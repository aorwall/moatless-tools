# astropy__astropy-14628

| **astropy/astropy** | `c667e73df92215cf1446c3eda71a56fdaebba426` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 944 |
| **Any found context length** | 944 |
| **Avg pos** | 4.0 |
| **Min pos** | 2 |
| **Max pos** | 2 |
| **Top file pos** | 2 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/astropy/coordinates/earth.py b/astropy/coordinates/earth.py
--- a/astropy/coordinates/earth.py
+++ b/astropy/coordinates/earth.py
@@ -655,21 +655,26 @@ def to_geocentric(self):
         """Convert to a tuple with X, Y, and Z as quantities."""
         return (self.x, self.y, self.z)
 
-    def get_itrs(self, obstime=None):
+    def get_itrs(self, obstime=None, location=None):
         """
         Generates an `~astropy.coordinates.ITRS` object with the location of
-        this object at the requested ``obstime``.
+        this object at the requested ``obstime``, either geocentric, or
+        topocentric relative to a given ``location``.
 
         Parameters
         ----------
         obstime : `~astropy.time.Time` or None
             The ``obstime`` to apply to the new `~astropy.coordinates.ITRS`, or
             if None, the default ``obstime`` will be used.
+        location : `~astropy.coordinates.EarthLocation` or None
+            A possible observer's location, for a topocentric ITRS position.
+            If not given (default), a geocentric ITRS object will be created.
 
         Returns
         -------
         itrs : `~astropy.coordinates.ITRS`
-            The new object in the ITRS frame
+            The new object in the ITRS frame, either geocentric or topocentric
+            relative to the given ``location``.
         """
         # Broadcast for a single position at multiple times, but don't attempt
         # to be more general here.
@@ -679,7 +684,18 @@ def get_itrs(self, obstime=None):
         # do this here to prevent a series of complicated circular imports
         from .builtin_frames import ITRS
 
-        return ITRS(x=self.x, y=self.y, z=self.z, obstime=obstime)
+        if location is None:
+            # No location provided, return geocentric ITRS coordinates
+            return ITRS(x=self.x, y=self.y, z=self.z, obstime=obstime)
+        else:
+            return ITRS(
+                self.x - location.x,
+                self.y - location.y,
+                self.z - location.z,
+                copy=False,
+                obstime=obstime,
+                location=location,
+            )
 
     itrs = property(
         get_itrs,

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| astropy/coordinates/earth.py | 658 | 672 | 2 | 2 | 944
| astropy/coordinates/earth.py | 682 | 682 | 2 | 2 | 944


## Problem Statement

```
Make getting a topocentric ITRS position easier
### What is the problem this feature will solve?

Right now, it is not easy to create ITRS coordinates for sources relative to a given location (rather than geocentric), to the level that we have specific instructions on how to calculate relative `CartesianCoordinates` and then put these into an `ITRS`: https://docs.astropy.org/en/latest/coordinates/common_errors.html#altaz-calculations-for-earth-based-objects

This has led to numerous issues, the latest of which is #12678

### Describe the desired outcome

It would be nice if as part of `EarthLocation.get_itrs()` it would be possible to get a topocentric rather than a geocentric position. In #12678, @tomfelker and @mkbrewer [suggested](https://github.com/astropy/astropy/issues/12678#issuecomment-1463366166) (and below) to extend `.get_itrs()` to take not just an `obstime` but also a `location` argument, with an implementation along [the following lines](https://github.com/astropy/astropy/issues/12678#issuecomment-1464065862):

> the idea would be to simply add a `location` argument to `get_itrs()` that defaults to `None`. Then if a location is provided, `get_itrs()` would return a topocentric ITRS frame containing the difference between the object's position and that of the `location` argument. One could also use `EARTH_CENTER` as the default and always return the difference.

### Additional context

See #12768. Labeling this a good first issue since it is easy code wise. However, writing the tests and documentation will require understanding of how ITRS and the associated coordinate transformations work.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 astropy/coordinates/builtin_frames/itrs.py | 29 | 90| 714 | 714 | 927 | 
| **-> 2 <-** | **2 astropy/coordinates/earth.py** | 658 | 682| 230 | 944 | 9536 | 
| 3 | 2 astropy/coordinates/builtin_frames/itrs.py | 3 | 26| 195 | 1139 | 9536 | 
| 4 | 3 astropy/coordinates/builtin_frames/itrs_observed_transforms.py | 125 | 148| 298 | 1437 | 11168 | 
| 5 | **3 astropy/coordinates/earth.py** | 684 | 709| 213 | 1650 | 11168 | 
| 6 | 4 astropy/coordinates/builtin_frames/intermediate_rotation_transforms.py | 90 | 108| 257 | 1907 | 14610 | 
| 7 | **4 astropy/coordinates/earth.py** | 177 | 207| 319 | 2226 | 14610 | 
| 8 | 5 astropy/coordinates/builtin_frames/cirs.py | 3 | 41| 305 | 2531 | 14932 | 
| 9 | **5 astropy/coordinates/earth.py** | 711 | 744| 461 | 2992 | 14932 | 
| 10 | 5 astropy/coordinates/builtin_frames/intermediate_rotation_transforms.py | 22 | 41| 254 | 3246 | 14932 | 
| 11 | **5 astropy/coordinates/earth.py** | 209 | 224| 173 | 3419 | 14932 | 
| 12 | 6 astropy/coordinates/builtin_frames/equatorial.py | 14 | 46| 265 | 3684 | 15992 | 
| 13 | 6 astropy/coordinates/builtin_frames/intermediate_rotation_transforms.py | 233 | 255| 237 | 3921 | 15992 | 
| 14 | 6 astropy/coordinates/builtin_frames/intermediate_rotation_transforms.py | 209 | 219| 140 | 4061 | 15992 | 
| 15 | 7 astropy/io/misc/asdf/tags/coordinates/earthlocation.py | 2 | 22| 127 | 4188 | 16136 | 
| 16 | 8 astropy/coordinates/builtin_frames/icrs_cirs_transforms.py | 110 | 151| 440 | 4628 | 19040 | 
| 17 | 8 astropy/coordinates/builtin_frames/intermediate_rotation_transforms.py | 222 | 230| 134 | 4762 | 19040 | 
| 18 | **8 astropy/coordinates/earth.py** | 226 | 273| 429 | 5191 | 19040 | 
| 19 | 8 astropy/coordinates/builtin_frames/intermediate_rotation_transforms.py | 287 | 305| 215 | 5406 | 19040 | 
| 20 | 9 astropy/coordinates/builtin_frames/gcrs.py | 62 | 86| 341 | 5747 | 20279 | 
| 21 | 9 astropy/coordinates/builtin_frames/equatorial.py | 49 | 83| 440 | 6187 | 20279 | 
| 22 | 9 astropy/coordinates/builtin_frames/intermediate_rotation_transforms.py | 153 | 163| 143 | 6330 | 20279 | 
| 23 | **9 astropy/coordinates/earth.py** | 275 | 320| 541 | 6871 | 20279 | 
| 24 | 9 astropy/coordinates/builtin_frames/itrs_observed_transforms.py | 101 | 122| 283 | 7154 | 20279 | 
| 25 | 9 astropy/coordinates/builtin_frames/intermediate_rotation_transforms.py | 44 | 61| 206 | 7360 | 20279 | 
| 26 | 9 astropy/coordinates/builtin_frames/intermediate_rotation_transforms.py | 135 | 150| 224 | 7584 | 20279 | 
| 27 | 9 astropy/coordinates/builtin_frames/intermediate_rotation_transforms.py | 166 | 174| 136 | 7720 | 20279 | 
| 28 | 9 astropy/coordinates/builtin_frames/intermediate_rotation_transforms.py | 194 | 206| 203 | 7923 | 20279 | 
| 29 | 10 astropy/coordinates/builtin_frames/utils.py | 7 | 37| 323 | 8246 | 24116 | 
| 30 | 10 astropy/coordinates/builtin_frames/intermediate_rotation_transforms.py | 111 | 132| 296 | 8542 | 24116 | 
| 31 | **10 astropy/coordinates/earth.py** | 470 | 494| 258 | 8800 | 24116 | 
| 32 | 10 astropy/coordinates/builtin_frames/intermediate_rotation_transforms.py | 276 | 284| 125 | 8925 | 24116 | 
| 33 | 10 astropy/coordinates/builtin_frames/icrs_cirs_transforms.py | 29 | 67| 419 | 9344 | 24116 | 
| 34 | 10 astropy/coordinates/builtin_frames/itrs_observed_transforms.py | 1 | 45| 378 | 9722 | 24116 | 
| 35 | 10 astropy/coordinates/builtin_frames/intermediate_rotation_transforms.py | 177 | 191| 185 | 9907 | 24116 | 
| 36 | 10 astropy/coordinates/builtin_frames/icrs_cirs_transforms.py | 244 | 259| 187 | 10094 | 24116 | 
| 37 | **10 astropy/coordinates/earth.py** | 746 | 772| 230 | 10324 | 24116 | 
| 38 | 11 astropy/coordinates/builtin_frames/altaz.py | 43 | 78| 428 | 10752 | 25319 | 
| 39 | 12 astropy/wcs/utils.py | 1212 | 1268| 471 | 11223 | 36309 | 
| 40 | 12 astropy/coordinates/builtin_frames/icrs_cirs_transforms.py | 195 | 241| 604 | 11827 | 36309 | 
| 41 | 13 astropy/coordinates/builtin_frames/ecliptic_transforms.py | 103 | 117| 180 | 12007 | 39421 | 
| 42 | 14 astropy/coordinates/builtin_frames/icrs_observed_transforms.py | 67 | 115| 525 | 12532 | 40666 | 
| 43 | 14 astropy/coordinates/builtin_frames/icrs_cirs_transforms.py | 154 | 192| 403 | 12935 | 40666 | 
| 44 | 14 astropy/coordinates/builtin_frames/icrs_cirs_transforms.py | 275 | 285| 143 | 13078 | 40666 | 
| 45 | 14 astropy/coordinates/builtin_frames/equatorial.py | 86 | 114| 243 | 13321 | 40666 | 
| 46 | **14 astropy/coordinates/earth.py** | 849 | 880| 205 | 13526 | 40666 | 
| 47 | 14 astropy/coordinates/builtin_frames/ecliptic_transforms.py | 154 | 169| 157 | 13683 | 40666 | 
| 48 | 14 astropy/coordinates/builtin_frames/icrs_cirs_transforms.py | 70 | 107| 394 | 14077 | 40666 | 
| 49 | 15 astropy/coordinates/builtin_frames/galactocentric.py | 596 | 610| 154 | 14231 | 46940 | 
| 50 | **15 astropy/coordinates/earth.py** | 882 | 891| 136 | 14367 | 46940 | 
| 51 | 15 astropy/coordinates/builtin_frames/icrs_cirs_transforms.py | 262 | 272| 126 | 14493 | 46940 | 
| 52 | 16 astropy/coordinates/builtin_frames/lsr.py | 197 | 223| 332 | 14825 | 49920 | 
| 53 | **16 astropy/coordinates/earth.py** | 396 | 468| 766 | 15591 | 49920 | 
| 54 | 17 astropy/coordinates/builtin_frames/ecliptic.py | 99 | 117| 207 | 15798 | 52267 | 
| 55 | 17 astropy/coordinates/builtin_frames/icrs_observed_transforms.py | 118 | 127| 134 | 15932 | 52267 | 
| 56 | 18 astropy/coordinates/builtin_frames/hcrs.py | 22 | 47| 238 | 16170 | 52630 | 
| 57 | 18 astropy/coordinates/builtin_frames/ecliptic_transforms.py | 73 | 88| 156 | 16326 | 52630 | 
| 58 | 19 astropy/coordinates/builtin_frames/__init__.py | 26 | 114| 737 | 17063 | 54460 | 
| 59 | 19 astropy/coordinates/builtin_frames/gcrs.py | 89 | 105| 211 | 17274 | 54460 | 
| 60 | 19 astropy/coordinates/builtin_frames/ecliptic_transforms.py | 228 | 241| 145 | 17419 | 54460 | 
| 61 | 19 astropy/coordinates/builtin_frames/lsr.py | 254 | 287| 362 | 17781 | 54460 | 
| 62 | 19 astropy/coordinates/builtin_frames/ecliptic_transforms.py | 212 | 225| 178 | 17959 | 54460 | 
| 63 | 19 astropy/coordinates/builtin_frames/ecliptic_transforms.py | 194 | 209| 173 | 18132 | 54460 | 
| 64 | 19 astropy/coordinates/builtin_frames/ecliptic_transforms.py | 259 | 298| 402 | 18534 | 54460 | 
| 65 | **19 astropy/coordinates/earth.py** | 633 | 656| 185 | 18719 | 54460 | 
| 66 | 19 astropy/coordinates/builtin_frames/ecliptic_transforms.py | 138 | 151| 176 | 18895 | 54460 | 
| 67 | 19 astropy/coordinates/builtin_frames/ecliptic_transforms.py | 172 | 181| 134 | 19029 | 54460 | 
| 68 | 19 astropy/coordinates/builtin_frames/gcrs.py | 38 | 59| 287 | 19316 | 54460 | 
| 69 | **19 astropy/coordinates/earth.py** | 582 | 631| 415 | 19731 | 54460 | 
| 70 | 19 astropy/coordinates/builtin_frames/icrs_observed_transforms.py | 24 | 64| 423 | 20154 | 54460 | 
| 71 | 19 astropy/coordinates/builtin_frames/ecliptic_transforms.py | 91 | 100| 133 | 20287 | 54460 | 
| 72 | 20 astropy/coordinates/sky_coordinate.py | 769 | 848| 805 | 21092 | 74154 | 
| 73 | 20 astropy/coordinates/builtin_frames/gcrs.py | 3 | 35| 383 | 21475 | 74154 | 
| 74 | 20 astropy/coordinates/builtin_frames/ecliptic_transforms.py | 244 | 256| 135 | 21610 | 74154 | 
| 75 | 20 astropy/coordinates/builtin_frames/ecliptic_transforms.py | 120 | 135| 172 | 21782 | 74154 | 
| 76 | 20 astropy/coordinates/sky_coordinate.py | 1950 | 2022| 795 | 22577 | 74154 | 
| 77 | 21 astropy/coordinates/builtin_frames/cirs_observed_transforms.py | 71 | 107| 337 | 22914 | 75114 | 
| 78 | 22 astropy/coordinates/erfa_astrom.py | 113 | 185| 716 | 23630 | 78835 | 
| 79 | 22 astropy/coordinates/builtin_frames/ecliptic.py | 78 | 96| 207 | 23837 | 78835 | 
| 80 | 22 astropy/coordinates/builtin_frames/galactocentric.py | 367 | 438| 1121 | 24958 | 78835 | 
| 81 | 22 astropy/coordinates/builtin_frames/ecliptic_transforms.py | 184 | 191| 121 | 25079 | 78835 | 
| 82 | **22 astropy/coordinates/earth.py** | 124 | 174| 451 | 25530 | 78835 | 
| 83 | 22 astropy/coordinates/builtin_frames/intermediate_rotation_transforms.py | 8 | 20| 111 | 25641 | 78835 | 
| 84 | 23 astropy/coordinates/solar_system.py | 538 | 552| 154 | 25795 | 84035 | 
| 85 | 24 astropy/coordinates/builtin_frames/icrs_fk5_transforms.py | 28 | 45| 229 | 26024 | 84469 | 
| 86 | 24 astropy/coordinates/builtin_frames/icrs_observed_transforms.py | 5 | 21| 125 | 26149 | 84469 | 
| 87 | 25 astropy/coordinates/attributes.py | 359 | 373| 121 | 26270 | 88052 | 
| 88 | 25 astropy/coordinates/builtin_frames/utils.py | 331 | 383| 501 | 26771 | 88052 | 
| 89 | 26 astropy/time/core.py | 2334 | 2396| 581 | 27352 | 116336 | 
| 90 | 27 examples/coordinates/plot_sgr-coordinate-frame.py | 107 | 179| 791 | 28143 | 118716 | 
| 91 | 28 astropy/utils/iers/iers.py | 948 | 990| 421 | 28564 | 130656 | 
| 92 | 29 astropy/coordinates/builtin_frames/icrs.py | 3 | 25| 207 | 28771 | 130880 | 
| 93 | 29 astropy/coordinates/builtin_frames/utils.py | 226 | 251| 312 | 29083 | 130880 | 
| 94 | 29 astropy/coordinates/builtin_frames/intermediate_rotation_transforms.py | 258 | 273| 159 | 29242 | 130880 | 
| 95 | 29 astropy/coordinates/builtin_frames/galactocentric.py | 526 | 572| 435 | 29677 | 130880 | 
| 96 | 29 astropy/coordinates/attributes.py | 375 | 410| 236 | 29913 | 130880 | 
| 97 | 30 astropy/coordinates/funcs.py | 132 | 182| 519 | 30432 | 134149 | 
| 98 | 30 astropy/coordinates/builtin_frames/icrs_cirs_transforms.py | 7 | 26| 139 | 30571 | 134149 | 
| 99 | 31 astropy/coordinates/transformations.py | 1252 | 1341| 943 | 31514 | 147892 | 
| 100 | 31 astropy/coordinates/builtin_frames/utils.py | 254 | 328| 828 | 32342 | 147892 | 
| 101 | 31 astropy/coordinates/builtin_frames/utils.py | 70 | 92| 173 | 32515 | 147892 | 
| 102 | 31 astropy/coordinates/solar_system.py | 555 | 573| 188 | 32703 | 147892 | 
| 103 | **31 astropy/coordinates/earth.py** | 3 | 64| 483 | 33186 | 147892 | 
| 104 | 31 astropy/coordinates/transformations.py | 794 | 870| 576 | 33762 | 147892 | 
| 105 | 32 astropy/io/misc/asdf/tags/coordinates/frames.py | 133 | 169| 275 | 34037 | 149032 | 
| 106 | **32 astropy/coordinates/earth.py** | 914 | 954| 359 | 34396 | 149032 | 
| 107 | 32 astropy/coordinates/builtin_frames/ecliptic.py | 147 | 161| 163 | 34559 | 149032 | 
| 108 | 32 astropy/coordinates/erfa_astrom.py | 300 | 313| 152 | 34711 | 149032 | 
| 109 | 32 astropy/coordinates/builtin_frames/cirs_observed_transforms.py | 6 | 68| 588 | 35299 | 149032 | 
| 110 | 32 astropy/io/misc/asdf/tags/coordinates/frames.py | 114 | 130| 139 | 35438 | 149032 | 
| 111 | 33 astropy/io/fits/fitstime.py | 151 | 190| 421 | 35859 | 154482 | 
| 112 | 33 astropy/coordinates/builtin_frames/ecliptic_transforms.py | 5 | 36| 310 | 36169 | 154482 | 
| 113 | **33 astropy/coordinates/earth.py** | 774 | 847| 701 | 36870 | 154482 | 
| 114 | 33 examples/coordinates/plot_sgr-coordinate-frame.py | 181 | 240| 607 | 37477 | 154482 | 
| 115 | 34 astropy/convolution/convolve.py | 130 | 215| 896 | 38373 | 165571 | 
| 116 | **34 astropy/coordinates/earth.py** | 322 | 394| 685 | 39058 | 165571 | 
| 117 | 35 astropy/coordinates/builtin_frames/hadec.py | 3 | 74| 751 | 39809 | 166784 | 
| 118 | **35 astropy/coordinates/earth.py** | 107 | 122| 152 | 39961 | 166784 | 
| 119 | **35 astropy/coordinates/earth.py** | 534 | 580| 381 | 40342 | 166784 | 
| 120 | 35 astropy/coordinates/builtin_frames/galactocentric.py | 3 | 37| 261 | 40603 | 166784 | 
| 121 | 35 astropy/coordinates/builtin_frames/ecliptic.py | 225 | 242| 192 | 40795 | 166784 | 
| 122 | 35 astropy/time/core.py | 2107 | 2131| 324 | 41119 | 166784 | 
| 123 | 35 astropy/coordinates/builtin_frames/lsr.py | 70 | 76| 115 | 41234 | 166784 | 
| 124 | 35 astropy/coordinates/sky_coordinate.py | 1851 | 1949| 1208 | 42442 | 166784 | 
| 125 | 35 astropy/coordinates/transformations.py | 1158 | 1250| 811 | 43253 | 166784 | 
| 126 | 35 astropy/coordinates/builtin_frames/lsr.py | 79 | 85| 116 | 43369 | 166784 | 
| 127 | 35 astropy/coordinates/builtin_frames/ecliptic.py | 65 | 75| 117 | 43486 | 166784 | 
| 128 | 35 astropy/coordinates/builtin_frames/ecliptic.py | 120 | 144| 230 | 43716 | 166784 | 
| 129 | 35 astropy/coordinates/erfa_astrom.py | 329 | 386| 509 | 44225 | 166784 | 
| 130 | 36 astropy/coordinates/name_resolve.py | 165 | 209| 344 | 44569 | 168331 | 
| 131 | 36 astropy/coordinates/builtin_frames/ecliptic.py | 201 | 222| 180 | 44749 | 168331 | 
| 132 | 36 astropy/coordinates/builtin_frames/altaz.py | 3 | 41| 333 | 45082 | 168331 | 
| 133 | 36 astropy/coordinates/builtin_frames/itrs_observed_transforms.py | 48 | 74| 391 | 45473 | 168331 | 
| 134 | 37 astropy/coordinates/builtin_frames/fk5.py | 3 | 20| 120 | 45593 | 168817 | 
| 135 | 37 astropy/coordinates/erfa_astrom.py | 258 | 280| 215 | 45808 | 168817 | 
| 136 | 38 examples/coordinates/plot_obs-planning.py | 36 | 126| 822 | 46630 | 170307 | 
| 137 | 38 astropy/coordinates/builtin_frames/galactocentric.py | 441 | 509| 752 | 47382 | 170307 | 
| 138 | 38 astropy/time/core.py | 2398 | 2433| 438 | 47820 | 170307 | 
| 139 | **38 astropy/coordinates/earth.py** | 67 | 104| 307 | 48127 | 170307 | 
| 140 | **38 astropy/coordinates/earth.py** | 957 | 976| 153 | 48280 | 170307 | 
| 141 | 38 astropy/coordinates/transformations.py | 1599 | 1678| 702 | 48982 | 170307 | 
| 142 | 38 astropy/time/core.py | 2133 | 2185| 465 | 49447 | 170307 | 
| 143 | 39 astropy/coordinates/builtin_frames/fk4.py | 3 | 33| 208 | 49655 | 172281 | 
| 144 | 39 astropy/coordinates/transformations.py | 1556 | 1578| 245 | 49900 | 172281 | 


### Hint

```
Hi, I am interested in signing up for this issue. This would be my first contribution here
@ninja18 - great! Easiest is to go ahead and make a PR. I assume you are familiar with the astronomy side of it? (see "Additional Context" above)
```

## Patch

```diff
diff --git a/astropy/coordinates/earth.py b/astropy/coordinates/earth.py
--- a/astropy/coordinates/earth.py
+++ b/astropy/coordinates/earth.py
@@ -655,21 +655,26 @@ def to_geocentric(self):
         """Convert to a tuple with X, Y, and Z as quantities."""
         return (self.x, self.y, self.z)
 
-    def get_itrs(self, obstime=None):
+    def get_itrs(self, obstime=None, location=None):
         """
         Generates an `~astropy.coordinates.ITRS` object with the location of
-        this object at the requested ``obstime``.
+        this object at the requested ``obstime``, either geocentric, or
+        topocentric relative to a given ``location``.
 
         Parameters
         ----------
         obstime : `~astropy.time.Time` or None
             The ``obstime`` to apply to the new `~astropy.coordinates.ITRS`, or
             if None, the default ``obstime`` will be used.
+        location : `~astropy.coordinates.EarthLocation` or None
+            A possible observer's location, for a topocentric ITRS position.
+            If not given (default), a geocentric ITRS object will be created.
 
         Returns
         -------
         itrs : `~astropy.coordinates.ITRS`
-            The new object in the ITRS frame
+            The new object in the ITRS frame, either geocentric or topocentric
+            relative to the given ``location``.
         """
         # Broadcast for a single position at multiple times, but don't attempt
         # to be more general here.
@@ -679,7 +684,18 @@ def get_itrs(self, obstime=None):
         # do this here to prevent a series of complicated circular imports
         from .builtin_frames import ITRS
 
-        return ITRS(x=self.x, y=self.y, z=self.z, obstime=obstime)
+        if location is None:
+            # No location provided, return geocentric ITRS coordinates
+            return ITRS(x=self.x, y=self.y, z=self.z, obstime=obstime)
+        else:
+            return ITRS(
+                self.x - location.x,
+                self.y - location.y,
+                self.z - location.z,
+                copy=False,
+                obstime=obstime,
+                location=location,
+            )
 
     itrs = property(
         get_itrs,

```

## Test Patch

```diff
diff --git a/astropy/coordinates/tests/test_intermediate_transformations.py b/astropy/coordinates/tests/test_intermediate_transformations.py
--- a/astropy/coordinates/tests/test_intermediate_transformations.py
+++ b/astropy/coordinates/tests/test_intermediate_transformations.py
@@ -1036,24 +1036,12 @@ def test_itrs_straight_overhead():
     obj = EarthLocation(-1 * u.deg, 52 * u.deg, height=10.0 * u.km)
     home = EarthLocation(-1 * u.deg, 52 * u.deg, height=0.0 * u.km)
 
-    # An object that appears straight overhead - FOR A GEOCENTRIC OBSERVER.
-    itrs_geo = obj.get_itrs(t).cartesian
-
-    # now get the Geocentric ITRS position of observatory
-    obsrepr = home.get_itrs(t).cartesian
-
-    # topocentric ITRS position of a straight overhead object
-    itrs_repr = itrs_geo - obsrepr
-
-    # create a ITRS object that appears straight overhead for a TOPOCENTRIC OBSERVER
-    itrs_topo = ITRS(itrs_repr, obstime=t, location=home)
-
     # Check AltAz (though Azimuth can be anything so is not tested).
-    aa = itrs_topo.transform_to(AltAz(obstime=t, location=home))
+    aa = obj.get_itrs(t, location=home).transform_to(AltAz(obstime=t, location=home))
     assert_allclose(aa.alt, 90 * u.deg, atol=1 * u.uas, rtol=0)
 
     # Check HADec.
-    hd = itrs_topo.transform_to(HADec(obstime=t, location=home))
+    hd = obj.get_itrs(t, location=home).transform_to(HADec(obstime=t, location=home))
     assert_allclose(hd.ha, 0 * u.hourangle, atol=1 * u.uas, rtol=0)
     assert_allclose(hd.dec, 52 * u.deg, atol=1 * u.uas, rtol=0)
 

```


## Code snippets

### 1 - astropy/coordinates/builtin_frames/itrs.py:

Start line: 29, End line: 90

```python
@format_doc(base_doc, components="", footer=doc_footer)
class ITRS(BaseCoordinateFrame):
    """
    A coordinate or frame in the International Terrestrial Reference System
    (ITRS).  This is approximately a geocentric system, although strictly it is
    defined by a series of reference locations near the surface of the Earth (the ITRF).
    For more background on the ITRS, see the references provided in the
    :ref:`astropy:astropy-coordinates-seealso` section of the documentation.

    This frame also includes frames that are defined *relative* to the center of the Earth,
    but that are offset (in both position and velocity) from the center of the Earth. You
    may see such non-geocentric coordinates referred to as "topocentric".

    Topocentric ITRS frames are convenient for observations of near Earth objects where
    stellar aberration is not included. One can merely subtract the observing site's
    EarthLocation geocentric ITRS coordinates from the object's geocentric ITRS coordinates,
    put the resulting vector into a topocentric ITRS frame and then transform to
    `~astropy.coordinates.AltAz` or `~astropy.coordinates.HADec`. The other way around is
    to transform an observed `~astropy.coordinates.AltAz` or `~astropy.coordinates.HADec`
    position to a topocentric ITRS frame and add the observing site's EarthLocation geocentric
    ITRS coordinates to yield the object's geocentric ITRS coordinates.

    On the other hand, using ``transform_to`` to transform geocentric ITRS coordinates to
    topocentric ITRS, observed `~astropy.coordinates.AltAz`, or observed
    `~astropy.coordinates.HADec` coordinates includes the difference between stellar aberration
    from the point of view of an observer at the geocenter and stellar aberration from the
    point of view of an observer on the surface of the Earth. If the geocentric ITRS
    coordinates of the object include stellar aberration at the geocenter (e.g. certain ILRS
    ephemerides), then this is the way to go.

    Note to ILRS ephemeris users: Astropy does not currently consider relativistic
    effects of the Earth's gravatational field. Nor do the `~astropy.coordinates.AltAz`
    or `~astropy.coordinates.HADec` refraction corrections compute the change in the
    range due to the curved path of light through the atmosphere, so Astropy is no
    substitute for the ILRS software in these respects.

    """

    default_representation = CartesianRepresentation
    default_differential = CartesianDifferential

    obstime = TimeAttribute(default=DEFAULT_OBSTIME)
    location = EarthLocationAttribute(default=EARTH_CENTER)

    @property
    def earth_location(self):
        """
        The data in this frame as an `~astropy.coordinates.EarthLocation` class.
        """
        from astropy.coordinates.earth import EarthLocation

        cart = self.represent_as(CartesianRepresentation)
        return EarthLocation(
            x=cart.x + self.location.x,
            y=cart.y + self.location.y,
            z=cart.z + self.location.z,
        )


# Self-transform is in intermediate_rotation_transforms.py with all the other
# ITRS transforms
```
### 2 - astropy/coordinates/earth.py:

Start line: 658, End line: 682

```python
class EarthLocation(u.Quantity):

    def get_itrs(self, obstime=None):
        """
        Generates an `~astropy.coordinates.ITRS` object with the location of
        this object at the requested ``obstime``.

        Parameters
        ----------
        obstime : `~astropy.time.Time` or None
            The ``obstime`` to apply to the new `~astropy.coordinates.ITRS`, or
            if None, the default ``obstime`` will be used.

        Returns
        -------
        itrs : `~astropy.coordinates.ITRS`
            The new object in the ITRS frame
        """
        # Broadcast for a single position at multiple times, but don't attempt
        # to be more general here.
        if obstime and self.size == 1 and obstime.shape:
            self = np.broadcast_to(self, obstime.shape, subok=True)

        # do this here to prevent a series of complicated circular imports
        from .builtin_frames import ITRS

        return ITRS(x=self.x, y=self.y, z=self.z, obstime=obstime)
```
### 3 - astropy/coordinates/builtin_frames/itrs.py:

Start line: 3, End line: 26

```python
from astropy.coordinates.attributes import EarthLocationAttribute, TimeAttribute
from astropy.coordinates.baseframe import BaseCoordinateFrame, base_doc
from astropy.coordinates.representation import (
    CartesianDifferential,
    CartesianRepresentation,
)
from astropy.utils.decorators import format_doc

from .utils import DEFAULT_OBSTIME, EARTH_CENTER

__all__ = ["ITRS"]

doc_footer = """
    Other parameters
    ----------------
    obstime : `~astropy.time.Time`
        The time at which the observation is taken.  Used for determining the
        position of the Earth and its precession.
    location : `~astropy.coordinates.EarthLocation`
        The location on the Earth.  This can be specified either as an
        `~astropy.coordinates.EarthLocation` object or as anything that can be
        transformed to an `~astropy.coordinates.ITRS` frame. The default is the
        centre of the Earth.
"""
```
### 4 - astropy/coordinates/builtin_frames/itrs_observed_transforms.py:

Start line: 125, End line: 148

```python
@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, AltAz, ITRS)
@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, HADec, ITRS)
def observed_to_itrs(observed_coo, itrs_frame):
    lon, lat, height = observed_coo.location.to_geodetic("WGS84")

    if isinstance(observed_coo, AltAz) or (observed_coo.pressure > 0.0):
        crepr = observed_coo.cartesian
        if observed_coo.pressure > 0.0:
            if isinstance(observed_coo, HADec):
                crepr = crepr.transform(matrix_transpose(altaz_to_hadec_mat(lat)))
            crepr = remove_refraction(crepr, observed_coo)
        crepr = crepr.transform(matrix_transpose(itrs_to_altaz_mat(lon, lat)))
    else:
        crepr = observed_coo.cartesian.transform(
            matrix_transpose(itrs_to_hadec_mat(lon))
        )

    itrs_at_obs_time = ITRS(
        crepr, obstime=observed_coo.obstime, location=observed_coo.location
    )
    # This final transform may be a no-op if the obstimes and locations are the same.
    # Otherwise, this transform will go through the CIRS and alter stellar aberration.
    return itrs_at_obs_time.transform_to(itrs_frame)
```
### 5 - astropy/coordinates/earth.py:

Start line: 684, End line: 709

```python
class EarthLocation(u.Quantity):

    itrs = property(
        get_itrs,
        doc="""An `~astropy.coordinates.ITRS` object
               for the location of this object at the
               default ``obstime``.""",
    )

    def get_gcrs(self, obstime):
        """GCRS position with velocity at ``obstime`` as a GCRS coordinate.

        Parameters
        ----------
        obstime : `~astropy.time.Time`
            The ``obstime`` to calculate the GCRS position/velocity at.

        Returns
        -------
        gcrs : `~astropy.coordinates.GCRS` instance
            With velocity included.
        """
        # do this here to prevent a series of complicated circular imports
        from .builtin_frames import GCRS

        loc, vel = self.get_gcrs_posvel(obstime)
        loc.differentials["s"] = CartesianDifferential.from_cartesian(vel)
        return GCRS(loc, obstime=obstime)
```
### 6 - astropy/coordinates/builtin_frames/intermediate_rotation_transforms.py:

Start line: 90, End line: 108

```python
def gcrs_precession_mat(equinox):
    gamb, phib, psib, epsa = erfa.pfw06(*get_jd12(equinox, "tt"))
    return erfa.fw2m(gamb, phib, psib, epsa)


def get_location_gcrs(location, obstime, ref_to_itrs, gcrs_to_ref):
    """Create a GCRS frame at the location and obstime.

    The reference frame z axis must point to the Celestial Intermediate Pole
    (as is the case for CIRS and TETE).

    This function is here to avoid location.get_gcrs(obstime), which would
    recalculate matrices that are already available below (and return a GCRS
    coordinate, rather than a frame with obsgeoloc and obsgeovel).  Instead,
    it uses the private method that allows passing in the matrices.

    """
    obsgeoloc, obsgeovel = location._get_gcrs_posvel(obstime, ref_to_itrs, gcrs_to_ref)
    return GCRS(obstime=obstime, obsgeoloc=obsgeoloc, obsgeovel=obsgeovel)
```
### 7 - astropy/coordinates/earth.py:

Start line: 177, End line: 207

```python
class EarthLocation(u.Quantity):
    """
    Location on the Earth.

    Initialization is first attempted assuming geocentric (x, y, z) coordinates
    are given; if that fails, another attempt is made assuming geodetic
    coordinates (longitude, latitude, height above a reference ellipsoid).
    When using the geodetic forms, Longitudes are measured increasing to the
    east, so west longitudes are negative. Internally, the coordinates are
    stored as geocentric.

    To ensure a specific type of coordinates is used, use the corresponding
    class methods (`from_geocentric` and `from_geodetic`) or initialize the
    arguments with names (``x``, ``y``, ``z`` for geocentric; ``lon``, ``lat``,
    ``height`` for geodetic).  See the class methods for details.


    Notes
    -----
    This class fits into the coordinates transformation framework in that it
    encodes a position on the `~astropy.coordinates.ITRS` frame.  To get a
    proper `~astropy.coordinates.ITRS` object from this object, use the ``itrs``
    property.
    """

    _ellipsoid = "WGS84"
    _location_dtype = np.dtype({"names": ["x", "y", "z"], "formats": [np.float64] * 3})
    _array_dtype = np.dtype((np.float64, (3,)))
    _site_registry = None

    info = EarthLocationInfo()
```
### 8 - astropy/coordinates/builtin_frames/cirs.py:

Start line: 3, End line: 41

```python
from astropy.coordinates.attributes import EarthLocationAttribute, TimeAttribute
from astropy.coordinates.baseframe import base_doc
from astropy.utils.decorators import format_doc

from .baseradec import BaseRADecFrame, doc_components
from .utils import DEFAULT_OBSTIME, EARTH_CENTER

__all__ = ["CIRS"]


doc_footer = """
    Other parameters
    ----------------
    obstime : `~astropy.time.Time`
        The time at which the observation is taken.  Used for determining the
        position of the Earth and its precession.
    location : `~astropy.coordinates.EarthLocation`
        The location on the Earth.  This can be specified either as an
        `~astropy.coordinates.EarthLocation` object or as anything that can be
        transformed to an `~astropy.coordinates.ITRS` frame. The default is the
        centre of the Earth.
"""


@format_doc(base_doc, components=doc_components, footer=doc_footer)
class CIRS(BaseRADecFrame):
    """
    A coordinate or frame in the Celestial Intermediate Reference System (CIRS).

    The frame attributes are listed under **Other Parameters**.
    """

    obstime = TimeAttribute(default=DEFAULT_OBSTIME)
    location = EarthLocationAttribute(default=EARTH_CENTER)


# The "self-transform" is defined in icrs_cirs_transformations.py, because in
# the current implementation it goes through ICRS (like GCRS)
```
### 9 - astropy/coordinates/earth.py:

Start line: 711, End line: 744

```python
class EarthLocation(u.Quantity):

    def _get_gcrs_posvel(self, obstime, ref_to_itrs, gcrs_to_ref):
        """Calculate GCRS position and velocity given transformation matrices.

        The reference frame z axis must point to the Celestial Intermediate Pole
        (as is the case for CIRS and TETE).

        This private method is used in intermediate_rotation_transforms,
        where some of the matrices are already available for the coordinate
        transformation.

        The method is faster by an order of magnitude than just adding a zero
        velocity to ITRS and transforming to GCRS, because it avoids calculating
        the velocity via finite differencing of the results of the transformation
        at three separate times.
        """
        # The simplest route is to transform to the reference frame where the
        # z axis is properly aligned with the Earth's rotation axis (CIRS or
        # TETE), then calculate the velocity, and then transform this
        # reference position and velocity to GCRS.  For speed, though, we
        # transform the coordinates to GCRS in one step, and calculate the
        # velocities by rotating around the earth's axis transformed to GCRS.
        ref_to_gcrs = matrix_transpose(gcrs_to_ref)
        itrs_to_gcrs = ref_to_gcrs @ matrix_transpose(ref_to_itrs)
        # Earth's rotation vector in the ref frame is rot_vec_ref = (0,0,OMEGA_EARTH),
        # so in GCRS it is rot_vec_gcrs[..., 2] @ OMEGA_EARTH.
        rot_vec_gcrs = CartesianRepresentation(
            ref_to_gcrs[..., 2] * OMEGA_EARTH, xyz_axis=-1, copy=False
        )
        # Get the position in the GCRS frame.
        # Since we just need the cartesian representation of ITRS, avoid get_itrs().
        itrs_cart = CartesianRepresentation(self.x, self.y, self.z, copy=False)
        pos = itrs_cart.transform(itrs_to_gcrs)
        vel = rot_vec_gcrs.cross(pos)
        return pos, vel
```
### 10 - astropy/coordinates/builtin_frames/intermediate_rotation_transforms.py:

Start line: 22, End line: 41

```python
# # first define helper functions


def teme_to_itrs_mat(time):
    # Sidereal time, rotates from ITRS to mean equinox
    # Use 1982 model for consistency with Vallado et al (2006)
    # http://www.celestrak.com/publications/aiaa/2006-6753/AIAA-2006-6753.pdf
    gst = erfa.gmst82(*get_jd12(time, "ut1"))

    # Polar Motion
    # Do not include TIO locator s' because it is not used in Vallado 2006
    xp, yp = get_polar_motion(time)
    pmmat = erfa.pom00(xp, yp, 0)

    # rotation matrix
    # c2tcio expects a GCRS->CIRS matrix as it's first argument.
    # Here, we just set that to an I-matrix, because we're already
    # in TEME and the difference between TEME and CIRS is just the
    # rotation by the sidereal time rather than the Earth Rotation Angle
    return erfa.c2tcio(np.eye(3), gst, pmmat)
```
### 11 - astropy/coordinates/earth.py:

Start line: 209, End line: 224

```python
class EarthLocation(u.Quantity):

    def __new__(cls, *args, **kwargs):
        # TODO: needs copy argument and better dealing with inputs.
        if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], EarthLocation):
            return args[0].copy()
        try:
            self = cls.from_geocentric(*args, **kwargs)
        except (u.UnitsError, TypeError) as exc_geocentric:
            try:
                self = cls.from_geodetic(*args, **kwargs)
            except Exception as exc_geodetic:
                raise TypeError(
                    "Coordinates could not be parsed as either "
                    "geocentric or geodetic, with respective "
                    f'exceptions "{exc_geocentric}" and "{exc_geodetic}"'
                )
        return self
```
### 18 - astropy/coordinates/earth.py:

Start line: 226, End line: 273

```python
class EarthLocation(u.Quantity):

    @classmethod
    def from_geocentric(cls, x, y, z, unit=None):
        """
        Location on Earth, initialized from geocentric coordinates.

        Parameters
        ----------
        x, y, z : `~astropy.units.Quantity` or array-like
            Cartesian coordinates.  If not quantities, ``unit`` should be given.
        unit : unit-like or None
            Physical unit of the coordinate values.  If ``x``, ``y``, and/or
            ``z`` are quantities, they will be converted to this unit.

        Raises
        ------
        astropy.units.UnitsError
            If the units on ``x``, ``y``, and ``z`` do not match or an invalid
            unit is given.
        ValueError
            If the shapes of ``x``, ``y``, and ``z`` do not match.
        TypeError
            If ``x`` is not a `~astropy.units.Quantity` and no unit is given.
        """
        if unit is None:
            try:
                unit = x.unit
            except AttributeError:
                raise TypeError(
                    "Geocentric coordinates should be Quantities "
                    "unless an explicit unit is given."
                ) from None
        else:
            unit = u.Unit(unit)

        if unit.physical_type != "length":
            raise u.UnitsError("Geocentric coordinates should be in units of length.")

        try:
            x = u.Quantity(x, unit, copy=False)
            y = u.Quantity(y, unit, copy=False)
            z = u.Quantity(z, unit, copy=False)
        except u.UnitsError:
            raise u.UnitsError("Geocentric coordinate units should all be consistent.")

        x, y, z = np.broadcast_arrays(x, y, z)
        struc = np.empty(x.shape, cls._location_dtype)
        struc["x"], struc["y"], struc["z"] = x, y, z
        return super().__new__(cls, struc, unit, copy=False)
```
### 23 - astropy/coordinates/earth.py:

Start line: 275, End line: 320

```python
class EarthLocation(u.Quantity):

    @classmethod
    def from_geodetic(cls, lon, lat, height=0.0, ellipsoid=None):
        """
        Location on Earth, initialized from geodetic coordinates.

        Parameters
        ----------
        lon : `~astropy.coordinates.Longitude` or float
            Earth East longitude.  Can be anything that initialises an
            `~astropy.coordinates.Angle` object (if float, in degrees).
        lat : `~astropy.coordinates.Latitude` or float
            Earth latitude.  Can be anything that initialises an
            `~astropy.coordinates.Latitude` object (if float, in degrees).
        height : `~astropy.units.Quantity` ['length'] or float, optional
            Height above reference ellipsoid (if float, in meters; default: 0).
        ellipsoid : str, optional
            Name of the reference ellipsoid to use (default: 'WGS84').
            Available ellipsoids are:  'WGS84', 'GRS80', 'WGS72'.

        Raises
        ------
        astropy.units.UnitsError
            If the units on ``lon`` and ``lat`` are inconsistent with angular
            ones, or that on ``height`` with a length.
        ValueError
            If ``lon``, ``lat``, and ``height`` do not have the same shape, or
            if ``ellipsoid`` is not recognized as among the ones implemented.

        Notes
        -----
        For the conversion to geocentric coordinates, the ERFA routine
        ``gd2gc`` is used.  See https://github.com/liberfa/erfa
        """
        ellipsoid = _check_ellipsoid(ellipsoid, default=cls._ellipsoid)
        # As wrapping fails on readonly input, we do so manually
        lon = Angle(lon, u.degree, copy=False).wrap_at(180 * u.degree)
        lat = Latitude(lat, u.degree, copy=False)
        # don't convert to m by default, so we can use the height unit below.
        if not isinstance(height, u.Quantity):
            height = u.Quantity(height, u.m, copy=False)
        # get geocentric coordinates.
        geodetic = ELLIPSOIDS[ellipsoid](lon, lat, height, copy=False)
        xyz = geodetic.to_cartesian().get_xyz(xyz_axis=-1) << height.unit
        self = xyz.view(cls._location_dtype, cls).reshape(geodetic.shape)
        self._ellipsoid = ellipsoid
        return self
```
### 31 - astropy/coordinates/earth.py:

Start line: 470, End line: 494

```python
class EarthLocation(u.Quantity):

    @classmethod
    def of_address(cls, address, get_height=False, google_api_key=None):
        # ... other code

        if use_google:
            loc = geo_result[0]["geometry"]["location"]
            lat = loc["lat"]
            lon = loc["lng"]

        else:
            loc = geo_result[0]
            lat = float(loc["lat"])  # strings are returned by OpenStreetMap
            lon = float(loc["lon"])

        if get_height:
            pars = {"locations": f"{lat:.8f},{lon:.8f}", "key": google_api_key}
            pars = urllib.parse.urlencode(pars)
            ele_url = f"https://maps.googleapis.com/maps/api/elevation/json?{pars}"

            err_str = f"Unable to retrieve elevation for address '{address}'; {{msg}}"
            ele_result = _get_json_result(
                ele_url, err_str=err_str, use_google=use_google
            )
            height = ele_result[0]["elevation"] * u.meter

        else:
            height = 0.0

        return cls.from_geodetic(lon=lon * u.deg, lat=lat * u.deg, height=height)
```
### 37 - astropy/coordinates/earth.py:

Start line: 746, End line: 772

```python
class EarthLocation(u.Quantity):

    def get_gcrs_posvel(self, obstime):
        """
        Calculate the GCRS position and velocity of this object at the
        requested ``obstime``.

        Parameters
        ----------
        obstime : `~astropy.time.Time`
            The ``obstime`` to calculate the GCRS position/velocity at.

        Returns
        -------
        obsgeoloc : `~astropy.coordinates.CartesianRepresentation`
            The GCRS position of the object
        obsgeovel : `~astropy.coordinates.CartesianRepresentation`
            The GCRS velocity of the object
        """
        # Local import to prevent circular imports.
        from .builtin_frames.intermediate_rotation_transforms import (
            cirs_to_itrs_mat,
            gcrs_to_cirs_mat,
        )

        # Get gcrs_posvel by transforming via CIRS (slightly faster than TETE).
        return self._get_gcrs_posvel(
            obstime, cirs_to_itrs_mat(obstime), gcrs_to_cirs_mat(obstime)
        )
```
### 46 - astropy/coordinates/earth.py:

Start line: 849, End line: 880

```python
class EarthLocation(u.Quantity):

    @property
    def x(self):
        """The X component of the geocentric coordinates."""
        return self["x"]

    @property
    def y(self):
        """The Y component of the geocentric coordinates."""
        return self["y"]

    @property
    def z(self):
        """The Z component of the geocentric coordinates."""
        return self["z"]

    def __getitem__(self, item):
        result = super().__getitem__(item)
        if result.dtype is self.dtype:
            return result.view(self.__class__)
        else:
            return result.view(u.Quantity)

    def __array_finalize__(self, obj):
        super().__array_finalize__(obj)
        if hasattr(obj, "_ellipsoid"):
            self._ellipsoid = obj._ellipsoid

    def __len__(self):
        if self.shape == ():
            raise IndexError("0-d EarthLocation arrays cannot be indexed")
        else:
            return super().__len__()
```
### 50 - astropy/coordinates/earth.py:

Start line: 882, End line: 891

```python
class EarthLocation(u.Quantity):

    def _to_value(self, unit, equivalencies=[]):
        """Helper method for to and to_value."""
        # Conversion to another unit in both ``to`` and ``to_value`` goes
        # via this routine. To make the regular quantity routines work, we
        # temporarily turn the structured array into a regular one.
        array_view = self.view(self._array_dtype, np.ndarray)
        if equivalencies == []:
            equivalencies = self._equivalencies
        new_array = self.unit.to(unit, array_view, equivalencies=equivalencies)
        return new_array.view(self.dtype).reshape(self.shape)
```
### 53 - astropy/coordinates/earth.py:

Start line: 396, End line: 468

```python
class EarthLocation(u.Quantity):

    @classmethod
    def of_address(cls, address, get_height=False, google_api_key=None):
        """
        Return an object of this class for a given address by querying either
        the OpenStreetMap Nominatim tool [1]_ (default) or the Google geocoding
        API [2]_, which requires a specified API key.

        This is intended as a quick convenience function to get easy access to
        locations. If you need to specify a precise location, you should use the
        initializer directly and pass in a longitude, latitude, and elevation.

        In the background, this just issues a web query to either of
        the APIs noted above. This is not meant to be abused! Both
        OpenStreetMap and Google use IP-based query limiting and will ban your
        IP if you send more than a few thousand queries per hour [2]_.

        .. warning::
            If the query returns more than one location (e.g., searching on
            ``address='springfield'``), this function will use the **first**
            returned location.

        Parameters
        ----------
        address : str
            The address to get the location for. As per the Google maps API,
            this can be a fully specified street address (e.g., 123 Main St.,
            New York, NY) or a city name (e.g., Danbury, CT), or etc.
        get_height : bool, optional
            This only works when using the Google API! See the ``google_api_key``
            block below. Use the retrieved location to perform a second query to
            the Google maps elevation API to retrieve the height of the input
            address [3]_.
        google_api_key : str, optional
            A Google API key with the Geocoding API and (optionally) the
            elevation API enabled. See [4]_ for more information.

        Returns
        -------
        location : `~astropy.coordinates.EarthLocation` (or subclass) instance
            The location of the input address.
            Will be type(this class)

        References
        ----------
        .. [1] https://nominatim.openstreetmap.org/
        .. [2] https://developers.google.com/maps/documentation/geocoding/start
        .. [3] https://developers.google.com/maps/documentation/elevation/start
        .. [4] https://developers.google.com/maps/documentation/geocoding/get-api-key

        """
        use_google = google_api_key is not None

        # Fail fast if invalid options are passed:
        if not use_google and get_height:
            raise ValueError(
                "Currently, `get_height` only works when using the Google geocoding"
                " API, which requires passing a Google API key with `google_api_key`."
                " See:"
                " https://developers.google.com/maps/documentation/geocoding/get-api-key"
                " for information on obtaining an API key."
            )

        if use_google:  # Google
            pars = urllib.parse.urlencode({"address": address, "key": google_api_key})
            geo_url = f"https://maps.googleapis.com/maps/api/geocode/json?{pars}"

        else:  # OpenStreetMap
            pars = urllib.parse.urlencode({"q": address, "format": "json"})
            geo_url = f"https://nominatim.openstreetmap.org/search?{pars}"

        # get longitude and latitude location
        err_str = f"Unable to retrieve coordinates for address '{address}'; {{msg}}"
        geo_result = _get_json_result(geo_url, err_str=err_str, use_google=use_google)
        # ... other code
```
### 65 - astropy/coordinates/earth.py:

Start line: 633, End line: 656

```python
class EarthLocation(u.Quantity):

    @property
    def lon(self):
        """Longitude of the location, for the default ellipsoid."""
        return self.geodetic[0]

    @property
    def lat(self):
        """Latitude of the location, for the default ellipsoid."""
        return self.geodetic[1]

    @property
    def height(self):
        """Height of the location, for the default ellipsoid."""
        return self.geodetic[2]

    # mostly for symmetry with geodetic and to_geodetic.
    @property
    def geocentric(self):
        """Convert to a tuple with X, Y, and Z as quantities."""
        return self.to_geocentric()

    def to_geocentric(self):
        """Convert to a tuple with X, Y, and Z as quantities."""
        return (self.x, self.y, self.z)
```
### 69 - astropy/coordinates/earth.py:

Start line: 582, End line: 631

```python
class EarthLocation(u.Quantity):

    @property
    def ellipsoid(self):
        """The default ellipsoid used to convert to geodetic coordinates."""
        return self._ellipsoid

    @ellipsoid.setter
    def ellipsoid(self, ellipsoid):
        self._ellipsoid = _check_ellipsoid(ellipsoid)

    @property
    def geodetic(self):
        """Convert to geodetic coordinates for the default ellipsoid."""
        return self.to_geodetic()

    def to_geodetic(self, ellipsoid=None):
        """Convert to geodetic coordinates.

        Parameters
        ----------
        ellipsoid : str, optional
            Reference ellipsoid to use.  Default is the one the coordinates
            were initialized with.  Available are: 'WGS84', 'GRS80', 'WGS72'

        Returns
        -------
        lon, lat, height : `~astropy.units.Quantity`
            The tuple is a ``GeodeticLocation`` namedtuple and is comprised of
            instances of `~astropy.coordinates.Longitude`,
            `~astropy.coordinates.Latitude`, and `~astropy.units.Quantity`.

        Raises
        ------
        ValueError
            if ``ellipsoid`` is not recognized as among the ones implemented.

        Notes
        -----
        For the conversion to geodetic coordinates, the ERFA routine
        ``gc2gd`` is used.  See https://github.com/liberfa/erfa
        """
        ellipsoid = _check_ellipsoid(ellipsoid, default=self.ellipsoid)
        xyz = self.view(self._array_dtype, u.Quantity)
        llh = CartesianRepresentation(xyz, xyz_axis=-1, copy=False).represent_as(
            ELLIPSOIDS[ellipsoid]
        )
        return GeodeticLocation(
            Longitude(llh.lon, u.deg, wrap_angle=180 * u.deg, copy=False),
            llh.lat << u.deg,
            llh.height << self.unit,
        )
```
### 82 - astropy/coordinates/earth.py:

Start line: 124, End line: 174

```python
class EarthLocationInfo(QuantityInfoBase):

    def new_like(self, cols, length, metadata_conflicts="warn", name=None):
        """
        Return a new EarthLocation instance which is consistent with the
        input ``cols`` and has ``length`` rows.

        This is intended for creating an empty column object whose elements can
        be set in-place for table operations like join or vstack.

        Parameters
        ----------
        cols : list
            List of input columns
        length : int
            Length of the output column object
        metadata_conflicts : str ('warn'|'error'|'silent')
            How to handle metadata conflicts
        name : str
            Output column name

        Returns
        -------
        col : EarthLocation (or subclass)
            Empty instance of this class consistent with ``cols``
        """
        # Very similar to QuantityInfo.new_like, but the creation of the
        # map is different enough that this needs its own rouinte.
        # Get merged info attributes shape, dtype, format, description.
        attrs = self.merge_cols_attributes(
            cols, metadata_conflicts, name, ("meta", "format", "description")
        )
        # The above raises an error if the dtypes do not match, but returns
        # just the string representation, which is not useful, so remove.
        attrs.pop("dtype")
        # Make empty EarthLocation using the dtype and unit of the last column.
        # Use zeros so we do not get problems for possible conversion to
        # geodetic coordinates.
        shape = (length,) + attrs.pop("shape")
        data = u.Quantity(
            np.zeros(shape=shape, dtype=cols[0].dtype), unit=cols[0].unit, copy=False
        )
        # Get arguments needed to reconstruct class
        map = {
            key: (data[key] if key in "xyz" else getattr(cols[-1], key))
            for key in self._represent_as_dict_attrs
        }
        out = self._construct_from_dict(map)
        # Set remaining info attributes
        for attr, value in attrs.items():
            setattr(out.info, attr, value)

        return out
```
### 103 - astropy/coordinates/earth.py:

Start line: 3, End line: 64

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
from .representation import (
    BaseRepresentation,
    CartesianDifferential,
    CartesianRepresentation,
)

__all__ = [
    "EarthLocation",
    "BaseGeodeticRepresentation",
    "WGS84GeodeticRepresentation",
    "WGS72GeodeticRepresentation",
    "GRS80GeodeticRepresentation",
]

GeodeticLocation = collections.namedtuple("GeodeticLocation", ["lon", "lat", "height"])

ELLIPSOIDS = {}
"""Available ellipsoids (defined in erfam.h, with numbers exposed in erfa)."""
# Note: they get filled by the creation of the geodetic classes.

OMEGA_EARTH = (1.002_737_811_911_354_48 * u.cycle / u.day).to(
    1 / u.s, u.dimensionless_angles()
)
"""
Rotational velocity of Earth, following SOFA's pvtob.

In UT1 seconds, this would be 2 pi / (24 * 3600), but we need the value
in SI seconds, so multiply by the ratio of stellar to solar day.
See Explanatory Supplement to the Astronomical Almanac, ed. P. Kenneth
Seidelmann (1992), University Science Books. The constant is the
conventional, exact one (IERS conventions 2003); see
http://hpiers.obspm.fr/eop-pc/index.php?index=constants.
"""


def _check_ellipsoid(ellipsoid=None, default="WGS84"):
    if ellipsoid is None:
        ellipsoid = default
    if ellipsoid not in ELLIPSOIDS:
        raise ValueError(f"Ellipsoid {ellipsoid} not among known ones ({ELLIPSOIDS})")
    return ellipsoid
```
### 106 - astropy/coordinates/earth.py:

Start line: 914, End line: 954

```python
@format_doc(geodetic_base_doc)
class BaseGeodeticRepresentation(BaseRepresentation):
    """Base geodetic representation."""

    attr_classes = {"lon": Longitude, "lat": Latitude, "height": u.Quantity}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "_ellipsoid" in cls.__dict__:
            ELLIPSOIDS[cls._ellipsoid] = cls

    def __init__(self, lon, lat=None, height=None, copy=True):
        if height is None and not isinstance(lon, self.__class__):
            height = 0 << u.m

        super().__init__(lon, lat, height, copy=copy)
        if not self.height.unit.is_equivalent(u.m):
            raise u.UnitTypeError(
                f"{self.__class__.__name__} requires height with units of length."
            )

    def to_cartesian(self):
        """
        Converts WGS84 geodetic coordinates to 3D rectangular (geocentric)
        cartesian coordinates.
        """
        xyz = erfa.gd2gc(
            getattr(erfa, self._ellipsoid), self.lon, self.lat, self.height
        )
        return CartesianRepresentation(xyz, xyz_axis=-1, copy=False)

    @classmethod
    def from_cartesian(cls, cart):
        """
        Converts 3D rectangular cartesian coordinates (assumed geocentric) to
        WGS84 geodetic coordinates.
        """
        lon, lat, height = erfa.gc2gd(
            getattr(erfa, cls._ellipsoid), cart.get_xyz(xyz_axis=-1)
        )
        return cls(lon, lat, height, copy=False)
```
### 113 - astropy/coordinates/earth.py:

Start line: 774, End line: 847

```python
class EarthLocation(u.Quantity):

    def gravitational_redshift(
        self, obstime, bodies=["sun", "jupiter", "moon"], masses={}
    ):
        """Return the gravitational redshift at this EarthLocation.

        Calculates the gravitational redshift, of order 3 m/s, due to the
        requested solar system bodies.

        Parameters
        ----------
        obstime : `~astropy.time.Time`
            The ``obstime`` to calculate the redshift at.

        bodies : iterable, optional
            The bodies (other than the Earth) to include in the redshift
            calculation.  List elements should be any body name
            `get_body_barycentric` accepts.  Defaults to Jupiter, the Sun, and
            the Moon.  Earth is always included (because the class represents
            an *Earth* location).

        masses : dict[str, `~astropy.units.Quantity`], optional
            The mass or gravitational parameters (G * mass) to assume for the
            bodies requested in ``bodies``. Can be used to override the
            defaults for the Sun, Jupiter, the Moon, and the Earth, or to
            pass in masses for other bodies.

        Returns
        -------
        redshift : `~astropy.units.Quantity`
            Gravitational redshift in velocity units at given obstime.
        """
        # needs to be here to avoid circular imports
        from .solar_system import get_body_barycentric

        bodies = list(bodies)
        # Ensure earth is included and last in the list.
        if "earth" in bodies:
            bodies.remove("earth")
        bodies.append("earth")
        _masses = {
            "sun": consts.GM_sun,
            "jupiter": consts.GM_jup,
            "moon": consts.G * 7.34767309e22 * u.kg,
            "earth": consts.GM_earth,
        }
        _masses.update(masses)
        GMs = []
        M_GM_equivalency = (u.kg, u.Unit(consts.G * u.kg))
        for body in bodies:
            try:
                GMs.append(_masses[body].to(u.m**3 / u.s**2, [M_GM_equivalency]))
            except KeyError as err:
                raise KeyError(f'body "{body}" does not have a mass.') from err
            except u.UnitsError as exc:
                exc.args += (
                    (
                        '"masses" argument values must be masses or '
                        "gravitational parameters."
                    ),
                )
                raise

        positions = [get_body_barycentric(name, obstime) for name in bodies]
        # Calculate distances to objects other than earth.
        distances = [(pos - positions[-1]).norm() for pos in positions[:-1]]
        # Append distance from Earth's center for Earth's contribution.
        distances.append(CartesianRepresentation(self.geocentric).norm())
        # Get redshifts due to all objects.
        redshifts = [
            -GM / consts.c / distance for (GM, distance) in zip(GMs, distances)
        ]
        # Reverse order of summing, to go from small to big, and to get
        # "earth" first, which gives m/s as unit.
        return sum(redshifts[::-1])
```
### 116 - astropy/coordinates/earth.py:

Start line: 322, End line: 394

```python
class EarthLocation(u.Quantity):

    @classmethod
    def of_site(cls, site_name, *, refresh_cache=False):
        """
        Return an object of this class for a known observatory/site by name.

        This is intended as a quick convenience function to get basic site
        information, not a fully-featured exhaustive registry of observatories
        and all their properties.

        Additional information about the site is stored in the ``.info.meta``
        dictionary of sites obtained using this method (see the examples below).

        .. note::
            This function is meant to access the site registry from the astropy
            data server, which is saved in the user's local cache.  If you would
            like a site to be added there, issue a pull request to the
            `astropy-data repository <https://github.com/astropy/astropy-data>`_ .
            If the cache already exists the function will use it even if the
            version in the astropy-data repository has been updated unless the
            ``refresh_cache=True`` option is used.  If there is no cache and the
            online version cannot be reached, this function falls back on a
            built-in list, which currently only contains the Greenwich Royal
            Observatory as an example case.

        Parameters
        ----------
        site_name : str
            Name of the observatory (case-insensitive).
        refresh_cache : bool, optional
            If `True`, force replacement of the cached registry with a
            newly downloaded version.  (Default: `False`)

            .. versionadded:: 5.3

        Returns
        -------
        site : `~astropy.coordinates.EarthLocation` (or subclass) instance
            The location of the observatory. The returned class will be the same
            as this class.

        Examples
        --------
        >>> from astropy.coordinates import EarthLocation
        >>> keck = EarthLocation.of_site('Keck Observatory')  # doctest: +REMOTE_DATA
        >>> keck.geodetic  # doctest: +REMOTE_DATA +FLOAT_CMP
        GeodeticLocation(lon=<Longitude -155.47833333 deg>, lat=<Latitude 19.82833333 deg>, height=<Quantity 4160. m>)
        >>> keck.info  # doctest: +REMOTE_DATA
        name = W. M. Keck Observatory
        dtype = (float64, float64, float64)
        unit = m
        class = EarthLocation
        n_bad = 0
        >>> keck.info.meta  # doctest: +REMOTE_DATA
        {'source': 'IRAF Observatory Database', 'timezone': 'US/Hawaii'}

        See Also
        --------
        get_site_names : the list of sites that this function can access
        """
        registry = cls._get_site_registry(force_download=refresh_cache)
        try:
            el = registry[site_name]
        except UnknownSiteException as e:
            raise UnknownSiteException(
                e.site, "EarthLocation.get_site_names", close_names=e.close_names
            ) from e

        if cls is el.__class__:
            return el
        else:
            newel = cls.from_geodetic(*el.to_geodetic())
            newel.info.name = el.info.name
            return newel
```
### 118 - astropy/coordinates/earth.py:

Start line: 107, End line: 122

```python
class EarthLocationInfo(QuantityInfoBase):
    """
    Container for meta information like name, description, format.  This is
    required when the object is used as a mixin column within a table, but can
    be used as a general way to store meta information.
    """

    _represent_as_dict_attrs = ("x", "y", "z", "ellipsoid")

    def _construct_from_dict(self, map):
        # Need to pop ellipsoid off and update post-instantiation.  This is
        # on the to-fix list in #4261.
        ellipsoid = map.pop("ellipsoid")
        out = self._parent_cls(**map)
        out.ellipsoid = ellipsoid
        return out
```
### 119 - astropy/coordinates/earth.py:

Start line: 534, End line: 580

```python
class EarthLocation(u.Quantity):

    @classmethod
    def _get_site_registry(cls, force_download=False, force_builtin=False):
        """
        Gets the site registry.  The first time this either downloads or loads
        from the data file packaged with astropy.  Subsequent calls will use the
        cached version unless explicitly overridden.

        Parameters
        ----------
        force_download : bool or str
            If not False, force replacement of the cached registry with a
            downloaded version. If a str, that will be used as the URL to
            download from (if just True, the default URL will be used).
        force_builtin : bool
            If True, load from the data file bundled with astropy and set the
            cache to that.

        Returns
        -------
        reg : astropy.coordinates.sites.SiteRegistry
        """
        # need to do this here at the bottom to avoid circular dependencies
        from .sites import get_builtin_sites, get_downloaded_sites

        if force_builtin and force_download:
            raise ValueError("Cannot have both force_builtin and force_download True")

        if force_builtin:
            cls._site_registry = get_builtin_sites()
        else:
            if force_download or not cls._site_registry:
                try:
                    if isinstance(force_download, str):
                        cls._site_registry = get_downloaded_sites(force_download)
                    else:
                        cls._site_registry = get_downloaded_sites()
                except OSError:
                    if force_download:
                        raise
                    msg = (
                        "Could not access the main site list. Falling back on the "
                        "built-in version, which is rather limited. If you want to "
                        "retry the download, use the option 'refresh_cache=True'."
                    )
                    warn(msg, AstropyUserWarning)
                    cls._site_registry = get_builtin_sites()
        return cls._site_registry
```
### 139 - astropy/coordinates/earth.py:

Start line: 67, End line: 104

```python
def _get_json_result(url, err_str, use_google):
    # need to do this here to prevent a series of complicated circular imports
    from .name_resolve import NameResolveError

    try:
        # Retrieve JSON response from Google maps API
        resp = urllib.request.urlopen(url, timeout=data.conf.remote_timeout)
        resp_data = json.loads(resp.read().decode("utf8"))

    except urllib.error.URLError as e:
        # This catches a timeout error, see:
        #   http://stackoverflow.com/questions/2712524/handling-urllib2s-timeout-python
        if isinstance(e.reason, socket.timeout):
            raise NameResolveError(err_str.format(msg="connection timed out")) from e
        else:
            raise NameResolveError(err_str.format(msg=e.reason)) from e

    except socket.timeout:
        # There are some cases where urllib2 does not catch socket.timeout
        # especially while receiving response data on an already previously
        # working request
        raise NameResolveError(err_str.format(msg="connection timed out"))

    if use_google:
        results = resp_data.get("results", [])

        if resp_data.get("status", None) != "OK":
            raise NameResolveError(
                err_str.format(msg="unknown failure with Google API")
            )

    else:  # OpenStreetMap returns a list
        results = resp_data

    if not results:
        raise NameResolveError(err_str.format(msg="no results returned"))

    return results
```
### 140 - astropy/coordinates/earth.py:

Start line: 957, End line: 976

```python
@format_doc(geodetic_base_doc)
class WGS84GeodeticRepresentation(BaseGeodeticRepresentation):
    """Representation of points in WGS84 3D geodetic coordinates."""

    _ellipsoid = "WGS84"


@format_doc(geodetic_base_doc)
class WGS72GeodeticRepresentation(BaseGeodeticRepresentation):
    """Representation of points in WGS72 3D geodetic coordinates."""

    _ellipsoid = "WGS72"


@format_doc(geodetic_base_doc)
class GRS80GeodeticRepresentation(BaseGeodeticRepresentation):
    """Representation of points in GRS80 3D geodetic coordinates."""

    _ellipsoid = "GRS80"
```
