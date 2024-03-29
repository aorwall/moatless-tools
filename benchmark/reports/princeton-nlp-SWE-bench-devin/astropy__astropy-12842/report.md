# astropy__astropy-12842

| **astropy/astropy** | `3a0cd2d8cd7b459cdc1e1b97a14f3040ccc1fffc` |
| ---- | ---- |
| **No of patches** | 2 |
| **All found context length** | - |
| **Any found context length** | 6685 |
| **Avg pos** | 5.5 |
| **Min pos** | 11 |
| **Max pos** | 11 |
| **Top file pos** | 5 |
| **Missing snippets** | 12 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/astropy/time/core.py b/astropy/time/core.py
--- a/astropy/time/core.py
+++ b/astropy/time/core.py
@@ -34,7 +34,7 @@
 
 from astropy.extern import _strptime
 
-__all__ = ['TimeBase', 'Time', 'TimeDelta', 'TimeInfo', 'update_leap_seconds',
+__all__ = ['TimeBase', 'Time', 'TimeDelta', 'TimeInfo', 'TimeInfoBase', 'update_leap_seconds',
            'TIME_SCALES', 'STANDARD_TIME_SCALES', 'TIME_DELTA_SCALES',
            'ScaleValueError', 'OperandTypeError', 'TimeDeltaMissingUnitWarning']
 
@@ -110,11 +110,13 @@ class _LeapSecondsCheck(enum.Enum):
 _LEAP_SECONDS_LOCK = threading.RLock()
 
 
-class TimeInfo(MixinInfo):
+class TimeInfoBase(MixinInfo):
     """
     Container for meta information like name, description, format.  This is
     required when the object is used as a mixin column within a table, but can
     be used as a general way to store meta information.
+
+    This base class is common between TimeInfo and TimeDeltaInfo.
     """
     attr_names = MixinInfo.attr_names | {'serialize_method'}
     _supports_indexing = True
@@ -133,6 +135,7 @@ class TimeInfo(MixinInfo):
     @property
     def _represent_as_dict_attrs(self):
         method = self.serialize_method[self._serialize_context]
+
         if method == 'formatted_value':
             out = ('value',)
         elif method == 'jd1_jd2':
@@ -182,7 +185,7 @@ def unit(self):
     # When Time has mean, std, min, max methods:
     # funcs = [lambda x: getattr(x, stat)() for stat_name in MixinInfo._stats])
 
-    def _construct_from_dict_base(self, map):
+    def _construct_from_dict(self, map):
         if 'jd1' in map and 'jd2' in map:
             # Initialize as JD but revert to desired format and out_subfmt (if needed)
             format = map.pop('format')
@@ -201,19 +204,6 @@ def _construct_from_dict_base(self, map):
 
         return out
 
-    def _construct_from_dict(self, map):
-        delta_ut1_utc = map.pop('_delta_ut1_utc', None)
-        delta_tdb_tt = map.pop('_delta_tdb_tt', None)
-
-        out = self._construct_from_dict_base(map)
-
-        if delta_ut1_utc is not None:
-            out._delta_ut1_utc = delta_ut1_utc
-        if delta_tdb_tt is not None:
-            out._delta_tdb_tt = delta_tdb_tt
-
-        return out
-
     def new_like(self, cols, length, metadata_conflicts='warn', name=None):
         """
         Return a new Time instance which is consistent with the input Time objects
@@ -276,11 +266,69 @@ def new_like(self, cols, length, metadata_conflicts='warn', name=None):
         return out
 
 
-class TimeDeltaInfo(TimeInfo):
-    _represent_as_dict_extra_attrs = ('format', 'scale')
+class TimeInfo(TimeInfoBase):
+    """
+    Container for meta information like name, description, format.  This is
+    required when the object is used as a mixin column within a table, but can
+    be used as a general way to store meta information.
+    """
+    def _represent_as_dict(self, attrs=None):
+        """Get the values for the parent ``attrs`` and return as a dict.
+
+        By default, uses '_represent_as_dict_attrs'.
+        """
+        map = super()._represent_as_dict(attrs=attrs)
+
+        # TODO: refactor these special cases into the TimeFormat classes?
+
+        # The datetime64 format requires special handling for ECSV (see #12840).
+        # The `value` has numpy dtype datetime64 but this is not an allowed
+        # datatype for ECSV. Instead convert to a string representation.
+        if (self._serialize_context == 'ecsv'
+                and map['format'] == 'datetime64'
+                and 'value' in map):
+            map['value'] = map['value'].astype('U')
+
+        # The datetime format is serialized as ISO with no loss of precision.
+        if map['format'] == 'datetime' and 'value' in map:
+            map['value'] = np.vectorize(lambda x: x.isoformat())(map['value'])
+
+        return map
 
     def _construct_from_dict(self, map):
-        return self._construct_from_dict_base(map)
+        # See comment above. May need to convert string back to datetime64.
+        # Note that _serialize_context is not set here so we just look for the
+        # string value directly.
+        if (map['format'] == 'datetime64'
+                and 'value' in map
+                and map['value'].dtype.kind == 'U'):
+            map['value'] = map['value'].astype('datetime64')
+
+        # Convert back to datetime objects for datetime format.
+        if map['format'] == 'datetime' and 'value' in map:
+            from datetime import datetime
+            map['value'] = np.vectorize(datetime.fromisoformat)(map['value'])
+
+        delta_ut1_utc = map.pop('_delta_ut1_utc', None)
+        delta_tdb_tt = map.pop('_delta_tdb_tt', None)
+
+        out = super()._construct_from_dict(map)
+
+        if delta_ut1_utc is not None:
+            out._delta_ut1_utc = delta_ut1_utc
+        if delta_tdb_tt is not None:
+            out._delta_tdb_tt = delta_tdb_tt
+
+        return out
+
+
+class TimeDeltaInfo(TimeInfoBase):
+    """
+    Container for meta information like name, description, format.  This is
+    required when the object is used as a mixin column within a table, but can
+    be used as a general way to store meta information.
+    """
+    _represent_as_dict_extra_attrs = ('format', 'scale')
 
     def new_like(self, cols, length, metadata_conflicts='warn', name=None):
         """
@@ -1815,7 +1863,7 @@ def earth_rotation_angle(self, longitude=None):
         and is rigorously corrected for polar motion.
         (except when ``longitude='tio'``).
 
-        """
+        """  # noqa
         if isinstance(longitude, str) and longitude == 'tio':
             longitude = 0
             include_tio = False
@@ -1877,7 +1925,7 @@ def sidereal_time(self, kind, longitude=None, model=None):
         the equator of the Celestial Intermediate Pole (CIP) and is rigorously
         corrected for polar motion (except when ``longitude='tio'`` or ``'greenwich'``).
 
-        """  # docstring is formatted below
+        """  # noqa (docstring is formatted below)
 
         if kind.lower() not in SIDEREAL_TIME_MODELS.keys():
             raise ValueError('The kind of sidereal time has to be {}'.format(
@@ -1929,7 +1977,7 @@ def _sid_time_or_earth_rot_ang(self, longitude, function, scales, include_tio=Tr
         `~astropy.coordinates.Longitude`
             Local sidereal time or Earth rotation angle, with units of hourangle.
 
-        """
+        """  # noqa
         from astropy.coordinates import Longitude, EarthLocation
         from astropy.coordinates.builtin_frames.utils import get_polar_motion
         from astropy.coordinates.matrix_utilities import rotation_matrix
@@ -1956,7 +2004,7 @@ def _sid_time_or_earth_rot_ang(self, longitude, function, scales, include_tio=Tr
             r = (rotation_matrix(longitude, 'z')
                  @ rotation_matrix(-yp, 'x', unit=u.radian)
                  @ rotation_matrix(-xp, 'y', unit=u.radian)
-                 @ rotation_matrix(theta+sp, 'z', unit=u.radian))
+                 @ rotation_matrix(theta + sp, 'z', unit=u.radian))
             # Solve for angle.
             angle = np.arctan2(r[..., 0, 1], r[..., 0, 0]) << u.radian
 
@@ -2781,7 +2829,6 @@ def __init__(self, left, right, op=None):
 def _check_leapsec():
     global _LEAP_SECONDS_CHECK
     if _LEAP_SECONDS_CHECK != _LeapSecondsCheck.DONE:
-        from astropy.utils import iers
         with _LEAP_SECONDS_LOCK:
             # There are three ways we can get here:
             # 1. First call (NOT_STARTED).
diff --git a/astropy/time/formats.py b/astropy/time/formats.py
--- a/astropy/time/formats.py
+++ b/astropy/time/formats.py
@@ -1745,7 +1745,7 @@ class TimeBesselianEpoch(TimeEpochDate):
 
     def _check_val_type(self, val1, val2):
         """Input value validation, typically overridden by derived classes"""
-        if hasattr(val1, 'to') and hasattr(val1, 'unit'):
+        if hasattr(val1, 'to') and hasattr(val1, 'unit') and val1.unit is not None:
             raise ValueError("Cannot use Quantities for 'byear' format, "
                              "as the interpretation would be ambiguous. "
                              "Use float with Besselian year instead. ")

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| astropy/time/core.py | 37 | 37 | 11 | 6 | 6685
| astropy/time/core.py | 113 | 113 | - | 6 | -
| astropy/time/core.py | 136 | 136 | - | 6 | -
| astropy/time/core.py | 185 | 185 | - | 6 | -
| astropy/time/core.py | 204 | 216 | - | 6 | -
| astropy/time/core.py | 279 | 283 | - | 6 | -
| astropy/time/core.py | 1818 | 1818 | - | 6 | -
| astropy/time/core.py | 1880 | 1880 | - | 6 | -
| astropy/time/core.py | 1932 | 1932 | - | 6 | -
| astropy/time/core.py | 1959 | 1959 | - | 6 | -
| astropy/time/core.py | 2784 | 2784 | - | 6 | -
| astropy/time/formats.py | 1748 | 1748 | - | 5 | -


## Problem Statement

```
No longer able to read BinnedTimeSeries with datetime column saved as ECSV after upgrading from 4.2.1 -> 5.0+
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
<!-- Provide a general description of the bug. -->
Hi, [This commit](https://github.com/astropy/astropy/commit/e807dbff9a5c72bdc42d18c7d6712aae69a0bddc) merged in PR #11569 breaks my ability to read an ECSV file created using Astropy v 4.2.1, BinnedTimeSeries class's write method, which has a datetime64 column. Downgrading astropy back to 4.2.1 fixes the issue because the strict type checking in line 177 of ecsv.py is not there.

Is there a reason why this strict type checking was added to ECSV? Is there a way to preserve reading and writing of ECSV files created with BinnedTimeSeries across versions? I am happy to make a PR on this if the strict type checking is allowed to be scaled back or we can add datetime64 as an allowed type. 

### Expected behavior
<!-- What did you expect to happen. -->

The file is read into a `BinnedTimeSeries` object from ecsv file without error.

### Actual behavior
<!-- What actually happened. -->
<!-- Was the output confusing or poorly described? -->

ValueError is produced and the file is not read because ECSV.py does not accept the datetime64 column.
`ValueError: datatype 'datetime64' of column 'time_bin_start' is not in allowed values ('bool', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64', 'float16', 'float32', 'float64', 'float128', 'string')`

### Steps to Reproduce
<!-- Ideally a code example could be provided so we can run it ourselves. -->
<!-- If you are pasting code, use triple backticks (\`\`\`) around
your code snippet. -->
<!-- If necessary, sanitize your screen output to be pasted so you do not
reveal secrets like tokens and passwords. -->

The file is read using:    
`BinnedTimeSeries.read('<file_path>', format='ascii.ecsv')`
which gives a long error. 


The file in question is a binned time series created by  `astropy.timeseries.aggregate_downsample`. which itself is a binned version of an `astropy.timeseries.TimeSeries` instance with some TESS data. (loaded via TimeSeries.from_pandas(Tess.set_index('datetime')). I.e., it has a datetime64 index.  The file was written using the classes own .write method in Astropy V4.2.1 from an instance of said class:   
`myBinnedTimeSeries.write('<file_path>',format='ascii.ecsv',overwrite=True)`

I'll attach a concatenated version of the file (as it contains private data). However, the relevant part from the header is on line 4:

\`\`\`
# %ECSV 0.9
# ---
# datatype:
# - {name: time_bin_start, datatype: datetime64}
\`\`\`

as you can see, the datatype is datetime64. This works fine with ECSV V0.9 but not V1.0 as some sort of strict type checking was added. 

### 
Full error log:
\`\`\`
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
Input In [3], in <module>
---> 49 tsrbin = BinnedTimeSeries.read('../Photometry/tsr_bin.dat', format='ascii.ecsv')

File ~/Apps/miniconda3/envs/py310_latest/lib/python3.10/site-packages/astropy/timeseries/binned.py:285, in BinnedTimeSeries.read(self, filename, time_bin_start_column, time_bin_end_column, time_bin_size_column, time_bin_size_unit, time_format, time_scale, format, *args, **kwargs)
    230 """
    231 Read and parse a file and returns a `astropy.timeseries.BinnedTimeSeries`.
    232 
   (...)
    279 
    280 """
    282 try:
    283 
    284     # First we try the readers defined for the BinnedTimeSeries class
--> 285     return super().read(filename, format=format, *args, **kwargs)
    287 except TypeError:
    288 
    289     # Otherwise we fall back to the default Table readers
    291     if time_bin_start_column is None:

File ~/Apps/miniconda3/envs/py310_latest/lib/python3.10/site-packages/astropy/table/connect.py:62, in TableRead.__call__(self, *args, **kwargs)
     59 units = kwargs.pop('units', None)
     60 descriptions = kwargs.pop('descriptions', None)
---> 62 out = self.registry.read(cls, *args, **kwargs)
     64 # For some readers (e.g., ascii.ecsv), the returned `out` class is not
     65 # guaranteed to be the same as the desired output `cls`.  If so,
     66 # try coercing to desired class without copying (io.registry.read
     67 # would normally do a copy).  The normal case here is swapping
     68 # Table <=> QTable.
     69 if cls is not out.__class__:

File ~/Apps/miniconda3/envs/py310_latest/lib/python3.10/site-packages/astropy/io/registry/core.py:199, in UnifiedInputRegistry.read(self, cls, format, cache, *args, **kwargs)
    195     format = self._get_valid_format(
    196         'read', cls, path, fileobj, args, kwargs)
    198 reader = self.get_reader(format, cls)
--> 199 data = reader(*args, **kwargs)
    201 if not isinstance(data, cls):
    202     # User has read with a subclass where only the parent class is
    203     # registered.  This returns the parent class, so try coercing
    204     # to desired subclass.
    205     try:

File ~/Apps/miniconda3/envs/py310_latest/lib/python3.10/site-packages/astropy/io/ascii/connect.py:18, in io_read(format, filename, **kwargs)
     16     format = re.sub(r'^ascii\.', '', format)
     17     kwargs['format'] = format
---> 18 return read(filename, **kwargs)

File ~/Apps/miniconda3/envs/py310_latest/lib/python3.10/site-packages/astropy/io/ascii/ui.py:376, in read(table, guess, **kwargs)
    374     else:
    375         reader = get_reader(**new_kwargs)
--> 376         dat = reader.read(table)
    377         _read_trace.append({'kwargs': copy.deepcopy(new_kwargs),
    378                             'Reader': reader.__class__,
    379                             'status': 'Success with specified Reader class '
    380                                       '(no guessing)'})
    382 # Static analysis (pyright) indicates `dat` might be left undefined, so just
    383 # to be sure define it at the beginning and check here.

File ~/Apps/miniconda3/envs/py310_latest/lib/python3.10/site-packages/astropy/io/ascii/core.py:1343, in BaseReader.read(self, table)
   1340 self.header.update_meta(self.lines, self.meta)
   1342 # Get the table column definitions
-> 1343 self.header.get_cols(self.lines)
   1345 # Make sure columns are valid
   1346 self.header.check_column_names(self.names, self.strict_names, self.guessing)

File ~/Apps/miniconda3/envs/py310_latest/lib/python3.10/site-packages/astropy/io/ascii/ecsv.py:177, in EcsvHeader.get_cols(self, lines)
    175 col.dtype = header_cols[col.name]['datatype']
    176 if col.dtype not in ECSV_DATATYPES:
--> 177     raise ValueError(f'datatype {col.dtype!r} of column {col.name!r} '
    178                      f'is not in allowed values {ECSV_DATATYPES}')
    180 # Subtype is written like "int64[2,null]" and we want to split this
    181 # out to "int64" and [2, None].
    182 subtype = col.subtype

ValueError: datatype 'datetime64' of column 'time_bin_start' is not in allowed values ('bool', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64', 'float16', 'float32', 'float64', 'float128', 'string')
\`\`\`
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
(For the version that does not work)
Python 3.10.2 | packaged by conda-forge | (main, Feb  1 2022, 19:28:35) [GCC 9.4.0]
Numpy 1.22.2
pyerfa 2.0.0.1
astropy 5.0.1
Scipy 1.8.0
Matplotlib 3.5.1

(For the version that does work)
Python 3.7.11 (default, Jul 27 2021, 14:32:16) [GCC 7.5.0]
Numpy 1.20.3
pyerfa 2.0.0.1
astropy 4.2.1
Scipy 1.7.0
Matplotlib 3.4.2


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 astropy/timeseries/binned.py | 282 | 346| 645 | 645 | 3254 | 
| 2 | 2 astropy/io/ascii/ecsv.py | 175 | 206| 378 | 1023 | 7451 | 
| 3 | 3 astropy/io/ascii/__init__.py | 7 | 48| 302 | 1325 | 7789 | 
| 4 | 4 astropy/utils/iers/iers.py | 12 | 77| 747 | 2072 | 19024 | 
| 5 | 4 astropy/timeseries/binned.py | 74 | 185| 1038 | 3110 | 19024 | 
| 6 | 4 astropy/io/ascii/ecsv.py | 412 | 452| 311 | 3421 | 19024 | 
| 7 | **5 astropy/time/formats.py** | 1 | 45| 443 | 3864 | 38186 | 
| 8 | 5 astropy/timeseries/binned.py | 226 | 280| 586 | 4450 | 38186 | 
| 9 | 5 astropy/io/ascii/ecsv.py | 7 | 31| 187 | 4637 | 38186 | 
| 10 | 5 astropy/io/ascii/ecsv.py | 236 | 340| 1111 | 5748 | 38186 | 
| **-> 11 <-** | **6 astropy/time/core.py** | 1 | 78| 937 | 6685 | 64169 | 
| 12 | 7 astropy/table/__init__.py | 3 | 14| 188 | 6873 | 65014 | 
| 13 | 7 astropy/table/__init__.py | 47 | 78| 338 | 7211 | 65014 | 
| 14 | 8 astropy/timeseries/core.py | 46 | 93| 376 | 7587 | 65661 | 
| 15 | 9 astropy/io/ascii/ui.py | 12 | 48| 193 | 7780 | 73357 | 
| 16 | 10 docs/conf.py | 1 | 101| 794 | 8574 | 77688 | 
| 17 | 11 astropy/io/misc/yaml.py | 59 | 112| 337 | 8911 | 80623 | 
| 18 | 12 astropy/utils/console.py | 1 | 39| 178 | 9089 | 88878 | 
| 19 | 13 astropy/io/fits/column.py | 3 | 77| 761 | 9850 | 111215 | 
| 20 | 13 astropy/io/fits/column.py | 78 | 128| 753 | 10603 | 111215 | 
| 21 | 13 astropy/io/ascii/ui.py | 480 | 553| 771 | 11374 | 111215 | 
| 22 | 14 astropy/io/misc/asdf/tags/time/time.py | 3 | 30| 194 | 11568 | 112209 | 
| 23 | 14 astropy/io/ascii/ecsv.py | 107 | 174| 601 | 12169 | 112209 | 
| 24 | 15 astropy/utils/compat/numpycompat.py | 7 | 24| 269 | 12438 | 112519 | 
| 25 | 16 astropy/io/votable/exceptions.py | 1423 | 1457| 223 | 12661 | 125851 | 
| 26 | 16 astropy/io/ascii/ecsv.py | 209 | 234| 228 | 12889 | 125851 | 
| 27 | 17 astropy/io/votable/tree.py | 5 | 69| 531 | 13420 | 154994 | 
| 28 | 18 astropy/wcs/docstrings.py | 2654 | 2821| 708 | 14128 | 176447 | 
| 29 | 19 astropy/io/ascii/cds.py | 196 | 300| 1129 | 15257 | 179532 | 
| 30 | 20 astropy/io/fits/hdu/table.py | 1001 | 1052| 676 | 15933 | 192641 | 


### Hint

```
I hope you don't mind me tagging you @taldcroft as it was your commit, maybe you can help me figure out if this is a bug or an evolution in `astropy.TimeSeries` that requires an alternative file format? I was pretty happy using ecsv formatted files to save complex data as they have been pretty stable, easy to visually inspect, and read in/out of scripts with astropy. 


[example_file.dat.txt](https://github.com/astropy/astropy/files/8043511/example_file.dat.txt)
(Also I had to add a .txt to the filename to allow github to put it up.)
@emirkmo - sorry, it was probably a mistake to make the reader be strict like that and raise an exception. Although that file is technically non-compliant with the ECSV spec, the reader should instead issue a warning but still carry on if possible (being liberal on input). I'll put in a PR to fix that.

The separate issue is that the `Time` object has a format of `datetime64` which leads to that unexpected numpy dtype in the output. I'm not immediately sure of what the right behavior for writing ECSV should be there. Maybe actually just `datetime64` as an allowed type, but that opens a small can of worms itself. Any thoughts @mhvk?

One curiosity @emirko is how you ended up with the timeseries object `time_bin_start` column having that `datetime64` format (`ts['time_bin_start'].format`). In my playing around it normally has `isot` format, which would not have led to this problem.
I would be happy to contribute this PR @taldcroft, as I have been working on it on a local copy anyway, and am keen to get it working. I currently monkey patched ecsv in my code to not raise, and it seems to work. If you let me know what the warning should say, I can make a first attempt. `UserWarning` of some sort? 

The `datetime64` comes through a chain:

 - Data is read into `pandas` with a `datetime64` index.
 - `TimeSeries` object is created using `.from_pandas`.
 - `aggregate_downsample` is used to turn this into a `BinnedTimeSeries`
 - `BinnedTimeSeries` object is written to an .ecsv file using its internal method.

Here is the raw code, although some of what you see may be illegible due to variable names. I didn't have easy access to the original raw data anymore, hence why I got stuck in trying to read it from the binned light curve. 
\`\`\`
perday = 12
Tess['datetime'] = pd.to_datetime(Tess.JD, unit='D', origin='julian')
ts = TimeSeries.from_pandas(Tess.set_index('datetime'))
tsb = aggregate_downsample(ts, time_bin_size=(1.0/perday)*u.day, 
                           time_bin_start=Time(beg.to_datetime64()), n_bins=nbin)
tsb.write('../Photometry/Tess_binned.ecsv', format='ascii.ecsv', overwrite=True)
\`\`\`
My PR above at least works for reading in the example file and my original file. Also passes my local tests on io module. 
Ouch, that is painful! Apart from changing the error to a warning (good idea!), I guess the writing somehow should change the data type from `datetime64` to `string`. Given that the format is stored as `datetime64`, I think this would still round-trip fine. I guess it would mean overwriting `_represent_as_dict` in `TimeInfo`.
> I guess it would mean overwriting _represent_as_dict in TimeInfo

That's where I got to, we need to be a little more careful about serializing `Time`. In some sense I'd like to just use `jd1_jd2` always for Time in ECSV (think of this as lossless serialization), but that change might not go down well.
Yes, what to pick is tricky: `jd1_jd2` is lossless, but much less readable.
As a user, I would expect the serializer picked to maintain the current time format in some way, or at least have a general mapping from all available  formats to the most nearby easily serializable ones if some of them are hard to work with. (Days as ISOT string, etc.)

ECSV seems designed to be human readable so I would find it strange if the format was majorly changed, although now I see that all other ways of saving the data use jd1_jd2. I assume a separate PR is needed for changing this.

Indeed, the other formats use `jd1_jd2`, but they are less explicitly meant to be human-readable.  I think this particular case of numpy datetime should not be too hard to fix, without actually changing how the file looks.
Agreed to keep the ECSV serialization as the `value` of the Time object.
```

## Patch

```diff
diff --git a/astropy/time/core.py b/astropy/time/core.py
--- a/astropy/time/core.py
+++ b/astropy/time/core.py
@@ -34,7 +34,7 @@
 
 from astropy.extern import _strptime
 
-__all__ = ['TimeBase', 'Time', 'TimeDelta', 'TimeInfo', 'update_leap_seconds',
+__all__ = ['TimeBase', 'Time', 'TimeDelta', 'TimeInfo', 'TimeInfoBase', 'update_leap_seconds',
            'TIME_SCALES', 'STANDARD_TIME_SCALES', 'TIME_DELTA_SCALES',
            'ScaleValueError', 'OperandTypeError', 'TimeDeltaMissingUnitWarning']
 
@@ -110,11 +110,13 @@ class _LeapSecondsCheck(enum.Enum):
 _LEAP_SECONDS_LOCK = threading.RLock()
 
 
-class TimeInfo(MixinInfo):
+class TimeInfoBase(MixinInfo):
     """
     Container for meta information like name, description, format.  This is
     required when the object is used as a mixin column within a table, but can
     be used as a general way to store meta information.
+
+    This base class is common between TimeInfo and TimeDeltaInfo.
     """
     attr_names = MixinInfo.attr_names | {'serialize_method'}
     _supports_indexing = True
@@ -133,6 +135,7 @@ class TimeInfo(MixinInfo):
     @property
     def _represent_as_dict_attrs(self):
         method = self.serialize_method[self._serialize_context]
+
         if method == 'formatted_value':
             out = ('value',)
         elif method == 'jd1_jd2':
@@ -182,7 +185,7 @@ def unit(self):
     # When Time has mean, std, min, max methods:
     # funcs = [lambda x: getattr(x, stat)() for stat_name in MixinInfo._stats])
 
-    def _construct_from_dict_base(self, map):
+    def _construct_from_dict(self, map):
         if 'jd1' in map and 'jd2' in map:
             # Initialize as JD but revert to desired format and out_subfmt (if needed)
             format = map.pop('format')
@@ -201,19 +204,6 @@ def _construct_from_dict_base(self, map):
 
         return out
 
-    def _construct_from_dict(self, map):
-        delta_ut1_utc = map.pop('_delta_ut1_utc', None)
-        delta_tdb_tt = map.pop('_delta_tdb_tt', None)
-
-        out = self._construct_from_dict_base(map)
-
-        if delta_ut1_utc is not None:
-            out._delta_ut1_utc = delta_ut1_utc
-        if delta_tdb_tt is not None:
-            out._delta_tdb_tt = delta_tdb_tt
-
-        return out
-
     def new_like(self, cols, length, metadata_conflicts='warn', name=None):
         """
         Return a new Time instance which is consistent with the input Time objects
@@ -276,11 +266,69 @@ def new_like(self, cols, length, metadata_conflicts='warn', name=None):
         return out
 
 
-class TimeDeltaInfo(TimeInfo):
-    _represent_as_dict_extra_attrs = ('format', 'scale')
+class TimeInfo(TimeInfoBase):
+    """
+    Container for meta information like name, description, format.  This is
+    required when the object is used as a mixin column within a table, but can
+    be used as a general way to store meta information.
+    """
+    def _represent_as_dict(self, attrs=None):
+        """Get the values for the parent ``attrs`` and return as a dict.
+
+        By default, uses '_represent_as_dict_attrs'.
+        """
+        map = super()._represent_as_dict(attrs=attrs)
+
+        # TODO: refactor these special cases into the TimeFormat classes?
+
+        # The datetime64 format requires special handling for ECSV (see #12840).
+        # The `value` has numpy dtype datetime64 but this is not an allowed
+        # datatype for ECSV. Instead convert to a string representation.
+        if (self._serialize_context == 'ecsv'
+                and map['format'] == 'datetime64'
+                and 'value' in map):
+            map['value'] = map['value'].astype('U')
+
+        # The datetime format is serialized as ISO with no loss of precision.
+        if map['format'] == 'datetime' and 'value' in map:
+            map['value'] = np.vectorize(lambda x: x.isoformat())(map['value'])
+
+        return map
 
     def _construct_from_dict(self, map):
-        return self._construct_from_dict_base(map)
+        # See comment above. May need to convert string back to datetime64.
+        # Note that _serialize_context is not set here so we just look for the
+        # string value directly.
+        if (map['format'] == 'datetime64'
+                and 'value' in map
+                and map['value'].dtype.kind == 'U'):
+            map['value'] = map['value'].astype('datetime64')
+
+        # Convert back to datetime objects for datetime format.
+        if map['format'] == 'datetime' and 'value' in map:
+            from datetime import datetime
+            map['value'] = np.vectorize(datetime.fromisoformat)(map['value'])
+
+        delta_ut1_utc = map.pop('_delta_ut1_utc', None)
+        delta_tdb_tt = map.pop('_delta_tdb_tt', None)
+
+        out = super()._construct_from_dict(map)
+
+        if delta_ut1_utc is not None:
+            out._delta_ut1_utc = delta_ut1_utc
+        if delta_tdb_tt is not None:
+            out._delta_tdb_tt = delta_tdb_tt
+
+        return out
+
+
+class TimeDeltaInfo(TimeInfoBase):
+    """
+    Container for meta information like name, description, format.  This is
+    required when the object is used as a mixin column within a table, but can
+    be used as a general way to store meta information.
+    """
+    _represent_as_dict_extra_attrs = ('format', 'scale')
 
     def new_like(self, cols, length, metadata_conflicts='warn', name=None):
         """
@@ -1815,7 +1863,7 @@ def earth_rotation_angle(self, longitude=None):
         and is rigorously corrected for polar motion.
         (except when ``longitude='tio'``).
 
-        """
+        """  # noqa
         if isinstance(longitude, str) and longitude == 'tio':
             longitude = 0
             include_tio = False
@@ -1877,7 +1925,7 @@ def sidereal_time(self, kind, longitude=None, model=None):
         the equator of the Celestial Intermediate Pole (CIP) and is rigorously
         corrected for polar motion (except when ``longitude='tio'`` or ``'greenwich'``).
 
-        """  # docstring is formatted below
+        """  # noqa (docstring is formatted below)
 
         if kind.lower() not in SIDEREAL_TIME_MODELS.keys():
             raise ValueError('The kind of sidereal time has to be {}'.format(
@@ -1929,7 +1977,7 @@ def _sid_time_or_earth_rot_ang(self, longitude, function, scales, include_tio=Tr
         `~astropy.coordinates.Longitude`
             Local sidereal time or Earth rotation angle, with units of hourangle.
 
-        """
+        """  # noqa
         from astropy.coordinates import Longitude, EarthLocation
         from astropy.coordinates.builtin_frames.utils import get_polar_motion
         from astropy.coordinates.matrix_utilities import rotation_matrix
@@ -1956,7 +2004,7 @@ def _sid_time_or_earth_rot_ang(self, longitude, function, scales, include_tio=Tr
             r = (rotation_matrix(longitude, 'z')
                  @ rotation_matrix(-yp, 'x', unit=u.radian)
                  @ rotation_matrix(-xp, 'y', unit=u.radian)
-                 @ rotation_matrix(theta+sp, 'z', unit=u.radian))
+                 @ rotation_matrix(theta + sp, 'z', unit=u.radian))
             # Solve for angle.
             angle = np.arctan2(r[..., 0, 1], r[..., 0, 0]) << u.radian
 
@@ -2781,7 +2829,6 @@ def __init__(self, left, right, op=None):
 def _check_leapsec():
     global _LEAP_SECONDS_CHECK
     if _LEAP_SECONDS_CHECK != _LeapSecondsCheck.DONE:
-        from astropy.utils import iers
         with _LEAP_SECONDS_LOCK:
             # There are three ways we can get here:
             # 1. First call (NOT_STARTED).
diff --git a/astropy/time/formats.py b/astropy/time/formats.py
--- a/astropy/time/formats.py
+++ b/astropy/time/formats.py
@@ -1745,7 +1745,7 @@ class TimeBesselianEpoch(TimeEpochDate):
 
     def _check_val_type(self, val1, val2):
         """Input value validation, typically overridden by derived classes"""
-        if hasattr(val1, 'to') and hasattr(val1, 'unit'):
+        if hasattr(val1, 'to') and hasattr(val1, 'unit') and val1.unit is not None:
             raise ValueError("Cannot use Quantities for 'byear' format, "
                              "as the interpretation would be ambiguous. "
                              "Use float with Besselian year instead. ")

```

## Test Patch

```diff
diff --git a/astropy/io/ascii/tests/test_ecsv.py b/astropy/io/ascii/tests/test_ecsv.py
--- a/astropy/io/ascii/tests/test_ecsv.py
+++ b/astropy/io/ascii/tests/test_ecsv.py
@@ -822,13 +822,13 @@ def _make_expected_values(cols):
      'name': '2-d regular array',
      'subtype': 'float16[2,2]'}]
 
-cols['scalar object'] = np.array([{'a': 1}, {'b':2}], dtype=object)
+cols['scalar object'] = np.array([{'a': 1}, {'b': 2}], dtype=object)
 exps['scalar object'] = [
     {'datatype': 'string', 'name': 'scalar object', 'subtype': 'json'}]
 
 cols['1-d object'] = np.array(
-    [[{'a': 1}, {'b':2}],
-     [{'a': 1}, {'b':2}]], dtype=object)
+    [[{'a': 1}, {'b': 2}],
+     [{'a': 1}, {'b': 2}]], dtype=object)
 exps['1-d object'] = [
     {'datatype': 'string',
      'name': '1-d object',
@@ -966,7 +966,7 @@ def test_masked_vals_in_array_subtypes():
     assert t2.colnames == t.colnames
     for name in t2.colnames:
         assert t2[name].dtype == t[name].dtype
-        assert type(t2[name]) is type(t[name])
+        assert type(t2[name]) is type(t[name])  # noqa
         for val1, val2 in zip(t2[name], t[name]):
             if isinstance(val1, np.ndarray):
                 assert val1.dtype == val2.dtype
diff --git a/astropy/time/tests/test_basic.py b/astropy/time/tests/test_basic.py
--- a/astropy/time/tests/test_basic.py
+++ b/astropy/time/tests/test_basic.py
@@ -6,6 +6,7 @@
 import datetime
 from copy import deepcopy
 from decimal import Decimal, localcontext
+from io import StringIO
 
 import numpy as np
 import pytest
@@ -20,7 +21,7 @@
 from astropy.coordinates import EarthLocation
 from astropy import units as u
 from astropy.table import Column, Table
-from astropy.utils.compat.optional_deps import HAS_PYTZ  # noqa
+from astropy.utils.compat.optional_deps import HAS_PYTZ, HAS_H5PY  # noqa
 
 
 allclose_jd = functools.partial(np.allclose, rtol=np.finfo(float).eps, atol=0)
@@ -2221,6 +2222,66 @@ def test_ymdhms_output():
     assert t.ymdhms.year == 2015
 
 
+@pytest.mark.parametrize('fmt', TIME_FORMATS)
+def test_write_every_format_to_ecsv(fmt):
+    """Test special-case serialization of certain Time formats"""
+    t = Table()
+    # Use a time that tests the default serialization of the time format
+    tm = (Time('2020-01-01')
+          + [[1, 1 / 7],
+             [3, 4.5]] * u.s)
+    tm.format = fmt
+    t['a'] = tm
+    out = StringIO()
+    t.write(out, format='ascii.ecsv')
+    t2 = Table.read(out.getvalue(), format='ascii.ecsv')
+    assert t['a'].format == t2['a'].format
+    # Some loss of precision in the serialization
+    assert not np.all(t['a'] == t2['a'])
+    # But no loss in the format representation
+    assert np.all(t['a'].value == t2['a'].value)
+
+
+@pytest.mark.parametrize('fmt', TIME_FORMATS)
+def test_write_every_format_to_fits(fmt, tmp_path):
+    """Test special-case serialization of certain Time formats"""
+    t = Table()
+    # Use a time that tests the default serialization of the time format
+    tm = (Time('2020-01-01')
+          + [[1, 1 / 7],
+             [3, 4.5]] * u.s)
+    tm.format = fmt
+    t['a'] = tm
+    out = tmp_path / 'out.fits'
+    t.write(out, format='fits')
+    t2 = Table.read(out, format='fits', astropy_native=True)
+    # Currently the format is lost in FITS so set it back
+    t2['a'].format = fmt
+    # No loss of precision in the serialization or representation
+    assert np.all(t['a'] == t2['a'])
+    assert np.all(t['a'].value == t2['a'].value)
+
+
+@pytest.mark.skipif(not HAS_H5PY, reason='Needs h5py')
+@pytest.mark.parametrize('fmt', TIME_FORMATS)
+def test_write_every_format_to_hdf5(fmt, tmp_path):
+    """Test special-case serialization of certain Time formats"""
+    t = Table()
+    # Use a time that tests the default serialization of the time format
+    tm = (Time('2020-01-01')
+          + [[1, 1 / 7],
+             [3, 4.5]] * u.s)
+    tm.format = fmt
+    t['a'] = tm
+    out = tmp_path / 'out.h5'
+    t.write(str(out), format='hdf5', path='root', serialize_meta=True)
+    t2 = Table.read(str(out), format='hdf5', path='root')
+    assert t['a'].format == t2['a'].format
+    # No loss of precision in the serialization or representation
+    assert np.all(t['a'] == t2['a'])
+    assert np.all(t['a'].value == t2['a'].value)
+
+
 # There are two stages of validation now - one on input into a format, so that
 # the format conversion code has tidy matched arrays to work with, and the
 # other when object construction does not go through a format object. Or at

```


## Code snippets

### 1 - astropy/timeseries/binned.py:

Start line: 282, End line: 346

```python
@autocheck_required_columns
class BinnedTimeSeries(BaseTimeSeries):

    @classmethod
    def read(self, filename, time_bin_start_column=None, time_bin_end_column=None,
             time_bin_size_column=None, time_bin_size_unit=None, time_format=None, time_scale=None,
             format=None, *args, **kwargs):

        try:

            # First we try the readers defined for the BinnedTimeSeries class
            return super().read(filename, format=format, *args, **kwargs)

        except TypeError:

            # Otherwise we fall back to the default Table readers

            if time_bin_start_column is None:
                raise ValueError("``time_bin_start_column`` should be provided since the default Table readers are being used.")
            if time_bin_end_column is None and time_bin_size_column is None:
                raise ValueError("Either `time_bin_end_column` or `time_bin_size_column` should be provided.")
            elif time_bin_end_column is not None and time_bin_size_column is not None:
                raise ValueError("Cannot specify both `time_bin_end_column` and `time_bin_size_column`.")

            table = Table.read(filename, format=format, *args, **kwargs)

            if time_bin_start_column in table.colnames:
                time_bin_start = Time(table.columns[time_bin_start_column],
                                      scale=time_scale, format=time_format)
                table.remove_column(time_bin_start_column)
            else:
                raise ValueError(f"Bin start time column '{time_bin_start_column}' not found in the input data.")

            if time_bin_end_column is not None:

                if time_bin_end_column in table.colnames:
                    time_bin_end = Time(table.columns[time_bin_end_column],
                                        scale=time_scale, format=time_format)
                    table.remove_column(time_bin_end_column)
                else:
                    raise ValueError(f"Bin end time column '{time_bin_end_column}' not found in the input data.")

                time_bin_size = None

            elif time_bin_size_column is not None:

                if time_bin_size_column in table.colnames:
                    time_bin_size = table.columns[time_bin_size_column]
                    table.remove_column(time_bin_size_column)
                else:
                    raise ValueError(f"Bin size column '{time_bin_size_column}' not found in the input data.")

                if time_bin_size.unit is None:
                    if time_bin_size_unit is None or not isinstance(time_bin_size_unit, u.UnitBase):
                        raise ValueError("The bin size unit should be specified as an astropy Unit using ``time_bin_size_unit``.")
                    time_bin_size = time_bin_size * time_bin_size_unit
                else:
                    time_bin_size = u.Quantity(time_bin_size)

                time_bin_end = None

            if time_bin_start.isscalar and time_bin_size.isscalar:
                return BinnedTimeSeries(data=table,
                                    time_bin_start=time_bin_start,
                                    time_bin_end=time_bin_end,
                                    time_bin_size=time_bin_size,
                                    n_bins=len(table))
            else:
                return BinnedTimeSeries(data=table,
                                    time_bin_start=time_bin_start,
                                    time_bin_end=time_bin_end,
                                    time_bin_size=time_bin_size)
```
### 2 - astropy/io/ascii/ecsv.py:

Start line: 175, End line: 206

```python
class EcsvHeader(basic.BasicHeader):

    def get_cols(self, lines):
        # ... other code
        for col in self.cols:
            for attr in ('description', 'format', 'unit', 'meta', 'subtype'):
                if attr in header_cols[col.name]:
                    setattr(col, attr, header_cols[col.name][attr])

            col.dtype = header_cols[col.name]['datatype']
            # Warn if col dtype is not a valid ECSV datatype, but allow reading for
            # back-compatibility with existing older files that have numpy datatypes
            # like datetime64 or object or python str, which are not in the ECSV standard.
            if col.dtype not in ECSV_DATATYPES:
                msg = (f'unexpected datatype {col.dtype!r} of column {col.name!r} '
                       f'is not in allowed ECSV datatypes {ECSV_DATATYPES}. '
                       'Using anyway as a numpy dtype but beware since unexpected '
                       'results are possible.')
                warnings.warn(msg, category=InvalidEcsvDatatypeWarning)

            # Subtype is written like "int64[2,null]" and we want to split this
            # out to "int64" and [2, None].
            subtype = col.subtype
            if subtype and '[' in subtype:
                idx = subtype.index('[')
                col.subtype = subtype[:idx]
                col.shape = json.loads(subtype[idx:])

            # Convert ECSV "string" to numpy "str"
            for attr in ('dtype', 'subtype'):
                if getattr(col, attr) == 'string':
                    setattr(col, attr, 'str')

            # ECSV subtype of 'json' maps to numpy 'object' dtype
            if col.subtype == 'json':
                col.subtype = 'object'
```
### 3 - astropy/io/ascii/__init__.py:

Start line: 7, End line: 48

```python
from .core import (InconsistentTableError,
                   ParameterError,
                   NoType, StrType, NumType, FloatType, IntType, AllType,
                   Column,
                   BaseInputter, ContinuationLinesInputter,
                   BaseHeader,
                   BaseData,
                   BaseOutputter, TableOutputter,
                   BaseReader,
                   BaseSplitter, DefaultSplitter, WhitespaceSplitter,
                   convert_numpy,
                   masked
                   )
from .basic import (Basic, BasicHeader, BasicData,
                    Rdb,
                    Csv,
                    Tab,
                    NoHeader,
                    CommentedHeader)
from .fastbasic import (FastBasic,
                        FastCsv,
                        FastTab,
                        FastNoHeader,
                        FastCommentedHeader,
                        FastRdb)
from .cds import Cds
from .mrt import Mrt
from .ecsv import Ecsv
from .latex import Latex, AASTex, latexdicts
from .html import HTML
from .ipac import Ipac
from .daophot import Daophot
from .qdp import QDP
from .sextractor import SExtractor
from .fixedwidth import (FixedWidth, FixedWidthNoHeader,
                         FixedWidthTwoLine, FixedWidthSplitter,
                         FixedWidthHeader, FixedWidthData)
from .rst import RST
from .ui import (set_guess, get_reader, read, get_writer, write, get_read_trace)

from . import connect
```
### 4 - astropy/utils/iers/iers.py:

Start line: 12, End line: 77

```python
import re
from datetime import datetime
from warnings import warn
from urllib.parse import urlparse

import numpy as np
import erfa

from astropy.time import Time, TimeDelta
from astropy import config as _config
from astropy import units as u
from astropy.table import QTable, MaskedColumn
from astropy.utils.data import (get_pkg_data_filename, clear_download_cache,
                                is_url_in_cache, get_readable_fileobj)
from astropy.utils.state import ScienceState
from astropy import utils
from astropy.utils.exceptions import AstropyWarning

__all__ = ['Conf', 'conf', 'earth_orientation_table',
           'IERS', 'IERS_B', 'IERS_A', 'IERS_Auto',
           'FROM_IERS_B', 'FROM_IERS_A', 'FROM_IERS_A_PREDICTION',
           'TIME_BEFORE_IERS_RANGE', 'TIME_BEYOND_IERS_RANGE',
           'IERS_A_FILE', 'IERS_A_URL', 'IERS_A_URL_MIRROR', 'IERS_A_README',
           'IERS_B_FILE', 'IERS_B_URL', 'IERS_B_README',
           'IERSRangeError', 'IERSStaleWarning',
           'LeapSeconds', 'IERS_LEAP_SECOND_FILE', 'IERS_LEAP_SECOND_URL',
           'IETF_LEAP_SECOND_URL']

# IERS-A default file name, URL, and ReadMe with content description
IERS_A_FILE = 'finals2000A.all'
IERS_A_URL = 'https://maia.usno.navy.mil/ser7/finals2000A.all'
IERS_A_URL_MIRROR = 'https://datacenter.iers.org/data/9/finals2000A.all'
IERS_A_README = get_pkg_data_filename('data/ReadMe.finals2000A')

# IERS-B default file name, URL, and ReadMe with content description
IERS_B_FILE = get_pkg_data_filename('data/eopc04_IAU2000.62-now')
IERS_B_URL = 'http://hpiers.obspm.fr/iers/eop/eopc04/eopc04_IAU2000.62-now'
IERS_B_README = get_pkg_data_filename('data/ReadMe.eopc04_IAU2000')

# LEAP SECONDS default file name, URL, and alternative format/URL
IERS_LEAP_SECOND_FILE = get_pkg_data_filename('data/Leap_Second.dat')
IERS_LEAP_SECOND_URL = 'https://hpiers.obspm.fr/iers/bul/bulc/Leap_Second.dat'
IETF_LEAP_SECOND_URL = 'https://www.ietf.org/timezones/data/leap-seconds.list'

# Status/source values returned by IERS.ut1_utc
FROM_IERS_B = 0
FROM_IERS_A = 1
FROM_IERS_A_PREDICTION = 2
TIME_BEFORE_IERS_RANGE = -1
TIME_BEYOND_IERS_RANGE = -2

MJD_ZERO = 2400000.5

INTERPOLATE_ERROR = """\
interpolating from IERS_Auto using predictive values that are more
than {0} days old.

Normally you should not see this error because this class
automatically downloads the latest IERS-A table.  Perhaps you are
offline?  If you understand what you are doing then this error can be
suppressed by setting the auto_max_age configuration variable to
``None``:

  from astropy.utils.iers import conf
  conf.auto_max_age = None
"""
```
### 5 - astropy/timeseries/binned.py:

Start line: 74, End line: 185

```python
@autocheck_required_columns
class BinnedTimeSeries(BaseTimeSeries):

    def __init__(self, data=None, *, time_bin_start=None, time_bin_end=None,
                 time_bin_size=None, n_bins=None, **kwargs):

        super().__init__(data=data, **kwargs)

        # For some operations, an empty time series needs to be created, then
        # columns added one by one. We should check that when columns are added
        # manually, time is added first and is of the right type.
        if (data is None and time_bin_start is None and time_bin_end is None and
                time_bin_size is None and n_bins is None):
            self._required_columns_relax = True
            return

        # First if time_bin_start and time_bin_end have been given in the table data, we
        # should extract them and treat them as if they had been passed as
        # keyword arguments.

        if 'time_bin_start' in self.colnames:
            if time_bin_start is None:
                time_bin_start = self.columns['time_bin_start']
            else:
                raise TypeError("'time_bin_start' has been given both in the table "
                                "and as a keyword argument")

        if 'time_bin_size' in self.colnames:
            if time_bin_size is None:
                time_bin_size = self.columns['time_bin_size']
            else:
                raise TypeError("'time_bin_size' has been given both in the table "
                                "and as a keyword argument")

        if time_bin_start is None:
            raise TypeError("'time_bin_start' has not been specified")

        if time_bin_end is None and time_bin_size is None:
            raise TypeError("Either 'time_bin_size' or 'time_bin_end' should be specified")

        if not isinstance(time_bin_start, (Time, TimeDelta)):
            time_bin_start = Time(time_bin_start)

        if time_bin_end is not None and not isinstance(time_bin_end, (Time, TimeDelta)):
            time_bin_end = Time(time_bin_end)

        if time_bin_size is not None and not isinstance(time_bin_size, (Quantity, TimeDelta)):
            raise TypeError("'time_bin_size' should be a Quantity or a TimeDelta")

        if isinstance(time_bin_size, TimeDelta):
            time_bin_size = time_bin_size.sec * u.s

        if n_bins is not None and time_bin_size is not None:
            if not (time_bin_start.isscalar and time_bin_size.isscalar):
                raise TypeError("'n_bins' cannot be specified if 'time_bin_start' or "
                                "'time_bin_size' are not scalar'")

        if time_bin_start.isscalar:

            # We interpret this as meaning that this is the start of the
            # first bin and that the bins are contiguous. In this case,
            # we require time_bin_size to be specified.

            if time_bin_size is None:
                raise TypeError("'time_bin_start' is scalar, so 'time_bin_size' is required")

            if time_bin_size.isscalar:
                if data is not None:
                    if n_bins is not None:
                        if n_bins != len(self):
                            raise TypeError("'n_bins' has been given and it is not the "
                                            "same length as the input data.")
                    else:
                        n_bins = len(self)

                time_bin_size = np.repeat(time_bin_size, n_bins)

            time_delta = np.cumsum(time_bin_size)
            time_bin_end = time_bin_start + time_delta

            # Now shift the array so that the first entry is 0
            time_delta = np.roll(time_delta, 1)
            time_delta[0] = 0. * u.s

            # Make time_bin_start into an array
            time_bin_start = time_bin_start + time_delta

        else:

            if len(self.colnames) > 0 and len(time_bin_start) != len(self):
                raise ValueError("Length of 'time_bin_start' ({}) should match "
                                 "table length ({})".format(len(time_bin_start), len(self)))

            if time_bin_end is not None:
                if time_bin_end.isscalar:
                    times = time_bin_start.copy()
                    times[:-1] = times[1:]
                    times[-1] = time_bin_end
                    time_bin_end = times
                time_bin_size = (time_bin_end - time_bin_start).sec * u.s

        if time_bin_size.isscalar:
            time_bin_size = np.repeat(time_bin_size, len(self))

        with self._delay_required_column_checks():

            if 'time_bin_start' in self.colnames:
                self.remove_column('time_bin_start')

            if 'time_bin_size' in self.colnames:
                self.remove_column('time_bin_size')

            self.add_column(time_bin_start, index=0, name='time_bin_start')
            self.add_index('time_bin_start')
            self.add_column(time_bin_size, index=1, name='time_bin_size')
```
### 6 - astropy/io/ascii/ecsv.py:

Start line: 412, End line: 452

```python
class Ecsv(basic.Basic):
    """ECSV (Enhanced Character Separated Values) format table.

    Th ECSV format allows for specification of key table and column meta-data, in
    particular the data type and unit.

    See: https://github.com/astropy/astropy-APEs/blob/main/APE6.rst

    Examples
    --------

    >>> from astropy.table import Table
    >>> ecsv_content = '''# %ECSV 0.9
    ... # ---
    ... # datatype:
    ... # - {name: a, unit: m / s, datatype: int64, format: '%03d'}
    ... # - {name: b, unit: km, datatype: int64, description: This is column b}
    ... a b
    ... 001 2
    ... 004 3
    ... '''

    >>> Table.read(ecsv_content, format='ascii.ecsv')
    <Table length=2>
      a     b
    m / s   km
    int64 int64
    ----- -----
      001     2
      004     3

    """
    _format_name = 'ecsv'
    _description = 'Enhanced CSV'
    _io_registry_suffix = '.ecsv'

    header_class = EcsvHeader
    data_class = EcsvData
    outputter_class = EcsvOutputter

    max_ndim = None  # No limit on column dimensionality
```
### 7 - astropy/time/formats.py:

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
### 8 - astropy/timeseries/binned.py:

Start line: 226, End line: 280

```python
@autocheck_required_columns
class BinnedTimeSeries(BaseTimeSeries):

    @classmethod
    def read(self, filename, time_bin_start_column=None, time_bin_end_column=None,
             time_bin_size_column=None, time_bin_size_unit=None, time_format=None, time_scale=None,
             format=None, *args, **kwargs):
        """
        Read and parse a file and returns a `astropy.timeseries.BinnedTimeSeries`.

        This method uses the unified I/O infrastructure in Astropy which makes
        it easy to define readers/writers for various classes
        (https://docs.astropy.org/en/stable/io/unified.html). By default, this
        method will try and use readers defined specifically for the
        `astropy.timeseries.BinnedTimeSeries` class - however, it is also
        possible to use the ``format`` keyword to specify formats defined for
        the `astropy.table.Table` class - in this case, you will need to also
        provide the column names for column containing the start times for the
        bins, as well as other column names (see the Parameters section below
        for details)::

            >>> from astropy.timeseries.binned import BinnedTimeSeries
            >>> ts = BinnedTimeSeries.read('binned.dat', format='ascii.ecsv',
            ...                            time_bin_start_column='date_start',
            ...                            time_bin_end_column='date_end')  # doctest: +SKIP

        Parameters
        ----------
        filename : str
            File to parse.
        format : str
            File format specifier.
        time_bin_start_column : str
            The name of the column with the start time for each bin.
        time_bin_end_column : str, optional
            The name of the column with the end time for each bin. Either this
            option or ``time_bin_size_column`` should be specified.
        time_bin_size_column : str, optional
            The name of the column with the size for each bin. Either this
            option or ``time_bin_end_column`` should be specified.
        time_bin_size_unit : `astropy.units.Unit`, optional
            If ``time_bin_size_column`` is specified but does not have a unit
            set in the table, you can specify the unit manually.
        time_format : str, optional
            The time format for the start and end columns.
        time_scale : str, optional
            The time scale for the start and end columns.
        *args : tuple, optional
            Positional arguments passed through to the data reader.
        **kwargs : dict, optional
            Keyword arguments passed through to the data reader.

        Returns
        -------
        out : `astropy.timeseries.binned.BinnedTimeSeries`
            BinnedTimeSeries corresponding to the file.

        """
        # ... other code
```
### 9 - astropy/io/ascii/ecsv.py:

Start line: 7, End line: 31

```python
import re
from collections import OrderedDict
import warnings
import json

import numpy as np

from . import core, basic
from astropy.table import meta, serialize
from astropy.utils.data_info import serialize_context_as
from astropy.utils.exceptions import AstropyUserWarning
from astropy.io.ascii.core import convert_numpy

ECSV_VERSION = '1.0'
DELIMITERS = (' ', ',')
ECSV_DATATYPES = (
    'bool', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16',
    'uint32', 'uint64', 'float16', 'float32', 'float64',
    'float128', 'string')  # Raise warning if not one of these standard dtypes


class InvalidEcsvDatatypeWarning(AstropyUserWarning):
    """
    ECSV specific Astropy warning class.
    """
```
### 10 - astropy/io/ascii/ecsv.py:

Start line: 236, End line: 340

```python
class EcsvOutputter(core.TableOutputter):

    def _convert_vals(self, cols):
        """READ: Convert str_vals in `cols` to final arrays with correct dtypes.

        This is adapted from ``BaseOutputter._convert_vals``. In the case of ECSV
        there is no guessing and all types are known in advance. A big change
        is handling the possibility of JSON-encoded values, both unstructured
        object data and structured values that may contain masked data.
        """
        for col in cols:
            try:
                # 1-d or N-d object columns are serialized as JSON.
                if col.subtype == 'object':
                    _check_dtype_is_str(col)
                    col_vals = [json.loads(val) for val in col.str_vals]
                    col.data = np.empty([len(col_vals)] + col.shape, dtype=object)
                    col.data[...] = col_vals

                # Variable length arrays with shape (n, m, ..., *) for fixed
                # n, m, .. and variable in last axis. Masked values here are
                # not currently supported.
                elif col.shape and col.shape[-1] is None:
                    _check_dtype_is_str(col)

                    # Empty (blank) values in original ECSV are changed to "0"
                    # in str_vals with corresponding col.mask being created and
                    # set accordingly. Instead use an empty list here.
                    if hasattr(col, 'mask'):
                        for idx in np.nonzero(col.mask)[0]:
                            col.str_vals[idx] = '[]'

                    # Remake as a 1-d object column of numpy ndarrays or
                    # MaskedArray using the datatype specified in the ECSV file.
                    col_vals = []
                    for str_val in col.str_vals:
                        obj_val = json.loads(str_val)  # list or nested lists
                        try:
                            arr_val = np.array(obj_val, dtype=col.subtype)
                        except TypeError:
                            # obj_val has entries that are inconsistent with
                            # dtype. For a valid ECSV file the only possibility
                            # is None values (indicating missing values).
                            data = np.array(obj_val, dtype=object)
                            # Replace all the None with an appropriate fill value
                            mask = (data == None)  # noqa: E711
                            kind = np.dtype(col.subtype).kind
                            data[mask] = {'U': '', 'S': b''}.get(kind, 0)
                            arr_val = np.ma.array(data.astype(col.subtype), mask=mask)

                        col_vals.append(arr_val)

                    col.shape = ()
                    col.dtype = np.dtype(object)
                    # np.array(col_vals_arr, dtype=object) fails ?? so this workaround:
                    col.data = np.empty(len(col_vals), dtype=object)
                    col.data[:] = col_vals

                # Multidim columns with consistent shape (n, m, ...). These
                # might be masked.
                elif col.shape:
                    _check_dtype_is_str(col)

                    # Change empty (blank) values in original ECSV to something
                    # like "[[null, null],[null,null]]" so subsequent JSON
                    # decoding works. Delete `col.mask` so that later code in
                    # core TableOutputter.__call__() that deals with col.mask
                    # does not run (since handling is done here already).
                    if hasattr(col, 'mask'):
                        all_none_arr = np.full(shape=col.shape, fill_value=None, dtype=object)
                        all_none_json = json.dumps(all_none_arr.tolist())
                        for idx in np.nonzero(col.mask)[0]:
                            col.str_vals[idx] = all_none_json
                        del col.mask

                    col_vals = [json.loads(val) for val in col.str_vals]
                    # Make a numpy object array of col_vals to look for None
                    # (masked values)
                    data = np.array(col_vals, dtype=object)
                    mask = (data == None)  # noqa: E711
                    if not np.any(mask):
                        # No None's, just convert to required dtype
                        col.data = data.astype(col.subtype)
                    else:
                        # Replace all the None with an appropriate fill value
                        kind = np.dtype(col.subtype).kind
                        data[mask] = {'U': '', 'S': b''}.get(kind, 0)
                        # Finally make a MaskedArray with the filled data + mask
                        col.data = np.ma.array(data.astype(col.subtype), mask=mask)

                # Regular scalar value column
                else:
                    if col.subtype:
                        warnings.warn(f'unexpected subtype {col.subtype!r} set for column '
                                      f'{col.name!r}, using dtype={col.dtype!r} instead.',
                                      category=InvalidEcsvDatatypeWarning)
                    converter_func, _ = convert_numpy(col.dtype)
                    col.data = converter_func(col.str_vals)

                if col.data.shape[1:] != tuple(col.shape):
                    raise ValueError('shape mismatch between value and column specifier')

            except json.JSONDecodeError:
                raise ValueError(f'column {col.name!r} failed to convert: '
                                 'column value is not valid JSON')
            except Exception as exc:
                raise ValueError(f'column {col.name!r} failed to convert: {exc}')
```
### 11 - astropy/time/core.py:

Start line: 1, End line: 78

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

from astropy.extern import _strptime

__all__ = ['TimeBase', 'Time', 'TimeDelta', 'TimeInfo', 'update_leap_seconds',
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
