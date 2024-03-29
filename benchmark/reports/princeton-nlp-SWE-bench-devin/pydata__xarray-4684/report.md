# pydata__xarray-4684

| **pydata/xarray** | `0f1eb96c924bad60ea87edd9139325adabfefa33` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | 3073 |
| **Avg pos** | 36.0 |
| **Min pos** | 5 |
| **Max pos** | 21 |
| **Top file pos** | 5 |
| **Missing snippets** | 6 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/xarray/coding/times.py b/xarray/coding/times.py
--- a/xarray/coding/times.py
+++ b/xarray/coding/times.py
@@ -26,6 +26,7 @@
 _STANDARD_CALENDARS = {"standard", "gregorian", "proleptic_gregorian"}
 
 _NS_PER_TIME_DELTA = {
+    "ns": 1,
     "us": int(1e3),
     "ms": int(1e6),
     "s": int(1e9),
@@ -35,7 +36,15 @@
 }
 
 TIME_UNITS = frozenset(
-    ["days", "hours", "minutes", "seconds", "milliseconds", "microseconds"]
+    [
+        "days",
+        "hours",
+        "minutes",
+        "seconds",
+        "milliseconds",
+        "microseconds",
+        "nanoseconds",
+    ]
 )
 
 
@@ -44,6 +53,7 @@ def _netcdf_to_numpy_timeunit(units):
     if not units.endswith("s"):
         units = "%ss" % units
     return {
+        "nanoseconds": "ns",
         "microseconds": "us",
         "milliseconds": "ms",
         "seconds": "s",
@@ -151,21 +161,22 @@ def _decode_datetime_with_pandas(flat_num_dates, units, calendar):
         # strings, in which case we fall back to using cftime
         raise OutOfBoundsDatetime
 
-    # fixes: https://github.com/pydata/pandas/issues/14068
-    # these lines check if the the lowest or the highest value in dates
-    # cause an OutOfBoundsDatetime (Overflow) error
-    with warnings.catch_warnings():
-        warnings.filterwarnings("ignore", "invalid value encountered", RuntimeWarning)
-        pd.to_timedelta(flat_num_dates.min(), delta) + ref_date
-        pd.to_timedelta(flat_num_dates.max(), delta) + ref_date
-
-    # Cast input dates to integers of nanoseconds because `pd.to_datetime`
-    # works much faster when dealing with integers
-    # make _NS_PER_TIME_DELTA an array to ensure type upcasting
-    flat_num_dates_ns_int = (
-        flat_num_dates.astype(np.float64) * _NS_PER_TIME_DELTA[delta]
-    ).astype(np.int64)
+    # To avoid integer overflow when converting to nanosecond units for integer
+    # dtypes smaller than np.int64 cast all integer-dtype arrays to np.int64
+    # (GH 2002).
+    if flat_num_dates.dtype.kind == "i":
+        flat_num_dates = flat_num_dates.astype(np.int64)
 
+    # Cast input ordinals to integers of nanoseconds because pd.to_timedelta
+    # works much faster when dealing with integers (GH 1399).
+    flat_num_dates_ns_int = (flat_num_dates * _NS_PER_TIME_DELTA[delta]).astype(
+        np.int64
+    )
+
+    # Use pd.to_timedelta to safely cast integer values to timedeltas,
+    # and add those to a Timestamp to safely produce a DatetimeIndex.  This
+    # ensures that we do not encounter integer overflow at any point in the
+    # process without raising OutOfBoundsDatetime.
     return (pd.to_timedelta(flat_num_dates_ns_int, "ns") + ref_date).values
 
 
@@ -252,11 +263,24 @@ def decode_cf_timedelta(num_timedeltas, units):
 
 
 def _infer_time_units_from_diff(unique_timedeltas):
-    for time_unit in ["days", "hours", "minutes", "seconds"]:
+    # Note that the modulus operator was only implemented for np.timedelta64
+    # arrays as of NumPy version 1.16.0.  Once our minimum version of NumPy
+    # supported is greater than or equal to this we will no longer need to cast
+    # unique_timedeltas to a TimedeltaIndex.  In the meantime, however, the
+    # modulus operator works for TimedeltaIndex objects.
+    unique_deltas_as_index = pd.TimedeltaIndex(unique_timedeltas)
+    for time_unit in [
+        "days",
+        "hours",
+        "minutes",
+        "seconds",
+        "milliseconds",
+        "microseconds",
+        "nanoseconds",
+    ]:
         delta_ns = _NS_PER_TIME_DELTA[_netcdf_to_numpy_timeunit(time_unit)]
         unit_delta = np.timedelta64(delta_ns, "ns")
-        diffs = unique_timedeltas / unit_delta
-        if np.all(diffs == diffs.astype(int)):
+        if np.all(unique_deltas_as_index % unit_delta == np.timedelta64(0, "ns")):
             return time_unit
     return "seconds"
 
@@ -416,7 +440,15 @@ def encode_cf_datetime(dates, units=None, calendar=None):
         # Wrap the dates in a DatetimeIndex to do the subtraction to ensure
         # an OverflowError is raised if the ref_date is too far away from
         # dates to be encoded (GH 2272).
-        num = (pd.DatetimeIndex(dates.ravel()) - ref_date) / time_delta
+        dates_as_index = pd.DatetimeIndex(dates.ravel())
+        time_deltas = dates_as_index - ref_date
+
+        # Use floor division if time_delta evenly divides all differences
+        # to preserve integer dtype if possible (GH 4045).
+        if np.all(time_deltas % time_delta == np.timedelta64(0, "ns")):
+            num = time_deltas // time_delta
+        else:
+            num = time_deltas / time_delta
         num = num.values.reshape(dates.shape)
 
     except (OutOfBoundsDatetime, OverflowError):

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| xarray/coding/times.py | 29 | 29 | 5 | 5 | 3073
| xarray/coding/times.py | 38 | 38 | 5 | 5 | 3073
| xarray/coding/times.py | 47 | 47 | 5 | 5 | 3073
| xarray/coding/times.py | 154 | 167 | - | 5 | -
| xarray/coding/times.py | 255 | 259 | 21 | 5 | 8903
| xarray/coding/times.py | 419 | 419 | - | 5 | -


## Problem Statement

```
Millisecond precision is lost on datetime64 during IO roundtrip
<!-- A short summary of the issue, if appropriate -->
I have millisecond-resolution time data as a coordinate on a DataArray. That data loses precision when round-tripping through disk.

#### MCVE Code Sample
<!-- In order for the maintainers to efficiently understand and prioritize issues, we ask you post a "Minimal, Complete and Verifiable Example" (MCVE): http://matthewrocklin.com/blog/work/2018/02/28/minimal-bug-reports -->

[bug_data.p.zip](https://github.com/pydata/xarray/files/4595145/bug_data.p.zip)

Unzip the data. It will result in a pickle file.

\`\`\`python
bug_data_path = '/path/to/unzipped/bug_data.p'
tmp_path = '~/Desktop/test.nc'

with open(bug_data_path, 'rb') as f:
    data = pickle.load(f)

selector = dict(animal=0, timepoint=0, wavelength='410', pair=0)

before_disk_ts = data.time.sel(**selector).values[()]

data.time.encoding = {'units': 'microseconds since 1900-01-01', 'calendar': 'proleptic_gregorian'}

data.to_netcdf(tmp_path)
after_disk_ts = xr.load_dataarray(tmp_path).time.sel(**selector).values[()]

print(f'before roundtrip: {before_disk_ts}')
print(f' after roundtrip: {after_disk_ts}')
\`\`\`
output:
\`\`\`
before roundtrip: 2017-02-22T16:24:10.586000000
after roundtrip:  2017-02-22T16:24:10.585999872
\`\`\`

#### Expected Output
\`\`\`
Before: 2017-02-22T16:24:10.586000000
After:  2017-02-22T16:24:10.586000000
\`\`\`

#### Problem Description
<!-- this should explain why the current behavior is a problem and why the expected output is a better solution -->

As you can see, I lose millisecond precision in this data. (The same happens when I use millisecond in the encoding).

#### Versions

<details><summary>Output of <tt>xr.show_versions()</tt></summary>

<!-- Paste the output here xr.show_versions() here -->
INSTALLED VERSIONS
------------------
commit: None
python: 3.7.6 | packaged by conda-forge | (default, Jan  7 2020, 22:05:27) 
[Clang 9.0.1 ]
python-bits: 64
OS: Darwin
OS-release: 19.4.0
machine: x86_64
processor: i386
byteorder: little
LC_ALL: None
LANG: en_US.UTF-8
LOCALE: None.UTF-8
libhdf5: 1.10.5
libnetcdf: 4.7.3

xarray: 0.15.1
pandas: 1.0.1
numpy: 1.18.1
scipy: 1.4.1
netCDF4: 1.5.3
pydap: None
h5netcdf: 0.8.0
h5py: 2.10.0
Nio: None
zarr: None
cftime: 1.0.4.2
nc_time_axis: None
PseudoNetCDF: None
rasterio: None
cfgrib: None
iris: None
bottleneck: None
dask: 2.11.0
distributed: 2.14.0
matplotlib: 3.1.3
cartopy: None
seaborn: 0.10.0
numbagg: None
setuptools: 45.2.0.post20200209
pip: 20.0.2
conda: None
pytest: 5.3.5
IPython: 7.12.0
sphinx: 2.4.3

</details>


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 xarray/core/dataset.py | 1 | 135| 643 | 643 | 57434 | 
| 2 | 2 asv_bench/benchmarks/dataset_io.py | 1 | 93| 715 | 1358 | 61039 | 
| 3 | 3 xarray/coding/cftime_offsets.py | 579 | 641| 832 | 2190 | 69356 | 
| 4 | 4 asv_bench/benchmarks/dataarray_missing.py | 1 | 75| 531 | 2721 | 69887 | 
| **-> 5 <-** | **5 xarray/coding/times.py** | 1 | 53| 352 | 3073 | 74163 | 
| 6 | 6 xarray/__init__.py | 1 | 93| 603 | 3676 | 74766 | 
| 7 | 7 asv_bench/benchmarks/reindexing.py | 1 | 49| 413 | 4089 | 75179 | 
| 8 | 8 xarray/core/dataarray.py | 1 | 82| 436 | 4525 | 111544 | 
| 9 | 9 xarray/plot/utils.py | 1 | 54| 283 | 4808 | 118172 | 
| 10 | 10 asv_bench/benchmarks/indexing.py | 1 | 59| 733 | 5541 | 119732 | 
| 11 | 10 asv_bench/benchmarks/dataset_io.py | 347 | 398| 442 | 5983 | 119732 | 
| 12 | 11 asv_bench/benchmarks/interp.py | 1 | 22| 184 | 6167 | 120217 | 
| 13 | 12 xarray/core/missing.py | 1 | 18| 133 | 6300 | 126250 | 
| 14 | 12 asv_bench/benchmarks/dataset_io.py | 401 | 432| 264 | 6564 | 126250 | 
| 15 | 12 asv_bench/benchmarks/dataset_io.py | 150 | 187| 349 | 6913 | 126250 | 
| 16 | 12 asv_bench/benchmarks/dataset_io.py | 96 | 126| 269 | 7182 | 126250 | 
| 17 | **12 xarray/coding/times.py** | 103 | 135| 283 | 7465 | 126250 | 
| 18 | 13 asv_bench/benchmarks/pandas.py | 1 | 25| 160 | 7625 | 126410 | 
| 19 | 13 asv_bench/benchmarks/dataset_io.py | 222 | 296| 613 | 8238 | 126410 | 
| 20 | 13 asv_bench/benchmarks/dataset_io.py | 190 | 219| 265 | 8503 | 126410 | 
| **-> 21 <-** | **13 xarray/coding/times.py** | 226 | 269| 400 | 8903 | 126410 | 
| 22 | 14 asv_bench/benchmarks/rolling.py | 1 | 17| 117 | 9020 | 127039 | 
| 23 | 14 asv_bench/benchmarks/dataset_io.py | 129 | 147| 163 | 9183 | 127039 | 
| 24 | 14 asv_bench/benchmarks/indexing.py | 145 | 162| 145 | 9328 | 127039 | 
| 25 | 14 asv_bench/benchmarks/dataset_io.py | 435 | 470| 186 | 9514 | 127039 | 
| 26 | 14 asv_bench/benchmarks/dataset_io.py | 331 | 344| 112 | 9626 | 127039 | 
| 27 | 15 xarray/core/duck_array_ops.py | 353 | 378| 270 | 9896 | 132271 | 
| 28 | 16 doc/conf.py | 1 | 105| 751 | 10647 | 135820 | 
| 29 | 16 asv_bench/benchmarks/dataset_io.py | 299 | 312| 113 | 10760 | 135820 | 
| 30 | 17 xarray/core/common.py | 1095 | 1152| 580 | 11340 | 151166 | 
| 31 | 18 doc/gallery/plot_rasterio.py | 1 | 55| 402 | 11742 | 151568 | 
| 32 | 19 xarray/core/utils.py | 644 | 677| 233 | 11975 | 157199 | 
| 33 | 20 xarray/core/alignment.py | 82 | 271| 1952 | 13927 | 163207 | 
| 34 | 20 asv_bench/benchmarks/dataset_io.py | 315 | 328| 114 | 14041 | 163207 | 
| 35 | 20 xarray/coding/cftime_offsets.py | 539 | 576| 184 | 14225 | 163207 | 
| 36 | 20 asv_bench/benchmarks/indexing.py | 62 | 76| 151 | 14376 | 163207 | 
| 37 | 21 xarray/core/merge.py | 635 | 842| 2740 | 17116 | 171074 | 
| 38 | 22 xarray/core/nputils.py | 280 | 291| 155 | 17271 | 173544 | 
| 39 | **22 xarray/coding/times.py** | 345 | 377| 253 | 17524 | 173544 | 
| 40 | 22 xarray/core/common.py | 985 | 1094| 1167 | 18691 | 173544 | 
| 41 | 22 xarray/core/common.py | 1 | 35| 183 | 18874 | 173544 | 
| 42 | 22 xarray/coding/cftime_offsets.py | 695 | 719| 197 | 19071 | 173544 | 
| 43 | 23 asv_bench/benchmarks/unstacking.py | 1 | 25| 159 | 19230 | 173703 | 
| 44 | 24 xarray/core/resample_cftime.py | 80 | 110| 261 | 19491 | 177017 | 
| 45 | 25 xarray/core/accessor_dt.py | 197 | 249| 327 | 19818 | 181624 | 
| 46 | 26 xarray/core/indexing.py | 1 | 20| 115 | 19933 | 193545 | 
| 47 | 26 xarray/core/nputils.py | 1 | 33| 244 | 20177 | 193545 | 
| 48 | 26 xarray/core/accessor_dt.py | 349 | 370| 230 | 20407 | 193545 | 
| 49 | 27 xarray/convert.py | 1 | 60| 330 | 20737 | 195881 | 


### Hint

```
This has something to do with the time values at some point being a float:

\`\`\`python
>>> import numpy as np
>>> np.datetime64("2017-02-22T16:24:10.586000000").astype("float64").astype(np.dtype('<M8[ns]'))
numpy.datetime64('2017-02-22T16:24:10.585999872')
\`\`\`

It looks like this is happening somewhere in the [cftime](https://github.com/Unidata/cftime/blob/master/cftime/_cftime.pyx#L870) library.
Thanks for the report @half-adder.

This indeed is related to times being encoded as floats, but actually is not cftime-related (the times here not being encoded using cftime; we only use cftime for non-standard calendars and out of nanosecond-resolution bounds dates).  

Here's a minimal working example that illustrates the issue with the current logic in [`coding.times.encode_cf_datetime`](https://github.com/pydata/xarray/blob/69548df9826cde9df6cbdae9c033c9fb1e62d493/xarray/coding/times.py#L343-L389):
\`\`\`
In [1]: import numpy as np; import pandas as pd

In [2]: times = pd.DatetimeIndex([np.datetime64("2017-02-22T16:27:08.732000000")])

In [3]: reference = pd.Timestamp("1900-01-01")

In [4]: units = np.timedelta64(1, "us")

In [5]: (times - reference).values[0]
Out[5]: numpy.timedelta64(3696769628732000000,'ns')

In [6]: ((times - reference) / units).values[0]
Out[6]: 3696769628732000.5
\`\`\`
In principle, we should be able to represent the difference between this date and the reference date in an integer amount of microseconds, but timedelta division produces a float.  We currently [try to cast these floats to integers when possible](https://github.com/pydata/xarray/blob/69548df9826cde9df6cbdae9c033c9fb1e62d493/xarray/coding/times.py#L388), but that's not always safe to do, e.g. in the case above.

It would be great to make roundtripping times -- particularly standard calendar datetimes like these -- more robust.  It's possible we could now leverage [floor division (i.e. `//`) of timedeltas within NumPy](https://github.com/numpy/numpy/pull/12308) for this (assuming we first check that the unit conversion divisor exactly divides each timedelta; if it doesn't we'd fall back to using floats):

\`\`\`
In [7]: ((times - reference) // units).values[0]
Out[7]: 3696769628732000
\`\`\`
These precision issues can be tricky, however, so we'd need to think things through carefully.  Even if we fixed this on the encoding side, [things are converted to floats during decoding](https://github.com/pydata/xarray/blob/69548df9826cde9df6cbdae9c033c9fb1e62d493/xarray/coding/times.py#L125-L132), so we'd need to make a change there too.
Just stumbled upon this as well. Internally, `datetime64[ns]` is simply an 8-byte int. Why on earth would it be serialized in a lossy way as a float64?...

Simply telling it to `encoding={...: {'dtype': 'int64'}}` won't work since then it complains about serializing float as an int.

Is there a way out of this, other than not using `M8[ns]` dtypes at all with xarray?

This is a huge issue, as anyone using nanosecond-precision timestamps with xarray would unknowingly and silently read wrong data after deserializing.
> Internally, datetime64[ns] is simply an 8-byte int. Why on earth would it be serialized in a lossy way as a float64?...

The short answer is that [CF conventions allow for dates to be encoded with floating point values](https://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#time-coordinate), so we encounter that in data that xarray ingests from other sources (i.e. files that were not even produced with Python, let alone xarray).  If we didn't have to worry about roundtripping files that followed those conventions, I agree we would just encode everything with nanosecond units as `int64` values.  

> This is a huge issue, as anyone using nanosecond-precision timestamps with xarray would unknowingly and silently read wrong data after deserializing.

Yes, I can see why this would be quite frustrating.  In principle we should be able to handle this (contributions are welcome); it just has not been a priority up to this point.  In my experience xarray's current encoding and decoding methods for standard calendar times work well up to at least second precision.
Can we use the `encoding["dtype"]` field to solve this? i.e. use `int64` when `encoding["dtype"]` is not set and use the specified value when available?
> In principle we should be able to handle this (contributions are welcome)

I don't mind contributing but not knowing the netcdf stuff inside out I'm not sure I have a good vision on what's the proper way to do it. My use case is very simple - I have an in-memory xr.Dataset that I want to save() and then load() without losses.

Should it just be an `xr.save(..., m8=True)` (or whatever that flag would be called), so that all of numpy's `M8[...]` and `m8[...]` would be serialized transparently (as int64, that is) without passing them through the whole cftime pipeline. It would be then nice, of course, if `xr.load` was also aware of this convention (via some special attribute or somehow else) and could convert them back like `.view('M8[ns]')` when loading. I think xarray should also throw an exception if it detects timestamps/timedeltas of nanosecond precision that it can't serialize without going through int-float-int routine (or automatically revert to using this transparent but netcdf-incompatible mode).

Maybe this is not the proper way to do it - ideas welcome (there's also an open PR - #4400 - mind checking that out?)
> Can we use the encoding["dtype"] field to solve this? i.e. use int64 when encoding["dtype"] is not set and use the specified value when available?

I think a lot of logic needs to be reshuffled, because as of right now it will complain "you can't store a float64 in int64" or something along those lines, when trying to do it with a nanosecond timestamp.
I would look here: https://github.com/pydata/xarray/blob/255bc8ee9cbe8b212e3262b0d4b2e32088a08064/xarray/coding/times.py#L440-L474
```

## Patch

```diff
diff --git a/xarray/coding/times.py b/xarray/coding/times.py
--- a/xarray/coding/times.py
+++ b/xarray/coding/times.py
@@ -26,6 +26,7 @@
 _STANDARD_CALENDARS = {"standard", "gregorian", "proleptic_gregorian"}
 
 _NS_PER_TIME_DELTA = {
+    "ns": 1,
     "us": int(1e3),
     "ms": int(1e6),
     "s": int(1e9),
@@ -35,7 +36,15 @@
 }
 
 TIME_UNITS = frozenset(
-    ["days", "hours", "minutes", "seconds", "milliseconds", "microseconds"]
+    [
+        "days",
+        "hours",
+        "minutes",
+        "seconds",
+        "milliseconds",
+        "microseconds",
+        "nanoseconds",
+    ]
 )
 
 
@@ -44,6 +53,7 @@ def _netcdf_to_numpy_timeunit(units):
     if not units.endswith("s"):
         units = "%ss" % units
     return {
+        "nanoseconds": "ns",
         "microseconds": "us",
         "milliseconds": "ms",
         "seconds": "s",
@@ -151,21 +161,22 @@ def _decode_datetime_with_pandas(flat_num_dates, units, calendar):
         # strings, in which case we fall back to using cftime
         raise OutOfBoundsDatetime
 
-    # fixes: https://github.com/pydata/pandas/issues/14068
-    # these lines check if the the lowest or the highest value in dates
-    # cause an OutOfBoundsDatetime (Overflow) error
-    with warnings.catch_warnings():
-        warnings.filterwarnings("ignore", "invalid value encountered", RuntimeWarning)
-        pd.to_timedelta(flat_num_dates.min(), delta) + ref_date
-        pd.to_timedelta(flat_num_dates.max(), delta) + ref_date
-
-    # Cast input dates to integers of nanoseconds because `pd.to_datetime`
-    # works much faster when dealing with integers
-    # make _NS_PER_TIME_DELTA an array to ensure type upcasting
-    flat_num_dates_ns_int = (
-        flat_num_dates.astype(np.float64) * _NS_PER_TIME_DELTA[delta]
-    ).astype(np.int64)
+    # To avoid integer overflow when converting to nanosecond units for integer
+    # dtypes smaller than np.int64 cast all integer-dtype arrays to np.int64
+    # (GH 2002).
+    if flat_num_dates.dtype.kind == "i":
+        flat_num_dates = flat_num_dates.astype(np.int64)
 
+    # Cast input ordinals to integers of nanoseconds because pd.to_timedelta
+    # works much faster when dealing with integers (GH 1399).
+    flat_num_dates_ns_int = (flat_num_dates * _NS_PER_TIME_DELTA[delta]).astype(
+        np.int64
+    )
+
+    # Use pd.to_timedelta to safely cast integer values to timedeltas,
+    # and add those to a Timestamp to safely produce a DatetimeIndex.  This
+    # ensures that we do not encounter integer overflow at any point in the
+    # process without raising OutOfBoundsDatetime.
     return (pd.to_timedelta(flat_num_dates_ns_int, "ns") + ref_date).values
 
 
@@ -252,11 +263,24 @@ def decode_cf_timedelta(num_timedeltas, units):
 
 
 def _infer_time_units_from_diff(unique_timedeltas):
-    for time_unit in ["days", "hours", "minutes", "seconds"]:
+    # Note that the modulus operator was only implemented for np.timedelta64
+    # arrays as of NumPy version 1.16.0.  Once our minimum version of NumPy
+    # supported is greater than or equal to this we will no longer need to cast
+    # unique_timedeltas to a TimedeltaIndex.  In the meantime, however, the
+    # modulus operator works for TimedeltaIndex objects.
+    unique_deltas_as_index = pd.TimedeltaIndex(unique_timedeltas)
+    for time_unit in [
+        "days",
+        "hours",
+        "minutes",
+        "seconds",
+        "milliseconds",
+        "microseconds",
+        "nanoseconds",
+    ]:
         delta_ns = _NS_PER_TIME_DELTA[_netcdf_to_numpy_timeunit(time_unit)]
         unit_delta = np.timedelta64(delta_ns, "ns")
-        diffs = unique_timedeltas / unit_delta
-        if np.all(diffs == diffs.astype(int)):
+        if np.all(unique_deltas_as_index % unit_delta == np.timedelta64(0, "ns")):
             return time_unit
     return "seconds"
 
@@ -416,7 +440,15 @@ def encode_cf_datetime(dates, units=None, calendar=None):
         # Wrap the dates in a DatetimeIndex to do the subtraction to ensure
         # an OverflowError is raised if the ref_date is too far away from
         # dates to be encoded (GH 2272).
-        num = (pd.DatetimeIndex(dates.ravel()) - ref_date) / time_delta
+        dates_as_index = pd.DatetimeIndex(dates.ravel())
+        time_deltas = dates_as_index - ref_date
+
+        # Use floor division if time_delta evenly divides all differences
+        # to preserve integer dtype if possible (GH 4045).
+        if np.all(time_deltas % time_delta == np.timedelta64(0, "ns")):
+            num = time_deltas // time_delta
+        else:
+            num = time_deltas / time_delta
         num = num.values.reshape(dates.shape)
 
     except (OutOfBoundsDatetime, OverflowError):

```

## Test Patch

```diff
diff --git a/xarray/tests/test_coding_times.py b/xarray/tests/test_coding_times.py
--- a/xarray/tests/test_coding_times.py
+++ b/xarray/tests/test_coding_times.py
@@ -6,7 +6,7 @@
 import pytest
 from pandas.errors import OutOfBoundsDatetime
 
-from xarray import DataArray, Dataset, Variable, coding, decode_cf
+from xarray import DataArray, Dataset, Variable, coding, conventions, decode_cf
 from xarray.coding.times import (
     cftime_to_nptime,
     decode_cf_datetime,
@@ -479,27 +479,36 @@ def test_decoded_cf_datetime_array_2d():
     assert_array_equal(np.asarray(result), expected)
 
 
+FREQUENCIES_TO_ENCODING_UNITS = {
+    "N": "nanoseconds",
+    "U": "microseconds",
+    "L": "milliseconds",
+    "S": "seconds",
+    "T": "minutes",
+    "H": "hours",
+    "D": "days",
+}
+
+
+@pytest.mark.parametrize(("freq", "units"), FREQUENCIES_TO_ENCODING_UNITS.items())
+def test_infer_datetime_units(freq, units):
+    dates = pd.date_range("2000", periods=2, freq=freq)
+    expected = f"{units} since 2000-01-01 00:00:00"
+    assert expected == coding.times.infer_datetime_units(dates)
+
+
 @pytest.mark.parametrize(
     ["dates", "expected"],
     [
-        (pd.date_range("1900-01-01", periods=5), "days since 1900-01-01 00:00:00"),
-        (
-            pd.date_range("1900-01-01 12:00:00", freq="H", periods=2),
-            "hours since 1900-01-01 12:00:00",
-        ),
         (
             pd.to_datetime(["1900-01-01", "1900-01-02", "NaT"]),
             "days since 1900-01-01 00:00:00",
         ),
-        (
-            pd.to_datetime(["1900-01-01", "1900-01-02T00:00:00.005"]),
-            "seconds since 1900-01-01 00:00:00",
-        ),
         (pd.to_datetime(["NaT", "1900-01-01"]), "days since 1900-01-01 00:00:00"),
         (pd.to_datetime(["NaT"]), "days since 1970-01-01 00:00:00"),
     ],
 )
-def test_infer_datetime_units(dates, expected):
+def test_infer_datetime_units_with_NaT(dates, expected):
     assert expected == coding.times.infer_datetime_units(dates)
 
 
@@ -535,6 +544,7 @@ def test_infer_cftime_datetime_units(calendar, date_args, expected):
         ("1h", "hours", np.int64(1)),
         ("1ms", "milliseconds", np.int64(1)),
         ("1us", "microseconds", np.int64(1)),
+        ("1ns", "nanoseconds", np.int64(1)),
         (["NaT", "0s", "1s"], None, [np.nan, 0, 1]),
         (["30m", "60m"], "hours", [0.5, 1.0]),
         ("NaT", "days", np.nan),
@@ -958,3 +968,30 @@ def test_decode_ambiguous_time_warns(calendar):
         assert not record
 
     np.testing.assert_array_equal(result, expected)
+
+
+@pytest.mark.parametrize("encoding_units", FREQUENCIES_TO_ENCODING_UNITS.values())
+@pytest.mark.parametrize("freq", FREQUENCIES_TO_ENCODING_UNITS.keys())
+def test_encode_cf_datetime_defaults_to_correct_dtype(encoding_units, freq):
+    times = pd.date_range("2000", periods=3, freq=freq)
+    units = f"{encoding_units} since 2000-01-01"
+    encoded, _, _ = coding.times.encode_cf_datetime(times, units)
+
+    numpy_timeunit = coding.times._netcdf_to_numpy_timeunit(encoding_units)
+    encoding_units_as_timedelta = np.timedelta64(1, numpy_timeunit)
+    if pd.to_timedelta(1, freq) >= encoding_units_as_timedelta:
+        assert encoded.dtype == np.int64
+    else:
+        assert encoded.dtype == np.float64
+
+
+@pytest.mark.parametrize("freq", FREQUENCIES_TO_ENCODING_UNITS.keys())
+def test_encode_decode_roundtrip(freq):
+    # See GH 4045. Prior to GH 4684 this test would fail for frequencies of
+    # "S", "L", "U", and "N".
+    initial_time = pd.date_range("1678-01-01", periods=1)
+    times = initial_time.append(pd.date_range("1968", periods=2, freq=freq))
+    variable = Variable(["time"], times)
+    encoded = conventions.encode_cf_variable(variable)
+    decoded = conventions.decode_cf_variable("time", encoded)
+    assert_equal(variable, decoded)

```


## Code snippets

### 1 - xarray/core/dataset.py:

Start line: 1, End line: 135

```python
import copy
import datetime
import functools
import sys
import warnings
from collections import defaultdict
from html import escape
from numbers import Number
from operator import methodcaller
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    DefaultDict,
    Dict,
    Hashable,
    Iterable,
    Iterator,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
    cast,
    overload,
)

import numpy as np
import pandas as pd

import xarray as xr

from ..coding.cftimeindex import _parse_array_of_cftime_strings
from ..plot.dataset_plot import _Dataset_PlotMethods
from . import (
    alignment,
    dtypes,
    duck_array_ops,
    formatting,
    formatting_html,
    groupby,
    ops,
    resample,
    rolling,
    utils,
    weighted,
)
from .alignment import _broadcast_helper, _get_broadcast_dims_map_common_coords, align
from .common import (
    DataWithCoords,
    ImplementsDatasetReduce,
    _contains_datetime_like_objects,
)
from .coordinates import (
    DatasetCoordinates,
    LevelCoordinatesSource,
    assert_coordinate_consistent,
    remap_label_indexers,
)
from .duck_array_ops import datetime_to_numeric
from .indexes import (
    Indexes,
    default_indexes,
    isel_variable_and_index,
    propagate_indexes,
    remove_unused_levels_categories,
    roll_index,
)
from .indexing import is_fancy_indexer
from .merge import (
    dataset_merge_method,
    dataset_update_method,
    merge_coordinates_without_align,
    merge_data_and_coords,
)
from .missing import get_clean_interp_index
from .options import OPTIONS, _get_keep_attrs
from .pycompat import is_duck_dask_array
from .utils import (
    Default,
    Frozen,
    SortedKeysDict,
    _default,
    decode_numpy_dict_values,
    drop_dims_from_indexers,
    either_dict_or_kwargs,
    hashable,
    infix_dims,
    is_dict_like,
    is_scalar,
    maybe_wrap_array,
)
from .variable import (
    IndexVariable,
    Variable,
    as_variable,
    assert_unique_multiindex_level_names,
    broadcast_variables,
)

if TYPE_CHECKING:
    from ..backends import AbstractDataStore, ZarrStore
    from .dataarray import DataArray
    from .merge import CoercibleMapping

    T_DSorDA = TypeVar("T_DSorDA", DataArray, "Dataset")

    try:
        from dask.delayed import Delayed
    except ImportError:
        Delayed = None


# list of attributes of pd.DatetimeIndex that are ndarrays of time info
_DATETIMEINDEX_COMPONENTS = [
    "year",
    "month",
    "day",
    "hour",
    "minute",
    "second",
    "microsecond",
    "nanosecond",
    "date",
    "time",
    "dayofyear",
    "weekofyear",
    "dayofweek",
    "quarter",
]
```
### 2 - asv_bench/benchmarks/dataset_io.py:

Start line: 1, End line: 93

```python
import os

import numpy as np
import pandas as pd

import xarray as xr

from . import randint, randn, requires_dask

try:
    import dask
    import dask.multiprocessing
except ImportError:
    pass


os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


class IOSingleNetCDF:
    """
    A few examples that benchmark reading/writing a single netCDF file with
    xarray
    """

    timeout = 300.0
    repeat = 1
    number = 5

    def make_ds(self):

        # single Dataset
        self.ds = xr.Dataset()
        self.nt = 1000
        self.nx = 90
        self.ny = 45

        self.block_chunks = {
            "time": self.nt / 4,
            "lon": self.nx / 3,
            "lat": self.ny / 3,
        }

        self.time_chunks = {"time": int(self.nt / 36)}

        times = pd.date_range("1970-01-01", periods=self.nt, freq="D")
        lons = xr.DataArray(
            np.linspace(0, 360, self.nx),
            dims=("lon",),
            attrs={"units": "degrees east", "long_name": "longitude"},
        )
        lats = xr.DataArray(
            np.linspace(-90, 90, self.ny),
            dims=("lat",),
            attrs={"units": "degrees north", "long_name": "latitude"},
        )
        self.ds["foo"] = xr.DataArray(
            randn((self.nt, self.nx, self.ny), frac_nan=0.2),
            coords={"lon": lons, "lat": lats, "time": times},
            dims=("time", "lon", "lat"),
            name="foo",
            encoding=None,
            attrs={"units": "foo units", "description": "a description"},
        )
        self.ds["bar"] = xr.DataArray(
            randn((self.nt, self.nx, self.ny), frac_nan=0.2),
            coords={"lon": lons, "lat": lats, "time": times},
            dims=("time", "lon", "lat"),
            name="bar",
            encoding=None,
            attrs={"units": "bar units", "description": "a description"},
        )
        self.ds["baz"] = xr.DataArray(
            randn((self.nx, self.ny), frac_nan=0.2).astype(np.float32),
            coords={"lon": lons, "lat": lats},
            dims=("lon", "lat"),
            name="baz",
            encoding=None,
            attrs={"units": "baz units", "description": "a description"},
        )

        self.ds.attrs = {"history": "created for xarray benchmarking"}

        self.oinds = {
            "time": randint(0, self.nt, 120),
            "lon": randint(0, self.nx, 20),
            "lat": randint(0, self.ny, 10),
        }
        self.vinds = {
            "time": xr.DataArray(randint(0, self.nt, 120), dims="x"),
            "lon": xr.DataArray(randint(0, self.nx, 120), dims="x"),
            "lat": slice(3, 20),
        }
```
### 3 - xarray/coding/cftime_offsets.py:

Start line: 579, End line: 641

```python
_FREQUENCIES = {
    "A": YearEnd,
    "AS": YearBegin,
    "Y": YearEnd,
    "YS": YearBegin,
    "Q": partial(QuarterEnd, month=12),
    "QS": partial(QuarterBegin, month=1),
    "M": MonthEnd,
    "MS": MonthBegin,
    "D": Day,
    "H": Hour,
    "T": Minute,
    "min": Minute,
    "S": Second,
    "AS-JAN": partial(YearBegin, month=1),
    "AS-FEB": partial(YearBegin, month=2),
    "AS-MAR": partial(YearBegin, month=3),
    "AS-APR": partial(YearBegin, month=4),
    "AS-MAY": partial(YearBegin, month=5),
    "AS-JUN": partial(YearBegin, month=6),
    "AS-JUL": partial(YearBegin, month=7),
    "AS-AUG": partial(YearBegin, month=8),
    "AS-SEP": partial(YearBegin, month=9),
    "AS-OCT": partial(YearBegin, month=10),
    "AS-NOV": partial(YearBegin, month=11),
    "AS-DEC": partial(YearBegin, month=12),
    "A-JAN": partial(YearEnd, month=1),
    "A-FEB": partial(YearEnd, month=2),
    "A-MAR": partial(YearEnd, month=3),
    "A-APR": partial(YearEnd, month=4),
    "A-MAY": partial(YearEnd, month=5),
    "A-JUN": partial(YearEnd, month=6),
    "A-JUL": partial(YearEnd, month=7),
    "A-AUG": partial(YearEnd, month=8),
    "A-SEP": partial(YearEnd, month=9),
    "A-OCT": partial(YearEnd, month=10),
    "A-NOV": partial(YearEnd, month=11),
    "A-DEC": partial(YearEnd, month=12),
    "QS-JAN": partial(QuarterBegin, month=1),
    "QS-FEB": partial(QuarterBegin, month=2),
    "QS-MAR": partial(QuarterBegin, month=3),
    "QS-APR": partial(QuarterBegin, month=4),
    "QS-MAY": partial(QuarterBegin, month=5),
    "QS-JUN": partial(QuarterBegin, month=6),
    "QS-JUL": partial(QuarterBegin, month=7),
    "QS-AUG": partial(QuarterBegin, month=8),
    "QS-SEP": partial(QuarterBegin, month=9),
    "QS-OCT": partial(QuarterBegin, month=10),
    "QS-NOV": partial(QuarterBegin, month=11),
    "QS-DEC": partial(QuarterBegin, month=12),
    "Q-JAN": partial(QuarterEnd, month=1),
    "Q-FEB": partial(QuarterEnd, month=2),
    "Q-MAR": partial(QuarterEnd, month=3),
    "Q-APR": partial(QuarterEnd, month=4),
    "Q-MAY": partial(QuarterEnd, month=5),
    "Q-JUN": partial(QuarterEnd, month=6),
    "Q-JUL": partial(QuarterEnd, month=7),
    "Q-AUG": partial(QuarterEnd, month=8),
    "Q-SEP": partial(QuarterEnd, month=9),
    "Q-OCT": partial(QuarterEnd, month=10),
    "Q-NOV": partial(QuarterEnd, month=11),
    "Q-DEC": partial(QuarterEnd, month=12),
}
```
### 4 - asv_bench/benchmarks/dataarray_missing.py:

Start line: 1, End line: 75

```python
import pandas as pd

import xarray as xr

from . import randn, requires_dask

try:
    import dask  # noqa: F401
except ImportError:
    pass


def make_bench_data(shape, frac_nan, chunks):
    vals = randn(shape, frac_nan)
    coords = {"time": pd.date_range("2000-01-01", freq="D", periods=shape[0])}
    da = xr.DataArray(vals, dims=("time", "x", "y"), coords=coords)

    if chunks is not None:
        da = da.chunk(chunks)

    return da


def time_interpolate_na(shape, chunks, method, limit):
    if chunks is not None:
        requires_dask()
    da = make_bench_data(shape, 0.1, chunks=chunks)
    actual = da.interpolate_na(dim="time", method="linear", limit=limit)

    if chunks is not None:
        actual = actual.compute()


time_interpolate_na.param_names = ["shape", "chunks", "method", "limit"]
time_interpolate_na.params = (
    [(3650, 200, 400), (100, 25, 25)],
    [None, {"x": 25, "y": 25}],
    ["linear", "spline", "quadratic", "cubic"],
    [None, 3],
)


def time_ffill(shape, chunks, limit):

    da = make_bench_data(shape, 0.1, chunks=chunks)
    actual = da.ffill(dim="time", limit=limit)

    if chunks is not None:
        actual = actual.compute()


time_ffill.param_names = ["shape", "chunks", "limit"]
time_ffill.params = (
    [(3650, 200, 400), (100, 25, 25)],
    [None, {"x": 25, "y": 25}],
    [None, 3],
)


def time_bfill(shape, chunks, limit):

    da = make_bench_data(shape, 0.1, chunks=chunks)
    actual = da.bfill(dim="time", limit=limit)

    if chunks is not None:
        actual = actual.compute()


time_bfill.param_names = ["shape", "chunks", "limit"]
time_bfill.params = (
    [(3650, 200, 400), (100, 25, 25)],
    [None, {"x": 25, "y": 25}],
    [None, 3],
)
```
### 5 - xarray/coding/times.py:

Start line: 1, End line: 53

```python
import re
import warnings
from datetime import datetime
from distutils.version import LooseVersion
from functools import partial

import numpy as np
import pandas as pd
from pandas.errors import OutOfBoundsDatetime

from ..core import indexing
from ..core.common import contains_cftime_datetimes
from ..core.formatting import first_n_items, format_timestamp, last_item
from ..core.variable import Variable
from .variables import (
    SerializationWarning,
    VariableCoder,
    lazy_elemwise_func,
    pop_to,
    safe_setitem,
    unpack_for_decoding,
    unpack_for_encoding,
)

# standard calendars recognized by cftime
_STANDARD_CALENDARS = {"standard", "gregorian", "proleptic_gregorian"}

_NS_PER_TIME_DELTA = {
    "us": int(1e3),
    "ms": int(1e6),
    "s": int(1e9),
    "m": int(1e9) * 60,
    "h": int(1e9) * 60 * 60,
    "D": int(1e9) * 60 * 60 * 24,
}

TIME_UNITS = frozenset(
    ["days", "hours", "minutes", "seconds", "milliseconds", "microseconds"]
)


def _netcdf_to_numpy_timeunit(units):
    units = units.lower()
    if not units.endswith("s"):
        units = "%ss" % units
    return {
        "microseconds": "us",
        "milliseconds": "ms",
        "seconds": "s",
        "minutes": "m",
        "hours": "h",
        "days": "D",
    }[units]
```
### 6 - xarray/__init__.py:

Start line: 1, End line: 93

```python
import pkg_resources

from . import testing, tutorial, ufuncs
from .backends.api import (
    load_dataarray,
    load_dataset,
    open_dataarray,
    open_dataset,
    open_mfdataset,
    save_mfdataset,
)
from .backends.rasterio_ import open_rasterio
from .backends.zarr import open_zarr
from .coding.cftime_offsets import cftime_range
from .coding.cftimeindex import CFTimeIndex
from .coding.frequencies import infer_freq
from .conventions import SerializationWarning, decode_cf
from .core.alignment import align, broadcast
from .core.combine import combine_by_coords, combine_nested
from .core.common import ALL_DIMS, full_like, ones_like, zeros_like
from .core.computation import apply_ufunc, corr, cov, dot, polyval, where
from .core.concat import concat
from .core.dataarray import DataArray
from .core.dataset import Dataset
from .core.extensions import register_dataarray_accessor, register_dataset_accessor
from .core.merge import MergeError, merge
from .core.options import set_options
from .core.parallel import map_blocks
from .core.variable import Coordinate, IndexVariable, Variable, as_variable
from .util.print_versions import show_versions

try:
    __version__ = pkg_resources.get_distribution("xarray").version
except Exception:
    # Local copy or not installed with setuptools.
    # Disable minimum version checks on downstream libraries.
    __version__ = "999"

# A hardcoded __all__ variable is necessary to appease
# `mypy --strict` running in projects that import xarray.
__all__ = (
    # Sub-packages
    "ufuncs",
    "testing",
    "tutorial",
    # Top-level functions
    "align",
    "apply_ufunc",
    "as_variable",
    "broadcast",
    "cftime_range",
    "combine_by_coords",
    "combine_nested",
    "concat",
    "decode_cf",
    "dot",
    "cov",
    "corr",
    "full_like",
    "infer_freq",
    "load_dataarray",
    "load_dataset",
    "map_blocks",
    "merge",
    "ones_like",
    "open_dataarray",
    "open_dataset",
    "open_mfdataset",
    "open_rasterio",
    "open_zarr",
    "polyval",
    "register_dataarray_accessor",
    "register_dataset_accessor",
    "save_mfdataset",
    "set_options",
    "show_versions",
    "where",
    "zeros_like",
    # Classes
    "CFTimeIndex",
    "Coordinate",
    "DataArray",
    "Dataset",
    "IndexVariable",
    "Variable",
    # Exceptions
    "MergeError",
    "SerializationWarning",
    # Constants
    "__version__",
    "ALL_DIMS",
)
```
### 7 - asv_bench/benchmarks/reindexing.py:

Start line: 1, End line: 49

```python
import numpy as np

import xarray as xr

from . import requires_dask


class Reindex:
    def setup(self):
        data = np.random.RandomState(0).randn(1000, 100, 100)
        self.ds = xr.Dataset(
            {"temperature": (("time", "x", "y"), data)},
            coords={"time": np.arange(1000), "x": np.arange(100), "y": np.arange(100)},
        )

    def time_1d_coarse(self):
        self.ds.reindex(time=np.arange(0, 1000, 5)).load()

    def time_1d_fine_all_found(self):
        self.ds.reindex(time=np.arange(0, 1000, 0.5), method="nearest").load()

    def time_1d_fine_some_missing(self):
        self.ds.reindex(
            time=np.arange(0, 1000, 0.5), method="nearest", tolerance=0.1
        ).load()

    def time_2d_coarse(self):
        self.ds.reindex(x=np.arange(0, 100, 2), y=np.arange(0, 100, 2)).load()

    def time_2d_fine_all_found(self):
        self.ds.reindex(
            x=np.arange(0, 100, 0.5), y=np.arange(0, 100, 0.5), method="nearest"
        ).load()

    def time_2d_fine_some_missing(self):
        self.ds.reindex(
            x=np.arange(0, 100, 0.5),
            y=np.arange(0, 100, 0.5),
            method="nearest",
            tolerance=0.1,
        ).load()


class ReindexDask(Reindex):
    def setup(self):
        requires_dask()
        super().setup()
        self.ds = self.ds.chunk({"time": 100})
```
### 8 - xarray/core/dataarray.py:

Start line: 1, End line: 82

```python
import datetime
import functools
import warnings
from numbers import Number
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import numpy as np
import pandas as pd

from ..plot.plot import _PlotMethods
from . import (
    computation,
    dtypes,
    groupby,
    indexing,
    ops,
    pdcompat,
    resample,
    rolling,
    utils,
    weighted,
)
from .accessor_dt import CombinedDatetimelikeAccessor
from .accessor_str import StringAccessor
from .alignment import (
    _broadcast_helper,
    _get_broadcast_dims_map_common_coords,
    align,
    reindex_like_indexers,
)
from .common import AbstractArray, DataWithCoords
from .coordinates import (
    DataArrayCoordinates,
    LevelCoordinatesSource,
    assert_coordinate_consistent,
    remap_label_indexers,
)
from .dataset import Dataset, split_indexes
from .formatting import format_item
from .indexes import Indexes, default_indexes, propagate_indexes
from .indexing import is_fancy_indexer
from .merge import PANDAS_TYPES, MergeError, _extract_indexes_from_coords
from .options import OPTIONS, _get_keep_attrs
from .utils import Default, ReprObject, _default, either_dict_or_kwargs
from .variable import (
    IndexVariable,
    Variable,
    as_compatible_data,
    as_variable,
    assert_unique_multiindex_level_names,
)

if TYPE_CHECKING:
    T_DSorDA = TypeVar("T_DSorDA", "DataArray", Dataset)

    try:
        from dask.delayed import Delayed
    except ImportError:
        Delayed = None
    try:
        from cdms2 import Variable as cdms2_Variable
    except ImportError:
        cdms2_Variable = None
    try:
        from iris.cube import Cube as iris_Cube
    except ImportError:
        iris_Cube = None
```
### 9 - xarray/plot/utils.py:

Start line: 1, End line: 54

```python
import itertools
import textwrap
import warnings
from datetime import datetime
from inspect import getfullargspec
from typing import Any, Iterable, Mapping, Tuple, Union

import numpy as np
import pandas as pd

from ..core.options import OPTIONS
from ..core.utils import is_scalar

try:
    import nc_time_axis  # noqa: F401

    nc_time_axis_available = True
except ImportError:
    nc_time_axis_available = False

ROBUST_PERCENTILE = 2.0


_registered = False


def register_pandas_datetime_converter_if_needed():
    # based on https://github.com/pandas-dev/pandas/pull/17710
    global _registered
    if not _registered:
        pd.plotting.register_matplotlib_converters()
        _registered = True


def import_matplotlib_pyplot():
    """Import pyplot as register appropriate converters."""
    register_pandas_datetime_converter_if_needed()
    import matplotlib.pyplot as plt

    return plt


def _determine_extend(calc_data, vmin, vmax):
    extend_min = calc_data.min() < vmin
    extend_max = calc_data.max() > vmax
    if extend_min and extend_max:
        extend = "both"
    elif extend_min:
        extend = "min"
    elif extend_max:
        extend = "max"
    else:
        extend = "neither"
    return extend
```
### 10 - asv_bench/benchmarks/indexing.py:

Start line: 1, End line: 59

```python
import os

import numpy as np
import pandas as pd

import xarray as xr

from . import randint, randn, requires_dask

nx = 3000
ny = 2000
nt = 1000

basic_indexes = {
    "1slice": {"x": slice(0, 3)},
    "1slice-1scalar": {"x": 0, "y": slice(None, None, 3)},
    "2slicess-1scalar": {"x": slice(3, -3, 3), "y": 1, "t": slice(None, -3, 3)},
}

basic_assignment_values = {
    "1slice": xr.DataArray(randn((3, ny), frac_nan=0.1), dims=["x", "y"]),
    "1slice-1scalar": xr.DataArray(randn(int(ny / 3) + 1, frac_nan=0.1), dims=["y"]),
    "2slicess-1scalar": xr.DataArray(
        randn(int((nx - 6) / 3), frac_nan=0.1), dims=["x"]
    ),
}

outer_indexes = {
    "1d": {"x": randint(0, nx, 400)},
    "2d": {"x": randint(0, nx, 500), "y": randint(0, ny, 400)},
    "2d-1scalar": {"x": randint(0, nx, 100), "y": 1, "t": randint(0, nt, 400)},
}

outer_assignment_values = {
    "1d": xr.DataArray(randn((400, ny), frac_nan=0.1), dims=["x", "y"]),
    "2d": xr.DataArray(randn((500, 400), frac_nan=0.1), dims=["x", "y"]),
    "2d-1scalar": xr.DataArray(randn(100, frac_nan=0.1), dims=["x"]),
}

vectorized_indexes = {
    "1-1d": {"x": xr.DataArray(randint(0, nx, 400), dims="a")},
    "2-1d": {
        "x": xr.DataArray(randint(0, nx, 400), dims="a"),
        "y": xr.DataArray(randint(0, ny, 400), dims="a"),
    },
    "3-2d": {
        "x": xr.DataArray(randint(0, nx, 400).reshape(4, 100), dims=["a", "b"]),
        "y": xr.DataArray(randint(0, ny, 400).reshape(4, 100), dims=["a", "b"]),
        "t": xr.DataArray(randint(0, nt, 400).reshape(4, 100), dims=["a", "b"]),
    },
}

vectorized_assignment_values = {
    "1-1d": xr.DataArray(randn((400, 2000)), dims=["a", "y"], coords={"a": randn(400)}),
    "2-1d": xr.DataArray(randn(400), dims=["a"], coords={"a": randn(400)}),
    "3-2d": xr.DataArray(
        randn((4, 100)), dims=["a", "b"], coords={"a": randn(4), "b": randn(100)}
    ),
}
```
### 17 - xarray/coding/times.py:

Start line: 103, End line: 135

```python
def _decode_cf_datetime_dtype(data, units, calendar, use_cftime):
    # Verify that at least the first and last date can be decoded
    # successfully. Otherwise, tracebacks end up swallowed by
    # Dataset.__repr__ when users try to view their lazily decoded array.
    values = indexing.ImplicitToExplicitIndexingAdapter(indexing.as_indexable(data))
    example_value = np.concatenate(
        [first_n_items(values, 1) or [0], last_item(values) or [0]]
    )

    try:
        result = decode_cf_datetime(example_value, units, calendar, use_cftime)
    except Exception:
        calendar_msg = (
            "the default calendar" if calendar is None else "calendar %r" % calendar
        )
        msg = (
            f"unable to decode time units {units!r} with {calendar_msg!r}. Try "
            "opening your dataset with decode_times=False or installing cftime "
            "if it is not installed."
        )
        raise ValueError(msg)
    else:
        dtype = getattr(result, "dtype", np.dtype("object"))

    return dtype


def _decode_datetime_with_cftime(num_dates, units, calendar):
    import cftime

    return np.asarray(
        cftime.num2date(num_dates, units, calendar, only_use_cftime_datetimes=True)
    )
```
### 21 - xarray/coding/times.py:

Start line: 226, End line: 269

```python
def to_timedelta_unboxed(value, **kwargs):
    if LooseVersion(pd.__version__) < "0.25.0":
        result = pd.to_timedelta(value, **kwargs, box=False)
    else:
        result = pd.to_timedelta(value, **kwargs).to_numpy()
    assert result.dtype == "timedelta64[ns]"
    return result


def to_datetime_unboxed(value, **kwargs):
    if LooseVersion(pd.__version__) < "0.25.0":
        result = pd.to_datetime(value, **kwargs, box=False)
    else:
        result = pd.to_datetime(value, **kwargs).to_numpy()
    assert result.dtype == "datetime64[ns]"
    return result


def decode_cf_timedelta(num_timedeltas, units):
    """Given an array of numeric timedeltas in netCDF format, convert it into a
    numpy timedelta64[ns] array.
    """
    num_timedeltas = np.asarray(num_timedeltas)
    units = _netcdf_to_numpy_timeunit(units)
    result = to_timedelta_unboxed(num_timedeltas.ravel(), unit=units)
    return result.reshape(num_timedeltas.shape)


def _infer_time_units_from_diff(unique_timedeltas):
    for time_unit in ["days", "hours", "minutes", "seconds"]:
        delta_ns = _NS_PER_TIME_DELTA[_netcdf_to_numpy_timeunit(time_unit)]
        unit_delta = np.timedelta64(delta_ns, "ns")
        diffs = unique_timedeltas / unit_delta
        if np.all(diffs == diffs.astype(int)):
            return time_unit
    return "seconds"


def infer_calendar_name(dates):
    """Given an array of datetimes, infer the CF calendar name"""
    if np.asarray(dates).dtype == "datetime64[ns]":
        return "proleptic_gregorian"
    else:
        return np.asarray(dates).ravel()[0].calendar
```
### 39 - xarray/coding/times.py:

Start line: 345, End line: 377

```python
def _cleanup_netcdf_time_units(units):
    delta, ref_date = _unpack_netcdf_time_units(units)
    try:
        units = "{} since {}".format(delta, format_timestamp(ref_date))
    except OutOfBoundsDatetime:
        # don't worry about reifying the units if they're out of bounds
        pass
    return units


def _encode_datetime_with_cftime(dates, units, calendar):
    """Fallback method for encoding dates using cftime.

    This method is more flexible than xarray's parsing using datetime64[ns]
    arrays but also slower because it loops over each element.
    """
    import cftime

    if np.issubdtype(dates.dtype, np.datetime64):
        # numpy's broken datetime conversion only works for us precision
        dates = dates.astype("M8[us]").astype(datetime)

    def encode_datetime(d):
        return np.nan if d is None else cftime.date2num(d, units, calendar)

    return np.vectorize(encode_datetime)(dates)


def cast_to_int_if_safe(num):
    int_num = np.array(num, dtype=np.int64)
    if (num == int_num).all():
        num = int_num
    return num
```
