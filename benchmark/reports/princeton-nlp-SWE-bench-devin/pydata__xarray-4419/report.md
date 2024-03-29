# pydata__xarray-4419

| **pydata/xarray** | `2ed6d57fa5e14e87e83c8194e619538f6edcd90a` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 33635 |
| **Any found context length** | 33635 |
| **Avg pos** | 20.0 |
| **Min pos** | 20 |
| **Max pos** | 20 |
| **Top file pos** | 6 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/xarray/core/concat.py b/xarray/core/concat.py
--- a/xarray/core/concat.py
+++ b/xarray/core/concat.py
@@ -463,6 +463,9 @@ def ensure_common_dims(vars):
             combined = concat_vars(vars, dim, positions)
             assert isinstance(combined, Variable)
             result_vars[k] = combined
+        elif k in result_vars:
+            # preserves original variable order
+            result_vars[k] = result_vars.pop(k)
 
     result = Dataset(result_vars, attrs=result_attrs)
     absent_coord_names = coord_names - set(result.variables)

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| xarray/core/concat.py | 466 | 466 | 20 | 6 | 33635


## Problem Statement

```
concat changes variable order
#### Code Sample, a copy-pastable example if possible

A "Minimal, Complete and Verifiable Example" will make it much easier for maintainers to help you:
http://matthewrocklin.com/blog/work/2018/02/28/minimal-bug-reports
 
- Case 1: Creation of Dataset without Coordinates
\`\`\`python
data = np.zeros((2,3))
ds = xr.Dataset({'test': (['c', 'b'],  data)})
print(ds.dims)
ds2 = xr.concat([ds, ds], dim='c')
print(ds2.dims)
\`\`\`
yields (assumed correct) output of:
\`\`\`
Frozen(SortedKeysDict({'c': 2, 'b': 3}))
Frozen(SortedKeysDict({'c': 4, 'b': 3}))
\`\`\`
- Case 2: Creation of Dataset with Coordinates
\`\`\`python
data = np.zeros((2,3))
ds = xr.Dataset({'test': (['c', 'b'],  data)}, 
                coords={'c': (['c'], np.arange(data.shape[0])),
                        'b': (['b'], np.arange(data.shape[1])),})
print(ds.dims)
ds2 = xr.concat([ds, ds], dim='c')
print(ds2.dims)
\`\`\`
yields (assumed false) output of:
\`\`\`
Frozen(SortedKeysDict({'c': 2, 'b': 3}))
Frozen(SortedKeysDict({'b': 3, 'c': 4}))
\`\`\`

#### Problem description

`xr.concat` changes the dimension order for `.dims` as well as `.sizes` to  an alphanumerically sorted representation.

#### Expected Output

`xr.concat` should not change the dimension order in any case.

\`\`\`
Frozen(SortedKeysDict({'c': 2, 'b': 3}))
Frozen(SortedKeysDict({'c': 4, 'b': 3}))
\`\`\`

#### Output of ``xr.show_versions()``

<details>
INSTALLED VERSIONS
------------------
commit: None
python: 3.7.1 | packaged by conda-forge | (default, Nov 13 2018, 18:33:04) 
[GCC 7.3.0]
python-bits: 64
OS: Linux
OS-release: 4.12.14-lp150.12.48-default
machine: x86_64
processor: x86_64
byteorder: little
LC_ALL: None
LANG: de_DE.UTF-8
LOCALE: de_DE.UTF-8
libhdf5: 1.10.4
libnetcdf: 4.6.2

xarray: 0.11.3
pandas: 0.24.1
numpy: 1.16.1
scipy: 1.2.0
netCDF4: 1.4.2
pydap: None
h5netcdf: 0.6.2
h5py: 2.9.0
Nio: None
zarr: None
cftime: 1.0.3.4
PseudonetCDF: None
rasterio: None
cfgrib: None
iris: None
bottleneck: 1.2.1
cyordereddict: None
dask: None
distributed: None
matplotlib: 3.0.2
cartopy: 0.17.0
seaborn: None
setuptools: 40.8.0
pip: 19.0.2
conda: None
pytest: 4.2.0
IPython: 7.2.0
sphinx: None

</details>

[BUG] xr.concat inverts coordinates order
<!-- A short summary of the issue, if appropriate -->

Following the issue #3969 
Merging two datasets using xr.concat inverts the coordinates order.

#### MCVE Code Sample

\`\`\`
import numpy as np
import xarray as xr

x = np.arange(0,10)
y = np.arange(0,10)
time = [0,1]
data = np.zeros((10,10), dtype=bool)
dataArray1 = xr.DataArray([data], coords={'time': [time[0]], 'y': y, 'x': x},
                             dims=['time', 'y', 'x'])
dataArray2 = xr.DataArray([data], coords={'time': [time[1]], 'y': y, 'x': x},
                             dims=['time', 'y', 'x'])
dataArray1 = dataArray1.to_dataset(name='data')
dataArray2 = dataArray2.to_dataset(name='data')

print(dataArray1)
print(xr.concat([dataArray1,dataArray2], dim='time'))
\`\`\`
#### Current Output

\`\`\`
<xarray.Dataset>
Dimensions:  (time: 1, x: 10, y: 10)
Coordinates:
  * time     (time) int64 0
  * y        (y) int64 0 1 2 3 4 5 6 7 8 9
  * x        (x) int64 0 1 2 3 4 5 6 7 8 9
Data variables:
    data     (time, y, x) bool False False False False ... False False False
<xarray.Dataset>
Dimensions:  (time: 2, x: 10, y: 10)
Coordinates:
  * x        (x) int64 0 1 2 3 4 5 6 7 8 9  ##Inverted x and y
  * y        (y) int64 0 1 2 3 4 5 6 7 8 9
  * time     (time) int64 0 1
Data variables:
    data     (time, y, x) bool False False False False ... False False False

\`\`\`

#### Expected Output
\`\`\`
<xarray.Dataset>
Dimensions:  (time: 1, x: 10, y: 10)
Coordinates:
  * time     (time) int64 0
  * y        (y) int64 0 1 2 3 4 5 6 7 8 9
  * x        (x) int64 0 1 2 3 4 5 6 7 8 9
Data variables:
    data     (time, y, x) bool False False False False ... False False False
<xarray.Dataset>
Dimensions:  (time: 2, x: 10, y: 10)
Coordinates:
  * y        (y) int64 0 1 2 3 4 5 6 7 8 9
  * x        (x) int64 0 1 2 3 4 5 6 7 8 9
  * time     (time) int64 0 1
Data variables:
    data     (time, y, x) bool False False False False ... False False False

\`\`\`


#### Problem Description

The concat function should not invert the coordinates but maintain the original order.

#### Versions

<details><summary>INSTALLED VERSIONS
</summary>
------------------
commit: None
python: 3.6.8 (default, May  7 2019, 14:58:50)
[GCC 8.3.0]
python-bits: 64
OS: Linux
OS-release: 4.15.0-88-generic
machine: x86_64
processor: x86_64
byteorder: little
LC_ALL: None
LANG: C.UTF-8
LOCALE: en_US.UTF-8
libhdf5: None
libnetcdf: None

xarray: 0.15.1
pandas: 1.0.3
numpy: 1.18.2
scipy: None
netCDF4: None
pydap: None
h5netcdf: None
h5py: None
Nio: None
zarr: None
cftime: None
nc_time_axis: None
PseudoNetCDF: None
rasterio: None
cfgrib: None
iris: None
bottleneck: None
dask: None
distributed: None
matplotlib: 3.2.0
cartopy: None
seaborn: None
numbagg: None
setuptools: 46.1.3
pip: 9.0.1
conda: None
pytest: None
IPython: 7.13.0
sphinx: 2.4.3</details>


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 xarray/core/merge.py | 635 | 842| 2740 | 2740 | 7866 | 
| 2 | 2 xarray/core/combine.py | 551 | 749| 2476 | 5216 | 15101 | 
| 3 | 3 xarray/core/alignment.py | 69 | 258| 1938 | 7154 | 21046 | 
| 4 | 4 xarray/core/dataset.py | 1 | 136| 649 | 7803 | 75628 | 
| 5 | 4 xarray/core/combine.py | 47 | 113| 528 | 8331 | 75628 | 
| 6 | 4 xarray/core/dataset.py | 4418 | 5177| 5915 | 14246 | 75628 | 
| 7 | 4 xarray/core/combine.py | 345 | 799| 2164 | 16410 | 75628 | 
| 8 | 5 asv_bench/benchmarks/indexing.py | 1 | 57| 730 | 17140 | 77040 | 
| 9 | **6 xarray/core/concat.py** | 165 | 193| 274 | 17414 | 81306 | 
| 10 | **6 xarray/core/concat.py** | 336 | 361| 239 | 17653 | 81306 | 
| 11 | 7 xarray/core/dataarray.py | 1 | 82| 436 | 18089 | 115925 | 
| 12 | 8 asv_bench/benchmarks/combine.py | 1 | 39| 332 | 18421 | 116257 | 
| 13 | **8 xarray/core/concat.py** | 243 | 333| 752 | 19173 | 116257 | 
| 14 | **8 xarray/core/concat.py** | 364 | 441| 667 | 19840 | 116257 | 
| 15 | 8 xarray/core/dataset.py | 3041 | 3796| 6149 | 25989 | 116257 | 
| 16 | 9 xarray/__init__.py | 1 | 93| 603 | 26592 | 116860 | 
| 17 | **9 xarray/core/concat.py** | 442 | 453| 239 | 26831 | 116860 | 
| 18 | 9 xarray/core/dataset.py | 2343 | 3039| 6072 | 32903 | 116860 | 
| 19 | 9 xarray/core/combine.py | 751 | 800| 409 | 33312 | 116860 | 
| **-> 20 <-** | **9 xarray/core/concat.py** | 455 | 483| 323 | 33635 | 116860 | 
| 21 | **9 xarray/core/concat.py** | 28 | 40| 108 | 33743 | 116860 | 
| 22 | 9 xarray/core/dataset.py | 555 | 1413| 6489 | 40232 | 116860 | 
| 23 | 10 asv_bench/benchmarks/reindexing.py | 1 | 49| 413 | 40645 | 117273 | 
| 24 | 11 asv_bench/benchmarks/dataarray_missing.py | 1 | 75| 531 | 41176 | 117804 | 
| 25 | 11 xarray/core/dataarray.py | 1579 | 1630| 381 | 41557 | 117804 | 
| 26 | **11 xarray/core/concat.py** | 1 | 25| 122 | 41679 | 117804 | 
| 27 | 12 xarray/core/coordinates.py | 222 | 245| 198 | 41877 | 120792 | 
| 28 | 12 xarray/core/coordinates.py | 1 | 30| 154 | 42031 | 120792 | 
| 29 | 13 xarray/core/common.py | 1100 | 1157| 580 | 42611 | 135463 | 
| 30 | **13 xarray/core/concat.py** | 486 | 533| 287 | 42898 | 135463 | 
| 31 | 13 xarray/core/dataarray.py | 1778 | 1813| 293 | 43191 | 135463 | 
| 32 | **13 xarray/core/concat.py** | 43 | 55| 110 | 43301 | 135463 | 
| 33 | 13 xarray/core/dataset.py | 3798 | 4416| 5367 | 48668 | 135463 | 
| 34 | 13 xarray/core/dataset.py | 5315 | 5925| 5688 | 54356 | 135463 | 
| 35 | **13 xarray/core/concat.py** | 220 | 241| 166 | 54522 | 135463 | 
| 36 | 13 xarray/core/dataarray.py | 1874 | 1936| 508 | 55030 | 135463 | 
| 37 | 14 asv_bench/benchmarks/unstacking.py | 1 | 25| 159 | 55189 | 135622 | 
| 38 | 14 xarray/core/dataset.py | 5179 | 5313| 1070 | 56259 | 135622 | 
| 39 | 14 xarray/core/dataset.py | 1561 | 2271| 6041 | 62300 | 135622 | 
| 40 | 14 xarray/core/alignment.py | 537 | 611| 697 | 62997 | 135622 | 
| 41 | 14 xarray/core/combine.py | 238 | 285| 264 | 63261 | 135622 | 
| 42 | 14 xarray/core/coordinates.py | 285 | 303| 174 | 63435 | 135622 | 
| 43 | 14 xarray/core/dataarray.py | 3546 | 3698| 1829 | 65264 | 135622 | 
| 44 | 15 doc/gallery/plot_rasterio.py | 1 | 55| 402 | 65666 | 136024 | 
| 45 | 15 xarray/core/dataarray.py | 3044 | 3100| 584 | 66250 | 136024 | 
| 46 | **15 xarray/core/concat.py** | 58 | 164| 1090 | 67340 | 136024 | 
| 47 | 16 xarray/core/utils.py | 1 | 84| 467 | 67807 | 141546 | 
| 48 | 16 xarray/core/coordinates.py | 33 | 77| 289 | 68096 | 141546 | 
| 49 | 16 xarray/core/combine.py | 210 | 235| 201 | 68297 | 141546 | 
| 50 | 17 asv_bench/benchmarks/dataset_io.py | 1 | 93| 715 | 69012 | 145151 | 
| 51 | **17 xarray/core/concat.py** | 196 | 217| 163 | 69175 | 145151 | 
| 52 | 17 xarray/core/merge.py | 1 | 77| 496 | 69671 | 145151 | 
| 53 | 18 xarray/core/groupby.py | 1 | 36| 229 | 69900 | 153020 | 
| 54 | 18 xarray/core/dataarray.py | 3237 | 3294| 567 | 70467 | 153020 | 
| 55 | 19 xarray/convert.py | 1 | 60| 330 | 70797 | 155356 | 
| 56 | 20 xarray/plot/plot.py | 636 | 793| 1739 | 72536 | 163693 | 
| 57 | 20 xarray/core/dataarray.py | 362 | 403| 384 | 72920 | 163693 | 
| 58 | 20 xarray/core/common.py | 1 | 35| 183 | 73103 | 163693 | 
| 59 | 20 xarray/core/combine.py | 148 | 207| 438 | 73541 | 163693 | 
| 60 | 20 xarray/core/common.py | 990 | 1099| 1167 | 74708 | 163693 | 
| 61 | 20 xarray/core/dataset.py | 214 | 296| 681 | 75389 | 163693 | 
| 62 | 21 doc/conf.py | 1 | 104| 738 | 76127 | 167076 | 
| 63 | 22 xarray/core/parallel.py | 1 | 47| 225 | 76352 | 172095 | 
| 64 | 23 xarray/core/rolling.py | 559 | 619| 502 | 76854 | 178227 | 
| 65 | 23 xarray/core/coordinates.py | 146 | 181| 262 | 77116 | 178227 | 
| 66 | 24 asv_bench/benchmarks/interp.py | 1 | 22| 184 | 77300 | 178712 | 


### Hint

```
Xref: [Gitter Chat](https://gitter.im/pydata/xarray?at=5c88ef25d3b35423cbb02afc)
This has also implications for the output using `.to_netcdf()`. If we read a netcdf dataset (same structure as above) with `xr.open_dataset` and then do the above `xr.concat` and save the resulting dataset with `.to_netcdf` then the dimensions of the dataset will be reversed in the resulting file.

Now, as the `xr.concat` operation need to change the length of the dimension ('c', which is not allowed by netCDF library), this is done by creating a new dataset. In this creation process xarray obviously uses the alphanumerically sorted representation of the source dataset dimension's and not the creation order as in the source dataset. 

I did not find any hints in the docs on that topic.  I need to preserve the original dimension ordering as declared in the source dataset. How can I achieve this using xarray?
Your system might print dataset dimensions like `Frozen(SortedKeysDict({'c': 2, 'b': 3}))`, but the iteration order will always be sorted (including if you write the dataset to disk as netcdf file).

When we drop support for Python 3.5, xarray might switch to dimensions matching order of insertion, since we'll get that for free with Python dictionary. But I still doubt we would make any guarantees about preserving dimension order in xarray operations, just like we don't guarantee variable order as part of xarray's API. It should be deterministic (with fixed versions of xarray and dependencies), but you shouldn't write your code in a way that breaks if changes.

What's your actual use-case here? What are you trying to do that needs preserving of dimension order?
Thanks for looking into this @shoyer.

> Your system might print dataset dimensions like `Frozen(SortedKeysDict({'c': 2, 'b': 3}))`, but the iteration order will always be sorted (including if you write the dataset to disk as netcdf file).

This isn't true for my system. If we consider this example:

\`\`\`python
data = np.zeros((2,3))
ds = xr.Dataset({'test': (['c', 'b'],  data)}, 
                coords={'c': (['c'], np.arange(data.shape[0])),
                        'b': (['b'], np.arange(data.shape[1])),})
ds.to_netcdf('test_dims.nc')
ds2 = xr.concat([ds, ds], dim='c')
ds2.to_netcdf('test_dims2.nc')
\`\`\`
Dumping the created files gives the following:

\`\`\`
netcdf test_dims {
dimensions:
        c = 2 ;
        b = 3 ;
variables:
        double test(c, b) ;
                test:_FillValue = NaN ;
        int64 c(c) ;
        int64 b(b) ;
data:

 test =
  0, 0, 0,
  0, 0, 0 ;

 c = 0, 1 ;

 b = 0, 1, 2 ;
}
\`\`\`
\`\`\`
netcdf test_dims2 {
dimensions:
        b = 3 ;
        c = 4 ;
variables:
        int64 b(b) ;
        double test(c, b) ;
                test:_FillValue = NaN ;
        int64 c(c) ;
data:

 b = 0, 1, 2 ;

 test =
  0, 0, 0,
  0, 0, 0,
  0, 0, 0,
  0, 0, 0 ;

 c = 0, 1, 0, 1 ;
}
\`\`\`
My use case is, well, I have to use some legacy code.

Concerning my code, yes I'm trying to write it as robust as possible. Finally I wan't to replace the legacy code with the implementation relying completely on xarray, but that's a long way to go.



Dimensions are written to netCDF files in the order in which they appear on variables in the Dataset:
https://github.com/pydata/xarray/blob/f382fd840dafa5fdd95e66a7ddd15a3d498c1bce/xarray/backends/common.py#L325-L329

It sounds like your use-case is writing netCDF files to disk with a desired dimension order? We could conceivably add an "encoding" option to datasets for specifying dimension order, like how we support controlling unlimited dimensions.
> Dimensions are written to netCDF files in the order in which they appear on variables in the Dataset:

I was assuming something along that lines. But in my variable `test` the 'c' dim is first. And it is written correctly, if there are no coordinates in that dataset (test_dim0). If there are coordinates (test_dims2) the dimensions are written in wrong order. So there is something working in one config and not in the other.

My use case is, that the dimensions should appear in the same order as in the source files.
> I was assuming something along that lines. But in my variable `test` the 'c' dim is first. And it is written correctly, if there are no coordinates in that dataset (test_dim0). If there are coordinates (test_dims2) the dimensions are written in wrong order. So there is something working in one config and not in the other.

The order of dimensions in the netCDF file matches the order of their appearance on variables in the netCDF files. In your first file, it's `(c, b)` on variable `test`. In the second file, it's `b` on variable `b` and `c` from variable `test`.

> My use case is, that the dimensions should appear in the same order as in the source files.

Sorry, xarray is not going to satisfy this use. If you want this guarantee in all cases, you should pick a different tool.
@shoyer I'm sorry if I did not explain well enough and if my intentions were vague. So let me first clarify, I really appreciate all your hard work to make xarray better. I've adapted many of my workflows to use xarray and I'm happy that such a library exist.

Let's consider just one more example where I hopefully get better to the point of my problems in understanding.

Two files are created, same dimensions, same data, but one without coordinates the other with coordinates.

\`\`\`python
data = np.zeros((2,3))
src_dim0 = xr.Dataset({'test': (['c', 'b'],  data)})
src_dim0.to_netcdf('src_dim0.nc')

src_dim1 = xr.Dataset({'test': (['c', 'b'],  data)}, 
                      coords={'c': (['c'], np.arange(data.shape[0])),
                              'b': (['b'], np.arange(data.shape[1])),})
src_dim1.to_netcdf('src_dim1.nc')
\`\`\`
The dump of both:
\`\`\`
netcdf src_dim0 {
dimensions:
	c = 2 ;
	b = 3 ;
variables:
	double test(c, b) ;
		test:_FillValue = NaN ;
data:

 test =
  0, 0, 0,
  0, 0, 0 ;
}

netcdf src_dim1 {
dimensions:
	c = 2 ;
	b = 3 ;
variables:
	double test(c, b) ;
		test:_FillValue = NaN ;
	int64 c(c) ;
	int64 b(b) ;
data:

 test =
  0, 0, 0,
  0, 0, 0 ;

 c = 0, 1 ;

 b = 0, 1, 2 ;
}
\`\`\`

Now, from the dump, the 'c' dimension is first in both. Lets read those files again and concat them along the `c`-dimension:

\`\`\`python
dst_dim0 = xr.open_dataset('src_dim0.nc')
dst_dim0 = xr.concat([dst_dim0, dst_dim0], dim='c')
dst_dim0.to_netcdf('dst_dim0.nc')

dst_dim1 = xr.open_dataset('src_dim1.nc')
dst_dim1 = xr.concat([dst_dim1, dst_dim1], dim='c')
dst_dim1.to_netcdf('dst_dim1.nc')
\`\`\`

Now, and this is what confuses me, the file without coordinates has 'c' dimension first and the file with coordinates has 'b' dimension first.:
\`\`\`
netcdf dst_dim0 {
dimensions:
	c = 4 ;
	b = 3 ;
variables:
	double test(c, b) ;
		test:_FillValue = NaN ;
data:

 test =
  0, 0, 0,
  0, 0, 0,
  0, 0, 0,
  0, 0, 0 ;
}

netcdf dst_dim1 {
dimensions:
	b = 3 ;
	c = 4 ;
variables:
	int64 b(b) ;
	double test(c, b) ;
		test:_FillValue = NaN ;
	int64 c(c) ;
data:

 b = 0, 1, 2 ;

 test =
  0, 0, 0,
  0, 0, 0,
  0, 0, 0,
  0, 0, 0 ;

 c = 0, 1, 0, 1 ;
}
\`\`\`

I really like to understand why there is this difference. Thanks for your patience!
This is due to the internal implementation of `xarray.concat`, which sometimes reorders variables. I doubt the reordering was intentional. It's probably just a side effect of how `concat` takes multiple passes over different types of variables to figure out how to combine them.

You are welcome to take a look at improving this, though I doubt this would be particularly easy to fix. Certainly the code in `concat` could use some clean-up, and if we can preserve the order of variables on outputs that would be an improvement in usability. But it's still not something I would consider a "bug" *per se*.
@shoyer Yes, that was what I was assuming. But was a bit confused too, as the concat docs say, that dimension order is not affected. But maybe I get this wrong and the order of dimensions is not affected only for DataArrays.

IIUC xarray creates a new dataset during concat, because the dimensions cannot be expanded (due to netCDF4 limitations). So I would need to look at that specific part, where this creation process takes place. 

I would also not speak of "bug" here, but if such reordering happens only in certain conditions users (I mean at least me) can get confused.

I'll try to find out under what conditions this happens and try to come up with some workaround. Will also try ti find my way through the concat-mechanism. Again, I really appreciate your help in this issue. 
I can rename the issue to a somewhat better name, do you have a suggestion? **Ambiguous dimension reorder in Dataset concat** maybe?


Sorry, fat fingers...
@shoyer I'm working on a notebook with all testing inside. Just found that if I have 3 dimensions ('c', 'd', 'b') the ordering is preserved in ~any case~ (~with and~ without coords) for concat along any dimension. Will link the notebook next day.

Update: Need to be more thorough...with coordinates it reorders also with 3 dims.
Just as note for me, to not have to reiterate:

- It seems, that variables are handled in creation order. Means that `concat ` reads them, handles them (even if the variable remains unchanged) and writes them to the new dataset in that order.
- This does not happen for coordinate for some reason. There only the changed dimension is handled and written after the variables. The unaffected coordinates are written at the beginning before the variables.

Example (dst concat over 'x'):
The ordering of the src_dim1 is because the dimensions in the variables/coordinates are x,y,z in that order. The ordering of the dst_dim1 is because the dimensions in the variables/coordinates are z, y, x.
\`\`\`
netcdf src_dim1 {
dimensions:
	x = 2 ;
	y = 3 ;
	z = 4 ;
variables:
	double test2(x, y) ;
		test2:_FillValue = NaN ;
	double test3(x, z) ;
		test3:_FillValue = NaN ;
	double test1(y, z) ;
		test1:_FillValue = NaN ;
	int64 z(z) ;
	int64 y(y) ;
	int64 x(x) ;
\`\`\`
\`\`\`
netcdf dst_dim1 {
dimensions:
	z = 4 ;
	y = 3 ;
	x = 4 ;
variables:
	int64 z(z) ;
	int64 y(y) ;
	double test2(x, y) ;
		test2:_FillValue = NaN ;
	double test3(x, z) ;
		test3:_FillValue = NaN ;
	double test1(x, y, z) ;
		test1:_FillValue = NaN ;
	int64 x(x) ;
\`\`\`

It seems, that the two coordinates (z and y) are written first, then the variables, and then the changed coordinate. Now trying to find, where this happens. If the two coordinates would be written in the same way as the variables (and after them), then the ordering would be x,y,z as in the source. 


@shoyer I think I found the relevant lines of code in `combine.py`. It might be possible to preserve the (*correct*) order of dimensions at least with regard to their occurrence in variables (and not in coordinates). But that would mean to treat variables and coordinates consecutively in some way..

In the docs there is a Warning: 
*We are changing the behavior of iterating over a Dataset the next major release of xarray, to only include data variables instead of both data variables and coordinates. In the meantime, prefer iterating over ds.data_vars or ds.coords.* [below here](http://xarray.pydata.org/en/stable/data-structures.html#dataset-contents).

Does that mean that this also affects internal machinery (like in concat)? If so, could you point me to some code where this is taken care of or give some explanation or links where this is discussed?

Update: I' working with latest 0.12.0 release.

That warning should be removed — we already finished that deprecation cycle!
see https://github.com/pydata/xarray/pull/2818 for removing that warning
@shoyer Attached the description of the issue source and kind of workaround.

During `concat` a `result_vars = OrderedDict()` is created.
After that it is iterated over the first dataset `datasets[0].variables.items()` and those variables which are not affected by the `concat` are added to the `result_vars` :
https://github.com/pydata/xarray/blob/a5ca64ac5988f0c9c9c6b741a5de16e81b90cad5/xarray/core/combine.py#L244-L246

After several checks the affected variables are treated and added to `result_vars`:
https://github.com/pydata/xarray/blob/a5ca64ac5988f0c9c9c6b741a5de16e81b90cad5/xarray/core/combine.py#L301-L306

The comment indicates what you already mentioned, that the reorder might be unintentional. But due to the handling in two separate iterations over `datasets[0].variables`, the source variable order is not preserved (and with that in some cases the order of the dimensions).

This can be worked around by changing the second iteration to:
\`\`\`python
# re-initialize result_vars to write in correct order
result_vars = OrderedDict()

# stack up each variable to fill-out the dataset (in order)
for k in datasets[0].variables:
    if k in concat_over:
        vars = ensure_common_dims([ds.variables[k] for ds in datasets])
        combined = concat_vars(vars, dim, positions)
        insert_result_variable(k, combined)
    else:
        insert_result_variable(k, datasets[0].variables[k])  
\`\`\`
With this workaround applied, the `concat` works as expected and the variable/coordinate order (and with that the dimension order) is preserved. I'm thinking about a better solution but wanted to get some feedback from you first, if I#m on the right track. Thanks!
After checking a bit more in older issues, this seems related: https://github.com/pydata/xarray/pull/1049, ping @fmaussion.

And also @shoyer's [comment](https://github.com/pydata/xarray/pull/1049/files#r83541671) suggest that those two iterations/loops I mentioned above need to be addressed correctly.


don't use the `Coordinates` section to check the order. `Dataset` objects may have multiple data variables (each with a possibly different order of dimensions) so for displaying it needs to define some kind of order.

What you want to compare is the order of dimensions referenced by a variable:
\`\`\`
<xarray.Dataset>
Dimensions:  (time: 1, x: 10, y: 10)
Coordinates:
  * time     (time) int64 0
  * y        (y) int64 0 1 2 3 4 5 6 7 8 9
  * x        (x) int64 0 1 2 3 4 5 6 7 8 9
Data variables:
    data     (time, y, x) bool False False False False ... False False False
             ^^^^^^^^^^^^
\`\`\`
and you will notice that the order does not change.

To make this less confusing, maybe we should always sort the coordinates when displaying them? I'm guessing that right now they are displayed in the order they were added to the coordinates.
This rings a bell! It has be discussed in https://github.com/pydata/xarray/issues/2811. I've done extensive checks back then, but didn't come up with an PR.

My workaround or solution is outlined [in this comment](https://github.com/pydata/xarray/issues/2811#issuecomment-473810333). Code might have been changed!
@keewis yes you are right, but still a consistent ordering would be less confusing, including maybe also the Dimensions field.
Now Dimensions shows first x and then y, even though the data itself has y,x ordering.
```

## Patch

```diff
diff --git a/xarray/core/concat.py b/xarray/core/concat.py
--- a/xarray/core/concat.py
+++ b/xarray/core/concat.py
@@ -463,6 +463,9 @@ def ensure_common_dims(vars):
             combined = concat_vars(vars, dim, positions)
             assert isinstance(combined, Variable)
             result_vars[k] = combined
+        elif k in result_vars:
+            # preserves original variable order
+            result_vars[k] = result_vars.pop(k)
 
     result = Dataset(result_vars, attrs=result_attrs)
     absent_coord_names = coord_names - set(result.variables)

```

## Test Patch

```diff
diff --git a/xarray/tests/test_concat.py b/xarray/tests/test_concat.py
--- a/xarray/tests/test_concat.py
+++ b/xarray/tests/test_concat.py
@@ -558,3 +558,36 @@ def test_concat_merge_single_non_dim_coord():
     for coords in ["different", "all"]:
         with raises_regex(ValueError, "'y' not present in all datasets"):
             concat([da1, da2, da3], dim="x")
+
+
+def test_concat_preserve_coordinate_order():
+    x = np.arange(0, 5)
+    y = np.arange(0, 10)
+    time = np.arange(0, 4)
+    data = np.zeros((4, 10, 5), dtype=bool)
+
+    ds1 = Dataset(
+        {"data": (["time", "y", "x"], data[0:2])},
+        coords={"time": time[0:2], "y": y, "x": x},
+    )
+    ds2 = Dataset(
+        {"data": (["time", "y", "x"], data[2:4])},
+        coords={"time": time[2:4], "y": y, "x": x},
+    )
+
+    expected = Dataset(
+        {"data": (["time", "y", "x"], data)},
+        coords={"time": time, "y": y, "x": x},
+    )
+
+    actual = concat([ds1, ds2], dim="time")
+
+    # check dimension order
+    for act, exp in zip(actual.dims, expected.dims):
+        assert act == exp
+        assert actual.dims[act] == expected.dims[exp]
+
+    # check coordinate order
+    for act, exp in zip(actual.coords, expected.coords):
+        assert act == exp
+        assert_identical(actual.coords[act], expected.coords[exp])

```


## Code snippets

### 1 - xarray/core/merge.py:

Start line: 635, End line: 842

```python
def merge(
    objects: Iterable[Union["DataArray", "CoercibleMapping"]],
    compat: str = "no_conflicts",
    join: str = "outer",
    fill_value: object = dtypes.NA,
    combine_attrs: str = "drop",
) -> "Dataset":
    """Merge any number of xarray objects into a single Dataset as variables.

    Parameters
    ----------
    objects : iterable of Dataset or iterable of DataArray or iterable of dict-like
        Merge together all variables from these objects. If any of them are
        DataArray objects, they must have a name.
    compat : {"identical", "equals", "broadcast_equals", "no_conflicts", "override"}, optional
        String indicating how to compare variables of the same name for
        potential conflicts:

        - "broadcast_equals": all values must be equal when variables are
          broadcast against each other to ensure common dimensions.
        - "equals": all values and dimensions must be the same.
        - "identical": all values, dimensions and attributes must be the
          same.
        - "no_conflicts": only values which are not null in both datasets
          must be equal. The returned dataset then contains the combination
          of all non-null values.
        - "override": skip comparing and pick variable from first dataset
    join : {"outer", "inner", "left", "right", "exact"}, optional
        String indicating how to combine differing indexes in objects.

        - "outer": use the union of object indexes
        - "inner": use the intersection of object indexes
        - "left": use indexes from the first object with each dimension
        - "right": use indexes from the last object with each dimension
        - "exact": instead of aligning, raise `ValueError` when indexes to be
          aligned are not equal
        - "override": if indexes are of same size, rewrite indexes to be
          those of the first object with that dimension. Indexes for the same
          dimension must have the same size in all objects.
    fill_value : scalar or dict-like, optional
        Value to use for newly missing values. If a dict-like, maps
        variable names to fill values. Use a data array's name to
        refer to its values.
    combine_attrs : {"drop", "identical", "no_conflicts", "override"}, \
                    default: "drop"
        String indicating how to combine attrs of the objects being merged:

        - "drop": empty attrs on returned Dataset.
        - "identical": all attrs must be the same on every object.
        - "no_conflicts": attrs from all objects are combined, any that have
          the same name must also have the same value.
        - "override": skip comparing and copy attrs from the first dataset to
          the result.

    Returns
    -------
    Dataset
        Dataset with combined variables from each object.

    Examples
    --------
    >>> import xarray as xr
    >>> x = xr.DataArray(
    ...     [[1.0, 2.0], [3.0, 5.0]],
    ...     dims=("lat", "lon"),
    ...     coords={"lat": [35.0, 40.0], "lon": [100.0, 120.0]},
    ...     name="var1",
    ... )
    >>> y = xr.DataArray(
    ...     [[5.0, 6.0], [7.0, 8.0]],
    ...     dims=("lat", "lon"),
    ...     coords={"lat": [35.0, 42.0], "lon": [100.0, 150.0]},
    ...     name="var2",
    ... )
    >>> z = xr.DataArray(
    ...     [[0.0, 3.0], [4.0, 9.0]],
    ...     dims=("time", "lon"),
    ...     coords={"time": [30.0, 60.0], "lon": [100.0, 150.0]},
    ...     name="var3",
    ... )

    >>> x
    <xarray.DataArray 'var1' (lat: 2, lon: 2)>
    array([[1., 2.],
           [3., 5.]])
    Coordinates:
      * lat      (lat) float64 35.0 40.0
      * lon      (lon) float64 100.0 120.0

    >>> y
    <xarray.DataArray 'var2' (lat: 2, lon: 2)>
    array([[5., 6.],
           [7., 8.]])
    Coordinates:
      * lat      (lat) float64 35.0 42.0
      * lon      (lon) float64 100.0 150.0

    >>> z
    <xarray.DataArray 'var3' (time: 2, lon: 2)>
    array([[0., 3.],
           [4., 9.]])
    Coordinates:
      * time     (time) float64 30.0 60.0
      * lon      (lon) float64 100.0 150.0

    >>> xr.merge([x, y, z])
    <xarray.Dataset>
    Dimensions:  (lat: 3, lon: 3, time: 2)
    Coordinates:
      * lat      (lat) float64 35.0 40.0 42.0
      * lon      (lon) float64 100.0 120.0 150.0
      * time     (time) float64 30.0 60.0
    Data variables:
        var1     (lat, lon) float64 1.0 2.0 nan 3.0 5.0 nan nan nan nan
        var2     (lat, lon) float64 5.0 nan 6.0 nan nan nan 7.0 nan 8.0
        var3     (time, lon) float64 0.0 nan 3.0 4.0 nan 9.0

    >>> xr.merge([x, y, z], compat="identical")
    <xarray.Dataset>
    Dimensions:  (lat: 3, lon: 3, time: 2)
    Coordinates:
      * lat      (lat) float64 35.0 40.0 42.0
      * lon      (lon) float64 100.0 120.0 150.0
      * time     (time) float64 30.0 60.0
    Data variables:
        var1     (lat, lon) float64 1.0 2.0 nan 3.0 5.0 nan nan nan nan
        var2     (lat, lon) float64 5.0 nan 6.0 nan nan nan 7.0 nan 8.0
        var3     (time, lon) float64 0.0 nan 3.0 4.0 nan 9.0

    >>> xr.merge([x, y, z], compat="equals")
    <xarray.Dataset>
    Dimensions:  (lat: 3, lon: 3, time: 2)
    Coordinates:
      * lat      (lat) float64 35.0 40.0 42.0
      * lon      (lon) float64 100.0 120.0 150.0
      * time     (time) float64 30.0 60.0
    Data variables:
        var1     (lat, lon) float64 1.0 2.0 nan 3.0 5.0 nan nan nan nan
        var2     (lat, lon) float64 5.0 nan 6.0 nan nan nan 7.0 nan 8.0
        var3     (time, lon) float64 0.0 nan 3.0 4.0 nan 9.0

    >>> xr.merge([x, y, z], compat="equals", fill_value=-999.0)
    <xarray.Dataset>
    Dimensions:  (lat: 3, lon: 3, time: 2)
    Coordinates:
      * lat      (lat) float64 35.0 40.0 42.0
      * lon      (lon) float64 100.0 120.0 150.0
      * time     (time) float64 30.0 60.0
    Data variables:
        var1     (lat, lon) float64 1.0 2.0 -999.0 3.0 ... -999.0 -999.0 -999.0
        var2     (lat, lon) float64 5.0 -999.0 6.0 -999.0 ... -999.0 7.0 -999.0 8.0
        var3     (time, lon) float64 0.0 -999.0 3.0 4.0 -999.0 9.0

    >>> xr.merge([x, y, z], join="override")
    <xarray.Dataset>
    Dimensions:  (lat: 2, lon: 2, time: 2)
    Coordinates:
      * lat      (lat) float64 35.0 40.0
      * lon      (lon) float64 100.0 120.0
      * time     (time) float64 30.0 60.0
    Data variables:
        var1     (lat, lon) float64 1.0 2.0 3.0 5.0
        var2     (lat, lon) float64 5.0 6.0 7.0 8.0
        var3     (time, lon) float64 0.0 3.0 4.0 9.0

    >>> xr.merge([x, y, z], join="inner")
    <xarray.Dataset>
    Dimensions:  (lat: 1, lon: 1, time: 2)
    Coordinates:
      * lat      (lat) float64 35.0
      * lon      (lon) float64 100.0
      * time     (time) float64 30.0 60.0
    Data variables:
        var1     (lat, lon) float64 1.0
        var2     (lat, lon) float64 5.0
        var3     (time, lon) float64 0.0 4.0

    >>> xr.merge([x, y, z], compat="identical", join="inner")
    <xarray.Dataset>
    Dimensions:  (lat: 1, lon: 1, time: 2)
    Coordinates:
      * lat      (lat) float64 35.0
      * lon      (lon) float64 100.0
      * time     (time) float64 30.0 60.0
    Data variables:
        var1     (lat, lon) float64 1.0
        var2     (lat, lon) float64 5.0
        var3     (time, lon) float64 0.0 4.0

    >>> xr.merge([x, y, z], compat="broadcast_equals", join="outer")
    <xarray.Dataset>
    Dimensions:  (lat: 3, lon: 3, time: 2)
    Coordinates:
      * lat      (lat) float64 35.0 40.0 42.0
      * lon      (lon) float64 100.0 120.0 150.0
      * time     (time) float64 30.0 60.0
    Data variables:
        var1     (lat, lon) float64 1.0 2.0 nan 3.0 5.0 nan nan nan nan
        var2     (lat, lon) float64 5.0 nan 6.0 nan nan nan 7.0 nan 8.0
        var3     (time, lon) float64 0.0 nan 3.0 4.0 nan 9.0

    >>> xr.merge([x, y, z], join="exact")
    Traceback (most recent call last):
    ...
    ValueError: indexes along dimension 'lat' are not equal

    Raises
    ------
    xarray.MergeError
        If any variables with the same name have conflicting values.

    See also
    --------
    concat
    """
    # ... other code
```
### 2 - xarray/core/combine.py:

Start line: 551, End line: 749

```python
def combine_by_coords(
    datasets,
    compat="no_conflicts",
    data_vars="all",
    coords="different",
    fill_value=dtypes.NA,
    join="outer",
    combine_attrs="no_conflicts",
):
    """
    Attempt to auto-magically combine the given datasets into one by using
    dimension coordinates.

    This method attempts to combine a group of datasets along any number of
    dimensions into a single entity by inspecting coords and metadata and using
    a combination of concat and merge.

    Will attempt to order the datasets such that the values in their dimension
    coordinates are monotonic along all dimensions. If it cannot determine the
    order in which to concatenate the datasets, it will raise a ValueError.
    Non-coordinate dimensions will be ignored, as will any coordinate
    dimensions which do not vary between each dataset.

    Aligns coordinates, but different variables on datasets can cause it
    to fail under some scenarios. In complex cases, you may need to clean up
    your data and use concat/merge explicitly (also see `manual_combine`).

    Works well if, for example, you have N years of data and M data variables,
    and each combination of a distinct time period and set of data variables is
    saved as its own dataset. Also useful for if you have a simulation which is
    parallelized in multiple dimensions, but has global coordinates saved in
    each file specifying the positions of points within the global domain.

    Parameters
    ----------
    datasets : sequence of xarray.Dataset
        Dataset objects to combine.
    compat : {"identical", "equals", "broadcast_equals", "no_conflicts", "override"}, optional
        String indicating how to compare variables of the same name for
        potential conflicts:

        - "broadcast_equals": all values must be equal when variables are
          broadcast against each other to ensure common dimensions.
        - "equals": all values and dimensions must be the same.
        - "identical": all values, dimensions and attributes must be the
          same.
        - "no_conflicts": only values which are not null in both datasets
          must be equal. The returned dataset then contains the combination
          of all non-null values.
        - "override": skip comparing and pick variable from first dataset
    data_vars : {"minimal", "different", "all" or list of str}, optional
        These data variables will be concatenated together:

        * "minimal": Only data variables in which the dimension already
          appears are included.
        * "different": Data variables which are not equal (ignoring
          attributes) across all datasets are also concatenated (as well as
          all for which dimension already appears). Beware: this option may
          load the data payload of data variables into memory if they are not
          already loaded.
        * "all": All data variables will be concatenated.
        * list of str: The listed data variables will be concatenated, in
          addition to the "minimal" data variables.

        If objects are DataArrays, `data_vars` must be "all".
    coords : {"minimal", "different", "all"} or list of str, optional
        As per the "data_vars" kwarg, but for coordinate variables.
    fill_value : scalar or dict-like, optional
        Value to use for newly missing values. If a dict-like, maps
        variable names to fill values. Use a data array's name to
        refer to its values. If None, raises a ValueError if
        the passed Datasets do not create a complete hypercube.
    join : {"outer", "inner", "left", "right", "exact"}, optional
        String indicating how to combine differing indexes
        (excluding concat_dim) in objects

        - "outer": use the union of object indexes
        - "inner": use the intersection of object indexes
        - "left": use indexes from the first object with each dimension
        - "right": use indexes from the last object with each dimension
        - "exact": instead of aligning, raise `ValueError` when indexes to be
          aligned are not equal
        - "override": if indexes are of same size, rewrite indexes to be
          those of the first object with that dimension. Indexes for the same
          dimension must have the same size in all objects.
    combine_attrs : {"drop", "identical", "no_conflicts", "override"}, \
                    default: "drop"
        String indicating how to combine attrs of the objects being merged:

        - "drop": empty attrs on returned Dataset.
        - "identical": all attrs must be the same on every object.
        - "no_conflicts": attrs from all objects are combined, any that have
          the same name must also have the same value.
        - "override": skip comparing and copy attrs from the first dataset to
          the result.

    Returns
    -------
    combined : xarray.Dataset

    See also
    --------
    concat
    merge
    combine_nested

    Examples
    --------

    Combining two datasets using their common dimension coordinates. Notice
    they are concatenated based on the values in their dimension coordinates,
    not on their position in the list passed to `combine_by_coords`.

    >>> import numpy as np
    >>> import xarray as xr

    >>> x1 = xr.Dataset(
    ...     {
    ...         "temperature": (("y", "x"), 20 * np.random.rand(6).reshape(2, 3)),
    ...         "precipitation": (("y", "x"), np.random.rand(6).reshape(2, 3)),
    ...     },
    ...     coords={"y": [0, 1], "x": [10, 20, 30]},
    ... )
    >>> x2 = xr.Dataset(
    ...     {
    ...         "temperature": (("y", "x"), 20 * np.random.rand(6).reshape(2, 3)),
    ...         "precipitation": (("y", "x"), np.random.rand(6).reshape(2, 3)),
    ...     },
    ...     coords={"y": [2, 3], "x": [10, 20, 30]},
    ... )
    >>> x3 = xr.Dataset(
    ...     {
    ...         "temperature": (("y", "x"), 20 * np.random.rand(6).reshape(2, 3)),
    ...         "precipitation": (("y", "x"), np.random.rand(6).reshape(2, 3)),
    ...     },
    ...     coords={"y": [2, 3], "x": [40, 50, 60]},
    ... )

    >>> x1
    <xarray.Dataset>
    Dimensions:        (x: 3, y: 2)
    Coordinates:
      * y              (y) int64 0 1
      * x              (x) int64 10 20 30
    Data variables:
        temperature    (y, x) float64 10.98 14.3 12.06 10.9 8.473 12.92
        precipitation  (y, x) float64 0.4376 0.8918 0.9637 0.3834 0.7917 0.5289

    >>> x2
    <xarray.Dataset>
    Dimensions:        (x: 3, y: 2)
    Coordinates:
      * y              (y) int64 2 3
      * x              (x) int64 10 20 30
    Data variables:
        temperature    (y, x) float64 11.36 18.51 1.421 1.743 0.4044 16.65
        precipitation  (y, x) float64 0.7782 0.87 0.9786 0.7992 0.4615 0.7805

    >>> x3
    <xarray.Dataset>
    Dimensions:        (x: 3, y: 2)
    Coordinates:
      * y              (y) int64 2 3
      * x              (x) int64 40 50 60
    Data variables:
        temperature    (y, x) float64 2.365 12.8 2.867 18.89 10.44 8.293
        precipitation  (y, x) float64 0.2646 0.7742 0.4562 0.5684 0.01879 0.6176

    >>> xr.combine_by_coords([x2, x1])
    <xarray.Dataset>
    Dimensions:        (x: 3, y: 4)
    Coordinates:
      * x              (x) int64 10 20 30
      * y              (y) int64 0 1 2 3
    Data variables:
        temperature    (y, x) float64 10.98 14.3 12.06 10.9 ... 1.743 0.4044 16.65
        precipitation  (y, x) float64 0.4376 0.8918 0.9637 ... 0.7992 0.4615 0.7805

    >>> xr.combine_by_coords([x3, x1])
    <xarray.Dataset>
    Dimensions:        (x: 6, y: 4)
    Coordinates:
      * x              (x) int64 10 20 30 40 50 60
      * y              (y) int64 0 1 2 3
    Data variables:
        temperature    (y, x) float64 10.98 14.3 12.06 nan ... nan 18.89 10.44 8.293
        precipitation  (y, x) float64 0.4376 0.8918 0.9637 ... 0.5684 0.01879 0.6176

    >>> xr.combine_by_coords([x3, x1], join="override")
    <xarray.Dataset>
    Dimensions:        (x: 3, y: 4)
    Coordinates:
      * x              (x) int64 10 20 30
      * y              (y) int64 0 1 2 3
    Data variables:
        temperature    (y, x) float64 10.98 14.3 12.06 10.9 ... 18.89 10.44 8.293
        precipitation  (y, x) float64 0.4376 0.8918 0.9637 ... 0.5684 0.01879 0.6176

    >>> xr.combine_by_coords([x1, x2, x3])
    <xarray.Dataset>
    Dimensions:        (x: 6, y: 4)
    Coordinates:
      * x              (x) int64 10 20 30 40 50 60
      * y              (y) int64 0 1 2 3
    Data variables:
        temperature    (y, x) float64 10.98 14.3 12.06 nan ... 18.89 10.44 8.293
        precipitation  (y, x) float64 0.4376 0.8918 0.9637 ... 0.5684 0.01879 0.6176
    """
    # ... other code
```
### 3 - xarray/core/alignment.py:

Start line: 69, End line: 258

```python
def align(
    *objects,
    join="inner",
    copy=True,
    indexes=None,
    exclude=frozenset(),
    fill_value=dtypes.NA,
):
    """
    Given any number of Dataset and/or DataArray objects, returns new
    objects with aligned indexes and dimension sizes.

    Array from the aligned objects are suitable as input to mathematical
    operators, because along each dimension they have the same index and size.

    Missing values (if ``join != 'inner'``) are filled with ``fill_value``.
    The default fill value is NaN.

    Parameters
    ----------
    *objects : Dataset or DataArray
        Objects to align.
    join : {"outer", "inner", "left", "right", "exact", "override"}, optional
        Method for joining the indexes of the passed objects along each
        dimension:

        - "outer": use the union of object indexes
        - "inner": use the intersection of object indexes
        - "left": use indexes from the first object with each dimension
        - "right": use indexes from the last object with each dimension
        - "exact": instead of aligning, raise `ValueError` when indexes to be
          aligned are not equal
        - "override": if indexes are of same size, rewrite indexes to be
          those of the first object with that dimension. Indexes for the same
          dimension must have the same size in all objects.
    copy : bool, optional
        If ``copy=True``, data in the return values is always copied. If
        ``copy=False`` and reindexing is unnecessary, or can be performed with
        only slice operations, then the output may share memory with the input.
        In either case, new xarray objects are always returned.
    indexes : dict-like, optional
        Any indexes explicitly provided with the `indexes` argument should be
        used in preference to the aligned indexes.
    exclude : sequence of str, optional
        Dimensions that must be excluded from alignment
    fill_value : scalar or dict-like, optional
        Value to use for newly missing values. If a dict-like, maps
        variable names to fill values. Use a data array's name to
        refer to its values.

    Returns
    -------
    aligned : DataArray or Dataset
        Tuple of objects with the same type as `*objects` with aligned
        coordinates.

    Raises
    ------
    ValueError
        If any dimensions without labels on the arguments have different sizes,
        or a different size than the size of the aligned dimension labels.

    Examples
    --------

    >>> import xarray as xr
    >>> x = xr.DataArray(
    ...     [[25, 35], [10, 24]],
    ...     dims=("lat", "lon"),
    ...     coords={"lat": [35.0, 40.0], "lon": [100.0, 120.0]},
    ... )
    >>> y = xr.DataArray(
    ...     [[20, 5], [7, 13]],
    ...     dims=("lat", "lon"),
    ...     coords={"lat": [35.0, 42.0], "lon": [100.0, 120.0]},
    ... )

    >>> x
    <xarray.DataArray (lat: 2, lon: 2)>
    array([[25, 35],
           [10, 24]])
    Coordinates:
      * lat      (lat) float64 35.0 40.0
      * lon      (lon) float64 100.0 120.0

    >>> y
    <xarray.DataArray (lat: 2, lon: 2)>
    array([[20,  5],
           [ 7, 13]])
    Coordinates:
      * lat      (lat) float64 35.0 42.0
      * lon      (lon) float64 100.0 120.0

    >>> a, b = xr.align(x, y)
    >>> a
    <xarray.DataArray (lat: 1, lon: 2)>
    array([[25, 35]])
    Coordinates:
      * lat      (lat) float64 35.0
      * lon      (lon) float64 100.0 120.0
    >>> b
    <xarray.DataArray (lat: 1, lon: 2)>
    array([[20,  5]])
    Coordinates:
      * lat      (lat) float64 35.0
      * lon      (lon) float64 100.0 120.0

    >>> a, b = xr.align(x, y, join="outer")
    >>> a
    <xarray.DataArray (lat: 3, lon: 2)>
    array([[25., 35.],
           [10., 24.],
           [nan, nan]])
    Coordinates:
      * lat      (lat) float64 35.0 40.0 42.0
      * lon      (lon) float64 100.0 120.0
    >>> b
    <xarray.DataArray (lat: 3, lon: 2)>
    array([[20.,  5.],
           [nan, nan],
           [ 7., 13.]])
    Coordinates:
      * lat      (lat) float64 35.0 40.0 42.0
      * lon      (lon) float64 100.0 120.0

    >>> a, b = xr.align(x, y, join="outer", fill_value=-999)
    >>> a
    <xarray.DataArray (lat: 3, lon: 2)>
    array([[  25,   35],
           [  10,   24],
           [-999, -999]])
    Coordinates:
      * lat      (lat) float64 35.0 40.0 42.0
      * lon      (lon) float64 100.0 120.0
    >>> b
    <xarray.DataArray (lat: 3, lon: 2)>
    array([[  20,    5],
           [-999, -999],
           [   7,   13]])
    Coordinates:
      * lat      (lat) float64 35.0 40.0 42.0
      * lon      (lon) float64 100.0 120.0

    >>> a, b = xr.align(x, y, join="left")
    >>> a
    <xarray.DataArray (lat: 2, lon: 2)>
    array([[25, 35],
           [10, 24]])
    Coordinates:
      * lat      (lat) float64 35.0 40.0
      * lon      (lon) float64 100.0 120.0
    >>> b
    <xarray.DataArray (lat: 2, lon: 2)>
    array([[20.,  5.],
           [nan, nan]])
    Coordinates:
      * lat      (lat) float64 35.0 40.0
      * lon      (lon) float64 100.0 120.0

    >>> a, b = xr.align(x, y, join="right")
    >>> a
    <xarray.DataArray (lat: 2, lon: 2)>
    array([[25., 35.],
           [nan, nan]])
    Coordinates:
      * lat      (lat) float64 35.0 42.0
      * lon      (lon) float64 100.0 120.0
    >>> b
    <xarray.DataArray (lat: 2, lon: 2)>
    array([[20,  5],
           [ 7, 13]])
    Coordinates:
      * lat      (lat) float64 35.0 42.0
      * lon      (lon) float64 100.0 120.0

    >>> a, b = xr.align(x, y, join="exact")
    Traceback (most recent call last):
    ...
        "indexes along dimension {!r} are not equal".format(dim)
    ValueError: indexes along dimension 'lat' are not equal

    >>> a, b = xr.align(x, y, join="override")
    >>> a
    <xarray.DataArray (lat: 2, lon: 2)>
    array([[25, 35],
           [10, 24]])
    Coordinates:
      * lat      (lat) float64 35.0 40.0
      * lon      (lon) float64 100.0 120.0
    >>> b
    <xarray.DataArray (lat: 2, lon: 2)>
    array([[20,  5],
           [ 7, 13]])
    Coordinates:
      * lat      (lat) float64 35.0 40.0
      * lon      (lon) float64 100.0 120.0

    """
    # ... other code
```
### 4 - xarray/core/dataset.py:

Start line: 1, End line: 136

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
    _check_inplace,
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
### 5 - xarray/core/combine.py:

Start line: 47, End line: 113

```python
def _infer_concat_order_from_coords(datasets):

    concat_dims = []
    tile_ids = [() for ds in datasets]

    # All datasets have same variables because they've been grouped as such
    ds0 = datasets[0]
    for dim in ds0.dims:

        # Check if dim is a coordinate dimension
        if dim in ds0:

            # Need to read coordinate values to do ordering
            indexes = [ds.indexes.get(dim) for ds in datasets]
            if any(index is None for index in indexes):
                raise ValueError(
                    "Every dimension needs a coordinate for "
                    "inferring concatenation order"
                )

            # If dimension coordinate values are same on every dataset then
            # should be leaving this dimension alone (it's just a "bystander")
            if not all(index.equals(indexes[0]) for index in indexes[1:]):

                # Infer order datasets should be arranged in along this dim
                concat_dims.append(dim)

                if all(index.is_monotonic_increasing for index in indexes):
                    ascending = True
                elif all(index.is_monotonic_decreasing for index in indexes):
                    ascending = False
                else:
                    raise ValueError(
                        "Coordinate variable {} is neither "
                        "monotonically increasing nor "
                        "monotonically decreasing on all datasets".format(dim)
                    )

                # Assume that any two datasets whose coord along dim starts
                # with the same value have the same coord values throughout.
                if any(index.size == 0 for index in indexes):
                    raise ValueError("Cannot handle size zero dimensions")
                first_items = pd.Index([index[0] for index in indexes])

                # Sort datasets along dim
                # We want rank but with identical elements given identical
                # position indices - they should be concatenated along another
                # dimension, not along this one
                series = first_items.to_series()
                rank = series.rank(method="dense", ascending=ascending)
                order = rank.astype(int).values - 1

                # Append positions along extra dimension to structure which
                # encodes the multi-dimensional concatentation order
                tile_ids = [
                    tile_id + (position,) for tile_id, position in zip(tile_ids, order)
                ]

    if len(datasets) > 1 and not concat_dims:
        raise ValueError(
            "Could not find any dimension coordinates to use to "
            "order the datasets for concatenation"
        )

    combined_ids = dict(zip(tile_ids, datasets))

    return combined_ids, concat_dims
```
### 6 - xarray/core/dataset.py:

Start line: 4418, End line: 5177

```python
class Dataset(Mapping, ImplementsDatasetReduce, DataWithCoords):

    def assign(
        self, variables: Mapping[Hashable, Any] = None, **variables_kwargs: Hashable
    ) -> "Dataset":
        """Assign new data variables to a Dataset, returning a new object
        with all the original variables in addition to the new ones.

        Parameters
        ----------
        variables : mapping of hashable to Any
            Mapping from variables names to the new values. If the new values
            are callable, they are computed on the Dataset and assigned to new
            data variables. If the values are not callable, (e.g. a DataArray,
            scalar, or array), they are simply assigned.
        **variables_kwargs
            The keyword arguments form of ``variables``.
            One of variables or variables_kwargs must be provided.

        Returns
        -------
        ds : Dataset
            A new Dataset with the new variables in addition to all the
            existing variables.

        Notes
        -----
        Since ``kwargs`` is a dictionary, the order of your arguments may not
        be preserved, and so the order of the new variables is not well
        defined. Assigning multiple variables within the same ``assign`` is
        possible, but you cannot reference other variables created within the
        same ``assign`` call.

        See Also
        --------
        pandas.DataFrame.assign

        Examples
        --------
        >>> x = xr.Dataset(
        ...     {
        ...         "temperature_c": (
        ...             ("lat", "lon"),
        ...             20 * np.random.rand(4).reshape(2, 2),
        ...         ),
        ...         "precipitation": (("lat", "lon"), np.random.rand(4).reshape(2, 2)),
        ...     },
        ...     coords={"lat": [10, 20], "lon": [150, 160]},
        ... )
        >>> x
        <xarray.Dataset>
        Dimensions:        (lat: 2, lon: 2)
        Coordinates:
          * lat            (lat) int64 10 20
          * lon            (lon) int64 150 160
        Data variables:
            temperature_c  (lat, lon) float64 10.98 14.3 12.06 10.9
            precipitation  (lat, lon) float64 0.4237 0.6459 0.4376 0.8918

        Where the value is a callable, evaluated on dataset:

        >>> x.assign(temperature_f=lambda x: x.temperature_c * 9 / 5 + 32)
        <xarray.Dataset>
        Dimensions:        (lat: 2, lon: 2)
        Coordinates:
          * lat            (lat) int64 10 20
          * lon            (lon) int64 150 160
        Data variables:
            temperature_c  (lat, lon) float64 10.98 14.3 12.06 10.9
            precipitation  (lat, lon) float64 0.4237 0.6459 0.4376 0.8918
            temperature_f  (lat, lon) float64 51.76 57.75 53.7 51.62

        Alternatively, the same behavior can be achieved by directly referencing an existing dataarray:

        >>> x.assign(temperature_f=x["temperature_c"] * 9 / 5 + 32)
        <xarray.Dataset>
        Dimensions:        (lat: 2, lon: 2)
        Coordinates:
          * lat            (lat) int64 10 20
          * lon            (lon) int64 150 160
        Data variables:
            temperature_c  (lat, lon) float64 10.98 14.3 12.06 10.9
            precipitation  (lat, lon) float64 0.4237 0.6459 0.4376 0.8918
            temperature_f  (lat, lon) float64 51.76 57.75 53.7 51.62

        """
        variables = either_dict_or_kwargs(variables, variables_kwargs, "assign")
        data = self.copy()
        # do all calculations first...
        results = data._calc_assign_results(variables)
        # ... and then assign
        data.update(results)
        return data

    def to_array(self, dim="variable", name=None):
        """Convert this dataset into an xarray.DataArray

        The data variables of this dataset will be broadcast against each other
        and stacked along the first axis of the new array. All coordinates of
        this dataset will remain coordinates.

        Parameters
        ----------
        dim : str, optional
            Name of the new dimension.
        name : str, optional
            Name of the new data array.

        Returns
        -------
        array : xarray.DataArray
        """
        from .dataarray import DataArray

        data_vars = [self.variables[k] for k in self.data_vars]
        broadcast_vars = broadcast_variables(*data_vars)
        data = duck_array_ops.stack([b.data for b in broadcast_vars], axis=0)

        coords = dict(self.coords)
        coords[dim] = list(self.data_vars)
        indexes = propagate_indexes(self._indexes)

        dims = (dim,) + broadcast_vars[0].dims

        return DataArray(
            data, coords, dims, attrs=self.attrs, name=name, indexes=indexes
        )

    def _normalize_dim_order(
        self, dim_order: List[Hashable] = None
    ) -> Dict[Hashable, int]:
        """
        Check the validity of the provided dimensions if any and return the mapping
        between dimension name and their size.

        Parameters
        ----------
        dim_order
            Dimension order to validate (default to the alphabetical order if None).

        Returns
        -------
        result
            Validated dimensions mapping.

        """
        if dim_order is None:
            dim_order = list(self.dims)
        elif set(dim_order) != set(self.dims):
            raise ValueError(
                "dim_order {} does not match the set of dimensions of this "
                "Dataset: {}".format(dim_order, list(self.dims))
            )

        ordered_dims = {k: self.dims[k] for k in dim_order}

        return ordered_dims

    def _to_dataframe(self, ordered_dims: Mapping[Hashable, int]):
        columns = [k for k in self.variables if k not in self.dims]
        data = [
            self._variables[k].set_dims(ordered_dims).values.reshape(-1)
            for k in columns
        ]
        index = self.coords.to_index([*ordered_dims])
        return pd.DataFrame(dict(zip(columns, data)), index=index)

    def to_dataframe(self, dim_order: List[Hashable] = None) -> pd.DataFrame:
        """Convert this dataset into a pandas.DataFrame.

        Non-index variables in this dataset form the columns of the
        DataFrame. The DataFrame is indexed by the Cartesian product of
        this dataset's indices.

        Parameters
        ----------
        dim_order
            Hierarchical dimension order for the resulting dataframe. All
            arrays are transposed to this order and then written out as flat
            vectors in contiguous order, so the last dimension in this list
            will be contiguous in the resulting DataFrame. This has a major
            influence on which operations are efficient on the resulting
            dataframe.

            If provided, must include all dimensions of this dataset. By
            default, dimensions are sorted alphabetically.

        Returns
        -------
        result
            Dataset as a pandas DataFrame.

        """

        ordered_dims = self._normalize_dim_order(dim_order=dim_order)

        return self._to_dataframe(ordered_dims=ordered_dims)

    def _set_sparse_data_from_dataframe(
        self, idx: pd.Index, arrays: List[Tuple[Hashable, np.ndarray]], dims: tuple
    ) -> None:
        from sparse import COO

        if isinstance(idx, pd.MultiIndex):
            coords = np.stack([np.asarray(code) for code in idx.codes], axis=0)
            is_sorted = idx.is_lexsorted()
            shape = tuple(lev.size for lev in idx.levels)
        else:
            coords = np.arange(idx.size).reshape(1, -1)
            is_sorted = True
            shape = (idx.size,)

        for name, values in arrays:
            # In virtually all real use cases, the sparse array will now have
            # missing values and needs a fill_value. For consistency, don't
            # special case the rare exceptions (e.g., dtype=int without a
            # MultiIndex).
            dtype, fill_value = dtypes.maybe_promote(values.dtype)
            values = np.asarray(values, dtype=dtype)

            data = COO(
                coords,
                values,
                shape,
                has_duplicates=False,
                sorted=is_sorted,
                fill_value=fill_value,
            )
            self[name] = (dims, data)

    def _set_numpy_data_from_dataframe(
        self, idx: pd.Index, arrays: List[Tuple[Hashable, np.ndarray]], dims: tuple
    ) -> None:
        if not isinstance(idx, pd.MultiIndex):
            for name, values in arrays:
                self[name] = (dims, values)
            return

        shape = tuple(lev.size for lev in idx.levels)
        indexer = tuple(idx.codes)

        # We already verified that the MultiIndex has all unique values, so
        # there are missing values if and only if the size of output arrays is
        # larger that the index.
        missing_values = np.prod(shape) > idx.shape[0]

        for name, values in arrays:
            # NumPy indexing is much faster than using DataFrame.reindex() to
            # fill in missing values:
            # https://stackoverflow.com/a/35049899/809705
            if missing_values:
                dtype, fill_value = dtypes.maybe_promote(values.dtype)
                data = np.full(shape, fill_value, dtype)
            else:
                # If there are no missing values, keep the existing dtype
                # instead of promoting to support NA, e.g., keep integer
                # columns as integers.
                # TODO: consider removing this special case, which doesn't
                # exist for sparse=True.
                data = np.zeros(shape, values.dtype)
            data[indexer] = values
            self[name] = (dims, data)

    @classmethod
    def from_dataframe(cls, dataframe: pd.DataFrame, sparse: bool = False) -> "Dataset":
        """Convert a pandas.DataFrame into an xarray.Dataset

        Each column will be converted into an independent variable in the
        Dataset. If the dataframe's index is a MultiIndex, it will be expanded
        into a tensor product of one-dimensional indices (filling in missing
        values with NaN). This method will produce a Dataset very similar to
        that on which the 'to_dataframe' method was called, except with
        possibly redundant dimensions (since all dataset variables will have
        the same dimensionality)

        Parameters
        ----------
        dataframe : DataFrame
            DataFrame from which to copy data and indices.
        sparse : bool, default: False
            If true, create a sparse arrays instead of dense numpy arrays. This
            can potentially save a large amount of memory if the DataFrame has
            a MultiIndex. Requires the sparse package (sparse.pydata.org).

        Returns
        -------
        New Dataset.

        See also
        --------
        xarray.DataArray.from_series
        pandas.DataFrame.to_xarray
        """
        # TODO: Add an option to remove dimensions along which the variables
        # are constant, to enable consistent serialization to/from a dataframe,
        # even if some variables have different dimensionality.

        if not dataframe.columns.is_unique:
            raise ValueError("cannot convert DataFrame with non-unique columns")

        idx = remove_unused_levels_categories(dataframe.index)

        if isinstance(idx, pd.MultiIndex) and not idx.is_unique:
            raise ValueError(
                "cannot convert a DataFrame with a non-unique MultiIndex into xarray"
            )

        # Cast to a NumPy array first, in case the Series is a pandas Extension
        # array (which doesn't have a valid NumPy dtype)
        # TODO: allow users to control how this casting happens, e.g., by
        # forwarding arguments to pandas.Series.to_numpy?
        arrays = [(k, np.asarray(v)) for k, v in dataframe.items()]

        obj = cls()

        if isinstance(idx, pd.MultiIndex):
            dims = tuple(
                name if name is not None else "level_%i" % n
                for n, name in enumerate(idx.names)
            )
            for dim, lev in zip(dims, idx.levels):
                obj[dim] = (dim, lev)
        else:
            index_name = idx.name if idx.name is not None else "index"
            dims = (index_name,)
            obj[index_name] = (dims, idx)

        if sparse:
            obj._set_sparse_data_from_dataframe(idx, arrays, dims)
        else:
            obj._set_numpy_data_from_dataframe(idx, arrays, dims)
        return obj

    def to_dask_dataframe(self, dim_order=None, set_index=False):
        """
        Convert this dataset into a dask.dataframe.DataFrame.

        The dimensions, coordinates and data variables in this dataset form
        the columns of the DataFrame.

        Parameters
        ----------
        dim_order : list, optional
            Hierarchical dimension order for the resulting dataframe. All
            arrays are transposed to this order and then written out as flat
            vectors in contiguous order, so the last dimension in this list
            will be contiguous in the resulting DataFrame. This has a major
            influence on which operations are efficient on the resulting dask
            dataframe.

            If provided, must include all dimensions of this dataset. By
            default, dimensions are sorted alphabetically.
        set_index : bool, optional
            If set_index=True, the dask DataFrame is indexed by this dataset's
            coordinate. Since dask DataFrames do not support multi-indexes,
            set_index only works if the dataset only contains one dimension.

        Returns
        -------
        dask.dataframe.DataFrame
        """

        import dask.array as da
        import dask.dataframe as dd

        ordered_dims = self._normalize_dim_order(dim_order=dim_order)

        columns = list(ordered_dims)
        columns.extend(k for k in self.coords if k not in self.dims)
        columns.extend(self.data_vars)

        series_list = []
        for name in columns:
            try:
                var = self.variables[name]
            except KeyError:
                # dimension without a matching coordinate
                size = self.dims[name]
                data = da.arange(size, chunks=size, dtype=np.int64)
                var = Variable((name,), data)

            # IndexVariable objects have a dummy .chunk() method
            if isinstance(var, IndexVariable):
                var = var.to_base_variable()

            dask_array = var.set_dims(ordered_dims).chunk(self.chunks).data
            series = dd.from_array(dask_array.reshape(-1), columns=[name])
            series_list.append(series)

        df = dd.concat(series_list, axis=1)

        if set_index:
            dim_order = [*ordered_dims]

            if len(dim_order) == 1:
                (dim,) = dim_order
                df = df.set_index(dim)
            else:
                # triggers an error about multi-indexes, even if only one
                # dimension is passed
                df = df.set_index(dim_order)

        return df

    def to_dict(self, data=True):
        """
        Convert this dataset to a dictionary following xarray naming
        conventions.

        Converts all variables and attributes to native Python objects
        Useful for converting to json. To avoid datetime incompatibility
        use decode_times=False kwarg in xarrray.open_dataset.

        Parameters
        ----------
        data : bool, optional
            Whether to include the actual data in the dictionary. When set to
            False, returns just the schema.

        See also
        --------
        Dataset.from_dict
        """
        d = {
            "coords": {},
            "attrs": decode_numpy_dict_values(self.attrs),
            "dims": dict(self.dims),
            "data_vars": {},
        }
        for k in self.coords:
            d["coords"].update({k: self[k].variable.to_dict(data=data)})
        for k in self.data_vars:
            d["data_vars"].update({k: self[k].variable.to_dict(data=data)})
        return d

    @classmethod
    def from_dict(cls, d):
        """
        Convert a dictionary into an xarray.Dataset.

        Input dict can take several forms:

        .. code:: python

            d = {
                "t": {"dims": ("t"), "data": t},
                "a": {"dims": ("t"), "data": x},
                "b": {"dims": ("t"), "data": y},
            }

            d = {
                "coords": {"t": {"dims": "t", "data": t, "attrs": {"units": "s"}}},
                "attrs": {"title": "air temperature"},
                "dims": "t",
                "data_vars": {
                    "a": {"dims": "t", "data": x},
                    "b": {"dims": "t", "data": y},
                },
            }

        where "t" is the name of the dimesion, "a" and "b" are names of data
        variables and t, x, and y are lists, numpy.arrays or pandas objects.

        Parameters
        ----------
        d : dict-like
            Mapping with a minimum structure of
                ``{"var_0": {"dims": [..], "data": [..]}, \
                            ...}``

        Returns
        -------
        obj : xarray.Dataset

        See also
        --------
        Dataset.to_dict
        DataArray.from_dict
        """

        if not {"coords", "data_vars"}.issubset(set(d)):
            variables = d.items()
        else:
            import itertools

            variables = itertools.chain(
                d.get("coords", {}).items(), d.get("data_vars", {}).items()
            )
        try:
            variable_dict = {
                k: (v["dims"], v["data"], v.get("attrs")) for k, v in variables
            }
        except KeyError as e:
            raise ValueError(
                "cannot convert dict without the key "
                "'{dims_data}'".format(dims_data=str(e.args[0]))
            )
        obj = cls(variable_dict)

        # what if coords aren't dims?
        coords = set(d.get("coords", {})) - set(d.get("dims", {}))
        obj = obj.set_coords(coords)

        obj.attrs.update(d.get("attrs", {}))

        return obj

    @staticmethod
    def _unary_op(f, keep_attrs=False):
        @functools.wraps(f)
        def func(self, *args, **kwargs):
            variables = {}
            for k, v in self._variables.items():
                if k in self._coord_names:
                    variables[k] = v
                else:
                    variables[k] = f(v, *args, **kwargs)
            attrs = self._attrs if keep_attrs else None
            return self._replace_with_new_dims(variables, attrs=attrs)

        return func

    @staticmethod
    def _binary_op(f, reflexive=False, join=None):
        @functools.wraps(f)
        def func(self, other):
            from .dataarray import DataArray

            if isinstance(other, groupby.GroupBy):
                return NotImplemented
            align_type = OPTIONS["arithmetic_join"] if join is None else join
            if isinstance(other, (DataArray, Dataset)):
                self, other = align(self, other, join=align_type, copy=False)
            g = f if not reflexive else lambda x, y: f(y, x)
            ds = self._calculate_binary_op(g, other, join=align_type)
            return ds

        return func

    @staticmethod
    def _inplace_binary_op(f):
        @functools.wraps(f)
        def func(self, other):
            from .dataarray import DataArray

            if isinstance(other, groupby.GroupBy):
                raise TypeError(
                    "in-place operations between a Dataset and "
                    "a grouped object are not permitted"
                )
            # we don't actually modify arrays in-place with in-place Dataset
            # arithmetic -- this lets us automatically align things
            if isinstance(other, (DataArray, Dataset)):
                other = other.reindex_like(self, copy=False)
            g = ops.inplace_to_noninplace_op(f)
            ds = self._calculate_binary_op(g, other, inplace=True)
            self._replace_with_new_dims(
                ds._variables,
                ds._coord_names,
                attrs=ds._attrs,
                indexes=ds._indexes,
                inplace=True,
            )
            return self

        return func

    def _calculate_binary_op(self, f, other, join="inner", inplace=False):
        def apply_over_both(lhs_data_vars, rhs_data_vars, lhs_vars, rhs_vars):
            if inplace and set(lhs_data_vars) != set(rhs_data_vars):
                raise ValueError(
                    "datasets must have the same data variables "
                    "for in-place arithmetic operations: %s, %s"
                    % (list(lhs_data_vars), list(rhs_data_vars))
                )

            dest_vars = {}

            for k in lhs_data_vars:
                if k in rhs_data_vars:
                    dest_vars[k] = f(lhs_vars[k], rhs_vars[k])
                elif join in ["left", "outer"]:
                    dest_vars[k] = f(lhs_vars[k], np.nan)
            for k in rhs_data_vars:
                if k not in dest_vars and join in ["right", "outer"]:
                    dest_vars[k] = f(rhs_vars[k], np.nan)
            return dest_vars

        if utils.is_dict_like(other) and not isinstance(other, Dataset):
            # can't use our shortcut of doing the binary operation with
            # Variable objects, so apply over our data vars instead.
            new_data_vars = apply_over_both(
                self.data_vars, other, self.data_vars, other
            )
            return Dataset(new_data_vars)

        other_coords = getattr(other, "coords", None)
        ds = self.coords.merge(other_coords)

        if isinstance(other, Dataset):
            new_vars = apply_over_both(
                self.data_vars, other.data_vars, self.variables, other.variables
            )
        else:
            other_variable = getattr(other, "variable", other)
            new_vars = {k: f(self.variables[k], other_variable) for k in self.data_vars}
        ds._variables.update(new_vars)
        ds._dims = calculate_dimensions(ds._variables)
        return ds

    def _copy_attrs_from(self, other):
        self.attrs = other.attrs
        for v in other.variables:
            if v in self.variables:
                self.variables[v].attrs = other.variables[v].attrs

    def diff(self, dim, n=1, label="upper"):
        """Calculate the n-th order discrete difference along given axis.

        Parameters
        ----------
        dim : str
            Dimension over which to calculate the finite difference.
        n : int, optional
            The number of times values are differenced.
        label : str, optional
            The new coordinate in dimension ``dim`` will have the
            values of either the minuend's or subtrahend's coordinate
            for values 'upper' and 'lower', respectively.  Other
            values are not supported.

        Returns
        -------
        difference : same type as caller
            The n-th order finite difference of this object.

        .. note::

            `n` matches numpy's behavior and is different from pandas' first
            argument named `periods`.

        Examples
        --------
        >>> ds = xr.Dataset({"foo": ("x", [5, 5, 6, 6])})
        >>> ds.diff("x")
        <xarray.Dataset>
        Dimensions:  (x: 3)
        Dimensions without coordinates: x
        Data variables:
            foo      (x) int64 0 1 0
        >>> ds.diff("x", 2)
        <xarray.Dataset>
        Dimensions:  (x: 2)
        Dimensions without coordinates: x
        Data variables:
            foo      (x) int64 1 -1

        See Also
        --------
        Dataset.differentiate
        """
        if n == 0:
            return self
        if n < 0:
            raise ValueError(f"order `n` must be non-negative but got {n}")

        # prepare slices
        kwargs_start = {dim: slice(None, -1)}
        kwargs_end = {dim: slice(1, None)}

        # prepare new coordinate
        if label == "upper":
            kwargs_new = kwargs_end
        elif label == "lower":
            kwargs_new = kwargs_start
        else:
            raise ValueError(
                "The 'label' argument has to be either " "'upper' or 'lower'"
            )

        variables = {}

        for name, var in self.variables.items():
            if dim in var.dims:
                if name in self.data_vars:
                    variables[name] = var.isel(**kwargs_end) - var.isel(**kwargs_start)
                else:
                    variables[name] = var.isel(**kwargs_new)
            else:
                variables[name] = var

        indexes = dict(self.indexes)
        if dim in indexes:
            indexes[dim] = indexes[dim][kwargs_new[dim]]

        difference = self._replace_with_new_dims(variables, indexes=indexes)

        if n > 1:
            return difference.diff(dim, n - 1)
        else:
            return difference

    def shift(self, shifts=None, fill_value=dtypes.NA, **shifts_kwargs):
        """Shift this dataset by an offset along one or more dimensions.

        Only data variables are moved; coordinates stay in place. This is
        consistent with the behavior of ``shift`` in pandas.

        Parameters
        ----------
        shifts : mapping of hashable to int
            Integer offset to shift along each of the given dimensions.
            Positive offsets shift to the right; negative offsets shift to the
            left.
        fill_value : scalar or dict-like, optional
            Value to use for newly missing values. If a dict-like, maps
            variable names (including coordinates) to fill values.
        **shifts_kwargs
            The keyword arguments form of ``shifts``.
            One of shifts or shifts_kwargs must be provided.

        Returns
        -------
        shifted : Dataset
            Dataset with the same coordinates and attributes but shifted data
            variables.

        See also
        --------
        roll

        Examples
        --------

        >>> ds = xr.Dataset({"foo": ("x", list("abcde"))})
        >>> ds.shift(x=2)
        <xarray.Dataset>
        Dimensions:  (x: 5)
        Dimensions without coordinates: x
        Data variables:
            foo      (x) object nan nan 'a' 'b' 'c'
        """
        shifts = either_dict_or_kwargs(shifts, shifts_kwargs, "shift")
        invalid = [k for k in shifts if k not in self.dims]
        if invalid:
            raise ValueError("dimensions %r do not exist" % invalid)

        variables = {}
        for name, var in self.variables.items():
            if name in self.data_vars:
                fill_value_ = (
                    fill_value.get(name, dtypes.NA)
                    if isinstance(fill_value, dict)
                    else fill_value
                )

                var_shifts = {k: v for k, v in shifts.items() if k in var.dims}
                variables[name] = var.shift(fill_value=fill_value_, shifts=var_shifts)
            else:
                variables[name] = var

        return self._replace(variables)
```
### 7 - xarray/core/combine.py:

Start line: 345, End line: 799

```python
def combine_nested(
    datasets,
    concat_dim,
    compat="no_conflicts",
    data_vars="all",
    coords="different",
    fill_value=dtypes.NA,
    join="outer",
    combine_attrs="drop",
):
    """
    Explicitly combine an N-dimensional grid of datasets into one by using a
    succession of concat and merge operations along each dimension of the grid.

    Does not sort the supplied datasets under any circumstances, so the
    datasets must be passed in the order you wish them to be concatenated. It
    does align coordinates, but different variables on datasets can cause it to
    fail under some scenarios. In complex cases, you may need to clean up your
    data and use concat/merge explicitly.

    To concatenate along multiple dimensions the datasets must be passed as a
    nested list-of-lists, with a depth equal to the length of ``concat_dims``.
    ``manual_combine`` will concatenate along the top-level list first.

    Useful for combining datasets from a set of nested directories, or for
    collecting the output of a simulation parallelized along multiple
    dimensions.

    Parameters
    ----------
    datasets : list or nested list of Dataset
        Dataset objects to combine.
        If concatenation or merging along more than one dimension is desired,
        then datasets must be supplied in a nested list-of-lists.
    concat_dim : str, or list of str, DataArray, Index or None
        Dimensions along which to concatenate variables, as used by
        :py:func:`xarray.concat`.
        Set ``concat_dim=[..., None, ...]`` explicitly to disable concatenation
        and merge instead along a particular dimension.
        The position of ``None`` in the list specifies the dimension of the
        nested-list input along which to merge.
        Must be the same length as the depth of the list passed to
        ``datasets``.
    compat : {"identical", "equals", "broadcast_equals", \
              "no_conflicts", "override"}, optional
        String indicating how to compare variables of the same name for
        potential merge conflicts:

        - "broadcast_equals": all values must be equal when variables are
          broadcast against each other to ensure common dimensions.
        - "equals": all values and dimensions must be the same.
        - "identical": all values, dimensions and attributes must be the
          same.
        - "no_conflicts": only values which are not null in both datasets
          must be equal. The returned dataset then contains the combination
          of all non-null values.
        - "override": skip comparing and pick variable from first dataset
    data_vars : {"minimal", "different", "all" or list of str}, optional
        Details are in the documentation of concat
    coords : {"minimal", "different", "all" or list of str}, optional
        Details are in the documentation of concat
    fill_value : scalar or dict-like, optional
        Value to use for newly missing values. If a dict-like, maps
        variable names to fill values. Use a data array's name to
        refer to its values.
    join : {"outer", "inner", "left", "right", "exact"}, optional
        String indicating how to combine differing indexes
        (excluding concat_dim) in objects

        - "outer": use the union of object indexes
        - "inner": use the intersection of object indexes
        - "left": use indexes from the first object with each dimension
        - "right": use indexes from the last object with each dimension
        - "exact": instead of aligning, raise `ValueError` when indexes to be
          aligned are not equal
        - "override": if indexes are of same size, rewrite indexes to be
          those of the first object with that dimension. Indexes for the same
          dimension must have the same size in all objects.
    combine_attrs : {"drop", "identical", "no_conflicts", "override"}, \
                    default: "drop"
        String indicating how to combine attrs of the objects being merged:

        - "drop": empty attrs on returned Dataset.
        - "identical": all attrs must be the same on every object.
        - "no_conflicts": attrs from all objects are combined, any that have
          the same name must also have the same value.
        - "override": skip comparing and copy attrs from the first dataset to
          the result.

    Returns
    -------
    combined : xarray.Dataset

    Examples
    --------

    A common task is collecting data from a parallelized simulation in which
    each process wrote out to a separate file. A domain which was decomposed
    into 4 parts, 2 each along both the x and y axes, requires organising the
    datasets into a doubly-nested list, e.g:

    >>> x1y1 = xr.Dataset(
    ...     {
    ...         "temperature": (("x", "y"), np.random.randn(2, 2)),
    ...         "precipitation": (("x", "y"), np.random.randn(2, 2)),
    ...     }
    ... )
    >>> x1y1
    <xarray.Dataset>
    Dimensions:        (x: 2, y: 2)
    Dimensions without coordinates: x, y
    Data variables:
        temperature    (x, y) float64 1.764 0.4002 0.9787 2.241
        precipitation  (x, y) float64 1.868 -0.9773 0.9501 -0.1514
    >>> x1y2 = xr.Dataset(
    ...     {
    ...         "temperature": (("x", "y"), np.random.randn(2, 2)),
    ...         "precipitation": (("x", "y"), np.random.randn(2, 2)),
    ...     }
    ... )
    >>> x2y1 = xr.Dataset(
    ...     {
    ...         "temperature": (("x", "y"), np.random.randn(2, 2)),
    ...         "precipitation": (("x", "y"), np.random.randn(2, 2)),
    ...     }
    ... )
    >>> x2y2 = xr.Dataset(
    ...     {
    ...         "temperature": (("x", "y"), np.random.randn(2, 2)),
    ...         "precipitation": (("x", "y"), np.random.randn(2, 2)),
    ...     }
    ... )


    >>> ds_grid = [[x1y1, x1y2], [x2y1, x2y2]]
    >>> combined = xr.combine_nested(ds_grid, concat_dim=["x", "y"])
    >>> combined
    <xarray.Dataset>
    Dimensions:        (x: 4, y: 4)
    Dimensions without coordinates: x, y
    Data variables:
        temperature    (x, y) float64 1.764 0.4002 -0.1032 ... 0.04576 -0.1872
        precipitation  (x, y) float64 1.868 -0.9773 0.761 ... -0.7422 0.1549 0.3782

    ``manual_combine`` can also be used to explicitly merge datasets with
    different variables. For example if we have 4 datasets, which are divided
    along two times, and contain two different variables, we can pass ``None``
    to ``concat_dim`` to specify the dimension of the nested list over which
    we wish to use ``merge`` instead of ``concat``:

    >>> t1temp = xr.Dataset({"temperature": ("t", np.random.randn(5))})
    >>> t1temp
    <xarray.Dataset>
    Dimensions:      (t: 5)
    Dimensions without coordinates: t
    Data variables:
        temperature  (t) float64 -0.8878 -1.981 -0.3479 0.1563 1.23

    >>> t1precip = xr.Dataset({"precipitation": ("t", np.random.randn(5))})
    >>> t1precip
    <xarray.Dataset>
    Dimensions:        (t: 5)
    Dimensions without coordinates: t
    Data variables:
        precipitation  (t) float64 1.202 -0.3873 -0.3023 -1.049 -1.42

    >>> t2temp = xr.Dataset({"temperature": ("t", np.random.randn(5))})
    >>> t2precip = xr.Dataset({"precipitation": ("t", np.random.randn(5))})


    >>> ds_grid = [[t1temp, t1precip], [t2temp, t2precip]]
    >>> combined = xr.combine_nested(ds_grid, concat_dim=["t", None])
    >>> combined
    <xarray.Dataset>
    Dimensions:        (t: 10)
    Dimensions without coordinates: t
    Data variables:
        temperature    (t) float64 -0.8878 -1.981 -0.3479 ... -0.5097 -0.4381 -1.253
        precipitation  (t) float64 1.202 -0.3873 -0.3023 ... -0.2127 -0.8955 0.3869

    See also
    --------
    concat
    merge
    auto_combine
    """
    if isinstance(concat_dim, (str, DataArray)) or concat_dim is None:
        concat_dim = [concat_dim]

    # The IDs argument tells _manual_combine that datasets aren't yet sorted
    return _nested_combine(
        datasets,
        concat_dims=concat_dim,
        compat=compat,
        data_vars=data_vars,
        coords=coords,
        ids=False,
        fill_value=fill_value,
        join=join,
        combine_attrs=combine_attrs,
    )


def vars_as_keys(ds):
    return tuple(sorted(ds))


def combine_by_coords(
    datasets,
    compat="no_conflicts",
    data_vars="all",
    coords="different",
    fill_value=dtypes.NA,
    join="outer",
    combine_attrs="no_conflicts",
):
    # ... other code
```
### 8 - asv_bench/benchmarks/indexing.py:

Start line: 1, End line: 57

```python
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
### 9 - xarray/core/concat.py:

Start line: 165, End line: 193

```python
def concat(
    objs,
    dim,
    data_vars="all",
    coords="different",
    compat="equals",
    positions=None,
    fill_value=dtypes.NA,
    join="outer",
    combine_attrs="override",
):
    # TODO: add ignore_index arguments copied from pandas.concat
    # TODO: support concatenating scalar coordinates even if the concatenated
    # dimension already exists
    from .dataarray import DataArray
    from .dataset import Dataset

    try:
        first_obj, objs = utils.peek_at(objs)
    except StopIteration:
        raise ValueError("must supply at least one object to concatenate")

    if compat not in _VALID_COMPAT:
        raise ValueError(
            "compat=%r invalid: must be 'broadcast_equals', 'equals', 'identical', 'no_conflicts' or 'override'"
            % compat
        )

    if isinstance(first_obj, DataArray):
        f = _dataarray_concat
    elif isinstance(first_obj, Dataset):
        f = _dataset_concat
    else:
        raise TypeError(
            "can only concatenate xarray Dataset and DataArray "
            "objects, got %s" % type(first_obj)
        )
    return f(
        objs, dim, data_vars, coords, compat, positions, fill_value, join, combine_attrs
    )
```
### 10 - xarray/core/concat.py:

Start line: 336, End line: 361

```python
# determine dimensional coordinate names and a dict mapping name to DataArray
def _parse_datasets(
    datasets: Iterable["Dataset"],
) -> Tuple[Dict[Hashable, Variable], Dict[Hashable, int], Set[Hashable], Set[Hashable]]:

    dims: Set[Hashable] = set()
    all_coord_names: Set[Hashable] = set()
    data_vars: Set[Hashable] = set()  # list of data_vars
    dim_coords: Dict[Hashable, Variable] = {}  # maps dim name to variable
    dims_sizes: Dict[Hashable, int] = {}  # shared dimension sizes to expand variables

    for ds in datasets:
        dims_sizes.update(ds.dims)
        all_coord_names.update(ds.coords)
        data_vars.update(ds.data_vars)

        # preserves ordering of dimensions
        for dim in ds.dims:
            if dim in dims:
                continue

            if dim not in dim_coords:
                dim_coords[dim] = ds.coords[dim].variable
        dims = dims | set(ds.dims)

    return dim_coords, dims_sizes, all_coord_names, data_vars
```
### 13 - xarray/core/concat.py:

Start line: 243, End line: 333

```python
def _calc_concat_over(datasets, dim, dim_names, data_vars, coords, compat):
    # ... other code

    def process_subset_opt(opt, subset):
        if isinstance(opt, str):
            if opt == "different":
                if compat == "override":
                    raise ValueError(
                        "Cannot specify both %s='different' and compat='override'."
                        % subset
                    )
                # all nonindexes that are not the same in each dataset
                for k in getattr(datasets[0], subset):
                    if k not in concat_over:
                        equals[k] = None

                        variables = []
                        for ds in datasets:
                            if k in ds.variables:
                                variables.append(ds.variables[k])

                        if len(variables) == 1:
                            # coords="different" doesn't make sense when only one object
                            # contains a particular variable.
                            break
                        elif len(variables) != len(datasets) and opt == "different":
                            raise ValueError(
                                f"{k!r} not present in all datasets and coords='different'. "
                                f"Either add {k!r} to datasets where it is missing or "
                                "specify coords='minimal'."
                            )

                        # first check without comparing values i.e. no computes
                        for var in variables[1:]:
                            equals[k] = getattr(variables[0], compat)(
                                var, equiv=lazy_array_equiv
                            )
                            if equals[k] is not True:
                                # exit early if we know these are not equal or that
                                # equality cannot be determined i.e. one or all of
                                # the variables wraps a numpy array
                                break

                        if equals[k] is False:
                            concat_over.add(k)

                        elif equals[k] is None:
                            # Compare the variable of all datasets vs. the one
                            # of the first dataset. Perform the minimum amount of
                            # loads in order to avoid multiple loads from disk
                            # while keeping the RAM footprint low.
                            v_lhs = datasets[0].variables[k].load()
                            # We'll need to know later on if variables are equal.
                            computed = []
                            for ds_rhs in datasets[1:]:
                                v_rhs = ds_rhs.variables[k].compute()
                                computed.append(v_rhs)
                                if not getattr(v_lhs, compat)(v_rhs):
                                    concat_over.add(k)
                                    equals[k] = False
                                    # computed variables are not to be re-computed
                                    # again in the future
                                    for ds, v in zip(datasets[1:], computed):
                                        ds.variables[k].data = v.data
                                    break
                            else:
                                equals[k] = True

            elif opt == "all":
                concat_over.update(
                    set(getattr(datasets[0], subset)) - set(datasets[0].dims)
                )
            elif opt == "minimal":
                pass
            else:
                raise ValueError(f"unexpected value for {subset}: {opt}")
        else:
            invalid_vars = [k for k in opt if k not in getattr(datasets[0], subset)]
            if invalid_vars:
                if subset == "coords":
                    raise ValueError(
                        "some variables in coords are not coordinates on "
                        "the first dataset: %s" % (invalid_vars,)
                    )
                else:
                    raise ValueError(
                        "some variables in data_vars are not data variables "
                        "on the first dataset: %s" % (invalid_vars,)
                    )
            concat_over.update(opt)

    process_subset_opt(data_vars, "data_vars")
    process_subset_opt(coords, "coords")
    return concat_over, equals, concat_dim_lengths
```
### 14 - xarray/core/concat.py:

Start line: 364, End line: 441

```python
def _dataset_concat(
    datasets: List["Dataset"],
    dim: Union[str, "DataArray", pd.Index],
    data_vars: Union[str, List[str]],
    coords: Union[str, List[str]],
    compat: str,
    positions: Optional[Iterable[int]],
    fill_value: object = dtypes.NA,
    join: str = "outer",
    combine_attrs: str = "override",
) -> "Dataset":
    """
    Concatenate a sequence of datasets along a new or existing dimension
    """
    from .dataset import Dataset

    dim, coord = _calc_concat_dim_coord(dim)
    # Make sure we're working on a copy (we'll be loading variables)
    datasets = [ds.copy() for ds in datasets]
    datasets = align(
        *datasets, join=join, copy=False, exclude=[dim], fill_value=fill_value
    )

    dim_coords, dims_sizes, coord_names, data_names = _parse_datasets(datasets)
    dim_names = set(dim_coords)
    unlabeled_dims = dim_names - coord_names

    both_data_and_coords = coord_names & data_names
    if both_data_and_coords:
        raise ValueError(
            "%r is a coordinate in some datasets but not others." % both_data_and_coords
        )
    # we don't want the concat dimension in the result dataset yet
    dim_coords.pop(dim, None)
    dims_sizes.pop(dim, None)

    # case where concat dimension is a coordinate or data_var but not a dimension
    if (dim in coord_names or dim in data_names) and dim not in dim_names:
        datasets = [ds.expand_dims(dim) for ds in datasets]

    # determine which variables to concatentate
    concat_over, equals, concat_dim_lengths = _calc_concat_over(
        datasets, dim, dim_names, data_vars, coords, compat
    )

    # determine which variables to merge, and then merge them according to compat
    variables_to_merge = (coord_names | data_names) - concat_over - dim_names

    result_vars = {}
    if variables_to_merge:
        to_merge: Dict[Hashable, List[Variable]] = {
            var: [] for var in variables_to_merge
        }

        for ds in datasets:
            for var in variables_to_merge:
                if var in ds:
                    to_merge[var].append(ds.variables[var])

        for var in variables_to_merge:
            result_vars[var] = unique_variable(
                var, to_merge[var], compat=compat, equals=equals.get(var, None)
            )
    else:
        result_vars = {}
    result_vars.update(dim_coords)

    # assign attrs and encoding from first dataset
    result_attrs = merge_attrs([ds.attrs for ds in datasets], combine_attrs)
    result_encoding = datasets[0].encoding

    # check that global attributes are fixed across all datasets if necessary
    for ds in datasets[1:]:
        if compat == "identical" and not utils.dict_equiv(ds.attrs, result_attrs):
            raise ValueError("Dataset global attributes not equal.")

    # we've already verified everything is consistent; now, calculate
    # shared dimension sizes so we can expand the necessary variables
    # ... other code
```
### 17 - xarray/core/concat.py:

Start line: 442, End line: 453

```python
def _dataset_concat(
    datasets: List["Dataset"],
    dim: Union[str, "DataArray", pd.Index],
    data_vars: Union[str, List[str]],
    coords: Union[str, List[str]],
    compat: str,
    positions: Optional[Iterable[int]],
    fill_value: object = dtypes.NA,
    join: str = "outer",
    combine_attrs: str = "override",
) -> "Dataset":
    # ... other code
    def ensure_common_dims(vars):
        # ensure each variable with the given name shares the same
        # dimensions and the same shape for all of them except along the
        # concat dimension
        common_dims = tuple(pd.unique([d for v in vars for d in v.dims]))
        if dim not in common_dims:
            common_dims = (dim,) + common_dims
        for var, dim_len in zip(vars, concat_dim_lengths):
            if var.dims != common_dims:
                common_shape = tuple(dims_sizes.get(d, dim_len) for d in common_dims)
                var = var.set_dims(common_dims, common_shape)
            yield var
    # ... other code
```
### 20 - xarray/core/concat.py:

Start line: 455, End line: 483

```python
def _dataset_concat(
    datasets: List["Dataset"],
    dim: Union[str, "DataArray", pd.Index],
    data_vars: Union[str, List[str]],
    coords: Union[str, List[str]],
    compat: str,
    positions: Optional[Iterable[int]],
    fill_value: object = dtypes.NA,
    join: str = "outer",
    combine_attrs: str = "override",
) -> "Dataset":

    # stack up each variable to fill-out the dataset (in order)
    # n.b. this loop preserves variable order, needed for groupby.
    for k in datasets[0].variables:
        if k in concat_over:
            try:
                vars = ensure_common_dims([ds.variables[k] for ds in datasets])
            except KeyError:
                raise ValueError("%r is not present in all datasets." % k)
            combined = concat_vars(vars, dim, positions)
            assert isinstance(combined, Variable)
            result_vars[k] = combined

    result = Dataset(result_vars, attrs=result_attrs)
    absent_coord_names = coord_names - set(result.variables)
    if absent_coord_names:
        raise ValueError(
            "Variables %r are coordinates in some datasets but not others."
            % absent_coord_names
        )
    result = result.set_coords(coord_names)
    result.encoding = result_encoding

    result = result.drop_vars(unlabeled_dims, errors="ignore")

    if coord is not None:
        # add concat dimension last to ensure that its in the final Dataset
        result[coord.name] = coord

    return result
```
### 21 - xarray/core/concat.py:

Start line: 28, End line: 40

```python
@overload
def concat(
    objs: Iterable["Dataset"],
    dim: Union[str, "DataArray", pd.Index],
    data_vars: Union[str, List[str]] = "all",
    coords: Union[str, List[str]] = "different",
    compat: str = "equals",
    positions: Optional[Iterable[int]] = None,
    fill_value: object = dtypes.NA,
    join: str = "outer",
    combine_attrs: str = "override",
) -> "Dataset":
    ...
```
### 26 - xarray/core/concat.py:

Start line: 1, End line: 25

```python
from typing import (
    TYPE_CHECKING,
    Dict,
    Hashable,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Union,
    overload,
)

import pandas as pd

from . import dtypes, utils
from .alignment import align
from .duck_array_ops import lazy_array_equiv
from .merge import _VALID_COMPAT, merge_attrs, unique_variable
from .variable import IndexVariable, Variable, as_variable
from .variable import concat as concat_vars

if TYPE_CHECKING:
    from .dataarray import DataArray
    from .dataset import Dataset
```
### 30 - xarray/core/concat.py:

Start line: 486, End line: 533

```python
def _dataarray_concat(
    arrays: Iterable["DataArray"],
    dim: Union[str, "DataArray", pd.Index],
    data_vars: Union[str, List[str]],
    coords: Union[str, List[str]],
    compat: str,
    positions: Optional[Iterable[int]],
    fill_value: object = dtypes.NA,
    join: str = "outer",
    combine_attrs: str = "override",
) -> "DataArray":
    arrays = list(arrays)

    if data_vars != "all":
        raise ValueError(
            "data_vars is not a valid argument when concatenating DataArray objects"
        )

    datasets = []
    for n, arr in enumerate(arrays):
        if n == 0:
            name = arr.name
        elif name != arr.name:
            if compat == "identical":
                raise ValueError("array names not identical")
            else:
                arr = arr.rename(name)
        datasets.append(arr._to_temp_dataset())

    ds = _dataset_concat(
        datasets,
        dim,
        data_vars,
        coords,
        compat,
        positions,
        fill_value=fill_value,
        join=join,
        combine_attrs="drop",
    )

    merged_attrs = merge_attrs([da.attrs for da in arrays], combine_attrs)

    result = arrays[0]._from_temp_dataset(ds, name)
    result.attrs = merged_attrs

    return result
```
### 32 - xarray/core/concat.py:

Start line: 43, End line: 55

```python
@overload
def concat(
    objs: Iterable["DataArray"],
    dim: Union[str, "DataArray", pd.Index],
    data_vars: Union[str, List[str]] = "all",
    coords: Union[str, List[str]] = "different",
    compat: str = "equals",
    positions: Optional[Iterable[int]] = None,
    fill_value: object = dtypes.NA,
    join: str = "outer",
    combine_attrs: str = "override",
) -> "DataArray":
    ...
```
### 35 - xarray/core/concat.py:

Start line: 220, End line: 241

```python
def _calc_concat_over(datasets, dim, dim_names, data_vars, coords, compat):
    """
    Determine which dataset variables need to be concatenated in the result,
    """
    # Return values
    concat_over = set()
    equals = {}

    if dim in dim_names:
        concat_over_existing_dim = True
        concat_over.add(dim)
    else:
        concat_over_existing_dim = False

    concat_dim_lengths = []
    for ds in datasets:
        if concat_over_existing_dim:
            if dim not in ds.dims:
                if dim in ds:
                    ds = ds.set_coords(dim)
        concat_over.update(k for k, v in ds.variables.items() if dim in v.dims)
        concat_dim_lengths.append(ds.dims.get(dim, 1))
    # ... other code
```
### 46 - xarray/core/concat.py:

Start line: 58, End line: 164

```python
def concat(
    objs,
    dim,
    data_vars="all",
    coords="different",
    compat="equals",
    positions=None,
    fill_value=dtypes.NA,
    join="outer",
    combine_attrs="override",
):
    """Concatenate xarray objects along a new or existing dimension.

    Parameters
    ----------
    objs : sequence of Dataset and DataArray
        xarray objects to concatenate together. Each object is expected to
        consist of variables and coordinates with matching shapes except for
        along the concatenated dimension.
    dim : str or DataArray or pandas.Index
        Name of the dimension to concatenate along. This can either be a new
        dimension name, in which case it is added along axis=0, or an existing
        dimension name, in which case the location of the dimension is
        unchanged. If dimension is provided as a DataArray or Index, its name
        is used as the dimension to concatenate along and the values are added
        as a coordinate.
    data_vars : {"minimal", "different", "all"} or list of str, optional
        These data variables will be concatenated together:
          * "minimal": Only data variables in which the dimension already
            appears are included.
          * "different": Data variables which are not equal (ignoring
            attributes) across all datasets are also concatenated (as well as
            all for which dimension already appears). Beware: this option may
            load the data payload of data variables into memory if they are not
            already loaded.
          * "all": All data variables will be concatenated.
          * list of str: The listed data variables will be concatenated, in
            addition to the "minimal" data variables.

        If objects are DataArrays, data_vars must be "all".
    coords : {"minimal", "different", "all"} or list of str, optional
        These coordinate variables will be concatenated together:
          * "minimal": Only coordinates in which the dimension already appears
            are included.
          * "different": Coordinates which are not equal (ignoring attributes)
            across all datasets are also concatenated (as well as all for which
            dimension already appears). Beware: this option may load the data
            payload of coordinate variables into memory if they are not already
            loaded.
          * "all": All coordinate variables will be concatenated, except
            those corresponding to other dimensions.
          * list of str: The listed coordinate variables will be concatenated,
            in addition to the "minimal" coordinates.
    compat : {"identical", "equals", "broadcast_equals", "no_conflicts", "override"}, optional
        String indicating how to compare non-concatenated variables of the same name for
        potential conflicts. This is passed down to merge.

        - "broadcast_equals": all values must be equal when variables are
          broadcast against each other to ensure common dimensions.
        - "equals": all values and dimensions must be the same.
        - "identical": all values, dimensions and attributes must be the
          same.
        - "no_conflicts": only values which are not null in both datasets
          must be equal. The returned dataset then contains the combination
          of all non-null values.
        - "override": skip comparing and pick variable from first dataset
    positions : None or list of integer arrays, optional
        List of integer arrays which specifies the integer positions to which
        to assign each dataset along the concatenated dimension. If not
        supplied, objects are concatenated in the provided order.
    fill_value : scalar or dict-like, optional
        Value to use for newly missing values. If a dict-like, maps
        variable names to fill values. Use a data array's name to
        refer to its values.
    join : {"outer", "inner", "left", "right", "exact"}, optional
        String indicating how to combine differing indexes
        (excluding dim) in objects

        - "outer": use the union of object indexes
        - "inner": use the intersection of object indexes
        - "left": use indexes from the first object with each dimension
        - "right": use indexes from the last object with each dimension
        - "exact": instead of aligning, raise `ValueError` when indexes to be
          aligned are not equal
        - "override": if indexes are of same size, rewrite indexes to be
          those of the first object with that dimension. Indexes for the same
          dimension must have the same size in all objects.
    combine_attrs : {"drop", "identical", "no_conflicts", "override"}, \
                    default: "override"
        String indicating how to combine attrs of the objects being merged:

        - "drop": empty attrs on returned Dataset.
        - "identical": all attrs must be the same on every object.
        - "no_conflicts": attrs from all objects are combined, any that have
          the same name must also have the same value.
        - "override": skip comparing and copy attrs from the first dataset to
          the result.

    Returns
    -------
    concatenated : type of objs

    See also
    --------
    merge
    auto_combine
    """
    # ... other code
```
### 51 - xarray/core/concat.py:

Start line: 196, End line: 217

```python
def _calc_concat_dim_coord(dim):
    """
    Infer the dimension name and 1d coordinate variable (if appropriate)
    for concatenating along the new dimension.
    """
    from .dataarray import DataArray

    if isinstance(dim, str):
        coord = None
    elif not isinstance(dim, (DataArray, Variable)):
        dim_name = getattr(dim, "name", None)
        if dim_name is None:
            dim_name = "concat_dim"
        coord = IndexVariable(dim_name, dim)
        dim = dim_name
    elif not isinstance(dim, DataArray):
        coord = as_variable(dim).to_index_variable()
        (dim,) = coord.dims
    else:
        coord = dim
        (dim,) = coord.dims
    return dim, coord
```
