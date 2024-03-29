# pydata__xarray-5131

| **pydata/xarray** | `e56905889c836c736152b11a7e6117a229715975` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | - |
| **Avg pos** | - |
| **Min pos** | - |
| **Max pos** | - |
| **Top file pos** | 5 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/xarray/core/groupby.py b/xarray/core/groupby.py
--- a/xarray/core/groupby.py
+++ b/xarray/core/groupby.py
@@ -436,7 +436,7 @@ def __iter__(self):
         return zip(self._unique_coord.values, self._iter_grouped())
 
     def __repr__(self):
-        return "{}, grouped over {!r} \n{!r} groups with labels {}.".format(
+        return "{}, grouped over {!r}\n{!r} groups with labels {}.".format(
             self.__class__.__name__,
             self._unique_coord.name,
             self._unique_coord.size,

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| xarray/core/groupby.py | 439 | 439 | - | 5 | -


## Problem Statement

```
Trailing whitespace in DatasetGroupBy text representation
When displaying a DatasetGroupBy in an interactive Python session, the first line of output contains a trailing whitespace. The first example in the documentation demonstrate this:

\`\`\`pycon
>>> import xarray as xr, numpy as np
>>> ds = xr.Dataset(
...     {"foo": (("x", "y"), np.random.rand(4, 3))},
...     coords={"x": [10, 20, 30, 40], "letters": ("x", list("abba"))},
... )
>>> ds.groupby("letters")
DatasetGroupBy, grouped over 'letters' 
2 groups with labels 'a', 'b'.
\`\`\`

There is a trailing whitespace in the first line of output which is "DatasetGroupBy, grouped over 'letters' ". This can be seen more clearly by converting the object to a string (note the whitespace before `\n`):

\`\`\`pycon
>>> str(ds.groupby("letters"))
"DatasetGroupBy, grouped over 'letters' \n2 groups with labels 'a', 'b'."
\`\`\`


While this isn't a problem in itself, it causes an issue for us because we use flake8 in continuous integration to verify that our code is correctly formatted and we also have doctests that rely on DatasetGroupBy textual representation. Flake8 reports a violation on the trailing whitespaces in our docstrings. If we remove the trailing whitespaces, our doctests fail because the expected output doesn't match the actual output. So we have conflicting constraints coming from our tools which both seem reasonable. Trailing whitespaces are forbidden by flake8 because, among other reasons, they lead to noisy git diffs. Doctest want the expected output to be exactly the same as the actual output and considers a trailing whitespace to be a significant difference. We could configure flake8 to ignore this particular violation for the files in which we have these doctests, but this may cause other trailing whitespaces to creep in our code, which we don't want. Unfortunately it's not possible to just add `# NoQA` comments to get flake8 to ignore the violation only for specific lines because that creates a difference between expected and actual output from doctest point of view. Flake8 doesn't allow to disable checks for blocks of code either.

Is there a reason for having this trailing whitespace in DatasetGroupBy representation? Whould it be OK to remove it? If so please let me know and I can make a pull request.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 xarray/core/formatting.py | 515 | 544| 206 | 206 | 5477 | 
| 2 | 2 xarray/core/dataset.py | 358 | 383| 191 | 397 | 67172 | 
| 3 | 2 xarray/core/formatting.py | 398 | 455| 387 | 784 | 67172 | 
| 4 | 3 xarray/core/accessor_str.py | 1295 | 1319| 188 | 972 | 87049 | 
| 5 | 4 doc/conf.py | 1 | 101| 670 | 1642 | 90379 | 
| 6 | 4 xarray/core/dataset.py | 419 | 444| 207 | 1849 | 90379 | 
| 7 | 4 xarray/core/dataset.py | 5788 | 6393| 5685 | 7534 | 90379 | 
| 8 | **5 xarray/core/groupby.py** | 488 | 512| 186 | 7720 | 98309 | 
| 9 | 5 doc/conf.py | 260 | 309| 586 | 8306 | 98309 | 
| 10 | 5 xarray/core/formatting.py | 671 | 692| 158 | 8464 | 98309 | 
| 11 | 6 xarray/core/rolling.py | 665 | 687| 146 | 8610 | 105850 | 
| 12 | 6 xarray/core/accessor_str.py | 1228 | 1267| 322 | 8932 | 105850 | 
| 13 | 6 xarray/core/dataset.py | 1395 | 2094| 5833 | 14765 | 105850 | 
| 14 | 6 xarray/core/dataset.py | 7260 | 7314| 563 | 15328 | 105850 | 
| 15 | 7 xarray/conventions.py | 1 | 29| 168 | 15496 | 112251 | 
| 16 | 7 xarray/core/rolling.py | 619 | 635| 161 | 15657 | 112251 | 
| 17 | 7 xarray/core/formatting.py | 219 | 245| 203 | 15860 | 112251 | 
| 18 | **7 xarray/core/groupby.py** | 185 | 199| 174 | 16034 | 112251 | 
| 19 | 7 xarray/core/accessor_str.py | 1269 | 1293| 188 | 16222 | 112251 | 
| 20 | **7 xarray/core/groupby.py** | 933 | 960| 240 | 16462 | 112251 | 
| 21 | 8 xarray/core/coordinates.py | 291 | 303| 116 | 16578 | 115388 | 
| 22 | 9 xarray/core/formatting_html.py | 253 | 289| 257 | 16835 | 117580 | 
| 23 | 9 xarray/core/dataset.py | 4217 | 4887| 6027 | 22862 | 117580 | 
| 24 | **9 xarray/core/groupby.py** | 1 | 36| 229 | 23091 | 117580 | 
| 25 | 10 xarray/testing.py | 177 | 201| 175 | 23266 | 120674 | 
| 26 | 11 xarray/core/ops.py | 108 | 136| 269 | 23535 | 123327 | 
| 27 | 12 xarray/backends/common.py | 1 | 39| 225 | 23760 | 125802 | 
| 28 | **12 xarray/core/groupby.py** | 894 | 931| 319 | 24079 | 125802 | 
| 29 | 13 xarray/plot/dataset_plot.py | 465 | 517| 437 | 24516 | 130488 | 
| 30 | 13 xarray/core/dataset.py | 1 | 137| 658 | 25174 | 130488 | 
| 31 | 13 xarray/core/dataset.py | 2096 | 2744| 6101 | 31275 | 130488 | 
| 32 | 14 xarray/core/concat.py | 490 | 501| 239 | 31514 | 135366 | 
| 33 | 14 xarray/plot/dataset_plot.py | 277 | 389| 828 | 32342 | 135366 | 
| 34 | 15 xarray/core/merge.py | 654 | 863| 2773 | 35115 | 143444 | 
| 35 | 15 xarray/plot/dataset_plot.py | 192 | 275| 977 | 36092 | 143444 | 
| 36 | 16 xarray/core/dataarray.py | 1 | 87| 442 | 36534 | 182321 | 
| 37 | 17 xarray/core/common.py | 1 | 46| 249 | 36783 | 198023 | 


### Hint

```
I don't think this is intentional and we are happy to take a PR. The problem seems to be here:

https://github.com/pydata/xarray/blob/c7c4aae1fa2bcb9417e498e7dcb4acc0792c402d/xarray/core/groupby.py#L439

You will also have to fix the tests (maybe other places):

https://github.com/pydata/xarray/blob/c7c4aae1fa2bcb9417e498e7dcb4acc0792c402d/xarray/tests/test_groupby.py#L391
https://github.com/pydata/xarray/blob/c7c4aae1fa2bcb9417e498e7dcb4acc0792c402d/xarray/tests/test_groupby.py#L408

```

## Patch

```diff
diff --git a/xarray/core/groupby.py b/xarray/core/groupby.py
--- a/xarray/core/groupby.py
+++ b/xarray/core/groupby.py
@@ -436,7 +436,7 @@ def __iter__(self):
         return zip(self._unique_coord.values, self._iter_grouped())
 
     def __repr__(self):
-        return "{}, grouped over {!r} \n{!r} groups with labels {}.".format(
+        return "{}, grouped over {!r}\n{!r} groups with labels {}.".format(
             self.__class__.__name__,
             self._unique_coord.name,
             self._unique_coord.size,

```

## Test Patch

```diff
diff --git a/xarray/tests/test_groupby.py b/xarray/tests/test_groupby.py
--- a/xarray/tests/test_groupby.py
+++ b/xarray/tests/test_groupby.py
@@ -388,7 +388,7 @@ def test_da_groupby_assign_coords():
 def test_groupby_repr(obj, dim):
     actual = repr(obj.groupby(dim))
     expected = "%sGroupBy" % obj.__class__.__name__
-    expected += ", grouped over %r " % dim
+    expected += ", grouped over %r" % dim
     expected += "\n%r groups with labels " % (len(np.unique(obj[dim])))
     if dim == "x":
         expected += "1, 2, 3, 4, 5."
@@ -405,7 +405,7 @@ def test_groupby_repr(obj, dim):
 def test_groupby_repr_datetime(obj):
     actual = repr(obj.groupby("t.month"))
     expected = "%sGroupBy" % obj.__class__.__name__
-    expected += ", grouped over 'month' "
+    expected += ", grouped over 'month'"
     expected += "\n%r groups with labels " % (len(np.unique(obj.t.dt.month)))
     expected += "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12."
     assert actual == expected

```


## Code snippets

### 1 - xarray/core/formatting.py:

Start line: 515, End line: 544

```python
def dataset_repr(ds):
    summary = ["<xarray.{}>".format(type(ds).__name__)]

    col_width = _calculate_col_width(_get_col_items(ds.variables))

    dims_start = pretty_print("Dimensions:", col_width)
    summary.append("{}({})".format(dims_start, dim_summary(ds)))

    if ds.coords:
        summary.append(coords_repr(ds.coords, col_width=col_width))

    unindexed_dims_str = unindexed_dims_repr(ds.dims, ds.coords)
    if unindexed_dims_str:
        summary.append(unindexed_dims_str)

    summary.append(data_vars_repr(ds.data_vars, col_width=col_width))

    if ds.attrs:
        summary.append(attrs_repr(ds.attrs))

    return "\n".join(summary)


def diff_dim_summary(a, b):
    if a.dims != b.dims:
        return "Differing dimensions:\n    ({}) != ({})".format(
            dim_summary(a), dim_summary(b)
        )
    else:
        return ""
```
### 2 - xarray/core/dataset.py:

Start line: 358, End line: 383

```python
def _assert_empty(args: tuple, msg: str = "%s") -> None:
    if args:
        raise ValueError(msg % args)


def _check_chunks_compatibility(var, chunks, preferred_chunks):
    for dim in var.dims:
        if dim not in chunks or (dim not in preferred_chunks):
            continue

        preferred_chunks_dim = preferred_chunks.get(dim)
        chunks_dim = chunks.get(dim)

        if isinstance(chunks_dim, int):
            chunks_dim = (chunks_dim,)
        else:
            chunks_dim = chunks_dim[:-1]

        if any(s % preferred_chunks_dim for s in chunks_dim):
            warnings.warn(
                f"Specified Dask chunks {chunks[dim]} would separate "
                f"on disks chunk shape {preferred_chunks[dim]} for dimension {dim}. "
                "This could degrade performance. "
                "Consider rechunking after loading instead.",
                stacklevel=2,
            )
```
### 3 - xarray/core/formatting.py:

Start line: 398, End line: 455

```python
data_vars_repr = functools.partial(
    _mapping_repr, title="Data variables", summarizer=summarize_datavar
)


attrs_repr = functools.partial(
    _mapping_repr, title="Attributes", summarizer=summarize_attr
)


def coords_repr(coords, col_width=None):
    if col_width is None:
        col_width = _calculate_col_width(_get_col_items(coords))
    return _mapping_repr(
        coords, title="Coordinates", summarizer=summarize_coord, col_width=col_width
    )


def indexes_repr(indexes):
    summary = []
    for k, v in indexes.items():
        summary.append(wrap_indent(repr(v), f"{k}: "))
    return "\n".join(summary)


def dim_summary(obj):
    elements = [f"{k}: {v}" for k, v in obj.sizes.items()]
    return ", ".join(elements)


def unindexed_dims_repr(dims, coords):
    unindexed_dims = [d for d in dims if d not in coords]
    if unindexed_dims:
        dims_str = ", ".join(f"{d}" for d in unindexed_dims)
        return "Dimensions without coordinates: " + dims_str
    else:
        return None


@contextlib.contextmanager
def set_numpy_options(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)


def limit_lines(string: str, *, limit: int):
    """
    If the string is more lines than the limit,
    this returns the middle lines replaced by an ellipsis
    """
    lines = string.splitlines()
    if len(lines) > limit:
        string = "\n".join(chain(lines[: limit // 2], ["..."], lines[-limit // 2 :]))
    return string
```
### 4 - xarray/core/accessor_str.py:

Start line: 1295, End line: 1319

```python
class StringAccessor:

    def rstrip(
        self,
        to_strip: Union[str, bytes, Any] = None,
    ) -> Any:
        """
        Remove trailing characters.

        Strip whitespaces (including newlines) or a set of specified characters
        from each string in the array from the right side.

        `to_strip` can either be a ``str`` or array-like of ``str``.
        If array-like, it will be broadcast and applied elementwise.

        Parameters
        ----------
        to_strip : str or array-like of str or None, default: None
            Specifying the set of characters to be removed.
            All combinations of this set of characters will be stripped.
            If None then whitespaces are removed. If array-like, it is broadcast.

        Returns
        -------
        stripped : same type as values
        """
        return self.strip(to_strip, side="right")
```
### 5 - doc/conf.py:

Start line: 1, End line: 101

```python
#
# xarray documentation build configuration file, created by
# sphinx-quickstart on Thu Feb  6 18:57:54 2014.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.


import datetime
import inspect
import os
import subprocess
import sys
from contextlib import suppress

import sphinx_autosummary_accessors

import xarray

allowed_failures = set()

print("python exec:", sys.executable)
print("sys.path:", sys.path)

if "conda" in sys.executable:
    print("conda environment:")
    subprocess.run(["conda", "list"])
else:
    print("pip environment:")
    subprocess.run([sys.executable, "-m", "pip", "list"])

print(f"xarray: {xarray.__version__}, {xarray.__file__}")

with suppress(ImportError):
    import matplotlib

    matplotlib.use("Agg")

try:
    import rasterio  # noqa: F401
except ImportError:
    allowed_failures.update(
        ["gallery/plot_rasterio_rgb.py", "gallery/plot_rasterio.py"]
    )

try:
    import cartopy  # noqa: F401
except ImportError:
    allowed_failures.update(
        [
            "gallery/plot_cartopy_facetgrid.py",
            "gallery/plot_rasterio_rgb.py",
            "gallery/plot_rasterio.py",
        ]
    )

# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.extlinks",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "IPython.sphinxext.ipython_directive",
    "IPython.sphinxext.ipython_console_highlighting",
    "nbsphinx",
    "sphinx_autosummary_accessors",
    "sphinx.ext.linkcode",
    "sphinx_panels",
    "sphinxext.opengraph",
    "sphinx_copybutton",
    "sphinxext.rediraffe",
]

extlinks = {
    "issue": ("https://github.com/pydata/xarray/issues/%s", "GH"),
    "pull": ("https://github.com/pydata/xarray/pull/%s", "PR"),
}

# sphinx-copybutton configurations
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# nbsphinx configurations

nbsphinx_timeout = 600
nbsphinx_execute = "always"
```
### 6 - xarray/core/dataset.py:

Start line: 419, End line: 444

```python
def _maybe_chunk(
    name,
    var,
    chunks,
    token=None,
    lock=None,
    name_prefix="xarray-",
    overwrite_encoded_chunks=False,
):
    from dask.base import tokenize

    if chunks is not None:
        chunks = {dim: chunks[dim] for dim in var.dims if dim in chunks}
    if var.ndim:
        # when rechunking by different amounts, make sure dask names change
        # by provinding chunks as an input to tokenize.
        # subtle bugs result otherwise. see GH3350
        token2 = tokenize(name, token if token else var._data, chunks)
        name2 = f"{name_prefix}{name}-{token2}"
        var = var.chunk(chunks, name=name2, lock=lock)

        if overwrite_encoded_chunks and var.chunks is not None:
            var.encoding["chunks"] = tuple(x[0] for x in var.chunks)
        return var
    else:
        return var
```
### 7 - xarray/core/dataset.py:

Start line: 5788, End line: 6393

```python
class Dataset(Mapping, ImplementsDatasetReduce, DataWithCoords):

    def quantile(
        self,
        q,
        dim=None,
        interpolation="linear",
        numeric_only=False,
        keep_attrs=None,
        skipna=True,
    ):
        """Compute the qth quantile of the data along the specified dimension.

        Returns the qth quantiles(s) of the array elements for each variable
        in the Dataset.

        Parameters
        ----------
        q : float or array-like of float
            Quantile to compute, which must be between 0 and 1 inclusive.
        dim : str or sequence of str, optional
            Dimension(s) over which to apply quantile.
        interpolation : {"linear", "lower", "higher", "midpoint", "nearest"}, default: "linear"
            This optional parameter specifies the interpolation method to
            use when the desired quantile lies between two data points
            ``i < j``:

                * linear: ``i + (j - i) * fraction``, where ``fraction`` is
                  the fractional part of the index surrounded by ``i`` and
                  ``j``.
                * lower: ``i``.
                * higher: ``j``.
                * nearest: ``i`` or ``j``, whichever is nearest.
                * midpoint: ``(i + j) / 2``.
        keep_attrs : bool, optional
            If True, the dataset's attributes (`attrs`) will be copied from
            the original object to the new one.  If False (default), the new
            object will be returned without attributes.
        numeric_only : bool, optional
            If True, only apply ``func`` to variables with a numeric dtype.
        skipna : bool, optional
            Whether to skip missing values when aggregating.

        Returns
        -------
        quantiles : Dataset
            If `q` is a single quantile, then the result is a scalar for each
            variable in data_vars. If multiple percentiles are given, first
            axis of the result corresponds to the quantile and a quantile
            dimension is added to the return Dataset. The other dimensions are
            the dimensions that remain after the reduction of the array.

        See Also
        --------
        numpy.nanquantile, numpy.quantile, pandas.Series.quantile, DataArray.quantile

        Examples
        --------
        >>> ds = xr.Dataset(
        ...     {"a": (("x", "y"), [[0.7, 4.2, 9.4, 1.5], [6.5, 7.3, 2.6, 1.9]])},
        ...     coords={"x": [7, 9], "y": [1, 1.5, 2, 2.5]},
        ... )
        >>> ds.quantile(0)  # or ds.quantile(0, dim=...)
        <xarray.Dataset>
        Dimensions:   ()
        Coordinates:
            quantile  float64 0.0
        Data variables:
            a         float64 0.7
        >>> ds.quantile(0, dim="x")
        <xarray.Dataset>
        Dimensions:   (y: 4)
        Coordinates:
          * y         (y) float64 1.0 1.5 2.0 2.5
            quantile  float64 0.0
        Data variables:
            a         (y) float64 0.7 4.2 2.6 1.5
        >>> ds.quantile([0, 0.5, 1])
        <xarray.Dataset>
        Dimensions:   (quantile: 3)
        Coordinates:
          * quantile  (quantile) float64 0.0 0.5 1.0
        Data variables:
            a         (quantile) float64 0.7 3.4 9.4
        >>> ds.quantile([0, 0.5, 1], dim="x")
        <xarray.Dataset>
        Dimensions:   (quantile: 3, y: 4)
        Coordinates:
          * y         (y) float64 1.0 1.5 2.0 2.5
          * quantile  (quantile) float64 0.0 0.5 1.0
        Data variables:
            a         (quantile, y) float64 0.7 4.2 2.6 1.5 3.6 ... 1.7 6.5 7.3 9.4 1.9
        """

        if isinstance(dim, str):
            dims = {dim}
        elif dim in [None, ...]:
            dims = set(self.dims)
        else:
            dims = set(dim)

        _assert_empty(
            [d for d in dims if d not in self.dims],
            "Dataset does not contain the dimensions: %s",
        )

        q = np.asarray(q, dtype=np.float64)

        variables = {}
        for name, var in self.variables.items():
            reduce_dims = [d for d in var.dims if d in dims]
            if reduce_dims or not var.dims:
                if name not in self.coords:
                    if (
                        not numeric_only
                        or np.issubdtype(var.dtype, np.number)
                        or var.dtype == np.bool_
                    ):
                        if len(reduce_dims) == var.ndim:
                            # prefer to aggregate over axis=None rather than
                            # axis=(0, 1) if they will be equivalent, because
                            # the former is often more efficient
                            reduce_dims = None
                        variables[name] = var.quantile(
                            q,
                            dim=reduce_dims,
                            interpolation=interpolation,
                            keep_attrs=keep_attrs,
                            skipna=skipna,
                        )

            else:
                variables[name] = var

        # construct the new dataset
        coord_names = {k for k in self.coords if k in variables}
        indexes = {k: v for k, v in self.indexes.items() if k in variables}
        if keep_attrs is None:
            keep_attrs = _get_keep_attrs(default=False)
        attrs = self.attrs if keep_attrs else None
        new = self._replace_with_new_dims(
            variables, coord_names=coord_names, attrs=attrs, indexes=indexes
        )
        return new.assign_coords(quantile=q)

    def rank(self, dim, pct=False, keep_attrs=None):
        """Ranks the data.

        Equal values are assigned a rank that is the average of the ranks that
        would have been otherwise assigned to all of the values within
        that set.
        Ranks begin at 1, not 0. If pct is True, computes percentage ranks.

        NaNs in the input array are returned as NaNs.

        The `bottleneck` library is required.

        Parameters
        ----------
        dim : str
            Dimension over which to compute rank.
        pct : bool, optional
            If True, compute percentage ranks, otherwise compute integer ranks.
        keep_attrs : bool, optional
            If True, the dataset's attributes (`attrs`) will be copied from
            the original object to the new one.  If False (default), the new
            object will be returned without attributes.

        Returns
        -------
        ranked : Dataset
            Variables that do not depend on `dim` are dropped.
        """
        if dim not in self.dims:
            raise ValueError("Dataset does not contain the dimension: %s" % dim)

        variables = {}
        for name, var in self.variables.items():
            if name in self.data_vars:
                if dim in var.dims:
                    variables[name] = var.rank(dim, pct=pct)
            else:
                variables[name] = var

        coord_names = set(self.coords)
        if keep_attrs is None:
            keep_attrs = _get_keep_attrs(default=False)
        attrs = self.attrs if keep_attrs else None
        return self._replace(variables, coord_names, attrs=attrs)

    def differentiate(self, coord, edge_order=1, datetime_unit=None):
        """ Differentiate with the second order accurate central
        differences.

        .. note::
            This feature is limited to simple cartesian geometry, i.e. coord
            must be one dimensional.

        Parameters
        ----------
        coord : str
            The coordinate to be used to compute the gradient.
        edge_order : {1, 2}, default: 1
            N-th order accurate differences at the boundaries.
        datetime_unit : None or {"Y", "M", "W", "D", "h", "m", "s", "ms", \
            "us", "ns", "ps", "fs", "as"}, default: None
            Unit to compute gradient. Only valid for datetime coordinate.

        Returns
        -------
        differentiated: Dataset

        See also
        --------
        numpy.gradient: corresponding numpy function
        """
        from .variable import Variable

        if coord not in self.variables and coord not in self.dims:
            raise ValueError(f"Coordinate {coord} does not exist.")

        coord_var = self[coord].variable
        if coord_var.ndim != 1:
            raise ValueError(
                "Coordinate {} must be 1 dimensional but is {}"
                " dimensional".format(coord, coord_var.ndim)
            )

        dim = coord_var.dims[0]
        if _contains_datetime_like_objects(coord_var):
            if coord_var.dtype.kind in "mM" and datetime_unit is None:
                datetime_unit, _ = np.datetime_data(coord_var.dtype)
            elif datetime_unit is None:
                datetime_unit = "s"  # Default to seconds for cftime objects
            coord_var = coord_var._to_numeric(datetime_unit=datetime_unit)

        variables = {}
        for k, v in self.variables.items():
            if k in self.data_vars and dim in v.dims and k not in self.coords:
                if _contains_datetime_like_objects(v):
                    v = v._to_numeric(datetime_unit=datetime_unit)
                grad = duck_array_ops.gradient(
                    v.data, coord_var, edge_order=edge_order, axis=v.get_axis_num(dim)
                )
                variables[k] = Variable(v.dims, grad)
            else:
                variables[k] = v
        return self._replace(variables)

    def integrate(
        self, coord: Union[Hashable, Sequence[Hashable]], datetime_unit: str = None
    ) -> "Dataset":
        """Integrate along the given coordinate using the trapezoidal rule.

        .. note::
            This feature is limited to simple cartesian geometry, i.e. coord
            must be one dimensional.

        Parameters
        ----------
        coord : hashable, or sequence of hashable
            Coordinate(s) used for the integration.
        datetime_unit : {'Y', 'M', 'W', 'D', 'h', 'm', 's', 'ms', 'us', 'ns', \
                        'ps', 'fs', 'as'}, optional
            Specify the unit if datetime coordinate is used.

        Returns
        -------
        integrated : Dataset

        See also
        --------
        DataArray.integrate
        numpy.trapz : corresponding numpy function

        Examples
        --------
        >>> ds = xr.Dataset(
        ...     data_vars={"a": ("x", [5, 5, 6, 6]), "b": ("x", [1, 2, 1, 0])},
        ...     coords={"x": [0, 1, 2, 3], "y": ("x", [1, 7, 3, 5])},
        ... )
        >>> ds
        <xarray.Dataset>
        Dimensions:  (x: 4)
        Coordinates:
          * x        (x) int64 0 1 2 3
            y        (x) int64 1 7 3 5
        Data variables:
            a        (x) int64 5 5 6 6
            b        (x) int64 1 2 1 0
        >>> ds.integrate("x")
        <xarray.Dataset>
        Dimensions:  ()
        Data variables:
            a        float64 16.5
            b        float64 3.5
        >>> ds.integrate("y")
        <xarray.Dataset>
        Dimensions:  ()
        Data variables:
            a        float64 20.0
            b        float64 4.0
        """
        if not isinstance(coord, (list, tuple)):
            coord = (coord,)
        result = self
        for c in coord:
            result = result._integrate_one(c, datetime_unit=datetime_unit)
        return result

    def _integrate_one(self, coord, datetime_unit=None):
        from .variable import Variable

        if coord not in self.variables and coord not in self.dims:
            raise ValueError(f"Coordinate {coord} does not exist.")

        coord_var = self[coord].variable
        if coord_var.ndim != 1:
            raise ValueError(
                "Coordinate {} must be 1 dimensional but is {}"
                " dimensional".format(coord, coord_var.ndim)
            )

        dim = coord_var.dims[0]
        if _contains_datetime_like_objects(coord_var):
            if coord_var.dtype.kind in "mM" and datetime_unit is None:
                datetime_unit, _ = np.datetime_data(coord_var.dtype)
            elif datetime_unit is None:
                datetime_unit = "s"  # Default to seconds for cftime objects
            coord_var = coord_var._replace(
                data=datetime_to_numeric(coord_var.data, datetime_unit=datetime_unit)
            )

        variables = {}
        coord_names = set()
        for k, v in self.variables.items():
            if k in self.coords:
                if dim not in v.dims:
                    variables[k] = v
                    coord_names.add(k)
            else:
                if k in self.data_vars and dim in v.dims:
                    if _contains_datetime_like_objects(v):
                        v = datetime_to_numeric(v, datetime_unit=datetime_unit)
                    integ = duck_array_ops.trapz(
                        v.data, coord_var.data, axis=v.get_axis_num(dim)
                    )
                    v_dims = list(v.dims)
                    v_dims.remove(dim)
                    variables[k] = Variable(v_dims, integ)
                else:
                    variables[k] = v
        indexes = {k: v for k, v in self.indexes.items() if k in variables}
        return self._replace_with_new_dims(
            variables, coord_names=coord_names, indexes=indexes
        )

    @property
    def real(self):
        return self.map(lambda x: x.real, keep_attrs=True)

    @property
    def imag(self):
        return self.map(lambda x: x.imag, keep_attrs=True)

    plot = utils.UncachedAccessor(_Dataset_PlotMethods)

    def filter_by_attrs(self, **kwargs):
        """Returns a ``Dataset`` with variables that match specific conditions.

        Can pass in ``key=value`` or ``key=callable``.  A Dataset is returned
        containing only the variables for which all the filter tests pass.
        These tests are either ``key=value`` for which the attribute ``key``
        has the exact value ``value`` or the callable passed into
        ``key=callable`` returns True. The callable will be passed a single
        value, either the value of the attribute ``key`` or ``None`` if the
        DataArray does not have an attribute with the name ``key``.

        Parameters
        ----------
        **kwargs
            key : str
                Attribute name.
            value : callable or obj
                If value is a callable, it should return a boolean in the form
                of bool = func(attr) where attr is da.attrs[key].
                Otherwise, value will be compared to the each
                DataArray's attrs[key].

        Returns
        -------
        new : Dataset
            New dataset with variables filtered by attribute.

        Examples
        --------
        >>> # Create an example dataset:
        >>> temp = 15 + 8 * np.random.randn(2, 2, 3)
        >>> precip = 10 * np.random.rand(2, 2, 3)
        >>> lon = [[-99.83, -99.32], [-99.79, -99.23]]
        >>> lat = [[42.25, 42.21], [42.63, 42.59]]
        >>> dims = ["x", "y", "time"]
        >>> temp_attr = dict(standard_name="air_potential_temperature")
        >>> precip_attr = dict(standard_name="convective_precipitation_flux")
        >>> ds = xr.Dataset(
        ...     {
        ...         "temperature": (dims, temp, temp_attr),
        ...         "precipitation": (dims, precip, precip_attr),
        ...     },
        ...     coords={
        ...         "lon": (["x", "y"], lon),
        ...         "lat": (["x", "y"], lat),
        ...         "time": pd.date_range("2014-09-06", periods=3),
        ...         "reference_time": pd.Timestamp("2014-09-05"),
        ...     },
        ... )
        >>> # Get variables matching a specific standard_name.
        >>> ds.filter_by_attrs(standard_name="convective_precipitation_flux")
        <xarray.Dataset>
        Dimensions:         (time: 3, x: 2, y: 2)
        Coordinates:
            lon             (x, y) float64 -99.83 -99.32 -99.79 -99.23
            lat             (x, y) float64 42.25 42.21 42.63 42.59
          * time            (time) datetime64[ns] 2014-09-06 2014-09-07 2014-09-08
            reference_time  datetime64[ns] 2014-09-05
        Dimensions without coordinates: x, y
        Data variables:
            precipitation   (x, y, time) float64 5.68 9.256 0.7104 ... 7.992 4.615 7.805
        >>> # Get all variables that have a standard_name attribute.
        >>> standard_name = lambda v: v is not None
        >>> ds.filter_by_attrs(standard_name=standard_name)
        <xarray.Dataset>
        Dimensions:         (time: 3, x: 2, y: 2)
        Coordinates:
            lon             (x, y) float64 -99.83 -99.32 -99.79 -99.23
            lat             (x, y) float64 42.25 42.21 42.63 42.59
          * time            (time) datetime64[ns] 2014-09-06 2014-09-07 2014-09-08
            reference_time  datetime64[ns] 2014-09-05
        Dimensions without coordinates: x, y
        Data variables:
            temperature     (x, y, time) float64 29.11 18.2 22.83 ... 18.28 16.15 26.63
            precipitation   (x, y, time) float64 5.68 9.256 0.7104 ... 7.992 4.615 7.805

        """
        selection = []
        for var_name, variable in self.variables.items():
            has_value_flag = False
            for attr_name, pattern in kwargs.items():
                attr_value = variable.attrs.get(attr_name)
                if (callable(pattern) and pattern(attr_value)) or attr_value == pattern:
                    has_value_flag = True
                else:
                    has_value_flag = False
                    break
            if has_value_flag is True:
                selection.append(var_name)
        return self[selection]

    def unify_chunks(self) -> "Dataset":
        """Unify chunk size along all chunked dimensions of this Dataset.

        Returns
        -------
        Dataset with consistent chunk sizes for all dask-array variables

        See Also
        --------
        dask.array.core.unify_chunks
        """

        try:
            self.chunks
        except ValueError:  # "inconsistent chunks"
            pass
        else:
            # No variables with dask backend, or all chunks are already aligned
            return self.copy()

        # import dask is placed after the quick exit test above to allow
        # running this method if dask isn't installed and there are no chunks
        import dask.array

        ds = self.copy()

        dims_pos_map = {dim: index for index, dim in enumerate(ds.dims)}

        dask_array_names = []
        dask_unify_args = []
        for name, variable in ds.variables.items():
            if isinstance(variable.data, dask.array.Array):
                dims_tuple = [dims_pos_map[dim] for dim in variable.dims]
                dask_array_names.append(name)
                dask_unify_args.append(variable.data)
                dask_unify_args.append(dims_tuple)

        _, rechunked_arrays = dask.array.core.unify_chunks(*dask_unify_args)

        for name, new_array in zip(dask_array_names, rechunked_arrays):
            ds.variables[name]._data = new_array

        return ds

    def map_blocks(
        self,
        func: "Callable[..., T_DSorDA]",
        args: Sequence[Any] = (),
        kwargs: Mapping[str, Any] = None,
        template: Union["DataArray", "Dataset"] = None,
    ) -> "T_DSorDA":
        """
        Apply a function to each block of this Dataset.

        .. warning::
            This method is experimental and its signature may change.

        Parameters
        ----------
        func : callable
            User-provided function that accepts a Dataset as its first
            parameter. The function will receive a subset or 'block' of this Dataset (see below),
            corresponding to one chunk along each chunked dimension. ``func`` will be
            executed as ``func(subset_dataset, *subset_args, **kwargs)``.

            This function must return either a single DataArray or a single Dataset.

            This function cannot add a new chunked dimension.
        args : sequence
            Passed to func after unpacking and subsetting any xarray objects by blocks.
            xarray objects in args must be aligned with obj, otherwise an error is raised.
        kwargs : mapping
            Passed verbatim to func after unpacking. xarray objects, if any, will not be
            subset to blocks. Passing dask collections in kwargs is not allowed.
        template : DataArray or Dataset, optional
            xarray object representing the final result after compute is called. If not provided,
            the function will be first run on mocked-up data, that looks like this object but
            has sizes 0, to determine properties of the returned object such as dtype,
            variable names, attributes, new dimensions and new indexes (if any).
            ``template`` must be provided if the function changes the size of existing dimensions.
            When provided, ``attrs`` on variables in `template` are copied over to the result. Any
            ``attrs`` set by ``func`` will be ignored.

        Returns
        -------
        A single DataArray or Dataset with dask backend, reassembled from the outputs of the
        function.

        Notes
        -----
        This function is designed for when ``func`` needs to manipulate a whole xarray object
        subset to each block. Each block is loaded into memory. In the more common case where
        ``func`` can work on numpy arrays, it is recommended to use ``apply_ufunc``.

        If none of the variables in this object is backed by dask arrays, calling this function is
        equivalent to calling ``func(obj, *args, **kwargs)``.

        See Also
        --------
        dask.array.map_blocks, xarray.apply_ufunc, xarray.Dataset.map_blocks
        xarray.DataArray.map_blocks

        Examples
        --------
        Calculate an anomaly from climatology using ``.groupby()``. Using
        ``xr.map_blocks()`` allows for parallel operations with knowledge of ``xarray``,
        its indices, and its methods like ``.groupby()``.

        >>> def calculate_anomaly(da, groupby_type="time.month"):
        ...     gb = da.groupby(groupby_type)
        ...     clim = gb.mean(dim="time")
        ...     return gb - clim
        ...
        >>> time = xr.cftime_range("1990-01", "1992-01", freq="M")
        >>> month = xr.DataArray(time.month, coords={"time": time}, dims=["time"])
        >>> np.random.seed(123)
        >>> array = xr.DataArray(
        ...     np.random.rand(len(time)),
        ...     dims=["time"],
        ...     coords={"time": time, "month": month},
        ... ).chunk()
        >>> ds = xr.Dataset({"a": array})
        >>> ds.map_blocks(calculate_anomaly, template=ds).compute()
        <xarray.Dataset>
        Dimensions:  (time: 24)
        Coordinates:
          * time     (time) object 1990-01-31 00:00:00 ... 1991-12-31 00:00:00
            month    (time) int64 1 2 3 4 5 6 7 8 9 10 11 12 1 2 3 4 5 6 7 8 9 10 11 12
        Data variables:
            a        (time) float64 0.1289 0.1132 -0.0856 ... 0.2287 0.1906 -0.05901

        Note that one must explicitly use ``args=[]`` and ``kwargs={}`` to pass arguments
        to the function being applied in ``xr.map_blocks()``:

        >>> ds.map_blocks(
        ...     calculate_anomaly,
        ...     kwargs={"groupby_type": "time.year"},
        ...     template=ds,
        ... )
        <xarray.Dataset>
        Dimensions:  (time: 24)
        Coordinates:
          * time     (time) object 1990-01-31 00:00:00 ... 1991-12-31 00:00:00
            month    (time) int64 dask.array<chunksize=(24,), meta=np.ndarray>
        Data variables:
            a        (time) float64 dask.array<chunksize=(24,), meta=np.ndarray>
        """
        from .parallel import map_blocks

        return map_blocks(func, self, args, kwargs, template)
```
### 8 - xarray/core/groupby.py:

Start line: 488, End line: 512

```python
class GroupBy(SupportsArithmetic):

    def _yield_binary_applied(self, func, other):
        dummy = None

        for group_value, obj in self:
            try:
                other_sel = other.sel(**{self._group.name: group_value})
            except AttributeError:
                raise TypeError(
                    "GroupBy objects only support binary ops "
                    "when the other argument is a Dataset or "
                    "DataArray"
                )
            except (KeyError, ValueError):
                if self._group.name not in other.dims:
                    raise ValueError(
                        "incompatible dimensions for a grouped "
                        "binary operation: the group variable %r "
                        "is not a dimension on the other argument" % self._group.name
                    )
                if dummy is None:
                    dummy = _dummy_copy(other)
                other_sel = dummy

            result = func(obj, other_sel)
            yield result
```
### 9 - doc/conf.py:

Start line: 260, End line: 309

```python
ogp_custom_meta_tags = [
    '<meta name="twitter:card" content="summary_large_image" />',
    '<meta property="twitter:site" content="@xarray_dev />',
    '<meta name="image" property="og:image" content="https://xarray.pydata.org/en/stable/_static/dataset-diagram-logo.png">',
]

# Redirects for pages that were moved to new locations

rediraffe_redirects = {
    "terminology.rst": "user-guide/terminology.rst",
    "data-structures.rst": "user-guide/data-structures.rst",
    "indexing.rst": "user-guide/indexing.rst",
    "interpolation.rst": "user-guide/interpolation.rst",
    "computation.rst": "user-guide/computation.rst",
    "groupby.rst": "user-guide/groupby.rst",
    "reshaping.rst": "user-guide/reshaping.rst",
    "combining.rst": "user-guide/combining.rst",
    "time-series.rst": "user-guide/time-series.rst",
    "weather-climate.rst": "user-guide/weather-climate.rst",
    "pandas.rst": "user-guide/pandas.rst",
    "io.rst": "user-guide/io.rst",
    "dask.rst": "user-guide/dask.rst",
    "plotting.rst": "user-guide/plotting.rst",
    "duckarrays.rst": "user-guide/duckarrays.rst",
    "related-projects.rst": "ecosystem.rst",
    "faq.rst": "getting-started-guide/faq.rst",
    "why-xarray.rst": "getting-started-guide/why-xarray.rst",
    "installing.rst": "getting-started-guide/installing.rst",
    "quick-overview.rst": "getting-started-guide/quick-overview.rst",
}

# Sometimes the savefig directory doesn't exist and needs to be created
# https://github.com/ipython/ipython/issues/8733
# becomes obsolete when we can pin ipython>=5.2; see ci/requirements/doc.yml
ipython_savefig_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "_build", "html", "_static"
)
if not os.path.exists(ipython_savefig_dir):
    os.makedirs(ipython_savefig_dir)


# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
html_last_updated_fmt = today_fmt

# Output file base name for HTML help builder.
htmlhelp_basename = "xarraydoc"


# Example configuration for intersphinx: refer to the Python standard library.
```
### 10 - xarray/core/formatting.py:

Start line: 671, End line: 692

```python
def diff_dataset_repr(a, b, compat):
    summary = [
        "Left and right {} objects are not {}".format(
            type(a).__name__, _compat_to_str(compat)
        )
    ]

    col_width = _calculate_col_width(
        set(_get_col_items(a.variables) + _get_col_items(b.variables))
    )

    summary.append(diff_dim_summary(a, b))
    summary.append(diff_coords_repr(a.coords, b.coords, compat, col_width=col_width))
    summary.append(
        diff_data_vars_repr(a.data_vars, b.data_vars, compat, col_width=col_width)
    )

    if compat == "identical":
        summary.append(diff_attrs_repr(a.attrs, b.attrs, compat))

    return "\n".join(summary)
```
### 18 - xarray/core/groupby.py:

Start line: 185, End line: 199

```python
def _ensure_1d(group, obj):
    if group.ndim != 1:
        # try to stack the dims of the group into a single dim
        orig_dims = group.dims
        stacked_dim = "stacked_" + "_".join(orig_dims)
        # these dimensions get created by the stack operation
        inserted_dims = [dim for dim in group.dims if dim not in group.coords]
        # the copy is necessary here, otherwise read only array raises error
        # in pandas: https://github.com/pydata/pandas/issues/12813
        group = group.stack(**{stacked_dim: orig_dims}).copy()
        obj = obj.stack(**{stacked_dim: orig_dims})
    else:
        stacked_dim = None
        inserted_dims = []
    return group, obj, stacked_dim, inserted_dims
```
### 20 - xarray/core/groupby.py:

Start line: 933, End line: 960

```python
class DatasetGroupBy(GroupBy, ImplementsDatasetReduce):

    def apply(self, func, args=(), shortcut=None, **kwargs):
        """
        Backward compatible implementation of ``map``

        See Also
        --------
        DatasetGroupBy.map
        """

        warnings.warn(
            "GroupBy.apply may be deprecated in the future. Using GroupBy.map is encouraged",
            PendingDeprecationWarning,
            stacklevel=2,
        )
        return self.map(func, shortcut=shortcut, args=args, **kwargs)

    def _combine(self, applied):
        """Recombine the applied objects like the original."""
        applied_example, applied = peek_at(applied)
        coord, dim, positions = self._infer_concat_args(applied_example)
        combined = concat(applied, dim)
        combined = _maybe_reorder(combined, dim, positions)
        # assign coord when the applied function does not return that coord
        if coord is not None and dim not in applied_example.dims:
            combined[coord.name] = coord
        combined = self._maybe_restore_empty_groups(combined)
        combined = self._maybe_unstack(combined)
        return combined
```
### 24 - xarray/core/groupby.py:

Start line: 1, End line: 36

```python
import datetime
import functools
import warnings

import numpy as np
import pandas as pd

from . import dtypes, duck_array_ops, nputils, ops
from .arithmetic import SupportsArithmetic
from .common import ImplementsArrayReduce, ImplementsDatasetReduce
from .concat import concat
from .formatting import format_array_flat
from .indexes import propagate_indexes
from .options import _get_keep_attrs
from .pycompat import integer_types
from .utils import (
    either_dict_or_kwargs,
    hashable,
    is_scalar,
    maybe_wrap_array,
    peek_at,
    safe_cast_to_index,
)
from .variable import IndexVariable, Variable, as_variable


def check_reduce_dims(reduce_dims, dimensions):

    if reduce_dims is not ...:
        if is_scalar(reduce_dims):
            reduce_dims = [reduce_dims]
        if any(dim not in dimensions for dim in reduce_dims):
            raise ValueError(
                "cannot reduce over dimensions %r. expected either '...' to reduce over all dimensions or one or more of %r."
                % (reduce_dims, dimensions)
            )
```
### 28 - xarray/core/groupby.py:

Start line: 894, End line: 931

```python
ops.inject_reduce_methods(DataArrayGroupBy)
ops.inject_binary_ops(DataArrayGroupBy)


class DatasetGroupBy(GroupBy, ImplementsDatasetReduce):
    def map(self, func, args=(), shortcut=None, **kwargs):
        """Apply a function to each Dataset in the group and concatenate them
        together into a new Dataset.

        `func` is called like `func(ds, *args, **kwargs)` for each dataset `ds`
        in this group.

        Apply uses heuristics (like `pandas.GroupBy.apply`) to figure out how
        to stack together the datasets. The rule is:

        1. If the dimension along which the group coordinate is defined is
           still in the first grouped item after applying `func`, then stack
           over this dimension.
        2. Otherwise, stack over the new dimension given by name of this
           grouping (the argument to the `groupby` function).

        Parameters
        ----------
        func : callable
            Callable to apply to each sub-dataset.
        args : tuple, optional
            Positional arguments to pass to `func`.
        **kwargs
            Used to call `func(ds, **kwargs)` for each sub-dataset `ar`.

        Returns
        -------
        applied : Dataset or DataArray
            The result of splitting, applying and combining this dataset.
        """
        # ignore shortcut if set (for now)
        applied = (func(ds, *args, **kwargs) for ds in self._iter_grouped())
        return self._combine(applied)
```
