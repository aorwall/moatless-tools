# astropy__astropy-13469

| **astropy/astropy** | `2b8631e7d64bfc16c70f5c51cda97964d8dd1ae0` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 200 |
| **Any found context length** | 200 |
| **Avg pos** | 1.0 |
| **Min pos** | 1 |
| **Max pos** | 1 |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/astropy/table/table.py b/astropy/table/table.py
--- a/astropy/table/table.py
+++ b/astropy/table/table.py
@@ -1070,7 +1070,12 @@ def __array__(self, dtype=None):
         supported and will raise a ValueError.
         """
         if dtype is not None:
-            raise ValueError('Datatype coercion is not allowed')
+            if np.dtype(dtype) != object:
+                raise ValueError('Datatype coercion is not allowed')
+
+            out = np.array(None, dtype=object)
+            out[()] = self
+            return out
 
         # This limitation is because of the following unexpected result that
         # should have made a table copy while changing the column names.

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| astropy/table/table.py | 1073 | 1073 | 1 | 1 | 200


## Problem Statement

```
Can't convert a list of Astropy tables to a NumPy array of tables
I recently stumbled upon [a StackOverflow question](https://stackoverflow.com/questions/69414829/convert-a-list-of-astropy-table-in-a-numpy-array-of-astropy-table) where someone likes to convert a list of Tables to a NumPy array.
By default, NumPy will convert the Table along the way, resulting in the wrong data structure. 
Using a specific `dtype=object`, however, fails with 
\`\`\`
ValueError: Datatype coercion is not allowed
\`\`\`

This error leads directly to the source of `table.__array__()`, which explicitly checks for any `dtype` to be not `None`, which will raise the error.
The reasoning behind that is clear, as given in the comments below. 

But I wonder if an exception is reasonable for `dtype=object` here, and let that pass through. For a single Table, this may be odd, but not necessarily incorrect. And for a list of Tables, to be converted to an array, this may be helpful.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 astropy/table/table.py** | 1066 | 1084| 200 | 200 | 32807 | 
| 2 | 2 astropy/table/np_utils.py | 116 | 155| 389 | 589 | 34161 | 
| 3 | 3 astropy/table/column.py | 218 | 287| 835 | 1424 | 48818 | 
| 4 | **3 astropy/table/table.py** | 3707 | 3761| 506 | 1930 | 48818 | 
| 5 | 4 astropy/table/row.py | 61 | 116| 401 | 2331 | 50203 | 
| 6 | 5 astropy/table/meta.py | 159 | 218| 511 | 2842 | 53338 | 
| 7 | **5 astropy/table/table.py** | 1086 | 1096| 121 | 2963 | 53338 | 
| 8 | 6 astropy/table/operations.py | 961 | 973| 111 | 3074 | 67664 | 
| 9 | **6 astropy/table/table.py** | 1252 | 1317| 706 | 3780 | 67664 | 
| 10 | 6 astropy/table/column.py | 149 | 216| 716 | 4496 | 67664 | 
| 11 | 7 astropy/io/fits/util.py | 700 | 716| 157 | 4653 | 74701 | 
| 12 | **7 astropy/table/table.py** | 3850 | 3898| 495 | 5148 | 74701 | 
| 13 | **7 astropy/table/table.py** | 2710 | 2730| 191 | 5339 | 74701 | 
| 14 | **7 astropy/table/table.py** | 1179 | 1250| 624 | 5963 | 74701 | 
| 15 | **7 astropy/table/table.py** | 2 | 75| 641 | 6604 | 74701 | 
| 16 | **7 astropy/table/table.py** | 1870 | 1898| 342 | 6946 | 74701 | 
| 17 | 8 astropy/io/votable/exceptions.py | 856 | 868| 153 | 7099 | 88025 | 
| 18 | 9 astropy/io/votable/converters.py | 1328 | 1359| 243 | 7342 | 97979 | 
| 19 | 9 astropy/io/votable/converters.py | 1398 | 1459| 521 | 7863 | 97979 | 
| 20 | 9 astropy/io/votable/exceptions.py | 1310 | 1349| 436 | 8299 | 97979 | 
| 21 | 10 astropy/io/misc/asdf/tags/table/table.py | 42 | 67| 218 | 8517 | 98978 | 
| 22 | **10 astropy/table/table.py** | 2680 | 2708| 252 | 8769 | 98978 | 
| 23 | **10 astropy/table/table.py** | 2161 | 2214| 558 | 9327 | 98978 | 
| 24 | **10 astropy/table/table.py** | 3946 | 3966| 219 | 9546 | 98978 | 
| 25 | 11 astropy/utils/metadata.py | 36 | 76| 394 | 9940 | 102938 | 
| 26 | 12 astropy/io/ascii/core.py | 183 | 267| 404 | 10344 | 116561 | 
| 27 | **12 astropy/table/table.py** | 1900 | 1950| 428 | 10772 | 116561 | 
| 28 | **12 astropy/table/table.py** | 1952 | 1966| 134 | 10906 | 116561 | 
| 29 | 12 astropy/io/votable/exceptions.py | 520 | 545| 248 | 11154 | 116561 | 
| 30 | **12 astropy/table/table.py** | 1319 | 1335| 182 | 11336 | 116561 | 
| 31 | 13 astropy/table/table_helpers.py | 9 | 53| 421 | 11757 | 118116 | 
| 32 | **13 astropy/table/table.py** | 3656 | 3705| 479 | 12236 | 118116 | 
| 33 | 14 astropy/table/ndarray_mixin.py | 24 | 66| 381 | 12617 | 118638 | 
| 34 | 15 astropy/io/fits/hdu/table.py | 753 | 783| 312 | 12929 | 131741 | 
| 35 | 16 astropy/io/fits/convenience.py | 487 | 527| 516 | 13445 | 141546 | 
| 36 | 16 astropy/io/misc/asdf/tags/table/table.py | 2 | 40| 292 | 13737 | 141546 | 
| 37 | 16 astropy/io/votable/converters.py | 1362 | 1395| 232 | 13969 | 141546 | 
| 38 | **16 astropy/table/table.py** | 542 | 601| 512 | 14481 | 141546 | 
| 39 | 17 astropy/io/fits/column.py | 2406 | 2440| 239 | 14720 | 163886 | 
| 40 | 17 astropy/io/fits/column.py | 1280 | 1343| 633 | 15353 | 163886 | 
| 41 | 17 astropy/io/ascii/core.py | 1295 | 1312| 138 | 15491 | 163886 | 
| 42 | 17 astropy/io/ascii/core.py | 1015 | 1057| 360 | 15851 | 163886 | 
| 43 | 18 astropy/io/ascii/ecsv.py | 209 | 234| 228 | 16079 | 168083 | 
| 44 | **18 astropy/table/table.py** | 1362 | 1374| 124 | 16203 | 168083 | 
| 45 | 18 astropy/table/operations.py | 41 | 70| 218 | 16421 | 168083 | 
| 46 | 18 astropy/io/votable/converters.py | 600 | 618| 203 | 16624 | 168083 | 
| 47 | 18 astropy/io/votable/exceptions.py | 1192 | 1222| 277 | 16901 | 168083 | 
| 48 | 18 astropy/io/misc/asdf/tags/table/table.py | 70 | 93| 211 | 17112 | 168083 | 
| 49 | **18 astropy/table/table.py** | 603 | 657| 445 | 17557 | 168083 | 
| 50 | 18 astropy/io/fits/convenience.py | 591 | 617| 232 | 17789 | 168083 | 
| 51 | 18 astropy/io/fits/column.py | 387 | 440| 515 | 18304 | 168083 | 
| 52 | 19 astropy/io/fits/fitsrec.py | 833 | 872| 418 | 18722 | 179863 | 
| 53 | 20 astropy/io/ascii/docs.py | 1 | 96| 991 | 19713 | 181667 | 
| 54 | 21 astropy/table/sorted_array.py | 28 | 52| 145 | 19858 | 183728 | 
| 55 | **21 astropy/table/table.py** | 3901 | 3944| 340 | 20198 | 183728 | 
| 56 | 22 astropy/table/mixins/dask.py | 1 | 40| 244 | 20442 | 183973 | 


### Hint

```
FYI, here is a fix that seems to work. If anyone else wants to put this (or some variation) into a PR and add a test etc then feel free!
\`\`\`diff
(astropy) ➜  astropy git:(main) ✗ git diff
diff --git a/astropy/table/table.py b/astropy/table/table.py
index d3bcaebeb5..6db399a7b8 100644
--- a/astropy/table/table.py
+++ b/astropy/table/table.py
@@ -1072,7 +1072,11 @@ class Table:
         Coercion to a different dtype via np.array(table, dtype) is not
         supported and will raise a ValueError.
         """
-        if dtype is not None:
+        if np.dtype(dtype).kind == 'O':
+            out = np.array(None, dtype=object)
+            out[()] = self
+            return out
+        elif dtype is not None:
             raise ValueError('Datatype coercion is not allowed')
 
         # This limitation is because of the following unexpected result that
\`\`\`

```

## Patch

```diff
diff --git a/astropy/table/table.py b/astropy/table/table.py
--- a/astropy/table/table.py
+++ b/astropy/table/table.py
@@ -1070,7 +1070,12 @@ def __array__(self, dtype=None):
         supported and will raise a ValueError.
         """
         if dtype is not None:
-            raise ValueError('Datatype coercion is not allowed')
+            if np.dtype(dtype) != object:
+                raise ValueError('Datatype coercion is not allowed')
+
+            out = np.array(None, dtype=object)
+            out[()] = self
+            return out
 
         # This limitation is because of the following unexpected result that
         # should have made a table copy while changing the column names.

```

## Test Patch

```diff
diff --git a/astropy/table/tests/test_table.py b/astropy/table/tests/test_table.py
--- a/astropy/table/tests/test_table.py
+++ b/astropy/table/tests/test_table.py
@@ -28,6 +28,7 @@
 from .conftest import MaskedTable, MIXIN_COLS
 
 from astropy.utils.compat.optional_deps import HAS_PANDAS  # noqa
+from astropy.utils.compat.numpycompat import NUMPY_LT_1_20
 
 
 @pytest.fixture
@@ -1405,6 +1406,22 @@ def test_byteswap_fits_array(self, table_types):
                 assert (data[colname].dtype.byteorder
                         == arr2[colname].dtype.byteorder)
 
+    def test_convert_numpy_object_array(self, table_types):
+        d = table_types.Table([[1, 2], [3, 4]], names=('a', 'b'))
+
+        # Single table
+        np_d = np.array(d, dtype=object)
+        assert isinstance(np_d, np.ndarray)
+        assert np_d[()] is d
+
+    @pytest.mark.xfail(NUMPY_LT_1_20, reason="numpy array introspection changed")
+    def test_convert_list_numpy_object_array(self, table_types):
+        d = table_types.Table([[1, 2], [3, 4]], names=('a', 'b'))
+        ds = [d, d, d]
+        np_ds = np.array(ds, dtype=object)
+        assert all([isinstance(t, table_types.Table) for t in np_ds])
+        assert all([np.array_equal(t, d) for t in np_ds])
+
 
 def _assert_copies(t, t2, deep=True):
     assert t.colnames == t2.colnames

```


## Code snippets

### 1 - astropy/table/table.py:

Start line: 1066, End line: 1084

```python
class Table:

    def __array__(self, dtype=None):
        """Support converting Table to np.array via np.array(table).

        Coercion to a different dtype via np.array(table, dtype) is not
        supported and will raise a ValueError.
        """
        if dtype is not None:
            raise ValueError('Datatype coercion is not allowed')

        # This limitation is because of the following unexpected result that
        # should have made a table copy while changing the column names.
        #
        # >>> d = astropy.table.Table([[1,2],[3,4]])
        # >>> np.array(d, dtype=[('a', 'i8'), ('b', 'i8')])
        # array([(0, 0), (0, 0)],
        #       dtype=[('a', '<i8'), ('b', '<i8')])

        out = self.as_array()
        return out.data if isinstance(out, np.ma.MaskedArray) else out
```
### 2 - astropy/table/np_utils.py:

Start line: 116, End line: 155

```python
def common_dtype(cols):
    """
    Use numpy to find the common dtype for a list of structured ndarray columns.

    Only allow columns within the following fundamental numpy data types:
    np.bool_, np.object_, np.number, np.character, np.void
    """
    np_types = (np.bool_, np.object_, np.number, np.character, np.void)
    uniq_types = {tuple(issubclass(col.dtype.type, np_type) for np_type in np_types)
                  for col in cols}
    if len(uniq_types) > 1:
        # Embed into the exception the actual list of incompatible types.
        incompat_types = [col.dtype.name for col in cols]
        tme = TableMergeError(f'Columns have incompatible types {incompat_types}')
        tme._incompat_types = incompat_types
        raise tme

    arrs = [np.empty(1, dtype=col.dtype) for col in cols]

    # For string-type arrays need to explicitly fill in non-zero
    # values or the final arr_common = .. step is unpredictable.
    for arr in arrs:
        if arr.dtype.kind in ('S', 'U'):
            arr[0] = '0' * arr.itemsize

    arr_common = np.array([arr[0] for arr in arrs])
    return arr_common.dtype.str


def _check_for_sequence_of_structured_arrays(arrays):
    err = '`arrays` arg must be a sequence (e.g. list) of structured arrays'
    if not isinstance(arrays, Sequence):
        raise TypeError(err)
    for array in arrays:
        # Must be structured array
        if not isinstance(array, np.ndarray) or array.dtype.names is None:
            raise TypeError(err)
    if len(arrays) == 0:
        raise ValueError('`arrays` arg must include at least one array')
```
### 3 - astropy/table/column.py:

Start line: 218, End line: 287

```python
def _convert_sequence_data_to_array(data, dtype=None):
    # ... other code

    if np_data.ndim == 0 or (np_data.ndim > 0 and len(np_data) == 0):
        # Implies input was a scalar or an empty list (e.g. initializing an
        # empty table with pre-declared names and dtypes but no data).  Here we
        # need to fall through to initializing with the original data=[].
        return data

    # If there were no warnings and the data are int or float, then we are done.
    # Other dtypes like string or complex can have masked values and the
    # np.array() conversion gives the wrong answer (e.g. converting np.ma.masked
    # to the string "0.0").
    if len(warns) == 0 and np_data.dtype.kind in ('i', 'f'):
        return np_data

    # Now we need to determine if there is an np.ma.masked anywhere in input data.

    # Make a statement like below to look for np.ma.masked in a nested sequence.
    # Because np.array(data) succeeded we know that `data` has a regular N-d
    # structure. Find ma_masked:
    #   any(any(any(d2 is ma_masked for d2 in d1) for d1 in d0) for d0 in data)
    # Using this eval avoids creating a copy of `data` in the more-usual case of
    # no masked elements.
    any_statement = 'd0 is ma_masked'
    for ii in reversed(range(np_data.ndim)):
        if ii == 0:
            any_statement = f'any({any_statement} for d0 in data)'
        elif ii == np_data.ndim - 1:
            any_statement = f'any(d{ii} is ma_masked for d{ii} in d{ii-1})'
        else:
            any_statement = f'any({any_statement} for d{ii} in d{ii-1})'
    context = {'ma_masked': np.ma.masked, 'data': data}
    has_masked = eval(any_statement, context)

    # If there are any masks then explicitly change each one to a fill value and
    # set a mask boolean array. If not has_masked then we're done.
    if has_masked:
        mask = np.zeros(np_data.shape, dtype=bool)
        data_filled = np.array(data, dtype=object)

        # Make type-appropriate fill value based on initial conversion.
        if np_data.dtype.kind == 'U':
            fill = ''
        elif np_data.dtype.kind == 'S':
            fill = b''
        else:
            # Zero works for every numeric type.
            fill = 0

        ranges = [range(dim) for dim in np_data.shape]
        for idxs in itertools.product(*ranges):
            val = data_filled[idxs]
            if val is np_ma_masked:
                data_filled[idxs] = fill
                mask[idxs] = True
            elif isinstance(val, bool) and dtype is None:
                # If we see a bool and dtype not specified then assume bool for
                # the entire array. Not perfect but in most practical cases OK.
                # Unfortunately numpy types [False, 0] as int, not bool (and
                # [False, np.ma.masked] => array([0.0, np.nan])).
                dtype = bool

        # If no dtype is provided then need to convert back to list so np.array
        # does type autodetection.
        if dtype is None:
            data_filled = data_filled.tolist()

        # Use np.array first to convert `data` to ndarray (fast) and then make
        # masked array from an ndarray with mask (fast) instead of from `data`.
        np_data = np.ma.array(np.array(data_filled, dtype=dtype), mask=mask)

    return np_data
```
### 4 - astropy/table/table.py:

Start line: 3707, End line: 3761

```python
class Table:

    def to_pandas(self, index=None, use_nullable_int=True):
        # ... other code

        tbl = _encode_mixins(self)

        badcols = [name for name, col in self.columns.items() if len(col.shape) > 1]
        if badcols:
            raise ValueError(
                f'Cannot convert a table with multidimensional columns to a '
                f'pandas DataFrame. Offending columns are: {badcols}\n'
                f'One can filter out such columns using:\n'
                f'names = [name for name in tbl.colnames if len(tbl[name].shape) <= 1]\n'
                f'tbl[names].to_pandas(...)')

        out = OrderedDict()

        for name, column in tbl.columns.items():
            if getattr(column.dtype, 'isnative', True):
                out[name] = column
            else:
                out[name] = column.data.byteswap().newbyteorder('=')

            if isinstance(column, MaskedColumn) and np.any(column.mask):
                if column.dtype.kind in ['i', 'u']:
                    pd_dtype = column.dtype.name
                    if use_nullable_int:
                        # Convert int64 to Int64, uint32 to UInt32, etc for nullable types
                        pd_dtype = pd_dtype.replace('i', 'I').replace('u', 'U')
                    out[name] = Series(out[name], dtype=pd_dtype)

                    # If pandas is older than 0.24 the type may have turned to float
                    if column.dtype.kind != out[name].dtype.kind:
                        warnings.warn(
                            f"converted column '{name}' from {column.dtype} to {out[name].dtype}",
                            TableReplaceWarning, stacklevel=3)
                elif column.dtype.kind not in ['f', 'c']:
                    out[name] = column.astype(object).filled(np.nan)

        kwargs = {}

        if index:
            idx = out.pop(index)

            kwargs['index'] = idx

            # We add the table index to Series inputs (MaskedColumn with int values) to override
            # its default RangeIndex, see #11432
            for v in out.values():
                if isinstance(v, Series):
                    v.index = idx

        df = DataFrame(out, **kwargs)
        if index:
            # Explicitly set the pandas DataFrame index to the original table
            # index name.
            df.index.name = idx.info.name

        return df
```
### 5 - astropy/table/row.py:

Start line: 61, End line: 116

```python
class Row:

    def __setitem__(self, item, val):
        if self._table._is_list_or_tuple_of_str(item):
            self._table._set_row(self._index, colnames=item, vals=val)
        else:
            self._table.columns[item][self._index] = val

    def _ipython_key_completions_(self):
        return self.colnames

    def __eq__(self, other):
        if self._table.masked:
            # Sent bug report to numpy-discussion group on 2012-Oct-21, subject:
            # "Comparing rows in a structured masked array raises exception"
            # No response, so this is still unresolved.
            raise ValueError('Unable to compare rows for masked table due to numpy.ma bug')
        return self.as_void() == other

    def __ne__(self, other):
        if self._table.masked:
            raise ValueError('Unable to compare rows for masked table due to numpy.ma bug')
        return self.as_void() != other

    def __array__(self, dtype=None):
        """Support converting Row to np.array via np.array(table).

        Coercion to a different dtype via np.array(table, dtype) is not
        supported and will raise a ValueError.

        If the parent table is masked then the mask information is dropped.
        """
        if dtype is not None:
            raise ValueError('Datatype coercion is not allowed')

        return np.asarray(self.as_void())

    def __len__(self):
        return len(self._table.columns)

    def __iter__(self):
        index = self._index
        for col in self._table.columns.values():
            yield col[index]

    def keys(self):
        return self._table.columns.keys()

    def values(self):
        return self.__iter__()

    @property
    def table(self):
        return self._table

    @property
    def index(self):
        return self._index
```
### 6 - astropy/table/meta.py:

Start line: 159, End line: 218

```python
def _get_variable_length_array_shape(col):
    """Check if object-type ``col`` is really a variable length list.

    That is true if the object consists purely of list of nested lists, where
    the shape of every item can be represented as (m, n, ..., *) where the (m,
    n, ...) are constant and only the lists in the last axis have variable
    shape. If so the returned value of shape will be a tuple in the form (m, n,
    ..., None).

    If ``col`` is a variable length array then the return ``dtype`` corresponds
    to the type found by numpy for all the individual values. Otherwise it will
    be ``np.dtype(object)``.

    Parameters
    ==========
    col : column-like
        Input table column, assumed to be object-type

    Returns
    =======
    shape : tuple
        Inferred variable length shape or None
    dtype : np.dtype
        Numpy dtype that applies to col
    """
    class ConvertError(ValueError):
        """Local conversion error used below"""

    # Numpy types supported as variable-length arrays
    np_classes = (np.floating, np.integer, np.bool_, np.unicode_)

    try:
        if len(col) == 0 or not all(isinstance(val, np.ndarray) for val in col):
            raise ConvertError
        dtype = col[0].dtype
        shape = col[0].shape[:-1]
        for val in col:
            if not issubclass(val.dtype.type, np_classes) or val.shape[:-1] != shape:
                raise ConvertError
            dtype = np.promote_types(dtype, val.dtype)
        shape = shape + (None,)

    except ConvertError:
        # `col` is not a variable length array, return shape and dtype to
        #  the original. Note that this function is only called if
        #  col.shape[1:] was () and col.info.dtype is object.
        dtype = col.info.dtype
        shape = ()

    return shape, dtype


def _get_datatype_from_dtype(dtype):
    """Return string version of ``dtype`` for writing to ECSV ``datatype``"""
    datatype = dtype.name
    if datatype.startswith(('bytes', 'str')):
        datatype = 'string'
    if datatype.endswith('_'):
        datatype = datatype[:-1]  # string_ and bool_ lose the final _ for ECSV
    return datatype
```
### 7 - astropy/table/table.py:

Start line: 1086, End line: 1096

```python
class Table:

    def _check_names_dtype(self, names, dtype, n_cols):
        """Make sure that names and dtype are both iterable and have
        the same length as data.
        """
        for inp_list, inp_str in ((dtype, 'dtype'), (names, 'names')):
            if not isiterable(inp_list):
                raise ValueError(f'{inp_str} must be a list or None')

        if len(names) != n_cols or len(dtype) != n_cols:
            raise ValueError(
                'Arguments "names" and "dtype" must match number of columns')
```
### 8 - astropy/table/operations.py:

Start line: 961, End line: 973

```python
def common_dtype(cols):
    """
    Use numpy to find the common dtype for a list of columns.

    Only allow columns within the following fundamental numpy data types:
    np.bool_, np.object_, np.number, np.character, np.void
    """
    try:
        return metadata.common_dtype(cols)
    except metadata.MergeConflictError as err:
        tme = TableMergeError(f'Columns have incompatible types {err._incompat_types}')
        tme._incompat_types = err._incompat_types
        raise tme from err
```
### 9 - astropy/table/table.py:

Start line: 1252, End line: 1317

```python
class Table:

    def _convert_data_to_col(self, data, copy=True, default_name=None, dtype=None, name=None):
        # ... other code

        if isinstance(data, Column):
            # If self.ColumnClass is a subclass of col, then "upgrade" to ColumnClass,
            # otherwise just use the original class.  The most common case is a
            # table with masked=True and ColumnClass=MaskedColumn.  Then a Column
            # gets upgraded to MaskedColumn, but the converse (pre-4.0) behavior
            # of downgrading from MaskedColumn to Column (for non-masked table)
            # does not happen.
            col_cls = self._get_col_cls_for_table(data)

        elif data_is_mixin:
            # Copy the mixin column attributes if they exist since the copy below
            # may not get this attribute.
            col = col_copy(data, copy_indices=self._init_indices) if copy else data
            col.info.name = name
            return col

        elif data0_is_mixin:
            # Handle case of a sequence of a mixin, e.g. [1*u.m, 2*u.m].
            try:
                col = data[0].__class__(data)
                col.info.name = name
                return col
            except Exception:
                # If that didn't work for some reason, just turn it into np.array of object
                data = np.array(data, dtype=object)
                col_cls = self.ColumnClass

        elif isinstance(data, (np.ma.MaskedArray, Masked)):
            # Require that col_cls be a subclass of MaskedColumn, remembering
            # that ColumnClass could be a user-defined subclass (though more-likely
            # could be MaskedColumn).
            col_cls = masked_col_cls

        elif data is None:
            # Special case for data passed as the None object (for broadcasting
            # to an object column). Need to turn data into numpy `None` scalar
            # object, otherwise `Column` interprets data=None as no data instead
            # of a object column of `None`.
            data = np.array(None)
            col_cls = self.ColumnClass

        elif not hasattr(data, 'dtype'):
            # `data` is none of the above, convert to numpy array or MaskedArray
            # assuming only that it is a scalar or sequence or N-d nested
            # sequence. This function is relatively intricate and tries to
            # maintain performance for common cases while handling things like
            # list input with embedded np.ma.masked entries. If `data` is a
            # scalar then it gets returned unchanged so the original object gets
            # passed to `Column` later.
            data = _convert_sequence_data_to_array(data, dtype)
            copy = False  # Already made a copy above
            col_cls = masked_col_cls if isinstance(data, np.ma.MaskedArray) else self.ColumnClass

        else:
            col_cls = self.ColumnClass

        try:
            col = col_cls(name=name, data=data, dtype=dtype,
                          copy=copy, copy_indices=self._init_indices)
        except Exception:
            # Broad exception class since we don't know what might go wrong
            raise ValueError('unable to convert data to Column for Table')

        col = self._convert_col_for_table(col)

        return col
```
### 10 - astropy/table/column.py:

Start line: 149, End line: 216

```python
def _convert_sequence_data_to_array(data, dtype=None):
    """Convert N-d sequence-like data to ndarray or MaskedArray.

    This is the core function for converting Python lists or list of lists to a
    numpy array. This handles embedded np.ma.masked constants in ``data`` along
    with the special case of an homogeneous list of MaskedArray elements.

    Considerations:

    - np.ma.array is about 50 times slower than np.array for list input. This
      function avoids using np.ma.array on list input.
    - np.array emits a UserWarning for embedded np.ma.masked, but only for int
      or float inputs. For those it converts to np.nan and forces float dtype.
      For other types np.array is inconsistent, for instance converting
      np.ma.masked to "0.0" for str types.
    - Searching in pure Python for np.ma.masked in ``data`` is comparable in
      speed to calling ``np.array(data)``.
    - This function may end up making two additional copies of input ``data``.

    Parameters
    ----------
    data : N-d sequence
        Input data, typically list or list of lists
    dtype : None or dtype-like
        Output datatype (None lets np.array choose)

    Returns
    -------
    np_data : np.ndarray or np.ma.MaskedArray

    """
    np_ma_masked = np.ma.masked  # Avoid repeated lookups of this object

    # Special case of an homogeneous list of MaskedArray elements (see #8977).
    # np.ma.masked is an instance of MaskedArray, so exclude those values.
    if (hasattr(data, '__len__')
        and len(data) > 0
        and all(isinstance(val, np.ma.MaskedArray)
                and val is not np_ma_masked for val in data)):
        np_data = np.ma.array(data, dtype=dtype)
        return np_data

    # First convert data to a plain ndarray. If there are instances of np.ma.masked
    # in the data this will issue a warning for int and float.
    with warnings.catch_warnings(record=True) as warns:
        # Ensure this warning from numpy is always enabled and that it is not
        # converted to an error (which can happen during pytest).
        warnings.filterwarnings('always', category=UserWarning,
                                message='.*converting a masked element.*')
        # FutureWarning in numpy 1.21. See https://github.com/astropy/astropy/issues/11291
        # and https://github.com/numpy/numpy/issues/18425.
        warnings.filterwarnings('always', category=FutureWarning,
                                message='.*Promotion of numbers and bools to strings.*')
        try:
            np_data = np.array(data, dtype=dtype)
        except np.ma.MaskError:
            # Catches case of dtype=int with masked values, instead let it
            # convert to float
            np_data = np.array(data)
        except Exception:
            # Conversion failed for some reason, e.g. [2, 1*u.m] gives TypeError in Quantity.
            # First try to interpret the data as Quantity. If that still fails then fall
            # through to object
            try:
                np_data = Quantity(data, dtype)
            except Exception:
                dtype = object
                np_data = np.array(data, dtype=dtype)
    # ... other code
```
### 12 - astropy/table/table.py:

Start line: 3850, End line: 3898

```python
class Table:

    @classmethod
    def from_pandas(cls, dataframe, index=False, units=None):
        # ... other code

        for name, column, data, mask, unit in zip(names, columns, datas, masks, units):

            if column.dtype.kind in ['u', 'i'] and np.any(mask):
                # Special-case support for pandas nullable int
                np_dtype = str(column.dtype).lower()
                data = np.zeros(shape=column.shape, dtype=np_dtype)
                data[~mask] = column[~mask]
                out[name] = MaskedColumn(data=data, name=name, mask=mask, unit=unit, copy=False)
                continue

            if data.dtype.kind == 'O':
                # If all elements of an object array are string-like or np.nan
                # then coerce back to a native numpy str/unicode array.
                string_types = (str, bytes)
                nan = np.nan
                if all(isinstance(x, string_types) or x is nan for x in data):
                    # Force any missing (null) values to b''.  Numpy will
                    # upcast to str/unicode as needed.
                    data[mask] = b''

                    # When the numpy object array is represented as a list then
                    # numpy initializes to the correct string or unicode type.
                    data = np.array([x for x in data])

            # Numpy datetime64
            if data.dtype.kind == 'M':
                from astropy.time import Time
                out[name] = Time(data, format='datetime64')
                if np.any(mask):
                    out[name][mask] = np.ma.masked
                out[name].format = 'isot'

            # Numpy timedelta64
            elif data.dtype.kind == 'm':
                from astropy.time import TimeDelta
                data_sec = data.astype('timedelta64[ns]').astype(np.float64) / 1e9
                out[name] = TimeDelta(data_sec, format='sec')
                if np.any(mask):
                    out[name][mask] = np.ma.masked

            else:
                if np.any(mask):
                    out[name] = MaskedColumn(data=data, name=name, mask=mask, unit=unit)
                else:
                    out[name] = Column(data=data, name=name, unit=unit)

        return cls(out)

    info = TableInfo()
```
### 13 - astropy/table/table.py:

Start line: 2710, End line: 2730

```python
class Table:

    def convert_bytestring_to_unicode(self):
        """
        Convert bytestring columns (dtype.kind='S') to unicode (dtype.kind='U')
        using UTF-8 encoding.

        Internally this changes string columns to represent each character
        in the string with a 4-byte UCS-4 equivalent, so it is inefficient
        for memory but allows scripts to manipulate string arrays with
        natural syntax.
        """
        self._convert_string_dtype('S', 'U', np.char.decode)

    def convert_unicode_to_bytestring(self):
        """
        Convert unicode columns (dtype.kind='U') to bytestring (dtype.kind='S')
        using UTF-8 encoding.

        When exporting a unicode string array to a file, it may be desirable
        to encode unicode columns as bytestrings.
        """
        self._convert_string_dtype('U', 'S', np.char.encode)
```
### 14 - astropy/table/table.py:

Start line: 1179, End line: 1250

```python
class Table:

    def _convert_data_to_col(self, data, copy=True, default_name=None, dtype=None, name=None):
        """
        Convert any allowed sequence data ``col`` to a column object that can be used
        directly in the self.columns dict.  This could be a Column, MaskedColumn,
        or mixin column.

        The final column name is determined by::

            name or data.info.name or def_name

        If ``data`` has no ``info`` then ``name = name or def_name``.

        The behavior of ``copy`` for Column objects is:
        - copy=True: new class instance with a copy of data and deep copy of meta
        - copy=False: new class instance with same data and a key-only copy of meta

        For mixin columns:
        - copy=True: new class instance with copy of data and deep copy of meta
        - copy=False: original instance (no copy at all)

        Parameters
        ----------
        data : object (column-like sequence)
            Input column data
        copy : bool
            Make a copy
        default_name : str
            Default name
        dtype : np.dtype or None
            Data dtype
        name : str or None
            Column name

        Returns
        -------
        col : Column, MaskedColumn, mixin-column type
            Object that can be used as a column in self
        """

        data_is_mixin = self._is_mixin_for_table(data)
        masked_col_cls = (self.ColumnClass
                          if issubclass(self.ColumnClass, self.MaskedColumn)
                          else self.MaskedColumn)

        try:
            data0_is_mixin = self._is_mixin_for_table(data[0])
        except Exception:
            # Need broad exception, cannot predict what data[0] raises for arbitrary data
            data0_is_mixin = False

        # If the data is not an instance of Column or a mixin class, we can
        # check the registry of mixin 'handlers' to see if the column can be
        # converted to a mixin class
        if (handler := get_mixin_handler(data)) is not None:
            original_data = data
            data = handler(data)
            if not (data_is_mixin := self._is_mixin_for_table(data)):
                fully_qualified_name = (original_data.__class__.__module__ + '.'
                                        + original_data.__class__.__name__)
                raise TypeError('Mixin handler for object of type '
                                f'{fully_qualified_name} '
                                'did not return a valid mixin column')

        # Get the final column name using precedence.  Some objects may not
        # have an info attribute. Also avoid creating info as a side effect.
        if not name:
            if isinstance(data, Column):
                name = data.name or default_name
            elif 'info' in getattr(data, '__dict__', ()):
                name = data.info.name or default_name
            else:
                name = default_name
        # ... other code
```
### 15 - astropy/table/table.py:

Start line: 2, End line: 75

```python
from .index import SlicedIndex, TableIndices, TableLoc, TableILoc, TableLocIndices

import sys
from collections import OrderedDict, defaultdict
from collections.abc import Mapping
import warnings
from copy import deepcopy
import types
import itertools
import weakref

import numpy as np
from numpy import ma

from astropy import log
from astropy.units import Quantity, QuantityInfo
from astropy.utils import isiterable, ShapedLikeNDArray
from astropy.utils.console import color_print
from astropy.utils.exceptions import AstropyUserWarning
from astropy.utils.masked import Masked
from astropy.utils.metadata import MetaData, MetaAttribute
from astropy.utils.data_info import BaseColumnInfo, MixinInfo, DataInfo
from astropy.utils.decorators import format_doc
from astropy.io.registry import UnifiedReadWriteMethod

from . import groups
from .pprint import TableFormatter
from .column import (BaseColumn, Column, MaskedColumn, _auto_names, FalseArray,
                     col_copy, _convert_sequence_data_to_array)
from .row import Row
from .info import TableInfo
from .index import Index, _IndexModeContext, get_index
from .connect import TableRead, TableWrite
from .ndarray_mixin import NdarrayMixin
from .mixins.registry import get_mixin_handler
from . import conf


_implementation_notes = """
This string has informal notes concerning Table implementation for developers.

Things to remember:

- Table has customizable attributes ColumnClass, Column, MaskedColumn.
  Table.Column is normally just column.Column (same w/ MaskedColumn)
  but in theory they can be different.  Table.ColumnClass is the default
  class used to create new non-mixin columns, and this is a function of
  the Table.masked attribute.  Column creation / manipulation in a Table
  needs to respect these.

- Column objects that get inserted into the Table.columns attribute must
  have the info.parent_table attribute set correctly.  Beware just dropping
  an object into the columns dict since an existing column may
  be part of another Table and have parent_table set to point at that
  table.  Dropping that column into `columns` of this Table will cause
  a problem for the old one so the column object needs to be copied (but
  not necessarily the data).

  Currently replace_column is always making a copy of both object and
  data if parent_table is set.  This could be improved but requires a
  generic way to copy a mixin object but not the data.

- Be aware of column objects that have indices set.

- `cls.ColumnClass` is a property that effectively uses the `masked` attribute
  to choose either `cls.Column` or `cls.MaskedColumn`.
"""

__doctest_skip__ = ['Table.read', 'Table.write', 'Table._read',
                    'Table.convert_bytestring_to_unicode',
                    'Table.convert_unicode_to_bytestring',
                    ]

__doctest_requires__ = {'*pandas': ['pandas>=1.1']}
```
### 16 - astropy/table/table.py:

Start line: 1870, End line: 1898

```python
class Table:

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.columns[item]
        elif isinstance(item, (int, np.integer)):
            return self.Row(self, item)
        elif (isinstance(item, np.ndarray) and item.shape == () and item.dtype.kind == 'i'):
            return self.Row(self, item.item())
        elif self._is_list_or_tuple_of_str(item):
            out = self.__class__([self[x] for x in item],
                                 copy_indices=self._copy_indices)
            out._groups = groups.TableGroups(out, indices=self.groups._indices,
                                             keys=self.groups._keys)
            out.meta = self.meta.copy()  # Shallow copy for meta
            return out
        elif ((isinstance(item, np.ndarray) and item.size == 0)
              or (isinstance(item, (tuple, list)) and not item)):
            # If item is an empty array/list/tuple then return the table with no rows
            return self._new_from_slice([])
        elif (isinstance(item, slice)
              or isinstance(item, np.ndarray)
              or isinstance(item, list)
              or isinstance(item, tuple) and all(isinstance(x, np.ndarray)
                                                 for x in item)):
            # here for the many ways to give a slice; a tuple of ndarray
            # is produced by np.where, as in t[np.where(t['a'] > 2)]
            # For all, a new table is constructed with slice of all columns
            return self._new_from_slice(item)
        else:
            raise ValueError(f'Illegal type {type(item)} for table item access')
```
### 22 - astropy/table/table.py:

Start line: 2680, End line: 2708

```python
class Table:

    def _convert_string_dtype(self, in_kind, out_kind, encode_decode_func):
        """
        Convert string-like columns to/from bytestring and unicode (internal only).

        Parameters
        ----------
        in_kind : str
            Input dtype.kind
        out_kind : str
            Output dtype.kind
        """

        for col in self.itercols():
            if col.dtype.kind == in_kind:
                try:
                    # This requires ASCII and is faster by a factor of up to ~8, so
                    # try that first.
                    newcol = col.__class__(col, dtype=out_kind)
                except (UnicodeEncodeError, UnicodeDecodeError):
                    newcol = col.__class__(encode_decode_func(col, 'utf-8'))

                    # Quasi-manually copy info attributes.  Unfortunately
                    # DataInfo.__set__ does not do the right thing in this case
                    # so newcol.info = col.info does not get the old info attributes.
                    for attr in col.info.attr_names - col.info._attrs_no_copy - {'dtype'}:
                        value = deepcopy(getattr(col.info, attr))
                        setattr(newcol.info, attr, value)

                self[col.name] = newcol
```
### 23 - astropy/table/table.py:

Start line: 2161, End line: 2214

```python
class Table:

    def add_column(self, col, index=None, name=None, rename_duplicate=False, copy=True,
                   default_name=None):
        if default_name is None:
            default_name = f'col{len(self.columns)}'

        # Convert col data to acceptable object for insertion into self.columns.
        # Note that along with the lines above and below, this allows broadcasting
        # of scalars to the correct shape for adding to table.
        col = self._convert_data_to_col(col, name=name, copy=copy,
                                        default_name=default_name)

        # Assigning a scalar column to an empty table should result in an
        # exception (see #3811).
        if col.shape == () and len(self) == 0:
            raise TypeError('Empty table cannot have column set to scalar value')
        # Make col data shape correct for scalars.  The second test is to allow
        # broadcasting an N-d element to a column, e.g. t['new'] = [[1, 2]].
        elif (col.shape == () or col.shape[0] == 1) and len(self) > 0:
            new_shape = (len(self),) + getattr(col, 'shape', ())[1:]
            if isinstance(col, np.ndarray):
                col = np.broadcast_to(col, shape=new_shape,
                                      subok=True)
            elif isinstance(col, ShapedLikeNDArray):
                col = col._apply(np.broadcast_to, shape=new_shape,
                                 subok=True)

            # broadcast_to() results in a read-only array.  Apparently it only changes
            # the view to look like the broadcasted array.  So copy.
            col = col_copy(col)

        name = col.info.name

        # Ensure that new column is the right length
        if len(self.columns) > 0 and len(col) != len(self):
            raise ValueError('Inconsistent data column lengths')

        if rename_duplicate:
            orig_name = name
            i = 1
            while name in self.columns:
                # Iterate until a unique name is found
                name = orig_name + '_' + str(i)
                i += 1
            col.info.name = name

        # Set col parent_table weakref and ensure col has mask attribute if table.masked
        self._set_col_parent_table_and_mask(col)

        # Add new column as last column
        self.columns[name] = col

        if index is not None:
            # Move the other cols to the right of the new one
            move_names = self.colnames[index:-1]
            for move_name in move_names:
                self.columns.move_to_end(move_name, last=True)
```
### 24 - astropy/table/table.py:

Start line: 3946, End line: 3966

```python
class QTable(Table):

    def _convert_col_for_table(self, col):
        if isinstance(col, Column) and getattr(col, 'unit', None) is not None:
            # We need to turn the column into a quantity; use subok=True to allow
            # Quantity subclasses identified in the unit (such as u.mag()).
            q_cls = Masked(Quantity) if isinstance(col, MaskedColumn) else Quantity
            try:
                qcol = q_cls(col.data, col.unit, copy=False, subok=True)
            except Exception as exc:
                warnings.warn(f"column {col.info.name} has a unit but is kept as "
                              f"a {col.__class__.__name__} as an attempt to "
                              f"convert it to Quantity failed with:\n{exc!r}",
                              AstropyUserWarning)
            else:
                qcol.info = col.info
                qcol.info.indices = col.info.indices
                col = qcol
        else:
            col = super()._convert_col_for_table(col)

        return col
```
### 27 - astropy/table/table.py:

Start line: 1900, End line: 1950

```python
class Table:

    def __setitem__(self, item, value):
        # If the item is a string then it must be the name of a column.
        # If that column doesn't already exist then create it now.
        if isinstance(item, str) and item not in self.colnames:
            self.add_column(value, name=item, copy=True)

        else:
            n_cols = len(self.columns)

            if isinstance(item, str):
                # Set an existing column by first trying to replace, and if
                # this fails do an in-place update.  See definition of mask
                # property for discussion of the _setitem_inplace attribute.
                if (not getattr(self, '_setitem_inplace', False)
                        and not conf.replace_inplace):
                    try:
                        self._replace_column_warnings(item, value)
                        return
                    except Exception:
                        pass
                self.columns[item][:] = value

            elif isinstance(item, (int, np.integer)):
                self._set_row(idx=item, colnames=self.colnames, vals=value)

            elif (isinstance(item, slice)
                  or isinstance(item, np.ndarray)
                  or isinstance(item, list)
                  or (isinstance(item, tuple)  # output from np.where
                      and all(isinstance(x, np.ndarray) for x in item))):

                if isinstance(value, Table):
                    vals = (col for col in value.columns.values())

                elif isinstance(value, np.ndarray) and value.dtype.names:
                    vals = (value[name] for name in value.dtype.names)

                elif np.isscalar(value):
                    vals = itertools.repeat(value, n_cols)

                else:  # Assume this is an iterable that will work
                    if len(value) != n_cols:
                        raise ValueError('Right side value needs {} elements (one for each column)'
                                         .format(n_cols))
                    vals = value

                for col, val in zip(self.columns.values(), vals):
                    col[item] = val

            else:
                raise ValueError(f'Illegal type {type(item)} for table item access')
```
### 28 - astropy/table/table.py:

Start line: 1952, End line: 1966

```python
class Table:

    def __delitem__(self, item):
        if isinstance(item, str):
            self.remove_column(item)
        elif isinstance(item, (int, np.integer)):
            self.remove_row(item)
        elif (isinstance(item, (list, tuple, np.ndarray))
              and all(isinstance(x, str) for x in item)):
            self.remove_columns(item)
        elif (isinstance(item, (list, np.ndarray))
              and np.asarray(item).dtype.kind == 'i'):
            self.remove_rows(item)
        elif isinstance(item, slice):
            self.remove_rows(item)
        else:
            raise IndexError('illegal key or index value')
```
### 30 - astropy/table/table.py:

Start line: 1319, End line: 1335

```python
class Table:

    def _init_from_ndarray(self, data, names, dtype, n_cols, copy):
        """Initialize table from an ndarray structured array"""

        data_names = data.dtype.names or _auto_names(n_cols)
        struct = data.dtype.names is not None
        names = [name or data_names[i] for i, name in enumerate(names)]

        cols = ([data[name] for name in data_names] if struct else
                [data[:, i] for i in range(n_cols)])

        self._init_from_list(cols, names, dtype, n_cols, copy)

    def _init_from_dict(self, data, names, dtype, n_cols, copy):
        """Initialize table from a dictionary of columns"""

        data_list = [data[name] for name in names]
        self._init_from_list(data_list, names, dtype, n_cols, copy)
```
### 32 - astropy/table/table.py:

Start line: 3656, End line: 3705

```python
class Table:

    def to_pandas(self, index=None, use_nullable_int=True):
        # ... other code

        if index is not False:
            if index in (None, True):
                # Default is to use the table primary key if available and a single column
                if self.primary_key and len(self.primary_key) == 1:
                    index = self.primary_key[0]
                else:
                    index = False
            else:
                if index not in self.colnames:
                    raise ValueError('index must be None, False, True or a table '
                                     'column name')

        def _encode_mixins(tbl):
            """Encode a Table ``tbl`` that may have mixin columns to a Table with only
            astropy Columns + appropriate meta-data to allow subsequent decoding.
            """
            from . import serialize
            from astropy.time import TimeBase, TimeDelta

            # Convert any Time or TimeDelta columns and pay attention to masking
            time_cols = [col for col in tbl.itercols() if isinstance(col, TimeBase)]
            if time_cols:

                # Make a light copy of table and clear any indices
                new_cols = []
                for col in tbl.itercols():
                    new_col = col_copy(col, copy_indices=False) if col.info.indices else col
                    new_cols.append(new_col)
                tbl = tbl.__class__(new_cols, copy=False)

                # Certain subclasses (e.g. TimeSeries) may generate new indices on
                # table creation, so make sure there are no indices on the table.
                for col in tbl.itercols():
                    col.info.indices.clear()

                for col in time_cols:
                    if isinstance(col, TimeDelta):
                        # Convert to nanoseconds (matches astropy datetime64 support)
                        new_col = (col.sec * 1e9).astype('timedelta64[ns]')
                        nat = np.timedelta64('NaT')
                    else:
                        new_col = col.datetime64.copy()
                        nat = np.datetime64('NaT')
                    if col.masked:
                        new_col[col.mask] = nat
                    tbl[col.info.name] = new_col

            # Convert the table to one with no mixins, only Column objects.
            encode_tbl = serialize.represent_mixins_as_columns(tbl)
            return encode_tbl
        # ... other code
```
### 38 - astropy/table/table.py:

Start line: 542, End line: 601

```python
class Table:
    """A class to represent tables of heterogeneous data.

    `~astropy.table.Table` provides a class for heterogeneous tabular data.
    A key enhancement provided by the `~astropy.table.Table` class over
    e.g. a `numpy` structured array is the ability to easily modify the
    structure of the table by adding or removing columns, or adding new
    rows of data.  In addition table and column metadata are fully supported.

    `~astropy.table.Table` differs from `~astropy.nddata.NDData` by the
    assumption that the input data consists of columns of homogeneous data,
    where each column has a unique identifier and may contain additional
    metadata such as the data unit, format, and description.

    See also: https://docs.astropy.org/en/stable/table/

    Parameters
    ----------
    data : numpy ndarray, dict, list, table-like object, optional
        Data to initialize table.
    masked : bool, optional
        Specify whether the table is masked.
    names : list, optional
        Specify column names.
    dtype : list, optional
        Specify column data types.
    meta : dict, optional
        Metadata associated with the table.
    copy : bool, optional
        Copy the input data. If the input is a Table the ``meta`` is always
        copied regardless of the ``copy`` parameter.
        Default is True.
    rows : numpy ndarray, list of list, optional
        Row-oriented data for table instead of ``data`` argument.
    copy_indices : bool, optional
        Copy any indices in the input data. Default is True.
    units : list, dict, optional
        List or dict of units to apply to columns.
    descriptions : list, dict, optional
        List or dict of descriptions to apply to columns.
    **kwargs : dict, optional
        Additional keyword args when converting table-like object.
    """

    meta = MetaData(copy=False)

    # Define class attributes for core container objects to allow for subclass
    # customization.
    Row = Row
    Column = Column
    MaskedColumn = MaskedColumn
    TableColumns = TableColumns
    TableFormatter = TableFormatter

    # Unified I/O read and write methods from .connect
    read = UnifiedReadWriteMethod(TableRead)
    write = UnifiedReadWriteMethod(TableWrite)

    pprint_exclude_names = PprintIncludeExclude()
    pprint_include_names = PprintIncludeExclude()
```
### 44 - astropy/table/table.py:

Start line: 1362, End line: 1374

```python
class Table:

    def _convert_col_for_table(self, col):
        """
        Make sure that all Column objects have correct base class for this type of
        Table.  For a base Table this most commonly means setting to
        MaskedColumn if the table is masked.  Table subclasses like QTable
        override this method.
        """
        if isinstance(col, Column) and not isinstance(col, self.ColumnClass):
            col_cls = self._get_col_cls_for_table(col)
            if col_cls is not col.__class__:
                col = col_cls(col, copy=False)

        return col
```
### 49 - astropy/table/table.py:

Start line: 603, End line: 657

```python
class Table:

    def as_array(self, keep_byteorder=False, names=None):
        """
        Return a new copy of the table in the form of a structured np.ndarray or
        np.ma.MaskedArray object (as appropriate).

        Parameters
        ----------
        keep_byteorder : bool, optional
            By default the returned array has all columns in native byte
            order.  However, if this option is `True` this preserves the
            byte order of all columns (if any are non-native).

        names : list, optional:
            List of column names to include for returned structured array.
            Default is to include all table columns.

        Returns
        -------
        table_array : array or `~numpy.ma.MaskedArray`
            Copy of table as a numpy structured array.
            ndarray for unmasked or `~numpy.ma.MaskedArray` for masked.
        """
        masked = self.masked or self.has_masked_columns or self.has_masked_values
        empty_init = ma.empty if masked else np.empty
        if len(self.columns) == 0:
            return empty_init(0, dtype=None)

        dtype = []

        cols = self.columns.values()

        if names is not None:
            cols = [col for col in cols if col.info.name in names]

        for col in cols:
            col_descr = descr(col)

            if not (col.info.dtype.isnative or keep_byteorder):
                new_dt = np.dtype(col_descr[1]).newbyteorder('=')
                col_descr = (col_descr[0], new_dt, col_descr[2])

            dtype.append(col_descr)

        data = empty_init(len(self), dtype=dtype)
        for col in cols:
            # When assigning from one array into a field of a structured array,
            # Numpy will automatically swap those columns to their destination
            # byte order where applicable
            data[col.info.name] = col

            # For masked out, masked mixin columns need to set output mask attribute.
            if masked and has_info_class(col, MixinInfo) and hasattr(col, 'mask'):
                data[col.info.name].mask = col.mask

        return data
```
### 55 - astropy/table/table.py:

Start line: 3901, End line: 3944

```python
class QTable(Table):
    """A class to represent tables of heterogeneous data.

    `~astropy.table.QTable` provides a class for heterogeneous tabular data
    which can be easily modified, for instance adding columns or new rows.

    The `~astropy.table.QTable` class is identical to `~astropy.table.Table`
    except that columns with an associated ``unit`` attribute are converted to
    `~astropy.units.Quantity` objects.

    See also:

    - https://docs.astropy.org/en/stable/table/
    - https://docs.astropy.org/en/stable/table/mixin_columns.html

    Parameters
    ----------
    data : numpy ndarray, dict, list, table-like object, optional
        Data to initialize table.
    masked : bool, optional
        Specify whether the table is masked.
    names : list, optional
        Specify column names.
    dtype : list, optional
        Specify column data types.
    meta : dict, optional
        Metadata associated with the table.
    copy : bool, optional
        Copy the input data. Default is True.
    rows : numpy ndarray, list of list, optional
        Row-oriented data for table instead of ``data`` argument.
    copy_indices : bool, optional
        Copy any indices in the input data. Default is True.
    **kwargs : dict, optional
        Additional keyword args when converting table-like object.

    """

    def _is_mixin_for_table(self, col):
        """
        Determine if ``col`` should be added to the table directly as
        a mixin column.
        """
        return has_info_class(col, MixinInfo)
```
