# astropy__astropy-14578

| **astropy/astropy** | `c748299218dcbd9e15caef558722cc04aa658fad` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | - |
| **Avg pos** | - |
| **Min pos** | - |
| **Max pos** | - |
| **Top file pos** | 3 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/astropy/io/fits/column.py b/astropy/io/fits/column.py
--- a/astropy/io/fits/column.py
+++ b/astropy/io/fits/column.py
@@ -1528,7 +1528,19 @@ def _init_from_array(self, array):
         for idx in range(len(array.dtype)):
             cname = array.dtype.names[idx]
             ftype = array.dtype.fields[cname][0]
-            format = self._col_format_cls.from_recformat(ftype)
+
+            if ftype.kind == "O":
+                dtypes = {np.array(array[cname][i]).dtype for i in range(len(array))}
+                if (len(dtypes) > 1) or (np.dtype("O") in dtypes):
+                    raise TypeError(
+                        f"Column '{cname}' contains unsupported object types or "
+                        f"mixed types: {dtypes}"
+                    )
+                ftype = dtypes.pop()
+                format = self._col_format_cls.from_recformat(ftype)
+                format = f"P{format}()"
+            else:
+                format = self._col_format_cls.from_recformat(ftype)
 
             # Determine the appropriate dimensions for items in the column
             dim = array.dtype[idx].shape[::-1]

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| astropy/io/fits/column.py | 1531 | 1531 | - | 3 | -


## Problem Statement

```
Writing a Table to FITS fails if the table contains objects
The following works fine:

\`\`\` Python
from astropy.table import Table
Table([{'col1': None}]).write('/tmp/tmp.txt', format='ascii')
\`\`\`

whereas the following fails:

\`\`\` Python
Table([{'col1': None}]).write('/tmp/tmp.fits', format='fits')
\`\`\`

with

\`\`\`
/home/gb/bin/anaconda/lib/python2.7/site-packages/astropy-0.4.dev6667-py2.7-linux-x86_64.egg/astropy/io/fits/column.pyc in _convert_record2fits(format)
   1727         output_format = repeat + NUMPY2FITS[recformat]
   1728     else:
-> 1729         raise ValueError('Illegal format %s.' % format)
   1730 
   1731     return output_format

ValueError: Illegal format object.
\`\`\`

This behaviour is seen whenever a Table contains an object, i.e. io/fits/column.py does not know how to deal with `dtype('O')`.

I wonder if we want the Table API to write objects to files by their string representation as a default, or otherwise provide a more meaningful error message?


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 astropy/io/fits/connect.py | 417 | 454| 263 | 263 | 3878 | 
| 2 | 2 astropy/io/votable/exceptions.py | 883 | 895| 153 | 416 | 17290 | 
| 3 | 2 astropy/io/fits/connect.py | 121 | 183| 758 | 1174 | 17290 | 
| 4 | **3 astropy/io/fits/column.py** | 291 | 315| 223 | 1397 | 39864 | 
| 5 | 4 astropy/table/connect.py | 83 | 131| 400 | 1797 | 40881 | 
| 6 | 4 astropy/io/fits/connect.py | 261 | 339| 650 | 2447 | 40881 | 
| 7 | 5 astropy/io/fits/hdu/table.py | 1034 | 1057| 245 | 2692 | 54310 | 
| 8 | 5 astropy/io/fits/connect.py | 184 | 260| 621 | 3313 | 54310 | 
| 9 | 5 astropy/io/fits/hdu/table.py | 1263 | 1287| 211 | 3524 | 54310 | 
| 10 | **5 astropy/io/fits/column.py** | 344 | 390| 439 | 3963 | 54310 | 
| 11 | 6 astropy/io/fits/convenience.py | 626 | 658| 238 | 4201 | 64202 | 
| 12 | 6 astropy/io/fits/hdu/table.py | 984 | 1032| 407 | 4608 | 64202 | 
| 13 | **6 astropy/io/fits/column.py** | 2523 | 2575| 544 | 5152 | 64202 | 
| 14 | 7 astropy/table/table.py | 2 | 90| 660 | 5812 | 97629 | 
| 15 | 8 astropy/io/fits/fitsrec.py | 847 | 885| 418 | 6230 | 109682 | 
| 16 | 8 astropy/io/fits/hdu/table.py | 803 | 834| 315 | 6545 | 109682 | 
| 17 | 8 astropy/io/fits/convenience.py | 556 | 624| 654 | 7199 | 109682 | 
| 18 | **8 astropy/io/fits/column.py** | 426 | 480| 514 | 7713 | 109682 | 
| 19 | 8 astropy/io/fits/hdu/table.py | 4 | 55| 337 | 8050 | 109682 | 
| 20 | 9 astropy/io/votable/tree.py | 2464 | 2483| 171 | 8221 | 139076 | 
| 21 | 9 astropy/io/fits/hdu/table.py | 1342 | 1366| 207 | 8428 | 139076 | 
| 22 | 10 astropy/io/ascii/ui.py | 922 | 996| 549 | 8977 | 147139 | 
| 23 | **10 astropy/io/fits/column.py** | 2443 | 2485| 359 | 9336 | 147139 | 
| 24 | **10 astropy/io/fits/column.py** | 2406 | 2440| 310 | 9646 | 147139 | 
| 25 | **10 astropy/io/fits/column.py** | 408 | 423| 130 | 9776 | 147139 | 
| 26 | 10 astropy/io/votable/tree.py | 2485 | 2504| 170 | 9946 | 147139 | 
| 27 | 10 astropy/io/fits/hdu/table.py | 1301 | 1340| 365 | 10311 | 147139 | 
| 28 | **10 astropy/io/fits/column.py** | 1324 | 1365| 414 | 10725 | 147139 | 
| 29 | **10 astropy/io/fits/column.py** | 249 | 288| 238 | 10963 | 147139 | 
| 30 | **10 astropy/io/fits/column.py** | 1289 | 1322| 386 | 11349 | 147139 | 
| 31 | 10 astropy/io/fits/hdu/table.py | 1288 | 1299| 157 | 11506 | 147139 | 
| 32 | 10 astropy/io/votable/tree.py | 3042 | 3086| 315 | 11821 | 147139 | 
| 33 | 10 astropy/io/votable/exceptions.py | 631 | 651| 168 | 11989 | 147139 | 
| 34 | **10 astropy/io/fits/column.py** | 897 | 964| 516 | 12505 | 147139 | 
| 35 | 11 astropy/io/ascii/core.py | 1173 | 1200| 221 | 12726 | 160978 | 
| 36 | 12 astropy/io/fits/scripts/fitsheader.py | 239 | 268| 205 | 12931 | 164602 | 
| 37 | 12 astropy/io/fits/convenience.py | 513 | 555| 521 | 13452 | 164602 | 
| 38 | 13 astropy/io/votable/connect.py | 131 | 182| 412 | 13864 | 166072 | 
| 39 | **13 astropy/io/fits/column.py** | 2488 | 2520| 239 | 14103 | 166072 | 
| 40 | 13 astropy/io/fits/hdu/table.py | 1462 | 1473| 163 | 14266 | 166072 | 
| 41 | 13 astropy/io/votable/tree.py | 3191 | 3225| 277 | 14543 | 166072 | 
| 42 | **13 astropy/io/fits/column.py** | 317 | 341| 192 | 14735 | 166072 | 
| 43 | 14 astropy/io/misc/asdf/connect.py | 59 | 119| 518 | 15253 | 167065 | 
| 44 | **14 astropy/io/fits/column.py** | 1565 | 1614| 447 | 15700 | 167065 | 
| 45 | 15 astropy/io/fits/fitstime.py | 587 | 656| 635 | 16335 | 172515 | 
| 46 | 16 astropy/io/ascii/docs.py | 97 | 190| 813 | 17148 | 174319 | 
| 47 | 16 astropy/io/votable/tree.py | 3130 | 3189| 420 | 17568 | 174319 | 
| 48 | 17 astropy/cosmology/io/table.py | 239 | 258| 131 | 17699 | 177129 | 
| 49 | 17 astropy/io/fits/hdu/table.py | 1059 | 1112| 678 | 18377 | 177129 | 
| 50 | **17 astropy/io/fits/column.py** | 1367 | 1434| 647 | 19024 | 177129 | 
| 51 | 17 astropy/io/ascii/ui.py | 571 | 650| 645 | 19669 | 177129 | 
| 52 | 17 astropy/io/fits/convenience.py | 941 | 998| 497 | 20166 | 177129 | 
| 53 | 17 astropy/table/table.py | 2382 | 2434| 568 | 20734 | 177129 | 
| 54 | 18 astropy/io/ascii/ecsv.py | 233 | 259| 228 | 20962 | 181361 | 
| 55 | 18 astropy/io/votable/exceptions.py | 541 | 567| 254 | 21216 | 181361 | 
| 56 | 19 astropy/io/misc/hdf5.py | 209 | 271| 582 | 21798 | 184643 | 
| 57 | 19 astropy/io/fits/hdu/table.py | 542 | 569| 342 | 22140 | 184643 | 
| 58 | 19 astropy/table/table.py | 3973 | 4033| 527 | 22667 | 184643 | 
| 59 | 19 astropy/io/fits/scripts/fitsheader.py | 296 | 329| 226 | 22893 | 184643 | 
| 60 | 19 astropy/io/fits/hdu/table.py | 1114 | 1184| 589 | 23482 | 184643 | 
| 61 | 19 astropy/io/fits/hdu/table.py | 1546 | 1613| 473 | 23955 | 184643 | 
| 62 | 19 astropy/table/table.py | 4220 | 4242| 223 | 24178 | 184643 | 
| 63 | 19 astropy/table/table.py | 1322 | 1398| 753 | 24931 | 184643 | 
| 64 | 19 astropy/table/table.py | 1148 | 1159| 124 | 25055 | 184643 | 
| 65 | **19 astropy/io/fits/column.py** | 860 | 895| 303 | 25358 | 184643 | 
| 66 | 19 astropy/io/misc/hdf5.py | 273 | 368| 778 | 26136 | 184643 | 
| 67 | 19 astropy/io/ascii/docs.py | 1 | 96| 991 | 27127 | 184643 | 
| 68 | 19 astropy/io/fits/convenience.py | 462 | 512| 401 | 27528 | 184643 | 
| 69 | 19 astropy/table/table.py | 297 | 318| 200 | 27728 | 184643 | 
| 70 | 19 astropy/io/fits/hdu/table.py | 1475 | 1510| 308 | 28036 | 184643 | 
| 71 | **19 astropy/io/fits/column.py** | 3 | 85| 770 | 28806 | 184643 | 
| 72 | 20 astropy/io/misc/asdf/tags/fits/fits.py | 2 | 35| 251 | 29057 | 185407 | 
| 73 | 21 astropy/io/misc/pandas/connect.py | 85 | 118| 284 | 29341 | 186349 | 
| 74 | 21 astropy/io/fits/connect.py | 4 | 45| 302 | 29643 | 186349 | 
| 75 | **21 astropy/io/fits/column.py** | 86 | 146| 757 | 30400 | 186349 | 
| 76 | 21 astropy/io/fits/hdu/table.py | 449 | 506| 500 | 30900 | 186349 | 
| 77 | 21 astropy/io/fits/hdu/table.py | 209 | 230| 200 | 31100 | 186349 | 
| 78 | **21 astropy/io/fits/column.py** | 147 | 231| 695 | 31795 | 186349 | 
| 79 | 21 astropy/io/ascii/core.py | 1049 | 1095| 364 | 32159 | 186349 | 
| 80 | 22 astropy/table/scripts/showtable.py | 45 | 93| 332 | 32491 | 187722 | 
| 81 | 22 astropy/io/votable/tree.py | 3088 | 3128| 359 | 32850 | 187722 | 
| 82 | 22 astropy/io/votable/exceptions.py | 654 | 664| 117 | 32967 | 187722 | 


### Hint

```
Hm. I wonder if there's a place in the I/O registry for readers/writers to provide some means of listing what data formats they can accept--or at least rejecting formats that they don't accept.  Maybe something to think about as part of #962 ?

I should add--I think the current behavior is "correct"--any convention for storing arbitrary Python objects in a FITS file would be ad-hoc and not helpful.  I think it's fine that this is currently rejected.  But I agree that it should have been handled differently.

I agree with @embray that the best solution here is just to provide a more helpful error message.  In addition `io.ascii` should probably check the column dtypes and make sure they can reliably serialized.  The fact that `None` worked was a bit of an accident and as @embray said not very helpful because it doesn't round trip back to `None`.

Agreed! I wouldn't have posted the issue had there been a clear error message explaining that object X isn't supported by FITS.

We could also consider skipping unsupported columns and raising a warning.

I would be more inclined to tell the user in the exception which columns need to be removed and how to do it.  But just raising warnings doesn't always get peoples attention, e.g. in the case of processing scripts with lots of output.

Not critical for 1.0 so removing milestone (but if someone feels like implementing it in the next few days, feel free to!)

```

## Patch

```diff
diff --git a/astropy/io/fits/column.py b/astropy/io/fits/column.py
--- a/astropy/io/fits/column.py
+++ b/astropy/io/fits/column.py
@@ -1528,7 +1528,19 @@ def _init_from_array(self, array):
         for idx in range(len(array.dtype)):
             cname = array.dtype.names[idx]
             ftype = array.dtype.fields[cname][0]
-            format = self._col_format_cls.from_recformat(ftype)
+
+            if ftype.kind == "O":
+                dtypes = {np.array(array[cname][i]).dtype for i in range(len(array))}
+                if (len(dtypes) > 1) or (np.dtype("O") in dtypes):
+                    raise TypeError(
+                        f"Column '{cname}' contains unsupported object types or "
+                        f"mixed types: {dtypes}"
+                    )
+                ftype = dtypes.pop()
+                format = self._col_format_cls.from_recformat(ftype)
+                format = f"P{format}()"
+            else:
+                format = self._col_format_cls.from_recformat(ftype)
 
             # Determine the appropriate dimensions for items in the column
             dim = array.dtype[idx].shape[::-1]

```

## Test Patch

```diff
diff --git a/astropy/io/fits/tests/test_connect.py b/astropy/io/fits/tests/test_connect.py
--- a/astropy/io/fits/tests/test_connect.py
+++ b/astropy/io/fits/tests/test_connect.py
@@ -414,6 +414,61 @@ def test_mask_str_on_read(self, tmp_path):
         tab = Table.read(filename, mask_invalid=False)
         assert tab.mask is None
 
+    def test_heterogeneous_VLA_tables(self, tmp_path):
+        """
+        Check the behaviour of heterogeneous VLA object.
+        """
+        filename = tmp_path / "test_table_object.fits"
+        msg = "Column 'col1' contains unsupported object types or mixed types: "
+
+        # The column format fix the type of the arrays in the VLF object.
+        a = np.array([45, 30])
+        b = np.array([11.0, 12.0, 13])
+        var = np.array([a, b], dtype=object)
+        tab = Table({"col1": var})
+        with pytest.raises(TypeError, match=msg):
+            tab.write(filename)
+
+        # Strings in the VLF object can't be added to the table
+        a = np.array(["five", "thirty"])
+        b = np.array([11.0, 12.0, 13])
+        var = np.array([a, b], dtype=object)
+        with pytest.raises(TypeError, match=msg):
+            tab.write(filename)
+
+    def test_write_object_tables_with_unified(self, tmp_path):
+        """
+        Write objects with the unified I/O interface.
+        See https://github.com/astropy/astropy/issues/1906
+        """
+        filename = tmp_path / "test_table_object.fits"
+        msg = r"Column 'col1' contains unsupported object types or mixed types: {dtype\('O'\)}"
+        # Make a FITS table with an object column
+        tab = Table({"col1": [None]})
+        with pytest.raises(TypeError, match=msg):
+            tab.write(filename)
+
+    def test_write_VLA_tables_with_unified(self, tmp_path):
+        """
+        Write VLA objects with the unified I/O interface.
+        See https://github.com/astropy/astropy/issues/11323
+        """
+
+        filename = tmp_path / "test_table_VLA.fits"
+        # Make a FITS table with a variable-length array column
+        a = np.array([45, 30])
+        b = np.array([11, 12, 13])
+        c = np.array([45, 55, 65, 75])
+        var = np.array([a, b, c], dtype=object)
+
+        tabw = Table({"col1": var})
+        tabw.write(filename)
+
+        tab = Table.read(filename)
+        assert np.array_equal(tab[0]["col1"], np.array([45, 30]))
+        assert np.array_equal(tab[1]["col1"], np.array([11, 12, 13]))
+        assert np.array_equal(tab[2]["col1"], np.array([45, 55, 65, 75]))
+
 
 class TestMultipleHDU:
     def setup_class(self):
diff --git a/astropy/io/fits/tests/test_table.py b/astropy/io/fits/tests/test_table.py
--- a/astropy/io/fits/tests/test_table.py
+++ b/astropy/io/fits/tests/test_table.py
@@ -3313,6 +3313,31 @@ def test_multidim_VLA_tables(self):
                 hdus[1].data["test"][1], np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
             )
 
+    def test_heterogeneous_VLA_tables(self):
+        """
+        Check the behaviour of heterogeneous VLA object.
+        """
+
+        # The column format fix the type of the arrays in the VLF object.
+        a = np.array([45, 30])
+        b = np.array([11.0, 12.0, 13])
+        var = np.array([a, b], dtype=object)
+
+        c1 = fits.Column(name="var", format="PJ()", array=var)
+        hdu = fits.BinTableHDU.from_columns([c1])
+        assert hdu.data[0].array.dtype[0].subdtype[0] == "int32"
+
+        # Strings in the VLF object can't be added to the table
+        a = np.array([45, "thirty"])
+        b = np.array([11.0, 12.0, 13])
+        var = np.array([a, b], dtype=object)
+
+        c1 = fits.Column(name="var", format="PJ()", array=var)
+        with pytest.raises(
+            ValueError, match=r"invalid literal for int\(\) with base 10"
+        ):
+            fits.BinTableHDU.from_columns([c1])
+
 
 # These are tests that solely test the Column and ColDefs interfaces and
 # related functionality without directly involving full tables; currently there

```


## Code snippets

### 1 - astropy/io/fits/connect.py:

Start line: 417, End line: 454

```python
def write_table_fits(input, output, overwrite=False, append=False):
    """
    Write a Table object to a FITS file.

    Parameters
    ----------
    input : Table
        The table to write out.
    output : str
        The filename to write the table to.
    overwrite : bool
        Whether to overwrite any existing file without warning.
    append : bool
        Whether to append the table to an existing file
    """
    # Encode any mixin columns into standard Columns.
    input = _encode_mixins(input)

    table_hdu = table_to_hdu(input, character_as_bytes=True)

    # Check if output file already exists
    if isinstance(output, str) and os.path.exists(output):
        if overwrite:
            os.remove(output)
        elif not append:
            raise OSError(NOT_OVERWRITING_MSG.format(output))

    if append:
        # verify=False stops it reading and checking the existing file.
        fits_append(output, table_hdu.data, table_hdu.header, verify=False)
    else:
        table_hdu.writeto(output)


io_registry.register_reader("fits", Table, read_table_fits)
io_registry.register_writer("fits", Table, write_table_fits)
io_registry.register_identifier("fits", Table, is_fits)
```
### 2 - astropy/io/votable/exceptions.py:

Start line: 883, End line: 895

```python
class W37(UnimplementedWarning):
    """
    The 3 datatypes defined in the VOTable specification and supported by
    ``astropy.io.votable`` are ``TABLEDATA``, ``BINARY`` and ``FITS``.

    **References:** `1.1
    <http://www.ivoa.net/documents/VOTable/20040811/REC-VOTable-1.1-20040811.html#sec:data>`__,
    `1.2
    <http://www.ivoa.net/documents/VOTable/20091130/REC-VOTable-1.2.html#sec:data>`__
    """

    message_template = "Unsupported data format '{}'"
    default_args = ("x",)
```
### 3 - astropy/io/fits/connect.py:

Start line: 121, End line: 183

```python
def read_table_fits(
    input,
    hdu=None,
    astropy_native=False,
    memmap=False,
    character_as_bytes=True,
    unit_parse_strict="warn",
    mask_invalid=True,
):
    """
    Read a Table object from an FITS file.

    If the ``astropy_native`` argument is ``True``, then input FITS columns
    which are representations of an astropy core object will be converted to
    that class and stored in the ``Table`` as "mixin columns".  Currently this
    is limited to FITS columns which adhere to the FITS Time standard, in which
    case they will be converted to a `~astropy.time.Time` column in the output
    table.

    Parameters
    ----------
    input : str or file-like or compatible `astropy.io.fits` HDU object
        If a string, the filename to read the table from. If a file object, or
        a compatible HDU object, the object to extract the table from. The
        following `astropy.io.fits` HDU objects can be used as input:
        - :class:`~astropy.io.fits.hdu.table.TableHDU`
        - :class:`~astropy.io.fits.hdu.table.BinTableHDU`
        - :class:`~astropy.io.fits.hdu.table.GroupsHDU`
        - :class:`~astropy.io.fits.hdu.hdulist.HDUList`
    hdu : int or str, optional
        The HDU to read the table from.
    astropy_native : bool, optional
        Read in FITS columns as native astropy objects where possible instead
        of standard Table Column objects. Default is False.
    memmap : bool, optional
        Whether to use memory mapping, which accesses data on disk as needed. If
        you are only accessing part of the data, this is often more efficient.
        If you want to access all the values in the table, and you are able to
        fit the table in memory, you may be better off leaving memory mapping
        off. However, if your table would not fit in memory, you should set this
        to `True`.
        When set to `True` then ``mask_invalid`` is set to `False` since the
        masking would cause loading the full data array.
    character_as_bytes : bool, optional
        If `True`, string columns are stored as Numpy byte arrays (dtype ``S``)
        and are converted on-the-fly to unicode strings when accessing
        individual elements. If you need to use Numpy unicode arrays (dtype
        ``U``) internally, you should set this to `False`, but note that this
        will use more memory. If set to `False`, string columns will not be
        memory-mapped even if ``memmap`` is `True`.
    unit_parse_strict : str, optional
        Behaviour when encountering invalid column units in the FITS header.
        Default is "warn", which will emit a ``UnitsWarning`` and create a
        :class:`~astropy.units.core.UnrecognizedUnit`.
        Values are the ones allowed by the ``parse_strict`` argument of
        :class:`~astropy.units.core.Unit`: ``raise``, ``warn`` and ``silent``.
    mask_invalid : bool, optional
        By default the code masks NaNs in float columns and empty strings in
        string columns. Set this parameter to `False` to avoid the performance
        penalty of doing this masking step. The masking is always deactivated
        when using ``memmap=True`` (see above).

    """
    # ... other code
```
### 4 - astropy/io/fits/column.py:

Start line: 291, End line: 315

```python
class _ColumnFormat(_BaseColumnFormat):
    """
    Represents a FITS binary table column format.

    This is an enhancement over using a normal string for the format, since the
    repeat count, format code, and option are available as separate attributes,
    and smart comparison is used.  For example 1J == J.
    """

    def __new__(cls, format):
        self = super().__new__(cls, format)
        self.repeat, self.format, self.option = _parse_tformat(format)
        self.format = self.format.upper()
        if self.format in ("P", "Q"):
            # TODO: There should be a generic factory that returns either
            # _FormatP or _FormatQ as appropriate for a given TFORMn
            if self.format == "P":
                recformat = _FormatP.from_tform(format)
            else:
                recformat = _FormatQ.from_tform(format)
            # Format of variable length arrays
            self.p_format = recformat.format
        else:
            self.p_format = None
        return self
```
### 5 - astropy/table/connect.py:

Start line: 83, End line: 131

```python
class TableWrite(registry.UnifiedReadWrite):
    """
    Write this Table object out in the specified format.

    This function provides the Table interface to the astropy unified I/O
    layer.  This allows easily writing a file in many supported data formats
    using syntax such as::

      >>> from astropy.table import Table
      >>> dat = Table([[1, 2], [3, 4]], names=('a', 'b'))
      >>> dat.write('table.dat', format='ascii')

    Get help on the available writers for ``Table`` using the``help()`` method::

      >>> Table.write.help()  # Get help writing Table and list supported formats
      >>> Table.write.help('fits')  # Get detailed help on Table FITS writer
      >>> Table.write.list_formats()  # Print list of available formats

    The ``serialize_method`` argument is explained in the section on
    `Table serialization methods
    <https://docs.astropy.org/en/latest/io/unified.html#table-serialization-methods>`_.

    See also: https://docs.astropy.org/en/stable/io/unified.html

    Parameters
    ----------
    *args : tuple, optional
        Positional arguments passed through to data writer. If supplied the
        first argument is the output filename.
    format : str
        File format specifier.
    serialize_method : str, dict, optional
        Serialization method specifier for columns.
    **kwargs : dict, optional
        Keyword arguments passed through to data writer.

    Notes
    -----
    """

    def __init__(self, instance, cls):
        super().__init__(instance, cls, "write", registry=None)
        # uses default global registry

    def __call__(self, *args, serialize_method=None, **kwargs):
        instance = self._instance
        with serialize_method_as(instance, serialize_method):
            self.registry.write(instance, *args, **kwargs)
```
### 6 - astropy/io/fits/connect.py:

Start line: 261, End line: 339

```python
def read_table_fits(
    input,
    hdu=None,
    astropy_native=False,
    memmap=False,
    character_as_bytes=True,
    unit_parse_strict="warn",
    mask_invalid=True,
):
    # ... other code
    for col in data.columns:
        # Check if column is masked. Here, we make a guess based on the
        # presence of FITS mask values. For integer columns, this is simply
        # the null header, for float and complex, the presence of NaN, and for
        # string, empty strings.
        # Since Multi-element columns with dtypes such as '2f8' have a subdtype,
        # we should look up the type of column on that.
        masked = mask = False
        coltype = col.dtype.subdtype[0].type if col.dtype.subdtype else col.dtype.type
        if col.null is not None:
            mask = data[col.name] == col.null
            # Return a MaskedColumn even if no elements are masked so
            # we roundtrip better.
            masked = True
        elif mask_invalid and issubclass(coltype, np.inexact):
            mask = np.isnan(data[col.name])
        elif mask_invalid and issubclass(coltype, np.character):
            mask = col.array == b""

        if masked or np.any(mask):
            column = MaskedColumn(
                data=data[col.name], name=col.name, mask=mask, copy=False
            )
        else:
            column = Column(data=data[col.name], name=col.name, copy=False)

        # Copy over units
        if col.unit is not None:
            column.unit = u.Unit(
                col.unit, format="fits", parse_strict=unit_parse_strict
            )

        # Copy over display format
        if col.disp is not None:
            column.format = _fortran_to_python_format(col.disp)

        columns.append(column)

    # Create Table object
    t = Table(columns, copy=False)

    # TODO: deal properly with unsigned integers

    hdr = table.header
    if astropy_native:
        # Avoid circular imports, and also only import if necessary.
        from .fitstime import fits_to_time

        hdr = fits_to_time(hdr, t)

    for key, value, comment in hdr.cards:
        if key in ["COMMENT", "HISTORY"]:
            # Convert to io.ascii format
            if key == "COMMENT":
                key = "comments"

            if key in t.meta:
                t.meta[key].append(value)
            else:
                t.meta[key] = [value]

        elif key in t.meta:  # key is duplicate
            if isinstance(t.meta[key], list):
                t.meta[key].append(value)
            else:
                t.meta[key] = [t.meta[key], value]

        elif is_column_keyword(key) or key in REMOVE_KEYWORDS:
            pass

        else:
            t.meta[key] = value

    # TODO: implement masking

    # Decode any mixin columns that have been stored as standard Columns.
    t = _decode_mixins(t)

    return t
```
### 7 - astropy/io/fits/hdu/table.py:

Start line: 1034, End line: 1057

```python
class BinTableHDU(_TableBaseHDU):

    def _writedata_by_row(self, fileobj):
        fields = [self.data.field(idx) for idx in range(len(self.data.columns))]

        # Creating Record objects is expensive (as in
        # `for row in self.data:` so instead we just iterate over the row
        # indices and get one field at a time:
        for idx in range(len(self.data)):
            for field in fields:
                item = field[idx]
                field_width = None

                if field.dtype.kind == "U":
                    # Read the field *width* by reading past the field kind.
                    i = field.dtype.str.index(field.dtype.kind)
                    field_width = int(field.dtype.str[i + 1 :])
                    item = np.char.encode(item, "ascii")

                fileobj.writearray(item)
                if field_width is not None:
                    j = item.dtype.str.index(item.dtype.kind)
                    item_length = int(item.dtype.str[j + 1 :])
                    # Fix padding problem (see #5296).
                    padding = "\x00" * (field_width - item_length)
                    fileobj.write(padding.encode("ascii"))
```
### 8 - astropy/io/fits/connect.py:

Start line: 184, End line: 260

```python
def read_table_fits(
    input,
    hdu=None,
    astropy_native=False,
    memmap=False,
    character_as_bytes=True,
    unit_parse_strict="warn",
    mask_invalid=True,
):
    if isinstance(input, HDUList):
        # Parse all table objects
        tables = dict()
        for ihdu, hdu_item in enumerate(input):
            if isinstance(hdu_item, (TableHDU, BinTableHDU, GroupsHDU)):
                tables[ihdu] = hdu_item

        if len(tables) > 1:
            if hdu is None:
                warnings.warn(
                    "hdu= was not specified but multiple tables"
                    " are present, reading in first available"
                    f" table (hdu={first(tables)})",
                    AstropyUserWarning,
                )
                hdu = first(tables)

            # hdu might not be an integer, so we first need to convert it
            # to the correct HDU index
            hdu = input.index_of(hdu)

            if hdu in tables:
                table = tables[hdu]
            else:
                raise ValueError(f"No table found in hdu={hdu}")

        elif len(tables) == 1:
            if hdu is not None:
                msg = None
                try:
                    hdi = input.index_of(hdu)
                except KeyError:
                    msg = f"Specified hdu={hdu} not found"
                else:
                    if hdi >= len(input):
                        msg = f"Specified hdu={hdu} not found"
                    elif hdi not in tables:
                        msg = f"No table found in specified hdu={hdu}"
                if msg is not None:
                    warnings.warn(
                        f"{msg}, reading in first available table "
                        f"(hdu={first(tables)}) instead. This will"
                        " result in an error in future versions!",
                        AstropyDeprecationWarning,
                    )
            table = tables[first(tables)]

        else:
            raise ValueError("No table found")

    elif isinstance(input, (TableHDU, BinTableHDU, GroupsHDU)):
        table = input

    else:
        if memmap:
            # using memmap is not compatible with masking invalid value by
            # default so we deactivate the masking
            mask_invalid = False

        hdulist = fits_open(input, character_as_bytes=character_as_bytes, memmap=memmap)

        try:
            return read_table_fits(
                hdulist,
                hdu=hdu,
                astropy_native=astropy_native,
                unit_parse_strict=unit_parse_strict,
                mask_invalid=mask_invalid,
            )
        finally:
            hdulist.close()

    # In the loop below we access the data using data[col.name] rather than
    # col.array to make sure that the data is scaled correctly if needed.
    data = table.data

    columns = []
    # ... other code
```
### 9 - astropy/io/fits/hdu/table.py:

Start line: 1263, End line: 1287

```python
class BinTableHDU(_TableBaseHDU):

    if isinstance(load.__doc__, str):
        load.__doc__ += _tdump_file_format.replace("\n", "\n        ")

    load = classmethod(load)
    # Have to create a classmethod from this here instead of as a decorator;
    # otherwise we can't update __doc__

    def _dump_data(self, fileobj):
        """
        Write the table data in the ASCII format read by BinTableHDU.load()
        to fileobj.
        """
        if not fileobj and self._file:
            root = os.path.splitext(self._file.name)[0]
            fileobj = root + ".txt"

        close_file = False

        if isinstance(fileobj, str):
            fileobj = open(fileobj, "w")
            close_file = True

        linewriter = csv.writer(fileobj, dialect=FITSTableDumpDialect)

        # Process each row of the table and output one row at a time
        # ... other code
```
### 10 - astropy/io/fits/column.py:

Start line: 344, End line: 390

```python
class _AsciiColumnFormat(_BaseColumnFormat):
    """Similar to _ColumnFormat but specifically for columns in ASCII tables.

    The formats of ASCII table columns and binary table columns are inherently
    incompatible in FITS.  They don't support the same ranges and types of
    values, and even reuse format codes in subtly different ways.  For example
    the format code 'Iw' in ASCII columns refers to any integer whose string
    representation is at most w characters wide, so 'I' can represent
    effectively any integer that will fit in a FITS columns.  Whereas for
    binary tables 'I' very explicitly refers to a 16-bit signed integer.

    Conversions between the two column formats can be performed using the
    ``to/from_binary`` methods on this class, or the ``to/from_ascii``
    methods on the `_ColumnFormat` class.  But again, not all conversions are
    possible and may result in a `ValueError`.
    """

    def __new__(cls, format, strict=False):
        self = super().__new__(cls, format)
        self.format, self.width, self.precision = _parse_ascii_tformat(format, strict)

        # If no width has been specified, set the dtype here to default as well
        if format == self.format:
            self.recformat = ASCII2NUMPY[format]

        # This is to support handling logical (boolean) data from binary tables
        # in an ASCII table
        self._pseudo_logical = False
        return self

    @classmethod
    def from_column_format(cls, format):
        inst = cls.from_recformat(format.recformat)
        # Hack
        if format.format == "L":
            inst._pseudo_logical = True
        return inst

    @classmethod
    def from_recformat(cls, recformat):
        """Creates a column format from a Numpy record dtype format."""
        return cls(_convert_ascii_format(recformat, reverse=True))

    @lazyproperty
    def recformat(self):
        """Returns the equivalent Numpy record format string."""
        return _convert_ascii_format(self)
```
### 13 - astropy/io/fits/column.py:

Start line: 2523, End line: 2575

```python
def _convert_ascii_format(format, reverse=False):
    """Convert ASCII table format spec to record format spec."""
    if reverse:
        recformat, kind, dtype = _dtype_to_recformat(format)
        itemsize = dtype.itemsize

        if kind == "a":
            return "A" + str(itemsize)
        elif NUMPY2FITS.get(recformat) == "L":
            # Special case for logical/boolean types--for ASCII tables we
            # represent these as single character columns containing 'T' or 'F'
            # (a la the storage format for Logical columns in binary tables)
            return "A1"
        elif kind == "i":
            # Use for the width the maximum required to represent integers
            # of that byte size plus 1 for signs, but use a minimum of the
            # default width (to keep with existing behavior)
            width = 1 + len(str(2 ** (itemsize * 8)))
            width = max(width, ASCII_DEFAULT_WIDTHS["I"][0])
            return "I" + str(width)
        elif kind == "f":
            # This is tricky, but go ahead and use D if float-64, and E
            # if float-32 with their default widths
            if itemsize >= 8:
                format = "D"
            else:
                format = "E"
            width = ".".join(str(w) for w in ASCII_DEFAULT_WIDTHS[format])
            return format + width
        # TODO: There may be reasonable ways to represent other Numpy types so
        # let's see what other possibilities there are besides just 'a', 'i',
        # and 'f'.  If it doesn't have a reasonable ASCII representation then
        # raise an exception
    else:
        format, width, precision = _parse_ascii_tformat(format)

        # This gives a sensible "default" dtype for a given ASCII
        # format code
        recformat = ASCII2NUMPY[format]

        # The following logic is taken from CFITSIO:
        # For integers, if the width <= 4 we can safely use 16-bit ints for all
        # values, if width >= 10 we may need to accommodate 64-bit ints.
        # values [for the non-standard J format code just always force 64-bit]
        if format == "I":
            if width <= 4:
                recformat = "i2"
            elif width > 9:
                recformat = "i8"
        elif format == "A":
            recformat += str(width)

        return recformat
```
### 18 - astropy/io/fits/column.py:

Start line: 426, End line: 480

```python
# TODO: Table column formats need to be verified upon first reading the file;
# as it is, an invalid P format will raise a VerifyError from some deep,
# unexpected place
class _FormatP(str):
    """For P format in variable length table."""

    # As far as I can tell from my reading of the FITS standard, a type code is
    # *required* for P and Q formats; there is no default
    _format_re_template = (
        r"(?P<repeat>\d+)?{}(?P<dtype>[LXBIJKAEDCM])(?:\((?P<max>\d*)\))?"
    )
    _format_code = "P"
    _format_re = re.compile(_format_re_template.format(_format_code))
    _descriptor_format = "2i4"

    def __new__(cls, dtype, repeat=None, max=None):
        obj = super().__new__(cls, cls._descriptor_format)
        obj.format = NUMPY2FITS[dtype]
        obj.dtype = dtype
        obj.repeat = repeat
        obj.max = max
        return obj

    def __getnewargs__(self):
        return (self.dtype, self.repeat, self.max)

    @classmethod
    def from_tform(cls, format):
        m = cls._format_re.match(format)
        if not m or m.group("dtype") not in FITS2NUMPY:
            raise VerifyError(f"Invalid column format: {format}")
        repeat = m.group("repeat")
        array_dtype = m.group("dtype")
        max = m.group("max")
        if not max:
            max = None
        return cls(FITS2NUMPY[array_dtype], repeat=repeat, max=max)

    @property
    def tform(self):
        repeat = "" if self.repeat is None else self.repeat
        max = "" if self.max is None else self.max
        return f"{repeat}{self._format_code}{self.format}({max})"


class _FormatQ(_FormatP):
    """Carries type description of the Q format for variable length arrays.

    The Q format is like the P format but uses 64-bit integers in the array
    descriptors, allowing for heaps stored beyond 2GB into a file.
    """

    _format_code = "Q"
    _format_re = re.compile(_FormatP._format_re_template.format(_format_code))
    _descriptor_format = "2i8"
```
### 23 - astropy/io/fits/column.py:

Start line: 2443, End line: 2485

```python
def _convert_record2fits(format):
    """
    Convert record format spec to FITS format spec.
    """
    recformat, kind, dtype = _dtype_to_recformat(format)
    shape = dtype.shape
    itemsize = dtype.base.itemsize
    if dtype.char == "U" or (
        dtype.subdtype is not None and dtype.subdtype[0].char == "U"
    ):
        # Unicode dtype--itemsize is 4 times actual ASCII character length,
        # which what matters for FITS column formats
        # Use dtype.base and dtype.subdtype --dtype for multi-dimensional items
        itemsize = itemsize // 4

    option = str(itemsize)

    ndims = len(shape)
    repeat = 1
    if ndims > 0:
        nel = np.array(shape, dtype="i8").prod()
        if nel > 1:
            repeat = nel

    if kind == "a":
        # This is a kludge that will place string arrays into a
        # single field, so at least we won't lose data.  Need to
        # use a TDIM keyword to fix this, declaring as (slength,
        # dim1, dim2, ...)  as mwrfits does

        ntot = int(repeat) * int(option)

        output_format = str(ntot) + "A"
    elif recformat in NUMPY2FITS:  # record format
        if repeat != 1:
            repeat = str(repeat)
        else:
            repeat = ""
        output_format = repeat + NUMPY2FITS[recformat]
    else:
        raise ValueError(f"Illegal format `{format}`.")

    return output_format
```
### 24 - astropy/io/fits/column.py:

Start line: 2406, End line: 2440

```python
def _convert_fits2record(format):
    """
    Convert FITS format spec to record format spec.
    """
    repeat, dtype, option = _parse_tformat(format)

    if dtype in FITS2NUMPY:
        if dtype == "A":
            output_format = FITS2NUMPY[dtype] + str(repeat)
            # to accommodate both the ASCII table and binary table column
            # format spec, i.e. A7 in ASCII table is the same as 7A in
            # binary table, so both will produce 'a7'.
            # Technically the FITS standard does not allow this but it's a very
            # common mistake
            if format.lstrip()[0] == "A" and option != "":
                # make sure option is integer
                output_format = FITS2NUMPY[dtype] + str(int(option))
        else:
            repeat_str = ""
            if repeat != 1:
                repeat_str = str(repeat)
            output_format = repeat_str + FITS2NUMPY[dtype]

    elif dtype == "X":
        output_format = _FormatX(repeat)
    elif dtype == "P":
        output_format = _FormatP.from_tform(format)
    elif dtype == "Q":
        output_format = _FormatQ.from_tform(format)
    elif dtype == "F":
        output_format = "f8"
    else:
        raise ValueError(f"Illegal format `{format}`.")

    return output_format
```
### 25 - astropy/io/fits/column.py:

Start line: 408, End line: 423

```python
class _FormatX(str):
    """For X format in binary tables."""

    def __new__(cls, repeat=1):
        nbytes = ((repeat - 1) // 8) + 1
        # use an array, even if it is only ONE u1 (i.e. use tuple always)
        obj = super().__new__(cls, repr((nbytes,)) + "u1")
        obj.repeat = repeat
        return obj

    def __getnewargs__(self):
        return (self.repeat,)

    @property
    def tform(self):
        return f"{self.repeat}X"
```
### 28 - astropy/io/fits/column.py:

Start line: 1324, End line: 1365

```python
class Column(NotifierMixin):

    @classmethod
    def _guess_format(cls, format, start, dim):
        if start and dim:
            # This is impossible; this can't be a valid FITS column
            raise ValueError(
                "Columns cannot have both a start (TCOLn) and dim "
                "(TDIMn) option, since the former is only applies to "
                "ASCII tables, and the latter is only valid for binary tables."
            )
        elif start:
            # Only ASCII table columns can have a 'start' option
            guess_format = _AsciiColumnFormat
        elif dim:
            # Only binary tables can have a dim option
            guess_format = _ColumnFormat
        else:
            # If the format is *technically* a valid binary column format
            # (i.e. it has a valid format code followed by arbitrary
            # "optional" codes), but it is also strictly a valid ASCII
            # table format, then assume an ASCII table column was being
            # requested (the more likely case, after all).
            with suppress(VerifyError):
                format = _AsciiColumnFormat(format, strict=True)

            # A safe guess which reflects the existing behavior of previous
            # Astropy versions
            guess_format = _ColumnFormat

        try:
            format, recformat = cls._convert_format(format, guess_format)
        except VerifyError:
            # For whatever reason our guess was wrong (for example if we got
            # just 'F' that's not a valid binary format, but it an ASCII format
            # code albeit with the width/precision omitted
            guess_format = (
                _AsciiColumnFormat if guess_format is _ColumnFormat else _ColumnFormat
            )
            # If this fails too we're out of options--it is truly an invalid
            # format, or at least not supported
            format, recformat = cls._convert_format(format, guess_format)

        return format, recformat
```
### 29 - astropy/io/fits/column.py:

Start line: 249, End line: 288

```python
class _BaseColumnFormat(str):
    """
    Base class for binary table column formats (just called _ColumnFormat)
    and ASCII table column formats (_AsciiColumnFormat).
    """

    def __eq__(self, other):
        if not other:
            return False

        if isinstance(other, str):
            if not isinstance(other, self.__class__):
                try:
                    other = self.__class__(other)
                except ValueError:
                    return False
        else:
            return False

        return self.canonical == other.canonical

    def __hash__(self):
        return hash(self.canonical)

    @lazyproperty
    def dtype(self):
        """
        The Numpy dtype object created from the format's associated recformat.
        """
        return np.dtype(self.recformat)

    @classmethod
    def from_column_format(cls, format):
        """Creates a column format object from another column format object
        regardless of their type.

        That is, this can convert a _ColumnFormat to an _AsciiColumnFormat
        or vice versa at least in cases where a direct translation is possible.
        """
        return cls.from_recformat(format.recformat)
```
### 30 - astropy/io/fits/column.py:

Start line: 1289, End line: 1322

```python
class Column(NotifierMixin):

    @classmethod
    def _determine_formats(cls, format, start, dim, ascii):
        """
        Given a format string and whether or not the Column is for an
        ASCII table (ascii=None means unspecified, but lean toward binary table
        where ambiguous) create an appropriate _BaseColumnFormat instance for
        the column's format, and determine the appropriate recarray format.

        The values of the start and dim keyword arguments are also useful, as
        the former is only valid for ASCII tables and the latter only for
        BINARY tables.
        """
        # If the given format string is unambiguously a Numpy dtype or one of
        # the Numpy record format type specifiers supported by Astropy then that
        # should take priority--otherwise assume it is a FITS format
        if isinstance(format, np.dtype):
            format, _, _ = _dtype_to_recformat(format)

        # check format
        if ascii is None and not isinstance(format, _BaseColumnFormat):
            # We're just give a string which could be either a Numpy format
            # code, or a format for a binary column array *or* a format for an
            # ASCII column array--there may be many ambiguities here.  Try our
            # best to guess what the user intended.
            format, recformat = cls._guess_format(format, start, dim)
        elif not ascii and not isinstance(format, _BaseColumnFormat):
            format, recformat = cls._convert_format(format, _ColumnFormat)
        elif ascii and not isinstance(format, _AsciiColumnFormat):
            format, recformat = cls._convert_format(format, _AsciiColumnFormat)
        else:
            # The format is already acceptable and unambiguous
            recformat = format.recformat

        return format, recformat
```
### 34 - astropy/io/fits/column.py:

Start line: 897, End line: 964

```python
class Column(NotifierMixin):

    @ColumnAttribute("TCTYP")
    def coord_type(col, coord_type):
        if coord_type is None:
            return

        if not isinstance(coord_type, str) or len(coord_type) > 8:
            raise AssertionError(
                "Coordinate/axis type must be a string of atmost 8 characters."
            )

    @ColumnAttribute("TCUNI")
    def coord_unit(col, coord_unit):
        if coord_unit is not None and not isinstance(coord_unit, str):
            raise AssertionError("Coordinate/axis unit must be a string.")

    @ColumnAttribute("TCRPX")
    def coord_ref_point(col, coord_ref_point):
        if coord_ref_point is not None and not isinstance(
            coord_ref_point, numbers.Real
        ):
            raise AssertionError(
                "Pixel coordinate of the reference point must be real floating type."
            )

    @ColumnAttribute("TCRVL")
    def coord_ref_value(col, coord_ref_value):
        if coord_ref_value is not None and not isinstance(
            coord_ref_value, numbers.Real
        ):
            raise AssertionError(
                "Coordinate value at reference point must be real floating type."
            )

    @ColumnAttribute("TCDLT")
    def coord_inc(col, coord_inc):
        if coord_inc is not None and not isinstance(coord_inc, numbers.Real):
            raise AssertionError("Coordinate increment must be real floating type.")

    @ColumnAttribute("TRPOS")
    def time_ref_pos(col, time_ref_pos):
        if time_ref_pos is not None and not isinstance(time_ref_pos, str):
            raise AssertionError("Time reference position must be a string.")

    format = ColumnAttribute("TFORM")
    unit = ColumnAttribute("TUNIT")
    null = ColumnAttribute("TNULL")
    bscale = ColumnAttribute("TSCAL")
    bzero = ColumnAttribute("TZERO")
    disp = ColumnAttribute("TDISP")
    start = ColumnAttribute("TBCOL")
    dim = ColumnAttribute("TDIM")

    @lazyproperty
    def ascii(self):
        """Whether this `Column` represents a column in an ASCII table."""
        return isinstance(self.format, _AsciiColumnFormat)

    @lazyproperty
    def dtype(self):
        return self.format.dtype

    def copy(self):
        """
        Return a copy of this `Column`.
        """
        tmp = Column(format="I")  # just use a throw-away format
        tmp.__dict__ = self.__dict__.copy()
        return tmp
```
### 39 - astropy/io/fits/column.py:

Start line: 2488, End line: 2520

```python
def _dtype_to_recformat(dtype):
    """
    Utility function for converting a dtype object or string that instantiates
    a dtype (e.g. 'float32') into one of the two character Numpy format codes
    that have been traditionally used by Astropy.

    In particular, use of 'a' to refer to character data is long since
    deprecated in Numpy, but Astropy remains heavily invested in its use
    (something to try to get away from sooner rather than later).
    """
    if not isinstance(dtype, np.dtype):
        dtype = np.dtype(dtype)

    kind = dtype.base.kind

    if kind in ("U", "S"):
        recformat = kind = "a"
    else:
        itemsize = dtype.base.itemsize
        recformat = kind + str(itemsize)

    return recformat, kind, dtype


def _convert_format(format, reverse=False):
    """
    Convert FITS format spec to record format spec.  Do the opposite if
    reverse=True.
    """
    if reverse:
        return _convert_record2fits(format)
    else:
        return _convert_fits2record(format)
```
### 42 - astropy/io/fits/column.py:

Start line: 317, End line: 341

```python
class _ColumnFormat(_BaseColumnFormat):

    @classmethod
    def from_recformat(cls, recformat):
        """Creates a column format from a Numpy record dtype format."""
        return cls(_convert_format(recformat, reverse=True))

    @lazyproperty
    def recformat(self):
        """Returns the equivalent Numpy record format string."""
        return _convert_format(self)

    @lazyproperty
    def canonical(self):
        """
        Returns a 'canonical' string representation of this format.

        This is in the proper form of rTa where T is the single character data
        type code, a is the optional part, and r is the repeat.  If repeat == 1
        (the default) it is left out of this representation.
        """
        if self.repeat == 1:
            repeat = ""
        else:
            repeat = str(self.repeat)

        return f"{repeat}{self.format}{self.option}"
```
### 44 - astropy/io/fits/column.py:

Start line: 1565, End line: 1614

```python
class ColDefs(NotifierMixin):

    def _init_from_table(self, table):
        hdr = table._header
        nfields = hdr["TFIELDS"]

        # go through header keywords to pick out column definition keywords
        # definition dictionaries for each field
        col_keywords = [{} for i in range(nfields)]
        for keyword in hdr:
            key = TDEF_RE.match(keyword)
            try:
                label = key.group("label")
            except Exception:
                continue  # skip if there is no match
            if label in KEYWORD_NAMES:
                col = int(key.group("num"))
                if 0 < col <= nfields:
                    attr = KEYWORD_TO_ATTRIBUTE[label]
                    value = hdr[keyword]
                    if attr == "format":
                        # Go ahead and convert the format value to the
                        # appropriate ColumnFormat container now
                        value = self._col_format_cls(value)
                    col_keywords[col - 1][attr] = value

        # Verify the column keywords and display any warnings if necessary;
        # we only want to pass on the valid keywords
        for idx, kwargs in enumerate(col_keywords):
            valid_kwargs, invalid_kwargs = Column._verify_keywords(**kwargs)
            for val in invalid_kwargs.values():
                warnings.warn(
                    f"Invalid keyword for column {idx + 1}: {val[1]}", VerifyWarning
                )
            # Special cases for recformat and dim
            # TODO: Try to eliminate the need for these special cases
            del valid_kwargs["recformat"]
            if "dim" in valid_kwargs:
                valid_kwargs["dim"] = kwargs["dim"]
            col_keywords[idx] = valid_kwargs

        # data reading will be delayed
        for col in range(nfields):
            col_keywords[col]["array"] = Delayed(table, col)

        # now build the columns
        self.columns = [Column(**attrs) for attrs in col_keywords]

        # Add the table HDU is a listener to changes to the columns
        # (either changes to individual columns, or changes to the set of
        # columns (add/remove/etc.))
        self._add_listener(table)
```
### 50 - astropy/io/fits/column.py:

Start line: 1367, End line: 1434

```python
class Column(NotifierMixin):

    def _convert_to_valid_data_type(self, array):
        # Convert the format to a type we understand
        if isinstance(array, Delayed):
            return array
        elif array is None:
            return array
        else:
            format = self.format
            dims = self._dims
            if dims and format.format not in "PQ":
                shape = dims[:-1] if "A" in format else dims
                shape = (len(array),) + shape
                array = array.reshape(shape)

            if "P" in format or "Q" in format:
                return array
            elif "A" in format:
                if array.dtype.char in "SU":
                    if dims:
                        # The 'last' dimension (first in the order given
                        # in the TDIMn keyword itself) is the number of
                        # characters in each string
                        fsize = dims[-1]
                    else:
                        fsize = np.dtype(format.recformat).itemsize
                    return chararray.array(array, itemsize=fsize, copy=False)
                else:
                    return _convert_array(array, np.dtype(format.recformat))
            elif "L" in format:
                # boolean needs to be scaled back to storage values ('T', 'F')
                if array.dtype == np.dtype("bool"):
                    return np.where(array == np.False_, ord("F"), ord("T"))
                else:
                    return np.where(array == 0, ord("F"), ord("T"))
            elif "X" in format:
                return _convert_array(array, np.dtype("uint8"))
            else:
                # Preserve byte order of the original array for now; see #77
                numpy_format = array.dtype.byteorder + format.recformat

                # Handle arrays passed in as unsigned ints as pseudo-unsigned
                # int arrays; blatantly tacked in here for now--we need columns
                # to have explicit knowledge of whether they treated as
                # pseudo-unsigned
                bzeros = {
                    2: np.uint16(2**15),
                    4: np.uint32(2**31),
                    8: np.uint64(2**63),
                }
                if (
                    array.dtype.kind == "u"
                    and array.dtype.itemsize in bzeros
                    and self.bscale in (1, None, "")
                    and self.bzero == bzeros[array.dtype.itemsize]
                ):
                    # Basically the array is uint, has scale == 1.0, and the
                    # bzero is the appropriate value for a pseudo-unsigned
                    # integer of the input dtype, then go ahead and assume that
                    # uint is assumed
                    numpy_format = numpy_format.replace("i", "u")
                    self._pseudo_unsigned_ints = True

                # The .base here means we're dropping the shape information,
                # which is only used to format recarray fields, and is not
                # useful for converting input arrays to the correct data type
                dtype = np.dtype(numpy_format).base

                return _convert_array(array, dtype)
```
### 65 - astropy/io/fits/column.py:

Start line: 860, End line: 895

```python
class Column(NotifierMixin):

    @array.deleter
    def array(self):
        try:
            del self.__dict__["array"]
        except KeyError:
            pass

        self._parent_fits_rec = None

    @ColumnAttribute("TTYPE")
    def name(col, name):
        if name is None:
            # Allow None to indicate deleting the name, or to just indicate an
            # unspecified name (when creating a new Column).
            return

        # Check that the name meets the recommended standard--other column
        # names are *allowed*, but will be discouraged
        if isinstance(name, str) and not TTYPE_RE.match(name):
            warnings.warn(
                "It is strongly recommended that column names contain only "
                "upper and lower-case ASCII letters, digits, or underscores "
                "for maximum compatibility with other software "
                "(got {!r}).".format(name),
                VerifyWarning,
            )

        # This ensures that the new name can fit into a single FITS card
        # without any special extension like CONTINUE cards or the like.
        if not isinstance(name, str) or len(str(Card("TTYPE", name))) != CARD_LENGTH:
            raise AssertionError(
                "Column name must be a string able to fit in a single "
                "FITS card--typically this means a maximum of 68 "
                "characters, though it may be fewer if the string "
                "contains special characters like quotes."
            )
```
### 71 - astropy/io/fits/column.py:

Start line: 3, End line: 85

```python
import copy
import numbers
import operator
import re
import sys
import warnings
import weakref
from collections import OrderedDict
from contextlib import suppress
from functools import reduce

import numpy as np
from numpy import char as chararray

from astropy.utils import indent, isiterable, lazyproperty
from astropy.utils.exceptions import AstropyUserWarning

from .card import CARD_LENGTH, Card
from .util import NotifierMixin, _convert_array, _is_int, cmp, encode_ascii, pairwise
from .verify import VerifyError, VerifyWarning

__all__ = ["Column", "ColDefs", "Delayed"]


# mapping from TFORM data type to numpy data type (code)
# L: Logical (Boolean)
# B: Unsigned Byte
# I: 16-bit Integer
# J: 32-bit Integer
# K: 64-bit Integer
# E: Single-precision Floating Point
# D: Double-precision Floating Point
# C: Single-precision Complex
# M: Double-precision Complex
# A: Character
FITS2NUMPY = {
    "L": "i1",
    "B": "u1",
    "I": "i2",
    "J": "i4",
    "K": "i8",
    "E": "f4",
    "D": "f8",
    "C": "c8",
    "M": "c16",
    "A": "a",
}

# the inverse dictionary of the above
NUMPY2FITS = {val: key for key, val in FITS2NUMPY.items()}
# Normally booleans are represented as ints in Astropy, but if passed in a numpy
# boolean array, that should be supported
NUMPY2FITS["b1"] = "L"
# Add unsigned types, which will be stored as signed ints with a TZERO card.
NUMPY2FITS["u2"] = "I"
NUMPY2FITS["u4"] = "J"
NUMPY2FITS["u8"] = "K"
# Add half precision floating point numbers which will be up-converted to
# single precision.
NUMPY2FITS["f2"] = "E"

# This is the order in which values are converted to FITS types
# Note that only double precision floating point/complex are supported
FORMATORDER = ["L", "B", "I", "J", "K", "D", "M", "A"]

# Convert single precision floating point/complex to double precision.
FITSUPCONVERTERS = {"E": "D", "C": "M"}

# mapping from ASCII table TFORM data type to numpy data type
# A: Character
# I: Integer (32-bit)
# J: Integer (64-bit; non-standard)
# F: Float (64-bit; fixed decimal notation)
# E: Float (64-bit; exponential notation)
# D: Float (64-bit; exponential notation, always 64-bit by convention)
ASCII2NUMPY = {"A": "a", "I": "i4", "J": "i8", "F": "f8", "E": "f8", "D": "f8"}

# Maps FITS ASCII column format codes to the appropriate Python string
# formatting codes for that type.
ASCII2STR = {"A": "", "I": "d", "J": "d", "F": "f", "E": "E", "D": "E"}

# For each ASCII table format code, provides a default width (and decimal
# precision) for when one isn't given explicitly in the column format
```
### 75 - astropy/io/fits/column.py:

Start line: 86, End line: 146

```python
ASCII_DEFAULT_WIDTHS = {
    "A": (1, 0),
    "I": (10, 0),
    "J": (15, 0),
    "E": (15, 7),
    "F": (16, 7),
    "D": (25, 17),
}

# TDISPn for both ASCII and Binary tables
TDISP_RE_DICT = {}
TDISP_RE_DICT["F"] = re.compile(
    r"(?:(?P<formatc>[F])(?:(?P<width>[0-9]+)\.{1}(?P<precision>[0-9])+)+)|"
)
TDISP_RE_DICT["A"] = TDISP_RE_DICT["L"] = re.compile(
    r"(?:(?P<formatc>[AL])(?P<width>[0-9]+)+)|"
)
TDISP_RE_DICT["I"] = TDISP_RE_DICT["B"] = TDISP_RE_DICT["O"] = TDISP_RE_DICT[
    "Z"
] = re.compile(
    r"(?:(?P<formatc>[IBOZ])(?:(?P<width>[0-9]+)"
    r"(?:\.{0,1}(?P<precision>[0-9]+))?))|"
)
TDISP_RE_DICT["E"] = TDISP_RE_DICT["G"] = TDISP_RE_DICT["D"] = re.compile(
    r"(?:(?P<formatc>[EGD])(?:(?P<width>[0-9]+)\."
    r"(?P<precision>[0-9]+))+)"
    r"(?:E{0,1}(?P<exponential>[0-9]+)?)|"
)
TDISP_RE_DICT["EN"] = TDISP_RE_DICT["ES"] = re.compile(
    r"(?:(?P<formatc>E[NS])(?:(?P<width>[0-9]+)\.{1}(?P<precision>[0-9])+)+)"
)

# mapping from TDISP format to python format
# A: Character
# L: Logical (Boolean)
# I: 16-bit Integer
#    Can't predefine zero padding and space padding before hand without
#    knowing the value being formatted, so grabbing precision and using that
#    to zero pad, ignoring width. Same with B, O, and Z
# B: Binary Integer
# O: Octal Integer
# Z: Hexadecimal Integer
# F: Float (64-bit; fixed decimal notation)
# EN: Float (engineering fortran format, exponential multiple of thee
# ES: Float (scientific, same as EN but non-zero leading digit
# E: Float, exponential notation
#    Can't get exponential restriction to work without knowing value
#    before hand, so just using width and precision, same with D, G, EN, and
#    ES formats
# D: Double-precision Floating Point with exponential
#    (E but for double precision)
# G: Double-precision Floating Point, may or may not show exponent
TDISP_FMT_DICT = {
    "I": "{{:{width}d}}",
    "B": "{{:{width}b}}",
    "O": "{{:{width}o}}",
    "Z": "{{:{width}x}}",
    "F": "{{:{width}.{precision}f}}",
    "G": "{{:{width}.{precision}g}}",
}
TDISP_FMT_DICT["A"] = TDISP_FMT_DICT["L"] = "{{:>{width}}}"
```
### 78 - astropy/io/fits/column.py:

Start line: 147, End line: 231

```python
TDISP_FMT_DICT["E"] = TDISP_FMT_DICT["D"] = TDISP_FMT_DICT["EN"] = TDISP_FMT_DICT[
    "ES"
] = "{{:{width}.{precision}e}}"

# tuple of column/field definition common names and keyword names, make
# sure to preserve the one-to-one correspondence when updating the list(s).
# Use lists, instead of dictionaries so the names can be displayed in a
# preferred order.
KEYWORD_NAMES = (
    "TTYPE",
    "TFORM",
    "TUNIT",
    "TNULL",
    "TSCAL",
    "TZERO",
    "TDISP",
    "TBCOL",
    "TDIM",
    "TCTYP",
    "TCUNI",
    "TCRPX",
    "TCRVL",
    "TCDLT",
    "TRPOS",
)
KEYWORD_ATTRIBUTES = (
    "name",
    "format",
    "unit",
    "null",
    "bscale",
    "bzero",
    "disp",
    "start",
    "dim",
    "coord_type",
    "coord_unit",
    "coord_ref_point",
    "coord_ref_value",
    "coord_inc",
    "time_ref_pos",
)
"""This is a list of the attributes that can be set on `Column` objects."""


KEYWORD_TO_ATTRIBUTE = OrderedDict(zip(KEYWORD_NAMES, KEYWORD_ATTRIBUTES))

ATTRIBUTE_TO_KEYWORD = OrderedDict(zip(KEYWORD_ATTRIBUTES, KEYWORD_NAMES))


# TODO: Define a list of default comments to associate with each table keyword

# TFORMn regular expression
TFORMAT_RE = re.compile(
    r"(?P<repeat>^[0-9]*)(?P<format>[LXBIJKAEDCMPQ])(?P<option>[!-~]*)", re.I
)

# TFORMn for ASCII tables; two different versions depending on whether
# the format is floating-point or not; allows empty values for width
# in which case defaults are used
TFORMAT_ASCII_RE = re.compile(
    r"(?:(?P<format>[AIJ])(?P<width>[0-9]+)?)|"
    r"(?:(?P<formatf>[FED])"
    r"(?:(?P<widthf>[0-9]+)(?:\."
    r"(?P<precision>[0-9]+))?)?)"
)

TTYPE_RE = re.compile(r"[0-9a-zA-Z_]+")
"""
Regular expression for valid table column names.  See FITS Standard v3.0 section 7.2.2.
"""

# table definition keyword regular expression
TDEF_RE = re.compile(r"(?P<label>^T[A-Z]*)(?P<num>[1-9][0-9 ]*$)")

# table dimension keyword regular expression (fairly flexible with whitespace)
TDIM_RE = re.compile(r"\(\s*(?P<dims>(?:\d+\s*)(?:,\s*\d+\s*)*\s*)\)\s*")

# value for ASCII table cell with value = TNULL
# this can be reset by user.
ASCIITNULL = 0

# The default placeholder to use for NULL values in ASCII tables when
# converting from binary to ASCII tables
DEFAULT_ASCII_TNULL = "---"
```
