# astropy__astropy-13390

| **astropy/astropy** | `1e75f298aef2540240c63b4075d06851d55fc19a` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | 6747 |
| **Avg pos** | 16.0 |
| **Min pos** | 16 |
| **Max pos** | 16 |
| **Top file pos** | 1 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/astropy/table/column.py b/astropy/table/column.py
--- a/astropy/table/column.py
+++ b/astropy/table/column.py
@@ -297,31 +297,23 @@ def _make_compare(oper):
     oper : str
         Operator name
     """
-    swapped_oper = {'__eq__': '__eq__',
-                    '__ne__': '__ne__',
-                    '__gt__': '__lt__',
-                    '__lt__': '__gt__',
-                    '__ge__': '__le__',
-                    '__le__': '__ge__'}[oper]
-
     def _compare(self, other):
         op = oper  # copy enclosed ref to allow swap below
 
-        # Special case to work around #6838.  Other combinations work OK,
-        # see tests.test_column.test_unicode_sandwich_compare().  In this
-        # case just swap self and other.
-        #
-        # This is related to an issue in numpy that was addressed in np 1.13.
-        # However that fix does not make this problem go away, but maybe
-        # future numpy versions will do so.  NUMPY_LT_1_13 to get the
-        # attention of future maintainers to check (by deleting or versioning
-        # the if block below).  See #6899 discussion.
-        # 2019-06-21: still needed with numpy 1.16.
-        if (isinstance(self, MaskedColumn) and self.dtype.kind == 'U'
-                and isinstance(other, MaskedColumn) and other.dtype.kind == 'S'):
-            self, other = other, self
-            op = swapped_oper
+        # If other is a Quantity, we should let it do the work, since
+        # it can deal with our possible unit (which, for MaskedColumn,
+        # would get dropped below, as '.data' is accessed in super()).
+        if isinstance(other, Quantity):
+            return NotImplemented
 
+        # If we are unicode and other is a column with bytes, defer to it for
+        # doing the unicode sandwich.  This avoids problems like those
+        # discussed in #6838 and #6899.
+        if (self.dtype.kind == 'U'
+                and isinstance(other, Column) and other.dtype.kind == 'S'):
+            return NotImplemented
+
+        # If we are bytes, encode other as needed.
         if self.dtype.char == 'S':
             other = self._encode_str(other)
 
@@ -1531,10 +1523,11 @@ def __new__(cls, data=None, name=None, mask=None, fill_value=None,
 
         # Note: do not set fill_value in the MaskedArray constructor because this does not
         # go through the fill_value workarounds.
-        if fill_value is None and getattr(data, 'fill_value', None) is not None:
-            # Coerce the fill_value to the correct type since `data` may be a
-            # different dtype than self.
-            fill_value = np.array(data.fill_value, self.dtype)[()]
+        if fill_value is None:
+            data_fill_value = getattr(data, 'fill_value', None)
+            if (data_fill_value is not None
+                    and data_fill_value != np.ma.default_fill_value(data.dtype)):
+                fill_value = np.array(data_fill_value, self.dtype)[()]
         self.fill_value = fill_value
 
         self.parent_table = None

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| astropy/table/column.py | 300 | 323 | - | 1 | -
| astropy/table/column.py | 1534 | 1537 | 16 | 1 | 6747


## Problem Statement

```
BUG: Table test failures with np 1.23.0rc3
\`\`\`
====================================================================== FAILURES =======================================================================
__________________________________________________________ test_col_unicode_sandwich_unicode __________________________________________________________
numpy.core._exceptions._UFuncNoLoopError: ufunc 'not_equal' did not contain a loop with signature matching types (<class 'numpy.dtype[str_]'>, <class 'numpy.dtype[bytes_]'>) -> None

The above exception was the direct cause of the following exception:

    def test_col_unicode_sandwich_unicode():
        """
        Sanity check that Unicode Column behaves normally.
        """
        uba = 'bä'
        uba8 = uba.encode('utf-8')
    
        c = table.Column([uba, 'def'], dtype='U')
        assert c[0] == uba
        assert isinstance(c[:0], table.Column)
        assert isinstance(c[0], str)
        assert np.all(c[:2] == np.array([uba, 'def']))
    
        assert isinstance(c[:], table.Column)
        assert c[:].dtype.char == 'U'
    
        ok = c == [uba, 'def']
        assert type(ok) == np.ndarray
        assert ok.dtype.char == '?'
        assert np.all(ok)
    
>       assert np.all(c != [uba8, b'def'])

astropy/table/tests/test_column.py:777: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

self = <Column dtype='str3' length=2>
 bä
def, other = [b'b\xc3\xa4', b'def']

    def _compare(self, other):
        op = oper  # copy enclosed ref to allow swap below
    
        # Special case to work around #6838.  Other combinations work OK,
        # see tests.test_column.test_unicode_sandwich_compare().  In this
        # case just swap self and other.
        #
        # This is related to an issue in numpy that was addressed in np 1.13.
        # However that fix does not make this problem go away, but maybe
        # future numpy versions will do so.  NUMPY_LT_1_13 to get the
        # attention of future maintainers to check (by deleting or versioning
        # the if block below).  See #6899 discussion.
        # 2019-06-21: still needed with numpy 1.16.
        if (isinstance(self, MaskedColumn) and self.dtype.kind == 'U'
                and isinstance(other, MaskedColumn) and other.dtype.kind == 'S'):
            self, other = other, self
            op = swapped_oper
    
        if self.dtype.char == 'S':
            other = self._encode_str(other)
    
        # Now just let the regular ndarray.__eq__, etc., take over.
>       result = getattr(super(Column, self), op)(other)
E       FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison

astropy/table/column.py:329: FutureWarning
______________________________________________ test_unicode_sandwich_compare[MaskedColumn-MaskedColumn] _______________________________________________

class1 = <class 'astropy.table.column.MaskedColumn'>, class2 = <class 'astropy.table.column.MaskedColumn'>

    @pytest.mark.parametrize('class1', [table.MaskedColumn, table.Column])
    @pytest.mark.parametrize('class2', [table.MaskedColumn, table.Column, str, list])
    def test_unicode_sandwich_compare(class1, class2):
        """Test that comparing a bytestring Column/MaskedColumn with various
        str (unicode) object types gives the expected result.  Tests #6838.
        """
        obj1 = class1([b'a', b'c'])
        if class2 is str:
            obj2 = 'a'
        elif class2 is list:
            obj2 = ['a', 'b']
        else:
            obj2 = class2(['a', 'b'])
    
        assert np.all((obj1 == obj2) == [True, False])
        assert np.all((obj2 == obj1) == [True, False])
    
        assert np.all((obj1 != obj2) == [False, True])
        assert np.all((obj2 != obj1) == [False, True])
    
>       assert np.all((obj1 > obj2) == [False, True])
E       TypeError: '>' not supported between instances of 'MaskedColumn' and 'MaskedColumn'

astropy/table/tests/test_column.py:857: TypeError
_________________________________________________ test_unicode_sandwich_compare[Column-MaskedColumn] __________________________________________________

class1 = <class 'astropy.table.column.MaskedColumn'>, class2 = <class 'astropy.table.column.Column'>

    @pytest.mark.parametrize('class1', [table.MaskedColumn, table.Column])
    @pytest.mark.parametrize('class2', [table.MaskedColumn, table.Column, str, list])
    def test_unicode_sandwich_compare(class1, class2):
        """Test that comparing a bytestring Column/MaskedColumn with various
        str (unicode) object types gives the expected result.  Tests #6838.
        """
        obj1 = class1([b'a', b'c'])
        if class2 is str:
            obj2 = 'a'
        elif class2 is list:
            obj2 = ['a', 'b']
        else:
            obj2 = class2(['a', 'b'])
    
        assert np.all((obj1 == obj2) == [True, False])
        assert np.all((obj2 == obj1) == [True, False])
    
        assert np.all((obj1 != obj2) == [False, True])
        assert np.all((obj2 != obj1) == [False, True])
    
>       assert np.all((obj1 > obj2) == [False, True])
E       TypeError: '>' not supported between instances of 'MaskedColumn' and 'Column'

astropy/table/tests/test_column.py:857: TypeError
____________________________________________________ test_unicode_sandwich_compare[Column-Column] _____________________________________________________
numpy.core._exceptions._UFuncNoLoopError: ufunc 'equal' did not contain a loop with signature matching types (<class 'numpy.dtype[str_]'>, <class 'numpy.dtype[bytes_]'>) -> None

The above exception was the direct cause of the following exception:

class1 = <class 'astropy.table.column.Column'>, class2 = <class 'astropy.table.column.Column'>

    @pytest.mark.parametrize('class1', [table.MaskedColumn, table.Column])
    @pytest.mark.parametrize('class2', [table.MaskedColumn, table.Column, str, list])
    def test_unicode_sandwich_compare(class1, class2):
        """Test that comparing a bytestring Column/MaskedColumn with various
        str (unicode) object types gives the expected result.  Tests #6838.
        """
        obj1 = class1([b'a', b'c'])
        if class2 is str:
            obj2 = 'a'
        elif class2 is list:
            obj2 = ['a', 'b']
        else:
            obj2 = class2(['a', 'b'])
    
        assert np.all((obj1 == obj2) == [True, False])
>       assert np.all((obj2 == obj1) == [True, False])

astropy/table/tests/test_column.py:852: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

self = <Column dtype='str1' length=2>
a
b, other = <Column dtype='bytes1' length=2>
a
c

    def _compare(self, other):
        op = oper  # copy enclosed ref to allow swap below
    
        # Special case to work around #6838.  Other combinations work OK,
        # see tests.test_column.test_unicode_sandwich_compare().  In this
        # case just swap self and other.
        #
        # This is related to an issue in numpy that was addressed in np 1.13.
        # However that fix does not make this problem go away, but maybe
        # future numpy versions will do so.  NUMPY_LT_1_13 to get the
        # attention of future maintainers to check (by deleting or versioning
        # the if block below).  See #6899 discussion.
        # 2019-06-21: still needed with numpy 1.16.
        if (isinstance(self, MaskedColumn) and self.dtype.kind == 'U'
                and isinstance(other, MaskedColumn) and other.dtype.kind == 'S'):
            self, other = other, self
            op = swapped_oper
    
        if self.dtype.char == 'S':
            other = self._encode_str(other)
    
        # Now just let the regular ndarray.__eq__, etc., take over.
>       result = getattr(super(Column, self), op)(other)
E       FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison

astropy/table/column.py:329: FutureWarning
___________________________________________________ test_unicode_sandwich_compare[str-MaskedColumn] ___________________________________________________

class1 = <class 'astropy.table.column.MaskedColumn'>, class2 = <class 'str'>

    @pytest.mark.parametrize('class1', [table.MaskedColumn, table.Column])
    @pytest.mark.parametrize('class2', [table.MaskedColumn, table.Column, str, list])
    def test_unicode_sandwich_compare(class1, class2):
        """Test that comparing a bytestring Column/MaskedColumn with various
        str (unicode) object types gives the expected result.  Tests #6838.
        """
        obj1 = class1([b'a', b'c'])
        if class2 is str:
            obj2 = 'a'
        elif class2 is list:
            obj2 = ['a', 'b']
        else:
            obj2 = class2(['a', 'b'])
    
        assert np.all((obj1 == obj2) == [True, False])
        assert np.all((obj2 == obj1) == [True, False])
    
        assert np.all((obj1 != obj2) == [False, True])
        assert np.all((obj2 != obj1) == [False, True])
    
>       assert np.all((obj1 > obj2) == [False, True])
E       TypeError: '>' not supported between instances of 'MaskedColumn' and 'str'

astropy/table/tests/test_column.py:857: TypeError
__________________________________________________ test_unicode_sandwich_compare[list-MaskedColumn] ___________________________________________________

class1 = <class 'astropy.table.column.MaskedColumn'>, class2 = <class 'list'>

    @pytest.mark.parametrize('class1', [table.MaskedColumn, table.Column])
    @pytest.mark.parametrize('class2', [table.MaskedColumn, table.Column, str, list])
    def test_unicode_sandwich_compare(class1, class2):
        """Test that comparing a bytestring Column/MaskedColumn with various
        str (unicode) object types gives the expected result.  Tests #6838.
        """
        obj1 = class1([b'a', b'c'])
        if class2 is str:
            obj2 = 'a'
        elif class2 is list:
            obj2 = ['a', 'b']
        else:
            obj2 = class2(['a', 'b'])
    
        assert np.all((obj1 == obj2) == [True, False])
        assert np.all((obj2 == obj1) == [True, False])
    
        assert np.all((obj1 != obj2) == [False, True])
        assert np.all((obj2 != obj1) == [False, True])
    
>       assert np.all((obj1 > obj2) == [False, True])
E       TypeError: '>' not supported between instances of 'MaskedColumn' and 'list'

astropy/table/tests/test_column.py:857: TypeError
=============================================================== short test summary info ===============================================================
FAILED astropy/table/tests/test_column.py::test_col_unicode_sandwich_unicode - FutureWarning: elementwise comparison failed; returning scalar instea...
FAILED astropy/table/tests/test_column.py::test_unicode_sandwich_compare[MaskedColumn-MaskedColumn] - TypeError: '>' not supported between instances...
FAILED astropy/table/tests/test_column.py::test_unicode_sandwich_compare[Column-MaskedColumn] - TypeError: '>' not supported between instances of 'M...
FAILED astropy/table/tests/test_column.py::test_unicode_sandwich_compare[Column-Column] - FutureWarning: elementwise comparison failed; returning sc...
FAILED astropy/table/tests/test_column.py::test_unicode_sandwich_compare[str-MaskedColumn] - TypeError: '>' not supported between instances of 'Mask...
FAILED astropy/table/tests/test_column.py::test_unicode_sandwich_compare[list-MaskedColumn] - TypeError: '>' not supported between instances of 'Mas...
=============================================== 6 failed, 3377 passed, 43 skipped, 14 xfailed in 25.62s ===============================================

\`\`\`


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 astropy/table/column.py** | 307 | 333| 314 | 314 | 14772 | 
| 2 | **1 astropy/table/column.py** | 778 | 805| 201 | 515 | 14772 | 
| 3 | 2 astropy/io/fits/column.py | 1137 | 1198| 629 | 1144 | 37109 | 
| 4 | **2 astropy/table/column.py** | 3 | 55| 391 | 1535 | 37109 | 
| 5 | **2 astropy/table/column.py** | 290 | 305| 123 | 1658 | 37109 | 
| 6 | 3 astropy/table/table.py | 2 | 75| 641 | 2299 | 69916 | 
| 7 | 3 astropy/io/fits/column.py | 676 | 702| 205 | 2504 | 69916 | 
| 8 | **3 astropy/table/column.py** | 1688 | 1715| 269 | 2773 | 69916 | 
| 9 | 3 astropy/io/fits/column.py | 998 | 1068| 745 | 3518 | 69916 | 
| 10 | **3 astropy/table/column.py** | 1266 | 1286| 195 | 3713 | 69916 | 
| 11 | 3 astropy/table/table.py | 3478 | 3551| 641 | 4354 | 69916 | 
| 12 | 3 astropy/io/fits/column.py | 847 | 920| 534 | 4888 | 69916 | 
| 13 | 3 astropy/io/fits/column.py | 1069 | 1135| 697 | 5585 | 69916 | 
| 14 | 4 astropy/io/misc/asdf/tags/table/table.py | 96 | 136| 262 | 5847 | 70915 | 
| 15 | 5 astropy/io/votable/converters.py | 1328 | 1359| 243 | 6090 | 80869 | 
| **-> 16 <-** | **5 astropy/table/column.py** | 1490 | 1546| 657 | 6747 | 80869 | 
| 17 | 5 astropy/table/table.py | 3431 | 3476| 388 | 7135 | 80869 | 
| 18 | 6 astropy/io/fits/diff.py | 1398 | 1447| 510 | 7645 | 93728 | 
| 19 | **6 astropy/table/column.py** | 1166 | 1178| 139 | 7784 | 93728 | 
| 20 | 7 astropy/io/ascii/core.py | 714 | 749| 353 | 8137 | 107351 | 
| 21 | 7 astropy/io/fits/column.py | 3 | 77| 761 | 8898 | 107351 | 
| 22 | 8 astropy/table/__init__.py | 3 | 14| 188 | 9086 | 108196 | 
| 23 | 8 astropy/io/fits/diff.py | 1258 | 1396| 1274 | 10360 | 108196 | 
| 24 | 8 astropy/io/fits/column.py | 129 | 183| 663 | 11023 | 108196 | 
| 25 | **8 astropy/table/column.py** | 480 | 537| 487 | 11510 | 108196 | 
| 26 | **8 astropy/table/column.py** | 1068 | 1092| 196 | 11706 | 108196 | 
| 27 | 8 astropy/io/fits/column.py | 1280 | 1343| 633 | 12339 | 108196 | 
| 28 | 8 astropy/table/table.py | 3850 | 3898| 495 | 12834 | 108196 | 
| 29 | 8 astropy/io/ascii/core.py | 183 | 267| 404 | 13238 | 108196 | 
| 30 | 8 astropy/table/table.py | 2319 | 2374| 468 | 13706 | 108196 | 
| 31 | **8 astropy/table/column.py** | 1226 | 1264| 322 | 14028 | 108196 | 
| 32 | 9 astropy/cosmology/funcs/comparison.py | 304 | 321| 201 | 14229 | 111699 | 
| 33 | **9 astropy/table/column.py** | 1376 | 1408| 325 | 14554 | 111699 | 
| 34 | 10 astropy/io/ascii/ecsv.py | 236 | 340| 1111 | 15665 | 115896 | 
| 35 | **10 astropy/table/column.py** | 1180 | 1192| 131 | 15796 | 115896 | 
| 36 | 10 astropy/table/table.py | 1337 | 1360| 204 | 16000 | 115896 | 
| 37 | **10 astropy/table/column.py** | 935 | 945| 159 | 16159 | 115896 | 
| 38 | 10 astropy/table/table.py | 1525 | 1583| 453 | 16612 | 115896 | 
| 39 | 11 astropy/utils/masked/core.py | 619 | 646| 296 | 16908 | 126136 | 
| 40 | **11 astropy/table/column.py** | 1194 | 1224| 253 | 17161 | 126136 | 
| 41 | 12 astropy/table/groups.py | 240 | 281| 436 | 17597 | 129312 | 
| 42 | 12 astropy/table/table.py | 1362 | 1374| 124 | 17721 | 129312 | 
| 43 | **12 astropy/table/column.py** | 1672 | 1686| 184 | 17905 | 129312 | 
| 44 | 12 astropy/cosmology/funcs/comparison.py | 205 | 303| 980 | 18885 | 129312 | 
| 45 | 13 astropy/io/ascii/mrt.py | 431 | 532| 1277 | 20162 | 135352 | 
| 46 | 14 astropy/io/votable/exceptions.py | 608 | 627| 166 | 20328 | 148676 | 
| 47 | 14 astropy/table/table.py | 1968 | 2028| 367 | 20695 | 148676 | 
| 48 | **14 astropy/table/column.py** | 1411 | 1488| 745 | 21440 | 148676 | 
| 49 | 14 astropy/io/votable/exceptions.py | 999 | 1036| 236 | 21676 | 148676 | 
| 50 | 15 astropy/io/misc/asdf/tags/transform/tabular.py | 46 | 88| 377 | 22053 | 149415 | 
| 51 | 15 astropy/io/fits/column.py | 812 | 845| 301 | 22354 | 149415 | 
| 52 | 15 astropy/io/fits/column.py | 1467 | 1522| 490 | 22844 | 149415 | 
| 53 | 16 astropy/io/misc/asdf/tags/transform/functional_models.py | 389 | 398| 120 | 22964 | 156874 | 
| 54 | 16 astropy/io/misc/asdf/tags/transform/functional_models.py | 454 | 462| 113 | 23077 | 156874 | 
| 55 | 16 astropy/io/fits/column.py | 951 | 997| 496 | 23573 | 156874 | 
| 56 | 17 astropy/table/table_helpers.py | 9 | 53| 421 | 23994 | 158429 | 
| 57 | 17 astropy/io/misc/asdf/tags/transform/functional_models.py | 423 | 433| 134 | 24128 | 158429 | 
| 58 | 17 astropy/io/fits/column.py | 1237 | 1278| 415 | 24543 | 158429 | 
| 59 | 17 astropy/table/table.py | 2710 | 2730| 191 | 24734 | 158429 | 
| 60 | 17 astropy/io/misc/asdf/tags/transform/functional_models.py | 303 | 311| 118 | 24852 | 158429 | 
| 61 | 17 astropy/io/misc/asdf/tags/transform/functional_models.py | 269 | 280| 145 | 24997 | 158429 | 
| 62 | 17 astropy/table/table.py | 1252 | 1317| 706 | 25703 | 158429 | 
| 63 | 18 astropy/nddata/_testing.py | 4 | 24| 179 | 25882 | 158924 | 
| 64 | 18 astropy/io/fits/column.py | 2311 | 2320| 110 | 25992 | 158924 | 
| 65 | 19 astropy/io/votable/validator/result.py | 236 | 330| 796 | 26788 | 161448 | 
| 66 | 20 astropy/table/meta.py | 159 | 218| 511 | 27299 | 164583 | 
| 67 | **20 astropy/table/column.py** | 351 | 409| 609 | 27908 | 164583 | 
| 68 | 21 astropy/timeseries/core.py | 46 | 101| 443 | 28351 | 165297 | 
| 69 | **21 astropy/table/column.py** | 662 | 678| 175 | 28526 | 165297 | 
| 70 | 21 astropy/io/votable/exceptions.py | 1054 | 1105| 365 | 28891 | 165297 | 
| 71 | 21 astropy/table/table.py | 2680 | 2708| 252 | 29143 | 165297 | 
| 72 | 22 astropy/utils/masked/function_helpers.py | 80 | 154| 773 | 29916 | 174349 | 
| 73 | 23 astropy/io/ascii/__init__.py | 7 | 48| 302 | 30218 | 174687 | 
| 74 | 24 astropy/wcs/docstrings.py | 684 | 773| 634 | 30852 | 196140 | 


### Hint

```
Related details: https://github.com/astropy/astroquery/issues/2440#issuecomment-1155588504
xref https://github.com/numpy/numpy/pull/21041
It was merged 4 days ago, so does this mean it went into the RC before it hits the "nightly wheel" that we tests against here?
ahh, good point, I forgot that the "nightly" is not in fact a daily build, that at least takes the confusion away of how a partial backport could happen that makes the RC fail but the dev still pass.
Perhaps Numpy could have a policy to refresh the "nightly wheel" along with RC to make sure last-minute backport like this won't go unnoticed for those who test against "nightly"? 🤔 
There you go: https://github.com/numpy/numpy/issues/21758
It seems there are two related problems.
1. When a column is unicode, a comparison with bytes now raises a `FutureWarning`, which leads to a failure in the tests. Here, we can either filter out the warning in our tests, or move to the future and raise a `TypeError`.
2. When one of the two is a `MaskedColumn`, the unicode sandwich somehow gets skipped. This is weird...
See https://github.com/numpy/numpy/issues/21770
Looks like Numpy is thinking to [undo the backport](https://github.com/numpy/numpy/issues/21770#issuecomment-1157077479). If that happens, then we have more time to think about this.
Are these errors related to the same numpy backport? Maybe we finally seeing it in "nightly wheel" and it does not look pretty (45 failures over several subpackages) -- https://github.com/astropy/astropy/runs/6918680788?check_suite_focus=true
@pllim - those other errors are actually due to a bug in `Quantity`, where the unit of an `initial` argument is not taken into account (and where units are no longer stripped in numpy). Working on a fix...
Well, *some* of the new failures are resolved by my fix - but at least it also fixes behaviour for all previous versions of numpy! See #13340.
The remainder all seem to be due to a new check on overflow on casting - we're trying to write `1e45` in a `float32` - see #13341
After merging a few PRs to fix other dev failures, these are the remaining ones in `main` now. Please advise on what we should do next to get rid of these 21 failures. Thanks!

Example log: https://github.com/astropy/astropy/runs/6936666794?check_suite_focus=true

\`\`\`
FAILED .../astropy/io/fits/tests/test_connect.py::TestSingleTable::test_simple
FAILED .../astropy/io/fits/tests/test_connect.py::TestSingleTable::test_simple_pathlib
FAILED .../astropy/io/fits/tests/test_connect.py::TestSingleTable::test_simple_meta
FAILED .../astropy/io/fits/tests/test_connect.py::TestSingleTable::test_simple_noextension
FAILED .../astropy/io/fits/tests/test_connect.py::TestSingleTable::test_with_units[Table]
FAILED .../astropy/io/fits/tests/test_connect.py::TestSingleTable::test_with_units[QTable]
FAILED .../astropy/io/fits/tests/test_connect.py::TestSingleTable::test_with_format[Table]
FAILED .../astropy/io/fits/tests/test_connect.py::TestSingleTable::test_with_format[QTable]
FAILED .../astropy/io/fits/tests/test_connect.py::TestSingleTable::test_character_as_bytes[False]
FAILED .../astropy/io/fits/tests/test_connect.py::TestSingleTable::test_character_as_bytes[True]
FAILED .../astropy/modeling/tests/test_models_quantities.py::test_models_evaluate_with_units[model11]
FAILED .../astropy/modeling/tests/test_models_quantities.py::test_models_evaluate_with_units[model22]
FAILED .../astropy/modeling/tests/test_models_quantities.py::test_models_evaluate_with_units_x_array[model11]
FAILED .../astropy/modeling/tests/test_models_quantities.py::test_models_evaluate_with_units_x_array[model22]
FAILED .../astropy/table/tests/test_column.py::test_col_unicode_sandwich_unicode
FAILED .../astropy/table/tests/test_column.py::test_unicode_sandwich_compare[MaskedColumn-MaskedColumn]
FAILED .../astropy/table/tests/test_column.py::test_unicode_sandwich_compare[Column-MaskedColumn]
FAILED .../astropy/table/tests/test_column.py::test_unicode_sandwich_compare[Column-Column]
FAILED .../astropy/table/tests/test_column.py::test_unicode_sandwich_compare[str-MaskedColumn]
FAILED .../astropy/table/tests/test_column.py::test_unicode_sandwich_compare[list-MaskedColumn]
FAILED .../astropy/table/tests/test_init_table.py::TestInitFromTable::test_partial_names_dtype[True]
\`\`\`
FWIW, I have #13349 that picked up the RC in question here and you can see there are only 17 failures (4 less from using numpy's "nightly wheel").

Example log: https://github.com/astropy/astropy/runs/6937240337?check_suite_focus=true

\`\`\`
FAILED .../astropy/io/fits/tests/test_connect.py::TestSingleTable::test_simple
FAILED .../astropy/io/fits/tests/test_connect.py::TestSingleTable::test_simple_pathlib
FAILED .../astropy/io/fits/tests/test_connect.py::TestSingleTable::test_simple_meta
FAILED .../astropy/io/fits/tests/test_connect.py::TestSingleTable::test_simple_noextension
FAILED .../astropy/io/fits/tests/test_connect.py::TestSingleTable::test_with_units[Table]
FAILED .../astropy/io/fits/tests/test_connect.py::TestSingleTable::test_with_units[QTable]
FAILED .../astropy/io/fits/tests/test_connect.py::TestSingleTable::test_with_format[Table]
FAILED .../astropy/io/fits/tests/test_connect.py::TestSingleTable::test_with_format[QTable]
FAILED .../astropy/io/fits/tests/test_connect.py::TestSingleTable::test_character_as_bytes[False]
FAILED .../astropy/io/fits/tests/test_connect.py::TestSingleTable::test_character_as_bytes[True]
FAILED .../astropy/io/misc/tests/test_hdf5.py::test_read_write_unicode_to_hdf5
FAILED .../astropy/table/tests/test_column.py::test_col_unicode_sandwich_unicode
FAILED .../astropy/table/tests/test_column.py::test_unicode_sandwich_compare[MaskedColumn-MaskedColumn]
FAILED .../astropy/table/tests/test_column.py::test_unicode_sandwich_compare[Column-MaskedColumn]
FAILED .../astropy/table/tests/test_column.py::test_unicode_sandwich_compare[Column-Column]
FAILED .../astropy/table/tests/test_column.py::test_unicode_sandwich_compare[str-MaskedColumn]
FAILED .../astropy/table/tests/test_column.py::test_unicode_sandwich_compare[list-MaskedColumn]
\`\`\`

So...

# In both "nightly wheel" and RC

\`\`\`
FAILED .../astropy/io/fits/tests/test_connect.py::TestSingleTable::test_simple
FAILED .../astropy/io/fits/tests/test_connect.py::TestSingleTable::test_simple_pathlib
FAILED .../astropy/io/fits/tests/test_connect.py::TestSingleTable::test_simple_meta
FAILED .../astropy/io/fits/tests/test_connect.py::TestSingleTable::test_simple_noextension
FAILED .../astropy/io/fits/tests/test_connect.py::TestSingleTable::test_with_units[Table]
FAILED .../astropy/io/fits/tests/test_connect.py::TestSingleTable::test_with_units[QTable]
FAILED .../astropy/io/fits/tests/test_connect.py::TestSingleTable::test_with_format[Table]
FAILED .../astropy/io/fits/tests/test_connect.py::TestSingleTable::test_with_format[QTable]
FAILED .../astropy/io/fits/tests/test_connect.py::TestSingleTable::test_character_as_bytes[False]
FAILED .../astropy/io/fits/tests/test_connect.py::TestSingleTable::test_character_as_bytes[True]
FAILED .../astropy/table/tests/test_column.py::test_col_unicode_sandwich_unicode
FAILED .../astropy/table/tests/test_column.py::test_unicode_sandwich_compare[MaskedColumn-MaskedColumn]
FAILED .../astropy/table/tests/test_column.py::test_unicode_sandwich_compare[Column-MaskedColumn]
FAILED .../astropy/table/tests/test_column.py::test_unicode_sandwich_compare[Column-Column]
FAILED .../astropy/table/tests/test_column.py::test_unicode_sandwich_compare[str-MaskedColumn]
FAILED .../astropy/table/tests/test_column.py::test_unicode_sandwich_compare[list-MaskedColumn]
\`\`\`

# RC only

I don't understand why this one only pops up in the RC but not in dev. 🤷 

\`\`\`
FAILED .../astropy/io/misc/tests/test_hdf5.py::test_read_write_unicode_to_hdf5
\`\`\`

# "nightly wheel" only

\`\`\`
FAILED .../astropy/modeling/tests/test_models_quantities.py::test_models_evaluate_with_units[model11]
FAILED .../astropy/modeling/tests/test_models_quantities.py::test_models_evaluate_with_units[model22]
FAILED .../astropy/modeling/tests/test_models_quantities.py::test_models_evaluate_with_units_x_array[model11]
FAILED .../astropy/modeling/tests/test_models_quantities.py::test_models_evaluate_with_units_x_array[model22]
FAILED .../astropy/table/tests/test_init_table.py::TestInitFromTable::test_partial_names_dtype[True]
\`\`\`
@pllim - with the corrections to the rc3, i.e., numpy 1.23.x (1.23.0rc3+10.gcc0e08d20), the failures in `io.fits`, `io.misc`, and `table` are all gone -- all tests pass! So, we can now move to address the problems in `numpy-dev`.
Will there be a rc4?
Looks like numpy released 1.23 🤞 
I am anxiously waiting for the "nightly wheel" to catch up. The other CI jobs passing even after the new release, so at least that is a good sign. 🤞 
I actually don't know that `-dev` was changed too - I think they just reverted the bad commit from 1.23, with the idea that for 1.24 there would be a fix (IIRC, https://github.com/numpy/numpy/pull/21812 would solve at least some of the problems)
```

## Patch

```diff
diff --git a/astropy/table/column.py b/astropy/table/column.py
--- a/astropy/table/column.py
+++ b/astropy/table/column.py
@@ -297,31 +297,23 @@ def _make_compare(oper):
     oper : str
         Operator name
     """
-    swapped_oper = {'__eq__': '__eq__',
-                    '__ne__': '__ne__',
-                    '__gt__': '__lt__',
-                    '__lt__': '__gt__',
-                    '__ge__': '__le__',
-                    '__le__': '__ge__'}[oper]
-
     def _compare(self, other):
         op = oper  # copy enclosed ref to allow swap below
 
-        # Special case to work around #6838.  Other combinations work OK,
-        # see tests.test_column.test_unicode_sandwich_compare().  In this
-        # case just swap self and other.
-        #
-        # This is related to an issue in numpy that was addressed in np 1.13.
-        # However that fix does not make this problem go away, but maybe
-        # future numpy versions will do so.  NUMPY_LT_1_13 to get the
-        # attention of future maintainers to check (by deleting or versioning
-        # the if block below).  See #6899 discussion.
-        # 2019-06-21: still needed with numpy 1.16.
-        if (isinstance(self, MaskedColumn) and self.dtype.kind == 'U'
-                and isinstance(other, MaskedColumn) and other.dtype.kind == 'S'):
-            self, other = other, self
-            op = swapped_oper
+        # If other is a Quantity, we should let it do the work, since
+        # it can deal with our possible unit (which, for MaskedColumn,
+        # would get dropped below, as '.data' is accessed in super()).
+        if isinstance(other, Quantity):
+            return NotImplemented
 
+        # If we are unicode and other is a column with bytes, defer to it for
+        # doing the unicode sandwich.  This avoids problems like those
+        # discussed in #6838 and #6899.
+        if (self.dtype.kind == 'U'
+                and isinstance(other, Column) and other.dtype.kind == 'S'):
+            return NotImplemented
+
+        # If we are bytes, encode other as needed.
         if self.dtype.char == 'S':
             other = self._encode_str(other)
 
@@ -1531,10 +1523,11 @@ def __new__(cls, data=None, name=None, mask=None, fill_value=None,
 
         # Note: do not set fill_value in the MaskedArray constructor because this does not
         # go through the fill_value workarounds.
-        if fill_value is None and getattr(data, 'fill_value', None) is not None:
-            # Coerce the fill_value to the correct type since `data` may be a
-            # different dtype than self.
-            fill_value = np.array(data.fill_value, self.dtype)[()]
+        if fill_value is None:
+            data_fill_value = getattr(data, 'fill_value', None)
+            if (data_fill_value is not None
+                    and data_fill_value != np.ma.default_fill_value(data.dtype)):
+                fill_value = np.array(data_fill_value, self.dtype)[()]
         self.fill_value = fill_value
 
         self.parent_table = None

```

## Test Patch

```diff
diff --git a/astropy/table/tests/test_column.py b/astropy/table/tests/test_column.py
--- a/astropy/table/tests/test_column.py
+++ b/astropy/table/tests/test_column.py
@@ -2,6 +2,7 @@
 
 from astropy.utils.tests.test_metadata import MetaBaseTest
 import operator
+import warnings
 
 import pytest
 import numpy as np
@@ -773,7 +774,10 @@ def test_col_unicode_sandwich_unicode():
     assert ok.dtype.char == '?'
     assert np.all(ok)
 
-    assert np.all(c != [uba8, b'def'])
+    with warnings.catch_warnings():
+        # Ignore the FutureWarning in numpy >=1.24 (it is OK).
+        warnings.filterwarnings('ignore', message='.*elementwise comparison failed.*')
+        assert np.all(c != [uba8, b'def'])
 
 
 def test_masked_col_unicode_sandwich():

```


## Code snippets

### 1 - astropy/table/column.py:

Start line: 307, End line: 333

```python
def _make_compare(oper):
    # ... other code

    def _compare(self, other):
        op = oper  # copy enclosed ref to allow swap below

        # Special case to work around #6838.  Other combinations work OK,
        # see tests.test_column.test_unicode_sandwich_compare().  In this
        # case just swap self and other.
        #
        # This is related to an issue in numpy that was addressed in np 1.13.
        # However that fix does not make this problem go away, but maybe
        # future numpy versions will do so.  NUMPY_LT_1_13 to get the
        # attention of future maintainers to check (by deleting or versioning
        # the if block below).  See #6899 discussion.
        # 2019-06-21: still needed with numpy 1.16.
        if (isinstance(self, MaskedColumn) and self.dtype.kind == 'U'
                and isinstance(other, MaskedColumn) and other.dtype.kind == 'S'):
            self, other = other, self
            op = swapped_oper

        if self.dtype.char == 'S':
            other = self._encode_str(other)

        # Now just let the regular ndarray.__eq__, etc., take over.
        result = getattr(super(Column, self), op)(other)
        # But we should not return Column instances for this case.
        return result.data if isinstance(result, Column) else result

    return _compare
```
### 2 - astropy/table/column.py:

Start line: 778, End line: 805

```python
class BaseColumn(_ColumnGetitemShim, np.ndarray):

    def attrs_equal(self, col):
        """Compare the column attributes of ``col`` to this object.

        The comparison attributes are: ``name``, ``unit``, ``dtype``,
        ``format``, ``description``, and ``meta``.

        Parameters
        ----------
        col : Column
            Comparison column

        Returns
        -------
        equal : bool
            True if all attributes are equal
        """
        if not isinstance(col, BaseColumn):
            raise ValueError('Comparison `col` must be a Column or '
                             'MaskedColumn object')

        attrs = ('name', 'unit', 'dtype', 'format', 'description', 'meta')
        equal = all(getattr(self, x) == getattr(col, x) for x in attrs)

        return equal

    @property
    def _formatter(self):
        return FORMATTER if (self.parent_table is None) else self.parent_table.formatter
```
### 3 - astropy/io/fits/column.py:

Start line: 1137, End line: 1198

```python
class Column(NotifierMixin):

    @classmethod
    def _verify_keywords(cls, name=None, format=None, unit=None, null=None,
                         bscale=None, bzero=None, disp=None, start=None,
                         dim=None, ascii=None, coord_type=None, coord_unit=None,
                         coord_ref_point=None, coord_ref_value=None,
                         coord_inc=None, time_ref_pos=None):
        # ... other code

        if coord_type is not None and coord_type != '':
            msg = None
            if not isinstance(coord_type, str):
                msg = (
                    "Coordinate/axis type option (TCTYPn) must be a string "
                    "(got {!r}). The invalid keyword will be ignored for the "
                    "purpose of formatting this column.".format(coord_type))
            elif len(coord_type) > 8:
                msg = (
                    "Coordinate/axis type option (TCTYPn) must be a string "
                    "of atmost 8 characters (got {!r}). The invalid keyword "
                    "will be ignored for the purpose of formatting this "
                    "column.".format(coord_type))

            if msg is None:
                valid['coord_type'] = coord_type
            else:
                invalid['coord_type'] = (coord_type, msg)

        if coord_unit is not None and coord_unit != '':
            msg = None
            if not isinstance(coord_unit, str):
                msg = (
                    "Coordinate/axis unit option (TCUNIn) must be a string "
                    "(got {!r}). The invalid keyword will be ignored for the "
                    "purpose of formatting this column.".format(coord_unit))

            if msg is None:
                valid['coord_unit'] = coord_unit
            else:
                invalid['coord_unit'] = (coord_unit, msg)

        for k, v in [('coord_ref_point', coord_ref_point),
                     ('coord_ref_value', coord_ref_value),
                     ('coord_inc', coord_inc)]:
            if v is not None and v != '':
                msg = None
                if not isinstance(v, numbers.Real):
                    msg = (
                        "Column {} option ({}n) must be a real floating type (got {!r}). "
                        "The invalid value will be ignored for the purpose of formatting "
                        "the data in this column.".format(k, ATTRIBUTE_TO_KEYWORD[k], v))

                if msg is None:
                    valid[k] = v
                else:
                    invalid[k] = (v, msg)

        if time_ref_pos is not None and time_ref_pos != '':
            msg = None
            if not isinstance(time_ref_pos, str):
                msg = (
                    "Time coordinate reference position option (TRPOSn) must be "
                    "a string (got {!r}). The invalid keyword will be ignored for "
                    "the purpose of formatting this column.".format(time_ref_pos))

            if msg is None:
                valid['time_ref_pos'] = time_ref_pos
            else:
                invalid['time_ref_pos'] = (time_ref_pos, msg)

        return valid, invalid
```
### 4 - astropy/table/column.py:

Start line: 3, End line: 55

```python
import itertools
import warnings
import weakref

from copy import deepcopy

import numpy as np
from numpy import ma

from astropy.units import Unit, Quantity, StructuredUnit
from astropy.utils.console import color_print
from astropy.utils.metadata import MetaData
from astropy.utils.data_info import BaseColumnInfo, dtype_info_name
from astropy.utils.misc import dtype_bytes_or_chars
from . import groups
from . import pprint

# These "shims" provide __getitem__ implementations for Column and MaskedColumn
from ._column_mixins import _ColumnGetitemShim, _MaskedColumnGetitemShim

# Create a generic TableFormatter object for use by bare columns with no
# parent table.
FORMATTER = pprint.TableFormatter()


class StringTruncateWarning(UserWarning):
    """
    Warning class for when a string column is assigned a value
    that gets truncated because the base (numpy) string length
    is too short.

    This does not inherit from AstropyWarning because we want to use
    stacklevel=2 to show the user where the issue occurred in their code.
    """
    pass


# Always emit this warning, not just the first instance
warnings.simplefilter('always', StringTruncateWarning)


def _auto_names(n_cols):
    from . import conf
    return [str(conf.auto_colname).format(i) for i in range(n_cols)]


# list of one and two-dimensional comparison functions, which sometimes return
# a Column class and sometimes a plain array. Used in __array_wrap__ to ensure
# they only return plain (masked) arrays (see #1446 and #1685)
_comparison_functions = {
    np.greater, np.greater_equal, np.less, np.less_equal,
    np.not_equal, np.equal,
    np.isfinite, np.isinf, np.isnan, np.sign, np.signbit}
```
### 5 - astropy/table/column.py:

Start line: 290, End line: 305

```python
def _make_compare(oper):
    """
    Make Column comparison methods which encode the ``other`` object to utf-8
    in the case of a bytestring dtype for Py3+.

    Parameters
    ----------
    oper : str
        Operator name
    """
    swapped_oper = {'__eq__': '__eq__',
                    '__ne__': '__ne__',
                    '__gt__': '__lt__',
                    '__lt__': '__gt__',
                    '__ge__': '__le__',
                    '__le__': '__ge__'}[oper]
    # ... other code
```
### 6 - astropy/table/table.py:

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
### 7 - astropy/io/fits/column.py:

Start line: 676, End line: 702

```python
class Column(NotifierMixin):

    def __repr__(self):
        text = ''
        for attr in KEYWORD_ATTRIBUTES:
            value = getattr(self, attr)
            if value is not None:
                text += attr + ' = ' + repr(value) + '; '
        return text[:-2]

    def __eq__(self, other):
        """
        Two columns are equal if their name and format are the same.  Other
        attributes aren't taken into account at this time.
        """

        # According to the FITS standard column names must be case-insensitive
        a = (self.name.lower(), self.format)
        b = (other.name.lower(), other.format)
        return a == b

    def __hash__(self):
        """
        Like __eq__, the hash of a column should be based on the unique column
        name and format, and be case-insensitive with respect to the column
        name.
        """

        return hash((self.name.lower(), self.format))
```
### 8 - astropy/table/column.py:

Start line: 1688, End line: 1715

```python
class MaskedColumn(Column, _MaskedColumnGetitemShim, ma.MaskedArray):

    def __setitem__(self, index, value):
        # Issue warning for string assignment that truncates ``value``
        if self.dtype.char == 'S':
            value = self._encode_str(value)

        if issubclass(self.dtype.type, np.character):
            # Account for a bug in np.ma.MaskedArray setitem.
            # https://github.com/numpy/numpy/issues/8624
            value = np.ma.asanyarray(value, dtype=self.dtype.type)

            # Check for string truncation after filling masked items with
            # empty (zero-length) string.  Note that filled() does not make
            # a copy if there are no masked items.
            self._check_string_truncate(value.filled(''))

        # update indices
        self.info.adjust_indices(index, value, len(self))

        ma.MaskedArray.__setitem__(self, index, value)

    # We do this to make the methods show up in the API docs
    name = BaseColumn.name
    copy = BaseColumn.copy
    more = BaseColumn.more
    pprint = BaseColumn.pprint
    pformat = BaseColumn.pformat
    convert_unit_to = BaseColumn.convert_unit_to
```
### 9 - astropy/io/fits/column.py:

Start line: 998, End line: 1068

```python
class Column(NotifierMixin):

    @classmethod
    def _verify_keywords(cls, name=None, format=None, unit=None, null=None,
                         bscale=None, bzero=None, disp=None, start=None,
                         dim=None, ascii=None, coord_type=None, coord_unit=None,
                         coord_ref_point=None, coord_ref_value=None,
                         coord_inc=None, time_ref_pos=None):
        # ... other code
        if null is not None and null != '':
            msg = None
            if isinstance(format, _AsciiColumnFormat):
                null = str(null)
                if len(null) > format.width:
                    msg = (
                        "ASCII table null option (TNULLn) is longer than "
                        "the column's character width and will be truncated "
                        "(got {!r}).".format(null))
            else:
                tnull_formats = ('B', 'I', 'J', 'K')

                if not _is_int(null):
                    # Make this an exception instead of a warning, since any
                    # non-int value is meaningless
                    msg = (
                        'Column null option (TNULLn) must be an integer for '
                        'binary table columns (got {!r}).  The invalid value '
                        'will be ignored for the purpose of formatting '
                        'the data in this column.'.format(null))

                elif not (format.format in tnull_formats or
                          (format.format in ('P', 'Q') and
                           format.p_format in tnull_formats)):
                    # TODO: We should also check that TNULLn's integer value
                    # is in the range allowed by the column's format
                    msg = (
                        'Column null option (TNULLn) is invalid for binary '
                        'table columns of type {!r} (got {!r}).  The invalid '
                        'value will be ignored for the purpose of formatting '
                        'the data in this column.'.format(format, null))

            if msg is None:
                valid['null'] = null
            else:
                invalid['null'] = (null, msg)

        # Validate the disp option
        # TODO: Add full parsing and validation of TDISPn keywords
        if disp is not None and disp != '':
            msg = None
            if not isinstance(disp, str):
                msg = (
                    f'Column disp option (TDISPn) must be a string (got '
                    f'{disp!r}). The invalid value will be ignored for the '
                    'purpose of formatting the data in this column.')

            elif (isinstance(format, _AsciiColumnFormat) and
                    disp[0].upper() == 'L'):
                # disp is at least one character long and has the 'L' format
                # which is not recognized for ASCII tables
                msg = (
                    "Column disp option (TDISPn) may not use the 'L' format "
                    "with ASCII table columns.  The invalid value will be "
                    "ignored for the purpose of formatting the data in this "
                    "column.")

            if msg is None:
                try:
                    _parse_tdisp_format(disp)
                    valid['disp'] = disp
                except VerifyError as err:
                    msg = (
                        f'Column disp option (TDISPn) failed verification: '
                        f'{err!s} The invalid value will be ignored for the '
                        'purpose of formatting the data in this column.')
                    invalid['disp'] = (disp, msg)
            else:
                invalid['disp'] = (disp, msg)

        # Validate the start option
        # ... other code
```
### 10 - astropy/table/column.py:

Start line: 1266, End line: 1286

```python
class Column(BaseColumn):

    def __setitem__(self, index, value):
        if self.dtype.char == 'S':
            value = self._encode_str(value)

        # Issue warning for string assignment that truncates ``value``
        if issubclass(self.dtype.type, np.character):
            self._check_string_truncate(value)

        # update indices
        self.info.adjust_indices(index, value, len(self))

        # Set items using a view of the underlying data, as it gives an
        # order-of-magnitude speed-up. [#2994]
        self.data[index] = value

    __eq__ = _make_compare('__eq__')
    __ne__ = _make_compare('__ne__')
    __gt__ = _make_compare('__gt__')
    __lt__ = _make_compare('__lt__')
    __ge__ = _make_compare('__ge__')
    __le__ = _make_compare('__le__')
```
### 16 - astropy/table/column.py:

Start line: 1490, End line: 1546

```python
class MaskedColumn(Column, _MaskedColumnGetitemShim, ma.MaskedArray):

    def __new__(cls, data=None, name=None, mask=None, fill_value=None,
                dtype=None, shape=(), length=0,
                description=None, unit=None, format=None, meta=None,
                copy=False, copy_indices=True):

        if mask is None:
            # If mask is None then we need to determine the mask (if any) from the data.
            # The naive method is looking for a mask attribute on data, but this can fail,
            # see #8816.  Instead use ``MaskedArray`` to do the work.
            mask = ma.MaskedArray(data).mask
            if mask is np.ma.nomask:
                # Handle odd-ball issue with np.ma.nomask (numpy #13758), and see below.
                mask = False
            elif copy:
                mask = mask.copy()

        elif mask is np.ma.nomask:
            # Force the creation of a full mask array as nomask is tricky to
            # use and will fail in an unexpected manner when setting a value
            # to the mask.
            mask = False
        else:
            mask = deepcopy(mask)

        # Create self using MaskedArray as a wrapper class, following the example of
        # class MSubArray in
        # https://github.com/numpy/numpy/blob/maintenance/1.8.x/numpy/ma/tests/test_subclassing.py
        # This pattern makes it so that __array_finalize__ is called as expected (e.g. #1471 and
        # https://github.com/astropy/astropy/commit/ff6039e8)

        # First just pass through all args and kwargs to BaseColumn, then wrap that object
        # with MaskedArray.
        self_data = BaseColumn(data, dtype=dtype, shape=shape, length=length, name=name,
                               unit=unit, format=format, description=description,
                               meta=meta, copy=copy, copy_indices=copy_indices)
        self = ma.MaskedArray.__new__(cls, data=self_data, mask=mask)
        # The above process preserves info relevant for Column, but this does
        # not include serialize_method (and possibly other future attributes)
        # relevant for MaskedColumn, so we set info explicitly.
        if 'info' in getattr(data, '__dict__', {}):
            self.info = data.info

        # Note: do not set fill_value in the MaskedArray constructor because this does not
        # go through the fill_value workarounds.
        if fill_value is None and getattr(data, 'fill_value', None) is not None:
            # Coerce the fill_value to the correct type since `data` may be a
            # different dtype than self.
            fill_value = np.array(data.fill_value, self.dtype)[()]
        self.fill_value = fill_value

        self.parent_table = None

        # needs to be done here since self doesn't come from BaseColumn.__new__
        for index in self.indices:
            index.replace_col(self_data, self)

        return self
```
### 19 - astropy/table/column.py:

Start line: 1166, End line: 1178

```python
class Column(BaseColumn):

    def __new__(cls, data=None, name=None,
                dtype=None, shape=(), length=0,
                description=None, unit=None, format=None, meta=None,
                copy=False, copy_indices=True):

        if isinstance(data, MaskedColumn) and np.any(data.mask):
            raise TypeError("Cannot convert a MaskedColumn with masked value to a Column")

        self = super().__new__(
            cls, data=data, name=name, dtype=dtype, shape=shape, length=length,
            description=description, unit=unit, format=format, meta=meta,
            copy=copy, copy_indices=copy_indices)
        return self
```
### 25 - astropy/table/column.py:

Start line: 480, End line: 537

```python
class BaseColumn(_ColumnGetitemShim, np.ndarray):

    meta = MetaData()

    def __new__(cls, data=None, name=None,
                dtype=None, shape=(), length=0,
                description=None, unit=None, format=None, meta=None,
                copy=False, copy_indices=True):
        if data is None:
            self_data = np.zeros((length,)+shape, dtype=dtype)
        elif isinstance(data, BaseColumn) and hasattr(data, '_name'):
            # When unpickling a MaskedColumn, ``data`` will be a bare
            # BaseColumn with none of the expected attributes.  In this case
            # do NOT execute this block which initializes from ``data``
            # attributes.
            self_data = np.array(data.data, dtype=dtype, copy=copy)
            if description is None:
                description = data.description
            if unit is None:
                unit = unit or data.unit
            if format is None:
                format = data.format
            if meta is None:
                meta = data.meta
            if name is None:
                name = data.name
        elif isinstance(data, Quantity):
            if unit is None:
                self_data = np.array(data, dtype=dtype, copy=copy)
                unit = data.unit
            else:
                self_data = Quantity(data, unit, dtype=dtype, copy=copy).value
            # If 'info' has been defined, copy basic properties (if needed).
            if 'info' in data.__dict__:
                if description is None:
                    description = data.info.description
                if format is None:
                    format = data.info.format
                if meta is None:
                    meta = data.info.meta

        else:
            if np.dtype(dtype).char == 'S':
                data = cls._encode_str(data)
            self_data = np.array(data, dtype=dtype, copy=copy)

        self = self_data.view(cls)
        self._name = None if name is None else str(name)
        self._parent_table = None
        self.unit = unit
        self._format = format
        self.description = description
        self.meta = meta
        self.indices = deepcopy(getattr(data, 'indices', [])) if copy_indices else []
        for index in self.indices:
            index.replace_col(data, self)

        return self
```
### 26 - astropy/table/column.py:

Start line: 1068, End line: 1092

```python
class BaseColumn(_ColumnGetitemShim, np.ndarray):

    @staticmethod
    def _encode_str(value):
        """
        Encode anything that is unicode-ish as utf-8.  This method is only
        called for Py3+.
        """
        if isinstance(value, str):
            value = value.encode('utf-8')
        elif isinstance(value, bytes) or value is np.ma.masked:
            pass
        else:
            arr = np.asarray(value)
            if arr.dtype.char == 'U':
                arr = np.char.encode(arr, encoding='utf-8')
                if isinstance(value, np.ma.MaskedArray):
                    arr = np.ma.array(arr, mask=value.mask, copy=False)
            value = arr

        return value

    def tolist(self):
        if self.dtype.kind == 'S':
            return np.chararray.decode(self, encoding='utf-8').tolist()
        else:
            return super().tolist()
```
### 31 - astropy/table/column.py:

Start line: 1226, End line: 1264

```python
class Column(BaseColumn):

    def _repr_html_(self):
        return self._base_repr_(html=True)

    def __repr__(self):
        return self._base_repr_(html=False)

    def __str__(self):
        # If scalar then just convert to correct numpy type and use numpy repr
        if self.ndim == 0:
            return str(self.item())

        lines, outs = self._formatter._pformat_col(self)
        return '\n'.join(lines)

    def __bytes__(self):
        return str(self).encode('utf-8')

    def _check_string_truncate(self, value):
        """
        Emit a warning if any elements of ``value`` will be truncated when
        ``value`` is assigned to self.
        """
        # Convert input ``value`` to the string dtype of this column and
        # find the length of the longest string in the array.
        value = np.asanyarray(value, dtype=self.dtype.type)
        if value.size == 0:
            return
        value_str_len = np.char.str_len(value).max()

        # Parse the array-protocol typestring (e.g. '|U15') of self.dtype which
        # has the character repeat count on the right side.
        self_str_len = dtype_bytes_or_chars(self.dtype)

        if value_str_len > self_str_len:
            warnings.warn('truncated right side string(s) longer than {} '
                          'character(s) during assignment'
                          .format(self_str_len),
                          StringTruncateWarning,
                          stacklevel=3)
```
### 33 - astropy/table/column.py:

Start line: 1376, End line: 1408

```python
class MaskedColumnInfo(ColumnInfo):

    def _represent_as_dict(self):
        out = super()._represent_as_dict()
        # If we are a structured masked column, then our parent class,
        # ColumnInfo, will already have set up a dict with masked parts,
        # which will be serialized later, so no further work needed here.
        if self._parent.dtype.names is not None:
            return out

        col = self._parent

        # If the serialize method for this context (e.g. 'fits' or 'ecsv') is
        # 'data_mask', that means to serialize using an explicit mask column.
        method = self.serialize_method[self._serialize_context]

        if method == 'data_mask':
            # Note: a driver here is a performance issue in #8443 where repr() of a
            # np.ma.MaskedArray value is up to 10 times slower than repr of a normal array
            # value.  So regardless of whether there are masked elements it is useful to
            # explicitly define this as a serialized column and use col.data.data (ndarray)
            # instead of letting it fall through to the "standard" serialization machinery.
            out['data'] = col.data.data

            if np.any(col.mask):
                # Only if there are actually masked elements do we add the ``mask`` column
                out['mask'] = col.mask

        elif method == 'null_value':
            pass

        else:
            raise ValueError('serialize method must be either "data_mask" or "null_value"')

        return out
```
### 35 - astropy/table/column.py:

Start line: 1180, End line: 1192

```python
class Column(BaseColumn):

    def __setattr__(self, item, value):
        if not isinstance(self, MaskedColumn) and item == "mask":
            raise AttributeError("cannot set mask value to a column in non-masked Table")
        super().__setattr__(item, value)

        if item == 'unit' and issubclass(self.dtype.type, np.number):
            try:
                converted = self.parent_table._convert_col_for_table(self)
            except AttributeError:  # Either no parent table or parent table is None
                pass
            else:
                if converted is not self:
                    self.parent_table.replace_column(self.name, converted)
```
### 37 - astropy/table/column.py:

Start line: 935, End line: 945

```python
class BaseColumn(_ColumnGetitemShim, np.ndarray):

    def searchsorted(self, v, side='left', sorter=None):
        # For bytes type data, encode the `v` value as UTF-8 (if necessary) before
        # calling searchsorted. This prevents a factor of 1000 slowdown in
        # searchsorted in this case.
        a = self.data
        if a.dtype.kind == 'S' and not isinstance(v, bytes):
            v = np.asarray(v)
            if v.dtype.kind == 'U':
                v = np.char.encode(v, 'utf-8')
        return np.searchsorted(a, v, side=side, sorter=sorter)
    searchsorted.__doc__ = np.ndarray.searchsorted.__doc__
```
### 40 - astropy/table/column.py:

Start line: 1194, End line: 1224

```python
class Column(BaseColumn):

    def _base_repr_(self, html=False):
        # If scalar then just convert to correct numpy type and use numpy repr
        if self.ndim == 0:
            return repr(self.item())

        descr_vals = [self.__class__.__name__]
        unit = None if self.unit is None else str(self.unit)
        shape = None if self.ndim <= 1 else self.shape[1:]
        for attr, val in (('name', self.name),
                          ('dtype', dtype_info_name(self.dtype)),
                          ('shape', shape),
                          ('unit', unit),
                          ('format', self.format),
                          ('description', self.description),
                          ('length', len(self))):

            if val is not None:
                descr_vals.append(f'{attr}={val!r}')

        descr = '<' + ' '.join(descr_vals) + '>\n'

        if html:
            from astropy.utils.xml.writer import xml_escape
            descr = xml_escape(descr)

        data_lines, outs = self._formatter._pformat_col(
            self, show_name=False, show_unit=False, show_length=False, html=html)

        out = descr + '\n'.join(data_lines)

        return out
```
### 43 - astropy/table/column.py:

Start line: 1672, End line: 1686

```python
class MaskedColumn(Column, _MaskedColumnGetitemShim, ma.MaskedArray):

    def _copy_attrs_slice(self, out):
        # Fixes issue #3023: when calling getitem with a MaskedArray subclass
        # the original object attributes are not copied.
        if out.__class__ is self.__class__:
            # TODO: this part is essentially the same as what is done in
            # __array_finalize__ and could probably be called directly in our
            # override of __getitem__ in _columns_mixins.pyx). Refactor?
            if 'info' in self.__dict__:
                out.info = self.info
            out.parent_table = None
            # we need this because __getitem__ does a shallow copy of indices
            if out.indices is self.indices:
                out.indices = []
            out._copy_attrs(self)
        return out
```
### 48 - astropy/table/column.py:

Start line: 1411, End line: 1488

```python
class MaskedColumn(Column, _MaskedColumnGetitemShim, ma.MaskedArray):
    """Define a masked data column for use in a Table object.

    Parameters
    ----------
    data : list, ndarray, or None
        Column data values
    name : str
        Column name and key for reference within Table
    mask : list, ndarray or None
        Boolean mask for which True indicates missing or invalid data
    fill_value : float, int, str, or None
        Value used when filling masked column elements
    dtype : `~numpy.dtype`-like
        Data type for column
    shape : tuple or ()
        Dimensions of a single row element in the column data
    length : int or 0
        Number of row elements in column data
    description : str or None
        Full description of column
    unit : str or None
        Physical unit
    format : str, None, or callable
        Format string for outputting column values.  This can be an
        "old-style" (``format % value``) or "new-style" (`str.format`)
        format specification string or a function or any callable object that
        accepts a single value and returns a string.
    meta : dict-like or None
        Meta-data associated with the column

    Examples
    --------
    A MaskedColumn is similar to a Column except that it includes ``mask`` and
    ``fill_value`` attributes.  It can be created in two different ways:

    - Provide a ``data`` value but not ``shape`` or ``length`` (which are
      inferred from the data).

      Examples::

        col = MaskedColumn(data=[1, 2], name='name')
        col = MaskedColumn(data=[1, 2], name='name', mask=[True, False])
        col = MaskedColumn(data=[1, 2], name='name', dtype=float, fill_value=99)

      The ``mask`` argument will be cast as a boolean array and specifies
      which elements are considered to be missing or invalid.

      The ``dtype`` argument can be any value which is an acceptable
      fixed-size data-type initializer for the numpy.dtype() method.  See
      `<https://numpy.org/doc/stable/reference/arrays.dtypes.html>`_.
      Examples include:

      - Python non-string type (float, int, bool)
      - Numpy non-string type (e.g. np.float32, np.int64, np.bool\\_)
      - Numpy.dtype array-protocol type strings (e.g. 'i4', 'f8', 'S15')

      If no ``dtype`` value is provide then the type is inferred using
      ``np.array(data)``.  When ``data`` is provided then the ``shape``
      and ``length`` arguments are ignored.

    - Provide ``length`` and optionally ``shape``, but not ``data``

      Examples::

        col = MaskedColumn(name='name', length=5)
        col = MaskedColumn(name='name', dtype=int, length=10, shape=(3,4))

      The default ``dtype`` is ``np.float64``.  The ``shape`` argument is the
      array shape of a single cell in the column.

    To access the ``Column`` data as a raw `numpy.ma.MaskedArray` object, you can
    use one of the ``data`` or ``value`` attributes (which are equivalent)::

        col.data
        col.value
    """
    info = MaskedColumnInfo()
```
### 67 - astropy/table/column.py:

Start line: 351, End line: 409

```python
class ColumnInfo(BaseColumnInfo):

    def _represent_as_dict(self):
        result = super()._represent_as_dict()
        names = self._parent.dtype.names
        # For a regular column, we are done, but for a structured
        # column, we use a SerializedColumns to store the pieces.
        if names is None:
            return result

        from .serialize import SerializedColumn

        data = SerializedColumn()
        # If this column has a StructuredUnit, we split it and store
        # it on the corresponding part. Otherwise, we just store it
        # as an attribute below.  All other attributes we remove from
        # the parts, so that we do not store them multiple times.
        # (Note that attributes are not linked to the parent, so it
        # is safe to reset them.)
        # TODO: deal with (some of) this in Column.__getitem__?
        # Alternatively: should we store info on the first part?
        # TODO: special-case format somehow? Can we have good formats
        # for structured columns?
        unit = self.unit
        if isinstance(unit, StructuredUnit) and len(unit) == len(names):
            units = unit.values()
            unit = None  # No need to store as an attribute as well.
        else:
            units = [None] * len(names)
        for name, part_unit in zip(names, units):
            part = self._parent[name]
            part.unit = part_unit
            part.description = None
            part.meta = {}
            part.format = None
            data[name] = part

        # Create the attributes required to reconstruct the column.
        result['data'] = data
        # Store the shape if needed. Just like scalar data, a structured data
        # column (e.g. with dtype `f8,i8`) can be multidimensional within each
        # row and have a shape, and that needs to be distinguished from the
        # case that each entry in the structure has the same shape (e.g.,
        # distinguist a column with dtype='f8,i8' and 2 elements per row from
        # one with dtype '2f8,2i8' and just one element per row).
        if shape := self._parent.shape[1:]:
            result['shape'] = list(shape)
        # Also store the standard info attributes since these are
        # stored on the parent and can thus just be passed on as
        # arguments.  TODO: factor out with essentially the same
        # code in serialize._represent_mixin_as_column.
        if unit is not None and unit != '':
            result['unit'] = unit
        if self.format is not None:
            result['format'] = self.format
        if self.description is not None:
            result['description'] = self.description
        if self.meta:
            result['meta'] = self.meta

        return result
```
### 69 - astropy/table/column.py:

Start line: 662, End line: 678

```python
class BaseColumn(_ColumnGetitemShim, np.ndarray):

    def __array_finalize__(self, obj):
        # Obj will be none for direct call to Column() creator
        if obj is None:
            return

        if callable(super().__array_finalize__):
            super().__array_finalize__(obj)

        # Self was created from template (e.g. obj[slice] or (obj * 2))
        # or viewcast e.g. obj.view(Column).  In either case we want to
        # init Column attributes for self from obj if possible.
        self.parent_table = None
        if not hasattr(self, 'indices'):  # may have been copied in __new__
            self.indices = []
        self._copy_attrs(obj)
        if 'info' in getattr(obj, '__dict__', {}):
            self.info = obj.info
```
