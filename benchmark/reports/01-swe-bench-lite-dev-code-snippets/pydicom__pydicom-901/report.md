# pydicom__pydicom-901

| **pydicom/pydicom** | `3746878d8edf1cbda6fbcf35eec69f9ba79301ca` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | 296 |
| **Avg pos** | 3.0 |
| **Min pos** | 1 |
| **Max pos** | 2 |
| **Top file pos** | 1 |
| **Missing snippets** | 3 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/pydicom/config.py b/pydicom/config.py
--- a/pydicom/config.py
+++ b/pydicom/config.py
@@ -62,10 +62,7 @@ def DS_decimal(use_Decimal_boolean=True):
 
 # Logging system and debug function to change logging level
 logger = logging.getLogger('pydicom')
-handler = logging.StreamHandler()
-formatter = logging.Formatter("%(message)s")
-handler.setFormatter(formatter)
-logger.addHandler(handler)
+logger.addHandler(logging.NullHandler())
 
 
 import pydicom.pixel_data_handlers.numpy_handler as np_handler  # noqa
@@ -110,16 +107,29 @@ def get_pixeldata(ds):
 """
 
 
-def debug(debug_on=True):
-    """Turn debugging of DICOM file reading and writing on or off.
+def debug(debug_on=True, default_handler=True):
+    """Turn on/off debugging of DICOM file reading and writing.
+
     When debugging is on, file location and details about the
     elements read at that location are logged to the 'pydicom'
     logger using python's logging module.
 
-    :param debug_on: True (default) to turn on debugging,
-    False to turn off.
+    Parameters
+    ----------
+    debug_on : bool, optional
+        If True (default) then turn on debugging, False to turn off.
+    default_handler : bool, optional
+        If True (default) then use ``logging.StreamHandler()`` as the handler
+        for log messages.
     """
     global logger, debugging
+
+    if default_handler:
+        handler = logging.StreamHandler()
+        formatter = logging.Formatter("%(message)s")
+        handler.setFormatter(formatter)
+        logger.addHandler(handler)
+
     if debug_on:
         logger.setLevel(logging.DEBUG)
         debugging = True
@@ -129,4 +139,4 @@ def debug(debug_on=True):
 
 
 # force level=WARNING, in case logging default is set differently (issue 103)
-debug(False)
+debug(False, False)

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| pydicom/config.py | 65 | 67 | 1 | 1 | 296
| pydicom/config.py | 113 | 119 | 2 | 1 | 656
| pydicom/config.py | 132 | 134 | - | 1 | -


## Problem Statement

```
pydicom should not define handler, formatter and log level.
The `config` module (imported when pydicom is imported) defines a handler and set the log level for the pydicom logger. This should not be the case IMO. It should be the responsibility of the client code of pydicom to configure the logging module to its convenience. Otherwise one end up having multiple logs record as soon as pydicom is imported:

Example:
\`\`\`
Could not import pillow
2018-03-25 15:27:29,744 :: DEBUG :: pydicom 
  Could not import pillow
Could not import jpeg_ls
2018-03-25 15:27:29,745 :: DEBUG :: pydicom 
  Could not import jpeg_ls
Could not import gdcm
2018-03-25 15:27:29,745 :: DEBUG :: pydicom 
  Could not import gdcm
\`\`\` 
Or am I missing something?

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- |
| **-> 1 <-** | **1 pydicom/config.py** | 43 | 82| 296 | 296 | 
| **-> 2 <-** | **1 pydicom/config.py** | 83 | 132| 360 | 656 | 
| 3 | 2 pydicom/pixel_data_handlers/pillow_handler.py | 3 | 79| 442 | 1098 | 
| 4 | 3 dicom.py | 0 | 11| 129 | 1227 | 
| 5 | 4 setup.py | 0 | 106| 784 | 2011 | 
| 6 | 5 pydicom/pixel_data_handlers/gdcm_handler.py | 3 | 76| 488 | 2499 | 
| 7 | 6 pydicom/compat.py | 6 | 34| 211 | 2710 | 
| 8 | 6 pydicom/pixel_data_handlers/pillow_handler.py | 171 | 219| 375 | 3085 | 
| 9 | 7 pydicom/dicomio.py | 0 | 5| 0 | 3085 | 
| 10 | 8 pydicom/pixel_data_handlers/jpeg_ls_handler.py | 5 | 59| 309 | 3394 | 
| 11 | 9 pydicom/filereader.py | 3 | 30| 265 | 3659 | 
| 12 | 10 pydicom/__init__.py | 31 | 55| 175 | 3834 | 
| 13 | 11 pydicom/data/__init__.py | 0 | 9| 0 | 3834 | 
| 14 | 12 doc/conf.py | 0 | 104| 805 | 4639 | 
| 15 | 13 pydicom/uid.py | 192 | 242| 550 | 5189 | 
| 16 | 14 pydicom/util/codify.py | 284 | 368| 667 | 5856 | 
| 17 | 15 pydicom/data/charset_files/charlist.py | 3 | 49| 311 | 6167 | 
| 18 | 16 pydicom/dataset.py | 17 | 61| 352 | 6519 | 
| 19 | 17 pydicom/filebase.py | 117 | 139| 200 | 6719 | 
| 20 | 18 pydicom/filewriter.py | 3 | 21| 175 | 6894 | 
| 21 | 19 pydicom/pixel_data_handlers/rle_handler.py | 35 | 84| 263 | 7157 | 
| 22 | 19 pydicom/util/codify.py | 66 | 78| 124 | 7281 | 
| 23 | 19 doc/conf.py | 243 | 305| 467 | 7748 | 
| 24 | 19 doc/conf.py | 105 | 240| 1068 | 8816 | 
| 25 | 19 pydicom/pixel_data_handlers/pillow_handler.py | 82 | 169| 783 | 9599 | 
| 26 | 20 pydicom/datadict.py | 4 | 32| 298 | 9897 | 
| 27 | 21 source/generate_dict/generate_dicom_dict.py | 213 | 307| 828 | 10725 | 
| 28 | 21 pydicom/filereader.py | 827 | 854| 264 | 10989 | 
| 29 | 22 doc/sphinxext/sphinx_issues.py | 122 | 134| 161 | 11150 | 
| 30 | 23 pydicom/data/test_files/test.py | 5 | 23| 184 | 11334 | 
| 31 | **23 pydicom/config.py** | 5 | 27| 135 | 11469 | 
| 32 | 23 pydicom/filebase.py | 87 | 115| 251 | 11720 | 
| 33 | 24 examples/input_output/plot_write_dicom.py | 0 | 76| 556 | 12276 | 
| 34 | 25 pydicom/benchmarks/bench_handler_numpy.py | 6 | 46| 655 | 12931 | 
| 35 | 26 pydicom/_version.py | 0 | 6| 0 | 12931 | 
| 36 | 26 pydicom/filebase.py | 142 | 182| 354 | 13285 | 
| 37 | 27 examples/dicomtree.py | 73 | 85| 110 | 13395 | 
| 38 | 28 pydicom/util/leanread.py | 36 | 67| 255 | 13650 | 
| 39 | 29 pydicom/errors.py | 4 | 19| 144 | 13794 | 
| 40 | 30 pydicom/pixel_data_handlers/numpy_handler.py | 38 | 93| 307 | 14101 | 
| 41 | 30 pydicom/filereader.py | 671 | 744| 893 | 14994 | 
| 42 | 31 pydicom/charset.py | 2 | 62| 783 | 15777 | 
| 43 | 31 pydicom/filewriter.py | 905 | 950| 510 | 16287 | 
| 44 | 31 pydicom/pixel_data_handlers/jpeg_ls_handler.py | 62 | 146| 686 | 16973 | 
| 45 | 32 pydicom/util/fixes.py | 3 | 5| 12 | 16985 | 
| 46 | **32 pydicom/config.py** | 30 | 40| 136 | 17121 | 
| 47 | 32 pydicom/util/codify.py | 16 | 63| 372 | 17493 | 
| 48 | 33 pydicom/valuerep.py | 2 | 39| 406 | 17899 | 
| 49 | 34 source/generate_dict/generate_private_dict.py | 94 | 112| 152 | 18051 | 
| 50 | 35 examples/input_output/plot_read_dicom.py | 0 | 49| 320 | 18371 | 
| 51 | 36 pydicom/benchmarks/bench_handler_rle_decode.py | 3 | 35| 559 | 18930 | 
| 52 | 36 source/generate_dict/generate_dicom_dict.py | 0 | 61| 529 | 19459 | 
| 53 | 36 examples/dicomtree.py | 26 | 23| 329 | 19788 | 
| 54 | 37 examples/input_output/plot_read_dicom_directory.py | 0 | 78| 601 | 20389 | 
| 55 | 38 pydicom/dicomdir.py | 58 | 101| 343 | 20732 | 
| 56 | 38 pydicom/util/codify.py | 201 | 233| 297 | 21029 | 
| 57 | 39 pydicom/_storage_sopclass_uids.py | 129 | 172| 736 | 21765 | 
| 58 | 40 pydicom/values.py | 379 | 428| 505 | 22270 | 
| 59 | 40 pydicom/_storage_sopclass_uids.py | 85 | 128| 725 | 22995 | 
| 60 | 40 pydicom/pixel_data_handlers/gdcm_handler.py | 174 | 253| 675 | 23670 | 
| 61 | 40 pydicom/util/leanread.py | 14 | 34| 143 | 23813 | 
| 62 | 40 pydicom/_storage_sopclass_uids.py | 253 | 299| 775 | 24588 | 
| 63 | 40 pydicom/filewriter.py | 868 | 902| 365 | 24953 | 
| 64 | 40 pydicom/pixel_data_handlers/gdcm_handler.py | 114 | 171| 433 | 25386 | 
| 65 | 41 pydicom/fileutil.py | 243 | 256| 136 | 25522 | 
| 66 | 41 pydicom/charset.py | 63 | 95| 424 | 25946 | 
| 67 | 41 pydicom/_storage_sopclass_uids.py | 0 | 44| 719 | 26665 | 
| 68 | 41 pydicom/filebase.py | 11 | 8| 396 | 27061 | 
| 69 | 42 pydicom/dataelem.py | 9 | 57| 366 | 27427 | 
| 70 | 42 pydicom/filewriter.py | 812 | 866| 573 | 28000 | 
| 71 | 42 examples/dicomtree.py | 52 | 49| 279 | 28279 | 
| 72 | 43 examples/metadata_processing/plot_add_dict_entries.py | 0 | 51| 333 | 28612 | 
| 73 | 43 pydicom/filewriter.py | 357 | 354| 178 | 28790 | 
| 74 | 43 pydicom/pixel_data_handlers/rle_handler.py | 304 | 323| 260 | 29050 | 
| 75 | 44 examples/metadata_processing/plot_anonymize.py | 0 | 93| 487 | 29537 | 
| 76 | 44 pydicom/dicomdir.py | 7 | 4| 423 | 29960 | 
| 77 | 44 pydicom/_storage_sopclass_uids.py | 211 | 252| 718 | 30678 | 
| 78 | 44 pydicom/filewriter.py | 385 | 382| 199 | 30877 | 
| 79 | 45 doc/_templates/numpydoc_docstring.py | 0 | 16| 0 | 30877 | 
| 80 | 45 pydicom/uid.py | 3 | 20| 199 | 31076 | 
| 81 | 46 examples/input_output/plot_printing_dataset.py | 20 | 17| 314 | 31390 | 
| 82 | 46 pydicom/benchmarks/bench_handler_numpy.py | 131 | 208| 827 | 32217 | 
| 83 | 47 examples/image_processing/plot_downsize_image.py | 0 | 43| 307 | 32524 | 
| 84 | 47 pydicom/filewriter.py | 415 | 412| 198 | 32722 | 
| 85 | 47 pydicom/filewriter.py | 545 | 567| 231 | 32953 | 
| 86 | 47 pydicom/_storage_sopclass_uids.py | 173 | 210| 738 | 33691 | 
| 87 | 47 pydicom/charset.py | 569 | 632| 644 | 34335 | 
| 88 | 47 pydicom/util/leanread.py | 3 | 14| 183 | 34518 | 
| 89 | 48 pydicom/util/dump.py | 69 | 105| 308 | 34826 | 
| 90 | 48 pydicom/filereader.py | 747 | 825| 732 | 35558 | 
| 91 | 49 pydicom/pixel_data_handlers/util.py | 254 | 322| 648 | 36206 | 
| 92 | 49 pydicom/dataset.py | 1698 | 1726| 215 | 36421 | 
| 93 | 49 pydicom/util/codify.py | 236 | 281| 397 | 36818 | 
| 94 | 50 source/generate_dict/generate_uid_dict.py | 156 | 179| 182 | 37000 | 
| 95 | 50 pydicom/dataset.py | 1201 | 1305| 813 | 37813 | 
| 96 | 50 pydicom/valuerep.py | 603 | 715| 787 | 38600 | 
| 97 | 50 pydicom/dataset.py | 200 | 342| 1192 | 39792 | 
| 98 | 50 pydicom/dataset.py | 1307 | 1348| 360 | 40152 | 
| 99 | 51 examples/image_processing/reslice.py | 0 | 84| 583 | 40735 | 
| 100 | 52 examples/plot_dicom_difference.py | 0 | 39| 223 | 40958 | 
| 101 | 53 pydicom/_uid_dict.py | 0 | 404| 20 | 40978 | 
| 102 | 54 pydicom/_private_dict.py | 0 | 9790| 70 | 41048 | 
| 103 | 54 pydicom/_storage_sopclass_uids.py | 45 | 84| 718 | 41766 | 
| 104 | 54 pydicom/util/fixes.py | 7 | 135| 989 | 42755 | 
| 105 | 54 pydicom/filebase.py | 60 | 85| 265 | 43020 | 
| 106 | 54 pydicom/filereader.py | 857 | 900| 336 | 43356 | 
| 107 | 55 pydicom/misc.py | 24 | 46| 151 | 43507 | 
| 108 | 55 pydicom/dataset.py | 1953 | 2007| 476 | 43983 | 
| 109 | 55 pydicom/fileutil.py | 12 | 9| 290 | 44273 | 
| 110 | 55 pydicom/benchmarks/bench_handler_numpy.py | 110 | 107| 461 | 44734 | 
| 111 | 55 pydicom/dataelem.py | 60 | 128| 578 | 45312 | 
| 112 | 55 pydicom/dataset.py | 2010 | 2090| 725 | 46037 | 
| 113 | 55 pydicom/pixel_data_handlers/gdcm_handler.py | 79 | 111| 230 | 46267 | 
| 114 | 56 pydicom/util/fixer.py | 80 | 77| 248 | 46515 | 
| 115 | 56 pydicom/util/dump.py | 3 | 26| 157 | 46672 | 
| 116 | 57 pydicom/benchmarks/bench_handler_rle_encode.py | 3 | 38| 397 | 47069 | 
| 117 | 57 pydicom/util/leanread.py | 70 | 90| 200 | 47269 | 
| 118 | 57 pydicom/dataset.py | 2116 | 2171| 544 | 47813 | 
| 119 | 57 pydicom/dataset.py | 1558 | 1600| 407 | 48220 | 
| 120 | 57 pydicom/filereader.py | 616 | 613| 464 | 48684 | 
| 121 | 57 source/generate_dict/generate_private_dict.py | 0 | 23| 214 | 48898 | 
| 122 | 57 pydicom/pixel_data_handlers/util.py | 92 | 176| 734 | 49632 | 
| 123 | 58 pydicom/tag.py | 16 | 13| 143 | 49775 | 
| 124 | 58 pydicom/benchmarks/bench_handler_rle_decode.py | 68 | 69| 166 | 49941 | 


### Hint

```
In addition, I don't understand what the purpose of the `config.debug` function since the default behavor of the logging module in absence of configuartion seems to already be the one you want.

From https://docs.python.org/3/howto/logging.html#configuring-logging-for-a-library:

> If the using application does not use logging, and library code makes logging calls, then (as described in the previous section) events of severity WARNING and greater will be printed to sys.stderr. This is regarded as the best default behaviour.

and

>**It is strongly advised that you do not add any handlers other than NullHandler to your library’s loggers.** This is because the configuration of handlers is the prerogative of the application developer who uses your library. The application developer knows their target audience and what handlers are most appropriate for their application: if you add handlers ‘under the hood’, you might well interfere with their ability to carry out unit tests and deliver logs which suit their requirements. 

I think you make good points here.  I support changing the logging to comply with python's suggested behavior.

> In addition, I don't understand what the purpose of the config.debug function

One reason is that the core loop in pydicom (data_element_generator in filereader.py) is extremely optimized for speed - it checks the `debugging` flag set by config.debug, to avoid composing messages and doing function calls to logger when not needed.
```

## Patch

```diff
diff --git a/pydicom/config.py b/pydicom/config.py
--- a/pydicom/config.py
+++ b/pydicom/config.py
@@ -62,10 +62,7 @@ def DS_decimal(use_Decimal_boolean=True):
 
 # Logging system and debug function to change logging level
 logger = logging.getLogger('pydicom')
-handler = logging.StreamHandler()
-formatter = logging.Formatter("%(message)s")
-handler.setFormatter(formatter)
-logger.addHandler(handler)
+logger.addHandler(logging.NullHandler())
 
 
 import pydicom.pixel_data_handlers.numpy_handler as np_handler  # noqa
@@ -110,16 +107,29 @@ def get_pixeldata(ds):
 """
 
 
-def debug(debug_on=True):
-    """Turn debugging of DICOM file reading and writing on or off.
+def debug(debug_on=True, default_handler=True):
+    """Turn on/off debugging of DICOM file reading and writing.
+
     When debugging is on, file location and details about the
     elements read at that location are logged to the 'pydicom'
     logger using python's logging module.
 
-    :param debug_on: True (default) to turn on debugging,
-    False to turn off.
+    Parameters
+    ----------
+    debug_on : bool, optional
+        If True (default) then turn on debugging, False to turn off.
+    default_handler : bool, optional
+        If True (default) then use ``logging.StreamHandler()`` as the handler
+        for log messages.
     """
     global logger, debugging
+
+    if default_handler:
+        handler = logging.StreamHandler()
+        formatter = logging.Formatter("%(message)s")
+        handler.setFormatter(formatter)
+        logger.addHandler(handler)
+
     if debug_on:
         logger.setLevel(logging.DEBUG)
         debugging = True
@@ -129,4 +139,4 @@ def debug(debug_on=True):
 
 
 # force level=WARNING, in case logging default is set differently (issue 103)
-debug(False)
+debug(False, False)

```

## Test Patch

```diff
diff --git a/pydicom/tests/test_config.py b/pydicom/tests/test_config.py
new file mode 100644
--- /dev/null
+++ b/pydicom/tests/test_config.py
@@ -0,0 +1,107 @@
+# Copyright 2008-2019 pydicom authors. See LICENSE file for details.
+"""Unit tests for the pydicom.config module."""
+
+import logging
+import sys
+
+import pytest
+
+from pydicom import dcmread
+from pydicom.config import debug
+from pydicom.data import get_testdata_files
+
+
+DS_PATH = get_testdata_files("CT_small.dcm")[0]
+PYTEST = [int(x) for x in pytest.__version__.split('.')]
+
+
+@pytest.mark.skipif(PYTEST[:2] < [3, 4], reason='no caplog')
+class TestDebug(object):
+    """Tests for config.debug()."""
+    def setup(self):
+        self.logger = logging.getLogger('pydicom')
+
+    def teardown(self):
+        # Reset to just NullHandler
+        self.logger.handlers = [self.logger.handlers[0]]
+
+    def test_default(self, caplog):
+        """Test that the default logging handler is a NullHandler."""
+        assert 1 == len(self.logger.handlers)
+        assert isinstance(self.logger.handlers[0], logging.NullHandler)
+
+        with caplog.at_level(logging.DEBUG, logger='pydicom'):
+            ds = dcmread(DS_PATH)
+
+            assert "Call to dcmread()" not in caplog.text
+            assert "Reading File Meta Information preamble..." in caplog.text
+            assert "Reading File Meta Information prefix..." in caplog.text
+            assert "00000080: 'DICM' prefix found" in caplog.text
+
+    def test_debug_on_handler_null(self, caplog):
+        """Test debug(True, False)."""
+        debug(True, False)
+        assert 1 == len(self.logger.handlers)
+        assert isinstance(self.logger.handlers[0], logging.NullHandler)
+
+        with caplog.at_level(logging.DEBUG, logger='pydicom'):
+            ds = dcmread(DS_PATH)
+
+            assert "Call to dcmread()" in caplog.text
+            assert "Reading File Meta Information preamble..." in caplog.text
+            assert "Reading File Meta Information prefix..." in caplog.text
+            assert "00000080: 'DICM' prefix found" in caplog.text
+            msg = (
+                "00009848: fc ff fc ff 4f 42 00 00 7e 00 00 00    "
+                "(fffc, fffc) OB Length: 126"
+            )
+            assert msg in caplog.text
+
+    def test_debug_off_handler_null(self, caplog):
+        """Test debug(False, False)."""
+        debug(False, False)
+        assert 1 == len(self.logger.handlers)
+        assert isinstance(self.logger.handlers[0], logging.NullHandler)
+
+        with caplog.at_level(logging.DEBUG, logger='pydicom'):
+            ds = dcmread(DS_PATH)
+
+            assert "Call to dcmread()" not in caplog.text
+            assert "Reading File Meta Information preamble..." in caplog.text
+            assert "Reading File Meta Information prefix..." in caplog.text
+            assert "00000080: 'DICM' prefix found" in caplog.text
+
+    def test_debug_on_handler_stream(self, caplog):
+        """Test debug(True, True)."""
+        debug(True, True)
+        assert 2 == len(self.logger.handlers)
+        assert isinstance(self.logger.handlers[0], logging.NullHandler)
+        assert isinstance(self.logger.handlers[1], logging.StreamHandler)
+
+        with caplog.at_level(logging.DEBUG, logger='pydicom'):
+            ds = dcmread(DS_PATH)
+
+            assert "Call to dcmread()" in caplog.text
+            assert "Reading File Meta Information preamble..." in caplog.text
+            assert "Reading File Meta Information prefix..." in caplog.text
+            assert "00000080: 'DICM' prefix found" in caplog.text
+            msg = (
+                "00009848: fc ff fc ff 4f 42 00 00 7e 00 00 00    "
+                "(fffc, fffc) OB Length: 126"
+            )
+            assert msg in caplog.text
+
+    def test_debug_off_handler_stream(self, caplog):
+        """Test debug(False, True)."""
+        debug(False, True)
+        assert 2 == len(self.logger.handlers)
+        assert isinstance(self.logger.handlers[0], logging.NullHandler)
+        assert isinstance(self.logger.handlers[1], logging.StreamHandler)
+
+        with caplog.at_level(logging.DEBUG, logger='pydicom'):
+            ds = dcmread(DS_PATH)
+
+            assert "Call to dcmread()" not in caplog.text
+            assert "Reading File Meta Information preamble..." in caplog.text
+            assert "Reading File Meta Information prefix..." in caplog.text
+            assert "00000080: 'DICM' prefix found" in caplog.text

```


## Code snippets

### 1 - pydicom/config.py:

Start line: 43, End line: 82

```python
# Configuration flags
allow_DS_float = False
"""Set allow_float to True to allow DSdecimal instances
to be created with floats; otherwise, they must be explicitly
converted to strings, with the user explicity setting the
precision of digits and rounding. Default: False"""

enforce_valid_values = False
"""Raise errors if any value is not allowed by DICOM standard,
e.g. DS strings that are longer than 16 characters;
IS strings outside the allowed range.
"""

datetime_conversion = False
"""Set datetime_conversion to convert DA, DT and TM
data elements to datetime.date, datetime.datetime
and datetime.time respectively. Default: False
"""

# Logging system and debug function to change logging level
logger = logging.getLogger('pydicom')
handler = logging.StreamHandler()
formatter = logging.Formatter("%(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


import pydicom.pixel_data_handlers.numpy_handler as np_handler  # noqa
import pydicom.pixel_data_handlers.rle_handler as rle_handler  # noqa
import pydicom.pixel_data_handlers.pillow_handler as pillow_handler  # noqa
import pydicom.pixel_data_handlers.jpeg_ls_handler as jpegls_handler  # noqa
import pydicom.pixel_data_handlers.gdcm_handler as gdcm_handler  # noqa

pixel_data_handlers = [
    np_handler,
    rle_handler,
    gdcm_handler,
    pillow_handler,
    jpegls_handler,
]
```
### 2 - pydicom/config.py:

Start line: 83, End line: 132

```python
"""Handlers for converting (7fe0,0010) Pixel Data.
This is an ordered list that the dataset.convert_pixel_data()
method will try to extract a correctly sized numpy array from the
PixelData element.

Handers shall have two methods:

def supports_transfer_syntax(ds)
  This returns True if the handler might support the transfer syntax
  indicated in the dicom_dataset

def get_pixeldata(ds):
  This shall either throw an exception or return a correctly sized numpy
  array derived from the PixelData.  Reshaping the array to the correct
  dimensions is handled outside the image handler

The first handler that both announces that it supports the transfer syntax
and does not throw an exception, either in getting the data or when the data
is reshaped to the correct dimensions, is the handler that will provide the
data.

If they all fail, the last one to throw an exception gets to see its
exception thrown up.

If no one throws an exception, but they all refuse to support the transfer
syntax, then this fact is announced in a NotImplementedError exception.
"""


def debug(debug_on=True):
    """Turn debugging of DICOM file reading and writing on or off.
    When debugging is on, file location and details about the
    elements read at that location are logged to the 'pydicom'
    logger using python's logging module.

    :param debug_on: True (default) to turn on debugging,
    False to turn off.
    """
    global logger, debugging
    if debug_on:
        logger.setLevel(logging.DEBUG)
        debugging = True
    else:
        logger.setLevel(logging.WARNING)
        debugging = False


# force level=WARNING, in case logging default is set differently (issue 103)
debug(False)
```
### 3 - pydicom/pixel_data_handlers/pillow_handler.py:

Start line: 3, End line: 79

```python
import io
import logging

try:
    import numpy
    HAVE_NP = True
except ImportError:
    HAVE_NP = False

try:
    from PIL import Image
    HAVE_PIL = True
except ImportError:
    HAVE_PIL = False

try:
    from PIL import _imaging
    HAVE_JPEG = getattr(_imaging, "jpeg_decoder", False)
    HAVE_JPEG2K = getattr(_imaging, "jpeg2k_decoder", False)
except ImportError:
    HAVE_JPEG = False
    HAVE_JPEG2K = False

import pydicom.encaps
from pydicom.pixel_data_handlers.util import dtype_corrected_for_endianness
import pydicom.uid


logger = logging.getLogger('pydicom')

PillowSupportedTransferSyntaxes = [
    pydicom.uid.JPEGBaseline,
    pydicom.uid.JPEGLossless,
    pydicom.uid.JPEGExtended,
    pydicom.uid.JPEG2000Lossless,
]
PillowJPEG2000TransferSyntaxes = [
    pydicom.uid.JPEG2000Lossless,
]
PillowJPEGTransferSyntaxes = [
    pydicom.uid.JPEGBaseline,
    pydicom.uid.JPEGExtended,
]

HANDLER_NAME = 'Pillow'

DEPENDENCIES = {
    'numpy': ('http://www.numpy.org/', 'NumPy'),
    'PIL': ('https://python-pillow.org/', 'Pillow'),
}


def is_available():
    """Return True if the handler has its dependencies met."""
    return HAVE_NP and HAVE_PIL


def supports_transfer_syntax(transfer_syntax):
    """
    Returns
    -------
    bool
        True if this pixel data handler might support this transfer syntax.

        False to prevent any attempt to try to use this handler
        to decode the given transfer syntax
    """
    return transfer_syntax in PillowSupportedTransferSyntaxes


def needs_to_convert_to_RGB(dicom_dataset):
    return False


def should_change_PhotometricInterpretation_to_RGB(dicom_dataset):
    should_change = dicom_dataset.SamplesPerPixel == 3
    return False
```
### 4 - dicom.py:

```python
msg = """
Pydicom via 'import dicom' has been removed in pydicom version 1.0.
Please install the `dicom` package to restore function of code relying
on pydicom 0.9.9 or earlier. E.g. `pip install dicom`.
Alternatively, most code can easily be converted to pydicom > 1.0 by
changing import lines from 'import dicom' to 'import pydicom'.
See the Transition Guide at
https://pydicom.github.io/pydicom/stable/transition_to_pydicom1.html.
"""

raise ImportError(msg)
```
### 5 - setup.py:

```python
#!/usr/bin/env python

import os
import os.path
import sys
from glob import glob
from setuptools import setup, find_packages

have_dicom = True
try:
    import dicom
except ImportError:
    have_dicom = False

# get __version__ from _version.py
base_dir = os.path.dirname(os.path.realpath(__file__))
ver_file = os.path.join(base_dir, 'pydicom', '_version.py')
with open(ver_file) as f:
    exec(f.read())

description = "Pure python package for DICOM medical file reading and writing"

needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
pytest_runner = ['pytest-runner'] if needs_pytest else []

# in_py2 check in next line - pytest>=5 requires Python 3
TESTS_REQUIRE = ['pytest<5'] if sys.version_info[0] == 2 else ['pytest']
_py_modules = []
if not have_dicom:
    _py_modules = ['dicom']

CLASSIFIERS = [
    "License :: OSI Approved :: MIT License",
    "Intended Audience :: Developers",
    "Intended Audience :: Healthcare Industry",
    "Intended Audience :: Science/Research",
    "Development Status :: 5 - Production/Stable",
    "Programming Language :: Python",
    "Programming Language :: Python :: 2.7",
    "Programming Language :: Python :: 3.4",
    "Programming Language :: Python :: 3.5",
    "Programming Language :: Python :: 3.6",
    "Operating System :: OS Independent",
    "Topic :: Scientific/Engineering :: Medical Science Apps.",
    "Topic :: Scientific/Engineering :: Physics",
    "Topic :: Software Development :: Libraries"]

KEYWORDS = "dicom python medical imaging"

NAME = "pydicom"
AUTHOR = "Darcy Mason and contributors"
AUTHOR_EMAIL = "darcymason@gmail.com"
MAINTAINER = "Darcy Mason and contributors"
MAINTAINER_EMAIL = "darcymason@gmail.com"
DESCRIPTION = description
URL = "https://github.com/pydicom/pydicom"
DOWNLOAD_URL = "https://github.com/pydicom/pydicom/archive/master.zip"
LICENSE = "MIT"
VERSION = __version__
REQUIRES = []
SETUP_REQUIRES = pytest_runner

# get long description from README.md
BASE_PATH = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(BASE_PATH, 'README.md')) as f:
    LONG_DESCRIPTION = f.read()


def data_files_inventory():
    data_files = []
    data_roots = ['pydicom/data']
    for data_root in data_roots:
        for root, subfolder, files in os.walk(data_root):
            files = [x.replace('pydicom/', '') for x in glob(root + '/*')
                     if not os.path.isdir(x)]
            data_files = data_files + files
    return data_files


PACKAGE_DATA = {'pydicom': data_files_inventory()}

opts = dict(name=NAME,
            version=VERSION,
            maintainer=MAINTAINER,
            maintainer_email=MAINTAINER_EMAIL,
            author=AUTHOR,
            author_email=AUTHOR_EMAIL,
            description=description,
            long_description=LONG_DESCRIPTION,
            long_description_content_type='text/markdown',
            url=URL,
            download_url=DOWNLOAD_URL,
            license=LICENSE,
            keywords=KEYWORDS,
            classifiers=CLASSIFIERS,
            packages=find_packages(),
            py_modules=_py_modules,
            package_data=PACKAGE_DATA,
            include_package_data=True,
            install_requires=REQUIRES,
            setup_requires=SETUP_REQUIRES,
            tests_require=TESTS_REQUIRE,
            zip_safe=False)

if __name__ == '__main__':
    setup(**opts)
```
### 6 - pydicom/pixel_data_handlers/gdcm_handler.py:

Start line: 3, End line: 76

```python
import sys

try:
    import numpy
    HAVE_NP = True
except ImportError:
    HAVE_NP = False

try:
    import gdcm
    HAVE_GDCM = True
    HAVE_GDCM_IN_MEMORY_SUPPORT = hasattr(gdcm.DataElement,
                                          'SetByteStringValue')
except ImportError:
    HAVE_GDCM = False
    HAVE_GDCM_IN_MEMORY_SUPPORT = False

import pydicom.uid
from pydicom import compat
from pydicom.pixel_data_handlers.util import get_expected_length, pixel_dtype


HANDLER_NAME = 'GDCM'

DEPENDENCIES = {
    'numpy': ('http://www.numpy.org/', 'NumPy'),
    'gdcm': ('http://gdcm.sourceforge.net/wiki/index.php/Main_Page', 'GDCM'),
}

SUPPORTED_TRANSFER_SYNTAXES = [
    pydicom.uid.JPEGBaseline,
    pydicom.uid.JPEGExtended,
    pydicom.uid.JPEGLosslessP14,
    pydicom.uid.JPEGLossless,
    pydicom.uid.JPEGLSLossless,
    pydicom.uid.JPEGLSLossy,
    pydicom.uid.JPEG2000Lossless,
    pydicom.uid.JPEG2000,
]

should_convert_these_syntaxes_to_RGB = [
    pydicom.uid.JPEGBaseline, ]


def is_available():
    """Return True if the handler has its dependencies met."""
    return HAVE_NP and HAVE_GDCM


def needs_to_convert_to_RGB(dicom_dataset):
    should_convert = (dicom_dataset.file_meta.TransferSyntaxUID in
                      should_convert_these_syntaxes_to_RGB)
    should_convert &= dicom_dataset.SamplesPerPixel == 3
    return False


def should_change_PhotometricInterpretation_to_RGB(dicom_dataset):
    should_change = (dicom_dataset.file_meta.TransferSyntaxUID in
                     should_convert_these_syntaxes_to_RGB)
    should_change &= dicom_dataset.SamplesPerPixel == 3
    return False


def supports_transfer_syntax(transfer_syntax):
    """
    Returns
    -------
    bool
        True if this pixel data handler might support this transfer syntax.

        False to prevent any attempt to try to use this handler
        to decode the given transfer syntax
    """
    return transfer_syntax in SUPPORTED_TRANSFER_SYNTAXES
```
### 7 - pydicom/compat.py:

Start line: 6, End line: 34

```python
import sys

in_py2 = sys.version_info[0] == 2
in_PyPy = 'PyPy' in sys.version

# Text types
# In py3+, the native text type ('str') is unicode
# In py2, str can be either bytes or text.
if in_py2:
    text_type = unicode
    string_types = (str, unicode)
    char_types = (str, unicode)
    number_types = (int, long)
    int_type = long
else:
    text_type = str
    string_types = (str, )
    char_types = (str, bytes)
    number_types = (int, )
    int_type = int

if in_py2:
    # Have to run through exec as the code is a syntax error in py 3
    exec('def reraise(tp, value, tb):\n raise tp, value, tb')
else:

    def reraise(tp, value, tb):
        raise value.with_traceback(tb)
```
### 8 - pydicom/pixel_data_handlers/pillow_handler.py:

Start line: 171, End line: 219

```python
def get_pixeldata(dicom_dataset):
    # ... other code

    try:
        UncompressedPixelData = bytearray()
        if ('NumberOfFrames' in dicom_dataset and
                dicom_dataset.NumberOfFrames > 1):
            # multiple compressed frames
            CompressedPixelDataSeq = \
                pydicom.encaps.decode_data_sequence(
                    dicom_dataset.PixelData)
            for frame in CompressedPixelDataSeq:
                data = generic_jpeg_file_header + \
                    frame[frame_start_from:]
                fio = io.BytesIO(data)
                try:
                    decompressed_image = Image.open(fio)
                except IOError as e:
                    raise NotImplementedError(e.strerror)
                UncompressedPixelData.extend(decompressed_image.tobytes())
        else:
            # single compressed frame
            pixel_data = pydicom.encaps.defragment_data(
                dicom_dataset.PixelData)
            pixel_data = generic_jpeg_file_header + \
                pixel_data[frame_start_from:]
            try:
                fio = io.BytesIO(pixel_data)
                decompressed_image = Image.open(fio)
            except IOError as e:
                raise NotImplementedError(e.strerror)
            UncompressedPixelData.extend(decompressed_image.tobytes())
    except Exception:
        raise

    logger.debug(
        "Successfully read %s pixel bytes", len(UncompressedPixelData)
    )

    pixel_array = numpy.frombuffer(UncompressedPixelData, numpy_format)

    if (transfer_syntax in
            PillowJPEG2000TransferSyntaxes and
            dicom_dataset.BitsStored == 16):
        # WHY IS THIS EVEN NECESSARY??
        pixel_array &= 0x7FFF

    if should_change_PhotometricInterpretation_to_RGB(dicom_dataset):
        dicom_dataset.PhotometricInterpretation = "RGB"

    return pixel_array
```
### 9 - pydicom/dicomio.py:

```python

```
### 10 - pydicom/pixel_data_handlers/jpeg_ls_handler.py:

Start line: 5, End line: 59

```python
try:
    import numpy
    HAVE_NP = True
except ImportError:
    HAVE_NP = False

try:
    import jpeg_ls
    HAVE_JPEGLS = True
except ImportError:
    HAVE_JPEGLS = False

import pydicom.encaps
from pydicom.pixel_data_handlers.util import dtype_corrected_for_endianness
import pydicom.uid


HANDLER_NAME = 'JPEG-LS'

DEPENDENCIES = {
    'numpy': ('http://www.numpy.org/', 'NumPy'),
    'jpeg_ls': ('https://github.com/Who8MyLunch/CharPyLS', 'CharPyLS'),
}

SUPPORTED_TRANSFER_SYNTAXES = [
    pydicom.uid.JPEGLSLossless,
    pydicom.uid.JPEGLSLossy,
]


def is_available():
    """Return True if the handler has its dependencies met."""
    return HAVE_NP and HAVE_JPEGLS


def needs_to_convert_to_RGB(dicom_dataset):
    return False


def should_change_PhotometricInterpretation_to_RGB(dicom_dataset):
    should_change = dicom_dataset.SamplesPerPixel == 3
    return False


def supports_transfer_syntax(transfer_syntax):
    """
    Returns
    -------
    bool
        True if this pixel data handler might support this transfer syntax.

        False to prevent any attempt to try to use this handler
        to decode the given transfer syntax
    """
    return transfer_syntax in SUPPORTED_TRANSFER_SYNTAXES
```
### 31 - pydicom/config.py:

Start line: 5, End line: 27

```python
import logging

# Set the type used to hold DS values
#    default False; was decimal-based in pydicom 0.9.7
use_DS_decimal = False


data_element_callback = None
"""Set data_element_callback to a function to be called from read_dataset
every time a RawDataElement has been returned, before it is added
to the dataset.
"""

data_element_callback_kwargs = {}
"""Set this to use as keyword arguments passed to the data_element_callback
function"""


def reset_data_element_callback():
    global data_element_callback
    global data_element_callback_kwargs
    data_element_callback = None
    data_element_callback_kwargs = {}
```
### 46 - pydicom/config.py:

Start line: 30, End line: 40

```python
def DS_decimal(use_Decimal_boolean=True):
    """Set DS class to be derived from Decimal (True) or from float (False)
    If this function is never called, the default in pydicom >= 0.9.8
    is for DS to be based on float.
    """
    use_DS_decimal = use_Decimal_boolean
    import pydicom.valuerep
    if use_DS_decimal:
        pydicom.valuerep.DSclass = pydicom.valuerep.DSdecimal
    else:
        pydicom.valuerep.DSclass = pydicom.valuerep.DSfloat
```
