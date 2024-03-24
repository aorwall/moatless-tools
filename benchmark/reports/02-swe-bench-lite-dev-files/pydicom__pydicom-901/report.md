# pydicom__pydicom-901

| **pydicom/pydicom** | `3746878d8edf1cbda6fbcf35eec69f9ba79301ca` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | 968 |
| **Avg pos** | 6.0 |
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
| pydicom/config.py | 65 | 67 | 2 | 1 | 1936
| pydicom/config.py | 113 | 119 | 2 | 1 | 1936
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
| **-> 1 <-** | **1 pydicom/config.py** | 0 | 133| 968 | 968 | 
| **-> 2 <-** | **1 pydicom/config.py** | 0 | 133| 968 | 1936 | 
| 3 | 2 pydicom/pixel_data_handlers/pillow_handler.py | 0 | 220| 1615 | 3551 | 
| 4 | 3 dicom.py | 0 | 12| 129 | 3680 | 
| 5 | 4 setup.py | 0 | 107| 784 | 4464 | 
| 6 | 5 pydicom/pixel_data_handlers/gdcm_handler.py | 0 | 254| 1862 | 6326 | 
| 7 | 6 pydicom/compat.py | 0 | 35| 283 | 6609 | 
| 8 | 6 pydicom/pixel_data_handlers/pillow_handler.py | 0 | 220| 1615 | 8224 | 
| 9 | 7 pydicom/dicomio.py | 0 | 6| 68 | 8292 | 
| 10 | 8 pydicom/pixel_data_handlers/jpeg_ls_handler.py | 0 | 147| 1036 | 9328 | 
| 11 | 9 pydicom/filereader.py | 0 | 947| 8724 | 18052 | 
| 12 | 10 pydicom/__init__.py | 0 | 56| 388 | 18440 | 
| 13 | 11 pydicom/data/__init__.py | 0 | 10| 69 | 18509 | 
| 14 | 12 doc/conf.py | 0 | 306| 2409 | 20918 | 
| 15 | 13 pydicom/uid.py | 0 | 308| 2568 | 23486 | 
| 16 | 14 pydicom/util/codify.py | 0 | 369| 3015 | 26501 | 
| 17 | 15 pydicom/data/charset_files/charlist.py | 0 | 50| 343 | 26844 | 
| 18 | 16 pydicom/dataset.py | 0 | 2172| 17317 | 44161 | 
| 19 | 17 pydicom/filebase.py | 0 | 183| 1488 | 45649 | 


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

```python
# Copyright 2008-2018 pydicom authors. See LICENSE file for details.
"""Pydicom configuration options."""

# doc strings following items are picked up by sphinx for documentation

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
### 2 - pydicom/config.py:

```python
# Copyright 2008-2018 pydicom authors. See LICENSE file for details.
"""Pydicom configuration options."""

# doc strings following items are picked up by sphinx for documentation

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

```python
# Copyright 2008-2018 pydicom authors. See LICENSE file for details.
"""Use the pillow python package to decode pixel transfer syntaxes."""

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


def get_pixeldata(dicom_dataset):
    """Use Pillow to decompress compressed Pixel Data.

    Returns
    -------
    numpy.ndarray
       The contents of the Pixel Data element (7FE0,0010) as an ndarray.

    Raises
    ------
    ImportError
        If PIL is not available.

    NotImplementedError
        if the transfer syntax is not supported

    TypeError
        if the pixel data type is unsupported
    """
    logger.debug("Trying to use Pillow to read pixel array "
                 "(has pillow = %s)", HAVE_PIL)
    transfer_syntax = dicom_dataset.file_meta.TransferSyntaxUID
    if not HAVE_PIL:
        msg = ("The pillow package is required to use pixel_array for "
               "this transfer syntax {0}, and pillow could not be "
               "imported.".format(transfer_syntax.name))
        raise ImportError(msg)

    if not HAVE_JPEG and transfer_syntax in PillowJPEGTransferSyntaxes:
        msg = ("this transfer syntax {0}, can not be read because "
               "Pillow lacks the jpeg decoder plugin"
               .format(transfer_syntax.name))
        raise NotImplementedError(msg)

    if not HAVE_JPEG2K and transfer_syntax in PillowJPEG2000TransferSyntaxes:
        msg = ("this transfer syntax {0}, can not be read because "
               "Pillow lacks the jpeg 2000 decoder plugin"
               .format(transfer_syntax.name))
        raise NotImplementedError(msg)

    if transfer_syntax not in PillowSupportedTransferSyntaxes:
        msg = ("this transfer syntax {0}, can not be read because "
               "Pillow does not support this syntax"
               .format(transfer_syntax.name))
        raise NotImplementedError(msg)

    # Make NumPy format code, e.g. "uint16", "int32" etc
    # from two pieces of info:
    # dicom_dataset.PixelRepresentation -- 0 for unsigned, 1 for signed;
    # dicom_dataset.BitsAllocated -- 8, 16, or 32
    if dicom_dataset.PixelRepresentation == 0:
        format_str = 'uint{}'.format(dicom_dataset.BitsAllocated)
    elif dicom_dataset.PixelRepresentation == 1:
        format_str = 'int{}'.format(dicom_dataset.BitsAllocated)
    else:
        format_str = 'bad_pixel_representation'
    try:
        numpy_format = numpy.dtype(format_str)
    except TypeError:
        msg = ("Data type not understood by NumPy: "
               "format='{}', PixelRepresentation={}, "
               "BitsAllocated={}".format(
                   format_str,
                   dicom_dataset.PixelRepresentation,
                   dicom_dataset.BitsAllocated))
        raise TypeError(msg)

    numpy_format = dtype_corrected_for_endianness(
        dicom_dataset.is_little_endian, numpy_format)

    # decompress here
    if transfer_syntax in PillowJPEGTransferSyntaxes:
        logger.debug("This is a JPEG lossy format")
        if dicom_dataset.BitsAllocated > 8:
            raise NotImplementedError("JPEG Lossy only supported if "
                                      "Bits Allocated = 8")
        generic_jpeg_file_header = b''
        frame_start_from = 0
    elif transfer_syntax in PillowJPEG2000TransferSyntaxes:
        logger.debug("This is a JPEG 2000 format")
        generic_jpeg_file_header = b''
        # generic_jpeg_file_header = b'\x00\x00\x00\x0C\x6A'
        #     b'\x50\x20\x20\x0D\x0A\x87\x0A'
        frame_start_from = 0
    else:
        logger.debug("This is a another pillow supported format")
        generic_jpeg_file_header = b''
        frame_start_from = 0

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

```python
# Copyright 2008-2018 pydicom authors. See LICENSE file for details.
"""Use the gdcm python package to decode pixel transfer syntaxes."""

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


def create_data_element(dicom_dataset):
    """Create a gdcm.DataElement containing PixelData from a FileDataset

    Parameters
    ----------
    dicom_dataset : FileDataset


    Returns
    -------
    gdcm.DataElement
        Converted PixelData element
    """
    data_element = gdcm.DataElement(gdcm.Tag(0x7fe0, 0x0010))
    if dicom_dataset.file_meta.TransferSyntaxUID.is_compressed:
        if getattr(dicom_dataset, 'NumberOfFrames', 1) > 1:
            pixel_data_sequence = pydicom.encaps.decode_data_sequence(
                dicom_dataset.PixelData)
        else:
            pixel_data_sequence = [
                pydicom.encaps.defragment_data(dicom_dataset.PixelData)
            ]

        fragments = gdcm.SequenceOfFragments.New()
        for pixel_data in pixel_data_sequence:
            fragment = gdcm.Fragment()
            fragment.SetByteStringValue(pixel_data)
            fragments.AddFragment(fragment)
        data_element.SetValue(fragments.__ref__())
    else:
        data_element.SetByteStringValue(dicom_dataset.PixelData)

    return data_element


def create_image(dicom_dataset, data_element):
    """Create a gdcm.Image from a FileDataset and a gdcm.DataElement containing
    PixelData (0x7fe0, 0x0010)

    Parameters
    ----------
    dicom_dataset : FileDataset
    data_element : gdcm.DataElement
        DataElement containing PixelData

    Returns
    -------
    gdcm.Image
    """
    image = gdcm.Image()
    number_of_frames = getattr(dicom_dataset, 'NumberOfFrames', 1)
    image.SetNumberOfDimensions(2 if number_of_frames == 1 else 3)
    image.SetDimensions(
        (dicom_dataset.Columns, dicom_dataset.Rows, number_of_frames))
    image.SetDataElement(data_element)
    pi_type = gdcm.PhotometricInterpretation.GetPIType(
        dicom_dataset.PhotometricInterpretation)
    image.SetPhotometricInterpretation(
        gdcm.PhotometricInterpretation(pi_type))
    ts_type = gdcm.TransferSyntax.GetTSType(
        str.__str__(dicom_dataset.file_meta.TransferSyntaxUID))
    image.SetTransferSyntax(gdcm.TransferSyntax(ts_type))
    pixel_format = gdcm.PixelFormat(
        dicom_dataset.SamplesPerPixel, dicom_dataset.BitsAllocated,
        dicom_dataset.BitsStored, dicom_dataset.HighBit,
        dicom_dataset.PixelRepresentation)
    image.SetPixelFormat(pixel_format)
    if 'PlanarConfiguration' in dicom_dataset:
        image.SetPlanarConfiguration(dicom_dataset.PlanarConfiguration)
    return image


def create_image_reader(filename):
    """Create a gdcm.ImageReader

    Parameters
    ----------
    filename: str or unicode (Python 2)

    Returns
    -------
    gdcm.ImageReader
    """
    image_reader = gdcm.ImageReader()
    if compat.in_py2:
        if isinstance(filename, unicode):
            image_reader.SetFileName(
                filename.encode(sys.getfilesystemencoding()))
        else:
            image_reader.SetFileName(filename)
    else:
        image_reader.SetFileName(filename)
    return image_reader


def get_pixeldata(dicom_dataset):
    """
    Use the GDCM package to decode the PixelData attribute

    Returns
    -------
    numpy.ndarray

        A correctly sized (but not shaped) numpy array
        of the entire data volume

    Raises
    ------
    ImportError
        if the required packages are not available

    TypeError
        if the image could not be read by GDCM
        if the pixel data type is unsupported

    AttributeError
        if the decoded amount of data does not match the expected amount
    """

    if not HAVE_GDCM:
        msg = ("GDCM requires both the gdcm package and numpy "
               "and one or more could not be imported")
        raise ImportError(msg)

    if HAVE_GDCM_IN_MEMORY_SUPPORT:
        gdcm_data_element = create_data_element(dicom_dataset)
        gdcm_image = create_image(dicom_dataset, gdcm_data_element)
    else:
        gdcm_image_reader = create_image_reader(dicom_dataset.filename)
        if not gdcm_image_reader.Read():
            raise TypeError("GDCM could not read DICOM image")
        gdcm_image = gdcm_image_reader.GetImage()

    # GDCM returns char* as type str. Under Python 2 `str` are
    # byte arrays by default. Python 3 decodes this to
    # unicode strings by default.
    # The SWIG docs mention that they always decode byte streams
    # as utf-8 strings for Python 3, with the `surrogateescape`
    # error handler configured.
    # Therefore, we can encode them back to their original bytearray
    # representation on Python 3 by using the same parameters.
    if compat.in_py2:
        pixel_bytearray = gdcm_image.GetBuffer()
    else:
        pixel_bytearray = gdcm_image.GetBuffer().encode(
            "utf-8", "surrogateescape")

    # Here we need to be careful because in some cases, GDCM reads a
    # buffer that is too large, so we need to make sure we only include
    # the first n_rows * n_columns * dtype_size bytes.
    expected_length_bytes = get_expected_length(dicom_dataset)
    if len(pixel_bytearray) > expected_length_bytes:
        # We make sure that all the bytes after are in fact zeros
        padding = pixel_bytearray[expected_length_bytes:]
        if numpy.any(numpy.frombuffer(padding, numpy.byte)):
            pixel_bytearray = pixel_bytearray[:expected_length_bytes]
        else:
            # We revert to the old behavior which should then result
            #   in a Numpy error later on.
            pass

    numpy_dtype = pixel_dtype(dicom_dataset)
    pixel_array = numpy.frombuffer(pixel_bytearray, dtype=numpy_dtype)

    expected_length_pixels = get_expected_length(dicom_dataset, 'pixels')
    if pixel_array.size != expected_length_pixels:
        raise AttributeError("Amount of pixel data %d does "
                             "not match the expected data %d" %
                             (pixel_array.size, expected_length_pixels))

    if should_change_PhotometricInterpretation_to_RGB(dicom_dataset):
        dicom_dataset.PhotometricInterpretation = "RGB"

    return pixel_array.copy()

```
### 7 - pydicom/compat.py:

```python
# Copyright 2008-2018 pydicom authors. See LICENSE file for details.
"""Compatibility functions for python 2 vs later versions"""

# These are largely modeled on Armin Ronacher's porting advice
# at http://lucumr.pocoo.org/2013/5/21/porting-to-python-3-redux/

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

```python
# Copyright 2008-2018 pydicom authors. See LICENSE file for details.
"""Use the pillow python package to decode pixel transfer syntaxes."""

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


def get_pixeldata(dicom_dataset):
    """Use Pillow to decompress compressed Pixel Data.

    Returns
    -------
    numpy.ndarray
       The contents of the Pixel Data element (7FE0,0010) as an ndarray.

    Raises
    ------
    ImportError
        If PIL is not available.

    NotImplementedError
        if the transfer syntax is not supported

    TypeError
        if the pixel data type is unsupported
    """
    logger.debug("Trying to use Pillow to read pixel array "
                 "(has pillow = %s)", HAVE_PIL)
    transfer_syntax = dicom_dataset.file_meta.TransferSyntaxUID
    if not HAVE_PIL:
        msg = ("The pillow package is required to use pixel_array for "
               "this transfer syntax {0}, and pillow could not be "
               "imported.".format(transfer_syntax.name))
        raise ImportError(msg)

    if not HAVE_JPEG and transfer_syntax in PillowJPEGTransferSyntaxes:
        msg = ("this transfer syntax {0}, can not be read because "
               "Pillow lacks the jpeg decoder plugin"
               .format(transfer_syntax.name))
        raise NotImplementedError(msg)

    if not HAVE_JPEG2K and transfer_syntax in PillowJPEG2000TransferSyntaxes:
        msg = ("this transfer syntax {0}, can not be read because "
               "Pillow lacks the jpeg 2000 decoder plugin"
               .format(transfer_syntax.name))
        raise NotImplementedError(msg)

    if transfer_syntax not in PillowSupportedTransferSyntaxes:
        msg = ("this transfer syntax {0}, can not be read because "
               "Pillow does not support this syntax"
               .format(transfer_syntax.name))
        raise NotImplementedError(msg)

    # Make NumPy format code, e.g. "uint16", "int32" etc
    # from two pieces of info:
    # dicom_dataset.PixelRepresentation -- 0 for unsigned, 1 for signed;
    # dicom_dataset.BitsAllocated -- 8, 16, or 32
    if dicom_dataset.PixelRepresentation == 0:
        format_str = 'uint{}'.format(dicom_dataset.BitsAllocated)
    elif dicom_dataset.PixelRepresentation == 1:
        format_str = 'int{}'.format(dicom_dataset.BitsAllocated)
    else:
        format_str = 'bad_pixel_representation'
    try:
        numpy_format = numpy.dtype(format_str)
    except TypeError:
        msg = ("Data type not understood by NumPy: "
               "format='{}', PixelRepresentation={}, "
               "BitsAllocated={}".format(
                   format_str,
                   dicom_dataset.PixelRepresentation,
                   dicom_dataset.BitsAllocated))
        raise TypeError(msg)

    numpy_format = dtype_corrected_for_endianness(
        dicom_dataset.is_little_endian, numpy_format)

    # decompress here
    if transfer_syntax in PillowJPEGTransferSyntaxes:
        logger.debug("This is a JPEG lossy format")
        if dicom_dataset.BitsAllocated > 8:
            raise NotImplementedError("JPEG Lossy only supported if "
                                      "Bits Allocated = 8")
        generic_jpeg_file_header = b''
        frame_start_from = 0
    elif transfer_syntax in PillowJPEG2000TransferSyntaxes:
        logger.debug("This is a JPEG 2000 format")
        generic_jpeg_file_header = b''
        # generic_jpeg_file_header = b'\x00\x00\x00\x0C\x6A'
        #     b'\x50\x20\x20\x0D\x0A\x87\x0A'
        frame_start_from = 0
    else:
        logger.debug("This is a another pillow supported format")
        generic_jpeg_file_header = b''
        frame_start_from = 0

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
# Copyright 2008-2018 pydicom authors. See LICENSE file for details.
"""Many point of entry for pydicom read and write functions"""

from pydicom.filereader import (dcmread, read_file, read_dicomdir)
from pydicom.filewriter import dcmwrite, write_file

```
### 10 - pydicom/pixel_data_handlers/jpeg_ls_handler.py:

```python
# Copyright 2008-2018 pydicom authors. See LICENSE file for details.
"""
Use the jpeg_ls (CharPyLS) python package to decode pixel transfer syntaxes.
"""

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


def get_pixeldata(dicom_dataset):
    """
    Use the jpeg_ls package to decode the PixelData attribute

    Returns
    -------
    numpy.ndarray

        A correctly sized (but not shaped) numpy array
        of the entire data volume

    Raises
    ------
    ImportError
        if the required packages are not available

    NotImplementedError
        if the transfer syntax is not supported

    TypeError
        if the pixel data type is unsupported
    """
    if (dicom_dataset.file_meta.TransferSyntaxUID
            not in SUPPORTED_TRANSFER_SYNTAXES):
        msg = ("The jpeg_ls does not support "
               "this transfer syntax {0}.".format(
                   dicom_dataset.file_meta.TransferSyntaxUID.name))
        raise NotImplementedError(msg)

    if not HAVE_JPEGLS:
        msg = ("The jpeg_ls package is required to use pixel_array "
               "for this transfer syntax {0}, and jpeg_ls could not "
               "be imported.".format(
                   dicom_dataset.file_meta.TransferSyntaxUID.name))
        raise ImportError(msg)
    # Make NumPy format code, e.g. "uint16", "int32" etc
    # from two pieces of info:
    # dicom_dataset.PixelRepresentation -- 0 for unsigned, 1 for signed;
    # dicom_dataset.BitsAllocated -- 8, 16, or 32
    if dicom_dataset.PixelRepresentation == 0:
        format_str = 'uint{}'.format(dicom_dataset.BitsAllocated)
    elif dicom_dataset.PixelRepresentation == 1:
        format_str = 'int{}'.format(dicom_dataset.BitsAllocated)
    else:
        format_str = 'bad_pixel_representation'
    try:
        numpy_format = numpy.dtype(format_str)
    except TypeError:
        msg = ("Data type not understood by NumPy: "
               "format='{}', PixelRepresentation={}, "
               "BitsAllocated={}".format(
                   format_str,
                   dicom_dataset.PixelRepresentation,
                   dicom_dataset.BitsAllocated))
        raise TypeError(msg)

    numpy_format = dtype_corrected_for_endianness(
        dicom_dataset.is_little_endian, numpy_format)

    # decompress here
    UncompressedPixelData = bytearray()
    if ('NumberOfFrames' in dicom_dataset and
            dicom_dataset.NumberOfFrames > 1):
        # multiple compressed frames
        CompressedPixelDataSeq = pydicom.encaps.decode_data_sequence(
            dicom_dataset.PixelData)
        # print len(CompressedPixelDataSeq)
        for frame in CompressedPixelDataSeq:
            decompressed_image = jpeg_ls.decode(
                numpy.frombuffer(frame, dtype=numpy.uint8))
            UncompressedPixelData.extend(decompressed_image.tobytes())
    else:
        # single compressed frame
        CompressedPixelData = pydicom.encaps.defragment_data(
            dicom_dataset.PixelData)
        decompressed_image = jpeg_ls.decode(
            numpy.frombuffer(CompressedPixelData, dtype=numpy.uint8))
        UncompressedPixelData.extend(decompressed_image.tobytes())

    pixel_array = numpy.frombuffer(UncompressedPixelData, numpy_format)
    if should_change_PhotometricInterpretation_to_RGB(dicom_dataset):
        dicom_dataset.PhotometricInterpretation = "RGB"

    return pixel_array

```
