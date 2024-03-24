# pydicom__pydicom-1694

| **pydicom/pydicom** | `f8cf45b6c121e5a4bf4a43f71aba3bc64af3db9c` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 13855 |
| **Any found context length** | 13855 |
| **Avg pos** | 29.0 |
| **Min pos** | 29 |
| **Max pos** | 29 |
| **Top file pos** | 2 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/pydicom/dataset.py b/pydicom/dataset.py
--- a/pydicom/dataset.py
+++ b/pydicom/dataset.py
@@ -2492,8 +2492,8 @@ def to_json_dict(
         json_dataset = {}
         for key in self.keys():
             json_key = '{:08X}'.format(key)
-            data_element = self[key]
             try:
+                data_element = self[key]
                 json_dataset[json_key] = data_element.to_json_dict(
                     bulk_data_element_handler=bulk_data_element_handler,
                     bulk_data_threshold=bulk_data_threshold

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| pydicom/dataset.py | 2495 | 2497 | 29 | 2 | 13855


## Problem Statement

```
Dataset.to_json_dict can still generate exceptions when suppress_invalid_tags=True
**Describe the bug**
I'm using `Dataset.to_json_dict(suppress_invalid_tags=True)` and can live with losing invalid tags.  Unfortunately, I can still trigger an exception with something like  `2.0` in an `IS` field.

**Expected behavior**
to_json_dict shouldn't throw an error about an invalid tag when `suppress_invalid_tags` is enabled.

My thought was simply to move the `data_element = self[key]` into the try/catch block that's right after it.

**Steps To Reproduce**

Traceback:
\`\`\`
  File "dicom.py", line 143, in create_dict
    json_ds = ds.to_json_dict(suppress_invalid_tags=True)
  File "/usr/lib/python3/dist-packages/pydicom/dataset.py", line 2495, in to_json_dict
    data_element = self[key]
  File "/usr/lib/python3/dist-packages/pydicom/dataset.py", line 939, in __getitem__
    self[tag] = DataElement_from_raw(elem, character_set, self)
  File "/usr/lib/python3/dist-packages/pydicom/dataelem.py", line 859, in DataElement_from_raw
    value = convert_value(vr, raw, encoding)
  File "/usr/lib/python3/dist-packages/pydicom/values.py", line 771, in convert_value
    return converter(byte_string, is_little_endian, num_format)
  File "/usr/lib/python3/dist-packages/pydicom/values.py", line 348, in convert_IS_string
    return MultiString(num_string, valtype=pydicom.valuerep.IS)
  File "/usr/lib/python3/dist-packages/pydicom/valuerep.py", line 1213, in MultiString
    return valtype(splitup[0])
  File "/usr/lib/python3/dist-packages/pydicom/valuerep.py", line 1131, in __new__
    raise TypeError("Could not convert value to integer without loss")
TypeError: Could not convert value to integer without loss
\`\`\`

**Your environment**
python 3.7, pydicom 2.3



```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- |
| 1 | 1 pydicom/config.py | 364 | 495| 1150 | 1150 | 
| 2 | **2 pydicom/dataset.py** | 2846 | 2877| 217 | 1367 | 
| 3 | 3 pydicom/dataelem.py | 9 | 39| 278 | 1645 | 
| 4 | **3 pydicom/dataset.py** | 219 | 220| 31 | 1676 | 
| 5 | **3 pydicom/dataset.py** | 921 | 947| 265 | 1941 | 
| 6 | **3 pydicom/dataset.py** | 1277 | 1321| 345 | 2286 | 
| 7 | **3 pydicom/dataset.py** | 1323 | 1393| 531 | 2817 | 
| 8 | **3 pydicom/dataset.py** | 16 | 62| 389 | 3206 | 
| 9 | 4 pydicom/tag.py | 144 | 240| 852 | 4058 | 
| 10 | **4 pydicom/dataset.py** | 223 | 364| 1239 | 5297 | 
| 11 | 5 pydicom/filewriter.py | 3 | 59| 769 | 6066 | 
| 12 | 6 pydicom/datadict.py | 3 | 27| 274 | 6340 | 
| 13 | **6 pydicom/dataset.py** | 483 | 523| 273 | 6613 | 
| 14 | 7 pydicom/fileset.py | 3 | 74| 596 | 7209 | 
| 15 | 8 source/generate_dict/generate_dicom_dict.py | 236 | 369| 1279 | 8488 | 
| 16 | 9 pydicom/valuerep.py | 378 | 404| 386 | 8874 | 
| 17 | **9 pydicom/dataset.py** | 2106 | 2161| 519 | 9393 | 
| 18 | **9 pydicom/dataset.py** | 2879 | 2907| 201 | 9594 | 
| 19 | 10 pydicom/errors.py | 6 | 3| 155 | 9749 | 
| 20 | 10 pydicom/filewriter.py | 69 | 161| 949 | 10698 | 
| 21 | **10 pydicom/dataset.py** | 1113 | 1111| 320 | 11018 | 
| 22 | 11 pydicom/util/codify.py | 16 | 48| 290 | 11308 | 
| 23 | **11 pydicom/dataset.py** | 2045 | 2069| 296 | 11604 | 
| 24 | **11 pydicom/dataset.py** | 1670 | 1731| 536 | 12140 | 
| 25 | **11 pydicom/dataset.py** | 2739 | 2802| 581 | 12721 | 
| 26 | **11 pydicom/dataset.py** | 162 | 178| 116 | 12837 | 
| 27 | **11 pydicom/dataset.py** | 596 | 643| 443 | 13280 | 
| 28 | 11 pydicom/datadict.py | 288 | 310| 158 | 13438 | 
| **-> 29 <-** | **11 pydicom/dataset.py** | 2459 | 2505| 417 | 13855 | 


## Patch

```diff
diff --git a/pydicom/dataset.py b/pydicom/dataset.py
--- a/pydicom/dataset.py
+++ b/pydicom/dataset.py
@@ -2492,8 +2492,8 @@ def to_json_dict(
         json_dataset = {}
         for key in self.keys():
             json_key = '{:08X}'.format(key)
-            data_element = self[key]
             try:
+                data_element = self[key]
                 json_dataset[json_key] = data_element.to_json_dict(
                     bulk_data_element_handler=bulk_data_element_handler,
                     bulk_data_threshold=bulk_data_threshold

```

## Test Patch

```diff
diff --git a/pydicom/tests/test_json.py b/pydicom/tests/test_json.py
--- a/pydicom/tests/test_json.py
+++ b/pydicom/tests/test_json.py
@@ -7,7 +7,7 @@
 
 from pydicom import dcmread
 from pydicom.data import get_testdata_file
-from pydicom.dataelem import DataElement
+from pydicom.dataelem import DataElement, RawDataElement
 from pydicom.dataset import Dataset
 from pydicom.tag import Tag, BaseTag
 from pydicom.valuerep import PersonName
@@ -284,7 +284,23 @@ def test_suppress_invalid_tags(self, _):
 
         ds_json = ds.to_json_dict(suppress_invalid_tags=True)
 
-        assert ds_json.get("00100010") is None
+        assert "00100010" not in ds_json
+
+    def test_suppress_invalid_tags_with_failed_dataelement(self):
+        """Test tags that raise exceptions don't if suppress_invalid_tags True.
+        """
+        ds = Dataset()
+        # we have to add a RawDataElement as creating a DataElement would
+        # already raise an exception
+        ds[0x00082128] = RawDataElement(
+            Tag(0x00082128), 'IS', 4, b'5.25', 0, True, True)
+
+        with pytest.raises(TypeError):
+            ds.to_json_dict()
+
+        ds_json = ds.to_json_dict(suppress_invalid_tags=True)
+
+        assert "00082128" not in ds_json
 
 
 class TestSequence:

```


## Code snippets

### 1 - pydicom/config.py:

Start line: 364, End line: 495

```python
import pydicom.pixel_data_handlers.rle_handler as rle_handler  # noqa
import pydicom.pixel_data_handlers.pillow_handler as pillow_handler  # noqa
import pydicom.pixel_data_handlers.jpeg_ls_handler as jpegls_handler  # noqa
import pydicom.pixel_data_handlers.gdcm_handler as gdcm_handler  # noqa
import pydicom.pixel_data_handlers.pylibjpeg_handler as pylibjpeg_handler  # noqa

pixel_data_handlers = [
    np_handler,
    gdcm_handler,
    pillow_handler,
    jpegls_handler,
    pylibjpeg_handler,
    rle_handler,
]
"""Handlers for converting (7FE0,0010) *Pixel Data*.

.. versionadded:: 1.2

.. currentmodule:: pydicom.dataset

This is an ordered list of *Pixel Data* handlers that the
:meth:`~Dataset.convert_pixel_data` method will use to try to extract a
correctly sized numpy array from the *Pixel Data* element.

Handlers shall have four methods:

def supports_transfer_syntax(transfer_syntax: UID)
    Return ``True`` if the handler supports the transfer syntax indicated in
    :class:`Dataset` `ds`, ``False`` otherwise.

def is_available():
    Return ``True`` if the handler's dependencies are installed, ``False``
    otherwise.

def get_pixeldata(ds):
    Return a correctly sized 1D :class:`numpy.ndarray` derived from the
    *Pixel Data* in :class:`Dataset` `ds` or raise an exception. Reshaping the
    returned array to the correct dimensions is handled automatically.

def needs_to_convert_to_RGB(ds):
    Return ``True`` if the *Pixel Data* in the :class:`Dataset` `ds` needs to
    be converted to the RGB colourspace, ``False`` otherwise.

The first handler that both announces that it supports the transfer syntax
and does not raise an exception, either in getting the data or when the data
is reshaped to the correct dimensions, is the handler that will provide the
data.

If they all fail only the last exception is raised.

If none raise an exception, but they all refuse to support the transfer
syntax, then this fact is announced in a :class:`NotImplementedError`
exception.
"""

APPLY_J2K_CORRECTIONS = True
"""Use the information within JPEG 2000 data to correct the returned pixel data

.. versionadded:: 2.1

If ``True`` (default), then for handlers that support JPEG 2000 pixel data,
use the component precision and sign to correct the returned ndarray when
using the pixel data handlers. If ``False`` then only rely on the element
values within the dataset when applying corrections.
"""

assume_implicit_vr_switch = True
"""If invalid VR encountered, assume file switched to implicit VR

.. versionadded:: 2.2

If ``True`` (default), when reading an explicit VR file,
if a VR is encountered that is not a valid two bytes within A-Z,
then assume the original writer switched to implicit VR.  This has been
seen in particular in some sequences.  This does not test that
the VR is a valid DICOM VR, just that it has valid characters.
"""


INVALID_KEYWORD_BEHAVIOR = "WARN"
"""Control the behavior when setting a :class:`~pydicom.dataset.Dataset`
attribute that's not a known element keyword.

.. versionadded:: 2.1

If ``"WARN"`` (default), then warn when an element value is set using
``Dataset.__setattr__()`` and the keyword is camel case but doesn't match a
known DICOM element keyword. If ``"RAISE"`` then raise a :class:`ValueError`
exception. If ``"IGNORE"`` then neither warn nor raise.

Examples
--------

>>> from pydicom import config
>>> config.INVALID_KEYWORD_BEHAVIOR = "WARN"
>>> ds = Dataset()
>>> ds.PatientName = "Citizen^Jan"  # OK
>>> ds.PatientsName = "Citizen^Jan"
../pydicom/dataset.py:1895: UserWarning: Camel case attribute 'PatientsName'
used which is not in the element keyword data dictionary
"""

INVALID_KEY_BEHAVIOR = "WARN"
"""Control the behavior when invalid keys are used with
:meth:`~pydicom.dataset.Dataset.__contains__` (e.g. ``'invalid' in ds``).

.. versionadded:: 2.1

Invalid keys are objects that cannot be converted to a
:class:`~pydicom.tag.BaseTag`, such as unknown element keywords or invalid
element tags like ``0x100100010``.

If ``"WARN"`` (default), then warn when an invalid key is used, if ``"RAISE"``
then raise a :class:`ValueError` exception. If ``"IGNORE"`` then neither warn
nor raise.

Examples
--------

>>> from pydicom import config
>>> config.INVALID_KEY_BEHAVIOR = "RAISE"
>>> ds = Dataset()
>>> 'PatientName' in ds  # OK
False
>>> 'PatientsName' in ds
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File ".../pydicom/dataset.py", line 494, in __contains__
    raise ValueError(msg) from exc
ValueError: Invalid value used with the 'in' operator: must be an
element tag as a 2-tuple or int, or an element keyword
"""
```
### 2 - pydicom/dataset.py:

Start line: 2846, End line: 2877

```python
class FileMetaDataset(Dataset):

    @staticmethod
    def validate(init_value: _DatasetType) -> None:
        """Raise errors if initialization value is not acceptable for file_meta

        Parameters
        ----------
        init_value: dict or Dataset
            The tag:data element pairs to initialize a file meta dataset

        Raises
        ------
        TypeError
            If the passed argument is not a :class:`dict` or :class:`Dataset`
        ValueError
            If any data elements passed are not group 2.
        """
        if init_value is None:
            return

        if not isinstance(init_value, (Dataset, dict)):
            raise TypeError(
                "Argument must be a dict or Dataset, not {}".format(
                    type(init_value)
                )
            )

        non_group2 = [
            Tag(tag) for tag in init_value.keys() if Tag(tag).group != 2
        ]
        if non_group2:
            msg = "Attempted to set non-group 2 elements: {}"
            raise ValueError(msg.format(non_group2))
```
### 3 - pydicom/dataelem.py:

Start line: 9, End line: 39

```python
import base64
import json
from typing import (
    Optional, Any, Tuple, Callable, Union, TYPE_CHECKING, Dict, Type,
    List, NamedTuple, MutableSequence
)
import warnings

from pydicom import config  # don't import datetime_conversion directly
from pydicom.config import logger
from pydicom.datadict import (dictionary_has_tag, dictionary_description,
                              dictionary_keyword, dictionary_is_retired,
                              private_dictionary_description, dictionary_VR,
                              repeater_has_tag, private_dictionary_VR)
from pydicom.errors import BytesLengthException
from pydicom.jsonrep import JsonDataElementConverter, BulkDataType
from pydicom.multival import MultiValue
from pydicom.tag import Tag, BaseTag
from pydicom.uid import UID
from pydicom import jsonrep
import pydicom.valuerep  # don't import DS directly as can be changed by config
from pydicom.valuerep import (
    PersonName, BYTES_VR, AMBIGUOUS_VR, STR_VR, ALLOW_BACKSLASH,
    DEFAULT_CHARSET_VR, LONG_VALUE_VR, VR as VR_, validate_value
)

if config.have_numpy:
    import numpy

if TYPE_CHECKING:  # pragma: no cover
    from pydicom.dataset import Dataset
```
### 4 - pydicom/dataset.py:

Start line: 219, End line: 220

```python
_DatasetValue = Union[DataElement, RawDataElement]
_DatasetType = Union["Dataset", MutableMapping[BaseTag, _DatasetValue]]
```
### 5 - pydicom/dataset.py:

Start line: 921, End line: 947

```python
class Dataset:

    def __getitem__(
        self, key: Union[slice, TagType]
    ) -> Union["Dataset", DataElement]:
        # ... other code

        if isinstance(elem, RawDataElement):
            # If a deferred read, then go get the value now
            if elem.value is None and elem.length != 0:
                from pydicom.filereader import read_deferred_data_element

                elem = read_deferred_data_element(
                    self.fileobj_type,
                    self.filename,
                    self.timestamp,
                    elem
                )

            if tag != BaseTag(0x00080005):
                character_set = self.read_encoding or self._character_set
            else:
                character_set = default_encoding
            # Not converted from raw form read from file yet; do so now
            self[tag] = DataElement_from_raw(elem, character_set, self)

            # If the Element has an ambiguous VR, try to correct it
            if self[tag].VR in AMBIGUOUS_VR:
                from pydicom.filewriter import correct_ambiguous_vr_element
                self[tag] = correct_ambiguous_vr_element(
                    self[tag], self, elem[6]
                )

        return cast(DataElement, self._dict.get(tag))
```
### 6 - pydicom/dataset.py:

Start line: 1277, End line: 1321

```python
class Dataset:

    def pop(self, key: Union[BaseTag, TagType], *args: Any) -> _DatasetValue:
        """Emulate :meth:`dict.pop` with support for tags and keywords.

        Removes the element for `key` if it exists and returns it,
        otherwise returns a default value if given or raises :class:`KeyError`.

        Parameters
        ----------
        key : int or str or 2-tuple

            * If :class:`tuple` - the group and element number of the DICOM tag
            * If :class:`int` - the combined group/element number
            * If :class:`str` - the DICOM keyword of the tag

        *args : zero or one argument
            Defines the behavior if no tag exists for `key`: if given,
            it defines the return value, if not given, :class:`KeyError` is
            raised

        Returns
        -------
        RawDataElement or DataElement
            The element for `key` if it exists, or the default value if given.

        Raises
        ------
        KeyError
            If the `key` is not a valid tag or keyword.
            If the tag does not exist and no default is given.
        """
        try:
            key = Tag(key)
        except Exception:
            pass

        return self._dict.pop(cast(BaseTag, key), *args)

    def popitem(self) -> Tuple[BaseTag, _DatasetValue]:
        """Emulate :meth:`dict.popitem`.

        Returns
        -------
        tuple of (BaseTag, DataElement)
        """
        return self._dict.popitem()
```
### 7 - pydicom/dataset.py:

Start line: 1323, End line: 1393

```python
class Dataset:

    def setdefault(
        self, key: TagType, default: Optional[Any] = None
    ) -> DataElement:
        """Emulate :meth:`dict.setdefault` with support for tags and keywords.

        Examples
        --------

        >>> ds = Dataset()
        >>> elem = ds.setdefault((0x0010, 0x0010), "Test")
        >>> elem
        (0010, 0010) Patient's Name                      PN: 'Test'
        >>> elem.value
        'Test'
        >>> elem = ds.setdefault('PatientSex',
        ...     DataElement(0x00100040, 'CS', 'F'))
        >>> elem.value
        'F'

        Parameters
        ----------
        key : int, str or 2-tuple of int

            * If :class:`tuple` - the group and element number of the DICOM tag
            * If :class:`int` - the combined group/element number
            * If :class:`str` - the DICOM keyword of the tag
        default : pydicom.dataelem.DataElement or object, optional
            The :class:`~pydicom.dataelem.DataElement` to use with `key`, or
            the value of the :class:`~pydicom.dataelem.DataElement` to use with
            `key` (default ``None``).

        Returns
        -------
        pydicom.dataelem.DataElement or object
            The :class:`~pydicom.dataelem.DataElement` for `key`.

        Raises
        ------
        ValueError
            If `key` is not convertible to a valid tag or a known element
            keyword.
        KeyError
            If :attr:`~pydicom.config.settings.reading_validation_mode` is
             ``RAISE`` and `key` is an unknown non-private tag.
        """
        tag = Tag(key)
        if tag in self:
            return self[tag]

        vr: Union[str, VR_]
        if not isinstance(default, DataElement):
            if tag.is_private:
                vr = VR_.UN
            else:
                try:
                    vr = dictionary_VR(tag)
                except KeyError:
                    if (config.settings.writing_validation_mode ==
                            config.RAISE):
                        raise KeyError(f"Unknown DICOM tag {tag}")

                    vr = VR_.UN
                    warnings.warn(
                        f"Unknown DICOM tag {tag} - setting VR to 'UN'"
                    )

            default = DataElement(tag, vr, default)

        self[key] = default

        return default
```
### 8 - pydicom/dataset.py:

Start line: 16, End line: 62

```python
import copy
from bisect import bisect_left
import io
from importlib.util import find_spec as have_package
import inspect  # for __dir__
from itertools import takewhile
import json
import os
import os.path
import re
from types import TracebackType
from typing import (
    Optional, Tuple, Union, List, Any, cast, Dict, ValuesView,
    Iterator, BinaryIO, AnyStr, Callable, TypeVar, Type, overload,
    MutableSequence, MutableMapping, AbstractSet
)
import warnings
import weakref

from pydicom.filebase import DicomFileLike

try:
    import numpy
except ImportError:
    pass

import pydicom  # for dcmwrite
from pydicom import jsonrep, config
from pydicom._version import __version_info__
from pydicom.charset import default_encoding, convert_encodings
from pydicom.config import logger
from pydicom.datadict import (
    dictionary_VR, tag_for_keyword, keyword_for_tag, repeater_has_keyword
)
from pydicom.dataelem import DataElement, DataElement_from_raw, RawDataElement
from pydicom.encaps import encapsulate, encapsulate_extended
from pydicom.fileutil import path_from_pathlike, PathType
from pydicom.pixel_data_handlers.util import (
    convert_color_space, reshape_pixel_array, get_image_pixel_ids
)
from pydicom.tag import Tag, BaseTag, tag_in_exception, TagType
from pydicom.uid import (
    ExplicitVRLittleEndian, ImplicitVRLittleEndian, ExplicitVRBigEndian,
    RLELossless, PYDICOM_IMPLEMENTATION_UID, UID
)
from pydicom.valuerep import VR as VR_, AMBIGUOUS_VR
from pydicom.waveforms import numpy_handler as wave_handler
```
### 9 - pydicom/tag.py:

Start line: 144, End line: 240

```python
class BaseTag(int):
    """Represents a DICOM element (group, element) tag.

    Tags are represented as an :class:`int`.
    """
    # Override comparisons so can convert "other" to Tag as necessary
    #   See Ordering Comparisons at:
    #   http://docs.python.org/dev/3.0/whatsnew/3.0.html
    def __le__(self, other: Any) -> Any:
        """Return ``True`` if `self`  is less than or equal to `other`."""
        return self == other or self < other

    def __lt__(self, other: Any) -> Any:
        """Return ``True`` if `self` is less than `other`."""
        # Check if comparing with another Tag object; if not, create a temp one
        if not isinstance(other, int):
            try:
                other = Tag(other)
            except Exception:
                raise TypeError("Cannot compare Tag with non-Tag item")

        return int(self) < int(other)

    def __ge__(self, other: Any) -> Any:
        """Return ``True`` if `self` is greater than or equal to `other`."""
        return self == other or self > other

    def __gt__(self, other: Any) -> Any:
        """Return ``True`` if `self` is greater than `other`."""
        return not (self == other or self < other)

    def __eq__(self, other: Any) -> Any:
        """Return ``True`` if `self` equals `other`."""
        # Check if comparing with another Tag object; if not, create a temp one
        if not isinstance(other, int):
            try:
                other = Tag(other)
            except Exception:
                return False

        return int(self) == int(other)

    def __ne__(self, other: Any) -> Any:
        """Return ``True`` if `self` does not equal `other`."""
        return not self == other

    # For python 3, any override of __cmp__ or __eq__
    # immutable requires explicit redirect of hash function
    # to the parent class
    #   See http://docs.python.org/dev/3.0/reference/
    #              datamodel.html#object.__hash__
    __hash__ = int.__hash__

    def __str__(self) -> str:
        """Return the tag value as a hex string '(gggg, eeee)'."""
        return "({0:04x}, {1:04x})".format(self.group, self.element)

    __repr__ = __str__

    @property
    def json_key(self) -> str:
        """Return the tag value as a JSON key string 'GGGGEEEE'."""
        return f"{self.group:04X}{self.element:04X}"

    @property
    def group(self) -> int:
        """Return the tag's group number as :class:`int`."""
        return self >> 16

    @property
    def element(self) -> int:
        """Return the tag's element number as :class:`int`."""
        return self & 0xffff

    elem = element  # alternate syntax

    @property
    def is_private(self) -> bool:
        """Return ``True`` if the tag is private (has an odd group number)."""
        return self.group % 2 == 1

    @property
    def is_private_creator(self) -> bool:
        """Return ``True`` if the tag is a private creator.

        .. versionadded:: 1.1
        """
        return self.is_private and 0x0010 <= self.element < 0x0100

    @property
    def private_creator(self) -> "BaseTag":
        """Return the private creator tag for the given tag.
        The result is meaningless if this is not a private tag.

        .. versionadded:: 2.4
        """
        return BaseTag((self & 0xffff0000) | self.element >> 8)
```
### 10 - pydicom/dataset.py:

Start line: 223, End line: 364

```python
class Dataset:
    """A DICOM dataset as a mutable mapping of DICOM Data Elements.

    Examples
    --------
    Add an element to the :class:`Dataset` (for elements in the DICOM
    dictionary):

    >>> ds = Dataset()
    >>> ds.PatientName = "CITIZEN^Joan"
    >>> ds.add_new(0x00100020, 'LO', '12345')
    >>> ds[0x0010, 0x0030] = DataElement(0x00100030, 'DA', '20010101')

    Add a sequence element to the :class:`Dataset`

    >>> ds.BeamSequence = [Dataset(), Dataset(), Dataset()]
    >>> ds.BeamSequence[0].Manufacturer = "Linac, co."
    >>> ds.BeamSequence[1].Manufacturer = "Linac and Sons, co."
    >>> ds.BeamSequence[2].Manufacturer = "Linac and Daughters, co."

    Add private elements to the :class:`Dataset`

    >>> block = ds.private_block(0x0041, 'My Creator', create=True)
    >>> block.add_new(0x01, 'LO', '12345')

    Updating and retrieving element values:

    >>> ds.PatientName = "CITIZEN^Joan"
    >>> ds.PatientName
    'CITIZEN^Joan'
    >>> ds.PatientName = "CITIZEN^John"
    >>> ds.PatientName
    'CITIZEN^John'

    Retrieving an element's value from a Sequence:

    >>> ds.BeamSequence[0].Manufacturer
    'Linac, co.'
    >>> ds.BeamSequence[1].Manufacturer
    'Linac and Sons, co.'

    Accessing the :class:`~pydicom.dataelem.DataElement` items:

    >>> elem = ds['PatientName']
    >>> elem
    (0010, 0010) Patient's Name                      PN: 'CITIZEN^John'
    >>> elem = ds[0x00100010]
    >>> elem
    (0010, 0010) Patient's Name                      PN: 'CITIZEN^John'
    >>> elem = ds.data_element('PatientName')
    >>> elem
    (0010, 0010) Patient's Name                      PN: 'CITIZEN^John'

    Accessing a private :class:`~pydicom.dataelem.DataElement`
    item:

    >>> block = ds.private_block(0x0041, 'My Creator')
    >>> elem = block[0x01]
    >>> elem
    (0041, 1001) Private tag data                    LO: '12345'
    >>> elem.value
    '12345'

    Alternatively:

    >>> ds.get_private_item(0x0041, 0x01, 'My Creator').value
    '12345'

    Deleting an element from the :class:`Dataset`

    >>> del ds.PatientID
    >>> del ds.BeamSequence[1].Manufacturer
    >>> del ds.BeamSequence[2]

    Deleting a private element from the :class:`Dataset`

    >>> block = ds.private_block(0x0041, 'My Creator')
    >>> if 0x01 in block:
    ...     del block[0x01]

    Determining if an element is present in the :class:`Dataset`

    >>> 'PatientName' in ds
    True
    >>> 'PatientID' in ds
    False
    >>> (0x0010, 0x0030) in ds
    True
    >>> 'Manufacturer' in ds.BeamSequence[0]
    True

    Iterating through the top level of a :class:`Dataset` only (excluding
    Sequences):

    >>> for elem in ds:
    ...    print(elem)
    (0010, 0010) Patient's Name                      PN: 'CITIZEN^John'

    Iterating through the entire :class:`Dataset` (including Sequences):

    >>> for elem in ds.iterall():
    ...     print(elem)
    (0010, 0010) Patient's Name                      PN: 'CITIZEN^John'

    Recursively iterate through a :class:`Dataset` (including Sequences):

    >>> def recurse(ds):
    ...     for elem in ds:
    ...         if elem.VR == 'SQ':
    ...             [recurse(item) for item in elem.value]
    ...         else:
    ...             # Do something useful with each DataElement

    Converting the :class:`Dataset` to and from JSON:

    >>> ds = Dataset()
    >>> ds.PatientName = "Some^Name"
    >>> jsonmodel = ds.to_json()
    >>> ds2 = Dataset()
    >>> ds2.from_json(jsonmodel)
    (0010, 0010) Patient's Name                      PN: 'Some^Name'

    Attributes
    ----------
    default_element_format : str
        The default formatting for string display.
    default_sequence_element_format : str
        The default formatting for string display of sequences.
    indent_chars : str
        For string display, the characters used to indent nested Sequences.
        Default is ``"   "``.
    is_little_endian : bool
        Shall be set before writing with ``write_like_original=False``.
        The :class:`Dataset` (excluding the pixel data) will be written using
        the given endianness.
    is_implicit_VR : bool
        Shall be set before writing with ``write_like_original=False``.
        The :class:`Dataset` will be written using the transfer syntax with
        the given VR handling, e.g *Little Endian Implicit VR* if ``True``,
        and *Little Endian Explicit VR* or *Big Endian Explicit VR* (depending
        on ``Dataset.is_little_endian``) if ``False``.
    """
```
### 13 - pydicom/dataset.py:

Start line: 483, End line: 523

```python
class Dataset:

    def __contains__(self, name: TagType) -> bool:
        """Simulate dict.__contains__() to handle DICOM keywords.

        Examples
        --------

        >>> ds = Dataset()
        >>> ds.SliceLocation = '2'
        >>> 'SliceLocation' in ds
        True

        Parameters
        ----------
        name : str or int or 2-tuple
            The element keyword or tag to search for.

        Returns
        -------
        bool
            ``True`` if the corresponding element is in the :class:`Dataset`,
            ``False`` otherwise.
        """
        try:
            return Tag(name) in self._dict
        except Exception as exc:
            msg = (
                f"Invalid value '{name}' used with the 'in' operator: must be "
                "an element tag as a 2-tuple or int, or an element keyword"
            )
            if isinstance(exc, OverflowError):
                msg = (
                    "Invalid element tag value used with the 'in' operator: "
                    "tags have a maximum value of (0xFFFF, 0xFFFF)"
                )

            if config.INVALID_KEY_BEHAVIOR == "WARN":
                warnings.warn(msg)
            elif config.INVALID_KEY_BEHAVIOR == "RAISE":
                raise ValueError(msg) from exc

        return False
```
### 17 - pydicom/dataset.py:

Start line: 2106, End line: 2161

```python
class Dataset:

    def __setattr__(self, name: str, value: Any) -> None:
        """Intercept any attempts to set a value for an instance attribute.

        If name is a DICOM keyword, set the corresponding tag and DataElement.
        Else, set an instance (python) attribute as any other class would do.

        Parameters
        ----------
        name : str
            The keyword for the element you wish to add/change. If
            `name` is not a DICOM element keyword then this will be the
            name of the attribute to be added/changed.
        value
            The value for the attribute to be added/changed.
        """
        tag = tag_for_keyword(name)
        if tag is not None:  # successfully mapped name to a tag
            if tag not in self:
                # don't have this tag yet->create the data_element instance
                vr = dictionary_VR(tag)
                data_element = DataElement(tag, vr, value)
                if vr == VR_.SQ:
                    # let a sequence know its parent dataset to pass it
                    # to its items, who may need parent dataset tags
                    # to resolve ambiguous tags
                    data_element.parent = self
            else:
                # already have this data_element, just changing its value
                data_element = self[tag]
                data_element.value = value
            # Now have data_element - store it in this dict
            self[tag] = data_element
        elif repeater_has_keyword(name):
            # Check if `name` is repeaters element
            raise ValueError(
                f"'{name}' is a DICOM repeating group element and must be "
                "added using the add() or add_new() methods."
            )
        elif name == "file_meta":
            self._set_file_meta(value)
        else:
            # Warn if `name` is camel case but not a keyword
            if _RE_CAMEL_CASE.match(name):
                msg = (
                    f"Camel case attribute '{name}' used which is not in the "
                    "element keyword data dictionary"
                )
                if config.INVALID_KEYWORD_BEHAVIOR == "WARN":
                    warnings.warn(msg)
                elif config.INVALID_KEYWORD_BEHAVIOR == "RAISE":
                    raise ValueError(msg)

            # name not in dicom dictionary - setting a non-dicom instance
            # attribute
            # XXX note if user mis-spells a dicom data_element - no error!!!
            object.__setattr__(self, name, value)
```
### 18 - pydicom/dataset.py:

Start line: 2879, End line: 2907

```python
class FileMetaDataset(Dataset):

    def __setitem__(
        self, key: Union[slice, TagType], value: _DatasetValue
    ) -> None:
        """Override parent class to only allow setting of group 2 elements.

        Parameters
        ----------
        key : int or Tuple[int, int] or str
            The tag for the element to be added to the Dataset.
        value : dataelem.DataElement or dataelem.RawDataElement
            The element to add to the :class:`FileMetaDataset`.

        Raises
        ------
        ValueError
            If `key` is not a DICOM Group 2 tag.
        """

        if isinstance(value.tag, BaseTag):
            tag = value.tag
        else:
            tag = Tag(value.tag)

        if tag.group != 2:
            raise ValueError(
                "Only group 2 data elements are allowed in a FileMetaDataset"
            )

        super().__setitem__(key, value)
```
### 21 - pydicom/dataset.py:

Start line: 1113, End line: 1111

```python
class Dataset:

    @overload
    def get_item(self, key: slice) -> "Dataset":
        pass  # pragma: no cover

    @overload
    def get_item(self, key: TagType) -> DataElement:
        pass  # pragma: no cover

    def get_item(
        self, key: Union[slice, TagType]
    ) -> Union["Dataset", DataElement, RawDataElement, None]:
        """Return the raw data element if possible.

        It will be raw if the user has never accessed the value, or set their
        own value. Note if the data element is a deferred-read element,
        then it is read and converted before being returned.

        Parameters
        ----------
        key
            The DICOM (group, element) tag in any form accepted by
            :func:`~pydicom.tag.Tag` such as ``[0x0010, 0x0010]``,
            ``(0x10, 0x10)``, ``0x00100010``, etc. May also be a :class:`slice`
            made up of DICOM tags.

        Returns
        -------
        dataelem.DataElement
            The corresponding element.
        """
        if isinstance(key, slice):
            return self._dataset_slice(key)

        elem = self._dict.get(Tag(key))
        # If a deferred read, return using __getitem__ to read and convert it
        if isinstance(elem, RawDataElement) and elem.value is None:
            return self[key]

        return elem
```
### 23 - pydicom/dataset.py:

Start line: 2045, End line: 2069

```python
class Dataset:

    def remove_private_tags(self) -> None:
        """Remove all private elements from the :class:`Dataset`."""

        def remove_callback(dataset: "Dataset", elem: DataElement) -> None:
            """Internal method to use as callback to walk() method."""
            if elem.tag.is_private:
                # can't del self[tag] - won't be right dataset on recursion
                del dataset[elem.tag]

        self.walk(remove_callback)

    def save_as(
        self,
        filename: Union[str, "os.PathLike[AnyStr]", BinaryIO],
        write_like_original: bool = True
    ) -> None:
        """Write the :class:`Dataset` to `filename`.

        Wrapper for pydicom.filewriter.dcmwrite, passing this dataset to it.
        See documentation for that function for details.

        See Also
        --------
        pydicom.filewriter.dcmwrite
            Write a DICOM file from a :class:`FileDataset` instance.
        """
        pydicom.dcmwrite(filename, self, write_like_original)

    def ensure_file_meta(self) -> None:
        """Create an empty ``Dataset.file_meta`` if none exists.

        .. versionadded:: 1.2
        """
        # Changed in v2.0 so does not re-assign self.file_meta with getattr()
        if not hasattr(self, "file_meta"):
            self.file_meta = FileMetaDataset()
```
### 24 - pydicom/dataset.py:

Start line: 1670, End line: 1731

```python
class Dataset:

    def compress(
        self,
        transfer_syntax_uid: str,
        arr: Optional["numpy.ndarray"] = None,
        encoding_plugin: str = '',
        decoding_plugin: str = '',
        encapsulate_ext: bool = False,
        **kwargs: Any,
    ) -> None:
        from pydicom.encoders import get_encoder

        uid = UID(transfer_syntax_uid)

        # Raises NotImplementedError if `uid` is not supported
        encoder = get_encoder(uid)
        if not encoder.is_available:
            missing = "\n".join(
                [f"    {s}" for s in encoder.missing_dependencies]
            )
            raise RuntimeError(
                f"The '{uid.name}' encoder is unavailable because its "
                f"encoding plugins are missing dependencies:\n"
                f"{missing}"
            )

        if arr is None:
            # Encode the current *Pixel Data*
            frame_iterator = encoder.iter_encode(
                self,
                encoding_plugin=encoding_plugin,
                decoding_plugin=decoding_plugin,
                **kwargs
            )
        else:
            # Encode from an uncompressed pixel data array
            kwargs.update(encoder.kwargs_from_ds(self))
            frame_iterator = encoder.iter_encode(
                arr,
                encoding_plugin=encoding_plugin,
                **kwargs
            )

        # Encode!
        encoded = [f for f in frame_iterator]

        # Encapsulate the encoded *Pixel Data*
        nr_frames = getattr(self, "NumberOfFrames", 1) or 1
        total = (nr_frames - 1) * 8 + sum([len(f) for f in encoded[:-1]])
        if encapsulate_ext or total > 2**32 - 1:
            (self.PixelData,
             self.ExtendedOffsetTable,
             self.ExtendedOffsetTableLengths) = encapsulate_extended(encoded)
        else:
            self.PixelData = encapsulate(encoded)

        # PS3.5 Annex A.4 - encapsulated pixel data uses undefined length
        self['PixelData'].is_undefined_length = True

        # PS3.5 Annex A.4 - encapsulated datasets use explicit VR little endian
        self.is_implicit_VR = False
        self.is_little_endian = True

        # Set the correct *Transfer Syntax UID*
        if not hasattr(self, 'file_meta'):
            self.file_meta = FileMetaDataset()

        self.file_meta.TransferSyntaxUID = uid

        # Add or update any other required elements
        if self.SamplesPerPixel > 1:
            self.PlanarConfiguration: int = 1 if uid == RLELossless else 0
```
### 25 - pydicom/dataset.py:

Start line: 2739, End line: 2802

```python
def validate_file_meta(
    file_meta: "FileMetaDataset", enforce_standard: bool = True
) -> None:
    """Validate the *File Meta Information* elements in `file_meta`.

    .. versionchanged:: 1.2

        Moved from :mod:`pydicom.filewriter`.

    Parameters
    ----------
    file_meta : Dataset
        The *File Meta Information* data elements.
    enforce_standard : bool, optional
        If ``False``, then only a check for invalid elements is performed.
        If ``True`` (default), the following elements will be added if not
        already present:

        * (0002,0001) *File Meta Information Version*
        * (0002,0012) *Implementation Class UID*
        * (0002,0013) *Implementation Version Name*

        and the following elements will be checked:

        * (0002,0002) *Media Storage SOP Class UID*
        * (0002,0003) *Media Storage SOP Instance UID*
        * (0002,0010) *Transfer Syntax UID*

    Raises
    ------
    ValueError
        If `enforce_standard` is ``True`` and any of the checked *File Meta
        Information* elements are missing from `file_meta`.
    ValueError
        If any non-Group 2 Elements are present in `file_meta`.
    """
    # Check that no non-Group 2 Elements are present
    for elem in file_meta.elements():
        if elem.tag.group != 0x0002:
            raise ValueError("Only File Meta Information Group (0002,eeee) "
                             "elements must be present in 'file_meta'.")

    if enforce_standard:
        if 'FileMetaInformationVersion' not in file_meta:
            file_meta.FileMetaInformationVersion = b'\x00\x01'

        if 'ImplementationClassUID' not in file_meta:
            file_meta.ImplementationClassUID = UID(PYDICOM_IMPLEMENTATION_UID)

        if 'ImplementationVersionName' not in file_meta:
            file_meta.ImplementationVersionName = (
                'PYDICOM ' + ".".join(str(x) for x in __version_info__))

        # Check that required File Meta Information elements are present
        missing = []
        for element in [0x0002, 0x0003, 0x0010]:
            if Tag(0x0002, element) not in file_meta:
                missing.append(Tag(0x0002, element))
        if missing:
            msg = ("Missing required File Meta Information elements from "
                   "'file_meta':\n")
            for tag in missing:
                msg += '\t{0} {1}\n'.format(tag, keyword_for_tag(tag))
            raise ValueError(msg[:-1])  # Remove final newline
```
### 26 - pydicom/dataset.py:

Start line: 162, End line: 178

```python
class PrivateBlock:

    def __delitem__(self, element_offset: int) -> None:
        """Delete the tag with the given `element_offset` from the dataset.

        Parameters
        ----------
        element_offset : int
            The lower 16 bits (e.g. 2 hex numbers) of the element tag
            to be deleted.

        Raises
        ------
        ValueError
            If `element_offset` is too large.
        KeyError
            If no data element exists at that offset.
        """
        del self.dataset[self.get_tag(element_offset)]
```
### 27 - pydicom/dataset.py:

Start line: 596, End line: 643

```python
class Dataset:

    def __delitem__(self, key: Union[slice, BaseTag, TagType]) -> None:
        """Intercept requests to delete an attribute by key.

        Examples
        --------
        Indexing using :class:`~pydicom.dataelem.DataElement` tag

        >>> ds = Dataset()
        >>> ds.CommandGroupLength = 100
        >>> ds.PatientName = 'CITIZEN^Jan'
        >>> del ds[0x00000000]
        >>> ds
        (0010, 0010) Patient's Name                      PN: 'CITIZEN^Jan'

        Slicing using :class:`~pydicom.dataelem.DataElement` tag

        >>> ds = Dataset()
        >>> ds.CommandGroupLength = 100
        >>> ds.SOPInstanceUID = '1.2.3'
        >>> ds.PatientName = 'CITIZEN^Jan'
        >>> del ds[:0x00100000]
        >>> ds
        (0010, 0010) Patient's Name                      PN: 'CITIZEN^Jan'

        Parameters
        ----------
        key
            The key for the attribute to be deleted. If a ``slice`` is used
            then the tags matching the slice conditions will be deleted.
        """
        # If passed a slice, delete the corresponding DataElements
        if isinstance(key, slice):
            for tag in self._slice_dataset(key.start, key.stop, key.step):
                del self._dict[tag]
                # invalidate private blocks in case a private creator is
                # deleted - will be re-created on next access
                if self._private_blocks and BaseTag(tag).is_private_creator:
                    self._private_blocks = {}
        elif isinstance(key, BaseTag):
            del self._dict[key]
            if self._private_blocks and key.is_private_creator:
                self._private_blocks = {}
        else:
            # If not a standard tag, than convert to Tag and try again
            tag = Tag(key)
            del self._dict[tag]
            if self._private_blocks and tag.is_private_creator:
                self._private_blocks = {}
```
### 29 - pydicom/dataset.py:

Start line: 2459, End line: 2505

```python
class Dataset:

    def to_json_dict(
        self,
        bulk_data_threshold: int = 1024,
        bulk_data_element_handler: Optional[Callable[[DataElement], str]] = None,  # noqa
        suppress_invalid_tags: bool = False,
    ) -> Dict[str, Any]:
        """Return a dictionary representation of the :class:`Dataset`
        conforming to the DICOM JSON Model as described in the DICOM
        Standard, Part 18, :dcm:`Annex F<part18/chapter_F.html>`.

        .. versionadded:: 1.4

        Parameters
        ----------
        bulk_data_threshold : int, optional
            Threshold for the length of a base64-encoded binary data element
            above which the element should be considered bulk data and the
            value provided as a URI rather than included inline (default:
            ``1024``). Ignored if no bulk data handler is given.
        bulk_data_element_handler : callable, optional
            Callable function that accepts a bulk data element and returns a
            JSON representation of the data element (dictionary including the
            "vr" key and either the "InlineBinary" or the "BulkDataURI" key).
        suppress_invalid_tags : bool, optional
            Flag to specify if errors while serializing tags should be logged
            and the tag dropped or if the error should be bubbled up.

        Returns
        -------
        dict
            :class:`Dataset` representation based on the DICOM JSON Model.
        """
        json_dataset = {}
        for key in self.keys():
            json_key = '{:08X}'.format(key)
            data_element = self[key]
            try:
                json_dataset[json_key] = data_element.to_json_dict(
                    bulk_data_element_handler=bulk_data_element_handler,
                    bulk_data_threshold=bulk_data_threshold
                )
            except Exception as exc:
                logger.error(f"Error while processing tag {json_key}")
                if not suppress_invalid_tags:
                    raise exc

        return json_dataset
```
