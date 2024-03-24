# pydicom__pydicom-1256

| **pydicom/pydicom** | `49a3da4a3d9c24d7e8427a25048a1c7d5c4f7724` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | - |
| **Avg pos** | - |
| **Min pos** | - |
| **Max pos** | - |
| **Top file pos** | 2 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/pydicom/jsonrep.py b/pydicom/jsonrep.py
--- a/pydicom/jsonrep.py
+++ b/pydicom/jsonrep.py
@@ -226,7 +226,8 @@ def get_sequence_item(self, value):
                     value_key = unique_value_keys[0]
                     elem = DataElement.from_json(
                         self.dataset_class, key, vr,
-                        val[value_key], value_key
+                        val[value_key], value_key,
+                        self.bulk_data_element_handler
                     )
                 ds.add(elem)
         return ds

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| pydicom/jsonrep.py | 229 | 231 | - | 2 | -


## Problem Statement

```
from_json does not correctly convert BulkDataURI's in SQ data elements
**Describe the bug**
When a DICOM object contains large data elements in SQ elements and is converted to JSON, those elements are correctly turned into BulkDataURI's. However, when the JSON is converted back to DICOM using from_json, the BulkDataURI's in SQ data elements are not converted back and warnings are thrown.

**Expected behavior**
The BulkDataURI's in SQ data elements get converted back correctly.

**Steps To Reproduce**
Take the `waveform_ecg.dcm` in the test data, convert it to JSON, and then convert the JSON to DICOM

**Your environment**
module       | version
------       | -------
platform     | macOS-10.15.7-x86_64-i386-64bit
Python       | 3.8.2 (v3.8.2:7b3ab5921f, Feb 24 2020, 17:52:18)  [Clang 6.0 (clang-600.0.57)]
pydicom      | 2.1.0
gdcm         | _module not found_
jpeg_ls      | _module not found_
numpy        | _module not found_
PIL          | _module not found_

The problem is in `jsonrep.py` at line 227. I plan on submitting a pull-request today for this.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- |
| 1 | 1 pydicom/dataelem.py | 9 | 42| 284 | 284 | 
| 2 | **2 pydicom/jsonrep.py** | 3 | 17| 143 | 427 | 
| 3 | 3 pydicom/dataset.py | 16 | 59| 358 | 785 | 
| 4 | 3 pydicom/dataelem.py | 283 | 370| 713 | 1498 | 
| 5 | **3 pydicom/jsonrep.py** | 53 | 107| 382 | 1880 | 
| 6 | 4 pydicom/_storage_sopclass_uids.py | 130 | 171| 720 | 2600 | 
| 7 | 5 pydicom/uid.py | 241 | 275| 749 | 3349 | 
| 8 | 5 pydicom/uid.py | 3 | 22| 146 | 3495 | 
| 9 | 6 pydicom/valuerep.py | 3 | 55| 502 | 3997 | 
| 10 | 7 pydicom/benchmarks/bench_handler_numpy.py | 6 | 46| 671 | 4668 | 
| 11 | 7 pydicom/_storage_sopclass_uids.py | 254 | 295| 715 | 5383 | 
| 12 | 8 pydicom/config.py | 233 | 322| 776 | 6159 | 
| 13 | 8 pydicom/dataset.py | 2223 | 2273| 400 | 6559 | 
| 14 | 9 pydicom/values.py | 772 | 833| 607 | 7166 | 
| 15 | 9 pydicom/dataelem.py | 228 | 281| 412 | 7578 | 
| 16 | 9 pydicom/_storage_sopclass_uids.py | 86 | 129| 732 | 8310 | 
| 17 | 9 pydicom/uid.py | 348 | 378| 286 | 8596 | 
| 18 | 9 pydicom/_storage_sopclass_uids.py | 46 | 85| 729 | 9325 | 
| 19 | 10 source/generate_cids/generate_concept_dicts.py | 0 | 74| 656 | 9981 | 
| 20 | 11 pydicom/fileset.py | 2750 | 2817| 947 | 10928 | 
| 21 | 11 pydicom/_storage_sopclass_uids.py | 296 | 342| 801 | 11729 | 
| 22 | 11 pydicom/values.py | 5 | 33| 241 | 11970 | 
| 23 | 12 pydicom/benchmarks/bench_handler_rle_decode.py | 3 | 35| 535 | 12505 | 
| 24 | 12 pydicom/dataelem.py | 372 | 414| 327 | 12832 | 
| 25 | 12 pydicom/_storage_sopclass_uids.py | 172 | 215| 749 | 13581 | 
| 26 | 13 pydicom/filewriter.py | 3 | 26| 193 | 13774 | 
| 27 | 14 source/generate_dict/generate_dicom_dict.py | 236 | 357| 1169 | 14943 | 
| 28 | 15 pydicom/charset.py | 3 | 65| 786 | 15729 | 
| 29 | 16 pydicom/data/download.py | 5 | 33| 139 | 15868 | 
| 30 | 17 source/generate_dict/generate_uid_dict.py | 185 | 233| 409 | 16277 | 
| 31 | 18 setup.py | 0 | 114| 796 | 17073 | 
| 32 | 18 pydicom/_storage_sopclass_uids.py | 216 | 253| 726 | 17799 | 
| 33 | 19 pydicom/filereader.py | 5 | 28| 251 | 18050 | 
| 34 | 19 pydicom/_storage_sopclass_uids.py | 0 | 45| 733 | 18783 | 
| 35 | 19 source/generate_cids/generate_concept_dicts.py | 334 | 445| 967 | 19750 | 
| 36 | 20 pydicom/pixel_data_handlers/gdcm_handler.py | 282 | 305| 251 | 20001 | 
| 37 | 20 pydicom/dataset.py | 2275 | 2311| 344 | 20345 | 
| 38 | **20 pydicom/jsonrep.py** | 109 | 157| 503 | 20848 | 
| 39 | 20 pydicom/valuerep.py | 502 | 517| 142 | 20990 | 
| 40 | 21 pydicom/__init__.py | 31 | 49| 128 | 21118 | 
| 41 | 21 pydicom/pixel_data_handlers/gdcm_handler.py | 214 | 280| 640 | 21758 | 
| 42 | 22 pydicom/pixel_data_handlers/pylibjpeg_handler.py | 235 | 267| 331 | 22089 | 
| 43 | 23 examples/metadata_processing/plot_anonymize.py | 0 | 92| 477 | 22566 | 
| 44 | 23 pydicom/config.py | 117 | 232| 858 | 23424 | 
| 45 | 24 pydicom/pixel_data_handlers/util.py | 1142 | 1156| 136 | 23560 | 
| 46 | 25 pydicom/pixel_data_handlers/numpy_handler.py | 300 | 368| 786 | 24346 | 
| 47 | 25 pydicom/dataset.py | 220 | 369| 1314 | 25660 | 
| 48 | 26 pydicom/util/leanread.py | 3 | 14| 170 | 25830 | 
| 49 | 27 pydicom/pixel_data_handlers/__init__.py | 1 | 7| 0 | 25830 | 
| 50 | 27 pydicom/config.py | 5 | 46| 252 | 26082 | 
| 51 | 28 pydicom/pixel_data_handlers/pillow_handler.py | 170 | 207| 390 | 26472 | 
| 52 | 29 pydicom/datadict.py | 4 | 27| 273 | 26745 | 
| 53 | 29 pydicom/dataset.py | 2313 | 2367| 458 | 27203 | 
| 54 | 29 pydicom/pixel_data_handlers/util.py | 1239 | 1308| 655 | 27858 | 
| 55 | 29 pydicom/uid.py | 276 | 346| 742 | 28600 | 
| 56 | 29 pydicom/charset.py | 66 | 96| 424 | 29024 | 
| 57 | 29 pydicom/dataelem.py | 716 | 806| 672 | 29696 | 
| 58 | 30 pydicom/benchmarks/bench_handler_rle_encode.py | 3 | 36| 379 | 30075 | 
| 59 | 30 pydicom/uid.py | 46 | 64| 262 | 30337 | 
| 60 | 31 pydicom/filebase.py | 143 | 165| 212 | 30549 | 
| 61 | 31 pydicom/filebase.py | 3 | 25| 156 | 30705 | 
| 62 | 31 pydicom/filereader.py | 853 | 892| 445 | 31150 | 
| 63 | 31 pydicom/filereader.py | 123 | 249| 1277 | 32427 | 
| 64 | 31 pydicom/dataset.py | 2651 | 2657| 105 | 32532 | 
| 65 | 31 pydicom/pixel_data_handlers/util.py | 807 | 869| 534 | 33066 | 
| 66 | 32 pydicom/pixel_data_handlers/jpeg_ls_handler.py | 6 | 69| 385 | 33451 | 
| 67 | 32 pydicom/pixel_data_handlers/pillow_handler.py | 5 | 88| 544 | 33995 | 
| 68 | 32 source/generate_dict/generate_dicom_dict.py | 0 | 62| 560 | 34555 | 
| 69 | 32 pydicom/pixel_data_handlers/util.py | 79 | 136| 758 | 35313 | 
| 70 | 32 pydicom/pixel_data_handlers/util.py | 236 | 267| 382 | 35695 | 
| 71 | 32 pydicom/filewriter.py | 179 | 215| 294 | 35989 | 
| 72 | 33 pydicom/pixel_data_handlers/rle_handler.py | 312 | 331| 260 | 36249 | 
| 73 | 34 examples/image_processing/plot_downsize_image.py | 0 | 43| 305 | 36554 | 
| 74 | 34 pydicom/pixel_data_handlers/gdcm_handler.py | 101 | 135| 250 | 36804 | 
| 75 | 35 pydicom/waveforms/__init__.py | 0 | 3| 0 | 36804 | 
| 76 | 35 pydicom/fileset.py | 3 | 71| 610 | 37414 | 
| 77 | 35 pydicom/charset.py | 624 | 700| 736 | 38150 | 
| 78 | 35 pydicom/dataset.py | 1472 | 1544| 601 | 38751 | 
| 79 | 36 pydicom/waveforms/numpy_handler.py | 66 | 137| 688 | 39439 | 
| 80 | **36 pydicom/jsonrep.py** | 159 | 186| 183 | 39622 | 
| 81 | 36 source/generate_cids/generate_concept_dicts.py | 176 | 173| 294 | 39916 | 
| 82 | 37 pydicom/compat.py | 4 | 25| 117 | 40033 | 
| 83 | 37 pydicom/filereader.py | 690 | 754| 792 | 40825 | 
| 84 | 38 pydicom/dicomdir.py | 115 | 140| 199 | 41024 | 
| 85 | 38 pydicom/filewriter.py | 139 | 176| 277 | 41301 | 
| 86 | 38 pydicom/util/leanread.py | 36 | 67| 254 | 41555 | 
| 87 | 38 pydicom/util/leanread.py | 93 | 184| 837 | 42392 | 
| 88 | 38 pydicom/dataelem.py | 697 | 713| 193 | 42585 | 
| 89 | 39 source/generate_dict/generate_private_dict.py | 96 | 115| 166 | 42751 | 
| 90 | 39 pydicom/valuerep.py | 897 | 912| 116 | 42867 | 
| 91 | 39 pydicom/pixel_data_handlers/pylibjpeg_handler.py | 175 | 234| 509 | 43376 | 
| 92 | 40 pydicom/_version.py | 0 | 14| 114 | 43490 | 
| 93 | 40 pydicom/config.py | 49 | 78| 200 | 43690 | 
| 94 | 41 pydicom/fileutil.py | 202 | 309| 891 | 44581 | 
| 95 | 41 pydicom/uid.py | 25 | 43| 190 | 44771 | 
| 96 | 42 examples/image_processing/plot_waveforms.py | 0 | 52| 402 | 45173 | 
| 97 | 42 pydicom/config.py | 323 | 387| 484 | 45657 | 
| 98 | 42 pydicom/fileset.py | 2307 | 2375| 492 | 46149 | 
| 99 | 42 pydicom/config.py | 390 | 405| 133 | 46282 | 
| 100 | 42 pydicom/dataset.py | 1565 | 1629| 569 | 46851 | 
| 101 | 42 pydicom/pixel_data_handlers/pylibjpeg_handler.py | 152 | 172| 185 | 47036 | 
| 102 | 43 pydicom/multival.py | 125 | 141| 125 | 47161 | 
| 103 | 43 pydicom/charset.py | 794 | 843| 417 | 47578 | 
| 104 | 44 pydicom/util/codify.py | 129 | 195| 613 | 48191 | 
| 105 | 44 pydicom/filereader.py | 113 | 121| 137 | 48328 | 
| 106 | 44 pydicom/filewriter.py | 939 | 1020| 799 | 49127 | 
| 107 | 44 pydicom/filewriter.py | 1021 | 1057| 344 | 49471 | 
| 108 | 44 pydicom/dataset.py | 1546 | 1563| 176 | 49647 | 


## Patch

```diff
diff --git a/pydicom/jsonrep.py b/pydicom/jsonrep.py
--- a/pydicom/jsonrep.py
+++ b/pydicom/jsonrep.py
@@ -226,7 +226,8 @@ def get_sequence_item(self, value):
                     value_key = unique_value_keys[0]
                     elem = DataElement.from_json(
                         self.dataset_class, key, vr,
-                        val[value_key], value_key
+                        val[value_key], value_key,
+                        self.bulk_data_element_handler
                     )
                 ds.add(elem)
         return ds

```

## Test Patch

```diff
diff --git a/pydicom/tests/test_json.py b/pydicom/tests/test_json.py
--- a/pydicom/tests/test_json.py
+++ b/pydicom/tests/test_json.py
@@ -354,3 +354,25 @@ def bulk_data_reader(tag, vr, value):
         ds = Dataset().from_json(json.dumps(json_data), bulk_data_reader)
 
         assert b'xyzzy' == ds[0x00091002].value
+
+    def test_bulk_data_reader_is_called_within_SQ(self):
+        def bulk_data_reader(_):
+            return b'xyzzy'
+
+        json_data = {
+            "003a0200": {
+                "vr": "SQ", 
+                "Value": [
+                    {
+                        "54001010": {
+                            "vr": "OW",
+                            "BulkDataURI": "https://a.dummy.url"
+                        }
+                    }
+                ]
+            }
+        }
+
+        ds = Dataset().from_json(json.dumps(json_data), bulk_data_reader)
+
+        assert b'xyzzy' == ds[0x003a0200].value[0][0x54001010].value

```


## Code snippets

### 1 - pydicom/dataelem.py:

Start line: 9, End line: 42

```python
import base64
import json
from typing import (
    Optional, Any, Optional, Tuple, Callable, Union, TYPE_CHECKING, Dict,
    TypeVar, Type, List, NamedTuple
)
import warnings

from pydicom import config  # don't import datetime_conversion directly
from pydicom.config import logger
from pydicom import config
from pydicom.datadict import (dictionary_has_tag, dictionary_description,
                              dictionary_keyword, dictionary_is_retired,
                              private_dictionary_description, dictionary_VR,
                              repeater_has_tag)
from pydicom.jsonrep import JsonDataElementConverter
from pydicom.multival import MultiValue
from pydicom.tag import Tag, BaseTag
from pydicom.uid import UID
from pydicom import jsonrep
import pydicom.valuerep  # don't import DS directly as can be changed by config
from pydicom.valuerep import PersonName

if config.have_numpy:
    import numpy

if TYPE_CHECKING:
    from pydicom.dataset import Dataset


BINARY_VR_VALUES = [
    'US', 'SS', 'UL', 'SL', 'OW', 'OB', 'OL', 'UN',
    'OB or OW', 'US or OW', 'US or SS or OW', 'FL', 'FD', 'OF', 'OD'
]
```
### 2 - pydicom/jsonrep.py:

Start line: 3, End line: 17

```python
import base64
from inspect import signature
import inspect
from typing import Callable, Optional, Union
import warnings

from pydicom.tag import BaseTag

# Order of keys is significant!
JSON_VALUE_KEYS = ('Value', 'BulkDataURI', 'InlineBinary',)

BINARY_VR_VALUES = ['OW', 'OB', 'OD', 'OF', 'OL', 'UN',
                    'OB or OW', 'US or OW', 'US or SS or OW']
VRs_TO_BE_FLOATS = ['DS', 'FL', 'FD', ]
VRs_TO_BE_INTS = ['IS', 'SL', 'SS', 'UL', 'US', 'US or SS']
```
### 3 - pydicom/dataset.py:

Start line: 16, End line: 59

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
from types import ModuleType, TracebackType
from typing import (
    Generator, TYPE_CHECKING, Optional, Tuple, Union, List, ItemsView,
    KeysView, Dict, ValuesView, Iterator, BinaryIO, AnyStr,
    Callable, TypeVar, Type, overload
)
import warnings
import weakref

if TYPE_CHECKING:
    try:
        import numpy as np
    except ImportError:
        pass

import pydicom  # for dcmwrite
import pydicom.charset
import pydicom.config
from pydicom import datadict, jsonrep, config
from pydicom._version import __version_info__
from pydicom.charset import default_encoding, convert_encodings
from pydicom.config import logger
from pydicom.datadict import (
    dictionary_VR, tag_for_keyword, keyword_for_tag, repeater_has_keyword
)
from pydicom.dataelem import DataElement, DataElement_from_raw, RawDataElement
from pydicom.fileutil import path_from_pathlike
from pydicom.pixel_data_handlers.util import (
    convert_color_space, reshape_pixel_array, get_image_pixel_ids
)
from pydicom.tag import Tag, BaseTag, tag_in_exception, TagType
from pydicom.uid import (ExplicitVRLittleEndian, ImplicitVRLittleEndian,
                         ExplicitVRBigEndian, PYDICOM_IMPLEMENTATION_UID)
from pydicom.waveforms import numpy_handler as wave_handler
```
### 4 - pydicom/dataelem.py:

Start line: 283, End line: 370

```python
class DataElement:

    def to_json_dict(
        self,
        bulk_data_element_handler: Optional[Callable[["DataElement"], str]],
        bulk_data_threshold: int
    ) -> Dict[str, object]:
        """Return a dictionary representation of the :class:`DataElement`
        conforming to the DICOM JSON Model as described in the DICOM
        Standard, Part 18, :dcm:`Annex F<part18/chaptr_F.html>`.

        .. versionadded:: 1.4

        Parameters
        ----------
        bulk_data_element_handler: callable or None
            Callable that accepts a bulk data element and returns the
            "BulkDataURI" for retrieving the value of the data element
            via DICOMweb WADO-RS
        bulk_data_threshold: int
            Size of base64 encoded data element above which a value will be
            provided in form of a "BulkDataURI" rather than "InlineBinary".
            Ignored if no bulk data handler is given.

        Returns
        -------
        dict
            Mapping representing a JSON encoded data element
        """
        json_element = {'vr': self.VR, }
        if self.VR in jsonrep.BINARY_VR_VALUES:
            if not self.is_empty:
                binary_value = self.value
                encoded_value = base64.b64encode(binary_value).decode('utf-8')
                if (
                    bulk_data_element_handler is not None
                    and len(encoded_value) > bulk_data_threshold
                ):
                    json_element['BulkDataURI'] = (
                        bulk_data_element_handler(self)
                    )
                else:
                    logger.info(
                        f"encode bulk data element '{self.name}' inline"
                    )
                    json_element['InlineBinary'] = encoded_value
        elif self.VR == 'SQ':
            # recursive call to get sequence item JSON dicts
            value = [
                ds.to_json(
                    bulk_data_element_handler=bulk_data_element_handler,
                    bulk_data_threshold=bulk_data_threshold,
                    dump_handler=lambda d: d
                )
                for ds in self.value
            ]
            json_element['Value'] = value
        elif self.VR == 'PN':
            if not self.is_empty:
                elem_value = []
                if self.VM > 1:
                    value = self.value
                else:
                    value = [self.value]
                for v in value:
                    comps = {'Alphabetic': v.components[0]}
                    if len(v.components) > 1:
                        comps['Ideographic'] = v.components[1]
                    if len(v.components) > 2:
                        comps['Phonetic'] = v.components[2]
                    elem_value.append(comps)
                json_element['Value'] = elem_value
        elif self.VR == 'AT':
            if not self.is_empty:
                value = self.value
                if self.VM == 1:
                    value = [value]
                json_element['Value'] = [format(v, '08X') for v in value]
        else:
            if not self.is_empty:
                if self.VM > 1:
                    value = self.value
                else:
                    value = [self.value]
                json_element['Value'] = [v for v in value]
        if hasattr(json_element, 'Value'):
            json_element['Value'] = jsonrep.convert_to_python_number(
                json_element['Value'], self.VR
            )
        return json_element
```
### 5 - pydicom/jsonrep.py:

Start line: 53, End line: 107

```python
class JsonDataElementConverter:
    """Handles conversion between JSON struct and :class:`DataElement`.

    .. versionadded:: 1.4
    """

    def __init__(
        self,
        dataset_class,
        tag,
        vr,
        value,
        value_key,
        bulk_data_uri_handler: Optional[
            Union[
                Callable[[BaseTag, str, str], object],
                Callable[[str], object]
            ]
        ] = None
    ):
        """Create a new converter instance.

        Parameters
        ----------
        dataset_class : dataset.Dataset derived class
            Class used to create sequence items.
        tag : BaseTag
            The data element tag or int.
        vr : str
            The data element value representation.
        value : list
            The data element's value(s).
        value_key : str or None
            Key of the data element that contains the value
            (options: ``{"Value", "InlineBinary", "BulkDataURI"}``)
        bulk_data_uri_handler: callable or None
            Callable function that accepts either the tag, vr and "BulkDataURI"
            or just the "BulkDataURI" of the JSON
            representation of a data element and returns the actual value of
            that data element (retrieved via DICOMweb WADO-RS)
        """
        self.dataset_class = dataset_class
        self.tag = tag
        self.vr = vr
        self.value = value
        self.value_key = value_key
        if (
            bulk_data_uri_handler and
            len(signature(bulk_data_uri_handler).parameters) == 1
        ):
            def wrapped_bulk_data_handler(tag, vr, value):
                return bulk_data_uri_handler(value)
            self.bulk_data_element_handler = wrapped_bulk_data_handler
        else:
            self.bulk_data_element_handler = bulk_data_uri_handler
```
### 6 - pydicom/_storage_sopclass_uids.py:

Start line: 130, End line: 171

```python
TomotherapeuticRadiationRecordStorage = UID(
    '1.2.840.10008.5.1.4.1.1.481.18')
CArmPhotonElectronRadiationRecordStorage = UID(
    '1.2.840.10008.5.1.4.1.1.481.19')
RTDoseStorage = UID(
    '1.2.840.10008.5.1.4.1.1.481.2')
RoboticRadiationRecordStorage = UID(
    '1.2.840.10008.5.1.4.1.1.481.20')
RTStructureSetStorage = UID(
    '1.2.840.10008.5.1.4.1.1.481.3')
RTBeamsTreatmentRecordStorage = UID(
    '1.2.840.10008.5.1.4.1.1.481.4')
RTPlanStorage = UID(
    '1.2.840.10008.5.1.4.1.1.481.5')
RTBrachyTreatmentRecordStorage = UID(
    '1.2.840.10008.5.1.4.1.1.481.6')
RTTreatmentSummaryRecordStorage = UID(
    '1.2.840.10008.5.1.4.1.1.481.7')
RTIonPlanStorage = UID(
    '1.2.840.10008.5.1.4.1.1.481.8')
RTIonBeamsTreatmentRecordStorage = UID(
    '1.2.840.10008.5.1.4.1.1.481.9')
DICOSCTImageStorage = UID(
    '1.2.840.10008.5.1.4.1.1.501.1')
DICOSDigitalXRayImageStorageForPresentation = UID(
    '1.2.840.10008.5.1.4.1.1.501.2.1')
DICOSDigitalXRayImageStorageForProcessing = UID(
    '1.2.840.10008.5.1.4.1.1.501.2.2')
DICOSThreatDetectionReportStorage = UID(
    '1.2.840.10008.5.1.4.1.1.501.3')
DICOS2DAITStorage = UID(
    '1.2.840.10008.5.1.4.1.1.501.4')
DICOS3DAITStorage = UID(
    '1.2.840.10008.5.1.4.1.1.501.5')
DICOSQuadrupoleResonanceStorage = UID(
    '1.2.840.10008.5.1.4.1.1.501.6')
UltrasoundImageStorage = UID(
    '1.2.840.10008.5.1.4.1.1.6.1')
EnhancedUSVolumeStorage = UID(
    '1.2.840.10008.5.1.4.1.1.6.2')
EddyCurrentImageStorage = UID(
    '1.2.840.10008.5.1.4.1.1.601.1')
```
### 7 - pydicom/uid.py:

Start line: 241, End line: 275

```python
# Pre-defined Transfer Syntax UIDs (for convenience)
ImplicitVRLittleEndian = UID('1.2.840.10008.1.2')
"""1.2.840.10008.1.2"""
ExplicitVRLittleEndian = UID('1.2.840.10008.1.2.1')
"""1.2.840.10008.1.2.1"""
DeflatedExplicitVRLittleEndian = UID('1.2.840.10008.1.2.1.99')
"""1.2.840.10008.1.2.1.99"""
ExplicitVRBigEndian = UID('1.2.840.10008.1.2.2')
"""1.2.840.10008.1.2.2"""
JPEGBaseline8Bit = UID('1.2.840.10008.1.2.4.50')
"""1.2.840.10008.1.2.4.50"""
JPEGExtended12Bit = UID('1.2.840.10008.1.2.4.51')
"""1.2.840.10008.1.2.4.51"""
JPEGLosslessP14 = UID('1.2.840.10008.1.2.4.57')  # needs to be updated
"""1.2.840.10008.1.2.4.57"""
JPEGLosslessSV1 = UID('1.2.840.10008.1.2.4.70')  # Old JPEGLossless
"""1.2.840.10008.1.2.4.70"""
JPEGLSLossless = UID('1.2.840.10008.1.2.4.80')
"""1.2.840.10008.1.2.4.80"""
JPEGLSNearLossless = UID('1.2.840.10008.1.2.4.81')
"""1.2.840.10008.1.2.4.81"""
JPEG2000Lossless = UID('1.2.840.10008.1.2.4.90')
"""1.2.840.10008.1.2.4.90"""
JPEG2000 = UID('1.2.840.10008.1.2.4.91')
"""1.2.840.10008.1.2.4.91"""
JPEG2000MCLossless = UID('1.2.840.10008.1.2.4.92')
"""1.2.840.10008.1.2.4.92"""
JPEG2000MC = UID('1.2.840.10008.1.2.4.93')
"""1.2.840.10008.1.2.4.93"""
MPEG2MPML = UID('1.2.840.10008.1.2.4.100')
"""1.2.840.10008.1.2.4.100"""
MPEG2MPHL = UID('1.2.840.10008.1.2.4.101')
"""1.2.840.10008.1.2.4.101"""
MPEG4HP41 = UID('1.2.840.10008.1.2.4.102')
"""1.2.840.10008.1.2.4.102"""
```
### 8 - pydicom/uid.py:

Start line: 3, End line: 22

```python
import os
import uuid
import random
import hashlib
import re
import sys
from typing import List, Optional, TypeVar, Type, Union
import warnings

from pydicom._uid_dict import UID_dictionary


_deprecations = {
    "JPEGBaseline": "JPEGBaseline8Bit",
    "JPEGExtended": "JPEGExtended12Bit",
    "JPEGLossless": "JPEGLosslessSV1",
    "JPEGLSLossy": "JPEGLSNearLossless",
    "JPEG2000MultiComponentLossless": "JPEG2000MCLossless",
    "JPEG2000MultiComponent": "JPEG2000MC",
}
```
### 9 - pydicom/valuerep.py:

Start line: 3, End line: 55

```python
import datetime
from decimal import Decimal
import platform
import re
import sys
from typing import (
    TypeVar, Type, Tuple, Optional, List, Dict, Union, Any, Generator, AnyStr,
    Callable, Iterator, overload
)
from typing import Sequence as SequenceType
import warnings

# don't import datetime_conversion directly
from pydicom import config
from pydicom.multival import MultiValue
from pydicom.uid import UID


# Types
_T = TypeVar('_T')
_DA = TypeVar("_DA", bound="DA")
_DT = TypeVar("_DT", bound="DT")
_TM = TypeVar("_TM", bound="TM")
_IS = TypeVar("_IS", bound="IS")
_DSfloat = TypeVar("_DSfloat", bound="DSfloat")
_DSdecimal = TypeVar("_DSdecimal", bound="DSdecimal")
_PersonName = TypeVar("_PersonName", bound="PersonName")

# can't import from charset or get circular import
default_encoding = "iso8859"

# For reading/writing data elements,
# these ones have longer explicit VR format
# Taken from PS3.5 Section 7.1.2
extra_length_VRs = ('OB', 'OD', 'OF', 'OL', 'OW', 'SQ', 'UC', 'UN', 'UR', 'UT')

# VRs that can be affected by character repertoire
# in (0008,0005) Specific Character Set
# See PS-3.5 (2011), section 6.1.2 Graphic Characters
# and PN, but it is handled separately.
text_VRs: Tuple[str, ...] = ('SH', 'LO', 'ST', 'LT', 'UC', 'UT')

# Delimiters for text strings and person name that reset the encoding.
# See PS3.5, Section 6.1.2.5.3
# Note: We use character codes for Python 3
# because those are the types yielded if iterating over a byte string.

# Characters/Character codes for text VR delimiters: LF, CR, TAB, FF
TEXT_VR_DELIMS = {0x0d, 0x0a, 0x09, 0x0c}

# Character/Character code for PN delimiter: name part separator '^'
# (the component separator '=' is handled separately)
PN_DELIMS = {0xe5}
```
### 10 - pydicom/benchmarks/bench_handler_numpy.py:

Start line: 6, End line: 46

```python
from platform import python_implementation
from tempfile import TemporaryFile

import numpy as np

from pydicom import dcmread
from pydicom.data import get_testdata_file
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.pixel_data_handlers.numpy_handler import get_pixeldata
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

# 1/1, 1 sample/pixel, 1 frame
EXPL_1_1_1F = get_testdata_file("liver_1frame.dcm")
# 1/1, 1 sample/pixel, 3 frame
EXPL_1_1_3F = get_testdata_file("liver.dcm")
# 8/8, 1 sample/pixel, 1 frame
EXPL_8_1_1F = get_testdata_file("OBXXXX1A.dcm")
# 8/8, 1 sample/pixel, 2 frame
EXPL_8_1_2F = get_testdata_file("OBXXXX1A_2frame.dcm")
# 8/8, 3 sample/pixel, 1 frame
EXPL_8_3_1F = get_testdata_file("SC_rgb.dcm")
# 8/8, 3 sample/pixel, 1 frame, YBR_FULL_422
EXPL_8_3_1F_YBR422 = get_testdata_file('SC_ybr_full_422_uncompressed.dcm')
# 8/8, 3 sample/pixel, 2 frame
EXPL_8_3_2F = get_testdata_file("SC_rgb_2frame.dcm")
# 16/16, 1 sample/pixel, 1 frame
EXPL_16_1_1F = get_testdata_file("MR_small.dcm")
# 16/12, 1 sample/pixel, 10 frame
EXPL_16_1_10F = get_testdata_file("emri_small.dcm")
# 16/16, 3 sample/pixel, 1 frame
EXPL_16_3_1F = get_testdata_file("SC_rgb_16bit.dcm")
# 16/16, 3 sample/pixel, 2 frame
EXPL_16_3_2F = get_testdata_file("SC_rgb_16bit_2frame.dcm")
# 32/32, 1 sample/pixel, 1 frame
IMPL_32_1_1F = get_testdata_file("rtdose_1frame.dcm")
# 32/32, 1 sample/pixel, 15 frame
IMPL_32_1_15F = get_testdata_file("rtdose.dcm")
# 32/32, 3 sample/pixel, 1 frame
EXPL_32_3_1F = get_testdata_file("SC_rgb_32bit.dcm")
# 32/32, 3 sample/pixel, 2 frame
EXPL_32_3_2F = get_testdata_file("SC_rgb_32bit_2frame.dcm")
```
### 38 - pydicom/jsonrep.py:

Start line: 109, End line: 157

```python
class JsonDataElementConverter:

    def get_element_values(self):
        """Return a the data element value or list of values.

        Returns
        -------
        str or bytes or int or float or dataset_class
        or PersonName or list of any of these types
            The value or value list of the newly created data element.
        """
        from pydicom.dataelem import empty_value_for_VR
        if self.value_key == 'Value':
            if not isinstance(self.value, list):
                fmt = '"{}" of data element "{}" must be a list.'
                raise TypeError(fmt.format(self.value_key, self.tag))
            if not self.value:
                return empty_value_for_VR(self.vr)
            element_value = [self.get_regular_element_value(v)
                             for v in self.value]
            if len(element_value) == 1 and self.vr != 'SQ':
                element_value = element_value[0]
            return convert_to_python_number(element_value, self.vr)

        # The value for "InlineBinary" shall be encoded as a base64 encoded
        # string, as shown in PS3.18, Table F.3.1-1, but the example in
        # PS3.18, Annex F.4 shows the string enclosed in a list.
        # We support both variants, as the standard is ambiguous here,
        # and do the same for "BulkDataURI".
        value = self.value
        if isinstance(value, list):
            value = value[0]

        if self.value_key == 'InlineBinary':
            if not isinstance(value, (str, bytes)):
                fmt = '"{}" of data element "{}" must be a bytes-like object.'
                raise TypeError(fmt.format(self.value_key, self.tag))
            return base64.b64decode(value)

        if self.value_key == 'BulkDataURI':
            if not isinstance(value, str):
                fmt = '"{}" of data element "{}" must be a string.'
                raise TypeError(fmt.format(self.value_key, self.tag))
            if self.bulk_data_element_handler is None:
                warnings.warn(
                    'no bulk data URI handler provided for retrieval '
                    'of value of data element "{}"'.format(self.tag)
                )
                return empty_value_for_VR(self.vr, raw=True)
            return self.bulk_data_element_handler(self.tag, self.vr, value)
        return empty_value_for_VR(self.vr)
```
### 80 - pydicom/jsonrep.py:

Start line: 159, End line: 186

```python
class JsonDataElementConverter:

    def get_regular_element_value(self, value):
        """Return a the data element value created from a json "Value" entry.

        Parameters
        ----------
        value : str or int or float or dict
            The data element's value from the json entry.

        Returns
        -------
        dataset_class or PersonName
        or str or int or float
            A single value of the corresponding :class:`DataElement`.
        """
        if self.vr == 'SQ':
            return self.get_sequence_item(value)

        if self.vr == 'PN':
            return self.get_pn_element_value(value)

        if self.vr == 'AT':
            try:
                return int(value, 16)
            except ValueError:
                warnings.warn('Invalid value "{}" for AT element - '
                              'ignoring it'.format(value))
            return
        return value
```
