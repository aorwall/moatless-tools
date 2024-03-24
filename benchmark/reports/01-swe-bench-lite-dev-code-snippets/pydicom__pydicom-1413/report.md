# pydicom__pydicom-1413

| **pydicom/pydicom** | `f909c76e31f759246cec3708dadd173c5d6e84b1` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | - |
| **Avg pos** | - |
| **Min pos** | - |
| **Max pos** | - |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/pydicom/dataelem.py b/pydicom/dataelem.py
--- a/pydicom/dataelem.py
+++ b/pydicom/dataelem.py
@@ -433,13 +433,24 @@ def value(self) -> Any:
     @value.setter
     def value(self, val: Any) -> None:
         """Convert (if necessary) and set the value of the element."""
+        # Ignore backslash characters in these VRs, based on:
+        # * Which str VRs can have backslashes in Part 5, Section 6.2
+        # * All byte VRs
+        exclusions = [
+            'LT', 'OB', 'OD', 'OF', 'OL', 'OV', 'OW', 'ST', 'UN', 'UT',
+            'OB/OW', 'OW/OB', 'OB or OW', 'OW or OB',
+            # Probably not needed
+            'AT', 'FD', 'FL', 'SQ', 'SS', 'SL', 'UL',
+        ]
+
         # Check if is a string with multiple values separated by '\'
         # If so, turn them into a list of separate strings
         #  Last condition covers 'US or SS' etc
-        if isinstance(val, (str, bytes)) and self.VR not in \
-                ['UT', 'ST', 'LT', 'FL', 'FD', 'AT', 'OB', 'OW', 'OF', 'SL',
-                 'SQ', 'SS', 'UL', 'OB/OW', 'OW/OB', 'OB or OW',
-                 'OW or OB', 'UN'] and 'US' not in self.VR:
+        if (
+            isinstance(val, (str, bytes))
+            and self.VR not in exclusions
+            and 'US' not in self.VR
+        ):
             try:
                 if _backslash_str in val:
                     val = cast(str, val).split(_backslash_str)

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| pydicom/dataelem.py | 439 | 444 | - | 1 | -


## Problem Statement

```
Error : a bytes-like object is required, not 'MultiValue'
Hello,

I am getting following error while updating the tag LongTrianglePointIndexList (0066,0040),
**TypeError: a bytes-like object is required, not 'MultiValue'**

I noticed that the error  gets produced only when the VR is given as "OL" , works fine with "OB", "OF" etc.

sample code (assume 'lineSeq' is the dicom dataset sequence):
\`\`\`python
import pydicom
import array
data=list(range(1,10))
data=array.array('H', indexData).tostring()  # to convert to unsigned short
lineSeq.add_new(0x00660040, 'OL', data)   
ds.save_as("mydicom")
\`\`\`
outcome: **TypeError: a bytes-like object is required, not 'MultiValue'**

using version - 2.0.0.0

Any help is appreciated.

Thank you

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- |
| 1 | **1 pydicom/dataelem.py** | 9 | 42| 309 | 309 | 
| 2 | 2 pydicom/values.py | 5 | 34| 258 | 567 | 
| 3 | 2 pydicom/values.py | 782 | 834| 533 | 1100 | 
| 4 | 3 pydicom/valuerep.py | 3 | 43| 397 | 1497 | 
| 5 | 4 pydicom/multival.py | 115 | 139| 305 | 1802 | 
| 6 | 5 pydicom/tag.py | 34 | 38| 58 | 1860 | 
| 7 | 6 pydicom/filewriter.py | 3 | 31| 254 | 2114 | 
| 8 | 7 pydicom/jsonrep.py | 3 | 17| 147 | 2261 | 
| 9 | 8 pydicom/config.py | 48 | 77| 200 | 2461 | 
| 10 | 9 pydicom/_storage_sopclass_uids.py | 172 | 215| 742 | 3203 | 
| 11 | 10 pydicom/dataset.py | 16 | 61| 377 | 3580 | 
| 12 | 11 pydicom/charset.py | 3 | 68| 817 | 4397 | 
| 13 | 12 pydicom/filereader.py | 5 | 36| 268 | 4665 | 
| 14 | 12 pydicom/_storage_sopclass_uids.py | 130 | 171| 726 | 5391 | 
| 15 | 12 pydicom/filewriter.py | 272 | 304| 284 | 5675 | 
| 16 | 13 pydicom/util/codify.py | 17 | 46| 256 | 5931 | 
| 17 | 13 pydicom/filewriter.py | 41 | 153| 1418 | 7349 | 
| 18 | 14 pydicom/errors.py | 4 | 23| 145 | 7494 | 
| 19 | 15 pydicom/fileset.py | 2767 | 2834| 947 | 8441 | 
| 20 | 16 pydicom/filebase.py | 143 | 165| 240 | 8681 | 
| 21 | 16 pydicom/fileset.py | 3 | 72| 616 | 9297 | 
| 22 | 16 pydicom/filebase.py | 3 | 25| 160 | 9457 | 
| 23 | 16 pydicom/multival.py | 16 | 13| 847 | 10304 | 
| 24 | 16 pydicom/dataset.py | 218 | 220| 43 | 10347 | 
| 25 | **16 pydicom/dataelem.py** | 516 | 552| 479 | 10826 | 
| 26 | 16 pydicom/config.py | 116 | 243| 959 | 11785 | 
| 27 | 17 pydicom/fileutil.py | 176 | 215| 449 | 12234 | 
| 28 | **17 pydicom/dataelem.py** | 495 | 514| 150 | 12384 | 
| 29 | 17 pydicom/_storage_sopclass_uids.py | 216 | 253| 729 | 13113 | 
| 30 | 18 pydicom/pixel_data_handlers/util.py | 448 | 456| 153 | 13266 | 
| 31 | 18 pydicom/values.py | 408 | 442| 233 | 13499 | 
| 32 | 18 pydicom/filereader.py | 130 | 282| 1515 | 15014 | 
| 33 | 18 pydicom/_storage_sopclass_uids.py | 254 | 295| 716 | 15730 | 
| 34 | 18 pydicom/values.py | 472 | 496| 176 | 15906 | 
| 35 | 19 pydicom/__init__.py | 29 | 54| 160 | 16066 | 
| 36 | 19 pydicom/filereader.py | 439 | 450| 237 | 16303 | 
| 37 | 19 pydicom/filewriter.py | 198 | 234| 309 | 16612 | 
| 38 | 19 pydicom/filewriter.py | 34 | 40| 80 | 16692 | 
| 39 | **19 pydicom/dataelem.py** | 729 | 758| 235 | 16927 | 
| 40 | 19 pydicom/dataset.py | 223 | 364| 1237 | 18164 | 
| 41 | 19 pydicom/_storage_sopclass_uids.py | 336 | 350| 205 | 18369 | 
| 42 | 19 pydicom/filewriter.py | 475 | 472| 212 | 18581 | 
| 43 | 19 pydicom/_storage_sopclass_uids.py | 296 | 335| 740 | 19321 | 
| 44 | 19 pydicom/_storage_sopclass_uids.py | 86 | 129| 729 | 20050 | 
| 45 | 19 pydicom/util/codify.py | 413 | 450| 290 | 20340 | 
| 46 | 20 pydicom/benchmarks/bench_rle_decode.py | 3 | 35| 535 | 20875 | 
| 47 | 20 pydicom/valuerep.py | 1513 | 1528| 116 | 20991 | 
| 48 | **20 pydicom/dataelem.py** | 110 | 107| 535 | 21526 | 
| 49 | 20 pydicom/dataset.py | 1883 | 1916| 338 | 21864 | 
| 50 | 20 pydicom/filewriter.py | 156 | 195| 298 | 22162 | 
| 51 | 20 pydicom/filereader.py | 791 | 853| 827 | 22989 | 
| 52 | 20 pydicom/filewriter.py | 440 | 437| 226 | 23215 | 
| 53 | 20 pydicom/filewriter.py | 1121 | 1183| 621 | 23836 | 
| 54 | 20 pydicom/filewriter.py | 578 | 596| 230 | 24066 | 
| 55 | 20 pydicom/fileutil.py | 218 | 312| 802 | 24868 | 
| 56 | 20 pydicom/filewriter.py | 405 | 402| 196 | 25064 | 
| 57 | 20 pydicom/util/codify.py | 139 | 221| 684 | 25748 | 
| 58 | 20 pydicom/values.py | 37 | 59| 178 | 25926 | 
| 59 | 20 pydicom/pixel_data_handlers/util.py | 1174 | 1199| 236 | 26162 | 
| 60 | 20 pydicom/pixel_data_handlers/util.py | 3 | 20| 115 | 26277 | 
| 61 | 20 pydicom/tag.py | 144 | 226| 733 | 27010 | 
| 62 | 21 pydicom/encoders/base.py | 3 | 23| 171 | 27181 | 
| 63 | 21 pydicom/config.py | 244 | 375| 1150 | 28331 | 
| 64 | 21 pydicom/pixel_data_handlers/util.py | 246 | 279| 424 | 28755 | 
| 65 | 21 pydicom/filereader.py | 855 | 874| 296 | 29051 | 
| 66 | 22 pydicom/dicomio.py | 0 | 11| 0 | 29051 | 
| 67 | 23 pydicom/pixel_data_handlers/rle_handler.py | 393 | 453| 477 | 29528 | 
| 68 | 23 pydicom/filereader.py | 109 | 128| 323 | 29851 | 
| 69 | 24 pydicom/_version.py | 0 | 14| 117 | 29968 | 
| 70 | 24 pydicom/values.py | 172 | 240| 629 | 30597 | 
| 71 | 25 pydicom/sequence.py | 148 | 166| 188 | 30785 | 
| 72 | 25 pydicom/pixel_data_handlers/util.py | 1282 | 1351| 662 | 31447 | 
| 73 | 25 pydicom/dataset.py | 2829 | 2860| 217 | 31664 | 
| 74 | 26 pydicom/cli/show.py | 37 | 62| 159 | 31823 | 
| 75 | 26 pydicom/values.py | 94 | 135| 265 | 32088 | 
| 76 | 26 pydicom/valuerep.py | 344 | 357| 145 | 32233 | 
| 77 | 26 pydicom/fileset.py | 2837 | 2844| 70 | 32303 | 
| 78 | 27 pydicom/env_info.py | 17 | 14| 124 | 32427 | 
| 79 | 27 pydicom/dataset.py | 922 | 948| 262 | 32689 | 
| 80 | 27 pydicom/charset.py | 69 | 99| 424 | 33113 | 
| 81 | 27 pydicom/util/codify.py | 77 | 136| 416 | 33529 | 
| 82 | 27 pydicom/pixel_data_handlers/rle_handler.py | 325 | 350| 348 | 33877 | 
| 83 | 27 pydicom/filereader.py | 966 | 1010| 467 | 34344 | 
| 84 | 27 pydicom/jsonrep.py | 193 | 237| 291 | 34635 | 
| 85 | 27 pydicom/filebase.py | 168 | 221| 462 | 35097 | 
| 86 | 27 pydicom/tag.py | 229 | 248| 158 | 35255 | 
| 87 | 27 pydicom/cli/show.py | 65 | 132| 608 | 35863 | 
| 88 | **27 pydicom/dataelem.py** | 599 | 617| 179 | 36042 | 
| 89 | 27 pydicom/fileset.py | 2847 | 2868| 158 | 36200 | 
| 90 | 27 pydicom/filereader.py | 489 | 549| 512 | 36712 | 
| 91 | 27 pydicom/filewriter.py | 713 | 710| 185 | 36897 | 
| 92 | 28 pydicom/cli/main.py | 9 | 60| 405 | 37302 | 
| 93 | 29 pydicom/benchmarks/bench_rle_encode.py | 3 | 20| 281 | 37583 | 
| 94 | 29 pydicom/fileset.py | 2324 | 2392| 492 | 38075 | 
| 95 | 29 pydicom/valuerep.py | 594 | 621| 242 | 38317 | 
| 96 | 29 pydicom/valuerep.py | 574 | 592| 130 | 38447 | 
| 97 | **29 pydicom/dataelem.py** | 710 | 726| 195 | 38642 | 
| 98 | 30 pydicom/waveforms/numpy_handler.py | 143 | 220| 753 | 39395 | 
| 99 | 30 pydicom/sequence.py | 130 | 146| 150 | 39545 | 
| 100 | 30 pydicom/filewriter.py | 366 | 392| 223 | 39768 | 
| 101 | 30 pydicom/fileutil.py | 314 | 327| 164 | 39932 | 
| 102 | 30 pydicom/jsonrep.py | 114 | 162| 503 | 40435 | 
| 103 | 30 pydicom/config.py | 415 | 430| 133 | 40568 | 
| 104 | 30 pydicom/encoders/base.py | 185 | 201| 179 | 40747 | 
| 105 | 30 pydicom/filereader.py | 730 | 727| 547 | 41294 | 
| 106 | **30 pydicom/dataelem.py** | 761 | 861| 810 | 42104 | 
| 107 | 30 pydicom/pixel_data_handlers/util.py | 846 | 908| 534 | 42638 | 
| 108 | 30 pydicom/filereader.py | 285 | 353| 685 | 43323 | 
| 109 | 30 pydicom/dataset.py | 2893 | 2899| 105 | 43428 | 
| 110 | 30 pydicom/filewriter.py | 307 | 323| 130 | 43558 | 
| 111 | 30 pydicom/values.py | 252 | 249| 330 | 43888 | 
| 112 | 30 pydicom/values.py | 346 | 405| 479 | 44367 | 
| 113 | 30 pydicom/encoders/base.py | 610 | 643| 320 | 44687 | 
| 114 | 30 pydicom/valuerep.py | 1262 | 1276| 195 | 44882 | 
| 115 | 30 pydicom/valuerep.py | 1241 | 1260| 225 | 45107 | 
| 116 | 31 pydicom/encoders/__init__.py | 0 | 1| 0 | 45107 | 
| 117 | 31 pydicom/valuerep.py | 732 | 737| 363 | 45470 | 
| 118 | 31 pydicom/encoders/base.py | 864 | 881| 153 | 45623 | 
| 119 | 31 pydicom/values.py | 690 | 779| 710 | 46333 | 
| 120 | 31 pydicom/values.py | 499 | 523| 174 | 46507 | 
| 121 | 31 pydicom/valuerep.py | 269 | 279| 145 | 46652 | 
| 122 | 31 pydicom/config.py | 5 | 45| 254 | 46906 | 
| 123 | 31 pydicom/valuerep.py | 467 | 571| 826 | 47732 | 
| 124 | 31 pydicom/tag.py | 41 | 141| 817 | 48549 | 
| 125 | **31 pydicom/dataelem.py** | 432 | 430| 255 | 48804 | 
| 126 | 31 pydicom/fileutil.py | 16 | 13| 337 | 49141 | 
| 127 | 31 pydicom/encoders/base.py | 203 | 225| 172 | 49313 | 


### Hint

```
Also tried following code to get the byte string, but same error.
1. data=array.array('L', indexData).tostring()  # to convert to long -> same error
2. data=array.array('Q', indexData).tostring()  # to convert to long long -> same error


O* VRs should be `bytes`. Use `array.tobytes()` instead of `tostring()`?

Also, in the future if have an issue it's much more helpful if you post the full traceback rather than the error since we can look at it to figure out where in the code the exception is occurring.

It would also help if you posted the version of Python you're using. 

This works fine for me with Python 3.9 and pydicom 2.1.2:
\`\`\`python
from pydicom import Dataset
import array

arr = array.array('H', range(10))
ds = Dataset()
ds.is_little_endian = True
ds.is_implicit_VR = False
ds.LongTrianglePointIndexList = arr.tobytes()
print(ds["LongTrianglePointIndexList"].VR)  # 'OL'
ds.save_as('temp.dcm')
\`\`\`
This also works fine:
\`\`\`python
ds = Dataset()
ds.add_new(0x00660040, 'OL', arr.tobytes())
\`\`\`
Thank you for the answer.
Unfortunately the error still persists with above code.
Please find the attached detailed error.
[error.txt](https://github.com/pydicom/pydicom/files/6661451/error.txt)

One more information is that the 'ds' is actually read from a file in the disk (ds=pydicom.read_file(filename)). 
and this byte array is stored under the following sequence
ds[0x0066,0x0002][0][0x0066,0x0013][0][0x0066,0x0028][0][0x0066,0x0040] = arr.tobytes()

pydicom - 2.0.0.0
python - 3.6.4

Thank you.
Could you post a minimal code sample that reproduces the issue please?

If you're using something like this:
`ds[0x0066,0x0002][0][0x0066,0x0013][0][0x0066,0x0028][0][0x0066,0x0040] = arr.tobytes()`

Then you're missing the `.value` assignment:
`ds[0x0066,0x0002][0][0x0066,0x0013][0][0x0066,0x0028][0][0x0066,0x0040].value = arr.tobytes()`
Hello,
above code line I just mentioned to give an idea where the actual data is stored (tree level).

Please find the actual code used below,
\`\`\`python
import pydicom
from pydicom.sequence import Sequence
from pydicom.dataelem import DataElement
from pydicom.dataset import Dataset

ds = pydicom.read_file(filename)
surfaceSeq= ds[0x0066,0x0002]

#// read existing sequence items in the dataset
seqlist=[]
for n in surfaceSeq:
    seqlist.append(n)

newDs = Dataset()
 
surfaceMeshPrimitiveSq = Dataset()
lineSeq = Dataset()
indexData = list(range(1,100))
indexData = array.array('H', indexData)
indexData = indexData.tobytes()
lineSeq.add_new(0x00660040, 'OL', indexData) 
surfaceMeshPrimitiveSq.add_new(0x00660028, 'SQ', [lineSeq])
newDs.add_new(0x00660013, 'SQ', [surfaceMeshPrimitiveSq])

#add the new sequnce item to the list
seqlist.append(newDs)
ds[0x0066,0x0002] = DataElement(0x00660002,"SQ",seqlist)
ds.save_as(filename)
\`\`\`
OK, I can reproduce with:
\`\`\`python

import array

from pydicom import Dataset
from pydicom.uid import ExplicitVRLittleEndian

ds = Dataset()
ds.file_meta = Dataset()
ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

b = array.array('H', range(100)).tobytes()

ds.LongPrimitivePointIndexList = b
ds.save_as('1421.dcm')
\`\`\`
And `print(ds)` gives:
\`\`\`
(0066, 0040) Long Primitive Point Index List     OL: [b'\x00\x00\x01\x00\x02\x00\x03\x00\x04\x00\x05\x00\x06\x00\x07\x00\x08\x00\t\x00\n\x00\x0b\x00\x0c\x00\r\x00\x0e\x00\x0f\x00\x10\x00\x11\x00\x12\x00\x13\x00\x14\x00\x15\x00\x16\x00\x17\x00\x18\x00\x19\x00\x1a\x00\x1b\x00\x1c\x00\x1d\x00\x1e\x00\x1f\x00 \x00!\x00"\x00#\x00$\x00%\x00&\x00\'\x00(\x00)\x00*\x00+\x00,\x00-\x00.\x00/\x000\x001\x002\x003\x004\x005\x006\x007\x008\x009\x00:\x00;\x00<\x00=\x00>\x00?\x00@\x00A\x00B\x00C\x00D\x00E\x00F\x00G\x00H\x00I\x00J\x00K\x00L\x00M\x00N\x00O\x00P\x00Q\x00R\x00S\x00T\x00U\x00V\x00W\x00X\x00Y\x00Z\x00[\x00', b'\x00]\x00^\x00_\x00`\x00a\x00b\x00c\x00']
\`\`\`
I think this is because the byte value is hitting the hex for the backslash character during assignment. Ouch, that's kinda nasty.
```

## Patch

```diff
diff --git a/pydicom/dataelem.py b/pydicom/dataelem.py
--- a/pydicom/dataelem.py
+++ b/pydicom/dataelem.py
@@ -433,13 +433,24 @@ def value(self) -> Any:
     @value.setter
     def value(self, val: Any) -> None:
         """Convert (if necessary) and set the value of the element."""
+        # Ignore backslash characters in these VRs, based on:
+        # * Which str VRs can have backslashes in Part 5, Section 6.2
+        # * All byte VRs
+        exclusions = [
+            'LT', 'OB', 'OD', 'OF', 'OL', 'OV', 'OW', 'ST', 'UN', 'UT',
+            'OB/OW', 'OW/OB', 'OB or OW', 'OW or OB',
+            # Probably not needed
+            'AT', 'FD', 'FL', 'SQ', 'SS', 'SL', 'UL',
+        ]
+
         # Check if is a string with multiple values separated by '\'
         # If so, turn them into a list of separate strings
         #  Last condition covers 'US or SS' etc
-        if isinstance(val, (str, bytes)) and self.VR not in \
-                ['UT', 'ST', 'LT', 'FL', 'FD', 'AT', 'OB', 'OW', 'OF', 'SL',
-                 'SQ', 'SS', 'UL', 'OB/OW', 'OW/OB', 'OB or OW',
-                 'OW or OB', 'UN'] and 'US' not in self.VR:
+        if (
+            isinstance(val, (str, bytes))
+            and self.VR not in exclusions
+            and 'US' not in self.VR
+        ):
             try:
                 if _backslash_str in val:
                     val = cast(str, val).split(_backslash_str)

```

## Test Patch

```diff
diff --git a/pydicom/tests/test_valuerep.py b/pydicom/tests/test_valuerep.py
--- a/pydicom/tests/test_valuerep.py
+++ b/pydicom/tests/test_valuerep.py
@@ -1546,3 +1546,16 @@ def test_set_value(vr, pytype, vm0, vmN, keyword):
     elem = ds[keyword]
     assert elem.value == list(vmN)
     assert list(vmN) == elem.value
+
+
+@pytest.mark.parametrize("vr, pytype, vm0, vmN, keyword", VALUE_REFERENCE)
+def test_assigning_bytes(vr, pytype, vm0, vmN, keyword):
+    """Test that byte VRs are excluded from the backslash check."""
+    if pytype == bytes:
+        ds = Dataset()
+        value = b"\x00\x01" + b"\\" + b"\x02\x03"
+        setattr(ds, keyword, value)
+        elem = ds[keyword]
+        assert elem.VR == vr
+        assert elem.value == value
+        assert elem.VM == 1

```


## Code snippets

### 1 - pydicom/dataelem.py:

Start line: 9, End line: 42

```python
import base64
import json
from typing import (
    Optional, Any, Tuple, Callable, Union, TYPE_CHECKING, Dict, TypeVar, Type,
    List, NamedTuple, MutableSequence, cast
)
import warnings

from pydicom import config  # don't import datetime_conversion directly
from pydicom.config import logger
from pydicom.datadict import (dictionary_has_tag, dictionary_description,
                              dictionary_keyword, dictionary_is_retired,
                              private_dictionary_description, dictionary_VR,
                              repeater_has_tag, private_dictionary_VR)
from pydicom.errors import BytesLengthException
from pydicom.jsonrep import JsonDataElementConverter
from pydicom.multival import MultiValue
from pydicom.tag import Tag, BaseTag
from pydicom.uid import UID
from pydicom import jsonrep
import pydicom.valuerep  # don't import DS directly as can be changed by config
from pydicom.valuerep import PersonName

if config.have_numpy:
    import numpy  # type: ignore[import]

if TYPE_CHECKING:  # pragma: no cover
    from pydicom.dataset import Dataset


BINARY_VR_VALUES = [
    'US', 'SS', 'UL', 'SL', 'OW', 'OB', 'OL', 'UN',
    'OB or OW', 'US or OW', 'US or SS or OW', 'FL', 'FD', 'OF', 'OD'
]
```
### 2 - pydicom/values.py:

Start line: 5, End line: 34

```python
import re
from io import BytesIO
from struct import (unpack, calcsize)
from typing import (
    Optional, Union, List, Tuple, Dict, Callable, cast, MutableSequence, Any
)

# don't import datetime_conversion directly
from pydicom import config
from pydicom.charset import default_encoding, decode_bytes
from pydicom.config import logger, have_numpy
from pydicom.dataelem import empty_value_for_VR, RawDataElement
from pydicom.errors import BytesLengthException
from pydicom.filereader import read_sequence
from pydicom.multival import MultiValue
from pydicom.sequence import Sequence
from pydicom.tag import (Tag, TupleTag, BaseTag)
import pydicom.uid
import pydicom.valuerep  # don't import DS directly as can be changed by config
from pydicom.valuerep import (
    MultiString, DA, DT, TM, TEXT_VR_DELIMS, DSfloat, DSdecimal, IS, text_VRs
)

try:
    import numpy  # type: ignore[import]
    have_numpy = True
except ImportError:
    have_numpy = False

from pydicom.valuerep import PersonName
```
### 3 - pydicom/values.py:

Start line: 782, End line: 834

```python
convert_retry_VR_order = [
    'SH', 'UL', 'SL', 'US', 'SS', 'FL', 'FD', 'OF', 'OB', 'UI', 'DA', 'TM',
    'PN', 'IS', 'DS', 'LT', 'SQ', 'UN', 'AT', 'OW', 'DT', 'UT', ]
# converters map a VR to the function
# to read the value(s). for convert_numbers,
# the converter maps to a tuple
# (function, struct_format)
# (struct_format in python struct module style)
converters = {
    'AE': convert_AE_string,
    'AS': convert_string,
    'AT': convert_ATvalue,
    'CS': convert_string,
    'DA': convert_DA_string,
    'DS': convert_DS_string,
    'DT': convert_DT_string,
    'FD': (convert_numbers, 'd'),
    'FL': (convert_numbers, 'f'),
    'IS': convert_IS_string,
    'LO': convert_text,
    'LT': convert_single_string,
    'OB': convert_OBvalue,
    'OD': convert_OBvalue,
    'OF': convert_OWvalue,
    'OL': convert_OBvalue,
    'OW': convert_OWvalue,
    'OV': convert_OVvalue,
    'PN': convert_PN,
    'SH': convert_text,
    'SL': (convert_numbers, 'l'),
    'SQ': convert_SQ,
    'SS': (convert_numbers, 'h'),
    'ST': convert_single_string,
    'SV': (convert_numbers, 'q'),
    'TM': convert_TM_string,
    'UC': convert_text,
    'UI': convert_UI,
    'UL': (convert_numbers, 'L'),
    'UN': convert_UN,
    'UR': convert_UR_string,
    'US': (convert_numbers, 'H'),
    'UT': convert_single_string,
    'UV': (convert_numbers, 'Q'),
    'OW/OB': convert_OBvalue,  # note OW/OB depends on other items,
    'OB/OW': convert_OBvalue,  # which we don't know at read time
    'OW or OB': convert_OBvalue,
    'OB or OW': convert_OBvalue,
    'US or SS': convert_OWvalue,
    'US or OW': convert_OWvalue,
    'US or SS or OW': convert_OWvalue,
    'US\\US or SS\\US': convert_OWvalue,
}
```
### 4 - pydicom/valuerep.py:

Start line: 3, End line: 43

```python
import datetime
import re
import sys
import warnings
from decimal import Decimal
from math import floor, isfinite, log10
from typing import (
    TypeVar, Type, Tuple, Optional, List, Dict, Union, Any, Callable,
    MutableSequence, Sequence, cast, Iterator
)

# don't import datetime_conversion directly
from pydicom import config
from pydicom.multival import MultiValue


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
### 5 - pydicom/multival.py:

Start line: 115, End line: 139

```python
class MultiValue(MutableSequence[_ItemType]):

    @overload
    def __setitem__(self, idx: int, val: _T) -> None: pass  # pragma: no cover

    @overload
    def __setitem__(self, idx: slice, val: Iterable[_T]) -> None:
        pass  # pragma: no cover

    def __setitem__(  # type: ignore[misc]
        self, idx: Union[int, slice], val: Union[_T, Iterable[_T]]
    ) -> None:
        """Set an item of the list, making sure it is of the right VR type"""
        if isinstance(idx, slice):
            val = cast(Iterable[_T], val)
            out = [self.type_constructor(v) for v in val]
            self._list.__setitem__(idx, out)
        else:
            val = cast(_T, val)
            self._list.__setitem__(idx, self.type_constructor(val))

    def sort(self, *args: Any, **kwargs: Any) -> None:
        self._list.sort(*args, **kwargs)

    def __str__(self) -> str:
        if not self:
            return ''
        lines = (
            f"{x!r}" if isinstance(x, (str, bytes)) else str(x) for x in self
        )
        return f"[{', '.join(lines)}]"

    __repr__ = __str__
```
### 6 - pydicom/tag.py:

Start line: 34, End line: 38

```python
# Type that can be used where a parameter is a tag or keyword
TagType = Union[int, str, Tuple[int, int], "BaseTag"]
TagListType = Union[
    List[int], List[str], List[Tuple[int, int]], List["BaseTag"]
]
```
### 7 - pydicom/filewriter.py:

Start line: 3, End line: 31

```python
from struct import pack
import sys
from typing import (
    Union, BinaryIO, Any, cast, Sequence, MutableSequence, Iterable, Optional,
    List
)
import warnings
import zlib

from pydicom.charset import (
    default_encoding, text_VRs, convert_encodings, encode_string
)
from pydicom.config import have_numpy
from pydicom.dataelem import DataElement_from_raw, DataElement, RawDataElement
from pydicom.dataset import Dataset, validate_file_meta, FileMetaDataset
from pydicom.filebase import DicomFile, DicomFileLike, DicomBytesIO, DicomIO
from pydicom.fileutil import path_from_pathlike, PathType
from pydicom.multival import MultiValue
from pydicom.tag import (Tag, ItemTag, ItemDelimiterTag, SequenceDelimiterTag,
                         tag_in_exception)
from pydicom.uid import DeflatedExplicitVRLittleEndian, UID
from pydicom.valuerep import (
    extra_length_VRs, PersonName, IS, DSclass, DA, DT, TM
)
from pydicom.values import convert_numbers


if have_numpy:
    import numpy  # type: ignore[import]
```
### 8 - pydicom/jsonrep.py:

Start line: 3, End line: 17

```python
import base64
from inspect import signature
import inspect
from typing import Callable, Optional, Union, Any, cast
import warnings

from pydicom.tag import BaseTag

# Order of keys is significant!
JSON_VALUE_KEYS = ('Value', 'BulkDataURI', 'InlineBinary',)

BINARY_VR_VALUES = ['OW', 'OB', 'OD', 'OF', 'OL', 'UN',
                    'OB or OW', 'US or OW', 'US or SS or OW']
VRs_TO_BE_FLOATS = ['DS', 'FL', 'FD', ]
VRs_TO_BE_INTS = ['IS', 'SL', 'SS', 'UL', 'US', 'US or SS']
```
### 9 - pydicom/config.py:

Start line: 48, End line: 77

```python
def DS_numpy(use_numpy=True):
    """Set whether multi-valued elements with VR of **DS** will be numpy arrays

    .. versionadded:: 2.0

    Parameters
    ----------
    use_numpy : bool, optional
        ``True`` (default) to read multi-value **DS** elements
        as :class:`~numpy.ndarray`, ``False`` to read multi-valued **DS**
        data elements as type :class:`~python.mulitval.MultiValue`

        Note: once a value has been accessed, changing this setting will
        no longer change its type

    Raises
    ------
    ValueError
        If :data:`use_DS_decimal` and `use_numpy` are both True.

    """

    global use_DS_numpy

    if use_DS_decimal and use_numpy:
        raise ValueError(
            "Cannot use numpy arrays to read DS elements"
            "if `use_DS_decimal` is True"
        )
    use_DS_numpy = use_numpy
```
### 10 - pydicom/_storage_sopclass_uids.py:

Start line: 172, End line: 215

```python
EnhancedUSVolumeStorage = UID(
    '1.2.840.10008.5.1.4.1.1.6.2')
EddyCurrentImageStorage = UID(
    '1.2.840.10008.5.1.4.1.1.601.1')
EddyCurrentMultiFrameImageStorage = UID(
    '1.2.840.10008.5.1.4.1.1.601.2')
RawDataStorage = UID(
    '1.2.840.10008.5.1.4.1.1.66')
SpatialRegistrationStorage = UID(
    '1.2.840.10008.5.1.4.1.1.66.1')
SpatialFiducialsStorage = UID(
    '1.2.840.10008.5.1.4.1.1.66.2')
DeformableSpatialRegistrationStorage = UID(
    '1.2.840.10008.5.1.4.1.1.66.3')
SegmentationStorage = UID(
    '1.2.840.10008.5.1.4.1.1.66.4')
SurfaceSegmentationStorage = UID(
    '1.2.840.10008.5.1.4.1.1.66.5')
TractographyResultsStorage = UID(
    '1.2.840.10008.5.1.4.1.1.66.6')
RealWorldValueMappingStorage = UID(
    '1.2.840.10008.5.1.4.1.1.67')
SurfaceScanMeshStorage = UID(
    '1.2.840.10008.5.1.4.1.1.68.1')
SurfaceScanPointCloudStorage = UID(
    '1.2.840.10008.5.1.4.1.1.68.2')
SecondaryCaptureImageStorage = UID(
    '1.2.840.10008.5.1.4.1.1.7')
MultiFrameSingleBitSecondaryCaptureImageStorage = UID(
    '1.2.840.10008.5.1.4.1.1.7.1')
MultiFrameGrayscaleByteSecondaryCaptureImageStorage = UID(
    '1.2.840.10008.5.1.4.1.1.7.2')
MultiFrameGrayscaleWordSecondaryCaptureImageStorage = UID(
    '1.2.840.10008.5.1.4.1.1.7.3')
MultiFrameTrueColorSecondaryCaptureImageStorage = UID(
    '1.2.840.10008.5.1.4.1.1.7.4')
VLEndoscopicImageStorage = UID(
    '1.2.840.10008.5.1.4.1.1.77.1.1')
VideoEndoscopicImageStorage = UID(
    '1.2.840.10008.5.1.4.1.1.77.1.1.1')
VLMicroscopicImageStorage = UID(
    '1.2.840.10008.5.1.4.1.1.77.1.2')
VideoMicroscopicImageStorage = UID(
    '1.2.840.10008.5.1.4.1.1.77.1.2.1')
```
### 25 - pydicom/dataelem.py:

Start line: 516, End line: 552

```python
class DataElement:

    def _convert(self, val: Any) -> Any:
        """Convert `val` to an appropriate type for the element's VR."""
        # If the value is a byte string and has a VR that can only be encoded
        # using the default character repertoire, we convert it to a string
        # here to allow for byte string input in these cases
        if _is_bytes(val) and self.VR in (
                'AE', 'AS', 'CS', 'DA', 'DS', 'DT', 'IS', 'TM', 'UI', 'UR'):
            val = val.decode()

        if self.VR == 'IS':
            return pydicom.valuerep.IS(val)
        elif self.VR == 'DA' and config.datetime_conversion:
            return pydicom.valuerep.DA(val)
        elif self.VR == 'DS':
            return pydicom.valuerep.DS(val)
        elif self.VR == 'DT' and config.datetime_conversion:
            return pydicom.valuerep.DT(val)
        elif self.VR == 'TM' and config.datetime_conversion:
            return pydicom.valuerep.TM(val)
        elif self.VR == "UI":
            return UID(val) if val is not None else None
        elif self.VR == "PN":
            return PersonName(val)
        elif self.VR == "AT" and (val == 0 or val):
            return val if isinstance(val, BaseTag) else Tag(val)
        # Later may need this for PersonName as for UI,
        #    but needs more thought
        # elif self.VR == "PN":
        #    return PersonName(val)
        else:  # is either a string or a type 2 optionally blank string
            return val  # this means a "numeric" value could be empty string ""
        # except TypeError:
            # print "Could not convert value '%s' to VR '%s' in tag %s" \
            # % (repr(val), self.VR, self.tag)
        # except ValueError:
            # print "Could not convert value '%s' to VR '%s' in tag %s" \
            # % (repr(val), self.VR, self.tag)
```
### 28 - pydicom/dataelem.py:

Start line: 495, End line: 514

```python
class DataElement:

    def _convert_value(self, val: Any) -> Any:
        """Convert `val` to an appropriate type and return the result.

        Uses the element's VR in order to determine the conversion method and
        resulting type.
        """
        if self.VR == 'SQ':  # a sequence - leave it alone
            from pydicom.sequence import Sequence
            if isinstance(val, Sequence):
                return val
            else:
                return Sequence(val)

        # if the value is a list, convert each element
        try:
            val.append
        except AttributeError:  # not a list
            return self._convert(val)
        else:
            return MultiValue(self._convert, val)
```
### 39 - pydicom/dataelem.py:

Start line: 729, End line: 758

```python
def _private_vr_for_tag(ds: Optional["Dataset"], tag: BaseTag) -> str:
    """Return the VR for a known private tag, otherwise "UN".

    Parameters
    ----------
    ds : Dataset, optional
        The dataset needed for the private creator lookup.
        If not given, "UN" is returned.
    tag : BaseTag
        The private tag to lookup. The caller has to ensure that the
        tag is private.

    Returns
    -------
    str
        "LO" if the tag is a private creator, the VR of the private tag if
        found in the private dictionary, or "UN".
    """
    if tag.is_private_creator:
        return "LO"
    # invalid private tags are handled as UN
    if ds is not None and (tag.element & 0xff00):
        private_creator_tag = tag.group << 16 | (tag.element >> 8)
        private_creator = ds.get(private_creator_tag, "")
        if private_creator:
            try:
                return private_dictionary_VR(tag, private_creator.value)
            except KeyError:
                pass
    return "UN"
```
### 48 - pydicom/dataelem.py:

Start line: 110, End line: 107

```python
def _is_bytes(val: object) -> bool:
    """Return True only if `val` is of type `bytes`."""
    return isinstance(val, bytes)


# double '\' because it is used as escape chr in Python
_backslash_str = "\\"
_backslash_byte = b"\\"


_DataElement = TypeVar("_DataElement", bound="DataElement")
_Dataset = TypeVar("_Dataset", bound="Dataset")


class DataElement:
    """Contain and manipulate a DICOM Element.

    Examples
    --------

    While its possible to create a new :class:`DataElement` directly and add
    it to a :class:`~pydicom.dataset.Dataset`:

    >>> from pydicom import Dataset
    >>> elem = DataElement(0x00100010, 'PN', 'CITIZEN^Joan')
    >>> ds = Dataset()
    >>> ds.add(elem)

    Its far more convenient to use a :class:`~pydicom.dataset.Dataset`
    to add a new :class:`DataElement`, as the VR and tag are determined
    automatically from the DICOM dictionary:

    >>> ds = Dataset()
    >>> ds.PatientName = 'CITIZEN^Joan'

    Empty DataElement objects (e.g. with VM = 0) show an empty string as
    value for text VRs and `None` for non-text (binary) VRs:

    >>> ds = Dataset()
    >>> ds.PatientName = None
    >>> ds.PatientName
    ''

    >>> ds.BitsAllocated = None
    >>> ds.BitsAllocated

    >>> str(ds.BitsAllocated)
    'None'

    Attributes
    ----------
    descripWidth : int
        For string display, this is the maximum width of the description
        field (default ``35``).
    is_undefined_length : bool
        Indicates whether the length field for the element was ``0xFFFFFFFFL``
        (ie undefined).
    maxBytesToDisplay : int
        For string display, elements with values containing data which is
        longer than this value will display ``"array of # bytes"``
        (default ``16``).
    showVR : bool
        For string display, include the element's VR just before it's value
        (default ``True``).
    tag : pydicom.tag.BaseTag
        The element's tag.
    VR : str
        The element's Value Representation.
    """

    descripWidth = 35
    maxBytesToDisplay = 16
    showVR = True
    is_raw = False
```
### 88 - pydicom/dataelem.py:

Start line: 599, End line: 617

```python
class DataElement:

    @property
    def repval(self) -> str:
        """Return a :class:`str` representation of the element's value."""
        long_VRs = {"OB", "OD", "OF", "OW", "UN", "UT"}
        if set(self.VR.split(" or ")) & long_VRs:
            try:
                length = len(self.value)
            except TypeError:
                pass
            else:
                if length > self.maxBytesToDisplay:
                    return "Array of %d elements" % length
        if self.VM > self.maxBytesToDisplay:
            repVal = "Array of %d elements" % self.VM
        elif isinstance(self.value, UID):
            repVal = self.value.name
        else:
            repVal = repr(self.value)  # will tolerate unicode too
        return repVal
```
### 97 - pydicom/dataelem.py:

Start line: 710, End line: 726

```python
class RawDataElement(NamedTuple):
    """Container for the data from a raw (mostly) undecoded element."""
    tag: BaseTag
    VR: Optional[str]
    length: int
    value: Optional[bytes]
    value_tell: int
    is_implicit_VR: bool
    is_little_endian: bool
    is_raw: bool = True


# The first and third values of the following elements are always US
#   even if the VR is SS (PS3.3 C.7.6.3.1.5, C.11.1, C.11.2).
# (0028,1101-1103) RGB Palette Color LUT Descriptor
# (0028,3002) LUT Descriptor
_LUT_DESCRIPTOR_TAGS = (0x00281101, 0x00281102, 0x00281103, 0x00283002)
```
### 106 - pydicom/dataelem.py:

Start line: 761, End line: 861

```python
def DataElement_from_raw(
    raw_data_element: RawDataElement,
    encoding: Optional[Union[str, MutableSequence[str]]] = None,
    dataset: Optional["Dataset"] = None
) -> DataElement:
    """Return a :class:`DataElement` created from `raw_data_element`.

    Parameters
    ----------
    raw_data_element : RawDataElement
        The raw data to convert to a :class:`DataElement`.
    encoding : str or list of str, optional
        The character encoding of the raw data.
    dataset : Dataset, optional
        If given, used to resolve the VR for known private tags.

    Returns
    -------
    DataElement

    Raises
    ------
    KeyError
        If `raw_data_element` belongs to an unknown non-private tag and
        `config.enforce_valid_values` is set.
    """
    # XXX buried here to avoid circular import
    # filereader->Dataset->convert_value->filereader
    # (for SQ parsing)

    from pydicom.values import convert_value
    raw = raw_data_element

    # If user has hooked into conversion of raw values, call his/her routine
    if config.data_element_callback:
        raw = config.data_element_callback(
            raw_data_element,
            encoding=encoding,
            **config.data_element_callback_kwargs
        )

    VR = raw.VR
    if VR is None:  # Can be if was implicit VR
        try:
            VR = dictionary_VR(raw.tag)
        except KeyError:
            # just read the bytes, no way to know what they mean
            if raw.tag.is_private:
                # for VR for private tags see PS3.5, 6.2.2
                VR = _private_vr_for_tag(dataset, raw.tag)

            # group length tag implied in versions < 3.0
            elif raw.tag.element == 0:
                VR = 'UL'
            else:
                msg = "Unknown DICOM tag {0:s}".format(str(raw.tag))
                if config.enforce_valid_values:
                    msg += " can't look up VR"
                    raise KeyError(msg)
                else:
                    VR = 'UN'
                    msg += " - setting VR to 'UN'"
                    warnings.warn(msg)
    elif VR == 'UN' and config.replace_un_with_known_vr:
        # handle rare case of incorrectly set 'UN' in explicit encoding
        # see also DataElement.__init__()
        if raw.tag.is_private:
            VR = _private_vr_for_tag(dataset, raw.tag)
        elif raw.value is None or len(raw.value) < 0xffff:
            try:
                VR = dictionary_VR(raw.tag)
            except KeyError:
                pass
    try:
        value = convert_value(VR, raw, encoding)
    except NotImplementedError as e:
        raise NotImplementedError("{0:s} in tag {1!r}".format(str(e), raw.tag))
    except BytesLengthException as e:
        message = (f"{e} This occurred while trying to parse "
                   f"{raw.tag} according to VR '{VR}'.")
        if config.convert_wrong_length_to_UN:
            warnings.warn(f"{message} Setting VR to 'UN'.")
            VR = "UN"
            value = raw.value
        else:
            raise BytesLengthException(
                f"{message} To replace this error with a warning set "
                "pydicom.config.convert_wrong_length_to_UN = True."
            )

    if raw.tag in _LUT_DESCRIPTOR_TAGS and value:
        # We only fix the first value as the third value is 8 or 16
        try:
            if value[0] < 0:
                value[0] += 65536
        except TypeError:
            pass

    return DataElement(raw.tag, VR, value, raw.value_tell,
                       raw.length == 0xFFFFFFFF, already_converted=True)
```
### 125 - pydicom/dataelem.py:

Start line: 432, End line: 430

```python
class DataElement:

    @property
    def value(self) -> Any:
        """Return the element's value."""
        return self._value

    @value.setter
    def value(self, val: Any) -> None:
        """Convert (if necessary) and set the value of the element."""
        # Check if is a string with multiple values separated by '\'
        # If so, turn them into a list of separate strings
        #  Last condition covers 'US or SS' etc
        if isinstance(val, (str, bytes)) and self.VR not in \
                ['UT', 'ST', 'LT', 'FL', 'FD', 'AT', 'OB', 'OW', 'OF', 'SL',
                 'SQ', 'SS', 'UL', 'OB/OW', 'OW/OB', 'OB or OW',
                 'OW or OB', 'UN'] and 'US' not in self.VR:
            try:
                if _backslash_str in val:
                    val = cast(str, val).split(_backslash_str)
            except TypeError:
                if _backslash_byte in val:
                    val = val.split(_backslash_byte)
        self._value = self._convert_value(val)
```
