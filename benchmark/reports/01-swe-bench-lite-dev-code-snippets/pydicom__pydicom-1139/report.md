# pydicom__pydicom-1139

| **pydicom/pydicom** | `b9fb05c177b685bf683f7f57b2d57374eb7d882d` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 20728 |
| **Any found context length** | 457 |
| **Avg pos** | 122.0 |
| **Min pos** | 2 |
| **Max pos** | 54 |
| **Top file pos** | 2 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/pydicom/valuerep.py b/pydicom/valuerep.py
--- a/pydicom/valuerep.py
+++ b/pydicom/valuerep.py
@@ -1,6 +1,5 @@
 # Copyright 2008-2018 pydicom authors. See LICENSE file for details.
 """Special classes for DICOM value representations (VR)"""
-from copy import deepcopy
 from decimal import Decimal
 import re
 
@@ -750,6 +749,25 @@ def __ne__(self, other):
     def __str__(self):
         return '='.join(self.components).__str__()
 
+    def __next__(self):
+        # Get next character or stop iteration
+        if self._i < self._rep_len:
+            c = self._str_rep[self._i]
+            self._i += 1
+            return c
+        else:
+            raise StopIteration
+
+    def __iter__(self):
+        # Get string rep. and length, initialize index counter
+        self._str_rep = self.__str__()
+        self._rep_len = len(self._str_rep)
+        self._i = 0
+        return self
+
+    def __contains__(self, x):
+        return x in self.__str__()
+
     def __repr__(self):
         return '='.join(self.components).__repr__()
 

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| pydicom/valuerep.py | 3 | 4 | 54 | 2 | 20728
| pydicom/valuerep.py | 750 | - | 54 | 2 | 20728


## Problem Statement

```
Make PersonName3 iterable
\`\`\`python
from pydicom import Dataset

ds = Dataset()
ds.PatientName = 'SomeName'

'S' in ds.PatientName
\`\`\`
\`\`\`
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: argument of type 'PersonName3' is not iterable
\`\`\`

I'm not really sure if this is intentional or if PN elements should support `str` methods. And yes I know I can `str(ds.PatientName)` but it's a bit silly, especially when I keep having to write exceptions to my element iterators just for PN elements.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- |
| 1 | 1 pydicom/jsonrep.py | 209 | 249| 315 | 315 | 
| **-> 2 <-** | **2 pydicom/valuerep.py** | 731 | 756| 142 | 457 | 
| 3 | 3 pydicom/dataset.py | 196 | 345| 1306 | 1763 | 
| **-> 4 <-** | **3 pydicom/valuerep.py** | 608 | 729| 833 | 2596 | 
| 5 | 3 pydicom/dataset.py | 1941 | 1977| 272 | 2868 | 
| 6 | 3 pydicom/dataset.py | 425 | 444| 133 | 3001 | 
| 7 | 4 examples/show_charset_name.py | 0 | 43| 426 | 3427 | 
| **-> 8 <-** | **4 pydicom/valuerep.py** | 576 | 605| 224 | 3651 | 
| 9 | 4 pydicom/dataset.py | 446 | 475| 189 | 3840 | 
| 10 | 4 pydicom/dataset.py | 504 | 542| 282 | 4122 | 
| 11 | 4 pydicom/dataset.py | 1979 | 1995| 120 | 4242 | 
| 12 | **4 pydicom/valuerep.py** | 758 | 784| 245 | 4487 | 
| 13 | 5 pydicom/values.py | 5 | 28| 191 | 4678 | 
| 14 | 5 pydicom/values.py | 374 | 400| 184 | 4862 | 
| 15 | 5 pydicom/dataset.py | 1108 | 1135| 230 | 5092 | 
| 16 | 5 pydicom/dataset.py | 1318 | 1352| 337 | 5429 | 
| 17 | 6 pydicom/dataelem.py | 590 | 601| 140 | 5569 | 
| 18 | 6 pydicom/dataset.py | 741 | 782| 306 | 5875 | 
| 19 | 6 pydicom/dataset.py | 1660 | 1717| 490 | 6365 | 
| 20 | 6 pydicom/dataset.py | 1615 | 1658| 375 | 6740 | 
| 21 | 6 pydicom/jsonrep.py | 3 | 14| 129 | 6869 | 
| 22 | **6 pydicom/valuerep.py** | 786 | 831| 376 | 7245 | 
| 23 | 7 pydicom/__init__.py | 31 | 50| 123 | 7368 | 
| 24 | 7 pydicom/dataset.py | 943 | 975| 226 | 7594 | 
| 25 | 7 pydicom/dataset.py | 1844 | 1842| 463 | 8057 | 
| 26 | 8 examples/memory_dataset.py | 13 | 84| 492 | 8549 | 
| 27 | 9 pydicom/config.py | 112 | 227| 861 | 9410 | 
| 28 | 9 pydicom/dataset.py | 1787 | 1831| 421 | 9831 | 
| 29 | 9 pydicom/dataset.py | 610 | 643| 288 | 10119 | 
| 30 | 10 pydicom/util/dump.py | 68 | 104| 308 | 10427 | 
| 31 | 11 examples/metadata_processing/plot_anonymize.py | 0 | 92| 479 | 10906 | 
| 32 | 12 pydicom/sr/_cid_dict.py | 24848 | 24944| 699 | 11605 | 
| 33 | 12 pydicom/dataset.py | 784 | 868| 784 | 12389 | 
| 34 | 12 pydicom/sr/_cid_dict.py | 20320 | 20390| 736 | 13125 | 
| 35 | 12 pydicom/dataelem.py | 90 | 151| 455 | 13580 | 
| 36 | 12 pydicom/dataelem.py | 603 | 627| 206 | 13786 | 
| 37 | 12 pydicom/dataset.py | 1719 | 1750| 261 | 14047 | 
| 38 | 12 pydicom/sr/_cid_dict.py | 8161 | 8226| 498 | 14545 | 
| 39 | 12 pydicom/dataset.py | 1269 | 1316| 370 | 14915 | 
| 40 | 13 examples/input_output/plot_printing_dataset.py | 18 | 15| 306 | 15221 | 
| 41 | 14 pydicom/uid.py | 33 | 51| 132 | 15353 | 
| 42 | 14 pydicom/dataelem.py | 10 | 35| 237 | 15590 | 
| 43 | 14 pydicom/dataset.py | 17 | 46| 255 | 15845 | 
| 44 | 14 pydicom/sr/_cid_dict.py | 454 | 545| 746 | 16591 | 
| 45 | 14 pydicom/dataset.py | 1047 | 1061| 158 | 16749 | 
| 46 | 14 pydicom/dataset.py | 346 | 397| 387 | 17136 | 
| 47 | 14 pydicom/sr/_cid_dict.py | 22931 | 23001| 670 | 17806 | 
| 48 | 14 pydicom/dataset.py | 1014 | 1045| 262 | 18068 | 
| 49 | 14 pydicom/sr/_cid_dict.py | 20659 | 20732| 649 | 18717 | 
| 50 | 14 pydicom/dataset.py | 2040 | 2080| 324 | 19041 | 
| 51 | 14 pydicom/sr/_cid_dict.py | 11709 | 11757| 378 | 19419 | 
| 52 | 15 pydicom/util/codify.py | 16 | 75| 457 | 19876 | 
| 53 | 15 pydicom/dataset.py | 1210 | 1267| 497 | 20373 | 
| **-> 54 <-** | **15 pydicom/valuerep.py** | 2 | 36| 355 | 20728 | 


### Hint

```
I think it is reasonable to support at least some `str` methods (definitely `__contains__` for the example above), but there are many that don't make a lot of sense in this context though - e.g. `join`, `ljust`, `maketrans`, `splitlines` just to name a few, but I suppose each would either never be actually used or would have no effect.

I have a vague memory that one or more of the `PersonName` classes was at one time subclassed from `str`, or at least that it was discussed... does anyone remember?  Maybe it would be easier now with only Python 3 supported.
`PersonName` was derived from `str` or `unicode` in Python 2, but that caused a number of problems, which is why you switched to `PersonName3` in Python 3, I think. I agree though that it makes sense to implement `str` methods, either by implementing some of them, or generically by adding `__getattr__` that converts it to `str` and applies the attribute to that string. 
```

## Patch

```diff
diff --git a/pydicom/valuerep.py b/pydicom/valuerep.py
--- a/pydicom/valuerep.py
+++ b/pydicom/valuerep.py
@@ -1,6 +1,5 @@
 # Copyright 2008-2018 pydicom authors. See LICENSE file for details.
 """Special classes for DICOM value representations (VR)"""
-from copy import deepcopy
 from decimal import Decimal
 import re
 
@@ -750,6 +749,25 @@ def __ne__(self, other):
     def __str__(self):
         return '='.join(self.components).__str__()
 
+    def __next__(self):
+        # Get next character or stop iteration
+        if self._i < self._rep_len:
+            c = self._str_rep[self._i]
+            self._i += 1
+            return c
+        else:
+            raise StopIteration
+
+    def __iter__(self):
+        # Get string rep. and length, initialize index counter
+        self._str_rep = self.__str__()
+        self._rep_len = len(self._str_rep)
+        self._i = 0
+        return self
+
+    def __contains__(self, x):
+        return x in self.__str__()
+
     def __repr__(self):
         return '='.join(self.components).__repr__()
 

```

## Test Patch

```diff
diff --git a/pydicom/tests/test_valuerep.py b/pydicom/tests/test_valuerep.py
--- a/pydicom/tests/test_valuerep.py
+++ b/pydicom/tests/test_valuerep.py
@@ -427,6 +427,62 @@ def test_hash(self):
         )
         assert hash(pn1) == hash(pn2)
 
+    def test_next(self):
+        """Test that the next function works on it's own"""
+        # Test getting the first character
+        pn1 = PersonName("John^Doe^^Dr", encodings=default_encoding)
+        pn1_itr = iter(pn1)
+        assert next(pn1_itr) == "J"
+
+        # Test getting multiple characters
+        pn2 = PersonName(
+            "Yamada^Tarou=山田^太郎=やまだ^たろう", [default_encoding, "iso2022_jp"]
+        )
+        pn2_itr = iter(pn2)
+        assert next(pn2_itr) == "Y"
+        assert next(pn2_itr) == "a"
+
+        # Test getting all characters
+        pn3 = PersonName("SomeName")
+        pn3_itr = iter(pn3)
+        assert next(pn3_itr) == "S"
+        assert next(pn3_itr) == "o"
+        assert next(pn3_itr) == "m"
+        assert next(pn3_itr) == "e"
+        assert next(pn3_itr) == "N"
+        assert next(pn3_itr) == "a"
+        assert next(pn3_itr) == "m"
+        assert next(pn3_itr) == "e"
+
+        # Attempting to get next characeter should stop the iteration
+        # I.e. next can only start once
+        with pytest.raises(StopIteration):
+            next(pn3_itr)
+
+        # Test that next() doesn't work without instantiating an iterator
+        pn4 = PersonName("SomeName")
+        with pytest.raises(AttributeError):
+            next(pn4)
+
+    def test_iterator(self):
+        """Test that iterators can be corretly constructed"""
+        name_str = "John^Doe^^Dr"
+        pn1 = PersonName(name_str)
+        
+        for i, c in enumerate(pn1):
+            assert name_str[i] == c
+
+        # Ensure that multiple iterators can be created on the same variable
+        for i, c in enumerate(pn1):
+            assert name_str[i] == c
+
+    def test_contains(self):
+        """Test that characters can be check if they are within the name"""
+        pn1 = PersonName("John^Doe")
+        assert ("J" in pn1) == True
+        assert ("o" in pn1) == True
+        assert ("x" in pn1) == False
+
 
 class TestDateTime:
     """Unit tests for DA, DT, TM conversion to datetime objects"""

```


## Code snippets

### 1 - pydicom/jsonrep.py:

Start line: 209, End line: 249

```python
class JsonDataElementConverter:

    def get_pn_element_value(self, value):
        """Return PersonName value from JSON value.

        Values with VR PN have a special JSON encoding, see the DICOM Standard,
        Part 18, :dcm:`Annex F.2.2<part18/sect_F.2.2.html>`.

        Parameters
        ----------
        value : dict
            The person name components in the JSON entry.

        Returns
        -------
        PersonName or str
            The decoded PersonName object or an empty string.
        """
        if not isinstance(value, dict):
            # Some DICOMweb services get this wrong, so we
            # workaround the issue and warn the user
            # rather than raising an error.
            warnings.warn(
                'value of data element "{}" with VR Person Name (PN) '
                'is not formatted correctly'.format(self.tag)
            )
            return value
        else:
            if 'Phonetic' in value:
                comps = ['', '', '']
            elif 'Ideographic' in value:
                comps = ['', '']
            else:
                comps = ['']
            if 'Alphabetic' in value:
                comps[0] = value['Alphabetic']
            if 'Ideographic' in value:
                comps[1] = value['Ideographic']
            if 'Phonetic' in value:
                comps[2] = value['Phonetic']
            elem_value = '='.join(comps)
            return elem_value
```
### 2 - pydicom/valuerep.py:

Start line: 731, End line: 756

```python
class PersonName:

    @property
    def phonetic(self):
        """Return the third (phonetic) person name component as a
        unicode string

        .. versionadded:: 1.2
        """
        try:
            return self.components[2]
        except IndexError:
            return ''

    def __eq__(self, other):
        return str(self) == other

    def __ne__(self, other):
        return not self == other

    def __str__(self):
        return '='.join(self.components).__str__()

    def __repr__(self):
        return '='.join(self.components).__repr__()

    def __hash__(self):
        return hash(self.components)
```
### 3 - pydicom/dataset.py:

Start line: 196, End line: 345

```python
class Dataset(dict):
    """Contains a collection (dictionary) of DICOM Data Elements.

    Behaves like a :class:`dict`.

    .. note::

        :class:`Dataset` is only derived from :class:`dict` to make it work in
        a NumPy :class:`~numpy.ndarray`. The parent :class:`dict` class
        is never called, as all :class:`dict` methods are overridden.

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
    ...             [recurse(item) for item in elem]
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
        the given endianess.
    is_implicit_VR : bool
        Shall be set before writing with ``write_like_original=False``.
        The :class:`Dataset` will be written using the transfer syntax with
        the given VR handling, e.g *Little Endian Implicit VR* if ``True``,
        and *Little Endian Explicit VR* or *Big Endian Explicit VR* (depending
        on ``Dataset.is_little_endian``) if ``False``.
    """
```
### 4 - pydicom/valuerep.py:

Start line: 608, End line: 729

```python
class PersonName:
    def __new__(cls, *args, **kwargs):
        # Handle None value by returning None instead of a PersonName object
        if len(args) and args[0] is None:
            return None
        return super(PersonName, cls).__new__(cls)

    def __init__(self, val, encodings=None, original_string=None):
        if isinstance(val, PersonName):
            encodings = val.encodings
            self.original_string = val.original_string
            self._components = tuple(str(val).split('='))
        elif isinstance(val, bytes):
            # this is the raw byte string - decode it on demand
            self.original_string = val
            self._components = None
        else:
            # handle None `val` as empty string
            val = val or ''

            # this is the decoded string - save the original string if
            # available for easier writing back
            self.original_string = original_string
            components = val.split('=')
            # Remove empty elements from the end to avoid trailing '='
            while len(components) and not components[-1]:
                components.pop()
            self._components = tuple(components)

            # if the encoding is not given, leave it as undefined (None)
        self.encodings = _verify_encodings(encodings)
        self._dict = {}

    def _create_dict(self):
        """Creates a dictionary of person name group and component names.

        Used exclusively for `formatted` for backwards compatibility.
        """
        if not self._dict:
            for name in ('family_name', 'given_name', 'middle_name',
                         'name_prefix', 'name_suffix',
                         'ideographic', 'phonetic'):
                self._dict[name] = getattr(self, name, '')

    @property
    def components(self):
        """Returns up to three decoded person name components.

        .. versionadded:: 1.2

        The returned components represent the alphabetic, ideographic and
        phonetic representations as a list of unicode strings.
        """
        if self._components is None:
            groups = self.original_string.split(b'=')
            encodings = self.encodings or [default_encoding]
            self._components = _decode_personname(groups, encodings)

        return self._components

    def _name_part(self, i):
        try:
            return self.components[0].split('^')[i]
        except IndexError:
            return ''

    @property
    def family_name(self):
        """Return the first (family name) group of the alphabetic person name
        representation as a unicode string

        .. versionadded:: 1.2
        """
        return self._name_part(0)

    @property
    def given_name(self):
        """Return the second (given name) group of the alphabetic person name
        representation as a unicode string

        .. versionadded:: 1.2
        """
        return self._name_part(1)

    @property
    def middle_name(self):
        """Return the third (middle name) group of the alphabetic person name
        representation as a unicode string

        .. versionadded:: 1.2
        """
        return self._name_part(2)

    @property
    def name_prefix(self):
        """Return the fourth (name prefix) group of the alphabetic person name
        representation as a unicode string

        .. versionadded:: 1.2
        """
        return self._name_part(3)

    @property
    def name_suffix(self):
        """Return the fifth (name suffix) group of the alphabetic person name
        representation as a unicode string

        .. versionadded:: 1.2
        """
        return self._name_part(4)

    @property
    def ideographic(self):
        """Return the second (ideographic) person name component as a
        unicode string

        .. versionadded:: 1.2
        """
        try:
            return self.components[1]
        except IndexError:
            return ''
```
### 5 - pydicom/dataset.py:

Start line: 1941, End line: 1977

```python
class Dataset(dict):

    def __str__(self):
        """Handle str(dataset).

        ..versionchanged:: 2.0

            The file meta information was added in its own section,
            if :data:`pydicom.config.show_file_meta` is ``True``

        """
        return self._pretty_str()

    def top(self):
        """Return a :class:`str` representation of the top level elements. """
        return self._pretty_str(top_level_only=True)

    def trait_names(self):
        """Return a :class:`list` of valid names for auto-completion code.

        Used in IPython, so that data element names can be found and offered
        for autocompletion on the IPython command line.
        """
        return dir(self)  # only valid python >=2.6, else use self.__dir__()

    def update(self, dictionary):
        """Extend :meth:`dict.update` to handle DICOM tags and keywords.

        Parameters
        ----------
        dictionary : dict or Dataset
            The :class:`dict` or :class:`Dataset` to use when updating the
            current object.
        """
        for key, value in list(dictionary.items()):
            if isinstance(key, str):
                setattr(self, key, value)
            else:
                self[Tag(key)] = value
```
### 6 - pydicom/dataset.py:

Start line: 425, End line: 444

```python
class Dataset(dict):

    def data_element(self, name):
        """Return the element corresponding to the element keyword `name`.

        Parameters
        ----------
        name : str
            A DICOM element keyword.

        Returns
        -------
        dataelem.DataElement or None
            For the given DICOM element `keyword`, return the corresponding
            :class:`~pydicom.dataelem.DataElement` if present, ``None``
            otherwise.
        """
        tag = tag_for_keyword(name)
        # Test against None as (0000,0000) is a possible tag
        if tag is not None:
            return self[tag]
        return None
```
### 7 - examples/show_charset_name.py:

```python
"""
============================
Display unicode person names
============================

Very simple app to display unicode person names.

"""

from pydicom.valuerep import PersonNameUnicode

import tkinter

print(__doc__)

default_encoding = 'iso8859'

root = tkinter.Tk()
# root.geometry("%dx%d%+d%+d" % (800, 600, 0, 0))

person_names = [
    PersonNameUnicode(
        b"Yamada^Tarou=\033$B;3ED\033(B^\033$BB@O:"
        b"\033(B=\033$B$d$^$@\033(B^\033$B$?$m$&\033(B",
        [default_encoding, 'iso2022_jp']),  # DICOM standard 2008-PS3.5 H.3 p98
    PersonNameUnicode(
        b"Wang^XiaoDong=\xcd\xf5\x5e\xd0\xa1\xb6\xab=",
        [default_encoding, 'GB18030']),  # DICOM standard 2008-PS3.5 J.3 p 105
    PersonNameUnicode(
        b"Wang^XiaoDong=\xe7\x8e\x8b\x5e\xe5\xb0\x8f\xe6\x9d\xb1=",
        [default_encoding, 'UTF-8']),  # DICOM standard 2008-PS3.5 J.1 p 104
    PersonNameUnicode(
        b"Hong^Gildong=\033$)C\373\363^\033$)C\321\316\324\327="
        b"\033$)C\310\253^\033$)C\261\346\265\277",
        [default_encoding, 'euc_kr']),  # DICOM standard 2008-PS3.5 I.2 p 101
]
for person_name in person_names:
    label = tkinter.Label(text=person_name)
    label.pack()
root.mainloop()
```
### 8 - pydicom/valuerep.py:

Start line: 576, End line: 605

```python
def _encode_personname(components, encodings):
    """Encode a list of text string person name components.

    Parameters
    ----------
    components : list of text type
        The list of the up to three unicode person name components
    encodings : list of str
        The Python encodings uses to encode `components`.

    Returns
    -------
    byte string
        The byte string that can be written as a PN DICOM tag value.
        If the encoding of some component parts is not possible using the
        given encodings, they are encoded with the first encoding using
        replacement bytes for characters that cannot be encoded.
    """
    from pydicom.charset import encode_string

    encoded_comps = []
    for comp in components:
        groups = [encode_string(group, encodings)
                  for group in comp.split('^')]
        encoded_comps.append(b'^'.join(groups))

    # Remove empty elements from the end
    while len(encoded_comps) and not encoded_comps[-1]:
        encoded_comps.pop()
    return b'='.join(encoded_comps)
```
### 9 - pydicom/dataset.py:

Start line: 446, End line: 475

```python
class Dataset(dict):

    def __contains__(self, name):
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
            tag = Tag(name)
        except (ValueError, OverflowError):
            return False
        # Test against None as (0000,0000) is a possible tag
        if tag is not None:
            return tag in self._dict
        return name in self._dict  # will no doubt raise an exception
```
### 10 - pydicom/dataset.py:

Start line: 504, End line: 542

```python
class Dataset(dict):

    def __delattr__(self, name):
        """Intercept requests to delete an attribute by `name`.

        Examples
        --------

        >>> ds = Dataset()
        >>> ds.PatientName = 'foo'
        >>> ds.some_attribute = True

        If `name` is a DICOM keyword - delete the corresponding
        :class:`~pydicom.dataelem.DataElement`

        >>> del ds.PatientName
        >>> 'PatientName' in ds
        False

        If `name` is another attribute - delete it

        >>> del ds.some_attribute
        >>> hasattr(ds, 'some_attribute')
        False

        Parameters
        ----------
        name : str
            The keyword for the DICOM element or the class attribute to delete.
        """
        # First check if a valid DICOM keyword and if we have that data element
        tag = tag_for_keyword(name)
        if tag is not None and tag in self._dict:
            del self._dict[tag]
        # If not a DICOM name in this dataset, check for regular instance name
        #   can't do delete directly, that will call __delattr__ again
        elif name in self.__dict__:
            del self.__dict__[name]
        # Not found, raise an error in same style as python does
        else:
            raise AttributeError(name)
```
### 12 - pydicom/valuerep.py:

Start line: 758, End line: 784

```python
class PersonName:

    def decode(self, encodings=None):
        """Return the patient name decoded by the given `encodings`.

        Parameters
        ----------
        encodings : list of str
            The list of encodings used for decoding the byte string. If not
            given, the initial encodings set in the object are used.

        Returns
        -------
        valuerep.PersonName
            A person name object that will return the decoded string with
            the given encodings on demand. If the encodings are not given,
            the current object is returned.
        """
        # in the common case (encoding did not change) we decode on demand
        if encodings is None or encodings == self.encodings:
            return self
        # the encoding was unknown or incorrect - create a new
        # PersonName object with the changed encoding
        encodings = _verify_encodings(encodings)
        if self.original_string is None:
            # if the original encoding was not set, we set it now
            self.original_string = _encode_personname(
                self.components, self.encodings or [default_encoding])
        return PersonName(self.original_string, encodings)
```
### 22 - pydicom/valuerep.py:

Start line: 786, End line: 831

```python
class PersonName:

    def encode(self, encodings=None):
        """Return the patient name decoded by the given `encodings`.

        Parameters
        ----------
        encodings : list of str
            The list of encodings used for encoding the unicode string. If
            not given, the initial encodings set in the object are used.

        Returns
        -------
        bytes
            The person name encoded with the given encodings as a byte string.
            If no encoding is given, the original byte string is returned, if
            available, otherwise each group of the patient name is encoded
            with the first matching of the given encodings.
        """
        encodings = _verify_encodings(encodings) or self.encodings

        # if the encoding is not the original encoding, we have to return
        # a re-encoded string (without updating the original string)
        if encodings != self.encodings and self.encodings is not None:
            return _encode_personname(self.components, encodings)
        if self.original_string is None:
            # if the original encoding was not set, we set it now
            self.original_string = _encode_personname(
                self.components, encodings or [default_encoding])
        return self.original_string

    def family_comma_given(self):
        return self.formatted('%(family_name)s, %(given_name)s')

    def formatted(self, format_str):
        self._create_dict()
        return format_str % self._dict

    def __bool__(self):
        if self.original_string is None:
            return (bool(self._components) and
                    (len(self._components) > 1 or bool(self._components[0])))
        return bool(self.original_string)


# Alias old class names for backwards compat in user code
PersonNameUnicode = PersonName = PersonName
```
### 54 - pydicom/valuerep.py:

Start line: 2, End line: 36

```python
from copy import deepcopy
from decimal import Decimal
import re

from datetime import (date, datetime, time, timedelta, timezone)

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
text_VRs = ('SH', 'LO', 'ST', 'LT', 'UC', 'UT')

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
