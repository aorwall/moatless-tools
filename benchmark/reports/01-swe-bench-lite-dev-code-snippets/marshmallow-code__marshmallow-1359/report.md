# marshmallow-code__marshmallow-1359

| **marshmallow-code/marshmallow** | `b40a0f4e33823e6d0f341f7e8684e359a99060d1` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 17236 |
| **Any found context length** | 17236 |
| **Avg pos** | 61.0 |
| **Min pos** | 61 |
| **Max pos** | 61 |
| **Top file pos** | 3 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/src/marshmallow/fields.py b/src/marshmallow/fields.py
--- a/src/marshmallow/fields.py
+++ b/src/marshmallow/fields.py
@@ -1114,7 +1114,7 @@ def _bind_to_schema(self, field_name, schema):
         super()._bind_to_schema(field_name, schema)
         self.format = (
             self.format
-            or getattr(schema.opts, self.SCHEMA_OPTS_VAR_NAME)
+            or getattr(self.root.opts, self.SCHEMA_OPTS_VAR_NAME)
             or self.DEFAULT_FORMAT
         )
 

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| src/marshmallow/fields.py | 1117 | 1119 | 61 | 3 | 17236


## Problem Statement

```
3.0: DateTime fields cannot be used as inner field for List or Tuple fields
Between releases 3.0.0rc8 and 3.0.0rc9, `DateTime` fields have started throwing an error when being instantiated as inner fields of container fields like `List` or `Tuple`. The snippet below works in <=3.0.0rc8 and throws the error below in >=3.0.0rc9 (and, worryingly, 3.0.0):

\`\`\`python
from marshmallow import fields, Schema

class MySchema(Schema):
    times = fields.List(fields.DateTime())

s = MySchema()
\`\`\`

Traceback:
\`\`\`
Traceback (most recent call last):
  File "test-mm.py", line 8, in <module>
    s = MySchema()
  File "/Users/victor/.pyenv/versions/marshmallow/lib/python3.6/site-packages/marshmallow/schema.py", line 383, in __init__
    self.fields = self._init_fields()
  File "/Users/victor/.pyenv/versions/marshmallow/lib/python3.6/site-packages/marshmallow/schema.py", line 913, in _init_fields
    self._bind_field(field_name, field_obj)
  File "/Users/victor/.pyenv/versions/marshmallow/lib/python3.6/site-packages/marshmallow/schema.py", line 969, in _bind_field
    field_obj._bind_to_schema(field_name, self)
  File "/Users/victor/.pyenv/versions/marshmallow/lib/python3.6/site-packages/marshmallow/fields.py", line 636, in _bind_to_schema
    self.inner._bind_to_schema(field_name, self)
  File "/Users/victor/.pyenv/versions/marshmallow/lib/python3.6/site-packages/marshmallow/fields.py", line 1117, in _bind_to_schema
    or getattr(schema.opts, self.SCHEMA_OPTS_VAR_NAME)
AttributeError: 'List' object has no attribute 'opts'
\`\`\`

It seems like it's treating the parent field as a Schema without checking that it is indeed a schema, so the `schema.opts` statement fails as fields don't have an `opts` attribute.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- |
| 1 | 1 src/marshmallow/__init__.py | 0 | 34| 180 | 180 | 
| 2 | 2 src/marshmallow/schema.py | 879 | 947| 534 | 714 | 
| 3 | **3 src/marshmallow/fields.py** | 407 | 461| 187 | 901 | 
| 4 | **3 src/marshmallow/fields.py** | 1130 | 1152| 216 | 1117 | 
| 5 | **3 src/marshmallow/fields.py** | 669 | 728| 431 | 1548 | 
| 6 | 3 src/marshmallow/schema.py | 352 | 410| 494 | 2042 | 
| 7 | 3 src/marshmallow/schema.py | 224 | 306| 756 | 2798 | 
| 8 | **3 src/marshmallow/fields.py** | 501 | 531| 268 | 3066 | 
| 9 | 3 src/marshmallow/schema.py | 186 | 221| 367 | 3433 | 
| 10 | 4 examples/flask_example.py | 47 | 64| 132 | 3565 | 
| 11 | 4 src/marshmallow/schema.py | 1017 | 1061| 311 | 3876 | 
| 12 | **4 src/marshmallow/fields.py** | 463 | 499| 281 | 4157 | 
| 13 | 4 src/marshmallow/schema.py | 0 | 35| 192 | 4349 | 
| 14 | **4 src/marshmallow/fields.py** | 382 | 404| 139 | 4488 | 
| 15 | **4 src/marshmallow/fields.py** | 648 | 666| 158 | 4646 | 
| 16 | 5 examples/inflection_example.py | 0 | 30| 198 | 4844 | 
| 17 | 6 src/marshmallow/exceptions.py | 11 | 8| 374 | 5218 | 
| 18 | 6 src/marshmallow/schema.py | 1106 | 1140| 266 | 5484 | 
| 19 | 6 src/marshmallow/schema.py | 1063 | 1104| 247 | 5731 | 
| 20 | 7 src/marshmallow/class_registry.py | 0 | 15| 108 | 5839 | 
| 21 | 7 src/marshmallow/schema.py | 310 | 308| 495 | 6334 | 
| 22 | **7 src/marshmallow/fields.py** | 730 | 749| 130 | 6464 | 
| 23 | **7 src/marshmallow/fields.py** | 0 | 63| 291 | 6755 | 
| 24 | 7 src/marshmallow/schema.py | 956 | 954| 263 | 7018 | 
| 25 | 7 src/marshmallow/schema.py | 711 | 729| 123 | 7141 | 
| 26 | 7 src/marshmallow/schema.py | 860 | 877| 202 | 7343 | 
| 27 | 8 examples/peewee_example.py | 76 | 98| 186 | 7529 | 
| 28 | **8 src/marshmallow/fields.py** | 546 | 594| 416 | 7945 | 
| 29 | 8 src/marshmallow/schema.py | 83 | 88| 395 | 8340 | 
| 30 | 8 src/marshmallow/schema.py | 412 | 439| 243 | 8583 | 
| 31 | 8 examples/flask_example.py | 0 | 44| 272 | 8855 | 
| 32 | **8 src/marshmallow/fields.py** | 344 | 342| 254 | 9109 | 
| 33 | 9 src/marshmallow/utils.py | 0 | 105| 673 | 9782 | 
| 34 | 9 src/marshmallow/schema.py | 996 | 1015| 132 | 9914 | 
| 35 | 10 src/marshmallow/decorators.py | 0 | 65| 433 | 10347 | 
| 36 | 10 examples/peewee_example.py | 50 | 73| 179 | 10526 | 
| 37 | **10 src/marshmallow/fields.py** | 126 | 198| 519 | 11045 | 
| 38 | **10 src/marshmallow/fields.py** | 533 | 543| 116 | 11161 | 
| 39 | 10 src/marshmallow/schema.py | 845 | 858| 157 | 11318 | 
| 40 | 10 src/marshmallow/schema.py | 984 | 982| 170 | 11488 | 
| 41 | 11 src/marshmallow/base.py | 0 | 45| 229 | 11717 | 
| 42 | **11 src/marshmallow/fields.py** | 597 | 646| 419 | 12136 | 
| 43 | **11 src/marshmallow/fields.py** | 200 | 213| 128 | 12264 | 
| 44 | **11 src/marshmallow/fields.py** | 1535 | 1581| 395 | 12659 | 
| 45 | **11 src/marshmallow/fields.py** | 1275 | 1341| 494 | 13153 | 
| 46 | 12 examples/package_json_example.py | 20 | 17| 293 | 13446 | 
| 47 | 12 src/marshmallow/schema.py | 125 | 144| 213 | 13659 | 
| 48 | **12 src/marshmallow/fields.py** | 1218 | 1244| 204 | 13863 | 
| 49 | **12 src/marshmallow/fields.py** | 1427 | 1467| 281 | 14144 | 
| 50 | 12 examples/peewee_example.py | 0 | 47| 206 | 14350 | 
| 51 | 12 src/marshmallow/decorators.py | 68 | 95| 239 | 14589 | 
| 52 | 12 src/marshmallow/schema.py | 443 | 468| 231 | 14820 | 
| 53 | **12 src/marshmallow/fields.py** | 1400 | 1425| 186 | 15006 | 
| 54 | 12 src/marshmallow/schema.py | 731 | 753| 220 | 15226 | 
| 55 | **12 src/marshmallow/fields.py** | 264 | 288| 211 | 15437 | 
| 56 | 12 src/marshmallow/schema.py | 660 | 684| 316 | 15753 | 
| 57 | 12 src/marshmallow/schema.py | 146 | 183| 255 | 16008 | 
| 58 | **12 src/marshmallow/fields.py** | 1470 | 1483| 86 | 16094 | 
| 59 | **12 src/marshmallow/fields.py** | 1344 | 1398| 431 | 16525 | 
| 60 | 12 src/marshmallow/schema.py | 58 | 80| 196 | 16721 | 
| **-> 61 <-** | **12 src/marshmallow/fields.py** | 1066 | 1128| 515 | 17236 | 


### Hint

```
Thanks for reporting. I don't think I'll have time to look into this until the weekend. Would you like to send a PR? 
I'm afraid I don't have any time either, and I don't really have enough context on the `_bind_to_schema` process to make sure I'm not breaking stuff.
OK, no problem. @lafrech Will you have a chance to look into this?
I've found the patch below to fix the minimal example above, but I'm not really sure what it's missing out on or how to test it properly:
\`\`\`patch
diff --git a/src/marshmallow/fields.py b/src/marshmallow/fields.py
index 0b18e7d..700732e 100644
--- a/src/marshmallow/fields.py
+++ b/src/marshmallow/fields.py
@@ -1114,7 +1114,7 @@ class DateTime(Field):
         super()._bind_to_schema(field_name, schema)
         self.format = (
             self.format
-            or getattr(schema.opts, self.SCHEMA_OPTS_VAR_NAME)
+            or getattr(getattr(schema, "opts", None), self.SCHEMA_OPTS_VAR_NAME, None)
             or self.DEFAULT_FORMAT
         )
\`\`\`
    git difftool 3.0.0rc8 3.0.0rc9 src/marshmallow/fields.py

When reworking container stuff, I changed

\`\`\`py
        self.inner.parent = self
        self.inner.name = field_name
\`\`\`
into

\`\`\`py
        self.inner._bind_to_schema(field_name, self)
\`\`\`

AFAIR, I did this merely to avoid duplication. On second thought, I think it was the right thing to do, not only for duplication but to actually bind inner fields to the `Schema`.

Reverting this avoids the error but the inner field's `_bind_to_schema` method is not called so I'm not sure it is desirable.

I think we really mean to call that method, not only in this case but also generally.

Changing

\`\`\`py
            or getattr(schema.opts, self.SCHEMA_OPTS_VAR_NAME)
\`\`\`

into

\`\`\`py
            or getattr(self.root.opts, self.SCHEMA_OPTS_VAR_NAME)
\`\`\`

might be a better fix. Can anyone confirm (@sloria, @deckar01)?

The fix in https://github.com/marshmallow-code/marshmallow/issues/1357#issuecomment-523465528 removes the error but also the feature: `DateTime` fields buried into container fields won't respect the format set in the `Schema`.

I didn't double-check that but AFAIU, the change I mentioned above (in container stuff rework) was the right thing to do. The feature was already broken (format set in `Schema` not respected if `DateTime` field in container field) and that's just one of the issues that may arise due to the inner field not being bound to the `Schema`. But I may be wrong.
On quick glance, your analysis and fix look correct @lafrech 
Let's do that, then.

Not much time either. The first who gets the time can do it.

For the non-reg tests :

1/ a test that checks the format set in the schema is respected if the `DateTime` field is in a container field

2/ a set of tests asserting the `_bind_to_schema` method of inner fields `List`, `Dict`, `Tuple` is called from container fields (we can use `DateTime` with the same test case for that)

Perhaps 1/ is useless if 2/ is done.
```

## Patch

```diff
diff --git a/src/marshmallow/fields.py b/src/marshmallow/fields.py
--- a/src/marshmallow/fields.py
+++ b/src/marshmallow/fields.py
@@ -1114,7 +1114,7 @@ def _bind_to_schema(self, field_name, schema):
         super()._bind_to_schema(field_name, schema)
         self.format = (
             self.format
-            or getattr(schema.opts, self.SCHEMA_OPTS_VAR_NAME)
+            or getattr(self.root.opts, self.SCHEMA_OPTS_VAR_NAME)
             or self.DEFAULT_FORMAT
         )
 

```

## Test Patch

```diff
diff --git a/tests/test_fields.py b/tests/test_fields.py
--- a/tests/test_fields.py
+++ b/tests/test_fields.py
@@ -169,6 +169,20 @@ class OtherSchema(MySchema):
         assert schema2.fields["foo"].key_field.root == schema2
         assert schema2.fields["foo"].value_field.root == schema2
 
+    # Regression test for https://github.com/marshmallow-code/marshmallow/issues/1357
+    def test_datetime_list_inner_format(self, schema):
+        class MySchema(Schema):
+            foo = fields.List(fields.DateTime())
+            bar = fields.Tuple((fields.DateTime(),))
+
+            class Meta:
+                datetimeformat = "iso8601"
+                dateformat = "iso8601"
+
+        schema = MySchema()
+        assert schema.fields["foo"].inner.format == "iso8601"
+        assert schema.fields["bar"].tuple_fields[0].format == "iso8601"
+
 
 class TestMetadata:
     @pytest.mark.parametrize("FieldClass", ALL_FIELDS)

```


## Code snippets

### 1 - src/marshmallow/__init__.py:

```python
from marshmallow.schema import Schema, SchemaOpts

from . import fields
from marshmallow.decorators import (
    pre_dump,
    post_dump,
    pre_load,
    post_load,
    validates,
    validates_schema,
)
from marshmallow.utils import EXCLUDE, INCLUDE, RAISE, pprint, missing
from marshmallow.exceptions import ValidationError
from distutils.version import LooseVersion

__version__ = "3.0.0"
__version_info__ = tuple(LooseVersion(__version__).version)
__all__ = [
    "EXCLUDE",
    "INCLUDE",
    "RAISE",
    "Schema",
    "SchemaOpts",
    "fields",
    "validates",
    "validates_schema",
    "pre_dump",
    "post_dump",
    "pre_load",
    "post_load",
    "pprint",
    "ValidationError",
    "missing",
]
```
### 2 - src/marshmallow/schema.py:

Start line: 879, End line: 947

```python
class BaseSchema(base.SchemaABC):

    def _init_fields(self):
        """Update fields based on schema options."""
        if self.opts.fields:
            available_field_names = self.set_class(self.opts.fields)
        else:
            available_field_names = self.set_class(self.declared_fields.keys())
            if self.opts.additional:
                available_field_names |= self.set_class(self.opts.additional)

        invalid_fields = self.set_class()

        if self.only is not None:
            # Return only fields specified in only option
            field_names = self.set_class(self.only)

            invalid_fields |= field_names - available_field_names
        else:
            field_names = available_field_names

        # If "exclude" option or param is specified, remove those fields.
        if self.exclude:
            # Note that this isn't available_field_names, since we want to
            # apply "only" for the actual calculation.
            field_names = field_names - self.exclude
            invalid_fields |= self.exclude - available_field_names

        if invalid_fields:
            message = "Invalid fields for {}: {}.".format(self, invalid_fields)
            raise ValueError(message)

        fields_dict = self.dict_class()
        for field_name in field_names:
            field_obj = self.declared_fields.get(field_name, ma_fields.Inferred())
            self._bind_field(field_name, field_obj)
            fields_dict[field_name] = field_obj

        dump_data_keys = [
            obj.data_key or name
            for name, obj in fields_dict.items()
            if not obj.load_only
        ]
        if len(dump_data_keys) != len(set(dump_data_keys)):
            data_keys_duplicates = {
                x for x in dump_data_keys if dump_data_keys.count(x) > 1
            }
            raise ValueError(
                "The data_key argument for one or more fields collides "
                "with another field's name or data_key argument. "
                "Check the following field names and "
                "data_key arguments: {}".format(list(data_keys_duplicates))
            )

        load_attributes = [
            obj.attribute or name
            for name, obj in fields_dict.items()
            if not obj.dump_only
        ]
        if len(load_attributes) != len(set(load_attributes)):
            attributes_duplicates = {
                x for x in load_attributes if load_attributes.count(x) > 1
            }
            raise ValueError(
                "The attribute argument for one or more fields collides "
                "with another field's name or attribute argument. "
                "Check the following field names and "
                "attribute arguments: {}".format(list(attributes_duplicates))
            )

        return fields_dict
```
### 3 - src/marshmallow/fields.py:

Start line: 407, End line: 461

```python
class Nested(Field):

    default_error_messages = {"type": "Invalid type."}

    def __init__(
        self, nested, *, default=missing_, exclude=tuple(), only=None, **kwargs
    ):
        # Raise error if only or exclude is passed as string, not list of strings
        if only is not None and not is_collection(only):
            raise StringNotCollectionError('"only" should be a collection of strings.')
        if exclude is not None and not is_collection(exclude):
            raise StringNotCollectionError(
                '"exclude" should be a collection of strings.'
            )
        self.nested = nested
        self.only = only
        self.exclude = exclude
        self.many = kwargs.get("many", False)
        self.unknown = kwargs.get("unknown")
        self._schema = None  # Cached Schema instance
        super().__init__(default=default, **kwargs)
```
### 4 - src/marshmallow/fields.py:

Start line: 1130, End line: 1152

```python
class DateTime(Field):

    def _deserialize(self, value, attr, data, **kwargs):
        if not value:  # Falsy values, e.g. '', None, [] are not valid
            raise self.make_error("invalid", input=value, obj_type=self.OBJ_TYPE)
        data_format = self.format or self.DEFAULT_FORMAT
        func = self.DESERIALIZATION_FUNCS.get(data_format)
        if func:
            try:
                return func(value)
            except (TypeError, AttributeError, ValueError) as error:
                raise self.make_error(
                    "invalid", input=value, obj_type=self.OBJ_TYPE
                ) from error
        else:
            try:
                return self._make_object_from_format(value, data_format)
            except (TypeError, AttributeError, ValueError) as error:
                raise self.make_error(
                    "invalid", input=value, obj_type=self.OBJ_TYPE
                ) from error

    @staticmethod
    def _make_object_from_format(value, data_format):
        return dt.datetime.strptime(value, data_format)
```
### 5 - src/marshmallow/fields.py:

Start line: 669, End line: 728

```python
class Tuple(Field):
    """A tuple field, composed of a fixed number of other `Field` classes or
    instances

    Example: ::

        row = Tuple((fields.String(), fields.Integer(), fields.Float()))

    .. note::
        Because of the structured nature of `collections.namedtuple` and
        `typing.NamedTuple`, using a Schema within a Nested field for them is
        more appropriate than using a `Tuple` field.

    :param Iterable[Field] tuple_fields: An iterable of field classes or
        instances.
    :param kwargs: The same keyword arguments that :class:`Field` receives.

    .. versionadded:: 3.0.0rc4
    """

    default_error_messages = {"invalid": "Not a valid tuple."}

    def __init__(self, tuple_fields, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not utils.is_collection(tuple_fields):
            raise ValueError(
                "tuple_fields must be an iterable of Field classes or " "instances."
            )

        try:
            self.tuple_fields = [
                resolve_field_instance(cls_or_instance)
                for cls_or_instance in tuple_fields
            ]
        except FieldInstanceResolutionError as error:
            raise ValueError(
                'Elements of "tuple_fields" must be subclasses or '
                "instances of marshmallow.base.FieldABC."
            ) from error

        self.validate_length = Length(equal=len(self.tuple_fields))

    def _bind_to_schema(self, field_name, schema):
        super()._bind_to_schema(field_name, schema)
        new_tuple_fields = []
        for field in self.tuple_fields:
            field = copy.deepcopy(field)
            field._bind_to_schema(field_name, self)
            new_tuple_fields.append(field)

        self.tuple_fields = new_tuple_fields

    def _serialize(self, value, attr, obj, **kwargs):
        if value is None:
            return None

        return tuple(
            field._serialize(each, attr, obj, **kwargs)
            for field, each in zip(self.tuple_fields, value)
        )
```
### 6 - src/marshmallow/schema.py:

Start line: 352, End line: 410

```python
class BaseSchema(base.SchemaABC):

    def __init__(
        self,
        *,
        only=None,
        exclude=(),
        many=False,
        context=None,
        load_only=(),
        dump_only=(),
        partial=False,
        unknown=None
    ):
        # Raise error if only or exclude is passed as string, not list of strings
        if only is not None and not is_collection(only):
            raise StringNotCollectionError('"only" should be a list of strings')
        if exclude is not None and not is_collection(exclude):
            raise StringNotCollectionError('"exclude" should be a list of strings')
        # copy declared fields from metaclass
        self.declared_fields = copy.deepcopy(self._declared_fields)
        self.many = many
        self.only = only
        self.exclude = set(self.opts.exclude) | set(exclude)
        self.ordered = self.opts.ordered
        self.load_only = set(load_only) or set(self.opts.load_only)
        self.dump_only = set(dump_only) or set(self.opts.dump_only)
        self.partial = partial
        self.unknown = unknown or self.opts.unknown
        self.context = context or {}
        self._normalize_nested_options()
        #: Dictionary mapping field_names -> :class:`Field` objects
        self.fields = self._init_fields()
        self.dump_fields, self.load_fields = self.dict_class(), self.dict_class()
        for field_name, field_obj in self.fields.items():
            if field_obj.load_only:
                self.load_fields[field_name] = field_obj
            elif field_obj.dump_only:
                self.dump_fields[field_name] = field_obj
            else:
                self.load_fields[field_name] = field_obj
                self.dump_fields[field_name] = field_obj
        messages = {}
        messages.update(self._default_error_messages)
        for cls in reversed(self.__class__.__mro__):
            messages.update(getattr(cls, "error_messages", {}))
        messages.update(self.error_messages or {})
        self.error_messages = messages

    def __repr__(self):
        return "<{ClassName}(many={self.many})>".format(
            ClassName=self.__class__.__name__, self=self
        )

    @property
    def dict_class(self):
        return OrderedDict if self.ordered else dict

    @property
    def set_class(self):
        return OrderedSet if self.ordered else set
```
### 7 - src/marshmallow/schema.py:

Start line: 224, End line: 306

```python
class BaseSchema(base.SchemaABC):
    """Base schema class with which to define custom schemas.

    Example usage:

    .. code-block:: python

        import datetime as dt
        from marshmallow import Schema, fields

        class Album:
            def __init__(self, title, release_date):
                self.title = title
                self.release_date = release_date

        class AlbumSchema(Schema):
            title = fields.Str()
            release_date = fields.Date()

        # Or, equivalently
        class AlbumSchema2(Schema):
            class Meta:
                fields = ("title", "release_date")

        album = Album("Beggars Banquet", dt.date(1968, 12, 6))
        schema = AlbumSchema()
        data = schema.dump(album)
        data  # {'release_date': '1968-12-06', 'title': 'Beggars Banquet'}

    :param tuple|list only: Whitelist of the declared fields to select when
        instantiating the Schema. If None, all fields are used. Nested fields
        can be represented with dot delimiters.
    :param tuple|list exclude: Blacklist of the declared fields to exclude
        when instantiating the Schema. If a field appears in both `only` and
        `exclude`, it is not used. Nested fields can be represented with dot
        delimiters.
    :param bool many: Should be set to `True` if ``obj`` is a collection
        so that the object will be serialized to a list.
    :param dict context: Optional context passed to :class:`fields.Method` and
        :class:`fields.Function` fields.
    :param tuple|list load_only: Fields to skip during serialization (write-only fields)
    :param tuple|list dump_only: Fields to skip during deserialization (read-only fields)
    :param bool|tuple partial: Whether to ignore missing fields and not require
        any fields declared. Propagates down to ``Nested`` fields as well. If
        its value is an iterable, only missing fields listed in that iterable
        will be ignored. Use dot delimiters to specify nested fields.
    :param unknown: Whether to exclude, include, or raise an error for unknown
        fields in the data. Use `EXCLUDE`, `INCLUDE` or `RAISE`.

    .. versionchanged:: 3.0.0
        `prefix` parameter removed.

    .. versionchanged:: 2.0.0
        `__validators__`, `__preprocessors__`, and `__data_handlers__` are removed in favor of
        `marshmallow.decorators.validates_schema`,
        `marshmallow.decorators.pre_load` and `marshmallow.decorators.post_dump`.
        `__accessor__` and `__error_handler__` are deprecated. Implement the
        `handle_error` and `get_attribute` methods instead.
    """

    TYPE_MAPPING = {
        str: ma_fields.String,
        bytes: ma_fields.String,
        dt.datetime: ma_fields.DateTime,
        float: ma_fields.Float,
        bool: ma_fields.Boolean,
        tuple: ma_fields.Raw,
        list: ma_fields.Raw,
        set: ma_fields.Raw,
        int: ma_fields.Integer,
        uuid.UUID: ma_fields.UUID,
        dt.time: ma_fields.Time,
        dt.date: ma_fields.Date,
        dt.timedelta: ma_fields.TimeDelta,
        decimal.Decimal: ma_fields.Decimal,
    }
    #: Overrides for default schema-level error messages
    error_messages = {}

    _default_error_messages = {
        "type": "Invalid input type.",
        "unknown": "Unknown field.",
    }
```
### 8 - src/marshmallow/fields.py:

Start line: 501, End line: 531

```python
class Nested(Field):

    def _nested_normalized_option(self, option_name):
        nested_field = "%s." % self.name
        return [
            field.split(nested_field, 1)[1]
            for field in getattr(self.root, option_name, set())
            if field.startswith(nested_field)
        ]

    def _serialize(self, nested_obj, attr, obj, many=False, **kwargs):
        # Load up the schema first. This allows a RegistryError to be raised
        # if an invalid schema name was passed
        schema = self.schema
        if nested_obj is None:
            return None
        return schema.dump(nested_obj, many=self.many or many)

    def _test_collection(self, value, many=False):
        many = self.many or many
        if many and not utils.is_collection(value):
            raise self.make_error("type", input=value, type=value.__class__.__name__)

    def _load(self, value, data, partial=None, many=False):
        try:
            valid_data = self.schema.load(
                value, unknown=self.unknown, partial=partial, many=self.many or many
            )
        except ValidationError as error:
            raise ValidationError(
                error.messages, valid_data=error.valid_data
            ) from error
        return valid_data
```
### 9 - src/marshmallow/schema.py:

Start line: 186, End line: 221

```python
class SchemaOpts:
    """class Meta options for the :class:`Schema`. Defines defaults."""

    def __init__(self, meta, ordered=False):
        self.fields = getattr(meta, "fields", ())
        if not isinstance(self.fields, (list, tuple)):
            raise ValueError("`fields` option must be a list or tuple.")
        self.additional = getattr(meta, "additional", ())
        if not isinstance(self.additional, (list, tuple)):
            raise ValueError("`additional` option must be a list or tuple.")
        if self.fields and self.additional:
            raise ValueError(
                "Cannot set both `fields` and `additional` options"
                " for the same Schema."
            )
        self.exclude = getattr(meta, "exclude", ())
        if not isinstance(self.exclude, (list, tuple)):
            raise ValueError("`exclude` must be a list or tuple.")
        self.dateformat = getattr(meta, "dateformat", None)
        self.datetimeformat = getattr(meta, "datetimeformat", None)
        if hasattr(meta, "json_module"):
            warnings.warn(
                "The json_module class Meta option is deprecated. Use render_module instead.",
                DeprecationWarning,
            )
            render_module = getattr(meta, "json_module", json)
        else:
            render_module = json
        self.render_module = getattr(meta, "render_module", render_module)
        self.ordered = getattr(meta, "ordered", ordered)
        self.index_errors = getattr(meta, "index_errors", True)
        self.include = getattr(meta, "include", {})
        self.load_only = getattr(meta, "load_only", ())
        self.dump_only = getattr(meta, "dump_only", ())
        self.unknown = getattr(meta, "unknown", RAISE)
        self.register = getattr(meta, "register", True)
```
### 10 - examples/flask_example.py:

Start line: 47, End line: 64

```python
class QuoteSchema(Schema):
    id = fields.Int(dump_only=True)
    author = fields.Nested(AuthorSchema, validate=must_not_be_blank)
    content = fields.Str(required=True, validate=must_not_be_blank)
    posted_at = fields.DateTime(dump_only=True)
    @pre_load
    def process_author(self, data, **kwargs):
        author_name = data.get("author")
        if author_name:
            first, last = author_name.split(" ")
            author_dict = dict(first=first, last=last)
        else:
            author_dict = {}
        data["author"] = author_dict
        return data
```
### 12 - src/marshmallow/fields.py:

Start line: 463, End line: 499

```python
class Nested(Field):

    @property
    def schema(self):
        """The nested Schema object.

        .. versionchanged:: 1.0.0
            Renamed from `serializer` to `schema`.
        """
        if not self._schema:
            # Inherit context from parent.
            context = getattr(self.parent, "context", {})
            if isinstance(self.nested, SchemaABC):
                self._schema = self.nested
                self._schema.context.update(context)
            else:
                if isinstance(self.nested, type) and issubclass(self.nested, SchemaABC):
                    schema_class = self.nested
                elif not isinstance(self.nested, (str, bytes)):
                    raise ValueError(
                        "Nested fields must be passed a "
                        "Schema, not {}.".format(self.nested.__class__)
                    )
                elif self.nested == "self":
                    ret = self
                    while not isinstance(ret, SchemaABC):
                        ret = ret.parent
                    schema_class = ret.__class__
                else:
                    schema_class = class_registry.get_class(self.nested)
                self._schema = schema_class(
                    many=self.many,
                    only=self.only,
                    exclude=self.exclude,
                    context=context,
                    load_only=self._nested_normalized_option("load_only"),
                    dump_only=self._nested_normalized_option("dump_only"),
                )
        return self._schema
```
### 14 - src/marshmallow/fields.py:

Start line: 382, End line: 404

```python
class Field(FieldABC):

    # Properties

    @property
    def context(self):
        """The context dictionary for the parent :class:`Schema`."""
        return self.parent.context

    @property
    def root(self):
        """Reference to the `Schema` that this field belongs to even if it is buried in a
        container field (e.g. `List`).
        Return `None` for unbound fields.
        """
        ret = self
        while hasattr(ret, "parent"):
            ret = ret.parent
        return ret if isinstance(ret, SchemaABC) else None


class Raw(Field):
    """Field that applies no formatting or validation."""

    pass
```
### 15 - src/marshmallow/fields.py:

Start line: 648, End line: 666

```python
class List(Field):

    def _deserialize(self, value, attr, data, **kwargs):
        if not utils.is_collection(value):
            raise self.make_error("invalid")
        # Optimize loading a list of Nested objects by calling load(many=True)
        if isinstance(self.inner, Nested) and not self.inner.many:
            return self.inner.deserialize(value, many=True, **kwargs)

        result = []
        errors = {}
        for idx, each in enumerate(value):
            try:
                result.append(self.inner.deserialize(each, **kwargs))
            except ValidationError as error:
                if error.valid_data is not None:
                    result.append(error.valid_data)
                errors.update({idx: error.messages})
        if errors:
            raise ValidationError(errors, valid_data=result)
        return result
```
### 22 - src/marshmallow/fields.py:

Start line: 730, End line: 749

```python
class Tuple(Field):

    def _deserialize(self, value, attr, data, **kwargs):
        if not utils.is_collection(value):
            raise self.make_error("invalid")

        self.validate_length(value)

        result = []
        errors = {}

        for idx, (field, each) in enumerate(zip(self.tuple_fields, value)):
            try:
                result.append(field.deserialize(each, **kwargs))
            except ValidationError as error:
                if error.valid_data is not None:
                    result.append(error.valid_data)
                errors.update({idx: error.messages})
        if errors:
            raise ValidationError(errors, valid_data=result)

        return tuple(result)
```
### 23 - src/marshmallow/fields.py:

```python
"""Field classes for various types of data."""

import collections
import copy
import datetime as dt
import numbers
import uuid
import decimal
import math
import warnings
from collections.abc import Mapping as _Mapping

from marshmallow import validate, utils, class_registry
from marshmallow.base import FieldABC, SchemaABC
from marshmallow.utils import (
    is_collection,
    missing as missing_,
    resolve_field_instance,
    is_aware,
)
from marshmallow.exceptions import (
    ValidationError,
    StringNotCollectionError,
    FieldInstanceResolutionError,
)
from marshmallow.validate import Validator, Length

__all__ = [
    "Field",
    "Raw",
    "Nested",
    "Mapping",
    "Dict",
    "List",
    "Tuple",
    "String",
    "UUID",
    "Number",
    "Integer",
    "Decimal",
    "Boolean",
    "Float",
    "DateTime",
    "NaiveDateTime",
    "AwareDateTime",
    "Time",
    "Date",
    "TimeDelta",
    "Url",
    "URL",
    "Email",
    "Method",
    "Function",
    "Str",
    "Bool",
    "Int",
    "Constant",
    "Pluck",
]

MISSING_ERROR_MESSAGE = (
    "ValidationError raised by `{class_name}`, but error key `{key}` does "
    "not exist in the `error_messages` dictionary."
)
```
### 28 - src/marshmallow/fields.py:

Start line: 546, End line: 594

```python
class Pluck(Nested):
    """Allows you to replace nested data with one of the data's fields.

    Example: ::

        from marshmallow import Schema, fields

        class ArtistSchema(Schema):
            id = fields.Int()
            name = fields.Str()

        class AlbumSchema(Schema):
            artist = fields.Pluck(ArtistSchema, 'id')


        in_data = {'artist': 42}
        loaded = AlbumSchema().load(in_data) # => {'artist': {'id': 42}}
        dumped = AlbumSchema().dump(loaded)  # => {'artist': 42}

    :param Schema nested: The Schema class or class name (string)
        to nest, or ``"self"`` to nest the :class:`Schema` within itself.
    :param str field_name: The key to pluck a value from.
    :param kwargs: The same keyword arguments that :class:`Nested` receives.
    """

    def __init__(self, nested, field_name, **kwargs):
        super().__init__(nested, only=(field_name,), **kwargs)
        self.field_name = field_name

    @property
    def _field_data_key(self):
        only_field = self.schema.fields[self.field_name]
        return only_field.data_key or self.field_name

    def _serialize(self, nested_obj, attr, obj, **kwargs):
        ret = super()._serialize(nested_obj, attr, obj, **kwargs)
        if ret is None:
            return None
        if self.many:
            return utils.pluck(ret, key=self._field_data_key)
        return ret[self._field_data_key]

    def _deserialize(self, value, attr, data, partial=None, **kwargs):
        self._test_collection(value)
        if self.many:
            value = [{self._field_data_key: v} for v in value]
        else:
            value = {self._field_data_key: value}
        return self._load(value, data, partial=partial)
```
### 32 - src/marshmallow/fields.py:

Start line: 344, End line: 342

```python
class Field(FieldABC):

    # Methods for concrete classes to override.

    def _bind_to_schema(self, field_name, schema):
        """Update field with values from its parent schema. Called by
        :meth:`Schema._bind_field <marshmallow.Schema._bind_field>`.

        :param str field_name: Field name set in schema.
        :param Schema schema: Parent schema.
        """
        self.parent = self.parent or schema
        self.name = self.name or field_name

    def _serialize(self, value, attr, obj, **kwargs):
        """Serializes ``value`` to a basic Python datatype. Noop by default.
        Concrete :class:`Field` classes should implement this method.

        Example: ::

            class TitleCase(Field):
                def _serialize(self, value, attr, obj, **kwargs):
                    if not value:
                        return ''
                    return str(value).title()

        :param value: The value to be serialized.
        :param str attr: The attribute or key on the object to be serialized.
        :param object obj: The object the value was pulled from.
        :param dict kwargs: Field-specific keyword arguments.
        :return: The serialized value
        """
        return value
```
### 37 - src/marshmallow/fields.py:

Start line: 126, End line: 198

```python
class Field(FieldABC):

    # Some fields, such as Method fields and Function fields, are not expected
    #  to exist as attributes on the objects to serialize. Set this to False
    #  for those fields
    _CHECK_ATTRIBUTE = True
    _creation_index = 0  # Used for sorting

    #: Default error messages for various kinds of errors. The keys in this dictionary
    #: are passed to `Field.fail`. The values are error messages passed to
    #: :exc:`marshmallow.exceptions.ValidationError`.
    default_error_messages = {
        "required": "Missing data for required field.",
        "null": "Field may not be null.",
        "validator_failed": "Invalid value.",
    }

    def __init__(
        self,
        *,
        default=missing_,
        missing=missing_,
        data_key=None,
        attribute=None,
        validate=None,
        required=False,
        allow_none=None,
        load_only=False,
        dump_only=False,
        error_messages=None,
        **metadata
    ):
        self.default = default
        self.attribute = attribute
        self.data_key = data_key
        self.validate = validate
        if utils.is_iterable_but_not_string(validate):
            if not utils.is_generator(validate):
                self.validators = validate
            else:
                self.validators = list(validate)
        elif callable(validate):
            self.validators = [validate]
        elif validate is None:
            self.validators = []
        else:
            raise ValueError(
                "The 'validate' parameter must be a callable "
                "or a collection of callables."
            )

        # If missing=None, None should be considered valid by default
        if allow_none is None:
            if missing is None:
                self.allow_none = True
            else:
                self.allow_none = False
        else:
            self.allow_none = allow_none
        self.load_only = load_only
        self.dump_only = dump_only
        if required is True and missing is not missing_:
            raise ValueError("'missing' must not be set for required fields.")
        self.required = required
        self.missing = missing
        self.metadata = metadata
        self._creation_index = Field._creation_index
        Field._creation_index += 1

        # Collect default error message from self and parent classes
        messages = {}
        for cls in reversed(self.__class__.__mro__):
            messages.update(getattr(cls, "default_error_messages", {}))
        messages.update(error_messages or {})
        self.error_messages = messages
```
### 38 - src/marshmallow/fields.py:

Start line: 533, End line: 543

```python
class Nested(Field):

    def _deserialize(self, value, attr, data, partial=None, many=False, **kwargs):
        """Same as :meth:`Field._deserialize` with additional ``partial`` argument.

        :param bool|tuple partial: For nested schemas, the ``partial``
            parameter passed to `Schema.load`.

        .. versionchanged:: 3.0.0
            Add ``partial`` parameter.
        """
        self._test_collection(value, many=many)
        return self._load(value, data, partial=partial, many=many)
```
### 42 - src/marshmallow/fields.py:

Start line: 597, End line: 646

```python
class List(Field):
    """A list field, composed with another `Field` class or
    instance.

    Example: ::

        numbers = fields.List(fields.Float())

    :param Field cls_or_instance: A field class or instance.
    :param bool default: Default value for serialization.
    :param kwargs: The same keyword arguments that :class:`Field` receives.

    .. versionchanged:: 2.0.0
        The ``allow_none`` parameter now applies to deserialization and
        has the same semantics as the other fields.

    .. versionchanged:: 3.0.0rc9
        Does not serialize scalar values to single-item lists.
    """

    default_error_messages = {"invalid": "Not a valid list."}

    def __init__(self, cls_or_instance, **kwargs):
        super().__init__(**kwargs)
        try:
            self.inner = resolve_field_instance(cls_or_instance)
        except FieldInstanceResolutionError as error:
            raise ValueError(
                "The list elements must be a subclass or instance of "
                "marshmallow.base.FieldABC."
            ) from error
        if isinstance(self.inner, Nested):
            self.only = self.inner.only
            self.exclude = self.inner.exclude

    def _bind_to_schema(self, field_name, schema):
        super()._bind_to_schema(field_name, schema)
        self.inner = copy.deepcopy(self.inner)
        self.inner._bind_to_schema(field_name, self)
        if isinstance(self.inner, Nested):
            self.inner.only = self.only
            self.inner.exclude = self.exclude

    def _serialize(self, value, attr, obj, **kwargs):
        if value is None:
            return None
        # Optimize dumping a list of Nested objects by calling dump(many=True)
        if isinstance(self.inner, Nested) and not self.inner.many:
            return self.inner._serialize(value, attr, obj, many=True, **kwargs)
        return [self.inner._serialize(each, attr, obj, **kwargs) for each in value]
```
### 43 - src/marshmallow/fields.py:

Start line: 200, End line: 213

```python
class Field(FieldABC):

    def __repr__(self):
        return (
            "<fields.{ClassName}(default={self.default!r}, "
            "attribute={self.attribute!r}, "
            "validate={self.validate}, required={self.required}, "
            "load_only={self.load_only}, dump_only={self.dump_only}, "
            "missing={self.missing}, allow_none={self.allow_none}, "
            "error_messages={self.error_messages})>".format(
                ClassName=self.__class__.__name__, self=self
            )
        )

    def __deepcopy__(self, memo):
        return copy.copy(self)
```
### 44 - src/marshmallow/fields.py:

Start line: 1535, End line: 1581

```python
class Method(Field):
    """A field that takes the value returned by a `Schema` method.

    :param str serialize: The name of the Schema method from which
        to retrieve the value. The method must take an argument ``obj``
        (in addition to self) that is the object to be serialized.
    :param str deserialize: Optional name of the Schema method for deserializing
        a value The method must take a single argument ``value``, which is the
        value to deserialize.

    .. versionchanged:: 2.0.0
        Removed optional ``context`` parameter on methods. Use ``self.context`` instead.

    .. versionchanged:: 2.3.0
        Deprecated ``method_name`` parameter in favor of ``serialize`` and allow
        ``serialize`` to not be passed at all.

    .. versionchanged:: 3.0.0
        Removed ``method_name`` parameter.
    """

    _CHECK_ATTRIBUTE = False

    def __init__(self, serialize=None, deserialize=None, **kwargs):
        # Set dump_only and load_only based on arguments
        kwargs["dump_only"] = bool(serialize) and not bool(deserialize)
        kwargs["load_only"] = bool(deserialize) and not bool(serialize)
        super().__init__(**kwargs)
        self.serialize_method_name = serialize
        self.deserialize_method_name = deserialize

    def _serialize(self, value, attr, obj, **kwargs):
        if not self.serialize_method_name:
            return missing_

        method = utils.callable_or_raise(
            getattr(self.parent, self.serialize_method_name, None)
        )
        return method(obj)

    def _deserialize(self, value, attr, data, **kwargs):
        if self.deserialize_method_name:
            method = utils.callable_or_raise(
                getattr(self.parent, self.deserialize_method_name, None)
            )
            return method(value)
        return value
```
### 45 - src/marshmallow/fields.py:

Start line: 1275, End line: 1341

```python
class TimeDelta(Field):
    """A field that (de)serializes a :class:`datetime.timedelta` object to an
    integer and vice versa. The integer can represent the number of days,
    seconds or microseconds.

    :param str precision: Influences how the integer is interpreted during
        (de)serialization. Must be 'days', 'seconds', 'microseconds',
        'milliseconds', 'minutes', 'hours' or 'weeks'.
    :param kwargs: The same keyword arguments that :class:`Field` receives.

    .. versionchanged:: 2.0.0
        Always serializes to an integer value to avoid rounding errors.
        Add `precision` parameter.
    """

    DAYS = "days"
    SECONDS = "seconds"
    MICROSECONDS = "microseconds"
    MILLISECONDS = "milliseconds"
    MINUTES = "minutes"
    HOURS = "hours"
    WEEKS = "weeks"

    default_error_messages = {
        "invalid": "Not a valid period of time.",
        "format": "{input!r} cannot be formatted as a timedelta.",
    }

    def __init__(self, precision=SECONDS, **kwargs):
        precision = precision.lower()
        units = (
            self.DAYS,
            self.SECONDS,
            self.MICROSECONDS,
            self.MILLISECONDS,
            self.MINUTES,
            self.HOURS,
            self.WEEKS,
        )

        if precision not in units:
            msg = 'The precision must be {} or "{}".'.format(
                ", ".join(['"{}"'.format(each) for each in units[:-1]]), units[-1]
            )
            raise ValueError(msg)

        self.precision = precision
        super().__init__(**kwargs)

    def _serialize(self, value, attr, obj, **kwargs):
        if value is None:
            return None
        base_unit = dt.timedelta(**{self.precision: 1})
        return int(value.total_seconds() / base_unit.total_seconds())

    def _deserialize(self, value, attr, data, **kwargs):
        try:
            value = int(value)
        except (TypeError, ValueError) as error:
            raise self.make_error("invalid") from error

        kwargs = {self.precision: value}

        try:
            return dt.timedelta(**kwargs)
        except OverflowError as error:
            raise self.make_error("invalid") from error
```
### 48 - src/marshmallow/fields.py:

Start line: 1218, End line: 1244

```python
class Time(Field):
    """ISO8601-formatted time string.

    :param kwargs: The same keyword arguments that :class:`Field` receives.
    """

    default_error_messages = {
        "invalid": "Not a valid time.",
        "format": '"{input}" cannot be formatted as a time.',
    }

    def _serialize(self, value, attr, obj, **kwargs):
        if value is None:
            return None
        ret = value.isoformat()
        if value.microsecond:
            return ret[:15]
        return ret

    def _deserialize(self, value, attr, data, **kwargs):
        """Deserialize an ISO8601-formatted time to a :class:`datetime.time` object."""
        if not value:  # falsy values are invalid
            raise self.make_error("invalid")
        try:
            return utils.from_iso_time(value)
        except (AttributeError, TypeError, ValueError) as error:
            raise self.make_error("invalid") from error
```
### 49 - src/marshmallow/fields.py:

Start line: 1427, End line: 1467

```python
class Mapping(Field):

    def _deserialize(self, value, attr, data, **kwargs):
        if not isinstance(value, _Mapping):
            raise self.make_error("invalid")
        if not self.value_field and not self.key_field:
            return value

        errors = collections.defaultdict(dict)

        #  Deserialize keys
        if self.key_field is None:
            keys = {k: k for k in value.keys()}
        else:
            keys = {}
            for key in value.keys():
                try:
                    keys[key] = self.key_field.deserialize(key, **kwargs)
                except ValidationError as error:
                    errors[key]["key"] = error.messages

        #  Deserialize values
        result = self.mapping_type()
        if self.value_field is None:
            for k, v in value.items():
                if k in keys:
                    result[keys[k]] = v
        else:
            for key, val in value.items():
                try:
                    deser_val = self.value_field.deserialize(val, **kwargs)
                except ValidationError as error:
                    errors[key]["value"] = error.messages
                    if error.valid_data is not None and key in keys:
                        result[keys[key]] = error.valid_data
                else:
                    if key in keys:
                        result[keys[key]] = deser_val

        if errors:
            raise ValidationError(errors, valid_data=result)

        return result
```
### 53 - src/marshmallow/fields.py:

Start line: 1400, End line: 1425

```python
class Mapping(Field):

    def _serialize(self, value, attr, obj, **kwargs):
        if value is None:
            return None
        if not self.value_field and not self.key_field:
            return value

        #  Serialize keys
        if self.key_field is None:
            keys = {k: k for k in value.keys()}
        else:
            keys = {
                k: self.key_field._serialize(k, None, None, **kwargs)
                for k in value.keys()
            }

        #  Serialize values
        result = self.mapping_type()
        if self.value_field is None:
            for k, v in value.items():
                if k in keys:
                    result[keys[k]] = v
        else:
            for k, v in value.items():
                result[keys[k]] = self.value_field._serialize(v, None, None, **kwargs)

        return result
```
### 55 - src/marshmallow/fields.py:

Start line: 264, End line: 288

```python
class Field(FieldABC):

    def fail(self, key: str, **kwargs):
        """Helper method that raises a `ValidationError` with an error message
        from ``self.error_messages``.

        .. deprecated:: 3.0.0
            Use `make_error <marshmallow.fields.Field.make_error>` instead.
        """
        warnings.warn(
            '`Field.fail` is deprecated. Use `raise self.make_error("{}", ...)` instead.'.format(
                key
            ),
            DeprecationWarning,
        )
        raise self.make_error(key=key, **kwargs)

    def _validate_missing(self, value):
        """Validate missing values. Raise a :exc:`ValidationError` if
        `value` should be considered missing.
        """
        if value is missing_:
            if hasattr(self, "required") and self.required:
                raise self.make_error("required")
        if value is None:
            if hasattr(self, "allow_none") and self.allow_none is not True:
                raise self.make_error("null")
```
### 58 - src/marshmallow/fields.py:

Start line: 1470, End line: 1483

```python
class Dict(Mapping):
    """A dict field. Supports dicts and dict-like objects. Extends
    Mapping with dict as the mapping_type.

    Example: ::

        numbers = fields.Dict(keys=fields.Str(), values=fields.Float())

    :param kwargs: The same keyword arguments that :class:`Mapping` receives.

    .. versionadded:: 2.1.0
    """

    mapping_type = dict
```
### 59 - src/marshmallow/fields.py:

Start line: 1344, End line: 1398

```python
class Mapping(Field):
    """An abstract class for objects with key-value pairs.

    :param Field keys: A field class or instance for dict keys.
    :param Field values: A field class or instance for dict values.
    :param kwargs: The same keyword arguments that :class:`Field` receives.

    .. note::
        When the structure of nested data is not known, you may omit the
        `keys` and `values` arguments to prevent content validation.

    .. versionadded:: 3.0.0rc4
    """

    mapping_type = dict
    default_error_messages = {"invalid": "Not a valid mapping type."}

    def __init__(self, keys=None, values=None, **kwargs):
        super().__init__(**kwargs)
        if keys is None:
            self.key_field = None
        else:
            try:
                self.key_field = resolve_field_instance(keys)
            except FieldInstanceResolutionError as error:
                raise ValueError(
                    '"keys" must be a subclass or instance of '
                    "marshmallow.base.FieldABC."
                ) from error

        if values is None:
            self.value_field = None
        else:
            try:
                self.value_field = resolve_field_instance(values)
            except FieldInstanceResolutionError as error:
                raise ValueError(
                    '"values" must be a subclass or instance of '
                    "marshmallow.base.FieldABC."
                ) from error
            if isinstance(self.value_field, Nested):
                self.only = self.value_field.only
                self.exclude = self.value_field.exclude

    def _bind_to_schema(self, field_name, schema):
        super()._bind_to_schema(field_name, schema)
        if self.value_field:
            self.value_field = copy.deepcopy(self.value_field)
            self.value_field._bind_to_schema(field_name, self)
        if isinstance(self.value_field, Nested):
            self.value_field.only = self.only
            self.value_field.exclude = self.exclude
        if self.key_field:
            self.key_field = copy.deepcopy(self.key_field)
            self.key_field._bind_to_schema(field_name, self)
```
### 61 - src/marshmallow/fields.py:

Start line: 1066, End line: 1128

```python
class DateTime(Field):
    """A formatted datetime string.

    Example: ``'2014-12-22T03:12:58.019077+00:00'``

    :param str format: Either ``"rfc"`` (for RFC822), ``"iso"`` (for ISO8601),
        or a date format string. If `None`, defaults to "iso".
    :param kwargs: The same keyword arguments that :class:`Field` receives.

    .. versionchanged:: 3.0.0rc9
        Does not modify timezone information on (de)serialization.
    """

    SERIALIZATION_FUNCS = {
        "iso": utils.isoformat,
        "iso8601": utils.isoformat,
        "rfc": utils.rfcformat,
        "rfc822": utils.rfcformat,
    }

    DESERIALIZATION_FUNCS = {
        "iso": utils.from_iso_datetime,
        "iso8601": utils.from_iso_datetime,
        "rfc": utils.from_rfc,
        "rfc822": utils.from_rfc,
    }

    DEFAULT_FORMAT = "iso"

    OBJ_TYPE = "datetime"

    SCHEMA_OPTS_VAR_NAME = "datetimeformat"

    default_error_messages = {
        "invalid": "Not a valid {obj_type}.",
        "invalid_awareness": "Not a valid {awareness} {obj_type}.",
        "format": '"{input}" cannot be formatted as a {obj_type}.',
    }

    def __init__(self, format=None, **kwargs):
        super().__init__(**kwargs)
        # Allow this to be None. It may be set later in the ``_serialize``
        # or ``_deserialize`` methods. This allows a Schema to dynamically set the
        # format, e.g. from a Meta option
        self.format = format

    def _bind_to_schema(self, field_name, schema):
        super()._bind_to_schema(field_name, schema)
        self.format = (
            self.format
            or getattr(schema.opts, self.SCHEMA_OPTS_VAR_NAME)
            or self.DEFAULT_FORMAT
        )

    def _serialize(self, value, attr, obj, **kwargs):
        if value is None:
            return None
        data_format = self.format or self.DEFAULT_FORMAT
        format_func = self.SERIALIZATION_FUNCS.get(data_format)
        if format_func:
            return format_func(value)
        else:
            return value.strftime(data_format)
```
