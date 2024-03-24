# marshmallow-code__marshmallow-1343

| **marshmallow-code/marshmallow** | `2be2d83a1a9a6d3d9b85804f3ab545cecc409bb0` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 571 |
| **Any found context length** | 571 |
| **Avg pos** | 4.0 |
| **Min pos** | 2 |
| **Max pos** | 2 |
| **Top file pos** | 1 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/src/marshmallow/schema.py b/src/marshmallow/schema.py
--- a/src/marshmallow/schema.py
+++ b/src/marshmallow/schema.py
@@ -877,7 +877,7 @@ def _invoke_field_validators(self, unmarshal, data, many):
                 for idx, item in enumerate(data):
                     try:
                         value = item[field_obj.attribute or field_name]
-                    except KeyError:
+                    except (KeyError, TypeError):
                         pass
                     else:
                         validated_value = unmarshal.call_and_store(
@@ -892,7 +892,7 @@ def _invoke_field_validators(self, unmarshal, data, many):
             else:
                 try:
                     value = data[field_obj.attribute or field_name]
-                except KeyError:
+                except (KeyError, TypeError):
                     pass
                 else:
                     validated_value = unmarshal.call_and_store(

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| src/marshmallow/schema.py | 880 | 882 | 2 | 1 | 571
| src/marshmallow/schema.py | 895 | 897 | 2 | 1 | 571


## Problem Statement

```
[version 2.20.0] TypeError: 'NoneType' object is not subscriptable
After update from version 2.19.5 to 2.20.0 I got error for code like:

\`\`\`python
from marshmallow import Schema, fields, validates


class Bar(Schema):
    value = fields.String()

    @validates('value')  # <- issue here
    def validate_value(self, value):
        pass


class Foo(Schema):
    bar = fields.Nested(Bar)


sch = Foo()

sch.validate({
    'bar': 'invalid',
})
\`\`\`

\`\`\`
Traceback (most recent call last):
  File "/_/bug_mschema.py", line 19, in <module>
    'bar': 'invalid',
  File "/_/env/lib/python3.7/site-packages/marshmallow/schema.py", line 628, in validate
    _, errors = self._do_load(data, many, partial=partial, postprocess=False)
  File "/_/env/lib/python3.7/site-packages/marshmallow/schema.py", line 670, in _do_load
    index_errors=self.opts.index_errors,
  File "/_/env/lib/python3.7/site-packages/marshmallow/marshalling.py", line 292, in deserialize
    index=(index if index_errors else None)
  File "/_/env/lib/python3.7/site-packages/marshmallow/marshalling.py", line 65, in call_and_store
    value = getter_func(data)
  File "/_/env/lib/python3.7/site-packages/marshmallow/marshalling.py", line 285, in <lambda>
    data
  File "/_/env/lib/python3.7/site-packages/marshmallow/fields.py", line 265, in deserialize
    output = self._deserialize(value, attr, data)
  File "/_/env/lib/python3.7/site-packages/marshmallow/fields.py", line 465, in _deserialize
    data, errors = self.schema.load(value)
  File "/_/env/lib/python3.7/site-packages/marshmallow/schema.py", line 588, in load
    result, errors = self._do_load(data, many, partial=partial, postprocess=True)
  File "/_/env/lib/python3.7/site-packages/marshmallow/schema.py", line 674, in _do_load
    self._invoke_field_validators(unmarshal, data=result, many=many)
  File "/_/env/lib/python3.7/site-packages/marshmallow/schema.py", line 894, in _invoke_field_validators
    value = data[field_obj.attribute or field_name]
TypeError: 'NoneType' object is not subscriptable
\`\`\`

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- |
| 1 | **1 src/marshmallow/schema.py** | 906 | 937| 262 | 262 | 
| **-> 2 <-** | **1 src/marshmallow/schema.py** | 862 | 904| 309 | 571 | 
| 3 | 2 src/marshmallow/__init__.py | 0 | 36| 202 | 773 | 
| 4 | **2 src/marshmallow/schema.py** | 341 | 416| 569 | 1342 | 
| 5 | **2 src/marshmallow/schema.py** | 750 | 778| 266 | 1608 | 
| 6 | 3 src/marshmallow/fields.py | 460 | 480| 170 | 1778 | 
| 7 | 3 src/marshmallow/fields.py | 253 | 278| 212 | 1990 | 
| 8 | **3 src/marshmallow/schema.py** | 939 | 968| 253 | 2243 | 
| 9 | 4 src/marshmallow/marshalling.py | 164 | 161| 360 | 2603 | 
| 10 | 5 examples/flask_example.py | 43 | 60| 129 | 2732 | 
| 11 | **5 src/marshmallow/schema.py** | 449 | 466| 154 | 2886 | 
| 12 | **5 src/marshmallow/schema.py** | 843 | 860| 231 | 3117 | 
| 13 | 5 examples/flask_example.py | 0 | 41| 276 | 3393 | 
| 14 | **5 src/marshmallow/schema.py** | 612 | 630| 190 | 3583 | 
| 15 | **5 src/marshmallow/schema.py** | 0 | 28| 245 | 3828 | 
| 16 | **5 src/marshmallow/schema.py** | 468 | 552| 627 | 4455 | 
| 17 | 6 src/marshmallow/exceptions.py | 10 | 7| 420 | 4875 | 
| 18 | 6 src/marshmallow/fields.py | 482 | 505| 183 | 5058 | 
| 19 | 6 src/marshmallow/fields.py | 0 | 53| 280 | 5338 | 
| 20 | 7 src/marshmallow/decorators.py | 0 | 68| 437 | 5775 | 
| 21 | 8 performance/benchmark.py | 32 | 47| 134 | 5909 | 
| 22 | 8 src/marshmallow/fields.py | 436 | 434| 288 | 6197 | 
| 23 | 8 src/marshmallow/fields.py | 337 | 384| 491 | 6688 | 
| 24 | 9 examples/peewee_example.py | 36 | 59| 171 | 6859 | 
| 25 | 9 src/marshmallow/decorators.py | 71 | 85| 171 | 7030 | 
| 26 | 9 examples/peewee_example.py | 62 | 84| 179 | 7209 | 
| 27 | 9 src/marshmallow/marshalling.py | 0 | 26| 130 | 7339 | 
| 28 | 9 src/marshmallow/fields.py | 116 | 185| 650 | 7989 | 
| 29 | 10 src/marshmallow/class_registry.py | 0 | 18| 124 | 8113 | 
| 30 | **10 src/marshmallow/schema.py** | 223 | 302| 731 | 8844 | 
| 31 | **10 src/marshmallow/schema.py** | 632 | 712| 629 | 9473 | 
| 32 | **10 src/marshmallow/schema.py** | 572 | 588| 215 | 9688 | 
| 33 | 10 src/marshmallow/fields.py | 187 | 205| 131 | 9819 | 
| 34 | 10 src/marshmallow/fields.py | 300 | 334| 261 | 10080 | 
| 35 | **10 src/marshmallow/schema.py** | 418 | 447| 244 | 10324 | 
| 36 | 10 src/marshmallow/marshalling.py | 209 | 308| 769 | 11093 | 
| 37 | 11 src/marshmallow/utils.py | 0 | 85| 571 | 11664 | 
| 38 | 11 src/marshmallow/marshalling.py | 29 | 49| 144 | 11808 | 
| 39 | **11 src/marshmallow/schema.py** | 554 | 570| 230 | 12038 | 
| 40 | 12 src/marshmallow/base.py | 0 | 45| 206 | 12244 | 
| 41 | 13 src/marshmallow/validate.py | 145 | 167| 145 | 12389 | 
| 42 | **13 src/marshmallow/schema.py** | 76 | 81| 439 | 12828 | 


### Hint

```
Thanks for reporting. I was able to reproduce this on 2.20.0. This is likely a regression from https://github.com/marshmallow-code/marshmallow/pull/1323 . I don't have time to look into it now. Would appreciate a PR.
```

## Patch

```diff
diff --git a/src/marshmallow/schema.py b/src/marshmallow/schema.py
--- a/src/marshmallow/schema.py
+++ b/src/marshmallow/schema.py
@@ -877,7 +877,7 @@ def _invoke_field_validators(self, unmarshal, data, many):
                 for idx, item in enumerate(data):
                     try:
                         value = item[field_obj.attribute or field_name]
-                    except KeyError:
+                    except (KeyError, TypeError):
                         pass
                     else:
                         validated_value = unmarshal.call_and_store(
@@ -892,7 +892,7 @@ def _invoke_field_validators(self, unmarshal, data, many):
             else:
                 try:
                     value = data[field_obj.attribute or field_name]
-                except KeyError:
+                except (KeyError, TypeError):
                     pass
                 else:
                     validated_value = unmarshal.call_and_store(

```

## Test Patch

```diff
diff --git a/tests/test_marshalling.py b/tests/test_marshalling.py
--- a/tests/test_marshalling.py
+++ b/tests/test_marshalling.py
@@ -2,7 +2,7 @@
 
 import pytest
 
-from marshmallow import fields, Schema
+from marshmallow import fields, Schema, validates
 from marshmallow.marshalling import Marshaller, Unmarshaller, missing
 from marshmallow.exceptions import ValidationError
 
@@ -283,3 +283,24 @@ class TestSchema(Schema):
 
             assert result is None
             assert excinfo.value.messages == {'foo': {'_schema': ['Invalid input type.']}}
+
+    # Regression test for https://github.com/marshmallow-code/marshmallow/issues/1342
+    def test_deserialize_wrong_nested_type_with_validates_method(self, unmarshal):
+        class TestSchema(Schema):
+            value = fields.String()
+
+            @validates('value')
+            def validate_value(self, value):
+                pass
+
+        data = {
+            'foo': 'not what we need'
+        }
+        fields_dict = {
+            'foo': fields.Nested(TestSchema, required=True)
+        }
+        with pytest.raises(ValidationError) as excinfo:
+            result = unmarshal.deserialize(data, fields_dict)
+
+            assert result is None
+            assert excinfo.value.messages == {'foo': {'_schema': ['Invalid input type.']}}

```


## Code snippets

### 1 - src/marshmallow/schema.py:

Start line: 906, End line: 937

```python
class BaseSchema(base.SchemaABC):

    def _invoke_validators(
            self, unmarshal, pass_many, data, original_data, many, field_errors=False):
        errors = {}
        for attr_name in self.__processors__[(VALIDATES_SCHEMA, pass_many)]:
            validator = getattr(self, attr_name)
            validator_kwargs = validator.__marshmallow_kwargs__[(VALIDATES_SCHEMA, pass_many)]
            pass_original = validator_kwargs.get('pass_original', False)

            skip_on_field_errors = validator_kwargs['skip_on_field_errors']
            if skip_on_field_errors and field_errors:
                continue

            if pass_many:
                validator = functools.partial(validator, many=many)
            if many and not pass_many:
                for idx, item in enumerate(data):
                    try:
                        unmarshal.run_validator(validator,
                                                item, original_data, self.fields, many=many,
                                                index=idx, pass_original=pass_original)
                    except ValidationError as err:
                        errors.update(err.messages)
            else:
                try:
                    unmarshal.run_validator(validator,
                                            data, original_data, self.fields, many=many,
                                            pass_original=pass_original)
                except ValidationError as err:
                    errors.update(err.messages)
        if errors:
            raise ValidationError(errors)
        return None
```
### 2 - src/marshmallow/schema.py:

Start line: 862, End line: 904

```python
class BaseSchema(base.SchemaABC):

    def _invoke_field_validators(self, unmarshal, data, many):
        for attr_name in self.__processors__[(VALIDATES, False)]:
            validator = getattr(self, attr_name)
            validator_kwargs = validator.__marshmallow_kwargs__[(VALIDATES, False)]
            field_name = validator_kwargs['field_name']

            try:
                field_obj = self.fields[field_name]
            except KeyError:
                if field_name in self.declared_fields:
                    continue
                raise ValueError('"{0}" field does not exist.'.format(field_name))

            if many:
                for idx, item in enumerate(data):
                    try:
                        value = item[field_obj.attribute or field_name]
                    except KeyError:
                        pass
                    else:
                        validated_value = unmarshal.call_and_store(
                            getter_func=validator,
                            data=value,
                            field_name=field_obj.load_from or field_name,
                            field_obj=field_obj,
                            index=(idx if self.opts.index_errors else None)
                        )
                        if validated_value is missing:
                            data[idx].pop(field_name, None)
            else:
                try:
                    value = data[field_obj.attribute or field_name]
                except KeyError:
                    pass
                else:
                    validated_value = unmarshal.call_and_store(
                        getter_func=validator,
                        data=value,
                        field_name=field_obj.load_from or field_name,
                        field_obj=field_obj
                    )
                    if validated_value is missing:
                        data.pop(field_name, None)
```
### 3 - src/marshmallow/__init__.py:

```python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

from marshmallow.schema import (
    Schema,
    SchemaOpts,
    MarshalResult,
    UnmarshalResult,
)
from . import fields
from marshmallow.decorators import (
    pre_dump, post_dump, pre_load, post_load, validates, validates_schema
)
from marshmallow.utils import pprint, missing
from marshmallow.exceptions import ValidationError
from distutils.version import LooseVersion

__version__ = '2.20.0'
__version_info__ = tuple(LooseVersion(__version__).version)
__author__ = 'Steven Loria'
__all__ = [
    'Schema',
    'SchemaOpts',
    'fields',
    'validates',
    'validates_schema',
    'pre_dump',
    'post_dump',
    'pre_load',
    'post_load',
    'pprint',
    'MarshalResult',
    'UnmarshalResult',
    'ValidationError',
    'missing',
]
```
### 4 - src/marshmallow/schema.py:

Start line: 341, End line: 416

```python
class BaseSchema(base.SchemaABC):

    def __init__(self, extra=None, only=None, exclude=(), prefix='', strict=None,
                 many=False, context=None, load_only=(), dump_only=(),
                 partial=False):
        # copy declared fields from metaclass
        self.declared_fields = copy.deepcopy(self._declared_fields)
        self.many = many
        self.only = only
        self.exclude = set(self.opts.exclude) | set(exclude)
        if prefix:
            warnings.warn(
                'The `prefix` argument is deprecated. Use a post_dump '
                'method to insert a prefix instead.',
                RemovedInMarshmallow3Warning
            )
        self.prefix = prefix
        self.strict = strict if strict is not None else self.opts.strict
        self.ordered = self.opts.ordered
        self.load_only = set(load_only) or set(self.opts.load_only)
        self.dump_only = set(dump_only) or set(self.opts.dump_only)
        self.partial = partial
        #: Dictionary mapping field_names -> :class:`Field` objects
        self.fields = self.dict_class()
        if extra:
            warnings.warn(
                'The `extra` argument is deprecated. Use a post_dump '
                'method to add additional data instead.',
                RemovedInMarshmallow3Warning
            )
        self.extra = extra
        self.context = context or {}
        self._normalize_nested_options()
        self._types_seen = set()
        self._update_fields(many=many)

    def __repr__(self):
        return '<{ClassName}(many={self.many}, strict={self.strict})>'.format(
            ClassName=self.__class__.__name__, self=self
        )

    def _postprocess(self, data, many, obj):
        if self.extra:
            if many:
                for each in data:
                    each.update(self.extra)
            else:
                data.update(self.extra)
        return data

    @property
    def dict_class(self):
        return OrderedDict if self.ordered else dict

    @property
    def set_class(self):
        return OrderedSet if self.ordered else set

    ##### Override-able methods #####

    def handle_error(self, error, data):
        """Custom error handler function for the schema.

        :param ValidationError error: The `ValidationError` raised during (de)serialization.
        :param data: The original input data.

        .. versionadded:: 2.0.0
        """
        pass

    def get_attribute(self, attr, obj, default):
        """Defines how to pull values from an object to serialize.

        .. versionadded:: 2.0.0
        """
        return utils.get_value(attr, obj, default)

    ##### Handler decorators (deprecated) #####
```
### 5 - src/marshmallow/schema.py:

Start line: 750, End line: 778

```python
class BaseSchema(base.SchemaABC):

    def _update_fields(self, obj=None, many=False):
        """Update fields based on the passed in object."""
        if self.only is not None:
            # Return only fields specified in only option
            if self.opts.fields:
                field_names = self.set_class(self.opts.fields) & self.set_class(self.only)
            else:
                field_names = self.set_class(self.only)
        elif self.opts.fields:
            # Return fields specified in fields option
            field_names = self.set_class(self.opts.fields)
        elif self.opts.additional:
            # Return declared fields + additional fields
            field_names = (self.set_class(self.declared_fields.keys()) |
                            self.set_class(self.opts.additional))
        else:
            field_names = self.set_class(self.declared_fields.keys())

        # If "exclude" option or param is specified, remove those fields
        field_names -= self.exclude
        ret = self.__filter_fields(field_names, obj, many=many)
        # Set parents
        self.__set_field_attrs(ret)
        self.fields = ret
        return self.fields

    def on_bind_field(self, field_name, field_obj):
        """Hook to modify a field when it is bound to the `Schema`. No-op by default."""
        return None
```
### 6 - src/marshmallow/fields.py:

Start line: 460, End line: 480

```python
class Nested(Field):

    def _deserialize(self, value, attr, data):
        if self.many and not utils.is_collection(value):
            self.fail('type', input=value, type=value.__class__.__name__)

        data, errors = self.schema.load(value)
        if errors:
            raise ValidationError(errors, data=data)
        return data

    def _validate_missing(self, value):
        """Validate missing values. Raise a :exc:`ValidationError` if
        `value` should be considered missing.
        """
        if value is missing_ and hasattr(self, 'required'):
            if self.nested == _RECURSIVE_NESTED:
                self.fail('required')
            errors = self._check_required()
            if errors:
                raise ValidationError(errors)
        else:
            super(Nested, self)._validate_missing(value)
```
### 7 - src/marshmallow/fields.py:

Start line: 253, End line: 278

```python
class Field(FieldABC):

    def deserialize(self, value, attr=None, data=None):
        """Deserialize ``value``.

        :raise ValidationError: If an invalid value is passed or if a required value
            is missing.
        """
        # Validate required fields, deserialize, then validate
        # deserialized value
        self._validate_missing(value)
        if getattr(self, 'allow_none', False) is True and value is None:
            return None
        output = self._deserialize(value, attr, data)
        self._validate(output)
        return output

    # Methods for concrete classes to override.

    def _add_to_schema(self, field_name, schema):
        """Update field with values from its parent schema. Called by
            :meth:`__set_field_attrs <marshmallow.Schema.__set_field_attrs>`.

        :param str field_name: Field name set in schema.
        :param Schema schema: Parent schema.
        """
        self.parent = self.parent or schema
        self.name = self.name or field_name
```
### 8 - src/marshmallow/schema.py:

Start line: 939, End line: 968

```python
class BaseSchema(base.SchemaABC):

    def _invoke_processors(self, tag_name, pass_many, data, many, original_data=None):
        for attr_name in self.__processors__[(tag_name, pass_many)]:
            # This will be a bound method.
            processor = getattr(self, attr_name)

            processor_kwargs = processor.__marshmallow_kwargs__[(tag_name, pass_many)]
            pass_original = processor_kwargs.get('pass_original', False)

            if pass_many:
                if pass_original:
                    data = utils.if_none(processor(data, many, original_data), data)
                else:
                    data = utils.if_none(processor(data, many), data)
            elif many:
                if pass_original:
                    data = [utils.if_none(processor(item, original_data), item)
                            for item in data]
                else:
                    data = [utils.if_none(processor(item), item) for item in data]
            else:
                if pass_original:
                    data = utils.if_none(processor(data, original_data), data)
                else:
                    data = utils.if_none(processor(data), data)
        return data


class Schema(with_metaclass(SchemaMeta, BaseSchema)):
    __doc__ = BaseSchema.__doc__
```
### 9 - src/marshmallow/marshalling.py:

Start line: 164, End line: 161

```python
# Key used for schema-level validation errors
SCHEMA = '_schema'


class Unmarshaller(ErrorStore):
    """Callable class responsible for deserializing data and storing errors.

    .. versionadded:: 1.0.0
    """

    default_schema_validation_error = 'Invalid data.'

    def run_validator(self, validator_func, output,
            original_data, fields_dict, index=None,
            many=False, pass_original=False):
        try:
            if pass_original:  # Pass original, raw data (before unmarshalling)
                res = validator_func(output, original_data)
            else:
                res = validator_func(output)
            if res is False:
                raise ValidationError(self.default_schema_validation_error)
        except ValidationError as err:
            errors = self.get_errors(index=index)
            self.error_kwargs.update(err.kwargs)
            # Store or reraise errors
            if err.field_names:
                field_names = err.field_names
                field_objs = [fields_dict[each] if each in fields_dict else None
                              for each in field_names]
            else:
                field_names = [SCHEMA]
                field_objs = []
            self.error_field_names = field_names
            self.error_fields = field_objs
            for field_name in field_names:
                if isinstance(err.messages, (list, tuple)):
                    # self.errors[field_name] may be a dict if schemas are nested
                    if isinstance(errors.get(field_name), dict):
                        errors[field_name].setdefault(
                            SCHEMA, []
                        ).extend(err.messages)
                    else:
                        errors.setdefault(field_name, []).extend(err.messages)
                elif isinstance(err.messages, dict):
                    errors.setdefault(field_name, []).append(err.messages)
                else:
                    errors.setdefault(field_name, []).append(text_type(err))
```
### 10 - examples/flask_example.py:

Start line: 43, End line: 60

```python
class QuoteSchema(Schema):
    id = fields.Int(dump_only=True)
    author = fields.Nested(AuthorSchema, validate=must_not_be_blank)
    content = fields.Str(required=True, validate=must_not_be_blank)
    posted_at = fields.DateTime(dump_only=True)
    @pre_load
    def process_author(self, data):
        author_name = data.get('author')
        if author_name:
            first, last = author_name.split(' ')
            author_dict = dict(first=first, last=last)
        else:
            author_dict = {}
        data['author'] = author_dict
        return data
```
### 11 - src/marshmallow/schema.py:

Start line: 449, End line: 466

```python
class BaseSchema(base.SchemaABC):

    @classmethod
    def accessor(cls, func):
        """Decorator that registers a function for pulling values from an object
        to serialize. The function receives the :class:`Schema` instance, the
        ``key`` of the value to get, the ``obj`` to serialize, and an optional
        ``default`` value.

        .. deprecated:: 2.0.0
            Set the ``error_handler`` class Meta option instead.
        """
        warnings.warn(
            'Schema.accessor is deprecated. Set the accessor class Meta option '
            'instead.', category=DeprecationWarning
        )
        cls.__accessor__ = func
        return func

    ##### Serialization/Deserialization API #####
```
### 12 - src/marshmallow/schema.py:

Start line: 843, End line: 860

```python
class BaseSchema(base.SchemaABC):

    def _invoke_dump_processors(self, tag_name, data, many, original_data=None):
        # The pass_many post-dump processors may do things like add an envelope, so
        # invoke those after invoking the non-pass_many processors which will expect
        # to get a list of items.
        data = self._invoke_processors(tag_name, pass_many=False,
            data=data, many=many, original_data=original_data)
        data = self._invoke_processors(tag_name, pass_many=True,
            data=data, many=many, original_data=original_data)
        return data

    def _invoke_load_processors(self, tag_name, data, many, original_data=None):
        # This has to invert the order of the dump processors, so run the pass_many
        # processors first.
        data = self._invoke_processors(tag_name, pass_many=True,
            data=data, many=many, original_data=original_data)
        data = self._invoke_processors(tag_name, pass_many=False,
            data=data, many=many, original_data=original_data)
        return data
```
### 14 - src/marshmallow/schema.py:

Start line: 612, End line: 630

```python
class BaseSchema(base.SchemaABC):

    def validate(self, data, many=None, partial=None):
        """Validate `data` against the schema, returning a dictionary of
        validation errors.

        :param dict data: The data to validate.
        :param bool many: Whether to validate `data` as a collection. If `None`, the
            value for `self.many` is used.
        :param bool|tuple partial: Whether to ignore missing fields. If `None`,
            the value for `self.partial` is used. If its value is an iterable,
            only missing fields listed in that iterable will be ignored.
        :return: A dictionary of validation errors.
        :rtype: dict

        .. versionadded:: 1.1.0
        """
        _, errors = self._do_load(data, many, partial=partial, postprocess=False)
        return errors

    ##### Private Helpers #####
```
### 15 - src/marshmallow/schema.py:

```python
# -*- coding: utf-8 -*-
"""The :class:`Schema` class, including its metaclass and options (class Meta)."""
from __future__ import absolute_import, unicode_literals

from collections import defaultdict, namedtuple
import copy
import datetime as dt
import decimal
import inspect
import json
import uuid
import warnings
import functools

from marshmallow import base, fields, utils, class_registry, marshalling
from marshmallow.compat import (with_metaclass, iteritems, text_type,
                                binary_type, Mapping, OrderedDict)
from marshmallow.exceptions import ValidationError
from marshmallow.orderedset import OrderedSet
from marshmallow.decorators import (PRE_DUMP, POST_DUMP, PRE_LOAD, POST_LOAD,
                                    VALIDATES, VALIDATES_SCHEMA)
from marshmallow.utils import missing
from marshmallow.warnings import RemovedInMarshmallow3Warning, ChangedInMarshmallow3Warning


#: Return type of :meth:`Schema.dump` including serialized data and errors
MarshalResult = namedtuple('MarshalResult', ['data', 'errors'])
#: Return type of :meth:`Schema.load`, including deserialized data and errors
UnmarshalResult = namedtuple('UnmarshalResult', ['data', 'errors'])
```
### 16 - src/marshmallow/schema.py:

Start line: 468, End line: 552

```python
class BaseSchema(base.SchemaABC):

    def dump(self, obj, many=None, update_fields=True, **kwargs):
        """Serialize an object to native Python data types according to this
        Schema's fields.

        :param obj: The object to serialize.
        :param bool many: Whether to serialize `obj` as a collection. If `None`, the value
            for `self.many` is used.
        :param bool update_fields: Whether to update the schema's field classes. Typically
            set to `True`, but may be `False` when serializing a homogenous collection.
            This parameter is used by `fields.Nested` to avoid multiple updates.
        :return: A tuple of the form (``data``, ``errors``)
        :rtype: `MarshalResult`, a `collections.namedtuple`

        .. versionadded:: 1.0.0
        """
        # Callable marshalling object
        marshal = marshalling.Marshaller(prefix=self.prefix)
        errors = {}
        many = self.many if many is None else bool(many)
        if many and utils.is_iterable_but_not_string(obj):
            obj = list(obj)

        if self._has_processors:
            try:
                processed_obj = self._invoke_dump_processors(
                    PRE_DUMP,
                    obj,
                    many,
                    original_data=obj)
            except ValidationError as error:
                errors = error.normalized_messages()
                result = None
        else:
            processed_obj = obj

        if not errors:
            if update_fields:
                obj_type = type(processed_obj)
                if obj_type not in self._types_seen:
                    self._update_fields(processed_obj, many=many)
                    if not isinstance(processed_obj, Mapping):
                        self._types_seen.add(obj_type)

            try:
                preresult = marshal(
                    processed_obj,
                    self.fields,
                    many=many,
                    # TODO: Remove self.__accessor__ in a later release
                    accessor=self.get_attribute or self.__accessor__,
                    dict_class=self.dict_class,
                    index_errors=self.opts.index_errors,
                    **kwargs
                )
            except ValidationError as error:
                errors = marshal.errors
                preresult = error.data

            result = self._postprocess(preresult, many, obj=obj)

        if not errors and self._has_processors:
            try:
                result = self._invoke_dump_processors(
                    POST_DUMP,
                    result,
                    many,
                    original_data=obj)
            except ValidationError as error:
                errors = error.normalized_messages()
        if errors:
            # TODO: Remove self.__error_handler__ in a later release
            if self.__error_handler__ and callable(self.__error_handler__):
                self.__error_handler__(errors, obj)
            exc = ValidationError(
                errors,
                field_names=marshal.error_field_names,
                fields=marshal.error_fields,
                data=obj,
                **marshal.error_kwargs
            )
            self.handle_error(exc, obj)
            if self.strict:
                raise exc

        return MarshalResult(result, errors)
```
### 30 - src/marshmallow/schema.py:

Start line: 223, End line: 302

```python
class BaseSchema(base.SchemaABC):
    """Base schema class with which to define custom schemas.

    Example usage:

    .. code-block:: python

        import datetime as dt
        from marshmallow import Schema, fields

        class Album(object):
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
        data, errors = schema.dump(album)
        data  # {'release_date': '1968-12-06', 'title': 'Beggars Banquet'}

    :param dict extra: A dict of extra attributes to bind to the serialized result.
    :param tuple|list only: Whitelist of fields to select when instantiating the Schema.
        If None, all fields are used.
        Nested fields can be represented with dot delimiters.
    :param tuple|list exclude: Blacklist of fields to exclude when instantiating the Schema.
        If a field appears in both `only` and `exclude`, it is not used.
        Nested fields can be represented with dot delimiters.
    :param str prefix: Optional prefix that will be prepended to all the
        serialized field names.
    :param bool strict: If `True`, raise errors if invalid data are passed in
        instead of failing silently and storing the errors.
    :param bool many: Should be set to `True` if ``obj`` is a collection
        so that the object will be serialized to a list.
    :param dict context: Optional context passed to :class:`fields.Method` and
        :class:`fields.Function` fields.
    :param tuple|list load_only: Fields to skip during serialization (write-only fields)
    :param tuple|list dump_only: Fields to skip during deserialization (read-only fields)
    :param bool|tuple partial: Whether to ignore missing fields. If its value
        is an iterable, only missing fields listed in that iterable will be
        ignored.

    .. versionchanged:: 2.0.0
        `__validators__`, `__preprocessors__`, and `__data_handlers__` are removed in favor of
        `marshmallow.decorators.validates_schema`,
        `marshmallow.decorators.pre_load` and `marshmallow.decorators.post_dump`.
        `__accessor__` and `__error_handler__` are deprecated. Implement the
        `handle_error` and `get_attribute` methods instead.
        """
    TYPE_MAPPING = {
        text_type: fields.String,
        binary_type: fields.String,
        dt.datetime: fields.DateTime,
        float: fields.Float,
        bool: fields.Boolean,
        tuple: fields.Raw,
        list: fields.Raw,
        set: fields.Raw,
        int: fields.Integer,
        uuid.UUID: fields.UUID,
        dt.time: fields.Time,
        dt.date: fields.Date,
        dt.timedelta: fields.TimeDelta,
        decimal.Decimal: fields.Decimal,
    }

    OPTIONS_CLASS = SchemaOpts

    #: DEPRECATED: Custom error handler function. May be `None`.
    __error_handler__ = None
    #: DEPRECATED: Function used to get values of an object.
    __accessor__ = None
```
### 31 - src/marshmallow/schema.py:

Start line: 632, End line: 712

```python
class BaseSchema(base.SchemaABC):

    def _do_load(self, data, many=None, partial=None, postprocess=True):
        """Deserialize `data`, returning the deserialized result and a dictonary of
        validation errors.

        :param data: The data to deserialize.
        :param bool many: Whether to deserialize `data` as a collection. If `None`, the
            value for `self.many` is used.
        :param bool|tuple partial: Whether to validate required fields. If its value is an iterable,
            only fields listed in that iterable will be ignored will be allowed missing.
            If `True`, all fields will be allowed missing.
            If `None`, the value for `self.partial` is used.
        :param bool postprocess: Whether to run post_load methods..
        :return: A tuple of the form (`data`, `errors`)
        """
        # Callable unmarshalling object
        unmarshal = marshalling.Unmarshaller()
        errors = {}
        many = self.many if many is None else bool(many)
        if partial is None:
            partial = self.partial
        try:
            processed_data = self._invoke_load_processors(
                PRE_LOAD,
                data,
                many,
                original_data=data)
        except ValidationError as err:
            errors = err.normalized_messages()
            result = None
        if not errors:
            try:
                result = unmarshal(
                    processed_data,
                    self.fields,
                    many=many,
                    partial=partial,
                    dict_class=self.dict_class,
                    index_errors=self.opts.index_errors,
                )
            except ValidationError as error:
                result = error.data
            self._invoke_field_validators(unmarshal, data=result, many=many)
            errors = unmarshal.errors
            field_errors = bool(errors)
            # Run schema-level migration
            try:
                self._invoke_validators(unmarshal, pass_many=True, data=result, original_data=data,
                                        many=many, field_errors=field_errors)
            except ValidationError as err:
                errors.update(err.messages)
            try:
                self._invoke_validators(unmarshal, pass_many=False, data=result, original_data=data,
                                        many=many, field_errors=field_errors)
            except ValidationError as err:
                errors.update(err.messages)
        # Run post processors
        if not errors and postprocess:
            try:
                result = self._invoke_load_processors(
                    POST_LOAD,
                    result,
                    many,
                    original_data=data)
            except ValidationError as err:
                errors = err.normalized_messages()
        if errors:
            # TODO: Remove self.__error_handler__ in a later release
            if self.__error_handler__ and callable(self.__error_handler__):
                self.__error_handler__(errors, data)
            exc = ValidationError(
                errors,
                field_names=unmarshal.error_field_names,
                fields=unmarshal.error_fields,
                data=data,
                **unmarshal.error_kwargs
            )
            self.handle_error(exc, data)
            if self.strict:
                raise exc

        return result, errors
```
### 32 - src/marshmallow/schema.py:

Start line: 572, End line: 588

```python
class BaseSchema(base.SchemaABC):

    def load(self, data, many=None, partial=None):
        """Deserialize a data structure to an object defined by this Schema's
        fields and :meth:`make_object`.

        :param dict data: The data to deserialize.
        :param bool many: Whether to deserialize `data` as a collection. If `None`, the
            value for `self.many` is used.
        :param bool|tuple partial: Whether to ignore missing fields. If `None`,
            the value for `self.partial` is used. If its value is an iterable,
            only missing fields listed in that iterable will be ignored.
        :return: A tuple of the form (``data``, ``errors``)
        :rtype: `UnmarshalResult`, a `collections.namedtuple`

        .. versionadded:: 1.0.0
        """
        result, errors = self._do_load(data, many, partial=partial, postprocess=True)
        return UnmarshalResult(data=result, errors=errors)
```
### 35 - src/marshmallow/schema.py:

Start line: 418, End line: 447

```python
class BaseSchema(base.SchemaABC):

    @classmethod
    def error_handler(cls, func):
        """Decorator that registers an error handler function for the schema.
        The function receives the :class:`Schema` instance, a dictionary of errors,
        and the serialized object (if serializing data) or data dictionary (if
        deserializing data) as arguments.

        Example: ::

            class UserSchema(Schema):
                email = fields.Email()

            @UserSchema.error_handler
            def handle_errors(schema, errors, obj):
                raise ValueError('An error occurred while marshalling {}'.format(obj))

            user = User(email='invalid')
            UserSchema().dump(user)  # => raises ValueError
            UserSchema().load({'email': 'bademail'})  # raises ValueError

        .. versionadded:: 0.7.0
        .. deprecated:: 2.0.0
            Set the ``error_handler`` class Meta option instead.
        """
        warnings.warn(
            'Schema.error_handler is deprecated. Set the error_handler class Meta option '
            'instead.', category=DeprecationWarning
        )
        cls.__error_handler__ = func
        return func
```
### 39 - src/marshmallow/schema.py:

Start line: 554, End line: 570

```python
class BaseSchema(base.SchemaABC):

    def dumps(self, obj, many=None, update_fields=True, *args, **kwargs):
        """Same as :meth:`dump`, except return a JSON-encoded string.

        :param obj: The object to serialize.
        :param bool many: Whether to serialize `obj` as a collection. If `None`, the value
            for `self.many` is used.
        :param bool update_fields: Whether to update the schema's field classes. Typically
            set to `True`, but may be `False` when serializing a homogenous collection.
            This parameter is used by `fields.Nested` to avoid multiple updates.
        :return: A tuple of the form (``data``, ``errors``)
        :rtype: `MarshalResult`, a `collections.namedtuple`

        .. versionadded:: 1.0.0
        """
        deserialized, errors = self.dump(obj, many=many, update_fields=update_fields)
        ret = self.opts.json_module.dumps(deserialized, *args, **kwargs)
        return MarshalResult(ret, errors)
```
### 42 - src/marshmallow/schema.py:

Start line: 76, End line: 81

```python
class SchemaMeta(type):
    """Metaclass for the Schema class. Binds the declared fields to
    a ``_declared_fields`` attribute, which is a dictionary mapping attribute
    names to field objects. Also sets the ``opts`` class attribute, which is
    the Schema class's ``class Meta`` options.
    """

    def __new__(mcs, name, bases, attrs):
        meta = attrs.get('Meta')
        ordered = getattr(meta, 'ordered', False)
        if not ordered:
            # Inherit 'ordered' option
            # Warning: We loop through bases instead of MRO because we don't
            # yet have access to the class object
            # (i.e. can't call super before we have fields)
            for base_ in bases:
                if hasattr(base_, 'Meta') and hasattr(base_.Meta, 'ordered'):
                    ordered = base_.Meta.ordered
                    break
            else:
                ordered = False
        cls_fields = _get_fields(attrs, base.FieldABC, pop=True, ordered=ordered)
        klass = super(SchemaMeta, mcs).__new__(mcs, name, bases, attrs)
        inherited_fields = _get_fields_by_mro(klass, base.FieldABC, ordered=ordered)

        # Use getattr rather than attrs['Meta'] so that we get inheritance for free
        meta = getattr(klass, 'Meta')
        # Set klass.opts in __new__ rather than __init__ so that it is accessible in
        # get_declared_fields
        klass.opts = klass.OPTIONS_CLASS(meta)
        # Pass the inherited `ordered` into opts
        klass.opts.ordered = ordered
        # Add fields specifid in the `include` class Meta option
        cls_fields += list(klass.opts.include.items())

        dict_cls = OrderedDict if ordered else dict
        # Assign _declared_fields on class
        klass._declared_fields = mcs.get_declared_fields(
            klass=klass,
            cls_fields=cls_fields,
            inherited_fields=inherited_fields,
            dict_cls=dict_cls
        )
        return klass
```
