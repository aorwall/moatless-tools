# marshmallow-code__marshmallow-1359

| **marshmallow-code/marshmallow** | `b40a0f4e33823e6d0f341f7e8684e359a99060d1` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 22589 |
| **Any found context length** | 22589 |
| **Avg pos** | 3.0 |
| **Min pos** | 3 |
| **Max pos** | 3 |
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
| src/marshmallow/fields.py | 1117 | 1119 | 3 | 3 | 22589


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
| 1 | 1 src/marshmallow/__init__.py | 0 | 35| 180 | 180 | 
| 2 | 2 src/marshmallow/schema.py | 0 | 1141| 9410 | 9590 | 
| **-> 3 <-** | **3 src/marshmallow/fields.py** | 0 | 1694| 12999 | 22589 | 


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

```python
"""The :class:`Schema` class, including its metaclass and options (class Meta)."""
from collections import defaultdict, OrderedDict
from collections.abc import Mapping
from functools import lru_cache
import datetime as dt
import uuid
import decimal
import copy
import inspect
import json
import typing
import warnings

from marshmallow import base, fields as ma_fields, class_registry
from marshmallow.error_store import ErrorStore
from marshmallow.exceptions import ValidationError, StringNotCollectionError
from marshmallow.orderedset import OrderedSet
from marshmallow.decorators import (
    POST_DUMP,
    POST_LOAD,
    PRE_DUMP,
    PRE_LOAD,
    VALIDATES,
    VALIDATES_SCHEMA,
)
from marshmallow.utils import (
    RAISE,
    EXCLUDE,
    INCLUDE,
    missing,
    set_value,
    get_value,
    is_collection,
    is_instance_or_subclass,
    is_iterable_but_not_string,
)


def _get_fields(attrs, field_class, pop=False, ordered=False):
    """Get fields from a class. If ordered=True, fields will sorted by creation index.

    :param attrs: Mapping of class attributes
    :param type field_class: Base field class
    :param bool pop: Remove matching fields
    """
    fields = [
        (field_name, field_value)
        for field_name, field_value in attrs.items()
        if is_instance_or_subclass(field_value, field_class)
    ]
    if pop:
        for field_name, _ in fields:
            del attrs[field_name]
    if ordered:
        fields.sort(key=lambda pair: pair[1]._creation_index)
    return fields


# This function allows Schemas to inherit from non-Schema classes and ensures
#   inheritance according to the MRO
def _get_fields_by_mro(klass, field_class, ordered=False):
    """Collect fields from a class, following its method resolution order. The
    class itself is excluded from the search; only its parents are checked. Get
    fields from ``_declared_fields`` if available, else use ``__dict__``.

    :param type klass: Class whose fields to retrieve
    :param type field_class: Base field class
    """
    mro = inspect.getmro(klass)
    # Loop over mro in reverse to maintain correct order of fields
    return sum(
        (
            _get_fields(
                getattr(base, "_declared_fields", base.__dict__),
                field_class,
                ordered=ordered,
            )
            for base in mro[:0:-1]
        ),
        [],
    )


class SchemaMeta(type):
    """Metaclass for the Schema class. Binds the declared fields to
    a ``_declared_fields`` attribute, which is a dictionary mapping attribute
    names to field objects. Also sets the ``opts`` class attribute, which is
    the Schema class's ``class Meta`` options.
    """

    def __new__(mcs, name, bases, attrs):
        meta = attrs.get("Meta")
        ordered = getattr(meta, "ordered", False)
        if not ordered:
            # Inherit 'ordered' option
            # Warning: We loop through bases instead of MRO because we don't
            # yet have access to the class object
            # (i.e. can't call super before we have fields)
            for base_ in bases:
                if hasattr(base_, "Meta") and hasattr(base_.Meta, "ordered"):
                    ordered = base_.Meta.ordered
                    break
            else:
                ordered = False
        cls_fields = _get_fields(attrs, base.FieldABC, pop=True, ordered=ordered)
        klass = super().__new__(mcs, name, bases, attrs)
        inherited_fields = _get_fields_by_mro(klass, base.FieldABC, ordered=ordered)

        meta = klass.Meta
        # Set klass.opts in __new__ rather than __init__ so that it is accessible in
        # get_declared_fields
        klass.opts = klass.OPTIONS_CLASS(meta, ordered=ordered)
        # Add fields specified in the `include` class Meta option
        cls_fields += list(klass.opts.include.items())

        dict_cls = OrderedDict if ordered else dict
        # Assign _declared_fields on class
        klass._declared_fields = mcs.get_declared_fields(
            klass=klass,
            cls_fields=cls_fields,
            inherited_fields=inherited_fields,
            dict_cls=dict_cls,
        )
        return klass

    @classmethod
    def get_declared_fields(mcs, klass, cls_fields, inherited_fields, dict_cls):
        """Returns a dictionary of field_name => `Field` pairs declard on the class.
        This is exposed mainly so that plugins can add additional fields, e.g. fields
        computed from class Meta options.

        :param type klass: The class object.
        :param list cls_fields: The fields declared on the class, including those added
            by the ``include`` class Meta option.
        :param list inherited_fields: Inherited fields.
        :param type dict_class: Either `dict` or `OrderedDict`, depending on the whether
            the user specified `ordered=True`.
        """
        return dict_cls(inherited_fields + cls_fields)

    def __init__(cls, name, bases, attrs):
        super().__init__(cls, bases, attrs)
        if name and cls.opts.register:
            class_registry.register(name, cls)
        cls._hooks = cls.resolve_hooks()

    def resolve_hooks(cls):
        """Add in the decorated processors

        By doing this after constructing the class, we let standard inheritance
        do all the hard work.
        """
        mro = inspect.getmro(cls)

        hooks = defaultdict(list)

        for attr_name in dir(cls):
            # Need to look up the actual descriptor, not whatever might be
            # bound to the class. This needs to come from the __dict__ of the
            # declaring class.
            for parent in mro:
                try:
                    attr = parent.__dict__[attr_name]
                except KeyError:
                    continue
                else:
                    break
            else:
                # In case we didn't find the attribute and didn't break above.
                # We should never hit this - it's just here for completeness
                # to exclude the possibility of attr being undefined.
                continue

            try:
                hook_config = attr.__marshmallow_hook__
            except AttributeError:
                pass
            else:
                for key in hook_config.keys():
                    # Use name here so we can get the bound method later, in
                    # case the processor was a descriptor or something.
                    hooks[key].append(attr_name)

        return hooks


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

    OPTIONS_CLASS = SchemaOpts

    class Meta:
        """Options object for a Schema.

        Example usage: ::

            class Meta:
                fields = ("id", "email", "date_created")
                exclude = ("password", "secret_attribute")

        Available options:

        - ``fields``: Tuple or list of fields to include in the serialized result.
        - ``additional``: Tuple or list of fields to include *in addition* to the
            explicitly declared fields. ``additional`` and ``fields`` are
            mutually-exclusive options.
        - ``include``: Dictionary of additional fields to include in the schema. It is
            usually better to define fields as class variables, but you may need to
            use this option, e.g., if your fields are Python keywords. May be an
            `OrderedDict`.
        - ``exclude``: Tuple or list of fields to exclude in the serialized result.
            Nested fields can be represented with dot delimiters.
        - ``dateformat``: Default format for `Date <fields.Date>` fields.
        - ``datetimeformat``: Default format for `DateTime <fields.DateTime>` fields.
        - ``render_module``: Module to use for `loads <Schema.loads>` and `dumps <Schema.dumps>`.
            Defaults to `json` from the standard library.
        - ``ordered``: If `True`, order serialization output according to the
            order in which fields were declared. Output of `Schema.dump` will be a
            `collections.OrderedDict`.
        - ``index_errors``: If `True`, errors dictionaries will include the index
            of invalid items in a collection.
        - ``load_only``: Tuple or list of fields to exclude from serialized results.
        - ``dump_only``: Tuple or list of fields to exclude from deserialization
        - ``unknown``: Whether to exclude, include, or raise an error for unknown
            fields in the data. Use `EXCLUDE`, `INCLUDE` or `RAISE`.
        - ``register``: Whether to register the `Schema` with marshmallow's internal
            class registry. Must be `True` if you intend to refer to this `Schema`
            by class name in `Nested` fields. Only set this to `False` when memory
            usage is critical. Defaults to `True`.
        """

        pass

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

    @classmethod
    def from_dict(
        cls, fields: typing.Dict[str, ma_fields.Field], *, name: str = "GeneratedSchema"
    ) -> typing.Type["Schema"]:
        """Generate a `Schema` class given a dictionary of fields.

        .. code-block:: python

            from marshmallow import Schema, fields

            PersonSchema = Schema.from_dict({"name": fields.Str()})
            print(PersonSchema().load({"name": "David"}))  # => {'name': 'David'}

        Generated schemas are not added to the class registry and therefore cannot
        be referred to by name in `Nested` fields.

        :param dict fields: Dictionary mapping field names to field instances.
        :param str name: Optional name for the class, which will appear in
            the ``repr`` for the class.

        .. versionadded:: 3.0.0
        """
        attrs = fields.copy()
        attrs["Meta"] = type(
            "GeneratedMeta", (getattr(cls, "Meta", object),), {"register": False}
        )
        schema_cls = type(name, (cls,), attrs)
        return schema_cls

    ##### Override-able methods #####

    def handle_error(self, error, data, *, many, **kwargs):
        """Custom error handler function for the schema.

        :param ValidationError error: The `ValidationError` raised during (de)serialization.
        :param data: The original input data.
        :param bool many: Value of ``many`` on dump or load.
        :param bool partial: Value of ``partial`` on load.

        .. versionadded:: 2.0.0

        .. versionchanged:: 3.0.0rc9
            Receives `many` and `partial` (on deserialization) as keyword arguments.
        """
        pass

    def get_attribute(self, obj, attr, default):
        """Defines how to pull values from an object to serialize.

        .. versionadded:: 2.0.0

        .. versionchanged:: 3.0.0a1
            Changed position of ``obj`` and ``attr``.
        """
        return get_value(obj, attr, default)

    ##### Serialization/Deserialization API #####

    @staticmethod
    def _call_and_store(getter_func, data, *, field_name, error_store, index=None):
        """Call ``getter_func`` with ``data`` as its argument, and store any `ValidationErrors`.

        :param callable getter_func: Function for getting the serialized/deserialized
            value from ``data``.
        :param data: The data passed to ``getter_func``.
        :param str field_name: Field name.
        :param int index: Index of the item being validated, if validating a collection,
            otherwise `None`.
        """
        try:
            value = getter_func(data)
        except ValidationError as error:
            error_store.store_error(error.messages, field_name, index=index)
            # When a Nested field fails validation, the marshalled data is stored
            # on the ValidationError's valid_data attribute
            return error.valid_data or missing
        return value

    def _serialize(self, obj, *, many=False):
        """Serialize ``obj``.

        :param obj: The object(s) to serialize.
        :param bool many: `True` if ``data`` should be serialized as a collection.
        :return: A dictionary of the serialized data

        .. versionchanged:: 1.0.0
            Renamed from ``marshal``.
        """
        if many and obj is not None:
            return [self._serialize(d, many=False) for d in obj]
        ret = self.dict_class()
        for attr_name, field_obj in self.dump_fields.items():
            value = field_obj.serialize(attr_name, obj, accessor=self.get_attribute)
            if value is missing:
                continue
            key = field_obj.data_key or attr_name
            ret[key] = value
        return ret

    def dump(self, obj, *, many=None):
        """Serialize an object to native Python data types according to this
        Schema's fields.

        :param obj: The object to serialize.
        :param bool many: Whether to serialize `obj` as a collection. If `None`, the value
            for `self.many` is used.
        :return: A dict of serialized data
        :rtype: dict

        .. versionadded:: 1.0.0
        .. versionchanged:: 3.0.0b7
            This method returns the serialized data rather than a ``(data, errors)`` duple.
            A :exc:`ValidationError <marshmallow.exceptions.ValidationError>` is raised
            if ``obj`` is invalid.
        .. versionchanged:: 3.0.0rc9
            Validation no longer occurs upon serialization.
        """
        many = self.many if many is None else bool(many)
        if many and is_iterable_but_not_string(obj):
            obj = list(obj)

        if self._has_processors(PRE_DUMP):
            processed_obj = self._invoke_dump_processors(
                PRE_DUMP, obj, many=many, original_data=obj
            )
        else:
            processed_obj = obj

        result = self._serialize(processed_obj, many=many)

        if self._has_processors(POST_DUMP):
            result = self._invoke_dump_processors(
                POST_DUMP, result, many=many, original_data=obj
            )

        return result

    def dumps(self, obj, *args, many=None, **kwargs):
        """Same as :meth:`dump`, except return a JSON-encoded string.

        :param obj: The object to serialize.
        :param bool many: Whether to serialize `obj` as a collection. If `None`, the value
            for `self.many` is used.
        :return: A ``json`` string
        :rtype: str

        .. versionadded:: 1.0.0
        .. versionchanged:: 3.0.0b7
            This method returns the serialized data rather than a ``(data, errors)`` duple.
            A :exc:`ValidationError <marshmallow.exceptions.ValidationError>` is raised
            if ``obj`` is invalid.
        """
        serialized = self.dump(obj, many=many)
        return self.opts.render_module.dumps(serialized, *args, **kwargs)

    def _deserialize(
        self, data, *, error_store, many=False, partial=False, unknown=RAISE, index=None
    ):
        """Deserialize ``data``.

        :param dict data: The data to deserialize.
        :param ErrorStore error_store: Structure to store errors.
        :param bool many: `True` if ``data`` should be deserialized as a collection.
        :param bool|tuple partial: Whether to ignore missing fields and not require
            any fields declared. Propagates down to ``Nested`` fields as well. If
            its value is an iterable, only missing fields listed in that iterable
            will be ignored. Use dot delimiters to specify nested fields.
        :param unknown: Whether to exclude, include, or raise an error for unknown
            fields in the data. Use `EXCLUDE`, `INCLUDE` or `RAISE`.
        :param int index: Index of the item being serialized (for storing errors) if
            serializing a collection, otherwise `None`.
        :return: A dictionary of the deserialized data.
        """
        index_errors = self.opts.index_errors
        index = index if index_errors else None
        if many:
            if not is_collection(data):
                error_store.store_error([self.error_messages["type"]], index=index)
                ret = []
            else:
                ret = [
                    self._deserialize(
                        d,
                        error_store=error_store,
                        many=False,
                        partial=partial,
                        unknown=unknown,
                        index=idx,
                    )
                    for idx, d in enumerate(data)
                ]
            return ret
        ret = self.dict_class()
        # Check data is a dict
        if not isinstance(data, Mapping):
            error_store.store_error([self.error_messages["type"]], index=index)
        else:
            partial_is_collection = is_collection(partial)
            for attr_name, field_obj in self.load_fields.items():
                field_name = field_obj.data_key or attr_name
                raw_value = data.get(field_name, missing)
                if raw_value is missing:
                    # Ignore missing field if we're allowed to.
                    if partial is True or (
                        partial_is_collection and attr_name in partial
                    ):
                        continue
                d_kwargs = {}
                # Allow partial loading of nested schemas.
                if partial_is_collection:
                    prefix = field_name + "."
                    len_prefix = len(prefix)
                    sub_partial = [
                        f[len_prefix:] for f in partial if f.startswith(prefix)
                    ]
                    d_kwargs["partial"] = sub_partial
                else:
                    d_kwargs["partial"] = partial
                getter = lambda val: field_obj.deserialize(
                    val, field_name, data, **d_kwargs
                )
                value = self._call_and_store(
                    getter_func=getter,
                    data=raw_value,
                    field_name=field_name,
                    error_store=error_store,
                    index=index,
                )
                if value is not missing:
                    key = field_obj.attribute or attr_name
                    set_value(ret, key, value)
            if unknown != EXCLUDE:
                fields = {
                    field_obj.data_key or field_name
                    for field_name, field_obj in self.load_fields.items()
                }
                for key in set(data) - fields:
                    value = data[key]
                    if unknown == INCLUDE:
                        set_value(ret, key, value)
                    elif unknown == RAISE:
                        error_store.store_error(
                            [self.error_messages["unknown"]],
                            key,
                            (index if index_errors else None),
                        )
        return ret

    def load(self, data, *, many=None, partial=None, unknown=None):
        """Deserialize a data structure to an object defined by this Schema's fields.

        :param dict data: The data to deserialize.
        :param bool many: Whether to deserialize `data` as a collection. If `None`, the
            value for `self.many` is used.
        :param bool|tuple partial: Whether to ignore missing fields and not require
            any fields declared. Propagates down to ``Nested`` fields as well. If
            its value is an iterable, only missing fields listed in that iterable
            will be ignored. Use dot delimiters to specify nested fields.
        :param unknown: Whether to exclude, include, or raise an error for unknown
            fields in the data. Use `EXCLUDE`, `INCLUDE` or `RAISE`.
            If `None`, the value for `self.unknown` is used.
        :return: A dict of deserialized data
        :rtype: dict

        .. versionadded:: 1.0.0
        .. versionchanged:: 3.0.0b7
            This method returns the deserialized data rather than a ``(data, errors)`` duple.
            A :exc:`ValidationError <marshmallow.exceptions.ValidationError>` is raised
            if invalid data are passed.
        """
        return self._do_load(
            data, many=many, partial=partial, unknown=unknown, postprocess=True
        )

    def loads(self, json_data, *, many=None, partial=None, unknown=None, **kwargs):
        """Same as :meth:`load`, except it takes a JSON string as input.

        :param str json_data: A JSON string of the data to deserialize.
        :param bool many: Whether to deserialize `obj` as a collection. If `None`, the
            value for `self.many` is used.
        :param bool|tuple partial: Whether to ignore missing fields and not require
            any fields declared. Propagates down to ``Nested`` fields as well. If
            its value is an iterable, only missing fields listed in that iterable
            will be ignored. Use dot delimiters to specify nested fields.
        :param unknown: Whether to exclude, include, or raise an error for unknown
            fields in the data. Use `EXCLUDE`, `INCLUDE` or `RAISE`.
            If `None`, the value for `self.unknown` is used.
        :return: A dict of deserialized data
        :rtype: dict

        .. versionadded:: 1.0.0
        .. versionchanged:: 3.0.0b7
            This method returns the deserialized data rather than a ``(data, errors)`` duple.
            A :exc:`ValidationError <marshmallow.exceptions.ValidationError>` is raised
            if invalid data are passed.
        """
        data = self.opts.render_module.loads(json_data, **kwargs)
        return self.load(data, many=many, partial=partial, unknown=unknown)

    def _run_validator(
        self,
        validator_func,
        output,
        *,
        original_data,
        error_store,
        many,
        partial,
        pass_original,
        index=None
    ):
        try:
            if pass_original:  # Pass original, raw data (before unmarshalling)
                validator_func(output, original_data, partial=partial, many=many)
            else:
                validator_func(output, partial=partial, many=many)
        except ValidationError as err:
            error_store.store_error(err.messages, err.field_name, index=index)

    def validate(self, data, *, many=None, partial=None):
        """Validate `data` against the schema, returning a dictionary of
        validation errors.

        :param dict data: The data to validate.
        :param bool many: Whether to validate `data` as a collection. If `None`, the
            value for `self.many` is used.
        :param bool|tuple partial: Whether to ignore missing fields and not require
            any fields declared. Propagates down to ``Nested`` fields as well. If
            its value is an iterable, only missing fields listed in that iterable
            will be ignored. Use dot delimiters to specify nested fields.
        :return: A dictionary of validation errors.
        :rtype: dict

        .. versionadded:: 1.1.0
        """
        try:
            self._do_load(data, many=many, partial=partial, postprocess=False)
        except ValidationError as exc:
            return exc.messages
        return {}

    ##### Private Helpers #####

    def _do_load(
        self, data, *, many=None, partial=None, unknown=None, postprocess=True
    ):
        """Deserialize `data`, returning the deserialized result.

        :param data: The data to deserialize.
        :param bool many: Whether to deserialize `data` as a collection. If `None`, the
            value for `self.many` is used.
        :param bool|tuple partial: Whether to validate required fields. If its
            value is an iterable, only fields listed in that iterable will be
            ignored will be allowed missing. If `True`, all fields will be allowed missing.
            If `None`, the value for `self.partial` is used.
        :param unknown: Whether to exclude, include, or raise an error for unknown
            fields in the data. Use `EXCLUDE`, `INCLUDE` or `RAISE`.
            If `None`, the value for `self.unknown` is used.
        :param bool postprocess: Whether to run post_load methods..
        :return: A dict of deserialized data
        :rtype: dict
        """
        error_store = ErrorStore()
        errors = {}
        many = self.many if many is None else bool(many)
        unknown = unknown or self.unknown
        if partial is None:
            partial = self.partial
        # Run preprocessors
        if self._has_processors(PRE_LOAD):
            try:
                processed_data = self._invoke_load_processors(
                    PRE_LOAD, data, many=many, original_data=data, partial=partial
                )
            except ValidationError as err:
                errors = err.normalized_messages()
                result = None
        else:
            processed_data = data
        if not errors:
            # Deserialize data
            result = self._deserialize(
                processed_data,
                error_store=error_store,
                many=many,
                partial=partial,
                unknown=unknown,
            )
            # Run field-level validation
            self._invoke_field_validators(
                error_store=error_store, data=result, many=many
            )
            # Run schema-level validation
            if self._has_processors(VALIDATES_SCHEMA):
                field_errors = bool(error_store.errors)
                self._invoke_schema_validators(
                    error_store=error_store,
                    pass_many=True,
                    data=result,
                    original_data=data,
                    many=many,
                    partial=partial,
                    field_errors=field_errors,
                )
                self._invoke_schema_validators(
                    error_store=error_store,
                    pass_many=False,
                    data=result,
                    original_data=data,
                    many=many,
                    partial=partial,
                    field_errors=field_errors,
                )
            errors = error_store.errors
            # Run post processors
            if not errors and postprocess and self._has_processors(POST_LOAD):
                try:
                    result = self._invoke_load_processors(
                        POST_LOAD,
                        result,
                        many=many,
                        original_data=data,
                        partial=partial,
                    )
                except ValidationError as err:
                    errors = err.normalized_messages()
        if errors:
            exc = ValidationError(errors, data=data, valid_data=result)
            self.handle_error(exc, data, many=many, partial=partial)
            raise exc

        return result

    def _normalize_nested_options(self):
        """Apply then flatten nested schema options"""
        if self.only is not None:
            # Apply the only option to nested fields.
            self.__apply_nested_option("only", self.only, "intersection")
            # Remove the child field names from the only option.
            self.only = self.set_class([field.split(".", 1)[0] for field in self.only])
        if self.exclude:
            # Apply the exclude option to nested fields.
            self.__apply_nested_option("exclude", self.exclude, "union")
            # Remove the parent field names from the exclude option.
            self.exclude = self.set_class(
                [field for field in self.exclude if "." not in field]
            )

    def __apply_nested_option(self, option_name, field_names, set_operation):
        """Apply nested options to nested fields"""
        # Split nested field names on the first dot.
        nested_fields = [name.split(".", 1) for name in field_names if "." in name]
        # Partition the nested field names by parent field.
        nested_options = defaultdict(list)
        for parent, nested_names in nested_fields:
            nested_options[parent].append(nested_names)
        # Apply the nested field options.
        for key, options in iter(nested_options.items()):
            new_options = self.set_class(options)
            original_options = getattr(self.declared_fields[key], option_name, ())
            if original_options:
                if set_operation == "union":
                    new_options |= self.set_class(original_options)
                if set_operation == "intersection":
                    new_options &= self.set_class(original_options)
            setattr(self.declared_fields[key], option_name, new_options)

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

    def on_bind_field(self, field_name, field_obj):
        """Hook to modify a field when it is bound to the `Schema`.

        No-op by default.
        """
        return None

    def _bind_field(self, field_name, field_obj):
        """Bind field to the schema, setting any necessary attributes on the
        field (e.g. parent and name).

        Also set field load_only and dump_only values if field_name was
        specified in ``class Meta``.
        """
        try:
            if field_name in self.load_only:
                field_obj.load_only = True
            if field_name in self.dump_only:
                field_obj.dump_only = True
            field_obj._bind_to_schema(field_name, self)
            self.on_bind_field(field_name, field_obj)
        except TypeError as error:
            # field declared as a class, not an instance
            if isinstance(field_obj, type) and issubclass(field_obj, base.FieldABC):
                msg = (
                    'Field for "{}" must be declared as a '
                    "Field instance, not a class. "
                    'Did you mean "fields.{}()"?'.format(field_name, field_obj.__name__)
                )
                raise TypeError(msg) from error

    @lru_cache(maxsize=8)
    def _has_processors(self, tag):
        return self._hooks[(tag, True)] or self._hooks[(tag, False)]

    def _invoke_dump_processors(self, tag, data, *, many, original_data=None):
        # The pass_many post-dump processors may do things like add an envelope, so
        # invoke those after invoking the non-pass_many processors which will expect
        # to get a list of items.
        data = self._invoke_processors(
            tag, pass_many=False, data=data, many=many, original_data=original_data
        )
        data = self._invoke_processors(
            tag, pass_many=True, data=data, many=many, original_data=original_data
        )
        return data

    def _invoke_load_processors(self, tag, data, *, many, original_data, partial):
        # This has to invert the order of the dump processors, so run the pass_many
        # processors first.
        data = self._invoke_processors(
            tag,
            pass_many=True,
            data=data,
            many=many,
            original_data=original_data,
            partial=partial,
        )
        data = self._invoke_processors(
            tag,
            pass_many=False,
            data=data,
            many=many,
            original_data=original_data,
            partial=partial,
        )
        return data

    def _invoke_field_validators(self, *, error_store, data, many):
        for attr_name in self._hooks[VALIDATES]:
            validator = getattr(self, attr_name)
            validator_kwargs = validator.__marshmallow_hook__[VALIDATES]
            field_name = validator_kwargs["field_name"]

            try:
                field_obj = self.fields[field_name]
            except KeyError as error:
                if field_name in self.declared_fields:
                    continue
                raise ValueError(
                    '"{}" field does not exist.'.format(field_name)
                ) from error

            if many:
                for idx, item in enumerate(data):
                    try:
                        value = item[field_obj.attribute or field_name]
                    except KeyError:
                        pass
                    else:
                        validated_value = self._call_and_store(
                            getter_func=validator,
                            data=value,
                            field_name=field_obj.data_key or field_name,
                            error_store=error_store,
                            index=(idx if self.opts.index_errors else None),
                        )
                        if validated_value is missing:
                            data[idx].pop(field_name, None)
            else:
                try:
                    value = data[field_obj.attribute or field_name]
                except KeyError:
                    pass
                else:
                    validated_value = self._call_and_store(
                        getter_func=validator,
                        data=value,
                        field_name=field_obj.data_key or field_name,
                        error_store=error_store,
                    )
                    if validated_value is missing:
                        data.pop(field_name, None)

    def _invoke_schema_validators(
        self,
        *,
        error_store,
        pass_many,
        data,
        original_data,
        many,
        partial,
        field_errors=False
    ):
        for attr_name in self._hooks[(VALIDATES_SCHEMA, pass_many)]:
            validator = getattr(self, attr_name)
            validator_kwargs = validator.__marshmallow_hook__[
                (VALIDATES_SCHEMA, pass_many)
            ]
            if field_errors and validator_kwargs["skip_on_field_errors"]:
                continue
            pass_original = validator_kwargs.get("pass_original", False)

            if many and not pass_many:
                for idx, (item, orig) in enumerate(zip(data, original_data)):
                    self._run_validator(
                        validator,
                        item,
                        original_data=orig,
                        error_store=error_store,
                        many=many,
                        partial=partial,
                        index=idx,
                        pass_original=pass_original,
                    )
            else:
                self._run_validator(
                    validator,
                    data,
                    original_data=original_data,
                    error_store=error_store,
                    many=many,
                    pass_original=pass_original,
                    partial=partial,
                )

    def _invoke_processors(
        self, tag, *, pass_many, data, many, original_data=None, **kwargs
    ):
        key = (tag, pass_many)
        for attr_name in self._hooks[key]:
            # This will be a bound method.
            processor = getattr(self, attr_name)

            processor_kwargs = processor.__marshmallow_hook__[key]
            pass_original = processor_kwargs.get("pass_original", False)

            if pass_many:
                if pass_original:
                    data = processor(data, original_data, many=many, **kwargs)
                else:
                    data = processor(data, many=many, **kwargs)
            elif many:
                if pass_original:
                    data = [
                        processor(item, original, many=many, **kwargs)
                        for item, original in zip(data, original_data)
                    ]
                else:
                    data = [processor(item, many=many, **kwargs) for item in data]
            else:
                if pass_original:
                    data = processor(data, original_data, many=many, **kwargs)
                else:
                    data = processor(data, many=many, **kwargs)
        return data


class Schema(BaseSchema, metaclass=SchemaMeta):
    __doc__ = BaseSchema.__doc__

```
### 3 - src/marshmallow/fields.py:

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


class Field(FieldABC):
    """Basic field from which other fields should extend. It applies no
    formatting by default, and should only be used in cases where
    data does not need to be formatted before being serialized or deserialized.
    On error, the name of the field will be returned.

    :param default: If set, this value will be used during serialization if the input value
        is missing. If not set, the field will be excluded from the serialized output if the
        input value is missing. May be a value or a callable.
    :param missing: Default deserialization value for the field if the field is not
        found in the input data. May be a value or a callable.
    :param str data_key: The name of the dict key in the external representation, i.e.
        the input of `load` and the output of `dump`.
        If `None`, the key will match the name of the field.
    :param str attribute: The name of the attribute to get the value from when serializing.
        If `None`, assumes the attribute has the same name as the field.
        Note: This should only be used for very specific use cases such as
        outputting multiple fields for a single attribute. In most cases,
        you should use ``data_key`` instead.
    :param callable validate: Validator or collection of validators that are called
        during deserialization. Validator takes a field's input value as
        its only parameter and returns a boolean.
        If it returns `False`, an :exc:`ValidationError` is raised.
    :param required: Raise a :exc:`ValidationError` if the field value
        is not supplied during deserialization.
    :param allow_none: Set this to `True` if `None` should be considered a valid value during
        validation/deserialization. If ``missing=None`` and ``allow_none`` is unset,
        will default to ``True``. Otherwise, the default is ``False``.
    :param bool load_only: If `True` skip this field during serialization, otherwise
        its value will be present in the serialized data.
    :param bool dump_only: If `True` skip this field during deserialization, otherwise
        its value will be present in the deserialized object. In the context of an
        HTTP API, this effectively marks the field as "read-only".
    :param dict error_messages: Overrides for `Field.default_error_messages`.
    :param metadata: Extra arguments to be stored as metadata.

    .. versionchanged:: 2.0.0
        Removed `error` parameter. Use ``error_messages`` instead.

    .. versionchanged:: 2.0.0
        Added `allow_none` parameter, which makes validation/deserialization of `None`
        consistent across fields.

    .. versionchanged:: 2.0.0
        Added `load_only` and `dump_only` parameters, which allow field skipping
        during the (de)serialization process.

    .. versionchanged:: 2.0.0
        Added `missing` parameter, which indicates the value for a field if the field
        is not found during deserialization.

    .. versionchanged:: 2.0.0
        ``default`` value is only used if explicitly set. Otherwise, missing values
        inputs are excluded from serialized output.

    .. versionchanged:: 3.0.0b8
        Add ``data_key`` parameter for the specifying the key in the input and
        output data. This parameter replaced both ``load_from`` and ``dump_to``.
    """

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

    def get_value(self, obj, attr, accessor=None, default=missing_):
        """Return the value for a given key from an object.

        :param object obj: The object to get the value from.
        :param str attr: The attribute/key in `obj` to get the value from.
        :param callable accessor: A callable used to retrieve the value of `attr` from
            the object `obj`. Defaults to `marshmallow.utils.get_value`.
        """
        # NOTE: Use getattr instead of direct attribute access here so that
        # subclasses aren't required to define `attribute` member
        attribute = getattr(self, "attribute", None)
        accessor_func = accessor or utils.get_value
        check_key = attr if attribute is None else attribute
        return accessor_func(obj, check_key, default)

    def _validate(self, value):
        """Perform validation on ``value``. Raise a :exc:`ValidationError` if validation
        does not succeed.
        """
        errors = []
        kwargs = {}
        for validator in self.validators:
            try:
                r = validator(value)
                if not isinstance(validator, Validator) and r is False:
                    raise self.make_error("validator_failed")
            except ValidationError as err:
                kwargs.update(err.kwargs)
                if isinstance(err.messages, dict):
                    errors.append(err.messages)
                else:
                    errors.extend(err.messages)
        if errors:
            raise ValidationError(errors, **kwargs)

    def make_error(self, key: str, **kwargs) -> ValidationError:
        """Helper method to make a `ValidationError` with an error message
        from ``self.error_messages``.
        """
        try:
            msg = self.error_messages[key]
        except KeyError as error:
            class_name = self.__class__.__name__
            msg = MISSING_ERROR_MESSAGE.format(class_name=class_name, key=key)
            raise AssertionError(msg) from error
        if isinstance(msg, (str, bytes)):
            msg = msg.format(**kwargs)
        return ValidationError(msg)

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

    def serialize(self, attr, obj, accessor=None, **kwargs):
        """Pulls the value for the given key from the object, applies the
        field's formatting and returns the result.

        :param str attr: The attribute/key to get from the object.
        :param str obj: The object to access the attribute/key from.
        :param callable accessor: Function used to access values from ``obj``.
        :param dict kwargs: Field-specific keyword arguments.
        """
        if self._CHECK_ATTRIBUTE:
            value = self.get_value(obj, attr, accessor=accessor)
            if value is missing_ and hasattr(self, "default"):
                default = self.default
                value = default() if callable(default) else default
            if value is missing_:
                return value
        else:
            value = None
        return self._serialize(value, attr, obj, **kwargs)

    def deserialize(self, value, attr=None, data=None, **kwargs):
        """Deserialize ``value``.

        :param value: The value to deserialize.
        :param str attr: The attribute/key in `data` to deserialize.
        :param dict data: The raw input data passed to `Schema.load`.
        :param dict kwargs: Field-specific keyword arguments.
        :raise ValidationError: If an invalid value is passed or if a required value
            is missing.
        """
        # Validate required fields, deserialize, then validate
        # deserialized value
        self._validate_missing(value)
        if value is missing_:
            _miss = self.missing
            return _miss() if callable(_miss) else _miss
        if getattr(self, "allow_none", False) is True and value is None:
            return None
        output = self._deserialize(value, attr, data, **kwargs)
        self._validate(output)
        return output

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

    def _deserialize(self, value, attr, data, **kwargs):
        """Deserialize value. Concrete :class:`Field` classes should implement this method.

        :param value: The value to be deserialized.
        :param str attr: The attribute/key in `data` to be deserialized.
        :param dict data: The raw input data passed to the `Schema.load`.
        :param dict kwargs: Field-specific keyword arguments.
        :raise ValidationError: In case of formatting or validation failure.
        :return: The deserialized value.

        .. versionchanged:: 2.0.0
            Added ``attr`` and ``data`` parameters.

        .. versionchanged:: 3.0.0
            Added ``**kwargs`` to signature.
        """
        return value

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


class Nested(Field):
    """Allows you to nest a :class:`Schema <marshmallow.Schema>`
    inside a field.

    Examples: ::

        user = fields.Nested(UserSchema)
        user2 = fields.Nested('UserSchema')  # Equivalent to above
        collaborators = fields.Nested(UserSchema, many=True, only=('id',))
        parent = fields.Nested('self')

    When passing a `Schema <marshmallow.Schema>` instance as the first argument,
    the instance's ``exclude``, ``only``, and ``many`` attributes will be respected.

    Therefore, when passing the ``exclude``, ``only``, or ``many`` arguments to `fields.Nested`,
    you should pass a `Schema <marshmallow.Schema>` class (not an instance) as the first argument.

    ::

        # Yes
        author = fields.Nested(UserSchema, only=('id', 'name'))

        # No
        author = fields.Nested(UserSchema(), only=('id', 'name'))

    :param Schema nested: The Schema class or class name (string)
        to nest, or ``"self"`` to nest the :class:`Schema` within itself.
    :param tuple exclude: A list or tuple of fields to exclude.
    :param only: A list or tuple of fields to marshal. If `None`, all fields are marshalled.
        This parameter takes precedence over ``exclude``.
    :param bool many: Whether the field is a collection of objects.
    :param unknown: Whether to exclude, include, or raise an error for unknown
        fields in the data. Use `EXCLUDE`, `INCLUDE` or `RAISE`.
    :param kwargs: The same keyword arguments that :class:`Field` receives.
    """

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

    def _deserialize(self, value, attr, data, partial=None, many=False, **kwargs):
        """Same as :meth:`Field._deserialize` with additional ``partial`` argument.

        :param bool|tuple partial: For nested schemas, the ``partial``
            parameter passed to `Schema.load`.

        .. versionchanged:: 3.0.0
            Add ``partial`` parameter.
        """
        self._test_collection(value, many=many)
        return self._load(value, data, partial=partial, many=many)


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


class String(Field):
    """A string field.

    :param kwargs: The same keyword arguments that :class:`Field` receives.
    """

    default_error_messages = {
        "invalid": "Not a valid string.",
        "invalid_utf8": "Not a valid utf-8 string.",
    }

    def _serialize(self, value, attr, obj, **kwargs):
        if value is None:
            return None
        return utils.ensure_text_type(value)

    def _deserialize(self, value, attr, data, **kwargs):
        if not isinstance(value, (str, bytes)):
            raise self.make_error("invalid")
        try:
            return utils.ensure_text_type(value)
        except UnicodeDecodeError as error:
            raise self.make_error("invalid_utf8") from error


class UUID(String):
    """A UUID field."""

    default_error_messages = {"invalid_uuid": "Not a valid UUID."}

    def _validated(self, value):
        """Format the value or raise a :exc:`ValidationError` if an error occurs."""
        if value is None:
            return None
        if isinstance(value, uuid.UUID):
            return value
        try:
            if isinstance(value, bytes) and len(value) == 16:
                return uuid.UUID(bytes=value)
            else:
                return uuid.UUID(value)
        except (ValueError, AttributeError, TypeError) as error:
            raise self.make_error("invalid_uuid") from error

    def _serialize(self, value, attr, obj, **kwargs):
        val = str(value) if value is not None else None
        return super()._serialize(val, attr, obj, **kwargs)

    def _deserialize(self, value, attr, data, **kwargs):
        return self._validated(value)


class Number(Field):
    """Base class for number fields.

    :param bool as_string: If True, format the serialized value as a string.
    :param kwargs: The same keyword arguments that :class:`Field` receives.
    """

    num_type = float
    default_error_messages = {
        "invalid": "Not a valid number.",
        "too_large": "Number too large.",
    }

    def __init__(self, *, as_string=False, **kwargs):
        self.as_string = as_string
        super().__init__(**kwargs)

    def _format_num(self, value):
        """Return the number value for value, given this field's `num_type`."""
        return self.num_type(value)

    def _validated(self, value):
        """Format the value or raise a :exc:`ValidationError` if an error occurs."""
        if value is None:
            return None
        # (value is True or value is False) is ~5x faster than isinstance(value, bool)
        if value is True or value is False:
            raise self.make_error("invalid", input=value)
        try:
            return self._format_num(value)
        except (TypeError, ValueError) as error:
            raise self.make_error("invalid", input=value) from error
        except OverflowError as error:
            raise self.make_error("too_large", input=value) from error

    def _to_string(self, value):
        return str(value)

    def _serialize(self, value, attr, obj, **kwargs):
        """Return a string if `self.as_string=True`, otherwise return this field's `num_type`."""
        if value is None:
            return None
        ret = self._format_num(value)
        return self._to_string(ret) if self.as_string else ret

    def _deserialize(self, value, attr, data, **kwargs):
        return self._validated(value)


class Integer(Number):
    """An integer field.

    :param kwargs: The same keyword arguments that :class:`Number` receives.
    """

    num_type = int
    default_error_messages = {"invalid": "Not a valid integer."}

    def __init__(self, *, strict=False, **kwargs):
        self.strict = strict
        super().__init__(**kwargs)

    # override Number
    def _validated(self, value):
        if self.strict:
            if isinstance(value, numbers.Number) and isinstance(
                value, numbers.Integral
            ):
                return super()._validated(value)
            raise self.make_error("invalid", input=value)
        return super()._validated(value)


class Float(Number):
    """A double as an IEEE-754 double precision string.

    :param bool allow_nan: If `True`, `NaN`, `Infinity` and `-Infinity` are allowed,
        even though they are illegal according to the JSON specification.
    :param bool as_string: If True, format the value as a string.
    :param kwargs: The same keyword arguments that :class:`Number` receives.
    """

    num_type = float
    default_error_messages = {
        "special": "Special numeric values (nan or infinity) are not permitted."
    }

    def __init__(self, *, allow_nan=False, as_string=False, **kwargs):
        self.allow_nan = allow_nan
        super().__init__(as_string=as_string, **kwargs)

    def _validated(self, value):
        num = super()._validated(value)
        if self.allow_nan is False:
            if math.isnan(num) or num == float("inf") or num == float("-inf"):
                raise self.make_error("special")
        return num


class Decimal(Number):
    """A field that (de)serializes to the Python ``decimal.Decimal`` type.
    It's safe to use when dealing with money values, percentages, ratios
    or other numbers where precision is critical.

    .. warning::

        This field serializes to a `decimal.Decimal` object by default. If you need
        to render your data as JSON, keep in mind that the `json` module from the
        standard library does not encode `decimal.Decimal`. Therefore, you must use
        a JSON library that can handle decimals, such as `simplejson`, or serialize
        to a string by passing ``as_string=True``.

    .. warning::

        If a JSON `float` value is passed to this field for deserialization it will
        first be cast to its corresponding `string` value before being deserialized
        to a `decimal.Decimal` object. The default `__str__` implementation of the
        built-in Python `float` type may apply a destructive transformation upon
        its input data and therefore cannot be relied upon to preserve precision.
        To avoid this, you can instead pass a JSON `string` to be deserialized
        directly.

    :param int places: How many decimal places to quantize the value. If `None`, does
        not quantize the value.
    :param rounding: How to round the value during quantize, for example
        `decimal.ROUND_UP`. If None, uses the rounding value from
        the current thread's context.
    :param bool allow_nan: If `True`, `NaN`, `Infinity` and `-Infinity` are allowed,
        even though they are illegal according to the JSON specification.
    :param bool as_string: If True, serialize to a string instead of a Python
        `decimal.Decimal` type.
    :param kwargs: The same keyword arguments that :class:`Number` receives.

    .. versionadded:: 1.2.0
    """

    num_type = decimal.Decimal

    default_error_messages = {
        "special": "Special numeric values (nan or infinity) are not permitted."
    }

    def __init__(
        self, places=None, rounding=None, *, allow_nan=False, as_string=False, **kwargs
    ):
        self.places = (
            decimal.Decimal((0, (1,), -places)) if places is not None else None
        )
        self.rounding = rounding
        self.allow_nan = allow_nan
        super().__init__(as_string=as_string, **kwargs)

    # override Number
    def _format_num(self, value):
        num = decimal.Decimal(str(value))
        if self.allow_nan:
            if num.is_nan():
                return decimal.Decimal("NaN")  # avoid sNaN, -sNaN and -NaN
        if self.places is not None and num.is_finite():
            num = num.quantize(self.places, rounding=self.rounding)
        return num

    # override Number
    def _validated(self, value):
        try:
            num = super()._validated(value)
        except decimal.InvalidOperation as error:
            raise self.make_error("invalid") from error
        if not self.allow_nan and (num.is_nan() or num.is_infinite()):
            raise self.make_error("special")
        return num

    # override Number
    def _to_string(self, value):
        return format(value, "f")


class Boolean(Field):
    """A boolean field.

    :param set truthy: Values that will (de)serialize to `True`. If an empty
        set, any non-falsy value will deserialize to `True`. If `None`,
        `marshmallow.fields.Boolean.truthy` will be used.
    :param set falsy: Values that will (de)serialize to `False`. If `None`,
        `marshmallow.fields.Boolean.falsy` will be used.
    :param kwargs: The same keyword arguments that :class:`Field` receives.
    """

    #: Default truthy values.
    truthy = {
        "t",
        "T",
        "true",
        "True",
        "TRUE",
        "on",
        "On",
        "ON",
        "y",
        "Y",
        "yes",
        "Yes",
        "YES",
        "1",
        1,
        True,
    }
    #: Default falsy values.
    falsy = {
        "f",
        "F",
        "false",
        "False",
        "FALSE",
        "off",
        "Off",
        "OFF",
        "n",
        "N",
        "no",
        "No",
        "NO",
        "0",
        0,
        0.0,
        False,
    }

    default_error_messages = {"invalid": "Not a valid boolean."}

    def __init__(self, *, truthy=None, falsy=None, **kwargs):
        super().__init__(**kwargs)

        if truthy is not None:
            self.truthy = set(truthy)
        if falsy is not None:
            self.falsy = set(falsy)

    def _serialize(self, value, attr, obj, **kwargs):
        if value is None:
            return None
        elif value in self.truthy:
            return True
        elif value in self.falsy:
            return False

        return bool(value)

    def _deserialize(self, value, attr, data, **kwargs):
        if not self.truthy:
            return bool(value)
        else:
            try:
                if value in self.truthy:
                    return True
                elif value in self.falsy:
                    return False
            except TypeError as error:
                raise self.make_error("invalid", input=value) from error
        raise self.make_error("invalid", input=value)


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


class NaiveDateTime(DateTime):
    """A formatted naive datetime string.

    :param str format: See :class:`DateTime`.
    :param timezone timezone: Used on deserialization. If `None`,
        aware datetimes are rejected. If not `None`, aware datetimes are
        converted to this timezone before their timezone information is
        removed.
    :param kwargs: The same keyword arguments that :class:`Field` receives.

    .. versionadded:: 3.0.0rc9
    """

    AWARENESS = "naive"

    def __init__(self, format=None, *, timezone=None, **kwargs):
        super().__init__(format=format, **kwargs)
        self.timezone = timezone

    def _deserialize(self, value, attr, data, **kwargs):
        ret = super()._deserialize(value, attr, data, **kwargs)
        if is_aware(ret):
            if self.timezone is None:
                raise self.make_error(
                    "invalid_awareness",
                    awareness=self.AWARENESS,
                    obj_type=self.OBJ_TYPE,
                )
            ret = ret.astimezone(self.timezone).replace(tzinfo=None)
        return ret


class AwareDateTime(DateTime):
    """A formatted aware datetime string.

    :param str format: See :class:`DateTime`.
    :param timezone default_timezone: Used on deserialization. If `None`, naive
        datetimes are rejected. If not `None`, naive datetimes are set this
        timezone.
    :param kwargs: The same keyword arguments that :class:`Field` receives.

    .. versionadded:: 3.0.0rc9
    """

    AWARENESS = "aware"

    def __init__(self, format=None, *, default_timezone=None, **kwargs):
        super().__init__(format=format, **kwargs)
        self.default_timezone = default_timezone

    def _deserialize(self, value, attr, data, **kwargs):
        ret = super()._deserialize(value, attr, data, **kwargs)
        if not is_aware(ret):
            if self.default_timezone is None:
                raise self.make_error(
                    "invalid_awareness",
                    awareness=self.AWARENESS,
                    obj_type=self.OBJ_TYPE,
                )
            ret = ret.replace(tzinfo=self.default_timezone)
        return ret


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


class Date(DateTime):
    """ISO8601-formatted date string.

    :param format: Either ``"iso"`` (for ISO8601) or a date format string.
        If `None`, defaults to "iso".
    :param kwargs: The same keyword arguments that :class:`Field` receives.
    """

    default_error_messages = {
        "invalid": "Not a valid date.",
        "format": '"{input}" cannot be formatted as a date.',
    }

    SERIALIZATION_FUNCS = {"iso": utils.to_iso_date, "iso8601": utils.to_iso_date}

    DESERIALIZATION_FUNCS = {"iso": utils.from_iso_date, "iso8601": utils.from_iso_date}

    DEFAULT_FORMAT = "iso"

    OBJ_TYPE = "date"

    SCHEMA_OPTS_VAR_NAME = "dateformat"

    @staticmethod
    def _make_object_from_format(value, data_format):
        return dt.datetime.strptime(value, data_format).date()


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


class Dict(Mapping):
    """A dict field. Supports dicts and dict-like objects. Extends
    Mapping with dict as the mapping_type.

    Example: ::

        numbers = fields.Dict(keys=fields.Str(), values=fields.Float())

    :param kwargs: The same keyword arguments that :class:`Mapping` receives.

    .. versionadded:: 2.1.0
    """

    mapping_type = dict


class Url(String):
    """A validated URL field. Validation occurs during both serialization and
    deserialization.

    :param default: Default value for the field if the attribute is not set.
    :param str attribute: The name of the attribute to get the value from. If
        `None`, assumes the attribute has the same name as the field.
    :param bool relative: Whether to allow relative URLs.
    :param bool require_tld: Whether to reject non-FQDN hostnames.
    :param kwargs: The same keyword arguments that :class:`String` receives.
    """

    default_error_messages = {"invalid": "Not a valid URL."}

    def __init__(self, *, relative=False, schemes=None, require_tld=True, **kwargs):
        super().__init__(**kwargs)

        self.relative = relative
        self.require_tld = require_tld
        # Insert validation into self.validators so that multiple errors can be
        # stored.
        self.validators.insert(
            0,
            validate.URL(
                relative=self.relative,
                schemes=schemes,
                require_tld=self.require_tld,
                error=self.error_messages["invalid"],
            ),
        )


class Email(String):
    """A validated email field. Validation occurs during both serialization and
    deserialization.

    :param args: The same positional arguments that :class:`String` receives.
    :param kwargs: The same keyword arguments that :class:`String` receives.
    """

    default_error_messages = {"invalid": "Not a valid email address."}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Insert validation into self.validators so that multiple errors can be
        # stored.
        self.validators.insert(0, validate.Email(error=self.error_messages["invalid"]))


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


class Function(Field):
    """A field that takes the value returned by a function.

    :param callable serialize: A callable from which to retrieve the value.
        The function must take a single argument ``obj`` which is the object
        to be serialized. It can also optionally take a ``context`` argument,
        which is a dictionary of context variables passed to the serializer.
        If no callable is provided then the ```load_only``` flag will be set
        to True.
    :param callable deserialize: A callable from which to retrieve the value.
        The function must take a single argument ``value`` which is the value
        to be deserialized. It can also optionally take a ``context`` argument,
        which is a dictionary of context variables passed to the deserializer.
        If no callable is provided then ```value``` will be passed through
        unchanged.

    .. versionchanged:: 2.3.0
        Deprecated ``func`` parameter in favor of ``serialize``.

    .. versionchanged:: 3.0.0a1
        Removed ``func`` parameter.
    """

    _CHECK_ATTRIBUTE = False

    def __init__(self, serialize=None, deserialize=None, **kwargs):
        # Set dump_only and load_only based on arguments
        kwargs["dump_only"] = bool(serialize) and not bool(deserialize)
        kwargs["load_only"] = bool(deserialize) and not bool(serialize)
        super().__init__(**kwargs)
        self.serialize_func = serialize and utils.callable_or_raise(serialize)
        self.deserialize_func = deserialize and utils.callable_or_raise(deserialize)

    def _serialize(self, value, attr, obj, **kwargs):
        return self._call_or_raise(self.serialize_func, obj, attr)

    def _deserialize(self, value, attr, data, **kwargs):
        if self.deserialize_func:
            return self._call_or_raise(self.deserialize_func, value, attr)
        return value

    def _call_or_raise(self, func, value, attr):
        if len(utils.get_func_args(func)) > 1:
            if self.parent.context is None:
                msg = "No context available for Function field {!r}".format(attr)
                raise ValidationError(msg)
            return func(value, self.parent.context)
        else:
            return func(value)


class Constant(Field):
    """A field that (de)serializes to a preset constant.  If you only want the
    constant added for serialization or deserialization, you should use
    ``dump_only=True`` or ``load_only=True`` respectively.

    :param constant: The constant to return for the field attribute.

    .. versionadded:: 2.0.0
    """

    _CHECK_ATTRIBUTE = False

    def __init__(self, constant, **kwargs):
        super().__init__(**kwargs)
        self.constant = constant
        self.missing = constant
        self.default = constant

    def _serialize(self, value, *args, **kwargs):
        return self.constant

    def _deserialize(self, value, *args, **kwargs):
        return self.constant


class Inferred(Field):
    """A field that infers how to serialize, based on the value type.

    .. warning::

        This class is treated as private API.
        Users should not need to use this class directly.
    """

    def __init__(self):
        super().__init__()
        # We memoize the fields to avoid creating and binding new fields
        # every time on serialization.
        self._field_cache = {}

    def _serialize(self, value, attr, obj, **kwargs):
        field_cls = self.root.TYPE_MAPPING.get(type(value))
        if field_cls is None:
            field = super()
        else:
            field = self._field_cache.get(field_cls)
            if field is None:
                field = field_cls()
                field._bind_to_schema(self.name, self.parent)
                self._field_cache[field_cls] = field
        return field._serialize(value, attr, obj, **kwargs)


# Aliases
URL = Url
Str = String
Bool = Boolean
Int = Integer

```
