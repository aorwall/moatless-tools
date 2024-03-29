# django__django-15996

| **django/django** | `b30c0081d4d8a31ab7dc7f72a4c7099af606ef29` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 4254 |
| **Any found context length** | 136 |
| **Avg pos** | 15.0 |
| **Min pos** | 1 |
| **Max pos** | 14 |
| **Top file pos** | 1 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/db/migrations/serializer.py b/django/db/migrations/serializer.py
--- a/django/db/migrations/serializer.py
+++ b/django/db/migrations/serializer.py
@@ -16,7 +16,7 @@
 from django.db.migrations.operations.base import Operation
 from django.db.migrations.utils import COMPILED_REGEX_TYPE, RegexObject
 from django.utils.functional import LazyObject, Promise
-from django.utils.version import get_docs_version
+from django.utils.version import PY311, get_docs_version
 
 
 class BaseSerializer:
@@ -125,8 +125,21 @@ class EnumSerializer(BaseSerializer):
     def serialize(self):
         enum_class = self.value.__class__
         module = enum_class.__module__
+        if issubclass(enum_class, enum.Flag):
+            if PY311:
+                members = list(self.value)
+            else:
+                members, _ = enum._decompose(enum_class, self.value)
+                members = reversed(members)
+        else:
+            members = (self.value,)
         return (
-            "%s.%s[%r]" % (module, enum_class.__qualname__, self.value.name),
+            " | ".join(
+                [
+                    f"{module}.{enum_class.__qualname__}[{item.name!r}]"
+                    for item in members
+                ]
+            ),
             {"import %s" % module},
         )
 

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/db/migrations/serializer.py | 19 | 19 | 14 | 1 | 4254
| django/db/migrations/serializer.py | 128 | 128 | 1 | 1 | 136


## Problem Statement

```
Support for serialization of combination of Enum flags.
Description
	 
		(last modified by Willem Van Onsem)
	 
If we work with a field:
regex_flags = models.IntegerField(default=re.UNICODE | re.IGNORECASE)
This is turned into a migration with:
default=re.RegexFlag[None]
This is due to the fact that the EnumSerializer aims to work with the .name of the item, but if there is no single item for the given value, then there is no such name.
In that case, we can use enum._decompose to obtain a list of names, and create an expression to create the enum value by "ORing" the items together.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 django/db/migrations/serializer.py** | 124 | 143| 136 | 136 | 2687 | 
| 2 | **1 django/db/migrations/serializer.py** | 248 | 261| 137 | 273 | 2687 | 
| 3 | 2 django/db/models/expressions.py | 504 | 600| 683 | 956 | 15812 | 
| 4 | 3 django/db/models/enums.py | 34 | 56| 183 | 1139 | 16425 | 
| 5 | **3 django/db/migrations/serializer.py** | 314 | 353| 297 | 1436 | 16425 | 
| 6 | 3 django/db/models/expressions.py | 464 | 502| 280 | 1716 | 16425 | 
| 7 | 3 django/db/models/expressions.py | 650 | 679| 252 | 1968 | 16425 | 
| 8 | 4 django/utils/regex_helper.py | 288 | 354| 482 | 2450 | 19066 | 
| 9 | 4 django/db/models/expressions.py | 37 | 145| 819 | 3269 | 19066 | 
| 10 | 5 django/db/models/constraints.py | 1 | 13| 111 | 3380 | 21994 | 
| 11 | 5 django/db/models/expressions.py | 147 | 163| 137 | 3517 | 21994 | 
| 12 | 6 django/forms/fields.py | 580 | 612| 212 | 3729 | 31630 | 
| 13 | **6 django/db/migrations/serializer.py** | 196 | 207| 117 | 3846 | 31630 | 
| **-> 14 <-** | **6 django/db/migrations/serializer.py** | 1 | 78| 408 | 4254 | 31630 | 
| 15 | **6 django/db/migrations/serializer.py** | 210 | 245| 281 | 4535 | 31630 | 
| 16 | 6 django/db/models/enums.py | 1 | 32| 245 | 4780 | 31630 | 
| 17 | 7 django/core/serializers/xml_serializer.py | 69 | 100| 228 | 5008 | 35226 | 
| 18 | 7 django/db/models/enums.py | 59 | 93| 191 | 5199 | 35226 | 
| 19 | 8 django/db/models/lookups.py | 578 | 595| 134 | 5333 | 40544 | 
| 20 | **8 django/db/migrations/serializer.py** | 111 | 121| 114 | 5447 | 40544 | 
| 21 | 9 django/db/models/fields/reverse_related.py | 20 | 151| 765 | 6212 | 43076 | 
| 22 | 9 django/db/models/expressions.py | 631 | 648| 145 | 6357 | 43076 | 
| 23 | 9 django/core/serializers/xml_serializer.py | 127 | 173| 366 | 6723 | 43076 | 
| 24 | 9 django/db/models/expressions.py | 603 | 628| 210 | 6933 | 43076 | 
| 25 | **9 django/db/migrations/serializer.py** | 290 | 311| 166 | 7099 | 43076 | 
| 26 | 10 django/db/models/fields/__init__.py | 1220 | 1243| 153 | 7252 | 61799 | 
| 27 | 10 django/db/models/fields/__init__.py | 2251 | 2293| 210 | 7462 | 61799 | 
| 28 | **10 django/db/migrations/serializer.py** | 264 | 287| 169 | 7631 | 61799 | 
| 29 | 10 django/db/models/expressions.py | 681 | 726| 335 | 7966 | 61799 | 
| 30 | 10 django/core/serializers/xml_serializer.py | 102 | 125| 196 | 8162 | 61799 | 
| 31 | 10 django/utils/regex_helper.py | 1 | 38| 250 | 8412 | 61799 | 
| 32 | 10 django/db/models/expressions.py | 1095 | 1132| 298 | 8710 | 61799 | 
| 33 | 10 django/db/models/fields/__init__.py | 2548 | 2609| 425 | 9135 | 61799 | 
| 34 | **10 django/db/migrations/serializer.py** | 81 | 108| 233 | 9368 | 61799 | 
| 35 | 10 django/db/models/fields/__init__.py | 2048 | 2069| 123 | 9491 | 61799 | 
| 36 | 11 django/db/models/options.py | 60 | 83| 203 | 9694 | 69387 | 
| 37 | 11 django/db/models/fields/reverse_related.py | 303 | 333| 159 | 9853 | 69387 | 
| 38 | 12 django/contrib/admin/migrations/0003_logentry_add_action_flag_choices.py | 1 | 21| 112 | 9965 | 69499 | 
| 39 | 13 django/core/serializers/base.py | 297 | 322| 224 | 10189 | 72175 | 
| 40 | 13 django/db/models/fields/reverse_related.py | 336 | 397| 366 | 10555 | 72175 | 
| 41 | 14 django/core/serializers/python.py | 64 | 85| 160 | 10715 | 73484 | 
| 42 | 14 django/db/models/lookups.py | 686 | 719| 141 | 10856 | 73484 | 
| 43 | 14 django/db/models/expressions.py | 1 | 34| 228 | 11084 | 73484 | 
| 44 | 15 django/db/models/fields/related.py | 604 | 670| 497 | 11581 | 88093 | 
| 45 | 15 django/core/serializers/base.py | 386 | 411| 209 | 11790 | 88093 | 
| 46 | 16 django/db/migrations/autodetector.py | 1 | 18| 115 | 11905 | 101552 | 
| 47 | **16 django/db/migrations/serializer.py** | 174 | 193| 156 | 12061 | 101552 | 
| 48 | 16 django/db/models/options.py | 626 | 654| 231 | 12292 | 101552 | 
| 49 | 16 django/utils/regex_helper.py | 78 | 192| 964 | 13256 | 101552 | 
| 50 | 16 django/db/models/fields/__init__.py | 1095 | 1114| 223 | 13479 | 101552 | 
| 51 | 16 django/db/models/fields/__init__.py | 643 | 664| 228 | 13707 | 101552 | 
| 52 | 17 django/db/migrations/utils.py | 1 | 24| 134 | 13841 | 102423 | 
| 53 | 17 django/utils/regex_helper.py | 41 | 76| 375 | 14216 | 102423 | 
| 54 | 18 django/core/serializers/__init__.py | 91 | 146| 369 | 14585 | 104193 | 
| 55 | 18 django/db/models/fields/__init__.py | 1 | 110| 663 | 15248 | 104193 | 
| 56 | 19 django/utils/jslex.py | 31 | 48| 126 | 15374 | 106021 | 
| 57 | 20 django/contrib/postgres/search.py | 68 | 84| 129 | 15503 | 108490 | 
| 58 | 21 django/contrib/gis/geometry.py | 1 | 18| 215 | 15718 | 108705 | 
| 59 | 21 django/db/models/expressions.py | 1024 | 1050| 195 | 15913 | 108705 | 
| 60 | 22 django/contrib/gis/serializers/geojson.py | 47 | 82| 308 | 16221 | 109339 | 
| 61 | 22 django/db/migrations/autodetector.py | 1432 | 1477| 318 | 16539 | 109339 | 
| 62 | 23 django/db/models/query_utils.py | 35 | 80| 282 | 16821 | 112132 | 
| 63 | 24 django/forms/models.py | 1520 | 1554| 254 | 17075 | 124326 | 
| 64 | 24 django/db/migrations/autodetector.py | 908 | 981| 623 | 17698 | 124326 | 
| 65 | 24 django/forms/models.py | 1632 | 1660| 207 | 17905 | 124326 | 
| 66 | 24 django/core/serializers/base.py | 351 | 383| 227 | 18132 | 124326 | 
| 67 | 24 django/db/models/fields/reverse_related.py | 241 | 300| 386 | 18518 | 124326 | 
| 68 | 24 django/db/models/fields/__init__.py | 984 | 1006| 162 | 18680 | 124326 | 
| 69 | 25 django/contrib/contenttypes/fields.py | 471 | 498| 258 | 18938 | 129957 | 
| 70 | 26 django/contrib/admin/widgets.py | 384 | 453| 373 | 19311 | 134147 | 
| 71 | 26 django/db/models/expressions.py | 795 | 827| 227 | 19538 | 134147 | 
| 72 | 26 django/db/models/lookups.py | 360 | 411| 306 | 19844 | 134147 | 
| 73 | 27 django/db/models/__init__.py | 1 | 116| 682 | 20526 | 134829 | 
| 74 | 27 django/db/models/expressions.py | 1397 | 1429| 302 | 20828 | 134829 | 
| 75 | 28 django/db/models/functions/comparison.py | 104 | 119| 178 | 21006 | 136584 | 
| 76 | 29 django/db/migrations/questioner.py | 57 | 87| 255 | 21261 | 139280 | 
| 77 | 29 django/db/models/fields/reverse_related.py | 1 | 17| 120 | 21381 | 139280 | 
| 78 | 30 django/db/models/fields/json.py | 498 | 588| 515 | 21896 | 143693 | 
| 79 | 31 django/contrib/auth/validators.py | 1 | 26| 165 | 22061 | 143859 | 
| 80 | 31 django/db/models/lookups.py | 1 | 50| 360 | 22421 | 143859 | 
| 81 | 31 django/db/models/fields/__init__.py | 2752 | 2774| 143 | 22564 | 143859 | 
| 82 | 31 django/db/models/fields/reverse_related.py | 205 | 238| 306 | 22870 | 143859 | 
| 83 | 31 django/db/models/options.py | 286 | 328| 355 | 23225 | 143859 | 
| 84 | 31 django/db/models/expressions.py | 1221 | 1251| 242 | 23467 | 143859 | 
| 85 | 31 django/db/models/options.py | 1 | 57| 347 | 23814 | 143859 | 
| 86 | 31 django/db/models/expressions.py | 996 | 1022| 258 | 24072 | 143859 | 
| 87 | 32 django/db/models/sql/query.py | 679 | 719| 383 | 24455 | 167030 | 
| 88 | 33 django/core/management/commands/loaddata.py | 111 | 137| 267 | 24722 | 170216 | 
| 89 | 33 django/db/models/lookups.py | 166 | 186| 200 | 24922 | 170216 | 
| 90 | **33 django/db/migrations/serializer.py** | 356 | 383| 231 | 25153 | 170216 | 
| 91 | 33 django/db/models/lookups.py | 521 | 575| 308 | 25461 | 170216 | 
| 92 | 33 django/db/models/fields/__init__.py | 494 | 571| 669 | 26130 | 170216 | 
| 93 | 34 django/db/models/indexes.py | 232 | 296| 487 | 26617 | 172602 | 
| 94 | 34 django/db/models/fields/__init__.py | 1952 | 1976| 147 | 26764 | 172602 | 
| 95 | 34 django/db/models/fields/json.py | 284 | 308| 185 | 26949 | 172602 | 
| 96 | 34 django/db/models/fields/json.py | 71 | 127| 366 | 27315 | 172602 | 
| 97 | 34 django/db/models/fields/__init__.py | 2612 | 2664| 343 | 27658 | 172602 | 
| 98 | 35 django/db/backends/base/schema.py | 39 | 72| 214 | 27872 | 186463 | 
| 99 | 36 django/contrib/postgres/fields/hstore.py | 1 | 71| 440 | 28312 | 187168 | 


### Hint

```
patch of the EnumSerializer
```

## Patch

```diff
diff --git a/django/db/migrations/serializer.py b/django/db/migrations/serializer.py
--- a/django/db/migrations/serializer.py
+++ b/django/db/migrations/serializer.py
@@ -16,7 +16,7 @@
 from django.db.migrations.operations.base import Operation
 from django.db.migrations.utils import COMPILED_REGEX_TYPE, RegexObject
 from django.utils.functional import LazyObject, Promise
-from django.utils.version import get_docs_version
+from django.utils.version import PY311, get_docs_version
 
 
 class BaseSerializer:
@@ -125,8 +125,21 @@ class EnumSerializer(BaseSerializer):
     def serialize(self):
         enum_class = self.value.__class__
         module = enum_class.__module__
+        if issubclass(enum_class, enum.Flag):
+            if PY311:
+                members = list(self.value)
+            else:
+                members, _ = enum._decompose(enum_class, self.value)
+                members = reversed(members)
+        else:
+            members = (self.value,)
         return (
-            "%s.%s[%r]" % (module, enum_class.__qualname__, self.value.name),
+            " | ".join(
+                [
+                    f"{module}.{enum_class.__qualname__}[{item.name!r}]"
+                    for item in members
+                ]
+            ),
             {"import %s" % module},
         )
 

```

## Test Patch

```diff
diff --git a/tests/migrations/test_writer.py b/tests/migrations/test_writer.py
--- a/tests/migrations/test_writer.py
+++ b/tests/migrations/test_writer.py
@@ -413,6 +413,14 @@ def test_serialize_enum_flags(self):
             "(2, migrations.test_writer.IntFlagEnum['B'])], "
             "default=migrations.test_writer.IntFlagEnum['A'])",
         )
+        self.assertSerializedResultEqual(
+            IntFlagEnum.A | IntFlagEnum.B,
+            (
+                "migrations.test_writer.IntFlagEnum['A'] | "
+                "migrations.test_writer.IntFlagEnum['B']",
+                {"import migrations.test_writer"},
+            ),
+        )
 
     def test_serialize_choices(self):
         class TextChoices(models.TextChoices):

```


## Code snippets

### 1 - django/db/migrations/serializer.py:

Start line: 124, End line: 143

```python
class EnumSerializer(BaseSerializer):
    def serialize(self):
        enum_class = self.value.__class__
        module = enum_class.__module__
        return (
            "%s.%s[%r]" % (module, enum_class.__qualname__, self.value.name),
            {"import %s" % module},
        )


class FloatSerializer(BaseSimpleSerializer):
    def serialize(self):
        if math.isnan(self.value) or math.isinf(self.value):
            return 'float("{}")'.format(self.value), set()
        return super().serialize()


class FrozensetSerializer(BaseSequenceSerializer):
    def _format(self):
        return "frozenset([%s])"
```
### 2 - django/db/migrations/serializer.py:

Start line: 248, End line: 261

```python
class RegexSerializer(BaseSerializer):
    def serialize(self):
        regex_pattern, pattern_imports = serializer_factory(
            self.value.pattern
        ).serialize()
        # Turn off default implicit flags (e.g. re.U) because regexes with the
        # same implicit and explicit flags aren't equal.
        flags = self.value.flags ^ re.compile("").flags
        regex_flags, flag_imports = serializer_factory(flags).serialize()
        imports = {"import re", *pattern_imports, *flag_imports}
        args = [regex_pattern]
        if flags:
            args.append(regex_flags)
        return "re.compile(%s)" % ", ".join(args), imports
```
### 3 - django/db/models/expressions.py:

Start line: 504, End line: 600

```python
_connector_combinations = [
    # Numeric operations - operands of same type.
    {
        connector: [
            (fields.IntegerField, fields.IntegerField, fields.IntegerField),
            (fields.FloatField, fields.FloatField, fields.FloatField),
            (fields.DecimalField, fields.DecimalField, fields.DecimalField),
        ]
        for connector in (
            Combinable.ADD,
            Combinable.SUB,
            Combinable.MUL,
            # Behavior for DIV with integer arguments follows Postgres/SQLite,
            # not MySQL/Oracle.
            Combinable.DIV,
            Combinable.MOD,
            Combinable.POW,
        )
    },
    # Numeric operations - operands of different type.
    {
        connector: [
            (fields.IntegerField, fields.DecimalField, fields.DecimalField),
            (fields.DecimalField, fields.IntegerField, fields.DecimalField),
            (fields.IntegerField, fields.FloatField, fields.FloatField),
            (fields.FloatField, fields.IntegerField, fields.FloatField),
        ]
        for connector in (
            Combinable.ADD,
            Combinable.SUB,
            Combinable.MUL,
            Combinable.DIV,
        )
    },
    # Bitwise operators.
    {
        connector: [
            (fields.IntegerField, fields.IntegerField, fields.IntegerField),
        ]
        for connector in (
            Combinable.BITAND,
            Combinable.BITOR,
            Combinable.BITLEFTSHIFT,
            Combinable.BITRIGHTSHIFT,
            Combinable.BITXOR,
        )
    },
    # Numeric with NULL.
    {
        connector: [
            (field_type, NoneType, field_type),
            (NoneType, field_type, field_type),
        ]
        for connector in (
            Combinable.ADD,
            Combinable.SUB,
            Combinable.MUL,
            Combinable.DIV,
            Combinable.MOD,
            Combinable.POW,
        )
        for field_type in (fields.IntegerField, fields.DecimalField, fields.FloatField)
    },
    # Date/DateTimeField/DurationField/TimeField.
    {
        Combinable.ADD: [
            # Date/DateTimeField.
            (fields.DateField, fields.DurationField, fields.DateTimeField),
            (fields.DateTimeField, fields.DurationField, fields.DateTimeField),
            (fields.DurationField, fields.DateField, fields.DateTimeField),
            (fields.DurationField, fields.DateTimeField, fields.DateTimeField),
            # DurationField.
            (fields.DurationField, fields.DurationField, fields.DurationField),
            # TimeField.
            (fields.TimeField, fields.DurationField, fields.TimeField),
            (fields.DurationField, fields.TimeField, fields.TimeField),
        ],
    },
    {
        Combinable.SUB: [
            # Date/DateTimeField.
            (fields.DateField, fields.DurationField, fields.DateTimeField),
            (fields.DateTimeField, fields.DurationField, fields.DateTimeField),
            (fields.DateField, fields.DateField, fields.DurationField),
            (fields.DateField, fields.DateTimeField, fields.DurationField),
            (fields.DateTimeField, fields.DateField, fields.DurationField),
            (fields.DateTimeField, fields.DateTimeField, fields.DurationField),
            # DurationField.
            (fields.DurationField, fields.DurationField, fields.DurationField),
            # TimeField.
            (fields.TimeField, fields.DurationField, fields.TimeField),
            (fields.TimeField, fields.TimeField, fields.DurationField),
        ],
    },
]

_connector_combinators = defaultdict(list)
```
### 4 - django/db/models/enums.py:

Start line: 34, End line: 56

```python
class ChoicesMeta(enum.EnumMeta):

    def __contains__(cls, member):
        if not isinstance(member, enum.Enum):
            # Allow non-enums to match against member values.
            return any(x.value == member for x in cls)
        return super().__contains__(member)

    @property
    def names(cls):
        empty = ["__empty__"] if hasattr(cls, "__empty__") else []
        return empty + [member.name for member in cls]

    @property
    def choices(cls):
        empty = [(None, cls.__empty__)] if hasattr(cls, "__empty__") else []
        return empty + [(member.value, member.label) for member in cls]

    @property
    def labels(cls):
        return [label for _, label in cls.choices]

    @property
    def values(cls):
        return [value for value, _ in cls.choices]
```
### 5 - django/db/migrations/serializer.py:

Start line: 314, End line: 353

```python
class Serializer:
    _registry = {
        # Some of these are order-dependent.
        frozenset: FrozensetSerializer,
        list: SequenceSerializer,
        set: SetSerializer,
        tuple: TupleSerializer,
        dict: DictionarySerializer,
        models.Choices: ChoicesSerializer,
        enum.Enum: EnumSerializer,
        datetime.datetime: DatetimeDatetimeSerializer,
        (datetime.date, datetime.timedelta, datetime.time): DateTimeSerializer,
        SettingsReference: SettingsReferenceSerializer,
        float: FloatSerializer,
        (bool, int, type(None), bytes, str, range): BaseSimpleSerializer,
        decimal.Decimal: DecimalSerializer,
        (functools.partial, functools.partialmethod): FunctoolsPartialSerializer,
        (
            types.FunctionType,
            types.BuiltinFunctionType,
            types.MethodType,
        ): FunctionTypeSerializer,
        collections.abc.Iterable: IterableSerializer,
        (COMPILED_REGEX_TYPE, RegexObject): RegexSerializer,
        uuid.UUID: UUIDSerializer,
        pathlib.PurePath: PathSerializer,
        os.PathLike: PathLikeSerializer,
    }

    @classmethod
    def register(cls, type_, serializer):
        if not issubclass(serializer, BaseSerializer):
            raise ValueError(
                "'%s' must inherit from 'BaseSerializer'." % serializer.__name__
            )
        cls._registry[type_] = serializer

    @classmethod
    def unregister(cls, type_):
        cls._registry.pop(type_)
```
### 6 - django/db/models/expressions.py:

Start line: 464, End line: 502

```python
@deconstructible
class Expression(BaseExpression, Combinable):
    """An expression that can be combined with other expressions."""

    @cached_property
    def identity(self):
        constructor_signature = inspect.signature(self.__init__)
        args, kwargs = self._constructor_args
        signature = constructor_signature.bind_partial(*args, **kwargs)
        signature.apply_defaults()
        arguments = signature.arguments.items()
        identity = [self.__class__]
        for arg, value in arguments:
            if isinstance(value, fields.Field):
                if value.name and value.model:
                    value = (value.model._meta.label, value.name)
                else:
                    value = type(value)
            else:
                value = make_hashable(value)
            identity.append((arg, value))
        return tuple(identity)

    def __eq__(self, other):
        if not isinstance(other, Expression):
            return NotImplemented
        return other.identity == self.identity

    def __hash__(self):
        return hash(self.identity)


# Type inference for CombinedExpression.output_field.
# Missing items will result in FieldError, by design.
#
# The current approach for NULL is based on lowest common denominator behavior
# i.e. if one of the supported databases is raising an error (rather than
# return NULL) for `val <op> NULL`, then Django raises FieldError.
NoneType = type(None)
```
### 7 - django/db/models/expressions.py:

Start line: 650, End line: 679

```python
class CombinedExpression(SQLiteNumericMixin, Expression):

    def _resolve_output_field(self):
        # We avoid using super() here for reasons given in
        # Expression._resolve_output_field()
        combined_type = _resolve_combined_type(
            self.connector,
            type(self.lhs._output_field_or_none),
            type(self.rhs._output_field_or_none),
        )
        if combined_type is None:
            raise FieldError(
                f"Cannot infer type of {self.connector!r} expression involving these "
                f"types: {self.lhs.output_field.__class__.__name__}, "
                f"{self.rhs.output_field.__class__.__name__}. You must set "
                f"output_field."
            )
        return combined_type()

    def as_sql(self, compiler, connection):
        expressions = []
        expression_params = []
        sql, params = compiler.compile(self.lhs)
        expressions.append(sql)
        expression_params.extend(params)
        sql, params = compiler.compile(self.rhs)
        expressions.append(sql)
        expression_params.extend(params)
        # order of precedence
        expression_wrapper = "(%s)"
        sql = connection.ops.combine_expression(self.connector, expressions)
        return expression_wrapper % sql, expression_params
```
### 8 - django/utils/regex_helper.py:

Start line: 288, End line: 354

```python
def flatten_result(source):
    """
    Turn the given source sequence into a list of reg-exp possibilities and
    their arguments. Return a list of strings and a list of argument lists.
    Each of the two lists will be of the same length.
    """
    if source is None:
        return [""], [[]]
    if isinstance(source, Group):
        if source[1] is None:
            params = []
        else:
            params = [source[1]]
        return [source[0]], [params]
    result = [""]
    result_args = [[]]
    pos = last = 0
    for pos, elt in enumerate(source):
        if isinstance(elt, str):
            continue
        piece = "".join(source[last:pos])
        if isinstance(elt, Group):
            piece += elt[0]
            param = elt[1]
        else:
            param = None
        last = pos + 1
        for i in range(len(result)):
            result[i] += piece
            if param:
                result_args[i].append(param)
        if isinstance(elt, (Choice, NonCapture)):
            if isinstance(elt, NonCapture):
                elt = [elt]
            inner_result, inner_args = [], []
            for item in elt:
                res, args = flatten_result(item)
                inner_result.extend(res)
                inner_args.extend(args)
            new_result = []
            new_args = []
            for item, args in zip(result, result_args):
                for i_item, i_args in zip(inner_result, inner_args):
                    new_result.append(item + i_item)
                    new_args.append(args[:] + i_args)
            result = new_result
            result_args = new_args
    if pos >= last:
        piece = "".join(source[last:])
        for i in range(len(result)):
            result[i] += piece
    return result, result_args


def _lazy_re_compile(regex, flags=0):
    """Lazily compile a regex with flags."""

    def _compile():
        # Compile the regex if it was not passed pre-compiled.
        if isinstance(regex, (str, bytes)):
            return re.compile(regex, flags)
        else:
            assert not flags, "flags must be empty if regex is passed pre-compiled"
            return regex

    return SimpleLazyObject(_compile)
```
### 9 - django/db/models/expressions.py:

Start line: 37, End line: 145

```python
class Combinable:
    """
    Provide the ability to combine one or two objects with
    some connector. For example F('foo') + F('bar').
    """

    # Arithmetic connectors
    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"
    POW = "^"
    # The following is a quoted % operator - it is quoted because it can be
    # used in strings that also have parameter substitution.
    MOD = "%%"

    # Bitwise operators - note that these are generated by .bitand()
    # and .bitor(), the '&' and '|' are reserved for boolean operator
    # usage.
    BITAND = "&"
    BITOR = "|"
    BITLEFTSHIFT = "<<"
    BITRIGHTSHIFT = ">>"
    BITXOR = "#"

    def _combine(self, other, connector, reversed):
        if not hasattr(other, "resolve_expression"):
            # everything must be resolvable to an expression
            other = Value(other)

        if reversed:
            return CombinedExpression(other, connector, self)
        return CombinedExpression(self, connector, other)

    #############
    # OPERATORS #
    #############

    def __neg__(self):
        return self._combine(-1, self.MUL, False)

    def __add__(self, other):
        return self._combine(other, self.ADD, False)

    def __sub__(self, other):
        return self._combine(other, self.SUB, False)

    def __mul__(self, other):
        return self._combine(other, self.MUL, False)

    def __truediv__(self, other):
        return self._combine(other, self.DIV, False)

    def __mod__(self, other):
        return self._combine(other, self.MOD, False)

    def __pow__(self, other):
        return self._combine(other, self.POW, False)

    def __and__(self, other):
        if getattr(self, "conditional", False) and getattr(other, "conditional", False):
            return Q(self) & Q(other)
        raise NotImplementedError(
            "Use .bitand(), .bitor(), and .bitxor() for bitwise logical operations."
        )

    def bitand(self, other):
        return self._combine(other, self.BITAND, False)

    def bitleftshift(self, other):
        return self._combine(other, self.BITLEFTSHIFT, False)

    def bitrightshift(self, other):
        return self._combine(other, self.BITRIGHTSHIFT, False)

    def __xor__(self, other):
        if getattr(self, "conditional", False) and getattr(other, "conditional", False):
            return Q(self) ^ Q(other)
        raise NotImplementedError(
            "Use .bitand(), .bitor(), and .bitxor() for bitwise logical operations."
        )

    def bitxor(self, other):
        return self._combine(other, self.BITXOR, False)

    def __or__(self, other):
        if getattr(self, "conditional", False) and getattr(other, "conditional", False):
            return Q(self) | Q(other)
        raise NotImplementedError(
            "Use .bitand(), .bitor(), and .bitxor() for bitwise logical operations."
        )

    def bitor(self, other):
        return self._combine(other, self.BITOR, False)

    def __radd__(self, other):
        return self._combine(other, self.ADD, True)

    def __rsub__(self, other):
        return self._combine(other, self.SUB, True)

    def __rmul__(self, other):
        return self._combine(other, self.MUL, True)

    def __rtruediv__(self, other):
        return self._combine(other, self.DIV, True)

    def __rmod__(self, other):
        return self._combine(other, self.MOD, True)
```
### 10 - django/db/models/constraints.py:

Start line: 1, End line: 13

```python
from enum import Enum

from django.core.exceptions import FieldError, ValidationError
from django.db import connections
from django.db.models.expressions import Exists, ExpressionList, F
from django.db.models.indexes import IndexExpression
from django.db.models.lookups import Exact
from django.db.models.query_utils import Q
from django.db.models.sql.query import Query
from django.db.utils import DEFAULT_DB_ALIAS
from django.utils.translation import gettext_lazy as _

__all__ = ["BaseConstraint", "CheckConstraint", "Deferrable", "UniqueConstraint"]
```
### 13 - django/db/migrations/serializer.py:

Start line: 196, End line: 207

```python
class IterableSerializer(BaseSerializer):
    def serialize(self):
        imports = set()
        strings = []
        for item in self.value:
            item_string, item_imports = serializer_factory(item).serialize()
            imports.update(item_imports)
            strings.append(item_string)
        # When len(strings)==0, the empty iterable should be serialized as
        # "()", not "(,)" because (,) is invalid Python syntax.
        value = "(%s)" if len(strings) != 1 else "(%s,)"
        return value % (", ".join(strings)), imports
```
### 14 - django/db/migrations/serializer.py:

Start line: 1, End line: 78

```python
import builtins
import collections.abc
import datetime
import decimal
import enum
import functools
import math
import os
import pathlib
import re
import types
import uuid

from django.conf import SettingsReference
from django.db import models
from django.db.migrations.operations.base import Operation
from django.db.migrations.utils import COMPILED_REGEX_TYPE, RegexObject
from django.utils.functional import LazyObject, Promise
from django.utils.version import get_docs_version


class BaseSerializer:
    def __init__(self, value):
        self.value = value

    def serialize(self):
        raise NotImplementedError(
            "Subclasses of BaseSerializer must implement the serialize() method."
        )


class BaseSequenceSerializer(BaseSerializer):
    def _format(self):
        raise NotImplementedError(
            "Subclasses of BaseSequenceSerializer must implement the _format() method."
        )

    def serialize(self):
        imports = set()
        strings = []
        for item in self.value:
            item_string, item_imports = serializer_factory(item).serialize()
            imports.update(item_imports)
            strings.append(item_string)
        value = self._format()
        return value % (", ".join(strings)), imports


class BaseSimpleSerializer(BaseSerializer):
    def serialize(self):
        return repr(self.value), set()


class ChoicesSerializer(BaseSerializer):
    def serialize(self):
        return serializer_factory(self.value.value).serialize()


class DateTimeSerializer(BaseSerializer):
    """For datetime.*, except datetime.datetime."""

    def serialize(self):
        return repr(self.value), {"import datetime"}


class DatetimeDatetimeSerializer(BaseSerializer):
    """For datetime.datetime."""

    def serialize(self):
        if self.value.tzinfo is not None and self.value.tzinfo != datetime.timezone.utc:
            self.value = self.value.astimezone(datetime.timezone.utc)
        imports = ["import datetime"]
        return repr(self.value), set(imports)


class DecimalSerializer(BaseSerializer):
    def serialize(self):
        return repr(self.value), {"from decimal import Decimal"}
```
### 15 - django/db/migrations/serializer.py:

Start line: 210, End line: 245

```python
class ModelFieldSerializer(DeconstructableSerializer):
    def serialize(self):
        attr_name, path, args, kwargs = self.value.deconstruct()
        return self.serialize_deconstructed(path, args, kwargs)


class ModelManagerSerializer(DeconstructableSerializer):
    def serialize(self):
        as_manager, manager_path, qs_path, args, kwargs = self.value.deconstruct()
        if as_manager:
            name, imports = self._serialize_path(qs_path)
            return "%s.as_manager()" % name, imports
        else:
            return self.serialize_deconstructed(manager_path, args, kwargs)


class OperationSerializer(BaseSerializer):
    def serialize(self):
        from django.db.migrations.writer import OperationWriter

        string, imports = OperationWriter(self.value, indentation=0).serialize()
        # Nested operation, trailing comma is handled in upper OperationWriter._write()
        return string.rstrip(","), imports


class PathLikeSerializer(BaseSerializer):
    def serialize(self):
        return repr(os.fspath(self.value)), {}


class PathSerializer(BaseSerializer):
    def serialize(self):
        # Convert concrete paths to pure paths to avoid issues with migrations
        # generated on one platform being used on a different platform.
        prefix = "Pure" if isinstance(self.value, pathlib.Path) else ""
        return "pathlib.%s%r" % (prefix, self.value), {"import pathlib"}
```
### 20 - django/db/migrations/serializer.py:

Start line: 111, End line: 121

```python
class DictionarySerializer(BaseSerializer):
    def serialize(self):
        imports = set()
        strings = []
        for k, v in sorted(self.value.items()):
            k_string, k_imports = serializer_factory(k).serialize()
            v_string, v_imports = serializer_factory(v).serialize()
            imports.update(k_imports)
            imports.update(v_imports)
            strings.append((k_string, v_string))
        return "{%s}" % (", ".join("%s: %s" % (k, v) for k, v in strings)), imports
```
### 25 - django/db/migrations/serializer.py:

Start line: 290, End line: 311

```python
class TypeSerializer(BaseSerializer):
    def serialize(self):
        special_cases = [
            (models.Model, "models.Model", ["from django.db import models"]),
            (type(None), "type(None)", []),
        ]
        for case, string, imports in special_cases:
            if case is self.value:
                return string, set(imports)
        if hasattr(self.value, "__module__"):
            module = self.value.__module__
            if module == builtins.__name__:
                return self.value.__name__, set()
            else:
                return "%s.%s" % (module, self.value.__qualname__), {
                    "import %s" % module
                }


class UUIDSerializer(BaseSerializer):
    def serialize(self):
        return "uuid.%s" % repr(self.value), {"import uuid"}
```
### 28 - django/db/migrations/serializer.py:

Start line: 264, End line: 287

```python
class SequenceSerializer(BaseSequenceSerializer):
    def _format(self):
        return "[%s]"


class SetSerializer(BaseSequenceSerializer):
    def _format(self):
        # Serialize as a set literal except when value is empty because {}
        # is an empty dict.
        return "{%s}" if self.value else "set(%s)"


class SettingsReferenceSerializer(BaseSerializer):
    def serialize(self):
        return "settings.%s" % self.value.setting_name, {
            "from django.conf import settings"
        }


class TupleSerializer(BaseSequenceSerializer):
    def _format(self):
        # When len(value)==0, the empty tuple should be serialized as "()",
        # not "(,)" because (,) is invalid Python syntax.
        return "(%s)" if len(self.value) != 1 else "(%s,)"
```
### 34 - django/db/migrations/serializer.py:

Start line: 81, End line: 108

```python
class DeconstructableSerializer(BaseSerializer):
    @staticmethod
    def serialize_deconstructed(path, args, kwargs):
        name, imports = DeconstructableSerializer._serialize_path(path)
        strings = []
        for arg in args:
            arg_string, arg_imports = serializer_factory(arg).serialize()
            strings.append(arg_string)
            imports.update(arg_imports)
        for kw, arg in sorted(kwargs.items()):
            arg_string, arg_imports = serializer_factory(arg).serialize()
            imports.update(arg_imports)
            strings.append("%s=%s" % (kw, arg_string))
        return "%s(%s)" % (name, ", ".join(strings)), imports

    @staticmethod
    def _serialize_path(path):
        module, name = path.rsplit(".", 1)
        if module == "django.db.models":
            imports = {"from django.db import models"}
            name = "models.%s" % name
        else:
            imports = {"import %s" % module}
            name = path
        return name, imports

    def serialize(self):
        return self.serialize_deconstructed(*self.value.deconstruct())
```
### 47 - django/db/migrations/serializer.py:

Start line: 174, End line: 193

```python
class FunctoolsPartialSerializer(BaseSerializer):
    def serialize(self):
        # Serialize functools.partial() arguments
        func_string, func_imports = serializer_factory(self.value.func).serialize()
        args_string, args_imports = serializer_factory(self.value.args).serialize()
        keywords_string, keywords_imports = serializer_factory(
            self.value.keywords
        ).serialize()
        # Add any imports needed by arguments
        imports = {"import functools", *func_imports, *args_imports, *keywords_imports}
        return (
            "functools.%s(%s, *%s, **%s)"
            % (
                self.value.__class__.__name__,
                func_string,
                args_string,
                keywords_string,
            ),
            imports,
        )
```
### 90 - django/db/migrations/serializer.py:

Start line: 356, End line: 383

```python
def serializer_factory(value):
    if isinstance(value, Promise):
        value = str(value)
    elif isinstance(value, LazyObject):
        # The unwrapped value is returned as the first item of the arguments
        # tuple.
        value = value.__reduce__()[1][0]

    if isinstance(value, models.Field):
        return ModelFieldSerializer(value)
    if isinstance(value, models.manager.BaseManager):
        return ModelManagerSerializer(value)
    if isinstance(value, Operation):
        return OperationSerializer(value)
    if isinstance(value, type):
        return TypeSerializer(value)
    # Anything that knows how to deconstruct itself.
    if hasattr(value, "deconstruct"):
        return DeconstructableSerializer(value)
    for type_, serializer_cls in Serializer._registry.items():
        if isinstance(value, type_):
            return serializer_cls(value)
    raise ValueError(
        "Cannot serialize: %r\nThere are some values Django cannot serialize into "
        "migration files.\nFor more, see https://docs.djangoproject.com/en/%s/"
        "topics/migrations/#migration-serializing" % (value, get_docs_version())
    )
```
