# django__django-17087

| **django/django** | `4a72da71001f154ea60906a2f74898d32b7322a7` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | - |
| **Avg pos** | - |
| **Min pos** | - |
| **Max pos** | - |
| **Top file pos** | 28 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/db/migrations/serializer.py b/django/db/migrations/serializer.py
--- a/django/db/migrations/serializer.py
+++ b/django/db/migrations/serializer.py
@@ -168,7 +168,7 @@ def serialize(self):
         ):
             klass = self.value.__self__
             module = klass.__module__
-            return "%s.%s.%s" % (module, klass.__name__, self.value.__name__), {
+            return "%s.%s.%s" % (module, klass.__qualname__, self.value.__name__), {
                 "import %s" % module
             }
         # Further error checking

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/db/migrations/serializer.py | 171 | 171 | - | 28 | -


## Problem Statement

```
Class methods from nested classes cannot be used as Field.default.
Description
	 
		(last modified by Mariusz Felisiak)
	 
Given the following model:
 
class Profile(models.Model):
	class Capability(models.TextChoices):
		BASIC = ("BASIC", "Basic")
		PROFESSIONAL = ("PROFESSIONAL", "Professional")
		
		@classmethod
		def default(cls) -> list[str]:
			return [cls.BASIC]
	capabilities = ArrayField(
		models.CharField(choices=Capability.choices, max_length=30, blank=True),
		null=True,
		default=Capability.default
	)
The resulting migration contained the following:
 # ...
	 migrations.AddField(
		 model_name='profile',
		 name='capabilities',
		 field=django.contrib.postgres.fields.ArrayField(base_field=models.CharField(blank=True, choices=[('BASIC', 'Basic'), ('PROFESSIONAL', 'Professional')], max_length=30), default=appname.models.Capability.default, null=True, size=None),
	 ),
 # ...
As you can see, migrations.AddField is passed as argument "default" a wrong value "appname.models.Capability.default", which leads to an error when trying to migrate. The right value should be "appname.models.Profile.Capability.default".

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 django/db/models/fields/mixins.py | 31 | 60| 178 | 178 | 348 | 
| 2 | 2 django/db/models/fields/__init__.py | 391 | 422| 239 | 417 | 19718 | 
| 3 | 2 django/db/models/fields/__init__.py | 1 | 110| 669 | 1086 | 19718 | 
| 4 | 3 django/db/migrations/state.py | 240 | 263| 247 | 1333 | 27886 | 
| 5 | 4 django/db/backends/mysql/schema.py | 104 | 118| 144 | 1477 | 30222 | 
| 6 | 5 django/db/backends/base/schema.py | 425 | 447| 176 | 1653 | 45369 | 
| 7 | 6 django/contrib/postgres/fields/array.py | 18 | 57| 313 | 1966 | 47861 | 
| 8 | 6 django/db/migrations/state.py | 265 | 289| 238 | 2204 | 47861 | 
| 9 | 6 django/db/backends/base/schema.py | 379 | 407| 205 | 2409 | 47861 | 
| 10 | 6 django/db/models/fields/__init__.py | 1036 | 1068| 256 | 2665 | 47861 | 
| 11 | 7 django/db/models/base.py | 459 | 572| 957 | 3622 | 66979 | 
| 12 | 7 django/db/models/fields/__init__.py | 1290 | 1305| 173 | 3795 | 66979 | 
| 13 | 8 django/db/migrations/questioner.py | 269 | 288| 195 | 3990 | 69675 | 
| 14 | 9 django/db/models/sql/compiler.py | 950 | 998| 483 | 4473 | 86462 | 
| 15 | 9 django/db/models/base.py | 1605 | 1635| 247 | 4720 | 86462 | 
| 16 | 10 django/db/models/fields/related.py | 1124 | 1160| 284 | 5004 | 101166 | 
| 17 | 10 django/db/models/base.py | 2078 | 2131| 359 | 5363 | 101166 | 
| 18 | 10 django/db/migrations/questioner.py | 291 | 342| 367 | 5730 | 101166 | 
| 19 | 11 django/db/models/fields/reverse_related.py | 173 | 191| 160 | 5890 | 103801 | 
| 20 | 12 django/contrib/postgres/fields/ranges.py | 54 | 112| 425 | 6315 | 106280 | 
| 21 | 12 django/db/models/fields/__init__.py | 2392 | 2426| 259 | 6574 | 106280 | 
| 22 | 12 django/contrib/postgres/fields/array.py | 59 | 100| 288 | 6862 | 106280 | 
| 23 | 12 django/db/migrations/questioner.py | 189 | 215| 238 | 7100 | 106280 | 
| 24 | 12 django/db/migrations/questioner.py | 166 | 187| 188 | 7288 | 106280 | 
| 25 | 13 django/db/models/options.py | 288 | 330| 355 | 7643 | 113923 | 
| 26 | 13 django/db/models/fields/related.py | 343 | 379| 294 | 7937 | 113923 | 
| 27 | 13 django/db/migrations/questioner.py | 57 | 87| 255 | 8192 | 113923 | 
| 28 | 13 django/db/models/fields/related.py | 1972 | 2006| 266 | 8458 | 113923 | 
| 29 | 13 django/db/backends/base/schema.py | 449 | 463| 129 | 8587 | 113923 | 
| 30 | 14 django/forms/models.py | 1535 | 1569| 254 | 8841 | 126252 | 
| 31 | 14 django/db/models/fields/__init__.py | 2708 | 2760| 343 | 9184 | 126252 | 
| 32 | 14 django/contrib/postgres/fields/array.py | 102 | 172| 506 | 9690 | 126252 | 
| 33 | 15 django/forms/fields.py | 1319 | 1356| 199 | 9889 | 136086 | 
| 34 | 15 django/db/models/fields/__init__.py | 319 | 389| 462 | 10351 | 136086 | 
| 35 | 15 django/db/backends/base/schema.py | 409 | 423| 160 | 10511 | 136086 | 
| 36 | 15 django/forms/models.py | 268 | 338| 558 | 11069 | 136086 | 
| 37 | 15 django/db/models/fields/related.py | 1463 | 1592| 984 | 12053 | 136086 | 
| 38 | 15 django/db/models/fields/related.py | 304 | 341| 296 | 12349 | 136086 | 
| 39 | 15 django/db/models/fields/__init__.py | 2465 | 2498| 256 | 12605 | 136086 | 
| 40 | 16 django/contrib/admin/options.py | 1 | 119| 799 | 13404 | 155491 | 
| 41 | 16 django/db/models/fields/__init__.py | 113 | 178| 488 | 13892 | 155491 | 
| 42 | 17 django/contrib/contenttypes/fields.py | 474 | 501| 258 | 14150 | 161319 | 
| 43 | 17 django/db/models/fields/related.py | 1162 | 1177| 136 | 14286 | 161319 | 
| 44 | 17 django/db/models/fields/__init__.py | 2429 | 2463| 229 | 14515 | 161319 | 
| 45 | 17 django/contrib/contenttypes/fields.py | 226 | 268| 384 | 14899 | 161319 | 
| 46 | 17 django/db/models/base.py | 1107 | 1159| 520 | 15419 | 161319 | 
| 47 | 17 django/db/models/fields/__init__.py | 1194 | 1214| 159 | 15578 | 161319 | 
| 48 | 17 django/db/models/options.py | 174 | 243| 645 | 16223 | 161319 | 
| 49 | 17 django/contrib/postgres/fields/array.py | 1 | 15| 114 | 16337 | 161319 | 
| 50 | 18 django/db/migrations/operations/fields.py | 227 | 237| 146 | 16483 | 163835 | 
| 51 | 18 django/db/models/fields/__init__.py | 1268 | 1288| 146 | 16629 | 163835 | 
| 52 | 18 django/forms/fields.py | 85 | 182| 804 | 17433 | 163835 | 
| 53 | 19 django/db/migrations/autodetector.py | 1098 | 1216| 994 | 18427 | 177610 | 
| 54 | 19 django/db/models/base.py | 1773 | 1841| 603 | 19030 | 177610 | 
| 55 | 20 django/contrib/postgres/forms/array.py | 183 | 210| 226 | 19256 | 179236 | 
| 56 | 20 django/db/models/fields/__init__.py | 1900 | 1923| 187 | 19443 | 179236 | 
| 57 | 20 django/db/models/base.py | 1190 | 1209| 232 | 19675 | 179236 | 
| 58 | 21 django/contrib/auth/migrations/0001_initial.py | 1 | 205| 1007 | 20682 | 180243 | 
| 59 | 22 django/contrib/postgres/fields/citext.py | 63 | 79| 138 | 20820 | 180829 | 
| 60 | 22 django/forms/fields.py | 1 | 82| 425 | 21245 | 180829 | 
| 61 | 22 django/db/models/options.py | 495 | 518| 155 | 21400 | 180829 | 
| 62 | 22 django/db/migrations/questioner.py | 126 | 164| 327 | 21727 | 180829 | 
| 63 | 22 django/db/migrations/state.py | 291 | 345| 476 | 22203 | 180829 | 
| 64 | 22 django/db/migrations/autodetector.py | 1023 | 1073| 396 | 22599 | 180829 | 
| 65 | 22 django/db/models/fields/__init__.py | 560 | 639| 685 | 23284 | 180829 | 
| 66 | 22 django/db/models/base.py | 574 | 612| 326 | 23610 | 180829 | 
| 67 | 22 django/contrib/postgres/fields/citext.py | 27 | 42| 142 | 23752 | 180829 | 
| 68 | 22 django/db/models/fields/related.py | 91 | 126| 232 | 23984 | 180829 | 
| 69 | 22 django/db/migrations/operations/fields.py | 1 | 43| 238 | 24222 | 180829 | 
| 70 | 23 django/db/models/fields/json.py | 24 | 52| 195 | 24417 | 185614 | 
| 71 | 24 django/contrib/gis/db/models/fields.py | 191 | 224| 252 | 24669 | 188743 | 
| 72 | 24 django/db/migrations/state.py | 437 | 458| 204 | 24873 | 188743 | 
| 73 | 25 django/contrib/flatpages/migrations/0001_initial.py | 1 | 69| 355 | 25228 | 189098 | 
| 74 | 25 django/db/models/fields/__init__.py | 510 | 535| 198 | 25426 | 189098 | 
| 75 | 25 django/db/models/fields/__init__.py | 1246 | 1266| 143 | 25569 | 189098 | 
| 76 | 25 django/db/backends/base/schema.py | 1065 | 1154| 826 | 26395 | 189098 | 
| 77 | 25 django/db/models/fields/__init__.py | 1542 | 1578| 276 | 26671 | 189098 | 
| 78 | 25 django/contrib/postgres/fields/array.py | 195 | 212| 146 | 26817 | 189098 | 
| 79 | 25 django/db/models/fields/related.py | 889 | 916| 237 | 27054 | 189098 | 
| 80 | 25 django/forms/fields.py | 872 | 935| 435 | 27489 | 189098 | 
| 81 | 25 django/db/models/fields/json.py | 133 | 166| 204 | 27693 | 189098 | 
| 82 | 25 django/db/models/fields/__init__.py | 2763 | 2813| 313 | 28006 | 189098 | 
| 83 | 25 django/db/models/options.py | 332 | 368| 338 | 28344 | 189098 | 
| 84 | 26 django/contrib/postgres/fields/hstore.py | 1 | 71| 440 | 28784 | 189803 | 
| 85 | 26 django/db/migrations/operations/fields.py | 239 | 267| 210 | 28994 | 189803 | 
| 86 | 27 django/db/models/fields/files.py | 224 | 356| 993 | 29987 | 193706 | 
| 87 | 27 django/db/models/fields/related.py | 1267 | 1321| 421 | 30408 | 193706 | 
| 88 | 27 django/db/models/fields/__init__.py | 669 | 687| 182 | 30590 | 193706 | 
| 89 | 27 django/db/models/fields/__init__.py | 1216 | 1244| 168 | 30758 | 193706 | 
| 90 | 27 django/db/models/fields/json.py | 54 | 76| 154 | 30912 | 193706 | 
| 91 | 27 django/db/models/base.py | 250 | 367| 874 | 31786 | 193706 | 
| 92 | 27 django/db/models/base.py | 1161 | 1188| 238 | 32024 | 193706 | 
| 93 | 27 django/forms/models.py | 1 | 43| 247 | 32271 | 193706 | 
| 94 | 27 django/db/models/fields/related.py | 424 | 462| 293 | 32564 | 193706 | 
| 95 | 27 django/db/models/fields/related.py | 1894 | 1941| 505 | 33069 | 193706 | 
| 96 | 27 django/db/models/fields/__init__.py | 711 | 732| 228 | 33297 | 193706 | 
| 97 | 27 django/db/migrations/operations/fields.py | 101 | 113| 130 | 33427 | 193706 | 
| 98 | 27 django/contrib/gis/db/models/fields.py | 280 | 291| 148 | 33575 | 193706 | 
| 99 | 27 django/db/backends/mysql/schema.py | 57 | 102| 381 | 33956 | 193706 | 
| 100 | **28 django/db/migrations/serializer.py** | 228 | 263| 281 | 34237 | 196509 | 
| 101 | 28 django/db/models/fields/__init__.py | 458 | 479| 151 | 34388 | 196509 | 
| 102 | 28 django/db/models/fields/__init__.py | 2199 | 2289| 587 | 34975 | 196509 | 
| 103 | 28 django/db/models/fields/related.py | 210 | 226| 142 | 35117 | 196509 | 
| 104 | 28 django/db/models/fields/__init__.py | 2619 | 2641| 167 | 35284 | 196509 | 
| 105 | 28 django/db/models/fields/related.py | 189 | 208| 155 | 35439 | 196509 | 
| 106 | 28 django/db/models/fields/__init__.py | 2292 | 2316| 208 | 35647 | 196509 | 
| 107 | 28 django/contrib/postgres/fields/array.py | 214 | 235| 147 | 35794 | 196509 | 
| 108 | 28 django/db/migrations/state.py | 711 | 765| 472 | 36266 | 196509 | 
| 109 | 28 django/db/models/fields/__init__.py | 2002 | 2041| 230 | 36496 | 196509 | 
| 110 | 28 django/contrib/contenttypes/fields.py | 177 | 224| 417 | 36913 | 196509 | 
| 111 | 28 django/db/models/options.py | 1 | 57| 350 | 37263 | 196509 | 
| 112 | 28 django/db/backends/mysql/schema.py | 208 | 225| 172 | 37435 | 196509 | 


### Hint

```
Thanks for the report. It seems that FunctionTypeSerializer should use __qualname__ instead of __name__: django/db/migrations/serializer.py diff --git a/django/db/migrations/serializer.py b/django/db/migrations/serializer.py index d88cda6e20..06657ebaab 100644 a b class FunctionTypeSerializer(BaseSerializer): 168168 ): 169169 klass = self.value.__self__ 170170 module = klass.__module__ 171 return "%s.%s.%s" % (module, klass.__name__, self.value.__name__), { 171 return "%s.%s.%s" % (module, klass.__qualname__, self.value.__name__), { 172172 "import %s" % module 173173 } 174174 # Further error checking Would you like to prepare a patch? (regression test is required)
Also to nitpick the terminology: Capability is a nested class, not a subclass. (fyi for anyone preparing tests/commit message)
Replying to David Sanders: Also to nitpick the terminology: Capability is a nested class, not a subclass. (fyi for anyone preparing tests/commit message) You're right, that was inaccurate. Thanks for having fixed the title
Replying to Mariusz Felisiak: Thanks for the report. It seems that FunctionTypeSerializer should use __qualname__ instead of __name__: django/db/migrations/serializer.py diff --git a/django/db/migrations/serializer.py b/django/db/migrations/serializer.py index d88cda6e20..06657ebaab 100644 a b class FunctionTypeSerializer(BaseSerializer): 168168 ): 169169 klass = self.value.__self__ 170170 module = klass.__module__ 171 return "%s.%s.%s" % (module, klass.__name__, self.value.__name__), { 171 return "%s.%s.%s" % (module, klass.__qualname__, self.value.__name__), { 172172 "import %s" % module 173173 } 174174 # Further error checking Would you like to prepare a patch? (regression test is required) I would be very happy to prepare a patch, i will do my best to write a test that's coherent with the current suite
I would be very happy to prepare a patch, i will do my best to write a test that's coherent with the current suite You can check tests in tests.migrations.test_writer.WriterTests, e.g. test_serialize_nested_class().
```

## Patch

```diff
diff --git a/django/db/migrations/serializer.py b/django/db/migrations/serializer.py
--- a/django/db/migrations/serializer.py
+++ b/django/db/migrations/serializer.py
@@ -168,7 +168,7 @@ def serialize(self):
         ):
             klass = self.value.__self__
             module = klass.__module__
-            return "%s.%s.%s" % (module, klass.__name__, self.value.__name__), {
+            return "%s.%s.%s" % (module, klass.__qualname__, self.value.__name__), {
                 "import %s" % module
             }
         # Further error checking

```

## Test Patch

```diff
diff --git a/tests/migrations/test_writer.py b/tests/migrations/test_writer.py
--- a/tests/migrations/test_writer.py
+++ b/tests/migrations/test_writer.py
@@ -211,6 +211,10 @@ class NestedChoices(models.TextChoices):
         X = "X", "X value"
         Y = "Y", "Y value"
 
+        @classmethod
+        def method(cls):
+            return cls.X
+
     def safe_exec(self, string, value=None):
         d = {}
         try:
@@ -468,6 +472,15 @@ def test_serialize_nested_class(self):
                     ),
                 )
 
+    def test_serialize_nested_class_method(self):
+        self.assertSerializedResultEqual(
+            self.NestedChoices.method,
+            (
+                "migrations.test_writer.WriterTests.NestedChoices.method",
+                {"import migrations.test_writer"},
+            ),
+        )
+
     def test_serialize_uuid(self):
         self.assertSerializedEqual(uuid.uuid1())
         self.assertSerializedEqual(uuid.uuid4())

```


## Code snippets

### 1 - django/db/models/fields/mixins.py:

Start line: 31, End line: 60

```python
class CheckFieldDefaultMixin:
    _default_hint = ("<valid default>", "<invalid default>")

    def _check_default(self):
        if (
            self.has_default()
            and self.default is not None
            and not callable(self.default)
        ):
            return [
                checks.Warning(
                    "%s default should be a callable instead of an instance "
                    "so that it's not shared between all field instances."
                    % (self.__class__.__name__,),
                    hint=(
                        "Use a callable instead, e.g., use `%s` instead of "
                        "`%s`." % self._default_hint
                    ),
                    obj=self,
                    id="fields.E010",
                )
            ]
        else:
            return []

    def check(self, **kwargs):
        errors = super().check(**kwargs)
        errors.extend(self._check_default())
        return errors
```
### 2 - django/db/models/fields/__init__.py:

Start line: 391, End line: 422

```python
@total_ordering
class Field(RegisterLookupMixin):

    def _check_db_default(self, databases=None, **kwargs):
        from django.db.models.expressions import Value

        if (
            self.db_default is NOT_PROVIDED
            or isinstance(self.db_default, Value)
            or databases is None
        ):
            return []
        errors = []
        for db in databases:
            if not router.allow_migrate_model(db, self.model):
                continue
            connection = connections[db]

            if not getattr(self.db_default, "allowed_default", False) and (
                connection.features.supports_expression_defaults
            ):
                msg = f"{self.db_default} cannot be used in db_default."
                errors.append(checks.Error(msg, obj=self, id="fields.E012"))

            if not (
                connection.features.supports_expression_defaults
                or "supports_expression_defaults"
                in self.model._meta.required_db_features
            ):
                msg = (
                    f"{connection.display_name} does not support default database "
                    "values with expressions (db_default)."
                )
                errors.append(checks.Error(msg, obj=self, id="fields.E011"))
        return errors
```
### 3 - django/db/models/fields/__init__.py:

Start line: 1, End line: 110

```python
import collections.abc
import copy
import datetime
import decimal
import operator
import uuid
import warnings
from base64 import b64decode, b64encode
from functools import partialmethod, total_ordering

from django import forms
from django.apps import apps
from django.conf import settings
from django.core import checks, exceptions, validators
from django.db import connection, connections, router
from django.db.models.constants import LOOKUP_SEP
from django.db.models.enums import ChoicesMeta
from django.db.models.query_utils import DeferredAttribute, RegisterLookupMixin
from django.utils import timezone
from django.utils.datastructures import DictWrapper
from django.utils.dateparse import (
    parse_date,
    parse_datetime,
    parse_duration,
    parse_time,
)
from django.utils.duration import duration_microseconds, duration_string
from django.utils.functional import Promise, cached_property
from django.utils.ipv6 import clean_ipv6_address
from django.utils.itercompat import is_iterable
from django.utils.text import capfirst
from django.utils.translation import gettext_lazy as _

__all__ = [
    "AutoField",
    "BLANK_CHOICE_DASH",
    "BigAutoField",
    "BigIntegerField",
    "BinaryField",
    "BooleanField",
    "CharField",
    "CommaSeparatedIntegerField",
    "DateField",
    "DateTimeField",
    "DecimalField",
    "DurationField",
    "EmailField",
    "Empty",
    "Field",
    "FilePathField",
    "FloatField",
    "GenericIPAddressField",
    "IPAddressField",
    "IntegerField",
    "NOT_PROVIDED",
    "NullBooleanField",
    "PositiveBigIntegerField",
    "PositiveIntegerField",
    "PositiveSmallIntegerField",
    "SlugField",
    "SmallAutoField",
    "SmallIntegerField",
    "TextField",
    "TimeField",
    "URLField",
    "UUIDField",
]


class Empty:
    pass


class NOT_PROVIDED:
    pass


# The values to use for "blank" in SelectFields. Will be appended to the start
# of most "choices" lists.
BLANK_CHOICE_DASH = [("", "---------")]


def _load_field(app_label, model_name, field_name):
    return apps.get_model(app_label, model_name)._meta.get_field(field_name)


# A guide to Field parameters:
#
#   * name:      The name of the field specified in the model.
#   * attname:   The attribute to use on the model object. This is the same as
#                "name", except in the case of ForeignKeys, where "_id" is
#                appended.
#   * db_column: The db_column specified in the model (or None).
#   * column:    The database column for this field. This is the same as
#                "attname", except if db_column is specified.
#
# Code that introspects values, or does other dynamic things, should use
# attname. For example, this gets the primary key value of object "obj":
#
#     getattr(obj, opts.pk.attname)


def _empty(of_cls):
    new = Empty()
    new.__class__ = of_cls
    return new


def return_None():
    return None
```
### 4 - django/db/migrations/state.py:

Start line: 240, End line: 263

```python
class ProjectState:

    def add_field(self, app_label, model_name, name, field, preserve_default):
        # If preserve default is off, don't use the default for future state.
        if not preserve_default:
            field = field.clone()
            field.default = NOT_PROVIDED
        else:
            field = field
        model_key = app_label, model_name
        self.models[model_key].fields[name] = field
        if self._relations is not None:
            self.resolve_model_field_relations(model_key, name, field)
        # Delay rendering of relationships if it's not a relational field.
        delay = not field.is_relation
        self.reload_model(*model_key, delay=delay)

    def remove_field(self, app_label, model_name, name):
        model_key = app_label, model_name
        model_state = self.models[model_key]
        old_field = model_state.fields.pop(name)
        if self._relations is not None:
            self.resolve_model_field_relations(model_key, name, old_field)
        # Delay rendering of relationships if it's not a relational field.
        delay = not old_field.is_relation
        self.reload_model(*model_key, delay=delay)
```
### 5 - django/db/backends/mysql/schema.py:

Start line: 104, End line: 118

```python
class DatabaseSchemaEditor(BaseDatabaseSchemaEditor):

    def add_field(self, model, field):
        super().add_field(model, field)

        # Simulate the effect of a one-off default.
        # field.default may be unhashable, so a set isn't used for "in" check.
        if self.skip_default(field) and field.default not in (None, NOT_PROVIDED):
            effective_default = self.effective_default(field)
            self.execute(
                "UPDATE %(table)s SET %(column)s = %%s"
                % {
                    "table": self.quote_name(model._meta.db_table),
                    "column": self.quote_name(field.column),
                },
                [effective_default],
            )
```
### 6 - django/db/backends/base/schema.py:

Start line: 425, End line: 447

```python
class BaseDatabaseSchemaEditor:

    @staticmethod
    def _effective_default(field):
        # This method allows testing its logic without a connection.
        if field.has_default():
            default = field.get_default()
        elif not field.null and field.blank and field.empty_strings_allowed:
            if field.get_internal_type() == "BinaryField":
                default = b""
            else:
                default = ""
        elif getattr(field, "auto_now", False) or getattr(field, "auto_now_add", False):
            internal_type = field.get_internal_type()
            if internal_type == "DateTimeField":
                default = timezone.now()
            else:
                default = datetime.now()
                if internal_type == "DateField":
                    default = default.date()
                elif internal_type == "TimeField":
                    default = default.time()
        else:
            default = None
        return default
```
### 7 - django/contrib/postgres/fields/array.py:

Start line: 18, End line: 57

```python
class ArrayField(CheckFieldDefaultMixin, Field):
    empty_strings_allowed = False
    default_error_messages = {
        "item_invalid": _("Item %(nth)s in the array did not validate:"),
        "nested_array_mismatch": _("Nested arrays must have the same length."),
    }
    _default_hint = ("list", "[]")

    def __init__(self, base_field, size=None, **kwargs):
        self.base_field = base_field
        self.db_collation = getattr(self.base_field, "db_collation", None)
        self.size = size
        if self.size:
            self.default_validators = [
                *self.default_validators,
                ArrayMaxLengthValidator(self.size),
            ]
        # For performance, only add a from_db_value() method if the base field
        # implements it.
        if hasattr(self.base_field, "from_db_value"):
            self.from_db_value = self._from_db_value
        super().__init__(**kwargs)

    @property
    def model(self):
        try:
            return self.__dict__["model"]
        except KeyError:
            raise AttributeError(
                "'%s' object has no attribute 'model'" % self.__class__.__name__
            )

    @model.setter
    def model(self, model):
        self.__dict__["model"] = model
        self.base_field.model = model

    @classmethod
    def _choices_is_value(cls, value):
        return isinstance(value, (list, tuple)) or super()._choices_is_value(value)
```
### 8 - django/db/migrations/state.py:

Start line: 265, End line: 289

```python
class ProjectState:

    def alter_field(self, app_label, model_name, name, field, preserve_default):
        if not preserve_default:
            field = field.clone()
            field.default = NOT_PROVIDED
        else:
            field = field
        model_key = app_label, model_name
        fields = self.models[model_key].fields
        if self._relations is not None:
            old_field = fields.pop(name)
            if old_field.is_relation:
                self.resolve_model_field_relations(model_key, name, old_field)
            fields[name] = field
            if field.is_relation:
                self.resolve_model_field_relations(model_key, name, field)
        else:
            fields[name] = field
        # TODO: investigate if old relational fields must be reloaded or if
        # it's sufficient if the new field is (#27737).
        # Delay rendering of relationships if it's not a relational field and
        # not referenced by a foreign key.
        delay = not field.is_relation and not field_is_referenced(
            self, model_key, (name, field)
        )
        self.reload_model(*model_key, delay=delay)
```
### 9 - django/db/backends/base/schema.py:

Start line: 379, End line: 407

```python
class BaseDatabaseSchemaEditor:

    def skip_default(self, field):
        """
        Some backends don't accept default values for certain columns types
        (i.e. MySQL longtext and longblob).
        """
        return False

    def skip_default_on_alter(self, field):
        """
        Some backends don't accept default values for certain columns types
        (i.e. MySQL longtext and longblob) in the ALTER COLUMN statement.
        """
        return False

    def prepare_default(self, value):
        """
        Only used for backends which have requires_literal_defaults feature
        """
        raise NotImplementedError(
            "subclasses of BaseDatabaseSchemaEditor for backends which have "
            "requires_literal_defaults must provide a prepare_default() method"
        )

    def _column_default_sql(self, field):
        """
        Return the SQL to use in a DEFAULT clause. The resulting string should
        contain a '%s' placeholder for a default value.
        """
        return "%s"
```
### 10 - django/db/models/fields/__init__.py:

Start line: 1036, End line: 1068

```python
@total_ordering
class Field(RegisterLookupMixin):

    def get_choices(
        self,
        include_blank=True,
        blank_choice=BLANK_CHOICE_DASH,
        limit_choices_to=None,
        ordering=(),
    ):
        """
        Return choices with a default blank choices included, for use
        as <select> choices for this field.
        """
        if self.choices is not None:
            choices = list(self.choices)
            if include_blank:
                blank_defined = any(
                    choice in ("", None) for choice, _ in self.flatchoices
                )
                if not blank_defined:
                    choices = blank_choice + choices
            return choices
        rel_model = self.remote_field.model
        limit_choices_to = limit_choices_to or self.get_limit_choices_to()
        choice_func = operator.attrgetter(
            self.remote_field.get_related_field().attname
            if hasattr(self.remote_field, "get_related_field")
            else "pk"
        )
        qs = rel_model._default_manager.complex_filter(limit_choices_to)
        if ordering:
            qs = qs.order_by(*ordering)
        return (blank_choice if include_blank else []) + [
            (choice_func(x), str(x)) for x in qs
        ]
```
### 100 - django/db/migrations/serializer.py:

Start line: 228, End line: 263

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
