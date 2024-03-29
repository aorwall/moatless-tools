# django__django-12209

| **django/django** | `5a68f024987e6d16c2626a31bf653a2edddea579` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 3381 |
| **Any found context length** | 3381 |
| **Avg pos** | 6.0 |
| **Min pos** | 6 |
| **Max pos** | 6 |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/db/models/base.py b/django/db/models/base.py
--- a/django/db/models/base.py
+++ b/django/db/models/base.py
@@ -849,6 +849,7 @@ def _save_table(self, raw=False, cls=None, force_insert=False,
         updated = False
         # Skip an UPDATE when adding an instance and primary key has a default.
         if (
+            not raw and
             not force_insert and
             self._state.adding and
             self._meta.pk.default and

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/db/models/base.py | 852 | 852 | 6 | 1 | 3381


## Problem Statement

```
Change in behaviour when saving a model instance with an explcit pk value if the pk field has a default
Description
	 
		(last modified by Reupen Shah)
	 
Consider the following model:
from uuid import uuid4
from django.db import models
class Sample(models.Model):
	id = models.UUIDField(primary_key=True, default=uuid4)
	name = models.CharField(blank=True, max_length=100)
In Django 2.2 and earlier, the following commands would result in an INSERT followed by an UPDATE:
s0 = Sample.objects.create()
s1 = Sample(pk=s0.pk, name='Test 1')
s1.save()
However, in Django 3.0, this results in two INSERTs (naturally the second one fails). The behaviour also changes if default=uuid4 is removed from the id field.
This seems related to https://code.djangoproject.com/ticket/29260.
The change in behaviour also has the side effect of changing the behaviour of the loaddata management command when the fixture contains explicit pk values and the objects already exist (e.g. when loading the fixture multiple times).
Perhaps the intention was to only change the behaviour if an explicit pk value was not set on the model instance being saved? (At least, that would be more backwards-compatible behaviour...)

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 django/db/models/base.py** | 748 | 797| 456 | 456 | 15311 | 
| 2 | 2 django/db/models/fields/__init__.py | 565 | 606| 287 | 743 | 32823 | 
| 3 | **2 django/db/models/base.py** | 663 | 747| 831 | 1574 | 32823 | 
| 4 | 3 django/forms/models.py | 775 | 813| 314 | 1888 | 44425 | 
| 5 | **3 django/db/models/base.py** | 404 | 503| 856 | 2744 | 44425 | 
| **-> 6 <-** | **3 django/db/models/base.py** | 829 | 890| 637 | 3381 | 44425 | 
| 7 | 3 django/db/models/fields/__init__.py | 2282 | 2332| 339 | 3720 | 44425 | 
| 8 | **3 django/db/models/base.py** | 1861 | 1912| 351 | 4071 | 44425 | 
| 9 | **3 django/db/models/base.py** | 1068 | 1111| 404 | 4475 | 44425 | 
| 10 | 4 django/db/backends/base/schema.py | 370 | 384| 182 | 4657 | 55740 | 
| 11 | 5 django/db/models/fields/related_descriptors.py | 309 | 323| 182 | 4839 | 66098 | 
| 12 | 5 django/db/backends/base/schema.py | 626 | 698| 792 | 5631 | 66098 | 
| 13 | 6 django/db/backends/sqlite3/schema.py | 101 | 138| 486 | 6117 | 70063 | 
| 14 | 7 django/db/models/options.py | 296 | 314| 136 | 6253 | 77162 | 
| 15 | **7 django/db/models/base.py** | 979 | 1007| 230 | 6483 | 77162 | 
| 16 | 7 django/forms/models.py | 918 | 950| 353 | 6836 | 77162 | 
| 17 | **7 django/db/models/base.py** | 799 | 827| 307 | 7143 | 77162 | 
| 18 | 8 django/db/migrations/questioner.py | 162 | 185| 246 | 7389 | 79236 | 
| 19 | 9 django/db/migrations/autodetector.py | 796 | 845| 570 | 7959 | 90971 | 
| 20 | 9 django/forms/models.py | 647 | 677| 217 | 8176 | 90971 | 
| 21 | 10 django/core/serializers/base.py | 219 | 230| 157 | 8333 | 93396 | 
| 22 | 10 django/db/backends/base/schema.py | 699 | 768| 740 | 9073 | 93396 | 
| 23 | 11 django/db/migrations/operations/models.py | 625 | 677| 342 | 9415 | 100092 | 
| 24 | 11 django/db/models/options.py | 262 | 294| 343 | 9758 | 100092 | 
| 25 | 12 django/db/backends/mysql/schema.py | 79 | 89| 138 | 9896 | 101487 | 
| 26 | 12 django/db/migrations/operations/models.py | 304 | 343| 406 | 10302 | 101487 | 
| 27 | 12 django/db/migrations/operations/models.py | 345 | 394| 493 | 10795 | 101487 | 
| 28 | 13 django/db/models/signals.py | 37 | 54| 231 | 11026 | 101974 | 
| 29 | **13 django/db/models/base.py** | 948 | 962| 212 | 11238 | 101974 | 
| 30 | 13 django/db/migrations/autodetector.py | 1182 | 1207| 245 | 11483 | 101974 | 
| 31 | **13 django/db/models/base.py** | 919 | 946| 256 | 11739 | 101974 | 
| 32 | 13 django/db/backends/base/schema.py | 278 | 299| 173 | 11912 | 101974 | 
| 33 | 13 django/db/backends/base/schema.py | 1168 | 1197| 268 | 12180 | 101974 | 
| 34 | 14 django/db/models/lookups.py | 595 | 628| 141 | 12321 | 106797 | 
| 35 | 14 django/db/backends/base/schema.py | 769 | 809| 511 | 12832 | 106797 | 
| 36 | 15 django/db/models/deletion.py | 361 | 428| 569 | 13401 | 110456 | 
| 37 | 15 django/db/migrations/operations/models.py | 591 | 607| 215 | 13616 | 110456 | 
| 38 | 15 django/db/backends/sqlite3/schema.py | 223 | 305| 731 | 14347 | 110456 | 
| 39 | 15 django/db/migrations/operations/models.py | 460 | 485| 279 | 14626 | 110456 | 
| 40 | 15 django/db/migrations/operations/models.py | 120 | 239| 827 | 15453 | 110456 | 
| 41 | 15 django/core/serializers/base.py | 232 | 249| 208 | 15661 | 110456 | 
| 42 | 15 django/forms/models.py | 444 | 471| 227 | 15888 | 110456 | 
| 43 | 15 django/db/migrations/autodetector.py | 525 | 671| 1109 | 16997 | 110456 | 
| 44 | 15 django/db/migrations/operations/models.py | 566 | 589| 183 | 17180 | 110456 | 
| 45 | **15 django/db/models/base.py** | 1113 | 1140| 286 | 17466 | 110456 | 
| 46 | **15 django/db/models/base.py** | 549 | 565| 142 | 17608 | 110456 | 
| 47 | **15 django/db/models/base.py** | 505 | 547| 347 | 17955 | 110456 | 
| 48 | 15 django/db/migrations/questioner.py | 56 | 81| 220 | 18175 | 110456 | 
| 49 | 16 django/contrib/admin/helpers.py | 355 | 364| 134 | 18309 | 113649 | 
| 50 | 16 django/db/migrations/questioner.py | 143 | 160| 183 | 18492 | 113649 | 
| 51 | 17 django/db/models/fields/related.py | 1202 | 1313| 939 | 19431 | 127161 | 
| 52 | 17 django/db/migrations/operations/models.py | 609 | 622| 137 | 19568 | 127161 | 
| 53 | **17 django/db/models/base.py** | 1250 | 1279| 242 | 19810 | 127161 | 
| 54 | 17 django/db/migrations/questioner.py | 207 | 224| 171 | 19981 | 127161 | 
| 55 | 17 django/db/backends/sqlite3/schema.py | 140 | 221| 820 | 20801 | 127161 | 
| 56 | 17 django/db/migrations/autodetector.py | 1146 | 1180| 296 | 21097 | 127161 | 
| 57 | 17 django/db/backends/base/schema.py | 567 | 625| 676 | 21773 | 127161 | 
| 58 | 17 django/db/models/fields/__init__.py | 1186 | 1204| 180 | 21953 | 127161 | 
| 59 | 18 django/forms/fields.py | 1173 | 1203| 182 | 22135 | 136108 | 
| 60 | 18 django/db/migrations/questioner.py | 227 | 240| 123 | 22258 | 136108 | 
| 61 | 18 django/forms/models.py | 679 | 750| 732 | 22990 | 136108 | 
| 62 | **18 django/db/models/base.py** | 1159 | 1187| 213 | 23203 | 136108 | 
| 63 | 18 django/db/migrations/autodetector.py | 1027 | 1043| 188 | 23391 | 136108 | 
| 64 | 18 django/forms/models.py | 382 | 410| 240 | 23631 | 136108 | 
| 65 | 19 django/db/models/fields/files.py | 212 | 324| 868 | 24499 | 139830 | 
| 66 | 20 django/bin/django-admin.py | 1 | 22| 138 | 24637 | 139968 | 
| 67 | 20 django/db/backends/sqlite3/schema.py | 86 | 99| 181 | 24818 | 139968 | 
| 68 | 20 django/core/serializers/base.py | 301 | 323| 207 | 25025 | 139968 | 
| 69 | 20 django/db/models/fields/related_descriptors.py | 120 | 154| 405 | 25430 | 139968 | 
| 70 | **20 django/db/models/base.py** | 385 | 401| 128 | 25558 | 139968 | 
| 71 | 20 django/db/models/fields/related_descriptors.py | 672 | 726| 511 | 26069 | 139968 | 
| 72 | 20 django/db/models/fields/__init__.py | 1336 | 1364| 281 | 26350 | 139968 | 
| 73 | 21 django/contrib/sites/models.py | 78 | 121| 236 | 26586 | 140756 | 
| 74 | 22 django/db/migrations/state.py | 154 | 164| 132 | 26718 | 145974 | 
| 75 | 23 django/db/models/query.py | 787 | 826| 322 | 27040 | 162979 | 
| 76 | 24 django/db/backends/postgresql/operations.py | 166 | 208| 454 | 27494 | 165738 | 
| 77 | 24 django/db/backends/sqlite3/schema.py | 348 | 365| 270 | 27764 | 165738 | 
| 78 | 25 django/db/backends/base/operations.py | 251 | 294| 324 | 28088 | 171229 | 
| 79 | 26 django/db/backends/postgresql/schema.py | 154 | 180| 351 | 28439 | 173212 | 
| 80 | 26 django/db/migrations/state.py | 106 | 152| 368 | 28807 | 173212 | 
| 81 | 26 django/db/backends/sqlite3/schema.py | 367 | 413| 444 | 29251 | 173212 | 
| 82 | 26 django/db/models/options.py | 1 | 36| 304 | 29555 | 173212 | 
| 83 | 26 django/db/backends/base/schema.py | 1048 | 1062| 123 | 29678 | 173212 | 
| 84 | 26 django/db/models/fields/related.py | 824 | 845| 169 | 29847 | 173212 | 
| 85 | 27 django/db/backends/oracle/schema.py | 57 | 77| 249 | 30096 | 174968 | 
| 86 | **27 django/db/models/base.py** | 212 | 322| 866 | 30962 | 174968 | 
| 87 | 27 django/db/backends/base/schema.py | 1106 | 1136| 240 | 31202 | 174968 | 
| 88 | 27 django/db/models/fields/related.py | 896 | 916| 178 | 31380 | 174968 | 
| 89 | 27 django/db/migrations/autodetector.py | 1123 | 1144| 231 | 31611 | 174968 | 
| 90 | **27 django/db/models/base.py** | 1470 | 1492| 171 | 31782 | 174968 | 
| 91 | 27 django/db/models/fields/__init__.py | 1052 | 1081| 218 | 32000 | 174968 | 
| 92 | 27 django/db/migrations/autodetector.py | 904 | 985| 876 | 32876 | 174968 | 
| 93 | 27 django/db/backends/postgresql/schema.py | 84 | 152| 539 | 33415 | 174968 | 
| 94 | 27 django/db/migrations/autodetector.py | 707 | 794| 789 | 34204 | 174968 | 
| 95 | 28 django/db/backends/base/features.py | 1 | 115| 904 | 35108 | 177547 | 
| 96 | **28 django/db/models/base.py** | 1 | 50| 330 | 35438 | 177547 | 
| 97 | 28 django/db/backends/base/schema.py | 386 | 400| 185 | 35623 | 177547 | 
| 98 | 28 django/db/models/fields/related_descriptors.py | 344 | 363| 156 | 35779 | 177547 | 
| 99 | 29 django/db/backends/mysql/operations.py | 1 | 35| 282 | 36061 | 180868 | 
| 100 | 30 django/db/models/sql/compiler.py | 1506 | 1546| 409 | 36470 | 194871 | 
| 101 | 30 django/forms/models.py | 1254 | 1284| 242 | 36712 | 194871 | 
| 102 | 30 django/db/models/fields/related.py | 847 | 873| 240 | 36952 | 194871 | 
| 103 | 30 django/db/migrations/autodetector.py | 1209 | 1221| 131 | 37083 | 194871 | 
| 104 | 30 django/db/migrations/autodetector.py | 1086 | 1121| 312 | 37395 | 194871 | 
| 105 | 30 django/db/models/query.py | 576 | 598| 213 | 37608 | 194871 | 
| 106 | 30 django/db/models/query.py | 770 | 786| 157 | 37765 | 194871 | 
| 107 | 30 django/db/models/fields/__init__.py | 1 | 81| 633 | 38398 | 194871 | 
| 108 | 30 django/db/models/fields/related_descriptors.py | 365 | 381| 184 | 38582 | 194871 | 


### Hint

```
It looks like ​the logic in _save_table should not force an insert if an explicit pk value is provided. The logic should likely take pk_set into account if ( not pk_set and self._state.adding and self._meta.pk.default and self._meta.pk.default is not NOT_PROVIDED ): force_insert = True I'm surprised this was not caught by the suite if this breaks fixtures loading.
I bisected the regression down to 85458e94e38c20e57939947ee515a1a53689659f if that helps.
I'm afraid we'll have to revert 85458e94e38c20e57939947ee515a1a53689659f if we want this pattern to work because we assign field defaults on Model.__init__ without tracking whether or not they were generated from field defaults or not which makes both Sample() and Sample(pk=s0.pk) have pk_set=True in _save_table. Note that this limitation was mentioned in https://code.djangoproject.com/ticket/29260#comment:3 so another option could be to document that force_update must be used in this particular case. I feel like this would be good compromise. Regarding the fixture loading we should branch of raw ​which is passed by the serialization framework to disable the optimiation. Happy to provide a patch for whatever solution we choose. I think it's worth adjusting the feature given it does reduce the number of queries significantly when using primary key defaults and documenting it as a backward incompatible change that can be worked around by passing force_update but simply reverting the feature and reopening #29260 to target the next release is likely less trouble.
If it helps, I noticed this through the changed behaviour of loaddata in such cases (rather than the example code given). (I don't think we have any code that directly looks like the example.)
Replying to Simon Charette: Note that this limitation was mentioned in https://code.djangoproject.com/ticket/29260#comment:3 so another option could be to document that force_update must be used in this particular case. I feel like this would be good compromise. Regarding the fixture loading we should branch of raw ​which is passed by the serialization framework to disable the optimiation. Happy to provide a patch for whatever solution we choose. I think it's worth adjusting the feature given it does reduce the number of queries significantly when using primary key defaults and documenting it as a backward incompatible change that can be worked around by passing force_update... I really like this proposition and I think it's fine to adjust the current fix and backport it to the Django 3.0.
```

## Patch

```diff
diff --git a/django/db/models/base.py b/django/db/models/base.py
--- a/django/db/models/base.py
+++ b/django/db/models/base.py
@@ -849,6 +849,7 @@ def _save_table(self, raw=False, cls=None, force_insert=False,
         updated = False
         # Skip an UPDATE when adding an instance and primary key has a default.
         if (
+            not raw and
             not force_insert and
             self._state.adding and
             self._meta.pk.default and

```

## Test Patch

```diff
diff --git a/tests/serializers/models/data.py b/tests/serializers/models/data.py
--- a/tests/serializers/models/data.py
+++ b/tests/serializers/models/data.py
@@ -4,6 +4,8 @@
 NULL values, where allowed.
 The basic idea is to have a model for each Django data type.
 """
+import uuid
+
 from django.contrib.contenttypes.fields import (
     GenericForeignKey, GenericRelation,
 )
@@ -257,6 +259,10 @@ class UUIDData(models.Model):
     data = models.UUIDField(primary_key=True)
 
 
+class UUIDDefaultData(models.Model):
+    data = models.UUIDField(primary_key=True, default=uuid.uuid4)
+
+
 class FKToUUID(models.Model):
     data = models.ForeignKey(UUIDData, models.CASCADE)
 
diff --git a/tests/serializers/test_data.py b/tests/serializers/test_data.py
--- a/tests/serializers/test_data.py
+++ b/tests/serializers/test_data.py
@@ -26,7 +26,7 @@
     ModifyingSaveData, NullBooleanData, O2OData, PositiveBigIntegerData,
     PositiveIntegerData, PositiveIntegerPKData, PositiveSmallIntegerData,
     PositiveSmallIntegerPKData, SlugData, SlugPKData, SmallData, SmallPKData,
-    Tag, TextData, TimeData, UniqueAnchor, UUIDData,
+    Tag, TextData, TimeData, UniqueAnchor, UUIDData, UUIDDefaultData,
 )
 from .tests import register_tests
 
@@ -351,6 +351,7 @@ def inherited_compare(testcase, pk, klass, data):
     # (pk_obj, 790, XMLPKData, "<foo></foo>"),
     (pk_obj, 791, UUIDData, uuid_obj),
     (fk_obj, 792, FKToUUID, uuid_obj),
+    (pk_obj, 793, UUIDDefaultData, uuid_obj),
 
     (data_obj, 800, AutoNowDateTimeData, datetime.datetime(2006, 6, 16, 10, 42, 37)),
     (data_obj, 810, ModifyingSaveData, 42),

```


## Code snippets

### 1 - django/db/models/base.py:

Start line: 748, End line: 797

```python
class Model(metaclass=ModelBase):
    save.alters_data = True

    def save_base(self, raw=False, force_insert=False,
                  force_update=False, using=None, update_fields=None):
        """
        Handle the parts of saving which should be done only once per save,
        yet need to be done in raw saves, too. This includes some sanity
        checks and signal sending.

        The 'raw' argument is telling save_base not to save any parent
        models and not to do any changes to the values before save. This
        is used by fixture loading.
        """
        using = using or router.db_for_write(self.__class__, instance=self)
        assert not (force_insert and (force_update or update_fields))
        assert update_fields is None or update_fields
        cls = origin = self.__class__
        # Skip proxies, but keep the origin as the proxy model.
        if cls._meta.proxy:
            cls = cls._meta.concrete_model
        meta = cls._meta
        if not meta.auto_created:
            pre_save.send(
                sender=origin, instance=self, raw=raw, using=using,
                update_fields=update_fields,
            )
        # A transaction isn't needed if one query is issued.
        if meta.parents:
            context_manager = transaction.atomic(using=using, savepoint=False)
        else:
            context_manager = transaction.mark_for_rollback_on_error(using=using)
        with context_manager:
            parent_inserted = False
            if not raw:
                parent_inserted = self._save_parents(cls, using, update_fields)
            updated = self._save_table(
                raw, cls, force_insert or parent_inserted,
                force_update, using, update_fields,
            )
        # Store the database on which the object was saved
        self._state.db = using
        # Once saved, this is no longer a to-be-added instance.
        self._state.adding = False

        # Signal that the save is complete
        if not meta.auto_created:
            post_save.send(
                sender=origin, instance=self, created=(not updated),
                update_fields=update_fields, raw=raw, using=using,
            )
```
### 2 - django/db/models/fields/__init__.py:

Start line: 565, End line: 606

```python
@total_ordering
class Field(RegisterLookupMixin):

    def get_pk_value_on_save(self, instance):
        """
        Hook to generate new PK values on save. This method is called when
        saving instances with no primary key value set. If this method returns
        something else than None, then the returned value is used when saving
        the new instance.
        """
        if self.default:
            return self.get_default()
        return None

    def to_python(self, value):
        """
        Convert the input value into the expected Python data type, raising
        django.core.exceptions.ValidationError if the data can't be converted.
        Return the converted value. Subclasses should override this.
        """
        return value

    @cached_property
    def validators(self):
        """
        Some validators can't be created at field initialization time.
        This method provides a way to delay their creation until required.
        """
        return [*self.default_validators, *self._validators]

    def run_validators(self, value):
        if value in self.empty_values:
            return

        errors = []
        for v in self.validators:
            try:
                v(value)
            except exceptions.ValidationError as e:
                if hasattr(e, 'code') and e.code in self.error_messages:
                    e.message = self.error_messages[e.code]
                errors.extend(e.error_list)

        if errors:
            raise exceptions.ValidationError(errors)
```
### 3 - django/db/models/base.py:

Start line: 663, End line: 747

```python
class Model(metaclass=ModelBase):

    def save(self, force_insert=False, force_update=False, using=None,
             update_fields=None):
        """
        Save the current instance. Override this in a subclass if you want to
        control the saving process.

        The 'force_insert' and 'force_update' parameters can be used to insist
        that the "save" must be an SQL insert or update (or equivalent for
        non-SQL backends), respectively. Normally, they should not be set.
        """
        # Ensure that a model instance without a PK hasn't been assigned to
        # a ForeignKey or OneToOneField on this model. If the field is
        # nullable, allowing the save() would result in silent data loss.
        for field in self._meta.concrete_fields:
            # If the related field isn't cached, then an instance hasn't
            # been assigned and there's no need to worry about this check.
            if field.is_relation and field.is_cached(self):
                obj = getattr(self, field.name, None)
                if not obj:
                    continue
                # A pk may have been assigned manually to a model instance not
                # saved to the database (or auto-generated in a case like
                # UUIDField), but we allow the save to proceed and rely on the
                # database to raise an IntegrityError if applicable. If
                # constraints aren't supported by the database, there's the
                # unavoidable risk of data corruption.
                if obj.pk is None:
                    # Remove the object from a related instance cache.
                    if not field.remote_field.multiple:
                        field.remote_field.delete_cached_value(obj)
                    raise ValueError(
                        "save() prohibited to prevent data loss due to "
                        "unsaved related object '%s'." % field.name
                    )
                elif getattr(self, field.attname) is None:
                    # Use pk from related object if it has been saved after
                    # an assignment.
                    setattr(self, field.attname, obj.pk)
                # If the relationship's pk/to_field was changed, clear the
                # cached relationship.
                if getattr(obj, field.target_field.attname) != getattr(self, field.attname):
                    field.delete_cached_value(self)

        using = using or router.db_for_write(self.__class__, instance=self)
        if force_insert and (force_update or update_fields):
            raise ValueError("Cannot force both insert and updating in model saving.")

        deferred_fields = self.get_deferred_fields()
        if update_fields is not None:
            # If update_fields is empty, skip the save. We do also check for
            # no-op saves later on for inheritance cases. This bailout is
            # still needed for skipping signal sending.
            if not update_fields:
                return

            update_fields = frozenset(update_fields)
            field_names = set()

            for field in self._meta.fields:
                if not field.primary_key:
                    field_names.add(field.name)

                    if field.name != field.attname:
                        field_names.add(field.attname)

            non_model_fields = update_fields.difference(field_names)

            if non_model_fields:
                raise ValueError("The following fields do not exist in this "
                                 "model or are m2m fields: %s"
                                 % ', '.join(non_model_fields))

        # If saving to the same database, and this model is deferred, then
        # automatically do an "update_fields" save on the loaded fields.
        elif not force_insert and deferred_fields and using == self._state.db:
            field_names = set()
            for field in self._meta.concrete_fields:
                if not field.primary_key and not hasattr(field, 'through'):
                    field_names.add(field.attname)
            loaded_fields = field_names.difference(deferred_fields)
            if loaded_fields:
                update_fields = frozenset(loaded_fields)

        self.save_base(using=using, force_insert=force_insert,
                       force_update=force_update, update_fields=update_fields)
```
### 4 - django/forms/models.py:

Start line: 775, End line: 813

```python
class BaseModelFormSet(BaseFormSet):

    def save_existing_objects(self, commit=True):
        self.changed_objects = []
        self.deleted_objects = []
        if not self.initial_forms:
            return []

        saved_instances = []
        forms_to_delete = self.deleted_forms
        for form in self.initial_forms:
            obj = form.instance
            # If the pk is None, it means either:
            # 1. The object is an unexpected empty model, created by invalid
            #    POST data such as an object outside the formset's queryset.
            # 2. The object was already deleted from the database.
            if obj.pk is None:
                continue
            if form in forms_to_delete:
                self.deleted_objects.append(obj)
                self.delete_existing(obj, commit=commit)
            elif form.has_changed():
                self.changed_objects.append((obj, form.changed_data))
                saved_instances.append(self.save_existing(form, obj, commit=commit))
                if not commit:
                    self.saved_forms.append(form)
        return saved_instances

    def save_new_objects(self, commit=True):
        self.new_objects = []
        for form in self.extra_forms:
            if not form.has_changed():
                continue
            # If someone has marked an add form for deletion, don't save the
            # object.
            if self.can_delete and self._should_delete_form(form):
                continue
            self.new_objects.append(self.save_new(form, commit=commit))
            if not commit:
                self.saved_forms.append(form)
        return self.new_objects
```
### 5 - django/db/models/base.py:

Start line: 404, End line: 503

```python
class Model(metaclass=ModelBase):

    def __init__(self, *args, **kwargs):
        # Alias some things as locals to avoid repeat global lookups
        cls = self.__class__
        opts = self._meta
        _setattr = setattr
        _DEFERRED = DEFERRED

        pre_init.send(sender=cls, args=args, kwargs=kwargs)

        # Set up the storage for instance state
        self._state = ModelState()

        # There is a rather weird disparity here; if kwargs, it's set, then args
        # overrides it. It should be one or the other; don't duplicate the work
        # The reason for the kwargs check is that standard iterator passes in by
        # args, and instantiation for iteration is 33% faster.
        if len(args) > len(opts.concrete_fields):
            # Daft, but matches old exception sans the err msg.
            raise IndexError("Number of args exceeds number of fields")

        if not kwargs:
            fields_iter = iter(opts.concrete_fields)
            # The ordering of the zip calls matter - zip throws StopIteration
            # when an iter throws it. So if the first iter throws it, the second
            # is *not* consumed. We rely on this, so don't change the order
            # without changing the logic.
            for val, field in zip(args, fields_iter):
                if val is _DEFERRED:
                    continue
                _setattr(self, field.attname, val)
        else:
            # Slower, kwargs-ready version.
            fields_iter = iter(opts.fields)
            for val, field in zip(args, fields_iter):
                if val is _DEFERRED:
                    continue
                _setattr(self, field.attname, val)
                kwargs.pop(field.name, None)

        # Now we're left with the unprocessed fields that *must* come from
        # keywords, or default.

        for field in fields_iter:
            is_related_object = False
            # Virtual field
            if field.attname not in kwargs and field.column is None:
                continue
            if kwargs:
                if isinstance(field.remote_field, ForeignObjectRel):
                    try:
                        # Assume object instance was passed in.
                        rel_obj = kwargs.pop(field.name)
                        is_related_object = True
                    except KeyError:
                        try:
                            # Object instance wasn't passed in -- must be an ID.
                            val = kwargs.pop(field.attname)
                        except KeyError:
                            val = field.get_default()
                else:
                    try:
                        val = kwargs.pop(field.attname)
                    except KeyError:
                        # This is done with an exception rather than the
                        # default argument on pop because we don't want
                        # get_default() to be evaluated, and then not used.
                        # Refs #12057.
                        val = field.get_default()
            else:
                val = field.get_default()

            if is_related_object:
                # If we are passed a related instance, set it using the
                # field.name instead of field.attname (e.g. "user" instead of
                # "user_id") so that the object gets properly cached (and type
                # checked) by the RelatedObjectDescriptor.
                if rel_obj is not _DEFERRED:
                    _setattr(self, field.name, rel_obj)
            else:
                if val is not _DEFERRED:
                    _setattr(self, field.attname, val)

        if kwargs:
            property_names = opts._property_names
            for prop in tuple(kwargs):
                try:
                    # Any remaining kwargs must correspond to properties or
                    # virtual fields.
                    if prop in property_names or opts.get_field(prop):
                        if kwargs[prop] is not _DEFERRED:
                            _setattr(self, prop, kwargs[prop])
                        del kwargs[prop]
                except (AttributeError, FieldDoesNotExist):
                    pass
            for kwarg in kwargs:
                raise TypeError("%s() got an unexpected keyword argument '%s'" % (cls.__name__, kwarg))
        super().__init__()
        post_init.send(sender=cls, instance=self)
```
### 6 - django/db/models/base.py:

Start line: 829, End line: 890

```python
class Model(metaclass=ModelBase):

    def _save_table(self, raw=False, cls=None, force_insert=False,
                    force_update=False, using=None, update_fields=None):
        """
        Do the heavy-lifting involved in saving. Update or insert the data
        for a single table.
        """
        meta = cls._meta
        non_pks = [f for f in meta.local_concrete_fields if not f.primary_key]

        if update_fields:
            non_pks = [f for f in non_pks
                       if f.name in update_fields or f.attname in update_fields]

        pk_val = self._get_pk_val(meta)
        if pk_val is None:
            pk_val = meta.pk.get_pk_value_on_save(self)
            setattr(self, meta.pk.attname, pk_val)
        pk_set = pk_val is not None
        if not pk_set and (force_update or update_fields):
            raise ValueError("Cannot force an update in save() with no primary key.")
        updated = False
        # Skip an UPDATE when adding an instance and primary key has a default.
        if (
            not force_insert and
            self._state.adding and
            self._meta.pk.default and
            self._meta.pk.default is not NOT_PROVIDED
        ):
            force_insert = True
        # If possible, try an UPDATE. If that doesn't update anything, do an INSERT.
        if pk_set and not force_insert:
            base_qs = cls._base_manager.using(using)
            values = [(f, None, (getattr(self, f.attname) if raw else f.pre_save(self, False)))
                      for f in non_pks]
            forced_update = update_fields or force_update
            updated = self._do_update(base_qs, using, pk_val, values, update_fields,
                                      forced_update)
            if force_update and not updated:
                raise DatabaseError("Forced update did not affect any rows.")
            if update_fields and not updated:
                raise DatabaseError("Save with update_fields did not affect any rows.")
        if not updated:
            if meta.order_with_respect_to:
                # If this is a model with an order_with_respect_to
                # autopopulate the _order field
                field = meta.order_with_respect_to
                filter_args = field.get_filter_kwargs_for_object(self)
                self._order = cls._base_manager.using(using).filter(**filter_args).aggregate(
                    _order__max=Coalesce(
                        ExpressionWrapper(Max('_order') + Value(1), output_field=IntegerField()),
                        Value(0),
                    ),
                )['_order__max']
            fields = meta.local_concrete_fields
            if not pk_set:
                fields = [f for f in fields if f is not meta.auto_field]

            returning_fields = meta.db_returning_fields
            results = self._do_insert(cls._base_manager, using, fields, returning_fields, raw)
            for result, field in zip(results, returning_fields):
                setattr(self, field.attname, result)
        return updated
```
### 7 - django/db/models/fields/__init__.py:

Start line: 2282, End line: 2332

```python
class UUIDField(Field):
    default_error_messages = {
        'invalid': _('“%(value)s” is not a valid UUID.'),
    }
    description = _('Universally unique identifier')
    empty_strings_allowed = False

    def __init__(self, verbose_name=None, **kwargs):
        kwargs['max_length'] = 32
        super().__init__(verbose_name, **kwargs)

    def deconstruct(self):
        name, path, args, kwargs = super().deconstruct()
        del kwargs['max_length']
        return name, path, args, kwargs

    def get_internal_type(self):
        return "UUIDField"

    def get_prep_value(self, value):
        value = super().get_prep_value(value)
        return self.to_python(value)

    def get_db_prep_value(self, value, connection, prepared=False):
        if value is None:
            return None
        if not isinstance(value, uuid.UUID):
            value = self.to_python(value)

        if connection.features.has_native_uuid_field:
            return value
        return value.hex

    def to_python(self, value):
        if value is not None and not isinstance(value, uuid.UUID):
            input_form = 'int' if isinstance(value, int) else 'hex'
            try:
                return uuid.UUID(**{input_form: value})
            except (AttributeError, ValueError):
                raise exceptions.ValidationError(
                    self.error_messages['invalid'],
                    code='invalid',
                    params={'value': value},
                )
        return value

    def formfield(self, **kwargs):
        return super().formfield(**{
            'form_class': forms.UUIDField,
            **kwargs,
        })
```
### 8 - django/db/models/base.py:

Start line: 1861, End line: 1912

```python
############################################
# HELPER FUNCTIONS (CURRIED MODEL METHODS) #
############################################

# ORDERING METHODS #########################

def method_set_order(self, ordered_obj, id_list, using=None):
    if using is None:
        using = DEFAULT_DB_ALIAS
    order_wrt = ordered_obj._meta.order_with_respect_to
    filter_args = order_wrt.get_forward_related_filter(self)
    ordered_obj.objects.db_manager(using).filter(**filter_args).bulk_update([
        ordered_obj(pk=pk, _order=order) for order, pk in enumerate(id_list)
    ], ['_order'])


def method_get_order(self, ordered_obj):
    order_wrt = ordered_obj._meta.order_with_respect_to
    filter_args = order_wrt.get_forward_related_filter(self)
    pk_name = ordered_obj._meta.pk.name
    return ordered_obj.objects.filter(**filter_args).values_list(pk_name, flat=True)


def make_foreign_order_accessors(model, related_model):
    setattr(
        related_model,
        'get_%s_order' % model.__name__.lower(),
        partialmethod(method_get_order, model)
    )
    setattr(
        related_model,
        'set_%s_order' % model.__name__.lower(),
        partialmethod(method_set_order, model)
    )

########
# MISC #
########


def model_unpickle(model_id):
    """Used to unpickle Model subclasses with deferred fields."""
    if isinstance(model_id, tuple):
        model = apps.get_model(*model_id)
    else:
        # Backwards compat - the model was cached directly in earlier versions.
        model = model_id
    return model.__new__(model)


model_unpickle.__safe_for_unpickle__ = True
```
### 9 - django/db/models/base.py:

Start line: 1068, End line: 1111

```python
class Model(metaclass=ModelBase):

    def _perform_unique_checks(self, unique_checks):
        errors = {}

        for model_class, unique_check in unique_checks:
            # Try to look up an existing object with the same values as this
            # object's values for all the unique field.

            lookup_kwargs = {}
            for field_name in unique_check:
                f = self._meta.get_field(field_name)
                lookup_value = getattr(self, f.attname)
                # TODO: Handle multiple backends with different feature flags.
                if (lookup_value is None or
                        (lookup_value == '' and connection.features.interprets_empty_strings_as_nulls)):
                    # no value, skip the lookup
                    continue
                if f.primary_key and not self._state.adding:
                    # no need to check for unique primary key when editing
                    continue
                lookup_kwargs[str(field_name)] = lookup_value

            # some fields were skipped, no reason to do the check
            if len(unique_check) != len(lookup_kwargs):
                continue

            qs = model_class._default_manager.filter(**lookup_kwargs)

            # Exclude the current object from the query if we are editing an
            # instance (as opposed to creating a new one)
            # Note that we need to use the pk as defined by model_class, not
            # self.pk. These can be different fields because model inheritance
            # allows single model to have effectively multiple primary keys.
            # Refs #17615.
            model_class_pk = self._get_pk_val(model_class._meta)
            if not self._state.adding and model_class_pk is not None:
                qs = qs.exclude(pk=model_class_pk)
            if qs.exists():
                if len(unique_check) == 1:
                    key = unique_check[0]
                else:
                    key = NON_FIELD_ERRORS
                errors.setdefault(key, []).append(self.unique_error_message(model_class, unique_check))

        return errors
```
### 10 - django/db/backends/base/schema.py:

Start line: 370, End line: 384

```python
class BaseDatabaseSchemaEditor:

    def alter_unique_together(self, model, old_unique_together, new_unique_together):
        """
        Deal with a model changing its unique_together. The input
        unique_togethers must be doubly-nested, not the single-nested
        ["foo", "bar"] format.
        """
        olds = {tuple(fields) for fields in old_unique_together}
        news = {tuple(fields) for fields in new_unique_together}
        # Deleted uniques
        for fields in olds.difference(news):
            self._delete_composed_index(model, fields, {'unique': True}, self.sql_delete_unique)
        # Created uniques
        for fields in news.difference(olds):
            columns = [model._meta.get_field(field).column for field in fields]
            self.execute(self._create_unique_sql(model, columns))
```
### 15 - django/db/models/base.py:

Start line: 979, End line: 1007

```python
class Model(metaclass=ModelBase):

    def prepare_database_save(self, field):
        if self.pk is None:
            raise ValueError("Unsaved model instance %r cannot be used in an ORM query." % self)
        return getattr(self, field.remote_field.get_related_field().attname)

    def clean(self):
        """
        Hook for doing any extra model-wide validation after clean() has been
        called on every field by self.clean_fields. Any ValidationError raised
        by this method will not be associated with a particular field; it will
        have a special-case association with the field defined by NON_FIELD_ERRORS.
        """
        pass

    def validate_unique(self, exclude=None):
        """
        Check unique constraints on the model and raise ValidationError if any
        failed.
        """
        unique_checks, date_checks = self._get_unique_checks(exclude=exclude)

        errors = self._perform_unique_checks(unique_checks)
        date_errors = self._perform_date_checks(date_checks)

        for k, v in date_errors.items():
            errors.setdefault(k, []).extend(v)

        if errors:
            raise ValidationError(errors)
```
### 17 - django/db/models/base.py:

Start line: 799, End line: 827

```python
class Model(metaclass=ModelBase):

    save_base.alters_data = True

    def _save_parents(self, cls, using, update_fields):
        """Save all the parents of cls using values from self."""
        meta = cls._meta
        inserted = False
        for parent, field in meta.parents.items():
            # Make sure the link fields are synced between parent and self.
            if (field and getattr(self, parent._meta.pk.attname) is None and
                    getattr(self, field.attname) is not None):
                setattr(self, parent._meta.pk.attname, getattr(self, field.attname))
            parent_inserted = self._save_parents(cls=parent, using=using, update_fields=update_fields)
            updated = self._save_table(
                cls=parent, using=using, update_fields=update_fields,
                force_insert=parent_inserted,
            )
            if not updated:
                inserted = True
            # Set the parent's PK value to self.
            if field:
                setattr(self, field.attname, self._get_pk_val(parent._meta))
                # Since we didn't have an instance of the parent handy set
                # attname directly, bypassing the descriptor. Invalidate
                # the related object cache, in case it's been accidentally
                # populated. A fresh instance will be re-built from the
                # database if necessary.
                if field.is_cached(self):
                    field.delete_cached_value(self)
        return inserted
```
### 29 - django/db/models/base.py:

Start line: 948, End line: 962

```python
class Model(metaclass=ModelBase):

    def _get_next_or_previous_by_FIELD(self, field, is_next, **kwargs):
        if not self.pk:
            raise ValueError("get_next/get_previous cannot be used on unsaved objects.")
        op = 'gt' if is_next else 'lt'
        order = '' if is_next else '-'
        param = getattr(self, field.attname)
        q = Q(**{'%s__%s' % (field.name, op): param})
        q = q | Q(**{field.name: param, 'pk__%s' % op: self.pk})
        qs = self.__class__._default_manager.using(self._state.db).filter(**kwargs).filter(q).order_by(
            '%s%s' % (order, field.name), '%spk' % order
        )
        try:
            return qs[0]
        except IndexError:
            raise self.DoesNotExist("%s matching query does not exist." % self.__class__._meta.object_name)
```
### 31 - django/db/models/base.py:

Start line: 919, End line: 946

```python
class Model(metaclass=ModelBase):

    def _do_insert(self, manager, using, fields, returning_fields, raw):
        """
        Do an INSERT. If returning_fields is defined then this method should
        return the newly created data for the model.
        """
        return manager._insert(
            [self], fields=fields, returning_fields=returning_fields,
            using=using, raw=raw,
        )

    def delete(self, using=None, keep_parents=False):
        using = using or router.db_for_write(self.__class__, instance=self)
        assert self.pk is not None, (
            "%s object can't be deleted because its %s attribute is set to None." %
            (self._meta.object_name, self._meta.pk.attname)
        )

        collector = Collector(using=using)
        collector.collect([self], keep_parents=keep_parents)
        return collector.delete()

    delete.alters_data = True

    def _get_FIELD_display(self, field):
        value = getattr(self, field.attname)
        choices_dict = dict(make_hashable(field.flatchoices))
        # force_str() to coerce lazy strings.
        return force_str(choices_dict.get(make_hashable(value), value), strings_only=True)
```
### 45 - django/db/models/base.py:

Start line: 1113, End line: 1140

```python
class Model(metaclass=ModelBase):

    def _perform_date_checks(self, date_checks):
        errors = {}
        for model_class, lookup_type, field, unique_for in date_checks:
            lookup_kwargs = {}
            # there's a ticket to add a date lookup, we can remove this special
            # case if that makes it's way in
            date = getattr(self, unique_for)
            if date is None:
                continue
            if lookup_type == 'date':
                lookup_kwargs['%s__day' % unique_for] = date.day
                lookup_kwargs['%s__month' % unique_for] = date.month
                lookup_kwargs['%s__year' % unique_for] = date.year
            else:
                lookup_kwargs['%s__%s' % (unique_for, lookup_type)] = getattr(date, lookup_type)
            lookup_kwargs[field] = getattr(self, field)

            qs = model_class._default_manager.filter(**lookup_kwargs)
            # Exclude the current object from the query if we are editing an
            # instance (as opposed to creating a new one)
            if not self._state.adding and self.pk is not None:
                qs = qs.exclude(pk=self.pk)

            if qs.exists():
                errors.setdefault(field, []).append(
                    self.date_error_message(lookup_type, field, unique_for)
                )
        return errors
```
### 46 - django/db/models/base.py:

Start line: 549, End line: 565

```python
class Model(metaclass=ModelBase):

    def __setstate__(self, state):
        msg = None
        pickled_version = state.get(DJANGO_VERSION_PICKLE_KEY)
        if pickled_version:
            current_version = get_version()
            if current_version != pickled_version:
                msg = (
                    "Pickled model instance's Django version %s does not match "
                    "the current version %s." % (pickled_version, current_version)
                )
        else:
            msg = "Pickled model instance's Django version is not specified."

        if msg:
            warnings.warn(msg, RuntimeWarning, stacklevel=2)

        self.__dict__.update(state)
```
### 47 - django/db/models/base.py:

Start line: 505, End line: 547

```python
class Model(metaclass=ModelBase):

    @classmethod
    def from_db(cls, db, field_names, values):
        if len(values) != len(cls._meta.concrete_fields):
            values_iter = iter(values)
            values = [
                next(values_iter) if f.attname in field_names else DEFERRED
                for f in cls._meta.concrete_fields
            ]
        new = cls(*values)
        new._state.adding = False
        new._state.db = db
        return new

    def __repr__(self):
        return '<%s: %s>' % (self.__class__.__name__, self)

    def __str__(self):
        return '%s object (%s)' % (self.__class__.__name__, self.pk)

    def __eq__(self, other):
        if not isinstance(other, Model):
            return NotImplemented
        if self._meta.concrete_model != other._meta.concrete_model:
            return False
        my_pk = self.pk
        if my_pk is None:
            return self is other
        return my_pk == other.pk

    def __hash__(self):
        if self.pk is None:
            raise TypeError("Model instances without primary key value are unhashable")
        return hash(self.pk)

    def __reduce__(self):
        data = self.__getstate__()
        data[DJANGO_VERSION_PICKLE_KEY] = get_version()
        class_id = self._meta.app_label, self._meta.object_name
        return model_unpickle, (class_id,), data

    def __getstate__(self):
        """Hook to allow choosing the attributes to pickle."""
        return self.__dict__
```
### 53 - django/db/models/base.py:

Start line: 1250, End line: 1279

```python
class Model(metaclass=ModelBase):

    @classmethod
    def check(cls, **kwargs):
        errors = [*cls._check_swappable(), *cls._check_model(), *cls._check_managers(**kwargs)]
        if not cls._meta.swapped:
            errors += [
                *cls._check_fields(**kwargs),
                *cls._check_m2m_through_same_relationship(),
                *cls._check_long_column_names(),
            ]
            clash_errors = (
                *cls._check_id_field(),
                *cls._check_field_name_clashes(),
                *cls._check_model_name_db_lookup_clashes(),
                *cls._check_property_name_related_field_accessor_clashes(),
                *cls._check_single_primary_key(),
            )
            errors.extend(clash_errors)
            # If there are field name clashes, hide consequent column name
            # clashes.
            if not clash_errors:
                errors.extend(cls._check_column_name_clashes())
            errors += [
                *cls._check_index_together(),
                *cls._check_unique_together(),
                *cls._check_indexes(),
                *cls._check_ordering(),
                *cls._check_constraints(),
            ]

        return errors
```
### 62 - django/db/models/base.py:

Start line: 1159, End line: 1187

```python
class Model(metaclass=ModelBase):

    def unique_error_message(self, model_class, unique_check):
        opts = model_class._meta

        params = {
            'model': self,
            'model_class': model_class,
            'model_name': capfirst(opts.verbose_name),
            'unique_check': unique_check,
        }

        # A unique field
        if len(unique_check) == 1:
            field = opts.get_field(unique_check[0])
            params['field_label'] = capfirst(field.verbose_name)
            return ValidationError(
                message=field.error_messages['unique'],
                code='unique',
                params=params,
            )

        # unique_together
        else:
            field_labels = [capfirst(opts.get_field(f).verbose_name) for f in unique_check]
            params['field_labels'] = get_text_list(field_labels, _('and'))
            return ValidationError(
                message=_("%(model_name)s with this %(field_labels)s already exists."),
                code='unique_together',
                params=params,
            )
```
### 70 - django/db/models/base.py:

Start line: 385, End line: 401

```python
class ModelStateFieldsCacheDescriptor:
    def __get__(self, instance, cls=None):
        if instance is None:
            return self
        res = instance.fields_cache = {}
        return res


class ModelState:
    """Store model instance state."""
    db = None
    # If true, uniqueness validation checks will consider this a new, unsaved
    # object. Necessary for correct validation of new instances of objects with
    # explicit (non-auto) PKs. This impacts validation only; it has no effect
    # on the actual save.
    adding = True
    fields_cache = ModelStateFieldsCacheDescriptor()
```
### 86 - django/db/models/base.py:

Start line: 212, End line: 322

```python
class ModelBase(type):
    def __new__(cls, name, bases, attrs, **kwargs):
        # ... other code
        for base in new_class.mro():
            if base not in parents or not hasattr(base, '_meta'):
                # Things without _meta aren't functional models, so they're
                # uninteresting parents.
                inherited_attributes.update(base.__dict__)
                continue

            parent_fields = base._meta.local_fields + base._meta.local_many_to_many
            if not base._meta.abstract:
                # Check for clashes between locally declared fields and those
                # on the base classes.
                for field in parent_fields:
                    if field.name in field_names:
                        raise FieldError(
                            'Local field %r in class %r clashes with field of '
                            'the same name from base class %r.' % (
                                field.name,
                                name,
                                base.__name__,
                            )
                        )
                    else:
                        inherited_attributes.add(field.name)

                # Concrete classes...
                base = base._meta.concrete_model
                base_key = make_model_tuple(base)
                if base_key in parent_links:
                    field = parent_links[base_key]
                elif not is_proxy:
                    attr_name = '%s_ptr' % base._meta.model_name
                    field = OneToOneField(
                        base,
                        on_delete=CASCADE,
                        name=attr_name,
                        auto_created=True,
                        parent_link=True,
                    )

                    if attr_name in field_names:
                        raise FieldError(
                            "Auto-generated field '%s' in class %r for "
                            "parent_link to base class %r clashes with "
                            "declared field of the same name." % (
                                attr_name,
                                name,
                                base.__name__,
                            )
                        )

                    # Only add the ptr field if it's not already present;
                    # e.g. migrations will already have it specified
                    if not hasattr(new_class, attr_name):
                        new_class.add_to_class(attr_name, field)
                else:
                    field = None
                new_class._meta.parents[base] = field
            else:
                base_parents = base._meta.parents.copy()

                # Add fields from abstract base class if it wasn't overridden.
                for field in parent_fields:
                    if (field.name not in field_names and
                            field.name not in new_class.__dict__ and
                            field.name not in inherited_attributes):
                        new_field = copy.deepcopy(field)
                        new_class.add_to_class(field.name, new_field)
                        # Replace parent links defined on this base by the new
                        # field. It will be appropriately resolved if required.
                        if field.one_to_one:
                            for parent, parent_link in base_parents.items():
                                if field == parent_link:
                                    base_parents[parent] = new_field

                # Pass any non-abstract parent classes onto child.
                new_class._meta.parents.update(base_parents)

            # Inherit private fields (like GenericForeignKey) from the parent
            # class
            for field in base._meta.private_fields:
                if field.name in field_names:
                    if not base._meta.abstract:
                        raise FieldError(
                            'Local field %r in class %r clashes with field of '
                            'the same name from base class %r.' % (
                                field.name,
                                name,
                                base.__name__,
                            )
                        )
                else:
                    field = copy.deepcopy(field)
                    if not base._meta.abstract:
                        field.mti_inherited = True
                    new_class.add_to_class(field.name, field)

        # Copy indexes so that index names are unique when models extend an
        # abstract model.
        new_class._meta.indexes = [copy.deepcopy(idx) for idx in new_class._meta.indexes]

        if abstract:
            # Abstract base models can't be instantiated and don't appear in
            # the list of models for an app. We do the final setup for them a
            # little differently from normal models.
            attr_meta.abstract = False
            new_class.Meta = attr_meta
            return new_class

        new_class._prepare()
        new_class._meta.apps.register_model(new_class._meta.app_label, new_class)
        return new_class
```
### 90 - django/db/models/base.py:

Start line: 1470, End line: 1492

```python
class Model(metaclass=ModelBase):

    @classmethod
    def _check_model_name_db_lookup_clashes(cls):
        errors = []
        model_name = cls.__name__
        if model_name.startswith('_') or model_name.endswith('_'):
            errors.append(
                checks.Error(
                    "The model name '%s' cannot start or end with an underscore "
                    "as it collides with the query lookup syntax." % model_name,
                    obj=cls,
                    id='models.E023'
                )
            )
        elif LOOKUP_SEP in model_name:
            errors.append(
                checks.Error(
                    "The model name '%s' cannot contain double underscores as "
                    "it collides with the query lookup syntax." % model_name,
                    obj=cls,
                    id='models.E024'
                )
            )
        return errors
```
### 96 - django/db/models/base.py:

Start line: 1, End line: 50

```python
import copy
import inspect
import warnings
from functools import partialmethod
from itertools import chain

from django.apps import apps
from django.conf import settings
from django.core import checks
from django.core.exceptions import (
    NON_FIELD_ERRORS, FieldDoesNotExist, FieldError, MultipleObjectsReturned,
    ObjectDoesNotExist, ValidationError,
)
from django.db import (
    DEFAULT_DB_ALIAS, DJANGO_VERSION_PICKLE_KEY, DatabaseError, connection,
    connections, router, transaction,
)
from django.db.models import (
    NOT_PROVIDED, ExpressionWrapper, IntegerField, Max, Value,
)
from django.db.models.constants import LOOKUP_SEP
from django.db.models.constraints import CheckConstraint, UniqueConstraint
from django.db.models.deletion import CASCADE, Collector
from django.db.models.fields.related import (
    ForeignObjectRel, OneToOneField, lazy_related_operation, resolve_relation,
)
from django.db.models.functions import Coalesce
from django.db.models.manager import Manager
from django.db.models.options import Options
from django.db.models.query import Q
from django.db.models.signals import (
    class_prepared, post_init, post_save, pre_init, pre_save,
)
from django.db.models.utils import make_model_tuple
from django.utils.encoding import force_str
from django.utils.hashable import make_hashable
from django.utils.text import capfirst, get_text_list
from django.utils.translation import gettext_lazy as _
from django.utils.version import get_version


class Deferred:
    def __repr__(self):
        return '<Deferred field>'

    def __str__(self):
        return '<Deferred field>'


DEFERRED = Deferred()
```
