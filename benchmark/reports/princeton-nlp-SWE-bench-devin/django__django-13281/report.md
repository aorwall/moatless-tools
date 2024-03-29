# django__django-13281

| **django/django** | `63300f7e686c2c452763cb512df9abf7734fd588` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 27674 |
| **Any found context length** | 27674 |
| **Avg pos** | 79.0 |
| **Min pos** | 79 |
| **Max pos** | 79 |
| **Top file pos** | 5 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/db/models/base.py b/django/db/models/base.py
--- a/django/db/models/base.py
+++ b/django/db/models/base.py
@@ -546,7 +546,10 @@ def __reduce__(self):
 
     def __getstate__(self):
         """Hook to allow choosing the attributes to pickle."""
-        return self.__dict__
+        state = self.__dict__.copy()
+        state['_state'] = copy.copy(state['_state'])
+        state['_state'].fields_cache = state['_state'].fields_cache.copy()
+        return state
 
     def __setstate__(self, state):
         pickled_version = state.get(DJANGO_VERSION_PICKLE_KEY)

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/db/models/base.py | 549 | 549 | 79 | 5 | 27674


## Problem Statement

```
FK field caching behavior change between 1.11.x and 2.x
Description
	
Whilst upgrading a codebase from 1.11.x to 2.0/2.2 I noticed a weird change in behavior of FK fields when copying model instances.
At the bottom of the post there is a testcase that succeeds on 1.11.x and fails on 2.x
I think the commit that changed the behavior is bfb746f983aa741afa3709794e70f1e0ab6040b5
So my question is two fold:
Is the behavior in >=2.0 correct? It seems quite unexpected.
What is the recommended way to clone a model instance? To date we have been using copy() in a similar fashion to the test without issue. deepcopy seems to work fine in >=2.0 but we haven’t done too much testing yet.
Test (placed in tests/model_fields/test_field_caching_change.py):
import copy
from django.test import TestCase
from .models import Bar, Foo
class ForeignKeyCachingBehaviorTest(TestCase):
	def test_copy(self):
		foo1 = Foo.objects.create(a='foo1', d=1)
		foo2 = Foo.objects.create(a='foo2', d=2)
		bar1 = Bar.objects.create(a=foo1, b='bar1')
		bar2 = copy.copy(bar1)
		bar2.pk = None
		bar2.a = foo2
		# bar2 points to foo2
		self.assertEqual(bar2.a, foo2)
		self.assertEqual(bar2.a.id, bar2.a_id)
		# bar1 is unchanged and must still point to foo1
		# These fail on Django >= 2.0
		self.assertEqual(bar1.a, foo1)
		self.assertEqual(bar1.a.id, bar1.a_id)
and executed that via:
python3.6 tests/runtests.py --parallel 1 model_fields
In ​https://groups.google.com/g/django-developers/c/QMhVPIqVVP4/m/mbezfaBEAwAJ Simon suggests:
..... Model.copy should make sure to make a deep-copy of self._state now that fields are cached in self._state.fields_cache.
which I will attempt to implement.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 django/db/models/fields/related.py | 255 | 282| 269 | 269 | 13876 | 
| 2 | 1 django/db/models/fields/related.py | 864 | 890| 240 | 509 | 13876 | 
| 3 | 1 django/db/models/fields/related.py | 190 | 254| 673 | 1182 | 13876 | 
| 4 | 2 django/core/cache/backends/db.py | 112 | 197| 794 | 1976 | 15998 | 
| 5 | 3 django/contrib/contenttypes/fields.py | 173 | 217| 411 | 2387 | 21431 | 
| 6 | 3 django/db/models/fields/related.py | 171 | 188| 166 | 2553 | 21431 | 
| 7 | 4 django/db/models/fields/__init__.py | 508 | 548| 329 | 2882 | 39120 | 
| 8 | 4 django/db/models/fields/related.py | 1235 | 1352| 963 | 3845 | 39120 | 
| 9 | 4 django/db/models/fields/related.py | 284 | 318| 293 | 4138 | 39120 | 
| 10 | **5 django/db/models/base.py** | 404 | 505| 871 | 5009 | 55716 | 
| 11 | 5 django/contrib/contenttypes/fields.py | 160 | 171| 123 | 5132 | 55716 | 
| 12 | 5 django/core/cache/backends/db.py | 199 | 228| 285 | 5417 | 55716 | 
| 13 | 5 django/core/cache/backends/db.py | 40 | 95| 431 | 5848 | 55716 | 
| 14 | 6 django/db/backends/sqlite3/creation.py | 51 | 82| 317 | 6165 | 56567 | 
| 15 | 6 django/core/cache/backends/db.py | 255 | 283| 324 | 6489 | 56567 | 
| 16 | 6 django/core/cache/backends/db.py | 230 | 253| 259 | 6748 | 56567 | 
| 17 | **6 django/db/models/base.py** | 169 | 211| 413 | 7161 | 56567 | 
| 18 | **6 django/db/models/base.py** | 1396 | 1451| 491 | 7652 | 56567 | 
| 19 | **6 django/db/models/base.py** | 2033 | 2084| 351 | 8003 | 56567 | 
| 20 | 6 django/core/cache/backends/db.py | 97 | 110| 234 | 8237 | 56567 | 
| 21 | 7 django/db/models/fields/related_descriptors.py | 120 | 154| 405 | 8642 | 66955 | 
| 22 | 7 django/db/models/fields/related.py | 750 | 768| 222 | 8864 | 66955 | 
| 23 | **7 django/db/models/base.py** | 1075 | 1118| 404 | 9268 | 66955 | 
| 24 | **7 django/db/models/base.py** | 956 | 970| 212 | 9480 | 66955 | 
| 25 | 7 django/db/models/fields/related.py | 127 | 154| 201 | 9681 | 66955 | 
| 26 | 7 django/db/models/fields/related.py | 841 | 862| 169 | 9850 | 66955 | 
| 27 | 8 django/core/serializers/xml_serializer.py | 93 | 114| 192 | 10042 | 70467 | 
| 28 | 8 django/db/models/fields/related.py | 997 | 1024| 215 | 10257 | 70467 | 
| 29 | 9 django/db/migrations/state.py | 591 | 607| 146 | 10403 | 75589 | 
| 30 | 10 django/db/models/options.py | 256 | 288| 331 | 10734 | 82695 | 
| 31 | 11 django/db/models/fields/mixins.py | 1 | 28| 168 | 10902 | 83038 | 
| 32 | **11 django/db/models/base.py** | 1 | 50| 328 | 11230 | 83038 | 
| 33 | 11 django/db/models/fields/related.py | 630 | 650| 168 | 11398 | 83038 | 
| 34 | 11 django/db/models/fields/related.py | 913 | 933| 178 | 11576 | 83038 | 
| 35 | 12 django/db/backends/base/schema.py | 574 | 636| 700 | 12276 | 94880 | 
| 36 | 13 django/db/models/query.py | 1651 | 1757| 1063 | 13339 | 112059 | 
| 37 | 13 django/db/models/options.py | 524 | 552| 231 | 13570 | 112059 | 
| 38 | 14 django/db/models/sql/query.py | 288 | 337| 444 | 14014 | 134506 | 
| 39 | 14 django/db/backends/base/schema.py | 370 | 384| 182 | 14196 | 134506 | 
| 40 | **14 django/db/models/base.py** | 1656 | 1704| 348 | 14544 | 134506 | 
| 41 | 14 django/db/models/fields/related_descriptors.py | 365 | 381| 184 | 14728 | 134506 | 
| 42 | **14 django/db/models/base.py** | 1257 | 1287| 259 | 14987 | 134506 | 
| 43 | 15 django/db/backends/sqlite3/schema.py | 384 | 430| 444 | 15431 | 138622 | 
| 44 | 16 django/db/backends/base/features.py | 1 | 113| 899 | 16330 | 141346 | 
| 45 | 16 django/db/models/fields/related.py | 935 | 948| 126 | 16456 | 141346 | 
| 46 | 16 django/db/models/fields/related.py | 576 | 609| 334 | 16790 | 141346 | 
| 47 | 16 django/db/models/fields/related.py | 509 | 574| 492 | 17282 | 141346 | 
| 48 | 16 django/db/backends/base/schema.py | 896 | 915| 296 | 17578 | 141346 | 
| 49 | **16 django/db/models/base.py** | 385 | 401| 128 | 17706 | 141346 | 
| 50 | **16 django/db/models/base.py** | 972 | 985| 180 | 17886 | 141346 | 
| 51 | **16 django/db/models/base.py** | 1502 | 1534| 231 | 18117 | 141346 | 
| 52 | 17 django/db/migrations/operations/models.py | 339 | 388| 493 | 18610 | 148241 | 
| 53 | 17 django/db/models/fields/related.py | 1354 | 1426| 616 | 19226 | 148241 | 
| 54 | 17 django/db/models/fields/related.py | 156 | 169| 144 | 19370 | 148241 | 
| 55 | 18 django/db/migrations/operations/fields.py | 216 | 234| 185 | 19555 | 151339 | 
| 56 | **18 django/db/models/base.py** | 551 | 568| 142 | 19697 | 151339 | 
| 57 | 18 django/db/models/fields/related_descriptors.py | 344 | 363| 156 | 19853 | 151339 | 
| 58 | 19 django/core/cache/backends/filebased.py | 61 | 96| 260 | 20113 | 152512 | 
| 59 | 19 django/db/migrations/operations/fields.py | 1 | 37| 241 | 20354 | 152512 | 
| 60 | 19 django/db/models/fields/related.py | 108 | 125| 155 | 20509 | 152512 | 
| 61 | **19 django/db/models/base.py** | 212 | 322| 866 | 21375 | 152512 | 
| 62 | 19 django/db/models/fields/related.py | 652 | 668| 163 | 21538 | 152512 | 
| 63 | 20 django/db/models/sql/datastructures.py | 104 | 115| 133 | 21671 | 153914 | 
| 64 | 20 django/db/migrations/state.py | 526 | 551| 228 | 21899 | 153914 | 
| 65 | **20 django/db/models/base.py** | 1347 | 1377| 244 | 22143 | 153914 | 
| 66 | 21 django/core/serializers/base.py | 301 | 323| 207 | 22350 | 156339 | 
| 67 | 22 django/db/models/fields/reverse_related.py | 156 | 181| 269 | 22619 | 158482 | 
| 68 | 22 django/db/migrations/operations/models.py | 1 | 38| 235 | 22854 | 158482 | 
| 69 | 23 django/db/backends/mysql/creation.py | 31 | 55| 253 | 23107 | 159092 | 
| 70 | 24 django/db/models/query_utils.py | 284 | 309| 293 | 23400 | 161798 | 
| 71 | 24 django/db/migrations/operations/models.py | 312 | 337| 290 | 23690 | 161798 | 
| 72 | 24 django/db/backends/base/schema.py | 637 | 709| 796 | 24486 | 161798 | 
| 73 | 24 django/db/backends/sqlite3/schema.py | 348 | 382| 422 | 24908 | 161798 | 
| 74 | 24 django/contrib/contenttypes/fields.py | 20 | 108| 571 | 25479 | 161798 | 
| 75 | 24 django/db/models/fields/related.py | 1428 | 1469| 418 | 25897 | 161798 | 
| 76 | 24 django/db/backends/sqlite3/schema.py | 101 | 138| 486 | 26383 | 161798 | 
| 77 | 25 django/db/backends/sqlite3/features.py | 1 | 80| 725 | 27108 | 162523 | 
| 78 | 25 django/core/serializers/base.py | 273 | 298| 218 | 27326 | 162523 | 
| **-> 79 <-** | **25 django/db/models/base.py** | 507 | 549| 348 | 27674 | 162523 | 
| 80 | 25 django/db/models/query.py | 787 | 803| 157 | 27831 | 162523 | 
| 81 | 26 django/db/backends/mysql/features.py | 1 | 112| 847 | 28678 | 163925 | 
| 82 | 26 django/db/backends/base/schema.py | 31 | 41| 120 | 28798 | 163925 | 
| 83 | 26 django/db/models/query.py | 1563 | 1619| 481 | 29279 | 163925 | 
| 84 | 27 django/db/backends/base/creation.py | 207 | 224| 154 | 29433 | 166334 | 
| 85 | 27 django/db/models/fields/reverse_related.py | 136 | 154| 172 | 29605 | 166334 | 
| 86 | 27 django/db/models/sql/query.py | 697 | 732| 389 | 29994 | 166334 | 
| 87 | 27 django/db/models/fields/related.py | 984 | 995| 128 | 30122 | 166334 | 
| 88 | 27 django/db/models/fields/related.py | 1606 | 1643| 484 | 30606 | 166334 | 
| 89 | 27 django/db/backends/sqlite3/schema.py | 140 | 221| 820 | 31426 | 166334 | 
| 90 | 27 django/core/cache/backends/filebased.py | 1 | 44| 284 | 31710 | 166334 | 
| 91 | 27 django/db/backends/sqlite3/schema.py | 223 | 305| 731 | 32441 | 166334 | 
| 92 | **27 django/db/models/base.py** | 1120 | 1147| 286 | 32727 | 166334 | 
| 93 | 27 django/db/models/fields/related.py | 444 | 485| 273 | 33000 | 166334 | 
| 94 | 28 django/db/backends/postgresql/creation.py | 53 | 78| 247 | 33247 | 166980 | 
| 95 | 29 django/core/serializers/python.py | 62 | 77| 156 | 33403 | 168242 | 
| 96 | 30 django/db/migrations/autodetector.py | 805 | 854| 567 | 33970 | 179861 | 
| 97 | 31 django/forms/fields.py | 175 | 207| 280 | 34250 | 189208 | 
| 98 | 31 django/db/backends/base/schema.py | 780 | 820| 519 | 34769 | 189208 | 
| 99 | **31 django/db/models/base.py** | 591 | 650| 527 | 35296 | 189208 | 
| 100 | 31 django/db/models/fields/related.py | 771 | 839| 521 | 35817 | 189208 | 
| 101 | 32 django/core/cache/backends/dummy.py | 1 | 40| 255 | 36072 | 189464 | 
| 102 | 32 django/db/models/fields/related.py | 950 | 982| 279 | 36351 | 189464 | 
| 103 | 32 django/contrib/contenttypes/fields.py | 219 | 270| 459 | 36810 | 189464 | 
| 104 | 32 django/db/models/fields/related_descriptors.py | 671 | 729| 548 | 37358 | 189464 | 
| 105 | 32 django/db/migrations/autodetector.py | 913 | 994| 876 | 38234 | 189464 | 
| 106 | 32 django/db/models/fields/related.py | 487 | 507| 138 | 38372 | 189464 | 
| 107 | **32 django/db/models/base.py** | 754 | 803| 456 | 38828 | 189464 | 
| 108 | 32 django/db/backends/base/creation.py | 226 | 242| 173 | 39001 | 189464 | 
| 109 | 32 django/db/models/options.py | 716 | 731| 144 | 39145 | 189464 | 
| 110 | 32 django/core/cache/backends/filebased.py | 46 | 59| 145 | 39290 | 189464 | 
| 111 | **32 django/db/models/base.py** | 1706 | 1806| 729 | 40019 | 189464 | 
| 112 | 32 django/db/models/options.py | 290 | 308| 124 | 40143 | 189464 | 
| 113 | 32 django/db/models/sql/query.py | 648 | 695| 511 | 40654 | 189464 | 
| 114 | 32 django/contrib/contenttypes/fields.py | 110 | 158| 328 | 40982 | 189464 | 
| 115 | 32 django/db/models/fields/related_descriptors.py | 309 | 323| 182 | 41164 | 189464 | 
| 116 | 32 django/db/models/sql/query.py | 339 | 362| 179 | 41343 | 189464 | 
| 117 | 32 django/db/migrations/autodetector.py | 1095 | 1130| 312 | 41655 | 189464 | 
| 118 | 32 django/db/models/fields/related.py | 1 | 34| 246 | 41901 | 189464 | 
| 119 | 32 django/db/models/fields/__init__.py | 550 | 568| 224 | 42125 | 189464 | 
| 120 | 33 django/db/models/fields/files.py | 1 | 141| 940 | 43065 | 193244 | 
| 121 | **33 django/db/models/base.py** | 1453 | 1476| 176 | 43241 | 193244 | 
| 122 | 34 django/contrib/auth/migrations/0011_update_proxy_permissions.py | 1 | 51| 444 | 43685 | 193797 | 
| 123 | 34 django/db/models/fields/related.py | 1027 | 1074| 368 | 44053 | 193797 | 
| 124 | 34 django/db/models/query.py | 1872 | 1904| 314 | 44367 | 193797 | 
| 125 | 34 django/db/migrations/operations/fields.py | 85 | 95| 124 | 44491 | 193797 | 
| 126 | 34 django/db/models/fields/related.py | 83 | 106| 162 | 44653 | 193797 | 
| 127 | 35 django/db/models/deletion.py | 379 | 448| 580 | 45233 | 197623 | 
| 128 | 35 django/db/models/fields/related.py | 611 | 628| 197 | 45430 | 197623 | 
| 129 | 36 django/core/cache/backends/memcached.py | 100 | 117| 173 | 45603 | 199535 | 
| 130 | 36 django/db/models/options.py | 357 | 379| 164 | 45767 | 199535 | 
| 131 | 36 django/db/backends/base/schema.py | 1073 | 1095| 199 | 45966 | 199535 | 


## Patch

```diff
diff --git a/django/db/models/base.py b/django/db/models/base.py
--- a/django/db/models/base.py
+++ b/django/db/models/base.py
@@ -546,7 +546,10 @@ def __reduce__(self):
 
     def __getstate__(self):
         """Hook to allow choosing the attributes to pickle."""
-        return self.__dict__
+        state = self.__dict__.copy()
+        state['_state'] = copy.copy(state['_state'])
+        state['_state'].fields_cache = state['_state'].fields_cache.copy()
+        return state
 
     def __setstate__(self, state):
         pickled_version = state.get(DJANGO_VERSION_PICKLE_KEY)

```

## Test Patch

```diff
diff --git a/tests/model_regress/tests.py b/tests/model_regress/tests.py
--- a/tests/model_regress/tests.py
+++ b/tests/model_regress/tests.py
@@ -1,3 +1,4 @@
+import copy
 import datetime
 from operator import attrgetter
 
@@ -256,3 +257,17 @@ def test_model_with_evaluate_method(self):
         dept = Department.objects.create(pk=1, name='abc')
         dept.evaluate = 'abc'
         Worker.objects.filter(department=dept)
+
+
+class ModelFieldsCacheTest(TestCase):
+    def test_fields_cache_reset_on_copy(self):
+        department1 = Department.objects.create(id=1, name='department1')
+        department2 = Department.objects.create(id=2, name='department2')
+        worker1 = Worker.objects.create(name='worker', department=department1)
+        worker2 = copy.copy(worker1)
+
+        self.assertEqual(worker2.department, department1)
+        # Changing related fields doesn't mutate the base object.
+        worker2.department = department2
+        self.assertEqual(worker2.department, department2)
+        self.assertEqual(worker1.department, department1)

```


## Code snippets

### 1 - django/db/models/fields/related.py:

Start line: 255, End line: 282

```python
class RelatedField(FieldCacheMixin, Field):

    def _check_clashes(self):
        # ... other code
        for clash_field in potential_clashes:
            clash_name = "%s.%s" % (  # i. e. "Model.m2m"
                clash_field.related_model._meta.object_name,
                clash_field.field.name)
            if not rel_is_hidden and clash_field.get_accessor_name() == rel_name:
                errors.append(
                    checks.Error(
                        "Reverse accessor for '%s' clashes with reverse accessor for '%s'." % (field_name, clash_name),
                        hint=("Add or change a related_name argument "
                              "to the definition for '%s' or '%s'.") % (field_name, clash_name),
                        obj=self,
                        id='fields.E304',
                    )
                )

            if clash_field.get_accessor_name() == rel_query_name:
                errors.append(
                    checks.Error(
                        "Reverse query name for '%s' clashes with reverse query name for '%s'."
                        % (field_name, clash_name),
                        hint=("Add or change a related_name argument "
                              "to the definition for '%s' or '%s'.") % (field_name, clash_name),
                        obj=self,
                        id='fields.E305',
                    )
                )

        return errors
```
### 2 - django/db/models/fields/related.py:

Start line: 864, End line: 890

```python
class ForeignKey(ForeignObject):

    def _check_unique(self, **kwargs):
        return [
            checks.Warning(
                'Setting unique=True on a ForeignKey has the same effect as using a OneToOneField.',
                hint='ForeignKey(unique=True) is usually better served by a OneToOneField.',
                obj=self,
                id='fields.W342',
            )
        ] if self.unique else []

    def deconstruct(self):
        name, path, args, kwargs = super().deconstruct()
        del kwargs['to_fields']
        del kwargs['from_fields']
        # Handle the simpler arguments
        if self.db_index:
            del kwargs['db_index']
        else:
            kwargs['db_index'] = False
        if self.db_constraint is not True:
            kwargs['db_constraint'] = self.db_constraint
        # Rel needs more work.
        to_meta = getattr(self.remote_field.model, "_meta", None)
        if self.remote_field.field_name and (
                not to_meta or (to_meta.pk and self.remote_field.field_name != to_meta.pk.name)):
            kwargs['to_field'] = self.remote_field.field_name
        return name, path, args, kwargs
```
### 3 - django/db/models/fields/related.py:

Start line: 190, End line: 254

```python
class RelatedField(FieldCacheMixin, Field):

    def _check_clashes(self):
        """Check accessor and reverse query name clashes."""
        from django.db.models.base import ModelBase

        errors = []
        opts = self.model._meta

        # `f.remote_field.model` may be a string instead of a model. Skip if model name is
        # not resolved.
        if not isinstance(self.remote_field.model, ModelBase):
            return []

        # Consider that we are checking field `Model.foreign` and the models
        # are:
        #
        #     class Target(models.Model):
        #         model = models.IntegerField()
        #         model_set = models.IntegerField()
        #
        #     class Model(models.Model):
        #         foreign = models.ForeignKey(Target)
        #         m2m = models.ManyToManyField(Target)

        # rel_opts.object_name == "Target"
        rel_opts = self.remote_field.model._meta
        # If the field doesn't install a backward relation on the target model
        # (so `is_hidden` returns True), then there are no clashes to check
        # and we can skip these fields.
        rel_is_hidden = self.remote_field.is_hidden()
        rel_name = self.remote_field.get_accessor_name()  # i. e. "model_set"
        rel_query_name = self.related_query_name()  # i. e. "model"
        field_name = "%s.%s" % (opts.object_name, self.name)  # i. e. "Model.field"

        # Check clashes between accessor or reverse query name of `field`
        # and any other field name -- i.e. accessor for Model.foreign is
        # model_set and it clashes with Target.model_set.
        potential_clashes = rel_opts.fields + rel_opts.many_to_many
        for clash_field in potential_clashes:
            clash_name = "%s.%s" % (rel_opts.object_name, clash_field.name)  # i.e. "Target.model_set"
            if not rel_is_hidden and clash_field.name == rel_name:
                errors.append(
                    checks.Error(
                        "Reverse accessor for '%s' clashes with field name '%s'." % (field_name, clash_name),
                        hint=("Rename field '%s', or add/change a related_name "
                              "argument to the definition for field '%s'.") % (clash_name, field_name),
                        obj=self,
                        id='fields.E302',
                    )
                )

            if clash_field.name == rel_query_name:
                errors.append(
                    checks.Error(
                        "Reverse query name for '%s' clashes with field name '%s'." % (field_name, clash_name),
                        hint=("Rename field '%s', or add/change a related_name "
                              "argument to the definition for field '%s'.") % (clash_name, field_name),
                        obj=self,
                        id='fields.E303',
                    )
                )

        # Check clashes between accessors/reverse query names of `field` and
        # any other field accessor -- i. e. Model.foreign accessor clashes with
        # Model.m2m accessor.
        potential_clashes = (r for r in rel_opts.related_objects if r.field is not self)
        # ... other code
```
### 4 - django/core/cache/backends/db.py:

Start line: 112, End line: 197

```python
class DatabaseCache(BaseDatabaseCache):

    # This class uses cursors provided by the database connection. This means
    # it reads expiration values as aware or naive datetimes, depending on the
    # value of USE_TZ and whether the database supports time zones. The ORM's
    # conversion and adaptation infrastructure is then used to avoid comparing

    pickle_protocol =
    # ... other code

    def _base_set(self, mode, key, value, timeout=DEFAULT_TIMEOUT):
        timeout = self.get_backend_timeout(timeout)
        db = router.db_for_write(self.cache_model_class)
        connection = connections[db]
        quote_name = connection.ops.quote_name
        table = quote_name(self._table)

        with connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM %s" % table)
            num = cursor.fetchone()[0]
            now = timezone.now()
            now = now.replace(microsecond=0)
            if timeout is None:
                exp = datetime.max
            elif settings.USE_TZ:
                exp = datetime.utcfromtimestamp(timeout)
            else:
                exp = datetime.fromtimestamp(timeout)
            exp = exp.replace(microsecond=0)
            if num > self._max_entries:
                self._cull(db, cursor, now)
            pickled = pickle.dumps(value, self.pickle_protocol)
            # The DB column is expecting a string, so make sure the value is a
            # string, not bytes. Refs #19274.
            b64encoded = base64.b64encode(pickled).decode('latin1')
            try:
                # Note: typecasting for datetimes is needed by some 3rd party
                # database backends. All core backends work without typecasting,
                # so be careful about changes here - test suite will NOT pick
                # regressions.
                with transaction.atomic(using=db):
                    cursor.execute(
                        'SELECT %s, %s FROM %s WHERE %s = %%s' % (
                            quote_name('cache_key'),
                            quote_name('expires'),
                            table,
                            quote_name('cache_key'),
                        ),
                        [key]
                    )
                    result = cursor.fetchone()

                    if result:
                        current_expires = result[1]
                        expression = models.Expression(output_field=models.DateTimeField())
                        for converter in (connection.ops.get_db_converters(expression) +
                                          expression.get_db_converters(connection)):
                            current_expires = converter(current_expires, expression, connection)

                    exp = connection.ops.adapt_datetimefield_value(exp)
                    if result and mode == 'touch':
                        cursor.execute(
                            'UPDATE %s SET %s = %%s WHERE %s = %%s' % (
                                table,
                                quote_name('expires'),
                                quote_name('cache_key')
                            ),
                            [exp, key]
                        )
                    elif result and (mode == 'set' or (mode == 'add' and current_expires < now)):
                        cursor.execute(
                            'UPDATE %s SET %s = %%s, %s = %%s WHERE %s = %%s' % (
                                table,
                                quote_name('value'),
                                quote_name('expires'),
                                quote_name('cache_key'),
                            ),
                            [b64encoded, exp, key]
                        )
                    elif mode != 'touch':
                        cursor.execute(
                            'INSERT INTO %s (%s, %s, %s) VALUES (%%s, %%s, %%s)' % (
                                table,
                                quote_name('cache_key'),
                                quote_name('value'),
                                quote_name('expires'),
                            ),
                            [key, b64encoded, exp]
                        )
                    else:
                        return False  # touch failed.
            except DatabaseError:
                # To be threadsafe, updates/inserts are allowed to fail silently
                return False
            else:
                return True
    # ... other code
```
### 5 - django/contrib/contenttypes/fields.py:

Start line: 173, End line: 217

```python
class GenericForeignKey(FieldCacheMixin):

    def get_prefetch_queryset(self, instances, queryset=None):
        if queryset is not None:
            raise ValueError("Custom queryset can't be used for this lookup.")

        # For efficiency, group the instances by content type and then do one
        # query per model
        fk_dict = defaultdict(set)
        # We need one instance for each group in order to get the right db:
        instance_dict = {}
        ct_attname = self.model._meta.get_field(self.ct_field).get_attname()
        for instance in instances:
            # We avoid looking for values if either ct_id or fkey value is None
            ct_id = getattr(instance, ct_attname)
            if ct_id is not None:
                fk_val = getattr(instance, self.fk_field)
                if fk_val is not None:
                    fk_dict[ct_id].add(fk_val)
                    instance_dict[ct_id] = instance

        ret_val = []
        for ct_id, fkeys in fk_dict.items():
            instance = instance_dict[ct_id]
            ct = self.get_content_type(id=ct_id, using=instance._state.db)
            ret_val.extend(ct.get_all_objects_for_this_type(pk__in=fkeys))

        # For doing the join in Python, we have to match both the FK val and the
        # content type, so we use a callable that returns a (fk, class) pair.
        def gfk_key(obj):
            ct_id = getattr(obj, ct_attname)
            if ct_id is None:
                return None
            else:
                model = self.get_content_type(id=ct_id,
                                              using=obj._state.db).model_class()
                return (model._meta.pk.get_prep_value(getattr(obj, self.fk_field)),
                        model)

        return (
            ret_val,
            lambda obj: (obj.pk, obj.__class__),
            gfk_key,
            True,
            self.name,
            True,
        )
```
### 6 - django/db/models/fields/related.py:

Start line: 171, End line: 188

```python
class RelatedField(FieldCacheMixin, Field):

    def _check_referencing_to_swapped_model(self):
        if (self.remote_field.model not in self.opts.apps.get_models() and
                not isinstance(self.remote_field.model, str) and
                self.remote_field.model._meta.swapped):
            model = "%s.%s" % (
                self.remote_field.model._meta.app_label,
                self.remote_field.model._meta.object_name
            )
            return [
                checks.Error(
                    "Field defines a relation with the model '%s', which has "
                    "been swapped out." % model,
                    hint="Update the relation to point at 'settings.%s'." % self.remote_field.model._meta.swappable,
                    obj=self,
                    id='fields.E301',
                )
            ]
        return []
```
### 7 - django/db/models/fields/__init__.py:

Start line: 508, End line: 548

```python
@total_ordering
class Field(RegisterLookupMixin):

    def clone(self):
        """
        Uses deconstruct() to clone a new copy of this Field.
        Will not preserve any class attachments/attribute names.
        """
        name, path, args, kwargs = self.deconstruct()
        return self.__class__(*args, **kwargs)

    def __eq__(self, other):
        # Needed for @total_ordering
        if isinstance(other, Field):
            return self.creation_counter == other.creation_counter
        return NotImplemented

    def __lt__(self, other):
        # This is needed because bisect does not take a comparison function.
        if isinstance(other, Field):
            return self.creation_counter < other.creation_counter
        return NotImplemented

    def __hash__(self):
        return hash(self.creation_counter)

    def __deepcopy__(self, memodict):
        # We don't have to deepcopy very much here, since most things are not
        # intended to be altered after initial creation.
        obj = copy.copy(self)
        if self.remote_field:
            obj.remote_field = copy.copy(self.remote_field)
            if hasattr(self.remote_field, 'field') and self.remote_field.field is self:
                obj.remote_field.field = obj
        memodict[id(self)] = obj
        return obj

    def __copy__(self):
        # We need to avoid hitting __reduce__, so define this
        # slightly weird copy construct.
        obj = Empty()
        obj.__class__ = self.__class__
        obj.__dict__ = self.__dict__.copy()
        return obj
```
### 8 - django/db/models/fields/related.py:

Start line: 1235, End line: 1352

```python
class ManyToManyField(RelatedField):

    def _check_relationship_model(self, from_model=None, **kwargs):
        if hasattr(self.remote_field.through, '_meta'):
            qualified_model_name = "%s.%s" % (
                self.remote_field.through._meta.app_label, self.remote_field.through.__name__)
        else:
            qualified_model_name = self.remote_field.through

        errors = []

        if self.remote_field.through not in self.opts.apps.get_models(include_auto_created=True):
            # The relationship model is not installed.
            errors.append(
                checks.Error(
                    "Field specifies a many-to-many relation through model "
                    "'%s', which has not been installed." % qualified_model_name,
                    obj=self,
                    id='fields.E331',
                )
            )

        else:
            assert from_model is not None, (
                "ManyToManyField with intermediate "
                "tables cannot be checked if you don't pass the model "
                "where the field is attached to."
            )
            # Set some useful local variables
            to_model = resolve_relation(from_model, self.remote_field.model)
            from_model_name = from_model._meta.object_name
            if isinstance(to_model, str):
                to_model_name = to_model
            else:
                to_model_name = to_model._meta.object_name
            relationship_model_name = self.remote_field.through._meta.object_name
            self_referential = from_model == to_model
            # Count foreign keys in intermediate model
            if self_referential:
                seen_self = sum(
                    from_model == getattr(field.remote_field, 'model', None)
                    for field in self.remote_field.through._meta.fields
                )

                if seen_self > 2 and not self.remote_field.through_fields:
                    errors.append(
                        checks.Error(
                            "The model is used as an intermediate model by "
                            "'%s', but it has more than two foreign keys "
                            "to '%s', which is ambiguous. You must specify "
                            "which two foreign keys Django should use via the "
                            "through_fields keyword argument." % (self, from_model_name),
                            hint="Use through_fields to specify which two foreign keys Django should use.",
                            obj=self.remote_field.through,
                            id='fields.E333',
                        )
                    )

            else:
                # Count foreign keys in relationship model
                seen_from = sum(
                    from_model == getattr(field.remote_field, 'model', None)
                    for field in self.remote_field.through._meta.fields
                )
                seen_to = sum(
                    to_model == getattr(field.remote_field, 'model', None)
                    for field in self.remote_field.through._meta.fields
                )

                if seen_from > 1 and not self.remote_field.through_fields:
                    errors.append(
                        checks.Error(
                            ("The model is used as an intermediate model by "
                             "'%s', but it has more than one foreign key "
                             "from '%s', which is ambiguous. You must specify "
                             "which foreign key Django should use via the "
                             "through_fields keyword argument.") % (self, from_model_name),
                            hint=(
                                'If you want to create a recursive relationship, '
                                'use ManyToManyField("%s", through="%s").'
                            ) % (
                                RECURSIVE_RELATIONSHIP_CONSTANT,
                                relationship_model_name,
                            ),
                            obj=self,
                            id='fields.E334',
                        )
                    )

                if seen_to > 1 and not self.remote_field.through_fields:
                    errors.append(
                        checks.Error(
                            "The model is used as an intermediate model by "
                            "'%s', but it has more than one foreign key "
                            "to '%s', which is ambiguous. You must specify "
                            "which foreign key Django should use via the "
                            "through_fields keyword argument." % (self, to_model_name),
                            hint=(
                                'If you want to create a recursive relationship, '
                                'use ManyToManyField("%s", through="%s").'
                            ) % (
                                RECURSIVE_RELATIONSHIP_CONSTANT,
                                relationship_model_name,
                            ),
                            obj=self,
                            id='fields.E335',
                        )
                    )

                if seen_from == 0 or seen_to == 0:
                    errors.append(
                        checks.Error(
                            "The model is used as an intermediate model by "
                            "'%s', but it does not have a foreign key to '%s' or '%s'." % (
                                self, from_model_name, to_model_name
                            ),
                            obj=self.remote_field.through,
                            id='fields.E336',
                        )
                    )
        # ... other code
```
### 9 - django/db/models/fields/related.py:

Start line: 284, End line: 318

```python
class RelatedField(FieldCacheMixin, Field):

    def db_type(self, connection):
        # By default related field will not have a column as it relates to
        # columns from another table.
        return None

    def contribute_to_class(self, cls, name, private_only=False, **kwargs):

        super().contribute_to_class(cls, name, private_only=private_only, **kwargs)

        self.opts = cls._meta

        if not cls._meta.abstract:
            if self.remote_field.related_name:
                related_name = self.remote_field.related_name
            else:
                related_name = self.opts.default_related_name
            if related_name:
                related_name = related_name % {
                    'class': cls.__name__.lower(),
                    'model_name': cls._meta.model_name.lower(),
                    'app_label': cls._meta.app_label.lower()
                }
                self.remote_field.related_name = related_name

            if self.remote_field.related_query_name:
                related_query_name = self.remote_field.related_query_name % {
                    'class': cls.__name__.lower(),
                    'app_label': cls._meta.app_label.lower(),
                }
                self.remote_field.related_query_name = related_query_name

            def resolve_related_class(model, related, field):
                field.remote_field.model = related
                field.do_related_class(related, model)
            lazy_related_operation(resolve_related_class, cls, self.remote_field.model, field=self)
```
### 10 - django/db/models/base.py:

Start line: 404, End line: 505

```python
class Model(metaclass=ModelBase):

    def __init__(self, *args, **kwargs):
        # Alias some things as locals to avoid repeat global lookups
        cls = self.__class__
        opts = self._meta
        _setattr = setattr
        _DEFERRED = DEFERRED
        if opts.abstract:
            raise TypeError('Abstract models cannot be instantiated.')

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
### 17 - django/db/models/base.py:

Start line: 169, End line: 211

```python
class ModelBase(type):
    def __new__(cls, name, bases, attrs, **kwargs):
        # ... other code
        field_names = {f.name for f in new_fields}

        # Basic setup for proxy models.
        if is_proxy:
            base = None
            for parent in [kls for kls in parents if hasattr(kls, '_meta')]:
                if parent._meta.abstract:
                    if parent._meta.fields:
                        raise TypeError(
                            "Abstract base class containing model fields not "
                            "permitted for proxy model '%s'." % name
                        )
                    else:
                        continue
                if base is None:
                    base = parent
                elif parent._meta.concrete_model is not base._meta.concrete_model:
                    raise TypeError("Proxy model '%s' has more than one non-abstract model base class." % name)
            if base is None:
                raise TypeError("Proxy model '%s' has no non-abstract model base class." % name)
            new_class._meta.setup_proxy(base)
            new_class._meta.concrete_model = base._meta.concrete_model
        else:
            new_class._meta.concrete_model = new_class

        # Collect the parent links for multi-table inheritance.
        parent_links = {}
        for base in reversed([new_class] + parents):
            # Conceptually equivalent to `if base is Model`.
            if not hasattr(base, '_meta'):
                continue
            # Skip concrete parent classes.
            if base != new_class and not base._meta.abstract:
                continue
            # Locate OneToOneField instances.
            for field in base._meta.local_fields:
                if isinstance(field, OneToOneField) and field.remote_field.parent_link:
                    related = resolve_relation(new_class, field.remote_field.model)
                    parent_links[make_model_tuple(related)] = field

        # Track fields inherited from base models.
        inherited_attributes = set()
        # Do the appropriate setup for any model parents.
        # ... other code
```
### 18 - django/db/models/base.py:

Start line: 1396, End line: 1451

```python
class Model(metaclass=ModelBase):

    @classmethod
    def _check_field_name_clashes(cls):
        """Forbid field shadowing in multi-table inheritance."""
        errors = []
        used_fields = {}  # name or attname -> field

        # Check that multi-inheritance doesn't cause field name shadowing.
        for parent in cls._meta.get_parent_list():
            for f in parent._meta.local_fields:
                clash = used_fields.get(f.name) or used_fields.get(f.attname) or None
                if clash:
                    errors.append(
                        checks.Error(
                            "The field '%s' from parent model "
                            "'%s' clashes with the field '%s' "
                            "from parent model '%s'." % (
                                clash.name, clash.model._meta,
                                f.name, f.model._meta
                            ),
                            obj=cls,
                            id='models.E005',
                        )
                    )
                used_fields[f.name] = f
                used_fields[f.attname] = f

        # Check that fields defined in the model don't clash with fields from
        # parents, including auto-generated fields like multi-table inheritance
        # child accessors.
        for parent in cls._meta.get_parent_list():
            for f in parent._meta.get_fields():
                if f not in used_fields:
                    used_fields[f.name] = f

        for f in cls._meta.local_fields:
            clash = used_fields.get(f.name) or used_fields.get(f.attname) or None
            # Note that we may detect clash between user-defined non-unique
            # field "id" and automatically added unique field "id", both
            # defined at the same model. This special case is considered in
            # _check_id_field and here we ignore it.
            id_conflict = f.name == "id" and clash and clash.name == "id" and clash.model == cls
            if clash and not id_conflict:
                errors.append(
                    checks.Error(
                        "The field '%s' clashes with the field '%s' "
                        "from model '%s'." % (
                            f.name, clash.name, clash.model._meta
                        ),
                        obj=f,
                        id='models.E006',
                    )
                )
            used_fields[f.name] = f
            used_fields[f.attname] = f

        return errors
```
### 19 - django/db/models/base.py:

Start line: 2033, End line: 2084

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
### 23 - django/db/models/base.py:

Start line: 1075, End line: 1118

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
### 24 - django/db/models/base.py:

Start line: 956, End line: 970

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
### 32 - django/db/models/base.py:

Start line: 1, End line: 50

```python
import copy
import inspect
import warnings
from functools import partialmethod
from itertools import chain

import django
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
from django.db.models.query import F, Q
from django.db.models.signals import (
    class_prepared, post_init, post_save, pre_init, pre_save,
)
from django.db.models.utils import make_model_tuple
from django.utils.encoding import force_str
from django.utils.hashable import make_hashable
from django.utils.text import capfirst, get_text_list
from django.utils.translation import gettext_lazy as _


class Deferred:
    def __repr__(self):
        return '<Deferred field>'

    def __str__(self):
        return '<Deferred field>'


DEFERRED = Deferred()
```
### 40 - django/db/models/base.py:

Start line: 1656, End line: 1704

```python
class Model(metaclass=ModelBase):

    @classmethod
    def _check_local_fields(cls, fields, option):
        from django.db import models

        # In order to avoid hitting the relation tree prematurely, we use our
        # own fields_map instead of using get_field()
        forward_fields_map = {}
        for field in cls._meta._get_fields(reverse=False):
            forward_fields_map[field.name] = field
            if hasattr(field, 'attname'):
                forward_fields_map[field.attname] = field

        errors = []
        for field_name in fields:
            try:
                field = forward_fields_map[field_name]
            except KeyError:
                errors.append(
                    checks.Error(
                        "'%s' refers to the nonexistent field '%s'." % (
                            option, field_name,
                        ),
                        obj=cls,
                        id='models.E012',
                    )
                )
            else:
                if isinstance(field.remote_field, models.ManyToManyRel):
                    errors.append(
                        checks.Error(
                            "'%s' refers to a ManyToManyField '%s', but "
                            "ManyToManyFields are not permitted in '%s'." % (
                                option, field_name, option,
                            ),
                            obj=cls,
                            id='models.E013',
                        )
                    )
                elif field not in cls._meta.local_fields:
                    errors.append(
                        checks.Error(
                            "'%s' refers to field '%s' which is not local to model '%s'."
                            % (option, field_name, cls._meta.object_name),
                            hint="This issue may be caused by multi-table inheritance.",
                            obj=cls,
                            id='models.E016',
                        )
                    )
        return errors
```
### 42 - django/db/models/base.py:

Start line: 1257, End line: 1287

```python
class Model(metaclass=ModelBase):

    @classmethod
    def check(cls, **kwargs):
        errors = [*cls._check_swappable(), *cls._check_model(), *cls._check_managers(**kwargs)]
        if not cls._meta.swapped:
            databases = kwargs.get('databases') or []
            errors += [
                *cls._check_fields(**kwargs),
                *cls._check_m2m_through_same_relationship(),
                *cls._check_long_column_names(databases),
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
                *cls._check_indexes(databases),
                *cls._check_ordering(),
                *cls._check_constraints(databases),
            ]

        return errors
```
### 49 - django/db/models/base.py:

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
### 50 - django/db/models/base.py:

Start line: 972, End line: 985

```python
class Model(metaclass=ModelBase):

    def _get_next_or_previous_in_order(self, is_next):
        cachename = "__%s_order_cache" % is_next
        if not hasattr(self, cachename):
            op = 'gt' if is_next else 'lt'
            order = '_order' if is_next else '-_order'
            order_field = self._meta.order_with_respect_to
            filter_args = order_field.get_filter_kwargs_for_object(self)
            obj = self.__class__._default_manager.filter(**filter_args).filter(**{
                '_order__%s' % op: self.__class__._default_manager.values('_order').filter(**{
                    self._meta.pk.name: self.pk
                })
            }).order_by(order)[:1].get()
            setattr(self, cachename, obj)
        return getattr(self, cachename)
```
### 51 - django/db/models/base.py:

Start line: 1502, End line: 1534

```python
class Model(metaclass=ModelBase):

    @classmethod
    def _check_property_name_related_field_accessor_clashes(cls):
        errors = []
        property_names = cls._meta._property_names
        related_field_accessors = (
            f.get_attname() for f in cls._meta._get_fields(reverse=False)
            if f.is_relation and f.related_model is not None
        )
        for accessor in related_field_accessors:
            if accessor in property_names:
                errors.append(
                    checks.Error(
                        "The property '%s' clashes with a related field "
                        "accessor." % accessor,
                        obj=cls,
                        id='models.E025',
                    )
                )
        return errors

    @classmethod
    def _check_single_primary_key(cls):
        errors = []
        if sum(1 for f in cls._meta.local_fields if f.primary_key) > 1:
            errors.append(
                checks.Error(
                    "The model cannot have more than one field with "
                    "'primary_key=True'.",
                    obj=cls,
                    id='models.E026',
                )
            )
        return errors
```
### 56 - django/db/models/base.py:

Start line: 551, End line: 568

```python
class Model(metaclass=ModelBase):

    def __setstate__(self, state):
        pickled_version = state.get(DJANGO_VERSION_PICKLE_KEY)
        if pickled_version:
            if pickled_version != django.__version__:
                warnings.warn(
                    "Pickled model instance's Django version %s does not "
                    "match the current version %s."
                    % (pickled_version, django.__version__),
                    RuntimeWarning,
                    stacklevel=2,
                )
        else:
            warnings.warn(
                "Pickled model instance's Django version is not specified.",
                RuntimeWarning,
                stacklevel=2,
            )
        self.__dict__.update(state)
```
### 61 - django/db/models/base.py:

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
### 65 - django/db/models/base.py:

Start line: 1347, End line: 1377

```python
class Model(metaclass=ModelBase):

    @classmethod
    def _check_m2m_through_same_relationship(cls):
        """ Check if no relationship model is used by more than one m2m field.
        """

        errors = []
        seen_intermediary_signatures = []

        fields = cls._meta.local_many_to_many

        # Skip when the target model wasn't found.
        fields = (f for f in fields if isinstance(f.remote_field.model, ModelBase))

        # Skip when the relationship model wasn't found.
        fields = (f for f in fields if isinstance(f.remote_field.through, ModelBase))

        for f in fields:
            signature = (f.remote_field.model, cls, f.remote_field.through, f.remote_field.through_fields)
            if signature in seen_intermediary_signatures:
                errors.append(
                    checks.Error(
                        "The model has two identical many-to-many relations "
                        "through the intermediate model '%s'." %
                        f.remote_field.through._meta.label,
                        obj=cls,
                        id='models.E003',
                    )
                )
            else:
                seen_intermediary_signatures.append(signature)
        return errors
```
### 79 - django/db/models/base.py:

Start line: 507, End line: 549

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
        data[DJANGO_VERSION_PICKLE_KEY] = django.__version__
        class_id = self._meta.app_label, self._meta.object_name
        return model_unpickle, (class_id,), data

    def __getstate__(self):
        """Hook to allow choosing the attributes to pickle."""
        return self.__dict__
```
### 92 - django/db/models/base.py:

Start line: 1120, End line: 1147

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
### 99 - django/db/models/base.py:

Start line: 591, End line: 650

```python
class Model(metaclass=ModelBase):

    def refresh_from_db(self, using=None, fields=None):
        """
        Reload field values from the database.

        By default, the reloading happens from the database this instance was
        loaded from, or by the read router if this instance wasn't loaded from
        any database. The using parameter will override the default.

        Fields can be used to specify which fields to reload. The fields
        should be an iterable of field attnames. If fields is None, then
        all non-deferred fields are reloaded.

        When accessing deferred fields of an instance, the deferred loading
        of the field will call this method.
        """
        if fields is None:
            self._prefetched_objects_cache = {}
        else:
            prefetched_objects_cache = getattr(self, '_prefetched_objects_cache', ())
            for field in fields:
                if field in prefetched_objects_cache:
                    del prefetched_objects_cache[field]
                    fields.remove(field)
            if not fields:
                return
            if any(LOOKUP_SEP in f for f in fields):
                raise ValueError(
                    'Found "%s" in fields argument. Relations and transforms '
                    'are not allowed in fields.' % LOOKUP_SEP)

        hints = {'instance': self}
        db_instance_qs = self.__class__._base_manager.db_manager(using, hints=hints).filter(pk=self.pk)

        # Use provided fields, if not set then reload all non-deferred fields.
        deferred_fields = self.get_deferred_fields()
        if fields is not None:
            fields = list(fields)
            db_instance_qs = db_instance_qs.only(*fields)
        elif deferred_fields:
            fields = [f.attname for f in self._meta.concrete_fields
                      if f.attname not in deferred_fields]
            db_instance_qs = db_instance_qs.only(*fields)

        db_instance = db_instance_qs.get()
        non_loaded_fields = db_instance.get_deferred_fields()
        for field in self._meta.concrete_fields:
            if field.attname in non_loaded_fields:
                # This field wasn't refreshed - skip ahead.
                continue
            setattr(self, field.attname, getattr(db_instance, field.attname))
            # Clear cached foreign keys.
            if field.is_relation and field.is_cached(self):
                field.delete_cached_value(self)

        # Clear cached relations.
        for field in self._meta.related_objects:
            if field.is_cached(self):
                field.delete_cached_value(self)

        self._state.db = db_instance._state.db
```
### 107 - django/db/models/base.py:

Start line: 754, End line: 803

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
### 111 - django/db/models/base.py:

Start line: 1706, End line: 1806

```python
class Model(metaclass=ModelBase):

    @classmethod
    def _check_ordering(cls):
        """
        Check "ordering" option -- is it a list of strings and do all fields
        exist?
        """
        if cls._meta._ordering_clash:
            return [
                checks.Error(
                    "'ordering' and 'order_with_respect_to' cannot be used together.",
                    obj=cls,
                    id='models.E021',
                ),
            ]

        if cls._meta.order_with_respect_to or not cls._meta.ordering:
            return []

        if not isinstance(cls._meta.ordering, (list, tuple)):
            return [
                checks.Error(
                    "'ordering' must be a tuple or list (even if you want to order by only one field).",
                    obj=cls,
                    id='models.E014',
                )
            ]

        errors = []
        fields = cls._meta.ordering

        # Skip expressions and '?' fields.
        fields = (f for f in fields if isinstance(f, str) and f != '?')

        # Convert "-field" to "field".
        fields = ((f[1:] if f.startswith('-') else f) for f in fields)

        # Separate related fields and non-related fields.
        _fields = []
        related_fields = []
        for f in fields:
            if LOOKUP_SEP in f:
                related_fields.append(f)
            else:
                _fields.append(f)
        fields = _fields

        # Check related fields.
        for field in related_fields:
            _cls = cls
            fld = None
            for part in field.split(LOOKUP_SEP):
                try:
                    # pk is an alias that won't be found by opts.get_field.
                    if part == 'pk':
                        fld = _cls._meta.pk
                    else:
                        fld = _cls._meta.get_field(part)
                    if fld.is_relation:
                        _cls = fld.get_path_info()[-1].to_opts.model
                    else:
                        _cls = None
                except (FieldDoesNotExist, AttributeError):
                    if fld is None or (
                        fld.get_transform(part) is None and fld.get_lookup(part) is None
                    ):
                        errors.append(
                            checks.Error(
                                "'ordering' refers to the nonexistent field, "
                                "related field, or lookup '%s'." % field,
                                obj=cls,
                                id='models.E015',
                            )
                        )

        # Skip ordering on pk. This is always a valid order_by field
        # but is an alias and therefore won't be found by opts.get_field.
        fields = {f for f in fields if f != 'pk'}

        # Check for invalid or nonexistent fields in ordering.
        invalid_fields = []

        # Any field name that is not present in field_names does not exist.
        # Also, ordering by m2m fields is not allowed.
        opts = cls._meta
        valid_fields = set(chain.from_iterable(
            (f.name, f.attname) if not (f.auto_created and not f.concrete) else (f.field.related_query_name(),)
            for f in chain(opts.fields, opts.related_objects)
        ))

        invalid_fields.extend(fields - valid_fields)

        for invalid_field in invalid_fields:
            errors.append(
                checks.Error(
                    "'ordering' refers to the nonexistent field, related "
                    "field, or lookup '%s'." % invalid_field,
                    obj=cls,
                    id='models.E015',
                )
            )
        return errors
```
### 121 - django/db/models/base.py:

Start line: 1453, End line: 1476

```python
class Model(metaclass=ModelBase):

    @classmethod
    def _check_column_name_clashes(cls):
        # Store a list of column names which have already been used by other fields.
        used_column_names = []
        errors = []

        for f in cls._meta.local_fields:
            _, column_name = f.get_attname_column()

            # Ensure the column name is not already in use.
            if column_name and column_name in used_column_names:
                errors.append(
                    checks.Error(
                        "Field '%s' has column name '%s' that is used by "
                        "another field." % (f.name, column_name),
                        hint="Specify a 'db_column' for the field.",
                        obj=cls,
                        id='models.E007'
                    )
                )
            else:
                used_column_names.append(column_name)

        return errors
```
