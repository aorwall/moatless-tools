# django__django-16511

| **django/django** | `ecafcaf634fcef93f9da8cb12795273dd1c3a576` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 612 |
| **Any found context length** | 353 |
| **Avg pos** | 3.0 |
| **Min pos** | 1 |
| **Max pos** | 2 |
| **Top file pos** | 1 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/db/models/query.py b/django/db/models/query.py
--- a/django/db/models/query.py
+++ b/django/db/models/query.py
@@ -926,25 +926,32 @@ async def aget_or_create(self, defaults=None, **kwargs):
             **kwargs,
         )
 
-    def update_or_create(self, defaults=None, **kwargs):
+    def update_or_create(self, defaults=None, create_defaults=None, **kwargs):
         """
         Look up an object with the given kwargs, updating one with defaults
-        if it exists, otherwise create a new one.
+        if it exists, otherwise create a new one. Optionally, an object can
+        be created with different values than defaults by using
+        create_defaults.
         Return a tuple (object, created), where created is a boolean
         specifying whether an object was created.
         """
-        defaults = defaults or {}
+        if create_defaults is None:
+            update_defaults = create_defaults = defaults or {}
+        else:
+            update_defaults = defaults or {}
         self._for_write = True
         with transaction.atomic(using=self.db):
             # Lock the row so that a concurrent update is blocked until
             # update_or_create() has performed its save.
-            obj, created = self.select_for_update().get_or_create(defaults, **kwargs)
+            obj, created = self.select_for_update().get_or_create(
+                create_defaults, **kwargs
+            )
             if created:
                 return obj, created
-            for k, v in resolve_callables(defaults):
+            for k, v in resolve_callables(update_defaults):
                 setattr(obj, k, v)
 
-            update_fields = set(defaults)
+            update_fields = set(update_defaults)
             concrete_field_names = self.model._meta._non_pk_concrete_field_names
             # update_fields does not support non-concrete fields.
             if concrete_field_names.issuperset(update_fields):
@@ -964,9 +971,10 @@ def update_or_create(self, defaults=None, **kwargs):
                 obj.save(using=self.db)
         return obj, False
 
-    async def aupdate_or_create(self, defaults=None, **kwargs):
+    async def aupdate_or_create(self, defaults=None, create_defaults=None, **kwargs):
         return await sync_to_async(self.update_or_create)(
             defaults=defaults,
+            create_defaults=create_defaults,
             **kwargs,
         )
 

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/db/models/query.py | 929 | 947 | 1 | 1 | 353
| django/db/models/query.py | 967 | 967 | 2 | 1 | 612


## Problem Statement

```
Support create defaults for update_or_create
Description
	
I proposed the idea of extending update_or_create to support specifying a different set of defaults for the create operation on the [forum](​https://forum.djangoproject.com/t/feature-idea-update-or-create-to-allow-different-defaults-for-create-and-update-operations/18300/15). There seems to be consensus it's a positive add to Django.
Adam raised concerns with my proposed approach of adding a create_defaults parameter to the function since this would conflict with any fields on a model named, create_defaults. Jeff did a code search on github for that term and didn't find any matches. I suspect if someone where using a field named create_defaults, it would be a JSON or object type field. Those don't seem like reasonable candidates to be part of a UniqueConstraint, which should be underlying the look-up arguments to update_or_create.
I do like the idea of having a separate parameter for create_defaults, but if we must preserve 100% backwards compatibility, Adam's suggestion of having defaults be set to another object makes the most sense.
My blocking question is, which approach should I take?
From the forum post:
I’ve run into a use-case in which it’d be helpful to have the ability to specify a different set of defaults for the update operation compared to the create operation. While I don’t expect my particular use case to translate, here’s a more generic one.
Given the following Record model:
class Record(models.Model):
	some_id = models.CharField(unique=True)
	created_by = models.ForeignKey(User, ...)
	modified_by = models.ForeignKey(User, null=True, blank=True, ...)
When a record is created, we would want to set created_by, but if it’s being updated, we’d want to set modified_by. This use case can’t be solved by using update_or_create, unless it allows for us to specify a different set of default values.
Record.objects.update_or_create(
	some_id=some_value,
	defaults={"modified_by": user},
	create_defaults={"created_by": user},
)

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 django/db/models/query.py** | 929 | 965| 353 | 353 | 20391 | 
| **-> 2 <-** | **1 django/db/models/query.py** | 967 | 998| 259 | 612 | 20391 | 
| 3 | **1 django/db/models/query.py** | 887 | 927| 298 | 910 | 20391 | 
| 4 | 2 django/db/models/options.py | 175 | 244| 645 | 1555 | 28087 | 
| 5 | 3 django/contrib/auth/management/commands/createsuperuser.py | 250 | 279| 211 | 1766 | 30378 | 
| 6 | 4 django/db/models/base.py | 459 | 572| 957 | 2723 | 49105 | 
| 7 | 5 django/db/migrations/questioner.py | 269 | 288| 195 | 2918 | 51801 | 
| 8 | **5 django/db/models/query.py** | 663 | 715| 424 | 3342 | 51801 | 
| 9 | 6 django/db/backends/base/schema.py | 937 | 1022| 769 | 4111 | 66217 | 
| 10 | 7 django/db/backends/mysql/schema.py | 106 | 120| 144 | 4255 | 68289 | 
| 11 | 8 django/db/backends/postgresql/schema.py | 142 | 255| 920 | 5175 | 71126 | 
| 12 | 9 django/db/models/constraints.py | 234 | 250| 139 | 5314 | 74083 | 
| 13 | 10 django/db/migrations/operations/models.py | 1065 | 1103| 283 | 5597 | 82082 | 
| 14 | 11 django/contrib/contenttypes/fields.py | 756 | 804| 407 | 6004 | 87910 | 
| 15 | 12 django/db/models/fields/__init__.py | 979 | 1011| 256 | 6260 | 106891 | 
| 16 | 12 django/db/migrations/questioner.py | 166 | 187| 188 | 6448 | 106891 | 
| 17 | 13 django/db/models/fields/related_descriptors.py | 788 | 893| 808 | 7256 | 118549 | 
| 18 | 13 django/db/backends/base/schema.py | 1023 | 1110| 822 | 8078 | 118549 | 
| 19 | 13 django/db/models/options.py | 289 | 331| 355 | 8433 | 118549 | 
| 20 | 13 django/db/models/fields/related_descriptors.py | 1238 | 1300| 544 | 8977 | 118549 | 
| 21 | 13 django/db/migrations/operations/models.py | 136 | 306| 968 | 9945 | 118549 | 
| 22 | 13 django/db/backends/postgresql/schema.py | 276 | 310| 277 | 10222 | 118549 | 
| 23 | 14 django/contrib/auth/forms.py | 149 | 190| 294 | 10516 | 121868 | 
| 24 | 14 django/db/backends/base/schema.py | 1111 | 1185| 725 | 11241 | 121868 | 
| 25 | 14 django/db/migrations/operations/models.py | 682 | 706| 231 | 11472 | 121868 | 
| 26 | 14 django/db/backends/postgresql/schema.py | 339 | 375| 212 | 11684 | 121868 | 
| 27 | 15 django/contrib/postgres/constraints.py | 167 | 207| 367 | 12051 | 123575 | 
| 28 | 16 django/contrib/admin/options.py | 1749 | 1851| 780 | 12831 | 142815 | 
| 29 | 17 django/forms/models.py | 380 | 423| 400 | 13231 | 155025 | 
| 30 | 18 django/db/models/fields/mixins.py | 31 | 60| 178 | 13409 | 155373 | 
| 31 | 18 django/db/models/constraints.py | 308 | 373| 517 | 13926 | 155373 | 
| 32 | 18 django/db/models/base.py | 821 | 899| 551 | 14477 | 155373 | 
| 33 | 19 django/db/models/fields/reverse_related.py | 165 | 183| 160 | 14637 | 157905 | 
| 34 | 19 django/db/models/options.py | 87 | 173| 682 | 15319 | 157905 | 
| 35 | 19 django/db/backends/base/schema.py | 373 | 401| 205 | 15524 | 157905 | 
| 36 | 19 django/db/migrations/questioner.py | 291 | 342| 367 | 15891 | 157905 | 
| 37 | 19 django/db/models/base.py | 1567 | 1597| 247 | 16138 | 157905 | 
| 38 | 19 django/db/models/constraints.py | 280 | 306| 221 | 16359 | 157905 | 
| 39 | 19 django/contrib/postgres/constraints.py | 104 | 154| 397 | 16756 | 157905 | 
| 40 | 19 django/db/migrations/questioner.py | 189 | 215| 238 | 16994 | 157905 | 
| 41 | 19 django/db/models/options.py | 333 | 369| 338 | 17332 | 157905 | 
| 42 | 19 django/db/backends/postgresql/schema.py | 257 | 274| 170 | 17502 | 157905 | 
| 43 | 19 django/db/backends/base/schema.py | 403 | 425| 176 | 17678 | 157905 | 
| 44 | 20 django/db/backends/mysql/features.py | 159 | 266| 728 | 18406 | 160315 | 
| 45 | 20 django/db/backends/base/schema.py | 1630 | 1681| 343 | 18749 | 160315 | 
| 46 | 20 django/db/backends/base/schema.py | 777 | 839| 524 | 19273 | 160315 | 
| 47 | 20 django/db/models/options.py | 1 | 58| 353 | 19626 | 160315 | 
| 48 | 20 django/db/migrations/questioner.py | 57 | 87| 255 | 19881 | 160315 | 
| 49 | 20 django/db/migrations/operations/models.py | 727 | 779| 320 | 20201 | 160315 | 
| 50 | 20 django/db/models/base.py | 1430 | 1455| 208 | 20409 | 160315 | 
| 51 | 20 django/db/migrations/questioner.py | 126 | 164| 327 | 20736 | 160315 | 
| 52 | 21 django/db/backends/base/features.py | 6 | 221| 1746 | 22482 | 163529 | 
| 53 | 22 django/db/backends/oracle/creation.py | 159 | 201| 411 | 22893 | 167544 | 
| 54 | 22 django/db/backends/postgresql/schema.py | 312 | 337| 235 | 23128 | 167544 | 
| 55 | 23 django/db/models/__init__.py | 1 | 116| 682 | 23810 | 168226 | 
| 56 | 24 django/contrib/auth/admin.py | 43 | 119| 528 | 24338 | 169997 | 
| 57 | 24 django/db/models/options.py | 496 | 519| 155 | 24493 | 169997 | 
| 58 | 24 django/db/models/constraints.py | 63 | 126| 522 | 25015 | 169997 | 
| 59 | 24 django/db/models/constraints.py | 17 | 45| 232 | 25247 | 169997 | 
| 60 | 25 django/db/backends/postgresql/features.py | 1 | 112| 895 | 26142 | 171042 | 
| 61 | 25 django/db/models/constraints.py | 216 | 232| 138 | 26280 | 171042 | 
| 62 | 25 django/db/backends/postgresql/schema.py | 121 | 140| 234 | 26514 | 171042 | 
| 63 | 26 django/db/backends/postgresql/operations.py | 397 | 416| 150 | 26664 | 174427 | 
| 64 | 26 django/db/models/constraints.py | 129 | 214| 684 | 27348 | 174427 | 
| 65 | 26 django/db/backends/oracle/creation.py | 262 | 300| 403 | 27751 | 174427 | 
| 66 | 26 django/db/models/base.py | 2286 | 2477| 1306 | 29057 | 174427 | 
| 67 | 26 django/db/models/fields/related_descriptors.py | 1353 | 1384| 348 | 29405 | 174427 | 
| 68 | **26 django/db/models/query.py** | 1202 | 1223| 192 | 29597 | 174427 | 
| 69 | 26 django/db/models/fields/__init__.py | 1277 | 1314| 245 | 29842 | 174427 | 
| 70 | 27 django/db/backends/mysql/operations.py | 436 | 465| 274 | 30116 | 178605 | 
| 71 | 27 django/db/models/base.py | 1069 | 1121| 520 | 30636 | 178605 | 
| 72 | 28 django/db/backends/sqlite3/schema.py | 381 | 400| 176 | 30812 | 183320 | 
| 73 | 28 django/contrib/contenttypes/fields.py | 734 | 754| 192 | 31004 | 183320 | 
| 74 | 28 django/forms/models.py | 791 | 882| 787 | 31791 | 183320 | 
| 75 | 28 django/db/backends/base/schema.py | 1507 | 1522| 206 | 31997 | 183320 | 
| 76 | 28 django/db/models/options.py | 630 | 658| 231 | 32228 | 183320 | 
| 77 | 28 django/db/backends/base/schema.py | 544 | 563| 196 | 32424 | 183320 | 
| 78 | 28 django/db/models/base.py | 1301 | 1348| 413 | 32837 | 183320 | 
| 79 | 28 django/db/backends/mysql/schema.py | 1 | 44| 484 | 33321 | 183320 | 
| 80 | 28 django/contrib/contenttypes/fields.py | 654 | 688| 288 | 33609 | 183320 | 
| 81 | 28 django/contrib/auth/management/commands/createsuperuser.py | 90 | 248| 1288 | 34897 | 183320 | 
| 82 | 29 django/db/models/fields/files.py | 224 | 356| 993 | 35890 | 187205 | 
| 83 | 29 django/contrib/contenttypes/fields.py | 690 | 732| 325 | 36215 | 187205 | 
| 84 | 29 django/db/models/fields/__init__.py | 688 | 737| 342 | 36557 | 187205 | 
| 85 | 29 django/db/backends/mysql/schema.py | 210 | 244| 324 | 36881 | 187205 | 
| 86 | 29 django/contrib/admin/options.py | 1 | 114| 776 | 37657 | 187205 | 
| 87 | 29 django/db/models/options.py | 443 | 465| 164 | 37821 | 187205 | 
| 88 | 29 django/db/models/base.py | 1 | 66| 361 | 38182 | 187205 | 
| 89 | **29 django/db/models/query.py** | 640 | 661| 183 | 38365 | 187205 | 
| 90 | 29 django/forms/models.py | 909 | 947| 318 | 38683 | 187205 | 
| 91 | 30 django/db/backends/oracle/features.py | 1 | 87| 744 | 39427 | 188548 | 
| 92 | 30 django/db/backends/base/schema.py | 1590 | 1628| 240 | 39667 | 188548 | 
| 93 | 31 django/db/backends/sqlite3/features.py | 1 | 61| 584 | 40251 | 189892 | 
| 94 | 31 django/db/models/base.py | 1196 | 1236| 310 | 40561 | 189892 | 
| 95 | 31 django/db/models/base.py | 938 | 1026| 690 | 41251 | 189892 | 
| 96 | 31 django/db/models/fields/__init__.py | 1 | 109| 660 | 41911 | 189892 | 
| 97 | 31 django/db/backends/base/schema.py | 1566 | 1588| 199 | 42110 | 189892 | 
| 98 | 31 django/db/migrations/operations/models.py | 608 | 632| 213 | 42323 | 189892 | 
| 99 | 31 django/db/backends/base/schema.py | 659 | 743| 742 | 43065 | 189892 | 


## Patch

```diff
diff --git a/django/db/models/query.py b/django/db/models/query.py
--- a/django/db/models/query.py
+++ b/django/db/models/query.py
@@ -926,25 +926,32 @@ async def aget_or_create(self, defaults=None, **kwargs):
             **kwargs,
         )
 
-    def update_or_create(self, defaults=None, **kwargs):
+    def update_or_create(self, defaults=None, create_defaults=None, **kwargs):
         """
         Look up an object with the given kwargs, updating one with defaults
-        if it exists, otherwise create a new one.
+        if it exists, otherwise create a new one. Optionally, an object can
+        be created with different values than defaults by using
+        create_defaults.
         Return a tuple (object, created), where created is a boolean
         specifying whether an object was created.
         """
-        defaults = defaults or {}
+        if create_defaults is None:
+            update_defaults = create_defaults = defaults or {}
+        else:
+            update_defaults = defaults or {}
         self._for_write = True
         with transaction.atomic(using=self.db):
             # Lock the row so that a concurrent update is blocked until
             # update_or_create() has performed its save.
-            obj, created = self.select_for_update().get_or_create(defaults, **kwargs)
+            obj, created = self.select_for_update().get_or_create(
+                create_defaults, **kwargs
+            )
             if created:
                 return obj, created
-            for k, v in resolve_callables(defaults):
+            for k, v in resolve_callables(update_defaults):
                 setattr(obj, k, v)
 
-            update_fields = set(defaults)
+            update_fields = set(update_defaults)
             concrete_field_names = self.model._meta._non_pk_concrete_field_names
             # update_fields does not support non-concrete fields.
             if concrete_field_names.issuperset(update_fields):
@@ -964,9 +971,10 @@ def update_or_create(self, defaults=None, **kwargs):
                 obj.save(using=self.db)
         return obj, False
 
-    async def aupdate_or_create(self, defaults=None, **kwargs):
+    async def aupdate_or_create(self, defaults=None, create_defaults=None, **kwargs):
         return await sync_to_async(self.update_or_create)(
             defaults=defaults,
+            create_defaults=create_defaults,
             **kwargs,
         )
 

```

## Test Patch

```diff
diff --git a/tests/async/test_async_queryset.py b/tests/async/test_async_queryset.py
--- a/tests/async/test_async_queryset.py
+++ b/tests/async/test_async_queryset.py
@@ -99,10 +99,17 @@ async def test_aupdate_or_create(self):
             id=self.s1.id, defaults={"field": 2}
         )
         self.assertEqual(instance, self.s1)
+        self.assertEqual(instance.field, 2)
         self.assertIs(created, False)
         instance, created = await SimpleModel.objects.aupdate_or_create(field=4)
         self.assertEqual(await SimpleModel.objects.acount(), 4)
         self.assertIs(created, True)
+        instance, created = await SimpleModel.objects.aupdate_or_create(
+            field=5, defaults={"field": 7}, create_defaults={"field": 6}
+        )
+        self.assertEqual(await SimpleModel.objects.acount(), 5)
+        self.assertIs(created, True)
+        self.assertEqual(instance.field, 6)
 
     @skipUnlessDBFeature("has_bulk_insert")
     @async_to_sync
diff --git a/tests/async/test_async_related_managers.py b/tests/async/test_async_related_managers.py
--- a/tests/async/test_async_related_managers.py
+++ b/tests/async/test_async_related_managers.py
@@ -44,12 +44,18 @@ async def test_aupdate_or_create(self):
         self.assertIs(created, True)
         self.assertEqual(await self.mtm1.simples.acount(), 1)
         self.assertEqual(new_simple.field, 2)
-        new_simple, created = await self.mtm1.simples.aupdate_or_create(
+        new_simple1, created = await self.mtm1.simples.aupdate_or_create(
             id=new_simple.id, defaults={"field": 3}
         )
         self.assertIs(created, False)
-        self.assertEqual(await self.mtm1.simples.acount(), 1)
-        self.assertEqual(new_simple.field, 3)
+        self.assertEqual(new_simple1.field, 3)
+
+        new_simple2, created = await self.mtm1.simples.aupdate_or_create(
+            field=4, defaults={"field": 6}, create_defaults={"field": 5}
+        )
+        self.assertIs(created, True)
+        self.assertEqual(new_simple2.field, 5)
+        self.assertEqual(await self.mtm1.simples.acount(), 2)
 
     async def test_aupdate_or_create_reverse(self):
         new_relatedmodel, created = await self.s1.relatedmodel_set.aupdate_or_create()
diff --git a/tests/generic_relations/tests.py b/tests/generic_relations/tests.py
--- a/tests/generic_relations/tests.py
+++ b/tests/generic_relations/tests.py
@@ -59,6 +59,19 @@ def test_generic_update_or_create_when_created(self):
         self.assertTrue(created)
         self.assertEqual(count + 1, self.bacon.tags.count())
 
+    def test_generic_update_or_create_when_created_with_create_defaults(self):
+        count = self.bacon.tags.count()
+        tag, created = self.bacon.tags.update_or_create(
+            # Since, the "stinky" tag doesn't exist create
+            # a "juicy" tag.
+            create_defaults={"tag": "juicy"},
+            defaults={"tag": "uncured"},
+            tag="stinky",
+        )
+        self.assertEqual(tag.tag, "juicy")
+        self.assertIs(created, True)
+        self.assertEqual(count + 1, self.bacon.tags.count())
+
     def test_generic_update_or_create_when_updated(self):
         """
         Should be able to use update_or_create from the generic related manager
@@ -74,6 +87,17 @@ def test_generic_update_or_create_when_updated(self):
         self.assertEqual(count + 1, self.bacon.tags.count())
         self.assertEqual(tag.tag, "juicy")
 
+    def test_generic_update_or_create_when_updated_with_defaults(self):
+        count = self.bacon.tags.count()
+        tag = self.bacon.tags.create(tag="stinky")
+        self.assertEqual(count + 1, self.bacon.tags.count())
+        tag, created = self.bacon.tags.update_or_create(
+            create_defaults={"tag": "uncured"}, defaults={"tag": "juicy"}, id=tag.id
+        )
+        self.assertIs(created, False)
+        self.assertEqual(count + 1, self.bacon.tags.count())
+        self.assertEqual(tag.tag, "juicy")
+
     async def test_generic_async_aupdate_or_create(self):
         tag, created = await self.bacon.tags.aupdate_or_create(
             id=self.fatty.id, defaults={"tag": "orange"}
@@ -86,6 +110,22 @@ async def test_generic_async_aupdate_or_create(self):
         self.assertEqual(await self.bacon.tags.acount(), 3)
         self.assertEqual(tag.tag, "pink")
 
+    async def test_generic_async_aupdate_or_create_with_create_defaults(self):
+        tag, created = await self.bacon.tags.aupdate_or_create(
+            id=self.fatty.id,
+            create_defaults={"tag": "pink"},
+            defaults={"tag": "orange"},
+        )
+        self.assertIs(created, False)
+        self.assertEqual(tag.tag, "orange")
+        self.assertEqual(await self.bacon.tags.acount(), 2)
+        tag, created = await self.bacon.tags.aupdate_or_create(
+            tag="pink", create_defaults={"tag": "brown"}
+        )
+        self.assertIs(created, True)
+        self.assertEqual(await self.bacon.tags.acount(), 3)
+        self.assertEqual(tag.tag, "brown")
+
     def test_generic_get_or_create_when_created(self):
         """
         Should be able to use get_or_create from the generic related manager
@@ -550,6 +590,26 @@ def test_update_or_create_defaults(self):
         self.assertFalse(created)
         self.assertEqual(tag.content_object.id, diamond.id)
 
+    def test_update_or_create_defaults_with_create_defaults(self):
+        # update_or_create() should work with virtual fields (content_object).
+        quartz = Mineral.objects.create(name="Quartz", hardness=7)
+        diamond = Mineral.objects.create(name="Diamond", hardness=7)
+        tag, created = TaggedItem.objects.update_or_create(
+            tag="shiny",
+            create_defaults={"content_object": quartz},
+            defaults={"content_object": diamond},
+        )
+        self.assertIs(created, True)
+        self.assertEqual(tag.content_object.id, quartz.id)
+
+        tag, created = TaggedItem.objects.update_or_create(
+            tag="shiny",
+            create_defaults={"content_object": quartz},
+            defaults={"content_object": diamond},
+        )
+        self.assertIs(created, False)
+        self.assertEqual(tag.content_object.id, diamond.id)
+
     def test_query_content_type(self):
         msg = "Field 'content_object' does not generate an automatic reverse relation"
         with self.assertRaisesMessage(FieldError, msg):
diff --git a/tests/get_or_create/models.py b/tests/get_or_create/models.py
--- a/tests/get_or_create/models.py
+++ b/tests/get_or_create/models.py
@@ -6,6 +6,7 @@ class Person(models.Model):
     last_name = models.CharField(max_length=100)
     birthday = models.DateField()
     defaults = models.TextField()
+    create_defaults = models.TextField()
 
 
 class DefaultPerson(models.Model):
diff --git a/tests/get_or_create/tests.py b/tests/get_or_create/tests.py
--- a/tests/get_or_create/tests.py
+++ b/tests/get_or_create/tests.py
@@ -330,15 +330,24 @@ def test_create(self):
         self.assertEqual(p.birthday, date(1940, 10, 10))
 
     def test_create_twice(self):
-        params = {
-            "first_name": "John",
-            "last_name": "Lennon",
-            "birthday": date(1940, 10, 10),
-        }
-        Person.objects.update_or_create(**params)
-        # If we execute the exact same statement, it won't create a Person.
-        p, created = Person.objects.update_or_create(**params)
-        self.assertFalse(created)
+        p, created = Person.objects.update_or_create(
+            first_name="John",
+            last_name="Lennon",
+            create_defaults={"birthday": date(1940, 10, 10)},
+            defaults={"birthday": date(1950, 2, 2)},
+        )
+        self.assertIs(created, True)
+        self.assertEqual(p.birthday, date(1940, 10, 10))
+        # If we execute the exact same statement, it won't create a Person, but
+        # will update the birthday.
+        p, created = Person.objects.update_or_create(
+            first_name="John",
+            last_name="Lennon",
+            create_defaults={"birthday": date(1940, 10, 10)},
+            defaults={"birthday": date(1950, 2, 2)},
+        )
+        self.assertIs(created, False)
+        self.assertEqual(p.birthday, date(1950, 2, 2))
 
     def test_integrity(self):
         """
@@ -391,8 +400,14 @@ def test_create_with_related_manager(self):
         """
         p = Publisher.objects.create(name="Acme Publishing")
         book, created = p.books.update_or_create(name="The Book of Ed & Fred")
-        self.assertTrue(created)
+        self.assertIs(created, True)
         self.assertEqual(p.books.count(), 1)
+        book, created = p.books.update_or_create(
+            name="Basics of Django", create_defaults={"name": "Advanced Django"}
+        )
+        self.assertIs(created, True)
+        self.assertEqual(book.name, "Advanced Django")
+        self.assertEqual(p.books.count(), 2)
 
     def test_update_with_related_manager(self):
         """
@@ -406,6 +421,14 @@ def test_update_with_related_manager(self):
         book, created = p.books.update_or_create(defaults={"name": name}, id=book.id)
         self.assertFalse(created)
         self.assertEqual(book.name, name)
+        # create_defaults should be ignored.
+        book, created = p.books.update_or_create(
+            create_defaults={"name": "Basics of Django"},
+            defaults={"name": name},
+            id=book.id,
+        )
+        self.assertIs(created, False)
+        self.assertEqual(book.name, name)
         self.assertEqual(p.books.count(), 1)
 
     def test_create_with_many(self):
@@ -418,8 +441,16 @@ def test_create_with_many(self):
         book, created = author.books.update_or_create(
             name="The Book of Ed & Fred", publisher=p
         )
-        self.assertTrue(created)
+        self.assertIs(created, True)
         self.assertEqual(author.books.count(), 1)
+        book, created = author.books.update_or_create(
+            name="Basics of Django",
+            publisher=p,
+            create_defaults={"name": "Advanced Django"},
+        )
+        self.assertIs(created, True)
+        self.assertEqual(book.name, "Advanced Django")
+        self.assertEqual(author.books.count(), 2)
 
     def test_update_with_many(self):
         """
@@ -437,6 +468,14 @@ def test_update_with_many(self):
         )
         self.assertFalse(created)
         self.assertEqual(book.name, name)
+        # create_defaults should be ignored.
+        book, created = author.books.update_or_create(
+            create_defaults={"name": "Basics of Django"},
+            defaults={"name": name},
+            id=book.id,
+        )
+        self.assertIs(created, False)
+        self.assertEqual(book.name, name)
         self.assertEqual(author.books.count(), 1)
 
     def test_defaults_exact(self):
@@ -467,6 +506,34 @@ def test_defaults_exact(self):
         self.assertFalse(created)
         self.assertEqual(obj.defaults, "another testing")
 
+    def test_create_defaults_exact(self):
+        """
+        If you have a field named create_defaults and want to use it as an
+        exact lookup, you need to use 'create_defaults__exact'.
+        """
+        obj, created = Person.objects.update_or_create(
+            first_name="George",
+            last_name="Harrison",
+            create_defaults__exact="testing",
+            create_defaults={
+                "birthday": date(1943, 2, 25),
+                "create_defaults": "testing",
+            },
+        )
+        self.assertIs(created, True)
+        self.assertEqual(obj.create_defaults, "testing")
+        obj, created = Person.objects.update_or_create(
+            first_name="George",
+            last_name="Harrison",
+            create_defaults__exact="testing",
+            create_defaults={
+                "birthday": date(1943, 2, 25),
+                "create_defaults": "another testing",
+            },
+        )
+        self.assertIs(created, False)
+        self.assertEqual(obj.create_defaults, "testing")
+
     def test_create_callable_default(self):
         obj, created = Person.objects.update_or_create(
             first_name="George",
@@ -476,6 +543,16 @@ def test_create_callable_default(self):
         self.assertIs(created, True)
         self.assertEqual(obj.birthday, date(1943, 2, 25))
 
+    def test_create_callable_create_defaults(self):
+        obj, created = Person.objects.update_or_create(
+            first_name="George",
+            last_name="Harrison",
+            defaults={},
+            create_defaults={"birthday": lambda: date(1943, 2, 25)},
+        )
+        self.assertIs(created, True)
+        self.assertEqual(obj.birthday, date(1943, 2, 25))
+
     def test_update_callable_default(self):
         Person.objects.update_or_create(
             first_name="George",
@@ -694,6 +771,12 @@ def test_update_or_create_with_invalid_defaults(self):
         with self.assertRaisesMessage(FieldError, self.msg):
             Thing.objects.update_or_create(name="a", defaults={"nonexistent": "b"})
 
+    def test_update_or_create_with_invalid_create_defaults(self):
+        with self.assertRaisesMessage(FieldError, self.msg):
+            Thing.objects.update_or_create(
+                name="a", create_defaults={"nonexistent": "b"}
+            )
+
     def test_update_or_create_with_invalid_kwargs(self):
         with self.assertRaisesMessage(FieldError, self.bad_field_msg):
             Thing.objects.update_or_create(name="a", nonexistent="b")

```


## Code snippets

### 1 - django/db/models/query.py:

Start line: 929, End line: 965

```python
class QuerySet(AltersData):

    def update_or_create(self, defaults=None, **kwargs):
        """
        Look up an object with the given kwargs, updating one with defaults
        if it exists, otherwise create a new one.
        Return a tuple (object, created), where created is a boolean
        specifying whether an object was created.
        """
        defaults = defaults or {}
        self._for_write = True
        with transaction.atomic(using=self.db):
            # Lock the row so that a concurrent update is blocked until
            # update_or_create() has performed its save.
            obj, created = self.select_for_update().get_or_create(defaults, **kwargs)
            if created:
                return obj, created
            for k, v in resolve_callables(defaults):
                setattr(obj, k, v)

            update_fields = set(defaults)
            concrete_field_names = self.model._meta._non_pk_concrete_field_names
            # update_fields does not support non-concrete fields.
            if concrete_field_names.issuperset(update_fields):
                # Add fields which are set on pre_save(), e.g. auto_now fields.
                # This is to maintain backward compatibility as these fields
                # are not updated unless explicitly specified in the
                # update_fields list.
                for field in self.model._meta.local_concrete_fields:
                    if not (
                        field.primary_key or field.__class__.pre_save is Field.pre_save
                    ):
                        update_fields.add(field.name)
                        if field.name != field.attname:
                            update_fields.add(field.attname)
                obj.save(using=self.db, update_fields=update_fields)
            else:
                obj.save(using=self.db)
        return obj, False
```
### 2 - django/db/models/query.py:

Start line: 967, End line: 998

```python
class QuerySet(AltersData):

    async def aupdate_or_create(self, defaults=None, **kwargs):
        return await sync_to_async(self.update_or_create)(
            defaults=defaults,
            **kwargs,
        )

    def _extract_model_params(self, defaults, **kwargs):
        """
        Prepare `params` for creating a model instance based on the given
        kwargs; for use by get_or_create().
        """
        defaults = defaults or {}
        params = {k: v for k, v in kwargs.items() if LOOKUP_SEP not in k}
        params.update(defaults)
        property_names = self.model._meta._property_names
        invalid_params = []
        for param in params:
            try:
                self.model._meta.get_field(param)
            except exceptions.FieldDoesNotExist:
                # It's okay to use a model's property if it has a setter.
                if not (param in property_names and getattr(self.model, param).fset):
                    invalid_params.append(param)
        if invalid_params:
            raise exceptions.FieldError(
                "Invalid field name(s) for model %s: '%s'."
                % (
                    self.model._meta.object_name,
                    "', '".join(sorted(invalid_params)),
                )
            )
        return params
```
### 3 - django/db/models/query.py:

Start line: 887, End line: 927

```python
class QuerySet(AltersData):

    bulk_update.alters_data = True

    async def abulk_update(self, objs, fields, batch_size=None):
        return await sync_to_async(self.bulk_update)(
            objs=objs,
            fields=fields,
            batch_size=batch_size,
        )

    abulk_update.alters_data = True

    def get_or_create(self, defaults=None, **kwargs):
        """
        Look up an object with the given kwargs, creating one if necessary.
        Return a tuple of (object, created), where created is a boolean
        specifying whether an object was created.
        """
        # The get() needs to be targeted at the write database in order
        # to avoid potential transaction consistency problems.
        self._for_write = True
        try:
            return self.get(**kwargs), False
        except self.model.DoesNotExist:
            params = self._extract_model_params(defaults, **kwargs)
            # Try to create an object using passed params.
            try:
                with transaction.atomic(using=self.db):
                    params = dict(resolve_callables(params))
                    return self.create(**params), True
            except IntegrityError:
                try:
                    return self.get(**kwargs), False
                except self.model.DoesNotExist:
                    pass
                raise

    async def aget_or_create(self, defaults=None, **kwargs):
        return await sync_to_async(self.get_or_create)(
            defaults=defaults,
            **kwargs,
        )
```
### 4 - django/db/models/options.py:

Start line: 175, End line: 244

```python
class Options:

    def contribute_to_class(self, cls, name):
        from django.db import connection
        from django.db.backends.utils import truncate_name

        cls._meta = self
        self.model = cls
        # First, construct the default values for these options.
        self.object_name = cls.__name__
        self.model_name = self.object_name.lower()
        self.verbose_name = camel_case_to_spaces(self.object_name)

        # Store the original user-defined values for each option,
        # for use when serializing the model definition
        self.original_attrs = {}

        # Next, apply any overridden values from 'class Meta'.
        if self.meta:
            meta_attrs = self.meta.__dict__.copy()
            for name in self.meta.__dict__:
                # Ignore any private attributes that Django doesn't care about.
                # NOTE: We can't modify a dictionary's contents while looping
                # over it, so we loop over the *original* dictionary instead.
                if name.startswith("_"):
                    del meta_attrs[name]
            for attr_name in DEFAULT_NAMES:
                if attr_name in meta_attrs:
                    setattr(self, attr_name, meta_attrs.pop(attr_name))
                    self.original_attrs[attr_name] = getattr(self, attr_name)
                elif hasattr(self.meta, attr_name):
                    setattr(self, attr_name, getattr(self.meta, attr_name))
                    self.original_attrs[attr_name] = getattr(self, attr_name)

            self.unique_together = normalize_together(self.unique_together)
            self.index_together = normalize_together(self.index_together)
            if self.index_together:
                warnings.warn(
                    f"'index_together' is deprecated. Use 'Meta.indexes' in "
                    f"{self.label!r} instead.",
                    RemovedInDjango51Warning,
                )
            # App label/class name interpolation for names of constraints and
            # indexes.
            if not getattr(cls._meta, "abstract", False):
                for attr_name in {"constraints", "indexes"}:
                    objs = getattr(self, attr_name, [])
                    setattr(self, attr_name, self._format_names_with_class(cls, objs))

            # verbose_name_plural is a special case because it uses a 's'
            # by default.
            if self.verbose_name_plural is None:
                self.verbose_name_plural = format_lazy("{}s", self.verbose_name)

            # order_with_respect_and ordering are mutually exclusive.
            self._ordering_clash = bool(self.ordering and self.order_with_respect_to)

            # Any leftover attributes must be invalid.
            if meta_attrs != {}:
                raise TypeError(
                    "'class Meta' got invalid attribute(s): %s" % ",".join(meta_attrs)
                )
        else:
            self.verbose_name_plural = format_lazy("{}s", self.verbose_name)
        del self.meta

        # If the db_table wasn't provided, use the app_label + model_name.
        if not self.db_table:
            self.db_table = "%s_%s" % (self.app_label, self.model_name)
            self.db_table = truncate_name(
                self.db_table, connection.ops.max_name_length()
            )
```
### 5 - django/contrib/auth/management/commands/createsuperuser.py:

Start line: 250, End line: 279

```python
class Command(BaseCommand):

    def get_input_data(self, field, message, default=None):
        """
        Override this method if you want to customize data inputs or
        validation exceptions.
        """
        raw_value = input(message)
        if default and raw_value == "":
            raw_value = default
        try:
            val = field.clean(raw_value, None)
        except exceptions.ValidationError as e:
            self.stderr.write("Error: %s" % "; ".join(e.messages))
            val = None

        return val

    def _get_input_message(self, field, default=None):
        return "%s%s%s: " % (
            capfirst(field.verbose_name),
            " (leave blank to use '%s')" % default if default else "",
            " (%s.%s)"
            % (
                field.remote_field.model._meta.object_name,
                field.m2m_target_field_name()
                if field.many_to_many
                else field.remote_field.field_name,
            )
            if field.remote_field
            else "",
        )
```
### 6 - django/db/models/base.py:

Start line: 459, End line: 572

```python
class Model(AltersData, metaclass=ModelBase):
    def __init__(self, *args, **kwargs):
        # Alias some things as locals to avoid repeat global lookups
        cls = self.__class__
        opts = self._meta
        _setattr = setattr
        _DEFERRED = DEFERRED
        if opts.abstract:
            raise TypeError("Abstract models cannot be instantiated.")

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
                if kwargs.pop(field.name, NOT_PROVIDED) is not NOT_PROVIDED:
                    raise TypeError(
                        f"{cls.__qualname__}() got both positional and "
                        f"keyword arguments for field '{field.name}'."
                    )

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
            unexpected = ()
            for prop, value in kwargs.items():
                # Any remaining kwargs must correspond to properties or virtual
                # fields.
                if prop in property_names:
                    if value is not _DEFERRED:
                        _setattr(self, prop, value)
                else:
                    try:
                        opts.get_field(prop)
                    except FieldDoesNotExist:
                        unexpected += (prop,)
                    else:
                        if value is not _DEFERRED:
                            _setattr(self, prop, value)
            if unexpected:
                unexpected_names = ", ".join(repr(n) for n in unexpected)
                raise TypeError(
                    f"{cls.__name__}() got unexpected keyword arguments: "
                    f"{unexpected_names}"
                )
        super().__init__()
        post_init.send(sender=cls, instance=self)
```
### 7 - django/db/migrations/questioner.py:

Start line: 269, End line: 288

```python
class InteractiveMigrationQuestioner(MigrationQuestioner):

    def ask_unique_callable_default_addition(self, field_name, model_name):
        """Adding a unique field with a callable default."""
        if not self.dry_run:
            version = get_docs_version()
            choice = self._choice_input(
                f"Callable default on unique field {model_name}.{field_name} "
                f"will not generate unique values upon migrating.\n"
                f"Please choose how to proceed:\n",
                [
                    f"Continue making this migration as the first step in "
                    f"writing a manual migration to generate unique values "
                    f"described here: "
                    f"https://docs.djangoproject.com/en/{version}/howto/"
                    f"writing-migrations/#migrations-that-add-unique-fields.",
                    "Quit and edit field options in models.py.",
                ],
            )
            if choice == 2:
                sys.exit(3)
        return None
```
### 8 - django/db/models/query.py:

Start line: 663, End line: 715

```python
class QuerySet(AltersData):

    def _check_bulk_create_options(
        self, ignore_conflicts, update_conflicts, update_fields, unique_fields
    ):
        if ignore_conflicts and update_conflicts:
            raise ValueError(
                "ignore_conflicts and update_conflicts are mutually exclusive."
            )
        db_features = connections[self.db].features
        if ignore_conflicts:
            if not db_features.supports_ignore_conflicts:
                raise NotSupportedError(
                    "This database backend does not support ignoring conflicts."
                )
            return OnConflict.IGNORE
        elif update_conflicts:
            if not db_features.supports_update_conflicts:
                raise NotSupportedError(
                    "This database backend does not support updating conflicts."
                )
            if not update_fields:
                raise ValueError(
                    "Fields that will be updated when a row insertion fails "
                    "on conflicts must be provided."
                )
            if unique_fields and not db_features.supports_update_conflicts_with_target:
                raise NotSupportedError(
                    "This database backend does not support updating "
                    "conflicts with specifying unique fields that can trigger "
                    "the upsert."
                )
            if not unique_fields and db_features.supports_update_conflicts_with_target:
                raise ValueError(
                    "Unique fields that can trigger the upsert must be provided."
                )
            # Updating primary keys and non-concrete fields is forbidden.
            if any(not f.concrete or f.many_to_many for f in update_fields):
                raise ValueError(
                    "bulk_create() can only be used with concrete fields in "
                    "update_fields."
                )
            if any(f.primary_key for f in update_fields):
                raise ValueError(
                    "bulk_create() cannot be used with primary keys in "
                    "update_fields."
                )
            if unique_fields:
                if any(not f.concrete or f.many_to_many for f in unique_fields):
                    raise ValueError(
                        "bulk_create() can only be used with concrete fields "
                        "in unique_fields."
                    )
            return OnConflict.UPDATE
        return None
```
### 9 - django/db/backends/base/schema.py:

Start line: 937, End line: 1022

```python
class BaseDatabaseSchemaEditor:

    def _alter_field(
        self,
        model,
        old_field,
        new_field,
        old_type,
        new_type,
        old_db_params,
        new_db_params,
        strict=False,
    ):
        # ... other code
        if (
            old_field.db_index
            and not old_field.unique
            and (not new_field.db_index or new_field.unique)
        ):
            # Find the index for this field
            meta_index_names = {index.name for index in model._meta.indexes}
            # Retrieve only BTREE indexes since this is what's created with
            # db_index=True.
            index_names = self._constraint_names(
                model,
                [old_field.column],
                index=True,
                type_=Index.suffix,
                exclude=meta_index_names,
            )
            for index_name in index_names:
                # The only way to check if an index was created with
                # db_index=True or with Index(['field'], name='foo')
                # is to look at its name (refs #28053).
                self.execute(self._delete_index_sql(model, index_name))
        # Change check constraints?
        if old_db_params["check"] != new_db_params["check"] and old_db_params["check"]:
            meta_constraint_names = {
                constraint.name for constraint in model._meta.constraints
            }
            constraint_names = self._constraint_names(
                model,
                [old_field.column],
                check=True,
                exclude=meta_constraint_names,
            )
            if strict and len(constraint_names) != 1:
                raise ValueError(
                    "Found wrong number (%s) of check constraints for %s.%s"
                    % (
                        len(constraint_names),
                        model._meta.db_table,
                        old_field.column,
                    )
                )
            for constraint_name in constraint_names:
                self.execute(self._delete_check_sql(model, constraint_name))
        # Have they renamed the column?
        if old_field.column != new_field.column:
            self.execute(
                self._rename_field_sql(
                    model._meta.db_table, old_field, new_field, new_type
                )
            )
            # Rename all references to the renamed column.
            for sql in self.deferred_sql:
                if isinstance(sql, Statement):
                    sql.rename_column_references(
                        model._meta.db_table, old_field.column, new_field.column
                    )
        # Next, start accumulating actions to do
        actions = []
        null_actions = []
        post_actions = []
        # Type suffix change? (e.g. auto increment).
        old_type_suffix = old_field.db_type_suffix(connection=self.connection)
        new_type_suffix = new_field.db_type_suffix(connection=self.connection)
        # Type, collation, or comment change?
        if (
            old_type != new_type
            or old_type_suffix != new_type_suffix
            or old_collation != new_collation
            or (
                self.connection.features.supports_comments
                and old_field.db_comment != new_field.db_comment
            )
        ):
            fragment, other_actions = self._alter_column_type_sql(
                model, old_field, new_field, new_type, old_collation, new_collation
            )
            actions.append(fragment)
            post_actions.extend(other_actions)
        # When changing a column NULL constraint to NOT NULL with a given
        # default value, we need to perform 4 steps:
        #  1. Add a default for new incoming writes
        #  2. Update existing NULL rows with new default
        #  3. Replace NULL constraint with NOT NULL
        #  4. Drop the default again.
        # Default change?
        needs_database_default = False
        # ... other code
```
### 10 - django/db/backends/mysql/schema.py:

Start line: 106, End line: 120

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
### 68 - django/db/models/query.py:

Start line: 1202, End line: 1223

```python
class QuerySet(AltersData):

    update.alters_data = True

    async def aupdate(self, **kwargs):
        return await sync_to_async(self.update)(**kwargs)

    aupdate.alters_data = True

    def _update(self, values):
        """
        A version of update() that accepts field objects instead of field names.
        Used primarily for model saving and not intended for use by general
        code (it requires too much poking around at model internals to be
        useful at that level).
        """
        if self.query.is_sliced:
            raise TypeError("Cannot update a query once a slice has been taken.")
        query = self.query.chain(sql.UpdateQuery)
        query.add_update_fields(values)
        # Clear any annotations so that they won't be present in subqueries.
        query.annotations = {}
        self._result_cache = None
        return query.get_compiler(self.db).execute_sql(CURSOR)
```
### 89 - django/db/models/query.py:

Start line: 640, End line: 661

```python
class QuerySet(AltersData):

    async def aget(self, *args, **kwargs):
        return await sync_to_async(self.get)(*args, **kwargs)

    def create(self, **kwargs):
        """
        Create a new object with the given kwargs, saving it to the database
        and returning the created object.
        """
        obj = self.model(**kwargs)
        self._for_write = True
        obj.save(force_insert=True, using=self.db)
        return obj

    async def acreate(self, **kwargs):
        return await sync_to_async(self.create)(**kwargs)

    def _prepare_for_bulk_create(self, objs):
        for obj in objs:
            if obj.pk is None:
                # Populate new PK values.
                obj.pk = obj._meta.pk.get_pk_value_on_save(obj)
            obj._prepare_related_fields_for_save(operation_name="bulk_create")
```
