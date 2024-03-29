# django__django-15969

| **django/django** | `081871bc20cc8b28481109b8dcadc321e177e6be` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | 569 |
| **Avg pos** | 29.0 |
| **Min pos** | 1 |
| **Max pos** | 16 |
| **Top file pos** | 1 |
| **Missing snippets** | 9 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/db/models/deletion.py b/django/db/models/deletion.py
--- a/django/db/models/deletion.py
+++ b/django/db/models/deletion.py
@@ -1,9 +1,9 @@
 from collections import Counter, defaultdict
-from functools import partial
+from functools import partial, reduce
 from itertools import chain
-from operator import attrgetter
+from operator import attrgetter, or_
 
-from django.db import IntegrityError, connections, transaction
+from django.db import IntegrityError, connections, models, transaction
 from django.db.models import query_utils, signals, sql
 
 
@@ -61,6 +61,7 @@ def set_on_delete(collector, field, sub_objs, using):
             collector.add_field_update(field, value, sub_objs)
 
     set_on_delete.deconstruct = lambda: ("django.db.models.SET", (value,), {})
+    set_on_delete.lazy_sub_objs = True
     return set_on_delete
 
 
@@ -68,10 +69,16 @@ def SET_NULL(collector, field, sub_objs, using):
     collector.add_field_update(field, None, sub_objs)
 
 
+SET_NULL.lazy_sub_objs = True
+
+
 def SET_DEFAULT(collector, field, sub_objs, using):
     collector.add_field_update(field, field.get_default(), sub_objs)
 
 
+SET_DEFAULT.lazy_sub_objs = True
+
+
 def DO_NOTHING(collector, field, sub_objs, using):
     pass
 
@@ -93,8 +100,8 @@ def __init__(self, using, origin=None):
         self.origin = origin
         # Initially, {model: {instances}}, later values become lists.
         self.data = defaultdict(set)
-        # {model: {(field, value): {instances}}}
-        self.field_updates = defaultdict(partial(defaultdict, set))
+        # {(field, value): [instances, …]}
+        self.field_updates = defaultdict(list)
         # {model: {field: {instances}}}
         self.restricted_objects = defaultdict(partial(defaultdict, set))
         # fast_deletes is a list of queryset-likes that can be deleted without
@@ -145,10 +152,7 @@ def add_field_update(self, field, value, objs):
         Schedule a field update. 'objs' must be a homogeneous iterable
         collection of model instances (e.g. a QuerySet).
         """
-        if not objs:
-            return
-        model = objs[0].__class__
-        self.field_updates[model][field, value].update(objs)
+        self.field_updates[field, value].append(objs)
 
     def add_restricted_objects(self, field, objs):
         if objs:
@@ -312,7 +316,8 @@ def collect(
             if keep_parents and related.model in parents:
                 continue
             field = related.field
-            if field.remote_field.on_delete == DO_NOTHING:
+            on_delete = field.remote_field.on_delete
+            if on_delete == DO_NOTHING:
                 continue
             related_model = related.related_model
             if self.can_fast_delete(related_model, from_field=field):
@@ -340,9 +345,9 @@ def collect(
                         )
                     )
                     sub_objs = sub_objs.only(*tuple(referenced_fields))
-                if sub_objs:
+                if getattr(on_delete, "lazy_sub_objs", False) or sub_objs:
                     try:
-                        field.remote_field.on_delete(self, field, sub_objs, self.using)
+                        on_delete(self, field, sub_objs, self.using)
                     except ProtectedError as error:
                         key = "'%s.%s'" % (field.model.__name__, field.name)
                         protected_objects[key] += error.protected_objects
@@ -469,11 +474,25 @@ def delete(self):
                     deleted_counter[qs.model._meta.label] += count
 
             # update fields
-            for model, instances_for_fieldvalues in self.field_updates.items():
-                for (field, value), instances in instances_for_fieldvalues.items():
+            for (field, value), instances_list in self.field_updates.items():
+                updates = []
+                objs = []
+                for instances in instances_list:
+                    if (
+                        isinstance(instances, models.QuerySet)
+                        and instances._result_cache is None
+                    ):
+                        updates.append(instances)
+                    else:
+                        objs.extend(instances)
+                if updates:
+                    combined_updates = reduce(or_, updates)
+                    combined_updates.update(**{field.name: value})
+                if objs:
+                    model = objs[0].__class__
                     query = sql.UpdateQuery(model)
                     query.update_batch(
-                        [obj.pk for obj in instances], {field.name: value}, self.using
+                        list({obj.pk for obj in objs}), {field.name: value}, self.using
                     )
 
             # reverse instance collections
@@ -497,11 +516,6 @@ def delete(self):
                             origin=self.origin,
                         )
 
-        # update collected instances
-        for instances_for_fieldvalues in self.field_updates.values():
-            for (field, value), instances in instances_for_fieldvalues.items():
-                for obj in instances:
-                    setattr(obj, field.attname, value)
         for model, instances in self.data.items():
             for instance in instances:
                 setattr(instance, model._meta.pk.attname, None)

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/db/models/deletion.py | 2 | 6 | 1 | 1 | 569
| django/db/models/deletion.py | 64 | 64 | 1 | 1 | 569
| django/db/models/deletion.py | 71 | 71 | 1 | 1 | 569
| django/db/models/deletion.py | 96 | 97 | 16 | 1 | 6005
| django/db/models/deletion.py | 148 | 151 | - | 1 | -
| django/db/models/deletion.py | 315 | 315 | 2 | 1 | 1185
| django/db/models/deletion.py | 343 | 345 | 2 | 1 | 1185
| django/db/models/deletion.py | 472 | 476 | 3 | 1 | 1786
| django/db/models/deletion.py | 500 | 504 | 3 | 1 | 1786


## Problem Statement

```
Performance issues with `on_delete=models.SET_NULL` on large tables
Description
	
Hello,
I have the following models configuration:
Parent model
Child model, with a parent_id foreign key to a Parent model, set with on_delete=models.SET_NULL
Each Parent can have a lot of children, in my case roughly 30k.
I'm starting to encounter performance issues that make my jobs timeout, because the SQL queries simply timeout.
I've enabled query logging, and noticed something weird (that is certainly that way on purpose, but I don't understand why).
# Select the parent
SELECT * FROM "parent" WHERE "parent"."id" = 'parent123';
# Select all children
SELECT * FROM "children" WHERE "children"."parent_id" IN ('parent123');
# Update all children `parent_id` column to `NULL`
UPDATE "children" SET "parent_id" = NULL WHERE "children"."id" IN ('child1', 'child2', 'child3', ..., 'child30000');
# Finally delete the parent
DELETE FROM "parent" WHERE "parent"."id" IN ('parent123');
I would have expected the update condition to simply be WHERE "children"."parent_id" = 'parent123', but for some reason it isn't.
In the meantime, I'll switch to on_delete=models.CASCADE, which in my case does the trick, but I was curious about the reason why this happens in the first place.
Thanks in advance

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 django/db/models/deletion.py** | 1 | 86| 569 | 569 | 3932 | 
| **-> 2 <-** | **1 django/db/models/deletion.py** | 310 | 370| 616 | 1185 | 3932 | 
| **-> 3 <-** | **1 django/db/models/deletion.py** | 431 | 509| 601 | 1786 | 3932 | 
| 4 | 2 django/db/models/query.py | 1148 | 1170| 156 | 1942 | 24382 | 
| 5 | 3 django/db/models/sql/subqueries.py | 1 | 45| 311 | 2253 | 25610 | 
| 6 | 4 django/db/migrations/autodetector.py | 811 | 906| 712 | 2965 | 39071 | 
| 7 | 4 django/db/models/query.py | 684 | 743| 479 | 3444 | 39071 | 
| 8 | 5 django/db/models/fields/related.py | 992 | 1016| 176 | 3620 | 53680 | 
| 9 | 5 django/db/models/query.py | 1112 | 1146| 306 | 3926 | 53680 | 
| 10 | 6 django/db/models/base.py | 1128 | 1147| 188 | 4114 | 72231 | 
| 11 | **6 django/db/models/deletion.py** | 372 | 396| 273 | 4387 | 72231 | 
| 12 | 6 django/db/models/base.py | 1074 | 1126| 516 | 4903 | 72231 | 
| 13 | 6 django/db/models/query.py | 795 | 853| 499 | 5402 | 72231 | 
| 14 | 7 django/db/backends/sqlite3/schema.py | 362 | 378| 132 | 5534 | 76814 | 
| 15 | 8 django/db/models/sql/compiler.py | 1764 | 1796| 254 | 5788 | 92967 | 
| **-> 16 <-** | **8 django/db/models/deletion.py** | 89 | 109| 217 | 6005 | 92967 | 
| 17 | 8 django/db/models/sql/compiler.py | 1907 | 1968| 588 | 6593 | 92967 | 
| 18 | 9 django/db/backends/base/schema.py | 563 | 595| 268 | 6861 | 106828 | 
| 19 | 10 django/db/backends/oracle/schema.py | 52 | 73| 146 | 7007 | 109170 | 
| 20 | 10 django/db/models/base.py | 906 | 941| 318 | 7325 | 109170 | 
| 21 | 10 django/db/models/sql/subqueries.py | 48 | 78| 212 | 7537 | 109170 | 
| 22 | 10 django/db/backends/base/schema.py | 1718 | 1755| 303 | 7840 | 109170 | 
| 23 | 10 django/db/models/query.py | 2232 | 2366| 1129 | 8969 | 109170 | 
| 24 | 10 django/db/models/base.py | 838 | 904| 475 | 9444 | 109170 | 
| 25 | 10 django/db/models/base.py | 1298 | 1345| 409 | 9853 | 109170 | 
| 26 | 11 django/db/backends/base/features.py | 6 | 221| 1745 | 11598 | 112239 | 
| 27 | 12 django/db/models/sql/query.py | 1981 | 2048| 667 | 12265 | 135410 | 
| 28 | 12 django/db/backends/base/schema.py | 39 | 72| 214 | 12479 | 135410 | 
| 29 | **12 django/db/models/deletion.py** | 183 | 225| 337 | 12816 | 135410 | 
| 30 | 12 django/db/backends/base/schema.py | 521 | 540| 196 | 13012 | 135410 | 
| 31 | 13 django/db/migrations/questioner.py | 189 | 215| 238 | 13250 | 138106 | 
| 32 | 13 django/db/backends/base/schema.py | 1614 | 1651| 243 | 13493 | 138106 | 
| 33 | 13 django/db/backends/base/schema.py | 456 | 475| 151 | 13644 | 138106 | 
| 34 | 13 django/db/backends/base/schema.py | 888 | 972| 778 | 14422 | 138106 | 
| 35 | 14 django/db/backends/mysql/schema.py | 138 | 156| 209 | 14631 | 139729 | 
| 36 | 14 django/db/backends/mysql/schema.py | 1 | 42| 456 | 15087 | 139729 | 
| 37 | 14 django/db/models/base.py | 943 | 1031| 686 | 15773 | 139729 | 
| 38 | 14 django/db/models/fields/related.py | 1450 | 1579| 984 | 16757 | 139729 | 
| 39 | 14 django/db/backends/base/schema.py | 797 | 887| 773 | 17530 | 139729 | 
| 40 | 15 django/db/backends/oracle/operations.py | 407 | 453| 385 | 17915 | 145949 | 
| 41 | 15 django/db/models/base.py | 2256 | 2447| 1302 | 19217 | 145949 | 
| 42 | 15 django/db/models/sql/query.py | 2050 | 2079| 259 | 19476 | 145949 | 
| 43 | 16 django/db/models/sql/where.py | 253 | 261| 115 | 19591 | 148460 | 
| 44 | 16 django/db/models/sql/query.py | 2081 | 2130| 332 | 19923 | 148460 | 
| 45 | 17 django/db/models/constraints.py | 251 | 265| 117 | 20040 | 151388 | 
| 46 | 18 django/db/backends/sqlite3/operations.py | 216 | 241| 218 | 20258 | 154875 | 
| 47 | 18 django/db/backends/base/schema.py | 1390 | 1406| 153 | 20411 | 154875 | 
| 48 | 19 django/db/models/options.py | 1 | 57| 347 | 20758 | 162463 | 
| 49 | 19 django/db/models/fields/related.py | 604 | 670| 497 | 21255 | 162463 | 
| 50 | 20 django/db/backends/mysql/compiler.py | 25 | 48| 252 | 21507 | 163099 | 
| 51 | 20 django/db/models/query.py | 2203 | 2231| 246 | 21753 | 163099 | 
| 52 | 20 django/db/models/base.py | 2157 | 2238| 588 | 22341 | 163099 | 
| 53 | 20 django/db/models/fields/related.py | 1418 | 1448| 172 | 22513 | 163099 | 
| 54 | 21 django/db/backends/postgresql/schema.py | 267 | 292| 235 | 22748 | 165546 | 
| 55 | 22 django/db/backends/oracle/creation.py | 159 | 201| 411 | 23159 | 169561 | 
| 56 | 22 django/db/backends/oracle/operations.py | 455 | 518| 532 | 23691 | 169561 | 
| 57 | 22 django/db/models/sql/query.py | 2391 | 2440| 398 | 24089 | 169561 | 
| 58 | 22 django/db/backends/postgresql/schema.py | 219 | 265| 408 | 24497 | 169561 | 
| 59 | 22 django/db/models/sql/query.py | 2639 | 2695| 823 | 25320 | 169561 | 
| 60 | 22 django/db/models/sql/query.py | 1077 | 1108| 307 | 25627 | 169561 | 
| 61 | 23 django/db/models/__init__.py | 1 | 116| 682 | 26309 | 170243 | 
| 62 | 24 django/db/backends/mysql/features.py | 86 | 160| 597 | 26906 | 172614 | 
| 63 | 24 django/db/models/sql/query.py | 1656 | 1754| 846 | 27752 | 172614 | 
| 64 | 24 django/db/models/sql/compiler.py | 1798 | 1817| 212 | 27964 | 172614 | 
| 65 | 24 django/db/migrations/autodetector.py | 908 | 981| 623 | 28587 | 172614 | 
| 66 | 24 django/db/models/sql/query.py | 976 | 1003| 272 | 28859 | 172614 | 
| 67 | 25 django/db/models/fields/related_descriptors.py | 789 | 864| 592 | 29451 | 183856 | 
| 68 | 26 django/db/backends/postgresql/operations.py | 212 | 244| 314 | 29765 | 186845 | 
| 69 | 27 django/db/backends/mysql/operations.py | 436 | 465| 274 | 30039 | 191023 | 
| 70 | 27 django/db/models/sql/query.py | 721 | 803| 845 | 30884 | 191023 | 
| 71 | 27 django/db/models/base.py | 2450 | 2502| 341 | 31225 | 191023 | 
| 72 | 27 django/db/models/sql/compiler.py | 171 | 230| 543 | 31768 | 191023 | 
| 73 | 27 django/db/models/sql/query.py | 2236 | 2280| 355 | 32123 | 191023 | 
| 74 | 27 django/db/models/query.py | 2604 | 2627| 201 | 32324 | 191023 | 
| 75 | 28 django/core/cache/backends/db.py | 206 | 233| 280 | 32604 | 193167 | 
| 76 | 28 django/db/backends/sqlite3/operations.py | 1 | 42| 314 | 32918 | 193167 | 
| 77 | 28 django/db/migrations/questioner.py | 57 | 87| 255 | 33173 | 193167 | 
| 78 | 28 django/db/models/query.py | 745 | 794| 518 | 33691 | 193167 | 
| 79 | 28 django/db/backends/postgresql/operations.py | 338 | 357| 150 | 33841 | 193167 | 
| 80 | 29 django/contrib/gis/db/backends/spatialite/schema.py | 91 | 110| 155 | 33996 | 194561 | 
| 81 | 29 django/db/models/sql/compiler.py | 1167 | 1272| 871 | 34867 | 194561 | 
| 82 | 29 django/db/models/query.py | 2142 | 2200| 487 | 35354 | 194561 | 
| 83 | 29 django/db/models/base.py | 1347 | 1376| 290 | 35644 | 194561 | 
| 84 | 29 django/db/models/options.py | 286 | 328| 355 | 35999 | 194561 | 
| 85 | 29 django/db/models/sql/query.py | 389 | 441| 497 | 36496 | 194561 | 
| 86 | 29 django/db/backends/sqlite3/schema.py | 265 | 360| 807 | 37303 | 194561 | 
| 87 | 29 django/db/backends/base/schema.py | 973 | 1060| 824 | 38127 | 194561 | 
| 88 | 29 django/db/models/query.py | 1695 | 1710| 149 | 38276 | 194561 | 


### Hint

```
You are right that is an opportunity for an optimization when SET, SET_DEFAULT, or SET_NULL is used but I wonder if it's worth doing given db_on_delete support (see #21961) make things even better for this use case. In the meantime, I'll switch to on_delete=models.CASCADE, which in my case does the trick, but I was curious about the reason why this happens in the first place. This is likely the case because the collector is able to take ​the fast delete route and avoid object fetching entirely. If you want to give a shot at a patch you'll want to have a look at Collector.collect and have it skip collection entirely when dealing with SET and friends likely by adding a branch that turns them into fast updates. ​One branch that worries me is the post-deletion assignment of values to in-memory instances but I can't understand why this is even necessary given all the instances that are collected for field updates are never make their way out of the collector so I would expect it to be entirely unnecessary at least all delete and delete_regress tests pass if I entirely remove it django/db/models/deletion.py diff --git a/django/db/models/deletion.py b/django/db/models/deletion.py index 2cb3c88444..2eb8e95281 100644 a b def delete(self): 496496 using=self.using, 497497 origin=self.origin, 498498 ) 499 500 # update collected instances 501 for instances_for_fieldvalues in self.field_updates.values(): 502 for (field, value), instances in instances_for_fieldvalues.items(): 503 for obj in instances: 504 setattr(obj, field.attname, value) 505499 for model, instances in self.data.items(): 506500 for instance in instances: 507501 setattr(instance, model._meta.pk.attname, None) You'll want to make sure to avoid breaking the ​admin's collector subclass used to display deletion confirmation pages but from a quick look it doesn't seem to care about field updates. Tentatively accepting but I think we should revisit when #21961 lands.
```

## Patch

```diff
diff --git a/django/db/models/deletion.py b/django/db/models/deletion.py
--- a/django/db/models/deletion.py
+++ b/django/db/models/deletion.py
@@ -1,9 +1,9 @@
 from collections import Counter, defaultdict
-from functools import partial
+from functools import partial, reduce
 from itertools import chain
-from operator import attrgetter
+from operator import attrgetter, or_
 
-from django.db import IntegrityError, connections, transaction
+from django.db import IntegrityError, connections, models, transaction
 from django.db.models import query_utils, signals, sql
 
 
@@ -61,6 +61,7 @@ def set_on_delete(collector, field, sub_objs, using):
             collector.add_field_update(field, value, sub_objs)
 
     set_on_delete.deconstruct = lambda: ("django.db.models.SET", (value,), {})
+    set_on_delete.lazy_sub_objs = True
     return set_on_delete
 
 
@@ -68,10 +69,16 @@ def SET_NULL(collector, field, sub_objs, using):
     collector.add_field_update(field, None, sub_objs)
 
 
+SET_NULL.lazy_sub_objs = True
+
+
 def SET_DEFAULT(collector, field, sub_objs, using):
     collector.add_field_update(field, field.get_default(), sub_objs)
 
 
+SET_DEFAULT.lazy_sub_objs = True
+
+
 def DO_NOTHING(collector, field, sub_objs, using):
     pass
 
@@ -93,8 +100,8 @@ def __init__(self, using, origin=None):
         self.origin = origin
         # Initially, {model: {instances}}, later values become lists.
         self.data = defaultdict(set)
-        # {model: {(field, value): {instances}}}
-        self.field_updates = defaultdict(partial(defaultdict, set))
+        # {(field, value): [instances, …]}
+        self.field_updates = defaultdict(list)
         # {model: {field: {instances}}}
         self.restricted_objects = defaultdict(partial(defaultdict, set))
         # fast_deletes is a list of queryset-likes that can be deleted without
@@ -145,10 +152,7 @@ def add_field_update(self, field, value, objs):
         Schedule a field update. 'objs' must be a homogeneous iterable
         collection of model instances (e.g. a QuerySet).
         """
-        if not objs:
-            return
-        model = objs[0].__class__
-        self.field_updates[model][field, value].update(objs)
+        self.field_updates[field, value].append(objs)
 
     def add_restricted_objects(self, field, objs):
         if objs:
@@ -312,7 +316,8 @@ def collect(
             if keep_parents and related.model in parents:
                 continue
             field = related.field
-            if field.remote_field.on_delete == DO_NOTHING:
+            on_delete = field.remote_field.on_delete
+            if on_delete == DO_NOTHING:
                 continue
             related_model = related.related_model
             if self.can_fast_delete(related_model, from_field=field):
@@ -340,9 +345,9 @@ def collect(
                         )
                     )
                     sub_objs = sub_objs.only(*tuple(referenced_fields))
-                if sub_objs:
+                if getattr(on_delete, "lazy_sub_objs", False) or sub_objs:
                     try:
-                        field.remote_field.on_delete(self, field, sub_objs, self.using)
+                        on_delete(self, field, sub_objs, self.using)
                     except ProtectedError as error:
                         key = "'%s.%s'" % (field.model.__name__, field.name)
                         protected_objects[key] += error.protected_objects
@@ -469,11 +474,25 @@ def delete(self):
                     deleted_counter[qs.model._meta.label] += count
 
             # update fields
-            for model, instances_for_fieldvalues in self.field_updates.items():
-                for (field, value), instances in instances_for_fieldvalues.items():
+            for (field, value), instances_list in self.field_updates.items():
+                updates = []
+                objs = []
+                for instances in instances_list:
+                    if (
+                        isinstance(instances, models.QuerySet)
+                        and instances._result_cache is None
+                    ):
+                        updates.append(instances)
+                    else:
+                        objs.extend(instances)
+                if updates:
+                    combined_updates = reduce(or_, updates)
+                    combined_updates.update(**{field.name: value})
+                if objs:
+                    model = objs[0].__class__
                     query = sql.UpdateQuery(model)
                     query.update_batch(
-                        [obj.pk for obj in instances], {field.name: value}, self.using
+                        list({obj.pk for obj in objs}), {field.name: value}, self.using
                     )
 
             # reverse instance collections
@@ -497,11 +516,6 @@ def delete(self):
                             origin=self.origin,
                         )
 
-        # update collected instances
-        for instances_for_fieldvalues in self.field_updates.values():
-            for (field, value), instances in instances_for_fieldvalues.items():
-                for obj in instances:
-                    setattr(obj, field.attname, value)
         for model, instances in self.data.items():
             for instance in instances:
                 setattr(instance, model._meta.pk.attname, None)

```

## Test Patch

```diff
diff --git a/tests/delete_regress/models.py b/tests/delete_regress/models.py
--- a/tests/delete_regress/models.py
+++ b/tests/delete_regress/models.py
@@ -90,6 +90,12 @@ class Location(models.Model):
 class Item(models.Model):
     version = models.ForeignKey(Version, models.CASCADE)
     location = models.ForeignKey(Location, models.SET_NULL, blank=True, null=True)
+    location_value = models.ForeignKey(
+        Location, models.SET(42), default=1, db_constraint=False, related_name="+"
+    )
+    location_default = models.ForeignKey(
+        Location, models.SET_DEFAULT, default=1, db_constraint=False, related_name="+"
+    )
 
 
 # Models for #16128
diff --git a/tests/delete_regress/tests.py b/tests/delete_regress/tests.py
--- a/tests/delete_regress/tests.py
+++ b/tests/delete_regress/tests.py
@@ -399,3 +399,19 @@ def test_disallowed_delete_distinct(self):
             Book.objects.distinct().delete()
         with self.assertRaisesMessage(TypeError, msg):
             Book.objects.distinct("id").delete()
+
+
+class SetQueryCountTests(TestCase):
+    def test_set_querycount(self):
+        policy = Policy.objects.create()
+        version = Version.objects.create(policy=policy)
+        location = Location.objects.create(version=version)
+        Item.objects.create(
+            version=version,
+            location=location,
+            location_default=location,
+            location_value=location,
+        )
+        # 3 UPDATEs for SET of item values and one for DELETE locations.
+        with self.assertNumQueries(4):
+            location.delete()

```


## Code snippets

### 1 - django/db/models/deletion.py:

Start line: 1, End line: 86

```python
from collections import Counter, defaultdict
from functools import partial
from itertools import chain
from operator import attrgetter

from django.db import IntegrityError, connections, transaction
from django.db.models import query_utils, signals, sql


class ProtectedError(IntegrityError):
    def __init__(self, msg, protected_objects):
        self.protected_objects = protected_objects
        super().__init__(msg, protected_objects)


class RestrictedError(IntegrityError):
    def __init__(self, msg, restricted_objects):
        self.restricted_objects = restricted_objects
        super().__init__(msg, restricted_objects)


def CASCADE(collector, field, sub_objs, using):
    collector.collect(
        sub_objs,
        source=field.remote_field.model,
        source_attr=field.name,
        nullable=field.null,
        fail_on_restricted=False,
    )
    if field.null and not connections[using].features.can_defer_constraint_checks:
        collector.add_field_update(field, None, sub_objs)


def PROTECT(collector, field, sub_objs, using):
    raise ProtectedError(
        "Cannot delete some instances of model '%s' because they are "
        "referenced through a protected foreign key: '%s.%s'"
        % (
            field.remote_field.model.__name__,
            sub_objs[0].__class__.__name__,
            field.name,
        ),
        sub_objs,
    )


def RESTRICT(collector, field, sub_objs, using):
    collector.add_restricted_objects(field, sub_objs)
    collector.add_dependency(field.remote_field.model, field.model)


def SET(value):
    if callable(value):

        def set_on_delete(collector, field, sub_objs, using):
            collector.add_field_update(field, value(), sub_objs)

    else:

        def set_on_delete(collector, field, sub_objs, using):
            collector.add_field_update(field, value, sub_objs)

    set_on_delete.deconstruct = lambda: ("django.db.models.SET", (value,), {})
    return set_on_delete


def SET_NULL(collector, field, sub_objs, using):
    collector.add_field_update(field, None, sub_objs)


def SET_DEFAULT(collector, field, sub_objs, using):
    collector.add_field_update(field, field.get_default(), sub_objs)


def DO_NOTHING(collector, field, sub_objs, using):
    pass


def get_candidate_relations_to_delete(opts):
    # The candidate relations are the ones that come from N-1 and 1-1 relations.
    # N-N  (i.e., many-to-many) relations aren't candidates for deletion.
    return (
        f
        for f in opts.get_fields(include_hidden=True)
        if f.auto_created and not f.concrete and (f.one_to_one or f.one_to_many)
    )
```
### 2 - django/db/models/deletion.py:

Start line: 310, End line: 370

```python
class Collector:

    def collect(
        self,
        objs,
        source=None,
        nullable=False,
        collect_related=True,
        source_attr=None,
        reverse_dependency=False,
        keep_parents=False,
        fail_on_restricted=True,
    ):
        # ... other code
        for related in get_candidate_relations_to_delete(model._meta):
            # Preserve parent reverse relationships if keep_parents=True.
            if keep_parents and related.model in parents:
                continue
            field = related.field
            if field.remote_field.on_delete == DO_NOTHING:
                continue
            related_model = related.related_model
            if self.can_fast_delete(related_model, from_field=field):
                model_fast_deletes[related_model].append(field)
                continue
            batches = self.get_del_batches(new_objs, [field])
            for batch in batches:
                sub_objs = self.related_objects(related_model, [field], batch)
                # Non-referenced fields can be deferred if no signal receivers
                # are connected for the related model as they'll never be
                # exposed to the user. Skip field deferring when some
                # relationships are select_related as interactions between both
                # features are hard to get right. This should only happen in
                # the rare cases where .related_objects is overridden anyway.
                if not (
                    sub_objs.query.select_related
                    or self._has_signal_listeners(related_model)
                ):
                    referenced_fields = set(
                        chain.from_iterable(
                            (rf.attname for rf in rel.field.foreign_related_fields)
                            for rel in get_candidate_relations_to_delete(
                                related_model._meta
                            )
                        )
                    )
                    sub_objs = sub_objs.only(*tuple(referenced_fields))
                if sub_objs:
                    try:
                        field.remote_field.on_delete(self, field, sub_objs, self.using)
                    except ProtectedError as error:
                        key = "'%s.%s'" % (field.model.__name__, field.name)
                        protected_objects[key] += error.protected_objects
        if protected_objects:
            raise ProtectedError(
                "Cannot delete some instances of model %r because they are "
                "referenced through protected foreign keys: %s."
                % (
                    model.__name__,
                    ", ".join(protected_objects),
                ),
                set(chain.from_iterable(protected_objects.values())),
            )
        for related_model, related_fields in model_fast_deletes.items():
            batches = self.get_del_batches(new_objs, related_fields)
            for batch in batches:
                sub_objs = self.related_objects(related_model, related_fields, batch)
                self.fast_deletes.append(sub_objs)
        for field in model._meta.private_fields:
            if hasattr(field, "bulk_related_objects"):
                # It's something like generic foreign key.
                sub_objs = field.bulk_related_objects(new_objs, self.using)
                self.collect(
                    sub_objs, source=model, nullable=True, fail_on_restricted=False
                )
        # ... other code
```
### 3 - django/db/models/deletion.py:

Start line: 431, End line: 509

```python
class Collector:

    def delete(self):
        # sort instance collections
        for model, instances in self.data.items():
            self.data[model] = sorted(instances, key=attrgetter("pk"))

        # if possible, bring the models in an order suitable for databases that
        # don't support transactions or cannot defer constraint checks until the
        # end of a transaction.
        self.sort()
        # number of objects deleted for each model label
        deleted_counter = Counter()

        # Optimize for the case with a single obj and no dependencies
        if len(self.data) == 1 and len(instances) == 1:
            instance = list(instances)[0]
            if self.can_fast_delete(instance):
                with transaction.mark_for_rollback_on_error(self.using):
                    count = sql.DeleteQuery(model).delete_batch(
                        [instance.pk], self.using
                    )
                setattr(instance, model._meta.pk.attname, None)
                return count, {model._meta.label: count}

        with transaction.atomic(using=self.using, savepoint=False):
            # send pre_delete signals
            for model, obj in self.instances_with_model():
                if not model._meta.auto_created:
                    signals.pre_delete.send(
                        sender=model,
                        instance=obj,
                        using=self.using,
                        origin=self.origin,
                    )

            # fast deletes
            for qs in self.fast_deletes:
                count = qs._raw_delete(using=self.using)
                if count:
                    deleted_counter[qs.model._meta.label] += count

            # update fields
            for model, instances_for_fieldvalues in self.field_updates.items():
                for (field, value), instances in instances_for_fieldvalues.items():
                    query = sql.UpdateQuery(model)
                    query.update_batch(
                        [obj.pk for obj in instances], {field.name: value}, self.using
                    )

            # reverse instance collections
            for instances in self.data.values():
                instances.reverse()

            # delete instances
            for model, instances in self.data.items():
                query = sql.DeleteQuery(model)
                pk_list = [obj.pk for obj in instances]
                count = query.delete_batch(pk_list, self.using)
                if count:
                    deleted_counter[model._meta.label] += count

                if not model._meta.auto_created:
                    for obj in instances:
                        signals.post_delete.send(
                            sender=model,
                            instance=obj,
                            using=self.using,
                            origin=self.origin,
                        )

        # update collected instances
        for instances_for_fieldvalues in self.field_updates.values():
            for (field, value), instances in instances_for_fieldvalues.items():
                for obj in instances:
                    setattr(obj, field.attname, value)
        for model, instances in self.data.items():
            for instance in instances:
                setattr(instance, model._meta.pk.attname, None)
        return sum(deleted_counter.values()), dict(deleted_counter)
```
### 4 - django/db/models/query.py:

Start line: 1148, End line: 1170

```python
class QuerySet:

    delete.alters_data = True
    delete.queryset_only = True

    async def adelete(self):
        return await sync_to_async(self.delete)()

    adelete.alters_data = True
    adelete.queryset_only = True

    def _raw_delete(self, using):
        """
        Delete objects found from the given queryset in single direct SQL
        query. No signals are sent and there is no protection for cascades.
        """
        query = self.query.clone()
        query.__class__ = sql.DeleteQuery
        cursor = query.get_compiler(using).execute_sql(CURSOR)
        if cursor:
            with cursor:
                return cursor.rowcount
        return 0

    _raw_delete.alters_data = True
```
### 5 - django/db/models/sql/subqueries.py:

Start line: 1, End line: 45

```python
"""
Query subclasses which provide extra functionality beyond simple data retrieval.
"""

from django.core.exceptions import FieldError
from django.db.models.sql.constants import CURSOR, GET_ITERATOR_CHUNK_SIZE, NO_RESULTS
from django.db.models.sql.query import Query

__all__ = ["DeleteQuery", "UpdateQuery", "InsertQuery", "AggregateQuery"]


class DeleteQuery(Query):
    """A DELETE SQL query."""

    compiler = "SQLDeleteCompiler"

    def do_query(self, table, where, using):
        self.alias_map = {table: self.alias_map[table]}
        self.where = where
        cursor = self.get_compiler(using).execute_sql(CURSOR)
        if cursor:
            with cursor:
                return cursor.rowcount
        return 0

    def delete_batch(self, pk_list, using):
        """
        Set up and execute delete queries for all the objects in pk_list.

        More than one physical query may be executed if there are a
        lot of values in pk_list.
        """
        # number of objects deleted
        num_deleted = 0
        field = self.get_meta().pk
        for offset in range(0, len(pk_list), GET_ITERATOR_CHUNK_SIZE):
            self.clear_where()
            self.add_filter(
                f"{field.attname}__in",
                pk_list[offset : offset + GET_ITERATOR_CHUNK_SIZE],
            )
            num_deleted += self.do_query(
                self.get_meta().db_table, self.where, using=using
            )
        return num_deleted
```
### 6 - django/db/migrations/autodetector.py:

Start line: 811, End line: 906

```python
class MigrationAutodetector:

    def generate_deleted_models(self):
        """
        Find all deleted models (managed and unmanaged) and make delete
        operations for them as well as separate operations to delete any
        foreign key or M2M relationships (these are optimized later, if
        possible).

        Also bring forward removal of any model options that refer to
        collections of fields - the inverse of generate_created_models().
        """
        new_keys = self.new_model_keys | self.new_unmanaged_keys
        deleted_models = self.old_model_keys - new_keys
        deleted_unmanaged_models = self.old_unmanaged_keys - new_keys
        all_deleted_models = chain(
            sorted(deleted_models), sorted(deleted_unmanaged_models)
        )
        for app_label, model_name in all_deleted_models:
            model_state = self.from_state.models[app_label, model_name]
            # Gather related fields
            related_fields = {}
            for field_name, field in model_state.fields.items():
                if field.remote_field:
                    if field.remote_field.model:
                        related_fields[field_name] = field
                    if getattr(field.remote_field, "through", None):
                        related_fields[field_name] = field
            # Generate option removal first
            unique_together = model_state.options.pop("unique_together", None)
            # RemovedInDjango51Warning.
            index_together = model_state.options.pop("index_together", None)
            if unique_together:
                self.add_operation(
                    app_label,
                    operations.AlterUniqueTogether(
                        name=model_name,
                        unique_together=None,
                    ),
                )
            # RemovedInDjango51Warning.
            if index_together:
                self.add_operation(
                    app_label,
                    operations.AlterIndexTogether(
                        name=model_name,
                        index_together=None,
                    ),
                )
            # Then remove each related field
            for name in sorted(related_fields):
                self.add_operation(
                    app_label,
                    operations.RemoveField(
                        model_name=model_name,
                        name=name,
                    ),
                )
            # Finally, remove the model.
            # This depends on both the removal/alteration of all incoming fields
            # and the removal of all its own related fields, and if it's
            # a through model the field that references it.
            dependencies = []
            relations = self.from_state.relations
            for (
                related_object_app_label,
                object_name,
            ), relation_related_fields in relations[app_label, model_name].items():
                for field_name, field in relation_related_fields.items():
                    dependencies.append(
                        (related_object_app_label, object_name, field_name, False),
                    )
                    if not field.many_to_many:
                        dependencies.append(
                            (
                                related_object_app_label,
                                object_name,
                                field_name,
                                "alter",
                            ),
                        )

            for name in sorted(related_fields):
                dependencies.append((app_label, model_name, name, False))
            # We're referenced in another field's through=
            through_user = self.through_users.get((app_label, model_state.name_lower))
            if through_user:
                dependencies.append(
                    (through_user[0], through_user[1], through_user[2], False)
                )
            # Finally, make the operation, deduping any dependencies
            self.add_operation(
                app_label,
                operations.DeleteModel(
                    name=model_state.name,
                ),
                dependencies=list(set(dependencies)),
            )
```
### 7 - django/db/models/query.py:

Start line: 684, End line: 743

```python
class QuerySet:

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
            update_fields = [self.model._meta.get_field(name) for name in update_fields]
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
                # Primary key is allowed in unique_fields.
                unique_fields = [
                    self.model._meta.get_field(name)
                    for name in unique_fields
                    if name != "pk"
                ]
                if any(not f.concrete or f.many_to_many for f in unique_fields):
                    raise ValueError(
                        "bulk_create() can only be used with concrete fields "
                        "in unique_fields."
                    )
            return OnConflict.UPDATE
        return None
```
### 8 - django/db/models/fields/related.py:

Start line: 992, End line: 1016

```python
class ForeignKey(ForeignObject):

    def _check_on_delete(self):
        on_delete = getattr(self.remote_field, "on_delete", None)
        if on_delete == SET_NULL and not self.null:
            return [
                checks.Error(
                    "Field specifies on_delete=SET_NULL, but cannot be null.",
                    hint=(
                        "Set null=True argument on the field, or change the on_delete "
                        "rule."
                    ),
                    obj=self,
                    id="fields.E320",
                )
            ]
        elif on_delete == SET_DEFAULT and not self.has_default():
            return [
                checks.Error(
                    "Field specifies on_delete=SET_DEFAULT, but has no default value.",
                    hint="Set a default value, or change the on_delete rule.",
                    obj=self,
                    id="fields.E321",
                )
            ]
        else:
            return []
```
### 9 - django/db/models/query.py:

Start line: 1112, End line: 1146

```python
class QuerySet:

    async def ain_bulk(self, id_list=None, *, field_name="pk"):
        return await sync_to_async(self.in_bulk)(
            id_list=id_list,
            field_name=field_name,
        )

    def delete(self):
        """Delete the records in the current QuerySet."""
        self._not_support_combined_queries("delete")
        if self.query.is_sliced:
            raise TypeError("Cannot use 'limit' or 'offset' with delete().")
        if self.query.distinct or self.query.distinct_fields:
            raise TypeError("Cannot call delete() after .distinct().")
        if self._fields is not None:
            raise TypeError("Cannot call delete() after .values() or .values_list()")

        del_query = self._chain()

        # The delete is actually 2 queries - one to find related objects,
        # and one to delete. Make sure that the discovery of related
        # objects is performed on the same database as the deletion.
        del_query._for_write = True

        # Disable non-supported fields.
        del_query.query.select_for_update = False
        del_query.query.select_related = False
        del_query.query.clear_ordering(force=True)

        collector = Collector(using=del_query.db, origin=self)
        collector.collect(del_query)
        deleted, _rows_count = collector.delete()

        # Clear the result cache, in case this QuerySet gets reused.
        self._result_cache = None
        return deleted, _rows_count
```
### 10 - django/db/models/base.py:

Start line: 1128, End line: 1147

```python
class Model(metaclass=ModelBase):

    def delete(self, using=None, keep_parents=False):
        if self.pk is None:
            raise ValueError(
                "%s object can't be deleted because its %s attribute is set "
                "to None." % (self._meta.object_name, self._meta.pk.attname)
            )
        using = using or router.db_for_write(self.__class__, instance=self)
        collector = Collector(using=using, origin=self)
        collector.collect([self], keep_parents=keep_parents)
        return collector.delete()

    delete.alters_data = True

    def _get_FIELD_display(self, field):
        value = getattr(self, field.attname)
        choices_dict = dict(make_hashable(field.flatchoices))
        # force_str() to coerce lazy strings.
        return force_str(
            choices_dict.get(make_hashable(value), value), strings_only=True
        )
```
### 11 - django/db/models/deletion.py:

Start line: 372, End line: 396

```python
class Collector:

    def collect(
        self,
        objs,
        source=None,
        nullable=False,
        collect_related=True,
        source_attr=None,
        reverse_dependency=False,
        keep_parents=False,
        fail_on_restricted=True,
    ):
        # ... other code

        if fail_on_restricted:
            # Raise an error if collected restricted objects (RESTRICT) aren't
            # candidates for deletion also collected via CASCADE.
            for related_model, instances in self.data.items():
                self.clear_restricted_objects_from_set(related_model, instances)
            for qs in self.fast_deletes:
                self.clear_restricted_objects_from_queryset(qs.model, qs)
            if self.restricted_objects.values():
                restricted_objects = defaultdict(list)
                for related_model, fields in self.restricted_objects.items():
                    for field, objs in fields.items():
                        if objs:
                            key = "'%s.%s'" % (related_model.__name__, field.name)
                            restricted_objects[key] += objs
                if restricted_objects:
                    raise RestrictedError(
                        "Cannot delete some instances of model %r because "
                        "they are referenced through restricted foreign keys: "
                        "%s."
                        % (
                            model.__name__,
                            ", ".join(restricted_objects),
                        ),
                        set(chain.from_iterable(restricted_objects.values())),
                    )
```
### 16 - django/db/models/deletion.py:

Start line: 89, End line: 109

```python
class Collector:
    def __init__(self, using, origin=None):
        self.using = using
        # A Model or QuerySet object.
        self.origin = origin
        # Initially, {model: {instances}}, later values become lists.
        self.data = defaultdict(set)
        # {model: {(field, value): {instances}}}
        self.field_updates = defaultdict(partial(defaultdict, set))
        # {model: {field: {instances}}}
        self.restricted_objects = defaultdict(partial(defaultdict, set))
        # fast_deletes is a list of queryset-likes that can be deleted without
        # fetching the objects into memory.
        self.fast_deletes = []

        # Tracks deletion-order dependency for databases without transactions
        # or ability to defer constraint checks. Only concrete model classes
        # should be included, as the dependencies exist only between actual
        # database tables; proxy models are represented here by their concrete
        # parent.
        self.dependencies = defaultdict(set)  # {model: {models}}
```
### 29 - django/db/models/deletion.py:

Start line: 183, End line: 225

```python
class Collector:

    def can_fast_delete(self, objs, from_field=None):
        """
        Determine if the objects in the given queryset-like or single object
        can be fast-deleted. This can be done if there are no cascades, no
        parents and no signal listeners for the object class.

        The 'from_field' tells where we are coming from - we need this to
        determine if the objects are in fact to be deleted. Allow also
        skipping parent -> child -> parent chain preventing fast delete of
        the child.
        """
        if from_field and from_field.remote_field.on_delete is not CASCADE:
            return False
        if hasattr(objs, "_meta"):
            model = objs._meta.model
        elif hasattr(objs, "model") and hasattr(objs, "_raw_delete"):
            model = objs.model
        else:
            return False
        if self._has_signal_listeners(model):
            return False
        # The use of from_field comes from the need to avoid cascade back to
        # parent when parent delete is cascading to child.
        opts = model._meta
        return (
            all(
                link == from_field
                for link in opts.concrete_model._meta.parents.values()
            )
            and
            # Foreign keys pointing to this model.
            all(
                related.field.remote_field.on_delete is DO_NOTHING
                for related in get_candidate_relations_to_delete(opts)
            )
            and (
                # Something like generic foreign key.
                not any(
                    hasattr(field, "bulk_related_objects")
                    for field in opts.private_fields
                )
            )
        )
```
