# django__django-16317

| **django/django** | `744a1af7f943106e30d538e6ace55c2c66ccd791` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 3418 |
| **Any found context length** | 483 |
| **Avg pos** | 21.0 |
| **Min pos** | 1 |
| **Max pos** | 10 |
| **Top file pos** | 1 |
| **Missing snippets** | 3 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/db/models/query.py b/django/db/models/query.py
--- a/django/db/models/query.py
+++ b/django/db/models/query.py
@@ -732,11 +732,8 @@ def _check_bulk_create_options(
                     "update_fields."
                 )
             if unique_fields:
-                # Primary key is allowed in unique_fields.
                 unique_fields = [
-                    self.model._meta.get_field(name)
-                    for name in unique_fields
-                    if name != "pk"
+                    self.model._meta.get_field(name) for name in unique_fields
                 ]
                 if any(not f.concrete or f.many_to_many for f in unique_fields):
                     raise ValueError(
@@ -785,6 +782,12 @@ def bulk_create(
                 raise ValueError("Can't bulk create a multi-table inherited model")
         if not objs:
             return objs
+        opts = self.model._meta
+        if unique_fields:
+            # Primary key is allowed in unique_fields.
+            unique_fields = [
+                opts.pk.name if name == "pk" else name for name in unique_fields
+            ]
         on_conflict = self._check_bulk_create_options(
             ignore_conflicts,
             update_conflicts,
@@ -792,7 +795,6 @@ def bulk_create(
             unique_fields,
         )
         self._for_write = True
-        opts = self.model._meta
         fields = opts.concrete_fields
         objs = list(objs)
         self._prepare_for_bulk_create(objs)

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/db/models/query.py | 735 | 739 | 1 | 1 | 483
| django/db/models/query.py | 788 | 788 | 10 | 1 | 3418
| django/db/models/query.py | 795 | 795 | 10 | 1 | 3418


## Problem Statement

```
QuerySet.bulk_create() crashes on "pk" in unique_fields.
Description
	 
		(last modified by Mariusz Felisiak)
	 
QuerySet.bulk_create() crashes on "pk" in unique_fields which should be allowed.
 File "/django/django/db/backends/utils.py", line 89, in _execute
	return self.cursor.execute(sql, params)
django.db.utils.ProgrammingError: column "pk" does not exist
LINE 1: ...S (3127, 3, 3, 'c'), (3128, 4, 4, 'd') ON CONFLICT("pk") DO ...
Bug in 0f6946495a8ec955b471ca1baaf408ceb53d4796.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 django/db/models/query.py** | 688 | 747| 483 | 483 | 20636 | 
| 2 | **1 django/db/models/query.py** | 799 | 857| 503 | 986 | 20636 | 
| 3 | 2 django/db/models/constraints.py | 307 | 370| 497 | 1483 | 23564 | 
| 4 | 2 django/db/models/constraints.py | 233 | 249| 139 | 1622 | 23564 | 
| 5 | 3 django/db/models/base.py | 1301 | 1348| 413 | 2035 | 42138 | 
| 6 | 4 django/db/backends/base/schema.py | 1563 | 1614| 343 | 2378 | 56002 | 
| 7 | 4 django/db/models/constraints.py | 215 | 231| 138 | 2516 | 56002 | 
| 8 | 4 django/db/models/constraints.py | 251 | 265| 117 | 2633 | 56002 | 
| 9 | **4 django/db/models/query.py** | 1827 | 1864| 263 | 2896 | 56002 | 
| **-> 10 <-** | **4 django/db/models/query.py** | 749 | 798| 522 | 3418 | 56002 | 
| 11 | **4 django/db/models/query.py** | 1134 | 1168| 310 | 3728 | 56002 | 
| 12 | 4 django/db/backends/base/schema.py | 1523 | 1561| 240 | 3968 | 56002 | 
| 13 | 4 django/db/backends/base/schema.py | 1616 | 1653| 243 | 4211 | 56002 | 
| 14 | 4 django/db/models/base.py | 1398 | 1428| 220 | 4431 | 56002 | 
| 15 | 4 django/db/models/constraints.py | 128 | 213| 684 | 5115 | 56002 | 
| 16 | 5 django/db/backends/mysql/operations.py | 436 | 465| 274 | 5389 | 60180 | 
| 17 | 5 django/db/backends/base/schema.py | 523 | 542| 196 | 5585 | 60180 | 
| 18 | 6 django/db/backends/sqlite3/operations.py | 415 | 435| 148 | 5733 | 63652 | 
| 19 | 7 django/db/backends/postgresql/operations.py | 352 | 371| 150 | 5883 | 66721 | 
| 20 | **7 django/db/models/query.py** | 911 | 951| 298 | 6181 | 66721 | 
| 21 | 8 django/db/models/fields/related.py | 604 | 670| 497 | 6678 | 81225 | 
| 22 | 9 django/forms/models.py | 1596 | 1631| 288 | 6966 | 93435 | 
| 23 | 9 django/db/models/base.py | 1877 | 1905| 192 | 7158 | 93435 | 
| 24 | 9 django/db/backends/base/schema.py | 1720 | 1757| 303 | 7461 | 93435 | 
| 25 | **9 django/db/models/query.py** | 665 | 686| 183 | 7644 | 93435 | 
| 26 | 9 django/db/models/constraints.py | 267 | 277| 163 | 7807 | 93435 | 
| 27 | **9 django/db/models/query.py** | 859 | 909| 564 | 8371 | 93435 | 
| 28 | 9 django/db/models/fields/related.py | 1665 | 1715| 431 | 8802 | 93435 | 
| 29 | 9 django/db/backends/base/schema.py | 565 | 597| 268 | 9070 | 93435 | 
| 30 | 9 django/forms/models.py | 791 | 882| 787 | 9857 | 93435 | 
| 31 | 9 django/forms/models.py | 884 | 907| 202 | 10059 | 93435 | 
| 32 | 9 django/db/backends/sqlite3/operations.py | 1 | 42| 314 | 10373 | 93435 | 
| 33 | 9 django/db/models/constraints.py | 279 | 305| 221 | 10594 | 93435 | 
| 34 | 9 django/db/models/fields/related.py | 1018 | 1054| 261 | 10855 | 93435 | 
| 35 | 10 django/db/backends/oracle/operations.py | 671 | 694| 311 | 11166 | 99655 | 
| 36 | 11 django/db/backends/mysql/schema.py | 156 | 204| 374 | 11540 | 101609 | 
| 37 | 11 django/forms/models.py | 1144 | 1179| 371 | 11911 | 101609 | 
| 38 | 12 django/db/models/fields/__init__.py | 376 | 409| 214 | 12125 | 120274 | 
| 39 | 13 django/db/backends/oracle/creation.py | 159 | 201| 411 | 12536 | 124289 | 
| 40 | 14 django/db/models/sql/query.py | 947 | 974| 272 | 12808 | 147108 | 
| 41 | 15 django/db/migrations/questioner.py | 269 | 288| 195 | 13003 | 149804 | 
| 42 | 16 django/db/backends/postgresql/creation.py | 41 | 56| 174 | 13177 | 150487 | 
| 43 | 16 django/db/backends/sqlite3/operations.py | 354 | 413| 577 | 13754 | 150487 | 
| 44 | 16 django/db/models/fields/__init__.py | 2606 | 2658| 343 | 14097 | 150487 | 
| 45 | **16 django/db/models/query.py** | 1170 | 1192| 160 | 14257 | 150487 | 
| 46 | 16 django/db/backends/base/schema.py | 1478 | 1497| 176 | 14433 | 150487 | 
| 47 | 17 django/db/models/deletion.py | 1 | 93| 601 | 15034 | 154469 | 
| 48 | **17 django/db/models/query.py** | 505 | 522| 136 | 15170 | 154469 | 
| 49 | **17 django/db/models/query.py** | 486 | 503| 150 | 15320 | 154469 | 
| 50 | **17 django/db/models/query.py** | 1509 | 1518| 130 | 15450 | 154469 | 
| 51 | 17 django/db/backends/base/schema.py | 1457 | 1476| 175 | 15625 | 154469 | 
| 52 | 18 django/db/models/sql/subqueries.py | 48 | 78| 212 | 15837 | 155697 | 
| 53 | 18 django/db/backends/mysql/schema.py | 1 | 43| 471 | 16308 | 155697 | 
| 54 | 18 django/db/models/base.py | 1812 | 1847| 246 | 16554 | 155697 | 
| 55 | 18 django/db/models/sql/subqueries.py | 1 | 45| 311 | 16865 | 155697 | 
| 56 | 18 django/db/models/sql/query.py | 546 | 566| 185 | 17050 | 155697 | 
| 57 | 18 django/db/models/base.py | 1763 | 1786| 180 | 17230 | 155697 | 
| 58 | 18 django/db/models/base.py | 1069 | 1121| 520 | 17750 | 155697 | 
| 59 | 19 django/db/backends/sqlite3/schema.py | 1 | 23| 191 | 17941 | 160412 | 
| 60 | 19 django/db/models/sql/query.py | 1 | 79| 576 | 18517 | 160412 | 
| 61 | 20 django/db/models/sql/compiler.py | 1688 | 1766| 657 | 19174 | 176864 | 
| 62 | 20 django/db/models/sql/compiler.py | 1768 | 1808| 280 | 19454 | 176864 | 
| 63 | 21 django/db/backends/mysql/creation.py | 1 | 29| 221 | 19675 | 177533 | 
| 64 | **21 django/db/models/query.py** | 1615 | 1668| 347 | 20022 | 177533 | 
| 65 | 22 django/db/backends/postgresql/schema.py | 330 | 366| 210 | 20232 | 180274 | 
| 66 | 22 django/db/backends/mysql/schema.py | 121 | 136| 116 | 20348 | 180274 | 
| 67 | 22 django/db/backends/postgresql/creation.py | 1 | 39| 266 | 20614 | 180274 | 
| 68 | 22 django/db/models/base.py | 1381 | 1396| 142 | 20756 | 180274 | 
| 69 | 22 django/db/backends/base/schema.py | 1499 | 1521| 199 | 20955 | 180274 | 
| 70 | **22 django/db/models/query.py** | 1916 | 1926| 118 | 21073 | 180274 | 
| 71 | 22 django/forms/models.py | 494 | 524| 247 | 21320 | 180274 | 
| 72 | 22 django/db/backends/oracle/creation.py | 302 | 311| 114 | 21434 | 180274 | 
| 73 | **22 django/db/models/query.py** | 452 | 484| 253 | 21687 | 180274 | 
| 74 | 23 django/db/backends/mysql/compiler.py | 1 | 22| 145 | 21832 | 180921 | 
| 75 | 23 django/db/models/base.py | 2259 | 2450| 1306 | 23138 | 180921 | 
| 76 | 23 django/db/models/fields/related.py | 302 | 339| 296 | 23434 | 180921 | 
| 77 | 23 django/db/backends/sqlite3/schema.py | 553 | 577| 162 | 23596 | 180921 | 
| 78 | 24 django/db/backends/oracle/utils.py | 63 | 98| 262 | 23858 | 181522 | 
| 79 | 24 django/db/backends/sqlite3/schema.py | 123 | 174| 527 | 24385 | 181522 | 
| 80 | 24 django/db/backends/base/schema.py | 439 | 456| 160 | 24545 | 181522 | 
| 81 | 24 django/db/backends/oracle/creation.py | 313 | 334| 180 | 24725 | 181522 | 
| 82 | 24 django/db/models/base.py | 1563 | 1593| 247 | 24972 | 181522 | 
| 83 | 24 django/db/backends/postgresql/schema.py | 267 | 301| 277 | 25249 | 181522 | 
| 84 | **24 django/db/models/query.py** | 354 | 381| 225 | 25474 | 181522 | 
| 85 | 24 django/db/models/fields/related.py | 992 | 1016| 176 | 25650 | 181522 | 
| 86 | 24 django/db/models/fields/related.py | 154 | 185| 209 | 25859 | 181522 | 
| 87 | 24 django/db/backends/sqlite3/schema.py | 25 | 41| 168 | 26027 | 181522 | 
| 88 | 24 django/db/models/fields/__init__.py | 665 | 714| 342 | 26369 | 181522 | 
| 89 | 24 django/db/models/base.py | 1788 | 1810| 175 | 26544 | 181522 | 
| 90 | 24 django/db/models/base.py | 1708 | 1761| 494 | 27038 | 181522 | 
| 91 | 24 django/db/models/sql/query.py | 2385 | 2435| 406 | 27444 | 181522 | 
| 92 | 24 django/db/models/sql/compiler.py | 1 | 27| 197 | 27641 | 181522 | 
| 93 | 25 django/db/backends/base/operations.py | 285 | 330| 328 | 27969 | 187535 | 
| 94 | 26 django/db/models/indexes.py | 92 | 130| 290 | 28259 | 189921 | 
| 95 | 26 django/db/models/sql/query.py | 1628 | 1726| 846 | 29105 | 189921 | 
| 96 | 26 django/db/backends/postgresql/operations.py | 223 | 255| 314 | 29419 | 189921 | 
| 97 | 26 django/db/backends/sqlite3/schema.py | 381 | 400| 176 | 29595 | 189921 | 
| 98 | 26 django/db/models/sql/query.py | 135 | 155| 173 | 29768 | 189921 | 


## Patch

```diff
diff --git a/django/db/models/query.py b/django/db/models/query.py
--- a/django/db/models/query.py
+++ b/django/db/models/query.py
@@ -732,11 +732,8 @@ def _check_bulk_create_options(
                     "update_fields."
                 )
             if unique_fields:
-                # Primary key is allowed in unique_fields.
                 unique_fields = [
-                    self.model._meta.get_field(name)
-                    for name in unique_fields
-                    if name != "pk"
+                    self.model._meta.get_field(name) for name in unique_fields
                 ]
                 if any(not f.concrete or f.many_to_many for f in unique_fields):
                     raise ValueError(
@@ -785,6 +782,12 @@ def bulk_create(
                 raise ValueError("Can't bulk create a multi-table inherited model")
         if not objs:
             return objs
+        opts = self.model._meta
+        if unique_fields:
+            # Primary key is allowed in unique_fields.
+            unique_fields = [
+                opts.pk.name if name == "pk" else name for name in unique_fields
+            ]
         on_conflict = self._check_bulk_create_options(
             ignore_conflicts,
             update_conflicts,
@@ -792,7 +795,6 @@ def bulk_create(
             unique_fields,
         )
         self._for_write = True
-        opts = self.model._meta
         fields = opts.concrete_fields
         objs = list(objs)
         self._prepare_for_bulk_create(objs)

```

## Test Patch

```diff
diff --git a/tests/bulk_create/tests.py b/tests/bulk_create/tests.py
--- a/tests/bulk_create/tests.py
+++ b/tests/bulk_create/tests.py
@@ -595,6 +595,39 @@ def test_update_conflicts_two_fields_unique_fields_first(self):
     def test_update_conflicts_two_fields_unique_fields_second(self):
         self._test_update_conflicts_two_fields(["f2"])
 
+    @skipUnlessDBFeature(
+        "supports_update_conflicts", "supports_update_conflicts_with_target"
+    )
+    def test_update_conflicts_unique_fields_pk(self):
+        TwoFields.objects.bulk_create(
+            [
+                TwoFields(f1=1, f2=1, name="a"),
+                TwoFields(f1=2, f2=2, name="b"),
+            ]
+        )
+        self.assertEqual(TwoFields.objects.count(), 2)
+
+        obj1 = TwoFields.objects.get(f1=1)
+        obj2 = TwoFields.objects.get(f1=2)
+        conflicting_objects = [
+            TwoFields(pk=obj1.pk, f1=3, f2=3, name="c"),
+            TwoFields(pk=obj2.pk, f1=4, f2=4, name="d"),
+        ]
+        TwoFields.objects.bulk_create(
+            conflicting_objects,
+            update_conflicts=True,
+            unique_fields=["pk"],
+            update_fields=["name"],
+        )
+        self.assertEqual(TwoFields.objects.count(), 2)
+        self.assertCountEqual(
+            TwoFields.objects.values("f1", "f2", "name"),
+            [
+                {"f1": 1, "f2": 1, "name": "c"},
+                {"f1": 2, "f2": 2, "name": "d"},
+            ],
+        )
+
     @skipUnlessDBFeature(
         "supports_update_conflicts", "supports_update_conflicts_with_target"
     )

```


## Code snippets

### 1 - django/db/models/query.py:

Start line: 688, End line: 747

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
### 2 - django/db/models/query.py:

Start line: 799, End line: 857

```python
class QuerySet(AltersData):

    def bulk_create(
        self,
        objs,
        batch_size=None,
        ignore_conflicts=False,
        update_conflicts=False,
        update_fields=None,
        unique_fields=None,
    ):
        # ... other code
        with transaction.atomic(using=self.db, savepoint=False):
            objs_with_pk, objs_without_pk = partition(lambda o: o.pk is None, objs)
            if objs_with_pk:
                returned_columns = self._batched_insert(
                    objs_with_pk,
                    fields,
                    batch_size,
                    on_conflict=on_conflict,
                    update_fields=update_fields,
                    unique_fields=unique_fields,
                )
                for obj_with_pk, results in zip(objs_with_pk, returned_columns):
                    for result, field in zip(results, opts.db_returning_fields):
                        if field != opts.pk:
                            setattr(obj_with_pk, field.attname, result)
                for obj_with_pk in objs_with_pk:
                    obj_with_pk._state.adding = False
                    obj_with_pk._state.db = self.db
            if objs_without_pk:
                fields = [f for f in fields if not isinstance(f, AutoField)]
                returned_columns = self._batched_insert(
                    objs_without_pk,
                    fields,
                    batch_size,
                    on_conflict=on_conflict,
                    update_fields=update_fields,
                    unique_fields=unique_fields,
                )
                connection = connections[self.db]
                if (
                    connection.features.can_return_rows_from_bulk_insert
                    and on_conflict is None
                ):
                    assert len(returned_columns) == len(objs_without_pk)
                for obj_without_pk, results in zip(objs_without_pk, returned_columns):
                    for result, field in zip(results, opts.db_returning_fields):
                        setattr(obj_without_pk, field.attname, result)
                    obj_without_pk._state.adding = False
                    obj_without_pk._state.db = self.db

        return objs

    async def abulk_create(
        self,
        objs,
        batch_size=None,
        ignore_conflicts=False,
        update_conflicts=False,
        update_fields=None,
        unique_fields=None,
    ):
        return await sync_to_async(self.bulk_create)(
            objs=objs,
            batch_size=batch_size,
            ignore_conflicts=ignore_conflicts,
            update_conflicts=update_conflicts,
            update_fields=update_fields,
            unique_fields=unique_fields,
        )
```
### 3 - django/db/models/constraints.py:

Start line: 307, End line: 370

```python
class UniqueConstraint(BaseConstraint):

    def validate(self, model, instance, exclude=None, using=DEFAULT_DB_ALIAS):
        queryset = model._default_manager.using(using)
        if self.fields:
            lookup_kwargs = {}
            for field_name in self.fields:
                if exclude and field_name in exclude:
                    return
                field = model._meta.get_field(field_name)
                lookup_value = getattr(instance, field.attname)
                if lookup_value is None or (
                    lookup_value == ""
                    and connections[using].features.interprets_empty_strings_as_nulls
                ):
                    # A composite constraint containing NULL value cannot cause
                    # a violation since NULL != NULL in SQL.
                    return
                lookup_kwargs[field.name] = lookup_value
            queryset = queryset.filter(**lookup_kwargs)
        else:
            # Ignore constraints with excluded fields.
            if exclude:
                for expression in self.expressions:
                    if hasattr(expression, "flatten"):
                        for expr in expression.flatten():
                            if isinstance(expr, F) and expr.name in exclude:
                                return
                    elif isinstance(expression, F) and expression.name in exclude:
                        return
            replacements = {
                F(field): value
                for field, value in instance._get_field_value_map(
                    meta=model._meta, exclude=exclude
                ).items()
            }
            expressions = [
                Exact(expr, expr.replace_expressions(replacements))
                for expr in self.expressions
            ]
            queryset = queryset.filter(*expressions)
        model_class_pk = instance._get_pk_val(model._meta)
        if not instance._state.adding and model_class_pk is not None:
            queryset = queryset.exclude(pk=model_class_pk)
        if not self.condition:
            if queryset.exists():
                if self.expressions:
                    raise ValidationError(self.get_violation_error_message())
                # When fields are defined, use the unique_error_message() for
                # backward compatibility.
                for model, constraints in instance.get_constraints():
                    for constraint in constraints:
                        if constraint is self:
                            raise ValidationError(
                                instance.unique_error_message(model, self.fields)
                            )
        else:
            against = instance._get_field_value_map(meta=model._meta, exclude=exclude)
            try:
                if (self.condition & Exists(queryset.filter(self.condition))).check(
                    against, using=using
                ):
                    raise ValidationError(self.get_violation_error_message())
            except FieldError:
                pass
```
### 4 - django/db/models/constraints.py:

Start line: 233, End line: 249

```python
class UniqueConstraint(BaseConstraint):

    def create_sql(self, model, schema_editor):
        fields = [model._meta.get_field(field_name) for field_name in self.fields]
        include = [
            model._meta.get_field(field_name).column for field_name in self.include
        ]
        condition = self._get_condition_sql(model, schema_editor)
        expressions = self._get_index_expressions(model, schema_editor)
        return schema_editor._create_unique_sql(
            model,
            fields,
            self.name,
            condition=condition,
            deferrable=self.deferrable,
            include=include,
            opclasses=self.opclasses,
            expressions=expressions,
        )
```
### 5 - django/db/models/base.py:

Start line: 1301, End line: 1348

```python
class Model(AltersData, metaclass=ModelBase):

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
                if lookup_value is None or (
                    lookup_value == ""
                    and connection.features.interprets_empty_strings_as_nulls
                ):
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
                errors.setdefault(key, []).append(
                    self.unique_error_message(model_class, unique_check)
                )

        return errors
```
### 6 - django/db/backends/base/schema.py:

Start line: 1563, End line: 1614

```python
class BaseDatabaseSchemaEditor:

    def _create_unique_sql(
        self,
        model,
        fields,
        name=None,
        condition=None,
        deferrable=None,
        include=None,
        opclasses=None,
        expressions=None,
    ):
        if (
            (
                deferrable
                and not self.connection.features.supports_deferrable_unique_constraints
            )
            or (condition and not self.connection.features.supports_partial_indexes)
            or (include and not self.connection.features.supports_covering_indexes)
            or (
                expressions and not self.connection.features.supports_expression_indexes
            )
        ):
            return None

        compiler = Query(model, alias_cols=False).get_compiler(
            connection=self.connection
        )
        table = model._meta.db_table
        columns = [field.column for field in fields]
        if name is None:
            name = self._unique_constraint_name(table, columns, quote=True)
        else:
            name = self.quote_name(name)
        if condition or include or opclasses or expressions:
            sql = self.sql_create_unique_index
        else:
            sql = self.sql_create_unique
        if columns:
            columns = self._index_columns(
                table, columns, col_suffixes=(), opclasses=opclasses
            )
        else:
            columns = Expressions(table, expressions, compiler, self.quote_value)
        return Statement(
            sql,
            table=Table(table, self.quote_name),
            name=name,
            columns=columns,
            condition=self._index_condition_sql(condition),
            deferrable=self._deferrable_constraint_sql(deferrable),
            include=self._index_include_sql(model, include),
        )
```
### 7 - django/db/models/constraints.py:

Start line: 215, End line: 231

```python
class UniqueConstraint(BaseConstraint):

    def constraint_sql(self, model, schema_editor):
        fields = [model._meta.get_field(field_name) for field_name in self.fields]
        include = [
            model._meta.get_field(field_name).column for field_name in self.include
        ]
        condition = self._get_condition_sql(model, schema_editor)
        expressions = self._get_index_expressions(model, schema_editor)
        return schema_editor._unique_sql(
            model,
            fields,
            self.name,
            condition=condition,
            deferrable=self.deferrable,
            include=include,
            opclasses=self.opclasses,
            expressions=expressions,
        )
```
### 8 - django/db/models/constraints.py:

Start line: 251, End line: 265

```python
class UniqueConstraint(BaseConstraint):

    def remove_sql(self, model, schema_editor):
        condition = self._get_condition_sql(model, schema_editor)
        include = [
            model._meta.get_field(field_name).column for field_name in self.include
        ]
        expressions = self._get_index_expressions(model, schema_editor)
        return schema_editor._delete_unique_sql(
            model,
            self.name,
            condition=condition,
            deferrable=self.deferrable,
            include=include,
            opclasses=self.opclasses,
            expressions=expressions,
        )
```
### 9 - django/db/models/query.py:

Start line: 1827, End line: 1864

```python
class QuerySet(AltersData):

    def _batched_insert(
        self,
        objs,
        fields,
        batch_size,
        on_conflict=None,
        update_fields=None,
        unique_fields=None,
    ):
        """
        Helper method for bulk_create() to insert objs one batch at a time.
        """
        connection = connections[self.db]
        ops = connection.ops
        max_batch_size = max(ops.bulk_batch_size(fields, objs), 1)
        batch_size = min(batch_size, max_batch_size) if batch_size else max_batch_size
        inserted_rows = []
        bulk_return = connection.features.can_return_rows_from_bulk_insert
        for item in [objs[i : i + batch_size] for i in range(0, len(objs), batch_size)]:
            if bulk_return and on_conflict is None:
                inserted_rows.extend(
                    self._insert(
                        item,
                        fields=fields,
                        using=self.db,
                        returning_fields=self.model._meta.db_returning_fields,
                    )
                )
            else:
                self._insert(
                    item,
                    fields=fields,
                    using=self.db,
                    on_conflict=on_conflict,
                    update_fields=update_fields,
                    unique_fields=unique_fields,
                )
        return inserted_rows
```
### 10 - django/db/models/query.py:

Start line: 749, End line: 798

```python
class QuerySet(AltersData):

    def bulk_create(
        self,
        objs,
        batch_size=None,
        ignore_conflicts=False,
        update_conflicts=False,
        update_fields=None,
        unique_fields=None,
    ):
        """
        Insert each of the instances into the database. Do *not* call
        save() on each of the instances, do not send any pre/post_save
        signals, and do not set the primary key attribute if it is an
        autoincrement field (except if features.can_return_rows_from_bulk_insert=True).
        Multi-table models are not supported.
        """
        # When you bulk insert you don't get the primary keys back (if it's an
        # autoincrement, except if can_return_rows_from_bulk_insert=True), so
        # you can't insert into the child tables which references this. There
        # are two workarounds:
        # 1) This could be implemented if you didn't have an autoincrement pk
        # 2) You could do it by doing O(n) normal inserts into the parent
        #    tables to get the primary keys back and then doing a single bulk
        #    insert into the childmost table.
        # We currently set the primary keys on the objects when using
        # PostgreSQL via the RETURNING ID clause. It should be possible for
        # Oracle as well, but the semantics for extracting the primary keys is
        # trickier so it's not done yet.
        if batch_size is not None and batch_size <= 0:
            raise ValueError("Batch size must be a positive integer.")
        # Check that the parents share the same concrete model with the our
        # model to detect the inheritance pattern ConcreteGrandParent ->
        # MultiTableParent -> ProxyChild. Simply checking self.model._meta.proxy
        # would not identify that case as involving multiple tables.
        for parent in self.model._meta.get_parent_list():
            if parent._meta.concrete_model is not self.model._meta.concrete_model:
                raise ValueError("Can't bulk create a multi-table inherited model")
        if not objs:
            return objs
        on_conflict = self._check_bulk_create_options(
            ignore_conflicts,
            update_conflicts,
            update_fields,
            unique_fields,
        )
        self._for_write = True
        opts = self.model._meta
        fields = opts.concrete_fields
        objs = list(objs)
        self._prepare_for_bulk_create(objs)
        # ... other code
```
### 11 - django/db/models/query.py:

Start line: 1134, End line: 1168

```python
class QuerySet(AltersData):

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
### 20 - django/db/models/query.py:

Start line: 911, End line: 951

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
### 25 - django/db/models/query.py:

Start line: 665, End line: 686

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
### 27 - django/db/models/query.py:

Start line: 859, End line: 909

```python
class QuerySet(AltersData):

    def bulk_update(self, objs, fields, batch_size=None):
        """
        Update the given fields in each of the given objects in the database.
        """
        if batch_size is not None and batch_size <= 0:
            raise ValueError("Batch size must be a positive integer.")
        if not fields:
            raise ValueError("Field names must be given to bulk_update().")
        objs = tuple(objs)
        if any(obj.pk is None for obj in objs):
            raise ValueError("All bulk_update() objects must have a primary key set.")
        fields = [self.model._meta.get_field(name) for name in fields]
        if any(not f.concrete or f.many_to_many for f in fields):
            raise ValueError("bulk_update() can only be used with concrete fields.")
        if any(f.primary_key for f in fields):
            raise ValueError("bulk_update() cannot be used with primary key fields.")
        if not objs:
            return 0
        for obj in objs:
            obj._prepare_related_fields_for_save(
                operation_name="bulk_update", fields=fields
            )
        # PK is used twice in the resulting update query, once in the filter
        # and once in the WHEN. Each field will also have one CAST.
        self._for_write = True
        connection = connections[self.db]
        max_batch_size = connection.ops.bulk_batch_size(["pk", "pk"] + fields, objs)
        batch_size = min(batch_size, max_batch_size) if batch_size else max_batch_size
        requires_casting = connection.features.requires_casted_case_in_updates
        batches = (objs[i : i + batch_size] for i in range(0, len(objs), batch_size))
        updates = []
        for batch_objs in batches:
            update_kwargs = {}
            for field in fields:
                when_statements = []
                for obj in batch_objs:
                    attr = getattr(obj, field.attname)
                    if not hasattr(attr, "resolve_expression"):
                        attr = Value(attr, output_field=field)
                    when_statements.append(When(pk=obj.pk, then=attr))
                case_statement = Case(*when_statements, output_field=field)
                if requires_casting:
                    case_statement = Cast(case_statement, output_field=field)
                update_kwargs[field.attname] = case_statement
            updates.append(([obj.pk for obj in batch_objs], update_kwargs))
        rows_updated = 0
        queryset = self.using(self.db)
        with transaction.atomic(using=self.db, savepoint=False):
            for pks, update_kwargs in updates:
                rows_updated += queryset.filter(pk__in=pks).update(**update_kwargs)
        return rows_updated
```
### 45 - django/db/models/query.py:

Start line: 1170, End line: 1192

```python
class QuerySet(AltersData):

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
### 48 - django/db/models/query.py:

Start line: 505, End line: 522

```python
class QuerySet(AltersData):

    ####################################
    # METHODS THAT DO DATABASE QUERIES #
    ####################################

    def _iterator(self, use_chunked_fetch, chunk_size):
        iterable = self._iterable_class(
            self,
            chunked_fetch=use_chunked_fetch,
            chunk_size=chunk_size or 2000,
        )
        if not self._prefetch_related_lookups or chunk_size is None:
            yield from iterable
            return

        iterator = iter(iterable)
        while results := list(islice(iterator, chunk_size)):
            prefetch_related_objects(results, *self._prefetch_related_lookups)
            yield from results
```
### 49 - django/db/models/query.py:

Start line: 486, End line: 503

```python
class QuerySet(AltersData):

    def __xor__(self, other):
        self._check_operator_queryset(other, "^")
        self._merge_sanity_check(other)
        if isinstance(self, EmptyQuerySet):
            return other
        if isinstance(other, EmptyQuerySet):
            return self
        query = (
            self
            if self.query.can_filter()
            else self.model._base_manager.filter(pk__in=self.values("pk"))
        )
        combined = query._chain()
        combined._merge_known_related_objects(other)
        if not other.query.can_filter():
            other = other.model._base_manager.filter(pk__in=other.values("pk"))
        combined.query.combine(other.query, sql.XOR)
        return combined
```
### 50 - django/db/models/query.py:

Start line: 1509, End line: 1518

```python
class QuerySet(AltersData):

    def union(self, *other_qs, all=False):
        # If the query is an EmptyQuerySet, combine all nonempty querysets.
        if isinstance(self, EmptyQuerySet):
            qs = [q for q in other_qs if not isinstance(q, EmptyQuerySet)]
            if not qs:
                return self
            if len(qs) == 1:
                return qs[0]
            return qs[0]._combinator_query("union", *qs[1:], all=all)
        return self._combinator_query("union", *other_qs, all=all)
```
### 64 - django/db/models/query.py:

Start line: 1615, End line: 1668

```python
class QuerySet(AltersData):

    def _annotate(self, args, kwargs, select=True):
        self._validate_values_are_expressions(
            args + tuple(kwargs.values()), method_name="annotate"
        )
        annotations = {}
        for arg in args:
            # The default_alias property may raise a TypeError.
            try:
                if arg.default_alias in kwargs:
                    raise ValueError(
                        "The named annotation '%s' conflicts with the "
                        "default name for another annotation." % arg.default_alias
                    )
            except TypeError:
                raise TypeError("Complex annotations require an alias")
            annotations[arg.default_alias] = arg
        annotations.update(kwargs)

        clone = self._chain()
        names = self._fields
        if names is None:
            names = set(
                chain.from_iterable(
                    (field.name, field.attname)
                    if hasattr(field, "attname")
                    else (field.name,)
                    for field in self.model._meta.get_fields()
                )
            )

        for alias, annotation in annotations.items():
            if alias in names:
                raise ValueError(
                    "The annotation '%s' conflicts with a field on "
                    "the model." % alias
                )
            if isinstance(annotation, FilteredRelation):
                clone.query.add_filtered_relation(annotation, alias)
            else:
                clone.query.add_annotation(
                    annotation,
                    alias,
                    is_summary=False,
                    select=select,
                )
        for alias, annotation in clone.query.annotations.items():
            if alias in annotations and annotation.contains_aggregate:
                if clone._fields is None:
                    clone.query.group_by = True
                else:
                    clone.query.set_group_by()
                break

        return clone
```
### 70 - django/db/models/query.py:

Start line: 1916, End line: 1926

```python
class QuerySet(AltersData):

    def _merge_sanity_check(self, other):
        """Check that two QuerySet classes may be merged."""
        if self._fields is not None and (
            set(self.query.values_select) != set(other.query.values_select)
            or set(self.query.extra_select) != set(other.query.extra_select)
            or set(self.query.annotation_select) != set(other.query.annotation_select)
        ):
            raise TypeError(
                "Merging '%s' classes must involve the same values in each case."
                % self.__class__.__name__
            )
```
### 73 - django/db/models/query.py:

Start line: 452, End line: 484

```python
class QuerySet(AltersData):

    def __class_getitem__(cls, *args, **kwargs):
        return cls

    def __and__(self, other):
        self._check_operator_queryset(other, "&")
        self._merge_sanity_check(other)
        if isinstance(other, EmptyQuerySet):
            return other
        if isinstance(self, EmptyQuerySet):
            return self
        combined = self._chain()
        combined._merge_known_related_objects(other)
        combined.query.combine(other.query, sql.AND)
        return combined

    def __or__(self, other):
        self._check_operator_queryset(other, "|")
        self._merge_sanity_check(other)
        if isinstance(self, EmptyQuerySet):
            return other
        if isinstance(other, EmptyQuerySet):
            return self
        query = (
            self
            if self.query.can_filter()
            else self.model._base_manager.filter(pk__in=self.values("pk"))
        )
        combined = query._chain()
        combined._merge_known_related_objects(other)
        if not other.query.can_filter():
            other = other.model._base_manager.filter(pk__in=other.values("pk"))
        combined.query.combine(other.query, sql.OR)
        return combined
```
### 84 - django/db/models/query.py:

Start line: 354, End line: 381

```python
class QuerySet(AltersData):

    def __setstate__(self, state):
        pickled_version = state.get(DJANGO_VERSION_PICKLE_KEY)
        if pickled_version:
            if pickled_version != django.__version__:
                warnings.warn(
                    "Pickled queryset instance's Django version %s does not "
                    "match the current version %s."
                    % (pickled_version, django.__version__),
                    RuntimeWarning,
                    stacklevel=2,
                )
        else:
            warnings.warn(
                "Pickled queryset instance's Django version is not specified.",
                RuntimeWarning,
                stacklevel=2,
            )
        self.__dict__.update(state)

    def __repr__(self):
        data = list(self[: REPR_OUTPUT_SIZE + 1])
        if len(data) > REPR_OUTPUT_SIZE:
            data[-1] = "...(remaining elements truncated)..."
        return "<%s %r>" % (self.__class__.__name__, data)

    def __len__(self):
        self._fetch_all()
        return len(self._result_cache)
```
