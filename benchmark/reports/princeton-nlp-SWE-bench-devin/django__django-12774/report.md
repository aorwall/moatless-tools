# django__django-12774

| **django/django** | `67f9d076cfc1858b94f9ed6d1a5ce2327dcc8d0d` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 7840 |
| **Any found context length** | 7840 |
| **Avg pos** | 24.0 |
| **Min pos** | 24 |
| **Max pos** | 24 |
| **Top file pos** | 3 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/db/models/query.py b/django/db/models/query.py
--- a/django/db/models/query.py
+++ b/django/db/models/query.py
@@ -689,7 +689,17 @@ def in_bulk(self, id_list=None, *, field_name='pk'):
         """
         assert not self.query.is_sliced, \
             "Cannot use 'limit' or 'offset' with in_bulk"
-        if field_name != 'pk' and not self.model._meta.get_field(field_name).unique:
+        opts = self.model._meta
+        unique_fields = [
+            constraint.fields[0]
+            for constraint in opts.total_unique_constraints
+            if len(constraint.fields) == 1
+        ]
+        if (
+            field_name != 'pk' and
+            not opts.get_field(field_name).unique and
+            field_name not in unique_fields
+        ):
             raise ValueError("in_bulk()'s field_name must be a unique field but %r isn't." % field_name)
         if id_list is not None:
             if not id_list:

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/db/models/query.py | 692 | 692 | 24 | 3 | 7840


## Problem Statement

```
Allow QuerySet.in_bulk() for fields with total UniqueConstraints.
Description
	
If a field is unique by UniqueConstraint instead of unique=True running in_bulk() on that field will fail.
Consider:
class Article(models.Model):
	slug = models.CharField(max_length=255)
	
	class Meta:
		constraints = [
			models.UniqueConstraint(fields=["slug"], name="%(app_label)s_%(class)s_slug_unq")
		]
>>> Article.objects.in_bulk(field_name="slug")
Traceback (most recent call last):
 File "/usr/local/lib/python3.8/code.py", line 90, in runcode
	exec(code, self.locals)
 File "<console>", line 1, in <module>
 File "/app/venv/lib/python3.8/site-packages/django/db/models/manager.py", line 82, in manager_method
	return getattr(self.get_queryset(), name)(*args, **kwargs)
 File "/app/venv/lib/python3.8/site-packages/django/db/models/query.py", line 680, in in_bulk
	raise ValueError("in_bulk()'s field_name must be a unique field but %r isn't." % field_name)
ValueError: in_bulk()'s field_name must be a unique field but 'slug' isn't.
It should be pretty simple to fix this and I have a patch if accepted.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 django/db/models/constraints.py | 72 | 126| 496 | 496 | 1049 | 
| 2 | 2 django/db/models/options.py | 831 | 863| 225 | 721 | 8155 | 
| 3 | **3 django/db/models/query.py** | 518 | 559| 495 | 1216 | 25182 | 
| 4 | 4 django/db/models/base.py | 1072 | 1115| 404 | 1620 | 40825 | 
| 5 | **4 django/db/models/query.py** | 1242 | 1262| 253 | 1873 | 40825 | 
| 6 | 5 django/db/backends/base/schema.py | 1068 | 1082| 126 | 1999 | 52176 | 
| 7 | 5 django/db/models/constraints.py | 1 | 27| 203 | 2202 | 52176 | 
| 8 | 6 django/forms/models.py | 680 | 751| 732 | 2934 | 63897 | 
| 9 | 7 django/db/models/fields/__init__.py | 308 | 336| 205 | 3139 | 81484 | 
| 10 | 8 django/db/models/fields/related.py | 509 | 574| 492 | 3631 | 95316 | 
| 11 | **8 django/db/models/query.py** | 1154 | 1169| 149 | 3780 | 95316 | 
| 12 | 8 django/forms/models.py | 953 | 986| 367 | 4147 | 95316 | 
| 13 | 8 django/db/models/base.py | 1560 | 1585| 183 | 4330 | 95316 | 
| 14 | 8 django/db/models/fields/__init__.py | 2004 | 2034| 252 | 4582 | 95316 | 
| 15 | 8 django/db/models/base.py | 1393 | 1448| 491 | 5073 | 95316 | 
| 16 | 9 django/forms/fields.py | 1184 | 1214| 182 | 5255 | 104329 | 
| 17 | 9 django/db/models/base.py | 1014 | 1070| 560 | 5815 | 104329 | 
| 18 | 9 django/db/models/fields/__init__.py | 244 | 306| 448 | 6263 | 104329 | 
| 19 | 10 django/contrib/admin/views/main.py | 332 | 392| 508 | 6771 | 108627 | 
| 20 | 11 django/db/models/sql/query.py | 2030 | 2052| 249 | 7020 | 130561 | 
| 21 | 12 django/db/models/sql/where.py | 230 | 246| 130 | 7150 | 132365 | 
| 22 | 12 django/db/models/sql/query.py | 2270 | 2286| 177 | 7327 | 132365 | 
| 23 | 13 django/db/backends/mysql/schema.py | 115 | 129| 201 | 7528 | 133861 | 
| **-> 24 <-** | **13 django/db/models/query.py** | 685 | 711| 312 | 7840 | 133861 | 
| 25 | 13 django/db/backends/base/schema.py | 370 | 384| 182 | 8022 | 133861 | 
| 26 | 13 django/db/backends/base/schema.py | 526 | 565| 470 | 8492 | 133861 | 
| 27 | 13 django/db/models/fields/related.py | 860 | 886| 240 | 8732 | 133861 | 
| 28 | 13 django/db/models/fields/__init__.py | 338 | 365| 203 | 8935 | 133861 | 
| 29 | 13 django/db/models/fields/__init__.py | 2289 | 2339| 339 | 9274 | 133861 | 
| 30 | **13 django/db/models/query.py** | 1171 | 1190| 209 | 9483 | 133861 | 
| 31 | 13 django/db/models/fields/related.py | 1424 | 1465| 418 | 9901 | 133861 | 
| 32 | 14 django/db/models/sql/compiler.py | 680 | 702| 202 | 10103 | 147946 | 
| 33 | 14 django/db/models/fields/__init__.py | 367 | 393| 199 | 10302 | 147946 | 
| 34 | 14 django/db/models/base.py | 1163 | 1191| 213 | 10515 | 147946 | 
| 35 | 15 django/contrib/admin/checks.py | 231 | 261| 229 | 10744 | 156953 | 
| 36 | 15 django/forms/models.py | 753 | 774| 194 | 10938 | 156953 | 
| 37 | 15 django/db/models/fields/related.py | 1198 | 1229| 180 | 11118 | 156953 | 
| 38 | **15 django/db/models/query.py** | 776 | 792| 157 | 11275 | 156953 | 
| 39 | 15 django/db/models/sql/query.py | 2124 | 2160| 292 | 11567 | 156953 | 
| 40 | 15 django/db/models/fields/__init__.py | 208 | 242| 235 | 11802 | 156953 | 
| 41 | 16 django/db/backends/sqlite3/operations.py | 1 | 39| 267 | 12069 | 160012 | 
| 42 | **16 django/db/models/query.py** | 454 | 516| 754 | 12823 | 160012 | 
| 43 | 16 django/db/models/sql/query.py | 2054 | 2076| 229 | 13052 | 160012 | 
| 44 | 16 django/db/backends/base/schema.py | 1110 | 1140| 240 | 13292 | 160012 | 
| 45 | 16 django/forms/models.py | 310 | 349| 387 | 13679 | 160012 | 
| 46 | **16 django/db/models/query.py** | 1455 | 1488| 297 | 13976 | 160012 | 
| 47 | 17 django/db/backends/mysql/validation.py | 33 | 70| 287 | 14263 | 160532 | 
| 48 | **17 django/db/models/query.py** | 793 | 832| 322 | 14585 | 160532 | 
| 49 | 17 django/db/models/fields/related.py | 255 | 282| 269 | 14854 | 160532 | 
| 50 | 17 django/db/models/sql/query.py | 1847 | 1883| 318 | 15172 | 160532 | 
| 51 | 18 django/db/models/fields/related_lookups.py | 62 | 99| 447 | 15619 | 161981 | 
| 52 | 18 django/db/models/fields/related.py | 127 | 154| 202 | 15821 | 161981 | 
| 53 | 19 django/contrib/contenttypes/fields.py | 173 | 217| 411 | 16232 | 167414 | 
| 54 | 19 django/db/models/sql/query.py | 1436 | 1514| 734 | 16966 | 167414 | 
| 55 | 19 django/db/models/sql/query.py | 1796 | 1845| 330 | 17296 | 167414 | 
| 56 | **19 django/db/models/query.py** | 1117 | 1152| 314 | 17610 | 167414 | 
| 57 | 19 django/db/backends/base/schema.py | 1084 | 1108| 193 | 17803 | 167414 | 
| 58 | 19 django/db/models/base.py | 1193 | 1227| 230 | 18033 | 167414 | 
| 59 | 19 django/contrib/admin/checks.py | 522 | 545| 230 | 18263 | 167414 | 
| 60 | 20 django/db/migrations/questioner.py | 143 | 160| 183 | 18446 | 169487 | 
| 61 | 20 django/db/models/base.py | 984 | 1012| 230 | 18676 | 169487 | 
| 62 | 20 django/db/models/fields/__init__.py | 611 | 640| 234 | 18910 | 169487 | 
| 63 | 20 django/db/models/fields/related.py | 401 | 419| 165 | 19075 | 169487 | 
| 64 | 21 django/db/models/query_utils.py | 25 | 54| 185 | 19260 | 172199 | 
| 65 | 22 django/db/migrations/operations/models.py | 825 | 860| 347 | 19607 | 178769 | 
| 66 | 22 django/db/models/fields/__init__.py | 2342 | 2391| 311 | 19918 | 178769 | 
| 67 | 22 django/db/backends/base/schema.py | 567 | 629| 700 | 20618 | 178769 | 
| 68 | 22 django/db/models/base.py | 1688 | 1786| 717 | 21335 | 178769 | 
| 69 | 22 django/db/migrations/questioner.py | 162 | 185| 246 | 21581 | 178769 | 
| 70 | 23 django/contrib/postgres/constraints.py | 53 | 63| 127 | 21708 | 179615 | 
| 71 | **23 django/db/models/query.py** | 1310 | 1319| 114 | 21822 | 179615 | 
| 72 | 23 django/forms/models.py | 413 | 443| 243 | 22065 | 179615 | 
| 73 | 24 django/db/models/deletion.py | 1 | 76| 566 | 22631 | 183438 | 
| 74 | 25 django/db/backends/sqlite3/introspection.py | 342 | 420| 750 | 23381 | 187147 | 
| 75 | 25 django/db/backends/mysql/schema.py | 100 | 113| 148 | 23529 | 187147 | 
| 76 | 26 django/db/backends/mysql/introspection.py | 167 | 253| 729 | 24258 | 189375 | 
| 77 | 26 django/contrib/contenttypes/fields.py | 110 | 158| 328 | 24586 | 189375 | 
| 78 | 26 django/db/models/fields/related.py | 1231 | 1348| 963 | 25549 | 189375 | 
| 79 | 26 django/contrib/admin/checks.py | 313 | 324| 138 | 25687 | 189375 | 
| 80 | **26 django/db/models/query.py** | 834 | 863| 248 | 25935 | 189375 | 
| 81 | 26 django/db/models/fields/__init__.py | 1713 | 1736| 146 | 26081 | 189375 | 
| 82 | 27 django/db/models/lookups.py | 356 | 386| 263 | 26344 | 194301 | 
| 83 | 27 django/db/migrations/questioner.py | 56 | 81| 220 | 26564 | 194301 | 
| 84 | 27 django/contrib/contenttypes/fields.py | 656 | 676| 188 | 26752 | 194301 | 
| 85 | 27 django/contrib/postgres/constraints.py | 1 | 51| 427 | 27179 | 194301 | 


## Patch

```diff
diff --git a/django/db/models/query.py b/django/db/models/query.py
--- a/django/db/models/query.py
+++ b/django/db/models/query.py
@@ -689,7 +689,17 @@ def in_bulk(self, id_list=None, *, field_name='pk'):
         """
         assert not self.query.is_sliced, \
             "Cannot use 'limit' or 'offset' with in_bulk"
-        if field_name != 'pk' and not self.model._meta.get_field(field_name).unique:
+        opts = self.model._meta
+        unique_fields = [
+            constraint.fields[0]
+            for constraint in opts.total_unique_constraints
+            if len(constraint.fields) == 1
+        ]
+        if (
+            field_name != 'pk' and
+            not opts.get_field(field_name).unique and
+            field_name not in unique_fields
+        ):
             raise ValueError("in_bulk()'s field_name must be a unique field but %r isn't." % field_name)
         if id_list is not None:
             if not id_list:

```

## Test Patch

```diff
diff --git a/tests/lookup/models.py b/tests/lookup/models.py
--- a/tests/lookup/models.py
+++ b/tests/lookup/models.py
@@ -67,6 +67,11 @@ class Season(models.Model):
     gt = models.IntegerField(null=True, blank=True)
     nulled_text_field = NulledTextField(null=True)
 
+    class Meta:
+        constraints = [
+            models.UniqueConstraint(fields=['year'], name='season_year_unique'),
+        ]
+
     def __str__(self):
         return str(self.year)
 
diff --git a/tests/lookup/tests.py b/tests/lookup/tests.py
--- a/tests/lookup/tests.py
+++ b/tests/lookup/tests.py
@@ -4,10 +4,11 @@
 from operator import attrgetter
 
 from django.core.exceptions import FieldError
-from django.db import connection
+from django.db import connection, models
 from django.db.models import Exists, Max, OuterRef
 from django.db.models.functions import Substr
 from django.test import TestCase, skipUnlessDBFeature
+from django.test.utils import isolate_apps
 from django.utils.deprecation import RemovedInDjango40Warning
 
 from .models import (
@@ -189,11 +190,49 @@ def test_in_bulk_with_field(self):
             }
         )
 
+    def test_in_bulk_meta_constraint(self):
+        season_2011 = Season.objects.create(year=2011)
+        season_2012 = Season.objects.create(year=2012)
+        Season.objects.create(year=2013)
+        self.assertEqual(
+            Season.objects.in_bulk(
+                [season_2011.year, season_2012.year],
+                field_name='year',
+            ),
+            {season_2011.year: season_2011, season_2012.year: season_2012},
+        )
+
     def test_in_bulk_non_unique_field(self):
         msg = "in_bulk()'s field_name must be a unique field but 'author' isn't."
         with self.assertRaisesMessage(ValueError, msg):
             Article.objects.in_bulk([self.au1], field_name='author')
 
+    @isolate_apps('lookup')
+    def test_in_bulk_non_unique_meta_constaint(self):
+        class Model(models.Model):
+            ean = models.CharField(max_length=100)
+            brand = models.CharField(max_length=100)
+            name = models.CharField(max_length=80)
+
+            class Meta:
+                constraints = [
+                    models.UniqueConstraint(
+                        fields=['ean'],
+                        name='partial_ean_unique',
+                        condition=models.Q(is_active=True)
+                    ),
+                    models.UniqueConstraint(
+                        fields=['brand', 'name'],
+                        name='together_brand_name_unique',
+                    ),
+                ]
+
+        msg = "in_bulk()'s field_name must be a unique field but '%s' isn't."
+        for field_name in ['brand', 'ean']:
+            with self.subTest(field_name=field_name):
+                with self.assertRaisesMessage(ValueError, msg % field_name):
+                    Model.objects.in_bulk(field_name=field_name)
+
     def test_values(self):
         # values() returns a list of dictionaries instead of object instances --
         # and you can specify which fields you want to retrieve.

```


## Code snippets

### 1 - django/db/models/constraints.py:

Start line: 72, End line: 126

```python
class UniqueConstraint(BaseConstraint):
    def __init__(self, *, fields, name, condition=None):
        if not fields:
            raise ValueError('At least one field is required to define a unique constraint.')
        if not isinstance(condition, (type(None), Q)):
            raise ValueError('UniqueConstraint.condition must be a Q instance.')
        self.fields = tuple(fields)
        self.condition = condition
        super().__init__(name)

    def _get_condition_sql(self, model, schema_editor):
        if self.condition is None:
            return None
        query = Query(model=model, alias_cols=False)
        where = query.build_where(self.condition)
        compiler = query.get_compiler(connection=schema_editor.connection)
        sql, params = where.as_sql(compiler, schema_editor.connection)
        return sql % tuple(schema_editor.quote_value(p) for p in params)

    def constraint_sql(self, model, schema_editor):
        fields = [model._meta.get_field(field_name).column for field_name in self.fields]
        condition = self._get_condition_sql(model, schema_editor)
        return schema_editor._unique_sql(model, fields, self.name, condition=condition)

    def create_sql(self, model, schema_editor):
        fields = [model._meta.get_field(field_name).column for field_name in self.fields]
        condition = self._get_condition_sql(model, schema_editor)
        return schema_editor._create_unique_sql(model, fields, self.name, condition=condition)

    def remove_sql(self, model, schema_editor):
        condition = self._get_condition_sql(model, schema_editor)
        return schema_editor._delete_unique_sql(model, self.name, condition=condition)

    def __repr__(self):
        return '<%s: fields=%r name=%r%s>' % (
            self.__class__.__name__, self.fields, self.name,
            '' if self.condition is None else ' condition=%s' % self.condition,
        )

    def __eq__(self, other):
        if isinstance(other, UniqueConstraint):
            return (
                self.name == other.name and
                self.fields == other.fields and
                self.condition == other.condition
            )
        return super().__eq__(other)

    def deconstruct(self):
        path, args, kwargs = super().deconstruct()
        kwargs['fields'] = self.fields
        if self.condition:
            kwargs['condition'] = self.condition
        return path, args, kwargs
```
### 2 - django/db/models/options.py:

Start line: 831, End line: 863

```python
class Options:

    @cached_property
    def total_unique_constraints(self):
        """
        Return a list of total unique constraints. Useful for determining set
        of fields guaranteed to be unique for all rows.
        """
        return [
            constraint
            for constraint in self.constraints
            if isinstance(constraint, UniqueConstraint) and constraint.condition is None
        ]

    @cached_property
    def _property_names(self):
        """Return a set of the names of the properties defined on the model."""
        names = []
        for name in dir(self.model):
            attr = inspect.getattr_static(self.model, name)
            if isinstance(attr, property):
                names.append(name)
        return frozenset(names)

    @cached_property
    def db_returning_fields(self):
        """
        Private API intended only to be used by Django itself.
        Fields to be returned after a database insert.
        """
        return [
            field for field in self._get_fields(forward=True, reverse=False, include_parents=PROXY_PARENTS)
            if getattr(field, 'db_returning', False)
        ]
```
### 3 - django/db/models/query.py:

Start line: 518, End line: 559

```python
class QuerySet:

    def bulk_update(self, objs, fields, batch_size=None):
        """
        Update the given fields in each of the given objects in the database.
        """
        if batch_size is not None and batch_size < 0:
            raise ValueError('Batch size must be a positive integer.')
        if not fields:
            raise ValueError('Field names must be given to bulk_update().')
        objs = tuple(objs)
        if any(obj.pk is None for obj in objs):
            raise ValueError('All bulk_update() objects must have a primary key set.')
        fields = [self.model._meta.get_field(name) for name in fields]
        if any(not f.concrete or f.many_to_many for f in fields):
            raise ValueError('bulk_update() can only be used with concrete fields.')
        if any(f.primary_key for f in fields):
            raise ValueError('bulk_update() cannot be used with primary key fields.')
        if not objs:
            return
        # PK is used twice in the resulting update query, once in the filter
        # and once in the WHEN. Each field will also have one CAST.
        max_batch_size = connections[self.db].ops.bulk_batch_size(['pk', 'pk'] + fields, objs)
        batch_size = min(batch_size, max_batch_size) if batch_size else max_batch_size
        requires_casting = connections[self.db].features.requires_casted_case_in_updates
        batches = (objs[i:i + batch_size] for i in range(0, len(objs), batch_size))
        updates = []
        for batch_objs in batches:
            update_kwargs = {}
            for field in fields:
                when_statements = []
                for obj in batch_objs:
                    attr = getattr(obj, field.attname)
                    if not isinstance(attr, Expression):
                        attr = Value(attr, output_field=field)
                    when_statements.append(When(pk=obj.pk, then=attr))
                case_statement = Case(*when_statements, output_field=field)
                if requires_casting:
                    case_statement = Cast(case_statement, output_field=field)
                update_kwargs[field.attname] = case_statement
            updates.append(([obj.pk for obj in batch_objs], update_kwargs))
        with transaction.atomic(using=self.db, savepoint=False):
            for pks, update_kwargs in updates:
                self.filter(pk__in=pks).update(**update_kwargs)
```
### 4 - django/db/models/base.py:

Start line: 1072, End line: 1115

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
### 5 - django/db/models/query.py:

Start line: 1242, End line: 1262

```python
class QuerySet:

    def _batched_insert(self, objs, fields, batch_size, ignore_conflicts=False):
        """
        Helper method for bulk_create() to insert objs one batch at a time.
        """
        if ignore_conflicts and not connections[self.db].features.supports_ignore_conflicts:
            raise NotSupportedError('This database backend does not support ignoring conflicts.')
        ops = connections[self.db].ops
        max_batch_size = max(ops.bulk_batch_size(fields, objs), 1)
        batch_size = min(batch_size, max_batch_size) if batch_size else max_batch_size
        inserted_rows = []
        bulk_return = connections[self.db].features.can_return_rows_from_bulk_insert
        for item in [objs[i:i + batch_size] for i in range(0, len(objs), batch_size)]:
            if bulk_return and not ignore_conflicts:
                inserted_rows.extend(self._insert(
                    item, fields=fields, using=self.db,
                    returning_fields=self.model._meta.db_returning_fields,
                    ignore_conflicts=ignore_conflicts,
                ))
            else:
                self._insert(item, fields=fields, using=self.db, ignore_conflicts=ignore_conflicts)
        return inserted_rows
```
### 6 - django/db/backends/base/schema.py:

Start line: 1068, End line: 1082

```python
class BaseDatabaseSchemaEditor:

    def _unique_sql(self, model, fields, name, condition=None):
        if condition:
            # Databases support conditional unique constraints via a unique
            # index.
            sql = self._create_unique_sql(model, fields, name=name, condition=condition)
            if sql:
                self.deferred_sql.append(sql)
            return None
        constraint = self.sql_unique_constraint % {
            'columns': ', '.join(map(self.quote_name, fields)),
        }
        return self.sql_constraint % {
            'name': self.quote_name(name),
            'constraint': constraint,
        }
```
### 7 - django/db/models/constraints.py:

Start line: 1, End line: 27

```python
from django.db.models.query_utils import Q
from django.db.models.sql.query import Query

__all__ = ['CheckConstraint', 'UniqueConstraint']


class BaseConstraint:
    def __init__(self, name):
        self.name = name

    def constraint_sql(self, model, schema_editor):
        raise NotImplementedError('This method must be implemented by a subclass.')

    def create_sql(self, model, schema_editor):
        raise NotImplementedError('This method must be implemented by a subclass.')

    def remove_sql(self, model, schema_editor):
        raise NotImplementedError('This method must be implemented by a subclass.')

    def deconstruct(self):
        path = '%s.%s' % (self.__class__.__module__, self.__class__.__name__)
        path = path.replace('django.db.models.constraints', 'django.db.models')
        return (path, (), {'name': self.name})

    def clone(self):
        _, args, kwargs = self.deconstruct()
        return self.__class__(*args, **kwargs)
```
### 8 - django/forms/models.py:

Start line: 680, End line: 751

```python
class BaseModelFormSet(BaseFormSet):

    def validate_unique(self):
        # Collect unique_checks and date_checks to run from all the forms.
        all_unique_checks = set()
        all_date_checks = set()
        forms_to_delete = self.deleted_forms
        valid_forms = [form for form in self.forms if form.is_valid() and form not in forms_to_delete]
        for form in valid_forms:
            exclude = form._get_validation_exclusions()
            unique_checks, date_checks = form.instance._get_unique_checks(exclude=exclude)
            all_unique_checks.update(unique_checks)
            all_date_checks.update(date_checks)

        errors = []
        # Do each of the unique checks (unique and unique_together)
        for uclass, unique_check in all_unique_checks:
            seen_data = set()
            for form in valid_forms:
                # Get the data for the set of fields that must be unique among the forms.
                row_data = (
                    field if field in self.unique_fields else form.cleaned_data[field]
                    for field in unique_check if field in form.cleaned_data
                )
                # Reduce Model instances to their primary key values
                row_data = tuple(
                    d._get_pk_val() if hasattr(d, '_get_pk_val')
                    # Prevent "unhashable type: list" errors later on.
                    else tuple(d) if isinstance(d, list)
                    else d for d in row_data
                )
                if row_data and None not in row_data:
                    # if we've already seen it then we have a uniqueness failure
                    if row_data in seen_data:
                        # poke error messages into the right places and mark
                        # the form as invalid
                        errors.append(self.get_unique_error_message(unique_check))
                        form._errors[NON_FIELD_ERRORS] = self.error_class([self.get_form_error()])
                        # remove the data from the cleaned_data dict since it was invalid
                        for field in unique_check:
                            if field in form.cleaned_data:
                                del form.cleaned_data[field]
                    # mark the data as seen
                    seen_data.add(row_data)
        # iterate over each of the date checks now
        for date_check in all_date_checks:
            seen_data = set()
            uclass, lookup, field, unique_for = date_check
            for form in valid_forms:
                # see if we have data for both fields
                if (form.cleaned_data and form.cleaned_data[field] is not None and
                        form.cleaned_data[unique_for] is not None):
                    # if it's a date lookup we need to get the data for all the fields
                    if lookup == 'date':
                        date = form.cleaned_data[unique_for]
                        date_data = (date.year, date.month, date.day)
                    # otherwise it's just the attribute on the date/datetime
                    # object
                    else:
                        date_data = (getattr(form.cleaned_data[unique_for], lookup),)
                    data = (form.cleaned_data[field],) + date_data
                    # if we've already seen it then we have a uniqueness failure
                    if data in seen_data:
                        # poke error messages into the right places and mark
                        # the form as invalid
                        errors.append(self.get_date_error_message(date_check))
                        form._errors[NON_FIELD_ERRORS] = self.error_class([self.get_form_error()])
                        # remove the data from the cleaned_data dict since it was invalid
                        del form.cleaned_data[field]
                    # mark the data as seen
                    seen_data.add(data)

        if errors:
            raise ValidationError(errors)
```
### 9 - django/db/models/fields/__init__.py:

Start line: 308, End line: 336

```python
@total_ordering
class Field(RegisterLookupMixin):

    def _check_db_index(self):
        if self.db_index not in (None, True, False):
            return [
                checks.Error(
                    "'db_index' must be None, True or False.",
                    obj=self,
                    id='fields.E006',
                )
            ]
        else:
            return []

    def _check_null_allowed_for_primary_keys(self):
        if (self.primary_key and self.null and
                not connection.features.interprets_empty_strings_as_nulls):
            # We cannot reliably check this for backends like Oracle which
            # consider NULL and '' to be equal (and thus set up
            # character-based fields a little differently).
            return [
                checks.Error(
                    'Primary keys must not have null=True.',
                    hint=('Set null=False on the field, or '
                          'remove primary_key=True argument.'),
                    obj=self,
                    id='fields.E007',
                )
            ]
        else:
            return []
```
### 10 - django/db/models/fields/related.py:

Start line: 509, End line: 574

```python
class ForeignObject(RelatedField):

    def _check_unique_target(self):
        rel_is_string = isinstance(self.remote_field.model, str)
        if rel_is_string or not self.requires_unique_target:
            return []

        try:
            self.foreign_related_fields
        except exceptions.FieldDoesNotExist:
            return []

        if not self.foreign_related_fields:
            return []

        unique_foreign_fields = {
            frozenset([f.name])
            for f in self.remote_field.model._meta.get_fields()
            if getattr(f, 'unique', False)
        }
        unique_foreign_fields.update({
            frozenset(ut)
            for ut in self.remote_field.model._meta.unique_together
        })
        unique_foreign_fields.update({
            frozenset(uc.fields)
            for uc in self.remote_field.model._meta.total_unique_constraints
        })
        foreign_fields = {f.name for f in self.foreign_related_fields}
        has_unique_constraint = any(u <= foreign_fields for u in unique_foreign_fields)

        if not has_unique_constraint and len(self.foreign_related_fields) > 1:
            field_combination = ', '.join(
                "'%s'" % rel_field.name for rel_field in self.foreign_related_fields
            )
            model_name = self.remote_field.model.__name__
            return [
                checks.Error(
                    "No subset of the fields %s on model '%s' is unique."
                    % (field_combination, model_name),
                    hint=(
                        'Mark a single field as unique=True or add a set of '
                        'fields to a unique constraint (via unique_together '
                        'or a UniqueConstraint (without condition) in the '
                        'model Meta.constraints).'
                    ),
                    obj=self,
                    id='fields.E310',
                )
            ]
        elif not has_unique_constraint:
            field_name = self.foreign_related_fields[0].name
            model_name = self.remote_field.model.__name__
            return [
                checks.Error(
                    "'%s.%s' must be unique because it is referenced by "
                    "a foreign key." % (model_name, field_name),
                    hint=(
                        'Add unique=True to this field or add a '
                        'UniqueConstraint (without condition) in the model '
                        'Meta.constraints.'
                    ),
                    obj=self,
                    id='fields.E311',
                )
            ]
        else:
            return []
```
### 11 - django/db/models/query.py:

Start line: 1154, End line: 1169

```python
class QuerySet:

    def defer(self, *fields):
        """
        Defer the loading of data for certain fields until they are accessed.
        Add the set of deferred fields to any existing set of deferred fields.
        The only exception to this is if None is passed in as the only
        parameter, in which case removal all deferrals.
        """
        self._not_support_combined_queries('defer')
        if self._fields is not None:
            raise TypeError("Cannot call defer() after .values() or .values_list()")
        clone = self._chain()
        if fields == (None,):
            clone.query.clear_deferred_loading()
        else:
            clone.query.add_deferred_loading(fields)
        return clone
```
### 24 - django/db/models/query.py:

Start line: 685, End line: 711

```python
class QuerySet:

    def in_bulk(self, id_list=None, *, field_name='pk'):
        """
        Return a dictionary mapping each of the given IDs to the object with
        that ID. If `id_list` isn't provided, evaluate the entire QuerySet.
        """
        assert not self.query.is_sliced, \
            "Cannot use 'limit' or 'offset' with in_bulk"
        if field_name != 'pk' and not self.model._meta.get_field(field_name).unique:
            raise ValueError("in_bulk()'s field_name must be a unique field but %r isn't." % field_name)
        if id_list is not None:
            if not id_list:
                return {}
            filter_key = '{}__in'.format(field_name)
            batch_size = connections[self.db].features.max_query_params
            id_list = tuple(id_list)
            # If the database has a limit on the number of query parameters
            # (e.g. SQLite), retrieve objects in batches if necessary.
            if batch_size and batch_size < len(id_list):
                qs = ()
                for offset in range(0, len(id_list), batch_size):
                    batch = id_list[offset:offset + batch_size]
                    qs += tuple(self.filter(**{filter_key: batch}).order_by())
            else:
                qs = self.filter(**{filter_key: id_list}).order_by()
        else:
            qs = self._chain()
        return {getattr(obj, field_name): obj for obj in qs}
```
### 30 - django/db/models/query.py:

Start line: 1171, End line: 1190

```python
class QuerySet:

    def only(self, *fields):
        """
        Essentially, the opposite of defer(). Only the fields passed into this
        method and that are not already specified as deferred are loaded
        immediately when the queryset is evaluated.
        """
        self._not_support_combined_queries('only')
        if self._fields is not None:
            raise TypeError("Cannot call only() after .values() or .values_list()")
        if fields == (None,):
            # Can only pass None to defer(), not only(), as the rest option.
            # That won't stop people trying to do this, so let's be explicit.
            raise TypeError("Cannot pass None as an argument to only().")
        for field in fields:
            field = field.split(LOOKUP_SEP, 1)[0]
            if field in self.query._filtered_relations:
                raise ValueError('only() is not supported with FilteredRelation.')
        clone = self._chain()
        clone.query.add_immediate_loading(fields)
        return clone
```
### 38 - django/db/models/query.py:

Start line: 776, End line: 792

```python
class QuerySet:
    update.alters_data = True

    def _update(self, values):
        """
        A version of update() that accepts field objects instead of field names.
        Used primarily for model saving and not intended for use by general
        code (it requires too much poking around at model internals to be
        useful at that level).
        """
        assert not self.query.is_sliced, \
            "Cannot update a query once a slice has been taken."
        query = self.query.chain(sql.UpdateQuery)
        query.add_update_fields(values)
        # Clear any annotations so that they won't be present in subqueries.
        query.annotations = {}
        self._result_cache = None
        return query.get_compiler(self.db).execute_sql(CURSOR)
```
### 42 - django/db/models/query.py:

Start line: 454, End line: 516

```python
class QuerySet:

    def bulk_create(self, objs, batch_size=None, ignore_conflicts=False):
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
        assert batch_size is None or batch_size > 0
        # Check that the parents share the same concrete model with the our
        # model to detect the inheritance pattern ConcreteGrandParent ->
        # MultiTableParent -> ProxyChild. Simply checking self.model._meta.proxy
        # would not identify that case as involving multiple tables.
        for parent in self.model._meta.get_parent_list():
            if parent._meta.concrete_model is not self.model._meta.concrete_model:
                raise ValueError("Can't bulk create a multi-table inherited model")
        if not objs:
            return objs
        self._for_write = True
        connection = connections[self.db]
        opts = self.model._meta
        fields = opts.concrete_fields
        objs = list(objs)
        self._populate_pk_values(objs)
        with transaction.atomic(using=self.db, savepoint=False):
            objs_with_pk, objs_without_pk = partition(lambda o: o.pk is None, objs)
            if objs_with_pk:
                returned_columns = self._batched_insert(
                    objs_with_pk, fields, batch_size, ignore_conflicts=ignore_conflicts,
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
                    objs_without_pk, fields, batch_size, ignore_conflicts=ignore_conflicts,
                )
                if connection.features.can_return_rows_from_bulk_insert and not ignore_conflicts:
                    assert len(returned_columns) == len(objs_without_pk)
                for obj_without_pk, results in zip(objs_without_pk, returned_columns):
                    for result, field in zip(results, opts.db_returning_fields):
                        setattr(obj_without_pk, field.attname, result)
                    obj_without_pk._state.adding = False
                    obj_without_pk._state.db = self.db

        return objs
```
### 46 - django/db/models/query.py:

Start line: 1455, End line: 1488

```python
class RawQuerySet:

    def iterator(self):
        # Cache some things for performance reasons outside the loop.
        db = self.db
        compiler = connections[db].ops.compiler('SQLCompiler')(
            self.query, connections[db], db
        )

        query = iter(self.query)

        try:
            model_init_names, model_init_pos, annotation_fields = self.resolve_model_init_order()
            if self.model._meta.pk.attname not in model_init_names:
                raise exceptions.FieldDoesNotExist(
                    'Raw query must include the primary key'
                )
            model_cls = self.model
            fields = [self.model_fields.get(c) for c in self.columns]
            converters = compiler.get_converters([
                f.get_col(f.model._meta.db_table) if f else None for f in fields
            ])
            if converters:
                query = compiler.apply_converters(query, converters)
            for values in query:
                # Associate fields to values
                model_init_values = [values[pos] for pos in model_init_pos]
                instance = model_cls.from_db(db, model_init_names, model_init_values)
                if annotation_fields:
                    for column, pos in annotation_fields:
                        setattr(instance, column, values[pos])
                yield instance
        finally:
            # Done iterating the Query. If it has its own cursor, close it.
            if hasattr(self.query, 'cursor') and self.query.cursor:
                self.query.cursor.close()
```
### 48 - django/db/models/query.py:

Start line: 793, End line: 832

```python
class QuerySet:
    _update.alters_data = True
    _update.queryset_only = False

    def exists(self):
        if self._result_cache is None:
            return self.query.has_results(using=self.db)
        return bool(self._result_cache)

    def _prefetch_related_objects(self):
        # This method can only be called once the result cache has been filled.
        prefetch_related_objects(self._result_cache, *self._prefetch_related_lookups)
        self._prefetch_done = True

    def explain(self, *, format=None, **options):
        return self.query.explain(using=self.db, format=format, **options)

    ##################################################
    # PUBLIC METHODS THAT RETURN A QUERYSET SUBCLASS #
    ##################################################

    def raw(self, raw_query, params=None, translations=None, using=None):
        if using is None:
            using = self.db
        qs = RawQuerySet(raw_query, model=self.model, params=params, translations=translations, using=using)
        qs._prefetch_related_lookups = self._prefetch_related_lookups[:]
        return qs

    def _values(self, *fields, **expressions):
        clone = self._chain()
        if expressions:
            clone = clone.annotate(**expressions)
        clone._fields = fields
        clone.query.set_values(fields)
        return clone

    def values(self, *fields, **expressions):
        fields += tuple(expressions)
        clone = self._values(*fields, **expressions)
        clone._iterable_class = ValuesIterable
        return clone
```
### 56 - django/db/models/query.py:

Start line: 1117, End line: 1152

```python
class QuerySet:

    def order_by(self, *field_names):
        """Return a new QuerySet instance with the ordering changed."""
        assert not self.query.is_sliced, \
            "Cannot reorder a query once a slice has been taken."
        obj = self._chain()
        obj.query.clear_ordering(force_empty=False)
        obj.query.add_ordering(*field_names)
        return obj

    def distinct(self, *field_names):
        """
        Return a new QuerySet instance that will select only distinct results.
        """
        assert not self.query.is_sliced, \
            "Cannot create distinct fields once a slice has been taken."
        obj = self._chain()
        obj.query.add_distinct_fields(*field_names)
        return obj

    def extra(self, select=None, where=None, params=None, tables=None,
              order_by=None, select_params=None):
        """Add extra SQL fragments to the query."""
        self._not_support_combined_queries('extra')
        assert not self.query.is_sliced, \
            "Cannot change a query once a slice has been taken"
        clone = self._chain()
        clone.query.add_extra(select, select_params, where, params, tables, order_by)
        return clone

    def reverse(self):
        """Reverse the ordering of the QuerySet."""
        if self.query.is_sliced:
            raise TypeError('Cannot reverse a query once a slice has been taken.')
        clone = self._chain()
        clone.query.standard_ordering = not clone.query.standard_ordering
        return clone
```
### 71 - django/db/models/query.py:

Start line: 1310, End line: 1319

```python
class QuerySet:

    def _merge_sanity_check(self, other):
        """Check that two QuerySet classes may be merged."""
        if self._fields is not None and (
                set(self.query.values_select) != set(other.query.values_select) or
                set(self.query.extra_select) != set(other.query.extra_select) or
                set(self.query.annotation_select) != set(other.query.annotation_select)):
            raise TypeError(
                "Merging '%s' classes must involve the same values in each case."
                % self.__class__.__name__
            )
```
### 80 - django/db/models/query.py:

Start line: 834, End line: 863

```python
class QuerySet:

    def values_list(self, *fields, flat=False, named=False):
        if flat and named:
            raise TypeError("'flat' and 'named' can't be used together.")
        if flat and len(fields) > 1:
            raise TypeError("'flat' is not valid when values_list is called with more than one field.")

        field_names = {f for f in fields if not hasattr(f, 'resolve_expression')}
        _fields = []
        expressions = {}
        counter = 1
        for field in fields:
            if hasattr(field, 'resolve_expression'):
                field_id_prefix = getattr(field, 'default_alias', field.__class__.__name__.lower())
                while True:
                    field_id = field_id_prefix + str(counter)
                    counter += 1
                    if field_id not in field_names:
                        break
                expressions[field_id] = field
                _fields.append(field_id)
            else:
                _fields.append(field)

        clone = self._values(*_fields, **expressions)
        clone._iterable_class = (
            NamedValuesListIterable if named
            else FlatValuesListIterable if flat
            else ValuesListIterable
        )
        return clone
```
