# django__django-11179

| **django/django** | `19fc6376ce67d01ca37a91ef2f55ef769f50513a` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 556 |
| **Any found context length** | 556 |
| **Avg pos** | 1.0 |
| **Min pos** | 1 |
| **Max pos** | 1 |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/db/models/deletion.py b/django/db/models/deletion.py
--- a/django/db/models/deletion.py
+++ b/django/db/models/deletion.py
@@ -277,6 +277,7 @@ def delete(self):
             if self.can_fast_delete(instance):
                 with transaction.mark_for_rollback_on_error():
                     count = sql.DeleteQuery(model).delete_batch([instance.pk], self.using)
+                setattr(instance, model._meta.pk.attname, None)
                 return count, {model._meta.label: count}
 
         with transaction.atomic(using=self.using, savepoint=False):

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/db/models/deletion.py | 280 | 280 | 1 | 1 | 556


## Problem Statement

```
delete() on instances of models without any dependencies doesn't clear PKs.
Description
	
Deleting any model with no dependencies not updates the PK on the model. It should be set to None after .delete() call.
See Django.db.models.deletion:276-281. Should update the model line 280.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 django/db/models/deletion.py** | 262 | 328| 556 | 556 | 2761 | 
| 2 | 2 django/db/backends/base/schema.py | 318 | 332| 141 | 697 | 13959 | 
| 3 | **2 django/db/models/deletion.py** | 63 | 78| 162 | 859 | 13959 | 
| 4 | 3 django/db/backends/oracle/schema.py | 41 | 55| 133 | 992 | 15704 | 
| 5 | 4 django/contrib/gis/db/backends/spatialite/schema.py | 84 | 101| 154 | 1146 | 17062 | 
| 6 | 5 django/db/migrations/operations/models.py | 242 | 274| 232 | 1378 | 23808 | 
| 7 | 6 django/db/models/sql/subqueries.py | 44 | 76| 312 | 1690 | 25327 | 
| 8 | 7 django/db/migrations/autodetector.py | 707 | 794| 789 | 2479 | 36998 | 
| 9 | 7 django/db/backends/base/schema.py | 1152 | 1181| 268 | 2747 | 36998 | 
| 10 | **7 django/db/models/deletion.py** | 1 | 60| 462 | 3209 | 36998 | 
| 11 | 8 django/db/models/query.py | 678 | 704| 223 | 3432 | 53378 | 
| 12 | 8 django/db/backends/base/schema.py | 480 | 508| 289 | 3721 | 53378 | 
| 13 | 9 django/db/models/fields/related.py | 822 | 843| 169 | 3890 | 66873 | 
| 14 | 10 django/contrib/sites/models.py | 103 | 121| 121 | 4011 | 67662 | 
| 15 | 10 django/db/models/query.py | 706 | 732| 226 | 4237 | 67662 | 
| 16 | 11 django/db/backends/mysql/schema.py | 73 | 87| 201 | 4438 | 68727 | 
| 17 | 12 django/db/backends/sqlite3/schema.py | 329 | 345| 173 | 4611 | 72681 | 
| 18 | 12 django/db/backends/base/schema.py | 386 | 400| 174 | 4785 | 72681 | 
| 19 | 12 django/db/backends/base/schema.py | 1090 | 1120| 240 | 5025 | 72681 | 
| 20 | 12 django/db/migrations/operations/models.py | 770 | 809| 360 | 5385 | 72681 | 
| 21 | 12 django/db/models/sql/subqueries.py | 1 | 42| 318 | 5703 | 72681 | 
| 22 | 13 django/db/models/base.py | 1048 | 1091| 404 | 6107 | 87571 | 
| 23 | 13 django/db/models/base.py | 875 | 900| 314 | 6421 | 87571 | 
| 24 | 13 django/db/models/base.py | 902 | 926| 238 | 6659 | 87571 | 
| 25 | 14 django/contrib/admin/options.py | 825 | 839| 124 | 6783 | 105899 | 
| 26 | **14 django/db/models/deletion.py** | 119 | 167| 450 | 7233 | 105899 | 
| 27 | 14 django/db/models/query.py | 1304 | 1335| 246 | 7479 | 105899 | 
| 28 | 14 django/contrib/admin/options.py | 2051 | 2103| 451 | 7930 | 105899 | 
| 29 | 14 django/db/models/base.py | 663 | 741| 785 | 8715 | 105899 | 
| 30 | 15 django/core/cache/backends/db.py | 199 | 226| 265 | 8980 | 107985 | 
| 31 | **15 django/db/models/deletion.py** | 169 | 229| 595 | 9575 | 107985 | 
| 32 | 15 django/db/backends/base/schema.py | 354 | 368| 182 | 9757 | 107985 | 
| 33 | 15 django/db/models/base.py | 959 | 987| 230 | 9987 | 107985 | 
| 34 | 15 django/contrib/gis/db/backends/spatialite/schema.py | 63 | 82| 133 | 10120 | 107985 | 
| 35 | 15 django/db/models/base.py | 399 | 503| 903 | 11023 | 107985 | 
| 36 | 15 django/db/migrations/autodetector.py | 796 | 845| 570 | 11593 | 107985 | 
| 37 | 15 django/db/models/base.py | 1807 | 1858| 351 | 11944 | 107985 | 
| 38 | 16 django/contrib/contenttypes/models.py | 118 | 130| 133 | 12077 | 109401 | 
| 39 | 17 django/db/migrations/operations/fields.py | 148 | 173| 183 | 12260 | 112668 | 
| 40 | 18 django/db/models/sql/query.py | 1739 | 1780| 283 | 12543 | 133717 | 
| 41 | 18 django/contrib/admin/options.py | 487 | 500| 165 | 12708 | 133717 | 
| 42 | 18 django/db/migrations/operations/models.py | 844 | 879| 347 | 13055 | 133717 | 
| 43 | 18 django/db/backends/sqlite3/schema.py | 306 | 327| 218 | 13273 | 133717 | 
| 44 | 18 django/core/cache/backends/db.py | 253 | 278| 308 | 13581 | 133717 | 
| 45 | 18 django/db/models/base.py | 380 | 396| 128 | 13709 | 133717 | 
| 46 | 18 django/db/models/base.py | 1169 | 1203| 230 | 13939 | 133717 | 
| 47 | 18 django/contrib/admin/options.py | 1805 | 1873| 584 | 14523 | 133717 | 
| 48 | 18 django/db/models/base.py | 1450 | 1472| 171 | 14694 | 133717 | 
| 49 | 18 django/db/models/fields/related.py | 845 | 871| 240 | 14934 | 133717 | 
| 50 | 19 django/forms/models.py | 770 | 808| 314 | 15248 | 145166 | 
| 51 | 19 django/db/backends/base/schema.py | 551 | 609| 676 | 15924 | 145166 | 
| 52 | 19 django/db/models/base.py | 1093 | 1120| 286 | 16210 | 145166 | 
| 53 | 19 django/db/backends/base/schema.py | 1032 | 1046| 123 | 16333 | 145166 | 
| 54 | 19 django/db/backends/sqlite3/schema.py | 222 | 304| 729 | 17062 | 145166 | 
| 55 | 19 django/db/models/base.py | 567 | 583| 133 | 17195 | 145166 | 
| 56 | 19 django/forms/models.py | 379 | 407| 240 | 17435 | 145166 | 
| 57 | 19 django/db/models/fields/related.py | 1193 | 1316| 1010 | 18445 | 145166 | 
| 58 | 19 django/db/models/base.py | 549 | 565| 142 | 18587 | 145166 | 
| 59 | 19 django/contrib/admin/options.py | 1442 | 1461| 133 | 18720 | 145166 | 
| 60 | 19 django/db/models/base.py | 1230 | 1259| 242 | 18962 | 145166 | 
| 61 | 19 django/db/backends/mysql/schema.py | 89 | 107| 192 | 19154 | 145166 | 
| 62 | 20 django/db/models/sql/compiler.py | 1329 | 1342| 141 | 19295 | 158610 | 
| 63 | 20 django/db/models/base.py | 1568 | 1614| 324 | 19619 | 158610 | 
| 64 | 20 django/db/models/base.py | 1783 | 1804| 155 | 19774 | 158610 | 
| 65 | 20 django/db/models/sql/query.py | 1930 | 1952| 249 | 20023 | 158610 | 
| 66 | **20 django/db/models/deletion.py** | 80 | 117| 327 | 20350 | 158610 | 
| 67 | 20 django/contrib/admin/options.py | 1914 | 1957| 403 | 20753 | 158610 | 
| 68 | 20 django/db/models/base.py | 1351 | 1366| 153 | 20906 | 158610 | 
| 69 | 20 django/db/models/query.py | 859 | 900| 287 | 21193 | 158610 | 
| 70 | 21 django/db/models/manager.py | 165 | 202| 211 | 21404 | 160044 | 
| 71 | 22 django/contrib/sessions/backends/base.py | 292 | 359| 459 | 21863 | 162569 | 
| 72 | 22 django/db/models/base.py | 1319 | 1349| 244 | 22107 | 162569 | 
| 73 | 22 django/db/backends/sqlite3/schema.py | 139 | 220| 820 | 22927 | 162569 | 
| 74 | 22 django/db/models/base.py | 793 | 821| 307 | 23234 | 162569 | 
| 75 | 22 django/db/models/base.py | 823 | 873| 539 | 23773 | 162569 | 
| 76 | 22 django/db/models/base.py | 1474 | 1506| 231 | 24004 | 162569 | 
| 77 | 22 django/db/models/fields/related.py | 509 | 563| 409 | 24413 | 162569 | 
| 78 | 23 django/views/generic/edit.py | 202 | 242| 263 | 24676 | 164285 | 
| 79 | 24 django/db/models/sql/__init__.py | 1 | 8| 0 | 24676 | 164357 | 
| 80 | 24 django/db/models/base.py | 585 | 644| 527 | 25203 | 164357 | 
| 81 | 24 django/db/models/base.py | 1139 | 1167| 213 | 25416 | 164357 | 
| 82 | 24 django/db/models/fields/related.py | 156 | 169| 144 | 25560 | 164357 | 
| 83 | 25 django/db/models/sql/where.py | 207 | 225| 131 | 25691 | 166132 | 
| 84 | 25 django/db/models/sql/query.py | 2163 | 2179| 177 | 25868 | 166132 | 
| 85 | 25 django/forms/models.py | 642 | 672| 217 | 26085 | 166132 | 
| 86 | 25 django/db/migrations/operations/models.py | 120 | 239| 827 | 26912 | 166132 | 
| 87 | 25 django/db/models/fields/related.py | 1160 | 1191| 180 | 27092 | 166132 | 
| 88 | 26 django/contrib/admin/checks.py | 1008 | 1035| 204 | 27296 | 175093 | 
| 89 | 26 django/db/backends/base/schema.py | 958 | 985| 245 | 27541 | 175093 | 
| 90 | 27 django/contrib/contenttypes/management/commands/remove_stale_contenttypes.py | 1 | 81| 598 | 28139 | 175711 | 
| 91 | 28 django/db/backends/oracle/creation.py | 130 | 165| 399 | 28538 | 179606 | 
| 92 | 29 django/core/checks/model_checks.py | 45 | 66| 168 | 28706 | 181008 | 
| 93 | 30 django/contrib/gis/db/backends/mysql/schema.py | 40 | 63| 190 | 28896 | 181608 | 
| 94 | 30 django/db/migrations/autodetector.py | 465 | 506| 418 | 29314 | 181608 | 
| 95 | 31 django/db/models/options.py | 278 | 296| 136 | 29450 | 188474 | 


### Hint

```
Reproduced at 1ffddfc233e2d5139cc6ec31a4ec6ef70b10f87f. Regression in bc7dd8490b882b2cefdc7faf431dc64c532b79c9. Thanks for the report.
Regression test.
I have attached a simple fix which mimics what ​https://github.com/django/django/blob/master/django/db/models/deletion.py#L324-L326 does for multiple objects. I am not sure if we need ​https://github.com/django/django/blob/master/django/db/models/deletion.py#L320-L323 (the block above) because I think field_updates is only ever filled if the objects are not fast-deletable -- ie ​https://github.com/django/django/blob/master/django/db/models/deletion.py#L224 is not called due to the can_fast_delete check at the beginning of the collect function. That said, if we want to be extra "safe" we can just move lines 320 - 326 into an extra function and call that from the old and new location (though I do not think it is needed).
```

## Patch

```diff
diff --git a/django/db/models/deletion.py b/django/db/models/deletion.py
--- a/django/db/models/deletion.py
+++ b/django/db/models/deletion.py
@@ -277,6 +277,7 @@ def delete(self):
             if self.can_fast_delete(instance):
                 with transaction.mark_for_rollback_on_error():
                     count = sql.DeleteQuery(model).delete_batch([instance.pk], self.using)
+                setattr(instance, model._meta.pk.attname, None)
                 return count, {model._meta.label: count}
 
         with transaction.atomic(using=self.using, savepoint=False):

```

## Test Patch

```diff
diff --git a/tests/delete/tests.py b/tests/delete/tests.py
--- a/tests/delete/tests.py
+++ b/tests/delete/tests.py
@@ -1,6 +1,7 @@
 from math import ceil
 
 from django.db import IntegrityError, connection, models
+from django.db.models.deletion import Collector
 from django.db.models.sql.constants import GET_ITERATOR_CHUNK_SIZE
 from django.test import TestCase, skipIfDBFeature, skipUnlessDBFeature
 
@@ -471,6 +472,14 @@ def test_fast_delete_qs(self):
         self.assertEqual(User.objects.count(), 1)
         self.assertTrue(User.objects.filter(pk=u2.pk).exists())
 
+    def test_fast_delete_instance_set_pk_none(self):
+        u = User.objects.create()
+        # User can be fast-deleted.
+        collector = Collector(using='default')
+        self.assertTrue(collector.can_fast_delete(u))
+        u.delete()
+        self.assertIsNone(u.pk)
+
     def test_fast_delete_joined_qs(self):
         a = Avatar.objects.create(desc='a')
         User.objects.create(avatar=a)

```


## Code snippets

### 1 - django/db/models/deletion.py:

Start line: 262, End line: 328

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
                with transaction.mark_for_rollback_on_error():
                    count = sql.DeleteQuery(model).delete_batch([instance.pk], self.using)
                return count, {model._meta.label: count}

        with transaction.atomic(using=self.using, savepoint=False):
            # send pre_delete signals
            for model, obj in self.instances_with_model():
                if not model._meta.auto_created:
                    signals.pre_delete.send(
                        sender=model, instance=obj, using=self.using
                    )

            # fast deletes
            for qs in self.fast_deletes:
                count = qs._raw_delete(using=self.using)
                deleted_counter[qs.model._meta.label] += count

            # update fields
            for model, instances_for_fieldvalues in self.field_updates.items():
                for (field, value), instances in instances_for_fieldvalues.items():
                    query = sql.UpdateQuery(model)
                    query.update_batch([obj.pk for obj in instances],
                                       {field.name: value}, self.using)

            # reverse instance collections
            for instances in self.data.values():
                instances.reverse()

            # delete instances
            for model, instances in self.data.items():
                query = sql.DeleteQuery(model)
                pk_list = [obj.pk for obj in instances]
                count = query.delete_batch(pk_list, self.using)
                deleted_counter[model._meta.label] += count

                if not model._meta.auto_created:
                    for obj in instances:
                        signals.post_delete.send(
                            sender=model, instance=obj, using=self.using
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
### 2 - django/db/backends/base/schema.py:

Start line: 318, End line: 332

```python
class BaseDatabaseSchemaEditor:

    def delete_model(self, model):
        """Delete a model from the database."""
        # Handle auto-created intermediary models
        for field in model._meta.local_many_to_many:
            if field.remote_field.through._meta.auto_created:
                self.delete_model(field.remote_field.through)

        # Delete the table
        self.execute(self.sql_delete_table % {
            "table": self.quote_name(model._meta.db_table),
        })
        # Remove all deferred statements referencing the deleted table.
        for sql in list(self.deferred_sql):
            if isinstance(sql, Statement) and sql.references_table(model._meta.db_table):
                self.deferred_sql.remove(sql)
```
### 3 - django/db/models/deletion.py:

Start line: 63, End line: 78

```python
class Collector:
    def __init__(self, using):
        self.using = using
        # Initially, {model: {instances}}, later values become lists.
        self.data = {}
        self.field_updates = {}  # {model: {(field, value): {instances}}}
        # fast_deletes is a list of queryset-likes that can be deleted without
        # fetching the objects into memory.
        self.fast_deletes = []

        # Tracks deletion-order dependency for databases without transactions
        # or ability to defer constraint checks. Only concrete model classes
        # should be included, as the dependencies exist only between actual
        # database tables; proxy models are represented here by their concrete
        # parent.
        self.dependencies = {}  # {model: {models}}
```
### 4 - django/db/backends/oracle/schema.py:

Start line: 41, End line: 55

```python
class DatabaseSchemaEditor(BaseDatabaseSchemaEditor):

    def delete_model(self, model):
        # Run superclass action
        super().delete_model(model)
        # Clean up manually created sequence.
        self.execute("""
            DECLARE
                i INTEGER;
            BEGIN
                SELECT COUNT(1) INTO i FROM USER_SEQUENCES
                    WHERE SEQUENCE_NAME = '%(sq_name)s';
                IF i = 1 THEN
                    EXECUTE IMMEDIATE 'DROP SEQUENCE "%(sq_name)s"';
                END IF;
            END;
        /""" % {'sq_name': self.connection.ops._get_no_autofield_sequence_name(model._meta.db_table)})
```
### 5 - django/contrib/gis/db/backends/spatialite/schema.py:

Start line: 84, End line: 101

```python
class SpatialiteSchemaEditor(DatabaseSchemaEditor):

    def delete_model(self, model, **kwargs):
        from django.contrib.gis.db.models.fields import GeometryField
        # Drop spatial metadata (dropping the table does not automatically remove them)
        for field in model._meta.local_fields:
            if isinstance(field, GeometryField):
                self.remove_geometry_metadata(model, field)
        # Make sure all geom stuff is gone
        for geom_table in self.geometry_tables:
            try:
                self.execute(
                    self.sql_discard_geometry_columns % {
                        "geom_table": geom_table,
                        "table": self.quote_name(model._meta.db_table),
                    }
                )
            except DatabaseError:
                pass
        super().delete_model(model, **kwargs)
```
### 6 - django/db/migrations/operations/models.py:

Start line: 242, End line: 274

```python
class DeleteModel(ModelOperation):
    """Drop a model's table."""

    def deconstruct(self):
        kwargs = {
            'name': self.name,
        }
        return (
            self.__class__.__qualname__,
            [],
            kwargs
        )

    def state_forwards(self, app_label, state):
        state.remove_model(app_label, self.name_lower)

    def database_forwards(self, app_label, schema_editor, from_state, to_state):
        model = from_state.apps.get_model(app_label, self.name)
        if self.allow_migrate_model(schema_editor.connection.alias, model):
            schema_editor.delete_model(model)

    def database_backwards(self, app_label, schema_editor, from_state, to_state):
        model = to_state.apps.get_model(app_label, self.name)
        if self.allow_migrate_model(schema_editor.connection.alias, model):
            schema_editor.create_model(model)

    def references_model(self, name, app_label=None):
        # The deleted model could be referencing the specified model through
        # related fields.
        return True

    def describe(self):
        return "Delete model %s" % self.name
```
### 7 - django/db/models/sql/subqueries.py:

Start line: 44, End line: 76

```python
class DeleteQuery(Query):

    def delete_qs(self, query, using):
        """
        Delete the queryset in one SQL query (if possible). For simple queries
        this is done by copying the query.query.where to self.query, for
        complex queries by using subquery.
        """
        innerq = query.query
        # Make sure the inner query has at least one table in use.
        innerq.get_initial_alias()
        # The same for our new query.
        self.get_initial_alias()
        innerq_used_tables = tuple([t for t in innerq.alias_map if innerq.alias_refcount[t]])
        if not innerq_used_tables or innerq_used_tables == tuple(self.alias_map):
            # There is only the base table in use in the query.
            self.where = innerq.where
        else:
            pk = query.model._meta.pk
            if not connections[using].features.update_can_self_select:
                # We can't do the delete using subquery.
                values = list(query.values_list('pk', flat=True))
                if not values:
                    return 0
                return self.delete_batch(values, using)
            else:
                innerq.clear_select_clause()
                innerq.select = [
                    pk.get_col(self.get_initial_alias())
                ]
                values = innerq
            self.where = self.where_class()
            self.add_q(Q(pk__in=values))
        cursor = self.get_compiler(using).execute_sql(CURSOR)
        return cursor.rowcount if cursor else 0
```
### 8 - django/db/migrations/autodetector.py:

Start line: 707, End line: 794

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
        all_deleted_models = chain(sorted(deleted_models), sorted(deleted_unmanaged_models))
        for app_label, model_name in all_deleted_models:
            model_state = self.from_state.models[app_label, model_name]
            model = self.old_apps.get_model(app_label, model_name)
            # Gather related fields
            related_fields = {}
            for field in model._meta.local_fields:
                if field.remote_field:
                    if field.remote_field.model:
                        related_fields[field.name] = field
                    # through will be none on M2Ms on swapped-out models;
                    # we can treat lack of through as auto_created=True, though.
                    if (getattr(field.remote_field, "through", None) and
                            not field.remote_field.through._meta.auto_created):
                        related_fields[field.name] = field
            for field in model._meta.local_many_to_many:
                if field.remote_field.model:
                    related_fields[field.name] = field
                if getattr(field.remote_field, "through", None) and not field.remote_field.through._meta.auto_created:
                    related_fields[field.name] = field
            # Generate option removal first
            unique_together = model_state.options.pop('unique_together', None)
            index_together = model_state.options.pop('index_together', None)
            if unique_together:
                self.add_operation(
                    app_label,
                    operations.AlterUniqueTogether(
                        name=model_name,
                        unique_together=None,
                    )
                )
            if index_together:
                self.add_operation(
                    app_label,
                    operations.AlterIndexTogether(
                        name=model_name,
                        index_together=None,
                    )
                )
            # Then remove each related field
            for name in sorted(related_fields):
                self.add_operation(
                    app_label,
                    operations.RemoveField(
                        model_name=model_name,
                        name=name,
                    )
                )
            # Finally, remove the model.
            # This depends on both the removal/alteration of all incoming fields
            # and the removal of all its own related fields, and if it's
            # a through model the field that references it.
            dependencies = []
            for related_object in model._meta.related_objects:
                related_object_app_label = related_object.related_model._meta.app_label
                object_name = related_object.related_model._meta.object_name
                field_name = related_object.field.name
                dependencies.append((related_object_app_label, object_name, field_name, False))
                if not related_object.many_to_many:
                    dependencies.append((related_object_app_label, object_name, field_name, "alter"))

            for name in sorted(related_fields):
                dependencies.append((app_label, model_name, name, False))
            # We're referenced in another field's through=
            through_user = self.through_users.get((app_label, model_state.name_lower))
            if through_user:
                dependencies.append((through_user[0], through_user[1], through_user[2], False))
            # Finally, make the operation, deduping any dependencies
            self.add_operation(
                app_label,
                operations.DeleteModel(
                    name=model_state.name,
                ),
                dependencies=list(set(dependencies)),
            )
```
### 9 - django/db/backends/base/schema.py:

Start line: 1152, End line: 1181

```python
class BaseDatabaseSchemaEditor:

    def _delete_primary_key(self, model, strict=False):
        constraint_names = self._constraint_names(model, primary_key=True)
        if strict and len(constraint_names) != 1:
            raise ValueError('Found wrong number (%s) of PK constraints for %s' % (
                len(constraint_names),
                model._meta.db_table,
            ))
        for constraint_name in constraint_names:
            self.execute(self._delete_primary_key_sql(model, constraint_name))

    def _create_primary_key_sql(self, model, field):
        return Statement(
            self.sql_create_pk,
            table=Table(model._meta.db_table, self.quote_name),
            name=self.quote_name(
                self._create_index_name(model._meta.db_table, [field.column], suffix="_pk")
            ),
            columns=Columns(model._meta.db_table, [field.column], self.quote_name),
        )

    def _delete_primary_key_sql(self, model, name):
        return self._delete_constraint_sql(self.sql_delete_pk, model, name)

    def remove_procedure(self, procedure_name, param_types=()):
        sql = self.sql_delete_procedure % {
            'procedure': self.quote_name(procedure_name),
            'param_types': ','.join(param_types),
        }
        self.execute(sql)
```
### 10 - django/db/models/deletion.py:

Start line: 1, End line: 60

```python
from collections import Counter
from operator import attrgetter

from django.db import IntegrityError, connections, transaction
from django.db.models import signals, sql


class ProtectedError(IntegrityError):
    def __init__(self, msg, protected_objects):
        self.protected_objects = protected_objects
        super().__init__(msg, protected_objects)


def CASCADE(collector, field, sub_objs, using):
    collector.collect(sub_objs, source=field.remote_field.model,
                      source_attr=field.name, nullable=field.null)
    if field.null and not connections[using].features.can_defer_constraint_checks:
        collector.add_field_update(field, None, sub_objs)


def PROTECT(collector, field, sub_objs, using):
    raise ProtectedError(
        "Cannot delete some instances of model '%s' because they are "
        "referenced through a protected foreign key: '%s.%s'" % (
            field.remote_field.model.__name__, sub_objs[0].__class__.__name__, field.name
        ),
        sub_objs
    )


def SET(value):
    if callable(value):
        def set_on_delete(collector, field, sub_objs, using):
            collector.add_field_update(field, value(), sub_objs)
    else:
        def set_on_delete(collector, field, sub_objs, using):
            collector.add_field_update(field, value, sub_objs)
    set_on_delete.deconstruct = lambda: ('django.db.models.SET', (value,), {})
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
        f for f in opts.get_fields(include_hidden=True)
        if f.auto_created and not f.concrete and (f.one_to_one or f.one_to_many)
    )
```
### 26 - django/db/models/deletion.py:

Start line: 119, End line: 167

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
        if hasattr(objs, '_meta'):
            model = type(objs)
        elif hasattr(objs, 'model') and hasattr(objs, '_raw_delete'):
            model = objs.model
        else:
            return False
        if (signals.pre_delete.has_listeners(model) or
                signals.post_delete.has_listeners(model) or
                signals.m2m_changed.has_listeners(model)):
            return False
        # The use of from_field comes from the need to avoid cascade back to
        # parent when parent delete is cascading to child.
        opts = model._meta
        return (
            all(link == from_field for link in opts.concrete_model._meta.parents.values()) and
            # Foreign keys pointing to this model.
            all(
                related.field.remote_field.on_delete is DO_NOTHING
                for related in get_candidate_relations_to_delete(opts)
            ) and (
                # Something like generic foreign key.
                not any(hasattr(field, 'bulk_related_objects') for field in opts.private_fields)
            )
        )

    def get_del_batches(self, objs, field):
        """
        Return the objs in suitably sized batches for the used connection.
        """
        conn_batch_size = max(
            connections[self.using].ops.bulk_batch_size([field.name], objs), 1)
        if len(objs) > conn_batch_size:
            return [objs[i:i + conn_batch_size]
                    for i in range(0, len(objs), conn_batch_size)]
        else:
            return [objs]
```
### 31 - django/db/models/deletion.py:

Start line: 169, End line: 229

```python
class Collector:

    def collect(self, objs, source=None, nullable=False, collect_related=True,
                source_attr=None, reverse_dependency=False, keep_parents=False):
        """
        Add 'objs' to the collection of objects to be deleted as well as all
        parent instances.  'objs' must be a homogeneous iterable collection of
        model instances (e.g. a QuerySet).  If 'collect_related' is True,
        related objects will be handled by their respective on_delete handler.

        If the call is the result of a cascade, 'source' should be the model
        that caused it and 'nullable' should be set to True, if the relation
        can be null.

        If 'reverse_dependency' is True, 'source' will be deleted before the
        current model, rather than after. (Needed for cascading to parent
        models, the one case in which the cascade follows the forwards
        direction of an FK rather than the reverse direction.)

        If 'keep_parents' is True, data of parent model's will be not deleted.
        """
        if self.can_fast_delete(objs):
            self.fast_deletes.append(objs)
            return
        new_objs = self.add(objs, source, nullable,
                            reverse_dependency=reverse_dependency)
        if not new_objs:
            return

        model = new_objs[0].__class__

        if not keep_parents:
            # Recursively collect concrete model's parent models, but not their
            # related objects. These will be found by meta.get_fields()
            concrete_model = model._meta.concrete_model
            for ptr in concrete_model._meta.parents.values():
                if ptr:
                    parent_objs = [getattr(obj, ptr.name) for obj in new_objs]
                    self.collect(parent_objs, source=model,
                                 source_attr=ptr.remote_field.related_name,
                                 collect_related=False,
                                 reverse_dependency=True)
        if collect_related:
            parents = model._meta.parents
            for related in get_candidate_relations_to_delete(model._meta):
                # Preserve parent reverse relationships if keep_parents=True.
                if keep_parents and related.model in parents:
                    continue
                field = related.field
                if field.remote_field.on_delete == DO_NOTHING:
                    continue
                batches = self.get_del_batches(new_objs, field)
                for batch in batches:
                    sub_objs = self.related_objects(related, batch)
                    if self.can_fast_delete(sub_objs, from_field=field):
                        self.fast_deletes.append(sub_objs)
                    elif sub_objs:
                        field.remote_field.on_delete(self, field, sub_objs, self.using)
            for field in model._meta.private_fields:
                if hasattr(field, 'bulk_related_objects'):
                    # It's something like generic foreign key.
                    sub_objs = field.bulk_related_objects(new_objs, self.using)
                    self.collect(sub_objs, source=model, nullable=True)
```
### 66 - django/db/models/deletion.py:

Start line: 80, End line: 117

```python
class Collector:

    def add(self, objs, source=None, nullable=False, reverse_dependency=False):
        """
        Add 'objs' to the collection of objects to be deleted.  If the call is
        the result of a cascade, 'source' should be the model that caused it,
        and 'nullable' should be set to True if the relation can be null.

        Return a list of all objects that were not already collected.
        """
        if not objs:
            return []
        new_objs = []
        model = objs[0].__class__
        instances = self.data.setdefault(model, set())
        for obj in objs:
            if obj not in instances:
                new_objs.append(obj)
        instances.update(new_objs)
        # Nullable relationships can be ignored -- they are nulled out before
        # deleting, and therefore do not affect the order in which objects have
        # to be deleted.
        if source is not None and not nullable:
            if reverse_dependency:
                source, model = model, source
            self.dependencies.setdefault(
                source._meta.concrete_model, set()).add(model._meta.concrete_model)
        return new_objs

    def add_field_update(self, field, value, objs):
        """
        Schedule a field update. 'objs' must be a homogeneous iterable
        collection of model instances (e.g. a QuerySet).
        """
        if not objs:
            return
        model = objs[0].__class__
        self.field_updates.setdefault(
            model, {}).setdefault(
            (field, value), set()).update(objs)
```
