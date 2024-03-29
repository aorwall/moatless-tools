# django__django-15732

| **django/django** | `ce69e34bd646558bb44ea92cecfd98b345a0b3e0` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 9803 |
| **Any found context length** | 1038 |
| **Avg pos** | 84.0 |
| **Min pos** | 5 |
| **Max pos** | 32 |
| **Top file pos** | 2 |
| **Missing snippets** | 4 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/db/backends/base/schema.py b/django/db/backends/base/schema.py
--- a/django/db/backends/base/schema.py
+++ b/django/db/backends/base/schema.py
@@ -528,7 +528,10 @@ def alter_unique_together(self, model, old_unique_together, new_unique_together)
         # Deleted uniques
         for fields in olds.difference(news):
             self._delete_composed_index(
-                model, fields, {"unique": True}, self.sql_delete_unique
+                model,
+                fields,
+                {"unique": True, "primary_key": False},
+                self.sql_delete_unique,
             )
         # Created uniques
         for field_names in news.difference(olds):
@@ -568,6 +571,17 @@ def _delete_composed_index(self, model, fields, constraint_kwargs, sql):
             exclude=meta_constraint_names | meta_index_names,
             **constraint_kwargs,
         )
+        if (
+            constraint_kwargs.get("unique") is True
+            and constraint_names
+            and self.connection.features.allows_multiple_constraints_on_same_fields
+        ):
+            # Constraint matching the unique_together name.
+            default_name = str(
+                self._unique_constraint_name(model._meta.db_table, columns, quote=False)
+            )
+            if default_name in constraint_names:
+                constraint_names = [default_name]
         if len(constraint_names) != 1:
             raise ValueError(
                 "Found wrong number (%s) of constraints for %s(%s)"
@@ -1560,16 +1574,13 @@ def _create_unique_sql(
         ):
             return None
 
-        def create_unique_name(*args, **kwargs):
-            return self.quote_name(self._create_index_name(*args, **kwargs))
-
         compiler = Query(model, alias_cols=False).get_compiler(
             connection=self.connection
         )
         table = model._meta.db_table
         columns = [field.column for field in fields]
         if name is None:
-            name = IndexName(table, columns, "_uniq", create_unique_name)
+            name = self._unique_constraint_name(table, columns, quote=True)
         else:
             name = self.quote_name(name)
         if condition or include or opclasses or expressions:
@@ -1592,6 +1603,17 @@ def create_unique_name(*args, **kwargs):
             include=self._index_include_sql(model, include),
         )
 
+    def _unique_constraint_name(self, table, columns, quote=True):
+        if quote:
+
+            def create_unique_name(*args, **kwargs):
+                return self.quote_name(self._create_index_name(*args, **kwargs))
+
+        else:
+            create_unique_name = self._create_index_name
+
+        return IndexName(table, columns, "_uniq", create_unique_name)
+
     def _delete_unique_sql(
         self,
         model,

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/db/backends/base/schema.py | 531 | 531 | 5 | 2 | 1038
| django/db/backends/base/schema.py | 571 | 571 | 29 | 2 | 8945
| django/db/backends/base/schema.py | 1563 | 1572 | 32 | 2 | 9803
| django/db/backends/base/schema.py | 1595 | 1595 | 18 | 2 | 5781


## Problem Statement

```
Cannot drop unique_together constraint on a single field with its own unique=True constraint
Description
	
I have an erroneous unique_together constraint on a model's primary key (unique_together = (('id',),)) that cannot be dropped by a migration. Apparently the migration tries to find all unique constraints on the column and expects there to be only one, but I've got two — the primary key and the unique_together constraint:
Indexes:
	"foo_bar_pkey" PRIMARY KEY, btree (id)
	"foo_bar_id_1c3b3088c74c3b17_uniq" UNIQUE CONSTRAINT, btree (id)
Database is PostgreSQL, if that makes any difference.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 django/db/models/constraints.py | 293 | 350| 457 | 457 | 2767 | 
| 2 | 1 django/db/models/constraints.py | 238 | 252| 117 | 574 | 2767 | 
| 3 | 1 django/db/models/constraints.py | 202 | 218| 138 | 712 | 2767 | 
| 4 | 1 django/db/models/constraints.py | 220 | 236| 139 | 851 | 2767 | 
| **-> 5 <-** | **2 django/db/backends/base/schema.py** | 520 | 536| 187 | 1038 | 16424 | 
| 6 | 2 django/db/models/constraints.py | 266 | 291| 205 | 1243 | 16424 | 
| 7 | 2 django/db/models/constraints.py | 115 | 200| 684 | 1927 | 16424 | 
| 8 | 3 django/db/models/base.py | 1873 | 1901| 188 | 2115 | 34969 | 
| 9 | 3 django/db/models/constraints.py | 254 | 264| 163 | 2278 | 34969 | 
| 10 | 4 django/db/migrations/operations/models.py | 598 | 619| 148 | 2426 | 42720 | 
| 11 | 5 django/db/backends/mysql/schema.py | 138 | 156| 209 | 2635 | 44343 | 
| 12 | 5 django/db/models/base.py | 1395 | 1425| 216 | 2851 | 44343 | 
| 13 | 6 django/db/models/fields/related.py | 603 | 669| 497 | 3348 | 58955 | 
| 14 | 6 django/db/models/base.py | 2255 | 2446| 1302 | 4650 | 58955 | 
| 15 | 7 django/db/migrations/autodetector.py | 1499 | 1525| 217 | 4867 | 72318 | 
| 16 | 7 django/db/models/base.py | 1298 | 1345| 409 | 5276 | 72318 | 
| 17 | 7 django/db/migrations/operations/models.py | 1069 | 1109| 337 | 5613 | 72318 | 
| **-> 18 <-** | **7 django/db/backends/base/schema.py** | 1595 | 1621| 168 | 5781 | 72318 | 
| 19 | **7 django/db/backends/base/schema.py** | 538 | 557| 199 | 5980 | 72318 | 
| 20 | **7 django/db/backends/base/schema.py** | 1499 | 1537| 240 | 6220 | 72318 | 
| 21 | 8 django/contrib/postgres/constraints.py | 195 | 235| 365 | 6585 | 74257 | 
| 22 | 8 django/db/models/base.py | 1844 | 1871| 187 | 6772 | 74257 | 
| 23 | 9 django/db/models/options.py | 950 | 989| 243 | 7015 | 81759 | 
| 24 | 9 django/db/models/fields/related.py | 1680 | 1730| 431 | 7446 | 81759 | 
| 25 | 9 django/db/models/fields/related.py | 1017 | 1053| 261 | 7707 | 81759 | 
| 26 | 9 django/db/migrations/operations/models.py | 1028 | 1066| 283 | 7990 | 81759 | 
| 27 | 10 django/db/models/deletion.py | 1 | 86| 569 | 8559 | 85691 | 
| 28 | 10 django/contrib/postgres/constraints.py | 107 | 127| 202 | 8761 | 85691 | 
| **-> 29 <-** | **10 django/db/backends/base/schema.py** | 559 | 580| 184 | 8945 | 85691 | 
| 30 | 10 django/contrib/postgres/constraints.py | 129 | 153| 184 | 9129 | 85691 | 
| 31 | **10 django/db/backends/base/schema.py** | 1688 | 1725| 303 | 9432 | 85691 | 
| **-> 32 <-** | **10 django/db/backends/base/schema.py** | 1539 | 1593| 371 | 9803 | 85691 | 
| 33 | 11 django/forms/models.py | 796 | 887| 783 | 10586 | 97935 | 
| 34 | 11 django/db/models/fields/related.py | 1211 | 1258| 368 | 10954 | 97935 | 
| 35 | 11 django/db/models/fields/related.py | 1449 | 1578| 984 | 11938 | 97935 | 
| 36 | 11 django/db/models/base.py | 1427 | 1449| 193 | 12131 | 97935 | 
| 37 | 11 django/db/migrations/operations/models.py | 571 | 595| 213 | 12344 | 97935 | 
| 38 | 11 django/db/models/base.py | 1809 | 1842| 232 | 12576 | 97935 | 
| 39 | 12 django/db/backends/oracle/introspection.py | 270 | 351| 614 | 13190 | 100676 | 
| 40 | 12 django/forms/models.py | 889 | 912| 198 | 13388 | 100676 | 
| 41 | 13 django/contrib/postgres/operations.py | 289 | 330| 278 | 13666 | 103024 | 
| 42 | 13 django/db/models/constraints.py | 16 | 50| 285 | 13951 | 103024 | 
| 43 | 13 django/contrib/postgres/constraints.py | 183 | 193| 141 | 14092 | 103024 | 
| 44 | 13 django/db/models/base.py | 1705 | 1758| 490 | 14582 | 103024 | 
| 45 | 14 django/db/backends/mysql/introspection.py | 226 | 311| 698 | 15280 | 105602 | 
| 46 | 14 django/contrib/postgres/operations.py | 255 | 286| 248 | 15528 | 105602 | 
| 47 | 14 django/contrib/postgres/operations.py | 140 | 162| 261 | 15789 | 105602 | 
| 48 | 15 django/db/backends/sqlite3/schema.py | 536 | 560| 162 | 15951 | 110175 | 
| 49 | 15 django/db/models/constraints.py | 1 | 13| 111 | 16062 | 110175 | 
| 50 | 15 django/contrib/postgres/constraints.py | 155 | 181| 231 | 16293 | 110175 | 
| 51 | 16 django/db/backends/sqlite3/introspection.py | 308 | 403| 820 | 17113 | 113522 | 
| 52 | **16 django/db/backends/base/schema.py** | 1475 | 1497| 199 | 17312 | 113522 | 
| 53 | **16 django/db/backends/base/schema.py** | 1433 | 1452| 175 | 17487 | 113522 | 
| 54 | 16 django/db/models/base.py | 1650 | 1684| 248 | 17735 | 113522 | 
| 55 | 17 django/db/models/query.py | 684 | 743| 479 | 18214 | 133695 | 
| 56 | 18 django/db/backends/postgresql/introspection.py | 176 | 263| 777 | 18991 | 136126 | 
| 57 | 18 django/db/models/constraints.py | 53 | 112| 500 | 19491 | 136126 | 
| 58 | 19 django/db/models/sql/where.py | 284 | 301| 141 | 19632 | 138291 | 
| 59 | 20 django/db/migrations/exceptions.py | 1 | 61| 249 | 19881 | 138541 | 
| 60 | 20 django/db/models/fields/related.py | 302 | 339| 296 | 20177 | 138541 | 
| 61 | **20 django/db/backends/base/schema.py** | 1 | 35| 217 | 20394 | 138541 | 
| 62 | 20 django/forms/models.py | 500 | 530| 243 | 20637 | 138541 | 
| 63 | 20 django/db/models/fields/related.py | 1580 | 1678| 655 | 21292 | 138541 | 
| 64 | 20 django/db/migrations/operations/models.py | 560 | 569| 129 | 21421 | 138541 | 
| 65 | 20 django/db/migrations/operations/models.py | 1 | 18| 137 | 21558 | 138541 | 
| 66 | 20 django/db/models/base.py | 1686 | 1703| 156 | 21714 | 138541 | 
| 67 | 21 django/db/backends/postgresql/schema.py | 189 | 235| 408 | 22122 | 140764 | 
| 68 | 22 django/db/backends/postgresql/operations.py | 331 | 350| 150 | 22272 | 143695 | 
| 69 | 22 django/db/models/options.py | 57 | 80| 203 | 22475 | 143695 | 
| 70 | 23 django/db/models/fields/__init__.py | 377 | 410| 214 | 22689 | 162418 | 
| 71 | 23 django/db/models/base.py | 1074 | 1126| 520 | 23209 | 162418 | 
| 72 | 23 django/db/models/base.py | 1760 | 1783| 176 | 23385 | 162418 | 
| 73 | **23 django/db/backends/base/schema.py** | 1647 | 1686| 299 | 23684 | 162418 | 
| 74 | 23 django/db/models/base.py | 1193 | 1233| 306 | 23990 | 162418 | 
| 75 | 23 django/contrib/postgres/constraints.py | 21 | 105| 701 | 24691 | 162418 | 
| 76 | 23 django/db/migrations/operations/models.py | 533 | 558| 163 | 24854 | 162418 | 
| 77 | **23 django/db/backends/base/schema.py** | 1259 | 1289| 331 | 25185 | 162418 | 
| 78 | 23 django/db/models/fields/related.py | 991 | 1015| 176 | 25361 | 162418 | 
| 79 | 23 django/db/migrations/autodetector.py | 1215 | 1302| 719 | 26080 | 162418 | 
| 80 | 23 django/db/migrations/autodetector.py | 1341 | 1362| 197 | 26277 | 162418 | 
| 81 | **23 django/db/backends/base/schema.py** | 38 | 71| 214 | 26491 | 162418 | 
| 82 | 24 django/db/migrations/questioner.py | 269 | 288| 195 | 26686 | 165114 | 
| 83 | 24 django/db/migrations/autodetector.py | 1474 | 1497| 161 | 26847 | 165114 | 
| 84 | 24 django/db/migrations/autodetector.py | 1427 | 1472| 318 | 27165 | 165114 | 
| 85 | 25 django/db/backends/oracle/operations.py | 391 | 437| 385 | 27550 | 171241 | 
| 86 | 25 django/db/models/base.py | 1378 | 1393| 138 | 27688 | 171241 | 
| 87 | 25 django/db/backends/postgresql/schema.py | 237 | 262| 235 | 27923 | 171241 | 
| 88 | 25 django/db/models/fields/related.py | 581 | 601| 138 | 28061 | 171241 | 
| 89 | 25 django/db/models/base.py | 1235 | 1296| 590 | 28651 | 171241 | 
| 90 | 25 django/db/models/base.py | 2156 | 2237| 588 | 29239 | 171241 | 
| 91 | 25 django/db/models/fields/related.py | 1078 | 1100| 180 | 29419 | 171241 | 
| 92 | 25 django/db/models/base.py | 1128 | 1147| 188 | 29607 | 171241 | 
| 93 | **25 django/db/backends/base/schema.py** | 780 | 870| 773 | 30380 | 171241 | 
| 94 | 26 django/db/backends/oracle/schema.py | 105 | 170| 740 | 31120 | 173573 | 
| 95 | 26 django/db/backends/oracle/introspection.py | 352 | 387| 291 | 31411 | 173573 | 
| 96 | 26 django/db/models/fields/related.py | 1417 | 1447| 172 | 31583 | 173573 | 
| 97 | 27 django/db/models/fields/related_descriptors.py | 375 | 396| 158 | 31741 | 184636 | 
| 98 | 28 django/db/backends/mysql/operations.py | 441 | 470| 274 | 32015 | 188835 | 
| 99 | 28 django/db/models/deletion.py | 372 | 396| 273 | 32288 | 188835 | 
| 100 | 28 django/db/models/fields/__init__.py | 2612 | 2664| 343 | 32631 | 188835 | 
| 101 | 28 django/db/backends/postgresql/schema.py | 115 | 187| 602 | 33233 | 188835 | 
| 102 | 29 django/contrib/gis/utils/layermapping.py | 326 | 342| 128 | 33361 | 194431 | 
| 103 | 29 django/db/models/deletion.py | 310 | 370| 616 | 33977 | 194431 | 
| 104 | 29 django/db/migrations/autodetector.py | 1364 | 1390| 144 | 34121 | 194431 | 
| 105 | 30 django/db/backends/sqlite3/operations.py | 417 | 437| 148 | 34269 | 197922 | 
| 106 | 30 django/db/backends/mysql/schema.py | 1 | 42| 456 | 34725 | 197922 | 
| 107 | 31 django/db/migrations/operations/__init__.py | 1 | 43| 227 | 34952 | 198149 | 


### Hint

```
Can you share a traceback that shows how and where the migration fails, please.
Here it is: Traceback (most recent call last): File "./bin/django", line 56, in <module> sys.exit(djangorecipe.manage.main('project.settings.local.foobar')) File "/foo/bar/eggs/djangorecipe-1.10-py2.7.egg/djangorecipe/manage.py", line 9, in main management.execute_from_command_line(sys.argv) File "/foo/bar/eggs/Django-1.7-py2.7.egg/django/core/management/__init__.py", line 385, in execute_from_command_line utility.execute() File "/foo/bar/eggs/Django-1.7-py2.7.egg/django/core/management/__init__.py", line 377, in execute self.fetch_command(subcommand).run_from_argv(self.argv) File "/foo/bar/eggs/Django-1.7-py2.7.egg/django/core/management/base.py", line 288, in run_from_argv self.execute(*args, **options.__dict__) File "/foo/bar/eggs/Django-1.7-py2.7.egg/django/core/management/base.py", line 338, in execute output = self.handle(*args, **options) File "/foo/bar/eggs/Django-1.7-py2.7.egg/django/core/management/commands/migrate.py", line 160, in handle executor.migrate(targets, plan, fake=options.get("fake", False)) File "/foo/bar/eggs/Django-1.7-py2.7.egg/django/db/migrations/executor.py", line 63, in migrate self.apply_migration(migration, fake=fake) File "/foo/bar/eggs/Django-1.7-py2.7.egg/django/db/migrations/executor.py", line 97, in apply_migration migration.apply(project_state, schema_editor) File "/foo/bar/eggs/Django-1.7-py2.7.egg/django/db/migrations/migration.py", line 107, in apply operation.database_forwards(self.app_label, schema_editor, project_state, new_state) File "/foo/bar/eggs/Django-1.7-py2.7.egg/django/db/migrations/operations/models.py", line 253, in database_forwards getattr(new_model._meta, self.option_name, set()), File "/foo/bar/eggs/Django-1.7-py2.7.egg/django/db/backends/schema.py", line 315, in alter_unique_together ", ".join(columns), ValueError: Found wrong number (2) of constraints for foo_bar(id) (some identifier names and paths were changed to protect the innocent :))
Confirmed on current master. On a new postgresql-based project: create a new app 'foo' with the following models.py from django.db import models class Bar(models.Model): name = models.CharField(max_length=255) class Meta: unique_together = (('id',),) ./manage.py makemigrations ./manage.py migrate comment out the 'class Meta' ./manage.py makemigrations ./manage.py migrate This is not specific to the primary key field - it happens more generally on single-field unique_together constraints that duplicate a unique=True constraint, such as: class Bar(models.Model): name = models.CharField(max_length=255, unique=True) class Meta: unique_together = (('name'),)
Can't see a good way of fixing this without breaking backwards compatibility. The alter_unique_together logic in db/backends/schema.py relies on the UNIQUE constraints generated by unique_together to be distinguishable within the database from constraints generated through other mechanisms (e.g. unique=True) - but this isn't possible when the unique_together rule only contains a single field. Another way this manifests itself is to perform the following steps, in separate migrations: create a model add unique=True to one if its fields add a unique_together constraint for that same field This results in two identical calls to _create_unique_sql, leading to a 'relation "foo_bar_name_1e64ed8ec8cfa1c7_uniq" already exists' error.
Replying to matthewwestcott: Can't see a good way of fixing this without breaking backwards compatibility. The alter_unique_together logic in db/backends/schema.py relies on the UNIQUE constraints generated by unique_together to be distinguishable within the database from constraints generated through other mechanisms (e.g. unique=True) - but this isn't possible when the unique_together rule only contains a single field. I have encountered this same issue. Could you explain the backwards compatibility issue here? It seems to me that the bug is in attempting to drop or create an index when that is not called for, either because the index is required by a different constraint or was already created by another constraint. Is that behavior that needs to be kept.
Just got hit by this bug, I believe that at least in recent versions of Django (at least 3.2.13) it is now possible to distinguish unique=True constraints from unique_together ones. At least in the application I'm working on, there seem to be two different naming schemes: For UNIQUE CONSTRAINT created with unique=True the naming scheme seems to be <table_name>_<field_name>_key For UNIQUE CONSTRAINT created with unique_together the naming scheme seems to be <table_name>_<field_names>_<hash>_uniq Note that this is extrapolated from me looking at my application's database schemas, I haven't read the code so I am unsure if this is in fact what happens all of the time. So in the case where we would want to delete a unique_together constraint that had a single field, we'd just look for foo_bar_%_uniq and drop that constraint, here is an SQL statement (in PostgreSQL) that I used in a manual migration (originally from ​https://stackoverflow.com/a/12396684) in case it can be useful to someone that comes across this bug again: DO $body$ DECLARE _con text := ( SELECT quote_ident(conname) FROM pg_constraint WHERE conrelid = 'my_table'::regclass AND contype = 'u' AND conname LIKE 'my_table_my_field_%_uniq' ); BEGIN EXECUTE 'ALTER TABLE my_table DROP CONSTRAINT ' || _con; END $body$; So I think it's possible to fix this bug now?
As a workaround, it may be possible to migrate the unique_together to a UniqueConstraint, keeping the same index name, and then dropping the UniqueConstraint. Just an idea, untested.
Hey there, It's indeed still (mostly) relevant. I've tried to tackle the issue, here is a first draft of the ​PR. "Mostly" because I couldn't reproduce the part that Matt is describing about adding a unique_together: Another way this manifests itself is to perform the following steps, in separate migrations: create a model add unique=True to one if its fields add a unique_together constraint for that same field This results in two identical calls to _create_unique_sql, leading to a 'relation "foo_bar_name_1e64ed8ec8cfa1c7_uniq" already exists' error. For the rest, removing the unique_together on a PK field and on a unique=True field should work now, when we base the constraint checking on how the unique_together names are generated by default. However, this will break dropping such a constraint if it has been renamed manually. I think this should be fine, as the name is generated by Django and never really exposed - so I guess it's okay to regard this name as internals of Django, and we can rely on it. Feel free to tell me what you think of the PR :)
```

## Patch

```diff
diff --git a/django/db/backends/base/schema.py b/django/db/backends/base/schema.py
--- a/django/db/backends/base/schema.py
+++ b/django/db/backends/base/schema.py
@@ -528,7 +528,10 @@ def alter_unique_together(self, model, old_unique_together, new_unique_together)
         # Deleted uniques
         for fields in olds.difference(news):
             self._delete_composed_index(
-                model, fields, {"unique": True}, self.sql_delete_unique
+                model,
+                fields,
+                {"unique": True, "primary_key": False},
+                self.sql_delete_unique,
             )
         # Created uniques
         for field_names in news.difference(olds):
@@ -568,6 +571,17 @@ def _delete_composed_index(self, model, fields, constraint_kwargs, sql):
             exclude=meta_constraint_names | meta_index_names,
             **constraint_kwargs,
         )
+        if (
+            constraint_kwargs.get("unique") is True
+            and constraint_names
+            and self.connection.features.allows_multiple_constraints_on_same_fields
+        ):
+            # Constraint matching the unique_together name.
+            default_name = str(
+                self._unique_constraint_name(model._meta.db_table, columns, quote=False)
+            )
+            if default_name in constraint_names:
+                constraint_names = [default_name]
         if len(constraint_names) != 1:
             raise ValueError(
                 "Found wrong number (%s) of constraints for %s(%s)"
@@ -1560,16 +1574,13 @@ def _create_unique_sql(
         ):
             return None
 
-        def create_unique_name(*args, **kwargs):
-            return self.quote_name(self._create_index_name(*args, **kwargs))
-
         compiler = Query(model, alias_cols=False).get_compiler(
             connection=self.connection
         )
         table = model._meta.db_table
         columns = [field.column for field in fields]
         if name is None:
-            name = IndexName(table, columns, "_uniq", create_unique_name)
+            name = self._unique_constraint_name(table, columns, quote=True)
         else:
             name = self.quote_name(name)
         if condition or include or opclasses or expressions:
@@ -1592,6 +1603,17 @@ def create_unique_name(*args, **kwargs):
             include=self._index_include_sql(model, include),
         )
 
+    def _unique_constraint_name(self, table, columns, quote=True):
+        if quote:
+
+            def create_unique_name(*args, **kwargs):
+                return self.quote_name(self._create_index_name(*args, **kwargs))
+
+        else:
+            create_unique_name = self._create_index_name
+
+        return IndexName(table, columns, "_uniq", create_unique_name)
+
     def _delete_unique_sql(
         self,
         model,

```

## Test Patch

```diff
diff --git a/tests/migrations/test_operations.py b/tests/migrations/test_operations.py
--- a/tests/migrations/test_operations.py
+++ b/tests/migrations/test_operations.py
@@ -2809,6 +2809,69 @@ def test_alter_unique_together_remove(self):
             operation.describe(), "Alter unique_together for Pony (0 constraint(s))"
         )
 
+    @skipUnlessDBFeature("allows_multiple_constraints_on_same_fields")
+    def test_remove_unique_together_on_pk_field(self):
+        app_label = "test_rutopkf"
+        project_state = self.apply_operations(
+            app_label,
+            ProjectState(),
+            operations=[
+                migrations.CreateModel(
+                    "Pony",
+                    fields=[("id", models.AutoField(primary_key=True))],
+                    options={"unique_together": {("id",)}},
+                ),
+            ],
+        )
+        table_name = f"{app_label}_pony"
+        pk_constraint_name = f"{table_name}_pkey"
+        unique_together_constraint_name = f"{table_name}_id_fb61f881_uniq"
+        self.assertConstraintExists(table_name, pk_constraint_name, value=False)
+        self.assertConstraintExists(
+            table_name, unique_together_constraint_name, value=False
+        )
+
+        new_state = project_state.clone()
+        operation = migrations.AlterUniqueTogether("Pony", set())
+        operation.state_forwards(app_label, new_state)
+        with connection.schema_editor() as editor:
+            operation.database_forwards(app_label, editor, project_state, new_state)
+        self.assertConstraintExists(table_name, pk_constraint_name, value=False)
+        self.assertConstraintNotExists(table_name, unique_together_constraint_name)
+
+    @skipUnlessDBFeature("allows_multiple_constraints_on_same_fields")
+    def test_remove_unique_together_on_unique_field(self):
+        app_label = "test_rutouf"
+        project_state = self.apply_operations(
+            app_label,
+            ProjectState(),
+            operations=[
+                migrations.CreateModel(
+                    "Pony",
+                    fields=[
+                        ("id", models.AutoField(primary_key=True)),
+                        ("name", models.CharField(max_length=30, unique=True)),
+                    ],
+                    options={"unique_together": {("name",)}},
+                ),
+            ],
+        )
+        table_name = f"{app_label}_pony"
+        unique_constraint_name = f"{table_name}_name_key"
+        unique_together_constraint_name = f"{table_name}_name_694f3b9f_uniq"
+        self.assertConstraintExists(table_name, unique_constraint_name, value=False)
+        self.assertConstraintExists(
+            table_name, unique_together_constraint_name, value=False
+        )
+
+        new_state = project_state.clone()
+        operation = migrations.AlterUniqueTogether("Pony", set())
+        operation.state_forwards(app_label, new_state)
+        with connection.schema_editor() as editor:
+            operation.database_forwards(app_label, editor, project_state, new_state)
+        self.assertConstraintExists(table_name, unique_constraint_name, value=False)
+        self.assertConstraintNotExists(table_name, unique_together_constraint_name)
+
     def test_add_index(self):
         """
         Test the AddIndex operation.

```


## Code snippets

### 1 - django/db/models/constraints.py:

Start line: 293, End line: 350

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
                    for expr in expression.flatten():
                        if isinstance(expr, F) and expr.name in exclude:
                            return
            replacement_map = instance._get_field_value_map(
                meta=model._meta, exclude=exclude
            )
            expressions = [
                Exact(expr, expr.replace_references(replacement_map))
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
### 2 - django/db/models/constraints.py:

Start line: 238, End line: 252

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
### 3 - django/db/models/constraints.py:

Start line: 202, End line: 218

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
### 4 - django/db/models/constraints.py:

Start line: 220, End line: 236

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
### 5 - django/db/backends/base/schema.py:

Start line: 520, End line: 536

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
            self._delete_composed_index(
                model, fields, {"unique": True}, self.sql_delete_unique
            )
        # Created uniques
        for field_names in news.difference(olds):
            fields = [model._meta.get_field(field) for field in field_names]
            self.execute(self._create_unique_sql(model, fields))
```
### 6 - django/db/models/constraints.py:

Start line: 266, End line: 291

```python
class UniqueConstraint(BaseConstraint):

    def __eq__(self, other):
        if isinstance(other, UniqueConstraint):
            return (
                self.name == other.name
                and self.fields == other.fields
                and self.condition == other.condition
                and self.deferrable == other.deferrable
                and self.include == other.include
                and self.opclasses == other.opclasses
                and self.expressions == other.expressions
            )
        return super().__eq__(other)

    def deconstruct(self):
        path, args, kwargs = super().deconstruct()
        if self.fields:
            kwargs["fields"] = self.fields
        if self.condition:
            kwargs["condition"] = self.condition
        if self.deferrable:
            kwargs["deferrable"] = self.deferrable
        if self.include:
            kwargs["include"] = self.include
        if self.opclasses:
            kwargs["opclasses"] = self.opclasses
        return path, self.expressions, kwargs
```
### 7 - django/db/models/constraints.py:

Start line: 115, End line: 200

```python
class UniqueConstraint(BaseConstraint):
    def __init__(
        self,
        *expressions,
        fields=(),
        name=None,
        condition=None,
        deferrable=None,
        include=None,
        opclasses=(),
        violation_error_message=None,
    ):
        if not name:
            raise ValueError("A unique constraint must be named.")
        if not expressions and not fields:
            raise ValueError(
                "At least one field or expression is required to define a "
                "unique constraint."
            )
        if expressions and fields:
            raise ValueError(
                "UniqueConstraint.fields and expressions are mutually exclusive."
            )
        if not isinstance(condition, (type(None), Q)):
            raise ValueError("UniqueConstraint.condition must be a Q instance.")
        if condition and deferrable:
            raise ValueError("UniqueConstraint with conditions cannot be deferred.")
        if include and deferrable:
            raise ValueError("UniqueConstraint with include fields cannot be deferred.")
        if opclasses and deferrable:
            raise ValueError("UniqueConstraint with opclasses cannot be deferred.")
        if expressions and deferrable:
            raise ValueError("UniqueConstraint with expressions cannot be deferred.")
        if expressions and opclasses:
            raise ValueError(
                "UniqueConstraint.opclasses cannot be used with expressions. "
                "Use django.contrib.postgres.indexes.OpClass() instead."
            )
        if not isinstance(deferrable, (type(None), Deferrable)):
            raise ValueError(
                "UniqueConstraint.deferrable must be a Deferrable instance."
            )
        if not isinstance(include, (type(None), list, tuple)):
            raise ValueError("UniqueConstraint.include must be a list or tuple.")
        if not isinstance(opclasses, (list, tuple)):
            raise ValueError("UniqueConstraint.opclasses must be a list or tuple.")
        if opclasses and len(fields) != len(opclasses):
            raise ValueError(
                "UniqueConstraint.fields and UniqueConstraint.opclasses must "
                "have the same number of elements."
            )
        self.fields = tuple(fields)
        self.condition = condition
        self.deferrable = deferrable
        self.include = tuple(include) if include else ()
        self.opclasses = opclasses
        self.expressions = tuple(
            F(expression) if isinstance(expression, str) else expression
            for expression in expressions
        )
        super().__init__(name, violation_error_message=violation_error_message)

    @property
    def contains_expressions(self):
        return bool(self.expressions)

    def _get_condition_sql(self, model, schema_editor):
        if self.condition is None:
            return None
        query = Query(model=model, alias_cols=False)
        where = query.build_where(self.condition)
        compiler = query.get_compiler(connection=schema_editor.connection)
        sql, params = where.as_sql(compiler, schema_editor.connection)
        return sql % tuple(schema_editor.quote_value(p) for p in params)

    def _get_index_expressions(self, model, schema_editor):
        if not self.expressions:
            return None
        index_expressions = []
        for expression in self.expressions:
            index_expression = IndexExpression(expression)
            index_expression.set_wrapper_classes(schema_editor.connection)
            index_expressions.append(index_expression)
        return ExpressionList(*index_expressions).resolve_expression(
            Query(model, alias_cols=False),
        )
```
### 8 - django/db/models/base.py:

Start line: 1873, End line: 1901

```python
class Model(metaclass=ModelBase):

    @classmethod
    def _check_unique_together(cls):
        """Check the value of "unique_together" option."""
        if not isinstance(cls._meta.unique_together, (tuple, list)):
            return [
                checks.Error(
                    "'unique_together' must be a list or tuple.",
                    obj=cls,
                    id="models.E010",
                )
            ]

        elif any(
            not isinstance(fields, (tuple, list))
            for fields in cls._meta.unique_together
        ):
            return [
                checks.Error(
                    "All 'unique_together' elements must be lists or tuples.",
                    obj=cls,
                    id="models.E011",
                )
            ]

        else:
            errors = []
            for fields in cls._meta.unique_together:
                errors.extend(cls._check_local_fields(fields, "unique_together"))
            return errors
```
### 9 - django/db/models/constraints.py:

Start line: 254, End line: 264

```python
class UniqueConstraint(BaseConstraint):

    def __repr__(self):
        return "<%s:%s%s%s%s%s%s%s>" % (
            self.__class__.__qualname__,
            "" if not self.fields else " fields=%s" % repr(self.fields),
            "" if not self.expressions else " expressions=%s" % repr(self.expressions),
            " name=%s" % repr(self.name),
            "" if self.condition is None else " condition=%s" % self.condition,
            "" if self.deferrable is None else " deferrable=%r" % self.deferrable,
            "" if not self.include else " include=%s" % repr(self.include),
            "" if not self.opclasses else " opclasses=%s" % repr(self.opclasses),
        )
```
### 10 - django/db/migrations/operations/models.py:

Start line: 598, End line: 619

```python
class AlterUniqueTogether(AlterTogetherOptionOperation):
    """
    Change the value of unique_together to the target one.
    Input value of unique_together must be a set of tuples.
    """

    option_name = "unique_together"

    def __init__(self, name, unique_together):
        super().__init__(name, unique_together)


class AlterIndexTogether(AlterTogetherOptionOperation):
    """
    Change the value of index_together to the target one.
    Input value of index_together must be a set of tuples.
    """

    option_name = "index_together"

    def __init__(self, name, index_together):
        super().__init__(name, index_together)
```
### 18 - django/db/backends/base/schema.py:

Start line: 1595, End line: 1621

```python
class BaseDatabaseSchemaEditor:

    def _delete_unique_sql(
        self,
        model,
        name,
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
        if condition or include or opclasses or expressions:
            sql = self.sql_delete_index
        else:
            sql = self.sql_delete_unique
        return self._delete_constraint_sql(sql, model, name)
```
### 19 - django/db/backends/base/schema.py:

Start line: 538, End line: 557

```python
class BaseDatabaseSchemaEditor:

    def alter_index_together(self, model, old_index_together, new_index_together):
        """
        Deal with a model changing its index_together. The input
        index_togethers must be doubly-nested, not the single-nested
        ["foo", "bar"] format.
        """
        olds = {tuple(fields) for fields in old_index_together}
        news = {tuple(fields) for fields in new_index_together}
        # Deleted indexes
        for fields in olds.difference(news):
            self._delete_composed_index(
                model,
                fields,
                {"index": True, "unique": False},
                self.sql_delete_index,
            )
        # Created indexes
        for field_names in news.difference(olds):
            fields = [model._meta.get_field(field) for field in field_names]
            self.execute(self._create_index_sql(model, fields=fields, suffix="_idx"))
```
### 20 - django/db/backends/base/schema.py:

Start line: 1499, End line: 1537

```python
class BaseDatabaseSchemaEditor:

    def _unique_sql(
        self,
        model,
        fields,
        name,
        condition=None,
        deferrable=None,
        include=None,
        opclasses=None,
        expressions=None,
    ):
        if (
            deferrable
            and not self.connection.features.supports_deferrable_unique_constraints
        ):
            return None
        if condition or include or opclasses or expressions:
            # Databases support conditional, covering, and functional unique
            # constraints via a unique index.
            sql = self._create_unique_sql(
                model,
                fields,
                name=name,
                condition=condition,
                include=include,
                opclasses=opclasses,
                expressions=expressions,
            )
            if sql:
                self.deferred_sql.append(sql)
            return None
        constraint = self.sql_unique_constraint % {
            "columns": ", ".join([self.quote_name(field.column) for field in fields]),
            "deferrable": self._deferrable_constraint_sql(deferrable),
        }
        return self.sql_constraint % {
            "name": self.quote_name(name),
            "constraint": constraint,
        }
```
### 29 - django/db/backends/base/schema.py:

Start line: 559, End line: 580

```python
class BaseDatabaseSchemaEditor:

    def _delete_composed_index(self, model, fields, constraint_kwargs, sql):
        meta_constraint_names = {
            constraint.name for constraint in model._meta.constraints
        }
        meta_index_names = {constraint.name for constraint in model._meta.indexes}
        columns = [model._meta.get_field(field).column for field in fields]
        constraint_names = self._constraint_names(
            model,
            columns,
            exclude=meta_constraint_names | meta_index_names,
            **constraint_kwargs,
        )
        if len(constraint_names) != 1:
            raise ValueError(
                "Found wrong number (%s) of constraints for %s(%s)"
                % (
                    len(constraint_names),
                    model._meta.db_table,
                    ", ".join(columns),
                )
            )
        self.execute(self._delete_constraint_sql(sql, model, constraint_names[0]))
```
### 31 - django/db/backends/base/schema.py:

Start line: 1688, End line: 1725

```python
class BaseDatabaseSchemaEditor:

    def _delete_primary_key(self, model, strict=False):
        constraint_names = self._constraint_names(model, primary_key=True)
        if strict and len(constraint_names) != 1:
            raise ValueError(
                "Found wrong number (%s) of PK constraints for %s"
                % (
                    len(constraint_names),
                    model._meta.db_table,
                )
            )
        for constraint_name in constraint_names:
            self.execute(self._delete_primary_key_sql(model, constraint_name))

    def _create_primary_key_sql(self, model, field):
        return Statement(
            self.sql_create_pk,
            table=Table(model._meta.db_table, self.quote_name),
            name=self.quote_name(
                self._create_index_name(
                    model._meta.db_table, [field.column], suffix="_pk"
                )
            ),
            columns=Columns(model._meta.db_table, [field.column], self.quote_name),
        )

    def _delete_primary_key_sql(self, model, name):
        return self._delete_constraint_sql(self.sql_delete_pk, model, name)

    def _collate_sql(self, collation):
        return "COLLATE " + self.quote_name(collation)

    def remove_procedure(self, procedure_name, param_types=()):
        sql = self.sql_delete_procedure % {
            "procedure": self.quote_name(procedure_name),
            "param_types": ",".join(param_types),
        }
        self.execute(sql)
```
### 32 - django/db/backends/base/schema.py:

Start line: 1539, End line: 1593

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

        def create_unique_name(*args, **kwargs):
            return self.quote_name(self._create_index_name(*args, **kwargs))

        compiler = Query(model, alias_cols=False).get_compiler(
            connection=self.connection
        )
        table = model._meta.db_table
        columns = [field.column for field in fields]
        if name is None:
            name = IndexName(table, columns, "_uniq", create_unique_name)
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
### 52 - django/db/backends/base/schema.py:

Start line: 1475, End line: 1497

```python
class BaseDatabaseSchemaEditor:

    def _fk_constraint_name(self, model, field, suffix):
        def create_fk_name(*args, **kwargs):
            return self.quote_name(self._create_index_name(*args, **kwargs))

        return ForeignKeyName(
            model._meta.db_table,
            [field.column],
            split_identifier(field.target_field.model._meta.db_table)[1],
            [field.target_field.column],
            suffix,
            create_fk_name,
        )

    def _delete_fk_sql(self, model, name):
        return self._delete_constraint_sql(self.sql_delete_fk, model, name)

    def _deferrable_constraint_sql(self, deferrable):
        if deferrable is None:
            return ""
        if deferrable == Deferrable.DEFERRED:
            return " DEFERRABLE INITIALLY DEFERRED"
        if deferrable == Deferrable.IMMEDIATE:
            return " DEFERRABLE INITIALLY IMMEDIATE"
```
### 53 - django/db/backends/base/schema.py:

Start line: 1433, End line: 1452

```python
class BaseDatabaseSchemaEditor:

    def _field_should_be_indexed(self, model, field):
        return field.db_index and not field.unique

    def _field_became_primary_key(self, old_field, new_field):
        return not old_field.primary_key and new_field.primary_key

    def _unique_should_be_added(self, old_field, new_field):
        return (
            not new_field.primary_key
            and new_field.unique
            and (not old_field.unique or old_field.primary_key)
        )

    def _rename_field_sql(self, table, old_field, new_field, new_type):
        return self.sql_rename_column % {
            "table": self.quote_name(table),
            "old_column": self.quote_name(old_field.column),
            "new_column": self.quote_name(new_field.column),
            "type": new_type,
        }
```
### 61 - django/db/backends/base/schema.py:

Start line: 1, End line: 35

```python
import logging
import operator
from datetime import datetime

from django.db.backends.ddl_references import (
    Columns,
    Expressions,
    ForeignKeyName,
    IndexName,
    Statement,
    Table,
)
from django.db.backends.utils import names_digest, split_identifier
from django.db.models import Deferrable, Index
from django.db.models.sql import Query
from django.db.transaction import TransactionManagementError, atomic
from django.utils import timezone

logger = logging.getLogger("django.db.backends.schema")


def _is_relevant_relation(relation, altered_field):
    """
    When altering the given field, must constraints on its model from the given
    relation be temporarily dropped?
    """
    field = relation.field
    if field.many_to_many:
        # M2M reverse field
        return False
    if altered_field.primary_key and field.to_fields == [None]:
        # Foreign key constraint on the primary key, which is being altered.
        return True
    # Is the constraint targeting the field being altered?
    return altered_field.name in field.to_fields
```
### 73 - django/db/backends/base/schema.py:

Start line: 1647, End line: 1686

```python
class BaseDatabaseSchemaEditor:

    def _constraint_names(
        self,
        model,
        column_names=None,
        unique=None,
        primary_key=None,
        index=None,
        foreign_key=None,
        check=None,
        type_=None,
        exclude=None,
    ):
        """Return all constraint names matching the columns and conditions."""
        if column_names is not None:
            column_names = [
                self.connection.introspection.identifier_converter(name)
                for name in column_names
            ]
        with self.connection.cursor() as cursor:
            constraints = self.connection.introspection.get_constraints(
                cursor, model._meta.db_table
            )
        result = []
        for name, infodict in constraints.items():
            if column_names is None or column_names == infodict["columns"]:
                if unique is not None and infodict["unique"] != unique:
                    continue
                if primary_key is not None and infodict["primary_key"] != primary_key:
                    continue
                if index is not None and infodict["index"] != index:
                    continue
                if check is not None and infodict["check"] != check:
                    continue
                if foreign_key is not None and not infodict["foreign_key"]:
                    continue
                if type_ is not None and infodict["type"] != type_:
                    continue
                if not exclude or name not in exclude:
                    result.append(name)
        return result
```
### 77 - django/db/backends/base/schema.py:

Start line: 1259, End line: 1289

```python
class BaseDatabaseSchemaEditor:

    def _create_index_name(self, table_name, column_names, suffix=""):
        """
        Generate a unique name for an index/unique constraint.

        The name is divided into 3 parts: the table name, the column names,
        and a unique digest and suffix.
        """
        _, table_name = split_identifier(table_name)
        hash_suffix_part = "%s%s" % (
            names_digest(table_name, *column_names, length=8),
            suffix,
        )
        max_length = self.connection.ops.max_name_length() or 200
        # If everything fits into max_length, use that name.
        index_name = "%s_%s_%s" % (table_name, "_".join(column_names), hash_suffix_part)
        if len(index_name) <= max_length:
            return index_name
        # Shorten a long suffix.
        if len(hash_suffix_part) > max_length / 3:
            hash_suffix_part = hash_suffix_part[: max_length // 3]
        other_length = (max_length - len(hash_suffix_part)) // 2 - 1
        index_name = "%s_%s_%s" % (
            table_name[:other_length],
            "_".join(column_names)[:other_length],
            hash_suffix_part,
        )
        # Prepend D if needed to prevent the name from starting with an
        # underscore or a number (not permitted on Oracle).
        if index_name[0] == "_" or index_name[0].isdigit():
            index_name = "D%s" % index_name[:-1]
        return index_name
```
### 81 - django/db/backends/base/schema.py:

Start line: 38, End line: 71

```python
def _all_related_fields(model):
    # Related fields must be returned in a deterministic order.
    return sorted(
        model._meta._get_fields(
            forward=False,
            reverse=True,
            include_hidden=True,
            include_parents=False,
        ),
        key=operator.attrgetter("name"),
    )


def _related_non_m2m_objects(old_field, new_field):
    # Filter out m2m objects from reverse relations.
    # Return (old_relation, new_relation) tuples.
    related_fields = zip(
        (
            obj
            for obj in _all_related_fields(old_field.model)
            if _is_relevant_relation(obj, old_field)
        ),
        (
            obj
            for obj in _all_related_fields(new_field.model)
            if _is_relevant_relation(obj, new_field)
        ),
    )
    for old_rel, new_rel in related_fields:
        yield old_rel, new_rel
        yield from _related_non_m2m_objects(
            old_rel.remote_field,
            new_rel.remote_field,
        )
```
### 93 - django/db/backends/base/schema.py:

Start line: 780, End line: 870

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
        """Perform a "physical" (non-ManyToMany) field update."""
        # Drop any FK constraints, we'll remake them later
        fks_dropped = set()
        if (
            self.connection.features.supports_foreign_keys
            and old_field.remote_field
            and old_field.db_constraint
        ):
            fk_names = self._constraint_names(
                model, [old_field.column], foreign_key=True
            )
            if strict and len(fk_names) != 1:
                raise ValueError(
                    "Found wrong number (%s) of foreign key constraints for %s.%s"
                    % (
                        len(fk_names),
                        model._meta.db_table,
                        old_field.column,
                    )
                )
            for fk_name in fk_names:
                fks_dropped.add((old_field.column,))
                self.execute(self._delete_fk_sql(model, fk_name))
        # Has unique been removed?
        if old_field.unique and (
            not new_field.unique or self._field_became_primary_key(old_field, new_field)
        ):
            # Find the unique constraint for this field
            meta_constraint_names = {
                constraint.name for constraint in model._meta.constraints
            }
            constraint_names = self._constraint_names(
                model,
                [old_field.column],
                unique=True,
                primary_key=False,
                exclude=meta_constraint_names,
            )
            if strict and len(constraint_names) != 1:
                raise ValueError(
                    "Found wrong number (%s) of unique constraints for %s.%s"
                    % (
                        len(constraint_names),
                        model._meta.db_table,
                        old_field.column,
                    )
                )
            for constraint_name in constraint_names:
                self.execute(self._delete_unique_sql(model, constraint_name))
        # Drop incoming FK constraints if the field is a primary key or unique,
        # which might be a to_field target, and things are going to change.
        old_collation = old_db_params.get("collation")
        new_collation = new_db_params.get("collation")
        drop_foreign_keys = (
            self.connection.features.supports_foreign_keys
            and (
                (old_field.primary_key and new_field.primary_key)
                or (old_field.unique and new_field.unique)
            )
            and ((old_type != new_type) or (old_collation != new_collation))
        )
        if drop_foreign_keys:
            # '_meta.related_field' also contains M2M reverse fields, these
            # will be filtered out
            for _old_rel, new_rel in _related_non_m2m_objects(old_field, new_field):
                rel_fk_names = self._constraint_names(
                    new_rel.related_model, [new_rel.field.column], foreign_key=True
                )
                for fk_name in rel_fk_names:
                    self.execute(self._delete_fk_sql(new_rel.related_model, fk_name))
        # Removed an index? (no strict check, as multiple indexes are possible)
        # Remove indexes if db_index switched to False or a unique constraint
        # will now be used in lieu of an index. The following lines from the
        # truth table show all True cases; the rest are False:
        #
        # old_field.db_index | old_field.unique | new_field.db_index | new_field.unique
        # ------------------------------------------------------------------------------
        # True               | False            | False              | False
        # True               | False            | False              | True
        # True               | False            | True               | True
        # ... other code
```
