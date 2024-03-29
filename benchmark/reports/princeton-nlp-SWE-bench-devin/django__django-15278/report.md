# django__django-15278

| **django/django** | `0ab58c120939093fea90822f376e1866fc714d1f` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 1228 |
| **Any found context length** | 1228 |
| **Avg pos** | 4.0 |
| **Min pos** | 4 |
| **Max pos** | 4 |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/db/backends/sqlite3/schema.py b/django/db/backends/sqlite3/schema.py
--- a/django/db/backends/sqlite3/schema.py
+++ b/django/db/backends/sqlite3/schema.py
@@ -324,10 +324,15 @@ def delete_model(self, model, handle_autom2m=True):
 
     def add_field(self, model, field):
         """Create a field on a model."""
-        # Fields with default values cannot by handled by ALTER TABLE ADD
-        # COLUMN statement because DROP DEFAULT is not supported in
-        # ALTER TABLE.
-        if not field.null or self.effective_default(field) is not None:
+        if (
+            # Primary keys and unique fields are not supported in ALTER TABLE
+            # ADD COLUMN.
+            field.primary_key or field.unique or
+            # Fields with default values cannot by handled by ALTER TABLE ADD
+            # COLUMN statement because DROP DEFAULT is not supported in
+            # ALTER TABLE.
+            not field.null or self.effective_default(field) is not None
+        ):
             self._remake_table(model, create_field=field)
         else:
             super().add_field(model, field)

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/db/backends/sqlite3/schema.py | 327 | 330 | 4 | 1 | 1228


## Problem Statement

```
Adding nullable OneToOneField crashes on SQLite.
Description
	
This new sqlite3 error has cropped up between building django-oauth-toolkit between Django 4.0 and main branch for migrations.AddField of a OneToOneField (see ​https://github.com/jazzband/django-oauth-toolkit/issues/1064):
self = <django.db.backends.sqlite3.base.SQLiteCursorWrapper object at 0x10b8038b0>
query = 'ALTER TABLE "oauth2_provider_accesstoken" ADD COLUMN "source_refresh_token_id" bigint NULL UNIQUE REFERENCES "oauth2_provider_refreshtoken" ("id") DEFERRABLE INITIALLY DEFERRED'
params = []
	def execute(self, query, params=None):
		if params is None:
			return Database.Cursor.execute(self, query)
		query = self.convert_query(query)
>	 return Database.Cursor.execute(self, query, params)
E	 django.db.utils.OperationalError: Cannot add a UNIQUE column
Here's the relevant migration snippet: 
		migrations.AddField(
			model_name='AccessToken',
			name='source_refresh_token',
			field=models.OneToOneField(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, to=oauth2_settings.REFRESH_TOKEN_MODEL, related_name="refreshed_access_token"),
		),
I see there have been a lot of sqlite3 changes in #33355 since the 4.0 release....

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 django/db/backends/sqlite3/schema.py** | 104 | 143| 505 | 505 | 4214 | 
| 2 | **1 django/db/backends/sqlite3/schema.py** | 89 | 102| 181 | 686 | 4214 | 
| 3 | **1 django/db/backends/sqlite3/schema.py** | 1 | 38| 329 | 1015 | 4214 | 
| **-> 4 <-** | **1 django/db/backends/sqlite3/schema.py** | 312 | 333| 213 | 1228 | 4214 | 
| 5 | 2 django/db/models/fields/related.py | 904 | 930| 240 | 1468 | 18373 | 
| 6 | 2 django/db/models/fields/related.py | 1067 | 1114| 368 | 1836 | 18373 | 
| 7 | 3 django/db/backends/sqlite3/operations.py | 1 | 41| 305 | 2141 | 21645 | 
| 8 | 4 django/db/backends/oracle/schema.py | 65 | 85| 249 | 2390 | 23817 | 
| 9 | 5 django/db/migrations/questioner.py | 160 | 179| 185 | 2575 | 26471 | 
| 10 | 5 django/db/backends/oracle/schema.py | 87 | 142| 715 | 3290 | 26471 | 
| 11 | 5 django/db/migrations/questioner.py | 57 | 87| 255 | 3545 | 26471 | 
| 12 | 5 django/db/models/fields/related.py | 881 | 902| 169 | 3714 | 26471 | 
| 13 | 6 django/db/backends/sqlite3/base.py | 170 | 227| 506 | 4220 | 29517 | 
| 14 | 6 django/db/migrations/questioner.py | 247 | 266| 195 | 4415 | 29517 | 
| 15 | 6 django/db/backends/sqlite3/operations.py | 330 | 387| 545 | 4960 | 29517 | 
| 16 | **6 django/db/backends/sqlite3/schema.py** | 353 | 387| 422 | 5382 | 29517 | 
| 17 | 7 django/db/backends/oracle/creation.py | 130 | 165| 399 | 5781 | 33410 | 
| 18 | 7 django/db/migrations/questioner.py | 181 | 205| 235 | 6016 | 33410 | 
| 19 | 8 django/db/backends/mysql/schema.py | 96 | 106| 138 | 6154 | 34984 | 
| 20 | 9 django/db/backends/sqlite3/features.py | 89 | 121| 241 | 6395 | 36057 | 
| 21 | 10 django/db/models/constraints.py | 198 | 216| 234 | 6629 | 38118 | 
| 22 | 10 django/db/backends/sqlite3/features.py | 54 | 87| 342 | 6971 | 38118 | 
| 23 | 10 django/db/models/fields/related.py | 538 | 603| 492 | 7463 | 38118 | 
| 24 | 10 django/db/models/fields/related.py | 1474 | 1515| 418 | 7881 | 38118 | 
| 25 | 10 django/db/backends/sqlite3/operations.py | 180 | 205| 190 | 8071 | 38118 | 
| 26 | 10 django/db/models/fields/related.py | 267 | 298| 284 | 8355 | 38118 | 
| 27 | 10 django/db/models/fields/related.py | 1281 | 1398| 963 | 9318 | 38118 | 
| 28 | 10 django/db/models/fields/related.py | 1024 | 1035| 128 | 9446 | 38118 | 
| 29 | 11 django/db/models/base.py | 947 | 981| 400 | 9846 | 55736 | 
| 30 | 11 django/db/backends/sqlite3/features.py | 1 | 52| 503 | 10349 | 55736 | 
| 31 | 11 django/db/models/constraints.py | 187 | 196| 130 | 10479 | 55736 | 
| 32 | 12 django/db/backends/base/schema.py | 1189 | 1211| 199 | 10678 | 68618 | 
| 33 | 12 django/db/backends/sqlite3/operations.py | 207 | 226| 209 | 10887 | 68618 | 
| 34 | 12 django/db/backends/base/schema.py | 684 | 754| 799 | 11686 | 68618 | 
| 35 | 12 django/db/models/fields/related.py | 139 | 166| 201 | 11887 | 68618 | 
| 36 | **12 django/db/backends/sqlite3/schema.py** | 228 | 310| 731 | 12618 | 68618 | 
| 37 | 12 django/db/backends/sqlite3/operations.py | 246 | 275| 198 | 12816 | 68618 | 
| 38 | 12 django/db/backends/base/schema.py | 834 | 874| 506 | 13322 | 68618 | 
| 39 | 13 django/db/backends/sqlite3/creation.py | 23 | 49| 239 | 13561 | 69469 | 
| 40 | 13 django/db/backends/base/schema.py | 755 | 833| 826 | 14387 | 69469 | 
| 41 | 14 django/db/backends/oracle/features.py | 1 | 125| 1113 | 15500 | 70583 | 
| 42 | 14 django/db/models/fields/related.py | 808 | 879| 549 | 16049 | 70583 | 
| 43 | 15 django/db/models/fields/__init__.py | 2381 | 2431| 339 | 16388 | 88835 | 
| 44 | 15 django/db/backends/base/schema.py | 1172 | 1187| 170 | 16558 | 88835 | 
| 45 | **15 django/db/backends/sqlite3/schema.py** | 389 | 422| 358 | 16916 | 88835 | 
| 46 | 15 django/db/models/fields/related.py | 975 | 988| 126 | 17042 | 88835 | 
| 47 | 15 django/db/backends/oracle/schema.py | 144 | 153| 142 | 17184 | 88835 | 
| 48 | 15 django/db/backends/base/schema.py | 1151 | 1170| 175 | 17359 | 88835 | 
| 49 | **15 django/db/backends/sqlite3/schema.py** | 424 | 448| 162 | 17521 | 88835 | 
| 50 | 16 django/db/migrations/operations/fields.py | 93 | 105| 130 | 17651 | 91328 | 
| 51 | 16 django/db/models/fields/related.py | 1037 | 1064| 216 | 17867 | 91328 | 
| 52 | 16 django/db/models/fields/related.py | 1249 | 1279| 172 | 18039 | 91328 | 
| 53 | 16 django/db/migrations/questioner.py | 227 | 245| 177 | 18216 | 91328 | 
| 54 | 16 django/db/models/fields/related.py | 300 | 334| 293 | 18509 | 91328 | 
| 55 | 16 django/db/backends/base/schema.py | 415 | 429| 183 | 18692 | 91328 | 
| 56 | 16 django/db/backends/base/schema.py | 1246 | 1287| 357 | 19049 | 91328 | 
| 57 | 17 django/db/backends/oracle/operations.py | 377 | 414| 369 | 19418 | 97349 | 
| 58 | 18 django/db/models/deletion.py | 1 | 75| 561 | 19979 | 101179 | 
| 59 | 18 django/db/models/fields/related.py | 787 | 805| 222 | 20201 | 101179 | 
| 60 | 18 django/db/backends/mysql/schema.py | 1 | 39| 428 | 20629 | 101179 | 
| 61 | 18 django/db/models/fields/related.py | 1400 | 1472| 616 | 21245 | 101179 | 
| 62 | 18 django/db/models/base.py | 1580 | 1612| 231 | 21476 | 101179 | 
| 63 | 19 django/db/backends/sqlite3/introspection.py | 230 | 258| 278 | 21754 | 104414 | 
| 64 | 19 django/db/backends/sqlite3/base.py | 49 | 123| 775 | 22529 | 104414 | 
| 65 | 19 django/db/backends/base/schema.py | 1213 | 1244| 233 | 22762 | 104414 | 
| 66 | 20 django/db/models/fields/related_descriptors.py | 344 | 363| 156 | 22918 | 115019 | 
| 67 | 20 django/db/backends/oracle/schema.py | 1 | 26| 284 | 23202 | 115019 | 
| 68 | 20 django/db/models/fields/related.py | 120 | 137| 155 | 23357 | 115019 | 
| 69 | 20 django/db/backends/base/schema.py | 1 | 29| 209 | 23566 | 115019 | 
| 70 | 20 django/db/models/fields/related.py | 183 | 196| 140 | 23706 | 115019 | 
| 71 | 21 django/db/models/sql/query.py | 1725 | 1767| 436 | 24142 | 137689 | 
| 72 | 21 django/db/backends/base/schema.py | 1289 | 1308| 163 | 24305 | 137689 | 
| 73 | 21 django/db/models/fields/related_descriptors.py | 309 | 323| 182 | 24487 | 137689 | 
| 74 | 21 django/db/backends/oracle/operations.py | 485 | 504| 240 | 24727 | 137689 | 
| 75 | 21 django/db/models/base.py | 1 | 50| 328 | 25055 | 137689 | 
| 76 | 22 django/db/models/options.py | 285 | 317| 331 | 25386 | 145051 | 
| 77 | 22 django/db/models/base.py | 1195 | 1210| 138 | 25524 | 145051 | 
| 78 | 22 django/db/backends/sqlite3/operations.py | 43 | 69| 232 | 25756 | 145051 | 
| 79 | 22 django/db/models/fields/related.py | 953 | 973| 178 | 25934 | 145051 | 
| 80 | 22 django/db/models/fields/related.py | 605 | 638| 334 | 26268 | 145051 | 
| 81 | 22 django/db/migrations/operations/fields.py | 107 | 117| 127 | 26395 | 145051 | 
| 82 | 23 django/db/backends/postgresql/schema.py | 101 | 182| 647 | 27042 | 147219 | 
| 83 | **23 django/db/backends/sqlite3/schema.py** | 145 | 226| 820 | 27862 | 147219 | 
| 84 | 23 django/db/models/fields/__init__.py | 2119 | 2152| 228 | 28090 | 147219 | 
| 85 | 23 django/db/migrations/operations/fields.py | 217 | 227| 146 | 28236 | 147219 | 
| 86 | 23 django/db/models/fields/related.py | 198 | 266| 687 | 28923 | 147219 | 
| 87 | 23 django/db/models/fields/related.py | 1659 | 1700| 497 | 29420 | 147219 | 
| 88 | 23 django/db/models/fields/__init__.py | 308 | 336| 205 | 29625 | 147219 | 
| 89 | 23 django/db/backends/sqlite3/creation.py | 51 | 82| 317 | 29942 | 147219 | 
| 90 | 23 django/db/models/fields/__init__.py | 1797 | 1824| 215 | 30157 | 147219 | 
| 91 | 24 django/db/backends/oracle/base.py | 60 | 99| 328 | 30485 | 152287 | 
| 92 | 25 django/db/models/expressions.py | 1 | 30| 207 | 30692 | 163634 | 
| 93 | 25 django/db/models/fields/__init__.py | 2434 | 2484| 313 | 31005 | 163634 | 
| 94 | 25 django/db/models/fields/related.py | 516 | 536| 138 | 31143 | 163634 | 
| 95 | 26 django/db/backends/sqlite3/_functions.py | 71 | 78| 155 | 31298 | 167405 | 
| 96 | 26 django/db/models/fields/__init__.py | 2515 | 2540| 143 | 31441 | 167405 | 
| 97 | **26 django/db/backends/sqlite3/schema.py** | 40 | 66| 243 | 31684 | 167405 | 
| 98 | 27 django/db/models/fields/related_lookups.py | 130 | 166| 244 | 31928 | 168876 | 
| 99 | 28 django/db/models/fields/json.py | 42 | 65| 155 | 32083 | 173013 | 
| 100 | 28 django/db/backends/oracle/operations.py | 589 | 604| 221 | 32304 | 173013 | 
| 101 | 28 django/db/backends/base/schema.py | 1364 | 1396| 293 | 32597 | 173013 | 
| 102 | 28 django/db/backends/base/schema.py | 491 | 546| 613 | 33210 | 173013 | 
| 103 | 28 django/db/models/fields/related.py | 1517 | 1551| 356 | 33566 | 173013 | 
| 104 | 28 django/db/models/fields/related.py | 168 | 181| 144 | 33710 | 173013 | 
| 105 | 28 django/db/backends/sqlite3/base.py | 229 | 318| 850 | 34560 | 173013 | 
| 106 | 29 django/db/migrations/autodetector.py | 939 | 1022| 919 | 35479 | 185128 | 
| 107 | 29 django/db/backends/postgresql/schema.py | 184 | 210| 351 | 35830 | 185128 | 
| 108 | 29 django/db/migrations/autodetector.py | 874 | 916| 394 | 36224 | 185128 | 
| 109 | 29 django/db/backends/oracle/operations.py | 156 | 173| 309 | 36533 | 185128 | 
| 110 | 29 django/db/models/fields/related.py | 640 | 657| 197 | 36730 | 185128 | 


### Hint

```
Thanks for the report! Regression in 2f73e5406d54cb8945e187eff302a3a3373350be.
```

## Patch

```diff
diff --git a/django/db/backends/sqlite3/schema.py b/django/db/backends/sqlite3/schema.py
--- a/django/db/backends/sqlite3/schema.py
+++ b/django/db/backends/sqlite3/schema.py
@@ -324,10 +324,15 @@ def delete_model(self, model, handle_autom2m=True):
 
     def add_field(self, model, field):
         """Create a field on a model."""
-        # Fields with default values cannot by handled by ALTER TABLE ADD
-        # COLUMN statement because DROP DEFAULT is not supported in
-        # ALTER TABLE.
-        if not field.null or self.effective_default(field) is not None:
+        if (
+            # Primary keys and unique fields are not supported in ALTER TABLE
+            # ADD COLUMN.
+            field.primary_key or field.unique or
+            # Fields with default values cannot by handled by ALTER TABLE ADD
+            # COLUMN statement because DROP DEFAULT is not supported in
+            # ALTER TABLE.
+            not field.null or self.effective_default(field) is not None
+        ):
             self._remake_table(model, create_field=field)
         else:
             super().add_field(model, field)

```

## Test Patch

```diff
diff --git a/tests/schema/tests.py b/tests/schema/tests.py
--- a/tests/schema/tests.py
+++ b/tests/schema/tests.py
@@ -624,6 +624,18 @@ def get_prep_value(self, value):
         # Make sure the values were transformed correctly
         self.assertEqual(Author.objects.extra(where=["thing = 1"]).count(), 2)
 
+    def test_add_field_o2o_nullable(self):
+        with connection.schema_editor() as editor:
+            editor.create_model(Author)
+            editor.create_model(Note)
+        new_field = OneToOneField(Note, CASCADE, null=True)
+        new_field.set_attributes_from_name('note')
+        with connection.schema_editor() as editor:
+            editor.add_field(Author, new_field)
+        columns = self.column_classes(Author)
+        self.assertIn('note_id', columns)
+        self.assertTrue(columns['note_id'][1][6])
+
     def test_add_field_binary(self):
         """
         Tests binary fields get a sane default (#22851)

```


## Code snippets

### 1 - django/db/backends/sqlite3/schema.py:

Start line: 104, End line: 143

```python
class DatabaseSchemaEditor(BaseDatabaseSchemaEditor):

    def alter_field(self, model, old_field, new_field, strict=False):
        if not self._field_should_be_altered(old_field, new_field):
            return
        old_field_name = old_field.name
        table_name = model._meta.db_table
        _, old_column_name = old_field.get_attname_column()
        if (new_field.name != old_field_name and
                not self.connection.features.supports_atomic_references_rename and
                self._is_referenced_by_fk_constraint(table_name, old_column_name, ignore_self=True)):
            if self.connection.in_atomic_block:
                raise NotSupportedError((
                    'Renaming the %r.%r column while in a transaction is not '
                    'supported on SQLite < 3.26 because it would break referential '
                    'integrity. Try adding `atomic = False` to the Migration class.'
                ) % (model._meta.db_table, old_field_name))
            with atomic(self.connection.alias):
                super().alter_field(model, old_field, new_field, strict=strict)
                # Follow SQLite's documented procedure for performing changes
                # that don't affect the on-disk content.
                # https://sqlite.org/lang_altertable.html#otheralter
                with self.connection.cursor() as cursor:
                    schema_version = cursor.execute('PRAGMA schema_version').fetchone()[0]
                    cursor.execute('PRAGMA writable_schema = 1')
                    references_template = ' REFERENCES "%s" ("%%s") ' % table_name
                    new_column_name = new_field.get_attname_column()[1]
                    search = references_template % old_column_name
                    replacement = references_template % new_column_name
                    cursor.execute('UPDATE sqlite_master SET sql = replace(sql, %s, %s)', (search, replacement))
                    cursor.execute('PRAGMA schema_version = %d' % (schema_version + 1))
                    cursor.execute('PRAGMA writable_schema = 0')
                    # The integrity check will raise an exception and rollback
                    # the transaction if the sqlite_master updates corrupt the
                    # database.
                    cursor.execute('PRAGMA integrity_check')
            # Perform a VACUUM to refresh the database representation from
            # the sqlite_master table.
            with self.connection.cursor() as cursor:
                cursor.execute('VACUUM')
        else:
            super().alter_field(model, old_field, new_field, strict=strict)
```
### 2 - django/db/backends/sqlite3/schema.py:

Start line: 89, End line: 102

```python
class DatabaseSchemaEditor(BaseDatabaseSchemaEditor):

    def alter_db_table(self, model, old_db_table, new_db_table, disable_constraints=True):
        if (not self.connection.features.supports_atomic_references_rename and
                disable_constraints and self._is_referenced_by_fk_constraint(old_db_table)):
            if self.connection.in_atomic_block:
                raise NotSupportedError((
                    'Renaming the %r table while in a transaction is not '
                    'supported on SQLite < 3.26 because it would break referential '
                    'integrity. Try adding `atomic = False` to the Migration class.'
                ) % old_db_table)
            self.connection.enable_constraint_checking()
            super().alter_db_table(model, old_db_table, new_db_table)
            self.connection.disable_constraint_checking()
        else:
            super().alter_db_table(model, old_db_table, new_db_table)
```
### 3 - django/db/backends/sqlite3/schema.py:

Start line: 1, End line: 38

```python
import copy
from decimal import Decimal

from django.apps.registry import Apps
from django.db import NotSupportedError
from django.db.backends.base.schema import BaseDatabaseSchemaEditor
from django.db.backends.ddl_references import Statement
from django.db.backends.utils import strip_quotes
from django.db.models import UniqueConstraint
from django.db.transaction import atomic


class DatabaseSchemaEditor(BaseDatabaseSchemaEditor):

    sql_delete_table = "DROP TABLE %(table)s"
    sql_create_fk = None
    sql_create_inline_fk = "REFERENCES %(to_table)s (%(to_column)s) DEFERRABLE INITIALLY DEFERRED"
    sql_create_column_inline_fk = sql_create_inline_fk
    sql_create_unique = "CREATE UNIQUE INDEX %(name)s ON %(table)s (%(columns)s)"
    sql_delete_unique = "DROP INDEX %(name)s"

    def __enter__(self):
        # Some SQLite schema alterations need foreign key constraints to be
        # disabled. Enforce it here for the duration of the schema edition.
        if not self.connection.disable_constraint_checking():
            raise NotSupportedError(
                'SQLite schema editor cannot be used while foreign key '
                'constraint checks are enabled. Make sure to disable them '
                'before entering a transaction.atomic() context because '
                'SQLite does not support disabling them in the middle of '
                'a multi-statement transaction.'
            )
        return super().__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        self.connection.check_constraints()
        super().__exit__(exc_type, exc_value, traceback)
        self.connection.enable_constraint_checking()
```
### 4 - django/db/backends/sqlite3/schema.py:

Start line: 312, End line: 333

```python
class DatabaseSchemaEditor(BaseDatabaseSchemaEditor):

    def delete_model(self, model, handle_autom2m=True):
        if handle_autom2m:
            super().delete_model(model)
        else:
            # Delete the table (and only that)
            self.execute(self.sql_delete_table % {
                "table": self.quote_name(model._meta.db_table),
            })
            # Remove all deferred statements referencing the deleted table.
            for sql in list(self.deferred_sql):
                if isinstance(sql, Statement) and sql.references_table(model._meta.db_table):
                    self.deferred_sql.remove(sql)

    def add_field(self, model, field):
        """Create a field on a model."""
        # Fields with default values cannot by handled by ALTER TABLE ADD
        # COLUMN statement because DROP DEFAULT is not supported in
        # ALTER TABLE.
        if not field.null or self.effective_default(field) is not None:
            self._remake_table(model, create_field=field)
        else:
            super().add_field(model, field)
```
### 5 - django/db/models/fields/related.py:

Start line: 904, End line: 930

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
### 6 - django/db/models/fields/related.py:

Start line: 1067, End line: 1114

```python
class OneToOneField(ForeignKey):
    """
    A OneToOneField is essentially the same as a ForeignKey, with the exception
    that it always carries a "unique" constraint with it and the reverse
    relation always returns the object pointed to (since there will only ever
    be one), rather than returning a list.
    """

    # Field flags
    many_to_many = False
    many_to_one = False
    one_to_many = False
    one_to_one = True

    related_accessor_class = ReverseOneToOneDescriptor
    forward_related_accessor_class = ForwardOneToOneDescriptor
    rel_class = OneToOneRel

    description = _("One-to-one relationship")

    def __init__(self, to, on_delete, to_field=None, **kwargs):
        kwargs['unique'] = True
        super().__init__(to, on_delete, to_field=to_field, **kwargs)

    def deconstruct(self):
        name, path, args, kwargs = super().deconstruct()
        if "unique" in kwargs:
            del kwargs['unique']
        return name, path, args, kwargs

    def formfield(self, **kwargs):
        if self.remote_field.parent_link:
            return None
        return super().formfield(**kwargs)

    def save_form_data(self, instance, data):
        if isinstance(data, self.remote_field.model):
            setattr(instance, self.name, data)
        else:
            setattr(instance, self.attname, data)
            # Remote field object must be cleared otherwise Model.save()
            # will reassign attname using the related object pk.
            if data is None:
                setattr(instance, self.name, data)

    def _check_unique(self, **kwargs):
        # Override ForeignKey since check isn't applicable here.
        return []
```
### 7 - django/db/backends/sqlite3/operations.py:

Start line: 1, End line: 41

```python
import datetime
import decimal
import uuid
from functools import lru_cache
from itertools import chain

from django.conf import settings
from django.core.exceptions import FieldError
from django.db import DatabaseError, NotSupportedError, models
from django.db.backends.base.operations import BaseDatabaseOperations
from django.db.models.expressions import Col
from django.utils import timezone
from django.utils.dateparse import parse_date, parse_datetime, parse_time
from django.utils.functional import cached_property


class DatabaseOperations(BaseDatabaseOperations):
    cast_char_field_without_max_length = 'text'
    cast_data_types = {
        'DateField': 'TEXT',
        'DateTimeField': 'TEXT',
    }
    explain_prefix = 'EXPLAIN QUERY PLAN'
    # List of datatypes to that cannot be extracted with JSON_EXTRACT() on
    # SQLite. Use JSON_TYPE() instead.
    jsonfield_datatype_values = frozenset(['null', 'false', 'true'])

    def bulk_batch_size(self, fields, objs):
        """
        SQLite has a compile-time default (SQLITE_LIMIT_VARIABLE_NUMBER) of
        999 variables per query.

        If there's only a single field to insert, the limit is 500
        (SQLITE_MAX_COMPOUND_SELECT).
        """
        if len(fields) == 1:
            return 500
        elif len(fields) > 1:
            return self.connection.features.max_query_params // len(fields)
        else:
            return len(objs)
```
### 8 - django/db/backends/oracle/schema.py:

Start line: 65, End line: 85

```python
class DatabaseSchemaEditor(BaseDatabaseSchemaEditor):

    def alter_field(self, model, old_field, new_field, strict=False):
        try:
            super().alter_field(model, old_field, new_field, strict)
        except DatabaseError as e:
            description = str(e)
            # If we're changing type to an unsupported type we need a
            # SQLite-ish workaround
            if 'ORA-22858' in description or 'ORA-22859' in description:
                self._alter_field_type_workaround(model, old_field, new_field)
            # If an identity column is changing to a non-numeric type, drop the
            # identity first.
            elif 'ORA-30675' in description:
                self._drop_identity(model._meta.db_table, old_field.column)
                self.alter_field(model, old_field, new_field, strict)
            # If a primary key column is changing to an identity column, drop
            # the primary key first.
            elif 'ORA-30673' in description and old_field.primary_key:
                self._delete_primary_key(model, strict=True)
                self._alter_field_type_workaround(model, old_field, new_field)
            else:
                raise
```
### 9 - django/db/migrations/questioner.py:

Start line: 160, End line: 179

```python
class InteractiveMigrationQuestioner(MigrationQuestioner):

    def ask_not_null_addition(self, field_name, model_name):
        """Adding a NOT NULL field to a model."""
        if not self.dry_run:
            choice = self._choice_input(
                f"It is impossible to add a non-nullable field '{field_name}' "
                f"to {model_name} without specifying a default. This is "
                f"because the database needs something to populate existing "
                f"rows.\n"
                f"Please select a fix:",
                [
                    ("Provide a one-off default now (will be set on all existing "
                     "rows with a null value for this column)"),
                    'Quit and manually define a default value in models.py.',
                ]
            )
            if choice == 2:
                sys.exit(3)
            else:
                return self._ask_default()
        return None
```
### 10 - django/db/backends/oracle/schema.py:

Start line: 87, End line: 142

```python
class DatabaseSchemaEditor(BaseDatabaseSchemaEditor):

    def _alter_field_type_workaround(self, model, old_field, new_field):
        """
        Oracle refuses to change from some type to other type.
        What we need to do instead is:
        - Add a nullable version of the desired field with a temporary name. If
          the new column is an auto field, then the temporary column can't be
          nullable.
        - Update the table to transfer values from old to new
        - Drop old column
        - Rename the new column and possibly drop the nullable property
        """
        # Make a new field that's like the new one but with a temporary
        # column name.
        new_temp_field = copy.deepcopy(new_field)
        new_temp_field.null = (new_field.get_internal_type() not in ('AutoField', 'BigAutoField', 'SmallAutoField'))
        new_temp_field.column = self._generate_temp_name(new_field.column)
        # Add it
        self.add_field(model, new_temp_field)
        # Explicit data type conversion
        # https://docs.oracle.com/en/database/oracle/oracle-database/18/sqlrf
        # /Data-Type-Comparison-Rules.html#GUID-D0C5A47E-6F93-4C2D-9E49-4F2B86B359DD
        new_value = self.quote_name(old_field.column)
        old_type = old_field.db_type(self.connection)
        if re.match('^N?CLOB', old_type):
            new_value = "TO_CHAR(%s)" % new_value
            old_type = 'VARCHAR2'
        if re.match('^N?VARCHAR2', old_type):
            new_internal_type = new_field.get_internal_type()
            if new_internal_type == 'DateField':
                new_value = "TO_DATE(%s, 'YYYY-MM-DD')" % new_value
            elif new_internal_type == 'DateTimeField':
                new_value = "TO_TIMESTAMP(%s, 'YYYY-MM-DD HH24:MI:SS.FF')" % new_value
            elif new_internal_type == 'TimeField':
                # TimeField are stored as TIMESTAMP with a 1900-01-01 date part.
                new_value = "TO_TIMESTAMP(CONCAT('1900-01-01 ', %s), 'YYYY-MM-DD HH24:MI:SS.FF')" % new_value
        # Transfer values across
        self.execute("UPDATE %s set %s=%s" % (
            self.quote_name(model._meta.db_table),
            self.quote_name(new_temp_field.column),
            new_value,
        ))
        # Drop the old field
        self.remove_field(model, old_field)
        # Rename and possibly make the new field NOT NULL
        super().alter_field(model, new_temp_field, new_field)
        # Recreate foreign key (if necessary) because the old field is not
        # passed to the alter_field() and data types of new_temp_field and
        # new_field always match.
        new_type = new_field.db_type(self.connection)
        if (
            (old_field.primary_key and new_field.primary_key) or
            (old_field.unique and new_field.unique)
        ) and old_type != new_type:
            for _, rel in _related_non_m2m_objects(new_temp_field, new_field):
                if rel.field.db_constraint:
                    self.execute(self._create_fk_sql(rel.related_model, rel.field, '_fk'))
```
### 16 - django/db/backends/sqlite3/schema.py:

Start line: 353, End line: 387

```python
class DatabaseSchemaEditor(BaseDatabaseSchemaEditor):

    def _alter_field(self, model, old_field, new_field, old_type, new_type,
                     old_db_params, new_db_params, strict=False):
        """Perform a "physical" (non-ManyToMany) field update."""
        # Use "ALTER TABLE ... RENAME COLUMN" if only the column name
        # changed and there aren't any constraints.
        if (self.connection.features.can_alter_table_rename_column and
            old_field.column != new_field.column and
            self.column_sql(model, old_field) == self.column_sql(model, new_field) and
            not (old_field.remote_field and old_field.db_constraint or
                 new_field.remote_field and new_field.db_constraint)):
            return self.execute(self._rename_field_sql(model._meta.db_table, old_field, new_field, new_type))
        # Alter by remaking table
        self._remake_table(model, alter_field=(old_field, new_field))
        # Rebuild tables with FKs pointing to this field.
        if new_field.unique and old_type != new_type:
            related_models = set()
            opts = new_field.model._meta
            for remote_field in opts.related_objects:
                # Ignore self-relationship since the table was already rebuilt.
                if remote_field.related_model == model:
                    continue
                if not remote_field.many_to_many:
                    if remote_field.field_name == new_field.name:
                        related_models.add(remote_field.related_model)
                elif new_field.primary_key and remote_field.through._meta.auto_created:
                    related_models.add(remote_field.through)
            if new_field.primary_key:
                for many_to_many in opts.many_to_many:
                    # Ignore self-relationship since the table was already rebuilt.
                    if many_to_many.related_model == model:
                        continue
                    if many_to_many.remote_field.through._meta.auto_created:
                        related_models.add(many_to_many.remote_field.through)
            for related_model in related_models:
                self._remake_table(related_model)
```
### 36 - django/db/backends/sqlite3/schema.py:

Start line: 228, End line: 310

```python
class DatabaseSchemaEditor(BaseDatabaseSchemaEditor):

    def _remake_table(self, model, create_field=None, delete_field=None, alter_field=None):

        # Work out the new value for index_together, taking renames into
        # account
        index_together = [
            [rename_mapping.get(n, n) for n in index]
            for index in model._meta.index_together
        ]

        indexes = model._meta.indexes
        if delete_field:
            indexes = [
                index for index in indexes
                if delete_field.name not in index.fields
            ]

        constraints = list(model._meta.constraints)

        # Provide isolated instances of the fields to the new model body so
        # that the existing model's internals aren't interfered with when
        # the dummy model is constructed.
        body_copy = copy.deepcopy(body)

        # Construct a new model with the new fields to allow self referential
        # primary key to resolve to. This model won't ever be materialized as a
        # table and solely exists for foreign key reference resolution purposes.
        # This wouldn't be required if the schema editor was operating on model
        # states instead of rendered models.
        meta_contents = {
            'app_label': model._meta.app_label,
            'db_table': model._meta.db_table,
            'unique_together': unique_together,
            'index_together': index_together,
            'indexes': indexes,
            'constraints': constraints,
            'apps': apps,
        }
        meta = type("Meta", (), meta_contents)
        body_copy['Meta'] = meta
        body_copy['__module__'] = model.__module__
        type(model._meta.object_name, model.__bases__, body_copy)

        # Construct a model with a renamed table name.
        body_copy = copy.deepcopy(body)
        meta_contents = {
            'app_label': model._meta.app_label,
            'db_table': 'new__%s' % strip_quotes(model._meta.db_table),
            'unique_together': unique_together,
            'index_together': index_together,
            'indexes': indexes,
            'constraints': constraints,
            'apps': apps,
        }
        meta = type("Meta", (), meta_contents)
        body_copy['Meta'] = meta
        body_copy['__module__'] = model.__module__
        new_model = type('New%s' % model._meta.object_name, model.__bases__, body_copy)

        # Create a new table with the updated schema.
        self.create_model(new_model)

        # Copy data from the old table into the new table
        self.execute("INSERT INTO %s (%s) SELECT %s FROM %s" % (
            self.quote_name(new_model._meta.db_table),
            ', '.join(self.quote_name(x) for x in mapping),
            ', '.join(mapping.values()),
            self.quote_name(model._meta.db_table),
        ))

        # Delete the old table to make way for the new
        self.delete_model(model, handle_autom2m=False)

        # Rename the new table to take way for the old
        self.alter_db_table(
            new_model, new_model._meta.db_table, model._meta.db_table,
            disable_constraints=False,
        )

        # Run deferred SQL on correct table
        for sql in self.deferred_sql:
            self.execute(sql)
        self.deferred_sql = []
        # Fix any PK-removed field
        if restore_pk_field:
            restore_pk_field.primary_key = True
```
### 45 - django/db/backends/sqlite3/schema.py:

Start line: 389, End line: 422

```python
class DatabaseSchemaEditor(BaseDatabaseSchemaEditor):

    def _alter_many_to_many(self, model, old_field, new_field, strict):
        """Alter M2Ms to repoint their to= endpoints."""
        if old_field.remote_field.through._meta.db_table == new_field.remote_field.through._meta.db_table:
            # The field name didn't change, but some options did; we have to propagate this altering.
            self._remake_table(
                old_field.remote_field.through,
                alter_field=(
                    # We need the field that points to the target model, so we can tell alter_field to change it -
                    # this is m2m_reverse_field_name() (as opposed to m2m_field_name, which points to our model)
                    old_field.remote_field.through._meta.get_field(old_field.m2m_reverse_field_name()),
                    new_field.remote_field.through._meta.get_field(new_field.m2m_reverse_field_name()),
                ),
            )
            return

        # Make a new through table
        self.create_model(new_field.remote_field.through)
        # Copy the data across
        self.execute("INSERT INTO %s (%s) SELECT %s FROM %s" % (
            self.quote_name(new_field.remote_field.through._meta.db_table),
            ', '.join([
                "id",
                new_field.m2m_column_name(),
                new_field.m2m_reverse_name(),
            ]),
            ', '.join([
                "id",
                old_field.m2m_column_name(),
                old_field.m2m_reverse_name(),
            ]),
            self.quote_name(old_field.remote_field.through._meta.db_table),
        ))
        # Delete the old through table
        self.delete_model(old_field.remote_field.through)
```
### 49 - django/db/backends/sqlite3/schema.py:

Start line: 424, End line: 448

```python
class DatabaseSchemaEditor(BaseDatabaseSchemaEditor):

    def add_constraint(self, model, constraint):
        if isinstance(constraint, UniqueConstraint) and (
            constraint.condition or
            constraint.contains_expressions or
            constraint.include or
            constraint.deferrable
        ):
            super().add_constraint(model, constraint)
        else:
            self._remake_table(model)

    def remove_constraint(self, model, constraint):
        if isinstance(constraint, UniqueConstraint) and (
            constraint.condition or
            constraint.contains_expressions or
            constraint.include or
            constraint.deferrable
        ):
            super().remove_constraint(model, constraint)
        else:
            self._remake_table(model)

    def _collate_sql(self, collation):
        return 'COLLATE ' + collation
```
### 83 - django/db/backends/sqlite3/schema.py:

Start line: 145, End line: 226

```python
class DatabaseSchemaEditor(BaseDatabaseSchemaEditor):

    def _remake_table(self, model, create_field=None, delete_field=None, alter_field=None):
        """
        Shortcut to transform a model from old_model into new_model

        This follows the correct procedure to perform non-rename or column
        addition operations based on SQLite's documentation

        https://www.sqlite.org/lang_altertable.html#caution

        The essential steps are:
          1. Create a table with the updated definition called "new__app_model"
          2. Copy the data from the existing "app_model" table to the new table
          3. Drop the "app_model" table
          4. Rename the "new__app_model" table to "app_model"
          5. Restore any index of the previous "app_model" table.
        """
        # Self-referential fields must be recreated rather than copied from
        # the old model to ensure their remote_field.field_name doesn't refer
        # to an altered field.
        def is_self_referential(f):
            return f.is_relation and f.remote_field.model is model
        # Work out the new fields dict / mapping
        body = {
            f.name: f.clone() if is_self_referential(f) else f
            for f in model._meta.local_concrete_fields
        }
        # Since mapping might mix column names and default values,
        # its values must be already quoted.
        mapping = {f.column: self.quote_name(f.column) for f in model._meta.local_concrete_fields}
        # This maps field names (not columns) for things like unique_together
        rename_mapping = {}
        # If any of the new or altered fields is introducing a new PK,
        # remove the old one
        restore_pk_field = None
        if getattr(create_field, 'primary_key', False) or (
                alter_field and getattr(alter_field[1], 'primary_key', False)):
            for name, field in list(body.items()):
                if field.primary_key:
                    field.primary_key = False
                    restore_pk_field = field
                    if field.auto_created:
                        del body[name]
                        del mapping[field.column]
        # Add in any created fields
        if create_field:
            body[create_field.name] = create_field
            # Choose a default and insert it into the copy map
            if not create_field.many_to_many and create_field.concrete:
                mapping[create_field.column] = self.prepare_default(
                    self.effective_default(create_field),
                )
        # Add in any altered fields
        if alter_field:
            old_field, new_field = alter_field
            body.pop(old_field.name, None)
            mapping.pop(old_field.column, None)
            body[new_field.name] = new_field
            if old_field.null and not new_field.null:
                case_sql = "coalesce(%(col)s, %(default)s)" % {
                    'col': self.quote_name(old_field.column),
                    'default': self.prepare_default(self.effective_default(new_field)),
                }
                mapping[new_field.column] = case_sql
            else:
                mapping[new_field.column] = self.quote_name(old_field.column)
            rename_mapping[old_field.name] = new_field.name
        # Remove any deleted fields
        if delete_field:
            del body[delete_field.name]
            del mapping[delete_field.column]
            # Remove any implicit M2M tables
            if delete_field.many_to_many and delete_field.remote_field.through._meta.auto_created:
                return self.delete_model(delete_field.remote_field.through)
        # Work inside a new app registry
        apps = Apps()

        # Work out the new value of unique_together, taking renames into
        # account
        unique_together = [
            [rename_mapping.get(n, n) for n in unique]
            for unique in model._meta.unique_together
        ]
        # ... other code
```
### 97 - django/db/backends/sqlite3/schema.py:

Start line: 40, End line: 66

```python
class DatabaseSchemaEditor(BaseDatabaseSchemaEditor):

    def quote_value(self, value):
        # The backend "mostly works" without this function and there are use
        # cases for compiling Python without the sqlite3 libraries (e.g.
        # security hardening).
        try:
            import sqlite3
            value = sqlite3.adapt(value)
        except ImportError:
            pass
        except sqlite3.ProgrammingError:
            pass
        # Manual emulation of SQLite parameter quoting
        if isinstance(value, bool):
            return str(int(value))
        elif isinstance(value, (Decimal, float, int)):
            return str(value)
        elif isinstance(value, str):
            return "'%s'" % value.replace("\'", "\'\'")
        elif value is None:
            return "NULL"
        elif isinstance(value, (bytes, bytearray, memoryview)):
            # Bytes are only allowed for BLOB fields, encoded as string
            # literals containing hexadecimal data and preceded by a single "X"
            # character.
            return "X'%s'" % value.hex()
        else:
            raise ValueError("Cannot quote parameter value %r of type %s" % (value, type(value)))
```
