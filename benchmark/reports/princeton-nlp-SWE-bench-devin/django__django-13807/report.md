# django__django-13807

| **django/django** | `89fc144dedc737a79929231438f035b1d4a993c9` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 1013 |
| **Any found context length** | 1013 |
| **Avg pos** | 2.0 |
| **Min pos** | 2 |
| **Max pos** | 2 |
| **Top file pos** | 2 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/db/backends/sqlite3/base.py b/django/db/backends/sqlite3/base.py
--- a/django/db/backends/sqlite3/base.py
+++ b/django/db/backends/sqlite3/base.py
@@ -327,19 +327,24 @@ def check_constraints(self, table_names=None):
                     violations = cursor.execute('PRAGMA foreign_key_check').fetchall()
                 else:
                     violations = chain.from_iterable(
-                        cursor.execute('PRAGMA foreign_key_check(%s)' % table_name).fetchall()
+                        cursor.execute(
+                            'PRAGMA foreign_key_check(%s)'
+                            % self.ops.quote_name(table_name)
+                        ).fetchall()
                         for table_name in table_names
                     )
                 # See https://www.sqlite.org/pragma.html#pragma_foreign_key_check
                 for table_name, rowid, referenced_table_name, foreign_key_index in violations:
                     foreign_key = cursor.execute(
-                        'PRAGMA foreign_key_list(%s)' % table_name
+                        'PRAGMA foreign_key_list(%s)' % self.ops.quote_name(table_name)
                     ).fetchall()[foreign_key_index]
                     column_name, referenced_column_name = foreign_key[3:5]
                     primary_key_column_name = self.introspection.get_primary_key_column(cursor, table_name)
                     primary_key_value, bad_value = cursor.execute(
                         'SELECT %s, %s FROM %s WHERE rowid = %%s' % (
-                            primary_key_column_name, column_name, table_name
+                            self.ops.quote_name(primary_key_column_name),
+                            self.ops.quote_name(column_name),
+                            self.ops.quote_name(table_name),
                         ),
                         (rowid,),
                     ).fetchone()

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/db/backends/sqlite3/base.py | 330 | 342 | 2 | 2 | 1013


## Problem Statement

```
loaddata crashes on SQLite when table names are SQL keywords.
Description
	
Steps to reproduce:
Create a Model called Order. (order is a SQL reserved word)
Create fixtures for the model
Use manage.py loaddata to load the fixture.
Notice that it fails with the following error. This is because the table name order is not quoted properly
(0.000) PRAGMA foreign_key_check(order); args=None
Traceback (most recent call last):
 File "python3.7/site-packages/django/db/backends/utils.py", line 82, in _execute
	return self.cursor.execute(sql)
 File "python3.7/site-packages/django/db/backends/sqlite3/base.py", line 411, in execute
	return Database.Cursor.execute(self, query)
sqlite3.OperationalError: near "order": syntax error
Root Cause
File: python3.7/site-packages/django/db/backends/sqlite3/base.py line 327
Function: check_constraints
Details: due to missing back ticks around %s in the SQL statement PRAGMA foreign_key_check(%s)
Here in check_constraints line 327 in context
				if table_names is None:
					violations = cursor.execute('PRAGMA foreign_key_check').fetchall()
				else:
					violations = chain.from_iterable(
						cursor.execute('PRAGMA foreign_key_check(%s)' % table_name).fetchall()
						for table_name in table_names
					)
And here line 333
				for table_name, rowid, referenced_table_name, foreign_key_index in violations:
					foreign_key = cursor.execute(
						'PRAGMA foreign_key_list(%s)' % table_name
					).fetchall()[foreign_key_index]
Issue confirmed in
3.1.0
3.1.2

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 django/db/backends/sqlite3/operations.py | 170 | 195| 190 | 190 | 3090 | 
| **-> 2 <-** | **2 django/db/backends/sqlite3/base.py** | 316 | 400| 823 | 1013 | 9107 | 
| 3 | 3 django/core/management/commands/loaddata.py | 69 | 85| 187 | 1200 | 12044 | 
| 4 | 3 django/core/management/commands/loaddata.py | 87 | 157| 640 | 1840 | 12044 | 
| 5 | 4 django/db/backends/sqlite3/features.py | 1 | 114| 1027 | 2867 | 13071 | 
| 6 | 5 django/db/backends/sqlite3/schema.py | 1 | 37| 317 | 3184 | 17227 | 
| 7 | 5 django/db/backends/sqlite3/schema.py | 86 | 99| 181 | 3365 | 17227 | 
| 8 | 6 django/db/models/base.py | 1511 | 1533| 171 | 3536 | 34085 | 
| 9 | 6 django/db/models/base.py | 1739 | 1839| 729 | 4265 | 34085 | 
| 10 | **6 django/db/backends/sqlite3/base.py** | 81 | 155| 757 | 5022 | 34085 | 
| 11 | 6 django/core/management/commands/loaddata.py | 38 | 67| 261 | 5283 | 34085 | 
| 12 | 7 django/db/models/__init__.py | 1 | 53| 619 | 5902 | 34704 | 
| 13 | 8 django/db/backends/sqlite3/introspection.py | 225 | 239| 146 | 6048 | 38746 | 
| 14 | 8 django/db/backends/sqlite3/schema.py | 101 | 140| 505 | 6553 | 38746 | 
| 15 | 9 django/db/backends/oracle/operations.py | 369 | 406| 369 | 6922 | 44717 | 
| 16 | 9 django/db/models/base.py | 1486 | 1509| 176 | 7098 | 44717 | 
| 17 | 10 django/db/models/sql/compiler.py | 1 | 19| 170 | 7268 | 59108 | 
| 18 | 10 django/db/backends/sqlite3/introspection.py | 331 | 359| 278 | 7546 | 59108 | 
| 19 | 10 django/db/models/base.py | 1265 | 1296| 267 | 7813 | 59108 | 
| 20 | 10 django/db/models/base.py | 1841 | 1914| 572 | 8385 | 59108 | 
| 21 | 11 django/db/backends/mysql/base.py | 290 | 328| 402 | 8787 | 62484 | 
| 22 | 11 django/db/models/base.py | 1932 | 2063| 976 | 9763 | 62484 | 
| 23 | 11 django/db/backends/sqlite3/operations.py | 197 | 216| 209 | 9972 | 62484 | 
| 24 | 11 django/db/models/base.py | 2066 | 2117| 351 | 10323 | 62484 | 
| 25 | 12 django/db/migrations/operations/__init__.py | 1 | 18| 195 | 10518 | 62679 | 
| 26 | 12 django/db/backends/sqlite3/operations.py | 1 | 38| 258 | 10776 | 62679 | 
| 27 | 12 django/db/backends/sqlite3/operations.py | 218 | 234| 142 | 10918 | 62679 | 
| 28 | 12 django/db/backends/oracle/operations.py | 477 | 496| 240 | 11158 | 62679 | 
| 29 | 13 django/db/backends/oracle/creation.py | 130 | 165| 399 | 11557 | 66572 | 
| 30 | 14 django/db/backends/base/features.py | 1 | 112| 895 | 12452 | 69493 | 
| 31 | 14 django/db/backends/sqlite3/introspection.py | 57 | 78| 218 | 12670 | 69493 | 
| 32 | 15 django/db/backends/oracle/features.py | 1 | 124| 1048 | 13718 | 70541 | 
| 33 | 16 django/db/backends/mysql/operations.py | 219 | 276| 431 | 14149 | 74245 | 
| 34 | 16 django/db/backends/sqlite3/introspection.py | 440 | 463| 183 | 14332 | 74245 | 
| 35 | 16 django/db/backends/oracle/operations.py | 333 | 344| 227 | 14559 | 74245 | 
| 36 | 17 django/db/backends/mysql/schema.py | 1 | 38| 409 | 14968 | 75767 | 
| 37 | **17 django/db/backends/sqlite3/base.py** | 403 | 423| 183 | 15151 | 75767 | 
| 38 | 17 django/db/models/sql/compiler.py | 1047 | 1087| 337 | 15488 | 75767 | 
| 39 | 17 django/db/models/base.py | 1535 | 1567| 231 | 15719 | 75767 | 
| 40 | 17 django/db/backends/oracle/operations.py | 21 | 73| 574 | 16293 | 75767 | 
| 41 | 17 django/db/models/base.py | 1 | 50| 328 | 16621 | 75767 | 
| 42 | 17 django/db/models/sql/compiler.py | 22 | 47| 257 | 16878 | 75767 | 
| 43 | 17 django/db/backends/sqlite3/introspection.py | 23 | 54| 296 | 17174 | 75767 | 
| 44 | 18 django/db/models/options.py | 1 | 35| 300 | 17474 | 83134 | 
| 45 | 18 django/db/models/base.py | 911 | 943| 383 | 17857 | 83134 | 
| 46 | 19 django/core/exceptions.py | 107 | 218| 752 | 18609 | 84323 | 
| 47 | **19 django/db/backends/sqlite3/base.py** | 268 | 314| 419 | 19028 | 84323 | 
| 48 | 19 django/db/models/options.py | 252 | 287| 341 | 19369 | 84323 | 
| 49 | 20 django/db/backends/base/schema.py | 1238 | 1260| 173 | 19542 | 96676 | 
| 50 | 21 django/db/backends/postgresql/introspection.py | 214 | 235| 208 | 19750 | 99010 | 
| 51 | 22 django/db/backends/oracle/schema.py | 138 | 198| 544 | 20294 | 101045 | 
| 52 | **22 django/db/backends/sqlite3/base.py** | 1 | 78| 499 | 20793 | 101045 | 
| 53 | 23 django/core/management/commands/dumpdata.py | 179 | 203| 224 | 21017 | 102655 | 
| 54 | 23 django/db/backends/oracle/operations.py | 461 | 475| 203 | 21220 | 102655 | 
| 55 | 24 django/db/backends/mysql/features.py | 99 | 184| 741 | 21961 | 104483 | 
| 56 | 24 django/db/backends/base/features.py | 113 | 216| 833 | 22794 | 104483 | 
| 57 | 24 django/db/models/sql/compiler.py | 365 | 405| 482 | 23276 | 104483 | 
| 58 | 24 django/db/backends/base/features.py | 217 | 318| 864 | 24140 | 104483 | 
| 59 | 24 django/db/backends/oracle/creation.py | 253 | 281| 277 | 24417 | 104483 | 
| 60 | 25 django/db/models/sql/query.py | 1487 | 1572| 801 | 25218 | 127029 | 
| 61 | 25 django/db/backends/sqlite3/schema.py | 225 | 307| 731 | 25949 | 127029 | 
| 62 | 25 django/core/management/commands/dumpdata.py | 67 | 139| 624 | 26573 | 127029 | 
| 63 | 25 django/db/backends/sqlite3/schema.py | 39 | 65| 243 | 26816 | 127029 | 
| 64 | 26 django/db/backends/postgresql/operations.py | 160 | 187| 311 | 27127 | 129590 | 
| 65 | 26 django/db/backends/oracle/operations.py | 1 | 18| 141 | 27268 | 129590 | 
| 66 | 26 django/db/backends/base/schema.py | 1126 | 1148| 199 | 27467 | 129590 | 
| 67 | 26 django/db/backends/base/schema.py | 1292 | 1324| 292 | 27759 | 129590 | 
| 68 | 27 django/db/backends/sqlite3/creation.py | 23 | 49| 239 | 27998 | 130441 | 
| 69 | 27 django/db/backends/base/schema.py | 1182 | 1217| 291 | 28289 | 130441 | 
| 70 | 27 django/db/backends/oracle/operations.py | 408 | 459| 516 | 28805 | 130441 | 
| 71 | 27 django/db/models/sql/compiler.py | 407 | 415| 133 | 28938 | 130441 | 
| 72 | 28 django/db/backends/mysql/introspection.py | 292 | 308| 184 | 29122 | 132993 | 
| 73 | 28 django/db/backends/mysql/features.py | 53 | 97| 419 | 29541 | 132993 | 
| 74 | **28 django/db/backends/sqlite3/base.py** | 156 | 173| 216 | 29757 | 132993 | 
| 75 | 29 django/core/management/commands/sqlmigrate.py | 31 | 69| 379 | 30136 | 133626 | 
| 76 | 29 django/db/backends/mysql/introspection.py | 1 | 17| 121 | 30257 | 133626 | 
| 77 | 29 django/db/backends/postgresql/operations.py | 189 | 276| 696 | 30953 | 133626 | 
| 78 | 30 django/db/backends/postgresql/features.py | 1 | 110| 883 | 31836 | 134509 | 
| 79 | 31 django/db/backends/postgresql/schema.py | 1 | 67| 626 | 32462 | 136668 | 
| 80 | 31 django/db/models/sql/compiler.py | 1341 | 1400| 617 | 33079 | 136668 | 
| 81 | 31 django/db/models/base.py | 1689 | 1737| 348 | 33427 | 136668 | 
| 82 | 31 django/db/backends/oracle/schema.py | 1 | 41| 427 | 33854 | 136668 | 
| 83 | 32 django/db/migrations/autodetector.py | 1191 | 1216| 245 | 34099 | 148287 | 
| 84 | 33 django/db/backends/postgresql/creation.py | 1 | 34| 238 | 34337 | 148933 | 
| 85 | 33 django/db/backends/oracle/operations.py | 598 | 617| 303 | 34640 | 148933 | 
| 86 | 34 django/core/management/commands/inspectdb.py | 38 | 173| 1291 | 35931 | 151566 | 
| 87 | 34 django/db/backends/base/schema.py | 44 | 111| 769 | 36700 | 151566 | 
| 88 | 35 django/db/backends/oracle/base.py | 60 | 99| 328 | 37028 | 156662 | 
| 89 | 36 django/db/backends/mysql/validation.py | 1 | 31| 239 | 37267 | 157182 | 
| 90 | 36 django/db/backends/base/schema.py | 1150 | 1180| 214 | 37481 | 157182 | 
| 91 | 36 django/core/management/commands/loaddata.py | 1 | 35| 177 | 37658 | 157182 | 
| 92 | 36 django/db/models/base.py | 1083 | 1126| 404 | 38062 | 157182 | 
| 93 | 36 django/db/backends/sqlite3/operations.py | 40 | 66| 232 | 38294 | 157182 | 
| 94 | 36 django/db/backends/oracle/operations.py | 257 | 271| 231 | 38525 | 157182 | 
| 95 | 36 django/db/backends/oracle/creation.py | 187 | 218| 319 | 38844 | 157182 | 
| 96 | 36 django/db/backends/oracle/operations.py | 581 | 596| 221 | 39065 | 157182 | 
| 97 | **36 django/db/backends/sqlite3/base.py** | 588 | 610| 143 | 39208 | 157182 | 
| 98 | 36 django/db/models/base.py | 1298 | 1320| 186 | 39394 | 157182 | 
| 99 | 36 django/db/models/base.py | 1623 | 1687| 514 | 39908 | 157182 | 
| 100 | 37 django/db/models/fields/related.py | 127 | 154| 201 | 40109 | 171058 | 
| 101 | 37 django/db/models/sql/query.py | 1 | 66| 479 | 40588 | 171058 | 
| 102 | 37 django/db/backends/base/schema.py | 1219 | 1236| 142 | 40730 | 171058 | 
| 103 | 37 django/db/backends/sqlite3/creation.py | 1 | 21| 140 | 40870 | 171058 | 
| 104 | 37 django/core/management/commands/loaddata.py | 159 | 224| 583 | 41453 | 171058 | 
| 105 | 38 django/contrib/admin/views/main.py | 340 | 400| 508 | 41961 | 175454 | 
| 106 | 38 django/db/backends/mysql/operations.py | 1 | 35| 282 | 42243 | 175454 | 
| 107 | 38 django/db/models/sql/compiler.py | 1402 | 1420| 203 | 42446 | 175454 | 
| 108 | 38 django/db/models/base.py | 1128 | 1155| 286 | 42732 | 175454 | 
| 109 | 39 django/db/migrations/operations/models.py | 619 | 636| 163 | 42895 | 182379 | 
| 110 | 39 django/db/backends/oracle/base.py | 102 | 155| 626 | 43521 | 182379 | 
| 111 | 40 django/db/backends/base/operations.py | 674 | 694| 187 | 43708 | 187975 | 
| 112 | 40 django/db/models/base.py | 1157 | 1172| 138 | 43846 | 187975 | 


### Hint

```
Thanks for the report, I was able to reproduce this issue with db_table = 'order'. Reproduced at 966b5b49b6521483f1c90b4499c4c80e80136de3.
Simply wrapping table_name in connection.ops.quote_name should address the issue for anyone interested in picking the issue up.
a little guidance needed as this is my first ticket.
will the issue be fixed if i just wrap %s around back ticks -> %s as in 'PRAGMA foreign_key_check(%s)' and 'PRAGMA foreign_key_list(%s)' % table_name?
Nayan, Have you seen Simon's comment? We should wrap with quote_name().
"Details: due to missing back ticks around %s in the SQL statement PRAGMA foreign_key_check(%s)" But it is quoted that this should fix the issue.
shall i wrap "table_name" in "quote_name" as in "quote_name(table_name)"?
shall i wrap "table_name" in "quote_name" as in "quote_name(table_name)"? yes, self.ops.quote_name(table_name)
First contribution, currently trying to understand the code to figure out how to write a regression test for this. Any help in how to unit test this is appreciated.
I believe I got it. Will put my test in tests/fixtures_regress which has a lot of regression tests for loaddata already, creating a new Order model and a fixture for it.
Suggested test improvements to avoid the creation of another model.
Things have been busy for me but I'm still on it, I'll do the changes to the patch later this week.
Since this issue hasn't received any activity recently, may I assign it to myself?
Sure, feel-free.
Patch awaiting review. PR: ​https://github.com/django/django/pull/13807
```

## Patch

```diff
diff --git a/django/db/backends/sqlite3/base.py b/django/db/backends/sqlite3/base.py
--- a/django/db/backends/sqlite3/base.py
+++ b/django/db/backends/sqlite3/base.py
@@ -327,19 +327,24 @@ def check_constraints(self, table_names=None):
                     violations = cursor.execute('PRAGMA foreign_key_check').fetchall()
                 else:
                     violations = chain.from_iterable(
-                        cursor.execute('PRAGMA foreign_key_check(%s)' % table_name).fetchall()
+                        cursor.execute(
+                            'PRAGMA foreign_key_check(%s)'
+                            % self.ops.quote_name(table_name)
+                        ).fetchall()
                         for table_name in table_names
                     )
                 # See https://www.sqlite.org/pragma.html#pragma_foreign_key_check
                 for table_name, rowid, referenced_table_name, foreign_key_index in violations:
                     foreign_key = cursor.execute(
-                        'PRAGMA foreign_key_list(%s)' % table_name
+                        'PRAGMA foreign_key_list(%s)' % self.ops.quote_name(table_name)
                     ).fetchall()[foreign_key_index]
                     column_name, referenced_column_name = foreign_key[3:5]
                     primary_key_column_name = self.introspection.get_primary_key_column(cursor, table_name)
                     primary_key_value, bad_value = cursor.execute(
                         'SELECT %s, %s FROM %s WHERE rowid = %%s' % (
-                            primary_key_column_name, column_name, table_name
+                            self.ops.quote_name(primary_key_column_name),
+                            self.ops.quote_name(column_name),
+                            self.ops.quote_name(table_name),
                         ),
                         (rowid,),
                     ).fetchone()

```

## Test Patch

```diff
diff --git a/tests/backends/models.py b/tests/backends/models.py
--- a/tests/backends/models.py
+++ b/tests/backends/models.py
@@ -140,3 +140,11 @@ class Author(models.Model):
 
 class Book(models.Model):
     author = models.ForeignKey(Author, models.CASCADE, to_field='name')
+
+
+class SQLKeywordsModel(models.Model):
+    id = models.AutoField(primary_key=True, db_column='select')
+    reporter = models.ForeignKey(Reporter, models.CASCADE, db_column='where')
+
+    class Meta:
+        db_table = 'order'
diff --git a/tests/backends/tests.py b/tests/backends/tests.py
--- a/tests/backends/tests.py
+++ b/tests/backends/tests.py
@@ -20,7 +20,7 @@
 
 from .models import (
     Article, Object, ObjectReference, Person, Post, RawData, Reporter,
-    ReporterProxy, SchoolClass, Square,
+    ReporterProxy, SchoolClass, SQLKeywordsModel, Square,
     VeryLongModelNameZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ,
 )
 
@@ -625,6 +625,17 @@ def test_check_constraints(self):
                     connection.check_constraints()
             transaction.set_rollback(True)
 
+    def test_check_constraints_sql_keywords(self):
+        with transaction.atomic():
+            obj = SQLKeywordsModel.objects.create(reporter=self.r)
+            obj.refresh_from_db()
+            obj.reporter_id = 30
+            with connection.constraint_checks_disabled():
+                obj.save()
+                with self.assertRaises(IntegrityError):
+                    connection.check_constraints(table_names=['order'])
+            transaction.set_rollback(True)
+
 
 class ThreadTests(TransactionTestCase):
 

```


## Code snippets

### 1 - django/db/backends/sqlite3/operations.py:

Start line: 170, End line: 195

```python
class DatabaseOperations(BaseDatabaseOperations):

    def quote_name(self, name):
        if name.startswith('"') and name.endswith('"'):
            return name  # Quoting once is enough.
        return '"%s"' % name

    def no_limit_value(self):
        return -1

    def __references_graph(self, table_name):
        query = """
        WITH tables AS (
            SELECT %s name
            UNION
            SELECT sqlite_master.name
            FROM sqlite_master
            JOIN tables ON (sql REGEXP %s || tables.name || %s)
        ) SELECT name FROM tables;
        """
        params = (
            table_name,
            r'(?i)\s+references\s+("|\')?',
            r'("|\')?\s*\(',
        )
        with self.connection.cursor() as cursor:
            results = cursor.execute(query, params)
            return [row[0] for row in results.fetchall()]
```
### 2 - django/db/backends/sqlite3/base.py:

Start line: 316, End line: 400

```python
class DatabaseWrapper(BaseDatabaseWrapper):

    def check_constraints(self, table_names=None):
        """
        Check each table name in `table_names` for rows with invalid foreign
        key references. This method is intended to be used in conjunction with
        `disable_constraint_checking()` and `enable_constraint_checking()`, to
        determine if rows with invalid references were entered while constraint
        checks were off.
        """
        if self.features.supports_pragma_foreign_key_check:
            with self.cursor() as cursor:
                if table_names is None:
                    violations = cursor.execute('PRAGMA foreign_key_check').fetchall()
                else:
                    violations = chain.from_iterable(
                        cursor.execute('PRAGMA foreign_key_check(%s)' % table_name).fetchall()
                        for table_name in table_names
                    )
                # See https://www.sqlite.org/pragma.html#pragma_foreign_key_check
                for table_name, rowid, referenced_table_name, foreign_key_index in violations:
                    foreign_key = cursor.execute(
                        'PRAGMA foreign_key_list(%s)' % table_name
                    ).fetchall()[foreign_key_index]
                    column_name, referenced_column_name = foreign_key[3:5]
                    primary_key_column_name = self.introspection.get_primary_key_column(cursor, table_name)
                    primary_key_value, bad_value = cursor.execute(
                        'SELECT %s, %s FROM %s WHERE rowid = %%s' % (
                            primary_key_column_name, column_name, table_name
                        ),
                        (rowid,),
                    ).fetchone()
                    raise IntegrityError(
                        "The row in table '%s' with primary key '%s' has an "
                        "invalid foreign key: %s.%s contains a value '%s' that "
                        "does not have a corresponding value in %s.%s." % (
                            table_name, primary_key_value, table_name, column_name,
                            bad_value, referenced_table_name, referenced_column_name
                        )
                    )
        else:
            with self.cursor() as cursor:
                if table_names is None:
                    table_names = self.introspection.table_names(cursor)
                for table_name in table_names:
                    primary_key_column_name = self.introspection.get_primary_key_column(cursor, table_name)
                    if not primary_key_column_name:
                        continue
                    key_columns = self.introspection.get_key_columns(cursor, table_name)
                    for column_name, referenced_table_name, referenced_column_name in key_columns:
                        cursor.execute(
                            """
                            SELECT REFERRING.`%s`, REFERRING.`%s` FROM `%s` as REFERRING
                            LEFT JOIN `%s` as REFERRED
                            ON (REFERRING.`%s` = REFERRED.`%s`)
                            WHERE REFERRING.`%s` IS NOT NULL AND REFERRED.`%s` IS NULL
                            """
                            % (
                                primary_key_column_name, column_name, table_name,
                                referenced_table_name, column_name, referenced_column_name,
                                column_name, referenced_column_name,
                            )
                        )
                        for bad_row in cursor.fetchall():
                            raise IntegrityError(
                                "The row in table '%s' with primary key '%s' has an "
                                "invalid foreign key: %s.%s contains a value '%s' that "
                                "does not have a corresponding value in %s.%s." % (
                                    table_name, bad_row[0], table_name, column_name,
                                    bad_row[1], referenced_table_name, referenced_column_name,
                                )
                            )

    def is_usable(self):
        return True

    def _start_transaction_under_autocommit(self):
        """
        Start a transaction explicitly in autocommit mode.

        Staying in autocommit mode works around a bug of sqlite3 that breaks
        savepoints when autocommit is disabled.
        """
        self.cursor().execute("BEGIN")

    def is_in_memory_db(self):
        return self.creation.is_in_memory_db(self.settings_dict['NAME'])
```
### 3 - django/core/management/commands/loaddata.py:

Start line: 69, End line: 85

```python
class Command(BaseCommand):

    def handle(self, *fixture_labels, **options):
        self.ignore = options['ignore']
        self.using = options['database']
        self.app_label = options['app_label']
        self.verbosity = options['verbosity']
        self.excluded_models, self.excluded_apps = parse_apps_and_model_labels(options['exclude'])
        self.format = options['format']

        with transaction.atomic(using=self.using):
            self.loaddata(fixture_labels)

        # Close the DB connection -- unless we're still in a transaction. This
        # is required as a workaround for an edge case in MySQL: if the same
        # connection is used to create tables, load data, and query, the query
        # can return incorrect results. See Django #7572, MySQL #37735.
        if transaction.get_autocommit(self.using):
            connections[self.using].close()
```
### 4 - django/core/management/commands/loaddata.py:

Start line: 87, End line: 157

```python
class Command(BaseCommand):

    def loaddata(self, fixture_labels):
        connection = connections[self.using]

        # Keep a count of the installed objects and fixtures
        self.fixture_count = 0
        self.loaded_object_count = 0
        self.fixture_object_count = 0
        self.models = set()

        self.serialization_formats = serializers.get_public_serializer_formats()
        # Forcing binary mode may be revisited after dropping Python 2 support (see #22399)
        self.compression_formats = {
            None: (open, 'rb'),
            'gz': (gzip.GzipFile, 'rb'),
            'zip': (SingleZipReader, 'r'),
            'stdin': (lambda *args: sys.stdin, None),
        }
        if has_bz2:
            self.compression_formats['bz2'] = (bz2.BZ2File, 'r')
        if has_lzma:
            self.compression_formats['lzma'] = (lzma.LZMAFile, 'r')
            self.compression_formats['xz'] = (lzma.LZMAFile, 'r')

        # Django's test suite repeatedly tries to load initial_data fixtures
        # from apps that don't have any fixtures. Because disabling constraint
        # checks can be expensive on some database (especially MSSQL), bail
        # out early if no fixtures are found.
        for fixture_label in fixture_labels:
            if self.find_fixtures(fixture_label):
                break
        else:
            return

        with connection.constraint_checks_disabled():
            self.objs_with_deferred_fields = []
            for fixture_label in fixture_labels:
                self.load_label(fixture_label)
            for obj in self.objs_with_deferred_fields:
                obj.save_deferred_fields(using=self.using)

        # Since we disabled constraint checks, we must manually check for
        # any invalid keys that might have been added
        table_names = [model._meta.db_table for model in self.models]
        try:
            connection.check_constraints(table_names=table_names)
        except Exception as e:
            e.args = ("Problem installing fixtures: %s" % e,)
            raise

        # If we found even one object in a fixture, we need to reset the
        # database sequences.
        if self.loaded_object_count > 0:
            sequence_sql = connection.ops.sequence_reset_sql(no_style(), self.models)
            if sequence_sql:
                if self.verbosity >= 2:
                    self.stdout.write('Resetting sequences')
                with connection.cursor() as cursor:
                    for line in sequence_sql:
                        cursor.execute(line)

        if self.verbosity >= 1:
            if self.fixture_object_count == self.loaded_object_count:
                self.stdout.write(
                    "Installed %d object(s) from %d fixture(s)"
                    % (self.loaded_object_count, self.fixture_count)
                )
            else:
                self.stdout.write(
                    "Installed %d object(s) (of %d) from %d fixture(s)"
                    % (self.loaded_object_count, self.fixture_object_count, self.fixture_count)
                )
```
### 5 - django/db/backends/sqlite3/features.py:

Start line: 1, End line: 114

```python
import operator
import platform

from django.db import transaction
from django.db.backends.base.features import BaseDatabaseFeatures
from django.db.utils import OperationalError
from django.utils.functional import cached_property

from .base import Database


class DatabaseFeatures(BaseDatabaseFeatures):
    # SQLite can read from a cursor since SQLite 3.6.5, subject to the caveat
    # that statements within a connection aren't isolated from each other. See
    # https://sqlite.org/isolation.html.
    can_use_chunked_reads = True
    test_db_allows_multiple_connections = False
    supports_unspecified_pk = True
    supports_timezones = False
    max_query_params = 999
    supports_mixed_date_datetime_comparisons = False
    supports_transactions = True
    atomic_transactions = False
    can_rollback_ddl = True
    can_create_inline_fk = False
    supports_paramstyle_pyformat = False
    can_clone_databases = True
    supports_temporal_subtraction = True
    ignores_table_name_case = True
    supports_cast_with_precision = False
    time_cast_precision = 3
    can_release_savepoints = True
    # Is "ALTER TABLE ... RENAME COLUMN" supported?
    can_alter_table_rename_column = Database.sqlite_version_info >= (3, 25, 0)
    supports_parentheses_in_compound = False
    # Deferred constraint checks can be emulated on SQLite < 3.20 but not in a
    # reasonably performant way.
    supports_pragma_foreign_key_check = Database.sqlite_version_info >= (3, 20, 0)
    can_defer_constraint_checks = supports_pragma_foreign_key_check
    supports_functions_in_partial_indexes = Database.sqlite_version_info >= (3, 15, 0)
    supports_over_clause = Database.sqlite_version_info >= (3, 25, 0)
    supports_frame_range_fixed_distance = Database.sqlite_version_info >= (3, 28, 0)
    supports_aggregate_filter_clause = Database.sqlite_version_info >= (3, 30, 1)
    supports_order_by_nulls_modifier = Database.sqlite_version_info >= (3, 30, 0)
    order_by_nulls_first = True
    supports_json_field_contains = False
    test_collations = {
        'ci': 'nocase',
        'cs': 'binary',
        'non_default': 'nocase',
    }

    @cached_property
    def django_test_skips(self):
        skips = {
            'SQLite stores values rounded to 15 significant digits.': {
                'model_fields.test_decimalfield.DecimalFieldTests.test_fetch_from_db_without_float_rounding',
            },
            'SQLite naively remakes the table on field alteration.': {
                'schema.tests.SchemaTests.test_unique_no_unnecessary_fk_drops',
                'schema.tests.SchemaTests.test_unique_and_reverse_m2m',
                'schema.tests.SchemaTests.test_alter_field_default_doesnt_perform_queries',
                'schema.tests.SchemaTests.test_rename_column_renames_deferred_sql_references',
            },
            "SQLite doesn't have a constraint.": {
                'model_fields.test_integerfield.PositiveIntegerFieldTests.test_negative_values',
            },
        }
        if Database.sqlite_version_info < (3, 27):
            skips.update({
                'Nondeterministic failure on SQLite < 3.27.': {
                    'expressions_window.tests.WindowFunctionTests.test_subquery_row_range_rank',
                },
            })
        if self.connection.is_in_memory_db():
            skips.update({
                "the sqlite backend's close() method is a no-op when using an "
                "in-memory database": {
                    'servers.test_liveserverthread.LiveServerThreadTest.test_closes_connections',
                },
            })
        return skips

    @cached_property
    def supports_atomic_references_rename(self):
        # SQLite 3.28.0 bundled with MacOS 10.15 does not support renaming
        # references atomically.
        if platform.mac_ver()[0].startswith('10.15.') and Database.sqlite_version_info == (3, 28, 0):
            return False
        return Database.sqlite_version_info >= (3, 26, 0)

    @cached_property
    def introspected_field_types(self):
        return{
            **super().introspected_field_types,
            'BigAutoField': 'AutoField',
            'DurationField': 'BigIntegerField',
            'GenericIPAddressField': 'CharField',
            'SmallAutoField': 'AutoField',
        }

    @cached_property
    def supports_json_field(self):
        with self.connection.cursor() as cursor:
            try:
                with transaction.atomic(self.connection.alias):
                    cursor.execute('SELECT JSON(\'{"a": "b"}\')')
            except OperationalError:
                return False
        return True

    can_introspect_json_field = property(operator.attrgetter('supports_json_field'))
    has_json_object_function = property(operator.attrgetter('supports_json_field'))
```
### 6 - django/db/backends/sqlite3/schema.py:

Start line: 1, End line: 37

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
### 7 - django/db/backends/sqlite3/schema.py:

Start line: 86, End line: 99

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
### 8 - django/db/models/base.py:

Start line: 1511, End line: 1533

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
### 9 - django/db/models/base.py:

Start line: 1739, End line: 1839

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
### 10 - django/db/backends/sqlite3/base.py:

Start line: 81, End line: 155

```python
class DatabaseWrapper(BaseDatabaseWrapper):
    vendor = 'sqlite'
    display_name = 'SQLite'
    # SQLite doesn't actually support most of these types, but it "does the right
    # thing" given more verbose field definitions, so leave them as is so that
    # schema inspection is more useful.
    data_types = {
        'AutoField': 'integer',
        'BigAutoField': 'integer',
        'BinaryField': 'BLOB',
        'BooleanField': 'bool',
        'CharField': 'varchar(%(max_length)s)',
        'DateField': 'date',
        'DateTimeField': 'datetime',
        'DecimalField': 'decimal',
        'DurationField': 'bigint',
        'FileField': 'varchar(%(max_length)s)',
        'FilePathField': 'varchar(%(max_length)s)',
        'FloatField': 'real',
        'IntegerField': 'integer',
        'BigIntegerField': 'bigint',
        'IPAddressField': 'char(15)',
        'GenericIPAddressField': 'char(39)',
        'JSONField': 'text',
        'NullBooleanField': 'bool',
        'OneToOneField': 'integer',
        'PositiveBigIntegerField': 'bigint unsigned',
        'PositiveIntegerField': 'integer unsigned',
        'PositiveSmallIntegerField': 'smallint unsigned',
        'SlugField': 'varchar(%(max_length)s)',
        'SmallAutoField': 'integer',
        'SmallIntegerField': 'smallint',
        'TextField': 'text',
        'TimeField': 'time',
        'UUIDField': 'char(32)',
    }
    data_type_check_constraints = {
        'PositiveBigIntegerField': '"%(column)s" >= 0',
        'JSONField': '(JSON_VALID("%(column)s") OR "%(column)s" IS NULL)',
        'PositiveIntegerField': '"%(column)s" >= 0',
        'PositiveSmallIntegerField': '"%(column)s" >= 0',
    }
    data_types_suffix = {
        'AutoField': 'AUTOINCREMENT',
        'BigAutoField': 'AUTOINCREMENT',
        'SmallAutoField': 'AUTOINCREMENT',
    }
    # SQLite requires LIKE statements to include an ESCAPE clause if the value
    # being escaped has a percent or underscore in it.
    # See https://www.sqlite.org/lang_expr.html for an explanation.
    operators = {
        'exact': '= %s',
        'iexact': "LIKE %s ESCAPE '\\'",
        'contains': "LIKE %s ESCAPE '\\'",
        'icontains': "LIKE %s ESCAPE '\\'",
        'regex': 'REGEXP %s',
        'iregex': "REGEXP '(?i)' || %s",
        'gt': '> %s',
        'gte': '>= %s',
        'lt': '< %s',
        'lte': '<= %s',
        'startswith': "LIKE %s ESCAPE '\\'",
        'endswith': "LIKE %s ESCAPE '\\'",
        'istartswith': "LIKE %s ESCAPE '\\'",
        'iendswith': "LIKE %s ESCAPE '\\'",
    }

    # The patterns below are used to generate SQL pattern lookup clauses when
    # the right-hand side of the lookup isn't a raw string (it might be an expression
    # or the result of a bilateral transformation).
    # In those cases, special characters for LIKE operators (e.g. \, *, _) should be
    # escaped on database side.
    #
    # Note: we use str.format() here for readability as '%' is used as a wildcard for
    # the LIKE operator.
```
### 37 - django/db/backends/sqlite3/base.py:

Start line: 403, End line: 423

```python
FORMAT_QMARK_REGEX = _lazy_re_compile(r'(?<!%)%s')


class SQLiteCursorWrapper(Database.Cursor):
    """
    Django uses "format" style placeholders, but pysqlite2 uses "qmark" style.
    This fixes it -- but note that if you want to use a literal "%s" in a query,
    you'll need to use "%%s".
    """
    def execute(self, query, params=None):
        if params is None:
            return Database.Cursor.execute(self, query)
        query = self.convert_query(query)
        return Database.Cursor.execute(self, query, params)

    def executemany(self, query, param_list):
        query = self.convert_query(query)
        return Database.Cursor.executemany(self, query, param_list)

    def convert_query(self, query):
        return FORMAT_QMARK_REGEX.sub('?', query).replace('%%', '%')
```
### 47 - django/db/backends/sqlite3/base.py:

Start line: 268, End line: 314

```python
class DatabaseWrapper(BaseDatabaseWrapper):

    def init_connection_state(self):
        pass

    def create_cursor(self, name=None):
        return self.connection.cursor(factory=SQLiteCursorWrapper)

    @async_unsafe
    def close(self):
        self.validate_thread_sharing()
        # If database is in memory, closing the connection destroys the
        # database. To prevent accidental data loss, ignore close requests on
        # an in-memory db.
        if not self.is_in_memory_db():
            BaseDatabaseWrapper.close(self)

    def _savepoint_allowed(self):
        # When 'isolation_level' is not None, sqlite3 commits before each
        # savepoint; it's a bug. When it is None, savepoints don't make sense
        # because autocommit is enabled. The only exception is inside 'atomic'
        # blocks. To work around that bug, on SQLite, 'atomic' starts a
        # transaction explicitly rather than simply disable autocommit.
        return self.in_atomic_block

    def _set_autocommit(self, autocommit):
        if autocommit:
            level = None
        else:
            # sqlite3's internal default is ''. It's different from None.
            # See Modules/_sqlite/connection.c.
            level = ''
        # 'isolation_level' is a misleading API.
        # SQLite always runs at the SERIALIZABLE isolation level.
        with self.wrap_database_errors:
            self.connection.isolation_level = level

    def disable_constraint_checking(self):
        with self.cursor() as cursor:
            cursor.execute('PRAGMA foreign_keys = OFF')
            # Foreign key constraints cannot be turned off while in a multi-
            # statement transaction. Fetch the current state of the pragma
            # to determine if constraints are effectively disabled.
            enabled = cursor.execute('PRAGMA foreign_keys').fetchone()[0]
        return not bool(enabled)

    def enable_constraint_checking(self):
        with self.cursor() as cursor:
            cursor.execute('PRAGMA foreign_keys = ON')
```
### 52 - django/db/backends/sqlite3/base.py:

Start line: 1, End line: 78

```python
"""
SQLite backend for the sqlite3 module in the standard library.
"""
import datetime
import decimal
import functools
import hashlib
import math
import operator
import random
import re
import statistics
import warnings
from itertools import chain
from sqlite3 import dbapi2 as Database

import pytz

from django.core.exceptions import ImproperlyConfigured
from django.db import IntegrityError
from django.db.backends import utils as backend_utils
from django.db.backends.base.base import BaseDatabaseWrapper
from django.utils import timezone
from django.utils.asyncio import async_unsafe
from django.utils.dateparse import parse_datetime, parse_time
from django.utils.duration import duration_microseconds
from django.utils.regex_helper import _lazy_re_compile
from django.utils.version import PY38

from .client import DatabaseClient
from .creation import DatabaseCreation
from .features import DatabaseFeatures
from .introspection import DatabaseIntrospection
from .operations import DatabaseOperations
from .schema import DatabaseSchemaEditor


def decoder(conv_func):
    """
    Convert bytestrings from Python's sqlite3 interface to a regular string.
    """
    return lambda s: conv_func(s.decode())


def none_guard(func):
    """
    Decorator that returns None if any of the arguments to the decorated
    function are None. Many SQL functions return NULL if any of their arguments
    are NULL. This decorator simplifies the implementation of this for the
    custom functions registered below.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return None if None in args else func(*args, **kwargs)
    return wrapper


def list_aggregate(function):
    """
    Return an aggregate class that accumulates values in a list and applies
    the provided function to the data.
    """
    return type('ListAggregate', (list,), {'finalize': function, 'step': list.append})


def check_sqlite_version():
    if Database.sqlite_version_info < (3, 8, 3):
        raise ImproperlyConfigured('SQLite 3.8.3 or later is required (found %s).' % Database.sqlite_version)


check_sqlite_version()

Database.register_converter("bool", b'1'.__eq__)
Database.register_converter("time", decoder(parse_time))
Database.register_converter("datetime", decoder(parse_datetime))
Database.register_converter("timestamp", decoder(parse_datetime))

Database.register_adapter(decimal.Decimal, str)
```
### 74 - django/db/backends/sqlite3/base.py:

Start line: 156, End line: 173

```python
class DatabaseWrapper(BaseDatabaseWrapper):
    pattern_esc = r"REPLACE(REPLACE(REPLACE({}, '\', '\\'), '%%', '\%%'), '_', '\_')"
    pattern_ops = {
        'contains': r"LIKE '%%' || {} || '%%' ESCAPE '\'",
        'icontains': r"LIKE '%%' || UPPER({}) || '%%' ESCAPE '\'",
        'startswith': r"LIKE {} || '%%' ESCAPE '\'",
        'istartswith': r"LIKE UPPER({}) || '%%' ESCAPE '\'",
        'endswith': r"LIKE '%%' || {} ESCAPE '\'",
        'iendswith': r"LIKE '%%' || UPPER({}) ESCAPE '\'",
    }

    Database = Database
    SchemaEditorClass = DatabaseSchemaEditor
    # Classes instantiated in __init__().
    client_class = DatabaseClient
    creation_class = DatabaseCreation
    features_class = DatabaseFeatures
    introspection_class = DatabaseIntrospection
    ops_class = DatabaseOperations
```
### 97 - django/db/backends/sqlite3/base.py:

Start line: 588, End line: 610

```python
@none_guard
def _sqlite_timestamp_diff(lhs, rhs):
    left = backend_utils.typecast_timestamp(lhs)
    right = backend_utils.typecast_timestamp(rhs)
    return duration_microseconds(left - right)


@none_guard
def _sqlite_regexp(re_pattern, re_string):
    return bool(re.search(re_pattern, str(re_string)))


@none_guard
def _sqlite_lpad(text, length, fill_text):
    if len(text) >= length:
        return text[:length]
    return (fill_text * length)[:length - len(text)] + text


@none_guard
def _sqlite_rpad(text, length, fill_text):
    return (text + fill_text * length)[:length]
```
