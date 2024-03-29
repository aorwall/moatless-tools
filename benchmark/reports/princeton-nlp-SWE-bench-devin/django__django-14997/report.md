# django__django-14997

| **django/django** | `0d4e575c96d408e0efb4dfd0cbfc864219776950` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | - |
| **Avg pos** | - |
| **Min pos** | - |
| **Max pos** | - |
| **Top file pos** | 34 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/db/backends/ddl_references.py b/django/db/backends/ddl_references.py
--- a/django/db/backends/ddl_references.py
+++ b/django/db/backends/ddl_references.py
@@ -212,11 +212,7 @@ def __init__(self, table, expressions, compiler, quote_value):
     def rename_table_references(self, old_table, new_table):
         if self.table != old_table:
             return
-        expressions = deepcopy(self.expressions)
-        self.columns = []
-        for col in self.compiler.query._gen_cols([expressions]):
-            col.alias = new_table
-        self.expressions = expressions
+        self.expressions = self.expressions.relabeled_clone({old_table: new_table})
         super().rename_table_references(old_table, new_table)
 
     def rename_column_references(self, table, old_column, new_column):

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/db/backends/ddl_references.py | 215 | 219 | - | 34 | -


## Problem Statement

```
Remaking table with unique constraint crashes on SQLite.
Description
	
In Django 4.0a1, this model:
class Tag(models.Model):
	name = models.SlugField(help_text="The tag key.")
	value = models.CharField(max_length=150, help_text="The tag value.")
	class Meta:
		ordering = ["name", "value"]
		constraints = [
			models.UniqueConstraint(
				"name",
				"value",
				name="unique_name_value",
			)
		]
	def __str__(self):
		return f"{self.name}={self.value}"
with these migrations, using sqlite:
class Migration(migrations.Migration):
	initial = True
	dependencies = [
	]
	operations = [
		migrations.CreateModel(
			name='Tag',
			fields=[
				('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
				('name', models.SlugField(help_text='The tag key.')),
				('value', models.CharField(help_text='The tag value.', max_length=200)),
			],
			options={
				'ordering': ['name', 'value'],
			},
		),
		migrations.AddConstraint(
			model_name='tag',
			constraint=models.UniqueConstraint(django.db.models.expressions.F('name'), django.db.models.expressions.F('value'), name='unique_name_value'),
		),
	]
class Migration(migrations.Migration):
	dependencies = [
		('myapp', '0001_initial'),
	]
	operations = [
		migrations.AlterField(
			model_name='tag',
			name='value',
			field=models.CharField(help_text='The tag value.', max_length=150),
		),
	]
raises this error:
manage.py migrate
Operations to perform:
 Apply all migrations: admin, auth, contenttypes, myapp, sessions
Running migrations:
 Applying myapp.0002_alter_tag_value...python-BaseException
Traceback (most recent call last):
 File "D:\Projects\Development\sqliteerror\.venv\lib\site-packages\django\db\backends\utils.py", line 84, in _execute
	return self.cursor.execute(sql, params)
 File "D:\Projects\Development\sqliteerror\.venv\lib\site-packages\django\db\backends\sqlite3\base.py", line 416, in execute
	return Database.Cursor.execute(self, query, params)
sqlite3.OperationalError: the "." operator prohibited in index expressions
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
 File "D:\Projects\Development\sqliteerror\.venv\lib\site-packages\django\core\management\base.py", line 373, in run_from_argv
	self.execute(*args, **cmd_options)
 File "D:\Projects\Development\sqliteerror\.venv\lib\site-packages\django\core\management\base.py", line 417, in execute
	output = self.handle(*args, **options)
 File "D:\Projects\Development\sqliteerror\.venv\lib\site-packages\django\core\management\base.py", line 90, in wrapped
	res = handle_func(*args, **kwargs)
 File "D:\Projects\Development\sqliteerror\.venv\lib\site-packages\django\core\management\commands\migrate.py", line 253, in handle
	post_migrate_state = executor.migrate(
 File "D:\Projects\Development\sqliteerror\.venv\lib\site-packages\django\db\migrations\executor.py", line 126, in migrate
	state = self._migrate_all_forwards(state, plan, full_plan, fake=fake, fake_initial=fake_initial)
 File "D:\Projects\Development\sqliteerror\.venv\lib\site-packages\django\db\migrations\executor.py", line 156, in _migrate_all_forwards
	state = self.apply_migration(state, migration, fake=fake, fake_initial=fake_initial)
 File "D:\Projects\Development\sqliteerror\.venv\lib\site-packages\django\db\migrations\executor.py", line 236, in apply_migration
	state = migration.apply(state, schema_editor)
 File "D:\Projects\Development\sqliteerror\.venv\lib\site-packages\django\db\migrations\migration.py", line 125, in apply
	operation.database_forwards(self.app_label, schema_editor, old_state, project_state)
 File "D:\Projects\Development\sqliteerror\.venv\lib\site-packages\django\db\migrations\operations\fields.py", line 225, in database_forwards
	schema_editor.alter_field(from_model, from_field, to_field)
 File "D:\Projects\Development\sqliteerror\.venv\lib\site-packages\django\db\backends\sqlite3\schema.py", line 140, in alter_field
	super().alter_field(model, old_field, new_field, strict=strict)
 File "D:\Projects\Development\sqliteerror\.venv\lib\site-packages\django\db\backends\base\schema.py", line 618, in alter_field
	self._alter_field(model, old_field, new_field, old_type, new_type,
 File "D:\Projects\Development\sqliteerror\.venv\lib\site-packages\django\db\backends\sqlite3\schema.py", line 362, in _alter_field
	self._remake_table(model, alter_field=(old_field, new_field))
 File "D:\Projects\Development\sqliteerror\.venv\lib\site-packages\django\db\backends\sqlite3\schema.py", line 303, in _remake_table
	self.execute(sql)
 File "D:\Projects\Development\sqliteerror\.venv\lib\site-packages\django\db\backends\base\schema.py", line 151, in execute
	cursor.execute(sql, params)
 File "D:\Projects\Development\sqliteerror\.venv\lib\site-packages\django\db\backends\utils.py", line 98, in execute
	return super().execute(sql, params)
 File "D:\Projects\Development\sqliteerror\.venv\lib\site-packages\django\db\backends\utils.py", line 66, in execute
	return self._execute_with_wrappers(sql, params, many=False, executor=self._execute)
 File "D:\Projects\Development\sqliteerror\.venv\lib\site-packages\django\db\backends\utils.py", line 75, in _execute_with_wrappers
	return executor(sql, params, many, context)
 File "D:\Projects\Development\sqliteerror\.venv\lib\site-packages\django\db\backends\utils.py", line 84, in _execute
	return self.cursor.execute(sql, params)
 File "D:\Projects\Development\sqliteerror\.venv\lib\site-packages\django\db\utils.py", line 90, in __exit__
	raise dj_exc_value.with_traceback(traceback) from exc_value
 File "D:\Projects\Development\sqliteerror\.venv\lib\site-packages\django\db\backends\utils.py", line 84, in _execute
	return self.cursor.execute(sql, params)
 File "D:\Projects\Development\sqliteerror\.venv\lib\site-packages\django\db\backends\sqlite3\base.py", line 416, in execute
	return Database.Cursor.execute(self, query, params)
django.db.utils.OperationalError: the "." operator prohibited in index expressions

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 django/db/models/constraints.py | 198 | 216| 234 | 234 | 2061 | 
| 2 | 2 django/db/backends/sqlite3/schema.py | 86 | 99| 181 | 415 | 6264 | 
| 3 | 2 django/db/backends/sqlite3/schema.py | 101 | 140| 505 | 920 | 6264 | 
| 4 | 2 django/db/models/constraints.py | 187 | 196| 130 | 1050 | 6264 | 
| 5 | 2 django/db/backends/sqlite3/schema.py | 1 | 37| 317 | 1367 | 6264 | 
| 6 | 3 django/db/migrations/operations/__init__.py | 1 | 18| 195 | 1562 | 6459 | 
| 7 | 3 django/db/backends/sqlite3/schema.py | 225 | 307| 731 | 2293 | 6459 | 
| 8 | 4 django/db/backends/base/schema.py | 1246 | 1287| 357 | 2650 | 19341 | 
| 9 | 4 django/db/backends/sqlite3/schema.py | 421 | 445| 162 | 2812 | 19341 | 
| 10 | 4 django/db/backends/base/schema.py | 1213 | 1244| 233 | 3045 | 19341 | 
| 11 | 5 django/db/models/base.py | 1178 | 1206| 213 | 3258 | 36671 | 
| 12 | 5 django/db/backends/base/schema.py | 1289 | 1308| 163 | 3421 | 36671 | 
| 13 | 6 django/db/backends/sqlite3/operations.py | 180 | 205| 190 | 3611 | 39943 | 
| 14 | 6 django/db/backends/base/schema.py | 1151 | 1170| 175 | 3786 | 39943 | 
| 15 | 7 django/db/migrations/operations/models.py | 815 | 846| 273 | 4059 | 46393 | 
| 16 | 7 django/db/models/base.py | 1087 | 1130| 404 | 4463 | 46393 | 
| 17 | 7 django/db/models/constraints.py | 93 | 185| 685 | 5148 | 46393 | 
| 18 | 7 django/db/models/constraints.py | 218 | 228| 163 | 5311 | 46393 | 
| 19 | 7 django/db/backends/sqlite3/schema.py | 142 | 223| 820 | 6131 | 46393 | 
| 20 | 7 django/db/backends/base/schema.py | 452 | 466| 174 | 6305 | 46393 | 
| 21 | 7 django/db/migrations/operations/models.py | 849 | 885| 331 | 6636 | 46393 | 
| 22 | 8 django/db/migrations/autodetector.py | 1153 | 1174| 231 | 6867 | 58078 | 
| 23 | 8 django/db/backends/base/schema.py | 51 | 119| 785 | 7652 | 58078 | 
| 24 | 8 django/db/models/base.py | 1161 | 1176| 138 | 7790 | 58078 | 
| 25 | 9 django/db/migrations/state.py | 259 | 309| 468 | 8258 | 65941 | 
| 26 | 9 django/db/backends/base/schema.py | 1189 | 1211| 199 | 8457 | 65941 | 
| 27 | 9 django/db/migrations/operations/models.py | 319 | 368| 493 | 8950 | 65941 | 
| 28 | 10 django/db/backends/oracle/schema.py | 1 | 44| 454 | 9404 | 68084 | 
| 29 | 10 django/db/backends/base/schema.py | 1364 | 1396| 293 | 9697 | 68084 | 
| 30 | 10 django/db/models/constraints.py | 230 | 256| 205 | 9902 | 68084 | 
| 31 | 10 django/db/backends/base/schema.py | 415 | 429| 183 | 10085 | 68084 | 
| 32 | 11 django/db/backends/postgresql/schema.py | 184 | 210| 351 | 10436 | 70252 | 
| 33 | 11 django/db/models/base.py | 1966 | 2124| 1178 | 11614 | 70252 | 
| 34 | 11 django/db/backends/postgresql/schema.py | 227 | 239| 152 | 11766 | 70252 | 
| 35 | 12 django/db/models/sql/query.py | 1699 | 1741| 436 | 12202 | 92588 | 
| 36 | 12 django/db/models/base.py | 1546 | 1578| 231 | 12433 | 92588 | 
| 37 | 12 django/db/backends/sqlite3/operations.py | 330 | 387| 545 | 12978 | 92588 | 
| 38 | 12 django/db/models/base.py | 1522 | 1544| 171 | 13149 | 92588 | 
| 39 | 12 django/db/backends/base/schema.py | 348 | 363| 154 | 13303 | 92588 | 
| 40 | 12 django/db/models/base.py | 1269 | 1300| 267 | 13570 | 92588 | 
| 41 | 13 django/db/backends/mysql/schema.py | 1 | 39| 428 | 13998 | 94162 | 
| 42 | 13 django/db/migrations/autodetector.py | 1043 | 1059| 188 | 14186 | 94162 | 
| 43 | 14 django/db/backends/sqlite3/introspection.py | 331 | 359| 278 | 14464 | 98251 | 
| 44 | 15 django/core/management/commands/sqlmigrate.py | 31 | 69| 379 | 14843 | 98884 | 
| 45 | 15 django/db/migrations/operations/models.py | 370 | 390| 213 | 15056 | 98884 | 
| 46 | 15 django/db/backends/postgresql/schema.py | 1 | 67| 626 | 15682 | 98884 | 
| 47 | 15 django/db/models/base.py | 1 | 50| 328 | 16010 | 98884 | 
| 48 | 15 django/db/backends/sqlite3/schema.py | 39 | 65| 243 | 16253 | 98884 | 
| 49 | 16 django/db/backends/sqlite3/features.py | 1 | 127| 1163 | 17416 | 100047 | 
| 50 | 17 django/contrib/gis/db/backends/spatialite/schema.py | 128 | 169| 376 | 17792 | 101399 | 
| 51 | 17 django/db/backends/oracle/schema.py | 62 | 82| 249 | 18041 | 101399 | 
| 52 | 18 django/db/models/indexes.py | 90 | 116| 251 | 18292 | 103722 | 
| 53 | 18 django/db/backends/sqlite3/operations.py | 246 | 275| 198 | 18490 | 103722 | 
| 54 | 19 django/db/backends/sqlite3/base.py | 83 | 157| 775 | 19265 | 109792 | 
| 55 | 19 django/db/backends/base/schema.py | 684 | 754| 799 | 20064 | 109792 | 
| 56 | 19 django/db/backends/sqlite3/schema.py | 350 | 384| 422 | 20486 | 109792 | 
| 57 | 20 django/db/models/deletion.py | 1 | 75| 561 | 21047 | 113622 | 
| 58 | 20 django/db/backends/sqlite3/operations.py | 1 | 41| 305 | 21352 | 113622 | 
| 59 | 20 django/db/migrations/operations/models.py | 580 | 596| 215 | 21567 | 113622 | 
| 60 | 20 django/db/backends/base/schema.py | 755 | 833| 826 | 22393 | 113622 | 
| 61 | 20 django/db/backends/base/schema.py | 1172 | 1187| 170 | 22563 | 113622 | 
| 62 | 21 django/db/models/fields/related.py | 1266 | 1383| 963 | 23526 | 127618 | 
| 63 | 21 django/db/migrations/operations/models.py | 598 | 615| 163 | 23689 | 127618 | 
| 64 | 21 django/db/backends/postgresql/schema.py | 101 | 182| 647 | 24336 | 127618 | 
| 65 | 21 django/db/models/base.py | 915 | 947| 385 | 24721 | 127618 | 
| 66 | 21 django/db/migrations/autodetector.py | 1003 | 1019| 188 | 24909 | 127618 | 
| 67 | 22 django/contrib/postgres/operations.py | 296 | 330| 267 | 25176 | 130004 | 
| 68 | 22 django/db/migrations/operations/models.py | 41 | 104| 513 | 25689 | 130004 | 
| 69 | 23 django/db/models/__init__.py | 1 | 53| 619 | 26308 | 130623 | 
| 70 | 23 django/db/backends/base/schema.py | 1310 | 1332| 173 | 26481 | 130623 | 
| 71 | 23 django/db/backends/sqlite3/operations.py | 207 | 226| 209 | 26690 | 130623 | 
| 72 | 24 django/db/backends/postgresql/operations.py | 160 | 187| 311 | 27001 | 133184 | 
| 73 | 24 django/db/migrations/state.py | 170 | 205| 407 | 27408 | 133184 | 
| 74 | 24 django/db/backends/base/schema.py | 989 | 1016| 327 | 27735 | 133184 | 
| 75 | 24 django/db/backends/sqlite3/base.py | 312 | 401| 850 | 28585 | 133184 | 
| 76 | 25 django/db/models/fields/__init__.py | 2077 | 2107| 252 | 28837 | 151352 | 
| 77 | 26 django/db/models/sql/compiler.py | 1 | 20| 171 | 29008 | 166144 | 
| 78 | 27 django/db/backends/mysql/operations.py | 222 | 279| 437 | 29445 | 169871 | 
| 79 | 27 django/db/migrations/state.py | 872 | 885| 138 | 29583 | 169871 | 
| 80 | 27 django/db/backends/base/schema.py | 834 | 874| 506 | 30089 | 169871 | 
| 81 | 27 django/db/backends/sqlite3/schema.py | 309 | 330| 218 | 30307 | 169871 | 
| 82 | 27 django/db/backends/sqlite3/introspection.py | 225 | 239| 146 | 30453 | 169871 | 
| 83 | 27 django/db/models/fields/related.py | 531 | 596| 492 | 30945 | 169871 | 
| 84 | 28 django/db/backends/oracle/creation.py | 130 | 165| 399 | 31344 | 173764 | 
| 85 | 28 django/db/models/base.py | 1497 | 1520| 176 | 31520 | 173764 | 
| 86 | 28 django/db/backends/postgresql/schema.py | 212 | 225| 182 | 31702 | 173764 | 
| 87 | 28 django/db/models/base.py | 1607 | 1632| 183 | 31885 | 173764 | 
| 88 | 29 django/db/backends/postgresql/creation.py | 1 | 37| 261 | 32146 | 174433 | 
| 89 | 29 django/db/backends/mysql/operations.py | 1 | 35| 282 | 32428 | 174433 | 
| 90 | 29 django/db/backends/base/schema.py | 1 | 29| 209 | 32637 | 174433 | 
| 91 | 29 django/db/backends/sqlite3/operations.py | 228 | 244| 142 | 32779 | 174433 | 
| 92 | 29 django/db/models/base.py | 1875 | 1948| 572 | 33351 | 174433 | 
| 93 | 29 django/db/migrations/operations/models.py | 416 | 466| 410 | 33761 | 174433 | 
| 94 | 29 django/db/backends/sqlite3/schema.py | 386 | 419| 358 | 34119 | 174433 | 
| 95 | 29 django/db/migrations/operations/models.py | 124 | 247| 853 | 34972 | 174433 | 
| 96 | 29 django/db/backends/mysql/schema.py | 108 | 122| 143 | 35115 | 174433 | 
| 97 | 29 django/db/migrations/state.py | 397 | 413| 199 | 35314 | 174433 | 
| 98 | 30 django/db/models/options.py | 1 | 35| 300 | 35614 | 181800 | 
| 99 | 31 django/db/backends/sqlite3/creation.py | 23 | 49| 239 | 35853 | 182651 | 
| 100 | 32 django/db/migrations/questioner.py | 56 | 86| 255 | 36108 | 185175 | 
| 101 | 33 django/db/migrations/operations/fields.py | 254 | 320| 519 | 36627 | 187652 | 
| 102 | 33 django/db/backends/oracle/schema.py | 46 | 60| 133 | 36760 | 187652 | 
| 103 | 33 django/db/models/fields/related.py | 267 | 298| 284 | 37044 | 187652 | 
| 104 | 33 django/db/models/fields/related.py | 1459 | 1500| 418 | 37462 | 187652 | 
| 105 | **34 django/db/backends/ddl_references.py** | 111 | 129| 163 | 37625 | 189274 | 
| 106 | 35 django/core/exceptions.py | 107 | 218| 752 | 38377 | 190463 | 
| 107 | 35 django/db/models/sql/query.py | 1470 | 1555| 801 | 39178 | 190463 | 
| 108 | 36 django/db/migrations/exceptions.py | 1 | 55| 249 | 39427 | 190713 | 
| 109 | 36 django/db/migrations/operations/fields.py | 217 | 227| 146 | 39573 | 190713 | 
| 110 | 37 django/contrib/postgres/constraints.py | 92 | 105| 179 | 39752 | 192235 | 
| 111 | 37 django/db/migrations/state.py | 232 | 257| 240 | 39992 | 192235 | 
| 112 | 38 django/db/backends/oracle/operations.py | 481 | 500| 240 | 40232 | 198228 | 
| 113 | 38 django/db/models/fields/related.py | 889 | 915| 240 | 40472 | 198228 | 
| 114 | 38 django/db/backends/oracle/operations.py | 373 | 410| 369 | 40841 | 198228 | 
| 115 | 38 django/db/backends/sqlite3/creation.py | 51 | 82| 317 | 41158 | 198228 | 
| 116 | 38 django/db/migrations/operations/models.py | 531 | 550| 148 | 41306 | 198228 | 
| 117 | 39 django/contrib/sites/migrations/0002_alter_domain_unique.py | 1 | 21| 0 | 41306 | 198325 | 
| 118 | 39 django/db/migrations/state.py | 133 | 168| 391 | 41697 | 198325 | 
| 119 | 39 django/db/models/base.py | 949 | 966| 181 | 41878 | 198325 | 
| 120 | 39 django/db/models/base.py | 999 | 1027| 230 | 42108 | 198325 | 
| 121 | 39 django/db/migrations/operations/models.py | 1 | 38| 235 | 42343 | 198325 | 
| 122 | 39 django/db/migrations/operations/fields.py | 229 | 251| 188 | 42531 | 198325 | 
| 123 | 39 django/db/models/sql/compiler.py | 1080 | 1120| 337 | 42868 | 198325 | 


### Hint

```
Thanks for the report. Regression in 3aa545281e0c0f9fac93753e3769df9e0334dbaa.
Thanks for the report! Looks like we don't check if an alias is set on the Col before we update it to new_table in Expressions.rename_table_references when running _remake_table.
```

## Patch

```diff
diff --git a/django/db/backends/ddl_references.py b/django/db/backends/ddl_references.py
--- a/django/db/backends/ddl_references.py
+++ b/django/db/backends/ddl_references.py
@@ -212,11 +212,7 @@ def __init__(self, table, expressions, compiler, quote_value):
     def rename_table_references(self, old_table, new_table):
         if self.table != old_table:
             return
-        expressions = deepcopy(self.expressions)
-        self.columns = []
-        for col in self.compiler.query._gen_cols([expressions]):
-            col.alias = new_table
-        self.expressions = expressions
+        self.expressions = self.expressions.relabeled_clone({old_table: new_table})
         super().rename_table_references(old_table, new_table)
 
     def rename_column_references(self, table, old_column, new_column):

```

## Test Patch

```diff
diff --git a/tests/backends/test_ddl_references.py b/tests/backends/test_ddl_references.py
--- a/tests/backends/test_ddl_references.py
+++ b/tests/backends/test_ddl_references.py
@@ -5,6 +5,7 @@
 from django.db.models import ExpressionList, F
 from django.db.models.functions import Upper
 from django.db.models.indexes import IndexExpression
+from django.db.models.sql import Query
 from django.test import SimpleTestCase, TransactionTestCase
 
 from .models import Person
@@ -229,6 +230,27 @@ def test_rename_table_references(self):
             str(self.expressions),
         )
 
+    def test_rename_table_references_without_alias(self):
+        compiler = Query(Person, alias_cols=False).get_compiler(connection=connection)
+        table = Person._meta.db_table
+        expressions = Expressions(
+            table=table,
+            expressions=ExpressionList(
+                IndexExpression(Upper('last_name')),
+                IndexExpression(F('first_name')),
+            ).resolve_expression(compiler.query),
+            compiler=compiler,
+            quote_value=self.editor.quote_value,
+        )
+        expressions.rename_table_references(table, 'other')
+        self.assertIs(expressions.references_table(table), False)
+        self.assertIs(expressions.references_table('other'), True)
+        expected_str = '(UPPER(%s)), %s' % (
+            self.editor.quote_name('last_name'),
+            self.editor.quote_name('first_name'),
+        )
+        self.assertEqual(str(expressions), expected_str)
+
     def test_rename_column_references(self):
         table = Person._meta.db_table
         self.expressions.rename_column_references(table, 'first_name', 'other')
diff --git a/tests/migrations/test_operations.py b/tests/migrations/test_operations.py
--- a/tests/migrations/test_operations.py
+++ b/tests/migrations/test_operations.py
@@ -2106,6 +2106,25 @@ def test_remove_func_index(self):
         self.assertEqual(definition[1], [])
         self.assertEqual(definition[2], {'model_name': 'Pony', 'name': index_name})
 
+    @skipUnlessDBFeature('supports_expression_indexes')
+    def test_alter_field_with_func_index(self):
+        app_label = 'test_alfuncin'
+        index_name = f'{app_label}_pony_idx'
+        table_name = f'{app_label}_pony'
+        project_state = self.set_up_test_model(
+            app_label,
+            indexes=[models.Index(Abs('pink'), name=index_name)],
+        )
+        operation = migrations.AlterField('Pony', 'pink', models.IntegerField(null=True))
+        new_state = project_state.clone()
+        operation.state_forwards(app_label, new_state)
+        with connection.schema_editor() as editor:
+            operation.database_forwards(app_label, editor, project_state, new_state)
+        self.assertIndexNameExists(table_name, index_name)
+        with connection.schema_editor() as editor:
+            operation.database_backwards(app_label, editor, new_state, project_state)
+        self.assertIndexNameExists(table_name, index_name)
+
     def test_alter_field_with_index(self):
         """
         Test AlterField operation with an index to ensure indexes created via
@@ -2664,6 +2683,26 @@ def test_remove_covering_unique_constraint(self):
             'name': 'covering_pink_constraint_rm',
         })
 
+    def test_alter_field_with_func_unique_constraint(self):
+        app_label = 'test_alfuncuc'
+        constraint_name = f'{app_label}_pony_uq'
+        table_name = f'{app_label}_pony'
+        project_state = self.set_up_test_model(
+            app_label,
+            constraints=[models.UniqueConstraint('pink', 'weight', name=constraint_name)]
+        )
+        operation = migrations.AlterField('Pony', 'pink', models.IntegerField(null=True))
+        new_state = project_state.clone()
+        operation.state_forwards(app_label, new_state)
+        with connection.schema_editor() as editor:
+            operation.database_forwards(app_label, editor, project_state, new_state)
+        if connection.features.supports_expression_indexes:
+            self.assertIndexNameExists(table_name, constraint_name)
+        with connection.schema_editor() as editor:
+            operation.database_backwards(app_label, editor, new_state, project_state)
+        if connection.features.supports_expression_indexes:
+            self.assertIndexNameExists(table_name, constraint_name)
+
     def test_add_func_unique_constraint(self):
         app_label = 'test_adfuncuc'
         constraint_name = f'{app_label}_pony_abs_uq'

```


## Code snippets

### 1 - django/db/models/constraints.py:

Start line: 198, End line: 216

```python
class UniqueConstraint(BaseConstraint):

    def create_sql(self, model, schema_editor):
        fields = [model._meta.get_field(field_name) for field_name in self.fields]
        include = [model._meta.get_field(field_name).column for field_name in self.include]
        condition = self._get_condition_sql(model, schema_editor)
        expressions = self._get_index_expressions(model, schema_editor)
        return schema_editor._create_unique_sql(
            model, fields, self.name, condition=condition,
            deferrable=self.deferrable, include=include,
            opclasses=self.opclasses, expressions=expressions,
        )

    def remove_sql(self, model, schema_editor):
        condition = self._get_condition_sql(model, schema_editor)
        include = [model._meta.get_field(field_name).column for field_name in self.include]
        expressions = self._get_index_expressions(model, schema_editor)
        return schema_editor._delete_unique_sql(
            model, self.name, condition=condition, deferrable=self.deferrable,
            include=include, opclasses=self.opclasses, expressions=expressions,
        )
```
### 2 - django/db/backends/sqlite3/schema.py:

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
### 3 - django/db/backends/sqlite3/schema.py:

Start line: 101, End line: 140

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
### 4 - django/db/models/constraints.py:

Start line: 187, End line: 196

```python
class UniqueConstraint(BaseConstraint):

    def constraint_sql(self, model, schema_editor):
        fields = [model._meta.get_field(field_name) for field_name in self.fields]
        include = [model._meta.get_field(field_name).column for field_name in self.include]
        condition = self._get_condition_sql(model, schema_editor)
        expressions = self._get_index_expressions(model, schema_editor)
        return schema_editor._unique_sql(
            model, fields, self.name, condition=condition,
            deferrable=self.deferrable, include=include,
            opclasses=self.opclasses, expressions=expressions,
        )
```
### 5 - django/db/backends/sqlite3/schema.py:

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
### 6 - django/db/migrations/operations/__init__.py:

Start line: 1, End line: 18

```python
from .fields import AddField, AlterField, RemoveField, RenameField
from .models import (
    AddConstraint, AddIndex, AlterIndexTogether, AlterModelManagers,
    AlterModelOptions, AlterModelTable, AlterOrderWithRespectTo,
    AlterUniqueTogether, CreateModel, DeleteModel, RemoveConstraint,
    RemoveIndex, RenameModel,
)
from .special import RunPython, RunSQL, SeparateDatabaseAndState

__all__ = [
    'CreateModel', 'DeleteModel', 'AlterModelTable', 'AlterUniqueTogether',
    'RenameModel', 'AlterIndexTogether', 'AlterModelOptions', 'AddIndex',
    'RemoveIndex', 'AddField', 'RemoveField', 'AlterField', 'RenameField',
    'AddConstraint', 'RemoveConstraint',
    'SeparateDatabaseAndState', 'RunSQL', 'RunPython',
    'AlterOrderWithRespectTo', 'AlterModelManagers',
]
```
### 7 - django/db/backends/sqlite3/schema.py:

Start line: 225, End line: 307

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
### 8 - django/db/backends/base/schema.py:

Start line: 1246, End line: 1287

```python
class BaseDatabaseSchemaEditor:

    def _create_unique_sql(
        self, model, fields, name=None, condition=None, deferrable=None,
        include=None, opclasses=None, expressions=None,
    ):
        if (
            (
                deferrable and
                not self.connection.features.supports_deferrable_unique_constraints
            ) or
            (condition and not self.connection.features.supports_partial_indexes) or
            (include and not self.connection.features.supports_covering_indexes) or
            (expressions and not self.connection.features.supports_expression_indexes)
        ):
            return None

        def create_unique_name(*args, **kwargs):
            return self.quote_name(self._create_index_name(*args, **kwargs))

        compiler = Query(model, alias_cols=False).get_compiler(connection=self.connection)
        table = model._meta.db_table
        columns = [field.column for field in fields]
        if name is None:
            name = IndexName(table, columns, '_uniq', create_unique_name)
        else:
            name = self.quote_name(name)
        if condition or include or opclasses or expressions:
            sql = self.sql_create_unique_index
        else:
            sql = self.sql_create_unique
        if columns:
            columns = self._index_columns(table, columns, col_suffixes=(), opclasses=opclasses)
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
### 9 - django/db/backends/sqlite3/schema.py:

Start line: 421, End line: 445

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
### 10 - django/db/backends/base/schema.py:

Start line: 1213, End line: 1244

```python
class BaseDatabaseSchemaEditor:

    def _unique_sql(
        self, model, fields, name, condition=None, deferrable=None,
        include=None, opclasses=None, expressions=None,
    ):
        if (
            deferrable and
            not self.connection.features.supports_deferrable_unique_constraints
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
            'columns': ', '.join([self.quote_name(field.column) for field in fields]),
            'deferrable': self._deferrable_constraint_sql(deferrable),
        }
        return self.sql_constraint % {
            'name': self.quote_name(name),
            'constraint': constraint,
        }
```
### 105 - django/db/backends/ddl_references.py:

Start line: 111, End line: 129

```python
class IndexColumns(Columns):
    def __init__(self, table, columns, quote_name, col_suffixes=(), opclasses=()):
        self.opclasses = opclasses
        super().__init__(table, columns, quote_name, col_suffixes)

    def __str__(self):
        def col_str(column, idx):
            # Index.__init__() guarantees that self.opclasses is the same
            # length as self.columns.
            col = '{} {}'.format(self.quote_name(column), self.opclasses[idx])
            try:
                suffix = self.col_suffixes[idx]
                if suffix:
                    col = '{} {}'.format(col, suffix)
            except IndexError:
                pass
            return col

        return ', '.join(col_str(column, idx) for idx, column in enumerate(self.columns))
```
