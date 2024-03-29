# django__django-14722

| **django/django** | `c1e4111c74ee9d9f48cbee5a5b7c40289203c93d` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | 16054 |
| **Avg pos** | 102.0 |
| **Min pos** | 51 |
| **Max pos** | 51 |
| **Top file pos** | 3 |
| **Missing snippets** | 3 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/db/migrations/autodetector.py b/django/db/migrations/autodetector.py
--- a/django/db/migrations/autodetector.py
+++ b/django/db/migrations/autodetector.py
@@ -178,8 +178,12 @@ def _detect_changes(self, convert_apps=None, graph=None):
         # Generate index removal operations before field is removed
         self.generate_removed_constraints()
         self.generate_removed_indexes()
-        # Generate field operations
+        # Generate field renaming operations.
         self.generate_renamed_fields()
+        # Generate removal of foo together.
+        self.generate_removed_altered_unique_together()
+        self.generate_removed_altered_index_together()
+        # Generate field operations.
         self.generate_removed_fields()
         self.generate_added_fields()
         self.generate_altered_fields()
@@ -1128,8 +1132,7 @@ def _get_dependencies_for_foreign_key(app_label, model_name, field, project_stat
             dependencies.append((through_app_label, through_object_name, None, True))
         return dependencies
 
-    def _generate_altered_foo_together(self, operation):
-        option_name = operation.option_name
+    def _get_altered_foo_together_operations(self, option_name):
         for app_label, model_name in sorted(self.kept_model_keys):
             old_model_name = self.renamed_models.get((app_label, model_name), model_name)
             old_model_state = self.from_state.models[app_label, old_model_name]
@@ -1157,13 +1160,49 @@ def _generate_altered_foo_together(self, operation):
                             dependencies.extend(self._get_dependencies_for_foreign_key(
                                 app_label, model_name, field, self.to_state,
                             ))
+                yield (
+                    old_value,
+                    new_value,
+                    app_label,
+                    model_name,
+                    dependencies,
+                )
 
+    def _generate_removed_altered_foo_together(self, operation):
+        for (
+            old_value,
+            new_value,
+            app_label,
+            model_name,
+            dependencies,
+        ) in self._get_altered_foo_together_operations(operation.option_name):
+            removal_value = new_value.intersection(old_value)
+            if removal_value or old_value:
                 self.add_operation(
                     app_label,
-                    operation(
-                        name=model_name,
-                        **{option_name: new_value}
-                    ),
+                    operation(name=model_name, **{operation.option_name: removal_value}),
+                    dependencies=dependencies,
+                )
+
+    def generate_removed_altered_unique_together(self):
+        self._generate_removed_altered_foo_together(operations.AlterUniqueTogether)
+
+    def generate_removed_altered_index_together(self):
+        self._generate_removed_altered_foo_together(operations.AlterIndexTogether)
+
+    def _generate_altered_foo_together(self, operation):
+        for (
+            old_value,
+            new_value,
+            app_label,
+            model_name,
+            dependencies,
+        ) in self._get_altered_foo_together_operations(operation.option_name):
+            removal_value = new_value.intersection(old_value)
+            if new_value != removal_value:
+                self.add_operation(
+                    app_label,
+                    operation(name=model_name, **{operation.option_name: new_value}),
                     dependencies=dependencies,
                 )
 

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/db/migrations/autodetector.py | 181 | 181 | - | 3 | -
| django/db/migrations/autodetector.py | 1131 | 1132 | 51 | 3 | 16054
| django/db/migrations/autodetector.py | 1160 | 1163 | 51 | 3 | 16054


## Problem Statement

```
Moving a unique constraint from unique_together to Field.unique generate an invalid migration.
Description
	
You can see a demo example to show the bug at [github](​https://github.com/ramwin/testunique/).
I met a problem when I convert a unique_together to the unique=True attribute. 
first commit, everything is ok
I create a model file.
// testapp/models.py
class MyModel(models.Model):
 name = models.CharField(max_length=32)
 class Meta:
	 unique_together = ("name", )
the migrations file looks like this.
// testapp/migrations/0001_initial.py
# Generated by Django 3.0.5 on 2020-04-22 12:47
from django.db import migrations, models
class Migration(migrations.Migration):
 dependencies = [
	 ('testapp', '0001_initial'),
 ]
 operations = [
	 migrations.AlterField(
		 model_name='mymodel',
		 name='name',
		 field=models.CharField(max_length=32, unique=True),
	 ),
	 migrations.AlterUniqueTogether(
		 name='mymodel',
		 unique_together=set(),
	 ),
 ]
second commit: then I remove the unique_together and add unique=True to the field MyModel.name
model file
class MyModel(models.Model):
 name = models.CharField(max_length=32, unique=True)
 class Meta:
	 pass
	 # unique_together = ("name", )
0002 migrations file
class Migration(migrations.Migration):
 dependencies = [
	 ('testapp', '0001_initial'),
 ]
 operations = [
	 migrations.AlterField(
		 model_name='mymodel',
		 name='name',
		 field=models.CharField(max_length=32, unique=True),
	 ),
	 migrations.AlterUniqueTogether(
		 name='mymodel',
		 unique_together=set(),
	 ),
 ]
However, when I apply the migrations, an error occurs;
wangx@aliyun:~/testunique$ python3 manage.py migrate
Operations to perform:
 Apply all migrations: admin, auth, contenttypes, sessions, testapp
Running migrations:
 Applying contenttypes.0001_initial... OK
 Applying auth.0001_initial... OK
 Applying admin.0001_initial... OK
 Applying admin.0002_logentry_remove_auto_add... OK
 Applying admin.0003_logentry_add_action_flag_choices... OK
 Applying contenttypes.0002_remove_content_type_name... OK
 Applying auth.0002_alter_permission_name_max_length... OK
 Applying auth.0003_alter_user_email_max_length... OK
 Applying auth.0004_alter_user_username_opts... OK
 Applying auth.0005_alter_user_last_login_null... OK
 Applying auth.0006_require_contenttypes_0002... OK
 Applying auth.0007_alter_validators_add_error_messages... OK
 Applying auth.0008_alter_user_username_max_length... OK
 Applying auth.0009_alter_user_last_name_max_length... OK
 Applying auth.0010_alter_group_name_max_length... OK
 Applying auth.0011_update_proxy_permissions... OK
 Applying sessions.0001_initial... OK
 Applying testapp.0001_initial... OK
 Applying testapp.0002_auto_20200422_1247...Traceback (most recent call last):
 File "/usr/local/lib/python3.6/dist-packages/django/db/backends/utils.py", line 86, in _execute
	return self.cursor.execute(sql, params)
 File "/usr/local/lib/python3.6/dist-packages/django/db/backends/mysql/base.py", line 74, in execute
	return self.cursor.execute(query, args)
 File "/usr/local/lib/python3.6/dist-packages/MySQLdb/cursors.py", line 209, in execute
	res = self._query(query)
 File "/usr/local/lib/python3.6/dist-packages/MySQLdb/cursors.py", line 315, in _query
	db.query(q)
 File "/usr/local/lib/python3.6/dist-packages/MySQLdb/connections.py", line 239, in query
	_mysql.connection.query(self, query)
MySQLdb._exceptions.OperationalError: (1061, "Duplicate key name 'testapp_mymodel_name_ba5e2bd2_uniq'")
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
 File "manage.py", line 21, in <module>
	main()
 File "manage.py", line 17, in main
	execute_from_command_line(sys.argv)
 File "/usr/local/lib/python3.6/dist-packages/django/core/management/__init__.py", line 401, in execute_from_command_line
	utility.execute()
 File "/usr/local/lib/python3.6/dist-packages/django/core/management/__init__.py", line 395, in execute
	self.fetch_command(subcommand).run_from_argv(self.argv)
 File "/usr/local/lib/python3.6/dist-packages/django/core/management/base.py", line 328, in run_from_argv
	self.execute(*args, **cmd_options)
 File "/usr/local/lib/python3.6/dist-packages/django/core/management/base.py", line 369, in execute
	output = self.handle(*args, **options)
 File "/usr/local/lib/python3.6/dist-packages/django/core/management/base.py", line 83, in wrapped
	res = handle_func(*args, **kwargs)
 File "/usr/local/lib/python3.6/dist-packages/django/core/management/commands/migrate.py", line 233, in handle
	fake_initial=fake_initial,
 File "/usr/local/lib/python3.6/dist-packages/django/db/migrations/executor.py", line 117, in migrate
	state = self._migrate_all_forwards(state, plan, full_plan, fake=fake, fake_initial=fake_initial)
 File "/usr/local/lib/python3.6/dist-packages/django/db/migrations/executor.py", line 147, in _migrate_all_forwards
	state = self.apply_migration(state, migration, fake=fake, fake_initial=fake_initial)
 File "/usr/local/lib/python3.6/dist-packages/django/db/migrations/executor.py", line 245, in apply_migration
	state = migration.apply(state, schema_editor)
 File "/usr/local/lib/python3.6/dist-packages/django/db/migrations/migration.py", line 124, in apply
	operation.database_forwards(self.app_label, schema_editor, old_state, project_state)
 File "/usr/local/lib/python3.6/dist-packages/django/db/migrations/operations/fields.py", line 249, in database_forwards
	schema_editor.alter_field(from_model, from_field, to_field)
 File "/usr/local/lib/python3.6/dist-packages/django/db/backends/base/schema.py", line 565, in alter_field
	old_db_params, new_db_params, strict)
 File "/usr/local/lib/python3.6/dist-packages/django/db/backends/base/schema.py", line 745, in _alter_field
	self.execute(self._create_unique_sql(model, [new_field.column]))
 File "/usr/local/lib/python3.6/dist-packages/django/db/backends/base/schema.py", line 142, in execute
	cursor.execute(sql, params)
 File "/usr/local/lib/python3.6/dist-packages/django/db/backends/utils.py", line 100, in execute
	return super().execute(sql, params)
 File "/usr/local/lib/python3.6/dist-packages/django/db/backends/utils.py", line 68, in execute
	return self._execute_with_wrappers(sql, params, many=False, executor=self._execute)
 File "/usr/local/lib/python3.6/dist-packages/django/db/backends/utils.py", line 77, in _execute_with_wrappers
	return executor(sql, params, many, context)
 File "/usr/local/lib/python3.6/dist-packages/django/db/backends/utils.py", line 86, in _execute
	return self.cursor.execute(sql, params)
 File "/usr/local/lib/python3.6/dist-packages/django/db/utils.py", line 90, in __exit__
	raise dj_exc_value.with_traceback(traceback) from exc_value
 File "/usr/local/lib/python3.6/dist-packages/django/db/backends/utils.py", line 86, in _execute
	return self.cursor.execute(sql, params)
 File "/usr/local/lib/python3.6/dist-packages/django/db/backends/mysql/base.py", line 74, in execute
	return self.cursor.execute(query, args)
 File "/usr/local/lib/python3.6/dist-packages/MySQLdb/cursors.py", line 209, in execute
	res = self._query(query)
 File "/usr/local/lib/python3.6/dist-packages/MySQLdb/cursors.py", line 315, in _query
	db.query(q)
 File "/usr/local/lib/python3.6/dist-packages/MySQLdb/connections.py", line 239, in query
	_mysql.connection.query(self, query)
django.db.utils.OperationalError: (1061, "Duplicate key name 'testapp_mymodel_name_ba5e2bd2_uniq'")
I check the sql for these migrations, it shows:
wangx@aliyun:~/testunique$ python3 manage.py sqlmigrate testapp 0001
--
-- Create model MyModel
--
CREATE TABLE `testapp_mymodel` (`id` integer AUTO_INCREMENT NOT NULL PRIMARY KEY, `name` varchar(32) NOT NULL);
ALTER TABLE `testapp_mymodel` ADD CONSTRAINT `testapp_mymodel_name_ba5e2bd2_uniq` UNIQUE (`name`);
wangx@aliyun:~/testunique$ python3 manage.py sqlmigrate testapp 0002
--
-- Alter field name on mymodel
--
ALTER TABLE `testapp_mymodel` ADD CONSTRAINT `testapp_mymodel_name_ba5e2bd2_uniq` UNIQUE (`name`);
--
-- Alter unique_together for mymodel (0 constraint(s))
--
ALTER TABLE `testapp_mymodel` DROP INDEX `testapp_mymodel_name_ba5e2bd2_uniq`;
it looks like django will
first create the index for unique=True
second drop the index for unique_together=('name', )
but the program for creating index name generates the same index name :testapp_mymodel_name_ba5e2bd2_uniq, so when django create the same index, Duplicate key name error occurs.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 django/db/models/constraints.py | 198 | 216| 234 | 234 | 2061 | 
| 2 | 2 django/db/backends/base/schema.py | 415 | 429| 183 | 417 | 14943 | 
| 3 | **3 django/db/migrations/autodetector.py** | 1170 | 1191| 231 | 648 | 26776 | 
| 4 | 4 django/db/models/base.py | 1087 | 1130| 404 | 1052 | 44106 | 
| 5 | 4 django/db/models/base.py | 1178 | 1206| 213 | 1265 | 44106 | 
| 6 | 4 django/db/models/constraints.py | 187 | 196| 130 | 1395 | 44106 | 
| 7 | 4 django/db/models/base.py | 1607 | 1632| 183 | 1578 | 44106 | 
| 8 | 4 django/db/backends/base/schema.py | 1246 | 1287| 357 | 1935 | 44106 | 
| 9 | 5 django/db/migrations/operations/models.py | 531 | 550| 148 | 2083 | 50556 | 
| 10 | 5 django/db/models/constraints.py | 93 | 185| 685 | 2768 | 50556 | 
| 11 | 6 django/db/backends/mysql/schema.py | 124 | 140| 205 | 2973 | 52130 | 
| 12 | 6 django/db/backends/base/schema.py | 1289 | 1308| 163 | 3136 | 52130 | 
| 13 | 6 django/db/models/constraints.py | 230 | 256| 205 | 3341 | 52130 | 
| 14 | 6 django/db/backends/base/schema.py | 1213 | 1244| 233 | 3574 | 52130 | 
| 15 | 6 django/db/models/constraints.py | 218 | 228| 163 | 3737 | 52130 | 
| 16 | 6 django/db/backends/base/schema.py | 1151 | 1170| 175 | 3912 | 52130 | 
| 17 | 6 django/db/models/base.py | 1546 | 1578| 231 | 4143 | 52130 | 
| 18 | 7 django/db/migrations/exceptions.py | 1 | 55| 249 | 4392 | 52380 | 
| 19 | 8 django/db/models/indexes.py | 142 | 170| 326 | 4718 | 54703 | 
| 20 | 8 django/db/models/base.py | 1875 | 1948| 572 | 5290 | 54703 | 
| 21 | 8 django/db/migrations/operations/models.py | 1 | 38| 235 | 5525 | 54703 | 
| 22 | 9 django/forms/models.py | 765 | 786| 194 | 5719 | 66621 | 
| 23 | 10 django/db/models/fields/related.py | 1459 | 1500| 418 | 6137 | 80617 | 
| 24 | 10 django/db/models/fields/related.py | 531 | 596| 492 | 6629 | 80617 | 
| 25 | 10 django/db/backends/base/schema.py | 431 | 450| 199 | 6828 | 80617 | 
| 26 | 10 django/db/models/base.py | 1522 | 1544| 171 | 6999 | 80617 | 
| 27 | 11 django/db/migrations/operations/__init__.py | 1 | 18| 195 | 7194 | 80812 | 
| 28 | 11 django/db/models/base.py | 1497 | 1520| 176 | 7370 | 80812 | 
| 29 | 11 django/db/backends/base/schema.py | 989 | 1016| 327 | 7697 | 80812 | 
| 30 | 11 django/db/models/base.py | 1580 | 1605| 183 | 7880 | 80812 | 
| 31 | 11 django/db/models/fields/related.py | 889 | 915| 240 | 8120 | 80812 | 
| 32 | 11 django/db/models/base.py | 1269 | 1300| 267 | 8387 | 80812 | 
| 33 | 12 django/db/migrations/questioner.py | 56 | 86| 255 | 8642 | 83336 | 
| 34 | 12 django/db/migrations/operations/models.py | 849 | 885| 331 | 8973 | 83336 | 
| 35 | 12 django/forms/models.py | 686 | 763| 750 | 9723 | 83336 | 
| 36 | 12 django/forms/models.py | 419 | 449| 243 | 9966 | 83336 | 
| 37 | 12 django/db/models/fields/related.py | 1266 | 1383| 963 | 10929 | 83336 | 
| 38 | 12 django/db/migrations/operations/models.py | 319 | 368| 493 | 11422 | 83336 | 
| 39 | 13 django/db/migrations/state.py | 259 | 309| 468 | 11890 | 91199 | 
| 40 | 13 django/db/models/base.py | 1966 | 2124| 1178 | 13068 | 91199 | 
| 41 | 14 django/contrib/admin/checks.py | 1011 | 1026| 136 | 13204 | 100381 | 
| 42 | 14 django/db/models/base.py | 1161 | 1176| 138 | 13342 | 100381 | 
| 43 | 14 django/db/migrations/operations/models.py | 815 | 846| 273 | 13615 | 100381 | 
| 44 | 14 django/db/models/base.py | 1391 | 1421| 244 | 13859 | 100381 | 
| 45 | 14 django/db/backends/base/schema.py | 1189 | 1211| 199 | 14058 | 100381 | 
| 46 | 15 django/contrib/sites/migrations/0002_alter_domain_unique.py | 1 | 21| 0 | 14058 | 100478 | 
| 47 | 15 django/db/migrations/state.py | 872 | 885| 138 | 14196 | 100478 | 
| 48 | 16 django/contrib/contenttypes/migrations/0001_initial.py | 1 | 35| 207 | 14403 | 100685 | 
| 49 | 16 django/db/models/base.py | 1440 | 1495| 491 | 14894 | 100685 | 
| 50 | 17 django/contrib/auth/migrations/0001_initial.py | 1 | 104| 843 | 15737 | 101528 | 
| **-> 51 <-** | **17 django/db/migrations/autodetector.py** | 1131 | 1168| 317 | 16054 | 101528 | 
| 52 | 17 django/db/backends/base/schema.py | 452 | 466| 174 | 16228 | 101528 | 
| 53 | 17 django/db/migrations/operations/models.py | 124 | 247| 853 | 17081 | 101528 | 
| 54 | 17 django/db/models/base.py | 999 | 1027| 230 | 17311 | 101528 | 
| 55 | 17 django/db/migrations/operations/models.py | 106 | 122| 156 | 17467 | 101528 | 
| 56 | 18 django/core/management/commands/migrate.py | 71 | 160| 774 | 18241 | 104865 | 
| 57 | 19 django/db/backends/sqlite3/schema.py | 225 | 307| 731 | 18972 | 109068 | 
| 58 | 19 django/db/migrations/operations/models.py | 370 | 390| 213 | 19185 | 109068 | 
| 59 | 19 django/db/models/base.py | 1634 | 1721| 672 | 19857 | 109068 | 
| 60 | **19 django/db/migrations/autodetector.py** | 1020 | 1036| 188 | 20045 | 109068 | 
| 61 | 20 django/contrib/admin/migrations/0001_initial.py | 1 | 47| 314 | 20359 | 109382 | 
| 62 | 20 django/db/models/fields/related.py | 198 | 266| 687 | 21046 | 109382 | 
| 63 | **20 django/db/migrations/autodetector.py** | 462 | 514| 465 | 21511 | 109382 | 
| 64 | 20 django/db/backends/base/schema.py | 1364 | 1396| 293 | 21804 | 109382 | 
| 65 | 20 django/db/migrations/operations/models.py | 511 | 528| 168 | 21972 | 109382 | 
| 66 | 20 django/db/backends/mysql/schema.py | 1 | 39| 428 | 22400 | 109382 | 
| 67 | 20 django/db/migrations/operations/models.py | 500 | 509| 129 | 22529 | 109382 | 
| 68 | 20 django/db/backends/sqlite3/schema.py | 1 | 37| 317 | 22846 | 109382 | 
| 69 | 20 django/db/models/fields/related.py | 267 | 298| 284 | 23130 | 109382 | 
| 70 | 21 django/contrib/sites/migrations/0001_initial.py | 1 | 32| 191 | 23321 | 109573 | 
| 71 | 21 django/db/models/base.py | 1723 | 1771| 348 | 23669 | 109573 | 
| 72 | **21 django/db/migrations/autodetector.py** | 719 | 802| 680 | 24349 | 109573 | 
| 73 | 21 django/db/migrations/questioner.py | 238 | 257| 195 | 24544 | 109573 | 
| 74 | 22 django/db/backends/sqlite3/base.py | 312 | 401| 850 | 25394 | 115643 | 
| 75 | 22 django/db/backends/mysql/schema.py | 108 | 122| 143 | 25537 | 115643 | 
| 76 | 22 django/db/models/base.py | 915 | 947| 385 | 25922 | 115643 | 
| 77 | **22 django/db/migrations/autodetector.py** | 533 | 684| 1175 | 27097 | 115643 | 
| 78 | 22 django/db/migrations/state.py | 170 | 205| 407 | 27504 | 115643 | 
| 79 | 22 django/db/backends/sqlite3/schema.py | 142 | 223| 820 | 28324 | 115643 | 
| 80 | 23 django/db/models/fields/__init__.py | 308 | 336| 205 | 28529 | 133811 | 
| 81 | 23 django/db/models/base.py | 1 | 50| 328 | 28857 | 133811 | 
| 82 | 23 django/db/models/fields/related.py | 1385 | 1457| 616 | 29473 | 133811 | 
| 83 | 24 django/db/models/lookups.py | 655 | 688| 141 | 29614 | 139150 | 
| 84 | 24 django/db/migrations/state.py | 397 | 413| 199 | 29813 | 139150 | 
| 85 | 24 django/db/migrations/state.py | 887 | 903| 146 | 29959 | 139150 | 
| 86 | 24 django/db/backends/sqlite3/schema.py | 101 | 140| 505 | 30464 | 139150 | 
| 87 | 24 django/db/models/base.py | 1029 | 1085| 560 | 31024 | 139150 | 
| 88 | 24 django/core/management/commands/migrate.py | 228 | 279| 537 | 31561 | 139150 | 
| 89 | 24 django/db/backends/sqlite3/schema.py | 421 | 445| 162 | 31723 | 139150 | 
| 90 | **24 django/db/migrations/autodetector.py** | 1060 | 1076| 188 | 31911 | 139150 | 
| 91 | 25 django/contrib/auth/migrations/0007_alter_validators_add_error_messages.py | 1 | 25| 137 | 32048 | 139287 | 
| 92 | 25 django/db/models/base.py | 1208 | 1242| 230 | 32278 | 139287 | 
| 93 | **25 django/db/migrations/autodetector.py** | 516 | 532| 186 | 32464 | 139287 | 
| 94 | 26 django/core/management/commands/makemigrations.py | 64 | 155| 805 | 33269 | 142135 | 
| 95 | 26 django/db/backends/sqlite3/schema.py | 386 | 419| 358 | 33627 | 142135 | 
| 96 | 26 django/db/migrations/questioner.py | 260 | 305| 361 | 33988 | 142135 | 
| 97 | 27 django/core/management/commands/sqlmigrate.py | 31 | 69| 379 | 34367 | 142768 | 
| 98 | 27 django/db/backends/base/schema.py | 348 | 363| 154 | 34521 | 142768 | 
| 99 | 27 django/db/migrations/operations/models.py | 772 | 812| 344 | 34865 | 142768 | 
| 100 | 28 django/contrib/redirects/migrations/0001_initial.py | 1 | 40| 268 | 35133 | 143036 | 
| 101 | 28 django/db/models/base.py | 1423 | 1438| 153 | 35286 | 143036 | 
| 102 | 29 django/db/migrations/operations/fields.py | 258 | 324| 519 | 35805 | 145529 | 
| 103 | 30 django/db/migrations/loader.py | 159 | 185| 291 | 36096 | 148637 | 
| 104 | **30 django/db/migrations/autodetector.py** | 1229 | 1254| 245 | 36341 | 148637 | 
| 105 | 31 django/db/backends/mysql/creation.py | 32 | 56| 253 | 36594 | 149276 | 
| 106 | 31 django/db/migrations/state.py | 133 | 168| 391 | 36985 | 149276 | 
| 107 | 31 django/db/migrations/operations/models.py | 41 | 104| 513 | 37498 | 149276 | 
| 108 | 31 django/db/migrations/questioner.py | 198 | 216| 233 | 37731 | 149276 | 
| 109 | **31 django/db/migrations/autodetector.py** | 1038 | 1058| 134 | 37865 | 149276 | 
| 110 | 31 django/db/backends/sqlite3/schema.py | 86 | 99| 181 | 38046 | 149276 | 
| 111 | 32 django/contrib/gis/db/backends/mysql/schema.py | 65 | 78| 121 | 38167 | 149897 | 
| 112 | 32 django/db/models/indexes.py | 90 | 116| 251 | 38418 | 149897 | 
| 113 | 32 django/db/migrations/operations/models.py | 598 | 615| 163 | 38581 | 149897 | 
| 114 | 32 django/db/models/fields/related.py | 139 | 166| 201 | 38782 | 149897 | 
| 115 | 32 django/db/models/base.py | 1333 | 1358| 184 | 38966 | 149897 | 
| 116 | 33 django/db/backends/mysql/base.py | 296 | 334| 402 | 39368 | 153346 | 
| 117 | 34 django/db/backends/mysql/operations.py | 1 | 35| 282 | 39650 | 157073 | 
| 118 | 35 django/contrib/sessions/migrations/0001_initial.py | 1 | 31| 162 | 39812 | 157235 | 
| 119 | 35 django/contrib/gis/db/backends/mysql/schema.py | 40 | 63| 190 | 40002 | 157235 | 
| 120 | 36 django/db/backends/oracle/creation.py | 130 | 165| 399 | 40401 | 161128 | 
| 121 | 37 django/contrib/auth/migrations/0008_alter_user_username_max_length.py | 1 | 25| 138 | 40539 | 161266 | 
| 122 | 38 django/core/checks/model_checks.py | 1 | 86| 665 | 41204 | 163051 | 
| 123 | 39 django/db/backends/postgresql/schema.py | 184 | 210| 351 | 41555 | 165219 | 
| 124 | 39 django/db/backends/base/schema.py | 684 | 754| 799 | 42354 | 165219 | 
| 125 | **39 django/db/migrations/autodetector.py** | 1256 | 1268| 131 | 42485 | 165219 | 
| 126 | 39 django/db/models/fields/__init__.py | 2372 | 2422| 339 | 42824 | 165219 | 
| 127 | 40 django/core/management/commands/squashmigrations.py | 206 | 219| 112 | 42936 | 167092 | 
| 128 | 40 django/db/migrations/operations/models.py | 580 | 596| 215 | 43151 | 167092 | 
| 129 | 41 django/db/backends/sqlite3/creation.py | 84 | 104| 174 | 43325 | 167943 | 
| 130 | 41 django/db/migrations/state.py | 415 | 439| 219 | 43544 | 167943 | 
| 131 | 41 django/db/backends/base/schema.py | 755 | 833| 826 | 44370 | 167943 | 
| 132 | **41 django/db/migrations/autodetector.py** | 804 | 868| 676 | 45046 | 167943 | 
| 133 | 41 django/db/migrations/state.py | 232 | 257| 240 | 45286 | 167943 | 
| 134 | 41 django/core/management/commands/migrate.py | 162 | 227| 632 | 45918 | 167943 | 
| 135 | 41 django/db/backends/postgresql/schema.py | 227 | 239| 152 | 46070 | 167943 | 
| 136 | 41 django/db/models/fields/related.py | 120 | 137| 155 | 46225 | 167943 | 
| 137 | **41 django/db/migrations/autodetector.py** | 1078 | 1098| 136 | 46361 | 167943 | 
| 138 | 41 django/db/migrations/operations/models.py | 416 | 466| 410 | 46771 | 167943 | 
| 139 | 41 django/db/migrations/state.py | 207 | 230| 247 | 47018 | 167943 | 
| 140 | 42 django/db/models/options.py | 864 | 896| 225 | 47243 | 175310 | 
| 141 | 42 django/db/backends/sqlite3/creation.py | 51 | 82| 317 | 47560 | 175310 | 
| 142 | 42 django/db/migrations/operations/models.py | 469 | 498| 168 | 47728 | 175310 | 
| 143 | 42 django/db/backends/base/schema.py | 1172 | 1187| 170 | 47898 | 175310 | 
| 144 | 42 django/db/models/fields/__init__.py | 2425 | 2475| 313 | 48211 | 175310 | 
| 145 | 42 django/db/backends/oracle/creation.py | 30 | 100| 722 | 48933 | 175310 | 
| 146 | 42 django/contrib/admin/checks.py | 255 | 285| 229 | 49162 | 175310 | 
| 147 | 42 django/db/backends/base/schema.py | 1310 | 1332| 173 | 49335 | 175310 | 
| 148 | 43 django/db/migrations/recorder.py | 46 | 97| 390 | 49725 | 175987 | 
| 149 | 43 django/core/management/commands/makemigrations.py | 247 | 334| 865 | 50590 | 175987 | 
| 150 | 44 django/contrib/contenttypes/migrations/0002_remove_content_type_name.py | 1 | 40| 217 | 50807 | 176204 | 
| 151 | 44 django/db/models/base.py | 1360 | 1389| 205 | 51012 | 176204 | 
| 152 | 45 django/db/backends/oracle/schema.py | 1 | 44| 454 | 51466 | 178347 | 
| 153 | 45 django/db/backends/base/schema.py | 968 | 987| 296 | 51762 | 178347 | 
| 154 | 45 django/db/models/options.py | 1 | 35| 300 | 52062 | 178347 | 
| 155 | **45 django/db/migrations/autodetector.py** | 431 | 460| 265 | 52327 | 178347 | 
| 156 | 45 django/db/models/indexes.py | 118 | 140| 215 | 52542 | 178347 | 
| 157 | 45 django/db/backends/base/schema.py | 1 | 29| 209 | 52751 | 178347 | 
| 158 | 45 django/db/backends/oracle/schema.py | 152 | 206| 493 | 53244 | 178347 | 
| 159 | 45 django/db/backends/base/schema.py | 51 | 119| 785 | 54029 | 178347 | 
| 160 | 45 django/db/models/base.py | 404 | 509| 913 | 54942 | 178347 | 
| 161 | 46 django/contrib/auth/migrations/0004_alter_user_username_opts.py | 1 | 24| 150 | 55092 | 178497 | 
| 162 | 46 django/db/models/indexes.py | 172 | 189| 205 | 55297 | 178497 | 
| 163 | 47 django/db/backends/mysql/features.py | 65 | 128| 640 | 55937 | 180691 | 
| 164 | 48 django/db/models/__init__.py | 1 | 53| 619 | 56556 | 181310 | 
| 165 | 48 django/db/migrations/state.py | 1 | 29| 230 | 56786 | 181310 | 
| 166 | 49 django/contrib/flatpages/migrations/0001_initial.py | 1 | 40| 307 | 57093 | 181617 | 
| 167 | 50 django/core/management/commands/showmigrations.py | 46 | 67| 158 | 57251 | 182887 | 
| 168 | 51 django/db/migrations/executor.py | 289 | 382| 843 | 58094 | 186241 | 
| 169 | 52 django/db/backends/postgresql/creation.py | 56 | 81| 247 | 58341 | 186910 | 
| 170 | 52 django/db/backends/mysql/creation.py | 1 | 30| 221 | 58562 | 186910 | 
| 171 | 53 django/contrib/gis/utils/layermapping.py | 286 | 300| 124 | 58686 | 192363 | 
| 172 | 53 django/db/models/fields/__init__.py | 338 | 365| 203 | 58889 | 192363 | 
| 173 | 53 django/db/backends/postgresql/creation.py | 39 | 54| 173 | 59062 | 192363 | 
| 174 | 53 django/db/models/base.py | 1132 | 1159| 286 | 59348 | 192363 | 
| 175 | 53 django/db/backends/mysql/operations.py | 222 | 279| 437 | 59785 | 192363 | 
| 176 | 54 django/contrib/auth/migrations/0010_alter_group_name_max_length.py | 1 | 17| 0 | 59785 | 192440 | 
| 177 | 55 django/db/backends/mysql/validation.py | 33 | 70| 287 | 60072 | 192960 | 
| 178 | 55 django/forms/models.py | 968 | 1001| 367 | 60439 | 192960 | 
| 179 | 55 django/db/backends/mysql/features.py | 130 | 226| 790 | 61229 | 192960 | 
| 180 | 55 django/db/backends/base/schema.py | 32 | 48| 166 | 61395 | 192960 | 
| 181 | 56 django/db/backends/base/features.py | 1 | 112| 895 | 62290 | 195967 | 
| 182 | 56 django/db/migrations/operations/fields.py | 229 | 255| 204 | 62494 | 195967 | 
| 183 | 56 django/db/models/fields/__init__.py | 208 | 242| 234 | 62728 | 195967 | 
| 184 | **56 django/db/migrations/autodetector.py** | 935 | 1018| 919 | 63647 | 195967 | 
| 185 | 56 django/db/backends/sqlite3/creation.py | 23 | 49| 239 | 63886 | 195967 | 
| 186 | 56 django/db/backends/mysql/creation.py | 58 | 69| 178 | 64064 | 195967 | 
| 187 | 57 django/db/utils.py | 255 | 297| 322 | 64386 | 197974 | 


### Hint

```
this is a small project which can repeat the bug.
Thank your for your report. I guess this is a bug in the auto-detector where the AlterUniqueTogether should appear before the AlterField that adds the Field.unique=True. I assume this is the case because generate_altered_unique_together is run after generate_altered_fields and the former doesn't add any dependencies on ensure operations are properly re-ordered. Not sure it's worth adjusting AddConstraint ordering as well since the chance of adding a UniqueConstraint with a colliding .name are really slim. Xiang, can you confirm that re-ordering your operations so that AlterUniqueTogether is performed before AlterField addresses your issue?
I think changing the format of index name if the CONSTRAINT is create by the unique_together will work either. e.g.: ALTER TABLE `testapp_mymodel` ADD CONSTRAINT `testapp_mymodel_name_ba5e2bd2_uniq_by_unique_together` UNIQUE (`name`); ALTER TABLE `testapp_mymodel` ADD CONSTRAINT `testapp_mymodel_name_ba5e2bd2_uniq_by_field` UNIQUE (`name`); I tried the sql in reverse order. It works. wangx@aliyun:~/testunique$ python3 manage.py migrate testapp 0002 Operations to perform: Target specific migration: 0002_auto_20200422_1247, from testapp Running migrations: Applying testapp.0002_auto_20200422_1247...Traceback (most recent call last): File "/usr/local/lib/python3.6/dist-packages/django/db/backends/utils.py", line 86, in _execute return self.cursor.execute(sql, params) File "/usr/local/lib/python3.6/dist-packages/django/db/backends/mysql/base.py", line 74, in execute return self.cursor.execute(query, args) File "/usr/local/lib/python3.6/dist-packages/MySQLdb/cursors.py", line 209, in execute res = self._query(query) File "/usr/local/lib/python3.6/dist-packages/MySQLdb/cursors.py", line 315, in _query db.query(q) File "/usr/local/lib/python3.6/dist-packages/MySQLdb/connections.py", line 239, in query _mysql.connection.query(self, query) MySQLdb._exceptions.OperationalError: (1061, "Duplicate key name 'testapp_mymodel_name_ba5e2bd2_uniq'") The above exception was the direct cause of the following exception: Traceback (most recent call last): File "manage.py", line 21, in <module> main() File "manage.py", line 17, in main execute_from_command_line(sys.argv) File "/usr/local/lib/python3.6/dist-packages/django/core/management/__init__.py", line 401, in execute_from_command_line utility.execute() File "/usr/local/lib/python3.6/dist-packages/django/core/management/__init__.py", line 395, in execute self.fetch_command(subcommand).run_from_argv(self.argv) File "/usr/local/lib/python3.6/dist-packages/django/core/management/base.py", line 328, in run_from_argv self.execute(*args, **cmd_options) File "/usr/local/lib/python3.6/dist-packages/django/core/management/base.py", line 369, in execute output = self.handle(*args, **options) File "/usr/local/lib/python3.6/dist-packages/django/core/management/base.py", line 83, in wrapped res = handle_func(*args, **kwargs) File "/usr/local/lib/python3.6/dist-packages/django/core/management/commands/migrate.py", line 233, in handle fake_initial=fake_initial, File "/usr/local/lib/python3.6/dist-packages/django/db/migrations/executor.py", line 117, in migrate state = self._migrate_all_forwards(state, plan, full_plan, fake=fake, fake_initial=fake_initial) File "/usr/local/lib/python3.6/dist-packages/django/db/migrations/executor.py", line 147, in _migrate_all_forwards state = self.apply_migration(state, migration, fake=fake, fake_initial=fake_initial) File "/usr/local/lib/python3.6/dist-packages/django/db/migrations/executor.py", line 245, in apply_migration state = migration.apply(state, schema_editor) File "/usr/local/lib/python3.6/dist-packages/django/db/migrations/migration.py", line 124, in apply operation.database_forwards(self.app_label, schema_editor, old_state, project_state) File "/usr/local/lib/python3.6/dist-packages/django/db/migrations/operations/fields.py", line 249, in database_forwards schema_editor.alter_field(from_model, from_field, to_field) File "/usr/local/lib/python3.6/dist-packages/django/db/backends/base/schema.py", line 565, in alter_field old_db_params, new_db_params, strict) File "/usr/local/lib/python3.6/dist-packages/django/db/backends/base/schema.py", line 745, in _alter_field self.execute(self._create_unique_sql(model, [new_field.column])) File "/usr/local/lib/python3.6/dist-packages/django/db/backends/base/schema.py", line 142, in execute cursor.execute(sql, params) File "/usr/local/lib/python3.6/dist-packages/django/db/backends/utils.py", line 100, in execute return super().execute(sql, params) File "/usr/local/lib/python3.6/dist-packages/django/db/backends/utils.py", line 68, in execute return self._execute_with_wrappers(sql, params, many=False, executor=self._execute) File "/usr/local/lib/python3.6/dist-packages/django/db/backends/utils.py", line 77, in _execute_with_wrappers return executor(sql, params, many, context) File "/usr/local/lib/python3.6/dist-packages/django/db/backends/utils.py", line 86, in _execute return self.cursor.execute(sql, params) File "/usr/local/lib/python3.6/dist-packages/django/db/utils.py", line 90, in __exit__ raise dj_exc_value.with_traceback(traceback) from exc_value File "/usr/local/lib/python3.6/dist-packages/django/db/backends/utils.py", line 86, in _execute return self.cursor.execute(sql, params) File "/usr/local/lib/python3.6/dist-packages/django/db/backends/mysql/base.py", line 74, in execute return self.cursor.execute(query, args) File "/usr/local/lib/python3.6/dist-packages/MySQLdb/cursors.py", line 209, in execute res = self._query(query) File "/usr/local/lib/python3.6/dist-packages/MySQLdb/cursors.py", line 315, in _query db.query(q) File "/usr/local/lib/python3.6/dist-packages/MySQLdb/connections.py", line 239, in query _mysql.connection.query(self, query) django.db.utils.OperationalError: (1061, "Duplicate key name 'testapp_mymodel_name_ba5e2bd2_uniq'") wangx@aliyun:~/testunique$ python3 manage.py sqlmigrate testapp 0002 -- -- Alter field name on mymodel -- ALTER TABLE `testapp_mymodel` ADD CONSTRAINT `testapp_mymodel_name_ba5e2bd2_uniq` UNIQUE (`name`); -- -- Alter unique_together for mymodel (0 constraint(s)) -- ALTER TABLE `testapp_mymodel` DROP INDEX `testapp_mymodel_name_ba5e2bd2_uniq`; wangx@aliyun:~/testunique$ mysql -u test -p Enter password: Welcome to the MySQL monitor. Commands end with ; or \g. Your MySQL connection id is 36 Server version: 5.7.29-0ubuntu0.18.04.1 (Ubuntu) Copyright (c) 2000, 2020, Oracle and/or its affiliates. All rights reserved. Oracle is a registered trademark of Oracle Corporation and/or its affiliates. Other names may be trademarks of their respective owners. Type 'help;' or '\h' for help. Type '\c' to clear the current input statement. mysql> use testunique; Reading table information for completion of table and column names You can turn off this feature to get a quicker startup with -A Database changed mysql> ALTER TABLE `testapp_mymodel` DROP INDEX `testapp_mymodel_name_ba5e2bd2_uniq`; Query OK, 0 rows affected (0.01 sec) Records: 0 Duplicates: 0 Warnings: 0 mysql> ALTER TABLE `testapp_mymodel` ADD CONSTRAINT `testapp_mymodel_name_ba5e2bd2_uniq` UNIQUE (`name`); Query OK, 0 rows affected (0.01 sec) Records: 0 Duplicates: 0 Warnings: 0 mysql> ALTER TABLE `testapp_mymodel` ADD CONSTRAINT `testapp_mymodel_name_ba5e2bd2_uniq_by_unique_together` UNIQUE (`name`); Query OK, 0 rows affected, 1 warning (0.00 sec) Records: 0 Duplicates: 0 Warnings: 1
I think we can consider limit the length of unique_together. If a user set unique_together containing only one field, we can raise an error and ask him to use unique=True instead.
Hi! I looked a bit into the issue. I'll share my thoughts and suggest/discuss some solutions so that we can agree on the approach. I'd be more than happy to tackle the issue then, if that is okay, once we know how we want to fix the issue :D TL;DR: some suboptimal solutions are presented, but 5/ and 6/ look the most promising to me. Please let's discuss those a bit at least :) So the issue occurs because we are changing an unique_together to a unique=True on the field (similarly, I believe the bug occurs with index_together and db_index), which will first try to create an existing index before deleting it. Some solutions: 1/ Changing the index name I think changing the format of index name if the CONSTRAINT is create by the unique_together will work either. It would somehow work and mitigate the issue at hand, but it might introduce complexity for backward compatibility. When upgrading your Django version and wanting to drop an index, Django will have to know whether the name of the index comes the previous or current version of Django, in order to know how the index is called and drop it. 2/ Forbid using unique_together (and index_together) with a single field If a user set unique_together containing only one field, we can raise an error and ask him to use unique=True instead It feels more like a workaround than a real fix of the issue. And more generally, we will have issues with backward compatibility. We can't break unique_togethers with one field from a release to the next. I guess we would need to add a deprecation warning, and really introduce a breaking behaviour in the next major release (Django 5.x then?). Which seems IMHO like a lot of trouble for a rather narrow issue. 3/ Leverage the existing foo_together_change dependency mecanism The autodetector has a similar re-order behaviour so the one we would need already implemented. When dropping a field, we add a dependency called foo_together_change to the field, which would re-order the AlterUniqueTogether operations, for the index to be dropped before the removal of the field. I tried it out for field altering (see code block below), and it would fix the issue at hand, but it wouldn't fix the reverse operation. If we changed from a unique=True to a unique_together, the dependency would still re-order the AlterUniqueTogether operation to happen before the AlterField, but we would need to first drop the index through the AlterField. So the very same issue occurs, just the other way around. diff --git a/django/db/migrations/autodetector.py b/django/db/migrations/autodetector.py index 2848adce7d..598d4649e9 100644 --- a/django/db/migrations/autodetector.py +++ b/django/db/migrations/autodetector.py @@ -987,7 +987,9 @@ class MigrationAutodetector: field=field, preserve_default=preserve_default, ), - dependencies=dependencies, + dependencies=dependencies + [ + (app_label, model_name, field_name, "foo_together_change"), + ], ) else: # We cannot alter between m2m and concrete fields 4/ Split the index dropping/creation out of the AlterField operation The bug seems related to the fact that AlterField does a lot of things, and among them is the creation/drop of indexes, which can clash with other structures. So one could probably move this part out of AlterField, but it feels like a very heavy and error-prone change to a part that is currently core to the autodetector and migrations framework. This idea is a long-shot, and also pretty vague in my head. I wouldn't actually consider this solution. 5/ Do multi-step AlterFooTogether operations This solution, and the next one, focus on the idea that index creation should operate in two steps (in migrations). First, index drops, and then, after field removal/creation/altering, add indexes. Today, AlterUniqueTogether is one step that will both create and drop indexes, which is a part of the problem. So would de-couple this, and first do the AlterFooTogether that would drop indexes, then field changes, and then again AlterFooTogether to add indexes. A small issue is that this operation is declarative with respect to the expected model state, we just go from one state to the next. So the idea would be to split the AlterFooTogether operation (in two mostly), so that the migration ends up in the correct state, but with an intermediate state that, by design, only drops indexes. An example would be (in pseudo-code): Initial model class MyModel(models.Model): name = models.CharField(max_length=32) age = models.IntegerField() class Meta: unique_together = ("name", ) becomes: class MyModel(models.Model): name = models.CharField(max_length=32, unique=True) age = models.IntegerField() class Meta: unique_together = ("age", ) would do operations like operations = [ # Dropping the "name" index migrations.AlterUniqueTogether( name='mymodel', unique_together=set(), ), # Adding the "name" index migrations.AlterField( model_name='mymodel', name='name', field=models.CharField(max_length=32, unique=True), ), # Adding the "age" index migrations.AlterUniqueTogether( name='mymodel', unique_together={("age", )}, ), ] (one could also imagine age being a unique=True first, where the AlterField will drop the index) I believe the amount of work is not that high for this solution, and we should have no issue with backward compatibility, since we keep the same logic for the operation itself. The tricky part is to the generate the intermediate state of foo_together, that will only drop the related indexes. An issue I see: we use implicitly the AlterFooTogether behaviour to serve our purpose, and not the purpose it was made for. 6/ Introduce explicit CreateFooTogether and DropFooTogether operations The cleaner solution to 5/ would be to introduce four new operations, creating and dropping unique_together and index_together. This would mimic what is done for constraints and indexes in the autodetector, having two separate steps. The issue is that, a) it will introduce a lot of new code and operations. Even if the code is not complicated, it's still content to document and maintain. And b) for backward compatibility, we would need to keep AlterFooTogether operations (forever?) the way they are, even if the operation is not generated by Django anymore. End note: From a quick look, 5/ seems like the most realistic approach, but I'd like to confirm with more experienced Django contributors/maintainers :) Thanks! David
A first PR ​https://github.com/django/django/pull/14722 with approach 5/ (from my comment above) basically.
Removing "patch needs improvement" to get some eyes to look at it :)
```

## Patch

```diff
diff --git a/django/db/migrations/autodetector.py b/django/db/migrations/autodetector.py
--- a/django/db/migrations/autodetector.py
+++ b/django/db/migrations/autodetector.py
@@ -178,8 +178,12 @@ def _detect_changes(self, convert_apps=None, graph=None):
         # Generate index removal operations before field is removed
         self.generate_removed_constraints()
         self.generate_removed_indexes()
-        # Generate field operations
+        # Generate field renaming operations.
         self.generate_renamed_fields()
+        # Generate removal of foo together.
+        self.generate_removed_altered_unique_together()
+        self.generate_removed_altered_index_together()
+        # Generate field operations.
         self.generate_removed_fields()
         self.generate_added_fields()
         self.generate_altered_fields()
@@ -1128,8 +1132,7 @@ def _get_dependencies_for_foreign_key(app_label, model_name, field, project_stat
             dependencies.append((through_app_label, through_object_name, None, True))
         return dependencies
 
-    def _generate_altered_foo_together(self, operation):
-        option_name = operation.option_name
+    def _get_altered_foo_together_operations(self, option_name):
         for app_label, model_name in sorted(self.kept_model_keys):
             old_model_name = self.renamed_models.get((app_label, model_name), model_name)
             old_model_state = self.from_state.models[app_label, old_model_name]
@@ -1157,13 +1160,49 @@ def _generate_altered_foo_together(self, operation):
                             dependencies.extend(self._get_dependencies_for_foreign_key(
                                 app_label, model_name, field, self.to_state,
                             ))
+                yield (
+                    old_value,
+                    new_value,
+                    app_label,
+                    model_name,
+                    dependencies,
+                )
 
+    def _generate_removed_altered_foo_together(self, operation):
+        for (
+            old_value,
+            new_value,
+            app_label,
+            model_name,
+            dependencies,
+        ) in self._get_altered_foo_together_operations(operation.option_name):
+            removal_value = new_value.intersection(old_value)
+            if removal_value or old_value:
                 self.add_operation(
                     app_label,
-                    operation(
-                        name=model_name,
-                        **{option_name: new_value}
-                    ),
+                    operation(name=model_name, **{operation.option_name: removal_value}),
+                    dependencies=dependencies,
+                )
+
+    def generate_removed_altered_unique_together(self):
+        self._generate_removed_altered_foo_together(operations.AlterUniqueTogether)
+
+    def generate_removed_altered_index_together(self):
+        self._generate_removed_altered_foo_together(operations.AlterIndexTogether)
+
+    def _generate_altered_foo_together(self, operation):
+        for (
+            old_value,
+            new_value,
+            app_label,
+            model_name,
+            dependencies,
+        ) in self._get_altered_foo_together_operations(operation.option_name):
+            removal_value = new_value.intersection(old_value)
+            if new_value != removal_value:
+                self.add_operation(
+                    app_label,
+                    operation(name=model_name, **{operation.option_name: new_value}),
                     dependencies=dependencies,
                 )
 

```

## Test Patch

```diff
diff --git a/tests/migrations/test_autodetector.py b/tests/migrations/test_autodetector.py
--- a/tests/migrations/test_autodetector.py
+++ b/tests/migrations/test_autodetector.py
@@ -1570,9 +1570,26 @@ def test_foo_together_ordering(self):
         )
         # Right number/type of migrations?
         self.assertNumberMigrations(changes, "otherapp", 1)
-        self.assertOperationTypes(changes, "otherapp", 0, ["AlterUniqueTogether", "AlterIndexTogether"])
-        self.assertOperationAttributes(changes, "otherapp", 0, 0, name="book", unique_together={("title", "author")})
-        self.assertOperationAttributes(changes, "otherapp", 0, 1, name="book", index_together={("title", "author")})
+        self.assertOperationTypes(changes, 'otherapp', 0, [
+            'AlterUniqueTogether',
+            'AlterIndexTogether',
+            'AlterUniqueTogether',
+            'AlterIndexTogether',
+        ])
+        self.assertOperationAttributes(
+            changes, 'otherapp', 0, 0, name='book', unique_together=set(),
+        )
+        self.assertOperationAttributes(
+            changes, 'otherapp', 0, 1, name='book', index_together=set(),
+        )
+        self.assertOperationAttributes(
+            changes, 'otherapp', 0, 2, name='book',
+            unique_together={('title', 'author')},
+        )
+        self.assertOperationAttributes(
+            changes, 'otherapp', 0, 3, name='book',
+            index_together={('title', 'author')},
+        )
 
     def test_add_field_and_foo_together(self):
         """
@@ -1619,10 +1636,100 @@ def test_remove_field_and_foo_together(self):
         )
         # Right number/type of migrations?
         self.assertNumberMigrations(changes, "otherapp", 1)
-        self.assertOperationTypes(changes, "otherapp", 0, ["AlterUniqueTogether", "AlterIndexTogether", "RemoveField"])
-        self.assertOperationAttributes(changes, "otherapp", 0, 0, name="book", unique_together={("author", "title")})
-        self.assertOperationAttributes(changes, "otherapp", 0, 1, name="book", index_together={("author", "title")})
-        self.assertOperationAttributes(changes, "otherapp", 0, 2, model_name="book", name="newfield")
+        self.assertOperationTypes(changes, 'otherapp', 0, [
+            'AlterUniqueTogether',
+            'AlterIndexTogether',
+            'AlterUniqueTogether',
+            'AlterIndexTogether',
+            'RemoveField',
+        ])
+        self.assertOperationAttributes(
+            changes, 'otherapp', 0, 0, name='book', unique_together=set(),
+        )
+        self.assertOperationAttributes(
+            changes, 'otherapp', 0, 1, name='book', index_together=set(),
+        )
+        self.assertOperationAttributes(
+            changes, 'otherapp', 0, 2, name='book',
+            unique_together={('author', 'title')},
+        )
+        self.assertOperationAttributes(
+            changes, 'otherapp', 0, 3, name='book',
+            index_together={('author', 'title')},
+        )
+        self.assertOperationAttributes(
+            changes, 'otherapp', 0, 4, model_name='book', name='newfield',
+        )
+
+    def test_alter_field_and_foo_together(self):
+        """Fields are altered after deleting some index/unique_together."""
+        initial_author = ModelState('testapp', 'Author', [
+            ('id', models.AutoField(primary_key=True)),
+            ('name', models.CharField(max_length=200)),
+            ('age', models.IntegerField(db_index=True)),
+        ], {
+            'unique_together': {('name',)},
+        })
+        author_reversed_constraints = ModelState('testapp', 'Author', [
+            ('id', models.AutoField(primary_key=True)),
+            ('name', models.CharField(max_length=200, unique=True)),
+            ('age', models.IntegerField()),
+        ], {
+            'index_together': {('age',)},
+        })
+        changes = self.get_changes([initial_author], [author_reversed_constraints])
+
+        self.assertNumberMigrations(changes, 'testapp', 1)
+        self.assertOperationTypes(changes, 'testapp', 0, [
+            'AlterUniqueTogether',
+            'AlterField',
+            'AlterField',
+            'AlterIndexTogether',
+        ])
+        self.assertOperationAttributes(
+            changes, 'testapp', 0, 0, name='author', unique_together=set(),
+        )
+        self.assertOperationAttributes(
+            changes, 'testapp', 0, 1, model_name='author', name='age',
+        )
+        self.assertOperationAttributes(
+            changes, 'testapp', 0, 2, model_name='author', name='name',
+        )
+        self.assertOperationAttributes(
+            changes, 'testapp', 0, 3, name='author', index_together={('age',)},
+        )
+
+    def test_partly_alter_foo_together(self):
+        initial_author = ModelState('testapp', 'Author', [
+            ('id', models.AutoField(primary_key=True)),
+            ('name', models.CharField(max_length=200)),
+            ('age', models.IntegerField()),
+        ], {
+            'unique_together': {('name',), ('age',)},
+            'index_together': {('name',)},
+        })
+        author_reversed_constraints = ModelState('testapp', 'Author', [
+            ('id', models.AutoField(primary_key=True)),
+            ('name', models.CharField(max_length=200)),
+            ('age', models.IntegerField()),
+        ], {
+            'unique_together': {('age',)},
+            'index_together': {('name',), ('age',)},
+        })
+        changes = self.get_changes([initial_author], [author_reversed_constraints])
+
+        self.assertNumberMigrations(changes, 'testapp', 1)
+        self.assertOperationTypes(changes, 'testapp', 0, [
+            'AlterUniqueTogether',
+            'AlterIndexTogether',
+        ])
+        self.assertOperationAttributes(
+            changes, 'testapp', 0, 0, name='author', unique_together={('age',)},
+        )
+        self.assertOperationAttributes(
+            changes, 'testapp', 0, 1, name='author',
+            index_together={('name',), ('age',)},
+        )
 
     def test_rename_field_and_foo_together(self):
         """
@@ -1635,11 +1742,27 @@ def test_rename_field_and_foo_together(self):
         )
         # Right number/type of migrations?
         self.assertNumberMigrations(changes, "otherapp", 1)
-        self.assertOperationTypes(changes, "otherapp", 0, ["RenameField", "AlterUniqueTogether", "AlterIndexTogether"])
-        self.assertOperationAttributes(changes, "otherapp", 0, 1, name="book", unique_together={
-            ("title", "newfield2")
-        })
-        self.assertOperationAttributes(changes, "otherapp", 0, 2, name="book", index_together={("title", "newfield2")})
+        self.assertOperationTypes(changes, 'otherapp', 0, [
+            'RenameField',
+            'AlterUniqueTogether',
+            'AlterIndexTogether',
+            'AlterUniqueTogether',
+            'AlterIndexTogether',
+        ])
+        self.assertOperationAttributes(
+            changes, 'otherapp', 0, 1, name='book', unique_together=set(),
+        )
+        self.assertOperationAttributes(
+            changes, 'otherapp', 0, 2, name='book', index_together=set(),
+        )
+        self.assertOperationAttributes(
+            changes, 'otherapp', 0, 3, name='book',
+            unique_together={('title', 'newfield2')},
+        )
+        self.assertOperationAttributes(
+            changes, 'otherapp', 0, 4, name='book',
+            index_together={('title', 'newfield2')},
+        )
 
     def test_proxy(self):
         """The autodetector correctly deals with proxy models."""

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
### 2 - django/db/backends/base/schema.py:

Start line: 415, End line: 429

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
            self._delete_composed_index(model, fields, {'unique': True}, self.sql_delete_unique)
        # Created uniques
        for field_names in news.difference(olds):
            fields = [model._meta.get_field(field) for field in field_names]
            self.execute(self._create_unique_sql(model, fields))
```
### 3 - django/db/migrations/autodetector.py:

Start line: 1170, End line: 1191

```python
class MigrationAutodetector:

    def generate_altered_unique_together(self):
        self._generate_altered_foo_together(operations.AlterUniqueTogether)

    def generate_altered_index_together(self):
        self._generate_altered_foo_together(operations.AlterIndexTogether)

    def generate_altered_db_table(self):
        models_to_check = self.kept_model_keys.union(self.kept_proxy_keys, self.kept_unmanaged_keys)
        for app_label, model_name in sorted(models_to_check):
            old_model_name = self.renamed_models.get((app_label, model_name), model_name)
            old_model_state = self.from_state.models[app_label, old_model_name]
            new_model_state = self.to_state.models[app_label, model_name]
            old_db_table_name = old_model_state.options.get('db_table')
            new_db_table_name = new_model_state.options.get('db_table')
            if old_db_table_name != new_db_table_name:
                self.add_operation(
                    app_label,
                    operations.AlterModelTable(
                        name=model_name,
                        table=new_db_table_name,
                    )
                )
```
### 4 - django/db/models/base.py:

Start line: 1087, End line: 1130

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
### 5 - django/db/models/base.py:

Start line: 1178, End line: 1206

```python
class Model(metaclass=ModelBase):

    def unique_error_message(self, model_class, unique_check):
        opts = model_class._meta

        params = {
            'model': self,
            'model_class': model_class,
            'model_name': capfirst(opts.verbose_name),
            'unique_check': unique_check,
        }

        # A unique field
        if len(unique_check) == 1:
            field = opts.get_field(unique_check[0])
            params['field_label'] = capfirst(field.verbose_name)
            return ValidationError(
                message=field.error_messages['unique'],
                code='unique',
                params=params,
            )

        # unique_together
        else:
            field_labels = [capfirst(opts.get_field(f).verbose_name) for f in unique_check]
            params['field_labels'] = get_text_list(field_labels, _('and'))
            return ValidationError(
                message=_("%(model_name)s with this %(field_labels)s already exists."),
                code='unique_together',
                params=params,
            )
```
### 6 - django/db/models/constraints.py:

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
### 7 - django/db/models/base.py:

Start line: 1607, End line: 1632

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
                    id='models.E010',
                )
            ]

        elif any(not isinstance(fields, (tuple, list)) for fields in cls._meta.unique_together):
            return [
                checks.Error(
                    "All 'unique_together' elements must be lists or tuples.",
                    obj=cls,
                    id='models.E011',
                )
            ]

        else:
            errors = []
            for fields in cls._meta.unique_together:
                errors.extend(cls._check_local_fields(fields, "unique_together"))
            return errors
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
### 9 - django/db/migrations/operations/models.py:

Start line: 531, End line: 550

```python
class AlterUniqueTogether(AlterTogetherOptionOperation):
    """
    Change the value of unique_together to the target one.
    Input value of unique_together must be a set of tuples.
    """
    option_name = 'unique_together'

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
### 10 - django/db/models/constraints.py:

Start line: 93, End line: 185

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
    ):
        if not name:
            raise ValueError('A unique constraint must be named.')
        if not expressions and not fields:
            raise ValueError(
                'At least one field or expression is required to define a '
                'unique constraint.'
            )
        if expressions and fields:
            raise ValueError(
                'UniqueConstraint.fields and expressions are mutually exclusive.'
            )
        if not isinstance(condition, (type(None), Q)):
            raise ValueError('UniqueConstraint.condition must be a Q instance.')
        if condition and deferrable:
            raise ValueError(
                'UniqueConstraint with conditions cannot be deferred.'
            )
        if include and deferrable:
            raise ValueError(
                'UniqueConstraint with include fields cannot be deferred.'
            )
        if opclasses and deferrable:
            raise ValueError(
                'UniqueConstraint with opclasses cannot be deferred.'
            )
        if expressions and deferrable:
            raise ValueError(
                'UniqueConstraint with expressions cannot be deferred.'
            )
        if expressions and opclasses:
            raise ValueError(
                'UniqueConstraint.opclasses cannot be used with expressions. '
                'Use django.contrib.postgres.indexes.OpClass() instead.'
            )
        if not isinstance(deferrable, (type(None), Deferrable)):
            raise ValueError(
                'UniqueConstraint.deferrable must be a Deferrable instance.'
            )
        if not isinstance(include, (type(None), list, tuple)):
            raise ValueError('UniqueConstraint.include must be a list or tuple.')
        if not isinstance(opclasses, (list, tuple)):
            raise ValueError('UniqueConstraint.opclasses must be a list or tuple.')
        if opclasses and len(fields) != len(opclasses):
            raise ValueError(
                'UniqueConstraint.fields and UniqueConstraint.opclasses must '
                'have the same number of elements.'
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
        super().__init__(name)

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
### 51 - django/db/migrations/autodetector.py:

Start line: 1131, End line: 1168

```python
class MigrationAutodetector:

    def _generate_altered_foo_together(self, operation):
        option_name = operation.option_name
        for app_label, model_name in sorted(self.kept_model_keys):
            old_model_name = self.renamed_models.get((app_label, model_name), model_name)
            old_model_state = self.from_state.models[app_label, old_model_name]
            new_model_state = self.to_state.models[app_label, model_name]

            # We run the old version through the field renames to account for those
            old_value = old_model_state.options.get(option_name)
            old_value = {
                tuple(
                    self.renamed_fields.get((app_label, model_name, n), n)
                    for n in unique
                )
                for unique in old_value
            } if old_value else set()

            new_value = new_model_state.options.get(option_name)
            new_value = set(new_value) if new_value else set()

            if old_value != new_value:
                dependencies = []
                for foo_togethers in new_value:
                    for field_name in foo_togethers:
                        field = new_model_state.get_field(field_name)
                        if field.remote_field and field.remote_field.model:
                            dependencies.extend(self._get_dependencies_for_foreign_key(
                                app_label, model_name, field, self.to_state,
                            ))

                self.add_operation(
                    app_label,
                    operation(
                        name=model_name,
                        **{option_name: new_value}
                    ),
                    dependencies=dependencies,
                )
```
### 60 - django/db/migrations/autodetector.py:

Start line: 1020, End line: 1036

```python
class MigrationAutodetector:

    def create_altered_indexes(self):
        option_name = operations.AddIndex.option_name
        for app_label, model_name in sorted(self.kept_model_keys):
            old_model_name = self.renamed_models.get((app_label, model_name), model_name)
            old_model_state = self.from_state.models[app_label, old_model_name]
            new_model_state = self.to_state.models[app_label, model_name]

            old_indexes = old_model_state.options[option_name]
            new_indexes = new_model_state.options[option_name]
            add_idx = [idx for idx in new_indexes if idx not in old_indexes]
            rem_idx = [idx for idx in old_indexes if idx not in new_indexes]

            self.altered_indexes.update({
                (app_label, model_name): {
                    'added_indexes': add_idx, 'removed_indexes': rem_idx,
                }
            })
```
### 63 - django/db/migrations/autodetector.py:

Start line: 462, End line: 514

```python
class MigrationAutodetector:

    def generate_renamed_models(self):
        """
        Find any renamed models, generate the operations for them, and remove
        the old entry from the model lists. Must be run before other
        model-level generation.
        """
        self.renamed_models = {}
        self.renamed_models_rel = {}
        added_models = self.new_model_keys - self.old_model_keys
        for app_label, model_name in sorted(added_models):
            model_state = self.to_state.models[app_label, model_name]
            model_fields_def = self.only_relation_agnostic_fields(model_state.fields)

            removed_models = self.old_model_keys - self.new_model_keys
            for rem_app_label, rem_model_name in removed_models:
                if rem_app_label == app_label:
                    rem_model_state = self.from_state.models[rem_app_label, rem_model_name]
                    rem_model_fields_def = self.only_relation_agnostic_fields(rem_model_state.fields)
                    if model_fields_def == rem_model_fields_def:
                        if self.questioner.ask_rename_model(rem_model_state, model_state):
                            dependencies = []
                            fields = list(model_state.fields.values()) + [
                                field.remote_field
                                for relations in self.to_state.relations[app_label, model_name].values()
                                for field in relations.values()
                            ]
                            for field in fields:
                                if field.is_relation:
                                    dependencies.extend(
                                        self._get_dependencies_for_foreign_key(
                                            app_label, model_name, field, self.to_state,
                                        )
                                    )
                            self.add_operation(
                                app_label,
                                operations.RenameModel(
                                    old_name=rem_model_state.name,
                                    new_name=model_state.name,
                                ),
                                dependencies=dependencies,
                            )
                            self.renamed_models[app_label, model_name] = rem_model_name
                            renamed_models_rel_key = '%s.%s' % (
                                rem_model_state.app_label,
                                rem_model_state.name_lower,
                            )
                            self.renamed_models_rel[renamed_models_rel_key] = '%s.%s' % (
                                model_state.app_label,
                                model_state.name_lower,
                            )
                            self.old_model_keys.remove((rem_app_label, rem_model_name))
                            self.old_model_keys.add((app_label, model_name))
                            break
```
### 72 - django/db/migrations/autodetector.py:

Start line: 719, End line: 802

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
            # Gather related fields
            related_fields = {}
            for field_name, field in model_state.fields.items():
                if field.remote_field:
                    if field.remote_field.model:
                        related_fields[field_name] = field
                    if getattr(field.remote_field, 'through', None):
                        related_fields[field_name] = field
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
            relations = self.from_state.relations
            for (related_object_app_label, object_name), relation_related_fields in (
                relations[app_label, model_name].items()
            ):
                for field_name, field in relation_related_fields.items():
                    dependencies.append(
                        (related_object_app_label, object_name, field_name, False),
                    )
                    if not field.many_to_many:
                        dependencies.append(
                            (related_object_app_label, object_name, field_name, 'alter'),
                        )

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
### 77 - django/db/migrations/autodetector.py:

Start line: 533, End line: 684

```python
class MigrationAutodetector:

    def generate_created_models(self):
        # ... other code
        for app_label, model_name in all_added_models:
            model_state = self.to_state.models[app_label, model_name]
            # Gather related fields
            related_fields = {}
            primary_key_rel = None
            for field_name, field in model_state.fields.items():
                if field.remote_field:
                    if field.remote_field.model:
                        if field.primary_key:
                            primary_key_rel = field.remote_field.model
                        elif not field.remote_field.parent_link:
                            related_fields[field_name] = field
                    if getattr(field.remote_field, 'through', None):
                        related_fields[field_name] = field

            # Are there indexes/unique|index_together to defer?
            indexes = model_state.options.pop('indexes')
            constraints = model_state.options.pop('constraints')
            unique_together = model_state.options.pop('unique_together', None)
            index_together = model_state.options.pop('index_together', None)
            order_with_respect_to = model_state.options.pop('order_with_respect_to', None)
            # Depend on the deletion of any possible proxy version of us
            dependencies = [
                (app_label, model_name, None, False),
            ]
            # Depend on all bases
            for base in model_state.bases:
                if isinstance(base, str) and "." in base:
                    base_app_label, base_name = base.split(".", 1)
                    dependencies.append((base_app_label, base_name, None, True))
                    # Depend on the removal of base fields if the new model has
                    # a field with the same name.
                    old_base_model_state = self.from_state.models.get((base_app_label, base_name))
                    new_base_model_state = self.to_state.models.get((base_app_label, base_name))
                    if old_base_model_state and new_base_model_state:
                        removed_base_fields = set(old_base_model_state.fields).difference(
                            new_base_model_state.fields,
                        ).intersection(model_state.fields)
                        for removed_base_field in removed_base_fields:
                            dependencies.append((base_app_label, base_name, removed_base_field, False))
            # Depend on the other end of the primary key if it's a relation
            if primary_key_rel:
                dependencies.append(
                    resolve_relation(
                        primary_key_rel, app_label, model_name,
                    ) + (None, True)
                )
            # Generate creation operation
            self.add_operation(
                app_label,
                operations.CreateModel(
                    name=model_state.name,
                    fields=[d for d in model_state.fields.items() if d[0] not in related_fields],
                    options=model_state.options,
                    bases=model_state.bases,
                    managers=model_state.managers,
                ),
                dependencies=dependencies,
                beginning=True,
            )

            # Don't add operations which modify the database for unmanaged models
            if not model_state.options.get('managed', True):
                continue

            # Generate operations for each related field
            for name, field in sorted(related_fields.items()):
                dependencies = self._get_dependencies_for_foreign_key(
                    app_label, model_name, field, self.to_state,
                )
                # Depend on our own model being created
                dependencies.append((app_label, model_name, None, True))
                # Make operation
                self.add_operation(
                    app_label,
                    operations.AddField(
                        model_name=model_name,
                        name=name,
                        field=field,
                    ),
                    dependencies=list(set(dependencies)),
                )
            # Generate other opns
            if order_with_respect_to:
                self.add_operation(
                    app_label,
                    operations.AlterOrderWithRespectTo(
                        name=model_name,
                        order_with_respect_to=order_with_respect_to,
                    ),
                    dependencies=[
                        (app_label, model_name, order_with_respect_to, True),
                        (app_label, model_name, None, True),
                    ]
                )
            related_dependencies = [
                (app_label, model_name, name, True)
                for name in sorted(related_fields)
            ]
            related_dependencies.append((app_label, model_name, None, True))
            for index in indexes:
                self.add_operation(
                    app_label,
                    operations.AddIndex(
                        model_name=model_name,
                        index=index,
                    ),
                    dependencies=related_dependencies,
                )
            for constraint in constraints:
                self.add_operation(
                    app_label,
                    operations.AddConstraint(
                        model_name=model_name,
                        constraint=constraint,
                    ),
                    dependencies=related_dependencies,
                )
            if unique_together:
                self.add_operation(
                    app_label,
                    operations.AlterUniqueTogether(
                        name=model_name,
                        unique_together=unique_together,
                    ),
                    dependencies=related_dependencies
                )
            if index_together:
                self.add_operation(
                    app_label,
                    operations.AlterIndexTogether(
                        name=model_name,
                        index_together=index_together,
                    ),
                    dependencies=related_dependencies
                )
            # Fix relationships if the model changed from a proxy model to a
            # concrete model.
            relations = self.to_state.relations
            if (app_label, model_name) in self.old_proxy_keys:
                for related_model_key, related_fields in relations[app_label, model_name].items():
                    related_model_state = self.to_state.models[related_model_key]
                    for related_field_name, related_field in related_fields.items():
                        self.add_operation(
                            related_model_state.app_label,
                            operations.AlterField(
                                model_name=related_model_state.name,
                                name=related_field_name,
                                field=related_field,
                            ),
                            dependencies=[(app_label, model_name, None, True)],
                        )
```
### 90 - django/db/migrations/autodetector.py:

Start line: 1060, End line: 1076

```python
class MigrationAutodetector:

    def create_altered_constraints(self):
        option_name = operations.AddConstraint.option_name
        for app_label, model_name in sorted(self.kept_model_keys):
            old_model_name = self.renamed_models.get((app_label, model_name), model_name)
            old_model_state = self.from_state.models[app_label, old_model_name]
            new_model_state = self.to_state.models[app_label, model_name]

            old_constraints = old_model_state.options[option_name]
            new_constraints = new_model_state.options[option_name]
            add_constraints = [c for c in new_constraints if c not in old_constraints]
            rem_constraints = [c for c in old_constraints if c not in new_constraints]

            self.altered_constraints.update({
                (app_label, model_name): {
                    'added_constraints': add_constraints, 'removed_constraints': rem_constraints,
                }
            })
```
### 93 - django/db/migrations/autodetector.py:

Start line: 516, End line: 532

```python
class MigrationAutodetector:

    def generate_created_models(self):
        """
        Find all new models (both managed and unmanaged) and make create
        operations for them as well as separate operations to create any
        foreign key or M2M relationships (these are optimized later, if
        possible).

        Defer any model options that refer to collections of fields that might
        be deferred (e.g. unique_together, index_together).
        """
        old_keys = self.old_model_keys | self.old_unmanaged_keys
        added_models = self.new_model_keys - old_keys
        added_unmanaged_models = self.new_unmanaged_keys - old_keys
        all_added_models = chain(
            sorted(added_models, key=self.swappable_first_key, reverse=True),
            sorted(added_unmanaged_models, key=self.swappable_first_key, reverse=True)
        )
        # ... other code
```
### 104 - django/db/migrations/autodetector.py:

Start line: 1229, End line: 1254

```python
class MigrationAutodetector:

    def generate_altered_order_with_respect_to(self):
        for app_label, model_name in sorted(self.kept_model_keys):
            old_model_name = self.renamed_models.get((app_label, model_name), model_name)
            old_model_state = self.from_state.models[app_label, old_model_name]
            new_model_state = self.to_state.models[app_label, model_name]
            if (old_model_state.options.get("order_with_respect_to") !=
                    new_model_state.options.get("order_with_respect_to")):
                # Make sure it comes second if we're adding
                # (removal dependency is part of RemoveField)
                dependencies = []
                if new_model_state.options.get("order_with_respect_to"):
                    dependencies.append((
                        app_label,
                        model_name,
                        new_model_state.options["order_with_respect_to"],
                        True,
                    ))
                # Actually generate the operation
                self.add_operation(
                    app_label,
                    operations.AlterOrderWithRespectTo(
                        name=model_name,
                        order_with_respect_to=new_model_state.options.get('order_with_respect_to'),
                    ),
                    dependencies=dependencies,
                )
```
### 109 - django/db/migrations/autodetector.py:

Start line: 1038, End line: 1058

```python
class MigrationAutodetector:

    def generate_added_indexes(self):
        for (app_label, model_name), alt_indexes in self.altered_indexes.items():
            for index in alt_indexes['added_indexes']:
                self.add_operation(
                    app_label,
                    operations.AddIndex(
                        model_name=model_name,
                        index=index,
                    )
                )

    def generate_removed_indexes(self):
        for (app_label, model_name), alt_indexes in self.altered_indexes.items():
            for index in alt_indexes['removed_indexes']:
                self.add_operation(
                    app_label,
                    operations.RemoveIndex(
                        model_name=model_name,
                        name=index.name,
                    )
                )
```
### 125 - django/db/migrations/autodetector.py:

Start line: 1256, End line: 1268

```python
class MigrationAutodetector:

    def generate_altered_managers(self):
        for app_label, model_name in sorted(self.kept_model_keys):
            old_model_name = self.renamed_models.get((app_label, model_name), model_name)
            old_model_state = self.from_state.models[app_label, old_model_name]
            new_model_state = self.to_state.models[app_label, model_name]
            if old_model_state.managers != new_model_state.managers:
                self.add_operation(
                    app_label,
                    operations.AlterModelManagers(
                        name=model_name,
                        managers=new_model_state.managers,
                    )
                )
```
### 132 - django/db/migrations/autodetector.py:

Start line: 804, End line: 868

```python
class MigrationAutodetector:

    def generate_deleted_proxies(self):
        """Make DeleteModel options for proxy models."""
        deleted = self.old_proxy_keys - self.new_proxy_keys
        for app_label, model_name in sorted(deleted):
            model_state = self.from_state.models[app_label, model_name]
            assert model_state.options.get("proxy")
            self.add_operation(
                app_label,
                operations.DeleteModel(
                    name=model_state.name,
                ),
            )

    def generate_renamed_fields(self):
        """Work out renamed fields."""
        self.renamed_fields = {}
        for app_label, model_name, field_name in sorted(self.new_field_keys - self.old_field_keys):
            old_model_name = self.renamed_models.get((app_label, model_name), model_name)
            old_model_state = self.from_state.models[app_label, old_model_name]
            new_model_state = self.to_state.models[app_label, old_model_name]
            field = new_model_state.get_field(field_name)
            # Scan to see if this is actually a rename!
            field_dec = self.deep_deconstruct(field)
            for rem_app_label, rem_model_name, rem_field_name in sorted(self.old_field_keys - self.new_field_keys):
                if rem_app_label == app_label and rem_model_name == model_name:
                    old_field = old_model_state.get_field(rem_field_name)
                    old_field_dec = self.deep_deconstruct(old_field)
                    if field.remote_field and field.remote_field.model and 'to' in old_field_dec[2]:
                        old_rel_to = old_field_dec[2]['to']
                        if old_rel_to in self.renamed_models_rel:
                            old_field_dec[2]['to'] = self.renamed_models_rel[old_rel_to]
                    old_field.set_attributes_from_name(rem_field_name)
                    old_db_column = old_field.get_attname_column()[1]
                    if (old_field_dec == field_dec or (
                            # Was the field renamed and db_column equal to the
                            # old field's column added?
                            old_field_dec[0:2] == field_dec[0:2] and
                            dict(old_field_dec[2], db_column=old_db_column) == field_dec[2])):
                        if self.questioner.ask_rename(model_name, rem_field_name, field_name, field):
                            # A db_column mismatch requires a prior noop
                            # AlterField for the subsequent RenameField to be a
                            # noop on attempts at preserving the old name.
                            if old_field.db_column != field.db_column:
                                altered_field = field.clone()
                                altered_field.name = rem_field_name
                                self.add_operation(
                                    app_label,
                                    operations.AlterField(
                                        model_name=model_name,
                                        name=rem_field_name,
                                        field=altered_field,
                                    ),
                                )
                            self.add_operation(
                                app_label,
                                operations.RenameField(
                                    model_name=model_name,
                                    old_name=rem_field_name,
                                    new_name=field_name,
                                )
                            )
                            self.old_field_keys.remove((rem_app_label, rem_model_name, rem_field_name))
                            self.old_field_keys.add((app_label, model_name, field_name))
                            self.renamed_fields[app_label, model_name, field_name] = rem_field_name
                            break
```
### 137 - django/db/migrations/autodetector.py:

Start line: 1078, End line: 1098

```python
class MigrationAutodetector:

    def generate_added_constraints(self):
        for (app_label, model_name), alt_constraints in self.altered_constraints.items():
            for constraint in alt_constraints['added_constraints']:
                self.add_operation(
                    app_label,
                    operations.AddConstraint(
                        model_name=model_name,
                        constraint=constraint,
                    )
                )

    def generate_removed_constraints(self):
        for (app_label, model_name), alt_constraints in self.altered_constraints.items():
            for constraint in alt_constraints['removed_constraints']:
                self.add_operation(
                    app_label,
                    operations.RemoveConstraint(
                        model_name=model_name,
                        name=constraint.name,
                    )
                )
```
### 155 - django/db/migrations/autodetector.py:

Start line: 431, End line: 460

```python
class MigrationAutodetector:

    def add_operation(self, app_label, operation, dependencies=None, beginning=False):
        # Dependencies are (app_label, model_name, field_name, create/delete as True/False)
        operation._auto_deps = dependencies or []
        if beginning:
            self.generated_operations.setdefault(app_label, []).insert(0, operation)
        else:
            self.generated_operations.setdefault(app_label, []).append(operation)

    def swappable_first_key(self, item):
        """
        Place potential swappable models first in lists of created models (only
        real way to solve #22783).
        """
        try:
            model_state = self.to_state.models[item]
            base_names = {
                base if isinstance(base, str) else base.__name__
                for base in model_state.bases
            }
            string_version = "%s.%s" % (item[0], item[1])
            if (
                model_state.options.get('swappable') or
                "AbstractUser" in base_names or
                "AbstractBaseUser" in base_names or
                settings.AUTH_USER_MODEL.lower() == string_version.lower()
            ):
                return ("___" + item[0], "___" + item[1])
        except LookupError:
            pass
        return item
```
### 184 - django/db/migrations/autodetector.py:

Start line: 935, End line: 1018

```python
class MigrationAutodetector:

    def generate_altered_fields(self):
        """
        Make AlterField operations, or possibly RemovedField/AddField if alter
        isn't possible.
        """
        for app_label, model_name, field_name in sorted(self.old_field_keys & self.new_field_keys):
            # Did the field change?
            old_model_name = self.renamed_models.get((app_label, model_name), model_name)
            old_field_name = self.renamed_fields.get((app_label, model_name, field_name), field_name)
            old_field = self.from_state.models[app_label, old_model_name].get_field(old_field_name)
            new_field = self.to_state.models[app_label, model_name].get_field(field_name)
            dependencies = []
            # Implement any model renames on relations; these are handled by RenameModel
            # so we need to exclude them from the comparison
            if hasattr(new_field, "remote_field") and getattr(new_field.remote_field, "model", None):
                rename_key = resolve_relation(new_field.remote_field.model, app_label, model_name)
                if rename_key in self.renamed_models:
                    new_field.remote_field.model = old_field.remote_field.model
                # Handle ForeignKey which can only have a single to_field.
                remote_field_name = getattr(new_field.remote_field, 'field_name', None)
                if remote_field_name:
                    to_field_rename_key = rename_key + (remote_field_name,)
                    if to_field_rename_key in self.renamed_fields:
                        # Repoint both model and field name because to_field
                        # inclusion in ForeignKey.deconstruct() is based on
                        # both.
                        new_field.remote_field.model = old_field.remote_field.model
                        new_field.remote_field.field_name = old_field.remote_field.field_name
                # Handle ForeignObjects which can have multiple from_fields/to_fields.
                from_fields = getattr(new_field, 'from_fields', None)
                if from_fields:
                    from_rename_key = (app_label, model_name)
                    new_field.from_fields = tuple([
                        self.renamed_fields.get(from_rename_key + (from_field,), from_field)
                        for from_field in from_fields
                    ])
                    new_field.to_fields = tuple([
                        self.renamed_fields.get(rename_key + (to_field,), to_field)
                        for to_field in new_field.to_fields
                    ])
                dependencies.extend(self._get_dependencies_for_foreign_key(
                    app_label, model_name, new_field, self.to_state,
                ))
            if (
                hasattr(new_field, 'remote_field') and
                getattr(new_field.remote_field, 'through', None)
            ):
                rename_key = resolve_relation(new_field.remote_field.through, app_label, model_name)
                if rename_key in self.renamed_models:
                    new_field.remote_field.through = old_field.remote_field.through
            old_field_dec = self.deep_deconstruct(old_field)
            new_field_dec = self.deep_deconstruct(new_field)
            # If the field was confirmed to be renamed it means that only
            # db_column was allowed to change which generate_renamed_fields()
            # already accounts for by adding an AlterField operation.
            if old_field_dec != new_field_dec and old_field_name == field_name:
                both_m2m = old_field.many_to_many and new_field.many_to_many
                neither_m2m = not old_field.many_to_many and not new_field.many_to_many
                if both_m2m or neither_m2m:
                    # Either both fields are m2m or neither is
                    preserve_default = True
                    if (old_field.null and not new_field.null and not new_field.has_default() and
                            not new_field.many_to_many):
                        field = new_field.clone()
                        new_default = self.questioner.ask_not_null_alteration(field_name, model_name)
                        if new_default is not models.NOT_PROVIDED:
                            field.default = new_default
                            preserve_default = False
                    else:
                        field = new_field
                    self.add_operation(
                        app_label,
                        operations.AlterField(
                            model_name=model_name,
                            name=field_name,
                            field=field,
                            preserve_default=preserve_default,
                        ),
                        dependencies=dependencies,
                    )
                else:
                    # We cannot alter between m2m and concrete fields
                    self._generate_removed_field(app_label, model_name, field_name)
                    self._generate_added_field(app_label, model_name, field_name)
```
