# django__django-11630

| **django/django** | `65e86948b80262574058a94ccaae3a9b59c3faea` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 4361 |
| **Any found context length** | 4361 |
| **Avg pos** | 30.0 |
| **Min pos** | 15 |
| **Max pos** | 15 |
| **Top file pos** | 7 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/core/checks/model_checks.py b/django/core/checks/model_checks.py
--- a/django/core/checks/model_checks.py
+++ b/django/core/checks/model_checks.py
@@ -4,7 +4,8 @@
 from itertools import chain
 
 from django.apps import apps
-from django.core.checks import Error, Tags, register
+from django.conf import settings
+from django.core.checks import Error, Tags, Warning, register
 
 
 @register(Tags.models)
@@ -35,14 +36,25 @@ def check_all_models(app_configs=None, **kwargs):
             indexes[model_index.name].append(model._meta.label)
         for model_constraint in model._meta.constraints:
             constraints[model_constraint.name].append(model._meta.label)
+    if settings.DATABASE_ROUTERS:
+        error_class, error_id = Warning, 'models.W035'
+        error_hint = (
+            'You have configured settings.DATABASE_ROUTERS. Verify that %s '
+            'are correctly routed to separate databases.'
+        )
+    else:
+        error_class, error_id = Error, 'models.E028'
+        error_hint = None
     for db_table, model_labels in db_table_models.items():
         if len(model_labels) != 1:
+            model_labels_str = ', '.join(model_labels)
             errors.append(
-                Error(
+                error_class(
                     "db_table '%s' is used by multiple models: %s."
-                    % (db_table, ', '.join(db_table_models[db_table])),
+                    % (db_table, model_labels_str),
                     obj=db_table,
-                    id='models.E028',
+                    hint=(error_hint % model_labels_str) if error_hint else None,
+                    id=error_id,
                 )
             )
     for index_name, model_labels in indexes.items():

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/core/checks/model_checks.py | 7 | 7 | 15 | 7 | 4361
| django/core/checks/model_checks.py | 38 | 42 | 15 | 7 | 4361


## Problem Statement

```
Django throws error when different apps with different models have the same name table name.
Description
	
Error message:
table_name: (models.E028) db_table 'table_name' is used by multiple models: base.ModelName, app2.ModelName.
We have a Base app that points to a central database and that has its own tables. We then have multiple Apps that talk to their own databases. Some share the same table names.
We have used this setup for a while, but after upgrading to Django 2.2 we're getting an error saying we're not allowed 2 apps, with 2 different models to have the same table names. 
Is this correct behavior? We've had to roll back to Django 2.0 for now.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 django/db/models/base.py | 1451 | 1473| 171 | 171 | 15090 | 
| 2 | 2 django/apps/registry.py | 212 | 232| 237 | 408 | 18497 | 
| 3 | 2 django/db/models/base.py | 1426 | 1449| 176 | 584 | 18497 | 
| 4 | 3 django/db/migrations/state.py | 1 | 24| 191 | 775 | 23715 | 
| 5 | 3 django/db/models/base.py | 1262 | 1287| 184 | 959 | 23715 | 
| 6 | 3 django/db/models/base.py | 1 | 45| 289 | 1248 | 23715 | 
| 7 | 3 django/db/models/base.py | 1140 | 1168| 213 | 1461 | 23715 | 
| 8 | 4 django/db/backends/base/introspection.py | 57 | 84| 220 | 1681 | 25118 | 
| 9 | 4 django/db/models/base.py | 1231 | 1260| 242 | 1923 | 25118 | 
| 10 | 4 django/db/models/base.py | 1736 | 1807| 565 | 2488 | 25118 | 
| 11 | 5 django/db/migrations/operations/models.py | 345 | 394| 493 | 2981 | 31814 | 
| 12 | 5 django/db/migrations/state.py | 293 | 317| 266 | 3247 | 31814 | 
| 13 | 6 django/db/utils.py | 272 | 314| 322 | 3569 | 33926 | 
| 14 | 6 django/db/models/base.py | 1475 | 1507| 231 | 3800 | 33926 | 
| **-> 15 <-** | **7 django/core/checks/model_checks.py** | 1 | 74| 561 | 4361 | 35607 | 
| 16 | 7 django/db/migrations/operations/models.py | 460 | 485| 279 | 4640 | 35607 | 
| 17 | 8 django/db/backends/sqlite3/schema.py | 1 | 37| 318 | 4958 | 39572 | 
| 18 | 8 django/db/models/base.py | 1369 | 1424| 491 | 5449 | 39572 | 
| 19 | 8 django/apps/registry.py | 185 | 210| 190 | 5639 | 39572 | 
| 20 | 8 django/db/migrations/operations/models.py | 438 | 458| 130 | 5769 | 39572 | 
| 21 | 9 django/db/migrations/exceptions.py | 1 | 55| 250 | 6019 | 39823 | 
| 22 | 10 django/db/backends/base/schema.py | 406 | 427| 234 | 6253 | 51056 | 
| 23 | 10 django/db/models/base.py | 1320 | 1350| 244 | 6497 | 51056 | 
| 24 | 11 django/db/models/fields/related.py | 1389 | 1419| 322 | 6819 | 64568 | 
| 25 | 12 django/db/models/options.py | 1 | 36| 304 | 7123 | 71587 | 
| 26 | 12 django/db/models/base.py | 1592 | 1640| 348 | 7471 | 71587 | 
| 27 | 12 django/db/models/base.py | 1123 | 1138| 138 | 7609 | 71587 | 
| 28 | 12 django/db/migrations/operations/models.py | 396 | 412| 182 | 7791 | 71587 | 
| 29 | 13 django/contrib/auth/checks.py | 97 | 167| 525 | 8316 | 72760 | 
| 30 | 14 django/apps/config.py | 167 | 202| 251 | 8567 | 74492 | 
| 31 | 14 django/apps/registry.py | 262 | 274| 112 | 8679 | 74492 | 
| 32 | 15 django/db/backends/sqlite3/base.py | 296 | 380| 829 | 9508 | 80173 | 
| 33 | 15 django/db/models/base.py | 1809 | 1830| 155 | 9663 | 80173 | 
| 34 | 16 django/db/backends/mysql/base.py | 284 | 322| 404 | 10067 | 83215 | 
| 35 | 16 django/apps/registry.py | 127 | 145| 166 | 10233 | 83215 | 
| 36 | 16 django/apps/config.py | 1 | 52| 427 | 10660 | 83215 | 
| 37 | 16 django/db/migrations/state.py | 601 | 612| 136 | 10796 | 83215 | 
| 38 | 16 django/db/backends/base/schema.py | 1094 | 1124| 240 | 11036 | 83215 | 
| 39 | 17 django/contrib/contenttypes/checks.py | 24 | 42| 125 | 11161 | 83464 | 
| 40 | 17 django/db/backends/base/schema.py | 1052 | 1066| 126 | 11287 | 83464 | 
| 41 | 17 django/db/migrations/state.py | 245 | 291| 444 | 11731 | 83464 | 
| 42 | 17 django/db/models/base.py | 544 | 560| 142 | 11873 | 83464 | 
| 43 | 17 django/db/models/fields/related.py | 1202 | 1313| 939 | 12812 | 83464 | 
| 44 | 17 django/db/backends/base/schema.py | 1068 | 1092| 193 | 13005 | 83464 | 
| 45 | 17 django/db/backends/base/schema.py | 358 | 372| 182 | 13187 | 83464 | 
| 46 | 17 django/db/migrations/operations/models.py | 304 | 343| 406 | 13593 | 83464 | 
| 47 | **17 django/core/checks/model_checks.py** | 143 | 164| 263 | 13856 | 83464 | 
| 48 | 18 django/contrib/contenttypes/management/__init__.py | 1 | 42| 359 | 14215 | 84441 | 
| 49 | 18 django/db/backends/base/schema.py | 1036 | 1050| 123 | 14338 | 84441 | 
| 50 | **18 django/core/checks/model_checks.py** | 166 | 199| 332 | 14670 | 84441 | 
| 51 | 18 django/db/migrations/state.py | 154 | 164| 132 | 14802 | 84441 | 
| 52 | 19 django/db/migrations/recorder.py | 1 | 22| 153 | 14955 | 85111 | 
| 53 | 20 django/core/management/commands/showmigrations.py | 42 | 63| 158 | 15113 | 86297 | 
| 54 | 20 django/db/migrations/operations/models.py | 1 | 38| 238 | 15351 | 86297 | 
| 55 | 20 django/db/backends/sqlite3/schema.py | 86 | 99| 181 | 15532 | 86297 | 
| 56 | 20 django/db/migrations/operations/models.py | 102 | 118| 147 | 15679 | 86297 | 
| 57 | 21 django/db/backends/base/base.py | 528 | 559| 227 | 15906 | 91155 | 
| 58 | 21 django/db/models/base.py | 1049 | 1092| 404 | 16310 | 91155 | 
| 59 | 21 django/db/backends/base/introspection.py | 86 | 100| 121 | 16431 | 91155 | 
| 60 | 22 django/forms/models.py | 752 | 773| 194 | 16625 | 102650 | 
| 61 | 22 django/db/backends/sqlite3/schema.py | 140 | 221| 820 | 17445 | 102650 | 
| 62 | 23 django/db/migrations/operations/__init__.py | 1 | 18| 195 | 17640 | 102845 | 
| 63 | 23 django/db/models/base.py | 1536 | 1561| 183 | 17823 | 102845 | 
| 64 | **23 django/core/checks/model_checks.py** | 117 | 141| 268 | 18091 | 102845 | 
| 65 | 23 django/db/utils.py | 1 | 49| 154 | 18245 | 102845 | 
| 66 | 23 django/db/migrations/state.py | 557 | 578| 229 | 18474 | 102845 | 
| 67 | 24 django/core/management/templates.py | 210 | 241| 236 | 18710 | 105513 | 
| 68 | 24 django/apps/registry.py | 276 | 296| 229 | 18939 | 105513 | 
| 69 | 24 django/contrib/contenttypes/checks.py | 1 | 21| 122 | 19061 | 105513 | 
| 70 | 24 django/db/backends/sqlite3/schema.py | 223 | 305| 731 | 19792 | 105513 | 
| 71 | 25 django/db/models/sql/__init__.py | 1 | 8| 0 | 19792 | 105585 | 
| 72 | 25 django/db/models/fields/related.py | 171 | 188| 166 | 19958 | 105585 | 
| 73 | 25 django/db/models/options.py | 65 | 147| 668 | 20626 | 105585 | 
| 74 | 25 django/db/models/base.py | 1833 | 1884| 351 | 20977 | 105585 | 
| 75 | 26 django/core/management/commands/sqlsequencereset.py | 1 | 26| 194 | 21171 | 105779 | 
| 76 | 26 django/apps/registry.py | 298 | 329| 236 | 21407 | 105779 | 
| 77 | 27 django/core/management/commands/dumpdata.py | 1 | 65| 507 | 21914 | 107314 | 
| 78 | 27 django/db/migrations/state.py | 349 | 399| 471 | 22385 | 107314 | 
| 79 | 28 django/db/backends/sqlite3/operations.py | 163 | 188| 190 | 22575 | 110198 | 
| 80 | 28 django/db/models/base.py | 960 | 988| 230 | 22805 | 110198 | 
| 81 | 28 django/db/migrations/operations/models.py | 242 | 274| 232 | 23037 | 110198 | 
| 82 | 29 django/core/checks/urls.py | 30 | 50| 165 | 23202 | 110899 | 
| 83 | 29 django/apps/registry.py | 61 | 125| 438 | 23640 | 110899 | 
| 84 | 29 django/apps/registry.py | 234 | 260| 219 | 23859 | 110899 | 
| 85 | 29 django/db/models/options.py | 338 | 361| 198 | 24057 | 110899 | 
| 86 | 30 django/db/backends/mysql/schema.py | 1 | 51| 523 | 24580 | 112066 | 
| 87 | 31 django/core/checks/templates.py | 1 | 36| 259 | 24839 | 112326 | 
| 88 | 32 django/db/backends/postgresql/creation.py | 1 | 34| 238 | 25077 | 112974 | 
| 89 | 32 django/db/backends/base/base.py | 1 | 23| 138 | 25215 | 112974 | 
| 90 | 33 django/db/__init__.py | 1 | 18| 141 | 25356 | 113367 | 
| 91 | 33 django/db/models/base.py | 1289 | 1318| 205 | 25561 | 113367 | 
| 92 | 34 django/apps/__init__.py | 1 | 5| 0 | 25561 | 113390 | 
| 93 | 35 django/db/backends/oracle/schema.py | 41 | 55| 133 | 25694 | 115146 | 
| 94 | 36 django/core/exceptions.py | 1 | 96| 405 | 26099 | 116201 | 
| 95 | 36 django/db/migrations/state.py | 580 | 599| 188 | 26287 | 116201 | 
| 96 | 36 django/db/models/fields/related.py | 127 | 154| 202 | 26489 | 116201 | 
| 97 | 37 django/db/models/sql/datastructures.py | 142 | 171| 231 | 26720 | 117630 | 
| 98 | 38 django/core/management/commands/migrate.py | 1 | 18| 148 | 26868 | 120782 | 
| 99 | 38 django/db/migrations/operations/models.py | 517 | 526| 129 | 26997 | 120782 | 
| 100 | 39 django/core/management/commands/squashmigrations.py | 202 | 215| 112 | 27109 | 122653 | 
| 101 | 39 django/db/models/fields/related.py | 1315 | 1387| 616 | 27725 | 122653 | 
| 102 | 40 django/db/backends/base/creation.py | 278 | 297| 121 | 27846 | 124978 | 
| 103 | 40 django/db/models/base.py | 164 | 206| 406 | 28252 | 124978 | 
| 104 | 41 django/contrib/sites/models.py | 78 | 100| 115 | 28367 | 125767 | 
| 105 | 41 django/db/models/fields/related.py | 1 | 34| 244 | 28611 | 125767 | 
| 106 | 42 django/contrib/contenttypes/migrations/0002_remove_content_type_name.py | 1 | 40| 217 | 28828 | 125984 | 
| 107 | 42 django/apps/registry.py | 165 | 183| 133 | 28961 | 125984 | 
| 108 | 43 django/db/backends/mysql/creation.py | 1 | 29| 219 | 29180 | 126593 | 
| 109 | 44 django/db/models/sql/compiler.py | 1 | 20| 167 | 29347 | 140298 | 
| 110 | 44 django/db/migrations/operations/models.py | 414 | 435| 178 | 29525 | 140298 | 
| 111 | 45 django/core/management/commands/sqlmigrate.py | 1 | 30| 266 | 29791 | 140930 | 
| 112 | 45 django/db/backends/base/schema.py | 390 | 404| 174 | 29965 | 140930 | 
| 113 | 46 django/contrib/messages/apps.py | 1 | 8| 0 | 29965 | 140967 | 
| 114 | 46 django/db/models/fields/related.py | 108 | 125| 155 | 30120 | 140967 | 
| 115 | 47 django/core/management/commands/loaddata.py | 1 | 29| 151 | 30271 | 143835 | 
| 116 | 47 django/db/models/fields/related.py | 255 | 282| 269 | 30540 | 143835 | 
| 117 | 47 django/core/management/commands/migrate.py | 293 | 340| 401 | 30941 | 143835 | 
| 118 | 47 django/db/backends/oracle/schema.py | 125 | 173| 419 | 31360 | 143835 | 
| 119 | 47 django/db/models/fields/related.py | 156 | 169| 144 | 31504 | 143835 | 
| 120 | 47 django/db/models/base.py | 207 | 317| 866 | 32370 | 143835 | 
| 121 | 48 django/db/backends/oracle/creation.py | 130 | 165| 399 | 32769 | 147730 | 
| 122 | 48 django/db/migrations/operations/models.py | 41 | 100| 497 | 33266 | 147730 | 
| 123 | 49 django/db/models/utils.py | 1 | 22| 185 | 33451 | 147915 | 
| 124 | 49 django/db/models/fields/related.py | 509 | 563| 409 | 33860 | 147915 | 
| 125 | 49 django/db/models/sql/datastructures.py | 106 | 117| 133 | 33993 | 147915 | 
| 126 | 49 django/db/backends/base/schema.py | 40 | 115| 790 | 34783 | 147915 | 
| 127 | 49 django/db/models/base.py | 1563 | 1590| 238 | 35021 | 147915 | 
| 128 | 50 django/utils/autoreload.py | 267 | 283| 162 | 35183 | 152631 | 
| 129 | 50 django/db/models/fields/related.py | 190 | 254| 673 | 35856 | 152631 | 
| 130 | 50 django/contrib/auth/checks.py | 1 | 94| 646 | 36502 | 152631 | 
| 131 | 50 django/db/models/base.py | 1094 | 1121| 286 | 36788 | 152631 | 
| 132 | 51 django/core/management/sql.py | 37 | 52| 116 | 36904 | 153016 | 
| 133 | 51 django/db/models/options.py | 210 | 260| 446 | 37350 | 153016 | 
| 134 | 51 django/db/backends/base/base.py | 424 | 501| 525 | 37875 | 153016 | 
| 135 | 52 django/db/models/__init__.py | 1 | 49| 548 | 38423 | 153564 | 
| 136 | 53 django/contrib/admin/sites.py | 1 | 29| 175 | 38598 | 157755 | 
| 137 | 53 django/core/management/commands/dumpdata.py | 67 | 140| 626 | 39224 | 157755 | 
| 138 | 53 django/db/backends/mysql/base.py | 1 | 50| 447 | 39671 | 157755 | 
| 139 | 53 django/db/models/fields/related.py | 847 | 873| 240 | 39911 | 157755 | 
| 140 | 54 django/db/migrations/autodetector.py | 1116 | 1137| 231 | 40142 | 169426 | 
| 141 | 54 django/db/backends/base/schema.py | 1019 | 1034| 170 | 40312 | 169426 | 
| 142 | 55 django/db/models/fields/__init__.py | 1 | 85| 678 | 40990 | 186632 | 
| 143 | 56 django/contrib/contenttypes/apps.py | 1 | 23| 150 | 41140 | 186782 | 
| 144 | 57 django/db/backends/mysql/validation.py | 1 | 27| 248 | 41388 | 187270 | 
| 145 | 58 django/db/migrations/loader.py | 148 | 174| 291 | 41679 | 190187 | 
| 146 | 58 django/db/migrations/autodetector.py | 465 | 506| 418 | 42097 | 190187 | 
| 147 | 58 django/db/migrations/state.py | 319 | 346| 255 | 42352 | 190187 | 
| 148 | 59 django/core/management/commands/check.py | 1 | 34| 226 | 42578 | 190622 | 
| 149 | 59 django/db/models/base.py | 1509 | 1534| 183 | 42761 | 190622 | 
| 150 | 60 django/db/models/constraints.py | 1 | 27| 203 | 42964 | 191621 | 
| 151 | 60 django/db/migrations/state.py | 166 | 190| 213 | 43177 | 191621 | 
| 152 | 61 django/db/migrations/operations/utils.py | 17 | 54| 340 | 43517 | 192101 | 
| 153 | 61 django/db/models/base.py | 1170 | 1204| 230 | 43747 | 192101 | 
| 154 | 61 django/db/backends/oracle/schema.py | 1 | 39| 406 | 44153 | 192101 | 
| 155 | 61 django/db/models/base.py | 1352 | 1367| 153 | 44306 | 192101 | 
| 156 | 62 django/template/backends/django.py | 1 | 45| 302 | 44608 | 192956 | 
| 157 | 63 django/db/backends/oracle/operations.py | 443 | 475| 326 | 44934 | 198514 | 
| 158 | 63 django/core/management/sql.py | 20 | 34| 116 | 45050 | 198514 | 
| 159 | 63 django/db/migrations/autodetector.py | 1202 | 1214| 131 | 45181 | 198514 | 


### Hint

```
Regression in [5d25804eaf81795c7d457e5a2a9f0b9b0989136c], ticket #20098. My opinion is that as soon as the project has a non-empty DATABASE_ROUTERS setting, the error should be turned into a warning, as it becomes difficult to say for sure that it's an error. And then the project can add the warning in SILENCED_SYSTEM_CHECKS.
I agree with your opinion. Assigning to myself, patch on its way Replying to Claude Paroz: Regression in [5d25804eaf81795c7d457e5a2a9f0b9b0989136c], ticket #20098. My opinion is that as soon as the project has a non-empty DATABASE_ROUTERS setting, the error should be turned into a warning, as it becomes difficult to say for sure that it's an error. And then the project can add the warning in SILENCED_SYSTEM_CHECKS.
```

## Patch

```diff
diff --git a/django/core/checks/model_checks.py b/django/core/checks/model_checks.py
--- a/django/core/checks/model_checks.py
+++ b/django/core/checks/model_checks.py
@@ -4,7 +4,8 @@
 from itertools import chain
 
 from django.apps import apps
-from django.core.checks import Error, Tags, register
+from django.conf import settings
+from django.core.checks import Error, Tags, Warning, register
 
 
 @register(Tags.models)
@@ -35,14 +36,25 @@ def check_all_models(app_configs=None, **kwargs):
             indexes[model_index.name].append(model._meta.label)
         for model_constraint in model._meta.constraints:
             constraints[model_constraint.name].append(model._meta.label)
+    if settings.DATABASE_ROUTERS:
+        error_class, error_id = Warning, 'models.W035'
+        error_hint = (
+            'You have configured settings.DATABASE_ROUTERS. Verify that %s '
+            'are correctly routed to separate databases.'
+        )
+    else:
+        error_class, error_id = Error, 'models.E028'
+        error_hint = None
     for db_table, model_labels in db_table_models.items():
         if len(model_labels) != 1:
+            model_labels_str = ', '.join(model_labels)
             errors.append(
-                Error(
+                error_class(
                     "db_table '%s' is used by multiple models: %s."
-                    % (db_table, ', '.join(db_table_models[db_table])),
+                    % (db_table, model_labels_str),
                     obj=db_table,
-                    id='models.E028',
+                    hint=(error_hint % model_labels_str) if error_hint else None,
+                    id=error_id,
                 )
             )
     for index_name, model_labels in indexes.items():

```

## Test Patch

```diff
diff --git a/tests/check_framework/test_model_checks.py b/tests/check_framework/test_model_checks.py
--- a/tests/check_framework/test_model_checks.py
+++ b/tests/check_framework/test_model_checks.py
@@ -1,12 +1,16 @@
 from django.core import checks
-from django.core.checks import Error
+from django.core.checks import Error, Warning
 from django.db import models
 from django.test import SimpleTestCase, TestCase, skipUnlessDBFeature
 from django.test.utils import (
-    isolate_apps, modify_settings, override_system_checks,
+    isolate_apps, modify_settings, override_settings, override_system_checks,
 )
 
 
+class EmptyRouter:
+    pass
+
+
 @isolate_apps('check_framework', attr_name='apps')
 @override_system_checks([checks.model_checks.check_all_models])
 class DuplicateDBTableTests(SimpleTestCase):
@@ -28,6 +32,30 @@ class Meta:
             )
         ])
 
+    @override_settings(DATABASE_ROUTERS=['check_framework.test_model_checks.EmptyRouter'])
+    def test_collision_in_same_app_database_routers_installed(self):
+        class Model1(models.Model):
+            class Meta:
+                db_table = 'test_table'
+
+        class Model2(models.Model):
+            class Meta:
+                db_table = 'test_table'
+
+        self.assertEqual(checks.run_checks(app_configs=self.apps.get_app_configs()), [
+            Warning(
+                "db_table 'test_table' is used by multiple models: "
+                "check_framework.Model1, check_framework.Model2.",
+                hint=(
+                    'You have configured settings.DATABASE_ROUTERS. Verify '
+                    'that check_framework.Model1, check_framework.Model2 are '
+                    'correctly routed to separate databases.'
+                ),
+                obj='test_table',
+                id='models.W035',
+            )
+        ])
+
     @modify_settings(INSTALLED_APPS={'append': 'basic'})
     @isolate_apps('basic', 'check_framework', kwarg_name='apps')
     def test_collision_across_apps(self, apps):
@@ -50,6 +78,34 @@ class Meta:
             )
         ])
 
+    @modify_settings(INSTALLED_APPS={'append': 'basic'})
+    @override_settings(DATABASE_ROUTERS=['check_framework.test_model_checks.EmptyRouter'])
+    @isolate_apps('basic', 'check_framework', kwarg_name='apps')
+    def test_collision_across_apps_database_routers_installed(self, apps):
+        class Model1(models.Model):
+            class Meta:
+                app_label = 'basic'
+                db_table = 'test_table'
+
+        class Model2(models.Model):
+            class Meta:
+                app_label = 'check_framework'
+                db_table = 'test_table'
+
+        self.assertEqual(checks.run_checks(app_configs=apps.get_app_configs()), [
+            Warning(
+                "db_table 'test_table' is used by multiple models: "
+                "basic.Model1, check_framework.Model2.",
+                hint=(
+                    'You have configured settings.DATABASE_ROUTERS. Verify '
+                    'that basic.Model1, check_framework.Model2 are correctly '
+                    'routed to separate databases.'
+                ),
+                obj='test_table',
+                id='models.W035',
+            )
+        ])
+
     def test_no_collision_for_unmanaged_models(self):
         class Unmanaged(models.Model):
             class Meta:

```


## Code snippets

### 1 - django/db/models/base.py:

Start line: 1451, End line: 1473

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
### 2 - django/apps/registry.py:

Start line: 212, End line: 232

```python
class Apps:

    def register_model(self, app_label, model):
        # Since this method is called when models are imported, it cannot
        # perform imports because of the risk of import loops. It mustn't
        # call get_app_config().
        model_name = model._meta.model_name
        app_models = self.all_models[app_label]
        if model_name in app_models:
            if (model.__name__ == app_models[model_name].__name__ and
                    model.__module__ == app_models[model_name].__module__):
                warnings.warn(
                    "Model '%s.%s' was already registered. "
                    "Reloading models is not advised as it can lead to inconsistencies, "
                    "most notably with related models." % (app_label, model_name),
                    RuntimeWarning, stacklevel=2)
            else:
                raise RuntimeError(
                    "Conflicting '%s' models in application '%s': %s and %s." %
                    (model_name, app_label, app_models[model_name], model))
        app_models[model_name] = model
        self.do_pending_operations(model)
        self.clear_cache()
```
### 3 - django/db/models/base.py:

Start line: 1426, End line: 1449

```python
class Model(metaclass=ModelBase):

    @classmethod
    def _check_column_name_clashes(cls):
        # Store a list of column names which have already been used by other fields.
        used_column_names = []
        errors = []

        for f in cls._meta.local_fields:
            _, column_name = f.get_attname_column()

            # Ensure the column name is not already in use.
            if column_name and column_name in used_column_names:
                errors.append(
                    checks.Error(
                        "Field '%s' has column name '%s' that is used by "
                        "another field." % (f.name, column_name),
                        hint="Specify a 'db_column' for the field.",
                        obj=cls,
                        id='models.E007'
                    )
                )
            else:
                used_column_names.append(column_name)

        return errors
```
### 4 - django/db/migrations/state.py:

Start line: 1, End line: 24

```python
import copy
from contextlib import contextmanager

from django.apps import AppConfig
from django.apps.registry import Apps, apps as global_apps
from django.conf import settings
from django.db import models
from django.db.models.fields.proxy import OrderWrt
from django.db.models.fields.related import RECURSIVE_RELATIONSHIP_CONSTANT
from django.db.models.options import DEFAULT_NAMES, normalize_together
from django.db.models.utils import make_model_tuple
from django.utils.functional import cached_property
from django.utils.module_loading import import_string
from django.utils.version import get_docs_version

from .exceptions import InvalidBasesError


def _get_app_label_and_model_name(model, app_label=''):
    if isinstance(model, str):
        split = model.split('.', 1)
        return tuple(split) if len(split) == 2 else (app_label, split[0])
    else:
        return model._meta.app_label, model._meta.model_name
```
### 5 - django/db/models/base.py:

Start line: 1262, End line: 1287

```python
class Model(metaclass=ModelBase):

    @classmethod
    def _check_swappable(cls):
        """Check if the swapped model exists."""
        errors = []
        if cls._meta.swapped:
            try:
                apps.get_model(cls._meta.swapped)
            except ValueError:
                errors.append(
                    checks.Error(
                        "'%s' is not of the form 'app_label.app_name'." % cls._meta.swappable,
                        id='models.E001',
                    )
                )
            except LookupError:
                app_label, model_name = cls._meta.swapped.split('.')
                errors.append(
                    checks.Error(
                        "'%s' references '%s.%s', which has not been "
                        "installed, or is abstract." % (
                            cls._meta.swappable, app_label, model_name
                        ),
                        id='models.E002',
                    )
                )
        return errors
```
### 6 - django/db/models/base.py:

Start line: 1, End line: 45

```python
import copy
import inspect
import warnings
from functools import partialmethod
from itertools import chain

from django.apps import apps
from django.conf import settings
from django.core import checks
from django.core.exceptions import (
    NON_FIELD_ERRORS, FieldDoesNotExist, FieldError, MultipleObjectsReturned,
    ObjectDoesNotExist, ValidationError,
)
from django.db import (
    DEFAULT_DB_ALIAS, DJANGO_VERSION_PICKLE_KEY, DatabaseError, connection,
    connections, router, transaction,
)
from django.db.models.constants import LOOKUP_SEP
from django.db.models.constraints import CheckConstraint, UniqueConstraint
from django.db.models.deletion import CASCADE, Collector
from django.db.models.fields.related import (
    ForeignObjectRel, OneToOneField, lazy_related_operation, resolve_relation,
)
from django.db.models.manager import Manager
from django.db.models.options import Options
from django.db.models.query import Q
from django.db.models.signals import (
    class_prepared, post_init, post_save, pre_init, pre_save,
)
from django.db.models.utils import make_model_tuple
from django.utils.encoding import force_str
from django.utils.text import capfirst, get_text_list
from django.utils.translation import gettext_lazy as _
from django.utils.version import get_version


class Deferred:
    def __repr__(self):
        return '<Deferred field>'

    def __str__(self):
        return '<Deferred field>'


DEFERRED = Deferred()
```
### 7 - django/db/models/base.py:

Start line: 1140, End line: 1168

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
### 8 - django/db/backends/base/introspection.py:

Start line: 57, End line: 84

```python
class BaseDatabaseIntrospection:

    def django_table_names(self, only_existing=False, include_views=True):
        """
        Return a list of all table names that have associated Django models and
        are in INSTALLED_APPS.

        If only_existing is True, include only the tables in the database.
        """
        from django.apps import apps
        from django.db import router
        tables = set()
        for app_config in apps.get_app_configs():
            for model in router.get_migratable_models(app_config, self.connection.alias):
                if not model._meta.managed:
                    continue
                tables.add(model._meta.db_table)
                tables.update(
                    f.m2m_db_table() for f in model._meta.local_many_to_many
                    if f.remote_field.through._meta.managed
                )
        tables = list(tables)
        if only_existing:
            existing_tables = set(self.table_names(include_views=include_views))
            tables = [
                t
                for t in tables
                if self.identifier_converter(t) in existing_tables
            ]
        return tables
```
### 9 - django/db/models/base.py:

Start line: 1231, End line: 1260

```python
class Model(metaclass=ModelBase):

    @classmethod
    def check(cls, **kwargs):
        errors = [*cls._check_swappable(), *cls._check_model(), *cls._check_managers(**kwargs)]
        if not cls._meta.swapped:
            errors += [
                *cls._check_fields(**kwargs),
                *cls._check_m2m_through_same_relationship(),
                *cls._check_long_column_names(),
            ]
            clash_errors = (
                *cls._check_id_field(),
                *cls._check_field_name_clashes(),
                *cls._check_model_name_db_lookup_clashes(),
                *cls._check_property_name_related_field_accessor_clashes(),
                *cls._check_single_primary_key(),
            )
            errors.extend(clash_errors)
            # If there are field name clashes, hide consequent column name
            # clashes.
            if not clash_errors:
                errors.extend(cls._check_column_name_clashes())
            errors += [
                *cls._check_index_together(),
                *cls._check_unique_together(),
                *cls._check_indexes(),
                *cls._check_ordering(),
                *cls._check_constraints(),
            ]

        return errors
```
### 10 - django/db/models/base.py:

Start line: 1736, End line: 1807

```python
class Model(metaclass=ModelBase):

    @classmethod
    def _check_long_column_names(cls):
        """
        Check that any auto-generated column names are shorter than the limits
        for each database in which the model will be created.
        """
        errors = []
        allowed_len = None
        db_alias = None

        # Find the minimum max allowed length among all specified db_aliases.
        for db in settings.DATABASES:
            # skip databases where the model won't be created
            if not router.allow_migrate_model(db, cls):
                continue
            connection = connections[db]
            max_name_length = connection.ops.max_name_length()
            if max_name_length is None or connection.features.truncates_names:
                continue
            else:
                if allowed_len is None:
                    allowed_len = max_name_length
                    db_alias = db
                elif max_name_length < allowed_len:
                    allowed_len = max_name_length
                    db_alias = db

        if allowed_len is None:
            return errors

        for f in cls._meta.local_fields:
            _, column_name = f.get_attname_column()

            # Check if auto-generated name for the field is too long
            # for the database.
            if f.db_column is None and column_name is not None and len(column_name) > allowed_len:
                errors.append(
                    checks.Error(
                        'Autogenerated column name too long for field "%s". '
                        'Maximum length is "%s" for database "%s".'
                        % (column_name, allowed_len, db_alias),
                        hint="Set the column name manually using 'db_column'.",
                        obj=cls,
                        id='models.E018',
                    )
                )

        for f in cls._meta.local_many_to_many:
            # Skip nonexistent models.
            if isinstance(f.remote_field.through, str):
                continue

            # Check if auto-generated name for the M2M field is too long
            # for the database.
            for m2m in f.remote_field.through._meta.local_fields:
                _, rel_name = m2m.get_attname_column()
                if m2m.db_column is None and rel_name is not None and len(rel_name) > allowed_len:
                    errors.append(
                        checks.Error(
                            'Autogenerated column name too long for M2M field '
                            '"%s". Maximum length is "%s" for database "%s".'
                            % (rel_name, allowed_len, db_alias),
                            hint=(
                                "Use 'through' to create a separate model for "
                                "M2M and then set column_name using 'db_column'."
                            ),
                            obj=cls,
                            id='models.E019',
                        )
                    )

        return errors
```
### 15 - django/core/checks/model_checks.py:

Start line: 1, End line: 74

```python
import inspect
import types
from collections import defaultdict
from itertools import chain

from django.apps import apps
from django.core.checks import Error, Tags, register


@register(Tags.models)
def check_all_models(app_configs=None, **kwargs):
    db_table_models = defaultdict(list)
    indexes = defaultdict(list)
    constraints = defaultdict(list)
    errors = []
    if app_configs is None:
        models = apps.get_models()
    else:
        models = chain.from_iterable(app_config.get_models() for app_config in app_configs)
    for model in models:
        if model._meta.managed and not model._meta.proxy:
            db_table_models[model._meta.db_table].append(model._meta.label)
        if not inspect.ismethod(model.check):
            errors.append(
                Error(
                    "The '%s.check()' class method is currently overridden by %r."
                    % (model.__name__, model.check),
                    obj=model,
                    id='models.E020'
                )
            )
        else:
            errors.extend(model.check(**kwargs))
        for model_index in model._meta.indexes:
            indexes[model_index.name].append(model._meta.label)
        for model_constraint in model._meta.constraints:
            constraints[model_constraint.name].append(model._meta.label)
    for db_table, model_labels in db_table_models.items():
        if len(model_labels) != 1:
            errors.append(
                Error(
                    "db_table '%s' is used by multiple models: %s."
                    % (db_table, ', '.join(db_table_models[db_table])),
                    obj=db_table,
                    id='models.E028',
                )
            )
    for index_name, model_labels in indexes.items():
        if len(model_labels) > 1:
            model_labels = set(model_labels)
            errors.append(
                Error(
                    "index name '%s' is not unique %s %s." % (
                        index_name,
                        'for model' if len(model_labels) == 1 else 'amongst models:',
                        ', '.join(sorted(model_labels)),
                    ),
                    id='models.E029' if len(model_labels) == 1 else 'models.E030',
                ),
            )
    for constraint_name, model_labels in constraints.items():
        if len(model_labels) > 1:
            model_labels = set(model_labels)
            errors.append(
                Error(
                    "constraint name '%s' is not unique %s %s." % (
                        constraint_name,
                        'for model' if len(model_labels) == 1 else 'amongst models:',
                        ', '.join(sorted(model_labels)),
                    ),
                    id='models.E031' if len(model_labels) == 1 else 'models.E032',
                ),
            )
    return errors
```
### 47 - django/core/checks/model_checks.py:

Start line: 143, End line: 164

```python
def _check_lazy_references(apps, ignore=None):
    # ... other code

    def signal_connect_error(model_key, func, args, keywords):
        error_msg = (
            "%(receiver)s was connected to the '%(signal)s' signal with a "
            "lazy reference to the sender '%(model)s', but %(model_error)s."
        )
        receiver = args[0]
        # The receiver is either a function or an instance of class
        # defining a `__call__` method.
        if isinstance(receiver, types.FunctionType):
            description = "The function '%s'" % receiver.__name__
        elif isinstance(receiver, types.MethodType):
            description = "Bound method '%s.%s'" % (receiver.__self__.__class__.__name__, receiver.__name__)
        else:
            description = "An instance of class '%s'" % receiver.__class__.__name__
        signal_name = model_signals.get(func.__self__, 'unknown')
        params = {
            'model': '.'.join(model_key),
            'receiver': description,
            'signal': signal_name,
            'model_error': app_model_error(model_key),
        }
        return Error(error_msg % params, obj=receiver.__module__, id='signals.E001')
    # ... other code
```
### 50 - django/core/checks/model_checks.py:

Start line: 166, End line: 199

```python
def _check_lazy_references(apps, ignore=None):
    # ... other code

    def default_error(model_key, func, args, keywords):
        error_msg = "%(op)s contains a lazy reference to %(model)s, but %(model_error)s."
        params = {
            'op': func,
            'model': '.'.join(model_key),
            'model_error': app_model_error(model_key),
        }
        return Error(error_msg % params, obj=func, id='models.E022')

    # Maps common uses of lazy operations to corresponding error functions
    # defined above. If a key maps to None, no error will be produced.
    # default_error() will be used for usages that don't appear in this dict.
    known_lazy = {
        ('django.db.models.fields.related', 'resolve_related_class'): field_error,
        ('django.db.models.fields.related', 'set_managed'): None,
        ('django.dispatch.dispatcher', 'connect'): signal_connect_error,
    }

    def build_error(model_key, func, args, keywords):
        key = (func.__module__, func.__name__)
        error_fn = known_lazy.get(key, default_error)
        return error_fn(model_key, func, args, keywords) if error_fn else None

    return sorted(filter(None, (
        build_error(model_key, *extract_operation(func))
        for model_key in pending_models
        for func in apps._pending_operations[model_key]
    )), key=lambda error: error.msg)


@register(Tags.models)
def check_lazy_references(app_configs=None, **kwargs):
    return _check_lazy_references(apps)
```
### 64 - django/core/checks/model_checks.py:

Start line: 117, End line: 141

```python
def _check_lazy_references(apps, ignore=None):
    # ... other code

    def app_model_error(model_key):
        try:
            apps.get_app_config(model_key[0])
            model_error = "app '%s' doesn't provide model '%s'" % model_key
        except LookupError:
            model_error = "app '%s' isn't installed" % model_key[0]
        return model_error

    # Here are several functions which return CheckMessage instances for the
    # most common usages of lazy operations throughout Django. These functions
    # take the model that was being waited on as an (app_label, modelname)
    # pair, the original lazy function, and its positional and keyword args as
    # determined by extract_operation().

    def field_error(model_key, func, args, keywords):
        error_msg = (
            "The field %(field)s was declared with a lazy reference "
            "to '%(model)s', but %(model_error)s."
        )
        params = {
            'model': '.'.join(model_key),
            'field': keywords['field'],
            'model_error': app_model_error(model_key),
        }
        return Error(error_msg % params, obj=keywords['field'], id='fields.E307')
    # ... other code
```
