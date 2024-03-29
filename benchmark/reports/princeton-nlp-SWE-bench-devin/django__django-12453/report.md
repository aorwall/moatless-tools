# django__django-12453

| **django/django** | `b330b918e979ea39a21d47b61172d112caf432c3` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 3247 |
| **Any found context length** | 1626 |
| **Avg pos** | 11.0 |
| **Min pos** | 4 |
| **Max pos** | 7 |
| **Top file pos** | 4 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/db/backends/base/creation.py b/django/db/backends/base/creation.py
--- a/django/db/backends/base/creation.py
+++ b/django/db/backends/base/creation.py
@@ -6,6 +6,7 @@
 from django.conf import settings
 from django.core import serializers
 from django.db import router
+from django.db.transaction import atomic
 
 # The prefix to put on the default database name when creating
 # the test database.
@@ -126,8 +127,16 @@ def deserialize_db_from_string(self, data):
         the serialize_db_to_string() method.
         """
         data = StringIO(data)
-        for obj in serializers.deserialize("json", data, using=self.connection.alias):
-            obj.save()
+        # Load data in a transaction to handle forward references and cycles.
+        with atomic(using=self.connection.alias):
+            # Disable constraint checks, because some databases (MySQL) doesn't
+            # support deferred checks.
+            with self.connection.constraint_checks_disabled():
+                for obj in serializers.deserialize('json', data, using=self.connection.alias):
+                    obj.save()
+            # Manually check for any invalid keys that might have been added,
+            # because constraint checks were disabled.
+            self.connection.check_constraints()
 
     def _get_database_display_str(self, verbosity, database_name):
         """

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/db/backends/base/creation.py | 9 | 9 | 7 | 4 | 3247
| django/db/backends/base/creation.py | 129 | 130 | 4 | 4 | 1626


## Problem Statement

```
`TransactionTestCase.serialized_rollback` fails to restore objects due to ordering constraints
Description
	
I hit this problem in a fairly complex projet and haven't had the time to write a minimal reproduction case. I think it can be understood just by inspecting the code so I'm going to describe it while I have it in mind.
Setting serialized_rollback = True on a TransactionTestCase triggers ​rollback emulation. In practice, for each database:
BaseDatabaseCreation.create_test_db calls connection._test_serialized_contents = connection.creation.serialize_db_to_string()
TransactionTestCase._fixture_setup calls connection.creation.deserialize_db_from_string(connection._test_serialized_contents)
(The actual code isn't written that way; it's equivalent but the symmetry is less visible.)
serialize_db_to_string orders models with serializers.sort_dependencies and serializes them. The sorting algorithm only deals with natural keys. It doesn't do anything to order models referenced by foreign keys before models containing said foreign keys. That wouldn't be possible in general because circular foreign keys are allowed.
deserialize_db_from_string deserializes and saves models without wrapping in a transaction. This can result in integrity errors if an instance containing a foreign key is saved before the instance it references. I'm suggesting to fix it as follows:
diff --git a/django/db/backends/base/creation.py b/django/db/backends/base/creation.py
index bca8376..7bed2be 100644
--- a/django/db/backends/base/creation.py
+++ b/django/db/backends/base/creation.py
@@ -4,7 +4,7 @@ import time
 from django.apps import apps
 from django.conf import settings
 from django.core import serializers
-from django.db import router
+from django.db import router, transaction
 from django.utils.six import StringIO
 from django.utils.six.moves import input
 
@@ -128,8 +128,9 @@ class BaseDatabaseCreation(object):
		 the serialize_db_to_string method.
		 """
		 data = StringIO(data)
-		for obj in serializers.deserialize("json", data, using=self.connection.alias):
-			obj.save()
+		with transaction.atomic(using=self.connection.alias):
+			for obj in serializers.deserialize("json", data, using=self.connection.alias):
+				obj.save()
 
	 def _get_database_display_str(self, verbosity, database_name):
		 """
Note that loaddata doesn't have this problem because it wraps everything in a transaction:
	def handle(self, *fixture_labels, **options):
		# ...
		with transaction.atomic(using=self.using):
			self.loaddata(fixture_labels)
		# ...
This suggest that the transaction was just forgotten in the implementation of deserialize_db_from_string.
It should be possible to write a deterministic test for this bug because the order in which serialize_db_to_string serializes models depends on the app registry, and the app registry uses OrderedDict to store apps and models in a deterministic order.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 django/db/backends/oracle/creation.py | 130 | 165| 399 | 399 | 3894 | 
| 2 | 2 django/db/backends/base/base.py | 423 | 500| 525 | 924 | 8777 | 
| 3 | 3 django/db/transaction.py | 1 | 77| 438 | 1362 | 10921 | 
| **-> 4 <-** | **4 django/db/backends/base/creation.py** | 123 | 153| 264 | 1626 | 13236 | 
| 5 | 4 django/db/backends/oracle/creation.py | 30 | 100| 722 | 2348 | 13236 | 
| 6 | 4 django/db/backends/oracle/creation.py | 253 | 281| 277 | 2625 | 13236 | 
| **-> 7 <-** | **4 django/db/backends/base/creation.py** | 1 | 84| 622 | 3247 | 13236 | 
| 8 | 5 django/core/management/commands/loaddata.py | 81 | 148| 593 | 3840 | 16104 | 
| 9 | 6 django/db/backends/postgresql/operations.py | 165 | 207| 454 | 4294 | 18773 | 
| 10 | 7 django/db/backends/postgresql/creation.py | 53 | 78| 247 | 4541 | 19419 | 
| 11 | 8 django/db/backends/mysql/creation.py | 31 | 55| 253 | 4794 | 20026 | 
| 12 | 9 django/db/backends/sqlite3/creation.py | 23 | 49| 239 | 5033 | 20877 | 
| 13 | 9 django/db/backends/sqlite3/creation.py | 51 | 82| 317 | 5350 | 20877 | 
| 14 | 9 django/db/transaction.py | 196 | 282| 622 | 5972 | 20877 | 
| 15 | 10 django/db/migrations/serializer.py | 76 | 103| 233 | 6205 | 23426 | 
| 16 | 10 django/db/transaction.py | 168 | 194| 262 | 6467 | 23426 | 
| 17 | **10 django/db/backends/base/creation.py** | 86 | 121| 307 | 6774 | 23426 | 
| 18 | 11 django/db/backends/base/features.py | 306 | 324| 161 | 6935 | 26020 | 
| 19 | 11 django/db/backends/postgresql/creation.py | 36 | 51| 173 | 7108 | 26020 | 
| 20 | **11 django/db/backends/base/creation.py** | 155 | 194| 365 | 7473 | 26020 | 
| 21 | 12 django/core/serializers/__init__.py | 85 | 140| 369 | 7842 | 27668 | 
| 22 | 12 django/db/backends/oracle/creation.py | 187 | 218| 319 | 8161 | 27668 | 
| 23 | 13 django/db/backends/mysql/base.py | 251 | 287| 259 | 8420 | 30798 | 
| 24 | **13 django/db/backends/base/creation.py** | 263 | 294| 230 | 8650 | 30798 | 
| 25 | 13 django/db/migrations/serializer.py | 234 | 255| 166 | 8816 | 30798 | 
| 26 | 14 django/core/serializers/base.py | 219 | 230| 157 | 8973 | 33223 | 
| 27 | 15 django/db/models/base.py | 1866 | 1917| 351 | 9324 | 48595 | 
| 28 | 15 django/db/backends/base/base.py | 343 | 357| 118 | 9442 | 48595 | 
| 29 | 15 django/db/backends/base/base.py | 207 | 280| 497 | 9939 | 48595 | 
| 30 | **15 django/db/backends/base/creation.py** | 196 | 213| 154 | 10093 | 48595 | 
| 31 | 15 django/db/backends/sqlite3/creation.py | 84 | 104| 174 | 10267 | 48595 | 
| 32 | 15 django/db/models/base.py | 1 | 50| 330 | 10597 | 48595 | 
| 33 | 15 django/core/serializers/base.py | 232 | 249| 208 | 10805 | 48595 | 
| 34 | 15 django/db/backends/base/base.py | 1 | 23| 138 | 10943 | 48595 | 
| 35 | 15 django/db/transaction.py | 95 | 128| 225 | 11168 | 48595 | 
| 36 | 16 django/db/backends/sqlite3/base.py | 254 | 300| 419 | 11587 | 54360 | 
| 37 | 16 django/db/backends/sqlite3/base.py | 302 | 386| 823 | 12410 | 54360 | 
| 38 | 16 django/db/transaction.py | 80 | 92| 135 | 12545 | 54360 | 
| 39 | 17 django/db/backends/sqlite3/operations.py | 189 | 206| 204 | 12749 | 57244 | 
| 40 | 17 django/db/backends/postgresql/creation.py | 1 | 34| 238 | 12987 | 57244 | 
| 41 | 17 django/db/migrations/serializer.py | 280 | 311| 270 | 13257 | 57244 | 
| 42 | 17 django/db/backends/base/base.py | 560 | 607| 300 | 13557 | 57244 | 
| 43 | 17 django/core/serializers/__init__.py | 1 | 50| 274 | 13831 | 57244 | 
| 44 | 18 django/db/backends/base/operations.py | 1 | 100| 829 | 14660 | 62824 | 
| 45 | 18 django/db/backends/oracle/creation.py | 300 | 315| 193 | 14853 | 62824 | 
| 46 | 18 django/db/backends/mysql/creation.py | 1 | 29| 218 | 15071 | 62824 | 
| 47 | 18 django/db/backends/base/features.py | 1 | 115| 903 | 15974 | 62824 | 
| 48 | 18 django/db/models/base.py | 1666 | 1764| 717 | 16691 | 62824 | 
| 49 | 19 django/core/management/commands/dumpdata.py | 142 | 168| 239 | 16930 | 64359 | 
| 50 | 20 django/core/serializers/python.py | 78 | 156| 679 | 17609 | 65594 | 
| 51 | 21 django/db/models/sql/compiler.py | 1 | 18| 157 | 17766 | 79650 | 
| 52 | 21 django/db/transaction.py | 285 | 310| 174 | 17940 | 79650 | 
| 53 | 22 django/bin/django-admin.py | 1 | 22| 138 | 18078 | 79788 | 
| 54 | 23 django/db/migrations/autodetector.py | 1182 | 1207| 245 | 18323 | 91523 | 
| 55 | 24 django/db/backends/base/schema.py | 44 | 119| 790 | 19113 | 102838 | 
| 56 | 24 django/db/backends/oracle/creation.py | 220 | 251| 390 | 19503 | 102838 | 
| 57 | 25 django/db/backends/postgresql/base.py | 294 | 337| 343 | 19846 | 105641 | 
| 58 | 25 django/core/management/commands/loaddata.py | 63 | 79| 187 | 20033 | 105641 | 
| 59 | 25 django/db/backends/sqlite3/creation.py | 1 | 21| 140 | 20173 | 105641 | 
| 60 | **25 django/db/backends/base/creation.py** | 215 | 231| 173 | 20346 | 105641 | 
| 61 | 25 django/db/migrations/serializer.py | 1 | 73| 428 | 20774 | 105641 | 
| 62 | 26 django/db/models/deletion.py | 1 | 76| 566 | 21340 | 109456 | 
| 63 | 27 django/db/backends/sqlite3/schema.py | 1 | 37| 317 | 21657 | 113420 | 
| 64 | 27 django/db/backends/base/base.py | 302 | 320| 148 | 21805 | 113420 | 
| 65 | 27 django/db/migrations/autodetector.py | 1 | 15| 110 | 21915 | 113420 | 
| 66 | 28 django/db/backends/oracle/operations.py | 469 | 500| 360 | 22275 | 119348 | 
| 67 | 29 django/db/backends/postgresql/introspection.py | 209 | 225| 179 | 22454 | 121587 | 
| 68 | 29 django/db/backends/oracle/operations.py | 405 | 451| 480 | 22934 | 121587 | 
| 69 | 29 django/db/backends/sqlite3/base.py | 1 | 77| 535 | 23469 | 121587 | 
| 70 | 30 django/db/backends/mysql/operations.py | 143 | 174| 294 | 23763 | 124918 | 
| 71 | 30 django/db/backends/postgresql/operations.py | 143 | 163| 221 | 23984 | 124918 | 
| 72 | 30 django/db/transaction.py | 131 | 166| 295 | 24279 | 124918 | 
| 73 | 31 django/db/models/__init__.py | 1 | 52| 605 | 24884 | 125523 | 
| 74 | 31 django/core/serializers/base.py | 1 | 38| 207 | 25091 | 125523 | 
| 75 | 32 django/db/backends/dummy/base.py | 50 | 74| 173 | 25264 | 125968 | 
| 76 | 33 django/core/cache/backends/db.py | 112 | 197| 794 | 26058 | 128074 | 
| 77 | 34 django/contrib/sessions/backends/db.py | 1 | 72| 461 | 26519 | 128798 | 
| 78 | 34 django/db/migrations/autodetector.py | 337 | 356| 196 | 26715 | 128798 | 
| 79 | 34 django/db/backends/base/base.py | 322 | 341| 137 | 26852 | 128798 | 
| 80 | 34 django/db/migrations/serializer.py | 106 | 116| 114 | 26966 | 128798 | 
| 81 | 35 django/db/backends/oracle/base.py | 277 | 319| 303 | 27269 | 133916 | 
| 82 | 35 django/db/models/deletion.py | 379 | 446| 569 | 27838 | 133916 | 
| 83 | 36 django/core/management/commands/inspectdb.py | 38 | 173| 1292 | 29130 | 136526 | 
| 84 | 37 django/core/management/commands/testserver.py | 29 | 55| 234 | 29364 | 136960 | 
| 85 | 37 django/db/backends/oracle/creation.py | 1 | 28| 225 | 29589 | 136960 | 
| 86 | 37 django/db/backends/sqlite3/base.py | 202 | 244| 783 | 30372 | 136960 | 
| 87 | 37 django/db/backends/postgresql/introspection.py | 95 | 112| 257 | 30629 | 136960 | 
| 88 | 38 django/utils/deconstruct.py | 1 | 56| 385 | 31014 | 137345 | 
| 89 | 39 django/core/serializers/pyyaml.py | 1 | 39| 287 | 31301 | 137982 | 
| 90 | 39 django/db/backends/mysql/operations.py | 1 | 35| 282 | 31583 | 137982 | 
| 91 | 39 django/db/models/base.py | 1254 | 1284| 255 | 31838 | 137982 | 
| 92 | 40 django/db/backends/mysql/features.py | 1 | 101| 834 | 32672 | 139307 | 
| 93 | 40 django/db/models/base.py | 1839 | 1863| 174 | 32846 | 139307 | 
| 94 | 40 django/db/backends/mysql/creation.py | 57 | 67| 149 | 32995 | 139307 | 
| 95 | 40 django/db/backends/postgresql/base.py | 239 | 269| 267 | 33262 | 139307 | 
| 96 | 40 django/core/cache/backends/db.py | 199 | 228| 285 | 33547 | 139307 | 
| 97 | 40 django/core/serializers/base.py | 195 | 217| 183 | 33730 | 139307 | 
| 98 | 40 django/db/backends/base/schema.py | 1 | 28| 194 | 33924 | 139307 | 
| 99 | 41 django/db/models/options.py | 1 | 34| 282 | 34206 | 146329 | 
| 100 | 41 django/db/backends/oracle/base.py | 60 | 91| 276 | 34482 | 146329 | 
| 101 | 41 django/core/management/commands/dumpdata.py | 170 | 194| 224 | 34706 | 146329 | 
| 102 | 42 django/core/management/commands/sqlsequencereset.py | 1 | 26| 194 | 34900 | 146523 | 
| 103 | 43 django/core/exceptions.py | 99 | 194| 649 | 35549 | 147578 | 
| 104 | 44 django/db/backends/sqlite3/introspection.py | 312 | 340| 278 | 35827 | 151287 | 
| 105 | 44 django/core/management/commands/dumpdata.py | 67 | 140| 626 | 36453 | 151287 | 
| 106 | 44 django/db/migrations/serializer.py | 196 | 217| 183 | 36636 | 151287 | 
| 107 | 44 django/db/backends/postgresql/base.py | 1 | 62| 480 | 37116 | 151287 | 
| 108 | 45 django/contrib/postgres/serializers.py | 1 | 11| 0 | 37116 | 151388 | 
| 109 | 45 django/core/serializers/base.py | 64 | 119| 496 | 37612 | 151388 | 
| 110 | 46 django/db/migrations/recorder.py | 1 | 21| 148 | 37760 | 152065 | 
| 111 | 46 django/db/migrations/autodetector.py | 1123 | 1144| 231 | 37991 | 152065 | 
| 112 | 47 django/db/migrations/operations/models.py | 609 | 622| 137 | 38128 | 158761 | 
| 113 | 47 django/db/models/base.py | 1766 | 1837| 565 | 38693 | 158761 | 
| 114 | 47 django/db/backends/base/base.py | 527 | 558| 227 | 38920 | 158761 | 
| 115 | 47 django/db/backends/oracle/base.py | 147 | 209| 780 | 39700 | 158761 | 
| 116 | 47 django/db/backends/sqlite3/schema.py | 86 | 99| 181 | 39881 | 158761 | 
| 117 | 47 django/db/backends/sqlite3/schema.py | 223 | 305| 731 | 40612 | 158761 | 
| 118 | 47 django/db/backends/mysql/base.py | 168 | 192| 199 | 40811 | 158761 | 
| 119 | 47 django/core/cache/backends/db.py | 255 | 280| 308 | 41119 | 158761 | 
| 120 | 47 django/db/models/base.py | 751 | 800| 456 | 41575 | 158761 | 
| 121 | 47 django/db/backends/base/base.py | 359 | 390| 204 | 41779 | 158761 | 
| 122 | 48 django/core/management/commands/migrate.py | 161 | 243| 794 | 42573 | 161922 | 
| 123 | 49 django/db/backends/mysql/introspection.py | 254 | 270| 184 | 42757 | 164150 | 
| 124 | 49 django/db/migrations/autodetector.py | 525 | 671| 1109 | 43866 | 164150 | 
| 125 | 49 django/db/backends/mysql/operations.py | 176 | 191| 163 | 44029 | 164150 | 
| 126 | 49 django/db/migrations/autodetector.py | 1027 | 1043| 188 | 44217 | 164150 | 
| 127 | 50 django/db/backends/postgresql/features.py | 1 | 81| 674 | 44891 | 164824 | 
| 128 | 51 django/db/backends/mysql/validation.py | 1 | 27| 248 | 45139 | 165312 | 
| 129 | 51 django/core/cache/backends/db.py | 40 | 95| 431 | 45570 | 165312 | 
| 130 | 52 django/db/backends/sqlite3/features.py | 1 | 48| 533 | 46103 | 165845 | 
| 131 | 52 django/db/backends/mysql/operations.py | 193 | 236| 329 | 46432 | 165845 | 
| 132 | 52 django/db/backends/mysql/base.py | 1 | 49| 458 | 46890 | 165845 | 
| 133 | 52 django/db/migrations/serializer.py | 141 | 160| 223 | 47113 | 165845 | 
| 134 | 52 django/core/serializers/base.py | 121 | 169| 309 | 47422 | 165845 | 
| 135 | 53 django/db/utils.py | 279 | 321| 322 | 47744 | 167991 | 
| 136 | 54 django/db/migrations/writer.py | 2 | 115| 886 | 48630 | 170238 | 
| 137 | 54 django/db/backends/sqlite3/base.py | 154 | 170| 189 | 48819 | 170238 | 
| 138 | 55 django/db/migrations/operations/special.py | 181 | 204| 246 | 49065 | 171796 | 
| 139 | 55 django/db/backends/mysql/features.py | 103 | 156| 497 | 49562 | 171796 | 
| 140 | 55 django/db/backends/oracle/creation.py | 317 | 401| 739 | 50301 | 171796 | 
| 141 | 55 django/db/backends/postgresql/base.py | 205 | 237| 260 | 50561 | 171796 | 
| 142 | 56 django/db/backends/utils.py | 1 | 45| 273 | 50834 | 173662 | 
| 143 | 56 django/db/backends/oracle/operations.py | 366 | 403| 369 | 51203 | 173662 | 
| 144 | 56 django/core/cache/backends/db.py | 97 | 110| 234 | 51437 | 173662 | 
| 145 | 56 django/db/backends/oracle/operations.py | 453 | 467| 203 | 51640 | 173662 | 
| 146 | 57 django/db/migrations/state.py | 1 | 23| 180 | 51820 | 178870 | 
| 147 | 58 django/contrib/admin/views/main.py | 332 | 384| 467 | 52287 | 183127 | 
| 148 | 58 django/db/migrations/serializer.py | 119 | 138| 136 | 52423 | 183127 | 
| 149 | 59 django/db/backends/postgresql/schema.py | 1 | 60| 585 | 53008 | 185110 | 
| 150 | 59 django/db/backends/sqlite3/base.py | 80 | 153| 754 | 53762 | 185110 | 
| 151 | 59 django/db/migrations/state.py | 600 | 611| 136 | 53898 | 185110 | 
| 152 | 59 django/db/models/base.py | 1072 | 1115| 404 | 54302 | 185110 | 
| 153 | 59 django/db/migrations/recorder.py | 46 | 97| 390 | 54692 | 185110 | 
| 154 | 59 django/db/migrations/autodetector.py | 707 | 794| 789 | 55481 | 185110 | 
| 155 | 59 django/db/backends/oracle/operations.py | 1 | 18| 137 | 55618 | 185110 | 
| 156 | 59 django/db/migrations/operations/models.py | 1 | 38| 238 | 55856 | 185110 | 
| 157 | 59 django/core/management/commands/loaddata.py | 32 | 61| 261 | 56117 | 185110 | 
| 158 | 59 django/db/backends/postgresql/base.py | 132 | 150| 177 | 56294 | 185110 | 
| 159 | 59 django/db/backends/mysql/base.py | 231 | 249| 165 | 56459 | 185110 | 
| 160 | 59 django/db/backends/postgresql/operations.py | 106 | 141| 267 | 56726 | 185110 | 
| 161 | 59 django/db/backends/base/operations.py | 672 | 692| 187 | 56913 | 185110 | 
| 162 | 60 django/db/migrations/executor.py | 298 | 391| 843 | 57756 | 188533 | 
| 163 | 61 django/db/migrations/loader.py | 275 | 299| 205 | 57961 | 191423 | 
| 164 | 61 django/db/backends/sqlite3/operations.py | 292 | 335| 431 | 58392 | 191423 | 
| 165 | 61 django/db/utils.py | 207 | 237| 194 | 58586 | 191423 | 
| 166 | 61 django/db/backends/sqlite3/operations.py | 162 | 187| 190 | 58776 | 191423 | 
| 167 | 61 django/core/serializers/__init__.py | 159 | 235| 676 | 59452 | 191423 | 
| 168 | **61 django/db/backends/base/creation.py** | 233 | 261| 232 | 59684 | 191423 | 
| 169 | 61 django/db/backends/oracle/operations.py | 21 | 73| 574 | 60258 | 191423 | 
| 170 | 62 django/db/migrations/operations/__init__.py | 1 | 18| 195 | 60453 | 191618 | 
| 171 | 62 django/db/backends/sqlite3/operations.py | 1 | 39| 267 | 60720 | 191618 | 
| 172 | 62 django/db/backends/oracle/creation.py | 102 | 128| 314 | 61034 | 191618 | 
| 173 | 62 django/db/backends/sqlite3/introspection.py | 206 | 220| 146 | 61180 | 191618 | 
| 174 | 62 django/db/backends/base/operations.py | 401 | 463| 510 | 61690 | 191618 | 
| 175 | 62 django/db/backends/postgresql/operations.py | 41 | 85| 483 | 62173 | 191618 | 


### Hint

```
I've run into a problem related to this one (just reported as #31051), so I ended up looking into this problem as well. The original report still seems accurate to me, with the proposed solution valid. I've been working on a fix and (most of the work), testcase for this problem. I'll do some more testing and provide a proper PR for this issue and #31051 soon. The testcase is not ideal yet (testing the testing framework is tricky), but I'll expand on that in the PR. Furthermore, I noticed that loaddata does not just wrap everything in a transaction, it also explicitly disables constraint checks inside the transaction: with connection.constraint_checks_disabled(): self.objs_with_deferred_fields = [] for fixture_label in fixture_labels: self.load_label(fixture_label) for obj in self.objs_with_deferred_fields: obj.save_deferred_fields(using=self.using) # Since we disabled constraint checks, we must manually check for # any invalid keys that might have been added table_names = [model._meta.db_table for model in self.models] try: connection.check_constraints(table_names=table_names) except Exception as e: e.args = ("Problem installing fixtures: %s" % e,) raise I had a closer look at how this works (since I understood that a transaction already implicitly disables constraint checks) and it turns out that MySQL/InnoDB is an exception and does *not* defer constraint checks to the end of the transaction, but instead needs extra handling (so constraint_checks_disabled() is a no-op on most database backends). See #3615.
```

## Patch

```diff
diff --git a/django/db/backends/base/creation.py b/django/db/backends/base/creation.py
--- a/django/db/backends/base/creation.py
+++ b/django/db/backends/base/creation.py
@@ -6,6 +6,7 @@
 from django.conf import settings
 from django.core import serializers
 from django.db import router
+from django.db.transaction import atomic
 
 # The prefix to put on the default database name when creating
 # the test database.
@@ -126,8 +127,16 @@ def deserialize_db_from_string(self, data):
         the serialize_db_to_string() method.
         """
         data = StringIO(data)
-        for obj in serializers.deserialize("json", data, using=self.connection.alias):
-            obj.save()
+        # Load data in a transaction to handle forward references and cycles.
+        with atomic(using=self.connection.alias):
+            # Disable constraint checks, because some databases (MySQL) doesn't
+            # support deferred checks.
+            with self.connection.constraint_checks_disabled():
+                for obj in serializers.deserialize('json', data, using=self.connection.alias):
+                    obj.save()
+            # Manually check for any invalid keys that might have been added,
+            # because constraint checks were disabled.
+            self.connection.check_constraints()
 
     def _get_database_display_str(self, verbosity, database_name):
         """

```

## Test Patch

```diff
diff --git a/tests/backends/base/test_creation.py b/tests/backends/base/test_creation.py
--- a/tests/backends/base/test_creation.py
+++ b/tests/backends/base/test_creation.py
@@ -7,6 +7,8 @@
 )
 from django.test import SimpleTestCase
 
+from ..models import Object, ObjectReference
+
 
 def get_connection_copy():
     # Get a copy of the default connection. (Can't use django.db.connection
@@ -73,3 +75,29 @@ def test_migrate_test_setting_true(self, mocked_migrate, mocked_ensure_connectio
         finally:
             with mock.patch.object(creation, '_destroy_test_db'):
                 creation.destroy_test_db(old_database_name, verbosity=0)
+
+
+class TestDeserializeDbFromString(SimpleTestCase):
+    databases = {'default'}
+
+    def test_circular_reference(self):
+        # deserialize_db_from_string() handles circular references.
+        data = """
+        [
+            {
+                "model": "backends.object",
+                "pk": 1,
+                "fields": {"obj_ref": 1, "related_objects": []}
+            },
+            {
+                "model": "backends.objectreference",
+                "pk": 1,
+                "fields": {"obj": 1}
+            }
+        ]
+        """
+        connection.creation.deserialize_db_from_string(data)
+        obj = Object.objects.get()
+        obj_ref = ObjectReference.objects.get()
+        self.assertEqual(obj.obj_ref, obj_ref)
+        self.assertEqual(obj_ref.obj, obj)
diff --git a/tests/backends/models.py b/tests/backends/models.py
--- a/tests/backends/models.py
+++ b/tests/backends/models.py
@@ -89,6 +89,7 @@ def __str__(self):
 
 class Object(models.Model):
     related_objects = models.ManyToManyField("self", db_constraint=False, symmetrical=False)
+    obj_ref = models.ForeignKey('ObjectReference', models.CASCADE, null=True)
 
     def __str__(self):
         return str(self.id)

```


## Code snippets

### 1 - django/db/backends/oracle/creation.py:

Start line: 130, End line: 165

```python
class DatabaseCreation(BaseDatabaseCreation):

    def _handle_objects_preventing_db_destruction(self, cursor, parameters, verbosity, autoclobber):
        # There are objects in the test tablespace which prevent dropping it
        # The easy fix is to drop the test user -- but are we allowed to do so?
        self.log(
            'There are objects in the old test database which prevent its destruction.\n'
            'If they belong to the test user, deleting the user will allow the test '
            'database to be recreated.\n'
            'Otherwise, you will need to find and remove each of these objects, '
            'or use a different tablespace.\n'
        )
        if self._test_user_create():
            if not autoclobber:
                confirm = input("Type 'yes' to delete user %s: " % parameters['user'])
            if autoclobber or confirm == 'yes':
                try:
                    if verbosity >= 1:
                        self.log('Destroying old test user...')
                    self._destroy_test_user(cursor, parameters, verbosity)
                except Exception as e:
                    self.log('Got an error destroying the test user: %s' % e)
                    sys.exit(2)
                try:
                    if verbosity >= 1:
                        self.log("Destroying old test database for alias '%s'..." % self.connection.alias)
                    self._execute_test_db_destruction(cursor, parameters, verbosity)
                except Exception as e:
                    self.log('Got an error destroying the test database: %s' % e)
                    sys.exit(2)
            else:
                self.log('Tests cancelled -- test database cannot be recreated.')
                sys.exit(1)
        else:
            self.log("Django is configured to use pre-existing test user '%s',"
                     " and will not attempt to delete it." % parameters['user'])
            self.log('Tests cancelled -- test database cannot be recreated.')
            sys.exit(1)
```
### 2 - django/db/backends/base/base.py:

Start line: 423, End line: 500

```python
class BaseDatabaseWrapper:

    def get_rollback(self):
        """Get the "needs rollback" flag -- for *advanced use* only."""
        if not self.in_atomic_block:
            raise TransactionManagementError(
                "The rollback flag doesn't work outside of an 'atomic' block.")
        return self.needs_rollback

    def set_rollback(self, rollback):
        """
        Set or unset the "needs rollback" flag -- for *advanced use* only.
        """
        if not self.in_atomic_block:
            raise TransactionManagementError(
                "The rollback flag doesn't work outside of an 'atomic' block.")
        self.needs_rollback = rollback

    def validate_no_atomic_block(self):
        """Raise an error if an atomic block is active."""
        if self.in_atomic_block:
            raise TransactionManagementError(
                "This is forbidden when an 'atomic' block is active.")

    def validate_no_broken_transaction(self):
        if self.needs_rollback:
            raise TransactionManagementError(
                "An error occurred in the current transaction. You can't "
                "execute queries until the end of the 'atomic' block.")

    # ##### Foreign key constraints checks handling #####

    @contextmanager
    def constraint_checks_disabled(self):
        """
        Disable foreign key constraint checking.
        """
        disabled = self.disable_constraint_checking()
        try:
            yield
        finally:
            if disabled:
                self.enable_constraint_checking()

    def disable_constraint_checking(self):
        """
        Backends can implement as needed to temporarily disable foreign key
        constraint checking. Should return True if the constraints were
        disabled and will need to be reenabled.
        """
        return False

    def enable_constraint_checking(self):
        """
        Backends can implement as needed to re-enable foreign key constraint
        checking.
        """
        pass

    def check_constraints(self, table_names=None):
        """
        Backends can override this method if they can apply constraint
        checking (e.g. via "SET CONSTRAINTS ALL IMMEDIATE"). Should raise an
        IntegrityError if any invalid foreign key references are encountered.
        """
        pass

    # ##### Connection termination handling #####

    def is_usable(self):
        """
        Test if the database connection is usable.

        This method may assume that self.connection is not None.

        Actual implementations should take care not to raise exceptions
        as that may prevent Django from recycling unusable connections.
        """
        raise NotImplementedError(
            "subclasses of BaseDatabaseWrapper may require an is_usable() method")
```
### 3 - django/db/transaction.py:

Start line: 1, End line: 77

```python
from contextlib import ContextDecorator, contextmanager

from django.db import (
    DEFAULT_DB_ALIAS, DatabaseError, Error, ProgrammingError, connections,
)


class TransactionManagementError(ProgrammingError):
    """Transaction management is used improperly."""
    pass


def get_connection(using=None):
    """
    Get a database connection by name, or the default database connection
    if no name is provided. This is a private API.
    """
    if using is None:
        using = DEFAULT_DB_ALIAS
    return connections[using]


def get_autocommit(using=None):
    """Get the autocommit status of the connection."""
    return get_connection(using).get_autocommit()


def set_autocommit(autocommit, using=None):
    """Set the autocommit status of the connection."""
    return get_connection(using).set_autocommit(autocommit)


def commit(using=None):
    """Commit a transaction."""
    get_connection(using).commit()


def rollback(using=None):
    """Roll back a transaction."""
    get_connection(using).rollback()


def savepoint(using=None):
    """
    Create a savepoint (if supported and required by the backend) inside the
    current transaction. Return an identifier for the savepoint that will be
    used for the subsequent rollback or commit.
    """
    return get_connection(using).savepoint()


def savepoint_rollback(sid, using=None):
    """
    Roll back the most recent savepoint (if one exists). Do nothing if
    savepoints are not supported.
    """
    get_connection(using).savepoint_rollback(sid)


def savepoint_commit(sid, using=None):
    """
    Commit the most recent savepoint (if one exists). Do nothing if
    savepoints are not supported.
    """
    get_connection(using).savepoint_commit(sid)


def clean_savepoints(using=None):
    """
    Reset the counter used to generate unique savepoint ids in this thread.
    """
    get_connection(using).clean_savepoints()


def get_rollback(using=None):
    """Get the "needs rollback" flag -- for *advanced use* only."""
    return get_connection(using).get_rollback()
```
### 4 - django/db/backends/base/creation.py:

Start line: 123, End line: 153

```python
class BaseDatabaseCreation:

    def deserialize_db_from_string(self, data):
        """
        Reload the database with data from a string generated by
        the serialize_db_to_string() method.
        """
        data = StringIO(data)
        for obj in serializers.deserialize("json", data, using=self.connection.alias):
            obj.save()

    def _get_database_display_str(self, verbosity, database_name):
        """
        Return display string for a database for use in various actions.
        """
        return "'%s'%s" % (
            self.connection.alias,
            (" ('%s')" % database_name) if verbosity >= 2 else '',
        )

    def _get_test_db_name(self):
        """
        Internal implementation - return the name of the test DB that will be
        created. Only useful when called from create_test_db() and
        _create_test_db() and when no external munging is done with the 'NAME'
        settings.
        """
        if self.connection.settings_dict['TEST']['NAME']:
            return self.connection.settings_dict['TEST']['NAME']
        return TEST_DATABASE_PREFIX + self.connection.settings_dict['NAME']

    def _execute_create_test_db(self, cursor, parameters, keepdb=False):
        cursor.execute('CREATE DATABASE %(dbname)s %(suffix)s' % parameters)
```
### 5 - django/db/backends/oracle/creation.py:

Start line: 30, End line: 100

```python
class DatabaseCreation(BaseDatabaseCreation):

    def _create_test_db(self, verbosity=1, autoclobber=False, keepdb=False):
        parameters = self._get_test_db_params()
        with self._maindb_connection.cursor() as cursor:
            if self._test_database_create():
                try:
                    self._execute_test_db_creation(cursor, parameters, verbosity, keepdb)
                except Exception as e:
                    if 'ORA-01543' not in str(e):
                        # All errors except "tablespace already exists" cancel tests
                        self.log('Got an error creating the test database: %s' % e)
                        sys.exit(2)
                    if not autoclobber:
                        confirm = input(
                            "It appears the test database, %s, already exists. "
                            "Type 'yes' to delete it, or 'no' to cancel: " % parameters['user'])
                    if autoclobber or confirm == 'yes':
                        if verbosity >= 1:
                            self.log("Destroying old test database for alias '%s'..." % self.connection.alias)
                        try:
                            self._execute_test_db_destruction(cursor, parameters, verbosity)
                        except DatabaseError as e:
                            if 'ORA-29857' in str(e):
                                self._handle_objects_preventing_db_destruction(cursor, parameters,
                                                                               verbosity, autoclobber)
                            else:
                                # Ran into a database error that isn't about leftover objects in the tablespace
                                self.log('Got an error destroying the old test database: %s' % e)
                                sys.exit(2)
                        except Exception as e:
                            self.log('Got an error destroying the old test database: %s' % e)
                            sys.exit(2)
                        try:
                            self._execute_test_db_creation(cursor, parameters, verbosity, keepdb)
                        except Exception as e:
                            self.log('Got an error recreating the test database: %s' % e)
                            sys.exit(2)
                    else:
                        self.log('Tests cancelled.')
                        sys.exit(1)

            if self._test_user_create():
                if verbosity >= 1:
                    self.log('Creating test user...')
                try:
                    self._create_test_user(cursor, parameters, verbosity, keepdb)
                except Exception as e:
                    if 'ORA-01920' not in str(e):
                        # All errors except "user already exists" cancel tests
                        self.log('Got an error creating the test user: %s' % e)
                        sys.exit(2)
                    if not autoclobber:
                        confirm = input(
                            "It appears the test user, %s, already exists. Type "
                            "'yes' to delete it, or 'no' to cancel: " % parameters['user'])
                    if autoclobber or confirm == 'yes':
                        try:
                            if verbosity >= 1:
                                self.log('Destroying old test user...')
                            self._destroy_test_user(cursor, parameters, verbosity)
                            if verbosity >= 1:
                                self.log('Creating test user...')
                            self._create_test_user(cursor, parameters, verbosity, keepdb)
                        except Exception as e:
                            self.log('Got an error recreating the test user: %s' % e)
                            sys.exit(2)
                    else:
                        self.log('Tests cancelled.')
                        sys.exit(1)
        self._maindb_connection.close()  # done with main user -- test user and tablespaces created
        self._switch_to_test_user(parameters)
        return self.connection.settings_dict['NAME']
```
### 6 - django/db/backends/oracle/creation.py:

Start line: 253, End line: 281

```python
class DatabaseCreation(BaseDatabaseCreation):

    def _execute_test_db_destruction(self, cursor, parameters, verbosity):
        if verbosity >= 2:
            self.log('_execute_test_db_destruction(): dbname=%s' % parameters['user'])
        statements = [
            'DROP TABLESPACE %(tblspace)s INCLUDING CONTENTS AND DATAFILES CASCADE CONSTRAINTS',
            'DROP TABLESPACE %(tblspace_temp)s INCLUDING CONTENTS AND DATAFILES CASCADE CONSTRAINTS',
        ]
        self._execute_statements(cursor, statements, parameters, verbosity)

    def _destroy_test_user(self, cursor, parameters, verbosity):
        if verbosity >= 2:
            self.log('_destroy_test_user(): user=%s' % parameters['user'])
            self.log('Be patient. This can take some time...')
        statements = [
            'DROP USER %(user)s CASCADE',
        ]
        self._execute_statements(cursor, statements, parameters, verbosity)

    def _execute_statements(self, cursor, statements, parameters, verbosity, allow_quiet_fail=False):
        for template in statements:
            stmt = template % parameters
            if verbosity >= 2:
                print(stmt)
            try:
                cursor.execute(stmt)
            except Exception as err:
                if (not allow_quiet_fail) or verbosity >= 2:
                    self.log('Failed (%s)' % (err))
                raise
```
### 7 - django/db/backends/base/creation.py:

Start line: 1, End line: 84

```python
import os
import sys
from io import StringIO

from django.apps import apps
from django.conf import settings
from django.core import serializers
from django.db import router

# The prefix to put on the default database name when creating
# the test database.
TEST_DATABASE_PREFIX = 'test_'


class BaseDatabaseCreation:
    """
    Encapsulate backend-specific differences pertaining to creation and
    destruction of the test database.
    """
    def __init__(self, connection):
        self.connection = connection

    def _nodb_cursor(self):
        return self.connection._nodb_cursor()

    def log(self, msg):
        sys.stderr.write(msg + os.linesep)

    def create_test_db(self, verbosity=1, autoclobber=False, serialize=True, keepdb=False):
        """
        Create a test database, prompting the user for confirmation if the
        database already exists. Return the name of the test database created.
        """
        # Don't import django.core.management if it isn't needed.
        from django.core.management import call_command

        test_database_name = self._get_test_db_name()

        if verbosity >= 1:
            action = 'Creating'
            if keepdb:
                action = "Using existing"

            self.log('%s test database for alias %s...' % (
                action,
                self._get_database_display_str(verbosity, test_database_name),
            ))

        # We could skip this call if keepdb is True, but we instead
        # give it the keepdb param. This is to handle the case
        # where the test DB doesn't exist, in which case we need to
        # create it, then just not destroy it. If we instead skip
        # this, we will get an exception.
        self._create_test_db(verbosity, autoclobber, keepdb)

        self.connection.close()
        settings.DATABASES[self.connection.alias]["NAME"] = test_database_name
        self.connection.settings_dict["NAME"] = test_database_name

        if self.connection.settings_dict['TEST']['MIGRATE']:
            # We report migrate messages at one level lower than that
            # requested. This ensures we don't get flooded with messages during
            # testing (unless you really ask to be flooded).
            call_command(
                'migrate',
                verbosity=max(verbosity - 1, 0),
                interactive=False,
                database=self.connection.alias,
                run_syncdb=True,
            )

        # We then serialize the current state of the database into a string
        # and store it on the connection. This slightly horrific process is so people
        # who are testing on databases without transactions or who are using
        # a TransactionTestCase still get a clean database on every test run.
        if serialize:
            self.connection._test_serialized_contents = self.serialize_db_to_string()

        call_command('createcachetable', database=self.connection.alias)

        # Ensure a connection for the side effect of initializing the test database.
        self.connection.ensure_connection()

        return test_database_name
```
### 8 - django/core/management/commands/loaddata.py:

Start line: 81, End line: 148

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
                    self.stdout.write("Resetting sequences\n")
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
### 9 - django/db/backends/postgresql/operations.py:

Start line: 165, End line: 207

```python
class DatabaseOperations(BaseDatabaseOperations):

    def sequence_reset_sql(self, style, model_list):
        from django.db import models
        output = []
        qn = self.quote_name
        for model in model_list:
            # Use `coalesce` to set the sequence for each model to the max pk value if there are records,
            # or 1 if there are none. Set the `is_called` property (the third argument to `setval`) to true
            # if there are records (as the max pk value is already in use), otherwise set it to false.
            # Use pg_get_serial_sequence to get the underlying sequence name from the table name
            # and column name (available since PostgreSQL 8)

            for f in model._meta.local_fields:
                if isinstance(f, models.AutoField):
                    output.append(
                        "%s setval(pg_get_serial_sequence('%s','%s'), "
                        "coalesce(max(%s), 1), max(%s) %s null) %s %s;" % (
                            style.SQL_KEYWORD('SELECT'),
                            style.SQL_TABLE(qn(model._meta.db_table)),
                            style.SQL_FIELD(f.column),
                            style.SQL_FIELD(qn(f.column)),
                            style.SQL_FIELD(qn(f.column)),
                            style.SQL_KEYWORD('IS NOT'),
                            style.SQL_KEYWORD('FROM'),
                            style.SQL_TABLE(qn(model._meta.db_table)),
                        )
                    )
                    break  # Only one AutoField is allowed per model, so don't bother continuing.
            for f in model._meta.many_to_many:
                if not f.remote_field.through:
                    output.append(
                        "%s setval(pg_get_serial_sequence('%s','%s'), "
                        "coalesce(max(%s), 1), max(%s) %s null) %s %s;" % (
                            style.SQL_KEYWORD('SELECT'),
                            style.SQL_TABLE(qn(f.m2m_db_table())),
                            style.SQL_FIELD('id'),
                            style.SQL_FIELD(qn('id')),
                            style.SQL_FIELD(qn('id')),
                            style.SQL_KEYWORD('IS NOT'),
                            style.SQL_KEYWORD('FROM'),
                            style.SQL_TABLE(qn(f.m2m_db_table()))
                        )
                    )
        return output
```
### 10 - django/db/backends/postgresql/creation.py:

Start line: 53, End line: 78

```python
class DatabaseCreation(BaseDatabaseCreation):

    def _clone_test_db(self, suffix, verbosity, keepdb=False):
        # CREATE DATABASE ... WITH TEMPLATE ... requires closing connections
        # to the template database.
        self.connection.close()

        source_database_name = self.connection.settings_dict['NAME']
        target_database_name = self.get_test_db_clone_settings(suffix)['NAME']
        test_db_params = {
            'dbname': self._quote_name(target_database_name),
            'suffix': self._get_database_create_suffix(template=source_database_name),
        }
        with self._nodb_cursor() as cursor:
            try:
                self._execute_create_test_db(cursor, test_db_params, keepdb)
            except Exception:
                try:
                    if verbosity >= 1:
                        self.log('Destroying old test database for alias %s...' % (
                            self._get_database_display_str(verbosity, target_database_name),
                        ))
                    cursor.execute('DROP DATABASE %(dbname)s' % test_db_params)
                    self._execute_create_test_db(cursor, test_db_params, keepdb)
                except Exception as e:
                    self.log('Got an error cloning the test database: %s' % e)
                    sys.exit(2)
```
### 17 - django/db/backends/base/creation.py:

Start line: 86, End line: 121

```python
class BaseDatabaseCreation:

    def set_as_test_mirror(self, primary_settings_dict):
        """
        Set this database up to be used in testing as a mirror of a primary
        database whose settings are given.
        """
        self.connection.settings_dict['NAME'] = primary_settings_dict['NAME']

    def serialize_db_to_string(self):
        """
        Serialize all data in the database into a JSON string.
        Designed only for test runner usage; will not handle large
        amounts of data.
        """
        # Build list of all apps to serialize
        from django.db.migrations.loader import MigrationLoader
        loader = MigrationLoader(self.connection)
        app_list = []
        for app_config in apps.get_app_configs():
            if (
                app_config.models_module is not None and
                app_config.label in loader.migrated_apps and
                app_config.name not in settings.TEST_NON_SERIALIZED_APPS
            ):
                app_list.append((app_config, None))

        # Make a function to iteratively return every object
        def get_objects():
            for model in serializers.sort_dependencies(app_list):
                if (model._meta.can_migrate(self.connection) and
                        router.allow_migrate_model(self.connection.alias, model)):
                    queryset = model._default_manager.using(self.connection.alias).order_by(model._meta.pk.name)
                    yield from queryset.iterator()
        # Serialize to a string
        out = StringIO()
        serializers.serialize("json", get_objects(), indent=None, stream=out)
        return out.getvalue()
```
### 20 - django/db/backends/base/creation.py:

Start line: 155, End line: 194

```python
class BaseDatabaseCreation:

    def _create_test_db(self, verbosity, autoclobber, keepdb=False):
        """
        Internal implementation - create the test db tables.
        """
        test_database_name = self._get_test_db_name()
        test_db_params = {
            'dbname': self.connection.ops.quote_name(test_database_name),
            'suffix': self.sql_table_creation_suffix(),
        }
        # Create the test database and connect to it.
        with self._nodb_cursor() as cursor:
            try:
                self._execute_create_test_db(cursor, test_db_params, keepdb)
            except Exception as e:
                # if we want to keep the db, then no need to do any of the below,
                # just return and skip it all.
                if keepdb:
                    return test_database_name

                self.log('Got an error creating the test database: %s' % e)
                if not autoclobber:
                    confirm = input(
                        "Type 'yes' if you would like to try deleting the test "
                        "database '%s', or 'no' to cancel: " % test_database_name)
                if autoclobber or confirm == 'yes':
                    try:
                        if verbosity >= 1:
                            self.log('Destroying old test database for alias %s...' % (
                                self._get_database_display_str(verbosity, test_database_name),
                            ))
                        cursor.execute('DROP DATABASE %(dbname)s' % test_db_params)
                        self._execute_create_test_db(cursor, test_db_params, keepdb)
                    except Exception as e:
                        self.log('Got an error recreating the test database: %s' % e)
                        sys.exit(2)
                else:
                    self.log('Tests cancelled.')
                    sys.exit(1)

        return test_database_name
```
### 24 - django/db/backends/base/creation.py:

Start line: 263, End line: 294

```python
class BaseDatabaseCreation:

    def _destroy_test_db(self, test_database_name, verbosity):
        """
        Internal implementation - remove the test db tables.
        """
        # Remove the test database to clean up after
        # ourselves. Connect to the previous database (not the test database)
        # to do so, because it's not allowed to delete a database while being
        # connected to it.
        with self._nodb_cursor() as cursor:
            cursor.execute("DROP DATABASE %s"
                           % self.connection.ops.quote_name(test_database_name))

    def sql_table_creation_suffix(self):
        """
        SQL to append to the end of the test table creation statements.
        """
        return ''

    def test_db_signature(self):
        """
        Return a tuple with elements of self.connection.settings_dict (a
        DATABASES setting value) that uniquely identify a database
        accordingly to the RDBMS particularities.
        """
        settings_dict = self.connection.settings_dict
        return (
            settings_dict['HOST'],
            settings_dict['PORT'],
            settings_dict['ENGINE'],
            self._get_test_db_name(),
        )
```
### 30 - django/db/backends/base/creation.py:

Start line: 196, End line: 213

```python
class BaseDatabaseCreation:

    def clone_test_db(self, suffix, verbosity=1, autoclobber=False, keepdb=False):
        """
        Clone a test database.
        """
        source_database_name = self.connection.settings_dict['NAME']

        if verbosity >= 1:
            action = 'Cloning test database'
            if keepdb:
                action = 'Using existing clone'
            self.log('%s for alias %s...' % (
                action,
                self._get_database_display_str(verbosity, source_database_name),
            ))

        # We could skip this call if keepdb is True, but we instead
        # give it the keepdb param. See create_test_db for details.
        self._clone_test_db(suffix, verbosity, keepdb)
```
### 60 - django/db/backends/base/creation.py:

Start line: 215, End line: 231

```python
class BaseDatabaseCreation:

    def get_test_db_clone_settings(self, suffix):
        """
        Return a modified connection settings dict for the n-th clone of a DB.
        """
        # When this function is called, the test database has been created
        # already and its name has been copied to settings_dict['NAME'] so
        # we don't need to call _get_test_db_name.
        orig_settings_dict = self.connection.settings_dict
        return {**orig_settings_dict, 'NAME': '{}_{}'.format(orig_settings_dict['NAME'], suffix)}

    def _clone_test_db(self, suffix, verbosity, keepdb=False):
        """
        Internal implementation - duplicate the test db tables.
        """
        raise NotImplementedError(
            "The database backend doesn't support cloning databases. "
            "Disable the option to run tests in parallel processes.")
```
### 168 - django/db/backends/base/creation.py:

Start line: 233, End line: 261

```python
class BaseDatabaseCreation:

    def destroy_test_db(self, old_database_name=None, verbosity=1, keepdb=False, suffix=None):
        """
        Destroy a test database, prompting the user for confirmation if the
        database already exists.
        """
        self.connection.close()
        if suffix is None:
            test_database_name = self.connection.settings_dict['NAME']
        else:
            test_database_name = self.get_test_db_clone_settings(suffix)['NAME']

        if verbosity >= 1:
            action = 'Destroying'
            if keepdb:
                action = 'Preserving'
            self.log('%s test database for alias %s...' % (
                action,
                self._get_database_display_str(verbosity, test_database_name),
            ))

        # if we want to preserve the database
        # skip the actual destroying piece.
        if not keepdb:
            self._destroy_test_db(test_database_name, verbosity)

        # Restore the original database name
        if old_database_name is not None:
            settings.DATABASES[self.connection.alias]["NAME"] = old_database_name
            self.connection.settings_dict["NAME"] = old_database_name
```
