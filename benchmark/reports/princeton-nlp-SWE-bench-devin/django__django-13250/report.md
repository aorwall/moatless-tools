# django__django-13250

| **django/django** | `bac5777bff8e8d8189193438b5af52f158a3f2a4` |
| ---- | ---- |
| **No of patches** | 5 |
| **All found context length** | 4566 |
| **Any found context length** | 4566 |
| **Avg pos** | 34.8 |
| **Min pos** | 17 |
| **Max pos** | 67 |
| **Top file pos** | 1 |
| **Missing snippets** | 7 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/db/backends/base/features.py b/django/db/backends/base/features.py
--- a/django/db/backends/base/features.py
+++ b/django/db/backends/base/features.py
@@ -295,6 +295,9 @@ class BaseDatabaseFeatures:
     has_native_json_field = False
     # Does the backend use PostgreSQL-style JSON operators like '->'?
     has_json_operators = False
+    # Does the backend support __contains and __contained_by lookups for
+    # a JSONField?
+    supports_json_field_contains = True
 
     def __init__(self, connection):
         self.connection = connection
diff --git a/django/db/backends/oracle/features.py b/django/db/backends/oracle/features.py
--- a/django/db/backends/oracle/features.py
+++ b/django/db/backends/oracle/features.py
@@ -60,6 +60,7 @@ class DatabaseFeatures(BaseDatabaseFeatures):
     allows_multiple_constraints_on_same_fields = False
     supports_boolean_expr_in_select_clause = False
     supports_primitives_in_json_field = False
+    supports_json_field_contains = False
 
     @cached_property
     def introspected_field_types(self):
diff --git a/django/db/backends/sqlite3/base.py b/django/db/backends/sqlite3/base.py
--- a/django/db/backends/sqlite3/base.py
+++ b/django/db/backends/sqlite3/base.py
@@ -5,7 +5,6 @@
 import decimal
 import functools
 import hashlib
-import json
 import math
 import operator
 import re
@@ -235,7 +234,6 @@ def get_new_connection(self, conn_params):
         create_deterministic_function('DEGREES', 1, none_guard(math.degrees))
         create_deterministic_function('EXP', 1, none_guard(math.exp))
         create_deterministic_function('FLOOR', 1, none_guard(math.floor))
-        create_deterministic_function('JSON_CONTAINS', 2, _sqlite_json_contains)
         create_deterministic_function('LN', 1, none_guard(math.log))
         create_deterministic_function('LOG', 2, none_guard(lambda x, y: math.log(y, x)))
         create_deterministic_function('LPAD', 3, _sqlite_lpad)
@@ -601,11 +599,3 @@ def _sqlite_lpad(text, length, fill_text):
 @none_guard
 def _sqlite_rpad(text, length, fill_text):
     return (text + fill_text * length)[:length]
-
-
-@none_guard
-def _sqlite_json_contains(haystack, needle):
-    target, candidate = json.loads(haystack), json.loads(needle)
-    if isinstance(target, dict) and isinstance(candidate, dict):
-        return target.items() >= candidate.items()
-    return target == candidate
diff --git a/django/db/backends/sqlite3/features.py b/django/db/backends/sqlite3/features.py
--- a/django/db/backends/sqlite3/features.py
+++ b/django/db/backends/sqlite3/features.py
@@ -43,6 +43,7 @@ class DatabaseFeatures(BaseDatabaseFeatures):
     supports_aggregate_filter_clause = Database.sqlite_version_info >= (3, 30, 1)
     supports_order_by_nulls_modifier = Database.sqlite_version_info >= (3, 30, 0)
     order_by_nulls_first = True
+    supports_json_field_contains = False
 
     @cached_property
     def supports_atomic_references_rename(self):
diff --git a/django/db/models/fields/json.py b/django/db/models/fields/json.py
--- a/django/db/models/fields/json.py
+++ b/django/db/models/fields/json.py
@@ -140,28 +140,30 @@ class DataContains(PostgresOperatorLookup):
     postgres_operator = '@>'
 
     def as_sql(self, compiler, connection):
+        if not connection.features.supports_json_field_contains:
+            raise NotSupportedError(
+                'contains lookup is not supported on this database backend.'
+            )
         lhs, lhs_params = self.process_lhs(compiler, connection)
         rhs, rhs_params = self.process_rhs(compiler, connection)
         params = tuple(lhs_params) + tuple(rhs_params)
         return 'JSON_CONTAINS(%s, %s)' % (lhs, rhs), params
 
-    def as_oracle(self, compiler, connection):
-        raise NotSupportedError('contains lookup is not supported on Oracle.')
-
 
 class ContainedBy(PostgresOperatorLookup):
     lookup_name = 'contained_by'
     postgres_operator = '<@'
 
     def as_sql(self, compiler, connection):
+        if not connection.features.supports_json_field_contains:
+            raise NotSupportedError(
+                'contained_by lookup is not supported on this database backend.'
+            )
         lhs, lhs_params = self.process_lhs(compiler, connection)
         rhs, rhs_params = self.process_rhs(compiler, connection)
         params = tuple(rhs_params) + tuple(lhs_params)
         return 'JSON_CONTAINS(%s, %s)' % (rhs, lhs), params
 
-    def as_oracle(self, compiler, connection):
-        raise NotSupportedError('contained_by lookup is not supported on Oracle.')
-
 
 class HasKeyLookup(PostgresOperatorLookup):
     logical_operator = None

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/db/backends/base/features.py | 298 | 298 | 29 | 13 | 8612
| django/db/backends/oracle/features.py | 63 | 63 | 61 | 24 | 19371
| django/db/backends/sqlite3/base.py | 8 | 8 | - | 25 | -
| django/db/backends/sqlite3/base.py | 238 | 238 | - | 25 | -
| django/db/backends/sqlite3/base.py | 604 | 611 | 67 | 25 | 21213
| django/db/backends/sqlite3/features.py | 46 | 46 | 17 | 6 | 4566
| django/db/models/fields/json.py | 143 | 159 | - | 1 | -


## Problem Statement

```
JSONField's __contains and __contained_by lookups don't work with nested values on SQLite.
Description
	
SQLite doesn't provide a native way for testing containment of JSONField. The current implementation works only for basic examples without supporting nested structures and doesn't follow "the general principle that the contained object must match the containing object as to structure and data contents, possibly after discarding some non-matching array elements or object key/value pairs from the containing object".
I'm not sure if it's feasible to emulate it in Python.
Some (not really complicated) examples that don't work:
diff --git a/tests/model_fields/test_jsonfield.py b/tests/model_fields/test_jsonfield.py
index 9a9e1a1286..1acc5af73e 100644
--- a/tests/model_fields/test_jsonfield.py
+++ b/tests/model_fields/test_jsonfield.py
@@ -449,9 +449,14 @@ class TestQuerying(TestCase):
		 tests = [
			 ({}, self.objs[2:5] + self.objs[6:8]),
			 ({'baz': {'a': 'b', 'c': 'd'}}, [self.objs[7]]),
+			({'baz': {'a': 'b'}}, [self.objs[7]]),
+			({'baz': {'c': 'd'}}, [self.objs[7]]),
			 ({'k': True, 'l': False}, [self.objs[6]]),
			 ({'d': ['e', {'f': 'g'}]}, [self.objs[4]]),
+			({'d': ['e']}, [self.objs[4]]),
			 ([1, [2]], [self.objs[5]]),
+			([1], [self.objs[5]]),
+			([[2]], [self.objs[5]]),
			 ({'n': [None]}, [self.objs[4]]),
			 ({'j': None}, [self.objs[4]]),
		 ]

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 django/db/models/fields/json.py** | 152 | 163| 120 | 120 | 3953 | 
| 2 | **1 django/db/models/fields/json.py** | 218 | 259| 318 | 438 | 3953 | 
| 3 | **1 django/db/models/fields/json.py** | 1 | 40| 273 | 711 | 3953 | 
| 4 | **1 django/db/models/fields/json.py** | 62 | 122| 398 | 1109 | 3953 | 
| 5 | **1 django/db/models/fields/json.py** | 125 | 149| 199 | 1308 | 3953 | 
| 6 | **1 django/db/models/fields/json.py** | 262 | 290| 236 | 1544 | 3953 | 
| 7 | 2 django/forms/fields.py | 1219 | 1272| 351 | 1895 | 13300 | 
| 8 | **2 django/db/models/fields/json.py** | 197 | 215| 232 | 2127 | 13300 | 
| 9 | **2 django/db/models/fields/json.py** | 42 | 60| 125 | 2252 | 13300 | 
| 10 | **2 django/db/models/fields/json.py** | 366 | 376| 131 | 2383 | 13300 | 
| 11 | **2 django/db/models/fields/json.py** | 166 | 195| 275 | 2658 | 13300 | 
| 12 | 3 django/db/backends/sqlite3/introspection.py | 23 | 54| 296 | 2954 | 17149 | 
| 13 | **3 django/db/models/fields/json.py** | 379 | 407| 301 | 3255 | 17149 | 
| 14 | 4 django/db/models/lookups.py | 449 | 486| 195 | 3450 | 22097 | 
| 15 | 5 django/db/models/query.py | 845 | 874| 248 | 3698 | 39199 | 
| 16 | **5 django/db/models/fields/json.py** | 409 | 421| 174 | 3872 | 39199 | 
| **-> 17 <-** | **6 django/db/backends/sqlite3/features.py** | 1 | 75| 694 | 4566 | 39893 | 
| 18 | 7 django/db/models/fields/related.py | 127 | 154| 201 | 4767 | 53769 | 
| 19 | 8 django/contrib/postgres/forms/hstore.py | 1 | 59| 339 | 5106 | 54108 | 
| 20 | 9 django/contrib/postgres/lookups.py | 1 | 61| 337 | 5443 | 54445 | 
| 21 | 10 django/db/models/sql/query.py | 1106 | 1138| 338 | 5781 | 76791 | 
| 22 | 10 django/db/models/fields/related.py | 190 | 254| 673 | 6454 | 76791 | 
| 23 | 10 django/db/models/fields/related.py | 487 | 507| 138 | 6592 | 76791 | 
| 24 | 10 django/db/models/fields/related.py | 255 | 282| 269 | 6861 | 76791 | 
| 25 | 11 django/contrib/postgres/fields/hstore.py | 1 | 69| 435 | 7296 | 77491 | 
| 26 | 11 django/db/models/lookups.py | 608 | 641| 141 | 7437 | 77491 | 
| 27 | 11 django/db/models/fields/related.py | 156 | 169| 144 | 7581 | 77491 | 
| 28 | 12 django/contrib/contenttypes/fields.py | 110 | 158| 328 | 7909 | 82924 | 
| **-> 29 <-** | **13 django/db/backends/base/features.py** | 219 | 305| 703 | 8612 | 85512 | 
| 30 | **13 django/db/models/fields/json.py** | 305 | 318| 184 | 8796 | 85512 | 
| 31 | 14 django/db/models/fields/__init__.py | 308 | 336| 205 | 9001 | 103201 | 
| 32 | 15 django/contrib/gis/db/models/lookups.py | 86 | 217| 762 | 9763 | 105814 | 
| 33 | 16 django/contrib/postgres/fields/array.py | 1 | 15| 110 | 9873 | 107895 | 
| 34 | **16 django/db/models/fields/json.py** | 424 | 509| 522 | 10395 | 107895 | 
| 35 | 16 django/db/models/sql/query.py | 1693 | 1727| 399 | 10794 | 107895 | 
| 36 | 16 django/db/models/fields/related.py | 509 | 574| 492 | 11286 | 107895 | 
| 37 | 17 django/contrib/admin/checks.py | 247 | 277| 229 | 11515 | 117032 | 
| 38 | 17 django/db/models/fields/related.py | 1235 | 1352| 963 | 12478 | 117032 | 
| 39 | **17 django/db/models/fields/json.py** | 321 | 343| 181 | 12659 | 117032 | 
| 40 | 17 django/db/models/lookups.py | 303 | 353| 306 | 12965 | 117032 | 
| 41 | 17 django/db/models/query.py | 804 | 843| 322 | 13287 | 117032 | 
| 42 | **17 django/db/models/fields/json.py** | 346 | 363| 170 | 13457 | 117032 | 
| 43 | 17 django/db/models/fields/related.py | 108 | 125| 155 | 13612 | 117032 | 
| 44 | 17 django/contrib/contenttypes/fields.py | 332 | 354| 185 | 13797 | 117032 | 
| 45 | **17 django/db/models/fields/json.py** | 292 | 303| 153 | 13950 | 117032 | 
| 46 | 17 django/db/models/lookups.py | 356 | 388| 294 | 14244 | 117032 | 
| 47 | 18 django/db/models/query_utils.py | 284 | 309| 293 | 14537 | 119738 | 
| 48 | 18 django/contrib/contenttypes/fields.py | 173 | 217| 411 | 14948 | 119738 | 
| 49 | 18 django/contrib/admin/checks.py | 294 | 327| 381 | 15329 | 119738 | 
| 50 | 18 django/contrib/gis/db/models/lookups.py | 220 | 247| 134 | 15463 | 119738 | 
| 51 | 18 django/db/models/lookups.py | 208 | 243| 308 | 15771 | 119738 | 
| 52 | 19 django/db/migrations/operations/fields.py | 1 | 37| 241 | 16012 | 122836 | 
| 53 | 20 django/contrib/postgres/fields/jsonb.py | 1 | 44| 312 | 16324 | 123148 | 
| 54 | 21 django/db/models/fields/related_lookups.py | 119 | 155| 244 | 16568 | 124597 | 
| 55 | 22 django/contrib/postgres/fields/ranges.py | 199 | 228| 279 | 16847 | 126689 | 
| 56 | 23 django/db/models/__init__.py | 1 | 53| 619 | 17466 | 127308 | 
| 57 | 23 django/contrib/gis/db/models/lookups.py | 301 | 323| 202 | 17668 | 127308 | 
| 58 | 23 django/db/models/sql/query.py | 2161 | 2206| 371 | 18039 | 127308 | 
| 59 | 23 django/db/models/fields/related.py | 284 | 318| 293 | 18332 | 127308 | 
| 60 | 23 django/db/models/fields/related_lookups.py | 62 | 99| 447 | 18779 | 127308 | 
| **-> 61 <-** | **24 django/db/backends/oracle/features.py** | 1 | 75| 592 | 19371 | 127900 | 
| 62 | 24 django/contrib/postgres/fields/ranges.py | 165 | 196| 272 | 19643 | 127900 | 
| 63 | 24 django/db/models/sql/query.py | 1464 | 1549| 801 | 20444 | 127900 | 
| 64 | 24 django/contrib/postgres/fields/hstore.py | 72 | 112| 264 | 20708 | 127900 | 
| 65 | 24 django/contrib/admin/checks.py | 329 | 340| 138 | 20846 | 127900 | 
| 66 | 24 django/db/models/fields/related.py | 171 | 188| 166 | 21012 | 127900 | 
| **-> 67 <-** | **25 django/db/backends/sqlite3/base.py** | 582 | 612| 201 | 21213 | 133917 | 
| 68 | 25 django/db/models/fields/__init__.py | 338 | 365| 203 | 21416 | 133917 | 
| 69 | 25 django/db/models/sql/query.py | 2349 | 2365| 119 | 21535 | 133917 | 
| 70 | 25 django/db/models/fields/related_lookups.py | 1 | 23| 170 | 21705 | 133917 | 
| 71 | 25 django/db/models/sql/query.py | 697 | 732| 389 | 22094 | 133917 | 
| 72 | 26 django/db/models/expressions.py | 1143 | 1171| 266 | 22360 | 144694 | 
| 73 | 26 django/contrib/gis/db/models/lookups.py | 338 | 360| 124 | 22484 | 144694 | 
| 74 | 26 django/db/models/fields/__init__.py | 208 | 242| 234 | 22718 | 144694 | 
| 75 | 26 django/db/models/lookups.py | 288 | 300| 168 | 22886 | 144694 | 
| 76 | 26 django/db/models/fields/__init__.py | 244 | 306| 448 | 23334 | 144694 | 
| 77 | 26 django/contrib/contenttypes/fields.py | 1 | 17| 134 | 23468 | 144694 | 
| 78 | 27 django/contrib/gis/db/backends/base/features.py | 1 | 100| 752 | 24220 | 145447 | 
| 79 | 28 django/core/serializers/xml_serializer.py | 93 | 114| 192 | 24412 | 148959 | 
| 80 | 28 django/db/models/fields/related.py | 1 | 34| 246 | 24658 | 148959 | 
| 81 | 28 django/contrib/admin/checks.py | 279 | 292| 135 | 24793 | 148959 | 
| 82 | 28 django/db/models/sql/query.py | 1077 | 1104| 285 | 25078 | 148959 | 
| 83 | 28 django/db/models/lookups.py | 100 | 142| 359 | 25437 | 148959 | 
| 84 | 28 django/db/models/sql/query.py | 2316 | 2332| 177 | 25614 | 148959 | 
| 85 | 29 django/db/models/sql/where.py | 157 | 190| 233 | 25847 | 150763 | 
| 86 | 29 django/db/models/fields/related.py | 630 | 650| 168 | 26015 | 150763 | 
| 87 | 29 django/db/models/query.py | 327 | 353| 222 | 26237 | 150763 | 
| 88 | 29 django/db/models/query.py | 787 | 803| 157 | 26394 | 150763 | 
| 89 | 30 django/contrib/gis/forms/fields.py | 108 | 134| 123 | 26517 | 151683 | 
| 90 | 30 django/contrib/gis/db/models/lookups.py | 250 | 275| 222 | 26739 | 151683 | 
| 91 | 30 django/db/models/query.py | 1323 | 1332| 114 | 26853 | 151683 | 
| 92 | 30 django/db/models/fields/related.py | 997 | 1024| 215 | 27068 | 151683 | 
| 93 | **30 django/db/backends/base/features.py** | 1 | 113| 899 | 27967 | 151683 | 
| 94 | 30 django/db/models/fields/related.py | 750 | 768| 222 | 28189 | 151683 | 
| 95 | 30 django/db/models/fields/related_lookups.py | 46 | 60| 224 | 28413 | 151683 | 
| 96 | 31 django/db/backends/postgresql/features.py | 1 | 91| 726 | 29139 | 152409 | 
| 97 | 32 django/forms/boundfield.py | 53 | 78| 180 | 29319 | 154565 | 
| 98 | 33 django/db/backends/sqlite3/operations.py | 40 | 66| 232 | 29551 | 157596 | 
| 99 | 33 django/db/models/fields/__init__.py | 495 | 506| 184 | 29735 | 157596 | 
| 100 | 34 django/contrib/admin/filters.py | 446 | 475| 226 | 29961 | 161689 | 
| 101 | 35 django/contrib/gis/db/models/fields.py | 168 | 194| 239 | 30200 | 164740 | 
| 102 | **35 django/db/backends/sqlite3/base.py** | 81 | 155| 757 | 30957 | 164740 | 
| 103 | 35 django/db/models/fields/related.py | 1354 | 1426| 616 | 31573 | 164740 | 
| 104 | 35 django/db/models/lookups.py | 188 | 205| 175 | 31748 | 164740 | 
| 105 | 35 django/db/models/sql/where.py | 230 | 246| 130 | 31878 | 164740 | 
| 106 | 35 django/db/models/expressions.py | 590 | 629| 290 | 32168 | 164740 | 
| 107 | 35 django/core/serializers/xml_serializer.py | 116 | 156| 360 | 32528 | 164740 | 
| 108 | 35 django/db/models/query_utils.py | 1 | 22| 178 | 32706 | 164740 | 
| 109 | 35 django/db/models/fields/__init__.py | 963 | 979| 176 | 32882 | 164740 | 
| 110 | 36 django/contrib/postgres/forms/jsonb.py | 1 | 17| 108 | 32990 | 164848 | 
| 111 | 36 django/contrib/admin/checks.py | 947 | 976| 243 | 33233 | 164848 | 
| 112 | 36 django/contrib/admin/checks.py | 232 | 245| 161 | 33394 | 164848 | 
| 113 | 36 django/db/models/fields/related_lookups.py | 102 | 117| 215 | 33609 | 164848 | 
| 114 | 36 django/forms/boundfield.py | 36 | 51| 149 | 33758 | 164848 | 
| 115 | 36 django/contrib/postgres/fields/ranges.py | 93 | 112| 160 | 33918 | 164848 | 
| 116 | 36 django/db/models/sql/query.py | 1419 | 1446| 283 | 34201 | 164848 | 
| 117 | 36 django/contrib/postgres/fields/ranges.py | 115 | 162| 262 | 34463 | 164848 | 
| 118 | 36 django/contrib/gis/db/models/fields.py | 1 | 20| 193 | 34656 | 164848 | 
| 119 | 36 django/forms/fields.py | 1179 | 1216| 199 | 34855 | 164848 | 
| 120 | 36 django/db/models/query.py | 1468 | 1501| 297 | 35152 | 164848 | 
| 121 | 37 django/contrib/postgres/forms/array.py | 62 | 102| 248 | 35400 | 166442 | 
| 122 | 38 django/db/models/base.py | 1654 | 1702| 348 | 35748 | 183023 | 
| 123 | 38 django/db/models/expressions.py | 1 | 30| 204 | 35952 | 183023 | 
| 124 | 38 django/contrib/gis/db/models/fields.py | 280 | 340| 361 | 36313 | 183023 | 
| 125 | 39 django/contrib/gis/serializers/geojson.py | 38 | 68| 280 | 36593 | 183602 | 
| 126 | 39 django/forms/fields.py | 175 | 207| 280 | 36873 | 183602 | 
| 127 | 40 django/contrib/gis/db/models/__init__.py | 1 | 19| 204 | 37077 | 183806 | 
| 128 | 40 django/contrib/gis/forms/fields.py | 87 | 105| 162 | 37239 | 183806 | 
| 129 | 40 django/db/models/query.py | 1184 | 1203| 209 | 37448 | 183806 | 
| 130 | 41 django/db/models/sql/datastructures.py | 1 | 21| 126 | 37574 | 185208 | 
| 131 | 41 django/contrib/postgres/fields/ranges.py | 231 | 321| 479 | 38053 | 185208 | 
| 132 | 41 django/contrib/admin/checks.py | 430 | 439| 125 | 38178 | 185208 | 
| 133 | 41 django/db/models/query.py | 1334 | 1382| 405 | 38583 | 185208 | 
| 134 | 41 django/db/models/fields/__init__.py | 367 | 393| 199 | 38782 | 185208 | 
| 135 | 41 django/contrib/admin/checks.py | 526 | 536| 134 | 38916 | 185208 | 
| 136 | 42 django/db/backends/mysql/features.py | 1 | 108| 816 | 39732 | 186579 | 
| 137 | 42 django/db/models/query.py | 1385 | 1416| 246 | 39978 | 186579 | 
| 138 | 42 django/db/models/base.py | 1588 | 1652| 514 | 40492 | 186579 | 
| 139 | 43 django/db/backends/oracle/introspection.py | 1 | 49| 431 | 40923 | 189230 | 
| 140 | 43 django/db/models/base.py | 1704 | 1804| 729 | 41652 | 189230 | 
| 141 | 43 django/forms/fields.py | 730 | 762| 241 | 41893 | 189230 | 
| 142 | 43 django/db/models/fields/related.py | 611 | 628| 197 | 42090 | 189230 | 
| 143 | 43 django/db/models/fields/related.py | 864 | 890| 240 | 42330 | 189230 | 
| 144 | 43 django/db/models/sql/query.py | 1285 | 1350| 772 | 43102 | 189230 | 
| 145 | 43 django/db/models/fields/__init__.py | 417 | 494| 667 | 43769 | 189230 | 
| 146 | 43 django/contrib/admin/checks.py | 342 | 368| 221 | 43990 | 189230 | 
| 147 | 43 django/contrib/postgres/fields/ranges.py | 43 | 91| 362 | 44352 | 189230 | 
| 148 | 43 django/db/models/sql/query.py | 1729 | 1800| 784 | 45136 | 189230 | 
| 149 | 43 django/db/models/sql/query.py | 364 | 414| 494 | 45630 | 189230 | 
| 150 | 43 django/contrib/gis/db/models/fields.py | 239 | 250| 148 | 45778 | 189230 | 
| 151 | 43 django/db/models/lookups.py | 390 | 419| 337 | 46115 | 189230 | 
| 152 | 44 django/contrib/contenttypes/admin.py | 1 | 80| 615 | 46730 | 190255 | 
| 153 | 44 django/contrib/admin/checks.py | 724 | 734| 115 | 46845 | 190255 | 
| 154 | 44 django/contrib/postgres/fields/array.py | 53 | 75| 172 | 47017 | 190255 | 
| 155 | 44 django/db/models/sql/query.py | 2115 | 2132| 156 | 47173 | 190255 | 
| 156 | 44 django/contrib/postgres/forms/array.py | 197 | 235| 271 | 47444 | 190255 | 
| 157 | 44 django/db/models/fields/__init__.py | 2235 | 2296| 425 | 47869 | 190255 | 
| 158 | 44 django/db/models/fields/related.py | 935 | 948| 126 | 47995 | 190255 | 
| 159 | 44 django/db/models/fields/related.py | 652 | 668| 163 | 48158 | 190255 | 
| 160 | 44 django/db/models/fields/related.py | 1428 | 1469| 418 | 48576 | 190255 | 
| 161 | 44 django/db/models/fields/__init__.py | 783 | 842| 431 | 49007 | 190255 | 
| 162 | 44 django/db/models/sql/query.py | 1050 | 1075| 214 | 49221 | 190255 | 
| 163 | 44 django/db/models/fields/related.py | 1202 | 1233| 180 | 49401 | 190255 | 
| 164 | 44 django/db/models/fields/__init__.py | 936 | 961| 209 | 49610 | 190255 | 
| 165 | 44 django/db/models/fields/related.py | 320 | 341| 225 | 49835 | 190255 | 
| 166 | 44 django/db/backends/sqlite3/operations.py | 1 | 38| 258 | 50093 | 190255 | 
| 167 | 44 django/contrib/admin/checks.py | 220 | 230| 127 | 50220 | 190255 | 
| 168 | 44 django/db/models/lookups.py | 1 | 40| 314 | 50534 | 190255 | 
| 169 | 44 django/contrib/contenttypes/fields.py | 160 | 171| 123 | 50657 | 190255 | 
| 170 | 44 django/contrib/admin/checks.py | 416 | 428| 137 | 50794 | 190255 | 
| 171 | 45 django/contrib/gis/db/models/functions.py | 325 | 345| 166 | 50960 | 194200 | 
| 172 | 46 django/core/serializers/base.py | 301 | 323| 207 | 51167 | 196625 | 
| 173 | 46 django/db/backends/sqlite3/introspection.py | 360 | 438| 750 | 51917 | 196625 | 
| 174 | 46 django/core/serializers/xml_serializer.py | 65 | 91| 219 | 52136 | 196625 | 
| 175 | 46 django/db/backends/sqlite3/operations.py | 312 | 357| 453 | 52589 | 196625 | 
| 176 | 46 django/db/models/base.py | 1394 | 1449| 491 | 53080 | 196625 | 
| 177 | 46 django/contrib/admin/filters.py | 229 | 244| 196 | 53276 | 196625 | 
| 178 | 46 django/db/models/fields/related.py | 841 | 862| 169 | 53445 | 196625 | 
| 179 | 46 django/contrib/admin/checks.py | 897 | 945| 416 | 53861 | 196625 | 
| 180 | 46 django/db/models/fields/__init__.py | 1061 | 1090| 218 | 54079 | 196625 | 
| 181 | 47 django/db/models/fields/reverse_related.py | 19 | 115| 635 | 54714 | 198768 | 
| 182 | 47 django/contrib/admin/checks.py | 492 | 502| 149 | 54863 | 198768 | 


### Hint

```
OK, grrrr... — Just a case of doc-ing the limitations. 🤨 (Thanks)
Draft.
I've attached a draft solution but it's really hot and it doesn't handle list with dicts (...deep rabbit hole). IMO we should drop support for these lookups on SQLite, at least for now. Testing containment of JSONField is really complicated, I hope SQLite and Oracle will prepare native solutions in future versions.
```

## Patch

```diff
diff --git a/django/db/backends/base/features.py b/django/db/backends/base/features.py
--- a/django/db/backends/base/features.py
+++ b/django/db/backends/base/features.py
@@ -295,6 +295,9 @@ class BaseDatabaseFeatures:
     has_native_json_field = False
     # Does the backend use PostgreSQL-style JSON operators like '->'?
     has_json_operators = False
+    # Does the backend support __contains and __contained_by lookups for
+    # a JSONField?
+    supports_json_field_contains = True
 
     def __init__(self, connection):
         self.connection = connection
diff --git a/django/db/backends/oracle/features.py b/django/db/backends/oracle/features.py
--- a/django/db/backends/oracle/features.py
+++ b/django/db/backends/oracle/features.py
@@ -60,6 +60,7 @@ class DatabaseFeatures(BaseDatabaseFeatures):
     allows_multiple_constraints_on_same_fields = False
     supports_boolean_expr_in_select_clause = False
     supports_primitives_in_json_field = False
+    supports_json_field_contains = False
 
     @cached_property
     def introspected_field_types(self):
diff --git a/django/db/backends/sqlite3/base.py b/django/db/backends/sqlite3/base.py
--- a/django/db/backends/sqlite3/base.py
+++ b/django/db/backends/sqlite3/base.py
@@ -5,7 +5,6 @@
 import decimal
 import functools
 import hashlib
-import json
 import math
 import operator
 import re
@@ -235,7 +234,6 @@ def get_new_connection(self, conn_params):
         create_deterministic_function('DEGREES', 1, none_guard(math.degrees))
         create_deterministic_function('EXP', 1, none_guard(math.exp))
         create_deterministic_function('FLOOR', 1, none_guard(math.floor))
-        create_deterministic_function('JSON_CONTAINS', 2, _sqlite_json_contains)
         create_deterministic_function('LN', 1, none_guard(math.log))
         create_deterministic_function('LOG', 2, none_guard(lambda x, y: math.log(y, x)))
         create_deterministic_function('LPAD', 3, _sqlite_lpad)
@@ -601,11 +599,3 @@ def _sqlite_lpad(text, length, fill_text):
 @none_guard
 def _sqlite_rpad(text, length, fill_text):
     return (text + fill_text * length)[:length]
-
-
-@none_guard
-def _sqlite_json_contains(haystack, needle):
-    target, candidate = json.loads(haystack), json.loads(needle)
-    if isinstance(target, dict) and isinstance(candidate, dict):
-        return target.items() >= candidate.items()
-    return target == candidate
diff --git a/django/db/backends/sqlite3/features.py b/django/db/backends/sqlite3/features.py
--- a/django/db/backends/sqlite3/features.py
+++ b/django/db/backends/sqlite3/features.py
@@ -43,6 +43,7 @@ class DatabaseFeatures(BaseDatabaseFeatures):
     supports_aggregate_filter_clause = Database.sqlite_version_info >= (3, 30, 1)
     supports_order_by_nulls_modifier = Database.sqlite_version_info >= (3, 30, 0)
     order_by_nulls_first = True
+    supports_json_field_contains = False
 
     @cached_property
     def supports_atomic_references_rename(self):
diff --git a/django/db/models/fields/json.py b/django/db/models/fields/json.py
--- a/django/db/models/fields/json.py
+++ b/django/db/models/fields/json.py
@@ -140,28 +140,30 @@ class DataContains(PostgresOperatorLookup):
     postgres_operator = '@>'
 
     def as_sql(self, compiler, connection):
+        if not connection.features.supports_json_field_contains:
+            raise NotSupportedError(
+                'contains lookup is not supported on this database backend.'
+            )
         lhs, lhs_params = self.process_lhs(compiler, connection)
         rhs, rhs_params = self.process_rhs(compiler, connection)
         params = tuple(lhs_params) + tuple(rhs_params)
         return 'JSON_CONTAINS(%s, %s)' % (lhs, rhs), params
 
-    def as_oracle(self, compiler, connection):
-        raise NotSupportedError('contains lookup is not supported on Oracle.')
-
 
 class ContainedBy(PostgresOperatorLookup):
     lookup_name = 'contained_by'
     postgres_operator = '<@'
 
     def as_sql(self, compiler, connection):
+        if not connection.features.supports_json_field_contains:
+            raise NotSupportedError(
+                'contained_by lookup is not supported on this database backend.'
+            )
         lhs, lhs_params = self.process_lhs(compiler, connection)
         rhs, rhs_params = self.process_rhs(compiler, connection)
         params = tuple(rhs_params) + tuple(lhs_params)
         return 'JSON_CONTAINS(%s, %s)' % (rhs, lhs), params
 
-    def as_oracle(self, compiler, connection):
-        raise NotSupportedError('contained_by lookup is not supported on Oracle.')
-
 
 class HasKeyLookup(PostgresOperatorLookup):
     logical_operator = None

```

## Test Patch

```diff
diff --git a/tests/model_fields/test_jsonfield.py b/tests/model_fields/test_jsonfield.py
--- a/tests/model_fields/test_jsonfield.py
+++ b/tests/model_fields/test_jsonfield.py
@@ -1,6 +1,6 @@
 import operator
 import uuid
-from unittest import mock, skipIf, skipUnless
+from unittest import mock, skipIf
 
 from django import forms
 from django.core import serializers
@@ -441,17 +441,20 @@ def test_has_any_keys(self):
             [self.objs[3], self.objs[4], self.objs[6]],
         )
 
-    @skipIf(
-        connection.vendor == 'oracle',
-        "Oracle doesn't support contains lookup.",
-    )
+    @skipUnlessDBFeature('supports_json_field_contains')
     def test_contains(self):
         tests = [
             ({}, self.objs[2:5] + self.objs[6:8]),
             ({'baz': {'a': 'b', 'c': 'd'}}, [self.objs[7]]),
+            ({'baz': {'a': 'b'}}, [self.objs[7]]),
+            ({'baz': {'c': 'd'}}, [self.objs[7]]),
             ({'k': True, 'l': False}, [self.objs[6]]),
             ({'d': ['e', {'f': 'g'}]}, [self.objs[4]]),
+            ({'d': ['e']}, [self.objs[4]]),
+            ({'d': [{'f': 'g'}]}, [self.objs[4]]),
             ([1, [2]], [self.objs[5]]),
+            ([1], [self.objs[5]]),
+            ([[2]], [self.objs[5]]),
             ({'n': [None]}, [self.objs[4]]),
             ({'j': None}, [self.objs[4]]),
         ]
@@ -460,38 +463,32 @@ def test_contains(self):
                 qs = NullableJSONModel.objects.filter(value__contains=value)
                 self.assertSequenceEqual(qs, expected)
 
-    @skipUnless(
-        connection.vendor == 'oracle',
-        "Oracle doesn't support contains lookup.",
-    )
+    @skipIfDBFeature('supports_json_field_contains')
     def test_contains_unsupported(self):
-        msg = 'contains lookup is not supported on Oracle.'
+        msg = 'contains lookup is not supported on this database backend.'
         with self.assertRaisesMessage(NotSupportedError, msg):
             NullableJSONModel.objects.filter(
                 value__contains={'baz': {'a': 'b', 'c': 'd'}},
             ).get()
 
-    @skipUnlessDBFeature('supports_primitives_in_json_field')
+    @skipUnlessDBFeature(
+        'supports_primitives_in_json_field',
+        'supports_json_field_contains',
+    )
     def test_contains_primitives(self):
         for value in self.primitives:
             with self.subTest(value=value):
                 qs = NullableJSONModel.objects.filter(value__contains=value)
                 self.assertIs(qs.exists(), True)
 
-    @skipIf(
-        connection.vendor == 'oracle',
-        "Oracle doesn't support contained_by lookup.",
-    )
+    @skipUnlessDBFeature('supports_json_field_contains')
     def test_contained_by(self):
         qs = NullableJSONModel.objects.filter(value__contained_by={'a': 'b', 'c': 14, 'h': True})
         self.assertSequenceEqual(qs, self.objs[2:4])
 
-    @skipUnless(
-        connection.vendor == 'oracle',
-        "Oracle doesn't support contained_by lookup.",
-    )
+    @skipIfDBFeature('supports_json_field_contains')
     def test_contained_by_unsupported(self):
-        msg = 'contained_by lookup is not supported on Oracle.'
+        msg = 'contained_by lookup is not supported on this database backend.'
         with self.assertRaisesMessage(NotSupportedError, msg):
             NullableJSONModel.objects.filter(value__contained_by={'a': 'b'}).get()
 
@@ -679,19 +676,25 @@ def test_lookups_with_key_transform(self):
             ('value__baz__has_any_keys', ['a', 'x']),
             ('value__has_key', KeyTextTransform('foo', 'value')),
         )
-        # contained_by and contains lookups are not supported on Oracle.
-        if connection.vendor != 'oracle':
-            tests += (
-                ('value__contains', KeyTransform('bax', 'value')),
-                ('value__baz__contained_by', {'a': 'b', 'c': 'd', 'e': 'f'}),
-                (
-                    'value__contained_by',
-                    KeyTransform('x', RawSQL(
-                        self.raw_sql,
-                        ['{"x": {"a": "b", "c": 1, "d": "e"}}'],
-                    )),
-                ),
-            )
+        for lookup, value in tests:
+            with self.subTest(lookup=lookup):
+                self.assertIs(NullableJSONModel.objects.filter(
+                    **{lookup: value},
+                ).exists(), True)
+
+    @skipUnlessDBFeature('supports_json_field_contains')
+    def test_contains_contained_by_with_key_transform(self):
+        tests = [
+            ('value__contains', KeyTransform('bax', 'value')),
+            ('value__baz__contained_by', {'a': 'b', 'c': 'd', 'e': 'f'}),
+            (
+                'value__contained_by',
+                KeyTransform('x', RawSQL(
+                    self.raw_sql,
+                    ['{"x": {"a": "b", "c": 1, "d": "e"}}'],
+                )),
+            ),
+        ]
         for lookup, value in tests:
             with self.subTest(lookup=lookup):
                 self.assertIs(NullableJSONModel.objects.filter(

```


## Code snippets

### 1 - django/db/models/fields/json.py:

Start line: 152, End line: 163

```python
class ContainedBy(PostgresOperatorLookup):
    lookup_name = 'contained_by'
    postgres_operator = '<@'

    def as_sql(self, compiler, connection):
        lhs, lhs_params = self.process_lhs(compiler, connection)
        rhs, rhs_params = self.process_rhs(compiler, connection)
        params = tuple(rhs_params) + tuple(lhs_params)
        return 'JSON_CONTAINS(%s, %s)' % (rhs, lhs), params

    def as_oracle(self, compiler, connection):
        raise NotSupportedError('contained_by lookup is not supported on Oracle.')
```
### 2 - django/db/models/fields/json.py:

Start line: 218, End line: 259

```python
class HasKey(HasKeyLookup):
    lookup_name = 'has_key'
    postgres_operator = '?'
    prepare_rhs = False


class HasKeys(HasKeyLookup):
    lookup_name = 'has_keys'
    postgres_operator = '?&'
    logical_operator = ' AND '

    def get_prep_lookup(self):
        return [str(item) for item in self.rhs]


class HasAnyKeys(HasKeys):
    lookup_name = 'has_any_keys'
    postgres_operator = '?|'
    logical_operator = ' OR '


class JSONExact(lookups.Exact):
    can_use_none_as_rhs = True

    def process_lhs(self, compiler, connection):
        lhs, lhs_params = super().process_lhs(compiler, connection)
        if connection.vendor == 'sqlite':
            rhs, rhs_params = super().process_rhs(compiler, connection)
            if rhs == '%s' and rhs_params == [None]:
                # Use JSON_TYPE instead of JSON_EXTRACT for NULLs.
                lhs = "JSON_TYPE(%s, '$')" % lhs
        return lhs, lhs_params

    def process_rhs(self, compiler, connection):
        rhs, rhs_params = super().process_rhs(compiler, connection)
        # Treat None lookup values as null.
        if rhs == '%s' and rhs_params == [None]:
            rhs_params = ['null']
        if connection.vendor == 'mysql':
            func = ["JSON_EXTRACT(%s, '$')"] * len(rhs_params)
            rhs = rhs % tuple(func)
        return rhs, rhs_params
```
### 3 - django/db/models/fields/json.py:

Start line: 1, End line: 40

```python
import json

from django import forms
from django.core import checks, exceptions
from django.db import NotSupportedError, connections, router
from django.db.models import lookups
from django.db.models.lookups import PostgresOperatorLookup, Transform
from django.utils.translation import gettext_lazy as _

from . import Field
from .mixins import CheckFieldDefaultMixin

__all__ = ['JSONField']


class JSONField(CheckFieldDefaultMixin, Field):
    empty_strings_allowed = False
    description = _('A JSON object')
    default_error_messages = {
        'invalid': _('Value must be valid JSON.'),
    }
    _default_hint = ('dict', '{}')

    def __init__(
        self, verbose_name=None, name=None, encoder=None, decoder=None,
        **kwargs,
    ):
        if encoder and not callable(encoder):
            raise ValueError('The encoder parameter must be a callable object.')
        if decoder and not callable(decoder):
            raise ValueError('The decoder parameter must be a callable object.')
        self.encoder = encoder
        self.decoder = decoder
        super().__init__(verbose_name, name, **kwargs)

    def check(self, **kwargs):
        errors = super().check(**kwargs)
        databases = kwargs.get('databases') or []
        errors.extend(self._check_supported(databases))
        return errors
```
### 4 - django/db/models/fields/json.py:

Start line: 62, End line: 122

```python
class JSONField(CheckFieldDefaultMixin, Field):

    def deconstruct(self):
        name, path, args, kwargs = super().deconstruct()
        if self.encoder is not None:
            kwargs['encoder'] = self.encoder
        if self.decoder is not None:
            kwargs['decoder'] = self.decoder
        return name, path, args, kwargs

    def from_db_value(self, value, expression, connection):
        if value is None:
            return value
        if connection.features.has_native_json_field and self.decoder is None:
            return value
        try:
            return json.loads(value, cls=self.decoder)
        except json.JSONDecodeError:
            return value

    def get_internal_type(self):
        return 'JSONField'

    def get_prep_value(self, value):
        if value is None:
            return value
        return json.dumps(value, cls=self.encoder)

    def get_transform(self, name):
        transform = super().get_transform(name)
        if transform:
            return transform
        return KeyTransformFactory(name)

    def select_format(self, compiler, sql, params):
        if (
            compiler.connection.features.has_native_json_field and
            self.decoder is not None
        ):
            return compiler.connection.ops.json_cast_text_sql(sql), params
        return super().select_format(compiler, sql, params)

    def validate(self, value, model_instance):
        super().validate(value, model_instance)
        try:
            json.dumps(value, cls=self.encoder)
        except TypeError:
            raise exceptions.ValidationError(
                self.error_messages['invalid'],
                code='invalid',
                params={'value': value},
            )

    def value_to_string(self, obj):
        return self.value_from_object(obj)

    def formfield(self, **kwargs):
        return super().formfield(**{
            'form_class': forms.JSONField,
            'encoder': self.encoder,
            'decoder': self.decoder,
            **kwargs,
        })
```
### 5 - django/db/models/fields/json.py:

Start line: 125, End line: 149

```python
def compile_json_path(key_transforms, include_root=True):
    path = ['$'] if include_root else []
    for key_transform in key_transforms:
        try:
            num = int(key_transform)
        except ValueError:  # non-integer
            path.append('.')
            path.append(json.dumps(key_transform))
        else:
            path.append('[%s]' % num)
    return ''.join(path)


class DataContains(PostgresOperatorLookup):
    lookup_name = 'contains'
    postgres_operator = '@>'

    def as_sql(self, compiler, connection):
        lhs, lhs_params = self.process_lhs(compiler, connection)
        rhs, rhs_params = self.process_rhs(compiler, connection)
        params = tuple(lhs_params) + tuple(rhs_params)
        return 'JSON_CONTAINS(%s, %s)' % (lhs, rhs), params

    def as_oracle(self, compiler, connection):
        raise NotSupportedError('contains lookup is not supported on Oracle.')
```
### 6 - django/db/models/fields/json.py:

Start line: 262, End line: 290

```python
JSONField.register_lookup(DataContains)
JSONField.register_lookup(ContainedBy)
JSONField.register_lookup(HasKey)
JSONField.register_lookup(HasKeys)
JSONField.register_lookup(HasAnyKeys)
JSONField.register_lookup(JSONExact)


class KeyTransform(Transform):
    postgres_operator = '->'
    postgres_nested_operator = '#>'

    def __init__(self, key_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.key_name = str(key_name)

    def preprocess_lhs(self, compiler, connection, lhs_only=False):
        if not lhs_only:
            key_transforms = [self.key_name]
        previous = self.lhs
        while isinstance(previous, KeyTransform):
            if not lhs_only:
                key_transforms.insert(0, previous.key_name)
            previous = previous.lhs
        lhs, params = compiler.compile(previous)
        if connection.vendor == 'oracle':
            # Escape string-formatting.
            key_transforms = [key.replace('%', '%%') for key in key_transforms]
        return (lhs, params, key_transforms) if not lhs_only else (lhs, params)
```
### 7 - django/forms/fields.py:

Start line: 1219, End line: 1272

```python
class JSONField(CharField):
    default_error_messages = {
        'invalid': _('Enter a valid JSON.'),
    }
    widget = Textarea

    def __init__(self, encoder=None, decoder=None, **kwargs):
        self.encoder = encoder
        self.decoder = decoder
        super().__init__(**kwargs)

    def to_python(self, value):
        if self.disabled:
            return value
        if value in self.empty_values:
            return None
        elif isinstance(value, (list, dict, int, float, JSONString)):
            return value
        try:
            converted = json.loads(value, cls=self.decoder)
        except json.JSONDecodeError:
            raise ValidationError(
                self.error_messages['invalid'],
                code='invalid',
                params={'value': value},
            )
        if isinstance(converted, str):
            return JSONString(converted)
        else:
            return converted

    def bound_data(self, data, initial):
        if self.disabled:
            return initial
        try:
            return json.loads(data, cls=self.decoder)
        except json.JSONDecodeError:
            return InvalidJSONInput(data)

    def prepare_value(self, value):
        if isinstance(value, InvalidJSONInput):
            return value
        return json.dumps(value, cls=self.encoder)

    def has_changed(self, initial, data):
        if super().has_changed(initial, data):
            return True
        # For purposes of seeing whether something has changed, True isn't the
        # same as 1 and the order of keys doesn't matter.
        return (
            json.dumps(initial, sort_keys=True, cls=self.encoder) !=
            json.dumps(self.to_python(data), sort_keys=True, cls=self.encoder)
        )
```
### 8 - django/db/models/fields/json.py:

Start line: 197, End line: 215

```python
class HasKeyLookup(PostgresOperatorLookup):

    def as_mysql(self, compiler, connection):
        return self.as_sql(compiler, connection, template="JSON_CONTAINS_PATH(%s, 'one', %%s)")

    def as_oracle(self, compiler, connection):
        sql, params = self.as_sql(compiler, connection, template="JSON_EXISTS(%s, '%%s')")
        # Add paths directly into SQL because path expressions cannot be passed
        # as bind variables on Oracle.
        return sql % tuple(params), []

    def as_postgresql(self, compiler, connection):
        if isinstance(self.rhs, KeyTransform):
            *_, rhs_key_transforms = self.rhs.preprocess_lhs(compiler, connection)
            for key in rhs_key_transforms[:-1]:
                self.lhs = KeyTransform(key, self.lhs)
            self.rhs = rhs_key_transforms[-1]
        return super().as_postgresql(compiler, connection)

    def as_sqlite(self, compiler, connection):
        return self.as_sql(compiler, connection, template='JSON_TYPE(%s, %%s) IS NOT NULL')
```
### 9 - django/db/models/fields/json.py:

Start line: 42, End line: 60

```python
class JSONField(CheckFieldDefaultMixin, Field):

    def _check_supported(self, databases):
        errors = []
        for db in databases:
            if not router.allow_migrate_model(db, self.model):
                continue
            connection = connections[db]
            if not (
                'supports_json_field' in self.model._meta.required_db_features or
                connection.features.supports_json_field
            ):
                errors.append(
                    checks.Error(
                        '%s does not support JSONFields.'
                        % connection.display_name,
                        obj=self.model,
                        id='fields.E180',
                    )
                )
        return errors
```
### 10 - django/db/models/fields/json.py:

Start line: 366, End line: 376

```python
class KeyTransformIsNull(lookups.IsNull):
    # key__isnull=False is the same as has_key='key'
    def as_oracle(self, compiler, connection):
        if not self.rhs:
            return HasKey(self.lhs.lhs, self.lhs.key_name).as_oracle(compiler, connection)
        return super().as_sql(compiler, connection)

    def as_sqlite(self, compiler, connection):
        if not self.rhs:
            return HasKey(self.lhs.lhs, self.lhs.key_name).as_sqlite(compiler, connection)
        return super().as_sql(compiler, connection)
```
### 11 - django/db/models/fields/json.py:

Start line: 166, End line: 195

```python
class HasKeyLookup(PostgresOperatorLookup):
    logical_operator = None

    def as_sql(self, compiler, connection, template=None):
        # Process JSON path from the left-hand side.
        if isinstance(self.lhs, KeyTransform):
            lhs, lhs_params, lhs_key_transforms = self.lhs.preprocess_lhs(compiler, connection)
            lhs_json_path = compile_json_path(lhs_key_transforms)
        else:
            lhs, lhs_params = self.process_lhs(compiler, connection)
            lhs_json_path = '$'
        sql = template % lhs
        # Process JSON path from the right-hand side.
        rhs = self.rhs
        rhs_params = []
        if not isinstance(rhs, (list, tuple)):
            rhs = [rhs]
        for key in rhs:
            if isinstance(key, KeyTransform):
                *_, rhs_key_transforms = key.preprocess_lhs(compiler, connection)
            else:
                rhs_key_transforms = [key]
            rhs_params.append('%s%s' % (
                lhs_json_path,
                compile_json_path(rhs_key_transforms, include_root=False),
            ))
        # Add condition for each key.
        if self.logical_operator:
            sql = '(%s)' % self.logical_operator.join([sql] * len(rhs_params))
        return sql, tuple(lhs_params) + tuple(rhs_params)
```
### 13 - django/db/models/fields/json.py:

Start line: 379, End line: 407

```python
class KeyTransformExact(JSONExact):
    def process_lhs(self, compiler, connection):
        lhs, lhs_params = super().process_lhs(compiler, connection)
        if connection.vendor == 'sqlite':
            rhs, rhs_params = super().process_rhs(compiler, connection)
            if rhs == '%s' and rhs_params == ['null']:
                lhs, _ = self.lhs.preprocess_lhs(compiler, connection, lhs_only=True)
                lhs = 'JSON_TYPE(%s, %%s)' % lhs
        return lhs, lhs_params

    def process_rhs(self, compiler, connection):
        if isinstance(self.rhs, KeyTransform):
            return super(lookups.Exact, self).process_rhs(compiler, connection)
        rhs, rhs_params = super().process_rhs(compiler, connection)
        if connection.vendor == 'oracle':
            func = []
            for value in rhs_params:
                value = json.loads(value)
                function = 'JSON_QUERY' if isinstance(value, (list, dict)) else 'JSON_VALUE'
                func.append("%s('%s', '$.value')" % (
                    function,
                    json.dumps({'value': value}),
                ))
            rhs = rhs % tuple(func)
            rhs_params = []
        elif connection.vendor == 'sqlite':
            func = ["JSON_EXTRACT(%s, '$')" if value != 'null' else '%s' for value in rhs_params]
            rhs = rhs % tuple(func)
        return rhs, rhs_params
```
### 16 - django/db/models/fields/json.py:

Start line: 409, End line: 421

```python
class KeyTransformExact(JSONExact):

    def as_oracle(self, compiler, connection):
        rhs, rhs_params = super().process_rhs(compiler, connection)
        if rhs_params == ['null']:
            # Field has key and it's NULL.
            has_key_expr = HasKey(self.lhs.lhs, self.lhs.key_name)
            has_key_sql, has_key_params = has_key_expr.as_oracle(compiler, connection)
            is_null_expr = self.lhs.get_lookup('isnull')(self.lhs, True)
            is_null_sql, is_null_params = is_null_expr.as_sql(compiler, connection)
            return (
                '%s AND %s' % (has_key_sql, is_null_sql),
                tuple(has_key_params) + tuple(is_null_params),
            )
        return super().as_sql(compiler, connection)
```
### 17 - django/db/backends/sqlite3/features.py:

Start line: 1, End line: 75

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
        try:
            with self.connection.cursor() as cursor, transaction.atomic():
                cursor.execute('SELECT JSON(\'{"a": "b"}\')')
        except OperationalError:
            return False
        return True

    can_introspect_json_field = property(operator.attrgetter('supports_json_field'))
```
### 29 - django/db/backends/base/features.py:

Start line: 219, End line: 305

```python
class BaseDatabaseFeatures:

    # Place FOR UPDATE right after FROM clause. Used on MSSQL.
    for_update_after_from = False

    # Combinatorial flags
    supports_select_union = True
    supports_select_intersection = True
    supports_select_difference = True
    supports_slicing_ordering_in_compound = False
    supports_parentheses_in_compound = True

    # Does the database support SQL 2003 FILTER (WHERE ...) in aggregate
    # expressions?
    supports_aggregate_filter_clause = False

    # Does the backend support indexing a TextField?
    supports_index_on_text_field = True

    # Does the backend support window expressions (expression OVER (...))?
    supports_over_clause = False
    supports_frame_range_fixed_distance = False
    only_supports_unbounded_with_preceding_and_following = False

    # Does the backend support CAST with precision?
    supports_cast_with_precision = True

    # How many second decimals does the database return when casting a value to
    # a type with time?
    time_cast_precision = 6

    # SQL to create a procedure for use by the Django test suite. The
    # functionality of the procedure isn't important.
    create_test_procedure_without_params_sql = None
    create_test_procedure_with_int_param_sql = None

    # Does the backend support keyword parameters for cursor.callproc()?
    supports_callproc_kwargs = False

    # What formats does the backend EXPLAIN syntax support?
    supported_explain_formats = set()

    # Does DatabaseOperations.explain_query_prefix() raise ValueError if
    # unknown kwargs are passed to QuerySet.explain()?
    validates_explain_options = True

    # Does the backend support the default parameter in lead() and lag()?
    supports_default_in_lead_lag = True

    # Does the backend support ignoring constraint or uniqueness errors during
    # INSERT?
    supports_ignore_conflicts = True

    # Does this backend require casting the results of CASE expressions used
    # in UPDATE statements to ensure the expression has the correct type?
    requires_casted_case_in_updates = False

    # Does the backend support partial indexes (CREATE INDEX ... WHERE ...)?
    supports_partial_indexes = True
    supports_functions_in_partial_indexes = True
    # Does the backend support covering indexes (CREATE INDEX ... INCLUDE ...)?
    supports_covering_indexes = False

    # Does the database allow more than one constraint or index on the same
    # field(s)?
    allows_multiple_constraints_on_same_fields = True

    # Does the backend support boolean expressions in SELECT and GROUP BY
    # clauses?
    supports_boolean_expr_in_select_clause = True

    # Does the backend support JSONField?
    supports_json_field = True
    # Can the backend introspect a JSONField?
    can_introspect_json_field = True
    # Does the backend support primitives in JSONField?
    supports_primitives_in_json_field = True
    # Is there a true datatype for JSON?
    has_native_json_field = False
    # Does the backend use PostgreSQL-style JSON operators like '->'?
    has_json_operators = False

    def __init__(self, connection):
        self.connection = connection

    @cached_property
    def supports_explaining_query_execution(self):
        """Does this backend support explaining query execution?"""
        return self.connection.ops.explain_prefix is not None
```
### 30 - django/db/models/fields/json.py:

Start line: 305, End line: 318

```python
class KeyTransform(Transform):

    def as_postgresql(self, compiler, connection):
        lhs, params, key_transforms = self.preprocess_lhs(compiler, connection)
        if len(key_transforms) > 1:
            return '(%s %s %%s)' % (lhs, self.postgres_nested_operator), params + [key_transforms]
        try:
            lookup = int(self.key_name)
        except ValueError:
            lookup = self.key_name
        return '(%s %s %%s)' % (lhs, self.postgres_operator), tuple(params) + (lookup,)

    def as_sqlite(self, compiler, connection):
        lhs, params, key_transforms = self.preprocess_lhs(compiler, connection)
        json_path = compile_json_path(key_transforms)
        return 'JSON_EXTRACT(%s, %%s)' % lhs, tuple(params) + (json_path,)
```
### 34 - django/db/models/fields/json.py:

Start line: 424, End line: 509

```python
class KeyTransformIExact(CaseInsensitiveMixin, KeyTransformTextLookupMixin, lookups.IExact):
    pass


class KeyTransformIContains(CaseInsensitiveMixin, KeyTransformTextLookupMixin, lookups.IContains):
    pass


class KeyTransformContains(KeyTransformTextLookupMixin, lookups.Contains):
    pass


class KeyTransformStartsWith(KeyTransformTextLookupMixin, lookups.StartsWith):
    pass


class KeyTransformIStartsWith(CaseInsensitiveMixin, KeyTransformTextLookupMixin, lookups.IStartsWith):
    pass


class KeyTransformEndsWith(KeyTransformTextLookupMixin, lookups.EndsWith):
    pass


class KeyTransformIEndsWith(CaseInsensitiveMixin, KeyTransformTextLookupMixin, lookups.IEndsWith):
    pass


class KeyTransformRegex(KeyTransformTextLookupMixin, lookups.Regex):
    pass


class KeyTransformIRegex(CaseInsensitiveMixin, KeyTransformTextLookupMixin, lookups.IRegex):
    pass


class KeyTransformNumericLookupMixin:
    def process_rhs(self, compiler, connection):
        rhs, rhs_params = super().process_rhs(compiler, connection)
        if not connection.features.has_native_json_field:
            rhs_params = [json.loads(value) for value in rhs_params]
        return rhs, rhs_params


class KeyTransformLt(KeyTransformNumericLookupMixin, lookups.LessThan):
    pass


class KeyTransformLte(KeyTransformNumericLookupMixin, lookups.LessThanOrEqual):
    pass


class KeyTransformGt(KeyTransformNumericLookupMixin, lookups.GreaterThan):
    pass


class KeyTransformGte(KeyTransformNumericLookupMixin, lookups.GreaterThanOrEqual):
    pass


KeyTransform.register_lookup(KeyTransformExact)
KeyTransform.register_lookup(KeyTransformIExact)
KeyTransform.register_lookup(KeyTransformIsNull)
KeyTransform.register_lookup(KeyTransformContains)
KeyTransform.register_lookup(KeyTransformIContains)
KeyTransform.register_lookup(KeyTransformStartsWith)
KeyTransform.register_lookup(KeyTransformIStartsWith)
KeyTransform.register_lookup(KeyTransformEndsWith)
KeyTransform.register_lookup(KeyTransformIEndsWith)
KeyTransform.register_lookup(KeyTransformRegex)
KeyTransform.register_lookup(KeyTransformIRegex)

KeyTransform.register_lookup(KeyTransformLt)
KeyTransform.register_lookup(KeyTransformLte)
KeyTransform.register_lookup(KeyTransformGt)
KeyTransform.register_lookup(KeyTransformGte)


class KeyTransformFactory:

    def __init__(self, key_name):
        self.key_name = key_name

    def __call__(self, *args, **kwargs):
        return KeyTransform(self.key_name, *args, **kwargs)
```
### 39 - django/db/models/fields/json.py:

Start line: 321, End line: 343

```python
class KeyTextTransform(KeyTransform):
    postgres_operator = '->>'
    postgres_nested_operator = '#>>'


class KeyTransformTextLookupMixin:
    """
    Mixin for combining with a lookup expecting a text lhs from a JSONField
    key lookup. On PostgreSQL, make use of the ->> operator instead of casting
    key values to text and performing the lookup on the resulting
    representation.
    """
    def __init__(self, key_transform, *args, **kwargs):
        if not isinstance(key_transform, KeyTransform):
            raise TypeError(
                'Transform should be an instance of KeyTransform in order to '
                'use this lookup.'
            )
        key_text_transform = KeyTextTransform(
            key_transform.key_name, *key_transform.source_expressions,
            **key_transform.extra,
        )
        super().__init__(key_text_transform, *args, **kwargs)
```
### 42 - django/db/models/fields/json.py:

Start line: 346, End line: 363

```python
class CaseInsensitiveMixin:
    """
    Mixin to allow case-insensitive comparison of JSON values on MySQL.
    MySQL handles strings used in JSON context using the utf8mb4_bin collation.
    Because utf8mb4_bin is a binary collation, comparison of JSON values is
    case-sensitive.
    """
    def process_lhs(self, compiler, connection):
        lhs, lhs_params = super().process_lhs(compiler, connection)
        if connection.vendor == 'mysql':
            return 'LOWER(%s)' % lhs, lhs_params
        return lhs, lhs_params

    def process_rhs(self, compiler, connection):
        rhs, rhs_params = super().process_rhs(compiler, connection)
        if connection.vendor == 'mysql':
            return 'LOWER(%s)' % rhs, rhs_params
        return rhs, rhs_params
```
### 45 - django/db/models/fields/json.py:

Start line: 292, End line: 303

```python
class KeyTransform(Transform):

    def as_mysql(self, compiler, connection):
        lhs, params, key_transforms = self.preprocess_lhs(compiler, connection)
        json_path = compile_json_path(key_transforms)
        return 'JSON_EXTRACT(%s, %%s)' % lhs, tuple(params) + (json_path,)

    def as_oracle(self, compiler, connection):
        lhs, params, key_transforms = self.preprocess_lhs(compiler, connection)
        json_path = compile_json_path(key_transforms)
        return (
            "COALESCE(JSON_QUERY(%s, '%s'), JSON_VALUE(%s, '%s'))" %
            ((lhs, json_path) * 2)
        ), tuple(params) * 2
```
### 61 - django/db/backends/oracle/features.py:

Start line: 1, End line: 75

```python
from django.db import InterfaceError
from django.db.backends.base.features import BaseDatabaseFeatures
from django.utils.functional import cached_property


class DatabaseFeatures(BaseDatabaseFeatures):
    interprets_empty_strings_as_nulls = True
    has_select_for_update = True
    has_select_for_update_nowait = True
    has_select_for_update_skip_locked = True
    has_select_for_update_of = True
    select_for_update_of_column = True
    can_return_columns_from_insert = True
    supports_subqueries_in_group_by = False
    supports_transactions = True
    supports_timezones = False
    has_native_duration_field = True
    can_defer_constraint_checks = True
    supports_partially_nullable_unique_constraints = False
    supports_deferrable_unique_constraints = True
    truncates_names = True
    supports_tablespaces = True
    supports_sequence_reset = False
    can_introspect_materialized_views = True
    atomic_transactions = False
    supports_combined_alters = False
    nulls_order_largest = True
    requires_literal_defaults = True
    closed_cursor_error_class = InterfaceError
    bare_select_suffix = " FROM DUAL"
    # select for update with limit can be achieved on Oracle, but not with the current backend.
    supports_select_for_update_with_limit = False
    supports_temporal_subtraction = True
    # Oracle doesn't ignore quoted identifiers case but the current backend
    # does by uppercasing all identifiers.
    ignores_table_name_case = True
    supports_index_on_text_field = False
    has_case_insensitive_like = False
    create_test_procedure_without_params_sql = """
        CREATE PROCEDURE "TEST_PROCEDURE" AS
            V_I INTEGER;
        BEGIN
            V_I := 1;
        END;
    """
    create_test_procedure_with_int_param_sql = """
        CREATE PROCEDURE "TEST_PROCEDURE" (P_I INTEGER) AS
            V_I INTEGER;
        BEGIN
            V_I := P_I;
        END;
    """
    supports_callproc_kwargs = True
    supports_over_clause = True
    supports_frame_range_fixed_distance = True
    supports_ignore_conflicts = False
    max_query_params = 2**16 - 1
    supports_partial_indexes = False
    supports_slicing_ordering_in_compound = True
    allows_multiple_constraints_on_same_fields = False
    supports_boolean_expr_in_select_clause = False
    supports_primitives_in_json_field = False

    @cached_property
    def introspected_field_types(self):
        return {
            **super().introspected_field_types,
            'GenericIPAddressField': 'CharField',
            'PositiveBigIntegerField': 'BigIntegerField',
            'PositiveIntegerField': 'IntegerField',
            'PositiveSmallIntegerField': 'IntegerField',
            'SmallIntegerField': 'IntegerField',
            'TimeField': 'DateTimeField',
        }
```
### 67 - django/db/backends/sqlite3/base.py:

Start line: 582, End line: 612

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


@none_guard
def _sqlite_json_contains(haystack, needle):
    target, candidate = json.loads(haystack), json.loads(needle)
    if isinstance(target, dict) and isinstance(candidate, dict):
        return target.items() >= candidate.items()
    return target == candidate
```
### 93 - django/db/backends/base/features.py:

Start line: 1, End line: 113

```python
from django.db import ProgrammingError
from django.utils.functional import cached_property


class BaseDatabaseFeatures:
    gis_enabled = False
    allows_group_by_pk = False
    allows_group_by_selected_pks = False
    empty_fetchmany_value = []
    update_can_self_select = True

    # Does the backend distinguish between '' and None?
    interprets_empty_strings_as_nulls = False

    # Does the backend allow inserting duplicate NULL rows in a nullable
    # unique field? All core backends implement this correctly, but other
    # databases such as SQL Server do not.
    supports_nullable_unique_constraints = True

    # Does the backend allow inserting duplicate rows when a unique_together
    # constraint exists and some fields are nullable but not all of them?
    supports_partially_nullable_unique_constraints = True
    # Does the backend support initially deferrable unique constraints?
    supports_deferrable_unique_constraints = False

    can_use_chunked_reads = True
    can_return_columns_from_insert = False
    can_return_rows_from_bulk_insert = False
    has_bulk_insert = True
    uses_savepoints = True
    can_release_savepoints = False

    # If True, don't use integer foreign keys referring to, e.g., positive
    # integer primary keys.
    related_fields_match_type = False
    allow_sliced_subqueries_with_in = True
    has_select_for_update = False
    has_select_for_update_nowait = False
    has_select_for_update_skip_locked = False
    has_select_for_update_of = False
    has_select_for_no_key_update = False
    # Does the database's SELECT FOR UPDATE OF syntax require a column rather
    # than a table?
    select_for_update_of_column = False

    # Does the default test database allow multiple connections?
    # Usually an indication that the test database is in-memory
    test_db_allows_multiple_connections = True

    # Can an object be saved without an explicit primary key?
    supports_unspecified_pk = False

    # Can a fixture contain forward references? i.e., are
    # FK constraints checked at the end of transaction, or
    # at the end of each save operation?
    supports_forward_references = True

    # Does the backend truncate names properly when they are too long?
    truncates_names = False

    # Is there a REAL datatype in addition to floats/doubles?
    has_real_datatype = False
    supports_subqueries_in_group_by = True

    # Is there a true datatype for uuid?
    has_native_uuid_field = False

    # Is there a true datatype for timedeltas?
    has_native_duration_field = False

    # Does the database driver supports same type temporal data subtraction
    # by returning the type used to store duration field?
    supports_temporal_subtraction = False

    # Does the __regex lookup support backreferencing and grouping?
    supports_regex_backreferencing = True

    # Can date/datetime lookups be performed using a string?
    supports_date_lookup_using_string = True

    # Can datetimes with timezones be used?
    supports_timezones = True

    # Does the database have a copy of the zoneinfo database?
    has_zoneinfo_database = True

    # When performing a GROUP BY, is an ORDER BY NULL required
    # to remove any ordering?
    requires_explicit_null_ordering_when_grouping = False

    # Does the backend order NULL values as largest or smallest?
    nulls_order_largest = False

    # Does the backend support NULLS FIRST and NULLS LAST in ORDER BY?
    supports_order_by_nulls_modifier = True

    # Does the backend orders NULLS FIRST by default?
    order_by_nulls_first = False

    # The database's limit on the number of query parameters.
    max_query_params = None

    # Can an object have an autoincrement primary key of 0?
    allows_auto_pk_0 = True

    # Do we need to NULL a ForeignKey out, or can the constraint check be
    # deferred
    can_defer_constraint_checks = False

    # date_interval_sql can properly handle mixed Date/DateTime fields and timedeltas
    supports_mixed_date_datetime_comparisons = True

    # Does the backend support tablespaces? Default to False because it isn't
```
### 102 - django/db/backends/sqlite3/base.py:

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
