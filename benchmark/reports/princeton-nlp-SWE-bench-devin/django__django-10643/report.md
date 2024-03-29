# django__django-10643

| **django/django** | `28e769dfe6a65bf604f5adc6a650ab47ba6b3bef` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | - |
| **Avg pos** | - |
| **Min pos** | - |
| **Max pos** | - |
| **Top file pos** | 12 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/db/models/lookups.py b/django/db/models/lookups.py
--- a/django/db/models/lookups.py
+++ b/django/db/models/lookups.py
@@ -5,7 +5,7 @@
 from django.core.exceptions import EmptyResultSet
 from django.db.models.expressions import Case, Exists, Func, Value, When
 from django.db.models.fields import (
-    BooleanField, DateTimeField, Field, IntegerField,
+    BooleanField, CharField, DateTimeField, Field, IntegerField, UUIDField,
 )
 from django.db.models.query_utils import RegisterLookupMixin
 from django.utils.datastructures import OrderedSet
@@ -548,3 +548,53 @@ def get_bound_params(self, start, finish):
 class YearLte(YearLookup, LessThanOrEqual):
     def get_bound_params(self, start, finish):
         return (finish,)
+
+
+class UUIDTextMixin:
+    """
+    Strip hyphens from a value when filtering a UUIDField on backends without
+    a native datatype for UUID.
+    """
+    def process_rhs(self, qn, connection):
+        if not connection.features.has_native_uuid_field:
+            from django.db.models.functions import Replace
+            if self.rhs_is_direct_value():
+                self.rhs = Value(self.rhs)
+            self.rhs = Replace(self.rhs, Value('-'), Value(''), output_field=CharField())
+        rhs, params = super().process_rhs(qn, connection)
+        return rhs, params
+
+
+@UUIDField.register_lookup
+class UUIDIExact(UUIDTextMixin, IExact):
+    pass
+
+
+@UUIDField.register_lookup
+class UUIDContains(UUIDTextMixin, Contains):
+    pass
+
+
+@UUIDField.register_lookup
+class UUIDIContains(UUIDTextMixin, IContains):
+    pass
+
+
+@UUIDField.register_lookup
+class UUIDStartsWith(UUIDTextMixin, StartsWith):
+    pass
+
+
+@UUIDField.register_lookup
+class UUIDIStartsWith(UUIDTextMixin, IStartsWith):
+    pass
+
+
+@UUIDField.register_lookup
+class UUIDEndsWith(UUIDTextMixin, EndsWith):
+    pass
+
+
+@UUIDField.register_lookup
+class UUIDIEndsWith(UUIDTextMixin, IEndsWith):
+    pass

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/db/models/lookups.py | 8 | 8 | - | 12 | -
| django/db/models/lookups.py | 551 | 551 | - | 12 | -


## Problem Statement

```
Allow icontains lookup to accept uuids with or without dashes
Description
	
We have Django 2.1 project with model admin which includes an UUIDField in list_display and search_fields. The UUID is displayed with dashes on changelist (e.g. "245ba2eb-6852-47be-82be-7dc07327cf9e") and if the user cut'n'paste it to the search field, I would expect admin to find it.
This works however only on Postgres but fails on Oracle. I can understand why this happens (Oracle backend stores uuid as string) and I believe I can workaround it by customizing get_search_results but I think should be internal thing that Django handles gracefully - search should be possible by the value as displayed in admin.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 django/db/models/fields/__init__.py | 2268 | 2318| 339 | 339 | 17457 | 
| 2 | 2 django/contrib/admin/options.py | 1008 | 1019| 149 | 488 | 35823 | 
| 3 | 3 django/db/backends/mysql/operations.py | 1 | 33| 254 | 742 | 38950 | 
| 4 | 3 django/contrib/admin/options.py | 973 | 1006| 310 | 1052 | 38950 | 
| 5 | 4 django/db/backends/oracle/base.py | 389 | 411| 200 | 1252 | 44039 | 
| 6 | 5 django/db/backends/oracle/operations.py | 1 | 17| 118 | 1370 | 49824 | 
| 7 | 6 django/contrib/admin/views/main.py | 1 | 45| 319 | 1689 | 54031 | 
| 8 | 7 django/contrib/postgres/lookups.py | 1 | 77| 469 | 2158 | 54501 | 
| 9 | 8 django/forms/fields.py | 1173 | 1203| 182 | 2340 | 63445 | 
| 10 | 8 django/db/backends/oracle/operations.py | 294 | 319| 243 | 2583 | 63445 | 
| 11 | 8 django/contrib/admin/options.py | 368 | 420| 504 | 3087 | 63445 | 
| 12 | 9 django/db/backends/oracle/introspection.py | 108 | 141| 286 | 3373 | 65804 | 
| 13 | 9 django/db/backends/oracle/operations.py | 321 | 332| 227 | 3600 | 65804 | 
| 14 | 9 django/db/backends/oracle/operations.py | 20 | 71| 556 | 4156 | 65804 | 
| 15 | 10 django/db/backends/oracle/features.py | 1 | 62| 504 | 4660 | 66309 | 
| 16 | 10 django/db/backends/oracle/operations.py | 456 | 488| 367 | 5027 | 66309 | 
| 17 | 10 django/db/backends/oracle/operations.py | 353 | 390| 369 | 5396 | 66309 | 
| 18 | 11 django/contrib/postgres/search.py | 131 | 145| 142 | 5538 | 68247 | 
| 19 | 11 django/db/backends/oracle/operations.py | 568 | 583| 221 | 5759 | 68247 | 
| 20 | **12 django/db/models/lookups.py** | 117 | 140| 220 | 5979 | 72539 | 
| 21 | 12 django/db/backends/oracle/operations.py | 201 | 249| 411 | 6390 | 72539 | 
| 22 | 12 django/db/backends/oracle/operations.py | 263 | 277| 198 | 6588 | 72539 | 
| 23 | 12 django/db/backends/oracle/base.py | 479 | 508| 427 | 7015 | 72539 | 
| 24 | 12 django/contrib/postgres/search.py | 147 | 154| 121 | 7136 | 72539 | 
| 25 | 13 django/contrib/admin/widgets.py | 352 | 378| 328 | 7464 | 76405 | 
| 26 | 13 django/contrib/admin/options.py | 1 | 96| 769 | 8233 | 76405 | 
| 27 | **13 django/db/models/lookups.py** | 420 | 470| 275 | 8508 | 76405 | 
| 28 | 14 django/contrib/postgres/fields/ranges.py | 195 | 218| 217 | 8725 | 78285 | 
| 29 | 14 django/db/backends/oracle/operations.py | 172 | 199| 315 | 9040 | 78285 | 
| 30 | 14 django/contrib/postgres/search.py | 156 | 188| 287 | 9327 | 78285 | 
| 31 | 14 django/contrib/postgres/search.py | 101 | 128| 245 | 9572 | 78285 | 
| 32 | 15 django/contrib/admin/checks.py | 929 | 958| 243 | 9815 | 87301 | 
| 33 | 15 django/contrib/postgres/search.py | 1 | 21| 201 | 10016 | 87301 | 
| 34 | 15 django/contrib/postgres/search.py | 24 | 44| 132 | 10148 | 87301 | 
| 35 | 15 django/db/backends/oracle/operations.py | 99 | 110| 210 | 10358 | 87301 | 
| 36 | 16 django/db/backends/oracle/schema.py | 125 | 173| 419 | 10777 | 89057 | 
| 37 | 16 django/contrib/admin/views/main.py | 326 | 378| 467 | 11244 | 89057 | 
| 38 | 17 django/db/backends/oracle/validation.py | 1 | 23| 146 | 11390 | 89204 | 
| 39 | 17 django/db/backends/oracle/operations.py | 334 | 351| 207 | 11597 | 89204 | 
| 40 | 17 django/db/backends/oracle/base.py | 60 | 91| 278 | 11875 | 89204 | 
| 41 | 17 django/db/backends/oracle/operations.py | 160 | 170| 206 | 12081 | 89204 | 
| 42 | 17 django/db/backends/oracle/base.py | 38 | 60| 235 | 12316 | 89204 | 
| 43 | 17 django/db/backends/oracle/operations.py | 88 | 97| 201 | 12517 | 89204 | 
| 44 | 18 django/contrib/gis/db/backends/oracle/models.py | 14 | 43| 217 | 12734 | 89688 | 
| 45 | 18 django/db/backends/oracle/operations.py | 392 | 438| 480 | 13214 | 89688 | 
| 46 | 18 django/db/backends/oracle/schema.py | 79 | 123| 583 | 13797 | 89688 | 
| 47 | 18 django/contrib/postgres/fields/ranges.py | 176 | 192| 173 | 13970 | 89688 | 
| 48 | 18 django/db/backends/oracle/operations.py | 534 | 550| 187 | 14157 | 89688 | 
| 49 | 18 django/contrib/postgres/search.py | 47 | 68| 227 | 14384 | 89688 | 
| 50 | 18 django/contrib/admin/widgets.py | 166 | 197| 243 | 14627 | 89688 | 
| 51 | 18 django/db/backends/oracle/operations.py | 73 | 86| 253 | 14880 | 89688 | 
| 52 | 18 django/db/backends/oracle/operations.py | 141 | 158| 309 | 15189 | 89688 | 
| 53 | 18 django/db/backends/oracle/introspection.py | 1 | 30| 245 | 15434 | 89688 | 
| 54 | 18 django/db/backends/oracle/base.py | 94 | 143| 578 | 16012 | 89688 | 
| 55 | 18 django/db/backends/oracle/operations.py | 128 | 139| 213 | 16225 | 89688 | 
| 56 | 18 django/db/backends/oracle/operations.py | 508 | 532| 242 | 16467 | 89688 | 
| 57 | 18 django/contrib/postgres/fields/ranges.py | 221 | 283| 348 | 16815 | 89688 | 
| 58 | 18 django/db/backends/oracle/operations.py | 552 | 566| 269 | 17084 | 89688 | 
| 59 | 19 django/contrib/gis/db/backends/oracle/introspection.py | 1 | 43| 379 | 17463 | 90068 | 
| 60 | 20 django/contrib/admin/views/autocomplete.py | 1 | 35| 246 | 17709 | 90460 | 
| 61 | 20 django/db/backends/oracle/introspection.py | 164 | 176| 158 | 17867 | 90460 | 
| 62 | 20 django/db/backends/oracle/schema.py | 57 | 77| 249 | 18116 | 90460 | 
| 63 | 21 django/db/models/functions/text.py | 227 | 243| 149 | 18265 | 92916 | 
| 64 | 21 django/db/backends/oracle/operations.py | 440 | 454| 203 | 18468 | 92916 | 
| 65 | 21 django/db/backends/oracle/base.py | 510 | 548| 318 | 18786 | 92916 | 
| 66 | 21 django/db/backends/oracle/introspection.py | 178 | 192| 136 | 18922 | 92916 | 
| 67 | 21 django/db/backends/oracle/operations.py | 251 | 261| 210 | 19132 | 92916 | 
| 68 | 21 django/contrib/postgres/fields/ranges.py | 110 | 157| 262 | 19394 | 92916 | 
| 69 | 21 django/db/backends/oracle/introspection.py | 32 | 49| 176 | 19570 | 92916 | 
| 70 | 21 django/db/backends/oracle/operations.py | 112 | 126| 225 | 19795 | 92916 | 
| 71 | 22 django/db/backends/oracle/client.py | 1 | 18| 0 | 19795 | 93020 | 
| 72 | **22 django/db/models/lookups.py** | 276 | 326| 306 | 20101 | 93020 | 
| 73 | 23 django/db/backends/postgresql/operations.py | 208 | 300| 756 | 20857 | 95751 | 
| 74 | 24 django/contrib/admin/templatetags/admin_list.py | 431 | 489| 343 | 21200 | 99580 | 
| 75 | 24 django/db/backends/postgresql/operations.py | 86 | 103| 202 | 21402 | 99580 | 
| 76 | 25 django/contrib/gis/db/backends/oracle/base.py | 1 | 17| 0 | 21402 | 99683 | 
| 77 | **25 django/db/models/lookups.py** | 257 | 273| 133 | 21535 | 99683 | 
| 78 | 25 django/db/backends/oracle/operations.py | 490 | 506| 190 | 21725 | 99683 | 
| 79 | 25 django/db/backends/oracle/operations.py | 606 | 629| 268 | 21993 | 99683 | 
| 80 | 26 django/contrib/gis/db/backends/oracle/features.py | 1 | 13| 0 | 21993 | 99772 | 
| 81 | 27 django/contrib/gis/db/backends/oracle/schema.py | 1 | 32| 326 | 22319 | 100605 | 
| 82 | 27 django/contrib/admin/checks.py | 706 | 716| 115 | 22434 | 100605 | 
| 83 | 27 django/db/backends/oracle/schema.py | 41 | 55| 133 | 22567 | 100605 | 
| 84 | 27 django/contrib/admin/views/autocomplete.py | 37 | 52| 154 | 22721 | 100605 | 
| 85 | **27 django/db/models/lookups.py** | 329 | 359| 263 | 22984 | 100605 | 
| 86 | 27 django/db/models/functions/text.py | 24 | 55| 217 | 23201 | 100605 | 
| 87 | 27 django/contrib/postgres/fields/ranges.py | 160 | 174| 136 | 23337 | 100605 | 
| 88 | 27 django/db/backends/oracle/operations.py | 585 | 604| 303 | 23640 | 100605 | 
| 89 | 28 django/contrib/admin/utils.py | 259 | 282| 174 | 23814 | 104678 | 
| 90 | 28 django/contrib/gis/db/backends/oracle/schema.py | 60 | 95| 297 | 24111 | 104678 | 
| 91 | 28 django/contrib/admin/checks.py | 160 | 201| 325 | 24436 | 104678 | 
| 92 | 28 django/contrib/admin/checks.py | 203 | 213| 127 | 24563 | 104678 | 
| 93 | 28 django/contrib/admin/utils.py | 52 | 100| 334 | 24897 | 104678 | 
| 94 | 28 django/contrib/admin/utils.py | 403 | 432| 203 | 25100 | 104678 | 
| 95 | 28 django/contrib/admin/options.py | 1623 | 1635| 167 | 25267 | 104678 | 
| 96 | 28 django/contrib/postgres/search.py | 70 | 98| 274 | 25541 | 104678 | 
| 97 | 28 django/db/backends/oracle/operations.py | 279 | 292| 240 | 25781 | 104678 | 
| 98 | 28 django/db/backends/oracle/base.py | 1 | 35| 242 | 26023 | 104678 | 
| 99 | 28 django/db/backends/oracle/schema.py | 1 | 39| 406 | 26429 | 104678 | 
| 100 | 29 django/db/backends/postgresql/base.py | 66 | 141| 772 | 27201 | 107438 | 
| 101 | 29 django/contrib/admin/checks.py | 148 | 158| 123 | 27324 | 107438 | 
| 102 | 30 django/db/models/fields/related_lookups.py | 62 | 99| 447 | 27771 | 108887 | 
| 103 | 31 django/contrib/admin/filters.py | 422 | 430| 107 | 27878 | 112621 | 
| 104 | 31 django/contrib/admin/options.py | 422 | 465| 350 | 28228 | 112621 | 
| 105 | 31 django/db/models/fields/related_lookups.py | 119 | 155| 244 | 28472 | 112621 | 
| 106 | 31 django/contrib/admin/utils.py | 285 | 303| 175 | 28647 | 112621 | 
| 107 | 31 django/contrib/admin/views/main.py | 465 | 496| 225 | 28872 | 112621 | 
| 108 | 31 django/db/backends/oracle/base.py | 446 | 477| 280 | 29152 | 112621 | 
| 109 | 31 django/contrib/gis/db/backends/oracle/schema.py | 34 | 58| 229 | 29381 | 112621 | 
| 110 | 31 django/contrib/gis/db/backends/oracle/models.py | 46 | 65| 158 | 29539 | 112621 | 
| 111 | 31 django/contrib/admin/options.py | 626 | 643| 136 | 29675 | 112621 | 
| 112 | 31 django/contrib/admin/views/main.py | 48 | 119| 636 | 30311 | 112621 | 
| 113 | 31 django/db/backends/oracle/introspection.py | 69 | 106| 364 | 30675 | 112621 | 
| 114 | 31 django/db/backends/oracle/introspection.py | 194 | 295| 842 | 31517 | 112621 | 
| 115 | 31 django/contrib/admin/templatetags/admin_list.py | 1 | 26| 170 | 31687 | 112621 | 
| 116 | 32 django/contrib/postgres/fields/utils.py | 1 | 4| 0 | 31687 | 112644 | 
| 117 | 32 django/contrib/admin/utils.py | 1 | 24| 218 | 31905 | 112644 | 
| 118 | 32 django/contrib/postgres/search.py | 191 | 240| 338 | 32243 | 112644 | 
| 119 | **32 django/db/models/lookups.py** | 361 | 390| 337 | 32580 | 112644 | 
| 120 | 32 django/contrib/admin/widgets.py | 97 | 122| 179 | 32759 | 112644 | 
| 121 | 33 django/contrib/postgres/constraints.py | 56 | 66| 123 | 32882 | 113501 | 
| 122 | 34 django/db/backends/sqlite3/operations.py | 293 | 335| 429 | 33311 | 116385 | 
| 123 | 34 django/db/models/fields/related_lookups.py | 102 | 117| 215 | 33526 | 116385 | 
| 124 | 35 django/contrib/gis/db/backends/oracle/operations.py | 52 | 115| 746 | 34272 | 118417 | 
| 125 | 35 django/contrib/admin/options.py | 933 | 971| 269 | 34541 | 118417 | 
| 126 | 36 django/contrib/postgres/indexes.py | 140 | 157| 131 | 34672 | 119814 | 
| 127 | 36 django/contrib/admin/options.py | 206 | 217| 135 | 34807 | 119814 | 
| 128 | 36 django/db/models/functions/text.py | 58 | 77| 153 | 34960 | 119814 | 
| 129 | 36 django/contrib/admin/filters.py | 1 | 17| 127 | 35087 | 119814 | 
| 130 | 36 django/contrib/postgres/indexes.py | 1 | 36| 270 | 35357 | 119814 | 
| 131 | 36 django/contrib/admin/options.py | 277 | 366| 641 | 35998 | 119814 | 
| 132 | 37 django/contrib/auth/admin.py | 25 | 37| 128 | 36126 | 121540 | 
| 133 | 38 django/contrib/gis/db/models/lookups.py | 303 | 325| 202 | 36328 | 124147 | 
| 134 | 38 django/db/backends/mysql/operations.py | 293 | 303| 120 | 36448 | 124147 | 
| 135 | 38 django/contrib/admin/options.py | 602 | 624| 280 | 36728 | 124147 | 
| 136 | 38 django/contrib/admin/checks.py | 879 | 927| 416 | 37144 | 124147 | 
| 137 | 39 django/db/backends/postgresql/introspection.py | 209 | 225| 179 | 37323 | 126388 | 
| 138 | 40 django/urls/converters.py | 1 | 67| 313 | 37636 | 126701 | 
| 139 | 41 django/contrib/postgres/fields/hstore.py | 73 | 113| 264 | 37900 | 127398 | 
| 140 | 41 django/contrib/admin/filters.py | 209 | 226| 190 | 38090 | 127398 | 
| 141 | 42 django/contrib/postgres/forms/ranges.py | 81 | 103| 149 | 38239 | 128075 | 
| 142 | 42 django/db/backends/oracle/base.py | 145 | 207| 780 | 39019 | 128075 | 
| 143 | 42 django/contrib/admin/views/main.py | 205 | 254| 423 | 39442 | 128075 | 
| 144 | 43 django/contrib/admin/models.py | 1 | 20| 118 | 39560 | 129200 | 
| 145 | 43 django/contrib/postgres/fields/hstore.py | 1 | 70| 432 | 39992 | 129200 | 
| 146 | 43 django/contrib/gis/db/models/lookups.py | 222 | 249| 134 | 40126 | 129200 | 
| 147 | 43 django/contrib/admin/checks.py | 768 | 789| 190 | 40316 | 129200 | 
| 148 | 44 django/contrib/postgres/fields/__init__.py | 1 | 6| 0 | 40316 | 129253 | 
| 149 | 44 django/db/models/fields/__init__.py | 311 | 339| 205 | 40521 | 129253 | 
| 150 | 45 django/contrib/admin/helpers.py | 355 | 364| 134 | 40655 | 132446 | 
| 151 | 46 django/db/backends/postgresql/utils.py | 1 | 8| 0 | 40655 | 132484 | 
| 152 | 46 django/contrib/admin/helpers.py | 366 | 382| 138 | 40793 | 132484 | 
| 153 | 46 django/db/models/fields/__init__.py | 1983 | 2013| 252 | 41045 | 132484 | 
| 154 | 47 django/contrib/admin/__init__.py | 1 | 30| 286 | 41331 | 132770 | 
| 155 | 47 django/contrib/admin/filters.py | 264 | 276| 149 | 41480 | 132770 | 
| 156 | 47 django/contrib/gis/db/backends/oracle/operations.py | 1 | 35| 309 | 41789 | 132770 | 
| 157 | 47 django/contrib/admin/options.py | 99 | 129| 223 | 42012 | 132770 | 
| 158 | 48 django/db/backends/base/features.py | 116 | 215| 852 | 42864 | 135332 | 
| 159 | 49 django/db/backends/postgresql/schema.py | 175 | 185| 132 | 42996 | 137109 | 
| 160 | 50 django/db/models/base.py | 1462 | 1484| 171 | 43167 | 152299 | 
| 161 | 50 django/contrib/postgres/fields/ranges.py | 42 | 86| 330 | 43497 | 152299 | 
| 162 | 51 django/db/backends/ddl_references.py | 106 | 121| 152 | 43649 | 153618 | 
| 163 | 51 django/db/backends/base/features.py | 1 | 115| 897 | 44546 | 153618 | 
| 164 | 51 django/contrib/gis/db/models/lookups.py | 340 | 362| 124 | 44670 | 153618 | 
| 165 | **51 django/db/models/lookups.py** | 186 | 203| 175 | 44845 | 153618 | 
| 166 | 51 django/contrib/admin/checks.py | 620 | 639| 183 | 45028 | 153618 | 
| 167 | 51 django/contrib/admin/widgets.py | 125 | 164| 349 | 45377 | 153618 | 
| 168 | 51 django/db/backends/postgresql/introspection.py | 137 | 208| 751 | 46128 | 153618 | 
| 169 | 51 django/contrib/admin/widgets.py | 449 | 477| 203 | 46331 | 153618 | 
| 170 | 51 django/db/models/fields/related_lookups.py | 46 | 60| 224 | 46555 | 153618 | 
| 171 | 51 django/contrib/admin/options.py | 1517 | 1596| 719 | 47274 | 153618 | 
| 172 | 52 django/db/models/options.py | 1 | 36| 304 | 47578 | 160717 | 
| 173 | 53 django/db/backends/base/schema.py | 1064 | 1078| 126 | 47704 | 172033 | 
| 174 | 53 django/db/backends/oracle/introspection.py | 51 | 67| 161 | 47865 | 172033 | 
| 175 | 53 django/contrib/admin/filters.py | 278 | 302| 217 | 48082 | 172033 | 
| 176 | 54 django/db/models/fields/related.py | 127 | 154| 202 | 48284 | 185545 | 
| 177 | 54 django/contrib/admin/options.py | 1653 | 1724| 653 | 48937 | 185545 | 
| 178 | 55 django/contrib/postgres/validators.py | 1 | 21| 181 | 49118 | 186096 | 
| 179 | 56 django/db/backends/oracle/creation.py | 130 | 165| 399 | 49517 | 189991 | 
| 180 | 57 django/db/backends/sqlite3/base.py | 383 | 403| 181 | 49698 | 195672 | 
| 181 | 57 django/contrib/admin/checks.py | 751 | 766| 182 | 49880 | 195672 | 
| 182 | 57 django/contrib/admin/options.py | 1725 | 1806| 744 | 50624 | 195672 | 
| 183 | 57 django/contrib/admin/options.py | 1597 | 1621| 279 | 50903 | 195672 | 
| 184 | 57 django/contrib/admin/widgets.py | 200 | 226| 216 | 51119 | 195672 | 
| 185 | 58 django/contrib/gis/db/models/functions.py | 88 | 119| 231 | 51350 | 199471 | 


### Hint

```
This isn't really an admin issue but rather it's due to the fact that the default admin lookup uses __icontains. You could fix the issue by using search_fields = ['uuidfield__exact'] (adding __exact) although that doesn't allow searching for part of the UUID value. I'll tentatively accept the ticket to allow QuerySet.objects.filter(uuidfield__icontains='...') to work with values with or without dashes
Thank you for the quick answer. However, the proposed solution does not work as changing the uuid lookup to exact causes an error when searching for any string other than valid uuid value. I get ValidationError raised from UUIDField.to_python() File "/home/vaclav/.local/share/virtualenvs/aaa-yRfablWV/lib/python3.7/site-packages/django/db/models/fields/__init__.py" in to_python 2325. return uuid.UUID(value) File "/usr/lib/python3.7/uuid.py" in __init__ 160. raise ValueError('badly formed hexadecimal UUID string')
​PR
```

## Patch

```diff
diff --git a/django/db/models/lookups.py b/django/db/models/lookups.py
--- a/django/db/models/lookups.py
+++ b/django/db/models/lookups.py
@@ -5,7 +5,7 @@
 from django.core.exceptions import EmptyResultSet
 from django.db.models.expressions import Case, Exists, Func, Value, When
 from django.db.models.fields import (
-    BooleanField, DateTimeField, Field, IntegerField,
+    BooleanField, CharField, DateTimeField, Field, IntegerField, UUIDField,
 )
 from django.db.models.query_utils import RegisterLookupMixin
 from django.utils.datastructures import OrderedSet
@@ -548,3 +548,53 @@ def get_bound_params(self, start, finish):
 class YearLte(YearLookup, LessThanOrEqual):
     def get_bound_params(self, start, finish):
         return (finish,)
+
+
+class UUIDTextMixin:
+    """
+    Strip hyphens from a value when filtering a UUIDField on backends without
+    a native datatype for UUID.
+    """
+    def process_rhs(self, qn, connection):
+        if not connection.features.has_native_uuid_field:
+            from django.db.models.functions import Replace
+            if self.rhs_is_direct_value():
+                self.rhs = Value(self.rhs)
+            self.rhs = Replace(self.rhs, Value('-'), Value(''), output_field=CharField())
+        rhs, params = super().process_rhs(qn, connection)
+        return rhs, params
+
+
+@UUIDField.register_lookup
+class UUIDIExact(UUIDTextMixin, IExact):
+    pass
+
+
+@UUIDField.register_lookup
+class UUIDContains(UUIDTextMixin, Contains):
+    pass
+
+
+@UUIDField.register_lookup
+class UUIDIContains(UUIDTextMixin, IContains):
+    pass
+
+
+@UUIDField.register_lookup
+class UUIDStartsWith(UUIDTextMixin, StartsWith):
+    pass
+
+
+@UUIDField.register_lookup
+class UUIDIStartsWith(UUIDTextMixin, IStartsWith):
+    pass
+
+
+@UUIDField.register_lookup
+class UUIDEndsWith(UUIDTextMixin, EndsWith):
+    pass
+
+
+@UUIDField.register_lookup
+class UUIDIEndsWith(UUIDTextMixin, IEndsWith):
+    pass

```

## Test Patch

```diff
diff --git a/tests/model_fields/test_uuid.py b/tests/model_fields/test_uuid.py
--- a/tests/model_fields/test_uuid.py
+++ b/tests/model_fields/test_uuid.py
@@ -2,7 +2,9 @@
 import uuid
 
 from django.core import exceptions, serializers
-from django.db import IntegrityError, models
+from django.db import IntegrityError, connection, models
+from django.db.models import CharField, F, Value
+from django.db.models.functions import Concat, Repeat
 from django.test import (
     SimpleTestCase, TestCase, TransactionTestCase, skipUnlessDBFeature,
 )
@@ -90,11 +92,41 @@ def setUpTestData(cls):
             NullableUUIDModel.objects.create(field=None),
         ]
 
+    def assertSequenceEqualWithoutHyphens(self, qs, result):
+        """
+        Backends with a native datatype for UUID don't support fragment lookups
+        without hyphens because they store values with them.
+        """
+        self.assertSequenceEqual(
+            qs,
+            [] if connection.features.has_native_uuid_field else result,
+        )
+
     def test_exact(self):
         self.assertSequenceEqual(
             NullableUUIDModel.objects.filter(field__exact='550e8400e29b41d4a716446655440000'),
             [self.objs[1]]
         )
+        self.assertSequenceEqual(
+            NullableUUIDModel.objects.filter(
+                field__exact='550e8400-e29b-41d4-a716-446655440000'
+            ),
+            [self.objs[1]],
+        )
+
+    def test_iexact(self):
+        self.assertSequenceEqualWithoutHyphens(
+            NullableUUIDModel.objects.filter(
+                field__iexact='550E8400E29B41D4A716446655440000'
+            ),
+            [self.objs[1]],
+        )
+        self.assertSequenceEqual(
+            NullableUUIDModel.objects.filter(
+                field__iexact='550E8400-E29B-41D4-A716-446655440000'
+            ),
+            [self.objs[1]],
+        )
 
     def test_isnull(self):
         self.assertSequenceEqual(
@@ -102,6 +134,86 @@ def test_isnull(self):
             [self.objs[2]]
         )
 
+    def test_contains(self):
+        self.assertSequenceEqualWithoutHyphens(
+            NullableUUIDModel.objects.filter(field__contains='8400e29b'),
+            [self.objs[1]],
+        )
+        self.assertSequenceEqual(
+            NullableUUIDModel.objects.filter(field__contains='8400-e29b'),
+            [self.objs[1]],
+        )
+
+    def test_icontains(self):
+        self.assertSequenceEqualWithoutHyphens(
+            NullableUUIDModel.objects.filter(field__icontains='8400E29B'),
+            [self.objs[1]],
+        )
+        self.assertSequenceEqual(
+            NullableUUIDModel.objects.filter(field__icontains='8400-E29B'),
+            [self.objs[1]],
+        )
+
+    def test_startswith(self):
+        self.assertSequenceEqualWithoutHyphens(
+            NullableUUIDModel.objects.filter(field__startswith='550e8400e29b4'),
+            [self.objs[1]],
+        )
+        self.assertSequenceEqual(
+            NullableUUIDModel.objects.filter(field__startswith='550e8400-e29b-4'),
+            [self.objs[1]],
+        )
+
+    def test_istartswith(self):
+        self.assertSequenceEqualWithoutHyphens(
+            NullableUUIDModel.objects.filter(field__istartswith='550E8400E29B4'),
+            [self.objs[1]],
+        )
+        self.assertSequenceEqual(
+            NullableUUIDModel.objects.filter(field__istartswith='550E8400-E29B-4'),
+            [self.objs[1]],
+        )
+
+    def test_endswith(self):
+        self.assertSequenceEqualWithoutHyphens(
+            NullableUUIDModel.objects.filter(field__endswith='a716446655440000'),
+            [self.objs[1]],
+        )
+        self.assertSequenceEqual(
+            NullableUUIDModel.objects.filter(field__endswith='a716-446655440000'),
+            [self.objs[1]],
+        )
+
+    def test_iendswith(self):
+        self.assertSequenceEqualWithoutHyphens(
+            NullableUUIDModel.objects.filter(field__iendswith='A716446655440000'),
+            [self.objs[1]],
+        )
+        self.assertSequenceEqual(
+            NullableUUIDModel.objects.filter(field__iendswith='A716-446655440000'),
+            [self.objs[1]],
+        )
+
+    def test_filter_with_expr(self):
+        self.assertSequenceEqualWithoutHyphens(
+            NullableUUIDModel.objects.annotate(
+                value=Concat(Value('8400'), Value('e29b'), output_field=CharField()),
+            ).filter(field__contains=F('value')),
+            [self.objs[1]],
+        )
+        self.assertSequenceEqual(
+            NullableUUIDModel.objects.annotate(
+                value=Concat(Value('8400'), Value('-'), Value('e29b'), output_field=CharField()),
+            ).filter(field__contains=F('value')),
+            [self.objs[1]],
+        )
+        self.assertSequenceEqual(
+            NullableUUIDModel.objects.annotate(
+                value=Repeat(Value('0'), 4, output_field=CharField()),
+            ).filter(field__contains=F('value')),
+            [self.objs[1]],
+        )
+
 
 class TestSerialization(SimpleTestCase):
     test_data = (

```


## Code snippets

### 1 - django/db/models/fields/__init__.py:

Start line: 2268, End line: 2318

```python
class UUIDField(Field):
    default_error_messages = {
        'invalid': _('“%(value)s” is not a valid UUID.'),
    }
    description = _('Universally unique identifier')
    empty_strings_allowed = False

    def __init__(self, verbose_name=None, **kwargs):
        kwargs['max_length'] = 32
        super().__init__(verbose_name, **kwargs)

    def deconstruct(self):
        name, path, args, kwargs = super().deconstruct()
        del kwargs['max_length']
        return name, path, args, kwargs

    def get_internal_type(self):
        return "UUIDField"

    def get_prep_value(self, value):
        value = super().get_prep_value(value)
        return self.to_python(value)

    def get_db_prep_value(self, value, connection, prepared=False):
        if value is None:
            return None
        if not isinstance(value, uuid.UUID):
            value = self.to_python(value)

        if connection.features.has_native_uuid_field:
            return value
        return value.hex

    def to_python(self, value):
        if value is not None and not isinstance(value, uuid.UUID):
            input_form = 'int' if isinstance(value, int) else 'hex'
            try:
                return uuid.UUID(**{input_form: value})
            except (AttributeError, ValueError):
                raise exceptions.ValidationError(
                    self.error_messages['invalid'],
                    code='invalid',
                    params={'value': value},
                )
        return value

    def formfield(self, **kwargs):
        return super().formfield(**{
            'form_class': forms.UUIDField,
            **kwargs,
        })
```
### 2 - django/contrib/admin/options.py:

Start line: 1008, End line: 1019

```python
class ModelAdmin(BaseModelAdmin):

    def get_search_results(self, request, queryset, search_term):
        # ... other code

        use_distinct = False
        search_fields = self.get_search_fields(request)
        if search_fields and search_term:
            orm_lookups = [construct_search(str(search_field))
                           for search_field in search_fields]
            for bit in search_term.split():
                or_queries = [models.Q(**{orm_lookup: bit})
                              for orm_lookup in orm_lookups]
                queryset = queryset.filter(reduce(operator.or_, or_queries))
            use_distinct |= any(lookup_needs_distinct(self.opts, search_spec) for search_spec in orm_lookups)

        return queryset, use_distinct
```
### 3 - django/db/backends/mysql/operations.py:

Start line: 1, End line: 33

```python
import uuid

from django.conf import settings
from django.db.backends.base.operations import BaseDatabaseOperations
from django.utils import timezone
from django.utils.duration import duration_microseconds
from django.utils.encoding import force_str


class DatabaseOperations(BaseDatabaseOperations):
    compiler_module = "django.db.backends.mysql.compiler"

    # MySQL stores positive fields as UNSIGNED ints.
    integer_field_ranges = {
        **BaseDatabaseOperations.integer_field_ranges,
        'PositiveSmallIntegerField': (0, 65535),
        'PositiveIntegerField': (0, 4294967295),
    }
    cast_data_types = {
        'AutoField': 'signed integer',
        'BigAutoField': 'signed integer',
        'SmallAutoField': 'signed integer',
        'CharField': 'char(%(max_length)s)',
        'DecimalField': 'decimal(%(max_digits)s, %(decimal_places)s)',
        'TextField': 'char',
        'IntegerField': 'signed integer',
        'BigIntegerField': 'signed integer',
        'SmallIntegerField': 'signed integer',
        'PositiveIntegerField': 'unsigned integer',
        'PositiveSmallIntegerField': 'unsigned integer',
    }
    cast_char_field_without_max_length = 'char'
    explain_prefix = 'EXPLAIN'
```
### 4 - django/contrib/admin/options.py:

Start line: 973, End line: 1006

```python
class ModelAdmin(BaseModelAdmin):

    def get_search_results(self, request, queryset, search_term):
        """
        Return a tuple containing a queryset to implement the search
        and a boolean indicating if the results may contain duplicates.
        """
        # Apply keyword searches.
        def construct_search(field_name):
            if field_name.startswith('^'):
                return "%s__istartswith" % field_name[1:]
            elif field_name.startswith('='):
                return "%s__iexact" % field_name[1:]
            elif field_name.startswith('@'):
                return "%s__search" % field_name[1:]
            # Use field_name if it includes a lookup.
            opts = queryset.model._meta
            lookup_fields = field_name.split(LOOKUP_SEP)
            # Go through the fields, following all relations.
            prev_field = None
            for path_part in lookup_fields:
                if path_part == 'pk':
                    path_part = opts.pk.name
                try:
                    field = opts.get_field(path_part)
                except FieldDoesNotExist:
                    # Use valid query lookups.
                    if prev_field and prev_field.get_lookup(path_part):
                        return field_name
                else:
                    prev_field = field
                    if hasattr(field, 'get_path_info'):
                        # Update opts to follow the relation.
                        opts = field.get_path_info()[-1].to_opts
            # Otherwise, use the field with icontains.
            return "%s__icontains" % field_name
        # ... other code
```
### 5 - django/db/backends/oracle/base.py:

Start line: 389, End line: 411

```python
class FormatStylePlaceholderCursor:
    """
    Django uses "format" (e.g. '%s') style placeholders, but Oracle uses ":var"
    style. This fixes it -- but note that if you want to use a literal "%s" in
    a query, you'll need to use "%%s".
    """
    charset = 'utf-8'

    def __init__(self, connection):
        self.cursor = connection.cursor()
        self.cursor.outputtypehandler = self._output_type_handler

    @staticmethod
    def _output_number_converter(value):
        return decimal.Decimal(value) if '.' in value else int(value)

    @staticmethod
    def _get_decimal_converter(precision, scale):
        if scale == 0:
            return int
        context = decimal.Context(prec=precision)
        quantize_value = decimal.Decimal(1).scaleb(-scale)
        return lambda v: decimal.Decimal(v).quantize(quantize_value, context=context)
```
### 6 - django/db/backends/oracle/operations.py:

Start line: 1, End line: 17

```python
import datetime
import re
import uuid
from functools import lru_cache

from django.conf import settings
from django.db.backends.base.operations import BaseDatabaseOperations
from django.db.backends.utils import strip_quotes, truncate_name
from django.db.models.expressions import Exists, ExpressionWrapper
from django.db.models.query_utils import Q
from django.db.utils import DatabaseError
from django.utils import timezone
from django.utils.encoding import force_bytes, force_str
from django.utils.functional import cached_property

from .base import Database
from .utils import BulkInsertMapper, InsertVar, Oracle_datetime
```
### 7 - django/contrib/admin/views/main.py:

Start line: 1, End line: 45

```python
from datetime import datetime, timedelta

from django import forms
from django.conf import settings
from django.contrib import messages
from django.contrib.admin import FieldListFilter
from django.contrib.admin.exceptions import (
    DisallowedModelAdminLookup, DisallowedModelAdminToField,
)
from django.contrib.admin.options import (
    IS_POPUP_VAR, TO_FIELD_VAR, IncorrectLookupParameters,
)
from django.contrib.admin.utils import (
    get_fields_from_path, lookup_needs_distinct, prepare_lookup_value, quote,
)
from django.core.exceptions import (
    FieldDoesNotExist, ImproperlyConfigured, SuspiciousOperation,
)
from django.core.paginator import InvalidPage
from django.db import models
from django.db.models.expressions import Combinable, F, OrderBy
from django.urls import reverse
from django.utils.http import urlencode
from django.utils.timezone import make_aware
from django.utils.translation import gettext

# Changelist settings
ALL_VAR = 'all'
ORDER_VAR = 'o'
ORDER_TYPE_VAR = 'ot'
PAGE_VAR = 'p'
SEARCH_VAR = 'q'
ERROR_FLAG = 'e'

IGNORED_PARAMS = (
    ALL_VAR, ORDER_VAR, ORDER_TYPE_VAR, SEARCH_VAR, IS_POPUP_VAR, TO_FIELD_VAR)


class ChangeListSearchForm(forms.Form):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Populate "fields" dynamically because SEARCH_VAR is a variable:
        self.fields = {
            SEARCH_VAR: forms.CharField(required=False, strip=False),
        }
```
### 8 - django/contrib/postgres/lookups.py:

Start line: 1, End line: 77

```python
from django.db.models import Lookup, Transform
from django.db.models.lookups import Exact, FieldGetDbPrepValueMixin

from .search import SearchVector, SearchVectorExact, SearchVectorField


class PostgresSimpleLookup(FieldGetDbPrepValueMixin, Lookup):
    def as_sql(self, qn, connection):
        lhs, lhs_params = self.process_lhs(qn, connection)
        rhs, rhs_params = self.process_rhs(qn, connection)
        params = lhs_params + rhs_params
        return '%s %s %s' % (lhs, self.operator, rhs), params


class DataContains(PostgresSimpleLookup):
    lookup_name = 'contains'
    operator = '@>'


class ContainedBy(PostgresSimpleLookup):
    lookup_name = 'contained_by'
    operator = '<@'


class Overlap(PostgresSimpleLookup):
    lookup_name = 'overlap'
    operator = '&&'


class HasKey(PostgresSimpleLookup):
    lookup_name = 'has_key'
    operator = '?'
    prepare_rhs = False


class HasKeys(PostgresSimpleLookup):
    lookup_name = 'has_keys'
    operator = '?&'

    def get_prep_lookup(self):
        return [str(item) for item in self.rhs]


class HasAnyKeys(HasKeys):
    lookup_name = 'has_any_keys'
    operator = '?|'


class Unaccent(Transform):
    bilateral = True
    lookup_name = 'unaccent'
    function = 'UNACCENT'


class SearchLookup(SearchVectorExact):
    lookup_name = 'search'

    def process_lhs(self, qn, connection):
        if not isinstance(self.lhs.output_field, SearchVectorField):
            self.lhs = SearchVector(self.lhs)
        lhs, lhs_params = super().process_lhs(qn, connection)
        return lhs, lhs_params


class TrigramSimilar(PostgresSimpleLookup):
    lookup_name = 'trigram_similar'
    operator = '%%'


class JSONExact(Exact):
    can_use_none_as_rhs = True

    def process_rhs(self, compiler, connection):
        result = super().process_rhs(compiler, connection)
        # Treat None lookup values as null.
        return ("'null'", []) if result == ('%s', [None]) else result
```
### 9 - django/forms/fields.py:

Start line: 1173, End line: 1203

```python
class SlugField(CharField):
    default_validators = [validators.validate_slug]

    def __init__(self, *, allow_unicode=False, **kwargs):
        self.allow_unicode = allow_unicode
        if self.allow_unicode:
            self.default_validators = [validators.validate_unicode_slug]
        super().__init__(**kwargs)


class UUIDField(CharField):
    default_error_messages = {
        'invalid': _('Enter a valid UUID.'),
    }

    def prepare_value(self, value):
        if isinstance(value, uuid.UUID):
            return str(value)
        return value

    def to_python(self, value):
        value = super().to_python(value)
        if value in self.empty_values:
            return None
        if not isinstance(value, uuid.UUID):
            try:
                value = uuid.UUID(value)
            except ValueError:
                raise ValidationError(self.error_messages['invalid'], code='invalid')
        return value
```
### 10 - django/db/backends/oracle/operations.py:

Start line: 294, End line: 319

```python
class DatabaseOperations(BaseDatabaseOperations):
    # Oracle uses NUMBER(5), NUMBER(11), and NUMBER(19) for integer fields.
    # SmallIntegerField uses NUMBER(11) instead of NUMBER(5), which is used by
    integer_field_ranges =
    # ... other code

    def last_insert_id(self, cursor, table_name, pk_name):
        sq_name = self._get_sequence_name(cursor, strip_quotes(table_name), pk_name)
        cursor.execute('"%s".currval' % sq_name)
        return cursor.fetchone()[0]

    def lookup_cast(self, lookup_type, internal_type=None):
        if lookup_type in ('iexact', 'icontains', 'istartswith', 'iendswith'):
            return "UPPER(%s)"
        return "%s"

    def max_in_list_size(self):
        return 1000

    def max_name_length(self):
        return 30

    def pk_default_value(self):
        return "NULL"

    def prep_for_iexact_query(self, x):
        return x

    def process_clob(self, value):
        if value is None:
            return ''
        return value.read()
    # ... other code
```
### 20 - django/db/models/lookups.py:

Start line: 117, End line: 140

```python
class Lookup:

    def as_oracle(self, compiler, connection):
        # Oracle doesn't allow EXISTS() to be compared to another expression
        # unless it's wrapped in a CASE WHEN.
        wrapped = False
        exprs = []
        for expr in (self.lhs, self.rhs):
            if isinstance(expr, Exists):
                expr = Case(When(expr, then=True), default=False, output_field=BooleanField())
                wrapped = True
            exprs.append(expr)
        lookup = type(self)(*exprs) if wrapped else self
        return lookup.as_sql(compiler, connection)

    @cached_property
    def contains_aggregate(self):
        return self.lhs.contains_aggregate or getattr(self.rhs, 'contains_aggregate', False)

    @cached_property
    def contains_over_clause(self):
        return self.lhs.contains_over_clause or getattr(self.rhs, 'contains_over_clause', False)

    @property
    def is_summary(self):
        return self.lhs.is_summary or getattr(self.rhs, 'is_summary', False)
```
### 27 - django/db/models/lookups.py:

Start line: 420, End line: 470

```python
@Field.register_lookup
class Contains(PatternLookup):
    lookup_name = 'contains'


@Field.register_lookup
class IContains(Contains):
    lookup_name = 'icontains'


@Field.register_lookup
class StartsWith(PatternLookup):
    lookup_name = 'startswith'
    param_pattern = '%s%%'


@Field.register_lookup
class IStartsWith(StartsWith):
    lookup_name = 'istartswith'


@Field.register_lookup
class EndsWith(PatternLookup):
    lookup_name = 'endswith'
    param_pattern = '%%%s'


@Field.register_lookup
class IEndsWith(EndsWith):
    lookup_name = 'iendswith'


@Field.register_lookup
class Range(FieldGetDbPrepValueIterableMixin, BuiltinLookup):
    lookup_name = 'range'

    def get_rhs_op(self, connection, rhs):
        return "BETWEEN %s AND %s" % (rhs[0], rhs[1])


@Field.register_lookup
class IsNull(BuiltinLookup):
    lookup_name = 'isnull'
    prepare_rhs = False

    def as_sql(self, compiler, connection):
        sql, params = compiler.compile(self.lhs)
        if self.rhs:
            return "%s IS NULL" % sql, params
        else:
            return "%s IS NOT NULL" % sql, params
```
### 72 - django/db/models/lookups.py:

Start line: 276, End line: 326

```python
@Field.register_lookup
class IExact(BuiltinLookup):
    lookup_name = 'iexact'
    prepare_rhs = False

    def process_rhs(self, qn, connection):
        rhs, params = super().process_rhs(qn, connection)
        if params:
            params[0] = connection.ops.prep_for_iexact_query(params[0])
        return rhs, params


@Field.register_lookup
class GreaterThan(FieldGetDbPrepValueMixin, BuiltinLookup):
    lookup_name = 'gt'


@Field.register_lookup
class GreaterThanOrEqual(FieldGetDbPrepValueMixin, BuiltinLookup):
    lookup_name = 'gte'


@Field.register_lookup
class LessThan(FieldGetDbPrepValueMixin, BuiltinLookup):
    lookup_name = 'lt'


@Field.register_lookup
class LessThanOrEqual(FieldGetDbPrepValueMixin, BuiltinLookup):
    lookup_name = 'lte'


class IntegerFieldFloatRounding:
    """
    Allow floats to work as query values for IntegerField. Without this, the
    decimal portion of the float would always be discarded.
    """
    def get_prep_lookup(self):
        if isinstance(self.rhs, float):
            self.rhs = math.ceil(self.rhs)
        return super().get_prep_lookup()


@IntegerField.register_lookup
class IntegerGreaterThanOrEqual(IntegerFieldFloatRounding, GreaterThanOrEqual):
    pass


@IntegerField.register_lookup
class IntegerLessThan(IntegerFieldFloatRounding, LessThan):
    pass
```
### 77 - django/db/models/lookups.py:

Start line: 257, End line: 273

```python
@Field.register_lookup
class Exact(FieldGetDbPrepValueMixin, BuiltinLookup):
    lookup_name = 'exact'

    def process_rhs(self, compiler, connection):
        from django.db.models.sql.query import Query
        if isinstance(self.rhs, Query):
            if self.rhs.has_limit_one():
                if not self.rhs.has_select_fields:
                    self.rhs.clear_select_clause()
                    self.rhs.add_fields(['pk'])
            else:
                raise ValueError(
                    'The QuerySet value for an exact lookup must be limited to '
                    'one result using slicing.'
                )
        return super().process_rhs(compiler, connection)
```
### 85 - django/db/models/lookups.py:

Start line: 329, End line: 359

```python
@Field.register_lookup
class In(FieldGetDbPrepValueIterableMixin, BuiltinLookup):
    lookup_name = 'in'

    def process_rhs(self, compiler, connection):
        db_rhs = getattr(self.rhs, '_db', None)
        if db_rhs is not None and db_rhs != connection.alias:
            raise ValueError(
                "Subqueries aren't allowed across different databases. Force "
                "the inner query to be evaluated using `list(inner_query)`."
            )

        if self.rhs_is_direct_value():
            try:
                rhs = OrderedSet(self.rhs)
            except TypeError:  # Unhashable items in self.rhs
                rhs = self.rhs

            if not rhs:
                raise EmptyResultSet

            # rhs should be an iterable; use batch_process_rhs() to
            # prepare/transform those values.
            sqls, sqls_params = self.batch_process_rhs(compiler, connection, rhs)
            placeholder = '(' + ', '.join(sqls) + ')'
            return (placeholder, sqls_params)
        else:
            if not getattr(self.rhs, 'has_select_fields', True):
                self.rhs.clear_select_clause()
                self.rhs.add_fields(['pk'])
            return super().process_rhs(compiler, connection)
```
### 119 - django/db/models/lookups.py:

Start line: 361, End line: 390

```python
@Field.register_lookup
class In(FieldGetDbPrepValueIterableMixin, BuiltinLookup):

    def get_rhs_op(self, connection, rhs):
        return 'IN %s' % rhs

    def as_sql(self, compiler, connection):
        max_in_list_size = connection.ops.max_in_list_size()
        if self.rhs_is_direct_value() and max_in_list_size and len(self.rhs) > max_in_list_size:
            return self.split_parameter_list_as_sql(compiler, connection)
        return super().as_sql(compiler, connection)

    def split_parameter_list_as_sql(self, compiler, connection):
        # This is a special case for databases which limit the number of
        # elements which can appear in an 'IN' clause.
        max_in_list_size = connection.ops.max_in_list_size()
        lhs, lhs_params = self.process_lhs(compiler, connection)
        rhs, rhs_params = self.batch_process_rhs(compiler, connection)
        in_clause_elements = ['(']
        params = []
        for offset in range(0, len(rhs_params), max_in_list_size):
            if offset > 0:
                in_clause_elements.append(' OR ')
            in_clause_elements.append('%s IN (' % lhs)
            params.extend(lhs_params)
            sqls = rhs[offset: offset + max_in_list_size]
            sqls_params = rhs_params[offset: offset + max_in_list_size]
            param_group = ', '.join(sqls)
            in_clause_elements.append(param_group)
            in_clause_elements.append(')')
            params.extend(sqls_params)
        in_clause_elements.append(')')
        return ''.join(in_clause_elements), params
```
### 165 - django/db/models/lookups.py:

Start line: 186, End line: 203

```python
class FieldGetDbPrepValueMixin:
    """
    Some lookups require Field.get_db_prep_value() to be called on their
    inputs.
    """
    get_db_prep_lookup_value_is_iterable = False

    def get_db_prep_lookup(self, value, connection):
        # For relational fields, use the 'target_field' attribute of the
        # output_field.
        field = getattr(self.lhs.output_field, 'target_field', None)
        get_db_prep_value = getattr(field, 'get_db_prep_value', None) or self.lhs.output_field.get_db_prep_value
        return (
            '%s',
            [get_db_prep_value(v, connection, prepared=True) for v in value]
            if self.get_db_prep_lookup_value_is_iterable else
            [get_db_prep_value(value, connection, prepared=True)]
        )
```
