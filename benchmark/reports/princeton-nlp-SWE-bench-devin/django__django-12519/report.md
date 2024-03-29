# django__django-12519

| **django/django** | `d4fff711d4c97356bd6ba1273d2a5e349326eb5f` |
| ---- | ---- |
| **No of patches** | 3 |
| **All found context length** | - |
| **Any found context length** | 33021 |
| **Avg pos** | 57.0 |
| **Min pos** | 62 |
| **Max pos** | 109 |
| **Top file pos** | 10 |
| **Missing snippets** | 6 |
| **Missing patch files** | 2 |


## Expected patch

```diff
diff --git a/django/db/backends/mysql/base.py b/django/db/backends/mysql/base.py
--- a/django/db/backends/mysql/base.py
+++ b/django/db/backends/mysql/base.py
@@ -364,3 +364,10 @@ def mysql_version(self):
     @cached_property
     def mysql_is_mariadb(self):
         return 'mariadb' in self.mysql_server_info.lower()
+
+    @cached_property
+    def sql_mode(self):
+        with self.cursor() as cursor:
+            cursor.execute('SELECT @@sql_mode')
+            sql_mode = cursor.fetchone()
+        return set(sql_mode[0].split(',') if sql_mode else ())
diff --git a/django/db/backends/mysql/validation.py b/django/db/backends/mysql/validation.py
--- a/django/db/backends/mysql/validation.py
+++ b/django/db/backends/mysql/validation.py
@@ -10,11 +10,7 @@ def check(self, **kwargs):
         return issues
 
     def _check_sql_mode(self, **kwargs):
-        with self.connection.cursor() as cursor:
-            cursor.execute("SELECT @@sql_mode")
-            sql_mode = cursor.fetchone()
-        modes = set(sql_mode[0].split(',') if sql_mode else ())
-        if not (modes & {'STRICT_TRANS_TABLES', 'STRICT_ALL_TABLES'}):
+        if not (self.connection.sql_mode & {'STRICT_TRANS_TABLES', 'STRICT_ALL_TABLES'}):
             return [checks.Warning(
                 "MySQL Strict Mode is not set for database connection '%s'" % self.connection.alias,
                 hint="MySQL's Strict Mode fixes many data integrity problems in MySQL, "
diff --git a/django/db/models/expressions.py b/django/db/models/expressions.py
--- a/django/db/models/expressions.py
+++ b/django/db/models/expressions.py
@@ -6,6 +6,7 @@
 from django.core.exceptions import EmptyResultSet, FieldError
 from django.db import NotSupportedError, connection
 from django.db.models import fields
+from django.db.models.constants import LOOKUP_SEP
 from django.db.models.query_utils import Q
 from django.utils.deconstruct import deconstructible
 from django.utils.functional import cached_property
@@ -559,6 +560,14 @@ def as_sql(self, *args, **kwargs):
             'only be used in a subquery.'
         )
 
+    def resolve_expression(self, *args, **kwargs):
+        col = super().resolve_expression(*args, **kwargs)
+        # FIXME: Rename possibly_multivalued to multivalued and fix detection
+        # for non-multivalued JOINs (e.g. foreign key fields). This should take
+        # into account only many-to-many and one-to-many relationships.
+        col.possibly_multivalued = LOOKUP_SEP in self.name
+        return col
+
     def relabeled_clone(self, relabels):
         return self
 
@@ -747,6 +756,7 @@ def as_sql(self, compiler, connection):
 class Col(Expression):
 
     contains_column_references = True
+    possibly_multivalued = False
 
     def __init__(self, alias, target, output_field=None):
         if output_field is None:
@@ -1042,7 +1052,10 @@ def as_sql(self, compiler, connection, template=None, **extra_context):
     def get_group_by_cols(self, alias=None):
         if alias:
             return [Ref(alias, self)]
-        return self.query.get_external_cols()
+        external_cols = self.query.get_external_cols()
+        if any(col.possibly_multivalued for col in external_cols):
+            return [self]
+        return external_cols
 
 
 class Exists(Subquery):

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/db/backends/mysql/base.py | 367 | 367 | - | - | -
| django/db/backends/mysql/validation.py | 13 | 17 | - | - | -
| django/db/models/expressions.py | 9 | 9 | - | 10 | -
| django/db/models/expressions.py | 562 | 562 | - | 10 | -
| django/db/models/expressions.py | 750 | 750 | 62 | 10 | 33021
| django/db/models/expressions.py | 1045 | 1045 | 109 | 10 | 55346


## Problem Statement

```
Subquery annotations are omitted in group by query section if multiple annotation are declared
Description
	 
		(last modified by Johannes Maron)
	 
Sadly there is more regression in Django 3.0.2 even after #31094.
Background: It's the same query as #31094. I tried upgrading to Django 3.0.2 and now I get duplicate results. Even tho they query should be distinct. Where on 2.2 the queryset yields 490 results, it's 519 on 3.0.
A quick diff on the queries still reveals a different grouped by section:
This is the new query on 3.0.2:
SELECT DISTINCT "camps_offer"."id",
				"camps_offer"."title",
				"camps_offer"."slug",
				"camps_offer"."is_active",
				"camps_offer"."modified",
				"camps_offer"."created",
				"camps_offer"."provider_id",
				"camps_offer"."activity_type",
				"camps_offer"."description",
				"camps_offer"."highlights",
				"camps_offer"."important_information",
				"camps_offer"."min_age",
				"camps_offer"."max_age",
				"camps_offer"."food",
				"camps_offer"."video",
				"camps_offer"."accommodation",
				"camps_offer"."accommodation_type",
				"camps_offer"."room_type",
				"camps_offer"."room_size_min",
				"camps_offer"."room_size_max",
				"camps_offer"."external_url",
				"camps_offer"."application_form",
				"camps_offer"."caseload",
				"camps_offer"."field_trips",
				MIN(T4."retail_price") AS "min_retail_price",
				(SELECT U0."id"
				 FROM "camps_servicepackage" U0
						 INNER JOIN "camps_region" U2 ON (U0."region_id" = U2."id")
				 WHERE (U0."company_id" = 1 AND U0."option" = "camps_offer"."activity_type" AND
						ST_Contains(U2."locations", T4."position"))
				 LIMIT 1)			 AS "in_package",
				"camps_provider"."id",
				"camps_provider"."title",
				"camps_provider"."slug",
				"camps_provider"."is_active",
				"camps_provider"."modified",
				"camps_provider"."created",
				"camps_provider"."logo",
				"camps_provider"."description",
				"camps_provider"."video",
				"camps_provider"."external_url",
				"camps_provider"."terms",
				"camps_provider"."cancellation_policy",
				"camps_provider"."privacy_policy",
				"camps_provider"."application_form"
FROM "camps_offer"
		 LEFT OUTER JOIN "camps_bookingoption" ON ("camps_offer"."id" = "camps_bookingoption"."offer_id")
		 INNER JOIN "camps_provider" ON ("camps_offer"."provider_id" = "camps_provider"."id")
		 INNER JOIN "camps_bookingoption" T4 ON ("camps_offer"."id" = T4."offer_id")
WHERE ("camps_offer"."is_active" = True AND "camps_provider"."is_active" = True AND
	 T4."end" >= STATEMENT_TIMESTAMP() AND T4."is_active" = True AND "camps_offer"."max_age" >= 5 AND
	 "camps_offer"."min_age" <= 13 AND (SELECT U0."id"
										 FROM "camps_servicepackage" U0
												 INNER JOIN "camps_region" U2 ON (U0."region_id" = U2."id")
										 WHERE (U0."company_id" = 1 AND U0."option" = "camps_offer"."activity_type" AND
												 ST_Contains(U2."locations", T4."position"))
										 LIMIT 1) IS NOT NULL)
GROUP BY "camps_offer"."id", T4."position", "camps_provider"."id"
ORDER BY "camps_offer"."created" ASC
And what it was (and should be) on 2.2.9:
SELECT DISTINCT "camps_offer"."id",
				"camps_offer"."title",
				"camps_offer"."slug",
				"camps_offer"."is_active",
				"camps_offer"."modified",
				"camps_offer"."created",
				"camps_offer"."provider_id",
				"camps_offer"."activity_type",
				"camps_offer"."description",
				"camps_offer"."highlights",
				"camps_offer"."important_information",
				"camps_offer"."min_age",
				"camps_offer"."max_age",
				"camps_offer"."food",
				"camps_offer"."video",
				"camps_offer"."accommodation",
				"camps_offer"."accommodation_type",
				"camps_offer"."room_type",
				"camps_offer"."room_size_min",
				"camps_offer"."room_size_max",
				"camps_offer"."external_url",
				"camps_offer"."application_form",
				"camps_offer"."caseload",
				"camps_offer"."field_trips",
				MIN(T4."retail_price") AS "min_retail_price",
				(SELECT U0."id"
				 FROM "camps_servicepackage" U0
						 INNER JOIN "camps_region" U2 ON (U0."region_id" = U2."id")
				 WHERE (U0."company_id" = 1 AND U0."option" = ("camps_offer"."activity_type") AND
						ST_Contains(U2."locations", (T4."position")))
				 LIMIT 1)			 AS "in_package",
				"camps_provider"."id",
				"camps_provider"."title",
				"camps_provider"."slug",
				"camps_provider"."is_active",
				"camps_provider"."modified",
				"camps_provider"."created",
				"camps_provider"."logo",
				"camps_provider"."description",
				"camps_provider"."video",
				"camps_provider"."external_url",
				"camps_provider"."terms",
				"camps_provider"."cancellation_policy",
				"camps_provider"."privacy_policy",
				"camps_provider"."application_form"
FROM "camps_offer"
		 LEFT OUTER JOIN "camps_bookingoption" ON ("camps_offer"."id" = "camps_bookingoption"."offer_id")
		 INNER JOIN "camps_provider" ON ("camps_offer"."provider_id" = "camps_provider"."id")
		 INNER JOIN "camps_bookingoption" T4 ON ("camps_offer"."id" = T4."offer_id")
WHERE ("camps_offer"."is_active" = True AND "camps_provider"."is_active" = True AND
	 T4."end" >= (STATEMENT_TIMESTAMP()) AND T4."is_active" = True AND (SELECT U0."id"
																		 FROM "camps_servicepackage" U0
																				 INNER JOIN "camps_region" U2 ON (U0."region_id" = U2."id")
																		 WHERE (U0."company_id" = 1 AND
																				 U0."option" = ("camps_offer"."activity_type") AND
																				 ST_Contains(U2."locations", (T4."position")))
																		 LIMIT 1) IS NOT NULL)
GROUP BY "camps_offer"."id",
		 (SELECT U0."id"
		 FROM "camps_servicepackage" U0
				 INNER JOIN "camps_region" U2 ON (U0."region_id" = U2."id")
		 WHERE (U0."company_id" = 1 AND U0."option" = ("camps_offer"."activity_type") AND
				 ST_Contains(U2."locations", (T4."position")))
		 LIMIT 1), "camps_provider"."id"
ORDER BY "camps_offer"."created" ASC

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 django/db/models/sql/query.py | 1281 | 1339| 711 | 711 | 21669 | 
| 2 | 2 django/db/models/sql/compiler.py | 58 | 141| 851 | 1562 | 35725 | 
| 3 | 2 django/db/models/sql/query.py | 363 | 413| 494 | 2056 | 35725 | 
| 4 | 2 django/db/models/sql/query.py | 2317 | 2372| 827 | 2883 | 35725 | 
| 5 | 2 django/db/models/sql/query.py | 1695 | 1766| 784 | 3667 | 35725 | 
| 6 | 3 django/db/models/query.py | 790 | 829| 322 | 3989 | 52744 | 
| 7 | 3 django/db/models/sql/query.py | 1921 | 1963| 356 | 4345 | 52744 | 
| 8 | 3 django/db/models/query.py | 243 | 269| 221 | 4566 | 52744 | 
| 9 | 3 django/db/models/sql/compiler.py | 193 | 263| 564 | 5130 | 52744 | 
| 10 | 4 django/db/models/__init__.py | 1 | 52| 605 | 5735 | 53349 | 
| 11 | 4 django/db/models/sql/query.py | 696 | 731| 389 | 6124 | 53349 | 
| 12 | 4 django/db/models/sql/query.py | 1659 | 1693| 399 | 6523 | 53349 | 
| 13 | 4 django/db/models/sql/compiler.py | 864 | 956| 829 | 7352 | 53349 | 
| 14 | 4 django/db/models/query.py | 1622 | 1728| 1063 | 8415 | 53349 | 
| 15 | 4 django/db/models/sql/compiler.py | 265 | 352| 696 | 9111 | 53349 | 
| 16 | 4 django/db/models/sql/query.py | 138 | 238| 874 | 9985 | 53349 | 
| 17 | 5 django/db/models/fields/related_lookups.py | 62 | 99| 447 | 10432 | 54798 | 
| 18 | 5 django/db/models/sql/query.py | 1797 | 1846| 330 | 10762 | 54798 | 
| 19 | 5 django/db/models/sql/compiler.py | 479 | 629| 1392 | 12154 | 54798 | 
| 20 | 5 django/db/models/sql/query.py | 1593 | 1617| 227 | 12381 | 54798 | 
| 21 | 5 django/db/models/query.py | 326 | 349| 203 | 12584 | 54798 | 
| 22 | 5 django/db/models/sql/query.py | 2129 | 2161| 228 | 12812 | 54798 | 
| 23 | 6 django/db/models/base.py | 1665 | 1763| 717 | 13529 | 70148 | 
| 24 | 7 django/contrib/admin/views/main.py | 206 | 255| 423 | 13952 | 74446 | 
| 25 | 7 django/contrib/admin/views/main.py | 121 | 204| 827 | 14779 | 74446 | 
| 26 | 7 django/db/models/query.py | 968 | 999| 341 | 15120 | 74446 | 
| 27 | 8 django/conf/locale/nl/formats.py | 32 | 67| 1371 | 16491 | 76329 | 
| 28 | 9 django/db/backends/postgresql/operations.py | 209 | 293| 674 | 17165 | 78998 | 
| 29 | 9 django/contrib/admin/views/main.py | 434 | 477| 390 | 17555 | 78998 | 
| 30 | 9 django/db/models/sql/query.py | 1619 | 1657| 356 | 17911 | 78998 | 
| 31 | 9 django/db/models/query.py | 1285 | 1303| 186 | 18097 | 78998 | 
| 32 | 9 django/db/models/sql/query.py | 1437 | 1515| 734 | 18831 | 78998 | 
| 33 | **10 django/db/models/expressions.py** | 856 | 917| 564 | 19395 | 89000 | 
| 34 | 10 django/db/models/query.py | 1108 | 1143| 314 | 19709 | 89000 | 
| 35 | 10 django/db/models/sql/query.py | 2163 | 2235| 774 | 20483 | 89000 | 
| 36 | 11 django/db/models/fields/related.py | 1235 | 1352| 963 | 21446 | 102861 | 
| 37 | 12 django/contrib/admin/templatetags/admin_list.py | 356 | 428| 631 | 22077 | 106690 | 
| 38 | 12 django/db/models/sql/query.py | 800 | 826| 280 | 22357 | 106690 | 
| 39 | 12 django/contrib/admin/views/main.py | 48 | 119| 636 | 22993 | 106690 | 
| 40 | 12 django/contrib/admin/views/main.py | 479 | 510| 224 | 23217 | 106690 | 
| 41 | 13 django/db/models/sql/where.py | 230 | 246| 130 | 23347 | 108494 | 
| 42 | 14 django/db/backends/postgresql/features.py | 1 | 81| 674 | 24021 | 109168 | 
| 43 | 14 django/db/models/sql/query.py | 766 | 798| 392 | 24413 | 109168 | 
| 44 | 15 django/db/backends/mysql/features.py | 1 | 101| 834 | 25247 | 110493 | 
| 45 | 16 django/db/backends/base/features.py | 1 | 115| 903 | 26150 | 113087 | 
| 46 | 16 django/db/models/query.py | 1061 | 1106| 349 | 26499 | 113087 | 
| 47 | 16 django/db/models/sql/query.py | 892 | 914| 248 | 26747 | 113087 | 
| 48 | 16 django/db/models/sql/where.py | 65 | 115| 396 | 27143 | 113087 | 
| 49 | 16 django/db/models/sql/query.py | 1341 | 1362| 250 | 27393 | 113087 | 
| 50 | 16 django/db/models/sql/query.py | 1408 | 1419| 137 | 27530 | 113087 | 
| 51 | 17 django/contrib/postgres/fields/ranges.py | 231 | 321| 489 | 28019 | 115204 | 
| 52 | 17 django/db/models/query.py | 1534 | 1590| 481 | 28500 | 115204 | 
| 53 | 18 django/db/backends/postgresql/introspection.py | 137 | 208| 751 | 29251 | 117443 | 
| 54 | 19 django/db/models/aggregates.py | 45 | 68| 294 | 29545 | 118744 | 
| 55 | 19 django/db/models/aggregates.py | 70 | 96| 266 | 29811 | 118744 | 
| 56 | 19 django/db/models/sql/where.py | 32 | 63| 317 | 30128 | 118744 | 
| 57 | 19 django/db/models/fields/related_lookups.py | 119 | 155| 244 | 30372 | 118744 | 
| 58 | 20 django/core/management/commands/inspectdb.py | 38 | 173| 1292 | 31664 | 121354 | 
| 59 | 20 django/db/backends/postgresql/operations.py | 165 | 207| 454 | 32118 | 121354 | 
| 60 | 20 django/db/models/sql/compiler.py | 1513 | 1553| 409 | 32527 | 121354 | 
| 61 | 21 django/db/models/sql/subqueries.py | 47 | 75| 210 | 32737 | 122567 | 
| **-> 62 <-** | **21 django/db/models/expressions.py** | 747 | 780| 284 | 33021 | 122567 | 
| 63 | 22 django/db/models/sql/datastructures.py | 117 | 137| 144 | 33165 | 123969 | 
| 64 | 22 django/db/models/sql/query.py | 240 | 286| 335 | 33500 | 123969 | 
| 65 | 23 django/db/migrations/autodetector.py | 525 | 671| 1109 | 34609 | 135704 | 
| 66 | **23 django/db/models/expressions.py** | 1125 | 1155| 218 | 34827 | 135704 | 
| 67 | **23 django/db/models/expressions.py** | 1048 | 1075| 261 | 35088 | 135704 | 
| 68 | 24 django/db/migrations/operations/models.py | 120 | 239| 827 | 35915 | 142400 | 
| 69 | 24 django/db/models/query.py | 1316 | 1364| 405 | 36320 | 142400 | 
| 70 | 24 django/db/models/sql/query.py | 2091 | 2127| 292 | 36612 | 142400 | 
| 71 | 24 django/db/models/sql/compiler.py | 143 | 191| 507 | 37119 | 142400 | 
| 72 | 24 django/db/migrations/autodetector.py | 707 | 794| 789 | 37908 | 142400 | 
| 73 | 25 django/db/backends/mysql/operations.py | 143 | 174| 294 | 38202 | 145731 | 
| 74 | 25 django/db/models/sql/query.py | 1 | 67| 471 | 38673 | 145731 | 
| 75 | 26 django/db/backends/oracle/operations.py | 21 | 73| 574 | 39247 | 151659 | 
| 76 | 26 django/db/backends/postgresql/operations.py | 41 | 85| 483 | 39730 | 151659 | 
| 77 | 26 django/db/models/sql/where.py | 157 | 190| 233 | 39963 | 151659 | 
| 78 | 26 django/contrib/admin/templatetags/admin_list.py | 105 | 194| 788 | 40751 | 151659 | 
| 79 | 27 django/db/backends/sqlite3/operations.py | 296 | 339| 431 | 41182 | 154555 | 
| 80 | 28 django/db/models/deletion.py | 1 | 76| 566 | 41748 | 158370 | 
| 81 | 28 django/db/models/sql/compiler.py | 1179 | 1200| 213 | 41961 | 158370 | 
| 82 | **28 django/db/models/expressions.py** | 1102 | 1123| 213 | 42174 | 158370 | 
| 83 | 28 django/contrib/admin/templatetags/admin_list.py | 214 | 298| 809 | 42983 | 158370 | 
| 84 | 28 django/contrib/admin/templatetags/admin_list.py | 431 | 489| 343 | 43326 | 158370 | 
| 85 | 28 django/db/models/sql/query.py | 1049 | 1074| 214 | 43540 | 158370 | 
| 86 | 28 django/db/migrations/autodetector.py | 264 | 335| 748 | 44288 | 158370 | 
| 87 | 28 django/db/models/query.py | 1450 | 1483| 297 | 44585 | 158370 | 
| 88 | 28 django/db/backends/mysql/operations.py | 297 | 315| 258 | 44843 | 158370 | 
| 89 | 28 django/db/models/sql/subqueries.py | 137 | 163| 173 | 45016 | 158370 | 
| 90 | 29 django/core/management/commands/squashmigrations.py | 136 | 200| 652 | 45668 | 160241 | 
| 91 | 29 django/db/models/sql/compiler.py | 958 | 979| 179 | 45847 | 160241 | 
| 92 | 30 django/contrib/postgres/search.py | 158 | 204| 432 | 46279 | 162131 | 
| 93 | 30 django/db/models/deletion.py | 269 | 344| 798 | 47077 | 162131 | 
| 94 | 30 django/db/backends/sqlite3/operations.py | 121 | 146| 279 | 47356 | 162131 | 
| 95 | 30 django/db/models/sql/compiler.py | 354 | 387| 388 | 47744 | 162131 | 
| 96 | 30 django/db/models/sql/datastructures.py | 59 | 102| 419 | 48163 | 162131 | 
| 97 | 30 django/db/models/sql/query.py | 621 | 645| 269 | 48432 | 162131 | 
| 98 | 31 django/db/migrations/operations/base.py | 1 | 102| 783 | 49215 | 163210 | 
| 99 | 31 django/db/models/sql/compiler.py | 1015 | 1055| 327 | 49542 | 163210 | 
| 100 | 31 django/contrib/admin/views/main.py | 332 | 392| 508 | 50050 | 163210 | 
| 101 | 31 django/db/models/sql/query.py | 415 | 508| 917 | 50967 | 163210 | 
| 102 | 32 django/db/backends/base/operations.py | 102 | 185| 718 | 51685 | 168790 | 
| 103 | 32 django/db/models/sql/query.py | 647 | 694| 511 | 52196 | 168790 | 
| 104 | 32 django/db/models/query.py | 1410 | 1448| 308 | 52504 | 168790 | 
| 105 | 32 django/db/models/sql/query.py | 546 | 620| 809 | 53313 | 168790 | 
| 106 | 32 django/core/management/commands/squashmigrations.py | 45 | 134| 791 | 54104 | 168790 | 
| 107 | 33 django/views/i18n.py | 88 | 191| 711 | 54815 | 171337 | 
| 108 | 33 django/db/backends/base/operations.py | 672 | 692| 187 | 55002 | 171337 | 
| **-> 109 <-** | **33 django/db/models/expressions.py** | 996 | 1045| 344 | 55346 | 171337 | 
| 110 | 33 django/db/models/sql/query.py | 510 | 544| 282 | 55628 | 171337 | 
| 111 | 33 django/contrib/postgres/fields/ranges.py | 199 | 228| 281 | 55909 | 171337 | 
| 112 | 33 django/db/models/query.py | 1162 | 1181| 209 | 56118 | 171337 | 
| 113 | 33 django/db/migrations/autodetector.py | 1146 | 1180| 296 | 56414 | 171337 | 
| 114 | 34 django/contrib/gis/db/models/lookups.py | 86 | 217| 762 | 57176 | 173950 | 
| 115 | 35 django/contrib/postgres/constraints.py | 1 | 51| 427 | 57603 | 174796 | 
| 116 | 35 django/db/models/sql/compiler.py | 424 | 477| 548 | 58151 | 174796 | 
| 117 | 35 django/db/models/query.py | 1843 | 1875| 314 | 58465 | 174796 | 
| 118 | 35 django/db/models/sql/subqueries.py | 1 | 44| 320 | 58785 | 174796 | 
| 119 | 36 django/db/models/manager.py | 1 | 162| 1223 | 60008 | 176230 | 
| 120 | 36 django/db/models/sql/query.py | 1886 | 1919| 279 | 60287 | 176230 | 
| 121 | 37 django/db/migrations/optimizer.py | 41 | 71| 249 | 60536 | 176826 | 
| 122 | 38 django/contrib/postgres/aggregates/statistics.py | 1 | 66| 419 | 60955 | 177245 | 
| 123 | 39 django/conf/locale/sr_Latn/formats.py | 5 | 40| 726 | 61681 | 178016 | 
| 124 | 39 django/db/backends/mysql/operations.py | 1 | 35| 282 | 61963 | 178016 | 
| 125 | 39 django/db/models/query.py | 184 | 241| 453 | 62416 | 178016 | 
| 126 | 39 django/contrib/admin/views/main.py | 289 | 330| 388 | 62804 | 178016 | 
| 127 | 39 django/db/models/query.py | 1731 | 1775| 439 | 63243 | 178016 | 
| 128 | 39 django/db/models/sql/compiler.py | 748 | 780| 332 | 63575 | 178016 | 
| 129 | 40 django/contrib/admin/options.py | 1740 | 1821| 744 | 64319 | 196504 | 
| 130 | 40 django/db/migrations/operations/models.py | 1 | 38| 238 | 64557 | 196504 | 
| 131 | 40 django/db/models/query.py | 1 | 41| 304 | 64861 | 196504 | 
| 132 | 40 django/db/migrations/operations/models.py | 609 | 622| 137 | 64998 | 196504 | 
| 133 | 40 django/db/models/sql/compiler.py | 1 | 18| 157 | 65155 | 196504 | 
| 134 | 40 django/db/models/base.py | 1865 | 1916| 351 | 65506 | 196504 | 
| 135 | 41 django/contrib/postgres/aggregates/mixins.py | 22 | 34| 128 | 65634 | 196946 | 
| 136 | 41 django/db/models/aggregates.py | 122 | 158| 245 | 65879 | 196946 | 
| 137 | **41 django/db/models/expressions.py** | 920 | 966| 377 | 66256 | 196946 | 
| 138 | 42 django/conf/locale/sr/formats.py | 5 | 40| 726 | 66982 | 197717 | 
| 139 | **42 django/db/models/expressions.py** | 436 | 471| 385 | 67367 | 197717 | 
| 140 | 42 django/contrib/postgres/aggregates/mixins.py | 36 | 49| 145 | 67512 | 197717 | 
| 141 | 42 django/db/backends/oracle/operations.py | 618 | 644| 303 | 67815 | 197717 | 
| 142 | 42 django/db/models/sql/compiler.py | 389 | 397| 119 | 67934 | 197717 | 
| 143 | 42 django/db/models/sql/compiler.py | 675 | 697| 186 | 68120 | 197717 | 
| 144 | 43 django/contrib/gis/db/models/__init__.py | 1 | 19| 204 | 68324 | 197921 | 
| 145 | 43 django/db/models/fields/related_lookups.py | 1 | 23| 170 | 68494 | 197921 | 


## Missing Patch Files

 * 1: django/db/backends/mysql/base.py
 * 2: django/db/backends/mysql/validation.py
 * 3: django/db/models/expressions.py

### Hint

```
Johannes, I need to repeat my gentle request for a queryset (see comment:2 and comment:6 ) Can you provide a queryset? It's really hard to restore the original queryset from a raw SQL.
I really try but without a queryset I was not able to reproduce this issue.
@felixxm, it seems that Subquery annotation are omitted from the grouping section in 3.0. I will try to create a test case today. If you add the following test to 2.2.* anywhere in tests.aggregation it will pass, in 3.0 it will fail, because the subquery is missing from the grouping bit: def test_filtered_aggregate_ref_subquery_annotation_e_31150(self): from django.db.models import OuterRef, Subquery aggs = Author.objects.annotate( earliest_book_year=Subquery( Book.objects.filter( contact__pk=OuterRef('pk'), ).order_by('pubdate').values('pubdate__year')[:1] ), ).annotate(max_id=Max('id')) print(aggs.query) self.assertIn(str(aggs.query), """ SELECT "aggregation_author"."id", "aggregation_author"."name", "aggregation_author"."age", (SELECT django_date_extract('year', U0."pubdate") FROM "aggregation_book" U0 WHERE U0."contact_id" = ("aggregation_author"."id") ORDER BY U0."pubdate" ASC LIMIT 1) AS "earliest_book_year", MAX("aggregation_author"."id") AS "max_id" FROM "aggregation_author" GROUP BY "aggregation_author"."id", "aggregation_author"."name", "aggregation_author"."age", (SELECT django_date_extract('year', U0."pubdate") FROM "aggregation_book" U0 WHERE U0."contact_id" = ("aggregation_author"."id") ORDER BY U0."pubdate" ASC LIMIT 1) """)
OK, thank you for the test case Joe. I added the import line so it pops straight into ,aggregation.tests.AggregateTestCase and then, yes, there's a change of behaviour between 2.2 and 3.0.
The generated query effectively changed but the returned resultset should be the same since the subquery is a function of the outer query's primary key and the query is grouped by the outer query primary key. I don't think asserting against the generated query string is a proper test case; the generated SQL will change between Django versions but the returned result set should be the same.
Yes. I didn’t take the test case as final. Just illustrating the issue. 😀 If it’s equivalent then there’s no issue, but I’m assuming Joe is seeing what he thinks is a substantive regression?
Replying to Simon Charette: The generated query effectively changed but the returned resultset should be the same since the subquery is a function of the outer query's primary key and the query is grouped by the outer query primary key. I came from a wrong result set and deduced the error from here on out. It took me a while to figure it out to, but sadly grouping by the outer reference is not the same as grouping by the query. Here an example: with t_w_douplicates as ( select abs(n) as pk, n as val from generate_series(-2, 2, 1) n ) select pk, ( select n from generate_series(2, 2) n where n = val ) from t_w_douplicates where ( select n from generate_series(2, 2) n where n = val ) is null GROUP BY pk, ( select n from generate_series(2, 2) n where n = val ); Which returns (0, null) (1, null) (2, null) And just using the outer ref: with t_w_douplicates as ( select abs(n) as pk, n as val from generate_series(-2, 2, 1) n ) select pk, ( select n from generate_series(2, 2) n where n = val ) from t_w_douplicates where ( select n from generate_series(2, 2) n where n = val ) is null GROUP BY pk, val; Which returns 4 results: (2, null) (1, null) (0, null) (1, null) I don't think asserting against the generated query string is a proper test case; the generated SQL will change between Django versions but the returned result set should be the same. Certainly, haha, that was just to illustrate how to get from the ORM to the SQL query in the description. The SQL sample now, shows how they produce different result sets.
Thanks for the extra details, I think I have a better picture of what's happening here. In the cases where an outer query spawns a multi-valued relationship and the subquery references one of these multi-valued relationship we must keep grouping by the subquery. In your original reports the INNER JOIN "camps_bookingoption" T4 joins spawns multiple rows for the same camps.Offer and then your subquery filter against (T4."position"). In ORM terms that means we must keep returning self in Subquery.get_group_by_cols when any of our OuterRef include a __ which could point to a multi-valued relationship. We could go a bit further and only disable the optimization if any of the outer-ref in the __ chain is multi-valued by relying on getattr(rel, 'many_to_many', True) introspection but in all cases that means adding a way to taint Col instances with whether or not they are/could be multi-valued so get_group_by_cols can return self if any of get_external_cols is multi-valued I could see this tainting as being useful to warn about #10060 in the future since the logic could be based on this column or alias tainting. I'm happy to submit a patch or discuss any alternatives but it feels like the above solution solves the reported problem while maintaining the optimization for most of the cases.
I give a quick shot at the above solution and came with this rough patch ​https://github.com/django/django/compare/master...charettes:ticket-31150 Johannes, could you give it a shot and report whether or not it addresses your problem? Do you confirm your original queryset was using an OuterRef with a reference to an outer-query multi-valued relationship?
I tested Simon's patch applied to 3.0.3 and it now works as expected. So yes, I'd say go ahead!
```

## Patch

```diff
diff --git a/django/db/backends/mysql/base.py b/django/db/backends/mysql/base.py
--- a/django/db/backends/mysql/base.py
+++ b/django/db/backends/mysql/base.py
@@ -364,3 +364,10 @@ def mysql_version(self):
     @cached_property
     def mysql_is_mariadb(self):
         return 'mariadb' in self.mysql_server_info.lower()
+
+    @cached_property
+    def sql_mode(self):
+        with self.cursor() as cursor:
+            cursor.execute('SELECT @@sql_mode')
+            sql_mode = cursor.fetchone()
+        return set(sql_mode[0].split(',') if sql_mode else ())
diff --git a/django/db/backends/mysql/validation.py b/django/db/backends/mysql/validation.py
--- a/django/db/backends/mysql/validation.py
+++ b/django/db/backends/mysql/validation.py
@@ -10,11 +10,7 @@ def check(self, **kwargs):
         return issues
 
     def _check_sql_mode(self, **kwargs):
-        with self.connection.cursor() as cursor:
-            cursor.execute("SELECT @@sql_mode")
-            sql_mode = cursor.fetchone()
-        modes = set(sql_mode[0].split(',') if sql_mode else ())
-        if not (modes & {'STRICT_TRANS_TABLES', 'STRICT_ALL_TABLES'}):
+        if not (self.connection.sql_mode & {'STRICT_TRANS_TABLES', 'STRICT_ALL_TABLES'}):
             return [checks.Warning(
                 "MySQL Strict Mode is not set for database connection '%s'" % self.connection.alias,
                 hint="MySQL's Strict Mode fixes many data integrity problems in MySQL, "
diff --git a/django/db/models/expressions.py b/django/db/models/expressions.py
--- a/django/db/models/expressions.py
+++ b/django/db/models/expressions.py
@@ -6,6 +6,7 @@
 from django.core.exceptions import EmptyResultSet, FieldError
 from django.db import NotSupportedError, connection
 from django.db.models import fields
+from django.db.models.constants import LOOKUP_SEP
 from django.db.models.query_utils import Q
 from django.utils.deconstruct import deconstructible
 from django.utils.functional import cached_property
@@ -559,6 +560,14 @@ def as_sql(self, *args, **kwargs):
             'only be used in a subquery.'
         )
 
+    def resolve_expression(self, *args, **kwargs):
+        col = super().resolve_expression(*args, **kwargs)
+        # FIXME: Rename possibly_multivalued to multivalued and fix detection
+        # for non-multivalued JOINs (e.g. foreign key fields). This should take
+        # into account only many-to-many and one-to-many relationships.
+        col.possibly_multivalued = LOOKUP_SEP in self.name
+        return col
+
     def relabeled_clone(self, relabels):
         return self
 
@@ -747,6 +756,7 @@ def as_sql(self, compiler, connection):
 class Col(Expression):
 
     contains_column_references = True
+    possibly_multivalued = False
 
     def __init__(self, alias, target, output_field=None):
         if output_field is None:
@@ -1042,7 +1052,10 @@ def as_sql(self, compiler, connection, template=None, **extra_context):
     def get_group_by_cols(self, alias=None):
         if alias:
             return [Ref(alias, self)]
-        return self.query.get_external_cols()
+        external_cols = self.query.get_external_cols()
+        if any(col.possibly_multivalued for col in external_cols):
+            return [self]
+        return external_cols
 
 
 class Exists(Subquery):

```

## Test Patch

```diff
diff --git a/tests/aggregation/tests.py b/tests/aggregation/tests.py
--- a/tests/aggregation/tests.py
+++ b/tests/aggregation/tests.py
@@ -1,6 +1,7 @@
 import datetime
 import re
 from decimal import Decimal
+from unittest import skipIf
 
 from django.core.exceptions import FieldError
 from django.db import connection
@@ -1190,6 +1191,26 @@ def test_aggregation_subquery_annotation_values(self):
             },
         ])
 
+    @skipUnlessDBFeature('supports_subqueries_in_group_by')
+    @skipIf(
+        connection.vendor == 'mysql' and 'ONLY_FULL_GROUP_BY' in connection.sql_mode,
+        'GROUP BY optimization does not work properly when ONLY_FULL_GROUP_BY '
+        'mode is enabled on MySQL, see #31331.',
+    )
+    def test_aggregation_subquery_annotation_multivalued(self):
+        """
+        Subquery annotations must be included in the GROUP BY if they use
+        potentially multivalued relations (contain the LOOKUP_SEP).
+        """
+        subquery_qs = Author.objects.filter(
+            pk=OuterRef('pk'),
+            book__name=OuterRef('book__name'),
+        ).values('pk')
+        author_qs = Author.objects.annotate(
+            subquery_id=Subquery(subquery_qs),
+        ).annotate(count=Count('book'))
+        self.assertEqual(author_qs.count(), Author.objects.count())
+
     def test_aggregation_order_by_not_selected_annotation_values(self):
         result_asc = [
             self.b4.pk,
@@ -1248,6 +1269,7 @@ def test_group_by_exists_annotation(self):
         ).annotate(total=Count('*'))
         self.assertEqual(dict(has_long_books_breakdown), {True: 2, False: 3})
 
+    @skipUnlessDBFeature('supports_subqueries_in_group_by')
     def test_aggregation_subquery_annotation_related_field(self):
         publisher = Publisher.objects.create(name=self.a9.name, num_awards=2)
         book = Book.objects.create(
@@ -1267,3 +1289,8 @@ def test_aggregation_subquery_annotation_related_field(self):
             contact_publisher__isnull=False,
         ).annotate(count=Count('authors'))
         self.assertSequenceEqual(books_qs, [book])
+        # FIXME: GROUP BY doesn't need to include a subquery with
+        # non-multivalued JOINs, see Col.possibly_multivalued (refs #31150):
+        # with self.assertNumQueries(1) as ctx:
+        #     self.assertSequenceEqual(books_qs, [book])
+        # self.assertEqual(ctx[0]['sql'].count('SELECT'), 2)
diff --git a/tests/check_framework/test_database.py b/tests/check_framework/test_database.py
--- a/tests/check_framework/test_database.py
+++ b/tests/check_framework/test_database.py
@@ -2,7 +2,7 @@
 from unittest import mock
 
 from django.core.checks.database import check_database_backends
-from django.db import connection
+from django.db import connection, connections
 from django.test import TestCase
 
 
@@ -18,6 +18,12 @@ def test_database_checks_called(self, mocked_check):
 
     @unittest.skipUnless(connection.vendor == 'mysql', 'Test only for MySQL')
     def test_mysql_strict_mode(self):
+        def _clean_sql_mode():
+            for alias in self.databases:
+                if hasattr(connections[alias], 'sql_mode'):
+                    del connections[alias].sql_mode
+
+        _clean_sql_mode()
         good_sql_modes = [
             'STRICT_TRANS_TABLES,STRICT_ALL_TABLES',
             'STRICT_TRANS_TABLES',
@@ -29,6 +35,7 @@ def test_mysql_strict_mode(self):
                 return_value=(response,)
             ):
                 self.assertEqual(check_database_backends(databases=self.databases), [])
+            _clean_sql_mode()
 
         bad_sql_modes = ['', 'WHATEVER']
         for response in bad_sql_modes:
@@ -40,3 +47,4 @@ def test_mysql_strict_mode(self):
                 result = check_database_backends(databases=self.databases)
                 self.assertEqual(len(result), 2)
                 self.assertEqual([r.id for r in result], ['mysql.W002', 'mysql.W002'])
+            _clean_sql_mode()

```


## Code snippets

### 1 - django/db/models/sql/query.py:

Start line: 1281, End line: 1339

```python
class Query(BaseExpression):

    def build_filter(self, filter_expr, branch_negated=False, current_negated=False,
                     can_reuse=None, allow_joins=True, split_subq=True,
                     reuse_with_filtered_relation=False, check_filterable=True):
        # ... other code

        try:
            join_info = self.setup_joins(
                parts, opts, alias, can_reuse=can_reuse, allow_many=allow_many,
                reuse_with_filtered_relation=reuse_with_filtered_relation,
            )

            # Prevent iterator from being consumed by check_related_objects()
            if isinstance(value, Iterator):
                value = list(value)
            self.check_related_objects(join_info.final_field, value, join_info.opts)

            # split_exclude() needs to know which joins were generated for the
            # lookup parts
            self._lookup_joins = join_info.joins
        except MultiJoin as e:
            return self.split_exclude(filter_expr, can_reuse, e.names_with_path)

        # Update used_joins before trimming since they are reused to determine
        # which joins could be later promoted to INNER.
        used_joins.update(join_info.joins)
        targets, alias, join_list = self.trim_joins(join_info.targets, join_info.joins, join_info.path)
        if can_reuse is not None:
            can_reuse.update(join_list)

        if join_info.final_field.is_relation:
            # No support for transforms for relational fields
            num_lookups = len(lookups)
            if num_lookups > 1:
                raise FieldError('Related Field got invalid lookup: {}'.format(lookups[0]))
            if len(targets) == 1:
                col = self._get_col(targets[0], join_info.final_field, alias)
            else:
                col = MultiColSource(alias, targets, join_info.targets, join_info.final_field)
        else:
            col = self._get_col(targets[0], join_info.final_field, alias)

        condition = self.build_lookup(lookups, col, value)
        lookup_type = condition.lookup_name
        clause.add(condition, AND)

        require_outer = lookup_type == 'isnull' and condition.rhs is True and not current_negated
        if current_negated and (lookup_type != 'isnull' or condition.rhs is False) and condition.rhs is not None:
            require_outer = True
            if (lookup_type != 'isnull' and (
                    self.is_nullable(targets[0]) or
                    self.alias_map[join_list[-1]].join_type == LOUTER)):
                # The condition added here will be SQL like this:
                # NOT (col IS NOT NULL), where the first NOT is added in
                # upper layers of code. The reason for addition is that if col
                # is null, then col != someval will result in SQL "unknown"
                # which isn't the same as in Python. The Python None handling
                # is wanted, and it can be gotten by
                # (col IS NULL OR col != someval)
                #   <=>
                # NOT (col IS NOT NULL AND col = someval).
                lookup_class = targets[0].get_lookup('isnull')
                col = self._get_col(targets[0], join_info.targets[0], alias)
                clause.add(lookup_class(col, False), AND)
        return clause, used_joins if not require_outer else ()
```
### 2 - django/db/models/sql/compiler.py:

Start line: 58, End line: 141

```python
class SQLCompiler:

    def get_group_by(self, select, order_by):
        """
        Return a list of 2-tuples of form (sql, params).

        The logic of what exactly the GROUP BY clause contains is hard
        to describe in other words than "if it passes the test suite,
        then it is correct".
        """
        # Some examples:
        #     SomeModel.objects.annotate(Count('somecol'))
        #     GROUP BY: all fields of the model
        #
        #    SomeModel.objects.values('name').annotate(Count('somecol'))
        #    GROUP BY: name
        #
        #    SomeModel.objects.annotate(Count('somecol')).values('name')
        #    GROUP BY: all cols of the model
        #
        #    SomeModel.objects.values('name', 'pk').annotate(Count('somecol')).values('pk')
        #    GROUP BY: name, pk
        #
        #    SomeModel.objects.values('name').annotate(Count('somecol')).values('pk')
        #    GROUP BY: name, pk
        #
        # In fact, the self.query.group_by is the minimal set to GROUP BY. It
        # can't be ever restricted to a smaller set, but additional columns in
        # HAVING, ORDER BY, and SELECT clauses are added to it. Unfortunately
        # the end result is that it is impossible to force the query to have
        # a chosen GROUP BY clause - you can almost do this by using the form:
        #     .values(*wanted_cols).annotate(AnAggregate())
        # but any later annotations, extra selects, values calls that
        # refer some column outside of the wanted_cols, order_by, or even
        # filter calls can alter the GROUP BY clause.

        # The query.group_by is either None (no GROUP BY at all), True
        # (group by select fields), or a list of expressions to be added
        # to the group by.
        if self.query.group_by is None:
            return []
        expressions = []
        if self.query.group_by is not True:
            # If the group by is set to a list (by .values() call most likely),
            # then we need to add everything in it to the GROUP BY clause.
            # Backwards compatibility hack for setting query.group_by. Remove
            # when  we have public API way of forcing the GROUP BY clause.
            # Converts string references to expressions.
            for expr in self.query.group_by:
                if not hasattr(expr, 'as_sql'):
                    expressions.append(self.query.resolve_ref(expr))
                else:
                    expressions.append(expr)
        # Note that even if the group_by is set, it is only the minimal
        # set to group by. So, we need to add cols in select, order_by, and
        # having into the select in any case.
        ref_sources = {
            expr.source for expr in expressions if isinstance(expr, Ref)
        }
        for expr, _, _ in select:
            # Skip members of the select clause that are already included
            # by reference.
            if expr in ref_sources:
                continue
            cols = expr.get_group_by_cols()
            for col in cols:
                expressions.append(col)
        for expr, (sql, params, is_ref) in order_by:
            # Skip References to the select clause, as all expressions in the
            # select clause are already part of the group by.
            if not is_ref:
                expressions.extend(expr.get_group_by_cols())
        having_group_by = self.having.get_group_by_cols() if self.having else ()
        for expr in having_group_by:
            expressions.append(expr)
        result = []
        seen = set()
        expressions = self.collapse_group_by(expressions, having_group_by)

        for expr in expressions:
            sql, params = self.compile(expr)
            params_hash = make_hashable(params)
            if (sql, params_hash) not in seen:
                result.append((sql, params))
                seen.add((sql, params_hash))
        return result
```
### 3 - django/db/models/sql/query.py:

Start line: 363, End line: 413

```python
class Query(BaseExpression):

    def rewrite_cols(self, annotation, col_cnt):
        # We must make sure the inner query has the referred columns in it.
        # If we are aggregating over an annotation, then Django uses Ref()
        # instances to note this. However, if we are annotating over a column
        # of a related model, then it might be that column isn't part of the
        # SELECT clause of the inner query, and we must manually make sure
        # the column is selected. An example case is:
        orig_exprs = annotation.get_source_expressions()
        new_exprs = []
        for expr in orig_exprs:
            # FIXME: These conditions are fairly arbitrary. Identify a better
            # method of having expressions decide which code path they should
            # take.
            if isinstance(expr, Ref):
                # Its already a Ref to subquery (see resolve_ref() for
                # details)
                new_exprs.append(expr)
            elif isinstance(expr, (WhereNode, Lookup)):
                # Decompose the subexpressions further. The code here is
                # copied from the else clause, but this condition must appear
                # before the contains_aggregate/is_summary condition below.
                new_expr, col_cnt = self.rewrite_cols(expr, col_cnt)
                new_exprs.append(new_expr)
            else:
                # Reuse aliases of expressions already selected in subquery.
                for col_alias, selected_annotation in self.annotation_select.items():
                    if selected_annotation == expr:
                        new_expr = Ref(col_alias, expr)
                        break
                else:
                    # An expression that is not selected the subquery.
                    if isinstance(expr, Col) or (expr.contains_aggregate and not expr.is_summary):
                        # Reference column or another aggregate. Select it
                        # under a non-conflicting alias.
                        col_cnt += 1
                        col_alias = '__col%d' % col_cnt
                        self.annotations[col_alias] = expr
                        self.append_annotation_mask([col_alias])
                        new_expr = Ref(col_alias, expr)
                    else:
                        # Some other expression not referencing database values
                        # directly. Its subexpression might contain Cols.
                        new_expr, col_cnt = self.rewrite_cols(expr, col_cnt)
                new_exprs.append(new_expr)
        annotation.set_source_expressions(new_exprs)
        return annotation, col_cnt
```
### 4 - django/db/models/sql/query.py:

Start line: 2317, End line: 2372

```python
class JoinPromoter:

    def update_join_types(self, query):
        """
        Change join types so that the generated query is as efficient as
        possible, but still correct. So, change as many joins as possible
        to INNER, but don't make OUTER joins INNER if that could remove
        results from the query.
        """
        to_promote = set()
        to_demote = set()
        # The effective_connector is used so that NOT (a AND b) is treated
        # similarly to (a OR b) for join promotion.
        for table, votes in self.votes.items():
            # We must use outer joins in OR case when the join isn't contained
            # in all of the joins. Otherwise the INNER JOIN itself could remove
            # valid results. Consider the case where a model with rel_a and
            # rel_b relations is queried with rel_a__col=1 | rel_b__col=2. Now,
            # if rel_a join doesn't produce any results is null (for example
            # reverse foreign key or null value in direct foreign key), and
            # there is a matching row in rel_b with col=2, then an INNER join
            # to rel_a would remove a valid match from the query. So, we need
            # to promote any existing INNER to LOUTER (it is possible this
            # promotion in turn will be demoted later on).
            if self.effective_connector == 'OR' and votes < self.num_children:
                to_promote.add(table)
            # If connector is AND and there is a filter that can match only
            # when there is a joinable row, then use INNER. For example, in
            # rel_a__col=1 & rel_b__col=2, if either of the rels produce NULL
            # as join output, then the col=1 or col=2 can't match (as
            # NULL=anything is always false).
            # For the OR case, if all children voted for a join to be inner,
            # then we can use INNER for the join. For example:
            #     (rel_a__col__icontains=Alex | rel_a__col__icontains=Russell)
            # then if rel_a doesn't produce any rows, the whole condition
            # can't match. Hence we can safely use INNER join.
            if self.effective_connector == 'AND' or (
                    self.effective_connector == 'OR' and votes == self.num_children):
                to_demote.add(table)
            # Finally, what happens in cases where we have:
            #    (rel_a__col=1|rel_b__col=2) & rel_a__col__gte=0
            # Now, we first generate the OR clause, and promote joins for it
            # in the first if branch above. Both rel_a and rel_b are promoted
            # to LOUTER joins. After that we do the AND case. The OR case
            # voted no inner joins but the rel_a__col__gte=0 votes inner join
            # for rel_a. We demote it back to INNER join (in AND case a single
            # vote is enough). The demotion is OK, if rel_a doesn't produce
            # rows, then the rel_a__col__gte=0 clause can't be true, and thus
            # the whole clause must be false. So, it is safe to use INNER
            # join.
            # Note that in this example we could just as well have the __gte
            # clause and the OR clause swapped. Or we could replace the __gte
            # clause with an OR clause containing rel_a__col=1|rel_a__col=2,
            # and again we could safely demote to INNER.
        query.promote_joins(to_promote)
        query.demote_joins(to_demote)
        return to_demote
```
### 5 - django/db/models/sql/query.py:

Start line: 1695, End line: 1766

```python
class Query(BaseExpression):

    def split_exclude(self, filter_expr, can_reuse, names_with_path):
        """
        When doing an exclude against any kind of N-to-many relation, we need
        to use a subquery. This method constructs the nested query, given the
        original exclude filter (filter_expr) and the portion up to the first
        N-to-many relation field.

        For example, if the origin filter is ~Q(child__name='foo'), filter_expr
        is ('child__name', 'foo') and can_reuse is a set of joins usable for
        filters in the original query.

        We will turn this into equivalent of:
            WHERE NOT (pk IN (SELECT parent_id FROM thetable
                              WHERE name = 'foo' AND parent_id IS NOT NULL))

        It might be worth it to consider using WHERE NOT EXISTS as that has
        saner null handling, and is easier for the backend's optimizer to
        handle.
        """
        filter_lhs, filter_rhs = filter_expr
        if isinstance(filter_rhs, OuterRef):
            filter_expr = (filter_lhs, OuterRef(filter_rhs))
        elif isinstance(filter_rhs, F):
            filter_expr = (filter_lhs, OuterRef(filter_rhs.name))
        # Generate the inner query.
        query = Query(self.model)
        query._filtered_relations = self._filtered_relations
        query.add_filter(filter_expr)
        query.clear_ordering(True)
        # Try to have as simple as possible subquery -> trim leading joins from
        # the subquery.
        trimmed_prefix, contains_louter = query.trim_start(names_with_path)

        # Add extra check to make sure the selected field will not be null
        # since we are adding an IN <subquery> clause. This prevents the
        # database from tripping over IN (...,NULL,...) selects and returning
        # nothing
        col = query.select[0]
        select_field = col.target
        alias = col.alias
        if self.is_nullable(select_field):
            lookup_class = select_field.get_lookup('isnull')
            lookup = lookup_class(select_field.get_col(alias), False)
            query.where.add(lookup, AND)
        if alias in can_reuse:
            pk = select_field.model._meta.pk
            # Need to add a restriction so that outer query's filters are in effect for
            # the subquery, too.
            query.bump_prefix(self)
            lookup_class = select_field.get_lookup('exact')
            # Note that the query.select[0].alias is different from alias
            # due to bump_prefix above.
            lookup = lookup_class(pk.get_col(query.select[0].alias),
                                  pk.get_col(alias))
            query.where.add(lookup, AND)
            query.external_aliases[alias] = True

        condition, needed_inner = self.build_filter(
            ('%s__in' % trimmed_prefix, query),
            current_negated=True, branch_negated=True, can_reuse=can_reuse)
        if contains_louter:
            or_null_condition, _ = self.build_filter(
                ('%s__isnull' % trimmed_prefix, True),
                current_negated=True, branch_negated=True, can_reuse=can_reuse)
            condition.add(or_null_condition, OR)
            # Note that the end result will be:
            # (outercol NOT IN innerq AND outercol IS NOT NULL) OR outercol IS NULL.
            # This might look crazy but due to how IN works, this seems to be
            # correct. If the IS NOT NULL check is removed then outercol NOT
            # IN will return UNKNOWN. If the IS NULL check is removed, then if
            # outercol IS NULL we will not match the row.
        return condition, needed_inner
```
### 6 - django/db/models/query.py:

Start line: 790, End line: 829

```python
class QuerySet:
    _update.alters_data = True
    _update.queryset_only = False

    def exists(self):
        if self._result_cache is None:
            return self.query.has_results(using=self.db)
        return bool(self._result_cache)

    def _prefetch_related_objects(self):
        # This method can only be called once the result cache has been filled.
        prefetch_related_objects(self._result_cache, *self._prefetch_related_lookups)
        self._prefetch_done = True

    def explain(self, *, format=None, **options):
        return self.query.explain(using=self.db, format=format, **options)

    ##################################################
    # PUBLIC METHODS THAT RETURN A QUERYSET SUBCLASS #
    ##################################################

    def raw(self, raw_query, params=None, translations=None, using=None):
        if using is None:
            using = self.db
        qs = RawQuerySet(raw_query, model=self.model, params=params, translations=translations, using=using)
        qs._prefetch_related_lookups = self._prefetch_related_lookups[:]
        return qs

    def _values(self, *fields, **expressions):
        clone = self._chain()
        if expressions:
            clone = clone.annotate(**expressions)
        clone._fields = fields
        clone.query.set_values(fields)
        return clone

    def values(self, *fields, **expressions):
        fields += tuple(expressions)
        clone = self._values(*fields, **expressions)
        clone._iterable_class = ValuesIterable
        return clone
```
### 7 - django/db/models/sql/query.py:

Start line: 1921, End line: 1963

```python
class Query(BaseExpression):

    def set_group_by(self, allow_aliases=True):
        """
        Expand the GROUP BY clause required by the query.

        This will usually be the set of all non-aggregate fields in the
        return data. If the database backend supports grouping by the
        primary key, and the query would be equivalent, the optimization
        will be made automatically.
        """
        group_by = list(self.select)
        if self.annotation_select:
            for alias, annotation in self.annotation_select.items():
                signature = inspect.signature(annotation.get_group_by_cols)
                if 'alias' not in signature.parameters:
                    annotation_class = annotation.__class__
                    msg = (
                        '`alias=None` must be added to the signature of '
                        '%s.%s.get_group_by_cols().'
                    ) % (annotation_class.__module__, annotation_class.__qualname__)
                    warnings.warn(msg, category=RemovedInDjango40Warning)
                    group_by_cols = annotation.get_group_by_cols()
                else:
                    if not allow_aliases:
                        alias = None
                    group_by_cols = annotation.get_group_by_cols(alias=alias)
                group_by.extend(group_by_cols)
        self.group_by = tuple(group_by)

    def add_select_related(self, fields):
        """
        Set up the select_related data structure so that we only select
        certain related models (as opposed to all models, when
        self.select_related=True).
        """
        if isinstance(self.select_related, bool):
            field_dict = {}
        else:
            field_dict = self.select_related
        for field in fields:
            d = field_dict
            for part in field.split(LOOKUP_SEP):
                d = d.setdefault(part, {})
        self.select_related = field_dict
```
### 8 - django/db/models/query.py:

Start line: 243, End line: 269

```python
class QuerySet:

    def __setstate__(self, state):
        msg = None
        pickled_version = state.get(DJANGO_VERSION_PICKLE_KEY)
        if pickled_version:
            current_version = get_version()
            if current_version != pickled_version:
                msg = (
                    "Pickled queryset instance's Django version %s does not "
                    "match the current version %s." % (pickled_version, current_version)
                )
        else:
            msg = "Pickled queryset instance's Django version is not specified."

        if msg:
            warnings.warn(msg, RuntimeWarning, stacklevel=2)

        self.__dict__.update(state)

    def __repr__(self):
        data = list(self[:REPR_OUTPUT_SIZE + 1])
        if len(data) > REPR_OUTPUT_SIZE:
            data[-1] = "...(remaining elements truncated)..."
        return '<%s %r>' % (self.__class__.__name__, data)

    def __len__(self):
        self._fetch_all()
        return len(self._result_cache)
```
### 9 - django/db/models/sql/compiler.py:

Start line: 193, End line: 263

```python
class SQLCompiler:

    def get_select(self):
        """
        Return three values:
        - a list of 3-tuples of (expression, (sql, params), alias)
        - a klass_info structure,
        - a dictionary of annotations

        The (sql, params) is what the expression will produce, and alias is the
        "AS alias" for the column (possibly None).

        The klass_info structure contains the following information:
        - The base model of the query.
        - Which columns for that model are present in the query (by
          position of the select clause).
        - related_klass_infos: [f, klass_info] to descent into

        The annotations is a dictionary of {'attname': column position} values.
        """
        select = []
        klass_info = None
        annotations = {}
        select_idx = 0
        for alias, (sql, params) in self.query.extra_select.items():
            annotations[alias] = select_idx
            select.append((RawSQL(sql, params), alias))
            select_idx += 1
        assert not (self.query.select and self.query.default_cols)
        if self.query.default_cols:
            cols = self.get_default_columns()
        else:
            # self.query.select is a special case. These columns never go to
            # any model.
            cols = self.query.select
        if cols:
            select_list = []
            for col in cols:
                select_list.append(select_idx)
                select.append((col, None))
                select_idx += 1
            klass_info = {
                'model': self.query.model,
                'select_fields': select_list,
            }
        for alias, annotation in self.query.annotation_select.items():
            annotations[alias] = select_idx
            select.append((annotation, alias))
            select_idx += 1

        if self.query.select_related:
            related_klass_infos = self.get_related_selections(select)
            klass_info['related_klass_infos'] = related_klass_infos

            def get_select_from_parent(klass_info):
                for ki in klass_info['related_klass_infos']:
                    if ki['from_parent']:
                        ki['select_fields'] = (klass_info['select_fields'] +
                                               ki['select_fields'])
                    get_select_from_parent(ki)
            get_select_from_parent(klass_info)

        ret = []
        for col, alias in select:
            try:
                sql, params = self.compile(col)
            except EmptyResultSet:
                # Select a predicate that's always False.
                sql, params = '0', ()
            else:
                sql, params = col.select_format(self, sql, params)
            ret.append((col, (sql, params), alias))
        return ret, klass_info, annotations
```
### 10 - django/db/models/__init__.py:

Start line: 1, End line: 52

```python
from django.core.exceptions import ObjectDoesNotExist
from django.db.models import signals
from django.db.models.aggregates import *  # NOQA
from django.db.models.aggregates import __all__ as aggregates_all
from django.db.models.constraints import *  # NOQA
from django.db.models.constraints import __all__ as constraints_all
from django.db.models.deletion import (
    CASCADE, DO_NOTHING, PROTECT, RESTRICT, SET, SET_DEFAULT, SET_NULL,
    ProtectedError, RestrictedError,
)
from django.db.models.enums import *  # NOQA
from django.db.models.enums import __all__ as enums_all
from django.db.models.expressions import (
    Case, Exists, Expression, ExpressionList, ExpressionWrapper, F, Func,
    OrderBy, OuterRef, RowRange, Subquery, Value, ValueRange, When, Window,
    WindowFrame,
)
from django.db.models.fields import *  # NOQA
from django.db.models.fields import __all__ as fields_all
from django.db.models.fields.files import FileField, ImageField
from django.db.models.fields.proxy import OrderWrt
from django.db.models.indexes import *  # NOQA
from django.db.models.indexes import __all__ as indexes_all
from django.db.models.lookups import Lookup, Transform
from django.db.models.manager import Manager
from django.db.models.query import Prefetch, QuerySet, prefetch_related_objects
from django.db.models.query_utils import FilteredRelation, Q

# Imports that would create circular imports if sorted
from django.db.models.base import DEFERRED, Model  # isort:skip
from django.db.models.fields.related import (  # isort:skip
    ForeignKey, ForeignObject, OneToOneField, ManyToManyField,
    ForeignObjectRel, ManyToOneRel, ManyToManyRel, OneToOneRel,
)


__all__ = aggregates_all + constraints_all + enums_all + fields_all + indexes_all
__all__ += [
    'ObjectDoesNotExist', 'signals',
    'CASCADE', 'DO_NOTHING', 'PROTECT', 'RESTRICT', 'SET', 'SET_DEFAULT',
    'SET_NULL', 'ProtectedError', 'RestrictedError',
    'Case', 'Exists', 'Expression', 'ExpressionList', 'ExpressionWrapper', 'F',
    'Func', 'OrderBy', 'OuterRef', 'RowRange', 'Subquery', 'Value',
    'ValueRange', 'When',
    'Window', 'WindowFrame',
    'FileField', 'ImageField', 'OrderWrt', 'Lookup', 'Transform', 'Manager',
    'Prefetch', 'Q', 'QuerySet', 'prefetch_related_objects', 'DEFERRED', 'Model',
    'FilteredRelation',
    'ForeignKey', 'ForeignObject', 'OneToOneField', 'ManyToManyField',
    'ForeignObjectRel', 'ManyToOneRel', 'ManyToManyRel', 'OneToOneRel',
]
```
### 33 - django/db/models/expressions.py:

Start line: 856, End line: 917

```python
class When(Expression):
    template = 'WHEN %(condition)s THEN %(result)s'
    # This isn't a complete conditional expression, must be used in Case().
    conditional = False

    def __init__(self, condition=None, then=None, **lookups):
        if lookups and condition is None:
            condition, lookups = Q(**lookups), None
        if condition is None or not getattr(condition, 'conditional', False) or lookups:
            raise TypeError(
                'When() supports a Q object, a boolean expression, or lookups '
                'as a condition.'
            )
        if isinstance(condition, Q) and not condition:
            raise ValueError("An empty Q() can't be used as a When() condition.")
        super().__init__(output_field=None)
        self.condition = condition
        self.result = self._parse_expressions(then)[0]

    def __str__(self):
        return "WHEN %r THEN %r" % (self.condition, self.result)

    def __repr__(self):
        return "<%s: %s>" % (self.__class__.__name__, self)

    def get_source_expressions(self):
        return [self.condition, self.result]

    def set_source_expressions(self, exprs):
        self.condition, self.result = exprs

    def get_source_fields(self):
        # We're only interested in the fields of the result expressions.
        return [self.result._output_field_or_none]

    def resolve_expression(self, query=None, allow_joins=True, reuse=None, summarize=False, for_save=False):
        c = self.copy()
        c.is_summary = summarize
        if hasattr(c.condition, 'resolve_expression'):
            c.condition = c.condition.resolve_expression(query, allow_joins, reuse, summarize, False)
        c.result = c.result.resolve_expression(query, allow_joins, reuse, summarize, for_save)
        return c

    def as_sql(self, compiler, connection, template=None, **extra_context):
        connection.ops.check_expression_support(self)
        template_params = extra_context
        sql_params = []
        condition_sql, condition_params = compiler.compile(self.condition)
        template_params['condition'] = condition_sql
        sql_params.extend(condition_params)
        result_sql, result_params = compiler.compile(self.result)
        template_params['result'] = result_sql
        sql_params.extend(result_params)
        template = template or self.template
        return template % template_params, sql_params

    def get_group_by_cols(self, alias=None):
        # This is not a complete expression and cannot be used in GROUP BY.
        cols = []
        for source in self.get_source_expressions():
            cols.extend(source.get_group_by_cols())
        return cols
```
### 62 - django/db/models/expressions.py:

Start line: 747, End line: 780

```python
class Col(Expression):

    contains_column_references = True

    def __init__(self, alias, target, output_field=None):
        if output_field is None:
            output_field = target
        super().__init__(output_field=output_field)
        self.alias, self.target = alias, target

    def __repr__(self):
        alias, target = self.alias, self.target
        identifiers = (alias, str(target)) if alias else (str(target),)
        return '{}({})'.format(self.__class__.__name__, ', '.join(identifiers))

    def as_sql(self, compiler, connection):
        alias, column = self.alias, self.target.column
        identifiers = (alias, column) if alias else (column,)
        sql = '.'.join(map(compiler.quote_name_unless_alias, identifiers))
        return sql, []

    def relabeled_clone(self, relabels):
        if self.alias is None:
            return self
        return self.__class__(relabels.get(self.alias, self.alias), self.target, self.output_field)

    def get_group_by_cols(self, alias=None):
        return [self]

    def get_db_converters(self, connection):
        if self.target == self.output_field:
            return self.output_field.get_db_converters(connection)
        return (self.output_field.get_db_converters(connection) +
                self.target.get_db_converters(connection))
```
### 66 - django/db/models/expressions.py:

Start line: 1125, End line: 1155

```python
class OrderBy(BaseExpression):

    def as_oracle(self, compiler, connection):
        # Oracle doesn't allow ORDER BY EXISTS() unless it's wrapped in
        # a CASE WHEN.
        if isinstance(self.expression, Exists):
            copy = self.copy()
            copy.expression = Case(
                When(self.expression, then=True),
                default=False,
                output_field=fields.BooleanField(),
            )
            return copy.as_sql(compiler, connection)
        return self.as_sql(compiler, connection)

    def get_group_by_cols(self, alias=None):
        cols = []
        for source in self.get_source_expressions():
            cols.extend(source.get_group_by_cols())
        return cols

    def reverse_ordering(self):
        self.descending = not self.descending
        if self.nulls_first or self.nulls_last:
            self.nulls_first = not self.nulls_first
            self.nulls_last = not self.nulls_last
        return self

    def asc(self):
        self.descending = False

    def desc(self):
        self.descending = True
```
### 67 - django/db/models/expressions.py:

Start line: 1048, End line: 1075

```python
class Exists(Subquery):
    template = 'EXISTS(%(subquery)s)'
    output_field = fields.BooleanField()

    def __init__(self, queryset, negated=False, **kwargs):
        # As a performance optimization, remove ordering since EXISTS doesn't
        # care about it, just whether or not a row matches.
        queryset = queryset.order_by()
        self.negated = negated
        super().__init__(queryset, **kwargs)

    def __invert__(self):
        clone = self.copy()
        clone.negated = not self.negated
        return clone

    def as_sql(self, compiler, connection, template=None, **extra_context):
        sql, params = super().as_sql(compiler, connection, template, **extra_context)
        if self.negated:
            sql = 'NOT {}'.format(sql)
        return sql, params

    def select_format(self, compiler, sql, params):
        # Wrap EXISTS() with a CASE WHEN expression if a database backend
        # (e.g. Oracle) doesn't support boolean expression in the SELECT list.
        if not compiler.connection.features.supports_boolean_expr_in_select_clause:
            sql = 'CASE WHEN {} THEN 1 ELSE 0 END'.format(sql)
        return sql, params
```
### 82 - django/db/models/expressions.py:

Start line: 1102, End line: 1123

```python
class OrderBy(BaseExpression):

    def as_sql(self, compiler, connection, template=None, **extra_context):
        template = template or self.template
        if connection.features.supports_order_by_nulls_modifier:
            if self.nulls_last:
                template = '%s NULLS LAST' % template
            elif self.nulls_first:
                template = '%s NULLS FIRST' % template
        else:
            if self.nulls_last:
                template = '%%(expression)s IS NULL, %s' % template
            elif self.nulls_first:
                template = '%%(expression)s IS NOT NULL, %s' % template
        connection.ops.check_expression_support(self)
        expression_sql, params = compiler.compile(self.expression)
        placeholders = {
            'expression': expression_sql,
            'ordering': 'DESC' if self.descending else 'ASC',
            **extra_context,
        }
        template = template or self.template
        params *= template.count('%(expression)s')
        return (template % placeholders).rstrip(), params
```
### 109 - django/db/models/expressions.py:

Start line: 996, End line: 1045

```python
class Subquery(Expression):
    """
    An explicit subquery. It may contain OuterRef() references to the outer
    query which will be resolved when it is applied to that query.
    """
    template = '(%(subquery)s)'
    contains_aggregate = False

    def __init__(self, queryset, output_field=None, **extra):
        self.query = queryset.query
        self.extra = extra
        super().__init__(output_field)

    def __getstate__(self):
        state = super().__getstate__()
        state.pop('_constructor_args', None)
        return state

    def get_source_expressions(self):
        return [self.query]

    def set_source_expressions(self, exprs):
        self.query = exprs[0]

    def _resolve_output_field(self):
        return self.query.output_field

    def copy(self):
        clone = super().copy()
        clone.query = clone.query.clone()
        return clone

    @property
    def external_aliases(self):
        return self.query.external_aliases

    def as_sql(self, compiler, connection, template=None, **extra_context):
        connection.ops.check_expression_support(self)
        template_params = {**self.extra, **extra_context}
        subquery_sql, sql_params = self.query.as_sql(compiler, connection)
        template_params['subquery'] = subquery_sql[1:-1]

        template = template or template_params.get('template', self.template)
        sql = template % template_params
        return sql, sql_params

    def get_group_by_cols(self, alias=None):
        if alias:
            return [Ref(alias, self)]
        return self.query.get_external_cols()
```
### 137 - django/db/models/expressions.py:

Start line: 920, End line: 966

```python
class Case(Expression):
    """
    An SQL searched CASE expression:

        CASE
            WHEN n > 0
                THEN 'positive'
            WHEN n < 0
                THEN 'negative'
            ELSE 'zero'
        END
    """
    template = 'CASE %(cases)s ELSE %(default)s END'
    case_joiner = ' '

    def __init__(self, *cases, default=None, output_field=None, **extra):
        if not all(isinstance(case, When) for case in cases):
            raise TypeError("Positional arguments must all be When objects.")
        super().__init__(output_field)
        self.cases = list(cases)
        self.default = self._parse_expressions(default)[0]
        self.extra = extra

    def __str__(self):
        return "CASE %s, ELSE %r" % (', '.join(str(c) for c in self.cases), self.default)

    def __repr__(self):
        return "<%s: %s>" % (self.__class__.__name__, self)

    def get_source_expressions(self):
        return self.cases + [self.default]

    def set_source_expressions(self, exprs):
        *self.cases, self.default = exprs

    def resolve_expression(self, query=None, allow_joins=True, reuse=None, summarize=False, for_save=False):
        c = self.copy()
        c.is_summary = summarize
        for pos, case in enumerate(c.cases):
            c.cases[pos] = case.resolve_expression(query, allow_joins, reuse, summarize, for_save)
        c.default = c.default.resolve_expression(query, allow_joins, reuse, summarize, for_save)
        return c

    def copy(self):
        c = super().copy()
        c.cases = c.cases[:]
        return c
```
### 139 - django/db/models/expressions.py:

Start line: 436, End line: 471

```python
class CombinedExpression(SQLiteNumericMixin, Expression):

    def as_sql(self, compiler, connection):
        try:
            lhs_output = self.lhs.output_field
        except FieldError:
            lhs_output = None
        try:
            rhs_output = self.rhs.output_field
        except FieldError:
            rhs_output = None
        if (not connection.features.has_native_duration_field and
                ((lhs_output and lhs_output.get_internal_type() == 'DurationField') or
                 (rhs_output and rhs_output.get_internal_type() == 'DurationField'))):
            return DurationExpression(self.lhs, self.connector, self.rhs).as_sql(compiler, connection)
        if (lhs_output and rhs_output and self.connector == self.SUB and
            lhs_output.get_internal_type() in {'DateField', 'DateTimeField', 'TimeField'} and
                lhs_output.get_internal_type() == rhs_output.get_internal_type()):
            return TemporalSubtraction(self.lhs, self.rhs).as_sql(compiler, connection)
        expressions = []
        expression_params = []
        sql, params = compiler.compile(self.lhs)
        expressions.append(sql)
        expression_params.extend(params)
        sql, params = compiler.compile(self.rhs)
        expressions.append(sql)
        expression_params.extend(params)
        # order of precedence
        expression_wrapper = '(%s)'
        sql = connection.ops.combine_expression(self.connector, expressions)
        return expression_wrapper % sql, expression_params

    def resolve_expression(self, query=None, allow_joins=True, reuse=None, summarize=False, for_save=False):
        c = self.copy()
        c.is_summary = summarize
        c.lhs = c.lhs.resolve_expression(query, allow_joins, reuse, summarize, for_save)
        c.rhs = c.rhs.resolve_expression(query, allow_joins, reuse, summarize, for_save)
        return c
```
