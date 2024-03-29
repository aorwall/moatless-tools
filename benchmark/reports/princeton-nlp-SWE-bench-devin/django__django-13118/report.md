# django__django-13118

| **django/django** | `b7b7df5fbcf44e6598396905136cab5a19e9faff` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 977 |
| **Any found context length** | 977 |
| **Avg pos** | 4.0 |
| **Min pos** | 2 |
| **Max pos** | 2 |
| **Top file pos** | 2 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/db/models/sql/query.py b/django/db/models/sql/query.py
--- a/django/db/models/sql/query.py
+++ b/django/db/models/sql/query.py
@@ -1324,9 +1324,7 @@ def build_filter(self, filter_expr, branch_negated=False, current_negated=False,
         require_outer = lookup_type == 'isnull' and condition.rhs is True and not current_negated
         if current_negated and (lookup_type != 'isnull' or condition.rhs is False) and condition.rhs is not None:
             require_outer = True
-            if (lookup_type != 'isnull' and (
-                    self.is_nullable(targets[0]) or
-                    self.alias_map[join_list[-1]].join_type == LOUTER)):
+            if lookup_type != 'isnull':
                 # The condition added here will be SQL like this:
                 # NOT (col IS NOT NULL), where the first NOT is added in
                 # upper layers of code. The reason for addition is that if col
@@ -1336,9 +1334,18 @@ def build_filter(self, filter_expr, branch_negated=False, current_negated=False,
                 # (col IS NULL OR col != someval)
                 #   <=>
                 # NOT (col IS NOT NULL AND col = someval).
-                lookup_class = targets[0].get_lookup('isnull')
-                col = self._get_col(targets[0], join_info.targets[0], alias)
-                clause.add(lookup_class(col, False), AND)
+                if (
+                    self.is_nullable(targets[0]) or
+                    self.alias_map[join_list[-1]].join_type == LOUTER
+                ):
+                    lookup_class = targets[0].get_lookup('isnull')
+                    col = self._get_col(targets[0], join_info.targets[0], alias)
+                    clause.add(lookup_class(col, False), AND)
+                # If someval is a nullable column, someval IS NOT NULL is
+                # added.
+                if isinstance(value, Col) and self.is_nullable(value.target):
+                    lookup_class = value.target.get_lookup('isnull')
+                    clause.add(lookup_class(value, False), AND)
         return clause, used_joins if not require_outer else ()
 
     def add_filter(self, filter_clause):

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/db/models/sql/query.py | 1327 | 1329 | 2 | 2 | 977
| django/db/models/sql/query.py | 1339 | 1341 | 2 | 2 | 977


## Problem Statement

```
SQL generated for negated F() expressions is incorrect
Description
	
Consider the following model definition.
from django.db import models
class Rectangle(models.Model):
	length = models.IntegerField(null=True)
	width = models.IntegerField(null=True)
We make the following queries: Q1) Rectangles that are squares. Q2a) Rectangles that are not squares (length != width). Q2b) Rectangles that are not squares (width != length). Queries Q2a and Q2b semantically mean the same. However, the ORM generates different SQL for these two queries.
import django
django.setup()
from django.db.models import F
from testapp.models import Rectangle
print '--- Q1: Get the rectangles that are squares'
print Rectangle.objects.filter(length=F('width')).values('pk').query
print '--- Q2a: Get the rectangles that are not squares'
print Rectangle.objects.exclude(length=F('width')).values('pk').query
print '--- Q2b: Get the rectangles that are not squares'
print Rectangle.objects.exclude(width=F('length')).values('pk').query
The generated SQL is
--- Q1: Get the rectangles that are squares
SELECT "testapp_rectangle"."id" FROM "testapp_rectangle" WHERE "testapp_rectangle"."length" = ("testapp_rectangle"."width")
--- Q2a: Get the rectangles that are not squares
SELECT "testapp_rectangle"."id" FROM "testapp_rectangle" WHERE NOT ("testapp_rectangle"."length" = ("testapp_rectangle"."width") AND "testapp_rectangle"."length" IS NOT NULL)
--- Q2b: Get the rectangles that are not squares
SELECT "testapp_rectangle"."id" FROM "testapp_rectangle" WHERE NOT ("testapp_rectangle"."width" = ("testapp_rectangle"."length") AND "testapp_rectangle"."width" IS NOT NULL)
Note the asymmetry between Q2a and Q2b. These queries will return different results.
Reddit user /u/charettes set up this useful SQL fiddle with the above mentioned schema and test values: ​http://sqlfiddle.com/#!12/c8283/4 Here's my reddit post on this issue: ​http://www.reddit.com/r/django/comments/2lxjcc/unintuitive_behavior_of_f_expression_with_exclude/

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 django/db/models/expressions.py | 1087 | 1115| 266 | 266 | 10383 | 
| **-> 2 <-** | **2 django/db/models/sql/query.py** | 1284 | 1342| 711 | 977 | 32420 | 
| 3 | 3 django/contrib/postgres/constraints.py | 69 | 90| 201 | 1178 | 33834 | 
| 4 | **3 django/db/models/sql/query.py** | 1662 | 1696| 399 | 1577 | 33834 | 
| 5 | **3 django/db/models/sql/query.py** | 1698 | 1769| 784 | 2361 | 33834 | 
| 6 | 3 django/db/models/expressions.py | 737 | 763| 196 | 2557 | 33834 | 
| 7 | **3 django/db/models/sql/query.py** | 1411 | 1422| 137 | 2694 | 33834 | 
| 8 | 3 django/db/models/expressions.py | 649 | 674| 268 | 2962 | 33834 | 
| 9 | 4 django/contrib/gis/db/models/functions.py | 345 | 374| 353 | 3315 | 37761 | 
| 10 | 5 django/db/models/sql/compiler.py | 1200 | 1221| 223 | 3538 | 51980 | 
| 11 | 5 django/db/models/expressions.py | 694 | 718| 238 | 3776 | 51980 | 
| 12 | 6 django/db/models/__init__.py | 1 | 53| 619 | 4395 | 52599 | 
| 13 | 6 django/db/models/expressions.py | 949 | 995| 377 | 4772 | 52599 | 
| 14 | 6 django/db/models/expressions.py | 882 | 946| 591 | 5363 | 52599 | 
| 15 | **6 django/db/models/sql/query.py** | 1440 | 1518| 734 | 6097 | 52599 | 
| 16 | 6 django/db/models/expressions.py | 721 | 735| 120 | 6217 | 52599 | 
| 17 | 6 django/contrib/postgres/constraints.py | 1 | 67| 550 | 6767 | 52599 | 
| 18 | 7 django/db/models/sql/where.py | 230 | 246| 130 | 6897 | 54403 | 
| 19 | 7 django/contrib/postgres/constraints.py | 92 | 105| 180 | 7077 | 54403 | 
| 20 | 7 django/db/models/expressions.py | 1247 | 1279| 246 | 7323 | 54403 | 
| 21 | 8 django/db/backends/sqlite3/operations.py | 40 | 66| 232 | 7555 | 57434 | 
| 22 | **8 django/db/models/sql/query.py** | 1105 | 1137| 338 | 7893 | 57434 | 
| 23 | 8 django/db/models/expressions.py | 459 | 484| 303 | 8196 | 57434 | 
| 24 | **8 django/db/models/sql/query.py** | 363 | 413| 494 | 8690 | 57434 | 
| 25 | **8 django/db/models/sql/query.py** | 1049 | 1074| 214 | 8904 | 57434 | 
| 26 | 8 django/contrib/gis/db/models/functions.py | 459 | 487| 225 | 9129 | 57434 | 
| 27 | 8 django/db/models/expressions.py | 1 | 28| 196 | 9325 | 57434 | 
| 28 | 8 django/db/models/expressions.py | 425 | 457| 244 | 9569 | 57434 | 
| 29 | 9 django/db/models/aggregates.py | 45 | 68| 294 | 9863 | 58735 | 
| 30 | 9 django/db/models/expressions.py | 603 | 647| 419 | 10282 | 58735 | 
| 31 | **9 django/db/models/sql/query.py** | 1596 | 1620| 227 | 10509 | 58735 | 
| 32 | 10 django/db/backends/base/features.py | 1 | 111| 886 | 11395 | 61327 | 
| 33 | 11 django/db/models/fields/related_lookups.py | 119 | 155| 244 | 11639 | 62776 | 
| 34 | 11 django/contrib/gis/db/models/functions.py | 254 | 294| 458 | 12097 | 62776 | 
| 35 | **11 django/db/models/sql/query.py** | 510 | 544| 282 | 12379 | 62776 | 
| 36 | 12 django/db/models/query.py | 804 | 843| 322 | 12701 | 79878 | 
| 37 | 12 django/db/models/expressions.py | 997 | 1022| 242 | 12943 | 79878 | 
| 38 | 12 django/db/models/aggregates.py | 70 | 96| 266 | 13209 | 79878 | 
| 39 | **12 django/db/models/sql/query.py** | 2283 | 2299| 177 | 13386 | 79878 | 
| 40 | 13 django/db/backends/mysql/features.py | 1 | 108| 853 | 14239 | 81163 | 
| 41 | 13 django/db/models/expressions.py | 766 | 800| 292 | 14531 | 81163 | 
| 42 | **13 django/db/models/sql/query.py** | 136 | 230| 833 | 15364 | 81163 | 
| 43 | 13 django/db/models/expressions.py | 1142 | 1167| 248 | 15612 | 81163 | 
| 44 | 14 django/contrib/postgres/aggregates/mixins.py | 1 | 20| 182 | 15794 | 81605 | 
| 45 | 15 django/db/models/lookups.py | 288 | 300| 168 | 15962 | 86562 | 
| 46 | 15 django/contrib/postgres/constraints.py | 156 | 166| 132 | 16094 | 86562 | 
| 47 | **15 django/db/models/sql/query.py** | 2128 | 2173| 371 | 16465 | 86562 | 
| 48 | 15 django/db/models/sql/compiler.py | 149 | 197| 523 | 16988 | 86562 | 
| 49 | **15 django/db/models/sql/query.py** | 892 | 914| 248 | 17236 | 86562 | 
| 50 | 15 django/db/models/lookups.py | 565 | 590| 163 | 17399 | 86562 | 
| 51 | 15 django/db/models/sql/compiler.py | 1 | 19| 170 | 17569 | 86562 | 
| 52 | 15 django/contrib/postgres/constraints.py | 128 | 154| 231 | 17800 | 86562 | 
| 53 | 15 django/db/models/sql/compiler.py | 433 | 486| 564 | 18364 | 86562 | 
| 54 | **15 django/db/models/sql/query.py** | 1800 | 1849| 330 | 18694 | 86562 | 
| 55 | 16 django/db/backends/postgresql/features.py | 1 | 87| 701 | 19395 | 87263 | 
| 56 | **16 django/db/models/sql/query.py** | 696 | 731| 389 | 19784 | 87263 | 
| 57 | 16 django/db/models/expressions.py | 1025 | 1084| 440 | 20224 | 87263 | 
| 58 | 16 django/db/models/lookups.py | 119 | 142| 220 | 20444 | 87263 | 
| 59 | 17 django/db/backends/oracle/operations.py | 629 | 655| 303 | 20747 | 93302 | 
| 60 | 17 django/db/models/query.py | 327 | 353| 222 | 20969 | 93302 | 
| 61 | 17 django/db/models/lookups.py | 303 | 353| 306 | 21275 | 93302 | 
| 62 | 17 django/db/models/query.py | 988 | 1019| 341 | 21616 | 93302 | 
| 63 | 17 django/db/models/expressions.py | 517 | 558| 298 | 21914 | 93302 | 
| 64 | **17 django/db/models/sql/query.py** | 621 | 645| 269 | 22183 | 93302 | 
| 65 | 17 django/db/models/sql/compiler.py | 1036 | 1076| 337 | 22520 | 93302 | 
| 66 | 18 django/db/models/functions/text.py | 144 | 188| 312 | 22832 | 95638 | 
| 67 | 18 django/db/models/functions/text.py | 1 | 39| 266 | 23098 | 95638 | 
| 68 | 18 django/db/models/sql/compiler.py | 199 | 269| 580 | 23678 | 95638 | 
| 69 | 19 django/contrib/gis/db/backends/mysql/operations.py | 51 | 71| 225 | 23903 | 96489 | 
| 70 | 19 django/contrib/gis/db/models/functions.py | 322 | 342| 166 | 24069 | 96489 | 
| 71 | 19 django/db/backends/oracle/operations.py | 573 | 589| 290 | 24359 | 96489 | 
| 72 | **19 django/db/models/sql/query.py** | 1771 | 1798| 244 | 24603 | 96489 | 
| 73 | **19 django/db/models/sql/query.py** | 1 | 65| 465 | 25068 | 96489 | 
| 74 | 19 django/db/models/sql/compiler.py | 360 | 396| 427 | 25495 | 96489 | 
| 75 | 20 django/db/models/sql/subqueries.py | 1 | 44| 320 | 25815 | 97702 | 
| 76 | 21 django/forms/models.py | 310 | 349| 387 | 26202 | 109476 | 
| 77 | 21 django/db/models/sql/compiler.py | 1415 | 1429| 127 | 26329 | 109476 | 
| 78 | 21 django/db/models/expressions.py | 1169 | 1199| 218 | 26547 | 109476 | 
| 79 | 22 django/db/models/fields/related.py | 997 | 1024| 215 | 26762 | 123352 | 
| 80 | 22 django/db/models/sql/where.py | 1 | 30| 167 | 26929 | 123352 | 
| 81 | 23 django/db/backends/mysql/operations.py | 274 | 285| 165 | 27094 | 126989 | 
| 82 | 24 django/db/models/base.py | 1806 | 1879| 572 | 27666 | 143570 | 
| 83 | 24 django/db/models/fields/related_lookups.py | 62 | 99| 447 | 28113 | 143570 | 
| 84 | 24 django/db/models/sql/compiler.py | 885 | 977| 839 | 28952 | 143570 | 
| 85 | 25 django/contrib/gis/db/models/aggregates.py | 29 | 46| 216 | 29168 | 144187 | 
| 86 | 26 django/db/models/sql/datastructures.py | 117 | 137| 144 | 29312 | 145589 | 
| 87 | 26 django/contrib/gis/db/models/functions.py | 18 | 53| 312 | 29624 | 145589 | 
| 88 | **26 django/db/models/sql/query.py** | 2175 | 2207| 228 | 29852 | 145589 | 
| 89 | 26 django/db/models/sql/compiler.py | 22 | 47| 257 | 30109 | 145589 | 
| 90 | 27 django/db/models/functions/window.py | 1 | 25| 153 | 30262 | 146232 | 
| 91 | 28 django/contrib/gis/db/backends/base/features.py | 1 | 100| 752 | 31014 | 146985 | 
| 92 | 28 django/db/models/sql/where.py | 65 | 115| 396 | 31410 | 146985 | 
| 93 | 28 django/db/models/expressions.py | 1202 | 1245| 373 | 31783 | 146985 | 
| 94 | 28 django/contrib/gis/db/models/functions.py | 377 | 417| 294 | 32077 | 146985 | 
| 95 | 28 django/db/models/expressions.py | 1296 | 1329| 276 | 32353 | 146985 | 
| 96 | 28 django/db/models/sql/where.py | 192 | 206| 156 | 32509 | 146985 | 
| 97 | 29 django/contrib/postgres/fields/ranges.py | 232 | 322| 479 | 32988 | 149102 | 
| 98 | **29 django/db/models/sql/query.py** | 1935 | 1984| 420 | 33408 | 149102 | 
| 99 | 29 django/contrib/postgres/constraints.py | 107 | 126| 155 | 33563 | 149102 | 
| 100 | 29 django/db/backends/base/features.py | 113 | 215| 840 | 34403 | 149102 | 
| 101 | 29 django/contrib/gis/db/models/functions.py | 440 | 456| 180 | 34583 | 149102 | 
| 102 | 29 django/db/models/base.py | 1704 | 1804| 729 | 35312 | 149102 | 
| 103 | 30 django/db/backends/mysql/compiler.py | 16 | 44| 232 | 35544 | 149451 | 
| 104 | 30 django/db/models/query.py | 845 | 874| 248 | 35792 | 149451 | 
| 105 | 31 django/db/models/functions/comparison.py | 42 | 53| 171 | 35963 | 150669 | 
| 106 | 32 django/contrib/gis/db/models/sql/conversion.py | 43 | 70| 203 | 36166 | 151152 | 
| 107 | 32 django/db/models/sql/compiler.py | 398 | 406| 133 | 36299 | 151152 | 
| 108 | 33 django/db/models/constraints.py | 32 | 76| 372 | 36671 | 152767 | 
| 109 | **33 django/db/models/sql/query.py** | 1139 | 1182| 469 | 37140 | 152767 | 
| 110 | 34 django/contrib/gis/db/backends/spatialite/operations.py | 1 | 23| 226 | 37366 | 154473 | 
| 111 | 34 django/db/models/sql/where.py | 209 | 227| 131 | 37497 | 154473 | 
| 112 | 34 django/contrib/gis/db/models/functions.py | 88 | 121| 249 | 37746 | 154473 | 
| 113 | 34 django/db/backends/mysql/compiler.py | 1 | 13| 115 | 37861 | 154473 | 
| 114 | 35 django/db/backends/oracle/features.py | 1 | 75| 592 | 38453 | 155065 | 
| 115 | 36 django/contrib/gis/db/models/__init__.py | 1 | 19| 204 | 38657 | 155269 | 
| 116 | 37 django/contrib/postgres/search.py | 160 | 195| 313 | 38970 | 157491 | 
| 117 | 37 django/db/models/expressions.py | 31 | 150| 866 | 39836 | 157491 | 
| 118 | 37 django/contrib/gis/db/models/functions.py | 55 | 85| 275 | 40111 | 157491 | 
| 119 | 37 django/contrib/gis/db/backends/mysql/operations.py | 73 | 103| 225 | 40336 | 157491 | 
| 120 | 38 django/db/backends/sqlite3/features.py | 1 | 67| 622 | 40958 | 158113 | 
| 121 | 39 django/db/models/query_utils.py | 312 | 352| 286 | 41244 | 160819 | 
| 122 | 40 django/db/backends/sqlite3/base.py | 81 | 155| 757 | 42001 | 166836 | 
| 123 | 41 django/contrib/gis/db/models/fields.py | 280 | 340| 361 | 42362 | 169887 | 
| 124 | 41 django/db/models/sql/where.py | 157 | 190| 233 | 42595 | 169887 | 
| 125 | 41 django/db/models/fields/related.py | 696 | 708| 116 | 42711 | 169887 | 
| 126 | 41 django/db/models/sql/where.py | 32 | 63| 317 | 43028 | 169887 | 
| 127 | 41 django/db/models/sql/compiler.py | 63 | 147| 881 | 43909 | 169887 | 
| 128 | 41 django/db/models/expressions.py | 1118 | 1140| 188 | 44097 | 169887 | 
| 129 | 41 django/db/models/expressions.py | 487 | 514| 214 | 44311 | 169887 | 
| 130 | 41 django/db/backends/sqlite3/base.py | 582 | 612| 201 | 44512 | 169887 | 
| 131 | 41 django/db/models/sql/compiler.py | 271 | 358| 712 | 45224 | 169887 | 
| 132 | 41 django/db/models/functions/comparison.py | 31 | 40| 131 | 45355 | 169887 | 
| 133 | 41 django/contrib/postgres/search.py | 130 | 157| 248 | 45603 | 169887 | 
| 134 | **41 django/db/models/sql/query.py** | 232 | 286| 400 | 46003 | 169887 | 
| 135 | 42 django/db/models/sql/constants.py | 1 | 25| 140 | 46143 | 170027 | 
| 136 | 42 django/db/models/expressions.py | 335 | 390| 368 | 46511 | 170027 | 
| 137 | **42 django/db/models/sql/query.py** | 1076 | 1103| 285 | 46796 | 170027 | 
| 138 | 42 django/db/backends/base/features.py | 216 | 305| 716 | 47512 | 170027 | 
| 139 | **42 django/db/models/sql/query.py** | 2363 | 2418| 827 | 48339 | 170027 | 
| 140 | 42 django/db/models/expressions.py | 561 | 600| 290 | 48629 | 170027 | 
| 141 | 42 django/db/models/fields/related.py | 1235 | 1352| 963 | 49592 | 170027 | 
| 142 | 42 django/db/models/query.py | 1184 | 1203| 209 | 49801 | 170027 | 
| 143 | 42 django/contrib/gis/db/models/functions.py | 124 | 146| 205 | 50006 | 170027 | 
| 144 | 43 django/db/models/fields/json.py | 214 | 232| 232 | 50238 | 174160 | 
| 145 | 44 django/contrib/gis/db/models/lookups.py | 86 | 217| 762 | 51000 | 176773 | 
| 146 | 44 django/db/models/lookups.py | 356 | 388| 294 | 51294 | 176773 | 
| 147 | 44 django/contrib/postgres/search.py | 1 | 24| 205 | 51499 | 176773 | 
| 148 | 44 django/db/models/base.py | 1897 | 2028| 976 | 52475 | 176773 | 
| 149 | 45 django/contrib/gis/db/backends/mysql/schema.py | 25 | 38| 146 | 52621 | 177404 | 
| 150 | 46 django/db/models/deletion.py | 1 | 76| 566 | 53187 | 181230 | 
| 151 | 46 django/db/models/constraints.py | 79 | 161| 729 | 53916 | 181230 | 
| 152 | **46 django/db/models/sql/query.py** | 2034 | 2056| 249 | 54165 | 181230 | 
| 153 | 46 django/db/models/sql/datastructures.py | 1 | 21| 126 | 54291 | 181230 | 
| 154 | 47 django/contrib/gis/db/backends/oracle/operations.py | 175 | 221| 350 | 54641 | 183304 | 
| 155 | 47 django/db/models/fields/json.py | 383 | 393| 131 | 54772 | 183304 | 
| 156 | 47 django/db/backends/mysql/features.py | 110 | 160| 438 | 55210 | 183304 | 
| 157 | 47 django/db/models/functions/comparison.py | 1 | 29| 317 | 55527 | 183304 | 
| 158 | 48 django/db/backends/postgresql/operations.py | 205 | 289| 674 | 56201 | 185964 | 
| 159 | **48 django/db/models/sql/query.py** | 984 | 1015| 307 | 56508 | 185964 | 
| 160 | 48 django/db/models/expressions.py | 1331 | 1367| 305 | 56813 | 185964 | 
| 161 | 48 django/contrib/gis/db/models/lookups.py | 326 | 335| 148 | 56961 | 185964 | 
| 162 | 48 django/db/backends/sqlite3/operations.py | 312 | 357| 453 | 57414 | 185964 | 
| 163 | 48 django/db/backends/oracle/operations.py | 370 | 407| 369 | 57783 | 185964 | 
| 164 | 49 django/db/models/functions/math.py | 114 | 167| 295 | 58078 | 187088 | 
| 165 | 49 django/db/models/expressions.py | 1281 | 1293| 111 | 58189 | 187088 | 
| 166 | 49 django/contrib/gis/db/models/aggregates.py | 49 | 84| 207 | 58396 | 187088 | 
| 167 | 49 django/db/models/sql/datastructures.py | 59 | 102| 419 | 58815 | 187088 | 
| 168 | 49 django/db/backends/oracle/operations.py | 21 | 73| 574 | 59389 | 187088 | 
| 169 | 49 django/contrib/postgres/fields/ranges.py | 1 | 40| 283 | 59672 | 187088 | 
| 170 | 49 django/db/models/functions/text.py | 119 | 141| 209 | 59881 | 187088 | 
| 171 | 50 django/contrib/gis/db/backends/base/operations.py | 1 | 163| 1281 | 61162 | 188370 | 
| 172 | 50 django/db/backends/sqlite3/operations.py | 296 | 310| 148 | 61310 | 188370 | 
| 173 | 50 django/db/models/expressions.py | 222 | 256| 285 | 61595 | 188370 | 
| 174 | 50 django/db/models/fields/related.py | 841 | 862| 169 | 61764 | 188370 | 
| 175 | **50 django/db/models/sql/query.py** | 1344 | 1365| 250 | 62014 | 188370 | 
| 176 | 50 django/contrib/gis/db/models/functions.py | 149 | 171| 190 | 62204 | 188370 | 
| 177 | 51 django/contrib/postgres/aggregates/statistics.py | 1 | 66| 419 | 62623 | 188789 | 
| 178 | 51 django/db/models/fields/related.py | 190 | 254| 673 | 63296 | 188789 | 
| 179 | 51 django/contrib/gis/db/models/lookups.py | 1 | 35| 286 | 63582 | 188789 | 
| 180 | 52 django/db/models/indexes.py | 125 | 137| 134 | 63716 | 190112 | 
| 181 | 52 django/db/models/query.py | 919 | 969| 381 | 64097 | 190112 | 
| 182 | 52 django/db/models/fields/related.py | 750 | 768| 222 | 64319 | 190112 | 
| 183 | 53 django/db/migrations/questioner.py | 227 | 240| 123 | 64442 | 192185 | 
| 184 | 53 django/db/models/functions/comparison.py | 97 | 126| 244 | 64686 | 192185 | 
| 185 | 53 django/contrib/gis/db/models/functions.py | 193 | 251| 395 | 65081 | 192185 | 
| 186 | 53 django/db/models/fields/json.py | 426 | 438| 174 | 65255 | 192185 | 
| 187 | 53 django/contrib/gis/db/models/functions.py | 174 | 190| 193 | 65448 | 192185 | 
| 188 | 53 django/db/models/functions/comparison.py | 56 | 74| 205 | 65653 | 192185 | 
| 189 | 53 django/db/backends/oracle/operations.py | 478 | 509| 360 | 66013 | 192185 | 
| 190 | 53 django/contrib/gis/db/backends/oracle/operations.py | 38 | 49| 202 | 66215 | 192185 | 
| 191 | 53 django/db/models/lookups.py | 390 | 419| 337 | 66552 | 192185 | 
| 192 | 53 django/db/models/functions/math.py | 52 | 97| 290 | 66842 | 192185 | 
| 193 | 53 django/db/models/expressions.py | 258 | 285| 184 | 67026 | 192185 | 
| 194 | 53 django/db/models/base.py | 1881 | 1895| 136 | 67162 | 192185 | 
| 195 | 53 django/db/models/lookups.py | 489 | 510| 172 | 67334 | 192185 | 


### Hint

```
Unfortunately this will be messy to fix. It is true that .exclude() should produce the complement of .filter(), and that doesn't happen here. One possible solution is to use WHERE ("testapp_rectangle"."length" = ("testapp_rectangle"."width")) IS NOT true. This should match both null and false values (the .filter() query could be written as WHERE ("testapp_rectangle"."length" = ("testapp_rectangle"."width")) IS true. I have no idea how well IS NOT true is optimized, and if it is available for all backends. Another possibility is to also filter F() NULL values, but as said that will be messy.
After a quick test, it seems the WHERE ("testapp_rectangle"."length" = ("testapp_rectangle"."width")) IS NOT true approach will not work that well. At least PostgreSQL wont use indexes for a query select * from foobar where (id > 1 and id < 20) is true, but will use the index for select * from foobar where (id > 1 and id < 20). This tells me PostgreSQL's optimizer will not handle the is true / is not true conditions well. If somebody is able to provide a somewhat clean solution to this, I am all for fixing this.
Is it the case that filter() and exclude() should produce the complement of each other? If that is not the case, the obvious queries would be WHERE (length = width) and WHERE NOT (length = width). If the results should be complimentary, couldn't we use WHERE (length = width) for the filter and WHERE (length = width)) IS NOT true for the exclude. The exclude would be slower, but at least the results would be consistent.
@akaariai what about adding another IS NOT NULL clause ​here based on value we should end up with NOT ((length = width) AND (length IS NULL) AND (width IS NULL))? If the query planer use the index when the generated constraint is NOT ((length = width) AND (length IS NULL)) then it should also be able to use it in this case.
Yes, .exclude() should be the complement of .filter(). As said in comment:2 the WHERE (length = width)) IS NOT true way is just way too slow at least on PostgreSQL (always a sequential scan according to my tests). Going for the NOT ((length = width) AND (length IS NULL) AND (width IS NULL)) should likely work. I'd like to do this in the Lookup class. We can't do this directly in as_sql() as we don't have enough context there. Maybe a new method get_extra_restriction(self, is_negated) could work. The method returns an expression (a Lookup instance or WhereNode instance) that will be ANDed to the lookup's main condition. I'll try to find the time to write a small proof of concept.
I have to take back what I said earlier. Unfortunately we have backwards compatibility requirements for the pre-Lookup classes way of writing custom lookups. So, we actually don't necessarily have a lookup class at the point where we want to add the (width IS NULL) condition. So, for now we have to do this directly in build_filter() method.
Just for the record, the (x=y) is true syntax is not supported on Oracle (or Mysql, IIRC). It's a database, it doesn't really handle logic.
```

## Patch

```diff
diff --git a/django/db/models/sql/query.py b/django/db/models/sql/query.py
--- a/django/db/models/sql/query.py
+++ b/django/db/models/sql/query.py
@@ -1324,9 +1324,7 @@ def build_filter(self, filter_expr, branch_negated=False, current_negated=False,
         require_outer = lookup_type == 'isnull' and condition.rhs is True and not current_negated
         if current_negated and (lookup_type != 'isnull' or condition.rhs is False) and condition.rhs is not None:
             require_outer = True
-            if (lookup_type != 'isnull' and (
-                    self.is_nullable(targets[0]) or
-                    self.alias_map[join_list[-1]].join_type == LOUTER)):
+            if lookup_type != 'isnull':
                 # The condition added here will be SQL like this:
                 # NOT (col IS NOT NULL), where the first NOT is added in
                 # upper layers of code. The reason for addition is that if col
@@ -1336,9 +1334,18 @@ def build_filter(self, filter_expr, branch_negated=False, current_negated=False,
                 # (col IS NULL OR col != someval)
                 #   <=>
                 # NOT (col IS NOT NULL AND col = someval).
-                lookup_class = targets[0].get_lookup('isnull')
-                col = self._get_col(targets[0], join_info.targets[0], alias)
-                clause.add(lookup_class(col, False), AND)
+                if (
+                    self.is_nullable(targets[0]) or
+                    self.alias_map[join_list[-1]].join_type == LOUTER
+                ):
+                    lookup_class = targets[0].get_lookup('isnull')
+                    col = self._get_col(targets[0], join_info.targets[0], alias)
+                    clause.add(lookup_class(col, False), AND)
+                # If someval is a nullable column, someval IS NOT NULL is
+                # added.
+                if isinstance(value, Col) and self.is_nullable(value.target):
+                    lookup_class = value.target.get_lookup('isnull')
+                    clause.add(lookup_class(value, False), AND)
         return clause, used_joins if not require_outer else ()
 
     def add_filter(self, filter_clause):

```

## Test Patch

```diff
diff --git a/tests/queries/models.py b/tests/queries/models.py
--- a/tests/queries/models.py
+++ b/tests/queries/models.py
@@ -142,6 +142,7 @@ def __str__(self):
 class Number(models.Model):
     num = models.IntegerField()
     other_num = models.IntegerField(null=True)
+    another_num = models.IntegerField(null=True)
 
     def __str__(self):
         return str(self.num)
diff --git a/tests/queries/tests.py b/tests/queries/tests.py
--- a/tests/queries/tests.py
+++ b/tests/queries/tests.py
@@ -2372,7 +2372,10 @@ def test_named_values_list_without_fields(self):
         qs = Number.objects.extra(select={'num2': 'num+1'}).annotate(Count('id'))
         values = qs.values_list(named=True).first()
         self.assertEqual(type(values).__name__, 'Row')
-        self.assertEqual(values._fields, ('num2', 'id', 'num', 'other_num', 'id__count'))
+        self.assertEqual(
+            values._fields,
+            ('num2', 'id', 'num', 'other_num', 'another_num', 'id__count'),
+        )
         self.assertEqual(values.num, 72)
         self.assertEqual(values.num2, 73)
         self.assertEqual(values.id__count, 1)
@@ -2855,6 +2858,18 @@ def test_subquery_exclude_outerref(self):
         self.r1.delete()
         self.assertFalse(qs.exists())
 
+    def test_exclude_nullable_fields(self):
+        number = Number.objects.create(num=1, other_num=1)
+        Number.objects.create(num=2, other_num=2, another_num=2)
+        self.assertSequenceEqual(
+            Number.objects.exclude(other_num=F('another_num')),
+            [number],
+        )
+        self.assertSequenceEqual(
+            Number.objects.exclude(num=F('another_num')),
+            [number],
+        )
+
 
 class ExcludeTest17600(TestCase):
     """

```


## Code snippets

### 1 - django/db/models/expressions.py:

Start line: 1087, End line: 1115

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
        # (e.g. Oracle) doesn't support boolean expression in SELECT or GROUP
        # BY list.
        if not compiler.connection.features.supports_boolean_expr_in_select_clause:
            sql = 'CASE WHEN {} THEN 1 ELSE 0 END'.format(sql)
        return sql, params
```
### 2 - django/db/models/sql/query.py:

Start line: 1284, End line: 1342

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
### 3 - django/contrib/postgres/constraints.py:

Start line: 69, End line: 90

```python
class ExclusionConstraint(BaseConstraint):

    def _get_expression_sql(self, compiler, connection, query):
        expressions = []
        for idx, (expression, operator) in enumerate(self.expressions):
            if isinstance(expression, str):
                expression = F(expression)
            expression = expression.resolve_expression(query=query)
            sql, params = expression.as_sql(compiler, connection)
            try:
                opclass = self.opclasses[idx]
                if opclass:
                    sql = '%s %s' % (sql, opclass)
            except IndexError:
                pass
            expressions.append('%s WITH %s' % (sql % params, operator))
        return expressions

    def _get_condition_sql(self, compiler, schema_editor, query):
        if self.condition is None:
            return None
        where = query.build_where(self.condition)
        sql, params = where.as_sql(compiler, schema_editor.connection)
        return sql % tuple(schema_editor.quote_value(p) for p in params)
```
### 4 - django/db/models/sql/query.py:

Start line: 1662, End line: 1696

```python
class Query(BaseExpression):

    def resolve_ref(self, name, allow_joins=True, reuse=None, summarize=False):
        if not allow_joins and LOOKUP_SEP in name:
            raise FieldError("Joined field references are not permitted in this query")
        annotation = self.annotations.get(name)
        if annotation is not None:
            if not allow_joins:
                for alias in self._gen_col_aliases([annotation]):
                    if isinstance(self.alias_map[alias], Join):
                        raise FieldError(
                            'Joined field references are not permitted in '
                            'this query'
                        )
            if summarize:
                # Summarize currently means we are doing an aggregate() query
                # which is executed as a wrapped subquery if any of the
                # aggregate() elements reference an existing annotation. In
                # that case we need to return a Ref to the subquery's annotation.
                return Ref(name, self.annotation_select[name])
            else:
                return annotation
        else:
            field_list = name.split(LOOKUP_SEP)
            join_info = self.setup_joins(field_list, self.get_meta(), self.get_initial_alias(), can_reuse=reuse)
            targets, final_alias, join_list = self.trim_joins(join_info.targets, join_info.joins, join_info.path)
            if not allow_joins and len(join_list) > 1:
                raise FieldError('Joined field references are not permitted in this query')
            if len(targets) > 1:
                raise FieldError("Referencing multicolumn fields with F() objects "
                                 "isn't supported")
            # Verify that the last lookup in name is a field or a transform:
            # transform_function() raises FieldError if not.
            join_info.transform_function(targets[0], final_alias)
            if reuse is not None:
                reuse.update(join_list)
            return self._get_col(targets[0], join_info.targets[0], join_list[-1])
```
### 5 - django/db/models/sql/query.py:

Start line: 1698, End line: 1769

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
### 6 - django/db/models/expressions.py:

Start line: 737, End line: 763

```python
class RawSQL(Expression):

    def resolve_expression(self, query=None, allow_joins=True, reuse=None, summarize=False, for_save=False):
        # Resolve parents fields used in raw SQL.
        for parent in query.model._meta.get_parent_list():
            for parent_field in parent._meta.local_fields:
                _, column_name = parent_field.get_attname_column()
                if column_name.lower() in self.sql.lower():
                    query.resolve_ref(parent_field.name, allow_joins, reuse, summarize)
                    break
        return super().resolve_expression(query, allow_joins, reuse, summarize, for_save)


class Star(Expression):
    def __repr__(self):
        return "'*'"

    def as_sql(self, compiler, connection):
        return '*', []


class Random(Expression):
    output_field = fields.FloatField()

    def __repr__(self):
        return "Random()"

    def as_sql(self, compiler, connection):
        return connection.ops.random_function_sql(), []
```
### 7 - django/db/models/sql/query.py:

Start line: 1411, End line: 1422

```python
class Query(BaseExpression):

    def add_filtered_relation(self, filtered_relation, alias):
        filtered_relation.alias = alias
        lookups = dict(get_children_from_q(filtered_relation.condition))
        for lookup in chain((filtered_relation.relation_name,), lookups):
            lookup_parts, field_parts, _ = self.solve_lookup_type(lookup)
            shift = 2 if not lookup_parts else 1
            if len(field_parts) > (shift + len(lookup_parts)):
                raise ValueError(
                    "FilteredRelation's condition doesn't support nested "
                    "relations (got %r)." % lookup
                )
        self._filtered_relations[filtered_relation.alias] = filtered_relation
```
### 8 - django/db/models/expressions.py:

Start line: 649, End line: 674

```python
class Func(SQLiteNumericMixin, Expression):

    def as_sql(self, compiler, connection, function=None, template=None, arg_joiner=None, **extra_context):
        connection.ops.check_expression_support(self)
        sql_parts = []
        params = []
        for arg in self.source_expressions:
            arg_sql, arg_params = compiler.compile(arg)
            sql_parts.append(arg_sql)
            params.extend(arg_params)
        data = {**self.extra, **extra_context}
        # Use the first supplied value in this order: the parameter to this
        # method, a value supplied in __init__()'s **extra (the value in
        # `data`), or the value defined on the class.
        if function is not None:
            data['function'] = function
        else:
            data.setdefault('function', self.function)
        template = template or data.get('template', self.template)
        arg_joiner = arg_joiner or data.get('arg_joiner', self.arg_joiner)
        data['expressions'] = data['field'] = arg_joiner.join(sql_parts)
        return template % data, params

    def copy(self):
        copy = super().copy()
        copy.source_expressions = self.source_expressions[:]
        copy.extra = self.extra.copy()
        return copy
```
### 9 - django/contrib/gis/db/models/functions.py:

Start line: 345, End line: 374

```python
class Length(DistanceResultMixin, OracleToleranceMixin, GeoFunc):
    def __init__(self, expr1, spheroid=True, **extra):
        self.spheroid = spheroid
        super().__init__(expr1, **extra)

    def as_sql(self, compiler, connection, **extra_context):
        if self.geo_field.geodetic(connection) and not connection.features.supports_length_geodetic:
            raise NotSupportedError("This backend doesn't support Length on geodetic fields")
        return super().as_sql(compiler, connection, **extra_context)

    def as_postgresql(self, compiler, connection, **extra_context):
        clone = self.copy()
        function = None
        if self.source_is_geography():
            clone.source_expressions.append(Value(self.spheroid))
        elif self.geo_field.geodetic(connection):
            # Geometry fields with geodetic (lon/lat) coordinates need length_spheroid
            function = connection.ops.spatial_function_name('LengthSpheroid')
            clone.source_expressions.append(Value(self.geo_field.spheroid(connection)))
        else:
            dim = min(f.dim for f in self.get_source_fields() if f)
            if dim > 2:
                function = connection.ops.length3d
        return super(Length, clone).as_sql(compiler, connection, function=function, **extra_context)

    def as_sqlite(self, compiler, connection, **extra_context):
        function = None
        if self.geo_field.geodetic(connection):
            function = 'GeodesicLength' if self.spheroid else 'GreatCircleLength'
        return super().as_sql(compiler, connection, function=function, **extra_context)
```
### 10 - django/db/models/sql/compiler.py:

Start line: 1200, End line: 1221

```python
class SQLCompiler:
    ordering_parts =
    # ... other code

    def as_subquery_condition(self, alias, columns, compiler):
        qn = compiler.quote_name_unless_alias
        qn2 = self.connection.ops.quote_name

        for index, select_col in enumerate(self.query.select):
            lhs_sql, lhs_params = self.compile(select_col)
            rhs = '%s.%s' % (qn(alias), qn2(columns[index]))
            self.query.where.add(
                RawSQL('%s = %s' % (lhs_sql, rhs), lhs_params), 'AND')

        sql, params = self.as_sql()
        return 'EXISTS (%s)' % sql, params

    def explain_query(self):
        result = list(self.execute_sql())
        # Some backends return 1 item tuples with strings, and others return
        # tuples with integers and strings. Flatten them out into strings.
        for row in result[0]:
            if not isinstance(row, str):
                yield ' '.join(str(c) for c in row)
            else:
                yield row
```
### 15 - django/db/models/sql/query.py:

Start line: 1440, End line: 1518

```python
class Query(BaseExpression):

    def names_to_path(self, names, opts, allow_many=True, fail_on_missing=False):
        # ... other code
        for pos, name in enumerate(names):
            cur_names_with_path = (name, [])
            if name == 'pk':
                name = opts.pk.name

            field = None
            filtered_relation = None
            try:
                field = opts.get_field(name)
            except FieldDoesNotExist:
                if name in self.annotation_select:
                    field = self.annotation_select[name].output_field
                elif name in self._filtered_relations and pos == 0:
                    filtered_relation = self._filtered_relations[name]
                    field = opts.get_field(filtered_relation.relation_name)
            if field is not None:
                # Fields that contain one-to-many relations with a generic
                # model (like a GenericForeignKey) cannot generate reverse
                # relations and therefore cannot be used for reverse querying.
                if field.is_relation and not field.related_model:
                    raise FieldError(
                        "Field %r does not generate an automatic reverse "
                        "relation and therefore cannot be used for reverse "
                        "querying. If it is a GenericForeignKey, consider "
                        "adding a GenericRelation." % name
                    )
                try:
                    model = field.model._meta.concrete_model
                except AttributeError:
                    # QuerySet.annotate() may introduce fields that aren't
                    # attached to a model.
                    model = None
            else:
                # We didn't find the current field, so move position back
                # one step.
                pos -= 1
                if pos == -1 or fail_on_missing:
                    available = sorted([
                        *get_field_names_from_opts(opts),
                        *self.annotation_select,
                        *self._filtered_relations,
                    ])
                    raise FieldError("Cannot resolve keyword '%s' into field. "
                                     "Choices are: %s" % (name, ", ".join(available)))
                break
            # Check if we need any joins for concrete inheritance cases (the
            # field lives in parent, but we are currently in one of its
            # children)
            if model is not opts.model:
                path_to_parent = opts.get_path_to_parent(model)
                if path_to_parent:
                    path.extend(path_to_parent)
                    cur_names_with_path[1].extend(path_to_parent)
                    opts = path_to_parent[-1].to_opts
            if hasattr(field, 'get_path_info'):
                pathinfos = field.get_path_info(filtered_relation)
                if not allow_many:
                    for inner_pos, p in enumerate(pathinfos):
                        if p.m2m:
                            cur_names_with_path[1].extend(pathinfos[0:inner_pos + 1])
                            names_with_path.append(cur_names_with_path)
                            raise MultiJoin(pos + 1, names_with_path)
                last = pathinfos[-1]
                path.extend(pathinfos)
                final_field = last.join_field
                opts = last.to_opts
                targets = last.target_fields
                cur_names_with_path[1].extend(pathinfos)
                names_with_path.append(cur_names_with_path)
            else:
                # Local non-relational field.
                final_field = field
                targets = (field,)
                if fail_on_missing and pos + 1 != len(names):
                    raise FieldError(
                        "Cannot resolve keyword %r into field. Join on '%s'"
                        " not permitted." % (names[pos + 1], name))
                break
        return path, final_field, targets, names[pos + 1:]
```
### 22 - django/db/models/sql/query.py:

Start line: 1105, End line: 1137

```python
class Query(BaseExpression):

    def check_related_objects(self, field, value, opts):
        """Check the type of object passed to query relations."""
        if field.is_relation:
            # Check that the field and the queryset use the same model in a
            # query like .filter(author=Author.objects.all()). For example, the
            # opts would be Author's (from the author field) and value.model
            # would be Author.objects.all() queryset's .model (Author also).
            # The field is the related field on the lhs side.
            if (isinstance(value, Query) and not value.has_select_fields and
                    not check_rel_lookup_compatibility(value.model, opts, field)):
                raise ValueError(
                    'Cannot use QuerySet for "%s": Use a QuerySet for "%s".' %
                    (value.model._meta.object_name, opts.object_name)
                )
            elif hasattr(value, '_meta'):
                self.check_query_object_type(value, opts, field)
            elif hasattr(value, '__iter__'):
                for v in value:
                    self.check_query_object_type(v, opts, field)

    def check_filterable(self, expression):
        """Raise an error if expression cannot be used in a WHERE clause."""
        if (
            hasattr(expression, 'resolve_expression') and
            not getattr(expression, 'filterable', True)
        ):
            raise NotSupportedError(
                expression.__class__.__name__ + ' is disallowed in the filter '
                'clause.'
            )
        if hasattr(expression, 'get_source_expressions'):
            for expr in expression.get_source_expressions():
                self.check_filterable(expr)
```
### 24 - django/db/models/sql/query.py:

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
                    if selected_annotation is expr:
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
### 25 - django/db/models/sql/query.py:

Start line: 1049, End line: 1074

```python
class Query(BaseExpression):

    def get_external_cols(self):
        exprs = chain(self.annotations.values(), self.where.children)
        return [
            col for col in self._gen_cols(exprs)
            if col.alias in self.external_aliases
        ]

    def as_sql(self, compiler, connection):
        sql, params = self.get_compiler(connection=connection).as_sql()
        if self.subquery:
            sql = '(%s)' % sql
        return sql, params

    def resolve_lookup_value(self, value, can_reuse, allow_joins):
        if hasattr(value, 'resolve_expression'):
            value = value.resolve_expression(
                self, reuse=can_reuse, allow_joins=allow_joins,
            )
        elif isinstance(value, (list, tuple)):
            # The items of the iterable may be expressions and therefore need
            # to be resolved independently.
            return type(value)(
                self.resolve_lookup_value(sub_value, can_reuse, allow_joins)
                for sub_value in value
            )
        return value
```
### 31 - django/db/models/sql/query.py:

Start line: 1596, End line: 1620

```python
class Query(BaseExpression):

    def setup_joins(self, names, opts, alias, can_reuse=None, allow_many=True,
                    reuse_with_filtered_relation=False):
        # ... other code
        for join in path:
            if join.filtered_relation:
                filtered_relation = join.filtered_relation.clone()
                table_alias = filtered_relation.alias
            else:
                filtered_relation = None
                table_alias = None
            opts = join.to_opts
            if join.direct:
                nullable = self.is_nullable(join.join_field)
            else:
                nullable = True
            connection = Join(
                opts.db_table, alias, table_alias, INNER, join.join_field,
                nullable, filtered_relation=filtered_relation,
            )
            reuse = can_reuse if join.m2m or reuse_with_filtered_relation else None
            alias = self.join(
                connection, reuse=reuse,
                reuse_with_filtered_relation=reuse_with_filtered_relation,
            )
            joins.append(alias)
            if filtered_relation:
                filtered_relation.path = joins[:]
        return JoinInfo(final_field, targets, opts, joins, path, final_transformer)
```
### 35 - django/db/models/sql/query.py:

Start line: 510, End line: 544

```python
class Query(BaseExpression):

    def get_count(self, using):
        """
        Perform a COUNT() query using the current filter constraints.
        """
        obj = self.clone()
        obj.add_annotation(Count('*'), alias='__count', is_summary=True)
        number = obj.get_aggregation(using, ['__count'])['__count']
        if number is None:
            number = 0
        return number

    def has_filters(self):
        return self.where

    def has_results(self, using):
        q = self.clone()
        if not q.distinct:
            if q.group_by is True:
                q.add_fields((f.attname for f in self.model._meta.concrete_fields), False)
                # Disable GROUP BY aliases to avoid orphaning references to the
                # SELECT clause which is about to be cleared.
                q.set_group_by(allow_aliases=False)
            q.clear_select_clause()
        q.clear_ordering(True)
        q.set_limits(high=1)
        compiler = q.get_compiler(using=using)
        return compiler.has_results()

    def explain(self, using, format=None, **options):
        q = self.clone()
        q.explain_query = True
        q.explain_format = format
        q.explain_options = options
        compiler = q.get_compiler(using=using)
        return '\n'.join(compiler.explain_query())
```
### 39 - django/db/models/sql/query.py:

Start line: 2283, End line: 2299

```python
class Query(BaseExpression):

    def is_nullable(self, field):
        """
        Check if the given field should be treated as nullable.

        Some backends treat '' as null and Django treats such fields as
        nullable for those backends. In such situations field.null can be
        False even if we should treat the field as nullable.
        """
        # We need to use DEFAULT_DB_ALIAS here, as QuerySet does not have
        # (nor should it have) knowledge of which connection is going to be
        # used. The proper fix would be to defer all decisions where
        # is_nullable() is needed to the compiler stage, but that is not easy
        # to do currently.
        return (
            connections[DEFAULT_DB_ALIAS].features.interprets_empty_strings_as_nulls and
            field.empty_strings_allowed
        ) or field.null
```
### 42 - django/db/models/sql/query.py:

Start line: 136, End line: 230

```python
class Query(BaseExpression):
    """A single SQL query."""

    alias_prefix = 'T'
    subq_aliases = frozenset([alias_prefix])

    compiler = 'SQLCompiler'

    def __init__(self, model, where=WhereNode, alias_cols=True):
        self.model = model
        self.alias_refcount = {}
        # alias_map is the most important data structure regarding joins.
        # It's used for recording which joins exist in the query and what
        # types they are. The key is the alias of the joined table (possibly
        # the table name) and the value is a Join-like object (see
        # sql.datastructures.Join for more information).
        self.alias_map = {}
        # Whether to provide alias to columns during reference resolving.
        self.alias_cols = alias_cols
        # Sometimes the query contains references to aliases in outer queries (as
        # a result of split_exclude). Correct alias quoting needs to know these
        # aliases too.
        # Map external tables to whether they are aliased.
        self.external_aliases = {}
        self.table_map = {}     # Maps table names to list of aliases.
        self.default_cols = True
        self.default_ordering = True
        self.standard_ordering = True
        self.used_aliases = set()
        self.filter_is_sticky = False
        self.subquery = False

        # SQL-related attributes
        # Select and related select clauses are expressions to use in the
        # SELECT clause of the query.
        # The select is used for cases where we want to set up the select
        # clause to contain other than default fields (values(), subqueries...)
        # Note that annotations go to annotations dictionary.
        self.select = ()
        self.where = where()
        self.where_class = where
        # The group_by attribute can have one of the following forms:
        #  - None: no group by at all in the query
        #  - A tuple of expressions: group by (at least) those expressions.
        #    String refs are also allowed for now.
        #  - True: group by all select fields of the model
        # See compiler.get_group_by() for details.
        self.group_by = None
        self.order_by = ()
        self.low_mark, self.high_mark = 0, None  # Used for offset/limit
        self.distinct = False
        self.distinct_fields = ()
        self.select_for_update = False
        self.select_for_update_nowait = False
        self.select_for_update_skip_locked = False
        self.select_for_update_of = ()
        self.select_for_no_key_update = False

        self.select_related = False
        # Arbitrary limit for select_related to prevents infinite recursion.
        self.max_depth = 5

        # Holds the selects defined by a call to values() or values_list()
        # excluding annotation_select and extra_select.
        self.values_select = ()

        # SQL annotation-related attributes
        self.annotations = {}  # Maps alias -> Annotation Expression
        self.annotation_select_mask = None
        self._annotation_select_cache = None

        # Set combination attributes
        self.combinator = None
        self.combinator_all = False
        self.combined_queries = ()

        # These are for extensions. The contents are more or less appended
        # verbatim to the appropriate clause.
        self.extra = {}  # Maps col_alias -> (col_sql, params).
        self.extra_select_mask = None
        self._extra_select_cache = None

        self.extra_tables = ()
        self.extra_order_by = ()

        # A tuple that is a set of model field names and either True, if these
        # are the fields to defer, or False if these are the only fields to
        # load.
        self.deferred_loading = (frozenset(), True)

        self._filtered_relations = {}

        self.explain_query = False
        self.explain_format = None
        self.explain_options = {}
```
### 47 - django/db/models/sql/query.py:

Start line: 2128, End line: 2173

```python
class Query(BaseExpression):

    def set_values(self, fields):
        self.select_related = False
        self.clear_deferred_loading()
        self.clear_select_fields()

        if fields:
            field_names = []
            extra_names = []
            annotation_names = []
            if not self.extra and not self.annotations:
                # Shortcut - if there are no extra or annotations, then
                # the values() clause must be just field names.
                field_names = list(fields)
            else:
                self.default_cols = False
                for f in fields:
                    if f in self.extra_select:
                        extra_names.append(f)
                    elif f in self.annotation_select:
                        annotation_names.append(f)
                    else:
                        field_names.append(f)
            self.set_extra_mask(extra_names)
            self.set_annotation_mask(annotation_names)
        else:
            field_names = [f.attname for f in self.model._meta.concrete_fields]
        # Selected annotations must be known before setting the GROUP BY
        # clause.
        if self.group_by is True:
            self.add_fields((f.attname for f in self.model._meta.concrete_fields), False)
            # Disable GROUP BY aliases to avoid orphaning references to the
            # SELECT clause which is about to be cleared.
            self.set_group_by(allow_aliases=False)
            self.clear_select_fields()
        elif self.group_by:
            # Resolve GROUP BY annotation references if they are not part of
            # the selected fields anymore.
            group_by = []
            for expr in self.group_by:
                if isinstance(expr, Ref) and expr.refs not in field_names:
                    expr = self.annotations[expr.refs]
                group_by.append(expr)
            self.group_by = tuple(group_by)

        self.values_select = tuple(field_names)
        self.add_fields(field_names, True)
```
### 49 - django/db/models/sql/query.py:

Start line: 892, End line: 914

```python
class Query(BaseExpression):

    def bump_prefix(self, outer_query):
        # ... other code

        if self.alias_prefix != outer_query.alias_prefix:
            # No clashes between self and outer query should be possible.
            return

        # Explicitly avoid infinite loop. The constant divider is based on how
        # much depth recursive subquery references add to the stack. This value
        # might need to be adjusted when adding or removing function calls from
        # the code path in charge of performing these operations.
        local_recursion_limit = sys.getrecursionlimit() // 16
        for pos, prefix in enumerate(prefix_gen()):
            if prefix not in self.subq_aliases:
                self.alias_prefix = prefix
                break
            if pos > local_recursion_limit:
                raise RecursionError(
                    'Maximum recursion depth exceeded: too many subqueries.'
                )
        self.subq_aliases = self.subq_aliases.union([self.alias_prefix])
        outer_query.subq_aliases = outer_query.subq_aliases.union(self.subq_aliases)
        self.change_aliases({
            alias: '%s%d' % (self.alias_prefix, pos)
            for pos, alias in enumerate(self.alias_map)
        })
```
### 54 - django/db/models/sql/query.py:

Start line: 1800, End line: 1849

```python
class Query(BaseExpression):

    def clear_limits(self):
        """Clear any existing limits."""
        self.low_mark, self.high_mark = 0, None

    @property
    def is_sliced(self):
        return self.low_mark != 0 or self.high_mark is not None

    def has_limit_one(self):
        return self.high_mark is not None and (self.high_mark - self.low_mark) == 1

    def can_filter(self):
        """
        Return True if adding filters to this instance is still possible.

        Typically, this means no limits or offsets have been put on the results.
        """
        return not self.is_sliced

    def clear_select_clause(self):
        """Remove all fields from SELECT clause."""
        self.select = ()
        self.default_cols = False
        self.select_related = False
        self.set_extra_mask(())
        self.set_annotation_mask(())

    def clear_select_fields(self):
        """
        Clear the list of fields to select (but not extra_select columns).
        Some queryset types completely replace any existing list of select
        columns.
        """
        self.select = ()
        self.values_select = ()

    def add_select_col(self, col):
        self.select += col,
        self.values_select += col.output_field.name,

    def set_select(self, cols):
        self.default_cols = False
        self.select = tuple(cols)

    def add_distinct_fields(self, *field_names):
        """
        Add and resolve the given fields to the query's "distinct on" clause.
        """
        self.distinct_fields = field_names
        self.distinct = True
```
### 56 - django/db/models/sql/query.py:

Start line: 696, End line: 731

```python
class Query(BaseExpression):

    def deferred_to_data(self, target, callback):
        # ... other code

        if defer:
            # We need to load all fields for each model, except those that
            # appear in "seen" (for all models that appear in "seen"). The only
            # slight complexity here is handling fields that exist on parent
            # models.
            workset = {}
            for model, values in seen.items():
                for field in model._meta.local_fields:
                    if field not in values:
                        m = field.model._meta.concrete_model
                        add_to_dict(workset, m, field)
            for model, values in must_include.items():
                # If we haven't included a model in workset, we don't add the
                # corresponding must_include fields for that model, since an
                # empty set means "include all fields". That's why there's no
                # "else" branch here.
                if model in workset:
                    workset[model].update(values)
            for model, values in workset.items():
                callback(target, model, values)
        else:
            for model, values in must_include.items():
                if model in seen:
                    seen[model].update(values)
                else:
                    # As we've passed through this model, but not explicitly
                    # included any fields, we have to make sure it's mentioned
                    # so that only the "must include" fields are pulled in.
                    seen[model] = values
            # Now ensure that every model in the inheritance chain is mentioned
            # in the parent list. Again, it must be mentioned to ensure that
            # only "must include" fields are pulled in.
            for model in orig_opts.get_parent_list():
                seen.setdefault(model, set())
            for model, values in seen.items():
                callback(target, model, values)
```
### 64 - django/db/models/sql/query.py:

Start line: 621, End line: 645

```python
class Query(BaseExpression):

    def combine(self, rhs, connector):
        # ... other code
        if rhs.select:
            self.set_select([col.relabeled_clone(change_map) for col in rhs.select])
        else:
            self.select = ()

        if connector == OR:
            # It would be nice to be able to handle this, but the queries don't
            # really make sense (or return consistent value sets). Not worth
            # the extra complexity when you can write a real query instead.
            if self.extra and rhs.extra:
                raise ValueError("When merging querysets using 'or', you cannot have extra(select=...) on both sides.")
        self.extra.update(rhs.extra)
        extra_select_mask = set()
        if self.extra_select_mask is not None:
            extra_select_mask.update(self.extra_select_mask)
        if rhs.extra_select_mask is not None:
            extra_select_mask.update(rhs.extra_select_mask)
        if extra_select_mask:
            self.set_extra_mask(extra_select_mask)
        self.extra_tables += rhs.extra_tables

        # Ordering uses the 'rhs' ordering, unless it has none, in which case
        # the current ordering is used.
        self.order_by = rhs.order_by or self.order_by
        self.extra_order_by = rhs.extra_order_by or self.extra_order_by
```
### 72 - django/db/models/sql/query.py:

Start line: 1771, End line: 1798

```python
class Query(BaseExpression):

    def set_empty(self):
        self.where.add(NothingNode(), AND)

    def is_empty(self):
        return any(isinstance(c, NothingNode) for c in self.where.children)

    def set_limits(self, low=None, high=None):
        """
        Adjust the limits on the rows retrieved. Use low/high to set these,
        as it makes it more Pythonic to read and write. When the SQL query is
        created, convert them to the appropriate offset and limit values.

        Apply any limits passed in here to the existing constraints. Add low
        to the current low value and clamp both to any existing high value.
        """
        if high is not None:
            if self.high_mark is not None:
                self.high_mark = min(self.high_mark, self.low_mark + high)
            else:
                self.high_mark = self.low_mark + high
        if low is not None:
            if self.high_mark is not None:
                self.low_mark = min(self.high_mark, self.low_mark + low)
            else:
                self.low_mark = self.low_mark + low

        if self.low_mark == self.high_mark:
            self.set_empty()
```
### 73 - django/db/models/sql/query.py:

Start line: 1, End line: 65

```python
"""
Create SQL statements for QuerySets.

The code in here encapsulates all of the SQL construction so that QuerySets
themselves do not have to (and could be backed by things other than SQL
databases). The abstraction barrier only works one way: this module has to know
all about the internals of models in order to get the information it needs.
"""
import copy
import difflib
import functools
import inspect
import sys
import warnings
from collections import Counter, namedtuple
from collections.abc import Iterator, Mapping
from itertools import chain, count, product
from string import ascii_uppercase

from django.core.exceptions import (
    EmptyResultSet, FieldDoesNotExist, FieldError,
)
from django.db import DEFAULT_DB_ALIAS, NotSupportedError, connections
from django.db.models.aggregates import Count
from django.db.models.constants import LOOKUP_SEP
from django.db.models.expressions import BaseExpression, Col, F, OuterRef, Ref
from django.db.models.fields import Field
from django.db.models.fields.related_lookups import MultiColSource
from django.db.models.lookups import Lookup
from django.db.models.query_utils import (
    Q, check_rel_lookup_compatibility, refs_expression,
)
from django.db.models.sql.constants import INNER, LOUTER, ORDER_DIR, SINGLE
from django.db.models.sql.datastructures import (
    BaseTable, Empty, Join, MultiJoin,
)
from django.db.models.sql.where import (
    AND, OR, ExtraWhere, NothingNode, WhereNode,
)
from django.utils.deprecation import RemovedInDjango40Warning
from django.utils.functional import cached_property
from django.utils.tree import Node

__all__ = ['Query', 'RawQuery']


def get_field_names_from_opts(opts):
    return set(chain.from_iterable(
        (f.name, f.attname) if f.concrete else (f.name,)
        for f in opts.get_fields()
    ))


def get_children_from_q(q):
    for child in q.children:
        if isinstance(child, Node):
            yield from get_children_from_q(child)
        else:
            yield child


JoinInfo = namedtuple(
    'JoinInfo',
    ('final_field', 'targets', 'opts', 'joins', 'path', 'transform_function')
)
```
### 88 - django/db/models/sql/query.py:

Start line: 2175, End line: 2207

```python
class Query(BaseExpression):

    @property
    def annotation_select(self):
        """
        Return the dictionary of aggregate columns that are not masked and
        should be used in the SELECT clause. Cache this result for performance.
        """
        if self._annotation_select_cache is not None:
            return self._annotation_select_cache
        elif not self.annotations:
            return {}
        elif self.annotation_select_mask is not None:
            self._annotation_select_cache = {
                k: v for k, v in self.annotations.items()
                if k in self.annotation_select_mask
            }
            return self._annotation_select_cache
        else:
            return self.annotations

    @property
    def extra_select(self):
        if self._extra_select_cache is not None:
            return self._extra_select_cache
        if not self.extra:
            return {}
        elif self.extra_select_mask is not None:
            self._extra_select_cache = {
                k: v for k, v in self.extra.items()
                if k in self.extra_select_mask
            }
            return self._extra_select_cache
        else:
            return self.extra
```
### 98 - django/db/models/sql/query.py:

Start line: 1935, End line: 1984

```python
class Query(BaseExpression):

    def clear_ordering(self, force_empty):
        """
        Remove any ordering settings. If 'force_empty' is True, there will be
        no ordering in the resulting query (not even the model's default).
        """
        self.order_by = ()
        self.extra_order_by = ()
        if force_empty:
            self.default_ordering = False

    def set_group_by(self, allow_aliases=True):
        """
        Expand the GROUP BY clause required by the query.

        This will usually be the set of all non-aggregate fields in the
        return data. If the database backend supports grouping by the
        primary key, and the query would be equivalent, the optimization
        will be made automatically.
        """
        # Column names from JOINs to check collisions with aliases.
        if allow_aliases:
            column_names = set()
            seen_models = set()
            for join in list(self.alias_map.values())[1:]:  # Skip base table.
                model = join.join_field.related_model
                if model not in seen_models:
                    column_names.update({
                        field.column
                        for field in model._meta.local_concrete_fields
                    })
                    seen_models.add(model)

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
                    if not allow_aliases or alias in column_names:
                        alias = None
                    group_by_cols = annotation.get_group_by_cols(alias=alias)
                group_by.extend(group_by_cols)
        self.group_by = tuple(group_by)
```
### 109 - django/db/models/sql/query.py:

Start line: 1139, End line: 1182

```python
class Query(BaseExpression):

    def build_lookup(self, lookups, lhs, rhs):
        """
        Try to extract transforms and lookup from given lhs.

        The lhs value is something that works like SQLExpression.
        The rhs value is what the lookup is going to compare against.
        The lookups is a list of names to extract using get_lookup()
        and get_transform().
        """
        # __exact is the default lookup if one isn't given.
        *transforms, lookup_name = lookups or ['exact']
        for name in transforms:
            lhs = self.try_transform(lhs, name)
        # First try get_lookup() so that the lookup takes precedence if the lhs
        # supports both transform and lookup for the name.
        lookup_class = lhs.get_lookup(lookup_name)
        if not lookup_class:
            if lhs.field.is_relation:
                raise FieldError('Related Field got invalid lookup: {}'.format(lookup_name))
            # A lookup wasn't found. Try to interpret the name as a transform
            # and do an Exact lookup against it.
            lhs = self.try_transform(lhs, lookup_name)
            lookup_name = 'exact'
            lookup_class = lhs.get_lookup(lookup_name)
            if not lookup_class:
                return

        lookup = lookup_class(lhs, rhs)
        # Interpret '__exact=None' as the sql 'is NULL'; otherwise, reject all
        # uses of None as a query value unless the lookup supports it.
        if lookup.rhs is None and not lookup.can_use_none_as_rhs:
            if lookup_name not in ('exact', 'iexact'):
                raise ValueError("Cannot use None as a query value")
            return lhs.get_lookup('isnull')(lhs, True)

        # For Oracle '' is equivalent to null. The check must be done at this
        # stage because join promotion can't be done in the compiler. Using
        # DEFAULT_DB_ALIAS isn't nice but it's the best that can be done here.
        # A similar thing is done in is_nullable(), too.
        if (connections[DEFAULT_DB_ALIAS].features.interprets_empty_strings_as_nulls and
                lookup_name == 'exact' and lookup.rhs == ''):
            return lhs.get_lookup('isnull')(lhs, True)

        return lookup
```
### 134 - django/db/models/sql/query.py:

Start line: 232, End line: 286

```python
class Query(BaseExpression):

    @property
    def output_field(self):
        if len(self.select) == 1:
            select = self.select[0]
            return getattr(select, 'target', None) or select.field
        elif len(self.annotation_select) == 1:
            return next(iter(self.annotation_select.values())).output_field

    @property
    def has_select_fields(self):
        return bool(self.select or self.annotation_select_mask or self.extra_select_mask)

    @cached_property
    def base_table(self):
        for alias in self.alias_map:
            return alias

    def __str__(self):
        """
        Return the query as a string of SQL with the parameter values
        substituted in (use sql_with_params() to see the unsubstituted string).

        Parameter values won't necessarily be quoted correctly, since that is
        done by the database interface at execution time.
        """
        sql, params = self.sql_with_params()
        return sql % params

    def sql_with_params(self):
        """
        Return the query as an SQL string and the parameters that will be
        substituted into the query.
        """
        return self.get_compiler(DEFAULT_DB_ALIAS).as_sql()

    def __deepcopy__(self, memo):
        """Limit the amount of work when a Query is deepcopied."""
        result = self.clone()
        memo[id(self)] = result
        return result

    def get_compiler(self, using=None, connection=None):
        if using is None and connection is None:
            raise ValueError("Need either using or connection")
        if using:
            connection = connections[using]
        return connection.ops.compiler(self.compiler)(self, connection, using)

    def get_meta(self):
        """
        Return the Options instance (the model._meta) from which to start
        processing. Normally, this is self.model._meta, but it can be changed
        by subclasses.
        """
        return self.model._meta
```
### 137 - django/db/models/sql/query.py:

Start line: 1076, End line: 1103

```python
class Query(BaseExpression):

    def solve_lookup_type(self, lookup):
        """
        Solve the lookup type from the lookup (e.g.: 'foobar__id__icontains').
        """
        lookup_splitted = lookup.split(LOOKUP_SEP)
        if self.annotations:
            expression, expression_lookups = refs_expression(lookup_splitted, self.annotations)
            if expression:
                return expression_lookups, (), expression
        _, field, _, lookup_parts = self.names_to_path(lookup_splitted, self.get_meta())
        field_parts = lookup_splitted[0:len(lookup_splitted) - len(lookup_parts)]
        if len(lookup_parts) > 1 and not field_parts:
            raise FieldError(
                'Invalid lookup "%s" for model %s".' %
                (lookup, self.get_meta().model.__name__)
            )
        return lookup_parts, field_parts, False

    def check_query_object_type(self, value, opts, field):
        """
        Check whether the object passed while querying is of the correct type.
        If not, raise a ValueError specifying the wrong object.
        """
        if hasattr(value, '_meta'):
            if not check_rel_lookup_compatibility(value._meta.model, opts, field):
                raise ValueError(
                    'Cannot query "%s": Must be "%s" instance.' %
                    (value, opts.object_name))
```
### 139 - django/db/models/sql/query.py:

Start line: 2363, End line: 2418

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
### 152 - django/db/models/sql/query.py:

Start line: 2034, End line: 2056

```python
class Query(BaseExpression):

    def clear_deferred_loading(self):
        """Remove any fields from the deferred loading set."""
        self.deferred_loading = (frozenset(), True)

    def add_deferred_loading(self, field_names):
        """
        Add the given list of model field names to the set of fields to
        exclude from loading from the database when automatic column selection
        is done. Add the new field names to any existing field names that
        are deferred (or removed from any existing field names that are marked
        as the only ones for immediate loading).
        """
        # Fields on related models are stored in the literal double-underscore
        # format, so that we can use a set datastructure. We do the foo__bar
        # splitting and handling when computing the SQL column names (as part of
        # get_columns()).
        existing, defer = self.deferred_loading
        if defer:
            # Add to existing deferred names.
            self.deferred_loading = existing.union(field_names), True
        else:
            # Remove names from the set of any existing "immediate load" names.
            self.deferred_loading = existing.difference(field_names), False
```
### 159 - django/db/models/sql/query.py:

Start line: 984, End line: 1015

```python
class Query(BaseExpression):

    def join_parent_model(self, opts, model, alias, seen):
        """
        Make sure the given 'model' is joined in the query. If 'model' isn't
        a parent of 'opts' or if it is None this method is a no-op.

        The 'alias' is the root alias for starting the join, 'seen' is a dict
        of model -> alias of existing joins. It must also contain a mapping
        of None -> some alias. This will be returned in the no-op case.
        """
        if model in seen:
            return seen[model]
        chain = opts.get_base_chain(model)
        if not chain:
            return alias
        curr_opts = opts
        for int_model in chain:
            if int_model in seen:
                curr_opts = int_model._meta
                alias = seen[int_model]
                continue
            # Proxy model have elements in base chain
            # with no parents, assign the new options
            # object and skip to the next base in that
            # case
            if not curr_opts.parents[int_model]:
                curr_opts = int_model._meta
                continue
            link_field = curr_opts.get_ancestor_link(int_model)
            join_info = self.setup_joins([link_field.name], curr_opts, alias)
            curr_opts = int_model._meta
            alias = seen[int_model] = join_info.joins[-1]
        return alias or seen[None]
```
### 175 - django/db/models/sql/query.py:

Start line: 1344, End line: 1365

```python
class Query(BaseExpression):

    def add_filter(self, filter_clause):
        self.add_q(Q(**{filter_clause[0]: filter_clause[1]}))

    def add_q(self, q_object):
        """
        A preprocessor for the internal _add_q(). Responsible for doing final
        join promotion.
        """
        # For join promotion this case is doing an AND for the added q_object
        # and existing conditions. So, any existing inner join forces the join
        # type to remain inner. Existing outer joins can however be demoted.
        # (Consider case where rel_a is LOUTER and rel_a__col=1 is added - if
        # rel_a doesn't produce any rows, then the whole condition must fail.
        # So, demotion is OK.
        existing_inner = {a for a in self.alias_map if self.alias_map[a].join_type == INNER}
        clause, _ = self._add_q(q_object, self.used_aliases)
        if clause:
            self.where.add(clause, AND)
        self.demote_joins(existing_inner)

    def build_where(self, filter_expr):
        return self.build_filter(filter_expr, allow_joins=False)[0]
```
