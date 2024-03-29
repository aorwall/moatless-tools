# django__django-15382

| **django/django** | `770d3e6a4ce8e0a91a9e27156036c1985e74d4a3` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 1618 |
| **Any found context length** | 1618 |
| **Avg pos** | 4.0 |
| **Min pos** | 4 |
| **Max pos** | 4 |
| **Top file pos** | 4 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/db/models/expressions.py b/django/db/models/expressions.py
--- a/django/db/models/expressions.py
+++ b/django/db/models/expressions.py
@@ -1211,13 +1211,18 @@ def __invert__(self):
 
     def as_sql(self, compiler, connection, template=None, **extra_context):
         query = self.query.exists(using=connection.alias)
-        sql, params = super().as_sql(
-            compiler,
-            connection,
-            template=template,
-            query=query,
-            **extra_context,
-        )
+        try:
+            sql, params = super().as_sql(
+                compiler,
+                connection,
+                template=template,
+                query=query,
+                **extra_context,
+            )
+        except EmptyResultSet:
+            if self.negated:
+                return '', ()
+            raise
         if self.negated:
             sql = 'NOT {}'.format(sql)
         return sql, params

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/db/models/expressions.py | 1214 | 1220 | 4 | 4 | 1618


## Problem Statement

```
filter on exists-subquery with empty queryset removes whole WHERE block
Description
	 
		(last modified by Tobias Bengfort)
	 
>>> qs = MyModel.objects.filter(~models.Exists(MyModel.objects.none()), name='test')
>>> qs
<QuerySet []>
>>> print(qs.query)
EmptyResultSet
With django-debug-toolbar I can still see the query, but there WHERE block is missing completely.
This seems to be very similar to #33018.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 django/db/models/query.py | 1061 | 1109| 371 | 371 | 18494 | 
| 2 | 2 django/contrib/admin/filters.py | 449 | 479| 233 | 604 | 22634 | 
| 3 | 3 django/db/models/sql/query.py | 1320 | 1384| 760 | 1364 | 45352 | 
| **-> 4 <-** | **4 django/db/models/expressions.py** | 1199 | 1231| 254 | 1618 | 56777 | 
| 5 | 4 django/db/models/sql/query.py | 1873 | 1922| 329 | 1947 | 56777 | 
| 6 | 4 django/db/models/query.py | 1564 | 1595| 245 | 2192 | 56777 | 
| 7 | 4 django/db/models/sql/query.py | 1775 | 1840| 665 | 2857 | 56777 | 
| 8 | 4 django/db/models/query.py | 1150 | 1163| 126 | 2983 | 56777 | 
| 9 | 4 django/db/models/sql/query.py | 1454 | 1481| 283 | 3266 | 56777 | 
| 10 | 4 django/db/models/query.py | 355 | 372| 132 | 3398 | 56777 | 
| 11 | 4 django/db/models/query.py | 325 | 353| 242 | 3640 | 56777 | 
| 12 | 5 django/db/models/sql/where.py | 228 | 246| 131 | 3771 | 58734 | 
| 13 | 5 django/db/models/query.py | 949 | 980| 272 | 4043 | 58734 | 
| 14 | 5 django/db/models/query.py | 1111 | 1126| 124 | 4167 | 58734 | 
| 15 | 5 django/db/models/sql/query.py | 1499 | 1587| 816 | 4983 | 58734 | 
| 16 | 5 django/db/models/query.py | 1128 | 1148| 240 | 5223 | 58734 | 
| 17 | 5 django/db/models/sql/query.py | 1386 | 1410| 260 | 5483 | 58734 | 
| 18 | 5 django/db/models/query.py | 839 | 867| 264 | 5747 | 58734 | 
| 19 | 5 django/db/models/query.py | 1647 | 1678| 295 | 6042 | 58734 | 
| 20 | 5 django/db/models/query.py | 1241 | 1282| 323 | 6365 | 58734 | 
| 21 | 5 django/db/models/query.py | 1478 | 1496| 186 | 6551 | 58734 | 
| 22 | 6 django/db/models/sql/subqueries.py | 1 | 45| 309 | 6860 | 59948 | 
| 23 | 6 django/db/models/query.py | 176 | 235| 469 | 7329 | 59948 | 
| 24 | 6 django/db/models/sql/query.py | 1663 | 1684| 202 | 7531 | 59948 | 
| 25 | 7 django/db/models/query_utils.py | 279 | 319| 286 | 7817 | 62436 | 
| 26 | 7 django/db/models/sql/where.py | 249 | 266| 141 | 7958 | 62436 | 
| 27 | 7 django/db/models/query.py | 237 | 264| 221 | 8179 | 62436 | 
| 28 | 7 django/db/models/sql/query.py | 513 | 555| 375 | 8554 | 62436 | 
| 29 | 7 django/db/models/query.py | 1339 | 1358| 209 | 8763 | 62436 | 
| 30 | 7 django/contrib/admin/filters.py | 435 | 447| 142 | 8905 | 62436 | 
| 31 | 7 django/db/models/sql/query.py | 1433 | 1452| 204 | 9109 | 62436 | 
| 32 | 7 django/db/models/sql/query.py | 1247 | 1318| 694 | 9803 | 62436 | 
| 33 | 7 django/db/models/query.py | 266 | 286| 180 | 9983 | 62436 | 
| 34 | 7 django/db/models/query.py | 869 | 902| 274 | 10257 | 62436 | 
| 35 | 8 django/db/models/manager.py | 168 | 204| 201 | 10458 | 63879 | 
| 36 | 8 django/db/models/sql/query.py | 2004 | 2016| 129 | 10587 | 63879 | 
| 37 | 8 django/db/models/query.py | 780 | 798| 178 | 10765 | 63879 | 
| 38 | 8 django/db/models/sql/query.py | 142 | 237| 828 | 11593 | 63879 | 
| 39 | 8 django/db/models/sql/query.py | 1412 | 1431| 245 | 11838 | 63879 | 
| 40 | 8 django/db/models/query.py | 982 | 1011| 248 | 12086 | 63879 | 
| 41 | 8 django/db/models/sql/query.py | 916 | 941| 268 | 12354 | 63879 | 
| 42 | 8 django/db/models/query.py | 928 | 947| 163 | 12517 | 63879 | 
| 43 | 8 django/db/models/sql/query.py | 717 | 752| 389 | 12906 | 63879 | 
| 44 | 8 django/db/models/query.py | 1509 | 1561| 448 | 13354 | 63879 | 
| 45 | 9 django/db/models/sql/compiler.py | 1481 | 1513| 254 | 13608 | 78782 | 
| 46 | 9 django/db/models/sql/query.py | 1 | 62| 450 | 14058 | 78782 | 
| 47 | 9 django/db/models/sql/query.py | 1145 | 1177| 338 | 14396 | 78782 | 
| 48 | 10 django/db/models/deletion.py | 347 | 363| 123 | 14519 | 82640 | 
| 49 | 10 django/db/models/query_utils.py | 29 | 57| 227 | 14746 | 82640 | 
| 50 | 10 django/db/models/sql/query.py | 2196 | 2243| 394 | 15140 | 82640 | 
| 51 | 10 django/db/models/sql/query.py | 1731 | 1773| 436 | 15576 | 82640 | 
| 52 | 10 django/db/models/query_utils.py | 59 | 91| 288 | 15864 | 82640 | 
| 53 | 11 django/db/models/__init__.py | 1 | 53| 619 | 16483 | 83259 | 
| 54 | 12 django/forms/models.py | 1216 | 1281| 543 | 17026 | 95221 | 
| 55 | 12 django/db/models/sql/query.py | 119 | 139| 173 | 17199 | 95221 | 
| 56 | 13 django/shortcuts.py | 81 | 99| 200 | 17399 | 96318 | 
| 57 | 13 django/db/models/query.py | 1 | 40| 311 | 17710 | 96318 | 
| 58 | **13 django/db/models/expressions.py** | 1141 | 1196| 437 | 18147 | 96318 | 
| 59 | 13 django/db/models/sql/where.py | 65 | 115| 396 | 18543 | 96318 | 
| 60 | 14 django/contrib/admin/options.py | 994 | 1027| 307 | 18850 | 115108 | 
| 61 | 14 django/db/models/query.py | 496 | 553| 476 | 19326 | 115108 | 
| 62 | 14 django/contrib/admin/filters.py | 20 | 59| 295 | 19621 | 115108 | 
| 63 | 14 django/db/models/sql/query.py | 1002 | 1033| 307 | 19928 | 115108 | 
| 64 | 14 django/db/models/sql/subqueries.py | 138 | 166| 187 | 20115 | 115108 | 
| 65 | 14 django/db/models/sql/query.py | 1098 | 1114| 152 | 20267 | 115108 | 
| 66 | 14 django/db/models/query.py | 1453 | 1476| 199 | 20466 | 115108 | 
| 67 | 14 django/db/models/sql/where.py | 1 | 30| 167 | 20633 | 115108 | 
| 68 | 14 django/db/models/query.py | 43 | 90| 457 | 21090 | 115108 | 
| 69 | 14 django/db/models/sql/query.py | 239 | 293| 409 | 21499 | 115108 | 
| 70 | 14 django/db/models/query.py | 1226 | 1239| 115 | 21614 | 115108 | 
| 71 | 14 django/db/models/sql/query.py | 1070 | 1096| 220 | 21834 | 115108 | 
| 72 | 14 django/db/models/sql/query.py | 1714 | 1729| 132 | 21966 | 115108 | 
| 73 | 14 django/db/models/sql/query.py | 368 | 418| 494 | 22460 | 115108 | 
| 74 | 14 django/db/models/sql/subqueries.py | 48 | 76| 208 | 22668 | 115108 | 
| 75 | 14 django/db/models/query.py | 1284 | 1320| 327 | 22995 | 115108 | 
| 76 | 15 django/views/generic/detail.py | 58 | 76| 154 | 23149 | 116423 | 
| 77 | 16 django/contrib/admin/views/main.py | 452 | 507| 463 | 23612 | 120890 | 
| 78 | 16 django/db/models/sql/query.py | 295 | 341| 425 | 24037 | 120890 | 
| 79 | 16 django/db/models/query.py | 903 | 926| 207 | 24244 | 120890 | 
| 80 | 16 django/db/models/sql/query.py | 2279 | 2354| 781 | 25025 | 120890 | 
| 81 | 17 django/views/generic/dates.py | 318 | 342| 229 | 25254 | 126331 | 
| 82 | 17 django/db/models/sql/query.py | 2065 | 2099| 295 | 25549 | 126331 | 
| 83 | 17 django/db/models/query.py | 1203 | 1224| 214 | 25763 | 126331 | 
| 84 | 18 django/contrib/gis/db/models/__init__.py | 1 | 19| 204 | 25967 | 126535 | 
| 85 | 18 django/contrib/admin/filters.py | 62 | 115| 411 | 26378 | 126535 | 
| 86 | 18 django/db/models/sql/compiler.py | 1251 | 1273| 244 | 26622 | 126535 | 
| 87 | 18 django/db/models/query.py | 1680 | 1726| 336 | 26958 | 126535 | 
| 88 | 19 django/db/models/base.py | 1130 | 1173| 404 | 27362 | 144196 | 
| 89 | 19 django/db/models/query.py | 1390 | 1421| 252 | 27614 | 144196 | 
| 90 | 19 django/db/models/sql/query.py | 65 | 117| 384 | 27998 | 144196 | 
| 91 | 19 django/db/models/base.py | 992 | 1009| 184 | 28182 | 144196 | 
| 92 | 19 django/db/models/sql/query.py | 1842 | 1871| 259 | 28441 | 144196 | 
| 93 | 19 django/contrib/admin/filters.py | 163 | 208| 427 | 28868 | 144196 | 
| 94 | 19 django/db/models/query.py | 1181 | 1201| 155 | 29023 | 144196 | 
| 95 | 19 django/db/models/query.py | 93 | 111| 138 | 29161 | 144196 | 
| 96 | 19 django/db/models/query.py | 1729 | 1785| 487 | 29648 | 144196 | 
| 97 | 19 django/forms/models.py | 635 | 652| 167 | 29815 | 144196 | 
| 98 | 19 django/forms/models.py | 1335 | 1352| 152 | 29967 | 144196 | 
| 99 | 19 django/forms/models.py | 316 | 355| 387 | 30354 | 144196 | 
| 100 | 19 django/contrib/admin/views/main.py | 131 | 222| 851 | 31205 | 144196 | 
| 101 | 19 django/db/models/sql/where.py | 192 | 200| 115 | 31320 | 144196 | 
| 102 | 19 django/db/models/query_utils.py | 1 | 26| 183 | 31503 | 144196 | 
| 103 | 19 django/db/models/query.py | 374 | 396| 220 | 31723 | 144196 | 
| 104 | 19 django/contrib/admin/options.py | 1029 | 1048| 199 | 31922 | 144196 | 
| 105 | 19 django/db/models/query.py | 1360 | 1388| 183 | 32105 | 144196 | 
| 106 | 19 django/db/models/sql/query.py | 1686 | 1712| 276 | 32381 | 144196 | 
| 107 | 19 django/db/models/query.py | 600 | 637| 390 | 32771 | 144196 | 
| 108 | 19 django/db/models/query.py | 114 | 141| 233 | 33004 | 144196 | 
| 109 | 19 django/contrib/admin/filters.py | 281 | 305| 217 | 33221 | 144196 | 
| 110 | 19 django/db/models/sql/query.py | 2356 | 2372| 177 | 33398 | 144196 | 
| 111 | 19 django/db/models/sql/query.py | 343 | 366| 179 | 33577 | 144196 | 
| 112 | 19 django/db/models/deletion.py | 270 | 345| 800 | 34377 | 144196 | 
| 113 | 19 django/db/models/sql/compiler.py | 1 | 20| 171 | 34548 | 144196 | 
| 114 | 19 django/db/models/sql/query.py | 2150 | 2167| 156 | 34704 | 144196 | 
| 115 | 19 django/db/models/sql/query.py | 629 | 666| 376 | 35080 | 144196 | 
| 116 | 19 django/db/models/base.py | 1816 | 1916| 727 | 35807 | 144196 | 
| 117 | 19 django/db/models/base.py | 1565 | 1587| 171 | 35978 | 144196 | 
| 118 | 19 django/contrib/admin/filters.py | 400 | 422| 211 | 36189 | 144196 | 
| 119 | 19 django/db/models/deletion.py | 1 | 75| 561 | 36750 | 144196 | 
| 120 | 20 django/db/models/lookups.py | 387 | 424| 337 | 37087 | 149460 | 
| 121 | 20 django/db/models/sql/query.py | 2018 | 2063| 356 | 37443 | 149460 | 
| 122 | 20 django/db/models/sql/query.py | 1116 | 1143| 285 | 37728 | 149460 | 
| 123 | 20 django/db/models/deletion.py | 78 | 98| 217 | 37945 | 149460 | 
| 124 | 20 django/db/models/sql/where.py | 158 | 190| 221 | 38166 | 149460 | 
| 125 | **20 django/db/models/expressions.py** | 612 | 651| 290 | 38456 | 149460 | 
| 126 | 21 django/db/backends/base/schema.py | 1291 | 1310| 163 | 38619 | 162349 | 
| 127 | 21 django/db/models/sql/compiler.py | 23 | 58| 360 | 38979 | 162349 | 
| 128 | 21 django/db/models/sql/where.py | 202 | 225| 199 | 39178 | 162349 | 
| 129 | 21 django/db/models/sql/compiler.py | 74 | 159| 890 | 40068 | 162349 | 
| 130 | 21 django/db/models/query.py | 144 | 173| 207 | 40275 | 162349 | 
| 131 | 21 django/db/models/sql/query.py | 1179 | 1225| 474 | 40749 | 162349 | 
| 132 | 21 django/db/models/lookups.py | 319 | 331| 168 | 40917 | 162349 | 
| 133 | 21 django/contrib/admin/filters.py | 210 | 227| 190 | 41107 | 162349 | 
| 134 | **21 django/db/models/expressions.py** | 991 | 1056| 602 | 41709 | 162349 | 
| 135 | 21 django/db/models/sql/query.py | 2101 | 2124| 257 | 41966 | 162349 | 
| 136 | 21 django/forms/models.py | 96 | 109| 157 | 42123 | 162349 | 
| 137 | 22 django/db/models/fields/related.py | 336 | 357| 219 | 42342 | 176508 | 
| 138 | 22 django/contrib/admin/options.py | 888 | 902| 125 | 42467 | 176508 | 
| 139 | 22 django/db/models/sql/query.py | 2245 | 2277| 228 | 42695 | 176508 | 
| 140 | 22 django/db/models/query_utils.py | 251 | 276| 293 | 42988 | 176508 | 
| 141 | 23 django/template/base.py | 699 | 734| 270 | 43258 | 184703 | 
| 142 | 23 django/db/models/query.py | 1817 | 1932| 1098 | 44356 | 184703 | 
| 143 | 23 django/db/models/base.py | 1175 | 1202| 286 | 44642 | 184703 | 
| 144 | 23 django/db/models/query.py | 479 | 494| 130 | 44772 | 184703 | 
| 145 | 24 django/db/backends/oracle/schema.py | 49 | 63| 133 | 44905 | 186875 | 
| 146 | 24 django/contrib/admin/options.py | 376 | 428| 499 | 45404 | 186875 | 
| 147 | 25 django/db/models/aggregates.py | 50 | 70| 294 | 45698 | 188309 | 
| 148 | 25 django/db/models/aggregates.py | 17 | 48| 297 | 45995 | 188309 | 
| 149 | 25 django/db/models/base.py | 1312 | 1343| 267 | 46262 | 188309 | 
| 150 | 25 django/db/models/lookups.py | 629 | 641| 124 | 46386 | 188309 | 
| 151 | 25 django/contrib/admin/options.py | 1050 | 1072| 198 | 46584 | 188309 | 
| 152 | 25 django/db/backends/base/schema.py | 32 | 50| 173 | 46757 | 188309 | 
| 153 | 25 django/db/models/fields/related.py | 139 | 166| 201 | 46958 | 188309 | 
| 154 | 26 django/contrib/postgres/search.py | 160 | 195| 313 | 47271 | 190635 | 
| 155 | 26 django/db/models/sql/compiler.py | 1515 | 1536| 214 | 47485 | 190635 | 
| 156 | 26 django/db/models/query.py | 686 | 711| 202 | 47687 | 190635 | 


### Hint

```
I think that this is an issue with Exists.as_sql when self.invert is True. Since Exists encapsulate its negation logic (see __invert__) it should catch EmptyResultSet when raised by its super() call in as_sql and return an always true predicate (e.g. 1=1). Does the following patch address your issue? django/db/models/expressions.py diff --git a/django/db/models/expressions.py b/django/db/models/expressions.py index 81f8f79c71..7ec5dad57e 100644 a b def __invert__(self): 12111211 12121212 def as_sql(self, compiler, connection, template=None, **extra_context): 12131213 query = self.query.exists(using=connection.alias) 1214 sql, params = super().as_sql( 1215 compiler, 1216 connection, 1217 template=template, 1218 query=query, 1219 **extra_context, 1220 ) 1214 try: 1215 sql, params = super().as_sql( 1216 compiler, 1217 connection, 1218 template=template, 1219 query=query, 1220 **extra_context, 1221 ) 1222 except EmptyResultSet: 1223 if self.negated: 1224 return '%s = %s', (1, 1) 1225 raise 12211226 if self.negated: 12221227 sql = 'NOT {}'.format(sql) 12231228 return sql, params tests/expressions/tests.py diff --git a/tests/expressions/tests.py b/tests/expressions/tests.py index 5cf9dd1ea5..5d902c86e8 100644 a b def test_optimizations(self): 19051905 ) 19061906 self.assertNotIn('ORDER BY', captured_sql) 19071907 1908 def test_negated_empty_exists(self): 1909 manager = Manager.objects.create() 1910 qs = Manager.objects.filter( 1911 ~Exists(Manager.objects.none()), pk=manager.pk 1912 ) 1913 self.assertQuerysetEqual(qs, Manager.objects.filter(pk=manager.pk)) 1914 19081915 19091916class FieldTransformTests(TestCase):
```

## Patch

```diff
diff --git a/django/db/models/expressions.py b/django/db/models/expressions.py
--- a/django/db/models/expressions.py
+++ b/django/db/models/expressions.py
@@ -1211,13 +1211,18 @@ def __invert__(self):
 
     def as_sql(self, compiler, connection, template=None, **extra_context):
         query = self.query.exists(using=connection.alias)
-        sql, params = super().as_sql(
-            compiler,
-            connection,
-            template=template,
-            query=query,
-            **extra_context,
-        )
+        try:
+            sql, params = super().as_sql(
+                compiler,
+                connection,
+                template=template,
+                query=query,
+                **extra_context,
+            )
+        except EmptyResultSet:
+            if self.negated:
+                return '', ()
+            raise
         if self.negated:
             sql = 'NOT {}'.format(sql)
         return sql, params

```

## Test Patch

```diff
diff --git a/tests/expressions/tests.py b/tests/expressions/tests.py
--- a/tests/expressions/tests.py
+++ b/tests/expressions/tests.py
@@ -1905,6 +1905,13 @@ def test_optimizations(self):
         )
         self.assertNotIn('ORDER BY', captured_sql)
 
+    def test_negated_empty_exists(self):
+        manager = Manager.objects.create()
+        qs = Manager.objects.filter(
+            ~Exists(Manager.objects.none()) & Q(pk=manager.pk)
+        )
+        self.assertSequenceEqual(qs, [manager])
+
 
 class FieldTransformTests(TestCase):
 

```


## Code snippets

### 1 - django/db/models/query.py:

Start line: 1061, End line: 1109

```python
class QuerySet:

    def none(self):
        """Return an empty QuerySet."""
        clone = self._chain()
        clone.query.set_empty()
        return clone

    ##################################################################
    # PUBLIC METHODS THAT ALTER ATTRIBUTES AND RETURN A NEW QUERYSET #
    ##################################################################

    def all(self):
        """
        Return a new QuerySet that is a copy of the current one. This allows a
        QuerySet to proxy for a model manager in some cases.
        """
        return self._chain()

    def filter(self, *args, **kwargs):
        """
        Return a new QuerySet instance with the args ANDed to the existing
        set.
        """
        self._not_support_combined_queries('filter')
        return self._filter_or_exclude(False, args, kwargs)

    def exclude(self, *args, **kwargs):
        """
        Return a new QuerySet instance with NOT (args) ANDed to the existing
        set.
        """
        self._not_support_combined_queries('exclude')
        return self._filter_or_exclude(True, args, kwargs)

    def _filter_or_exclude(self, negate, args, kwargs):
        if (args or kwargs) and self.query.is_sliced:
            raise TypeError('Cannot filter a query once a slice has been taken.')
        clone = self._chain()
        if self._defer_next_filter:
            self._defer_next_filter = False
            clone._deferred_filter = negate, args, kwargs
        else:
            clone._filter_or_exclude_inplace(negate, args, kwargs)
        return clone

    def _filter_or_exclude_inplace(self, negate, args, kwargs):
        if negate:
            self._query.add_q(~Q(*args, **kwargs))
        else:
            self._query.add_q(Q(*args, **kwargs))
```
### 2 - django/contrib/admin/filters.py:

Start line: 449, End line: 479

```python
class EmptyFieldListFilter(FieldListFilter):

    def queryset(self, request, queryset):
        if self.lookup_kwarg not in self.used_parameters:
            return queryset
        if self.lookup_val not in ('0', '1'):
            raise IncorrectLookupParameters

        lookup_conditions = []
        if self.field.empty_strings_allowed:
            lookup_conditions.append((self.field_path, ''))
        if self.field.null:
            lookup_conditions.append((f'{self.field_path}__isnull', True))
        lookup_condition = models.Q(*lookup_conditions, _connector=models.Q.OR)
        if self.lookup_val == '1':
            return queryset.filter(lookup_condition)
        return queryset.exclude(lookup_condition)

    def expected_parameters(self):
        return [self.lookup_kwarg]

    def choices(self, changelist):
        for lookup, title in (
            (None, _('All')),
            ('1', _('Empty')),
            ('0', _('Not empty')),
        ):
            yield {
                'selected': self.lookup_val == lookup,
                'query_string': changelist.get_query_string({self.lookup_kwarg: lookup}),
                'display': title,
            }
```
### 3 - django/db/models/sql/query.py:

Start line: 1320, End line: 1384

```python
class Query(BaseExpression):

    def build_filter(self, filter_expr, branch_negated=False, current_negated=False,
                     can_reuse=None, allow_joins=True, split_subq=True,
                     check_filterable=True):
        # ... other code

        try:
            join_info = self.setup_joins(
                parts, opts, alias, can_reuse=can_reuse, allow_many=allow_many,
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
        clause = WhereNode([condition], connector=AND)

        require_outer = lookup_type == 'isnull' and condition.rhs is True and not current_negated
        if current_negated and (lookup_type != 'isnull' or condition.rhs is False) and condition.rhs is not None:
            require_outer = True
            if lookup_type != 'isnull':
                # The condition added here will be SQL like this:
                # NOT (col IS NOT NULL), where the first NOT is added in
                # upper layers of code. The reason for addition is that if col
                # is null, then col != someval will result in SQL "unknown"
                # which isn't the same as in Python. The Python None handling
                # is wanted, and it can be gotten by
                # (col IS NULL OR col != someval)
                #   <=>
                # NOT (col IS NOT NULL AND col = someval).
                if (
                    self.is_nullable(targets[0]) or
                    self.alias_map[join_list[-1]].join_type == LOUTER
                ):
                    lookup_class = targets[0].get_lookup('isnull')
                    col = self._get_col(targets[0], join_info.targets[0], alias)
                    clause.add(lookup_class(col, False), AND)
                # If someval is a nullable column, someval IS NOT NULL is
                # added.
                if isinstance(value, Col) and self.is_nullable(value.target):
                    lookup_class = value.target.get_lookup('isnull')
                    clause.add(lookup_class(value, False), AND)
        return clause, used_joins if not require_outer else ()
```
### 4 - django/db/models/expressions.py:

Start line: 1199, End line: 1231

```python
class Exists(Subquery):
    template = 'EXISTS(%(subquery)s)'
    output_field = fields.BooleanField()

    def __init__(self, queryset, negated=False, **kwargs):
        self.negated = negated
        super().__init__(queryset, **kwargs)

    def __invert__(self):
        clone = self.copy()
        clone.negated = not self.negated
        return clone

    def as_sql(self, compiler, connection, template=None, **extra_context):
        query = self.query.exists(using=connection.alias)
        sql, params = super().as_sql(
            compiler,
            connection,
            template=template,
            query=query,
            **extra_context,
        )
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
### 5 - django/db/models/sql/query.py:

Start line: 1873, End line: 1922

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

    def add_select_col(self, col, name):
        self.select += col,
        self.values_select += name,

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
### 6 - django/db/models/query.py:

Start line: 1564, End line: 1595

```python
class InstanceCheckMeta(type):
    def __instancecheck__(self, instance):
        return isinstance(instance, QuerySet) and instance.query.is_empty()


class EmptyQuerySet(metaclass=InstanceCheckMeta):
    """
    Marker class to checking if a queryset is empty by .none():
        isinstance(qs.none(), EmptyQuerySet) -> True
    """

    def __init__(self, *args, **kwargs):
        raise TypeError("EmptyQuerySet can't be instantiated")


class RawQuerySet:
    """
    Provide an iterator which converts the results of raw SQL queries into
    annotated model instances.
    """
    def __init__(self, raw_query, model=None, query=None, params=(),
                 translations=None, using=None, hints=None):
        self.raw_query = raw_query
        self.model = model
        self._db = using
        self._hints = hints or {}
        self.query = query or sql.RawQuery(sql=raw_query, using=self.db, params=params)
        self.params = params
        self.translations = translations or {}
        self._result_cache = None
        self._prefetch_related_lookups = ()
        self._prefetch_done = False
```
### 7 - django/db/models/sql/query.py:

Start line: 1775, End line: 1840

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
            WHERE NOT EXISTS(
                SELECT 1
                FROM child
                WHERE name = 'foo' AND child.parent_id = parent.id
                LIMIT 1
            )
        """
        # Generate the inner query.
        query = self.__class__(self.model)
        query._filtered_relations = self._filtered_relations
        filter_lhs, filter_rhs = filter_expr
        if isinstance(filter_rhs, OuterRef):
            filter_rhs = OuterRef(filter_rhs)
        elif isinstance(filter_rhs, F):
            filter_rhs = OuterRef(filter_rhs.name)
        query.add_filter(filter_lhs, filter_rhs)
        query.clear_ordering(force=True)
        # Try to have as simple as possible subquery -> trim leading joins from
        # the subquery.
        trimmed_prefix, contains_louter = query.trim_start(names_with_path)

        col = query.select[0]
        select_field = col.target
        alias = col.alias
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

        lookup_class = select_field.get_lookup('exact')
        lookup = lookup_class(col, ResolvedOuterRef(trimmed_prefix))
        query.where.add(lookup, AND)
        condition, needed_inner = self.build_filter(Exists(query))

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
### 8 - django/db/models/query.py:

Start line: 1150, End line: 1163

```python
class QuerySet:

    def intersection(self, *other_qs):
        # If any query is an EmptyQuerySet, return it.
        if isinstance(self, EmptyQuerySet):
            return self
        for other in other_qs:
            if isinstance(other, EmptyQuerySet):
                return other
        return self._combinator_query('intersection', *other_qs)

    def difference(self, *other_qs):
        # If the query is an EmptyQuerySet, return it.
        if isinstance(self, EmptyQuerySet):
            return self
        return self._combinator_query('difference', *other_qs)
```
### 9 - django/db/models/sql/query.py:

Start line: 1454, End line: 1481

```python
class Query(BaseExpression):

    def add_filtered_relation(self, filtered_relation, alias):
        filtered_relation.alias = alias
        lookups = dict(get_children_from_q(filtered_relation.condition))
        relation_lookup_parts, relation_field_parts, _ = self.solve_lookup_type(filtered_relation.relation_name)
        if relation_lookup_parts:
            raise ValueError(
                "FilteredRelation's relation_name cannot contain lookups "
                "(got %r)." % filtered_relation.relation_name
            )
        for lookup in chain(lookups):
            lookup_parts, lookup_field_parts, _ = self.solve_lookup_type(lookup)
            shift = 2 if not lookup_parts else 1
            lookup_field_path = lookup_field_parts[:-shift]
            for idx, lookup_field_part in enumerate(lookup_field_path):
                if len(relation_field_parts) > idx:
                    if relation_field_parts[idx] != lookup_field_part:
                        raise ValueError(
                            "FilteredRelation's condition doesn't support "
                            "relations outside the %r (got %r)."
                            % (filtered_relation.relation_name, lookup)
                        )
                else:
                    raise ValueError(
                        "FilteredRelation's condition doesn't support nested "
                        "relations deeper than the relation_name (got %r for "
                        "%r)." % (lookup, filtered_relation.relation_name)
                    )
        self._filtered_relations[filtered_relation.alias] = filtered_relation
```
### 10 - django/db/models/query.py:

Start line: 355, End line: 372

```python
class QuerySet:

    ####################################
    # METHODS THAT DO DATABASE QUERIES #
    ####################################

    def _iterator(self, use_chunked_fetch, chunk_size):
        iterable = self._iterable_class(
            self,
            chunked_fetch=use_chunked_fetch,
            chunk_size=chunk_size or 2000,
        )
        if not self._prefetch_related_lookups or chunk_size is None:
            yield from iterable
            return

        iterator = iter(iterable)
        while results := list(islice(iterator, chunk_size)):
            prefetch_related_objects(results, *self._prefetch_related_lookups)
            yield from results
```
### 58 - django/db/models/expressions.py:

Start line: 1141, End line: 1196

```python
class Subquery(BaseExpression, Combinable):
    """
    An explicit subquery. It may contain OuterRef() references to the outer
    query which will be resolved when it is applied to that query.
    """
    template = '(%(subquery)s)'
    contains_aggregate = False
    empty_result_set_value = None

    def __init__(self, queryset, output_field=None, **extra):
        # Allow the usage of both QuerySet and sql.Query objects.
        self.query = getattr(queryset, 'query', queryset).clone()
        self.query.subquery = True
        self.extra = extra
        super().__init__(output_field)

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

    def get_external_cols(self):
        return self.query.get_external_cols()

    def as_sql(self, compiler, connection, template=None, query=None, **extra_context):
        connection.ops.check_expression_support(self)
        template_params = {**self.extra, **extra_context}
        query = query or self.query
        subquery_sql, sql_params = query.as_sql(compiler, connection)
        template_params['subquery'] = subquery_sql[1:-1]

        template = template or template_params.get('template', self.template)
        sql = template % template_params
        return sql, sql_params

    def get_group_by_cols(self, alias=None):
        # If this expression is referenced by an alias for an explicit GROUP BY
        # through values() a reference to this expression and not the
        # underlying .query must be returned to ensure external column
        # references are not grouped against as well.
        if alias:
            return [Ref(alias, self)]
        return self.query.get_group_by_cols()
```
### 125 - django/db/models/expressions.py:

Start line: 612, End line: 651

```python
class ResolvedOuterRef(F):
    """
    An object that contains a reference to an outer query.

    In this case, the reference to the outer query has been resolved because
    the inner query has been used as a subquery.
    """
    contains_aggregate = False

    def as_sql(self, *args, **kwargs):
        raise ValueError(
            'This queryset contains a reference to an outer query and may '
            'only be used in a subquery.'
        )

    def resolve_expression(self, *args, **kwargs):
        col = super().resolve_expression(*args, **kwargs)
        # FIXME: Rename possibly_multivalued to multivalued and fix detection
        # for non-multivalued JOINs (e.g. foreign key fields). This should take
        # into account only many-to-many and one-to-many relationships.
        col.possibly_multivalued = LOOKUP_SEP in self.name
        return col

    def relabeled_clone(self, relabels):
        return self

    def get_group_by_cols(self, alias=None):
        return []


class OuterRef(F):
    contains_aggregate = False

    def resolve_expression(self, *args, **kwargs):
        if isinstance(self.name, self.__class__):
            return self.name
        return ResolvedOuterRef(self.name)

    def relabeled_clone(self, relabels):
        return self
```
### 134 - django/db/models/expressions.py:

Start line: 991, End line: 1056

```python
@deconstructible(path='django.db.models.When')
class When(Expression):
    template = 'WHEN %(condition)s THEN %(result)s'
    # This isn't a complete conditional expression, must be used in Case().
    conditional = False

    def __init__(self, condition=None, then=None, **lookups):
        if lookups:
            if condition is None:
                condition, lookups = Q(**lookups), None
            elif getattr(condition, 'conditional', False):
                condition, lookups = Q(condition, **lookups), None
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
