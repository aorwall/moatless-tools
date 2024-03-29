# django__django-13569

| **django/django** | `257f8495d6c93e30ab0f52af4c488d7344bcf112` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 13268 |
| **Any found context length** | 13268 |
| **Avg pos** | 37.0 |
| **Min pos** | 37 |
| **Max pos** | 37 |
| **Top file pos** | 11 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/db/models/functions/math.py b/django/db/models/functions/math.py
--- a/django/db/models/functions/math.py
+++ b/django/db/models/functions/math.py
@@ -154,6 +154,9 @@ def as_oracle(self, compiler, connection, **extra_context):
     def as_sqlite(self, compiler, connection, **extra_context):
         return super().as_sql(compiler, connection, function='RAND', **extra_context)
 
+    def get_group_by_cols(self, alias=None):
+        return []
+
 
 class Round(Transform):
     function = 'ROUND'

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/db/models/functions/math.py | 157 | 157 | 37 | 11 | 13268


## Problem Statement

```
order_by('?') unexpectedly breaking queryset aggregation
Description
	
Steps to reproduce:
class Thing(models.Model):
	pass
class Related(models.Model):
	models.ForeignKey(Thing)
With data
t = Thing.objects.create()
rs = [Related.objects.create(thing=t) for _ in range(2)]
The following query works as expected. The aggregation with Count produces a GROUP BY clause on related.id.
>>> Thing.objects.annotate(rc=Count('related')).order_by('rc').values('id', 'rc')
<QuerySet [{'id': 1, 'rc': 2}]>
This also works as expected (at least to me). Although there is an aggregation, ordering by related means that the grouping will be broken down.
>>> Thing.objects.annotate(rc=Count('related')).order_by('related').values('id', 'rc')
<QuerySet [{'id': 1, 'rc': 1}, {'id': 1, 'rc': 1}]>
But the following seems wrong to me.
>>> Thing.objects.annotate(rc=Count('related')).order_by('?').values('id', 'rc')
<QuerySet [{'id': 1, 'rc': 1}, {'id': 1, 'rc': 1}]>
The random function call has nothing to do with the aggregation, and I see no reason it should break it. Dumping the query seems that indeed the random call breaks the group by call: (I simpilfied the table names a little)
>>> print(Thing.objects.annotate(rc=Count('related')).order_by('?').values('id', 'rc').query)
SELECT "thing"."id", COUNT("related"."id") AS "rc" FROM "thing" LEFT OUTER JOIN "related" ON ("thing"."id" = "related"."thing_id") GROUP BY "thing"."id", RANDOM() ORDER BY RANDOM() ASC
I dug into the SQL compiler, and it seems to me the problem is inside django.db.models.sql.compiler.get_group_by, where the compiler combines all non-aggregate, non-ref order_by expressions into group_by. I patched it like this
for expr, (sql, params, is_ref) in order_by:
	if expr.contains_aggregate:
		continue
	if is_ref:
		continue
	expressions.extend([
		exp for exp in expr.get_source_expressions()
		if not isinstance(exp, Random)
	])
and things seem to work correctly. No failed tests against SQLite3 with default settings.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 django/db/models/aggregates.py | 45 | 68| 294 | 294 | 1301 | 
| 2 | 1 django/db/models/aggregates.py | 70 | 96| 266 | 560 | 1301 | 
| 3 | 2 django/db/models/sql/compiler.py | 360 | 396| 430 | 990 | 15571 | 
| 4 | 2 django/db/models/sql/compiler.py | 63 | 147| 881 | 1871 | 15571 | 
| 5 | 3 django/contrib/postgres/aggregates/mixins.py | 22 | 34| 124 | 1995 | 16009 | 
| 6 | 3 django/contrib/postgres/aggregates/mixins.py | 1 | 20| 182 | 2177 | 16009 | 
| 7 | 3 django/db/models/sql/compiler.py | 271 | 358| 712 | 2889 | 16009 | 
| 8 | 3 django/db/models/sql/compiler.py | 149 | 197| 523 | 3412 | 16009 | 
| 9 | 4 django/db/models/query.py | 1096 | 1137| 323 | 3735 | 33221 | 
| 10 | 4 django/db/models/sql/compiler.py | 756 | 767| 163 | 3898 | 33221 | 
| 11 | 5 django/db/models/sql/query.py | 1941 | 1985| 364 | 4262 | 55711 | 
| 12 | 6 django/db/models/expressions.py | 1192 | 1217| 248 | 4510 | 66567 | 
| 13 | 6 django/db/models/sql/query.py | 1987 | 2036| 420 | 4930 | 66567 | 
| 14 | 6 django/db/models/expressions.py | 1219 | 1248| 211 | 5141 | 66567 | 
| 15 | 7 django/db/models/sql/where.py | 233 | 249| 130 | 5271 | 68381 | 
| 16 | 7 django/db/models/sql/query.py | 899 | 921| 248 | 5519 | 68381 | 
| 17 | 7 django/db/models/sql/query.py | 1702 | 1741| 439 | 5958 | 68381 | 
| 18 | 7 django/db/models/sql/compiler.py | 885 | 977| 839 | 6797 | 68381 | 
| 19 | 7 django/db/models/sql/query.py | 364 | 414| 494 | 7291 | 68381 | 
| 20 | 8 django/db/models/__init__.py | 1 | 53| 619 | 7910 | 69000 | 
| 21 | 9 django/contrib/admin/views/main.py | 340 | 400| 508 | 8418 | 73396 | 
| 22 | 9 django/db/models/sql/compiler.py | 1 | 19| 170 | 8588 | 73396 | 
| 23 | 9 django/db/models/sql/compiler.py | 433 | 486| 564 | 9152 | 73396 | 
| 24 | 9 django/db/models/sql/compiler.py | 1199 | 1220| 223 | 9375 | 73396 | 
| 25 | 9 django/db/models/sql/compiler.py | 22 | 47| 257 | 9632 | 73396 | 
| 26 | 10 django/contrib/gis/db/models/aggregates.py | 29 | 46| 216 | 9848 | 74013 | 
| 27 | 10 django/db/models/query.py | 799 | 838| 322 | 10170 | 74013 | 
| 28 | 10 django/db/models/query.py | 1139 | 1175| 324 | 10494 | 74013 | 
| 29 | 10 django/contrib/postgres/aggregates/mixins.py | 36 | 49| 145 | 10639 | 74013 | 
| 30 | 10 django/db/models/sql/query.py | 628 | 652| 269 | 10908 | 74013 | 
| 31 | 10 django/db/models/sql/compiler.py | 1038 | 1078| 337 | 11245 | 74013 | 
| 32 | 10 django/db/models/sql/compiler.py | 398 | 406| 133 | 11378 | 74013 | 
| 33 | 10 django/db/models/sql/query.py | 1473 | 1558| 801 | 12179 | 74013 | 
| 34 | 10 django/db/models/query.py | 320 | 346| 222 | 12401 | 74013 | 
| 35 | 10 django/db/models/sql/query.py | 703 | 738| 389 | 12790 | 74013 | 
| 36 | 10 django/db/models/aggregates.py | 122 | 158| 245 | 13035 | 74013 | 
| **-> 37 <-** | **11 django/db/models/functions/math.py** | 144 | 181| 233 | 13268 | 75263 | 
| 38 | 12 django/contrib/postgres/aggregates/general.py | 1 | 68| 413 | 13681 | 75677 | 
| 39 | 13 django/contrib/postgres/aggregates/statistics.py | 1 | 66| 419 | 14100 | 76096 | 
| 40 | 13 django/db/models/query.py | 1483 | 1516| 297 | 14397 | 76096 | 
| 41 | 13 django/db/models/expressions.py | 1168 | 1190| 188 | 14585 | 76096 | 
| 42 | 13 django/db/models/sql/query.py | 1115 | 1147| 338 | 14923 | 76096 | 
| 43 | 13 django/db/models/sql/query.py | 1294 | 1359| 772 | 15695 | 76096 | 
| 44 | 13 django/db/models/query.py | 365 | 399| 314 | 16009 | 76096 | 
| 45 | 14 django/db/models/base.py | 1711 | 1811| 729 | 16738 | 92740 | 
| 46 | 14 django/db/models/sql/query.py | 1428 | 1455| 283 | 17021 | 92740 | 
| 47 | 14 django/db/models/query.py | 1972 | 1995| 200 | 17221 | 92740 | 
| 48 | 15 django/db/models/sql/subqueries.py | 137 | 163| 173 | 17394 | 93953 | 
| 49 | 16 django/db/backends/mysql/compiler.py | 41 | 63| 176 | 17570 | 94473 | 
| 50 | 16 django/db/models/sql/subqueries.py | 47 | 75| 210 | 17780 | 94473 | 
| 51 | 16 django/db/models/query.py | 983 | 1003| 239 | 18019 | 94473 | 
| 52 | 16 django/db/models/query.py | 1655 | 1761| 1063 | 19082 | 94473 | 
| 53 | 16 django/db/models/sql/query.py | 118 | 133| 145 | 19227 | 94473 | 
| 54 | 16 django/db/models/sql/compiler.py | 199 | 269| 580 | 19807 | 94473 | 
| 55 | 16 django/db/models/sql/query.py | 874 | 897| 203 | 20010 | 94473 | 
| 56 | 16 django/db/models/sql/query.py | 2180 | 2225| 371 | 20381 | 94473 | 
| 57 | 17 django/contrib/admin/options.py | 207 | 218| 135 | 20516 | 113055 | 
| 58 | 18 django/db/models/fields/related_lookups.py | 121 | 157| 244 | 20760 | 114508 | 
| 59 | 18 django/db/models/sql/query.py | 416 | 509| 917 | 21677 | 114508 | 
| 60 | 18 django/db/models/sql/query.py | 1636 | 1660| 227 | 21904 | 114508 | 
| 61 | 18 django/db/models/aggregates.py | 99 | 119| 158 | 22062 | 114508 | 
| 62 | 18 django/db/models/sql/compiler.py | 979 | 1001| 207 | 22269 | 114508 | 
| 63 | 19 django/contrib/postgres/search.py | 130 | 157| 248 | 22517 | 116730 | 
| 64 | 19 django/db/models/sql/query.py | 136 | 230| 833 | 23350 | 116730 | 
| 65 | 19 django/db/models/query.py | 1215 | 1243| 183 | 23533 | 116730 | 
| 66 | 19 django/db/models/sql/compiler.py | 1016 | 1037| 199 | 23732 | 116730 | 
| 67 | 19 django/db/models/sql/query.py | 1743 | 1814| 784 | 24516 | 116730 | 
| 68 | 19 django/db/models/sql/query.py | 1816 | 1845| 259 | 24775 | 116730 | 
| 69 | 19 django/db/models/sql/compiler.py | 1583 | 1614| 244 | 25019 | 116730 | 
| 70 | 19 django/db/models/sql/query.py | 511 | 551| 325 | 25344 | 116730 | 
| 71 | 20 django/views/generic/list.py | 50 | 75| 244 | 25588 | 118302 | 
| 72 | 20 django/db/models/sql/compiler.py | 1540 | 1580| 409 | 25997 | 118302 | 
| 73 | 20 django/db/models/sql/query.py | 1 | 65| 465 | 26462 | 118302 | 
| 74 | 21 django/db/models/deletion.py | 1 | 76| 566 | 27028 | 122130 | 
| 75 | 21 django/db/models/sql/query.py | 68 | 116| 361 | 27389 | 122130 | 
| 76 | 21 django/db/models/sql/compiler.py | 1132 | 1197| 527 | 27916 | 122130 | 
| 77 | 22 django/db/backends/postgresql/features.py | 1 | 104| 844 | 28760 | 122974 | 
| 78 | 22 django/db/models/fields/related_lookups.py | 62 | 101| 451 | 29211 | 122974 | 
| 79 | 22 django/db/models/sql/compiler.py | 488 | 645| 1469 | 30680 | 122974 | 
| 80 | 22 django/db/models/base.py | 2038 | 2089| 351 | 31031 | 122974 | 
| 81 | 22 django/db/models/query.py | 664 | 678| 132 | 31163 | 122974 | 
| 82 | 22 django/db/models/sql/compiler.py | 49 | 61| 155 | 31318 | 122974 | 
| 83 | 22 django/contrib/gis/db/models/aggregates.py | 1 | 27| 199 | 31517 | 122974 | 
| 84 | 22 django/db/models/query.py | 1318 | 1336| 186 | 31703 | 122974 | 
| 85 | 22 django/db/models/aggregates.py | 1 | 43| 344 | 32047 | 122974 | 
| 86 | 22 django/db/models/query.py | 1349 | 1397| 405 | 32452 | 122974 | 
| 87 | 22 django/db/models/sql/query.py | 1847 | 1896| 330 | 32782 | 122974 | 
| 88 | 22 django/db/models/sql/query.py | 1361 | 1382| 250 | 33032 | 122974 | 
| 89 | 22 django/db/models/sql/query.py | 2227 | 2259| 228 | 33260 | 122974 | 
| 90 | 22 django/db/models/sql/compiler.py | 691 | 713| 202 | 33462 | 122974 | 
| 91 | 22 django/db/models/sql/query.py | 553 | 627| 809 | 34271 | 122974 | 
| 92 | 22 django/db/models/query.py | 401 | 459| 474 | 34745 | 122974 | 
| 93 | 22 django/db/models/expressions.py | 1251 | 1294| 378 | 35123 | 122974 | 
| 94 | 22 django/db/models/sql/compiler.py | 1111 | 1130| 206 | 35329 | 122974 | 
| 95 | 22 django/db/models/query.py | 236 | 263| 221 | 35550 | 122974 | 
| 96 | 23 django/contrib/gis/db/models/__init__.py | 1 | 19| 204 | 35754 | 123178 | 
| 97 | 24 django/db/models/lookups.py | 102 | 144| 359 | 36113 | 128127 | 
| 98 | 25 django/db/backends/oracle/features.py | 1 | 94| 736 | 36849 | 128864 | 
| 99 | 25 django/db/models/query.py | 265 | 285| 180 | 37029 | 128864 | 
| 100 | 26 django/db/backends/sqlite3/operations.py | 40 | 66| 232 | 37261 | 131954 | 
| 101 | 26 django/db/models/expressions.py | 929 | 993| 591 | 37852 | 131954 | 
| 102 | 26 django/contrib/postgres/search.py | 198 | 230| 243 | 38095 | 131954 | 
| 103 | 26 django/db/models/sql/subqueries.py | 1 | 44| 320 | 38415 | 131954 | 
| 104 | 26 django/db/models/expressions.py | 1044 | 1074| 281 | 38696 | 131954 | 
| 105 | 27 django/db/models/fields/related.py | 190 | 254| 673 | 39369 | 145830 | 
| 106 | 27 django/contrib/gis/db/models/aggregates.py | 49 | 84| 207 | 39576 | 145830 | 
| 107 | 27 django/db/models/query.py | 1 | 39| 294 | 39870 | 145830 | 
| 108 | 27 django/db/models/expressions.py | 1330 | 1352| 215 | 40085 | 145830 | 
| 109 | 28 django/db/models/query_utils.py | 25 | 54| 185 | 40270 | 148536 | 
| 110 | 28 django/contrib/postgres/search.py | 160 | 195| 313 | 40583 | 148536 | 
| 111 | 28 django/db/models/sql/query.py | 2038 | 2084| 370 | 40953 | 148536 | 
| 112 | 28 django/db/models/expressions.py | 779 | 793| 120 | 41073 | 148536 | 
| 113 | 28 django/db/models/sql/query.py | 1059 | 1084| 214 | 41287 | 148536 | 
| 114 | 28 django/db/models/fields/related.py | 127 | 154| 201 | 41488 | 148536 | 
| 115 | 29 django/db/models/functions/comparison.py | 78 | 92| 175 | 41663 | 149940 | 
| 116 | 29 django/db/models/fields/related.py | 1235 | 1352| 963 | 42626 | 149940 | 
| 117 | 29 django/db/models/query.py | 1005 | 1018| 126 | 42752 | 149940 | 
| 118 | 29 django/db/models/sql/query.py | 991 | 1022| 307 | 43059 | 149940 | 
| 119 | 29 django/db/models/expressions.py | 795 | 811| 153 | 43212 | 149940 | 
| 120 | 29 django/db/models/query.py | 348 | 363| 146 | 43358 | 149940 | 
| 121 | 29 django/db/models/sql/compiler.py | 803 | 883| 717 | 44075 | 149940 | 
| 122 | 30 django/db/backends/postgresql/operations.py | 160 | 187| 311 | 44386 | 152501 | 
| 123 | 30 django/db/models/fields/related.py | 255 | 282| 269 | 44655 | 152501 | 
| 124 | 30 django/db/models/sql/query.py | 807 | 833| 280 | 44935 | 152501 | 
| 125 | 30 django/db/models/lookups.py | 305 | 355| 306 | 45241 | 152501 | 
| 126 | 30 django/db/models/query.py | 1911 | 1970| 772 | 46013 | 152501 | 
| 127 | 30 django/db/models/expressions.py | 1077 | 1136| 440 | 46453 | 152501 | 
| 128 | 30 django/db/models/query.py | 1058 | 1079| 214 | 46667 | 152501 | 
| 129 | 31 django/db/migrations/operations/special.py | 63 | 114| 390 | 47057 | 154059 | 
| 130 | 32 django/forms/models.py | 629 | 646| 167 | 47224 | 165833 | 
| 131 | 32 django/db/backends/sqlite3/operations.py | 1 | 38| 258 | 47482 | 165833 | 
| 132 | 32 django/db/models/expressions.py | 332 | 387| 368 | 47850 | 165833 | 
| 133 | 32 django/db/models/sql/query.py | 2415 | 2470| 827 | 48677 | 165833 | 
| 134 | 33 django/contrib/gis/db/backends/mysql/operations.py | 51 | 71| 225 | 48902 | 166684 | 
| 135 | 33 django/db/models/query.py | 719 | 746| 235 | 49137 | 166684 | 
| 136 | 33 django/db/models/sql/compiler.py | 1519 | 1538| 164 | 49301 | 166684 | 
| 137 | 33 django/db/models/deletion.py | 346 | 359| 116 | 49417 | 166684 | 
| 138 | 34 django/contrib/admin/checks.py | 563 | 598| 304 | 49721 | 175821 | 
| 139 | 34 django/db/backends/mysql/compiler.py | 1 | 14| 123 | 49844 | 175821 | 
| 140 | 34 django/db/models/base.py | 961 | 975| 212 | 50056 | 175821 | 
| 141 | 35 django/db/backends/oracle/operations.py | 369 | 406| 369 | 50425 | 181792 | 
| 142 | 36 django/core/management/commands/dumpdata.py | 142 | 178| 316 | 50741 | 183404 | 


### Hint

```
Patch to SQLCompiler.get_group_by that excluds Random expressions
​PR
I wonder what would happen if we skipped all expressions that have no cols as source expressions (plus, we need to include any raw sql).
I wonder what would happen if we skipped all expressions that have no cols as source expressions (plus, we need to include any raw sql). This seems like a very good idea, and I can’t think of a scenario where this will break things. I’ve updated the PR.
The new test isn't passing on MySQL/PostgreSQL.
Some test additions are still needed.
It's documented that ordering will be included in the grouping clause so I wouldn't say that this behavior is unexpected. It seems to me that trying to remove some (but not all) columns from the group by clause according to new rules is less clear than the simple rule that is in place now.
If you need to filter on an annotated value while still using .order_by('?'), this could work: Thing.objects.filter(pk__in=Thing.objects.annotate(rc=Count('related')).filter(rc__gte=2)).order_by('?') This avoids the GROUP BY RANDOM() ORDER BY RANDOM() ASC issue while still allowing .annotate() and .order_by('?') to be used together.
```

## Patch

```diff
diff --git a/django/db/models/functions/math.py b/django/db/models/functions/math.py
--- a/django/db/models/functions/math.py
+++ b/django/db/models/functions/math.py
@@ -154,6 +154,9 @@ def as_oracle(self, compiler, connection, **extra_context):
     def as_sqlite(self, compiler, connection, **extra_context):
         return super().as_sql(compiler, connection, function='RAND', **extra_context)
 
+    def get_group_by_cols(self, alias=None):
+        return []
+
 
 class Round(Transform):
     function = 'ROUND'

```

## Test Patch

```diff
diff --git a/tests/aggregation/tests.py b/tests/aggregation/tests.py
--- a/tests/aggregation/tests.py
+++ b/tests/aggregation/tests.py
@@ -1315,3 +1315,18 @@ def test_aggregation_subquery_annotation_related_field(self):
         # with self.assertNumQueries(1) as ctx:
         #     self.assertSequenceEqual(books_qs, [book])
         # self.assertEqual(ctx[0]['sql'].count('SELECT'), 2)
+
+    def test_aggregation_random_ordering(self):
+        """Random() is not included in the GROUP BY when used for ordering."""
+        authors = Author.objects.annotate(contact_count=Count('book')).order_by('?')
+        self.assertQuerysetEqual(authors, [
+            ('Adrian Holovaty', 1),
+            ('Jacob Kaplan-Moss', 1),
+            ('Brad Dayley', 1),
+            ('James Bennett', 1),
+            ('Jeffrey Forcier', 1),
+            ('Paul Bissex', 1),
+            ('Wesley J. Chun', 1),
+            ('Stuart Russell', 1),
+            ('Peter Norvig', 2),
+        ], lambda a: (a.name, a.contact_count), ordered=False)

```


## Code snippets

### 1 - django/db/models/aggregates.py:

Start line: 45, End line: 68

```python
class Aggregate(Func):

    def resolve_expression(self, query=None, allow_joins=True, reuse=None, summarize=False, for_save=False):
        # Aggregates are not allowed in UPDATE queries, so ignore for_save
        c = super().resolve_expression(query, allow_joins, reuse, summarize)
        c.filter = c.filter and c.filter.resolve_expression(query, allow_joins, reuse, summarize)
        if not summarize:
            # Call Aggregate.get_source_expressions() to avoid
            # returning self.filter and including that in this loop.
            expressions = super(Aggregate, c).get_source_expressions()
            for index, expr in enumerate(expressions):
                if expr.contains_aggregate:
                    before_resolved = self.get_source_expressions()[index]
                    name = before_resolved.name if hasattr(before_resolved, 'name') else repr(before_resolved)
                    raise FieldError("Cannot compute %s('%s'): '%s' is an aggregate" % (c.name, name, name))
        return c

    @property
    def default_alias(self):
        expressions = self.get_source_expressions()
        if len(expressions) == 1 and hasattr(expressions[0], 'name'):
            return '%s__%s' % (expressions[0].name, self.name.lower())
        raise TypeError("Complex expressions require an alias")

    def get_group_by_cols(self, alias=None):
        return []
```
### 2 - django/db/models/aggregates.py:

Start line: 70, End line: 96

```python
class Aggregate(Func):

    def as_sql(self, compiler, connection, **extra_context):
        extra_context['distinct'] = 'DISTINCT ' if self.distinct else ''
        if self.filter:
            if connection.features.supports_aggregate_filter_clause:
                filter_sql, filter_params = self.filter.as_sql(compiler, connection)
                template = self.filter_template % extra_context.get('template', self.template)
                sql, params = super().as_sql(
                    compiler, connection, template=template, filter=filter_sql,
                    **extra_context
                )
                return sql, params + filter_params
            else:
                copy = self.copy()
                copy.filter = None
                source_expressions = copy.get_source_expressions()
                condition = When(self.filter, then=source_expressions[0])
                copy.set_source_expressions([Case(condition)] + source_expressions[1:])
                return super(Aggregate, copy).as_sql(compiler, connection, **extra_context)
        return super().as_sql(compiler, connection, **extra_context)

    def _get_repr_options(self):
        options = super()._get_repr_options()
        if self.distinct:
            options['distinct'] = self.distinct
        if self.filter:
            options['filter'] = self.filter
        return options
```
### 3 - django/db/models/sql/compiler.py:

Start line: 360, End line: 396

```python
class SQLCompiler:
    ordering_parts =

    def get_order_by(self):
        # ... other code

        for expr, is_ref in order_by:
            resolved = expr.resolve_expression(self.query, allow_joins=True, reuse=None)
            if self.query.combinator and self.select:
                src = resolved.get_source_expressions()[0]
                expr_src = expr.get_source_expressions()[0]
                # Relabel order by columns to raw numbers if this is a combined
                # query; necessary since the columns can't be referenced by the
                # fully qualified name and the simple column names may collide.
                for idx, (sel_expr, _, col_alias) in enumerate(self.select):
                    if is_ref and col_alias == src.refs:
                        src = src.source
                    elif col_alias and not (
                        isinstance(expr_src, F) and col_alias == expr_src.name
                    ):
                        continue
                    if src == sel_expr:
                        resolved.set_source_expressions([RawSQL('%d' % (idx + 1), ())])
                        break
                else:
                    if col_alias:
                        raise DatabaseError('ORDER BY term does not match any column in the result set.')
                    # Add column used in ORDER BY clause without an alias to
                    # the selected columns.
                    self.query.add_select_col(src)
                    resolved.set_source_expressions([RawSQL('%d' % len(self.query.select), ())])
            sql, params = self.compile(resolved)
            # Don't add the same column twice, but the order direction is
            # not taken into account so we strip it. When this entire method
            # is refactored into expressions, then we can check each part as we
            # generate it.
            without_ordering = self.ordering_parts.search(sql)[1]
            params_hash = make_hashable(params)
            if (without_ordering, params_hash) in seen:
                continue
            seen.add((without_ordering, params_hash))
            result.append((resolved, (sql, params, is_ref)))
        return result
    # ... other code
```
### 4 - django/db/models/sql/compiler.py:

Start line: 63, End line: 147

```python
class SQLCompiler:
    ordering_parts =
    # ... other code

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
            sql, params = expr.select_format(self, sql, params)
            params_hash = make_hashable(params)
            if (sql, params_hash) not in seen:
                result.append((sql, params))
                seen.add((sql, params_hash))
        return result
    # ... other code
```
### 5 - django/contrib/postgres/aggregates/mixins.py:

Start line: 22, End line: 34

```python
class OrderableAggMixin:

    def as_sql(self, compiler, connection):
        if self.ordering:
            ordering_params = []
            ordering_expr_sql = []
            for expr in self.ordering:
                expr_sql, expr_params = compiler.compile(expr)
                ordering_expr_sql.append(expr_sql)
                ordering_params.extend(expr_params)
            sql, sql_params = super().as_sql(compiler, connection, ordering=(
                'ORDER BY ' + ', '.join(ordering_expr_sql)
            ))
            return sql, sql_params + ordering_params
        return super().as_sql(compiler, connection, ordering='')
```
### 6 - django/contrib/postgres/aggregates/mixins.py:

Start line: 1, End line: 20

```python
from django.db.models import F, OrderBy


class OrderableAggMixin:

    def __init__(self, *expressions, ordering=(), **extra):
        if not isinstance(ordering, (list, tuple)):
            ordering = [ordering]
        ordering = ordering or []
        # Transform minus sign prefixed strings into an OrderBy() expression.
        ordering = (
            (OrderBy(F(o[1:]), descending=True) if isinstance(o, str) and o[0] == '-' else o)
            for o in ordering
        )
        super().__init__(*expressions, **extra)
        self.ordering = self._parse_expressions(*ordering)

    def resolve_expression(self, *args, **kwargs):
        self.ordering = [expr.resolve_expression(*args, **kwargs) for expr in self.ordering]
        return super().resolve_expression(*args, **kwargs)
```
### 7 - django/db/models/sql/compiler.py:

Start line: 271, End line: 358

```python
class SQLCompiler:
    ordering_parts =
    # ... other code

    def get_order_by(self):
        """
        Return a list of 2-tuples of form (expr, (sql, params, is_ref)) for the
        ORDER BY clause.

        The order_by clause can alter the select clause (for example it
        can add aliases to clauses that do not yet have one, or it can
        add totally new select clauses).
        """
        if self.query.extra_order_by:
            ordering = self.query.extra_order_by
        elif not self.query.default_ordering:
            ordering = self.query.order_by
        elif self.query.order_by:
            ordering = self.query.order_by
        elif self.query.get_meta().ordering:
            ordering = self.query.get_meta().ordering
            self._meta_ordering = ordering
        else:
            ordering = []
        if self.query.standard_ordering:
            asc, desc = ORDER_DIR['ASC']
        else:
            asc, desc = ORDER_DIR['DESC']

        order_by = []
        for field in ordering:
            if hasattr(field, 'resolve_expression'):
                if isinstance(field, Value):
                    # output_field must be resolved for constants.
                    field = Cast(field, field.output_field)
                if not isinstance(field, OrderBy):
                    field = field.asc()
                if not self.query.standard_ordering:
                    field = field.copy()
                    field.reverse_ordering()
                order_by.append((field, False))
                continue
            if field == '?':  # random
                order_by.append((OrderBy(Random()), False))
                continue

            col, order = get_order_dir(field, asc)
            descending = order == 'DESC'

            if col in self.query.annotation_select:
                # Reference to expression in SELECT clause
                order_by.append((
                    OrderBy(Ref(col, self.query.annotation_select[col]), descending=descending),
                    True))
                continue
            if col in self.query.annotations:
                # References to an expression which is masked out of the SELECT
                # clause.
                expr = self.query.annotations[col]
                if isinstance(expr, Value):
                    # output_field must be resolved for constants.
                    expr = Cast(expr, expr.output_field)
                order_by.append((OrderBy(expr, descending=descending), False))
                continue

            if '.' in field:
                # This came in through an extra(order_by=...) addition. Pass it
                # on verbatim.
                table, col = col.split('.', 1)
                order_by.append((
                    OrderBy(
                        RawSQL('%s.%s' % (self.quote_name_unless_alias(table), col), []),
                        descending=descending
                    ), False))
                continue

            if not self.query.extra or col not in self.query.extra:
                # 'col' is of the form 'field' or 'field1__field2' or
                # '-field1__field2__field', etc.
                order_by.extend(self.find_ordering_name(
                    field, self.query.get_meta(), default_order=asc))
            else:
                if col not in self.query.extra_select:
                    order_by.append((
                        OrderBy(RawSQL(*self.query.extra[col]), descending=descending),
                        False))
                else:
                    order_by.append((
                        OrderBy(Ref(col, RawSQL(*self.query.extra[col])), descending=descending),
                        True))
        result = []
        seen = set()
        # ... other code
    # ... other code
```
### 8 - django/db/models/sql/compiler.py:

Start line: 149, End line: 197

```python
class SQLCompiler:
    ordering_parts =
    # ... other code

    def collapse_group_by(self, expressions, having):
        # If the DB can group by primary key, then group by the primary key of
        # query's main model. Note that for PostgreSQL the GROUP BY clause must
        # include the primary key of every table, but for MySQL it is enough to
        # have the main table's primary key.
        if self.connection.features.allows_group_by_pk:
            # Determine if the main model's primary key is in the query.
            pk = None
            for expr in expressions:
                # Is this a reference to query's base table primary key? If the
                # expression isn't a Col-like, then skip the expression.
                if (getattr(expr, 'target', None) == self.query.model._meta.pk and
                        getattr(expr, 'alias', None) == self.query.base_table):
                    pk = expr
                    break
            # If the main model's primary key is in the query, group by that
            # field, HAVING expressions, and expressions associated with tables
            # that don't have a primary key included in the grouped columns.
            if pk:
                pk_aliases = {
                    expr.alias for expr in expressions
                    if hasattr(expr, 'target') and expr.target.primary_key
                }
                expressions = [pk] + [
                    expr for expr in expressions
                    if expr in having or (
                        getattr(expr, 'alias', None) is not None and expr.alias not in pk_aliases
                    )
                ]
        elif self.connection.features.allows_group_by_selected_pks:
            # Filter out all expressions associated with a table's primary key
            # present in the grouped columns. This is done by identifying all
            # tables that have their primary key included in the grouped
            # columns and removing non-primary key columns referring to them.
            # Unmanaged models are excluded because they could be representing
            # database views on which the optimization might not be allowed.
            pks = {
                expr for expr in expressions
                if (
                    hasattr(expr, 'target') and
                    expr.target.primary_key and
                    self.connection.features.allows_group_by_selected_pks_on_model(expr.target.model)
                )
            }
            aliases = {expr.alias for expr in pks}
            expressions = [
                expr for expr in expressions if expr in pks or getattr(expr, 'alias', None) not in aliases
            ]
        return expressions
    # ... other code
```
### 9 - django/db/models/query.py:

Start line: 1096, End line: 1137

```python
class QuerySet:

    def _annotate(self, args, kwargs, select=True):
        self._validate_values_are_expressions(args + tuple(kwargs.values()), method_name='annotate')
        annotations = {}
        for arg in args:
            # The default_alias property may raise a TypeError.
            try:
                if arg.default_alias in kwargs:
                    raise ValueError("The named annotation '%s' conflicts with the "
                                     "default name for another annotation."
                                     % arg.default_alias)
            except TypeError:
                raise TypeError("Complex annotations require an alias")
            annotations[arg.default_alias] = arg
        annotations.update(kwargs)

        clone = self._chain()
        names = self._fields
        if names is None:
            names = set(chain.from_iterable(
                (field.name, field.attname) if hasattr(field, 'attname') else (field.name,)
                for field in self.model._meta.get_fields()
            ))

        for alias, annotation in annotations.items():
            if alias in names:
                raise ValueError("The annotation '%s' conflicts with a field on "
                                 "the model." % alias)
            if isinstance(annotation, FilteredRelation):
                clone.query.add_filtered_relation(annotation, alias)
            else:
                clone.query.add_annotation(
                    annotation, alias, is_summary=False, select=select,
                )
        for alias, annotation in clone.query.annotations.items():
            if alias in annotations and annotation.contains_aggregate:
                if clone._fields is None:
                    clone.query.group_by = True
                else:
                    clone.query.set_group_by()
                break

        return clone
```
### 10 - django/db/models/sql/compiler.py:

Start line: 756, End line: 767

```python
class SQLCompiler:
    ordering_parts =
    # ... other code

    def _setup_joins(self, pieces, opts, alias):
        """
        Helper method for get_order_by() and get_distinct().

        get_ordering() and get_distinct() must produce same target columns on
        same input, as the prefixes of get_ordering() and get_distinct() must
        match. Executing SQL where this is not true is an error.
        """
        alias = alias or self.query.get_initial_alias()
        field, targets, opts, joins, path, transform_function = self.query.setup_joins(pieces, opts, alias)
        alias = joins[-1]
        return field, targets, alias, joins, path, opts, transform_function
    # ... other code
```
### 37 - django/db/models/functions/math.py:

Start line: 144, End line: 181

```python
class Random(NumericOutputFieldMixin, Func):
    function = 'RANDOM'
    arity = 0

    def as_mysql(self, compiler, connection, **extra_context):
        return super().as_sql(compiler, connection, function='RAND', **extra_context)

    def as_oracle(self, compiler, connection, **extra_context):
        return super().as_sql(compiler, connection, function='DBMS_RANDOM.VALUE', **extra_context)

    def as_sqlite(self, compiler, connection, **extra_context):
        return super().as_sql(compiler, connection, function='RAND', **extra_context)


class Round(Transform):
    function = 'ROUND'
    lookup_name = 'round'


class Sign(Transform):
    function = 'SIGN'
    lookup_name = 'sign'


class Sin(NumericOutputFieldMixin, Transform):
    function = 'SIN'
    lookup_name = 'sin'


class Sqrt(NumericOutputFieldMixin, Transform):
    function = 'SQRT'
    lookup_name = 'sqrt'


class Tan(NumericOutputFieldMixin, Transform):
    function = 'TAN'
    lookup_name = 'tan'
```
