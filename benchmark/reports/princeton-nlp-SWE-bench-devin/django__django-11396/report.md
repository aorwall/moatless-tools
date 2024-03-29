# django__django-11396

| **django/django** | `59f04d6b8f6c7c7a1039185bd2c5653ea91f7ff7` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 12282 |
| **Any found context length** | 6518 |
| **Avg pos** | 84.0 |
| **Min pos** | 21 |
| **Max pos** | 42 |
| **Top file pos** | 2 |
| **Missing snippets** | 3 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/db/models/sql/compiler.py b/django/db/models/sql/compiler.py
--- a/django/db/models/sql/compiler.py
+++ b/django/db/models/sql/compiler.py
@@ -5,7 +5,8 @@
 
 from django.core.exceptions import EmptyResultSet, FieldError
 from django.db.models.constants import LOOKUP_SEP
-from django.db.models.expressions import OrderBy, Random, RawSQL, Ref
+from django.db.models.expressions import OrderBy, Random, RawSQL, Ref, Value
+from django.db.models.functions import Cast
 from django.db.models.query_utils import QueryWrapper, select_related_descend
 from django.db.models.sql.constants import (
     CURSOR, GET_ITERATOR_CHUNK_SIZE, MULTI, NO_RESULTS, ORDER_DIR, SINGLE,
@@ -278,6 +279,9 @@ def get_order_by(self):
         order_by = []
         for field in ordering:
             if hasattr(field, 'resolve_expression'):
+                if isinstance(field, Value):
+                    # output_field must be resolved for constants.
+                    field = Cast(field, field.output_field)
                 if not isinstance(field, OrderBy):
                     field = field.asc()
                 if not self.query.standard_ordering:
@@ -299,10 +303,13 @@ def get_order_by(self):
                     True))
                 continue
             if col in self.query.annotations:
-                # References to an expression which is masked out of the SELECT clause
-                order_by.append((
-                    OrderBy(self.query.annotations[col], descending=descending),
-                    False))
+                # References to an expression which is masked out of the SELECT
+                # clause.
+                expr = self.query.annotations[col]
+                if isinstance(expr, Value):
+                    # output_field must be resolved for constants.
+                    expr = Cast(expr, expr.output_field)
+                order_by.append((OrderBy(expr, descending=descending), False))
                 continue
 
             if '.' in field:

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/db/models/sql/compiler.py | 8 | 8 | 42 | 2 | 12282
| django/db/models/sql/compiler.py | 281 | 281 | 21 | 2 | 6518
| django/db/models/sql/compiler.py | 302 | 305 | 21 | 2 | 6518


## Problem Statement

```
Cannot order query by constant value on PostgreSQL
Description
	 
		(last modified by Sven R. Kunze)
	 
MyModel.objects.annotate(my_column=Value('asdf')).order_by('my_column').values_list('id')
ProgrammingError: non-integer constant in ORDER BY
LINE 1: ...odel"."id" FROM "mymodel" ORDER BY 'asdf' ASC...
Does it qualify as a bug this time? ;-)

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 django/db/models/expressions.py | 1102 | 1135| 292 | 292 | 9860 | 
| 2 | **2 django/db/models/sql/compiler.py** | 336 | 364| 335 | 627 | 23385 | 
| 3 | 2 django/db/models/expressions.py | 1085 | 1100| 148 | 775 | 23385 | 
| 4 | 3 django/contrib/postgres/aggregates/mixins.py | 22 | 34| 128 | 903 | 23895 | 
| 5 | 4 django/db/models/base.py | 1624 | 1716| 673 | 1576 | 38855 | 
| 6 | 4 django/contrib/postgres/aggregates/mixins.py | 1 | 20| 181 | 1757 | 38855 | 
| 7 | 4 django/db/models/expressions.py | 641 | 665| 238 | 1995 | 38855 | 
| 8 | 5 django/db/models/sql/query.py | 1826 | 1859| 279 | 2274 | 60020 | 
| 9 | **5 django/db/models/sql/compiler.py** | 687 | 716| 351 | 2625 | 60020 | 
| 10 | 6 django/db/backends/mysql/operations.py | 127 | 158| 294 | 2919 | 63070 | 
| 11 | 7 django/db/models/sql/where.py | 228 | 244| 130 | 3049 | 64845 | 
| 12 | 7 django/db/models/sql/query.py | 883 | 905| 248 | 3297 | 64845 | 
| 13 | 7 django/db/models/expressions.py | 1062 | 1083| 183 | 3480 | 64845 | 
| 14 | 8 django/db/backends/postgresql/operations.py | 157 | 199| 454 | 3934 | 67473 | 
| 15 | 8 django/db/models/base.py | 934 | 948| 212 | 4146 | 67473 | 
| 16 | 9 django/contrib/admin/views/main.py | 309 | 361| 467 | 4613 | 71550 | 
| 17 | 9 django/db/models/sql/query.py | 1043 | 1067| 243 | 4856 | 71550 | 
| 18 | 9 django/db/models/base.py | 950 | 963| 180 | 5036 | 71550 | 
| 19 | 9 django/db/models/sql/query.py | 1616 | 1643| 351 | 5387 | 71550 | 
| 20 | 9 django/db/models/sql/query.py | 358 | 408| 494 | 5881 | 71550 | 
| **-> 21 <-** | **9 django/db/models/sql/compiler.py** | 253 | 334| 637 | 6518 | 71550 | 
| 22 | 9 django/db/models/base.py | 1815 | 1866| 351 | 6869 | 71550 | 
| 23 | **9 django/db/models/sql/compiler.py** | 718 | 729| 147 | 7016 | 71550 | 
| 24 | 10 django/db/migrations/operations/models.py | 614 | 627| 137 | 7153 | 78296 | 
| 25 | 11 django/db/models/sql/constants.py | 1 | 28| 160 | 7313 | 78456 | 
| 26 | 11 django/db/backends/postgresql/operations.py | 201 | 285| 700 | 8013 | 78456 | 
| 27 | 11 django/db/models/sql/query.py | 858 | 881| 203 | 8216 | 78456 | 
| 28 | 11 django/db/models/sql/query.py | 1265 | 1302| 512 | 8728 | 78456 | 
| 29 | 11 django/contrib/postgres/aggregates/mixins.py | 36 | 58| 214 | 8942 | 78456 | 
| 30 | 11 django/db/models/sql/query.py | 2030 | 2063| 246 | 9188 | 78456 | 
| 31 | 12 django/contrib/admin/options.py | 206 | 217| 135 | 9323 | 96809 | 
| 32 | 13 django/db/models/aggregates.py | 45 | 68| 294 | 9617 | 98096 | 
| 33 | 13 django/db/models/sql/query.py | 690 | 725| 389 | 10006 | 98096 | 
| 34 | **13 django/db/models/sql/compiler.py** | 1121 | 1142| 213 | 10219 | 98096 | 
| 35 | 13 django/db/migrations/operations/models.py | 571 | 594| 183 | 10402 | 98096 | 
| 36 | 14 django/db/backends/base/operations.py | 193 | 224| 285 | 10687 | 103489 | 
| 37 | 15 django/db/models/query.py | 1138 | 1172| 234 | 10921 | 119960 | 
| 38 | 15 django/db/models/base.py | 1431 | 1454| 176 | 11097 | 119960 | 
| 39 | 15 django/db/migrations/operations/models.py | 596 | 612| 215 | 11312 | 119960 | 
| 40 | 15 django/db/models/base.py | 1718 | 1789| 565 | 11877 | 119960 | 
| 41 | 15 django/db/models/query.py | 800 | 829| 248 | 12125 | 119960 | 
| **-> 42 <-** | **15 django/db/models/sql/compiler.py** | 1 | 19| 157 | 12282 | 119960 | 
| 43 | 15 django/db/models/sql/query.py | 1406 | 1484| 734 | 13016 | 119960 | 
| 44 | 16 django/core/paginator.py | 110 | 136| 203 | 13219 | 121275 | 
| 45 | 16 django/db/models/sql/query.py | 1716 | 1743| 244 | 13463 | 121275 | 
| 46 | 17 django/contrib/postgres/search.py | 147 | 154| 121 | 13584 | 123213 | 
| 47 | 17 django/db/models/sql/query.py | 248 | 290| 308 | 13892 | 123213 | 
| 48 | 17 django/db/models/expressions.py | 668 | 708| 247 | 14139 | 123213 | 
| 49 | **17 django/db/models/sql/compiler.py** | 366 | 374| 119 | 14258 | 123213 | 
| 50 | 17 django/db/models/query.py | 1346 | 1354| 136 | 14394 | 123213 | 
| 51 | 18 django/views/generic/list.py | 50 | 75| 244 | 14638 | 124786 | 
| 52 | 18 django/db/models/query.py | 1066 | 1100| 298 | 14936 | 124786 | 
| 53 | 19 django/db/backends/mysql/compiler.py | 1 | 26| 163 | 15099 | 124950 | 
| 54 | **19 django/db/models/sql/compiler.py** | 963 | 997| 299 | 15398 | 124950 | 
| 55 | 19 django/db/models/sql/query.py | 145 | 246| 867 | 16265 | 124950 | 
| 56 | 20 django/contrib/postgres/aggregates/general.py | 1 | 64| 372 | 16637 | 125323 | 
| 57 | 21 django/db/models/lookups.py | 96 | 125| 243 | 16880 | 129477 | 
| 58 | 22 django/db/models/options.py | 204 | 242| 362 | 17242 | 136343 | 
| 59 | 23 django/db/models/__init__.py | 1 | 49| 548 | 17790 | 136891 | 
| 60 | **23 django/db/models/sql/compiler.py** | 1437 | 1477| 409 | 18199 | 136891 | 
| 61 | 23 django/contrib/postgres/search.py | 156 | 188| 287 | 18486 | 136891 | 
| 62 | 24 django/db/backends/sqlite3/schema.py | 38 | 64| 243 | 18729 | 140845 | 
| 63 | 25 django/db/backends/oracle/operations.py | 435 | 467| 326 | 19055 | 146329 | 
| 64 | 26 django/db/backends/postgresql/features.py | 1 | 70| 571 | 19626 | 146900 | 
| 65 | 26 django/db/models/sql/query.py | 1562 | 1586| 227 | 19853 | 146900 | 
| 66 | 27 django/db/backends/sqlite3/operations.py | 163 | 188| 190 | 20043 | 149784 | 
| 67 | 27 django/db/models/base.py | 1456 | 1478| 171 | 20214 | 149784 | 
| 68 | 28 django/db/backends/postgresql/schema.py | 1 | 40| 404 | 20618 | 151253 | 
| 69 | 29 django/db/models/sql/subqueries.py | 79 | 107| 210 | 20828 | 152772 | 
| 70 | 29 django/db/models/expressions.py | 742 | 773| 222 | 21050 | 152772 | 
| 71 | 29 django/contrib/postgres/search.py | 131 | 145| 142 | 21192 | 152772 | 
| 72 | 29 django/db/models/sql/query.py | 975 | 1006| 307 | 21499 | 152772 | 
| 73 | 30 django/db/models/fields/__init__.py | 242 | 290| 320 | 21819 | 169766 | 
| 74 | 30 django/db/backends/postgresql/schema.py | 64 | 125| 436 | 22255 | 169766 | 
| 75 | 30 django/db/backends/sqlite3/schema.py | 100 | 137| 486 | 22741 | 169766 | 
| 76 | 30 django/db/models/query.py | 759 | 798| 322 | 23063 | 169766 | 
| 77 | **30 django/db/models/sql/compiler.py** | 22 | 43| 248 | 23311 | 169766 | 
| 78 | 31 django/db/backends/postgresql/introspection.py | 145 | 217| 756 | 24067 | 172125 | 
| 79 | 31 django/db/models/sql/query.py | 1098 | 1116| 232 | 24299 | 172125 | 
| 80 | 31 django/db/models/sql/query.py | 1069 | 1096| 285 | 24584 | 172125 | 
| 81 | 31 django/db/backends/sqlite3/operations.py | 1 | 40| 269 | 24853 | 172125 | 
| 82 | 31 django/db/models/query.py | 643 | 657| 132 | 24985 | 172125 | 
| 83 | 31 django/db/backends/mysql/operations.py | 1 | 32| 244 | 25229 | 172125 | 
| 84 | 32 django/contrib/gis/db/models/functions.py | 88 | 119| 231 | 25460 | 175924 | 
| 85 | 32 django/db/models/sql/query.py | 2192 | 2203| 116 | 25576 | 175924 | 
| 86 | 32 django/db/models/sql/query.py | 1745 | 1786| 283 | 25859 | 175924 | 
| 87 | 32 django/db/models/sql/where.py | 65 | 115| 396 | 26255 | 175924 | 
| 88 | 32 django/db/models/fields/__init__.py | 592 | 621| 234 | 26489 | 175924 | 
| 89 | 33 django/db/models/constants.py | 1 | 7| 0 | 26489 | 175949 | 
| 90 | 33 django/contrib/postgres/search.py | 1 | 21| 201 | 26690 | 175949 | 
| 91 | **33 django/db/models/sql/compiler.py** | 137 | 181| 490 | 27180 | 175949 | 
| 92 | 33 django/db/models/aggregates.py | 70 | 96| 266 | 27446 | 175949 | 
| 93 | 34 django/forms/models.py | 1295 | 1330| 286 | 27732 | 187398 | 
| 94 | 34 django/contrib/postgres/search.py | 191 | 240| 338 | 28070 | 187398 | 
| 95 | 34 django/contrib/admin/views/main.py | 363 | 401| 334 | 28404 | 187398 | 
| 96 | 34 django/db/models/expressions.py | 711 | 739| 230 | 28634 | 187398 | 
| 97 | 34 django/db/models/base.py | 1514 | 1539| 183 | 28817 | 187398 | 
| 98 | 34 django/db/backends/oracle/operations.py | 1 | 63| 554 | 29371 | 187398 | 
| 99 | 35 django/contrib/admin/checks.py | 546 | 581| 303 | 29674 | 196412 | 
| 100 | 35 django/db/backends/oracle/operations.py | 279 | 304| 202 | 29876 | 196412 | 
| 101 | 36 django/db/models/fields/related_lookups.py | 26 | 43| 157 | 30033 | 197858 | 
| 102 | 36 django/db/models/base.py | 1236 | 1265| 242 | 30275 | 197858 | 
| 103 | 36 django/db/models/sql/query.py | 1377 | 1388| 137 | 30412 | 197858 | 
| 104 | 36 django/db/models/fields/related_lookups.py | 62 | 99| 447 | 30859 | 197858 | 
| 105 | 37 django/db/models/functions/comparison.py | 31 | 40| 158 | 31017 | 198936 | 
| 106 | **37 django/db/models/sql/compiler.py** | 1180 | 1216| 341 | 31358 | 198936 | 
| 107 | 38 django/contrib/postgres/aggregates/statistics.py | 1 | 19| 206 | 31564 | 199403 | 
| 108 | 38 django/db/backends/sqlite3/operations.py | 118 | 143| 279 | 31843 | 199403 | 
| 109 | 38 django/db/models/fields/related_lookups.py | 119 | 155| 244 | 32087 | 199403 | 
| 110 | 38 django/db/backends/oracle/operations.py | 332 | 369| 328 | 32415 | 199403 | 
| 111 | 38 django/contrib/admin/views/main.py | 239 | 269| 270 | 32685 | 199403 | 


### Hint

```
I don't see what the use of ordering by a constant string value is. Anyway, it looks like this is a database limitation.
I don't see what the use of ordering by a constant string value is. Reducing code complexity (e.g. fewer ifs). The code overview from #26192: # 1 create complex queryset ... more code # 2 annotate and add extra where ... more code # 3 add order (maybe referring to the extra column) ... more code # 4 wrap paginator around queryset ... more code # 5 create values_list of paged queryset ... more code # 6 evaluate <<<< crash The code above spread over several files and few thousand lines builds up a quite complex query. Each point contribute to the final queryset (actually more than one queryset). In order to reduce coupling and code complexity, adding a constant column make things straightforward (otherwise we would need to check if the column was added). it looks like this is a database limitation. Are you sure? From what I know of SQL, it's possible to order by column name (a string) or by column index (a number). Django just creates invalid SQL. Why does Django not refer the column order_by by name (or index) and instead inserts a value?
Sorry for misunderstanding, however, I cannot reproduce a crash on the stable/1.8.x branch: from django.db.models import Value from polls.models import Question Question.objects.annotate(my_column=Value('asdf')).order_by('my_column').values_list('id') [(1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,), (10,)] Tested on SQLite, PostgreSQL, and MySQL. Can you provide more details to reproduce?
Interesting. We upgraded to 1.8.9 and use PostgreSQL 9.3.10. Our testsuite runs without errors. Can you provide your query string? I would be interested in how the order clause looks. Ours: SELECT "mymodel"."id" FROM "app_mymodel" ORDER BY 'asdf' ASC
str(Question.objects.annotate(my_column=Value('asdf')).order_by('my_column').values_list('id').query) 'SELECT "polls_question"."id" FROM "polls_question" ORDER BY asdf ASC' PostgreSQL 9.5.0 and psycopg2 2.6.1 here.
psycopg2 2.5.1 psql: =# SELECT "mymodel"."id" FROM "mymodel" ORDER BY 'asdf' ASC; ERROR: non-integer constant in ORDER BY LINE 1: ...odel"."id" FROM "mymodel" ORDER BY 'asdf' ASC... ^ =# SELECT "mymodel"."id" FROM "mymodel" ORDER BY "asdf" ASC; ERROR: column "asdf" does not exist LINE 1: ...odel"."id" FROM "mymodel" ORDER BY "asdf" ASC... ^ SELECT "mymodel"."id" FROM "mymodel" ORDER BY asdf ASC; ERROR: column "asdf" does not exist LINE 1: ...odel"."id" FROM "mymodel" ORDER BY asdf ASC; ^
I can reproduce on PostgreSQL now (not sure if the issue affects other databases). Looks like the query needs to be something like SELECT "polls_question"."id" FROM "polls_question" ORDER BY 'asdf'::text ASC; ​reference
That seems to work. On the other hand, your comment ("I don't see what the use of ordering by a constant string value is") made me think of another way: SELECT "polls_question"."id" FROM "polls_question"; Dropping the order clause entirely might even solve that problem for other databases.
Hi I added a ​PR. IMO when we use value expression in order by but don't include in select columns, the query occurs incorrectly.
```

## Patch

```diff
diff --git a/django/db/models/sql/compiler.py b/django/db/models/sql/compiler.py
--- a/django/db/models/sql/compiler.py
+++ b/django/db/models/sql/compiler.py
@@ -5,7 +5,8 @@
 
 from django.core.exceptions import EmptyResultSet, FieldError
 from django.db.models.constants import LOOKUP_SEP
-from django.db.models.expressions import OrderBy, Random, RawSQL, Ref
+from django.db.models.expressions import OrderBy, Random, RawSQL, Ref, Value
+from django.db.models.functions import Cast
 from django.db.models.query_utils import QueryWrapper, select_related_descend
 from django.db.models.sql.constants import (
     CURSOR, GET_ITERATOR_CHUNK_SIZE, MULTI, NO_RESULTS, ORDER_DIR, SINGLE,
@@ -278,6 +279,9 @@ def get_order_by(self):
         order_by = []
         for field in ordering:
             if hasattr(field, 'resolve_expression'):
+                if isinstance(field, Value):
+                    # output_field must be resolved for constants.
+                    field = Cast(field, field.output_field)
                 if not isinstance(field, OrderBy):
                     field = field.asc()
                 if not self.query.standard_ordering:
@@ -299,10 +303,13 @@ def get_order_by(self):
                     True))
                 continue
             if col in self.query.annotations:
-                # References to an expression which is masked out of the SELECT clause
-                order_by.append((
-                    OrderBy(self.query.annotations[col], descending=descending),
-                    False))
+                # References to an expression which is masked out of the SELECT
+                # clause.
+                expr = self.query.annotations[col]
+                if isinstance(expr, Value):
+                    # output_field must be resolved for constants.
+                    expr = Cast(expr, expr.output_field)
+                order_by.append((OrderBy(expr, descending=descending), False))
                 continue
 
             if '.' in field:

```

## Test Patch

```diff
diff --git a/tests/ordering/tests.py b/tests/ordering/tests.py
--- a/tests/ordering/tests.py
+++ b/tests/ordering/tests.py
@@ -1,7 +1,10 @@
 from datetime import datetime
 from operator import attrgetter
 
-from django.db.models import Count, DateTimeField, F, Max, OuterRef, Subquery
+from django.core.exceptions import FieldError
+from django.db.models import (
+    CharField, Count, DateTimeField, F, Max, OuterRef, Subquery, Value,
+)
 from django.db.models.functions import Upper
 from django.test import TestCase
 from django.utils.deprecation import RemovedInDjango31Warning
@@ -402,6 +405,36 @@ def test_order_by_f_expression_duplicates(self):
             attrgetter("headline")
         )
 
+    def test_order_by_constant_value(self):
+        # Order by annotated constant from selected columns.
+        qs = Article.objects.annotate(
+            constant=Value('1', output_field=CharField()),
+        ).order_by('constant', '-headline')
+        self.assertSequenceEqual(qs, [self.a4, self.a3, self.a2, self.a1])
+        # Order by annotated constant which is out of selected columns.
+        self.assertSequenceEqual(
+            qs.values_list('headline', flat=True), [
+                'Article 4',
+                'Article 3',
+                'Article 2',
+                'Article 1',
+            ],
+        )
+        # Order by constant.
+        qs = Article.objects.order_by(Value('1', output_field=CharField()), '-headline')
+        self.assertSequenceEqual(qs, [self.a4, self.a3, self.a2, self.a1])
+
+    def test_order_by_constant_value_without_output_field(self):
+        msg = 'Cannot resolve expression type, unknown output_field'
+        qs = Article.objects.annotate(constant=Value('1')).order_by('constant')
+        for ordered_qs in (
+            qs,
+            qs.values('headline'),
+            Article.objects.order_by(Value('1')),
+        ):
+            with self.subTest(ordered_qs=ordered_qs), self.assertRaisesMessage(FieldError, msg):
+                ordered_qs.first()
+
     def test_related_ordering_duplicate_table_reference(self):
         """
         An ordering referencing a model with an ordering referencing a model

```


## Code snippets

### 1 - django/db/models/expressions.py:

Start line: 1102, End line: 1135

```python
class OrderBy(BaseExpression):

    def as_sqlite(self, compiler, connection):
        template = None
        if self.nulls_last:
            template = '%(expression)s IS NULL, %(expression)s %(ordering)s'
        elif self.nulls_first:
            template = '%(expression)s IS NOT NULL, %(expression)s %(ordering)s'
        return self.as_sql(compiler, connection, template=template)

    def as_mysql(self, compiler, connection):
        template = None
        if self.nulls_last:
            template = 'IF(ISNULL(%(expression)s),1,0), %(expression)s %(ordering)s '
        elif self.nulls_first:
            template = 'IF(ISNULL(%(expression)s),0,1), %(expression)s %(ordering)s '
        return self.as_sql(compiler, connection, template=template)

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
### 2 - django/db/models/sql/compiler.py:

Start line: 336, End line: 364

```python
class SQLCompiler:

    def get_order_by(self):
        # ... other code

        for expr, is_ref in order_by:
            resolved = expr.resolve_expression(self.query, allow_joins=True, reuse=None)
            if self.query.combinator:
                src = resolved.get_source_expressions()[0]
                # Relabel order by columns to raw numbers if this is a combined
                # query; necessary since the columns can't be referenced by the
                # fully qualified name and the simple column names may collide.
                for idx, (sel_expr, _, col_alias) in enumerate(self.select):
                    if is_ref and col_alias == src.refs:
                        src = src.source
                    elif col_alias:
                        continue
                    if src == sel_expr:
                        resolved.set_source_expressions([RawSQL('%d' % (idx + 1), ())])
                        break
                else:
                    raise DatabaseError('ORDER BY term does not match any column in the result set.')
            sql, params = self.compile(resolved)
            # Don't add the same column twice, but the order direction is
            # not taken into account so we strip it. When this entire method
            # is refactored into expressions, then we can check each part as we
            # generate it.
            without_ordering = self.ordering_parts.search(sql).group(1)
            params_hash = make_hashable(params)
            if (without_ordering, params_hash) in seen:
                continue
            seen.add((without_ordering, params_hash))
            result.append((resolved, (sql, params, is_ref)))
        return result
```
### 3 - django/db/models/expressions.py:

Start line: 1085, End line: 1100

```python
class OrderBy(BaseExpression):

    def as_sql(self, compiler, connection, template=None, **extra_context):
        if not template:
            if self.nulls_last:
                template = '%s NULLS LAST' % self.template
            elif self.nulls_first:
                template = '%s NULLS FIRST' % self.template
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
### 4 - django/contrib/postgres/aggregates/mixins.py:

Start line: 22, End line: 34

```python
class OrderableAggMixin:

    def as_sql(self, compiler, connection):
        if self.ordering:
            ordering_params = []
            ordering_expr_sql = []
            for expr in self.ordering:
                expr_sql, expr_params = expr.as_sql(compiler, connection)
                ordering_expr_sql.append(expr_sql)
                ordering_params.extend(expr_params)
            sql, sql_params = super().as_sql(compiler, connection, ordering=(
                'ORDER BY ' + ', '.join(ordering_expr_sql)
            ))
            return sql, sql_params + ordering_params
        return super().as_sql(compiler, connection, ordering='')
```
### 5 - django/db/models/base.py:

Start line: 1624, End line: 1716

```python
class Model(metaclass=ModelBase):

    @classmethod
    def _check_ordering(cls):
        """
        Check "ordering" option -- is it a list of strings and do all fields
        exist?
        """
        if cls._meta._ordering_clash:
            return [
                checks.Error(
                    "'ordering' and 'order_with_respect_to' cannot be used together.",
                    obj=cls,
                    id='models.E021',
                ),
            ]

        if cls._meta.order_with_respect_to or not cls._meta.ordering:
            return []

        if not isinstance(cls._meta.ordering, (list, tuple)):
            return [
                checks.Error(
                    "'ordering' must be a tuple or list (even if you want to order by only one field).",
                    obj=cls,
                    id='models.E014',
                )
            ]

        errors = []
        fields = cls._meta.ordering

        # Skip expressions and '?' fields.
        fields = (f for f in fields if isinstance(f, str) and f != '?')

        # Convert "-field" to "field".
        fields = ((f[1:] if f.startswith('-') else f) for f in fields)

        # Separate related fields and non-related fields.
        _fields = []
        related_fields = []
        for f in fields:
            if LOOKUP_SEP in f:
                related_fields.append(f)
            else:
                _fields.append(f)
        fields = _fields

        # Check related fields.
        for field in related_fields:
            _cls = cls
            fld = None
            for part in field.split(LOOKUP_SEP):
                try:
                    fld = _cls._meta.get_field(part)
                    if fld.is_relation:
                        _cls = fld.get_path_info()[-1].to_opts.model
                except (FieldDoesNotExist, AttributeError):
                    if fld is None or fld.get_transform(part) is None:
                        errors.append(
                            checks.Error(
                                "'ordering' refers to the nonexistent field, "
                                "related field, or lookup '%s'." % field,
                                obj=cls,
                                id='models.E015',
                            )
                        )

        # Skip ordering on pk. This is always a valid order_by field
        # but is an alias and therefore won't be found by opts.get_field.
        fields = {f for f in fields if f != 'pk'}

        # Check for invalid or nonexistent fields in ordering.
        invalid_fields = []

        # Any field name that is not present in field_names does not exist.
        # Also, ordering by m2m fields is not allowed.
        opts = cls._meta
        valid_fields = set(chain.from_iterable(
            (f.name, f.attname) if not (f.auto_created and not f.concrete) else (f.field.related_query_name(),)
            for f in chain(opts.fields, opts.related_objects)
        ))

        invalid_fields.extend(fields - valid_fields)

        for invalid_field in invalid_fields:
            errors.append(
                checks.Error(
                    "'ordering' refers to the nonexistent field, related "
                    "field, or lookup '%s'." % invalid_field,
                    obj=cls,
                    id='models.E015',
                )
            )
        return errors
```
### 6 - django/contrib/postgres/aggregates/mixins.py:

Start line: 1, End line: 20

```python
from django.db.models.expressions import F, OrderBy


class OrderableAggMixin:

    def __init__(self, expression, ordering=(), **extra):
        if not isinstance(ordering, (list, tuple)):
            ordering = [ordering]
        ordering = ordering or []
        # Transform minus sign prefixed strings into an OrderBy() expression.
        ordering = (
            (OrderBy(F(o[1:]), descending=True) if isinstance(o, str) and o[0] == '-' else o)
            for o in ordering
        )
        super().__init__(expression, **extra)
        self.ordering = self._parse_expressions(*ordering)

    def resolve_expression(self, *args, **kwargs):
        self.ordering = [expr.resolve_expression(*args, **kwargs) for expr in self.ordering]
        return super().resolve_expression(*args, **kwargs)
```
### 7 - django/db/models/expressions.py:

Start line: 641, End line: 665

```python
class Value(Expression):

    def as_sql(self, compiler, connection):
        connection.ops.check_expression_support(self)
        val = self.value
        output_field = self._output_field_or_none
        if output_field is not None:
            if self.for_save:
                val = output_field.get_db_prep_save(val, connection=connection)
            else:
                val = output_field.get_db_prep_value(val, connection=connection)
            if hasattr(output_field, 'get_placeholder'):
                return output_field.get_placeholder(val, compiler, connection), [val]
        if val is None:
            # cx_Oracle does not always convert None to the appropriate
            # NULL type (like in case expressions using numbers), so we
            # use a literal SQL NULL
            return 'NULL', []
        return '%s', [val]

    def resolve_expression(self, query=None, allow_joins=True, reuse=None, summarize=False, for_save=False):
        c = super().resolve_expression(query, allow_joins, reuse, summarize, for_save)
        c.for_save = for_save
        return c

    def get_group_by_cols(self, alias=None):
        return []
```
### 8 - django/db/models/sql/query.py:

Start line: 1826, End line: 1859

```python
class Query(BaseExpression):

    def add_ordering(self, *ordering):
        """
        Add items from the 'ordering' sequence to the query's "order by"
        clause. These items are either field names (not column names) --
        possibly with a direction prefix ('-' or '?') -- or OrderBy
        expressions.

        If 'ordering' is empty, clear all ordering from the query.
        """
        errors = []
        for item in ordering:
            if not hasattr(item, 'resolve_expression') and not ORDER_PATTERN.match(item):
                errors.append(item)
            if getattr(item, 'contains_aggregate', False):
                raise FieldError(
                    'Using an aggregate in order_by() without also including '
                    'it in annotate() is not allowed: %s' % item
                )
        if errors:
            raise FieldError('Invalid order_by arguments: %s' % errors)
        if ordering:
            self.order_by += ordering
        else:
            self.default_ordering = False

    def clear_ordering(self, force_empty):
        """
        Remove any ordering settings. If 'force_empty' is True, there will be
        no ordering in the resulting query (not even the model's default).
        """
        self.order_by = ()
        self.extra_order_by = ()
        if force_empty:
            self.default_ordering = False
```
### 9 - django/db/models/sql/compiler.py:

Start line: 687, End line: 716

```python
class SQLCompiler:

    def find_ordering_name(self, name, opts, alias=None, default_order='ASC',
                           already_seen=None):
        """
        Return the table alias (the name might be ambiguous, the alias will
        not be) and column name for ordering by the given 'name' parameter.
        The 'name' is of the form 'field1__field2__...__fieldN'.
        """
        name, order = get_order_dir(name, default_order)
        descending = order == 'DESC'
        pieces = name.split(LOOKUP_SEP)
        field, targets, alias, joins, path, opts, transform_function = self._setup_joins(pieces, opts, alias)

        # If we get to this point and the field is a relation to another model,
        # append the default ordering for that model unless the attribute name
        # of the field is specified.
        if field.is_relation and opts.ordering and getattr(field, 'attname', None) != name:
            # Firstly, avoid infinite loops.
            already_seen = already_seen or set()
            join_tuple = tuple(getattr(self.query.alias_map[j], 'join_cols', None) for j in joins)
            if join_tuple in already_seen:
                raise FieldError('Infinite loop caused by ordering.')
            already_seen.add(join_tuple)

            results = []
            for item in opts.ordering:
                results.extend(self.find_ordering_name(item, opts, alias,
                                                       order, already_seen))
            return results
        targets, alias, _ = self.query.trim_joins(targets, joins, path)
        return [(OrderBy(transform_function(t, alias), descending=descending), False) for t in targets]
```
### 10 - django/db/backends/mysql/operations.py:

Start line: 127, End line: 158

```python
class DatabaseOperations(BaseDatabaseOperations):

    def date_interval_sql(self, timedelta):
        return 'INTERVAL %s MICROSECOND' % duration_microseconds(timedelta)

    def format_for_duration_arithmetic(self, sql):
        return 'INTERVAL %s MICROSECOND' % sql

    def force_no_ordering(self):
        """
        "ORDER BY NULL" prevents MySQL from implicitly ordering by grouped
        columns. If no ordering would otherwise be applied, we don't want any
        implicit sorting going on.
        """
        return [(None, ("NULL", [], False))]

    def last_executed_query(self, cursor, sql, params):
        # With MySQLdb, cursor objects have an (undocumented) "_executed"
        # attribute where the exact query sent to the database is saved.
        # See MySQLdb/cursors.py in the source distribution.
        # MySQLdb returns string, PyMySQL bytes.
        return force_str(getattr(cursor, '_executed', None), errors='replace')

    def no_limit_value(self):
        # 2**64 - 1, as recommended by the MySQL documentation
        return 18446744073709551615

    def quote_name(self, name):
        if name.startswith("`") and name.endswith("`"):
            return name  # Quoting once is enough.
        return "`%s`" % name

    def random_function_sql(self):
        return 'RAND()'
```
### 21 - django/db/models/sql/compiler.py:

Start line: 253, End line: 334

```python
class SQLCompiler:

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
                # References to an expression which is masked out of the SELECT clause
                order_by.append((
                    OrderBy(self.query.annotations[col], descending=descending),
                    False))
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
```
### 23 - django/db/models/sql/compiler.py:

Start line: 718, End line: 729

```python
class SQLCompiler:

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
```
### 34 - django/db/models/sql/compiler.py:

Start line: 1121, End line: 1142

```python
class SQLCompiler:

    def as_subquery_condition(self, alias, columns, compiler):
        qn = compiler.quote_name_unless_alias
        qn2 = self.connection.ops.quote_name

        for index, select_col in enumerate(self.query.select):
            lhs_sql, lhs_params = self.compile(select_col)
            rhs = '%s.%s' % (qn(alias), qn2(columns[index]))
            self.query.where.add(
                QueryWrapper('%s = %s' % (lhs_sql, rhs), lhs_params), 'AND')

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
### 42 - django/db/models/sql/compiler.py:

Start line: 1, End line: 19

```python
import collections
import re
import warnings
from itertools import chain

from django.core.exceptions import EmptyResultSet, FieldError
from django.db.models.constants import LOOKUP_SEP
from django.db.models.expressions import OrderBy, Random, RawSQL, Ref
from django.db.models.query_utils import QueryWrapper, select_related_descend
from django.db.models.sql.constants import (
    CURSOR, GET_ITERATOR_CHUNK_SIZE, MULTI, NO_RESULTS, ORDER_DIR, SINGLE,
)
from django.db.models.sql.query import Query, get_order_dir
from django.db.transaction import TransactionManagementError
from django.db.utils import DatabaseError, NotSupportedError
from django.utils.deprecation import RemovedInDjango31Warning
from django.utils.hashable import make_hashable

FORCE = object()
```
### 49 - django/db/models/sql/compiler.py:

Start line: 366, End line: 374

```python
class SQLCompiler:

    def get_extra_select(self, order_by, select):
        extra_select = []
        if self.query.distinct and not self.query.distinct_fields:
            select_sql = [t[1] for t in select]
            for expr, (sql, params, is_ref) in order_by:
                without_ordering = self.ordering_parts.search(sql).group(1)
                if not is_ref and (without_ordering, params) not in select_sql:
                    extra_select.append((expr, (without_ordering, params), None))
        return extra_select
```
### 54 - django/db/models/sql/compiler.py:

Start line: 963, End line: 997

```python
class SQLCompiler:

    def get_select_for_update_of_arguments(self):
        # ... other code
        result = []
        invalid_names = []
        for name in self.query.select_for_update_of:
            parts = [] if name == 'self' else name.split(LOOKUP_SEP)
            klass_info = self.klass_info
            for part in parts:
                for related_klass_info in klass_info.get('related_klass_infos', []):
                    field = related_klass_info['field']
                    if related_klass_info['reverse']:
                        field = field.remote_field
                    if field.name == part:
                        klass_info = related_klass_info
                        break
                else:
                    klass_info = None
                    break
            if klass_info is None:
                invalid_names.append(name)
                continue
            select_index = klass_info['select_fields'][0]
            col = self.select[select_index][0]
            if self.connection.features.select_for_update_of_column:
                result.append(self.compile(col)[0])
            else:
                result.append(self.quote_name_unless_alias(col.alias))
        if invalid_names:
            raise FieldError(
                'Invalid field name(s) given in select_for_update(of=(...)): %s. '
                'Only relational fields followed in the query are allowed. '
                'Choices are: %s.' % (
                    ', '.join(invalid_names),
                    ', '.join(_get_field_choices()),
                )
            )
        return result
```
### 60 - django/db/models/sql/compiler.py:

Start line: 1437, End line: 1477

```python
class SQLUpdateCompiler(SQLCompiler):

    def pre_sql_setup(self):
        """
        If the update depends on results from other tables, munge the "where"
        conditions to match the format required for (portable) SQL updates.

        If multiple updates are required, pull out the id values to update at
        this point so that they don't change as a result of the progressive
        updates.
        """
        refcounts_before = self.query.alias_refcount.copy()
        # Ensure base table is in the query
        self.query.get_initial_alias()
        count = self.query.count_active_tables()
        if not self.query.related_updates and count == 1:
            return
        query = self.query.chain(klass=Query)
        query.select_related = False
        query.clear_ordering(True)
        query.extra = {}
        query.select = []
        query.add_fields([query.get_meta().pk.name])
        super().pre_sql_setup()

        must_pre_select = count > 1 and not self.connection.features.update_can_self_select

        # Now we adjust the current query: reset the where clause and get rid
        # of all the tables we don't need (since they're in the sub-select).
        self.query.where = self.query.where_class()
        if self.query.related_updates or must_pre_select:
            # Either we're using the idents in multiple update queries (so
            # don't want them to change), or the db backend doesn't support
            # selecting from the updating table (e.g. MySQL).
            idents = []
            for rows in query.get_compiler(self.using).execute_sql(MULTI):
                idents.extend(r[0] for r in rows)
            self.query.add_filter(('pk__in', idents))
            self.query.related_ids = idents
        else:
            # The fast path. Filters and updates in one query.
            self.query.add_filter(('pk__in', query))
        self.query.reset_refcounts(refcounts_before)
```
### 77 - django/db/models/sql/compiler.py:

Start line: 22, End line: 43

```python
class SQLCompiler:
    def __init__(self, query, connection, using):
        self.query = query
        self.connection = connection
        self.using = using
        self.quote_cache = {'*': '*'}
        # The select, klass_info, and annotations are needed by QuerySet.iterator()
        # these are set as a side-effect of executing the query. Note that we calculate
        # separately a list of extra select columns needed for grammatical correctness
        # of the query, but these columns are not included in self.select.
        self.select = None
        self.annotation_col_map = None
        self.klass_info = None
        # Multiline ordering SQL clause may appear from RawSQL.
        self.ordering_parts = re.compile(r'^(.*)\s(ASC|DESC)(.*)', re.MULTILINE | re.DOTALL)
        self._meta_ordering = None

    def setup_query(self):
        if all(self.query.alias_refcount[a] == 0 for a in self.query.alias_map):
            self.query.get_initial_alias()
        self.select, self.klass_info, self.annotation_col_map = self.get_select()
        self.col_count = len(self.select)
```
### 91 - django/db/models/sql/compiler.py:

Start line: 137, End line: 181

```python
class SQLCompiler:

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
                if hasattr(expr, 'target') and expr.target.primary_key and expr.target.model._meta.managed
            }
            aliases = {expr.alias for expr in pks}
            expressions = [
                expr for expr in expressions if expr in pks or getattr(expr, 'alias', None) not in aliases
            ]
        return expressions
```
### 106 - django/db/models/sql/compiler.py:

Start line: 1180, End line: 1216

```python
class SQLInsertCompiler(SQLCompiler):

    def prepare_value(self, field, value):
        """
        Prepare a value to be used in a query by resolving it if it is an
        expression and otherwise calling the field's get_db_prep_save().
        """
        if hasattr(value, 'resolve_expression'):
            value = value.resolve_expression(self.query, allow_joins=False, for_save=True)
            # Don't allow values containing Col expressions. They refer to
            # existing columns on a row, but in the case of insert the row
            # doesn't exist yet.
            if value.contains_column_references:
                raise ValueError(
                    'Failed to insert expression "%s" on %s. F() expressions '
                    'can only be used to update, not to insert.' % (value, field)
                )
            if value.contains_aggregate:
                raise FieldError(
                    'Aggregate functions are not allowed in this query '
                    '(%s=%r).' % (field.name, value)
                )
            if value.contains_over_clause:
                raise FieldError(
                    'Window expressions are not allowed in this query (%s=%r).'
                    % (field.name, value)
                )
        else:
            value = field.get_db_prep_save(value, connection=self.connection)
        return value

    def pre_save_val(self, field, obj):
        """
        Get the given field's value off the given obj. pre_save() is used for
        things like auto_now on DateTimeField. Skip it if this is a raw query.
        """
        if self.query.raw:
            return getattr(obj, field.attname)
        return field.pre_save(obj, add=True)
```
