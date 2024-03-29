# django__django-14894

| **django/django** | `9466fd78420a851460c92673dad50a5737c75b12` |
| ---- | ---- |
| **No of patches** | 6 |
| **All found context length** | 6886 |
| **Any found context length** | 6180 |
| **Avg pos** | 52.0 |
| **Min pos** | 16 |
| **Max pos** | 125 |
| **Top file pos** | 2 |
| **Missing snippets** | 12 |
| **Missing patch files** | 1 |


## Expected patch

```diff
diff --git a/django/contrib/postgres/aggregates/statistics.py b/django/contrib/postgres/aggregates/statistics.py
--- a/django/contrib/postgres/aggregates/statistics.py
+++ b/django/contrib/postgres/aggregates/statistics.py
@@ -36,7 +36,7 @@ class RegrAvgY(StatAggregate):
 class RegrCount(StatAggregate):
     function = 'REGR_COUNT'
     output_field = IntegerField()
-    empty_aggregate_value = 0
+    empty_result_set_value = 0
 
 
 class RegrIntercept(StatAggregate):
diff --git a/django/db/models/aggregates.py b/django/db/models/aggregates.py
--- a/django/db/models/aggregates.py
+++ b/django/db/models/aggregates.py
@@ -21,12 +21,12 @@ class Aggregate(Func):
     filter_template = '%s FILTER (WHERE %%(filter)s)'
     window_compatible = True
     allow_distinct = False
-    empty_aggregate_value = None
+    empty_result_set_value = None
 
     def __init__(self, *expressions, distinct=False, filter=None, default=None, **extra):
         if distinct and not self.allow_distinct:
             raise TypeError("%s does not allow distinct." % self.__class__.__name__)
-        if default is not None and self.empty_aggregate_value is not None:
+        if default is not None and self.empty_result_set_value is not None:
             raise TypeError(f'{self.__class__.__name__} does not allow default.')
         self.distinct = distinct
         self.filter = filter
@@ -117,7 +117,7 @@ class Count(Aggregate):
     name = 'Count'
     output_field = IntegerField()
     allow_distinct = True
-    empty_aggregate_value = 0
+    empty_result_set_value = 0
 
     def __init__(self, expression, filter=None, **extra):
         if expression == '*':
diff --git a/django/db/models/expressions.py b/django/db/models/expressions.py
--- a/django/db/models/expressions.py
+++ b/django/db/models/expressions.py
@@ -150,10 +150,10 @@ def __ror__(self, other):
 class BaseExpression:
     """Base class for all query expressions."""
 
+    empty_result_set_value = NotImplemented
     # aggregate specific fields
     is_summary = False
     _output_field_resolved_to_none = False
-    empty_aggregate_value = NotImplemented
     # Can the expression be used in a WHERE clause?
     filterable = True
     # Can the expression can be used as a source expression in Window?
@@ -702,7 +702,13 @@ def as_sql(self, compiler, connection, function=None, template=None, arg_joiner=
         sql_parts = []
         params = []
         for arg in self.source_expressions:
-            arg_sql, arg_params = compiler.compile(arg)
+            try:
+                arg_sql, arg_params = compiler.compile(arg)
+            except EmptyResultSet:
+                empty_result_set_value = getattr(arg, 'empty_result_set_value', NotImplemented)
+                if empty_result_set_value is NotImplemented:
+                    raise
+                arg_sql, arg_params = compiler.compile(Value(empty_result_set_value))
             sql_parts.append(arg_sql)
             params.extend(arg_params)
         data = {**self.extra, **extra_context}
@@ -797,7 +803,7 @@ def _resolve_output_field(self):
             return fields.UUIDField()
 
     @property
-    def empty_aggregate_value(self):
+    def empty_result_set_value(self):
         return self.value
 
 
@@ -1114,6 +1120,7 @@ class Subquery(BaseExpression, Combinable):
     """
     template = '(%(subquery)s)'
     contains_aggregate = False
+    empty_result_set_value = None
 
     def __init__(self, queryset, output_field=None, **extra):
         # Allow the usage of both QuerySet and sql.Query objects.
diff --git a/django/db/models/functions/comparison.py b/django/db/models/functions/comparison.py
--- a/django/db/models/functions/comparison.py
+++ b/django/db/models/functions/comparison.py
@@ -66,9 +66,9 @@ def __init__(self, *expressions, **extra):
         super().__init__(*expressions, **extra)
 
     @property
-    def empty_aggregate_value(self):
+    def empty_result_set_value(self):
         for expression in self.get_source_expressions():
-            result = expression.empty_aggregate_value
+            result = expression.empty_result_set_value
             if result is NotImplemented or result is not None:
                 return result
         return None
diff --git a/django/db/models/sql/compiler.py b/django/db/models/sql/compiler.py
--- a/django/db/models/sql/compiler.py
+++ b/django/db/models/sql/compiler.py
@@ -266,8 +266,12 @@ def get_select_from_parent(klass_info):
             try:
                 sql, params = self.compile(col)
             except EmptyResultSet:
-                # Select a predicate that's always False.
-                sql, params = '0', ()
+                empty_result_set_value = getattr(col, 'empty_result_set_value', NotImplemented)
+                if empty_result_set_value is NotImplemented:
+                    # Select a predicate that's always False.
+                    sql, params = '0', ()
+                else:
+                    sql, params = self.compile(Value(empty_result_set_value))
             else:
                 sql, params = col.select_format(self, sql, params)
             ret.append((col, (sql, params), alias))
diff --git a/django/db/models/sql/query.py b/django/db/models/sql/query.py
--- a/django/db/models/sql/query.py
+++ b/django/db/models/sql/query.py
@@ -143,6 +143,7 @@ class Query(BaseExpression):
     """A single SQL query."""
 
     alias_prefix = 'T'
+    empty_result_set_value = None
     subq_aliases = frozenset([alias_prefix])
 
     compiler = 'SQLCompiler'
@@ -487,11 +488,11 @@ def get_aggregation(self, using, added_aggregate_names):
             self.default_cols = False
             self.extra = {}
 
-        empty_aggregate_result = [
-            expression.empty_aggregate_value
+        empty_set_result = [
+            expression.empty_result_set_value
             for expression in outer_query.annotation_select.values()
         ]
-        elide_empty = not any(result is NotImplemented for result in empty_aggregate_result)
+        elide_empty = not any(result is NotImplemented for result in empty_set_result)
         outer_query.clear_ordering(force=True)
         outer_query.clear_limits()
         outer_query.select_for_update = False
@@ -499,7 +500,7 @@ def get_aggregation(self, using, added_aggregate_names):
         compiler = outer_query.get_compiler(using, elide_empty=elide_empty)
         result = compiler.execute_sql(SINGLE)
         if result is None:
-            result = empty_aggregate_result
+            result = empty_set_result
 
         converters = compiler.get_converters(outer_query.annotation_select.values())
         result = next(compiler.apply_converters((result,), converters))

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/contrib/postgres/aggregates/statistics.py | 39 | 39 | 58 | 15 | 21811
| django/db/models/aggregates.py | 24 | 29 | 70 | 7 | 26663
| django/db/models/aggregates.py | 120 | 120 | - | 7 | -
| django/db/models/expressions.py | 153 | 153 | - | 6 | -
| django/db/models/expressions.py | 705 | 705 | - | 6 | -
| django/db/models/expressions.py | 800 | 800 | 125 | 6 | 47180
| django/db/models/expressions.py | 1117 | 1117 | 16 | 6 | 6180
| django/db/models/functions/comparison.py | 69 | 71 | - | - | -
| django/db/models/sql/compiler.py | 269 | 270 | 18 | 3 | 6886
| django/db/models/sql/query.py | 146 | 146 | 25 | 2 | 9774
| django/db/models/sql/query.py | 490 | 494 | - | 2 | -
| django/db/models/sql/query.py | 502 | 502 | - | 2 | -


## Problem Statement

```
Incorrect annotation value when doing a subquery with empty queryset
Description
	
ORM seems to generate annotation/subqueries incorrectly if empty queryset is used. 
Models:
class Article(models.Model):
	author_name = models.CharField(max_length=100)
	content = models.TextField()
	is_public = models.BooleanField()
class Comment(models.Model):
	article = models.ForeignKey(Article, related_name="comments", on_delete=models.CASCADE)
	author_name = models.CharField(max_length=100)
	content = models.TextField()
test data:
article = Article.objects.create(author_name="Jack", content="Example content", is_public=True)
comment = Comment.objects.create(article=article, author_name="John", content="Example comment")
queries:
qs = Article.objects.all()
# keep one list_x uncommented to see the difference:
list_x = ["random_thing_that_is_not_equal_to_any_authors_name"] # list not empty, bug doesnt occur
#list_x = [] # if this list is empty, then the bug occurs
comment_qs = Comment.objects.filter(author_name__in=list_x)
qs = qs.annotate(
	A=Coalesce(Subquery(
		comment_qs.annotate(x=Count('content')).values('x')[:1], output_field=IntegerField(),
	), 101) # if list_x == [], Coalesce wont work and A will be 0 instead of 101
)
# please note that above annotation doesnt make much logical sense, its just for testing purposes
qs = qs.annotate(
	B=Value(99, output_field=IntegerField())
)
qs = qs.annotate(
	C=F("A") + F("B") # if list_x == [], C will result in 0 sic! instead of 101 + 99 = 200
)
data = {
	"A": qs.last().A,
	"B": qs.last().B,
	"C": qs.last().C,
}
print(data)
print(format_sql(qs.query))
console output for list_x=["random_thing_that_is_not_equal_to_any_authors_name"] (expected, correct):
{'A': 101, 'B': 99, 'C': 200}
SELECT "articles_article"."id",
	 "articles_article"."author_name",
	 "articles_article"."content",
	 "articles_article"."is_public",
	 COALESCE(
				 (SELECT COUNT(U0."content") AS "x"
				 FROM "articles_comment" U0
				 WHERE U0."author_name" IN (random_thing_that_is_not_equal_to_any_authors_name)
				 GROUP BY U0."id", U0."article_id", U0."author_name", U0."content"
				 LIMIT 1), 101) AS "A",
	 99 AS "B",
	 (COALESCE(
				 (SELECT COUNT(U0."content") AS "x"
					FROM "articles_comment" U0
					WHERE U0."author_name" IN (random_thing_that_is_not_equal_to_any_authors_name)
					GROUP BY U0."id", U0."article_id", U0."author_name", U0."content"
					LIMIT 1), 101) + 99) AS "C"
FROM "articles_article"
console output for list_x=[] (incorrect):
{'A': 0, 'B': 99, 'C': 0}
SELECT "articles_article"."id",
	 "articles_article"."author_name",
	 "articles_article"."content",
	 "articles_article"."is_public",
	 0 AS "A",
	 99 AS "B",
	 0 AS "C"
FROM "articles_article"
Background story: Above queries are made up (simplified), but based on some parts of logic that I had in my code. list_x was generated dynamically, and it was very hard to detect what is causing unexpected results. This behavior is very strange, I believe its a bug and needs to be fixed, because it is totally unintuitive that:
SomeModel.objects.filter(x__in=["something_that_causes_this_qs_lenth_to_be_0"])
and 
SomeModel.objects.filter(x__in=[]) 
may yield different results when used in queries later, even though results of this querysets are logically equivalent
I will attach a minimal repro project (with code from above)

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 django/db/models/query.py | 1134 | 1175| 323 | 323 | 17659 | 
| 2 | **2 django/db/models/sql/query.py** | 364 | 414| 494 | 817 | 40004 | 
| 3 | **2 django/db/models/sql/query.py** | 1701 | 1743| 436 | 1253 | 40004 | 
| 4 | 2 django/db/models/query.py | 842 | 873| 272 | 1525 | 40004 | 
| 5 | **3 django/db/models/sql/compiler.py** | 1233 | 1255| 244 | 1769 | 54754 | 
| 6 | **3 django/db/models/sql/query.py** | 1293 | 1357| 760 | 2529 | 54754 | 
| 7 | 3 django/db/models/query.py | 324 | 352| 242 | 2771 | 54754 | 
| 8 | 4 django/db/models/sql/where.py | 245 | 262| 141 | 2912 | 56687 | 
| 9 | 5 django/db/models/__init__.py | 1 | 53| 619 | 3531 | 57306 | 
| 10 | **5 django/db/models/sql/query.py** | 2166 | 2213| 394 | 3925 | 57306 | 
| 11 | **5 django/db/models/sql/query.py** | 906 | 928| 248 | 4173 | 57306 | 
| 12 | **5 django/db/models/sql/query.py** | 1427 | 1454| 283 | 4456 | 57306 | 
| 13 | **6 django/db/models/expressions.py** | 1165 | 1197| 254 | 4710 | 68417 | 
| 14 | **7 django/db/models/aggregates.py** | 50 | 68| 278 | 4988 | 69832 | 
| 15 | **7 django/db/models/sql/query.py** | 1472 | 1557| 801 | 5789 | 69832 | 
| **-> 16 <-** | **7 django/db/models/expressions.py** | 1110 | 1162| 391 | 6180 | 69832 | 
| 17 | 7 django/db/models/query.py | 1043 | 1056| 126 | 6306 | 69832 | 
| **-> 18 <-** | **7 django/db/models/sql/compiler.py** | 204 | 274| 580 | 6886 | 69832 | 
| 19 | **7 django/db/models/sql/compiler.py** | 23 | 51| 302 | 7188 | 69832 | 
| 20 | **7 django/db/models/sql/query.py** | 710 | 745| 389 | 7577 | 69832 | 
| 21 | **7 django/db/models/sql/query.py** | 1633 | 1654| 200 | 7777 | 69832 | 
| 22 | **7 django/db/models/sql/query.py** | 1745 | 1810| 662 | 8439 | 69832 | 
| 23 | **7 django/db/models/aggregates.py** | 70 | 106| 349 | 8788 | 69832 | 
| 24 | 7 django/db/models/query.py | 1119 | 1132| 115 | 8903 | 69832 | 
| **-> 25 <-** | **7 django/db/models/sql/query.py** | 142 | 241| 871 | 9774 | 69832 | 
| 26 | **7 django/db/models/sql/compiler.py** | 67 | 152| 890 | 10664 | 69832 | 
| 27 | 8 django/db/models/fields/related_lookups.py | 65 | 104| 451 | 11115 | 71301 | 
| 28 | 8 django/db/models/query.py | 1525 | 1556| 295 | 11410 | 71301 | 
| 29 | 8 django/db/models/fields/related_lookups.py | 124 | 160| 244 | 11654 | 71301 | 
| 30 | 9 django/db/models/fields/related.py | 1262 | 1379| 963 | 12617 | 85271 | 
| 31 | 9 django/db/models/query.py | 1442 | 1473| 245 | 12862 | 85271 | 
| 32 | 9 django/db/models/query.py | 954 | 1002| 371 | 13233 | 85271 | 
| 33 | **9 django/db/models/expressions.py** | 839 | 873| 292 | 13525 | 85271 | 
| 34 | **9 django/db/models/sql/query.py** | 1118 | 1150| 338 | 13863 | 85271 | 
| 35 | **9 django/db/models/sql/compiler.py** | 923 | 1015| 839 | 14702 | 85271 | 
| 36 | **9 django/db/models/sql/query.py** | 1 | 62| 450 | 15152 | 85271 | 
| 37 | **9 django/db/models/sql/compiler.py** | 1076 | 1116| 337 | 15489 | 85271 | 
| 38 | 9 django/db/models/query.py | 1021 | 1041| 240 | 15729 | 85271 | 
| 39 | 10 django/contrib/postgres/aggregates/general.py | 55 | 86| 217 | 15946 | 86066 | 
| 40 | **10 django/db/models/sql/compiler.py** | 276 | 371| 710 | 16656 | 86066 | 
| 41 | 10 django/db/models/query.py | 1387 | 1439| 448 | 17104 | 86066 | 
| 42 | 11 django/db/models/query_utils.py | 59 | 91| 288 | 17392 | 88554 | 
| 43 | **11 django/db/models/expressions.py** | 962 | 1026| 591 | 17983 | 88554 | 
| 44 | 12 django/db/backends/mysql/compiler.py | 1 | 15| 132 | 18115 | 89148 | 
| 45 | 12 django/db/models/query.py | 875 | 904| 248 | 18363 | 89148 | 
| 46 | **12 django/db/models/sql/compiler.py** | 462 | 515| 569 | 18932 | 89148 | 
| 47 | 12 django/db/models/query.py | 236 | 263| 221 | 19153 | 89148 | 
| 48 | **12 django/db/models/sql/query.py** | 1051 | 1069| 162 | 19315 | 89148 | 
| 49 | 13 django/db/models/sql/subqueries.py | 48 | 76| 208 | 19523 | 90336 | 
| 50 | **13 django/db/models/sql/query.py** | 1359 | 1383| 260 | 19783 | 90336 | 
| 51 | **13 django/db/models/sql/query.py** | 2215 | 2247| 228 | 20011 | 90336 | 
| 52 | **13 django/db/models/sql/compiler.py** | 1 | 20| 171 | 20182 | 90336 | 
| 53 | **13 django/db/models/sql/query.py** | 1684 | 1699| 132 | 20314 | 90336 | 
| 54 | 13 django/db/models/query.py | 1356 | 1374| 186 | 20500 | 90336 | 
| 55 | 14 django/db/models/base.py | 1087 | 1130| 404 | 20904 | 107666 | 
| 56 | **14 django/db/models/sql/query.py** | 1843 | 1892| 329 | 21233 | 107666 | 
| 57 | 14 django/db/models/sql/subqueries.py | 138 | 164| 161 | 21394 | 107666 | 
| **-> 58 <-** | **15 django/contrib/postgres/aggregates/statistics.py** | 1 | 64| 417 | 21811 | 108083 | 
| 59 | 16 django/db/models/lookups.py | 387 | 426| 329 | 22140 | 113422 | 
| 60 | **16 django/db/models/sql/query.py** | 1022 | 1049| 283 | 22423 | 113422 | 
| 61 | **16 django/db/models/sql/compiler.py** | 517 | 683| 1521 | 23944 | 113422 | 
| 62 | **16 django/db/models/sql/query.py** | 119 | 139| 173 | 24117 | 113422 | 
| 63 | **16 django/db/models/expressions.py** | 749 | 773| 243 | 24360 | 113422 | 
| 64 | 16 django/db/models/base.py | 1269 | 1300| 267 | 24627 | 113422 | 
| 65 | **16 django/db/models/sql/compiler.py** | 154 | 202| 523 | 25150 | 113422 | 
| 66 | **16 django/db/models/sql/compiler.py** | 1458 | 1490| 254 | 25404 | 113422 | 
| 67 | 17 django/contrib/admin/options.py | 1029 | 1046| 184 | 25588 | 132114 | 
| 68 | 17 django/db/models/sql/subqueries.py | 1 | 45| 309 | 25897 | 132114 | 
| 69 | 17 django/db/models/query.py | 175 | 234| 469 | 26366 | 132114 | 
| **-> 70 <-** | **17 django/db/models/aggregates.py** | 17 | 48| 297 | 26663 | 132114 | 
| 71 | 17 django/db/models/base.py | 404 | 509| 913 | 27576 | 132114 | 
| 72 | **17 django/db/models/sql/query.py** | 509 | 554| 392 | 27968 | 132114 | 
| 73 | 18 django/db/models/options.py | 1 | 35| 300 | 28268 | 139481 | 
| 74 | 18 django/db/models/fields/related_lookups.py | 1 | 26| 185 | 28453 | 139481 | 
| 75 | 18 django/db/models/query.py | 1376 | 1385| 114 | 28567 | 139481 | 
| 76 | **18 django/db/models/sql/query.py** | 634 | 659| 277 | 28844 | 139481 | 
| 77 | 19 django/contrib/admin/filters.py | 448 | 478| 233 | 29077 | 143611 | 
| 78 | **19 django/db/models/sql/query.py** | 2402 | 2457| 827 | 29904 | 143611 | 
| 79 | **19 django/db/models/expressions.py** | 820 | 836| 153 | 30057 | 143611 | 
| 80 | **19 django/db/models/sql/query.py** | 989 | 1020| 307 | 30364 | 143611 | 
| 81 | **19 django/db/models/sql/query.py** | 1071 | 1087| 152 | 30516 | 143611 | 
| 82 | 19 django/db/models/base.py | 1634 | 1721| 672 | 31188 | 143611 | 
| 83 | 19 django/db/models/query.py | 1695 | 1810| 1098 | 32286 | 143611 | 
| 84 | 19 django/db/models/base.py | 1580 | 1605| 183 | 32469 | 143611 | 
| 85 | **19 django/db/models/sql/query.py** | 1988 | 2033| 356 | 32825 | 143611 | 
| 86 | 20 django/contrib/gis/db/models/__init__.py | 1 | 19| 204 | 33029 | 143815 | 
| 87 | 20 django/db/models/base.py | 1773 | 1873| 729 | 33758 | 143815 | 
| 88 | 20 django/db/models/query.py | 265 | 285| 180 | 33938 | 143815 | 
| 89 | 21 django/contrib/admin/checks.py | 546 | 569| 230 | 34168 | 152997 | 
| 90 | 22 django/db/models/manager.py | 1 | 165| 1242 | 35410 | 154440 | 
| 91 | **22 django/db/models/sql/compiler.py** | 1017 | 1039| 207 | 35617 | 154440 | 
| 92 | **22 django/db/models/sql/compiler.py** | 1601 | 1641| 406 | 36023 | 154440 | 
| 93 | 22 django/db/models/fields/related_lookups.py | 49 | 63| 224 | 36247 | 154440 | 
| 94 | 22 django/db/models/manager.py | 168 | 204| 201 | 36448 | 154440 | 
| 95 | 23 django/db/models/indexes.py | 210 | 271| 480 | 36928 | 156763 | 
| 96 | **23 django/db/models/sql/compiler.py** | 1367 | 1426| 617 | 37545 | 156763 | 
| 97 | 23 django/db/models/base.py | 1522 | 1544| 171 | 37716 | 156763 | 
| 98 | 24 django/db/models/deletion.py | 1 | 75| 561 | 38277 | 160593 | 
| 99 | **24 django/db/models/sql/query.py** | 2035 | 2069| 295 | 38572 | 160593 | 
| 100 | **24 django/db/models/expressions.py** | 612 | 651| 290 | 38862 | 160593 | 
| 101 | 25 django/contrib/admin/views/main.py | 452 | 507| 463 | 39325 | 165060 | 
| 102 | 26 django/db/migrations/questioner.py | 233 | 246| 123 | 39448 | 167110 | 
| 103 | 26 django/db/models/query.py | 1475 | 1483| 136 | 39584 | 167110 | 
| 104 | **26 django/db/models/sql/query.py** | 243 | 289| 344 | 39928 | 167110 | 
| 105 | **26 django/db/models/sql/compiler.py** | 794 | 805| 163 | 40091 | 167110 | 
| 106 | 26 django/db/models/fields/related.py | 139 | 166| 201 | 40292 | 167110 | 
| 107 | **26 django/db/models/expressions.py** | 1077 | 1107| 281 | 40573 | 167110 | 
| 108 | 26 django/db/models/base.py | 1723 | 1771| 348 | 40921 | 167110 | 
| 109 | **26 django/db/models/sql/compiler.py** | 427 | 435| 133 | 41054 | 167110 | 
| 110 | **26 django/db/models/expressions.py** | 804 | 818| 120 | 41174 | 167110 | 
| 111 | 27 django/db/backends/base/features.py | 1 | 112| 895 | 42069 | 170117 | 
| 112 | **27 django/db/models/sql/compiler.py** | 841 | 921| 717 | 42786 | 170117 | 
| 113 | 28 django/core/management/commands/inspectdb.py | 38 | 173| 1291 | 44077 | 172750 | 
| 114 | 28 django/db/models/sql/subqueries.py | 112 | 135| 192 | 44269 | 172750 | 
| 115 | 28 django/db/models/base.py | 1132 | 1159| 286 | 44555 | 172750 | 
| 116 | 29 django/forms/models.py | 635 | 652| 167 | 44722 | 184655 | 
| 117 | **29 django/db/models/sql/query.py** | 1812 | 1841| 259 | 44981 | 184655 | 
| 118 | 29 django/db/models/lookups.py | 446 | 466| 243 | 45224 | 184655 | 
| 119 | 29 django/db/models/query.py | 1607 | 1663| 487 | 45711 | 184655 | 
| 120 | 29 django/db/models/query.py | 673 | 691| 178 | 45889 | 184655 | 
| 121 | 29 django/db/models/fields/related.py | 1 | 34| 246 | 46135 | 184655 | 
| 122 | 30 django/contrib/contenttypes/fields.py | 173 | 217| 411 | 46546 | 190113 | 
| 123 | 30 django/db/models/indexes.py | 172 | 189| 205 | 46751 | 190113 | 
| 124 | **30 django/db/models/sql/query.py** | 1385 | 1404| 245 | 46996 | 190113 | 
| **-> 125 <-** | **30 django/db/models/expressions.py** | 775 | 801| 184 | 47180 | 190113 | 
| 126 | **30 django/db/models/sql/query.py** | 65 | 117| 384 | 47564 | 190113 | 
| 127 | 30 django/db/models/base.py | 2127 | 2178| 351 | 47915 | 190113 | 


## Missing Patch Files

 * 1: django/contrib/postgres/aggregates/statistics.py
 * 2: django/db/models/aggregates.py
 * 3: django/db/models/expressions.py
 * 4: django/db/models/functions/comparison.py
 * 5: django/db/models/sql/compiler.py
 * 6: django/db/models/sql/query.py

### Hint

```
The 0 assignment on empty result set comes from ​this line. I assume we could adjust the logic to rely on getattr(col, 'empty_aggregate_value', NotImplemented) and fallback to '0' if it's missing. Makes me wonder if we'd want to rename empty_aggregate_value to empty_result_set_value instead since it would not entirely be bound to aggregation anymore. e.g. the following should exhibit the same behavior Author.objects.annotate(annotation=Coalesce(Author.objects.empty(), 42)) It also seems weird that we default to 0 as opposed to NULL which would be a more correct value for a non-coalesced annotation. Alternatively we could adjust Coalesce.as_sql to catch EmptyResultSet when it's compiling its source expressions but that's more involved as most of the logic for that currently lives in Func.as_sql. We could also use both of these approaches.
Hi! Thanks for the hints Simon, I tried a first patch where we catch empty values here ​https://github.com/django/django/pull/14770 Should we expand the solution a bit further and rename empty_aggregate_value as you suggested, and use it in the SQLCompiler too?
Hi, thanks for your PR, do we have any updates on this?
Hi, the patch is waiting on some review I believe. So once a maintainer has a bit of time available, we'll be able to move forward :)
```

## Patch

```diff
diff --git a/django/contrib/postgres/aggregates/statistics.py b/django/contrib/postgres/aggregates/statistics.py
--- a/django/contrib/postgres/aggregates/statistics.py
+++ b/django/contrib/postgres/aggregates/statistics.py
@@ -36,7 +36,7 @@ class RegrAvgY(StatAggregate):
 class RegrCount(StatAggregate):
     function = 'REGR_COUNT'
     output_field = IntegerField()
-    empty_aggregate_value = 0
+    empty_result_set_value = 0
 
 
 class RegrIntercept(StatAggregate):
diff --git a/django/db/models/aggregates.py b/django/db/models/aggregates.py
--- a/django/db/models/aggregates.py
+++ b/django/db/models/aggregates.py
@@ -21,12 +21,12 @@ class Aggregate(Func):
     filter_template = '%s FILTER (WHERE %%(filter)s)'
     window_compatible = True
     allow_distinct = False
-    empty_aggregate_value = None
+    empty_result_set_value = None
 
     def __init__(self, *expressions, distinct=False, filter=None, default=None, **extra):
         if distinct and not self.allow_distinct:
             raise TypeError("%s does not allow distinct." % self.__class__.__name__)
-        if default is not None and self.empty_aggregate_value is not None:
+        if default is not None and self.empty_result_set_value is not None:
             raise TypeError(f'{self.__class__.__name__} does not allow default.')
         self.distinct = distinct
         self.filter = filter
@@ -117,7 +117,7 @@ class Count(Aggregate):
     name = 'Count'
     output_field = IntegerField()
     allow_distinct = True
-    empty_aggregate_value = 0
+    empty_result_set_value = 0
 
     def __init__(self, expression, filter=None, **extra):
         if expression == '*':
diff --git a/django/db/models/expressions.py b/django/db/models/expressions.py
--- a/django/db/models/expressions.py
+++ b/django/db/models/expressions.py
@@ -150,10 +150,10 @@ def __ror__(self, other):
 class BaseExpression:
     """Base class for all query expressions."""
 
+    empty_result_set_value = NotImplemented
     # aggregate specific fields
     is_summary = False
     _output_field_resolved_to_none = False
-    empty_aggregate_value = NotImplemented
     # Can the expression be used in a WHERE clause?
     filterable = True
     # Can the expression can be used as a source expression in Window?
@@ -702,7 +702,13 @@ def as_sql(self, compiler, connection, function=None, template=None, arg_joiner=
         sql_parts = []
         params = []
         for arg in self.source_expressions:
-            arg_sql, arg_params = compiler.compile(arg)
+            try:
+                arg_sql, arg_params = compiler.compile(arg)
+            except EmptyResultSet:
+                empty_result_set_value = getattr(arg, 'empty_result_set_value', NotImplemented)
+                if empty_result_set_value is NotImplemented:
+                    raise
+                arg_sql, arg_params = compiler.compile(Value(empty_result_set_value))
             sql_parts.append(arg_sql)
             params.extend(arg_params)
         data = {**self.extra, **extra_context}
@@ -797,7 +803,7 @@ def _resolve_output_field(self):
             return fields.UUIDField()
 
     @property
-    def empty_aggregate_value(self):
+    def empty_result_set_value(self):
         return self.value
 
 
@@ -1114,6 +1120,7 @@ class Subquery(BaseExpression, Combinable):
     """
     template = '(%(subquery)s)'
     contains_aggregate = False
+    empty_result_set_value = None
 
     def __init__(self, queryset, output_field=None, **extra):
         # Allow the usage of both QuerySet and sql.Query objects.
diff --git a/django/db/models/functions/comparison.py b/django/db/models/functions/comparison.py
--- a/django/db/models/functions/comparison.py
+++ b/django/db/models/functions/comparison.py
@@ -66,9 +66,9 @@ def __init__(self, *expressions, **extra):
         super().__init__(*expressions, **extra)
 
     @property
-    def empty_aggregate_value(self):
+    def empty_result_set_value(self):
         for expression in self.get_source_expressions():
-            result = expression.empty_aggregate_value
+            result = expression.empty_result_set_value
             if result is NotImplemented or result is not None:
                 return result
         return None
diff --git a/django/db/models/sql/compiler.py b/django/db/models/sql/compiler.py
--- a/django/db/models/sql/compiler.py
+++ b/django/db/models/sql/compiler.py
@@ -266,8 +266,12 @@ def get_select_from_parent(klass_info):
             try:
                 sql, params = self.compile(col)
             except EmptyResultSet:
-                # Select a predicate that's always False.
-                sql, params = '0', ()
+                empty_result_set_value = getattr(col, 'empty_result_set_value', NotImplemented)
+                if empty_result_set_value is NotImplemented:
+                    # Select a predicate that's always False.
+                    sql, params = '0', ()
+                else:
+                    sql, params = self.compile(Value(empty_result_set_value))
             else:
                 sql, params = col.select_format(self, sql, params)
             ret.append((col, (sql, params), alias))
diff --git a/django/db/models/sql/query.py b/django/db/models/sql/query.py
--- a/django/db/models/sql/query.py
+++ b/django/db/models/sql/query.py
@@ -143,6 +143,7 @@ class Query(BaseExpression):
     """A single SQL query."""
 
     alias_prefix = 'T'
+    empty_result_set_value = None
     subq_aliases = frozenset([alias_prefix])
 
     compiler = 'SQLCompiler'
@@ -487,11 +488,11 @@ def get_aggregation(self, using, added_aggregate_names):
             self.default_cols = False
             self.extra = {}
 
-        empty_aggregate_result = [
-            expression.empty_aggregate_value
+        empty_set_result = [
+            expression.empty_result_set_value
             for expression in outer_query.annotation_select.values()
         ]
-        elide_empty = not any(result is NotImplemented for result in empty_aggregate_result)
+        elide_empty = not any(result is NotImplemented for result in empty_set_result)
         outer_query.clear_ordering(force=True)
         outer_query.clear_limits()
         outer_query.select_for_update = False
@@ -499,7 +500,7 @@ def get_aggregation(self, using, added_aggregate_names):
         compiler = outer_query.get_compiler(using, elide_empty=elide_empty)
         result = compiler.execute_sql(SINGLE)
         if result is None:
-            result = empty_aggregate_result
+            result = empty_set_result
 
         converters = compiler.get_converters(outer_query.annotation_select.values())
         result = next(compiler.apply_converters((result,), converters))

```

## Test Patch

```diff
diff --git a/tests/aggregation/tests.py b/tests/aggregation/tests.py
--- a/tests/aggregation/tests.py
+++ b/tests/aggregation/tests.py
@@ -1367,7 +1367,7 @@ def test_empty_result_optimization(self):
                     'books_count': 0,
                 }
             )
-        # Expression without empty_aggregate_value forces queries to be
+        # Expression without empty_result_set_value forces queries to be
         # executed even if they would return an empty result set.
         raw_books_count = Func('book', function='COUNT')
         raw_books_count.contains_aggregate = True
diff --git a/tests/annotations/tests.py b/tests/annotations/tests.py
--- a/tests/annotations/tests.py
+++ b/tests/annotations/tests.py
@@ -210,6 +210,12 @@ def test_empty_expression_annotation(self):
         self.assertEqual(len(books), Book.objects.count())
         self.assertTrue(all(not book.selected for book in books))
 
+    def test_empty_queryset_annotation(self):
+        qs = Author.objects.annotate(
+            empty=Subquery(Author.objects.values('id').none())
+        )
+        self.assertIsNone(qs.first().empty)
+
     def test_annotate_with_aggregation(self):
         books = Book.objects.annotate(is_book=Value(1), rating_count=Count('rating'))
         for book in books:
diff --git a/tests/db_functions/comparison/test_coalesce.py b/tests/db_functions/comparison/test_coalesce.py
--- a/tests/db_functions/comparison/test_coalesce.py
+++ b/tests/db_functions/comparison/test_coalesce.py
@@ -1,4 +1,4 @@
-from django.db.models import TextField
+from django.db.models import Subquery, TextField
 from django.db.models.functions import Coalesce, Lower
 from django.test import TestCase
 from django.utils import timezone
@@ -70,3 +70,14 @@ def test_ordering(self):
             authors, ['John Smith', 'Rhonda'],
             lambda a: a.name
         )
+
+    def test_empty_queryset(self):
+        Author.objects.create(name='John Smith')
+        tests = [
+            Author.objects.none(),
+            Subquery(Author.objects.none()),
+        ]
+        for empty_query in tests:
+            with self.subTest(empty_query.__class__.__name__):
+                qs = Author.objects.annotate(annotation=Coalesce(empty_query, 42))
+                self.assertEqual(qs.first().annotation, 42)

```


## Code snippets

### 1 - django/db/models/query.py:

Start line: 1134, End line: 1175

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
### 2 - django/db/models/sql/query.py:

Start line: 364, End line: 414

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
### 3 - django/db/models/sql/query.py:

Start line: 1701, End line: 1743

```python
class Query(BaseExpression):

    def resolve_ref(self, name, allow_joins=True, reuse=None, summarize=False):
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
                if name not in self.annotation_select:
                    raise FieldError(
                        "Cannot aggregate over the '%s' alias. Use annotate() "
                        "to promote it." % name
                    )
                return Ref(name, self.annotation_select[name])
            else:
                return annotation
        else:
            field_list = name.split(LOOKUP_SEP)
            annotation = self.annotations.get(field_list[0])
            if annotation is not None:
                for transform in field_list[1:]:
                    annotation = self.try_transform(annotation, transform)
                return annotation
            join_info = self.setup_joins(field_list, self.get_meta(), self.get_initial_alias(), can_reuse=reuse)
            targets, final_alias, join_list = self.trim_joins(join_info.targets, join_info.joins, join_info.path)
            if not allow_joins and len(join_list) > 1:
                raise FieldError('Joined field references are not permitted in this query')
            if len(targets) > 1:
                raise FieldError("Referencing multicolumn fields with F() objects "
                                 "isn't supported")
            # Verify that the last lookup in name is a field or a transform:
            # transform_function() raises FieldError if not.
            transform = join_info.transform_function(targets[0], final_alias)
            if reuse is not None:
                reuse.update(join_list)
            return transform
```
### 4 - django/db/models/query.py:

Start line: 842, End line: 873

```python
class QuerySet:

    def _prefetch_related_objects(self):
        # This method can only be called once the result cache has been filled.
        prefetch_related_objects(self._result_cache, *self._prefetch_related_lookups)
        self._prefetch_done = True

    def explain(self, *, format=None, **options):
        return self.query.explain(using=self.db, format=format, **options)

    ##################################################
    # PUBLIC METHODS THAT RETURN A QUERYSET SUBCLASS #
    ##################################################

    def raw(self, raw_query, params=(), translations=None, using=None):
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
### 5 - django/db/models/sql/compiler.py:

Start line: 1233, End line: 1255

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
        output_formatter = json.dumps if self.query.explain_info.format == 'json' else str
        for row in result[0]:
            if not isinstance(row, str):
                yield ' '.join(output_formatter(c) for c in row)
            else:
                yield row
```
### 6 - django/db/models/sql/query.py:

Start line: 1293, End line: 1357

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
### 7 - django/db/models/query.py:

Start line: 324, End line: 352

```python
class QuerySet:

    def __class_getitem__(cls, *args, **kwargs):
        return cls

    def __and__(self, other):
        self._check_operator_queryset(other, '&')
        self._merge_sanity_check(other)
        if isinstance(other, EmptyQuerySet):
            return other
        if isinstance(self, EmptyQuerySet):
            return self
        combined = self._chain()
        combined._merge_known_related_objects(other)
        combined.query.combine(other.query, sql.AND)
        return combined

    def __or__(self, other):
        self._check_operator_queryset(other, '|')
        self._merge_sanity_check(other)
        if isinstance(self, EmptyQuerySet):
            return other
        if isinstance(other, EmptyQuerySet):
            return self
        query = self if self.query.can_filter() else self.model._base_manager.filter(pk__in=self.values('pk'))
        combined = query._chain()
        combined._merge_known_related_objects(other)
        if not other.query.can_filter():
            other = other.model._base_manager.filter(pk__in=other.values('pk'))
        combined.query.combine(other.query, sql.OR)
        return combined
```
### 8 - django/db/models/sql/where.py:

Start line: 245, End line: 262

```python
class SubqueryConstraint:
    # Even if aggregates would be used in a subquery, the outer query isn't
    # interested about those.
    contains_aggregate = False

    def __init__(self, alias, columns, targets, query_object):
        self.alias = alias
        self.columns = columns
        self.targets = targets
        query_object.clear_ordering(clear_default=True)
        self.query_object = query_object

    def as_sql(self, compiler, connection):
        query = self.query_object
        query.set_values(self.targets)
        query_compiler = query.get_compiler(connection=connection)
        return query_compiler.as_subquery_condition(self.alias, self.columns, compiler)
```
### 9 - django/db/models/__init__.py:

Start line: 1, End line: 53

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
from django.db.models.fields.json import JSONField
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
    'FileField', 'ImageField', 'JSONField', 'OrderWrt', 'Lookup', 'Transform',
    'Manager', 'Prefetch', 'Q', 'QuerySet', 'prefetch_related_objects',
    'DEFERRED', 'Model', 'FilteredRelation',
    'ForeignKey', 'ForeignObject', 'OneToOneField', 'ManyToManyField',
    'ForeignObjectRel', 'ManyToOneRel', 'ManyToManyRel', 'OneToOneRel',
]
```
### 10 - django/db/models/sql/query.py:

Start line: 2166, End line: 2213

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
            selected = frozenset(field_names + extra_names + annotation_names)
        else:
            field_names = [f.attname for f in self.model._meta.concrete_fields]
            selected = frozenset(field_names)
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
                if isinstance(expr, Ref) and expr.refs not in selected:
                    expr = self.annotations[expr.refs]
                group_by.append(expr)
            self.group_by = tuple(group_by)

        self.values_select = tuple(field_names)
        self.add_fields(field_names, True)
```
### 11 - django/db/models/sql/query.py:

Start line: 906, End line: 928

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
### 12 - django/db/models/sql/query.py:

Start line: 1427, End line: 1454

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
### 13 - django/db/models/expressions.py:

Start line: 1165, End line: 1197

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
### 14 - django/db/models/aggregates.py:

Start line: 50, End line: 68

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
        if (default := c.default) is None:
            return c
        if hasattr(default, 'resolve_expression'):
            default = default.resolve_expression(query, allow_joins, reuse, summarize)
        c.default = None  # Reset the default argument before wrapping.
        return Coalesce(c, default, output_field=c._output_field_or_none)
```
### 15 - django/db/models/sql/query.py:

Start line: 1472, End line: 1557

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
                    if LOOKUP_SEP in filtered_relation.relation_name:
                        parts = filtered_relation.relation_name.split(LOOKUP_SEP)
                        filtered_relation_path, field, _, _ = self.names_to_path(
                            parts, opts, allow_many, fail_on_missing,
                        )
                        path.extend(filtered_relation_path[:-1])
                    else:
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
### 16 - django/db/models/expressions.py:

Start line: 1110, End line: 1162

```python
class Subquery(BaseExpression, Combinable):
    """
    An explicit subquery. It may contain OuterRef() references to the outer
    query which will be resolved when it is applied to that query.
    """
    template = '(%(subquery)s)'
    contains_aggregate = False

    def __init__(self, queryset, output_field=None, **extra):
        # Allow the usage of both QuerySet and sql.Query objects.
        self.query = getattr(queryset, 'query', queryset)
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
        if alias:
            return [Ref(alias, self)]
        external_cols = self.get_external_cols()
        if any(col.possibly_multivalued for col in external_cols):
            return [self]
        return external_cols
```
### 18 - django/db/models/sql/compiler.py:

Start line: 204, End line: 274

```python
class SQLCompiler:
    ordering_parts =
    # ... other code

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
    # ... other code
```
### 19 - django/db/models/sql/compiler.py:

Start line: 23, End line: 51

```python
class SQLCompiler:
    # Multiline ordering SQL clause may appear from RawSQL.
    ordering_parts = _lazy_re_compile(
        r'^(.*)\s(?:ASC|DESC).*',
        re.MULTILINE | re.DOTALL,
    )

    def __init__(self, query, connection, using, elide_empty=True):
        self.query = query
        self.connection = connection
        self.using = using
        # Some queries, e.g. coalesced aggregation, need to be executed even if
        # they would return an empty result set.
        self.elide_empty = elide_empty
        self.quote_cache = {'*': '*'}
        # The select, klass_info, and annotations are needed by QuerySet.iterator()
        # these are set as a side-effect of executing the query. Note that we calculate
        # separately a list of extra select columns needed for grammatical correctness
        # of the query, but these columns are not included in self.select.
        self.select = None
        self.annotation_col_map = None
        self.klass_info = None
        self._meta_ordering = None

    def setup_query(self):
        if all(self.query.alias_refcount[a] == 0 for a in self.query.alias_map):
            self.query.get_initial_alias()
        self.select, self.klass_info, self.annotation_col_map = self.get_select()
        self.col_count = len(self.select)
    # ... other code
```
### 20 - django/db/models/sql/query.py:

Start line: 710, End line: 745

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
### 21 - django/db/models/sql/query.py:

Start line: 1633, End line: 1654

```python
class Query(BaseExpression):

    def setup_joins(self, names, opts, alias, can_reuse=None, allow_many=True):
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
            reuse = can_reuse if join.m2m else None
            alias = self.join(connection, reuse=reuse)
            joins.append(alias)
            if filtered_relation:
                filtered_relation.path = joins[:]
        return JoinInfo(final_field, targets, opts, joins, path, final_transformer)
```
### 22 - django/db/models/sql/query.py:

Start line: 1745, End line: 1810

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
        query = Query(self.model)
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
### 23 - django/db/models/aggregates.py:

Start line: 70, End line: 106

```python
class Aggregate(Func):

    @property
    def default_alias(self):
        expressions = self.get_source_expressions()
        if len(expressions) == 1 and hasattr(expressions[0], 'name'):
            return '%s__%s' % (expressions[0].name, self.name.lower())
        raise TypeError("Complex expressions require an alias")

    def get_group_by_cols(self, alias=None):
        return []

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
### 25 - django/db/models/sql/query.py:

Start line: 142, End line: 241

```python
class Query(BaseExpression):
    """A single SQL query."""

    alias_prefix = 'T'
    subq_aliases = frozenset([alias_prefix])

    compiler = 'SQLCompiler'

    def __init__(self, model, alias_cols=True):
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
        self.where = WhereNode()
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

        self.explain_info = None

    @property
    def output_field(self):
        if len(self.select) == 1:
            select = self.select[0]
            return getattr(select, 'target', None) or select.field
        elif len(self.annotation_select) == 1:
            return next(iter(self.annotation_select.values())).output_field
```
### 26 - django/db/models/sql/compiler.py:

Start line: 67, End line: 152

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
        if not self._meta_ordering:
            for expr, (sql, params, is_ref) in order_by:
                # Skip references to the SELECT clause, as all expressions in
                # the SELECT clause are already part of the GROUP BY.
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
### 33 - django/db/models/expressions.py:

Start line: 839, End line: 873

```python
class Col(Expression):

    contains_column_references = True
    possibly_multivalued = False

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
### 34 - django/db/models/sql/query.py:

Start line: 1118, End line: 1150

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
### 35 - django/db/models/sql/compiler.py:

Start line: 923, End line: 1015

```python
class SQLCompiler:
    ordering_parts =

    def get_related_selections(self, select, opts=None, root_alias=None, cur_depth=1,
                               requested=None, restricted=None):
        # ... other code

        if restricted:
            related_fields = [
                (o.field, o.related_model)
                for o in opts.related_objects
                if o.field.unique and not o.many_to_many
            ]
            for f, model in related_fields:
                if not select_related_descend(f, restricted, requested,
                                              only_load.get(model), reverse=True):
                    continue

                related_field_name = f.related_query_name()
                fields_found.add(related_field_name)

                join_info = self.query.setup_joins([related_field_name], opts, root_alias)
                alias = join_info.joins[-1]
                from_parent = issubclass(model, opts.model) and model is not opts.model
                klass_info = {
                    'model': model,
                    'field': f,
                    'reverse': True,
                    'local_setter': f.remote_field.set_cached_value,
                    'remote_setter': f.set_cached_value,
                    'from_parent': from_parent,
                }
                related_klass_infos.append(klass_info)
                select_fields = []
                columns = self.get_default_columns(
                    start_alias=alias, opts=model._meta, from_parent=opts.model)
                for col in columns:
                    select_fields.append(len(select))
                    select.append((col, None))
                klass_info['select_fields'] = select_fields
                next = requested.get(f.related_query_name(), {})
                next_klass_infos = self.get_related_selections(
                    select, model._meta, alias, cur_depth + 1,
                    next, restricted)
                get_related_klass_infos(klass_info, next_klass_infos)

            def local_setter(obj, from_obj):
                # Set a reverse fk object when relation is non-empty.
                if from_obj:
                    f.remote_field.set_cached_value(from_obj, obj)

            def remote_setter(name, obj, from_obj):
                setattr(from_obj, name, obj)

            for name in list(requested):
                # Filtered relations work only on the topmost level.
                if cur_depth > 1:
                    break
                if name in self.query._filtered_relations:
                    fields_found.add(name)
                    f, _, join_opts, joins, _, _ = self.query.setup_joins([name], opts, root_alias)
                    model = join_opts.model
                    alias = joins[-1]
                    from_parent = issubclass(model, opts.model) and model is not opts.model
                    klass_info = {
                        'model': model,
                        'field': f,
                        'reverse': True,
                        'local_setter': local_setter,
                        'remote_setter': partial(remote_setter, name),
                        'from_parent': from_parent,
                    }
                    related_klass_infos.append(klass_info)
                    select_fields = []
                    columns = self.get_default_columns(
                        start_alias=alias, opts=model._meta,
                        from_parent=opts.model,
                    )
                    for col in columns:
                        select_fields.append(len(select))
                        select.append((col, None))
                    klass_info['select_fields'] = select_fields
                    next_requested = requested.get(name, {})
                    next_klass_infos = self.get_related_selections(
                        select, opts=model._meta, root_alias=alias,
                        cur_depth=cur_depth + 1, requested=next_requested,
                        restricted=restricted,
                    )
                    get_related_klass_infos(klass_info, next_klass_infos)
            fields_not_found = set(requested).difference(fields_found)
            if fields_not_found:
                invalid_fields = ("'%s'" % s for s in fields_not_found)
                raise FieldError(
                    'Invalid field name(s) given in select_related: %s. '
                    'Choices are: %s' % (
                        ', '.join(invalid_fields),
                        ', '.join(_get_field_choices()) or '(none)',
                    )
                )
        return related_klass_infos
    # ... other code
```
### 36 - django/db/models/sql/query.py:

Start line: 1, End line: 62

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
import sys
from collections import Counter, namedtuple
from collections.abc import Iterator, Mapping
from itertools import chain, count, product
from string import ascii_uppercase

from django.core.exceptions import FieldDoesNotExist, FieldError
from django.db import DEFAULT_DB_ALIAS, NotSupportedError, connections
from django.db.models.aggregates import Count
from django.db.models.constants import LOOKUP_SEP
from django.db.models.expressions import (
    BaseExpression, Col, Exists, F, OuterRef, Ref, ResolvedOuterRef,
)
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
### 37 - django/db/models/sql/compiler.py:

Start line: 1076, End line: 1116

```python
class SQLCompiler:
    ordering_parts =

    def get_select_for_update_of_arguments(self):
        # ... other code
        result = []
        invalid_names = []
        for name in self.query.select_for_update_of:
            klass_info = self.klass_info
            if name == 'self':
                col = _get_first_selected_col_from_model(klass_info)
            else:
                for part in name.split(LOOKUP_SEP):
                    klass_infos = (
                        *klass_info.get('related_klass_infos', []),
                        *_get_parent_klass_info(klass_info),
                    )
                    for related_klass_info in klass_infos:
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
                col = _get_first_selected_col_from_model(klass_info)
            if col is not None:
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
    # ... other code
```
### 40 - django/db/models/sql/compiler.py:

Start line: 276, End line: 371

```python
class SQLCompiler:
    ordering_parts =
    # ... other code

    def _order_by_pairs(self):
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
            default_order, _ = ORDER_DIR['ASC']
        else:
            default_order, _ = ORDER_DIR['DESC']

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
                yield field, False
                continue
            if field == '?':  # random
                yield OrderBy(Random()), False
                continue

            col, order = get_order_dir(field, default_order)
            descending = order == 'DESC'

            if col in self.query.annotation_select:
                # Reference to expression in SELECT clause
                yield (
                    OrderBy(
                        Ref(col, self.query.annotation_select[col]),
                        descending=descending,
                    ),
                    True,
                )
                continue
            if col in self.query.annotations:
                # References to an expression which is masked out of the SELECT
                # clause.
                if self.query.combinator and self.select:
                    # Don't use the resolved annotation because other
                    # combinated queries might define it differently.
                    expr = F(col)
                else:
                    expr = self.query.annotations[col]
                    if isinstance(expr, Value):
                        # output_field must be resolved for constants.
                        expr = Cast(expr, expr.output_field)
                yield OrderBy(expr, descending=descending), False
                continue

            if '.' in field:
                # This came in through an extra(order_by=...) addition. Pass it
                # on verbatim.
                table, col = col.split('.', 1)
                yield (
                    OrderBy(
                        RawSQL('%s.%s' % (self.quote_name_unless_alias(table), col), []),
                        descending=descending,
                    ),
                    False,
                )
                continue

            if self.query.extra and col in self.query.extra:
                if col in self.query.extra_select:
                    yield (
                        OrderBy(Ref(col, RawSQL(*self.query.extra[col])), descending=descending),
                        True,
                    )
                else:
                    yield (
                        OrderBy(RawSQL(*self.query.extra[col]), descending=descending),
                        False,
                    )
            else:
                if self.query.combinator and self.select:
                    # Don't use the first model's field because other
                    # combinated queries might define it differently.
                    yield OrderBy(F(col), descending=descending), False
                else:
                    # 'col' is of the form 'field' or 'field1__field2' or
                    # '-field1__field2__field', etc.
                    yield from self.find_ordering_name(
                        field, self.query.get_meta(), default_order=default_order,
                    )
    # ... other code
```
### 43 - django/db/models/expressions.py:

Start line: 962, End line: 1026

```python
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
### 46 - django/db/models/sql/compiler.py:

Start line: 462, End line: 515

```python
class SQLCompiler:
    ordering_parts =
    # ... other code

    def get_combinator_sql(self, combinator, all):
        features = self.connection.features
        compilers = [
            query.get_compiler(self.using, self.connection, self.elide_empty)
            for query in self.query.combined_queries if not query.is_empty()
        ]
        if not features.supports_slicing_ordering_in_compound:
            for query, compiler in zip(self.query.combined_queries, compilers):
                if query.low_mark or query.high_mark:
                    raise DatabaseError('LIMIT/OFFSET not allowed in subqueries of compound statements.')
                if compiler.get_order_by():
                    raise DatabaseError('ORDER BY not allowed in subqueries of compound statements.')
        parts = ()
        for compiler in compilers:
            try:
                # If the columns list is limited, then all combined queries
                # must have the same columns list. Set the selects defined on
                # the query on all combined queries, if not already set.
                if not compiler.query.values_select and self.query.values_select:
                    compiler.query = compiler.query.clone()
                    compiler.query.set_values((
                        *self.query.extra_select,
                        *self.query.values_select,
                        *self.query.annotation_select,
                    ))
                part_sql, part_args = compiler.as_sql()
                if compiler.query.combinator:
                    # Wrap in a subquery if wrapping in parentheses isn't
                    # supported.
                    if not features.supports_parentheses_in_compound:
                        part_sql = 'SELECT * FROM ({})'.format(part_sql)
                    # Add parentheses when combining with compound query if not
                    # already added for all compound queries.
                    elif not features.supports_slicing_ordering_in_compound:
                        part_sql = '({})'.format(part_sql)
                parts += ((part_sql, part_args),)
            except EmptyResultSet:
                # Omit the empty queryset with UNION and with DIFFERENCE if the
                # first queryset is nonempty.
                if combinator == 'union' or (combinator == 'difference' and parts):
                    continue
                raise
        if not parts:
            raise EmptyResultSet
        combinator_sql = self.connection.ops.set_operators[combinator]
        if all and combinator == 'union':
            combinator_sql += ' ALL'
        braces = '({})' if features.supports_slicing_ordering_in_compound else '{}'
        sql_parts, args_parts = zip(*((braces.format(sql), args) for sql, args in parts))
        result = [' {} '.format(combinator_sql).join(sql_parts)]
        params = []
        for part in args_parts:
            params.extend(part)
        return result, params
    # ... other code
```
### 48 - django/db/models/sql/query.py:

Start line: 1051, End line: 1069

```python
class Query(BaseExpression):

    def get_external_cols(self):
        exprs = chain(self.annotations.values(), self.where.children)
        return [
            col for col in self._gen_cols(exprs, include_external=True)
            if col.alias in self.external_aliases
        ]

    def as_sql(self, compiler, connection):
        # Some backends (e.g. Oracle) raise an error when a subquery contains
        # unnecessary ORDER BY clause.
        if (
            self.subquery and
            not connection.features.ignores_unnecessary_order_by_in_subqueries
        ):
            self.clear_ordering(force=False)
        sql, params = self.get_compiler(connection=connection).as_sql()
        if self.subquery:
            sql = '(%s)' % sql
        return sql, params
```
### 50 - django/db/models/sql/query.py:

Start line: 1359, End line: 1383

```python
class Query(BaseExpression):

    def add_filter(self, filter_lhs, filter_rhs):
        self.add_q(Q((filter_lhs, filter_rhs)))

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

    def clear_where(self):
        self.where = WhereNode()
```
### 51 - django/db/models/sql/query.py:

Start line: 2215, End line: 2247

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
### 52 - django/db/models/sql/compiler.py:

Start line: 1, End line: 20

```python
import collections
import json
import re
from functools import partial
from itertools import chain

from django.core.exceptions import EmptyResultSet, FieldError
from django.db import DatabaseError, NotSupportedError
from django.db.models.constants import LOOKUP_SEP
from django.db.models.expressions import F, OrderBy, RawSQL, Ref, Value
from django.db.models.functions import Cast, Random
from django.db.models.query_utils import select_related_descend
from django.db.models.sql.constants import (
    CURSOR, GET_ITERATOR_CHUNK_SIZE, MULTI, NO_RESULTS, ORDER_DIR, SINGLE,
)
from django.db.models.sql.query import Query, get_order_dir
from django.db.transaction import TransactionManagementError
from django.utils.functional import cached_property
from django.utils.hashable import make_hashable
from django.utils.regex_helper import _lazy_re_compile
```
### 53 - django/db/models/sql/query.py:

Start line: 1684, End line: 1699

```python
class Query(BaseExpression):

    @classmethod
    def _gen_cols(cls, exprs, include_external=False):
        for expr in exprs:
            if isinstance(expr, Col):
                yield expr
            elif include_external and callable(getattr(expr, 'get_external_cols', None)):
                yield from expr.get_external_cols()
            elif hasattr(expr, 'get_source_expressions'):
                yield from cls._gen_cols(
                    expr.get_source_expressions(),
                    include_external=include_external,
                )

    @classmethod
    def _gen_col_aliases(cls, exprs):
        yield from (expr.alias for expr in cls._gen_cols(exprs))
```
### 56 - django/db/models/sql/query.py:

Start line: 1843, End line: 1892

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
### 58 - django/contrib/postgres/aggregates/statistics.py:

Start line: 1, End line: 64

```python
from django.db.models import Aggregate, FloatField, IntegerField

__all__ = [
    'CovarPop', 'Corr', 'RegrAvgX', 'RegrAvgY', 'RegrCount', 'RegrIntercept',
    'RegrR2', 'RegrSlope', 'RegrSXX', 'RegrSXY', 'RegrSYY', 'StatAggregate',
]


class StatAggregate(Aggregate):
    output_field = FloatField()

    def __init__(self, y, x, output_field=None, filter=None, default=None):
        if not x or not y:
            raise ValueError('Both y and x must be provided.')
        super().__init__(y, x, output_field=output_field, filter=filter, default=default)


class Corr(StatAggregate):
    function = 'CORR'


class CovarPop(StatAggregate):
    def __init__(self, y, x, sample=False, filter=None, default=None):
        self.function = 'COVAR_SAMP' if sample else 'COVAR_POP'
        super().__init__(y, x, filter=filter, default=default)


class RegrAvgX(StatAggregate):
    function = 'REGR_AVGX'


class RegrAvgY(StatAggregate):
    function = 'REGR_AVGY'


class RegrCount(StatAggregate):
    function = 'REGR_COUNT'
    output_field = IntegerField()
    empty_aggregate_value = 0


class RegrIntercept(StatAggregate):
    function = 'REGR_INTERCEPT'


class RegrR2(StatAggregate):
    function = 'REGR_R2'


class RegrSlope(StatAggregate):
    function = 'REGR_SLOPE'


class RegrSXX(StatAggregate):
    function = 'REGR_SXX'


class RegrSXY(StatAggregate):
    function = 'REGR_SXY'


class RegrSYY(StatAggregate):
    function = 'REGR_SYY'
```
### 60 - django/db/models/sql/query.py:

Start line: 1022, End line: 1049

```python
class Query(BaseExpression):

    def add_annotation(self, annotation, alias, is_summary=False, select=True):
        """Add a single annotation expression to the Query."""
        annotation = annotation.resolve_expression(self, allow_joins=True, reuse=None,
                                                   summarize=is_summary)
        if select:
            self.append_annotation_mask([alias])
        else:
            self.set_annotation_mask(set(self.annotation_select).difference({alias}))
        self.annotations[alias] = annotation

    def resolve_expression(self, query, *args, **kwargs):
        clone = self.clone()
        # Subqueries need to use a different set of aliases than the outer query.
        clone.bump_prefix(query)
        clone.subquery = True
        clone.where.resolve_expression(query, *args, **kwargs)
        for key, value in clone.annotations.items():
            resolved = value.resolve_expression(query, *args, **kwargs)
            if hasattr(resolved, 'external_aliases'):
                resolved.external_aliases.update(clone.external_aliases)
            clone.annotations[key] = resolved
        # Outer query's aliases are considered external.
        for alias, table in query.alias_map.items():
            clone.external_aliases[alias] = (
                (isinstance(table, Join) and table.join_field.related_model._meta.db_table != alias) or
                (isinstance(table, BaseTable) and table.table_name != table.table_alias)
            )
        return clone
```
### 61 - django/db/models/sql/compiler.py:

Start line: 517, End line: 683

```python
class SQLCompiler:
    ordering_parts =
    # ... other code

    def as_sql(self, with_limits=True, with_col_aliases=False):
        """
        Create the SQL for this query. Return the SQL string and list of
        parameters.

        If 'with_limits' is False, any limit/offset information is not included
        in the query.
        """
        refcounts_before = self.query.alias_refcount.copy()
        try:
            extra_select, order_by, group_by = self.pre_sql_setup()
            for_update_part = None
            # Is a LIMIT/OFFSET clause needed?
            with_limit_offset = with_limits and (self.query.high_mark is not None or self.query.low_mark)
            combinator = self.query.combinator
            features = self.connection.features
            if combinator:
                if not getattr(features, 'supports_select_{}'.format(combinator)):
                    raise NotSupportedError('{} is not supported on this database backend.'.format(combinator))
                result, params = self.get_combinator_sql(combinator, self.query.combinator_all)
            else:
                distinct_fields, distinct_params = self.get_distinct()
                # This must come after 'select', 'ordering', and 'distinct'
                # (see docstring of get_from_clause() for details).
                from_, f_params = self.get_from_clause()
                try:
                    where, w_params = self.compile(self.where) if self.where is not None else ('', [])
                except EmptyResultSet:
                    if self.elide_empty:
                        raise
                    # Use a predicate that's always False.
                    where, w_params = '0 = 1', []
                having, h_params = self.compile(self.having) if self.having is not None else ("", [])
                result = ['SELECT']
                params = []

                if self.query.distinct:
                    distinct_result, distinct_params = self.connection.ops.distinct_sql(
                        distinct_fields,
                        distinct_params,
                    )
                    result += distinct_result
                    params += distinct_params

                out_cols = []
                col_idx = 1
                for _, (s_sql, s_params), alias in self.select + extra_select:
                    if alias:
                        s_sql = '%s AS %s' % (s_sql, self.connection.ops.quote_name(alias))
                    elif with_col_aliases:
                        s_sql = '%s AS %s' % (
                            s_sql,
                            self.connection.ops.quote_name('col%d' % col_idx),
                        )
                        col_idx += 1
                    params.extend(s_params)
                    out_cols.append(s_sql)

                result += [', '.join(out_cols), 'FROM', *from_]
                params.extend(f_params)

                if self.query.select_for_update and self.connection.features.has_select_for_update:
                    if self.connection.get_autocommit():
                        raise TransactionManagementError('select_for_update cannot be used outside of a transaction.')

                    if with_limit_offset and not self.connection.features.supports_select_for_update_with_limit:
                        raise NotSupportedError(
                            'LIMIT/OFFSET is not supported with '
                            'select_for_update on this database backend.'
                        )
                    nowait = self.query.select_for_update_nowait
                    skip_locked = self.query.select_for_update_skip_locked
                    of = self.query.select_for_update_of
                    no_key = self.query.select_for_no_key_update
                    # If it's a NOWAIT/SKIP LOCKED/OF/NO KEY query but the
                    # backend doesn't support it, raise NotSupportedError to
                    # prevent a possible deadlock.
                    if nowait and not self.connection.features.has_select_for_update_nowait:
                        raise NotSupportedError('NOWAIT is not supported on this database backend.')
                    elif skip_locked and not self.connection.features.has_select_for_update_skip_locked:
                        raise NotSupportedError('SKIP LOCKED is not supported on this database backend.')
                    elif of and not self.connection.features.has_select_for_update_of:
                        raise NotSupportedError('FOR UPDATE OF is not supported on this database backend.')
                    elif no_key and not self.connection.features.has_select_for_no_key_update:
                        raise NotSupportedError(
                            'FOR NO KEY UPDATE is not supported on this '
                            'database backend.'
                        )
                    for_update_part = self.connection.ops.for_update_sql(
                        nowait=nowait,
                        skip_locked=skip_locked,
                        of=self.get_select_for_update_of_arguments(),
                        no_key=no_key,
                    )

                if for_update_part and self.connection.features.for_update_after_from:
                    result.append(for_update_part)

                if where:
                    result.append('WHERE %s' % where)
                    params.extend(w_params)

                grouping = []
                for g_sql, g_params in group_by:
                    grouping.append(g_sql)
                    params.extend(g_params)
                if grouping:
                    if distinct_fields:
                        raise NotImplementedError('annotate() + distinct(fields) is not implemented.')
                    order_by = order_by or self.connection.ops.force_no_ordering()
                    result.append('GROUP BY %s' % ', '.join(grouping))
                    if self._meta_ordering:
                        order_by = None
                if having:
                    result.append('HAVING %s' % having)
                    params.extend(h_params)

            if self.query.explain_info:
                result.insert(0, self.connection.ops.explain_query_prefix(
                    self.query.explain_info.format,
                    **self.query.explain_info.options
                ))

            if order_by:
                ordering = []
                for _, (o_sql, o_params, _) in order_by:
                    ordering.append(o_sql)
                    params.extend(o_params)
                result.append('ORDER BY %s' % ', '.join(ordering))

            if with_limit_offset:
                result.append(self.connection.ops.limit_offset_sql(self.query.low_mark, self.query.high_mark))

            if for_update_part and not self.connection.features.for_update_after_from:
                result.append(for_update_part)

            if self.query.subquery and extra_select:
                # If the query is used as a subquery, the extra selects would
                # result in more columns than the left-hand side expression is
                # expecting. This can happen when a subquery uses a combination
                # of order_by() and distinct(), forcing the ordering expressions
                # to be selected as well. Wrap the query in another subquery
                # to exclude extraneous selects.
                sub_selects = []
                sub_params = []
                for index, (select, _, alias) in enumerate(self.select, start=1):
                    if not alias and with_col_aliases:
                        alias = 'col%d' % index
                    if alias:
                        sub_selects.append("%s.%s" % (
                            self.connection.ops.quote_name('subquery'),
                            self.connection.ops.quote_name(alias),
                        ))
                    else:
                        select_clone = select.relabeled_clone({select.alias: 'subquery'})
                        subselect, subparams = select_clone.as_sql(self, self.connection)
                        sub_selects.append(subselect)
                        sub_params.extend(subparams)
                return 'SELECT %s FROM (%s) subquery' % (
                    ', '.join(sub_selects),
                    ' '.join(result),
                ), tuple(sub_params + params)

            return ' '.join(result), tuple(params)
        finally:
            # Finally do cleanup - get rid of the joins we created above.
            self.query.reset_refcounts(refcounts_before)
    # ... other code
```
### 62 - django/db/models/sql/query.py:

Start line: 119, End line: 139

```python
class RawQuery:

    def _execute_query(self):
        connection = connections[self.using]

        # Adapt parameters to the database, as much as possible considering
        # that the target type isn't known. See #17755.
        params_type = self.params_type
        adapter = connection.ops.adapt_unknown_value
        if params_type is tuple:
            params = tuple(adapter(val) for val in self.params)
        elif params_type is dict:
            params = {key: adapter(val) for key, val in self.params.items()}
        elif params_type is None:
            params = None
        else:
            raise RuntimeError("Unexpected params type: %s" % params_type)

        self.cursor = connection.cursor()
        self.cursor.execute(self.sql, params)


ExplainInfo = namedtuple('ExplainInfo', ('format', 'options'))
```
### 63 - django/db/models/expressions.py:

Start line: 749, End line: 773

```python
class Value(SQLiteNumericMixin, Expression):

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
### 65 - django/db/models/sql/compiler.py:

Start line: 154, End line: 202

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
### 66 - django/db/models/sql/compiler.py:

Start line: 1458, End line: 1490

```python
class SQLDeleteCompiler(SQLCompiler):
    @cached_property
    def single_alias(self):
        # Ensure base table is in aliases.
        self.query.get_initial_alias()
        return sum(self.query.alias_refcount[t] > 0 for t in self.query.alias_map) == 1

    @classmethod
    def _expr_refs_base_model(cls, expr, base_model):
        if isinstance(expr, Query):
            return expr.model == base_model
        if not hasattr(expr, 'get_source_expressions'):
            return False
        return any(
            cls._expr_refs_base_model(source_expr, base_model)
            for source_expr in expr.get_source_expressions()
        )

    @cached_property
    def contains_self_reference_subquery(self):
        return any(
            self._expr_refs_base_model(expr, self.query.model)
            for expr in chain(self.query.annotations.values(), self.query.where.children)
        )

    def _as_sql(self, query):
        result = [
            'DELETE FROM %s' % self.quote_name_unless_alias(query.base_table)
        ]
        where, params = self.compile(query.where)
        if where:
            result.append('WHERE %s' % where)
        return ' '.join(result), tuple(params)
```
### 70 - django/db/models/aggregates.py:

Start line: 17, End line: 48

```python
class Aggregate(Func):
    template = '%(function)s(%(distinct)s%(expressions)s)'
    contains_aggregate = True
    name = None
    filter_template = '%s FILTER (WHERE %%(filter)s)'
    window_compatible = True
    allow_distinct = False
    empty_aggregate_value = None

    def __init__(self, *expressions, distinct=False, filter=None, default=None, **extra):
        if distinct and not self.allow_distinct:
            raise TypeError("%s does not allow distinct." % self.__class__.__name__)
        if default is not None and self.empty_aggregate_value is not None:
            raise TypeError(f'{self.__class__.__name__} does not allow default.')
        self.distinct = distinct
        self.filter = filter
        self.default = default
        super().__init__(*expressions, **extra)

    def get_source_fields(self):
        # Don't return the filter expression since it's not a source field.
        return [e._output_field_or_none for e in super().get_source_expressions()]

    def get_source_expressions(self):
        source_expressions = super().get_source_expressions()
        if self.filter:
            return source_expressions + [self.filter]
        return source_expressions

    def set_source_expressions(self, exprs):
        self.filter = self.filter and exprs.pop()
        return super().set_source_expressions(exprs)
```
### 72 - django/db/models/sql/query.py:

Start line: 509, End line: 554

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

    def exists(self, using, limit=True):
        q = self.clone()
        if not q.distinct:
            if q.group_by is True:
                q.add_fields((f.attname for f in self.model._meta.concrete_fields), False)
                # Disable GROUP BY aliases to avoid orphaning references to the
                # SELECT clause which is about to be cleared.
                q.set_group_by(allow_aliases=False)
            q.clear_select_clause()
        if q.combined_queries and q.combinator == 'union':
            limit_combined = connections[using].features.supports_slicing_ordering_in_compound
            q.combined_queries = tuple(
                combined_query.exists(using, limit=limit_combined)
                for combined_query in q.combined_queries
            )
        q.clear_ordering(force=True)
        if limit:
            q.set_limits(high=1)
        q.add_extra({'a': 1}, None, None, None, None, None)
        q.set_extra_mask(['a'])
        return q

    def has_results(self, using):
        q = self.exists(using)
        compiler = q.get_compiler(using=using)
        return compiler.has_results()

    def explain(self, using, format=None, **options):
        q = self.clone()
        q.explain_info = ExplainInfo(format, options)
        compiler = q.get_compiler(using=using)
        return '\n'.join(compiler.explain_query())
```
### 76 - django/db/models/sql/query.py:

Start line: 634, End line: 659

```python
class Query(BaseExpression):

    def combine(self, rhs, connector):

        # Selection columns and extra extensions are those provided by 'rhs'.
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
### 78 - django/db/models/sql/query.py:

Start line: 2402, End line: 2457

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
### 79 - django/db/models/expressions.py:

Start line: 820, End line: 836

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
```
### 80 - django/db/models/sql/query.py:

Start line: 989, End line: 1020

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
### 81 - django/db/models/sql/query.py:

Start line: 1071, End line: 1087

```python
class Query(BaseExpression):

    def resolve_lookup_value(self, value, can_reuse, allow_joins):
        if hasattr(value, 'resolve_expression'):
            value = value.resolve_expression(
                self, reuse=can_reuse, allow_joins=allow_joins,
            )
        elif isinstance(value, (list, tuple)):
            # The items of the iterable may be expressions and therefore need
            # to be resolved independently.
            values = (
                self.resolve_lookup_value(sub_value, can_reuse, allow_joins)
                for sub_value in value
            )
            type_ = type(value)
            if hasattr(type_, '_make'):  # namedtuple
                return type_(*values)
            return type_(values)
        return value
```
### 85 - django/db/models/sql/query.py:

Start line: 1988, End line: 2033

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
                if not allow_aliases or alias in column_names:
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
### 91 - django/db/models/sql/compiler.py:

Start line: 1017, End line: 1039

```python
class SQLCompiler:
    ordering_parts =
    # ... other code

    def get_select_for_update_of_arguments(self):
        """
        Return a quoted list of arguments for the SELECT FOR UPDATE OF part of
        the query.
        """
        def _get_parent_klass_info(klass_info):
            concrete_model = klass_info['model']._meta.concrete_model
            for parent_model, parent_link in concrete_model._meta.parents.items():
                parent_list = parent_model._meta.get_parent_list()
                yield {
                    'model': parent_model,
                    'field': parent_link,
                    'reverse': False,
                    'select_fields': [
                        select_index
                        for select_index in klass_info['select_fields']
                        # Selected columns from a model or its parents.
                        if (
                            self.select[select_index][0].target.model == parent_model or
                            self.select[select_index][0].target.model in parent_list
                        )
                    ],
                }
        # ... other code
    # ... other code
```
### 92 - django/db/models/sql/compiler.py:

Start line: 1601, End line: 1641

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
        query.clear_ordering(force=True)
        query.extra = {}
        query.select = []
        query.add_fields([query.get_meta().pk.name])
        super().pre_sql_setup()

        must_pre_select = count > 1 and not self.connection.features.update_can_self_select

        # Now we adjust the current query: reset the where clause and get rid
        # of all the tables we don't need (since they're in the sub-select).
        self.query.clear_where()
        if self.query.related_updates or must_pre_select:
            # Either we're using the idents in multiple update queries (so
            # don't want them to change), or the db backend doesn't support
            # selecting from the updating table (e.g. MySQL).
            idents = []
            for rows in query.get_compiler(self.using).execute_sql(MULTI):
                idents.extend(r[0] for r in rows)
            self.query.add_filter('pk__in', idents)
            self.query.related_ids = idents
        else:
            # The fast path. Filters and updates in one query.
            self.query.add_filter('pk__in', query)
        self.query.reset_refcounts(refcounts_before)
```
### 96 - django/db/models/sql/compiler.py:

Start line: 1367, End line: 1426

```python
class SQLInsertCompiler(SQLCompiler):

    def as_sql(self):
        # We don't need quote_name_unless_alias() here, since these are all
        # going to be column names (so we can avoid the extra overhead).
        qn = self.connection.ops.quote_name
        opts = self.query.get_meta()
        insert_statement = self.connection.ops.insert_statement(ignore_conflicts=self.query.ignore_conflicts)
        result = ['%s %s' % (insert_statement, qn(opts.db_table))]
        fields = self.query.fields or [opts.pk]
        result.append('(%s)' % ', '.join(qn(f.column) for f in fields))

        if self.query.fields:
            value_rows = [
                [self.prepare_value(field, self.pre_save_val(field, obj)) for field in fields]
                for obj in self.query.objs
            ]
        else:
            # An empty object.
            value_rows = [[self.connection.ops.pk_default_value()] for _ in self.query.objs]
            fields = [None]

        # Currently the backends just accept values when generating bulk
        # queries and generate their own placeholders. Doing that isn't
        # necessary and it should be possible to use placeholders and
        # expressions in bulk inserts too.
        can_bulk = (not self.returning_fields and self.connection.features.has_bulk_insert)

        placeholder_rows, param_rows = self.assemble_as_sql(fields, value_rows)

        ignore_conflicts_suffix_sql = self.connection.ops.ignore_conflicts_suffix_sql(
            ignore_conflicts=self.query.ignore_conflicts
        )
        if self.returning_fields and self.connection.features.can_return_columns_from_insert:
            if self.connection.features.can_return_rows_from_bulk_insert:
                result.append(self.connection.ops.bulk_insert_sql(fields, placeholder_rows))
                params = param_rows
            else:
                result.append("VALUES (%s)" % ", ".join(placeholder_rows[0]))
                params = [param_rows[0]]
            if ignore_conflicts_suffix_sql:
                result.append(ignore_conflicts_suffix_sql)
            # Skip empty r_sql to allow subclasses to customize behavior for
            # 3rd party backends. Refs #19096.
            r_sql, self.returning_params = self.connection.ops.return_insert_columns(self.returning_fields)
            if r_sql:
                result.append(r_sql)
                params += [self.returning_params]
            return [(" ".join(result), tuple(chain.from_iterable(params)))]

        if can_bulk:
            result.append(self.connection.ops.bulk_insert_sql(fields, placeholder_rows))
            if ignore_conflicts_suffix_sql:
                result.append(ignore_conflicts_suffix_sql)
            return [(" ".join(result), tuple(p for ps in param_rows for p in ps))]
        else:
            if ignore_conflicts_suffix_sql:
                result.append(ignore_conflicts_suffix_sql)
            return [
                (" ".join(result + ["VALUES (%s)" % ", ".join(p)]), vals)
                for p, vals in zip(placeholder_rows, param_rows)
            ]
```
### 99 - django/db/models/sql/query.py:

Start line: 2035, End line: 2069

```python
class Query(BaseExpression):

    def add_extra(self, select, select_params, where, params, tables, order_by):
        """
        Add data to the various extra_* attributes for user-created additions
        to the query.
        """
        if select:
            # We need to pair any placeholder markers in the 'select'
            # dictionary with their parameters in 'select_params' so that
            # subsequent updates to the select dictionary also adjust the
            # parameters appropriately.
            select_pairs = {}
            if select_params:
                param_iter = iter(select_params)
            else:
                param_iter = iter([])
            for name, entry in select.items():
                entry = str(entry)
                entry_params = []
                pos = entry.find("%s")
                while pos != -1:
                    if pos == 0 or entry[pos - 1] != '%':
                        entry_params.append(next(param_iter))
                    pos = entry.find("%s", pos + 2)
                select_pairs[name] = (entry, entry_params)
            self.extra.update(select_pairs)
        if where or params:
            self.where.add(ExtraWhere(where, params), AND)
        if tables:
            self.extra_tables += tuple(tables)
        if order_by:
            self.extra_order_by = order_by

    def clear_deferred_loading(self):
        """Remove any fields from the deferred loading set."""
        self.deferred_loading = (frozenset(), True)
```
### 100 - django/db/models/expressions.py:

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
### 104 - django/db/models/sql/query.py:

Start line: 243, End line: 289

```python
class Query(BaseExpression):

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

    def get_compiler(self, using=None, connection=None, elide_empty=True):
        if using is None and connection is None:
            raise ValueError("Need either using or connection")
        if using:
            connection = connections[using]
        return connection.ops.compiler(self.compiler)(self, connection, using, elide_empty)

    def get_meta(self):
        """
        Return the Options instance (the model._meta) from which to start
        processing. Normally, this is self.model._meta, but it can be changed
        by subclasses.
        """
        return self.model._meta
```
### 105 - django/db/models/sql/compiler.py:

Start line: 794, End line: 805

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
### 107 - django/db/models/expressions.py:

Start line: 1077, End line: 1107

```python
class Case(Expression):

    def as_sql(self, compiler, connection, template=None, case_joiner=None, **extra_context):
        connection.ops.check_expression_support(self)
        if not self.cases:
            return compiler.compile(self.default)
        template_params = {**self.extra, **extra_context}
        case_parts = []
        sql_params = []
        for case in self.cases:
            try:
                case_sql, case_params = compiler.compile(case)
            except EmptyResultSet:
                continue
            case_parts.append(case_sql)
            sql_params.extend(case_params)
        default_sql, default_params = compiler.compile(self.default)
        if not case_parts:
            return default_sql, default_params
        case_joiner = case_joiner or self.case_joiner
        template_params['cases'] = case_joiner.join(case_parts)
        template_params['default'] = default_sql
        sql_params.extend(default_params)
        template = template or template_params.get('template', self.template)
        sql = template % template_params
        if self._output_field_or_none is not None:
            sql = connection.ops.unification_cast_sql(self.output_field) % sql
        return sql, sql_params

    def get_group_by_cols(self, alias=None):
        if not self.cases:
            return self.default.get_group_by_cols(alias)
        return super().get_group_by_cols(alias)
```
### 109 - django/db/models/sql/compiler.py:

Start line: 427, End line: 435

```python
class SQLCompiler:
    ordering_parts =
    # ... other code

    def get_extra_select(self, order_by, select):
        extra_select = []
        if self.query.distinct and not self.query.distinct_fields:
            select_sql = [t[1] for t in select]
            for expr, (sql, params, is_ref) in order_by:
                without_ordering = self.ordering_parts.search(sql)[1]
                if not is_ref and (without_ordering, params) not in select_sql:
                    extra_select.append((expr, (without_ordering, params), None))
        return extra_select
    # ... other code
```
### 110 - django/db/models/expressions.py:

Start line: 804, End line: 818

```python
class RawSQL(Expression):
    def __init__(self, sql, params, output_field=None):
        if output_field is None:
            output_field = fields.Field()
        self.sql, self.params = sql, params
        super().__init__(output_field=output_field)

    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, self.sql, self.params)

    def as_sql(self, compiler, connection):
        return '(%s)' % self.sql, self.params

    def get_group_by_cols(self, alias=None):
        return [self]
```
### 112 - django/db/models/sql/compiler.py:

Start line: 841, End line: 921

```python
class SQLCompiler:
    ordering_parts =
    # ... other code

    def get_related_selections(self, select, opts=None, root_alias=None, cur_depth=1,
                               requested=None, restricted=None):
        """
        Fill in the information needed for a select_related query. The current
        depth is measured as the number of connections away from the root model
        (for example, cur_depth=1 means we are looking at models with direct
        connections to the root model).
        """
        def _get_field_choices():
            direct_choices = (f.name for f in opts.fields if f.is_relation)
            reverse_choices = (
                f.field.related_query_name()
                for f in opts.related_objects if f.field.unique
            )
            return chain(direct_choices, reverse_choices, self.query._filtered_relations)

        related_klass_infos = []
        if not restricted and cur_depth > self.query.max_depth:
            # We've recursed far enough; bail out.
            return related_klass_infos

        if not opts:
            opts = self.query.get_meta()
            root_alias = self.query.get_initial_alias()
        only_load = self.query.get_loaded_field_names()

        # Setup for the case when only particular related fields should be
        # included in the related selection.
        fields_found = set()
        if requested is None:
            restricted = isinstance(self.query.select_related, dict)
            if restricted:
                requested = self.query.select_related

        def get_related_klass_infos(klass_info, related_klass_infos):
            klass_info['related_klass_infos'] = related_klass_infos

        for f in opts.fields:
            field_model = f.model._meta.concrete_model
            fields_found.add(f.name)

            if restricted:
                next = requested.get(f.name, {})
                if not f.is_relation:
                    # If a non-related field is used like a relation,
                    # or if a single non-relational field is given.
                    if next or f.name in requested:
                        raise FieldError(
                            "Non-relational field given in select_related: '%s'. "
                            "Choices are: %s" % (
                                f.name,
                                ", ".join(_get_field_choices()) or '(none)',
                            )
                        )
            else:
                next = False

            if not select_related_descend(f, restricted, requested,
                                          only_load.get(field_model)):
                continue
            klass_info = {
                'model': f.remote_field.model,
                'field': f,
                'reverse': False,
                'local_setter': f.set_cached_value,
                'remote_setter': f.remote_field.set_cached_value if f.unique else lambda x, y: None,
                'from_parent': False,
            }
            related_klass_infos.append(klass_info)
            select_fields = []
            _, _, _, joins, _, _ = self.query.setup_joins(
                [f.name], opts, root_alias)
            alias = joins[-1]
            columns = self.get_default_columns(start_alias=alias, opts=f.remote_field.model._meta)
            for col in columns:
                select_fields.append(len(select))
                select.append((col, None))
            klass_info['select_fields'] = select_fields
            next_klass_infos = self.get_related_selections(
                select, f.remote_field.model._meta, alias, cur_depth + 1, next, restricted)
            get_related_klass_infos(klass_info, next_klass_infos)
        # ... other code
    # ... other code
```
### 117 - django/db/models/sql/query.py:

Start line: 1812, End line: 1841

```python
class Query(BaseExpression):

    def set_empty(self):
        self.where.add(NothingNode(), AND)
        for query in self.combined_queries:
            query.set_empty()

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
### 124 - django/db/models/sql/query.py:

Start line: 1385, End line: 1404

```python
class Query(BaseExpression):

    def _add_q(self, q_object, used_aliases, branch_negated=False,
               current_negated=False, allow_joins=True, split_subq=True,
               check_filterable=True):
        """Add a Q-object to the current filter."""
        connector = q_object.connector
        current_negated = current_negated ^ q_object.negated
        branch_negated = branch_negated or q_object.negated
        target_clause = WhereNode(connector=connector, negated=q_object.negated)
        joinpromoter = JoinPromoter(q_object.connector, len(q_object.children), current_negated)
        for child in q_object.children:
            child_clause, needed_inner = self.build_filter(
                child, can_reuse=used_aliases, branch_negated=branch_negated,
                current_negated=current_negated, allow_joins=allow_joins,
                split_subq=split_subq, check_filterable=check_filterable,
            )
            joinpromoter.add_votes(needed_inner)
            if child_clause:
                target_clause.add(child_clause, connector)
        needed_inner = joinpromoter.update_join_types(self)
        return target_clause, needed_inner
```
### 125 - django/db/models/expressions.py:

Start line: 775, End line: 801

```python
class Value(SQLiteNumericMixin, Expression):

    def _resolve_output_field(self):
        if isinstance(self.value, str):
            return fields.CharField()
        if isinstance(self.value, bool):
            return fields.BooleanField()
        if isinstance(self.value, int):
            return fields.IntegerField()
        if isinstance(self.value, float):
            return fields.FloatField()
        if isinstance(self.value, datetime.datetime):
            return fields.DateTimeField()
        if isinstance(self.value, datetime.date):
            return fields.DateField()
        if isinstance(self.value, datetime.time):
            return fields.TimeField()
        if isinstance(self.value, datetime.timedelta):
            return fields.DurationField()
        if isinstance(self.value, Decimal):
            return fields.DecimalField()
        if isinstance(self.value, bytes):
            return fields.BinaryField()
        if isinstance(self.value, UUID):
            return fields.UUIDField()

    @property
    def empty_aggregate_value(self):
        return self.value
```
### 126 - django/db/models/sql/query.py:

Start line: 65, End line: 117

```python
class RawQuery:
    """A single raw SQL query."""

    def __init__(self, sql, using, params=()):
        self.params = params
        self.sql = sql
        self.using = using
        self.cursor = None

        # Mirror some properties of a normal query so that
        # the compiler can be used to process results.
        self.low_mark, self.high_mark = 0, None  # Used for offset/limit
        self.extra_select = {}
        self.annotation_select = {}

    def chain(self, using):
        return self.clone(using)

    def clone(self, using):
        return RawQuery(self.sql, using, params=self.params)

    def get_columns(self):
        if self.cursor is None:
            self._execute_query()
        converter = connections[self.using].introspection.identifier_converter
        return [converter(column_meta[0])
                for column_meta in self.cursor.description]

    def __iter__(self):
        # Always execute a new query for a new iterator.
        # This could be optimized with a cache at the expense of RAM.
        self._execute_query()
        if not connections[self.using].features.can_use_chunked_reads:
            # If the database can't use chunked reads we need to make sure we
            # evaluate the entire query up front.
            result = list(self.cursor)
        else:
            result = self.cursor
        return iter(result)

    def __repr__(self):
        return "<%s: %s>" % (self.__class__.__name__, self)

    @property
    def params_type(self):
        if self.params is None:
            return None
        return dict if isinstance(self.params, Mapping) else tuple

    def __str__(self):
        if self.params_type is None:
            return self.sql
        return self.sql % self.params_type(self.params)
```
