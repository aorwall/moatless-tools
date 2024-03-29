# django__django-13667

| **django/django** | `4cce1d13cfe9d8e56921c5fa8c61e3034dc8e20c` |
| ---- | ---- |
| **No of patches** | 2 |
| **All found context length** | 5254 |
| **Any found context length** | 561 |
| **Avg pos** | 33.5 |
| **Min pos** | 2 |
| **Max pos** | 35 |
| **Top file pos** | 2 |
| **Missing snippets** | 5 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/db/models/expressions.py b/django/db/models/expressions.py
--- a/django/db/models/expressions.py
+++ b/django/db/models/expressions.py
@@ -1116,10 +1116,11 @@ def copy(self):
     def external_aliases(self):
         return self.query.external_aliases
 
-    def as_sql(self, compiler, connection, template=None, **extra_context):
+    def as_sql(self, compiler, connection, template=None, query=None, **extra_context):
         connection.ops.check_expression_support(self)
         template_params = {**self.extra, **extra_context}
-        subquery_sql, sql_params = self.query.as_sql(compiler, connection)
+        query = query or self.query
+        subquery_sql, sql_params = query.as_sql(compiler, connection)
         template_params['subquery'] = subquery_sql[1:-1]
 
         template = template or template_params.get('template', self.template)
@@ -1142,7 +1143,6 @@ class Exists(Subquery):
     def __init__(self, queryset, negated=False, **kwargs):
         self.negated = negated
         super().__init__(queryset, **kwargs)
-        self.query = self.query.exists()
 
     def __invert__(self):
         clone = self.copy()
@@ -1150,7 +1150,14 @@ def __invert__(self):
         return clone
 
     def as_sql(self, compiler, connection, template=None, **extra_context):
-        sql, params = super().as_sql(compiler, connection, template, **extra_context)
+        query = self.query.exists(using=connection.alias)
+        sql, params = super().as_sql(
+            compiler,
+            connection,
+            template=template,
+            query=query,
+            **extra_context,
+        )
         if self.negated:
             sql = 'NOT {}'.format(sql)
         return sql, params
diff --git a/django/db/models/sql/query.py b/django/db/models/sql/query.py
--- a/django/db/models/sql/query.py
+++ b/django/db/models/sql/query.py
@@ -525,7 +525,7 @@ def get_count(self, using):
     def has_filters(self):
         return self.where
 
-    def exists(self):
+    def exists(self, using, limit=True):
         q = self.clone()
         if not q.distinct:
             if q.group_by is True:
@@ -534,14 +534,21 @@ def exists(self):
                 # SELECT clause which is about to be cleared.
                 q.set_group_by(allow_aliases=False)
             q.clear_select_clause()
+        if q.combined_queries and q.combinator == 'union':
+            limit_combined = connections[using].features.supports_slicing_ordering_in_compound
+            q.combined_queries = tuple(
+                combined_query.exists(using, limit=limit_combined)
+                for combined_query in q.combined_queries
+            )
         q.clear_ordering(True)
-        q.set_limits(high=1)
+        if limit:
+            q.set_limits(high=1)
         q.add_extra({'a': 1}, None, None, None, None, None)
         q.set_extra_mask(['a'])
         return q
 
     def has_results(self, using):
-        q = self.exists()
+        q = self.exists(using)
         compiler = q.get_compiler(using=using)
         return compiler.has_results()
 

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/db/models/expressions.py | 1119 | 1122 | 35 | 2 | 12946
| django/db/models/expressions.py | 1145 | 1145 | 2 | 2 | 561
| django/db/models/expressions.py | 1153 | 1153 | 2 | 2 | 561
| django/db/models/sql/query.py | 528 | 528 | 14 | 3 | 5254
| django/db/models/sql/query.py | 537 | 543 | 14 | 3 | 5254


## Problem Statement

```
Augment QuerySet.exists() optimizations to .union().exists().
Description
	 
		(last modified by Simon Charette)
	 
The QuerySet.exists method performs optimization by ​clearing the select clause, ​dropping ordering, and limiting the number of results to 1 if possible.
A similar optimization can be applied for combined queries when using QuerySet.union() as some query planers (e.g. MySQL) are not smart enough to prune changes down into combined queries.
For example, given filtered_authors.union(other_authors).exists() the currently generated is
SELECT 1 FROM (SELECT * FROM authors WHERE ... ORDER BY ... UNION SELECT * FROM authors WHERE ... ORDER BY ...) LIMIT 1;
But some planers won't be smart enough to realize that both * and ORDER BY are not necessary and fetch all matching rows. In order to help them we should generate the following SQL
SELECT 1 FROM (SELECT 1 FROM authors WHERE ... LIMIT 1 UNION SELECT 1 FROM authors WHERE ... LIMIT 1) LIMIT 1;
This can already be done manually through filtered_authors.order_by().values(Value(1))[:1].union(other_authors.order_by().values(Value(1))[:1]).exists() but that involves a lot of boilerplate.
Note that the optimization is only possible in this form for union and not for intersection and difference since they require both sets to be unaltered.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 django/db/models/query.py | 801 | 840| 322 | 322 | 17234 | 
| **-> 2 <-** | **2 django/db/models/expressions.py** | 1138 | 1164| 239 | 561 | 28082 | 
| 3 | 2 django/db/models/query.py | 985 | 1005| 239 | 800 | 28082 | 
| 4 | 2 django/db/models/query.py | 320 | 346| 222 | 1022 | 28082 | 
| 5 | **3 django/db/models/sql/query.py** | 1750 | 1815| 666 | 1688 | 50540 | 
| 6 | **3 django/db/models/sql/query.py** | 1301 | 1366| 772 | 2460 | 50540 | 
| 7 | 3 django/db/models/query.py | 1007 | 1020| 126 | 2586 | 50540 | 
| 8 | **3 django/db/models/sql/query.py** | 631 | 655| 269 | 2855 | 50540 | 
| 9 | 3 django/db/models/query.py | 1351 | 1399| 405 | 3260 | 50540 | 
| 10 | **3 django/db/models/sql/query.py** | 1643 | 1667| 227 | 3487 | 50540 | 
| 11 | **3 django/db/models/sql/query.py** | 1848 | 1897| 329 | 3816 | 50540 | 
| 12 | 4 django/db/models/query_utils.py | 312 | 352| 286 | 4102 | 53246 | 
| 13 | **4 django/db/models/sql/query.py** | 2418 | 2473| 827 | 4929 | 53246 | 
| **-> 14 <-** | **4 django/db/models/sql/query.py** | 514 | 554| 325 | 5254 | 53246 | 
| 15 | 4 django/db/models/query.py | 916 | 966| 371 | 5625 | 53246 | 
| 16 | **4 django/db/models/sql/query.py** | 556 | 630| 809 | 6434 | 53246 | 
| 17 | 4 django/db/models/query.py | 1196 | 1215| 209 | 6643 | 53246 | 
| 18 | 4 django/db/models/query.py | 1340 | 1349| 114 | 6757 | 53246 | 
| 19 | 4 django/db/models/query.py | 1098 | 1139| 323 | 7080 | 53246 | 
| 20 | **4 django/db/models/sql/query.py** | 1368 | 1389| 250 | 7330 | 53246 | 
| 21 | 4 django/db/models/query.py | 1217 | 1245| 183 | 7513 | 53246 | 
| 22 | **4 django/db/models/sql/query.py** | 1435 | 1462| 283 | 7796 | 53246 | 
| 23 | **4 django/db/models/sql/query.py** | 2181 | 2228| 394 | 8190 | 53246 | 
| 24 | 4 django/db/models/query.py | 1060 | 1081| 214 | 8404 | 53246 | 
| 25 | 4 django/db/models/query.py | 1141 | 1177| 324 | 8728 | 53246 | 
| 26 | **4 django/db/models/sql/query.py** | 373 | 423| 494 | 9222 | 53246 | 
| 27 | **4 django/db/models/sql/query.py** | 2039 | 2085| 370 | 9592 | 53246 | 
| 28 | 4 django/db/models/query.py | 175 | 234| 469 | 10061 | 53246 | 
| 29 | **4 django/db/models/sql/query.py** | 1988 | 2037| 420 | 10481 | 53246 | 
| 30 | **4 django/db/models/sql/query.py** | 706 | 741| 389 | 10870 | 53246 | 
| 31 | 5 django/db/models/sql/where.py | 233 | 249| 130 | 11000 | 55060 | 
| 32 | 5 django/db/models/query_utils.py | 57 | 108| 396 | 11396 | 55060 | 
| 33 | **5 django/db/models/sql/query.py** | 137 | 231| 833 | 12229 | 55060 | 
| 34 | **5 django/db/models/expressions.py** | 219 | 253| 285 | 12514 | 55060 | 
| **-> 35 <-** | **5 django/db/models/expressions.py** | 1077 | 1135| 432 | 12946 | 55060 | 
| 36 | 5 django/db/models/query.py | 842 | 871| 248 | 13194 | 55060 | 
| 37 | 5 django/db/models/query.py | 1083 | 1096| 115 | 13309 | 55060 | 
| 38 | **5 django/db/models/sql/query.py** | 1 | 66| 479 | 13788 | 55060 | 
| 39 | 5 django/db/models/query.py | 968 | 983| 124 | 13912 | 55060 | 
| 40 | **5 django/db/models/sql/query.py** | 1817 | 1846| 259 | 14171 | 55060 | 
| 41 | 5 django/db/models/query.py | 1038 | 1058| 155 | 14326 | 55060 | 
| 42 | **5 django/db/models/sql/query.py** | 2264 | 2336| 774 | 15100 | 55060 | 
| 43 | 5 django/db/models/query.py | 1294 | 1318| 210 | 15310 | 55060 | 
| 44 | **5 django/db/models/sql/query.py** | 1480 | 1565| 801 | 16111 | 55060 | 
| 45 | **5 django/db/models/sql/query.py** | 902 | 924| 248 | 16359 | 55060 | 
| 46 | 6 django/db/models/sql/subqueries.py | 1 | 44| 320 | 16679 | 56261 | 
| 47 | 6 django/db/models/query.py | 1 | 39| 294 | 16973 | 56261 | 
| 48 | **6 django/db/models/sql/query.py** | 1709 | 1748| 439 | 17412 | 56261 | 
| 49 | 6 django/db/models/query.py | 784 | 800| 157 | 17569 | 56261 | 
| 50 | **6 django/db/models/sql/query.py** | 1062 | 1091| 245 | 17814 | 56261 | 
| 51 | 6 django/db/models/query.py | 1485 | 1518| 297 | 18111 | 56261 | 
| 52 | 6 django/db/models/query.py | 1320 | 1338| 186 | 18297 | 56261 | 
| 53 | 6 django/db/models/query.py | 1657 | 1763| 1063 | 19360 | 56261 | 
| 54 | **6 django/db/models/sql/query.py** | 2230 | 2262| 228 | 19588 | 56261 | 
| 55 | 6 django/db/models/query.py | 721 | 748| 235 | 19823 | 56261 | 
| 56 | 7 django/db/models/sql/compiler.py | 1208 | 1229| 223 | 20046 | 70652 | 
| 57 | **7 django/db/models/sql/query.py** | 1669 | 1707| 356 | 20402 | 70652 | 
| 58 | **7 django/db/models/sql/query.py** | 776 | 808| 392 | 20794 | 70652 | 
| 59 | **7 django/db/models/sql/query.py** | 994 | 1025| 307 | 21101 | 70652 | 
| 60 | **7 django/db/models/sql/query.py** | 1122 | 1154| 338 | 21439 | 70652 | 
| 61 | 7 django/db/models/query.py | 1445 | 1483| 308 | 21747 | 70652 | 
| 62 | 7 django/db/models/sql/compiler.py | 442 | 495| 564 | 22311 | 70652 | 
| 63 | 8 django/db/models/sql/datastructures.py | 1 | 21| 126 | 22437 | 72120 | 
| 64 | 8 django/db/models/query.py | 666 | 680| 132 | 22569 | 72120 | 
| 65 | **8 django/db/models/sql/query.py** | 810 | 836| 280 | 22849 | 72120 | 
| 66 | 8 django/db/models/query.py | 1022 | 1036| 145 | 22994 | 72120 | 
| 67 | **8 django/db/models/sql/query.py** | 1221 | 1299| 729 | 23723 | 72120 | 
| 68 | **8 django/db/models/sql/query.py** | 1391 | 1411| 247 | 23970 | 72120 | 
| 69 | **8 django/db/models/sql/query.py** | 1413 | 1433| 212 | 24182 | 72120 | 
| 70 | **8 django/db/models/sql/query.py** | 2390 | 2416| 176 | 24358 | 72120 | 
| 71 | 9 django/contrib/postgres/search.py | 130 | 157| 248 | 24606 | 74342 | 
| 72 | **9 django/db/models/sql/query.py** | 297 | 346| 444 | 25050 | 74342 | 
| 73 | 9 django/db/models/query.py | 265 | 285| 180 | 25230 | 74342 | 
| 74 | 9 django/db/models/query.py | 1878 | 1910| 314 | 25544 | 74342 | 
| 75 | **9 django/db/models/sql/query.py** | 348 | 371| 179 | 25723 | 74342 | 
| 76 | 9 django/db/models/query.py | 1179 | 1194| 149 | 25872 | 74342 | 
| 77 | 9 django/db/models/sql/subqueries.py | 47 | 75| 210 | 26082 | 74342 | 
| 78 | 9 django/db/models/sql/compiler.py | 271 | 363| 753 | 26835 | 74342 | 
| 79 | 9 django/db/models/query.py | 750 | 783| 274 | 27109 | 74342 | 
| 80 | 9 django/db/models/sql/compiler.py | 149 | 197| 523 | 27632 | 74342 | 
| 81 | 9 django/db/models/sql/compiler.py | 765 | 776| 163 | 27795 | 74342 | 
| 82 | 10 django/db/models/lookups.py | 103 | 157| 427 | 28222 | 79369 | 
| 83 | 10 django/db/models/sql/subqueries.py | 137 | 163| 161 | 28383 | 79369 | 
| 84 | **10 django/db/models/sql/query.py** | 425 | 512| 890 | 29273 | 79369 | 
| 85 | 10 django/db/models/query.py | 401 | 444| 370 | 29643 | 79369 | 
| 86 | 10 django/db/models/query.py | 236 | 263| 221 | 29864 | 79369 | 
| 87 | 11 django/db/models/fields/related_lookups.py | 62 | 101| 451 | 30315 | 80822 | 
| 88 | **11 django/db/models/sql/query.py** | 946 | 992| 439 | 30754 | 80822 | 
| 89 | **11 django/db/models/expressions.py** | 929 | 993| 591 | 31345 | 80822 | 
| 90 | **11 django/db/models/sql/query.py** | 1156 | 1199| 469 | 31814 | 80822 | 
| 91 | 12 django/db/models/aggregates.py | 70 | 96| 266 | 32080 | 82123 | 
| 92 | 13 django/db/backends/oracle/operations.py | 619 | 645| 303 | 32383 | 88094 | 
| 93 | 13 django/db/models/query.py | 1569 | 1625| 481 | 32864 | 88094 | 
| 94 | 13 django/db/models/sql/compiler.py | 778 | 810| 348 | 33212 | 88094 | 
| 95 | 13 django/db/models/sql/compiler.py | 497 | 654| 1469 | 34681 | 88094 | 
| 96 | 13 django/db/models/sql/compiler.py | 22 | 47| 257 | 34938 | 88094 | 
| 97 | **13 django/db/models/sql/query.py** | 1899 | 1940| 355 | 35293 | 88094 | 
| 98 | **13 django/db/models/expressions.py** | 1218 | 1247| 211 | 35504 | 88094 | 
| 99 | 13 django/db/models/sql/compiler.py | 700 | 722| 202 | 35706 | 88094 | 
| 100 | 14 django/contrib/postgres/constraints.py | 69 | 91| 213 | 35919 | 89519 | 
| 101 | **14 django/db/models/sql/query.py** | 2087 | 2109| 249 | 36168 | 89519 | 
| 102 | 14 django/db/models/aggregates.py | 45 | 68| 294 | 36462 | 89519 | 
| 103 | 14 django/db/models/sql/compiler.py | 63 | 147| 881 | 37343 | 89519 | 
| 104 | **14 django/db/models/expressions.py** | 489 | 514| 303 | 37646 | 89519 | 
| 105 | **14 django/db/models/sql/query.py** | 233 | 295| 446 | 38092 | 89519 | 
| 106 | **14 django/db/models/sql/query.py** | 1567 | 1642| 758 | 38850 | 89519 | 
| 107 | 14 django/db/models/sql/compiler.py | 365 | 405| 482 | 39332 | 89519 | 
| 108 | 15 django/forms/models.py | 98 | 111| 157 | 39489 | 101365 | 
| 109 | **15 django/db/models/expressions.py** | 795 | 811| 153 | 39642 | 101365 | 
| 110 | 15 django/db/models/query.py | 92 | 110| 138 | 39780 | 101365 | 
| 111 | 15 django/db/models/fields/related_lookups.py | 121 | 157| 244 | 40024 | 101365 | 
| 112 | 15 django/db/models/sql/where.py | 32 | 63| 317 | 40341 | 101365 | 
| 113 | **15 django/db/models/sql/query.py** | 657 | 704| 511 | 40852 | 101365 | 
| 114 | 15 django/db/models/sql/compiler.py | 407 | 415| 133 | 40985 | 101365 | 
| 115 | 15 django/db/models/query.py | 113 | 140| 233 | 41218 | 101365 | 
| 116 | 15 django/db/models/sql/compiler.py | 1549 | 1589| 409 | 41627 | 101365 | 
| 117 | 15 django/forms/models.py | 1188 | 1253| 543 | 42170 | 101365 | 
| 118 | 15 django/db/models/sql/subqueries.py | 111 | 134| 192 | 42362 | 101365 | 
| 119 | 15 django/db/models/sql/datastructures.py | 59 | 102| 419 | 42781 | 101365 | 
| 120 | 16 django/db/models/fields/related.py | 696 | 708| 116 | 42897 | 115241 | 
| 121 | **16 django/db/models/expressions.py** | 442 | 487| 314 | 43211 | 115241 | 
| 122 | 16 django/db/models/query.py | 1813 | 1876| 658 | 43869 | 115241 | 
| 123 | 16 django/db/models/sql/where.py | 65 | 115| 396 | 44265 | 115241 | 
| 124 | 16 django/db/models/query.py | 143 | 172| 207 | 44472 | 115241 | 
| 125 | **16 django/db/models/sql/query.py** | 1027 | 1060| 349 | 44821 | 115241 | 
| 126 | **16 django/db/models/sql/query.py** | 1942 | 1986| 364 | 45185 | 115241 | 
| 127 | 16 django/db/models/query.py | 348 | 363| 146 | 45331 | 115241 | 
| 128 | 16 django/db/models/lookups.py | 303 | 315| 168 | 45499 | 115241 | 
| 129 | 17 django/db/models/constraints.py | 79 | 161| 729 | 46228 | 116856 | 
| 130 | 17 django/contrib/postgres/search.py | 198 | 230| 243 | 46471 | 116856 | 
| 131 | 18 django/db/models/base.py | 1714 | 1814| 729 | 47200 | 133529 | 
| 132 | 19 django/db/models/deletion.py | 1 | 76| 566 | 47766 | 137357 | 
| 133 | **19 django/db/models/sql/query.py** | 926 | 944| 146 | 47912 | 137357 | 
| 134 | 20 django/contrib/postgres/lookups.py | 1 | 61| 337 | 48249 | 137694 | 
| 135 | 21 django/db/models/functions/comparison.py | 57 | 75| 205 | 48454 | 139098 | 
| 136 | **21 django/db/models/expressions.py** | 814 | 848| 292 | 48746 | 139098 | 
| 137 | **21 django/db/models/sql/query.py** | 743 | 774| 296 | 49042 | 139098 | 
| 138 | **21 django/db/models/expressions.py** | 547 | 588| 298 | 49340 | 139098 | 
| 139 | 21 django/db/models/query.py | 1628 | 1656| 246 | 49586 | 139098 | 
| 140 | 22 django/contrib/postgres/aggregates/mixins.py | 22 | 34| 124 | 49710 | 139536 | 
| 141 | 22 django/contrib/postgres/search.py | 160 | 195| 313 | 50023 | 139536 | 
| 142 | 23 django/contrib/gis/geos/geometry.py | 580 | 638| 461 | 50484 | 145451 | 
| 143 | 24 django/db/migrations/optimizer.py | 1 | 38| 344 | 50828 | 146041 | 
| 144 | **24 django/db/models/sql/query.py** | 69 | 117| 361 | 51189 | 146041 | 
| 145 | **24 django/db/models/expressions.py** | 1191 | 1216| 248 | 51437 | 146041 | 
| 146 | 24 django/db/models/sql/compiler.py | 894 | 986| 839 | 52276 | 146041 | 
| 147 | 25 django/contrib/admin/options.py | 988 | 1021| 310 | 52586 | 164634 | 
| 148 | 25 django/db/models/sql/compiler.py | 812 | 892| 717 | 53303 | 164634 | 
| 149 | 25 django/contrib/postgres/search.py | 1 | 24| 205 | 53508 | 164634 | 
| 150 | 26 django/db/models/__init__.py | 1 | 53| 619 | 54127 | 165253 | 
| 151 | 26 django/db/models/query.py | 569 | 594| 202 | 54329 | 165253 | 
| 152 | 26 django/db/models/deletion.py | 346 | 359| 116 | 54445 | 165253 | 
| 153 | 27 django/db/backends/postgresql/features.py | 1 | 104| 844 | 55289 | 166097 | 
| 154 | 27 django/db/models/sql/datastructures.py | 117 | 148| 170 | 55459 | 166097 | 
| 155 | 27 django/db/models/query.py | 1272 | 1292| 253 | 55712 | 166097 | 
| 156 | 27 django/db/models/fields/related.py | 401 | 419| 165 | 55877 | 166097 | 
| 157 | 27 django/db/models/query.py | 1974 | 1997| 200 | 56077 | 166097 | 
| 158 | 27 django/contrib/postgres/constraints.py | 93 | 106| 179 | 56256 | 166097 | 
| 159 | **27 django/db/models/expressions.py** | 591 | 630| 290 | 56546 | 166097 | 
| 160 | 27 django/db/models/sql/where.py | 212 | 230| 131 | 56677 | 166097 | 
| 161 | 27 django/db/models/query.py | 365 | 399| 314 | 56991 | 166097 | 
| 162 | 28 django/views/generic/detail.py | 58 | 76| 154 | 57145 | 167412 | 
| 163 | 28 django/db/models/sql/where.py | 1 | 30| 167 | 57312 | 167412 | 
| 164 | 28 django/contrib/postgres/aggregates/mixins.py | 1 | 20| 182 | 57494 | 167412 | 
| 165 | 28 django/db/models/sql/datastructures.py | 24 | 57| 354 | 57848 | 167412 | 
| 166 | 28 django/db/models/query.py | 42 | 89| 457 | 58305 | 167412 | 
| 167 | 28 django/db/models/sql/compiler.py | 199 | 269| 580 | 58885 | 167412 | 
| 168 | 28 django/db/models/lookups.py | 371 | 403| 294 | 59179 | 167412 | 
| 169 | 28 django/db/models/query.py | 1402 | 1433| 246 | 59425 | 167412 | 
| 170 | 28 django/db/models/query_utils.py | 284 | 309| 293 | 59718 | 167412 | 
| 171 | 29 django/db/backends/mysql/operations.py | 278 | 289| 165 | 59883 | 171116 | 
| 172 | 30 django/db/backends/sqlite3/operations.py | 125 | 150| 279 | 60162 | 174206 | 
| 173 | 30 django/db/models/query.py | 527 | 568| 495 | 60657 | 174206 | 
| 174 | 30 django/db/models/sql/compiler.py | 1047 | 1087| 337 | 60994 | 174206 | 
| 175 | 30 django/db/models/query.py | 596 | 614| 178 | 61172 | 174206 | 
| 176 | 30 django/contrib/postgres/search.py | 95 | 127| 277 | 61449 | 174206 | 
| 177 | 30 django/db/models/sql/compiler.py | 1025 | 1046| 199 | 61648 | 174206 | 
| 178 | 30 django/db/models/sql/where.py | 157 | 193| 243 | 61891 | 174206 | 


### Hint

```
PR ​https://github.com/django/django/pull/13667
```

## Patch

```diff
diff --git a/django/db/models/expressions.py b/django/db/models/expressions.py
--- a/django/db/models/expressions.py
+++ b/django/db/models/expressions.py
@@ -1116,10 +1116,11 @@ def copy(self):
     def external_aliases(self):
         return self.query.external_aliases
 
-    def as_sql(self, compiler, connection, template=None, **extra_context):
+    def as_sql(self, compiler, connection, template=None, query=None, **extra_context):
         connection.ops.check_expression_support(self)
         template_params = {**self.extra, **extra_context}
-        subquery_sql, sql_params = self.query.as_sql(compiler, connection)
+        query = query or self.query
+        subquery_sql, sql_params = query.as_sql(compiler, connection)
         template_params['subquery'] = subquery_sql[1:-1]
 
         template = template or template_params.get('template', self.template)
@@ -1142,7 +1143,6 @@ class Exists(Subquery):
     def __init__(self, queryset, negated=False, **kwargs):
         self.negated = negated
         super().__init__(queryset, **kwargs)
-        self.query = self.query.exists()
 
     def __invert__(self):
         clone = self.copy()
@@ -1150,7 +1150,14 @@ def __invert__(self):
         return clone
 
     def as_sql(self, compiler, connection, template=None, **extra_context):
-        sql, params = super().as_sql(compiler, connection, template, **extra_context)
+        query = self.query.exists(using=connection.alias)
+        sql, params = super().as_sql(
+            compiler,
+            connection,
+            template=template,
+            query=query,
+            **extra_context,
+        )
         if self.negated:
             sql = 'NOT {}'.format(sql)
         return sql, params
diff --git a/django/db/models/sql/query.py b/django/db/models/sql/query.py
--- a/django/db/models/sql/query.py
+++ b/django/db/models/sql/query.py
@@ -525,7 +525,7 @@ def get_count(self, using):
     def has_filters(self):
         return self.where
 
-    def exists(self):
+    def exists(self, using, limit=True):
         q = self.clone()
         if not q.distinct:
             if q.group_by is True:
@@ -534,14 +534,21 @@ def exists(self):
                 # SELECT clause which is about to be cleared.
                 q.set_group_by(allow_aliases=False)
             q.clear_select_clause()
+        if q.combined_queries and q.combinator == 'union':
+            limit_combined = connections[using].features.supports_slicing_ordering_in_compound
+            q.combined_queries = tuple(
+                combined_query.exists(using, limit=limit_combined)
+                for combined_query in q.combined_queries
+            )
         q.clear_ordering(True)
-        q.set_limits(high=1)
+        if limit:
+            q.set_limits(high=1)
         q.add_extra({'a': 1}, None, None, None, None, None)
         q.set_extra_mask(['a'])
         return q
 
     def has_results(self, using):
-        q = self.exists()
+        q = self.exists(using)
         compiler = q.get_compiler(using=using)
         return compiler.has_results()
 

```

## Test Patch

```diff
diff --git a/tests/queries/test_qs_combinators.py b/tests/queries/test_qs_combinators.py
--- a/tests/queries/test_qs_combinators.py
+++ b/tests/queries/test_qs_combinators.py
@@ -3,6 +3,7 @@
 from django.db import DatabaseError, NotSupportedError, connection
 from django.db.models import Exists, F, IntegerField, OuterRef, Value
 from django.test import TestCase, skipIfDBFeature, skipUnlessDBFeature
+from django.test.utils import CaptureQueriesContext
 
 from .models import Number, ReservedName
 
@@ -254,6 +255,41 @@ def test_count_intersection(self):
         qs2 = Number.objects.filter(num__lte=5)
         self.assertEqual(qs1.intersection(qs2).count(), 1)
 
+    def test_exists_union(self):
+        qs1 = Number.objects.filter(num__gte=5)
+        qs2 = Number.objects.filter(num__lte=5)
+        with CaptureQueriesContext(connection) as context:
+            self.assertIs(qs1.union(qs2).exists(), True)
+        captured_queries = context.captured_queries
+        self.assertEqual(len(captured_queries), 1)
+        captured_sql = captured_queries[0]['sql']
+        self.assertNotIn(
+            connection.ops.quote_name(Number._meta.pk.column),
+            captured_sql,
+        )
+        self.assertEqual(
+            captured_sql.count(connection.ops.limit_offset_sql(None, 1)),
+            3 if connection.features.supports_slicing_ordering_in_compound else 1
+        )
+
+    def test_exists_union_empty_result(self):
+        qs = Number.objects.filter(pk__in=[])
+        self.assertIs(qs.union(qs).exists(), False)
+
+    @skipUnlessDBFeature('supports_select_intersection')
+    def test_exists_intersection(self):
+        qs1 = Number.objects.filter(num__gt=5)
+        qs2 = Number.objects.filter(num__lt=5)
+        self.assertIs(qs1.intersection(qs1).exists(), True)
+        self.assertIs(qs1.intersection(qs2).exists(), False)
+
+    @skipUnlessDBFeature('supports_select_difference')
+    def test_exists_difference(self):
+        qs1 = Number.objects.filter(num__gte=5)
+        qs2 = Number.objects.filter(num__gte=3)
+        self.assertIs(qs1.difference(qs2).exists(), False)
+        self.assertIs(qs2.difference(qs1).exists(), True)
+
     def test_get_union(self):
         qs = Number.objects.filter(num=2)
         self.assertEqual(qs.union(qs).get().num, 2)

```


## Code snippets

### 1 - django/db/models/query.py:

Start line: 801, End line: 840

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
### 2 - django/db/models/expressions.py:

Start line: 1138, End line: 1164

```python
class Exists(Subquery):
    template = 'EXISTS(%(subquery)s)'
    output_field = fields.BooleanField()

    def __init__(self, queryset, negated=False, **kwargs):
        self.negated = negated
        super().__init__(queryset, **kwargs)
        self.query = self.query.exists()

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
### 3 - django/db/models/query.py:

Start line: 985, End line: 1005

```python
class QuerySet:

    def _combinator_query(self, combinator, *other_qs, all=False):
        # Clone the query to inherit the select list and everything
        clone = self._chain()
        # Clear limits and ordering so they can be reapplied
        clone.query.clear_ordering(True)
        clone.query.clear_limits()
        clone.query.combined_queries = (self.query,) + tuple(qs.query for qs in other_qs)
        clone.query.combinator = combinator
        clone.query.combinator_all = all
        return clone

    def union(self, *other_qs, all=False):
        # If the query is an EmptyQuerySet, combine all nonempty querysets.
        if isinstance(self, EmptyQuerySet):
            qs = [q for q in other_qs if not isinstance(q, EmptyQuerySet)]
            if not qs:
                return self
            if len(qs) == 1:
                return qs[0]
            return qs[0]._combinator_query('union', *qs[1:], all=all)
        return self._combinator_query('union', *other_qs, all=all)
```
### 4 - django/db/models/query.py:

Start line: 320, End line: 346

```python
class QuerySet:

    def __class_getitem__(cls, *args, **kwargs):
        return cls

    def __and__(self, other):
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
### 5 - django/db/models/sql/query.py:

Start line: 1750, End line: 1815

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
### 6 - django/db/models/sql/query.py:

Start line: 1301, End line: 1366

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

Start line: 1007, End line: 1020

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
### 8 - django/db/models/sql/query.py:

Start line: 631, End line: 655

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
### 9 - django/db/models/query.py:

Start line: 1351, End line: 1399

```python
class QuerySet:

    def _merge_known_related_objects(self, other):
        """
        Keep track of all known related objects from either QuerySet instance.
        """
        for field, objects in other._known_related_objects.items():
            self._known_related_objects.setdefault(field, {}).update(objects)

    def resolve_expression(self, *args, **kwargs):
        if self._fields and len(self._fields) > 1:
            # values() queryset can only be used as nested queries
            # if they are set up to select only a single field.
            raise TypeError('Cannot use multi-field values as a filter value.')
        query = self.query.resolve_expression(*args, **kwargs)
        query._db = self._db
        return query
    resolve_expression.queryset_only = True

    def _add_hints(self, **hints):
        """
        Update hinting information for use by routers. Add new key/values or
        overwrite existing key/values.
        """
        self._hints.update(hints)

    def _has_filters(self):
        """
        Check if this QuerySet has any filtering going on. This isn't
        equivalent with checking if all objects are present in results, for
        example, qs[1:]._has_filters() -> False.
        """
        return self.query.has_filters()

    @staticmethod
    def _validate_values_are_expressions(values, method_name):
        invalid_args = sorted(str(arg) for arg in values if not hasattr(arg, 'resolve_expression'))
        if invalid_args:
            raise TypeError(
                'QuerySet.%s() received non-expression(s): %s.' % (
                    method_name,
                    ', '.join(invalid_args),
                )
            )

    def _not_support_combined_queries(self, operation_name):
        if self.query.combinator:
            raise NotSupportedError(
                'Calling QuerySet.%s() after %s() is not supported.'
                % (operation_name, self.query.combinator)
            )
```
### 10 - django/db/models/sql/query.py:

Start line: 1643, End line: 1667

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
### 11 - django/db/models/sql/query.py:

Start line: 1848, End line: 1897

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
### 13 - django/db/models/sql/query.py:

Start line: 2418, End line: 2473

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
### 14 - django/db/models/sql/query.py:

Start line: 514, End line: 554

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

    def exists(self):
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
        q.add_extra({'a': 1}, None, None, None, None, None)
        q.set_extra_mask(['a'])
        return q

    def has_results(self, using):
        q = self.exists()
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
### 16 - django/db/models/sql/query.py:

Start line: 556, End line: 630

```python
class Query(BaseExpression):

    def combine(self, rhs, connector):
        """
        Merge the 'rhs' query into the current one (with any 'rhs' effects
        being applied *after* (that is, "to the right of") anything in the
        current query. 'rhs' is not modified during a call to this function.

        The 'connector' parameter describes how to connect filters from the
        'rhs' query.
        """
        assert self.model == rhs.model, \
            "Cannot combine queries on two different base models."
        assert not self.is_sliced, \
            "Cannot combine queries once a slice has been taken."
        assert self.distinct == rhs.distinct, \
            "Cannot combine a unique query with a non-unique query."
        assert self.distinct_fields == rhs.distinct_fields, \
            "Cannot combine queries with different distinct fields."

        # Work out how to relabel the rhs aliases, if necessary.
        change_map = {}
        conjunction = (connector == AND)

        # Determine which existing joins can be reused. When combining the
        # query with AND we must recreate all joins for m2m filters. When
        # combining with OR we can reuse joins. The reason is that in AND
        # case a single row can't fulfill a condition like:
        #     revrel__col=1 & revrel__col=2
        # But, there might be two different related rows matching this
        # condition. In OR case a single True is enough, so single row is
        # enough, too.
        #
        # Note that we will be creating duplicate joins for non-m2m joins in
        # the AND case. The results will be correct but this creates too many
        # joins. This is something that could be fixed later on.
        reuse = set() if conjunction else set(self.alias_map)
        # Base table must be present in the query - this is the same
        # table on both sides.
        self.get_initial_alias()
        joinpromoter = JoinPromoter(connector, 2, False)
        joinpromoter.add_votes(
            j for j in self.alias_map if self.alias_map[j].join_type == INNER)
        rhs_votes = set()
        # Now, add the joins from rhs query into the new query (skipping base
        # table).
        rhs_tables = list(rhs.alias_map)[1:]
        for alias in rhs_tables:
            join = rhs.alias_map[alias]
            # If the left side of the join was already relabeled, use the
            # updated alias.
            join = join.relabeled_clone(change_map)
            new_alias = self.join(join, reuse=reuse)
            if join.join_type == INNER:
                rhs_votes.add(new_alias)
            # We can't reuse the same join again in the query. If we have two
            # distinct joins for the same connection in rhs query, then the
            # combined query must have two joins, too.
            reuse.discard(new_alias)
            if alias != new_alias:
                change_map[alias] = new_alias
            if not rhs.alias_refcount[alias]:
                # The alias was unused in the rhs query. Unref it so that it
                # will be unused in the new query, too. We have to add and
                # unref the alias so that join promotion has information of
                # the join type for the unused alias.
                self.unref_alias(new_alias)
        joinpromoter.add_votes(rhs_votes)
        joinpromoter.update_join_types(self)

        # Now relabel a copy of the rhs where-clause and add it to the current
        # one.
        w = rhs.where.clone()
        w.relabel_aliases(change_map)
        self.where.add(w, connector)

        # Selection columns and extra extensions are those provided by 'rhs'.
        # ... other code
```
### 20 - django/db/models/sql/query.py:

Start line: 1368, End line: 1389

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
### 22 - django/db/models/sql/query.py:

Start line: 1435, End line: 1462

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
### 23 - django/db/models/sql/query.py:

Start line: 2181, End line: 2228

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
### 26 - django/db/models/sql/query.py:

Start line: 373, End line: 423

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
### 27 - django/db/models/sql/query.py:

Start line: 2039, End line: 2085

```python
class Query(BaseExpression):

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
```
### 29 - django/db/models/sql/query.py:

Start line: 1988, End line: 2037

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
### 30 - django/db/models/sql/query.py:

Start line: 706, End line: 741

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
### 33 - django/db/models/sql/query.py:

Start line: 137, End line: 231

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
### 34 - django/db/models/expressions.py:

Start line: 219, End line: 253

```python
@deconstructible
class BaseExpression:

    @cached_property
    def contains_aggregate(self):
        return any(expr and expr.contains_aggregate for expr in self.get_source_expressions())

    @cached_property
    def contains_over_clause(self):
        return any(expr and expr.contains_over_clause for expr in self.get_source_expressions())

    @cached_property
    def contains_column_references(self):
        return any(expr and expr.contains_column_references for expr in self.get_source_expressions())

    def resolve_expression(self, query=None, allow_joins=True, reuse=None, summarize=False, for_save=False):
        """
        Provide the chance to do any preprocessing or validation before being
        added to the query.

        Arguments:
         * query: the backend query implementation
         * allow_joins: boolean allowing or denying use of joins
           in this query
         * reuse: a set of reusable joins for multijoins
         * summarize: a terminal aggregate clause
         * for_save: whether this expression about to be used in a save or update

        Return: an Expression to be added to the query.
        """
        c = self.copy()
        c.is_summary = summarize
        c.set_source_expressions([
            expr.resolve_expression(query, allow_joins, reuse, summarize)
            if expr else None
            for expr in c.get_source_expressions()
        ])
        return c
```
### 35 - django/db/models/expressions.py:

Start line: 1077, End line: 1135

```python
class Subquery(Expression):
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

    def __getstate__(self):
        state = super().__getstate__()
        args, kwargs = state['_constructor_args']
        if args:
            args = (self.query, *args[1:])
        else:
            kwargs['queryset'] = self.query
        state['_constructor_args'] = args, kwargs
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
        external_cols = self.query.get_external_cols()
        if any(col.possibly_multivalued for col in external_cols):
            return [self]
        return external_cols
```
### 38 - django/db/models/sql/query.py:

Start line: 1, End line: 66

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
from django.utils.deprecation import RemovedInDjango40Warning
from django.utils.functional import cached_property
from django.utils.hashable import make_hashable
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
### 40 - django/db/models/sql/query.py:

Start line: 1817, End line: 1846

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
### 42 - django/db/models/sql/query.py:

Start line: 2264, End line: 2336

```python
class Query(BaseExpression):

    def trim_start(self, names_with_path):
        """
        Trim joins from the start of the join path. The candidates for trim
        are the PathInfos in names_with_path structure that are m2m joins.

        Also set the select column so the start matches the join.

        This method is meant to be used for generating the subquery joins &
        cols in split_exclude().

        Return a lookup usable for doing outerq.filter(lookup=self) and a
        boolean indicating if the joins in the prefix contain a LEFT OUTER join.
        _"""
        all_paths = []
        for _, paths in names_with_path:
            all_paths.extend(paths)
        contains_louter = False
        # Trim and operate only on tables that were generated for
        # the lookup part of the query. That is, avoid trimming
        # joins generated for F() expressions.
        lookup_tables = [
            t for t in self.alias_map
            if t in self._lookup_joins or t == self.base_table
        ]
        for trimmed_paths, path in enumerate(all_paths):
            if path.m2m:
                break
            if self.alias_map[lookup_tables[trimmed_paths + 1]].join_type == LOUTER:
                contains_louter = True
            alias = lookup_tables[trimmed_paths]
            self.unref_alias(alias)
        # The path.join_field is a Rel, lets get the other side's field
        join_field = path.join_field.field
        # Build the filter prefix.
        paths_in_prefix = trimmed_paths
        trimmed_prefix = []
        for name, path in names_with_path:
            if paths_in_prefix - len(path) < 0:
                break
            trimmed_prefix.append(name)
            paths_in_prefix -= len(path)
        trimmed_prefix.append(
            join_field.foreign_related_fields[0].name)
        trimmed_prefix = LOOKUP_SEP.join(trimmed_prefix)
        # Lets still see if we can trim the first join from the inner query
        # (that is, self). We can't do this for:
        # - LEFT JOINs because we would miss those rows that have nothing on
        #   the outer side,
        # - INNER JOINs from filtered relations because we would miss their
        #   filters.
        first_join = self.alias_map[lookup_tables[trimmed_paths + 1]]
        if first_join.join_type != LOUTER and not first_join.filtered_relation:
            select_fields = [r[0] for r in join_field.related_fields]
            select_alias = lookup_tables[trimmed_paths + 1]
            self.unref_alias(lookup_tables[trimmed_paths])
            extra_restriction = join_field.get_extra_restriction(
                self.where_class, None, lookup_tables[trimmed_paths + 1])
            if extra_restriction:
                self.where.add(extra_restriction, AND)
        else:
            # TODO: It might be possible to trim more joins from the start of the
            # inner query if it happens to have a longer join chain containing the
            # values in select_fields. Lets punt this one for now.
            select_fields = [r[1] for r in join_field.related_fields]
            select_alias = lookup_tables[trimmed_paths]
        # The found starting point is likely a Join instead of a BaseTable reference.
        # But the first entry in the query's FROM clause must not be a JOIN.
        for table in self.alias_map:
            if self.alias_refcount[table] > 0:
                self.alias_map[table] = BaseTable(self.alias_map[table].table_name, table)
                break
        self.set_select([f.get_col(select_alias) for f in select_fields])
        return trimmed_prefix, contains_louter
```
### 44 - django/db/models/sql/query.py:

Start line: 1480, End line: 1565

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
### 45 - django/db/models/sql/query.py:

Start line: 902, End line: 924

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
### 48 - django/db/models/sql/query.py:

Start line: 1709, End line: 1748

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
### 50 - django/db/models/sql/query.py:

Start line: 1062, End line: 1091

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
### 54 - django/db/models/sql/query.py:

Start line: 2230, End line: 2262

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
### 57 - django/db/models/sql/query.py:

Start line: 1669, End line: 1707

```python
class Query(BaseExpression):

    def trim_joins(self, targets, joins, path):
        """
        The 'target' parameter is the final field being joined to, 'joins'
        is the full list of join aliases. The 'path' contain the PathInfos
        used to create the joins.

        Return the final target field and table alias and the new active
        joins.

        Always trim any direct join if the target column is already in the
        previous table. Can't trim reverse joins as it's unknown if there's
        anything on the other side of the join.
        """
        joins = joins[:]
        for pos, info in enumerate(reversed(path)):
            if len(joins) == 1 or not info.direct:
                break
            if info.filtered_relation:
                break
            join_targets = {t.column for t in info.join_field.foreign_related_fields}
            cur_targets = {t.column for t in targets}
            if not cur_targets.issubset(join_targets):
                break
            targets_dict = {r[1].column: r[0] for r in info.join_field.related_fields if r[1].column in cur_targets}
            targets = tuple(targets_dict[t.column] for t in targets)
            self.unref_alias(joins.pop())
        return targets, joins[-1], joins

    @classmethod
    def _gen_cols(cls, exprs):
        for expr in exprs:
            if isinstance(expr, Col):
                yield expr
            else:
                yield from cls._gen_cols(expr.get_source_expressions())

    @classmethod
    def _gen_col_aliases(cls, exprs):
        yield from (expr.alias for expr in cls._gen_cols(exprs))
```
### 58 - django/db/models/sql/query.py:

Start line: 776, End line: 808

```python
class Query(BaseExpression):

    def promote_joins(self, aliases):
        """
        Promote recursively the join type of given aliases and its children to
        an outer join. If 'unconditional' is False, only promote the join if
        it is nullable or the parent join is an outer join.

        The children promotion is done to avoid join chains that contain a LOUTER
        b INNER c. So, if we have currently a INNER b INNER c and a->b is promoted,
        then we must also promote b->c automatically, or otherwise the promotion
        of a->b doesn't actually change anything in the query results.
        """
        aliases = list(aliases)
        while aliases:
            alias = aliases.pop(0)
            if self.alias_map[alias].join_type is None:
                # This is the base table (first FROM entry) - this table
                # isn't really joined at all in the query, so we should not
                # alter its join type.
                continue
            # Only the first alias (skipped above) should have None join_type
            assert self.alias_map[alias].join_type is not None
            parent_alias = self.alias_map[alias].parent_alias
            parent_louter = parent_alias and self.alias_map[parent_alias].join_type == LOUTER
            already_louter = self.alias_map[alias].join_type == LOUTER
            if ((self.alias_map[alias].nullable or parent_louter) and
                    not already_louter):
                self.alias_map[alias] = self.alias_map[alias].promote()
                # Join type of 'alias' changed, so re-examine all aliases that
                # refer to this one.
                aliases.extend(
                    join for join in self.alias_map
                    if self.alias_map[join].parent_alias == alias and join not in aliases
                )
```
### 59 - django/db/models/sql/query.py:

Start line: 994, End line: 1025

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
### 60 - django/db/models/sql/query.py:

Start line: 1122, End line: 1154

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
### 65 - django/db/models/sql/query.py:

Start line: 810, End line: 836

```python
class Query(BaseExpression):

    def demote_joins(self, aliases):
        """
        Change join type from LOUTER to INNER for all joins in aliases.

        Similarly to promote_joins(), this method must ensure no join chains
        containing first an outer, then an inner join are generated. If we
        are demoting b->c join in chain a LOUTER b LOUTER c then we must
        demote a->b automatically, or otherwise the demotion of b->c doesn't
        actually change anything in the query results. .
        """
        aliases = list(aliases)
        while aliases:
            alias = aliases.pop(0)
            if self.alias_map[alias].join_type == LOUTER:
                self.alias_map[alias] = self.alias_map[alias].demote()
                parent_alias = self.alias_map[alias].parent_alias
                if self.alias_map[parent_alias].join_type == INNER:
                    aliases.append(parent_alias)

    def reset_refcounts(self, to_counts):
        """
        Reset reference counts for aliases so that they match the value passed
        in `to_counts`.
        """
        for alias, cur_refcount in self.alias_refcount.copy().items():
            unref_amount = cur_refcount - to_counts.get(alias, 0)
            self.unref_alias(alias, unref_amount)
```
### 67 - django/db/models/sql/query.py:

Start line: 1221, End line: 1299

```python
class Query(BaseExpression):

    def build_filter(self, filter_expr, branch_negated=False, current_negated=False,
                     can_reuse=None, allow_joins=True, split_subq=True,
                     reuse_with_filtered_relation=False, check_filterable=True):
        """
        Build a WhereNode for a single filter clause but don't add it
        to this Query. Query.add_q() will then add this filter to the where
        Node.

        The 'branch_negated' tells us if the current branch contains any
        negations. This will be used to determine if subqueries are needed.

        The 'current_negated' is used to determine if the current filter is
        negated or not and this will be used to determine if IS NULL filtering
        is needed.

        The difference between current_negated and branch_negated is that
        branch_negated is set on first negation, but current_negated is
        flipped for each negation.

        Note that add_filter will not do any negating itself, that is done
        upper in the code by add_q().

        The 'can_reuse' is a set of reusable joins for multijoins.

        If 'reuse_with_filtered_relation' is True, then only joins in can_reuse
        will be reused.

        The method will create a filter clause that can be added to the current
        query. However, if the filter isn't added to the query then the caller
        is responsible for unreffing the joins used.
        """
        if isinstance(filter_expr, dict):
            raise FieldError("Cannot parse keyword query as dict")
        if isinstance(filter_expr, Q):
            return self._add_q(
                filter_expr,
                branch_negated=branch_negated,
                current_negated=current_negated,
                used_aliases=can_reuse,
                allow_joins=allow_joins,
                split_subq=split_subq,
                check_filterable=check_filterable,
            )
        if hasattr(filter_expr, 'resolve_expression'):
            if not getattr(filter_expr, 'conditional', False):
                raise TypeError('Cannot filter against a non-conditional expression.')
            condition = self.build_lookup(
                ['exact'], filter_expr.resolve_expression(self, allow_joins=allow_joins), True
            )
            clause = self.where_class()
            clause.add(condition, AND)
            return clause, []
        arg, value = filter_expr
        if not arg:
            raise FieldError("Cannot parse keyword query %r" % arg)
        lookups, parts, reffed_expression = self.solve_lookup_type(arg)

        if check_filterable:
            self.check_filterable(reffed_expression)

        if not allow_joins and len(parts) > 1:
            raise FieldError("Joined field references are not permitted in this query")

        pre_joins = self.alias_refcount.copy()
        value = self.resolve_lookup_value(value, can_reuse, allow_joins)
        used_joins = {k for k, v in self.alias_refcount.items() if v > pre_joins.get(k, 0)}

        if check_filterable:
            self.check_filterable(value)

        clause = self.where_class()
        if reffed_expression:
            condition = self.build_lookup(lookups, reffed_expression, value)
            clause.add(condition, AND)
            return clause, []

        opts = self.get_meta()
        alias = self.get_initial_alias()
        allow_many = not branch_negated or not split_subq
        # ... other code
```
### 68 - django/db/models/sql/query.py:

Start line: 1391, End line: 1411

```python
class Query(BaseExpression):

    def _add_q(self, q_object, used_aliases, branch_negated=False,
               current_negated=False, allow_joins=True, split_subq=True,
               check_filterable=True):
        """Add a Q-object to the current filter."""
        connector = q_object.connector
        current_negated = current_negated ^ q_object.negated
        branch_negated = branch_negated or q_object.negated
        target_clause = self.where_class(connector=connector,
                                         negated=q_object.negated)
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
### 69 - django/db/models/sql/query.py:

Start line: 1413, End line: 1433

```python
class Query(BaseExpression):

    def build_filtered_relation_q(self, q_object, reuse, branch_negated=False, current_negated=False):
        """Add a FilteredRelation object to the current filter."""
        connector = q_object.connector
        current_negated ^= q_object.negated
        branch_negated = branch_negated or q_object.negated
        target_clause = self.where_class(connector=connector, negated=q_object.negated)
        for child in q_object.children:
            if isinstance(child, Node):
                child_clause = self.build_filtered_relation_q(
                    child, reuse=reuse, branch_negated=branch_negated,
                    current_negated=current_negated,
                )
            else:
                child_clause, _ = self.build_filter(
                    child, can_reuse=reuse, branch_negated=branch_negated,
                    current_negated=current_negated,
                    allow_joins=True, split_subq=False,
                    reuse_with_filtered_relation=True,
                )
            target_clause.add(child_clause, connector)
        return target_clause
```
### 70 - django/db/models/sql/query.py:

Start line: 2390, End line: 2416

```python
class JoinPromoter:
    """
    A class to abstract away join promotion problems for complex filter
    conditions.
    """

    def __init__(self, connector, num_children, negated):
        self.connector = connector
        self.negated = negated
        if self.negated:
            if connector == AND:
                self.effective_connector = OR
            else:
                self.effective_connector = AND
        else:
            self.effective_connector = self.connector
        self.num_children = num_children
        # Maps of table alias to how many times it is seen as required for
        # inner and/or outer joins.
        self.votes = Counter()

    def add_votes(self, votes):
        """
        Add single vote per item to self.votes. Parameter can be any
        iterable.
        """
        self.votes.update(votes)
```
### 72 - django/db/models/sql/query.py:

Start line: 297, End line: 346

```python
class Query(BaseExpression):

    def clone(self):
        """
        Return a copy of the current Query. A lightweight alternative to
        to deepcopy().
        """
        obj = Empty()
        obj.__class__ = self.__class__
        # Copy references to everything.
        obj.__dict__ = self.__dict__.copy()
        # Clone attributes that can't use shallow copy.
        obj.alias_refcount = self.alias_refcount.copy()
        obj.alias_map = self.alias_map.copy()
        obj.external_aliases = self.external_aliases.copy()
        obj.table_map = self.table_map.copy()
        obj.where = self.where.clone()
        obj.annotations = self.annotations.copy()
        if self.annotation_select_mask is None:
            obj.annotation_select_mask = None
        else:
            obj.annotation_select_mask = self.annotation_select_mask.copy()
        obj.combined_queries = tuple(query.clone() for query in self.combined_queries)
        # _annotation_select_cache cannot be copied, as doing so breaks the
        # (necessary) state in which both annotations and
        # _annotation_select_cache point to the same underlying objects.
        # It will get re-populated in the cloned queryset the next time it's
        # used.
        obj._annotation_select_cache = None
        obj.extra = self.extra.copy()
        if self.extra_select_mask is None:
            obj.extra_select_mask = None
        else:
            obj.extra_select_mask = self.extra_select_mask.copy()
        if self._extra_select_cache is None:
            obj._extra_select_cache = None
        else:
            obj._extra_select_cache = self._extra_select_cache.copy()
        if self.select_related is not False:
            # Use deepcopy because select_related stores fields in nested
            # dicts.
            obj.select_related = copy.deepcopy(obj.select_related)
        if 'subq_aliases' in self.__dict__:
            obj.subq_aliases = self.subq_aliases.copy()
        obj.used_aliases = self.used_aliases.copy()
        obj._filtered_relations = self._filtered_relations.copy()
        # Clear the cached_property
        try:
            del obj.base_table
        except AttributeError:
            pass
        return obj
```
### 75 - django/db/models/sql/query.py:

Start line: 348, End line: 371

```python
class Query(BaseExpression):

    def chain(self, klass=None):
        """
        Return a copy of the current Query that's ready for another operation.
        The klass argument changes the type of the Query, e.g. UpdateQuery.
        """
        obj = self.clone()
        if klass and obj.__class__ != klass:
            obj.__class__ = klass
        if not obj.filter_is_sticky:
            obj.used_aliases = set()
        obj.filter_is_sticky = False
        if hasattr(obj, '_setup_query'):
            obj._setup_query()
        return obj

    def relabeled_clone(self, change_map):
        clone = self.clone()
        clone.change_aliases(change_map)
        return clone

    def _get_col(self, target, field, alias):
        if not self.alias_cols:
            alias = None
        return target.get_col(alias, field)
```
### 84 - django/db/models/sql/query.py:

Start line: 425, End line: 512

```python
class Query(BaseExpression):

    def get_aggregation(self, using, added_aggregate_names):
        """
        Return the dictionary with the values of the existing aggregations.
        """
        if not self.annotation_select:
            return {}
        existing_annotations = [
            annotation for alias, annotation
            in self.annotations.items()
            if alias not in added_aggregate_names
        ]
        # Decide if we need to use a subquery.
        #
        # Existing annotations would cause incorrect results as get_aggregation()
        # must produce just one result and thus must not use GROUP BY. But we
        # aren't smart enough to remove the existing annotations from the
        # query, so those would force us to use GROUP BY.
        #
        # If the query has limit or distinct, or uses set operations, then
        # those operations must be done in a subquery so that the query
        # aggregates on the limit and/or distinct results instead of applying
        # the distinct and limit after the aggregation.
        if (isinstance(self.group_by, tuple) or self.is_sliced or existing_annotations or
                self.distinct or self.combinator):
            from django.db.models.sql.subqueries import AggregateQuery
            inner_query = self.clone()
            inner_query.subquery = True
            outer_query = AggregateQuery(self.model, inner_query)
            inner_query.select_for_update = False
            inner_query.select_related = False
            inner_query.set_annotation_mask(self.annotation_select)
            if not self.is_sliced and not self.distinct_fields:
                # Queries with distinct_fields need ordering and when a limit
                # is applied we must take the slice from the ordered query.
                # Otherwise no need for ordering.
                inner_query.clear_ordering(True)
            if not inner_query.distinct:
                # If the inner query uses default select and it has some
                # aggregate annotations, then we must make sure the inner
                # query is grouped by the main model's primary key. However,
                # clearing the select clause can alter results if distinct is
                # used.
                has_existing_aggregate_annotations = any(
                    annotation for annotation in existing_annotations
                    if getattr(annotation, 'contains_aggregate', True)
                )
                if inner_query.default_cols and has_existing_aggregate_annotations:
                    inner_query.group_by = (self.model._meta.pk.get_col(inner_query.get_initial_alias()),)
                inner_query.default_cols = False

            relabels = {t: 'subquery' for t in inner_query.alias_map}
            relabels[None] = 'subquery'
            # Remove any aggregates marked for reduction from the subquery
            # and move them to the outer AggregateQuery.
            col_cnt = 0
            for alias, expression in list(inner_query.annotation_select.items()):
                annotation_select_mask = inner_query.annotation_select_mask
                if expression.is_summary:
                    expression, col_cnt = inner_query.rewrite_cols(expression, col_cnt)
                    outer_query.annotations[alias] = expression.relabeled_clone(relabels)
                    del inner_query.annotations[alias]
                    annotation_select_mask.remove(alias)
                # Make sure the annotation_select wont use cached results.
                inner_query.set_annotation_mask(inner_query.annotation_select_mask)
            if inner_query.select == () and not inner_query.default_cols and not inner_query.annotation_select_mask:
                # In case of Model.objects[0:3].count(), there would be no
                # field selected in the inner query, yet we must use a subquery.
                # So, make sure at least one field is selected.
                inner_query.select = (self.model._meta.pk.get_col(inner_query.get_initial_alias()),)
        else:
            outer_query = self
            self.select = ()
            self.default_cols = False
            self.extra = {}

        outer_query.clear_ordering(True)
        outer_query.clear_limits()
        outer_query.select_for_update = False
        outer_query.select_related = False
        compiler = outer_query.get_compiler(using)
        result = compiler.execute_sql(SINGLE)
        if result is None:
            result = [None] * len(outer_query.annotation_select)

        converters = compiler.get_converters(outer_query.annotation_select.values())
        result = next(compiler.apply_converters((result,), converters))

        return dict(zip(outer_query.annotation_select, result))
```
### 88 - django/db/models/sql/query.py:

Start line: 946, End line: 992

```python
class Query(BaseExpression):

    def join(self, join, reuse=None, reuse_with_filtered_relation=False):
        """
        Return an alias for the 'join', either reusing an existing alias for
        that join or creating a new one. 'join' is either a
        sql.datastructures.BaseTable or Join.

        The 'reuse' parameter can be either None which means all joins are
        reusable, or it can be a set containing the aliases that can be reused.

        The 'reuse_with_filtered_relation' parameter is used when computing
        FilteredRelation instances.

        A join is always created as LOUTER if the lhs alias is LOUTER to make
        sure chains like t1 LOUTER t2 INNER t3 aren't generated. All new
        joins are created as LOUTER if the join is nullable.
        """
        if reuse_with_filtered_relation and reuse:
            reuse_aliases = [
                a for a, j in self.alias_map.items()
                if a in reuse and j.equals(join, with_filtered_relation=False)
            ]
        else:
            reuse_aliases = [
                a for a, j in self.alias_map.items()
                if (reuse is None or a in reuse) and j == join
            ]
        if reuse_aliases:
            if join.table_alias in reuse_aliases:
                reuse_alias = join.table_alias
            else:
                # Reuse the most recent alias of the joined table
                # (a many-to-many relation may be joined multiple times).
                reuse_alias = reuse_aliases[-1]
            self.ref_alias(reuse_alias)
            return reuse_alias

        # No reuse is possible, so we need a new alias.
        alias, _ = self.table_alias(join.table_name, create=True, filtered_relation=join.filtered_relation)
        if join.join_type:
            if self.alias_map[join.parent_alias].join_type == LOUTER or join.nullable:
                join_type = LOUTER
            else:
                join_type = INNER
            join.join_type = join_type
        join.table_alias = alias
        self.alias_map[alias] = join
        return alias
```
### 89 - django/db/models/expressions.py:

Start line: 929, End line: 993

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
### 90 - django/db/models/sql/query.py:

Start line: 1156, End line: 1199

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
### 97 - django/db/models/sql/query.py:

Start line: 1899, End line: 1940

```python
class Query(BaseExpression):

    def add_fields(self, field_names, allow_m2m=True):
        """
        Add the given (model) fields to the select set. Add the field names in
        the order specified.
        """
        alias = self.get_initial_alias()
        opts = self.get_meta()

        try:
            cols = []
            for name in field_names:
                # Join promotion note - we must not remove any rows here, so
                # if there is no existing joins, use outer join.
                join_info = self.setup_joins(name.split(LOOKUP_SEP), opts, alias, allow_many=allow_m2m)
                targets, final_alias, joins = self.trim_joins(
                    join_info.targets,
                    join_info.joins,
                    join_info.path,
                )
                for target in targets:
                    cols.append(join_info.transform_function(target, final_alias))
            if cols:
                self.set_select(cols)
        except MultiJoin:
            raise FieldError("Invalid field name: '%s'" % name)
        except FieldError:
            if LOOKUP_SEP in name:
                # For lookups spanning over relationships, show the error
                # from the model on which the lookup failed.
                raise
            elif name in self.annotations:
                raise FieldError(
                    "Cannot select the '%s' alias. Use annotate() to promote "
                    "it." % name
                )
            else:
                names = sorted([
                    *get_field_names_from_opts(opts), *self.extra,
                    *self.annotation_select, *self._filtered_relations
                ])
                raise FieldError("Cannot resolve keyword %r into field. "
                                 "Choices are: %s" % (name, ", ".join(names)))
```
### 98 - django/db/models/expressions.py:

Start line: 1218, End line: 1247

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
### 101 - django/db/models/sql/query.py:

Start line: 2087, End line: 2109

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
### 104 - django/db/models/expressions.py:

Start line: 489, End line: 514

```python
class CombinedExpression(SQLiteNumericMixin, Expression):

    def resolve_expression(self, query=None, allow_joins=True, reuse=None, summarize=False, for_save=False):
        lhs = self.lhs.resolve_expression(query, allow_joins, reuse, summarize, for_save)
        rhs = self.rhs.resolve_expression(query, allow_joins, reuse, summarize, for_save)
        if not isinstance(self, (DurationExpression, TemporalSubtraction)):
            try:
                lhs_type = lhs.output_field.get_internal_type()
            except (AttributeError, FieldError):
                lhs_type = None
            try:
                rhs_type = rhs.output_field.get_internal_type()
            except (AttributeError, FieldError):
                rhs_type = None
            if 'DurationField' in {lhs_type, rhs_type} and lhs_type != rhs_type:
                return DurationExpression(self.lhs, self.connector, self.rhs).resolve_expression(
                    query, allow_joins, reuse, summarize, for_save,
                )
            datetime_fields = {'DateField', 'DateTimeField', 'TimeField'}
            if self.connector == self.SUB and lhs_type in datetime_fields and lhs_type == rhs_type:
                return TemporalSubtraction(self.lhs, self.rhs).resolve_expression(
                    query, allow_joins, reuse, summarize, for_save,
                )
        c = self.copy()
        c.is_summary = summarize
        c.lhs = lhs
        c.rhs = rhs
        return c
```
### 105 - django/db/models/sql/query.py:

Start line: 233, End line: 295

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

    @property
    def identity(self):
        identity = (
            (arg, make_hashable(value))
            for arg, value in self.__dict__.items()
        )
        return (self.__class__, *identity)

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
### 106 - django/db/models/sql/query.py:

Start line: 1567, End line: 1642

```python
class Query(BaseExpression):

    def setup_joins(self, names, opts, alias, can_reuse=None, allow_many=True,
                    reuse_with_filtered_relation=False):
        """
        Compute the necessary table joins for the passage through the fields
        given in 'names'. 'opts' is the Options class for the current model
        (which gives the table we are starting from), 'alias' is the alias for
        the table to start the joining from.

        The 'can_reuse' defines the reverse foreign key joins we can reuse. It
        can be None in which case all joins are reusable or a set of aliases
        that can be reused. Note that non-reverse foreign keys are always
        reusable when using setup_joins().

        The 'reuse_with_filtered_relation' can be used to force 'can_reuse'
        parameter and force the relation on the given connections.

        If 'allow_many' is False, then any reverse foreign key seen will
        generate a MultiJoin exception.

        Return the final field involved in the joins, the target field (used
        for any 'where' constraint), the final 'opts' value, the joins, the
        field path traveled to generate the joins, and a transform function
        that takes a field and alias and is equivalent to `field.get_col(alias)`
        in the simple case but wraps field transforms if they were included in
        names.

        The target field is the field containing the concrete value. Final
        field can be something different, for example foreign key pointing to
        that value. Final field is needed for example in some value
        conversions (convert 'obj' in fk__id=obj to pk val using the foreign
        key field for example).
        """
        joins = [alias]
        # The transform can't be applied yet, as joins must be trimmed later.
        # To avoid making every caller of this method look up transforms
        # directly, compute transforms here and create a partial that converts
        # fields to the appropriate wrapped version.

        def final_transformer(field, alias):
            return field.get_col(alias)

        # Try resolving all the names as fields first. If there's an error,
        # treat trailing names as lookups until a field can be resolved.
        last_field_exception = None
        for pivot in range(len(names), 0, -1):
            try:
                path, final_field, targets, rest = self.names_to_path(
                    names[:pivot], opts, allow_many, fail_on_missing=True,
                )
            except FieldError as exc:
                if pivot == 1:
                    # The first item cannot be a lookup, so it's safe
                    # to raise the field error here.
                    raise
                else:
                    last_field_exception = exc
            else:
                # The transforms are the remaining items that couldn't be
                # resolved into fields.
                transforms = names[pivot:]
                break
        for name in transforms:
            def transform(field, alias, *, name, previous):
                try:
                    wrapped = previous(field, alias)
                    return self.try_transform(wrapped, name)
                except FieldError:
                    # FieldError is raised if the transform doesn't exist.
                    if isinstance(final_field, Field) and last_field_exception:
                        raise last_field_exception
                    else:
                        raise
            final_transformer = functools.partial(transform, name=name, previous=final_transformer)
        # Then, add the path to the query's joins. Note that we can't trim
        # joins at this stage - we will need the information about join type
        # of the trimmed joins.
        # ... other code
```
### 109 - django/db/models/expressions.py:

Start line: 795, End line: 811

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
### 113 - django/db/models/sql/query.py:

Start line: 657, End line: 704

```python
class Query(BaseExpression):

    def deferred_to_data(self, target, callback):
        """
        Convert the self.deferred_loading data structure to an alternate data
        structure, describing the field that *will* be loaded. This is used to
        compute the columns to select from the database and also by the
        QuerySet class to work out which fields are being initialized on each
        model. Models that have all their fields included aren't mentioned in
        the result, only those that have field restrictions in place.

        The "target" parameter is the instance that is populated (in place).
        The "callback" is a function that is called whenever a (model, field)
        pair need to be added to "target". It accepts three parameters:
        "target", and the model and list of fields being added for that model.
        """
        field_names, defer = self.deferred_loading
        if not field_names:
            return
        orig_opts = self.get_meta()
        seen = {}
        must_include = {orig_opts.concrete_model: {orig_opts.pk}}
        for field_name in field_names:
            parts = field_name.split(LOOKUP_SEP)
            cur_model = self.model._meta.concrete_model
            opts = orig_opts
            for name in parts[:-1]:
                old_model = cur_model
                if name in self._filtered_relations:
                    name = self._filtered_relations[name].relation_name
                source = opts.get_field(name)
                if is_reverse_o2o(source):
                    cur_model = source.related_model
                else:
                    cur_model = source.remote_field.model
                opts = cur_model._meta
                # Even if we're "just passing through" this model, we must add
                # both the current model's pk and the related reference field
                # (if it's not a reverse relation) to the things we select.
                if not is_reverse_o2o(source):
                    must_include[old_model].add(source)
                add_to_dict(must_include, cur_model, opts.pk)
            field = opts.get_field(parts[-1])
            is_reverse_object = field.auto_created and not field.concrete
            model = field.related_model if is_reverse_object else field.model
            model = model._meta.concrete_model
            if model == opts.model:
                model = cur_model
            if not is_reverse_o2o(field):
                add_to_dict(seen, model, field)
        # ... other code
```
### 121 - django/db/models/expressions.py:

Start line: 442, End line: 487

```python
class CombinedExpression(SQLiteNumericMixin, Expression):

    def __init__(self, lhs, connector, rhs, output_field=None):
        super().__init__(output_field=output_field)
        self.connector = connector
        self.lhs = lhs
        self.rhs = rhs

    def __repr__(self):
        return "<{}: {}>".format(self.__class__.__name__, self)

    def __str__(self):
        return "{} {} {}".format(self.lhs, self.connector, self.rhs)

    def get_source_expressions(self):
        return [self.lhs, self.rhs]

    def set_source_expressions(self, exprs):
        self.lhs, self.rhs = exprs

    def _resolve_output_field(self):
        try:
            return super()._resolve_output_field()
        except FieldError:
            combined_type = _resolve_combined_type(
                self.connector,
                type(self.lhs.output_field),
                type(self.rhs.output_field),
            )
            if combined_type is None:
                raise
            return combined_type()

    def as_sql(self, compiler, connection):
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
```
### 125 - django/db/models/sql/query.py:

Start line: 1027, End line: 1060

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
        # It's safe to drop ordering if the queryset isn't using slicing,
        # distinct(*fields) or select_for_update().
        if (self.low_mark == 0 and self.high_mark is None and
                not self.distinct_fields and
                not self.select_for_update):
            clone.clear_ordering(True)
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
### 126 - django/db/models/sql/query.py:

Start line: 1942, End line: 1986

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
            if isinstance(item, str):
                if '.' in item:
                    warnings.warn(
                        'Passing column raw column aliases to order_by() is '
                        'deprecated. Wrap %r in a RawSQL expression before '
                        'passing it to order_by().' % item,
                        category=RemovedInDjango40Warning,
                        stacklevel=3,
                    )
                    continue
                if item == '?':
                    continue
                if item.startswith('-'):
                    item = item[1:]
                if item in self.annotations:
                    continue
                if self.extra and item in self.extra:
                    continue
                # names_to_path() validates the lookup. A descriptive
                # FieldError will be raise if it's not.
                self.names_to_path(item.split(LOOKUP_SEP), self.model._meta)
            elif not hasattr(item, 'resolve_expression'):
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
```
### 133 - django/db/models/sql/query.py:

Start line: 926, End line: 944

```python
class Query(BaseExpression):

    def get_initial_alias(self):
        """
        Return the first alias for this query, after increasing its reference
        count.
        """
        if self.alias_map:
            alias = self.base_table
            self.ref_alias(alias)
        else:
            alias = self.join(BaseTable(self.get_meta().db_table, None))
        return alias

    def count_active_tables(self):
        """
        Return the number of tables in this query with a non-zero reference
        count. After execution, the reference counts are zeroed, so tables
        added in compiler will not be seen by this method.
        """
        return len([1 for count in self.alias_refcount.values() if count])
```
### 136 - django/db/models/expressions.py:

Start line: 814, End line: 848

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
### 137 - django/db/models/sql/query.py:

Start line: 743, End line: 774

```python
class Query(BaseExpression):

    def table_alias(self, table_name, create=False, filtered_relation=None):
        """
        Return a table alias for the given table_name and whether this is a
        new alias or not.

        If 'create' is true, a new alias is always created. Otherwise, the
        most recently created alias for the table (if one exists) is reused.
        """
        alias_list = self.table_map.get(table_name)
        if not create and alias_list:
            alias = alias_list[0]
            self.alias_refcount[alias] += 1
            return alias, False

        # Create a new alias for this table.
        if alias_list:
            alias = '%s%d' % (self.alias_prefix, len(self.alias_map) + 1)
            alias_list.append(alias)
        else:
            # The first occurrence of a table uses the table name directly.
            alias = filtered_relation.alias if filtered_relation is not None else table_name
            self.table_map[table_name] = [alias]
        self.alias_refcount[alias] = 1
        return alias, True

    def ref_alias(self, alias):
        """Increases the reference count for this alias."""
        self.alias_refcount[alias] += 1

    def unref_alias(self, alias, amount=1):
        """Decreases the reference count for this alias."""
        self.alias_refcount[alias] -= amount
```
### 138 - django/db/models/expressions.py:

Start line: 547, End line: 588

```python
class TemporalSubtraction(CombinedExpression):
    output_field = fields.DurationField()

    def __init__(self, lhs, rhs):
        super().__init__(lhs, self.SUB, rhs)

    def as_sql(self, compiler, connection):
        connection.ops.check_expression_support(self)
        lhs = compiler.compile(self.lhs)
        rhs = compiler.compile(self.rhs)
        return connection.ops.subtract_temporals(self.lhs.output_field.get_internal_type(), lhs, rhs)


@deconstructible
class F(Combinable):
    """An object capable of resolving references to existing query objects."""

    def __init__(self, name):
        """
        Arguments:
         * name: the name of the field this expression references
        """
        self.name = name

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.name)

    def resolve_expression(self, query=None, allow_joins=True, reuse=None,
                           summarize=False, for_save=False):
        return query.resolve_ref(self.name, allow_joins, reuse, summarize)

    def asc(self, **kwargs):
        return OrderBy(self, **kwargs)

    def desc(self, **kwargs):
        return OrderBy(self, descending=True, **kwargs)

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.name == other.name

    def __hash__(self):
        return hash(self.name)
```
### 144 - django/db/models/sql/query.py:

Start line: 69, End line: 117

```python
class RawQuery:
    """A single raw SQL query."""

    def __init__(self, sql, using, params=None):
        self.params = params or ()
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
        return dict if isinstance(self.params, Mapping) else tuple

    def __str__(self):
        return self.sql % self.params_type(self.params)
```
### 145 - django/db/models/expressions.py:

Start line: 1191, End line: 1216

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
            if self.nulls_last and not (
                self.descending and connection.features.order_by_nulls_first
            ):
                template = '%%(expression)s IS NULL, %s' % template
            elif self.nulls_first and not (
                not self.descending and connection.features.order_by_nulls_first
            ):
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
### 159 - django/db/models/expressions.py:

Start line: 591, End line: 630

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
