# django__django-14480

| **django/django** | `795da6306a048b18c0158496b0d49e8e4f197a32` |
| ---- | ---- |
| **No of patches** | 7 |
| **All found context length** | 8650 |
| **Any found context length** | 1038 |
| **Avg pos** | 122.42857142857143 |
| **Min pos** | 4 |
| **Max pos** | 120 |
| **Top file pos** | 4 |
| **Missing snippets** | 13 |
| **Missing patch files** | 1 |


## Expected patch

```diff
diff --git a/django/db/backends/base/features.py b/django/db/backends/base/features.py
--- a/django/db/backends/base/features.py
+++ b/django/db/backends/base/features.py
@@ -325,6 +325,9 @@ class BaseDatabaseFeatures:
     # Does the backend support non-deterministic collations?
     supports_non_deterministic_collations = True
 
+    # Does the backend support the logical XOR operator?
+    supports_logical_xor = False
+
     # Collation names for use by the Django test suite.
     test_collations = {
         "ci": None,  # Case-insensitive.
diff --git a/django/db/backends/mysql/features.py b/django/db/backends/mysql/features.py
--- a/django/db/backends/mysql/features.py
+++ b/django/db/backends/mysql/features.py
@@ -47,6 +47,7 @@ class DatabaseFeatures(BaseDatabaseFeatures):
 
     supports_order_by_nulls_modifier = False
     order_by_nulls_first = True
+    supports_logical_xor = True
 
     @cached_property
     def minimum_database_version(self):
diff --git a/django/db/models/expressions.py b/django/db/models/expressions.py
--- a/django/db/models/expressions.py
+++ b/django/db/models/expressions.py
@@ -94,7 +94,7 @@ def __and__(self, other):
         if getattr(self, "conditional", False) and getattr(other, "conditional", False):
             return Q(self) & Q(other)
         raise NotImplementedError(
-            "Use .bitand() and .bitor() for bitwise logical operations."
+            "Use .bitand(), .bitor(), and .bitxor() for bitwise logical operations."
         )
 
     def bitand(self, other):
@@ -106,6 +106,13 @@ def bitleftshift(self, other):
     def bitrightshift(self, other):
         return self._combine(other, self.BITRIGHTSHIFT, False)
 
+    def __xor__(self, other):
+        if getattr(self, "conditional", False) and getattr(other, "conditional", False):
+            return Q(self) ^ Q(other)
+        raise NotImplementedError(
+            "Use .bitand(), .bitor(), and .bitxor() for bitwise logical operations."
+        )
+
     def bitxor(self, other):
         return self._combine(other, self.BITXOR, False)
 
@@ -113,7 +120,7 @@ def __or__(self, other):
         if getattr(self, "conditional", False) and getattr(other, "conditional", False):
             return Q(self) | Q(other)
         raise NotImplementedError(
-            "Use .bitand() and .bitor() for bitwise logical operations."
+            "Use .bitand(), .bitor(), and .bitxor() for bitwise logical operations."
         )
 
     def bitor(self, other):
@@ -139,12 +146,17 @@ def __rpow__(self, other):
 
     def __rand__(self, other):
         raise NotImplementedError(
-            "Use .bitand() and .bitor() for bitwise logical operations."
+            "Use .bitand(), .bitor(), and .bitxor() for bitwise logical operations."
         )
 
     def __ror__(self, other):
         raise NotImplementedError(
-            "Use .bitand() and .bitor() for bitwise logical operations."
+            "Use .bitand(), .bitor(), and .bitxor() for bitwise logical operations."
+        )
+
+    def __rxor__(self, other):
+        raise NotImplementedError(
+            "Use .bitand(), .bitor(), and .bitxor() for bitwise logical operations."
         )
 
 
diff --git a/django/db/models/query.py b/django/db/models/query.py
--- a/django/db/models/query.py
+++ b/django/db/models/query.py
@@ -396,6 +396,25 @@ def __or__(self, other):
         combined.query.combine(other.query, sql.OR)
         return combined
 
+    def __xor__(self, other):
+        self._check_operator_queryset(other, "^")
+        self._merge_sanity_check(other)
+        if isinstance(self, EmptyQuerySet):
+            return other
+        if isinstance(other, EmptyQuerySet):
+            return self
+        query = (
+            self
+            if self.query.can_filter()
+            else self.model._base_manager.filter(pk__in=self.values("pk"))
+        )
+        combined = query._chain()
+        combined._merge_known_related_objects(other)
+        if not other.query.can_filter():
+            other = other.model._base_manager.filter(pk__in=other.values("pk"))
+        combined.query.combine(other.query, sql.XOR)
+        return combined
+
     ####################################
     # METHODS THAT DO DATABASE QUERIES #
     ####################################
diff --git a/django/db/models/query_utils.py b/django/db/models/query_utils.py
--- a/django/db/models/query_utils.py
+++ b/django/db/models/query_utils.py
@@ -38,6 +38,7 @@ class Q(tree.Node):
     # Connection types
     AND = "AND"
     OR = "OR"
+    XOR = "XOR"
     default = AND
     conditional = True
 
@@ -70,6 +71,9 @@ def __or__(self, other):
     def __and__(self, other):
         return self._combine(other, self.AND)
 
+    def __xor__(self, other):
+        return self._combine(other, self.XOR)
+
     def __invert__(self):
         obj = type(self)()
         obj.add(self, self.AND)
diff --git a/django/db/models/sql/__init__.py b/django/db/models/sql/__init__.py
--- a/django/db/models/sql/__init__.py
+++ b/django/db/models/sql/__init__.py
@@ -1,6 +1,6 @@
 from django.db.models.sql.query import *  # NOQA
 from django.db.models.sql.query import Query
 from django.db.models.sql.subqueries import *  # NOQA
-from django.db.models.sql.where import AND, OR
+from django.db.models.sql.where import AND, OR, XOR
 
-__all__ = ["Query", "AND", "OR"]
+__all__ = ["Query", "AND", "OR", "XOR"]
diff --git a/django/db/models/sql/where.py b/django/db/models/sql/where.py
--- a/django/db/models/sql/where.py
+++ b/django/db/models/sql/where.py
@@ -1,14 +1,19 @@
 """
 Code to manage the creation and SQL rendering of 'where' constraints.
 """
+import operator
+from functools import reduce
 
 from django.core.exceptions import EmptyResultSet
+from django.db.models.expressions import Case, When
+from django.db.models.lookups import Exact
 from django.utils import tree
 from django.utils.functional import cached_property
 
 # Connection types
 AND = "AND"
 OR = "OR"
+XOR = "XOR"
 
 
 class WhereNode(tree.Node):
@@ -39,10 +44,12 @@ def split_having(self, negated=False):
         if not self.contains_aggregate:
             return self, None
         in_negated = negated ^ self.negated
-        # If the effective connector is OR and this node contains an aggregate,
-        # then we need to push the whole branch to HAVING clause.
-        may_need_split = (in_negated and self.connector == AND) or (
-            not in_negated and self.connector == OR
+        # If the effective connector is OR or XOR and this node contains an
+        # aggregate, then we need to push the whole branch to HAVING clause.
+        may_need_split = (
+            (in_negated and self.connector == AND)
+            or (not in_negated and self.connector == OR)
+            or self.connector == XOR
         )
         if may_need_split and self.contains_aggregate:
             return None, self
@@ -85,6 +92,21 @@ def as_sql(self, compiler, connection):
         else:
             full_needed, empty_needed = 1, len(self.children)
 
+        if self.connector == XOR and not connection.features.supports_logical_xor:
+            # Convert if the database doesn't support XOR:
+            #   a XOR b XOR c XOR ...
+            # to:
+            #   (a OR b OR c OR ...) AND (a + b + c + ...) == 1
+            lhs = self.__class__(self.children, OR)
+            rhs_sum = reduce(
+                operator.add,
+                (Case(When(c, then=1), default=0) for c in self.children),
+            )
+            rhs = Exact(1, rhs_sum)
+            return self.__class__([lhs, rhs], AND, self.negated).as_sql(
+                compiler, connection
+            )
+
         for child in self.children:
             try:
                 sql, params = compiler.compile(child)

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/db/backends/base/features.py | 328 | 328 | 24 | 12 | 8650
| django/db/backends/mysql/features.py | 50 | 50 | 112 | 35 | 36921
| django/db/models/expressions.py | 97 | 97 | 120 | 5 | 39502
| django/db/models/expressions.py | 109 | 109 | 120 | 5 | 39502
| django/db/models/expressions.py | 116 | 116 | 120 | 5 | 39502
| django/db/models/expressions.py | 142 | 147 | 120 | 5 | 39502
| django/db/models/query.py | 399 | 399 | 81 | 6 | 27977
| django/db/models/query_utils.py | 41 | 41 | 73 | 4 | 24559
| django/db/models/query_utils.py | 73 | 73 | 4 | 4 | 1038
| django/db/models/sql/__init__.py | 4 | 6 | - | - | -
| django/db/models/sql/where.py | 4 | 4 | 83 | 18 | 28263
| django/db/models/sql/where.py | 42 | 45 | - | 18 | -
| django/db/models/sql/where.py | 88 | 88 | - | 18 | -


## Problem Statement

```
Add logical XOR support to Q() and QuerySet().
Description
	
XOR seems to be available in ​Postgresql, ​MySQL, ​SequelServer and ​Oracle but NOT ​sqlite. Two stackoverflow questions cover this sort of thing: ​https://stackoverflow.com/questions/50408142/django-models-xor-at-the-model-level and ​https://stackoverflow.com/questions/14711203/perform-a-logical-exclusive-or-on-a-django-q-object.
I propose adding XOR to work with Q queries like the ​answer to the second question above. This will be my first time making a major contribution so we'll see how this goes (apologies in advance if this is annoying!).

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 django/contrib/postgres/search.py | 147 | 174| 248 | 248 | 2385 | 
| 2 | 2 django/contrib/postgres/aggregates/general.py | 60 | 95| 233 | 481 | 3205 | 
| 3 | 3 django/db/models/sql/query.py | 1505 | 1537| 262 | 743 | 26158 | 
| **-> 4 <-** | **4 django/db/models/query_utils.py** | 67 | 104| 295 | 1038 | 28663 | 
| 5 | **5 django/db/models/expressions.py** | 1311 | 1351| 305 | 1343 | 40353 | 
| 6 | **6 django/db/models/query.py** | 365 | 397| 249 | 1592 | 59173 | 
| 7 | 6 django/db/models/sql/query.py | 1477 | 1503| 263 | 1855 | 59173 | 
| 8 | 7 django/db/backends/postgresql/features.py | 1 | 109| 872 | 2727 | 60045 | 
| 9 | 8 django/db/backends/oracle/operations.py | 617 | 633| 291 | 3018 | 66172 | 
| 10 | 8 django/db/models/sql/query.py | 1 | 67| 455 | 3473 | 66172 | 
| 11 | **8 django/db/models/query.py** | 1250 | 1263| 126 | 3599 | 66172 | 
| 12 | 8 django/db/models/sql/query.py | 1395 | 1475| 795 | 4394 | 66172 | 
| 13 | **8 django/db/models/query.py** | 1239 | 1248| 126 | 4520 | 66172 | 
| 14 | 8 django/contrib/postgres/search.py | 177 | 220| 322 | 4842 | 66172 | 
| 15 | **8 django/db/models/query.py** | 1024 | 1061| 280 | 5122 | 66172 | 
| 16 | 8 django/db/models/sql/query.py | 2223 | 2257| 295 | 5417 | 66172 | 
| 17 | 8 django/db/models/sql/query.py | 1793 | 1819| 207 | 5624 | 66172 | 
| 18 | 9 django/contrib/postgres/constraints.py | 127 | 160| 248 | 5872 | 67783 | 
| 19 | **9 django/db/models/query.py** | 1226 | 1237| 120 | 5992 | 67783 | 
| 20 | 9 django/db/models/sql/query.py | 661 | 701| 383 | 6375 | 67783 | 
| 21 | 10 django/db/models/__init__.py | 1 | 116| 682 | 7057 | 68465 | 
| 22 | 11 django/db/models/sql/compiler.py | 1411 | 1434| 247 | 7304 | 83678 | 
| 23 | 11 django/contrib/postgres/constraints.py | 105 | 125| 202 | 7506 | 83678 | 
| **-> 24 <-** | **12 django/db/backends/base/features.py** | 220 | 351| 1144 | 8650 | 86714 | 
| 25 | 13 django/contrib/postgres/lookups.py | 1 | 66| 365 | 9015 | 87079 | 
| 26 | 14 django/contrib/gis/db/backends/oracle/operations.py | 56 | 116| 708 | 9723 | 89181 | 
| 27 | 14 django/db/models/sql/query.py | 752 | 787| 389 | 10112 | 89181 | 
| 28 | 14 django/db/models/sql/query.py | 960 | 987| 272 | 10384 | 89181 | 
| 29 | 15 django/db/models/fields/json.py | 208 | 232| 241 | 10625 | 93356 | 
| 30 | 15 django/db/models/sql/query.py | 1567 | 1596| 287 | 10912 | 93356 | 
| 31 | 15 django/contrib/postgres/constraints.py | 20 | 103| 686 | 11598 | 93356 | 
| 32 | 16 django/db/models/sql/subqueries.py | 1 | 45| 311 | 11909 | 94583 | 
| 33 | **16 django/db/models/query.py** | 1658 | 1714| 453 | 12362 | 94583 | 
| 34 | 16 django/db/backends/oracle/operations.py | 680 | 711| 314 | 12676 | 94583 | 
| 35 | 17 django/db/backends/oracle/features.py | 77 | 137| 563 | 13239 | 95785 | 
| 36 | **18 django/db/models/sql/where.py** | 262 | 279| 141 | 13380 | 97757 | 
| 37 | 18 django/contrib/gis/db/backends/oracle/operations.py | 39 | 53| 208 | 13588 | 97757 | 
| 38 | 19 django/contrib/gis/db/backends/mysql/operations.py | 57 | 91| 217 | 13805 | 98641 | 
| 39 | 19 django/db/backends/oracle/features.py | 1 | 75| 644 | 14449 | 98641 | 
| 40 | 19 django/contrib/postgres/search.py | 1 | 31| 212 | 14661 | 98641 | 
| 41 | **19 django/db/models/expressions.py** | 221 | 267| 304 | 14965 | 98641 | 
| 42 | 20 django/contrib/gis/db/models/functions.py | 517 | 545| 225 | 15190 | 102678 | 
| 43 | 20 django/db/models/sql/query.py | 1153 | 1171| 154 | 15344 | 102678 | 
| 44 | 20 django/db/models/sql/query.py | 1124 | 1151| 222 | 15566 | 102678 | 
| 45 | **20 django/db/models/query.py** | 1485 | 1514| 185 | 15751 | 102678 | 
| 46 | 21 django/db/models/functions/text.py | 1 | 39| 266 | 16017 | 105047 | 
| 47 | 22 django/db/models/lookups.py | 150 | 185| 273 | 16290 | 110354 | 
| 48 | 22 django/db/models/lookups.py | 311 | 340| 235 | 16525 | 110354 | 
| 49 | 22 django/db/models/sql/query.py | 1205 | 1239| 339 | 16864 | 110354 | 
| 50 | 22 django/contrib/postgres/constraints.py | 1 | 17| 131 | 16995 | 110354 | 
| 51 | 23 django/db/backends/mysql/operations.py | 300 | 311| 165 | 17160 | 114574 | 
| 52 | 23 django/db/models/sql/compiler.py | 506 | 573| 606 | 17766 | 114574 | 
| 53 | 23 django/db/models/sql/compiler.py | 28 | 63| 360 | 18126 | 114574 | 
| 54 | 24 django/db/models/fields/related_lookups.py | 170 | 210| 250 | 18376 | 116177 | 
| 55 | **24 django/db/models/query.py** | 998 | 1022| 207 | 18583 | 116177 | 
| 56 | 24 django/db/models/fields/json.py | 172 | 206| 283 | 18866 | 116177 | 
| 57 | 24 django/db/models/sql/query.py | 2022 | 2071| 332 | 19198 | 116177 | 
| 58 | 24 django/db/models/sql/query.py | 1614 | 1710| 829 | 20027 | 116177 | 
| 59 | **24 django/db/models/query.py** | 1209 | 1224| 124 | 20151 | 116177 | 
| 60 | 24 django/db/models/lookups.py | 359 | 410| 306 | 20457 | 116177 | 
| 61 | 25 django/contrib/postgres/aggregates/mixins.py | 1 | 27| 238 | 20695 | 116415 | 
| 62 | 26 django/db/models/functions/comparison.py | 50 | 68| 180 | 20875 | 118170 | 
| 63 | 26 django/db/models/lookups.py | 128 | 148| 186 | 21061 | 118170 | 
| 64 | 26 django/db/models/sql/query.py | 376 | 428| 497 | 21558 | 118170 | 
| 65 | 27 django/contrib/postgres/fields/ranges.py | 279 | 369| 479 | 22037 | 120556 | 
| 66 | **27 django/db/models/query.py** | 1909 | 1967| 487 | 22524 | 120556 | 
| 67 | 27 django/db/models/functions/comparison.py | 89 | 101| 139 | 22663 | 120556 | 
| 68 | **27 django/db/models/query.py** | 1345 | 1398| 343 | 23006 | 120556 | 
| 69 | 27 django/db/backends/oracle/operations.py | 323 | 350| 271 | 23277 | 120556 | 
| 70 | 27 django/db/models/sql/query.py | 2177 | 2221| 355 | 23632 | 120556 | 
| 71 | 27 django/db/backends/oracle/operations.py | 220 | 268| 411 | 24043 | 120556 | 
| 72 | 27 django/contrib/postgres/search.py | 34 | 79| 282 | 24325 | 120556 | 
| **-> 73 <-** | **27 django/db/models/query_utils.py** | 32 | 65| 234 | 24559 | 120556 | 
| 74 | 27 django/db/models/sql/query.py | 1539 | 1565| 213 | 24772 | 120556 | 
| 75 | 28 django/contrib/postgres/operations.py | 187 | 212| 214 | 24986 | 122953 | 
| 76 | **28 django/db/models/expressions.py** | 908 | 928| 161 | 25147 | 122953 | 
| 77 | 28 django/db/models/sql/query.py | 1853 | 1870| 135 | 25282 | 122953 | 
| 78 | 28 django/db/backends/oracle/operations.py | 439 | 502| 532 | 25814 | 122953 | 
| 79 | **28 django/db/backends/base/features.py** | 6 | 219| 1745 | 27559 | 122953 | 
| 80 | **28 django/db/models/query_utils.py** | 297 | 337| 286 | 27845 | 122953 | 
| **-> 81 <-** | **28 django/db/models/query.py** | 399 | 416| 132 | 27977 | 122953 | 
| 82 | 28 django/db/models/sql/query.py | 2549 | 2565| 119 | 28096 | 122953 | 
| **-> 83 <-** | **28 django/db/models/sql/where.py** | 1 | 31| 167 | 28263 | 122953 | 
| 84 | **28 django/db/models/query.py** | 1159 | 1207| 371 | 28634 | 122953 | 
| 85 | 29 django/contrib/gis/db/backends/oracle/features.py | 1 | 29| 191 | 28825 | 123145 | 
| 86 | 30 django/db/models/fields/__init__.py | 1077 | 1096| 223 | 29048 | 141692 | 
| 87 | 31 django/db/backends/sqlite3/features.py | 1 | 57| 581 | 29629 | 142888 | 
| 88 | 31 django/db/models/sql/query.py | 1991 | 2020| 259 | 29888 | 142888 | 
| 89 | 32 django/db/backends/postgresql/operations.py | 316 | 335| 150 | 30038 | 145724 | 
| 90 | 32 django/db/backends/mysql/operations.py | 1 | 38| 315 | 30353 | 145724 | 
| 91 | **32 django/db/models/sql/where.py** | 213 | 237| 199 | 30552 | 145724 | 
| 92 | 33 django/db/backends/mysql/compiler.py | 1 | 22| 142 | 30694 | 146324 | 
| 93 | 33 django/db/backends/sqlite3/features.py | 108 | 142| 245 | 30939 | 146324 | 
| 94 | 33 django/db/backends/postgresql/operations.py | 227 | 314| 662 | 31601 | 146324 | 
| 95 | 34 django/db/backends/sqlite3/operations.py | 44 | 70| 231 | 31832 | 149815 | 
| 96 | **34 django/db/models/query.py** | 1646 | 1656| 114 | 31946 | 149815 | 
| 97 | 34 django/db/models/sql/compiler.py | 466 | 474| 133 | 32079 | 149815 | 
| 98 | 34 django/db/backends/oracle/operations.py | 21 | 78| 602 | 32681 | 149815 | 
| 99 | **34 django/db/models/expressions.py** | 835 | 861| 258 | 32939 | 149815 | 
| 100 | 34 django/contrib/postgres/operations.py | 66 | 112| 248 | 33187 | 149815 | 
| 101 | 34 django/db/models/sql/query.py | 2354 | 2403| 398 | 33585 | 149815 | 
| 102 | 34 django/contrib/postgres/operations.py | 1 | 37| 271 | 33856 | 149815 | 
| 103 | **34 django/db/models/expressions.py** | 892 | 906| 120 | 33976 | 149815 | 
| 104 | 34 django/contrib/postgres/search.py | 106 | 144| 286 | 34262 | 149815 | 
| 105 | 34 django/db/models/sql/query.py | 1084 | 1122| 337 | 34599 | 149815 | 
| 106 | **34 django/db/models/query.py** | 1400 | 1445| 340 | 34939 | 149815 | 
| 107 | 34 django/db/models/sql/query.py | 934 | 958| 215 | 35154 | 149815 | 
| 108 | 34 django/db/models/lookups.py | 520 | 574| 308 | 35462 | 149815 | 
| 109 | 34 django/contrib/postgres/constraints.py | 162 | 188| 231 | 35693 | 149815 | 
| 110 | **34 django/db/models/expressions.py** | 1084 | 1155| 614 | 36307 | 149815 | 
| 111 | 34 django/db/models/sql/subqueries.py | 48 | 78| 212 | 36519 | 149815 | 
| **-> 112 <-** | **35 django/db/backends/mysql/features.py** | 1 | 56| 402 | 36921 | 152165 | 
| 113 | 36 django/db/backends/base/operations.py | 745 | 767| 198 | 37119 | 158121 | 
| 114 | 36 django/db/models/lookups.py | 342 | 356| 170 | 37289 | 158121 | 
| 115 | 36 django/db/models/sql/query.py | 1051 | 1082| 307 | 37596 | 158121 | 
| 116 | 36 django/db/models/functions/comparison.py | 143 | 180| 262 | 37858 | 158121 | 
| 117 | 36 django/contrib/postgres/search.py | 223 | 259| 247 | 38105 | 158121 | 
| 118 | 36 django/db/models/functions/comparison.py | 104 | 119| 178 | 38283 | 158121 | 
| 119 | 36 django/db/backends/oracle/operations.py | 391 | 437| 385 | 38668 | 158121 | 
| **-> 120 <-** | **36 django/db/models/expressions.py** | 34 | 148| 834 | 39502 | 158121 | 
| 121 | 36 django/contrib/gis/db/models/functions.py | 137 | 150| 104 | 39606 | 158121 | 
| 122 | **36 django/db/models/query.py** | 852 | 870| 178 | 39784 | 158121 | 
| 123 | 37 django/db/migrations/operations/special.py | 63 | 117| 396 | 40180 | 159694 | 
| 124 | 37 django/db/models/sql/query.py | 1872 | 1920| 445 | 40625 | 159694 | 
| 125 | **37 django/db/models/query.py** | 1305 | 1328| 218 | 40843 | 159694 | 
| 126 | **37 django/db/models/expressions.py** | 1408 | 1439| 233 | 41076 | 159694 | 
| 127 | **37 django/db/models/expressions.py** | 1198 | 1215| 153 | 41229 | 159694 | 
| 128 | **37 django/db/models/query.py** | 277 | 304| 221 | 41450 | 159694 | 
| 129 | 37 django/db/models/sql/subqueries.py | 142 | 172| 190 | 41640 | 159694 | 
| 130 | 37 django/db/models/fields/json.py | 157 | 169| 125 | 41765 | 159694 | 
| 131 | 37 django/db/backends/oracle/operations.py | 352 | 363| 226 | 41991 | 159694 | 
| 132 | 37 django/db/models/sql/compiler.py | 1218 | 1239| 199 | 42190 | 159694 | 
| 133 | 37 django/db/backends/oracle/operations.py | 284 | 302| 213 | 42403 | 159694 | 
| 134 | 37 django/db/models/sql/compiler.py | 1814 | 1856| 410 | 42813 | 159694 | 
| 135 | **37 django/db/models/expressions.py** | 1252 | 1308| 436 | 43249 | 159694 | 
| 136 | **37 django/db/backends/base/features.py** | 353 | 371| 173 | 43422 | 159694 | 
| 137 | **37 django/db/models/expressions.py** | 1217 | 1249| 302 | 43724 | 159694 | 
| 138 | 37 django/db/models/sql/query.py | 123 | 143| 173 | 43897 | 159694 | 
| 139 | **37 django/db/backends/mysql/features.py** | 276 | 329| 394 | 44291 | 159694 | 
| 140 | 38 django/contrib/gis/db/models/aggregates.py | 32 | 57| 230 | 44521 | 160329 | 
| 141 | 38 django/db/backends/sqlite3/operations.py | 72 | 140| 617 | 45138 | 160329 | 
| 142 | 38 django/db/models/sql/compiler.py | 1241 | 1284| 349 | 45487 | 160329 | 
| 143 | 38 django/db/models/sql/compiler.py | 1671 | 1703| 254 | 45741 | 160329 | 
| 144 | **38 django/db/backends/mysql/features.py** | 75 | 164| 701 | 46442 | 160329 | 
| 145 | 38 django/db/models/sql/compiler.py | 65 | 77| 155 | 46597 | 160329 | 
| 146 | 38 django/contrib/postgres/fields/ranges.py | 1 | 46| 285 | 46882 | 160329 | 
| 147 | 38 django/contrib/gis/db/backends/mysql/operations.py | 1 | 55| 461 | 47343 | 160329 | 
| 148 | 38 django/db/backends/sqlite3/features.py | 59 | 106| 383 | 47726 | 160329 | 
| 149 | 38 django/db/backends/mysql/operations.py | 406 | 439| 254 | 47980 | 160329 | 
| 150 | 38 django/db/backends/sqlite3/operations.py | 356 | 415| 577 | 48557 | 160329 | 
| 151 | 38 django/contrib/postgres/constraints.py | 190 | 201| 141 | 48698 | 160329 | 
| 152 | 38 django/db/models/sql/compiler.py | 1073 | 1178| 871 | 49569 | 160329 | 
| 153 | **38 django/db/models/query.py** | 551 | 610| 479 | 50048 | 160329 | 
| 154 | 38 django/db/backends/oracle/operations.py | 304 | 321| 250 | 50298 | 160329 | 
| 155 | 39 django/db/models/fields/related.py | 804 | 816| 113 | 50411 | 174892 | 
| 156 | 40 django/db/backends/postgresql/schema.py | 256 | 281| 202 | 50613 | 177178 | 
| 157 | **40 django/db/models/query.py** | 1464 | 1483| 209 | 50822 | 177178 | 
| 158 | 40 django/db/models/functions/text.py | 238 | 255| 150 | 50972 | 177178 | 
| 159 | 41 django/contrib/postgres/aggregates/statistics.py | 1 | 76| 430 | 51402 | 177608 | 
| 160 | 41 django/db/models/sql/compiler.py | 917 | 930| 166 | 51568 | 177608 | 
| 161 | 41 django/db/backends/oracle/operations.py | 135 | 153| 292 | 51860 | 177608 | 
| 162 | 41 django/db/models/sql/query.py | 146 | 252| 882 | 52742 | 177608 | 
| 163 | 41 django/db/models/sql/query.py | 2568 | 2600| 228 | 52970 | 177608 | 
| 164 | **41 django/db/models/expressions.py** | 478 | 522| 314 | 53284 | 177608 | 
| 165 | 41 django/contrib/postgres/fields/ranges.py | 159 | 206| 264 | 53548 | 177608 | 
| 166 | 41 django/db/backends/oracle/operations.py | 522 | 546| 250 | 53798 | 177608 | 
| 167 | 41 django/db/models/sql/compiler.py | 165 | 224| 543 | 54341 | 177608 | 
| 168 | **41 django/db/models/query.py** | 1 | 46| 318 | 54659 | 177608 | 
| 169 | 41 django/db/backends/mysql/operations.py | 371 | 390| 225 | 54884 | 177608 | 
| 170 | 41 django/contrib/gis/db/backends/oracle/operations.py | 1 | 36| 311 | 55195 | 177608 | 
| 171 | 41 django/db/models/fields/json.py | 304 | 335| 310 | 55505 | 177608 | 
| 172 | 41 django/db/models/sql/query.py | 351 | 374| 179 | 55684 | 177608 | 
| 173 | 42 django/db/models/sql/datastructures.py | 25 | 68| 365 | 56049 | 179101 | 


## Missing Patch Files

 * 1: django/db/backends/base/features.py
 * 2: django/db/backends/mysql/features.py
 * 3: django/db/models/expressions.py
 * 4: django/db/models/query.py
 * 5: django/db/models/query_utils.py
 * 6: django/db/models/sql/__init__.py
 * 7: django/db/models/sql/where.py

### Hint

```
It's probably best to write to the DevelopersMailingList to see if there's consensus about this (although having a working patch may help evaluate the idea). I wonder if it's possible to emulate XOR on SQLite similar to what we do for some other database functions.
XOR is not officially supported on Oracle (see ​doc) you pointed to the old MySQL documentation.
To be clear, you're talking about logical XOR, and not bitwise XOR? You linked to PostgreSQL's bitwise XOR operator, #. At the moment it does not have a logical XOR operator. The only ​logical operators it supports are AND, OR and NOT.
Replying to Marten Kenbeek: To be clear, you're talking about logical XOR, and not bitwise XOR? As you've highlighted, this should be for logical XOR and not bitwise XOR. So this is only supported for MariaDB and MySQL which have XOR. This could be implemented by defining Q.XOR and Q.__xor__() and then propagating that around the place. It could be possible to support this for other backends by specifying connection.features.supports_logical_xor = False and then writing out the query differently. For Q(a=1) ^ Q(b=2), the supporting backends would output (a = 1 XOR a = 2), while the others could output ((a = 1 OR b = 2) AND NOT (a = 1 AND b = 2)).
XOR can be implemented by def __xor__(self,other): return self.__or__(other).__and__(self.__invert__().__or__(other.__invert__())) it works for sqlite (possibly for others) wouldn't it solves the problem
Replying to Griffith Rees: XOR seems to be available in ​Postgresql, ​MySQL, ​SequelServer and ​Oracle but NOT ​sqlite. Two stackoverflow questions cover this sort of thing: ​https://stackoverflow.com/questions/50408142/django-models-xor-at-the-model-level and ​https://stackoverflow.com/questions/14711203/perform-a-logical-exclusive-or-on-a-django-q-object. I propose adding XOR to work with Q queries like the ​answer to the second question above. This will be my first time making a major contribution so we'll see how this goes (apologies in advance if this is annoying!). I started on this hoping to use it on my own postgres site, only to realize that postgres does not support logical XOR. Too bad, as it would help with not executing large subqueries multiple times. Never-the-less I have created a PR with the proposed changes for this ​here, which probably needs some TLC from a more advanced contributor. This code should add support for XOR across the codebase, to both Q objects and QuerySets, and ensure it gets down the SQL fed to the database. Note that a TypeError is raised if XOR is attempted on an unsupported backend. This seemed safer than converting on the fly to (A AND ~B) OR (~A AND B), since doing that could lead to some unintended results when the user is expecting XOR to be used. If it is decided that a conversion would be more desirable, then the code can be changed.
After careful consideration I have decided not to raise a TypeError on unsupported backends, and instead convert on the fly from A XOR B to (A OR B) AND NOT (A AND B). MySQL will still take advantage of logical XOR.
```

## Patch

```diff
diff --git a/django/db/backends/base/features.py b/django/db/backends/base/features.py
--- a/django/db/backends/base/features.py
+++ b/django/db/backends/base/features.py
@@ -325,6 +325,9 @@ class BaseDatabaseFeatures:
     # Does the backend support non-deterministic collations?
     supports_non_deterministic_collations = True
 
+    # Does the backend support the logical XOR operator?
+    supports_logical_xor = False
+
     # Collation names for use by the Django test suite.
     test_collations = {
         "ci": None,  # Case-insensitive.
diff --git a/django/db/backends/mysql/features.py b/django/db/backends/mysql/features.py
--- a/django/db/backends/mysql/features.py
+++ b/django/db/backends/mysql/features.py
@@ -47,6 +47,7 @@ class DatabaseFeatures(BaseDatabaseFeatures):
 
     supports_order_by_nulls_modifier = False
     order_by_nulls_first = True
+    supports_logical_xor = True
 
     @cached_property
     def minimum_database_version(self):
diff --git a/django/db/models/expressions.py b/django/db/models/expressions.py
--- a/django/db/models/expressions.py
+++ b/django/db/models/expressions.py
@@ -94,7 +94,7 @@ def __and__(self, other):
         if getattr(self, "conditional", False) and getattr(other, "conditional", False):
             return Q(self) & Q(other)
         raise NotImplementedError(
-            "Use .bitand() and .bitor() for bitwise logical operations."
+            "Use .bitand(), .bitor(), and .bitxor() for bitwise logical operations."
         )
 
     def bitand(self, other):
@@ -106,6 +106,13 @@ def bitleftshift(self, other):
     def bitrightshift(self, other):
         return self._combine(other, self.BITRIGHTSHIFT, False)
 
+    def __xor__(self, other):
+        if getattr(self, "conditional", False) and getattr(other, "conditional", False):
+            return Q(self) ^ Q(other)
+        raise NotImplementedError(
+            "Use .bitand(), .bitor(), and .bitxor() for bitwise logical operations."
+        )
+
     def bitxor(self, other):
         return self._combine(other, self.BITXOR, False)
 
@@ -113,7 +120,7 @@ def __or__(self, other):
         if getattr(self, "conditional", False) and getattr(other, "conditional", False):
             return Q(self) | Q(other)
         raise NotImplementedError(
-            "Use .bitand() and .bitor() for bitwise logical operations."
+            "Use .bitand(), .bitor(), and .bitxor() for bitwise logical operations."
         )
 
     def bitor(self, other):
@@ -139,12 +146,17 @@ def __rpow__(self, other):
 
     def __rand__(self, other):
         raise NotImplementedError(
-            "Use .bitand() and .bitor() for bitwise logical operations."
+            "Use .bitand(), .bitor(), and .bitxor() for bitwise logical operations."
         )
 
     def __ror__(self, other):
         raise NotImplementedError(
-            "Use .bitand() and .bitor() for bitwise logical operations."
+            "Use .bitand(), .bitor(), and .bitxor() for bitwise logical operations."
+        )
+
+    def __rxor__(self, other):
+        raise NotImplementedError(
+            "Use .bitand(), .bitor(), and .bitxor() for bitwise logical operations."
         )
 
 
diff --git a/django/db/models/query.py b/django/db/models/query.py
--- a/django/db/models/query.py
+++ b/django/db/models/query.py
@@ -396,6 +396,25 @@ def __or__(self, other):
         combined.query.combine(other.query, sql.OR)
         return combined
 
+    def __xor__(self, other):
+        self._check_operator_queryset(other, "^")
+        self._merge_sanity_check(other)
+        if isinstance(self, EmptyQuerySet):
+            return other
+        if isinstance(other, EmptyQuerySet):
+            return self
+        query = (
+            self
+            if self.query.can_filter()
+            else self.model._base_manager.filter(pk__in=self.values("pk"))
+        )
+        combined = query._chain()
+        combined._merge_known_related_objects(other)
+        if not other.query.can_filter():
+            other = other.model._base_manager.filter(pk__in=other.values("pk"))
+        combined.query.combine(other.query, sql.XOR)
+        return combined
+
     ####################################
     # METHODS THAT DO DATABASE QUERIES #
     ####################################
diff --git a/django/db/models/query_utils.py b/django/db/models/query_utils.py
--- a/django/db/models/query_utils.py
+++ b/django/db/models/query_utils.py
@@ -38,6 +38,7 @@ class Q(tree.Node):
     # Connection types
     AND = "AND"
     OR = "OR"
+    XOR = "XOR"
     default = AND
     conditional = True
 
@@ -70,6 +71,9 @@ def __or__(self, other):
     def __and__(self, other):
         return self._combine(other, self.AND)
 
+    def __xor__(self, other):
+        return self._combine(other, self.XOR)
+
     def __invert__(self):
         obj = type(self)()
         obj.add(self, self.AND)
diff --git a/django/db/models/sql/__init__.py b/django/db/models/sql/__init__.py
--- a/django/db/models/sql/__init__.py
+++ b/django/db/models/sql/__init__.py
@@ -1,6 +1,6 @@
 from django.db.models.sql.query import *  # NOQA
 from django.db.models.sql.query import Query
 from django.db.models.sql.subqueries import *  # NOQA
-from django.db.models.sql.where import AND, OR
+from django.db.models.sql.where import AND, OR, XOR
 
-__all__ = ["Query", "AND", "OR"]
+__all__ = ["Query", "AND", "OR", "XOR"]
diff --git a/django/db/models/sql/where.py b/django/db/models/sql/where.py
--- a/django/db/models/sql/where.py
+++ b/django/db/models/sql/where.py
@@ -1,14 +1,19 @@
 """
 Code to manage the creation and SQL rendering of 'where' constraints.
 """
+import operator
+from functools import reduce
 
 from django.core.exceptions import EmptyResultSet
+from django.db.models.expressions import Case, When
+from django.db.models.lookups import Exact
 from django.utils import tree
 from django.utils.functional import cached_property
 
 # Connection types
 AND = "AND"
 OR = "OR"
+XOR = "XOR"
 
 
 class WhereNode(tree.Node):
@@ -39,10 +44,12 @@ def split_having(self, negated=False):
         if not self.contains_aggregate:
             return self, None
         in_negated = negated ^ self.negated
-        # If the effective connector is OR and this node contains an aggregate,
-        # then we need to push the whole branch to HAVING clause.
-        may_need_split = (in_negated and self.connector == AND) or (
-            not in_negated and self.connector == OR
+        # If the effective connector is OR or XOR and this node contains an
+        # aggregate, then we need to push the whole branch to HAVING clause.
+        may_need_split = (
+            (in_negated and self.connector == AND)
+            or (not in_negated and self.connector == OR)
+            or self.connector == XOR
         )
         if may_need_split and self.contains_aggregate:
             return None, self
@@ -85,6 +92,21 @@ def as_sql(self, compiler, connection):
         else:
             full_needed, empty_needed = 1, len(self.children)
 
+        if self.connector == XOR and not connection.features.supports_logical_xor:
+            # Convert if the database doesn't support XOR:
+            #   a XOR b XOR c XOR ...
+            # to:
+            #   (a OR b OR c OR ...) AND (a + b + c + ...) == 1
+            lhs = self.__class__(self.children, OR)
+            rhs_sum = reduce(
+                operator.add,
+                (Case(When(c, then=1), default=0) for c in self.children),
+            )
+            rhs = Exact(1, rhs_sum)
+            return self.__class__([lhs, rhs], AND, self.negated).as_sql(
+                compiler, connection
+            )
+
         for child in self.children:
             try:
                 sql, params = compiler.compile(child)

```

## Test Patch

```diff
diff --git a/tests/aggregation_regress/tests.py b/tests/aggregation_regress/tests.py
--- a/tests/aggregation_regress/tests.py
+++ b/tests/aggregation_regress/tests.py
@@ -1704,6 +1704,28 @@ def test_filter_aggregates_negated_and_connector(self):
             attrgetter("pk"),
         )
 
+    def test_filter_aggregates_xor_connector(self):
+        q1 = Q(price__gt=50)
+        q2 = Q(authors__count__gt=1)
+        query = Book.objects.annotate(Count("authors")).filter(q1 ^ q2).order_by("pk")
+        self.assertQuerysetEqual(
+            query,
+            [self.b1.pk, self.b4.pk, self.b6.pk],
+            attrgetter("pk"),
+        )
+
+    def test_filter_aggregates_negated_xor_connector(self):
+        q1 = Q(price__gt=50)
+        q2 = Q(authors__count__gt=1)
+        query = (
+            Book.objects.annotate(Count("authors")).filter(~(q1 ^ q2)).order_by("pk")
+        )
+        self.assertQuerysetEqual(
+            query,
+            [self.b2.pk, self.b3.pk, self.b5.pk],
+            attrgetter("pk"),
+        )
+
     def test_ticket_11293_q_immutable(self):
         """
         Splitting a q object to parts for where/having doesn't alter
diff --git a/tests/expressions/tests.py b/tests/expressions/tests.py
--- a/tests/expressions/tests.py
+++ b/tests/expressions/tests.py
@@ -2339,7 +2339,9 @@ def test_filtered_aggregates(self):
 
 
 class CombinableTests(SimpleTestCase):
-    bitwise_msg = "Use .bitand() and .bitor() for bitwise logical operations."
+    bitwise_msg = (
+        "Use .bitand(), .bitor(), and .bitxor() for bitwise logical operations."
+    )
 
     def test_negation(self):
         c = Combinable()
@@ -2353,6 +2355,10 @@ def test_or(self):
         with self.assertRaisesMessage(NotImplementedError, self.bitwise_msg):
             Combinable() | Combinable()
 
+    def test_xor(self):
+        with self.assertRaisesMessage(NotImplementedError, self.bitwise_msg):
+            Combinable() ^ Combinable()
+
     def test_reversed_and(self):
         with self.assertRaisesMessage(NotImplementedError, self.bitwise_msg):
             object() & Combinable()
@@ -2361,6 +2367,10 @@ def test_reversed_or(self):
         with self.assertRaisesMessage(NotImplementedError, self.bitwise_msg):
             object() | Combinable()
 
+    def test_reversed_xor(self):
+        with self.assertRaisesMessage(NotImplementedError, self.bitwise_msg):
+            object() ^ Combinable()
+
 
 class CombinedExpressionTests(SimpleTestCase):
     def test_resolve_output_field(self):
diff --git a/tests/queries/test_q.py b/tests/queries/test_q.py
--- a/tests/queries/test_q.py
+++ b/tests/queries/test_q.py
@@ -27,6 +27,15 @@ def test_combine_or_empty(self):
         self.assertEqual(q | Q(), q)
         self.assertEqual(Q() | q, q)
 
+    def test_combine_xor_empty(self):
+        q = Q(x=1)
+        self.assertEqual(q ^ Q(), q)
+        self.assertEqual(Q() ^ q, q)
+
+        q = Q(x__in={}.keys())
+        self.assertEqual(q ^ Q(), q)
+        self.assertEqual(Q() ^ q, q)
+
     def test_combine_empty_copy(self):
         base_q = Q(x=1)
         tests = [
@@ -34,6 +43,8 @@ def test_combine_empty_copy(self):
             Q() | base_q,
             base_q & Q(),
             Q() & base_q,
+            base_q ^ Q(),
+            Q() ^ base_q,
         ]
         for i, q in enumerate(tests):
             with self.subTest(i=i):
@@ -43,6 +54,9 @@ def test_combine_empty_copy(self):
     def test_combine_or_both_empty(self):
         self.assertEqual(Q() | Q(), Q())
 
+    def test_combine_xor_both_empty(self):
+        self.assertEqual(Q() ^ Q(), Q())
+
     def test_combine_not_q_object(self):
         obj = object()
         q = Q(x=1)
@@ -50,12 +64,15 @@ def test_combine_not_q_object(self):
             q | obj
         with self.assertRaisesMessage(TypeError, str(obj)):
             q & obj
+        with self.assertRaisesMessage(TypeError, str(obj)):
+            q ^ obj
 
     def test_combine_negated_boolean_expression(self):
         tagged = Tag.objects.filter(category=OuterRef("pk"))
         tests = [
             Q() & ~Exists(tagged),
             Q() | ~Exists(tagged),
+            Q() ^ ~Exists(tagged),
         ]
         for q in tests:
             with self.subTest(q=q):
@@ -88,6 +105,20 @@ def test_deconstruct_or(self):
         )
         self.assertEqual(kwargs, {"_connector": "OR"})
 
+    def test_deconstruct_xor(self):
+        q1 = Q(price__gt=F("discounted_price"))
+        q2 = Q(price=F("discounted_price"))
+        q = q1 ^ q2
+        path, args, kwargs = q.deconstruct()
+        self.assertEqual(
+            args,
+            (
+                ("price__gt", F("discounted_price")),
+                ("price", F("discounted_price")),
+            ),
+        )
+        self.assertEqual(kwargs, {"_connector": "XOR"})
+
     def test_deconstruct_and(self):
         q1 = Q(price__gt=F("discounted_price"))
         q2 = Q(price=F("discounted_price"))
@@ -144,6 +175,13 @@ def test_reconstruct_or(self):
         path, args, kwargs = q.deconstruct()
         self.assertEqual(Q(*args, **kwargs), q)
 
+    def test_reconstruct_xor(self):
+        q1 = Q(price__gt=F("discounted_price"))
+        q2 = Q(price=F("discounted_price"))
+        q = q1 ^ q2
+        path, args, kwargs = q.deconstruct()
+        self.assertEqual(Q(*args, **kwargs), q)
+
     def test_reconstruct_and(self):
         q1 = Q(price__gt=F("discounted_price"))
         q2 = Q(price=F("discounted_price"))
diff --git a/tests/queries/test_qs_combinators.py b/tests/queries/test_qs_combinators.py
--- a/tests/queries/test_qs_combinators.py
+++ b/tests/queries/test_qs_combinators.py
@@ -526,6 +526,7 @@ def test_operator_on_combined_qs_error(self):
         operators = [
             ("|", operator.or_),
             ("&", operator.and_),
+            ("^", operator.xor),
         ]
         for combinator in combinators:
             combined_qs = getattr(qs, combinator)(qs)
diff --git a/tests/queries/tests.py b/tests/queries/tests.py
--- a/tests/queries/tests.py
+++ b/tests/queries/tests.py
@@ -1883,6 +1883,10 @@ def test_ticket5261(self):
             Note.objects.exclude(~Q() & ~Q()),
             [self.n1, self.n2],
         )
+        self.assertSequenceEqual(
+            Note.objects.exclude(~Q() ^ ~Q()),
+            [self.n1, self.n2],
+        )
 
     def test_extra_select_literal_percent_s(self):
         # Allow %%s to escape select clauses
@@ -2129,6 +2133,15 @@ def test_col_alias_quoted(self):
         sql = captured_queries[0]["sql"]
         self.assertIn("AS %s" % connection.ops.quote_name("col1"), sql)
 
+    def test_xor_subquery(self):
+        self.assertSequenceEqual(
+            Tag.objects.filter(
+                Exists(Tag.objects.filter(id=OuterRef("id"), name="t3"))
+                ^ Exists(Tag.objects.filter(id=OuterRef("id"), parent=self.t1))
+            ),
+            [self.t2],
+        )
+
 
 class RawQueriesTests(TestCase):
     @classmethod
@@ -2432,6 +2445,30 @@ def test_or_with_both_slice_and_ordering(self):
         qs2 = Classroom.objects.filter(has_blackboard=True).order_by("-name")[:1]
         self.assertCountEqual(qs1 | qs2, [self.room_3, self.room_4])
 
+    @skipUnlessDBFeature("allow_sliced_subqueries_with_in")
+    def test_xor_with_rhs_slice(self):
+        qs1 = Classroom.objects.filter(has_blackboard=True)
+        qs2 = Classroom.objects.filter(has_blackboard=False)[:1]
+        self.assertCountEqual(qs1 ^ qs2, [self.room_1, self.room_2, self.room_3])
+
+    @skipUnlessDBFeature("allow_sliced_subqueries_with_in")
+    def test_xor_with_lhs_slice(self):
+        qs1 = Classroom.objects.filter(has_blackboard=True)[:1]
+        qs2 = Classroom.objects.filter(has_blackboard=False)
+        self.assertCountEqual(qs1 ^ qs2, [self.room_1, self.room_2, self.room_4])
+
+    @skipUnlessDBFeature("allow_sliced_subqueries_with_in")
+    def test_xor_with_both_slice(self):
+        qs1 = Classroom.objects.filter(has_blackboard=False)[:1]
+        qs2 = Classroom.objects.filter(has_blackboard=True)[:1]
+        self.assertCountEqual(qs1 ^ qs2, [self.room_1, self.room_2])
+
+    @skipUnlessDBFeature("allow_sliced_subqueries_with_in")
+    def test_xor_with_both_slice_and_ordering(self):
+        qs1 = Classroom.objects.filter(has_blackboard=False).order_by("-pk")[:1]
+        qs2 = Classroom.objects.filter(has_blackboard=True).order_by("-name")[:1]
+        self.assertCountEqual(qs1 ^ qs2, [self.room_3, self.room_4])
+
     def test_subquery_aliases(self):
         combined = School.objects.filter(pk__isnull=False) & School.objects.filter(
             Exists(
diff --git a/tests/xor_lookups/__init__.py b/tests/xor_lookups/__init__.py
new file mode 100644
diff --git a/tests/xor_lookups/models.py b/tests/xor_lookups/models.py
new file mode 100644
--- /dev/null
+++ b/tests/xor_lookups/models.py
@@ -0,0 +1,8 @@
+from django.db import models
+
+
+class Number(models.Model):
+    num = models.IntegerField()
+
+    def __str__(self):
+        return str(self.num)
diff --git a/tests/xor_lookups/tests.py b/tests/xor_lookups/tests.py
new file mode 100644
--- /dev/null
+++ b/tests/xor_lookups/tests.py
@@ -0,0 +1,67 @@
+from django.db.models import Q
+from django.test import TestCase
+
+from .models import Number
+
+
+class XorLookupsTests(TestCase):
+    @classmethod
+    def setUpTestData(cls):
+        cls.numbers = [Number.objects.create(num=i) for i in range(10)]
+
+    def test_filter(self):
+        self.assertCountEqual(
+            Number.objects.filter(num__lte=7) ^ Number.objects.filter(num__gte=3),
+            self.numbers[:3] + self.numbers[8:],
+        )
+        self.assertCountEqual(
+            Number.objects.filter(Q(num__lte=7) ^ Q(num__gte=3)),
+            self.numbers[:3] + self.numbers[8:],
+        )
+
+    def test_filter_negated(self):
+        self.assertCountEqual(
+            Number.objects.filter(Q(num__lte=7) ^ ~Q(num__lt=3)),
+            self.numbers[:3] + self.numbers[8:],
+        )
+        self.assertCountEqual(
+            Number.objects.filter(~Q(num__gt=7) ^ ~Q(num__lt=3)),
+            self.numbers[:3] + self.numbers[8:],
+        )
+        self.assertCountEqual(
+            Number.objects.filter(Q(num__lte=7) ^ ~Q(num__lt=3) ^ Q(num__lte=1)),
+            [self.numbers[2]] + self.numbers[8:],
+        )
+        self.assertCountEqual(
+            Number.objects.filter(~(Q(num__lte=7) ^ ~Q(num__lt=3) ^ Q(num__lte=1))),
+            self.numbers[:2] + self.numbers[3:8],
+        )
+
+    def test_exclude(self):
+        self.assertCountEqual(
+            Number.objects.exclude(Q(num__lte=7) ^ Q(num__gte=3)),
+            self.numbers[3:8],
+        )
+
+    def test_stages(self):
+        numbers = Number.objects.all()
+        self.assertSequenceEqual(
+            numbers.filter(num__gte=0) ^ numbers.filter(num__lte=11),
+            [],
+        )
+        self.assertSequenceEqual(
+            numbers.filter(num__gt=0) ^ numbers.filter(num__lt=11),
+            [self.numbers[0]],
+        )
+
+    def test_pk_q(self):
+        self.assertCountEqual(
+            Number.objects.filter(Q(pk=self.numbers[0].pk) ^ Q(pk=self.numbers[1].pk)),
+            self.numbers[:2],
+        )
+
+    def test_empty_in(self):
+        self.assertCountEqual(
+            Number.objects.filter(Q(pk__in=[]) ^ Q(num__gte=5)),
+            self.numbers[5:],
+        )

```


## Code snippets

### 1 - django/contrib/postgres/search.py:

Start line: 147, End line: 174

```python
class SearchQueryCombinable:
    BITAND = "&&"
    BITOR = "||"

    def _combine(self, other, connector, reversed):
        if not isinstance(other, SearchQueryCombinable):
            raise TypeError(
                "SearchQuery can only be combined with other SearchQuery "
                "instances, got %s." % type(other).__name__
            )
        if reversed:
            return CombinedSearchQuery(other, connector, self, self.config)
        return CombinedSearchQuery(self, connector, other, self.config)

    # On Combinable, these are not implemented to reduce confusion with Q. In
    # this case we are actually (ab)using them to do logical combination so
    # it's consistent with other usage in Django.
    def __or__(self, other):
        return self._combine(other, self.BITOR, False)

    def __ror__(self, other):
        return self._combine(other, self.BITOR, True)

    def __and__(self, other):
        return self._combine(other, self.BITAND, False)

    def __rand__(self, other):
        return self._combine(other, self.BITAND, True)
```
### 2 - django/contrib/postgres/aggregates/general.py:

Start line: 60, End line: 95

```python
class BitAnd(Aggregate):
    function = "BIT_AND"


class BitOr(Aggregate):
    function = "BIT_OR"


class BitXor(Aggregate):
    function = "BIT_XOR"


class BoolAnd(Aggregate):
    function = "BOOL_AND"
    output_field = BooleanField()


class BoolOr(Aggregate):
    function = "BOOL_OR"
    output_field = BooleanField()


class JSONBAgg(DeprecatedConvertValueMixin, OrderableAggMixin, Aggregate):
    function = "JSONB_AGG"
    template = "%(function)s(%(distinct)s%(expressions)s %(ordering)s)"
    allow_distinct = True
    output_field = JSONField()

    # RemovedInDjango50Warning
    deprecation_value = "[]"
    deprecation_msg = (
        "In Django 5.0, JSONBAgg() will return None instead of an empty list "
        "if there are no rows. Pass default=None to opt into the new behavior "
        "and silence this warning or default=Value('[]') to keep the previous "
        "behavior."
    )
```
### 3 - django/db/models/sql/query.py:

Start line: 1505, End line: 1537

```python
class Query(BaseExpression):

    def _add_q(
        self,
        q_object,
        used_aliases,
        branch_negated=False,
        current_negated=False,
        allow_joins=True,
        split_subq=True,
        check_filterable=True,
    ):
        """Add a Q-object to the current filter."""
        connector = q_object.connector
        current_negated = current_negated ^ q_object.negated
        branch_negated = branch_negated or q_object.negated
        target_clause = WhereNode(connector=connector, negated=q_object.negated)
        joinpromoter = JoinPromoter(
            q_object.connector, len(q_object.children), current_negated
        )
        for child in q_object.children:
            child_clause, needed_inner = self.build_filter(
                child,
                can_reuse=used_aliases,
                branch_negated=branch_negated,
                current_negated=current_negated,
                allow_joins=allow_joins,
                split_subq=split_subq,
                check_filterable=check_filterable,
            )
            joinpromoter.add_votes(needed_inner)
            if child_clause:
                target_clause.add(child_clause, connector)
        needed_inner = joinpromoter.update_join_types(self)
        return target_clause, needed_inner
```
### 4 - django/db/models/query_utils.py:

Start line: 67, End line: 104

```python
class Q(tree.Node):

    def __or__(self, other):
        return self._combine(other, self.OR)

    def __and__(self, other):
        return self._combine(other, self.AND)

    def __invert__(self):
        obj = type(self)()
        obj.add(self, self.AND)
        obj.negate()
        return obj

    def resolve_expression(
        self, query=None, allow_joins=True, reuse=None, summarize=False, for_save=False
    ):
        # We must promote any new joins to left outer joins so that when Q is
        # used as an expression, rows aren't filtered due to joins.
        clause, joins = query._add_q(
            self,
            reuse,
            allow_joins=allow_joins,
            split_subq=False,
            check_filterable=False,
        )
        query.promote_joins(joins)
        return clause

    def deconstruct(self):
        path = "%s.%s" % (self.__class__.__module__, self.__class__.__name__)
        if path.startswith("django.db.models.query_utils"):
            path = path.replace("django.db.models.query_utils", "django.db.models")
        args = tuple(self.children)
        kwargs = {}
        if self.connector != self.default:
            kwargs["_connector"] = self.connector
        if self.negated:
            kwargs["_negated"] = True
        return path, args, kwargs
```
### 5 - django/db/models/expressions.py:

Start line: 1311, End line: 1351

```python
class Exists(Subquery):
    template = "EXISTS(%(subquery)s)"
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
        try:
            sql, params = super().as_sql(
                compiler,
                connection,
                template=template,
                query=query,
                **extra_context,
            )
        except EmptyResultSet:
            if self.negated:
                features = compiler.connection.features
                if not features.supports_boolean_expr_in_select_clause:
                    return "1=1", ()
                return compiler.compile(Value(True))
            raise
        if self.negated:
            sql = "NOT {}".format(sql)
        return sql, params

    def select_format(self, compiler, sql, params):
        # Wrap EXISTS() with a CASE WHEN expression if a database backend
        # (e.g. Oracle) doesn't support boolean expression in SELECT or GROUP
        # BY list.
        if not compiler.connection.features.supports_boolean_expr_in_select_clause:
            sql = "CASE WHEN {} THEN 1 ELSE 0 END".format(sql)
        return sql, params
```
### 6 - django/db/models/query.py:

Start line: 365, End line: 397

```python
class QuerySet:

    def __class_getitem__(cls, *args, **kwargs):
        return cls

    def __and__(self, other):
        self._check_operator_queryset(other, "&")
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
        self._check_operator_queryset(other, "|")
        self._merge_sanity_check(other)
        if isinstance(self, EmptyQuerySet):
            return other
        if isinstance(other, EmptyQuerySet):
            return self
        query = (
            self
            if self.query.can_filter()
            else self.model._base_manager.filter(pk__in=self.values("pk"))
        )
        combined = query._chain()
        combined._merge_known_related_objects(other)
        if not other.query.can_filter():
            other = other.model._base_manager.filter(pk__in=other.values("pk"))
        combined.query.combine(other.query, sql.OR)
        return combined
```
### 7 - django/db/models/sql/query.py:

Start line: 1477, End line: 1503

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
        existing_inner = {
            a for a in self.alias_map if self.alias_map[a].join_type == INNER
        }
        clause, _ = self._add_q(q_object, self.used_aliases)
        if clause:
            self.where.add(clause, AND)
        self.demote_joins(existing_inner)

    def build_where(self, filter_expr):
        return self.build_filter(filter_expr, allow_joins=False)[0]

    def clear_where(self):
        self.where = WhereNode()
```
### 8 - django/db/backends/postgresql/features.py:

Start line: 1, End line: 109

```python
import operator

from django.db import InterfaceError
from django.db.backends.base.features import BaseDatabaseFeatures
from django.utils.functional import cached_property


class DatabaseFeatures(BaseDatabaseFeatures):
    minimum_database_version = (10,)
    allows_group_by_selected_pks = True
    can_return_columns_from_insert = True
    can_return_rows_from_bulk_insert = True
    has_real_datatype = True
    has_native_uuid_field = True
    has_native_duration_field = True
    has_native_json_field = True
    can_defer_constraint_checks = True
    has_select_for_update = True
    has_select_for_update_nowait = True
    has_select_for_update_of = True
    has_select_for_update_skip_locked = True
    has_select_for_no_key_update = True
    can_release_savepoints = True
    supports_tablespaces = True
    supports_transactions = True
    can_introspect_materialized_views = True
    can_distinct_on_fields = True
    can_rollback_ddl = True
    supports_combined_alters = True
    nulls_order_largest = True
    closed_cursor_error_class = InterfaceError
    greatest_least_ignores_nulls = True
    can_clone_databases = True
    supports_temporal_subtraction = True
    supports_slicing_ordering_in_compound = True
    create_test_procedure_without_params_sql = """
        CREATE FUNCTION test_procedure () RETURNS void AS $$
        DECLARE
            V_I INTEGER;
        BEGIN
            V_I := 1;
        END;
    $$ LANGUAGE plpgsql;"""
    create_test_procedure_with_int_param_sql = """
        CREATE FUNCTION test_procedure (P_I INTEGER) RETURNS void AS $$
        DECLARE
            V_I INTEGER;
        BEGIN
            V_I := P_I;
        END;
    $$ LANGUAGE plpgsql;"""
    requires_casted_case_in_updates = True
    supports_over_clause = True
    only_supports_unbounded_with_preceding_and_following = True
    supports_aggregate_filter_clause = True
    supported_explain_formats = {"JSON", "TEXT", "XML", "YAML"}
    validates_explain_options = False  # A query will error on invalid options.
    supports_deferrable_unique_constraints = True
    has_json_operators = True
    json_key_contains_list_matching_requires_list = True
    supports_update_conflicts = True
    supports_update_conflicts_with_target = True
    test_collations = {
        "non_default": "sv-x-icu",
        "swedish_ci": "sv-x-icu",
    }
    test_now_utc_template = "STATEMENT_TIMESTAMP() AT TIME ZONE 'UTC'"

    django_test_skips = {
        "opclasses are PostgreSQL only.": {
            "indexes.tests.SchemaIndexesNotPostgreSQLTests."
            "test_create_index_ignores_opclasses",
        },
    }

    @cached_property
    def introspected_field_types(self):
        return {
            **super().introspected_field_types,
            "PositiveBigIntegerField": "BigIntegerField",
            "PositiveIntegerField": "IntegerField",
            "PositiveSmallIntegerField": "SmallIntegerField",
        }

    @cached_property
    def is_postgresql_11(self):
        return self.connection.pg_version >= 110000

    @cached_property
    def is_postgresql_12(self):
        return self.connection.pg_version >= 120000

    @cached_property
    def is_postgresql_13(self):
        return self.connection.pg_version >= 130000

    @cached_property
    def is_postgresql_14(self):
        return self.connection.pg_version >= 140000

    has_bit_xor = property(operator.attrgetter("is_postgresql_14"))
    has_websearch_to_tsquery = property(operator.attrgetter("is_postgresql_11"))
    supports_covering_indexes = property(operator.attrgetter("is_postgresql_11"))
    supports_covering_gist_indexes = property(operator.attrgetter("is_postgresql_12"))
    supports_covering_spgist_indexes = property(operator.attrgetter("is_postgresql_14"))
    supports_non_deterministic_collations = property(
        operator.attrgetter("is_postgresql_12")
    )
```
### 9 - django/db/backends/oracle/operations.py:

Start line: 617, End line: 633

```python
class DatabaseOperations(BaseDatabaseOperations):
    # Oracle uses NUMBER(5), NUMBER(11), and NUMBER(19) for integer fields.
    # SmallIntegerField uses NUMBER(11) instead of NUMBER(5), which is used by
    integer_field_ranges =
    # ... other code

    def combine_expression(self, connector, sub_expressions):
        lhs, rhs = sub_expressions
        if connector == "%%":
            return "MOD(%s)" % ",".join(sub_expressions)
        elif connector == "&":
            return "BITAND(%s)" % ",".join(sub_expressions)
        elif connector == "|":
            return "BITAND(-%(lhs)s-1,%(rhs)s)+%(lhs)s" % {"lhs": lhs, "rhs": rhs}
        elif connector == "<<":
            return "(%(lhs)s * POWER(2, %(rhs)s))" % {"lhs": lhs, "rhs": rhs}
        elif connector == ">>":
            return "FLOOR(%(lhs)s / POWER(2, %(rhs)s))" % {"lhs": lhs, "rhs": rhs}
        elif connector == "^":
            return "POWER(%s)" % ",".join(sub_expressions)
        elif connector == "#":
            raise NotSupportedError("Bitwise XOR is not supported in Oracle.")
        return super().combine_expression(connector, sub_expressions)
    # ... other code
```
### 10 - django/db/models/sql/query.py:

Start line: 1, End line: 67

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
    BaseExpression,
    Col,
    Exists,
    F,
    OuterRef,
    Ref,
    ResolvedOuterRef,
)
from django.db.models.fields import Field
from django.db.models.fields.related_lookups import MultiColSource
from django.db.models.lookups import Lookup
from django.db.models.query_utils import (
    Q,
    check_rel_lookup_compatibility,
    refs_expression,
)
from django.db.models.sql.constants import INNER, LOUTER, ORDER_DIR, SINGLE
from django.db.models.sql.datastructures import BaseTable, Empty, Join, MultiJoin
from django.db.models.sql.where import AND, OR, ExtraWhere, NothingNode, WhereNode
from django.utils.functional import cached_property
from django.utils.tree import Node

__all__ = ["Query", "RawQuery"]


def get_field_names_from_opts(opts):
    return set(
        chain.from_iterable(
            (f.name, f.attname) if f.concrete else (f.name,) for f in opts.get_fields()
        )
    )


def get_children_from_q(q):
    for child in q.children:
        if isinstance(child, Node):
            yield from get_children_from_q(child)
        else:
            yield child


JoinInfo = namedtuple(
    "JoinInfo",
    ("final_field", "targets", "opts", "joins", "path", "transform_function"),
)
```
### 11 - django/db/models/query.py:

Start line: 1250, End line: 1263

```python
class QuerySet:

    def intersection(self, *other_qs):
        # If any query is an EmptyQuerySet, return it.
        if isinstance(self, EmptyQuerySet):
            return self
        for other in other_qs:
            if isinstance(other, EmptyQuerySet):
                return other
        return self._combinator_query("intersection", *other_qs)

    def difference(self, *other_qs):
        # If the query is an EmptyQuerySet, return it.
        if isinstance(self, EmptyQuerySet):
            return self
        return self._combinator_query("difference", *other_qs)
```
### 13 - django/db/models/query.py:

Start line: 1239, End line: 1248

```python
class QuerySet:

    def union(self, *other_qs, all=False):
        # If the query is an EmptyQuerySet, combine all nonempty querysets.
        if isinstance(self, EmptyQuerySet):
            qs = [q for q in other_qs if not isinstance(q, EmptyQuerySet)]
            if not qs:
                return self
            if len(qs) == 1:
                return qs[0]
            return qs[0]._combinator_query("union", *qs[1:], all=all)
        return self._combinator_query("union", *other_qs, all=all)
```
### 15 - django/db/models/query.py:

Start line: 1024, End line: 1061

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
        qs = RawQuerySet(
            raw_query,
            model=self.model,
            params=params,
            translations=translations,
            using=using,
        )
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
### 19 - django/db/models/query.py:

Start line: 1226, End line: 1237

```python
class QuerySet:

    def _combinator_query(self, combinator, *other_qs, all=False):
        # Clone the query to inherit the select list and everything
        clone = self._chain()
        # Clear limits and ordering so they can be reapplied
        clone.query.clear_ordering(force=True)
        clone.query.clear_limits()
        clone.query.combined_queries = (self.query,) + tuple(
            qs.query for qs in other_qs
        )
        clone.query.combinator = combinator
        clone.query.combinator_all = all
        return clone
```
### 24 - django/db/backends/base/features.py:

Start line: 220, End line: 351

```python
class BaseDatabaseFeatures:
    minimum_database_version =
    # ... other code
    can_clone_databases = False

    # Does the backend consider table names with different casing to
    # be equal?
    ignores_table_name_case = False

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
    # Does the backend support updating rows on constraint or uniqueness errors
    # during INSERT?
    supports_update_conflicts = False
    supports_update_conflicts_with_target = False

    # Does this backend require casting the results of CASE expressions used
    # in UPDATE statements to ensure the expression has the correct type?
    requires_casted_case_in_updates = False

    # Does the backend support partial indexes (CREATE INDEX ... WHERE ...)?
    supports_partial_indexes = True
    supports_functions_in_partial_indexes = True
    # Does the backend support covering indexes (CREATE INDEX ... INCLUDE ...)?
    supports_covering_indexes = False
    # Does the backend support indexes on expressions?
    supports_expression_indexes = True
    # Does the backend treat COLLATE as an indexed expression?
    collate_as_index_expression = False

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
    # Does the backend support __contains and __contained_by lookups for
    # a JSONField?
    supports_json_field_contains = True
    # Does value__d__contains={'f': 'g'} (without a list around the dict) match
    # {'d': [{'f': 'g'}]}?
    json_key_contains_list_matching_requires_list = False
    # Does the backend support JSONObject() database function?
    has_json_object_function = True

    # Does the backend support column collations?
    supports_collation_on_charfield = True
    supports_collation_on_textfield = True
    # Does the backend support non-deterministic collations?
    supports_non_deterministic_collations = True

    # Collation names for use by the Django test suite.
    test_collations = {
        "ci": None,  # Case-insensitive.
        "cs": None,  # Case-sensitive.
        "non_default": None,  # Non-default.
        "swedish_ci": None,  # Swedish case-insensitive.
    }
    # SQL template override for tests.aggregation.tests.NowUTC
    test_now_utc_template = None

    # A set of dotted paths to tests in Django's test suite that are expected
    # to fail on this database.
    django_test_expected_failures = set()
    # A map of reasons to sets of dotted paths to tests in Django's test suite
    # that should be skipped for this database.
    django_test_skips = {}

    def __init__(self, connection):
        self.connection = connection

    @cached_property
    def supports_explaining_query_execution(self):
        """Does this backend support explaining query execution?"""
        return self.connection.ops.explain_prefix is not None
    # ... other code
```
### 33 - django/db/models/query.py:

Start line: 1658, End line: 1714

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
            raise TypeError("Cannot use multi-field values as a filter value.")
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
        invalid_args = sorted(
            str(arg) for arg in values if not hasattr(arg, "resolve_expression")
        )
        if invalid_args:
            raise TypeError(
                "QuerySet.%s() received non-expression(s): %s."
                % (
                    method_name,
                    ", ".join(invalid_args),
                )
            )

    def _not_support_combined_queries(self, operation_name):
        if self.query.combinator:
            raise NotSupportedError(
                "Calling QuerySet.%s() after %s() is not supported."
                % (operation_name, self.query.combinator)
            )

    def _check_operator_queryset(self, other, operator_):
        if self.query.combinator or other.query.combinator:
            raise TypeError(f"Cannot use {operator_} operator with combined queryset.")
```
### 36 - django/db/models/sql/where.py:

Start line: 262, End line: 279

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
### 41 - django/db/models/expressions.py:

Start line: 221, End line: 267

```python
class BaseExpression:

    @cached_property
    def contains_aggregate(self):
        return any(
            expr and expr.contains_aggregate for expr in self.get_source_expressions()
        )

    @cached_property
    def contains_over_clause(self):
        return any(
            expr and expr.contains_over_clause for expr in self.get_source_expressions()
        )

    @cached_property
    def contains_column_references(self):
        return any(
            expr and expr.contains_column_references
            for expr in self.get_source_expressions()
        )

    def resolve_expression(
        self, query=None, allow_joins=True, reuse=None, summarize=False, for_save=False
    ):
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
        c.set_source_expressions(
            [
                expr.resolve_expression(query, allow_joins, reuse, summarize)
                if expr
                else None
                for expr in c.get_source_expressions()
            ]
        )
        return c
```
### 45 - django/db/models/query.py:

Start line: 1485, End line: 1514

```python
class QuerySet:

    def using(self, alias):
        """Select which database this QuerySet should execute against."""
        clone = self._chain()
        clone._db = alias
        return clone

    ###################################
    # PUBLIC INTROSPECTION ATTRIBUTES #
    ###################################

    @property
    def ordered(self):
        """
        Return True if the QuerySet is ordered -- i.e. has an order_by()
        clause or a default ordering on the model (or is empty).
        """
        if isinstance(self, EmptyQuerySet):
            return True
        if self.query.extra_order_by or self.query.order_by:
            return True
        elif (
            self.query.default_ordering
            and self.query.get_meta().ordering
            and
            # A default ordering doesn't affect GROUP BY queries.
            not self.query.group_by
        ):
            return True
        else:
            return False
```
### 55 - django/db/models/query.py:

Start line: 998, End line: 1022

```python
class QuerySet:

    _update.alters_data = True
    _update.queryset_only = False

    def exists(self):
        if self._result_cache is None:
            return self.query.has_results(using=self.db)
        return bool(self._result_cache)

    def contains(self, obj):
        """Return True if the queryset contains an object."""
        self._not_support_combined_queries("contains")
        if self._fields is not None:
            raise TypeError(
                "Cannot call QuerySet.contains() after .values() or .values_list()."
            )
        try:
            if obj._meta.concrete_model != self.model._meta.concrete_model:
                return False
        except AttributeError:
            raise TypeError("'obj' must be a model instance.")
        if obj.pk is None:
            raise ValueError("QuerySet.contains() cannot be used on unsaved objects.")
        if self._result_cache is not None:
            return obj in self._result_cache
        return self.filter(pk=obj.pk).exists()
```
### 59 - django/db/models/query.py:

Start line: 1209, End line: 1224

```python
class QuerySet:

    def complex_filter(self, filter_obj):
        """
        Return a new QuerySet instance with filter_obj added to the filters.

        filter_obj can be a Q object or a dictionary of keyword lookup
        arguments.

        This exists to support framework features such as 'limit_choices_to',
        and usually it will be more natural to use other methods.
        """
        if isinstance(filter_obj, Q):
            clone = self._chain()
            clone.query.add_q(filter_obj)
            return clone
        else:
            return self._filter_or_exclude(False, args=(), kwargs=filter_obj)
```
### 66 - django/db/models/query.py:

Start line: 1909, End line: 1967

```python
class Prefetch:
    def __init__(self, lookup, queryset=None, to_attr=None):
        # `prefetch_through` is the path we traverse to perform the prefetch.
        self.prefetch_through = lookup
        # `prefetch_to` is the path to the attribute that stores the result.
        self.prefetch_to = lookup
        if queryset is not None and (
            isinstance(queryset, RawQuerySet)
            or (
                hasattr(queryset, "_iterable_class")
                and not issubclass(queryset._iterable_class, ModelIterable)
            )
        ):
            raise ValueError(
                "Prefetch querysets cannot use raw(), values(), and values_list()."
            )
        if to_attr:
            self.prefetch_to = LOOKUP_SEP.join(
                lookup.split(LOOKUP_SEP)[:-1] + [to_attr]
            )

        self.queryset = queryset
        self.to_attr = to_attr

    def __getstate__(self):
        obj_dict = self.__dict__.copy()
        if self.queryset is not None:
            queryset = self.queryset._chain()
            # Prevent the QuerySet from being evaluated
            queryset._result_cache = []
            queryset._prefetch_done = True
            obj_dict["queryset"] = queryset
        return obj_dict

    def add_prefix(self, prefix):
        self.prefetch_through = prefix + LOOKUP_SEP + self.prefetch_through
        self.prefetch_to = prefix + LOOKUP_SEP + self.prefetch_to

    def get_current_prefetch_to(self, level):
        return LOOKUP_SEP.join(self.prefetch_to.split(LOOKUP_SEP)[: level + 1])

    def get_current_to_attr(self, level):
        parts = self.prefetch_to.split(LOOKUP_SEP)
        to_attr = parts[level]
        as_attr = self.to_attr and level == len(parts) - 1
        return to_attr, as_attr

    def get_current_queryset(self, level):
        if self.get_current_prefetch_to(level) == self.prefetch_to:
            return self.queryset
        return None

    def __eq__(self, other):
        if not isinstance(other, Prefetch):
            return NotImplemented
        return self.prefetch_to == other.prefetch_to

    def __hash__(self):
        return hash((self.__class__, self.prefetch_to))
```
### 68 - django/db/models/query.py:

Start line: 1345, End line: 1398

```python
class QuerySet:

    def _annotate(self, args, kwargs, select=True):
        self._validate_values_are_expressions(
            args + tuple(kwargs.values()), method_name="annotate"
        )
        annotations = {}
        for arg in args:
            # The default_alias property may raise a TypeError.
            try:
                if arg.default_alias in kwargs:
                    raise ValueError(
                        "The named annotation '%s' conflicts with the "
                        "default name for another annotation." % arg.default_alias
                    )
            except TypeError:
                raise TypeError("Complex annotations require an alias")
            annotations[arg.default_alias] = arg
        annotations.update(kwargs)

        clone = self._chain()
        names = self._fields
        if names is None:
            names = set(
                chain.from_iterable(
                    (field.name, field.attname)
                    if hasattr(field, "attname")
                    else (field.name,)
                    for field in self.model._meta.get_fields()
                )
            )

        for alias, annotation in annotations.items():
            if alias in names:
                raise ValueError(
                    "The annotation '%s' conflicts with a field on "
                    "the model." % alias
                )
            if isinstance(annotation, FilteredRelation):
                clone.query.add_filtered_relation(annotation, alias)
            else:
                clone.query.add_annotation(
                    annotation,
                    alias,
                    is_summary=False,
                    select=select,
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
### 73 - django/db/models/query_utils.py:

Start line: 32, End line: 65

```python
class Q(tree.Node):
    """
    Encapsulate filters as objects that can then be combined logically (using
    `&` and `|`).
    """

    # Connection types
    AND = "AND"
    OR = "OR"
    default = AND
    conditional = True

    def __init__(self, *args, _connector=None, _negated=False, **kwargs):
        super().__init__(
            children=[*args, *sorted(kwargs.items())],
            connector=_connector,
            negated=_negated,
        )

    def _combine(self, other, conn):
        if not (isinstance(other, Q) or getattr(other, "conditional", False) is True):
            raise TypeError(other)

        if not self:
            return other.copy() if hasattr(other, "copy") else copy.copy(other)
        elif isinstance(other, Q) and not other:
            _, args, kwargs = self.deconstruct()
            return type(self)(*args, **kwargs)

        obj = type(self)()
        obj.connector = conn
        obj.add(self, conn)
        obj.add(other, conn)
        return obj
```
### 76 - django/db/models/expressions.py:

Start line: 908, End line: 928

```python
class RawSQL(Expression):

    def resolve_expression(
        self, query=None, allow_joins=True, reuse=None, summarize=False, for_save=False
    ):
        # Resolve parents fields used in raw SQL.
        for parent in query.model._meta.get_parent_list():
            for parent_field in parent._meta.local_fields:
                _, column_name = parent_field.get_attname_column()
                if column_name.lower() in self.sql.lower():
                    query.resolve_ref(parent_field.name, allow_joins, reuse, summarize)
                    break
        return super().resolve_expression(
            query, allow_joins, reuse, summarize, for_save
        )


class Star(Expression):
    def __repr__(self):
        return "'*'"

    def as_sql(self, compiler, connection):
        return "*", []
```
### 79 - django/db/backends/base/features.py:

Start line: 6, End line: 219

```python
class BaseDatabaseFeatures:
    # An optional tuple indicating the minimum supported database version.
    minimum_database_version = None
    gis_enabled = False
    # Oracle can't group by LOB (large object) data types.
    allows_group_by_lob = True
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

    # Does the backend ignore unnecessary ORDER BY clauses in subqueries?
    ignores_unnecessary_order_by_in_subqueries = True

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

    # Does the backend support tablespaces? Default to False because it isn't
    # in the SQL standard.
    supports_tablespaces = False

    # Does the backend reset sequences between tests?
    supports_sequence_reset = True

    # Can the backend introspect the default value of a column?
    can_introspect_default = True

    # Confirm support for introspected foreign keys
    # Every database can do this reliably, except MySQL,
    # which can't do it for MyISAM tables
    can_introspect_foreign_keys = True

    # Map fields which some backends may not be able to differentiate to the
    # field it's introspected as.
    introspected_field_types = {
        "AutoField": "AutoField",
        "BigAutoField": "BigAutoField",
        "BigIntegerField": "BigIntegerField",
        "BinaryField": "BinaryField",
        "BooleanField": "BooleanField",
        "CharField": "CharField",
        "DurationField": "DurationField",
        "GenericIPAddressField": "GenericIPAddressField",
        "IntegerField": "IntegerField",
        "PositiveBigIntegerField": "PositiveBigIntegerField",
        "PositiveIntegerField": "PositiveIntegerField",
        "PositiveSmallIntegerField": "PositiveSmallIntegerField",
        "SmallAutoField": "SmallAutoField",
        "SmallIntegerField": "SmallIntegerField",
        "TimeField": "TimeField",
    }

    # Can the backend introspect the column order (ASC/DESC) for indexes?
    supports_index_column_ordering = True

    # Does the backend support introspection of materialized views?
    can_introspect_materialized_views = False

    # Support for the DISTINCT ON clause
    can_distinct_on_fields = False

    # Does the backend prevent running SQL queries in broken transactions?
    atomic_transactions = True

    # Can we roll back DDL in a transaction?
    can_rollback_ddl = False

    # Does it support operations requiring references rename in a transaction?
    supports_atomic_references_rename = True

    # Can we issue more than one ALTER COLUMN clause in an ALTER TABLE?
    supports_combined_alters = False

    # Does it support foreign keys?
    supports_foreign_keys = True

    # Can it create foreign key constraints inline when adding columns?
    can_create_inline_fk = True

    # Does it automatically index foreign keys?
    indexes_foreign_keys = True

    # Does it support CHECK constraints?
    supports_column_check_constraints = True
    supports_table_check_constraints = True
    # Does the backend support introspection of CHECK constraints?
    can_introspect_check_constraints = True

    # Does the backend support 'pyformat' style ("... %(name)s ...", {'name': value})
    # parameter passing? Note this can be provided by the backend even if not
    # supported by the Python driver
    supports_paramstyle_pyformat = True

    # Does the backend require literal defaults, rather than parameterized ones?
    requires_literal_defaults = False

    # Does the backend require a connection reset after each material schema change?
    connection_persists_old_columns = False

    # What kind of error does the backend throw when accessing closed cursor?
    closed_cursor_error_class = ProgrammingError

    # Does 'a' LIKE 'A' match?
    has_case_insensitive_like = False

    # Suffix for backends that don't support "SELECT xxx;" queries.
    bare_select_suffix = ""

    # If NULL is implied on columns without needing to be explicitly specified
    implied_column_null = False

    # Does the backend support "select for update" queries with limit (and offset)?
    supports_select_for_update_with_limit = True

    # Does the backend ignore null expressions in GREATEST and LEAST queries unless
    # every expression is null?
    greatest_least_ignores_nulls = False

    # Can the backend clone databases for parallel test execution?
    # Defaults to False to allow third-party backends to opt-in.
    # ... other code
```
### 80 - django/db/models/query_utils.py:

Start line: 297, End line: 337

```python
class FilteredRelation:
    """Specify custom filtering in the ON clause of SQL joins."""

    def __init__(self, relation_name, *, condition=Q()):
        if not relation_name:
            raise ValueError("relation_name cannot be empty.")
        self.relation_name = relation_name
        self.alias = None
        if not isinstance(condition, Q):
            raise ValueError("condition argument must be a Q() instance.")
        self.condition = condition
        self.path = []

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return (
            self.relation_name == other.relation_name
            and self.alias == other.alias
            and self.condition == other.condition
        )

    def clone(self):
        clone = FilteredRelation(self.relation_name, condition=self.condition)
        clone.alias = self.alias
        clone.path = self.path[:]
        return clone

    def resolve_expression(self, *args, **kwargs):
        """
        QuerySet.annotate() only accepts expression-like arguments
        (with a resolve_expression() method).
        """
        raise NotImplementedError("FilteredRelation.resolve_expression() is unused.")

    def as_sql(self, compiler, connection):
        # Resolve the condition in Join.filtered_relation.
        query = compiler.query
        where = query.build_filtered_relation_q(self.condition, reuse=set(self.path))
        return compiler.compile(where)
```
### 81 - django/db/models/query.py:

Start line: 399, End line: 416

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
### 83 - django/db/models/sql/where.py:

Start line: 1, End line: 31

```python
"""
Code to manage the creation and SQL rendering of 'where' constraints.
"""

from django.core.exceptions import EmptyResultSet
from django.utils import tree
from django.utils.functional import cached_property

# Connection types
AND = "AND"
OR = "OR"


class WhereNode(tree.Node):
    """
    An SQL WHERE clause.

    The class is tied to the Query class that created it (in order to create
    the correct SQL).

    A child is usually an expression producing boolean values. Most likely the
    expression is a Lookup instance.

    However, a child could also be any class with as_sql() and either
    relabeled_clone() method or relabel_aliases() and clone() methods and
    contains_aggregate attribute.
    """

    default = AND
    resolved = False
    conditional = True
```
### 84 - django/db/models/query.py:

Start line: 1159, End line: 1207

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
        self._not_support_combined_queries("filter")
        return self._filter_or_exclude(False, args, kwargs)

    def exclude(self, *args, **kwargs):
        """
        Return a new QuerySet instance with NOT (args) ANDed to the existing
        set.
        """
        self._not_support_combined_queries("exclude")
        return self._filter_or_exclude(True, args, kwargs)

    def _filter_or_exclude(self, negate, args, kwargs):
        if (args or kwargs) and self.query.is_sliced:
            raise TypeError("Cannot filter a query once a slice has been taken.")
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
### 91 - django/db/models/sql/where.py:

Start line: 213, End line: 237

```python
class WhereNode(tree.Node):

    def resolve_expression(self, *args, **kwargs):
        clone = self.clone()
        clone._resolve_node(clone, *args, **kwargs)
        clone.resolved = True
        return clone

    @cached_property
    def output_field(self):
        from django.db.models import BooleanField

        return BooleanField()

    def select_format(self, compiler, sql, params):
        # Wrap filters with a CASE WHEN expression if a database backend
        # (e.g. Oracle) doesn't support boolean expression in SELECT or GROUP
        # BY list.
        if not compiler.connection.features.supports_boolean_expr_in_select_clause:
            sql = f"CASE WHEN {sql} THEN 1 ELSE 0 END"
        return sql, params

    def get_db_converters(self, connection):
        return self.output_field.get_db_converters(connection)

    def get_lookup(self, lookup):
        return self.output_field.get_lookup(lookup)
```
### 96 - django/db/models/query.py:

Start line: 1646, End line: 1656

```python
class QuerySet:

    def _merge_sanity_check(self, other):
        """Check that two QuerySet classes may be merged."""
        if self._fields is not None and (
            set(self.query.values_select) != set(other.query.values_select)
            or set(self.query.extra_select) != set(other.query.extra_select)
            or set(self.query.annotation_select) != set(other.query.annotation_select)
        ):
            raise TypeError(
                "Merging '%s' classes must involve the same values in each case."
                % self.__class__.__name__
            )
```
### 99 - django/db/models/expressions.py:

Start line: 835, End line: 861

```python
@deconstructible(path="django.db.models.Value")
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
            if hasattr(output_field, "get_placeholder"):
                return output_field.get_placeholder(val, compiler, connection), [val]
        if val is None:
            # cx_Oracle does not always convert None to the appropriate
            # NULL type (like in case expressions using numbers), so we
            # use a literal SQL NULL
            return "NULL", []
        return "%s", [val]

    def resolve_expression(
        self, query=None, allow_joins=True, reuse=None, summarize=False, for_save=False
    ):
        c = super().resolve_expression(query, allow_joins, reuse, summarize, for_save)
        c.for_save = for_save
        return c

    def get_group_by_cols(self, alias=None):
        return []
```
### 103 - django/db/models/expressions.py:

Start line: 892, End line: 906

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
        return "(%s)" % self.sql, self.params

    def get_group_by_cols(self, alias=None):
        return [self]
```
### 106 - django/db/models/query.py:

Start line: 1400, End line: 1445

```python
class QuerySet:

    def order_by(self, *field_names):
        """Return a new QuerySet instance with the ordering changed."""
        if self.query.is_sliced:
            raise TypeError("Cannot reorder a query once a slice has been taken.")
        obj = self._chain()
        obj.query.clear_ordering(force=True, clear_default=False)
        obj.query.add_ordering(*field_names)
        return obj

    def distinct(self, *field_names):
        """
        Return a new QuerySet instance that will select only distinct results.
        """
        self._not_support_combined_queries("distinct")
        if self.query.is_sliced:
            raise TypeError(
                "Cannot create distinct fields once a slice has been taken."
            )
        obj = self._chain()
        obj.query.add_distinct_fields(*field_names)
        return obj

    def extra(
        self,
        select=None,
        where=None,
        params=None,
        tables=None,
        order_by=None,
        select_params=None,
    ):
        """Add extra SQL fragments to the query."""
        self._not_support_combined_queries("extra")
        if self.query.is_sliced:
            raise TypeError("Cannot change a query once a slice has been taken.")
        clone = self._chain()
        clone.query.add_extra(select, select_params, where, params, tables, order_by)
        return clone

    def reverse(self):
        """Reverse the ordering of the QuerySet."""
        if self.query.is_sliced:
            raise TypeError("Cannot reverse a query once a slice has been taken.")
        clone = self._chain()
        clone.query.standard_ordering = not clone.query.standard_ordering
        return clone
```
### 110 - django/db/models/expressions.py:

Start line: 1084, End line: 1155

```python
@deconstructible(path="django.db.models.When")
class When(Expression):
    template = "WHEN %(condition)s THEN %(result)s"
    # This isn't a complete conditional expression, must be used in Case().
    conditional = False

    def __init__(self, condition=None, then=None, **lookups):
        if lookups:
            if condition is None:
                condition, lookups = Q(**lookups), None
            elif getattr(condition, "conditional", False):
                condition, lookups = Q(condition, **lookups), None
        if condition is None or not getattr(condition, "conditional", False) or lookups:
            raise TypeError(
                "When() supports a Q object, a boolean expression, or lookups "
                "as a condition."
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

    def resolve_expression(
        self, query=None, allow_joins=True, reuse=None, summarize=False, for_save=False
    ):
        c = self.copy()
        c.is_summary = summarize
        if hasattr(c.condition, "resolve_expression"):
            c.condition = c.condition.resolve_expression(
                query, allow_joins, reuse, summarize, False
            )
        c.result = c.result.resolve_expression(
            query, allow_joins, reuse, summarize, for_save
        )
        return c

    def as_sql(self, compiler, connection, template=None, **extra_context):
        connection.ops.check_expression_support(self)
        template_params = extra_context
        sql_params = []
        condition_sql, condition_params = compiler.compile(self.condition)
        template_params["condition"] = condition_sql
        sql_params.extend(condition_params)
        result_sql, result_params = compiler.compile(self.result)
        template_params["result"] = result_sql
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
### 112 - django/db/backends/mysql/features.py:

Start line: 1, End line: 56

```python
import operator

from django.db.backends.base.features import BaseDatabaseFeatures
from django.utils.functional import cached_property


class DatabaseFeatures(BaseDatabaseFeatures):
    empty_fetchmany_value = ()
    allows_group_by_pk = True
    related_fields_match_type = True
    # MySQL doesn't support sliced subqueries with IN/ALL/ANY/SOME.
    allow_sliced_subqueries_with_in = False
    has_select_for_update = True
    supports_forward_references = False
    supports_regex_backreferencing = False
    supports_date_lookup_using_string = False
    supports_timezones = False
    requires_explicit_null_ordering_when_grouping = True
    can_release_savepoints = True
    atomic_transactions = False
    can_clone_databases = True
    supports_temporal_subtraction = True
    supports_select_intersection = False
    supports_select_difference = False
    supports_slicing_ordering_in_compound = True
    supports_index_on_text_field = False
    supports_update_conflicts = True
    create_test_procedure_without_params_sql = """
        CREATE PROCEDURE test_procedure ()
        BEGIN
            DECLARE V_I INTEGER;
            SET V_I = 1;
        END;
    """
    create_test_procedure_with_int_param_sql = """
        CREATE PROCEDURE test_procedure (P_I INTEGER)
        BEGIN
            DECLARE V_I INTEGER;
            SET V_I = P_I;
        END;
    """
    # Neither MySQL nor MariaDB support partial indexes.
    supports_partial_indexes = False
    # COLLATE must be wrapped in parentheses because MySQL treats COLLATE as an
    # indexed expression.
    collate_as_index_expression = True

    supports_order_by_nulls_modifier = False
    order_by_nulls_first = True

    @cached_property
    def minimum_database_version(self):
        if self.connection.mysql_is_mariadb:
            return (10, 2)
        else:
            return (5, 7)
```
### 120 - django/db/models/expressions.py:

Start line: 34, End line: 148

```python
class Combinable:
    """
    Provide the ability to combine one or two objects with
    some connector. For example F('foo') + F('bar').
    """

    # Arithmetic connectors
    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"
    POW = "^"
    # The following is a quoted % operator - it is quoted because it can be
    # used in strings that also have parameter substitution.
    MOD = "%%"

    # Bitwise operators - note that these are generated by .bitand()
    # and .bitor(), the '&' and '|' are reserved for boolean operator
    # usage.
    BITAND = "&"
    BITOR = "|"
    BITLEFTSHIFT = "<<"
    BITRIGHTSHIFT = ">>"
    BITXOR = "#"

    def _combine(self, other, connector, reversed):
        if not hasattr(other, "resolve_expression"):
            # everything must be resolvable to an expression
            other = Value(other)

        if reversed:
            return CombinedExpression(other, connector, self)
        return CombinedExpression(self, connector, other)

    #############
    # OPERATORS #
    #############

    def __neg__(self):
        return self._combine(-1, self.MUL, False)

    def __add__(self, other):
        return self._combine(other, self.ADD, False)

    def __sub__(self, other):
        return self._combine(other, self.SUB, False)

    def __mul__(self, other):
        return self._combine(other, self.MUL, False)

    def __truediv__(self, other):
        return self._combine(other, self.DIV, False)

    def __mod__(self, other):
        return self._combine(other, self.MOD, False)

    def __pow__(self, other):
        return self._combine(other, self.POW, False)

    def __and__(self, other):
        if getattr(self, "conditional", False) and getattr(other, "conditional", False):
            return Q(self) & Q(other)
        raise NotImplementedError(
            "Use .bitand() and .bitor() for bitwise logical operations."
        )

    def bitand(self, other):
        return self._combine(other, self.BITAND, False)

    def bitleftshift(self, other):
        return self._combine(other, self.BITLEFTSHIFT, False)

    def bitrightshift(self, other):
        return self._combine(other, self.BITRIGHTSHIFT, False)

    def bitxor(self, other):
        return self._combine(other, self.BITXOR, False)

    def __or__(self, other):
        if getattr(self, "conditional", False) and getattr(other, "conditional", False):
            return Q(self) | Q(other)
        raise NotImplementedError(
            "Use .bitand() and .bitor() for bitwise logical operations."
        )

    def bitor(self, other):
        return self._combine(other, self.BITOR, False)

    def __radd__(self, other):
        return self._combine(other, self.ADD, True)

    def __rsub__(self, other):
        return self._combine(other, self.SUB, True)

    def __rmul__(self, other):
        return self._combine(other, self.MUL, True)

    def __rtruediv__(self, other):
        return self._combine(other, self.DIV, True)

    def __rmod__(self, other):
        return self._combine(other, self.MOD, True)

    def __rpow__(self, other):
        return self._combine(other, self.POW, True)

    def __rand__(self, other):
        raise NotImplementedError(
            "Use .bitand() and .bitor() for bitwise logical operations."
        )

    def __ror__(self, other):
        raise NotImplementedError(
            "Use .bitand() and .bitor() for bitwise logical operations."
        )
```
### 122 - django/db/models/query.py:

Start line: 852, End line: 870

```python
class QuerySet:

    def earliest(self, *fields):
        if self.query.is_sliced:
            raise TypeError("Cannot change a query once a slice has been taken.")
        return self._earliest(*fields)

    def latest(self, *fields):
        if self.query.is_sliced:
            raise TypeError("Cannot change a query once a slice has been taken.")
        return self.reverse()._earliest(*fields)

    def first(self):
        """Return the first object of a query or None if no match is found."""
        for obj in (self if self.ordered else self.order_by("pk"))[:1]:
            return obj

    def last(self):
        """Return the last object of a query or None if no match is found."""
        for obj in (self.reverse() if self.ordered else self.order_by("-pk"))[:1]:
            return obj
```
### 125 - django/db/models/query.py:

Start line: 1305, End line: 1328

```python
class QuerySet:

    def prefetch_related(self, *lookups):
        """
        Return a new QuerySet instance that will prefetch the specified
        Many-To-One and Many-To-Many related objects when the QuerySet is
        evaluated.

        When prefetch_related() is called more than once, append to the list of
        prefetch lookups. If prefetch_related(None) is called, clear the list.
        """
        self._not_support_combined_queries("prefetch_related")
        clone = self._chain()
        if lookups == (None,):
            clone._prefetch_related_lookups = ()
        else:
            for lookup in lookups:
                if isinstance(lookup, Prefetch):
                    lookup = lookup.prefetch_to
                lookup = lookup.split(LOOKUP_SEP, 1)[0]
                if lookup in self.query._filtered_relations:
                    raise ValueError(
                        "prefetch_related() is not supported with FilteredRelation."
                    )
            clone._prefetch_related_lookups = clone._prefetch_related_lookups + lookups
        return clone
```
### 126 - django/db/models/expressions.py:

Start line: 1408, End line: 1439

```python
@deconstructible(path="django.db.models.OrderBy")
class OrderBy(Expression):

    def as_oracle(self, compiler, connection):
        # Oracle doesn't allow ORDER BY EXISTS() or filters unless it's wrapped
        # in a CASE WHEN.
        if connection.ops.conditional_expression_supported_in_where_clause(
            self.expression
        ):
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
### 127 - django/db/models/expressions.py:

Start line: 1198, End line: 1215

```python
@deconstructible(path="django.db.models.Case")
class Case(SQLiteNumericMixin, Expression):

    def resolve_expression(
        self, query=None, allow_joins=True, reuse=None, summarize=False, for_save=False
    ):
        c = self.copy()
        c.is_summary = summarize
        for pos, case in enumerate(c.cases):
            c.cases[pos] = case.resolve_expression(
                query, allow_joins, reuse, summarize, for_save
            )
        c.default = c.default.resolve_expression(
            query, allow_joins, reuse, summarize, for_save
        )
        return c

    def copy(self):
        c = super().copy()
        c.cases = c.cases[:]
        return c
```
### 128 - django/db/models/query.py:

Start line: 277, End line: 304

```python
class QuerySet:

    def __setstate__(self, state):
        pickled_version = state.get(DJANGO_VERSION_PICKLE_KEY)
        if pickled_version:
            if pickled_version != django.__version__:
                warnings.warn(
                    "Pickled queryset instance's Django version %s does not "
                    "match the current version %s."
                    % (pickled_version, django.__version__),
                    RuntimeWarning,
                    stacklevel=2,
                )
        else:
            warnings.warn(
                "Pickled queryset instance's Django version is not specified.",
                RuntimeWarning,
                stacklevel=2,
            )
        self.__dict__.update(state)

    def __repr__(self):
        data = list(self[: REPR_OUTPUT_SIZE + 1])
        if len(data) > REPR_OUTPUT_SIZE:
            data[-1] = "...(remaining elements truncated)..."
        return "<%s %r>" % (self.__class__.__name__, data)

    def __len__(self):
        self._fetch_all()
        return len(self._result_cache)
```
### 135 - django/db/models/expressions.py:

Start line: 1252, End line: 1308

```python
class Subquery(BaseExpression, Combinable):
    """
    An explicit subquery. It may contain OuterRef() references to the outer
    query which will be resolved when it is applied to that query.
    """

    template = "(%(subquery)s)"
    contains_aggregate = False
    empty_result_set_value = None

    def __init__(self, queryset, output_field=None, **extra):
        # Allow the usage of both QuerySet and sql.Query objects.
        self.query = getattr(queryset, "query", queryset).clone()
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
        template_params["subquery"] = subquery_sql[1:-1]

        template = template or template_params.get("template", self.template)
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
### 136 - django/db/backends/base/features.py:

Start line: 353, End line: 371

```python
class BaseDatabaseFeatures:
    minimum_database_version =
    # ... other code

    @cached_property
    def supports_transactions(self):
        """Confirm support for transactions."""
        with self.connection.cursor() as cursor:
            cursor.execute("CREATE TABLE ROLLBACK_TEST (X INT)")
            self.connection.set_autocommit(False)
            cursor.execute("INSERT INTO ROLLBACK_TEST (X) VALUES (8)")
            self.connection.rollback()
            self.connection.set_autocommit(True)
            cursor.execute("SELECT COUNT(X) FROM ROLLBACK_TEST")
            (count,) = cursor.fetchone()
            cursor.execute("DROP TABLE ROLLBACK_TEST")
        return count == 0

    def allows_group_by_selected_pks_on_model(self, model):
        if not self.allows_group_by_selected_pks:
            return False
        return model._meta.managed
```
### 137 - django/db/models/expressions.py:

Start line: 1217, End line: 1249

```python
@deconstructible(path="django.db.models.Case")
class Case(SQLiteNumericMixin, Expression):

    def as_sql(
        self, compiler, connection, template=None, case_joiner=None, **extra_context
    ):
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
        template_params["cases"] = case_joiner.join(case_parts)
        template_params["default"] = default_sql
        sql_params.extend(default_params)
        template = template or template_params.get("template", self.template)
        sql = template % template_params
        if self._output_field_or_none is not None:
            sql = connection.ops.unification_cast_sql(self.output_field) % sql
        return sql, sql_params

    def get_group_by_cols(self, alias=None):
        if not self.cases:
            return self.default.get_group_by_cols(alias)
        return super().get_group_by_cols(alias)
```
### 139 - django/db/backends/mysql/features.py:

Start line: 276, End line: 329

```python
class DatabaseFeatures(BaseDatabaseFeatures):

    @cached_property
    def supported_explain_formats(self):
        # Alias MySQL's TRADITIONAL to TEXT for consistency with other
        # backends.
        formats = {"JSON", "TEXT", "TRADITIONAL"}
        if not self.connection.mysql_is_mariadb and self.connection.mysql_version >= (
            8,
            0,
            16,
        ):
            formats.add("TREE")
        return formats

    @cached_property
    def supports_transactions(self):
        """
        All storage engines except MyISAM support transactions.
        """
        return self._mysql_storage_engine != "MyISAM"

    @cached_property
    def ignores_table_name_case(self):
        return self.connection.mysql_server_data["lower_case_table_names"]

    @cached_property
    def supports_default_in_lead_lag(self):
        # To be added in https://jira.mariadb.org/browse/MDEV-12981.
        return not self.connection.mysql_is_mariadb

    @cached_property
    def supports_json_field(self):
        if self.connection.mysql_is_mariadb:
            return True
        return self.connection.mysql_version >= (5, 7, 8)

    @cached_property
    def can_introspect_json_field(self):
        if self.connection.mysql_is_mariadb:
            return self.supports_json_field and self.can_introspect_check_constraints
        return self.supports_json_field

    @cached_property
    def supports_index_column_ordering(self):
        if self.connection.mysql_is_mariadb:
            return self.connection.mysql_version >= (10, 8)
        return self.connection.mysql_version >= (8, 0, 1)

    @cached_property
    def supports_expression_indexes(self):
        return (
            not self.connection.mysql_is_mariadb
            and self.connection.mysql_version >= (8, 0, 13)
        )
```
### 144 - django/db/backends/mysql/features.py:

Start line: 75, End line: 164

```python
class DatabaseFeatures(BaseDatabaseFeatures):

    @cached_property
    def django_test_skips(self):
        skips = {
            "This doesn't work on MySQL.": {
                "db_functions.comparison.test_greatest.GreatestTests."
                "test_coalesce_workaround",
                "db_functions.comparison.test_least.LeastTests."
                "test_coalesce_workaround",
            },
            "Running on MySQL requires utf8mb4 encoding (#18392).": {
                "model_fields.test_textfield.TextFieldTests.test_emoji",
                "model_fields.test_charfield.TestCharField.test_emoji",
            },
            "MySQL doesn't support functional indexes on a function that "
            "returns JSON": {
                "schema.tests.SchemaTests.test_func_index_json_key_transform",
            },
            "MySQL supports multiplying and dividing DurationFields by a "
            "scalar value but it's not implemented (#25287).": {
                "expressions.tests.FTimeDeltaTests.test_durationfield_multiply_divide",
            },
        }
        if "ONLY_FULL_GROUP_BY" in self.connection.sql_mode:
            skips.update(
                {
                    "GROUP BY optimization does not work properly when "
                    "ONLY_FULL_GROUP_BY mode is enabled on MySQL, see #31331.": {
                        "aggregation.tests.AggregateTestCase."
                        "test_aggregation_subquery_annotation_multivalued",
                        "annotations.tests.NonAggregateAnnotationTestCase."
                        "test_annotation_aggregate_with_m2o",
                    },
                }
            )
        if not self.connection.mysql_is_mariadb and self.connection.mysql_version < (
            8,
        ):
            skips.update(
                {
                    "Casting to datetime/time is not supported by MySQL < 8.0. "
                    "(#30224)": {
                        "aggregation.tests.AggregateTestCase."
                        "test_aggregation_default_using_time_from_python",
                        "aggregation.tests.AggregateTestCase."
                        "test_aggregation_default_using_datetime_from_python",
                    },
                    "MySQL < 8.0 returns string type instead of datetime/time. "
                    "(#30224)": {
                        "aggregation.tests.AggregateTestCase."
                        "test_aggregation_default_using_time_from_database",
                        "aggregation.tests.AggregateTestCase."
                        "test_aggregation_default_using_datetime_from_database",
                    },
                }
            )
        if self.connection.mysql_is_mariadb and (
            10,
            4,
            3,
        ) < self.connection.mysql_version < (10, 5, 2):
            skips.update(
                {
                    "https://jira.mariadb.org/browse/MDEV-19598": {
                        "schema.tests.SchemaTests."
                        "test_alter_not_unique_field_to_primary_key",
                    },
                }
            )
        if self.connection.mysql_is_mariadb and (
            10,
            4,
            12,
        ) < self.connection.mysql_version < (10, 5):
            skips.update(
                {
                    "https://jira.mariadb.org/browse/MDEV-22775": {
                        "schema.tests.SchemaTests."
                        "test_alter_pk_with_self_referential_field",
                    },
                }
            )
        if not self.supports_explain_analyze:
            skips.update(
                {
                    "MariaDB and MySQL >= 8.0.18 specific.": {
                        "queries.test_explain.ExplainTests.test_mysql_analyze",
                    },
                }
            )
        return skips
```
### 153 - django/db/models/query.py:

Start line: 551, End line: 610

```python
class QuerySet:

    def _check_bulk_create_options(
        self, ignore_conflicts, update_conflicts, update_fields, unique_fields
    ):
        if ignore_conflicts and update_conflicts:
            raise ValueError(
                "ignore_conflicts and update_conflicts are mutually exclusive."
            )
        db_features = connections[self.db].features
        if ignore_conflicts:
            if not db_features.supports_ignore_conflicts:
                raise NotSupportedError(
                    "This database backend does not support ignoring conflicts."
                )
            return OnConflict.IGNORE
        elif update_conflicts:
            if not db_features.supports_update_conflicts:
                raise NotSupportedError(
                    "This database backend does not support updating conflicts."
                )
            if not update_fields:
                raise ValueError(
                    "Fields that will be updated when a row insertion fails "
                    "on conflicts must be provided."
                )
            if unique_fields and not db_features.supports_update_conflicts_with_target:
                raise NotSupportedError(
                    "This database backend does not support updating "
                    "conflicts with specifying unique fields that can trigger "
                    "the upsert."
                )
            if not unique_fields and db_features.supports_update_conflicts_with_target:
                raise ValueError(
                    "Unique fields that can trigger the upsert must be provided."
                )
            # Updating primary keys and non-concrete fields is forbidden.
            update_fields = [self.model._meta.get_field(name) for name in update_fields]
            if any(not f.concrete or f.many_to_many for f in update_fields):
                raise ValueError(
                    "bulk_create() can only be used with concrete fields in "
                    "update_fields."
                )
            if any(f.primary_key for f in update_fields):
                raise ValueError(
                    "bulk_create() cannot be used with primary keys in "
                    "update_fields."
                )
            if unique_fields:
                # Primary key is allowed in unique_fields.
                unique_fields = [
                    self.model._meta.get_field(name)
                    for name in unique_fields
                    if name != "pk"
                ]
                if any(not f.concrete or f.many_to_many for f in unique_fields):
                    raise ValueError(
                        "bulk_create() can only be used with concrete fields "
                        "in unique_fields."
                    )
            return OnConflict.UPDATE
        return None
```
### 157 - django/db/models/query.py:

Start line: 1464, End line: 1483

```python
class QuerySet:

    def only(self, *fields):
        """
        Essentially, the opposite of defer(). Only the fields passed into this
        method and that are not already specified as deferred are loaded
        immediately when the queryset is evaluated.
        """
        self._not_support_combined_queries("only")
        if self._fields is not None:
            raise TypeError("Cannot call only() after .values() or .values_list()")
        if fields == (None,):
            # Can only pass None to defer(), not only(), as the rest option.
            # That won't stop people trying to do this, so let's be explicit.
            raise TypeError("Cannot pass None as an argument to only().")
        for field in fields:
            field = field.split(LOOKUP_SEP, 1)[0]
            if field in self.query._filtered_relations:
                raise ValueError("only() is not supported with FilteredRelation.")
        clone = self._chain()
        clone.query.add_immediate_loading(fields)
        return clone
```
### 164 - django/db/models/expressions.py:

Start line: 478, End line: 522

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
        expression_wrapper = "(%s)"
        sql = connection.ops.combine_expression(self.connector, expressions)
        return expression_wrapper % sql, expression_params
```
### 168 - django/db/models/query.py:

Start line: 1, End line: 46

```python
"""
The main QuerySet implementation. This provides the public API for the ORM.
"""

import copy
import operator
import warnings
from itertools import chain, islice

import django
from django.conf import settings
from django.core import exceptions
from django.db import (
    DJANGO_VERSION_PICKLE_KEY,
    IntegrityError,
    NotSupportedError,
    connections,
    router,
    transaction,
)
from django.db.models import AutoField, DateField, DateTimeField, sql
from django.db.models.constants import LOOKUP_SEP, OnConflict
from django.db.models.deletion import Collector
from django.db.models.expressions import Case, F, Ref, Value, When
from django.db.models.functions import Cast, Trunc
from django.db.models.query_utils import FilteredRelation, Q
from django.db.models.sql.constants import CURSOR, GET_ITERATOR_CHUNK_SIZE
from django.db.models.utils import create_namedtuple_class, resolve_callables
from django.utils import timezone
from django.utils.deprecation import RemovedInDjango50Warning
from django.utils.functional import cached_property, partition

# The maximum number of results to fetch in a get() query.
MAX_GET_RESULTS = 21

# The maximum number of items to display in a QuerySet.__repr__
REPR_OUTPUT_SIZE = 20


class BaseIterable:
    def __init__(
        self, queryset, chunked_fetch=False, chunk_size=GET_ITERATOR_CHUNK_SIZE
    ):
        self.queryset = queryset
        self.chunked_fetch = chunked_fetch
        self.chunk_size = chunk_size
```
