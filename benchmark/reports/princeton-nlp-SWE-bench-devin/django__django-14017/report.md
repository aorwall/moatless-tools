# django__django-14017

| **django/django** | `466920f6d726eee90d5566e0a9948e92b33a122e` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 1839 |
| **Any found context length** | 1839 |
| **Avg pos** | 8.0 |
| **Min pos** | 8 |
| **Max pos** | 8 |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/db/models/query_utils.py b/django/db/models/query_utils.py
--- a/django/db/models/query_utils.py
+++ b/django/db/models/query_utils.py
@@ -40,7 +40,7 @@ def __init__(self, *args, _connector=None, _negated=False, **kwargs):
         super().__init__(children=[*args, *sorted(kwargs.items())], connector=_connector, negated=_negated)
 
     def _combine(self, other, conn):
-        if not isinstance(other, Q):
+        if not(isinstance(other, Q) or getattr(other, 'conditional', False) is True):
             raise TypeError(other)
 
         # If the other Q() is empty, ignore it and just use `self`.

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/db/models/query_utils.py | 43 | 43 | 8 | 1 | 1839


## Problem Statement

```
Q(...) & Exists(...) raises a TypeError
Description
	
Exists(...) & Q(...) works, but Q(...) & Exists(...) raise a TypeError
Here's a minimal example:
In [3]: Exists(Product.objects.all()) & Q()
Out[3]: <Q: (AND: <django.db.models.expressions.Exists object at 0x7fc18dd0ed90>, (AND: ))>
In [4]: Q() & Exists(Product.objects.all())
---------------------------------------------------------------------------
TypeError								 Traceback (most recent call last)
<ipython-input-4-21d3dea0fcb9> in <module>
----> 1 Q() & Exists(Product.objects.all())
~/Code/venv/ecom/lib/python3.8/site-packages/django/db/models/query_utils.py in __and__(self, other)
	 90 
	 91	 def __and__(self, other):
---> 92		 return self._combine(other, self.AND)
	 93 
	 94	 def __invert__(self):
~/Code/venv/ecom/lib/python3.8/site-packages/django/db/models/query_utils.py in _combine(self, other, conn)
	 71	 def _combine(self, other, conn):
	 72		 if not isinstance(other, Q):
---> 73			 raise TypeError(other)
	 74 
	 75		 # If the other Q() is empty, ignore it and just use `self`.
TypeError: <django.db.models.expressions.Exists object at 0x7fc18dd21400>
The & (and |) operators should be commutative on Q-Exists pairs, but it's not
I think there's a missing definition of __rand__ somewhere.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 django/db/models/query_utils.py** | 61 | 81| 181 | 181 | 2551 | 
| 2 | 2 django/db/models/expressions.py | 1150 | 1182| 254 | 435 | 13524 | 
| 3 | 3 django/db/models/query.py | 320 | 346| 222 | 657 | 30859 | 
| 4 | 3 django/db/models/query.py | 1009 | 1022| 126 | 783 | 30859 | 
| 5 | 4 django/contrib/postgres/search.py | 130 | 157| 248 | 1031 | 33081 | 
| 6 | 4 django/db/models/query.py | 987 | 1007| 239 | 1270 | 33081 | 
| 7 | 4 django/db/models/query.py | 803 | 842| 322 | 1592 | 33081 | 
| **-> 8 <-** | **4 django/db/models/query_utils.py** | 28 | 59| 247 | 1839 | 33081 | 
| 9 | 5 django/db/models/sql/query.py | 1401 | 1421| 247 | 2086 | 55472 | 
| 10 | 5 django/db/models/sql/query.py | 1378 | 1399| 250 | 2336 | 55472 | 
| 11 | 5 django/db/models/sql/query.py | 641 | 665| 269 | 2605 | 55472 | 
| 12 | 6 django/db/models/__init__.py | 1 | 53| 619 | 3224 | 56091 | 
| 13 | 6 django/db/models/sql/query.py | 1132 | 1164| 338 | 3562 | 56091 | 
| 14 | 6 django/db/models/sql/query.py | 1655 | 1679| 227 | 3789 | 56091 | 
| 15 | 6 django/db/models/query.py | 1342 | 1351| 114 | 3903 | 56091 | 
| 16 | 6 django/db/models/sql/query.py | 1765 | 1830| 666 | 4569 | 56091 | 
| 17 | 6 django/db/models/query.py | 1353 | 1401| 405 | 4974 | 56091 | 
| 18 | 6 django/db/models/sql/query.py | 1311 | 1376| 772 | 5746 | 56091 | 
| 19 | 6 django/db/models/sql/query.py | 1445 | 1472| 283 | 6029 | 56091 | 
| 20 | 6 django/db/models/expressions.py | 940 | 1004| 591 | 6620 | 56091 | 
| 21 | 6 django/db/models/query.py | 918 | 968| 371 | 6991 | 56091 | 
| 22 | 6 django/db/models/sql/query.py | 1490 | 1575| 801 | 7792 | 56091 | 
| 23 | **6 django/db/models/query_utils.py** | 83 | 97| 157 | 7949 | 56091 | 
| 24 | 6 django/db/models/expressions.py | 33 | 147| 836 | 8785 | 56091 | 
| 25 | 6 django/db/models/sql/query.py | 1 | 63| 460 | 9245 | 56091 | 
| 26 | 6 django/db/models/sql/query.py | 566 | 640| 809 | 10054 | 56091 | 
| 27 | 7 django/core/exceptions.py | 107 | 218| 752 | 10806 | 57280 | 
| 28 | 7 django/db/models/sql/query.py | 716 | 751| 389 | 11195 | 57280 | 
| 29 | 7 django/db/models/sql/query.py | 1004 | 1035| 307 | 11502 | 57280 | 
| 30 | 8 django/db/models/base.py | 1767 | 1867| 729 | 12231 | 74340 | 
| 31 | 9 django/db/models/sql/compiler.py | 1208 | 1229| 223 | 12454 | 88731 | 
| 32 | 10 django/db/models/constraints.py | 32 | 76| 372 | 12826 | 90346 | 
| 33 | 11 django/db/models/lookups.py | 101 | 155| 427 | 13253 | 95301 | 
| 34 | 12 django/db/models/fields/related_lookups.py | 121 | 157| 244 | 13497 | 96754 | 
| 35 | 12 django/db/models/sql/query.py | 1721 | 1763| 436 | 13933 | 96754 | 
| 36 | 12 django/db/models/query.py | 1100 | 1141| 323 | 14256 | 96754 | 
| 37 | 12 django/db/models/lookups.py | 316 | 366| 306 | 14562 | 96754 | 
| 38 | 12 django/db/models/lookups.py | 1 | 41| 313 | 14875 | 96754 | 
| 39 | 12 django/contrib/postgres/search.py | 160 | 195| 313 | 15188 | 96754 | 
| 40 | 12 django/db/models/sql/query.py | 912 | 934| 248 | 15436 | 96754 | 
| 41 | **12 django/db/models/query_utils.py** | 257 | 282| 293 | 15729 | 96754 | 
| 42 | 12 django/db/models/query.py | 1404 | 1435| 245 | 15974 | 96754 | 
| 43 | 12 django/db/models/sql/query.py | 2414 | 2469| 827 | 16801 | 96754 | 
| 44 | 13 django/db/backends/postgresql/features.py | 1 | 97| 767 | 17568 | 97521 | 
| 45 | 13 django/db/models/query.py | 265 | 285| 180 | 17748 | 97521 | 
| 46 | 13 django/db/models/sql/compiler.py | 1 | 19| 170 | 17918 | 97521 | 
| 47 | 13 django/db/models/sql/query.py | 1863 | 1912| 329 | 18247 | 97521 | 
| 48 | 14 django/db/models/deletion.py | 1 | 76| 566 | 18813 | 101349 | 
| 49 | 14 django/db/models/query.py | 596 | 614| 178 | 18991 | 101349 | 
| 50 | 14 django/db/models/query.py | 569 | 594| 202 | 19193 | 101349 | 
| 51 | 14 django/db/models/fields/related_lookups.py | 62 | 101| 451 | 19644 | 101349 | 
| 52 | 14 django/db/models/lookups.py | 572 | 597| 163 | 19807 | 101349 | 
| 53 | 14 django/db/models/sql/query.py | 1103 | 1130| 285 | 20092 | 101349 | 
| 54 | 15 django/db/models/sql/where.py | 233 | 249| 130 | 20222 | 103163 | 
| 55 | 15 django/db/models/sql/query.py | 1072 | 1101| 245 | 20467 | 103163 | 
| 56 | 16 django/contrib/gis/db/models/lookups.py | 86 | 217| 762 | 21229 | 105776 | 
| 57 | 17 django/contrib/postgres/lookups.py | 1 | 61| 337 | 21566 | 106113 | 
| 58 | 17 django/db/models/query.py | 1322 | 1340| 186 | 21752 | 106113 | 
| 59 | 18 django/db/models/sql/datastructures.py | 1 | 21| 126 | 21878 | 107581 | 
| 60 | 18 django/db/models/constraints.py | 1 | 29| 213 | 22091 | 107581 | 
| 61 | 18 django/db/models/sql/query.py | 1166 | 1209| 469 | 22560 | 107581 | 
| 62 | 18 django/db/models/sql/where.py | 1 | 30| 167 | 22727 | 107581 | 
| 63 | 18 django/db/models/sql/query.py | 517 | 564| 403 | 23130 | 107581 | 
| 64 | 19 django/db/migrations/questioner.py | 226 | 239| 123 | 23253 | 109638 | 
| 65 | 20 django/db/models/fields/related.py | 1 | 34| 246 | 23499 | 123461 | 
| 66 | 20 django/db/models/query.py | 1198 | 1217| 209 | 23708 | 123461 | 
| 67 | 20 django/contrib/postgres/search.py | 198 | 230| 243 | 23951 | 123461 | 
| 68 | 21 django/contrib/postgres/constraints.py | 1 | 67| 550 | 24501 | 124886 | 
| 69 | 21 django/db/models/sql/where.py | 157 | 193| 243 | 24744 | 124886 | 
| 70 | 22 django/db/backends/oracle/operations.py | 561 | 577| 290 | 25034 | 130797 | 
| 71 | 22 django/db/models/expressions.py | 492 | 517| 303 | 25337 | 130797 | 
| 72 | 23 django/db/models/aggregates.py | 45 | 68| 294 | 25631 | 132098 | 
| 73 | **23 django/db/models/query_utils.py** | 285 | 325| 286 | 25917 | 132098 | 
| 74 | 24 django/db/models/sql/subqueries.py | 1 | 44| 320 | 26237 | 133299 | 
| 75 | 24 django/db/models/sql/query.py | 376 | 426| 494 | 26731 | 133299 | 
| 76 | 24 django/db/models/expressions.py | 445 | 490| 314 | 27045 | 133299 | 
| 77 | 24 django/db/models/query.py | 1487 | 1520| 297 | 27342 | 133299 | 
| 78 | 24 django/db/models/query.py | 236 | 263| 221 | 27563 | 133299 | 
| 79 | 24 django/db/models/constraints.py | 79 | 161| 729 | 28292 | 133299 | 
| 80 | 24 django/db/models/sql/compiler.py | 442 | 495| 564 | 28856 | 133299 | 
| 81 | 24 django/db/models/query.py | 666 | 680| 132 | 28988 | 133299 | 
| 82 | 24 django/db/models/sql/datastructures.py | 117 | 148| 170 | 29158 | 133299 | 
| 83 | 24 django/db/models/query.py | 175 | 234| 469 | 29627 | 133299 | 
| 84 | 24 django/contrib/postgres/search.py | 1 | 24| 205 | 29832 | 133299 | 
| 85 | 24 django/db/models/sql/query.py | 140 | 234| 833 | 30665 | 133299 | 
| 86 | 25 django/db/backends/sqlite3/features.py | 1 | 114| 1027 | 31692 | 134326 | 
| 87 | 25 django/db/models/expressions.py | 1088 | 1147| 442 | 32134 | 134326 | 
| 88 | 25 django/db/models/sql/where.py | 212 | 230| 131 | 32265 | 134326 | 
| 89 | 25 django/db/models/sql/compiler.py | 22 | 47| 257 | 32522 | 134326 | 
| 90 | 25 django/db/models/sql/query.py | 120 | 137| 157 | 32679 | 134326 | 
| 91 | 25 django/db/models/fields/related.py | 1235 | 1352| 963 | 33642 | 134326 | 
| 92 | 26 django/db/models/functions/math.py | 144 | 184| 247 | 33889 | 135590 | 
| 93 | 26 django/db/models/sql/query.py | 1423 | 1443| 212 | 34101 | 135590 | 
| 94 | 26 django/db/models/query.py | 1 | 39| 294 | 34395 | 135590 | 
| 95 | 27 django/db/backends/base/features.py | 217 | 319| 879 | 35274 | 138550 | 
| 96 | 27 django/db/models/query.py | 1992 | 2015| 200 | 35474 | 138550 | 
| 97 | 27 django/db/models/sql/where.py | 195 | 209| 156 | 35630 | 138550 | 
| 98 | 27 django/db/models/sql/query.py | 351 | 374| 179 | 35809 | 138550 | 
| 99 | 27 django/db/models/query.py | 92 | 110| 138 | 35947 | 138550 | 
| 100 | 27 django/db/models/expressions.py | 798 | 814| 153 | 36100 | 138550 | 
| 101 | 27 django/db/models/query.py | 1659 | 1774| 1098 | 37198 | 138550 | 
| 102 | 27 django/db/models/query.py | 1143 | 1179| 324 | 37522 | 138550 | 
| 103 | 27 django/db/models/expressions.py | 817 | 851| 292 | 37814 | 138550 | 
| 104 | 28 django/db/backends/sqlite3/operations.py | 170 | 195| 190 | 38004 | 141635 | 
| 105 | 28 django/db/models/constraints.py | 172 | 196| 188 | 38192 | 141635 | 
| 106 | 28 django/db/backends/sqlite3/operations.py | 40 | 66| 232 | 38424 | 141635 | 
| 107 | 29 django/db/backends/mysql/features.py | 1 | 54| 406 | 38830 | 143607 | 
| 108 | 29 django/db/models/aggregates.py | 70 | 96| 266 | 39096 | 143607 | 
| 109 | 30 django/db/models/fields/__init__.py | 1110 | 1139| 218 | 39314 | 162025 | 
| 110 | 30 django/contrib/postgres/constraints.py | 129 | 155| 231 | 39545 | 162025 | 
| 111 | 30 django/db/models/sql/compiler.py | 63 | 147| 881 | 40426 | 162025 | 
| 112 | 30 django/db/models/sql/subqueries.py | 137 | 163| 161 | 40587 | 162025 | 
| 113 | 30 django/contrib/postgres/constraints.py | 69 | 91| 213 | 40800 | 162025 | 
| 114 | 30 django/db/models/expressions.py | 594 | 633| 290 | 41090 | 162025 | 
| 115 | 31 django/forms/models.py | 1186 | 1251| 543 | 41633 | 173799 | 
| 116 | 32 django/db/backends/mysql/operations.py | 278 | 289| 165 | 41798 | 177498 | 
| 117 | 32 django/db/models/query.py | 1219 | 1247| 183 | 41981 | 177498 | 
| 118 | 32 django/db/models/sql/compiler.py | 765 | 776| 163 | 42144 | 177498 | 
| 119 | 33 django/contrib/gis/db/models/__init__.py | 1 | 19| 204 | 42348 | 177702 | 
| 120 | 34 django/utils/datastructures.py | 214 | 248| 226 | 42574 | 179926 | 
| 121 | 35 django/db/migrations/operations/models.py | 1 | 38| 235 | 42809 | 186903 | 
| 122 | 35 django/db/backends/oracle/operations.py | 617 | 643| 303 | 43112 | 186903 | 
| 123 | 35 django/db/models/base.py | 1960 | 2091| 976 | 44088 | 186903 | 
| 124 | 36 django/db/models/enums.py | 37 | 59| 183 | 44271 | 187504 | 
| 125 | 36 django/db/models/base.py | 1 | 50| 328 | 44599 | 187504 | 
| 126 | 36 django/db/models/sql/query.py | 2035 | 2081| 370 | 44969 | 187504 | 
| 127 | 36 django/db/models/lookups.py | 369 | 401| 294 | 45263 | 187504 | 
| 128 | 36 django/db/models/sql/query.py | 1994 | 2033| 320 | 45583 | 187504 | 
| 129 | 37 django/db/models/fields/json.py | 198 | 216| 232 | 45815 | 191727 | 
| 130 | 37 django/db/models/query.py | 446 | 461| 130 | 45945 | 191727 | 
| 131 | 38 django/db/backends/sqlite3/base.py | 403 | 423| 183 | 46128 | 197712 | 
| 132 | 38 django/db/models/query.py | 970 | 985| 124 | 46252 | 197712 | 
| 133 | 39 django/core/checks/model_checks.py | 178 | 211| 332 | 46584 | 199497 | 


### Hint

```
Reproduced on 3.1.6. The exception is raised by this two lines in the Q._combine, which are not present in the Combinable._combine from which Exists inherit. if not isinstance(other, Q): raise TypeError(other)
Tests: diff --git a/tests/expressions/tests.py b/tests/expressions/tests.py index 08ea0a51d3..20d0404f44 100644 --- a/tests/expressions/tests.py +++ b/tests/expressions/tests.py @@ -815,6 +815,15 @@ class BasicExpressionsTests(TestCase): Employee.objects.filter(Exists(is_poc) | Q(salary__lt=15)), [self.example_inc.ceo, self.max], ) + self.assertCountEqual( + Employee.objects.filter(Q(salary__gte=30) & Exists(is_ceo)), + [self.max], + ) + self.assertCountEqual( + Employee.objects.filter(Q(salary__lt=15) | Exists(is_poc)), + [self.example_inc.ceo, self.max], + ) + class IterableLookupInnerExpressionsTests(TestCase):
​PR
```

## Patch

```diff
diff --git a/django/db/models/query_utils.py b/django/db/models/query_utils.py
--- a/django/db/models/query_utils.py
+++ b/django/db/models/query_utils.py
@@ -40,7 +40,7 @@ def __init__(self, *args, _connector=None, _negated=False, **kwargs):
         super().__init__(children=[*args, *sorted(kwargs.items())], connector=_connector, negated=_negated)
 
     def _combine(self, other, conn):
-        if not isinstance(other, Q):
+        if not(isinstance(other, Q) or getattr(other, 'conditional', False) is True):
             raise TypeError(other)
 
         # If the other Q() is empty, ignore it and just use `self`.

```

## Test Patch

```diff
diff --git a/tests/expressions/tests.py b/tests/expressions/tests.py
--- a/tests/expressions/tests.py
+++ b/tests/expressions/tests.py
@@ -815,6 +815,28 @@ def test_boolean_expression_combined(self):
             Employee.objects.filter(Exists(is_poc) | Q(salary__lt=15)),
             [self.example_inc.ceo, self.max],
         )
+        self.assertCountEqual(
+            Employee.objects.filter(Q(salary__gte=30) & Exists(is_ceo)),
+            [self.max],
+        )
+        self.assertCountEqual(
+            Employee.objects.filter(Q(salary__lt=15) | Exists(is_poc)),
+            [self.example_inc.ceo, self.max],
+        )
+
+    def test_boolean_expression_combined_with_empty_Q(self):
+        is_poc = Company.objects.filter(point_of_contact=OuterRef('pk'))
+        self.gmbh.point_of_contact = self.max
+        self.gmbh.save()
+        tests = [
+            Exists(is_poc) & Q(),
+            Q() & Exists(is_poc),
+            Exists(is_poc) | Q(),
+            Q() | Exists(is_poc),
+        ]
+        for conditions in tests:
+            with self.subTest(conditions):
+                self.assertCountEqual(Employee.objects.filter(conditions), [self.max])
 
 
 class IterableLookupInnerExpressionsTests(TestCase):

```


## Code snippets

### 1 - django/db/models/query_utils.py:

Start line: 61, End line: 81

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

    def resolve_expression(self, query=None, allow_joins=True, reuse=None, summarize=False, for_save=False):
        # We must promote any new joins to left outer joins so that when Q is
        # used as an expression, rows aren't filtered due to joins.
        clause, joins = query._add_q(
            self, reuse, allow_joins=allow_joins, split_subq=False,
            check_filterable=False,
        )
        query.promote_joins(joins)
        return clause
```
### 2 - django/db/models/expressions.py:

Start line: 1150, End line: 1182

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
### 3 - django/db/models/query.py:

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
### 4 - django/db/models/query.py:

Start line: 1009, End line: 1022

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
### 5 - django/contrib/postgres/search.py:

Start line: 130, End line: 157

```python
class SearchQueryCombinable:
    BITAND = '&&'
    BITOR = '||'

    def _combine(self, other, connector, reversed):
        if not isinstance(other, SearchQueryCombinable):
            raise TypeError(
                'SearchQuery can only be combined with other SearchQuery '
                'instances, got %s.' % type(other).__name__
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
### 6 - django/db/models/query.py:

Start line: 987, End line: 1007

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
### 7 - django/db/models/query.py:

Start line: 803, End line: 842

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
### 8 - django/db/models/query_utils.py:

Start line: 28, End line: 59

```python
class Q(tree.Node):
    """
    Encapsulate filters as objects that can then be combined logically (using
    `&` and `|`).
    """
    # Connection types
    AND = 'AND'
    OR = 'OR'
    default = AND
    conditional = True

    def __init__(self, *args, _connector=None, _negated=False, **kwargs):
        super().__init__(children=[*args, *sorted(kwargs.items())], connector=_connector, negated=_negated)

    def _combine(self, other, conn):
        if not isinstance(other, Q):
            raise TypeError(other)

        # If the other Q() is empty, ignore it and just use `self`.
        if not other:
            _, args, kwargs = self.deconstruct()
            return type(self)(*args, **kwargs)
        # Or if this Q is empty, ignore it and just use `other`.
        elif not self:
            _, args, kwargs = other.deconstruct()
            return type(other)(*args, **kwargs)

        obj = type(self)()
        obj.connector = conn
        obj.add(self, conn)
        obj.add(other, conn)
        return obj
```
### 9 - django/db/models/sql/query.py:

Start line: 1401, End line: 1421

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
### 10 - django/db/models/sql/query.py:

Start line: 1378, End line: 1399

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
### 23 - django/db/models/query_utils.py:

Start line: 83, End line: 97

```python
class Q(tree.Node):

    def deconstruct(self):
        path = '%s.%s' % (self.__class__.__module__, self.__class__.__name__)
        if path.startswith('django.db.models.query_utils'):
            path = path.replace('django.db.models.query_utils', 'django.db.models')
        args, kwargs = (), {}
        if len(self.children) == 1 and not isinstance(self.children[0], Q):
            child = self.children[0]
            kwargs = {child[0]: child[1]}
        else:
            args = tuple(self.children)
            if self.connector != self.default:
                kwargs = {'_connector': self.connector}
        if self.negated:
            kwargs['_negated'] = True
        return path, args, kwargs
```
### 41 - django/db/models/query_utils.py:

Start line: 257, End line: 282

```python
def check_rel_lookup_compatibility(model, target_opts, field):
    """
    Check that self.model is compatible with target_opts. Compatibility
    is OK if:
      1) model and opts match (where proxy inheritance is removed)
      2) model is parent of opts' model or the other way around
    """
    def check(opts):
        return (
            model._meta.concrete_model == opts.concrete_model or
            opts.concrete_model in model._meta.get_parent_list() or
            model in opts.get_parent_list()
        )
    # If the field is a primary key, then doing a query against the field's
    # model is ok, too. Consider the case:
    # class Restaurant(models.Model):
    #     place = OneToOneField(Place, primary_key=True):
    # Restaurant.objects.filter(pk__in=Restaurant.objects.all()).
    # If we didn't have the primary key check, then pk__in (== place__in) would
    # give Place's opts as the target opts, but Restaurant isn't compatible
    # with that. This logic applies only to primary keys, as when doing __in=qs,
    # we are going to turn this into __in=qs.values('pk') later on.
    return (
        check(target_opts) or
        (getattr(field, 'primary_key', False) and check(field.model._meta))
    )
```
### 73 - django/db/models/query_utils.py:

Start line: 285, End line: 325

```python
class FilteredRelation:
    """Specify custom filtering in the ON clause of SQL joins."""

    def __init__(self, relation_name, *, condition=Q()):
        if not relation_name:
            raise ValueError('relation_name cannot be empty.')
        self.relation_name = relation_name
        self.alias = None
        if not isinstance(condition, Q):
            raise ValueError('condition argument must be a Q() instance.')
        self.condition = condition
        self.path = []

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return (
            self.relation_name == other.relation_name and
            self.alias == other.alias and
            self.condition == other.condition
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
        raise NotImplementedError('FilteredRelation.resolve_expression() is unused.')

    def as_sql(self, compiler, connection):
        # Resolve the condition in Join.filtered_relation.
        query = compiler.query
        where = query.build_filtered_relation_q(self.condition, reuse=set(self.path))
        return compiler.compile(where)
```
