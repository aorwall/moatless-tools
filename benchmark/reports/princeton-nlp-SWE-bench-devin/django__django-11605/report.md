# django__django-11605

| **django/django** | `194d1dfc186cc8d2b35dabf64f3ed38b757cbd98` |
| ---- | ---- |
| **No of patches** | 2 |
| **All found context length** | 6595 |
| **Any found context length** | 6595 |
| **Avg pos** | 12.5 |
| **Min pos** | 25 |
| **Max pos** | 25 |
| **Top file pos** | 1 |
| **Missing snippets** | 4 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/db/models/expressions.py b/django/db/models/expressions.py
--- a/django/db/models/expressions.py
+++ b/django/db/models/expressions.py
@@ -500,8 +500,6 @@ def as_sql(self, compiler, connection):
 @deconstructible
 class F(Combinable):
     """An object capable of resolving references to existing query objects."""
-    # Can the expression be used in a WHERE clause?
-    filterable = True
 
     def __init__(self, name):
         """
diff --git a/django/db/models/sql/query.py b/django/db/models/sql/query.py
--- a/django/db/models/sql/query.py
+++ b/django/db/models/sql/query.py
@@ -1114,6 +1114,17 @@ def check_related_objects(self, field, value, opts):
                 for v in value:
                     self.check_query_object_type(v, opts, field)
 
+    def check_filterable(self, expression):
+        """Raise an error if expression cannot be used in a WHERE clause."""
+        if not getattr(expression, 'filterable', 'True'):
+            raise NotSupportedError(
+                expression.__class__.__name__ + ' is disallowed in the filter '
+                'clause.'
+            )
+        if hasattr(expression, 'get_source_expressions'):
+            for expr in expression.get_source_expressions():
+                self.check_filterable(expr)
+
     def build_lookup(self, lookups, lhs, rhs):
         """
         Try to extract transforms and lookup from given lhs.
@@ -1217,11 +1228,7 @@ def build_filter(self, filter_expr, branch_negated=False, current_negated=False,
             raise FieldError("Cannot parse keyword query %r" % arg)
         lookups, parts, reffed_expression = self.solve_lookup_type(arg)
 
-        if not getattr(reffed_expression, 'filterable', True):
-            raise NotSupportedError(
-                reffed_expression.__class__.__name__ + ' is disallowed in '
-                'the filter clause.'
-            )
+        self.check_filterable(reffed_expression)
 
         if not allow_joins and len(parts) > 1:
             raise FieldError("Joined field references are not permitted in this query")
@@ -1230,6 +1237,8 @@ def build_filter(self, filter_expr, branch_negated=False, current_negated=False,
         value = self.resolve_lookup_value(value, can_reuse, allow_joins, simple_col)
         used_joins = {k for k, v in self.alias_refcount.items() if v > pre_joins.get(k, 0)}
 
+        self.check_filterable(value)
+
         clause = self.where_class()
         if reffed_expression:
             condition = self.build_lookup(lookups, reffed_expression, value)

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/db/models/expressions.py | 503 | 504 | 25 | 1 | 6595
| django/db/models/sql/query.py | 1117 | 1117 | - | 4 | -
| django/db/models/sql/query.py | 1220 | 1224 | - | 4 | -
| django/db/models/sql/query.py | 1233 | 1233 | - | 4 | -


## Problem Statement

```
Filter by window expression should raise a descriptive error.
Description
	
Django has a check that filter does not contain window expressions. 
But it is shallow, neither right side of the expression nor combined expressions are checked.
class Employee(models.Model):
	grade = models.IntegerField()
# raises NotSupportedError
Employee.objects.annotate(
	prev_grade=Window(expression=Lag('grade'))
).filter(prev_grade=F('grade'))
# Do not raise anything, fail on database backend once executed.
Employee.objects.annotate(
	prev_grade=Window(expression=Lag('grade'))
).filter(grade=F('prev_grade'))
Employee.objects.annotate(
	prev_grade=Window(expression=Lag('grade')),
	dec_grade=F('prev_grade') - Value(1)
).filter(dec_grade=F('grade'))

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 django/db/models/expressions.py** | 1203 | 1235| 246 | 246 | 10035 | 
| 2 | **1 django/db/models/expressions.py** | 1158 | 1201| 373 | 619 | 10035 | 
| 3 | **1 django/db/models/expressions.py** | 1287 | 1323| 305 | 924 | 10035 | 
| 4 | **1 django/db/models/expressions.py** | 1237 | 1249| 111 | 1035 | 10035 | 
| 5 | **1 django/db/models/expressions.py** | 1252 | 1285| 276 | 1311 | 10035 | 
| 6 | 2 django/db/models/functions/window.py | 52 | 79| 182 | 1493 | 10678 | 
| 7 | 2 django/db/models/functions/window.py | 28 | 49| 154 | 1647 | 10678 | 
| 8 | 2 django/db/models/functions/window.py | 1 | 25| 153 | 1800 | 10678 | 
| 9 | 3 django/db/backends/sqlite3/operations.py | 42 | 64| 220 | 2020 | 13562 | 
| 10 | **4 django/db/models/sql/query.py** | 1264 | 1301| 512 | 2532 | 34871 | 
| 11 | 4 django/db/models/functions/window.py | 82 | 109| 150 | 2682 | 34871 | 
| 12 | 5 django/template/base.py | 667 | 702| 272 | 2954 | 42732 | 
| 13 | 6 django/db/models/__init__.py | 1 | 49| 548 | 3502 | 43280 | 
| 14 | 6 django/template/base.py | 704 | 723| 177 | 3679 | 43280 | 
| 15 | **6 django/db/models/expressions.py** | 869 | 925| 532 | 4211 | 43280 | 
| 16 | **6 django/db/models/expressions.py** | 143 | 182| 263 | 4474 | 43280 | 
| 17 | **6 django/db/models/expressions.py** | 651 | 675| 238 | 4712 | 43280 | 
| 18 | **6 django/db/models/expressions.py** | 1082 | 1103| 183 | 4895 | 43280 | 
| 19 | **6 django/db/models/expressions.py** | 300 | 319| 184 | 5079 | 43280 | 
| 20 | **6 django/db/models/expressions.py** | 420 | 455| 385 | 5464 | 43280 | 
| 21 | **6 django/db/models/sql/query.py** | 1376 | 1387| 137 | 5601 | 43280 | 
| 22 | **6 django/db/models/expressions.py** | 1 | 28| 191 | 5792 | 43280 | 
| 23 | **6 django/db/models/expressions.py** | 1122 | 1155| 292 | 6084 | 43280 | 
| 24 | **6 django/db/models/expressions.py** | 212 | 246| 285 | 6369 | 43280 | 
| **-> 25 <-** | **6 django/db/models/expressions.py** | 500 | 530| 226 | 6595 | 43280 | 
| 26 | **6 django/db/models/expressions.py** | 1105 | 1120| 148 | 6743 | 43280 | 
| 27 | **6 django/db/models/sql/query.py** | 1623 | 1658| 409 | 7152 | 43280 | 
| 28 | 7 django/db/models/aggregates.py | 45 | 68| 294 | 7446 | 44581 | 
| 29 | 8 django/db/backends/base/operations.py | 557 | 652| 804 | 8250 | 49976 | 
| 30 | **8 django/db/models/expressions.py** | 273 | 298| 257 | 8507 | 49976 | 
| 31 | **8 django/db/models/expressions.py** | 702 | 728| 196 | 8703 | 49976 | 
| 32 | 9 django/contrib/admin/options.py | 368 | 420| 504 | 9207 | 68342 | 
| 33 | **9 django/db/models/expressions.py** | 731 | 759| 230 | 9437 | 68342 | 
| 34 | **9 django/db/models/expressions.py** | 369 | 392| 180 | 9617 | 68342 | 
| 35 | **9 django/db/models/expressions.py** | 458 | 497| 305 | 9922 | 68342 | 
| 36 | **9 django/db/models/expressions.py** | 976 | 1001| 242 | 10164 | 68342 | 
| 37 | 9 django/db/models/aggregates.py | 122 | 158| 245 | 10409 | 68342 | 
| 38 | **9 django/db/models/expressions.py** | 321 | 367| 298 | 10707 | 68342 | 
| 39 | **9 django/db/models/sql/query.py** | 1097 | 1115| 232 | 10939 | 68342 | 
| 40 | 10 django/views/debug.py | 179 | 191| 146 | 11085 | 72560 | 
| 41 | **10 django/db/models/expressions.py** | 1051 | 1079| 286 | 11371 | 72560 | 
| 42 | 11 django/db/models/fields/__init__.py | 1123 | 1152| 218 | 11589 | 89631 | 
| 43 | 11 django/views/debug.py | 125 | 152| 248 | 11837 | 89631 | 
| 44 | 12 django/contrib/admin/checks.py | 929 | 958| 243 | 12080 | 98647 | 
| 45 | 13 django/core/checks/model_checks.py | 117 | 141| 268 | 12348 | 100328 | 
| 46 | 13 django/contrib/admin/checks.py | 791 | 841| 443 | 12791 | 100328 | 
| 47 | **13 django/db/models/sql/query.py** | 2196 | 2212| 177 | 12968 | 100328 | 
| 48 | 14 django/db/backends/base/features.py | 216 | 297| 633 | 13601 | 102820 | 
| 49 | 15 django/contrib/postgres/constraints.py | 1 | 54| 450 | 14051 | 103677 | 
| 50 | **15 django/db/models/sql/query.py** | 1042 | 1066| 243 | 14294 | 103677 | 
| 51 | **15 django/db/models/expressions.py** | 395 | 418| 169 | 14463 | 103677 | 
| 52 | **15 django/db/models/expressions.py** | 248 | 271| 165 | 14628 | 103677 | 
| 53 | **15 django/db/models/expressions.py** | 928 | 974| 377 | 15005 | 103677 | 
| 54 | 16 django/contrib/postgres/search.py | 147 | 154| 121 | 15126 | 105615 | 
| 55 | 16 django/db/backends/base/features.py | 116 | 215| 848 | 15974 | 105615 | 
| 56 | 17 django/core/exceptions.py | 99 | 194| 649 | 16623 | 106670 | 
| 57 | **17 django/db/models/sql/query.py** | 1760 | 1809| 330 | 16953 | 106670 | 
| 58 | 17 django/db/backends/sqlite3/operations.py | 277 | 291| 148 | 17101 | 106670 | 
| 59 | **17 django/db/models/expressions.py** | 678 | 700| 174 | 17275 | 106670 | 
| 60 | 17 django/contrib/admin/checks.py | 371 | 397| 281 | 17556 | 106670 | 
| 61 | 17 django/contrib/postgres/constraints.py | 56 | 66| 123 | 17679 | 106670 | 
| 62 | **17 django/db/models/expressions.py** | 1004 | 1048| 310 | 17989 | 106670 | 
| 63 | 18 django/db/models/base.py | 1094 | 1121| 286 | 18275 | 121760 | 
| 64 | 18 django/contrib/admin/checks.py | 399 | 411| 137 | 18412 | 121760 | 
| 65 | **18 django/db/models/sql/query.py** | 1405 | 1483| 734 | 19146 | 121760 | 
| 66 | 18 django/core/checks/model_checks.py | 166 | 199| 332 | 19478 | 121760 | 
| 67 | **18 django/db/models/sql/query.py** | 882 | 904| 248 | 19726 | 121760 | 
| 68 | 19 django/contrib/postgres/aggregates/mixins.py | 1 | 20| 181 | 19907 | 122270 | 
| 69 | 19 django/db/models/aggregates.py | 99 | 119| 158 | 20065 | 122270 | 
| 70 | 19 django/contrib/admin/checks.py | 1087 | 1117| 188 | 20253 | 122270 | 
| 71 | **19 django/db/models/expressions.py** | 606 | 631| 268 | 20521 | 122270 | 
| 72 | 19 django/db/models/base.py | 1123 | 1138| 138 | 20659 | 122270 | 
| 73 | 19 django/contrib/admin/checks.py | 1013 | 1040| 204 | 20863 | 122270 | 
| 74 | 19 django/core/checks/model_checks.py | 143 | 164| 263 | 21126 | 122270 | 
| 75 | 19 django/db/models/fields/__init__.py | 1313 | 1354| 332 | 21458 | 122270 | 
| 76 | 20 django/contrib/admin/filters.py | 365 | 389| 294 | 21752 | 125939 | 
| 77 | 20 django/db/models/base.py | 1231 | 1260| 242 | 21994 | 125939 | 
| 78 | **20 django/db/models/sql/query.py** | 689 | 724| 389 | 22383 | 125939 | 
| 79 | **20 django/db/models/sql/query.py** | 1303 | 1324| 259 | 22642 | 125939 | 
| 80 | 20 django/db/models/fields/__init__.py | 1173 | 1211| 293 | 22935 | 125939 | 
| 81 | 21 django/core/checks/messages.py | 53 | 76| 161 | 23096 | 126512 | 
| 82 | 22 django/db/models/sql/compiler.py | 138 | 182| 490 | 23586 | 140217 | 
| 83 | 22 django/db/models/sql/compiler.py | 1 | 20| 167 | 23753 | 140217 | 
| 84 | **22 django/db/models/sql/query.py** | 358 | 408| 494 | 24247 | 140217 | 
| 85 | 23 django/db/backends/mysql/operations.py | 234 | 244| 151 | 24398 | 143344 | 
| 86 | 24 django/contrib/admin/views/main.py | 1 | 34| 244 | 24642 | 147421 | 
| 87 | 25 django/db/models/query_utils.py | 298 | 337| 281 | 24923 | 150042 | 
| 88 | 25 django/contrib/admin/checks.py | 768 | 789| 190 | 25113 | 150042 | 
| 89 | 26 django/db/models/lookups.py | 510 | 536| 163 | 25276 | 154201 | 
| 90 | 26 django/db/models/base.py | 1289 | 1318| 205 | 25481 | 154201 | 
| 91 | 26 django/contrib/admin/checks.py | 879 | 927| 416 | 25897 | 154201 | 
| 92 | 27 django/core/validators.py | 341 | 386| 308 | 26205 | 158518 | 
| 93 | **27 django/db/models/sql/query.py** | 1731 | 1758| 244 | 26449 | 158518 | 
| 94 | 27 django/core/checks/messages.py | 1 | 24| 156 | 26605 | 158518 | 
| 95 | 27 django/contrib/admin/checks.py | 353 | 369| 134 | 26739 | 158518 | 
| 96 | 28 django/contrib/postgres/aggregates/statistics.py | 1 | 19| 206 | 26945 | 158985 | 
| 97 | 28 django/contrib/admin/filters.py | 1 | 17| 127 | 27072 | 158985 | 
| 98 | 28 django/contrib/admin/filters.py | 62 | 115| 411 | 27483 | 158985 | 
| 99 | 29 django/shortcuts.py | 81 | 99| 200 | 27683 | 160083 | 
| 100 | 30 django/db/backends/mysql/validation.py | 1 | 27| 248 | 27931 | 160571 | 
| 101 | 31 django/contrib/postgres/forms/ranges.py | 66 | 110| 284 | 28215 | 161275 | 
| 102 | 32 django/forms/models.py | 752 | 773| 194 | 28409 | 172770 | 
| 103 | 32 django/contrib/admin/checks.py | 706 | 716| 115 | 28524 | 172770 | 
| 104 | 33 django/db/models/sql/where.py | 228 | 244| 130 | 28654 | 174545 | 
| 105 | **33 django/db/models/expressions.py** | 829 | 843| 123 | 28777 | 174545 | 
| 106 | 33 django/db/models/aggregates.py | 1 | 43| 344 | 29121 | 174545 | 
| 107 | 33 django/contrib/admin/filters.py | 162 | 201| 381 | 29502 | 174545 | 
| 108 | 33 django/forms/models.py | 309 | 348| 387 | 29889 | 174545 | 
| 109 | 33 django/db/models/fields/__init__.py | 2129 | 2170| 325 | 30214 | 174545 | 
| 110 | 33 django/db/models/lookups.py | 261 | 311| 306 | 30520 | 174545 | 
| 111 | 33 django/db/models/base.py | 1736 | 1807| 565 | 31085 | 174545 | 
| 112 | 33 django/db/models/base.py | 1140 | 1168| 213 | 31298 | 174545 | 
| 113 | 33 django/contrib/postgres/search.py | 191 | 240| 338 | 31636 | 174545 | 
| 114 | 34 django/forms/utils.py | 139 | 146| 132 | 31768 | 175781 | 
| 115 | 34 django/core/validators.py | 29 | 71| 330 | 32098 | 175781 | 
| 116 | 34 django/contrib/admin/filters.py | 20 | 59| 295 | 32393 | 175781 | 
| 117 | 34 django/contrib/postgres/search.py | 1 | 21| 201 | 32594 | 175781 | 
| 118 | 35 django/db/models/fields/related.py | 964 | 991| 215 | 32809 | 189293 | 
| 119 | 35 django/template/base.py | 571 | 606| 359 | 33168 | 189293 | 
| 120 | **35 django/db/models/sql/query.py** | 1561 | 1585| 227 | 33395 | 189293 | 
| 121 | 36 django/db/models/functions/comparison.py | 1 | 29| 317 | 33712 | 190371 | 
| 122 | 36 django/core/checks/messages.py | 26 | 50| 259 | 33971 | 190371 | 
| 123 | 37 django/views/generic/__init__.py | 1 | 23| 189 | 34160 | 190561 | 
| 124 | 37 django/db/models/base.py | 1369 | 1424| 491 | 34651 | 190561 | 
| 125 | 37 django/contrib/admin/filters.py | 391 | 423| 299 | 34950 | 190561 | 
| 126 | 37 django/contrib/admin/views/main.py | 104 | 186| 818 | 35768 | 190561 | 
| 127 | 37 django/db/models/base.py | 1049 | 1092| 404 | 36172 | 190561 | 
| 128 | 38 django/db/models/constraints.py | 30 | 66| 309 | 36481 | 191560 | 
| 129 | 38 django/core/exceptions.py | 1 | 96| 405 | 36886 | 191560 | 
| 130 | 39 django/db/backends/mysql/features.py | 1 | 102| 842 | 37728 | 192653 | 
| 131 | 39 django/contrib/postgres/constraints.py | 68 | 107| 298 | 38026 | 192653 | 
| 132 | 39 django/db/models/lookups.py | 55 | 80| 218 | 38244 | 192653 | 
| 133 | 39 django/db/models/aggregates.py | 70 | 96| 266 | 38510 | 192653 | 
| 134 | 39 django/contrib/admin/filters.py | 223 | 238| 196 | 38706 | 192653 | 
| 135 | 39 django/db/models/fields/related.py | 127 | 154| 202 | 38908 | 192653 | 
| 136 | 39 django/db/models/functions/comparison.py | 84 | 113| 244 | 39152 | 192653 | 
| 137 | 39 django/db/models/base.py | 1809 | 1830| 155 | 39307 | 192653 | 
| 138 | 39 django/contrib/admin/filters.py | 299 | 362| 627 | 39934 | 192653 | 
| 139 | 39 django/db/backends/base/features.py | 1 | 115| 900 | 40834 | 192653 | 
| 140 | 39 django/contrib/admin/checks.py | 843 | 865| 217 | 41051 | 192653 | 
| 141 | 39 django/db/models/base.py | 1451 | 1473| 171 | 41222 | 192653 | 
| 142 | 39 django/db/models/base.py | 1592 | 1640| 348 | 41570 | 192653 | 


### Hint

```
Thanks for this report. Agreed, we should raise a helpful message because it is currently unsupported (see also a ticket #28333 to support this feature). ​PR
```

## Patch

```diff
diff --git a/django/db/models/expressions.py b/django/db/models/expressions.py
--- a/django/db/models/expressions.py
+++ b/django/db/models/expressions.py
@@ -500,8 +500,6 @@ def as_sql(self, compiler, connection):
 @deconstructible
 class F(Combinable):
     """An object capable of resolving references to existing query objects."""
-    # Can the expression be used in a WHERE clause?
-    filterable = True
 
     def __init__(self, name):
         """
diff --git a/django/db/models/sql/query.py b/django/db/models/sql/query.py
--- a/django/db/models/sql/query.py
+++ b/django/db/models/sql/query.py
@@ -1114,6 +1114,17 @@ def check_related_objects(self, field, value, opts):
                 for v in value:
                     self.check_query_object_type(v, opts, field)
 
+    def check_filterable(self, expression):
+        """Raise an error if expression cannot be used in a WHERE clause."""
+        if not getattr(expression, 'filterable', 'True'):
+            raise NotSupportedError(
+                expression.__class__.__name__ + ' is disallowed in the filter '
+                'clause.'
+            )
+        if hasattr(expression, 'get_source_expressions'):
+            for expr in expression.get_source_expressions():
+                self.check_filterable(expr)
+
     def build_lookup(self, lookups, lhs, rhs):
         """
         Try to extract transforms and lookup from given lhs.
@@ -1217,11 +1228,7 @@ def build_filter(self, filter_expr, branch_negated=False, current_negated=False,
             raise FieldError("Cannot parse keyword query %r" % arg)
         lookups, parts, reffed_expression = self.solve_lookup_type(arg)
 
-        if not getattr(reffed_expression, 'filterable', True):
-            raise NotSupportedError(
-                reffed_expression.__class__.__name__ + ' is disallowed in '
-                'the filter clause.'
-            )
+        self.check_filterable(reffed_expression)
 
         if not allow_joins and len(parts) > 1:
             raise FieldError("Joined field references are not permitted in this query")
@@ -1230,6 +1237,8 @@ def build_filter(self, filter_expr, branch_negated=False, current_negated=False,
         value = self.resolve_lookup_value(value, can_reuse, allow_joins, simple_col)
         used_joins = {k for k, v in self.alias_refcount.items() if v > pre_joins.get(k, 0)}
 
+        self.check_filterable(value)
+
         clause = self.where_class()
         if reffed_expression:
             condition = self.build_lookup(lookups, reffed_expression, value)

```

## Test Patch

```diff
diff --git a/tests/expressions_window/tests.py b/tests/expressions_window/tests.py
--- a/tests/expressions_window/tests.py
+++ b/tests/expressions_window/tests.py
@@ -4,7 +4,8 @@
 from django.core.exceptions import FieldError
 from django.db import NotSupportedError, connection
 from django.db.models import (
-    F, OuterRef, RowRange, Subquery, Value, ValueRange, Window, WindowFrame,
+    F, Func, OuterRef, Q, RowRange, Subquery, Value, ValueRange, Window,
+    WindowFrame,
 )
 from django.db.models.aggregates import Avg, Max, Min, Sum
 from django.db.models.functions import (
@@ -833,8 +834,17 @@ def test_frame_window_frame_notimplemented(self):
 
     def test_invalid_filter(self):
         msg = 'Window is disallowed in the filter clause'
+        qs = Employee.objects.annotate(dense_rank=Window(expression=DenseRank()))
         with self.assertRaisesMessage(NotSupportedError, msg):
-            Employee.objects.annotate(dense_rank=Window(expression=DenseRank())).filter(dense_rank__gte=1)
+            qs.filter(dense_rank__gte=1)
+        with self.assertRaisesMessage(NotSupportedError, msg):
+            qs.annotate(inc_rank=F('dense_rank') + Value(1)).filter(inc_rank__gte=1)
+        with self.assertRaisesMessage(NotSupportedError, msg):
+            qs.filter(id=F('dense_rank'))
+        with self.assertRaisesMessage(NotSupportedError, msg):
+            qs.filter(id=Func('dense_rank', 2, function='div'))
+        with self.assertRaisesMessage(NotSupportedError, msg):
+            qs.annotate(total=Sum('dense_rank', filter=Q(name='Jones'))).filter(total=1)
 
     def test_invalid_order_by(self):
         msg = 'order_by must be either an Expression or a sequence of expressions'

```


## Code snippets

### 1 - django/db/models/expressions.py:

Start line: 1203, End line: 1235

```python
class Window(Expression):

    def as_sql(self, compiler, connection, template=None):
        connection.ops.check_expression_support(self)
        if not connection.features.supports_over_clause:
            raise NotSupportedError('This backend does not support window expressions.')
        expr_sql, params = compiler.compile(self.source_expression)
        window_sql, window_params = [], []

        if self.partition_by is not None:
            sql_expr, sql_params = self.partition_by.as_sql(
                compiler=compiler, connection=connection,
                template='PARTITION BY %(expressions)s',
            )
            window_sql.extend(sql_expr)
            window_params.extend(sql_params)

        if self.order_by is not None:
            window_sql.append(' ORDER BY ')
            order_sql, order_params = compiler.compile(self.order_by)
            window_sql.extend(order_sql)
            window_params.extend(order_params)

        if self.frame:
            frame_sql, frame_params = compiler.compile(self.frame)
            window_sql.append(' ' + frame_sql)
            window_params.extend(frame_params)

        params.extend(window_params)
        template = template or self.template

        return template % {
            'expression': expr_sql,
            'window': ''.join(window_sql).strip()
        }, params
```
### 2 - django/db/models/expressions.py:

Start line: 1158, End line: 1201

```python
class Window(Expression):
    template = '%(expression)s OVER (%(window)s)'
    # Although the main expression may either be an aggregate or an
    # expression with an aggregate function, the GROUP BY that will
    # be introduced in the query as a result is not desired.
    contains_aggregate = False
    contains_over_clause = True
    filterable = False

    def __init__(self, expression, partition_by=None, order_by=None, frame=None, output_field=None):
        self.partition_by = partition_by
        self.order_by = order_by
        self.frame = frame

        if not getattr(expression, 'window_compatible', False):
            raise ValueError(
                "Expression '%s' isn't compatible with OVER clauses." %
                expression.__class__.__name__
            )

        if self.partition_by is not None:
            if not isinstance(self.partition_by, (tuple, list)):
                self.partition_by = (self.partition_by,)
            self.partition_by = ExpressionList(*self.partition_by)

        if self.order_by is not None:
            if isinstance(self.order_by, (list, tuple)):
                self.order_by = ExpressionList(*self.order_by)
            elif not isinstance(self.order_by, BaseExpression):
                raise ValueError(
                    'order_by must be either an Expression or a sequence of '
                    'expressions.'
                )
        super().__init__(output_field=output_field)
        self.source_expression = self._parse_expressions(expression)[0]

    def _resolve_output_field(self):
        return self.source_expression.output_field

    def get_source_expressions(self):
        return [self.source_expression, self.partition_by, self.order_by, self.frame]

    def set_source_expressions(self, exprs):
        self.source_expression, self.partition_by, self.order_by, self.frame = exprs
```
### 3 - django/db/models/expressions.py:

Start line: 1287, End line: 1323

```python
class WindowFrame(Expression):

    def __str__(self):
        if self.start.value is not None and self.start.value < 0:
            start = '%d %s' % (abs(self.start.value), connection.ops.PRECEDING)
        elif self.start.value is not None and self.start.value == 0:
            start = connection.ops.CURRENT_ROW
        else:
            start = connection.ops.UNBOUNDED_PRECEDING

        if self.end.value is not None and self.end.value > 0:
            end = '%d %s' % (self.end.value, connection.ops.FOLLOWING)
        elif self.end.value is not None and self.end.value == 0:
            end = connection.ops.CURRENT_ROW
        else:
            end = connection.ops.UNBOUNDED_FOLLOWING
        return self.template % {
            'frame_type': self.frame_type,
            'start': start,
            'end': end,
        }

    def window_frame_start_end(self, connection, start, end):
        raise NotImplementedError('Subclasses must implement window_frame_start_end().')


class RowRange(WindowFrame):
    frame_type = 'ROWS'

    def window_frame_start_end(self, connection, start, end):
        return connection.ops.window_frame_rows_start_end(start, end)


class ValueRange(WindowFrame):
    frame_type = 'RANGE'

    def window_frame_start_end(self, connection, start, end):
        return connection.ops.window_frame_range_start_end(start, end)
```
### 4 - django/db/models/expressions.py:

Start line: 1237, End line: 1249

```python
class Window(Expression):

    def __str__(self):
        return '{} OVER ({}{}{})'.format(
            str(self.source_expression),
            'PARTITION BY ' + str(self.partition_by) if self.partition_by else '',
            'ORDER BY ' + str(self.order_by) if self.order_by else '',
            str(self.frame or ''),
        )

    def __repr__(self):
        return '<%s: %s>' % (self.__class__.__name__, self)

    def get_group_by_cols(self, alias=None):
        return []
```
### 5 - django/db/models/expressions.py:

Start line: 1252, End line: 1285

```python
class WindowFrame(Expression):
    """
    Model the frame clause in window expressions. There are two types of frame
    clauses which are subclasses, however, all processing and validation (by no
    means intended to be complete) is done here. Thus, providing an end for a
    frame is optional (the default is UNBOUNDED FOLLOWING, which is the last
    row in the frame).
    """
    template = '%(frame_type)s BETWEEN %(start)s AND %(end)s'

    def __init__(self, start=None, end=None):
        self.start = Value(start)
        self.end = Value(end)

    def set_source_expressions(self, exprs):
        self.start, self.end = exprs

    def get_source_expressions(self):
        return [self.start, self.end]

    def as_sql(self, compiler, connection):
        connection.ops.check_expression_support(self)
        start, end = self.window_frame_start_end(connection, self.start.value, self.end.value)
        return self.template % {
            'frame_type': self.frame_type,
            'start': start,
            'end': end,
        }, []

    def __repr__(self):
        return '<%s: %s>' % (self.__class__.__name__, self)

    def get_group_by_cols(self, alias=None):
        return []
```
### 6 - django/db/models/functions/window.py:

Start line: 52, End line: 79

```python
class Lag(LagLeadFunction):
    function = 'LAG'


class LastValue(Func):
    arity = 1
    function = 'LAST_VALUE'
    window_compatible = True


class Lead(LagLeadFunction):
    function = 'LEAD'


class NthValue(Func):
    function = 'NTH_VALUE'
    window_compatible = True

    def __init__(self, expression, nth=1, **extra):
        if expression is None:
            raise ValueError('%s requires a non-null source expression.' % self.__class__.__name__)
        if nth is None or nth <= 0:
            raise ValueError('%s requires a positive integer as for nth.' % self.__class__.__name__)
        super().__init__(expression, nth, **extra)

    def _resolve_output_field(self):
        sources = self.get_source_expressions()
        return sources[0].output_field
```
### 7 - django/db/models/functions/window.py:

Start line: 28, End line: 49

```python
class LagLeadFunction(Func):
    window_compatible = True

    def __init__(self, expression, offset=1, default=None, **extra):
        if expression is None:
            raise ValueError(
                '%s requires a non-null source expression.' %
                self.__class__.__name__
            )
        if offset is None or offset <= 0:
            raise ValueError(
                '%s requires a positive integer for the offset.' %
                self.__class__.__name__
            )
        args = (expression, offset)
        if default is not None:
            args += (default,)
        super().__init__(*args, **extra)

    def _resolve_output_field(self):
        sources = self.get_source_expressions()
        return sources[0].output_field
```
### 8 - django/db/models/functions/window.py:

Start line: 1, End line: 25

```python
from django.db.models.expressions import Func
from django.db.models.fields import FloatField, IntegerField

__all__ = [
    'CumeDist', 'DenseRank', 'FirstValue', 'Lag', 'LastValue', 'Lead',
    'NthValue', 'Ntile', 'PercentRank', 'Rank', 'RowNumber',
]


class CumeDist(Func):
    function = 'CUME_DIST'
    output_field = FloatField()
    window_compatible = True


class DenseRank(Func):
    function = 'DENSE_RANK'
    output_field = IntegerField()
    window_compatible = True


class FirstValue(Func):
    arity = 1
    function = 'FIRST_VALUE'
    window_compatible = True
```
### 9 - django/db/backends/sqlite3/operations.py:

Start line: 42, End line: 64

```python
class DatabaseOperations(BaseDatabaseOperations):

    def check_expression_support(self, expression):
        bad_fields = (fields.DateField, fields.DateTimeField, fields.TimeField)
        bad_aggregates = (aggregates.Sum, aggregates.Avg, aggregates.Variance, aggregates.StdDev)
        if isinstance(expression, bad_aggregates):
            for expr in expression.get_source_expressions():
                try:
                    output_field = expr.output_field
                except FieldError:
                    # Not every subexpression has an output_field which is fine
                    # to ignore.
                    pass
                else:
                    if isinstance(output_field, bad_fields):
                        raise utils.NotSupportedError(
                            'You cannot use Sum, Avg, StdDev, and Variance '
                            'aggregations on date/time fields in sqlite3 '
                            'since date/time is saved as text.'
                        )
        if isinstance(expression, aggregates.Aggregate) and len(expression.source_expressions) > 1:
            raise utils.NotSupportedError(
                "SQLite doesn't support DISTINCT on aggregate functions "
                "accepting multiple arguments."
            )
```
### 10 - django/db/models/sql/query.py:

Start line: 1264, End line: 1301

```python
class Query(BaseExpression):

    def build_filter(self, filter_expr, branch_negated=False, current_negated=False,
                     can_reuse=None, allow_joins=True, split_subq=True,
                     reuse_with_filtered_relation=False, simple_col=False):
        # ... other code
        if can_reuse is not None:
            can_reuse.update(join_list)

        if join_info.final_field.is_relation:
            # No support for transforms for relational fields
            num_lookups = len(lookups)
            if num_lookups > 1:
                raise FieldError('Related Field got invalid lookup: {}'.format(lookups[0]))
            if len(targets) == 1:
                col = _get_col(targets[0], join_info.final_field, alias, simple_col)
            else:
                col = MultiColSource(alias, targets, join_info.targets, join_info.final_field)
        else:
            col = _get_col(targets[0], join_info.final_field, alias, simple_col)

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
                col = _get_col(targets[0], join_info.targets[0], alias, simple_col)
                clause.add(lookup_class(col, False), AND)
        return clause, used_joins if not require_outer else ()
```
### 15 - django/db/models/expressions.py:

Start line: 869, End line: 925

```python
class When(Expression):
    template = 'WHEN %(condition)s THEN %(result)s'

    def __init__(self, condition=None, then=None, **lookups):
        if lookups and condition is None:
            condition, lookups = Q(**lookups), None
        if condition is None or not getattr(condition, 'conditional', False) or lookups:
            raise TypeError("__init__() takes either a Q object or lookups as keyword arguments")
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
### 16 - django/db/models/expressions.py:

Start line: 143, End line: 182

```python
@deconstructible
class BaseExpression:
    """Base class for all query expressions."""

    # aggregate specific fields
    is_summary = False
    _output_field_resolved_to_none = False
    # Can the expression be used in a WHERE clause?
    filterable = True
    # Can the expression can be used as a source expression in Window?
    window_compatible = False

    def __init__(self, output_field=None):
        if output_field is not None:
            self.output_field = output_field

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('convert_value', None)
        return state

    def get_db_converters(self, connection):
        return (
            []
            if self.convert_value is self._convert_value_noop else
            [self.convert_value]
        ) + self.output_field.get_db_converters(connection)

    def get_source_expressions(self):
        return []

    def set_source_expressions(self, exprs):
        assert not exprs

    def _parse_expressions(self, *expressions):
        return [
            arg if hasattr(arg, 'resolve_expression') else (
                F(arg) if isinstance(arg, str) else Value(arg)
            ) for arg in expressions
        ]
```
### 17 - django/db/models/expressions.py:

Start line: 651, End line: 675

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
### 18 - django/db/models/expressions.py:

Start line: 1082, End line: 1103

```python
class OrderBy(BaseExpression):
    template = '%(expression)s %(ordering)s'

    def __init__(self, expression, descending=False, nulls_first=False, nulls_last=False):
        if nulls_first and nulls_last:
            raise ValueError('nulls_first and nulls_last are mutually exclusive')
        self.nulls_first = nulls_first
        self.nulls_last = nulls_last
        self.descending = descending
        if not hasattr(expression, 'resolve_expression'):
            raise ValueError('expression must be an expression type')
        self.expression = expression

    def __repr__(self):
        return "{}({}, descending={})".format(
            self.__class__.__name__, self.expression, self.descending)

    def set_source_expressions(self, exprs):
        self.expression = exprs[0]

    def get_source_expressions(self):
        return [self.expression]
```
### 19 - django/db/models/expressions.py:

Start line: 300, End line: 319

```python
@deconstructible
class BaseExpression:

    @staticmethod
    def _convert_value_noop(value, expression, connection):
        return value

    @cached_property
    def convert_value(self):
        """
        Expressions provide their own converters because users have the option
        of manually specifying the output_field which may be a different type
        from the one the database returns.
        """
        field = self.output_field
        internal_type = field.get_internal_type()
        if internal_type == 'FloatField':
            return lambda value, expression, connection: None if value is None else float(value)
        elif internal_type.endswith('IntegerField'):
            return lambda value, expression, connection: None if value is None else int(value)
        elif internal_type == 'DecimalField':
            return lambda value, expression, connection: None if value is None else Decimal(value)
        return self._convert_value_noop
```
### 20 - django/db/models/expressions.py:

Start line: 420, End line: 455

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
### 21 - django/db/models/sql/query.py:

Start line: 1376, End line: 1387

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
### 22 - django/db/models/expressions.py:

Start line: 1, End line: 28

```python
import copy
import datetime
import inspect
from decimal import Decimal

from django.core.exceptions import EmptyResultSet, FieldError
from django.db import connection
from django.db.models import fields
from django.db.models.query_utils import Q
from django.db.utils import NotSupportedError
from django.utils.deconstruct import deconstructible
from django.utils.functional import cached_property
from django.utils.hashable import make_hashable


class SQLiteNumericMixin:
    """
    Some expressions with output_field=DecimalField() must be cast to
    numeric to be properly filtered.
    """
    def as_sqlite(self, compiler, connection, **extra_context):
        sql, params = self.as_sql(compiler, connection, **extra_context)
        try:
            if self.output_field.get_internal_type() == 'DecimalField':
                sql = 'CAST(%s AS NUMERIC)' % sql
        except FieldError:
            pass
        return sql, params
```
### 23 - django/db/models/expressions.py:

Start line: 1122, End line: 1155

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
### 24 - django/db/models/expressions.py:

Start line: 212, End line: 246

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
### 25 - django/db/models/expressions.py:

Start line: 500, End line: 530

```python
@deconstructible
class F(Combinable):
    """An object capable of resolving references to existing query objects."""
    # Can the expression be used in a WHERE clause?
    filterable = True

    def __init__(self, name):
        """
        Arguments:
         * name: the name of the field this expression references
        """
        self.name = name

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.name)

    def resolve_expression(self, query=None, allow_joins=True, reuse=None,
                           summarize=False, for_save=False, simple_col=False):
        return query.resolve_ref(self.name, allow_joins, reuse, summarize, simple_col)

    def asc(self, **kwargs):
        return OrderBy(self, **kwargs)

    def desc(self, **kwargs):
        return OrderBy(self, descending=True, **kwargs)

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.name == other.name

    def __hash__(self):
        return hash(self.name)
```
### 26 - django/db/models/expressions.py:

Start line: 1105, End line: 1120

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
### 27 - django/db/models/sql/query.py:

Start line: 1623, End line: 1658

```python
class Query(BaseExpression):

    def resolve_ref(self, name, allow_joins=True, reuse=None, summarize=False, simple_col=False):
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
            col = _get_col(targets[0], join_info.targets[0], join_list[-1], simple_col)
            return col
```
### 30 - django/db/models/expressions.py:

Start line: 273, End line: 298

```python
@deconstructible
class BaseExpression:

    def _resolve_output_field(self):
        """
        Attempt to infer the output type of the expression. If the output
        fields of all source fields match then, simply infer the same type
        here. This isn't always correct, but it makes sense most of the time.

        Consider the difference between `2 + 2` and `2 / 3`. Inferring
        the type here is a convenience for the common case. The user should
        supply their own output_field with more complex computations.

        If a source's output field resolves to None, exclude it from this check.
        If all sources are None, then an error is raised higher up the stack in
        the output_field property.
        """
        sources_iter = (source for source in self.get_source_fields() if source is not None)
        for output_field in sources_iter:
            for source in sources_iter:
                if not isinstance(output_field, source.__class__):
                    raise FieldError(
                        'Expression contains mixed types: %s, %s. You must '
                        'set output_field.' % (
                            output_field.__class__.__name__,
                            source.__class__.__name__,
                        )
                    )
            return output_field
```
### 31 - django/db/models/expressions.py:

Start line: 702, End line: 728

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
### 33 - django/db/models/expressions.py:

Start line: 731, End line: 759

```python
class Col(Expression):

    contains_column_references = True

    def __init__(self, alias, target, output_field=None):
        if output_field is None:
            output_field = target
        super().__init__(output_field=output_field)
        self.alias, self.target = alias, target

    def __repr__(self):
        return "{}({}, {})".format(
            self.__class__.__name__, self.alias, self.target)

    def as_sql(self, compiler, connection):
        qn = compiler.quote_name_unless_alias
        return "%s.%s" % (qn(self.alias), qn(self.target.column)), []

    def relabeled_clone(self, relabels):
        return self.__class__(relabels.get(self.alias, self.alias), self.target, self.output_field)

    def get_group_by_cols(self, alias=None):
        return [self]

    def get_db_converters(self, connection):
        if self.target == self.output_field:
            return self.output_field.get_db_converters(connection)
        return (self.output_field.get_db_converters(connection) +
                self.target.get_db_converters(connection))
```
### 34 - django/db/models/expressions.py:

Start line: 369, End line: 392

```python
@deconstructible
class BaseExpression:

    @cached_property
    def identity(self):
        constructor_signature = inspect.signature(self.__init__)
        args, kwargs = self._constructor_args
        signature = constructor_signature.bind_partial(*args, **kwargs)
        signature.apply_defaults()
        arguments = signature.arguments.items()
        identity = [self.__class__]
        for arg, value in arguments:
            if isinstance(value, fields.Field):
                if value.name and value.model:
                    value = (value.model._meta.label, value.name)
                else:
                    value = type(value)
            else:
                value = make_hashable(value)
            identity.append((arg, value))
        return tuple(identity)

    def __eq__(self, other):
        return isinstance(other, BaseExpression) and other.identity == self.identity

    def __hash__(self):
        return hash(self.identity)
```
### 35 - django/db/models/expressions.py:

Start line: 458, End line: 497

```python
class DurationExpression(CombinedExpression):
    def compile(self, side, compiler, connection):
        if not isinstance(side, DurationValue):
            try:
                output = side.output_field
            except FieldError:
                pass
            else:
                if output.get_internal_type() == 'DurationField':
                    sql, params = compiler.compile(side)
                    return connection.ops.format_for_duration_arithmetic(sql), params
        return compiler.compile(side)

    def as_sql(self, compiler, connection):
        connection.ops.check_expression_support(self)
        expressions = []
        expression_params = []
        sql, params = self.compile(self.lhs, compiler, connection)
        expressions.append(sql)
        expression_params.extend(params)
        sql, params = self.compile(self.rhs, compiler, connection)
        expressions.append(sql)
        expression_params.extend(params)
        # order of precedence
        expression_wrapper = '(%s)'
        sql = connection.ops.combine_duration_expression(self.connector, expressions)
        return expression_wrapper % sql, expression_params


class TemporalSubtraction(CombinedExpression):
    output_field = fields.DurationField()

    def __init__(self, lhs, rhs):
        super().__init__(lhs, self.SUB, rhs)

    def as_sql(self, compiler, connection):
        connection.ops.check_expression_support(self)
        lhs = compiler.compile(self.lhs, connection)
        rhs = compiler.compile(self.rhs, connection)
        return connection.ops.subtract_temporals(self.lhs.output_field.get_internal_type(), lhs, rhs)
```
### 36 - django/db/models/expressions.py:

Start line: 976, End line: 1001

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
```
### 38 - django/db/models/expressions.py:

Start line: 321, End line: 367

```python
@deconstructible
class BaseExpression:

    def get_lookup(self, lookup):
        return self.output_field.get_lookup(lookup)

    def get_transform(self, name):
        return self.output_field.get_transform(name)

    def relabeled_clone(self, change_map):
        clone = self.copy()
        clone.set_source_expressions([
            e.relabeled_clone(change_map) if e is not None else None
            for e in self.get_source_expressions()
        ])
        return clone

    def copy(self):
        return copy.copy(self)

    def get_group_by_cols(self, alias=None):
        if not self.contains_aggregate:
            return [self]
        cols = []
        for source in self.get_source_expressions():
            cols.extend(source.get_group_by_cols())
        return cols

    def get_source_fields(self):
        """Return the underlying field types used by this aggregate."""
        return [e._output_field_or_none for e in self.get_source_expressions()]

    def asc(self, **kwargs):
        return OrderBy(self, **kwargs)

    def desc(self, **kwargs):
        return OrderBy(self, descending=True, **kwargs)

    def reverse_ordering(self):
        return self

    def flatten(self):
        """
        Recursively yield this expression and all subexpressions, in
        depth-first order.
        """
        yield self
        for expr in self.get_source_expressions():
            if expr:
                yield from expr.flatten()
```
### 39 - django/db/models/sql/query.py:

Start line: 1097, End line: 1115

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
```
### 41 - django/db/models/expressions.py:

Start line: 1051, End line: 1079

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

    def as_oracle(self, compiler, connection, template=None, **extra_context):
        # Oracle doesn't allow EXISTS() in the SELECT list, so wrap it with a
        # CASE WHEN expression. Change the template since the When expression
        # requires a left hand side (column) to compare against.
        sql, params = self.as_sql(compiler, connection, template, **extra_context)
        sql = 'CASE WHEN {} THEN 1 ELSE 0 END'.format(sql)
        return sql, params
```
### 47 - django/db/models/sql/query.py:

Start line: 2196, End line: 2212

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
### 50 - django/db/models/sql/query.py:

Start line: 1042, End line: 1066

```python
class Query(BaseExpression):

    def as_sql(self, compiler, connection):
        sql, params = self.get_compiler(connection=connection).as_sql()
        if self.subquery:
            sql = '(%s)' % sql
        return sql, params

    def resolve_lookup_value(self, value, can_reuse, allow_joins, simple_col):
        if hasattr(value, 'resolve_expression'):
            kwargs = {'reuse': can_reuse, 'allow_joins': allow_joins}
            if isinstance(value, F):
                kwargs['simple_col'] = simple_col
            value = value.resolve_expression(self, **kwargs)
        elif isinstance(value, (list, tuple)):
            # The items of the iterable may be expressions and therefore need
            # to be resolved independently.
            for sub_value in value:
                if hasattr(sub_value, 'resolve_expression'):
                    if isinstance(sub_value, F):
                        sub_value.resolve_expression(
                            self, reuse=can_reuse, allow_joins=allow_joins,
                            simple_col=simple_col,
                        )
                    else:
                        sub_value.resolve_expression(self, reuse=can_reuse, allow_joins=allow_joins)
        return value
```
### 51 - django/db/models/expressions.py:

Start line: 395, End line: 418

```python
class Expression(BaseExpression, Combinable):
    """An expression that can be combined with other expressions."""
    pass


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
```
### 52 - django/db/models/expressions.py:

Start line: 248, End line: 271

```python
@deconstructible
class BaseExpression:

    @property
    def field(self):
        return self.output_field

    @cached_property
    def output_field(self):
        """Return the output type of this expressions."""
        output_field = self._resolve_output_field()
        if output_field is None:
            self._output_field_resolved_to_none = True
            raise FieldError('Cannot resolve expression type, unknown output_field')
        return output_field

    @cached_property
    def _output_field_or_none(self):
        """
        Return the output field of this expression, or None if
        _resolve_output_field() didn't return an output type.
        """
        try:
            return self.output_field
        except FieldError:
            if not self._output_field_resolved_to_none:
                raise
```
### 53 - django/db/models/expressions.py:

Start line: 928, End line: 974

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
### 57 - django/db/models/sql/query.py:

Start line: 1760, End line: 1809

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
### 59 - django/db/models/expressions.py:

Start line: 678, End line: 700

```python
class DurationValue(Value):
    def as_sql(self, compiler, connection):
        connection.ops.check_expression_support(self)
        if connection.features.has_native_duration_field:
            return super().as_sql(compiler, connection)
        return connection.ops.date_interval_sql(self.value), []


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
### 62 - django/db/models/expressions.py:

Start line: 1004, End line: 1048

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
        return []
```
### 65 - django/db/models/sql/query.py:

Start line: 1405, End line: 1483

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
### 67 - django/db/models/sql/query.py:

Start line: 882, End line: 904

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
### 71 - django/db/models/expressions.py:

Start line: 606, End line: 631

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
### 78 - django/db/models/sql/query.py:

Start line: 689, End line: 724

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
### 79 - django/db/models/sql/query.py:

Start line: 1303, End line: 1324

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

    def build_where(self, q_object):
        return self._add_q(q_object, used_aliases=set(), allow_joins=False, simple_col=True)[0]
```
### 84 - django/db/models/sql/query.py:

Start line: 358, End line: 408

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
### 93 - django/db/models/sql/query.py:

Start line: 1731, End line: 1758

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
### 105 - django/db/models/expressions.py:

Start line: 829, End line: 843

```python
class ExpressionList(Func):
    """
    An expression containing multiple expressions. Can be used to provide a
    list of expressions as an argument to another expression, like an
    ordering clause.
    """
    template = '%(expressions)s'

    def __init__(self, *expressions, **extra):
        if not expressions:
            raise ValueError('%s requires at least one expression.' % self.__class__.__name__)
        super().__init__(*expressions, **extra)

    def __str__(self):
        return self.arg_joiner.join(str(arg) for arg in self.source_expressions)
```
### 120 - django/db/models/sql/query.py:

Start line: 1561, End line: 1585

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
