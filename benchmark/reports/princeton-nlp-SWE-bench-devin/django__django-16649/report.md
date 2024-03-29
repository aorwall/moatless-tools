# django__django-16649

| **django/django** | `39d1e45227e060746ed461fddde80fa2b6cf0dcd` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 1978 |
| **Any found context length** | 514 |
| **Avg pos** | 16.0 |
| **Min pos** | 2 |
| **Max pos** | 8 |
| **Top file pos** | 2 |
| **Missing snippets** | 3 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/db/models/sql/query.py b/django/db/models/sql/query.py
--- a/django/db/models/sql/query.py
+++ b/django/db/models/sql/query.py
@@ -1087,7 +1087,12 @@ def add_annotation(self, annotation, alias, select=True):
         if select:
             self.append_annotation_mask([alias])
         else:
-            self.set_annotation_mask(set(self.annotation_select).difference({alias}))
+            annotation_mask = (
+                value
+                for value in dict.fromkeys(self.annotation_select)
+                if value != alias
+            )
+            self.set_annotation_mask(annotation_mask)
         self.annotations[alias] = annotation
 
     def resolve_expression(self, query, *args, **kwargs):
@@ -2341,12 +2346,12 @@ def set_annotation_mask(self, names):
         if names is None:
             self.annotation_select_mask = None
         else:
-            self.annotation_select_mask = set(names)
+            self.annotation_select_mask = list(dict.fromkeys(names))
         self._annotation_select_cache = None
 
     def append_annotation_mask(self, names):
         if self.annotation_select_mask is not None:
-            self.set_annotation_mask(self.annotation_select_mask.union(names))
+            self.set_annotation_mask((*self.annotation_select_mask, *names))
 
     def set_extra_mask(self, names):
         """
@@ -2423,9 +2428,9 @@ def annotation_select(self):
             return {}
         elif self.annotation_select_mask is not None:
             self._annotation_select_cache = {
-                k: v
-                for k, v in self.annotations.items()
-                if k in self.annotation_select_mask
+                k: self.annotations[k]
+                for k in self.annotation_select_mask
+                if k in self.annotations
             }
             return self._annotation_select_cache
         else:

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/db/models/sql/query.py | 1090 | 1090 | 6 | 2 | 1304
| django/db/models/sql/query.py | 2344 | 2349 | 2 | 2 | 514
| django/db/models/sql/query.py | 2426 | 2428 | 8 | 2 | 1978


## Problem Statement

```
Querysets: annotate() columns are forced into a certain position which may disrupt union()
Description
	
(Reporting possible issue found by a user on #django)
Using values() to force selection of certain columns in a certain order proved useful unioning querysets with union() for the aforementioned user. The positioning of columns added with annotate() is not controllable with values() and has the potential to disrupt union() unless this fact is known and the ordering done in a certain way to accommodate it.
I'm reporting this mainly for posterity but also as a highlight that perhaps this should be mentioned in the documentation. I'm sure there are reasons why the annotations are appended to the select but if someone feels that this is changeable then it would be a bonus outcome.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 django/db/models/query.py | 1601 | 1653| 342 | 342 | 20490 | 
| **-> 2 <-** | **2 django/db/models/sql/query.py** | 2339 | 2360| 172 | 514 | 43128 | 
| 3 | 2 django/db/models/query.py | 1586 | 1599| 119 | 633 | 43128 | 
| 4 | 2 django/db/models/query.py | 1495 | 1504| 130 | 763 | 43128 | 
| 5 | **2 django/db/models/sql/query.py** | 2362 | 2412| 406 | 1169 | 43128 | 
| **-> 6 <-** | **2 django/db/models/sql/query.py** | 1076 | 1091| 135 | 1304 | 43128 | 
| 7 | **2 django/db/models/sql/query.py** | 1900 | 1948| 445 | 1749 | 43128 | 
| **-> 8 <-** | **2 django/db/models/sql/query.py** | 2414 | 2446| 229 | 1978 | 43128 | 
| 9 | 2 django/db/models/query.py | 1740 | 1769| 189 | 2167 | 43128 | 
| 10 | 2 django/db/models/query.py | 451 | 483| 253 | 2420 | 43128 | 
| 11 | 2 django/db/models/query.py | 1324 | 1360| 267 | 2687 | 43128 | 
| 12 | 2 django/db/models/query.py | 1217 | 1238| 192 | 2879 | 43128 | 
| 13 | 3 django/db/models/sql/where.py | 338 | 356| 149 | 3028 | 45710 | 
| 14 | 3 django/db/models/query.py | 663 | 715| 424 | 3452 | 45710 | 
| 15 | 3 django/db/models/query.py | 1901 | 1911| 118 | 3570 | 45710 | 
| 16 | **3 django/db/models/sql/query.py** | 419 | 529| 949 | 4519 | 45710 | 
| 17 | 3 django/db/models/query.py | 485 | 502| 150 | 4669 | 45710 | 
| 18 | **3 django/db/models/sql/query.py** | 2101 | 2150| 369 | 5038 | 45710 | 
| 19 | **3 django/db/models/sql/query.py** | 2592 | 2648| 823 | 5861 | 45710 | 
| 20 | **3 django/db/models/sql/query.py** | 1814 | 1844| 244 | 6105 | 45710 | 
| 21 | 4 django/db/models/sql/compiler.py | 229 | 314| 697 | 6802 | 62453 | 
| 22 | 4 django/db/models/query.py | 1913 | 1978| 543 | 7345 | 62453 | 
| 23 | **4 django/db/models/sql/query.py** | 871 | 914| 445 | 7790 | 62453 | 
| 24 | 4 django/db/models/query.py | 2023 | 2038| 147 | 7937 | 62453 | 
| 25 | 5 django/db/models/base.py | 2080 | 2185| 733 | 8670 | 81180 | 
| 26 | 6 django/db/backends/postgresql/schema.py | 142 | 255| 920 | 9590 | 84017 | 
| 27 | 6 django/db/models/query.py | 1178 | 1215| 300 | 9890 | 84017 | 
| 28 | 6 django/db/models/query.py | 1506 | 1519| 130 | 10020 | 84017 | 
| 29 | 6 django/db/models/sql/where.py | 184 | 264| 549 | 10569 | 84017 | 
| 30 | 6 django/db/models/sql/compiler.py | 1436 | 1479| 349 | 10918 | 84017 | 
| 31 | 6 django/db/models/base.py | 1790 | 1813| 180 | 11098 | 84017 | 
| 32 | 6 django/db/backends/postgresql/schema.py | 257 | 274| 170 | 11268 | 84017 | 
| 33 | 6 django/db/models/sql/compiler.py | 341 | 444| 772 | 12040 | 84017 | 
| 34 | 6 django/db/models/query.py | 1702 | 1717| 153 | 12193 | 84017 | 
| 35 | 6 django/db/models/sql/compiler.py | 1996 | 2057| 588 | 12781 | 84017 | 
| 36 | 7 django/db/models/sql/subqueries.py | 48 | 78| 212 | 12993 | 85245 | 
| 37 | 7 django/db/models/query.py | 504 | 521| 136 | 13129 | 85245 | 
| 38 | **7 django/db/models/sql/query.py** | 653 | 693| 383 | 13512 | 85245 | 
| 39 | 7 django/db/models/query.py | 1655 | 1700| 344 | 13856 | 85245 | 
| 40 | 8 django/db/models/query_utils.py | 396 | 436| 286 | 14142 | 88457 | 
| 41 | 8 django/db/models/sql/compiler.py | 1089 | 1102| 166 | 14308 | 88457 | 
| 42 | 9 django/db/migrations/operations/models.py | 674 | 698| 231 | 14539 | 96337 | 
| 43 | 9 django/db/models/sql/subqueries.py | 116 | 139| 193 | 14732 | 96337 | 
| 44 | 9 django/db/models/sql/compiler.py | 1596 | 1618| 254 | 14986 | 96337 | 
| 45 | 9 django/db/migrations/operations/models.py | 651 | 672| 162 | 15148 | 96337 | 
| 46 | 9 django/db/models/query.py | 1482 | 1493| 124 | 15272 | 96337 | 
| 47 | **9 django/db/models/sql/query.py** | 1404 | 1479| 765 | 16037 | 96337 | 
| 48 | 10 django/contrib/postgres/aggregates/general.py | 1 | 51| 263 | 16300 | 96993 | 
| 49 | **10 django/db/models/sql/query.py** | 843 | 869| 280 | 16580 | 96993 | 
| 50 | 10 django/contrib/postgres/aggregates/general.py | 54 | 99| 393 | 16973 | 96993 | 
| 51 | 11 django/contrib/postgres/aggregates/mixins.py | 1 | 30| 268 | 17241 | 97261 | 
| 52 | 11 django/db/models/query.py | 353 | 380| 225 | 17466 | 97261 | 
| 53 | 11 django/db/models/base.py | 1875 | 1902| 191 | 17657 | 97261 | 
| 54 | 12 django/db/backends/ddl_references.py | 115 | 135| 167 | 17824 | 98890 | 
| 55 | 13 django/db/backends/mysql/compiler.py | 55 | 85| 240 | 18064 | 99537 | 
| 56 | 13 django/db/backends/postgresql/schema.py | 276 | 310| 277 | 18341 | 99537 | 
| 57 | **13 django/db/models/sql/query.py** | 2204 | 2235| 282 | 18623 | 99537 | 
| 58 | 14 django/db/backends/base/schema.py | 1313 | 1357| 412 | 19035 | 114119 | 
| 59 | 14 django/db/models/query.py | 1881 | 1899| 190 | 19225 | 114119 | 
| 60 | 15 django/db/models/aggregates.py | 97 | 144| 372 | 19597 | 115649 | 
| 61 | 15 django/db/models/sql/compiler.py | 316 | 339| 223 | 19820 | 115649 | 
| 62 | 15 django/db/backends/base/schema.py | 1520 | 1535| 206 | 20026 | 115649 | 
| 63 | 16 django/db/backends/sqlite3/schema.py | 122 | 173| 527 | 20553 | 120364 | 
| 64 | **16 django/db/models/sql/query.py** | 1621 | 1719| 846 | 21399 | 120364 | 
| 65 | 17 django/db/models/fields/__init__.py | 495 | 516| 169 | 21568 | 139368 | 
| 66 | 17 django/db/models/sql/compiler.py | 39 | 76| 376 | 21944 | 139368 | 
| 67 | 17 django/db/models/sql/compiler.py | 1400 | 1411| 145 | 22089 | 139368 | 
| 68 | 18 django/db/models/expressions.py | 1113 | 1149| 295 | 22384 | 152632 | 
| 69 | **18 django/db/models/sql/query.py** | 2237 | 2284| 376 | 22760 | 152632 | 
| 70 | 18 django/db/backends/postgresql/schema.py | 312 | 337| 235 | 22995 | 152632 | 
| 71 | 19 django/contrib/admin/options.py | 243 | 256| 139 | 23134 | 171942 | 
| 72 | 19 django/db/models/query.py | 1719 | 1738| 213 | 23347 | 171942 | 
| 73 | 19 django/db/models/sql/compiler.py | 1 | 36| 249 | 23596 | 171942 | 
| 74 | **19 django/db/models/sql/query.py** | 1574 | 1603| 287 | 23883 | 171942 | 
| 75 | 19 django/db/backends/sqlite3/schema.py | 553 | 577| 162 | 24045 | 171942 | 
| 76 | 19 django/db/backends/ddl_references.py | 223 | 255| 252 | 24297 | 171942 | 
| 77 | 19 django/db/backends/postgresql/schema.py | 121 | 140| 234 | 24531 | 171942 | 
| 78 | 19 django/db/models/sql/compiler.py | 666 | 946| 503 | 25034 | 171942 | 
| 79 | 19 django/db/backends/base/schema.py | 948 | 1035| 795 | 25829 | 171942 | 
| 80 | 19 django/db/models/sql/compiler.py | 1375 | 1398| 207 | 26036 | 171942 | 
| 81 | 20 django/contrib/admin/views/main.py | 410 | 473| 513 | 26549 | 176700 | 
| 82 | 20 django/db/models/base.py | 1904 | 1932| 192 | 26741 | 176700 | 
| 83 | 20 django/db/models/query.py | 1275 | 1322| 361 | 27102 | 176700 | 
| 84 | **20 django/db/models/sql/query.py** | 1 | 79| 576 | 27678 | 176700 | 
| 85 | 20 django/db/models/sql/compiler.py | 647 | 664| 170 | 27848 | 176700 | 
| 86 | 21 django/contrib/gis/db/backends/postgis/schema.py | 53 | 82| 227 | 28075 | 177398 | 
| 87 | 21 django/db/models/sql/compiler.py | 446 | 505| 561 | 28636 | 177398 | 
| 88 | 21 django/db/models/query.py | 1154 | 1176| 160 | 28796 | 177398 | 
| 89 | 22 django/db/backends/mysql/schema.py | 1 | 44| 484 | 29280 | 179589 | 
| 90 | **22 django/db/models/sql/query.py** | 2152 | 2186| 288 | 29568 | 179589 | 
| 91 | 23 django/db/backends/mysql/operations.py | 436 | 465| 274 | 29842 | 183767 | 
| 92 | 24 django/db/migrations/autodetector.py | 1097 | 1214| 982 | 30824 | 197518 | 
| 93 | 24 django/db/models/aggregates.py | 60 | 95| 353 | 31177 | 197518 | 
| 94 | **24 django/db/models/sql/query.py** | 2188 | 2202| 132 | 31309 | 197518 | 
| 95 | 24 django/db/migrations/operations/models.py | 627 | 648| 148 | 31457 | 197518 | 


### Hint

```
(The ticket component should change to "Documentation" if there aren't any code changes to make here. I'm not sure.)
Probably duplicate of #28900.
I've stumbled upon a case in production where this limitation prevents me for making a useful query. I've been able to create a test to reproduce this problem. It works with sqlite but fails with postgres. Add this test to tests/queries/test_qs_combinators.py: from django.db.models import F, IntegerField, TextField, Value def test_union_with_two_annotated_values_on_different_models(self): qs1 = Number.objects.annotate( text_annotation=Value('Foo', TextField()) ).values('text_annotation', 'num') qs2 = ReservedName.objects.annotate( int_annotation=Value(1, IntegerField()), ).values('name', 'int_annotation') self.assertEqual(qs1.union(qs2).count(), 10) In current master (78f8b80f9b215e50618375adce4c97795dabbb84), running ./runtests.py --parallel=1 --settings=tests.test_postgres queries.test_qs_combinators.QuerySetSetOperationTests.test_union_with_two_annotated_values_on_different_models fails: Testing against Django installed in 'django/django' Creating test database for alias 'default'... Creating test database for alias 'other'... System check identified no issues (1 silenced). E ====================================================================== ERROR: test_union_with_two_annotated_values_on_different_models (queries.test_qs_combinators.QuerySetSetOperationTests) ---------------------------------------------------------------------- Traceback (most recent call last): File "django/django/db/backends/utils.py", line 85, in _execute return self.cursor.execute(sql, params) psycopg2.ProgrammingError: UNION types integer and character varying cannot be matched LINE 1: ..._annotation" FROM "queries_number") UNION (SELECT "queries_r... ^ The above exception was the direct cause of the following exception: Traceback (most recent call last): File "django/tests/queries/test_qs_combinators.py", line 140, in test_union_with_two_annotated_values_on_different_models self.assertEqual(qs1.union(qs2).count(), 10) File "django/django/db/models/query.py", line 382, in count return self.query.get_count(using=self.db) File "django/django/db/models/sql/query.py", line 494, in get_count number = obj.get_aggregation(using, ['__count'])['__count'] File "django/django/db/models/sql/query.py", line 479, in get_aggregation result = compiler.execute_sql(SINGLE) File "django/django/db/models/sql/compiler.py", line 1054, in execute_sql cursor.execute(sql, params) File "django/django/db/backends/utils.py", line 68, in execute return self._execute_with_wrappers(sql, params, many=False, executor=self._execute) File "django/django/db/backends/utils.py", line 77, in _execute_with_wrappers return executor(sql, params, many, context) File "django/django/db/backends/utils.py", line 85, in _execute return self.cursor.execute(sql, params) File "django/django/db/utils.py", line 89, in __exit__ raise dj_exc_value.with_traceback(traceback) from exc_value File "django/django/db/backends/utils.py", line 85, in _execute return self.cursor.execute(sql, params) django.db.utils.ProgrammingError: UNION types integer and character varying cannot be matched LINE 1: ..._annotation" FROM "queries_number") UNION (SELECT "queries_r... ^ ---------------------------------------------------------------------- Ran 1 test in 0.007s FAILED (errors=1) Destroying test database for alias 'default'... Destroying test database for alias 'other'... My tests/test_postgres.py is: DATABASES = { 'default': { 'ENGINE': 'django.db.backends.postgresql_psycopg2', 'NAME': 'django-test', 'HOST': '127.0.0.1', }, 'other': { 'ENGINE': 'django.db.backends.postgresql_psycopg2', 'NAME': 'django-test-other', } } SECRET_KEY = "django_tests_secret_key" # Use a fast hasher to speed up tests. PASSWORD_HASHERS = [ 'django.contrib.auth.hashers.MD5PasswordHasher', ]
There's a ticket with a test case in #30211. That ticket was reported as a bug, and I think this ticket should be a bug too, so I'm changing the classification for now (apologies if that's inappropriate). I think it's a bug because the documentation in my reading implies that as long as the columns match, union will work. So it really comes as a bit of a surprise that Django overrides the order in values_list(). In my particular case, I'm using union() to combine two different tables that I need to get out in sorted order, and I was trying to use annotate() + values_list() to add a NULL filler to one table as it lacks a column from the other. Also, I suppose the ORM could possibly also be a bit more efficient if it could return values_list() tuples directly from the select instead of having to rearrange them?
Actually there could be a different problem as well. We were lucky in that the ORM generated SQL where the data types of the columns do not match. But what happens when the data types of the columns match? I'm afraid you would get a query that doesn't throw an exception but is in fact subtly broken, especially if the values of the different columns happen to be similar, in which case it might take a long time for app developers to realize that the query is broken.
I just hit this as well. The solution (workaround) seems to be using F() and Value() freely and consistently for all querysets even if it doesn't look necessary on the surface. See ​https://github.com/matthiask/feincms3-forms/commit/c112a7d613e991780f383393fd05f1c84c81a279 (It's a bit surprising that values_list doesn't produce a SQL query with the exact same ordering of values, but after thinking about it some more I'm not sure if that's really a bug or just a sharp edge of the current implementation.)
I submitted a failing unit test for this issue which demonstrates the problem: ​https://github.com/django/django/pull/16577 This seems hard to fix. I don't expect to do any work on this in the next few months since I have found a workaround for my use case. Hopefully the test is useful to someone.
```

## Patch

```diff
diff --git a/django/db/models/sql/query.py b/django/db/models/sql/query.py
--- a/django/db/models/sql/query.py
+++ b/django/db/models/sql/query.py
@@ -1087,7 +1087,12 @@ def add_annotation(self, annotation, alias, select=True):
         if select:
             self.append_annotation_mask([alias])
         else:
-            self.set_annotation_mask(set(self.annotation_select).difference({alias}))
+            annotation_mask = (
+                value
+                for value in dict.fromkeys(self.annotation_select)
+                if value != alias
+            )
+            self.set_annotation_mask(annotation_mask)
         self.annotations[alias] = annotation
 
     def resolve_expression(self, query, *args, **kwargs):
@@ -2341,12 +2346,12 @@ def set_annotation_mask(self, names):
         if names is None:
             self.annotation_select_mask = None
         else:
-            self.annotation_select_mask = set(names)
+            self.annotation_select_mask = list(dict.fromkeys(names))
         self._annotation_select_cache = None
 
     def append_annotation_mask(self, names):
         if self.annotation_select_mask is not None:
-            self.set_annotation_mask(self.annotation_select_mask.union(names))
+            self.set_annotation_mask((*self.annotation_select_mask, *names))
 
     def set_extra_mask(self, names):
         """
@@ -2423,9 +2428,9 @@ def annotation_select(self):
             return {}
         elif self.annotation_select_mask is not None:
             self._annotation_select_cache = {
-                k: v
-                for k, v in self.annotations.items()
-                if k in self.annotation_select_mask
+                k: self.annotations[k]
+                for k in self.annotation_select_mask
+                if k in self.annotations
             }
             return self._annotation_select_cache
         else:

```

## Test Patch

```diff
diff --git a/tests/postgres_tests/test_array.py b/tests/postgres_tests/test_array.py
--- a/tests/postgres_tests/test_array.py
+++ b/tests/postgres_tests/test_array.py
@@ -466,8 +466,8 @@ def test_group_by_order_by_select_index(self):
                 ],
             )
         sql = ctx[0]["sql"]
-        self.assertIn("GROUP BY 1", sql)
-        self.assertIn("ORDER BY 1", sql)
+        self.assertIn("GROUP BY 2", sql)
+        self.assertIn("ORDER BY 2", sql)
 
     def test_index(self):
         self.assertSequenceEqual(
diff --git a/tests/queries/test_qs_combinators.py b/tests/queries/test_qs_combinators.py
--- a/tests/queries/test_qs_combinators.py
+++ b/tests/queries/test_qs_combinators.py
@@ -246,7 +246,7 @@ def test_union_with_two_annotated_values_list(self):
             )
             .values_list("num", "count")
         )
-        self.assertCountEqual(qs1.union(qs2), [(1, 0), (2, 1)])
+        self.assertCountEqual(qs1.union(qs2), [(1, 0), (1, 2)])
 
     def test_union_with_extra_and_values_list(self):
         qs1 = (
@@ -368,6 +368,20 @@ def test_union_multiple_models_with_values_list_and_order_by_extra_select(self):
             [reserved_name.pk],
         )
 
+    def test_union_multiple_models_with_values_list_and_annotations(self):
+        ReservedName.objects.create(name="rn1", order=10)
+        Celebrity.objects.create(name="c1")
+        qs1 = ReservedName.objects.annotate(row_type=Value("rn")).values_list(
+            "name", "order", "row_type"
+        )
+        qs2 = Celebrity.objects.annotate(
+            row_type=Value("cb"), order=Value(-10)
+        ).values_list("name", "order", "row_type")
+        self.assertSequenceEqual(
+            qs1.union(qs2).order_by("order"),
+            [("c1", -10, "cb"), ("rn1", 10, "rn")],
+        )
+
     def test_union_in_subquery(self):
         ReservedName.objects.bulk_create(
             [

```


## Code snippets

### 1 - django/db/models/query.py:

Start line: 1601, End line: 1653

```python
class QuerySet(AltersData):

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
### 2 - django/db/models/sql/query.py:

Start line: 2339, End line: 2360

```python
class Query(BaseExpression):

    def set_annotation_mask(self, names):
        """Set the mask of annotations that will be returned by the SELECT."""
        if names is None:
            self.annotation_select_mask = None
        else:
            self.annotation_select_mask = set(names)
        self._annotation_select_cache = None

    def append_annotation_mask(self, names):
        if self.annotation_select_mask is not None:
            self.set_annotation_mask(self.annotation_select_mask.union(names))

    def set_extra_mask(self, names):
        """
        Set the mask of extra select items that will be returned by SELECT.
        Don't remove them from the Query since they might be used later.
        """
        if names is None:
            self.extra_select_mask = None
        else:
            self.extra_select_mask = set(names)
        self._extra_select_cache = None
```
### 3 - django/db/models/query.py:

Start line: 1586, End line: 1599

```python
class QuerySet(AltersData):

    def annotate(self, *args, **kwargs):
        """
        Return a query set in which the returned objects have been annotated
        with extra data or aggregations.
        """
        self._not_support_combined_queries("annotate")
        return self._annotate(args, kwargs, select=True)

    def alias(self, *args, **kwargs):
        """
        Return a query set with added aliases for extra data or aggregations.
        """
        self._not_support_combined_queries("alias")
        return self._annotate(args, kwargs, select=False)
```
### 4 - django/db/models/query.py:

Start line: 1495, End line: 1504

```python
class QuerySet(AltersData):

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
### 5 - django/db/models/sql/query.py:

Start line: 2362, End line: 2412

```python
class Query(BaseExpression):

    def set_values(self, fields):
        self.select_related = False
        self.clear_deferred_loading()
        self.clear_select_fields()
        self.has_select_fields = True

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
            self.add_fields(
                (f.attname for f in self.model._meta.concrete_fields), False
            )
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
### 6 - django/db/models/sql/query.py:

Start line: 1076, End line: 1091

```python
class Query(BaseExpression):

    def check_alias(self, alias):
        if FORBIDDEN_ALIAS_PATTERN.search(alias):
            raise ValueError(
                "Column aliases cannot contain whitespace characters, quotation marks, "
                "semicolons, or SQL comments."
            )

    def add_annotation(self, annotation, alias, select=True):
        """Add a single annotation expression to the Query."""
        self.check_alias(alias)
        annotation = annotation.resolve_expression(self, allow_joins=True, reuse=None)
        if select:
            self.append_annotation_mask([alias])
        else:
            self.set_annotation_mask(set(self.annotation_select).difference({alias}))
        self.annotations[alias] = annotation
```
### 7 - django/db/models/sql/query.py:

Start line: 1900, End line: 1948

```python
class Query(BaseExpression):

    def resolve_ref(self, name, allow_joins=True, reuse=None, summarize=False):
        annotation = self.annotations.get(name)
        if annotation is not None:
            if not allow_joins:
                for alias in self._gen_col_aliases([annotation]):
                    if isinstance(self.alias_map[alias], Join):
                        raise FieldError(
                            "Joined field references are not permitted in this query"
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
            join_info = self.setup_joins(
                field_list, self.get_meta(), self.get_initial_alias(), can_reuse=reuse
            )
            targets, final_alias, join_list = self.trim_joins(
                join_info.targets, join_info.joins, join_info.path
            )
            if not allow_joins and len(join_list) > 1:
                raise FieldError(
                    "Joined field references are not permitted in this query"
                )
            if len(targets) > 1:
                raise FieldError(
                    "Referencing multicolumn fields with F() objects isn't supported"
                )
            # Verify that the last lookup in name is a field or a transform:
            # transform_function() raises FieldError if not.
            transform = join_info.transform_function(targets[0], final_alias)
            if reuse is not None:
                reuse.update(join_list)
            return transform
```
### 8 - django/db/models/sql/query.py:

Start line: 2414, End line: 2446

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
                k: v
                for k, v in self.annotations.items()
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
                k: v for k, v in self.extra.items() if k in self.extra_select_mask
            }
            return self._extra_select_cache
        else:
            return self.extra
```
### 9 - django/db/models/query.py:

Start line: 1740, End line: 1769

```python
class QuerySet(AltersData):

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
### 10 - django/db/models/query.py:

Start line: 451, End line: 483

```python
class QuerySet(AltersData):

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
### 16 - django/db/models/sql/query.py:

Start line: 419, End line: 529

```python
class Query(BaseExpression):

    def get_aggregation(self, using, aggregate_exprs):
        # ... other code
        if (
            isinstance(self.group_by, tuple)
            or self.is_sliced
            or has_existing_aggregation
            or qualify
            or self.distinct
            or self.combinator
        ):
            from django.db.models.sql.subqueries import AggregateQuery

            inner_query = self.clone()
            inner_query.subquery = True
            outer_query = AggregateQuery(self.model, inner_query)
            inner_query.select_for_update = False
            inner_query.select_related = False
            inner_query.set_annotation_mask(self.annotation_select)
            # Queries with distinct_fields need ordering and when a limit is
            # applied we must take the slice from the ordered query. Otherwise
            # no need for ordering.
            inner_query.clear_ordering(force=False)
            if not inner_query.distinct:
                # If the inner query uses default select and it has some
                # aggregate annotations, then we must make sure the inner
                # query is grouped by the main model's primary key. However,
                # clearing the select clause can alter results if distinct is
                # used.
                if inner_query.default_cols and has_existing_aggregation:
                    inner_query.group_by = (
                        self.model._meta.pk.get_col(inner_query.get_initial_alias()),
                    )
                inner_query.default_cols = False
                if not qualify:
                    # Mask existing annotations that are not referenced by
                    # aggregates to be pushed to the outer query unless
                    # filtering against window functions is involved as it
                    # requires complex realising.
                    annotation_mask = set()
                    for aggregate in aggregates.values():
                        annotation_mask |= aggregate.get_refs()
                    inner_query.set_annotation_mask(annotation_mask)

            # Add aggregates to the outer AggregateQuery. This requires making
            # sure all columns referenced by the aggregates are selected in the
            # inner query. It is achieved by retrieving all column references
            # by the aggregates, explicitly selecting them in the inner query,
            # and making sure the aggregates are repointed to them.
            col_refs = {}
            for alias, aggregate in aggregates.items():
                replacements = {}
                for col in self._gen_cols([aggregate], resolve_refs=False):
                    if not (col_ref := col_refs.get(col)):
                        index = len(col_refs) + 1
                        col_alias = f"__col{index}"
                        col_ref = Ref(col_alias, col)
                        col_refs[col] = col_ref
                        inner_query.annotations[col_alias] = col
                        inner_query.append_annotation_mask([col_alias])
                    replacements[col] = col_ref
                outer_query.annotations[alias] = aggregate.replace_expressions(
                    replacements
                )
            if (
                inner_query.select == ()
                and not inner_query.default_cols
                and not inner_query.annotation_select_mask
            ):
                # In case of Model.objects[0:3].count(), there would be no
                # field selected in the inner query, yet we must use a subquery.
                # So, make sure at least one field is selected.
                inner_query.select = (
                    self.model._meta.pk.get_col(inner_query.get_initial_alias()),
                )
        else:
            outer_query = self
            self.select = ()
            self.default_cols = False
            self.extra = {}
            if self.annotations:
                # Inline reference to existing annotations and mask them as
                # they are unnecessary given only the summarized aggregations
                # are requested.
                replacements = {
                    Ref(alias, annotation): annotation
                    for alias, annotation in self.annotations.items()
                }
                self.annotations = {
                    alias: aggregate.replace_expressions(replacements)
                    for alias, aggregate in aggregates.items()
                }
            else:
                self.annotations = aggregates
            self.set_annotation_mask(aggregates)

        empty_set_result = [
            expression.empty_result_set_value
            for expression in outer_query.annotation_select.values()
        ]
        elide_empty = not any(result is NotImplemented for result in empty_set_result)
        outer_query.clear_ordering(force=True)
        outer_query.clear_limits()
        outer_query.select_for_update = False
        outer_query.select_related = False
        compiler = outer_query.get_compiler(using, elide_empty=elide_empty)
        result = compiler.execute_sql(SINGLE)
        if result is None:
            result = empty_set_result
        else:
            converters = compiler.get_converters(outer_query.annotation_select.values())
            result = next(compiler.apply_converters((result,), converters))

        return dict(zip(outer_query.annotation_select, result))
```
### 18 - django/db/models/sql/query.py:

Start line: 2101, End line: 2150

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
                join_info = self.setup_joins(
                    name.split(LOOKUP_SEP), opts, alias, allow_many=allow_m2m
                )
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
                names = sorted(
                    [
                        *get_field_names_from_opts(opts),
                        *self.extra,
                        *self.annotation_select,
                        *self._filtered_relations,
                    ]
                )
                raise FieldError(
                    "Cannot resolve keyword %r into field. "
                    "Choices are: %s" % (name, ", ".join(names))
                )
```
### 19 - django/db/models/sql/query.py:

Start line: 2592, End line: 2648

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
            if self.effective_connector == OR and votes < self.num_children:
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
            if self.effective_connector == AND or (
                self.effective_connector == OR and votes == self.num_children
            ):
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
### 20 - django/db/models/sql/query.py:

Start line: 1814, End line: 1844

```python
class Query(BaseExpression):

    def setup_joins(
        self,
        names,
        opts,
        alias,
        can_reuse=None,
        allow_many=True,
        reuse_with_filtered_relation=False,
    ):
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
            connection = self.join_class(
                opts.db_table,
                alias,
                table_alias,
                INNER,
                join.join_field,
                nullable,
                filtered_relation=filtered_relation,
            )
            reuse = can_reuse if join.m2m or reuse_with_filtered_relation else None
            alias = self.join(
                connection,
                reuse=reuse,
                reuse_with_filtered_relation=reuse_with_filtered_relation,
            )
            joins.append(alias)
            if filtered_relation:
                filtered_relation.path = joins[:]
        return JoinInfo(final_field, targets, opts, joins, path, final_transformer)
```
### 23 - django/db/models/sql/query.py:

Start line: 871, End line: 914

```python
class Query(BaseExpression):

    def change_aliases(self, change_map):
        """
        Change the aliases in change_map (which maps old-alias -> new-alias),
        relabelling any references to them in select columns and the where
        clause.
        """
        # If keys and values of change_map were to intersect, an alias might be
        # updated twice (e.g. T4 -> T5, T5 -> T6, so also T4 -> T6) depending
        # on their order in change_map.
        assert set(change_map).isdisjoint(change_map.values())

        # 1. Update references in "select" (normal columns plus aliases),
        # "group by" and "where".
        self.where.relabel_aliases(change_map)
        if isinstance(self.group_by, tuple):
            self.group_by = tuple(
                [col.relabeled_clone(change_map) for col in self.group_by]
            )
        self.select = tuple([col.relabeled_clone(change_map) for col in self.select])
        self.annotations = self.annotations and {
            key: col.relabeled_clone(change_map)
            for key, col in self.annotations.items()
        }

        # 2. Rename the alias in the internal table/alias datastructures.
        for old_alias, new_alias in change_map.items():
            if old_alias not in self.alias_map:
                continue
            alias_data = self.alias_map[old_alias].relabeled_clone(change_map)
            self.alias_map[new_alias] = alias_data
            self.alias_refcount[new_alias] = self.alias_refcount[old_alias]
            del self.alias_refcount[old_alias]
            del self.alias_map[old_alias]

            table_aliases = self.table_map[alias_data.table_name]
            for pos, alias in enumerate(table_aliases):
                if alias == old_alias:
                    table_aliases[pos] = new_alias
                    break
        self.external_aliases = {
            # Table is aliased or it's being changed and thus is aliased.
            change_map.get(alias, alias): (aliased or alias in change_map)
            for alias, aliased in self.external_aliases.items()
        }
```
### 38 - django/db/models/sql/query.py:

Start line: 653, End line: 693

```python
class Query(BaseExpression):

    def combine(self, rhs, connector):
        # ... other code
        joinpromoter.update_join_types(self)

        # Combine subqueries aliases to ensure aliases relabelling properly
        # handle subqueries when combining where and select clauses.
        self.subq_aliases |= rhs.subq_aliases

        # Now relabel a copy of the rhs where-clause and add it to the current
        # one.
        w = rhs.where.clone()
        w.relabel_aliases(change_map)
        self.where.add(w, connector)

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
                raise ValueError(
                    "When merging querysets using 'or', you cannot have "
                    "extra(select=...) on both sides."
                )
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
### 47 - django/db/models/sql/query.py:

Start line: 1404, End line: 1479

```python
class Query(BaseExpression):

    def build_filter(
        self,
        filter_expr,
        branch_negated=False,
        current_negated=False,
        can_reuse=None,
        allow_joins=True,
        split_subq=True,
        reuse_with_filtered_relation=False,
        check_filterable=True,
        summarize=False,
    ):
        # ... other code

        try:
            join_info = self.setup_joins(
                parts,
                opts,
                alias,
                can_reuse=can_reuse,
                allow_many=allow_many,
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
        targets, alias, join_list = self.trim_joins(
            join_info.targets, join_info.joins, join_info.path
        )
        if can_reuse is not None:
            can_reuse.update(join_list)

        if join_info.final_field.is_relation:
            if len(targets) == 1:
                col = self._get_col(targets[0], join_info.final_field, alias)
            else:
                col = MultiColSource(
                    alias, targets, join_info.targets, join_info.final_field
                )
        else:
            col = self._get_col(targets[0], join_info.final_field, alias)

        condition = self.build_lookup(lookups, col, value)
        lookup_type = condition.lookup_name
        clause = WhereNode([condition], connector=AND)

        require_outer = (
            lookup_type == "isnull" and condition.rhs is True and not current_negated
        )
        if (
            current_negated
            and (lookup_type != "isnull" or condition.rhs is False)
            and condition.rhs is not None
        ):
            require_outer = True
            if lookup_type != "isnull":
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
                    self.is_nullable(targets[0])
                    or self.alias_map[join_list[-1]].join_type == LOUTER
                ):
                    lookup_class = targets[0].get_lookup("isnull")
                    col = self._get_col(targets[0], join_info.targets[0], alias)
                    clause.add(lookup_class(col, False), AND)
                # If someval is a nullable column, someval IS NOT NULL is
                # added.
                if isinstance(value, Col) and self.is_nullable(value.target):
                    lookup_class = value.target.get_lookup("isnull")
                    clause.add(lookup_class(value, False), AND)
        return clause, used_joins if not require_outer else ()
```
### 49 - django/db/models/sql/query.py:

Start line: 843, End line: 869

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
### 57 - django/db/models/sql/query.py:

Start line: 2204, End line: 2235

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
        if allow_aliases and self.values_select:
            # If grouping by aliases is allowed assign selected value aliases
            # by moving them to annotations.
            group_by_annotations = {}
            values_select = {}
            for alias, expr in zip(self.values_select, self.select):
                if isinstance(expr, Col):
                    values_select[alias] = expr
                else:
                    group_by_annotations[alias] = expr
            self.annotations = {**group_by_annotations, **self.annotations}
            self.append_annotation_mask(group_by_annotations)
            self.select = tuple(values_select.values())
            self.values_select = tuple(values_select)
        group_by = list(self.select)
        for alias, annotation in self.annotation_select.items():
            if not (group_by_cols := annotation.get_group_by_cols()):
                continue
            if allow_aliases and not annotation.contains_aggregate:
                group_by.append(Ref(alias, annotation))
            else:
                group_by.extend(group_by_cols)
        self.group_by = tuple(group_by)
```
### 64 - django/db/models/sql/query.py:

Start line: 1621, End line: 1719

```python
class Query(BaseExpression):

    def names_to_path(self, names, opts, allow_many=True, fail_on_missing=False):
        # ... other code
        for pos, name in enumerate(names):
            cur_names_with_path = (name, [])
            if name == "pk":
                name = opts.pk.name

            field = None
            filtered_relation = None
            try:
                if opts is None:
                    raise FieldDoesNotExist
                field = opts.get_field(name)
            except FieldDoesNotExist:
                if name in self.annotation_select:
                    field = self.annotation_select[name].output_field
                elif name in self._filtered_relations and pos == 0:
                    filtered_relation = self._filtered_relations[name]
                    if LOOKUP_SEP in filtered_relation.relation_name:
                        parts = filtered_relation.relation_name.split(LOOKUP_SEP)
                        filtered_relation_path, field, _, _ = self.names_to_path(
                            parts,
                            opts,
                            allow_many,
                            fail_on_missing,
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
                    available = sorted(
                        [
                            *get_field_names_from_opts(opts),
                            *self.annotation_select,
                            *self._filtered_relations,
                        ]
                    )
                    raise FieldError(
                        "Cannot resolve keyword '%s' into field. "
                        "Choices are: %s" % (name, ", ".join(available))
                    )
                break
            # Check if we need any joins for concrete inheritance cases (the
            # field lives in parent, but we are currently in one of its
            # children)
            if opts is not None and model is not opts.model:
                path_to_parent = opts.get_path_to_parent(model)
                if path_to_parent:
                    path.extend(path_to_parent)
                    cur_names_with_path[1].extend(path_to_parent)
                    opts = path_to_parent[-1].to_opts
            if hasattr(field, "path_infos"):
                if filtered_relation:
                    pathinfos = field.get_path_info(filtered_relation)
                else:
                    pathinfos = field.path_infos
                if not allow_many:
                    for inner_pos, p in enumerate(pathinfos):
                        if p.m2m:
                            cur_names_with_path[1].extend(pathinfos[0 : inner_pos + 1])
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
                        " not permitted." % (names[pos + 1], name)
                    )
                break
        return path, final_field, targets, names[pos + 1 :]
```
### 69 - django/db/models/sql/query.py:

Start line: 2237, End line: 2284

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
                self.check_alias(name)
                entry = str(entry)
                entry_params = []
                pos = entry.find("%s")
                while pos != -1:
                    if pos == 0 or entry[pos - 1] != "%":
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
### 74 - django/db/models/sql/query.py:

Start line: 1574, End line: 1603

```python
class Query(BaseExpression):

    def add_filtered_relation(self, filtered_relation, alias):
        filtered_relation.alias = alias
        lookups = dict(get_children_from_q(filtered_relation.condition))
        relation_lookup_parts, relation_field_parts, _ = self.solve_lookup_type(
            filtered_relation.relation_name
        )
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
### 84 - django/db/models/sql/query.py:

Start line: 1, End line: 79

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
    Value,
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
from django.utils.regex_helper import _lazy_re_compile
from django.utils.tree import Node

__all__ = ["Query", "RawQuery"]

# Quotation marks ('"`[]), whitespace characters, semicolons, or inline
# SQL comments are forbidden in column aliases.
FORBIDDEN_ALIAS_PATTERN = _lazy_re_compile(r"['`\"\]\[;\s]|--|/\*|\*/")

# Inspired from
# https://www.postgresql.org/docs/current/sql-syntax-lexical.html#SQL-SYNTAX-IDENTIFIERS
EXPLAIN_OPTIONS_PATTERN = _lazy_re_compile(r"[\w\-]+")


def get_field_names_from_opts(opts):
    if opts is None:
        return set()
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
### 90 - django/db/models/sql/query.py:

Start line: 2152, End line: 2186

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
                if item == "?":
                    continue
                item = item.removeprefix("-")
                if item in self.annotations:
                    continue
                if self.extra and item in self.extra:
                    continue
                # names_to_path() validates the lookup. A descriptive
                # FieldError will be raise if it's not.
                self.names_to_path(item.split(LOOKUP_SEP), self.model._meta)
            elif not hasattr(item, "resolve_expression"):
                errors.append(item)
            if getattr(item, "contains_aggregate", False):
                raise FieldError(
                    "Using an aggregate in order_by() without also including "
                    "it in annotate() is not allowed: %s" % item
                )
        if errors:
            raise FieldError("Invalid order_by arguments: %s" % errors)
        if ordering:
            self.order_by += ordering
        else:
            self.default_ordering = False
```
### 94 - django/db/models/sql/query.py:

Start line: 2188, End line: 2202

```python
class Query(BaseExpression):

    def clear_ordering(self, force=False, clear_default=True):
        """
        Remove any ordering settings if the current query allows it without
        side effects, set 'force' to True to clear the ordering regardless.
        If 'clear_default' is True, there will be no ordering in the resulting
        query (not even the model's default).
        """
        if not force and (
            self.is_sliced or self.distinct_fields or self.select_for_update
        ):
            return
        self.order_by = ()
        self.extra_order_by = ()
        if clear_default:
            self.default_ordering = False
```
