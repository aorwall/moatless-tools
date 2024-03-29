# django__django-13371

| **django/django** | `3a9f192b131f7a9b0fe5783c684b23015fa67cc8` |
| ---- | ---- |
| **No of patches** | 2 |
| **All found context length** | 7896 |
| **Any found context length** | 263 |
| **Avg pos** | 28.0 |
| **Min pos** | 1 |
| **Max pos** | 27 |
| **Top file pos** | 1 |
| **Missing snippets** | 6 |
| **Missing patch files** | 1 |


## Expected patch

```diff
diff --git a/django/db/models/query.py b/django/db/models/query.py
--- a/django/db/models/query.py
+++ b/django/db/models/query.py
@@ -5,8 +5,6 @@
 import copy
 import operator
 import warnings
-from collections import namedtuple
-from functools import lru_cache
 from itertools import chain
 
 import django
@@ -23,7 +21,7 @@
 from django.db.models.functions import Cast, Trunc
 from django.db.models.query_utils import FilteredRelation, Q
 from django.db.models.sql.constants import CURSOR, GET_ITERATOR_CHUNK_SIZE
-from django.db.models.utils import resolve_callables
+from django.db.models.utils import create_namedtuple_class, resolve_callables
 from django.utils import timezone
 from django.utils.functional import cached_property, partition
 
@@ -148,13 +146,6 @@ class NamedValuesListIterable(ValuesListIterable):
     namedtuple for each row.
     """
 
-    @staticmethod
-    @lru_cache()
-    def create_namedtuple_class(*names):
-        # Cache namedtuple() with @lru_cache() since it's too slow to be
-        # called for every QuerySet evaluation.
-        return namedtuple('Row', names)
-
     def __iter__(self):
         queryset = self.queryset
         if queryset._fields:
@@ -162,7 +153,7 @@ def __iter__(self):
         else:
             query = queryset.query
             names = [*query.extra_select, *query.values_select, *query.annotation_select]
-        tuple_class = self.create_namedtuple_class(*names)
+        tuple_class = create_namedtuple_class(*names)
         new = tuple.__new__
         for row in super().__iter__():
             yield new(tuple_class, row)
diff --git a/django/db/models/utils.py b/django/db/models/utils.py
--- a/django/db/models/utils.py
+++ b/django/db/models/utils.py
@@ -1,3 +1,7 @@
+import functools
+from collections import namedtuple
+
+
 def make_model_tuple(model):
     """
     Take a model or a string of the form "app_label.ModelName" and return a
@@ -28,3 +32,17 @@ def resolve_callables(mapping):
     """
     for k, v in mapping.items():
         yield k, v() if callable(v) else v
+
+
+def unpickle_named_row(names, values):
+    return create_namedtuple_class(*names)(*values)
+
+
+@functools.lru_cache()
+def create_namedtuple_class(*names):
+    # Cache type() with @lru_cache() since it's too slow to be called for every
+    # QuerySet evaluation.
+    def __reduce__(self):
+        return unpickle_named_row, (names, tuple(self))
+
+    return type('Row', (namedtuple('Row', names),), {'__reduce__': __reduce__})

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/db/models/query.py | 8 | 9 | 27 | 1 | 7896
| django/db/models/query.py | 26 | 26 | 27 | 1 | 7896
| django/db/models/query.py | 151 | 157 | 1 | 1 | 263
| django/db/models/query.py | 165 | 165 | 1 | 1 | 263
| django/db/models/utils.py | 1 | 1 | - | - | -
| django/db/models/utils.py | 31 | 31 | - | - | -


## Problem Statement

```
django.db.models.query.Row is not pickleable.
Description
	 
		(last modified by Mariusz Felisiak)
	 
The new named parameter of QuerySet.values_list() was released In Django 2.0 (#15648).
But resulted namedtuple-s can't be pickled:
class ModelA(models.Model):
	value = models.CharField(max_length=12)
In [12]: row = ModelA.objects.values_list('id', 'value', named=True).first()
In [14]: type(row)																																																						 
Out[14]: django.db.models.query.Row
In [16]: pickle.dumps(row)																																																				 
PicklingError: Can't pickle <class 'django.db.models.query.Row'>: attribute lookup Row on django.db.models.query failed
In particular, as a result, such requests do not work with cacheops package.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 django/db/models/query.py** | 145 | 181| 263 | 263 | 17133 | 
| 2 | **1 django/db/models/query.py** | 245 | 272| 221 | 484 | 17133 | 
| 3 | **1 django/db/models/query.py** | 836 | 865| 248 | 732 | 17133 | 
| 4 | **1 django/db/models/query.py** | 94 | 112| 138 | 870 | 17133 | 
| 5 | **1 django/db/models/query.py** | 795 | 834| 322 | 1192 | 17133 | 
| 6 | **1 django/db/models/query.py** | 115 | 142| 233 | 1425 | 17133 | 
| 7 | **1 django/db/models/query.py** | 778 | 794| 157 | 1582 | 17133 | 
| 8 | **1 django/db/models/query.py** | 1475 | 1508| 297 | 1879 | 17133 | 
| 9 | **1 django/db/models/query.py** | 44 | 91| 457 | 2336 | 17133 | 
| 10 | **1 django/db/models/query.py** | 715 | 742| 235 | 2571 | 17133 | 
| 11 | **1 django/db/models/query.py** | 1088 | 1129| 323 | 2894 | 17133 | 
| 12 | 2 django/db/models/sql/query.py | 2180 | 2225| 371 | 3265 | 39623 | 
| 13 | **2 django/db/models/query.py** | 184 | 243| 469 | 3734 | 39623 | 
| 14 | **2 django/db/models/query.py** | 274 | 294| 180 | 3914 | 39623 | 
| 15 | **2 django/db/models/query.py** | 1510 | 1556| 336 | 4250 | 39623 | 
| 16 | 2 django/db/models/sql/query.py | 1473 | 1558| 801 | 5051 | 39623 | 
| 17 | 2 django/db/models/sql/query.py | 232 | 286| 400 | 5451 | 39623 | 
| 18 | 3 django/db/models/base.py | 507 | 552| 382 | 5833 | 56267 | 
| 19 | **3 django/db/models/query.py** | 1131 | 1167| 324 | 6157 | 56267 | 
| 20 | **3 django/db/models/query.py** | 1237 | 1260| 220 | 6377 | 56267 | 
| 21 | 3 django/db/models/sql/query.py | 118 | 133| 145 | 6522 | 56267 | 
| 22 | **3 django/db/models/query.py** | 1964 | 1987| 200 | 6722 | 56267 | 
| 23 | 4 django/db/models/sql/compiler.py | 1 | 19| 170 | 6892 | 70534 | 
| 24 | 5 django/db/models/sql/subqueries.py | 77 | 97| 202 | 7094 | 71747 | 
| 25 | **5 django/db/models/query.py** | 660 | 674| 132 | 7226 | 71747 | 
| 26 | **5 django/db/models/query.py** | 910 | 960| 371 | 7597 | 71747 | 
| **-> 27 <-** | **5 django/db/models/query.py** | 1 | 41| 299 | 7896 | 71747 | 
| 28 | 5 django/db/models/sql/query.py | 1 | 65| 465 | 8361 | 71747 | 
| 29 | **5 django/db/models/query.py** | 357 | 372| 146 | 8507 | 71747 | 
| 30 | 6 django/forms/models.py | 1338 | 1373| 288 | 8795 | 83521 | 
| 31 | 6 django/forms/models.py | 1180 | 1245| 543 | 9338 | 83521 | 
| 32 | **6 django/db/models/query.py** | 1028 | 1048| 155 | 9493 | 83521 | 
| 33 | **6 django/db/models/query.py** | 329 | 355| 222 | 9715 | 83521 | 
| 34 | **6 django/db/models/query.py** | 1073 | 1086| 115 | 9830 | 83521 | 
| 35 | 6 django/db/models/sql/query.py | 1702 | 1741| 439 | 10269 | 83521 | 
| 36 | 7 django/http/request.py | 532 | 568| 285 | 10554 | 88993 | 
| 37 | **7 django/db/models/query.py** | 867 | 881| 184 | 10738 | 88993 | 
| 38 | 7 django/db/models/sql/subqueries.py | 1 | 44| 320 | 11058 | 88993 | 
| 39 | 7 django/forms/models.py | 1295 | 1319| 220 | 11278 | 88993 | 
| 40 | 7 django/db/models/sql/query.py | 288 | 337| 444 | 11722 | 88993 | 
| 41 | 7 django/db/models/sql/query.py | 68 | 116| 361 | 12083 | 88993 | 
| 42 | **7 django/db/models/query.py** | 1012 | 1026| 145 | 12228 | 88993 | 
| 43 | **7 django/db/models/query.py** | 1284 | 1308| 210 | 12438 | 88993 | 
| 44 | 7 django/db/models/sql/subqueries.py | 137 | 163| 173 | 12611 | 88993 | 
| 45 | 8 django/db/models/query_utils.py | 25 | 54| 185 | 12796 | 91699 | 
| 46 | 8 django/db/models/base.py | 2038 | 2089| 351 | 13147 | 91699 | 
| 47 | 8 django/db/models/sql/query.py | 703 | 738| 389 | 13536 | 91699 | 
| 48 | 8 django/db/models/sql/query.py | 1847 | 1896| 330 | 13866 | 91699 | 
| 49 | **8 django/db/models/query.py** | 1559 | 1615| 481 | 14347 | 91699 | 
| 50 | 9 django/core/serializers/python.py | 1 | 60| 430 | 14777 | 92961 | 
| 51 | 10 django/db/models/fields/__init__.py | 570 | 588| 224 | 15001 | 110838 | 
| 52 | 10 django/db/models/sql/query.py | 1115 | 1147| 338 | 15339 | 110838 | 
| 53 | **10 django/db/models/query.py** | 744 | 777| 274 | 15613 | 110838 | 
| 54 | 10 django/db/models/sql/query.py | 1059 | 1084| 214 | 15827 | 110838 | 
| 55 | **10 django/db/models/query.py** | 1207 | 1235| 183 | 16010 | 110838 | 
| 56 | **10 django/db/models/query.py** | 296 | 327| 265 | 16275 | 110838 | 
| 57 | 10 django/db/models/sql/query.py | 2134 | 2151| 156 | 16431 | 110838 | 
| 58 | 10 django/db/models/sql/subqueries.py | 47 | 75| 210 | 16641 | 110838 | 
| 59 | 10 django/db/models/sql/query.py | 2227 | 2259| 228 | 16869 | 110838 | 
| 60 | **10 django/db/models/query.py** | 676 | 713| 378 | 17247 | 110838 | 
| 61 | 10 django/http/request.py | 432 | 530| 792 | 18039 | 110838 | 
| 62 | 11 django/db/backends/sqlite3/base.py | 399 | 419| 183 | 18222 | 116774 | 
| 63 | 11 django/db/models/sql/query.py | 1636 | 1660| 227 | 18449 | 116774 | 
| 64 | 11 django/db/models/sql/query.py | 339 | 362| 179 | 18628 | 116774 | 
| 65 | 11 django/db/models/sql/query.py | 2086 | 2108| 249 | 18877 | 116774 | 
| 66 | 12 django/db/models/__init__.py | 1 | 53| 619 | 19496 | 117393 | 
| 67 | 12 django/db/models/sql/query.py | 1987 | 2036| 420 | 19916 | 117393 | 
| 68 | **12 django/db/models/query.py** | 1050 | 1071| 214 | 20130 | 117393 | 
| 69 | 12 django/db/models/base.py | 554 | 571| 142 | 20272 | 117393 | 
| 70 | 12 django/db/models/sql/subqueries.py | 111 | 134| 192 | 20464 | 117393 | 
| 71 | **12 django/db/models/query.py** | 1435 | 1473| 308 | 20772 | 117393 | 
| 72 | 12 django/db/models/sql/query.py | 1428 | 1455| 283 | 21055 | 117393 | 
| 73 | 12 django/db/models/sql/query.py | 1898 | 1939| 355 | 21410 | 117393 | 
| 74 | 12 django/db/models/sql/query.py | 1086 | 1113| 285 | 21695 | 117393 | 
| 75 | **12 django/db/models/query.py** | 1310 | 1328| 186 | 21881 | 117393 | 
| 76 | 12 django/db/models/sql/query.py | 1816 | 1845| 259 | 22140 | 117393 | 
| 77 | 13 django/db/backends/mysql/operations.py | 143 | 191| 414 | 22554 | 121065 | 
| 78 | 13 django/db/models/sql/query.py | 416 | 509| 917 | 23471 | 121065 | 
| 79 | 13 django/db/models/sql/query.py | 136 | 230| 833 | 24304 | 121065 | 
| 80 | 13 django/db/models/sql/query.py | 654 | 701| 511 | 24815 | 121065 | 
| 81 | 13 django/db/models/sql/compiler.py | 1132 | 1197| 527 | 25342 | 121065 | 
| 82 | 13 django/db/models/sql/compiler.py | 63 | 147| 881 | 26223 | 121065 | 
| 83 | 13 django/db/models/base.py | 961 | 975| 212 | 26435 | 121065 | 
| 84 | 14 django/db/backends/sqlite3/operations.py | 162 | 187| 190 | 26625 | 124096 | 
| 85 | 14 django/db/models/sql/compiler.py | 1583 | 1614| 244 | 26869 | 124096 | 
| 86 | 14 django/forms/models.py | 629 | 646| 167 | 27036 | 124096 | 
| 87 | 15 django/apps/config.py | 58 | 83| 270 | 27306 | 126543 | 
| 88 | 16 django/core/serializers/__init__.py | 86 | 141| 369 | 27675 | 128300 | 
| 89 | **16 django/db/models/query.py** | 1169 | 1184| 149 | 27824 | 128300 | 
| 90 | 16 django/forms/models.py | 1135 | 1177| 310 | 28134 | 128300 | 
| 91 | 17 django/core/cache/backends/db.py | 230 | 253| 259 | 28393 | 130422 | 
| 92 | 17 django/db/models/sql/compiler.py | 1332 | 1391| 617 | 29010 | 130422 | 
| 93 | 17 django/core/cache/backends/db.py | 40 | 95| 431 | 29441 | 130422 | 
| 94 | **17 django/db/models/query.py** | 1186 | 1205| 209 | 29650 | 130422 | 
| 95 | **17 django/db/models/query.py** | 962 | 977| 124 | 29774 | 130422 | 
| 96 | 17 django/db/models/query_utils.py | 1 | 22| 178 | 29952 | 130422 | 
| 97 | **17 django/db/models/query.py** | 979 | 1010| 341 | 30293 | 130422 | 
| 98 | 17 django/db/models/sql/query.py | 2110 | 2132| 229 | 30522 | 130422 | 
| 99 | 17 django/db/models/sql/query.py | 364 | 414| 494 | 31016 | 130422 | 
| 100 | 18 django/contrib/gis/db/models/__init__.py | 1 | 19| 204 | 31220 | 130626 | 
| 101 | 18 django/db/models/sql/compiler.py | 22 | 47| 257 | 31477 | 130626 | 
| 102 | 19 django/views/generic/detail.py | 58 | 76| 154 | 31631 | 131941 | 
| 103 | 20 django/db/models/expressions.py | 779 | 793| 120 | 31751 | 142702 | 
| 104 | 21 django/core/exceptions.py | 107 | 219| 770 | 32521 | 143909 | 
| 105 | 21 django/db/models/sql/subqueries.py | 99 | 109| 129 | 32650 | 143909 | 
| 106 | **21 django/db/models/query.py** | 1425 | 1433| 136 | 32786 | 143909 | 
| 107 | 21 django/db/models/sql/query.py | 2368 | 2384| 119 | 32905 | 143909 | 
| 108 | 21 django/db/models/sql/compiler.py | 271 | 358| 712 | 33617 | 143909 | 
| 109 | 22 django/db/backends/postgresql/operations.py | 158 | 185| 311 | 33928 | 146425 | 
| 110 | 22 django/db/models/sql/query.py | 2038 | 2084| 370 | 34298 | 146425 | 
| 111 | 22 django/db/models/sql/compiler.py | 1297 | 1330| 344 | 34642 | 146425 | 
| 112 | 22 django/db/models/sql/compiler.py | 1259 | 1295| 341 | 34983 | 146425 | 
| 113 | **22 django/db/models/query.py** | 883 | 908| 264 | 35247 | 146425 | 
| 114 | 22 django/core/cache/backends/db.py | 199 | 228| 285 | 35532 | 146425 | 
| 115 | 23 django/db/models/manager.py | 168 | 205| 211 | 35743 | 147878 | 
| 116 | 24 django/db/backends/sqlite3/features.py | 1 | 80| 725 | 36468 | 148603 | 
| 117 | 25 django/core/serializers/pyyaml.py | 41 | 64| 244 | 36712 | 149246 | 
| 118 | 25 django/db/models/expressions.py | 795 | 821| 196 | 36908 | 149246 | 
| 119 | 25 django/core/cache/backends/db.py | 255 | 283| 324 | 37232 | 149246 | 
| 120 | 26 django/db/models/lookups.py | 1 | 42| 319 | 37551 | 154199 | 
| 121 | 27 django/db/backends/sqlite3/schema.py | 39 | 65| 243 | 37794 | 158334 | 
| 122 | 28 django/db/models/aggregates.py | 45 | 68| 294 | 38088 | 159635 | 
| 123 | 28 django/db/models/sql/compiler.py | 1393 | 1411| 203 | 38291 | 159635 | 
| 124 | 28 django/db/models/expressions.py | 746 | 776| 233 | 38524 | 159635 | 
| 125 | 29 django/db/backends/utils.py | 92 | 129| 297 | 38821 | 161501 | 
| 126 | 29 django/db/models/sql/compiler.py | 1038 | 1078| 337 | 39158 | 161501 | 
| 127 | **29 django/db/models/query.py** | 1330 | 1339| 114 | 39272 | 161501 | 
| 128 | 29 django/db/models/base.py | 573 | 592| 170 | 39442 | 161501 | 
| 129 | 30 django/contrib/postgres/search.py | 160 | 195| 313 | 39755 | 163723 | 
| 130 | 31 django/views/generic/list.py | 50 | 75| 244 | 39999 | 165295 | 
| 131 | 31 django/db/backends/sqlite3/operations.py | 144 | 160| 184 | 40183 | 165295 | 
| 132 | **31 django/db/models/query.py** | 415 | 455| 343 | 40526 | 165295 | 
| 133 | 32 django/db/models/deletion.py | 346 | 359| 116 | 40642 | 169121 | 
| 134 | 32 django/db/models/base.py | 1 | 50| 328 | 40970 | 169121 | 
| 135 | 32 django/db/models/lookups.py | 358 | 390| 294 | 41264 | 169121 | 
| 136 | 32 django/db/models/base.py | 977 | 990| 180 | 41444 | 169121 | 
| 137 | 32 django/db/models/sql/query.py | 2335 | 2351| 177 | 41621 | 169121 | 
| 138 | 32 django/db/models/sql/query.py | 1294 | 1359| 772 | 42393 | 169121 | 
| 139 | 32 django/db/models/sql/query.py | 899 | 921| 248 | 42641 | 169121 | 
| 140 | 32 django/db/models/aggregates.py | 70 | 96| 266 | 42907 | 169121 | 
| 141 | 32 django/db/models/lookups.py | 247 | 258| 153 | 43060 | 169121 | 
| 142 | 32 django/core/cache/backends/db.py | 97 | 110| 234 | 43294 | 169121 | 
| 143 | 33 django/contrib/postgres/aggregates/mixins.py | 22 | 34| 128 | 43422 | 169563 | 
| 144 | 34 django/db/backends/mysql/features.py | 1 | 112| 847 | 44269 | 170965 | 
| 145 | **34 django/db/models/query.py** | 1341 | 1389| 405 | 44674 | 170965 | 
| 146 | 34 django/db/models/query_utils.py | 110 | 124| 157 | 44831 | 170965 | 
| 147 | **34 django/db/models/query.py** | 1647 | 1753| 1063 | 45894 | 170965 | 
| 148 | 34 django/db/models/expressions.py | 1 | 30| 204 | 46098 | 170965 | 
| 149 | 35 django/utils/datastructures.py | 151 | 190| 300 | 46398 | 173222 | 
| 150 | 36 django/contrib/admin/options.py | 1670 | 1684| 132 | 46530 | 191809 | 
| 151 | 37 django/db/backends/sqlite3/introspection.py | 23 | 54| 296 | 46826 | 195658 | 
| 152 | 37 django/db/models/sql/query.py | 2153 | 2178| 214 | 47040 | 195658 | 
| 153 | 37 django/forms/models.py | 1262 | 1292| 242 | 47282 | 195658 | 
| 154 | 38 django/db/backends/mysql/schema.py | 50 | 86| 349 | 47631 | 197154 | 
| 155 | 38 django/db/models/sql/compiler.py | 1199 | 1220| 223 | 47854 | 197154 | 
| 156 | 38 django/db/backends/mysql/operations.py | 220 | 277| 431 | 48285 | 197154 | 
| 157 | 38 django/core/cache/backends/db.py | 1 | 37| 229 | 48514 | 197154 | 
| 158 | 38 django/db/models/expressions.py | 1301 | 1333| 246 | 48760 | 197154 | 
| 159 | 39 django/db/backends/oracle/schema.py | 1 | 39| 405 | 49165 | 199042 | 
| 160 | 39 django/db/models/sql/query.py | 628 | 652| 269 | 49434 | 199042 | 
| 161 | **39 django/db/models/query.py** | 374 | 413| 325 | 49759 | 199042 | 
| 162 | 39 django/db/models/sql/compiler.py | 1223 | 1257| 332 | 50091 | 199042 | 


## Missing Patch Files

 * 1: django/db/models/query.py
 * 2: django/db/models/utils.py

### Hint

```
In addition to the data contained in the instances, pickle also stores a string reference to the original class. This means that Row tuple class should be in query module globals so pickle can find the reference. But besides that, it should have different class names for each model and for each list of values.
​PR
You shouldn't mark your own PRs as ready for checkin.
```

## Patch

```diff
diff --git a/django/db/models/query.py b/django/db/models/query.py
--- a/django/db/models/query.py
+++ b/django/db/models/query.py
@@ -5,8 +5,6 @@
 import copy
 import operator
 import warnings
-from collections import namedtuple
-from functools import lru_cache
 from itertools import chain
 
 import django
@@ -23,7 +21,7 @@
 from django.db.models.functions import Cast, Trunc
 from django.db.models.query_utils import FilteredRelation, Q
 from django.db.models.sql.constants import CURSOR, GET_ITERATOR_CHUNK_SIZE
-from django.db.models.utils import resolve_callables
+from django.db.models.utils import create_namedtuple_class, resolve_callables
 from django.utils import timezone
 from django.utils.functional import cached_property, partition
 
@@ -148,13 +146,6 @@ class NamedValuesListIterable(ValuesListIterable):
     namedtuple for each row.
     """
 
-    @staticmethod
-    @lru_cache()
-    def create_namedtuple_class(*names):
-        # Cache namedtuple() with @lru_cache() since it's too slow to be
-        # called for every QuerySet evaluation.
-        return namedtuple('Row', names)
-
     def __iter__(self):
         queryset = self.queryset
         if queryset._fields:
@@ -162,7 +153,7 @@ def __iter__(self):
         else:
             query = queryset.query
             names = [*query.extra_select, *query.values_select, *query.annotation_select]
-        tuple_class = self.create_namedtuple_class(*names)
+        tuple_class = create_namedtuple_class(*names)
         new = tuple.__new__
         for row in super().__iter__():
             yield new(tuple_class, row)
diff --git a/django/db/models/utils.py b/django/db/models/utils.py
--- a/django/db/models/utils.py
+++ b/django/db/models/utils.py
@@ -1,3 +1,7 @@
+import functools
+from collections import namedtuple
+
+
 def make_model_tuple(model):
     """
     Take a model or a string of the form "app_label.ModelName" and return a
@@ -28,3 +32,17 @@ def resolve_callables(mapping):
     """
     for k, v in mapping.items():
         yield k, v() if callable(v) else v
+
+
+def unpickle_named_row(names, values):
+    return create_namedtuple_class(*names)(*values)
+
+
+@functools.lru_cache()
+def create_namedtuple_class(*names):
+    # Cache type() with @lru_cache() since it's too slow to be called for every
+    # QuerySet evaluation.
+    def __reduce__(self):
+        return unpickle_named_row, (names, tuple(self))
+
+    return type('Row', (namedtuple('Row', names),), {'__reduce__': __reduce__})

```

## Test Patch

```diff
diff --git a/tests/queries/tests.py b/tests/queries/tests.py
--- a/tests/queries/tests.py
+++ b/tests/queries/tests.py
@@ -2408,6 +2408,11 @@ def test_named_values_list_expression(self):
         values = qs.first()
         self.assertEqual(values._fields, ('combinedexpression2', 'combinedexpression1'))
 
+    def test_named_values_pickle(self):
+        value = Number.objects.values_list('num', 'other_num', named=True).get()
+        self.assertEqual(value, (72, None))
+        self.assertEqual(pickle.loads(pickle.dumps(value)), value)
+
 
 class QuerySetSupportsPythonIdioms(TestCase):
 

```


## Code snippets

### 1 - django/db/models/query.py:

Start line: 145, End line: 181

```python
class NamedValuesListIterable(ValuesListIterable):
    """
    Iterable returned by QuerySet.values_list(named=True) that yields a
    namedtuple for each row.
    """

    @staticmethod
    @lru_cache()
    def create_namedtuple_class(*names):
        # Cache namedtuple() with @lru_cache() since it's too slow to be
        # called for every QuerySet evaluation.
        return namedtuple('Row', names)

    def __iter__(self):
        queryset = self.queryset
        if queryset._fields:
            names = queryset._fields
        else:
            query = queryset.query
            names = [*query.extra_select, *query.values_select, *query.annotation_select]
        tuple_class = self.create_namedtuple_class(*names)
        new = tuple.__new__
        for row in super().__iter__():
            yield new(tuple_class, row)


class FlatValuesListIterable(BaseIterable):
    """
    Iterable returned by QuerySet.values_list(flat=True) that yields single
    values.
    """

    def __iter__(self):
        queryset = self.queryset
        compiler = queryset.query.get_compiler(queryset.db)
        for row in compiler.results_iter(chunked_fetch=self.chunked_fetch, chunk_size=self.chunk_size):
            yield row[0]
```
### 2 - django/db/models/query.py:

Start line: 245, End line: 272

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
        data = list(self[:REPR_OUTPUT_SIZE + 1])
        if len(data) > REPR_OUTPUT_SIZE:
            data[-1] = "...(remaining elements truncated)..."
        return '<%s %r>' % (self.__class__.__name__, data)

    def __len__(self):
        self._fetch_all()
        return len(self._result_cache)
```
### 3 - django/db/models/query.py:

Start line: 836, End line: 865

```python
class QuerySet:

    def values_list(self, *fields, flat=False, named=False):
        if flat and named:
            raise TypeError("'flat' and 'named' can't be used together.")
        if flat and len(fields) > 1:
            raise TypeError("'flat' is not valid when values_list is called with more than one field.")

        field_names = {f for f in fields if not hasattr(f, 'resolve_expression')}
        _fields = []
        expressions = {}
        counter = 1
        for field in fields:
            if hasattr(field, 'resolve_expression'):
                field_id_prefix = getattr(field, 'default_alias', field.__class__.__name__.lower())
                while True:
                    field_id = field_id_prefix + str(counter)
                    counter += 1
                    if field_id not in field_names:
                        break
                expressions[field_id] = field
                _fields.append(field_id)
            else:
                _fields.append(field)

        clone = self._values(*_fields, **expressions)
        clone._iterable_class = (
            NamedValuesListIterable if named
            else FlatValuesListIterable if flat
            else ValuesListIterable
        )
        return clone
```
### 4 - django/db/models/query.py:

Start line: 94, End line: 112

```python
class ValuesIterable(BaseIterable):
    """
    Iterable returned by QuerySet.values() that yields a dict for each row.
    """

    def __iter__(self):
        queryset = self.queryset
        query = queryset.query
        compiler = query.get_compiler(queryset.db)

        # extra(select=...) cols are always at the start of the row.
        names = [
            *query.extra_select,
            *query.values_select,
            *query.annotation_select,
        ]
        indexes = range(len(names))
        for row in compiler.results_iter(chunked_fetch=self.chunked_fetch, chunk_size=self.chunk_size):
            yield {names[i]: row[i] for i in indexes}
```
### 5 - django/db/models/query.py:

Start line: 795, End line: 834

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
### 6 - django/db/models/query.py:

Start line: 115, End line: 142

```python
class ValuesListIterable(BaseIterable):
    """
    Iterable returned by QuerySet.values_list(flat=False) that yields a tuple
    for each row.
    """

    def __iter__(self):
        queryset = self.queryset
        query = queryset.query
        compiler = query.get_compiler(queryset.db)

        if queryset._fields:
            # extra(select=...) cols are always at the start of the row.
            names = [
                *query.extra_select,
                *query.values_select,
                *query.annotation_select,
            ]
            fields = [*queryset._fields, *(f for f in query.annotation_select if f not in queryset._fields)]
            if fields != names:
                # Reorder according to fields.
                index_map = {name: idx for idx, name in enumerate(names)}
                rowfactory = operator.itemgetter(*[index_map[f] for f in fields])
                return map(
                    rowfactory,
                    compiler.results_iter(chunked_fetch=self.chunked_fetch, chunk_size=self.chunk_size)
                )
        return compiler.results_iter(tuple_expected=True, chunked_fetch=self.chunked_fetch, chunk_size=self.chunk_size)
```
### 7 - django/db/models/query.py:

Start line: 778, End line: 794

```python
class QuerySet:
    update.alters_data = True

    def _update(self, values):
        """
        A version of update() that accepts field objects instead of field names.
        Used primarily for model saving and not intended for use by general
        code (it requires too much poking around at model internals to be
        useful at that level).
        """
        assert not self.query.is_sliced, \
            "Cannot update a query once a slice has been taken."
        query = self.query.chain(sql.UpdateQuery)
        query.add_update_fields(values)
        # Clear any annotations so that they won't be present in subqueries.
        query.annotations = {}
        self._result_cache = None
        return query.get_compiler(self.db).execute_sql(CURSOR)
```
### 8 - django/db/models/query.py:

Start line: 1475, End line: 1508

```python
class RawQuerySet:

    def iterator(self):
        # Cache some things for performance reasons outside the loop.
        db = self.db
        compiler = connections[db].ops.compiler('SQLCompiler')(
            self.query, connections[db], db
        )

        query = iter(self.query)

        try:
            model_init_names, model_init_pos, annotation_fields = self.resolve_model_init_order()
            if self.model._meta.pk.attname not in model_init_names:
                raise exceptions.FieldDoesNotExist(
                    'Raw query must include the primary key'
                )
            model_cls = self.model
            fields = [self.model_fields.get(c) for c in self.columns]
            converters = compiler.get_converters([
                f.get_col(f.model._meta.db_table) if f else None for f in fields
            ])
            if converters:
                query = compiler.apply_converters(query, converters)
            for values in query:
                # Associate fields to values
                model_init_values = [values[pos] for pos in model_init_pos]
                instance = model_cls.from_db(db, model_init_names, model_init_values)
                if annotation_fields:
                    for column, pos in annotation_fields:
                        setattr(instance, column, values[pos])
                yield instance
        finally:
            # Done iterating the Query. If it has its own cursor, close it.
            if hasattr(self.query, 'cursor') and self.query.cursor:
                self.query.cursor.close()
```
### 9 - django/db/models/query.py:

Start line: 44, End line: 91

```python
class ModelIterable(BaseIterable):
    """Iterable that yields a model instance for each row."""

    def __iter__(self):
        queryset = self.queryset
        db = queryset.db
        compiler = queryset.query.get_compiler(using=db)
        # Execute the query. This will also fill compiler.select, klass_info,
        # and annotations.
        results = compiler.execute_sql(chunked_fetch=self.chunked_fetch, chunk_size=self.chunk_size)
        select, klass_info, annotation_col_map = (compiler.select, compiler.klass_info,
                                                  compiler.annotation_col_map)
        model_cls = klass_info['model']
        select_fields = klass_info['select_fields']
        model_fields_start, model_fields_end = select_fields[0], select_fields[-1] + 1
        init_list = [f[0].target.attname
                     for f in select[model_fields_start:model_fields_end]]
        related_populators = get_related_populators(klass_info, select, db)
        known_related_objects = [
            (field, related_objs, operator.attrgetter(*[
                field.attname
                if from_field == 'self' else
                queryset.model._meta.get_field(from_field).attname
                for from_field in field.from_fields
            ])) for field, related_objs in queryset._known_related_objects.items()
        ]
        for row in compiler.results_iter(results):
            obj = model_cls.from_db(db, init_list, row[model_fields_start:model_fields_end])
            for rel_populator in related_populators:
                rel_populator.populate(row, obj)
            if annotation_col_map:
                for attr_name, col_pos in annotation_col_map.items():
                    setattr(obj, attr_name, row[col_pos])

            # Add the known related objects to the model.
            for field, rel_objs, rel_getter in known_related_objects:
                # Avoid overwriting objects loaded by, e.g., select_related().
                if field.is_cached(obj):
                    continue
                rel_obj_id = rel_getter(obj)
                try:
                    rel_obj = rel_objs[rel_obj_id]
                except KeyError:
                    pass  # May happen in qs1 | qs2 scenarios.
                else:
                    setattr(obj, field.name, rel_obj)

            yield obj
```
### 10 - django/db/models/query.py:

Start line: 715, End line: 742

```python
class QuerySet:

    def delete(self):
        """Delete the records in the current QuerySet."""
        self._not_support_combined_queries('delete')
        assert not self.query.is_sliced, \
            "Cannot use 'limit' or 'offset' with delete."

        if self._fields is not None:
            raise TypeError("Cannot call delete() after .values() or .values_list()")

        del_query = self._chain()

        # The delete is actually 2 queries - one to find related objects,
        # and one to delete. Make sure that the discovery of related
        # objects is performed on the same database as the deletion.
        del_query._for_write = True

        # Disable non-supported fields.
        del_query.query.select_for_update = False
        del_query.query.select_related = False
        del_query.query.clear_ordering(force_empty=True)

        collector = Collector(using=del_query.db)
        collector.collect(del_query)
        deleted, _rows_count = collector.delete()

        # Clear the result cache, in case this QuerySet gets reused.
        self._result_cache = None
        return deleted, _rows_count
```
### 11 - django/db/models/query.py:

Start line: 1088, End line: 1129

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
### 13 - django/db/models/query.py:

Start line: 184, End line: 243

```python
class QuerySet:
    """Represent a lazy database lookup for a set of objects."""

    def __init__(self, model=None, query=None, using=None, hints=None):
        self.model = model
        self._db = using
        self._hints = hints or {}
        self._query = query or sql.Query(self.model)
        self._result_cache = None
        self._sticky_filter = False
        self._for_write = False
        self._prefetch_related_lookups = ()
        self._prefetch_done = False
        self._known_related_objects = {}  # {rel_field: {pk: rel_obj}}
        self._iterable_class = ModelIterable
        self._fields = None
        self._defer_next_filter = False
        self._deferred_filter = None

    @property
    def query(self):
        if self._deferred_filter:
            negate, args, kwargs = self._deferred_filter
            self._filter_or_exclude_inplace(negate, args, kwargs)
            self._deferred_filter = None
        return self._query

    @query.setter
    def query(self, value):
        if value.values_select:
            self._iterable_class = ValuesIterable
        self._query = value

    def as_manager(cls):
        # Address the circular dependency between `Queryset` and `Manager`.
        from django.db.models.manager import Manager
        manager = Manager.from_queryset(cls)()
        manager._built_with_as_manager = True
        return manager
    as_manager.queryset_only = True
    as_manager = classmethod(as_manager)

    ########################
    # PYTHON MAGIC METHODS #
    ########################

    def __deepcopy__(self, memo):
        """Don't populate the QuerySet's cache."""
        obj = self.__class__()
        for k, v in self.__dict__.items():
            if k == '_result_cache':
                obj.__dict__[k] = None
            else:
                obj.__dict__[k] = copy.deepcopy(v, memo)
        return obj

    def __getstate__(self):
        # Force the cache to be fully populated.
        self._fetch_all()
        return {**self.__dict__, DJANGO_VERSION_PICKLE_KEY: django.__version__}
```
### 14 - django/db/models/query.py:

Start line: 274, End line: 294

```python
class QuerySet:

    def __iter__(self):
        """
        The queryset iterator protocol uses three nested iterators in the
        default case:
            1. sql.compiler.execute_sql()
               - Returns 100 rows at time (constants.GET_ITERATOR_CHUNK_SIZE)
                 using cursor.fetchmany(). This part is responsible for
                 doing some column masking, and returning the rows in chunks.
            2. sql.compiler.results_iter()
               - Returns one row at time. At this point the rows are still just
                 tuples. In some cases the return values are converted to
                 Python values at this location.
            3. self.iterator()
               - Responsible for turning the rows into model objects.
        """
        self._fetch_all()
        return iter(self._result_cache)

    def __bool__(self):
        self._fetch_all()
        return bool(self._result_cache)
```
### 15 - django/db/models/query.py:

Start line: 1510, End line: 1556

```python
class RawQuerySet:

    def __repr__(self):
        return "<%s: %s>" % (self.__class__.__name__, self.query)

    def __getitem__(self, k):
        return list(self)[k]

    @property
    def db(self):
        """Return the database used if this query is executed now."""
        return self._db or router.db_for_read(self.model, **self._hints)

    def using(self, alias):
        """Select the database this RawQuerySet should execute against."""
        return RawQuerySet(
            self.raw_query, model=self.model,
            query=self.query.chain(using=alias),
            params=self.params, translations=self.translations,
            using=alias,
        )

    @cached_property
    def columns(self):
        """
        A list of model field names in the order they'll appear in the
        query results.
        """
        columns = self.query.get_columns()
        # Adjust any column names which don't match field names
        for (query_name, model_name) in self.translations.items():
            # Ignore translations for nonexistent column names
            try:
                index = columns.index(query_name)
            except ValueError:
                pass
            else:
                columns[index] = model_name
        return columns

    @cached_property
    def model_fields(self):
        """A dict mapping column names to model field names."""
        converter = connections[self.db].introspection.identifier_converter
        model_fields = {}
        for field in self.model._meta.fields:
            name, column = field.get_attname_column()
            model_fields[converter(column)] = field
        return model_fields
```
### 19 - django/db/models/query.py:

Start line: 1131, End line: 1167

```python
class QuerySet:

    def order_by(self, *field_names):
        """Return a new QuerySet instance with the ordering changed."""
        assert not self.query.is_sliced, \
            "Cannot reorder a query once a slice has been taken."
        obj = self._chain()
        obj.query.clear_ordering(force_empty=False)
        obj.query.add_ordering(*field_names)
        return obj

    def distinct(self, *field_names):
        """
        Return a new QuerySet instance that will select only distinct results.
        """
        self._not_support_combined_queries('distinct')
        assert not self.query.is_sliced, \
            "Cannot create distinct fields once a slice has been taken."
        obj = self._chain()
        obj.query.add_distinct_fields(*field_names)
        return obj

    def extra(self, select=None, where=None, params=None, tables=None,
              order_by=None, select_params=None):
        """Add extra SQL fragments to the query."""
        self._not_support_combined_queries('extra')
        assert not self.query.is_sliced, \
            "Cannot change a query once a slice has been taken"
        clone = self._chain()
        clone.query.add_extra(select, select_params, where, params, tables, order_by)
        return clone

    def reverse(self):
        """Reverse the ordering of the QuerySet."""
        if self.query.is_sliced:
            raise TypeError('Cannot reverse a query once a slice has been taken.')
        clone = self._chain()
        clone.query.standard_ordering = not clone.query.standard_ordering
        return clone
```
### 20 - django/db/models/query.py:

Start line: 1237, End line: 1260

```python
class QuerySet:

    @property
    def db(self):
        """Return the database used if this query is executed now."""
        if self._for_write:
            return self._db or router.db_for_write(self.model, **self._hints)
        return self._db or router.db_for_read(self.model, **self._hints)

    ###################
    # PRIVATE METHODS #
    ###################

    def _insert(self, objs, fields, returning_fields=None, raw=False, using=None, ignore_conflicts=False):
        """
        Insert a new record for the given model. This provides an interface to
        the InsertQuery class and is how Model.save() is implemented.
        """
        self._for_write = True
        if using is None:
            using = self.db
        query = sql.InsertQuery(self.model, ignore_conflicts=ignore_conflicts)
        query.insert_values(fields, objs, raw=raw)
        return query.get_compiler(using=using).execute_sql(returning_fields)
    _insert.alters_data = True
    _insert.queryset_only = False
```
### 22 - django/db/models/query.py:

Start line: 1964, End line: 1987

```python
class RelatedPopulator:

    def populate(self, row, from_obj):
        if self.reorder_for_init:
            obj_data = self.reorder_for_init(row)
        else:
            obj_data = row[self.cols_start:self.cols_end]
        if obj_data[self.pk_idx] is None:
            obj = None
        else:
            obj = self.model_cls.from_db(self.db, self.init_list, obj_data)
            for rel_iter in self.related_populators:
                rel_iter.populate(row, obj)
        self.local_setter(from_obj, obj)
        if obj is not None:
            self.remote_setter(obj, from_obj)


def get_related_populators(klass_info, select, db):
    iterators = []
    related_klass_infos = klass_info.get('related_klass_infos', [])
    for rel_klass_info in related_klass_infos:
        rel_cls = RelatedPopulator(rel_klass_info, select, db)
        iterators.append(rel_cls)
    return iterators
```
### 25 - django/db/models/query.py:

Start line: 660, End line: 674

```python
class QuerySet:

    def earliest(self, *fields):
        return self._earliest(*fields)

    def latest(self, *fields):
        return self.reverse()._earliest(*fields)

    def first(self):
        """Return the first object of a query or None if no match is found."""
        for obj in (self if self.ordered else self.order_by('pk'))[:1]:
            return obj

    def last(self):
        """Return the last object of a query or None if no match is found."""
        for obj in (self.reverse() if self.ordered else self.order_by('-pk'))[:1]:
            return obj
```
### 26 - django/db/models/query.py:

Start line: 910, End line: 960

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
        if args or kwargs:
            assert not self.query.is_sliced, \
                "Cannot filter a query once a slice has been taken."

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
### 27 - django/db/models/query.py:

Start line: 1, End line: 41

```python
"""
The main QuerySet implementation. This provides the public API for the ORM.
"""

import copy
import operator
import warnings
from collections import namedtuple
from functools import lru_cache
from itertools import chain

import django
from django.conf import settings
from django.core import exceptions
from django.db import (
    DJANGO_VERSION_PICKLE_KEY, IntegrityError, NotSupportedError, connections,
    router, transaction,
)
from django.db.models import AutoField, DateField, DateTimeField, sql
from django.db.models.constants import LOOKUP_SEP
from django.db.models.deletion import Collector
from django.db.models.expressions import Case, Expression, F, Value, When
from django.db.models.functions import Cast, Trunc
from django.db.models.query_utils import FilteredRelation, Q
from django.db.models.sql.constants import CURSOR, GET_ITERATOR_CHUNK_SIZE
from django.db.models.utils import resolve_callables
from django.utils import timezone
from django.utils.functional import cached_property, partition

# The maximum number of results to fetch in a get() query.
MAX_GET_RESULTS = 21

# The maximum number of items to display in a QuerySet.__repr__
REPR_OUTPUT_SIZE = 20


class BaseIterable:
    def __init__(self, queryset, chunked_fetch=False, chunk_size=GET_ITERATOR_CHUNK_SIZE):
        self.queryset = queryset
        self.chunked_fetch = chunked_fetch
        self.chunk_size = chunk_size
```
### 29 - django/db/models/query.py:

Start line: 357, End line: 372

```python
class QuerySet:

    ####################################
    # METHODS THAT DO DATABASE QUERIES #
    ####################################

    def _iterator(self, use_chunked_fetch, chunk_size):
        yield from self._iterable_class(self, chunked_fetch=use_chunked_fetch, chunk_size=chunk_size)

    def iterator(self, chunk_size=2000):
        """
        An iterator over the results from applying this QuerySet to the
        database.
        """
        if chunk_size <= 0:
            raise ValueError('Chunk size must be strictly positive.')
        use_chunked_fetch = not connections[self.db].settings_dict.get('DISABLE_SERVER_SIDE_CURSORS')
        return self._iterator(use_chunked_fetch, chunk_size)
```
### 32 - django/db/models/query.py:

Start line: 1028, End line: 1048

```python
class QuerySet:

    def select_related(self, *fields):
        """
        Return a new QuerySet instance that will select related objects.

        If fields are specified, they must be ForeignKey fields and only those
        related objects are included in the selection.

        If select_related(None) is called, clear the list.
        """
        self._not_support_combined_queries('select_related')
        if self._fields is not None:
            raise TypeError("Cannot call select_related() after .values() or .values_list()")

        obj = self._chain()
        if fields == (None,):
            obj.query.select_related = False
        elif fields:
            obj.query.add_select_related(fields)
        else:
            obj.query.select_related = True
        return obj
```
### 33 - django/db/models/query.py:

Start line: 329, End line: 355

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
### 34 - django/db/models/query.py:

Start line: 1073, End line: 1086

```python
class QuerySet:

    def annotate(self, *args, **kwargs):
        """
        Return a query set in which the returned objects have been annotated
        with extra data or aggregations.
        """
        self._not_support_combined_queries('annotate')
        return self._annotate(args, kwargs, select=True)

    def alias(self, *args, **kwargs):
        """
        Return a query set with added aliases for extra data or aggregations.
        """
        self._not_support_combined_queries('alias')
        return self._annotate(args, kwargs, select=False)
```
### 37 - django/db/models/query.py:

Start line: 867, End line: 881

```python
class QuerySet:

    def dates(self, field_name, kind, order='ASC'):
        """
        Return a list of date objects representing all available dates for
        the given field_name, scoped to 'kind'.
        """
        assert kind in ('year', 'month', 'week', 'day'), \
            "'kind' must be one of 'year', 'month', 'week', or 'day'."
        assert order in ('ASC', 'DESC'), \
            "'order' must be either 'ASC' or 'DESC'."
        return self.annotate(
            datefield=Trunc(field_name, kind, output_field=DateField()),
            plain_field=F(field_name)
        ).values_list(
            'datefield', flat=True
        ).distinct().filter(plain_field__isnull=False).order_by(('-' if order == 'DESC' else '') + 'datefield')
```
### 42 - django/db/models/query.py:

Start line: 1012, End line: 1026

```python
class QuerySet:

    def select_for_update(self, nowait=False, skip_locked=False, of=(), no_key=False):
        """
        Return a new QuerySet instance that will select objects with a
        FOR UPDATE lock.
        """
        if nowait and skip_locked:
            raise ValueError('The nowait option cannot be used with skip_locked.')
        obj = self._chain()
        obj._for_write = True
        obj.query.select_for_update = True
        obj.query.select_for_update_nowait = nowait
        obj.query.select_for_update_skip_locked = skip_locked
        obj.query.select_for_update_of = of
        obj.query.select_for_no_key_update = no_key
        return obj
```
### 43 - django/db/models/query.py:

Start line: 1284, End line: 1308

```python
class QuerySet:

    def _chain(self, **kwargs):
        """
        Return a copy of the current QuerySet that's ready for another
        operation.
        """
        obj = self._clone()
        if obj._sticky_filter:
            obj.query.filter_is_sticky = True
            obj._sticky_filter = False
        obj.__dict__.update(kwargs)
        return obj

    def _clone(self):
        """
        Return a copy of the current QuerySet. A lightweight alternative
        to deepcopy().
        """
        c = self.__class__(model=self.model, query=self.query.chain(), using=self._db, hints=self._hints)
        c._sticky_filter = self._sticky_filter
        c._for_write = self._for_write
        c._prefetch_related_lookups = self._prefetch_related_lookups[:]
        c._known_related_objects = self._known_related_objects
        c._iterable_class = self._iterable_class
        c._fields = self._fields
        return c
```
### 49 - django/db/models/query.py:

Start line: 1559, End line: 1615

```python
class Prefetch:
    def __init__(self, lookup, queryset=None, to_attr=None):
        # `prefetch_through` is the path we traverse to perform the prefetch.
        self.prefetch_through = lookup
        # `prefetch_to` is the path to the attribute that stores the result.
        self.prefetch_to = lookup
        if queryset is not None and (
            isinstance(queryset, RawQuerySet) or (
                hasattr(queryset, '_iterable_class') and
                not issubclass(queryset._iterable_class, ModelIterable)
            )
        ):
            raise ValueError(
                'Prefetch querysets cannot use raw(), values(), and '
                'values_list().'
            )
        if to_attr:
            self.prefetch_to = LOOKUP_SEP.join(lookup.split(LOOKUP_SEP)[:-1] + [to_attr])

        self.queryset = queryset
        self.to_attr = to_attr

    def __getstate__(self):
        obj_dict = self.__dict__.copy()
        if self.queryset is not None:
            # Prevent the QuerySet from being evaluated
            obj_dict['queryset'] = self.queryset._chain(
                _result_cache=[],
                _prefetch_done=True,
            )
        return obj_dict

    def add_prefix(self, prefix):
        self.prefetch_through = prefix + LOOKUP_SEP + self.prefetch_through
        self.prefetch_to = prefix + LOOKUP_SEP + self.prefetch_to

    def get_current_prefetch_to(self, level):
        return LOOKUP_SEP.join(self.prefetch_to.split(LOOKUP_SEP)[:level + 1])

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
### 53 - django/db/models/query.py:

Start line: 744, End line: 777

```python
class QuerySet:

    delete.alters_data = True
    delete.queryset_only = True

    def _raw_delete(self, using):
        """
        Delete objects found from the given queryset in single direct SQL
        query. No signals are sent and there is no protection for cascades.
        """
        query = self.query.clone()
        query.__class__ = sql.DeleteQuery
        cursor = query.get_compiler(using).execute_sql(CURSOR)
        if cursor:
            with cursor:
                return cursor.rowcount
        return 0
    _raw_delete.alters_data = True

    def update(self, **kwargs):
        """
        Update all elements in the current QuerySet, setting all the given
        fields to the appropriate values.
        """
        self._not_support_combined_queries('update')
        assert not self.query.is_sliced, \
            "Cannot update a query once a slice has been taken."
        self._for_write = True
        query = self.query.chain(sql.UpdateQuery)
        query.add_update_values(kwargs)
        # Clear any annotations so that they won't be present in subqueries.
        query.annotations = {}
        with transaction.mark_for_rollback_on_error(using=self.db):
            rows = query.get_compiler(self.db).execute_sql(CURSOR)
        self._result_cache = None
        return rows
```
### 55 - django/db/models/query.py:

Start line: 1207, End line: 1235

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
            self.query.default_ordering and
            self.query.get_meta().ordering and
            # A default ordering doesn't affect GROUP BY queries.
            not self.query.group_by
        ):
            return True
        else:
            return False
```
### 56 - django/db/models/query.py:

Start line: 296, End line: 327

```python
class QuerySet:

    def __getitem__(self, k):
        """Retrieve an item or slice from the set of results."""
        if not isinstance(k, (int, slice)):
            raise TypeError(
                'QuerySet indices must be integers or slices, not %s.'
                % type(k).__name__
            )
        assert ((not isinstance(k, slice) and (k >= 0)) or
                (isinstance(k, slice) and (k.start is None or k.start >= 0) and
                 (k.stop is None or k.stop >= 0))), \
            "Negative indexing is not supported."

        if self._result_cache is not None:
            return self._result_cache[k]

        if isinstance(k, slice):
            qs = self._chain()
            if k.start is not None:
                start = int(k.start)
            else:
                start = None
            if k.stop is not None:
                stop = int(k.stop)
            else:
                stop = None
            qs.query.set_limits(start, stop)
            return list(qs)[::k.step] if k.step else qs

        qs = self._chain()
        qs.query.set_limits(k, k + 1)
        qs._fetch_all()
        return qs._result_cache[0]
```
### 60 - django/db/models/query.py:

Start line: 676, End line: 713

```python
class QuerySet:

    def in_bulk(self, id_list=None, *, field_name='pk'):
        """
        Return a dictionary mapping each of the given IDs to the object with
        that ID. If `id_list` isn't provided, evaluate the entire QuerySet.
        """
        assert not self.query.is_sliced, \
            "Cannot use 'limit' or 'offset' with in_bulk"
        opts = self.model._meta
        unique_fields = [
            constraint.fields[0]
            for constraint in opts.total_unique_constraints
            if len(constraint.fields) == 1
        ]
        if (
            field_name != 'pk' and
            not opts.get_field(field_name).unique and
            field_name not in unique_fields and
            not self.query.distinct_fields == (field_name,)
        ):
            raise ValueError("in_bulk()'s field_name must be a unique field but %r isn't." % field_name)
        if id_list is not None:
            if not id_list:
                return {}
            filter_key = '{}__in'.format(field_name)
            batch_size = connections[self.db].features.max_query_params
            id_list = tuple(id_list)
            # If the database has a limit on the number of query parameters
            # (e.g. SQLite), retrieve objects in batches if necessary.
            if batch_size and batch_size < len(id_list):
                qs = ()
                for offset in range(0, len(id_list), batch_size):
                    batch = id_list[offset:offset + batch_size]
                    qs += tuple(self.filter(**{filter_key: batch}).order_by())
            else:
                qs = self.filter(**{filter_key: id_list}).order_by()
        else:
            qs = self._chain()
        return {getattr(obj, field_name): obj for obj in qs}
```
### 68 - django/db/models/query.py:

Start line: 1050, End line: 1071

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
        self._not_support_combined_queries('prefetch_related')
        clone = self._chain()
        if lookups == (None,):
            clone._prefetch_related_lookups = ()
        else:
            for lookup in lookups:
                if isinstance(lookup, Prefetch):
                    lookup = lookup.prefetch_to
                lookup = lookup.split(LOOKUP_SEP, 1)[0]
                if lookup in self.query._filtered_relations:
                    raise ValueError('prefetch_related() is not supported with FilteredRelation.')
            clone._prefetch_related_lookups = clone._prefetch_related_lookups + lookups
        return clone
```
### 71 - django/db/models/query.py:

Start line: 1435, End line: 1473

```python
class RawQuerySet:

    def prefetch_related(self, *lookups):
        """Same as QuerySet.prefetch_related()"""
        clone = self._clone()
        if lookups == (None,):
            clone._prefetch_related_lookups = ()
        else:
            clone._prefetch_related_lookups = clone._prefetch_related_lookups + lookups
        return clone

    def _prefetch_related_objects(self):
        prefetch_related_objects(self._result_cache, *self._prefetch_related_lookups)
        self._prefetch_done = True

    def _clone(self):
        """Same as QuerySet._clone()"""
        c = self.__class__(
            self.raw_query, model=self.model, query=self.query, params=self.params,
            translations=self.translations, using=self._db, hints=self._hints
        )
        c._prefetch_related_lookups = self._prefetch_related_lookups[:]
        return c

    def _fetch_all(self):
        if self._result_cache is None:
            self._result_cache = list(self.iterator())
        if self._prefetch_related_lookups and not self._prefetch_done:
            self._prefetch_related_objects()

    def __len__(self):
        self._fetch_all()
        return len(self._result_cache)

    def __bool__(self):
        self._fetch_all()
        return bool(self._result_cache)

    def __iter__(self):
        self._fetch_all()
        return iter(self._result_cache)
```
### 75 - django/db/models/query.py:

Start line: 1310, End line: 1328

```python
class QuerySet:

    def _fetch_all(self):
        if self._result_cache is None:
            self._result_cache = list(self._iterable_class(self))
        if self._prefetch_related_lookups and not self._prefetch_done:
            self._prefetch_related_objects()

    def _next_is_sticky(self):
        """
        Indicate that the next filter call and the one following that should
        be treated as a single filter. This is only important when it comes to
        determining when to reuse tables for many-to-many filters. Required so
        that we can filter naturally on the results of related managers.

        This doesn't return a clone of the current QuerySet (it returns
        "self"). The method is only used internally and should be immediately
        followed by a filter() that does create a clone.
        """
        self._sticky_filter = True
        return self
```
### 89 - django/db/models/query.py:

Start line: 1169, End line: 1184

```python
class QuerySet:

    def defer(self, *fields):
        """
        Defer the loading of data for certain fields until they are accessed.
        Add the set of deferred fields to any existing set of deferred fields.
        The only exception to this is if None is passed in as the only
        parameter, in which case removal all deferrals.
        """
        self._not_support_combined_queries('defer')
        if self._fields is not None:
            raise TypeError("Cannot call defer() after .values() or .values_list()")
        clone = self._chain()
        if fields == (None,):
            clone.query.clear_deferred_loading()
        else:
            clone.query.add_deferred_loading(fields)
        return clone
```
### 94 - django/db/models/query.py:

Start line: 1186, End line: 1205

```python
class QuerySet:

    def only(self, *fields):
        """
        Essentially, the opposite of defer(). Only the fields passed into this
        method and that are not already specified as deferred are loaded
        immediately when the queryset is evaluated.
        """
        self._not_support_combined_queries('only')
        if self._fields is not None:
            raise TypeError("Cannot call only() after .values() or .values_list()")
        if fields == (None,):
            # Can only pass None to defer(), not only(), as the rest option.
            # That won't stop people trying to do this, so let's be explicit.
            raise TypeError("Cannot pass None as an argument to only().")
        for field in fields:
            field = field.split(LOOKUP_SEP, 1)[0]
            if field in self.query._filtered_relations:
                raise ValueError('only() is not supported with FilteredRelation.')
        clone = self._chain()
        clone.query.add_immediate_loading(fields)
        return clone
```
### 95 - django/db/models/query.py:

Start line: 962, End line: 977

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
### 97 - django/db/models/query.py:

Start line: 979, End line: 1010

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
            return qs[0]._combinator_query('union', *qs[1:], all=all) if qs else self
        return self._combinator_query('union', *other_qs, all=all)

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
### 106 - django/db/models/query.py:

Start line: 1425, End line: 1433

```python
class RawQuerySet:

    def resolve_model_init_order(self):
        """Resolve the init field names and value positions."""
        converter = connections[self.db].introspection.identifier_converter
        model_init_fields = [f for f in self.model._meta.fields if converter(f.column) in self.columns]
        annotation_fields = [(column, pos) for pos, column in enumerate(self.columns)
                             if column not in self.model_fields]
        model_init_order = [self.columns.index(converter(f.column)) for f in model_init_fields]
        model_init_names = [f.attname for f in model_init_fields]
        return model_init_names, model_init_order, annotation_fields
```
### 113 - django/db/models/query.py:

Start line: 883, End line: 908

```python
class QuerySet:

    def datetimes(self, field_name, kind, order='ASC', tzinfo=None, is_dst=None):
        """
        Return a list of datetime objects representing all available
        datetimes for the given field_name, scoped to 'kind'.
        """
        assert kind in ('year', 'month', 'week', 'day', 'hour', 'minute', 'second'), \
            "'kind' must be one of 'year', 'month', 'week', 'day', 'hour', 'minute', or 'second'."
        assert order in ('ASC', 'DESC'), \
            "'order' must be either 'ASC' or 'DESC'."
        if settings.USE_TZ:
            if tzinfo is None:
                tzinfo = timezone.get_current_timezone()
        else:
            tzinfo = None
        return self.annotate(
            datetimefield=Trunc(
                field_name,
                kind,
                output_field=DateTimeField(),
                tzinfo=tzinfo,
                is_dst=is_dst,
            ),
            plain_field=F(field_name)
        ).values_list(
            'datetimefield', flat=True
        ).distinct().filter(plain_field__isnull=False).order_by(('-' if order == 'DESC' else '') + 'datetimefield')
```
### 127 - django/db/models/query.py:

Start line: 1330, End line: 1339

```python
class QuerySet:

    def _merge_sanity_check(self, other):
        """Check that two QuerySet classes may be merged."""
        if self._fields is not None and (
                set(self.query.values_select) != set(other.query.values_select) or
                set(self.query.extra_select) != set(other.query.extra_select) or
                set(self.query.annotation_select) != set(other.query.annotation_select)):
            raise TypeError(
                "Merging '%s' classes must involve the same values in each case."
                % self.__class__.__name__
            )
```
### 132 - django/db/models/query.py:

Start line: 415, End line: 455

```python
class QuerySet:

    def get(self, *args, **kwargs):
        """
        Perform the query and return a single object matching the given
        keyword arguments.
        """
        clone = self._chain() if self.query.combinator else self.filter(*args, **kwargs)
        if self.query.can_filter() and not self.query.distinct_fields:
            clone = clone.order_by()
        limit = None
        if not clone.query.select_for_update or connections[clone.db].features.supports_select_for_update_with_limit:
            limit = MAX_GET_RESULTS
            clone.query.set_limits(high=limit)
        num = len(clone)
        if num == 1:
            return clone._result_cache[0]
        if not num:
            raise self.model.DoesNotExist(
                "%s matching query does not exist." %
                self.model._meta.object_name
            )
        raise self.model.MultipleObjectsReturned(
            'get() returned more than one %s -- it returned %s!' % (
                self.model._meta.object_name,
                num if not limit or num < limit else 'more than %s' % (limit - 1),
            )
        )

    def create(self, **kwargs):
        """
        Create a new object with the given kwargs, saving it to the database
        and returning the created object.
        """
        obj = self.model(**kwargs)
        self._for_write = True
        obj.save(force_insert=True, using=self.db)
        return obj

    def _populate_pk_values(self, objs):
        for obj in objs:
            if obj.pk is None:
                obj.pk = obj._meta.pk.get_pk_value_on_save(obj)
```
### 145 - django/db/models/query.py:

Start line: 1341, End line: 1389

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
### 147 - django/db/models/query.py:

Start line: 1647, End line: 1753

```python
def prefetch_related_objects(model_instances, *related_lookups):
    # ... other code
    while all_lookups:
        lookup = all_lookups.pop()
        if lookup.prefetch_to in done_queries:
            if lookup.queryset is not None:
                raise ValueError("'%s' lookup was already seen with a different queryset. "
                                 "You may need to adjust the ordering of your lookups." % lookup.prefetch_to)

            continue

        # Top level, the list of objects to decorate is the result cache
        # from the primary QuerySet. It won't be for deeper levels.
        obj_list = model_instances

        through_attrs = lookup.prefetch_through.split(LOOKUP_SEP)
        for level, through_attr in enumerate(through_attrs):
            # Prepare main instances
            if not obj_list:
                break

            prefetch_to = lookup.get_current_prefetch_to(level)
            if prefetch_to in done_queries:
                # Skip any prefetching, and any object preparation
                obj_list = done_queries[prefetch_to]
                continue

            # Prepare objects:
            good_objects = True
            for obj in obj_list:
                # Since prefetching can re-use instances, it is possible to have
                # the same instance multiple times in obj_list, so obj might
                # already be prepared.
                if not hasattr(obj, '_prefetched_objects_cache'):
                    try:
                        obj._prefetched_objects_cache = {}
                    except (AttributeError, TypeError):
                        # Must be an immutable object from
                        # values_list(flat=True), for example (TypeError) or
                        # a QuerySet subclass that isn't returning Model
                        # instances (AttributeError), either in Django or a 3rd
                        # party. prefetch_related() doesn't make sense, so quit.
                        good_objects = False
                        break
            if not good_objects:
                break

            # Descend down tree

            # We assume that objects retrieved are homogeneous (which is the premise
            # of prefetch_related), so what applies to first object applies to all.
            first_obj = obj_list[0]
            to_attr = lookup.get_current_to_attr(level)[0]
            prefetcher, descriptor, attr_found, is_fetched = get_prefetcher(first_obj, through_attr, to_attr)

            if not attr_found:
                raise AttributeError("Cannot find '%s' on %s object, '%s' is an invalid "
                                     "parameter to prefetch_related()" %
                                     (through_attr, first_obj.__class__.__name__, lookup.prefetch_through))

            if level == len(through_attrs) - 1 and prefetcher is None:
                # Last one, this *must* resolve to something that supports
                # prefetching, otherwise there is no point adding it and the
                # developer asking for it has made a mistake.
                raise ValueError("'%s' does not resolve to an item that supports "
                                 "prefetching - this is an invalid parameter to "
                                 "prefetch_related()." % lookup.prefetch_through)

            if prefetcher is not None and not is_fetched:
                obj_list, additional_lookups = prefetch_one_level(obj_list, prefetcher, lookup, level)
                # We need to ensure we don't keep adding lookups from the
                # same relationships to stop infinite recursion. So, if we
                # are already on an automatically added lookup, don't add
                # the new lookups from relationships we've seen already.
                if not (prefetch_to in done_queries and lookup in auto_lookups and descriptor in followed_descriptors):
                    done_queries[prefetch_to] = obj_list
                    new_lookups = normalize_prefetch_lookups(reversed(additional_lookups), prefetch_to)
                    auto_lookups.update(new_lookups)
                    all_lookups.extend(new_lookups)
                followed_descriptors.add(descriptor)
            else:
                # Either a singly related object that has already been fetched
                # (e.g. via select_related), or hopefully some other property
                # that doesn't support prefetching but needs to be traversed.

                # We replace the current list of parent objects with the list
                # of related objects, filtering out empty or missing values so
                # that we can continue with nullable or reverse relations.
                new_obj_list = []
                for obj in obj_list:
                    if through_attr in getattr(obj, '_prefetched_objects_cache', ()):
                        # If related objects have been prefetched, use the
                        # cache rather than the object's through_attr.
                        new_obj = list(obj._prefetched_objects_cache.get(through_attr))
                    else:
                        try:
                            new_obj = getattr(obj, through_attr)
                        except exceptions.ObjectDoesNotExist:
                            continue
                    if new_obj is None:
                        continue
                    # We special-case `list` rather than something more generic
                    # like `Iterable` because we don't want to accidentally match
                    # user models that define __iter__.
                    if isinstance(new_obj, list):
                        new_obj_list.extend(new_obj)
                    else:
                        new_obj_list.append(new_obj)
                obj_list = new_obj_list
```
### 161 - django/db/models/query.py:

Start line: 374, End line: 413

```python
class QuerySet:

    def aggregate(self, *args, **kwargs):
        """
        Return a dictionary containing the calculations (aggregation)
        over the current queryset.

        If args is present the expression is passed as a kwarg using
        the Aggregate object's default alias.
        """
        if self.query.distinct_fields:
            raise NotImplementedError("aggregate() + distinct(fields) not implemented.")
        self._validate_values_are_expressions((*args, *kwargs.values()), method_name='aggregate')
        for arg in args:
            # The default_alias property raises TypeError if default_alias
            # can't be set automatically or AttributeError if it isn't an
            # attribute.
            try:
                arg.default_alias
            except (AttributeError, TypeError):
                raise TypeError("Complex aggregates require an alias")
            kwargs[arg.default_alias] = arg

        query = self.query.chain()
        for (alias, aggregate_expr) in kwargs.items():
            query.add_annotation(aggregate_expr, alias, is_summary=True)
            if not query.annotations[alias].contains_aggregate:
                raise TypeError("%s is not an aggregate expression" % alias)
        return query.get_aggregation(self.db, kwargs)

    def count(self):
        """
        Perform a SELECT COUNT() and return the number of records as an
        integer.

        If the QuerySet is already fully cached, return the length of the
        cached results set to avoid multiple SELECT COUNT(*) calls.
        """
        if self._result_cache is not None:
            return len(self._result_cache)

        return self.query.get_count(using=self.db)
```
