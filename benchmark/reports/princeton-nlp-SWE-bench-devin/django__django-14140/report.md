# django__django-14140

| **django/django** | `45814af6197cfd8f4dc72ee43b90ecde305a1d5a` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 338 |
| **Any found context length** | 338 |
| **Avg pos** | 2.0 |
| **Min pos** | 2 |
| **Max pos** | 2 |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/db/models/query_utils.py b/django/db/models/query_utils.py
--- a/django/db/models/query_utils.py
+++ b/django/db/models/query_utils.py
@@ -84,14 +84,10 @@ def deconstruct(self):
         path = '%s.%s' % (self.__class__.__module__, self.__class__.__name__)
         if path.startswith('django.db.models.query_utils'):
             path = path.replace('django.db.models.query_utils', 'django.db.models')
-        args, kwargs = (), {}
-        if len(self.children) == 1 and not isinstance(self.children[0], Q):
-            child = self.children[0]
-            kwargs = {child[0]: child[1]}
-        else:
-            args = tuple(self.children)
-            if self.connector != self.default:
-                kwargs = {'_connector': self.connector}
+        args = tuple(self.children)
+        kwargs = {}
+        if self.connector != self.default:
+            kwargs['_connector'] = self.connector
         if self.negated:
             kwargs['_negated'] = True
         return path, args, kwargs

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/db/models/query_utils.py | 87 | 94 | 2 | 1 | 338


## Problem Statement

```
Combining Q() objects with boolean expressions crashes.
Description
	 
		(last modified by jonathan-golorry)
	 
Currently Q objects with 1 child are treated differently during deconstruct.
>>> from django.db.models import Q
>>> Q(x=1).deconstruct()
('django.db.models.Q', (), {'x': 1})
>>> Q(x=1, y=2).deconstruct()
('django.db.models.Q', (('x', 1), ('y', 2)), {})
This causes issues when deconstructing Q objects with a non-subscriptable child.
>>> from django.contrib.auth import get_user_model
>>> from django.db.models import Exists
>>> Q(Exists(get_user_model().objects.filter(username='jim'))).deconstruct()
Traceback (most recent call last):
 File "<console>", line 1, in <module>
 File "...", line 90, in deconstruct
	kwargs = {child[0]: child[1]}
TypeError: 'Exists' object is not subscriptable
Patch ​https://github.com/django/django/pull/14126 removes the special case, meaning single-child Q objects deconstruct into args instead of kwargs. A more backward-compatible approach would be to keep the special case and explicitly check that the child is a length-2 tuple, but it's unlikely that anyone is relying on this undocumented behavior.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 django/db/models/query_utils.py** | 61 | 81| 181 | 181 | 2563 | 
| **-> 2 <-** | **1 django/db/models/query_utils.py** | 83 | 97| 157 | 338 | 2563 | 
| 3 | 2 django/db/models/sql/query.py | 1401 | 1421| 247 | 585 | 24998 | 
| 4 | **2 django/db/models/query_utils.py** | 28 | 59| 259 | 844 | 24998 | 
| 5 | 2 django/db/models/sql/query.py | 1378 | 1399| 250 | 1094 | 24998 | 
| 6 | 3 django/db/models/expressions.py | 33 | 147| 836 | 1930 | 35985 | 
| 7 | 4 django/db/models/query.py | 1006 | 1026| 239 | 2169 | 53504 | 
| 8 | 5 django/db/models/__init__.py | 1 | 53| 619 | 2788 | 54123 | 
| 9 | 6 django/contrib/postgres/search.py | 130 | 157| 248 | 3036 | 56345 | 
| 10 | 6 django/db/models/sql/query.py | 641 | 665| 269 | 3305 | 56345 | 
| 11 | 7 django/db/models/sql/subqueries.py | 1 | 44| 320 | 3625 | 57546 | 
| 12 | 8 django/db/models/deletion.py | 1 | 76| 566 | 4191 | 61374 | 
| 13 | 8 django/db/models/sql/query.py | 1311 | 1376| 772 | 4963 | 61374 | 
| 14 | 8 django/db/models/expressions.py | 1153 | 1185| 254 | 5217 | 61374 | 
| 15 | 8 django/db/models/query.py | 320 | 346| 222 | 5439 | 61374 | 
| 16 | 8 django/db/models/expressions.py | 392 | 417| 186 | 5625 | 61374 | 
| 17 | 8 django/db/models/expressions.py | 550 | 591| 298 | 5923 | 61374 | 
| 18 | 9 django/db/models/constraints.py | 38 | 86| 378 | 6301 | 63398 | 
| 19 | 10 django/db/models/fields/related.py | 576 | 609| 334 | 6635 | 77221 | 
| 20 | 10 django/db/models/sql/query.py | 1 | 63| 460 | 7095 | 77221 | 
| 21 | 11 django/db/migrations/serializer.py | 78 | 105| 233 | 7328 | 79892 | 
| 22 | 11 django/db/models/sql/query.py | 1132 | 1164| 338 | 7666 | 79892 | 
| 23 | 11 django/db/models/sql/query.py | 1490 | 1575| 801 | 8467 | 79892 | 
| 24 | 11 django/db/models/expressions.py | 940 | 1004| 591 | 9058 | 79892 | 
| 25 | 12 django/db/models/sql/compiler.py | 443 | 496| 564 | 9622 | 94292 | 
| 26 | 12 django/db/models/sql/query.py | 716 | 751| 389 | 10011 | 94292 | 
| 27 | 12 django/db/models/query.py | 1372 | 1420| 405 | 10416 | 94292 | 
| 28 | 12 django/db/models/query.py | 1119 | 1160| 323 | 10739 | 94292 | 
| 29 | 12 django/db/models/sql/compiler.py | 150 | 198| 523 | 11262 | 94292 | 
| 30 | 12 django/db/models/sql/query.py | 566 | 640| 809 | 12071 | 94292 | 
| 31 | 12 django/db/models/sql/query.py | 1445 | 1472| 283 | 12354 | 94292 | 
| 32 | 12 django/db/models/sql/query.py | 1868 | 1917| 329 | 12683 | 94292 | 
| 33 | 12 django/db/models/sql/compiler.py | 1209 | 1230| 223 | 12906 | 94292 | 
| 34 | 12 django/db/models/constraints.py | 226 | 252| 205 | 13111 | 94292 | 
| 35 | 12 django/db/models/fields/related.py | 1471 | 1505| 356 | 13467 | 94292 | 
| 36 | 12 django/db/models/sql/query.py | 1770 | 1835| 666 | 14133 | 94292 | 
| 37 | 13 django/db/models/base.py | 404 | 509| 913 | 15046 | 111576 | 
| 38 | 13 django/db/models/sql/query.py | 1726 | 1768| 436 | 15482 | 111576 | 
| 39 | 13 django/db/models/sql/query.py | 1655 | 1679| 227 | 15709 | 111576 | 
| 40 | 14 django/db/migrations/questioner.py | 226 | 239| 123 | 15832 | 113633 | 
| 41 | 15 django/contrib/postgres/constraints.py | 129 | 155| 231 | 16063 | 115067 | 
| 42 | 15 django/db/models/constraints.py | 1 | 35| 250 | 16313 | 115067 | 
| 43 | 15 django/db/models/fields/related.py | 1 | 34| 246 | 16559 | 115067 | 
| 44 | 16 django/db/migrations/autodetector.py | 47 | 85| 322 | 16881 | 126686 | 
| 45 | 16 django/db/models/base.py | 1 | 50| 328 | 17209 | 126686 | 
| 46 | 17 django/db/models/sql/where.py | 233 | 249| 130 | 17339 | 128500 | 
| 47 | 17 django/db/models/query.py | 1028 | 1041| 126 | 17465 | 128500 | 
| 48 | 17 django/contrib/postgres/search.py | 160 | 195| 313 | 17778 | 128500 | 
| 49 | 17 django/db/models/query.py | 832 | 863| 272 | 18050 | 128500 | 
| 50 | 17 django/db/models/expressions.py | 492 | 517| 303 | 18353 | 128500 | 
| 51 | 17 django/db/models/base.py | 1771 | 1871| 729 | 19082 | 128500 | 
| 52 | 17 django/db/models/deletion.py | 269 | 344| 800 | 19882 | 128500 | 
| 53 | 17 django/db/models/query.py | 596 | 614| 178 | 20060 | 128500 | 
| 54 | 17 django/db/models/expressions.py | 420 | 442| 204 | 20264 | 128500 | 
| 55 | 17 django/db/models/sql/compiler.py | 1 | 19| 170 | 20434 | 128500 | 
| 56 | 17 django/db/models/deletion.py | 79 | 97| 199 | 20633 | 128500 | 
| 57 | 17 django/db/models/sql/query.py | 912 | 934| 248 | 20881 | 128500 | 
| 58 | 17 django/db/models/query.py | 752 | 785| 274 | 21155 | 128500 | 
| 59 | 18 django/db/backends/base/schema.py | 391 | 405| 182 | 21337 | 141192 | 
| 60 | 18 django/contrib/postgres/constraints.py | 1 | 67| 550 | 21887 | 141192 | 
| 61 | 18 django/db/models/fields/related.py | 864 | 890| 240 | 22127 | 141192 | 
| 62 | 18 django/db/models/query.py | 236 | 263| 221 | 22348 | 141192 | 
| 63 | 18 django/db/models/sql/query.py | 1004 | 1035| 307 | 22655 | 141192 | 
| 64 | 18 django/db/models/base.py | 1964 | 2120| 1158 | 23813 | 141192 | 
| 65 | 18 django/db/models/deletion.py | 165 | 199| 325 | 24138 | 141192 | 
| 66 | 18 django/db/models/query.py | 1361 | 1370| 114 | 24252 | 141192 | 
| 67 | 19 django/contrib/admin/utils.py | 160 | 186| 239 | 24491 | 145354 | 
| 68 | 19 django/contrib/postgres/search.py | 198 | 230| 243 | 24734 | 145354 | 
| 69 | 19 django/db/models/base.py | 1948 | 1962| 136 | 24870 | 145354 | 
| 70 | 19 django/db/models/sql/compiler.py | 22 | 47| 257 | 25127 | 145354 | 
| 71 | 19 django/db/models/query.py | 569 | 594| 202 | 25329 | 145354 | 
| 72 | 20 django/db/models/aggregates.py | 45 | 68| 294 | 25623 | 146655 | 
| 73 | **20 django/db/models/query_utils.py** | 1 | 25| 180 | 25803 | 146655 | 
| 74 | 20 django/db/models/sql/query.py | 1103 | 1130| 285 | 26088 | 146655 | 
| 75 | 21 django/core/serializers/__init__.py | 86 | 141| 369 | 26457 | 148412 | 
| 76 | 22 django/db/models/sql/datastructures.py | 1 | 21| 126 | 26583 | 149880 | 
| 77 | 22 django/db/models/expressions.py | 445 | 490| 314 | 26897 | 149880 | 
| 78 | 22 django/db/models/query.py | 722 | 750| 262 | 27159 | 149880 | 
| 79 | 23 django/contrib/gis/db/models/__init__.py | 1 | 19| 204 | 27363 | 150084 | 
| 80 | 23 django/db/models/constraints.py | 89 | 181| 685 | 28048 | 150084 | 
| 81 | 23 django/db/models/expressions.py | 219 | 253| 285 | 28333 | 150084 | 
| 82 | 23 django/db/models/sql/where.py | 195 | 209| 156 | 28489 | 150084 | 
| 83 | 23 django/db/models/query.py | 1217 | 1236| 209 | 28698 | 150084 | 
| 84 | 23 django/db/models/base.py | 1605 | 1630| 183 | 28881 | 150084 | 
| 85 | 23 django/db/models/deletion.py | 379 | 448| 580 | 29461 | 150084 | 
| 86 | **23 django/db/models/query_utils.py** | 140 | 204| 492 | 29953 | 150084 | 
| 87 | 23 django/db/models/sql/query.py | 376 | 426| 494 | 30447 | 150084 | 
| 88 | 23 django/db/models/base.py | 1269 | 1300| 267 | 30714 | 150084 | 
| 89 | 23 django/db/models/deletion.py | 346 | 359| 116 | 30830 | 150084 | 
| 90 | 23 django/db/models/base.py | 1087 | 1130| 404 | 31234 | 150084 | 
| 91 | 23 django/db/models/base.py | 1578 | 1603| 183 | 31417 | 150084 | 
| 92 | 23 django/db/models/sql/compiler.py | 1424 | 1438| 127 | 31544 | 150084 | 
| 93 | 24 django/db/models/fields/__init__.py | 507 | 522| 130 | 31674 | 168502 | 
| 94 | 24 django/db/models/query.py | 939 | 987| 371 | 32045 | 168502 | 
| 95 | 25 django/forms/models.py | 1186 | 1251| 543 | 32588 | 180276 | 
| 96 | 25 django/db/models/sql/query.py | 1072 | 1101| 249 | 32837 | 180276 | 
| 97 | 25 django/db/models/fields/__init__.py | 416 | 493| 667 | 33504 | 180276 | 
| 98 | 26 django/db/models/options.py | 1 | 35| 300 | 33804 | 187643 | 
| 99 | 26 django/db/models/sql/query.py | 140 | 234| 833 | 34637 | 187643 | 
| 100 | 27 django/db/backends/oracle/creation.py | 130 | 165| 399 | 35036 | 191536 | 
| 101 | 27 django/db/models/deletion.py | 361 | 377| 130 | 35166 | 191536 | 
| 102 | 28 django/db/models/indexes.py | 171 | 188| 205 | 35371 | 193856 | 
| 103 | 28 django/db/models/sql/where.py | 157 | 193| 243 | 35614 | 193856 | 
| 104 | 29 django/db/models/lookups.py | 101 | 155| 427 | 36041 | 198811 | 
| 105 | 29 django/db/models/lookups.py | 1 | 41| 313 | 36354 | 198811 | 
| 106 | 29 django/db/models/query.py | 786 | 809| 207 | 36561 | 198811 | 
| 107 | 29 django/db/models/query.py | 989 | 1004| 124 | 36685 | 198811 | 
| 108 | 29 django/db/models/sql/query.py | 300 | 349| 444 | 37129 | 198811 | 
| 109 | **29 django/db/models/query_utils.py** | 257 | 282| 293 | 37422 | 198811 | 
| 110 | 29 django/db/models/sql/compiler.py | 63 | 148| 890 | 38312 | 198811 | 


### Hint

```
Conditional expressions can be combined together, so it's not necessary to encapsulate Exists() with Q(). Moreover it's an undocumented and untested to pass conditional expressions to Q(). Nevertheless I think it makes sense to support this. There is no need to change the current format of deconstruct(), it should be enough to handle conditional expressions, e.g. diff --git a/django/db/models/query_utils.py b/django/db/models/query_utils.py index ae0f886107..5dc71ad619 100644 --- a/django/db/models/query_utils.py +++ b/django/db/models/query_utils.py @@ -85,7 +85,7 @@ class Q(tree.Node): if path.startswith('django.db.models.query_utils'): path = path.replace('django.db.models.query_utils', 'django.db.models') args, kwargs = (), {} - if len(self.children) == 1 and not isinstance(self.children[0], Q): + if len(self.children) == 1 and not isinstance(self.children[0], Q) and getattr(self.children[0], 'conditional', False) is False: child = self.children[0] kwargs = {child[0]: child[1]} else:
As Q() has .conditional set to True, we can amend Mariusz' example above to the following: django/db/models/query_utils.py diff --git a/django/db/models/query_utils.py b/django/db/models/query_utils.py index ae0f886107..5dc71ad619 100644 a b class Q(tree.Node): 8585 if path.startswith('django.db.models.query_utils'): 8686 path = path.replace('django.db.models.query_utils', 'django.db.models') 8787 args, kwargs = (), {} 88 if len(self.children) == 1 and not isinstance(self.children[0], Q): 88 if len(self.children) == 1 and getattr(self.children[0], 'conditional', False) is False: 8989 child = self.children[0] 9090 kwargs = {child[0]: child[1]} 9191 else:
Django already passes conditional expressions to Q objects internally. ​https://github.com/django/django/blob/main/django/db/models/expressions.py#L113 Tested here ​https://github.com/django/django/blob/main/tests/expressions/tests.py#L827 That test is only succeeding because Q(...) | Q(Q()) is treated differently from Q(...) | Q(). As for the form of .deconstruct(), is there any reason for keeping the special case? It's: Inconsistent: Q(x=1).deconstruct() vs Q(x=1, y=2).deconstruct() Fragile: Unsupported inputs like Q(False) sometimes (but not always!) lead to "not subscriptable" errors. Incorrect: Q(("x", 1)).deconstruct() incorrectly puts the condition in kwargs instead of args.
Django already passes conditional expressions to Q objects internally. ​https://github.com/django/django/blob/main/django/db/models/expressions.py#L113 Tested here ​https://github.com/django/django/blob/main/tests/expressions/tests.py#L827 These are example of combining conditional expressions. Again, initializing Q objects with conditional expressions is undocumented and untested. That test is only succeeding because Q(...) | Q(Q()) is treated differently from Q(...) | Q(). I'm not sure how it's related with the ticket 🤔 As for the form of .deconstruct(), is there any reason for keeping the special case? It's: Inconsistent: Q(x=1).deconstruct() vs Q(x=1, y=2).deconstruct() First is a single condition without a connector. Fragile: Unsupported inputs like Q(False) sometimes (but not always!) lead to "not subscriptable" errors. Incorrect: Q(("x", 1)).deconstruct() incorrectly puts the condition in kwargs instead of args. I wouldn't say that is incorrect Q(('x', 1)) is equivalent to the Q(x=1) so I don't see anything wrong with this behavior.
I suppose it's a semantics argument whether Q(Exists...) is untested if there's a test that runs that exact expression, but isn't solely checking that functionality. My point is that Q(("x", 1)) and Q(x=1) are equivalent, so it's impossible for the deconstruct to correctly recreate the original args and kwargs in all cases. Therefore, unless there's an argument for keeping the special case, it's better to consistently use args for both Q(x=1).deconstruct() and Q(x=1, y=2).deconstruct(). I point out Q(Exists...) | Q(Q()) to show that the fragility of the special case is problematic and hard to catch. An internal optimization for nested empty Q objects can cause conditional expression combination to fail. That's why I'd like this patch to be focused on removing the special case and making Q objects more robust for all inputs, rather than only adding support for expressions. Both would make my future work on Q objects possible, but the current patch would put django in a better position for future development. Edit: To clarify on my future work, I intend to add .empty() to detect nested empty Q objects -- such as Q(Q()) -- and remove them from logical operations. This would currently cause a regression in test_boolean_expression_combined_with_empty_Q. Ultimately the goal is to add robust implementations of Q.any() and Q.all() that can't return empty Q objects that accidentally leak the entire table ​https://forum.djangoproject.com/t/improving-q-objects-with-true-false-and-none/851/9 Edit 2: Patches are up. https://code.djangoproject.com/ticket/32549 and https://code.djangoproject.com/ticket/32554
Ian, can I ask for your opinion? We need another pair of eyes, I really don't see why the current format of deconstruct() is problematic 🤷.
I like the consistency and simplicity of the reworked deconstruct method. I think removing weird edge-cases in Q is a good thing. I think I would personally prefer a deconstruct api that always uses kwargs where possible, rather than args: ('django.db.models.Q', (), {'x': 1, 'y': 2}) looks nicer than ('django.db.models.Q', (('x', 1), ('y', 2)), {}) to me. I don't know how much harder this would be to implement though, and it's a machine facing interface, so human readability isn't the highest priority.
Unfortunately we can't use kwargs for Q objects with multiple children. (Q(x=1) & Q(x=2)).deconstruct() would lose an argument.
Replying to jonathan-golorry: Unfortunately we can't use kwargs for Q objects with multiple children. (Q(x=1) & Q(x=2)).deconstruct() would lose an argument. Excellent point!
```

## Patch

```diff
diff --git a/django/db/models/query_utils.py b/django/db/models/query_utils.py
--- a/django/db/models/query_utils.py
+++ b/django/db/models/query_utils.py
@@ -84,14 +84,10 @@ def deconstruct(self):
         path = '%s.%s' % (self.__class__.__module__, self.__class__.__name__)
         if path.startswith('django.db.models.query_utils'):
             path = path.replace('django.db.models.query_utils', 'django.db.models')
-        args, kwargs = (), {}
-        if len(self.children) == 1 and not isinstance(self.children[0], Q):
-            child = self.children[0]
-            kwargs = {child[0]: child[1]}
-        else:
-            args = tuple(self.children)
-            if self.connector != self.default:
-                kwargs = {'_connector': self.connector}
+        args = tuple(self.children)
+        kwargs = {}
+        if self.connector != self.default:
+            kwargs['_connector'] = self.connector
         if self.negated:
             kwargs['_negated'] = True
         return path, args, kwargs

```

## Test Patch

```diff
diff --git a/tests/expressions/tests.py b/tests/expressions/tests.py
--- a/tests/expressions/tests.py
+++ b/tests/expressions/tests.py
@@ -833,11 +833,21 @@ def test_boolean_expression_combined_with_empty_Q(self):
             Q() & Exists(is_poc),
             Exists(is_poc) | Q(),
             Q() | Exists(is_poc),
+            Q(Exists(is_poc)) & Q(),
+            Q() & Q(Exists(is_poc)),
+            Q(Exists(is_poc)) | Q(),
+            Q() | Q(Exists(is_poc)),
         ]
         for conditions in tests:
             with self.subTest(conditions):
                 self.assertCountEqual(Employee.objects.filter(conditions), [self.max])
 
+    def test_boolean_expression_in_Q(self):
+        is_poc = Company.objects.filter(point_of_contact=OuterRef('pk'))
+        self.gmbh.point_of_contact = self.max
+        self.gmbh.save()
+        self.assertCountEqual(Employee.objects.filter(Q(Exists(is_poc))), [self.max])
+
 
 class IterableLookupInnerExpressionsTests(TestCase):
     @classmethod
diff --git a/tests/queries/test_q.py b/tests/queries/test_q.py
--- a/tests/queries/test_q.py
+++ b/tests/queries/test_q.py
@@ -1,6 +1,8 @@
-from django.db.models import F, Q
+from django.db.models import Exists, F, OuterRef, Q
 from django.test import SimpleTestCase
 
+from .models import Tag
+
 
 class QTests(SimpleTestCase):
     def test_combine_and_empty(self):
@@ -39,17 +41,14 @@ def test_deconstruct(self):
         q = Q(price__gt=F('discounted_price'))
         path, args, kwargs = q.deconstruct()
         self.assertEqual(path, 'django.db.models.Q')
-        self.assertEqual(args, ())
-        self.assertEqual(kwargs, {'price__gt': F('discounted_price')})
+        self.assertEqual(args, (('price__gt', F('discounted_price')),))
+        self.assertEqual(kwargs, {})
 
     def test_deconstruct_negated(self):
         q = ~Q(price__gt=F('discounted_price'))
         path, args, kwargs = q.deconstruct()
-        self.assertEqual(args, ())
-        self.assertEqual(kwargs, {
-            'price__gt': F('discounted_price'),
-            '_negated': True,
-        })
+        self.assertEqual(args, (('price__gt', F('discounted_price')),))
+        self.assertEqual(kwargs, {'_negated': True})
 
     def test_deconstruct_or(self):
         q1 = Q(price__gt=F('discounted_price'))
@@ -88,6 +87,13 @@ def test_deconstruct_nested(self):
         self.assertEqual(args, (Q(price__gt=F('discounted_price')),))
         self.assertEqual(kwargs, {})
 
+    def test_deconstruct_boolean_expression(self):
+        tagged = Tag.objects.filter(category=OuterRef('pk'))
+        q = Q(Exists(tagged))
+        _, args, kwargs = q.deconstruct()
+        self.assertEqual(args, (Exists(tagged),))
+        self.assertEqual(kwargs, {})
+
     def test_reconstruct(self):
         q = Q(price__gt=F('discounted_price'))
         path, args, kwargs = q.deconstruct()
diff --git a/tests/queryset_pickle/tests.py b/tests/queryset_pickle/tests.py
--- a/tests/queryset_pickle/tests.py
+++ b/tests/queryset_pickle/tests.py
@@ -172,6 +172,17 @@ def test_pickle_prefetch_related_with_m2m_and_objects_deletion(self):
         m2ms = pickle.loads(pickle.dumps(m2ms))
         self.assertSequenceEqual(m2ms, [m2m])
 
+    def test_pickle_boolean_expression_in_Q__queryset(self):
+        group = Group.objects.create(name='group')
+        Event.objects.create(title='event', group=group)
+        groups = Group.objects.filter(
+            models.Q(models.Exists(
+                Event.objects.filter(group_id=models.OuterRef('id')),
+            )),
+        )
+        groups2 = pickle.loads(pickle.dumps(groups))
+        self.assertSequenceEqual(groups2, [group])
+
     def test_pickle_exists_queryset_still_usable(self):
         group = Group.objects.create(name='group')
         Event.objects.create(title='event', group=group)

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
### 2 - django/db/models/query_utils.py:

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
### 3 - django/db/models/sql/query.py:

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
### 4 - django/db/models/query_utils.py:

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
        if not(isinstance(other, Q) or getattr(other, 'conditional', False) is True):
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
### 5 - django/db/models/sql/query.py:

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
### 6 - django/db/models/expressions.py:

Start line: 33, End line: 147

```python
class Combinable:
    """
    Provide the ability to combine one or two objects with
    some connector. For example F('foo') + F('bar').
    """

    # Arithmetic connectors
    ADD = '+'
    SUB = '-'
    MUL = '*'
    DIV = '/'
    POW = '^'
    # The following is a quoted % operator - it is quoted because it can be
    # used in strings that also have parameter substitution.
    MOD = '%%'

    # Bitwise operators - note that these are generated by .bitand()
    # and .bitor(), the '&' and '|' are reserved for boolean operator
    # usage.
    BITAND = '&'
    BITOR = '|'
    BITLEFTSHIFT = '<<'
    BITRIGHTSHIFT = '>>'
    BITXOR = '#'

    def _combine(self, other, connector, reversed):
        if not hasattr(other, 'resolve_expression'):
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
        if getattr(self, 'conditional', False) and getattr(other, 'conditional', False):
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
        if getattr(self, 'conditional', False) and getattr(other, 'conditional', False):
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
### 7 - django/db/models/query.py:

Start line: 1006, End line: 1026

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
### 8 - django/db/models/__init__.py:

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
### 9 - django/contrib/postgres/search.py:

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
### 10 - django/db/models/sql/query.py:

Start line: 641, End line: 665

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
### 73 - django/db/models/query_utils.py:

Start line: 1, End line: 25

```python
"""
Various data structures used in query construction.

Factored out from django.db.models.query to avoid making the main module very
large and/or so that they can be used by other modules without getting into
circular import difficulties.
"""
import functools
import inspect
from collections import namedtuple

from django.core.exceptions import FieldError
from django.db.models.constants import LOOKUP_SEP
from django.utils import tree

# PathInfo is used when converting lookups (fk__somecol). The contents
# describe the relation in Model terms (model Options and Fields for both
# sides of the relation. The join_field is the field backing the relation.
PathInfo = namedtuple('PathInfo', 'from_opts to_opts target_fields join_field m2m direct filtered_relation')


def subclasses(cls):
    yield cls
    for subclass in cls.__subclasses__():
        yield from subclasses(subclass)
```
### 86 - django/db/models/query_utils.py:

Start line: 140, End line: 204

```python
class RegisterLookupMixin:

    @classmethod
    def _get_lookup(cls, lookup_name):
        return cls.get_lookups().get(lookup_name, None)

    @classmethod
    @functools.lru_cache(maxsize=None)
    def get_lookups(cls):
        class_lookups = [parent.__dict__.get('class_lookups', {}) for parent in inspect.getmro(cls)]
        return cls.merge_dicts(class_lookups)

    def get_lookup(self, lookup_name):
        from django.db.models.lookups import Lookup
        found = self._get_lookup(lookup_name)
        if found is None and hasattr(self, 'output_field'):
            return self.output_field.get_lookup(lookup_name)
        if found is not None and not issubclass(found, Lookup):
            return None
        return found

    def get_transform(self, lookup_name):
        from django.db.models.lookups import Transform
        found = self._get_lookup(lookup_name)
        if found is None and hasattr(self, 'output_field'):
            return self.output_field.get_transform(lookup_name)
        if found is not None and not issubclass(found, Transform):
            return None
        return found

    @staticmethod
    def merge_dicts(dicts):
        """
        Merge dicts in reverse to preference the order of the original list. e.g.,
        merge_dicts([a, b]) will preference the keys in 'a' over those in 'b'.
        """
        merged = {}
        for d in reversed(dicts):
            merged.update(d)
        return merged

    @classmethod
    def _clear_cached_lookups(cls):
        for subclass in subclasses(cls):
            subclass.get_lookups.cache_clear()

    @classmethod
    def register_lookup(cls, lookup, lookup_name=None):
        if lookup_name is None:
            lookup_name = lookup.lookup_name
        if 'class_lookups' not in cls.__dict__:
            cls.class_lookups = {}
        cls.class_lookups[lookup_name] = lookup
        cls._clear_cached_lookups()
        return lookup

    @classmethod
    def _unregister_lookup(cls, lookup, lookup_name=None):
        """
        Remove given lookup from cls lookups. For use in tests only as it's
        not thread-safe.
        """
        if lookup_name is None:
            lookup_name = lookup.lookup_name
        del cls.class_lookups[lookup_name]
```
### 109 - django/db/models/query_utils.py:

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
