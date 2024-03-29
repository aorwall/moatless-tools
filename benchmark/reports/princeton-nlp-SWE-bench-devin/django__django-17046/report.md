# django__django-17046

| **django/django** | `95cdf9dc6627135f3893095892816eb3f2785e2e` |
| ---- | ---- |
| **No of patches** | 2 |
| **All found context length** | 10174 |
| **Any found context length** | 5939 |
| **Avg pos** | 53.5 |
| **Min pos** | 13 |
| **Max pos** | 42 |
| **Top file pos** | 6 |
| **Missing snippets** | 4 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/contrib/admin/views/main.py b/django/contrib/admin/views/main.py
--- a/django/contrib/admin/views/main.py
+++ b/django/contrib/admin/views/main.py
@@ -29,7 +29,7 @@
     SuspiciousOperation,
 )
 from django.core.paginator import InvalidPage
-from django.db.models import Exists, F, Field, ManyToOneRel, OrderBy, OuterRef
+from django.db.models import F, Field, ManyToOneRel, OrderBy
 from django.db.models.expressions import Combinable
 from django.urls import reverse
 from django.utils.deprecation import RemovedInDjango60Warning
@@ -566,6 +566,13 @@ def get_queryset(self, request, exclude_parameters=None):
             # ValueError, ValidationError, or ?.
             raise IncorrectLookupParameters(e)
 
+        if not qs.query.select_related:
+            qs = self.apply_select_related(qs)
+
+        # Set ordering.
+        ordering = self.get_ordering(request, qs)
+        qs = qs.order_by(*ordering)
+
         # Apply search results
         qs, search_may_have_duplicates = self.model_admin.get_search_results(
             request,
@@ -580,17 +587,9 @@ def get_queryset(self, request, exclude_parameters=None):
         )
         # Remove duplicates from results, if necessary
         if filters_may_have_duplicates | search_may_have_duplicates:
-            qs = qs.filter(pk=OuterRef("pk"))
-            qs = self.root_queryset.filter(Exists(qs))
-
-        # Set ordering.
-        ordering = self.get_ordering(request, qs)
-        qs = qs.order_by(*ordering)
-
-        if not qs.query.select_related:
-            qs = self.apply_select_related(qs)
-
-        return qs
+            return qs.distinct()
+        else:
+            return qs
 
     def apply_select_related(self, qs):
         if self.list_select_related is True:
diff --git a/django/db/models/query.py b/django/db/models/query.py
--- a/django/db/models/query.py
+++ b/django/db/models/query.py
@@ -1135,8 +1135,8 @@ def delete(self):
         self._not_support_combined_queries("delete")
         if self.query.is_sliced:
             raise TypeError("Cannot use 'limit' or 'offset' with delete().")
-        if self.query.distinct or self.query.distinct_fields:
-            raise TypeError("Cannot call delete() after .distinct().")
+        if self.query.distinct_fields:
+            raise TypeError("Cannot call delete() after .distinct(*fields).")
         if self._fields is not None:
             raise TypeError("Cannot call delete() after .values() or .values_list()")
 

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/contrib/admin/views/main.py | 32 | 32 | 13 | 6 | 5939
| django/contrib/admin/views/main.py | 569 | 569 | 26 | 6 | 10174
| django/contrib/admin/views/main.py | 583 | 593 | 26 | 6 | 10174
| django/db/models/query.py | 1138 | 1139 | 42 | 9 | 16755


## Problem Statement

```
Deleting objects after searching related many to many field crashes the admin page
Description
	
Minimal reproduction:
# models.py
class Post(models.Model):
 title = models.String(...)
 authors = models.ManyToMany("User", ...)
class User(models.Model):
 email = models.String(...)
# admin.py
class PostAdmin(admin.ModelAdmin):
 search_fields = ("title", "authors__email")
then opening the admin site, opening the post page that contains only one post (any title and author assigned) and entering a search term (e.g the first 2 characters of the title), selecting the post and then using the delete action results in an Internal Sever Error 500 with an error/stack-trace:
Internal Server Error: /admin/post/post/
Traceback (most recent call last):
 File "...lib/python3.7/site-packages/django/core/handlers/exception.py", line 47, in inner
	response = get_response(request)
 File "...lib/python3.7/site-packages/django/core/handlers/base.py", line 181, in _get_response
	response = wrapped_callback(request, *callback_args, **callback_kwargs)
 File "...lib/python3.7/site-packages/django/contrib/admin/options.py", line 616, in wrapper
	return self.admin_site.admin_view(view)(*args, **kwargs)
 File "...lib/python3.7/site-packages/django/utils/decorators.py", line 130, in _wrapped_view
	response = view_func(request, *args, **kwargs)
 File "...lib/python3.7/site-packages/django/views/decorators/cache.py", line 44, in _wrapped_view_func
	response = view_func(request, *args, **kwargs)
 File "...lib/python3.7/site-packages/django/contrib/admin/sites.py", line 241, in inner
	return view(request, *args, **kwargs)
 File "...lib/python3.7/site-packages/django/utils/decorators.py", line 43, in _wrapper
	return bound_method(*args, **kwargs)
 File "...lib/python3.7/site-packages/django/utils/decorators.py", line 130, in _wrapped_view
	response = view_func(request, *args, **kwargs)
 File "...lib/python3.7/site-packages/django/contrib/admin/options.py", line 1737, in changelist_view
	response = self.response_action(request, queryset=cl.get_queryset(request))
 File "...lib/python3.7/site-packages/django/contrib/admin/options.py", line 1406, in response_action
	response = func(self, request, queryset)
 File "...lib/python3.7/site-packages/django/contrib/admin/actions.py", line 45, in delete_selected
	modeladmin.delete_queryset(request, queryset)
 File "...lib/python3.7/site-packages/django/contrib/admin/options.py", line 1107, in delete_queryset
	queryset.delete()
 File "...lib/python3.7/site-packages/django/db/models/query.py", line 728, in delete
	raise TypeError('Cannot call delete() after .distinct().')
TypeError: Cannot call delete() after .distinct().
"POST /admin/post/post/?q=my HTTP/1.1" 500 137654
I can confirm that pip install django==3.1.8 fixes the error, and after having a look at the diff between stable/3.2.x and 3.1.8, I suspect the "regression" comes about from the work done on preserving the filters on delete or something along those lines - I haven't done a thorough investigation yet. Presumably .distinct() is being called because of the search involving the many to many field.
I am using a Postgres database.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 django/contrib/admin/options.py | 1161 | 1181| 201 | 201 | 19382 | 
| 2 | 2 django/db/models/deletion.py | 377 | 401| 273 | 474 | 23364 | 
| 3 | 2 django/contrib/admin/options.py | 2117 | 2192| 599 | 1073 | 23364 | 
| 4 | 2 django/db/models/deletion.py | 436 | 523| 630 | 1703 | 23364 | 
| 5 | 2 django/db/models/deletion.py | 314 | 375| 632 | 2335 | 23364 | 
| 6 | 2 django/contrib/admin/options.py | 2254 | 2303| 450 | 2785 | 23364 | 
| 7 | 3 django/db/models/base.py | 1161 | 1188| 238 | 3023 | 42482 | 
| 8 | 3 django/db/models/base.py | 1339 | 1386| 413 | 3436 | 42482 | 
| 9 | 3 django/db/models/deletion.py | 1 | 93| 601 | 4037 | 42482 | 
| 10 | 3 django/contrib/admin/options.py | 1531 | 1557| 233 | 4270 | 42482 | 
| 11 | 4 django/contrib/admin/utils.py | 140 | 177| 302 | 4572 | 46864 | 
| 12 | 5 django/db/models/fields/related.py | 1463 | 1592| 984 | 5556 | 61568 | 
| **-> 13 <-** | **6 django/contrib/admin/views/main.py** | 1 | 64| 383 | 5939 | 66485 | 
| 14 | 6 django/contrib/admin/options.py | 441 | 506| 550 | 6489 | 66485 | 
| 15 | 6 django/db/models/base.py | 1892 | 1927| 246 | 6735 | 66485 | 
| 16 | 7 django/contrib/postgres/search.py | 186 | 229| 322 | 7057 | 68965 | 
| 17 | 8 django/db/models/fields/related_descriptors.py | 788 | 893| 808 | 7865 | 80623 | 
| 18 | **9 django/db/models/query.py** | 1163 | 1185| 160 | 8025 | 101183 | 
| 19 | 10 django/db/backends/sqlite3/schema.py | 372 | 388| 132 | 8157 | 106031 | 
| 20 | 11 django/contrib/admin/checks.py | 1159 | 1195| 252 | 8409 | 115686 | 
| 21 | 11 django/db/models/deletion.py | 96 | 116| 210 | 8619 | 115686 | 
| 22 | 11 django/db/models/base.py | 1868 | 1890| 175 | 8794 | 115686 | 
| 23 | 11 django/db/models/base.py | 1567 | 1603| 288 | 9082 | 115686 | 
| 24 | 11 django/contrib/admin/options.py | 2400 | 2457| 465 | 9547 | 115686 | 
| 25 | 11 django/contrib/admin/options.py | 965 | 980| 123 | 9670 | 115686 | 
| **-> 26 <-** | **11 django/contrib/admin/views/main.py** | 531 | 593| 504 | 10174 | 115686 | 
| 27 | 11 django/contrib/admin/options.py | 1919 | 1933| 132 | 10306 | 115686 | 
| 28 | 12 django/db/migrations/autodetector.py | 807 | 902| 712 | 11018 | 129461 | 
| 29 | 12 django/db/models/deletion.py | 418 | 434| 130 | 11148 | 129461 | 
| 30 | 13 django/db/models/__init__.py | 1 | 116| 682 | 11830 | 130143 | 
| 31 | 13 django/contrib/admin/checks.py | 800 | 818| 183 | 12013 | 130143 | 
| 32 | 13 django/contrib/admin/utils.py | 122 | 138| 139 | 12152 | 130143 | 
| 33 | 13 django/contrib/admin/options.py | 2024 | 2115| 784 | 12936 | 130143 | 
| 34 | 13 django/contrib/admin/options.py | 1906 | 1917| 149 | 13085 | 130143 | 
| 35 | 13 django/db/models/base.py | 459 | 572| 957 | 14042 | 130143 | 
| 36 | 14 django/contrib/contenttypes/management/commands/remove_stale_contenttypes.py | 36 | 113| 598 | 14640 | 130943 | 
| 37 | **14 django/db/models/query.py** | 504 | 521| 136 | 14776 | 130943 | 
| 38 | **14 django/db/models/query.py** | 672 | 724| 424 | 15200 | 130943 | 
| 39 | 14 django/contrib/admin/options.py | 1770 | 1872| 780 | 15980 | 130943 | 
| 40 | 15 django/db/models/sql/query.py | 2076 | 2125| 332 | 16312 | 153870 | 
| 41 | 15 django/contrib/admin/options.py | 1676 | 1697| 133 | 16445 | 153870 | 
| **-> 42 <-** | **15 django/db/models/query.py** | 1127 | 1161| 310 | 16755 | 153870 | 
| 43 | 15 django/contrib/admin/options.py | 1935 | 2023| 676 | 17431 | 153870 | 
| 44 | 15 django/db/models/base.py | 1436 | 1466| 220 | 17651 | 153870 | 
| 45 | 16 django/contrib/admin/migrations/0001_initial.py | 1 | 76| 363 | 18014 | 154233 | 
| 46 | 17 django/contrib/sites/models.py | 79 | 121| 236 | 18250 | 155022 | 
| 47 | 17 django/db/models/fields/related.py | 304 | 341| 296 | 18546 | 155022 | 
| 48 | 17 django/contrib/admin/options.py | 2194 | 2252| 444 | 18990 | 155022 | 
| 49 | **17 django/contrib/admin/views/main.py** | 307 | 345| 356 | 19346 | 155022 | 
| 50 | **17 django/contrib/admin/views/main.py** | 595 | 627| 227 | 19573 | 155022 | 
| 51 | 17 django/contrib/admin/options.py | 2488 | 2523| 315 | 19888 | 155022 | 
| 52 | 17 django/contrib/admin/options.py | 243 | 256| 139 | 20027 | 155022 | 
| 53 | 17 django/db/models/deletion.py | 403 | 416| 117 | 20144 | 155022 | 
| 54 | 17 django/contrib/admin/options.py | 578 | 591| 165 | 20309 | 155022 | 
| 55 | 17 django/contrib/admin/checks.py | 176 | 192| 155 | 20464 | 155022 | 
| 56 | 18 django/db/backends/postgresql/features.py | 1 | 109| 867 | 21331 | 156105 | 
| 57 | **18 django/db/models/query.py** | 451 | 483| 253 | 21584 | 156105 | 
| 58 | 19 django/db/backends/postgresql/operations.py | 398 | 427| 234 | 21818 | 159583 | 
| 59 | 19 django/db/models/base.py | 1388 | 1417| 294 | 22112 | 159583 | 
| 60 | 19 django/contrib/admin/options.py | 364 | 440| 531 | 22643 | 159583 | 
| 61 | 19 django/contrib/admin/checks.py | 217 | 264| 333 | 22976 | 159583 | 
| 62 | 19 django/db/models/fields/related.py | 1693 | 1743| 431 | 23407 | 159583 | 
| 63 | 20 django/core/management/commands/flush.py | 31 | 93| 498 | 23905 | 160286 | 
| 64 | 20 django/contrib/admin/checks.py | 1054 | 1085| 229 | 24134 | 160286 | 
| 65 | 20 django/db/models/fields/related.py | 1036 | 1072| 261 | 24395 | 160286 | 
| 66 | 21 django/db/backends/oracle/schema.py | 50 | 71| 146 | 24541 | 162641 | 
| 67 | 22 django/db/backends/base/schema.py | 608 | 640| 268 | 24809 | 177788 | 
| 68 | 23 django/db/backends/oracle/creation.py | 159 | 201| 411 | 25220 | 181803 | 
| 69 | 24 django/contrib/gis/db/backends/spatialite/schema.py | 91 | 110| 155 | 25375 | 183197 | 
| 70 | 24 django/db/models/fields/related.py | 156 | 187| 209 | 25584 | 183197 | 
| 71 | 24 django/db/backends/postgresql/features.py | 111 | 138| 221 | 25805 | 183197 | 
| 72 | **24 django/db/models/query.py** | 1922 | 1987| 543 | 26348 | 183197 | 
| 73 | 25 django/contrib/admin/__init__.py | 1 | 53| 292 | 26640 | 183489 | 
| 74 | 26 django/contrib/admin/filters.py | 220 | 280| 541 | 27181 | 189215 | 
| 75 | 27 django/db/backends/postgresql/schema.py | 312 | 337| 235 | 27416 | 192056 | 
| 76 | 27 django/contrib/admin/checks.py | 1258 | 1287| 196 | 27612 | 192056 | 
| 77 | 27 django/contrib/postgres/search.py | 315 | 333| 124 | 27736 | 192056 | 
| 78 | 27 django/db/backends/base/schema.py | 501 | 520| 151 | 27887 | 192056 | 
| 79 | 27 django/db/models/fields/related.py | 1423 | 1461| 213 | 28100 | 192056 | 
| 80 | 27 django/db/models/fields/related_descriptors.py | 406 | 427| 158 | 28258 | 192056 | 
| 81 | 28 django/db/models/fields/related_lookups.py | 160 | 200| 250 | 28508 | 193593 | 
| 82 | 28 django/db/models/fields/related_lookups.py | 141 | 158| 216 | 28724 | 193593 | 
| 83 | 29 django/contrib/auth/admin.py | 28 | 40| 130 | 28854 | 195400 | 


### Hint

```
This exception was introduce in 6307c3f1a123f5975c73b231e8ac4f115fd72c0d and revealed a possible data loss issue in the admin. IMO we should use Exists() instead of distinct(), e.g. diff --git a/django/contrib/admin/views/main.py b/django/contrib/admin/views/main.py index fefed29933..e9816ddd15 100644 --- a/django/contrib/admin/views/main.py +++ b/django/contrib/admin/views/main.py @@ -475,9 +475,8 @@ class ChangeList: if not qs.query.select_related: qs = self.apply_select_related(qs) - # Set ordering. + # Get ordering. ordering = self.get_ordering(request, qs) - qs = qs.order_by(*ordering) # Apply search results qs, search_use_distinct = self.model_admin.get_search_results(request, qs, self.query) @@ -487,11 +486,14 @@ class ChangeList: new_params=remaining_lookup_params, remove=self.get_filters_params(), ) + # Remove duplicates from results, if necessary if filters_use_distinct | search_use_distinct: - return qs.distinct() + from django.db.models import Exists, OuterRef + qs = qs.filter(pk=OuterRef('pk')) + return self.root_queryset.filter(Exists(qs)).order_by(*ordering) else: - return qs + return qs.order_by(*ordering) def apply_select_related(self, qs): if self.list_select_related is True:
​PR
In 4074f38e: Refs #32682 -- Fixed QuerySet.delete() crash on querysets with self-referential subqueries on MySQL.
In cd74aad: Refs #32682 -- Renamed use_distinct variable to may_have_duplicates. QuerySet.distinct() is not the only way to avoid duplicate, it's also not preferred.
In 18711820: Fixed #32682 -- Made admin changelist use Exists() instead of distinct() for preventing duplicates. Thanks Zain Patel for the report and Simon Charette for reviews. The exception introduced in 6307c3f1a123f5975c73b231e8ac4f115fd72c0d revealed a possible data loss issue in the admin.
In 7ad70340: [3.2.x] Refs #32682 -- Fixed QuerySet.delete() crash on querysets with self-referential subqueries on MySQL. Backport of 4074f38e1dcc93b859bbbfd6abd8441c3bca36b3 from main
In fbea64b: [3.2.x] Refs #32682 -- Renamed use_distinct variable to may_have_duplicates. QuerySet.distinct() is not the only way to avoid duplicate, it's also not preferred. Backport of cd74aad90e09865ae6cd8ca0377ef0a5008d14e9 from main
In 34981f39: [3.2.x] Fixed #32682 -- Made admin changelist use Exists() instead of distinct() for preventing duplicates. Thanks Zain Patel for the report and Simon Charette for reviews. The exception introduced in 6307c3f1a123f5975c73b231e8ac4f115fd72c0d revealed a possible data loss issue in the admin. Backport of 187118203197801c6cb72dc8b06b714b23b6dd3d from main
In baba733d: Refs #32682 -- Renamed lookup_needs_distinct() to lookup_spawns_duplicates(). Follow up to 187118203197801c6cb72dc8b06b714b23b6dd3d.
```

## Patch

```diff
diff --git a/django/contrib/admin/views/main.py b/django/contrib/admin/views/main.py
--- a/django/contrib/admin/views/main.py
+++ b/django/contrib/admin/views/main.py
@@ -29,7 +29,7 @@
     SuspiciousOperation,
 )
 from django.core.paginator import InvalidPage
-from django.db.models import Exists, F, Field, ManyToOneRel, OrderBy, OuterRef
+from django.db.models import F, Field, ManyToOneRel, OrderBy
 from django.db.models.expressions import Combinable
 from django.urls import reverse
 from django.utils.deprecation import RemovedInDjango60Warning
@@ -566,6 +566,13 @@ def get_queryset(self, request, exclude_parameters=None):
             # ValueError, ValidationError, or ?.
             raise IncorrectLookupParameters(e)
 
+        if not qs.query.select_related:
+            qs = self.apply_select_related(qs)
+
+        # Set ordering.
+        ordering = self.get_ordering(request, qs)
+        qs = qs.order_by(*ordering)
+
         # Apply search results
         qs, search_may_have_duplicates = self.model_admin.get_search_results(
             request,
@@ -580,17 +587,9 @@ def get_queryset(self, request, exclude_parameters=None):
         )
         # Remove duplicates from results, if necessary
         if filters_may_have_duplicates | search_may_have_duplicates:
-            qs = qs.filter(pk=OuterRef("pk"))
-            qs = self.root_queryset.filter(Exists(qs))
-
-        # Set ordering.
-        ordering = self.get_ordering(request, qs)
-        qs = qs.order_by(*ordering)
-
-        if not qs.query.select_related:
-            qs = self.apply_select_related(qs)
-
-        return qs
+            return qs.distinct()
+        else:
+            return qs
 
     def apply_select_related(self, qs):
         if self.list_select_related is True:
diff --git a/django/db/models/query.py b/django/db/models/query.py
--- a/django/db/models/query.py
+++ b/django/db/models/query.py
@@ -1135,8 +1135,8 @@ def delete(self):
         self._not_support_combined_queries("delete")
         if self.query.is_sliced:
             raise TypeError("Cannot use 'limit' or 'offset' with delete().")
-        if self.query.distinct or self.query.distinct_fields:
-            raise TypeError("Cannot call delete() after .distinct().")
+        if self.query.distinct_fields:
+            raise TypeError("Cannot call delete() after .distinct(*fields).")
         if self._fields is not None:
             raise TypeError("Cannot call delete() after .values() or .values_list()")
 

```

## Test Patch

```diff
diff --git a/tests/admin_changelist/tests.py b/tests/admin_changelist/tests.py
--- a/tests/admin_changelist/tests.py
+++ b/tests/admin_changelist/tests.py
@@ -467,7 +467,7 @@ def test_custom_paginator(self):
         cl.get_results(request)
         self.assertIsInstance(cl.paginator, CustomPaginator)
 
-    def test_no_duplicates_for_m2m_in_list_filter(self):
+    def test_distinct_for_m2m_in_list_filter(self):
         """
         Regression test for #13902: When using a ManyToMany in list_filter,
         results shouldn't appear more than once. Basic ManyToMany.
@@ -488,11 +488,10 @@ def test_no_duplicates_for_m2m_in_list_filter(self):
         # There's only one Group instance
         self.assertEqual(cl.result_count, 1)
         # Queryset must be deletable.
-        self.assertIs(cl.queryset.query.distinct, False)
         cl.queryset.delete()
         self.assertEqual(cl.queryset.count(), 0)
 
-    def test_no_duplicates_for_through_m2m_in_list_filter(self):
+    def test_distinct_for_through_m2m_in_list_filter(self):
         """
         Regression test for #13902: When using a ManyToMany in list_filter,
         results shouldn't appear more than once. With an intermediate model.
@@ -512,14 +511,14 @@ def test_no_duplicates_for_through_m2m_in_list_filter(self):
         # There's only one Group instance
         self.assertEqual(cl.result_count, 1)
         # Queryset must be deletable.
-        self.assertIs(cl.queryset.query.distinct, False)
         cl.queryset.delete()
         self.assertEqual(cl.queryset.count(), 0)
 
-    def test_no_duplicates_for_through_m2m_at_second_level_in_list_filter(self):
+    def test_distinct_for_through_m2m_at_second_level_in_list_filter(self):
         """
         When using a ManyToMany in list_filter at the second level behind a
-        ForeignKey, results shouldn't appear more than once.
+        ForeignKey, distinct() must be called and results shouldn't appear more
+        than once.
         """
         lead = Musician.objects.create(name="Vox")
         band = Group.objects.create(name="The Hype")
@@ -537,11 +536,10 @@ def test_no_duplicates_for_through_m2m_at_second_level_in_list_filter(self):
         # There's only one Concert instance
         self.assertEqual(cl.result_count, 1)
         # Queryset must be deletable.
-        self.assertIs(cl.queryset.query.distinct, False)
         cl.queryset.delete()
         self.assertEqual(cl.queryset.count(), 0)
 
-    def test_no_duplicates_for_inherited_m2m_in_list_filter(self):
+    def test_distinct_for_inherited_m2m_in_list_filter(self):
         """
         Regression test for #13902: When using a ManyToMany in list_filter,
         results shouldn't appear more than once. Model managed in the
@@ -562,11 +560,10 @@ def test_no_duplicates_for_inherited_m2m_in_list_filter(self):
         # There's only one Quartet instance
         self.assertEqual(cl.result_count, 1)
         # Queryset must be deletable.
-        self.assertIs(cl.queryset.query.distinct, False)
         cl.queryset.delete()
         self.assertEqual(cl.queryset.count(), 0)
 
-    def test_no_duplicates_for_m2m_to_inherited_in_list_filter(self):
+    def test_distinct_for_m2m_to_inherited_in_list_filter(self):
         """
         Regression test for #13902: When using a ManyToMany in list_filter,
         results shouldn't appear more than once. Target of the relationship
@@ -586,15 +583,11 @@ def test_no_duplicates_for_m2m_to_inherited_in_list_filter(self):
 
         # There's only one ChordsBand instance
         self.assertEqual(cl.result_count, 1)
-        # Queryset must be deletable.
-        self.assertIs(cl.queryset.query.distinct, False)
-        cl.queryset.delete()
-        self.assertEqual(cl.queryset.count(), 0)
 
-    def test_no_duplicates_for_non_unique_related_object_in_list_filter(self):
+    def test_distinct_for_non_unique_related_object_in_list_filter(self):
         """
-        Regressions tests for #15819: If a field listed in list_filters is a
-        non-unique related object, results shouldn't appear more than once.
+        Regressions tests for #15819: If a field listed in list_filters
+        is a non-unique related object, distinct() must be called.
         """
         parent = Parent.objects.create(name="Mary")
         # Two children with the same name
@@ -606,10 +599,9 @@ def test_no_duplicates_for_non_unique_related_object_in_list_filter(self):
         request.user = self.superuser
 
         cl = m.get_changelist_instance(request)
-        # Exists() is applied.
+        # Make sure distinct() was called
         self.assertEqual(cl.queryset.count(), 1)
         # Queryset must be deletable.
-        self.assertIs(cl.queryset.query.distinct, False)
         cl.queryset.delete()
         self.assertEqual(cl.queryset.count(), 0)
 
@@ -629,10 +621,10 @@ def test_changelist_search_form_validation(self):
                 self.assertEqual(1, len(messages))
                 self.assertEqual(error, messages[0])
 
-    def test_no_duplicates_for_non_unique_related_object_in_search_fields(self):
+    def test_distinct_for_non_unique_related_object_in_search_fields(self):
         """
         Regressions tests for #15819: If a field listed in search_fields
-        is a non-unique related object, Exists() must be applied.
+        is a non-unique related object, distinct() must be called.
         """
         parent = Parent.objects.create(name="Mary")
         Child.objects.create(parent=parent, name="Danielle")
@@ -643,17 +635,16 @@ def test_no_duplicates_for_non_unique_related_object_in_search_fields(self):
         request.user = self.superuser
 
         cl = m.get_changelist_instance(request)
-        # Exists() is applied.
+        # Make sure distinct() was called
         self.assertEqual(cl.queryset.count(), 1)
         # Queryset must be deletable.
-        self.assertIs(cl.queryset.query.distinct, False)
         cl.queryset.delete()
         self.assertEqual(cl.queryset.count(), 0)
 
-    def test_no_duplicates_for_many_to_many_at_second_level_in_search_fields(self):
+    def test_distinct_for_many_to_many_at_second_level_in_search_fields(self):
         """
         When using a ManyToMany in search_fields at the second level behind a
-        ForeignKey, Exists() must be applied and results shouldn't appear more
+        ForeignKey, distinct() must be called and results shouldn't appear more
         than once.
         """
         lead = Musician.objects.create(name="Vox")
@@ -670,7 +661,6 @@ def test_no_duplicates_for_many_to_many_at_second_level_in_search_fields(self):
         # There's only one Concert instance
         self.assertEqual(cl.queryset.count(), 1)
         # Queryset must be deletable.
-        self.assertIs(cl.queryset.query.distinct, False)
         cl.queryset.delete()
         self.assertEqual(cl.queryset.count(), 0)
 
@@ -820,23 +810,23 @@ def test_custom_lookup_with_pk_shortcut(self):
         cl = m.get_changelist_instance(request)
         self.assertCountEqual(cl.queryset, [abcd])
 
-    def test_no_exists_for_m2m_in_list_filter_without_params(self):
+    def test_no_distinct_for_m2m_in_list_filter_without_params(self):
         """
         If a ManyToManyField is in list_filter but isn't in any lookup params,
-        the changelist's query shouldn't have Exists().
+        the changelist's query shouldn't have distinct.
         """
         m = BandAdmin(Band, custom_site)
         for lookup_params in ({}, {"name": "test"}):
             request = self.factory.get("/band/", lookup_params)
             request.user = self.superuser
             cl = m.get_changelist_instance(request)
-            self.assertNotIn(" EXISTS", str(cl.queryset.query))
+            self.assertIs(cl.queryset.query.distinct, False)
 
-        # A ManyToManyField in params does have Exists() applied.
+        # A ManyToManyField in params does have distinct applied.
         request = self.factory.get("/band/", {"genres": "0"})
         request.user = self.superuser
         cl = m.get_changelist_instance(request)
-        self.assertIn(" EXISTS", str(cl.queryset.query))
+        self.assertIs(cl.queryset.query.distinct, True)
 
     def test_pagination(self):
         """
diff --git a/tests/delete_regress/tests.py b/tests/delete_regress/tests.py
--- a/tests/delete_regress/tests.py
+++ b/tests/delete_regress/tests.py
@@ -396,10 +396,8 @@ def test_self_reference_with_through_m2m_at_second_level(self):
 
 
 class DeleteDistinct(SimpleTestCase):
-    def test_disallowed_delete_distinct(self):
-        msg = "Cannot call delete() after .distinct()."
-        with self.assertRaisesMessage(TypeError, msg):
-            Book.objects.distinct().delete()
+    def test_disallowed_delete_distinct_on(self):
+        msg = "Cannot call delete() after .distinct(*fields)."
         with self.assertRaisesMessage(TypeError, msg):
             Book.objects.distinct("id").delete()
 

```


## Code snippets

### 1 - django/contrib/admin/options.py:

Start line: 1161, End line: 1181

```python
class ModelAdmin(BaseModelAdmin):

    def get_search_results(self, request, queryset, search_term):
        # ... other code

        may_have_duplicates = False
        search_fields = self.get_search_fields(request)
        if search_fields and search_term:
            orm_lookups = [
                construct_search(str(search_field)) for search_field in search_fields
            ]
            term_queries = []
            for bit in smart_split(search_term):
                if bit.startswith(('"', "'")) and bit[0] == bit[-1]:
                    bit = unescape_string_literal(bit)
                or_queries = models.Q.create(
                    [(orm_lookup, bit) for orm_lookup in orm_lookups],
                    connector=models.Q.OR,
                )
                term_queries.append(or_queries)
            queryset = queryset.filter(models.Q.create(term_queries))
            may_have_duplicates |= any(
                lookup_spawns_duplicates(self.opts, search_spec)
                for search_spec in orm_lookups
            )
        return queryset, may_have_duplicates
```
### 2 - django/db/models/deletion.py:

Start line: 377, End line: 401

```python
class Collector:

    def collect(
        self,
        objs,
        source=None,
        nullable=False,
        collect_related=True,
        source_attr=None,
        reverse_dependency=False,
        keep_parents=False,
        fail_on_restricted=True,
    ):
        # ... other code

        if fail_on_restricted:
            # Raise an error if collected restricted objects (RESTRICT) aren't
            # candidates for deletion also collected via CASCADE.
            for related_model, instances in self.data.items():
                self.clear_restricted_objects_from_set(related_model, instances)
            for qs in self.fast_deletes:
                self.clear_restricted_objects_from_queryset(qs.model, qs)
            if self.restricted_objects.values():
                restricted_objects = defaultdict(list)
                for related_model, fields in self.restricted_objects.items():
                    for field, objs in fields.items():
                        if objs:
                            key = "'%s.%s'" % (related_model.__name__, field.name)
                            restricted_objects[key] += objs
                if restricted_objects:
                    raise RestrictedError(
                        "Cannot delete some instances of model %r because "
                        "they are referenced through restricted foreign keys: "
                        "%s."
                        % (
                            model.__name__,
                            ", ".join(restricted_objects),
                        ),
                        set(chain.from_iterable(restricted_objects.values())),
                    )
```
### 3 - django/contrib/admin/options.py:

Start line: 2117, End line: 2192

```python
class ModelAdmin(BaseModelAdmin):

    def get_deleted_objects(self, objs, request):
        """
        Hook for customizing the delete process for the delete view and the
        "delete selected" action.
        """
        return get_deleted_objects(objs, request, self.admin_site)

    @csrf_protect_m
    def delete_view(self, request, object_id, extra_context=None):
        with transaction.atomic(using=router.db_for_write(self.model)):
            return self._delete_view(request, object_id, extra_context)

    def _delete_view(self, request, object_id, extra_context):
        "The 'delete' admin view for this model."
        app_label = self.opts.app_label

        to_field = request.POST.get(TO_FIELD_VAR, request.GET.get(TO_FIELD_VAR))
        if to_field and not self.to_field_allowed(request, to_field):
            raise DisallowedModelAdminToField(
                "The field %s cannot be referenced." % to_field
            )

        obj = self.get_object(request, unquote(object_id), to_field)

        if not self.has_delete_permission(request, obj):
            raise PermissionDenied

        if obj is None:
            return self._get_obj_does_not_exist_redirect(request, self.opts, object_id)

        # Populate deleted_objects, a data structure of all related objects that
        # will also be deleted.
        (
            deleted_objects,
            model_count,
            perms_needed,
            protected,
        ) = self.get_deleted_objects([obj], request)

        if request.POST and not protected:  # The user has confirmed the deletion.
            if perms_needed:
                raise PermissionDenied
            obj_display = str(obj)
            attr = str(to_field) if to_field else self.opts.pk.attname
            obj_id = obj.serializable_value(attr)
            self.log_deletion(request, obj, obj_display)
            self.delete_model(request, obj)

            return self.response_delete(request, obj_display, obj_id)

        object_name = str(self.opts.verbose_name)

        if perms_needed or protected:
            title = _("Cannot delete %(name)s") % {"name": object_name}
        else:
            title = _("Are you sure?")

        context = {
            **self.admin_site.each_context(request),
            "title": title,
            "subtitle": None,
            "object_name": object_name,
            "object": obj,
            "deleted_objects": deleted_objects,
            "model_count": dict(model_count).items(),
            "perms_lacking": perms_needed,
            "protected": protected,
            "opts": self.opts,
            "app_label": app_label,
            "preserved_filters": self.get_preserved_filters(request),
            "is_popup": IS_POPUP_VAR in request.POST or IS_POPUP_VAR in request.GET,
            "to_field": to_field,
            **(extra_context or {}),
        }

        return self.render_delete_form(request, context)
```
### 4 - django/db/models/deletion.py:

Start line: 436, End line: 523

```python
class Collector:

    def delete(self):
        # sort instance collections
        for model, instances in self.data.items():
            self.data[model] = sorted(instances, key=attrgetter("pk"))

        # if possible, bring the models in an order suitable for databases that
        # don't support transactions or cannot defer constraint checks until the
        # end of a transaction.
        self.sort()
        # number of objects deleted for each model label
        deleted_counter = Counter()

        # Optimize for the case with a single obj and no dependencies
        if len(self.data) == 1 and len(instances) == 1:
            instance = list(instances)[0]
            if self.can_fast_delete(instance):
                with transaction.mark_for_rollback_on_error(self.using):
                    count = sql.DeleteQuery(model).delete_batch(
                        [instance.pk], self.using
                    )
                setattr(instance, model._meta.pk.attname, None)
                return count, {model._meta.label: count}

        with transaction.atomic(using=self.using, savepoint=False):
            # send pre_delete signals
            for model, obj in self.instances_with_model():
                if not model._meta.auto_created:
                    signals.pre_delete.send(
                        sender=model,
                        instance=obj,
                        using=self.using,
                        origin=self.origin,
                    )

            # fast deletes
            for qs in self.fast_deletes:
                count = qs._raw_delete(using=self.using)
                if count:
                    deleted_counter[qs.model._meta.label] += count

            # update fields
            for (field, value), instances_list in self.field_updates.items():
                updates = []
                objs = []
                for instances in instances_list:
                    if (
                        isinstance(instances, models.QuerySet)
                        and instances._result_cache is None
                    ):
                        updates.append(instances)
                    else:
                        objs.extend(instances)
                if updates:
                    combined_updates = reduce(or_, updates)
                    combined_updates.update(**{field.name: value})
                if objs:
                    model = objs[0].__class__
                    query = sql.UpdateQuery(model)
                    query.update_batch(
                        list({obj.pk for obj in objs}), {field.name: value}, self.using
                    )

            # reverse instance collections
            for instances in self.data.values():
                instances.reverse()

            # delete instances
            for model, instances in self.data.items():
                query = sql.DeleteQuery(model)
                pk_list = [obj.pk for obj in instances]
                count = query.delete_batch(pk_list, self.using)
                if count:
                    deleted_counter[model._meta.label] += count

                if not model._meta.auto_created:
                    for obj in instances:
                        signals.post_delete.send(
                            sender=model,
                            instance=obj,
                            using=self.using,
                            origin=self.origin,
                        )

        for model, instances in self.data.items():
            for instance in instances:
                setattr(instance, model._meta.pk.attname, None)
        return sum(deleted_counter.values()), dict(deleted_counter)
```
### 5 - django/db/models/deletion.py:

Start line: 314, End line: 375

```python
class Collector:

    def collect(
        self,
        objs,
        source=None,
        nullable=False,
        collect_related=True,
        source_attr=None,
        reverse_dependency=False,
        keep_parents=False,
        fail_on_restricted=True,
    ):
        # ... other code
        for related in get_candidate_relations_to_delete(model._meta):
            # Preserve parent reverse relationships if keep_parents=True.
            if keep_parents and related.model in parents:
                continue
            field = related.field
            on_delete = field.remote_field.on_delete
            if on_delete == DO_NOTHING:
                continue
            related_model = related.related_model
            if self.can_fast_delete(related_model, from_field=field):
                model_fast_deletes[related_model].append(field)
                continue
            batches = self.get_del_batches(new_objs, [field])
            for batch in batches:
                sub_objs = self.related_objects(related_model, [field], batch)
                # Non-referenced fields can be deferred if no signal receivers
                # are connected for the related model as they'll never be
                # exposed to the user. Skip field deferring when some
                # relationships are select_related as interactions between both
                # features are hard to get right. This should only happen in
                # the rare cases where .related_objects is overridden anyway.
                if not (
                    sub_objs.query.select_related
                    or self._has_signal_listeners(related_model)
                ):
                    referenced_fields = set(
                        chain.from_iterable(
                            (rf.attname for rf in rel.field.foreign_related_fields)
                            for rel in get_candidate_relations_to_delete(
                                related_model._meta
                            )
                        )
                    )
                    sub_objs = sub_objs.only(*tuple(referenced_fields))
                if getattr(on_delete, "lazy_sub_objs", False) or sub_objs:
                    try:
                        on_delete(self, field, sub_objs, self.using)
                    except ProtectedError as error:
                        key = "'%s.%s'" % (field.model.__name__, field.name)
                        protected_objects[key] += error.protected_objects
        if protected_objects:
            raise ProtectedError(
                "Cannot delete some instances of model %r because they are "
                "referenced through protected foreign keys: %s."
                % (
                    model.__name__,
                    ", ".join(protected_objects),
                ),
                set(chain.from_iterable(protected_objects.values())),
            )
        for related_model, related_fields in model_fast_deletes.items():
            batches = self.get_del_batches(new_objs, related_fields)
            for batch in batches:
                sub_objs = self.related_objects(related_model, related_fields, batch)
                self.fast_deletes.append(sub_objs)
        for field in model._meta.private_fields:
            if hasattr(field, "bulk_related_objects"):
                # It's something like generic foreign key.
                sub_objs = field.bulk_related_objects(new_objs, self.using)
                self.collect(
                    sub_objs, source=model, nullable=True, fail_on_restricted=False
                )
        # ... other code
```
### 6 - django/contrib/admin/options.py:

Start line: 2254, End line: 2303

```python
class ModelAdmin(BaseModelAdmin):

    def get_formset_kwargs(self, request, obj, inline, prefix):
        formset_params = {
            "instance": obj,
            "prefix": prefix,
            "queryset": inline.get_queryset(request),
        }
        if request.method == "POST":
            formset_params.update(
                {
                    "data": request.POST.copy(),
                    "files": request.FILES,
                    "save_as_new": "_saveasnew" in request.POST,
                }
            )
        return formset_params

    def _create_formsets(self, request, obj, change):
        "Helper function to generate formsets for add/change_view."
        formsets = []
        inline_instances = []
        prefixes = {}
        get_formsets_args = [request]
        if change:
            get_formsets_args.append(obj)
        for FormSet, inline in self.get_formsets_with_inlines(*get_formsets_args):
            prefix = FormSet.get_default_prefix()
            prefixes[prefix] = prefixes.get(prefix, 0) + 1
            if prefixes[prefix] != 1 or not prefix:
                prefix = "%s-%s" % (prefix, prefixes[prefix])
            formset_params = self.get_formset_kwargs(request, obj, inline, prefix)
            formset = FormSet(**formset_params)

            def user_deleted_form(request, obj, formset, index, inline):
                """Return whether or not the user deleted the form."""
                return (
                    inline.has_delete_permission(request, obj)
                    and "{}-{}-DELETE".format(formset.prefix, index) in request.POST
                )

            # Bypass validation of each view-only inline form (since the form's
            # data won't be in request.POST), unless the form was deleted.
            if not inline.has_change_permission(request, obj if change else None):
                for index, form in enumerate(formset.initial_forms):
                    if user_deleted_form(request, obj, formset, index, inline):
                        continue
                    form._errors = {}
                    form.cleaned_data = form.initial
            formsets.append(formset)
            inline_instances.append(inline)
        return formsets, inline_instances
```
### 7 - django/db/models/base.py:

Start line: 1161, End line: 1188

```python
class Model(AltersData, metaclass=ModelBase):

    def delete(self, using=None, keep_parents=False):
        if self.pk is None:
            raise ValueError(
                "%s object can't be deleted because its %s attribute is set "
                "to None." % (self._meta.object_name, self._meta.pk.attname)
            )
        using = using or router.db_for_write(self.__class__, instance=self)
        collector = Collector(using=using, origin=self)
        collector.collect([self], keep_parents=keep_parents)
        return collector.delete()

    delete.alters_data = True

    async def adelete(self, using=None, keep_parents=False):
        return await sync_to_async(self.delete)(
            using=using,
            keep_parents=keep_parents,
        )

    adelete.alters_data = True

    def _get_FIELD_display(self, field):
        value = getattr(self, field.attname)
        choices_dict = dict(make_hashable(field.flatchoices))
        # force_str() to coerce lazy strings.
        return force_str(
            choices_dict.get(make_hashable(value), value), strings_only=True
        )
```
### 8 - django/db/models/base.py:

Start line: 1339, End line: 1386

```python
class Model(AltersData, metaclass=ModelBase):

    def _perform_unique_checks(self, unique_checks):
        errors = {}

        for model_class, unique_check in unique_checks:
            # Try to look up an existing object with the same values as this
            # object's values for all the unique field.

            lookup_kwargs = {}
            for field_name in unique_check:
                f = self._meta.get_field(field_name)
                lookup_value = getattr(self, f.attname)
                # TODO: Handle multiple backends with different feature flags.
                if lookup_value is None or (
                    lookup_value == ""
                    and connection.features.interprets_empty_strings_as_nulls
                ):
                    # no value, skip the lookup
                    continue
                if f.primary_key and not self._state.adding:
                    # no need to check for unique primary key when editing
                    continue
                lookup_kwargs[str(field_name)] = lookup_value

            # some fields were skipped, no reason to do the check
            if len(unique_check) != len(lookup_kwargs):
                continue

            qs = model_class._default_manager.filter(**lookup_kwargs)

            # Exclude the current object from the query if we are editing an
            # instance (as opposed to creating a new one)
            # Note that we need to use the pk as defined by model_class, not
            # self.pk. These can be different fields because model inheritance
            # allows single model to have effectively multiple primary keys.
            # Refs #17615.
            model_class_pk = self._get_pk_val(model_class._meta)
            if not self._state.adding and model_class_pk is not None:
                qs = qs.exclude(pk=model_class_pk)
            if qs.exists():
                if len(unique_check) == 1:
                    key = unique_check[0]
                else:
                    key = NON_FIELD_ERRORS
                errors.setdefault(key, []).append(
                    self.unique_error_message(model_class, unique_check)
                )

        return errors
```
### 9 - django/db/models/deletion.py:

Start line: 1, End line: 93

```python
from collections import Counter, defaultdict
from functools import partial, reduce
from itertools import chain
from operator import attrgetter, or_

from django.db import IntegrityError, connections, models, transaction
from django.db.models import query_utils, signals, sql


class ProtectedError(IntegrityError):
    def __init__(self, msg, protected_objects):
        self.protected_objects = protected_objects
        super().__init__(msg, protected_objects)


class RestrictedError(IntegrityError):
    def __init__(self, msg, restricted_objects):
        self.restricted_objects = restricted_objects
        super().__init__(msg, restricted_objects)


def CASCADE(collector, field, sub_objs, using):
    collector.collect(
        sub_objs,
        source=field.remote_field.model,
        source_attr=field.name,
        nullable=field.null,
        fail_on_restricted=False,
    )
    if field.null and not connections[using].features.can_defer_constraint_checks:
        collector.add_field_update(field, None, sub_objs)


def PROTECT(collector, field, sub_objs, using):
    raise ProtectedError(
        "Cannot delete some instances of model '%s' because they are "
        "referenced through a protected foreign key: '%s.%s'"
        % (
            field.remote_field.model.__name__,
            sub_objs[0].__class__.__name__,
            field.name,
        ),
        sub_objs,
    )


def RESTRICT(collector, field, sub_objs, using):
    collector.add_restricted_objects(field, sub_objs)
    collector.add_dependency(field.remote_field.model, field.model)


def SET(value):
    if callable(value):

        def set_on_delete(collector, field, sub_objs, using):
            collector.add_field_update(field, value(), sub_objs)

    else:

        def set_on_delete(collector, field, sub_objs, using):
            collector.add_field_update(field, value, sub_objs)

    set_on_delete.deconstruct = lambda: ("django.db.models.SET", (value,), {})
    set_on_delete.lazy_sub_objs = True
    return set_on_delete


def SET_NULL(collector, field, sub_objs, using):
    collector.add_field_update(field, None, sub_objs)


SET_NULL.lazy_sub_objs = True


def SET_DEFAULT(collector, field, sub_objs, using):
    collector.add_field_update(field, field.get_default(), sub_objs)


SET_DEFAULT.lazy_sub_objs = True


def DO_NOTHING(collector, field, sub_objs, using):
    pass


def get_candidate_relations_to_delete(opts):
    # The candidate relations are the ones that come from N-1 and 1-1 relations.
    # N-N  (i.e., many-to-many) relations aren't candidates for deletion.
    return (
        f
        for f in opts.get_fields(include_hidden=True)
        if f.auto_created and not f.concrete and (f.one_to_one or f.one_to_many)
    )
```
### 10 - django/contrib/admin/options.py:

Start line: 1531, End line: 1557

```python
class ModelAdmin(BaseModelAdmin):

    def _response_post_save(self, request, obj):
        if self.has_view_or_change_permission(request):
            post_url = reverse(
                "admin:%s_%s_changelist" % (self.opts.app_label, self.opts.model_name),
                current_app=self.admin_site.name,
            )
            preserved_filters = self.get_preserved_filters(request)
            post_url = add_preserved_filters(
                {"preserved_filters": preserved_filters, "opts": self.opts}, post_url
            )
        else:
            post_url = reverse("admin:index", current_app=self.admin_site.name)
        return HttpResponseRedirect(post_url)

    def response_post_save_add(self, request, obj):
        """
        Figure out where to redirect after the 'Save' button has been pressed
        when adding a new object.
        """
        return self._response_post_save(request, obj)

    def response_post_save_change(self, request, obj):
        """
        Figure out where to redirect after the 'Save' button has been pressed
        when editing an existing object.
        """
        return self._response_post_save(request, obj)
```
### 13 - django/contrib/admin/views/main.py:

Start line: 1, End line: 64

```python
import warnings
from datetime import datetime, timedelta

from django import forms
from django.conf import settings
from django.contrib import messages
from django.contrib.admin import FieldListFilter
from django.contrib.admin.exceptions import (
    DisallowedModelAdminLookup,
    DisallowedModelAdminToField,
)
from django.contrib.admin.options import (
    IS_FACETS_VAR,
    IS_POPUP_VAR,
    TO_FIELD_VAR,
    IncorrectLookupParameters,
    ShowFacets,
)
from django.contrib.admin.utils import (
    build_q_object_from_lookup_parameters,
    get_fields_from_path,
    lookup_spawns_duplicates,
    prepare_lookup_value,
    quote,
)
from django.core.exceptions import (
    FieldDoesNotExist,
    ImproperlyConfigured,
    SuspiciousOperation,
)
from django.core.paginator import InvalidPage
from django.db.models import Exists, F, Field, ManyToOneRel, OrderBy, OuterRef
from django.db.models.expressions import Combinable
from django.urls import reverse
from django.utils.deprecation import RemovedInDjango60Warning
from django.utils.http import urlencode
from django.utils.inspect import func_supports_parameter
from django.utils.timezone import make_aware
from django.utils.translation import gettext

# Changelist settings
ALL_VAR = "all"
ORDER_VAR = "o"
PAGE_VAR = "p"
SEARCH_VAR = "q"
ERROR_FLAG = "e"

IGNORED_PARAMS = (
    ALL_VAR,
    ORDER_VAR,
    SEARCH_VAR,
    IS_FACETS_VAR,
    IS_POPUP_VAR,
    TO_FIELD_VAR,
)


class ChangeListSearchForm(forms.Form):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Populate "fields" dynamically because SEARCH_VAR is a variable:
        self.fields = {
            SEARCH_VAR: forms.CharField(required=False, strip=False),
        }
```
### 18 - django/db/models/query.py:

Start line: 1163, End line: 1185

```python
class QuerySet(AltersData):

    delete.alters_data = True
    delete.queryset_only = True

    async def adelete(self):
        return await sync_to_async(self.delete)()

    adelete.alters_data = True
    adelete.queryset_only = True

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
```
### 26 - django/contrib/admin/views/main.py:

Start line: 531, End line: 593

```python
class ChangeList:

    def get_queryset(self, request, exclude_parameters=None):
        # First, we collect all the declared list filters.
        (
            self.filter_specs,
            self.has_filters,
            remaining_lookup_params,
            filters_may_have_duplicates,
            self.has_active_filters,
        ) = self.get_filters(request)
        # Then, we let every list filter modify the queryset to its liking.
        qs = self.root_queryset
        for filter_spec in self.filter_specs:
            if (
                exclude_parameters is None
                or filter_spec.expected_parameters() != exclude_parameters
            ):
                new_qs = filter_spec.queryset(request, qs)
                if new_qs is not None:
                    qs = new_qs

        try:
            # Finally, we apply the remaining lookup parameters from the query
            # string (i.e. those that haven't already been processed by the
            # filters).
            q_object = build_q_object_from_lookup_parameters(remaining_lookup_params)
            qs = qs.filter(q_object)
        except (SuspiciousOperation, ImproperlyConfigured):
            # Allow certain types of errors to be re-raised as-is so that the
            # caller can treat them in a special way.
            raise
        except Exception as e:
            # Every other error is caught with a naked except, because we don't
            # have any other way of validating lookup parameters. They might be
            # invalid if the keyword arguments are incorrect, or if the values
            # are not in the correct type, so we might get FieldError,
            # ValueError, ValidationError, or ?.
            raise IncorrectLookupParameters(e)

        # Apply search results
        qs, search_may_have_duplicates = self.model_admin.get_search_results(
            request,
            qs,
            self.query,
        )

        # Set query string for clearing all filters.
        self.clear_all_filters_qs = self.get_query_string(
            new_params=remaining_lookup_params,
            remove=self.get_filters_params(),
        )
        # Remove duplicates from results, if necessary
        if filters_may_have_duplicates | search_may_have_duplicates:
            qs = qs.filter(pk=OuterRef("pk"))
            qs = self.root_queryset.filter(Exists(qs))

        # Set ordering.
        ordering = self.get_ordering(request, qs)
        qs = qs.order_by(*ordering)

        if not qs.query.select_related:
            qs = self.apply_select_related(qs)

        return qs
```
### 37 - django/db/models/query.py:

Start line: 504, End line: 521

```python
class QuerySet(AltersData):

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
### 38 - django/db/models/query.py:

Start line: 672, End line: 724

```python
class QuerySet(AltersData):

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
                if any(not f.concrete or f.many_to_many for f in unique_fields):
                    raise ValueError(
                        "bulk_create() can only be used with concrete fields "
                        "in unique_fields."
                    )
            return OnConflict.UPDATE
        return None
```
### 42 - django/db/models/query.py:

Start line: 1127, End line: 1161

```python
class QuerySet(AltersData):

    async def ain_bulk(self, id_list=None, *, field_name="pk"):
        return await sync_to_async(self.in_bulk)(
            id_list=id_list,
            field_name=field_name,
        )

    def delete(self):
        """Delete the records in the current QuerySet."""
        self._not_support_combined_queries("delete")
        if self.query.is_sliced:
            raise TypeError("Cannot use 'limit' or 'offset' with delete().")
        if self.query.distinct or self.query.distinct_fields:
            raise TypeError("Cannot call delete() after .distinct().")
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
        del_query.query.clear_ordering(force=True)

        collector = Collector(using=del_query.db, origin=self)
        collector.collect(del_query)
        deleted, _rows_count = collector.delete()

        # Clear the result cache, in case this QuerySet gets reused.
        self._result_cache = None
        return deleted, _rows_count
```
### 49 - django/contrib/admin/views/main.py:

Start line: 307, End line: 345

```python
class ChangeList:

    def get_results(self, request):
        paginator = self.model_admin.get_paginator(
            request, self.queryset, self.list_per_page
        )
        # Get the number of objects, with admin filters applied.
        result_count = paginator.count

        # Get the total number of objects, with no admin filters applied.
        # Note this isn't necessarily the same as result_count in the case of
        # no filtering. Filters defined in list_filters may still apply some
        # default filtering which may be removed with query parameters.
        if self.model_admin.show_full_result_count:
            full_result_count = self.root_queryset.count()
        else:
            full_result_count = None
        can_show_all = result_count <= self.list_max_show_all
        multi_page = result_count > self.list_per_page

        # Get the list of objects to display on this page.
        if (self.show_all and can_show_all) or not multi_page:
            result_list = self.queryset._clone()
        else:
            try:
                result_list = paginator.page(self.page_num).object_list
            except InvalidPage:
                raise IncorrectLookupParameters

        self.result_count = result_count
        self.show_full_result_count = self.model_admin.show_full_result_count
        # Admin actions are shown if there is at least one entry
        # or if entries are not counted because show_full_result_count is disabled
        self.show_admin_actions = not self.show_full_result_count or bool(
            full_result_count
        )
        self.full_result_count = full_result_count
        self.result_list = result_list
        self.can_show_all = can_show_all
        self.multi_page = multi_page
        self.paginator = paginator
```
### 50 - django/contrib/admin/views/main.py:

Start line: 595, End line: 627

```python
class ChangeList:

    def apply_select_related(self, qs):
        if self.list_select_related is True:
            return qs.select_related()

        if self.list_select_related is False:
            if self.has_related_field_in_list_display():
                return qs.select_related()

        if self.list_select_related:
            return qs.select_related(*self.list_select_related)
        return qs

    def has_related_field_in_list_display(self):
        for field_name in self.list_display:
            try:
                field = self.lookup_opts.get_field(field_name)
            except FieldDoesNotExist:
                pass
            else:
                if isinstance(field.remote_field, ManyToOneRel):
                    # <FK>_id field names don't require a join.
                    if field_name != field.get_attname():
                        return True
        return False

    def url_for_result(self, result):
        pk = getattr(result, self.pk_attname)
        return reverse(
            "admin:%s_%s_change" % (self.opts.app_label, self.opts.model_name),
            args=(quote(pk),),
            current_app=self.model_admin.admin_site.name,
        )
```
### 57 - django/db/models/query.py:

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
### 72 - django/db/models/query.py:

Start line: 1922, End line: 1987

```python
class QuerySet(AltersData):

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

    def _check_ordering_first_last_queryset_aggregation(self, method):
        if isinstance(self.query.group_by, tuple) and not any(
            col.output_field is self.model._meta.pk for col in self.query.group_by
        ):
            raise TypeError(
                f"Cannot use QuerySet.{method}() on an unordered queryset performing "
                f"aggregation. Add an ordering with order_by()."
            )
```
