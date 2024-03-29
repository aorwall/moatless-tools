# django__django-15957

| **django/django** | `f387d024fc75569d2a4a338bfda76cc2f328f627` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | - |
| **Avg pos** | - |
| **Min pos** | - |
| **Max pos** | - |
| **Top file pos** | 3 |
| **Missing snippets** | 4 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/db/models/fields/related_descriptors.py b/django/db/models/fields/related_descriptors.py
--- a/django/db/models/fields/related_descriptors.py
+++ b/django/db/models/fields/related_descriptors.py
@@ -64,8 +64,10 @@ class Child(Model):
 """
 
 from django.core.exceptions import FieldError
-from django.db import connections, router, transaction
-from django.db.models import Q, signals
+from django.db import DEFAULT_DB_ALIAS, connections, router, transaction
+from django.db.models import Q, Window, signals
+from django.db.models.functions import RowNumber
+from django.db.models.lookups import GreaterThan, LessThanOrEqual
 from django.db.models.query import QuerySet
 from django.db.models.query_utils import DeferredAttribute
 from django.db.models.utils import resolve_callables
@@ -81,6 +83,24 @@ def __set__(self, instance, value):
         instance.__dict__[self.field.attname] = value
 
 
+def _filter_prefetch_queryset(queryset, field_name, instances):
+    predicate = Q(**{f"{field_name}__in": instances})
+    if queryset.query.is_sliced:
+        low_mark, high_mark = queryset.query.low_mark, queryset.query.high_mark
+        order_by = [
+            expr
+            for expr, _ in queryset.query.get_compiler(
+                using=queryset._db or DEFAULT_DB_ALIAS
+            ).get_order_by()
+        ]
+        window = Window(RowNumber(), partition_by=field_name, order_by=order_by)
+        predicate &= GreaterThan(window, low_mark)
+        if high_mark is not None:
+            predicate &= LessThanOrEqual(window, high_mark)
+        queryset.query.clear_limits()
+    return queryset.filter(predicate)
+
+
 class ForwardManyToOneDescriptor:
     """
     Accessor to the related object on the forward side of a many-to-one or
@@ -718,8 +738,7 @@ def get_prefetch_queryset(self, instances, queryset=None):
             rel_obj_attr = self.field.get_local_related_value
             instance_attr = self.field.get_foreign_related_value
             instances_dict = {instance_attr(inst): inst for inst in instances}
-            query = {"%s__in" % self.field.name: instances}
-            queryset = queryset.filter(**query)
+            queryset = _filter_prefetch_queryset(queryset, self.field.name, instances)
 
             # Since we just bypassed this class' get_queryset(), we must manage
             # the reverse relation manually.
@@ -1050,9 +1069,9 @@ def get_prefetch_queryset(self, instances, queryset=None):
 
             queryset._add_hints(instance=instances[0])
             queryset = queryset.using(queryset._db or self._db)
-
-            query = {"%s__in" % self.query_field_name: instances}
-            queryset = queryset._next_is_sticky().filter(**query)
+            queryset = _filter_prefetch_queryset(
+                queryset._next_is_sticky(), self.query_field_name, instances
+            )
 
             # M2M: need to annotate the query in order to get the primary model
             # that the secondary model was actually related to. We know that

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/db/models/fields/related_descriptors.py | 67 | 68 | - | 3 | -
| django/db/models/fields/related_descriptors.py | 84 | 84 | - | 3 | -
| django/db/models/fields/related_descriptors.py | 721 | 722 | - | 3 | -
| django/db/models/fields/related_descriptors.py | 1053 | 1055 | - | 3 | -


## Problem Statement

```
Prefetch objects don't work with slices
Description
	
​Prefetch() objects does not work with sliced querysets. For example the following code results in AssertionError: Cannot filter a query once a slice has been taken.:
Category.objects.prefetch_related(Prefetch(
	'post_set',
	queryset=Post.objects.all()[:3],
	to_attr='example_posts',
))
This behavior is also mentioned in ​this StackOverflow answer. On the other hand it does not seem to be documented in Django Docs.
Why is it needed?
My use case seems to be a common one: I want to display a list of categories while displaying couple of example objects from each category next to it. If I'm not mistaken there isn't currently an efficient way of doing this. Prefetching without slicing would prefetch all objects (and there may be thousands of them) instead of the three examples that are needed.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 django/db/models/query.py | 2142 | 2200| 487 | 487 | 20450 | 
| 2 | 1 django/db/models/query.py | 2232 | 2366| 1129 | 1616 | 20450 | 
| 3 | 1 django/db/models/query.py | 2499 | 2531| 314 | 1930 | 20450 | 
| 4 | 1 django/db/models/query.py | 2428 | 2497| 665 | 2595 | 20450 | 
| 5 | 1 django/db/models/query.py | 1553 | 1576| 218 | 2813 | 20450 | 
| 6 | 1 django/db/models/query.py | 2203 | 2231| 246 | 3059 | 20450 | 
| 7 | 1 django/db/models/query.py | 2369 | 2425| 483 | 3542 | 20450 | 
| 8 | 1 django/db/models/query.py | 1695 | 1710| 149 | 3691 | 20450 | 
| 9 | 1 django/db/models/query.py | 501 | 518| 132 | 3823 | 20450 | 
| 10 | 1 django/db/models/query.py | 1262 | 1309| 357 | 4180 | 20450 | 
| 11 | 1 django/db/models/query.py | 411 | 446| 271 | 4451 | 20450 | 
| 12 | 2 django/contrib/contenttypes/fields.py | 174 | 221| 417 | 4868 | 26081 | 
| 13 | 2 django/db/models/query.py | 1874 | 1892| 186 | 5054 | 26081 | 
| 14 | **3 django/db/models/fields/related_descriptors.py** | 398 | 421| 194 | 5248 | 37152 | 
| 15 | 3 django/db/models/query.py | 2033 | 2139| 725 | 5973 | 37152 | 
| 16 | 3 django/db/models/query.py | 287 | 348| 469 | 6442 | 37152 | 
| 17 | 3 django/db/models/query.py | 42 | 64| 173 | 6615 | 37152 | 
| 18 | 4 django/db/models/__init__.py | 1 | 116| 682 | 7297 | 37834 | 
| 19 | 5 django/views/generic/list.py | 49 | 79| 251 | 7548 | 39436 | 
| 20 | **5 django/db/models/fields/related_descriptors.py** | 124 | 168| 418 | 7966 | 39436 | 
| 21 | 5 django/db/models/query.py | 350 | 377| 221 | 8187 | 39436 | 
| 22 | 5 django/db/models/query.py | 448 | 480| 249 | 8436 | 39436 | 
| 23 | 5 django/db/models/query.py | 546 | 563| 153 | 8589 | 39436 | 
| 24 | 5 django/db/models/query.py | 1712 | 1731| 209 | 8798 | 39436 | 
| 25 | 5 django/db/models/query.py | 1906 | 1971| 539 | 9337 | 39436 | 
| 26 | 5 django/db/models/query.py | 1024 | 1068| 337 | 9674 | 39436 | 
| 27 | 6 django/contrib/postgres/indexes.py | 1 | 42| 287 | 9961 | 41292 | 
| 28 | 6 django/db/models/query.py | 1593 | 1646| 343 | 10304 | 41292 | 
| 29 | 6 django/db/models/query.py | 1204 | 1225| 188 | 10492 | 41292 | 
| 30 | 7 django/views/generic/dates.py | 328 | 355| 234 | 10726 | 46802 | 
| 31 | 8 django/views/generic/detail.py | 61 | 77| 151 | 10877 | 48132 | 
| 32 | 8 django/db/models/query.py | 520 | 544| 224 | 11101 | 48132 | 
| 33 | 9 django/core/paginator.py | 1 | 164| 1263 | 12364 | 49824 | 
| 34 | 9 django/db/models/query.py | 1407 | 1455| 371 | 12735 | 49824 | 
| 35 | 10 django/db/backends/mysql/features.py | 86 | 160| 597 | 13332 | 52195 | 
| 36 | 11 django/db/backends/sqlite3/features.py | 66 | 113| 383 | 13715 | 53442 | 
| 37 | 11 django/contrib/contenttypes/fields.py | 623 | 657| 337 | 14052 | 53442 | 
| 38 | 11 django/views/generic/list.py | 1 | 47| 333 | 14385 | 53442 | 
| 39 | 11 django/db/models/query.py | 1844 | 1872| 205 | 14590 | 53442 | 
| 40 | 11 django/db/models/query.py | 1474 | 1485| 120 | 14710 | 53442 | 
| 41 | 12 django/contrib/admin/utils.py | 192 | 228| 251 | 14961 | 57648 | 
| 42 | 12 django/db/models/query.py | 1529 | 1551| 158 | 15119 | 57648 | 
| 43 | 12 django/views/generic/list.py | 81 | 120| 278 | 15397 | 57648 | 
| 44 | 12 django/db/models/query.py | 1648 | 1693| 340 | 15737 | 57648 | 
| 45 | 12 django/db/models/query.py | 1578 | 1591| 115 | 15852 | 57648 | 
| 46 | 13 django/contrib/contenttypes/management/commands/remove_stale_contenttypes.py | 34 | 111| 598 | 16450 | 58436 | 
| 47 | 13 django/db/models/query.py | 379 | 409| 246 | 16696 | 58436 | 
| 48 | 13 django/contrib/postgres/indexes.py | 166 | 188| 200 | 16896 | 58436 | 
| 49 | 14 django/db/backends/postgresql/features.py | 1 | 103| 792 | 17688 | 59228 | 
| 50 | 15 django/db/models/deletion.py | 398 | 411| 117 | 17805 | 63160 | 
| 51 | 16 django/contrib/postgres/aggregates/general.py | 41 | 57| 164 | 17969 | 63980 | 
| 52 | 16 django/db/models/query.py | 2604 | 2627| 201 | 18170 | 63980 | 
| 53 | 16 django/core/paginator.py | 167 | 225| 428 | 18598 | 63980 | 
| 54 | 16 django/contrib/postgres/indexes.py | 211 | 243| 233 | 18831 | 63980 | 
| 55 | 17 django/db/models/base.py | 1170 | 1191| 196 | 19027 | 82531 | 
| 56 | 18 django/contrib/gis/geos/mutable_list.py | 231 | 261| 248 | 19275 | 84794 | 
| 57 | 18 django/db/models/query.py | 661 | 682| 179 | 19454 | 84794 | 
| 58 | 18 django/db/models/base.py | 2450 | 2502| 341 | 19795 | 84794 | 
| 59 | 19 django/db/models/sql/query.py | 1656 | 1754| 846 | 20641 | 107965 | 
| 60 | 19 django/db/models/query.py | 66 | 75| 135 | 20776 | 107965 | 
| 61 | 19 django/db/models/query.py | 1148 | 1170| 156 | 20932 | 107965 | 
| 62 | 19 django/db/models/query.py | 684 | 743| 479 | 21411 | 107965 | 
| 63 | 20 django/utils/functional.py | 143 | 212| 520 | 21931 | 111248 | 
| 64 | 20 django/db/models/query.py | 795 | 853| 499 | 22430 | 111248 | 
| 65 | 21 django/db/models/sql/subqueries.py | 1 | 45| 311 | 22741 | 112476 | 
| 66 | 21 django/db/models/query.py | 1112 | 1146| 306 | 23047 | 112476 | 
| 67 | 21 django/contrib/gis/geos/mutable_list.py | 299 | 315| 133 | 23180 | 112476 | 
| 68 | 21 django/contrib/gis/geos/mutable_list.py | 286 | 297| 134 | 23314 | 112476 | 
| 69 | 21 django/contrib/gis/geos/mutable_list.py | 263 | 284| 191 | 23505 | 112476 | 
| 70 | 21 django/db/models/query.py | 1513 | 1527| 145 | 23650 | 112476 | 
| 71 | 21 django/db/models/query.py | 1349 | 1370| 219 | 23869 | 112476 | 
| 72 | 21 django/db/models/sql/query.py | 976 | 1003| 272 | 24141 | 112476 | 
| 73 | 21 django/db/models/query.py | 626 | 659| 293 | 24434 | 112476 | 
| 74 | 21 django/db/models/query.py | 1 | 39| 274 | 24708 | 112476 | 
| 75 | 21 django/db/models/query.py | 482 | 499| 146 | 24854 | 112476 | 
| 76 | 22 django/contrib/postgres/aggregates/mixins.py | 1 | 27| 238 | 25092 | 112714 | 
| 77 | 22 django/db/models/query.py | 943 | 967| 215 | 25307 | 112714 | 
| 78 | 23 django/core/cache/backends/base.py | 192 | 214| 199 | 25506 | 115822 | 
| 79 | 23 django/contrib/postgres/indexes.py | 191 | 208| 137 | 25643 | 115822 | 
| 80 | 23 django/db/models/sql/query.py | 589 | 604| 136 | 25779 | 115822 | 
| 81 | 23 django/db/models/query.py | 607 | 624| 131 | 25910 | 115822 | 
| 82 | 23 django/db/models/sql/query.py | 721 | 803| 845 | 26755 | 115822 | 
| 83 | 23 django/db/models/sql/query.py | 1609 | 1638| 287 | 27042 | 115822 | 
| 84 | 24 django/http/request.py | 548 | 584| 285 | 27327 | 121121 | 
| 85 | 25 django/middleware/cache.py | 130 | 146| 132 | 27459 | 122733 | 
| 86 | 25 django/middleware/cache.py | 148 | 176| 256 | 27715 | 122733 | 
| 87 | 25 django/db/models/sql/query.py | 1241 | 1275| 339 | 28054 | 122733 | 
| 88 | 26 django/contrib/postgres/fields/array.py | 260 | 353| 599 | 28653 | 125106 | 
| 89 | 27 django/db/backends/postgresql/operations.py | 319 | 336| 153 | 28806 | 128095 | 
| 90 | 28 django/db/models/sql/compiler.py | 1167 | 1272| 871 | 29677 | 144248 | 
| 91 | 29 django/db/models/query_utils.py | 150 | 188| 300 | 29977 | 147041 | 
| 92 | 29 django/db/models/deletion.py | 183 | 225| 337 | 30314 | 147041 | 
| 93 | 30 django/db/models/manager.py | 176 | 214| 207 | 30521 | 148489 | 
| 94 | 30 django/db/models/sql/query.py | 1518 | 1544| 263 | 30784 | 148489 | 
| 95 | 30 django/db/models/query.py | 1457 | 1472| 124 | 30908 | 148489 | 
| 96 | 30 django/contrib/postgres/aggregates/general.py | 60 | 95| 233 | 31141 | 148489 | 
| 97 | 30 django/db/models/sql/compiler.py | 1063 | 1165| 747 | 31888 | 148489 | 
| 98 | 31 django/utils/cache.py | 165 | 210| 459 | 32347 | 152282 | 
| 99 | 31 django/views/generic/list.py | 150 | 175| 205 | 32552 | 152282 | 
| 100 | 31 django/views/generic/dates.py | 459 | 504| 304 | 32856 | 152282 | 
| 101 | 31 django/db/models/sql/subqueries.py | 142 | 172| 190 | 33046 | 152282 | 
| 102 | 31 django/db/models/sql/query.py | 1435 | 1516| 813 | 33859 | 152282 | 
| 103 | 31 django/contrib/gis/geos/mutable_list.py | 82 | 100| 161 | 34020 | 152282 | 
| 104 | 31 django/db/models/sql/query.py | 2236 | 2280| 355 | 34375 | 152282 | 
| 105 | 32 django/db/models/fields/related.py | 154 | 185| 209 | 34584 | 166891 | 
| 106 | 33 django/contrib/postgres/search.py | 182 | 225| 322 | 34906 | 169360 | 
| 107 | 33 django/utils/functional.py | 121 | 141| 211 | 35117 | 169360 | 
| 108 | 34 django/contrib/admin/options.py | 1897 | 1911| 132 | 35249 | 188580 | 
| 109 | 34 django/views/generic/dates.py | 507 | 556| 370 | 35619 | 188580 | 
| 110 | 34 django/db/models/sql/query.py | 553 | 587| 286 | 35905 | 188580 | 
| 111 | 34 django/db/models/sql/query.py | 2081 | 2130| 332 | 36237 | 188580 | 
| 112 | 34 django/db/models/query.py | 1227 | 1260| 252 | 36489 | 188580 | 
| 113 | 35 django/db/backends/postgresql/base.py | 276 | 302| 228 | 36717 | 191640 | 
| 114 | 35 django/db/models/query.py | 907 | 941| 256 | 36973 | 191640 | 
| 115 | 35 django/db/models/deletion.py | 372 | 396| 273 | 37246 | 191640 | 
| 116 | 35 django/db/models/query.py | 1002 | 1022| 175 | 37421 | 191640 | 
| 117 | 36 django/db/models/fields/related_lookups.py | 171 | 211| 250 | 37671 | 193250 | 
| 118 | 36 django/db/models/base.py | 1149 | 1168| 228 | 37899 | 193250 | 
| 119 | 37 django/db/models/fields/reverse_related.py | 165 | 183| 160 | 38059 | 195782 | 
| 120 | 37 django/db/models/query.py | 1498 | 1511| 126 | 38185 | 195782 | 
| 121 | 38 django/db/models/sql/where.py | 325 | 343| 149 | 38334 | 198293 | 
| 122 | 38 django/db/models/query.py | 1487 | 1496| 126 | 38460 | 198293 | 


### Hint

```
It seems to me that #26565 ("Allow Prefetch query to use .values()") is related.
Hi ludwik, #26565 is about a different issue. The only way we could support slices would be to extract it from the provided queryset to make it filterable again and apply it when in-memory joining. The prefect queryset would still have to fetch all results so I'm not sure it's worth the trouble as accessing category.example_posts[:3] would have the same effect.
Charettes, Thank you for the reply. accessing category.example_posts[:3] would have the same effect I'm pretty sure that is not correct. The end effect would obviously be the same, but the way it is achieved behind the scenes and the performance characteristics that follows would be widely different. What you propose (prefetching without limit, adding slicing in a loop after the fact) would make Django perform a database query selecting all objects from the database table and loading them into memory. This would happen when the main queryset is evaluated (that's what prefetching is all about). Then the slicing would be performed by Python in memory, on a queryset that was already evaluated. That's what I understood from ​the documentation and also how Django actually behaved in an experiment I performed couple of minutes ago. What I want to avoid is exactly this behavior - loading thousands of objects from the database to display first three of them. I would be happy with a sane workaround, but to my knowledge there isn't any.
Tentatively accepting but as you've noticed it would require a large refactor of the way prefetching is actually done. I could see Category.objects.prefetch_related(Prefetch('articles', Article.object.order_by('-published_data')[0:3], to_attr='latest_articles')) being useful but the underlying prefetch query would need to rely on subqueries, a feature the ORM is not good at for now. Maybe the addition of the ​Subquery expression could help here.
The only way to limit this Query (and to not load all the records into memory) would be to do a top-n-per-group query which I don't think the ORM is capable of. What I can suggest is: Instead of limiting the QuerySet, find a way to filter it. For example, get only the Posts in the last day, week or a month (based on the post-frequency). Category.objects.prefetch_related(Prefetch( 'post_set', queryset=Post.objects.filter( date_published__gte=datetime.date.today()-timedelta(days=7)), to_attr='example_posts', )) This way you won't load all the Posts into Memory.
If window queries are implemented (https://code.djangoproject.com/ticket/26608) then this use case could be possible in something similar to: Prefetch( 'post_set', queryset= Post.objects .annotate(_rank=Window(Rank(), partition_by='cateogry') .filter(_rank__lte=3) )
I tried the filter by rank suggestion using Django 2.0 but I get this error: django.db.utils.NotSupportedError: Window is disallowed in the filter clause.
I think there is a workaround now to in django new version as we have OuterRef and Subquery. from django.db.models import OuterRef, Subquery User.objects.all().prefetch_related('comments',queryset=Comment.objects.filter(id__in=Subquery(Comment.objects.filter(user_id=OuterRef('user_id')).values_list('id', flat=True)[:5])))
Note that the solution above might not be portable to some MySql databases, with error 1235, "This version of MySQL doesn't yet support 'LIMIT & IN/ALL/ANY/SOME subquery'"
As pointed out by others support for filter against window functions would allow prefetch_related_objects to use Rank(partition_by) to support this feature. If someone wants to give a shot at solving this particular issue before #28333 is resolved it should be doable by using a combination of Queryset.raw and Query.compile combinations.
```

## Patch

```diff
diff --git a/django/db/models/fields/related_descriptors.py b/django/db/models/fields/related_descriptors.py
--- a/django/db/models/fields/related_descriptors.py
+++ b/django/db/models/fields/related_descriptors.py
@@ -64,8 +64,10 @@ class Child(Model):
 """
 
 from django.core.exceptions import FieldError
-from django.db import connections, router, transaction
-from django.db.models import Q, signals
+from django.db import DEFAULT_DB_ALIAS, connections, router, transaction
+from django.db.models import Q, Window, signals
+from django.db.models.functions import RowNumber
+from django.db.models.lookups import GreaterThan, LessThanOrEqual
 from django.db.models.query import QuerySet
 from django.db.models.query_utils import DeferredAttribute
 from django.db.models.utils import resolve_callables
@@ -81,6 +83,24 @@ def __set__(self, instance, value):
         instance.__dict__[self.field.attname] = value
 
 
+def _filter_prefetch_queryset(queryset, field_name, instances):
+    predicate = Q(**{f"{field_name}__in": instances})
+    if queryset.query.is_sliced:
+        low_mark, high_mark = queryset.query.low_mark, queryset.query.high_mark
+        order_by = [
+            expr
+            for expr, _ in queryset.query.get_compiler(
+                using=queryset._db or DEFAULT_DB_ALIAS
+            ).get_order_by()
+        ]
+        window = Window(RowNumber(), partition_by=field_name, order_by=order_by)
+        predicate &= GreaterThan(window, low_mark)
+        if high_mark is not None:
+            predicate &= LessThanOrEqual(window, high_mark)
+        queryset.query.clear_limits()
+    return queryset.filter(predicate)
+
+
 class ForwardManyToOneDescriptor:
     """
     Accessor to the related object on the forward side of a many-to-one or
@@ -718,8 +738,7 @@ def get_prefetch_queryset(self, instances, queryset=None):
             rel_obj_attr = self.field.get_local_related_value
             instance_attr = self.field.get_foreign_related_value
             instances_dict = {instance_attr(inst): inst for inst in instances}
-            query = {"%s__in" % self.field.name: instances}
-            queryset = queryset.filter(**query)
+            queryset = _filter_prefetch_queryset(queryset, self.field.name, instances)
 
             # Since we just bypassed this class' get_queryset(), we must manage
             # the reverse relation manually.
@@ -1050,9 +1069,9 @@ def get_prefetch_queryset(self, instances, queryset=None):
 
             queryset._add_hints(instance=instances[0])
             queryset = queryset.using(queryset._db or self._db)
-
-            query = {"%s__in" % self.query_field_name: instances}
-            queryset = queryset._next_is_sticky().filter(**query)
+            queryset = _filter_prefetch_queryset(
+                queryset._next_is_sticky(), self.query_field_name, instances
+            )
 
             # M2M: need to annotate the query in order to get the primary model
             # that the secondary model was actually related to. We know that

```

## Test Patch

```diff
diff --git a/tests/prefetch_related/tests.py b/tests/prefetch_related/tests.py
--- a/tests/prefetch_related/tests.py
+++ b/tests/prefetch_related/tests.py
@@ -1908,3 +1908,67 @@ def test_nested_prefetch_is_not_overwritten_by_related_object(self):
         self.assertIs(Room.house.is_cached(self.room), True)
         with self.assertNumQueries(0):
             house.rooms.first().house.address
+
+
+class PrefetchLimitTests(TestDataMixin, TestCase):
+    def test_m2m_forward(self):
+        authors = Author.objects.all()  # Meta.ordering
+        with self.assertNumQueries(3):
+            books = list(
+                Book.objects.prefetch_related(
+                    Prefetch("authors", authors),
+                    Prefetch("authors", authors[1:], to_attr="authors_sliced"),
+                )
+            )
+        for book in books:
+            with self.subTest(book=book):
+                self.assertEqual(book.authors_sliced, list(book.authors.all())[1:])
+
+    def test_m2m_reverse(self):
+        books = Book.objects.order_by("title")
+        with self.assertNumQueries(3):
+            authors = list(
+                Author.objects.prefetch_related(
+                    Prefetch("books", books),
+                    Prefetch("books", books[1:2], to_attr="books_sliced"),
+                )
+            )
+        for author in authors:
+            with self.subTest(author=author):
+                self.assertEqual(author.books_sliced, list(author.books.all())[1:2])
+
+    def test_foreignkey_reverse(self):
+        authors = Author.objects.order_by("-name")
+        with self.assertNumQueries(3):
+            books = list(
+                Book.objects.prefetch_related(
+                    Prefetch(
+                        "first_time_authors",
+                        authors,
+                    ),
+                    Prefetch(
+                        "first_time_authors",
+                        authors[1:],
+                        to_attr="first_time_authors_sliced",
+                    ),
+                )
+            )
+        for book in books:
+            with self.subTest(book=book):
+                self.assertEqual(
+                    book.first_time_authors_sliced,
+                    list(book.first_time_authors.all())[1:],
+                )
+
+    def test_reverse_ordering(self):
+        authors = Author.objects.reverse()  # Reverse Meta.ordering
+        with self.assertNumQueries(3):
+            books = list(
+                Book.objects.prefetch_related(
+                    Prefetch("authors", authors),
+                    Prefetch("authors", authors[1:], to_attr="authors_sliced"),
+                )
+            )
+        for book in books:
+            with self.subTest(book=book):
+                self.assertEqual(book.authors_sliced, list(book.authors.all())[1:])

```


## Code snippets

### 1 - django/db/models/query.py:

Start line: 2142, End line: 2200

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
### 2 - django/db/models/query.py:

Start line: 2232, End line: 2366

```python
def prefetch_related_objects(model_instances, *related_lookups):
    # ... other code
    while all_lookups:
        lookup = all_lookups.pop()
        if lookup.prefetch_to in done_queries:
            if lookup.queryset is not None:
                raise ValueError(
                    "'%s' lookup was already seen with a different queryset. "
                    "You may need to adjust the ordering of your lookups."
                    % lookup.prefetch_to
                )

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
                if not hasattr(obj, "_prefetched_objects_cache"):
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
            prefetcher, descriptor, attr_found, is_fetched = get_prefetcher(
                first_obj, through_attr, to_attr
            )

            if not attr_found:
                raise AttributeError(
                    "Cannot find '%s' on %s object, '%s' is an invalid "
                    "parameter to prefetch_related()"
                    % (
                        through_attr,
                        first_obj.__class__.__name__,
                        lookup.prefetch_through,
                    )
                )

            if level == len(through_attrs) - 1 and prefetcher is None:
                # Last one, this *must* resolve to something that supports
                # prefetching, otherwise there is no point adding it and the
                # developer asking for it has made a mistake.
                raise ValueError(
                    "'%s' does not resolve to an item that supports "
                    "prefetching - this is an invalid parameter to "
                    "prefetch_related()." % lookup.prefetch_through
                )

            obj_to_fetch = None
            if prefetcher is not None:
                obj_to_fetch = [obj for obj in obj_list if not is_fetched(obj)]

            if obj_to_fetch:
                obj_list, additional_lookups = prefetch_one_level(
                    obj_to_fetch,
                    prefetcher,
                    lookup,
                    level,
                )
                # We need to ensure we don't keep adding lookups from the
                # same relationships to stop infinite recursion. So, if we
                # are already on an automatically added lookup, don't add
                # the new lookups from relationships we've seen already.
                if not (
                    prefetch_to in done_queries
                    and lookup in auto_lookups
                    and descriptor in followed_descriptors
                ):
                    done_queries[prefetch_to] = obj_list
                    new_lookups = normalize_prefetch_lookups(
                        reversed(additional_lookups), prefetch_to
                    )
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
                    if through_attr in getattr(obj, "_prefetched_objects_cache", ()):
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
### 3 - django/db/models/query.py:

Start line: 2499, End line: 2531

```python
def prefetch_one_level(instances, prefetcher, lookup, level):
    # ... other code

    for obj in instances:
        instance_attr_val = instance_attr(obj)
        vals = rel_obj_cache.get(instance_attr_val, [])

        if single:
            val = vals[0] if vals else None
            if as_attr:
                # A to_attr has been given for the prefetch.
                setattr(obj, to_attr, val)
            elif is_descriptor:
                # cache_name points to a field name in obj.
                # This field is a descriptor for a related object.
                setattr(obj, cache_name, val)
            else:
                # No to_attr has been given for this prefetch operation and the
                # cache_name does not point to a descriptor. Store the value of
                # the field in the object's field cache.
                obj._state.fields_cache[cache_name] = val
        else:
            if as_attr:
                setattr(obj, to_attr, vals)
            else:
                manager = getattr(obj, to_attr)
                if leaf and lookup.queryset is not None:
                    qs = manager._apply_rel_filters(lookup.queryset)
                else:
                    qs = manager.get_queryset()
                qs._result_cache = vals
                # We don't want the individual qs doing prefetch_related now,
                # since we have merged this into the current work.
                qs._prefetch_done = True
                obj._prefetched_objects_cache[cache_name] = qs
    return all_related_objects, additional_lookups
```
### 4 - django/db/models/query.py:

Start line: 2428, End line: 2497

```python
def prefetch_one_level(instances, prefetcher, lookup, level):
    """
    Helper function for prefetch_related_objects().

    Run prefetches on all instances using the prefetcher object,
    assigning results to relevant caches in instance.

    Return the prefetched objects along with any additional prefetches that
    must be done due to prefetch_related lookups found from default managers.
    """
    # prefetcher must have a method get_prefetch_queryset() which takes a list
    # of instances, and returns a tuple:

    # (queryset of instances of self.model that are related to passed in instances,
    #  callable that gets value to be matched for returned instances,
    #  callable that gets value to be matched for passed in instances,
    #  boolean that is True for singly related objects,
    #  cache or field name to assign to,
    #  boolean that is True when the previous argument is a cache name vs a field name).

    # The 'values to be matched' must be hashable as they will be used
    # in a dictionary.

    (
        rel_qs,
        rel_obj_attr,
        instance_attr,
        single,
        cache_name,
        is_descriptor,
    ) = prefetcher.get_prefetch_queryset(instances, lookup.get_current_queryset(level))
    # We have to handle the possibility that the QuerySet we just got back
    # contains some prefetch_related lookups. We don't want to trigger the
    # prefetch_related functionality by evaluating the query. Rather, we need
    # to merge in the prefetch_related lookups.
    # Copy the lookups in case it is a Prefetch object which could be reused
    # later (happens in nested prefetch_related).
    additional_lookups = [
        copy.copy(additional_lookup)
        for additional_lookup in getattr(rel_qs, "_prefetch_related_lookups", ())
    ]
    if additional_lookups:
        # Don't need to clone because the manager should have given us a fresh
        # instance, so we access an internal instead of using public interface
        # for performance reasons.
        rel_qs._prefetch_related_lookups = ()

    all_related_objects = list(rel_qs)

    rel_obj_cache = {}
    for rel_obj in all_related_objects:
        rel_attr_val = rel_obj_attr(rel_obj)
        rel_obj_cache.setdefault(rel_attr_val, []).append(rel_obj)

    to_attr, as_attr = lookup.get_current_to_attr(level)
    # Make sure `to_attr` does not conflict with a field.
    if as_attr and instances:
        # We assume that objects retrieved are homogeneous (which is the premise
        # of prefetch_related), so what applies to first object applies to all.
        model = instances[0].__class__
        try:
            model._meta.get_field(to_attr)
        except exceptions.FieldDoesNotExist:
            pass
        else:
            msg = "to_attr={} conflicts with a field on the {} model."
            raise ValueError(msg.format(to_attr, model.__name__))

    # Whether or not we're prefetching the last part of the lookup.
    leaf = len(lookup.prefetch_through.split(LOOKUP_SEP)) - 1 == level
    # ... other code
```
### 5 - django/db/models/query.py:

Start line: 1553, End line: 1576

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
### 6 - django/db/models/query.py:

Start line: 2203, End line: 2231

```python
def normalize_prefetch_lookups(lookups, prefix=None):
    """Normalize lookups into Prefetch objects."""
    ret = []
    for lookup in lookups:
        if not isinstance(lookup, Prefetch):
            lookup = Prefetch(lookup)
        if prefix:
            lookup.add_prefix(prefix)
        ret.append(lookup)
    return ret


def prefetch_related_objects(model_instances, *related_lookups):
    """
    Populate prefetched object caches for a list of model instances based on
    the lookups/Prefetch instances given.
    """
    if not model_instances:
        return  # nothing to do

    # We need to be able to dynamically add to the list of prefetch_related
    # lookups that we look up (see below).  So we need some book keeping to
    # ensure we don't do duplicate work.
    done_queries = {}  # dictionary of things like 'foo__bar': [results]

    auto_lookups = set()  # we add to this as we go through.
    followed_descriptors = set()  # recursion protection

    all_lookups = normalize_prefetch_lookups(reversed(related_lookups))
    # ... other code
```
### 7 - django/db/models/query.py:

Start line: 2369, End line: 2425

```python
def get_prefetcher(instance, through_attr, to_attr):
    """
    For the attribute 'through_attr' on the given instance, find
    an object that has a get_prefetch_queryset().
    Return a 4 tuple containing:
    (the object with get_prefetch_queryset (or None),
     the descriptor object representing this relationship (or None),
     a boolean that is False if the attribute was not found at all,
     a function that takes an instance and returns a boolean that is True if
     the attribute has already been fetched for that instance)
    """

    def has_to_attr_attribute(instance):
        return hasattr(instance, to_attr)

    prefetcher = None
    is_fetched = has_to_attr_attribute

    # For singly related objects, we have to avoid getting the attribute
    # from the object, as this will trigger the query. So we first try
    # on the class, in order to get the descriptor object.
    rel_obj_descriptor = getattr(instance.__class__, through_attr, None)
    if rel_obj_descriptor is None:
        attr_found = hasattr(instance, through_attr)
    else:
        attr_found = True
        if rel_obj_descriptor:
            # singly related object, descriptor object has the
            # get_prefetch_queryset() method.
            if hasattr(rel_obj_descriptor, "get_prefetch_queryset"):
                prefetcher = rel_obj_descriptor
                is_fetched = rel_obj_descriptor.is_cached
            else:
                # descriptor doesn't support prefetching, so we go ahead and get
                # the attribute on the instance rather than the class to
                # support many related managers
                rel_obj = getattr(instance, through_attr)
                if hasattr(rel_obj, "get_prefetch_queryset"):
                    prefetcher = rel_obj
                if through_attr != to_attr:
                    # Special case cached_property instances because hasattr
                    # triggers attribute computation and assignment.
                    if isinstance(
                        getattr(instance.__class__, to_attr, None), cached_property
                    ):

                        def has_cached_property(instance):
                            return to_attr in instance.__dict__

                        is_fetched = has_cached_property
                else:

                    def in_prefetched_cache(instance):
                        return through_attr in instance._prefetched_objects_cache

                    is_fetched = in_prefetched_cache
    return prefetcher, rel_obj_descriptor, attr_found, is_fetched
```
### 8 - django/db/models/query.py:

Start line: 1695, End line: 1710

```python
class QuerySet:

    def defer(self, *fields):
        """
        Defer the loading of data for certain fields until they are accessed.
        Add the set of deferred fields to any existing set of deferred fields.
        The only exception to this is if None is passed in as the only
        parameter, in which case removal all deferrals.
        """
        self._not_support_combined_queries("defer")
        if self._fields is not None:
            raise TypeError("Cannot call defer() after .values() or .values_list()")
        clone = self._chain()
        if fields == (None,):
            clone.query.clear_deferred_loading()
        else:
            clone.query.add_deferred_loading(fields)
        return clone
```
### 9 - django/db/models/query.py:

Start line: 501, End line: 518

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
### 10 - django/db/models/query.py:

Start line: 1262, End line: 1309

```python
class QuerySet:

    async def acontains(self, obj):
        return await sync_to_async(self.contains)(obj=obj)

    def _prefetch_related_objects(self):
        # This method can only be called once the result cache has been filled.
        prefetch_related_objects(self._result_cache, *self._prefetch_related_lookups)
        self._prefetch_done = True

    def explain(self, *, format=None, **options):
        """
        Runs an EXPLAIN on the SQL query this QuerySet would perform, and
        returns the results.
        """
        return self.query.explain(using=self.db, format=format, **options)

    async def aexplain(self, *, format=None, **options):
        return await sync_to_async(self.explain)(format=format, **options)

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
### 14 - django/db/models/fields/related_descriptors.py:

Start line: 398, End line: 421

```python
class ReverseOneToOneDescriptor:

    def get_prefetch_queryset(self, instances, queryset=None):
        if queryset is None:
            queryset = self.get_queryset()
        queryset._add_hints(instance=instances[0])

        rel_obj_attr = self.related.field.get_local_related_value
        instance_attr = self.related.field.get_foreign_related_value
        instances_dict = {instance_attr(inst): inst for inst in instances}
        query = {"%s__in" % self.related.field.name: instances}
        queryset = queryset.filter(**query)

        # Since we're going to assign directly in the cache,
        # we must manage the reverse relation cache manually.
        for rel_obj in queryset:
            instance = instances_dict[rel_obj_attr(rel_obj)]
            self.related.field.set_cached_value(rel_obj, instance)
        return (
            queryset,
            rel_obj_attr,
            instance_attr,
            True,
            self.related.get_cache_name(),
            False,
        )
```
### 20 - django/db/models/fields/related_descriptors.py:

Start line: 124, End line: 168

```python
class ForwardManyToOneDescriptor:

    def get_prefetch_queryset(self, instances, queryset=None):
        if queryset is None:
            queryset = self.get_queryset()
        queryset._add_hints(instance=instances[0])

        rel_obj_attr = self.field.get_foreign_related_value
        instance_attr = self.field.get_local_related_value
        instances_dict = {instance_attr(inst): inst for inst in instances}
        related_field = self.field.foreign_related_fields[0]
        remote_field = self.field.remote_field

        # FIXME: This will need to be revisited when we introduce support for
        # composite fields. In the meantime we take this practical approach to
        # solve a regression on 1.6 when the reverse manager in hidden
        # (related_name ends with a '+'). Refs #21410.
        # The check for len(...) == 1 is a special case that allows the query
        # to be join-less and smaller. Refs #21760.
        if remote_field.is_hidden() or len(self.field.foreign_related_fields) == 1:
            query = {
                "%s__in"
                % related_field.name: {instance_attr(inst)[0] for inst in instances}
            }
        else:
            query = {"%s__in" % self.field.related_query_name(): instances}
        queryset = queryset.filter(**query)

        # Since we're going to assign directly in the cache,
        # we must manage the reverse relation cache manually.
        if not remote_field.multiple:
            for rel_obj in queryset:
                instance = instances_dict[rel_obj_attr(rel_obj)]
                remote_field.set_cached_value(rel_obj, instance)
        return (
            queryset,
            rel_obj_attr,
            instance_attr,
            True,
            self.field.get_cache_name(),
            False,
        )

    def get_object(self, instance):
        qs = self.get_queryset(instance=instance)
        # Assuming the database enforces foreign keys, this won't fail.
        return qs.get(self.field.get_reverse_related_filter(instance))
```
