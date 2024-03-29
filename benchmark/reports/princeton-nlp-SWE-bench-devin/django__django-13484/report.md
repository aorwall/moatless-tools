# django__django-13484

| **django/django** | `292b3be698ef58aff9c215d62a444f66ead578c3` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | - |
| **Avg pos** | - |
| **Min pos** | - |
| **Max pos** | - |
| **Top file pos** | - |
| **Missing snippets** | 4 |
| **Missing patch files** | 1 |


## Expected patch

```diff
diff --git a/django/db/models/fields/reverse_related.py b/django/db/models/fields/reverse_related.py
--- a/django/db/models/fields/reverse_related.py
+++ b/django/db/models/fields/reverse_related.py
@@ -11,6 +11,7 @@
 
 from django.core import exceptions
 from django.utils.functional import cached_property
+from django.utils.hashable import make_hashable
 
 from . import BLANK_CHOICE_DASH
 from .mixins import FieldCacheMixin
@@ -115,6 +116,28 @@ def __repr__(self):
             self.related_model._meta.model_name,
         )
 
+    @property
+    def identity(self):
+        return (
+            self.field,
+            self.model,
+            self.related_name,
+            self.related_query_name,
+            tuple(sorted(make_hashable(self.limit_choices_to))),
+            self.parent_link,
+            self.on_delete,
+            self.symmetrical,
+            self.multiple,
+        )
+
+    def __eq__(self, other):
+        if not isinstance(other, self.__class__):
+            return NotImplemented
+        return self.identity == other.identity
+
+    def __hash__(self):
+        return hash(self.identity)
+
     def get_choices(
         self, include_blank=True, blank_choice=BLANK_CHOICE_DASH,
         limit_choices_to=None, ordering=(),
@@ -215,6 +238,10 @@ def __getstate__(self):
         state.pop('related_model', None)
         return state
 
+    @property
+    def identity(self):
+        return super().identity + (self.field_name,)
+
     def get_related_field(self):
         """
         Return the Field in the 'to' object to which this relationship is tied.
@@ -279,6 +306,14 @@ def __init__(self, field, to, related_name=None, related_query_name=None,
         self.symmetrical = symmetrical
         self.db_constraint = db_constraint
 
+    @property
+    def identity(self):
+        return super().identity + (
+            self.through,
+            self.through_fields,
+            self.db_constraint,
+        )
+
     def get_related_field(self):
         """
         Return the field in the 'to' object to which this relationship is tied.

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/db/models/fields/reverse_related.py | 14 | 14 | - | - | -
| django/db/models/fields/reverse_related.py | 118 | 118 | - | - | -
| django/db/models/fields/reverse_related.py | 218 | 218 | - | - | -
| django/db/models/fields/reverse_related.py | 282 | 282 | - | - | -


## Problem Statement

```
Queryset crashes when recreated from a pickled query with FilteredRelation used in aggregation.
Description
	
I am pickling query objects (queryset.query) for later re-evaluation as per ​https://docs.djangoproject.com/en/2.2/ref/models/querysets/#pickling-querysets. However, when I tried to rerun a query that contains a FilteredRelation inside a filter, I get an psycopg2.errors.UndefinedTable: missing FROM-clause entry for table "t3" error.
I created a minimum reproducible example.
models.py
from django.db import models
class Publication(models.Model):
	title = models.CharField(max_length=64)
class Session(models.Model):
	TYPE_CHOICES = (('A', 'A'), ('B', 'B'))
	publication = models.ForeignKey(Publication, on_delete=models.CASCADE)
	session_type = models.CharField(choices=TYPE_CHOICES, default='A', max_length=1)
	place = models.CharField(max_length=16)
	value = models.PositiveIntegerField(default=1)
The actual code to cause the crash:
import pickle
from django.db.models import FilteredRelation, Q, Sum
from django_error.models import Publication, Session
p1 = Publication.objects.create(title='Foo')
p2 = Publication.objects.create(title='Bar')
Session.objects.create(publication=p1, session_type='A', place='X', value=1)
Session.objects.create(publication=p1, session_type='B', place='X', value=2)
Session.objects.create(publication=p2, session_type='A', place='X', value=4)
Session.objects.create(publication=p2, session_type='B', place='X', value=8)
Session.objects.create(publication=p1, session_type='A', place='Y', value=1)
Session.objects.create(publication=p1, session_type='B', place='Y', value=2)
Session.objects.create(publication=p2, session_type='A', place='Y', value=4)
Session.objects.create(publication=p2, session_type='B', place='Y', value=8)
qs = Publication.objects.all().annotate(
	relevant_sessions=FilteredRelation('session', condition=Q(session__session_type='A'))
).annotate(x=Sum('relevant_sessions__value'))
# just print it out to make sure the query works
print(list(qs))
qs2 = Publication.objects.all()
qs2.query = pickle.loads(pickle.dumps(qs.query))
# the following crashes with an error
#	 psycopg2.errors.UndefinedTable: missing FROM-clause entry for table "t3"
#	 LINE 1: ...n"."id" = relevant_sessions."publication_id" AND (T3."sessio...
print(list(qs2))
In the crashing query, there seems to be a difference in the table_map attribute - this is probably where the t3 table is coming from.
Please let me know if there is any more info required for hunting this down.
Cheers
Beda
p.s.- I also tried in Django 3.1 and the behavior is the same.
p.p.s.- just to make sure, I am not interested in ideas on how to rewrite the query - the above is a very simplified version of what I use, so it would probably not be applicable anyway.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 django/db/models/sql/query.py | 1428 | 1455| 283 | 283 | 22490 | 
| 2 | 2 django/db/models/query.py | 236 | 263| 221 | 504 | 39634 | 
| 3 | 2 django/db/models/sql/query.py | 1294 | 1359| 772 | 1276 | 39634 | 
| 4 | 2 django/db/models/query.py | 794 | 833| 322 | 1598 | 39634 | 
| 5 | 2 django/db/models/sql/query.py | 1406 | 1426| 212 | 1810 | 39634 | 
| 6 | 2 django/db/models/sql/query.py | 1115 | 1147| 338 | 2148 | 39634 | 
| 7 | 2 django/db/models/sql/query.py | 1636 | 1660| 227 | 2375 | 39634 | 
| 8 | 2 django/db/models/query.py | 1087 | 1128| 323 | 2698 | 39634 | 
| 9 | 2 django/db/models/sql/query.py | 1702 | 1741| 439 | 3137 | 39634 | 
| 10 | 2 django/db/models/sql/query.py | 1473 | 1558| 801 | 3938 | 39634 | 
| 11 | 2 django/db/models/query.py | 1309 | 1327| 186 | 4124 | 39634 | 
| 12 | 2 django/db/models/query.py | 320 | 346| 222 | 4346 | 39634 | 
| 13 | 2 django/db/models/sql/query.py | 1847 | 1896| 330 | 4676 | 39634 | 
| 14 | 2 django/db/models/sql/query.py | 1743 | 1814| 784 | 5460 | 39634 | 
| 15 | 2 django/db/models/query.py | 1340 | 1388| 405 | 5865 | 39634 | 
| 16 | 2 django/db/models/query.py | 714 | 741| 235 | 6100 | 39634 | 
| 17 | 3 django/db/models/aggregates.py | 70 | 96| 266 | 6366 | 40935 | 
| 18 | 3 django/db/models/query.py | 978 | 1009| 341 | 6707 | 40935 | 
| 19 | 4 django/db/models/query_utils.py | 312 | 352| 286 | 6993 | 43641 | 
| 20 | 4 django/db/models/sql/query.py | 1361 | 1382| 250 | 7243 | 43641 | 
| 21 | 5 django/db/models/__init__.py | 1 | 53| 619 | 7862 | 44260 | 
| 22 | 5 django/db/models/sql/query.py | 899 | 921| 248 | 8110 | 44260 | 
| 23 | 5 django/db/models/aggregates.py | 45 | 68| 294 | 8404 | 44260 | 
| 24 | 5 django/db/models/query.py | 1049 | 1070| 214 | 8618 | 44260 | 
| 25 | 5 django/db/models/sql/query.py | 1384 | 1404| 247 | 8865 | 44260 | 
| 26 | 5 django/db/models/query.py | 1072 | 1085| 115 | 8980 | 44260 | 
| 27 | 6 django/db/models/fields/related.py | 1 | 34| 246 | 9226 | 58136 | 
| 28 | 6 django/db/models/sql/query.py | 364 | 414| 494 | 9720 | 58136 | 
| 29 | 6 django/db/models/query.py | 1329 | 1338| 114 | 9834 | 58136 | 
| 30 | 7 django/db/models/sql/compiler.py | 1 | 19| 170 | 10004 | 72406 | 
| 31 | 7 django/db/models/sql/query.py | 2180 | 2225| 371 | 10375 | 72406 | 
| 32 | 7 django/db/models/query.py | 1963 | 1986| 200 | 10575 | 72406 | 
| 33 | 7 django/db/models/sql/query.py | 1 | 65| 465 | 11040 | 72406 | 
| 34 | 7 django/db/models/sql/query.py | 703 | 738| 389 | 11429 | 72406 | 
| 35 | 7 django/db/models/query.py | 1646 | 1752| 1063 | 12492 | 72406 | 
| 36 | 7 django/db/models/sql/compiler.py | 885 | 977| 839 | 13331 | 72406 | 
| 37 | 7 django/db/models/query.py | 1474 | 1507| 297 | 13628 | 72406 | 
| 38 | 8 django/db/models/sql/subqueries.py | 137 | 163| 173 | 13801 | 73619 | 
| 39 | 8 django/db/models/query_utils.py | 25 | 54| 185 | 13986 | 73619 | 
| 40 | 9 django/db/models/deletion.py | 346 | 359| 116 | 14102 | 77445 | 
| 41 | 10 django/contrib/postgres/search.py | 160 | 195| 313 | 14415 | 79667 | 
| 42 | 11 django/db/models/fields/related_lookups.py | 121 | 157| 244 | 14659 | 81120 | 
| 43 | 11 django/db/models/fields/related_lookups.py | 62 | 101| 451 | 15110 | 81120 | 
| 44 | 11 django/db/models/sql/subqueries.py | 1 | 44| 320 | 15430 | 81120 | 
| 45 | 11 django/db/models/fields/related.py | 127 | 154| 201 | 15631 | 81120 | 
| 46 | 12 django/db/models/sql/datastructures.py | 117 | 137| 144 | 15775 | 82522 | 
| 47 | 12 django/db/models/fields/related.py | 1235 | 1352| 963 | 16738 | 82522 | 
| 48 | 12 django/db/models/query.py | 909 | 959| 371 | 17109 | 82522 | 
| 49 | 12 django/db/models/query.py | 1027 | 1047| 155 | 17264 | 82522 | 
| 50 | 13 django/contrib/gis/db/models/__init__.py | 1 | 19| 204 | 17468 | 82726 | 
| 51 | 13 django/db/models/query.py | 1 | 39| 294 | 17762 | 82726 | 
| 52 | 13 django/db/models/sql/query.py | 2038 | 2084| 370 | 18132 | 82726 | 
| 53 | 13 django/db/models/query.py | 1130 | 1166| 324 | 18456 | 82726 | 
| 54 | 13 django/db/models/deletion.py | 269 | 344| 798 | 19254 | 82726 | 
| 55 | 13 django/db/models/query.py | 659 | 673| 132 | 19386 | 82726 | 
| 56 | 13 django/db/models/sql/query.py | 136 | 230| 833 | 20219 | 82726 | 
| 57 | 14 django/db/models/sql/where.py | 233 | 249| 130 | 20349 | 84540 | 
| 58 | 14 django/db/models/deletion.py | 1 | 76| 566 | 20915 | 84540 | 
| 59 | 14 django/db/models/sql/compiler.py | 63 | 147| 881 | 21796 | 84540 | 
| 60 | 14 django/db/models/sql/subqueries.py | 47 | 75| 210 | 22006 | 84540 | 
| 61 | 15 django/db/models/base.py | 2038 | 2089| 351 | 22357 | 101184 | 
| 62 | 15 django/db/models/query.py | 175 | 234| 469 | 22826 | 101184 | 
| 63 | 15 django/db/models/query.py | 961 | 976| 124 | 22950 | 101184 | 
| 64 | 16 django/db/backends/base/schema.py | 31 | 41| 120 | 23070 | 113437 | 
| 65 | 16 django/db/models/query.py | 835 | 864| 248 | 23318 | 113437 | 
| 66 | 16 django/db/models/query_utils.py | 110 | 124| 157 | 23475 | 113437 | 
| 67 | 16 django/db/models/query_utils.py | 1 | 22| 178 | 23653 | 113437 | 
| 68 | 16 django/db/models/query_utils.py | 57 | 108| 396 | 24049 | 113437 | 
| 69 | 16 django/db/models/sql/query.py | 628 | 652| 269 | 24318 | 113437 | 
| 70 | 16 django/db/models/sql/query.py | 1987 | 2036| 420 | 24738 | 113437 | 
| 71 | 16 django/db/models/query.py | 265 | 285| 180 | 24918 | 113437 | 
| 72 | 16 django/db/models/sql/compiler.py | 149 | 197| 523 | 25441 | 113437 | 
| 73 | 16 django/db/models/sql/query.py | 118 | 133| 145 | 25586 | 113437 | 
| 74 | 16 django/db/models/sql/query.py | 1059 | 1084| 214 | 25800 | 113437 | 
| 75 | 16 django/db/models/sql/query.py | 1086 | 1113| 285 | 26085 | 113437 | 
| 76 | 16 django/db/models/query.py | 1558 | 1614| 481 | 26566 | 113437 | 
| 77 | 16 django/db/models/fields/related.py | 255 | 282| 269 | 26835 | 113437 | 
| 78 | 16 django/db/models/sql/query.py | 232 | 286| 400 | 27235 | 113437 | 
| 79 | 16 django/db/models/query.py | 348 | 363| 146 | 27381 | 113437 | 
| 80 | 16 django/db/models/sql/compiler.py | 1038 | 1078| 337 | 27718 | 113437 | 
| 81 | 16 django/db/models/query.py | 743 | 776| 274 | 27992 | 113437 | 
| 82 | 16 django/db/models/sql/subqueries.py | 111 | 134| 192 | 28184 | 113437 | 
| 83 | 16 django/db/models/query.py | 1206 | 1234| 183 | 28367 | 113437 | 
| 84 | 16 django/db/models/query.py | 1434 | 1472| 308 | 28675 | 113437 | 
| 85 | 16 django/db/models/sql/compiler.py | 199 | 269| 580 | 29255 | 113437 | 
| 86 | 16 django/db/models/query.py | 1902 | 1961| 772 | 30027 | 113437 | 
| 87 | 16 django/db/models/sql/datastructures.py | 104 | 115| 133 | 30160 | 113437 | 
| 88 | 17 django/contrib/postgres/aggregates/statistics.py | 1 | 66| 419 | 30579 | 113856 | 
| 89 | 17 django/db/models/sql/query.py | 288 | 337| 444 | 31023 | 113856 | 
| 90 | 17 django/db/models/sql/compiler.py | 22 | 47| 257 | 31280 | 113856 | 
| 91 | 17 django/db/models/sql/query.py | 339 | 362| 179 | 31459 | 113856 | 
| 92 | 17 django/contrib/postgres/search.py | 130 | 157| 248 | 31707 | 113856 | 
| 93 | 17 django/db/models/sql/query.py | 654 | 701| 511 | 32218 | 113856 | 
| 94 | 17 django/db/models/fields/related.py | 190 | 254| 673 | 32891 | 113856 | 
| 95 | 18 django/contrib/postgres/aggregates/mixins.py | 22 | 34| 128 | 33019 | 114298 | 
| 96 | 19 django/db/models/fields/related_descriptors.py | 365 | 381| 184 | 33203 | 124686 | 
| 97 | 19 django/db/models/query.py | 1509 | 1555| 336 | 33539 | 124686 | 
| 98 | 19 django/db/models/sql/query.py | 511 | 551| 325 | 33864 | 124686 | 
| 99 | 19 django/db/models/fields/related_lookups.py | 104 | 119| 215 | 34079 | 124686 | 
| 100 | 19 django/db/models/deletion.py | 379 | 448| 580 | 34659 | 124686 | 
| 101 | 19 django/db/models/query.py | 1283 | 1307| 210 | 34869 | 124686 | 
| 102 | 20 django/contrib/gis/db/models/aggregates.py | 29 | 46| 216 | 35085 | 125303 | 
| 103 | 20 django/db/models/deletion.py | 361 | 377| 130 | 35215 | 125303 | 
| 104 | 20 django/db/models/fields/related_lookups.py | 46 | 60| 224 | 35439 | 125303 | 
| 105 | 20 django/db/models/sql/query.py | 991 | 1022| 307 | 35746 | 125303 | 
| 106 | 21 django/db/models/manager.py | 168 | 205| 211 | 35957 | 126756 | 
| 107 | 21 django/db/models/sql/compiler.py | 1199 | 1220| 223 | 36180 | 126756 | 
| 108 | 22 django/core/serializers/__init__.py | 86 | 141| 369 | 36549 | 128513 | 
| 109 | 23 django/contrib/admin/views/main.py | 123 | 212| 861 | 37410 | 132909 | 
| 110 | 23 django/db/models/sql/query.py | 2086 | 2108| 249 | 37659 | 132909 | 
| 111 | 23 django/db/models/query.py | 1755 | 1799| 439 | 38098 | 132909 | 
| 112 | 24 django/core/management/commands/inspectdb.py | 38 | 173| 1291 | 39389 | 135542 | 
| 113 | 25 django/db/backends/postgresql/features.py | 1 | 97| 790 | 40179 | 136332 | 
| 114 | 25 django/db/models/fields/related_lookups.py | 1 | 23| 170 | 40349 | 136332 | 
| 115 | 25 django/db/models/sql/compiler.py | 803 | 883| 717 | 41066 | 136332 | 
| 116 | 25 django/db/models/fields/related.py | 320 | 341| 225 | 41291 | 136332 | 
| 117 | 25 django/db/models/sql/datastructures.py | 140 | 169| 231 | 41522 | 136332 | 
| 118 | 25 django/db/models/fields/related_descriptors.py | 344 | 363| 156 | 41678 | 136332 | 
| 119 | 26 django/contrib/admin/filters.py | 162 | 207| 427 | 42105 | 140455 | 
| 120 | 26 django/db/models/sql/query.py | 553 | 627| 809 | 42914 | 140455 | 
| 121 | 26 django/db/models/sql/compiler.py | 271 | 358| 712 | 43626 | 140455 | 
| 122 | 26 django/db/models/query.py | 1185 | 1204| 209 | 43835 | 140455 | 
| 123 | 26 django/db/models/fields/related.py | 576 | 609| 334 | 44169 | 140455 | 
| 124 | 27 django/contrib/admin/options.py | 377 | 429| 504 | 44673 | 159042 | 
| 125 | 28 django/db/backends/sqlite3/schema.py | 225 | 307| 731 | 45404 | 163198 | 
| 126 | 28 django/contrib/postgres/aggregates/mixins.py | 1 | 20| 182 | 45586 | 163198 | 
| 127 | 29 django/db/models/lookups.py | 358 | 390| 294 | 45880 | 168151 | 
| 128 | 29 django/db/models/fields/related.py | 935 | 948| 126 | 46006 | 168151 | 
| 129 | 30 django/forms/models.py | 629 | 646| 167 | 46173 | 179925 | 
| 130 | 31 django/db/backends/oracle/creation.py | 130 | 165| 399 | 46572 | 183818 | 
| 131 | 31 django/contrib/postgres/search.py | 1 | 24| 205 | 46777 | 183818 | 
| 132 | 31 django/contrib/gis/db/models/aggregates.py | 49 | 84| 207 | 46984 | 183818 | 
| 133 | 31 django/db/models/query_utils.py | 284 | 309| 293 | 47277 | 183818 | 
| 134 | 31 django/db/models/sql/query.py | 2227 | 2259| 228 | 47505 | 183818 | 
| 135 | 31 django/db/models/sql/compiler.py | 1540 | 1580| 409 | 47914 | 183818 | 
| 136 | 31 django/contrib/admin/filters.py | 209 | 226| 190 | 48104 | 183818 | 
| 137 | 32 django/db/backends/postgresql/schema.py | 1 | 67| 626 | 48730 | 185958 | 
| 138 | 33 django/db/models/options.py | 1 | 34| 285 | 49015 | 193064 | 
| 139 | 33 django/db/models/fields/related.py | 611 | 628| 197 | 49212 | 193064 | 
| 140 | 34 django/db/backends/postgresql/operations.py | 158 | 185| 311 | 49523 | 195580 | 
| 141 | 34 django/db/models/sql/compiler.py | 488 | 645| 1469 | 50992 | 195580 | 
| 142 | 35 django/views/generic/detail.py | 58 | 76| 154 | 51146 | 196895 | 
| 143 | 35 django/db/models/base.py | 1080 | 1123| 404 | 51550 | 196895 | 
| 144 | 35 django/db/models/sql/query.py | 68 | 116| 361 | 51911 | 196895 | 
| 145 | 35 django/contrib/admin/views/main.py | 496 | 527| 224 | 52135 | 196895 | 
| 146 | 35 django/db/models/fields/related.py | 864 | 890| 240 | 52375 | 196895 | 
| 147 | 35 django/db/models/deletion.py | 79 | 97| 199 | 52574 | 196895 | 
| 148 | 35 django/db/models/sql/query.py | 2387 | 2413| 176 | 52750 | 196895 | 
| 149 | 35 django/db/models/fields/related_descriptors.py | 883 | 904| 199 | 52949 | 196895 | 
| 150 | 35 django/contrib/admin/views/main.py | 442 | 494| 440 | 53389 | 196895 | 
| 151 | 35 django/db/models/sql/compiler.py | 1414 | 1428| 127 | 53516 | 196895 | 
| 152 | 36 django/db/backends/mysql/compiler.py | 1 | 14| 123 | 53639 | 197415 | 
| 153 | 36 django/db/models/query.py | 777 | 793| 157 | 53796 | 197415 | 
| 155 | 37 django/db/models/query.py | 1424 | 1432| 136 | 54225 | 199749 | 
| 156 | 37 django/contrib/gis/db/models/aggregates.py | 1 | 27| 199 | 54424 | 199749 | 
| 157 | 37 django/db/models/fields/related.py | 670 | 694| 218 | 54642 | 199749 | 
| 158 | 37 django/db/models/fields/related.py | 1539 | 1556| 184 | 54826 | 199749 | 
| 159 | 37 django/db/models/fields/related.py | 171 | 188| 166 | 54992 | 199749 | 


## Missing Patch Files

 * 1: django/db/models/fields/reverse_related.py

### Hint

```
Thanks for this ticket, I was able to reproduce this issue.
Tests.
Just a note, the failing queryset does not have to be constructed by setting the query param of a newly created queryset - like this: qs2 = Publication.objects.all() qs2.query = pickle.loads(pickle.dumps(qs.query)) The same problem occurs even if the whole queryset is pickled and unpickled and then a copy is created by calling .all(). qs2 = pickle.loads(pickle.dumps(qs)).all()
```

## Patch

```diff
diff --git a/django/db/models/fields/reverse_related.py b/django/db/models/fields/reverse_related.py
--- a/django/db/models/fields/reverse_related.py
+++ b/django/db/models/fields/reverse_related.py
@@ -11,6 +11,7 @@
 
 from django.core import exceptions
 from django.utils.functional import cached_property
+from django.utils.hashable import make_hashable
 
 from . import BLANK_CHOICE_DASH
 from .mixins import FieldCacheMixin
@@ -115,6 +116,28 @@ def __repr__(self):
             self.related_model._meta.model_name,
         )
 
+    @property
+    def identity(self):
+        return (
+            self.field,
+            self.model,
+            self.related_name,
+            self.related_query_name,
+            tuple(sorted(make_hashable(self.limit_choices_to))),
+            self.parent_link,
+            self.on_delete,
+            self.symmetrical,
+            self.multiple,
+        )
+
+    def __eq__(self, other):
+        if not isinstance(other, self.__class__):
+            return NotImplemented
+        return self.identity == other.identity
+
+    def __hash__(self):
+        return hash(self.identity)
+
     def get_choices(
         self, include_blank=True, blank_choice=BLANK_CHOICE_DASH,
         limit_choices_to=None, ordering=(),
@@ -215,6 +238,10 @@ def __getstate__(self):
         state.pop('related_model', None)
         return state
 
+    @property
+    def identity(self):
+        return super().identity + (self.field_name,)
+
     def get_related_field(self):
         """
         Return the Field in the 'to' object to which this relationship is tied.
@@ -279,6 +306,14 @@ def __init__(self, field, to, related_name=None, related_query_name=None,
         self.symmetrical = symmetrical
         self.db_constraint = db_constraint
 
+    @property
+    def identity(self):
+        return super().identity + (
+            self.through,
+            self.through_fields,
+            self.db_constraint,
+        )
+
     def get_related_field(self):
         """
         Return the field in the 'to' object to which this relationship is tied.

```

## Test Patch

```diff
diff --git a/tests/queryset_pickle/tests.py b/tests/queryset_pickle/tests.py
--- a/tests/queryset_pickle/tests.py
+++ b/tests/queryset_pickle/tests.py
@@ -219,6 +219,40 @@ def test_pickle_subquery_queryset_not_evaluated(self):
         with self.assertNumQueries(0):
             self.assert_pickles(groups)
 
+    def test_pickle_filteredrelation(self):
+        group = Group.objects.create(name='group')
+        event_1 = Event.objects.create(title='Big event', group=group)
+        event_2 = Event.objects.create(title='Small event', group=group)
+        Happening.objects.bulk_create([
+            Happening(event=event_1, number1=5),
+            Happening(event=event_2, number1=3),
+        ])
+        groups = Group.objects.annotate(
+            big_events=models.FilteredRelation(
+                'event',
+                condition=models.Q(event__title__startswith='Big'),
+            ),
+        ).annotate(sum_number=models.Sum('big_events__happening__number1'))
+        groups_query = pickle.loads(pickle.dumps(groups.query))
+        groups = Group.objects.all()
+        groups.query = groups_query
+        self.assertEqual(groups.get().sum_number, 5)
+
+    def test_pickle_filteredrelation_m2m(self):
+        group = Group.objects.create(name='group')
+        m2mmodel = M2MModel.objects.create()
+        m2mmodel.groups.add(group)
+        groups = Group.objects.annotate(
+            first_m2mmodels=models.FilteredRelation(
+                'm2mmodel',
+                condition=models.Q(m2mmodel__pk__lt=10),
+            ),
+        ).annotate(count_groups=models.Count('first_m2mmodels__groups'))
+        groups_query = pickle.loads(pickle.dumps(groups.query))
+        groups = Group.objects.all()
+        groups.query = groups_query
+        self.assertEqual(groups.get().count_groups, 1)
+
     def test_annotation_with_callable_default(self):
         # Happening.when has a callable default of datetime.datetime.now.
         qs = Happening.objects.annotate(latest_time=models.Max('when'))

```


## Code snippets

### 1 - django/db/models/sql/query.py:

Start line: 1428, End line: 1455

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
### 2 - django/db/models/query.py:

Start line: 236, End line: 263

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
### 3 - django/db/models/sql/query.py:

Start line: 1294, End line: 1359

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
### 4 - django/db/models/query.py:

Start line: 794, End line: 833

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
### 5 - django/db/models/sql/query.py:

Start line: 1406, End line: 1426

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
### 6 - django/db/models/sql/query.py:

Start line: 1115, End line: 1147

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
### 7 - django/db/models/sql/query.py:

Start line: 1636, End line: 1660

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
### 8 - django/db/models/query.py:

Start line: 1087, End line: 1128

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
### 9 - django/db/models/sql/query.py:

Start line: 1702, End line: 1741

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
### 10 - django/db/models/sql/query.py:

Start line: 1473, End line: 1558

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
