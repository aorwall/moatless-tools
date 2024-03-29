# django__django-13030

| **django/django** | `36db4dd937ae11c5b687c5d2e5fa3c27e4140001` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 2338 |
| **Any found context length** | 2338 |
| **Avg pos** | 6.0 |
| **Min pos** | 6 |
| **Max pos** | 6 |
| **Top file pos** | 3 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/db/models/lookups.py b/django/db/models/lookups.py
--- a/django/db/models/lookups.py
+++ b/django/db/models/lookups.py
@@ -366,10 +366,12 @@ def process_rhs(self, compiler, connection):
             )
 
         if self.rhs_is_direct_value():
+            # Remove None from the list as NULL is never equal to anything.
             try:
                 rhs = OrderedSet(self.rhs)
+                rhs.discard(None)
             except TypeError:  # Unhashable items in self.rhs
-                rhs = self.rhs
+                rhs = [r for r in self.rhs if r is not None]
 
             if not rhs:
                 raise EmptyResultSet

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/db/models/lookups.py | 369 | 369 | 6 | 3 | 2338


## Problem Statement

```
Avoid passing NULL to the IN lookup
Description
	
Currently prefetch_related on a FK passes the NULL through to the database for e.g. author_id IN (NULL, 2). Passing NULL is always unnecessary, since it's not allowed in FK's. There's a small risk from passing NULL that it could lead to incorrect with complex prefetch querysets using PK refs because of NULL's weirdness in SQL.
For example with these models:
from django.db import models
class Author(models.Model):
	pass
class Book(models.Model):
	author = models.ForeignKey(Author, null=True, on_delete=models.DO_NOTHING)
Prefetching authors on Books, when at least one Book has author=None, uses IN (..., NULL, ...) in the query:
In [1]: from example.core.models import Author, Book
In [2]: a1 = Author.objects.create()
In [3]: Book.objects.create(author=a1)
Out[3]: <Book: Book object (3)>
In [4]: Book.objects.create(author=None)
Out[4]: <Book: Book object (4)>
In [5]: Book.objects.prefetch_related('author')
Out[5]: <QuerySet [<Book: Book object (3)>, <Book: Book object (4)>]>
In [6]: from django.db import connection
In [7]: connection.queries
Out[7]:
[{'sql': 'INSERT INTO "core_author" ("id") VALUES (NULL)', 'time': '0.001'},
 {'sql': 'INSERT INTO "core_book" ("author_id") VALUES (2)', 'time': '0.001'},
 {'sql': 'INSERT INTO "core_book" ("author_id") VALUES (NULL)',
 'time': '0.001'},
 {'sql': 'SELECT "core_book"."id", "core_book"."author_id" FROM "core_book" LIMIT 21',
 'time': '0.000'},
 {'sql': 'SELECT "core_author"."id" FROM "core_author" WHERE "core_author"."id" IN (NULL, 2)',
 'time': '0.000'}]
Maybe this could generally be extended to use of __in with non-nullable fields?

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 django/db/models/fields/related_lookups.py | 62 | 99| 447 | 447 | 1449 | 
| 2 | 1 django/db/models/fields/related_lookups.py | 46 | 60| 224 | 671 | 1449 | 
| 3 | 2 django/db/models/query.py | 1640 | 1746| 1063 | 1734 | 18551 | 
| 4 | **3 django/db/models/lookups.py** | 487 | 508| 172 | 1906 | 23477 | 
| 5 | 4 django/db/models/fields/related.py | 837 | 858| 169 | 2075 | 37308 | 
| **-> 6 <-** | **4 django/db/models/lookups.py** | 356 | 386| 263 | 2338 | 37308 | 
| 7 | 4 django/db/models/query.py | 1861 | 1893| 314 | 2652 | 37308 | 
| 8 | 4 django/db/models/query.py | 1552 | 1608| 481 | 3133 | 37308 | 
| 9 | 4 django/db/models/fields/related_lookups.py | 102 | 117| 215 | 3348 | 37308 | 
| 10 | 4 django/db/models/fields/related.py | 746 | 764| 222 | 3570 | 37308 | 
| 11 | 4 django/db/models/fields/related_lookups.py | 119 | 155| 244 | 3814 | 37308 | 
| 12 | **4 django/db/models/lookups.py** | 388 | 417| 337 | 4151 | 37308 | 
| 13 | 5 django/db/backends/base/features.py | 1 | 111| 886 | 5037 | 39924 | 
| 14 | 5 django/db/models/query.py | 1796 | 1859| 658 | 5695 | 39924 | 
| 15 | 6 django/contrib/contenttypes/fields.py | 173 | 217| 411 | 6106 | 45357 | 
| 16 | 7 django/db/models/fields/__init__.py | 308 | 336| 205 | 6311 | 63046 | 
| 17 | 7 django/db/models/fields/related.py | 1231 | 1348| 963 | 7274 | 63046 | 
| 18 | 7 django/db/models/fields/related.py | 1198 | 1229| 180 | 7454 | 63046 | 
| 19 | 8 django/db/models/sql/query.py | 2283 | 2299| 177 | 7631 | 85083 | 
| 20 | 8 django/db/models/fields/related.py | 860 | 886| 240 | 7871 | 85083 | 
| 21 | 9 django/db/models/fields/related_descriptors.py | 120 | 154| 405 | 8276 | 95478 | 
| 22 | 10 django/db/models/fields/json.py | 383 | 393| 131 | 8407 | 99611 | 
| 23 | 10 django/db/models/query.py | 1611 | 1639| 246 | 8653 | 99611 | 
| 24 | 11 django/db/migrations/questioner.py | 162 | 185| 246 | 8899 | 101684 | 
| 25 | 12 django/db/models/sql/compiler.py | 803 | 883| 717 | 9616 | 115903 | 
| 26 | 12 django/db/models/sql/query.py | 696 | 731| 389 | 10005 | 115903 | 
| 27 | 12 django/db/models/fields/related.py | 993 | 1020| 215 | 10220 | 115903 | 
| 28 | 12 django/db/models/query.py | 1059 | 1080| 214 | 10434 | 115903 | 
| 29 | **12 django/db/models/lookups.py** | 188 | 205| 175 | 10609 | 115903 | 
| 30 | 12 django/db/models/fields/related.py | 946 | 978| 279 | 10888 | 115903 | 
| 31 | 12 django/db/models/fields/related_descriptors.py | 365 | 381| 184 | 11072 | 115903 | 
| 32 | 12 django/db/models/query.py | 1428 | 1466| 308 | 11380 | 115903 | 
| 33 | 12 django/db/models/fields/related.py | 509 | 574| 492 | 11872 | 115903 | 
| 34 | 12 django/db/models/fields/related.py | 1350 | 1422| 616 | 12488 | 115903 | 
| 35 | 12 django/db/models/fields/related.py | 909 | 929| 178 | 12666 | 115903 | 
| 36 | 12 django/db/models/fields/related.py | 767 | 835| 521 | 13187 | 115903 | 
| 37 | **12 django/db/models/lookups.py** | 288 | 300| 168 | 13355 | 115903 | 
| 38 | 12 django/db/models/fields/related.py | 487 | 507| 138 | 13493 | 115903 | 
| 39 | 12 django/db/migrations/questioner.py | 143 | 160| 183 | 13676 | 115903 | 
| 40 | 12 django/db/models/fields/__init__.py | 1928 | 1955| 234 | 13910 | 115903 | 
| 41 | **12 django/db/models/lookups.py** | 208 | 243| 308 | 14218 | 115903 | 
| 42 | 12 django/db/models/query.py | 1167 | 1182| 149 | 14367 | 115903 | 
| 43 | 12 django/db/models/sql/query.py | 1698 | 1769| 784 | 15151 | 115903 | 
| 44 | 12 django/db/models/fields/related.py | 127 | 154| 201 | 15352 | 115903 | 
| 45 | 12 django/db/models/fields/__init__.py | 1 | 81| 633 | 15985 | 115903 | 
| 46 | 12 django/contrib/contenttypes/fields.py | 20 | 108| 571 | 16556 | 115903 | 
| 47 | 12 django/db/models/fields/related.py | 284 | 318| 293 | 16849 | 115903 | 
| 48 | 12 django/db/models/sql/compiler.py | 885 | 977| 839 | 17688 | 115903 | 
| 49 | **12 django/db/models/lookups.py** | 303 | 353| 306 | 17994 | 115903 | 
| 50 | 13 django/contrib/admin/options.py | 375 | 427| 504 | 18498 | 134441 | 
| 51 | **13 django/db/models/lookups.py** | 1 | 40| 317 | 18815 | 134441 | 
| 52 | 13 django/contrib/contenttypes/fields.py | 219 | 270| 459 | 19274 | 134441 | 
| 53 | 13 django/db/models/sql/query.py | 1440 | 1518| 734 | 20008 | 134441 | 
| 54 | 14 django/db/models/__init__.py | 1 | 53| 619 | 20627 | 135060 | 
| 55 | 14 django/db/models/fields/related.py | 255 | 282| 269 | 20896 | 135060 | 
| 56 | 15 django/db/backends/base/schema.py | 31 | 41| 120 | 21016 | 146819 | 
| 57 | 16 django/db/models/fields/reverse_related.py | 117 | 134| 161 | 21177 | 148962 | 
| 58 | 16 django/db/models/fields/related.py | 648 | 664| 163 | 21340 | 148962 | 
| 59 | 17 django/db/models/base.py | 1073 | 1116| 404 | 21744 | 165057 | 
| 60 | 17 django/db/models/sql/query.py | 1284 | 1342| 711 | 22455 | 165057 | 
| 61 | 17 django/db/models/fields/related.py | 980 | 991| 128 | 22583 | 165057 | 
| 62 | 17 django/db/models/fields/related.py | 156 | 169| 144 | 22727 | 165057 | 
| 63 | 17 django/db/migrations/questioner.py | 56 | 81| 220 | 22947 | 165057 | 
| 64 | 18 django/contrib/admin/utils.py | 287 | 305| 175 | 23122 | 169209 | 
| 65 | 18 django/db/models/fields/related.py | 1602 | 1639| 484 | 23606 | 169209 | 
| 66 | 18 django/db/models/base.py | 404 | 503| 856 | 24462 | 169209 | 
| 67 | 18 django/db/models/fields/related.py | 108 | 125| 155 | 24617 | 169209 | 
| 68 | 18 django/db/backends/base/features.py | 113 | 215| 840 | 25457 | 169209 | 
| 69 | 18 django/db/models/query.py | 1749 | 1793| 439 | 25896 | 169209 | 
| 70 | 18 django/db/models/fields/related_descriptors.py | 1 | 79| 683 | 26579 | 169209 | 
| 71 | 18 django/db/models/fields/related.py | 1659 | 1693| 266 | 26845 | 169209 | 
| 72 | 19 django/db/backends/mysql/schema.py | 115 | 129| 201 | 27046 | 170705 | 
| 73 | 19 django/db/models/fields/related.py | 1424 | 1465| 418 | 27464 | 170705 | 
| 74 | 19 django/db/models/fields/related.py | 931 | 944| 126 | 27590 | 170705 | 
| 75 | 20 django/db/models/deletion.py | 1 | 76| 566 | 28156 | 174531 | 
| 76 | 20 django/db/models/fields/related.py | 444 | 485| 273 | 28429 | 174531 | 
| 77 | 21 django/db/models/query_utils.py | 284 | 309| 293 | 28722 | 177237 | 
| 78 | 21 django/db/models/fields/reverse_related.py | 136 | 154| 172 | 28894 | 177237 | 
| 79 | 21 django/db/models/fields/related.py | 607 | 624| 197 | 29091 | 177237 | 
| 80 | 21 django/db/models/fields/related.py | 692 | 704| 116 | 29207 | 177237 | 
| 81 | 22 django/contrib/admin/filters.py | 162 | 207| 427 | 29634 | 181330 | 
| 82 | 23 django/db/backends/sqlite3/introspection.py | 224 | 238| 146 | 29780 | 185179 | 
| 83 | 23 django/db/models/fields/related_lookups.py | 1 | 23| 170 | 29950 | 185179 | 
| 84 | 23 django/db/models/fields/related.py | 1023 | 1070| 368 | 30318 | 185179 | 
| 85 | 23 django/db/models/sql/query.py | 1662 | 1696| 399 | 30717 | 185179 | 
| 86 | 23 django/db/models/fields/related.py | 626 | 646| 168 | 30885 | 185179 | 
| 87 | 23 django/db/models/fields/__init__.py | 367 | 393| 199 | 31084 | 185179 | 
| 88 | 23 django/db/models/fields/related_descriptors.py | 907 | 944| 374 | 31458 | 185179 | 
| 89 | 23 django/db/models/query.py | 1957 | 1980| 200 | 31658 | 185179 | 
| 90 | 23 django/db/models/fields/related_descriptors.py | 309 | 323| 182 | 31840 | 185179 | 
| 91 | 23 django/db/backends/base/schema.py | 1071 | 1093| 199 | 32039 | 185179 | 
| 92 | 23 django/db/models/fields/related_descriptors.py | 344 | 363| 156 | 32195 | 185179 | 
| 93 | 23 django/db/models/fields/related.py | 171 | 188| 166 | 32361 | 185179 | 
| 94 | 23 django/db/models/base.py | 1971 | 2022| 351 | 32712 | 185179 | 
| 95 | 23 django/db/models/fields/related_descriptors.py | 672 | 730| 548 | 33260 | 185179 | 
| 96 | 23 django/db/backends/mysql/schema.py | 131 | 149| 192 | 33452 | 185179 | 
| 97 | 23 django/db/models/sql/query.py | 1105 | 1137| 338 | 33790 | 185179 | 
| 98 | 23 django/db/models/fields/related.py | 190 | 254| 673 | 34463 | 185179 | 
| 99 | 24 django/db/backends/mysql/features.py | 1 | 104| 828 | 35291 | 186619 | 
| 100 | 24 django/db/models/sql/query.py | 2082 | 2099| 156 | 35447 | 186619 | 
| 101 | 24 django/db/models/sql/compiler.py | 1036 | 1076| 337 | 35784 | 186619 | 
| 102 | 24 django/contrib/admin/options.py | 429 | 472| 350 | 36134 | 186619 | 
| 103 | 24 django/db/models/fields/json.py | 214 | 232| 232 | 36366 | 186619 | 
| 104 | 24 django/db/models/fields/reverse_related.py | 19 | 115| 635 | 37001 | 186619 | 
| 105 | 24 django/db/models/fields/json.py | 235 | 276| 318 | 37319 | 186619 | 
| 106 | 24 django/db/models/sql/compiler.py | 1536 | 1576| 409 | 37728 | 186619 | 
| 107 | 24 django/db/models/base.py | 954 | 968| 212 | 37940 | 186619 | 
| 108 | 24 django/db/models/base.py | 1394 | 1449| 491 | 38431 | 186619 | 
| 109 | 24 django/db/models/fields/related.py | 1120 | 1196| 524 | 38955 | 186619 | 
| 110 | **24 django/db/models/lookups.py** | 119 | 142| 220 | 39175 | 186619 | 
| 111 | 24 django/db/models/fields/related.py | 1 | 34| 246 | 39421 | 186619 | 
| 112 | 24 django/db/backends/base/schema.py | 820 | 840| 191 | 39612 | 186619 | 
| 113 | 25 django/db/backends/base/operations.py | 259 | 302| 324 | 39936 | 192279 | 
| 114 | 26 django/core/serializers/xml_serializer.py | 88 | 109| 192 | 40128 | 195703 | 
| 115 | 26 django/db/models/query.py | 1184 | 1203| 209 | 40337 | 195703 | 
| 116 | 26 django/db/models/query_utils.py | 234 | 268| 303 | 40640 | 195703 | 
| 117 | 26 django/contrib/contenttypes/fields.py | 564 | 596| 330 | 40970 | 195703 | 
| 118 | 26 django/db/backends/base/schema.py | 255 | 276| 154 | 41124 | 195703 | 
| 119 | 26 django/db/models/sql/compiler.py | 1224 | 1258| 332 | 41456 | 195703 | 
| 120 | 26 django/db/models/query.py | 184 | 241| 455 | 41911 | 195703 | 
| 121 | 26 django/db/models/sql/query.py | 2034 | 2056| 249 | 42160 | 195703 | 
| 122 | 26 django/db/models/base.py | 1 | 50| 326 | 42486 | 195703 | 
| 123 | 26 django/db/models/fields/json.py | 183 | 212| 275 | 42761 | 195703 | 
| 124 | 26 django/db/backends/base/schema.py | 635 | 707| 796 | 43557 | 195703 | 
| 125 | 26 django/contrib/contenttypes/fields.py | 430 | 451| 248 | 43805 | 195703 | 
| 126 | 26 django/db/models/sql/query.py | 2128 | 2173| 371 | 44176 | 195703 | 
| 127 | 26 django/db/models/fields/related_descriptors.py | 1016 | 1043| 334 | 44510 | 195703 | 


### Hint

```
Maybe this could generally be extended to use of __in with non-nullable fields? Since IN translates to OR = for each elements and NULL != NULL I assume it could be done at the __in lookup level even for non-nullable fields.
```

## Patch

```diff
diff --git a/django/db/models/lookups.py b/django/db/models/lookups.py
--- a/django/db/models/lookups.py
+++ b/django/db/models/lookups.py
@@ -366,10 +366,12 @@ def process_rhs(self, compiler, connection):
             )
 
         if self.rhs_is_direct_value():
+            # Remove None from the list as NULL is never equal to anything.
             try:
                 rhs = OrderedSet(self.rhs)
+                rhs.discard(None)
             except TypeError:  # Unhashable items in self.rhs
-                rhs = self.rhs
+                rhs = [r for r in self.rhs if r is not None]
 
             if not rhs:
                 raise EmptyResultSet

```

## Test Patch

```diff
diff --git a/tests/lookup/tests.py b/tests/lookup/tests.py
--- a/tests/lookup/tests.py
+++ b/tests/lookup/tests.py
@@ -576,8 +576,6 @@ def test_none(self):
         self.assertQuerysetEqual(Article.objects.none().iterator(), [])
 
     def test_in(self):
-        # using __in with an empty list should return an empty query set
-        self.assertQuerysetEqual(Article.objects.filter(id__in=[]), [])
         self.assertQuerysetEqual(
             Article.objects.exclude(id__in=[]),
             [
@@ -591,6 +589,9 @@ def test_in(self):
             ]
         )
 
+    def test_in_empty_list(self):
+        self.assertSequenceEqual(Article.objects.filter(id__in=[]), [])
+
     def test_in_different_database(self):
         with self.assertRaisesMessage(
             ValueError,
@@ -603,6 +604,31 @@ def test_in_keeps_value_ordering(self):
         query = Article.objects.filter(slug__in=['a%d' % i for i in range(1, 8)]).values('pk').query
         self.assertIn(' IN (a1, a2, a3, a4, a5, a6, a7) ', str(query))
 
+    def test_in_ignore_none(self):
+        with self.assertNumQueries(1) as ctx:
+            self.assertSequenceEqual(
+                Article.objects.filter(id__in=[None, self.a1.id]),
+                [self.a1],
+            )
+        sql = ctx.captured_queries[0]['sql']
+        self.assertIn('IN (%s)' % self.a1.pk, sql)
+
+    def test_in_ignore_solo_none(self):
+        with self.assertNumQueries(0):
+            self.assertSequenceEqual(Article.objects.filter(id__in=[None]), [])
+
+    def test_in_ignore_none_with_unhashable_items(self):
+        class UnhashableInt(int):
+            __hash__ = None
+
+        with self.assertNumQueries(1) as ctx:
+            self.assertSequenceEqual(
+                Article.objects.filter(id__in=[None, UnhashableInt(self.a1.id)]),
+                [self.a1],
+            )
+        sql = ctx.captured_queries[0]['sql']
+        self.assertIn('IN (%s)' % self.a1.pk, sql)
+
     def test_error_messages(self):
         # Programming errors are pointed out with nice error messages
         with self.assertRaisesMessage(

```


## Code snippets

### 1 - django/db/models/fields/related_lookups.py:

Start line: 62, End line: 99

```python
class RelatedIn(In):

    def as_sql(self, compiler, connection):
        if isinstance(self.lhs, MultiColSource):
            # For multicolumn lookups we need to build a multicolumn where clause.
            # This clause is either a SubqueryConstraint (for values that need to be compiled to
            # SQL) or an OR-combined list of (col1 = val1 AND col2 = val2 AND ...) clauses.
            from django.db.models.sql.where import WhereNode, SubqueryConstraint, AND, OR

            root_constraint = WhereNode(connector=OR)
            if self.rhs_is_direct_value():
                values = [get_normalized_value(value, self.lhs) for value in self.rhs]
                for value in values:
                    value_constraint = WhereNode()
                    for source, target, val in zip(self.lhs.sources, self.lhs.targets, value):
                        lookup_class = target.get_lookup('exact')
                        lookup = lookup_class(target.get_col(self.lhs.alias, source), val)
                        value_constraint.add(lookup, AND)
                    root_constraint.add(value_constraint, OR)
            else:
                root_constraint.add(
                    SubqueryConstraint(
                        self.lhs.alias, [target.column for target in self.lhs.targets],
                        [source.name for source in self.lhs.sources], self.rhs),
                    AND)
            return root_constraint.as_sql(compiler, connection)
        else:
            if (not getattr(self.rhs, 'has_select_fields', True) and
                    not getattr(self.lhs.field.target_field, 'primary_key', False)):
                self.rhs.clear_select_clause()
                if (getattr(self.lhs.output_field, 'primary_key', False) and
                        self.lhs.output_field.model == self.rhs.model):
                    # A case like Restaurant.objects.filter(place__in=restaurant_qs),
                    # where place is a OneToOneField and the primary key of
                    # Restaurant.
                    target_field = self.lhs.field.name
                else:
                    target_field = self.lhs.field.target_field.name
                self.rhs.add_fields([target_field], True)
            return super().as_sql(compiler, connection)
```
### 2 - django/db/models/fields/related_lookups.py:

Start line: 46, End line: 60

```python
class RelatedIn(In):
    def get_prep_lookup(self):
        if not isinstance(self.lhs, MultiColSource) and self.rhs_is_direct_value():
            # If we get here, we are dealing with single-column relations.
            self.rhs = [get_normalized_value(val, self.lhs)[0] for val in self.rhs]
            # We need to run the related field's get_prep_value(). Consider case
            # ForeignKey to IntegerField given value 'abc'. The ForeignKey itself
            # doesn't have validation for non-integers, so we must run validation
            # using the target field.
            if hasattr(self.lhs.output_field, 'get_path_info'):
                # Run the target field's get_prep_value. We can safely assume there is
                # only one as we don't get to the direct value branch otherwise.
                target_field = self.lhs.output_field.get_path_info()[-1].target_fields[-1]
                self.rhs = [target_field.get_prep_value(v) for v in self.rhs]
        return super().get_prep_lookup()
```
### 3 - django/db/models/query.py:

Start line: 1640, End line: 1746

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
### 4 - django/db/models/lookups.py:

Start line: 487, End line: 508

```python
@Field.register_lookup
class IsNull(BuiltinLookup):
    lookup_name = 'isnull'
    prepare_rhs = False

    def as_sql(self, compiler, connection):
        if not isinstance(self.rhs, bool):
            # When the deprecation ends, replace with:
            # raise ValueError(
            #     'The QuerySet value for an isnull lookup must be True or '
            #     'False.'
            # )
            warnings.warn(
                'Using a non-boolean value for an isnull lookup is '
                'deprecated, use True or False instead.',
                RemovedInDjango40Warning,
            )
        sql, params = compiler.compile(self.lhs)
        if self.rhs:
            return "%s IS NULL" % sql, params
        else:
            return "%s IS NOT NULL" % sql, params
```
### 5 - django/db/models/fields/related.py:

Start line: 837, End line: 858

```python
class ForeignKey(ForeignObject):

    def _check_on_delete(self):
        on_delete = getattr(self.remote_field, 'on_delete', None)
        if on_delete == SET_NULL and not self.null:
            return [
                checks.Error(
                    'Field specifies on_delete=SET_NULL, but cannot be null.',
                    hint='Set null=True argument on the field, or change the on_delete rule.',
                    obj=self,
                    id='fields.E320',
                )
            ]
        elif on_delete == SET_DEFAULT and not self.has_default():
            return [
                checks.Error(
                    'Field specifies on_delete=SET_DEFAULT, but has no default value.',
                    hint='Set a default value, or change the on_delete rule.',
                    obj=self,
                    id='fields.E321',
                )
            ]
        else:
            return []
```
### 6 - django/db/models/lookups.py:

Start line: 356, End line: 386

```python
@Field.register_lookup
class In(FieldGetDbPrepValueIterableMixin, BuiltinLookup):
    lookup_name = 'in'

    def process_rhs(self, compiler, connection):
        db_rhs = getattr(self.rhs, '_db', None)
        if db_rhs is not None and db_rhs != connection.alias:
            raise ValueError(
                "Subqueries aren't allowed across different databases. Force "
                "the inner query to be evaluated using `list(inner_query)`."
            )

        if self.rhs_is_direct_value():
            try:
                rhs = OrderedSet(self.rhs)
            except TypeError:  # Unhashable items in self.rhs
                rhs = self.rhs

            if not rhs:
                raise EmptyResultSet

            # rhs should be an iterable; use batch_process_rhs() to
            # prepare/transform those values.
            sqls, sqls_params = self.batch_process_rhs(compiler, connection, rhs)
            placeholder = '(' + ', '.join(sqls) + ')'
            return (placeholder, sqls_params)
        else:
            if not getattr(self.rhs, 'has_select_fields', True):
                self.rhs.clear_select_clause()
                self.rhs.add_fields(['pk'])
            return super().process_rhs(compiler, connection)
```
### 7 - django/db/models/query.py:

Start line: 1861, End line: 1893

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
### 8 - django/db/models/query.py:

Start line: 1552, End line: 1608

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
### 9 - django/db/models/fields/related_lookups.py:

Start line: 102, End line: 117

```python
class RelatedLookupMixin:
    def get_prep_lookup(self):
        if not isinstance(self.lhs, MultiColSource) and not hasattr(self.rhs, 'resolve_expression'):
            # If we get here, we are dealing with single-column relations.
            self.rhs = get_normalized_value(self.rhs, self.lhs)[0]
            # We need to run the related field's get_prep_value(). Consider case
            # ForeignKey to IntegerField given value 'abc'. The ForeignKey itself
            # doesn't have validation for non-integers, so we must run validation
            # using the target field.
            if self.prepare_rhs and hasattr(self.lhs.output_field, 'get_path_info'):
                # Get the target field. We can safely assume there is only one
                # as we don't get to the direct value branch otherwise.
                target_field = self.lhs.output_field.get_path_info()[-1].target_fields[-1]
                self.rhs = target_field.get_prep_value(self.rhs)

        return super().get_prep_lookup()
```
### 10 - django/db/models/fields/related.py:

Start line: 746, End line: 764

```python
class ForeignObject(RelatedField):

    def contribute_to_related_class(self, cls, related):
        # Internal FK's - i.e., those with a related name ending with '+' -
        # and swapped models don't get a related descriptor.
        if not self.remote_field.is_hidden() and not related.related_model._meta.swapped:
            setattr(cls._meta.concrete_model, related.get_accessor_name(), self.related_accessor_class(related))
            # While 'limit_choices_to' might be a callable, simply pass
            # it along for later - this is too early because it's still
            # model load time.
            if self.remote_field.limit_choices_to:
                cls._meta.related_fkey_lookups.append(self.remote_field.limit_choices_to)


ForeignObject.register_lookup(RelatedIn)
ForeignObject.register_lookup(RelatedExact)
ForeignObject.register_lookup(RelatedLessThan)
ForeignObject.register_lookup(RelatedGreaterThan)
ForeignObject.register_lookup(RelatedGreaterThanOrEqual)
ForeignObject.register_lookup(RelatedLessThanOrEqual)
ForeignObject.register_lookup(RelatedIsNull)
```
### 12 - django/db/models/lookups.py:

Start line: 388, End line: 417

```python
@Field.register_lookup
class In(FieldGetDbPrepValueIterableMixin, BuiltinLookup):

    def get_rhs_op(self, connection, rhs):
        return 'IN %s' % rhs

    def as_sql(self, compiler, connection):
        max_in_list_size = connection.ops.max_in_list_size()
        if self.rhs_is_direct_value() and max_in_list_size and len(self.rhs) > max_in_list_size:
            return self.split_parameter_list_as_sql(compiler, connection)
        return super().as_sql(compiler, connection)

    def split_parameter_list_as_sql(self, compiler, connection):
        # This is a special case for databases which limit the number of
        # elements which can appear in an 'IN' clause.
        max_in_list_size = connection.ops.max_in_list_size()
        lhs, lhs_params = self.process_lhs(compiler, connection)
        rhs, rhs_params = self.batch_process_rhs(compiler, connection)
        in_clause_elements = ['(']
        params = []
        for offset in range(0, len(rhs_params), max_in_list_size):
            if offset > 0:
                in_clause_elements.append(' OR ')
            in_clause_elements.append('%s IN (' % lhs)
            params.extend(lhs_params)
            sqls = rhs[offset: offset + max_in_list_size]
            sqls_params = rhs_params[offset: offset + max_in_list_size]
            param_group = ', '.join(sqls)
            in_clause_elements.append(param_group)
            in_clause_elements.append(')')
            params.extend(sqls_params)
        in_clause_elements.append(')')
        return ''.join(in_clause_elements), params
```
### 29 - django/db/models/lookups.py:

Start line: 188, End line: 205

```python
class FieldGetDbPrepValueMixin:
    """
    Some lookups require Field.get_db_prep_value() to be called on their
    inputs.
    """
    get_db_prep_lookup_value_is_iterable = False

    def get_db_prep_lookup(self, value, connection):
        # For relational fields, use the 'target_field' attribute of the
        # output_field.
        field = getattr(self.lhs.output_field, 'target_field', None)
        get_db_prep_value = getattr(field, 'get_db_prep_value', None) or self.lhs.output_field.get_db_prep_value
        return (
            '%s',
            [get_db_prep_value(v, connection, prepared=True) for v in value]
            if self.get_db_prep_lookup_value_is_iterable else
            [get_db_prep_value(value, connection, prepared=True)]
        )
```
### 37 - django/db/models/lookups.py:

Start line: 288, End line: 300

```python
@Field.register_lookup
class Exact(FieldGetDbPrepValueMixin, BuiltinLookup):

    def as_sql(self, compiler, connection):
        # Avoid comparison against direct rhs if lhs is a boolean value. That
        # turns "boolfield__exact=True" into "WHERE boolean_field" instead of
        # "WHERE boolean_field = True" when allowed.
        if (
            isinstance(self.rhs, bool) and
            getattr(self.lhs, 'conditional', False) and
            connection.ops.conditional_expression_supported_in_where_clause(self.lhs)
        ):
            lhs_sql, params = self.process_lhs(compiler, connection)
            template = '%s' if self.rhs else 'NOT %s'
            return template % lhs_sql, params
        return super().as_sql(compiler, connection)
```
### 41 - django/db/models/lookups.py:

Start line: 208, End line: 243

```python
class FieldGetDbPrepValueIterableMixin(FieldGetDbPrepValueMixin):
    """
    Some lookups require Field.get_db_prep_value() to be called on each value
    in an iterable.
    """
    get_db_prep_lookup_value_is_iterable = True

    def get_prep_lookup(self):
        if hasattr(self.rhs, 'resolve_expression'):
            return self.rhs
        prepared_values = []
        for rhs_value in self.rhs:
            if hasattr(rhs_value, 'resolve_expression'):
                # An expression will be handled by the database but can coexist
                # alongside real values.
                pass
            elif self.prepare_rhs and hasattr(self.lhs.output_field, 'get_prep_value'):
                rhs_value = self.lhs.output_field.get_prep_value(rhs_value)
            prepared_values.append(rhs_value)
        return prepared_values

    def process_rhs(self, compiler, connection):
        if self.rhs_is_direct_value():
            # rhs should be an iterable of values. Use batch_process_rhs()
            # to prepare/transform those values.
            return self.batch_process_rhs(compiler, connection)
        else:
            return super().process_rhs(compiler, connection)

    def resolve_expression_parameter(self, compiler, connection, sql, param):
        params = [param]
        if hasattr(param, 'resolve_expression'):
            param = param.resolve_expression(compiler.query)
        if hasattr(param, 'as_sql'):
            sql, params = param.as_sql(compiler, connection)
        return sql, params
```
### 49 - django/db/models/lookups.py:

Start line: 303, End line: 353

```python
@Field.register_lookup
class IExact(BuiltinLookup):
    lookup_name = 'iexact'
    prepare_rhs = False

    def process_rhs(self, qn, connection):
        rhs, params = super().process_rhs(qn, connection)
        if params:
            params[0] = connection.ops.prep_for_iexact_query(params[0])
        return rhs, params


@Field.register_lookup
class GreaterThan(FieldGetDbPrepValueMixin, BuiltinLookup):
    lookup_name = 'gt'


@Field.register_lookup
class GreaterThanOrEqual(FieldGetDbPrepValueMixin, BuiltinLookup):
    lookup_name = 'gte'


@Field.register_lookup
class LessThan(FieldGetDbPrepValueMixin, BuiltinLookup):
    lookup_name = 'lt'


@Field.register_lookup
class LessThanOrEqual(FieldGetDbPrepValueMixin, BuiltinLookup):
    lookup_name = 'lte'


class IntegerFieldFloatRounding:
    """
    Allow floats to work as query values for IntegerField. Without this, the
    decimal portion of the float would always be discarded.
    """
    def get_prep_lookup(self):
        if isinstance(self.rhs, float):
            self.rhs = math.ceil(self.rhs)
        return super().get_prep_lookup()


@IntegerField.register_lookup
class IntegerGreaterThanOrEqual(IntegerFieldFloatRounding, GreaterThanOrEqual):
    pass


@IntegerField.register_lookup
class IntegerLessThan(IntegerFieldFloatRounding, LessThan):
    pass
```
### 51 - django/db/models/lookups.py:

Start line: 1, End line: 40

```python
import itertools
import math
import warnings
from copy import copy

from django.core.exceptions import EmptyResultSet
from django.db.models.expressions import Case, Exists, Func, Value, When
from django.db.models.fields import (
    BooleanField, CharField, DateTimeField, Field, IntegerField, UUIDField,
)
from django.db.models.query_utils import RegisterLookupMixin
from django.utils.datastructures import OrderedSet
from django.utils.deprecation import RemovedInDjango40Warning
from django.utils.functional import cached_property


class Lookup:
    lookup_name = None
    prepare_rhs = True
    can_use_none_as_rhs = False

    def __init__(self, lhs, rhs):
        self.lhs, self.rhs = lhs, rhs
        self.rhs = self.get_prep_lookup()
        if hasattr(self.lhs, 'get_bilateral_transforms'):
            bilateral_transforms = self.lhs.get_bilateral_transforms()
        else:
            bilateral_transforms = []
        if bilateral_transforms:
            # Warn the user as soon as possible if they are trying to apply
            # a bilateral transformation on a nested QuerySet: that won't work.
            from django.db.models.sql.query import Query  # avoid circular import
            if isinstance(rhs, Query):
                raise NotImplementedError("Bilateral transformations on nested querysets are not implemented.")
        self.bilateral_transforms = bilateral_transforms

    def apply_bilateral_transforms(self, value):
        for transform in self.bilateral_transforms:
            value = transform(value)
        return value
```
### 110 - django/db/models/lookups.py:

Start line: 119, End line: 142

```python
class Lookup:

    def as_oracle(self, compiler, connection):
        # Oracle doesn't allow EXISTS() to be compared to another expression
        # unless it's wrapped in a CASE WHEN.
        wrapped = False
        exprs = []
        for expr in (self.lhs, self.rhs):
            if isinstance(expr, Exists):
                expr = Case(When(expr, then=True), default=False, output_field=BooleanField())
                wrapped = True
            exprs.append(expr)
        lookup = type(self)(*exprs) if wrapped else self
        return lookup.as_sql(compiler, connection)

    @cached_property
    def contains_aggregate(self):
        return self.lhs.contains_aggregate or getattr(self.rhs, 'contains_aggregate', False)

    @cached_property
    def contains_over_clause(self):
        return self.lhs.contains_over_clause or getattr(self.rhs, 'contains_over_clause', False)

    @property
    def is_summary(self):
        return self.lhs.is_summary or getattr(self.rhs, 'is_summary', False)
```
