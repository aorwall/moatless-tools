# django__django-11405

| **django/django** | `2007e11d7069b0f6ed673c7520ee7f480f07de68` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 2492 |
| **Any found context length** | 2492 |
| **Avg pos** | 9.0 |
| **Min pos** | 9 |
| **Max pos** | 9 |
| **Top file pos** | 2 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/db/models/sql/compiler.py b/django/db/models/sql/compiler.py
--- a/django/db/models/sql/compiler.py
+++ b/django/db/models/sql/compiler.py
@@ -281,6 +281,7 @@ def get_order_by(self):
                 if not isinstance(field, OrderBy):
                     field = field.asc()
                 if not self.query.standard_ordering:
+                    field = field.copy()
                     field.reverse_ordering()
                 order_by.append((field, False))
                 continue

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/db/models/sql/compiler.py | 284 | 284 | 9 | 2 | 2492


## Problem Statement

```
Queryset ordering and Meta.ordering are mutable on expressions with reverse().
Description
	
Queryset order and Meta.ordering are mutable with reverse().
Bug revealed by running ./runtests.py ordering.test --reverse (reproduced at a2c31e12da272acc76f3a3a0157fae9a7f6477ac).
It seems that test added in f218a2ff455b5f7391dd38038994f2c5f8b0eca1 wasn't correct because order mutates on queryset execution in ​SQLCompiler.get_order_by().

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 django/db/models/expressions.py | 1102 | 1135| 292 | 292 | 9860 | 
| 2 | 1 django/db/models/expressions.py | 1085 | 1100| 148 | 440 | 9860 | 
| 3 | 1 django/db/models/expressions.py | 1062 | 1083| 183 | 623 | 9860 | 
| 4 | **2 django/db/models/sql/compiler.py** | 335 | 363| 335 | 958 | 23379 | 
| 5 | 3 django/db/models/query.py | 1066 | 1100| 298 | 1256 | 39850 | 
| 6 | 4 django/contrib/postgres/aggregates/mixins.py | 1 | 20| 181 | 1437 | 40291 | 
| 7 | 5 django/db/models/sql/query.py | 1826 | 1859| 279 | 1716 | 61456 | 
| 8 | 5 django/contrib/postgres/aggregates/mixins.py | 36 | 52| 145 | 1861 | 61456 | 
| **-> 9 <-** | **5 django/db/models/sql/compiler.py** | 253 | 333| 631 | 2492 | 61456 | 
| 10 | 6 django/db/models/functions/text.py | 227 | 243| 149 | 2641 | 63912 | 
| 11 | 6 django/contrib/postgres/aggregates/mixins.py | 22 | 34| 128 | 2769 | 63912 | 
| 12 | 6 django/db/models/sql/query.py | 1406 | 1484| 734 | 3503 | 63912 | 
| 13 | 7 django/contrib/admin/views/main.py | 309 | 361| 467 | 3970 | 67989 | 
| 14 | 7 django/db/models/query.py | 759 | 798| 322 | 4292 | 67989 | 
| 15 | 7 django/db/models/sql/query.py | 1616 | 1643| 351 | 4643 | 67989 | 
| 16 | 7 django/db/models/sql/query.py | 1265 | 1302| 512 | 5155 | 67989 | 
| 17 | 7 django/db/models/sql/query.py | 358 | 408| 494 | 5649 | 67989 | 
| 18 | 8 django/db/models/base.py | 949 | 962| 180 | 5829 | 82947 | 
| 19 | 8 django/db/models/base.py | 1623 | 1715| 673 | 6502 | 82947 | 
| 20 | 8 django/db/models/sql/query.py | 883 | 905| 248 | 6750 | 82947 | 
| 21 | 9 django/db/backends/mysql/operations.py | 127 | 158| 294 | 7044 | 85997 | 
| 22 | 10 django/views/generic/list.py | 50 | 75| 244 | 7288 | 87570 | 
| 23 | 11 django/db/migrations/autodetector.py | 337 | 356| 196 | 7484 | 99241 | 
| 24 | 11 django/db/models/sql/query.py | 2030 | 2063| 246 | 7730 | 99241 | 
| 25 | 11 django/db/models/sql/query.py | 1562 | 1586| 227 | 7957 | 99241 | 
| 26 | 11 django/db/models/sql/query.py | 690 | 725| 389 | 8346 | 99241 | 
| 27 | 11 django/db/models/sql/query.py | 1043 | 1067| 243 | 8589 | 99241 | 
| 28 | **11 django/db/models/sql/compiler.py** | 365 | 373| 119 | 8708 | 99241 | 
| 29 | 11 django/db/models/sql/query.py | 2192 | 2203| 116 | 8824 | 99241 | 
| 30 | 11 django/db/models/sql/query.py | 615 | 639| 269 | 9093 | 99241 | 
| 31 | 11 django/db/models/sql/query.py | 1716 | 1743| 244 | 9337 | 99241 | 
| 32 | 11 django/db/models/query.py | 1396 | 1427| 291 | 9628 | 99241 | 
| 33 | 11 django/db/models/query.py | 1346 | 1354| 136 | 9764 | 99241 | 
| 34 | 12 django/db/migrations/operations/special.py | 63 | 114| 390 | 10154 | 100799 | 
| 35 | 13 django/db/migrations/operations/models.py | 614 | 627| 137 | 10291 | 107545 | 
| 36 | 13 django/db/models/expressions.py | 212 | 246| 285 | 10576 | 107545 | 
| 37 | **13 django/db/models/sql/compiler.py** | 717 | 728| 147 | 10723 | 107545 | 
| 38 | 13 django/db/models/query.py | 1138 | 1172| 234 | 10957 | 107545 | 
| 39 | 13 django/db/models/sql/query.py | 1377 | 1388| 137 | 11094 | 107545 | 
| 40 | **13 django/db/models/sql/compiler.py** | 1 | 19| 157 | 11251 | 107545 | 
| 41 | 13 django/db/models/query.py | 643 | 657| 132 | 11383 | 107545 | 
| 42 | 14 django/db/models/__init__.py | 1 | 49| 548 | 11931 | 108093 | 
| 43 | 14 django/db/migrations/operations/models.py | 571 | 594| 183 | 12114 | 108093 | 
| 44 | 15 django/contrib/admin/options.py | 206 | 217| 135 | 12249 | 126446 | 
| 45 | 15 django/db/models/base.py | 933 | 947| 212 | 12461 | 126446 | 
| 46 | 15 django/db/models/expressions.py | 314 | 382| 445 | 12906 | 126446 | 
| 47 | 15 django/db/models/sql/query.py | 1304 | 1325| 259 | 13165 | 126446 | 
| 48 | 16 django/db/backends/base/operations.py | 193 | 224| 285 | 13450 | 131839 | 
| 49 | 16 django/db/models/query.py | 1118 | 1136| 199 | 13649 | 131839 | 
| 50 | 16 django/db/models/query.py | 1478 | 1524| 433 | 14082 | 131839 | 
| 51 | 17 django/db/models/fields/related_descriptors.py | 356 | 372| 184 | 14266 | 142124 | 
| 52 | 17 django/db/models/sql/query.py | 292 | 336| 388 | 14654 | 142124 | 
| 53 | 17 django/db/models/query.py | 800 | 829| 248 | 14902 | 142124 | 
| 54 | 17 django/db/models/expressions.py | 490 | 520| 226 | 15128 | 142124 | 
| 55 | 17 django/db/models/base.py | 1814 | 1865| 351 | 15479 | 142124 | 
| 56 | 17 django/db/models/expressions.py | 1031 | 1059| 286 | 15765 | 142124 | 
| 57 | 18 django/contrib/admin/checks.py | 546 | 581| 303 | 16068 | 151138 | 
| 58 | 18 django/db/models/expressions.py | 1183 | 1215| 246 | 16314 | 151138 | 
| 59 | 18 django/db/models/query.py | 618 | 641| 197 | 16511 | 151138 | 
| 60 | 19 django/contrib/postgres/search.py | 147 | 154| 121 | 16632 | 153076 | 
| 61 | 19 django/contrib/admin/views/main.py | 271 | 307| 351 | 16983 | 153076 | 
| 62 | **19 django/db/models/sql/compiler.py** | 22 | 43| 248 | 17231 | 153076 | 
| 63 | 19 django/db/models/sql/query.py | 858 | 881| 203 | 17434 | 153076 | 
| 64 | 19 django/db/models/query.py | 261 | 281| 180 | 17614 | 153076 | 
| 65 | 19 django/db/models/sql/query.py | 1745 | 1786| 283 | 17897 | 153076 | 
| 66 | 19 django/db/models/sql/query.py | 794 | 820| 280 | 18177 | 153076 | 
| 67 | **19 django/db/models/sql/compiler.py** | 1029 | 1051| 239 | 18416 | 153076 | 
| 68 | 19 django/db/models/sql/query.py | 127 | 142| 145 | 18561 | 153076 | 
| 69 | 20 django/db/models/fields/related.py | 190 | 254| 673 | 19234 | 166647 | 
| 70 | **20 django/db/models/sql/compiler.py** | 137 | 181| 490 | 19724 | 166647 | 
| 71 | 20 django/db/models/query.py | 233 | 259| 221 | 19945 | 166647 | 
| 72 | 20 django/db/models/sql/query.py | 1008 | 1041| 326 | 20271 | 166647 | 
| 73 | 20 django/db/models/query.py | 928 | 959| 341 | 20612 | 166647 | 
| 74 | **20 django/db/models/sql/compiler.py** | 686 | 715| 351 | 20963 | 166647 | 
| 75 | 20 django/db/models/expressions.py | 410 | 445| 385 | 21348 | 166647 | 
| 76 | 20 django/db/models/expressions.py | 984 | 1028| 310 | 21658 | 166647 | 
| 77 | 20 django/db/models/expressions.py | 1138 | 1181| 373 | 22031 | 166647 | 
| 78 | 20 django/db/migrations/operations/models.py | 596 | 612| 215 | 22246 | 166647 | 
| 79 | 20 django/db/models/expressions.py | 641 | 665| 238 | 22484 | 166647 | 
| 80 | 21 django/db/models/aggregates.py | 45 | 68| 294 | 22778 | 167934 | 
| 81 | 21 django/db/models/query.py | 1 | 45| 324 | 23102 | 167934 | 
| 82 | 21 django/db/models/sql/query.py | 338 | 356| 144 | 23246 | 167934 | 
| 83 | 21 django/db/models/expressions.py | 956 | 981| 242 | 23488 | 167934 | 
| 84 | 22 django/db/backends/sqlite3/operations.py | 42 | 64| 220 | 23708 | 170818 | 
| 85 | 23 django/db/models/query_utils.py | 154 | 218| 492 | 24200 | 173449 | 
| 86 | 23 django/db/models/expressions.py | 184 | 210| 223 | 24423 | 173449 | 
| 87 | 23 django/db/models/expressions.py | 908 | 954| 377 | 24800 | 173449 | 
| 88 | 23 django/db/models/sql/query.py | 248 | 290| 308 | 25108 | 173449 | 
| 89 | 24 django/db/backends/postgresql/operations.py | 157 | 199| 454 | 25562 | 176077 | 
| 90 | 24 django/db/backends/mysql/operations.py | 222 | 232| 151 | 25713 | 176077 | 
| 91 | 25 django/core/paginator.py | 110 | 136| 203 | 25916 | 177392 | 
| 92 | **25 django/db/models/sql/compiler.py** | 1120 | 1141| 213 | 26129 | 177392 | 
| 93 | 25 django/db/models/sql/query.py | 907 | 925| 146 | 26275 | 177392 | 
| 94 | 26 django/db/backends/oracle/operations.py | 435 | 467| 326 | 26601 | 182876 | 
| 95 | 26 django/db/models/sql/query.py | 822 | 856| 350 | 26951 | 182876 | 
| 96 | 26 django/db/models/expressions.py | 849 | 905| 532 | 27483 | 182876 | 
| 97 | 26 django/db/models/query.py | 1238 | 1256| 186 | 27669 | 182876 | 
| 98 | 26 django/db/models/sql/query.py | 1904 | 1934| 260 | 27929 | 182876 | 
| 99 | 26 django/db/models/sql/query.py | 1 | 74| 509 | 28438 | 182876 | 
| 100 | 26 django/db/backends/sqlite3/operations.py | 240 | 255| 156 | 28594 | 182876 | 
| 101 | **26 django/db/models/sql/compiler.py** | 1436 | 1476| 409 | 29003 | 182876 | 
| 102 | 26 django/db/migrations/autodetector.py | 1175 | 1200| 245 | 29248 | 182876 | 
| 103 | 27 django/db/models/sql/subqueries.py | 79 | 107| 210 | 29458 | 184395 | 
| 104 | 27 django/db/backends/sqlite3/operations.py | 293 | 335| 429 | 29887 | 184395 | 
| 105 | 27 django/db/models/sql/query.py | 1861 | 1902| 342 | 30229 | 184395 | 
| 106 | 27 django/db/models/expressions.py | 668 | 708| 247 | 30476 | 184395 | 
| 107 | 28 django/db/backends/postgresql/introspection.py | 218 | 234| 179 | 30655 | 186754 | 
| 108 | 28 django/db/backends/mysql/operations.py | 234 | 262| 248 | 30903 | 186754 | 
| 109 | 28 django/db/models/sql/query.py | 145 | 246| 867 | 31770 | 186754 | 
| 110 | 28 django/db/models/expressions.py | 523 | 547| 174 | 31944 | 186754 | 
| 111 | 28 django/db/models/query.py | 1429 | 1475| 336 | 32280 | 186754 | 
| 112 | 28 django/db/models/sql/query.py | 2253 | 2308| 827 | 33107 | 186754 | 
| 113 | 28 django/db/models/expressions.py | 711 | 739| 230 | 33337 | 186754 | 
| 114 | 29 django/forms/forms.py | 114 | 134| 177 | 33514 | 190827 | 
| 115 | 29 django/db/backends/sqlite3/operations.py | 277 | 291| 148 | 33662 | 190827 | 
| 116 | **29 django/db/models/sql/compiler.py** | 939 | 961| 180 | 33842 | 190827 | 
| 117 | **29 django/db/models/sql/compiler.py** | 402 | 454| 540 | 34382 | 190827 | 
| 118 | 29 django/db/models/query_utils.py | 1 | 44| 287 | 34669 | 190827 | 
| 119 | 29 django/db/models/sql/query.py | 760 | 792| 392 | 35061 | 190827 | 
| 120 | 30 django/urls/resolvers.py | 603 | 669| 654 | 35715 | 196231 | 
| 121 | 30 django/db/models/expressions.py | 596 | 621| 268 | 35983 | 196231 | 
| 122 | 30 django/db/models/query.py | 1258 | 1267| 114 | 36097 | 196231 | 
| 123 | 30 django/db/models/sql/query.py | 641 | 688| 511 | 36608 | 196231 | 
| 124 | 30 django/db/migrations/operations/special.py | 181 | 204| 246 | 36854 | 196231 | 
| 125 | 30 django/db/backends/postgresql/introspection.py | 93 | 111| 269 | 37123 | 196231 | 
| 126 | 30 django/db/models/sql/query.py | 540 | 614| 807 | 37930 | 196231 | 
| 127 | 30 django/contrib/admin/views/main.py | 363 | 401| 334 | 38264 | 196231 | 
| 128 | 31 django/db/models/sql/where.py | 228 | 244| 130 | 38394 | 198006 | 
| 129 | 31 django/db/backends/sqlite3/operations.py | 118 | 143| 279 | 38673 | 198006 | 


## Patch

```diff
diff --git a/django/db/models/sql/compiler.py b/django/db/models/sql/compiler.py
--- a/django/db/models/sql/compiler.py
+++ b/django/db/models/sql/compiler.py
@@ -281,6 +281,7 @@ def get_order_by(self):
                 if not isinstance(field, OrderBy):
                     field = field.asc()
                 if not self.query.standard_ordering:
+                    field = field.copy()
                     field.reverse_ordering()
                 order_by.append((field, False))
                 continue

```

## Test Patch

```diff
diff --git a/tests/ordering/tests.py b/tests/ordering/tests.py
--- a/tests/ordering/tests.py
+++ b/tests/ordering/tests.py
@@ -210,6 +210,15 @@ def test_reversed_ordering(self):
     def test_reverse_ordering_pure(self):
         qs1 = Article.objects.order_by(F('headline').asc())
         qs2 = qs1.reverse()
+        self.assertQuerysetEqual(
+            qs2, [
+                'Article 4',
+                'Article 3',
+                'Article 2',
+                'Article 1',
+            ],
+            attrgetter('headline'),
+        )
         self.assertQuerysetEqual(
             qs1, [
                 "Article 1",
@@ -219,14 +228,29 @@ def test_reverse_ordering_pure(self):
             ],
             attrgetter("headline")
         )
+
+    def test_reverse_meta_ordering_pure(self):
+        Article.objects.create(
+            headline='Article 5',
+            pub_date=datetime(2005, 7, 30),
+            author=self.author_1,
+            second_author=self.author_2,
+        )
+        Article.objects.create(
+            headline='Article 5',
+            pub_date=datetime(2005, 7, 30),
+            author=self.author_2,
+            second_author=self.author_1,
+        )
         self.assertQuerysetEqual(
-            qs2, [
-                "Article 4",
-                "Article 3",
-                "Article 2",
-                "Article 1",
-            ],
-            attrgetter("headline")
+            Article.objects.filter(headline='Article 5').reverse(),
+            ['Name 2', 'Name 1'],
+            attrgetter('author.name'),
+        )
+        self.assertQuerysetEqual(
+            Article.objects.filter(headline='Article 5'),
+            ['Name 1', 'Name 2'],
+            attrgetter('author.name'),
         )
 
     def test_no_reordering_after_slicing(self):

```


## Code snippets

### 1 - django/db/models/expressions.py:

Start line: 1102, End line: 1135

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
### 2 - django/db/models/expressions.py:

Start line: 1085, End line: 1100

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
### 3 - django/db/models/expressions.py:

Start line: 1062, End line: 1083

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
### 4 - django/db/models/sql/compiler.py:

Start line: 335, End line: 363

```python
class SQLCompiler:

    def get_order_by(self):
        # ... other code

        for expr, is_ref in order_by:
            resolved = expr.resolve_expression(self.query, allow_joins=True, reuse=None)
            if self.query.combinator:
                src = resolved.get_source_expressions()[0]
                # Relabel order by columns to raw numbers if this is a combined
                # query; necessary since the columns can't be referenced by the
                # fully qualified name and the simple column names may collide.
                for idx, (sel_expr, _, col_alias) in enumerate(self.select):
                    if is_ref and col_alias == src.refs:
                        src = src.source
                    elif col_alias:
                        continue
                    if src == sel_expr:
                        resolved.set_source_expressions([RawSQL('%d' % (idx + 1), ())])
                        break
                else:
                    raise DatabaseError('ORDER BY term does not match any column in the result set.')
            sql, params = self.compile(resolved)
            # Don't add the same column twice, but the order direction is
            # not taken into account so we strip it. When this entire method
            # is refactored into expressions, then we can check each part as we
            # generate it.
            without_ordering = self.ordering_parts.search(sql).group(1)
            params_hash = make_hashable(params)
            if (without_ordering, params_hash) in seen:
                continue
            seen.add((without_ordering, params_hash))
            result.append((resolved, (sql, params, is_ref)))
        return result
```
### 5 - django/db/models/query.py:

Start line: 1066, End line: 1100

```python
class QuerySet:

    def order_by(self, *field_names):
        """Return a new QuerySet instance with the ordering changed."""
        assert self.query.can_filter(), \
            "Cannot reorder a query once a slice has been taken."
        obj = self._chain()
        obj.query.clear_ordering(force_empty=False)
        obj.query.add_ordering(*field_names)
        return obj

    def distinct(self, *field_names):
        """
        Return a new QuerySet instance that will select only distinct results.
        """
        assert self.query.can_filter(), \
            "Cannot create distinct fields once a slice has been taken."
        obj = self._chain()
        obj.query.add_distinct_fields(*field_names)
        return obj

    def extra(self, select=None, where=None, params=None, tables=None,
              order_by=None, select_params=None):
        """Add extra SQL fragments to the query."""
        assert self.query.can_filter(), \
            "Cannot change a query once a slice has been taken"
        clone = self._chain()
        clone.query.add_extra(select, select_params, where, params, tables, order_by)
        return clone

    def reverse(self):
        """Reverse the ordering of the QuerySet."""
        if not self.query.can_filter():
            raise TypeError('Cannot reverse a query once a slice has been taken.')
        clone = self._chain()
        clone.query.standard_ordering = not clone.query.standard_ordering
        return clone
```
### 6 - django/contrib/postgres/aggregates/mixins.py:

Start line: 1, End line: 20

```python
from django.db.models.expressions import F, OrderBy


class OrderableAggMixin:

    def __init__(self, expression, ordering=(), **extra):
        if not isinstance(ordering, (list, tuple)):
            ordering = [ordering]
        ordering = ordering or []
        # Transform minus sign prefixed strings into an OrderBy() expression.
        ordering = (
            (OrderBy(F(o[1:]), descending=True) if isinstance(o, str) and o[0] == '-' else o)
            for o in ordering
        )
        super().__init__(expression, **extra)
        self.ordering = self._parse_expressions(*ordering)

    def resolve_expression(self, *args, **kwargs):
        self.ordering = [expr.resolve_expression(*args, **kwargs) for expr in self.ordering]
        return super().resolve_expression(*args, **kwargs)
```
### 7 - django/db/models/sql/query.py:

Start line: 1826, End line: 1859

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
            if not hasattr(item, 'resolve_expression') and not ORDER_PATTERN.match(item):
                errors.append(item)
            if getattr(item, 'contains_aggregate', False):
                raise FieldError(
                    'Using an aggregate in order_by() without also including '
                    'it in annotate() is not allowed: %s' % item
                )
        if errors:
            raise FieldError('Invalid order_by arguments: %s' % errors)
        if ordering:
            self.order_by += ordering
        else:
            self.default_ordering = False

    def clear_ordering(self, force_empty):
        """
        Remove any ordering settings. If 'force_empty' is True, there will be
        no ordering in the resulting query (not even the model's default).
        """
        self.order_by = ()
        self.extra_order_by = ()
        if force_empty:
            self.default_ordering = False
```
### 8 - django/contrib/postgres/aggregates/mixins.py:

Start line: 36, End line: 52

```python
class OrderableAggMixin:

    def get_source_expressions(self):
        return self.source_expressions + self.ordering

    def get_source_fields(self):
        # Filter out fields contributed by the ordering expressions as
        # these should not be used to determine which the return type of the
        # expression.
        return [
            e._output_field_or_none
            for e in self.get_source_expressions()[:self._get_ordering_expressions_index()]
        ]

    def _get_ordering_expressions_index(self):
        """Return the index at which the ordering expressions start."""
        source_expressions = self.get_source_expressions()
        return len(source_expressions) - len(self.ordering)
```
### 9 - django/db/models/sql/compiler.py:

Start line: 253, End line: 333

```python
class SQLCompiler:

    def get_order_by(self):
        """
        Return a list of 2-tuples of form (expr, (sql, params, is_ref)) for the
        ORDER BY clause.

        The order_by clause can alter the select clause (for example it
        can add aliases to clauses that do not yet have one, or it can
        add totally new select clauses).
        """
        if self.query.extra_order_by:
            ordering = self.query.extra_order_by
        elif not self.query.default_ordering:
            ordering = self.query.order_by
        elif self.query.order_by:
            ordering = self.query.order_by
        elif self.query.get_meta().ordering:
            ordering = self.query.get_meta().ordering
            self._meta_ordering = ordering
        else:
            ordering = []
        if self.query.standard_ordering:
            asc, desc = ORDER_DIR['ASC']
        else:
            asc, desc = ORDER_DIR['DESC']

        order_by = []
        for field in ordering:
            if hasattr(field, 'resolve_expression'):
                if not isinstance(field, OrderBy):
                    field = field.asc()
                if not self.query.standard_ordering:
                    field.reverse_ordering()
                order_by.append((field, False))
                continue
            if field == '?':  # random
                order_by.append((OrderBy(Random()), False))
                continue

            col, order = get_order_dir(field, asc)
            descending = order == 'DESC'

            if col in self.query.annotation_select:
                # Reference to expression in SELECT clause
                order_by.append((
                    OrderBy(Ref(col, self.query.annotation_select[col]), descending=descending),
                    True))
                continue
            if col in self.query.annotations:
                # References to an expression which is masked out of the SELECT clause
                order_by.append((
                    OrderBy(self.query.annotations[col], descending=descending),
                    False))
                continue

            if '.' in field:
                # This came in through an extra(order_by=...) addition. Pass it
                # on verbatim.
                table, col = col.split('.', 1)
                order_by.append((
                    OrderBy(
                        RawSQL('%s.%s' % (self.quote_name_unless_alias(table), col), []),
                        descending=descending
                    ), False))
                continue

            if not self.query.extra or col not in self.query.extra:
                # 'col' is of the form 'field' or 'field1__field2' or
                # '-field1__field2__field', etc.
                order_by.extend(self.find_ordering_name(
                    field, self.query.get_meta(), default_order=asc))
            else:
                if col not in self.query.extra_select:
                    order_by.append((
                        OrderBy(RawSQL(*self.query.extra[col]), descending=descending),
                        False))
                else:
                    order_by.append((
                        OrderBy(Ref(col, RawSQL(*self.query.extra[col])), descending=descending),
                        True))
        result = []
        seen = set()
        # ... other code
```
### 10 - django/db/models/functions/text.py:

Start line: 227, End line: 243

```python
class Reverse(Transform):
    function = 'REVERSE'
    lookup_name = 'reverse'

    def as_oracle(self, compiler, connection, **extra_context):
        # REVERSE in Oracle is undocumented and doesn't support multi-byte
        # strings. Use a special subquery instead.
        return super().as_sql(
            compiler, connection,
            template=(
                '(SELECT LISTAGG(s) WITHIN GROUP (ORDER BY n DESC) FROM '
                '(SELECT LEVEL n, SUBSTR(%(expressions)s, LEVEL, 1) s '
                'FROM DUAL CONNECT BY LEVEL <= LENGTH(%(expressions)s)) '
                'GROUP BY %(expressions)s)'
            ),
            **extra_context
        )
```
### 28 - django/db/models/sql/compiler.py:

Start line: 365, End line: 373

```python
class SQLCompiler:

    def get_extra_select(self, order_by, select):
        extra_select = []
        if self.query.distinct and not self.query.distinct_fields:
            select_sql = [t[1] for t in select]
            for expr, (sql, params, is_ref) in order_by:
                without_ordering = self.ordering_parts.search(sql).group(1)
                if not is_ref and (without_ordering, params) not in select_sql:
                    extra_select.append((expr, (without_ordering, params), None))
        return extra_select
```
### 37 - django/db/models/sql/compiler.py:

Start line: 717, End line: 728

```python
class SQLCompiler:

    def _setup_joins(self, pieces, opts, alias):
        """
        Helper method for get_order_by() and get_distinct().

        get_ordering() and get_distinct() must produce same target columns on
        same input, as the prefixes of get_ordering() and get_distinct() must
        match. Executing SQL where this is not true is an error.
        """
        alias = alias or self.query.get_initial_alias()
        field, targets, opts, joins, path, transform_function = self.query.setup_joins(pieces, opts, alias)
        alias = joins[-1]
        return field, targets, alias, joins, path, opts, transform_function
```
### 40 - django/db/models/sql/compiler.py:

Start line: 1, End line: 19

```python
import collections
import re
import warnings
from itertools import chain

from django.core.exceptions import EmptyResultSet, FieldError
from django.db.models.constants import LOOKUP_SEP
from django.db.models.expressions import OrderBy, Random, RawSQL, Ref
from django.db.models.query_utils import QueryWrapper, select_related_descend
from django.db.models.sql.constants import (
    CURSOR, GET_ITERATOR_CHUNK_SIZE, MULTI, NO_RESULTS, ORDER_DIR, SINGLE,
)
from django.db.models.sql.query import Query, get_order_dir
from django.db.transaction import TransactionManagementError
from django.db.utils import DatabaseError, NotSupportedError
from django.utils.deprecation import RemovedInDjango31Warning
from django.utils.hashable import make_hashable

FORCE = object()
```
### 62 - django/db/models/sql/compiler.py:

Start line: 22, End line: 43

```python
class SQLCompiler:
    def __init__(self, query, connection, using):
        self.query = query
        self.connection = connection
        self.using = using
        self.quote_cache = {'*': '*'}
        # The select, klass_info, and annotations are needed by QuerySet.iterator()
        # these are set as a side-effect of executing the query. Note that we calculate
        # separately a list of extra select columns needed for grammatical correctness
        # of the query, but these columns are not included in self.select.
        self.select = None
        self.annotation_col_map = None
        self.klass_info = None
        # Multiline ordering SQL clause may appear from RawSQL.
        self.ordering_parts = re.compile(r'^(.*)\s(ASC|DESC)(.*)', re.MULTILINE | re.DOTALL)
        self._meta_ordering = None

    def setup_query(self):
        if all(self.query.alias_refcount[a] == 0 for a in self.query.alias_map):
            self.query.get_initial_alias()
        self.select, self.klass_info, self.annotation_col_map = self.get_select()
        self.col_count = len(self.select)
```
### 67 - django/db/models/sql/compiler.py:

Start line: 1029, End line: 1051

```python
class SQLCompiler:

    def results_iter(self, results=None, tuple_expected=False, chunked_fetch=False,
                     chunk_size=GET_ITERATOR_CHUNK_SIZE):
        """Return an iterator over the results from executing this query."""
        if results is None:
            results = self.execute_sql(MULTI, chunked_fetch=chunked_fetch, chunk_size=chunk_size)
        fields = [s[0] for s in self.select[0:self.col_count]]
        converters = self.get_converters(fields)
        rows = chain.from_iterable(results)
        if converters:
            rows = self.apply_converters(rows, converters)
            if tuple_expected:
                rows = map(tuple, rows)
        return rows

    def has_results(self):
        """
        Backends (e.g. NoSQL) can override this in order to use optimized
        versions of "query has any results."
        """
        # This is always executed on a query clone, so we can modify self.query
        self.query.add_extra({'a': 1}, None, None, None, None, None)
        self.query.set_extra_mask(['a'])
        return bool(self.execute_sql(SINGLE))
```
### 70 - django/db/models/sql/compiler.py:

Start line: 137, End line: 181

```python
class SQLCompiler:

    def collapse_group_by(self, expressions, having):
        # If the DB can group by primary key, then group by the primary key of
        # query's main model. Note that for PostgreSQL the GROUP BY clause must
        # include the primary key of every table, but for MySQL it is enough to
        # have the main table's primary key.
        if self.connection.features.allows_group_by_pk:
            # Determine if the main model's primary key is in the query.
            pk = None
            for expr in expressions:
                # Is this a reference to query's base table primary key? If the
                # expression isn't a Col-like, then skip the expression.
                if (getattr(expr, 'target', None) == self.query.model._meta.pk and
                        getattr(expr, 'alias', None) == self.query.base_table):
                    pk = expr
                    break
            # If the main model's primary key is in the query, group by that
            # field, HAVING expressions, and expressions associated with tables
            # that don't have a primary key included in the grouped columns.
            if pk:
                pk_aliases = {
                    expr.alias for expr in expressions
                    if hasattr(expr, 'target') and expr.target.primary_key
                }
                expressions = [pk] + [
                    expr for expr in expressions
                    if expr in having or (
                        getattr(expr, 'alias', None) is not None and expr.alias not in pk_aliases
                    )
                ]
        elif self.connection.features.allows_group_by_selected_pks:
            # Filter out all expressions associated with a table's primary key
            # present in the grouped columns. This is done by identifying all
            # tables that have their primary key included in the grouped
            # columns and removing non-primary key columns referring to them.
            # Unmanaged models are excluded because they could be representing
            # database views on which the optimization might not be allowed.
            pks = {
                expr for expr in expressions
                if hasattr(expr, 'target') and expr.target.primary_key and expr.target.model._meta.managed
            }
            aliases = {expr.alias for expr in pks}
            expressions = [
                expr for expr in expressions if expr in pks or getattr(expr, 'alias', None) not in aliases
            ]
        return expressions
```
### 74 - django/db/models/sql/compiler.py:

Start line: 686, End line: 715

```python
class SQLCompiler:

    def find_ordering_name(self, name, opts, alias=None, default_order='ASC',
                           already_seen=None):
        """
        Return the table alias (the name might be ambiguous, the alias will
        not be) and column name for ordering by the given 'name' parameter.
        The 'name' is of the form 'field1__field2__...__fieldN'.
        """
        name, order = get_order_dir(name, default_order)
        descending = order == 'DESC'
        pieces = name.split(LOOKUP_SEP)
        field, targets, alias, joins, path, opts, transform_function = self._setup_joins(pieces, opts, alias)

        # If we get to this point and the field is a relation to another model,
        # append the default ordering for that model unless the attribute name
        # of the field is specified.
        if field.is_relation and opts.ordering and getattr(field, 'attname', None) != name:
            # Firstly, avoid infinite loops.
            already_seen = already_seen or set()
            join_tuple = tuple(getattr(self.query.alias_map[j], 'join_cols', None) for j in joins)
            if join_tuple in already_seen:
                raise FieldError('Infinite loop caused by ordering.')
            already_seen.add(join_tuple)

            results = []
            for item in opts.ordering:
                results.extend(self.find_ordering_name(item, opts, alias,
                                                       order, already_seen))
            return results
        targets, alias, _ = self.query.trim_joins(targets, joins, path)
        return [(OrderBy(transform_function(t, alias), descending=descending), False) for t in targets]
```
### 92 - django/db/models/sql/compiler.py:

Start line: 1120, End line: 1141

```python
class SQLCompiler:

    def as_subquery_condition(self, alias, columns, compiler):
        qn = compiler.quote_name_unless_alias
        qn2 = self.connection.ops.quote_name

        for index, select_col in enumerate(self.query.select):
            lhs_sql, lhs_params = self.compile(select_col)
            rhs = '%s.%s' % (qn(alias), qn2(columns[index]))
            self.query.where.add(
                QueryWrapper('%s = %s' % (lhs_sql, rhs), lhs_params), 'AND')

        sql, params = self.as_sql()
        return 'EXISTS (%s)' % sql, params

    def explain_query(self):
        result = list(self.execute_sql())
        # Some backends return 1 item tuples with strings, and others return
        # tuples with integers and strings. Flatten them out into strings.
        for row in result[0]:
            if not isinstance(row, str):
                yield ' '.join(str(c) for c in row)
            else:
                yield row
```
### 101 - django/db/models/sql/compiler.py:

Start line: 1436, End line: 1476

```python
class SQLUpdateCompiler(SQLCompiler):

    def pre_sql_setup(self):
        """
        If the update depends on results from other tables, munge the "where"
        conditions to match the format required for (portable) SQL updates.

        If multiple updates are required, pull out the id values to update at
        this point so that they don't change as a result of the progressive
        updates.
        """
        refcounts_before = self.query.alias_refcount.copy()
        # Ensure base table is in the query
        self.query.get_initial_alias()
        count = self.query.count_active_tables()
        if not self.query.related_updates and count == 1:
            return
        query = self.query.chain(klass=Query)
        query.select_related = False
        query.clear_ordering(True)
        query.extra = {}
        query.select = []
        query.add_fields([query.get_meta().pk.name])
        super().pre_sql_setup()

        must_pre_select = count > 1 and not self.connection.features.update_can_self_select

        # Now we adjust the current query: reset the where clause and get rid
        # of all the tables we don't need (since they're in the sub-select).
        self.query.where = self.query.where_class()
        if self.query.related_updates or must_pre_select:
            # Either we're using the idents in multiple update queries (so
            # don't want them to change), or the db backend doesn't support
            # selecting from the updating table (e.g. MySQL).
            idents = []
            for rows in query.get_compiler(self.using).execute_sql(MULTI):
                idents.extend(r[0] for r in rows)
            self.query.add_filter(('pk__in', idents))
            self.query.related_ids = idents
        else:
            # The fast path. Filters and updates in one query.
            self.query.add_filter(('pk__in', query))
        self.query.reset_refcounts(refcounts_before)
```
### 116 - django/db/models/sql/compiler.py:

Start line: 939, End line: 961

```python
class SQLCompiler:

    def get_select_for_update_of_arguments(self):
        """
        Return a quoted list of arguments for the SELECT FOR UPDATE OF part of
        the query.
        """
        def _get_field_choices():
            """Yield all allowed field paths in breadth-first search order."""
            queue = collections.deque([(None, self.klass_info)])
            while queue:
                parent_path, klass_info = queue.popleft()
                if parent_path is None:
                    path = []
                    yield 'self'
                else:
                    field = klass_info['field']
                    if klass_info['reverse']:
                        field = field.remote_field
                    path = parent_path + [field.name]
                    yield LOOKUP_SEP.join(path)
                queue.extend(
                    (path, klass_info)
                    for klass_info in klass_info.get('related_klass_infos', [])
                )
        # ... other code
```
### 117 - django/db/models/sql/compiler.py:

Start line: 402, End line: 454

```python
class SQLCompiler:

    def get_combinator_sql(self, combinator, all):
        features = self.connection.features
        compilers = [
            query.get_compiler(self.using, self.connection)
            for query in self.query.combined_queries if not query.is_empty()
        ]
        if not features.supports_slicing_ordering_in_compound:
            for query, compiler in zip(self.query.combined_queries, compilers):
                if query.low_mark or query.high_mark:
                    raise DatabaseError('LIMIT/OFFSET not allowed in subqueries of compound statements.')
                if compiler.get_order_by():
                    raise DatabaseError('ORDER BY not allowed in subqueries of compound statements.')
        parts = ()
        for compiler in compilers:
            try:
                # If the columns list is limited, then all combined queries
                # must have the same columns list. Set the selects defined on
                # the query on all combined queries, if not already set.
                if not compiler.query.values_select and self.query.values_select:
                    compiler.query.set_values((
                        *self.query.extra_select,
                        *self.query.values_select,
                        *self.query.annotation_select,
                    ))
                part_sql, part_args = compiler.as_sql()
                if compiler.query.combinator:
                    # Wrap in a subquery if wrapping in parentheses isn't
                    # supported.
                    if not features.supports_parentheses_in_compound:
                        part_sql = 'SELECT * FROM ({})'.format(part_sql)
                    # Add parentheses when combining with compound query if not
                    # already added for all compound queries.
                    elif not features.supports_slicing_ordering_in_compound:
                        part_sql = '({})'.format(part_sql)
                parts += ((part_sql, part_args),)
            except EmptyResultSet:
                # Omit the empty queryset with UNION and with DIFFERENCE if the
                # first queryset is nonempty.
                if combinator == 'union' or (combinator == 'difference' and parts):
                    continue
                raise
        if not parts:
            raise EmptyResultSet
        combinator_sql = self.connection.ops.set_operators[combinator]
        if all and combinator == 'union':
            combinator_sql += ' ALL'
        braces = '({})' if features.supports_slicing_ordering_in_compound else '{}'
        sql_parts, args_parts = zip(*((braces.format(sql), args) for sql, args in parts))
        result = [' {} '.format(combinator_sql).join(sql_parts)]
        params = []
        for part in args_parts:
            params.extend(part)
        return result, params
```
