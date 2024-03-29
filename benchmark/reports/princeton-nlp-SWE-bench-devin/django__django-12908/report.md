# django__django-12908

| **django/django** | `49ae7ce50a874f8a04cd910882fb9571ff3a0d7a` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 1559 |
| **Any found context length** | 1559 |
| **Avg pos** | 6.0 |
| **Min pos** | 6 |
| **Max pos** | 6 |
| **Top file pos** | 2 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/db/models/query.py b/django/db/models/query.py
--- a/django/db/models/query.py
+++ b/django/db/models/query.py
@@ -1138,6 +1138,7 @@ def distinct(self, *field_names):
         """
         Return a new QuerySet instance that will select only distinct results.
         """
+        self._not_support_combined_queries('distinct')
         assert not self.query.is_sliced, \
             "Cannot create distinct fields once a slice has been taken."
         obj = self._chain()

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/db/models/query.py | 1141 | 1141 | 6 | 2 | 1559


## Problem Statement

```
Union queryset should raise on distinct().
Description
	 
		(last modified by Sielc Technologies)
	 
After using
.annotate() on 2 different querysets
and then .union()
.distinct() will not affect the queryset
	def setUp(self) -> None:
		user = self.get_or_create_admin_user()
		Sample.h.create(user, name="Sam1")
		Sample.h.create(user, name="Sam2 acid")
		Sample.h.create(user, name="Sam3")
		Sample.h.create(user, name="Sam4 acid")
		Sample.h.create(user, name="Dub")
		Sample.h.create(user, name="Dub")
		Sample.h.create(user, name="Dub")
		self.user = user
	def test_union_annotated_diff_distinct(self):
		qs = Sample.objects.filter(user=self.user)
		qs1 = qs.filter(name='Dub').annotate(rank=Value(0, IntegerField()))
		qs2 = qs.filter(name='Sam1').annotate(rank=Value(1, IntegerField()))
		qs = qs1.union(qs2)
		qs = qs.order_by('name').distinct('name') # THIS DISTINCT DOESN'T WORK
		self.assertEqual(qs.count(), 2)
expected to get wrapped union
	SELECT DISTINCT ON (siebox_sample.name) * FROM (SELECT ... UNION SELECT ...) AS siebox_sample

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 django/db/models/sql/compiler.py | 680 | 702| 202 | 202 | 14081 | 
| 2 | **2 django/db/models/query.py** | 988 | 1019| 341 | 543 | 31157 | 
| 3 | **2 django/db/models/query.py** | 1321 | 1330| 114 | 657 | 31157 | 
| 4 | **2 django/db/models/query.py** | 804 | 843| 322 | 979 | 31157 | 
| 5 | 3 django/db/models/aggregates.py | 70 | 96| 266 | 1245 | 32458 | 
| **-> 6 <-** | **3 django/db/models/query.py** | 1128 | 1163| 314 | 1559 | 32458 | 
| 7 | **3 django/db/models/query.py** | 327 | 353| 222 | 1781 | 32458 | 
| 8 | 4 django/contrib/admin/utils.py | 27 | 49| 199 | 1980 | 36610 | 
| 9 | 4 django/db/models/sql/compiler.py | 740 | 751| 163 | 2143 | 36610 | 
| 10 | 5 django/db/backends/base/schema.py | 370 | 384| 182 | 2325 | 48210 | 
| 11 | 6 django/db/models/base.py | 1073 | 1116| 404 | 2729 | 63998 | 
| 12 | 7 django/db/models/sql/where.py | 230 | 246| 130 | 2859 | 65802 | 
| 13 | 7 django/db/models/base.py | 1561 | 1586| 183 | 3042 | 65802 | 
| 14 | **7 django/db/models/query.py** | 1301 | 1319| 186 | 3228 | 65802 | 
| 15 | **7 django/db/models/query.py** | 1081 | 1126| 349 | 3577 | 65802 | 
| 16 | 8 django/db/models/constraints.py | 79 | 154| 666 | 4243 | 67054 | 
| 17 | 9 django/db/models/sql/query.py | 1658 | 1692| 399 | 4642 | 88988 | 
| 18 | **9 django/db/models/query.py** | 1332 | 1380| 405 | 5047 | 88988 | 
| 19 | **9 django/db/models/query.py** | 1203 | 1237| 234 | 5281 | 88988 | 
| 20 | 10 django/db/models/sql/datastructures.py | 117 | 137| 144 | 5425 | 90390 | 
| 21 | 10 django/db/models/sql/query.py | 509 | 543| 282 | 5707 | 90390 | 
| 22 | 11 django/db/models/query_utils.py | 312 | 352| 286 | 5993 | 93102 | 
| 23 | 11 django/db/models/sql/query.py | 1592 | 1616| 227 | 6220 | 93102 | 
| 24 | 11 django/db/models/sql/query.py | 620 | 644| 269 | 6489 | 93102 | 
| 25 | 11 django/db/models/aggregates.py | 45 | 68| 294 | 6783 | 93102 | 
| 26 | 12 django/contrib/admin/options.py | 1020 | 1031| 149 | 6932 | 111657 | 
| 27 | **12 django/db/models/query.py** | 919 | 969| 381 | 7313 | 111657 | 
| 28 | 12 django/db/models/sql/query.py | 891 | 913| 248 | 7561 | 111657 | 
| 29 | 12 django/db/models/sql/query.py | 1796 | 1845| 330 | 7891 | 111657 | 
| 30 | 13 django/db/migrations/operations/models.py | 530 | 549| 148 | 8039 | 118227 | 
| 31 | 13 django/db/models/sql/query.py | 1407 | 1418| 137 | 8176 | 118227 | 
| 32 | 13 django/db/models/sql/query.py | 545 | 619| 809 | 8985 | 118227 | 
| 33 | 14 django/contrib/gis/db/backends/postgis/operations.py | 160 | 188| 302 | 9287 | 121835 | 
| 34 | **14 django/db/models/query.py** | 243 | 270| 221 | 9508 | 121835 | 
| 35 | 14 django/db/models/base.py | 1164 | 1192| 213 | 9721 | 121835 | 
| 36 | 14 django/db/models/sql/query.py | 2162 | 2194| 228 | 9949 | 121835 | 
| 37 | 14 django/db/backends/base/schema.py | 1108 | 1137| 233 | 10182 | 121835 | 
| 38 | 15 django/contrib/admin/checks.py | 988 | 1003| 136 | 10318 | 130842 | 
| 39 | 15 django/db/models/sql/query.py | 799 | 825| 280 | 10598 | 130842 | 
| 40 | 15 django/db/backends/base/schema.py | 1086 | 1106| 176 | 10774 | 130842 | 
| 41 | 16 django/db/models/fields/related.py | 509 | 574| 492 | 11266 | 144673 | 
| 42 | **16 django/db/models/query.py** | 753 | 786| 274 | 11540 | 144673 | 
| 43 | 16 django/db/models/sql/query.py | 1694 | 1765| 784 | 12324 | 144673 | 
| 44 | 16 django/contrib/admin/options.py | 985 | 1018| 310 | 12634 | 144673 | 
| 45 | 16 django/contrib/admin/checks.py | 354 | 370| 134 | 12768 | 144673 | 
| 46 | **16 django/db/models/query.py** | 1466 | 1499| 297 | 13065 | 144673 | 
| 47 | 16 django/db/models/sql/compiler.py | 63 | 146| 867 | 13932 | 144673 | 
| 48 | 17 django/db/models/sql/subqueries.py | 1 | 44| 320 | 14252 | 145886 | 
| 49 | **17 django/db/models/query.py** | 724 | 751| 235 | 14487 | 145886 | 
| 50 | 17 django/db/models/sql/query.py | 1931 | 1980| 420 | 14907 | 145886 | 
| 51 | **17 django/db/models/query.py** | 272 | 292| 180 | 15087 | 145886 | 
| 52 | 18 django/contrib/gis/db/models/functions.py | 459 | 487| 225 | 15312 | 149813 | 
| 53 | 18 django/db/models/sql/query.py | 1280 | 1338| 711 | 16023 | 149813 | 
| 54 | 19 django/contrib/admin/views/main.py | 332 | 392| 508 | 16531 | 154111 | 
| 55 | **19 django/db/models/query.py** | 845 | 874| 248 | 16779 | 154111 | 
| 56 | 19 django/db/backends/base/schema.py | 1139 | 1174| 273 | 17052 | 154111 | 
| 57 | 20 django/forms/models.py | 680 | 751| 732 | 17784 | 165832 | 
| 58 | 20 django/db/models/sql/subqueries.py | 47 | 75| 210 | 17994 | 165832 | 
| 59 | 20 django/db/models/sql/query.py | 2124 | 2160| 292 | 18286 | 165832 | 
| 60 | 20 django/db/models/sql/query.py | 1 | 65| 465 | 18751 | 165832 | 
| 61 | 20 django/contrib/admin/checks.py | 231 | 261| 229 | 18980 | 165832 | 
| 62 | 21 django/contrib/postgres/aggregates/mixins.py | 22 | 34| 128 | 19108 | 166274 | 
| 63 | 21 django/db/models/sql/query.py | 1016 | 1046| 325 | 19433 | 166274 | 
| 64 | 21 django/db/models/query_utils.py | 167 | 231| 492 | 19925 | 166274 | 
| 65 | 22 django/contrib/gis/db/models/aggregates.py | 49 | 84| 207 | 20132 | 166891 | 
| 66 | 22 django/db/models/sql/query.py | 136 | 229| 823 | 20955 | 166891 | 
| 67 | 23 django/db/models/expressions.py | 1013 | 1065| 371 | 21326 | 177117 | 
| 68 | 23 django/forms/models.py | 753 | 774| 194 | 21520 | 177117 | 
| 69 | 23 django/db/models/sql/query.py | 915 | 933| 146 | 21666 | 177117 | 
| 70 | **23 django/db/models/query.py** | 670 | 684| 132 | 21798 | 177117 | 
| 71 | 24 django/db/models/lookups.py | 606 | 639| 141 | 21939 | 182043 | 
| 72 | 24 django/db/models/sql/query.py | 1436 | 1514| 734 | 22673 | 182043 | 
| 73 | 24 django/db/models/sql/subqueries.py | 137 | 163| 173 | 22846 | 182043 | 
| 74 | 24 django/db/models/sql/query.py | 1982 | 2028| 370 | 23216 | 182043 | 
| 75 | 24 django/db/models/aggregates.py | 122 | 158| 245 | 23461 | 182043 | 
| 76 | 25 django/db/models/deletion.py | 1 | 76| 566 | 24027 | 185866 | 
| 77 | 25 django/contrib/admin/checks.py | 263 | 276| 135 | 24162 | 185866 | 
| 78 | 26 django/db/migrations/autodetector.py | 1125 | 1146| 231 | 24393 | 197595 | 
| 79 | 26 django/db/models/base.py | 1015 | 1071| 560 | 24953 | 197595 | 
| 80 | 27 django/db/backends/sqlite3/creation.py | 84 | 104| 174 | 25127 | 198446 | 
| 81 | **27 django/db/models/query.py** | 1182 | 1201| 209 | 25336 | 198446 | 
| 82 | **27 django/db/models/query.py** | 1275 | 1299| 210 | 25546 | 198446 | 
| 83 | 27 django/contrib/admin/checks.py | 753 | 768| 182 | 25728 | 198446 | 
| 84 | **27 django/db/models/query.py** | 1036 | 1056| 155 | 25883 | 198446 | 
| 85 | 27 django/db/models/sql/compiler.py | 22 | 47| 257 | 26140 | 198446 | 
| 86 | 27 django/db/models/sql/query.py | 866 | 889| 203 | 26343 | 198446 | 
| 87 | 27 django/db/models/deletion.py | 379 | 448| 577 | 26920 | 198446 | 
| 88 | 27 django/db/models/sql/query.py | 1104 | 1133| 324 | 27244 | 198446 | 
| 89 | 27 django/db/models/sql/query.py | 1340 | 1361| 250 | 27494 | 198446 | 
| 90 | 27 django/contrib/postgres/aggregates/mixins.py | 1 | 20| 182 | 27676 | 198446 | 
| 91 | 27 django/db/models/constraints.py | 1 | 29| 213 | 27889 | 198446 | 
| 92 | 27 django/db/models/expressions.py | 1068 | 1095| 261 | 28150 | 198446 | 
| 93 | 27 django/db/models/sql/query.py | 1048 | 1073| 214 | 28364 | 198446 | 
| 94 | 27 django/db/migrations/operations/models.py | 1 | 38| 235 | 28599 | 198446 | 
| 95 | **27 django/db/models/query.py** | 1 | 41| 299 | 28898 | 198446 | 
| 96 | **27 django/db/models/query.py** | 372 | 411| 325 | 29223 | 198446 | 
| 97 | 28 django/db/models/__init__.py | 1 | 53| 619 | 29842 | 199065 | 
| 98 | **28 django/db/models/query.py** | 1426 | 1464| 308 | 30150 | 199065 | 
| 99 | 28 django/db/models/sql/datastructures.py | 1 | 21| 126 | 30276 | 199065 | 
| 100 | 28 django/db/models/expressions.py | 763 | 797| 292 | 30568 | 199065 | 
| 101 | 28 django/db/models/sql/query.py | 287 | 335| 424 | 30992 | 199065 | 


### Hint

```
distinct() is not supported but doesn't raise an error yet. As ​​per the documentation, "only LIMIT, OFFSET, COUNT(*), ORDER BY, and specifying columns (i.e. slicing, count(), order_by(), and values()/values_list()) are allowed on the resulting QuerySet.". Follow up to #27995.
```

## Patch

```diff
diff --git a/django/db/models/query.py b/django/db/models/query.py
--- a/django/db/models/query.py
+++ b/django/db/models/query.py
@@ -1138,6 +1138,7 @@ def distinct(self, *field_names):
         """
         Return a new QuerySet instance that will select only distinct results.
         """
+        self._not_support_combined_queries('distinct')
         assert not self.query.is_sliced, \
             "Cannot create distinct fields once a slice has been taken."
         obj = self._chain()

```

## Test Patch

```diff
diff --git a/tests/queries/test_qs_combinators.py b/tests/queries/test_qs_combinators.py
--- a/tests/queries/test_qs_combinators.py
+++ b/tests/queries/test_qs_combinators.py
@@ -272,6 +272,7 @@ def test_unsupported_operations_on_combined_qs(self):
                 'annotate',
                 'defer',
                 'delete',
+                'distinct',
                 'exclude',
                 'extra',
                 'filter',

```


## Code snippets

### 1 - django/db/models/sql/compiler.py:

Start line: 680, End line: 702

```python
class SQLCompiler:
    ordering_parts =
    # ... other code

    def get_distinct(self):
        """
        Return a quoted list of fields to use in DISTINCT ON part of the query.

        This method can alter the tables in the query, and thus it must be
        called before get_from_clause().
        """
        result = []
        params = []
        opts = self.query.get_meta()

        for name in self.query.distinct_fields:
            parts = name.split(LOOKUP_SEP)
            _, targets, alias, joins, path, _, transform_function = self._setup_joins(parts, opts, None)
            targets, alias, _ = self.query.trim_joins(targets, joins, path)
            for target in targets:
                if name in self.query.annotation_select:
                    result.append(name)
                else:
                    r, p = self.compile(transform_function(target, alias))
                    result.append(r)
                    params.append(p)
        return result, params
    # ... other code
```
### 2 - django/db/models/query.py:

Start line: 988, End line: 1019

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
### 3 - django/db/models/query.py:

Start line: 1321, End line: 1330

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
### 4 - django/db/models/query.py:

Start line: 804, End line: 843

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
### 5 - django/db/models/aggregates.py:

Start line: 70, End line: 96

```python
class Aggregate(Func):

    def as_sql(self, compiler, connection, **extra_context):
        extra_context['distinct'] = 'DISTINCT ' if self.distinct else ''
        if self.filter:
            if connection.features.supports_aggregate_filter_clause:
                filter_sql, filter_params = self.filter.as_sql(compiler, connection)
                template = self.filter_template % extra_context.get('template', self.template)
                sql, params = super().as_sql(
                    compiler, connection, template=template, filter=filter_sql,
                    **extra_context
                )
                return sql, params + filter_params
            else:
                copy = self.copy()
                copy.filter = None
                source_expressions = copy.get_source_expressions()
                condition = When(self.filter, then=source_expressions[0])
                copy.set_source_expressions([Case(condition)] + source_expressions[1:])
                return super(Aggregate, copy).as_sql(compiler, connection, **extra_context)
        return super().as_sql(compiler, connection, **extra_context)

    def _get_repr_options(self):
        options = super()._get_repr_options()
        if self.distinct:
            options['distinct'] = self.distinct
        if self.filter:
            options['filter'] = self.filter
        return options
```
### 6 - django/db/models/query.py:

Start line: 1128, End line: 1163

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
### 7 - django/db/models/query.py:

Start line: 327, End line: 353

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
### 8 - django/contrib/admin/utils.py:

Start line: 27, End line: 49

```python
def lookup_needs_distinct(opts, lookup_path):
    """
    Return True if 'distinct()' should be used to query the given lookup path.
    """
    lookup_fields = lookup_path.split(LOOKUP_SEP)
    # Go through the fields (following all relations) and look for an m2m.
    for field_name in lookup_fields:
        if field_name == 'pk':
            field_name = opts.pk.name
        try:
            field = opts.get_field(field_name)
        except FieldDoesNotExist:
            # Ignore query lookups.
            continue
        else:
            if hasattr(field, 'get_path_info'):
                # This field is a relation; update opts to follow the relation.
                path_info = field.get_path_info()
                opts = path_info[-1].to_opts
                if any(path.m2m for path in path_info):
                    # This field is a m2m relation so distinct must be called.
                    return True
    return False
```
### 9 - django/db/models/sql/compiler.py:

Start line: 740, End line: 751

```python
class SQLCompiler:
    ordering_parts =
    # ... other code

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
    # ... other code
```
### 10 - django/db/backends/base/schema.py:

Start line: 370, End line: 384

```python
class BaseDatabaseSchemaEditor:

    def alter_unique_together(self, model, old_unique_together, new_unique_together):
        """
        Deal with a model changing its unique_together. The input
        unique_togethers must be doubly-nested, not the single-nested
        ["foo", "bar"] format.
        """
        olds = {tuple(fields) for fields in old_unique_together}
        news = {tuple(fields) for fields in new_unique_together}
        # Deleted uniques
        for fields in olds.difference(news):
            self._delete_composed_index(model, fields, {'unique': True}, self.sql_delete_unique)
        # Created uniques
        for fields in news.difference(olds):
            columns = [model._meta.get_field(field).column for field in fields]
            self.execute(self._create_unique_sql(model, columns))
```
### 14 - django/db/models/query.py:

Start line: 1301, End line: 1319

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
### 15 - django/db/models/query.py:

Start line: 1081, End line: 1126

```python
class QuerySet:

    def annotate(self, *args, **kwargs):
        """
        Return a query set in which the returned objects have been annotated
        with extra data or aggregations.
        """
        self._not_support_combined_queries('annotate')
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
                clone.query.add_annotation(annotation, alias, is_summary=False)

        for alias, annotation in clone.query.annotations.items():
            if alias in annotations and annotation.contains_aggregate:
                if clone._fields is None:
                    clone.query.group_by = True
                else:
                    clone.query.set_group_by()
                break

        return clone
```
### 18 - django/db/models/query.py:

Start line: 1332, End line: 1380

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
### 19 - django/db/models/query.py:

Start line: 1203, End line: 1237

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
        elif self.query.default_ordering and self.query.get_meta().ordering:
            return True
        else:
            return False

    @property
    def db(self):
        """Return the database used if this query is executed now."""
        if self._for_write:
            return self._db or router.db_for_write(self.model, **self._hints)
        return self._db or router.db_for_read(self.model, **self._hints)

    ###################
    # PRIVATE METHODS #
    ###################
```
### 27 - django/db/models/query.py:

Start line: 919, End line: 969

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
        return self._filter_or_exclude(False, *args, **kwargs)

    def exclude(self, *args, **kwargs):
        """
        Return a new QuerySet instance with NOT (args) ANDed to the existing
        set.
        """
        self._not_support_combined_queries('exclude')
        return self._filter_or_exclude(True, *args, **kwargs)

    def _filter_or_exclude(self, negate, *args, **kwargs):
        if args or kwargs:
            assert not self.query.is_sliced, \
                "Cannot filter a query once a slice has been taken."

        clone = self._chain()
        if self._defer_next_filter:
            self._defer_next_filter = False
            clone._deferred_filter = negate, args, kwargs
        else:
            clone._filter_or_exclude_inplace(negate, *args, **kwargs)
        return clone

    def _filter_or_exclude_inplace(self, negate, *args, **kwargs):
        if negate:
            self._query.add_q(~Q(*args, **kwargs))
        else:
            self._query.add_q(Q(*args, **kwargs))
```
### 34 - django/db/models/query.py:

Start line: 243, End line: 270

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
### 42 - django/db/models/query.py:

Start line: 753, End line: 786

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
### 46 - django/db/models/query.py:

Start line: 1466, End line: 1499

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
### 49 - django/db/models/query.py:

Start line: 724, End line: 751

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
### 51 - django/db/models/query.py:

Start line: 272, End line: 292

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
### 55 - django/db/models/query.py:

Start line: 845, End line: 874

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
### 70 - django/db/models/query.py:

Start line: 670, End line: 684

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
### 81 - django/db/models/query.py:

Start line: 1182, End line: 1201

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
### 82 - django/db/models/query.py:

Start line: 1275, End line: 1299

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
### 84 - django/db/models/query.py:

Start line: 1036, End line: 1056

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
### 95 - django/db/models/query.py:

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
### 96 - django/db/models/query.py:

Start line: 372, End line: 411

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
### 98 - django/db/models/query.py:

Start line: 1426, End line: 1464

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
