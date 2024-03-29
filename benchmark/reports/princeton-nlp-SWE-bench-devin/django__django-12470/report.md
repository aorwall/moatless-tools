# django__django-12470

| **django/django** | `142ab6846ac09d6d401e26fc8b6b988a583ac0f5` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 4244 |
| **Any found context length** | 4244 |
| **Avg pos** | 14.0 |
| **Min pos** | 14 |
| **Max pos** | 14 |
| **Top file pos** | 5 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/db/models/sql/compiler.py b/django/db/models/sql/compiler.py
--- a/django/db/models/sql/compiler.py
+++ b/django/db/models/sql/compiler.py
@@ -709,9 +709,9 @@ def find_ordering_name(self, name, opts, alias=None, default_order='ASC',
         field, targets, alias, joins, path, opts, transform_function = self._setup_joins(pieces, opts, alias)
 
         # If we get to this point and the field is a relation to another model,
-        # append the default ordering for that model unless the attribute name
-        # of the field is specified.
-        if field.is_relation and opts.ordering and getattr(field, 'attname', None) != name:
+        # append the default ordering for that model unless it is the pk
+        # shortcut or the attribute name of the field that is specified.
+        if field.is_relation and opts.ordering and getattr(field, 'attname', None) != name and name != 'pk':
             # Firstly, avoid infinite loops.
             already_seen = already_seen or set()
             join_tuple = tuple(getattr(self.query.alias_map[j], 'join_cols', None) for j in joins)

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/db/models/sql/compiler.py | 712 | 714 | 14 | 5 | 4244


## Problem Statement

```
Inherited model doesn't correctly order by "-pk" when specified on Parent.Meta.ordering
Description
	
Given the following model definition:
from django.db import models
class Parent(models.Model):
	class Meta:
		ordering = ["-pk"]
class Child(Parent):
	pass
Querying the Child class results in the following:
>>> print(Child.objects.all().query)
SELECT "myapp_parent"."id", "myapp_child"."parent_ptr_id" FROM "myapp_child" INNER JOIN "myapp_parent" ON ("myapp_child"."parent_ptr_id" = "myapp_parent"."id") ORDER BY "myapp_parent"."id" ASC
The query is ordered ASC but I expect the order to be DESC.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 django/db/models/base.py | 968 | 981| 180 | 180 | 15355 | 
| 2 | 1 django/db/models/base.py | 1867 | 1918| 351 | 531 | 15355 | 
| 3 | 2 django/contrib/admin/options.py | 205 | 216| 135 | 666 | 33843 | 
| 4 | 2 django/db/models/base.py | 1665 | 1763| 717 | 1383 | 33843 | 
| 5 | 2 django/db/models/base.py | 952 | 966| 212 | 1595 | 33843 | 
| 6 | 3 django/contrib/admin/views/main.py | 332 | 392| 508 | 2103 | 38141 | 
| 7 | 4 django/db/models/expressions.py | 1115 | 1140| 248 | 2351 | 48322 | 
| 8 | **5 django/db/models/sql/compiler.py** | 354 | 387| 388 | 2739 | 62378 | 
| 9 | 6 django/db/models/options.py | 579 | 605| 197 | 2936 | 69474 | 
| 10 | 6 django/db/models/expressions.py | 1142 | 1172| 218 | 3154 | 69474 | 
| 11 | 6 django/db/models/expressions.py | 1091 | 1113| 188 | 3342 | 69474 | 
| 12 | 7 django/db/migrations/autodetector.py | 337 | 356| 196 | 3538 | 81209 | 
| 13 | 8 django/db/models/sql/query.py | 984 | 1015| 307 | 3845 | 102878 | 
| **-> 14 <-** | **8 django/db/models/sql/compiler.py** | 699 | 733| 399 | 4244 | 102878 | 
| 15 | 8 django/db/models/sql/query.py | 1886 | 1919| 279 | 4523 | 102878 | 
| 16 | 9 django/db/migrations/operations/models.py | 609 | 622| 137 | 4660 | 109574 | 
| 17 | 9 django/db/migrations/autodetector.py | 1182 | 1207| 245 | 4905 | 109574 | 
| 18 | 9 django/contrib/admin/views/main.py | 257 | 287| 270 | 5175 | 109574 | 
| 19 | 10 django/db/models/fields/related_descriptors.py | 120 | 154| 405 | 5580 | 119969 | 
| 20 | 10 django/db/migrations/operations/models.py | 566 | 589| 183 | 5763 | 119969 | 
| 21 | 10 django/db/models/base.py | 404 | 503| 856 | 6619 | 119969 | 
| 22 | 11 django/db/models/__init__.py | 1 | 52| 605 | 7224 | 120574 | 
| 23 | 11 django/db/models/fields/related_descriptors.py | 365 | 381| 184 | 7408 | 120574 | 
| 24 | 12 django/db/models/query.py | 1400 | 1408| 136 | 7544 | 137593 | 
| 25 | 13 django/views/generic/list.py | 50 | 75| 244 | 7788 | 139165 | 
| 26 | 13 django/db/models/sql/query.py | 1437 | 1515| 734 | 8522 | 139165 | 
| 27 | 13 django/db/models/sql/query.py | 2256 | 2267| 116 | 8638 | 139165 | 
| 28 | 13 django/db/models/base.py | 802 | 830| 307 | 8945 | 139165 | 
| 29 | 14 django/contrib/contenttypes/fields.py | 356 | 394| 405 | 9350 | 144598 | 
| 30 | 14 django/db/migrations/operations/models.py | 591 | 607| 215 | 9565 | 144598 | 
| 31 | 14 django/db/models/base.py | 212 | 322| 866 | 10431 | 144598 | 
| 32 | 14 django/db/models/fields/related_descriptors.py | 494 | 548| 338 | 10769 | 144598 | 
| 33 | 15 django/db/models/fields/related.py | 1235 | 1352| 963 | 11732 | 158459 | 
| 34 | 15 django/db/models/fields/related_descriptors.py | 344 | 363| 156 | 11888 | 158459 | 
| 35 | 16 django/db/models/deletion.py | 361 | 377| 130 | 12018 | 162274 | 
| 36 | 16 django/db/models/options.py | 1 | 34| 285 | 12303 | 162274 | 
| 37 | **16 django/db/models/sql/compiler.py** | 735 | 746| 147 | 12450 | 162274 | 
| 38 | **16 django/db/models/sql/compiler.py** | 265 | 352| 696 | 13146 | 162274 | 
| 39 | 17 django/db/backends/mysql/operations.py | 143 | 174| 294 | 13440 | 165605 | 
| 40 | 18 django/core/serializers/__init__.py | 159 | 235| 676 | 14116 | 167253 | 
| 41 | 19 django/forms/models.py | 629 | 646| 167 | 14283 | 178974 | 
| 42 | 20 django/contrib/postgres/aggregates/mixins.py | 1 | 20| 182 | 14465 | 179416 | 
| 43 | 20 django/db/models/query.py | 1108 | 1143| 314 | 14779 | 179416 | 
| 44 | 20 django/db/models/base.py | 1498 | 1530| 231 | 15010 | 179416 | 
| 45 | 20 django/db/models/sql/query.py | 867 | 890| 203 | 15213 | 179416 | 
| 46 | 21 django/db/models/sql/datastructures.py | 104 | 115| 133 | 15346 | 180818 | 
| 47 | 21 django/db/models/fields/related_descriptors.py | 309 | 323| 182 | 15528 | 180818 | 
| 48 | 22 django/db/models/sql/where.py | 117 | 140| 185 | 15713 | 182622 | 
| 49 | 22 django/db/models/base.py | 169 | 211| 413 | 16126 | 182622 | 
| 50 | 22 django/db/models/query.py | 1534 | 1590| 481 | 16607 | 182622 | 
| 51 | 22 django/db/models/query.py | 666 | 680| 132 | 16739 | 182622 | 
| 52 | 22 django/db/models/fields/related_descriptors.py | 156 | 201| 445 | 17184 | 182622 | 
| 53 | 22 django/forms/models.py | 989 | 1042| 470 | 17654 | 182622 | 
| 54 | 23 django/core/paginator.py | 113 | 129| 142 | 17796 | 183909 | 
| 55 | 23 django/db/models/sql/query.py | 696 | 731| 389 | 18185 | 183909 | 
| 56 | 23 django/db/models/base.py | 1253 | 1283| 257 | 18442 | 183909 | 
| 57 | 23 django/db/models/query.py | 1939 | 1962| 200 | 18642 | 183909 | 
| 58 | 24 django/contrib/admin/checks.py | 547 | 582| 304 | 18946 | 192916 | 
| 59 | 24 django/db/models/options.py | 627 | 655| 214 | 19160 | 192916 | 
| 60 | 24 django/db/models/fields/related_descriptors.py | 1 | 79| 683 | 19843 | 192916 | 
| 61 | 24 django/db/models/sql/query.py | 1408 | 1419| 137 | 19980 | 192916 | 
| 62 | 24 django/contrib/admin/views/main.py | 289 | 330| 388 | 20368 | 192916 | 
| 63 | 24 django/db/models/query.py | 641 | 664| 199 | 20567 | 192916 | 
| 64 | 24 django/db/models/query.py | 1183 | 1217| 234 | 20801 | 192916 | 


## Patch

```diff
diff --git a/django/db/models/sql/compiler.py b/django/db/models/sql/compiler.py
--- a/django/db/models/sql/compiler.py
+++ b/django/db/models/sql/compiler.py
@@ -709,9 +709,9 @@ def find_ordering_name(self, name, opts, alias=None, default_order='ASC',
         field, targets, alias, joins, path, opts, transform_function = self._setup_joins(pieces, opts, alias)
 
         # If we get to this point and the field is a relation to another model,
-        # append the default ordering for that model unless the attribute name
-        # of the field is specified.
-        if field.is_relation and opts.ordering and getattr(field, 'attname', None) != name:
+        # append the default ordering for that model unless it is the pk
+        # shortcut or the attribute name of the field that is specified.
+        if field.is_relation and opts.ordering and getattr(field, 'attname', None) != name and name != 'pk':
             # Firstly, avoid infinite loops.
             already_seen = already_seen or set()
             join_tuple = tuple(getattr(self.query.alias_map[j], 'join_cols', None) for j in joins)

```

## Test Patch

```diff
diff --git a/tests/model_inheritance/models.py b/tests/model_inheritance/models.py
--- a/tests/model_inheritance/models.py
+++ b/tests/model_inheritance/models.py
@@ -181,6 +181,8 @@ class GrandParent(models.Model):
     place = models.ForeignKey(Place, models.CASCADE, null=True, related_name='+')
 
     class Meta:
+        # Ordering used by test_inherited_ordering_pk_desc.
+        ordering = ['-pk']
         unique_together = ('first_name', 'last_name')
 
 
diff --git a/tests/model_inheritance/tests.py b/tests/model_inheritance/tests.py
--- a/tests/model_inheritance/tests.py
+++ b/tests/model_inheritance/tests.py
@@ -7,7 +7,7 @@
 
 from .models import (
     Base, Chef, CommonInfo, GrandChild, GrandParent, ItalianRestaurant,
-    MixinModel, ParkingLot, Place, Post, Restaurant, Student, SubBase,
+    MixinModel, Parent, ParkingLot, Place, Post, Restaurant, Student, SubBase,
     Supplier, Title, Worker,
 )
 
@@ -204,6 +204,19 @@ class A(models.Model):
 
         self.assertEqual(A.attr.called, (A, 'attr'))
 
+    def test_inherited_ordering_pk_desc(self):
+        p1 = Parent.objects.create(first_name='Joe', email='joe@email.com')
+        p2 = Parent.objects.create(first_name='Jon', email='jon@email.com')
+        expected_order_by_sql = 'ORDER BY %s.%s DESC' % (
+            connection.ops.quote_name(Parent._meta.db_table),
+            connection.ops.quote_name(
+                Parent._meta.get_field('grandparent_ptr').column
+            ),
+        )
+        qs = Parent.objects.all()
+        self.assertSequenceEqual(qs, [p2, p1])
+        self.assertIn(expected_order_by_sql, str(qs.query))
+
 
 class ModelInheritanceDataTests(TestCase):
     @classmethod

```


## Code snippets

### 1 - django/db/models/base.py:

Start line: 968, End line: 981

```python
class Model(metaclass=ModelBase):

    def _get_next_or_previous_in_order(self, is_next):
        cachename = "__%s_order_cache" % is_next
        if not hasattr(self, cachename):
            op = 'gt' if is_next else 'lt'
            order = '_order' if is_next else '-_order'
            order_field = self._meta.order_with_respect_to
            filter_args = order_field.get_filter_kwargs_for_object(self)
            obj = self.__class__._default_manager.filter(**filter_args).filter(**{
                '_order__%s' % op: self.__class__._default_manager.values('_order').filter(**{
                    self._meta.pk.name: self.pk
                })
            }).order_by(order)[:1].get()
            setattr(self, cachename, obj)
        return getattr(self, cachename)
```
### 2 - django/db/models/base.py:

Start line: 1867, End line: 1918

```python
############################################
# HELPER FUNCTIONS (CURRIED MODEL METHODS) #
############################################

# ORDERING METHODS #########################

def method_set_order(self, ordered_obj, id_list, using=None):
    if using is None:
        using = DEFAULT_DB_ALIAS
    order_wrt = ordered_obj._meta.order_with_respect_to
    filter_args = order_wrt.get_forward_related_filter(self)
    ordered_obj.objects.db_manager(using).filter(**filter_args).bulk_update([
        ordered_obj(pk=pk, _order=order) for order, pk in enumerate(id_list)
    ], ['_order'])


def method_get_order(self, ordered_obj):
    order_wrt = ordered_obj._meta.order_with_respect_to
    filter_args = order_wrt.get_forward_related_filter(self)
    pk_name = ordered_obj._meta.pk.name
    return ordered_obj.objects.filter(**filter_args).values_list(pk_name, flat=True)


def make_foreign_order_accessors(model, related_model):
    setattr(
        related_model,
        'get_%s_order' % model.__name__.lower(),
        partialmethod(method_get_order, model)
    )
    setattr(
        related_model,
        'set_%s_order' % model.__name__.lower(),
        partialmethod(method_set_order, model)
    )

########
# MISC #
########


def model_unpickle(model_id):
    """Used to unpickle Model subclasses with deferred fields."""
    if isinstance(model_id, tuple):
        model = apps.get_model(*model_id)
    else:
        # Backwards compat - the model was cached directly in earlier versions.
        model = model_id
    return model.__new__(model)


model_unpickle.__safe_for_unpickle__ = True
```
### 3 - django/contrib/admin/options.py:

Start line: 205, End line: 216

```python
class BaseModelAdmin(metaclass=forms.MediaDefiningClass):

    def get_field_queryset(self, db, db_field, request):
        """
        If the ModelAdmin specifies ordering, the queryset should respect that
        ordering.  Otherwise don't specify the queryset, let the field decide
        (return None in that case).
        """
        related_admin = self.admin_site._registry.get(db_field.remote_field.model)
        if related_admin is not None:
            ordering = related_admin.get_ordering(request)
            if ordering is not None and ordering != ():
                return db_field.remote_field.model._default_manager.using(db).order_by(*ordering)
        return None
```
### 4 - django/db/models/base.py:

Start line: 1665, End line: 1763

```python
class Model(metaclass=ModelBase):

    @classmethod
    def _check_ordering(cls):
        """
        Check "ordering" option -- is it a list of strings and do all fields
        exist?
        """
        if cls._meta._ordering_clash:
            return [
                checks.Error(
                    "'ordering' and 'order_with_respect_to' cannot be used together.",
                    obj=cls,
                    id='models.E021',
                ),
            ]

        if cls._meta.order_with_respect_to or not cls._meta.ordering:
            return []

        if not isinstance(cls._meta.ordering, (list, tuple)):
            return [
                checks.Error(
                    "'ordering' must be a tuple or list (even if you want to order by only one field).",
                    obj=cls,
                    id='models.E014',
                )
            ]

        errors = []
        fields = cls._meta.ordering

        # Skip expressions and '?' fields.
        fields = (f for f in fields if isinstance(f, str) and f != '?')

        # Convert "-field" to "field".
        fields = ((f[1:] if f.startswith('-') else f) for f in fields)

        # Separate related fields and non-related fields.
        _fields = []
        related_fields = []
        for f in fields:
            if LOOKUP_SEP in f:
                related_fields.append(f)
            else:
                _fields.append(f)
        fields = _fields

        # Check related fields.
        for field in related_fields:
            _cls = cls
            fld = None
            for part in field.split(LOOKUP_SEP):
                try:
                    # pk is an alias that won't be found by opts.get_field.
                    if part == 'pk':
                        fld = _cls._meta.pk
                    else:
                        fld = _cls._meta.get_field(part)
                    if fld.is_relation:
                        _cls = fld.get_path_info()[-1].to_opts.model
                    else:
                        _cls = None
                except (FieldDoesNotExist, AttributeError):
                    if fld is None or fld.get_transform(part) is None:
                        errors.append(
                            checks.Error(
                                "'ordering' refers to the nonexistent field, "
                                "related field, or lookup '%s'." % field,
                                obj=cls,
                                id='models.E015',
                            )
                        )

        # Skip ordering on pk. This is always a valid order_by field
        # but is an alias and therefore won't be found by opts.get_field.
        fields = {f for f in fields if f != 'pk'}

        # Check for invalid or nonexistent fields in ordering.
        invalid_fields = []

        # Any field name that is not present in field_names does not exist.
        # Also, ordering by m2m fields is not allowed.
        opts = cls._meta
        valid_fields = set(chain.from_iterable(
            (f.name, f.attname) if not (f.auto_created and not f.concrete) else (f.field.related_query_name(),)
            for f in chain(opts.fields, opts.related_objects)
        ))

        invalid_fields.extend(fields - valid_fields)

        for invalid_field in invalid_fields:
            errors.append(
                checks.Error(
                    "'ordering' refers to the nonexistent field, related "
                    "field, or lookup '%s'." % invalid_field,
                    obj=cls,
                    id='models.E015',
                )
            )
        return errors
```
### 5 - django/db/models/base.py:

Start line: 952, End line: 966

```python
class Model(metaclass=ModelBase):

    def _get_next_or_previous_by_FIELD(self, field, is_next, **kwargs):
        if not self.pk:
            raise ValueError("get_next/get_previous cannot be used on unsaved objects.")
        op = 'gt' if is_next else 'lt'
        order = '' if is_next else '-'
        param = getattr(self, field.attname)
        q = Q(**{'%s__%s' % (field.name, op): param})
        q = q | Q(**{field.name: param, 'pk__%s' % op: self.pk})
        qs = self.__class__._default_manager.using(self._state.db).filter(**kwargs).filter(q).order_by(
            '%s%s' % (order, field.name), '%spk' % order
        )
        try:
            return qs[0]
        except IndexError:
            raise self.DoesNotExist("%s matching query does not exist." % self.__class__._meta.object_name)
```
### 6 - django/contrib/admin/views/main.py:

Start line: 332, End line: 392

```python
class ChangeList:

    def _get_deterministic_ordering(self, ordering):
        """
        Ensure a deterministic order across all database backends. Search for a
        single field or unique together set of fields providing a total
        ordering. If these are missing, augment the ordering with a descendant
        primary key.
        """
        ordering = list(ordering)
        ordering_fields = set()
        total_ordering_fields = {'pk'} | {
            field.attname for field in self.lookup_opts.fields
            if field.unique and not field.null
        }
        for part in ordering:
            # Search for single field providing a total ordering.
            field_name = None
            if isinstance(part, str):
                field_name = part.lstrip('-')
            elif isinstance(part, F):
                field_name = part.name
            elif isinstance(part, OrderBy) and isinstance(part.expression, F):
                field_name = part.expression.name
            if field_name:
                # Normalize attname references by using get_field().
                try:
                    field = self.lookup_opts.get_field(field_name)
                except FieldDoesNotExist:
                    # Could be "?" for random ordering or a related field
                    # lookup. Skip this part of introspection for now.
                    continue
                # Ordering by a related field name orders by the referenced
                # model's ordering. Skip this part of introspection for now.
                if field.remote_field and field_name == field.name:
                    continue
                if field.attname in total_ordering_fields:
                    break
                ordering_fields.add(field.attname)
        else:
            # No single total ordering field, try unique_together and total
            # unique constraints.
            constraint_field_names = (
                *self.lookup_opts.unique_together,
                *(
                    constraint.fields
                    for constraint in self.lookup_opts.total_unique_constraints
                ),
            )
            for field_names in constraint_field_names:
                # Normalize attname references by using get_field().
                fields = [self.lookup_opts.get_field(field_name) for field_name in field_names]
                # Composite unique constraints containing a nullable column
                # cannot ensure total ordering.
                if any(field.null for field in fields):
                    continue
                if ordering_fields.issuperset(field.attname for field in fields):
                    break
            else:
                # If no set of unique fields is present in the ordering, rely
                # on the primary key to provide total ordering.
                ordering.append('-pk')
        return ordering
```
### 7 - django/db/models/expressions.py:

Start line: 1115, End line: 1140

```python
class OrderBy(BaseExpression):

    def as_sql(self, compiler, connection, template=None, **extra_context):
        template = template or self.template
        if connection.features.supports_order_by_nulls_modifier:
            if self.nulls_last:
                template = '%s NULLS LAST' % template
            elif self.nulls_first:
                template = '%s NULLS FIRST' % template
        else:
            if self.nulls_last and not (
                self.descending and connection.features.order_by_nulls_first
            ):
                template = '%%(expression)s IS NULL, %s' % template
            elif self.nulls_first and not (
                not self.descending and connection.features.order_by_nulls_first
            ):
                template = '%%(expression)s IS NOT NULL, %s' % template
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
### 8 - django/db/models/sql/compiler.py:

Start line: 354, End line: 387

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
                    if col_alias:
                        raise DatabaseError('ORDER BY term does not match any column in the result set.')
                    # Add column used in ORDER BY clause without an alias to
                    # the selected columns.
                    self.query.add_select_col(src)
                    resolved.set_source_expressions([RawSQL('%d' % len(self.query.select), ())])
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
### 9 - django/db/models/options.py:

Start line: 579, End line: 605

```python
class Options:

    def get_base_chain(self, model):
        """
        Return a list of parent classes leading to `model` (ordered from
        closest to most distant ancestor). This has to handle the case where
        `model` is a grandparent or even more distant relation.
        """
        if not self.parents:
            return []
        if model in self.parents:
            return [model]
        for parent in self.parents:
            res = parent._meta.get_base_chain(model)
            if res:
                res.insert(0, parent)
                return res
        return []

    def get_parent_list(self):
        """
        Return all the ancestors of this model as a list ordered by MRO.
        Useful for determining if something is an ancestor, regardless of lineage.
        """
        result = OrderedSet(self.parents)
        for parent in self.parents:
            for ancestor in parent._meta.get_parent_list():
                result.add(ancestor)
        return list(result)
```
### 10 - django/db/models/expressions.py:

Start line: 1142, End line: 1172

```python
class OrderBy(BaseExpression):

    def as_oracle(self, compiler, connection):
        # Oracle doesn't allow ORDER BY EXISTS() unless it's wrapped in
        # a CASE WHEN.
        if isinstance(self.expression, Exists):
            copy = self.copy()
            copy.expression = Case(
                When(self.expression, then=True),
                default=False,
                output_field=fields.BooleanField(),
            )
            return copy.as_sql(compiler, connection)
        return self.as_sql(compiler, connection)

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
### 14 - django/db/models/sql/compiler.py:

Start line: 699, End line: 733

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
                if hasattr(item, 'resolve_expression') and not isinstance(item, OrderBy):
                    item = item.desc() if descending else item.asc()
                if isinstance(item, OrderBy):
                    results.append((item, False))
                    continue
                results.extend(self.find_ordering_name(item, opts, alias,
                                                       order, already_seen))
            return results
        targets, alias, _ = self.query.trim_joins(targets, joins, path)
        return [(OrderBy(transform_function(t, alias), descending=descending), False) for t in targets]
```
### 37 - django/db/models/sql/compiler.py:

Start line: 735, End line: 746

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
### 38 - django/db/models/sql/compiler.py:

Start line: 265, End line: 352

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
                if isinstance(field, Value):
                    # output_field must be resolved for constants.
                    field = Cast(field, field.output_field)
                if not isinstance(field, OrderBy):
                    field = field.asc()
                if not self.query.standard_ordering:
                    field = field.copy()
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
                # References to an expression which is masked out of the SELECT
                # clause.
                expr = self.query.annotations[col]
                if isinstance(expr, Value):
                    # output_field must be resolved for constants.
                    expr = Cast(expr, expr.output_field)
                order_by.append((OrderBy(expr, descending=descending), False))
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
