# django__django-13251

| **django/django** | `b6dfdaff33f19757b1cb9b3bf1d17f28b94859d4` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | - |
| **Avg pos** | - |
| **Min pos** | - |
| **Max pos** | - |
| **Top file pos** | 18 |
| **Missing snippets** | 5 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/db/models/query.py b/django/db/models/query.py
--- a/django/db/models/query.py
+++ b/django/db/models/query.py
@@ -204,7 +204,7 @@ def __init__(self, model=None, query=None, using=None, hints=None):
     def query(self):
         if self._deferred_filter:
             negate, args, kwargs = self._deferred_filter
-            self._filter_or_exclude_inplace(negate, *args, **kwargs)
+            self._filter_or_exclude_inplace(negate, args, kwargs)
             self._deferred_filter = None
         return self._query
 
@@ -939,7 +939,7 @@ def filter(self, *args, **kwargs):
         set.
         """
         self._not_support_combined_queries('filter')
-        return self._filter_or_exclude(False, *args, **kwargs)
+        return self._filter_or_exclude(False, args, kwargs)
 
     def exclude(self, *args, **kwargs):
         """
@@ -947,9 +947,9 @@ def exclude(self, *args, **kwargs):
         set.
         """
         self._not_support_combined_queries('exclude')
-        return self._filter_or_exclude(True, *args, **kwargs)
+        return self._filter_or_exclude(True, args, kwargs)
 
-    def _filter_or_exclude(self, negate, *args, **kwargs):
+    def _filter_or_exclude(self, negate, args, kwargs):
         if args or kwargs:
             assert not self.query.is_sliced, \
                 "Cannot filter a query once a slice has been taken."
@@ -959,10 +959,10 @@ def _filter_or_exclude(self, negate, *args, **kwargs):
             self._defer_next_filter = False
             clone._deferred_filter = negate, args, kwargs
         else:
-            clone._filter_or_exclude_inplace(negate, *args, **kwargs)
+            clone._filter_or_exclude_inplace(negate, args, kwargs)
         return clone
 
-    def _filter_or_exclude_inplace(self, negate, *args, **kwargs):
+    def _filter_or_exclude_inplace(self, negate, args, kwargs):
         if negate:
             self._query.add_q(~Q(*args, **kwargs))
         else:
@@ -983,7 +983,7 @@ def complex_filter(self, filter_obj):
             clone.query.add_q(filter_obj)
             return clone
         else:
-            return self._filter_or_exclude(False, **filter_obj)
+            return self._filter_or_exclude(False, args=(), kwargs=filter_obj)
 
     def _combinator_query(self, combinator, *other_qs, all=False):
         # Clone the query to inherit the select list and everything

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/db/models/query.py | 207 | 207 | - | 18 | -
| django/db/models/query.py | 942 | 942 | - | 18 | -
| django/db/models/query.py | 950 | 952 | - | 18 | -
| django/db/models/query.py | 962 | 965 | - | 18 | -
| django/db/models/query.py | 986 | 986 | - | 18 | -


## Problem Statement

```
Filtering on a field named `negate` raises a TypeError
Description
	
Filtering on a model with a field named negate raises a TypeError.
For example:
class Foo(models.Model):
	negate = models.BooleanField()
Foo.objects.filter(negate=True)
raises TypeError: _filter_or_exclude() got multiple values for argument 'negate'
negate is not documented as a reserved argument for .filter(). I'm currently using .filter(negate__exact=True) as a workaround.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 django/contrib/admin/filters.py | 229 | 244| 196 | 196 | 4093 | 
| 2 | 2 django/db/models/fields/__init__.py | 1928 | 1955| 234 | 430 | 21782 | 
| 3 | 2 django/contrib/admin/filters.py | 246 | 261| 154 | 584 | 21782 | 
| 4 | 2 django/contrib/admin/filters.py | 264 | 276| 149 | 733 | 21782 | 
| 5 | 2 django/db/models/fields/__init__.py | 936 | 961| 209 | 942 | 21782 | 
| 6 | 3 django/forms/models.py | 310 | 349| 387 | 1329 | 33556 | 
| 7 | 4 django/db/models/sql/query.py | 2316 | 2332| 177 | 1506 | 55902 | 
| 8 | 4 django/contrib/admin/filters.py | 432 | 444| 142 | 1648 | 55902 | 
| 9 | 4 django/db/models/fields/__init__.py | 963 | 979| 176 | 1824 | 55902 | 
| 10 | 5 django/db/models/fields/related.py | 343 | 360| 163 | 1987 | 69778 | 
| 11 | 5 django/db/models/fields/related.py | 1202 | 1233| 180 | 2167 | 69778 | 
| 12 | 5 django/contrib/admin/filters.py | 446 | 475| 226 | 2393 | 69778 | 
| 13 | 5 django/contrib/admin/filters.py | 371 | 395| 294 | 2687 | 69778 | 
| 14 | 5 django/contrib/admin/filters.py | 162 | 207| 427 | 3114 | 69778 | 
| 15 | 5 django/db/models/sql/query.py | 1285 | 1350| 772 | 3886 | 69778 | 
| 16 | 5 django/db/models/sql/query.py | 1106 | 1138| 338 | 4224 | 69778 | 
| 17 | 5 django/db/models/sql/query.py | 1419 | 1446| 283 | 4507 | 69778 | 
| 18 | 6 django/contrib/admin/options.py | 377 | 429| 504 | 5011 | 88347 | 
| 19 | 6 django/contrib/admin/options.py | 431 | 474| 350 | 5361 | 88347 | 
| 20 | 6 django/contrib/admin/filters.py | 278 | 302| 217 | 5578 | 88347 | 
| 21 | 7 django/contrib/admin/checks.py | 416 | 428| 137 | 5715 | 97484 | 
| 22 | 7 django/contrib/admin/filters.py | 305 | 368| 627 | 6342 | 97484 | 
| 23 | 8 django/db/models/expressions.py | 1143 | 1171| 266 | 6608 | 108261 | 
| 24 | 9 django/db/models/query_utils.py | 25 | 54| 185 | 6793 | 110967 | 
| 25 | 10 django/forms/fields.py | 730 | 762| 241 | 7034 | 120314 | 
| 26 | 11 django/db/models/base.py | 1394 | 1449| 491 | 7525 | 136895 | 
| 27 | 12 django/template/defaulttags.py | 824 | 839| 163 | 7688 | 148038 | 
| 28 | 12 django/db/models/fields/__init__.py | 783 | 842| 431 | 8119 | 148038 | 
| 29 | 12 django/db/models/query_utils.py | 57 | 108| 396 | 8515 | 148038 | 
| 30 | 13 django/contrib/contenttypes/fields.py | 110 | 158| 328 | 8843 | 153471 | 
| 31 | 13 django/db/models/sql/query.py | 1729 | 1800| 784 | 9627 | 153471 | 
| 32 | 13 django/contrib/admin/filters.py | 422 | 429| 107 | 9734 | 153471 | 
| 33 | 13 django/contrib/admin/checks.py | 809 | 859| 443 | 10177 | 153471 | 
| 34 | 13 django/contrib/admin/filters.py | 397 | 419| 211 | 10388 | 153471 | 
| 35 | 13 django/db/models/base.py | 954 | 968| 212 | 10600 | 153471 | 
| 36 | 13 django/db/models/sql/query.py | 1464 | 1549| 801 | 11401 | 153471 | 
| 37 | 14 django/shortcuts.py | 81 | 99| 200 | 11601 | 154569 | 
| 38 | 14 django/db/models/fields/related.py | 670 | 694| 218 | 11819 | 154569 | 
| 39 | 14 django/db/models/base.py | 1654 | 1702| 348 | 12167 | 154569 | 
| 40 | 15 django/db/models/__init__.py | 1 | 53| 619 | 12786 | 155188 | 
| 41 | 16 django/template/base.py | 668 | 703| 272 | 13058 | 163066 | 
| 42 | 16 django/db/models/fields/__init__.py | 1975 | 2011| 198 | 13256 | 163066 | 
| 43 | 16 django/db/models/fields/related.py | 127 | 154| 201 | 13457 | 163066 | 
| 44 | 16 django/db/models/fields/related.py | 841 | 862| 169 | 13626 | 163066 | 
| 45 | 17 django/db/migrations/questioner.py | 56 | 81| 220 | 13846 | 165139 | 
| 46 | 17 django/db/models/fields/related.py | 1235 | 1352| 963 | 14809 | 165139 | 
| 47 | 17 django/db/models/sql/query.py | 1077 | 1104| 285 | 15094 | 165139 | 
| 48 | 17 django/forms/fields.py | 703 | 727| 226 | 15320 | 165139 | 
| 49 | 17 django/contrib/admin/filters.py | 118 | 159| 365 | 15685 | 165139 | 
| 50 | 17 django/contrib/admin/filters.py | 209 | 226| 190 | 15875 | 165139 | 
| 51 | **18 django/db/models/query.py** | 1184 | 1203| 209 | 16084 | 182241 | 
| 52 | 18 django/db/migrations/questioner.py | 162 | 185| 246 | 16330 | 182241 | 
| 53 | 18 django/db/models/fields/related.py | 255 | 282| 269 | 16599 | 182241 | 
| 54 | 19 django/contrib/admin/utils.py | 287 | 305| 175 | 16774 | 186393 | 
| 55 | 20 django/db/models/fields/reverse_related.py | 19 | 115| 635 | 17409 | 188536 | 
| 56 | 20 django/db/models/fields/related.py | 320 | 341| 225 | 17634 | 188536 | 
| 57 | 20 django/db/models/query_utils.py | 312 | 352| 286 | 17920 | 188536 | 
| 58 | 20 django/db/models/base.py | 1147 | 1162| 138 | 18058 | 188536 | 
| 59 | 20 django/db/models/fields/related.py | 487 | 507| 138 | 18196 | 188536 | 
| 60 | 20 django/db/models/fields/related.py | 190 | 254| 673 | 18869 | 188536 | 
| 61 | 20 django/db/models/fields/__init__.py | 1675 | 1712| 226 | 19095 | 188536 | 
| 62 | 21 django/db/migrations/operations/fields.py | 146 | 189| 394 | 19489 | 191634 | 


### Hint

```
We should either document this limitation or change _filter_or_exclude and friends signature from (negate, *args, **kwargs) to (negate, args, kwargs). I think the second approach is favourable as there's not much benefits in using arguments unpacking in these private methods as long as the public methods filter and exclude preserve their signature.
Aaron, would you be interested in submitting a PR with the changes below plus a regression test in tests/query/tests.py? django/db/models/query.py diff --git a/django/db/models/query.py b/django/db/models/query.py index 07d6ffd4ca..d655ede8d9 100644 a b class QuerySet: 204204 def query(self): 205205 if self._deferred_filter: 206206 negate, args, kwargs = self._deferred_filter 207 self._filter_or_exclude_inplace(negate, *args, **kwargs) 207 self._filter_or_exclude_inplace(negate, args, kwargs) 208208 self._deferred_filter = None 209209 return self._query 210210 … … class QuerySet: 939939 set. 940940 """ 941941 self._not_support_combined_queries('filter') 942 return self._filter_or_exclude(False, *args, **kwargs) 942 return self._filter_or_exclude(False, args, kwargs) 943943 944944 def exclude(self, *args, **kwargs): 945945 """ … … class QuerySet: 947947 set. 948948 """ 949949 self._not_support_combined_queries('exclude') 950 return self._filter_or_exclude(True, *args, **kwargs) 950 return self._filter_or_exclude(True, args, kwargs) 951951 952 def _filter_or_exclude(self, negate, *args, **kwargs): 952 def _filter_or_exclude(self, negate, args, kwargs): 953953 if args or kwargs: 954954 assert not self.query.is_sliced, \ 955955 "Cannot filter a query once a slice has been taken." … … class QuerySet: 959959 self._defer_next_filter = False 960960 clone._deferred_filter = negate, args, kwargs 961961 else: 962 clone._filter_or_exclude_inplace(negate, *args, **kwargs) 962 clone._filter_or_exclude_inplace(negate, args, kwargs) 963963 return clone 964964 965 def _filter_or_exclude_inplace(self, negate, *args, **kwargs): 965 def _filter_or_exclude_inplace(self, negate, args, kwargs): 966966 if negate: 967967 self._query.add_q(~Q(*args, **kwargs)) 968968 else: … … class QuerySet: 983983 clone.query.add_q(filter_obj) 984984 return clone 985985 else: 986 return self._filter_or_exclude(False, **filter_obj) 986 return self._filter_or_exclude(False, args=(), kwargs=filter_obj) 987987 988988 def _combinator_query(self, combinator, *other_qs, all=False): 989989 # Clone the query to inherit the select list and everything
Replying to Simon Charette: Aaron, would you be interested in submitting a PR with the changes below plus a regression test in tests/query/tests.py? Sure! I can do that this evening. Thanks for the patch :)
```

## Patch

```diff
diff --git a/django/db/models/query.py b/django/db/models/query.py
--- a/django/db/models/query.py
+++ b/django/db/models/query.py
@@ -204,7 +204,7 @@ def __init__(self, model=None, query=None, using=None, hints=None):
     def query(self):
         if self._deferred_filter:
             negate, args, kwargs = self._deferred_filter
-            self._filter_or_exclude_inplace(negate, *args, **kwargs)
+            self._filter_or_exclude_inplace(negate, args, kwargs)
             self._deferred_filter = None
         return self._query
 
@@ -939,7 +939,7 @@ def filter(self, *args, **kwargs):
         set.
         """
         self._not_support_combined_queries('filter')
-        return self._filter_or_exclude(False, *args, **kwargs)
+        return self._filter_or_exclude(False, args, kwargs)
 
     def exclude(self, *args, **kwargs):
         """
@@ -947,9 +947,9 @@ def exclude(self, *args, **kwargs):
         set.
         """
         self._not_support_combined_queries('exclude')
-        return self._filter_or_exclude(True, *args, **kwargs)
+        return self._filter_or_exclude(True, args, kwargs)
 
-    def _filter_or_exclude(self, negate, *args, **kwargs):
+    def _filter_or_exclude(self, negate, args, kwargs):
         if args or kwargs:
             assert not self.query.is_sliced, \
                 "Cannot filter a query once a slice has been taken."
@@ -959,10 +959,10 @@ def _filter_or_exclude(self, negate, *args, **kwargs):
             self._defer_next_filter = False
             clone._deferred_filter = negate, args, kwargs
         else:
-            clone._filter_or_exclude_inplace(negate, *args, **kwargs)
+            clone._filter_or_exclude_inplace(negate, args, kwargs)
         return clone
 
-    def _filter_or_exclude_inplace(self, negate, *args, **kwargs):
+    def _filter_or_exclude_inplace(self, negate, args, kwargs):
         if negate:
             self._query.add_q(~Q(*args, **kwargs))
         else:
@@ -983,7 +983,7 @@ def complex_filter(self, filter_obj):
             clone.query.add_q(filter_obj)
             return clone
         else:
-            return self._filter_or_exclude(False, **filter_obj)
+            return self._filter_or_exclude(False, args=(), kwargs=filter_obj)
 
     def _combinator_query(self, combinator, *other_qs, all=False):
         # Clone the query to inherit the select list and everything

```

## Test Patch

```diff
diff --git a/tests/queries/models.py b/tests/queries/models.py
--- a/tests/queries/models.py
+++ b/tests/queries/models.py
@@ -42,6 +42,7 @@ class Note(models.Model):
     note = models.CharField(max_length=100)
     misc = models.CharField(max_length=10)
     tag = models.ForeignKey(Tag, models.SET_NULL, blank=True, null=True)
+    negate = models.BooleanField(default=True)
 
     class Meta:
         ordering = ['note']
diff --git a/tests/queries/tests.py b/tests/queries/tests.py
--- a/tests/queries/tests.py
+++ b/tests/queries/tests.py
@@ -47,7 +47,7 @@ def setUpTestData(cls):
 
         cls.n1 = Note.objects.create(note='n1', misc='foo', id=1)
         cls.n2 = Note.objects.create(note='n2', misc='bar', id=2)
-        cls.n3 = Note.objects.create(note='n3', misc='foo', id=3)
+        cls.n3 = Note.objects.create(note='n3', misc='foo', id=3, negate=False)
 
         ann1 = Annotation.objects.create(name='a1', tag=cls.t1)
         ann1.notes.add(cls.n1)
@@ -1216,6 +1216,13 @@ def test_field_with_filterable(self):
             [self.a3, self.a4],
         )
 
+    def test_negate_field(self):
+        self.assertSequenceEqual(
+            Note.objects.filter(negate=True),
+            [self.n1, self.n2],
+        )
+        self.assertSequenceEqual(Note.objects.exclude(negate=True), [self.n3])
+
 
 class Queries2Tests(TestCase):
     @classmethod

```


## Code snippets

### 1 - django/contrib/admin/filters.py:

Start line: 229, End line: 244

```python
FieldListFilter.register(lambda f: f.remote_field, RelatedFieldListFilter)


class BooleanFieldListFilter(FieldListFilter):
    def __init__(self, field, request, params, model, model_admin, field_path):
        self.lookup_kwarg = '%s__exact' % field_path
        self.lookup_kwarg2 = '%s__isnull' % field_path
        self.lookup_val = params.get(self.lookup_kwarg)
        self.lookup_val2 = params.get(self.lookup_kwarg2)
        super().__init__(field, request, params, model, model_admin, field_path)
        if (self.used_parameters and self.lookup_kwarg in self.used_parameters and
                self.used_parameters[self.lookup_kwarg] in ('1', '0')):
            self.used_parameters[self.lookup_kwarg] = bool(int(self.used_parameters[self.lookup_kwarg]))

    def expected_parameters(self):
        return [self.lookup_kwarg, self.lookup_kwarg2]
```
### 2 - django/db/models/fields/__init__.py:

Start line: 1928, End line: 1955

```python
class NullBooleanField(BooleanField):
    default_error_messages = {
        'invalid': _('“%(value)s” value must be either None, True or False.'),
        'invalid_nullable': _('“%(value)s” value must be either None, True or False.'),
    }
    description = _("Boolean (Either True, False or None)")
    system_check_deprecated_details = {
        'msg': (
            'NullBooleanField is deprecated. Support for it (except in '
            'historical migrations) will be removed in Django 4.0.'
        ),
        'hint': 'Use BooleanField(null=True) instead.',
        'id': 'fields.W903',
    }

    def __init__(self, *args, **kwargs):
        kwargs['null'] = True
        kwargs['blank'] = True
        super().__init__(*args, **kwargs)

    def deconstruct(self):
        name, path, args, kwargs = super().deconstruct()
        del kwargs['null']
        del kwargs['blank']
        return name, path, args, kwargs

    def get_internal_type(self):
        return "NullBooleanField"
```
### 3 - django/contrib/admin/filters.py:

Start line: 246, End line: 261

```python
class BooleanFieldListFilter(FieldListFilter):

    def choices(self, changelist):
        for lookup, title in (
                (None, _('All')),
                ('1', _('Yes')),
                ('0', _('No'))):
            yield {
                'selected': self.lookup_val == lookup and not self.lookup_val2,
                'query_string': changelist.get_query_string({self.lookup_kwarg: lookup}, [self.lookup_kwarg2]),
                'display': title,
            }
        if self.field.null:
            yield {
                'selected': self.lookup_val2 == 'True',
                'query_string': changelist.get_query_string({self.lookup_kwarg2: 'True'}, [self.lookup_kwarg]),
                'display': _('Unknown'),
            }
```
### 4 - django/contrib/admin/filters.py:

Start line: 264, End line: 276

```python
FieldListFilter.register(lambda f: isinstance(f, models.BooleanField), BooleanFieldListFilter)


class ChoicesFieldListFilter(FieldListFilter):
    def __init__(self, field, request, params, model, model_admin, field_path):
        self.lookup_kwarg = '%s__exact' % field_path
        self.lookup_kwarg_isnull = '%s__isnull' % field_path
        self.lookup_val = params.get(self.lookup_kwarg)
        self.lookup_val_isnull = params.get(self.lookup_kwarg_isnull)
        super().__init__(field, request, params, model, model_admin, field_path)

    def expected_parameters(self):
        return [self.lookup_kwarg, self.lookup_kwarg_isnull]
```
### 5 - django/db/models/fields/__init__.py:

Start line: 936, End line: 961

```python
class BooleanField(Field):
    empty_strings_allowed = False
    default_error_messages = {
        'invalid': _('“%(value)s” value must be either True or False.'),
        'invalid_nullable': _('“%(value)s” value must be either True, False, or None.'),
    }
    description = _("Boolean (Either True or False)")

    def get_internal_type(self):
        return "BooleanField"

    def to_python(self, value):
        if self.null and value in self.empty_values:
            return None
        if value in (True, False):
            # 1/0 are equal to True/False. bool() converts former to latter.
            return bool(value)
        if value in ('t', 'True', '1'):
            return True
        if value in ('f', 'False', '0'):
            return False
        raise exceptions.ValidationError(
            self.error_messages['invalid_nullable' if self.null else 'invalid'],
            code='invalid',
            params={'value': value},
        )
```
### 6 - django/forms/models.py:

Start line: 310, End line: 349

```python
class BaseModelForm(BaseForm):

    def _get_validation_exclusions(self):
        """
        For backwards-compatibility, exclude several types of fields from model
        validation. See tickets #12507, #12521, #12553.
        """
        exclude = []
        # Build up a list of fields that should be excluded from model field
        # validation and unique checks.
        for f in self.instance._meta.fields:
            field = f.name
            # Exclude fields that aren't on the form. The developer may be
            # adding these values to the model after form validation.
            if field not in self.fields:
                exclude.append(f.name)

            # Don't perform model validation on fields that were defined
            # manually on the form and excluded via the ModelForm's Meta
            # class. See #12901.
            elif self._meta.fields and field not in self._meta.fields:
                exclude.append(f.name)
            elif self._meta.exclude and field in self._meta.exclude:
                exclude.append(f.name)

            # Exclude fields that failed form validation. There's no need for
            # the model fields to validate them as well.
            elif field in self._errors:
                exclude.append(f.name)

            # Exclude empty fields that are not required by the form, if the
            # underlying model field is required. This keeps the model field
            # from raising a required error. Note: don't exclude the field from
            # validation if the model field allows blanks. If it does, the blank
            # value may be included in a unique check, so cannot be excluded
            # from validation.
            else:
                form_field = self.fields[field]
                field_value = self.cleaned_data.get(field)
                if not f.blank and not form_field.required and field_value in form_field.empty_values:
                    exclude.append(f.name)
        return exclude
```
### 7 - django/db/models/sql/query.py:

Start line: 2316, End line: 2332

```python
class Query(BaseExpression):

    def is_nullable(self, field):
        """
        Check if the given field should be treated as nullable.

        Some backends treat '' as null and Django treats such fields as
        nullable for those backends. In such situations field.null can be
        False even if we should treat the field as nullable.
        """
        # We need to use DEFAULT_DB_ALIAS here, as QuerySet does not have
        # (nor should it have) knowledge of which connection is going to be
        # used. The proper fix would be to defer all decisions where
        # is_nullable() is needed to the compiler stage, but that is not easy
        # to do currently.
        return (
            connections[DEFAULT_DB_ALIAS].features.interprets_empty_strings_as_nulls and
            field.empty_strings_allowed
        ) or field.null
```
### 8 - django/contrib/admin/filters.py:

Start line: 432, End line: 444

```python
class EmptyFieldListFilter(FieldListFilter):
    def __init__(self, field, request, params, model, model_admin, field_path):
        if not field.empty_strings_allowed and not field.null:
            raise ImproperlyConfigured(
                "The list filter '%s' cannot be used with field '%s' which "
                "doesn't allow empty strings and nulls." % (
                    self.__class__.__name__,
                    field.name,
                )
            )
        self.lookup_kwarg = '%s__isempty' % field_path
        self.lookup_val = params.get(self.lookup_kwarg)
        super().__init__(field, request, params, model, model_admin, field_path)
```
### 9 - django/db/models/fields/__init__.py:

Start line: 963, End line: 979

```python
class BooleanField(Field):

    def get_prep_value(self, value):
        value = super().get_prep_value(value)
        if value is None:
            return None
        return self.to_python(value)

    def formfield(self, **kwargs):
        if self.choices is not None:
            include_blank = not (self.has_default() or 'initial' in kwargs)
            defaults = {'choices': self.get_choices(include_blank=include_blank)}
        else:
            form_class = forms.NullBooleanField if self.null else forms.BooleanField
            # In HTML checkboxes, 'required' means "must be checked" which is
            # different from the choices case ("must select some value").
            # required=False allows unchecked checkboxes.
            defaults = {'form_class': form_class, 'required': False}
        return super().formfield(**{**defaults, **kwargs})
```
### 10 - django/db/models/fields/related.py:

Start line: 343, End line: 360

```python
class RelatedField(FieldCacheMixin, Field):

    def get_reverse_related_filter(self, obj):
        """
        Complement to get_forward_related_filter(). Return the keyword
        arguments that when passed to self.related_field.model.object.filter()
        select all instances of self.related_field.model related through
        this field to obj. obj is an instance of self.model.
        """
        base_filter = {
            rh_field.attname: getattr(obj, lh_field.attname)
            for lh_field, rh_field in self.related_fields
        }
        descriptor_filter = self.get_extra_descriptor_filter(obj)
        base_q = Q(**base_filter)
        if isinstance(descriptor_filter, dict):
            return base_q & Q(**descriptor_filter)
        elif descriptor_filter:
            return base_q & descriptor_filter
        return base_q
```
### 51 - django/db/models/query.py:

Start line: 1184, End line: 1203

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
