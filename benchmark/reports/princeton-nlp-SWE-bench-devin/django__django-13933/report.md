# django__django-13933

| **django/django** | `42e8cf47c7ee2db238bf91197ea398126c546741` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 813 |
| **Any found context length** | 813 |
| **Avg pos** | 4.0 |
| **Min pos** | 4 |
| **Max pos** | 4 |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/forms/models.py b/django/forms/models.py
--- a/django/forms/models.py
+++ b/django/forms/models.py
@@ -1284,7 +1284,11 @@ def to_python(self, value):
                 value = getattr(value, key)
             value = self.queryset.get(**{key: value})
         except (ValueError, TypeError, self.queryset.model.DoesNotExist):
-            raise ValidationError(self.error_messages['invalid_choice'], code='invalid_choice')
+            raise ValidationError(
+                self.error_messages['invalid_choice'],
+                code='invalid_choice',
+                params={'value': value},
+            )
         return value
 
     def validate(self, value):

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/forms/models.py | 1287 | 1287 | 4 | 1 | 813


## Problem Statement

```
ModelChoiceField does not provide value of invalid choice when raising ValidationError
Description
	 
		(last modified by Aaron Wiegel)
	 
Compared with ChoiceField and others, ModelChoiceField does not show the value of the invalid choice when raising a validation error. Passing in parameters with the invalid value and modifying the default error message for the code invalid_choice should fix this.
From source code:
class ModelMultipleChoiceField(ModelChoiceField):
	"""A MultipleChoiceField whose choices are a model QuerySet."""
	widget = SelectMultiple
	hidden_widget = MultipleHiddenInput
	default_error_messages = {
		'invalid_list': _('Enter a list of values.'),
		'invalid_choice': _('Select a valid choice. %(value)s is not one of the'
							' available choices.'),
		'invalid_pk_value': _('“%(pk)s” is not a valid value.')
	}
	...
class ModelChoiceField(ChoiceField):
	"""A ChoiceField whose choices are a model QuerySet."""
	# This class is a subclass of ChoiceField for purity, but it doesn't
	# actually use any of ChoiceField's implementation.
	default_error_messages = {
		'invalid_choice': _('Select a valid choice. That choice is not one of'
							' the available choices.'),
	}
	...

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 django/forms/models.py** | 1337 | 1372| 288 | 288 | 11762 | 
| 2 | **1 django/forms/models.py** | 1301 | 1318| 152 | 440 | 11762 | 
| 3 | **1 django/forms/models.py** | 1320 | 1335| 131 | 571 | 11762 | 
| **-> 4 <-** | **1 django/forms/models.py** | 1268 | 1298| 242 | 813 | 11762 | 
| 5 | **1 django/forms/models.py** | 1186 | 1251| 543 | 1356 | 11762 | 
| 6 | 2 django/forms/fields.py | 853 | 892| 298 | 1654 | 21106 | 
| 7 | **2 django/forms/models.py** | 1374 | 1401| 209 | 1863 | 21106 | 
| 8 | 3 django/db/models/fields/__init__.py | 632 | 661| 234 | 2097 | 39524 | 
| 9 | 3 django/forms/fields.py | 764 | 823| 416 | 2513 | 39524 | 
| 10 | **3 django/forms/models.py** | 1141 | 1183| 310 | 2823 | 39524 | 
| 11 | **3 django/forms/models.py** | 1253 | 1266| 176 | 2999 | 39524 | 
| 12 | 3 django/forms/fields.py | 895 | 928| 235 | 3234 | 39524 | 
| 13 | 3 django/db/models/fields/__init__.py | 243 | 305| 448 | 3682 | 39524 | 
| 14 | 3 django/forms/fields.py | 826 | 850| 177 | 3859 | 39524 | 
| 15 | 4 django/db/models/base.py | 1174 | 1202| 213 | 4072 | 56582 | 
| 16 | 5 django/forms/widgets.py | 672 | 703| 261 | 4333 | 64680 | 
| 17 | 6 django/core/exceptions.py | 107 | 218| 752 | 5085 | 65869 | 
| 18 | **6 django/forms/models.py** | 759 | 780| 194 | 5279 | 65869 | 
| 19 | 6 django/db/models/fields/__init__.py | 863 | 887| 244 | 5523 | 65869 | 
| 20 | 6 django/db/models/base.py | 1157 | 1172| 138 | 5661 | 65869 | 
| 21 | 6 django/db/models/base.py | 1240 | 1263| 172 | 5833 | 65869 | 
| 22 | 6 django/forms/widgets.py | 744 | 763| 144 | 5977 | 65869 | 
| 23 | 7 django/contrib/admin/filters.py | 280 | 304| 217 | 6194 | 69992 | 
| 24 | 8 django/contrib/admin/widgets.py | 420 | 445| 267 | 6461 | 73835 | 
| 25 | 9 django/db/models/fields/related.py | 1202 | 1233| 180 | 6641 | 87658 | 
| 26 | 9 django/contrib/admin/filters.py | 399 | 421| 211 | 6852 | 87658 | 
| 27 | 9 django/forms/fields.py | 1000 | 1052| 450 | 7302 | 87658 | 
| 28 | 9 django/db/models/fields/__init__.py | 2397 | 2447| 339 | 7641 | 87658 | 
| 29 | 9 django/forms/widgets.py | 766 | 789| 203 | 7844 | 87658 | 
| 30 | 9 django/db/models/fields/__init__.py | 1 | 81| 633 | 8477 | 87658 | 
| 31 | 10 django/contrib/admin/options.py | 189 | 205| 171 | 8648 | 106197 | 
| 32 | 10 django/db/models/base.py | 1717 | 1765| 348 | 8996 | 106197 | 
| 33 | 11 django/db/models/fields/reverse_related.py | 141 | 158| 161 | 9157 | 108519 | 
| 34 | 12 django/forms/boundfield.py | 36 | 51| 149 | 9306 | 110675 | 
| 35 | 13 django/db/migrations/questioner.py | 143 | 160| 183 | 9489 | 112748 | 
| 36 | 13 django/forms/widgets.py | 551 | 585| 277 | 9766 | 112748 | 
| 37 | 13 django/db/models/fields/__init__.py | 207 | 241| 234 | 10000 | 112748 | 
| 38 | 13 django/db/models/fields/related.py | 1235 | 1352| 963 | 10963 | 112748 | 
| 39 | 13 django/db/models/fields/__init__.py | 1441 | 1464| 171 | 11134 | 112748 | 
| 40 | 13 django/forms/widgets.py | 621 | 638| 179 | 11313 | 112748 | 
| 41 | 14 django/db/models/enums.py | 37 | 59| 183 | 11496 | 113341 | 
| 42 | 14 django/contrib/admin/filters.py | 209 | 226| 190 | 11686 | 113341 | 
| 43 | 14 django/db/models/base.py | 995 | 1023| 230 | 11916 | 113341 | 
| 44 | 14 django/db/models/fields/__init__.py | 1514 | 1573| 403 | 12319 | 113341 | 
| 45 | 14 django/db/models/base.py | 1204 | 1238| 230 | 12549 | 113341 | 
| 46 | 15 django/db/models/fields/mixins.py | 31 | 57| 173 | 12722 | 113684 | 
| 47 | 15 django/db/migrations/questioner.py | 162 | 185| 246 | 12968 | 113684 | 
| 48 | **15 django/forms/models.py** | 686 | 757| 732 | 13700 | 113684 | 
| 49 | **15 django/forms/models.py** | 1 | 27| 218 | 13918 | 113684 | 
| 50 | 16 django/db/models/options.py | 252 | 287| 341 | 14259 | 121051 | 
| 51 | 16 django/forms/fields.py | 729 | 761| 241 | 14500 | 121051 | 
| 52 | 16 django/db/migrations/questioner.py | 56 | 81| 220 | 14720 | 121051 | 
| 53 | 16 django/db/models/fields/related.py | 913 | 933| 178 | 14898 | 121051 | 
| 54 | 16 django/db/models/fields/related.py | 1354 | 1426| 616 | 15514 | 121051 | 
| 55 | 16 django/db/models/base.py | 1083 | 1126| 404 | 15918 | 121051 | 
| 56 | 16 django/db/models/fields/__init__.py | 337 | 364| 203 | 16121 | 121051 | 
| 57 | 16 django/db/models/fields/__init__.py | 1764 | 1787| 146 | 16267 | 121051 | 
| 58 | 16 django/forms/fields.py | 1178 | 1215| 199 | 16466 | 121051 | 
| 59 | 16 django/db/models/fields/__init__.py | 982 | 998| 176 | 16642 | 121051 | 
| 60 | **16 django/forms/models.py** | 389 | 417| 240 | 16882 | 121051 | 
| 61 | 16 django/contrib/admin/options.py | 1 | 97| 762 | 17644 | 121051 | 
| 62 | **16 django/forms/models.py** | 316 | 355| 387 | 18031 | 121051 | 
| 63 | 16 django/db/models/enums.py | 62 | 83| 118 | 18149 | 121051 | 
| 64 | 17 django/db/models/query.py | 842 | 871| 248 | 18397 | 138360 | 
| 65 | **17 django/forms/models.py** | 357 | 387| 233 | 18630 | 138360 | 
| 66 | 18 django/db/models/sql/query.py | 2177 | 2224| 394 | 19024 | 160751 | 
| 67 | 18 django/contrib/admin/filters.py | 266 | 278| 149 | 19173 | 160751 | 
| 68 | 18 django/db/models/base.py | 1601 | 1626| 183 | 19356 | 160751 | 
| 69 | 18 django/contrib/admin/options.py | 2085 | 2137| 451 | 19807 | 160751 | 
| 70 | 18 django/db/models/fields/__init__.py | 1724 | 1761| 226 | 20033 | 160751 | 
| 71 | 18 django/db/models/fields/__init__.py | 1818 | 1848| 186 | 20219 | 160751 | 
| 72 | 19 django/db/models/fields/json.py | 67 | 121| 362 | 20581 | 164949 | 
| 73 | 19 django/db/models/query.py | 784 | 800| 157 | 20738 | 164949 | 
| 74 | 19 django/db/models/base.py | 1960 | 2091| 976 | 21714 | 164949 | 
| 75 | 19 django/db/models/fields/__init__.py | 663 | 687| 206 | 21920 | 164949 | 
| 76 | 19 django/db/models/base.py | 1265 | 1296| 267 | 22187 | 164949 | 
| 77 | **19 django/forms/models.py** | 201 | 211| 131 | 22318 | 164949 | 
| 78 | 20 django/contrib/admin/utils.py | 411 | 440| 203 | 22521 | 169111 | 
| 79 | 21 django/contrib/postgres/fields/ranges.py | 43 | 91| 362 | 22883 | 171203 | 
| 80 | 21 django/db/models/base.py | 945 | 962| 177 | 23060 | 171203 | 
| 81 | 21 django/forms/widgets.py | 640 | 669| 238 | 23298 | 171203 | 
| 82 | 22 django/contrib/postgres/forms/ranges.py | 31 | 78| 325 | 23623 | 171880 | 
| 83 | 23 django/db/migrations/state.py | 346 | 393| 428 | 24051 | 176983 | 
| 84 | 24 django/db/models/constraints.py | 79 | 161| 729 | 24780 | 178598 | 
| 85 | 24 django/db/models/fields/__init__.py | 1094 | 1107| 104 | 24884 | 178598 | 
| 86 | 25 django/contrib/gis/forms/fields.py | 62 | 85| 204 | 25088 | 179517 | 
| 87 | 25 django/forms/fields.py | 130 | 173| 290 | 25378 | 179517 | 
| 88 | 25 django/forms/widgets.py | 706 | 741| 237 | 25615 | 179517 | 
| 89 | 25 django/db/models/fields/__init__.py | 1984 | 2011| 220 | 25835 | 179517 | 
| 90 | **25 django/forms/models.py** | 419 | 449| 243 | 26078 | 179517 | 
| 91 | 25 django/db/models/fields/__init__.py | 1789 | 1816| 215 | 26293 | 179517 | 
| 92 | 25 django/db/models/base.py | 1298 | 1325| 228 | 26521 | 179517 | 
| 93 | 25 django/db/models/sql/query.py | 1132 | 1164| 338 | 26859 | 179517 | 
| 94 | 25 django/db/models/sql/query.py | 1490 | 1575| 801 | 27660 | 179517 | 
| 95 | 26 django/core/management/base.py | 120 | 155| 241 | 27901 | 184154 | 
| 96 | 26 django/forms/fields.py | 1218 | 1271| 355 | 28256 | 184154 | 
| 97 | 26 django/db/models/fields/related.py | 1661 | 1695| 266 | 28522 | 184154 | 
| 98 | 27 django/contrib/postgres/fields/array.py | 18 | 51| 288 | 28810 | 186337 | 
| 99 | 27 django/db/models/fields/__init__.py | 1110 | 1139| 218 | 29028 | 186337 | 
| 100 | 28 django/db/backends/mysql/schema.py | 51 | 87| 349 | 29377 | 187859 | 
| 101 | 28 django/db/backends/mysql/schema.py | 89 | 99| 138 | 29515 | 187859 | 
| 102 | 28 django/db/models/fields/__init__.py | 1076 | 1091| 173 | 29688 | 187859 | 
| 103 | 28 django/contrib/admin/options.py | 1540 | 1626| 760 | 30448 | 187859 | 
| 104 | 29 django/contrib/postgres/forms/array.py | 62 | 102| 248 | 30696 | 189453 | 
| 105 | 29 django/db/migrations/questioner.py | 227 | 240| 123 | 30819 | 189453 | 
| 106 | 30 django/core/validators.py | 328 | 358| 227 | 31046 | 193998 | 
| 107 | 30 django/db/models/query.py | 801 | 840| 322 | 31368 | 193998 | 
| 108 | 30 django/forms/fields.py | 325 | 362| 330 | 31698 | 193998 | 
| 109 | 30 django/db/models/sql/query.py | 2334 | 2350| 177 | 31875 | 193998 | 
| 110 | 30 django/db/models/fields/__init__.py | 366 | 392| 199 | 32074 | 193998 | 
| 111 | 30 django/forms/fields.py | 1083 | 1124| 353 | 32427 | 193998 | 
| 112 | 30 django/db/models/fields/related.py | 1428 | 1469| 418 | 32845 | 193998 | 
| 113 | 31 django/db/models/deletion.py | 1 | 76| 566 | 33411 | 197826 | 
| 114 | 31 django/core/validators.py | 361 | 406| 308 | 33719 | 197826 | 
| 115 | 31 django/db/models/sql/query.py | 1914 | 1955| 355 | 34074 | 197826 | 
| 116 | 32 django/contrib/postgres/validators.py | 1 | 21| 181 | 34255 | 198377 | 
| 117 | 32 django/db/models/fields/__init__.py | 1466 | 1488| 121 | 34376 | 198377 | 
| 118 | 32 django/forms/fields.py | 47 | 128| 773 | 35149 | 198377 | 
| 119 | 32 django/db/models/fields/__init__.py | 2450 | 2499| 311 | 35460 | 198377 | 
| 120 | 32 django/db/models/fields/json.py | 42 | 65| 155 | 35615 | 198377 | 
| 121 | 32 django/forms/fields.py | 244 | 261| 164 | 35779 | 198377 | 
| 122 | 32 django/db/models/fields/__init__.py | 1001 | 1015| 122 | 35901 | 198377 | 
| 123 | 33 django/db/backends/mysql/validation.py | 1 | 31| 239 | 36140 | 198897 | 
| 124 | 33 django/db/models/query.py | 1098 | 1139| 323 | 36463 | 198897 | 
| 125 | 33 django/contrib/admin/filters.py | 448 | 477| 226 | 36689 | 198897 | 
| 126 | 33 django/db/models/fields/related.py | 984 | 995| 128 | 36817 | 198897 | 
| 127 | 33 django/db/models/base.py | 1516 | 1538| 171 | 36988 | 198897 | 
| 128 | 33 django/contrib/postgres/fields/array.py | 160 | 177| 146 | 37134 | 198897 | 
| 129 | 33 django/core/validators.py | 438 | 484| 440 | 37574 | 198897 | 
| 130 | 33 django/contrib/admin/options.py | 207 | 218| 135 | 37709 | 198897 | 
| 131 | 33 django/db/models/fields/__init__.py | 1038 | 1074| 248 | 37957 | 198897 | 
| 132 | **33 django/forms/models.py** | 961 | 994| 367 | 38324 | 198897 | 
| 133 | 33 django/contrib/admin/filters.py | 246 | 263| 184 | 38508 | 198897 | 
| 134 | **33 django/forms/models.py** | 96 | 109| 157 | 38665 | 198897 | 
| 135 | **33 django/forms/models.py** | 214 | 283| 616 | 39281 | 198897 | 
| 136 | 33 django/forms/boundfield.py | 53 | 78| 180 | 39461 | 198897 | 


### Hint

```
This message has been the same literally forever b2b6fc8e3c78671c8b6af2709358c3213c84d119. ​Given that ChoiceField passes the value when raising the error, if you set ​error_messages you should be able to get the result you want.
Replying to Carlton Gibson: This message has been the same literally forever b2b6fc8e3c78671c8b6af2709358c3213c84d119. ​Given that ChoiceField passes the value when raising the error, if you set ​error_messages you should be able to get the result you want. That is ChoiceField. ModelChoiceField ​does not pass the value to the validation error. So, when the invalid value error is raised, you can't display the offending value even if you override the defaults.
OK, if you want to look at submitting a PR we can see if any objections come up in review. Thanks.
PR: ​https://github.com/django/django/pull/13933
```

## Patch

```diff
diff --git a/django/forms/models.py b/django/forms/models.py
--- a/django/forms/models.py
+++ b/django/forms/models.py
@@ -1284,7 +1284,11 @@ def to_python(self, value):
                 value = getattr(value, key)
             value = self.queryset.get(**{key: value})
         except (ValueError, TypeError, self.queryset.model.DoesNotExist):
-            raise ValidationError(self.error_messages['invalid_choice'], code='invalid_choice')
+            raise ValidationError(
+                self.error_messages['invalid_choice'],
+                code='invalid_choice',
+                params={'value': value},
+            )
         return value
 
     def validate(self, value):

```

## Test Patch

```diff
diff --git a/tests/forms_tests/tests/test_error_messages.py b/tests/forms_tests/tests/test_error_messages.py
--- a/tests/forms_tests/tests/test_error_messages.py
+++ b/tests/forms_tests/tests/test_error_messages.py
@@ -308,3 +308,16 @@ def test_modelchoicefield(self):
         self.assertFormErrors(['REQUIRED'], f.clean, '')
         self.assertFormErrors(['NOT A LIST OF VALUES'], f.clean, '3')
         self.assertFormErrors(['4 IS INVALID CHOICE'], f.clean, ['4'])
+
+    def test_modelchoicefield_value_placeholder(self):
+        f = ModelChoiceField(
+            queryset=ChoiceModel.objects.all(),
+            error_messages={
+                'invalid_choice': '"%(value)s" is not one of the available choices.',
+            },
+        )
+        self.assertFormErrors(
+            ['"invalid" is not one of the available choices.'],
+            f.clean,
+            'invalid',
+        )

```


## Code snippets

### 1 - django/forms/models.py:

Start line: 1337, End line: 1372

```python
class ModelMultipleChoiceField(ModelChoiceField):

    def _check_values(self, value):
        """
        Given a list of possible PK values, return a QuerySet of the
        corresponding objects. Raise a ValidationError if a given value is
        invalid (not a valid PK, not in the queryset, etc.)
        """
        key = self.to_field_name or 'pk'
        # deduplicate given values to avoid creating many querysets or
        # requiring the database backend deduplicate efficiently.
        try:
            value = frozenset(value)
        except TypeError:
            # list of lists isn't hashable, for example
            raise ValidationError(
                self.error_messages['invalid_list'],
                code='invalid_list',
            )
        for pk in value:
            try:
                self.queryset.filter(**{key: pk})
            except (ValueError, TypeError):
                raise ValidationError(
                    self.error_messages['invalid_pk_value'],
                    code='invalid_pk_value',
                    params={'pk': pk},
                )
        qs = self.queryset.filter(**{'%s__in' % key: value})
        pks = {str(getattr(o, key)) for o in qs}
        for val in value:
            if str(val) not in pks:
                raise ValidationError(
                    self.error_messages['invalid_choice'],
                    code='invalid_choice',
                    params={'value': val},
                )
        return qs
```
### 2 - django/forms/models.py:

Start line: 1301, End line: 1318

```python
class ModelMultipleChoiceField(ModelChoiceField):
    """A MultipleChoiceField whose choices are a model QuerySet."""
    widget = SelectMultiple
    hidden_widget = MultipleHiddenInput
    default_error_messages = {
        'invalid_list': _('Enter a list of values.'),
        'invalid_choice': _('Select a valid choice. %(value)s is not one of the'
                            ' available choices.'),
        'invalid_pk_value': _('“%(pk)s” is not a valid value.')
    }

    def __init__(self, queryset, **kwargs):
        super().__init__(queryset, empty_label=None, **kwargs)

    def to_python(self, value):
        if not value:
            return []
        return list(self._check_values(value))
```
### 3 - django/forms/models.py:

Start line: 1320, End line: 1335

```python
class ModelMultipleChoiceField(ModelChoiceField):

    def clean(self, value):
        value = self.prepare_value(value)
        if self.required and not value:
            raise ValidationError(self.error_messages['required'], code='required')
        elif not self.required and not value:
            return self.queryset.none()
        if not isinstance(value, (list, tuple)):
            raise ValidationError(
                self.error_messages['invalid_list'],
                code='invalid_list',
            )
        qs = self._check_values(value)
        # Since this overrides the inherited ModelChoiceField.clean
        # we run custom validators here
        self.run_validators(value)
        return qs
```
### 4 - django/forms/models.py:

Start line: 1268, End line: 1298

```python
class ModelChoiceField(ChoiceField):

    choices = property(_get_choices, ChoiceField._set_choices)

    def prepare_value(self, value):
        if hasattr(value, '_meta'):
            if self.to_field_name:
                return value.serializable_value(self.to_field_name)
            else:
                return value.pk
        return super().prepare_value(value)

    def to_python(self, value):
        if value in self.empty_values:
            return None
        try:
            key = self.to_field_name or 'pk'
            if isinstance(value, self.queryset.model):
                value = getattr(value, key)
            value = self.queryset.get(**{key: value})
        except (ValueError, TypeError, self.queryset.model.DoesNotExist):
            raise ValidationError(self.error_messages['invalid_choice'], code='invalid_choice')
        return value

    def validate(self, value):
        return Field.validate(self, value)

    def has_changed(self, initial, data):
        if self.disabled:
            return False
        initial_value = initial if initial is not None else ''
        data_value = data if data is not None else ''
        return str(self.prepare_value(initial_value)) != str(data_value)
```
### 5 - django/forms/models.py:

Start line: 1186, End line: 1251

```python
class ModelChoiceField(ChoiceField):
    """A ChoiceField whose choices are a model QuerySet."""
    # This class is a subclass of ChoiceField for purity, but it doesn't
    # actually use any of ChoiceField's implementation.
    default_error_messages = {
        'invalid_choice': _('Select a valid choice. That choice is not one of'
                            ' the available choices.'),
    }
    iterator = ModelChoiceIterator

    def __init__(self, queryset, *, empty_label="---------",
                 required=True, widget=None, label=None, initial=None,
                 help_text='', to_field_name=None, limit_choices_to=None,
                 blank=False, **kwargs):
        # Call Field instead of ChoiceField __init__() because we don't need
        # ChoiceField.__init__().
        Field.__init__(
            self, required=required, widget=widget, label=label,
            initial=initial, help_text=help_text, **kwargs
        )
        if (
            (required and initial is not None) or
            (isinstance(self.widget, RadioSelect) and not blank)
        ):
            self.empty_label = None
        else:
            self.empty_label = empty_label
        self.queryset = queryset
        self.limit_choices_to = limit_choices_to   # limit the queryset later.
        self.to_field_name = to_field_name

    def get_limit_choices_to(self):
        """
        Return ``limit_choices_to`` for this form field.

        If it is a callable, invoke it and return the result.
        """
        if callable(self.limit_choices_to):
            return self.limit_choices_to()
        return self.limit_choices_to

    def __deepcopy__(self, memo):
        result = super(ChoiceField, self).__deepcopy__(memo)
        # Need to force a new ModelChoiceIterator to be created, bug #11183
        if self.queryset is not None:
            result.queryset = self.queryset.all()
        return result

    def _get_queryset(self):
        return self._queryset

    def _set_queryset(self, queryset):
        self._queryset = None if queryset is None else queryset.all()
        self.widget.choices = self.choices

    queryset = property(_get_queryset, _set_queryset)

    # this method will be used to create object labels by the QuerySetIterator.
    # Override it to customize the label.
    def label_from_instance(self, obj):
        """
        Convert objects into strings and generate the labels for the choices
        presented by this object. Subclasses can override this method to
        customize the display of the choices.
        """
        return str(obj)
```
### 6 - django/forms/fields.py:

Start line: 853, End line: 892

```python
class MultipleChoiceField(ChoiceField):
    hidden_widget = MultipleHiddenInput
    widget = SelectMultiple
    default_error_messages = {
        'invalid_choice': _('Select a valid choice. %(value)s is not one of the available choices.'),
        'invalid_list': _('Enter a list of values.'),
    }

    def to_python(self, value):
        if not value:
            return []
        elif not isinstance(value, (list, tuple)):
            raise ValidationError(self.error_messages['invalid_list'], code='invalid_list')
        return [str(val) for val in value]

    def validate(self, value):
        """Validate that the input is a list or tuple."""
        if self.required and not value:
            raise ValidationError(self.error_messages['required'], code='required')
        # Validate that each value in the value list is in self.choices.
        for val in value:
            if not self.valid_value(val):
                raise ValidationError(
                    self.error_messages['invalid_choice'],
                    code='invalid_choice',
                    params={'value': val},
                )

    def has_changed(self, initial, data):
        if self.disabled:
            return False
        if initial is None:
            initial = []
        if data is None:
            data = []
        if len(initial) != len(data):
            return True
        initial_set = {str(value) for value in initial}
        data_set = {str(value) for value in data}
        return data_set != initial_set
```
### 7 - django/forms/models.py:

Start line: 1374, End line: 1401

```python
class ModelMultipleChoiceField(ModelChoiceField):

    def prepare_value(self, value):
        if (hasattr(value, '__iter__') and
                not isinstance(value, str) and
                not hasattr(value, '_meta')):
            prepare_value = super().prepare_value
            return [prepare_value(v) for v in value]
        return super().prepare_value(value)

    def has_changed(self, initial, data):
        if self.disabled:
            return False
        if initial is None:
            initial = []
        if data is None:
            data = []
        if len(initial) != len(data):
            return True
        initial_set = {str(value) for value in self.prepare_value(initial)}
        data_set = {str(value) for value in data}
        return data_set != initial_set


def modelform_defines_fields(form_class):
    return hasattr(form_class, '_meta') and (
        form_class._meta.fields is not None or
        form_class._meta.exclude is not None
    )
```
### 8 - django/db/models/fields/__init__.py:

Start line: 632, End line: 661

```python
@total_ordering
class Field(RegisterLookupMixin):

    def validate(self, value, model_instance):
        """
        Validate value and raise ValidationError if necessary. Subclasses
        should override this to provide validation logic.
        """
        if not self.editable:
            # Skip validation for non-editable fields.
            return

        if self.choices is not None and value not in self.empty_values:
            for option_key, option_value in self.choices:
                if isinstance(option_value, (list, tuple)):
                    # This is an optgroup, so look inside the group for
                    # options.
                    for optgroup_key, optgroup_value in option_value:
                        if value == optgroup_key:
                            return
                elif value == option_key:
                    return
            raise exceptions.ValidationError(
                self.error_messages['invalid_choice'],
                code='invalid_choice',
                params={'value': value},
            )

        if value is None and not self.null:
            raise exceptions.ValidationError(self.error_messages['null'], code='null')

        if not self.blank and value in self.empty_values:
            raise exceptions.ValidationError(self.error_messages['blank'], code='blank')
```
### 9 - django/forms/fields.py:

Start line: 764, End line: 823

```python
class ChoiceField(Field):
    widget = Select
    default_error_messages = {
        'invalid_choice': _('Select a valid choice. %(value)s is not one of the available choices.'),
    }

    def __init__(self, *, choices=(), **kwargs):
        super().__init__(**kwargs)
        self.choices = choices

    def __deepcopy__(self, memo):
        result = super().__deepcopy__(memo)
        result._choices = copy.deepcopy(self._choices, memo)
        return result

    def _get_choices(self):
        return self._choices

    def _set_choices(self, value):
        # Setting choices also sets the choices on the widget.
        # choices can be any iterable, but we call list() on it because
        # it will be consumed more than once.
        if callable(value):
            value = CallableChoiceIterator(value)
        else:
            value = list(value)

        self._choices = self.widget.choices = value

    choices = property(_get_choices, _set_choices)

    def to_python(self, value):
        """Return a string."""
        if value in self.empty_values:
            return ''
        return str(value)

    def validate(self, value):
        """Validate that the input is in self.choices."""
        super().validate(value)
        if value and not self.valid_value(value):
            raise ValidationError(
                self.error_messages['invalid_choice'],
                code='invalid_choice',
                params={'value': value},
            )

    def valid_value(self, value):
        """Check to see if the provided value is a valid choice."""
        text_value = str(value)
        for k, v in self.choices:
            if isinstance(v, (list, tuple)):
                # This is an optgroup, so look inside the group for options
                for k2, v2 in v:
                    if value == k2 or text_value == str(k2):
                        return True
            else:
                if value == k or text_value == str(k):
                    return True
        return False
```
### 10 - django/forms/models.py:

Start line: 1141, End line: 1183

```python
class ModelChoiceIteratorValue:
    def __init__(self, value, instance):
        self.value = value
        self.instance = instance

    def __str__(self):
        return str(self.value)

    def __eq__(self, other):
        if isinstance(other, ModelChoiceIteratorValue):
            other = other.value
        return self.value == other


class ModelChoiceIterator:
    def __init__(self, field):
        self.field = field
        self.queryset = field.queryset

    def __iter__(self):
        if self.field.empty_label is not None:
            yield ("", self.field.empty_label)
        queryset = self.queryset
        # Can't use iterator() when queryset uses prefetch_related()
        if not queryset._prefetch_related_lookups:
            queryset = queryset.iterator()
        for obj in queryset:
            yield self.choice(obj)

    def __len__(self):
        # count() adds a query but uses less memory since the QuerySet results
        # won't be cached. In most cases, the choices will only be iterated on,
        # and __len__() won't be called.
        return self.queryset.count() + (1 if self.field.empty_label is not None else 0)

    def __bool__(self):
        return self.field.empty_label is not None or self.queryset.exists()

    def choice(self, obj):
        return (
            ModelChoiceIteratorValue(self.field.prepare_value(obj), obj),
            self.field.label_from_instance(obj),
        )
```
### 11 - django/forms/models.py:

Start line: 1253, End line: 1266

```python
class ModelChoiceField(ChoiceField):

    def _get_choices(self):
        # If self._choices is set, then somebody must have manually set
        # the property self.choices. In this case, just return self._choices.
        if hasattr(self, '_choices'):
            return self._choices

        # Otherwise, execute the QuerySet in self.queryset to determine the
        # choices dynamically. Return a fresh ModelChoiceIterator that has not been
        # consumed. Note that we're instantiating a new ModelChoiceIterator *each*
        # time _get_choices() is called (and, thus, each time self.choices is
        # accessed) so that we can ensure the QuerySet has not been consumed. This
        # construct might look complicated but it allows for lazy evaluation of
        # the queryset.
        return self.iterator(self)
```
### 18 - django/forms/models.py:

Start line: 759, End line: 780

```python
class BaseModelFormSet(BaseFormSet):

    def get_unique_error_message(self, unique_check):
        if len(unique_check) == 1:
            return gettext("Please correct the duplicate data for %(field)s.") % {
                "field": unique_check[0],
            }
        else:
            return gettext("Please correct the duplicate data for %(field)s, which must be unique.") % {
                "field": get_text_list(unique_check, _("and")),
            }

    def get_date_error_message(self, date_check):
        return gettext(
            "Please correct the duplicate data for %(field_name)s "
            "which must be unique for the %(lookup)s in %(date_field)s."
        ) % {
            'field_name': date_check[2],
            'date_field': date_check[3],
            'lookup': str(date_check[1]),
        }

    def get_form_error(self):
        return gettext("Please correct the duplicate values below.")
```
### 48 - django/forms/models.py:

Start line: 686, End line: 757

```python
class BaseModelFormSet(BaseFormSet):

    def validate_unique(self):
        # Collect unique_checks and date_checks to run from all the forms.
        all_unique_checks = set()
        all_date_checks = set()
        forms_to_delete = self.deleted_forms
        valid_forms = [form for form in self.forms if form.is_valid() and form not in forms_to_delete]
        for form in valid_forms:
            exclude = form._get_validation_exclusions()
            unique_checks, date_checks = form.instance._get_unique_checks(exclude=exclude)
            all_unique_checks.update(unique_checks)
            all_date_checks.update(date_checks)

        errors = []
        # Do each of the unique checks (unique and unique_together)
        for uclass, unique_check in all_unique_checks:
            seen_data = set()
            for form in valid_forms:
                # Get the data for the set of fields that must be unique among the forms.
                row_data = (
                    field if field in self.unique_fields else form.cleaned_data[field]
                    for field in unique_check if field in form.cleaned_data
                )
                # Reduce Model instances to their primary key values
                row_data = tuple(
                    d._get_pk_val() if hasattr(d, '_get_pk_val')
                    # Prevent "unhashable type: list" errors later on.
                    else tuple(d) if isinstance(d, list)
                    else d for d in row_data
                )
                if row_data and None not in row_data:
                    # if we've already seen it then we have a uniqueness failure
                    if row_data in seen_data:
                        # poke error messages into the right places and mark
                        # the form as invalid
                        errors.append(self.get_unique_error_message(unique_check))
                        form._errors[NON_FIELD_ERRORS] = self.error_class([self.get_form_error()])
                        # remove the data from the cleaned_data dict since it was invalid
                        for field in unique_check:
                            if field in form.cleaned_data:
                                del form.cleaned_data[field]
                    # mark the data as seen
                    seen_data.add(row_data)
        # iterate over each of the date checks now
        for date_check in all_date_checks:
            seen_data = set()
            uclass, lookup, field, unique_for = date_check
            for form in valid_forms:
                # see if we have data for both fields
                if (form.cleaned_data and form.cleaned_data[field] is not None and
                        form.cleaned_data[unique_for] is not None):
                    # if it's a date lookup we need to get the data for all the fields
                    if lookup == 'date':
                        date = form.cleaned_data[unique_for]
                        date_data = (date.year, date.month, date.day)
                    # otherwise it's just the attribute on the date/datetime
                    # object
                    else:
                        date_data = (getattr(form.cleaned_data[unique_for], lookup),)
                    data = (form.cleaned_data[field],) + date_data
                    # if we've already seen it then we have a uniqueness failure
                    if data in seen_data:
                        # poke error messages into the right places and mark
                        # the form as invalid
                        errors.append(self.get_date_error_message(date_check))
                        form._errors[NON_FIELD_ERRORS] = self.error_class([self.get_form_error()])
                        # remove the data from the cleaned_data dict since it was invalid
                        del form.cleaned_data[field]
                    # mark the data as seen
                    seen_data.add(data)

        if errors:
            raise ValidationError(errors)
```
### 49 - django/forms/models.py:

Start line: 1, End line: 27

```python
"""
Helper functions for creating Form classes from Django models
and database field objects.
"""
from itertools import chain

from django.core.exceptions import (
    NON_FIELD_ERRORS, FieldError, ImproperlyConfigured, ValidationError,
)
from django.forms.fields import ChoiceField, Field
from django.forms.forms import BaseForm, DeclarativeFieldsMetaclass
from django.forms.formsets import BaseFormSet, formset_factory
from django.forms.utils import ErrorList
from django.forms.widgets import (
    HiddenInput, MultipleHiddenInput, RadioSelect, SelectMultiple,
)
from django.utils.text import capfirst, get_text_list
from django.utils.translation import gettext, gettext_lazy as _

__all__ = (
    'ModelForm', 'BaseModelForm', 'model_to_dict', 'fields_for_model',
    'ModelChoiceField', 'ModelMultipleChoiceField', 'ALL_FIELDS',
    'BaseModelFormSet', 'modelformset_factory', 'BaseInlineFormSet',
    'inlineformset_factory', 'modelform_factory',
)

ALL_FIELDS = '__all__'
```
### 60 - django/forms/models.py:

Start line: 389, End line: 417

```python
class BaseModelForm(BaseForm):

    def _post_clean(self):
        opts = self._meta

        exclude = self._get_validation_exclusions()

        # Foreign Keys being used to represent inline relationships
        # are excluded from basic field value validation. This is for two
        # reasons: firstly, the value may not be supplied (#12507; the
        # case of providing new values to the admin); secondly the
        # object being referred to may not yet fully exist (#12749).
        # However, these fields *must* be included in uniqueness checks,
        # so this can't be part of _get_validation_exclusions().
        for name, field in self.fields.items():
            if isinstance(field, InlineForeignKeyField):
                exclude.append(name)

        try:
            self.instance = construct_instance(self, self.instance, opts.fields, opts.exclude)
        except ValidationError as e:
            self._update_errors(e)

        try:
            self.instance.full_clean(exclude=exclude, validate_unique=False)
        except ValidationError as e:
            self._update_errors(e)

        # Validate uniqueness if needed.
        if self._validate_unique:
            self.validate_unique()
```
### 62 - django/forms/models.py:

Start line: 316, End line: 355

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
### 65 - django/forms/models.py:

Start line: 357, End line: 387

```python
class BaseModelForm(BaseForm):

    def clean(self):
        self._validate_unique = True
        return self.cleaned_data

    def _update_errors(self, errors):
        # Override any validation error messages defined at the model level
        # with those defined at the form level.
        opts = self._meta

        # Allow the model generated by construct_instance() to raise
        # ValidationError and have them handled in the same way as others.
        if hasattr(errors, 'error_dict'):
            error_dict = errors.error_dict
        else:
            error_dict = {NON_FIELD_ERRORS: errors}

        for field, messages in error_dict.items():
            if (field == NON_FIELD_ERRORS and opts.error_messages and
                    NON_FIELD_ERRORS in opts.error_messages):
                error_messages = opts.error_messages[NON_FIELD_ERRORS]
            elif field in self.fields:
                error_messages = self.fields[field].error_messages
            else:
                continue

            for message in messages:
                if (isinstance(message, ValidationError) and
                        message.code in error_messages):
                    message.message = error_messages[message.code]

        self.add_error(None, errors)
```
### 77 - django/forms/models.py:

Start line: 201, End line: 211

```python
class ModelFormOptions:
    def __init__(self, options=None):
        self.model = getattr(options, 'model', None)
        self.fields = getattr(options, 'fields', None)
        self.exclude = getattr(options, 'exclude', None)
        self.widgets = getattr(options, 'widgets', None)
        self.localized_fields = getattr(options, 'localized_fields', None)
        self.labels = getattr(options, 'labels', None)
        self.help_texts = getattr(options, 'help_texts', None)
        self.error_messages = getattr(options, 'error_messages', None)
        self.field_classes = getattr(options, 'field_classes', None)
```
### 90 - django/forms/models.py:

Start line: 419, End line: 449

```python
class BaseModelForm(BaseForm):

    def validate_unique(self):
        """
        Call the instance's validate_unique() method and update the form's
        validation errors if any were raised.
        """
        exclude = self._get_validation_exclusions()
        try:
            self.instance.validate_unique(exclude=exclude)
        except ValidationError as e:
            self._update_errors(e)

    def _save_m2m(self):
        """
        Save the many-to-many fields and generic relations for this form.
        """
        cleaned_data = self.cleaned_data
        exclude = self._meta.exclude
        fields = self._meta.fields
        opts = self.instance._meta
        # Note that for historical reasons we want to include also
        # private_fields here. (GenericRelation was previously a fake
        # m2m field).
        for f in chain(opts.many_to_many, opts.private_fields):
            if not hasattr(f, 'save_form_data'):
                continue
            if fields and f.name not in fields:
                continue
            if exclude and f.name in exclude:
                continue
            if f.name in cleaned_data:
                f.save_form_data(self.instance, cleaned_data[f.name])
```
### 132 - django/forms/models.py:

Start line: 961, End line: 994

```python
class BaseInlineFormSet(BaseModelFormSet):

    def add_fields(self, form, index):
        super().add_fields(form, index)
        if self._pk_field == self.fk:
            name = self._pk_field.name
            kwargs = {'pk_field': True}
        else:
            # The foreign key field might not be on the form, so we poke at the
            # Model field to get the label, since we need that for error messages.
            name = self.fk.name
            kwargs = {
                'label': getattr(form.fields.get(name), 'label', capfirst(self.fk.verbose_name))
            }

        # The InlineForeignKeyField assumes that the foreign key relation is
        # based on the parent model's pk. If this isn't the case, set to_field
        # to correctly resolve the initial form value.
        if self.fk.remote_field.field_name != self.fk.remote_field.model._meta.pk.name:
            kwargs['to_field'] = self.fk.remote_field.field_name

        # If we're adding a new object, ignore a parent's auto-generated key
        # as it will be regenerated on the save request.
        if self.instance._state.adding:
            if kwargs.get('to_field') is not None:
                to_field = self.instance._meta.get_field(kwargs['to_field'])
            else:
                to_field = self.instance._meta.pk
            if to_field.has_default():
                setattr(self.instance, to_field.attname, None)

        form.fields[name] = InlineForeignKeyField(self.instance, **kwargs)

    def get_unique_error_message(self, unique_check):
        unique_check = [field for field in unique_check if field != self.fk.name]
        return super().get_unique_error_message(unique_check)
```
### 134 - django/forms/models.py:

Start line: 96, End line: 109

```python
def apply_limit_choices_to_to_formfield(formfield):
    """Apply limit_choices_to to the formfield's queryset if needed."""
    from django.db.models import Exists, OuterRef, Q
    if hasattr(formfield, 'queryset') and hasattr(formfield, 'get_limit_choices_to'):
        limit_choices_to = formfield.get_limit_choices_to()
        if limit_choices_to:
            complex_filter = limit_choices_to
            if not isinstance(complex_filter, Q):
                complex_filter = Q(**limit_choices_to)
            complex_filter &= Q(pk=OuterRef('pk'))
            # Use Exists() to avoid potential duplicates.
            formfield.queryset = formfield.queryset.filter(
                Exists(formfield.queryset.model._base_manager.filter(complex_filter)),
            )
```
### 135 - django/forms/models.py:

Start line: 214, End line: 283

```python
class ModelFormMetaclass(DeclarativeFieldsMetaclass):
    def __new__(mcs, name, bases, attrs):
        base_formfield_callback = None
        for b in bases:
            if hasattr(b, 'Meta') and hasattr(b.Meta, 'formfield_callback'):
                base_formfield_callback = b.Meta.formfield_callback
                break

        formfield_callback = attrs.pop('formfield_callback', base_formfield_callback)

        new_class = super().__new__(mcs, name, bases, attrs)

        if bases == (BaseModelForm,):
            return new_class

        opts = new_class._meta = ModelFormOptions(getattr(new_class, 'Meta', None))

        # We check if a string was passed to `fields` or `exclude`,
        # which is likely to be a mistake where the user typed ('foo') instead
        # of ('foo',)
        for opt in ['fields', 'exclude', 'localized_fields']:
            value = getattr(opts, opt)
            if isinstance(value, str) and value != ALL_FIELDS:
                msg = ("%(model)s.Meta.%(opt)s cannot be a string. "
                       "Did you mean to type: ('%(value)s',)?" % {
                           'model': new_class.__name__,
                           'opt': opt,
                           'value': value,
                       })
                raise TypeError(msg)

        if opts.model:
            # If a model is defined, extract form fields from it.
            if opts.fields is None and opts.exclude is None:
                raise ImproperlyConfigured(
                    "Creating a ModelForm without either the 'fields' attribute "
                    "or the 'exclude' attribute is prohibited; form %s "
                    "needs updating." % name
                )

            if opts.fields == ALL_FIELDS:
                # Sentinel for fields_for_model to indicate "get the list of
                # fields from the model"
                opts.fields = None

            fields = fields_for_model(
                opts.model, opts.fields, opts.exclude, opts.widgets,
                formfield_callback, opts.localized_fields, opts.labels,
                opts.help_texts, opts.error_messages, opts.field_classes,
                # limit_choices_to will be applied during ModelForm.__init__().
                apply_limit_choices_to=False,
            )

            # make sure opts.fields doesn't specify an invalid field
            none_model_fields = {k for k, v in fields.items() if not v}
            missing_fields = none_model_fields.difference(new_class.declared_fields)
            if missing_fields:
                message = 'Unknown field(s) (%s) specified for %s'
                message = message % (', '.join(missing_fields),
                                     opts.model.__name__)
                raise FieldError(message)
            # Override default model fields with any custom declared ones
            # (plus, include all the other declared fields).
            fields.update(new_class.declared_fields)
        else:
            fields = new_class.declared_fields

        new_class.base_fields = fields

        return new_class
```
