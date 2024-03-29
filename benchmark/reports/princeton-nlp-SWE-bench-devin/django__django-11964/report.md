# django__django-11964

| **django/django** | `fc2b1cc926e34041953738e58fa6ad3053059b22` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 259 |
| **Any found context length** | 259 |
| **Avg pos** | 1.0 |
| **Min pos** | 1 |
| **Max pos** | 1 |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/db/models/enums.py b/django/db/models/enums.py
--- a/django/db/models/enums.py
+++ b/django/db/models/enums.py
@@ -60,7 +60,13 @@ def values(cls):
 
 class Choices(enum.Enum, metaclass=ChoicesMeta):
     """Class for creating enumerated choices."""
-    pass
+
+    def __str__(self):
+        """
+        Use value when cast to str, so that Choices set as model instance
+        attributes are rendered as expected in templates and similar contexts.
+        """
+        return str(self.value)
 
 
 class IntegerChoices(int, Choices):

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/db/models/enums.py | 63 | 63 | 1 | 1 | 259


## Problem Statement

```
The value of a TextChoices/IntegerChoices field has a differing type
Description
	
If we create an instance of a model having a CharField or IntegerField with the keyword choices pointing to IntegerChoices or TextChoices, the value returned by the getter of the field will be of the same type as the one created by enum.Enum (enum value).
For example, this model:
from django.db import models
from django.utils.translation import gettext_lazy as _
class MyChoice(models.TextChoices):
	FIRST_CHOICE = "first", _("The first choice, it is")
	SECOND_CHOICE = "second", _("The second choice, it is")
class MyObject(models.Model):
	my_str_value = models.CharField(max_length=10, choices=MyChoice.choices)
Then this test:
from django.test import TestCase
from testing.pkg.models import MyObject, MyChoice
class EnumTest(TestCase):
	def setUp(self) -> None:
		self.my_object = MyObject.objects.create(my_str_value=MyChoice.FIRST_CHOICE)
	def test_created_object_is_str(self):
		my_object = self.my_object
		self.assertIsInstance(my_object.my_str_value, str)
		self.assertEqual(str(my_object.my_str_value), "first")
	def test_retrieved_object_is_str(self):
		my_object = MyObject.objects.last()
		self.assertIsInstance(my_object.my_str_value, str)
		self.assertEqual(str(my_object.my_str_value), "first")
And then the results:
(django30-venv) ➜ django30 ./manage.py test
Creating test database for alias 'default'...
System check identified no issues (0 silenced).
F.
======================================================================
FAIL: test_created_object_is_str (testing.tests.EnumTest)
----------------------------------------------------------------------
Traceback (most recent call last):
 File "/Users/mikailkocak/Development/django30/testing/tests.py", line 14, in test_created_object_is_str
	self.assertEqual(str(my_object.my_str_value), "first")
AssertionError: 'MyChoice.FIRST_CHOICE' != 'first'
- MyChoice.FIRST_CHOICE
+ first
----------------------------------------------------------------------
Ran 2 tests in 0.002s
FAILED (failures=1)
We notice when invoking __str__(...) we don't actually get the value property of the enum value which can lead to some unexpected issues, especially when communicating to an external API with a freshly created instance that will send MyEnum.MyValue, and the one that was retrieved would send my_value.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 django/db/models/enums.py** | 36 | 76| 259 | 259 | 541 | 
| 2 | 2 django/forms/models.py | 1236 | 1266| 242 | 501 | 12054 | 
| 3 | **2 django/db/models/enums.py** | 1 | 34| 288 | 789 | 12054 | 
| 4 | 3 django/forms/fields.py | 758 | 817| 416 | 1205 | 20988 | 
| 5 | 4 django/db/models/fields/__init__.py | 240 | 305| 467 | 1672 | 38395 | 
| 6 | 4 django/forms/fields.py | 820 | 844| 177 | 1849 | 38395 | 
| 7 | 4 django/forms/models.py | 1339 | 1366| 209 | 2058 | 38395 | 
| 8 | 4 django/forms/fields.py | 889 | 922| 235 | 2293 | 38395 | 
| 9 | 4 django/forms/models.py | 1221 | 1234| 176 | 2469 | 38395 | 
| 10 | 4 django/forms/models.py | 1156 | 1219| 520 | 2989 | 38395 | 
| 11 | 4 django/db/models/fields/__init__.py | 856 | 877| 162 | 3151 | 38395 | 
| 12 | 4 django/forms/models.py | 1128 | 1153| 226 | 3377 | 38395 | 
| 13 | 4 django/forms/fields.py | 847 | 886| 298 | 3675 | 38395 | 
| 14 | 4 django/db/models/fields/__init__.py | 1755 | 1800| 279 | 3954 | 38395 | 
| 15 | 4 django/db/models/fields/__init__.py | 949 | 965| 176 | 4130 | 38395 | 
| 16 | 4 django/db/models/fields/__init__.py | 2012 | 2042| 199 | 4329 | 38395 | 
| 17 | 4 django/db/models/fields/__init__.py | 1 | 81| 628 | 4957 | 38395 | 
| 18 | 5 django/contrib/admin/filters.py | 278 | 302| 217 | 5174 | 42129 | 
| 19 | 5 django/db/models/fields/__init__.py | 830 | 854| 244 | 5418 | 42129 | 
| 20 | 5 django/contrib/admin/filters.py | 397 | 419| 211 | 5629 | 42129 | 
| 21 | 6 django/db/models/fields/reverse_related.py | 117 | 134| 161 | 5790 | 44272 | 
| 22 | 7 django/db/migrations/serializer.py | 119 | 138| 136 | 5926 | 46820 | 
| 23 | 7 django/forms/models.py | 1269 | 1300| 266 | 6192 | 46820 | 
| 24 | 7 django/db/models/fields/__init__.py | 922 | 947| 209 | 6401 | 46820 | 
| 25 | 7 django/db/models/fields/__init__.py | 1953 | 1976| 131 | 6532 | 46820 | 
| 26 | 8 django/db/models/options.py | 338 | 361| 198 | 6730 | 53919 | 
| 27 | 9 django/forms/widgets.py | 671 | 702| 261 | 6991 | 61985 | 
| 28 | 9 django/db/models/fields/__init__.py | 1701 | 1724| 146 | 7137 | 61985 | 
| 29 | 9 django/forms/widgets.py | 639 | 668| 238 | 7375 | 61985 | 
| 30 | 9 django/forms/widgets.py | 618 | 637| 189 | 7564 | 61985 | 
| 31 | 9 django/db/models/fields/__init__.py | 1726 | 1753| 215 | 7779 | 61985 | 
| 32 | 9 django/db/models/fields/__init__.py | 1002 | 1028| 229 | 8008 | 61985 | 
| 33 | 10 django/contrib/admin/options.py | 188 | 204| 171 | 8179 | 80351 | 
| 34 | 10 django/contrib/admin/filters.py | 246 | 261| 154 | 8333 | 80351 | 
| 35 | 10 django/forms/fields.py | 1077 | 1118| 353 | 8686 | 80351 | 
| 36 | 11 django/db/models/functions/text.py | 1 | 21| 183 | 8869 | 82807 | 
| 37 | 11 django/forms/models.py | 1085 | 1125| 306 | 9175 | 82807 | 
| 38 | 12 django/db/models/base.py | 504 | 546| 347 | 9522 | 98096 | 
| 39 | 13 django/db/models/fields/related.py | 640 | 656| 163 | 9685 | 111608 | 
| 40 | 13 django/db/models/fields/__init__.py | 1031 | 1044| 104 | 9789 | 111608 | 
| 41 | 13 django/db/models/fields/__init__.py | 607 | 636| 234 | 10023 | 111608 | 
| 42 | 13 django/forms/fields.py | 723 | 755| 241 | 10264 | 111608 | 
| 43 | 13 django/db/models/fields/__init__.py | 2264 | 2314| 339 | 10603 | 111608 | 
| 44 | 13 django/db/models/fields/__init__.py | 1914 | 1933| 164 | 10767 | 111608 | 
| 45 | 14 django/db/migrations/questioner.py | 84 | 107| 187 | 10954 | 113682 | 
| 46 | 14 django/contrib/admin/filters.py | 264 | 276| 149 | 11103 | 113682 | 
| 47 | 14 django/forms/models.py | 1302 | 1337| 286 | 11389 | 113682 | 
| 48 | 14 django/db/models/functions/text.py | 58 | 77| 153 | 11542 | 113682 | 
| 49 | 14 django/db/migrations/questioner.py | 187 | 205| 237 | 11779 | 113682 | 
| 50 | 14 django/db/models/fields/__init__.py | 1661 | 1698| 226 | 12005 | 113682 | 
| 51 | 14 django/db/models/base.py | 1659 | 1757| 717 | 12722 | 113682 | 
| 52 | 15 django/contrib/admin/migrations/0003_logentry_add_action_flag_choices.py | 1 | 21| 111 | 12833 | 113793 | 
| 53 | 16 django/db/models/lookups.py | 278 | 328| 306 | 13139 | 118467 | 
| 54 | 17 django/template/defaultfilters.py | 771 | 801| 273 | 13412 | 124532 | 
| 55 | 17 django/forms/fields.py | 1172 | 1202| 182 | 13594 | 124532 | 
| 56 | 17 django/db/migrations/questioner.py | 162 | 185| 246 | 13840 | 124532 | 
| 57 | 17 django/db/models/options.py | 149 | 208| 587 | 14427 | 124532 | 
| 58 | 17 django/db/migrations/serializer.py | 1 | 73| 428 | 14855 | 124532 | 
| 59 | 17 django/db/models/base.py | 1553 | 1578| 183 | 15038 | 124532 | 
| 60 | 17 django/contrib/admin/filters.py | 209 | 226| 190 | 15228 | 124532 | 
| 61 | 17 django/db/migrations/questioner.py | 109 | 141| 290 | 15518 | 124532 | 
| 62 | 17 django/db/models/fields/__init__.py | 638 | 662| 206 | 15724 | 124532 | 
| 63 | 17 django/db/models/fields/__init__.py | 769 | 828| 431 | 16155 | 124532 | 
| 64 | 18 django/forms/boundfield.py | 35 | 50| 149 | 16304 | 126653 | 
| 65 | 19 django/contrib/admin/widgets.py | 423 | 447| 232 | 16536 | 130519 | 
| 66 | 19 django/db/models/fields/__init__.py | 1047 | 1076| 218 | 16754 | 130519 | 
| 67 | 19 django/db/models/base.py | 1157 | 1185| 213 | 16967 | 130519 | 
| 68 | 20 django/core/checks/messages.py | 26 | 50| 259 | 17226 | 131092 | 
| 69 | 20 django/forms/widgets.py | 548 | 582| 277 | 17503 | 131092 | 
| 70 | 20 django/db/models/base.py | 1492 | 1524| 231 | 17734 | 131092 | 
| 71 | 21 django/contrib/admin/utils.py | 366 | 400| 305 | 18039 | 135180 | 
| 72 | 21 django/forms/models.py | 1 | 28| 215 | 18254 | 135180 | 
| 73 | 22 django/contrib/admin/checks.py | 424 | 445| 191 | 18445 | 144196 | 
| 74 | 22 django/db/models/base.py | 403 | 502| 856 | 19301 | 144196 | 
| 75 | 22 django/contrib/admin/checks.py | 1087 | 1117| 188 | 19489 | 144196 | 
| 76 | 23 django/db/migrations/state.py | 349 | 399| 471 | 19960 | 149414 | 
| 77 | 24 django/db/models/fields/mixins.py | 31 | 57| 173 | 20133 | 149757 | 
| 78 | 24 django/forms/fields.py | 260 | 284| 195 | 20328 | 149757 | 
| 79 | 24 django/db/models/fields/__init__.py | 1378 | 1401| 171 | 20499 | 149757 | 
| 80 | 24 django/db/models/options.py | 210 | 260| 446 | 20945 | 149757 | 
| 81 | 25 django/contrib/humanize/templatetags/humanize.py | 1 | 57| 648 | 21593 | 152917 | 
| 82 | 25 django/db/models/fields/__init__.py | 1451 | 1510| 398 | 21991 | 152917 | 
| 83 | 25 django/forms/models.py | 752 | 773| 194 | 22185 | 152917 | 
| 84 | 26 django/core/serializers/base.py | 297 | 319| 207 | 22392 | 155310 | 
| 85 | 26 django/contrib/admin/widgets.py | 352 | 378| 328 | 22720 | 155310 | 
| 86 | 26 django/contrib/admin/checks.py | 521 | 544| 230 | 22950 | 155310 | 
| 87 | 26 django/db/models/base.py | 1468 | 1490| 171 | 23121 | 155310 | 
| 88 | 26 django/db/models/fields/__init__.py | 968 | 1000| 208 | 23329 | 155310 | 
| 89 | 26 django/db/migrations/questioner.py | 56 | 81| 220 | 23549 | 155310 | 
| 90 | 26 django/db/models/options.py | 415 | 437| 154 | 23703 | 155310 | 
| 91 | 26 django/contrib/admin/utils.py | 403 | 432| 203 | 23906 | 155310 | 
| 92 | 26 django/db/models/base.py | 1066 | 1109| 404 | 24310 | 155310 | 
| 93 | 27 django/db/backends/oracle/creation.py | 317 | 401| 739 | 25049 | 159205 | 
| 94 | 27 django/contrib/admin/options.py | 1 | 96| 769 | 25818 | 159205 | 
| 95 | 28 django/core/validators.py | 419 | 465| 415 | 26233 | 163527 | 
| 96 | 29 django/contrib/auth/migrations/0001_initial.py | 1 | 105| 849 | 27082 | 164376 | 
| 97 | 29 django/db/models/base.py | 1443 | 1466| 176 | 27258 | 164376 | 
| 98 | 30 django/contrib/admin/models.py | 74 | 94| 161 | 27419 | 165501 | 
| 99 | 30 django/db/migrations/questioner.py | 143 | 160| 183 | 27602 | 165501 | 
| 100 | 31 django/db/models/expressions.py | 666 | 690| 238 | 27840 | 175801 | 
| 101 | 31 django/db/migrations/serializer.py | 314 | 341| 231 | 28071 | 175801 | 
| 102 | 31 django/contrib/admin/filters.py | 20 | 59| 295 | 28366 | 175801 | 
| 103 | 31 django/db/models/base.py | 645 | 660| 153 | 28519 | 175801 | 
| 104 | 31 django/db/migrations/state.py | 580 | 599| 188 | 28707 | 175801 | 
| 105 | 31 django/db/models/fields/__init__.py | 1834 | 1911| 567 | 29274 | 175801 | 
| 106 | 31 django/forms/fields.py | 241 | 258| 162 | 29436 | 175801 | 
| 107 | 32 django/contrib/admin/tests.py | 164 | 185| 205 | 29641 | 177217 | 
| 108 | 32 django/db/models/functions/text.py | 246 | 301| 372 | 30013 | 177217 | 
| 109 | 32 django/db/models/fields/__init__.py | 2397 | 2422| 143 | 30156 | 177217 | 
| 110 | 32 django/db/models/fields/related.py | 1202 | 1313| 939 | 31095 | 177217 | 
| 111 | 32 django/db/models/fields/__init__.py | 1237 | 1278| 332 | 31427 | 177217 | 
| 112 | 32 django/db/models/lookups.py | 581 | 614| 141 | 31568 | 177217 | 
| 113 | 33 django/contrib/gis/gdal/geomtype.py | 1 | 96| 757 | 32325 | 177975 | 
| 114 | 33 django/forms/fields.py | 207 | 238| 274 | 32599 | 177975 | 
| 115 | 33 django/db/models/fields/__init__.py | 879 | 919| 387 | 32986 | 177975 | 
| 116 | 33 django/db/migrations/state.py | 601 | 612| 136 | 33122 | 177975 | 
| 117 | 33 django/db/models/base.py | 1526 | 1551| 183 | 33305 | 177975 | 
| 118 | 33 django/db/models/options.py | 296 | 314| 136 | 33441 | 177975 | 
| 119 | 33 django/db/models/fields/__init__.py | 1222 | 1235| 157 | 33598 | 177975 | 
| 120 | 33 django/db/models/fields/__init__.py | 84 | 176| 822 | 34420 | 177975 | 
| 121 | 33 django/contrib/admin/checks.py | 447 | 473| 190 | 34610 | 177975 | 
| 122 | 33 django/db/models/base.py | 1140 | 1155| 138 | 34748 | 177975 | 
| 123 | 33 django/db/models/base.py | 1609 | 1657| 348 | 35096 | 177975 | 
| 124 | 33 django/db/models/base.py | 946 | 960| 212 | 35308 | 177975 | 
| 125 | 34 django/db/backends/base/schema.py | 278 | 299| 174 | 35482 | 189291 | 
| 126 | 35 django/contrib/postgres/fields/citext.py | 1 | 25| 113 | 35595 | 189405 | 
| 127 | 35 django/db/models/fields/__init__.py | 1936 | 1950| 131 | 35726 | 189405 | 
| 128 | 35 django/db/models/fields/__init__.py | 1280 | 1329| 342 | 36068 | 189405 | 
| 129 | 35 django/db/models/base.py | 1248 | 1277| 242 | 36310 | 189405 | 
| 130 | 35 django/db/models/options.py | 1 | 36| 304 | 36614 | 189405 | 
| 131 | 35 django/db/models/fields/related.py | 1611 | 1645| 266 | 36880 | 189405 | 
| 132 | 36 django/db/models/functions/comparison.py | 1 | 29| 317 | 37197 | 190483 | 
| 133 | 36 django/forms/widgets.py | 705 | 740| 237 | 37434 | 190483 | 
| 134 | 36 django/contrib/admin/checks.py | 879 | 927| 416 | 37850 | 190483 | 
| 135 | 37 django/utils/encoding.py | 1 | 45| 290 | 38140 | 192843 | 
| 136 | 37 django/db/models/fields/related.py | 964 | 991| 215 | 38355 | 192843 | 
| 137 | 37 django/db/migrations/questioner.py | 227 | 240| 123 | 38478 | 192843 | 
| 138 | 38 django/db/backends/base/features.py | 1 | 115| 900 | 39378 | 195395 | 
| 139 | 38 django/db/models/fields/__init__.py | 690 | 747| 425 | 39803 | 195395 | 


### Hint

```
Hi NyanKiyoshi, what a lovely report. Thank you. Clearly :) the expected behaviour is that test_created_object_is_str should pass. It's interesting that the underlying __dict__ values differ, which explains all I guess: Created: {'_state': <django.db.models.base.ModelState object at 0x10730efd0>, 'id': 1, 'my_str_value': <MyChoice.FIRST_CHOICE: 'first'>} Retrieved: {'_state': <django.db.models.base.ModelState object at 0x1072b5eb8>, 'id': 1, 'my_str_value': 'first'} Good catch. Thanks again.
Sample project with provided models. Run ./manage.py test
```

## Patch

```diff
diff --git a/django/db/models/enums.py b/django/db/models/enums.py
--- a/django/db/models/enums.py
+++ b/django/db/models/enums.py
@@ -60,7 +60,13 @@ def values(cls):
 
 class Choices(enum.Enum, metaclass=ChoicesMeta):
     """Class for creating enumerated choices."""
-    pass
+
+    def __str__(self):
+        """
+        Use value when cast to str, so that Choices set as model instance
+        attributes are rendered as expected in templates and similar contexts.
+        """
+        return str(self.value)
 
 
 class IntegerChoices(int, Choices):

```

## Test Patch

```diff
diff --git a/tests/model_enums/tests.py b/tests/model_enums/tests.py
--- a/tests/model_enums/tests.py
+++ b/tests/model_enums/tests.py
@@ -143,6 +143,12 @@ class Fruit(models.IntegerChoices):
                 APPLE = 1, 'Apple'
                 PINEAPPLE = 1, 'Pineapple'
 
+    def test_str(self):
+        for test in [Gender, Suit, YearInSchool, Vehicle]:
+            for member in test:
+                with self.subTest(member=member):
+                    self.assertEqual(str(test[member.name]), str(member.value))
+
 
 class Separator(bytes, models.Choices):
     FS = b'\x1c', 'File Separator'

```


## Code snippets

### 1 - django/db/models/enums.py:

Start line: 36, End line: 76

```python
class ChoicesMeta(enum.EnumMeta):

    def __contains__(cls, member):
        if not isinstance(member, enum.Enum):
            # Allow non-enums to match against member values.
            return member in {x.value for x in cls}
        return super().__contains__(member)

    @property
    def names(cls):
        empty = ['__empty__'] if hasattr(cls, '__empty__') else []
        return empty + [member.name for member in cls]

    @property
    def choices(cls):
        empty = [(None, cls.__empty__)] if hasattr(cls, '__empty__') else []
        return empty + [(member.value, member.label) for member in cls]

    @property
    def labels(cls):
        return [label for _, label in cls.choices]

    @property
    def values(cls):
        return [value for value, _ in cls.choices]


class Choices(enum.Enum, metaclass=ChoicesMeta):
    """Class for creating enumerated choices."""
    pass


class IntegerChoices(int, Choices):
    """Class for creating enumerated integer choices."""
    pass


class TextChoices(str, Choices):
    """Class for creating enumerated string choices."""

    def _generate_next_value_(name, start, count, last_values):
        return name
```
### 2 - django/forms/models.py:

Start line: 1236, End line: 1266

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
### 3 - django/db/models/enums.py:

Start line: 1, End line: 34

```python
import enum

from django.utils.functional import Promise

__all__ = ['Choices', 'IntegerChoices', 'TextChoices']


class ChoicesMeta(enum.EnumMeta):
    """A metaclass for creating a enum choices."""

    def __new__(metacls, classname, bases, classdict):
        labels = []
        for key in classdict._member_names:
            value = classdict[key]
            if (
                isinstance(value, (list, tuple)) and
                len(value) > 1 and
                isinstance(value[-1], (Promise, str))
            ):
                *value, label = value
                value = tuple(value)
            else:
                label = key.replace('_', ' ').title()
            labels.append(label)
            # Use dict.__setitem__() to suppress defenses against double
            # assignment in enum's classdict.
            dict.__setitem__(classdict, key, value)
        cls = super().__new__(metacls, classname, bases, classdict)
        cls._value2label_map_ = dict(zip(cls._value2member_map_, labels))
        # Add a label property to instances of enum which uses the enum member
        # that is passed in as "self" as the value to use when looking up the
        # label in the choices.
        cls.label = property(lambda self: cls._value2label_map_.get(self.value))
        return enum.unique(cls)
```
### 4 - django/forms/fields.py:

Start line: 758, End line: 817

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
### 5 - django/db/models/fields/__init__.py:

Start line: 240, End line: 305

```python
@total_ordering
class Field(RegisterLookupMixin):

    def _check_choices(self):
        if not self.choices:
            return []

        def is_value(value, accept_promise=True):
            return isinstance(value, (str, Promise) if accept_promise else str) or not is_iterable(value)

        if is_value(self.choices, accept_promise=False):
            return [
                checks.Error(
                    "'choices' must be an iterable (e.g., a list or tuple).",
                    obj=self,
                    id='fields.E004',
                )
            ]

        choice_max_length = 0
        # Expect [group_name, [value, display]]
        for choices_group in self.choices:
            try:
                group_name, group_choices = choices_group
            except (TypeError, ValueError):
                # Containing non-pairs
                break
            try:
                if not all(
                    is_value(value) and is_value(human_name)
                    for value, human_name in group_choices
                ):
                    break
                if self.max_length is not None and group_choices:
                    choice_max_length = max(
                        choice_max_length,
                        *(len(value) for value, _ in group_choices if isinstance(value, str)),
                    )
            except (TypeError, ValueError):
                # No groups, choices in the form [value, display]
                value, human_name = group_name, group_choices
                if not is_value(value) or not is_value(human_name):
                    break
                if self.max_length is not None and isinstance(value, str):
                    choice_max_length = max(choice_max_length, len(value))

            # Special case: choices=['ab']
            if isinstance(choices_group, str):
                break
        else:
            if self.max_length is not None and choice_max_length > self.max_length:
                return [
                    checks.Error(
                        "'max_length' is too small to fit the longest value "
                        "in 'choices' (%d characters)." % choice_max_length,
                        obj=self,
                        id='fields.E009',
                    ),
                ]
            return []

        return [
            checks.Error(
                "'choices' must be an iterable containing "
                "(actual value, human readable name) tuples.",
                obj=self,
                id='fields.E005',
            )
        ]
```
### 6 - django/forms/fields.py:

Start line: 820, End line: 844

```python
class TypedChoiceField(ChoiceField):
    def __init__(self, *, coerce=lambda val: val, empty_value='', **kwargs):
        self.coerce = coerce
        self.empty_value = empty_value
        super().__init__(**kwargs)

    def _coerce(self, value):
        """
        Validate that the value can be coerced to the right type (if not empty).
        """
        if value == self.empty_value or value in self.empty_values:
            return self.empty_value
        try:
            value = self.coerce(value)
        except (ValueError, TypeError, ValidationError):
            raise ValidationError(
                self.error_messages['invalid_choice'],
                code='invalid_choice',
                params={'value': value},
            )
        return value

    def clean(self, value):
        value = super().clean(value)
        return self._coerce(value)
```
### 7 - django/forms/models.py:

Start line: 1339, End line: 1366

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
### 8 - django/forms/fields.py:

Start line: 889, End line: 922

```python
class TypedMultipleChoiceField(MultipleChoiceField):
    def __init__(self, *, coerce=lambda val: val, **kwargs):
        self.coerce = coerce
        self.empty_value = kwargs.pop('empty_value', [])
        super().__init__(**kwargs)

    def _coerce(self, value):
        """
        Validate that the values are in self.choices and can be coerced to the
        right type.
        """
        if value == self.empty_value or value in self.empty_values:
            return self.empty_value
        new_value = []
        for choice in value:
            try:
                new_value.append(self.coerce(choice))
            except (ValueError, TypeError, ValidationError):
                raise ValidationError(
                    self.error_messages['invalid_choice'],
                    code='invalid_choice',
                    params={'value': choice},
                )
        return new_value

    def clean(self, value):
        value = super().clean(value)
        return self._coerce(value)

    def validate(self, value):
        if value != self.empty_value:
            super().validate(value)
        elif self.required:
            raise ValidationError(self.error_messages['required'], code='required')
```
### 9 - django/forms/models.py:

Start line: 1221, End line: 1234

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
### 10 - django/forms/models.py:

Start line: 1156, End line: 1219

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
                 **kwargs):
        if required and (initial is not None):
            self.empty_label = None
        else:
            self.empty_label = empty_label

        # Call Field instead of ChoiceField __init__() because we don't need
        # ChoiceField.__init__().
        Field.__init__(
            self, required=required, widget=widget, label=label,
            initial=initial, help_text=help_text, **kwargs
        )
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
