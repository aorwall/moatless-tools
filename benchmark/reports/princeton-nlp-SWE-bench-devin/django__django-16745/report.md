# django__django-16745

| **django/django** | `549d6ffeb6d626b023acc40c3bb2093b4b25b3d6` |
| ---- | ---- |
| **No of patches** | 2 |
| **All found context length** | 195 |
| **Any found context length** | 195 |
| **Avg pos** | 1.5 |
| **Min pos** | 1 |
| **Max pos** | 2 |
| **Top file pos** | 1 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/core/validators.py b/django/core/validators.py
--- a/django/core/validators.py
+++ b/django/core/validators.py
@@ -397,8 +397,37 @@ class StepValueValidator(BaseValidator):
     message = _("Ensure this value is a multiple of step size %(limit_value)s.")
     code = "step_size"
 
+    def __init__(self, limit_value, message=None, offset=None):
+        super().__init__(limit_value, message)
+        if offset is not None:
+            self.message = _(
+                "Ensure this value is a multiple of step size %(limit_value)s, "
+                "starting from %(offset)s, e.g. %(offset)s, %(valid_value1)s, "
+                "%(valid_value2)s, and so on."
+            )
+        self.offset = offset
+
+    def __call__(self, value):
+        if self.offset is None:
+            super().__call__(value)
+        else:
+            cleaned = self.clean(value)
+            limit_value = (
+                self.limit_value() if callable(self.limit_value) else self.limit_value
+            )
+            if self.compare(cleaned, limit_value):
+                offset = cleaned.__class__(self.offset)
+                params = {
+                    "limit_value": limit_value,
+                    "offset": offset,
+                    "valid_value1": offset + limit_value,
+                    "valid_value2": offset + 2 * limit_value,
+                }
+                raise ValidationError(self.message, code=self.code, params=params)
+
     def compare(self, a, b):
-        return not math.isclose(math.remainder(a, b), 0, abs_tol=1e-9)
+        offset = 0 if self.offset is None else self.offset
+        return not math.isclose(math.remainder(a - offset, b), 0, abs_tol=1e-9)
 
 
 @deconstructible
diff --git a/django/forms/fields.py b/django/forms/fields.py
--- a/django/forms/fields.py
+++ b/django/forms/fields.py
@@ -316,7 +316,9 @@ def __init__(self, *, max_value=None, min_value=None, step_size=None, **kwargs):
         if min_value is not None:
             self.validators.append(validators.MinValueValidator(min_value))
         if step_size is not None:
-            self.validators.append(validators.StepValueValidator(step_size))
+            self.validators.append(
+                validators.StepValueValidator(step_size, offset=min_value)
+            )
 
     def to_python(self, value):
         """

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/core/validators.py | 400 | 400 | 2 | 2 | 370
| django/forms/fields.py | 319 | 319 | 1 | 1 | 195


## Problem Statement

```
StepValueValidator does not take into account min_value
Description
	
If you define a number input with <input type="number" min=1 step=2>, client side this will only allow positive odd numbers. 
We could generate the same input in a Django form with IntegerField(min_value=1, step_size=2) and Field.localize is False, which would then use MinValueValidator and StepValueValidator.
We then get into a problem as StepValueValidator always uses 0 as the base, so step_size=2 only even numbers are allowed. This then conflicts with the client side validation, and the user cannot submit any value for the input.
I'm unsure if this is a bug or whether this is just a configuration problem, apologies if so, but the behaviour does seem to me to conflict with how min and step is handled by browsers.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 django/forms/fields.py** | 300 | 319| 195 | 195 | 9792 | 
| **-> 2 <-** | **2 django/core/validators.py** | 377 | 401| 175 | 370 | 14413 | 
| 3 | **2 django/forms/fields.py** | 424 | 445| 172 | 542 | 14413 | 
| 4 | **2 django/core/validators.py** | 472 | 525| 459 | 1001 | 14413 | 
| 5 | **2 django/core/validators.py** | 440 | 470| 230 | 1231 | 14413 | 
| 6 | **2 django/forms/fields.py** | 389 | 422| 229 | 1460 | 14413 | 
| 7 | **2 django/forms/fields.py** | 350 | 386| 253 | 1713 | 14413 | 
| 8 | **2 django/core/validators.py** | 404 | 419| 110 | 1823 | 14413 | 
| 9 | **2 django/forms/fields.py** | 321 | 347| 214 | 2037 | 14413 | 
| 10 | 3 django/contrib/postgres/validators.py | 76 | 92| 113 | 2150 | 14997 | 
| 11 | 4 django/db/models/fields/__init__.py | 2070 | 2103| 221 | 2371 | 34367 | 
| 12 | **4 django/core/validators.py** | 162 | 203| 400 | 2771 | 34367 | 
| 13 | **4 django/core/validators.py** | 342 | 374| 231 | 3002 | 34367 | 
| 14 | **4 django/core/validators.py** | 326 | 339| 114 | 3116 | 34367 | 
| 15 | 5 django/contrib/auth/password_validation.py | 99 | 126| 195 | 3311 | 36261 | 
| 16 | 5 django/db/models/fields/__init__.py | 1680 | 1709| 178 | 3489 | 36261 | 
| 17 | 5 django/db/models/fields/__init__.py | 1759 | 1789| 211 | 3700 | 36261 | 
| 18 | 6 django/contrib/postgres/forms/ranges.py | 98 | 120| 149 | 3849 | 36996 | 
| 19 | 6 django/db/models/fields/__init__.py | 1735 | 1757| 119 | 3968 | 36996 | 
| 20 | 6 django/db/models/fields/__init__.py | 2347 | 2389| 210 | 4178 | 36996 | 
| 21 | 6 django/db/models/fields/__init__.py | 1711 | 1733| 121 | 4299 | 36996 | 
| 22 | 7 django/forms/models.py | 1335 | 1359| 178 | 4477 | 49269 | 
| 23 | **7 django/core/validators.py** | 422 | 437| 110 | 4587 | 49269 | 
| 24 | 7 django/db/models/fields/__init__.py | 2044 | 2068| 147 | 4734 | 49269 | 
| 25 | 7 django/contrib/postgres/validators.py | 1 | 30| 204 | 4938 | 49269 | 
| 26 | 7 django/db/models/fields/__init__.py | 2105 | 2141| 236 | 5174 | 49269 | 
| 27 | 8 django/template/defaultfilters.py | 754 | 772| 150 | 5324 | 55906 | 
| 28 | 9 django/contrib/admin/widgets.py | 392 | 461| 373 | 5697 | 60138 | 
| 29 | 9 django/db/models/fields/__init__.py | 785 | 814| 234 | 5931 | 60138 | 
| 30 | 10 django/utils/formats.py | 172 | 190| 192 | 6123 | 62547 | 
| 31 | **10 django/core/validators.py** | 113 | 159| 474 | 6597 | 62547 | 
| 32 | 10 django/db/models/fields/__init__.py | 1791 | 1811| 132 | 6729 | 62547 | 
| 33 | 10 django/utils/formats.py | 276 | 306| 216 | 6945 | 62547 | 
| 34 | 11 django/contrib/gis/serializers/geojson.py | 1 | 45| 330 | 7275 | 63181 | 
| 35 | 12 django/db/models/lookups.py | 414 | 450| 203 | 7478 | 68735 | 
| 36 | 12 django/db/models/fields/__init__.py | 2144 | 2165| 123 | 7601 | 68735 | 
| 37 | 12 django/db/models/fields/__init__.py | 1813 | 1832| 137 | 7738 | 68735 | 
| 38 | 12 django/contrib/postgres/forms/ranges.py | 40 | 95| 369 | 8107 | 68735 | 
| 39 | **12 django/core/validators.py** | 250 | 305| 367 | 8474 | 68735 | 
| 40 | **12 django/core/validators.py** | 205 | 225| 170 | 8644 | 68735 | 
| 41 | 12 django/db/models/fields/__init__.py | 2002 | 2041| 230 | 8874 | 68735 | 
| 42 | **12 django/forms/fields.py** | 468 | 489| 144 | 9018 | 68735 | 
| 43 | **12 django/forms/fields.py** | 554 | 582| 179 | 9197 | 68735 | 
| 44 | 12 django/utils/formats.py | 218 | 240| 209 | 9406 | 68735 | 
| 45 | **12 django/forms/fields.py** | 960 | 1003| 307 | 9713 | 68735 | 
| 46 | **12 django/forms/fields.py** | 518 | 551| 225 | 9938 | 68735 | 
| 47 | **12 django/forms/fields.py** | 1314 | 1351| 199 | 10137 | 68735 | 
| 48 | **12 django/forms/fields.py** | 1112 | 1168| 458 | 10595 | 68735 | 
| 49 | 13 django/forms/widgets.py | 331 | 367| 204 | 10799 | 77139 | 
| 50 | **13 django/forms/fields.py** | 1006 | 1039| 235 | 11034 | 77139 | 
| 51 | **13 django/forms/fields.py** | 933 | 957| 177 | 11211 | 77139 | 
| 52 | 13 django/contrib/postgres/forms/ranges.py | 1 | 37| 215 | 11426 | 77139 | 
| 53 | **13 django/forms/fields.py** | 85 | 182| 804 | 12230 | 77139 | 
| 54 | 14 django/forms/formsets.py | 1 | 25| 210 | 12440 | 81450 | 
| 55 | **14 django/core/validators.py** | 1 | 16| 119 | 12559 | 81450 | 
| 56 | 15 django/conf/locale/nl/formats.py | 36 | 93| 1456 | 14015 | 83432 | 
| 57 | **15 django/forms/fields.py** | 492 | 515| 165 | 14180 | 83432 | 
| 58 | **15 django/core/validators.py** | 68 | 111| 609 | 14789 | 83432 | 
| 59 | 15 django/db/models/fields/__init__.py | 1308 | 1331| 153 | 14942 | 83432 | 
| 60 | **15 django/core/validators.py** | 19 | 65| 340 | 15282 | 83432 | 
| 61 | 15 django/forms/widgets.py | 408 | 449| 266 | 15548 | 83432 | 
| 62 | 16 django/db/models/functions/math.py | 176 | 213| 251 | 15799 | 84861 | 
| 63 | 17 django/conf/locale/sv/formats.py | 5 | 36| 480 | 16279 | 85386 | 
| 64 | 17 django/db/models/lookups.py | 396 | 411| 129 | 16408 | 85386 | 
| 65 | 17 django/contrib/auth/password_validation.py | 129 | 156| 279 | 16687 | 85386 | 
| 66 | 17 django/forms/widgets.py | 765 | 780| 122 | 16809 | 85386 | 
| 67 | 18 django/conf/locale/sr/formats.py | 5 | 45| 742 | 17551 | 86173 | 
| 68 | **18 django/forms/fields.py** | 751 | 771| 182 | 17733 | 86173 | 
| 69 | 19 django/conf/locale/sr_Latn/formats.py | 5 | 45| 741 | 18474 | 86959 | 
| 70 | 19 django/forms/models.py | 388 | 431| 400 | 18874 | 86959 | 
| 71 | 19 django/contrib/auth/password_validation.py | 179 | 214| 246 | 19120 | 86959 | 
| 72 | 20 django/conf/locale/hr/formats.py | 5 | 45| 747 | 19867 | 87751 | 
| 73 | 21 django/conf/locale/lt/formats.py | 5 | 46| 681 | 20548 | 88477 | 
| 74 | 21 django/contrib/postgres/validators.py | 33 | 73| 267 | 20815 | 88477 | 
| 75 | 21 django/forms/models.py | 1361 | 1396| 225 | 21040 | 88477 | 
| 76 | 22 django/conf/locale/sl/formats.py | 5 | 45| 713 | 21753 | 89235 | 
| 77 | 23 django/conf/locale/en_IE/formats.py | 5 | 38| 586 | 22339 | 89866 | 
| 78 | 24 django/conf/locale/ml/formats.py | 5 | 44| 631 | 22970 | 90542 | 
| 79 | 25 django/conf/locale/nb/formats.py | 5 | 42| 610 | 23580 | 91197 | 
| 80 | 26 django/conf/locale/lv/formats.py | 5 | 47| 705 | 24285 | 91947 | 
| 81 | 27 django/contrib/postgres/forms/array.py | 1 | 42| 266 | 24551 | 93573 | 
| 82 | 28 django/conf/locale/id/formats.py | 5 | 50| 678 | 25229 | 94296 | 
| 83 | 29 django/contrib/auth/migrations/0007_alter_validators_add_error_messages.py | 1 | 27| 143 | 25372 | 94439 | 
| 84 | 29 django/db/models/fields/__init__.py | 2708 | 2760| 343 | 25715 | 94439 | 
| 85 | 30 django/contrib/humanize/templatetags/humanize.py | 64 | 83| 179 | 25894 | 97438 | 
| 86 | 31 django/conf/locale/pt/formats.py | 5 | 40| 589 | 26483 | 98072 | 
| 87 | 32 django/conf/locale/nn/formats.py | 5 | 42| 610 | 27093 | 98727 | 
| 88 | 32 django/forms/models.py | 1587 | 1602| 131 | 27224 | 98727 | 
| 89 | 33 django/conf/locale/en_GB/formats.py | 5 | 42| 673 | 27897 | 99445 | 
| 90 | 34 django/db/models/functions/mixins.py | 1 | 23| 168 | 28065 | 99875 | 
| 91 | **34 django/forms/fields.py** | 448 | 465| 134 | 28199 | 99875 | 
| 92 | 35 django/contrib/gis/forms/fields.py | 67 | 96| 216 | 28415 | 100814 | 
| 93 | 35 django/forms/models.py | 1529 | 1563| 254 | 28669 | 100814 | 
| 94 | **35 django/forms/fields.py** | 1253 | 1278| 191 | 28860 | 100814 | 
| 95 | 36 django/contrib/auth/migrations/0008_alter_user_username_max_length.py | 1 | 27| 144 | 29004 | 100958 | 
| 96 | 37 django/conf/locale/cs/formats.py | 5 | 44| 609 | 29613 | 101612 | 
| 97 | 38 django/db/backends/mysql/validation.py | 1 | 36| 246 | 29859 | 102143 | 
| 98 | 39 django/conf/locale/mk/formats.py | 5 | 41| 599 | 30458 | 102787 | 
| 99 | 40 django/contrib/auth/migrations/0004_alter_user_username_opts.py | 1 | 28| 158 | 30616 | 102945 | 
| 100 | 41 django/conf/locale/en_AU/formats.py | 5 | 42| 673 | 31289 | 103663 | 
| 101 | 42 django/conf/locale/ka/formats.py | 29 | 49| 476 | 31765 | 104502 | 
| 102 | 43 django/contrib/gis/db/models/functions.py | 110 | 136| 169 | 31934 | 108674 | 
| 103 | **43 django/core/validators.py** | 572 | 611| 227 | 32161 | 108674 | 
| 104 | 44 django/conf/locale/zh_Hant/formats.py | 5 | 43| 635 | 32796 | 109354 | 
| 105 | 45 django/conf/locale/ig/formats.py | 5 | 33| 387 | 33183 | 109786 | 
| 106 | **45 django/core/validators.py** | 528 | 569| 263 | 33446 | 109786 | 
| 107 | 46 django/conf/locale/pt_BR/formats.py | 5 | 35| 468 | 33914 | 110299 | 
| 108 | 47 django/conf/locale/ms/formats.py | 5 | 39| 607 | 34521 | 110951 | 
| 109 | 48 django/conf/locale/uk/formats.py | 5 | 36| 424 | 34945 | 111420 | 
| 110 | 49 django/conf/locale/el/formats.py | 5 | 35| 460 | 35405 | 111925 | 
| 111 | **49 django/forms/fields.py** | 264 | 297| 277 | 35682 | 111925 | 
| 112 | 49 django/forms/models.py | 799 | 890| 787 | 36469 | 111925 | 
| 113 | 49 django/template/defaultfilters.py | 95 | 184| 787 | 37256 | 111925 | 
| 114 | 49 django/forms/widgets.py | 370 | 405| 279 | 37535 | 111925 | 
| 115 | 50 django/conf/locale/ru/formats.py | 5 | 31| 366 | 37901 | 112336 | 
| 116 | 51 django/conf/locale/en/formats.py | 53 | 66| 120 | 38021 | 113216 | 
| 117 | 52 django/conf/locale/uz/formats.py | 5 | 31| 416 | 38437 | 113677 | 
| 118 | 53 django/conf/locale/de_CH/formats.py | 5 | 36| 409 | 38846 | 114131 | 
| 119 | 54 django/conf/locale/da/formats.py | 5 | 27| 249 | 39095 | 114425 | 
| 120 | 55 django/conf/locale/zh_Hans/formats.py | 5 | 43| 635 | 39730 | 115105 | 
| 121 | 56 django/contrib/auth/validators.py | 1 | 26| 174 | 39904 | 115280 | 
| 122 | 56 django/forms/models.py | 470 | 500| 248 | 40152 | 115280 | 
| 123 | **56 django/forms/fields.py** | 620 | 639| 187 | 40339 | 115280 | 
| 124 | 56 django/utils/formats.py | 193 | 215| 242 | 40581 | 115280 | 
| 125 | **56 django/core/validators.py** | 308 | 323| 122 | 40703 | 115280 | 
| 126 | 57 django/conf/locale/ko/formats.py | 38 | 55| 384 | 41087 | 116189 | 
| 127 | **57 django/forms/fields.py** | 184 | 227| 290 | 41377 | 116189 | 
| 128 | 58 django/contrib/admin/checks.py | 580 | 608| 195 | 41572 | 125772 | 
| 129 | 58 django/db/backends/mysql/validation.py | 38 | 78| 291 | 41863 | 125772 | 
| 130 | 59 django/contrib/postgres/fields/ranges.py | 54 | 112| 425 | 42288 | 128251 | 
| 131 | **59 django/forms/fields.py** | 831 | 864| 241 | 42529 | 128251 | 
| 132 | 60 django/utils/translation/__init__.py | 114 | 163| 341 | 42870 | 130127 | 
| 133 | 61 django/conf/locale/it/formats.py | 5 | 44| 772 | 43642 | 130944 | 
| 134 | 61 django/contrib/humanize/templatetags/humanize.py | 145 | 167| 124 | 43766 | 130944 | 
| 135 | 61 django/db/models/fields/__init__.py | 816 | 842| 210 | 43976 | 130944 | 
| 136 | 61 django/contrib/postgres/forms/array.py | 69 | 113| 256 | 44232 | 130944 | 
| 137 | 61 django/forms/widgets.py | 528 | 570| 279 | 44511 | 130944 | 
| 138 | **61 django/forms/fields.py** | 867 | 930| 435 | 44946 | 130944 | 
| 139 | 62 django/db/migrations/questioner.py | 291 | 342| 367 | 45313 | 133640 | 
| 140 | 63 django/conf/locale/hu/formats.py | 5 | 31| 304 | 45617 | 133989 | 
| 141 | 63 django/db/models/fields/__init__.py | 2619 | 2641| 167 | 45784 | 133989 | 
| 142 | **63 django/forms/fields.py** | 1354 | 1408| 364 | 46148 | 133989 | 
| 143 | 64 django/db/models/aggregates.py | 154 | 211| 390 | 46538 | 135582 | 
| 144 | 65 django/conf/locale/cy/formats.py | 5 | 34| 531 | 47069 | 136158 | 
| 145 | 66 django/conf/locale/tg/formats.py | 5 | 33| 401 | 47470 | 136604 | 
| 146 | 66 django/db/models/fields/__init__.py | 319 | 389| 462 | 47932 | 136604 | 
| 147 | 67 django/db/models/expressions.py | 1055 | 1081| 195 | 48127 | 150076 | 
| 148 | 68 django/conf/locale/sk/formats.py | 5 | 31| 336 | 48463 | 150457 | 
| 149 | **68 django/core/validators.py** | 227 | 247| 135 | 48598 | 150457 | 
| 150 | 69 django/conf/locale/az/formats.py | 5 | 31| 363 | 48961 | 150865 | 
| 151 | 69 django/forms/models.py | 433 | 468| 243 | 49204 | 150865 | 
| 152 | 70 django/conf/locale/fi/formats.py | 5 | 37| 434 | 49638 | 151344 | 
| 153 | 71 django/conf/locale/bn/formats.py | 5 | 33| 293 | 49931 | 151681 | 
| 154 | 71 django/contrib/humanize/templatetags/humanize.py | 118 | 142| 201 | 50132 | 151681 | 
| 155 | 72 django/conf/locale/fr/formats.py | 5 | 34| 458 | 50590 | 152184 | 
| 156 | 73 django/contrib/sites/models.py | 1 | 22| 130 | 50720 | 152973 | 
| 157 | 73 django/contrib/auth/password_validation.py | 159 | 177| 179 | 50899 | 152973 | 
| 158 | 74 django/conf/locale/tk/formats.py | 5 | 33| 401 | 51300 | 153419 | 
| 159 | 75 django/conf/locale/eo/formats.py | 5 | 45| 705 | 52005 | 154169 | 
| 160 | 75 django/db/models/fields/__init__.py | 391 | 422| 239 | 52244 | 154169 | 
| 161 | 75 django/db/migrations/questioner.py | 126 | 164| 327 | 52571 | 154169 | 
| 162 | 76 django/conf/locale/de/formats.py | 5 | 30| 311 | 52882 | 154525 | 
| 163 | 77 django/contrib/auth/forms.py | 149 | 196| 298 | 53180 | 157848 | 
| 164 | **77 django/forms/fields.py** | 664 | 689| 228 | 53408 | 157848 | 
| 165 | 78 django/conf/locale/ro/formats.py | 5 | 36| 261 | 53669 | 158154 | 
| 166 | 79 django/conf/locale/ca/formats.py | 5 | 31| 279 | 53948 | 158478 | 
| 167 | 79 django/contrib/admin/widgets.py | 179 | 212| 244 | 54192 | 158478 | 
| 168 | 79 django/db/models/fields/__init__.py | 2319 | 2344| 206 | 54398 | 158478 | 
| 169 | 79 django/contrib/auth/password_validation.py | 217 | 267| 386 | 54784 | 158478 | 
| 170 | 79 django/contrib/admin/checks.py | 1326 | 1355| 176 | 54960 | 158478 | 
| 171 | 79 django/contrib/auth/forms.py | 63 | 81| 124 | 55084 | 158478 | 
| 172 | 80 django/conf/locale/es_PR/formats.py | 3 | 28| 253 | 55337 | 158747 | 
| 173 | 80 django/contrib/postgres/fields/ranges.py | 114 | 136| 168 | 55505 | 158747 | 
| 174 | 81 django/contrib/messages/storage/base.py | 155 | 179| 172 | 55677 | 159986 | 
| 175 | 82 django/conf/locale/ky/formats.py | 5 | 33| 414 | 56091 | 160445 | 
| 176 | 82 django/forms/widgets.py | 506 | 525| 208 | 56299 | 160445 | 
| 177 | 82 django/db/models/fields/__init__.py | 1900 | 1923| 187 | 56486 | 160445 | 
| 178 | 82 django/forms/models.py | 892 | 915| 202 | 56688 | 160445 | 
| 179 | 82 django/template/defaultfilters.py | 891 | 936| 379 | 57067 | 160445 | 
| 180 | 82 django/forms/widgets.py | 573 | 609| 331 | 57398 | 160445 | 
| 181 | **82 django/forms/fields.py** | 641 | 662| 175 | 57573 | 160445 | 
| 182 | 82 django/db/models/fields/__init__.py | 1430 | 1461| 230 | 57803 | 160445 | 
| 183 | 83 django/db/models/constraints.py | 93 | 134| 358 | 58161 | 163958 | 
| 184 | 83 django/contrib/postgres/fields/ranges.py | 139 | 164| 237 | 58398 | 163958 | 
| 185 | 84 django/conf/locale/es/formats.py | 5 | 31| 294 | 58692 | 164297 | 
| 186 | 84 django/forms/models.py | 502 | 532| 247 | 58939 | 164297 | 
| 187 | 84 django/contrib/postgres/fields/ranges.py | 167 | 221| 315 | 59254 | 164297 | 
| 188 | 84 django/db/models/constraints.py | 19 | 73| 456 | 59710 | 164297 | 


### Hint

```
Thanks for the report! As far as I'm aware we should pass min_value to the StepValueValidator. Bug in 3a82b5f655446f0ca89e3b6a92b100aa458f348f.
Thanks for the report. I think this is a bug. We need to consider min value also with step_size
```

## Patch

```diff
diff --git a/django/core/validators.py b/django/core/validators.py
--- a/django/core/validators.py
+++ b/django/core/validators.py
@@ -397,8 +397,37 @@ class StepValueValidator(BaseValidator):
     message = _("Ensure this value is a multiple of step size %(limit_value)s.")
     code = "step_size"
 
+    def __init__(self, limit_value, message=None, offset=None):
+        super().__init__(limit_value, message)
+        if offset is not None:
+            self.message = _(
+                "Ensure this value is a multiple of step size %(limit_value)s, "
+                "starting from %(offset)s, e.g. %(offset)s, %(valid_value1)s, "
+                "%(valid_value2)s, and so on."
+            )
+        self.offset = offset
+
+    def __call__(self, value):
+        if self.offset is None:
+            super().__call__(value)
+        else:
+            cleaned = self.clean(value)
+            limit_value = (
+                self.limit_value() if callable(self.limit_value) else self.limit_value
+            )
+            if self.compare(cleaned, limit_value):
+                offset = cleaned.__class__(self.offset)
+                params = {
+                    "limit_value": limit_value,
+                    "offset": offset,
+                    "valid_value1": offset + limit_value,
+                    "valid_value2": offset + 2 * limit_value,
+                }
+                raise ValidationError(self.message, code=self.code, params=params)
+
     def compare(self, a, b):
-        return not math.isclose(math.remainder(a, b), 0, abs_tol=1e-9)
+        offset = 0 if self.offset is None else self.offset
+        return not math.isclose(math.remainder(a - offset, b), 0, abs_tol=1e-9)
 
 
 @deconstructible
diff --git a/django/forms/fields.py b/django/forms/fields.py
--- a/django/forms/fields.py
+++ b/django/forms/fields.py
@@ -316,7 +316,9 @@ def __init__(self, *, max_value=None, min_value=None, step_size=None, **kwargs):
         if min_value is not None:
             self.validators.append(validators.MinValueValidator(min_value))
         if step_size is not None:
-            self.validators.append(validators.StepValueValidator(step_size))
+            self.validators.append(
+                validators.StepValueValidator(step_size, offset=min_value)
+            )
 
     def to_python(self, value):
         """

```

## Test Patch

```diff
diff --git a/tests/forms_tests/field_tests/test_decimalfield.py b/tests/forms_tests/field_tests/test_decimalfield.py
--- a/tests/forms_tests/field_tests/test_decimalfield.py
+++ b/tests/forms_tests/field_tests/test_decimalfield.py
@@ -152,6 +152,25 @@ def test_decimalfield_6(self):
         with self.assertRaisesMessage(ValidationError, msg):
             f.clean("1.1")
 
+    def test_decimalfield_step_size_min_value(self):
+        f = DecimalField(
+            step_size=decimal.Decimal("0.3"),
+            min_value=decimal.Decimal("-0.4"),
+        )
+        self.assertWidgetRendersTo(
+            f,
+            '<input name="f" min="-0.4" step="0.3" type="number" id="id_f" required>',
+        )
+        msg = (
+            "Ensure this value is a multiple of step size 0.3, starting from -0.4, "
+            "e.g. -0.4, -0.1, 0.2, and so on."
+        )
+        with self.assertRaisesMessage(ValidationError, msg):
+            f.clean("1")
+        self.assertEqual(f.clean("0.2"), decimal.Decimal("0.2"))
+        self.assertEqual(f.clean(2), decimal.Decimal(2))
+        self.assertEqual(f.step_size, decimal.Decimal("0.3"))
+
     def test_decimalfield_scientific(self):
         f = DecimalField(max_digits=4, decimal_places=2)
         with self.assertRaisesMessage(ValidationError, "Ensure that there are no more"):
diff --git a/tests/forms_tests/field_tests/test_floatfield.py b/tests/forms_tests/field_tests/test_floatfield.py
--- a/tests/forms_tests/field_tests/test_floatfield.py
+++ b/tests/forms_tests/field_tests/test_floatfield.py
@@ -84,6 +84,18 @@ def test_floatfield_4(self):
         self.assertEqual(-1.26, f.clean("-1.26"))
         self.assertEqual(f.step_size, 0.02)
 
+    def test_floatfield_step_size_min_value(self):
+        f = FloatField(step_size=0.02, min_value=0.01)
+        msg = (
+            "Ensure this value is a multiple of step size 0.02, starting from 0.01, "
+            "e.g. 0.01, 0.03, 0.05, and so on."
+        )
+        with self.assertRaisesMessage(ValidationError, msg):
+            f.clean("0.02")
+        self.assertEqual(f.clean("2.33"), 2.33)
+        self.assertEqual(f.clean("0.11"), 0.11)
+        self.assertEqual(f.step_size, 0.02)
+
     def test_floatfield_widget_attrs(self):
         f = FloatField(widget=NumberInput(attrs={"step": 0.01, "max": 1.0, "min": 0.0}))
         self.assertWidgetRendersTo(
diff --git a/tests/forms_tests/field_tests/test_integerfield.py b/tests/forms_tests/field_tests/test_integerfield.py
--- a/tests/forms_tests/field_tests/test_integerfield.py
+++ b/tests/forms_tests/field_tests/test_integerfield.py
@@ -126,6 +126,22 @@ def test_integerfield_6(self):
         self.assertEqual(12, f.clean("12"))
         self.assertEqual(f.step_size, 3)
 
+    def test_integerfield_step_size_min_value(self):
+        f = IntegerField(step_size=3, min_value=-1)
+        self.assertWidgetRendersTo(
+            f,
+            '<input name="f" min="-1" step="3" type="number" id="id_f" required>',
+        )
+        msg = (
+            "Ensure this value is a multiple of step size 3, starting from -1, e.g. "
+            "-1, 2, 5, and so on."
+        )
+        with self.assertRaisesMessage(ValidationError, msg):
+            f.clean("9")
+        self.assertEqual(f.clean("2"), 2)
+        self.assertEqual(f.clean("-1"), -1)
+        self.assertEqual(f.step_size, 3)
+
     def test_integerfield_localized(self):
         """
         A localized IntegerField's widget renders to a text input without any
diff --git a/tests/validators/tests.py b/tests/validators/tests.py
--- a/tests/validators/tests.py
+++ b/tests/validators/tests.py
@@ -451,11 +451,39 @@
     (StepValueValidator(3), 1, ValidationError),
     (StepValueValidator(3), 8, ValidationError),
     (StepValueValidator(3), 9, None),
+    (StepValueValidator(2), 4, None),
+    (StepValueValidator(2, offset=1), 3, None),
+    (StepValueValidator(2, offset=1), 4, ValidationError),
     (StepValueValidator(0.001), 0.55, None),
     (StepValueValidator(0.001), 0.5555, ValidationError),
+    (StepValueValidator(0.001, offset=0.0005), 0.5555, None),
+    (StepValueValidator(0.001, offset=0.0005), 0.555, ValidationError),
     (StepValueValidator(Decimal(0.02)), 0.88, None),
     (StepValueValidator(Decimal(0.02)), Decimal(0.88), None),
     (StepValueValidator(Decimal(0.02)), Decimal(0.77), ValidationError),
+    (StepValueValidator(Decimal(0.02), offset=Decimal(0.01)), Decimal(0.77), None),
+    (StepValueValidator(Decimal(2.0), offset=Decimal(0.1)), Decimal(0.1), None),
+    (
+        StepValueValidator(Decimal(0.02), offset=Decimal(0.01)),
+        Decimal(0.88),
+        ValidationError,
+    ),
+    (StepValueValidator(Decimal("1.2"), offset=Decimal("2.2")), Decimal("3.4"), None),
+    (
+        StepValueValidator(Decimal("1.2"), offset=Decimal("2.2")),
+        Decimal("1.2"),
+        ValidationError,
+    ),
+    (
+        StepValueValidator(Decimal("-1.2"), offset=Decimal("2.2")),
+        Decimal("1.1"),
+        ValidationError,
+    ),
+    (
+        StepValueValidator(Decimal("-1.2"), offset=Decimal("2.2")),
+        Decimal("1.0"),
+        None,
+    ),
     (URLValidator(EXTENDED_SCHEMES), "file://localhost/path", None),
     (URLValidator(EXTENDED_SCHEMES), "git://example.com/", None),
     (

```


## Code snippets

### 1 - django/forms/fields.py:

Start line: 300, End line: 319

```python
class IntegerField(Field):
    widget = NumberInput
    default_error_messages = {
        "invalid": _("Enter a whole number."),
    }
    re_decimal = _lazy_re_compile(r"\.0*\s*$")

    def __init__(self, *, max_value=None, min_value=None, step_size=None, **kwargs):
        self.max_value, self.min_value, self.step_size = max_value, min_value, step_size
        if kwargs.get("localize") and self.widget == NumberInput:
            # Localized number input is not well supported on most browsers
            kwargs.setdefault("widget", super().widget)
        super().__init__(**kwargs)

        if max_value is not None:
            self.validators.append(validators.MaxValueValidator(max_value))
        if min_value is not None:
            self.validators.append(validators.MinValueValidator(min_value))
        if step_size is not None:
            self.validators.append(validators.StepValueValidator(step_size))
```
### 2 - django/core/validators.py:

Start line: 377, End line: 401

```python
@deconstructible
class MaxValueValidator(BaseValidator):
    message = _("Ensure this value is less than or equal to %(limit_value)s.")
    code = "max_value"

    def compare(self, a, b):
        return a > b


@deconstructible
class MinValueValidator(BaseValidator):
    message = _("Ensure this value is greater than or equal to %(limit_value)s.")
    code = "min_value"

    def compare(self, a, b):
        return a < b


@deconstructible
class StepValueValidator(BaseValidator):
    message = _("Ensure this value is a multiple of step size %(limit_value)s.")
    code = "step_size"

    def compare(self, a, b):
        return not math.isclose(math.remainder(a, b), 0, abs_tol=1e-9)
```
### 3 - django/forms/fields.py:

Start line: 424, End line: 445

```python
class DecimalField(IntegerField):

    def validate(self, value):
        super().validate(value)
        if value in self.empty_values:
            return
        if not value.is_finite():
            raise ValidationError(
                self.error_messages["invalid"],
                code="invalid",
                params={"value": value},
            )

    def widget_attrs(self, widget):
        attrs = super().widget_attrs(widget)
        if isinstance(widget, NumberInput) and "step" not in widget.attrs:
            if self.decimal_places is not None:
                # Use exponential notation for small values since they might
                # be parsed as 0 otherwise. ref #20765
                step = str(Decimal(1).scaleb(-self.decimal_places)).lower()
            else:
                step = "any"
            attrs.setdefault("step", step)
        return attrs
```
### 4 - django/core/validators.py:

Start line: 472, End line: 525

```python
@deconstructible
class DecimalValidator:

    def __call__(self, value):
        digit_tuple, exponent = value.as_tuple()[1:]
        if exponent in {"F", "n", "N"}:
            raise ValidationError(
                self.messages["invalid"], code="invalid", params={"value": value}
            )
        if exponent >= 0:
            digits = len(digit_tuple)
            if digit_tuple != (0,):
                # A positive exponent adds that many trailing zeros.
                digits += exponent
            decimals = 0
        else:
            # If the absolute value of the negative exponent is larger than the
            # number of digits, then it's the same as the number of digits,
            # because it'll consume all of the digits in digit_tuple and then
            # add abs(exponent) - len(digit_tuple) leading zeros after the
            # decimal point.
            if abs(exponent) > len(digit_tuple):
                digits = decimals = abs(exponent)
            else:
                digits = len(digit_tuple)
                decimals = abs(exponent)
        whole_digits = digits - decimals

        if self.max_digits is not None and digits > self.max_digits:
            raise ValidationError(
                self.messages["max_digits"],
                code="max_digits",
                params={"max": self.max_digits, "value": value},
            )
        if self.decimal_places is not None and decimals > self.decimal_places:
            raise ValidationError(
                self.messages["max_decimal_places"],
                code="max_decimal_places",
                params={"max": self.decimal_places, "value": value},
            )
        if (
            self.max_digits is not None
            and self.decimal_places is not None
            and whole_digits > (self.max_digits - self.decimal_places)
        ):
            raise ValidationError(
                self.messages["max_whole_digits"],
                code="max_whole_digits",
                params={"max": (self.max_digits - self.decimal_places), "value": value},
            )

    def __eq__(self, other):
        return (
            isinstance(other, self.__class__)
            and self.max_digits == other.max_digits
            and self.decimal_places == other.decimal_places
        )
```
### 5 - django/core/validators.py:

Start line: 440, End line: 470

```python
@deconstructible
class DecimalValidator:
    """
    Validate that the input does not exceed the maximum number of digits
    expected, otherwise raise ValidationError.
    """

    messages = {
        "invalid": _("Enter a number."),
        "max_digits": ngettext_lazy(
            "Ensure that there are no more than %(max)s digit in total.",
            "Ensure that there are no more than %(max)s digits in total.",
            "max",
        ),
        "max_decimal_places": ngettext_lazy(
            "Ensure that there are no more than %(max)s decimal place.",
            "Ensure that there are no more than %(max)s decimal places.",
            "max",
        ),
        "max_whole_digits": ngettext_lazy(
            "Ensure that there are no more than %(max)s digit before the decimal "
            "point.",
            "Ensure that there are no more than %(max)s digits before the decimal "
            "point.",
            "max",
        ),
    }

    def __init__(self, max_digits, decimal_places):
        self.max_digits = max_digits
        self.decimal_places = decimal_places
```
### 6 - django/forms/fields.py:

Start line: 389, End line: 422

```python
class DecimalField(IntegerField):
    default_error_messages = {
        "invalid": _("Enter a number."),
    }

    def __init__(
        self,
        *,
        max_value=None,
        min_value=None,
        max_digits=None,
        decimal_places=None,
        **kwargs,
    ):
        self.max_digits, self.decimal_places = max_digits, decimal_places
        super().__init__(max_value=max_value, min_value=min_value, **kwargs)
        self.validators.append(validators.DecimalValidator(max_digits, decimal_places))

    def to_python(self, value):
        """
        Validate that the input is a decimal number. Return a Decimal
        instance or None for empty values. Ensure that there are no more
        than max_digits in the number and no more than decimal_places digits
        after the decimal point.
        """
        if value in self.empty_values:
            return None
        if self.localize:
            value = formats.sanitize_separators(value)
        try:
            value = Decimal(str(value))
        except DecimalException:
            raise ValidationError(self.error_messages["invalid"], code="invalid")
        return value
```
### 7 - django/forms/fields.py:

Start line: 350, End line: 386

```python
class FloatField(IntegerField):
    default_error_messages = {
        "invalid": _("Enter a number."),
    }

    def to_python(self, value):
        """
        Validate that float() can be called on the input. Return the result
        of float() or None for empty values.
        """
        value = super(IntegerField, self).to_python(value)
        if value in self.empty_values:
            return None
        if self.localize:
            value = formats.sanitize_separators(value)
        try:
            value = float(value)
        except (ValueError, TypeError):
            raise ValidationError(self.error_messages["invalid"], code="invalid")
        return value

    def validate(self, value):
        super().validate(value)
        if value in self.empty_values:
            return
        if not math.isfinite(value):
            raise ValidationError(self.error_messages["invalid"], code="invalid")

    def widget_attrs(self, widget):
        attrs = super().widget_attrs(widget)
        if isinstance(widget, NumberInput) and "step" not in widget.attrs:
            if self.step_size is not None:
                step = str(self.step_size)
            else:
                step = "any"
            attrs.setdefault("step", step)
        return attrs
```
### 8 - django/core/validators.py:

Start line: 404, End line: 419

```python
@deconstructible
class MinLengthValidator(BaseValidator):
    message = ngettext_lazy(
        "Ensure this value has at least %(limit_value)d character (it has "
        "%(show_value)d).",
        "Ensure this value has at least %(limit_value)d characters (it has "
        "%(show_value)d).",
        "limit_value",
    )
    code = "min_length"

    def compare(self, a, b):
        return a < b

    def clean(self, x):
        return len(x)
```
### 9 - django/forms/fields.py:

Start line: 321, End line: 347

```python
class IntegerField(Field):

    def to_python(self, value):
        """
        Validate that int() can be called on the input. Return the result
        of int() or None for empty values.
        """
        value = super().to_python(value)
        if value in self.empty_values:
            return None
        if self.localize:
            value = formats.sanitize_separators(value)
        # Strip trailing decimal and zeros.
        try:
            value = int(self.re_decimal.sub("", str(value)))
        except (ValueError, TypeError):
            raise ValidationError(self.error_messages["invalid"], code="invalid")
        return value

    def widget_attrs(self, widget):
        attrs = super().widget_attrs(widget)
        if isinstance(widget, NumberInput):
            if self.min_value is not None:
                attrs["min"] = self.min_value
            if self.max_value is not None:
                attrs["max"] = self.max_value
            if self.step_size is not None:
                attrs["step"] = self.step_size
        return attrs
```
### 10 - django/contrib/postgres/validators.py:

Start line: 76, End line: 92

```python
class RangeMaxValueValidator(MaxValueValidator):
    def compare(self, a, b):
        return a.upper is None or a.upper > b

    message = _(
        "Ensure that the upper bound of the range is not greater than %(limit_value)s."
    )


class RangeMinValueValidator(MinValueValidator):
    def compare(self, a, b):
        return a.lower is None or a.lower < b

    message = _(
        "Ensure that the lower bound of the range is not less than %(limit_value)s."
    )
```
### 12 - django/core/validators.py:

Start line: 162, End line: 203

```python
integer_validator = RegexValidator(
    _lazy_re_compile(r"^-?\d+\Z"),
    message=_("Enter a valid integer."),
    code="invalid",
)


def validate_integer(value):
    return integer_validator(value)


@deconstructible
class EmailValidator:
    message = _("Enter a valid email address.")
    code = "invalid"
    user_regex = _lazy_re_compile(
        # dot-atom
        r"(^[-!#$%&'*+/=?^_`{}|~0-9A-Z]+(\.[-!#$%&'*+/=?^_`{}|~0-9A-Z]+)*\Z"
        # quoted-string
        r'|^"([\001-\010\013\014\016-\037!#-\[\]-\177]|\\[\001-\011\013\014\016-\177])'
        r'*"\Z)',
        re.IGNORECASE,
    )
    domain_regex = _lazy_re_compile(
        # max length for domain name labels is 63 characters per RFC 1034
        r"((?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+)(?:[A-Z0-9-]{2,63}(?<!-))\Z",
        re.IGNORECASE,
    )
    literal_regex = _lazy_re_compile(
        # literal form, ipv4 or ipv6 address (SMTP 4.1.3)
        r"\[([A-F0-9:.]+)\]\Z",
        re.IGNORECASE,
    )
    domain_allowlist = ["localhost"]

    def __init__(self, message=None, code=None, allowlist=None):
        if message is not None:
            self.message = message
        if code is not None:
            self.code = code
        if allowlist is not None:
            self.domain_allowlist = allowlist
```
### 13 - django/core/validators.py:

Start line: 342, End line: 374

```python
@deconstructible
class BaseValidator:
    message = _("Ensure this value is %(limit_value)s (it is %(show_value)s).")
    code = "limit_value"

    def __init__(self, limit_value, message=None):
        self.limit_value = limit_value
        if message:
            self.message = message

    def __call__(self, value):
        cleaned = self.clean(value)
        limit_value = (
            self.limit_value() if callable(self.limit_value) else self.limit_value
        )
        params = {"limit_value": limit_value, "show_value": cleaned, "value": value}
        if self.compare(cleaned, limit_value):
            raise ValidationError(self.message, code=self.code, params=params)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return (
            self.limit_value == other.limit_value
            and self.message == other.message
            and self.code == other.code
        )

    def compare(self, a, b):
        return a is not b

    def clean(self, x):
        return x
```
### 14 - django/core/validators.py:

Start line: 326, End line: 339

```python
def int_list_validator(sep=",", message=None, code="invalid", allow_negative=False):
    regexp = _lazy_re_compile(
        r"^%(neg)s\d+(?:%(sep)s%(neg)s\d+)*\Z"
        % {
            "neg": "(-)?" if allow_negative else "",
            "sep": re.escape(sep),
        }
    )
    return RegexValidator(regexp, message=message, code=code)


validate_comma_separated_integer_list = int_list_validator(
    message=_("Enter only digits separated by commas."),
)
```
### 23 - django/core/validators.py:

Start line: 422, End line: 437

```python
@deconstructible
class MaxLengthValidator(BaseValidator):
    message = ngettext_lazy(
        "Ensure this value has at most %(limit_value)d character (it has "
        "%(show_value)d).",
        "Ensure this value has at most %(limit_value)d characters (it has "
        "%(show_value)d).",
        "limit_value",
    )
    code = "max_length"

    def compare(self, a, b):
        return a > b

    def clean(self, x):
        return len(x)
```
### 31 - django/core/validators.py:

Start line: 113, End line: 159

```python
@deconstructible
class URLValidator(RegexValidator):

    def __call__(self, value):
        if not isinstance(value, str):
            raise ValidationError(self.message, code=self.code, params={"value": value})
        if self.unsafe_chars.intersection(value):
            raise ValidationError(self.message, code=self.code, params={"value": value})
        # Check if the scheme is valid.
        scheme = value.split("://")[0].lower()
        if scheme not in self.schemes:
            raise ValidationError(self.message, code=self.code, params={"value": value})

        # Then check full URL
        try:
            splitted_url = urlsplit(value)
        except ValueError:
            raise ValidationError(self.message, code=self.code, params={"value": value})
        try:
            super().__call__(value)
        except ValidationError as e:
            # Trivial case failed. Try for possible IDN domain
            if value:
                scheme, netloc, path, query, fragment = splitted_url
                try:
                    netloc = punycode(netloc)  # IDN -> ACE
                except UnicodeError:  # invalid domain part
                    raise e
                url = urlunsplit((scheme, netloc, path, query, fragment))
                super().__call__(url)
            else:
                raise
        else:
            # Now verify IPv6 in the netloc part
            host_match = re.search(r"^\[(.+)\](?::[0-9]{1,5})?$", splitted_url.netloc)
            if host_match:
                potential_ip = host_match[1]
                try:
                    validate_ipv6_address(potential_ip)
                except ValidationError:
                    raise ValidationError(
                        self.message, code=self.code, params={"value": value}
                    )

        # The maximum length of a full host name is 253 characters per RFC 1034
        # section 3.1. It's defined to be 255 bytes or less, but this includes
        # one byte for the length of the name and one byte for the trailing dot
        # that's used to indicate absolute names in DNS.
        if splitted_url.hostname is None or len(splitted_url.hostname) > 253:
            raise ValidationError(self.message, code=self.code, params={"value": value})
```
### 39 - django/core/validators.py:

Start line: 250, End line: 305

```python
validate_email = EmailValidator()

slug_re = _lazy_re_compile(r"^[-a-zA-Z0-9_]+\Z")
validate_slug = RegexValidator(
    slug_re,
    # Translators: "letters" means latin letters: a-z and A-Z.
    _("Enter a valid “slug” consisting of letters, numbers, underscores or hyphens."),
    "invalid",
)

slug_unicode_re = _lazy_re_compile(r"^[-\w]+\Z")
validate_unicode_slug = RegexValidator(
    slug_unicode_re,
    _(
        "Enter a valid “slug” consisting of Unicode letters, numbers, underscores, or "
        "hyphens."
    ),
    "invalid",
)


def validate_ipv4_address(value):
    try:
        ipaddress.IPv4Address(value)
    except ValueError:
        raise ValidationError(
            _("Enter a valid IPv4 address."), code="invalid", params={"value": value}
        )


def validate_ipv6_address(value):
    if not is_valid_ipv6_address(value):
        raise ValidationError(
            _("Enter a valid IPv6 address."), code="invalid", params={"value": value}
        )


def validate_ipv46_address(value):
    try:
        validate_ipv4_address(value)
    except ValidationError:
        try:
            validate_ipv6_address(value)
        except ValidationError:
            raise ValidationError(
                _("Enter a valid IPv4 or IPv6 address."),
                code="invalid",
                params={"value": value},
            )


ip_address_validator_map = {
    "both": ([validate_ipv46_address], _("Enter a valid IPv4 or IPv6 address.")),
    "ipv4": ([validate_ipv4_address], _("Enter a valid IPv4 address.")),
    "ipv6": ([validate_ipv6_address], _("Enter a valid IPv6 address.")),
}
```
### 40 - django/core/validators.py:

Start line: 205, End line: 225

```python
@deconstructible
class EmailValidator:

    def __call__(self, value):
        if not value or "@" not in value:
            raise ValidationError(self.message, code=self.code, params={"value": value})

        user_part, domain_part = value.rsplit("@", 1)

        if not self.user_regex.match(user_part):
            raise ValidationError(self.message, code=self.code, params={"value": value})

        if domain_part not in self.domain_allowlist and not self.validate_domain_part(
            domain_part
        ):
            # Try for possible IDN domain-part
            try:
                domain_part = punycode(domain_part)
            except UnicodeError:
                pass
            else:
                if self.validate_domain_part(domain_part):
                    return
            raise ValidationError(self.message, code=self.code, params={"value": value})
```
### 42 - django/forms/fields.py:

Start line: 468, End line: 489

```python
class DateField(BaseTemporalField):
    widget = DateInput
    input_formats = formats.get_format_lazy("DATE_INPUT_FORMATS")
    default_error_messages = {
        "invalid": _("Enter a valid date."),
    }

    def to_python(self, value):
        """
        Validate that the input can be converted to a date. Return a Python
        datetime.date object.
        """
        if value in self.empty_values:
            return None
        if isinstance(value, datetime.datetime):
            return value.date()
        if isinstance(value, datetime.date):
            return value
        return super().to_python(value)

    def strptime(self, value, format):
        return datetime.datetime.strptime(value, format).date()
```
### 43 - django/forms/fields.py:

Start line: 554, End line: 582

```python
class DurationField(Field):
    default_error_messages = {
        "invalid": _("Enter a valid duration."),
        "overflow": _("The number of days must be between {min_days} and {max_days}."),
    }

    def prepare_value(self, value):
        if isinstance(value, datetime.timedelta):
            return duration_string(value)
        return value

    def to_python(self, value):
        if value in self.empty_values:
            return None
        if isinstance(value, datetime.timedelta):
            return value
        try:
            value = parse_duration(str(value))
        except OverflowError:
            raise ValidationError(
                self.error_messages["overflow"].format(
                    min_days=datetime.timedelta.min.days,
                    max_days=datetime.timedelta.max.days,
                ),
                code="overflow",
            )
        if value is None:
            raise ValidationError(self.error_messages["invalid"], code="invalid")
        return value
```
### 45 - django/forms/fields.py:

Start line: 960, End line: 1003

```python
class MultipleChoiceField(ChoiceField):
    hidden_widget = MultipleHiddenInput
    widget = SelectMultiple
    default_error_messages = {
        "invalid_choice": _(
            "Select a valid choice. %(value)s is not one of the available choices."
        ),
        "invalid_list": _("Enter a list of values."),
    }

    def to_python(self, value):
        if not value:
            return []
        elif not isinstance(value, (list, tuple)):
            raise ValidationError(
                self.error_messages["invalid_list"], code="invalid_list"
            )
        return [str(val) for val in value]

    def validate(self, value):
        """Validate that the input is a list or tuple."""
        if self.required and not value:
            raise ValidationError(self.error_messages["required"], code="required")
        # Validate that each value in the value list is in self.choices.
        for val in value:
            if not self.valid_value(val):
                raise ValidationError(
                    self.error_messages["invalid_choice"],
                    code="invalid_choice",
                    params={"value": val},
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
### 46 - django/forms/fields.py:

Start line: 518, End line: 551

```python
class DateTimeField(BaseTemporalField):
    widget = DateTimeInput
    input_formats = DateTimeFormatsIterator()
    default_error_messages = {
        "invalid": _("Enter a valid date/time."),
    }

    def prepare_value(self, value):
        if isinstance(value, datetime.datetime):
            value = to_current_timezone(value)
        return value

    def to_python(self, value):
        """
        Validate that the input can be converted to a datetime. Return a
        Python datetime.datetime object.
        """
        if value in self.empty_values:
            return None
        if isinstance(value, datetime.datetime):
            return from_current_timezone(value)
        if isinstance(value, datetime.date):
            result = datetime.datetime(value.year, value.month, value.day)
            return from_current_timezone(result)
        try:
            result = parse_datetime(value.strip())
        except ValueError:
            raise ValidationError(self.error_messages["invalid"], code="invalid")
        if not result:
            result = super().to_python(value)
        return from_current_timezone(result)

    def strptime(self, value, format):
        return datetime.datetime.strptime(value, format)
```
### 47 - django/forms/fields.py:

Start line: 1314, End line: 1351

```python
class SlugField(CharField):
    default_validators = [validators.validate_slug]

    def __init__(self, *, allow_unicode=False, **kwargs):
        self.allow_unicode = allow_unicode
        if self.allow_unicode:
            self.default_validators = [validators.validate_unicode_slug]
        super().__init__(**kwargs)


class UUIDField(CharField):
    default_error_messages = {
        "invalid": _("Enter a valid UUID."),
    }

    def prepare_value(self, value):
        if isinstance(value, uuid.UUID):
            return str(value)
        return value

    def to_python(self, value):
        value = super().to_python(value)
        if value in self.empty_values:
            return None
        if not isinstance(value, uuid.UUID):
            try:
                value = uuid.UUID(value)
            except ValueError:
                raise ValidationError(self.error_messages["invalid"], code="invalid")
        return value


class InvalidJSONInput(str):
    pass


class JSONString(str):
    pass
```
### 48 - django/forms/fields.py:

Start line: 1112, End line: 1168

```python
class MultiValueField(Field):

    def clean(self, value):
        """
        Validate every value in the given list. A value is validated against
        the corresponding Field in self.fields.

        For example, if this MultiValueField was instantiated with
        fields=(DateField(), TimeField()), clean() would call
        DateField.clean(value[0]) and TimeField.clean(value[1]).
        """
        clean_data = []
        errors = []
        if self.disabled and not isinstance(value, list):
            value = self.widget.decompress(value)
        if not value or isinstance(value, (list, tuple)):
            if not value or not [v for v in value if v not in self.empty_values]:
                if self.required:
                    raise ValidationError(
                        self.error_messages["required"], code="required"
                    )
                else:
                    return self.compress([])
        else:
            raise ValidationError(self.error_messages["invalid"], code="invalid")
        for i, field in enumerate(self.fields):
            try:
                field_value = value[i]
            except IndexError:
                field_value = None
            if field_value in self.empty_values:
                if self.require_all_fields:
                    # Raise a 'required' error if the MultiValueField is
                    # required and any field is empty.
                    if self.required:
                        raise ValidationError(
                            self.error_messages["required"], code="required"
                        )
                elif field.required:
                    # Otherwise, add an 'incomplete' error to the list of
                    # collected errors and skip field cleaning, if a required
                    # field is empty.
                    if field.error_messages["incomplete"] not in errors:
                        errors.append(field.error_messages["incomplete"])
                    continue
            try:
                clean_data.append(field.clean(field_value))
            except ValidationError as e:
                # Collect all validation errors in a single list, which we'll
                # raise at the end of clean(), rather than raising a single
                # exception for the first error we encounter. Skip duplicates.
                errors.extend(m for m in e.error_list if m not in errors)
        if errors:
            raise ValidationError(errors)

        out = self.compress(clean_data)
        self.validate(out)
        self.run_validators(out)
        return out
```
### 50 - django/forms/fields.py:

Start line: 1006, End line: 1039

```python
class TypedMultipleChoiceField(MultipleChoiceField):
    def __init__(self, *, coerce=lambda val: val, **kwargs):
        self.coerce = coerce
        self.empty_value = kwargs.pop("empty_value", [])
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
                    self.error_messages["invalid_choice"],
                    code="invalid_choice",
                    params={"value": choice},
                )
        return new_value

    def clean(self, value):
        value = super().clean(value)
        return self._coerce(value)

    def validate(self, value):
        if value != self.empty_value:
            super().validate(value)
        elif self.required:
            raise ValidationError(self.error_messages["required"], code="required")
```
### 51 - django/forms/fields.py:

Start line: 933, End line: 957

```python
class TypedChoiceField(ChoiceField):
    def __init__(self, *, coerce=lambda val: val, empty_value="", **kwargs):
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
                self.error_messages["invalid_choice"],
                code="invalid_choice",
                params={"value": value},
            )
        return value

    def clean(self, value):
        value = super().clean(value)
        return self._coerce(value)
```
### 53 - django/forms/fields.py:

Start line: 85, End line: 182

```python
class Field:
    widget = TextInput  # Default widget to use when rendering this type of Field.
    hidden_widget = (
        HiddenInput  # Default widget to use when rendering this as "hidden".
    )
    default_validators = []  # Default set of validators
    # Add an 'invalid' entry to default_error_message if you want a specific
    # field error message not raised by the field validators.
    default_error_messages = {
        "required": _("This field is required."),
    }
    empty_values = list(validators.EMPTY_VALUES)

    def __init__(
        self,
        *,
        required=True,
        widget=None,
        label=None,
        initial=None,
        help_text="",
        error_messages=None,
        show_hidden_initial=False,
        validators=(),
        localize=False,
        disabled=False,
        label_suffix=None,
        template_name=None,
    ):
        # required -- Boolean that specifies whether the field is required.
        #             True by default.
        # widget -- A Widget class, or instance of a Widget class, that should
        #           be used for this Field when displaying it. Each Field has a
        #           default Widget that it'll use if you don't specify this. In
        #           most cases, the default widget is TextInput.
        # label -- A verbose name for this field, for use in displaying this
        #          field in a form. By default, Django will use a "pretty"
        #          version of the form field name, if the Field is part of a
        #          Form.
        # initial -- A value to use in this Field's initial display. This value
        #            is *not* used as a fallback if data isn't given.
        # help_text -- An optional string to use as "help text" for this Field.
        # error_messages -- An optional dictionary to override the default
        #                   messages that the field will raise.
        # show_hidden_initial -- Boolean that specifies if it is needed to render a
        #                        hidden widget with initial value after widget.
        # validators -- List of additional validators to use
        # localize -- Boolean that specifies if the field should be localized.
        # disabled -- Boolean that specifies whether the field is disabled, that
        #             is its widget is shown in the form but not editable.
        # label_suffix -- Suffix to be added to the label. Overrides
        #                 form's label_suffix.
        self.required, self.label, self.initial = required, label, initial
        self.show_hidden_initial = show_hidden_initial
        self.help_text = help_text
        self.disabled = disabled
        self.label_suffix = label_suffix
        widget = widget or self.widget
        if isinstance(widget, type):
            widget = widget()
        else:
            widget = copy.deepcopy(widget)

        # Trigger the localization machinery if needed.
        self.localize = localize
        if self.localize:
            widget.is_localized = True

        # Let the widget know whether it should display as required.
        widget.is_required = self.required

        # Hook into self.widget_attrs() for any Field-specific HTML attributes.
        extra_attrs = self.widget_attrs(widget)
        if extra_attrs:
            widget.attrs.update(extra_attrs)

        self.widget = widget

        messages = {}
        for c in reversed(self.__class__.__mro__):
            messages.update(getattr(c, "default_error_messages", {}))
        messages.update(error_messages or {})
        self.error_messages = messages

        self.validators = [*self.default_validators, *validators]
        self.template_name = template_name

        super().__init__()

    def prepare_value(self, value):
        return value

    def to_python(self, value):
        return value

    def validate(self, value):
        if value in self.empty_values and self.required:
            raise ValidationError(self.error_messages["required"], code="required")
```
### 55 - django/core/validators.py:

Start line: 1, End line: 16

```python
import ipaddress
import math
import re
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from django.core.exceptions import ValidationError
from django.utils.deconstruct import deconstructible
from django.utils.encoding import punycode
from django.utils.ipv6 import is_valid_ipv6_address
from django.utils.regex_helper import _lazy_re_compile
from django.utils.translation import gettext_lazy as _
from django.utils.translation import ngettext_lazy

# These values, if given to validate(), will trigger the self.required check.
EMPTY_VALUES = (None, "", [], (), {})
```
### 57 - django/forms/fields.py:

Start line: 492, End line: 515

```python
class TimeField(BaseTemporalField):
    widget = TimeInput
    input_formats = formats.get_format_lazy("TIME_INPUT_FORMATS")
    default_error_messages = {"invalid": _("Enter a valid time.")}

    def to_python(self, value):
        """
        Validate that the input can be converted to a time. Return a Python
        datetime.time object.
        """
        if value in self.empty_values:
            return None
        if isinstance(value, datetime.time):
            return value
        return super().to_python(value)

    def strptime(self, value, format):
        return datetime.datetime.strptime(value, format).time()


class DateTimeFormatsIterator:
    def __iter__(self):
        yield from formats.get_format("DATETIME_INPUT_FORMATS")
        yield from formats.get_format("DATE_INPUT_FORMATS")
```
### 58 - django/core/validators.py:

Start line: 68, End line: 111

```python
@deconstructible
class URLValidator(RegexValidator):
    ul = "\u00a1-\uffff"  # Unicode letters range (must not be a raw string).

    # IP patterns
    ipv4_re = (
        r"(?:0|25[0-5]|2[0-4][0-9]|1[0-9]?[0-9]?|[1-9][0-9]?)"
        r"(?:\.(?:0|25[0-5]|2[0-4][0-9]|1[0-9]?[0-9]?|[1-9][0-9]?)){3}"
    )
    ipv6_re = r"\[[0-9a-f:.]+\]"  # (simple regex, validated later)

    # Host patterns
    hostname_re = (
        r"[a-z" + ul + r"0-9](?:[a-z" + ul + r"0-9-]{0,61}[a-z" + ul + r"0-9])?"
    )
    # Max length for domain name labels is 63 characters per RFC 1034 sec. 3.1
    domain_re = r"(?:\.(?!-)[a-z" + ul + r"0-9-]{1,63}(?<!-))*"
    tld_re = (
        r"\."  # dot
        r"(?!-)"  # can't start with a dash
        r"(?:[a-z" + ul + "-]{2,63}"  # domain label
        r"|xn--[a-z0-9]{1,59})"  # or punycode label
        r"(?<!-)"  # can't end with a dash
        r"\.?"  # may have a trailing dot
    )
    host_re = "(" + hostname_re + domain_re + tld_re + "|localhost)"

    regex = _lazy_re_compile(
        r"^(?:[a-z0-9.+-]*)://"  # scheme is validated separately
        r"(?:[^\s:@/]+(?::[^\s:@/]*)?@)?"  # user:pass authentication
        r"(?:" + ipv4_re + "|" + ipv6_re + "|" + host_re + ")"
        r"(?::[0-9]{1,5})?"  # port
        r"(?:[/?#][^\s]*)?"  # resource path
        r"\Z",
        re.IGNORECASE,
    )
    message = _("Enter a valid URL.")
    schemes = ["http", "https", "ftp", "ftps"]
    unsafe_chars = frozenset("\t\r\n")

    def __init__(self, schemes=None, **kwargs):
        super().__init__(**kwargs)
        if schemes is not None:
            self.schemes = schemes
```
### 60 - django/core/validators.py:

Start line: 19, End line: 65

```python
@deconstructible
class RegexValidator:
    regex = ""
    message = _("Enter a valid value.")
    code = "invalid"
    inverse_match = False
    flags = 0

    def __init__(
        self, regex=None, message=None, code=None, inverse_match=None, flags=None
    ):
        if regex is not None:
            self.regex = regex
        if message is not None:
            self.message = message
        if code is not None:
            self.code = code
        if inverse_match is not None:
            self.inverse_match = inverse_match
        if flags is not None:
            self.flags = flags
        if self.flags and not isinstance(self.regex, str):
            raise TypeError(
                "If the flags are set, regex must be a regular expression string."
            )

        self.regex = _lazy_re_compile(self.regex, self.flags)

    def __call__(self, value):
        """
        Validate that the input contains (or does *not* contain, if
        inverse_match is True) a match for the regular expression.
        """
        regex_matches = self.regex.search(str(value))
        invalid_input = regex_matches if self.inverse_match else not regex_matches
        if invalid_input:
            raise ValidationError(self.message, code=self.code, params={"value": value})

    def __eq__(self, other):
        return (
            isinstance(other, RegexValidator)
            and self.regex.pattern == other.regex.pattern
            and self.regex.flags == other.regex.flags
            and (self.message == other.message)
            and (self.code == other.code)
            and (self.inverse_match == other.inverse_match)
        )
```
### 68 - django/forms/fields.py:

Start line: 751, End line: 771

```python
class URLField(CharField):
    widget = URLInput
    default_error_messages = {
        "invalid": _("Enter a valid URL."),
    }
    default_validators = [validators.URLValidator()]

    def __init__(self, *, assume_scheme=None, **kwargs):
        if assume_scheme is None:
            warnings.warn(
                "The default scheme will be changed from 'http' to 'https' in Django "
                "6.0. Pass the forms.URLField.assume_scheme argument to silence this "
                "warning.",
                RemovedInDjango60Warning,
                stacklevel=2,
            )
            assume_scheme = "http"
        # RemovedInDjango60Warning: When the deprecation ends, replace with:
        # self.assume_scheme = assume_scheme or "https"
        self.assume_scheme = assume_scheme
        super().__init__(strip=True, **kwargs)
```
### 91 - django/forms/fields.py:

Start line: 448, End line: 465

```python
class BaseTemporalField(Field):
    def __init__(self, *, input_formats=None, **kwargs):
        super().__init__(**kwargs)
        if input_formats is not None:
            self.input_formats = input_formats

    def to_python(self, value):
        value = value.strip()
        # Try to strptime against each input format.
        for format in self.input_formats:
            try:
                return self.strptime(value, format)
            except (ValueError, TypeError):
                continue
        raise ValidationError(self.error_messages["invalid"], code="invalid")

    def strptime(self, value, format):
        raise NotImplementedError("Subclasses must define this method.")
```
### 94 - django/forms/fields.py:

Start line: 1253, End line: 1278

```python
class SplitDateTimeField(MultiValueField):
    widget = SplitDateTimeWidget
    hidden_widget = SplitHiddenDateTimeWidget
    default_error_messages = {
        "invalid_date": _("Enter a valid date."),
        "invalid_time": _("Enter a valid time."),
    }

    def __init__(self, *, input_date_formats=None, input_time_formats=None, **kwargs):
        errors = self.default_error_messages.copy()
        if "error_messages" in kwargs:
            errors.update(kwargs["error_messages"])
        localize = kwargs.get("localize", False)
        fields = (
            DateField(
                input_formats=input_date_formats,
                error_messages={"invalid": errors["invalid_date"]},
                localize=localize,
            ),
            TimeField(
                input_formats=input_time_formats,
                error_messages={"invalid": errors["invalid_time"]},
                localize=localize,
            ),
        )
        super().__init__(fields, **kwargs)
```
### 103 - django/core/validators.py:

Start line: 572, End line: 611

```python
def get_available_image_extensions():
    try:
        from PIL import Image
    except ImportError:
        return []
    else:
        Image.init()
        return [ext.lower()[1:] for ext in Image.EXTENSION]


def validate_image_file_extension(value):
    return FileExtensionValidator(allowed_extensions=get_available_image_extensions())(
        value
    )


@deconstructible
class ProhibitNullCharactersValidator:
    """Validate that the string doesn't contain the null character."""

    message = _("Null characters are not allowed.")
    code = "null_characters_not_allowed"

    def __init__(self, message=None, code=None):
        if message is not None:
            self.message = message
        if code is not None:
            self.code = code

    def __call__(self, value):
        if "\x00" in str(value):
            raise ValidationError(self.message, code=self.code, params={"value": value})

    def __eq__(self, other):
        return (
            isinstance(other, self.__class__)
            and self.message == other.message
            and self.code == other.code
        )
```
### 106 - django/core/validators.py:

Start line: 528, End line: 569

```python
@deconstructible
class FileExtensionValidator:
    message = _(
        "File extension “%(extension)s” is not allowed. "
        "Allowed extensions are: %(allowed_extensions)s."
    )
    code = "invalid_extension"

    def __init__(self, allowed_extensions=None, message=None, code=None):
        if allowed_extensions is not None:
            allowed_extensions = [
                allowed_extension.lower() for allowed_extension in allowed_extensions
            ]
        self.allowed_extensions = allowed_extensions
        if message is not None:
            self.message = message
        if code is not None:
            self.code = code

    def __call__(self, value):
        extension = Path(value.name).suffix[1:].lower()
        if (
            self.allowed_extensions is not None
            and extension not in self.allowed_extensions
        ):
            raise ValidationError(
                self.message,
                code=self.code,
                params={
                    "extension": extension,
                    "allowed_extensions": ", ".join(self.allowed_extensions),
                    "value": value,
                },
            )

    def __eq__(self, other):
        return (
            isinstance(other, self.__class__)
            and self.allowed_extensions == other.allowed_extensions
            and self.message == other.message
            and self.code == other.code
        )
```
### 111 - django/forms/fields.py:

Start line: 264, End line: 297

```python
class CharField(Field):
    def __init__(
        self, *, max_length=None, min_length=None, strip=True, empty_value="", **kwargs
    ):
        self.max_length = max_length
        self.min_length = min_length
        self.strip = strip
        self.empty_value = empty_value
        super().__init__(**kwargs)
        if min_length is not None:
            self.validators.append(validators.MinLengthValidator(int(min_length)))
        if max_length is not None:
            self.validators.append(validators.MaxLengthValidator(int(max_length)))
        self.validators.append(validators.ProhibitNullCharactersValidator())

    def to_python(self, value):
        """Return a string."""
        if value not in self.empty_values:
            value = str(value)
            if self.strip:
                value = value.strip()
        if value in self.empty_values:
            return self.empty_value
        return value

    def widget_attrs(self, widget):
        attrs = super().widget_attrs(widget)
        if self.max_length is not None and not widget.is_hidden:
            # The HTML attribute is maxlength, not max_length.
            attrs["maxlength"] = str(self.max_length)
        if self.min_length is not None and not widget.is_hidden:
            # The HTML attribute is minlength, not min_length.
            attrs["minlength"] = str(self.min_length)
        return attrs
```
### 123 - django/forms/fields.py:

Start line: 620, End line: 639

```python
class FileField(Field):
    widget = ClearableFileInput
    default_error_messages = {
        "invalid": _("No file was submitted. Check the encoding type on the form."),
        "missing": _("No file was submitted."),
        "empty": _("The submitted file is empty."),
        "max_length": ngettext_lazy(
            "Ensure this filename has at most %(max)d character (it has %(length)d).",
            "Ensure this filename has at most %(max)d characters (it has %(length)d).",
            "max",
        ),
        "contradiction": _(
            "Please either submit a file or check the clear checkbox, not both."
        ),
    }

    def __init__(self, *, max_length=None, allow_empty_file=False, **kwargs):
        self.max_length = max_length
        self.allow_empty_file = allow_empty_file
        super().__init__(**kwargs)
```
### 125 - django/core/validators.py:

Start line: 308, End line: 323

```python
def ip_address_validators(protocol, unpack_ipv4):
    """
    Depending on the given parameters, return the appropriate validators for
    the GenericIPAddressField.
    """
    if protocol != "both" and unpack_ipv4:
        raise ValueError(
            "You can only use `unpack_ipv4` if `protocol` is set to 'both'"
        )
    try:
        return ip_address_validator_map[protocol.lower()]
    except KeyError:
        raise ValueError(
            "The protocol '%s' is unknown. Supported: %s"
            % (protocol, list(ip_address_validator_map))
        )
```
### 127 - django/forms/fields.py:

Start line: 184, End line: 227

```python
class Field:

    def run_validators(self, value):
        if value in self.empty_values:
            return
        errors = []
        for v in self.validators:
            try:
                v(value)
            except ValidationError as e:
                if hasattr(e, "code") and e.code in self.error_messages:
                    e.message = self.error_messages[e.code]
                errors.extend(e.error_list)
        if errors:
            raise ValidationError(errors)

    def clean(self, value):
        """
        Validate the given value and return its "cleaned" value as an
        appropriate Python object. Raise ValidationError for any errors.
        """
        value = self.to_python(value)
        self.validate(value)
        self.run_validators(value)
        return value

    def bound_data(self, data, initial):
        """
        Return the value that should be shown for this field on render of a
        bound form, given the submitted POST data for the field and the initial
        data, if any.

        For most fields, this will simply be data; FileFields need to handle it
        a bit differently.
        """
        if self.disabled:
            return initial
        return data

    def widget_attrs(self, widget):
        """
        Given a Widget instance (*not* a Widget class), return a dictionary of
        any HTML attributes that should be added to the Widget, based on this
        Field.
        """
        return {}
```
### 131 - django/forms/fields.py:

Start line: 831, End line: 864

```python
class NullBooleanField(BooleanField):
    """
    A field whose valid values are None, True, and False. Clean invalid values
    to None.
    """

    widget = NullBooleanSelect

    def to_python(self, value):
        """
        Explicitly check for the string 'True' and 'False', which is what a
        hidden field will submit for True and False, for 'true' and 'false',
        which are likely to be returned by JavaScript serializations of forms,
        and for '1' and '0', which is what a RadioField will submit. Unlike
        the Booleanfield, this field must check for True because it doesn't
        use the bool() function.
        """
        if value in (True, "True", "true", "1"):
            return True
        elif value in (False, "False", "false", "0"):
            return False
        else:
            return None

    def validate(self, value):
        pass


class CallableChoiceIterator:
    def __init__(self, choices_func):
        self.choices_func = choices_func

    def __iter__(self):
        yield from self.choices_func()
```
### 138 - django/forms/fields.py:

Start line: 867, End line: 930

```python
class ChoiceField(Field):
    widget = Select
    default_error_messages = {
        "invalid_choice": _(
            "Select a valid choice. %(value)s is not one of the available choices."
        ),
    }

    def __init__(self, *, choices=(), **kwargs):
        super().__init__(**kwargs)
        if isinstance(choices, ChoicesMeta):
            choices = choices.choices
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
            return ""
        return str(value)

    def validate(self, value):
        """Validate that the input is in self.choices."""
        super().validate(value)
        if value and not self.valid_value(value):
            raise ValidationError(
                self.error_messages["invalid_choice"],
                code="invalid_choice",
                params={"value": value},
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
### 142 - django/forms/fields.py:

Start line: 1354, End line: 1408

```python
class JSONField(CharField):
    default_error_messages = {
        "invalid": _("Enter a valid JSON."),
    }
    widget = Textarea

    def __init__(self, encoder=None, decoder=None, **kwargs):
        self.encoder = encoder
        self.decoder = decoder
        super().__init__(**kwargs)

    def to_python(self, value):
        if self.disabled:
            return value
        if value in self.empty_values:
            return None
        elif isinstance(value, (list, dict, int, float, JSONString)):
            return value
        try:
            converted = json.loads(value, cls=self.decoder)
        except json.JSONDecodeError:
            raise ValidationError(
                self.error_messages["invalid"],
                code="invalid",
                params={"value": value},
            )
        if isinstance(converted, str):
            return JSONString(converted)
        else:
            return converted

    def bound_data(self, data, initial):
        if self.disabled:
            return initial
        if data is None:
            return None
        try:
            return json.loads(data, cls=self.decoder)
        except json.JSONDecodeError:
            return InvalidJSONInput(data)

    def prepare_value(self, value):
        if isinstance(value, InvalidJSONInput):
            return value
        return json.dumps(value, ensure_ascii=False, cls=self.encoder)

    def has_changed(self, initial, data):
        if super().has_changed(initial, data):
            return True
        # For purposes of seeing whether something has changed, True isn't the
        # same as 1 and the order of keys doesn't matter.
        return json.dumps(initial, sort_keys=True, cls=self.encoder) != json.dumps(
            self.to_python(data), sort_keys=True, cls=self.encoder
        )
```
### 149 - django/core/validators.py:

Start line: 227, End line: 247

```python
@deconstructible
class EmailValidator:

    def validate_domain_part(self, domain_part):
        if self.domain_regex.match(domain_part):
            return True

        literal_match = self.literal_regex.match(domain_part)
        if literal_match:
            ip_address = literal_match[1]
            try:
                validate_ipv46_address(ip_address)
                return True
            except ValidationError:
                pass
        return False

    def __eq__(self, other):
        return (
            isinstance(other, EmailValidator)
            and (self.domain_allowlist == other.domain_allowlist)
            and (self.message == other.message)
            and (self.code == other.code)
        )
```
### 164 - django/forms/fields.py:

Start line: 664, End line: 689

```python
class FileField(Field):

    def clean(self, data, initial=None):
        # If the widget got contradictory inputs, we raise a validation error
        if data is FILE_INPUT_CONTRADICTION:
            raise ValidationError(
                self.error_messages["contradiction"], code="contradiction"
            )
        # False means the field value should be cleared; further validation is
        # not needed.
        if data is False:
            if not self.required:
                return False
            # If the field is required, clearing is not possible (the widget
            # shouldn't return False data in that case anyway). False is not
            # in self.empty_value; if a False value makes it this far
            # it should be validated from here on out as None (so it will be
            # caught by the required check).
            data = None
        if not data and initial:
            return initial
        return super().clean(data)

    def bound_data(self, _, initial):
        return initial

    def has_changed(self, initial, data):
        return not self.disabled and data is not None
```
### 181 - django/forms/fields.py:

Start line: 641, End line: 662

```python
class FileField(Field):

    def to_python(self, data):
        if data in self.empty_values:
            return None

        # UploadedFile objects should have name and size attributes.
        try:
            file_name = data.name
            file_size = data.size
        except AttributeError:
            raise ValidationError(self.error_messages["invalid"], code="invalid")

        if self.max_length is not None and len(file_name) > self.max_length:
            params = {"max": self.max_length, "length": len(file_name)}
            raise ValidationError(
                self.error_messages["max_length"], code="max_length", params=params
            )
        if not file_name:
            raise ValidationError(self.error_messages["invalid"], code="invalid")
        if not self.allow_empty_file and not file_size:
            raise ValidationError(self.error_messages["empty"], code="empty")

        return data
```
