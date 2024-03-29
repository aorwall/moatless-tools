# django__django-14733

| **django/django** | `ae89daf46f83a7b39d599d289624c3377bfa4ab1` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 17803 |
| **Any found context length** | 2056 |
| **Avg pos** | 89.0 |
| **Min pos** | 6 |
| **Max pos** | 58 |
| **Top file pos** | 2 |
| **Missing snippets** | 4 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/forms/formsets.py b/django/forms/formsets.py
--- a/django/forms/formsets.py
+++ b/django/forms/formsets.py
@@ -2,7 +2,7 @@
 from django.forms import Form
 from django.forms.fields import BooleanField, IntegerField
 from django.forms.utils import ErrorList
-from django.forms.widgets import HiddenInput, NumberInput
+from django.forms.widgets import CheckboxInput, HiddenInput, NumberInput
 from django.utils.functional import cached_property
 from django.utils.html import html_safe
 from django.utils.safestring import mark_safe
@@ -55,6 +55,7 @@ class BaseFormSet:
     """
     A collection of instances of the same Form class.
     """
+    deletion_widget = CheckboxInput
     ordering_widget = NumberInput
     default_error_messages = {
         'missing_management_form': _(
@@ -283,6 +284,10 @@ def compare_ordering_key(k):
     def get_default_prefix(cls):
         return 'form'
 
+    @classmethod
+    def get_deletion_widget(cls):
+        return cls.deletion_widget
+
     @classmethod
     def get_ordering_widget(cls):
         return cls.ordering_widget
@@ -417,7 +422,11 @@ def add_fields(self, form, index):
                     widget=self.get_ordering_widget(),
                 )
         if self.can_delete and (self.can_delete_extra or index < initial_form_count):
-            form.fields[DELETION_FIELD_NAME] = BooleanField(label=_('Delete'), required=False)
+            form.fields[DELETION_FIELD_NAME] = BooleanField(
+                label=_('Delete'),
+                required=False,
+                widget=self.get_deletion_widget(),
+            )
 
     def add_prefix(self, index):
         return '%s-%s' % (self.prefix, index)

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/forms/formsets.py | 5 | 5 | 6 | 2 | 2056
| django/forms/formsets.py | 58 | 58 | 15 | 2 | 4769
| django/forms/formsets.py | 286 | 286 | 58 | 2 | 17803
| django/forms/formsets.py | 420 | 420 | 10 | 2 | 3521


## Problem Statement

```
Allow overriding of deletion widget in formsets
Description
	
In Django 3.0 ordering_widget and get_ordering_widget() were introduced (see #29956). The typical use case was to easily override the ORDER field in formsets that are updated in the frontend. For the exact same use case, I'd find it useful to see deletion_widget and get_deletion_widget() getting introduced.
Discussion ​initiated here for this feature.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 django/contrib/admin/options.py | 2102 | 2154| 451 | 451 | 18656 | 
| 2 | **2 django/forms/formsets.py** | 228 | 243| 176 | 627 | 22803 | 
| 3 | 3 django/forms/widgets.py | 1 | 41| 290 | 917 | 30913 | 
| 4 | 3 django/contrib/admin/options.py | 1976 | 2009| 345 | 1262 | 30913 | 
| 5 | 3 django/contrib/admin/options.py | 1850 | 1919| 590 | 1852 | 30913 | 
| **-> 6 <-** | **3 django/forms/formsets.py** | 1 | 25| 204 | 2056 | 30913 | 
| 7 | **3 django/forms/formsets.py** | 245 | 280| 410 | 2466 | 30913 | 
| 8 | 4 django/contrib/admin/widgets.py | 273 | 308| 382 | 2848 | 34785 | 
| 9 | 4 django/forms/widgets.py | 984 | 1018| 391 | 3239 | 34785 | 
| **-> 10 <-** | **4 django/forms/formsets.py** | 388 | 420| 282 | 3521 | 34785 | 
| 11 | 5 django/forms/forms.py | 110 | 130| 177 | 3698 | 38761 | 
| 12 | 6 django/contrib/admin/helpers.py | 390 | 406| 138 | 3836 | 42144 | 
| 13 | 6 django/contrib/admin/options.py | 1472 | 1491| 133 | 3969 | 42144 | 
| 14 | 6 django/contrib/admin/options.py | 787 | 841| 414 | 4383 | 42144 | 
| **-> 15 <-** | **6 django/forms/formsets.py** | 53 | 106| 386 | 4769 | 42144 | 
| 16 | 6 django/forms/widgets.py | 640 | 669| 238 | 5007 | 42144 | 
| 17 | 6 django/forms/widgets.py | 672 | 703| 261 | 5268 | 42144 | 
| 18 | 7 django/views/generic/edit.py | 202 | 225| 174 | 5442 | 43997 | 
| 19 | 7 django/forms/widgets.py | 448 | 466| 195 | 5637 | 43997 | 
| 20 | 7 django/forms/widgets.py | 621 | 638| 179 | 5816 | 43997 | 
| 21 | 7 django/forms/widgets.py | 820 | 849| 269 | 6085 | 43997 | 
| 22 | 7 django/contrib/admin/widgets.py | 224 | 271| 423 | 6508 | 43997 | 
| 23 | 7 django/contrib/admin/options.py | 1551 | 1637| 760 | 7268 | 43997 | 
| 24 | 7 django/forms/widgets.py | 434 | 446| 122 | 7390 | 43997 | 
| 25 | 7 django/forms/widgets.py | 199 | 281| 617 | 8007 | 43997 | 
| 26 | 7 django/forms/widgets.py | 1066 | 1089| 250 | 8257 | 43997 | 
| 27 | 8 django/forms/models.py | 635 | 652| 167 | 8424 | 55771 | 
| 28 | 8 django/contrib/admin/options.py | 843 | 868| 211 | 8635 | 55771 | 
| 29 | 8 django/contrib/admin/options.py | 1 | 96| 756 | 9391 | 55771 | 
| 30 | 9 django/forms/boundfield.py | 246 | 285| 273 | 9664 | 57959 | 
| 31 | 9 django/contrib/admin/options.py | 1493 | 1516| 319 | 9983 | 57959 | 
| 32 | 9 django/forms/models.py | 782 | 820| 314 | 10297 | 57959 | 
| 33 | 10 django/contrib/gis/forms/widgets.py | 44 | 74| 291 | 10588 | 58834 | 
| 34 | 10 django/contrib/admin/widgets.py | 311 | 323| 114 | 10702 | 58834 | 
| 35 | 10 django/contrib/admin/helpers.py | 271 | 294| 235 | 10937 | 58834 | 
| 36 | 10 django/contrib/admin/options.py | 285 | 374| 636 | 11573 | 58834 | 
| 37 | 10 django/contrib/admin/widgets.py | 49 | 70| 168 | 11741 | 58834 | 
| 38 | 10 django/contrib/admin/options.py | 774 | 785| 114 | 11855 | 58834 | 
| 39 | 10 django/forms/models.py | 927 | 959| 353 | 12208 | 58834 | 
| 40 | 11 django/contrib/admin/templatetags/admin_modify.py | 48 | 86| 391 | 12599 | 59802 | 
| 41 | 12 django/db/models/fields/__init__.py | 366 | 392| 199 | 12798 | 77949 | 
| 42 | 12 django/forms/widgets.py | 1045 | 1064| 144 | 12942 | 77949 | 
| 43 | 12 django/views/generic/edit.py | 1 | 67| 481 | 13423 | 77949 | 
| 44 | 12 django/views/generic/edit.py | 70 | 101| 269 | 13692 | 77949 | 
| 45 | 12 django/forms/widgets.py | 402 | 432| 193 | 13885 | 77949 | 
| 46 | 12 django/forms/widgets.py | 551 | 585| 277 | 14162 | 77949 | 
| 47 | 13 django/contrib/auth/forms.py | 33 | 54| 175 | 14337 | 81075 | 
| 48 | 14 django/forms/fields.py | 47 | 128| 773 | 15110 | 90496 | 
| 49 | 14 django/forms/widgets.py | 744 | 773| 219 | 15329 | 90496 | 
| 50 | 14 django/contrib/admin/options.py | 1638 | 1663| 291 | 15620 | 90496 | 
| 51 | 15 django/contrib/admin/views/main.py | 350 | 410| 508 | 16128 | 94963 | 
| 52 | 15 django/forms/widgets.py | 776 | 790| 139 | 16267 | 94963 | 
| 53 | 15 django/contrib/admin/widgets.py | 161 | 192| 243 | 16510 | 94963 | 
| 54 | 15 django/forms/widgets.py | 469 | 509| 275 | 16785 | 94963 | 
| 55 | 15 django/contrib/admin/helpers.py | 79 | 97| 152 | 16937 | 94963 | 
| 56 | 15 django/contrib/admin/widgets.py | 120 | 159| 325 | 17262 | 94963 | 
| 57 | 15 django/forms/models.py | 564 | 599| 300 | 17562 | 94963 | 
| **-> 58 <-** | **15 django/forms/formsets.py** | 282 | 314| 241 | 17803 | 94963 | 
| 59 | 15 django/contrib/admin/helpers.py | 296 | 328| 282 | 18085 | 94963 | 
| 60 | 15 django/contrib/admin/views/main.py | 1 | 44| 317 | 18402 | 94963 | 
| 61 | 15 django/contrib/admin/views/main.py | 275 | 305| 270 | 18672 | 94963 | 
| 62 | 15 django/forms/widgets.py | 936 | 982| 402 | 19074 | 94963 | 
| 63 | 16 django/contrib/postgres/forms/ranges.py | 1 | 28| 201 | 19275 | 95640 | 
| 64 | 16 django/forms/forms.py | 409 | 469| 419 | 19694 | 95640 | 
| 65 | 16 django/contrib/admin/helpers.py | 330 | 356| 171 | 19865 | 95640 | 
| 66 | 16 django/contrib/admin/widgets.py | 195 | 221| 216 | 20081 | 95640 | 
| 67 | 16 django/views/generic/edit.py | 152 | 199| 340 | 20421 | 95640 | 
| 68 | 16 django/forms/models.py | 686 | 757| 732 | 21153 | 95640 | 
| 69 | 16 django/contrib/admin/options.py | 1138 | 1183| 482 | 21635 | 95640 | 
| 70 | 16 django/forms/widgets.py | 587 | 619| 233 | 21868 | 95640 | 
| 71 | 16 django/contrib/admin/options.py | 1766 | 1848| 750 | 22618 | 95640 | 
| 72 | 16 django/forms/models.py | 214 | 283| 616 | 23234 | 95640 | 
| 73 | 16 django/forms/models.py | 866 | 892| 323 | 23557 | 95640 | 
| 74 | 17 django/db/models/deletion.py | 78 | 96| 199 | 23756 | 99470 | 
| 75 | 17 django/contrib/admin/views/main.py | 307 | 348| 388 | 24144 | 99470 | 
| 76 | 17 django/contrib/admin/helpers.py | 35 | 76| 282 | 24426 | 99470 | 
| 77 | 17 django/forms/models.py | 961 | 994| 367 | 24793 | 99470 | 
| 78 | 18 django/db/models/fields/proxy.py | 1 | 19| 117 | 24910 | 99587 | 
| 79 | **18 django/forms/formsets.py** | 108 | 121| 126 | 25036 | 99587 | 
| 80 | 19 django/contrib/gis/admin/options.py | 52 | 63| 139 | 25175 | 100783 | 
| 81 | 19 django/db/models/deletion.py | 122 | 162| 339 | 25514 | 100783 | 
| 82 | 20 django/db/migrations/operations/models.py | 553 | 578| 167 | 25681 | 107233 | 
| 83 | 20 django/forms/forms.py | 52 | 108| 525 | 26206 | 107233 | 
| 84 | 20 django/forms/models.py | 201 | 211| 131 | 26337 | 107233 | 
| 85 | 21 django/contrib/postgres/forms/array.py | 133 | 165| 265 | 26602 | 108827 | 
| 86 | 22 django/views/generic/list.py | 50 | 75| 244 | 26846 | 110399 | 
| 87 | 22 django/forms/models.py | 316 | 355| 387 | 27233 | 110399 | 
| 88 | 22 django/forms/models.py | 601 | 633| 270 | 27503 | 110399 | 
| 89 | 22 django/contrib/admin/options.py | 2183 | 2218| 315 | 27818 | 110399 | 
| 90 | 22 django/forms/forms.py | 393 | 407| 135 | 27953 | 110399 | 
| 91 | 22 django/forms/models.py | 895 | 925| 289 | 28242 | 110399 | 
| 92 | 22 django/contrib/admin/widgets.py | 347 | 373| 328 | 28570 | 110399 | 
| 93 | 22 django/contrib/admin/widgets.py | 73 | 89| 145 | 28715 | 110399 | 
| 94 | 22 django/forms/models.py | 389 | 417| 240 | 28955 | 110399 | 
| 95 | 22 django/contrib/admin/helpers.py | 409 | 432| 195 | 29150 | 110399 | 
| 96 | 23 django/contrib/admin/checks.py | 571 | 606| 304 | 29454 | 119581 | 
| 97 | 23 django/contrib/admin/options.py | 206 | 217| 135 | 29589 | 119581 | 
| 98 | 23 django/db/models/deletion.py | 1 | 75| 561 | 30150 | 119581 | 
| 99 | 23 django/forms/widgets.py | 512 | 548| 331 | 30481 | 119581 | 
| 100 | 23 django/forms/widgets.py | 924 | 933| 107 | 30588 | 119581 | 
| 101 | 23 django/forms/widgets.py | 706 | 741| 237 | 30825 | 119581 | 
| 102 | 24 django/contrib/admin/actions.py | 1 | 81| 616 | 31441 | 120197 | 
| 103 | 24 django/contrib/admin/helpers.py | 247 | 269| 211 | 31652 | 120197 | 
| 104 | 24 django/forms/widgets.py | 114 | 157| 367 | 32019 | 120197 | 
| 105 | **24 django/forms/formsets.py** | 28 | 50| 259 | 32278 | 120197 | 
| 106 | **24 django/forms/formsets.py** | 197 | 226| 217 | 32495 | 120197 | 
| 107 | 24 django/forms/fields.py | 130 | 173| 290 | 32785 | 120197 | 
| 108 | 24 django/contrib/postgres/forms/array.py | 105 | 131| 219 | 33004 | 120197 | 
| 109 | 24 django/forms/widgets.py | 793 | 818| 214 | 33218 | 120197 | 
| 110 | 24 django/contrib/admin/options.py | 2067 | 2100| 376 | 33594 | 120197 | 
| 111 | 24 django/db/migrations/operations/models.py | 580 | 596| 215 | 33809 | 120197 | 
| 112 | **24 django/forms/formsets.py** | 330 | 386| 472 | 34281 | 120197 | 
| 113 | 24 django/contrib/admin/options.py | 99 | 129| 223 | 34504 | 120197 | 
| 114 | 24 django/forms/boundfield.py | 80 | 98| 174 | 34678 | 120197 | 
| 115 | 24 django/contrib/admin/widgets.py | 326 | 344| 168 | 34846 | 120197 | 
| 116 | 24 django/forms/widgets.py | 379 | 399| 138 | 34984 | 120197 | 
| 117 | 24 django/forms/widgets.py | 851 | 894| 315 | 35299 | 120197 | 
| 118 | 24 django/contrib/admin/widgets.py | 449 | 477| 203 | 35502 | 120197 | 
| 119 | 24 django/forms/widgets.py | 303 | 339| 204 | 35706 | 120197 | 
| 120 | 25 django/contrib/auth/admin.py | 40 | 99| 504 | 36210 | 121934 | 
| 121 | 25 django/contrib/admin/helpers.py | 359 | 377| 181 | 36391 | 121934 | 
| 122 | 26 django/contrib/contenttypes/admin.py | 81 | 128| 410 | 36801 | 122915 | 
| 123 | 26 django/forms/boundfield.py | 36 | 51| 149 | 36950 | 122915 | 
| 124 | 26 django/contrib/admin/widgets.py | 376 | 393| 129 | 37079 | 122915 | 
| 125 | 26 django/contrib/admin/options.py | 675 | 721| 449 | 37528 | 122915 | 
| 126 | 26 django/forms/models.py | 1 | 27| 218 | 37746 | 122915 | 
| 127 | 27 django/contrib/admin/utils.py | 123 | 158| 303 | 38049 | 127073 | 
| 128 | 27 django/contrib/auth/forms.py | 57 | 75| 124 | 38173 | 127073 | 
| 129 | 27 django/forms/widgets.py | 897 | 921| 172 | 38345 | 127073 | 
| 130 | 27 django/contrib/admin/widgets.py | 1 | 46| 330 | 38675 | 127073 | 
| 131 | 27 django/forms/widgets.py | 1020 | 1043| 233 | 38908 | 127073 | 
| 132 | 27 django/forms/fields.py | 775 | 834| 416 | 39324 | 127073 | 
| 133 | 28 django/contrib/contenttypes/forms.py | 1 | 49| 401 | 39725 | 127860 | 
| 134 | 28 django/forms/forms.py | 207 | 283| 637 | 40362 | 127860 | 
| 135 | 28 django/forms/widgets.py | 342 | 376| 278 | 40640 | 127860 | 
| 136 | **28 django/forms/formsets.py** | 142 | 167| 201 | 40841 | 127860 | 
| 137 | 28 django/forms/fields.py | 584 | 609| 243 | 41084 | 127860 | 
| 138 | 28 django/forms/forms.py | 471 | 494| 247 | 41331 | 127860 | 
| 139 | 28 django/contrib/admin/views/main.py | 412 | 450| 334 | 41665 | 127860 | 
| 140 | 28 django/contrib/postgres/forms/ranges.py | 31 | 78| 325 | 41990 | 127860 | 
| 141 | 28 django/forms/fields.py | 545 | 561| 180 | 42170 | 127860 | 
| 142 | 28 django/db/models/deletion.py | 381 | 450| 580 | 42750 | 127860 | 
| 143 | 28 django/db/models/deletion.py | 363 | 379| 130 | 42880 | 127860 | 
| 144 | 28 django/contrib/admin/options.py | 1518 | 1549| 295 | 43175 | 127860 | 
| 145 | 29 django/db/models/sql/subqueries.py | 1 | 45| 309 | 43484 | 129048 | 
| 146 | 30 django/db/models/base.py | 2127 | 2178| 351 | 43835 | 146378 | 
| 147 | 30 django/forms/widgets.py | 284 | 300| 127 | 43962 | 146378 | 
| 148 | 30 django/contrib/admin/options.py | 499 | 512| 165 | 44127 | 146378 | 
| 149 | 31 django/contrib/contenttypes/fields.py | 631 | 655| 214 | 44341 | 151836 | 
| 150 | 31 django/contrib/admin/helpers.py | 131 | 157| 220 | 44561 | 151836 | 
| 151 | 31 django/contrib/admin/views/main.py | 452 | 507| 463 | 45024 | 151836 | 
| 152 | 31 django/forms/forms.py | 1 | 20| 122 | 45146 | 151836 | 
| 153 | 32 django/contrib/auth/management/commands/createsuperuser.py | 204 | 228| 204 | 45350 | 153899 | 
| 154 | 33 django/contrib/auth/views.py | 337 | 369| 239 | 45589 | 156629 | 
| 155 | 33 django/forms/models.py | 654 | 684| 217 | 45806 | 156629 | 
| 156 | 33 django/contrib/admin/options.py | 1921 | 1974| 433 | 46239 | 156629 | 
| 157 | 33 django/forms/widgets.py | 160 | 196| 228 | 46467 | 156629 | 
| 158 | 33 django/db/migrations/operations/models.py | 598 | 615| 163 | 46630 | 156629 | 
| 159 | 34 django/db/migrations/autodetector.py | 804 | 854| 576 | 47206 | 168209 | 
| 160 | 34 django/db/models/base.py | 1773 | 1873| 729 | 47935 | 168209 | 
| 161 | 35 django/db/models/sql/query.py | 1975 | 1987| 129 | 48064 | 190572 | 
| 162 | 35 django/forms/boundfield.py | 1 | 34| 246 | 48310 | 190572 | 
| 163 | **35 django/forms/formsets.py** | 169 | 195| 240 | 48550 | 190572 | 
| 164 | 35 django/contrib/admin/options.py | 131 | 186| 604 | 49154 | 190572 | 
| 165 | 35 django/contrib/admin/options.py | 219 | 241| 230 | 49384 | 190572 | 
| 166 | 35 django/forms/models.py | 286 | 314| 288 | 49672 | 190572 | 
| 167 | 35 django/contrib/admin/options.py | 2012 | 2065| 454 | 50126 | 190572 | 
| 168 | 35 django/contrib/auth/forms.py | 138 | 160| 179 | 50305 | 190572 | 
| 169 | 35 django/forms/models.py | 357 | 387| 233 | 50538 | 190572 | 
| 170 | 35 django/forms/models.py | 759 | 780| 194 | 50732 | 190572 | 
| 171 | 36 django/contrib/flatpages/forms.py | 53 | 71| 142 | 50874 | 191059 | 
| 172 | 36 django/db/models/fields/__init__.py | 524 | 541| 182 | 51056 | 191059 | 


### Hint

```
Thanks, sounds reasonable.
​PR
```

## Patch

```diff
diff --git a/django/forms/formsets.py b/django/forms/formsets.py
--- a/django/forms/formsets.py
+++ b/django/forms/formsets.py
@@ -2,7 +2,7 @@
 from django.forms import Form
 from django.forms.fields import BooleanField, IntegerField
 from django.forms.utils import ErrorList
-from django.forms.widgets import HiddenInput, NumberInput
+from django.forms.widgets import CheckboxInput, HiddenInput, NumberInput
 from django.utils.functional import cached_property
 from django.utils.html import html_safe
 from django.utils.safestring import mark_safe
@@ -55,6 +55,7 @@ class BaseFormSet:
     """
     A collection of instances of the same Form class.
     """
+    deletion_widget = CheckboxInput
     ordering_widget = NumberInput
     default_error_messages = {
         'missing_management_form': _(
@@ -283,6 +284,10 @@ def compare_ordering_key(k):
     def get_default_prefix(cls):
         return 'form'
 
+    @classmethod
+    def get_deletion_widget(cls):
+        return cls.deletion_widget
+
     @classmethod
     def get_ordering_widget(cls):
         return cls.ordering_widget
@@ -417,7 +422,11 @@ def add_fields(self, form, index):
                     widget=self.get_ordering_widget(),
                 )
         if self.can_delete and (self.can_delete_extra or index < initial_form_count):
-            form.fields[DELETION_FIELD_NAME] = BooleanField(label=_('Delete'), required=False)
+            form.fields[DELETION_FIELD_NAME] = BooleanField(
+                label=_('Delete'),
+                required=False,
+                widget=self.get_deletion_widget(),
+            )
 
     def add_prefix(self, index):
         return '%s-%s' % (self.prefix, index)

```

## Test Patch

```diff
diff --git a/tests/forms_tests/tests/test_formsets.py b/tests/forms_tests/tests/test_formsets.py
--- a/tests/forms_tests/tests/test_formsets.py
+++ b/tests/forms_tests/tests/test_formsets.py
@@ -551,6 +551,38 @@ def test_formset_with_deletion_invalid_deleted_form(self):
         self.assertEqual(formset._errors, [])
         self.assertEqual(len(formset.deleted_forms), 1)
 
+    def test_formset_with_deletion_custom_widget(self):
+        class DeletionAttributeFormSet(BaseFormSet):
+            deletion_widget = HiddenInput
+
+        class DeletionMethodFormSet(BaseFormSet):
+            def get_deletion_widget(self):
+                return HiddenInput(attrs={'class': 'deletion'})
+
+        tests = [
+            (DeletionAttributeFormSet, '<input type="hidden" name="form-0-DELETE">'),
+            (
+                DeletionMethodFormSet,
+                '<input class="deletion" type="hidden" name="form-0-DELETE">',
+            ),
+        ]
+        for formset_class, delete_html in tests:
+            with self.subTest(formset_class=formset_class.__name__):
+                ArticleFormSet = formset_factory(
+                    ArticleForm,
+                    formset=formset_class,
+                    can_delete=True,
+                )
+                formset = ArticleFormSet(auto_id=False)
+                self.assertHTMLEqual(
+                    '\n'.join([form.as_ul() for form in formset.forms]),
+                    (
+                        f'<li>Title: <input type="text" name="form-0-title"></li>'
+                        f'<li>Pub date: <input type="text" name="form-0-pub_date">'
+                        f'{delete_html}</li>'
+                    ),
+                )
+
     def test_formsets_with_ordering(self):
         """
         formset_factory's can_order argument adds an integer field to each
@@ -602,8 +634,8 @@ def test_formsets_with_ordering(self):
             ],
         )
 
-    def test_formsets_with_order_custom_widget(self):
-        class OrderingAttributFormSet(BaseFormSet):
+    def test_formsets_with_ordering_custom_widget(self):
+        class OrderingAttributeFormSet(BaseFormSet):
             ordering_widget = HiddenInput
 
         class OrderingMethodFormSet(BaseFormSet):
@@ -611,7 +643,7 @@ def get_ordering_widget(self):
                 return HiddenInput(attrs={'class': 'ordering'})
 
         tests = (
-            (OrderingAttributFormSet, '<input type="hidden" name="form-0-ORDER">'),
+            (OrderingAttributeFormSet, '<input type="hidden" name="form-0-ORDER">'),
             (OrderingMethodFormSet, '<input class="ordering" type="hidden" name="form-0-ORDER">'),
         )
         for formset_class, order_html in tests:

```


## Code snippets

### 1 - django/contrib/admin/options.py:

Start line: 2102, End line: 2154

```python
class InlineModelAdmin(BaseModelAdmin):

    def get_formset(self, request, obj=None, **kwargs):
        # ... other code

        class DeleteProtectedModelForm(base_model_form):

            def hand_clean_DELETE(self):
                """
                We don't validate the 'DELETE' field itself because on
                templates it's not rendered using the field information, but
                just using a generic "deletion_field" of the InlineModelAdmin.
                """
                if self.cleaned_data.get(DELETION_FIELD_NAME, False):
                    using = router.db_for_write(self._meta.model)
                    collector = NestedObjects(using=using)
                    if self.instance._state.adding:
                        return
                    collector.collect([self.instance])
                    if collector.protected:
                        objs = []
                        for p in collector.protected:
                            objs.append(
                                # Translators: Model verbose name and instance representation,
                                # suitable to be an item in a list.
                                _('%(class_name)s %(instance)s') % {
                                    'class_name': p._meta.verbose_name,
                                    'instance': p}
                            )
                        params = {
                            'class_name': self._meta.model._meta.verbose_name,
                            'instance': self.instance,
                            'related_objects': get_text_list(objs, _('and')),
                        }
                        msg = _("Deleting %(class_name)s %(instance)s would require "
                                "deleting the following protected related objects: "
                                "%(related_objects)s")
                        raise ValidationError(msg, code='deleting_protected', params=params)

            def is_valid(self):
                result = super().is_valid()
                self.hand_clean_DELETE()
                return result

            def has_changed(self):
                # Protect against unauthorized edits.
                if not can_change and not self.instance._state.adding:
                    return False
                if not can_add and self.instance._state.adding:
                    return False
                return super().has_changed()

        defaults['form'] = DeleteProtectedModelForm

        if defaults['fields'] is None and not modelform_defines_fields(defaults['form']):
            defaults['fields'] = forms.ALL_FIELDS

        return inlineformset_factory(self.parent_model, self.model, **defaults)
```
### 2 - django/forms/formsets.py:

Start line: 228, End line: 243

```python
@html_safe
class BaseFormSet:

    @property
    def deleted_forms(self):
        """Return a list of forms that have been marked for deletion."""
        if not self.is_valid() or not self.can_delete:
            return []
        # construct _deleted_form_indexes which is just a list of form indexes
        # that have had their deletion widget set to True
        if not hasattr(self, '_deleted_form_indexes'):
            self._deleted_form_indexes = []
            for i, form in enumerate(self.forms):
                # if this is an extra form and hasn't changed, don't consider it
                if i >= self.initial_form_count() and not form.has_changed():
                    continue
                if self._should_delete_form(form):
                    self._deleted_form_indexes.append(i)
        return [self.forms[i] for i in self._deleted_form_indexes]
```
### 3 - django/forms/widgets.py:

Start line: 1, End line: 41

```python
"""
HTML Widget classes
"""

import copy
import datetime
import warnings
from collections import defaultdict
from itertools import chain

from django.forms.utils import to_current_timezone
from django.templatetags.static import static
from django.utils import formats
from django.utils.datastructures import OrderedSet
from django.utils.dates import MONTHS
from django.utils.formats import get_format
from django.utils.html import format_html, html_safe
from django.utils.regex_helper import _lazy_re_compile
from django.utils.safestring import mark_safe
from django.utils.topological_sort import (
    CyclicDependencyError, stable_topological_sort,
)
from django.utils.translation import gettext_lazy as _

from .renderers import get_default_renderer

__all__ = (
    'Media', 'MediaDefiningClass', 'Widget', 'TextInput', 'NumberInput',
    'EmailInput', 'URLInput', 'PasswordInput', 'HiddenInput',
    'MultipleHiddenInput', 'FileInput', 'ClearableFileInput', 'Textarea',
    'DateInput', 'DateTimeInput', 'TimeInput', 'CheckboxInput', 'Select',
    'NullBooleanSelect', 'SelectMultiple', 'RadioSelect',
    'CheckboxSelectMultiple', 'MultiWidget', 'SplitDateTimeWidget',
    'SplitHiddenDateTimeWidget', 'SelectDateWidget',
)

MEDIA_TYPES = ('css', 'js')


class MediaOrderConflictWarning(RuntimeWarning):
    pass
```
### 4 - django/contrib/admin/options.py:

Start line: 1976, End line: 2009

```python
class ModelAdmin(BaseModelAdmin):

    def _create_formsets(self, request, obj, change):
        "Helper function to generate formsets for add/change_view."
        formsets = []
        inline_instances = []
        prefixes = {}
        get_formsets_args = [request]
        if change:
            get_formsets_args.append(obj)
        for FormSet, inline in self.get_formsets_with_inlines(*get_formsets_args):
            prefix = FormSet.get_default_prefix()
            prefixes[prefix] = prefixes.get(prefix, 0) + 1
            if prefixes[prefix] != 1 or not prefix:
                prefix = "%s-%s" % (prefix, prefixes[prefix])
            formset_params = self.get_formset_kwargs(request, obj, inline, prefix)
            formset = FormSet(**formset_params)

            def user_deleted_form(request, obj, formset, index):
                """Return whether or not the user deleted the form."""
                return (
                    inline.has_delete_permission(request, obj) and
                    '{}-{}-DELETE'.format(formset.prefix, index) in request.POST
                )

            # Bypass validation of each view-only inline form (since the form's
            # data won't be in request.POST), unless the form was deleted.
            if not inline.has_change_permission(request, obj if change else None):
                for index, form in enumerate(formset.initial_forms):
                    if user_deleted_form(request, obj, formset, index):
                        continue
                    form._errors = {}
                    form.cleaned_data = form.initial
            formsets.append(formset)
            inline_instances.append(inline)
        return formsets, inline_instances
```
### 5 - django/contrib/admin/options.py:

Start line: 1850, End line: 1919

```python
class ModelAdmin(BaseModelAdmin):

    def get_deleted_objects(self, objs, request):
        """
        Hook for customizing the delete process for the delete view and the
        "delete selected" action.
        """
        return get_deleted_objects(objs, request, self.admin_site)

    @csrf_protect_m
    def delete_view(self, request, object_id, extra_context=None):
        with transaction.atomic(using=router.db_for_write(self.model)):
            return self._delete_view(request, object_id, extra_context)

    def _delete_view(self, request, object_id, extra_context):
        "The 'delete' admin view for this model."
        opts = self.model._meta
        app_label = opts.app_label

        to_field = request.POST.get(TO_FIELD_VAR, request.GET.get(TO_FIELD_VAR))
        if to_field and not self.to_field_allowed(request, to_field):
            raise DisallowedModelAdminToField("The field %s cannot be referenced." % to_field)

        obj = self.get_object(request, unquote(object_id), to_field)

        if not self.has_delete_permission(request, obj):
            raise PermissionDenied

        if obj is None:
            return self._get_obj_does_not_exist_redirect(request, opts, object_id)

        # Populate deleted_objects, a data structure of all related objects that
        # will also be deleted.
        deleted_objects, model_count, perms_needed, protected = self.get_deleted_objects([obj], request)

        if request.POST and not protected:  # The user has confirmed the deletion.
            if perms_needed:
                raise PermissionDenied
            obj_display = str(obj)
            attr = str(to_field) if to_field else opts.pk.attname
            obj_id = obj.serializable_value(attr)
            self.log_deletion(request, obj, obj_display)
            self.delete_model(request, obj)

            return self.response_delete(request, obj_display, obj_id)

        object_name = str(opts.verbose_name)

        if perms_needed or protected:
            title = _("Cannot delete %(name)s") % {"name": object_name}
        else:
            title = _("Are you sure?")

        context = {
            **self.admin_site.each_context(request),
            'title': title,
            'subtitle': None,
            'object_name': object_name,
            'object': obj,
            'deleted_objects': deleted_objects,
            'model_count': dict(model_count).items(),
            'perms_lacking': perms_needed,
            'protected': protected,
            'opts': opts,
            'app_label': app_label,
            'preserved_filters': self.get_preserved_filters(request),
            'is_popup': IS_POPUP_VAR in request.POST or IS_POPUP_VAR in request.GET,
            'to_field': to_field,
            **(extra_context or {}),
        }

        return self.render_delete_form(request, context)
```
### 6 - django/forms/formsets.py:

Start line: 1, End line: 25

```python
from django.core.exceptions import ValidationError
from django.forms import Form
from django.forms.fields import BooleanField, IntegerField
from django.forms.utils import ErrorList
from django.forms.widgets import HiddenInput, NumberInput
from django.utils.functional import cached_property
from django.utils.html import html_safe
from django.utils.safestring import mark_safe
from django.utils.translation import gettext_lazy as _, ngettext

__all__ = ('BaseFormSet', 'formset_factory', 'all_valid')

# special field names
TOTAL_FORM_COUNT = 'TOTAL_FORMS'
INITIAL_FORM_COUNT = 'INITIAL_FORMS'
MIN_NUM_FORM_COUNT = 'MIN_NUM_FORMS'
MAX_NUM_FORM_COUNT = 'MAX_NUM_FORMS'
ORDERING_FIELD_NAME = 'ORDER'
DELETION_FIELD_NAME = 'DELETE'

# default minimum number of forms in a formset
DEFAULT_MIN_NUM = 0

# default maximum number of forms in a formset, to prevent memory exhaustion
DEFAULT_MAX_NUM = 1000
```
### 7 - django/forms/formsets.py:

Start line: 245, End line: 280

```python
@html_safe
class BaseFormSet:

    @property
    def ordered_forms(self):
        """
        Return a list of form in the order specified by the incoming data.
        Raise an AttributeError if ordering is not allowed.
        """
        if not self.is_valid() or not self.can_order:
            raise AttributeError("'%s' object has no attribute 'ordered_forms'" % self.__class__.__name__)
        # Construct _ordering, which is a list of (form_index, order_field_value)
        # tuples. After constructing this list, we'll sort it by order_field_value
        # so we have a way to get to the form indexes in the order specified
        # by the form data.
        if not hasattr(self, '_ordering'):
            self._ordering = []
            for i, form in enumerate(self.forms):
                # if this is an extra form and hasn't changed, don't consider it
                if i >= self.initial_form_count() and not form.has_changed():
                    continue
                # don't add data marked for deletion to self.ordered_data
                if self.can_delete and self._should_delete_form(form):
                    continue
                self._ordering.append((i, form.cleaned_data[ORDERING_FIELD_NAME]))
            # After we're done populating self._ordering, sort it.
            # A sort function to order things numerically ascending, but
            # None should be sorted below anything else. Allowing None as
            # a comparison value makes it so we can leave ordering fields
            # blank.

            def compare_ordering_key(k):
                if k[1] is None:
                    return (1, 0)  # +infinity, larger than any number
                return (0, k[1])
            self._ordering.sort(key=compare_ordering_key)
        # Return a list of form.cleaned_data dicts in the order specified by
        # the form data.
        return [self.forms[i[0]] for i in self._ordering]
```
### 8 - django/contrib/admin/widgets.py:

Start line: 273, End line: 308

```python
class RelatedFieldWidgetWrapper(forms.Widget):

    def get_context(self, name, value, attrs):
        from django.contrib.admin.views.main import IS_POPUP_VAR, TO_FIELD_VAR
        rel_opts = self.rel.model._meta
        info = (rel_opts.app_label, rel_opts.model_name)
        self.widget.choices = self.choices
        url_params = '&'.join("%s=%s" % param for param in [
            (TO_FIELD_VAR, self.rel.get_related_field().name),
            (IS_POPUP_VAR, 1),
        ])
        context = {
            'rendered_widget': self.widget.render(name, value, attrs),
            'is_hidden': self.is_hidden,
            'name': name,
            'url_params': url_params,
            'model': rel_opts.verbose_name,
            'can_add_related': self.can_add_related,
            'can_change_related': self.can_change_related,
            'can_delete_related': self.can_delete_related,
            'can_view_related': self.can_view_related,
        }
        if self.can_add_related:
            context['add_related_url'] = self.get_related_url(info, 'add')
        if self.can_delete_related:
            context['delete_related_template_url'] = self.get_related_url(info, 'delete', '__fk__')
        if self.can_view_related or self.can_change_related:
            context['change_related_template_url'] = self.get_related_url(info, 'change', '__fk__')
        return context

    def value_from_datadict(self, data, files, name):
        return self.widget.value_from_datadict(data, files, name)

    def value_omitted_from_data(self, data, files, name):
        return self.widget.value_omitted_from_data(data, files, name)

    def id_for_label(self, id_):
        return self.widget.id_for_label(id_)
```
### 9 - django/forms/widgets.py:

Start line: 984, End line: 1018

```python
class SelectDateWidget(Widget):

    def get_context(self, name, value, attrs):
        context = super().get_context(name, value, attrs)
        date_context = {}
        year_choices = [(i, str(i)) for i in self.years]
        if not self.is_required:
            year_choices.insert(0, self.year_none_value)
        year_name = self.year_field % name
        date_context['year'] = self.select_widget(attrs, choices=year_choices).get_context(
            name=year_name,
            value=context['widget']['value']['year'],
            attrs={**context['widget']['attrs'], 'id': 'id_%s' % year_name},
        )
        month_choices = list(self.months.items())
        if not self.is_required:
            month_choices.insert(0, self.month_none_value)
        month_name = self.month_field % name
        date_context['month'] = self.select_widget(attrs, choices=month_choices).get_context(
            name=month_name,
            value=context['widget']['value']['month'],
            attrs={**context['widget']['attrs'], 'id': 'id_%s' % month_name},
        )
        day_choices = [(i, i) for i in range(1, 32)]
        if not self.is_required:
            day_choices.insert(0, self.day_none_value)
        day_name = self.day_field % name
        date_context['day'] = self.select_widget(attrs, choices=day_choices,).get_context(
            name=day_name,
            value=context['widget']['value']['day'],
            attrs={**context['widget']['attrs'], 'id': 'id_%s' % day_name},
        )
        subwidgets = []
        for field in self._parse_date_fmt():
            subwidgets.append(date_context[field]['widget'])
        context['widget']['subwidgets'] = subwidgets
        return context
```
### 10 - django/forms/formsets.py:

Start line: 388, End line: 420

```python
@html_safe
class BaseFormSet:

    def clean(self):
        """
        Hook for doing any extra formset-wide cleaning after Form.clean() has
        been called on every form. Any ValidationError raised by this method
        will not be associated with a particular form; it will be accessible
        via formset.non_form_errors()
        """
        pass

    def has_changed(self):
        """Return True if data in any form differs from initial."""
        return any(form.has_changed() for form in self)

    def add_fields(self, form, index):
        """A hook for adding extra fields on to each form instance."""
        initial_form_count = self.initial_form_count()
        if self.can_order:
            # Only pre-fill the ordering field for initial forms.
            if index is not None and index < initial_form_count:
                form.fields[ORDERING_FIELD_NAME] = IntegerField(
                    label=_('Order'),
                    initial=index + 1,
                    required=False,
                    widget=self.get_ordering_widget(),
                )
            else:
                form.fields[ORDERING_FIELD_NAME] = IntegerField(
                    label=_('Order'),
                    required=False,
                    widget=self.get_ordering_widget(),
                )
        if self.can_delete and (self.can_delete_extra or index < initial_form_count):
            form.fields[DELETION_FIELD_NAME] = BooleanField(label=_('Delete'), required=False)
```
### 15 - django/forms/formsets.py:

Start line: 53, End line: 106

```python
@html_safe
class BaseFormSet:
    """
    A collection of instances of the same Form class.
    """
    ordering_widget = NumberInput
    default_error_messages = {
        'missing_management_form': _(
            'ManagementForm data is missing or has been tampered with. Missing fields: '
            '%(field_names)s. You may need to file a bug report if the issue persists.'
        ),
    }

    def __init__(self, data=None, files=None, auto_id='id_%s', prefix=None,
                 initial=None, error_class=ErrorList, form_kwargs=None,
                 error_messages=None):
        self.is_bound = data is not None or files is not None
        self.prefix = prefix or self.get_default_prefix()
        self.auto_id = auto_id
        self.data = data or {}
        self.files = files or {}
        self.initial = initial
        self.form_kwargs = form_kwargs or {}
        self.error_class = error_class
        self._errors = None
        self._non_form_errors = None

        messages = {}
        for cls in reversed(type(self).__mro__):
            messages.update(getattr(cls, 'default_error_messages', {}))
        if error_messages is not None:
            messages.update(error_messages)
        self.error_messages = messages

    def __str__(self):
        return self.as_table()

    def __iter__(self):
        """Yield the forms in the order they should be rendered."""
        return iter(self.forms)

    def __getitem__(self, index):
        """Return the form at the given index, based on the rendering order."""
        return self.forms[index]

    def __len__(self):
        return len(self.forms)

    def __bool__(self):
        """
        Return True since all formsets have a management form which is not
        included in the length.
        """
        return True
```
### 58 - django/forms/formsets.py:

Start line: 282, End line: 314

```python
@html_safe
class BaseFormSet:

    @classmethod
    def get_default_prefix(cls):
        return 'form'

    @classmethod
    def get_ordering_widget(cls):
        return cls.ordering_widget

    def non_form_errors(self):
        """
        Return an ErrorList of errors that aren't associated with a particular
        form -- i.e., from formset.clean(). Return an empty ErrorList if there
        are none.
        """
        if self._non_form_errors is None:
            self.full_clean()
        return self._non_form_errors

    @property
    def errors(self):
        """Return a list of form.errors for every form in self.forms."""
        if self._errors is None:
            self.full_clean()
        return self._errors

    def total_error_count(self):
        """Return the number of errors across all forms in the formset."""
        return len(self.non_form_errors()) +\
            sum(len(form_errors) for form_errors in self.errors)

    def _should_delete_form(self, form):
        """Return whether or not the form was marked for deletion."""
        return form.cleaned_data.get(DELETION_FIELD_NAME, False)
```
### 79 - django/forms/formsets.py:

Start line: 108, End line: 121

```python
@html_safe
class BaseFormSet:

    @cached_property
    def management_form(self):
        """Return the ManagementForm instance for this FormSet."""
        if self.is_bound:
            form = ManagementForm(self.data, auto_id=self.auto_id, prefix=self.prefix)
            form.full_clean()
        else:
            form = ManagementForm(auto_id=self.auto_id, prefix=self.prefix, initial={
                TOTAL_FORM_COUNT: self.total_form_count(),
                INITIAL_FORM_COUNT: self.initial_form_count(),
                MIN_NUM_FORM_COUNT: self.min_num,
                MAX_NUM_FORM_COUNT: self.max_num
            })
        return form
```
### 105 - django/forms/formsets.py:

Start line: 28, End line: 50

```python
class ManagementForm(Form):
    """
    Keep track of how many form instances are displayed on the page. If adding
    new forms via JavaScript, you should increment the count field of this form
    as well.
    """
    def __init__(self, *args, **kwargs):
        self.base_fields[TOTAL_FORM_COUNT] = IntegerField(widget=HiddenInput)
        self.base_fields[INITIAL_FORM_COUNT] = IntegerField(widget=HiddenInput)
        # MIN_NUM_FORM_COUNT and MAX_NUM_FORM_COUNT are output with the rest of
        # the management form, but only for the convenience of client-side
        # code. The POST value of them returned from the client is not checked.
        self.base_fields[MIN_NUM_FORM_COUNT] = IntegerField(required=False, widget=HiddenInput)
        self.base_fields[MAX_NUM_FORM_COUNT] = IntegerField(required=False, widget=HiddenInput)
        super().__init__(*args, **kwargs)

    def clean(self):
        cleaned_data = super().clean()
        # When the management form is invalid, we don't know how many forms
        # were submitted.
        cleaned_data.setdefault(TOTAL_FORM_COUNT, 0)
        cleaned_data.setdefault(INITIAL_FORM_COUNT, 0)
        return cleaned_data
```
### 106 - django/forms/formsets.py:

Start line: 197, End line: 226

```python
@html_safe
class BaseFormSet:

    @property
    def initial_forms(self):
        """Return a list of all the initial forms in this formset."""
        return self.forms[:self.initial_form_count()]

    @property
    def extra_forms(self):
        """Return a list of all the extra forms in this formset."""
        return self.forms[self.initial_form_count():]

    @property
    def empty_form(self):
        form = self.form(
            auto_id=self.auto_id,
            prefix=self.add_prefix('__prefix__'),
            empty_permitted=True,
            use_required_attribute=False,
            **self.get_form_kwargs(None)
        )
        self.add_fields(form, None)
        return form

    @property
    def cleaned_data(self):
        """
        Return a list of form.cleaned_data dicts for every form in self.forms.
        """
        if not self.is_valid():
            raise AttributeError("'%s' object has no attribute 'cleaned_data'" % self.__class__.__name__)
        return [form.cleaned_data for form in self.forms]
```
### 112 - django/forms/formsets.py:

Start line: 330, End line: 386

```python
@html_safe
class BaseFormSet:

    def full_clean(self):
        """
        Clean all of self.data and populate self._errors and
        self._non_form_errors.
        """
        self._errors = []
        self._non_form_errors = self.error_class(error_class='nonform')
        empty_forms_count = 0

        if not self.is_bound:  # Stop further processing.
            return

        if not self.management_form.is_valid():
            error = ValidationError(
                self.error_messages['missing_management_form'],
                params={
                    'field_names': ', '.join(
                        self.management_form.add_prefix(field_name)
                        for field_name in self.management_form.errors
                    ),
                },
                code='missing_management_form',
            )
            self._non_form_errors.append(error)

        for i, form in enumerate(self.forms):
            # Empty forms are unchanged forms beyond those with initial data.
            if not form.has_changed() and i >= self.initial_form_count():
                empty_forms_count += 1
            # Accessing errors calls full_clean() if necessary.
            # _should_delete_form() requires cleaned_data.
            form_errors = form.errors
            if self.can_delete and self._should_delete_form(form):
                continue
            self._errors.append(form_errors)
        try:
            if (self.validate_max and
                    self.total_form_count() - len(self.deleted_forms) > self.max_num) or \
                    self.management_form.cleaned_data[TOTAL_FORM_COUNT] > self.absolute_max:
                raise ValidationError(ngettext(
                    "Please submit at most %d form.",
                    "Please submit at most %d forms.", self.max_num) % self.max_num,
                    code='too_many_forms',
                )
            if (self.validate_min and
                    self.total_form_count() - len(self.deleted_forms) - empty_forms_count < self.min_num):
                raise ValidationError(ngettext(
                    "Please submit at least %d form.",
                    "Please submit at least %d forms.", self.min_num) % self.min_num,
                    code='too_few_forms')
            # Give self.clean() a chance to do cross-form validation.
            self.clean()
        except ValidationError as e:
            self._non_form_errors = self.error_class(
                e.error_list,
                error_class='nonform'
            )
```
### 136 - django/forms/formsets.py:

Start line: 142, End line: 167

```python
@html_safe
class BaseFormSet:

    def initial_form_count(self):
        """Return the number of forms that are required in this FormSet."""
        if self.is_bound:
            return self.management_form.cleaned_data[INITIAL_FORM_COUNT]
        else:
            # Use the length of the initial data if it's there, 0 otherwise.
            initial_forms = len(self.initial) if self.initial else 0
        return initial_forms

    @cached_property
    def forms(self):
        """Instantiate forms at first property access."""
        # DoS protection is included in total_form_count()
        return [
            self._construct_form(i, **self.get_form_kwargs(i))
            for i in range(self.total_form_count())
        ]

    def get_form_kwargs(self, index):
        """
        Return additional keyword arguments for each individual formset form.

        index will be None if the form being constructed is a new empty
        form.
        """
        return self.form_kwargs.copy()
```
### 163 - django/forms/formsets.py:

Start line: 169, End line: 195

```python
@html_safe
class BaseFormSet:

    def _construct_form(self, i, **kwargs):
        """Instantiate and return the i-th form instance in a formset."""
        defaults = {
            'auto_id': self.auto_id,
            'prefix': self.add_prefix(i),
            'error_class': self.error_class,
            # Don't render the HTML 'required' attribute as it may cause
            # incorrect validation for extra, optional, and deleted
            # forms in the formset.
            'use_required_attribute': False,
        }
        if self.is_bound:
            defaults['data'] = self.data
            defaults['files'] = self.files
        if self.initial and 'initial' not in kwargs:
            try:
                defaults['initial'] = self.initial[i]
            except IndexError:
                pass
        # Allow extra forms to be empty, unless they're part of
        # the minimum forms.
        if i >= self.initial_form_count() and i >= self.min_num:
            defaults['empty_permitted'] = True
        defaults.update(kwargs)
        form = self.form(**defaults)
        self.add_fields(form, i)
        return form
```
