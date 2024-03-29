# django__django-15799

| **django/django** | `90d2f9f41671ef01c8e8e7b5648f95c9bf512aae` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 7715 |
| **Any found context length** | 7715 |
| **Avg pos** | 23.0 |
| **Min pos** | 23 |
| **Max pos** | 23 |
| **Top file pos** | 4 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/contrib/admin/options.py b/django/contrib/admin/options.py
--- a/django/contrib/admin/options.py
+++ b/django/contrib/admin/options.py
@@ -314,8 +314,12 @@ def formfield_for_manytomany(self, db_field, request, **kwargs):
                 kwargs["queryset"] = queryset
 
         form_field = db_field.formfield(**kwargs)
-        if isinstance(form_field.widget, SelectMultiple) and not isinstance(
-            form_field.widget, (CheckboxSelectMultiple, AutocompleteSelectMultiple)
+        if (
+            isinstance(form_field.widget, SelectMultiple)
+            and form_field.widget.allow_multiple_selected
+            and not isinstance(
+                form_field.widget, (CheckboxSelectMultiple, AutocompleteSelectMultiple)
+            )
         ):
             msg = _(
                 "Hold down “Control”, or “Command” on a Mac, to select more than one."

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/contrib/admin/options.py | 317 | 318 | 23 | 4 | 7715


## Problem Statement

```
SelectMultiple in ModelAdminForm display help text when allow_multiple_selected is False.
Description
	
In AdminForm Help text on render for SelectMultiple widget don't check, if widget.allow_multiple_selected = False.
Widget himself on render checks it
# django.forms.widgets rows 684-685
if self.allow_multiple_selected:
 context['widget']['attrs']['multiple'] = True
But help_text for widget, whose is rendered behind widget - don't checks it. There we check only "isinstance" 
# django.contrib.admin.options.py rows 280-281
if (isinstance(form_field.widget, SelectMultiple) ann not isinstance(form_field.widget, (CheckboxSelectMultiple, AutocompleteSelectMultiple))):
	... # do some stuff with help text
as a result I get "msg", which should not be.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 django/forms/widgets.py | 805 | 835| 226 | 226 | 8260 | 
| 2 | 1 django/forms/widgets.py | 838 | 852| 139 | 365 | 8260 | 
| 3 | 1 django/forms/widgets.py | 725 | 743| 153 | 518 | 8260 | 
| 4 | 1 django/forms/widgets.py | 745 | 760| 122 | 640 | 8260 | 
| 5 | 2 django/forms/fields.py | 943 | 986| 307 | 947 | 17896 | 
| 6 | 3 django/contrib/admin/widgets.py | 1 | 48| 338 | 1285 | 22080 | 
| 7 | 3 django/contrib/admin/widgets.py | 555 | 588| 201 | 1486 | 22080 | 
| 8 | **4 django/contrib/admin/options.py** | 1744 | 1846| 782 | 2268 | 41287 | 
| 9 | **4 django/contrib/admin/options.py** | 1847 | 1878| 303 | 2571 | 41287 | 
| 10 | 5 django/forms/models.py | 1638 | 1666| 207 | 2778 | 53531 | 
| 11 | **5 django/contrib/admin/options.py** | 1 | 114| 776 | 3554 | 53531 | 
| 12 | 6 django/contrib/admin/utils.py | 388 | 427| 350 | 3904 | 57737 | 
| 13 | 6 django/forms/models.py | 1584 | 1599| 131 | 4035 | 57737 | 
| 14 | 7 django/contrib/admin/checks.py | 1091 | 1143| 430 | 4465 | 67269 | 
| 15 | **7 django/contrib/admin/options.py** | 1252 | 1314| 511 | 4976 | 67269 | 
| 16 | **7 django/contrib/admin/options.py** | 1998 | 2087| 767 | 5743 | 67269 | 
| 17 | 7 django/forms/widgets.py | 377 | 412| 279 | 6022 | 67269 | 
| 18 | 7 django/forms/widgets.py | 882 | 915| 282 | 6304 | 67269 | 
| 19 | 7 django/forms/models.py | 1563 | 1582| 154 | 6458 | 67269 | 
| 20 | **7 django/contrib/admin/options.py** | 2431 | 2458| 254 | 6712 | 67269 | 
| 21 | 7 django/forms/widgets.py | 1067 | 1108| 403 | 7115 | 67269 | 
| 22 | 7 django/contrib/admin/widgets.py | 207 | 234| 217 | 7332 | 67269 | 
| **-> 23 <-** | **7 django/contrib/admin/options.py** | 283 | 327| 383 | 7715 | 67269 | 
| 24 | 7 django/forms/widgets.py | 763 | 802| 240 | 7955 | 67269 | 
| 25 | 7 django/contrib/admin/widgets.py | 384 | 466| 485 | 8440 | 67269 | 
| 26 | 7 django/contrib/admin/widgets.py | 469 | 488| 141 | 8581 | 67269 | 
| 27 | 8 django/forms/boundfield.py | 36 | 53| 153 | 8734 | 69733 | 
| 28 | 8 django/contrib/admin/checks.py | 955 | 979| 197 | 8931 | 69733 | 
| 29 | 8 django/contrib/admin/checks.py | 217 | 264| 333 | 9264 | 69733 | 
| 30 | 8 django/contrib/admin/widgets.py | 520 | 553| 307 | 9571 | 69733 | 
| 31 | 8 django/forms/widgets.py | 1135 | 1154| 144 | 9715 | 69733 | 
| 32 | 8 django/contrib/admin/widgets.py | 346 | 358| 114 | 9829 | 69733 | 
| 33 | 8 django/forms/widgets.py | 592 | 626| 277 | 10106 | 69733 | 
| 34 | **8 django/contrib/admin/options.py** | 2372 | 2429| 465 | 10571 | 69733 | 
| 35 | 8 django/contrib/admin/checks.py | 894 | 928| 222 | 10793 | 69733 | 
| 36 | 8 django/forms/widgets.py | 668 | 689| 187 | 10980 | 69733 | 
| 37 | 8 django/contrib/admin/checks.py | 194 | 215| 141 | 11121 | 69733 | 
| 38 | 9 django/contrib/admin/helpers.py | 42 | 99| 324 | 11445 | 73399 | 
| 39 | 9 django/contrib/admin/checks.py | 1040 | 1071| 229 | 11674 | 73399 | 
| 40 | 9 django/forms/widgets.py | 691 | 722| 242 | 11916 | 73399 | 
| 41 | **9 django/contrib/admin/options.py** | 1673 | 1709| 338 | 12254 | 73399 | 
| 42 | 9 django/contrib/admin/checks.py | 1073 | 1089| 140 | 12394 | 73399 | 
| 43 | **9 django/contrib/admin/options.py** | 428 | 486| 509 | 12903 | 73399 | 
| 44 | 9 django/contrib/admin/helpers.py | 1 | 39| 227 | 13130 | 73399 | 
| 45 | 9 django/contrib/admin/helpers.py | 382 | 422| 327 | 13457 | 73399 | 
| 46 | 9 django/forms/widgets.py | 1156 | 1179| 250 | 13707 | 73399 | 
| 47 | 9 django/contrib/admin/checks.py | 879 | 892| 121 | 13828 | 73399 | 
| 48 | 9 django/contrib/admin/helpers.py | 168 | 195| 221 | 14049 | 73399 | 
| 49 | 9 django/forms/widgets.py | 553 | 589| 331 | 14380 | 73399 | 
| 50 | 9 django/contrib/admin/widgets.py | 297 | 343| 439 | 14819 | 73399 | 
| 51 | 9 django/forms/widgets.py | 628 | 666| 240 | 15059 | 73399 | 
| 52 | 9 django/contrib/admin/checks.py | 480 | 503| 188 | 15247 | 73399 | 
| 53 | 9 django/forms/fields.py | 852 | 913| 421 | 15668 | 73399 | 
| 54 | 9 django/contrib/admin/checks.py | 790 | 808| 183 | 15851 | 73399 | 
| 55 | 9 django/forms/fields.py | 989 | 1022| 235 | 16086 | 73399 | 
| 56 | 9 django/forms/widgets.py | 1017 | 1065| 407 | 16493 | 73399 | 
| 57 | 9 django/contrib/admin/checks.py | 521 | 537| 144 | 16637 | 73399 | 
| 58 | 10 django/contrib/gis/admin/options.py | 1 | 31| 227 | 16864 | 74876 | 
| 59 | 10 django/contrib/admin/checks.py | 981 | 1038| 457 | 17321 | 74876 | 
| 60 | 10 django/forms/widgets.py | 1 | 58| 306 | 17627 | 74876 | 
| 61 | **10 django/contrib/admin/options.py** | 2278 | 2335| 476 | 18103 | 74876 | 
| 62 | 10 django/contrib/admin/helpers.py | 499 | 512| 142 | 18245 | 74876 | 
| 63 | 10 django/contrib/admin/checks.py | 430 | 458| 224 | 18469 | 74876 | 
| 64 | **10 django/contrib/admin/options.py** | 488 | 532| 348 | 18817 | 74876 | 
| 65 | **10 django/contrib/admin/options.py** | 1909 | 1997| 676 | 19493 | 74876 | 
| 66 | 10 django/forms/models.py | 1601 | 1636| 288 | 19781 | 74876 | 
| 67 | 10 django/contrib/admin/widgets.py | 51 | 72| 168 | 19949 | 74876 | 
| 68 | 10 django/contrib/admin/checks.py | 1275 | 1319| 343 | 20292 | 74876 | 
| 69 | 10 django/contrib/gis/admin/options.py | 90 | 101| 139 | 20431 | 74876 | 
| 70 | **10 django/contrib/admin/options.py** | 1650 | 1671| 133 | 20564 | 74876 | 
| 71 | 10 django/forms/widgets.py | 855 | 880| 218 | 20782 | 74876 | 
| 72 | **10 django/contrib/admin/options.py** | 329 | 351| 169 | 20951 | 74876 | 
| 73 | 10 django/forms/widgets.py | 508 | 550| 279 | 21230 | 74876 | 
| 74 | 10 django/contrib/admin/checks.py | 1244 | 1273| 196 | 21426 | 74876 | 
| 75 | 10 django/contrib/admin/checks.py | 176 | 192| 155 | 21581 | 74876 | 
| 76 | 10 django/contrib/admin/widgets.py | 123 | 169| 358 | 21939 | 74876 | 
| 77 | 10 django/contrib/admin/helpers.py | 514 | 532| 138 | 22077 | 74876 | 
| 78 | 10 django/contrib/admin/checks.py | 841 | 877| 268 | 22345 | 74876 | 
| 79 | **10 django/contrib/admin/options.py** | 2226 | 2275| 446 | 22791 | 74876 | 
| 80 | 11 django/db/models/fields/related.py | 1417 | 1447| 172 | 22963 | 89488 | 
| 81 | **11 django/contrib/admin/options.py** | 117 | 147| 223 | 23186 | 89488 | 
| 82 | 11 django/forms/models.py | 1526 | 1560| 254 | 23440 | 89488 | 
| 83 | 12 django/contrib/admin/filters.py | 299 | 313| 152 | 23592 | 93757 | 
| 84 | **12 django/contrib/admin/options.py** | 353 | 426| 500 | 24092 | 93757 | 
| 85 | 12 django/contrib/admin/checks.py | 930 | 953| 195 | 24287 | 93757 | 
| 86 | 12 django/contrib/admin/widgets.py | 490 | 518| 257 | 24544 | 93757 | 
| 87 | 12 django/contrib/admin/checks.py | 348 | 367| 146 | 24690 | 93757 | 
| 88 | 12 django/forms/models.py | 245 | 255| 131 | 24821 | 93757 | 
| 89 | 12 django/forms/models.py | 889 | 912| 198 | 25019 | 93757 | 
| 90 | 13 django/contrib/postgres/forms/array.py | 146 | 180| 270 | 25289 | 95383 | 
| 91 | 14 django/contrib/admin/tests.py | 150 | 183| 279 | 25568 | 97025 | 
| 92 | 14 django/contrib/admin/checks.py | 762 | 787| 158 | 25726 | 97025 | 
| 93 | **14 django/contrib/admin/options.py** | 2460 | 2495| 315 | 26041 | 97025 | 
| 94 | 14 django/contrib/gis/admin/options.py | 123 | 181| 565 | 26606 | 97025 | 
| 95 | 14 django/contrib/admin/widgets.py | 171 | 204| 244 | 26850 | 97025 | 
| 96 | 14 django/contrib/admin/checks.py | 580 | 608| 195 | 27045 | 97025 | 
| 97 | 15 django/contrib/contenttypes/admin.py | 1 | 88| 585 | 27630 | 98030 | 
| 98 | 15 django/forms/widgets.py | 232 | 315| 627 | 28257 | 98030 | 
| 99 | 15 django/forms/widgets.py | 917 | 959| 302 | 28559 | 98030 | 
| 100 | 15 django/contrib/admin/helpers.py | 133 | 165| 256 | 28815 | 98030 | 
| 101 | 15 django/contrib/admin/filters.py | 180 | 225| 427 | 29242 | 98030 | 
| 102 | 15 django/contrib/admin/helpers.py | 424 | 460| 210 | 29452 | 98030 | 
| 103 | **15 django/contrib/admin/options.py** | 1135 | 1155| 201 | 29653 | 98030 | 
| 104 | 16 django/contrib/admindocs/views.py | 212 | 296| 615 | 30268 | 101512 | 
| 105 | 17 django/contrib/auth/admin.py | 28 | 40| 130 | 30398 | 103283 | 
| 106 | 17 django/contrib/admin/checks.py | 1231 | 1242| 116 | 30514 | 103283 | 
| 107 | 17 django/contrib/admin/checks.py | 1145 | 1181| 252 | 30766 | 103283 | 
| 108 | 17 django/forms/widgets.py | 338 | 374| 204 | 30970 | 103283 | 
| 109 | 17 django/contrib/admin/widgets.py | 75 | 92| 145 | 31115 | 103283 | 
| 110 | 17 django/contrib/admin/checks.py | 539 | 554| 136 | 31251 | 103283 | 
| 111 | 17 django/contrib/admin/tests.py | 205 | 235| 226 | 31477 | 103283 | 
| 112 | 17 django/contrib/admin/filters.py | 315 | 345| 229 | 31706 | 103283 | 
| 113 | 17 django/forms/widgets.py | 470 | 484| 126 | 31832 | 103283 | 
| 114 | 17 django/contrib/admin/helpers.py | 339 | 380| 257 | 32089 | 103283 | 
| 115 | 17 django/contrib/admin/checks.py | 810 | 824| 127 | 32216 | 103283 | 
| 116 | **17 django/contrib/admin/options.py** | 217 | 234| 174 | 32390 | 103283 | 
| 117 | **17 django/contrib/admin/options.py** | 866 | 879| 117 | 32507 | 103283 | 
| 118 | 17 django/forms/widgets.py | 415 | 435| 138 | 32645 | 103283 | 
| 119 | 17 django/contrib/admin/helpers.py | 264 | 302| 323 | 32968 | 103283 | 
| 120 | 17 django/contrib/admin/checks.py | 556 | 578| 194 | 33162 | 103283 | 
| 121 | 17 django/db/models/fields/related.py | 1449 | 1578| 984 | 34146 | 103283 | 
| 122 | 17 django/contrib/admin/filters.py | 227 | 250| 202 | 34348 | 103283 | 
| 123 | 18 django/contrib/admin/views/main.py | 554 | 586| 227 | 34575 | 107814 | 
| 124 | 18 django/contrib/admin/widgets.py | 237 | 295| 436 | 35011 | 107814 | 
| 125 | 18 django/forms/models.py | 386 | 429| 396 | 35407 | 107814 | 
| 126 | 19 django/contrib/admin/templatetags/admin_modify.py | 61 | 112| 414 | 35821 | 108858 | 
| 127 | 19 django/forms/widgets.py | 61 | 108| 311 | 36132 | 108858 | 
| 128 | **19 django/contrib/admin/options.py** | 236 | 249| 139 | 36271 | 108858 | 
| 129 | 20 django/contrib/postgres/forms/ranges.py | 1 | 34| 206 | 36477 | 109584 | 
| 130 | 20 django/contrib/admindocs/views.py | 297 | 391| 640 | 37117 | 109584 | 
| 131 | 20 django/forms/fields.py | 82 | 177| 791 | 37908 | 109584 | 
| 132 | 20 django/contrib/admin/checks.py | 743 | 760| 139 | 38047 | 109584 | 
| 133 | **20 django/contrib/admin/options.py** | 1079 | 1098| 131 | 38178 | 109584 | 
| 134 | 20 django/contrib/contenttypes/admin.py | 91 | 144| 420 | 38598 | 109584 | 
| 135 | 20 django/contrib/admin/helpers.py | 102 | 130| 165 | 38763 | 109584 | 
| 136 | 20 django/contrib/admin/views/main.py | 1 | 51| 324 | 39087 | 109584 | 
| 137 | 20 django/contrib/admin/checks.py | 284 | 312| 218 | 39305 | 109584 | 
| 138 | 20 django/db/models/fields/related.py | 1580 | 1678| 655 | 39960 | 109584 | 
| 139 | 20 django/db/models/fields/related.py | 1959 | 1993| 266 | 40226 | 109584 | 
| 140 | 20 django/contrib/admin/checks.py | 610 | 628| 164 | 40390 | 109584 | 
| 141 | 20 django/contrib/postgres/forms/array.py | 116 | 144| 222 | 40612 | 109584 | 
| 142 | 20 django/forms/boundfield.py | 1 | 34| 233 | 40845 | 109584 | 
| 143 | **20 django/contrib/admin/options.py** | 721 | 738| 128 | 40973 | 109584 | 
| 144 | 20 django/contrib/gis/admin/options.py | 103 | 121| 156 | 41129 | 109584 | 
| 145 | 21 django/db/models/fields/__init__.py | 2333 | 2367| 229 | 41358 | 128307 | 
| 146 | 22 django/forms/formsets.py | 1 | 25| 210 | 41568 | 132591 | 
| 147 | 22 django/contrib/admin/filters.py | 457 | 485| 223 | 41791 | 132591 | 
| 148 | 22 django/contrib/admin/checks.py | 673 | 702| 241 | 42032 | 132591 | 
| 149 | 22 django/db/models/fields/__init__.py | 1246 | 1283| 245 | 42277 | 132591 | 
| 150 | 22 django/contrib/admin/checks.py | 826 | 839| 117 | 42394 | 132591 | 
| 151 | 22 django/contrib/admin/filters.py | 488 | 501| 119 | 42513 | 132591 | 
| 152 | 23 django/contrib/admin/views/autocomplete.py | 66 | 123| 425 | 42938 | 133432 | 
| 153 | 24 django/contrib/auth/forms.py | 33 | 60| 186 | 43124 | 136624 | 
| 154 | 24 django/forms/widgets.py | 486 | 505| 195 | 43319 | 136624 | 
| 155 | **24 django/contrib/admin/options.py** | 881 | 896| 114 | 43433 | 136624 | 
| 156 | 24 django/forms/models.py | 796 | 887| 783 | 44216 | 136624 | 
| 157 | 25 django/db/models/base.py | 1994 | 2047| 355 | 44571 | 155165 | 
| 158 | 25 django/contrib/admin/checks.py | 657 | 671| 141 | 44712 | 155165 | 
| 159 | 25 django/contrib/admin/checks.py | 266 | 282| 138 | 44850 | 155165 | 
| 160 | **25 django/contrib/admin/options.py** | 898 | 942| 314 | 45164 | 155165 | 
| 161 | 25 django/contrib/admin/helpers.py | 535 | 561| 199 | 45363 | 155165 | 
| 162 | **25 django/contrib/admin/options.py** | 610 | 660| 349 | 45712 | 155165 | 
| 163 | 25 django/contrib/admin/checks.py | 413 | 428| 145 | 45857 | 155165 | 
| 164 | 25 django/forms/models.py | 258 | 336| 627 | 46484 | 155165 | 
| 165 | 25 django/forms/fields.py | 1236 | 1261| 191 | 46675 | 155165 | 
| 166 | 25 django/forms/models.py | 1511 | 1524| 176 | 46851 | 155165 | 
| 167 | **25 django/contrib/admin/options.py** | 662 | 677| 123 | 46974 | 155165 | 
| 168 | **25 django/contrib/admin/options.py** | 1893 | 1907| 132 | 47106 | 155165 | 
| 169 | 25 django/contrib/admin/utils.py | 430 | 459| 200 | 47306 | 155165 | 
| 170 | 25 django/db/models/fields/related.py | 1680 | 1730| 431 | 47737 | 155165 | 
| 171 | 25 django/forms/widgets.py | 110 | 126| 127 | 47864 | 155165 | 
| 172 | 25 django/contrib/admin/checks.py | 460 | 478| 137 | 48001 | 155165 | 
| 173 | 25 django/contrib/admin/widgets.py | 361 | 381| 172 | 48173 | 155165 | 
| 174 | **25 django/contrib/admin/options.py** | 2089 | 2164| 599 | 48772 | 155165 | 
| 175 | 26 django/forms/forms.py | 457 | 517| 419 | 49191 | 159246 | 
| 176 | 26 django/db/models/fields/__init__.py | 1095 | 1114| 223 | 49414 | 159246 | 
| 177 | 26 django/contrib/admin/filters.py | 275 | 296| 192 | 49606 | 159246 | 
| 178 | 26 django/db/models/fields/__init__.py | 2667 | 2717| 313 | 49919 | 159246 | 
| 179 | 26 django/forms/models.py | 468 | 498| 244 | 50163 | 159246 | 
| 180 | 27 django/views/generic/edit.py | 78 | 110| 269 | 50432 | 161250 | 
| 181 | 27 django/contrib/admin/checks.py | 505 | 519| 122 | 50554 | 161250 | 
| 182 | 27 django/contrib/admin/checks.py | 314 | 346| 233 | 50787 | 161250 | 


### Hint

```
rendered m2m field with allow_multiple_selected=False
Thanks for the report. As far as I understand correctly, you have a subclass of SelectMultiple with allow_multiple_selected set to False, that's quite niche. However, I agree that we should check allow_multiple_selected in both places: django/contrib/admin/options.py diff --git a/django/contrib/admin/options.py b/django/contrib/admin/options.py index a25814b5fb..f959a8dc48 100644 a b class BaseModelAdmin(metaclass=forms.MediaDefiningClass): 314314 kwargs["queryset"] = queryset 315315 316316 form_field = db_field.formfield(**kwargs) 317 if isinstance(form_field.widget, SelectMultiple) and not isinstance( 318 form_field.widget, (CheckboxSelectMultiple, AutocompleteSelectMultiple) 317 if ( 318 isinstance(form_field.widget, SelectMultiple) 319 and form_field.widget.allow_multiple_selected 320 and not isinstance( 321 form_field.widget, (CheckboxSelectMultiple, AutocompleteSelectMultiple) 322 ) 319323 ): 320324 msg = _( 321325 "Hold down “Control”, or “Command” on a Mac, to select more than one."
As far as I understand correctly, you have a subclass of SelectMultiple with allow_multiple_selected set to False. Exactly.
```

## Patch

```diff
diff --git a/django/contrib/admin/options.py b/django/contrib/admin/options.py
--- a/django/contrib/admin/options.py
+++ b/django/contrib/admin/options.py
@@ -314,8 +314,12 @@ def formfield_for_manytomany(self, db_field, request, **kwargs):
                 kwargs["queryset"] = queryset
 
         form_field = db_field.formfield(**kwargs)
-        if isinstance(form_field.widget, SelectMultiple) and not isinstance(
-            form_field.widget, (CheckboxSelectMultiple, AutocompleteSelectMultiple)
+        if (
+            isinstance(form_field.widget, SelectMultiple)
+            and form_field.widget.allow_multiple_selected
+            and not isinstance(
+                form_field.widget, (CheckboxSelectMultiple, AutocompleteSelectMultiple)
+            )
         ):
             msg = _(
                 "Hold down “Control”, or “Command” on a Mac, to select more than one."

```

## Test Patch

```diff
diff --git a/tests/admin_widgets/tests.py b/tests/admin_widgets/tests.py
--- a/tests/admin_widgets/tests.py
+++ b/tests/admin_widgets/tests.py
@@ -273,6 +273,26 @@ class AdvisorAdmin(admin.ModelAdmin):
             "Hold down “Control”, or “Command” on a Mac, to select more than one.",
         )
 
+    def test_m2m_widgets_no_allow_multiple_selected(self):
+        class NoAllowMultipleSelectedWidget(forms.SelectMultiple):
+            allow_multiple_selected = False
+
+        class AdvisorAdmin(admin.ModelAdmin):
+            filter_vertical = ["companies"]
+            formfield_overrides = {
+                ManyToManyField: {"widget": NoAllowMultipleSelectedWidget},
+            }
+
+        self.assertFormfield(
+            Advisor,
+            "companies",
+            widgets.FilteredSelectMultiple,
+            filter_vertical=["companies"],
+        )
+        ma = AdvisorAdmin(Advisor, admin.site)
+        f = ma.formfield_for_dbfield(Advisor._meta.get_field("companies"), request=None)
+        self.assertEqual(f.help_text, "")
+
 
 @override_settings(ROOT_URLCONF="admin_widgets.urls")
 class AdminFormfieldForDBFieldWithRequestTests(TestDataMixin, TestCase):

```


## Code snippets

### 1 - django/forms/widgets.py:

Start line: 805, End line: 835

```python
class SelectMultiple(Select):
    allow_multiple_selected = True

    def value_from_datadict(self, data, files, name):
        try:
            getter = data.getlist
        except AttributeError:
            getter = data.get
        return getter(name)

    def value_omitted_from_data(self, data, files, name):
        # An unselected <select multiple> doesn't appear in POST data, so it's
        # never known if the value is actually omitted.
        return False


class RadioSelect(ChoiceWidget):
    input_type = "radio"
    template_name = "django/forms/widgets/radio.html"
    option_template_name = "django/forms/widgets/radio_option.html"
    use_fieldset = True

    def id_for_label(self, id_, index=None):
        """
        Don't include for="field_0" in <label> to improve accessibility when
        using a screen reader, in addition clicking such a label would toggle
        the first input.
        """
        if index is None:
            return ""
        return super().id_for_label(id_, index)
```
### 2 - django/forms/widgets.py:

Start line: 838, End line: 852

```python
class CheckboxSelectMultiple(RadioSelect):
    allow_multiple_selected = True
    input_type = "checkbox"
    template_name = "django/forms/widgets/checkbox_select.html"
    option_template_name = "django/forms/widgets/checkbox_option.html"

    def use_required_attribute(self, initial):
        # Don't use the 'required' attribute because browser validation would
        # require all checkboxes to be checked instead of at least one.
        return False

    def value_omitted_from_data(self, data, files, name):
        # HTML checkboxes don't appear in POST data if not checked, so it's
        # never known if the value is actually omitted.
        return False
```
### 3 - django/forms/widgets.py:

Start line: 725, End line: 743

```python
class Select(ChoiceWidget):
    input_type = "select"
    template_name = "django/forms/widgets/select.html"
    option_template_name = "django/forms/widgets/select_option.html"
    add_id_index = False
    checked_attribute = {"selected": True}
    option_inherits_attrs = False

    def get_context(self, name, value, attrs):
        context = super().get_context(name, value, attrs)
        if self.allow_multiple_selected:
            context["widget"]["attrs"]["multiple"] = True
        return context

    @staticmethod
    def _choice_has_empty_value(choice):
        """Return True if the choice's value is empty string or None."""
        value, _ = choice
        return value is None or value == ""
```
### 4 - django/forms/widgets.py:

Start line: 745, End line: 760

```python
class Select(ChoiceWidget):

    def use_required_attribute(self, initial):
        """
        Don't render 'required' if the first <option> has a value, as that's
        invalid HTML.
        """
        use_required_attribute = super().use_required_attribute(initial)
        # 'required' is always okay for <select multiple>.
        if self.allow_multiple_selected:
            return use_required_attribute

        first_choice = next(iter(self.choices), None)
        return (
            use_required_attribute
            and first_choice is not None
            and self._choice_has_empty_value(first_choice)
        )
```
### 5 - django/forms/fields.py:

Start line: 943, End line: 986

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
### 6 - django/contrib/admin/widgets.py:

Start line: 1, End line: 48

```python
"""
Form Widget classes specific to the Django admin site.
"""
import copy
import json

from django import forms
from django.conf import settings
from django.core.exceptions import ValidationError
from django.core.validators import URLValidator
from django.db.models import CASCADE, UUIDField
from django.urls import reverse
from django.urls.exceptions import NoReverseMatch
from django.utils.html import smart_urlquote
from django.utils.http import urlencode
from django.utils.text import Truncator
from django.utils.translation import get_language
from django.utils.translation import gettext as _


class FilteredSelectMultiple(forms.SelectMultiple):
    """
    A SelectMultiple with a JavaScript filter interface.

    Note that the resulting JavaScript assumes that the jsi18n
    catalog has been loaded in the page
    """

    class Media:
        js = [
            "admin/js/core.js",
            "admin/js/SelectBox.js",
            "admin/js/SelectFilter2.js",
        ]

    def __init__(self, verbose_name, is_stacked, attrs=None, choices=()):
        self.verbose_name = verbose_name
        self.is_stacked = is_stacked
        super().__init__(attrs, choices)

    def get_context(self, name, value, attrs):
        context = super().get_context(name, value, attrs)
        context["widget"]["attrs"]["class"] = "selectfilter"
        if self.is_stacked:
            context["widget"]["attrs"]["class"] += "stacked"
        context["widget"]["attrs"]["data-field-name"] = self.verbose_name
        context["widget"]["attrs"]["data-is-stacked"] = int(self.is_stacked)
        return context
```
### 7 - django/contrib/admin/widgets.py:

Start line: 555, End line: 588

```python
class AutocompleteMixin:

    @property
    def media(self):
        extra = "" if settings.DEBUG else ".min"
        i18n_file = (
            ("admin/js/vendor/select2/i18n/%s.js" % self.i18n_name,)
            if self.i18n_name
            else ()
        )
        return forms.Media(
            js=(
                "admin/js/vendor/jquery/jquery%s.js" % extra,
                "admin/js/vendor/select2/select2.full%s.js" % extra,
            )
            + i18n_file
            + (
                "admin/js/jquery.init.js",
                "admin/js/autocomplete.js",
            ),
            css={
                "screen": (
                    "admin/css/vendor/select2/select2%s.css" % extra,
                    "admin/css/autocomplete.css",
                ),
            },
        )


class AutocompleteSelect(AutocompleteMixin, forms.Select):
    pass


class AutocompleteSelectMultiple(AutocompleteMixin, forms.SelectMultiple):
    pass
```
### 8 - django/contrib/admin/options.py:

Start line: 1744, End line: 1846

```python
class ModelAdmin(BaseModelAdmin):

    def _changeform_view(self, request, object_id, form_url, extra_context):
        to_field = request.POST.get(TO_FIELD_VAR, request.GET.get(TO_FIELD_VAR))
        if to_field and not self.to_field_allowed(request, to_field):
            raise DisallowedModelAdminToField(
                "The field %s cannot be referenced." % to_field
            )

        if request.method == "POST" and "_saveasnew" in request.POST:
            object_id = None

        add = object_id is None

        if add:
            if not self.has_add_permission(request):
                raise PermissionDenied
            obj = None

        else:
            obj = self.get_object(request, unquote(object_id), to_field)

            if request.method == "POST":
                if not self.has_change_permission(request, obj):
                    raise PermissionDenied
            else:
                if not self.has_view_or_change_permission(request, obj):
                    raise PermissionDenied

            if obj is None:
                return self._get_obj_does_not_exist_redirect(
                    request, self.opts, object_id
                )

        fieldsets = self.get_fieldsets(request, obj)
        ModelForm = self.get_form(
            request, obj, change=not add, fields=flatten_fieldsets(fieldsets)
        )
        if request.method == "POST":
            form = ModelForm(request.POST, request.FILES, instance=obj)
            formsets, inline_instances = self._create_formsets(
                request,
                form.instance,
                change=not add,
            )
            form_validated = form.is_valid()
            if form_validated:
                new_object = self.save_form(request, form, change=not add)
            else:
                new_object = form.instance
            if all_valid(formsets) and form_validated:
                self.save_model(request, new_object, form, not add)
                self.save_related(request, form, formsets, not add)
                change_message = self.construct_change_message(
                    request, form, formsets, add
                )
                if add:
                    self.log_addition(request, new_object, change_message)
                    return self.response_add(request, new_object)
                else:
                    self.log_change(request, new_object, change_message)
                    return self.response_change(request, new_object)
            else:
                form_validated = False
        else:
            if add:
                initial = self.get_changeform_initial_data(request)
                form = ModelForm(initial=initial)
                formsets, inline_instances = self._create_formsets(
                    request, form.instance, change=False
                )
            else:
                form = ModelForm(instance=obj)
                formsets, inline_instances = self._create_formsets(
                    request, obj, change=True
                )

        if not add and not self.has_change_permission(request, obj):
            readonly_fields = flatten_fieldsets(fieldsets)
        else:
            readonly_fields = self.get_readonly_fields(request, obj)
        adminForm = helpers.AdminForm(
            form,
            list(fieldsets),
            # Clear prepopulated fields on a view-only form to avoid a crash.
            self.get_prepopulated_fields(request, obj)
            if add or self.has_change_permission(request, obj)
            else {},
            readonly_fields,
            model_admin=self,
        )
        media = self.media + adminForm.media

        inline_formsets = self.get_inline_formsets(
            request, formsets, inline_instances, obj
        )
        for inline_formset in inline_formsets:
            media = media + inline_formset.media

        if add:
            title = _("Add %s")
        elif self.has_change_permission(request, obj):
            title = _("Change %s")
        else:
            title = _("View %s")
        # ... other code
```
### 9 - django/contrib/admin/options.py:

Start line: 1847, End line: 1878

```python
class ModelAdmin(BaseModelAdmin):

    def _changeform_view(self, request, object_id, form_url, extra_context):
        # ... other code
        context = {
            **self.admin_site.each_context(request),
            "title": title % self.opts.verbose_name,
            "subtitle": str(obj) if obj else None,
            "adminform": adminForm,
            "object_id": object_id,
            "original": obj,
            "is_popup": IS_POPUP_VAR in request.POST or IS_POPUP_VAR in request.GET,
            "to_field": to_field,
            "media": media,
            "inline_admin_formsets": inline_formsets,
            "errors": helpers.AdminErrorList(form, formsets),
            "preserved_filters": self.get_preserved_filters(request),
        }

        # Hide the "Save" and "Save and continue" buttons if "Save as New" was
        # previously chosen to prevent the interface from getting confusing.
        if (
            request.method == "POST"
            and not form_validated
            and "_saveasnew" in request.POST
        ):
            context["show_save"] = False
            context["show_save_and_continue"] = False
            # Use the change template instead of the add template.
            add = False

        context.update(extra_context or {})

        return self.render_change_form(
            request, context, add=add, change=not add, obj=obj, form_url=form_url
        )
```
### 10 - django/forms/models.py:

Start line: 1638, End line: 1666

```python
class ModelMultipleChoiceField(ModelChoiceField):

    def prepare_value(self, value):
        if (
            hasattr(value, "__iter__")
            and not isinstance(value, str)
            and not hasattr(value, "_meta")
        ):
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
    return hasattr(form_class, "_meta") and (
        form_class._meta.fields is not None or form_class._meta.exclude is not None
    )
```
### 11 - django/contrib/admin/options.py:

Start line: 1, End line: 114

```python
import copy
import json
import re
from functools import partial, update_wrapper
from urllib.parse import quote as urlquote

from django import forms
from django.conf import settings
from django.contrib import messages
from django.contrib.admin import helpers, widgets
from django.contrib.admin.checks import (
    BaseModelAdminChecks,
    InlineModelAdminChecks,
    ModelAdminChecks,
)
from django.contrib.admin.decorators import display
from django.contrib.admin.exceptions import DisallowedModelAdminToField
from django.contrib.admin.templatetags.admin_urls import add_preserved_filters
from django.contrib.admin.utils import (
    NestedObjects,
    construct_change_message,
    flatten_fieldsets,
    get_deleted_objects,
    lookup_spawns_duplicates,
    model_format_dict,
    model_ngettext,
    quote,
    unquote,
)
from django.contrib.admin.widgets import AutocompleteSelect, AutocompleteSelectMultiple
from django.contrib.auth import get_permission_codename
from django.core.exceptions import (
    FieldDoesNotExist,
    FieldError,
    PermissionDenied,
    ValidationError,
)
from django.core.paginator import Paginator
from django.db import models, router, transaction
from django.db.models.constants import LOOKUP_SEP
from django.forms.formsets import DELETION_FIELD_NAME, all_valid
from django.forms.models import (
    BaseInlineFormSet,
    inlineformset_factory,
    modelform_defines_fields,
    modelform_factory,
    modelformset_factory,
)
from django.forms.widgets import CheckboxSelectMultiple, SelectMultiple
from django.http import HttpResponseRedirect
from django.http.response import HttpResponseBase
from django.template.response import SimpleTemplateResponse, TemplateResponse
from django.urls import reverse
from django.utils.decorators import method_decorator
from django.utils.html import format_html
from django.utils.http import urlencode
from django.utils.safestring import mark_safe
from django.utils.text import (
    capfirst,
    format_lazy,
    get_text_list,
    smart_split,
    unescape_string_literal,
)
from django.utils.translation import gettext as _
from django.utils.translation import ngettext
from django.views.decorators.csrf import csrf_protect
from django.views.generic import RedirectView

IS_POPUP_VAR = "_popup"
TO_FIELD_VAR = "_to_field"


HORIZONTAL, VERTICAL = 1, 2


def get_content_type_for_model(obj):
    # Since this module gets imported in the application's root package,
    # it cannot import models from other applications at the module level.
    from django.contrib.contenttypes.models import ContentType

    return ContentType.objects.get_for_model(obj, for_concrete_model=False)


def get_ul_class(radio_style):
    return "radiolist" if radio_style == VERTICAL else "radiolist inline"


class IncorrectLookupParameters(Exception):
    pass


# Defaults for formfield_overrides. ModelAdmin subclasses can change this
# by adding to ModelAdmin.formfield_overrides.

FORMFIELD_FOR_DBFIELD_DEFAULTS = {
    models.DateTimeField: {
        "form_class": forms.SplitDateTimeField,
        "widget": widgets.AdminSplitDateTime,
    },
    models.DateField: {"widget": widgets.AdminDateWidget},
    models.TimeField: {"widget": widgets.AdminTimeWidget},
    models.TextField: {"widget": widgets.AdminTextareaWidget},
    models.URLField: {"widget": widgets.AdminURLFieldWidget},
    models.IntegerField: {"widget": widgets.AdminIntegerFieldWidget},
    models.BigIntegerField: {"widget": widgets.AdminBigIntegerFieldWidget},
    models.CharField: {"widget": widgets.AdminTextInputWidget},
    models.ImageField: {"widget": widgets.AdminFileWidget},
    models.FileField: {"widget": widgets.AdminFileWidget},
    models.EmailField: {"widget": widgets.AdminEmailInputWidget},
    models.UUIDField: {"widget": widgets.AdminUUIDInputWidget},
}

csrf_protect_m = method_decorator(csrf_protect)
```
### 15 - django/contrib/admin/options.py:

Start line: 1252, End line: 1314

```python
class ModelAdmin(BaseModelAdmin):

    def render_change_form(
        self, request, context, add=False, change=False, form_url="", obj=None
    ):
        app_label = self.opts.app_label
        preserved_filters = self.get_preserved_filters(request)
        form_url = add_preserved_filters(
            {"preserved_filters": preserved_filters, "opts": self.opts}, form_url
        )
        view_on_site_url = self.get_view_on_site_url(obj)
        has_editable_inline_admin_formsets = False
        for inline in context["inline_admin_formsets"]:
            if (
                inline.has_add_permission
                or inline.has_change_permission
                or inline.has_delete_permission
            ):
                has_editable_inline_admin_formsets = True
                break
        context.update(
            {
                "add": add,
                "change": change,
                "has_view_permission": self.has_view_permission(request, obj),
                "has_add_permission": self.has_add_permission(request),
                "has_change_permission": self.has_change_permission(request, obj),
                "has_delete_permission": self.has_delete_permission(request, obj),
                "has_editable_inline_admin_formsets": (
                    has_editable_inline_admin_formsets
                ),
                "has_file_field": context["adminform"].form.is_multipart()
                or any(
                    admin_formset.formset.is_multipart()
                    for admin_formset in context["inline_admin_formsets"]
                ),
                "has_absolute_url": view_on_site_url is not None,
                "absolute_url": view_on_site_url,
                "form_url": form_url,
                "opts": self.opts,
                "content_type_id": get_content_type_for_model(self.model).pk,
                "save_as": self.save_as,
                "save_on_top": self.save_on_top,
                "to_field_var": TO_FIELD_VAR,
                "is_popup_var": IS_POPUP_VAR,
                "app_label": app_label,
            }
        )
        if add and self.add_form_template is not None:
            form_template = self.add_form_template
        else:
            form_template = self.change_form_template

        request.current_app = self.admin_site.name

        return TemplateResponse(
            request,
            form_template
            or [
                "admin/%s/%s/change_form.html" % (app_label, self.opts.model_name),
                "admin/%s/change_form.html" % app_label,
                "admin/change_form.html",
            ],
            context,
        )
```
### 16 - django/contrib/admin/options.py:

Start line: 1998, End line: 2087

```python
class ModelAdmin(BaseModelAdmin):

    @csrf_protect_m
    def changelist_view(self, request, extra_context=None):
        # ... other code
        if request.method == "POST" and cl.list_editable and "_save" in request.POST:
            if not self.has_change_permission(request):
                raise PermissionDenied
            FormSet = self.get_changelist_formset(request)
            modified_objects = self._get_list_editable_queryset(
                request, FormSet.get_default_prefix()
            )
            formset = cl.formset = FormSet(
                request.POST, request.FILES, queryset=modified_objects
            )
            if formset.is_valid():
                changecount = 0
                for form in formset.forms:
                    if form.has_changed():
                        obj = self.save_form(request, form, change=True)
                        self.save_model(request, obj, form, change=True)
                        self.save_related(request, form, formsets=[], change=True)
                        change_msg = self.construct_change_message(request, form, None)
                        self.log_change(request, obj, change_msg)
                        changecount += 1

                if changecount:
                    msg = ngettext(
                        "%(count)s %(name)s was changed successfully.",
                        "%(count)s %(name)s were changed successfully.",
                        changecount,
                    ) % {
                        "count": changecount,
                        "name": model_ngettext(self.opts, changecount),
                    }
                    self.message_user(request, msg, messages.SUCCESS)

                return HttpResponseRedirect(request.get_full_path())

        # Handle GET -- construct a formset for display.
        elif cl.list_editable and self.has_change_permission(request):
            FormSet = self.get_changelist_formset(request)
            formset = cl.formset = FormSet(queryset=cl.result_list)

        # Build the list of media to be used by the formset.
        if formset:
            media = self.media + formset.media
        else:
            media = self.media

        # Build the action form and populate it with available actions.
        if actions:
            action_form = self.action_form(auto_id=None)
            action_form.fields["action"].choices = self.get_action_choices(request)
            media += action_form.media
        else:
            action_form = None

        selection_note_all = ngettext(
            "%(total_count)s selected", "All %(total_count)s selected", cl.result_count
        )

        context = {
            **self.admin_site.each_context(request),
            "module_name": str(self.opts.verbose_name_plural),
            "selection_note": _("0 of %(cnt)s selected") % {"cnt": len(cl.result_list)},
            "selection_note_all": selection_note_all % {"total_count": cl.result_count},
            "title": cl.title,
            "subtitle": None,
            "is_popup": cl.is_popup,
            "to_field": cl.to_field,
            "cl": cl,
            "media": media,
            "has_add_permission": self.has_add_permission(request),
            "opts": cl.opts,
            "action_form": action_form,
            "actions_on_top": self.actions_on_top,
            "actions_on_bottom": self.actions_on_bottom,
            "actions_selection_counter": self.actions_selection_counter,
            "preserved_filters": self.get_preserved_filters(request),
            **(extra_context or {}),
        }

        request.current_app = self.admin_site.name

        return TemplateResponse(
            request,
            self.change_list_template
            or [
                "admin/%s/%s/change_list.html" % (app_label, self.opts.model_name),
                "admin/%s/change_list.html" % app_label,
                "admin/change_list.html",
            ],
            context,
        )
```
### 20 - django/contrib/admin/options.py:

Start line: 2431, End line: 2458

```python
class InlineModelAdmin(BaseModelAdmin):

    def _get_form_for_get_fields(self, request, obj=None):
        return self.get_formset(request, obj, fields=None).form

    def get_queryset(self, request):
        queryset = super().get_queryset(request)
        if not self.has_view_or_change_permission(request):
            queryset = queryset.none()
        return queryset

    def _has_any_perms_for_target_model(self, request, perms):
        """
        This method is called only when the ModelAdmin's model is for an
        ManyToManyField's implicit through model (if self.opts.auto_created).
        Return True if the user has any of the given permissions ('add',
        'change', etc.) for the model that points to the through model.
        """
        opts = self.opts
        # Find the target model of an auto-created many-to-many relationship.
        for field in opts.fields:
            if field.remote_field and field.remote_field.model != self.parent_model:
                opts = field.remote_field.model._meta
                break
        return any(
            request.user.has_perm(
                "%s.%s" % (opts.app_label, get_permission_codename(perm, opts))
            )
            for perm in perms
        )
```
### 23 - django/contrib/admin/options.py:

Start line: 283, End line: 327

```python
class BaseModelAdmin(metaclass=forms.MediaDefiningClass):

    def formfield_for_manytomany(self, db_field, request, **kwargs):
        """
        Get a form Field for a ManyToManyField.
        """
        # If it uses an intermediary model that isn't auto created, don't show
        # a field in admin.
        if not db_field.remote_field.through._meta.auto_created:
            return None
        db = kwargs.get("using")

        if "widget" not in kwargs:
            autocomplete_fields = self.get_autocomplete_fields(request)
            if db_field.name in autocomplete_fields:
                kwargs["widget"] = AutocompleteSelectMultiple(
                    db_field,
                    self.admin_site,
                    using=db,
                )
            elif db_field.name in self.raw_id_fields:
                kwargs["widget"] = widgets.ManyToManyRawIdWidget(
                    db_field.remote_field,
                    self.admin_site,
                    using=db,
                )
            elif db_field.name in [*self.filter_vertical, *self.filter_horizontal]:
                kwargs["widget"] = widgets.FilteredSelectMultiple(
                    db_field.verbose_name, db_field.name in self.filter_vertical
                )
        if "queryset" not in kwargs:
            queryset = self.get_field_queryset(db, db_field, request)
            if queryset is not None:
                kwargs["queryset"] = queryset

        form_field = db_field.formfield(**kwargs)
        if isinstance(form_field.widget, SelectMultiple) and not isinstance(
            form_field.widget, (CheckboxSelectMultiple, AutocompleteSelectMultiple)
        ):
            msg = _(
                "Hold down “Control”, or “Command” on a Mac, to select more than one."
            )
            help_text = form_field.help_text
            form_field.help_text = (
                format_lazy("{} {}", help_text, msg) if help_text else msg
            )
        return form_field
```
### 34 - django/contrib/admin/options.py:

Start line: 2372, End line: 2429

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
                                # Translators: Model verbose name and instance
                                # representation, suitable to be an item in a
                                # list.
                                _("%(class_name)s %(instance)s")
                                % {"class_name": p._meta.verbose_name, "instance": p}
                            )
                        params = {
                            "class_name": self._meta.model._meta.verbose_name,
                            "instance": self.instance,
                            "related_objects": get_text_list(objs, _("and")),
                        }
                        msg = _(
                            "Deleting %(class_name)s %(instance)s would require "
                            "deleting the following protected related objects: "
                            "%(related_objects)s"
                        )
                        raise ValidationError(
                            msg, code="deleting_protected", params=params
                        )

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

        defaults["form"] = DeleteProtectedModelForm

        if defaults["fields"] is None and not modelform_defines_fields(
            defaults["form"]
        ):
            defaults["fields"] = forms.ALL_FIELDS

        return inlineformset_factory(self.parent_model, self.model, **defaults)
```
### 41 - django/contrib/admin/options.py:

Start line: 1673, End line: 1709

```python
class ModelAdmin(BaseModelAdmin):

    def get_inline_formsets(self, request, formsets, inline_instances, obj=None):
        # Edit permissions on parent model are required for editable inlines.
        can_edit_parent = (
            self.has_change_permission(request, obj)
            if obj
            else self.has_add_permission(request)
        )
        inline_admin_formsets = []
        for inline, formset in zip(inline_instances, formsets):
            fieldsets = list(inline.get_fieldsets(request, obj))
            readonly = list(inline.get_readonly_fields(request, obj))
            if can_edit_parent:
                has_add_permission = inline.has_add_permission(request, obj)
                has_change_permission = inline.has_change_permission(request, obj)
                has_delete_permission = inline.has_delete_permission(request, obj)
            else:
                # Disable all edit-permissions, and overide formset settings.
                has_add_permission = (
                    has_change_permission
                ) = has_delete_permission = False
                formset.extra = formset.max_num = 0
            has_view_permission = inline.has_view_permission(request, obj)
            prepopulated = dict(inline.get_prepopulated_fields(request, obj))
            inline_admin_formset = helpers.InlineAdminFormSet(
                inline,
                formset,
                fieldsets,
                prepopulated,
                readonly,
                model_admin=self,
                has_add_permission=has_add_permission,
                has_change_permission=has_change_permission,
                has_delete_permission=has_delete_permission,
                has_view_permission=has_view_permission,
            )
            inline_admin_formsets.append(inline_admin_formset)
        return inline_admin_formsets
```
### 43 - django/contrib/admin/options.py:

Start line: 428, End line: 486

```python
class BaseModelAdmin(metaclass=forms.MediaDefiningClass):

    def lookup_allowed(self, lookup, value):
        from django.contrib.admin.filters import SimpleListFilter

        model = self.model
        # Check FKey lookups that are allowed, so that popups produced by
        # ForeignKeyRawIdWidget, on the basis of ForeignKey.limit_choices_to,
        # are allowed to work.
        for fk_lookup in model._meta.related_fkey_lookups:
            # As ``limit_choices_to`` can be a callable, invoke it here.
            if callable(fk_lookup):
                fk_lookup = fk_lookup()
            if (lookup, value) in widgets.url_params_from_lookup_dict(
                fk_lookup
            ).items():
                return True

        relation_parts = []
        prev_field = None
        for part in lookup.split(LOOKUP_SEP):
            try:
                field = model._meta.get_field(part)
            except FieldDoesNotExist:
                # Lookups on nonexistent fields are ok, since they're ignored
                # later.
                break
            # It is allowed to filter on values that would be found from local
            # model anyways. For example, if you filter on employee__department__id,
            # then the id value would be found already from employee__department_id.
            if not prev_field or (
                prev_field.is_relation
                and field not in prev_field.path_infos[-1].target_fields
            ):
                relation_parts.append(part)
            if not getattr(field, "path_infos", None):
                # This is not a relational field, so further parts
                # must be transforms.
                break
            prev_field = field
            model = field.path_infos[-1].to_opts.model

        if len(relation_parts) <= 1:
            # Either a local field filter, or no fields at all.
            return True
        valid_lookups = {self.date_hierarchy}
        for filter_item in self.list_filter:
            if isinstance(filter_item, type) and issubclass(
                filter_item, SimpleListFilter
            ):
                valid_lookups.add(filter_item.parameter_name)
            elif isinstance(filter_item, (list, tuple)):
                valid_lookups.add(filter_item[0])
            else:
                valid_lookups.add(filter_item)

        # Is it a valid relational lookup?
        return not {
            LOOKUP_SEP.join(relation_parts),
            LOOKUP_SEP.join(relation_parts + [part]),
        }.isdisjoint(valid_lookups)
```
### 61 - django/contrib/admin/options.py:

Start line: 2278, End line: 2335

```python
class InlineModelAdmin(BaseModelAdmin):
    """
    Options for inline editing of ``model`` instances.

    Provide ``fk_name`` to specify the attribute name of the ``ForeignKey``
    from ``model`` to its parent. This is required if ``model`` has more than
    one ``ForeignKey`` to its parent.
    """

    model = None
    fk_name = None
    formset = BaseInlineFormSet
    extra = 3
    min_num = None
    max_num = None
    template = None
    verbose_name = None
    verbose_name_plural = None
    can_delete = True
    show_change_link = False
    checks_class = InlineModelAdminChecks
    classes = None

    def __init__(self, parent_model, admin_site):
        self.admin_site = admin_site
        self.parent_model = parent_model
        self.opts = self.model._meta
        self.has_registered_model = admin_site.is_registered(self.model)
        super().__init__()
        if self.verbose_name_plural is None:
            if self.verbose_name is None:
                self.verbose_name_plural = self.opts.verbose_name_plural
            else:
                self.verbose_name_plural = format_lazy("{}s", self.verbose_name)
        if self.verbose_name is None:
            self.verbose_name = self.opts.verbose_name

    @property
    def media(self):
        extra = "" if settings.DEBUG else ".min"
        js = ["vendor/jquery/jquery%s.js" % extra, "jquery.init.js", "inlines.js"]
        if self.filter_vertical or self.filter_horizontal:
            js.extend(["SelectBox.js", "SelectFilter2.js"])
        if self.classes and "collapse" in self.classes:
            js.append("collapse.js")
        return forms.Media(js=["admin/js/%s" % url for url in js])

    def get_extra(self, request, obj=None, **kwargs):
        """Hook for customizing the number of extra inline forms."""
        return self.extra

    def get_min_num(self, request, obj=None, **kwargs):
        """Hook for customizing the min number of inline forms."""
        return self.min_num

    def get_max_num(self, request, obj=None, **kwargs):
        """Hook for customizing the max number of extra inline forms."""
        return self.max_num
```
### 64 - django/contrib/admin/options.py:

Start line: 488, End line: 532

```python
class BaseModelAdmin(metaclass=forms.MediaDefiningClass):

    def to_field_allowed(self, request, to_field):
        """
        Return True if the model associated with this admin should be
        allowed to be referenced by the specified field.
        """
        try:
            field = self.opts.get_field(to_field)
        except FieldDoesNotExist:
            return False

        # Always allow referencing the primary key since it's already possible
        # to get this information from the change view URL.
        if field.primary_key:
            return True

        # Allow reverse relationships to models defining m2m fields if they
        # target the specified field.
        for many_to_many in self.opts.many_to_many:
            if many_to_many.m2m_target_field_name() == to_field:
                return True

        # Make sure at least one of the models registered for this site
        # references this field through a FK or a M2M relationship.
        registered_models = set()
        for model, admin in self.admin_site._registry.items():
            registered_models.add(model)
            for inline in admin.inlines:
                registered_models.add(inline.model)

        related_objects = (
            f
            for f in self.opts.get_fields(include_hidden=True)
            if (f.auto_created and not f.concrete)
        )
        for related_object in related_objects:
            related_model = related_object.related_model
            remote_field = related_object.field.remote_field
            if (
                any(issubclass(model, related_model) for model in registered_models)
                and hasattr(remote_field, "get_related_field")
                and remote_field.get_related_field() == field
            ):
                return True

        return False
```
### 65 - django/contrib/admin/options.py:

Start line: 1909, End line: 1997

```python
class ModelAdmin(BaseModelAdmin):

    @csrf_protect_m
    def changelist_view(self, request, extra_context=None):
        """
        The 'change list' admin view for this model.
        """
        from django.contrib.admin.views.main import ERROR_FLAG

        app_label = self.opts.app_label
        if not self.has_view_or_change_permission(request):
            raise PermissionDenied

        try:
            cl = self.get_changelist_instance(request)
        except IncorrectLookupParameters:
            # Wacky lookup parameters were given, so redirect to the main
            # changelist page, without parameters, and pass an 'invalid=1'
            # parameter via the query string. If wacky parameters were given
            # and the 'invalid=1' parameter was already in the query string,
            # something is screwed up with the database, so display an error
            # page.
            if ERROR_FLAG in request.GET:
                return SimpleTemplateResponse(
                    "admin/invalid_setup.html",
                    {
                        "title": _("Database error"),
                    },
                )
            return HttpResponseRedirect(request.path + "?" + ERROR_FLAG + "=1")

        # If the request was POSTed, this might be a bulk action or a bulk
        # edit. Try to look up an action or confirmation first, but if this
        # isn't an action the POST will fall through to the bulk edit check,
        # below.
        action_failed = False
        selected = request.POST.getlist(helpers.ACTION_CHECKBOX_NAME)

        actions = self.get_actions(request)
        # Actions with no confirmation
        if (
            actions
            and request.method == "POST"
            and "index" in request.POST
            and "_save" not in request.POST
        ):
            if selected:
                response = self.response_action(
                    request, queryset=cl.get_queryset(request)
                )
                if response:
                    return response
                else:
                    action_failed = True
            else:
                msg = _(
                    "Items must be selected in order to perform "
                    "actions on them. No items have been changed."
                )
                self.message_user(request, msg, messages.WARNING)
                action_failed = True

        # Actions with confirmation
        if (
            actions
            and request.method == "POST"
            and helpers.ACTION_CHECKBOX_NAME in request.POST
            and "index" not in request.POST
            and "_save" not in request.POST
        ):
            if selected:
                response = self.response_action(
                    request, queryset=cl.get_queryset(request)
                )
                if response:
                    return response
                else:
                    action_failed = True

        if action_failed:
            # Redirect back to the changelist page to avoid resubmitting the
            # form if the user refreshes the browser or uses the "No, take
            # me back" button on the action confirmation page.
            return HttpResponseRedirect(request.get_full_path())

        # If we're allowing changelist editing, we need to construct a formset
        # for the changelist given all the fields to be edited. Then we'll
        # use the formset to validate/process POSTed data.
        formset = cl.formset = None

        # Handle POSTed bulk-edit data.
        # ... other code
```
### 70 - django/contrib/admin/options.py:

Start line: 1650, End line: 1671

```python
class ModelAdmin(BaseModelAdmin):

    def render_delete_form(self, request, context):
        app_label = self.opts.app_label

        request.current_app = self.admin_site.name
        context.update(
            to_field_var=TO_FIELD_VAR,
            is_popup_var=IS_POPUP_VAR,
            media=self.media,
        )

        return TemplateResponse(
            request,
            self.delete_confirmation_template
            or [
                "admin/{}/{}/delete_confirmation.html".format(
                    app_label, self.opts.model_name
                ),
                "admin/{}/delete_confirmation.html".format(app_label),
                "admin/delete_confirmation.html",
            ],
            context,
        )
```
### 72 - django/contrib/admin/options.py:

Start line: 329, End line: 351

```python
class BaseModelAdmin(metaclass=forms.MediaDefiningClass):

    def get_autocomplete_fields(self, request):
        """
        Return a list of ForeignKey and/or ManyToMany fields which should use
        an autocomplete widget.
        """
        return self.autocomplete_fields

    def get_view_on_site_url(self, obj=None):
        if obj is None or not self.view_on_site:
            return None

        if callable(self.view_on_site):
            return self.view_on_site(obj)
        elif hasattr(obj, "get_absolute_url"):
            # use the ContentType lookup if view_on_site is True
            return reverse(
                "admin:view_on_site",
                kwargs={
                    "content_type_id": get_content_type_for_model(obj).pk,
                    "object_id": obj.pk,
                },
                current_app=self.admin_site.name,
            )
```
### 79 - django/contrib/admin/options.py:

Start line: 2226, End line: 2275

```python
class ModelAdmin(BaseModelAdmin):

    def get_formset_kwargs(self, request, obj, inline, prefix):
        formset_params = {
            "instance": obj,
            "prefix": prefix,
            "queryset": inline.get_queryset(request),
        }
        if request.method == "POST":
            formset_params.update(
                {
                    "data": request.POST.copy(),
                    "files": request.FILES,
                    "save_as_new": "_saveasnew" in request.POST,
                }
            )
        return formset_params

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
                    inline.has_delete_permission(request, obj)
                    and "{}-{}-DELETE".format(formset.prefix, index) in request.POST
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
### 81 - django/contrib/admin/options.py:

Start line: 117, End line: 147

```python
class BaseModelAdmin(metaclass=forms.MediaDefiningClass):
    """Functionality common to both ModelAdmin and InlineAdmin."""

    autocomplete_fields = ()
    raw_id_fields = ()
    fields = None
    exclude = None
    fieldsets = None
    form = forms.ModelForm
    filter_vertical = ()
    filter_horizontal = ()
    radio_fields = {}
    prepopulated_fields = {}
    formfield_overrides = {}
    readonly_fields = ()
    ordering = None
    sortable_by = None
    view_on_site = True
    show_full_result_count = True
    checks_class = BaseModelAdminChecks

    def check(self, **kwargs):
        return self.checks_class().check(self, **kwargs)

    def __init__(self):
        # Merge FORMFIELD_FOR_DBFIELD_DEFAULTS with the formfield_overrides
        # rather than simply overwriting.
        overrides = copy.deepcopy(FORMFIELD_FOR_DBFIELD_DEFAULTS)
        for k, v in self.formfield_overrides.items():
            overrides.setdefault(k, {}).update(v)
        self.formfield_overrides = overrides
```
### 84 - django/contrib/admin/options.py:

Start line: 353, End line: 426

```python
class BaseModelAdmin(metaclass=forms.MediaDefiningClass):

    def get_empty_value_display(self):
        """
        Return the empty_value_display set on ModelAdmin or AdminSite.
        """
        try:
            return mark_safe(self.empty_value_display)
        except AttributeError:
            return mark_safe(self.admin_site.empty_value_display)

    def get_exclude(self, request, obj=None):
        """
        Hook for specifying exclude.
        """
        return self.exclude

    def get_fields(self, request, obj=None):
        """
        Hook for specifying fields.
        """
        if self.fields:
            return self.fields
        # _get_form_for_get_fields() is implemented in subclasses.
        form = self._get_form_for_get_fields(request, obj)
        return [*form.base_fields, *self.get_readonly_fields(request, obj)]

    def get_fieldsets(self, request, obj=None):
        """
        Hook for specifying fieldsets.
        """
        if self.fieldsets:
            return self.fieldsets
        return [(None, {"fields": self.get_fields(request, obj)})]

    def get_inlines(self, request, obj):
        """Hook for specifying custom inlines."""
        return self.inlines

    def get_ordering(self, request):
        """
        Hook for specifying field ordering.
        """
        return self.ordering or ()  # otherwise we might try to *None, which is bad ;)

    def get_readonly_fields(self, request, obj=None):
        """
        Hook for specifying custom readonly fields.
        """
        return self.readonly_fields

    def get_prepopulated_fields(self, request, obj=None):
        """
        Hook for specifying custom prepopulated fields.
        """
        return self.prepopulated_fields

    def get_queryset(self, request):
        """
        Return a QuerySet of all model instances that can be edited by the
        admin site. This is used by changelist_view.
        """
        qs = self.model._default_manager.get_queryset()
        # TODO: this should be handled by some parameter to the ChangeList.
        ordering = self.get_ordering(request)
        if ordering:
            qs = qs.order_by(*ordering)
        return qs

    def get_sortable_by(self, request):
        """Hook for specifying which fields can be sorted in the changelist."""
        return (
            self.sortable_by
            if self.sortable_by is not None
            else self.get_list_display(request)
        )
```
### 93 - django/contrib/admin/options.py:

Start line: 2460, End line: 2495

```python
class InlineModelAdmin(BaseModelAdmin):

    def has_add_permission(self, request, obj):
        if self.opts.auto_created:
            # Auto-created intermediate models don't have their own
            # permissions. The user needs to have the change permission for the
            # related model in order to be able to do anything with the
            # intermediate model.
            return self._has_any_perms_for_target_model(request, ["change"])
        return super().has_add_permission(request)

    def has_change_permission(self, request, obj=None):
        if self.opts.auto_created:
            # Same comment as has_add_permission().
            return self._has_any_perms_for_target_model(request, ["change"])
        return super().has_change_permission(request)

    def has_delete_permission(self, request, obj=None):
        if self.opts.auto_created:
            # Same comment as has_add_permission().
            return self._has_any_perms_for_target_model(request, ["change"])
        return super().has_delete_permission(request, obj)

    def has_view_permission(self, request, obj=None):
        if self.opts.auto_created:
            # Same comment as has_add_permission(). The 'change' permission
            # also implies the 'view' permission.
            return self._has_any_perms_for_target_model(request, ["view", "change"])
        return super().has_view_permission(request)


class StackedInline(InlineModelAdmin):
    template = "admin/edit_inline/stacked.html"


class TabularInline(InlineModelAdmin):
    template = "admin/edit_inline/tabular.html"
```
### 103 - django/contrib/admin/options.py:

Start line: 1135, End line: 1155

```python
class ModelAdmin(BaseModelAdmin):

    def get_search_results(self, request, queryset, search_term):
        # ... other code

        may_have_duplicates = False
        search_fields = self.get_search_fields(request)
        if search_fields and search_term:
            orm_lookups = [
                construct_search(str(search_field)) for search_field in search_fields
            ]
            term_queries = []
            for bit in smart_split(search_term):
                if bit.startswith(('"', "'")) and bit[0] == bit[-1]:
                    bit = unescape_string_literal(bit)
                or_queries = models.Q(
                    *((orm_lookup, bit) for orm_lookup in orm_lookups),
                    _connector=models.Q.OR,
                )
                term_queries.append(or_queries)
            queryset = queryset.filter(models.Q(*term_queries))
            may_have_duplicates |= any(
                lookup_spawns_duplicates(self.opts, search_spec)
                for search_spec in orm_lookups
            )
        return queryset, may_have_duplicates
```
### 116 - django/contrib/admin/options.py:

Start line: 217, End line: 234

```python
class BaseModelAdmin(metaclass=forms.MediaDefiningClass):

    def formfield_for_choice_field(self, db_field, request, **kwargs):
        """
        Get a form Field for a database Field that has declared choices.
        """
        # If the field is named as a radio_field, use a RadioSelect
        if db_field.name in self.radio_fields:
            # Avoid stomping on custom widget/choices arguments.
            if "widget" not in kwargs:
                kwargs["widget"] = widgets.AdminRadioSelect(
                    attrs={
                        "class": get_ul_class(self.radio_fields[db_field.name]),
                    }
                )
            if "choices" not in kwargs:
                kwargs["choices"] = db_field.get_choices(
                    include_blank=db_field.blank, blank_choice=[("", _("None"))]
                )
        return db_field.formfield(**kwargs)
```
### 117 - django/contrib/admin/options.py:

Start line: 866, End line: 879

```python
class ModelAdmin(BaseModelAdmin):

    def get_changelist_form(self, request, **kwargs):
        """
        Return a Form class for use in the Formset on the changelist page.
        """
        defaults = {
            "formfield_callback": partial(self.formfield_for_dbfield, request=request),
            **kwargs,
        }
        if defaults.get("fields") is None and not modelform_defines_fields(
            defaults.get("form")
        ):
            defaults["fields"] = forms.ALL_FIELDS

        return modelform_factory(self.model, **defaults)
```
### 128 - django/contrib/admin/options.py:

Start line: 236, End line: 249

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
                return db_field.remote_field.model._default_manager.using(db).order_by(
                    *ordering
                )
        return None
```
### 133 - django/contrib/admin/options.py:

Start line: 1079, End line: 1098

```python
class ModelAdmin(BaseModelAdmin):

    def get_list_filter(self, request):
        """
        Return a sequence containing the fields to be displayed as filters in
        the right sidebar of the changelist page.
        """
        return self.list_filter

    def get_list_select_related(self, request):
        """
        Return a list of fields to add to the select_related() part of the
        changelist items query.
        """
        return self.list_select_related

    def get_search_fields(self, request):
        """
        Return a sequence containing the fields to be searched whenever
        somebody submits a search query.
        """
        return self.search_fields
```
### 143 - django/contrib/admin/options.py:

Start line: 721, End line: 738

```python
class ModelAdmin(BaseModelAdmin):

    @property
    def urls(self):
        return self.get_urls()

    @property
    def media(self):
        extra = "" if settings.DEBUG else ".min"
        js = [
            "vendor/jquery/jquery%s.js" % extra,
            "jquery.init.js",
            "core.js",
            "admin/RelatedObjectLookups.js",
            "actions.js",
            "urlify.js",
            "prepopulate.js",
            "vendor/xregexp/xregexp%s.js" % extra,
        ]
        return forms.Media(js=["admin/js/%s" % url for url in js])
```
### 155 - django/contrib/admin/options.py:

Start line: 881, End line: 896

```python
class ModelAdmin(BaseModelAdmin):

    def get_changelist_formset(self, request, **kwargs):
        """
        Return a FormSet class for use on the changelist page if list_editable
        is used.
        """
        defaults = {
            "formfield_callback": partial(self.formfield_for_dbfield, request=request),
            **kwargs,
        }
        return modelformset_factory(
            self.model,
            self.get_changelist_form(request),
            extra=0,
            fields=self.list_editable,
            **defaults,
        )
```
### 160 - django/contrib/admin/options.py:

Start line: 898, End line: 942

```python
class ModelAdmin(BaseModelAdmin):

    def get_formsets_with_inlines(self, request, obj=None):
        """
        Yield formsets and the corresponding inlines.
        """
        for inline in self.get_inline_instances(request, obj):
            yield inline.get_formset(request, obj), inline

    def get_paginator(
        self, request, queryset, per_page, orphans=0, allow_empty_first_page=True
    ):
        return self.paginator(queryset, per_page, orphans, allow_empty_first_page)

    def log_addition(self, request, obj, message):
        """
        Log that an object has been successfully added.

        The default implementation creates an admin LogEntry object.
        """
        from django.contrib.admin.models import ADDITION, LogEntry

        return LogEntry.objects.log_action(
            user_id=request.user.pk,
            content_type_id=get_content_type_for_model(obj).pk,
            object_id=obj.pk,
            object_repr=str(obj),
            action_flag=ADDITION,
            change_message=message,
        )

    def log_change(self, request, obj, message):
        """
        Log that an object has been successfully changed.

        The default implementation creates an admin LogEntry object.
        """
        from django.contrib.admin.models import CHANGE, LogEntry

        return LogEntry.objects.log_action(
            user_id=request.user.pk,
            content_type_id=get_content_type_for_model(obj).pk,
            object_id=obj.pk,
            object_repr=str(obj),
            action_flag=CHANGE,
            change_message=message,
        )
```
### 162 - django/contrib/admin/options.py:

Start line: 610, End line: 660

```python
class ModelAdmin(BaseModelAdmin):
    """Encapsulate all admin options and functionality for a given model."""

    list_display = ("__str__",)
    list_display_links = ()
    list_filter = ()
    list_select_related = False
    list_per_page = 100
    list_max_show_all = 200
    list_editable = ()
    search_fields = ()
    search_help_text = None
    date_hierarchy = None
    save_as = False
    save_as_continue = True
    save_on_top = False
    paginator = Paginator
    preserve_filters = True
    inlines = ()

    # Custom templates (designed to be over-ridden in subclasses)
    add_form_template = None
    change_form_template = None
    change_list_template = None
    delete_confirmation_template = None
    delete_selected_confirmation_template = None
    object_history_template = None
    popup_response_template = None

    # Actions
    actions = ()
    action_form = helpers.ActionForm
    actions_on_top = True
    actions_on_bottom = False
    actions_selection_counter = True
    checks_class = ModelAdminChecks

    def __init__(self, model, admin_site):
        self.model = model
        self.opts = model._meta
        self.admin_site = admin_site
        super().__init__()

    def __str__(self):
        return "%s.%s" % (self.opts.app_label, self.__class__.__name__)

    def __repr__(self):
        return (
            f"<{self.__class__.__qualname__}: model={self.model.__qualname__} "
            f"site={self.admin_site!r}>"
        )
```
### 167 - django/contrib/admin/options.py:

Start line: 662, End line: 677

```python
class ModelAdmin(BaseModelAdmin):

    def get_inline_instances(self, request, obj=None):
        inline_instances = []
        for inline_class in self.get_inlines(request, obj):
            inline = inline_class(self.model, self.admin_site)
            if request:
                if not (
                    inline.has_view_or_change_permission(request, obj)
                    or inline.has_add_permission(request, obj)
                    or inline.has_delete_permission(request, obj)
                ):
                    continue
                if not inline.has_add_permission(request, obj):
                    inline.max_num = 0
            inline_instances.append(inline)

        return inline_instances
```
### 168 - django/contrib/admin/options.py:

Start line: 1893, End line: 1907

```python
class ModelAdmin(BaseModelAdmin):

    def _get_list_editable_queryset(self, request, prefix):
        """
        Based on POST data, return a queryset of the objects that were edited
        via list_editable.
        """
        object_pks = self._get_edited_object_pks(request, prefix)
        queryset = self.get_queryset(request)
        validate = queryset.model._meta.pk.to_python
        try:
            for pk in object_pks:
                validate(pk)
        except ValidationError:
            # Disable the optimization if the POST data was tampered with.
            return queryset
        return queryset.filter(pk__in=object_pks)
```
### 174 - django/contrib/admin/options.py:

Start line: 2089, End line: 2164

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
        app_label = self.opts.app_label

        to_field = request.POST.get(TO_FIELD_VAR, request.GET.get(TO_FIELD_VAR))
        if to_field and not self.to_field_allowed(request, to_field):
            raise DisallowedModelAdminToField(
                "The field %s cannot be referenced." % to_field
            )

        obj = self.get_object(request, unquote(object_id), to_field)

        if not self.has_delete_permission(request, obj):
            raise PermissionDenied

        if obj is None:
            return self._get_obj_does_not_exist_redirect(request, self.opts, object_id)

        # Populate deleted_objects, a data structure of all related objects that
        # will also be deleted.
        (
            deleted_objects,
            model_count,
            perms_needed,
            protected,
        ) = self.get_deleted_objects([obj], request)

        if request.POST and not protected:  # The user has confirmed the deletion.
            if perms_needed:
                raise PermissionDenied
            obj_display = str(obj)
            attr = str(to_field) if to_field else self.opts.pk.attname
            obj_id = obj.serializable_value(attr)
            self.log_deletion(request, obj, obj_display)
            self.delete_model(request, obj)

            return self.response_delete(request, obj_display, obj_id)

        object_name = str(self.opts.verbose_name)

        if perms_needed or protected:
            title = _("Cannot delete %(name)s") % {"name": object_name}
        else:
            title = _("Are you sure?")

        context = {
            **self.admin_site.each_context(request),
            "title": title,
            "subtitle": None,
            "object_name": object_name,
            "object": obj,
            "deleted_objects": deleted_objects,
            "model_count": dict(model_count).items(),
            "perms_lacking": perms_needed,
            "protected": protected,
            "opts": self.opts,
            "app_label": app_label,
            "preserved_filters": self.get_preserved_filters(request),
            "is_popup": IS_POPUP_VAR in request.POST or IS_POPUP_VAR in request.GET,
            "to_field": to_field,
            **(extra_context or {}),
        }

        return self.render_delete_form(request, context)
```
