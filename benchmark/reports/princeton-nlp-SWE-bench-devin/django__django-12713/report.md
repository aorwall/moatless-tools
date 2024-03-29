# django__django-12713

| **django/django** | `5b884d45ac5b76234eca614d90c83b347294c332` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 496 |
| **Any found context length** | 496 |
| **Avg pos** | 2.0 |
| **Min pos** | 2 |
| **Max pos** | 2 |
| **Top file pos** | 2 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/contrib/admin/options.py b/django/contrib/admin/options.py
--- a/django/contrib/admin/options.py
+++ b/django/contrib/admin/options.py
@@ -249,17 +249,25 @@ def formfield_for_manytomany(self, db_field, request, **kwargs):
             return None
         db = kwargs.get('using')
 
-        autocomplete_fields = self.get_autocomplete_fields(request)
-        if db_field.name in autocomplete_fields:
-            kwargs['widget'] = AutocompleteSelectMultiple(db_field.remote_field, self.admin_site, using=db)
-        elif db_field.name in self.raw_id_fields:
-            kwargs['widget'] = widgets.ManyToManyRawIdWidget(db_field.remote_field, self.admin_site, using=db)
-        elif db_field.name in [*self.filter_vertical, *self.filter_horizontal]:
-            kwargs['widget'] = widgets.FilteredSelectMultiple(
-                db_field.verbose_name,
-                db_field.name in self.filter_vertical
-            )
-
+        if 'widget' not in kwargs:
+            autocomplete_fields = self.get_autocomplete_fields(request)
+            if db_field.name in autocomplete_fields:
+                kwargs['widget'] = AutocompleteSelectMultiple(
+                    db_field.remote_field,
+                    self.admin_site,
+                    using=db,
+                )
+            elif db_field.name in self.raw_id_fields:
+                kwargs['widget'] = widgets.ManyToManyRawIdWidget(
+                    db_field.remote_field,
+                    self.admin_site,
+                    using=db,
+                )
+            elif db_field.name in [*self.filter_vertical, *self.filter_horizontal]:
+                kwargs['widget'] = widgets.FilteredSelectMultiple(
+                    db_field.verbose_name,
+                    db_field.name in self.filter_vertical
+                )
         if 'queryset' not in kwargs:
             queryset = self.get_field_queryset(db, db_field, request)
             if queryset is not None:

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/contrib/admin/options.py | 252 | 262 | 2 | 2 | 496


## Problem Statement

```
Allow overridding widget in formfield_for_manytomany().
Description
	 
		(last modified by Mariusz Felisiak)
	 
It does not work when I set widget param to function formfield_for_manytomany().
This is different from the formfield_for_foreignkey() function.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 django/contrib/gis/admin/options.py | 52 | 63| 139 | 139 | 1196 | 
| **-> 2 <-** | **2 django/contrib/admin/options.py** | 242 | 274| 357 | 496 | 19684 | 
| 3 | 3 django/db/models/fields/related.py | 1198 | 1229| 180 | 676 | 33516 | 
| 4 | 4 django/contrib/admin/widgets.py | 195 | 221| 216 | 892 | 37337 | 
| 5 | 4 django/contrib/admin/widgets.py | 273 | 308| 382 | 1274 | 37337 | 
| 6 | 5 django/forms/widgets.py | 763 | 786| 204 | 1478 | 45414 | 
| 7 | 5 django/contrib/admin/widgets.py | 120 | 159| 349 | 1827 | 45414 | 
| 8 | 6 django/forms/boundfield.py | 80 | 98| 174 | 2001 | 47570 | 
| 9 | 6 django/contrib/admin/widgets.py | 161 | 192| 243 | 2244 | 47570 | 
| 10 | 6 django/forms/boundfield.py | 36 | 51| 149 | 2393 | 47570 | 
| 11 | 6 django/forms/widgets.py | 816 | 845| 269 | 2662 | 47570 | 
| 12 | **6 django/contrib/admin/options.py** | 130 | 185| 604 | 3266 | 47570 | 
| 13 | **6 django/contrib/admin/options.py** | 218 | 240| 232 | 3498 | 47570 | 
| 14 | **6 django/contrib/admin/options.py** | 1523 | 1609| 760 | 4258 | 47570 | 
| 15 | 6 django/forms/boundfield.py | 236 | 275| 273 | 4531 | 47570 | 
| 16 | 7 django/contrib/auth/forms.py | 53 | 78| 162 | 4693 | 50779 | 
| 17 | 7 django/db/models/fields/related.py | 980 | 991| 128 | 4821 | 50779 | 
| 18 | 7 django/db/models/fields/related.py | 1659 | 1693| 266 | 5087 | 50779 | 
| 19 | 7 django/forms/widgets.py | 669 | 700| 261 | 5348 | 50779 | 
| 20 | 8 django/contrib/postgres/forms/array.py | 133 | 165| 265 | 5613 | 52373 | 
| 21 | 9 django/contrib/postgres/forms/ranges.py | 31 | 78| 325 | 5938 | 53050 | 
| 22 | 9 django/forms/widgets.py | 337 | 371| 278 | 6216 | 53050 | 
| 23 | 9 django/forms/widgets.py | 741 | 760| 144 | 6360 | 53050 | 
| 24 | 9 django/forms/widgets.py | 980 | 1014| 391 | 6751 | 53050 | 
| 25 | 9 django/db/models/fields/related.py | 1231 | 1348| 963 | 7714 | 53050 | 
| 26 | **9 django/contrib/admin/options.py** | 1 | 95| 756 | 8470 | 53050 | 
| 27 | 9 django/forms/widgets.py | 374 | 394| 138 | 8608 | 53050 | 
| 28 | 9 django/contrib/admin/widgets.py | 224 | 271| 423 | 9031 | 53050 | 
| 29 | 10 django/forms/fields.py | 45 | 126| 773 | 9804 | 62063 | 
| 30 | 11 django/forms/models.py | 310 | 349| 387 | 10191 | 73784 | 
| 31 | 11 django/contrib/postgres/forms/array.py | 105 | 131| 219 | 10410 | 73784 | 
| 32 | 11 django/forms/models.py | 1370 | 1397| 209 | 10619 | 73784 | 
| 33 | 11 django/contrib/auth/forms.py | 32 | 50| 161 | 10780 | 73784 | 
| 34 | 11 django/forms/widgets.py | 1041 | 1060| 144 | 10924 | 73784 | 
| 35 | 11 django/db/models/fields/related.py | 1424 | 1465| 418 | 11342 | 73784 | 
| 36 | 11 django/forms/widgets.py | 1062 | 1085| 248 | 11590 | 73784 | 
| 37 | 11 django/forms/widgets.py | 703 | 738| 237 | 11827 | 73784 | 
| 38 | 11 django/db/models/fields/related.py | 255 | 282| 269 | 12096 | 73784 | 
| 39 | 11 django/forms/fields.py | 859 | 898| 298 | 12394 | 73784 | 
| 40 | 11 django/db/models/fields/related.py | 1350 | 1422| 616 | 13010 | 73784 | 
| 41 | 11 django/contrib/postgres/forms/ranges.py | 1 | 28| 201 | 13211 | 73784 | 
| 42 | 11 django/forms/widgets.py | 194 | 276| 622 | 13833 | 73784 | 
| 43 | 12 django/contrib/admin/helpers.py | 366 | 382| 138 | 13971 | 76975 | 
| 44 | 12 django/contrib/postgres/forms/array.py | 168 | 195| 226 | 14197 | 76975 | 
| 45 | **12 django/contrib/admin/options.py** | 98 | 128| 223 | 14420 | 76975 | 
| 46 | 12 django/forms/fields.py | 351 | 368| 160 | 14580 | 76975 | 
| 47 | 12 django/db/models/fields/related.py | 1602 | 1639| 484 | 15064 | 76975 | 
| 48 | 12 django/forms/models.py | 208 | 277| 616 | 15680 | 76975 | 
| 49 | 13 django/db/models/base.py | 1393 | 1448| 491 | 16171 | 92618 | 
| 50 | 14 django/db/backends/oracle/schema.py | 79 | 123| 583 | 16754 | 94373 | 
| 51 | 15 django/contrib/auth/admin.py | 25 | 37| 128 | 16882 | 96099 | 
| 52 | 15 django/forms/fields.py | 173 | 205| 280 | 17162 | 96099 | 
| 53 | 15 django/contrib/admin/widgets.py | 1 | 46| 333 | 17495 | 96099 | 
| 54 | 15 django/forms/widgets.py | 443 | 461| 195 | 17690 | 96099 | 
| 55 | 16 django/contrib/gis/forms/widgets.py | 44 | 73| 271 | 17961 | 96954 | 
| 56 | 17 django/db/migrations/operations/fields.py | 237 | 247| 146 | 18107 | 100010 | 
| 57 | 17 django/contrib/postgres/forms/array.py | 62 | 102| 248 | 18355 | 100010 | 
| 58 | 17 django/forms/widgets.py | 1 | 42| 299 | 18654 | 100010 | 
| 59 | 17 django/db/models/fields/related.py | 1641 | 1657| 286 | 18940 | 100010 | 
| 60 | 18 django/db/models/fields/related_descriptors.py | 1016 | 1043| 334 | 19274 | 110405 | 
| 61 | 18 django/forms/fields.py | 288 | 320| 228 | 19502 | 110405 | 
| 62 | 19 django/forms/forms.py | 429 | 451| 195 | 19697 | 114419 | 
| 63 | 19 django/contrib/admin/widgets.py | 311 | 323| 114 | 19811 | 114419 | 
| 64 | 19 django/forms/models.py | 195 | 205| 131 | 19942 | 114419 | 
| 65 | 19 django/forms/widgets.py | 546 | 580| 277 | 20219 | 114419 | 
| 66 | 19 django/db/models/fields/related_descriptors.py | 1165 | 1206| 392 | 20611 | 114419 | 
| 67 | 19 django/forms/fields.py | 540 | 556| 180 | 20791 | 114419 | 
| 68 | 19 django/forms/models.py | 383 | 411| 240 | 21031 | 114419 | 
| 69 | 19 django/db/models/fields/related.py | 1023 | 1070| 368 | 21399 | 114419 | 
| 70 | 19 django/forms/fields.py | 242 | 259| 164 | 21563 | 114419 | 
| 71 | 19 django/contrib/admin/widgets.py | 49 | 70| 168 | 21731 | 114419 | 
| 72 | 19 django/db/migrations/operations/fields.py | 214 | 235| 211 | 21942 | 114419 | 
| 73 | 19 django/db/migrations/operations/fields.py | 344 | 371| 291 | 22233 | 114419 | 
| 74 | 19 django/forms/widgets.py | 789 | 814| 214 | 22447 | 114419 | 
| 75 | 19 django/db/migrations/operations/fields.py | 298 | 342| 430 | 22877 | 114419 | 
| 76 | 19 django/db/models/fields/related_descriptors.py | 309 | 323| 182 | 23059 | 114419 | 
| 77 | 19 django/forms/widgets.py | 616 | 635| 189 | 23248 | 114419 | 
| 78 | 19 django/db/models/fields/related.py | 1073 | 1117| 407 | 23655 | 114419 | 
| 79 | 20 django/contrib/postgres/forms/jsonb.py | 1 | 63| 345 | 24000 | 114764 | 
| 80 | 20 django/forms/forms.py | 378 | 398| 210 | 24210 | 114764 | 
| 81 | 20 django/contrib/gis/admin/options.py | 65 | 78| 150 | 24360 | 114764 | 
| 82 | 20 django/contrib/admin/helpers.py | 192 | 220| 276 | 24636 | 114764 | 
| 83 | **20 django/contrib/admin/options.py** | 1610 | 1634| 279 | 24915 | 114764 | 
| 84 | 21 django/contrib/admin/utils.py | 287 | 305| 175 | 25090 | 118881 | 
| 85 | 21 django/forms/widgets.py | 429 | 441| 122 | 25212 | 118881 | 
| 86 | 22 django/db/models/fields/__init__.py | 367 | 393| 199 | 25411 | 136468 | 
| 87 | 23 django/contrib/postgres/fields/array.py | 179 | 198| 143 | 25554 | 138549 | 
| 88 | 23 django/contrib/admin/widgets.py | 347 | 373| 328 | 25882 | 138549 | 
| 89 | 23 django/db/models/fields/related_descriptors.py | 946 | 968| 218 | 26100 | 138549 | 
| 90 | 23 django/db/models/fields/related_descriptors.py | 970 | 987| 190 | 26290 | 138549 | 
| 91 | 23 django/forms/models.py | 953 | 986| 367 | 26657 | 138549 | 
| 92 | 24 django/contrib/admin/filters.py | 229 | 244| 196 | 26853 | 142642 | 
| 93 | 24 django/forms/widgets.py | 637 | 666| 238 | 27091 | 142642 | 
| 94 | 24 django/db/models/fields/related.py | 190 | 254| 673 | 27764 | 142642 | 
| 95 | 25 django/contrib/gis/forms/fields.py | 1 | 31| 258 | 28022 | 143559 | 
| 96 | 25 django/db/models/fields/related.py | 746 | 764| 222 | 28244 | 143559 | 
| 97 | 25 django/forms/fields.py | 128 | 171| 290 | 28534 | 143559 | 
| 98 | 25 django/db/models/fields/related_descriptors.py | 1118 | 1163| 484 | 29018 | 143559 | 
| 99 | 25 django/contrib/admin/helpers.py | 33 | 67| 230 | 29248 | 143559 | 
| 100 | 25 django/db/models/fields/related_descriptors.py | 672 | 730| 548 | 29796 | 143559 | 
| 101 | 26 django/db/backends/base/schema.py | 889 | 908| 296 | 30092 | 154910 | 
| 102 | 26 django/forms/models.py | 1316 | 1331| 131 | 30223 | 154910 | 
| 103 | **26 django/contrib/admin/options.py** | 2069 | 2121| 451 | 30674 | 154910 | 
| 104 | 27 django/contrib/gis/admin/widgets.py | 81 | 118| 344 | 31018 | 155874 | 
| 105 | 27 django/forms/forms.py | 152 | 166| 125 | 31143 | 155874 | 
| 106 | **27 django/contrib/admin/options.py** | 367 | 419| 504 | 31647 | 155874 | 
| 107 | 28 django/db/migrations/operations/models.py | 577 | 593| 215 | 31862 | 162448 | 
| 108 | 28 django/forms/boundfield.py | 1 | 34| 242 | 32104 | 162448 | 
| 109 | 28 django/contrib/postgres/fields/array.py | 53 | 75| 172 | 32276 | 162448 | 
| 110 | 28 django/contrib/admin/widgets.py | 376 | 394| 153 | 32429 | 162448 | 
| 111 | **28 django/contrib/admin/options.py** | 187 | 203| 171 | 32600 | 162448 | 
| 112 | 28 django/forms/fields.py | 770 | 829| 416 | 33016 | 162448 | 
| 113 | 29 django/db/models/fields/reverse_related.py | 136 | 154| 172 | 33188 | 164591 | 
| 114 | 29 django/forms/widgets.py | 847 | 890| 315 | 33503 | 164591 | 
| 115 | 29 django/db/migrations/operations/fields.py | 249 | 267| 157 | 33660 | 164591 | 
| 116 | 29 django/db/models/fields/related.py | 909 | 929| 178 | 33838 | 164591 | 
| 117 | 29 django/forms/forms.py | 453 | 500| 382 | 34220 | 164591 | 
| 118 | 29 django/forms/models.py | 413 | 443| 243 | 34463 | 164591 | 
| 119 | 29 django/db/models/fields/related.py | 1467 | 1507| 399 | 34862 | 164591 | 
| 120 | 29 django/contrib/admin/helpers.py | 355 | 364| 134 | 34996 | 164591 | 
| 121 | 29 django/forms/fields.py | 901 | 934| 235 | 35231 | 164591 | 
| 122 | 30 django/db/backends/mysql/schema.py | 88 | 98| 138 | 35369 | 166087 | 
| 123 | 30 django/db/models/fields/related.py | 401 | 419| 165 | 35534 | 166087 | 
| 124 | 30 django/db/models/fields/related.py | 127 | 154| 202 | 35736 | 166087 | 
| 125 | 30 django/contrib/admin/widgets.py | 444 | 472| 203 | 35939 | 166087 | 
| 126 | 31 django/contrib/contenttypes/fields.py | 20 | 108| 571 | 36510 | 171520 | 
| 127 | 32 django/db/migrations/autodetector.py | 907 | 988| 876 | 37386 | 183258 | 
| 128 | 32 django/db/models/fields/__init__.py | 2342 | 2391| 311 | 37697 | 183258 | 
| 129 | 32 django/db/backends/base/schema.py | 526 | 565| 470 | 38167 | 183258 | 
| 130 | 32 django/contrib/gis/forms/widgets.py | 76 | 118| 304 | 38471 | 183258 | 
| 131 | 32 django/db/models/fields/related_descriptors.py | 120 | 154| 405 | 38876 | 183258 | 
| 132 | 32 django/contrib/contenttypes/fields.py | 430 | 451| 248 | 39124 | 183258 | 
| 133 | 32 django/db/models/fields/related.py | 171 | 188| 166 | 39290 | 183258 | 
| 134 | 32 django/contrib/admin/widgets.py | 326 | 344| 168 | 39458 | 183258 | 
| 135 | 32 django/db/models/fields/__init__.py | 961 | 977| 176 | 39634 | 183258 | 
| 136 | 32 django/db/models/fields/related.py | 1535 | 1552| 184 | 39818 | 183258 | 
| 137 | 33 django/contrib/contenttypes/admin.py | 1 | 80| 615 | 40433 | 184283 | 
| 138 | 33 django/contrib/admin/filters.py | 422 | 429| 107 | 40540 | 184283 | 
| 139 | 33 django/forms/forms.py | 111 | 131| 177 | 40717 | 184283 | 
| 140 | 33 django/contrib/admin/filters.py | 264 | 276| 149 | 40866 | 184283 | 
| 141 | 33 django/db/models/fields/related_descriptors.py | 989 | 1015| 248 | 41114 | 184283 | 
| 142 | 34 django/db/backends/sqlite3/schema.py | 367 | 413| 444 | 41558 | 188247 | 
| 143 | 34 django/forms/widgets.py | 507 | 543| 331 | 41889 | 188247 | 
| 144 | 35 django/contrib/admin/views/main.py | 1 | 45| 324 | 42213 | 192545 | 
| 145 | 35 django/contrib/admin/helpers.py | 123 | 149| 220 | 42433 | 192545 | 
| 146 | **35 django/contrib/admin/options.py** | 421 | 464| 350 | 42783 | 192545 | 
| 147 | 35 django/db/models/fields/related.py | 487 | 507| 138 | 42921 | 192545 | 
| 148 | 35 django/db/migrations/operations/fields.py | 169 | 187| 232 | 43153 | 192545 | 
| 149 | 35 django/contrib/admin/filters.py | 371 | 395| 294 | 43447 | 192545 | 
| 150 | 35 django/forms/widgets.py | 464 | 504| 275 | 43722 | 192545 | 
| 151 | 35 django/forms/fields.py | 579 | 604| 243 | 43965 | 192545 | 
| 152 | 35 django/contrib/admin/helpers.py | 92 | 120| 249 | 44214 | 192545 | 
| 153 | 35 django/db/migrations/operations/fields.py | 85 | 95| 125 | 44339 | 192545 | 
| 154 | 35 django/forms/forms.py | 53 | 109| 525 | 44864 | 192545 | 
| 155 | 35 django/contrib/postgres/forms/array.py | 197 | 235| 271 | 45135 | 192545 | 
| 156 | 35 django/forms/fields.py | 1133 | 1166| 293 | 45428 | 192545 | 
| 157 | 35 django/contrib/admin/helpers.py | 247 | 270| 235 | 45663 | 192545 | 
| 158 | 35 django/contrib/postgres/forms/ranges.py | 81 | 103| 149 | 45812 | 192545 | 
| 159 | 35 django/db/models/fields/related_descriptors.py | 1076 | 1087| 138 | 45950 | 192545 | 
| 160 | 36 django/core/serializers/xml_serializer.py | 111 | 147| 330 | 46280 | 195939 | 
| 161 | 36 django/contrib/admin/filters.py | 162 | 207| 427 | 46707 | 195939 | 
| 162 | **36 django/contrib/admin/options.py** | 1110 | 1155| 482 | 47189 | 195939 | 
| 163 | 36 django/contrib/admin/helpers.py | 70 | 89| 167 | 47356 | 195939 | 


## Patch

```diff
diff --git a/django/contrib/admin/options.py b/django/contrib/admin/options.py
--- a/django/contrib/admin/options.py
+++ b/django/contrib/admin/options.py
@@ -249,17 +249,25 @@ def formfield_for_manytomany(self, db_field, request, **kwargs):
             return None
         db = kwargs.get('using')
 
-        autocomplete_fields = self.get_autocomplete_fields(request)
-        if db_field.name in autocomplete_fields:
-            kwargs['widget'] = AutocompleteSelectMultiple(db_field.remote_field, self.admin_site, using=db)
-        elif db_field.name in self.raw_id_fields:
-            kwargs['widget'] = widgets.ManyToManyRawIdWidget(db_field.remote_field, self.admin_site, using=db)
-        elif db_field.name in [*self.filter_vertical, *self.filter_horizontal]:
-            kwargs['widget'] = widgets.FilteredSelectMultiple(
-                db_field.verbose_name,
-                db_field.name in self.filter_vertical
-            )
-
+        if 'widget' not in kwargs:
+            autocomplete_fields = self.get_autocomplete_fields(request)
+            if db_field.name in autocomplete_fields:
+                kwargs['widget'] = AutocompleteSelectMultiple(
+                    db_field.remote_field,
+                    self.admin_site,
+                    using=db,
+                )
+            elif db_field.name in self.raw_id_fields:
+                kwargs['widget'] = widgets.ManyToManyRawIdWidget(
+                    db_field.remote_field,
+                    self.admin_site,
+                    using=db,
+                )
+            elif db_field.name in [*self.filter_vertical, *self.filter_horizontal]:
+                kwargs['widget'] = widgets.FilteredSelectMultiple(
+                    db_field.verbose_name,
+                    db_field.name in self.filter_vertical
+                )
         if 'queryset' not in kwargs:
             queryset = self.get_field_queryset(db, db_field, request)
             if queryset is not None:

```

## Test Patch

```diff
diff --git a/tests/admin_widgets/tests.py b/tests/admin_widgets/tests.py
--- a/tests/admin_widgets/tests.py
+++ b/tests/admin_widgets/tests.py
@@ -14,7 +14,9 @@
 from django.contrib.auth.models import User
 from django.core.files.storage import default_storage
 from django.core.files.uploadedfile import SimpleUploadedFile
-from django.db.models import CharField, DateField, DateTimeField, UUIDField
+from django.db.models import (
+    CharField, DateField, DateTimeField, ManyToManyField, UUIDField,
+)
 from django.test import SimpleTestCase, TestCase, override_settings
 from django.urls import reverse
 from django.utils import translation
@@ -138,6 +140,21 @@ class BandAdmin(admin.ModelAdmin):
         self.assertEqual(f2.widget.attrs['maxlength'], '20')
         self.assertEqual(f2.widget.attrs['size'], '10')
 
+    def test_formfield_overrides_m2m_filter_widget(self):
+        """
+        The autocomplete_fields, raw_id_fields, filter_vertical, and
+        filter_horizontal widgets for ManyToManyFields may be overridden by
+        specifying a widget in formfield_overrides.
+        """
+        class BandAdmin(admin.ModelAdmin):
+            filter_vertical = ['members']
+            formfield_overrides = {
+                ManyToManyField: {'widget': forms.CheckboxSelectMultiple},
+            }
+        ma = BandAdmin(Band, admin.site)
+        field = ma.formfield_for_dbfield(Band._meta.get_field('members'), request=None)
+        self.assertIsInstance(field.widget.widget, forms.CheckboxSelectMultiple)
+
     def test_formfield_overrides_for_datetime_field(self):
         """
         Overriding the widget for DateTimeField doesn't overrides the default

```


## Code snippets

### 1 - django/contrib/gis/admin/options.py:

Start line: 52, End line: 63

```python
class GeoModelAdmin(ModelAdmin):

    def formfield_for_dbfield(self, db_field, request, **kwargs):
        """
        Overloaded from ModelAdmin so that an OpenLayersWidget is used
        for viewing/editing 2D GeometryFields (OpenLayers 2 does not support
        3D editing).
        """
        if isinstance(db_field, models.GeometryField) and db_field.dim < 3:
            # Setting the widget with the newly defined widget.
            kwargs['widget'] = self.get_map_widget(db_field)
            return db_field.formfield(**kwargs)
        else:
            return super().formfield_for_dbfield(db_field, request, **kwargs)
```
### 2 - django/contrib/admin/options.py:

Start line: 242, End line: 274

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
        db = kwargs.get('using')

        autocomplete_fields = self.get_autocomplete_fields(request)
        if db_field.name in autocomplete_fields:
            kwargs['widget'] = AutocompleteSelectMultiple(db_field.remote_field, self.admin_site, using=db)
        elif db_field.name in self.raw_id_fields:
            kwargs['widget'] = widgets.ManyToManyRawIdWidget(db_field.remote_field, self.admin_site, using=db)
        elif db_field.name in [*self.filter_vertical, *self.filter_horizontal]:
            kwargs['widget'] = widgets.FilteredSelectMultiple(
                db_field.verbose_name,
                db_field.name in self.filter_vertical
            )

        if 'queryset' not in kwargs:
            queryset = self.get_field_queryset(db, db_field, request)
            if queryset is not None:
                kwargs['queryset'] = queryset

        form_field = db_field.formfield(**kwargs)
        if (isinstance(form_field.widget, SelectMultiple) and
                not isinstance(form_field.widget, (CheckboxSelectMultiple, AutocompleteSelectMultiple))):
            msg = _('Hold down “Control”, or “Command” on a Mac, to select more than one.')
            help_text = form_field.help_text
            form_field.help_text = format_lazy('{} {}', help_text, msg) if help_text else msg
        return form_field
```
### 3 - django/db/models/fields/related.py:

Start line: 1198, End line: 1229

```python
class ManyToManyField(RelatedField):

    def _check_ignored_options(self, **kwargs):
        warnings = []

        if self.has_null_arg:
            warnings.append(
                checks.Warning(
                    'null has no effect on ManyToManyField.',
                    obj=self,
                    id='fields.W340',
                )
            )

        if self._validators:
            warnings.append(
                checks.Warning(
                    'ManyToManyField does not support validators.',
                    obj=self,
                    id='fields.W341',
                )
            )
        if (self.remote_field.limit_choices_to and self.remote_field.through and
                not self.remote_field.through._meta.auto_created):
            warnings.append(
                checks.Warning(
                    'limit_choices_to has no effect on ManyToManyField '
                    'with a through model.',
                    obj=self,
                    id='fields.W343',
                )
            )

        return warnings
```
### 4 - django/contrib/admin/widgets.py:

Start line: 195, End line: 221

```python
class ManyToManyRawIdWidget(ForeignKeyRawIdWidget):
    """
    A Widget for displaying ManyToMany ids in the "raw_id" interface rather than
    in a <select multiple> box.
    """
    template_name = 'admin/widgets/many_to_many_raw_id.html'

    def get_context(self, name, value, attrs):
        context = super().get_context(name, value, attrs)
        if self.rel.model in self.admin_site._registry:
            # The related object is registered with the same AdminSite
            context['widget']['attrs']['class'] = 'vManyToManyRawIdAdminField'
        return context

    def url_parameters(self):
        return self.base_url_parameters()

    def label_and_url_for_value(self, value):
        return '', ''

    def value_from_datadict(self, data, files, name):
        value = data.get(name)
        if value:
            return value.split(',')

    def format_value(self, value):
        return ','.join(str(v) for v in value) if value else ''
```
### 5 - django/contrib/admin/widgets.py:

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
### 6 - django/forms/widgets.py:

Start line: 763, End line: 786

```python
class CheckboxSelectMultiple(ChoiceWidget):
    allow_multiple_selected = True
    input_type = 'checkbox'
    template_name = 'django/forms/widgets/checkbox_select.html'
    option_template_name = 'django/forms/widgets/checkbox_option.html'

    def use_required_attribute(self, initial):
        # Don't use the 'required' attribute because browser validation would
        # require all checkboxes to be checked instead of at least one.
        return False

    def value_omitted_from_data(self, data, files, name):
        # HTML checkboxes don't appear in POST data if not checked, so it's
        # never known if the value is actually omitted.
        return False

    def id_for_label(self, id_, index=None):
        """"
        Don't include for="field_0" in <label> because clicking such a label
        would toggle the first checkbox.
        """
        if index is None:
            return ''
        return super().id_for_label(id_, index)
```
### 7 - django/contrib/admin/widgets.py:

Start line: 120, End line: 159

```python
class ForeignKeyRawIdWidget(forms.TextInput):
    """
    A Widget for displaying ForeignKeys in the "raw_id" interface rather than
    in a <select> box.
    """
    template_name = 'admin/widgets/foreign_key_raw_id.html'

    def __init__(self, rel, admin_site, attrs=None, using=None):
        self.rel = rel
        self.admin_site = admin_site
        self.db = using
        super().__init__(attrs)

    def get_context(self, name, value, attrs):
        context = super().get_context(name, value, attrs)
        rel_to = self.rel.model
        if rel_to in self.admin_site._registry:
            # The related object is registered with the same AdminSite
            related_url = reverse(
                'admin:%s_%s_changelist' % (
                    rel_to._meta.app_label,
                    rel_to._meta.model_name,
                ),
                current_app=self.admin_site.name,
            )

            params = self.url_parameters()
            if params:
                related_url += '?' + '&amp;'.join('%s=%s' % (k, v) for k, v in params.items())
            context['related_url'] = mark_safe(related_url)
            context['link_title'] = _('Lookup')
            # The JavaScript code looks for this class.
            context['widget']['attrs'].setdefault('class', 'vForeignKeyRawIdAdminField')
        else:
            context['related_url'] = None
        if context['widget']['value']:
            context['link_label'], context['link_url'] = self.label_and_url_for_value(value)
        else:
            context['link_label'] = None
        return context
```
### 8 - django/forms/boundfield.py:

Start line: 80, End line: 98

```python
@html_safe
class BoundField:

    def as_widget(self, widget=None, attrs=None, only_initial=False):
        """
        Render the field by rendering the passed widget, adding any HTML
        attributes passed as attrs. If a widget isn't specified, use the
        field's default widget.
        """
        widget = widget or self.field.widget
        if self.field.localize:
            widget.is_localized = True
        attrs = attrs or {}
        attrs = self.build_widget_attrs(attrs, widget)
        if self.auto_id and 'id' not in widget.attrs:
            attrs.setdefault('id', self.html_initial_id if only_initial else self.auto_id)
        return widget.render(
            name=self.html_initial_name if only_initial else self.html_name,
            value=self.value(),
            attrs=attrs,
            renderer=self.form.renderer,
        )
```
### 9 - django/contrib/admin/widgets.py:

Start line: 161, End line: 192

```python
class ForeignKeyRawIdWidget(forms.TextInput):

    def base_url_parameters(self):
        limit_choices_to = self.rel.limit_choices_to
        if callable(limit_choices_to):
            limit_choices_to = limit_choices_to()
        return url_params_from_lookup_dict(limit_choices_to)

    def url_parameters(self):
        from django.contrib.admin.views.main import TO_FIELD_VAR
        params = self.base_url_parameters()
        params.update({TO_FIELD_VAR: self.rel.get_related_field().name})
        return params

    def label_and_url_for_value(self, value):
        key = self.rel.get_related_field().name
        try:
            obj = self.rel.model._default_manager.using(self.db).get(**{key: value})
        except (ValueError, self.rel.model.DoesNotExist, ValidationError):
            return '', ''

        try:
            url = reverse(
                '%s:%s_%s_change' % (
                    self.admin_site.name,
                    obj._meta.app_label,
                    obj._meta.object_name.lower(),
                ),
                args=(obj.pk,)
            )
        except NoReverseMatch:
            url = ''  # Admin not registered for target model.

        return Truncator(obj).words(14), url
```
### 10 - django/forms/boundfield.py:

Start line: 36, End line: 51

```python
@html_safe
class BoundField:

    @cached_property
    def subwidgets(self):
        """
        Most widgets yield a single subwidget, but others like RadioSelect and
        CheckboxSelectMultiple produce one subwidget for each choice.

        This property is cached so that only one database query occurs when
        rendering ModelChoiceFields.
        """
        id_ = self.field.widget.attrs.get('id') or self.auto_id
        attrs = {'id': id_} if id_ else {}
        attrs = self.build_widget_attrs(attrs)
        return [
            BoundWidget(self.field.widget, widget, self.form.renderer)
            for widget in self.field.widget.subwidgets(self.html_name, self.value(), attrs=attrs)
        ]
```
### 12 - django/contrib/admin/options.py:

Start line: 130, End line: 185

```python
class BaseModelAdmin(metaclass=forms.MediaDefiningClass):

    def formfield_for_dbfield(self, db_field, request, **kwargs):
        """
        Hook for specifying the form Field instance for a given database Field
        instance.

        If kwargs are given, they're passed to the form Field's constructor.
        """
        # If the field specifies choices, we don't need to look for special
        # admin widgets - we just need to use a select widget of some kind.
        if db_field.choices:
            return self.formfield_for_choice_field(db_field, request, **kwargs)

        # ForeignKey or ManyToManyFields
        if isinstance(db_field, (models.ForeignKey, models.ManyToManyField)):
            # Combine the field kwargs with any options for formfield_overrides.
            # Make sure the passed in **kwargs override anything in
            # formfield_overrides because **kwargs is more specific, and should
            # always win.
            if db_field.__class__ in self.formfield_overrides:
                kwargs = {**self.formfield_overrides[db_field.__class__], **kwargs}

            # Get the correct formfield.
            if isinstance(db_field, models.ForeignKey):
                formfield = self.formfield_for_foreignkey(db_field, request, **kwargs)
            elif isinstance(db_field, models.ManyToManyField):
                formfield = self.formfield_for_manytomany(db_field, request, **kwargs)

            # For non-raw_id fields, wrap the widget with a wrapper that adds
            # extra HTML -- the "add other" interface -- to the end of the
            # rendered output. formfield can be None if it came from a
            # OneToOneField with parent_link=True or a M2M intermediary.
            if formfield and db_field.name not in self.raw_id_fields:
                related_modeladmin = self.admin_site._registry.get(db_field.remote_field.model)
                wrapper_kwargs = {}
                if related_modeladmin:
                    wrapper_kwargs.update(
                        can_add_related=related_modeladmin.has_add_permission(request),
                        can_change_related=related_modeladmin.has_change_permission(request),
                        can_delete_related=related_modeladmin.has_delete_permission(request),
                        can_view_related=related_modeladmin.has_view_permission(request),
                    )
                formfield.widget = widgets.RelatedFieldWidgetWrapper(
                    formfield.widget, db_field.remote_field, self.admin_site, **wrapper_kwargs
                )

            return formfield

        # If we've got overrides for the formfield defined, use 'em. **kwargs
        # passed to formfield_for_dbfield override the defaults.
        for klass in db_field.__class__.mro():
            if klass in self.formfield_overrides:
                kwargs = {**copy.deepcopy(self.formfield_overrides[klass]), **kwargs}
                return db_field.formfield(**kwargs)

        # For any other type of field, just call its formfield() method.
        return db_field.formfield(**kwargs)
```
### 13 - django/contrib/admin/options.py:

Start line: 218, End line: 240

```python
class BaseModelAdmin(metaclass=forms.MediaDefiningClass):

    def formfield_for_foreignkey(self, db_field, request, **kwargs):
        """
        Get a form Field for a ForeignKey.
        """
        db = kwargs.get('using')

        if 'widget' not in kwargs:
            if db_field.name in self.get_autocomplete_fields(request):
                kwargs['widget'] = AutocompleteSelect(db_field.remote_field, self.admin_site, using=db)
            elif db_field.name in self.raw_id_fields:
                kwargs['widget'] = widgets.ForeignKeyRawIdWidget(db_field.remote_field, self.admin_site, using=db)
            elif db_field.name in self.radio_fields:
                kwargs['widget'] = widgets.AdminRadioSelect(attrs={
                    'class': get_ul_class(self.radio_fields[db_field.name]),
                })
                kwargs['empty_label'] = _('None') if db_field.blank else None

        if 'queryset' not in kwargs:
            queryset = self.get_field_queryset(db, db_field, request)
            if queryset is not None:
                kwargs['queryset'] = queryset

        return db_field.formfield(**kwargs)
```
### 14 - django/contrib/admin/options.py:

Start line: 1523, End line: 1609

```python
class ModelAdmin(BaseModelAdmin):

    def _changeform_view(self, request, object_id, form_url, extra_context):
        to_field = request.POST.get(TO_FIELD_VAR, request.GET.get(TO_FIELD_VAR))
        if to_field and not self.to_field_allowed(request, to_field):
            raise DisallowedModelAdminToField("The field %s cannot be referenced." % to_field)

        model = self.model
        opts = model._meta

        if request.method == 'POST' and '_saveasnew' in request.POST:
            object_id = None

        add = object_id is None

        if add:
            if not self.has_add_permission(request):
                raise PermissionDenied
            obj = None

        else:
            obj = self.get_object(request, unquote(object_id), to_field)

            if request.method == 'POST':
                if not self.has_change_permission(request, obj):
                    raise PermissionDenied
            else:
                if not self.has_view_or_change_permission(request, obj):
                    raise PermissionDenied

            if obj is None:
                return self._get_obj_does_not_exist_redirect(request, opts, object_id)

        fieldsets = self.get_fieldsets(request, obj)
        ModelForm = self.get_form(
            request, obj, change=not add, fields=flatten_fieldsets(fieldsets)
        )
        if request.method == 'POST':
            form = ModelForm(request.POST, request.FILES, instance=obj)
            form_validated = form.is_valid()
            if form_validated:
                new_object = self.save_form(request, form, change=not add)
            else:
                new_object = form.instance
            formsets, inline_instances = self._create_formsets(request, new_object, change=not add)
            if all_valid(formsets) and form_validated:
                self.save_model(request, new_object, form, not add)
                self.save_related(request, form, formsets, not add)
                change_message = self.construct_change_message(request, form, formsets, add)
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
                formsets, inline_instances = self._create_formsets(request, form.instance, change=False)
            else:
                form = ModelForm(instance=obj)
                formsets, inline_instances = self._create_formsets(request, obj, change=True)

        if not add and not self.has_change_permission(request, obj):
            readonly_fields = flatten_fieldsets(fieldsets)
        else:
            readonly_fields = self.get_readonly_fields(request, obj)
        adminForm = helpers.AdminForm(
            form,
            list(fieldsets),
            # Clear prepopulated fields on a view-only form to avoid a crash.
            self.get_prepopulated_fields(request, obj) if add or self.has_change_permission(request, obj) else {},
            readonly_fields,
            model_admin=self)
        media = self.media + adminForm.media

        inline_formsets = self.get_inline_formsets(request, formsets, inline_instances, obj)
        for inline_formset in inline_formsets:
            media = media + inline_formset.media

        if add:
            title = _('Add %s')
        elif self.has_change_permission(request, obj):
            title = _('Change %s')
        else:
            title = _('View %s')
        # ... other code
```
### 26 - django/contrib/admin/options.py:

Start line: 1, End line: 95

```python
import copy
import json
import operator
import re
from functools import partial, reduce, update_wrapper
from urllib.parse import quote as urlquote

from django import forms
from django.conf import settings
from django.contrib import messages
from django.contrib.admin import helpers, widgets
from django.contrib.admin.checks import (
    BaseModelAdminChecks, InlineModelAdminChecks, ModelAdminChecks,
)
from django.contrib.admin.exceptions import DisallowedModelAdminToField
from django.contrib.admin.templatetags.admin_urls import add_preserved_filters
from django.contrib.admin.utils import (
    NestedObjects, construct_change_message, flatten_fieldsets,
    get_deleted_objects, lookup_needs_distinct, model_format_dict,
    model_ngettext, quote, unquote,
)
from django.contrib.admin.views.autocomplete import AutocompleteJsonView
from django.contrib.admin.widgets import (
    AutocompleteSelect, AutocompleteSelectMultiple,
)
from django.contrib.auth import get_permission_codename
from django.core.exceptions import (
    FieldDoesNotExist, FieldError, PermissionDenied, ValidationError,
)
from django.core.paginator import Paginator
from django.db import models, router, transaction
from django.db.models.constants import LOOKUP_SEP
from django.forms.formsets import DELETION_FIELD_NAME, all_valid
from django.forms.models import (
    BaseInlineFormSet, inlineformset_factory, modelform_defines_fields,
    modelform_factory, modelformset_factory,
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
from django.utils.text import capfirst, format_lazy, get_text_list
from django.utils.translation import gettext as _, ngettext
from django.views.decorators.csrf import csrf_protect
from django.views.generic import RedirectView

IS_POPUP_VAR = '_popup'
TO_FIELD_VAR = '_to_field'


HORIZONTAL, VERTICAL = 1, 2


def get_content_type_for_model(obj):
    # Since this module gets imported in the application's root package,
    # it cannot import models from other applications at the module level.
    from django.contrib.contenttypes.models import ContentType
    return ContentType.objects.get_for_model(obj, for_concrete_model=False)


def get_ul_class(radio_style):
    return 'radiolist' if radio_style == VERTICAL else 'radiolist inline'


class IncorrectLookupParameters(Exception):
    pass


# Defaults for formfield_overrides. ModelAdmin subclasses can change this
# by adding to ModelAdmin.formfield_overrides.

FORMFIELD_FOR_DBFIELD_DEFAULTS = {
    models.DateTimeField: {
        'form_class': forms.SplitDateTimeField,
        'widget': widgets.AdminSplitDateTime
    },
    models.DateField: {'widget': widgets.AdminDateWidget},
    models.TimeField: {'widget': widgets.AdminTimeWidget},
    models.TextField: {'widget': widgets.AdminTextareaWidget},
    models.URLField: {'widget': widgets.AdminURLFieldWidget},
    models.IntegerField: {'widget': widgets.AdminIntegerFieldWidget},
    models.BigIntegerField: {'widget': widgets.AdminBigIntegerFieldWidget},
    models.CharField: {'widget': widgets.AdminTextInputWidget},
    models.ImageField: {'widget': widgets.AdminFileWidget},
    models.FileField: {'widget': widgets.AdminFileWidget},
    models.EmailField: {'widget': widgets.AdminEmailInputWidget},
    models.UUIDField: {'widget': widgets.AdminUUIDInputWidget},
}

csrf_protect_m = method_decorator(csrf_protect)
```
### 45 - django/contrib/admin/options.py:

Start line: 98, End line: 128

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
### 83 - django/contrib/admin/options.py:

Start line: 1610, End line: 1634

```python
class ModelAdmin(BaseModelAdmin):

    def _changeform_view(self, request, object_id, form_url, extra_context):
        # ... other code
        context = {
            **self.admin_site.each_context(request),
            'title': title % opts.verbose_name,
            'adminform': adminForm,
            'object_id': object_id,
            'original': obj,
            'is_popup': IS_POPUP_VAR in request.POST or IS_POPUP_VAR in request.GET,
            'to_field': to_field,
            'media': media,
            'inline_admin_formsets': inline_formsets,
            'errors': helpers.AdminErrorList(form, formsets),
            'preserved_filters': self.get_preserved_filters(request),
        }

        # Hide the "Save" and "Save and continue" buttons if "Save as New" was
        # previously chosen to prevent the interface from getting confusing.
        if request.method == 'POST' and not form_validated and "_saveasnew" in request.POST:
            context['show_save'] = False
            context['show_save_and_continue'] = False
            # Use the change template instead of the add template.
            add = False

        context.update(extra_context or {})

        return self.render_change_form(request, context, add=add, change=not add, obj=obj, form_url=form_url)
```
### 103 - django/contrib/admin/options.py:

Start line: 2069, End line: 2121

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
### 106 - django/contrib/admin/options.py:

Start line: 367, End line: 419

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
            if (lookup, value) in widgets.url_params_from_lookup_dict(fk_lookup).items():
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
            if not prev_field or (prev_field.is_relation and
                                  field not in prev_field.get_path_info()[-1].target_fields):
                relation_parts.append(part)
            if not getattr(field, 'get_path_info', None):
                # This is not a relational field, so further parts
                # must be transforms.
                break
            prev_field = field
            model = field.get_path_info()[-1].to_opts.model

        if len(relation_parts) <= 1:
            # Either a local field filter, or no fields at all.
            return True
        valid_lookups = {self.date_hierarchy}
        for filter_item in self.list_filter:
            if isinstance(filter_item, type) and issubclass(filter_item, SimpleListFilter):
                valid_lookups.add(filter_item.parameter_name)
            elif isinstance(filter_item, (list, tuple)):
                valid_lookups.add(filter_item[0])
            else:
                valid_lookups.add(filter_item)

        # Is it a valid relational lookup?
        return not {
            LOOKUP_SEP.join(relation_parts),
            LOOKUP_SEP.join(relation_parts + [part])
        }.isdisjoint(valid_lookups)
```
### 111 - django/contrib/admin/options.py:

Start line: 187, End line: 203

```python
class BaseModelAdmin(metaclass=forms.MediaDefiningClass):

    def formfield_for_choice_field(self, db_field, request, **kwargs):
        """
        Get a form Field for a database Field that has declared choices.
        """
        # If the field is named as a radio_field, use a RadioSelect
        if db_field.name in self.radio_fields:
            # Avoid stomping on custom widget/choices arguments.
            if 'widget' not in kwargs:
                kwargs['widget'] = widgets.AdminRadioSelect(attrs={
                    'class': get_ul_class(self.radio_fields[db_field.name]),
                })
            if 'choices' not in kwargs:
                kwargs['choices'] = db_field.get_choices(
                    include_blank=db_field.blank,
                    blank_choice=[('', _('None'))]
                )
        return db_field.formfield(**kwargs)
```
### 146 - django/contrib/admin/options.py:

Start line: 421, End line: 464

```python
class BaseModelAdmin(metaclass=forms.MediaDefiningClass):

    def to_field_allowed(self, request, to_field):
        """
        Return True if the model associated with this admin should be
        allowed to be referenced by the specified field.
        """
        opts = self.model._meta

        try:
            field = opts.get_field(to_field)
        except FieldDoesNotExist:
            return False

        # Always allow referencing the primary key since it's already possible
        # to get this information from the change view URL.
        if field.primary_key:
            return True

        # Allow reverse relationships to models defining m2m fields if they
        # target the specified field.
        for many_to_many in opts.many_to_many:
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
            f for f in opts.get_fields(include_hidden=True)
            if (f.auto_created and not f.concrete)
        )
        for related_object in related_objects:
            related_model = related_object.related_model
            remote_field = related_object.field.remote_field
            if (any(issubclass(model, related_model) for model in registered_models) and
                    hasattr(remote_field, 'get_related_field') and
                    remote_field.get_related_field() == field):
                return True

        return False
```
### 162 - django/contrib/admin/options.py:

Start line: 1110, End line: 1155

```python
class ModelAdmin(BaseModelAdmin):

    def render_change_form(self, request, context, add=False, change=False, form_url='', obj=None):
        opts = self.model._meta
        app_label = opts.app_label
        preserved_filters = self.get_preserved_filters(request)
        form_url = add_preserved_filters({'preserved_filters': preserved_filters, 'opts': opts}, form_url)
        view_on_site_url = self.get_view_on_site_url(obj)
        has_editable_inline_admin_formsets = False
        for inline in context['inline_admin_formsets']:
            if inline.has_add_permission or inline.has_change_permission or inline.has_delete_permission:
                has_editable_inline_admin_formsets = True
                break
        context.update({
            'add': add,
            'change': change,
            'has_view_permission': self.has_view_permission(request, obj),
            'has_add_permission': self.has_add_permission(request),
            'has_change_permission': self.has_change_permission(request, obj),
            'has_delete_permission': self.has_delete_permission(request, obj),
            'has_editable_inline_admin_formsets': has_editable_inline_admin_formsets,
            'has_file_field': context['adminform'].form.is_multipart() or any(
                admin_formset.formset.is_multipart()
                for admin_formset in context['inline_admin_formsets']
            ),
            'has_absolute_url': view_on_site_url is not None,
            'absolute_url': view_on_site_url,
            'form_url': form_url,
            'opts': opts,
            'content_type_id': get_content_type_for_model(self.model).pk,
            'save_as': self.save_as,
            'save_on_top': self.save_on_top,
            'to_field_var': TO_FIELD_VAR,
            'is_popup_var': IS_POPUP_VAR,
            'app_label': app_label,
        })
        if add and self.add_form_template is not None:
            form_template = self.add_form_template
        else:
            form_template = self.change_form_template

        request.current_app = self.admin_site.name

        return TemplateResponse(request, form_template or [
            "admin/%s/%s/change_form.html" % (app_label, opts.model_name),
            "admin/%s/change_form.html" % app_label,
            "admin/change_form.html"
        ], context)
```
