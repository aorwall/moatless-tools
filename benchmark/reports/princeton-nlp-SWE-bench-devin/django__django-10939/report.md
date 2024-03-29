# django__django-10939

| **django/django** | `1933e56eca1ad17de7dd133bfb7cbee9858a75a3` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 9519 |
| **Any found context length** | 612 |
| **Avg pos** | 62.0 |
| **Min pos** | 2 |
| **Max pos** | 30 |
| **Top file pos** | 2 |
| **Missing snippets** | 3 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/forms/widgets.py b/django/forms/widgets.py
--- a/django/forms/widgets.py
+++ b/django/forms/widgets.py
@@ -48,8 +48,8 @@ def __init__(self, media=None, css=None, js=None):
                 css = {}
             if js is None:
                 js = []
-        self._css = css
-        self._js = js
+        self._css_lists = [css]
+        self._js_lists = [js]
 
     def __repr__(self):
         return 'Media(css=%r, js=%r)' % (self._css, self._js)
@@ -57,6 +57,25 @@ def __repr__(self):
     def __str__(self):
         return self.render()
 
+    @property
+    def _css(self):
+        css = self._css_lists[0]
+        # filter(None, ...) avoids calling merge with empty dicts.
+        for obj in filter(None, self._css_lists[1:]):
+            css = {
+                medium: self.merge(css.get(medium, []), obj.get(medium, []))
+                for medium in css.keys() | obj.keys()
+            }
+        return css
+
+    @property
+    def _js(self):
+        js = self._js_lists[0]
+        # filter(None, ...) avoids calling merge() with empty lists.
+        for obj in filter(None, self._js_lists[1:]):
+            js = self.merge(js, obj)
+        return js
+
     def render(self):
         return mark_safe('\n'.join(chain.from_iterable(getattr(self, 'render_' + name)() for name in MEDIA_TYPES)))
 
@@ -132,11 +151,8 @@ def merge(list_1, list_2):
 
     def __add__(self, other):
         combined = Media()
-        combined._js = self.merge(self._js, other._js)
-        combined._css = {
-            medium: self.merge(self._css.get(medium, []), other._css.get(medium, []))
-            for medium in self._css.keys() | other._css.keys()
-        }
+        combined._css_lists = self._css_lists + other._css_lists
+        combined._js_lists = self._js_lists + other._js_lists
         return combined
 
 

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/forms/widgets.py | 51 | 52 | 30 | 2 | 9519
| django/forms/widgets.py | 60 | 60 | 30 | 2 | 9519
| django/forms/widgets.py | 135 | 139 | 2 | 2 | 612


## Problem Statement

```
ModelAdmin with custom widgets, inlines, and filter_horizontal can merge media in broken order
Description
	
when a modeadmin have a inline with a filed has its own media js no need jquery , and have a one to many field show filter_horizontal, the problem appear.
there will be MediaOrderConflictWarning and inlines.js load before jquery.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 django/contrib/admin/options.py | 1993 | 2014| 221 | 221 | 18325 | 
| **-> 2 <-** | **2 django/forms/widgets.py** | 98 | 140| 391 | 612 | 26303 | 
| 3 | **2 django/forms/widgets.py** | 1 | 37| 257 | 869 | 26303 | 
| 4 | 3 django/contrib/admin/widgets.py | 440 | 468| 203 | 1072 | 30101 | 
| 5 | 3 django/contrib/admin/widgets.py | 53 | 74| 168 | 1240 | 30101 | 
| 6 | 4 django/contrib/contenttypes/admin.py | 83 | 130| 410 | 1650 | 31126 | 
| 7 | 4 django/contrib/admin/options.py | 99 | 129| 223 | 1873 | 31126 | 
| 8 | 4 django/contrib/admin/options.py | 1 | 96| 769 | 2642 | 31126 | 
| 9 | 4 django/contrib/admin/widgets.py | 1 | 50| 370 | 3012 | 31126 | 
| 10 | 4 django/contrib/admin/options.py | 623 | 640| 136 | 3148 | 31126 | 
| 11 | 4 django/contrib/admin/options.py | 2141 | 2163| 201 | 3349 | 31126 | 
| 12 | 5 django/contrib/admin/checks.py | 649 | 683| 265 | 3614 | 39969 | 
| 13 | 6 django/contrib/admin/helpers.py | 70 | 89| 167 | 3781 | 43139 | 
| 14 | 6 django/contrib/admin/options.py | 1514 | 1593| 719 | 4500 | 43139 | 
| 15 | 6 django/contrib/admin/options.py | 2105 | 2139| 359 | 4859 | 43139 | 
| 16 | 6 django/contrib/admin/options.py | 1463 | 1479| 231 | 5090 | 43139 | 
| 17 | 6 django/contrib/admin/options.py | 1722 | 1803| 744 | 5834 | 43139 | 
| 18 | 6 django/contrib/admin/widgets.py | 199 | 225| 216 | 6050 | 43139 | 
| 19 | **6 django/forms/widgets.py** | 830 | 870| 309 | 6359 | 43139 | 
| 20 | 6 django/contrib/admin/options.py | 364 | 416| 504 | 6863 | 43139 | 
| 21 | 6 django/contrib/admin/widgets.py | 373 | 391| 153 | 7016 | 43139 | 
| 22 | 6 django/contrib/admin/checks.py | 978 | 990| 116 | 7132 | 43139 | 
| 23 | 6 django/contrib/admin/helpers.py | 301 | 327| 171 | 7303 | 43139 | 
| 24 | 6 django/contrib/admin/options.py | 2051 | 2103| 451 | 7754 | 43139 | 
| 25 | 7 django/forms/forms.py | 454 | 501| 382 | 8136 | 47196 | 
| 26 | 8 django/contrib/admin/views/main.py | 1 | 34| 244 | 8380 | 51251 | 
| 27 | 8 django/contrib/admin/options.py | 538 | 596| 408 | 8788 | 51251 | 
| 28 | 8 django/contrib/admin/widgets.py | 77 | 93| 145 | 8933 | 51251 | 
| 29 | 8 django/contrib/admin/helpers.py | 361 | 377| 138 | 9071 | 51251 | 
| **-> 30 <-** | **8 django/forms/widgets.py** | 40 | 96| 448 | 9519 | 51251 | 
| 31 | 8 django/contrib/admin/options.py | 1960 | 1991| 249 | 9768 | 51251 | 
| 32 | 8 django/contrib/admin/checks.py | 992 | 1019| 204 | 9972 | 51251 | 
| 33 | 8 django/contrib/admin/widgets.py | 277 | 312| 382 | 10354 | 51251 | 
| 34 | 8 django/contrib/admin/helpers.py | 272 | 299| 265 | 10619 | 51251 | 
| 35 | 8 django/contrib/admin/options.py | 277 | 362| 615 | 11234 | 51251 | 
| 36 | 8 django/contrib/admin/options.py | 1108 | 1153| 484 | 11718 | 51251 | 
| 37 | **8 django/forms/widgets.py** | 753 | 776| 204 | 11922 | 51251 | 
| 38 | **8 django/forms/widgets.py** | 143 | 179| 235 | 12157 | 51251 | 
| 39 | 9 django/contrib/gis/admin/options.py | 80 | 135| 555 | 12712 | 52447 | 
| 40 | 9 django/contrib/contenttypes/admin.py | 1 | 80| 615 | 13327 | 52447 | 
| 41 | 9 django/contrib/admin/helpers.py | 33 | 67| 230 | 13557 | 52447 | 
| 42 | 9 django/contrib/admin/helpers.py | 247 | 270| 235 | 13792 | 52447 | 
| 43 | 9 django/contrib/admin/options.py | 1594 | 1618| 279 | 14071 | 52447 | 
| 44 | 9 django/contrib/admin/helpers.py | 330 | 348| 181 | 14252 | 52447 | 
| 45 | 9 django/contrib/admin/widgets.py | 228 | 275| 423 | 14675 | 52447 | 
| 46 | 9 django/contrib/admin/views/main.py | 104 | 186| 818 | 15493 | 52447 | 
| 47 | 9 django/contrib/admin/widgets.py | 315 | 327| 114 | 15607 | 52447 | 
| 48 | 9 django/contrib/admin/checks.py | 378 | 390| 137 | 15744 | 52447 | 
| 49 | 10 django/contrib/admin/filters.py | 365 | 389| 294 | 16038 | 56116 | 
| 50 | 10 django/contrib/admin/filters.py | 118 | 159| 365 | 16403 | 56116 | 
| 51 | 10 django/contrib/gis/admin/options.py | 52 | 63| 139 | 16542 | 56116 | 
| 52 | 10 django/contrib/admin/widgets.py | 124 | 163| 349 | 16891 | 56116 | 
| 53 | 10 django/contrib/admin/checks.py | 747 | 768| 190 | 17081 | 56116 | 
| 54 | 11 django/contrib/admindocs/views.py | 86 | 114| 285 | 17366 | 59426 | 
| 55 | 11 django/contrib/admin/helpers.py | 380 | 403| 195 | 17561 | 59426 | 
| 56 | 11 django/contrib/admin/options.py | 206 | 217| 135 | 17696 | 59426 | 
| 57 | 11 django/contrib/admin/filters.py | 299 | 362| 627 | 18323 | 59426 | 
| 58 | 11 django/contrib/admin/widgets.py | 165 | 196| 243 | 18566 | 59426 | 
| 59 | **11 django/forms/widgets.py** | 799 | 828| 265 | 18831 | 59426 | 
| 60 | 11 django/contrib/gis/admin/options.py | 1 | 50| 394 | 19225 | 59426 | 
| 61 | 11 django/contrib/admin/helpers.py | 123 | 149| 220 | 19445 | 59426 | 
| 62 | 11 django/contrib/admin/options.py | 598 | 621| 284 | 19729 | 59426 | 
| 63 | 12 django/contrib/gis/admin/__init__.py | 1 | 13| 130 | 19859 | 59556 | 
| 64 | 12 django/contrib/admindocs/views.py | 182 | 248| 584 | 20443 | 59556 | 
| 65 | **12 django/forms/widgets.py** | 325 | 359| 278 | 20721 | 59556 | 
| 66 | **12 django/forms/widgets.py** | 362 | 379| 116 | 20837 | 59556 | 
| 67 | **12 django/forms/widgets.py** | 779 | 797| 153 | 20990 | 59556 | 
| 68 | 13 django/contrib/admin/__init__.py | 1 | 30| 286 | 21276 | 59842 | 
| 69 | 13 django/contrib/admindocs/views.py | 249 | 314| 585 | 21861 | 59842 | 
| 70 | 13 django/contrib/admin/filters.py | 1 | 17| 127 | 21988 | 59842 | 
| 71 | 13 django/contrib/admin/filters.py | 162 | 201| 381 | 22369 | 59842 | 
| 72 | **13 django/forms/widgets.py** | 900 | 909| 107 | 22476 | 59842 | 
| 73 | 14 django/contrib/auth/admin.py | 25 | 37| 128 | 22604 | 61568 | 
| 74 | 14 django/contrib/admin/checks.py | 599 | 618| 183 | 22787 | 61568 | 
| 75 | **14 django/forms/widgets.py** | 912 | 958| 400 | 23187 | 61568 | 
| 76 | 14 django/contrib/admin/checks.py | 858 | 906| 416 | 23603 | 61568 | 
| 77 | 15 django/contrib/admin/templatetags/admin_modify.py | 1 | 45| 372 | 23975 | 62488 | 
| 78 | **15 django/forms/widgets.py** | 731 | 750| 144 | 24119 | 62488 | 
| 79 | 15 django/contrib/admin/helpers.py | 92 | 120| 249 | 24368 | 62488 | 
| 80 | 15 django/contrib/admin/helpers.py | 223 | 245| 211 | 24579 | 62488 | 
| 81 | 15 django/contrib/admin/checks.py | 697 | 728| 227 | 24806 | 62488 | 
| 82 | 15 django/contrib/admin/options.py | 1914 | 1957| 403 | 25209 | 62488 | 
| 83 | 16 django/contrib/admin/templatetags/admin_list.py | 1 | 26| 170 | 25379 | 66271 | 
| 84 | 17 django/contrib/postgres/forms/array.py | 133 | 165| 265 | 25644 | 67724 | 
| 85 | 17 django/contrib/admin/views/main.py | 401 | 444| 390 | 26034 | 67724 | 
| 86 | 18 django/forms/fields.py | 1122 | 1155| 293 | 26327 | 76668 | 
| 87 | **18 django/forms/widgets.py** | 873 | 897| 172 | 26499 | 76668 | 
| 88 | 18 django/contrib/admin/filters.py | 391 | 423| 299 | 26798 | 76668 | 
| 89 | **18 django/forms/widgets.py** | 428 | 449| 217 | 27015 | 76668 | 
| 90 | 18 django/contrib/admin/widgets.py | 344 | 370| 328 | 27343 | 76668 | 
| 91 | 19 django/db/models/fields/related_descriptors.py | 871 | 891| 190 | 27533 | 86694 | 
| 92 | 19 django/contrib/admin/helpers.py | 350 | 359| 134 | 27667 | 86694 | 
| 93 | 19 django/contrib/admin/options.py | 2016 | 2049| 376 | 28043 | 86694 | 
| 94 | 19 django/contrib/admin/filters.py | 258 | 270| 149 | 28192 | 86694 | 
| 95 | 19 django/db/models/fields/related_descriptors.py | 973 | 997| 240 | 28432 | 86694 | 
| 96 | 19 django/contrib/admindocs/views.py | 155 | 179| 234 | 28666 | 86694 | 
| 97 | **19 django/forms/widgets.py** | 182 | 264| 622 | 29288 | 86694 | 
| 98 | 19 django/contrib/admin/checks.py | 525 | 560| 303 | 29591 | 86694 | 
| 99 | 19 django/contrib/admin/filters.py | 203 | 220| 190 | 29781 | 86694 | 
| 100 | 20 django/contrib/gis/forms/widgets.py | 76 | 118| 304 | 30085 | 87549 | 
| 101 | 20 django/contrib/admin/checks.py | 908 | 937| 243 | 30328 | 87549 | 
| 102 | 20 django/contrib/admin/checks.py | 108 | 125| 155 | 30483 | 87549 | 
| 103 | **20 django/forms/widgets.py** | 960 | 994| 391 | 30874 | 87549 | 
| 104 | 20 django/contrib/admin/options.py | 1805 | 1873| 584 | 31458 | 87549 | 
| 105 | 20 django/contrib/admin/options.py | 1650 | 1721| 653 | 32111 | 87549 | 
| 106 | 20 django/contrib/admin/widgets.py | 414 | 438| 232 | 32343 | 87549 | 
| 107 | 20 django/db/models/fields/related_descriptors.py | 1124 | 1164| 378 | 32721 | 87549 | 
| 108 | 20 django/db/models/fields/related_descriptors.py | 855 | 869| 190 | 32911 | 87549 | 
| 109 | 20 django/contrib/admin/checks.py | 304 | 330| 221 | 33132 | 87549 | 
| 110 | 20 django/contrib/admin/checks.py | 139 | 180| 325 | 33457 | 87549 | 
| 111 | **20 django/forms/widgets.py** | 452 | 492| 275 | 33732 | 87549 | 
| 112 | 20 django/contrib/admin/options.py | 841 | 859| 187 | 33919 | 87549 | 
| 113 | **20 django/forms/widgets.py** | 659 | 690| 261 | 34180 | 87549 | 
| 114 | 20 django/contrib/admin/checks.py | 1021 | 1063| 343 | 34523 | 87549 | 
| 115 | 20 django/contrib/admin/views/main.py | 37 | 102| 581 | 35104 | 87549 | 
| 116 | 21 django/forms/models.py | 947 | 980| 367 | 35471 | 98995 | 
| 117 | 21 django/contrib/admin/checks.py | 350 | 376| 281 | 35752 | 98995 | 
| 118 | 22 django/forms/boundfield.py | 228 | 267| 273 | 36025 | 101092 | 
| 119 | 22 django/contrib/admindocs/views.py | 317 | 347| 208 | 36233 | 101092 | 
| 120 | 22 django/contrib/admin/widgets.py | 96 | 121| 179 | 36412 | 101092 | 
| 121 | 22 django/db/models/fields/related_descriptors.py | 1025 | 1052| 334 | 36746 | 101092 | 
| 122 | 22 django/contrib/admin/checks.py | 575 | 596| 162 | 36908 | 101092 | 
| 123 | 22 django/forms/fields.py | 46 | 127| 773 | 37681 | 101092 | 
| 124 | 22 django/db/models/fields/related_descriptors.py | 1054 | 1122| 661 | 38342 | 101092 | 
| 125 | 22 django/contrib/admin/filters.py | 223 | 238| 196 | 38538 | 101092 | 
| 126 | 22 django/db/models/fields/related_descriptors.py | 663 | 717| 511 | 39049 | 101092 | 
| 127 | 22 django/contrib/admin/views/main.py | 446 | 477| 225 | 39274 | 101092 | 
| 128 | 22 django/contrib/admin/checks.py | 500 | 523| 230 | 39504 | 101092 | 
| 129 | 23 django/core/management/commands/makemigrations.py | 231 | 311| 824 | 40328 | 103841 | 
| 130 | 23 django/db/models/fields/related_descriptors.py | 599 | 631| 323 | 40651 | 103841 | 
| 131 | 23 django/contrib/admin/checks.py | 685 | 695| 115 | 40766 | 103841 | 
| 132 | 24 django/db/models/fields/related.py | 1160 | 1191| 180 | 40946 | 117336 | 
| 133 | 24 django/contrib/admin/options.py | 1875 | 1912| 330 | 41276 | 117336 | 
| 134 | 24 django/forms/boundfield.py | 35 | 50| 149 | 41425 | 117336 | 
| 135 | 25 django/contrib/admin/templatetags/base.py | 22 | 34| 135 | 41560 | 117635 | 
| 136 | 26 django/db/models/base.py | 1224 | 1253| 242 | 41802 | 132316 | 
| 137 | 27 django/db/models/__init__.py | 1 | 49| 548 | 42350 | 132864 | 
| 138 | 27 django/db/models/fields/related_descriptors.py | 542 | 597| 478 | 42828 | 132864 | 
| 139 | **27 django/forms/widgets.py** | 1021 | 1040| 144 | 42972 | 132864 | 
| 140 | **27 django/forms/widgets.py** | 1042 | 1068| 263 | 43235 | 132864 | 
| 141 | 27 django/db/models/fields/related_descriptors.py | 951 | 972| 248 | 43483 | 132864 | 
| 142 | 27 django/db/models/fields/related.py | 1193 | 1316| 1010 | 44493 | 132864 | 
| 143 | 27 django/contrib/admin/filters.py | 272 | 296| 217 | 44710 | 132864 | 
| 144 | 27 django/contrib/admin/checks.py | 770 | 820| 443 | 45153 | 132864 | 
| 145 | 28 django/contrib/admin/sites.py | 211 | 230| 221 | 45374 | 136980 | 
| 146 | 28 django/contrib/gis/admin/options.py | 65 | 78| 150 | 45524 | 136980 | 
| 147 | 28 django/contrib/admin/widgets.py | 393 | 412| 193 | 45717 | 136980 | 
| 148 | 28 django/db/models/fields/related_descriptors.py | 633 | 662| 254 | 45971 | 136980 | 
| 149 | 28 django/contrib/admin/options.py | 1307 | 1332| 232 | 46203 | 136980 | 
| 150 | 28 django/db/models/fields/related.py | 1035 | 1079| 407 | 46610 | 136980 | 
| 151 | 29 django/db/migrations/serializer.py | 190 | 211| 183 | 46793 | 139516 | 
| 152 | **29 django/forms/widgets.py** | 414 | 426| 122 | 46915 | 139516 | 
| 153 | 29 django/contrib/postgres/forms/array.py | 105 | 131| 219 | 47134 | 139516 | 
| 154 | 29 django/db/models/fields/related.py | 255 | 282| 269 | 47403 | 139516 | 
| 155 | 30 django/contrib/contenttypes/fields.py | 664 | 689| 254 | 47657 | 144819 | 
| 156 | 30 django/contrib/admindocs/views.py | 55 | 83| 285 | 47942 | 144819 | 
| 157 | 30 django/db/models/fields/related_descriptors.py | 719 | 745| 222 | 48164 | 144819 | 
| 158 | 30 django/db/models/fields/related_descriptors.py | 999 | 1024| 241 | 48405 | 144819 | 
| 159 | 30 django/contrib/admin/templatetags/admin_list.py | 428 | 486| 343 | 48748 | 144819 | 
| 160 | 31 django/contrib/admin/decorators.py | 1 | 31| 134 | 48882 | 145013 | 
| 161 | 31 django/contrib/admin/options.py | 1442 | 1461| 133 | 49015 | 145013 | 
| 162 | 31 django/forms/boundfield.py | 76 | 94| 174 | 49189 | 145013 | 
| 163 | 31 django/db/models/base.py | 1362 | 1417| 491 | 49680 | 145013 | 
| 164 | 31 django/contrib/admin/views/main.py | 307 | 359| 467 | 50147 | 145013 | 
| 165 | 31 django/contrib/admin/templatetags/admin_list.py | 353 | 425| 631 | 50778 | 145013 | 
| 166 | **31 django/forms/widgets.py** | 606 | 625| 189 | 50967 | 145013 | 
| 167 | 32 django/contrib/admin/views/autocomplete.py | 1 | 36| 256 | 51223 | 145415 | 
| 168 | 32 django/contrib/admin/checks.py | 426 | 452| 190 | 51413 | 145415 | 
| 169 | 32 django/contrib/admindocs/views.py | 135 | 153| 187 | 51600 | 145415 | 
| 170 | 32 django/db/models/fields/related_descriptors.py | 794 | 853| 576 | 52176 | 145415 | 
| 171 | 32 django/contrib/admin/views/autocomplete.py | 38 | 53| 154 | 52330 | 145415 | 
| 172 | 33 django/db/models/fields/__init__.py | 1114 | 1143| 218 | 52548 | 162254 | 
| 173 | 33 django/contrib/admin/sites.py | 1 | 28| 172 | 52720 | 162254 | 
| 174 | 33 django/contrib/admin/options.py | 1620 | 1632| 167 | 52887 | 162254 | 
| 175 | 34 django/contrib/postgres/forms/ranges.py | 66 | 110| 284 | 53171 | 162958 | 
| 176 | 34 django/contrib/admin/checks.py | 846 | 856| 129 | 53300 | 162958 | 
| 177 | 34 django/db/models/fields/__init__.py | 322 | 346| 184 | 53484 | 162958 | 
| 178 | 34 django/contrib/admin/options.py | 131 | 186| 604 | 54088 | 162958 | 
| 179 | 34 django/forms/models.py | 1080 | 1120| 306 | 54394 | 162958 | 
| 180 | 34 django/contrib/admin/widgets.py | 330 | 341| 118 | 54512 | 162958 | 
| 181 | 34 django/contrib/admin/filters.py | 20 | 59| 295 | 54807 | 162958 | 
| 182 | 34 django/contrib/admin/templatetags/admin_modify.py | 84 | 112| 203 | 55010 | 162958 | 


### Hint

```
This is the test program
I dove into this for a while. I'm not sure if there's anything that Django can do about it. Reverting 03974d81220ffd237754a82c77913799dd5909a4 solves the problem but that'll break other cases. The root causes seems to be that ckeditor/ckeditor/ckeditor.js is collected before jquery in one case and after it in another.
It seems to me that Django assumes too much. Here are some assets that have been collected in my case: [ 'imagefield/ppoi.js', <JS(ckeditor/ckeditor-init.js, {'id': 'ckeditor-init-script', 'data-ckeditor-basepath': '/static/ckeditor/ckeditor/'})>, 'ckeditor/ckeditor/ckeditor.js', 'admin/js/vendor/jquery/jquery.js', 'admin/js/jquery.init.js', 'admin/js/core.js', 'admin/js/admin/RelatedObjectLookups.js', 'admin/js/actions.js', 'admin/js/urlify.js', 'admin/js/prepopulate.js', 'admin/js/vendor/xregexp/xregexp.js', <JS(https://use.fontawesome.com/releases/v5.3.1/js/all.js, {'async': 'async', 'integrity': 'sha384-kW+oWsYx3YpxvjtZjFXqazFpA7UP/MbiY4jvs+RWZo2+N94PFZ36T6TFkc9O3qoB', 'crossorigin': 'anonymous'})>, 'app/plugin_buttons.js', 'admin/js/calendar.js', 'admin/js/admin/DateTimeShortcuts.js' ] The imagefield and ckeditor assets are there because of widgets. When using the same widgets in inlines the inlines' Media class will contain jquery, jquery.init.js, inlines.js, imagefield/ppoi.js. When merging the two JS lists Django will find that imagefield/ppoi.js is at index 0, will continue with inlines.js (and of course not find it) and insert it at index 0 as well (because that's the value of last_insert_index now). As soon as jquery.init.js is encountered it notices that something is amiss and emits a MediaOrderConflictWarning. The problem was produced one iteration earlier and the error message is not very helpful. I don't have a good suggestion yet. It also baffles me that only one of two candidate models/modeladmins shows the problem, and (for now) luckily only in development.
Following up. This problem exists since Django 2.0. It doesn't show itself in Django 1.11.x (not surprising since the MediaOrderConflictWarning change was introduced with Django 2.0). It's good that I'm able to reproduce this on a different computer, so that gives me hope that it's not impossible to debug & fix. It's unfortunate that the media files which cause the reordering don't even depend on jQuery.
Hi there, I was the one introducing the warning. The warning is emitted when the defined asset order can not be maintained. Which is a good thing to warn about, but not always a problem. It is particularly not an issue, if you have nested forms, where assets from multiple fields are merged into preliminary forms (like inlines) just to be merged again. I think a real solution could be to retain information about the explicitly defined order-constraints and ignore the implicit once. The merging algorithm can stay as is, it is correct. All that would need to be done, is order violation violates an implicit or explicit asset order before emitting the warning. This would however use a bit more memory, since we would need to keep a record of constraints (list of tuples). What do you think? I could invest a bit of time, to draft a solution.
Hey, yes that might be the right thing to do but maybe there is a solution which requires less effort. I suspect that the problem is an inconsistency in the way Django collects media from different places. I inserted a few print() statements here ​https://github.com/django/django/blob/893b80d95dd76642e478893ba6d4f46bb31388f1/django/contrib/admin/options.py#L1595 (Sorry for the big blob in advance) self.media <script type="text/javascript" src="/static/admin/js/vendor/jquery/jquery.js"></script> <script type="text/javascript" src="/static/admin/js/jquery.init.js"></script> <script type="text/javascript" src="/static/admin/js/core.js"></script> <script type="text/javascript" src="/static/admin/js/admin/RelatedObjectLookups.js"></script> <script type="text/javascript" src="/static/admin/js/actions.js"></script> <script type="text/javascript" src="/static/admin/js/urlify.js"></script> <script type="text/javascript" src="/static/admin/js/prepopulate.js"></script> <script type="text/javascript" src="/static/admin/js/vendor/xregexp/xregexp.js"></script> <script type="text/javascript" src="https://use.fontawesome.com/releases/v5.3.1/js/all.js" async="async" crossorigin="anonymous" integrity="sha384-kW+oWsYx3YpxvjtZjFXqazFpA7UP/MbiY4jvs+RWZo2+N94PFZ36T6TFkc9O3qoB"></script> <script type="text/javascript" src="/static/app/plugin_buttons.js"></script> adminForm.media <link href="/static/imagefield/ppoi.css" type="text/css" media="screen" rel="stylesheet"> <script type="text/javascript" src="/static/imagefield/ppoi.js"></script> <script type="text/javascript" src="/static/ckeditor/ckeditor-init.js" data-ckeditor-basepath="/static/ckeditor/ckeditor/" id="ckeditor-init-script"></script> <script type="text/javascript" src="/static/ckeditor/ckeditor/ckeditor.js"></script> <script type="text/javascript" src="/static/admin/js/vendor/jquery/jquery.js"></script> <script type="text/javascript" src="/static/admin/js/jquery.init.js"></script> <script type="text/javascript" src="/static/admin/js/calendar.js"></script> <script type="text/javascript" src="/static/admin/js/admin/DateTimeShortcuts.js"></script> inline_formset.media <class 'app.articles.models.RichText'> <script type="text/javascript" src="/static/admin/js/vendor/jquery/jquery.js"></script> <script type="text/javascript" src="/static/admin/js/jquery.init.js"></script> <script type="text/javascript" src="/static/admin/js/inlines.js"></script> <script type="text/javascript" src="/static/feincms3/plugin_ckeditor.js"></script> <script type="text/javascript" src="/static/ckeditor/ckeditor-init.js" data-ckeditor-basepath="/static/ckeditor/ckeditor/" id="ckeditor-init-script"></script> <script type="text/javascript" src="/static/ckeditor/ckeditor/ckeditor.js"></script> inline_formset.media <class 'app.articles.models.Image'> <link href="/static/imagefield/ppoi.css" type="text/css" media="screen" rel="stylesheet"> <script type="text/javascript" src="/static/admin/js/vendor/jquery/jquery.js"></script> <script type="text/javascript" src="/static/admin/js/jquery.init.js"></script> <script type="text/javascript" src="/static/admin/js/inlines.js"></script> <script type="text/javascript" src="/static/imagefield/ppoi.js"></script> So somehow the widget media files (imagefield and ckeditor) come after the files added by Django's inlines in inline formsets but before them in the helpers.AdminForm belonging to the ArticleAdmin class. The problem manifests itself when having any third-party widget (which does not reference Django's jquery asset) before any Django date field, prepopulated field or filter_* field in the fieldsets structure. Reordering fieldsets (or fields) avoids the problem completely. Now I still don't understand why the exact same project would work on the server and not locally. The problem can be worked around by including "admin/js/vendor/jquery/jquery%s.js", "admin/js/jquery.init.js" in third-party widgets' Media definitions. This sucks big time though, especially since those widgets don't even require jQuery to work. jQuery is included on all modeladmin pages anyway, so a good fix might be to remove the jquery and jquery.init.js entries from admin widgets Media definitions (which makes them incomplete when used outside of Django's admin panel, but that's probably not the intention anyway) or make the Media merging algorithm aware of libraries which are supposed to always come first. Just for reference, here's the form class and its fields. Putting e.g. publication_date before image makes everything work fine. form.media <link href="/static/imagefield/ppoi.css" type="text/css" media="screen" rel="stylesheet"> <script type="text/javascript" src="/static/imagefield/ppoi.js"></script> <script type="text/javascript" src="/static/ckeditor/ckeditor-init.js" data-ckeditor-basepath="/static/ckeditor/ckeditor/" id="ckeditor-init-script"></script> <script type="text/javascript" src="/static/ckeditor/ckeditor/ckeditor.js"></script> <script type="text/javascript" src="/static/admin/js/vendor/jquery/jquery.js"></script> <script type="text/javascript" src="/static/admin/js/jquery.init.js"></script> <script type="text/javascript" src="/static/admin/js/calendar.js"></script> <script type="text/javascript" src="/static/admin/js/admin/DateTimeShortcuts.js"></script> form.fields is_active: <class 'django.forms.fields.BooleanField'> title: <class 'django.forms.fields.CharField'> excerpt: <class 'django.forms.fields.CharField'> image: <class 'django.forms.fields.ImageField'> image_ppoi: <class 'django.forms.fields.CharField'> meta_title: <class 'django.forms.fields.CharField'> meta_description: <class 'django.forms.fields.CharField'> meta_image: <class 'django.forms.fields.ImageField'> meta_canonical: <class 'django.forms.fields.URLField'> meta_author: <class 'django.forms.fields.CharField'> meta_robots: <class 'django.forms.fields.CharField'> show_teaser_need: <class 'django.forms.fields.BooleanField'> show_teaser_competency: <class 'django.forms.fields.BooleanField'> teaser_title: <class 'django.forms.fields.CharField'> teaser_text_need: <class 'ckeditor.fields.RichTextFormField'> teaser_text_competency: <class 'ckeditor.fields.RichTextFormField'> teaser_image: <class 'django.forms.fields.ImageField'> teaser_image_ppoi: <class 'django.forms.fields.CharField'> slug: <class 'django.forms.fields.SlugField'> publication_date: <class 'django.forms.fields.SplitDateTimeField'> is_featured: <class 'django.forms.fields.BooleanField'> author: <class 'django.forms.models.ModelChoiceField'> categories: <class 'django.forms.models.ModelMultipleChoiceField'>
I just attached a minimal patch demonstrating the problem: https://code.djangoproject.com/attachment/ticket/30153/test.patch Simply reordering date and dummy in Holder3 makes the problem go away in this case (since Holder3's modeladmin class does not define fieldsets)
Test showing the problem
Replacing the patch with https://code.djangoproject.com/attachment/ticket/30153/test.2.patch because Django 2.2 and 3 will not use jquery in date fields anymore and therefore cannot be used to demonstrate the problem. I changed the code to use a filter_horizontal widget and now it fails again: ====================================================================== ERROR: test_inline_media_only_inline (admin_inlines.tests.TestInlineMedia) ---------------------------------------------------------------------- Traceback (most recent call last): File "/usr/lib/python3.6/unittest/case.py", line 59, in testPartExecutor yield File "/usr/lib/python3.6/unittest/case.py", line 605, in run testMethod() File "/home/matthias/Projects/django/tests/admin_inlines/tests.py", line 495, in test_inline_media_only_inline response = self.client.get(change_url) File "/home/matthias/Projects/django/django/test/client.py", line 535, in get response = super().get(path, data=data, secure=secure, **extra) File "/home/matthias/Projects/django/django/test/client.py", line 347, in get **extra, File "/home/matthias/Projects/django/django/test/client.py", line 422, in generic return self.request(**r) File "/home/matthias/Projects/django/django/test/client.py", line 503, in request raise exc_value File "/home/matthias/Projects/django/django/core/handlers/exception.py", line 34, in inner response = get_response(request) File "/home/matthias/Projects/django/django/core/handlers/base.py", line 115, in _get_response response = self.process_exception_by_middleware(e, request) File "/home/matthias/Projects/django/django/core/handlers/base.py", line 113, in _get_response response = wrapped_callback(request, *callback_args, **callback_kwargs) File "/home/matthias/Projects/django/django/contrib/admin/options.py", line 604, in wrapper return self.admin_site.admin_view(view)(*args, **kwargs) File "/home/matthias/Projects/django/django/utils/decorators.py", line 142, in _wrapped_view response = view_func(request, *args, **kwargs) File "/home/matthias/Projects/django/django/views/decorators/cache.py", line 44, in _wrapped_view_func response = view_func(request, *args, **kwargs) File "/home/matthias/Projects/django/django/contrib/admin/sites.py", line 223, in inner return view(request, *args, **kwargs) File "/home/matthias/Projects/django/django/contrib/admin/options.py", line 1635, in change_view return self.changeform_view(request, object_id, form_url, extra_context) File "/home/matthias/Projects/django/django/utils/decorators.py", line 45, in _wrapper return bound_method(*args, **kwargs) File "/home/matthias/Projects/django/django/utils/decorators.py", line 142, in _wrapped_view response = view_func(request, *args, **kwargs) File "/home/matthias/Projects/django/django/contrib/admin/options.py", line 1520, in changeform_view return self._changeform_view(request, object_id, form_url, extra_context) File "/home/matthias/Projects/django/django/contrib/admin/options.py", line 1594, in _changeform_view media = media + inline_formset.media File "/home/matthias/Projects/django/django/forms/widgets.py", line 135, in __add__ combined._js = self.merge(self._js, other._js) File "/home/matthias/Projects/django/django/forms/widgets.py", line 126, in merge MediaOrderConflictWarning, django.forms.widgets.MediaOrderConflictWarning: Detected duplicate Media files in an opposite order: admin/js/inlines.min.js admin/js/jquery.init.js
@codingjoe Here's a partial fix for the issue -- ​https://github.com/matthiask/django/commit/0640ba9f5f6272987b77c35d5ad992844d6a8822 Preserving the JavaScript lists longer and only merging them at the end works. Or maybe you have a better idea? Feel free to take over if you do (or even if you don't -- if you want to)
```

## Patch

```diff
diff --git a/django/forms/widgets.py b/django/forms/widgets.py
--- a/django/forms/widgets.py
+++ b/django/forms/widgets.py
@@ -48,8 +48,8 @@ def __init__(self, media=None, css=None, js=None):
                 css = {}
             if js is None:
                 js = []
-        self._css = css
-        self._js = js
+        self._css_lists = [css]
+        self._js_lists = [js]
 
     def __repr__(self):
         return 'Media(css=%r, js=%r)' % (self._css, self._js)
@@ -57,6 +57,25 @@ def __repr__(self):
     def __str__(self):
         return self.render()
 
+    @property
+    def _css(self):
+        css = self._css_lists[0]
+        # filter(None, ...) avoids calling merge with empty dicts.
+        for obj in filter(None, self._css_lists[1:]):
+            css = {
+                medium: self.merge(css.get(medium, []), obj.get(medium, []))
+                for medium in css.keys() | obj.keys()
+            }
+        return css
+
+    @property
+    def _js(self):
+        js = self._js_lists[0]
+        # filter(None, ...) avoids calling merge() with empty lists.
+        for obj in filter(None, self._js_lists[1:]):
+            js = self.merge(js, obj)
+        return js
+
     def render(self):
         return mark_safe('\n'.join(chain.from_iterable(getattr(self, 'render_' + name)() for name in MEDIA_TYPES)))
 
@@ -132,11 +151,8 @@ def merge(list_1, list_2):
 
     def __add__(self, other):
         combined = Media()
-        combined._js = self.merge(self._js, other._js)
-        combined._css = {
-            medium: self.merge(self._css.get(medium, []), other._css.get(medium, []))
-            for medium in self._css.keys() | other._css.keys()
-        }
+        combined._css_lists = self._css_lists + other._css_lists
+        combined._js_lists = self._js_lists + other._js_lists
         return combined
 
 

```

## Test Patch

```diff
diff --git a/tests/forms_tests/tests/test_media.py b/tests/forms_tests/tests/test_media.py
--- a/tests/forms_tests/tests/test_media.py
+++ b/tests/forms_tests/tests/test_media.py
@@ -541,3 +541,33 @@ def test_merge_warning(self):
         msg = 'Detected duplicate Media files in an opposite order:\n1\n2'
         with self.assertWarnsMessage(RuntimeWarning, msg):
             self.assertEqual(Media.merge([1, 2], [2, 1]), [1, 2])
+
+    def test_merge_js_three_way(self):
+        """
+        The relative order of scripts is preserved in a three-way merge.
+        """
+        # custom_widget.js doesn't depend on jquery.js.
+        widget1 = Media(js=['custom_widget.js'])
+        widget2 = Media(js=['jquery.js', 'uses_jquery.js'])
+        form_media = widget1 + widget2
+        # The relative ordering of custom_widget.js and jquery.js has been
+        # established (but without a real need to).
+        self.assertEqual(form_media._js, ['custom_widget.js', 'jquery.js', 'uses_jquery.js'])
+        # The inline also uses custom_widget.js. This time, it's at the end.
+        inline_media = Media(js=['jquery.js', 'also_jquery.js']) + Media(js=['custom_widget.js'])
+        merged = form_media + inline_media
+        self.assertEqual(merged._js, ['custom_widget.js', 'jquery.js', 'uses_jquery.js', 'also_jquery.js'])
+
+    def test_merge_css_three_way(self):
+        widget1 = Media(css={'screen': ['a.css']})
+        widget2 = Media(css={'screen': ['b.css']})
+        widget3 = Media(css={'all': ['c.css']})
+        form1 = widget1 + widget2
+        form2 = widget2 + widget1
+        # form1 and form2 have a.css and b.css in different order...
+        self.assertEqual(form1._css, {'screen': ['a.css', 'b.css']})
+        self.assertEqual(form2._css, {'screen': ['b.css', 'a.css']})
+        # ...but merging succeeds as the relative ordering of a.css and b.css
+        # was never specified.
+        merged = widget3 + form1 + form2
+        self.assertEqual(merged._css, {'screen': ['a.css', 'b.css'], 'all': ['c.css']})

```


## Code snippets

### 1 - django/contrib/admin/options.py:

Start line: 1993, End line: 2014

```python
class InlineModelAdmin(BaseModelAdmin):

    @property
    def media(self):
        extra = '' if settings.DEBUG else '.min'
        js = ['vendor/jquery/jquery%s.js' % extra, 'jquery.init.js',
              'inlines%s.js' % extra]
        if self.filter_vertical or self.filter_horizontal:
            js.extend(['SelectBox.js', 'SelectFilter2.js'])
        if self.classes and 'collapse' in self.classes:
            js.append('collapse%s.js' % extra)
        return forms.Media(js=['admin/js/%s' % url for url in js])

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
### 2 - django/forms/widgets.py:

Start line: 98, End line: 140

```python
@html_safe
class Media:

    @staticmethod
    def merge(list_1, list_2):
        """
        Merge two lists while trying to keep the relative order of the elements.
        Warn if the lists have the same two elements in a different relative
        order.

        For static assets it can be important to have them included in the DOM
        in a certain order. In JavaScript you may not be able to reference a
        global or in CSS you might want to override a style.
        """
        # Start with a copy of list_1.
        combined_list = list(list_1)
        last_insert_index = len(list_1)
        # Walk list_2 in reverse, inserting each element into combined_list if
        # it doesn't already exist.
        for path in reversed(list_2):
            try:
                # Does path already exist in the list?
                index = combined_list.index(path)
            except ValueError:
                # Add path to combined_list since it doesn't exist.
                combined_list.insert(last_insert_index, path)
            else:
                if index > last_insert_index:
                    warnings.warn(
                        'Detected duplicate Media files in an opposite order:\n'
                        '%s\n%s' % (combined_list[last_insert_index], combined_list[index]),
                        MediaOrderConflictWarning,
                    )
                # path already exists in the list. Update last_insert_index so
                # that the following elements are inserted in front of this one.
                last_insert_index = index
        return combined_list

    def __add__(self, other):
        combined = Media()
        combined._js = self.merge(self._js, other._js)
        combined._css = {
            medium: self.merge(self._css.get(medium, []), other._css.get(medium, []))
            for medium in self._css.keys() | other._css.keys()
        }
        return combined
```
### 3 - django/forms/widgets.py:

Start line: 1, End line: 37

```python
"""
HTML Widget classes
"""

import copy
import datetime
import re
import warnings
from itertools import chain

from django.conf import settings
from django.forms.utils import to_current_timezone
from django.templatetags.static import static
from django.utils import datetime_safe, formats
from django.utils.dates import MONTHS
from django.utils.formats import get_format
from django.utils.html import format_html, html_safe
from django.utils.safestring import mark_safe
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
### 4 - django/contrib/admin/widgets.py:

Start line: 440, End line: 468

```python
class AutocompleteMixin:

    @property
    def media(self):
        extra = '' if settings.DEBUG else '.min'
        i18n_name = SELECT2_TRANSLATIONS.get(get_language())
        i18n_file = ('admin/js/vendor/select2/i18n/%s.js' % i18n_name,) if i18n_name else ()
        return forms.Media(
            js=(
                'admin/js/vendor/jquery/jquery%s.js' % extra,
                'admin/js/vendor/select2/select2.full%s.js' % extra,
            ) + i18n_file + (
                'admin/js/jquery.init.js',
                'admin/js/autocomplete.js',
            ),
            css={
                'screen': (
                    'admin/css/vendor/select2/select2%s.css' % extra,
                    'admin/css/autocomplete.css',
                ),
            },
        )


class AutocompleteSelect(AutocompleteMixin, forms.Select):
    pass


class AutocompleteSelectMultiple(AutocompleteMixin, forms.SelectMultiple):
    pass
```
### 5 - django/contrib/admin/widgets.py:

Start line: 53, End line: 74

```python
class AdminDateWidget(forms.DateInput):
    class Media:
        js = [
            'admin/js/calendar.js',
            'admin/js/admin/DateTimeShortcuts.js',
        ]

    def __init__(self, attrs=None, format=None):
        attrs = {'class': 'vDateField', 'size': '10', **(attrs or {})}
        super().__init__(attrs=attrs, format=format)


class AdminTimeWidget(forms.TimeInput):
    class Media:
        js = [
            'admin/js/calendar.js',
            'admin/js/admin/DateTimeShortcuts.js',
        ]

    def __init__(self, attrs=None, format=None):
        attrs = {'class': 'vTimeField', 'size': '8', **(attrs or {})}
        super().__init__(attrs=attrs, format=format)
```
### 6 - django/contrib/contenttypes/admin.py:

Start line: 83, End line: 130

```python
class GenericInlineModelAdmin(InlineModelAdmin):
    ct_field = "content_type"
    ct_fk_field = "object_id"
    formset = BaseGenericInlineFormSet

    checks_class = GenericInlineModelAdminChecks

    def get_formset(self, request, obj=None, **kwargs):
        if 'fields' in kwargs:
            fields = kwargs.pop('fields')
        else:
            fields = flatten_fieldsets(self.get_fieldsets(request, obj))
        exclude = [*(self.exclude or []), *self.get_readonly_fields(request, obj)]
        if self.exclude is None and hasattr(self.form, '_meta') and self.form._meta.exclude:
            # Take the custom ModelForm's Meta.exclude into account only if the
            # GenericInlineModelAdmin doesn't define its own.
            exclude.extend(self.form._meta.exclude)
        exclude = exclude or None
        can_delete = self.can_delete and self.has_delete_permission(request, obj)
        defaults = {
            'ct_field': self.ct_field,
            'fk_field': self.ct_fk_field,
            'form': self.form,
            'formfield_callback': partial(self.formfield_for_dbfield, request=request),
            'formset': self.formset,
            'extra': self.get_extra(request, obj),
            'can_delete': can_delete,
            'can_order': False,
            'fields': fields,
            'min_num': self.get_min_num(request, obj),
            'max_num': self.get_max_num(request, obj),
            'exclude': exclude,
            **kwargs,
        }

        if defaults['fields'] is None and not modelform_defines_fields(defaults['form']):
            defaults['fields'] = ALL_FIELDS

        return generic_inlineformset_factory(self.model, **defaults)


class GenericStackedInline(GenericInlineModelAdmin):
    template = 'admin/edit_inline/stacked.html'


class GenericTabularInline(GenericInlineModelAdmin):
    template = 'admin/edit_inline/tabular.html'
```
### 7 - django/contrib/admin/options.py:

Start line: 99, End line: 129

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
### 8 - django/contrib/admin/options.py:

Start line: 1, End line: 96

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
from django.db.models.fields import BLANK_CHOICE_DASH
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
### 9 - django/contrib/admin/widgets.py:

Start line: 1, End line: 50

```python
"""
Form Widget classes specific to the Django admin site.
"""
import copy
import json

from django import forms
from django.conf import settings
from django.core.exceptions import ValidationError
from django.db.models.deletion import CASCADE
from django.urls import reverse
from django.urls.exceptions import NoReverseMatch
from django.utils.html import smart_urlquote
from django.utils.safestring import mark_safe
from django.utils.text import Truncator
from django.utils.translation import get_language, gettext as _


class FilteredSelectMultiple(forms.SelectMultiple):
    """
    A SelectMultiple with a JavaScript filter interface.

    Note that the resulting JavaScript assumes that the jsi18n
    catalog has been loaded in the page
    """
    @property
    def media(self):
        extra = '' if settings.DEBUG else '.min'
        js = [
            'vendor/jquery/jquery%s.js' % extra,
            'jquery.init.js',
            'core.js',
            'SelectBox.js',
            'SelectFilter2.js',
        ]
        return forms.Media(js=["admin/js/%s" % path for path in js])

    def __init__(self, verbose_name, is_stacked, attrs=None, choices=()):
        self.verbose_name = verbose_name
        self.is_stacked = is_stacked
        super().__init__(attrs, choices)

    def get_context(self, name, value, attrs):
        context = super().get_context(name, value, attrs)
        context['widget']['attrs']['class'] = 'selectfilter'
        if self.is_stacked:
            context['widget']['attrs']['class'] += 'stacked'
        context['widget']['attrs']['data-field-name'] = self.verbose_name
        context['widget']['attrs']['data-is-stacked'] = int(self.is_stacked)
        return context
```
### 10 - django/contrib/admin/options.py:

Start line: 623, End line: 640

```python
class ModelAdmin(BaseModelAdmin):

    @property
    def urls(self):
        return self.get_urls()

    @property
    def media(self):
        extra = '' if settings.DEBUG else '.min'
        js = [
            'vendor/jquery/jquery%s.js' % extra,
            'jquery.init.js',
            'core.js',
            'admin/RelatedObjectLookups.js',
            'actions%s.js' % extra,
            'urlify.js',
            'prepopulate%s.js' % extra,
            'vendor/xregexp/xregexp%s.js' % extra,
        ]
        return forms.Media(js=['admin/js/%s' % url for url in js])
```
### 19 - django/forms/widgets.py:

Start line: 830, End line: 870

```python
class MultiWidget(Widget):

    def id_for_label(self, id_):
        if id_:
            id_ += '_0'
        return id_

    def value_from_datadict(self, data, files, name):
        return [widget.value_from_datadict(data, files, name + '_%s' % i) for i, widget in enumerate(self.widgets)]

    def value_omitted_from_data(self, data, files, name):
        return all(
            widget.value_omitted_from_data(data, files, name + '_%s' % i)
            for i, widget in enumerate(self.widgets)
        )

    def decompress(self, value):
        """
        Return a list of decompressed values for the given compressed value.
        The given value can be assumed to be valid, but not necessarily
        non-empty.
        """
        raise NotImplementedError('Subclasses must implement this method.')

    def _get_media(self):
        """
        Media for a multiwidget is the combination of all media of the
        subwidgets.
        """
        media = Media()
        for w in self.widgets:
            media = media + w.media
        return media
    media = property(_get_media)

    def __deepcopy__(self, memo):
        obj = super().__deepcopy__(memo)
        obj.widgets = copy.deepcopy(self.widgets)
        return obj

    @property
    def needs_multipart_form(self):
        return any(w.needs_multipart_form for w in self.widgets)
```
### 30 - django/forms/widgets.py:

Start line: 40, End line: 96

```python
@html_safe
class Media:
    def __init__(self, media=None, css=None, js=None):
        if media is not None:
            css = getattr(media, 'css', {})
            js = getattr(media, 'js', [])
        else:
            if css is None:
                css = {}
            if js is None:
                js = []
        self._css = css
        self._js = js

    def __repr__(self):
        return 'Media(css=%r, js=%r)' % (self._css, self._js)

    def __str__(self):
        return self.render()

    def render(self):
        return mark_safe('\n'.join(chain.from_iterable(getattr(self, 'render_' + name)() for name in MEDIA_TYPES)))

    def render_js(self):
        return [
            format_html(
                '<script type="text/javascript" src="{}"></script>',
                self.absolute_path(path)
            ) for path in self._js
        ]

    def render_css(self):
        # To keep rendering order consistent, we can't just iterate over items().
        # We need to sort the keys, and iterate over the sorted list.
        media = sorted(self._css)
        return chain.from_iterable([
            format_html(
                '<link href="{}" type="text/css" media="{}" rel="stylesheet">',
                self.absolute_path(path), medium
            ) for path in self._css[medium]
        ] for medium in media)

    def absolute_path(self, path):
        """
        Given a relative or absolute path to a static asset, return an absolute
        path. An absolute path will be returned unchanged while a relative path
        will be passed to django.templatetags.static.static().
        """
        if path.startswith(('http://', 'https://', '/')):
            return path
        return static(path)

    def __getitem__(self, name):
        """Return a Media object that only contains media of the given type."""
        if name in MEDIA_TYPES:
            return Media(**{str(name): getattr(self, '_' + name)})
        raise KeyError('Unknown media type "%s"' % name)
```
### 37 - django/forms/widgets.py:

Start line: 753, End line: 776

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
### 38 - django/forms/widgets.py:

Start line: 143, End line: 179

```python
def media_property(cls):
    def _media(self):
        # Get the media property of the superclass, if it exists
        sup_cls = super(cls, self)
        try:
            base = sup_cls.media
        except AttributeError:
            base = Media()

        # Get the media definition for this class
        definition = getattr(cls, 'Media', None)
        if definition:
            extend = getattr(definition, 'extend', True)
            if extend:
                if extend is True:
                    m = base
                else:
                    m = Media()
                    for medium in extend:
                        m = m + base[medium]
                return m + Media(definition)
            return Media(definition)
        return base
    return property(_media)


class MediaDefiningClass(type):
    """
    Metaclass for classes that can have media definitions.
    """
    def __new__(mcs, name, bases, attrs):
        new_class = super(MediaDefiningClass, mcs).__new__(mcs, name, bases, attrs)

        if 'media' not in attrs:
            new_class.media = media_property(new_class)

        return new_class
```
### 59 - django/forms/widgets.py:

Start line: 799, End line: 828

```python
class MultiWidget(Widget):

    def get_context(self, name, value, attrs):
        context = super().get_context(name, value, attrs)
        if self.is_localized:
            for widget in self.widgets:
                widget.is_localized = self.is_localized
        # value is a list of values, each corresponding to a widget
        # in self.widgets.
        if not isinstance(value, list):
            value = self.decompress(value)

        final_attrs = context['widget']['attrs']
        input_type = final_attrs.pop('type', None)
        id_ = final_attrs.get('id')
        subwidgets = []
        for i, widget in enumerate(self.widgets):
            if input_type is not None:
                widget.input_type = input_type
            widget_name = '%s_%s' % (name, i)
            try:
                widget_value = value[i]
            except IndexError:
                widget_value = None
            if id_:
                widget_attrs = final_attrs.copy()
                widget_attrs['id'] = '%s_%s' % (id_, i)
            else:
                widget_attrs = final_attrs
            subwidgets.append(widget.get_context(widget_name, widget_value, widget_attrs)['widget'])
        context['widget']['subwidgets'] = subwidgets
        return context
```
### 65 - django/forms/widgets.py:

Start line: 325, End line: 359

```python
class MultipleHiddenInput(HiddenInput):
    """
    Handle <input type="hidden"> for fields that have a list
    of values.
    """
    template_name = 'django/forms/widgets/multiple_hidden.html'

    def get_context(self, name, value, attrs):
        context = super().get_context(name, value, attrs)
        final_attrs = context['widget']['attrs']
        id_ = context['widget']['attrs'].get('id')

        subwidgets = []
        for index, value_ in enumerate(context['widget']['value']):
            widget_attrs = final_attrs.copy()
            if id_:
                # An ID attribute was given. Add a numeric index as a suffix
                # so that the inputs don't all have the same ID attribute.
                widget_attrs['id'] = '%s_%s' % (id_, index)
            widget = HiddenInput()
            widget.is_required = self.is_required
            subwidgets.append(widget.get_context(name, value_, widget_attrs)['widget'])

        context['widget']['subwidgets'] = subwidgets
        return context

    def value_from_datadict(self, data, files, name):
        try:
            getter = data.getlist
        except AttributeError:
            getter = data.get
        return getter(name)

    def format_value(self, value):
        return [] if value is None else value
```
### 66 - django/forms/widgets.py:

Start line: 362, End line: 379

```python
class FileInput(Input):
    input_type = 'file'
    needs_multipart_form = True
    template_name = 'django/forms/widgets/file.html'

    def format_value(self, value):
        """File input never renders a value."""
        return

    def value_from_datadict(self, data, files, name):
        "File widgets take data from FILES, not POST"
        return files.get(name)

    def value_omitted_from_data(self, data, files, name):
        return name not in files


FILE_INPUT_CONTRADICTION = object()
```
### 67 - django/forms/widgets.py:

Start line: 779, End line: 797

```python
class MultiWidget(Widget):
    """
    A widget that is composed of multiple widgets.

    In addition to the values added by Widget.get_context(), this widget
    adds a list of subwidgets to the context as widget['subwidgets'].
    These can be looped over and rendered like normal widgets.

    You'll probably want to use this class with MultiValueField.
    """
    template_name = 'django/forms/widgets/multiwidget.html'

    def __init__(self, widgets, attrs=None):
        self.widgets = [w() if isinstance(w, type) else w for w in widgets]
        super().__init__(attrs)

    @property
    def is_hidden(self):
        return all(w.is_hidden for w in self.widgets)
```
### 72 - django/forms/widgets.py:

Start line: 900, End line: 909

```python
class SplitHiddenDateTimeWidget(SplitDateTimeWidget):
    """
    A widget that splits datetime input into two <input type="hidden"> inputs.
    """
    template_name = 'django/forms/widgets/splithiddendatetime.html'

    def __init__(self, attrs=None, date_format=None, time_format=None, date_attrs=None, time_attrs=None):
        super().__init__(attrs, date_format, time_format, date_attrs, time_attrs)
        for widget in self.widgets:
            widget.input_type = 'hidden'
```
### 75 - django/forms/widgets.py:

Start line: 912, End line: 958

```python
class SelectDateWidget(Widget):
    """
    A widget that splits date input into three <select> boxes.

    This also serves as an example of a Widget that has more than one HTML
    element and hence implements value_from_datadict.
    """
    none_value = ('', '---')
    month_field = '%s_month'
    day_field = '%s_day'
    year_field = '%s_year'
    template_name = 'django/forms/widgets/select_date.html'
    input_type = 'select'
    select_widget = Select
    date_re = re.compile(r'(\d{4}|0)-(\d\d?)-(\d\d?)$')

    def __init__(self, attrs=None, years=None, months=None, empty_label=None):
        self.attrs = attrs or {}

        # Optional list or tuple of years to use in the "year" select box.
        if years:
            self.years = years
        else:
            this_year = datetime.date.today().year
            self.years = range(this_year, this_year + 10)

        # Optional dict of months to use in the "month" select box.
        if months:
            self.months = months
        else:
            self.months = MONTHS

        # Optional string, list, or tuple to use as empty_label.
        if isinstance(empty_label, (list, tuple)):
            if not len(empty_label) == 3:
                raise ValueError('empty_label list/tuple must have 3 elements.')

            self.year_none_value = ('', empty_label[0])
            self.month_none_value = ('', empty_label[1])
            self.day_none_value = ('', empty_label[2])
        else:
            if empty_label is not None:
                self.none_value = ('', empty_label)

            self.year_none_value = self.none_value
            self.month_none_value = self.none_value
            self.day_none_value = self.none_value
```
### 78 - django/forms/widgets.py:

Start line: 731, End line: 750

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
    input_type = 'radio'
    template_name = 'django/forms/widgets/radio.html'
    option_template_name = 'django/forms/widgets/radio_option.html'
```
### 87 - django/forms/widgets.py:

Start line: 873, End line: 897

```python
class SplitDateTimeWidget(MultiWidget):
    """
    A widget that splits datetime input into two <input type="text"> boxes.
    """
    supports_microseconds = False
    template_name = 'django/forms/widgets/splitdatetime.html'

    def __init__(self, attrs=None, date_format=None, time_format=None, date_attrs=None, time_attrs=None):
        widgets = (
            DateInput(
                attrs=attrs if date_attrs is None else date_attrs,
                format=date_format,
            ),
            TimeInput(
                attrs=attrs if time_attrs is None else time_attrs,
                format=time_format,
            ),
        )
        super().__init__(widgets)

    def decompress(self, value):
        if value:
            value = to_current_timezone(value)
            return [value.date(), value.time()]
        return [None, None]
```
### 89 - django/forms/widgets.py:

Start line: 428, End line: 449

```python
class ClearableFileInput(FileInput):

    def value_from_datadict(self, data, files, name):
        upload = super().value_from_datadict(data, files, name)
        if not self.is_required and CheckboxInput().value_from_datadict(
                data, files, self.clear_checkbox_name(name)):

            if upload:
                # If the user contradicts themselves (uploads a new file AND
                # checks the "clear" checkbox), we return a unique marker
                # object that FileField will turn into a ValidationError.
                return FILE_INPUT_CONTRADICTION
            # False signals to clear any existing value, as opposed to just None
            return False
        return upload

    def use_required_attribute(self, initial):
        return super().use_required_attribute(initial) and not initial

    def value_omitted_from_data(self, data, files, name):
        return (
            super().value_omitted_from_data(data, files, name) and
            self.clear_checkbox_name(name) not in data
        )
```
### 97 - django/forms/widgets.py:

Start line: 182, End line: 264

```python
class Widget(metaclass=MediaDefiningClass):
    needs_multipart_form = False  # Determines does this widget need multipart form
    is_localized = False
    is_required = False
    supports_microseconds = True

    def __init__(self, attrs=None):
        self.attrs = {} if attrs is None else attrs.copy()

    def __deepcopy__(self, memo):
        obj = copy.copy(self)
        obj.attrs = self.attrs.copy()
        memo[id(self)] = obj
        return obj

    @property
    def is_hidden(self):
        return self.input_type == 'hidden' if hasattr(self, 'input_type') else False

    def subwidgets(self, name, value, attrs=None):
        context = self.get_context(name, value, attrs)
        yield context['widget']

    def format_value(self, value):
        """
        Return a value as it should appear when rendered in a template.
        """
        if value == '' or value is None:
            return None
        if self.is_localized:
            return formats.localize_input(value)
        return str(value)

    def get_context(self, name, value, attrs):
        context = {}
        context['widget'] = {
            'name': name,
            'is_hidden': self.is_hidden,
            'required': self.is_required,
            'value': self.format_value(value),
            'attrs': self.build_attrs(self.attrs, attrs),
            'template_name': self.template_name,
        }
        return context

    def render(self, name, value, attrs=None, renderer=None):
        """Render the widget as an HTML string."""
        context = self.get_context(name, value, attrs)
        return self._render(self.template_name, context, renderer)

    def _render(self, template_name, context, renderer=None):
        if renderer is None:
            renderer = get_default_renderer()
        return mark_safe(renderer.render(template_name, context))

    def build_attrs(self, base_attrs, extra_attrs=None):
        """Build an attribute dictionary."""
        return {**base_attrs, **(extra_attrs or {})}

    def value_from_datadict(self, data, files, name):
        """
        Given a dictionary of data and this widget's name, return the value
        of this widget or None if it's not provided.
        """
        return data.get(name)

    def value_omitted_from_data(self, data, files, name):
        return name not in data

    def id_for_label(self, id_):
        """
        Return the HTML ID attribute of this Widget for use by a <label>,
        given the ID of the field. Return None if no ID is available.

        This hook is necessary because some widgets have multiple HTML
        elements and, thus, multiple IDs. In that case, this method should
        return an ID value that corresponds to the first ID in the widget's
        tags.
        """
        return id_

    def use_required_attribute(self, initial):
        return not self.is_hidden
```
### 103 - django/forms/widgets.py:

Start line: 960, End line: 994

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
### 111 - django/forms/widgets.py:

Start line: 452, End line: 492

```python
class Textarea(Widget):
    template_name = 'django/forms/widgets/textarea.html'

    def __init__(self, attrs=None):
        # Use slightly better defaults than HTML's 20x2 box
        default_attrs = {'cols': '40', 'rows': '10'}
        if attrs:
            default_attrs.update(attrs)
        super().__init__(default_attrs)


class DateTimeBaseInput(TextInput):
    format_key = ''
    supports_microseconds = False

    def __init__(self, attrs=None, format=None):
        super().__init__(attrs)
        self.format = format or None

    def format_value(self, value):
        return formats.localize_input(value, self.format or formats.get_format(self.format_key)[0])


class DateInput(DateTimeBaseInput):
    format_key = 'DATE_INPUT_FORMATS'
    template_name = 'django/forms/widgets/date.html'


class DateTimeInput(DateTimeBaseInput):
    format_key = 'DATETIME_INPUT_FORMATS'
    template_name = 'django/forms/widgets/datetime.html'


class TimeInput(DateTimeBaseInput):
    format_key = 'TIME_INPUT_FORMATS'
    template_name = 'django/forms/widgets/time.html'


# Defined at module level so that CheckboxInput is picklable (#17976)
def boolean_check(v):
    return not (v is False or v is None or v == '')
```
### 113 - django/forms/widgets.py:

Start line: 659, End line: 690

```python
class Select(ChoiceWidget):
    input_type = 'select'
    template_name = 'django/forms/widgets/select.html'
    option_template_name = 'django/forms/widgets/select_option.html'
    add_id_index = False
    checked_attribute = {'selected': True}
    option_inherits_attrs = False

    def get_context(self, name, value, attrs):
        context = super().get_context(name, value, attrs)
        if self.allow_multiple_selected:
            context['widget']['attrs']['multiple'] = True
        return context

    @staticmethod
    def _choice_has_empty_value(choice):
        """Return True if the choice's value is empty string or None."""
        value, _ = choice
        return value is None or value == ''

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
        return use_required_attribute and first_choice is not None and self._choice_has_empty_value(first_choice)
```
### 139 - django/forms/widgets.py:

Start line: 1021, End line: 1040

```python
class SelectDateWidget(Widget):

    @staticmethod
    def _parse_date_fmt():
        fmt = get_format('DATE_FORMAT')
        escaped = False
        for char in fmt:
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char in 'Yy':
                yield 'year'
            elif char in 'bEFMmNn':
                yield 'month'
            elif char in 'dj':
                yield 'day'

    def id_for_label(self, id_):
        for first_select in self._parse_date_fmt():
            return '%s_%s' % (id_, first_select)
        return '%s_month' % id_
```
### 140 - django/forms/widgets.py:

Start line: 1042, End line: 1068

```python
class SelectDateWidget(Widget):

    def value_from_datadict(self, data, files, name):
        y = data.get(self.year_field % name)
        m = data.get(self.month_field % name)
        d = data.get(self.day_field % name)
        if y == m == d == '':
            return None
        if y is not None and m is not None and d is not None:
            if settings.USE_L10N:
                input_format = get_format('DATE_INPUT_FORMATS')[0]
                try:
                    date_value = datetime.date(int(y), int(m), int(d))
                except ValueError:
                    pass
                else:
                    date_value = datetime_safe.new_date(date_value)
                    return date_value.strftime(input_format)
            # Return pseudo-ISO dates with zeros for any unselected values,
            # e.g. '2017-0-23'.
            return '%s-%s-%s' % (y or 0, m or 0, d or 0)
        return data.get(name)

    def value_omitted_from_data(self, data, files, name):
        return not any(
            ('{}_{}'.format(name, interval) in data)
            for interval in ('year', 'month', 'day')
        )
```
### 152 - django/forms/widgets.py:

Start line: 414, End line: 426

```python
class ClearableFileInput(FileInput):

    def get_context(self, name, value, attrs):
        context = super().get_context(name, value, attrs)
        checkbox_name = self.clear_checkbox_name(name)
        checkbox_id = self.clear_checkbox_id(checkbox_name)
        context['widget'].update({
            'checkbox_name': checkbox_name,
            'checkbox_id': checkbox_id,
            'is_initial': self.is_initial(value),
            'input_text': self.input_text,
            'initial_text': self.initial_text,
            'clear_checkbox_label': self.clear_checkbox_label,
        })
        return context
```
### 166 - django/forms/widgets.py:

Start line: 606, End line: 625

```python
class ChoiceWidget(Widget):

    def create_option(self, name, value, label, selected, index, subindex=None, attrs=None):
        index = str(index) if subindex is None else "%s_%s" % (index, subindex)
        if attrs is None:
            attrs = {}
        option_attrs = self.build_attrs(self.attrs, attrs) if self.option_inherits_attrs else {}
        if selected:
            option_attrs.update(self.checked_attribute)
        if 'id' in option_attrs:
            option_attrs['id'] = self.id_for_label(option_attrs['id'], index)
        return {
            'name': name,
            'value': value,
            'label': label,
            'selected': selected,
            'index': index,
            'attrs': option_attrs,
            'type': self.input_type,
            'template_name': self.option_template_name,
            'wrap_label': True,
        }
```
