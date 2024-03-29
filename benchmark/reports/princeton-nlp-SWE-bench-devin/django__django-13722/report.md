# django__django-13722

| **django/django** | `600ff26a85752071da36e3a94c66dd8a77ee314a` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 5803 |
| **Any found context length** | 5803 |
| **Avg pos** | 34.0 |
| **Min pos** | 17 |
| **Max pos** | 17 |
| **Top file pos** | 2 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/contrib/admin/options.py b/django/contrib/admin/options.py
--- a/django/contrib/admin/options.py
+++ b/django/contrib/admin/options.py
@@ -1946,6 +1946,20 @@ def history_view(self, request, object_id, extra_context=None):
             "admin/object_history.html"
         ], context)
 
+    def get_formset_kwargs(self, request, obj, inline, prefix):
+        formset_params = {
+            'instance': obj,
+            'prefix': prefix,
+            'queryset': inline.get_queryset(request),
+        }
+        if request.method == 'POST':
+            formset_params.update({
+                'data': request.POST.copy(),
+                'files': request.FILES,
+                'save_as_new': '_saveasnew' in request.POST
+            })
+        return formset_params
+
     def _create_formsets(self, request, obj, change):
         "Helper function to generate formsets for add/change_view."
         formsets = []
@@ -1959,17 +1973,7 @@ def _create_formsets(self, request, obj, change):
             prefixes[prefix] = prefixes.get(prefix, 0) + 1
             if prefixes[prefix] != 1 or not prefix:
                 prefix = "%s-%s" % (prefix, prefixes[prefix])
-            formset_params = {
-                'instance': obj,
-                'prefix': prefix,
-                'queryset': inline.get_queryset(request),
-            }
-            if request.method == 'POST':
-                formset_params.update({
-                    'data': request.POST.copy(),
-                    'files': request.FILES,
-                    'save_as_new': '_saveasnew' in request.POST
-                })
+            formset_params = self.get_formset_kwargs(request, obj, inline, prefix)
             formset = FormSet(**formset_params)
 
             def user_deleted_form(request, obj, formset, index):

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/contrib/admin/options.py | 1949 | 1949 | 17 | 2 | 5803
| django/contrib/admin/options.py | 1962 | 1972 | 17 | 2 | 5803


## Problem Statement

```
Add a hook to customize the admin's formsets parameters
Description
	
New feature that adds a method on InlineModelAdmin for providing initial data for the inline formset. By default there is no implementation, although one could be implemented to use GET parameters like get_changeform_initial_data, but it wouldn't be trivial due to the list nature of formset initial data.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 django/contrib/admin/helpers.py | 323 | 349| 171 | 171 | 3331 | 
| 2 | 1 django/contrib/admin/helpers.py | 264 | 287| 235 | 406 | 3331 | 
| 3 | 1 django/contrib/admin/helpers.py | 289 | 321| 282 | 688 | 3331 | 
| 4 | **2 django/contrib/admin/options.py** | 1482 | 1505| 319 | 1007 | 21870 | 
| 5 | 2 django/contrib/admin/helpers.py | 240 | 262| 211 | 1218 | 21870 | 
| 6 | **2 django/contrib/admin/options.py** | 2050 | 2083| 376 | 1594 | 21870 | 
| 7 | 2 django/contrib/admin/helpers.py | 352 | 370| 181 | 1775 | 21870 | 
| 8 | **2 django/contrib/admin/options.py** | 1995 | 2048| 454 | 2229 | 21870 | 
| 9 | **2 django/contrib/admin/options.py** | 1507 | 1538| 295 | 2524 | 21870 | 
| 10 | 3 django/contrib/contenttypes/admin.py | 81 | 128| 410 | 2934 | 22851 | 
| 11 | **3 django/contrib/admin/options.py** | 1540 | 1626| 760 | 3694 | 22851 | 
| 12 | 3 django/contrib/admin/helpers.py | 383 | 399| 138 | 3832 | 22851 | 
| 13 | 4 django/forms/models.py | 927 | 959| 353 | 4185 | 34625 | 
| 14 | **4 django/contrib/admin/options.py** | 1127 | 1172| 482 | 4667 | 34625 | 
| 15 | **4 django/contrib/admin/options.py** | 2085 | 2137| 451 | 5118 | 34625 | 
| 16 | 5 django/forms/formsets.py | 385 | 417| 282 | 5400 | 38756 | 
| **-> 17 <-** | **5 django/contrib/admin/options.py** | 1949 | 1992| 403 | 5803 | 38756 | 
| 18 | 5 django/contrib/admin/helpers.py | 35 | 69| 230 | 6033 | 38756 | 
| 19 | **5 django/contrib/admin/options.py** | 780 | 802| 207 | 6240 | 38756 | 
| 20 | **5 django/contrib/admin/options.py** | 1627 | 1652| 291 | 6531 | 38756 | 
| 21 | **5 django/contrib/admin/options.py** | 100 | 130| 223 | 6754 | 38756 | 
| 22 | 5 django/contrib/admin/helpers.py | 372 | 381| 134 | 6888 | 38756 | 
| 23 | **5 django/contrib/admin/options.py** | 132 | 187| 604 | 7492 | 38756 | 
| 24 | **5 django/contrib/admin/options.py** | 286 | 375| 636 | 8128 | 38756 | 
| 25 | 5 django/forms/models.py | 961 | 994| 367 | 8495 | 38756 | 
| 26 | 5 django/forms/models.py | 895 | 925| 289 | 8784 | 38756 | 
| 27 | 6 django/contrib/contenttypes/forms.py | 1 | 49| 401 | 9185 | 39543 | 
| 28 | 6 django/contrib/admin/helpers.py | 402 | 425| 195 | 9380 | 39543 | 
| 29 | **6 django/contrib/admin/options.py** | 2166 | 2201| 315 | 9695 | 39543 | 
| 30 | **6 django/contrib/admin/options.py** | 767 | 778| 114 | 9809 | 39543 | 
| 31 | 6 django/contrib/admin/helpers.py | 72 | 90| 152 | 9961 | 39543 | 
| 32 | 6 django/forms/models.py | 601 | 633| 270 | 10231 | 39543 | 
| 33 | **6 django/contrib/admin/options.py** | 669 | 715| 449 | 10680 | 39543 | 
| 34 | **6 django/contrib/admin/options.py** | 1 | 97| 762 | 11442 | 39543 | 
| 35 | **6 django/contrib/admin/options.py** | 2139 | 2164| 250 | 11692 | 39543 | 
| 36 | **6 django/contrib/admin/options.py** | 1115 | 1125| 125 | 11817 | 39543 | 
| 37 | **6 django/contrib/admin/options.py** | 596 | 609| 122 | 11939 | 39543 | 
| 38 | 6 django/forms/models.py | 1053 | 1095| 380 | 12319 | 39543 | 
| 39 | 6 django/forms/models.py | 286 | 314| 288 | 12607 | 39543 | 
| 40 | **6 django/contrib/admin/options.py** | 1654 | 1665| 151 | 12758 | 39543 | 
| 41 | 6 django/contrib/contenttypes/forms.py | 52 | 85| 385 | 13143 | 39543 | 
| 42 | 6 django/forms/models.py | 635 | 652| 167 | 13310 | 39543 | 
| 43 | **6 django/contrib/admin/options.py** | 1839 | 1907| 584 | 13894 | 39543 | 
| 44 | **6 django/contrib/admin/options.py** | 1755 | 1837| 750 | 14644 | 39543 | 
| 45 | 7 django/contrib/admin/templatetags/admin_modify.py | 1 | 45| 372 | 15016 | 40511 | 
| 46 | 7 django/forms/models.py | 564 | 599| 300 | 15316 | 40511 | 
| 47 | 7 django/forms/models.py | 782 | 820| 314 | 15630 | 40511 | 
| 48 | 7 django/forms/formsets.py | 197 | 226| 217 | 15847 | 40511 | 
| 49 | 8 django/views/generic/edit.py | 1 | 67| 479 | 16326 | 42227 | 
| 50 | 9 django/contrib/admin/views/main.py | 1 | 45| 324 | 16650 | 46623 | 
| 51 | **9 django/contrib/admin/options.py** | 551 | 594| 297 | 16947 | 46623 | 
| 52 | 9 django/forms/formsets.py | 108 | 121| 126 | 17073 | 46623 | 
| 53 | 9 django/forms/models.py | 389 | 417| 240 | 17313 | 46623 | 
| 54 | 10 django/contrib/admin/forms.py | 1 | 31| 185 | 17498 | 46808 | 
| 55 | **10 django/contrib/admin/options.py** | 1086 | 1113| 188 | 17686 | 46808 | 
| 56 | 10 django/forms/models.py | 201 | 211| 131 | 17817 | 46808 | 
| 57 | 11 django/forms/forms.py | 398 | 425| 190 | 18007 | 50813 | 
| 58 | 11 django/forms/models.py | 214 | 283| 616 | 18623 | 50813 | 
| 59 | 11 django/contrib/contenttypes/admin.py | 1 | 78| 571 | 19194 | 50813 | 
| 60 | 11 django/forms/formsets.py | 142 | 167| 201 | 19395 | 50813 | 
| 61 | 11 django/contrib/admin/helpers.py | 124 | 150| 220 | 19615 | 50813 | 
| 62 | 11 django/forms/models.py | 822 | 863| 449 | 20064 | 50813 | 
| 63 | 11 django/forms/forms.py | 427 | 449| 195 | 20259 | 50813 | 
| 64 | 12 django/contrib/auth/admin.py | 40 | 99| 504 | 20763 | 52539 | 
| 65 | 12 django/contrib/admin/helpers.py | 93 | 121| 249 | 21012 | 52539 | 
| 66 | **12 django/contrib/admin/options.py** | 244 | 284| 376 | 21388 | 52539 | 
| 67 | 12 django/views/generic/edit.py | 70 | 101| 269 | 21657 | 52539 | 
| 68 | 13 django/contrib/admin/checks.py | 1021 | 1033| 116 | 21773 | 61676 | 
| 69 | 13 django/forms/formsets.py | 419 | 457| 359 | 22132 | 61676 | 
| 70 | 13 django/contrib/admin/helpers.py | 1 | 32| 215 | 22347 | 61676 | 
| 71 | **13 django/contrib/admin/options.py** | 1461 | 1480| 133 | 22480 | 61676 | 
| 72 | 13 django/contrib/admin/views/main.py | 48 | 121| 653 | 23133 | 61676 | 
| 73 | 13 django/forms/formsets.py | 169 | 195| 240 | 23373 | 61676 | 
| 74 | 13 django/forms/models.py | 686 | 757| 732 | 24105 | 61676 | 
| 75 | 13 django/forms/formsets.py | 53 | 106| 386 | 24491 | 61676 | 
| 76 | 13 django/forms/formsets.py | 1 | 25| 204 | 24695 | 61676 | 
| 77 | 14 django/contrib/gis/admin/options.py | 52 | 63| 139 | 24834 | 62872 | 
| 78 | 14 django/contrib/admin/checks.py | 688 | 722| 265 | 25099 | 62872 | 
| 79 | 15 django/contrib/admin/__init__.py | 1 | 25| 255 | 25354 | 63127 | 
| 80 | 15 django/contrib/gis/admin/options.py | 1 | 50| 394 | 25748 | 63127 | 
| 81 | **15 django/contrib/admin/options.py** | 1667 | 1681| 132 | 25880 | 63127 | 
| 82 | **15 django/contrib/admin/options.py** | 611 | 632| 258 | 26138 | 63127 | 
| 83 | 15 django/views/generic/edit.py | 103 | 126| 194 | 26332 | 63127 | 
| 84 | **15 django/contrib/admin/options.py** | 220 | 242| 230 | 26562 | 63127 | 
| 85 | 15 django/forms/models.py | 1 | 27| 218 | 26780 | 63127 | 
| 86 | 15 django/forms/formsets.py | 245 | 280| 410 | 27190 | 63127 | 
| 87 | 15 django/forms/models.py | 654 | 684| 217 | 27407 | 63127 | 
| 88 | 16 django/contrib/flatpages/admin.py | 1 | 20| 144 | 27551 | 63271 | 
| 89 | **16 django/contrib/admin/options.py** | 377 | 429| 504 | 28055 | 63271 | 
| 90 | 16 django/forms/formsets.py | 330 | 383| 456 | 28511 | 63271 | 
| 91 | 16 django/contrib/admin/checks.py | 1064 | 1106| 343 | 28854 | 63271 | 
| 92 | 17 django/contrib/auth/forms.py | 390 | 441| 356 | 29210 | 66383 | 
| 93 | 17 django/forms/models.py | 1378 | 1405| 209 | 29419 | 66383 | 
| 94 | 17 django/forms/forms.py | 51 | 107| 525 | 29944 | 66383 | 
| 95 | **17 django/contrib/admin/options.py** | 189 | 205| 171 | 30115 | 66383 | 
| 96 | 17 django/forms/models.py | 357 | 387| 233 | 30348 | 66383 | 
| 97 | **17 django/contrib/admin/options.py** | 634 | 651| 128 | 30476 | 66383 | 
| 98 | 17 django/contrib/gis/admin/options.py | 80 | 135| 555 | 31031 | 66383 | 
| 99 | 17 django/forms/models.py | 481 | 561| 718 | 31749 | 66383 | 
| 100 | 17 django/forms/models.py | 451 | 478| 227 | 31976 | 66383 | 
| 101 | **17 django/contrib/admin/options.py** | 1326 | 1351| 232 | 32208 | 66383 | 
| 102 | 18 django/contrib/admindocs/views.py | 186 | 252| 584 | 32792 | 69711 | 
| 103 | 19 django/contrib/admin/widgets.py | 447 | 475| 203 | 32995 | 73554 | 
| 104 | 20 django/db/models/base.py | 324 | 382| 525 | 33520 | 90614 | 
| 105 | 20 django/contrib/admin/views/main.py | 123 | 212| 861 | 34381 | 90614 | 
| 106 | 20 django/contrib/admin/widgets.py | 395 | 418| 241 | 34622 | 90614 | 
| 107 | 20 django/contrib/admin/widgets.py | 311 | 323| 114 | 34736 | 90614 | 
| 108 | 21 django/contrib/admin/filters.py | 1 | 17| 127 | 34863 | 94737 | 
| 109 | 21 django/contrib/admin/checks.py | 388 | 414| 281 | 35144 | 94737 | 
| 110 | 22 django/contrib/admin/sites.py | 37 | 77| 313 | 35457 | 99107 | 
| 111 | 22 django/forms/models.py | 866 | 892| 323 | 35780 | 99107 | 
| 112 | **22 django/contrib/admin/options.py** | 1683 | 1754| 653 | 36433 | 99107 | 
| 113 | **22 django/contrib/admin/options.py** | 897 | 918| 221 | 36654 | 99107 | 
| 114 | **22 django/contrib/admin/options.py** | 1022 | 1035| 169 | 36823 | 99107 | 
| 115 | 22 django/forms/forms.py | 451 | 498| 382 | 37205 | 99107 | 
| 116 | 22 django/contrib/auth/admin.py | 128 | 189| 465 | 37670 | 99107 | 
| 117 | 22 django/contrib/admin/widgets.py | 376 | 393| 129 | 37799 | 99107 | 
| 118 | 22 django/forms/forms.py | 376 | 396| 210 | 38009 | 99107 | 
| 119 | 23 django/db/models/query.py | 616 | 639| 215 | 38224 | 116442 | 
| 120 | 24 django/contrib/gis/admin/__init__.py | 1 | 13| 130 | 38354 | 116572 | 
| 121 | 24 django/contrib/admin/widgets.py | 49 | 70| 168 | 38522 | 116572 | 
| 122 | 24 django/contrib/admindocs/views.py | 253 | 319| 589 | 39111 | 116572 | 
| 123 | 24 django/contrib/auth/forms.py | 135 | 157| 179 | 39290 | 116572 | 
| 124 | 24 django/contrib/admin/checks.py | 1035 | 1062| 194 | 39484 | 116572 | 
| 125 | **24 django/contrib/admin/options.py** | 1037 | 1059| 198 | 39682 | 116572 | 
| 126 | 24 django/forms/formsets.py | 28 | 50| 259 | 39941 | 116572 | 
| 127 | 24 django/views/generic/edit.py | 129 | 149| 182 | 40123 | 116572 | 
| 128 | 24 django/forms/models.py | 69 | 93| 195 | 40318 | 116572 | 
| 129 | 24 django/contrib/admin/sites.py | 416 | 431| 129 | 40447 | 116572 | 
| 130 | **24 django/contrib/admin/options.py** | 1353 | 1418| 581 | 41028 | 116572 | 
| 131 | 24 django/contrib/auth/admin.py | 1 | 22| 188 | 41216 | 116572 | 
| 132 | 24 django/contrib/admin/templatetags/admin_modify.py | 89 | 117| 203 | 41419 | 116572 | 
| 133 | 24 django/contrib/admin/filters.py | 307 | 370| 627 | 42046 | 116572 | 
| 134 | 24 django/contrib/admin/sites.py | 536 | 569| 281 | 42327 | 116572 | 
| 135 | 24 django/forms/models.py | 1098 | 1138| 306 | 42633 | 116572 | 
| 136 | 24 django/contrib/auth/admin.py | 101 | 126| 286 | 42919 | 116572 | 
| 137 | 24 django/contrib/auth/admin.py | 25 | 37| 128 | 43047 | 116572 | 
| 138 | 24 django/contrib/admin/filters.py | 373 | 397| 294 | 43341 | 116572 | 
| 139 | 24 django/contrib/admin/views/main.py | 214 | 263| 420 | 43761 | 116572 | 
| 140 | 24 django/contrib/admin/widgets.py | 161 | 192| 243 | 44004 | 116572 | 
| 141 | 24 django/forms/models.py | 316 | 355| 387 | 44391 | 116572 | 
| 142 | **24 django/contrib/admin/options.py** | 1909 | 1947| 330 | 44721 | 116572 | 
| 143 | 24 django/contrib/admin/filters.py | 266 | 278| 149 | 44870 | 116572 | 
| 144 | 24 django/contrib/admindocs/views.py | 87 | 115| 285 | 45155 | 116572 | 
| 145 | 24 django/contrib/admin/widgets.py | 1 | 46| 330 | 45485 | 116572 | 
| 146 | 24 django/contrib/admin/sites.py | 95 | 141| 443 | 45928 | 116572 | 
| 147 | 25 django/db/models/fields/__init__.py | 589 | 630| 287 | 46215 | 134990 | 
| 148 | 25 django/contrib/admindocs/views.py | 322 | 351| 201 | 46416 | 134990 | 
| 149 | 25 django/forms/formsets.py | 228 | 243| 176 | 46592 | 134990 | 
| 150 | 26 django/db/backends/mysql/schema.py | 90 | 100| 138 | 46730 | 136531 | 
| 151 | 26 django/contrib/admin/widgets.py | 92 | 117| 179 | 46909 | 136531 | 
| 152 | 27 django/contrib/admin/migrations/0001_initial.py | 1 | 48| 320 | 47229 | 136851 | 
| 153 | **27 django/contrib/admin/options.py** | 653 | 667| 136 | 47365 | 136851 | 
| 154 | **27 django/contrib/admin/options.py** | 207 | 218| 135 | 47500 | 136851 | 
| 155 | 27 django/db/models/base.py | 404 | 505| 871 | 48371 | 136851 | 
| 156 | 27 django/contrib/admin/views/main.py | 442 | 494| 440 | 48811 | 136851 | 
| 157 | 28 django/db/backends/base/schema.py | 874 | 909| 267 | 49078 | 149445 | 
| 158 | 29 django/contrib/admin/templatetags/admin_list.py | 182 | 264| 791 | 49869 | 153105 | 
| 159 | 30 django/contrib/admin/decorators.py | 74 | 104| 134 | 50003 | 153748 | 
| 160 | 30 django/contrib/admin/checks.py | 146 | 163| 155 | 50158 | 153748 | 
| 161 | 31 django/contrib/contenttypes/fields.py | 679 | 704| 254 | 50412 | 159199 | 
| 162 | 31 django/contrib/admin/widgets.py | 347 | 373| 328 | 50740 | 159199 | 
| 163 | 31 django/contrib/admin/widgets.py | 420 | 445| 267 | 51007 | 159199 | 
| 164 | 32 django/contrib/admin/views/autocomplete.py | 48 | 102| 403 | 51410 | 159949 | 
| 165 | 33 django/contrib/admin/templatetags/base.py | 1 | 20| 173 | 51583 | 160247 | 
| 166 | 33 django/contrib/admin/templatetags/admin_list.py | 1 | 25| 175 | 51758 | 160247 | 
| 167 | 33 django/forms/forms.py | 131 | 148| 142 | 51900 | 160247 | 
| 168 | 33 django/forms/models.py | 30 | 66| 313 | 52213 | 160247 | 
| 169 | 33 django/contrib/admin/widgets.py | 273 | 308| 382 | 52595 | 160247 | 
| 170 | 33 django/contrib/admindocs/views.py | 159 | 183| 234 | 52829 | 160247 | 
| 171 | 34 django/contrib/admin/templatetags/admin_urls.py | 1 | 57| 405 | 53234 | 160652 | 
| 172 | 34 django/contrib/admin/templatetags/admin_list.py | 267 | 319| 350 | 53584 | 160652 | 
| 173 | **34 django/contrib/admin/options.py** | 920 | 945| 216 | 53800 | 160652 | 
| 174 | 34 django/contrib/admin/sites.py | 241 | 295| 516 | 54316 | 160652 | 
| 175 | 34 django/forms/models.py | 1268 | 1302| 254 | 54570 | 160652 | 
| 176 | 34 django/db/models/fields/__init__.py | 2450 | 2499| 311 | 54881 | 160652 | 
| 177 | 35 django/forms/widgets.py | 303 | 339| 204 | 55085 | 168751 | 
| 178 | 35 django/contrib/admin/filters.py | 118 | 159| 365 | 55450 | 168751 | 
| 179 | 35 django/forms/formsets.py | 123 | 140| 194 | 55644 | 168751 | 
| 180 | 35 django/forms/models.py | 419 | 449| 243 | 55887 | 168751 | 
| 181 | 35 django/forms/forms.py | 166 | 188| 182 | 56069 | 168751 | 
| 182 | 35 django/contrib/admin/sites.py | 321 | 336| 158 | 56227 | 168751 | 
| 183 | 36 django/contrib/admin/utils.py | 53 | 101| 333 | 56560 | 172913 | 
| 184 | 36 django/contrib/admin/utils.py | 262 | 285| 174 | 56734 | 172913 | 
| 185 | 37 django/db/migrations/autodetector.py | 805 | 854| 567 | 57301 | 184532 | 
| 186 | 38 django/db/migrations/questioner.py | 206 | 223| 171 | 57472 | 186589 | 
| 187 | 38 django/forms/models.py | 112 | 198| 747 | 58219 | 186589 | 
| 188 | 39 django/contrib/admin/models.py | 1 | 20| 118 | 58337 | 187712 | 
| 189 | 40 django/contrib/auth/models.py | 232 | 285| 352 | 58689 | 190984 | 
| 190 | 41 django/http/request.py | 323 | 350| 279 | 58968 | 196260 | 


### Hint

```
Currently the PR has merge conflicts
I think we should add a more general customization hook that allows customizing the parameters passed to the ​formset initialization (which includes initial data). That could also allow the use case of #27240 which requires adding form_kwargs': {'request': request} to formset_params.
consider the model and admin as defined below. models.py class Author(models.Model): name = models.CharField(max_length=100) class Book(models.Model): author = models.ForeignKey(Author, on_delete=models.CASCADE) name = models.CharField(max_length=100) admin.py class BookInline(admin.StackedInline): model = Book class AuthorAdmin(admin.ModelAdmin): inlines = [ BookInline, ] admin.site.register(Author, AuthorAdmin) Is it a good idea to pass the initial vales of bookinline fields by using get request in such a way like http://127.0.0.1:8000/admin/polls/author/add/?name=Author_name&book_1_name=book1_name_value&book_2_name=book2_name_value Example: http://127.0.0.1:8000/admin/polls/author/add/?name=william_shakespeare&book_1_name=Hamlet&book_2_name=Romeo_and_Juliet Please update me if the idea seems fine so that I may create a PR in order to solve the issue.
```

## Patch

```diff
diff --git a/django/contrib/admin/options.py b/django/contrib/admin/options.py
--- a/django/contrib/admin/options.py
+++ b/django/contrib/admin/options.py
@@ -1946,6 +1946,20 @@ def history_view(self, request, object_id, extra_context=None):
             "admin/object_history.html"
         ], context)
 
+    def get_formset_kwargs(self, request, obj, inline, prefix):
+        formset_params = {
+            'instance': obj,
+            'prefix': prefix,
+            'queryset': inline.get_queryset(request),
+        }
+        if request.method == 'POST':
+            formset_params.update({
+                'data': request.POST.copy(),
+                'files': request.FILES,
+                'save_as_new': '_saveasnew' in request.POST
+            })
+        return formset_params
+
     def _create_formsets(self, request, obj, change):
         "Helper function to generate formsets for add/change_view."
         formsets = []
@@ -1959,17 +1973,7 @@ def _create_formsets(self, request, obj, change):
             prefixes[prefix] = prefixes.get(prefix, 0) + 1
             if prefixes[prefix] != 1 or not prefix:
                 prefix = "%s-%s" % (prefix, prefixes[prefix])
-            formset_params = {
-                'instance': obj,
-                'prefix': prefix,
-                'queryset': inline.get_queryset(request),
-            }
-            if request.method == 'POST':
-                formset_params.update({
-                    'data': request.POST.copy(),
-                    'files': request.FILES,
-                    'save_as_new': '_saveasnew' in request.POST
-                })
+            formset_params = self.get_formset_kwargs(request, obj, inline, prefix)
             formset = FormSet(**formset_params)
 
             def user_deleted_form(request, obj, formset, index):

```

## Test Patch

```diff
diff --git a/tests/admin_views/admin.py b/tests/admin_views/admin.py
--- a/tests/admin_views/admin.py
+++ b/tests/admin_views/admin.py
@@ -951,6 +951,12 @@ class CityAdmin(admin.ModelAdmin):
     inlines = [RestaurantInlineAdmin]
     view_on_site = True
 
+    def get_formset_kwargs(self, request, obj, inline, prefix):
+        return {
+            **super().get_formset_kwargs(request, obj, inline, prefix),
+            'form_kwargs': {'initial': {'name': 'overridden_name'}},
+        }
+
 
 class WorkerAdmin(admin.ModelAdmin):
     def view_on_site(self, obj):
diff --git a/tests/admin_views/tests.py b/tests/admin_views/tests.py
--- a/tests/admin_views/tests.py
+++ b/tests/admin_views/tests.py
@@ -1117,6 +1117,10 @@ def test_view_subtitle_per_object(self):
         self.assertContains(response, '<h1>View article</h1>')
         self.assertContains(response, '<h2>Article 2</h2>')
 
+    def test_formset_kwargs_can_be_overridden(self):
+        response = self.client.get(reverse('admin:admin_views_city_add'))
+        self.assertContains(response, 'overridden_name')
+
 
 @override_settings(TEMPLATES=[{
     'BACKEND': 'django.template.backends.django.DjangoTemplates',

```


## Code snippets

### 1 - django/contrib/admin/helpers.py:

Start line: 323, End line: 349

```python
class InlineAdminFormSet:

    def inline_formset_data(self):
        verbose_name = self.opts.verbose_name
        return json.dumps({
            'name': '#%s' % self.formset.prefix,
            'options': {
                'prefix': self.formset.prefix,
                'addText': gettext('Add another %(verbose_name)s') % {
                    'verbose_name': capfirst(verbose_name),
                },
                'deleteText': gettext('Remove'),
            }
        })

    @property
    def forms(self):
        return self.formset.forms

    @property
    def non_form_errors(self):
        return self.formset.non_form_errors

    @property
    def media(self):
        media = self.opts.media + self.formset.media
        for fs in self:
            media = media + fs.media
        return media
```
### 2 - django/contrib/admin/helpers.py:

Start line: 264, End line: 287

```python
class InlineAdminFormSet:

    def __iter__(self):
        if self.has_change_permission:
            readonly_fields_for_editing = self.readonly_fields
        else:
            readonly_fields_for_editing = self.readonly_fields + flatten_fieldsets(self.fieldsets)

        for form, original in zip(self.formset.initial_forms, self.formset.get_queryset()):
            view_on_site_url = self.opts.get_view_on_site_url(original)
            yield InlineAdminForm(
                self.formset, form, self.fieldsets, self.prepopulated_fields,
                original, readonly_fields_for_editing, model_admin=self.opts,
                view_on_site_url=view_on_site_url,
            )
        for form in self.formset.extra_forms:
            yield InlineAdminForm(
                self.formset, form, self.fieldsets, self.prepopulated_fields,
                None, self.readonly_fields, model_admin=self.opts,
            )
        if self.has_add_permission:
            yield InlineAdminForm(
                self.formset, self.formset.empty_form,
                self.fieldsets, self.prepopulated_fields, None,
                self.readonly_fields, model_admin=self.opts,
            )
```
### 3 - django/contrib/admin/helpers.py:

Start line: 289, End line: 321

```python
class InlineAdminFormSet:

    def fields(self):
        fk = getattr(self.formset, "fk", None)
        empty_form = self.formset.empty_form
        meta_labels = empty_form._meta.labels or {}
        meta_help_texts = empty_form._meta.help_texts or {}
        for i, field_name in enumerate(flatten_fieldsets(self.fieldsets)):
            if fk and fk.name == field_name:
                continue
            if not self.has_change_permission or field_name in self.readonly_fields:
                yield {
                    'name': field_name,
                    'label': meta_labels.get(field_name) or label_for_field(
                        field_name,
                        self.opts.model,
                        self.opts,
                        form=empty_form,
                    ),
                    'widget': {'is_hidden': False},
                    'required': False,
                    'help_text': meta_help_texts.get(field_name) or help_text_for_field(field_name, self.opts.model),
                }
            else:
                form_field = empty_form.fields[field_name]
                label = form_field.label
                if label is None:
                    label = label_for_field(field_name, self.opts.model, self.opts, form=empty_form)
                yield {
                    'name': field_name,
                    'label': label,
                    'widget': form_field.widget,
                    'required': form_field.required,
                    'help_text': form_field.help_text,
                }
```
### 4 - django/contrib/admin/options.py:

Start line: 1482, End line: 1505

```python
class ModelAdmin(BaseModelAdmin):

    def get_inline_formsets(self, request, formsets, inline_instances, obj=None):
        # Edit permissions on parent model are required for editable inlines.
        can_edit_parent = self.has_change_permission(request, obj) if obj else self.has_add_permission(request)
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
                has_add_permission = has_change_permission = has_delete_permission = False
                formset.extra = formset.max_num = 0
            has_view_permission = inline.has_view_permission(request, obj)
            prepopulated = dict(inline.get_prepopulated_fields(request, obj))
            inline_admin_formset = helpers.InlineAdminFormSet(
                inline, formset, fieldsets, prepopulated, readonly, model_admin=self,
                has_add_permission=has_add_permission, has_change_permission=has_change_permission,
                has_delete_permission=has_delete_permission, has_view_permission=has_view_permission,
            )
            inline_admin_formsets.append(inline_admin_formset)
        return inline_admin_formsets
```
### 5 - django/contrib/admin/helpers.py:

Start line: 240, End line: 262

```python
class InlineAdminFormSet:
    """
    A wrapper around an inline formset for use in the admin system.
    """
    def __init__(self, inline, formset, fieldsets, prepopulated_fields=None,
                 readonly_fields=None, model_admin=None, has_add_permission=True,
                 has_change_permission=True, has_delete_permission=True,
                 has_view_permission=True):
        self.opts = inline
        self.formset = formset
        self.fieldsets = fieldsets
        self.model_admin = model_admin
        if readonly_fields is None:
            readonly_fields = ()
        self.readonly_fields = readonly_fields
        if prepopulated_fields is None:
            prepopulated_fields = {}
        self.prepopulated_fields = prepopulated_fields
        self.classes = ' '.join(inline.classes) if inline.classes else ''
        self.has_add_permission = has_add_permission
        self.has_change_permission = has_change_permission
        self.has_delete_permission = has_delete_permission
        self.has_view_permission = has_view_permission
```
### 6 - django/contrib/admin/options.py:

Start line: 2050, End line: 2083

```python
class InlineModelAdmin(BaseModelAdmin):

    def get_formset(self, request, obj=None, **kwargs):
        """Return a BaseInlineFormSet class for use in admin add/change views."""
        if 'fields' in kwargs:
            fields = kwargs.pop('fields')
        else:
            fields = flatten_fieldsets(self.get_fieldsets(request, obj))
        excluded = self.get_exclude(request, obj)
        exclude = [] if excluded is None else list(excluded)
        exclude.extend(self.get_readonly_fields(request, obj))
        if excluded is None and hasattr(self.form, '_meta') and self.form._meta.exclude:
            # Take the custom ModelForm's Meta.exclude into account only if the
            # InlineModelAdmin doesn't define its own.
            exclude.extend(self.form._meta.exclude)
        # If exclude is an empty list we use None, since that's the actual
        # default.
        exclude = exclude or None
        can_delete = self.can_delete and self.has_delete_permission(request, obj)
        defaults = {
            'form': self.form,
            'formset': self.formset,
            'fk_name': self.fk_name,
            'fields': fields,
            'exclude': exclude,
            'formfield_callback': partial(self.formfield_for_dbfield, request=request),
            'extra': self.get_extra(request, obj, **kwargs),
            'min_num': self.get_min_num(request, obj, **kwargs),
            'max_num': self.get_max_num(request, obj, **kwargs),
            'can_delete': can_delete,
            **kwargs,
        }

        base_model_form = defaults['form']
        can_change = self.has_change_permission(request, obj) if request else True
        can_add = self.has_add_permission(request, obj) if request else True
        # ... other code
```
### 7 - django/contrib/admin/helpers.py:

Start line: 352, End line: 370

```python
class InlineAdminForm(AdminForm):
    """
    A wrapper around an inline form for use in the admin system.
    """
    def __init__(self, formset, form, fieldsets, prepopulated_fields, original,
                 readonly_fields=None, model_admin=None, view_on_site_url=None):
        self.formset = formset
        self.model_admin = model_admin
        self.original = original
        self.show_url = original and view_on_site_url is not None
        self.absolute_url = view_on_site_url
        super().__init__(form, fieldsets, prepopulated_fields, readonly_fields, model_admin)

    def __iter__(self):
        for name, options in self.fieldsets:
            yield InlineFieldset(
                self.formset, self.form, name, self.readonly_fields,
                model_admin=self.model_admin, **options
            )
```
### 8 - django/contrib/admin/options.py:

Start line: 1995, End line: 2048

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
        if self.verbose_name is None:
            self.verbose_name = self.model._meta.verbose_name
        if self.verbose_name_plural is None:
            self.verbose_name_plural = self.model._meta.verbose_name_plural

    @property
    def media(self):
        extra = '' if settings.DEBUG else '.min'
        js = ['vendor/jquery/jquery%s.js' % extra, 'jquery.init.js', 'inlines.js']
        if self.filter_vertical or self.filter_horizontal:
            js.extend(['SelectBox.js', 'SelectFilter2.js'])
        if self.classes and 'collapse' in self.classes:
            js.append('collapse.js')
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
### 9 - django/contrib/admin/options.py:

Start line: 1507, End line: 1538

```python
class ModelAdmin(BaseModelAdmin):

    def get_changeform_initial_data(self, request):
        """
        Get the initial form data from the request's GET params.
        """
        initial = dict(request.GET.items())
        for k in initial:
            try:
                f = self.model._meta.get_field(k)
            except FieldDoesNotExist:
                continue
            # We have to special-case M2Ms as a list of comma-separated PKs.
            if isinstance(f, models.ManyToManyField):
                initial[k] = initial[k].split(",")
        return initial

    def _get_obj_does_not_exist_redirect(self, request, opts, object_id):
        """
        Create a message informing the user that the object doesn't exist
        and return a redirect to the admin index page.
        """
        msg = _('%(name)s with ID “%(key)s” doesn’t exist. Perhaps it was deleted?') % {
            'name': opts.verbose_name,
            'key': unquote(object_id),
        }
        self.message_user(request, msg, messages.WARNING)
        url = reverse('admin:index', current_app=self.admin_site.name)
        return HttpResponseRedirect(url)

    @csrf_protect_m
    def changeform_view(self, request, object_id=None, form_url='', extra_context=None):
        with transaction.atomic(using=router.db_for_write(self.model)):
            return self._changeform_view(request, object_id, form_url, extra_context)
```
### 10 - django/contrib/contenttypes/admin.py:

Start line: 81, End line: 128

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
### 11 - django/contrib/admin/options.py:

Start line: 1540, End line: 1626

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
### 14 - django/contrib/admin/options.py:

Start line: 1127, End line: 1172

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
### 15 - django/contrib/admin/options.py:

Start line: 2085, End line: 2137

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
### 17 - django/contrib/admin/options.py:

Start line: 1949, End line: 1992

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
            formset_params = {
                'instance': obj,
                'prefix': prefix,
                'queryset': inline.get_queryset(request),
            }
            if request.method == 'POST':
                formset_params.update({
                    'data': request.POST.copy(),
                    'files': request.FILES,
                    'save_as_new': '_saveasnew' in request.POST
                })
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
### 19 - django/contrib/admin/options.py:

Start line: 780, End line: 802

```python
class ModelAdmin(BaseModelAdmin):

    def get_changelist_formset(self, request, **kwargs):
        """
        Return a FormSet class for use on the changelist page if list_editable
        is used.
        """
        defaults = {
            'formfield_callback': partial(self.formfield_for_dbfield, request=request),
            **kwargs,
        }
        return modelformset_factory(
            self.model, self.get_changelist_form(request), extra=0,
            fields=self.list_editable, **defaults
        )

    def get_formsets_with_inlines(self, request, obj=None):
        """
        Yield formsets and the corresponding inlines.
        """
        for inline in self.get_inline_instances(request, obj):
            yield inline.get_formset(request, obj), inline

    def get_paginator(self, request, queryset, per_page, orphans=0, allow_empty_first_page=True):
        return self.paginator(queryset, per_page, orphans, allow_empty_first_page)
```
### 20 - django/contrib/admin/options.py:

Start line: 1627, End line: 1652

```python
class ModelAdmin(BaseModelAdmin):

    def _changeform_view(self, request, object_id, form_url, extra_context):
        # ... other code
        context = {
            **self.admin_site.each_context(request),
            'title': title % opts.verbose_name,
            'subtitle': str(obj) if obj else None,
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
### 21 - django/contrib/admin/options.py:

Start line: 100, End line: 130

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
### 23 - django/contrib/admin/options.py:

Start line: 132, End line: 187

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
### 24 - django/contrib/admin/options.py:

Start line: 286, End line: 375

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
        elif hasattr(obj, 'get_absolute_url'):
            # use the ContentType lookup if view_on_site is True
            return reverse('admin:view_on_site', kwargs={
                'content_type_id': get_content_type_for_model(obj).pk,
                'object_id': obj.pk
            })

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
        return [(None, {'fields': self.get_fields(request, obj)})]

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
        return self.sortable_by if self.sortable_by is not None else self.get_list_display(request)
```
### 29 - django/contrib/admin/options.py:

Start line: 2166, End line: 2201

```python
class InlineModelAdmin(BaseModelAdmin):

    def has_add_permission(self, request, obj):
        if self.opts.auto_created:
            # Auto-created intermediate models don't have their own
            # permissions. The user needs to have the change permission for the
            # related model in order to be able to do anything with the
            # intermediate model.
            return self._has_any_perms_for_target_model(request, ['change'])
        return super().has_add_permission(request)

    def has_change_permission(self, request, obj=None):
        if self.opts.auto_created:
            # Same comment as has_add_permission().
            return self._has_any_perms_for_target_model(request, ['change'])
        return super().has_change_permission(request)

    def has_delete_permission(self, request, obj=None):
        if self.opts.auto_created:
            # Same comment as has_add_permission().
            return self._has_any_perms_for_target_model(request, ['change'])
        return super().has_delete_permission(request, obj)

    def has_view_permission(self, request, obj=None):
        if self.opts.auto_created:
            # Same comment as has_add_permission(). The 'change' permission
            # also implies the 'view' permission.
            return self._has_any_perms_for_target_model(request, ['view', 'change'])
        return super().has_view_permission(request)


class StackedInline(InlineModelAdmin):
    template = 'admin/edit_inline/stacked.html'


class TabularInline(InlineModelAdmin):
    template = 'admin/edit_inline/tabular.html'
```
### 30 - django/contrib/admin/options.py:

Start line: 767, End line: 778

```python
class ModelAdmin(BaseModelAdmin):

    def get_changelist_form(self, request, **kwargs):
        """
        Return a Form class for use in the Formset on the changelist page.
        """
        defaults = {
            'formfield_callback': partial(self.formfield_for_dbfield, request=request),
            **kwargs,
        }
        if defaults.get('fields') is None and not modelform_defines_fields(defaults.get('form')):
            defaults['fields'] = forms.ALL_FIELDS

        return modelform_factory(self.model, **defaults)
```
### 33 - django/contrib/admin/options.py:

Start line: 669, End line: 715

```python
class ModelAdmin(BaseModelAdmin):

    def get_form(self, request, obj=None, change=False, **kwargs):
        """
        Return a Form class for use in the admin add view. This is used by
        add_view and change_view.
        """
        if 'fields' in kwargs:
            fields = kwargs.pop('fields')
        else:
            fields = flatten_fieldsets(self.get_fieldsets(request, obj))
        excluded = self.get_exclude(request, obj)
        exclude = [] if excluded is None else list(excluded)
        readonly_fields = self.get_readonly_fields(request, obj)
        exclude.extend(readonly_fields)
        # Exclude all fields if it's a change form and the user doesn't have
        # the change permission.
        if change and hasattr(request, 'user') and not self.has_change_permission(request, obj):
            exclude.extend(fields)
        if excluded is None and hasattr(self.form, '_meta') and self.form._meta.exclude:
            # Take the custom ModelForm's Meta.exclude into account only if the
            # ModelAdmin doesn't define its own.
            exclude.extend(self.form._meta.exclude)
        # if exclude is an empty list we pass None to be consistent with the
        # default on modelform_factory
        exclude = exclude or None

        # Remove declared form fields which are in readonly_fields.
        new_attrs = dict.fromkeys(f for f in readonly_fields if f in self.form.declared_fields)
        form = type(self.form.__name__, (self.form,), new_attrs)

        defaults = {
            'form': form,
            'fields': fields,
            'exclude': exclude,
            'formfield_callback': partial(self.formfield_for_dbfield, request=request),
            **kwargs,
        }

        if defaults['fields'] is None and not modelform_defines_fields(defaults['form']):
            defaults['fields'] = forms.ALL_FIELDS

        try:
            return modelform_factory(self.model, **defaults)
        except FieldError as e:
            raise FieldError(
                '%s. Check fields/fieldsets/exclude attributes of class %s.'
                % (e, self.__class__.__name__)
            )
```
### 34 - django/contrib/admin/options.py:

Start line: 1, End line: 97

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
from django.contrib.admin.decorators import display
from django.contrib.admin.exceptions import DisallowedModelAdminToField
from django.contrib.admin.templatetags.admin_urls import add_preserved_filters
from django.contrib.admin.utils import (
    NestedObjects, construct_change_message, flatten_fieldsets,
    get_deleted_objects, lookup_needs_distinct, model_format_dict,
    model_ngettext, quote, unquote,
)
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
from django.utils.text import (
    capfirst, format_lazy, get_text_list, smart_split, unescape_string_literal,
)
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
### 35 - django/contrib/admin/options.py:

Start line: 2139, End line: 2164

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
            request.user.has_perm('%s.%s' % (opts.app_label, get_permission_codename(perm, opts)))
            for perm in perms
        )
```
### 36 - django/contrib/admin/options.py:

Start line: 1115, End line: 1125

```python
class ModelAdmin(BaseModelAdmin):

    def save_related(self, request, form, formsets, change):
        """
        Given the ``HttpRequest``, the parent ``ModelForm`` instance, the
        list of inline formsets and a boolean value based on whether the
        parent is being added or changed, save the related objects to the
        database. Note that at this point save_form() and save_model() have
        already been called.
        """
        form.save_m2m()
        for formset in formsets:
            self.save_formset(request, form, formset, change=change)
```
### 37 - django/contrib/admin/options.py:

Start line: 596, End line: 609

```python
class ModelAdmin(BaseModelAdmin):

    def get_inline_instances(self, request, obj=None):
        inline_instances = []
        for inline_class in self.get_inlines(request, obj):
            inline = inline_class(self.model, self.admin_site)
            if request:
                if not (inline.has_view_or_change_permission(request, obj) or
                        inline.has_add_permission(request, obj) or
                        inline.has_delete_permission(request, obj)):
                    continue
                if not inline.has_add_permission(request, obj):
                    inline.max_num = 0
            inline_instances.append(inline)

        return inline_instances
```
### 40 - django/contrib/admin/options.py:

Start line: 1654, End line: 1665

```python
class ModelAdmin(BaseModelAdmin):

    def add_view(self, request, form_url='', extra_context=None):
        return self.changeform_view(request, None, form_url, extra_context)

    def change_view(self, request, object_id, form_url='', extra_context=None):
        return self.changeform_view(request, object_id, form_url, extra_context)

    def _get_edited_object_pks(self, request, prefix):
        """Return POST data values of list_editable primary keys."""
        pk_pattern = re.compile(
            r'{}-\d+-{}$'.format(re.escape(prefix), self.model._meta.pk.name)
        )
        return [value for key, value in request.POST.items() if pk_pattern.match(key)]
```
### 43 - django/contrib/admin/options.py:

Start line: 1839, End line: 1907

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
### 44 - django/contrib/admin/options.py:

Start line: 1755, End line: 1837

```python
class ModelAdmin(BaseModelAdmin):

    @csrf_protect_m
    def changelist_view(self, request, extra_context=None):
        # ... other code
        if request.method == 'POST' and cl.list_editable and '_save' in request.POST:
            if not self.has_change_permission(request):
                raise PermissionDenied
            FormSet = self.get_changelist_formset(request)
            modified_objects = self._get_list_editable_queryset(request, FormSet.get_default_prefix())
            formset = cl.formset = FormSet(request.POST, request.FILES, queryset=modified_objects)
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
                        changecount
                    ) % {
                        'count': changecount,
                        'name': model_ngettext(opts, changecount),
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
            action_form.fields['action'].choices = self.get_action_choices(request)
            media += action_form.media
        else:
            action_form = None

        selection_note_all = ngettext(
            '%(total_count)s selected',
            'All %(total_count)s selected',
            cl.result_count
        )

        context = {
            **self.admin_site.each_context(request),
            'module_name': str(opts.verbose_name_plural),
            'selection_note': _('0 of %(cnt)s selected') % {'cnt': len(cl.result_list)},
            'selection_note_all': selection_note_all % {'total_count': cl.result_count},
            'title': cl.title,
            'subtitle': None,
            'is_popup': cl.is_popup,
            'to_field': cl.to_field,
            'cl': cl,
            'media': media,
            'has_add_permission': self.has_add_permission(request),
            'opts': cl.opts,
            'action_form': action_form,
            'actions_on_top': self.actions_on_top,
            'actions_on_bottom': self.actions_on_bottom,
            'actions_selection_counter': self.actions_selection_counter,
            'preserved_filters': self.get_preserved_filters(request),
            **(extra_context or {}),
        }

        request.current_app = self.admin_site.name

        return TemplateResponse(request, self.change_list_template or [
            'admin/%s/%s/change_list.html' % (app_label, opts.model_name),
            'admin/%s/change_list.html' % app_label,
            'admin/change_list.html'
        ], context)
```
### 51 - django/contrib/admin/options.py:

Start line: 551, End line: 594

```python
class ModelAdmin(BaseModelAdmin):
    """Encapsulate all admin options and functionality for a given model."""

    list_display = ('__str__',)
    list_display_links = ()
    list_filter = ()
    list_select_related = False
    list_per_page = 100
    list_max_show_all = 200
    list_editable = ()
    search_fields = ()
    date_hierarchy = None
    save_as = False
    save_as_continue = True
    save_on_top = False
    paginator = Paginator
    preserve_filters = True
    inlines = []

    # Custom templates (designed to be over-ridden in subclasses)
    add_form_template = None
    change_form_template = None
    change_list_template = None
    delete_confirmation_template = None
    delete_selected_confirmation_template = None
    object_history_template = None
    popup_response_template = None

    # Actions
    actions = []
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
        return "%s.%s" % (self.model._meta.app_label, self.__class__.__name__)
```
### 55 - django/contrib/admin/options.py:

Start line: 1086, End line: 1113

```python
class ModelAdmin(BaseModelAdmin):

    def save_form(self, request, form, change):
        """
        Given a ModelForm return an unsaved instance. ``change`` is True if
        the object is being changed, and False if it's being added.
        """
        return form.save(commit=False)

    def save_model(self, request, obj, form, change):
        """
        Given a model instance save it to the database.
        """
        obj.save()

    def delete_model(self, request, obj):
        """
        Given a model instance delete it from the database.
        """
        obj.delete()

    def delete_queryset(self, request, queryset):
        """Given a queryset, delete it from the database."""
        queryset.delete()

    def save_formset(self, request, form, formset, change):
        """
        Given an inline formset save it to the database.
        """
        formset.save()
```
### 66 - django/contrib/admin/options.py:

Start line: 244, End line: 284

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

        if 'widget' not in kwargs:
            autocomplete_fields = self.get_autocomplete_fields(request)
            if db_field.name in autocomplete_fields:
                kwargs['widget'] = AutocompleteSelectMultiple(
                    db_field,
                    self.admin_site,
                    using=db,
                )
            elif db_field.name in self.raw_id_fields:
                kwargs['widget'] = widgets.ManyToManyRawIdWidget(
                    db_field.remote_field,
                    self.admin_site,
                    using=db,
                )
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
### 71 - django/contrib/admin/options.py:

Start line: 1461, End line: 1480

```python
class ModelAdmin(BaseModelAdmin):

    def render_delete_form(self, request, context):
        opts = self.model._meta
        app_label = opts.app_label

        request.current_app = self.admin_site.name
        context.update(
            to_field_var=TO_FIELD_VAR,
            is_popup_var=IS_POPUP_VAR,
            media=self.media,
        )

        return TemplateResponse(
            request,
            self.delete_confirmation_template or [
                "admin/{}/{}/delete_confirmation.html".format(app_label, opts.model_name),
                "admin/{}/delete_confirmation.html".format(app_label),
                "admin/delete_confirmation.html",
            ],
            context,
        )
```
### 81 - django/contrib/admin/options.py:

Start line: 1667, End line: 1681

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
### 82 - django/contrib/admin/options.py:

Start line: 611, End line: 632

```python
class ModelAdmin(BaseModelAdmin):

    def get_urls(self):
        from django.urls import path

        def wrap(view):
            def wrapper(*args, **kwargs):
                return self.admin_site.admin_view(view)(*args, **kwargs)
            wrapper.model_admin = self
            return update_wrapper(wrapper, view)

        info = self.model._meta.app_label, self.model._meta.model_name

        return [
            path('', wrap(self.changelist_view), name='%s_%s_changelist' % info),
            path('add/', wrap(self.add_view), name='%s_%s_add' % info),
            path('<path:object_id>/history/', wrap(self.history_view), name='%s_%s_history' % info),
            path('<path:object_id>/delete/', wrap(self.delete_view), name='%s_%s_delete' % info),
            path('<path:object_id>/change/', wrap(self.change_view), name='%s_%s_change' % info),
            # For backwards compatibility (was the change url before 1.9)
            path('<path:object_id>/', wrap(RedirectView.as_view(
                pattern_name='%s:%s_%s_change' % ((self.admin_site.name,) + info)
            ))),
        ]
```
### 84 - django/contrib/admin/options.py:

Start line: 220, End line: 242

```python
class BaseModelAdmin(metaclass=forms.MediaDefiningClass):

    def formfield_for_foreignkey(self, db_field, request, **kwargs):
        """
        Get a form Field for a ForeignKey.
        """
        db = kwargs.get('using')

        if 'widget' not in kwargs:
            if db_field.name in self.get_autocomplete_fields(request):
                kwargs['widget'] = AutocompleteSelect(db_field, self.admin_site, using=db)
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
### 89 - django/contrib/admin/options.py:

Start line: 377, End line: 429

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
### 95 - django/contrib/admin/options.py:

Start line: 189, End line: 205

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
### 97 - django/contrib/admin/options.py:

Start line: 634, End line: 651

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
            'actions.js',
            'urlify.js',
            'prepopulate.js',
            'vendor/xregexp/xregexp%s.js' % extra,
        ]
        return forms.Media(js=['admin/js/%s' % url for url in js])
```
### 101 - django/contrib/admin/options.py:

Start line: 1326, End line: 1351

```python
class ModelAdmin(BaseModelAdmin):

    def _response_post_save(self, request, obj):
        opts = self.model._meta
        if self.has_view_or_change_permission(request):
            post_url = reverse('admin:%s_%s_changelist' %
                               (opts.app_label, opts.model_name),
                               current_app=self.admin_site.name)
            preserved_filters = self.get_preserved_filters(request)
            post_url = add_preserved_filters({'preserved_filters': preserved_filters, 'opts': opts}, post_url)
        else:
            post_url = reverse('admin:index',
                               current_app=self.admin_site.name)
        return HttpResponseRedirect(post_url)

    def response_post_save_add(self, request, obj):
        """
        Figure out where to redirect after the 'Save' button has been pressed
        when adding a new object.
        """
        return self._response_post_save(request, obj)

    def response_post_save_change(self, request, obj):
        """
        Figure out where to redirect after the 'Save' button has been pressed
        when editing an existing object.
        """
        return self._response_post_save(request, obj)
```
### 112 - django/contrib/admin/options.py:

Start line: 1683, End line: 1754

```python
class ModelAdmin(BaseModelAdmin):

    @csrf_protect_m
    def changelist_view(self, request, extra_context=None):
        """
        The 'change list' admin view for this model.
        """
        from django.contrib.admin.views.main import ERROR_FLAG
        opts = self.model._meta
        app_label = opts.app_label
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
                return SimpleTemplateResponse('admin/invalid_setup.html', {
                    'title': _('Database error'),
                })
            return HttpResponseRedirect(request.path + '?' + ERROR_FLAG + '=1')

        # If the request was POSTed, this might be a bulk action or a bulk
        # edit. Try to look up an action or confirmation first, but if this
        # isn't an action the POST will fall through to the bulk edit check,
        # below.
        action_failed = False
        selected = request.POST.getlist(helpers.ACTION_CHECKBOX_NAME)

        actions = self.get_actions(request)
        # Actions with no confirmation
        if (actions and request.method == 'POST' and
                'index' in request.POST and '_save' not in request.POST):
            if selected:
                response = self.response_action(request, queryset=cl.get_queryset(request))
                if response:
                    return response
                else:
                    action_failed = True
            else:
                msg = _("Items must be selected in order to perform "
                        "actions on them. No items have been changed.")
                self.message_user(request, msg, messages.WARNING)
                action_failed = True

        # Actions with confirmation
        if (actions and request.method == 'POST' and
                helpers.ACTION_CHECKBOX_NAME in request.POST and
                'index' not in request.POST and '_save' not in request.POST):
            if selected:
                response = self.response_action(request, queryset=cl.get_queryset(request))
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
### 113 - django/contrib/admin/options.py:

Start line: 897, End line: 918

```python
class ModelAdmin(BaseModelAdmin):

    def get_actions(self, request):
        """
        Return a dictionary mapping the names of all actions for this
        ModelAdmin to a tuple of (callable, name, description) for each action.
        """
        # If self.actions is set to None that means actions are disabled on
        # this page.
        if self.actions is None or IS_POPUP_VAR in request.GET:
            return {}
        actions = self._filter_actions_by_permissions(request, self._get_base_actions())
        return {name: (func, name, desc) for func, name, desc in actions}

    def get_action_choices(self, request, default_choices=models.BLANK_CHOICE_DASH):
        """
        Return a list of choices for use in a form object.  Each choice is a
        tuple (name, description).
        """
        choices = [] + default_choices
        for func, name, description in self.get_actions(request).values():
            choice = (name, description % model_format_dict(self.opts))
            choices.append(choice)
        return choices
```
### 114 - django/contrib/admin/options.py:

Start line: 1022, End line: 1035

```python
class ModelAdmin(BaseModelAdmin):

    def get_search_results(self, request, queryset, search_term):
        # ... other code

        use_distinct = False
        search_fields = self.get_search_fields(request)
        if search_fields and search_term:
            orm_lookups = [construct_search(str(search_field))
                           for search_field in search_fields]
            for bit in smart_split(search_term):
                if bit.startswith(('"', "'")):
                    bit = unescape_string_literal(bit)
                or_queries = [models.Q(**{orm_lookup: bit})
                              for orm_lookup in orm_lookups]
                queryset = queryset.filter(reduce(operator.or_, or_queries))
            use_distinct |= any(lookup_needs_distinct(self.opts, search_spec) for search_spec in orm_lookups)

        return queryset, use_distinct
```
### 125 - django/contrib/admin/options.py:

Start line: 1037, End line: 1059

```python
class ModelAdmin(BaseModelAdmin):

    def get_preserved_filters(self, request):
        """
        Return the preserved filters querystring.
        """
        match = request.resolver_match
        if self.preserve_filters and match:
            opts = self.model._meta
            current_url = '%s:%s' % (match.app_name, match.url_name)
            changelist_url = 'admin:%s_%s_changelist' % (opts.app_label, opts.model_name)
            if current_url == changelist_url:
                preserved_filters = request.GET.urlencode()
            else:
                preserved_filters = request.GET.get('_changelist_filters')

            if preserved_filters:
                return urlencode({'_changelist_filters': preserved_filters})
        return ''

    def construct_change_message(self, request, form, formsets, add=False):
        """
        Construct a JSON structure describing changes from a changed object.
        """
        return construct_change_message(form, formsets, add)
```
### 130 - django/contrib/admin/options.py:

Start line: 1353, End line: 1418

```python
class ModelAdmin(BaseModelAdmin):

    def response_action(self, request, queryset):
        """
        Handle an admin action. This is called if a request is POSTed to the
        changelist; it returns an HttpResponse if the action was handled, and
        None otherwise.
        """

        # There can be multiple action forms on the page (at the top
        # and bottom of the change list, for example). Get the action
        # whose button was pushed.
        try:
            action_index = int(request.POST.get('index', 0))
        except ValueError:
            action_index = 0

        # Construct the action form.
        data = request.POST.copy()
        data.pop(helpers.ACTION_CHECKBOX_NAME, None)
        data.pop("index", None)

        # Use the action whose button was pushed
        try:
            data.update({'action': data.getlist('action')[action_index]})
        except IndexError:
            # If we didn't get an action from the chosen form that's invalid
            # POST data, so by deleting action it'll fail the validation check
            # below. So no need to do anything here
            pass

        action_form = self.action_form(data, auto_id=None)
        action_form.fields['action'].choices = self.get_action_choices(request)

        # If the form's valid we can handle the action.
        if action_form.is_valid():
            action = action_form.cleaned_data['action']
            select_across = action_form.cleaned_data['select_across']
            func = self.get_actions(request)[action][0]

            # Get the list of selected PKs. If nothing's selected, we can't
            # perform an action on it, so bail. Except we want to perform
            # the action explicitly on all objects.
            selected = request.POST.getlist(helpers.ACTION_CHECKBOX_NAME)
            if not selected and not select_across:
                # Reminder that something needs to be selected or nothing will happen
                msg = _("Items must be selected in order to perform "
                        "actions on them. No items have been changed.")
                self.message_user(request, msg, messages.WARNING)
                return None

            if not select_across:
                # Perform the action only on the selected objects
                queryset = queryset.filter(pk__in=selected)

            response = func(self, request, queryset)

            # Actions may return an HttpResponse-like object, which will be
            # used as the response from the POST. If not, we'll be a good
            # little HTTP citizen and redirect back to the changelist page.
            if isinstance(response, HttpResponseBase):
                return response
            else:
                return HttpResponseRedirect(request.get_full_path())
        else:
            msg = _("No action selected.")
            self.message_user(request, msg, messages.WARNING)
            return None
```
### 142 - django/contrib/admin/options.py:

Start line: 1909, End line: 1947

```python
class ModelAdmin(BaseModelAdmin):

    def history_view(self, request, object_id, extra_context=None):
        "The 'history' admin view for this model."
        from django.contrib.admin.models import LogEntry

        # First check if the user can see this history.
        model = self.model
        obj = self.get_object(request, unquote(object_id))
        if obj is None:
            return self._get_obj_does_not_exist_redirect(request, model._meta, object_id)

        if not self.has_view_or_change_permission(request, obj):
            raise PermissionDenied

        # Then get the history for this object.
        opts = model._meta
        app_label = opts.app_label
        action_list = LogEntry.objects.filter(
            object_id=unquote(object_id),
            content_type=get_content_type_for_model(model)
        ).select_related().order_by('action_time')

        context = {
            **self.admin_site.each_context(request),
            'title': _('Change history: %s') % obj,
            'action_list': action_list,
            'module_name': str(capfirst(opts.verbose_name_plural)),
            'object': obj,
            'opts': opts,
            'preserved_filters': self.get_preserved_filters(request),
            **(extra_context or {}),
        }

        request.current_app = self.admin_site.name

        return TemplateResponse(request, self.object_history_template or [
            "admin/%s/%s/object_history.html" % (app_label, opts.model_name),
            "admin/%s/object_history.html" % app_label,
            "admin/object_history.html"
        ], context)
```
### 153 - django/contrib/admin/options.py:

Start line: 653, End line: 667

```python
class ModelAdmin(BaseModelAdmin):

    def get_model_perms(self, request):
        """
        Return a dict of all perms for this model. This dict has the keys
        ``add``, ``change``, ``delete``, and ``view`` mapping to the True/False
        for each of those actions.
        """
        return {
            'add': self.has_add_permission(request),
            'change': self.has_change_permission(request),
            'delete': self.has_delete_permission(request),
            'view': self.has_view_permission(request),
        }

    def _get_form_for_get_fields(self, request, obj):
        return self.get_form(request, obj, fields=None)
```
### 154 - django/contrib/admin/options.py:

Start line: 207, End line: 218

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
### 173 - django/contrib/admin/options.py:

Start line: 920, End line: 945

```python
class ModelAdmin(BaseModelAdmin):

    def get_action(self, action):
        """
        Return a given action from a parameter, which can either be a callable,
        or the name of a method on the ModelAdmin.  Return is a tuple of
        (callable, name, description).
        """
        # If the action is a callable, just use it.
        if callable(action):
            func = action
            action = action.__name__

        # Next, look for a method. Grab it off self.__class__ to get an unbound
        # method instead of a bound one; this ensures that the calling
        # conventions are the same for functions and methods.
        elif hasattr(self.__class__, action):
            func = getattr(self.__class__, action)

        # Finally, look for a named method on the admin site
        else:
            try:
                func = self.admin_site.get_action(action)
            except KeyError:
                return None

        description = self._get_action_description(func, action)
        return func, action, description
```
