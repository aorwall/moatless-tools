# django__django-13447

| **django/django** | `0456d3e42795481a186db05719300691fe2a1029` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 476 |
| **Any found context length** | 476 |
| **Avg pos** | 1.0 |
| **Min pos** | 1 |
| **Max pos** | 1 |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/contrib/admin/sites.py b/django/contrib/admin/sites.py
--- a/django/contrib/admin/sites.py
+++ b/django/contrib/admin/sites.py
@@ -461,6 +461,7 @@ def _build_app_dict(self, request, label=None):
 
             info = (app_label, model._meta.model_name)
             model_dict = {
+                'model': model,
                 'name': capfirst(model._meta.verbose_name_plural),
                 'object_name': model._meta.object_name,
                 'perms': perms,

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/contrib/admin/sites.py | 464 | 464 | 1 | 1 | 476


## Problem Statement

```
Added model class to app_list context
Description
	 
		(last modified by Raffaele Salmaso)
	 
I need to manipulate the app_list in my custom admin view, and the easiest way to get the result is to have access to the model class (currently the dictionary is a serialized model).
In addition I would make the _build_app_dict method public, as it is used by the two views index and app_index.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 django/contrib/admin/sites.py** | 433 | 499| 476 | 476 | 4383 | 
| 2 | 2 django/contrib/admindocs/views.py | 186 | 252| 584 | 1060 | 7711 | 
| 3 | 2 django/contrib/admindocs/views.py | 159 | 183| 234 | 1294 | 7711 | 
| 4 | 2 django/contrib/admindocs/views.py | 253 | 319| 589 | 1883 | 7711 | 
| 5 | 3 django/contrib/admin/options.py | 1757 | 1839| 750 | 2633 | 26313 | 
| 6 | **3 django/contrib/admin/sites.py** | 536 | 570| 287 | 2920 | 26313 | 
| 7 | **3 django/contrib/admin/sites.py** | 501 | 534| 229 | 3149 | 26313 | 
| 8 | **3 django/contrib/admin/sites.py** | 242 | 296| 516 | 3665 | 26313 | 
| 9 | 3 django/contrib/admin/options.py | 1685 | 1756| 653 | 4318 | 26313 | 
| 10 | 3 django/contrib/admin/options.py | 1542 | 1628| 760 | 5078 | 26313 | 
| 11 | 3 django/contrib/admin/options.py | 1129 | 1174| 482 | 5560 | 26313 | 
| 12 | 3 django/contrib/admin/options.py | 1629 | 1654| 291 | 5851 | 26313 | 
| 13 | 3 django/contrib/admindocs/views.py | 118 | 136| 168 | 6019 | 26313 | 
| 14 | 3 django/contrib/admin/options.py | 611 | 632| 258 | 6277 | 26313 | 
| 15 | 3 django/contrib/admin/options.py | 1912 | 1965| 433 | 6710 | 26313 | 
| 16 | 3 django/contrib/admindocs/views.py | 1 | 30| 225 | 6935 | 26313 | 
| 17 | 3 django/contrib/admin/options.py | 551 | 594| 297 | 7232 | 26313 | 
| 18 | 3 django/contrib/admindocs/views.py | 33 | 53| 159 | 7391 | 26313 | 
| 19 | 3 django/contrib/admindocs/views.py | 139 | 157| 187 | 7578 | 26313 | 
| 20 | 4 django/db/migrations/state.py | 1 | 26| 197 | 7775 | 32055 | 
| 21 | 5 django/apps/registry.py | 213 | 233| 237 | 8012 | 35462 | 
| 22 | 5 django/db/migrations/state.py | 304 | 350| 444 | 8456 | 35462 | 
| 23 | 6 django/contrib/contenttypes/models.py | 1 | 32| 223 | 8679 | 36877 | 
| 24 | 6 django/contrib/admin/options.py | 1 | 97| 761 | 9440 | 36877 | 
| 25 | 7 django/contrib/contenttypes/management/__init__.py | 88 | 136| 319 | 9759 | 37852 | 
| 26 | 7 django/contrib/admin/options.py | 286 | 375| 636 | 10395 | 37852 | 
| 27 | 7 django/contrib/admin/options.py | 2174 | 2209| 315 | 10710 | 37852 | 
| 28 | 8 django/contrib/admin/models.py | 1 | 20| 118 | 10828 | 38975 | 
| 29 | 8 django/contrib/admin/options.py | 717 | 750| 245 | 11073 | 38975 | 
| 30 | **8 django/contrib/admin/sites.py** | 38 | 78| 313 | 11386 | 38975 | 
| 31 | 8 django/contrib/admin/options.py | 1463 | 1482| 133 | 11519 | 38975 | 
| 32 | 8 django/contrib/admin/options.py | 653 | 667| 136 | 11655 | 38975 | 
| 33 | 8 django/contrib/admin/options.py | 669 | 715| 449 | 12104 | 38975 | 
| 34 | 8 django/contrib/admin/options.py | 1656 | 1667| 151 | 12255 | 38975 | 
| 35 | 9 django/db/models/options.py | 64 | 146| 668 | 12923 | 46342 | 
| 36 | 10 django/core/serializers/__init__.py | 160 | 246| 772 | 13695 | 48099 | 
| 37 | 11 django/db/backends/sqlite3/schema.py | 142 | 223| 820 | 14515 | 52273 | 
| 38 | 11 django/contrib/admindocs/views.py | 56 | 84| 285 | 14800 | 52273 | 
| 39 | 12 django/contrib/admin/views/main.py | 445 | 500| 463 | 15263 | 56687 | 
| 40 | 12 django/contrib/admin/views/main.py | 502 | 533| 224 | 15487 | 56687 | 
| 41 | 13 django/db/models/base.py | 2125 | 2176| 351 | 15838 | 73991 | 
| 42 | 14 django/contrib/admin/views/autocomplete.py | 35 | 46| 131 | 15969 | 74759 | 
| 43 | 14 django/contrib/admindocs/views.py | 322 | 351| 201 | 16170 | 74759 | 
| 44 | 14 django/contrib/admin/options.py | 596 | 609| 122 | 16292 | 74759 | 
| 45 | **14 django/contrib/admin/sites.py** | 298 | 320| 187 | 16479 | 74759 | 
| 46 | 15 django/contrib/admindocs/urls.py | 1 | 51| 307 | 16786 | 75066 | 
| 47 | 15 django/contrib/admin/options.py | 1841 | 1910| 590 | 17376 | 75066 | 
| 48 | 15 django/contrib/admin/options.py | 767 | 778| 114 | 17490 | 75066 | 
| 49 | 15 django/contrib/admin/options.py | 1022 | 1037| 181 | 17671 | 75066 | 
| 50 | 15 django/db/migrations/state.py | 620 | 641| 227 | 17898 | 75066 | 
| 51 | 16 django/apps/config.py | 1 | 70| 558 | 18456 | 77609 | 
| 52 | 17 django/contrib/contenttypes/admin.py | 81 | 128| 410 | 18866 | 78590 | 
| 53 | 18 docs/_ext/djangodocs.py | 26 | 71| 398 | 19264 | 81746 | 
| 54 | 18 django/contrib/admin/options.py | 534 | 548| 169 | 19433 | 81746 | 
| 55 | 19 django/contrib/admin/__init__.py | 1 | 25| 255 | 19688 | 82001 | 
| 56 | **19 django/contrib/admin/sites.py** | 221 | 240| 221 | 19909 | 82001 | 
| 57 | 19 django/contrib/admin/views/autocomplete.py | 48 | 103| 421 | 20330 | 82001 | 
| 58 | 19 django/contrib/admindocs/views.py | 87 | 115| 285 | 20615 | 82001 | 
| 59 | 19 django/db/models/base.py | 324 | 382| 525 | 21140 | 82001 | 
| 60 | 19 django/contrib/admin/views/main.py | 49 | 122| 653 | 21793 | 82001 | 
| 61 | 19 django/contrib/admin/options.py | 100 | 130| 223 | 22016 | 82001 | 
| 62 | 19 django/contrib/admin/options.py | 897 | 918| 221 | 22237 | 82001 | 
| 63 | 19 django/contrib/admin/views/main.py | 217 | 266| 420 | 22657 | 82001 | 
| 64 | 20 django/contrib/admin/apps.py | 1 | 28| 164 | 22821 | 82165 | 
| 65 | 21 django/contrib/admin/templatetags/admin_list.py | 267 | 319| 350 | 23171 | 85825 | 
| 66 | 21 django/contrib/admin/options.py | 1328 | 1353| 232 | 23403 | 85825 | 
| 67 | 22 django/contrib/gis/admin/options.py | 80 | 135| 555 | 23958 | 87021 | 
| 68 | 22 django/contrib/admin/options.py | 1484 | 1507| 319 | 24277 | 87021 | 
| 69 | 22 django/contrib/admin/options.py | 1967 | 2000| 345 | 24622 | 87021 | 
| 70 | 23 django/contrib/admin/templatetags/base.py | 1 | 20| 173 | 24795 | 87319 | 
| 71 | **23 django/contrib/admin/sites.py** | 416 | 431| 129 | 24924 | 87319 | 
| 72 | 23 django/contrib/admin/options.py | 947 | 985| 269 | 25193 | 87319 | 
| 73 | 23 django/db/models/base.py | 212 | 322| 866 | 26059 | 87319 | 
| 74 | 23 django/contrib/admin/views/main.py | 124 | 215| 851 | 26910 | 87319 | 
| 75 | 24 django/contrib/gis/admin/__init__.py | 1 | 13| 140 | 27050 | 87459 | 
| 76 | **24 django/contrib/admin/sites.py** | 1 | 35| 225 | 27275 | 87459 | 
| 77 | 25 django/db/models/manager.py | 1 | 165| 1242 | 28517 | 88902 | 
| 78 | 26 django/contrib/admindocs/apps.py | 1 | 8| 0 | 28517 | 88944 | 
| 79 | 26 django/contrib/admin/views/main.py | 1 | 46| 329 | 28846 | 88944 | 
| 80 | 26 django/db/migrations/state.py | 352 | 376| 266 | 29112 | 88944 | 
| 81 | 26 django/contrib/admin/options.py | 377 | 429| 504 | 29616 | 88944 | 
| 82 | 26 django/apps/config.py | 293 | 306| 117 | 29733 | 88944 | 
| 83 | 26 django/contrib/admin/templatetags/admin_list.py | 409 | 467| 343 | 30076 | 88944 | 
| 84 | 26 django/contrib/admin/templatetags/base.py | 22 | 34| 134 | 30210 | 88944 | 
| 85 | 27 django/views/generic/edit.py | 70 | 101| 269 | 30479 | 90660 | 
| 86 | 27 django/contrib/admin/options.py | 634 | 651| 128 | 30607 | 90660 | 
| 87 | 27 docs/_ext/djangodocs.py | 173 | 206| 286 | 30893 | 90660 | 
| 88 | 27 django/db/models/base.py | 404 | 509| 913 | 31806 | 90660 | 
| 89 | **27 django/contrib/admin/sites.py** | 80 | 94| 129 | 31935 | 90660 | 
| 90 | 28 django/db/migrations/autodetector.py | 431 | 460| 265 | 32200 | 102240 | 
| 91 | 28 django/contrib/admin/options.py | 863 | 879| 171 | 32371 | 102240 | 
| 92 | 29 django/contrib/admindocs/utils.py | 1 | 28| 184 | 32555 | 104178 | 
| 93 | 29 django/contrib/admin/options.py | 1669 | 1683| 132 | 32687 | 104178 | 
| 94 | 30 django/db/models/utils.py | 1 | 25| 193 | 32880 | 104532 | 
| 95 | 30 django/views/generic/edit.py | 1 | 67| 479 | 33359 | 104532 | 
| 96 | 31 django/contrib/admin/utils.py | 226 | 242| 132 | 33491 | 108690 | 
| 97 | 31 django/db/models/options.py | 148 | 207| 587 | 34078 | 108690 | 
| 98 | **31 django/contrib/admin/sites.py** | 198 | 220| 167 | 34245 | 108690 | 
| 99 | 32 django/core/management/commands/dumpdata.py | 81 | 153| 624 | 34869 | 110608 | 
| 100 | 32 django/contrib/admin/templatetags/admin_list.py | 1 | 25| 175 | 35044 | 110608 | 
| 101 | 33 django/views/generic/list.py | 113 | 136| 205 | 35249 | 112180 | 
| 102 | 33 django/apps/registry.py | 166 | 184| 133 | 35382 | 112180 | 
| 103 | 34 django/db/utils.py | 255 | 297| 322 | 35704 | 114187 | 
| 104 | 34 django/contrib/gis/admin/options.py | 1 | 50| 394 | 36098 | 114187 | 
| 105 | 34 django/apps/registry.py | 61 | 125| 438 | 36536 | 114187 | 
| 106 | 34 django/apps/registry.py | 263 | 275| 112 | 36648 | 114187 | 
| 107 | 34 django/contrib/admin/options.py | 2003 | 2056| 454 | 37102 | 114187 | 
| 108 | 34 django/db/models/options.py | 390 | 412| 164 | 37266 | 114187 | 
| 109 | 34 django/contrib/admin/templatetags/admin_list.py | 182 | 264| 791 | 38057 | 114187 | 
| 110 | 34 django/views/generic/edit.py | 152 | 199| 340 | 38397 | 114187 | 
| 111 | 34 django/db/migrations/autodetector.py | 1230 | 1242| 131 | 38528 | 114187 | 
| 112 | 35 django/db/models/__init__.py | 1 | 53| 619 | 39147 | 114806 | 
| 113 | 36 django/views/generic/dates.py | 293 | 316| 203 | 39350 | 120247 | 
| 114 | 37 django/contrib/auth/management/__init__.py | 35 | 86| 471 | 39821 | 121357 | 
| 115 | 37 django/views/generic/list.py | 139 | 158| 196 | 40017 | 121357 | 
| 116 | 37 django/db/models/options.py | 1 | 35| 300 | 40317 | 121357 | 
| 117 | 37 django/contrib/admin/options.py | 2058 | 2091| 376 | 40693 | 121357 | 
| 118 | 37 django/db/models/options.py | 712 | 747| 388 | 41081 | 121357 | 
| 119 | 37 django/db/migrations/state.py | 110 | 156| 367 | 41448 | 121357 | 
| 120 | 37 django/apps/config.py | 256 | 291| 251 | 41699 | 121357 | 
| 121 | 38 django/contrib/admin/filters.py | 1 | 17| 127 | 41826 | 125480 | 
| 122 | 38 django/contrib/admin/models.py | 23 | 36| 111 | 41937 | 125480 | 
| 123 | 39 django/core/checks/model_checks.py | 129 | 153| 268 | 42205 | 127265 | 
| 124 | **39 django/contrib/admin/sites.py** | 96 | 142| 443 | 42648 | 127265 | 
| 125 | 40 django/contrib/auth/admin.py | 101 | 126| 286 | 42934 | 128991 | 
| 126 | 40 django/contrib/admin/options.py | 804 | 834| 216 | 43150 | 128991 | 
| 127 | 41 django/contrib/admin/checks.py | 646 | 665| 183 | 43333 | 138173 | 
| 128 | 42 django/contrib/auth/checks.py | 105 | 211| 776 | 44109 | 139671 | 
| 129 | 42 django/contrib/contenttypes/models.py | 133 | 185| 381 | 44490 | 139671 | 
| 130 | 42 django/db/migrations/state.py | 158 | 168| 132 | 44622 | 139671 | 
| 131 | 42 django/contrib/admin/templatetags/admin_list.py | 165 | 179| 136 | 44758 | 139671 | 
| 132 | 42 django/db/migrations/state.py | 289 | 301| 125 | 44883 | 139671 | 
| 133 | 42 django/contrib/admin/options.py | 780 | 802| 207 | 45090 | 139671 | 
| 134 | 43 django/core/management/commands/showmigrations.py | 42 | 63| 158 | 45248 | 140866 | 
| 135 | 43 django/db/models/manager.py | 168 | 204| 201 | 45449 | 140866 | 
| 136 | 43 django/db/models/options.py | 612 | 638| 197 | 45646 | 140866 | 
| 137 | 44 django/db/migrations/operations/models.py | 125 | 248| 853 | 46499 | 147850 | 
| 138 | 45 django/contrib/flatpages/apps.py | 1 | 9| 0 | 46499 | 147901 | 
| 139 | 45 django/contrib/admin/utils.py | 123 | 158| 303 | 46802 | 147901 | 
| 140 | 45 django/contrib/admin/checks.py | 744 | 775| 219 | 47021 | 147901 | 
| 141 | 46 django/contrib/syndication/apps.py | 1 | 8| 0 | 47021 | 147943 | 
| 142 | 46 django/contrib/admin/views/autocomplete.py | 1 | 33| 231 | 47252 | 147943 | 
| 143 | 46 django/db/migrations/operations/models.py | 417 | 438| 170 | 47422 | 147943 | 
| 144 | 46 django/contrib/gis/admin/options.py | 65 | 78| 150 | 47572 | 147943 | 
| 145 | 46 django/contrib/admin/options.py | 515 | 532| 200 | 47772 | 147943 | 
| 146 | 46 django/contrib/admin/models.py | 39 | 72| 241 | 48013 | 147943 | 
| 147 | 47 django/template/context.py | 133 | 167| 288 | 48301 | 149824 | 
| 148 | 47 django/db/models/base.py | 1331 | 1356| 184 | 48485 | 149824 | 
| 149 | 48 django/contrib/admin/helpers.py | 35 | 69| 230 | 48715 | 153155 | 
| 150 | 49 django/contrib/admin/templatetags/admin_urls.py | 1 | 57| 405 | 49120 | 153560 | 
| 151 | 49 django/db/models/base.py | 72 | 168| 827 | 49947 | 153560 | 
| 152 | 49 django/db/migrations/autodetector.py | 533 | 684| 1174 | 51121 | 153560 | 
| 153 | 49 django/db/models/options.py | 252 | 287| 341 | 51462 | 153560 | 
| 154 | 49 django/apps/registry.py | 186 | 211| 190 | 51652 | 153560 | 
| 155 | 49 django/views/generic/dates.py | 375 | 394| 140 | 51792 | 153560 | 
| 156 | 49 django/db/models/options.py | 414 | 440| 175 | 51967 | 153560 | 
| 157 | 49 django/db/migrations/operations/models.py | 742 | 795| 431 | 52398 | 153560 | 
| 158 | 49 django/contrib/admin/checks.py | 794 | 815| 190 | 52588 | 153560 | 
| 159 | 49 django/db/models/base.py | 169 | 211| 413 | 53001 | 153560 | 
| 160 | 49 django/db/migrations/state.py | 407 | 462| 474 | 53475 | 153560 | 
| 161 | 49 django/contrib/admin/checks.py | 904 | 952| 416 | 53891 | 153560 | 
| 162 | 50 django/contrib/syndication/views.py | 165 | 218| 475 | 54366 | 155287 | 
| 163 | 51 django/contrib/gis/db/models/__init__.py | 1 | 19| 204 | 54570 | 155491 | 
| 164 | 52 django/contrib/flatpages/admin.py | 1 | 20| 144 | 54714 | 155635 | 
| 165 | 52 django/contrib/contenttypes/management/__init__.py | 1 | 43| 357 | 55071 | 155635 | 
| 166 | 52 django/db/backends/sqlite3/schema.py | 225 | 307| 731 | 55802 | 155635 | 
| 167 | 52 django/db/models/base.py | 1269 | 1300| 267 | 56069 | 155635 | 
| 168 | 53 django/db/backends/base/schema.py | 357 | 389| 246 | 56315 | 168330 | 
| 169 | 54 django/db/backends/base/introspection.py | 107 | 137| 266 | 56581 | 169842 | 
| 170 | 54 django/db/migrations/autodetector.py | 1203 | 1228| 245 | 56826 | 169842 | 
| 171 | 54 django/db/migrations/operations/models.py | 317 | 342| 290 | 57116 | 169842 | 
| 172 | 54 django/db/migrations/autodetector.py | 719 | 802| 679 | 57795 | 169842 | 
| 173 | 54 django/apps/config.py | 98 | 192| 808 | 58603 | 169842 | 
| 174 | 54 django/contrib/admin/checks.py | 732 | 742| 115 | 58718 | 169842 | 
| 175 | 54 django/db/migrations/operations/models.py | 699 | 739| 250 | 58968 | 169842 | 
| 176 | 54 django/db/migrations/autodetector.py | 804 | 854| 576 | 59544 | 169842 | 
| 177 | 54 django/core/management/commands/showmigrations.py | 105 | 148| 340 | 59884 | 169842 | 
| 178 | 55 django/contrib/sites/models.py | 25 | 46| 192 | 60076 | 170630 | 
| 179 | 55 django/db/migrations/operations/models.py | 395 | 415| 213 | 60289 | 170630 | 
| 180 | 55 django/contrib/admin/options.py | 1355 | 1420| 581 | 60870 | 170630 | 
| 181 | 56 django/contrib/admin/actions.py | 1 | 81| 616 | 61486 | 171246 | 
| 182 | **56 django/contrib/admin/sites.py** | 144 | 196| 349 | 61835 | 171246 | 
| 183 | 57 django/contrib/gis/sitemaps/views.py | 1 | 62| 521 | 62356 | 171767 | 
| 184 | 57 django/core/management/commands/showmigrations.py | 65 | 103| 420 | 62776 | 171767 | 
| 185 | 57 django/contrib/admin/utils.py | 189 | 223| 248 | 63024 | 171767 | 
| 186 | 57 django/core/management/commands/dumpdata.py | 1 | 79| 565 | 63589 | 171767 | 
| 187 | 57 django/db/models/options.py | 209 | 250| 321 | 63910 | 171767 | 
| 188 | 57 django/core/checks/model_checks.py | 1 | 86| 665 | 64575 | 171767 | 
| 189 | 57 django/contrib/admin/utils.py | 263 | 286| 174 | 64749 | 171767 | 
| 190 | 57 django/contrib/admin/utils.py | 161 | 187| 239 | 64988 | 171767 | 
| 191 | 57 django/contrib/admin/checks.py | 892 | 902| 129 | 65117 | 171767 | 
| 192 | 58 django/contrib/admin/migrations/0001_initial.py | 1 | 47| 314 | 65431 | 172081 | 
| 193 | 58 django/apps/config.py | 193 | 254| 580 | 66011 | 172081 | 
| 194 | 58 django/apps/registry.py | 379 | 429| 465 | 66476 | 172081 | 
| 195 | 58 django/db/models/base.py | 984 | 997| 180 | 66656 | 172081 | 
| 196 | 58 django/contrib/admin/helpers.py | 153 | 202| 437 | 67093 | 172081 | 
| 197 | 59 django/contrib/messages/apps.py | 1 | 8| 0 | 67093 | 172118 | 
| 198 | 60 django/contrib/admin/decorators.py | 74 | 104| 134 | 67227 | 172761 | 


## Patch

```diff
diff --git a/django/contrib/admin/sites.py b/django/contrib/admin/sites.py
--- a/django/contrib/admin/sites.py
+++ b/django/contrib/admin/sites.py
@@ -461,6 +461,7 @@ def _build_app_dict(self, request, label=None):
 
             info = (app_label, model._meta.model_name)
             model_dict = {
+                'model': model,
                 'name': capfirst(model._meta.verbose_name_plural),
                 'object_name': model._meta.object_name,
                 'perms': perms,

```

## Test Patch

```diff
diff --git a/tests/admin_views/test_adminsite.py b/tests/admin_views/test_adminsite.py
--- a/tests/admin_views/test_adminsite.py
+++ b/tests/admin_views/test_adminsite.py
@@ -55,7 +55,9 @@ def test_available_apps(self):
         admin_views = apps[0]
         self.assertEqual(admin_views['app_label'], 'admin_views')
         self.assertEqual(len(admin_views['models']), 1)
-        self.assertEqual(admin_views['models'][0]['object_name'], 'Article')
+        article = admin_views['models'][0]
+        self.assertEqual(article['object_name'], 'Article')
+        self.assertEqual(article['model'], Article)
 
         # auth.User
         auth = apps[1]
@@ -63,6 +65,7 @@ def test_available_apps(self):
         self.assertEqual(len(auth['models']), 1)
         user = auth['models'][0]
         self.assertEqual(user['object_name'], 'User')
+        self.assertEqual(user['model'], User)
 
         self.assertEqual(auth['app_url'], '/test_admin/admin/auth/')
         self.assertIs(auth['has_module_perms'], True)

```


## Code snippets

### 1 - django/contrib/admin/sites.py:

Start line: 433, End line: 499

```python
class AdminSite:

    def _build_app_dict(self, request, label=None):
        """
        Build the app dictionary. The optional `label` parameter filters models
        of a specific app.
        """
        app_dict = {}

        if label:
            models = {
                m: m_a for m, m_a in self._registry.items()
                if m._meta.app_label == label
            }
        else:
            models = self._registry

        for model, model_admin in models.items():
            app_label = model._meta.app_label

            has_module_perms = model_admin.has_module_permission(request)
            if not has_module_perms:
                continue

            perms = model_admin.get_model_perms(request)

            # Check whether user has any perm for this module.
            # If so, add the module to the model_list.
            if True not in perms.values():
                continue

            info = (app_label, model._meta.model_name)
            model_dict = {
                'name': capfirst(model._meta.verbose_name_plural),
                'object_name': model._meta.object_name,
                'perms': perms,
                'admin_url': None,
                'add_url': None,
            }
            if perms.get('change') or perms.get('view'):
                model_dict['view_only'] = not perms.get('change')
                try:
                    model_dict['admin_url'] = reverse('admin:%s_%s_changelist' % info, current_app=self.name)
                except NoReverseMatch:
                    pass
            if perms.get('add'):
                try:
                    model_dict['add_url'] = reverse('admin:%s_%s_add' % info, current_app=self.name)
                except NoReverseMatch:
                    pass

            if app_label in app_dict:
                app_dict[app_label]['models'].append(model_dict)
            else:
                app_dict[app_label] = {
                    'name': apps.get_app_config(app_label).verbose_name,
                    'app_label': app_label,
                    'app_url': reverse(
                        'admin:app_list',
                        kwargs={'app_label': app_label},
                        current_app=self.name,
                    ),
                    'has_module_perms': has_module_perms,
                    'models': [model_dict],
                }

        if label:
            return app_dict.get(label)
        return app_dict
```
### 2 - django/contrib/admindocs/views.py:

Start line: 186, End line: 252

```python
class ModelDetailView(BaseAdminDocsView):
    template_name = 'admin_doc/model_detail.html'

    def get_context_data(self, **kwargs):
        model_name = self.kwargs['model_name']
        # Get the model class.
        try:
            app_config = apps.get_app_config(self.kwargs['app_label'])
        except LookupError:
            raise Http404(_("App %(app_label)r not found") % self.kwargs)
        try:
            model = app_config.get_model(model_name)
        except LookupError:
            raise Http404(_("Model %(model_name)r not found in app %(app_label)r") % self.kwargs)

        opts = model._meta

        title, body, metadata = utils.parse_docstring(model.__doc__)
        title = title and utils.parse_rst(title, 'model', _('model:') + model_name)
        body = body and utils.parse_rst(body, 'model', _('model:') + model_name)

        # Gather fields/field descriptions.
        fields = []
        for field in opts.fields:
            # ForeignKey is a special case since the field will actually be a
            # descriptor that returns the other object
            if isinstance(field, models.ForeignKey):
                data_type = field.remote_field.model.__name__
                app_label = field.remote_field.model._meta.app_label
                verbose = utils.parse_rst(
                    (_("the related `%(app_label)s.%(data_type)s` object") % {
                        'app_label': app_label, 'data_type': data_type,
                    }),
                    'model',
                    _('model:') + data_type,
                )
            else:
                data_type = get_readable_field_data_type(field)
                verbose = field.verbose_name
            fields.append({
                'name': field.name,
                'data_type': data_type,
                'verbose': verbose or '',
                'help_text': field.help_text,
            })

        # Gather many-to-many fields.
        for field in opts.many_to_many:
            data_type = field.remote_field.model.__name__
            app_label = field.remote_field.model._meta.app_label
            verbose = _("related `%(app_label)s.%(object_name)s` objects") % {
                'app_label': app_label,
                'object_name': data_type,
            }
            fields.append({
                'name': "%s.all" % field.name,
                "data_type": 'List',
                'verbose': utils.parse_rst(_("all %s") % verbose, 'model', _('model:') + opts.model_name),
            })
            fields.append({
                'name': "%s.count" % field.name,
                'data_type': 'Integer',
                'verbose': utils.parse_rst(_("number of %s") % verbose, 'model', _('model:') + opts.model_name),
            })

        methods = []
        # Gather model methods.
        # ... other code
```
### 3 - django/contrib/admindocs/views.py:

Start line: 159, End line: 183

```python
class ViewDetailView(BaseAdminDocsView):

    def get_context_data(self, **kwargs):
        view = self.kwargs['view']
        view_func = self._get_view_func(view)
        if view_func is None:
            raise Http404
        title, body, metadata = utils.parse_docstring(view_func.__doc__)
        title = title and utils.parse_rst(title, 'view', _('view:') + view)
        body = body and utils.parse_rst(body, 'view', _('view:') + view)
        for key in metadata:
            metadata[key] = utils.parse_rst(metadata[key], 'model', _('view:') + view)
        return super().get_context_data(**{
            **kwargs,
            'name': view,
            'summary': title,
            'body': body,
            'meta': metadata,
        })


class ModelIndexView(BaseAdminDocsView):
    template_name = 'admin_doc/model_index.html'

    def get_context_data(self, **kwargs):
        m_list = [m._meta for m in apps.get_models()]
        return super().get_context_data(**{**kwargs, 'models': m_list})
```
### 4 - django/contrib/admindocs/views.py:

Start line: 253, End line: 319

```python
class ModelDetailView(BaseAdminDocsView):

    def get_context_data(self, **kwargs):
        # ... other code
        for func_name, func in model.__dict__.items():
            if inspect.isfunction(func) or isinstance(func, (cached_property, property)):
                try:
                    for exclude in MODEL_METHODS_EXCLUDE:
                        if func_name.startswith(exclude):
                            raise StopIteration
                except StopIteration:
                    continue
                verbose = func.__doc__
                verbose = verbose and (
                    utils.parse_rst(cleandoc(verbose), 'model', _('model:') + opts.model_name)
                )
                # Show properties, cached_properties, and methods without
                # arguments as fields. Otherwise, show as a 'method with
                # arguments'.
                if isinstance(func, (cached_property, property)):
                    fields.append({
                        'name': func_name,
                        'data_type': get_return_data_type(func_name),
                        'verbose': verbose or ''
                    })
                elif method_has_no_args(func) and not func_accepts_kwargs(func) and not func_accepts_var_args(func):
                    fields.append({
                        'name': func_name,
                        'data_type': get_return_data_type(func_name),
                        'verbose': verbose or '',
                    })
                else:
                    arguments = get_func_full_args(func)
                    # Join arguments with ', ' and in case of default value,
                    # join it with '='. Use repr() so that strings will be
                    # correctly displayed.
                    print_arguments = ', '.join([
                        '='.join([arg_el[0], *map(repr, arg_el[1:])])
                        for arg_el in arguments
                    ])
                    methods.append({
                        'name': func_name,
                        'arguments': print_arguments,
                        'verbose': verbose or '',
                    })

        # Gather related objects
        for rel in opts.related_objects:
            verbose = _("related `%(app_label)s.%(object_name)s` objects") % {
                'app_label': rel.related_model._meta.app_label,
                'object_name': rel.related_model._meta.object_name,
            }
            accessor = rel.get_accessor_name()
            fields.append({
                'name': "%s.all" % accessor,
                'data_type': 'List',
                'verbose': utils.parse_rst(_("all %s") % verbose, 'model', _('model:') + opts.model_name),
            })
            fields.append({
                'name': "%s.count" % accessor,
                'data_type': 'Integer',
                'verbose': utils.parse_rst(_("number of %s") % verbose, 'model', _('model:') + opts.model_name),
            })
        return super().get_context_data(**{
            **kwargs,
            'name': opts.label,
            'summary': title,
            'description': body,
            'fields': fields,
            'methods': methods,
        })
```
### 5 - django/contrib/admin/options.py:

Start line: 1757, End line: 1839

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
### 6 - django/contrib/admin/sites.py:

Start line: 536, End line: 570

```python
class AdminSite:

    def app_index(self, request, app_label, extra_context=None):
        app_dict = self._build_app_dict(request, app_label)
        if not app_dict:
            raise Http404('The requested admin page does not exist.')
        # Sort the models alphabetically within each app.
        app_dict['models'].sort(key=lambda x: x['name'])
        context = {
            **self.each_context(request),
            'title': _('%(app)s administration') % {'app': app_dict['name']},
            'subtitle': None,
            'app_list': [app_dict],
            'app_label': app_label,
            **(extra_context or {}),
        }

        request.current_app = self.name

        return TemplateResponse(request, self.app_index_template or [
            'admin/%s/app_index.html' % app_label,
            'admin/app_index.html'
        ], context)


class DefaultAdminSite(LazyObject):
    def _setup(self):
        AdminSiteClass = import_string(apps.get_app_config('admin').default_site)
        self._wrapped = AdminSiteClass()


# This global object represents the default admin site, for the common case.
# You can provide your own AdminSite using the (Simple)AdminConfig.default_site
# attribute. You can also instantiate AdminSite in your own code to create a
# custom admin site.
site = DefaultAdminSite()
```
### 7 - django/contrib/admin/sites.py:

Start line: 501, End line: 534

```python
class AdminSite:

    def get_app_list(self, request):
        """
        Return a sorted list of all the installed apps that have been
        registered in this site.
        """
        app_dict = self._build_app_dict(request)

        # Sort the apps alphabetically.
        app_list = sorted(app_dict.values(), key=lambda x: x['name'].lower())

        # Sort the models alphabetically within each app.
        for app in app_list:
            app['models'].sort(key=lambda x: x['name'])

        return app_list

    def index(self, request, extra_context=None):
        """
        Display the main admin index page, which lists all of the installed
        apps that have been registered in this site.
        """
        app_list = self.get_app_list(request)

        context = {
            **self.each_context(request),
            'title': self.index_title,
            'subtitle': None,
            'app_list': app_list,
            **(extra_context or {}),
        }

        request.current_app = self.name

        return TemplateResponse(request, self.index_template or 'admin/index.html', context)
```
### 8 - django/contrib/admin/sites.py:

Start line: 242, End line: 296

```python
class AdminSite:

    def get_urls(self):
        # Since this module gets imported in the application's root package,
        # it cannot import models from other applications at the module level,
        # and django.contrib.contenttypes.views imports ContentType.
        from django.contrib.contenttypes import views as contenttype_views
        from django.urls import include, path, re_path

        def wrap(view, cacheable=False):
            def wrapper(*args, **kwargs):
                return self.admin_view(view, cacheable)(*args, **kwargs)
            wrapper.admin_site = self
            return update_wrapper(wrapper, view)

        # Admin-site-wide views.
        urlpatterns = [
            path('', wrap(self.index), name='index'),
            path('login/', self.login, name='login'),
            path('logout/', wrap(self.logout), name='logout'),
            path('password_change/', wrap(self.password_change, cacheable=True), name='password_change'),
            path(
                'password_change/done/',
                wrap(self.password_change_done, cacheable=True),
                name='password_change_done',
            ),
            path('autocomplete/', wrap(self.autocomplete_view), name='autocomplete'),
            path('jsi18n/', wrap(self.i18n_javascript, cacheable=True), name='jsi18n'),
            path(
                'r/<int:content_type_id>/<path:object_id>/',
                wrap(contenttype_views.shortcut),
                name='view_on_site',
            ),
        ]

        # Add in each model's views, and create a list of valid URLS for the
        # app_index
        valid_app_labels = []
        for model, model_admin in self._registry.items():
            urlpatterns += [
                path('%s/%s/' % (model._meta.app_label, model._meta.model_name), include(model_admin.urls)),
            ]
            if model._meta.app_label not in valid_app_labels:
                valid_app_labels.append(model._meta.app_label)

        # If there were ModelAdmins registered, we should have a list of app
        # labels for which we need to allow access to the app_index view,
        if valid_app_labels:
            regex = r'^(?P<app_label>' + '|'.join(valid_app_labels) + ')/$'
            urlpatterns += [
                re_path(regex, wrap(self.app_index), name='app_list'),
            ]

        if self.final_catch_all_view:
            urlpatterns.append(re_path(r'(?P<url>.*)$', wrap(self.catch_all_view)))

        return urlpatterns
```
### 9 - django/contrib/admin/options.py:

Start line: 1685, End line: 1756

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
### 10 - django/contrib/admin/options.py:

Start line: 1542, End line: 1628

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
### 30 - django/contrib/admin/sites.py:

Start line: 38, End line: 78

```python
class AdminSite:
    """
    An AdminSite object encapsulates an instance of the Django admin application, ready
    to be hooked in to your URLconf. Models are registered with the AdminSite using the
    register() method, and the get_urls() method can then be used to access Django view
    functions that present a full admin interface for the collection of registered
    models.
    """

    # Text to put at the end of each page's <title>.
    site_title = gettext_lazy('Django site admin')

    # Text to put in each page's <h1>.
    site_header = gettext_lazy('Django administration')

    # Text to put at the top of the admin index page.
    index_title = gettext_lazy('Site administration')

    # URL for the "View site" link at the top of each admin page.
    site_url = '/'

    enable_nav_sidebar = True

    empty_value_display = '-'

    login_form = None
    index_template = None
    app_index_template = None
    login_template = None
    logout_template = None
    password_change_template = None
    password_change_done_template = None

    final_catch_all_view = True

    def __init__(self, name='admin'):
        self._registry = {}  # model_class class -> admin_class instance
        self.name = name
        self._actions = {'delete_selected': actions.delete_selected}
        self._global_actions = self._actions.copy()
        all_sites.add(self)
```
### 45 - django/contrib/admin/sites.py:

Start line: 298, End line: 320

```python
class AdminSite:

    @property
    def urls(self):
        return self.get_urls(), 'admin', self.name

    def each_context(self, request):
        """
        Return a dictionary of variables to put in the template context for
        *every* page in the admin site.

        For sites running on a subpath, use the SCRIPT_NAME value if site_url
        hasn't been customized.
        """
        script_name = request.META['SCRIPT_NAME']
        site_url = script_name if self.site_url == '/' and script_name else self.site_url
        return {
            'site_title': self.site_title,
            'site_header': self.site_header,
            'site_url': site_url,
            'has_permission': self.has_permission(request),
            'available_apps': self.get_app_list(request),
            'is_popup': False,
            'is_nav_sidebar_enabled': self.enable_nav_sidebar,
        }
```
### 56 - django/contrib/admin/sites.py:

Start line: 221, End line: 240

```python
class AdminSite:

    def admin_view(self, view, cacheable=False):
        def inner(request, *args, **kwargs):
            if not self.has_permission(request):
                if request.path == reverse('admin:logout', current_app=self.name):
                    index_path = reverse('admin:index', current_app=self.name)
                    return HttpResponseRedirect(index_path)
                # Inner import to prevent django.contrib.admin (app) from
                # importing django.contrib.auth.models.User (unrelated model).
                from django.contrib.auth.views import redirect_to_login
                return redirect_to_login(
                    request.get_full_path(),
                    reverse('admin:login', current_app=self.name)
                )
            return view(request, *args, **kwargs)
        if not cacheable:
            inner = never_cache(inner)
        # We add csrf_protect here so this function can be used as a utility
        # function for any view, without having to repeat 'csrf_protect'.
        if not getattr(view, 'csrf_exempt', False):
            inner = csrf_protect(inner)
        return update_wrapper(inner, view)
```
### 71 - django/contrib/admin/sites.py:

Start line: 416, End line: 431

```python
class AdminSite:

    def autocomplete_view(self, request):
        return AutocompleteJsonView.as_view(admin_site=self)(request)

    @no_append_slash
    def catch_all_view(self, request, url):
        if settings.APPEND_SLASH and not url.endswith('/'):
            urlconf = getattr(request, 'urlconf', None)
            path = '%s/' % request.path_info
            try:
                match = resolve(path, urlconf)
            except Resolver404:
                pass
            else:
                if getattr(match.func, 'should_append_slash', True):
                    return HttpResponsePermanentRedirect(path)
        raise Http404
```
### 76 - django/contrib/admin/sites.py:

Start line: 1, End line: 35

```python
import re
from functools import update_wrapper
from weakref import WeakSet

from django.apps import apps
from django.conf import settings
from django.contrib.admin import ModelAdmin, actions
from django.contrib.admin.views.autocomplete import AutocompleteJsonView
from django.contrib.auth import REDIRECT_FIELD_NAME
from django.core.exceptions import ImproperlyConfigured
from django.db.models.base import ModelBase
from django.http import (
    Http404, HttpResponsePermanentRedirect, HttpResponseRedirect,
)
from django.template.response import TemplateResponse
from django.urls import NoReverseMatch, Resolver404, resolve, reverse
from django.utils.decorators import method_decorator
from django.utils.functional import LazyObject
from django.utils.module_loading import import_string
from django.utils.text import capfirst
from django.utils.translation import gettext as _, gettext_lazy
from django.views.decorators.cache import never_cache
from django.views.decorators.common import no_append_slash
from django.views.decorators.csrf import csrf_protect
from django.views.i18n import JavaScriptCatalog

all_sites = WeakSet()


class AlreadyRegistered(Exception):
    pass


class NotRegistered(Exception):
    pass
```
### 89 - django/contrib/admin/sites.py:

Start line: 80, End line: 94

```python
class AdminSite:

    def check(self, app_configs):
        """
        Run the system checks on all ModelAdmins, except if they aren't
        customized at all.
        """
        if app_configs is None:
            app_configs = apps.get_app_configs()
        app_configs = set(app_configs)  # Speed up lookups below

        errors = []
        modeladmins = (o for o in self._registry.values() if o.__class__ is not ModelAdmin)
        for modeladmin in modeladmins:
            if modeladmin.model._meta.app_config in app_configs:
                errors.extend(modeladmin.check())
        return errors
```
### 98 - django/contrib/admin/sites.py:

Start line: 198, End line: 220

```python
class AdminSite:

    def admin_view(self, view, cacheable=False):
        """
        Decorator to create an admin view attached to this ``AdminSite``. This
        wraps the view and provides permission checking by calling
        ``self.has_permission``.

        You'll want to use this from within ``AdminSite.get_urls()``:

            class MyAdminSite(AdminSite):

                def get_urls(self):
                    from django.urls import path

                    urls = super().get_urls()
                    urls += [
                        path('my_view/', self.admin_view(some_view))
                    ]
                    return urls

        By default, admin_views are marked non-cacheable using the
        ``never_cache`` decorator. If the view can be safely cached, set
        cacheable=True.
        """
        # ... other code
```
### 124 - django/contrib/admin/sites.py:

Start line: 96, End line: 142

```python
class AdminSite:

    def register(self, model_or_iterable, admin_class=None, **options):
        """
        Register the given model(s) with the given admin class.

        The model(s) should be Model classes, not instances.

        If an admin class isn't given, use ModelAdmin (the default admin
        options). If keyword arguments are given -- e.g., list_display --
        apply them as options to the admin class.

        If a model is already registered, raise AlreadyRegistered.

        If a model is abstract, raise ImproperlyConfigured.
        """
        admin_class = admin_class or ModelAdmin
        if isinstance(model_or_iterable, ModelBase):
            model_or_iterable = [model_or_iterable]
        for model in model_or_iterable:
            if model._meta.abstract:
                raise ImproperlyConfigured(
                    'The model %s is abstract, so it cannot be registered with admin.' % model.__name__
                )

            if model in self._registry:
                registered_admin = str(self._registry[model])
                msg = 'The model %s is already registered ' % model.__name__
                if registered_admin.endswith('.ModelAdmin'):
                    # Most likely registered without a ModelAdmin subclass.
                    msg += 'in app %r.' % re.sub(r'\.ModelAdmin$', '', registered_admin)
                else:
                    msg += 'with %r.' % registered_admin
                raise AlreadyRegistered(msg)

            # Ignore the registration if the model has been
            # swapped out.
            if not model._meta.swapped:
                # If we got **options then dynamically construct a subclass of
                # admin_class with those **options.
                if options:
                    # For reasons I don't quite understand, without a __module__
                    # the created class appears to "live" in the wrong place,
                    # which causes issues later on.
                    options['__module__'] = __name__
                    admin_class = type("%sAdmin" % model.__name__, (admin_class,), options)

                # Instantiate the admin class to save in the registry
                self._registry[model] = admin_class(model, self)
```
### 182 - django/contrib/admin/sites.py:

Start line: 144, End line: 196

```python
class AdminSite:

    def unregister(self, model_or_iterable):
        """
        Unregister the given model(s).

        If a model isn't already registered, raise NotRegistered.
        """
        if isinstance(model_or_iterable, ModelBase):
            model_or_iterable = [model_or_iterable]
        for model in model_or_iterable:
            if model not in self._registry:
                raise NotRegistered('The model %s is not registered' % model.__name__)
            del self._registry[model]

    def is_registered(self, model):
        """
        Check if a model class is registered with this `AdminSite`.
        """
        return model in self._registry

    def add_action(self, action, name=None):
        """
        Register an action to be available globally.
        """
        name = name or action.__name__
        self._actions[name] = action
        self._global_actions[name] = action

    def disable_action(self, name):
        """
        Disable a globally-registered action. Raise KeyError for invalid names.
        """
        del self._actions[name]

    def get_action(self, name):
        """
        Explicitly get a registered global action whether it's enabled or
        not. Raise KeyError for invalid names.
        """
        return self._global_actions[name]

    @property
    def actions(self):
        """
        Get all the enabled actions as an iterable of (name, func).
        """
        return self._actions.items()

    def has_permission(self, request):
        """
        Return True if the given HttpRequest has permission to view
        *at least one* page in the admin site.
        """
        return request.user.is_active and request.user.is_staff
```
