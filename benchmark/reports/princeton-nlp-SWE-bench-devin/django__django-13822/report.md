# django__django-13822

| **django/django** | `74fd233b1433da8c68de636172ee1c9c6d1c08c9` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 770 |
| **Any found context length** | 770 |
| **Avg pos** | 2.0 |
| **Min pos** | 2 |
| **Max pos** | 2 |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/db/models/fields/related.py b/django/db/models/fields/related.py
--- a/django/db/models/fields/related.py
+++ b/django/db/models/fields/related.py
@@ -1614,7 +1614,11 @@ def contribute_to_class(self, cls, name, **kwargs):
             # related_name with one generated from the m2m field name. Django
             # still uses backwards relations internally and we need to avoid
             # clashes between multiple m2m fields with related_name == '+'.
-            self.remote_field.related_name = "_%s_%s_+" % (cls.__name__.lower(), name)
+            self.remote_field.related_name = '_%s_%s_%s_+' % (
+                cls._meta.app_label,
+                cls.__name__.lower(),
+                name,
+            )
 
         super().contribute_to_class(cls, name, **kwargs)
 

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/db/models/fields/related.py | 1617 | 1617 | 2 | 1 | 770


## Problem Statement

```
fields.E305 is raised on ManyToManyFields with related_name='+' in models in different apps but with the same name.
Description
	 
		(last modified by Aleksey Ruban)
	 
Django raises an error during creation a db migration if two models with the same name refer to the same model in m2m field. related_name='+' or 'foo+' don't impact anything.
In some my project there are 50 apps and almost each one has a model with the same name. So I have to come up with a related name and write it in for each m2m field.
Just try to make a migration for my test project
​https://github.com/rafick1983/django_related_name_bug

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 django/db/models/fields/related.py** | 1639 | 1655| 286 | 286 | 13810 | 
| **-> 2 <-** | **1 django/db/models/fields/related.py** | 1600 | 1637| 484 | 770 | 13810 | 
| 3 | **1 django/db/models/fields/related.py** | 108 | 125| 155 | 925 | 13810 | 
| 4 | **1 django/db/models/fields/related.py** | 1428 | 1469| 418 | 1343 | 13810 | 
| 5 | **1 django/db/models/fields/related.py** | 1235 | 1352| 963 | 2306 | 13810 | 
| 6 | **1 django/db/models/fields/related.py** | 253 | 282| 271 | 2577 | 13810 | 
| 7 | **1 django/db/models/fields/related.py** | 127 | 154| 201 | 2778 | 13810 | 
| 8 | **1 django/db/models/fields/related.py** | 156 | 169| 144 | 2922 | 13810 | 
| 9 | **1 django/db/models/fields/related.py** | 1354 | 1426| 616 | 3538 | 13810 | 
| 10 | **1 django/db/models/fields/related.py** | 171 | 184| 140 | 3678 | 13810 | 
| 11 | **1 django/db/models/fields/related.py** | 935 | 948| 126 | 3804 | 13810 | 
| 12 | **1 django/db/models/fields/related.py** | 284 | 318| 293 | 4097 | 13810 | 
| 13 | **1 django/db/models/fields/related.py** | 509 | 574| 492 | 4589 | 13810 | 
| 14 | 2 django/db/backends/base/schema.py | 31 | 41| 120 | 4709 | 26163 | 
| 15 | **2 django/db/models/fields/related.py** | 487 | 507| 138 | 4847 | 26163 | 
| 16 | **2 django/db/models/fields/related.py** | 750 | 768| 222 | 5069 | 26163 | 
| 17 | **2 django/db/models/fields/related.py** | 1657 | 1691| 266 | 5335 | 26163 | 
| 18 | 3 django/db/models/base.py | 911 | 943| 383 | 5718 | 43021 | 
| 19 | **3 django/db/models/fields/related.py** | 421 | 441| 166 | 5884 | 43021 | 
| 20 | **3 django/db/models/fields/related.py** | 611 | 628| 197 | 6081 | 43021 | 
| 21 | **3 django/db/models/fields/related.py** | 864 | 890| 240 | 6321 | 43021 | 
| 22 | 3 django/db/models/base.py | 1535 | 1567| 231 | 6552 | 43021 | 
| 23 | **3 django/db/models/fields/related.py** | 984 | 995| 128 | 6680 | 43021 | 
| 24 | **3 django/db/models/fields/related.py** | 1077 | 1121| 407 | 7087 | 43021 | 
| 25 | **3 django/db/models/fields/related.py** | 913 | 933| 178 | 7265 | 43021 | 
| 26 | **3 django/db/models/fields/related.py** | 1533 | 1550| 184 | 7449 | 43021 | 
| 27 | **3 django/db/models/fields/related.py** | 1202 | 1233| 180 | 7629 | 43021 | 
| 28 | **3 django/db/models/fields/related.py** | 841 | 862| 169 | 7798 | 43021 | 
| 29 | **3 django/db/models/fields/related.py** | 186 | 252| 674 | 8472 | 43021 | 
| 30 | **3 django/db/models/fields/related.py** | 1 | 34| 246 | 8718 | 43021 | 
| 31 | **3 django/db/models/fields/related.py** | 630 | 650| 168 | 8886 | 43021 | 
| 32 | 3 django/db/models/base.py | 1380 | 1410| 244 | 9130 | 43021 | 
| 33 | 4 django/contrib/contenttypes/fields.py | 432 | 453| 248 | 9378 | 48472 | 
| 34 | **4 django/db/models/fields/related.py** | 1552 | 1568| 184 | 9562 | 48472 | 
| 35 | 5 django/db/models/fields/reverse_related.py | 160 | 178| 172 | 9734 | 50794 | 
| 36 | 6 django/db/models/fields/related_descriptors.py | 946 | 968| 218 | 9952 | 61191 | 
| 37 | 6 django/db/models/fields/related_descriptors.py | 1118 | 1163| 484 | 10436 | 61191 | 
| 38 | **6 django/db/models/fields/related.py** | 1570 | 1598| 275 | 10711 | 61191 | 
| 39 | 6 django/db/models/fields/related_descriptors.py | 1016 | 1043| 334 | 11045 | 61191 | 
| 40 | **6 django/db/models/fields/related.py** | 997 | 1024| 215 | 11260 | 61191 | 
| 41 | 6 django/db/models/fields/related_descriptors.py | 1165 | 1206| 392 | 11652 | 61191 | 
| 42 | **6 django/db/models/fields/related.py** | 1124 | 1200| 524 | 12176 | 61191 | 
| 43 | 6 django/contrib/contenttypes/fields.py | 335 | 356| 173 | 12349 | 61191 | 
| 44 | **6 django/db/models/fields/related.py** | 83 | 106| 162 | 12511 | 61191 | 
| 45 | 6 django/contrib/contenttypes/fields.py | 679 | 704| 254 | 12765 | 61191 | 
| 46 | 7 django/db/models/fields/related_lookups.py | 121 | 157| 244 | 13009 | 62644 | 
| 47 | **7 django/db/models/fields/related.py** | 1471 | 1505| 356 | 13365 | 62644 | 
| 48 | 7 django/db/models/fields/related_descriptors.py | 970 | 987| 190 | 13555 | 62644 | 
| 49 | 7 django/db/models/fields/related_descriptors.py | 907 | 944| 374 | 13929 | 62644 | 
| 50 | 7 django/db/models/fields/related_descriptors.py | 672 | 730| 548 | 14477 | 62644 | 
| 51 | **7 django/db/models/fields/related.py** | 950 | 982| 279 | 14756 | 62644 | 
| 52 | **7 django/db/models/fields/related.py** | 771 | 839| 521 | 15277 | 62644 | 
| 53 | 7 django/db/models/fields/related_descriptors.py | 344 | 363| 156 | 15433 | 62644 | 
| 54 | 8 django/db/migrations/operations/models.py | 343 | 392| 493 | 15926 | 69569 | 
| 55 | 8 django/db/models/fields/related_descriptors.py | 989 | 1015| 248 | 16174 | 69569 | 
| 56 | 8 django/contrib/contenttypes/fields.py | 600 | 631| 278 | 16452 | 69569 | 
| 57 | 8 django/db/models/fields/related_descriptors.py | 309 | 323| 182 | 16634 | 69569 | 
| 58 | 9 django/core/serializers/xml_serializer.py | 116 | 156| 360 | 16994 | 73081 | 
| 59 | 9 django/db/models/fields/related_lookups.py | 104 | 119| 215 | 17209 | 73081 | 
| 60 | 9 django/db/models/fields/related_descriptors.py | 732 | 758| 222 | 17431 | 73081 | 
| 61 | 9 django/db/models/fields/related_descriptors.py | 643 | 671| 247 | 17678 | 73081 | 
| 62 | 9 django/db/models/base.py | 1841 | 1914| 572 | 18250 | 73081 | 
| 63 | 9 django/contrib/contenttypes/fields.py | 632 | 656| 214 | 18464 | 73081 | 
| 64 | **9 django/db/models/fields/related.py** | 1027 | 1074| 368 | 18832 | 73081 | 
| 65 | 9 django/db/models/fields/related_descriptors.py | 1076 | 1087| 138 | 18970 | 73081 | 
| 66 | 10 django/db/migrations/state.py | 165 | 189| 213 | 19183 | 78184 | 
| 67 | 10 django/db/models/fields/related_descriptors.py | 120 | 154| 405 | 19588 | 78184 | 
| 68 | 10 django/contrib/contenttypes/fields.py | 658 | 678| 188 | 19776 | 78184 | 
| 69 | 11 django/db/migrations/autodetector.py | 87 | 99| 116 | 19892 | 89803 | 
| 70 | 11 django/db/models/fields/related_lookups.py | 1 | 23| 170 | 20062 | 89803 | 
| 71 | 11 django/db/models/fields/related_lookups.py | 62 | 101| 451 | 20513 | 89803 | 
| 72 | 11 django/db/models/base.py | 1689 | 1737| 348 | 20861 | 89803 | 
| 73 | 11 django/db/migrations/state.py | 26 | 53| 233 | 21094 | 89803 | 
| 74 | 12 django/db/migrations/exceptions.py | 1 | 55| 249 | 21343 | 90053 | 
| 75 | 12 django/db/migrations/state.py | 1 | 23| 180 | 21523 | 90053 | 
| 76 | 12 django/db/models/fields/reverse_related.py | 180 | 205| 269 | 21792 | 90053 | 
| 77 | 12 django/db/migrations/operations/models.py | 1 | 38| 235 | 22027 | 90053 | 
| 78 | 12 django/db/models/base.py | 1486 | 1509| 176 | 22203 | 90053 | 
| 79 | 12 django/db/models/fields/related_descriptors.py | 807 | 866| 576 | 22779 | 90053 | 
| 80 | **12 django/db/models/fields/related.py** | 1507 | 1531| 295 | 23074 | 90053 | 
| 81 | **12 django/db/models/fields/related.py** | 576 | 609| 334 | 23408 | 90053 | 
| 82 | 13 django/db/backends/sqlite3/schema.py | 386 | 435| 464 | 23872 | 94209 | 
| 83 | 13 django/db/models/fields/related_lookups.py | 46 | 60| 224 | 24096 | 94209 | 
| 84 | 13 django/db/models/fields/related_descriptors.py | 609 | 641| 323 | 24419 | 94209 | 
| 85 | 14 django/apps/registry.py | 213 | 233| 237 | 24656 | 97616 | 
| 86 | 14 django/db/models/fields/related_descriptors.py | 1045 | 1074| 277 | 24933 | 97616 | 
| 87 | 14 django/contrib/contenttypes/fields.py | 20 | 107| 557 | 25490 | 97616 | 
| 88 | 14 django/db/migrations/operations/models.py | 106 | 122| 156 | 25646 | 97616 | 
| 89 | **14 django/db/models/fields/related.py** | 652 | 668| 163 | 25809 | 97616 | 
| 90 | 14 django/db/models/base.py | 1511 | 1533| 171 | 25980 | 97616 | 
| 91 | 15 django/db/migrations/questioner.py | 56 | 81| 220 | 26200 | 99689 | 
| 92 | 15 django/db/backends/base/schema.py | 1126 | 1148| 199 | 26399 | 99689 | 
| 93 | 15 django/db/migrations/state.py | 56 | 75| 209 | 26608 | 99689 | 
| 94 | 15 django/db/models/fields/related_descriptors.py | 1 | 79| 683 | 27291 | 99689 | 
| 95 | 16 django/db/utils.py | 256 | 298| 322 | 27613 | 101713 | 
| 96 | **16 django/db/models/fields/related.py** | 670 | 694| 218 | 27831 | 101713 | 
| 97 | 16 django/db/models/fields/reverse_related.py | 208 | 255| 372 | 28203 | 101713 | 
| 98 | 16 django/db/migrations/state.py | 153 | 163| 132 | 28335 | 101713 | 
| 99 | 16 django/db/migrations/state.py | 105 | 151| 367 | 28702 | 101713 | 
| 100 | **16 django/db/models/fields/related.py** | 892 | 911| 145 | 28847 | 101713 | 
| 101 | 16 django/db/models/fields/reverse_related.py | 258 | 277| 147 | 28994 | 101713 | 
| 102 | 17 django/db/migrations/operations/fields.py | 346 | 381| 335 | 29329 | 104811 | 
| 103 | 18 django/core/serializers/python.py | 62 | 77| 156 | 29485 | 106073 | 
| 104 | 18 django/db/backends/base/schema.py | 920 | 939| 296 | 29781 | 106073 | 
| 105 | **18 django/db/models/fields/related.py** | 710 | 748| 335 | 30116 | 106073 | 
| 106 | **18 django/db/models/fields/related.py** | 37 | 59| 201 | 30317 | 106073 | 
| 107 | 18 django/db/models/base.py | 1174 | 1202| 213 | 30530 | 106073 | 
| 108 | 18 django/db/models/fields/related_descriptors.py | 868 | 882| 190 | 30720 | 106073 | 
| 109 | 18 django/db/models/base.py | 995 | 1023| 230 | 30950 | 106073 | 
| 110 | 18 django/db/models/fields/reverse_related.py | 280 | 331| 352 | 31302 | 106073 | 
| 111 | 19 django/contrib/contenttypes/migrations/0002_remove_content_type_name.py | 1 | 40| 217 | 31519 | 106290 | 
| 112 | 19 django/db/models/fields/reverse_related.py | 20 | 139| 749 | 32268 | 106290 | 
| 113 | 19 django/db/migrations/state.py | 589 | 605| 146 | 32414 | 106290 | 
| 114 | 19 django/db/migrations/operations/models.py | 394 | 414| 213 | 32627 | 106290 | 
| 115 | 19 django/db/models/fields/related_descriptors.py | 884 | 905| 199 | 32826 | 106290 | 
| 116 | 19 django/db/models/base.py | 1429 | 1484| 491 | 33317 | 106290 | 
| 117 | 19 django/db/migrations/autodetector.py | 526 | 681| 1240 | 34557 | 106290 | 
| 118 | 19 django/db/models/fields/related_descriptors.py | 1089 | 1116| 338 | 34895 | 106290 | 
| 119 | 19 django/db/migrations/state.py | 291 | 315| 266 | 35161 | 106290 | 
| 120 | **19 django/db/models/fields/related.py** | 444 | 485| 273 | 35434 | 106290 | 
| 121 | 20 django/db/models/options.py | 540 | 555| 146 | 35580 | 113657 | 
| 122 | 21 django/db/models/sql/subqueries.py | 111 | 134| 192 | 35772 | 114858 | 
| 123 | 21 django/contrib/contenttypes/fields.py | 566 | 598| 330 | 36102 | 114858 | 
| 124 | 22 django/db/models/fields/__init__.py | 2450 | 2499| 311 | 36413 | 133290 | 
| 125 | 22 django/db/models/base.py | 1 | 50| 328 | 36741 | 133290 | 
| 126 | 22 django/db/models/fields/reverse_related.py | 1 | 17| 120 | 36861 | 133290 | 
| 127 | 22 django/contrib/contenttypes/fields.py | 218 | 269| 459 | 37320 | 133290 | 
| 128 | 22 django/db/migrations/operations/fields.py | 97 | 109| 130 | 37450 | 133290 | 
| 129 | 22 django/db/models/fields/related_descriptors.py | 551 | 607| 487 | 37937 | 133290 | 
| 130 | 22 django/db/migrations/operations/models.py | 462 | 491| 302 | 38239 | 133290 | 
| 131 | 22 django/db/migrations/operations/fields.py | 111 | 121| 127 | 38366 | 133290 | 
| 132 | 23 django/contrib/contenttypes/checks.py | 24 | 42| 125 | 38491 | 133539 | 
| 133 | **23 django/db/models/fields/related.py** | 62 | 80| 223 | 38714 | 133539 | 
| 134 | 23 django/db/models/fields/related_descriptors.py | 365 | 381| 184 | 38898 | 133539 | 
| 135 | 24 django/forms/models.py | 421 | 451| 243 | 39141 | 145385 | 
| 136 | 24 django/db/migrations/operations/fields.py | 85 | 95| 124 | 39265 | 145385 | 
| 137 | 25 django/core/checks/model_checks.py | 178 | 211| 332 | 39597 | 147170 | 
| 138 | 26 django/contrib/contenttypes/migrations/0001_initial.py | 1 | 35| 207 | 39804 | 147377 | 
| 139 | 26 django/db/backends/base/schema.py | 375 | 389| 182 | 39986 | 147377 | 
| 140 | 27 django/db/migrations/operations/__init__.py | 1 | 18| 195 | 40181 | 147572 | 
| 141 | 27 django/db/backends/base/schema.py | 1109 | 1124| 170 | 40351 | 147572 | 
| 142 | 28 django/contrib/sites/migrations/0001_initial.py | 1 | 32| 191 | 40542 | 147763 | 
| 143 | 28 django/db/migrations/operations/models.py | 316 | 341| 290 | 40832 | 147763 | 
| 144 | 28 django/db/migrations/operations/fields.py | 301 | 344| 410 | 41242 | 147763 | 
| 145 | 28 django/db/models/base.py | 2066 | 2117| 351 | 41593 | 147763 | 
| 146 | 28 django/db/models/base.py | 1265 | 1296| 267 | 41860 | 147763 | 
| 147 | 28 django/db/migrations/operations/models.py | 124 | 247| 853 | 42713 | 147763 | 
| 148 | 28 django/db/models/fields/__init__.py | 337 | 364| 203 | 42916 | 147763 | 
| 149 | 29 django/contrib/contenttypes/management/__init__.py | 1 | 43| 357 | 43273 | 148738 | 
| 150 | 30 django/contrib/flatpages/migrations/0001_initial.py | 1 | 40| 307 | 43580 | 149045 | 
| 151 | 30 django/db/migrations/autodetector.py | 1095 | 1130| 312 | 43892 | 149045 | 
| 152 | 30 django/db/migrations/operations/fields.py | 1 | 37| 241 | 44133 | 149045 | 
| 153 | 30 django/db/models/base.py | 1932 | 2063| 976 | 45109 | 149045 | 
| 154 | 30 django/db/migrations/operations/fields.py | 39 | 61| 183 | 45292 | 149045 | 
| 155 | 31 django/db/migrations/operations/utils.py | 1 | 33| 231 | 45523 | 149800 | 
| 156 | 31 django/core/checks/model_checks.py | 155 | 176| 263 | 45786 | 149800 | 
| 157 | 31 django/db/migrations/autodetector.py | 1132 | 1153| 231 | 46017 | 149800 | 
| 158 | 32 django/contrib/admin/views/main.py | 496 | 527| 224 | 46241 | 154196 | 
| 159 | 32 django/contrib/contenttypes/fields.py | 109 | 157| 328 | 46569 | 154196 | 
| 160 | 32 django/db/models/base.py | 1083 | 1126| 404 | 46973 | 154196 | 
| 161 | **32 django/db/models/fields/related.py** | 320 | 341| 225 | 47198 | 154196 | 
| 162 | 32 django/db/models/fields/__init__.py | 2114 | 2147| 228 | 47426 | 154196 | 
| 163 | 32 django/db/migrations/operations/models.py | 619 | 636| 163 | 47589 | 154196 | 
| 164 | 32 django/db/backends/sqlite3/schema.py | 309 | 330| 218 | 47807 | 154196 | 
| 165 | 32 django/db/models/options.py | 1 | 35| 300 | 48107 | 154196 | 
| 166 | 32 django/db/migrations/operations/fields.py | 236 | 246| 146 | 48253 | 154196 | 
| 167 | 33 django/core/management/commands/migrate.py | 253 | 270| 208 | 48461 | 157452 | 
| 168 | 33 django/db/migrations/autodetector.py | 716 | 803| 789 | 49250 | 157452 | 
| 169 | 33 django/db/migrations/operations/fields.py | 123 | 143| 129 | 49379 | 157452 | 
| 170 | 34 django/core/serializers/base.py | 273 | 298| 218 | 49597 | 159877 | 
| 171 | 34 django/db/migrations/autodetector.py | 463 | 507| 424 | 50021 | 159877 | 
| 172 | 35 django/contrib/admin/checks.py | 177 | 218| 325 | 50346 | 169014 | 
| 173 | 35 django/contrib/contenttypes/checks.py | 1 | 21| 122 | 50468 | 169014 | 
| 174 | 36 django/db/models/constraints.py | 163 | 170| 124 | 50592 | 170629 | 
| 175 | 36 django/db/models/options.py | 289 | 321| 331 | 50923 | 170629 | 
| 176 | 36 django/db/models/fields/related_descriptors.py | 82 | 118| 264 | 51187 | 170629 | 
| 177 | 37 django/contrib/redirects/migrations/0001_initial.py | 1 | 41| 274 | 51461 | 170903 | 
| 178 | 37 django/db/migrations/questioner.py | 143 | 160| 183 | 51644 | 170903 | 
| 179 | 37 django/forms/models.py | 391 | 419| 240 | 51884 | 170903 | 
| 180 | 37 django/db/migrations/operations/models.py | 523 | 532| 129 | 52013 | 170903 | 
| 181 | 37 django/db/migrations/operations/fields.py | 216 | 234| 185 | 52198 | 170903 | 
| 182 | 37 django/core/checks/model_checks.py | 129 | 153| 268 | 52466 | 170903 | 
| 183 | 38 django/db/backends/mysql/schema.py | 89 | 99| 138 | 52604 | 172425 | 
| 184 | 38 django/db/models/fields/__init__.py | 2530 | 2555| 143 | 52747 | 172425 | 
| 185 | 38 django/db/migrations/operations/fields.py | 248 | 270| 188 | 52935 | 172425 | 
| 186 | 38 django/db/migrations/operations/fields.py | 383 | 400| 135 | 53070 | 172425 | 
| 187 | 38 django/db/migrations/autodetector.py | 1191 | 1216| 245 | 53315 | 172425 | 
| 188 | 38 django/db/models/base.py | 1596 | 1621| 183 | 53498 | 172425 | 
| 189 | 38 django/db/models/fields/__init__.py | 715 | 772| 425 | 53923 | 172425 | 
| 190 | 39 django/db/models/fields/files.py | 216 | 337| 960 | 54883 | 176235 | 
| 191 | 40 django/contrib/redirects/migrations/0002_alter_redirect_new_path_help_text.py | 1 | 25| 117 | 55000 | 176352 | 
| 192 | **40 django/db/models/fields/related.py** | 696 | 708| 116 | 55116 | 176352 | 
| 193 | 41 django/db/models/__init__.py | 1 | 53| 619 | 55735 | 176971 | 
| 194 | 41 django/db/migrations/autodetector.py | 435 | 461| 256 | 55991 | 176971 | 
| 195 | 41 django/db/migrations/autodetector.py | 1076 | 1093| 180 | 56171 | 176971 | 
| 196 | 41 django/db/migrations/operations/fields.py | 64 | 83| 128 | 56299 | 176971 | 
| 198 | 43 django/contrib/auth/migrations/0010_alter_group_name_max_length.py | 1 | 17| 0 | 57100 | 199594 | 


### Hint

```
Thanks for the report. It looks that app_label is missing in ​ManyToManyField.contribute_to_class(): self.remote_field.related_name = "_%s_%s_%s_+" % (cls._meta.app_label, cls.__name__.lower(), name)
```

## Patch

```diff
diff --git a/django/db/models/fields/related.py b/django/db/models/fields/related.py
--- a/django/db/models/fields/related.py
+++ b/django/db/models/fields/related.py
@@ -1614,7 +1614,11 @@ def contribute_to_class(self, cls, name, **kwargs):
             # related_name with one generated from the m2m field name. Django
             # still uses backwards relations internally and we need to avoid
             # clashes between multiple m2m fields with related_name == '+'.
-            self.remote_field.related_name = "_%s_%s_+" % (cls.__name__.lower(), name)
+            self.remote_field.related_name = '_%s_%s_%s_+' % (
+                cls._meta.app_label,
+                cls.__name__.lower(),
+                name,
+            )
 
         super().contribute_to_class(cls, name, **kwargs)
 

```

## Test Patch

```diff
diff --git a/tests/invalid_models_tests/test_relative_fields.py b/tests/invalid_models_tests/test_relative_fields.py
--- a/tests/invalid_models_tests/test_relative_fields.py
+++ b/tests/invalid_models_tests/test_relative_fields.py
@@ -3,7 +3,7 @@
 from django.core.checks import Error, Warning as DjangoWarning
 from django.db import connection, models
 from django.test.testcases import SimpleTestCase
-from django.test.utils import isolate_apps, override_settings
+from django.test.utils import isolate_apps, modify_settings, override_settings
 
 
 @isolate_apps('invalid_models_tests')
@@ -1025,6 +1025,32 @@ class Model(models.Model):
             ),
         ])
 
+    @modify_settings(INSTALLED_APPS={'append': 'basic'})
+    @isolate_apps('basic', 'invalid_models_tests')
+    def test_no_clash_across_apps_without_accessor(self):
+        class Target(models.Model):
+            class Meta:
+                app_label = 'invalid_models_tests'
+
+        class Model(models.Model):
+            m2m = models.ManyToManyField(Target, related_name='+')
+
+            class Meta:
+                app_label = 'basic'
+
+        def _test():
+            # Define model with the same name.
+            class Model(models.Model):
+                m2m = models.ManyToManyField(Target, related_name='+')
+
+                class Meta:
+                    app_label = 'invalid_models_tests'
+
+            self.assertEqual(Model.check(), [])
+
+        _test()
+        self.assertEqual(Model.check(), [])
+
 
 @isolate_apps('invalid_models_tests')
 class ExplicitRelatedNameClashTests(SimpleTestCase):
diff --git a/tests/model_meta/results.py b/tests/model_meta/results.py
--- a/tests/model_meta/results.py
+++ b/tests/model_meta/results.py
@@ -321,7 +321,7 @@
     'get_all_related_objects_with_model_hidden_local': {
         Person: (
             ('+', None),
-            ('_relating_people_hidden_+', None),
+            ('_model_meta_relating_people_hidden_+', None),
             ('Person_following_inherited+', None),
             ('Person_following_inherited+', None),
             ('Person_friends_inherited+', None),
@@ -339,7 +339,7 @@
         ),
         ProxyPerson: (
             ('+', Person),
-            ('_relating_people_hidden_+', Person),
+            ('_model_meta_relating_people_hidden_+', Person),
             ('Person_following_inherited+', Person),
             ('Person_following_inherited+', Person),
             ('Person_friends_inherited+', Person),
@@ -357,7 +357,7 @@
         ),
         BasePerson: (
             ('+', None),
-            ('_relating_basepeople_hidden_+', None),
+            ('_model_meta_relating_basepeople_hidden_+', None),
             ('BasePerson_following_abstract+', None),
             ('BasePerson_following_abstract+', None),
             ('BasePerson_following_base+', None),
@@ -408,8 +408,8 @@
         Person: (
             ('+', BasePerson),
             ('+', None),
-            ('_relating_basepeople_hidden_+', BasePerson),
-            ('_relating_people_hidden_+', None),
+            ('_model_meta_relating_basepeople_hidden_+', BasePerson),
+            ('_model_meta_relating_people_hidden_+', None),
             ('BasePerson_following_abstract+', BasePerson),
             ('BasePerson_following_abstract+', BasePerson),
             ('BasePerson_following_base+', BasePerson),
@@ -446,8 +446,8 @@
         ProxyPerson: (
             ('+', BasePerson),
             ('+', Person),
-            ('_relating_basepeople_hidden_+', BasePerson),
-            ('_relating_people_hidden_+', Person),
+            ('_model_meta_relating_basepeople_hidden_+', BasePerson),
+            ('_model_meta_relating_people_hidden_+', Person),
             ('BasePerson_following_abstract+', BasePerson),
             ('BasePerson_following_abstract+', BasePerson),
             ('BasePerson_following_base+', BasePerson),
@@ -483,7 +483,7 @@
         ),
         BasePerson: (
             ('+', None),
-            ('_relating_basepeople_hidden_+', None),
+            ('_model_meta_relating_basepeople_hidden_+', None),
             ('BasePerson_following_abstract+', None),
             ('BasePerson_following_abstract+', None),
             ('BasePerson_following_base+', None),
@@ -822,7 +822,7 @@
             ('friends_base_rel_+', None),
             ('followers_base', None),
             ('relating_basepeople', None),
-            ('_relating_basepeople_hidden_+', None),
+            ('_model_meta_relating_basepeople_hidden_+', None),
         ),
         Person: (
             ('friends_abstract_rel_+', BasePerson),
@@ -830,7 +830,7 @@
             ('friends_base_rel_+', BasePerson),
             ('followers_base', BasePerson),
             ('relating_basepeople', BasePerson),
-            ('_relating_basepeople_hidden_+', BasePerson),
+            ('_model_meta_relating_basepeople_hidden_+', BasePerson),
             ('friends_inherited_rel_+', None),
             ('followers_concrete', None),
             ('relating_people', None),
@@ -849,7 +849,7 @@
             'friends_base_rel_+',
             'followers_base',
             'relating_basepeople',
-            '_relating_basepeople_hidden_+',
+            '_model_meta_relating_basepeople_hidden_+',
         ],
         Person: [
             'friends_inherited_rel_+',
diff --git a/tests/model_meta/tests.py b/tests/model_meta/tests.py
--- a/tests/model_meta/tests.py
+++ b/tests/model_meta/tests.py
@@ -257,7 +257,7 @@ def test_relations_related_objects(self):
         self.assertEqual(
             sorted(field.related_query_name() for field in BasePerson._meta._relation_tree),
             sorted([
-                '+', '_relating_basepeople_hidden_+', 'BasePerson_following_abstract+',
+                '+', '_model_meta_relating_basepeople_hidden_+', 'BasePerson_following_abstract+',
                 'BasePerson_following_abstract+', 'BasePerson_following_base+', 'BasePerson_following_base+',
                 'BasePerson_friends_abstract+', 'BasePerson_friends_abstract+', 'BasePerson_friends_base+',
                 'BasePerson_friends_base+', 'BasePerson_m2m_abstract+', 'BasePerson_m2m_base+', 'Relating_basepeople+',

```


## Code snippets

### 1 - django/db/models/fields/related.py:

Start line: 1639, End line: 1655

```python
class ManyToManyField(RelatedField):

    def contribute_to_related_class(self, cls, related):
        # Internal M2Ms (i.e., those with a related name ending with '+')
        # and swapped models don't get a related descriptor.
        if not self.remote_field.is_hidden() and not related.related_model._meta.swapped:
            setattr(cls, related.get_accessor_name(), ManyToManyDescriptor(self.remote_field, reverse=True))

        # Set up the accessors for the column names on the m2m table.
        self.m2m_column_name = partial(self._get_m2m_attr, related, 'column')
        self.m2m_reverse_name = partial(self._get_m2m_reverse_attr, related, 'column')

        self.m2m_field_name = partial(self._get_m2m_attr, related, 'name')
        self.m2m_reverse_field_name = partial(self._get_m2m_reverse_attr, related, 'name')

        get_m2m_rel = partial(self._get_m2m_attr, related, 'remote_field')
        self.m2m_target_field_name = lambda: get_m2m_rel().field_name
        get_m2m_reverse_rel = partial(self._get_m2m_reverse_attr, related, 'remote_field')
        self.m2m_reverse_target_field_name = lambda: get_m2m_reverse_rel().field_name
```
### 2 - django/db/models/fields/related.py:

Start line: 1600, End line: 1637

```python
class ManyToManyField(RelatedField):

    def contribute_to_class(self, cls, name, **kwargs):
        # To support multiple relations to self, it's useful to have a non-None
        # related name on symmetrical relations for internal reasons. The
        # concept doesn't make a lot of sense externally ("you want me to
        # specify *what* on my non-reversible relation?!"), so we set it up
        # automatically. The funky name reduces the chance of an accidental
        # clash.
        if self.remote_field.symmetrical and (
            self.remote_field.model == RECURSIVE_RELATIONSHIP_CONSTANT or
            self.remote_field.model == cls._meta.object_name
        ):
            self.remote_field.related_name = "%s_rel_+" % name
        elif self.remote_field.is_hidden():
            # If the backwards relation is disabled, replace the original
            # related_name with one generated from the m2m field name. Django
            # still uses backwards relations internally and we need to avoid
            # clashes between multiple m2m fields with related_name == '+'.
            self.remote_field.related_name = "_%s_%s_+" % (cls.__name__.lower(), name)

        super().contribute_to_class(cls, name, **kwargs)

        # The intermediate m2m model is not auto created if:
        #  1) There is a manually specified intermediate, or
        #  2) The class owning the m2m field is abstract.
        #  3) The class owning the m2m field has been swapped out.
        if not cls._meta.abstract:
            if self.remote_field.through:
                def resolve_through_model(_, model, field):
                    field.remote_field.through = model
                lazy_related_operation(resolve_through_model, cls, self.remote_field.through, field=self)
            elif not cls._meta.swapped:
                self.remote_field.through = create_many_to_many_intermediary_model(self, cls)

        # Add the descriptor for the m2m relation.
        setattr(cls, self.name, ManyToManyDescriptor(self.remote_field, reverse=False))

        # Set up the accessor for the m2m table name for the relation.
        self.m2m_db_table = partial(self._get_m2m_db_table, cls._meta)
```
### 3 - django/db/models/fields/related.py:

Start line: 108, End line: 125

```python
class RelatedField(FieldCacheMixin, Field):

    def _check_related_name_is_valid(self):
        import keyword
        related_name = self.remote_field.related_name
        if related_name is None:
            return []
        is_valid_id = not keyword.iskeyword(related_name) and related_name.isidentifier()
        if not (is_valid_id or related_name.endswith('+')):
            return [
                checks.Error(
                    "The name '%s' is invalid related_name for field %s.%s" %
                    (self.remote_field.related_name, self.model._meta.object_name,
                     self.name),
                    hint="Related name must be a valid Python identifier or end with a '+'",
                    obj=self,
                    id='fields.E306',
                )
            ]
        return []
```
### 4 - django/db/models/fields/related.py:

Start line: 1428, End line: 1469

```python
class ManyToManyField(RelatedField):

    def _check_table_uniqueness(self, **kwargs):
        if isinstance(self.remote_field.through, str) or not self.remote_field.through._meta.managed:
            return []
        registered_tables = {
            model._meta.db_table: model
            for model in self.opts.apps.get_models(include_auto_created=True)
            if model != self.remote_field.through and model._meta.managed
        }
        m2m_db_table = self.m2m_db_table()
        model = registered_tables.get(m2m_db_table)
        # The second condition allows multiple m2m relations on a model if
        # some point to a through model that proxies another through model.
        if model and model._meta.concrete_model != self.remote_field.through._meta.concrete_model:
            if model._meta.auto_created:
                def _get_field_name(model):
                    for field in model._meta.auto_created._meta.many_to_many:
                        if field.remote_field.through is model:
                            return field.name
                opts = model._meta.auto_created._meta
                clashing_obj = '%s.%s' % (opts.label, _get_field_name(model))
            else:
                clashing_obj = model._meta.label
            if settings.DATABASE_ROUTERS:
                error_class, error_id = checks.Warning, 'fields.W344'
                error_hint = (
                    'You have configured settings.DATABASE_ROUTERS. Verify '
                    'that the table of %r is correctly routed to a separate '
                    'database.' % clashing_obj
                )
            else:
                error_class, error_id = checks.Error, 'fields.E340'
                error_hint = None
            return [
                error_class(
                    "The field's intermediary table '%s' clashes with the "
                    "table name of '%s'." % (m2m_db_table, clashing_obj),
                    obj=self,
                    hint=error_hint,
                    id=error_id,
                )
            ]
        return []
```
### 5 - django/db/models/fields/related.py:

Start line: 1235, End line: 1352

```python
class ManyToManyField(RelatedField):

    def _check_relationship_model(self, from_model=None, **kwargs):
        if hasattr(self.remote_field.through, '_meta'):
            qualified_model_name = "%s.%s" % (
                self.remote_field.through._meta.app_label, self.remote_field.through.__name__)
        else:
            qualified_model_name = self.remote_field.through

        errors = []

        if self.remote_field.through not in self.opts.apps.get_models(include_auto_created=True):
            # The relationship model is not installed.
            errors.append(
                checks.Error(
                    "Field specifies a many-to-many relation through model "
                    "'%s', which has not been installed." % qualified_model_name,
                    obj=self,
                    id='fields.E331',
                )
            )

        else:
            assert from_model is not None, (
                "ManyToManyField with intermediate "
                "tables cannot be checked if you don't pass the model "
                "where the field is attached to."
            )
            # Set some useful local variables
            to_model = resolve_relation(from_model, self.remote_field.model)
            from_model_name = from_model._meta.object_name
            if isinstance(to_model, str):
                to_model_name = to_model
            else:
                to_model_name = to_model._meta.object_name
            relationship_model_name = self.remote_field.through._meta.object_name
            self_referential = from_model == to_model
            # Count foreign keys in intermediate model
            if self_referential:
                seen_self = sum(
                    from_model == getattr(field.remote_field, 'model', None)
                    for field in self.remote_field.through._meta.fields
                )

                if seen_self > 2 and not self.remote_field.through_fields:
                    errors.append(
                        checks.Error(
                            "The model is used as an intermediate model by "
                            "'%s', but it has more than two foreign keys "
                            "to '%s', which is ambiguous. You must specify "
                            "which two foreign keys Django should use via the "
                            "through_fields keyword argument." % (self, from_model_name),
                            hint="Use through_fields to specify which two foreign keys Django should use.",
                            obj=self.remote_field.through,
                            id='fields.E333',
                        )
                    )

            else:
                # Count foreign keys in relationship model
                seen_from = sum(
                    from_model == getattr(field.remote_field, 'model', None)
                    for field in self.remote_field.through._meta.fields
                )
                seen_to = sum(
                    to_model == getattr(field.remote_field, 'model', None)
                    for field in self.remote_field.through._meta.fields
                )

                if seen_from > 1 and not self.remote_field.through_fields:
                    errors.append(
                        checks.Error(
                            ("The model is used as an intermediate model by "
                             "'%s', but it has more than one foreign key "
                             "from '%s', which is ambiguous. You must specify "
                             "which foreign key Django should use via the "
                             "through_fields keyword argument.") % (self, from_model_name),
                            hint=(
                                'If you want to create a recursive relationship, '
                                'use ManyToManyField("%s", through="%s").'
                            ) % (
                                RECURSIVE_RELATIONSHIP_CONSTANT,
                                relationship_model_name,
                            ),
                            obj=self,
                            id='fields.E334',
                        )
                    )

                if seen_to > 1 and not self.remote_field.through_fields:
                    errors.append(
                        checks.Error(
                            "The model is used as an intermediate model by "
                            "'%s', but it has more than one foreign key "
                            "to '%s', which is ambiguous. You must specify "
                            "which foreign key Django should use via the "
                            "through_fields keyword argument." % (self, to_model_name),
                            hint=(
                                'If you want to create a recursive relationship, '
                                'use ManyToManyField("%s", through="%s").'
                            ) % (
                                RECURSIVE_RELATIONSHIP_CONSTANT,
                                relationship_model_name,
                            ),
                            obj=self,
                            id='fields.E335',
                        )
                    )

                if seen_from == 0 or seen_to == 0:
                    errors.append(
                        checks.Error(
                            "The model is used as an intermediate model by "
                            "'%s', but it does not have a foreign key to '%s' or '%s'." % (
                                self, from_model_name, to_model_name
                            ),
                            obj=self.remote_field.through,
                            id='fields.E336',
                        )
                    )
        # ... other code
```
### 6 - django/db/models/fields/related.py:

Start line: 253, End line: 282

```python
class RelatedField(FieldCacheMixin, Field):

    def _check_clashes(self):
        # ... other code
        for clash_field in potential_clashes:
            # i.e. "app_label.Model.m2m".
            clash_name = '%s.%s' % (
                clash_field.related_model._meta.label,
                clash_field.field.name,
            )
            if not rel_is_hidden and clash_field.get_accessor_name() == rel_name:
                errors.append(
                    checks.Error(
                        "Reverse accessor for '%s' clashes with reverse accessor for '%s'." % (field_name, clash_name),
                        hint=("Add or change a related_name argument "
                              "to the definition for '%s' or '%s'.") % (field_name, clash_name),
                        obj=self,
                        id='fields.E304',
                    )
                )

            if clash_field.get_accessor_name() == rel_query_name:
                errors.append(
                    checks.Error(
                        "Reverse query name for '%s' clashes with reverse query name for '%s'."
                        % (field_name, clash_name),
                        hint=("Add or change a related_name argument "
                              "to the definition for '%s' or '%s'.") % (field_name, clash_name),
                        obj=self,
                        id='fields.E305',
                    )
                )

        return errors
```
### 7 - django/db/models/fields/related.py:

Start line: 127, End line: 154

```python
class RelatedField(FieldCacheMixin, Field):

    def _check_related_query_name_is_valid(self):
        if self.remote_field.is_hidden():
            return []
        rel_query_name = self.related_query_name()
        errors = []
        if rel_query_name.endswith('_'):
            errors.append(
                checks.Error(
                    "Reverse query name '%s' must not end with an underscore."
                    % rel_query_name,
                    hint=("Add or change a related_name or related_query_name "
                          "argument for this field."),
                    obj=self,
                    id='fields.E308',
                )
            )
        if LOOKUP_SEP in rel_query_name:
            errors.append(
                checks.Error(
                    "Reverse query name '%s' must not contain '%s'."
                    % (rel_query_name, LOOKUP_SEP),
                    hint=("Add or change a related_name or related_query_name "
                          "argument for this field."),
                    obj=self,
                    id='fields.E309',
                )
            )
        return errors
```
### 8 - django/db/models/fields/related.py:

Start line: 156, End line: 169

```python
class RelatedField(FieldCacheMixin, Field):

    def _check_relation_model_exists(self):
        rel_is_missing = self.remote_field.model not in self.opts.apps.get_models()
        rel_is_string = isinstance(self.remote_field.model, str)
        model_name = self.remote_field.model if rel_is_string else self.remote_field.model._meta.object_name
        if rel_is_missing and (rel_is_string or not self.remote_field.model._meta.swapped):
            return [
                checks.Error(
                    "Field defines a relation with model '%s', which is either "
                    "not installed, or is abstract." % model_name,
                    obj=self,
                    id='fields.E300',
                )
            ]
        return []
```
### 9 - django/db/models/fields/related.py:

Start line: 1354, End line: 1426

```python
class ManyToManyField(RelatedField):

    def _check_relationship_model(self, from_model=None, **kwargs):

        # Validate `through_fields`.
        if self.remote_field.through_fields is not None:
            # Validate that we're given an iterable of at least two items
            # and that none of them is "falsy".
            if not (len(self.remote_field.through_fields) >= 2 and
                    self.remote_field.through_fields[0] and self.remote_field.through_fields[1]):
                errors.append(
                    checks.Error(
                        "Field specifies 'through_fields' but does not provide "
                        "the names of the two link fields that should be used "
                        "for the relation through model '%s'." % qualified_model_name,
                        hint="Make sure you specify 'through_fields' as through_fields=('field1', 'field2')",
                        obj=self,
                        id='fields.E337',
                    )
                )

            # Validate the given through fields -- they should be actual
            # fields on the through model, and also be foreign keys to the
            # expected models.
            else:
                assert from_model is not None, (
                    "ManyToManyField with intermediate "
                    "tables cannot be checked if you don't pass the model "
                    "where the field is attached to."
                )

                source, through, target = from_model, self.remote_field.through, self.remote_field.model
                source_field_name, target_field_name = self.remote_field.through_fields[:2]

                for field_name, related_model in ((source_field_name, source),
                                                  (target_field_name, target)):

                    possible_field_names = []
                    for f in through._meta.fields:
                        if hasattr(f, 'remote_field') and getattr(f.remote_field, 'model', None) == related_model:
                            possible_field_names.append(f.name)
                    if possible_field_names:
                        hint = "Did you mean one of the following foreign keys to '%s': %s?" % (
                            related_model._meta.object_name,
                            ', '.join(possible_field_names),
                        )
                    else:
                        hint = None

                    try:
                        field = through._meta.get_field(field_name)
                    except exceptions.FieldDoesNotExist:
                        errors.append(
                            checks.Error(
                                "The intermediary model '%s' has no field '%s'."
                                % (qualified_model_name, field_name),
                                hint=hint,
                                obj=self,
                                id='fields.E338',
                            )
                        )
                    else:
                        if not (hasattr(field, 'remote_field') and
                                getattr(field.remote_field, 'model', None) == related_model):
                            errors.append(
                                checks.Error(
                                    "'%s.%s' is not a foreign key to '%s'." % (
                                        through._meta.object_name, field_name,
                                        related_model._meta.object_name,
                                    ),
                                    hint=hint,
                                    obj=self,
                                    id='fields.E339',
                                )
                            )

        return errors
```
### 10 - django/db/models/fields/related.py:

Start line: 171, End line: 184

```python
class RelatedField(FieldCacheMixin, Field):

    def _check_referencing_to_swapped_model(self):
        if (self.remote_field.model not in self.opts.apps.get_models() and
                not isinstance(self.remote_field.model, str) and
                self.remote_field.model._meta.swapped):
            return [
                checks.Error(
                    "Field defines a relation with the model '%s', which has "
                    "been swapped out." % self.remote_field.model._meta.label,
                    hint="Update the relation to point at 'settings.%s'." % self.remote_field.model._meta.swappable,
                    obj=self,
                    id='fields.E301',
                )
            ]
        return []
```
### 11 - django/db/models/fields/related.py:

Start line: 935, End line: 948

```python
class ForeignKey(ForeignObject):

    def resolve_related_fields(self):
        related_fields = super().resolve_related_fields()
        for from_field, to_field in related_fields:
            if to_field and to_field.model != self.remote_field.model._meta.concrete_model:
                raise exceptions.FieldError(
                    "'%s.%s' refers to field '%s' which is not local to model "
                    "'%s'." % (
                        self.model._meta.label,
                        self.name,
                        to_field.name,
                        self.remote_field.model._meta.concrete_model._meta.label,
                    )
                )
        return related_fields
```
### 12 - django/db/models/fields/related.py:

Start line: 284, End line: 318

```python
class RelatedField(FieldCacheMixin, Field):

    def db_type(self, connection):
        # By default related field will not have a column as it relates to
        # columns from another table.
        return None

    def contribute_to_class(self, cls, name, private_only=False, **kwargs):

        super().contribute_to_class(cls, name, private_only=private_only, **kwargs)

        self.opts = cls._meta

        if not cls._meta.abstract:
            if self.remote_field.related_name:
                related_name = self.remote_field.related_name
            else:
                related_name = self.opts.default_related_name
            if related_name:
                related_name = related_name % {
                    'class': cls.__name__.lower(),
                    'model_name': cls._meta.model_name.lower(),
                    'app_label': cls._meta.app_label.lower()
                }
                self.remote_field.related_name = related_name

            if self.remote_field.related_query_name:
                related_query_name = self.remote_field.related_query_name % {
                    'class': cls.__name__.lower(),
                    'app_label': cls._meta.app_label.lower(),
                }
                self.remote_field.related_query_name = related_query_name

            def resolve_related_class(model, related, field):
                field.remote_field.model = related
                field.do_related_class(related, model)
            lazy_related_operation(resolve_related_class, cls, self.remote_field.model, field=self)
```
### 13 - django/db/models/fields/related.py:

Start line: 509, End line: 574

```python
class ForeignObject(RelatedField):

    def _check_unique_target(self):
        rel_is_string = isinstance(self.remote_field.model, str)
        if rel_is_string or not self.requires_unique_target:
            return []

        try:
            self.foreign_related_fields
        except exceptions.FieldDoesNotExist:
            return []

        if not self.foreign_related_fields:
            return []

        unique_foreign_fields = {
            frozenset([f.name])
            for f in self.remote_field.model._meta.get_fields()
            if getattr(f, 'unique', False)
        }
        unique_foreign_fields.update({
            frozenset(ut)
            for ut in self.remote_field.model._meta.unique_together
        })
        unique_foreign_fields.update({
            frozenset(uc.fields)
            for uc in self.remote_field.model._meta.total_unique_constraints
        })
        foreign_fields = {f.name for f in self.foreign_related_fields}
        has_unique_constraint = any(u <= foreign_fields for u in unique_foreign_fields)

        if not has_unique_constraint and len(self.foreign_related_fields) > 1:
            field_combination = ', '.join(
                "'%s'" % rel_field.name for rel_field in self.foreign_related_fields
            )
            model_name = self.remote_field.model.__name__
            return [
                checks.Error(
                    "No subset of the fields %s on model '%s' is unique."
                    % (field_combination, model_name),
                    hint=(
                        'Mark a single field as unique=True or add a set of '
                        'fields to a unique constraint (via unique_together '
                        'or a UniqueConstraint (without condition) in the '
                        'model Meta.constraints).'
                    ),
                    obj=self,
                    id='fields.E310',
                )
            ]
        elif not has_unique_constraint:
            field_name = self.foreign_related_fields[0].name
            model_name = self.remote_field.model.__name__
            return [
                checks.Error(
                    "'%s.%s' must be unique because it is referenced by "
                    "a foreign key." % (model_name, field_name),
                    hint=(
                        'Add unique=True to this field or add a '
                        'UniqueConstraint (without condition) in the model '
                        'Meta.constraints.'
                    ),
                    obj=self,
                    id='fields.E311',
                )
            ]
        else:
            return []
```
### 15 - django/db/models/fields/related.py:

Start line: 487, End line: 507

```python
class ForeignObject(RelatedField):

    def _check_to_fields_exist(self):
        # Skip nonexistent models.
        if isinstance(self.remote_field.model, str):
            return []

        errors = []
        for to_field in self.to_fields:
            if to_field:
                try:
                    self.remote_field.model._meta.get_field(to_field)
                except exceptions.FieldDoesNotExist:
                    errors.append(
                        checks.Error(
                            "The to_field '%s' doesn't exist on the related "
                            "model '%s'."
                            % (to_field, self.remote_field.model._meta.label),
                            obj=self,
                            id='fields.E312',
                        )
                    )
        return errors
```
### 16 - django/db/models/fields/related.py:

Start line: 750, End line: 768

```python
class ForeignObject(RelatedField):

    def contribute_to_related_class(self, cls, related):
        # Internal FK's - i.e., those with a related name ending with '+' -
        # and swapped models don't get a related descriptor.
        if not self.remote_field.is_hidden() and not related.related_model._meta.swapped:
            setattr(cls._meta.concrete_model, related.get_accessor_name(), self.related_accessor_class(related))
            # While 'limit_choices_to' might be a callable, simply pass
            # it along for later - this is too early because it's still
            # model load time.
            if self.remote_field.limit_choices_to:
                cls._meta.related_fkey_lookups.append(self.remote_field.limit_choices_to)


ForeignObject.register_lookup(RelatedIn)
ForeignObject.register_lookup(RelatedExact)
ForeignObject.register_lookup(RelatedLessThan)
ForeignObject.register_lookup(RelatedGreaterThan)
ForeignObject.register_lookup(RelatedGreaterThanOrEqual)
ForeignObject.register_lookup(RelatedLessThanOrEqual)
ForeignObject.register_lookup(RelatedIsNull)
```
### 17 - django/db/models/fields/related.py:

Start line: 1657, End line: 1691

```python
class ManyToManyField(RelatedField):

    def set_attributes_from_rel(self):
        pass

    def value_from_object(self, obj):
        return [] if obj.pk is None else list(getattr(obj, self.attname).all())

    def save_form_data(self, instance, data):
        getattr(instance, self.attname).set(data)

    def formfield(self, *, using=None, **kwargs):
        defaults = {
            'form_class': forms.ModelMultipleChoiceField,
            'queryset': self.remote_field.model._default_manager.using(using),
            **kwargs,
        }
        # If initial is passed in, it's a list of related objects, but the
        # MultipleChoiceField takes a list of IDs.
        if defaults.get('initial') is not None:
            initial = defaults['initial']
            if callable(initial):
                initial = initial()
            defaults['initial'] = [i.pk for i in initial]
        return super().formfield(**defaults)

    def db_check(self, connection):
        return None

    def db_type(self, connection):
        # A ManyToManyField is not represented by a single column,
        # so return None.
        return None

    def db_parameters(self, connection):
        return {"type": None, "check": None}
```
### 19 - django/db/models/fields/related.py:

Start line: 421, End line: 441

```python
class RelatedField(FieldCacheMixin, Field):

    def related_query_name(self):
        """
        Define the name that can be used to identify this related object in a
        table-spanning query.
        """
        return self.remote_field.related_query_name or self.remote_field.related_name or self.opts.model_name

    @property
    def target_field(self):
        """
        When filtering against this relation, return the field on the remote
        model against which the filtering should happen.
        """
        target_fields = self.get_path_info()[-1].target_fields
        if len(target_fields) > 1:
            raise exceptions.FieldError(
                "The relation has multiple target fields, but only single target field was asked for")
        return target_fields[0]

    def get_cache_name(self):
        return self.name
```
### 20 - django/db/models/fields/related.py:

Start line: 611, End line: 628

```python
class ForeignObject(RelatedField):

    def resolve_related_fields(self):
        if not self.from_fields or len(self.from_fields) != len(self.to_fields):
            raise ValueError('Foreign Object from and to fields must be the same non-zero length')
        if isinstance(self.remote_field.model, str):
            raise ValueError('Related model %r cannot be resolved' % self.remote_field.model)
        related_fields = []
        for index in range(len(self.from_fields)):
            from_field_name = self.from_fields[index]
            to_field_name = self.to_fields[index]
            from_field = (
                self
                if from_field_name == RECURSIVE_RELATIONSHIP_CONSTANT
                else self.opts.get_field(from_field_name)
            )
            to_field = (self.remote_field.model._meta.pk if to_field_name is None
                        else self.remote_field.model._meta.get_field(to_field_name))
            related_fields.append((from_field, to_field))
        return related_fields
```
### 21 - django/db/models/fields/related.py:

Start line: 864, End line: 890

```python
class ForeignKey(ForeignObject):

    def _check_unique(self, **kwargs):
        return [
            checks.Warning(
                'Setting unique=True on a ForeignKey has the same effect as using a OneToOneField.',
                hint='ForeignKey(unique=True) is usually better served by a OneToOneField.',
                obj=self,
                id='fields.W342',
            )
        ] if self.unique else []

    def deconstruct(self):
        name, path, args, kwargs = super().deconstruct()
        del kwargs['to_fields']
        del kwargs['from_fields']
        # Handle the simpler arguments
        if self.db_index:
            del kwargs['db_index']
        else:
            kwargs['db_index'] = False
        if self.db_constraint is not True:
            kwargs['db_constraint'] = self.db_constraint
        # Rel needs more work.
        to_meta = getattr(self.remote_field.model, "_meta", None)
        if self.remote_field.field_name and (
                not to_meta or (to_meta.pk and self.remote_field.field_name != to_meta.pk.name)):
            kwargs['to_field'] = self.remote_field.field_name
        return name, path, args, kwargs
```
### 23 - django/db/models/fields/related.py:

Start line: 984, End line: 995

```python
class ForeignKey(ForeignObject):

    def formfield(self, *, using=None, **kwargs):
        if isinstance(self.remote_field.model, str):
            raise ValueError("Cannot create form field for %r yet, because "
                             "its related model %r has not been loaded yet" %
                             (self.name, self.remote_field.model))
        return super().formfield(**{
            'form_class': forms.ModelChoiceField,
            'queryset': self.remote_field.model._default_manager.using(using),
            'to_field_name': self.remote_field.field_name,
            **kwargs,
            'blank': self.blank,
        })
```
### 24 - django/db/models/fields/related.py:

Start line: 1077, End line: 1121

```python
def create_many_to_many_intermediary_model(field, klass):
    from django.db import models

    def set_managed(model, related, through):
        through._meta.managed = model._meta.managed or related._meta.managed

    to_model = resolve_relation(klass, field.remote_field.model)
    name = '%s_%s' % (klass._meta.object_name, field.name)
    lazy_related_operation(set_managed, klass, to_model, name)

    to = make_model_tuple(to_model)[1]
    from_ = klass._meta.model_name
    if to == from_:
        to = 'to_%s' % to
        from_ = 'from_%s' % from_

    meta = type('Meta', (), {
        'db_table': field._get_m2m_db_table(klass._meta),
        'auto_created': klass,
        'app_label': klass._meta.app_label,
        'db_tablespace': klass._meta.db_tablespace,
        'unique_together': (from_, to),
        'verbose_name': _('%(from)s-%(to)s relationship') % {'from': from_, 'to': to},
        'verbose_name_plural': _('%(from)s-%(to)s relationships') % {'from': from_, 'to': to},
        'apps': field.model._meta.apps,
    })
    # Construct and return the new class.
    return type(name, (models.Model,), {
        'Meta': meta,
        '__module__': klass.__module__,
        from_: models.ForeignKey(
            klass,
            related_name='%s+' % name,
            db_tablespace=field.db_tablespace,
            db_constraint=field.remote_field.db_constraint,
            on_delete=CASCADE,
        ),
        to: models.ForeignKey(
            to_model,
            related_name='%s+' % name,
            db_tablespace=field.db_tablespace,
            db_constraint=field.remote_field.db_constraint,
            on_delete=CASCADE,
        )
    })
```
### 25 - django/db/models/fields/related.py:

Start line: 913, End line: 933

```python
class ForeignKey(ForeignObject):

    def validate(self, value, model_instance):
        if self.remote_field.parent_link:
            return
        super().validate(value, model_instance)
        if value is None:
            return

        using = router.db_for_read(self.remote_field.model, instance=model_instance)
        qs = self.remote_field.model._base_manager.using(using).filter(
            **{self.remote_field.field_name: value}
        )
        qs = qs.complex_filter(self.get_limit_choices_to())
        if not qs.exists():
            raise exceptions.ValidationError(
                self.error_messages['invalid'],
                code='invalid',
                params={
                    'model': self.remote_field.model._meta.verbose_name, 'pk': value,
                    'field': self.remote_field.field_name, 'value': value,
                },  # 'pk' is included for backwards compatibility
            )
```
### 26 - django/db/models/fields/related.py:

Start line: 1533, End line: 1550

```python
class ManyToManyField(RelatedField):

    def get_path_info(self, filtered_relation=None):
        return self._get_path_info(direct=True, filtered_relation=filtered_relation)

    def get_reverse_path_info(self, filtered_relation=None):
        return self._get_path_info(direct=False, filtered_relation=filtered_relation)

    def _get_m2m_db_table(self, opts):
        """
        Function that can be curried to provide the m2m table name for this
        relation.
        """
        if self.remote_field.through is not None:
            return self.remote_field.through._meta.db_table
        elif self.db_table:
            return self.db_table
        else:
            m2m_table_name = '%s_%s' % (utils.strip_quotes(opts.db_table), self.name)
            return utils.truncate_name(m2m_table_name, connection.ops.max_name_length())
```
### 27 - django/db/models/fields/related.py:

Start line: 1202, End line: 1233

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
### 28 - django/db/models/fields/related.py:

Start line: 841, End line: 862

```python
class ForeignKey(ForeignObject):

    def _check_on_delete(self):
        on_delete = getattr(self.remote_field, 'on_delete', None)
        if on_delete == SET_NULL and not self.null:
            return [
                checks.Error(
                    'Field specifies on_delete=SET_NULL, but cannot be null.',
                    hint='Set null=True argument on the field, or change the on_delete rule.',
                    obj=self,
                    id='fields.E320',
                )
            ]
        elif on_delete == SET_DEFAULT and not self.has_default():
            return [
                checks.Error(
                    'Field specifies on_delete=SET_DEFAULT, but has no default value.',
                    hint='Set a default value, or change the on_delete rule.',
                    obj=self,
                    id='fields.E321',
                )
            ]
        else:
            return []
```
### 29 - django/db/models/fields/related.py:

Start line: 186, End line: 252

```python
class RelatedField(FieldCacheMixin, Field):

    def _check_clashes(self):
        """Check accessor and reverse query name clashes."""
        from django.db.models.base import ModelBase

        errors = []
        opts = self.model._meta

        # `f.remote_field.model` may be a string instead of a model. Skip if model name is
        # not resolved.
        if not isinstance(self.remote_field.model, ModelBase):
            return []

        # Consider that we are checking field `Model.foreign` and the models
        # are:
        #
        #     class Target(models.Model):
        #         model = models.IntegerField()
        #         model_set = models.IntegerField()
        #
        #     class Model(models.Model):
        #         foreign = models.ForeignKey(Target)
        #         m2m = models.ManyToManyField(Target)

        # rel_opts.object_name == "Target"
        rel_opts = self.remote_field.model._meta
        # If the field doesn't install a backward relation on the target model
        # (so `is_hidden` returns True), then there are no clashes to check
        # and we can skip these fields.
        rel_is_hidden = self.remote_field.is_hidden()
        rel_name = self.remote_field.get_accessor_name()  # i. e. "model_set"
        rel_query_name = self.related_query_name()  # i. e. "model"
        # i.e. "app_label.Model.field".
        field_name = '%s.%s' % (opts.label, self.name)

        # Check clashes between accessor or reverse query name of `field`
        # and any other field name -- i.e. accessor for Model.foreign is
        # model_set and it clashes with Target.model_set.
        potential_clashes = rel_opts.fields + rel_opts.many_to_many
        for clash_field in potential_clashes:
            # i.e. "app_label.Target.model_set".
            clash_name = '%s.%s' % (rel_opts.label, clash_field.name)
            if not rel_is_hidden and clash_field.name == rel_name:
                errors.append(
                    checks.Error(
                        "Reverse accessor for '%s' clashes with field name '%s'." % (field_name, clash_name),
                        hint=("Rename field '%s', or add/change a related_name "
                              "argument to the definition for field '%s'.") % (clash_name, field_name),
                        obj=self,
                        id='fields.E302',
                    )
                )

            if clash_field.name == rel_query_name:
                errors.append(
                    checks.Error(
                        "Reverse query name for '%s' clashes with field name '%s'." % (field_name, clash_name),
                        hint=("Rename field '%s', or add/change a related_name "
                              "argument to the definition for field '%s'.") % (clash_name, field_name),
                        obj=self,
                        id='fields.E303',
                    )
                )

        # Check clashes between accessors/reverse query names of `field` and
        # any other field accessor -- i. e. Model.foreign accessor clashes with
        # Model.m2m accessor.
        potential_clashes = (r for r in rel_opts.related_objects if r.field is not self)
        # ... other code
```
### 30 - django/db/models/fields/related.py:

Start line: 1, End line: 34

```python
import functools
import inspect
from functools import partial

from django import forms
from django.apps import apps
from django.conf import SettingsReference, settings
from django.core import checks, exceptions
from django.db import connection, router
from django.db.backends import utils
from django.db.models import Q
from django.db.models.constants import LOOKUP_SEP
from django.db.models.deletion import CASCADE, SET_DEFAULT, SET_NULL
from django.db.models.query_utils import PathInfo
from django.db.models.utils import make_model_tuple
from django.utils.functional import cached_property
from django.utils.translation import gettext_lazy as _

from . import Field
from .mixins import FieldCacheMixin
from .related_descriptors import (
    ForeignKeyDeferredAttribute, ForwardManyToOneDescriptor,
    ForwardOneToOneDescriptor, ManyToManyDescriptor,
    ReverseManyToOneDescriptor, ReverseOneToOneDescriptor,
)
from .related_lookups import (
    RelatedExact, RelatedGreaterThan, RelatedGreaterThanOrEqual, RelatedIn,
    RelatedIsNull, RelatedLessThan, RelatedLessThanOrEqual,
)
from .reverse_related import (
    ForeignObjectRel, ManyToManyRel, ManyToOneRel, OneToOneRel,
)

RECURSIVE_RELATIONSHIP_CONSTANT = 'self'
```
### 31 - django/db/models/fields/related.py:

Start line: 630, End line: 650

```python
class ForeignObject(RelatedField):

    @cached_property
    def related_fields(self):
        return self.resolve_related_fields()

    @cached_property
    def reverse_related_fields(self):
        return [(rhs_field, lhs_field) for lhs_field, rhs_field in self.related_fields]

    @cached_property
    def local_related_fields(self):
        return tuple(lhs_field for lhs_field, rhs_field in self.related_fields)

    @cached_property
    def foreign_related_fields(self):
        return tuple(rhs_field for lhs_field, rhs_field in self.related_fields if rhs_field)

    def get_local_related_value(self, instance):
        return self.get_instance_value_for_fields(instance, self.local_related_fields)

    def get_foreign_related_value(self, instance):
        return self.get_instance_value_for_fields(instance, self.foreign_related_fields)
```
### 34 - django/db/models/fields/related.py:

Start line: 1552, End line: 1568

```python
class ManyToManyField(RelatedField):

    def _get_m2m_attr(self, related, attr):
        """
        Function that can be curried to provide the source accessor or DB
        column name for the m2m table.
        """
        cache_attr = '_m2m_%s_cache' % attr
        if hasattr(self, cache_attr):
            return getattr(self, cache_attr)
        if self.remote_field.through_fields is not None:
            link_field_name = self.remote_field.through_fields[0]
        else:
            link_field_name = None
        for f in self.remote_field.through._meta.fields:
            if (f.is_relation and f.remote_field.model == related.related_model and
                    (link_field_name is None or link_field_name == f.name)):
                setattr(self, cache_attr, getattr(f, attr))
                return getattr(self, cache_attr)
```
### 38 - django/db/models/fields/related.py:

Start line: 1570, End line: 1598

```python
class ManyToManyField(RelatedField):

    def _get_m2m_reverse_attr(self, related, attr):
        """
        Function that can be curried to provide the related accessor or DB
        column name for the m2m table.
        """
        cache_attr = '_m2m_reverse_%s_cache' % attr
        if hasattr(self, cache_attr):
            return getattr(self, cache_attr)
        found = False
        if self.remote_field.through_fields is not None:
            link_field_name = self.remote_field.through_fields[1]
        else:
            link_field_name = None
        for f in self.remote_field.through._meta.fields:
            if f.is_relation and f.remote_field.model == related.model:
                if link_field_name is None and related.related_model == related.model:
                    # If this is an m2m-intermediate to self,
                    # the first foreign key you find will be
                    # the source column. Keep searching for
                    # the second foreign key.
                    if found:
                        setattr(self, cache_attr, getattr(f, attr))
                        break
                    else:
                        found = True
                elif link_field_name is None or link_field_name == f.name:
                    setattr(self, cache_attr, getattr(f, attr))
                    break
        return getattr(self, cache_attr)
```
### 40 - django/db/models/fields/related.py:

Start line: 997, End line: 1024

```python
class ForeignKey(ForeignObject):

    def db_check(self, connection):
        return []

    def db_type(self, connection):
        return self.target_field.rel_db_type(connection=connection)

    def db_parameters(self, connection):
        return {"type": self.db_type(connection), "check": self.db_check(connection)}

    def convert_empty_strings(self, value, expression, connection):
        if (not value) and isinstance(value, str):
            return None
        return value

    def get_db_converters(self, connection):
        converters = super().get_db_converters(connection)
        if connection.features.interprets_empty_strings_as_nulls:
            converters += [self.convert_empty_strings]
        return converters

    def get_col(self, alias, output_field=None):
        if output_field is None:
            output_field = self.target_field
            while isinstance(output_field, ForeignKey):
                output_field = output_field.target_field
                if output_field is self:
                    raise ValueError('Cannot resolve output_field.')
        return super().get_col(alias, output_field)
```
### 42 - django/db/models/fields/related.py:

Start line: 1124, End line: 1200

```python
class ManyToManyField(RelatedField):
    """
    Provide a many-to-many relation by using an intermediary model that
    holds two ForeignKey fields pointed at the two sides of the relation.

    Unless a ``through`` model was provided, ManyToManyField will use the
    create_many_to_many_intermediary_model factory to automatically generate
    the intermediary model.
    """

    # Field flags
    many_to_many = True
    many_to_one = False
    one_to_many = False
    one_to_one = False

    rel_class = ManyToManyRel

    description = _("Many-to-many relationship")

    def __init__(self, to, related_name=None, related_query_name=None,
                 limit_choices_to=None, symmetrical=None, through=None,
                 through_fields=None, db_constraint=True, db_table=None,
                 swappable=True, **kwargs):
        try:
            to._meta
        except AttributeError:
            assert isinstance(to, str), (
                "%s(%r) is invalid. First parameter to ManyToManyField must be "
                "either a model, a model name, or the string %r" %
                (self.__class__.__name__, to, RECURSIVE_RELATIONSHIP_CONSTANT)
            )

        if symmetrical is None:
            symmetrical = (to == RECURSIVE_RELATIONSHIP_CONSTANT)

        if through is not None:
            assert db_table is None, (
                "Cannot specify a db_table if an intermediary model is used."
            )

        kwargs['rel'] = self.rel_class(
            self, to,
            related_name=related_name,
            related_query_name=related_query_name,
            limit_choices_to=limit_choices_to,
            symmetrical=symmetrical,
            through=through,
            through_fields=through_fields,
            db_constraint=db_constraint,
        )
        self.has_null_arg = 'null' in kwargs

        super().__init__(**kwargs)

        self.db_table = db_table
        self.swappable = swappable

    def check(self, **kwargs):
        return [
            *super().check(**kwargs),
            *self._check_unique(**kwargs),
            *self._check_relationship_model(**kwargs),
            *self._check_ignored_options(**kwargs),
            *self._check_table_uniqueness(**kwargs),
        ]

    def _check_unique(self, **kwargs):
        if self.unique:
            return [
                checks.Error(
                    'ManyToManyFields cannot be unique.',
                    obj=self,
                    id='fields.E330',
                )
            ]
        return []
```
### 44 - django/db/models/fields/related.py:

Start line: 83, End line: 106

```python
class RelatedField(FieldCacheMixin, Field):
    """Base class that all relational fields inherit from."""

    # Field flags
    one_to_many = False
    one_to_one = False
    many_to_many = False
    many_to_one = False

    @cached_property
    def related_model(self):
        # Can't cache this property until all the models are loaded.
        apps.check_models_ready()
        return self.remote_field.model

    def check(self, **kwargs):
        return [
            *super().check(**kwargs),
            *self._check_related_name_is_valid(),
            *self._check_related_query_name_is_valid(),
            *self._check_relation_model_exists(),
            *self._check_referencing_to_swapped_model(),
            *self._check_clashes(),
        ]
```
### 47 - django/db/models/fields/related.py:

Start line: 1471, End line: 1505

```python
class ManyToManyField(RelatedField):

    def deconstruct(self):
        name, path, args, kwargs = super().deconstruct()
        # Handle the simpler arguments.
        if self.db_table is not None:
            kwargs['db_table'] = self.db_table
        if self.remote_field.db_constraint is not True:
            kwargs['db_constraint'] = self.remote_field.db_constraint
        # Rel needs more work.
        if isinstance(self.remote_field.model, str):
            kwargs['to'] = self.remote_field.model
        else:
            kwargs['to'] = self.remote_field.model._meta.label
        if getattr(self.remote_field, 'through', None) is not None:
            if isinstance(self.remote_field.through, str):
                kwargs['through'] = self.remote_field.through
            elif not self.remote_field.through._meta.auto_created:
                kwargs['through'] = self.remote_field.through._meta.label
        # If swappable is True, then see if we're actually pointing to the target
        # of a swap.
        swappable_setting = self.swappable_setting
        if swappable_setting is not None:
            # If it's already a settings reference, error.
            if hasattr(kwargs['to'], "setting_name"):
                if kwargs['to'].setting_name != swappable_setting:
                    raise ValueError(
                        "Cannot deconstruct a ManyToManyField pointing to a "
                        "model that is swapped in place of more than one model "
                        "(%s and %s)" % (kwargs['to'].setting_name, swappable_setting)
                    )

            kwargs['to'] = SettingsReference(
                kwargs['to'],
                swappable_setting,
            )
        return name, path, args, kwargs
```
### 51 - django/db/models/fields/related.py:

Start line: 950, End line: 982

```python
class ForeignKey(ForeignObject):

    def get_attname(self):
        return '%s_id' % self.name

    def get_attname_column(self):
        attname = self.get_attname()
        column = self.db_column or attname
        return attname, column

    def get_default(self):
        """Return the to_field if the default value is an object."""
        field_default = super().get_default()
        if isinstance(field_default, self.remote_field.model):
            return getattr(field_default, self.target_field.attname)
        return field_default

    def get_db_prep_save(self, value, connection):
        if value is None or (value == '' and
                             (not self.target_field.empty_strings_allowed or
                              connection.features.interprets_empty_strings_as_nulls)):
            return None
        else:
            return self.target_field.get_db_prep_save(value, connection=connection)

    def get_db_prep_value(self, value, connection, prepared=False):
        return self.target_field.get_db_prep_value(value, connection, prepared)

    def get_prep_value(self, value):
        return self.target_field.get_prep_value(value)

    def contribute_to_related_class(self, cls, related):
        super().contribute_to_related_class(cls, related)
        if self.remote_field.field_name is None:
            self.remote_field.field_name = cls._meta.pk.name
```
### 52 - django/db/models/fields/related.py:

Start line: 771, End line: 839

```python
class ForeignKey(ForeignObject):
    """
    Provide a many-to-one relation by adding a column to the local model
    to hold the remote value.

    By default ForeignKey will target the pk of the remote model but this
    behavior can be changed by using the ``to_field`` argument.
    """
    descriptor_class = ForeignKeyDeferredAttribute
    # Field flags
    many_to_many = False
    many_to_one = True
    one_to_many = False
    one_to_one = False

    rel_class = ManyToOneRel

    empty_strings_allowed = False
    default_error_messages = {
        'invalid': _('%(model)s instance with %(field)s %(value)r does not exist.')
    }
    description = _("Foreign Key (type determined by related field)")

    def __init__(self, to, on_delete, related_name=None, related_query_name=None,
                 limit_choices_to=None, parent_link=False, to_field=None,
                 db_constraint=True, **kwargs):
        try:
            to._meta.model_name
        except AttributeError:
            assert isinstance(to, str), (
                "%s(%r) is invalid. First parameter to ForeignKey must be "
                "either a model, a model name, or the string %r" % (
                    self.__class__.__name__, to,
                    RECURSIVE_RELATIONSHIP_CONSTANT,
                )
            )
        else:
            # For backwards compatibility purposes, we need to *try* and set
            # the to_field during FK construction. It won't be guaranteed to
            # be correct until contribute_to_class is called. Refs #12190.
            to_field = to_field or (to._meta.pk and to._meta.pk.name)
        if not callable(on_delete):
            raise TypeError('on_delete must be callable.')

        kwargs['rel'] = self.rel_class(
            self, to, to_field,
            related_name=related_name,
            related_query_name=related_query_name,
            limit_choices_to=limit_choices_to,
            parent_link=parent_link,
            on_delete=on_delete,
        )
        kwargs.setdefault('db_index', True)

        super().__init__(
            to,
            on_delete,
            from_fields=[RECURSIVE_RELATIONSHIP_CONSTANT],
            to_fields=[to_field],
            **kwargs,
        )
        self.db_constraint = db_constraint

    def check(self, **kwargs):
        return [
            *super().check(**kwargs),
            *self._check_on_delete(),
            *self._check_unique(),
        ]
```
### 64 - django/db/models/fields/related.py:

Start line: 1027, End line: 1074

```python
class OneToOneField(ForeignKey):
    """
    A OneToOneField is essentially the same as a ForeignKey, with the exception
    that it always carries a "unique" constraint with it and the reverse
    relation always returns the object pointed to (since there will only ever
    be one), rather than returning a list.
    """

    # Field flags
    many_to_many = False
    many_to_one = False
    one_to_many = False
    one_to_one = True

    related_accessor_class = ReverseOneToOneDescriptor
    forward_related_accessor_class = ForwardOneToOneDescriptor
    rel_class = OneToOneRel

    description = _("One-to-one relationship")

    def __init__(self, to, on_delete, to_field=None, **kwargs):
        kwargs['unique'] = True
        super().__init__(to, on_delete, to_field=to_field, **kwargs)

    def deconstruct(self):
        name, path, args, kwargs = super().deconstruct()
        if "unique" in kwargs:
            del kwargs['unique']
        return name, path, args, kwargs

    def formfield(self, **kwargs):
        if self.remote_field.parent_link:
            return None
        return super().formfield(**kwargs)

    def save_form_data(self, instance, data):
        if isinstance(data, self.remote_field.model):
            setattr(instance, self.name, data)
        else:
            setattr(instance, self.attname, data)
            # Remote field object must be cleared otherwise Model.save()
            # will reassign attname using the related object pk.
            if data is None:
                setattr(instance, self.name, data)

    def _check_unique(self, **kwargs):
        # Override ForeignKey since check isn't applicable here.
        return []
```
### 80 - django/db/models/fields/related.py:

Start line: 1507, End line: 1531

```python
class ManyToManyField(RelatedField):

    def _get_path_info(self, direct=False, filtered_relation=None):
        """Called by both direct and indirect m2m traversal."""
        int_model = self.remote_field.through
        linkfield1 = int_model._meta.get_field(self.m2m_field_name())
        linkfield2 = int_model._meta.get_field(self.m2m_reverse_field_name())
        if direct:
            join1infos = linkfield1.get_reverse_path_info()
            join2infos = linkfield2.get_path_info(filtered_relation)
        else:
            join1infos = linkfield2.get_reverse_path_info()
            join2infos = linkfield1.get_path_info(filtered_relation)

        # Get join infos between the last model of join 1 and the first model
        # of join 2. Assume the only reason these may differ is due to model
        # inheritance.
        join1_final = join1infos[-1].to_opts
        join2_initial = join2infos[0].from_opts
        if join1_final is join2_initial:
            intermediate_infos = []
        elif issubclass(join1_final.model, join2_initial.model):
            intermediate_infos = join1_final.get_path_to_parent(join2_initial.model)
        else:
            intermediate_infos = join2_initial.get_path_from_parent(join1_final.model)

        return [*join1infos, *intermediate_infos, *join2infos]
```
### 81 - django/db/models/fields/related.py:

Start line: 576, End line: 609

```python
class ForeignObject(RelatedField):

    def deconstruct(self):
        name, path, args, kwargs = super().deconstruct()
        kwargs['on_delete'] = self.remote_field.on_delete
        kwargs['from_fields'] = self.from_fields
        kwargs['to_fields'] = self.to_fields

        if self.remote_field.parent_link:
            kwargs['parent_link'] = self.remote_field.parent_link
        if isinstance(self.remote_field.model, str):
            if '.' in self.remote_field.model:
                app_label, model_name = self.remote_field.model.split('.')
                kwargs['to'] = '%s.%s' % (app_label, model_name.lower())
            else:
                kwargs['to'] = self.remote_field.model.lower()
        else:
            kwargs['to'] = self.remote_field.model._meta.label_lower
        # If swappable is True, then see if we're actually pointing to the target
        # of a swap.
        swappable_setting = self.swappable_setting
        if swappable_setting is not None:
            # If it's already a settings reference, error
            if hasattr(kwargs['to'], "setting_name"):
                if kwargs['to'].setting_name != swappable_setting:
                    raise ValueError(
                        "Cannot deconstruct a ForeignKey pointing to a model "
                        "that is swapped in place of more than one model (%s and %s)"
                        % (kwargs['to'].setting_name, swappable_setting)
                    )
            # Set it
            kwargs['to'] = SettingsReference(
                kwargs['to'],
                swappable_setting,
            )
        return name, path, args, kwargs
```
### 89 - django/db/models/fields/related.py:

Start line: 652, End line: 668

```python
class ForeignObject(RelatedField):

    @staticmethod
    def get_instance_value_for_fields(instance, fields):
        ret = []
        opts = instance._meta
        for field in fields:
            # Gotcha: in some cases (like fixture loading) a model can have
            # different values in parent_ptr_id and parent's id. So, use
            # instance.pk (that is, parent_ptr_id) when asked for instance.id.
            if field.primary_key:
                possible_parent_link = opts.get_ancestor_link(field.model)
                if (not possible_parent_link or
                        possible_parent_link.primary_key or
                        possible_parent_link.model._meta.abstract):
                    ret.append(instance.pk)
                    continue
            ret.append(getattr(instance, field.attname))
        return tuple(ret)
```
### 96 - django/db/models/fields/related.py:

Start line: 670, End line: 694

```python
class ForeignObject(RelatedField):

    def get_attname_column(self):
        attname, column = super().get_attname_column()
        return attname, None

    def get_joining_columns(self, reverse_join=False):
        source = self.reverse_related_fields if reverse_join else self.related_fields
        return tuple((lhs_field.column, rhs_field.column) for lhs_field, rhs_field in source)

    def get_reverse_joining_columns(self):
        return self.get_joining_columns(reverse_join=True)

    def get_extra_descriptor_filter(self, instance):
        """
        Return an extra filter condition for related object fetching when
        user does 'instance.fieldname', that is the extra filter is used in
        the descriptor of the field.

        The filter should be either a dict usable in .filter(**kwargs) call or
        a Q-object. The condition will be ANDed together with the relation's
        joining columns.

        A parallel method is get_extra_restriction() which is used in
        JOIN and subquery conditions.
        """
        return {}
```
### 100 - django/db/models/fields/related.py:

Start line: 892, End line: 911

```python
class ForeignKey(ForeignObject):

    def to_python(self, value):
        return self.target_field.to_python(value)

    @property
    def target_field(self):
        return self.foreign_related_fields[0]

    def get_reverse_path_info(self, filtered_relation=None):
        """Get path from the related model to this field's model."""
        opts = self.model._meta
        from_opts = self.remote_field.model._meta
        return [PathInfo(
            from_opts=from_opts,
            to_opts=opts,
            target_fields=(opts.pk,),
            join_field=self.remote_field,
            m2m=not self.unique,
            direct=False,
            filtered_relation=filtered_relation,
        )]
```
### 105 - django/db/models/fields/related.py:

Start line: 710, End line: 748

```python
class ForeignObject(RelatedField):

    def get_path_info(self, filtered_relation=None):
        """Get path from this field to the related model."""
        opts = self.remote_field.model._meta
        from_opts = self.model._meta
        return [PathInfo(
            from_opts=from_opts,
            to_opts=opts,
            target_fields=self.foreign_related_fields,
            join_field=self,
            m2m=False,
            direct=True,
            filtered_relation=filtered_relation,
        )]

    def get_reverse_path_info(self, filtered_relation=None):
        """Get path from the related model to this field's model."""
        opts = self.model._meta
        from_opts = self.remote_field.model._meta
        return [PathInfo(
            from_opts=from_opts,
            to_opts=opts,
            target_fields=(opts.pk,),
            join_field=self.remote_field,
            m2m=not self.unique,
            direct=False,
            filtered_relation=filtered_relation,
        )]

    @classmethod
    @functools.lru_cache(maxsize=None)
    def get_lookups(cls):
        bases = inspect.getmro(cls)
        bases = bases[:bases.index(ForeignObject) + 1]
        class_lookups = [parent.__dict__.get('class_lookups', {}) for parent in bases]
        return cls.merge_dicts(class_lookups)

    def contribute_to_class(self, cls, name, private_only=False, **kwargs):
        super().contribute_to_class(cls, name, private_only=private_only, **kwargs)
        setattr(cls, self.name, self.forward_related_accessor_class(self))
```
### 106 - django/db/models/fields/related.py:

Start line: 37, End line: 59

```python
def resolve_relation(scope_model, relation):
    """
    Transform relation into a model or fully-qualified model string of the form
    "app_label.ModelName", relative to scope_model.

    The relation argument can be:
      * RECURSIVE_RELATIONSHIP_CONSTANT, i.e. the string "self", in which case
        the model argument will be returned.
      * A bare model name without an app_label, in which case scope_model's
        app_label will be prepended.
      * An "app_label.ModelName" string.
      * A model class, which will be returned unchanged.
    """
    # Check for recursive relations
    if relation == RECURSIVE_RELATIONSHIP_CONSTANT:
        relation = scope_model

    # Look for an "app.Model" relation
    if isinstance(relation, str):
        if "." not in relation:
            relation = "%s.%s" % (scope_model._meta.app_label, relation)

    return relation
```
### 120 - django/db/models/fields/related.py:

Start line: 444, End line: 485

```python
class ForeignObject(RelatedField):
    """
    Abstraction of the ForeignKey relation to support multi-column relations.
    """

    # Field flags
    many_to_many = False
    many_to_one = True
    one_to_many = False
    one_to_one = False

    requires_unique_target = True
    related_accessor_class = ReverseManyToOneDescriptor
    forward_related_accessor_class = ForwardManyToOneDescriptor
    rel_class = ForeignObjectRel

    def __init__(self, to, on_delete, from_fields, to_fields, rel=None, related_name=None,
                 related_query_name=None, limit_choices_to=None, parent_link=False,
                 swappable=True, **kwargs):

        if rel is None:
            rel = self.rel_class(
                self, to,
                related_name=related_name,
                related_query_name=related_query_name,
                limit_choices_to=limit_choices_to,
                parent_link=parent_link,
                on_delete=on_delete,
            )

        super().__init__(rel=rel, **kwargs)

        self.from_fields = from_fields
        self.to_fields = to_fields
        self.swappable = swappable

    def check(self, **kwargs):
        return [
            *super().check(**kwargs),
            *self._check_to_fields_exist(),
            *self._check_unique_target(),
        ]
```
### 133 - django/db/models/fields/related.py:

Start line: 62, End line: 80

```python
def lazy_related_operation(function, model, *related_models, **kwargs):
    """
    Schedule `function` to be called once `model` and all `related_models`
    have been imported and registered with the app registry. `function` will
    be called with the newly-loaded model classes as its positional arguments,
    plus any optional keyword arguments.

    The `model` argument must be a model class. Each subsequent positional
    argument is another model, or a reference to another model - see
    `resolve_relation()` for the various forms these may take. Any relative
    references will be resolved relative to `model`.

    This is a convenience wrapper for `Apps.lazy_model_operation` - the app
    registry model used is the one found in `model._meta.apps`.
    """
    models = [model] + [resolve_relation(model, rel) for rel in related_models]
    model_keys = (make_model_tuple(m) for m in models)
    apps = model._meta.apps
    return apps.lazy_model_operation(partial(function, **kwargs), *model_keys)
```
### 161 - django/db/models/fields/related.py:

Start line: 320, End line: 341

```python
class RelatedField(FieldCacheMixin, Field):

    def deconstruct(self):
        name, path, args, kwargs = super().deconstruct()
        if self.remote_field.limit_choices_to:
            kwargs['limit_choices_to'] = self.remote_field.limit_choices_to
        if self.remote_field.related_name is not None:
            kwargs['related_name'] = self.remote_field.related_name
        if self.remote_field.related_query_name is not None:
            kwargs['related_query_name'] = self.remote_field.related_query_name
        return name, path, args, kwargs

    def get_forward_related_filter(self, obj):
        """
        Return the keyword arguments that when supplied to
        self.model.object.filter(), would select all instances related through
        this field to the remote obj. This is used to build the querysets
        returned by related descriptors. obj is an instance of
        self.related_field.model.
        """
        return {
            '%s__%s' % (self.name, rh_field.name): getattr(obj, rh_field.attname)
            for _, rh_field in self.related_fields
        }
```
### 192 - django/db/models/fields/related.py:

Start line: 696, End line: 708

```python
class ForeignObject(RelatedField):

    def get_extra_restriction(self, where_class, alias, related_alias):
        """
        Return a pair condition used for joining and subquery pushdown. The
        condition is something that responds to as_sql(compiler, connection)
        method.

        Note that currently referring both the 'alias' and 'related_alias'
        will not work in some conditions, like subquery pushdown.

        A parallel method is get_extra_descriptor_filter() which is used in
        instance.fieldname related object fetching.
        """
        return None
```
