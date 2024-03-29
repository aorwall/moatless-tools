# django__django-16983

| **django/django** | `ddb6506618ea52c6b20e97eefad03ed847a1e3de` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 144 |
| **Any found context length** | 144 |
| **Avg pos** | 1.0 |
| **Min pos** | 1 |
| **Max pos** | 1 |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/contrib/admin/checks.py b/django/contrib/admin/checks.py
--- a/django/contrib/admin/checks.py
+++ b/django/contrib/admin/checks.py
@@ -533,6 +533,16 @@ def _check_filter_item(self, obj, field_name, label):
                 return must_be(
                     "a many-to-many field", option=label, obj=obj, id="admin.E020"
                 )
+            elif not field.remote_field.through._meta.auto_created:
+                return [
+                    checks.Error(
+                        f"The value of '{label}' cannot include the ManyToManyField "
+                        f"'{field_name}', because that field manually specifies a "
+                        f"relationship model.",
+                        obj=obj.__class__,
+                        id="admin.E013",
+                    )
+                ]
             else:
                 return []
 

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/contrib/admin/checks.py | 536 | 536 | 1 | 1 | 144


## Problem Statement

```
Add system check for filter_horizontal/filter_vertical on ManyToManyFields with intermediary models.
Description
	
Hi team,
I'm a huge fan of Django and have been using it since 0.95 but I stumbled over this one.
Neither of
​https://docs.djangoproject.com/en/4.1/ref/contrib/admin/#django.contrib.admin.ModelAdmin.filter_horizontal and 
​https://docs.djangoproject.com/en/4.1/ref/contrib/admin/#django.contrib.admin.ModelAdmin.filter_vertical
call out the requirement to not use 
ManyToManyField(through="")
In the same way:
​https://docs.djangoproject.com/en/4.1/ref/models/fields/#django.db.models.ManyToManyField.through
doesn't call out the consequence that filter_horizontal and filter_vertical will stop working if one goes down the pathway of:
ManyToManyField(through="")
I just wasted half a day chasing this down.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 django/contrib/admin/checks.py** | 521 | 537| 144 | 144 | 9583 | 
| 2 | **1 django/contrib/admin/checks.py** | 505 | 519| 122 | 266 | 9583 | 
| 3 | 2 django/db/models/fields/related.py | 1463 | 1592| 984 | 1250 | 24287 | 
| 4 | 2 django/db/models/fields/related.py | 1594 | 1691| 655 | 1905 | 24287 | 
| 5 | 2 django/db/models/fields/related.py | 1693 | 1743| 431 | 2336 | 24287 | 
| 6 | 2 django/db/models/fields/related.py | 1423 | 1461| 213 | 2549 | 24287 | 
| 7 | 3 django/contrib/admin/options.py | 2459 | 2486| 254 | 2803 | 43669 | 
| 8 | 4 django/db/models/fields/related_descriptors.py | 1066 | 1087| 203 | 3006 | 55327 | 
| 9 | 4 django/contrib/admin/options.py | 441 | 506| 550 | 3556 | 55327 | 
| 10 | 4 django/db/models/fields/related.py | 1894 | 1941| 505 | 4061 | 55327 | 
| 11 | 4 django/db/models/fields/related_descriptors.py | 1046 | 1064| 199 | 4260 | 55327 | 
| 12 | 4 django/db/models/fields/related_descriptors.py | 1238 | 1300| 544 | 4804 | 55327 | 
| 13 | **4 django/contrib/admin/checks.py** | 790 | 808| 183 | 4987 | 55327 | 
| 14 | 4 django/db/models/fields/related_descriptors.py | 665 | 703| 354 | 5341 | 55327 | 
| 15 | 4 django/db/models/fields/related.py | 1324 | 1421| 574 | 5915 | 55327 | 
| 16 | 4 django/db/models/fields/related_descriptors.py | 1173 | 1204| 228 | 6143 | 55327 | 
| 17 | 5 django/contrib/admin/filters.py | 220 | 280| 541 | 6684 | 61053 | 
| 18 | 5 django/db/models/fields/related_descriptors.py | 1457 | 1507| 407 | 7091 | 61053 | 
| 19 | **5 django/contrib/admin/checks.py** | 959 | 983| 197 | 7288 | 61053 | 
| 20 | **5 django/contrib/admin/checks.py** | 217 | 264| 333 | 7621 | 61053 | 
| 21 | **5 django/contrib/admin/checks.py** | 1248 | 1277| 196 | 7817 | 61053 | 
| 22 | **5 django/contrib/admin/checks.py** | 176 | 192| 155 | 7972 | 61053 | 
| 23 | **5 django/contrib/admin/checks.py** | 480 | 503| 188 | 8160 | 61053 | 
| 24 | 5 django/db/models/fields/related_descriptors.py | 1133 | 1151| 174 | 8334 | 61053 | 
| 25 | 5 django/db/models/fields/related_descriptors.py | 1153 | 1171| 155 | 8489 | 61053 | 
| 26 | 5 django/db/models/fields/related.py | 1745 | 1783| 413 | 8902 | 61053 | 
| 27 | **5 django/contrib/admin/checks.py** | 1149 | 1185| 252 | 9154 | 61053 | 
| 28 | 5 django/contrib/admin/filters.py | 555 | 592| 371 | 9525 | 61053 | 
| 29 | 5 django/db/models/fields/related_descriptors.py | 788 | 893| 808 | 10333 | 61053 | 
| 30 | 5 django/db/models/fields/related_descriptors.py | 1206 | 1236| 260 | 10593 | 61053 | 
| 31 | 5 django/db/models/fields/related.py | 1943 | 1970| 305 | 10898 | 61053 | 
| 32 | 5 django/db/models/fields/related.py | 1785 | 1814| 331 | 11229 | 61053 | 
| 33 | 5 django/contrib/admin/filters.py | 1 | 21| 144 | 11373 | 61053 | 
| 34 | 6 django/contrib/auth/admin.py | 28 | 40| 130 | 11503 | 62860 | 
| 35 | 6 django/contrib/admin/filters.py | 319 | 339| 215 | 11718 | 62860 | 
| 36 | **6 django/contrib/admin/checks.py** | 1235 | 1246| 116 | 11834 | 62860 | 
| 37 | 7 django/db/models/base.py | 1691 | 1725| 252 | 12086 | 81775 | 
| 38 | **7 django/contrib/admin/checks.py** | 1095 | 1147| 430 | 12516 | 81775 | 
| 39 | 7 django/db/models/fields/related_descriptors.py | 1386 | 1455| 526 | 13042 | 81775 | 
| 40 | 7 django/db/models/fields/related_descriptors.py | 730 | 749| 231 | 13273 | 81775 | 
| 41 | 7 django/contrib/admin/filters.py | 632 | 645| 119 | 13392 | 81775 | 
| 42 | 8 django/contrib/admin/__init__.py | 1 | 53| 292 | 13684 | 82067 | 
| 43 | **8 django/contrib/admin/checks.py** | 284 | 312| 218 | 13902 | 82067 | 
| 44 | 8 django/db/models/fields/related_descriptors.py | 978 | 1044| 592 | 14494 | 82067 | 
| 45 | 8 django/db/models/fields/related_descriptors.py | 1334 | 1351| 154 | 14648 | 82067 | 
| 46 | 8 django/db/models/base.py | 1660 | 1689| 209 | 14857 | 82067 | 
| 47 | 8 django/db/models/fields/related_descriptors.py | 1089 | 1131| 381 | 15238 | 82067 | 
| 48 | **8 django/contrib/admin/checks.py** | 841 | 877| 268 | 15506 | 82067 | 
| 49 | 8 django/contrib/admin/options.py | 508 | 552| 348 | 15854 | 82067 | 
| 50 | 9 django/contrib/contenttypes/admin.py | 1 | 88| 585 | 16439 | 83072 | 
| 51 | **9 django/contrib/admin/checks.py** | 266 | 282| 138 | 16577 | 83072 | 
| 52 | 9 django/db/models/fields/related_descriptors.py | 895 | 928| 279 | 16856 | 83072 | 
| 53 | 9 django/db/models/fields/related.py | 156 | 187| 209 | 17065 | 83072 | 
| 54 | **9 django/contrib/admin/checks.py** | 985 | 1042| 457 | 17522 | 83072 | 
| 55 | 9 django/db/models/fields/related_descriptors.py | 705 | 728| 227 | 17749 | 83072 | 
| 56 | **9 django/contrib/admin/checks.py** | 194 | 215| 141 | 17890 | 83072 | 
| 57 | 9 django/db/models/fields/related.py | 304 | 341| 296 | 18186 | 83072 | 
| 58 | **9 django/contrib/admin/checks.py** | 430 | 458| 224 | 18410 | 83072 | 
| 59 | 10 django/contrib/admin/views/main.py | 175 | 270| 828 | 19238 | 87989 | 
| 60 | 10 django/db/models/fields/related_descriptors.py | 632 | 663| 243 | 19481 | 87989 | 
| 61 | 11 django/core/checks/model_checks.py | 187 | 228| 345 | 19826 | 89799 | 
| 62 | 11 django/db/models/fields/related.py | 1105 | 1122| 133 | 19959 | 89799 | 
| 63 | 11 django/db/models/fields/related_descriptors.py | 751 | 786| 268 | 20227 | 89799 | 
| 64 | 11 django/db/models/fields/related.py | 1972 | 2006| 266 | 20493 | 89799 | 
| 65 | **11 django/contrib/admin/checks.py** | 826 | 839| 117 | 20610 | 89799 | 
| 66 | 11 django/contrib/admin/options.py | 2488 | 2523| 315 | 20925 | 89799 | 
| 67 | 11 django/db/models/fields/related.py | 582 | 602| 138 | 21063 | 89799 | 
| 68 | **11 django/contrib/admin/checks.py** | 879 | 892| 121 | 21184 | 89799 | 
| 69 | 11 django/db/models/fields/related.py | 210 | 226| 142 | 21326 | 89799 | 
| 70 | 12 django/contrib/sites/managers.py | 1 | 46| 277 | 21603 | 90194 | 
| 71 | 12 django/db/models/base.py | 2051 | 2104| 359 | 21962 | 90194 | 
| 72 | 12 django/db/models/fields/related.py | 1267 | 1321| 421 | 22383 | 90194 | 
| 73 | 12 django/db/models/fields/related_descriptors.py | 1353 | 1384| 348 | 22731 | 90194 | 
| 74 | **12 django/contrib/admin/checks.py** | 704 | 740| 301 | 23032 | 90194 | 
| 75 | 12 django/db/models/fields/related_descriptors.py | 1302 | 1332| 281 | 23313 | 90194 | 
| 76 | 12 django/contrib/admin/options.py | 1 | 119| 796 | 24109 | 90194 | 
| 77 | **12 django/contrib/admin/checks.py** | 1044 | 1075| 229 | 24338 | 90194 | 
| 78 | 12 django/contrib/admin/views/main.py | 271 | 287| 158 | 24496 | 90194 | 
| 79 | 12 django/db/models/fields/related.py | 1816 | 1841| 222 | 24718 | 90194 | 
| 80 | 12 django/contrib/admin/options.py | 121 | 154| 235 | 24953 | 90194 | 
| 81 | 12 django/db/models/fields/related_descriptors.py | 406 | 427| 158 | 25111 | 90194 | 
| 82 | 12 django/contrib/admin/filters.py | 282 | 316| 311 | 25422 | 90194 | 
| 83 | 12 django/db/models/fields/related.py | 228 | 303| 696 | 26118 | 90194 | 
| 84 | 12 django/contrib/admin/views/main.py | 1 | 64| 383 | 26501 | 90194 | 
| 85 | 12 django/db/models/fields/related.py | 1 | 42| 267 | 26768 | 90194 | 
| 86 | 12 django/core/checks/model_checks.py | 135 | 159| 268 | 27036 | 90194 | 
| 87 | **12 django/contrib/admin/checks.py** | 1279 | 1323| 343 | 27379 | 90194 | 
| 88 | 12 django/db/models/base.py | 1540 | 1576| 288 | 27667 | 90194 | 
| 89 | **12 django/contrib/admin/checks.py** | 742 | 759| 139 | 27806 | 90194 | 
| 90 | 12 django/contrib/admin/options.py | 1770 | 1872| 780 | 28586 | 90194 | 
| 91 | 12 django/contrib/admin/filters.py | 169 | 196| 222 | 28808 | 90194 | 
| 92 | **12 django/contrib/admin/checks.py** | 673 | 702| 241 | 29049 | 90194 | 
| 93 | **12 django/contrib/admin/checks.py** | 314 | 346| 233 | 29282 | 90194 | 
| 94 | **12 django/contrib/admin/checks.py** | 894 | 932| 269 | 29551 | 90194 | 
| 95 | 13 django/db/models/fields/__init__.py | 481 | 508| 185 | 29736 | 109564 | 
| 96 | 14 django/contrib/contenttypes/fields.py | 25 | 114| 560 | 30296 | 115392 | 
| 97 | 14 django/db/models/fields/__init__.py | 1334 | 1371| 245 | 30541 | 115392 | 
| 98 | 15 django/db/backends/base/schema.py | 52 | 72| 152 | 30693 | 130538 | 
| 99 | 15 django/db/models/fields/related.py | 1036 | 1072| 261 | 30954 | 130538 | 
| 100 | 15 django/contrib/admin/filters.py | 458 | 533| 631 | 31585 | 130538 | 
| 101 | **15 django/contrib/admin/checks.py** | 761 | 787| 168 | 31753 | 130538 | 
| 102 | 15 django/core/checks/model_checks.py | 1 | 90| 671 | 32424 | 130538 | 
| 103 | 15 django/contrib/admin/filters.py | 388 | 417| 251 | 32675 | 130538 | 
| 104 | 16 django/db/models/__init__.py | 1 | 116| 682 | 33357 | 131220 | 
| 105 | 16 django/db/models/fields/related.py | 189 | 208| 155 | 33512 | 131220 | 
| 106 | **16 django/contrib/admin/checks.py** | 348 | 367| 146 | 33658 | 131220 | 
| 107 | 16 django/contrib/admin/filters.py | 24 | 87| 418 | 34076 | 131220 | 
| 108 | 16 django/contrib/admin/options.py | 1104 | 1123| 131 | 34207 | 131220 | 
| 109 | 16 django/db/models/fields/related_descriptors.py | 155 | 199| 418 | 34625 | 131220 | 
| 110 | **16 django/contrib/admin/checks.py** | 1326 | 1355| 176 | 34801 | 131220 | 
| 111 | **16 django/contrib/admin/checks.py** | 460 | 478| 137 | 34938 | 131220 | 
| 112 | 16 django/db/models/fields/related.py | 128 | 154| 171 | 35109 | 131220 | 
| 113 | 16 django/contrib/admin/filters.py | 648 | 690| 371 | 35480 | 131220 | 
| 114 | 17 django/contrib/admin/sites.py | 81 | 97| 132 | 35612 | 135721 | 
| 115 | 17 django/core/checks/model_checks.py | 93 | 116| 170 | 35782 | 135721 | 
| 116 | **17 django/contrib/admin/checks.py** | 580 | 608| 195 | 35977 | 135721 | 
| 117 | 17 django/core/checks/model_checks.py | 161 | 185| 267 | 36244 | 135721 | 
| 118 | 17 django/db/models/base.py | 1865 | 1900| 246 | 36490 | 135721 | 
| 119 | 18 django/contrib/auth/checks.py | 1 | 104| 728 | 37218 | 137237 | 
| 120 | 19 django/contrib/contenttypes/checks.py | 1 | 25| 130 | 37348 | 137498 | 
| 121 | 19 django/contrib/admin/options.py | 364 | 440| 531 | 37879 | 137498 | 
| 122 | 19 django/db/models/fields/related.py | 733 | 755| 172 | 38051 | 137498 | 
| 123 | 19 django/contrib/admin/options.py | 1015 | 1029| 125 | 38176 | 137498 | 
| 124 | 19 django/db/models/fields/__init__.py | 424 | 456| 210 | 38386 | 137498 | 
| 125 | 19 django/db/models/fields/related.py | 707 | 731| 210 | 38596 | 137498 | 
| 126 | 19 django/db/backends/base/schema.py | 1387 | 1431| 412 | 39008 | 137498 | 
| 127 | 19 django/db/models/fields/related.py | 1074 | 1103| 217 | 39225 | 137498 | 
| 128 | 19 django/db/models/fields/related.py | 1179 | 1214| 267 | 39492 | 137498 | 
| 129 | **19 django/contrib/admin/checks.py** | 610 | 628| 164 | 39656 | 137498 | 
| 130 | **19 django/contrib/admin/checks.py** | 1187 | 1212| 182 | 39838 | 137498 | 
| 131 | 20 django/db/models/options.py | 1 | 57| 350 | 40188 | 145141 | 
| 132 | 20 django/contrib/admin/views/main.py | 595 | 627| 227 | 40415 | 145141 | 
| 133 | 20 django/contrib/auth/checks.py | 107 | 221| 786 | 41201 | 145141 | 
| 134 | **20 django/contrib/admin/checks.py** | 55 | 173| 772 | 41973 | 145141 | 
| 135 | 21 django/db/backends/sqlite3/schema.py | 505 | 567| 472 | 42445 | 149989 | 
| 136 | 21 django/db/models/base.py | 1746 | 1814| 603 | 43048 | 149989 | 
| 137 | 21 django/db/models/base.py | 2106 | 2211| 733 | 43781 | 149989 | 
| 138 | 21 django/db/models/base.py | 1361 | 1390| 294 | 44075 | 149989 | 
| 139 | 21 django/db/models/fields/__init__.py | 510 | 535| 198 | 44273 | 149989 | 
| 140 | **21 django/contrib/admin/checks.py** | 934 | 957| 195 | 44468 | 149989 | 
| 141 | **21 django/contrib/admin/checks.py** | 413 | 428| 145 | 44613 | 149989 | 
| 142 | 21 django/contrib/admin/filters.py | 341 | 352| 112 | 44725 | 149989 | 
| 143 | **21 django/contrib/admin/checks.py** | 539 | 554| 136 | 44861 | 149989 | 
| 144 | 22 django/contrib/gis/admin/__init__.py | 1 | 30| 130 | 44991 | 150119 | 
| 145 | 22 django/contrib/admin/options.py | 1873 | 1904| 303 | 45294 | 150119 | 
| 146 | **22 django/contrib/admin/checks.py** | 810 | 824| 127 | 45421 | 150119 | 
| 147 | 22 django/contrib/admin/filters.py | 419 | 455| 302 | 45723 | 150119 | 
| 148 | 23 django/db/models/constraints.py | 93 | 134| 358 | 46081 | 153632 | 
| 149 | **23 django/contrib/admin/checks.py** | 1077 | 1093| 140 | 46221 | 153632 | 
| 150 | 23 django/db/models/constraints.py | 136 | 152| 134 | 46355 | 153632 | 
| 151 | 23 django/db/models/fields/related.py | 889 | 916| 237 | 46592 | 153632 | 
| 152 | 23 django/db/models/base.py | 1 | 66| 361 | 46953 | 153632 | 
| 153 | 23 django/db/models/fields/related.py | 837 | 887| 373 | 47326 | 153632 | 
| 154 | 23 django/contrib/contenttypes/fields.py | 756 | 804| 407 | 47733 | 153632 | 
| 155 | 24 django/db/models/fields/related_lookups.py | 141 | 158| 216 | 47949 | 155169 | 
| 156 | 24 django/db/models/fields/related_descriptors.py | 575 | 629| 338 | 48287 | 155169 | 
| 157 | 24 django/contrib/contenttypes/fields.py | 116 | 161| 322 | 48609 | 155169 | 
| 158 | 24 django/db/models/fields/related.py | 381 | 402| 219 | 48828 | 155169 | 
| 159 | 25 django/contrib/admin/utils.py | 311 | 337| 189 | 49017 | 159559 | 
| 160 | **25 django/contrib/admin/checks.py** | 657 | 671| 141 | 49158 | 159559 | 
| 161 | 26 django/db/migrations/autodetector.py | 807 | 902| 712 | 49870 | 173334 | 
| 162 | 27 django/contrib/admindocs/views.py | 102 | 138| 301 | 50171 | 176822 | 
| 163 | 27 django/db/models/fields/related.py | 404 | 422| 161 | 50332 | 176822 | 
| 164 | 27 django/contrib/contenttypes/fields.py | 366 | 392| 181 | 50513 | 176822 | 
| 165 | 27 django/contrib/contenttypes/fields.py | 1 | 22| 171 | 50684 | 176822 | 
| 166 | 27 django/contrib/admin/filters.py | 198 | 217| 192 | 50876 | 176822 | 
| 167 | 28 django/db/models/query_utils.py | 367 | 393| 289 | 51165 | 180135 | 
| 168 | 28 django/contrib/admin/options.py | 1531 | 1557| 233 | 51398 | 180135 | 
| 169 | 28 django/contrib/admin/options.py | 290 | 338| 397 | 51795 | 180135 | 
| 170 | 28 django/contrib/admin/filters.py | 354 | 385| 299 | 52094 | 180135 | 
| 171 | 28 django/contrib/contenttypes/fields.py | 474 | 501| 258 | 52352 | 180135 | 
| 172 | 28 django/db/models/fields/related_descriptors.py | 931 | 975| 315 | 52667 | 180135 | 
| 173 | 28 django/db/models/fields/related.py | 1010 | 1034| 176 | 52843 | 180135 | 
| 174 | **28 django/contrib/admin/checks.py** | 556 | 578| 194 | 53037 | 180135 | 
| 175 | 29 django/forms/models.py | 1361 | 1396| 225 | 53262 | 192408 | 
| 176 | 29 django/db/models/base.py | 2506 | 2558| 341 | 53603 | 192408 | 
| 177 | 29 django/db/models/options.py | 588 | 604| 115 | 53718 | 192408 | 
| 178 | 29 django/db/models/fields/related.py | 343 | 379| 294 | 54012 | 192408 | 
| 179 | 29 django/db/models/base.py | 1901 | 1928| 191 | 54203 | 192408 | 
| 180 | 29 django/db/models/base.py | 2213 | 2294| 592 | 54795 | 192408 | 
| 181 | 29 django/contrib/admindocs/views.py | 213 | 297| 615 | 55410 | 192408 | 
| 182 | 29 django/db/models/fields/related_lookups.py | 65 | 98| 369 | 55779 | 192408 | 
| 183 | 29 django/db/migrations/autodetector.py | 234 | 265| 256 | 56035 | 192408 | 
| 184 | **29 django/contrib/admin/checks.py** | 1 | 52| 321 | 56356 | 192408 | 


### Hint

```
Neither of ​https://docs.djangoproject.com/en/4.1/ref/contrib/admin/#django.contrib.admin.ModelAdmin.filter_horizontal and ​https://docs.djangoproject.com/en/4.1/ref/contrib/admin/#django.contrib.admin.ModelAdmin.filter_vertical call out the requirement to not use ManyToManyField(through="") There is a separate section in the same docs that describes ​Working with many-to-many intermediary models. I don't think it is necessary to cross-refer this section in all places where ManyToManyField is mentioned (see also #12203.) In the same way: ​https://docs.djangoproject.com/en/4.1/ref/models/fields/#django.db.models.ManyToManyField.through doesn't call out the consequence that filter_horizontal and filter_vertical will stop working if one goes down the pathway of: ManyToManyField(through="") Models docs are not the right place to describe how contrib apps work.
What do you think about raising admin.E013 in this case? "fields[n]/fieldsets[n][m]/filter_vertical[n]/filter_horizontal[n] cannot include the ManyToManyField <field name>, because that field manually specifies a relationship model."
Thanks. Sounds like a good outcome.
Replying to David Pratten: Thanks. Sounds like a good outcome. Would you like to prepare a patch via GitHub PR? The following should work: django/contrib/admin/checks.py diff --git a/django/contrib/admin/checks.py b/django/contrib/admin/checks.py index 27537d9614..a844b3f16f 100644 a b class BaseModelAdminChecks: 533533 return must_be( 534534 "a many-to-many field", option=label, obj=obj, id="admin.E020" 535535 ) 536 elif not field.remote_field.through._meta.auto_created: 537 return [ 538 checks.Error( 539 f"The value of '{label}' cannot include the ManyToManyField " 540 f"'{field_name}', because that field manually specifies a " 541 f"relationship model.", 542 obj=obj.__class__, 543 id="admin.E013", 544 ) 545 ] 536546 else: 537547 return [] 538548 Tests and ​docs changes (in the admin.E013 description) are also required.
Ok I'll take this up.
Replying to Mariusz Felisiak: I'm happy to work through this, but it won't be quick. Are we redefining admin.E013 there seems to already be a description of this error? Could you direct me to an explanation of where the documentation for the errors is held and how it is updated? Could you direct me to an explanation of how to add a test case? Thanks Replying to David Pratten: Thanks. Sounds like a good outcome. Would you like to prepare a patch via GitHub PR? The following should work: django/contrib/admin/checks.py diff --git a/django/contrib/admin/checks.py b/django/contrib/admin/checks.py index 27537d9614..a844b3f16f 100644 a b class BaseModelAdminChecks: 533533 return must_be( 534534 "a many-to-many field", option=label, obj=obj, id="admin.E020" 535535 ) 536 elif not field.remote_field.through._meta.auto_created: 537 return [ 538 checks.Error( 539 f"The value of '{label}' cannot include the ManyToManyField " 540 f"'{field_name}', because that field manually specifies a " 541 f"relationship model.", 542 obj=obj.__class__, 543 id="admin.E013", 544 ) 545 ] 536546 else: 537547 return [] 538548 Tests and ​docs changes (in the admin.E013 description) are also required.
Replying to David Pratten: Are we redefining admin.E013 there seems to already be a description of this error? Could you direct me to an explanation of where the documentation for the errors is held and how it is updated? We want to add filter_vertical[n] and filter_horizontal[n] to the existing error admin.E013 that is documented in ​docs/ref/checks.txt, so we need to update the message in docs to the: "fields[n]/filter_horizontal[n]/filter_vertical[n]/fieldsets[n][m] cannot include the ManyToManyField <field name>, because that field manually specifies a relationship model." Docs are wrapped at 79 chars. Could you direct me to an explanation of how to add a test case? I would add test methods to the tests.modeladmin.test_checks.FilterHorizontalCheckTests and tests.modeladmin.test_checks.FilterVerticalCheckTests.
Hello David, would you have time to work on this? Thanks!
Since David seems to be busy, I'll try to take this up.
```

## Patch

```diff
diff --git a/django/contrib/admin/checks.py b/django/contrib/admin/checks.py
--- a/django/contrib/admin/checks.py
+++ b/django/contrib/admin/checks.py
@@ -533,6 +533,16 @@ def _check_filter_item(self, obj, field_name, label):
                 return must_be(
                     "a many-to-many field", option=label, obj=obj, id="admin.E020"
                 )
+            elif not field.remote_field.through._meta.auto_created:
+                return [
+                    checks.Error(
+                        f"The value of '{label}' cannot include the ManyToManyField "
+                        f"'{field_name}', because that field manually specifies a "
+                        f"relationship model.",
+                        obj=obj.__class__,
+                        id="admin.E013",
+                    )
+                ]
             else:
                 return []
 

```

## Test Patch

```diff
diff --git a/tests/modeladmin/test_checks.py b/tests/modeladmin/test_checks.py
--- a/tests/modeladmin/test_checks.py
+++ b/tests/modeladmin/test_checks.py
@@ -4,10 +4,11 @@
 from django.contrib.admin.options import VERTICAL, ModelAdmin, TabularInline
 from django.contrib.admin.sites import AdminSite
 from django.core.checks import Error
-from django.db.models import CASCADE, F, Field, ForeignKey, Model
+from django.db.models import CASCADE, F, Field, ForeignKey, ManyToManyField, Model
 from django.db.models.functions import Upper
 from django.forms.models import BaseModelFormSet
 from django.test import SimpleTestCase
+from django.test.utils import isolate_apps
 
 from .models import Band, Song, User, ValidationTestInlineModel, ValidationTestModel
 
@@ -321,6 +322,26 @@ class TestModelAdmin(ModelAdmin):
             "admin.E020",
         )
 
+    @isolate_apps("modeladmin")
+    def test_invalid_m2m_field_with_through(self):
+        class Artist(Model):
+            bands = ManyToManyField("Band", through="BandArtist")
+
+        class BandArtist(Model):
+            artist = ForeignKey("Artist", on_delete=CASCADE)
+            band = ForeignKey("Band", on_delete=CASCADE)
+
+        class TestModelAdmin(ModelAdmin):
+            filter_vertical = ["bands"]
+
+        self.assertIsInvalid(
+            TestModelAdmin,
+            Artist,
+            "The value of 'filter_vertical[0]' cannot include the ManyToManyField "
+            "'bands', because that field manually specifies a relationship model.",
+            "admin.E013",
+        )
+
     def test_valid_case(self):
         class TestModelAdmin(ModelAdmin):
             filter_vertical = ("users",)
@@ -363,6 +384,26 @@ class TestModelAdmin(ModelAdmin):
             "admin.E020",
         )
 
+    @isolate_apps("modeladmin")
+    def test_invalid_m2m_field_with_through(self):
+        class Artist(Model):
+            bands = ManyToManyField("Band", through="BandArtist")
+
+        class BandArtist(Model):
+            artist = ForeignKey("Artist", on_delete=CASCADE)
+            band = ForeignKey("Band", on_delete=CASCADE)
+
+        class TestModelAdmin(ModelAdmin):
+            filter_horizontal = ["bands"]
+
+        self.assertIsInvalid(
+            TestModelAdmin,
+            Artist,
+            "The value of 'filter_horizontal[0]' cannot include the ManyToManyField "
+            "'bands', because that field manually specifies a relationship model.",
+            "admin.E013",
+        )
+
     def test_valid_case(self):
         class TestModelAdmin(ModelAdmin):
             filter_horizontal = ("users",)

```


## Code snippets

### 1 - django/contrib/admin/checks.py:

Start line: 521, End line: 537

```python
class BaseModelAdminChecks:

    def _check_filter_item(self, obj, field_name, label):
        """Check one item of `filter_vertical` or `filter_horizontal`, i.e.
        check that given field exists and is a ManyToManyField."""

        try:
            field = obj.model._meta.get_field(field_name)
        except FieldDoesNotExist:
            return refer_to_missing_field(
                field=field_name, option=label, obj=obj, id="admin.E019"
            )
        else:
            if not field.many_to_many:
                return must_be(
                    "a many-to-many field", option=label, obj=obj, id="admin.E020"
                )
            else:
                return []
```
### 2 - django/contrib/admin/checks.py:

Start line: 505, End line: 519

```python
class BaseModelAdminChecks:

    def _check_filter_horizontal(self, obj):
        """Check that filter_horizontal is a sequence of field names."""
        if not isinstance(obj.filter_horizontal, (list, tuple)):
            return must_be(
                "a list or tuple", option="filter_horizontal", obj=obj, id="admin.E018"
            )
        else:
            return list(
                chain.from_iterable(
                    self._check_filter_item(
                        obj, field_name, "filter_horizontal[%d]" % index
                    )
                    for index, field_name in enumerate(obj.filter_horizontal)
                )
            )
```
### 3 - django/db/models/fields/related.py:

Start line: 1463, End line: 1592

```python
class ManyToManyField(RelatedField):

    def _check_relationship_model(self, from_model=None, **kwargs):
        if hasattr(self.remote_field.through, "_meta"):
            qualified_model_name = "%s.%s" % (
                self.remote_field.through._meta.app_label,
                self.remote_field.through.__name__,
            )
        else:
            qualified_model_name = self.remote_field.through

        errors = []

        if self.remote_field.through not in self.opts.apps.get_models(
            include_auto_created=True
        ):
            # The relationship model is not installed.
            errors.append(
                checks.Error(
                    "Field specifies a many-to-many relation through model "
                    "'%s', which has not been installed." % qualified_model_name,
                    obj=self,
                    id="fields.E331",
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
                    from_model == getattr(field.remote_field, "model", None)
                    for field in self.remote_field.through._meta.fields
                )

                if seen_self > 2 and not self.remote_field.through_fields:
                    errors.append(
                        checks.Error(
                            "The model is used as an intermediate model by "
                            "'%s', but it has more than two foreign keys "
                            "to '%s', which is ambiguous. You must specify "
                            "which two foreign keys Django should use via the "
                            "through_fields keyword argument."
                            % (self, from_model_name),
                            hint=(
                                "Use through_fields to specify which two foreign keys "
                                "Django should use."
                            ),
                            obj=self.remote_field.through,
                            id="fields.E333",
                        )
                    )

            else:
                # Count foreign keys in relationship model
                seen_from = sum(
                    from_model == getattr(field.remote_field, "model", None)
                    for field in self.remote_field.through._meta.fields
                )
                seen_to = sum(
                    to_model == getattr(field.remote_field, "model", None)
                    for field in self.remote_field.through._meta.fields
                )

                if seen_from > 1 and not self.remote_field.through_fields:
                    errors.append(
                        checks.Error(
                            (
                                "The model is used as an intermediate model by "
                                "'%s', but it has more than one foreign key "
                                "from '%s', which is ambiguous. You must specify "
                                "which foreign key Django should use via the "
                                "through_fields keyword argument."
                            )
                            % (self, from_model_name),
                            hint=(
                                "If you want to create a recursive relationship, "
                                'use ManyToManyField("%s", through="%s").'
                            )
                            % (
                                RECURSIVE_RELATIONSHIP_CONSTANT,
                                relationship_model_name,
                            ),
                            obj=self,
                            id="fields.E334",
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
                                "If you want to create a recursive relationship, "
                                'use ManyToManyField("%s", through="%s").'
                            )
                            % (
                                RECURSIVE_RELATIONSHIP_CONSTANT,
                                relationship_model_name,
                            ),
                            obj=self,
                            id="fields.E335",
                        )
                    )

                if seen_from == 0 or seen_to == 0:
                    errors.append(
                        checks.Error(
                            "The model is used as an intermediate model by "
                            "'%s', but it does not have a foreign key to '%s' or '%s'."
                            % (self, from_model_name, to_model_name),
                            obj=self.remote_field.through,
                            id="fields.E336",
                        )
                    )
        # ... other code
```
### 4 - django/db/models/fields/related.py:

Start line: 1594, End line: 1691

```python
class ManyToManyField(RelatedField):

    def _check_relationship_model(self, from_model=None, **kwargs):

        # Validate `through_fields`.
        if self.remote_field.through_fields is not None:
            # Validate that we're given an iterable of at least two items
            # and that none of them is "falsy".
            if not (
                len(self.remote_field.through_fields) >= 2
                and self.remote_field.through_fields[0]
                and self.remote_field.through_fields[1]
            ):
                errors.append(
                    checks.Error(
                        "Field specifies 'through_fields' but does not provide "
                        "the names of the two link fields that should be used "
                        "for the relation through model '%s'." % qualified_model_name,
                        hint=(
                            "Make sure you specify 'through_fields' as "
                            "through_fields=('field1', 'field2')"
                        ),
                        obj=self,
                        id="fields.E337",
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

                source, through, target = (
                    from_model,
                    self.remote_field.through,
                    self.remote_field.model,
                )
                source_field_name, target_field_name = self.remote_field.through_fields[
                    :2
                ]

                for field_name, related_model in (
                    (source_field_name, source),
                    (target_field_name, target),
                ):
                    possible_field_names = []
                    for f in through._meta.fields:
                        if (
                            hasattr(f, "remote_field")
                            and getattr(f.remote_field, "model", None) == related_model
                        ):
                            possible_field_names.append(f.name)
                    if possible_field_names:
                        hint = (
                            "Did you mean one of the following foreign keys to '%s': "
                            "%s?"
                            % (
                                related_model._meta.object_name,
                                ", ".join(possible_field_names),
                            )
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
                                id="fields.E338",
                            )
                        )
                    else:
                        if not (
                            hasattr(field, "remote_field")
                            and getattr(field.remote_field, "model", None)
                            == related_model
                        ):
                            errors.append(
                                checks.Error(
                                    "'%s.%s' is not a foreign key to '%s'."
                                    % (
                                        through._meta.object_name,
                                        field_name,
                                        related_model._meta.object_name,
                                    ),
                                    hint=hint,
                                    obj=self,
                                    id="fields.E339",
                                )
                            )

        return errors
```
### 5 - django/db/models/fields/related.py:

Start line: 1693, End line: 1743

```python
class ManyToManyField(RelatedField):

    def _check_table_uniqueness(self, **kwargs):
        if (
            isinstance(self.remote_field.through, str)
            or not self.remote_field.through._meta.managed
        ):
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
        if (
            model
            and model._meta.concrete_model
            != self.remote_field.through._meta.concrete_model
        ):
            if model._meta.auto_created:

                def _get_field_name(model):
                    for field in model._meta.auto_created._meta.many_to_many:
                        if field.remote_field.through is model:
                            return field.name

                opts = model._meta.auto_created._meta
                clashing_obj = "%s.%s" % (opts.label, _get_field_name(model))
            else:
                clashing_obj = model._meta.label
            if settings.DATABASE_ROUTERS:
                error_class, error_id = checks.Warning, "fields.W344"
                error_hint = (
                    "You have configured settings.DATABASE_ROUTERS. Verify "
                    "that the table of %r is correctly routed to a separate "
                    "database." % clashing_obj
                )
            else:
                error_class, error_id = checks.Error, "fields.E340"
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
### 6 - django/db/models/fields/related.py:

Start line: 1423, End line: 1461

```python
class ManyToManyField(RelatedField):

    def _check_ignored_options(self, **kwargs):
        warnings = []

        if self.has_null_arg:
            warnings.append(
                checks.Warning(
                    "null has no effect on ManyToManyField.",
                    obj=self,
                    id="fields.W340",
                )
            )

        if self._validators:
            warnings.append(
                checks.Warning(
                    "ManyToManyField does not support validators.",
                    obj=self,
                    id="fields.W341",
                )
            )
        if self.remote_field.symmetrical and self._related_name:
            warnings.append(
                checks.Warning(
                    "related_name has no effect on ManyToManyField "
                    'with a symmetrical relationship, e.g. to "self".',
                    obj=self,
                    id="fields.W345",
                )
            )
        if self.db_comment:
            warnings.append(
                checks.Warning(
                    "db_comment has no effect on ManyToManyField.",
                    obj=self,
                    id="fields.W346",
                )
            )

        return warnings
```
### 7 - django/contrib/admin/options.py:

Start line: 2459, End line: 2486

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
### 8 - django/db/models/fields/related_descriptors.py:

Start line: 1066, End line: 1087

```python
def create_forward_many_to_many_manager(superclass, rel, reverse):

    class ManyRelatedManager(superclass, AltersData):

        def _apply_rel_filters(self, queryset):
            """
            Filter the queryset for the instance this manager is bound to.
            """
            queryset._add_hints(instance=self.instance)
            if self._db:
                queryset = queryset.using(self._db)
            queryset._defer_next_filter = True
            return queryset._next_is_sticky().filter(**self.core_filters)

        def _remove_prefetched_objects(self):
            try:
                self.instance._prefetched_objects_cache.pop(self.prefetch_cache_name)
            except (AttributeError, KeyError):
                pass  # nothing to clear from cache

        def get_queryset(self):
            try:
                return self.instance._prefetched_objects_cache[self.prefetch_cache_name]
            except (AttributeError, KeyError):
                queryset = super().get_queryset()
                return self._apply_rel_filters(queryset)
    # ... other code
```
### 9 - django/contrib/admin/options.py:

Start line: 441, End line: 506

```python
class BaseModelAdmin(metaclass=forms.MediaDefiningClass):
    def lookup_allowed(self, lookup, value, request=None):
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
            if not prev_field or (
                prev_field.is_relation
                and field not in model._meta.parents.values()
                and field is not model._meta.auto_field
                and (
                    model._meta.auto_field is None
                    or part not in getattr(prev_field, "to_fields", [])
                )
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
        # RemovedInDjango60Warning: when the deprecation ends, replace with:
        # for filter_item in self.get_list_filter(request):
        list_filter = (
            self.get_list_filter(request) if request is not None else self.list_filter
        )
        for filter_item in list_filter:
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
### 10 - django/db/models/fields/related.py:

Start line: 1894, End line: 1941

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
            self.remote_field.model == RECURSIVE_RELATIONSHIP_CONSTANT
            or self.remote_field.model == cls._meta.object_name
        ):
            self.remote_field.related_name = "%s_rel_+" % name
        elif self.remote_field.is_hidden():
            # If the backwards relation is disabled, replace the original
            # related_name with one generated from the m2m field name. Django
            # still uses backwards relations internally and we need to avoid
            # clashes between multiple m2m fields with related_name == '+'.
            self.remote_field.related_name = "_%s_%s_%s_+" % (
                cls._meta.app_label,
                cls.__name__.lower(),
                name,
            )

        super().contribute_to_class(cls, name, **kwargs)

        # The intermediate m2m model is not auto created if:
        #  1) There is a manually specified intermediate, or
        #  2) The class owning the m2m field is abstract.
        #  3) The class owning the m2m field has been swapped out.
        if not cls._meta.abstract:
            if self.remote_field.through:

                def resolve_through_model(_, model, field):
                    field.remote_field.through = model

                lazy_related_operation(
                    resolve_through_model, cls, self.remote_field.through, field=self
                )
            elif not cls._meta.swapped:
                self.remote_field.through = create_many_to_many_intermediary_model(
                    self, cls
                )

        # Add the descriptor for the m2m relation.
        setattr(cls, self.name, ManyToManyDescriptor(self.remote_field, reverse=False))

        # Set up the accessor for the m2m table name for the relation.
        self.m2m_db_table = partial(self._get_m2m_db_table, cls._meta)
```
### 13 - django/contrib/admin/checks.py:

Start line: 790, End line: 808

```python
class ModelAdminChecks(BaseModelAdminChecks):
    def check(self, admin_obj, **kwargs):
        return [
            *super().check(admin_obj),
            *self._check_save_as(admin_obj),
            *self._check_save_on_top(admin_obj),
            *self._check_inlines(admin_obj),
            *self._check_list_display(admin_obj),
            *self._check_list_display_links(admin_obj),
            *self._check_list_filter(admin_obj),
            *self._check_list_select_related(admin_obj),
            *self._check_list_per_page(admin_obj),
            *self._check_list_max_show_all(admin_obj),
            *self._check_list_editable(admin_obj),
            *self._check_search_fields(admin_obj),
            *self._check_date_hierarchy(admin_obj),
            *self._check_action_permission_methods(admin_obj),
            *self._check_actions_uniqueness(admin_obj),
        ]
```
### 19 - django/contrib/admin/checks.py:

Start line: 959, End line: 983

```python
class ModelAdminChecks(BaseModelAdminChecks):

    def _check_list_display_links_item(self, obj, field_name, label):
        if field_name not in obj.list_display:
            return [
                checks.Error(
                    "The value of '%s' refers to '%s', which is not defined in "
                    "'list_display'." % (label, field_name),
                    obj=obj.__class__,
                    id="admin.E111",
                )
            ]
        else:
            return []

    def _check_list_filter(self, obj):
        if not isinstance(obj.list_filter, (list, tuple)):
            return must_be(
                "a list or tuple", option="list_filter", obj=obj, id="admin.E112"
            )
        else:
            return list(
                chain.from_iterable(
                    self._check_list_filter_item(obj, item, "list_filter[%d]" % index)
                    for index, item in enumerate(obj.list_filter)
                )
            )
```
### 20 - django/contrib/admin/checks.py:

Start line: 217, End line: 264

```python
class BaseModelAdminChecks:

    def _check_autocomplete_fields_item(self, obj, field_name, label):
        """
        Check that an item in `autocomplete_fields` is a ForeignKey or a
        ManyToManyField and that the item has a related ModelAdmin with
        search_fields defined.
        """
        try:
            field = obj.model._meta.get_field(field_name)
        except FieldDoesNotExist:
            return refer_to_missing_field(
                field=field_name, option=label, obj=obj, id="admin.E037"
            )
        else:
            if not field.many_to_many and not isinstance(field, models.ForeignKey):
                return must_be(
                    "a foreign key or a many-to-many field",
                    option=label,
                    obj=obj,
                    id="admin.E038",
                )
            related_admin = obj.admin_site._registry.get(field.remote_field.model)
            if related_admin is None:
                return [
                    checks.Error(
                        'An admin for model "%s" has to be registered '
                        "to be referenced by %s.autocomplete_fields."
                        % (
                            field.remote_field.model.__name__,
                            type(obj).__name__,
                        ),
                        obj=obj.__class__,
                        id="admin.E039",
                    )
                ]
            elif not related_admin.search_fields:
                return [
                    checks.Error(
                        '%s must define "search_fields", because it\'s '
                        "referenced by %s.autocomplete_fields."
                        % (
                            related_admin.__class__.__name__,
                            type(obj).__name__,
                        ),
                        obj=obj.__class__,
                        id="admin.E040",
                    )
                ]
            return []
```
### 21 - django/contrib/admin/checks.py:

Start line: 1248, End line: 1277

```python
class InlineModelAdminChecks(BaseModelAdminChecks):

    def _check_exclude_of_parent_model(self, obj, parent_model):
        # Do not perform more specific checks if the base checks result in an
        # error.
        errors = super()._check_exclude(obj)
        if errors:
            return []

        # Skip if `fk_name` is invalid.
        if self._check_relation(obj, parent_model):
            return []

        if obj.exclude is None:
            return []

        fk = _get_foreign_key(parent_model, obj.model, fk_name=obj.fk_name)
        if fk.name in obj.exclude:
            return [
                checks.Error(
                    "Cannot exclude the field '%s', because it is the foreign key "
                    "to the parent model '%s'."
                    % (
                        fk.name,
                        parent_model._meta.label,
                    ),
                    obj=obj.__class__,
                    id="admin.E201",
                )
            ]
        else:
            return []
```
### 22 - django/contrib/admin/checks.py:

Start line: 176, End line: 192

```python
class BaseModelAdminChecks:
    def check(self, admin_obj, **kwargs):
        return [
            *self._check_autocomplete_fields(admin_obj),
            *self._check_raw_id_fields(admin_obj),
            *self._check_fields(admin_obj),
            *self._check_fieldsets(admin_obj),
            *self._check_exclude(admin_obj),
            *self._check_form(admin_obj),
            *self._check_filter_vertical(admin_obj),
            *self._check_filter_horizontal(admin_obj),
            *self._check_radio_fields(admin_obj),
            *self._check_prepopulated_fields(admin_obj),
            *self._check_view_on_site_url(admin_obj),
            *self._check_ordering(admin_obj),
            *self._check_readonly_fields(admin_obj),
        ]
```
### 23 - django/contrib/admin/checks.py:

Start line: 480, End line: 503

```python
class BaseModelAdminChecks:

    def _check_form(self, obj):
        """Check that form subclasses BaseModelForm."""
        if not _issubclass(obj.form, BaseModelForm):
            return must_inherit_from(
                parent="BaseModelForm", option="form", obj=obj, id="admin.E016"
            )
        else:
            return []

    def _check_filter_vertical(self, obj):
        """Check that filter_vertical is a sequence of field names."""
        if not isinstance(obj.filter_vertical, (list, tuple)):
            return must_be(
                "a list or tuple", option="filter_vertical", obj=obj, id="admin.E017"
            )
        else:
            return list(
                chain.from_iterable(
                    self._check_filter_item(
                        obj, field_name, "filter_vertical[%d]" % index
                    )
                    for index, field_name in enumerate(obj.filter_vertical)
                )
            )
```
### 27 - django/contrib/admin/checks.py:

Start line: 1149, End line: 1185

```python
class ModelAdminChecks(BaseModelAdminChecks):

    def _check_search_fields(self, obj):
        """Check search_fields is a sequence."""

        if not isinstance(obj.search_fields, (list, tuple)):
            return must_be(
                "a list or tuple", option="search_fields", obj=obj, id="admin.E126"
            )
        else:
            return []

    def _check_date_hierarchy(self, obj):
        """Check that date_hierarchy refers to DateField or DateTimeField."""

        if obj.date_hierarchy is None:
            return []
        else:
            try:
                field = get_fields_from_path(obj.model, obj.date_hierarchy)[-1]
            except (NotRelationField, FieldDoesNotExist):
                return [
                    checks.Error(
                        "The value of 'date_hierarchy' refers to '%s', which "
                        "does not refer to a Field." % obj.date_hierarchy,
                        obj=obj.__class__,
                        id="admin.E127",
                    )
                ]
            else:
                if not isinstance(field, (models.DateField, models.DateTimeField)):
                    return must_be(
                        "a DateField or DateTimeField",
                        option="date_hierarchy",
                        obj=obj,
                        id="admin.E128",
                    )
                else:
                    return []
```
### 36 - django/contrib/admin/checks.py:

Start line: 1235, End line: 1246

```python
class InlineModelAdminChecks(BaseModelAdminChecks):
    def check(self, inline_obj, **kwargs):
        parent_model = inline_obj.parent_model
        return [
            *super().check(inline_obj),
            *self._check_relation(inline_obj, parent_model),
            *self._check_exclude_of_parent_model(inline_obj, parent_model),
            *self._check_extra(inline_obj),
            *self._check_max_num(inline_obj),
            *self._check_min_num(inline_obj),
            *self._check_formset(inline_obj),
        ]
```
### 38 - django/contrib/admin/checks.py:

Start line: 1095, End line: 1147

```python
class ModelAdminChecks(BaseModelAdminChecks):

    def _check_list_editable_item(self, obj, field_name, label):
        try:
            field = obj.model._meta.get_field(field_name)
        except FieldDoesNotExist:
            return refer_to_missing_field(
                field=field_name, option=label, obj=obj, id="admin.E121"
            )
        else:
            if field_name not in obj.list_display:
                return [
                    checks.Error(
                        "The value of '%s' refers to '%s', which is not "
                        "contained in 'list_display'." % (label, field_name),
                        obj=obj.__class__,
                        id="admin.E122",
                    )
                ]
            elif obj.list_display_links and field_name in obj.list_display_links:
                return [
                    checks.Error(
                        "The value of '%s' cannot be in both 'list_editable' and "
                        "'list_display_links'." % field_name,
                        obj=obj.__class__,
                        id="admin.E123",
                    )
                ]
            # If list_display[0] is in list_editable, check that
            # list_display_links is set. See #22792 and #26229 for use cases.
            elif (
                obj.list_display[0] == field_name
                and not obj.list_display_links
                and obj.list_display_links is not None
            ):
                return [
                    checks.Error(
                        "The value of '%s' refers to the first field in 'list_display' "
                        "('%s'), which cannot be used unless 'list_display_links' is "
                        "set." % (label, obj.list_display[0]),
                        obj=obj.__class__,
                        id="admin.E124",
                    )
                ]
            elif not field.editable or field.primary_key:
                return [
                    checks.Error(
                        "The value of '%s' refers to '%s', which is not editable "
                        "through the admin." % (label, field_name),
                        obj=obj.__class__,
                        id="admin.E125",
                    )
                ]
            else:
                return []
```
### 43 - django/contrib/admin/checks.py:

Start line: 284, End line: 312

```python
class BaseModelAdminChecks:

    def _check_raw_id_fields_item(self, obj, field_name, label):
        """Check an item of `raw_id_fields`, i.e. check that field named
        `field_name` exists in model `model` and is a ForeignKey or a
        ManyToManyField."""

        try:
            field = obj.model._meta.get_field(field_name)
        except FieldDoesNotExist:
            return refer_to_missing_field(
                field=field_name, option=label, obj=obj, id="admin.E002"
            )
        else:
            # Using attname is not supported.
            if field.name != field_name:
                return refer_to_missing_field(
                    field=field_name,
                    option=label,
                    obj=obj,
                    id="admin.E002",
                )
            if not field.many_to_many and not isinstance(field, models.ForeignKey):
                return must_be(
                    "a foreign key or a many-to-many field",
                    option=label,
                    obj=obj,
                    id="admin.E003",
                )
            else:
                return []
```
### 48 - django/contrib/admin/checks.py:

Start line: 841, End line: 877

```python
class ModelAdminChecks(BaseModelAdminChecks):

    def _check_inlines_item(self, obj, inline, label):
        """Check one inline model admin."""
        try:
            inline_label = inline.__module__ + "." + inline.__name__
        except AttributeError:
            return [
                checks.Error(
                    "'%s' must inherit from 'InlineModelAdmin'." % obj,
                    obj=obj.__class__,
                    id="admin.E104",
                )
            ]

        from django.contrib.admin.options import InlineModelAdmin

        if not _issubclass(inline, InlineModelAdmin):
            return [
                checks.Error(
                    "'%s' must inherit from 'InlineModelAdmin'." % inline_label,
                    obj=obj.__class__,
                    id="admin.E104",
                )
            ]
        elif not inline.model:
            return [
                checks.Error(
                    "'%s' must have a 'model' attribute." % inline_label,
                    obj=obj.__class__,
                    id="admin.E105",
                )
            ]
        elif not _issubclass(inline.model, models.Model):
            return must_be(
                "a Model", option="%s.model" % inline_label, obj=obj, id="admin.E106"
            )
        else:
            return inline(obj.model, obj.admin_site).check()
```
### 51 - django/contrib/admin/checks.py:

Start line: 266, End line: 282

```python
class BaseModelAdminChecks:

    def _check_raw_id_fields(self, obj):
        """Check that `raw_id_fields` only contains field names that are listed
        on the model."""

        if not isinstance(obj.raw_id_fields, (list, tuple)):
            return must_be(
                "a list or tuple", option="raw_id_fields", obj=obj, id="admin.E001"
            )
        else:
            return list(
                chain.from_iterable(
                    self._check_raw_id_fields_item(
                        obj, field_name, "raw_id_fields[%d]" % index
                    )
                    for index, field_name in enumerate(obj.raw_id_fields)
                )
            )
```
### 54 - django/contrib/admin/checks.py:

Start line: 985, End line: 1042

```python
class ModelAdminChecks(BaseModelAdminChecks):

    def _check_list_filter_item(self, obj, item, label):
        """
        Check one item of `list_filter`, i.e. check if it is one of three options:
        1. 'field' -- a basic field filter, possibly w/ relationships (e.g.
           'field__rel')
        2. ('field', SomeFieldListFilter) - a field-based list filter class
        3. SomeListFilter - a non-field list filter class
        """
        from django.contrib.admin import FieldListFilter, ListFilter

        if callable(item) and not isinstance(item, models.Field):
            # If item is option 3, it should be a ListFilter...
            if not _issubclass(item, ListFilter):
                return must_inherit_from(
                    parent="ListFilter", option=label, obj=obj, id="admin.E113"
                )
            # ...  but not a FieldListFilter.
            elif issubclass(item, FieldListFilter):
                return [
                    checks.Error(
                        "The value of '%s' must not inherit from 'FieldListFilter'."
                        % label,
                        obj=obj.__class__,
                        id="admin.E114",
                    )
                ]
            else:
                return []
        elif isinstance(item, (tuple, list)):
            # item is option #2
            field, list_filter_class = item
            if not _issubclass(list_filter_class, FieldListFilter):
                return must_inherit_from(
                    parent="FieldListFilter",
                    option="%s[1]" % label,
                    obj=obj,
                    id="admin.E115",
                )
            else:
                return []
        else:
            # item is option #1
            field = item

            # Validate the field string
            try:
                get_fields_from_path(obj.model, field)
            except (NotRelationField, FieldDoesNotExist):
                return [
                    checks.Error(
                        "The value of '%s' refers to '%s', which does not refer to a "
                        "Field." % (label, field),
                        obj=obj.__class__,
                        id="admin.E116",
                    )
                ]
            else:
                return []
```
### 56 - django/contrib/admin/checks.py:

Start line: 194, End line: 215

```python
class BaseModelAdminChecks:

    def _check_autocomplete_fields(self, obj):
        """
        Check that `autocomplete_fields` is a list or tuple of model fields.
        """
        if not isinstance(obj.autocomplete_fields, (list, tuple)):
            return must_be(
                "a list or tuple",
                option="autocomplete_fields",
                obj=obj,
                id="admin.E036",
            )
        else:
            return list(
                chain.from_iterable(
                    [
                        self._check_autocomplete_fields_item(
                            obj, field_name, "autocomplete_fields[%d]" % index
                        )
                        for index, field_name in enumerate(obj.autocomplete_fields)
                    ]
                )
            )
```
### 58 - django/contrib/admin/checks.py:

Start line: 430, End line: 458

```python
class BaseModelAdminChecks:

    def _check_field_spec_item(self, obj, field_name, label):
        if field_name in obj.readonly_fields:
            # Stuff can be put in fields that isn't actually a model field if
            # it's in readonly_fields, readonly_fields will handle the
            # validation of such things.
            return []
        else:
            try:
                field = obj.model._meta.get_field(field_name)
            except FieldDoesNotExist:
                # If we can't find a field on the model that matches, it could
                # be an extra field on the form.
                return []
            else:
                if (
                    isinstance(field, models.ManyToManyField)
                    and not field.remote_field.through._meta.auto_created
                ):
                    return [
                        checks.Error(
                            "The value of '%s' cannot include the ManyToManyField "
                            "'%s', because that field manually specifies a "
                            "relationship model." % (label, field_name),
                            obj=obj.__class__,
                            id="admin.E013",
                        )
                    ]
                else:
                    return []
```
### 65 - django/contrib/admin/checks.py:

Start line: 826, End line: 839

```python
class ModelAdminChecks(BaseModelAdminChecks):

    def _check_inlines(self, obj):
        """Check all inline model admin classes."""

        if not isinstance(obj.inlines, (list, tuple)):
            return must_be(
                "a list or tuple", option="inlines", obj=obj, id="admin.E103"
            )
        else:
            return list(
                chain.from_iterable(
                    self._check_inlines_item(obj, item, "inlines[%d]" % index)
                    for index, item in enumerate(obj.inlines)
                )
            )
```
### 68 - django/contrib/admin/checks.py:

Start line: 879, End line: 892

```python
class ModelAdminChecks(BaseModelAdminChecks):

    def _check_list_display(self, obj):
        """Check that list_display only contains fields or usable attributes."""

        if not isinstance(obj.list_display, (list, tuple)):
            return must_be(
                "a list or tuple", option="list_display", obj=obj, id="admin.E107"
            )
        else:
            return list(
                chain.from_iterable(
                    self._check_list_display_item(obj, item, "list_display[%d]" % index)
                    for index, item in enumerate(obj.list_display)
                )
            )
```
### 74 - django/contrib/admin/checks.py:

Start line: 704, End line: 740

```python
class BaseModelAdminChecks:

    def _check_ordering_item(self, obj, field_name, label):
        """Check that `ordering` refers to existing fields."""
        if isinstance(field_name, (Combinable, models.OrderBy)):
            if not isinstance(field_name, models.OrderBy):
                field_name = field_name.asc()
            if isinstance(field_name.expression, models.F):
                field_name = field_name.expression.name
            else:
                return []
        if field_name == "?" and len(obj.ordering) != 1:
            return [
                checks.Error(
                    "The value of 'ordering' has the random ordering marker '?', "
                    "but contains other fields as well.",
                    hint='Either remove the "?", or remove the other fields.',
                    obj=obj.__class__,
                    id="admin.E032",
                )
            ]
        elif field_name == "?":
            return []
        elif LOOKUP_SEP in field_name:
            # Skip ordering in the format field1__field2 (FIXME: checking
            # this format would be nice, but it's a little fiddly).
            return []
        else:
            field_name = field_name.removeprefix("-")
            if field_name == "pk":
                return []
            try:
                obj.model._meta.get_field(field_name)
            except FieldDoesNotExist:
                return refer_to_missing_field(
                    field=field_name, option=label, obj=obj, id="admin.E033"
                )
            else:
                return []
```
### 77 - django/contrib/admin/checks.py:

Start line: 1044, End line: 1075

```python
class ModelAdminChecks(BaseModelAdminChecks):

    def _check_list_select_related(self, obj):
        """Check that list_select_related is a boolean, a list or a tuple."""

        if not isinstance(obj.list_select_related, (bool, list, tuple)):
            return must_be(
                "a boolean, tuple or list",
                option="list_select_related",
                obj=obj,
                id="admin.E117",
            )
        else:
            return []

    def _check_list_per_page(self, obj):
        """Check that list_per_page is an integer."""

        if not isinstance(obj.list_per_page, int):
            return must_be(
                "an integer", option="list_per_page", obj=obj, id="admin.E118"
            )
        else:
            return []

    def _check_list_max_show_all(self, obj):
        """Check that list_max_show_all is an integer."""

        if not isinstance(obj.list_max_show_all, int):
            return must_be(
                "an integer", option="list_max_show_all", obj=obj, id="admin.E119"
            )
        else:
            return []
```
### 87 - django/contrib/admin/checks.py:

Start line: 1279, End line: 1323

```python
class InlineModelAdminChecks(BaseModelAdminChecks):

    def _check_relation(self, obj, parent_model):
        try:
            _get_foreign_key(parent_model, obj.model, fk_name=obj.fk_name)
        except ValueError as e:
            return [checks.Error(e.args[0], obj=obj.__class__, id="admin.E202")]
        else:
            return []

    def _check_extra(self, obj):
        """Check that extra is an integer."""

        if not isinstance(obj.extra, int):
            return must_be("an integer", option="extra", obj=obj, id="admin.E203")
        else:
            return []

    def _check_max_num(self, obj):
        """Check that max_num is an integer."""

        if obj.max_num is None:
            return []
        elif not isinstance(obj.max_num, int):
            return must_be("an integer", option="max_num", obj=obj, id="admin.E204")
        else:
            return []

    def _check_min_num(self, obj):
        """Check that min_num is an integer."""

        if obj.min_num is None:
            return []
        elif not isinstance(obj.min_num, int):
            return must_be("an integer", option="min_num", obj=obj, id="admin.E205")
        else:
            return []

    def _check_formset(self, obj):
        """Check formset is a subclass of BaseModelFormSet."""

        if not _issubclass(obj.formset, BaseModelFormSet):
            return must_inherit_from(
                parent="BaseModelFormSet", option="formset", obj=obj, id="admin.E206"
            )
        else:
            return []
```
### 89 - django/contrib/admin/checks.py:

Start line: 742, End line: 759

```python
class BaseModelAdminChecks:

    def _check_readonly_fields(self, obj):
        """Check that readonly_fields refers to proper attribute or field."""

        if obj.readonly_fields == ():
            return []
        elif not isinstance(obj.readonly_fields, (list, tuple)):
            return must_be(
                "a list or tuple", option="readonly_fields", obj=obj, id="admin.E034"
            )
        else:
            return list(
                chain.from_iterable(
                    self._check_readonly_fields_item(
                        obj, field_name, "readonly_fields[%d]" % index
                    )
                    for index, field_name in enumerate(obj.readonly_fields)
                )
            )
```
### 92 - django/contrib/admin/checks.py:

Start line: 673, End line: 702

```python
class BaseModelAdminChecks:

    def _check_prepopulated_fields_value_item(self, obj, field_name, label):
        """For `prepopulated_fields` equal to {"slug": ("title",)},
        `field_name` is "title"."""

        try:
            obj.model._meta.get_field(field_name)
        except FieldDoesNotExist:
            return refer_to_missing_field(
                field=field_name, option=label, obj=obj, id="admin.E030"
            )
        else:
            return []

    def _check_ordering(self, obj):
        """Check that ordering refers to existing fields or is random."""

        # ordering = None
        if obj.ordering is None:  # The default value is None
            return []
        elif not isinstance(obj.ordering, (list, tuple)):
            return must_be(
                "a list or tuple", option="ordering", obj=obj, id="admin.E031"
            )
        else:
            return list(
                chain.from_iterable(
                    self._check_ordering_item(obj, field_name, "ordering[%d]" % index)
                    for index, field_name in enumerate(obj.ordering)
                )
            )
```
### 93 - django/contrib/admin/checks.py:

Start line: 314, End line: 346

```python
class BaseModelAdminChecks:

    def _check_fields(self, obj):
        """Check that `fields` only refer to existing fields, doesn't contain
        duplicates. Check if at most one of `fields` and `fieldsets` is defined.
        """

        if obj.fields is None:
            return []
        elif not isinstance(obj.fields, (list, tuple)):
            return must_be("a list or tuple", option="fields", obj=obj, id="admin.E004")
        elif obj.fieldsets:
            return [
                checks.Error(
                    "Both 'fieldsets' and 'fields' are specified.",
                    obj=obj.__class__,
                    id="admin.E005",
                )
            ]
        fields = flatten(obj.fields)
        if len(fields) != len(set(fields)):
            return [
                checks.Error(
                    "The value of 'fields' contains duplicate field(s).",
                    obj=obj.__class__,
                    id="admin.E006",
                )
            ]

        return list(
            chain.from_iterable(
                self._check_field_spec(obj, field_name, "fields")
                for field_name in obj.fields
            )
        )
```
### 94 - django/contrib/admin/checks.py:

Start line: 894, End line: 932

```python
class ModelAdminChecks(BaseModelAdminChecks):

    def _check_list_display_item(self, obj, item, label):
        if callable(item):
            return []
        elif hasattr(obj, item):
            return []
        try:
            field = obj.model._meta.get_field(item)
        except FieldDoesNotExist:
            try:
                field = getattr(obj.model, item)
            except AttributeError:
                return [
                    checks.Error(
                        "The value of '%s' refers to '%s', which is not a "
                        "callable, an attribute of '%s', or an attribute or "
                        "method on '%s'."
                        % (
                            label,
                            item,
                            obj.__class__.__name__,
                            obj.model._meta.label,
                        ),
                        obj=obj.__class__,
                        id="admin.E108",
                    )
                ]
        if (
            getattr(field, "is_relation", False)
            and (field.many_to_many or field.one_to_many)
        ) or (getattr(field, "rel", None) and field.rel.field.many_to_one):
            return [
                checks.Error(
                    f"The value of '{label}' must not be a many-to-many field or a "
                    f"reverse foreign key.",
                    obj=obj.__class__,
                    id="admin.E109",
                )
            ]
        return []
```
### 101 - django/contrib/admin/checks.py:

Start line: 761, End line: 787

```python
class BaseModelAdminChecks:

    def _check_readonly_fields_item(self, obj, field_name, label):
        if callable(field_name):
            return []
        elif hasattr(obj, field_name):
            return []
        elif hasattr(obj.model, field_name):
            return []
        else:
            try:
                obj.model._meta.get_field(field_name)
            except FieldDoesNotExist:
                return [
                    checks.Error(
                        "The value of '%s' refers to '%s', which is not a callable, "
                        "an attribute of '%s', or an attribute of '%s'."
                        % (
                            label,
                            field_name,
                            obj.__class__.__name__,
                            obj.model._meta.label,
                        ),
                        obj=obj.__class__,
                        id="admin.E035",
                    )
                ]
            else:
                return []
```
### 106 - django/contrib/admin/checks.py:

Start line: 348, End line: 367

```python
class BaseModelAdminChecks:

    def _check_fieldsets(self, obj):
        """Check that fieldsets is properly formatted and doesn't contain
        duplicates."""

        if obj.fieldsets is None:
            return []
        elif not isinstance(obj.fieldsets, (list, tuple)):
            return must_be(
                "a list or tuple", option="fieldsets", obj=obj, id="admin.E007"
            )
        else:
            seen_fields = []
            return list(
                chain.from_iterable(
                    self._check_fieldsets_item(
                        obj, fieldset, "fieldsets[%d]" % index, seen_fields
                    )
                    for index, fieldset in enumerate(obj.fieldsets)
                )
            )
```
### 110 - django/contrib/admin/checks.py:

Start line: 1326, End line: 1355

```python
def must_be(type, option, obj, id):
    return [
        checks.Error(
            "The value of '%s' must be %s." % (option, type),
            obj=obj.__class__,
            id=id,
        ),
    ]


def must_inherit_from(parent, option, obj, id):
    return [
        checks.Error(
            "The value of '%s' must inherit from '%s'." % (option, parent),
            obj=obj.__class__,
            id=id,
        ),
    ]


def refer_to_missing_field(field, option, obj, id):
    return [
        checks.Error(
            "The value of '%s' refers to '%s', which is not a field of '%s'."
            % (option, field, obj.model._meta.label),
            obj=obj.__class__,
            id=id,
        ),
    ]
```
### 111 - django/contrib/admin/checks.py:

Start line: 460, End line: 478

```python
class BaseModelAdminChecks:

    def _check_exclude(self, obj):
        """Check that exclude is a sequence without duplicates."""

        if obj.exclude is None:  # default value is None
            return []
        elif not isinstance(obj.exclude, (list, tuple)):
            return must_be(
                "a list or tuple", option="exclude", obj=obj, id="admin.E014"
            )
        elif len(obj.exclude) > len(set(obj.exclude)):
            return [
                checks.Error(
                    "The value of 'exclude' contains duplicate field(s).",
                    obj=obj.__class__,
                    id="admin.E015",
                )
            ]
        else:
            return []
```
### 116 - django/contrib/admin/checks.py:

Start line: 580, End line: 608

```python
class BaseModelAdminChecks:

    def _check_radio_fields_value(self, obj, val, label):
        """Check type of a value of `radio_fields` dictionary."""

        from django.contrib.admin.options import HORIZONTAL, VERTICAL

        if val not in (HORIZONTAL, VERTICAL):
            return [
                checks.Error(
                    "The value of '%s' must be either admin.HORIZONTAL or "
                    "admin.VERTICAL." % label,
                    obj=obj.__class__,
                    id="admin.E024",
                )
            ]
        else:
            return []

    def _check_view_on_site_url(self, obj):
        if not callable(obj.view_on_site) and not isinstance(obj.view_on_site, bool):
            return [
                checks.Error(
                    "The value of 'view_on_site' must be a callable or a boolean "
                    "value.",
                    obj=obj.__class__,
                    id="admin.E025",
                )
            ]
        else:
            return []
```
### 129 - django/contrib/admin/checks.py:

Start line: 610, End line: 628

```python
class BaseModelAdminChecks:

    def _check_prepopulated_fields(self, obj):
        """Check that `prepopulated_fields` is a dictionary containing allowed
        field types."""
        if not isinstance(obj.prepopulated_fields, dict):
            return must_be(
                "a dictionary", option="prepopulated_fields", obj=obj, id="admin.E026"
            )
        else:
            return list(
                chain.from_iterable(
                    self._check_prepopulated_fields_key(
                        obj, field_name, "prepopulated_fields"
                    )
                    + self._check_prepopulated_fields_value(
                        obj, val, 'prepopulated_fields["%s"]' % field_name
                    )
                    for field_name, val in obj.prepopulated_fields.items()
                )
            )
```
### 130 - django/contrib/admin/checks.py:

Start line: 1187, End line: 1212

```python
class ModelAdminChecks(BaseModelAdminChecks):

    def _check_action_permission_methods(self, obj):
        """
        Actions with an allowed_permission attribute require the ModelAdmin to
        implement a has_<perm>_permission() method for each permission.
        """
        actions = obj._get_base_actions()
        errors = []
        for func, name, _ in actions:
            if not hasattr(func, "allowed_permissions"):
                continue
            for permission in func.allowed_permissions:
                method_name = "has_%s_permission" % permission
                if not hasattr(obj, method_name):
                    errors.append(
                        checks.Error(
                            "%s must define a %s() method for the %s action."
                            % (
                                obj.__class__.__name__,
                                method_name,
                                func.__name__,
                            ),
                            obj=obj.__class__,
                            id="admin.E129",
                        )
                    )
        return errors
```
### 134 - django/contrib/admin/checks.py:

Start line: 55, End line: 173

```python
def check_dependencies(**kwargs):
    """
    Check that the admin's dependencies are correctly installed.
    """
    from django.contrib.admin.sites import all_sites

    if not apps.is_installed("django.contrib.admin"):
        return []
    errors = []
    app_dependencies = (
        ("django.contrib.contenttypes", 401),
        ("django.contrib.auth", 405),
        ("django.contrib.messages", 406),
    )
    for app_name, error_code in app_dependencies:
        if not apps.is_installed(app_name):
            errors.append(
                checks.Error(
                    "'%s' must be in INSTALLED_APPS in order to use the admin "
                    "application." % app_name,
                    id="admin.E%d" % error_code,
                )
            )
    for engine in engines.all():
        if isinstance(engine, DjangoTemplates):
            django_templates_instance = engine.engine
            break
    else:
        django_templates_instance = None
    if not django_templates_instance:
        errors.append(
            checks.Error(
                "A 'django.template.backends.django.DjangoTemplates' instance "
                "must be configured in TEMPLATES in order to use the admin "
                "application.",
                id="admin.E403",
            )
        )
    else:
        if (
            "django.contrib.auth.context_processors.auth"
            not in django_templates_instance.context_processors
            and _contains_subclass(
                "django.contrib.auth.backends.ModelBackend",
                settings.AUTHENTICATION_BACKENDS,
            )
        ):
            errors.append(
                checks.Error(
                    "'django.contrib.auth.context_processors.auth' must be "
                    "enabled in DjangoTemplates (TEMPLATES) if using the default "
                    "auth backend in order to use the admin application.",
                    id="admin.E402",
                )
            )
        if (
            "django.contrib.messages.context_processors.messages"
            not in django_templates_instance.context_processors
        ):
            errors.append(
                checks.Error(
                    "'django.contrib.messages.context_processors.messages' must "
                    "be enabled in DjangoTemplates (TEMPLATES) in order to use "
                    "the admin application.",
                    id="admin.E404",
                )
            )
        sidebar_enabled = any(site.enable_nav_sidebar for site in all_sites)
        if (
            sidebar_enabled
            and "django.template.context_processors.request"
            not in django_templates_instance.context_processors
        ):
            errors.append(
                checks.Warning(
                    "'django.template.context_processors.request' must be enabled "
                    "in DjangoTemplates (TEMPLATES) in order to use the admin "
                    "navigation sidebar.",
                    id="admin.W411",
                )
            )

    if not _contains_subclass(
        "django.contrib.auth.middleware.AuthenticationMiddleware", settings.MIDDLEWARE
    ):
        errors.append(
            checks.Error(
                "'django.contrib.auth.middleware.AuthenticationMiddleware' must "
                "be in MIDDLEWARE in order to use the admin application.",
                id="admin.E408",
            )
        )
    if not _contains_subclass(
        "django.contrib.messages.middleware.MessageMiddleware", settings.MIDDLEWARE
    ):
        errors.append(
            checks.Error(
                "'django.contrib.messages.middleware.MessageMiddleware' must "
                "be in MIDDLEWARE in order to use the admin application.",
                id="admin.E409",
            )
        )
    if not _contains_subclass(
        "django.contrib.sessions.middleware.SessionMiddleware", settings.MIDDLEWARE
    ):
        errors.append(
            checks.Error(
                "'django.contrib.sessions.middleware.SessionMiddleware' must "
                "be in MIDDLEWARE in order to use the admin application.",
                hint=(
                    "Insert "
                    "'django.contrib.sessions.middleware.SessionMiddleware' "
                    "before "
                    "'django.contrib.auth.middleware.AuthenticationMiddleware'."
                ),
                id="admin.E410",
            )
        )
    return errors
```
### 140 - django/contrib/admin/checks.py:

Start line: 934, End line: 957

```python
class ModelAdminChecks(BaseModelAdminChecks):

    def _check_list_display_links(self, obj):
        """Check that list_display_links is a unique subset of list_display."""
        from django.contrib.admin.options import ModelAdmin

        if obj.list_display_links is None:
            return []
        elif not isinstance(obj.list_display_links, (list, tuple)):
            return must_be(
                "a list, a tuple, or None",
                option="list_display_links",
                obj=obj,
                id="admin.E110",
            )
        # Check only if ModelAdmin.get_list_display() isn't overridden.
        elif obj.get_list_display.__func__ is ModelAdmin.get_list_display:
            return list(
                chain.from_iterable(
                    self._check_list_display_links_item(
                        obj, field_name, "list_display_links[%d]" % index
                    )
                    for index, field_name in enumerate(obj.list_display_links)
                )
            )
        return []
```
### 141 - django/contrib/admin/checks.py:

Start line: 413, End line: 428

```python
class BaseModelAdminChecks:

    def _check_field_spec(self, obj, fields, label):
        """`fields` should be an item of `fields` or an item of
        fieldset[1]['fields'] for any `fieldset` in `fieldsets`. It should be a
        field name or a tuple of field names."""

        if isinstance(fields, tuple):
            return list(
                chain.from_iterable(
                    self._check_field_spec_item(
                        obj, field_name, "%s[%d]" % (label, index)
                    )
                    for index, field_name in enumerate(fields)
                )
            )
        else:
            return self._check_field_spec_item(obj, fields, label)
```
### 143 - django/contrib/admin/checks.py:

Start line: 539, End line: 554

```python
class BaseModelAdminChecks:

    def _check_radio_fields(self, obj):
        """Check that `radio_fields` is a dictionary."""
        if not isinstance(obj.radio_fields, dict):
            return must_be(
                "a dictionary", option="radio_fields", obj=obj, id="admin.E021"
            )
        else:
            return list(
                chain.from_iterable(
                    self._check_radio_fields_key(obj, field_name, "radio_fields")
                    + self._check_radio_fields_value(
                        obj, val, 'radio_fields["%s"]' % field_name
                    )
                    for field_name, val in obj.radio_fields.items()
                )
            )
```
### 146 - django/contrib/admin/checks.py:

Start line: 810, End line: 824

```python
class ModelAdminChecks(BaseModelAdminChecks):

    def _check_save_as(self, obj):
        """Check save_as is a boolean."""

        if not isinstance(obj.save_as, bool):
            return must_be("a boolean", option="save_as", obj=obj, id="admin.E101")
        else:
            return []

    def _check_save_on_top(self, obj):
        """Check save_on_top is a boolean."""

        if not isinstance(obj.save_on_top, bool):
            return must_be("a boolean", option="save_on_top", obj=obj, id="admin.E102")
        else:
            return []
```
### 149 - django/contrib/admin/checks.py:

Start line: 1077, End line: 1093

```python
class ModelAdminChecks(BaseModelAdminChecks):

    def _check_list_editable(self, obj):
        """Check that list_editable is a sequence of editable fields from
        list_display without first element."""

        if not isinstance(obj.list_editable, (list, tuple)):
            return must_be(
                "a list or tuple", option="list_editable", obj=obj, id="admin.E120"
            )
        else:
            return list(
                chain.from_iterable(
                    self._check_list_editable_item(
                        obj, item, "list_editable[%d]" % index
                    )
                    for index, item in enumerate(obj.list_editable)
                )
            )
```
### 160 - django/contrib/admin/checks.py:

Start line: 657, End line: 671

```python
class BaseModelAdminChecks:

    def _check_prepopulated_fields_value(self, obj, val, label):
        """Check a value of `prepopulated_fields` dictionary, i.e. it's an
        iterable of existing fields."""

        if not isinstance(val, (list, tuple)):
            return must_be("a list or tuple", option=label, obj=obj, id="admin.E029")
        else:
            return list(
                chain.from_iterable(
                    self._check_prepopulated_fields_value_item(
                        obj, subfield_name, "%s[%r]" % (label, index)
                    )
                    for index, subfield_name in enumerate(val)
                )
            )
```
### 174 - django/contrib/admin/checks.py:

Start line: 556, End line: 578

```python
class BaseModelAdminChecks:

    def _check_radio_fields_key(self, obj, field_name, label):
        """Check that a key of `radio_fields` dictionary is name of existing
        field and that the field is a ForeignKey or has `choices` defined."""

        try:
            field = obj.model._meta.get_field(field_name)
        except FieldDoesNotExist:
            return refer_to_missing_field(
                field=field_name, option=label, obj=obj, id="admin.E022"
            )
        else:
            if not (isinstance(field, models.ForeignKey) or field.choices):
                return [
                    checks.Error(
                        "The value of '%s' refers to '%s', which is not an "
                        "instance of ForeignKey, and does not have a 'choices' "
                        "definition." % (label, field_name),
                        obj=obj.__class__,
                        id="admin.E023",
                    )
                ]
            else:
                return []
```
### 184 - django/contrib/admin/checks.py:

Start line: 1, End line: 52

```python
import collections
from itertools import chain

from django.apps import apps
from django.conf import settings
from django.contrib.admin.utils import NotRelationField, flatten, get_fields_from_path
from django.core import checks
from django.core.exceptions import FieldDoesNotExist
from django.db import models
from django.db.models.constants import LOOKUP_SEP
from django.db.models.expressions import Combinable
from django.forms.models import BaseModelForm, BaseModelFormSet, _get_foreign_key
from django.template import engines
from django.template.backends.django import DjangoTemplates
from django.utils.module_loading import import_string


def _issubclass(cls, classinfo):
    """
    issubclass() variant that doesn't raise an exception if cls isn't a
    class.
    """
    try:
        return issubclass(cls, classinfo)
    except TypeError:
        return False


def _contains_subclass(class_path, candidate_paths):
    """
    Return whether or not a dotted class path (or a subclass of that class) is
    found in a list of candidate paths.
    """
    cls = import_string(class_path)
    for path in candidate_paths:
        try:
            candidate_cls = import_string(path)
        except ImportError:
            # ImportErrors are raised elsewhere.
            continue
        if _issubclass(candidate_cls, cls):
            return True
    return False


def check_admin_app(app_configs, **kwargs):
    from django.contrib.admin.sites import all_sites

    errors = []
    for site in all_sites:
        errors.extend(site.check(app_configs))
    return errors
```
