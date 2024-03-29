# django__django-12856

| **django/django** | `8328811f048fed0dd22573224def8c65410c9f2e` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 1876 |
| **Any found context length** | 1876 |
| **Avg pos** | 5.0 |
| **Min pos** | 5 |
| **Max pos** | 5 |
| **Top file pos** | 2 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/db/models/base.py b/django/db/models/base.py
--- a/django/db/models/base.py
+++ b/django/db/models/base.py
@@ -1926,6 +1926,12 @@ def _check_constraints(cls, databases):
                         id='models.W038',
                     )
                 )
+            fields = (
+                field
+                for constraint in cls._meta.constraints if isinstance(constraint, UniqueConstraint)
+                for field in constraint.fields
+            )
+            errors.extend(cls._check_local_fields(fields, 'constraints'))
         return errors
 
 

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/db/models/base.py | 1929 | 1929 | 5 | 2 | 1876


## Problem Statement

```
Add check for fields of UniqueConstraints.
Description
	 
		(last modified by Marnanel Thurman)
	 
When a model gains a UniqueConstraint, makemigrations doesn't check that the fields named therein actually exist.
This is in contrast to the older unique_together syntax, which raises models.E012 if the fields don't exist.
In the attached demonstration, you'll need to uncomment "with_unique_together" in settings.py in order to show that unique_together raises E012.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 django/db/models/constraints.py | 79 | 154| 666 | 666 | 1252 | 
| 2 | **2 django/db/models/base.py** | 1561 | 1586| 183 | 849 | 17040 | 
| 3 | **2 django/db/models/base.py** | 1073 | 1116| 404 | 1253 | 17040 | 
| 4 | 3 django/db/backends/base/schema.py | 370 | 384| 182 | 1435 | 28640 | 
| **-> 5 <-** | **3 django/db/models/base.py** | 1866 | 1929| 441 | 1876 | 28640 | 
| 6 | 3 django/db/backends/base/schema.py | 1086 | 1106| 176 | 2052 | 28640 | 
| 7 | 4 django/db/models/fields/related.py | 509 | 574| 492 | 2544 | 42471 | 
| 8 | **4 django/db/models/base.py** | 1164 | 1192| 213 | 2757 | 42471 | 
| 9 | 5 django/forms/models.py | 680 | 751| 732 | 3489 | 54192 | 
| 10 | **5 django/db/models/base.py** | 1015 | 1071| 560 | 4049 | 54192 | 
| 11 | 6 django/contrib/admin/checks.py | 241 | 271| 229 | 4278 | 63296 | 
| 12 | 6 django/db/models/constraints.py | 32 | 76| 372 | 4650 | 63296 | 
| 13 | **6 django/db/models/base.py** | 1534 | 1559| 183 | 4833 | 63296 | 
| 14 | 6 django/forms/models.py | 753 | 774| 194 | 5027 | 63296 | 
| 15 | 6 django/db/backends/base/schema.py | 1139 | 1174| 273 | 5300 | 63296 | 
| 16 | 6 django/db/models/fields/related.py | 860 | 886| 240 | 5540 | 63296 | 
| 17 | **6 django/db/models/base.py** | 985 | 1013| 230 | 5770 | 63296 | 
| 18 | **6 django/db/models/base.py** | 1639 | 1687| 348 | 6118 | 63296 | 
| 19 | 6 django/forms/models.py | 413 | 443| 243 | 6361 | 63296 | 
| 20 | 7 django/db/models/options.py | 831 | 863| 225 | 6586 | 70402 | 
| 21 | 7 django/db/models/fields/related.py | 1424 | 1465| 418 | 7004 | 70402 | 
| 22 | 7 django/db/backends/base/schema.py | 1108 | 1137| 233 | 7237 | 70402 | 
| 23 | 8 django/db/migrations/autodetector.py | 1125 | 1146| 231 | 7468 | 82131 | 
| 24 | 8 django/db/models/fields/related.py | 1231 | 1348| 963 | 8431 | 82131 | 
| 25 | **8 django/db/models/base.py** | 1394 | 1449| 491 | 8922 | 82131 | 
| 26 | **8 django/db/models/base.py** | 1500 | 1532| 231 | 9153 | 82131 | 
| 27 | 8 django/db/models/fields/related.py | 1350 | 1422| 616 | 9769 | 82131 | 
| 28 | **8 django/db/models/base.py** | 1345 | 1375| 244 | 10013 | 82131 | 
| 29 | 8 django/db/models/constraints.py | 1 | 29| 213 | 10226 | 82131 | 
| 30 | 8 django/db/models/fields/related.py | 487 | 507| 138 | 10364 | 82131 | 
| 31 | 8 django/db/models/fields/related.py | 255 | 282| 269 | 10633 | 82131 | 
| 32 | 9 django/db/models/fields/__init__.py | 1061 | 1090| 218 | 10851 | 99815 | 
| 33 | **9 django/db/models/base.py** | 1255 | 1285| 259 | 11110 | 99815 | 
| 34 | 9 django/db/models/fields/__init__.py | 338 | 365| 203 | 11313 | 99815 | 
| 35 | 9 django/db/models/fields/__init__.py | 308 | 336| 205 | 11518 | 99815 | 
| 36 | **9 django/db/models/base.py** | 1588 | 1637| 384 | 11902 | 99815 | 
| 37 | 9 django/db/migrations/autodetector.py | 1029 | 1045| 188 | 12090 | 99815 | 
| 38 | 10 django/db/migrations/operations/models.py | 530 | 549| 148 | 12238 | 106385 | 
| 39 | **10 django/db/models/base.py** | 1377 | 1392| 153 | 12391 | 106385 | 
| 40 | 10 django/contrib/admin/checks.py | 273 | 286| 135 | 12526 | 106385 | 
| 41 | 11 django/contrib/contenttypes/fields.py | 110 | 158| 328 | 12854 | 111818 | 
| 42 | 12 django/db/backends/mysql/schema.py | 115 | 129| 201 | 13055 | 113314 | 
| 43 | **12 django/db/models/base.py** | 1791 | 1864| 572 | 13627 | 113314 | 
| 44 | 12 django/db/models/fields/related.py | 156 | 169| 144 | 13771 | 113314 | 
| 45 | **12 django/db/models/base.py** | 1451 | 1474| 176 | 13947 | 113314 | 
| 46 | 12 django/db/models/fields/related.py | 1198 | 1229| 180 | 14127 | 113314 | 
| 47 | 12 django/db/migrations/autodetector.py | 1047 | 1067| 136 | 14263 | 113314 | 
| 48 | 12 django/db/migrations/operations/models.py | 793 | 822| 278 | 14541 | 113314 | 
| 49 | 13 django/core/checks/model_checks.py | 1 | 86| 667 | 15208 | 115101 | 
| 50 | **13 django/db/models/base.py** | 1689 | 1789| 729 | 15937 | 115101 | 
| 51 | 13 django/contrib/admin/checks.py | 226 | 239| 161 | 16098 | 115101 | 
| 52 | 13 django/contrib/admin/checks.py | 998 | 1013| 136 | 16234 | 115101 | 
| 53 | 14 django/contrib/postgres/constraints.py | 78 | 122| 362 | 16596 | 116113 | 
| 54 | 14 django/db/models/fields/__init__.py | 208 | 242| 234 | 16830 | 116113 | 
| 55 | 14 django/db/models/fields/related.py | 190 | 254| 673 | 17503 | 116113 | 
| 56 | 14 django/db/models/fields/related.py | 127 | 154| 201 | 17704 | 116113 | 
| 57 | 14 django/db/models/fields/related.py | 108 | 125| 155 | 17859 | 116113 | 
| 58 | 15 django/contrib/contenttypes/admin.py | 1 | 80| 615 | 18474 | 117138 | 
| 59 | 15 django/contrib/admin/checks.py | 763 | 778| 182 | 18656 | 117138 | 
| 60 | **15 django/db/models/base.py** | 1147 | 1162| 138 | 18794 | 117138 | 
| 61 | 15 django/db/migrations/operations/models.py | 1 | 38| 235 | 19029 | 117138 | 
| 62 | 15 django/db/models/fields/related.py | 1023 | 1070| 368 | 19397 | 117138 | 
| 63 | 15 django/contrib/admin/checks.py | 214 | 224| 127 | 19524 | 117138 | 
| 64 | **15 django/db/models/base.py** | 1476 | 1498| 171 | 19695 | 117138 | 
| 65 | 15 django/db/migrations/operations/models.py | 825 | 860| 347 | 20042 | 117138 | 
| 66 | 15 django/contrib/admin/checks.py | 171 | 212| 325 | 20367 | 117138 | 
| 67 | **15 django/db/models/base.py** | 1194 | 1228| 230 | 20597 | 117138 | 
| 68 | 15 django/forms/models.py | 953 | 986| 367 | 20964 | 117138 | 
| 69 | 15 django/contrib/postgres/constraints.py | 65 | 76| 146 | 21110 | 117138 | 
| 70 | 15 django/db/models/fields/__init__.py | 2299 | 2349| 339 | 21449 | 117138 | 
| 71 | 16 django/db/models/fields/mixins.py | 31 | 57| 173 | 21622 | 117481 | 
| 72 | 16 django/db/models/fields/related.py | 837 | 858| 169 | 21791 | 117481 | 
| 73 | 16 django/contrib/admin/checks.py | 364 | 380| 134 | 21925 | 117481 | 
| 74 | 17 django/db/models/lookups.py | 606 | 639| 141 | 22066 | 122407 | 
| 75 | 18 django/db/migrations/questioner.py | 56 | 81| 220 | 22286 | 124480 | 
| 76 | 18 django/db/backends/base/schema.py | 407 | 421| 174 | 22460 | 124480 | 
| 77 | 18 django/contrib/admin/checks.py | 1059 | 1101| 343 | 22803 | 124480 | 
| 78 | 18 django/db/models/fields/__init__.py | 367 | 393| 199 | 23002 | 124480 | 
| 79 | 18 django/db/models/fields/related.py | 171 | 188| 166 | 23168 | 124480 | 
| 80 | 18 django/contrib/admin/checks.py | 336 | 362| 221 | 23389 | 124480 | 
| 81 | 18 django/contrib/admin/checks.py | 1030 | 1057| 194 | 23583 | 124480 | 
| 82 | 19 django/contrib/contenttypes/checks.py | 1 | 21| 122 | 23705 | 124729 | 
| 83 | 19 django/forms/models.py | 310 | 349| 387 | 24092 | 124729 | 
| 84 | 19 django/core/checks/model_checks.py | 129 | 153| 268 | 24360 | 124729 | 
| 85 | 19 django/core/checks/model_checks.py | 178 | 211| 332 | 24692 | 124729 | 
| 86 | 19 django/db/models/fields/__init__.py | 2352 | 2401| 311 | 25003 | 124729 | 
| 87 | 19 django/contrib/admin/checks.py | 159 | 169| 123 | 25126 | 124729 | 
| 88 | 20 django/db/backends/mysql/introspection.py | 186 | 272| 729 | 25855 | 127161 | 
| 89 | 20 django/contrib/admin/checks.py | 941 | 970| 243 | 26098 | 127161 | 
| 90 | 21 django/db/migrations/operations/__init__.py | 1 | 18| 195 | 26293 | 127356 | 
| 91 | 22 django/contrib/gis/utils/layermapping.py | 284 | 298| 124 | 26417 | 132801 | 
| 92 | 22 django/contrib/admin/checks.py | 288 | 321| 381 | 26798 | 132801 | 
| 93 | 22 django/db/backends/base/schema.py | 531 | 570| 470 | 27268 | 132801 | 
| 94 | 22 django/db/backends/base/schema.py | 1062 | 1084| 199 | 27467 | 132801 | 
| 95 | 23 django/db/backends/mysql/base.py | 290 | 328| 402 | 27869 | 136061 | 
| 96 | 23 django/db/backends/mysql/introspection.py | 273 | 289| 184 | 28053 | 136061 | 
| 97 | 23 django/db/backends/base/schema.py | 1 | 28| 198 | 28251 | 136061 | 
| 98 | 23 django/db/backends/base/schema.py | 386 | 405| 197 | 28448 | 136061 | 
| 99 | 23 django/db/migrations/questioner.py | 143 | 160| 183 | 28631 | 136061 | 
| 100 | **23 django/db/models/base.py** | 1314 | 1343| 205 | 28836 | 136061 | 
| 101 | 23 django/contrib/admin/checks.py | 323 | 334| 138 | 28974 | 136061 | 
| 102 | 23 django/db/models/fields/__init__.py | 613 | 642| 234 | 29208 | 136061 | 
| 103 | 23 django/db/models/fields/__init__.py | 244 | 306| 448 | 29656 | 136061 | 
| 104 | 23 django/contrib/admin/checks.py | 498 | 518| 200 | 29856 | 136061 | 
| 105 | 24 django/contrib/auth/checks.py | 1 | 99| 694 | 30550 | 137533 | 
| 106 | 24 django/contrib/admin/checks.py | 891 | 939| 416 | 30966 | 137533 | 
| 107 | 25 django/core/checks/urls.py | 30 | 50| 165 | 31131 | 138234 | 
| 108 | 25 django/contrib/contenttypes/fields.py | 332 | 354| 185 | 31316 | 138234 | 
| 109 | 26 django/db/backends/sqlite3/base.py | 315 | 399| 823 | 32139 | 144262 | 
| 110 | 27 django/db/backends/sqlite3/schema.py | 384 | 430| 444 | 32583 | 148378 | 
| 111 | 27 django/db/backends/mysql/schema.py | 100 | 113| 148 | 32731 | 148378 | 
| 112 | **27 django/db/models/base.py** | 1287 | 1312| 184 | 32915 | 148378 | 
| 113 | 27 django/db/migrations/autodetector.py | 89 | 101| 116 | 33031 | 148378 | 
| 114 | 27 django/contrib/admin/checks.py | 557 | 592| 304 | 33335 | 148378 | 
| 115 | 27 django/contrib/postgres/constraints.py | 1 | 63| 518 | 33853 | 148378 | 
| 116 | 27 django/contrib/admin/checks.py | 532 | 555| 230 | 34083 | 148378 | 
| 117 | 27 django/core/checks/model_checks.py | 89 | 110| 168 | 34251 | 148378 | 
| 118 | **27 django/db/models/base.py** | 1118 | 1145| 286 | 34537 | 148378 | 
| 119 | 27 django/db/models/fields/related.py | 909 | 929| 178 | 34715 | 148378 | 
| 120 | 28 django/db/migrations/state.py | 591 | 607| 146 | 34861 | 153500 | 
| 121 | 29 django/contrib/auth/migrations/0007_alter_validators_add_error_messages.py | 1 | 25| 137 | 34998 | 153637 | 
| 122 | 29 django/db/models/fields/related.py | 1 | 34| 246 | 35244 | 153637 | 
| 123 | 29 django/db/backends/base/schema.py | 635 | 707| 796 | 36040 | 153637 | 
| 124 | 29 django/db/backends/base/schema.py | 1206 | 1235| 268 | 36308 | 153637 | 
| 125 | 29 django/db/models/fields/related.py | 993 | 1020| 215 | 36523 | 153637 | 
| 126 | 30 django/db/migrations/operations/fields.py | 1 | 37| 241 | 36764 | 156600 | 
| 127 | 30 django/db/migrations/autodetector.py | 1088 | 1123| 312 | 37076 | 156600 | 
| 128 | 30 django/core/checks/model_checks.py | 155 | 176| 263 | 37339 | 156600 | 
| 129 | 30 django/db/migrations/questioner.py | 227 | 240| 123 | 37462 | 156600 | 
| 130 | 30 django/contrib/admin/checks.py | 410 | 422| 137 | 37599 | 156600 | 
| 131 | 30 django/contrib/admin/checks.py | 435 | 456| 191 | 37790 | 156600 | 
| 132 | 31 django/db/backends/mysql/validation.py | 33 | 70| 287 | 38077 | 157120 | 
| 133 | 31 django/db/models/fields/__init__.py | 1715 | 1738| 146 | 38223 | 157120 | 
| 134 | 31 django/forms/models.py | 383 | 411| 240 | 38463 | 157120 | 
| 135 | 31 django/db/backends/base/schema.py | 317 | 332| 154 | 38617 | 157120 | 
| 136 | 32 django/db/models/query_utils.py | 284 | 309| 293 | 38910 | 159832 | 
| 137 | 32 django/contrib/contenttypes/checks.py | 24 | 42| 125 | 39035 | 159832 | 
| 138 | 32 django/contrib/auth/checks.py | 102 | 208| 776 | 39811 | 159832 | 
| 139 | 33 django/db/backends/sqlite3/introspection.py | 224 | 238| 146 | 39957 | 163681 | 
| 140 | 33 django/db/backends/sqlite3/introspection.py | 360 | 438| 750 | 40707 | 163681 | 
| 141 | 34 django/contrib/sites/migrations/0002_alter_domain_unique.py | 1 | 21| 0 | 40707 | 163778 | 
| 142 | 35 django/db/models/deletion.py | 1 | 76| 566 | 41273 | 167601 | 
| 143 | 35 django/contrib/admin/checks.py | 607 | 629| 155 | 41428 | 167601 | 
| 144 | 36 django/db/models/fields/json.py | 42 | 60| 125 | 41553 | 171734 | 
| 145 | 36 django/db/backends/mysql/schema.py | 1 | 37| 387 | 41940 | 171734 | 
| 146 | 36 django/db/backends/sqlite3/introspection.py | 330 | 358| 278 | 42218 | 171734 | 
| 147 | 36 django/contrib/admin/checks.py | 486 | 496| 149 | 42367 | 171734 | 
| 148 | 36 django/db/backends/base/schema.py | 1176 | 1204| 284 | 42651 | 171734 | 
| 149 | 36 django/db/migrations/autodetector.py | 528 | 674| 1111 | 43762 | 171734 | 
| 150 | 36 django/contrib/admin/checks.py | 718 | 728| 115 | 43877 | 171734 | 
| 151 | 36 django/db/backends/sqlite3/introspection.py | 240 | 328| 749 | 44626 | 171734 | 
| 152 | **36 django/db/models/base.py** | 385 | 401| 128 | 44754 | 171734 | 
| 153 | 36 django/db/models/fields/related.py | 746 | 764| 222 | 44976 | 171734 | 
| 154 | 36 django/db/models/fields/related.py | 1602 | 1639| 484 | 45460 | 171734 | 
| 155 | 36 django/contrib/admin/checks.py | 1104 | 1133| 178 | 45638 | 171734 | 
| 156 | 36 django/db/backends/mysql/validation.py | 1 | 31| 239 | 45877 | 171734 | 
| 157 | 36 django/contrib/admin/checks.py | 855 | 877| 217 | 46094 | 171734 | 
| 158 | 37 django/db/migrations/loader.py | 282 | 306| 205 | 46299 | 174776 | 
| 159 | 38 django/db/models/query.py | 1322 | 1331| 114 | 46413 | 191862 | 
| 160 | **38 django/db/models/base.py** | 1 | 50| 326 | 46739 | 191862 | 
| 161 | 38 django/db/models/fields/__init__.py | 982 | 1014| 208 | 46947 | 191862 | 
| 162 | **38 django/db/models/base.py** | 1230 | 1253| 172 | 47119 | 191862 | 
| 163 | 38 django/db/migrations/operations/models.py | 102 | 118| 156 | 47275 | 191862 | 
| 164 | 38 django/db/migrations/questioner.py | 162 | 185| 246 | 47521 | 191862 | 
| 165 | 38 django/db/backends/base/schema.py | 1045 | 1060| 170 | 47691 | 191862 | 
| 166 | 38 django/contrib/admin/checks.py | 520 | 530| 134 | 47825 | 191862 | 
| 167 | 38 django/db/migrations/state.py | 576 | 589| 138 | 47963 | 191862 | 
| 168 | 38 django/db/backends/mysql/schema.py | 88 | 98| 138 | 48101 | 191862 | 
| 169 | 38 django/db/migrations/autodetector.py | 906 | 987| 876 | 48977 | 191862 | 
| 170 | 39 django/db/migrations/executor.py | 280 | 373| 843 | 49820 | 195135 | 
| 171 | 40 django/db/backends/oracle/introspection.py | 207 | 308| 842 | 50662 | 197598 | 
| 172 | 40 django/db/backends/base/schema.py | 708 | 777| 740 | 51402 | 197598 | 
| 173 | 40 django/db/backends/base/schema.py | 572 | 634| 700 | 52102 | 197598 | 
| 174 | 40 django/db/backends/base/schema.py | 31 | 41| 120 | 52222 | 197598 | 
| 175 | 40 django/contrib/admin/checks.py | 632 | 651| 183 | 52405 | 197598 | 
| 176 | 40 django/contrib/admin/checks.py | 594 | 605| 128 | 52533 | 197598 | 
| 177 | 41 django/db/models/fields/related_lookups.py | 62 | 99| 447 | 52980 | 199047 | 
| 178 | 41 django/db/models/fields/__init__.py | 570 | 611| 287 | 53267 | 199047 | 
| 179 | 41 django/db/migrations/autodetector.py | 511 | 527| 186 | 53453 | 199047 | 


### Hint

```
Demonstration
Agreed. We can simply call cls._check_local_fields() for UniqueConstraint's fields. I attached tests.
Tests.
Hello Django Team, My name is Jannah Mandwee, and I am working on my final project for my undergraduate software engineering class (here is the link to the assignment: ​https://web.eecs.umich.edu/~weimerw/481/hw6.html). I have to contribute to an open-source project and was advised to look through easy ticket pickings. I am wondering if it is possible to contribute to this ticket or if there is another ticket you believe would be a better fit for me. Thank you for your help.
Replying to Jannah Mandwee: Hello Django Team, My name is Jannah Mandwee, and I am working on my final project for my undergraduate software engineering class (here is the link to the assignment: ​https://web.eecs.umich.edu/~weimerw/481/hw6.html). I have to contribute to an open-source project and was advised to look through easy ticket pickings. I am wondering if it is possible to contribute to this ticket or if there is another ticket you believe would be a better fit for me. Thank you for your help. Hi Jannah, I'm working in this ticket. You can consult this report: https://code.djangoproject.com/query?status=!closed&easy=1&stage=Accepted&order=priority there are all the tickets marked as easy.
CheckConstraint might have the same bug.
```

## Patch

```diff
diff --git a/django/db/models/base.py b/django/db/models/base.py
--- a/django/db/models/base.py
+++ b/django/db/models/base.py
@@ -1926,6 +1926,12 @@ def _check_constraints(cls, databases):
                         id='models.W038',
                     )
                 )
+            fields = (
+                field
+                for constraint in cls._meta.constraints if isinstance(constraint, UniqueConstraint)
+                for field in constraint.fields
+            )
+            errors.extend(cls._check_local_fields(fields, 'constraints'))
         return errors
 
 

```

## Test Patch

```diff
diff --git a/tests/invalid_models_tests/test_models.py b/tests/invalid_models_tests/test_models.py
--- a/tests/invalid_models_tests/test_models.py
+++ b/tests/invalid_models_tests/test_models.py
@@ -1501,3 +1501,70 @@ class Meta:
                 ]
 
         self.assertEqual(Model.check(databases=self.databases), [])
+
+    def test_unique_constraint_pointing_to_missing_field(self):
+        class Model(models.Model):
+            class Meta:
+                constraints = [models.UniqueConstraint(fields=['missing_field'], name='name')]
+
+        self.assertEqual(Model.check(databases=self.databases), [
+            Error(
+                "'constraints' refers to the nonexistent field "
+                "'missing_field'.",
+                obj=Model,
+                id='models.E012',
+            ),
+        ])
+
+    def test_unique_constraint_pointing_to_m2m_field(self):
+        class Model(models.Model):
+            m2m = models.ManyToManyField('self')
+
+            class Meta:
+                constraints = [models.UniqueConstraint(fields=['m2m'], name='name')]
+
+        self.assertEqual(Model.check(databases=self.databases), [
+            Error(
+                "'constraints' refers to a ManyToManyField 'm2m', but "
+                "ManyToManyFields are not permitted in 'constraints'.",
+                obj=Model,
+                id='models.E013',
+            ),
+        ])
+
+    def test_unique_constraint_pointing_to_non_local_field(self):
+        class Parent(models.Model):
+            field1 = models.IntegerField()
+
+        class Child(Parent):
+            field2 = models.IntegerField()
+
+            class Meta:
+                constraints = [
+                    models.UniqueConstraint(fields=['field2', 'field1'], name='name'),
+                ]
+
+        self.assertEqual(Child.check(databases=self.databases), [
+            Error(
+                "'constraints' refers to field 'field1' which is not local to "
+                "model 'Child'.",
+                hint='This issue may be caused by multi-table inheritance.',
+                obj=Child,
+                id='models.E016',
+            ),
+        ])
+
+    def test_unique_constraint_pointing_to_fk(self):
+        class Target(models.Model):
+            pass
+
+        class Model(models.Model):
+            fk_1 = models.ForeignKey(Target, models.CASCADE, related_name='target_1')
+            fk_2 = models.ForeignKey(Target, models.CASCADE, related_name='target_2')
+
+            class Meta:
+                constraints = [
+                    models.UniqueConstraint(fields=['fk_1_id', 'fk_2'], name='name'),
+                ]
+
+        self.assertEqual(Model.check(databases=self.databases), [])

```


## Code snippets

### 1 - django/db/models/constraints.py:

Start line: 79, End line: 154

```python
class UniqueConstraint(BaseConstraint):
    def __init__(self, *, fields, name, condition=None, deferrable=None):
        if not fields:
            raise ValueError('At least one field is required to define a unique constraint.')
        if not isinstance(condition, (type(None), Q)):
            raise ValueError('UniqueConstraint.condition must be a Q instance.')
        if condition and deferrable:
            raise ValueError(
                'UniqueConstraint with conditions cannot be deferred.'
            )
        if not isinstance(deferrable, (type(None), Deferrable)):
            raise ValueError(
                'UniqueConstraint.deferrable must be a Deferrable instance.'
            )
        self.fields = tuple(fields)
        self.condition = condition
        self.deferrable = deferrable
        super().__init__(name)

    def _get_condition_sql(self, model, schema_editor):
        if self.condition is None:
            return None
        query = Query(model=model, alias_cols=False)
        where = query.build_where(self.condition)
        compiler = query.get_compiler(connection=schema_editor.connection)
        sql, params = where.as_sql(compiler, schema_editor.connection)
        return sql % tuple(schema_editor.quote_value(p) for p in params)

    def constraint_sql(self, model, schema_editor):
        fields = [model._meta.get_field(field_name).column for field_name in self.fields]
        condition = self._get_condition_sql(model, schema_editor)
        return schema_editor._unique_sql(
            model, fields, self.name, condition=condition,
            deferrable=self.deferrable,
        )

    def create_sql(self, model, schema_editor):
        fields = [model._meta.get_field(field_name).column for field_name in self.fields]
        condition = self._get_condition_sql(model, schema_editor)
        return schema_editor._create_unique_sql(
            model, fields, self.name, condition=condition,
            deferrable=self.deferrable,
        )

    def remove_sql(self, model, schema_editor):
        condition = self._get_condition_sql(model, schema_editor)
        return schema_editor._delete_unique_sql(
            model, self.name, condition=condition, deferrable=self.deferrable,
        )

    def __repr__(self):
        return '<%s: fields=%r name=%r%s%s>' % (
            self.__class__.__name__, self.fields, self.name,
            '' if self.condition is None else ' condition=%s' % self.condition,
            '' if self.deferrable is None else ' deferrable=%s' % self.deferrable,
        )

    def __eq__(self, other):
        if isinstance(other, UniqueConstraint):
            return (
                self.name == other.name and
                self.fields == other.fields and
                self.condition == other.condition and
                self.deferrable == other.deferrable
            )
        return super().__eq__(other)

    def deconstruct(self):
        path, args, kwargs = super().deconstruct()
        kwargs['fields'] = self.fields
        if self.condition:
            kwargs['condition'] = self.condition
        if self.deferrable:
            kwargs['deferrable'] = self.deferrable
        return path, args, kwargs
```
### 2 - django/db/models/base.py:

Start line: 1561, End line: 1586

```python
class Model(metaclass=ModelBase):

    @classmethod
    def _check_unique_together(cls):
        """Check the value of "unique_together" option."""
        if not isinstance(cls._meta.unique_together, (tuple, list)):
            return [
                checks.Error(
                    "'unique_together' must be a list or tuple.",
                    obj=cls,
                    id='models.E010',
                )
            ]

        elif any(not isinstance(fields, (tuple, list)) for fields in cls._meta.unique_together):
            return [
                checks.Error(
                    "All 'unique_together' elements must be lists or tuples.",
                    obj=cls,
                    id='models.E011',
                )
            ]

        else:
            errors = []
            for fields in cls._meta.unique_together:
                errors.extend(cls._check_local_fields(fields, "unique_together"))
            return errors
```
### 3 - django/db/models/base.py:

Start line: 1073, End line: 1116

```python
class Model(metaclass=ModelBase):

    def _perform_unique_checks(self, unique_checks):
        errors = {}

        for model_class, unique_check in unique_checks:
            # Try to look up an existing object with the same values as this
            # object's values for all the unique field.

            lookup_kwargs = {}
            for field_name in unique_check:
                f = self._meta.get_field(field_name)
                lookup_value = getattr(self, f.attname)
                # TODO: Handle multiple backends with different feature flags.
                if (lookup_value is None or
                        (lookup_value == '' and connection.features.interprets_empty_strings_as_nulls)):
                    # no value, skip the lookup
                    continue
                if f.primary_key and not self._state.adding:
                    # no need to check for unique primary key when editing
                    continue
                lookup_kwargs[str(field_name)] = lookup_value

            # some fields were skipped, no reason to do the check
            if len(unique_check) != len(lookup_kwargs):
                continue

            qs = model_class._default_manager.filter(**lookup_kwargs)

            # Exclude the current object from the query if we are editing an
            # instance (as opposed to creating a new one)
            # Note that we need to use the pk as defined by model_class, not
            # self.pk. These can be different fields because model inheritance
            # allows single model to have effectively multiple primary keys.
            # Refs #17615.
            model_class_pk = self._get_pk_val(model_class._meta)
            if not self._state.adding and model_class_pk is not None:
                qs = qs.exclude(pk=model_class_pk)
            if qs.exists():
                if len(unique_check) == 1:
                    key = unique_check[0]
                else:
                    key = NON_FIELD_ERRORS
                errors.setdefault(key, []).append(self.unique_error_message(model_class, unique_check))

        return errors
```
### 4 - django/db/backends/base/schema.py:

Start line: 370, End line: 384

```python
class BaseDatabaseSchemaEditor:

    def alter_unique_together(self, model, old_unique_together, new_unique_together):
        """
        Deal with a model changing its unique_together. The input
        unique_togethers must be doubly-nested, not the single-nested
        ["foo", "bar"] format.
        """
        olds = {tuple(fields) for fields in old_unique_together}
        news = {tuple(fields) for fields in new_unique_together}
        # Deleted uniques
        for fields in olds.difference(news):
            self._delete_composed_index(model, fields, {'unique': True}, self.sql_delete_unique)
        # Created uniques
        for fields in news.difference(olds):
            columns = [model._meta.get_field(field).column for field in fields]
            self.execute(self._create_unique_sql(model, columns))
```
### 5 - django/db/models/base.py:

Start line: 1866, End line: 1929

```python
class Model(metaclass=ModelBase):

    @classmethod
    def _check_constraints(cls, databases):
        errors = []
        for db in databases:
            if not router.allow_migrate_model(db, cls):
                continue
            connection = connections[db]
            if not (
                connection.features.supports_table_check_constraints or
                'supports_table_check_constraints' in cls._meta.required_db_features
            ) and any(
                isinstance(constraint, CheckConstraint)
                for constraint in cls._meta.constraints
            ):
                errors.append(
                    checks.Warning(
                        '%s does not support check constraints.' % connection.display_name,
                        hint=(
                            "A constraint won't be created. Silence this "
                            "warning if you don't care about it."
                        ),
                        obj=cls,
                        id='models.W027',
                    )
                )
            if not (
                connection.features.supports_partial_indexes or
                'supports_partial_indexes' in cls._meta.required_db_features
            ) and any(
                isinstance(constraint, UniqueConstraint) and constraint.condition is not None
                for constraint in cls._meta.constraints
            ):
                errors.append(
                    checks.Warning(
                        '%s does not support unique constraints with '
                        'conditions.' % connection.display_name,
                        hint=(
                            "A constraint won't be created. Silence this "
                            "warning if you don't care about it."
                        ),
                        obj=cls,
                        id='models.W036',
                    )
                )
            if not (
                connection.features.supports_deferrable_unique_constraints or
                'supports_deferrable_unique_constraints' in cls._meta.required_db_features
            ) and any(
                isinstance(constraint, UniqueConstraint) and constraint.deferrable is not None
                for constraint in cls._meta.constraints
            ):
                errors.append(
                    checks.Warning(
                        '%s does not support deferrable unique constraints.'
                        % connection.display_name,
                        hint=(
                            "A constraint won't be created. Silence this "
                            "warning if you don't care about it."
                        ),
                        obj=cls,
                        id='models.W038',
                    )
                )
        return errors
```
### 6 - django/db/backends/base/schema.py:

Start line: 1086, End line: 1106

```python
class BaseDatabaseSchemaEditor:

    def _unique_sql(self, model, fields, name, condition=None, deferrable=None):
        if (
            deferrable and
            not self.connection.features.supports_deferrable_unique_constraints
        ):
            return None
        if condition:
            # Databases support conditional unique constraints via a unique
            # index.
            sql = self._create_unique_sql(model, fields, name=name, condition=condition)
            if sql:
                self.deferred_sql.append(sql)
            return None
        constraint = self.sql_unique_constraint % {
            'columns': ', '.join(map(self.quote_name, fields)),
            'deferrable': self._deferrable_constraint_sql(deferrable),
        }
        return self.sql_constraint % {
            'name': self.quote_name(name),
            'constraint': constraint,
        }
```
### 7 - django/db/models/fields/related.py:

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
### 8 - django/db/models/base.py:

Start line: 1164, End line: 1192

```python
class Model(metaclass=ModelBase):

    def unique_error_message(self, model_class, unique_check):
        opts = model_class._meta

        params = {
            'model': self,
            'model_class': model_class,
            'model_name': capfirst(opts.verbose_name),
            'unique_check': unique_check,
        }

        # A unique field
        if len(unique_check) == 1:
            field = opts.get_field(unique_check[0])
            params['field_label'] = capfirst(field.verbose_name)
            return ValidationError(
                message=field.error_messages['unique'],
                code='unique',
                params=params,
            )

        # unique_together
        else:
            field_labels = [capfirst(opts.get_field(f).verbose_name) for f in unique_check]
            params['field_labels'] = get_text_list(field_labels, _('and'))
            return ValidationError(
                message=_("%(model_name)s with this %(field_labels)s already exists."),
                code='unique_together',
                params=params,
            )
```
### 9 - django/forms/models.py:

Start line: 680, End line: 751

```python
class BaseModelFormSet(BaseFormSet):

    def validate_unique(self):
        # Collect unique_checks and date_checks to run from all the forms.
        all_unique_checks = set()
        all_date_checks = set()
        forms_to_delete = self.deleted_forms
        valid_forms = [form for form in self.forms if form.is_valid() and form not in forms_to_delete]
        for form in valid_forms:
            exclude = form._get_validation_exclusions()
            unique_checks, date_checks = form.instance._get_unique_checks(exclude=exclude)
            all_unique_checks.update(unique_checks)
            all_date_checks.update(date_checks)

        errors = []
        # Do each of the unique checks (unique and unique_together)
        for uclass, unique_check in all_unique_checks:
            seen_data = set()
            for form in valid_forms:
                # Get the data for the set of fields that must be unique among the forms.
                row_data = (
                    field if field in self.unique_fields else form.cleaned_data[field]
                    for field in unique_check if field in form.cleaned_data
                )
                # Reduce Model instances to their primary key values
                row_data = tuple(
                    d._get_pk_val() if hasattr(d, '_get_pk_val')
                    # Prevent "unhashable type: list" errors later on.
                    else tuple(d) if isinstance(d, list)
                    else d for d in row_data
                )
                if row_data and None not in row_data:
                    # if we've already seen it then we have a uniqueness failure
                    if row_data in seen_data:
                        # poke error messages into the right places and mark
                        # the form as invalid
                        errors.append(self.get_unique_error_message(unique_check))
                        form._errors[NON_FIELD_ERRORS] = self.error_class([self.get_form_error()])
                        # remove the data from the cleaned_data dict since it was invalid
                        for field in unique_check:
                            if field in form.cleaned_data:
                                del form.cleaned_data[field]
                    # mark the data as seen
                    seen_data.add(row_data)
        # iterate over each of the date checks now
        for date_check in all_date_checks:
            seen_data = set()
            uclass, lookup, field, unique_for = date_check
            for form in valid_forms:
                # see if we have data for both fields
                if (form.cleaned_data and form.cleaned_data[field] is not None and
                        form.cleaned_data[unique_for] is not None):
                    # if it's a date lookup we need to get the data for all the fields
                    if lookup == 'date':
                        date = form.cleaned_data[unique_for]
                        date_data = (date.year, date.month, date.day)
                    # otherwise it's just the attribute on the date/datetime
                    # object
                    else:
                        date_data = (getattr(form.cleaned_data[unique_for], lookup),)
                    data = (form.cleaned_data[field],) + date_data
                    # if we've already seen it then we have a uniqueness failure
                    if data in seen_data:
                        # poke error messages into the right places and mark
                        # the form as invalid
                        errors.append(self.get_date_error_message(date_check))
                        form._errors[NON_FIELD_ERRORS] = self.error_class([self.get_form_error()])
                        # remove the data from the cleaned_data dict since it was invalid
                        del form.cleaned_data[field]
                    # mark the data as seen
                    seen_data.add(data)

        if errors:
            raise ValidationError(errors)
```
### 10 - django/db/models/base.py:

Start line: 1015, End line: 1071

```python
class Model(metaclass=ModelBase):

    def _get_unique_checks(self, exclude=None):
        """
        Return a list of checks to perform. Since validate_unique() could be
        called from a ModelForm, some fields may have been excluded; we can't
        perform a unique check on a model that is missing fields involved
        in that check. Fields that did not validate should also be excluded,
        but they need to be passed in via the exclude argument.
        """
        if exclude is None:
            exclude = []
        unique_checks = []

        unique_togethers = [(self.__class__, self._meta.unique_together)]
        constraints = [(self.__class__, self._meta.total_unique_constraints)]
        for parent_class in self._meta.get_parent_list():
            if parent_class._meta.unique_together:
                unique_togethers.append((parent_class, parent_class._meta.unique_together))
            if parent_class._meta.total_unique_constraints:
                constraints.append(
                    (parent_class, parent_class._meta.total_unique_constraints)
                )

        for model_class, unique_together in unique_togethers:
            for check in unique_together:
                if not any(name in exclude for name in check):
                    # Add the check if the field isn't excluded.
                    unique_checks.append((model_class, tuple(check)))

        for model_class, model_constraints in constraints:
            for constraint in model_constraints:
                if not any(name in exclude for name in constraint.fields):
                    unique_checks.append((model_class, constraint.fields))

        # These are checks for the unique_for_<date/year/month>.
        date_checks = []

        # Gather a list of checks for fields declared as unique and add them to
        # the list of checks.

        fields_with_class = [(self.__class__, self._meta.local_fields)]
        for parent_class in self._meta.get_parent_list():
            fields_with_class.append((parent_class, parent_class._meta.local_fields))

        for model_class, fields in fields_with_class:
            for f in fields:
                name = f.name
                if name in exclude:
                    continue
                if f.unique:
                    unique_checks.append((model_class, (name,)))
                if f.unique_for_date and f.unique_for_date not in exclude:
                    date_checks.append((model_class, 'date', name, f.unique_for_date))
                if f.unique_for_year and f.unique_for_year not in exclude:
                    date_checks.append((model_class, 'year', name, f.unique_for_year))
                if f.unique_for_month and f.unique_for_month not in exclude:
                    date_checks.append((model_class, 'month', name, f.unique_for_month))
        return unique_checks, date_checks
```
### 13 - django/db/models/base.py:

Start line: 1534, End line: 1559

```python
class Model(metaclass=ModelBase):

    @classmethod
    def _check_index_together(cls):
        """Check the value of "index_together" option."""
        if not isinstance(cls._meta.index_together, (tuple, list)):
            return [
                checks.Error(
                    "'index_together' must be a list or tuple.",
                    obj=cls,
                    id='models.E008',
                )
            ]

        elif any(not isinstance(fields, (tuple, list)) for fields in cls._meta.index_together):
            return [
                checks.Error(
                    "All 'index_together' elements must be lists or tuples.",
                    obj=cls,
                    id='models.E009',
                )
            ]

        else:
            errors = []
            for fields in cls._meta.index_together:
                errors.extend(cls._check_local_fields(fields, "index_together"))
            return errors
```
### 17 - django/db/models/base.py:

Start line: 985, End line: 1013

```python
class Model(metaclass=ModelBase):

    def prepare_database_save(self, field):
        if self.pk is None:
            raise ValueError("Unsaved model instance %r cannot be used in an ORM query." % self)
        return getattr(self, field.remote_field.get_related_field().attname)

    def clean(self):
        """
        Hook for doing any extra model-wide validation after clean() has been
        called on every field by self.clean_fields. Any ValidationError raised
        by this method will not be associated with a particular field; it will
        have a special-case association with the field defined by NON_FIELD_ERRORS.
        """
        pass

    def validate_unique(self, exclude=None):
        """
        Check unique constraints on the model and raise ValidationError if any
        failed.
        """
        unique_checks, date_checks = self._get_unique_checks(exclude=exclude)

        errors = self._perform_unique_checks(unique_checks)
        date_errors = self._perform_date_checks(date_checks)

        for k, v in date_errors.items():
            errors.setdefault(k, []).extend(v)

        if errors:
            raise ValidationError(errors)
```
### 18 - django/db/models/base.py:

Start line: 1639, End line: 1687

```python
class Model(metaclass=ModelBase):

    @classmethod
    def _check_local_fields(cls, fields, option):
        from django.db import models

        # In order to avoid hitting the relation tree prematurely, we use our
        # own fields_map instead of using get_field()
        forward_fields_map = {}
        for field in cls._meta._get_fields(reverse=False):
            forward_fields_map[field.name] = field
            if hasattr(field, 'attname'):
                forward_fields_map[field.attname] = field

        errors = []
        for field_name in fields:
            try:
                field = forward_fields_map[field_name]
            except KeyError:
                errors.append(
                    checks.Error(
                        "'%s' refers to the nonexistent field '%s'." % (
                            option, field_name,
                        ),
                        obj=cls,
                        id='models.E012',
                    )
                )
            else:
                if isinstance(field.remote_field, models.ManyToManyRel):
                    errors.append(
                        checks.Error(
                            "'%s' refers to a ManyToManyField '%s', but "
                            "ManyToManyFields are not permitted in '%s'." % (
                                option, field_name, option,
                            ),
                            obj=cls,
                            id='models.E013',
                        )
                    )
                elif field not in cls._meta.local_fields:
                    errors.append(
                        checks.Error(
                            "'%s' refers to field '%s' which is not local to model '%s'."
                            % (option, field_name, cls._meta.object_name),
                            hint="This issue may be caused by multi-table inheritance.",
                            obj=cls,
                            id='models.E016',
                        )
                    )
        return errors
```
### 25 - django/db/models/base.py:

Start line: 1394, End line: 1449

```python
class Model(metaclass=ModelBase):

    @classmethod
    def _check_field_name_clashes(cls):
        """Forbid field shadowing in multi-table inheritance."""
        errors = []
        used_fields = {}  # name or attname -> field

        # Check that multi-inheritance doesn't cause field name shadowing.
        for parent in cls._meta.get_parent_list():
            for f in parent._meta.local_fields:
                clash = used_fields.get(f.name) or used_fields.get(f.attname) or None
                if clash:
                    errors.append(
                        checks.Error(
                            "The field '%s' from parent model "
                            "'%s' clashes with the field '%s' "
                            "from parent model '%s'." % (
                                clash.name, clash.model._meta,
                                f.name, f.model._meta
                            ),
                            obj=cls,
                            id='models.E005',
                        )
                    )
                used_fields[f.name] = f
                used_fields[f.attname] = f

        # Check that fields defined in the model don't clash with fields from
        # parents, including auto-generated fields like multi-table inheritance
        # child accessors.
        for parent in cls._meta.get_parent_list():
            for f in parent._meta.get_fields():
                if f not in used_fields:
                    used_fields[f.name] = f

        for f in cls._meta.local_fields:
            clash = used_fields.get(f.name) or used_fields.get(f.attname) or None
            # Note that we may detect clash between user-defined non-unique
            # field "id" and automatically added unique field "id", both
            # defined at the same model. This special case is considered in
            # _check_id_field and here we ignore it.
            id_conflict = f.name == "id" and clash and clash.name == "id" and clash.model == cls
            if clash and not id_conflict:
                errors.append(
                    checks.Error(
                        "The field '%s' clashes with the field '%s' "
                        "from model '%s'." % (
                            f.name, clash.name, clash.model._meta
                        ),
                        obj=f,
                        id='models.E006',
                    )
                )
            used_fields[f.name] = f
            used_fields[f.attname] = f

        return errors
```
### 26 - django/db/models/base.py:

Start line: 1500, End line: 1532

```python
class Model(metaclass=ModelBase):

    @classmethod
    def _check_property_name_related_field_accessor_clashes(cls):
        errors = []
        property_names = cls._meta._property_names
        related_field_accessors = (
            f.get_attname() for f in cls._meta._get_fields(reverse=False)
            if f.is_relation and f.related_model is not None
        )
        for accessor in related_field_accessors:
            if accessor in property_names:
                errors.append(
                    checks.Error(
                        "The property '%s' clashes with a related field "
                        "accessor." % accessor,
                        obj=cls,
                        id='models.E025',
                    )
                )
        return errors

    @classmethod
    def _check_single_primary_key(cls):
        errors = []
        if sum(1 for f in cls._meta.local_fields if f.primary_key) > 1:
            errors.append(
                checks.Error(
                    "The model cannot have more than one field with "
                    "'primary_key=True'.",
                    obj=cls,
                    id='models.E026',
                )
            )
        return errors
```
### 28 - django/db/models/base.py:

Start line: 1345, End line: 1375

```python
class Model(metaclass=ModelBase):

    @classmethod
    def _check_m2m_through_same_relationship(cls):
        """ Check if no relationship model is used by more than one m2m field.
        """

        errors = []
        seen_intermediary_signatures = []

        fields = cls._meta.local_many_to_many

        # Skip when the target model wasn't found.
        fields = (f for f in fields if isinstance(f.remote_field.model, ModelBase))

        # Skip when the relationship model wasn't found.
        fields = (f for f in fields if isinstance(f.remote_field.through, ModelBase))

        for f in fields:
            signature = (f.remote_field.model, cls, f.remote_field.through, f.remote_field.through_fields)
            if signature in seen_intermediary_signatures:
                errors.append(
                    checks.Error(
                        "The model has two identical many-to-many relations "
                        "through the intermediate model '%s'." %
                        f.remote_field.through._meta.label,
                        obj=cls,
                        id='models.E003',
                    )
                )
            else:
                seen_intermediary_signatures.append(signature)
        return errors
```
### 33 - django/db/models/base.py:

Start line: 1255, End line: 1285

```python
class Model(metaclass=ModelBase):

    @classmethod
    def check(cls, **kwargs):
        errors = [*cls._check_swappable(), *cls._check_model(), *cls._check_managers(**kwargs)]
        if not cls._meta.swapped:
            databases = kwargs.get('databases') or []
            errors += [
                *cls._check_fields(**kwargs),
                *cls._check_m2m_through_same_relationship(),
                *cls._check_long_column_names(databases),
            ]
            clash_errors = (
                *cls._check_id_field(),
                *cls._check_field_name_clashes(),
                *cls._check_model_name_db_lookup_clashes(),
                *cls._check_property_name_related_field_accessor_clashes(),
                *cls._check_single_primary_key(),
            )
            errors.extend(clash_errors)
            # If there are field name clashes, hide consequent column name
            # clashes.
            if not clash_errors:
                errors.extend(cls._check_column_name_clashes())
            errors += [
                *cls._check_index_together(),
                *cls._check_unique_together(),
                *cls._check_indexes(databases),
                *cls._check_ordering(),
                *cls._check_constraints(databases),
            ]

        return errors
```
### 36 - django/db/models/base.py:

Start line: 1588, End line: 1637

```python
class Model(metaclass=ModelBase):

    @classmethod
    def _check_indexes(cls, databases):
        """Check fields, names, and conditions of indexes."""
        errors = []
        for index in cls._meta.indexes:
            # Index name can't start with an underscore or a number, restricted
            # for cross-database compatibility with Oracle.
            if index.name[0] == '_' or index.name[0].isdigit():
                errors.append(
                    checks.Error(
                        "The index name '%s' cannot start with an underscore "
                        "or a number." % index.name,
                        obj=cls,
                        id='models.E033',
                    ),
                )
            if len(index.name) > index.max_name_length:
                errors.append(
                    checks.Error(
                        "The index name '%s' cannot be longer than %d "
                        "characters." % (index.name, index.max_name_length),
                        obj=cls,
                        id='models.E034',
                    ),
                )
        for db in databases:
            if not router.allow_migrate_model(db, cls):
                continue
            connection = connections[db]
            if (
                connection.features.supports_partial_indexes or
                'supports_partial_indexes' in cls._meta.required_db_features
            ):
                continue
            if any(index.condition is not None for index in cls._meta.indexes):
                errors.append(
                    checks.Warning(
                        '%s does not support indexes with conditions.'
                        % connection.display_name,
                        hint=(
                            "Conditions will be ignored. Silence this warning "
                            "if you don't care about it."
                        ),
                        obj=cls,
                        id='models.W037',
                    )
                )
        fields = [field for index in cls._meta.indexes for field, _ in index.fields_orders]
        errors.extend(cls._check_local_fields(fields, 'indexes'))
        return errors
```
### 39 - django/db/models/base.py:

Start line: 1377, End line: 1392

```python
class Model(metaclass=ModelBase):

    @classmethod
    def _check_id_field(cls):
        """Check if `id` field is a primary key."""
        fields = [f for f in cls._meta.local_fields if f.name == 'id' and f != cls._meta.pk]
        # fields is empty or consists of the invalid "id" field
        if fields and not fields[0].primary_key and cls._meta.pk.name == 'id':
            return [
                checks.Error(
                    "'id' can only be used as a field name if the field also "
                    "sets 'primary_key=True'.",
                    obj=cls,
                    id='models.E004',
                )
            ]
        else:
            return []
```
### 43 - django/db/models/base.py:

Start line: 1791, End line: 1864

```python
class Model(metaclass=ModelBase):

    @classmethod
    def _check_long_column_names(cls, databases):
        """
        Check that any auto-generated column names are shorter than the limits
        for each database in which the model will be created.
        """
        if not databases:
            return []
        errors = []
        allowed_len = None
        db_alias = None

        # Find the minimum max allowed length among all specified db_aliases.
        for db in databases:
            # skip databases where the model won't be created
            if not router.allow_migrate_model(db, cls):
                continue
            connection = connections[db]
            max_name_length = connection.ops.max_name_length()
            if max_name_length is None or connection.features.truncates_names:
                continue
            else:
                if allowed_len is None:
                    allowed_len = max_name_length
                    db_alias = db
                elif max_name_length < allowed_len:
                    allowed_len = max_name_length
                    db_alias = db

        if allowed_len is None:
            return errors

        for f in cls._meta.local_fields:
            _, column_name = f.get_attname_column()

            # Check if auto-generated name for the field is too long
            # for the database.
            if f.db_column is None and column_name is not None and len(column_name) > allowed_len:
                errors.append(
                    checks.Error(
                        'Autogenerated column name too long for field "%s". '
                        'Maximum length is "%s" for database "%s".'
                        % (column_name, allowed_len, db_alias),
                        hint="Set the column name manually using 'db_column'.",
                        obj=cls,
                        id='models.E018',
                    )
                )

        for f in cls._meta.local_many_to_many:
            # Skip nonexistent models.
            if isinstance(f.remote_field.through, str):
                continue

            # Check if auto-generated name for the M2M field is too long
            # for the database.
            for m2m in f.remote_field.through._meta.local_fields:
                _, rel_name = m2m.get_attname_column()
                if m2m.db_column is None and rel_name is not None and len(rel_name) > allowed_len:
                    errors.append(
                        checks.Error(
                            'Autogenerated column name too long for M2M field '
                            '"%s". Maximum length is "%s" for database "%s".'
                            % (rel_name, allowed_len, db_alias),
                            hint=(
                                "Use 'through' to create a separate model for "
                                "M2M and then set column_name using 'db_column'."
                            ),
                            obj=cls,
                            id='models.E019',
                        )
                    )

        return errors
```
### 45 - django/db/models/base.py:

Start line: 1451, End line: 1474

```python
class Model(metaclass=ModelBase):

    @classmethod
    def _check_column_name_clashes(cls):
        # Store a list of column names which have already been used by other fields.
        used_column_names = []
        errors = []

        for f in cls._meta.local_fields:
            _, column_name = f.get_attname_column()

            # Ensure the column name is not already in use.
            if column_name and column_name in used_column_names:
                errors.append(
                    checks.Error(
                        "Field '%s' has column name '%s' that is used by "
                        "another field." % (f.name, column_name),
                        hint="Specify a 'db_column' for the field.",
                        obj=cls,
                        id='models.E007'
                    )
                )
            else:
                used_column_names.append(column_name)

        return errors
```
### 50 - django/db/models/base.py:

Start line: 1689, End line: 1789

```python
class Model(metaclass=ModelBase):

    @classmethod
    def _check_ordering(cls):
        """
        Check "ordering" option -- is it a list of strings and do all fields
        exist?
        """
        if cls._meta._ordering_clash:
            return [
                checks.Error(
                    "'ordering' and 'order_with_respect_to' cannot be used together.",
                    obj=cls,
                    id='models.E021',
                ),
            ]

        if cls._meta.order_with_respect_to or not cls._meta.ordering:
            return []

        if not isinstance(cls._meta.ordering, (list, tuple)):
            return [
                checks.Error(
                    "'ordering' must be a tuple or list (even if you want to order by only one field).",
                    obj=cls,
                    id='models.E014',
                )
            ]

        errors = []
        fields = cls._meta.ordering

        # Skip expressions and '?' fields.
        fields = (f for f in fields if isinstance(f, str) and f != '?')

        # Convert "-field" to "field".
        fields = ((f[1:] if f.startswith('-') else f) for f in fields)

        # Separate related fields and non-related fields.
        _fields = []
        related_fields = []
        for f in fields:
            if LOOKUP_SEP in f:
                related_fields.append(f)
            else:
                _fields.append(f)
        fields = _fields

        # Check related fields.
        for field in related_fields:
            _cls = cls
            fld = None
            for part in field.split(LOOKUP_SEP):
                try:
                    # pk is an alias that won't be found by opts.get_field.
                    if part == 'pk':
                        fld = _cls._meta.pk
                    else:
                        fld = _cls._meta.get_field(part)
                    if fld.is_relation:
                        _cls = fld.get_path_info()[-1].to_opts.model
                    else:
                        _cls = None
                except (FieldDoesNotExist, AttributeError):
                    if fld is None or (
                        fld.get_transform(part) is None and fld.get_lookup(part) is None
                    ):
                        errors.append(
                            checks.Error(
                                "'ordering' refers to the nonexistent field, "
                                "related field, or lookup '%s'." % field,
                                obj=cls,
                                id='models.E015',
                            )
                        )

        # Skip ordering on pk. This is always a valid order_by field
        # but is an alias and therefore won't be found by opts.get_field.
        fields = {f for f in fields if f != 'pk'}

        # Check for invalid or nonexistent fields in ordering.
        invalid_fields = []

        # Any field name that is not present in field_names does not exist.
        # Also, ordering by m2m fields is not allowed.
        opts = cls._meta
        valid_fields = set(chain.from_iterable(
            (f.name, f.attname) if not (f.auto_created and not f.concrete) else (f.field.related_query_name(),)
            for f in chain(opts.fields, opts.related_objects)
        ))

        invalid_fields.extend(fields - valid_fields)

        for invalid_field in invalid_fields:
            errors.append(
                checks.Error(
                    "'ordering' refers to the nonexistent field, related "
                    "field, or lookup '%s'." % invalid_field,
                    obj=cls,
                    id='models.E015',
                )
            )
        return errors
```
### 60 - django/db/models/base.py:

Start line: 1147, End line: 1162

```python
class Model(metaclass=ModelBase):

    def date_error_message(self, lookup_type, field_name, unique_for):
        opts = self._meta
        field = opts.get_field(field_name)
        return ValidationError(
            message=field.error_messages['unique_for_date'],
            code='unique_for_date',
            params={
                'model': self,
                'model_name': capfirst(opts.verbose_name),
                'lookup_type': lookup_type,
                'field': field_name,
                'field_label': capfirst(field.verbose_name),
                'date_field': unique_for,
                'date_field_label': capfirst(opts.get_field(unique_for).verbose_name),
            }
        )
```
### 64 - django/db/models/base.py:

Start line: 1476, End line: 1498

```python
class Model(metaclass=ModelBase):

    @classmethod
    def _check_model_name_db_lookup_clashes(cls):
        errors = []
        model_name = cls.__name__
        if model_name.startswith('_') or model_name.endswith('_'):
            errors.append(
                checks.Error(
                    "The model name '%s' cannot start or end with an underscore "
                    "as it collides with the query lookup syntax." % model_name,
                    obj=cls,
                    id='models.E023'
                )
            )
        elif LOOKUP_SEP in model_name:
            errors.append(
                checks.Error(
                    "The model name '%s' cannot contain double underscores as "
                    "it collides with the query lookup syntax." % model_name,
                    obj=cls,
                    id='models.E024'
                )
            )
        return errors
```
### 67 - django/db/models/base.py:

Start line: 1194, End line: 1228

```python
class Model(metaclass=ModelBase):

    def full_clean(self, exclude=None, validate_unique=True):
        """
        Call clean_fields(), clean(), and validate_unique() on the model.
        Raise a ValidationError for any errors that occur.
        """
        errors = {}
        if exclude is None:
            exclude = []
        else:
            exclude = list(exclude)

        try:
            self.clean_fields(exclude=exclude)
        except ValidationError as e:
            errors = e.update_error_dict(errors)

        # Form.clean() is run even if other validation fails, so do the
        # same with Model.clean() for consistency.
        try:
            self.clean()
        except ValidationError as e:
            errors = e.update_error_dict(errors)

        # Run unique checks, but only for fields that passed validation.
        if validate_unique:
            for name in errors:
                if name != NON_FIELD_ERRORS and name not in exclude:
                    exclude.append(name)
            try:
                self.validate_unique(exclude=exclude)
            except ValidationError as e:
                errors = e.update_error_dict(errors)

        if errors:
            raise ValidationError(errors)
```
### 100 - django/db/models/base.py:

Start line: 1314, End line: 1343

```python
class Model(metaclass=ModelBase):

    @classmethod
    def _check_model(cls):
        errors = []
        if cls._meta.proxy:
            if cls._meta.local_fields or cls._meta.local_many_to_many:
                errors.append(
                    checks.Error(
                        "Proxy model '%s' contains model fields." % cls.__name__,
                        id='models.E017',
                    )
                )
        return errors

    @classmethod
    def _check_managers(cls, **kwargs):
        """Perform all manager checks."""
        errors = []
        for manager in cls._meta.managers:
            errors.extend(manager.check(**kwargs))
        return errors

    @classmethod
    def _check_fields(cls, **kwargs):
        """Perform all field checks."""
        errors = []
        for field in cls._meta.local_fields:
            errors.extend(field.check(**kwargs))
        for field in cls._meta.local_many_to_many:
            errors.extend(field.check(from_model=cls, **kwargs))
        return errors
```
### 112 - django/db/models/base.py:

Start line: 1287, End line: 1312

```python
class Model(metaclass=ModelBase):

    @classmethod
    def _check_swappable(cls):
        """Check if the swapped model exists."""
        errors = []
        if cls._meta.swapped:
            try:
                apps.get_model(cls._meta.swapped)
            except ValueError:
                errors.append(
                    checks.Error(
                        "'%s' is not of the form 'app_label.app_name'." % cls._meta.swappable,
                        id='models.E001',
                    )
                )
            except LookupError:
                app_label, model_name = cls._meta.swapped.split('.')
                errors.append(
                    checks.Error(
                        "'%s' references '%s.%s', which has not been "
                        "installed, or is abstract." % (
                            cls._meta.swappable, app_label, model_name
                        ),
                        id='models.E002',
                    )
                )
        return errors
```
### 118 - django/db/models/base.py:

Start line: 1118, End line: 1145

```python
class Model(metaclass=ModelBase):

    def _perform_date_checks(self, date_checks):
        errors = {}
        for model_class, lookup_type, field, unique_for in date_checks:
            lookup_kwargs = {}
            # there's a ticket to add a date lookup, we can remove this special
            # case if that makes it's way in
            date = getattr(self, unique_for)
            if date is None:
                continue
            if lookup_type == 'date':
                lookup_kwargs['%s__day' % unique_for] = date.day
                lookup_kwargs['%s__month' % unique_for] = date.month
                lookup_kwargs['%s__year' % unique_for] = date.year
            else:
                lookup_kwargs['%s__%s' % (unique_for, lookup_type)] = getattr(date, lookup_type)
            lookup_kwargs[field] = getattr(self, field)

            qs = model_class._default_manager.filter(**lookup_kwargs)
            # Exclude the current object from the query if we are editing an
            # instance (as opposed to creating a new one)
            if not self._state.adding and self.pk is not None:
                qs = qs.exclude(pk=self.pk)

            if qs.exists():
                errors.setdefault(field, []).append(
                    self.date_error_message(lookup_type, field, unique_for)
                )
        return errors
```
### 152 - django/db/models/base.py:

Start line: 385, End line: 401

```python
class ModelStateFieldsCacheDescriptor:
    def __get__(self, instance, cls=None):
        if instance is None:
            return self
        res = instance.fields_cache = {}
        return res


class ModelState:
    """Store model instance state."""
    db = None
    # If true, uniqueness validation checks will consider this a new, unsaved
    # object. Necessary for correct validation of new instances of objects with
    # explicit (non-auto) PKs. This impacts validation only; it has no effect
    # on the actual save.
    adding = True
    fields_cache = ModelStateFieldsCacheDescriptor()
```
### 160 - django/db/models/base.py:

Start line: 1, End line: 50

```python
import copy
import inspect
import warnings
from functools import partialmethod
from itertools import chain

import django
from django.apps import apps
from django.conf import settings
from django.core import checks
from django.core.exceptions import (
    NON_FIELD_ERRORS, FieldDoesNotExist, FieldError, MultipleObjectsReturned,
    ObjectDoesNotExist, ValidationError,
)
from django.db import (
    DEFAULT_DB_ALIAS, DJANGO_VERSION_PICKLE_KEY, DatabaseError, connection,
    connections, router, transaction,
)
from django.db.models import (
    NOT_PROVIDED, ExpressionWrapper, IntegerField, Max, Value,
)
from django.db.models.constants import LOOKUP_SEP
from django.db.models.constraints import CheckConstraint, UniqueConstraint
from django.db.models.deletion import CASCADE, Collector
from django.db.models.fields.related import (
    ForeignObjectRel, OneToOneField, lazy_related_operation, resolve_relation,
)
from django.db.models.functions import Coalesce
from django.db.models.manager import Manager
from django.db.models.options import Options
from django.db.models.query import Q
from django.db.models.signals import (
    class_prepared, post_init, post_save, pre_init, pre_save,
)
from django.db.models.utils import make_model_tuple
from django.utils.encoding import force_str
from django.utils.hashable import make_hashable
from django.utils.text import capfirst, get_text_list
from django.utils.translation import gettext_lazy as _


class Deferred:
    def __repr__(self):
        return '<Deferred field>'

    def __str__(self):
        return '<Deferred field>'


DEFERRED = Deferred()
```
### 162 - django/db/models/base.py:

Start line: 1230, End line: 1253

```python
class Model(metaclass=ModelBase):

    def clean_fields(self, exclude=None):
        """
        Clean all fields and raise a ValidationError containing a dict
        of all validation errors if any occur.
        """
        if exclude is None:
            exclude = []

        errors = {}
        for f in self._meta.fields:
            if f.name in exclude:
                continue
            # Skip validation for empty fields with blank=True. The developer
            # is responsible for making sure they have a valid value.
            raw_value = getattr(self, f.attname)
            if f.blank and raw_value in f.empty_values:
                continue
            try:
                setattr(self, f.attname, f.clean(raw_value, self))
            except ValidationError as e:
                errors[f.name] = e.error_list

        if errors:
            raise ValidationError(errors)
```
