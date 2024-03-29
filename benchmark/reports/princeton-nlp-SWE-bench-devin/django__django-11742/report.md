# django__django-11742

| **django/django** | `fee75d2aed4e58ada6567c464cfd22e89dc65f4a` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 528 |
| **Any found context length** | 528 |
| **Avg pos** | 4.0 |
| **Min pos** | 2 |
| **Max pos** | 2 |
| **Top file pos** | 1 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/db/models/fields/__init__.py b/django/db/models/fields/__init__.py
--- a/django/db/models/fields/__init__.py
+++ b/django/db/models/fields/__init__.py
@@ -257,6 +257,7 @@ def is_value(value, accept_promise=True):
                 )
             ]
 
+        choice_max_length = 0
         # Expect [group_name, [value, display]]
         for choices_group in self.choices:
             try:
@@ -270,16 +271,32 @@ def is_value(value, accept_promise=True):
                     for value, human_name in group_choices
                 ):
                     break
+                if self.max_length is not None and group_choices:
+                    choice_max_length = max(
+                        choice_max_length,
+                        *(len(value) for value, _ in group_choices if isinstance(value, str)),
+                    )
             except (TypeError, ValueError):
                 # No groups, choices in the form [value, display]
                 value, human_name = group_name, group_choices
                 if not is_value(value) or not is_value(human_name):
                     break
+                if self.max_length is not None and isinstance(value, str):
+                    choice_max_length = max(choice_max_length, len(value))
 
             # Special case: choices=['ab']
             if isinstance(choices_group, str):
                 break
         else:
+            if self.max_length is not None and choice_max_length > self.max_length:
+                return [
+                    checks.Error(
+                        "'max_length' is too small to fit the longest value "
+                        "in 'choices' (%d characters)." % choice_max_length,
+                        obj=self,
+                        id='fields.E009',
+                    ),
+                ]
             return []
 
         return [

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/db/models/fields/__init__.py | 260 | 260 | 2 | 1 | 528
| django/db/models/fields/__init__.py | 273 | 273 | 2 | 1 | 528


## Problem Statement

```
Add check to ensure max_length fits longest choice.
Description
	
There is currently no check to ensure that Field.max_length is large enough to fit the longest value in Field.choices.
This would be very helpful as often this mistake is not noticed until an attempt is made to save a record with those values that are too long.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 django/db/models/fields/__init__.py** | 947 | 979| 208 | 208 | 17257 | 
| **-> 2 <-** | **1 django/db/models/fields/__init__.py** | 244 | 292| 320 | 528 | 17257 | 
| 3 | 2 django/db/backends/mysql/validation.py | 29 | 61| 246 | 774 | 17745 | 
| 4 | 3 django/db/models/base.py | 1745 | 1816| 565 | 1339 | 32925 | 
| 5 | 4 django/forms/fields.py | 848 | 887| 298 | 1637 | 41869 | 
| 6 | **4 django/db/models/fields/__init__.py** | 1680 | 1703| 146 | 1783 | 41869 | 
| 7 | 4 django/forms/fields.py | 890 | 923| 235 | 2018 | 41869 | 
| 8 | 4 django/forms/fields.py | 759 | 818| 416 | 2434 | 41869 | 
| 9 | 4 django/forms/fields.py | 821 | 845| 177 | 2611 | 41869 | 
| 10 | 5 django/contrib/postgres/validators.py | 1 | 21| 181 | 2792 | 42420 | 
| 11 | **5 django/db/models/fields/__init__.py** | 594 | 623| 234 | 3026 | 42420 | 
| 12 | 6 django/forms/models.py | 1236 | 1264| 224 | 3250 | 53915 | 
| 13 | 7 django/contrib/postgres/fields/mixins.py | 1 | 30| 179 | 3429 | 54095 | 
| 14 | **7 django/db/models/fields/__init__.py** | 1406 | 1428| 119 | 3548 | 54095 | 
| 15 | **7 django/db/models/fields/__init__.py** | 324 | 348| 184 | 3732 | 54095 | 
| 16 | 7 django/forms/models.py | 1337 | 1364| 209 | 3941 | 54095 | 
| 17 | 8 django/contrib/contenttypes/checks.py | 24 | 42| 125 | 4066 | 54344 | 
| 18 | 9 django/contrib/auth/migrations/0008_alter_user_username_max_length.py | 1 | 25| 138 | 4204 | 54482 | 
| 19 | 10 django/core/validators.py | 341 | 386| 308 | 4512 | 58799 | 
| 20 | **10 django/db/models/fields/__init__.py** | 1026 | 1055| 218 | 4730 | 58799 | 
| 21 | 10 django/forms/fields.py | 529 | 545| 180 | 4910 | 58799 | 
| 22 | 11 django/db/migrations/operations/fields.py | 196 | 218| 147 | 5057 | 62066 | 
| 23 | **11 django/db/models/fields/__init__.py** | 294 | 322| 205 | 5262 | 62066 | 
| 24 | **11 django/db/models/fields/__init__.py** | 350 | 376| 199 | 5461 | 62066 | 
| 25 | 12 django/contrib/admin/checks.py | 325 | 351| 221 | 5682 | 71082 | 
| 26 | 13 django/contrib/auth/migrations/0002_alter_permission_name_max_length.py | 1 | 17| 0 | 5682 | 71150 | 
| 27 | **13 django/db/models/fields/__init__.py** | 212 | 242| 204 | 5886 | 71150 | 
| 28 | 13 django/contrib/admin/checks.py | 879 | 927| 416 | 6302 | 71150 | 
| 29 | 13 django/contrib/admin/checks.py | 1087 | 1117| 188 | 6490 | 71150 | 
| 30 | 14 django/db/models/fields/related.py | 1169 | 1200| 180 | 6670 | 84662 | 
| 31 | **14 django/db/models/fields/__init__.py** | 1430 | 1489| 398 | 7068 | 84662 | 
| 32 | **14 django/db/models/fields/__init__.py** | 981 | 1007| 229 | 7297 | 84662 | 
| 33 | 15 django/contrib/postgres/forms/array.py | 1 | 37| 258 | 7555 | 86147 | 
| 34 | **15 django/db/models/fields/__init__.py** | 1552 | 1573| 183 | 7738 | 86147 | 
| 35 | **15 django/db/models/fields/__init__.py** | 809 | 833| 244 | 7982 | 86147 | 
| 36 | 16 django/db/backends/oracle/validation.py | 1 | 23| 146 | 8128 | 86294 | 
| 37 | 16 django/forms/fields.py | 208 | 239| 274 | 8402 | 86294 | 
| 38 | 16 django/contrib/admin/checks.py | 230 | 260| 229 | 8631 | 86294 | 
| 39 | 17 django/db/migrations/questioner.py | 143 | 160| 183 | 8814 | 88368 | 
| 40 | 18 django/contrib/admin/filters.py | 264 | 276| 149 | 8963 | 92102 | 
| 41 | 18 django/forms/fields.py | 469 | 494| 174 | 9137 | 92102 | 
| 42 | 18 django/db/models/base.py | 1601 | 1649| 348 | 9485 | 92102 | 
| 43 | 18 django/contrib/admin/filters.py | 278 | 302| 217 | 9702 | 92102 | 
| 44 | 19 django/contrib/auth/migrations/0009_alter_user_last_name_max_length.py | 1 | 17| 0 | 9702 | 92180 | 
| 45 | 20 django/contrib/postgres/fields/array.py | 1 | 47| 362 | 10064 | 94226 | 
| 46 | **20 django/db/models/fields/__init__.py** | 1705 | 1732| 215 | 10279 | 94226 | 
| 47 | 20 django/contrib/admin/checks.py | 424 | 445| 191 | 10470 | 94226 | 
| 48 | 20 django/contrib/admin/checks.py | 1042 | 1084| 343 | 10813 | 94226 | 
| 49 | 21 django/db/backends/postgresql/operations.py | 209 | 293| 702 | 11515 | 96922 | 
| 50 | **21 django/db/models/fields/__init__.py** | 1958 | 1988| 252 | 11767 | 96922 | 
| 51 | 22 django/contrib/admin/options.py | 188 | 204| 171 | 11938 | 115288 | 
| 52 | **22 django/db/models/fields/__init__.py** | 1357 | 1380| 171 | 12109 | 115288 | 
| 53 | 23 django/contrib/auth/migrations/0010_alter_group_name_max_length.py | 1 | 17| 0 | 12109 | 115365 | 
| 54 | 24 django/db/backends/base/schema.py | 622 | 694| 792 | 12901 | 126665 | 
| 55 | 25 django/forms/widgets.py | 765 | 788| 204 | 13105 | 134731 | 
| 56 | 25 django/contrib/admin/filters.py | 397 | 419| 211 | 13316 | 134731 | 
| 57 | 26 django/contrib/auth/checks.py | 97 | 167| 525 | 13841 | 135904 | 
| 58 | 27 django/db/backends/mysql/schema.py | 79 | 89| 138 | 13979 | 137299 | 
| 59 | 27 django/forms/fields.py | 995 | 1047| 450 | 14429 | 137299 | 
| 60 | 28 django/contrib/auth/password_validation.py | 91 | 115| 189 | 14618 | 138783 | 
| 61 | 28 django/contrib/admin/checks.py | 546 | 581| 303 | 14921 | 138783 | 
| 62 | **28 django/db/models/fields/__init__.py** | 1076 | 1114| 293 | 15214 | 138783 | 
| 63 | 28 django/db/backends/base/schema.py | 251 | 272| 154 | 15368 | 138783 | 
| 64 | 28 django/db/models/base.py | 1651 | 1743| 673 | 16041 | 138783 | 
| 65 | 28 django/contrib/admin/checks.py | 148 | 158| 123 | 16164 | 138783 | 
| 66 | 28 django/db/backends/mysql/validation.py | 1 | 27| 248 | 16412 | 138783 | 
| 67 | 28 django/db/backends/mysql/schema.py | 1 | 77| 751 | 17163 | 138783 | 
| 68 | 28 django/contrib/admin/checks.py | 160 | 201| 325 | 17488 | 138783 | 
| 69 | 28 django/contrib/admin/checks.py | 215 | 228| 161 | 17649 | 138783 | 
| 70 | 28 django/forms/models.py | 752 | 773| 194 | 17843 | 138783 | 
| 71 | 28 django/forms/fields.py | 242 | 259| 162 | 18005 | 138783 | 
| 72 | **28 django/db/models/fields/__init__.py** | 2042 | 2083| 325 | 18330 | 138783 | 
| 73 | 29 django/db/models/options.py | 262 | 294| 343 | 18673 | 145802 | 
| 74 | 29 django/db/backends/base/schema.py | 274 | 295| 174 | 18847 | 145802 | 
| 75 | **29 django/db/models/fields/__init__.py** | 1991 | 2021| 199 | 19046 | 145802 | 
| 76 | 29 django/contrib/admin/checks.py | 596 | 617| 162 | 19208 | 145802 | 
| 77 | 29 django/contrib/admin/checks.py | 509 | 519| 134 | 19342 | 145802 | 
| 78 | 29 django/db/models/fields/related.py | 190 | 254| 673 | 20015 | 145802 | 
| 79 | 29 django/db/models/fields/related.py | 1202 | 1313| 939 | 20954 | 145802 | 
| 80 | 29 django/db/models/fields/related.py | 255 | 282| 269 | 21223 | 145802 | 
| 81 | 29 django/forms/fields.py | 1173 | 1203| 182 | 21405 | 145802 | 
| 82 | 29 django/db/models/fields/related.py | 1315 | 1387| 616 | 22021 | 145802 | 
| 83 | 29 django/db/models/base.py | 1435 | 1458| 176 | 22197 | 145802 | 
| 84 | **29 django/db/models/fields/__init__.py** | 1382 | 1404| 121 | 22318 | 145802 | 
| 85 | 29 django/contrib/admin/checks.py | 487 | 507| 200 | 22518 | 145802 | 
| 86 | 29 django/contrib/admin/checks.py | 203 | 213| 127 | 22645 | 145802 | 
| 87 | 29 django/contrib/admin/checks.py | 791 | 841| 443 | 23088 | 145802 | 
| 88 | 29 django/db/migrations/questioner.py | 162 | 185| 246 | 23334 | 145802 | 
| 89 | 29 django/db/models/fields/related.py | 401 | 419| 165 | 23499 | 145802 | 
| 90 | 29 django/forms/models.py | 1300 | 1335| 286 | 23785 | 145802 | 
| 91 | 29 django/forms/fields.py | 1078 | 1119| 353 | 24138 | 145802 | 
| 92 | 30 django/forms/boundfield.py | 35 | 50| 149 | 24287 | 147923 | 
| 93 | 30 django/db/models/base.py | 1818 | 1842| 175 | 24462 | 147923 | 
| 94 | **30 django/db/models/fields/__init__.py** | 928 | 944| 176 | 24638 | 147923 | 
| 95 | 30 django/contrib/admin/checks.py | 475 | 485| 149 | 24787 | 147923 | 
| 96 | **30 django/db/models/fields/__init__.py** | 551 | 592| 287 | 25074 | 147923 | 
| 97 | 30 django/contrib/admin/checks.py | 706 | 716| 115 | 25189 | 147923 | 
| 98 | 31 django/contrib/auth/migrations/0004_alter_user_username_opts.py | 1 | 24| 150 | 25339 | 148073 | 
| 99 | 31 django/contrib/admin/filters.py | 209 | 226| 190 | 25529 | 148073 | 
| 100 | 31 django/contrib/postgres/fields/array.py | 175 | 194| 143 | 25672 | 148073 | 
| 101 | 31 django/contrib/admin/checks.py | 312 | 323| 138 | 25810 | 148073 | 
| 102 | 31 django/contrib/admin/checks.py | 768 | 789| 190 | 26000 | 148073 | 
| 103 | 31 django/db/backends/base/schema.py | 522 | 561| 470 | 26470 | 148073 | 
| 104 | 32 django/db/backends/oracle/schema.py | 79 | 123| 583 | 27053 | 149829 | 
| 105 | **32 django/db/models/fields/__init__.py** | 2243 | 2293| 339 | 27392 | 149829 | 
| 106 | 32 django/contrib/postgres/fields/array.py | 49 | 71| 172 | 27564 | 149829 | 
| 107 | 32 django/db/migrations/questioner.py | 207 | 224| 171 | 27735 | 149829 | 
| 108 | **32 django/db/models/fields/__init__.py** | 2296 | 2344| 304 | 28039 | 149829 | 
| 109 | 32 django/forms/widgets.py | 671 | 702| 261 | 28300 | 149829 | 
| 110 | 32 django/db/backends/oracle/schema.py | 57 | 77| 249 | 28549 | 149829 | 
| 111 | **32 django/db/models/fields/__init__.py** | 2024 | 2040| 185 | 28734 | 149829 | 
| 112 | 32 django/contrib/admin/checks.py | 277 | 310| 381 | 29115 | 149829 | 
| 113 | 32 django/contrib/admin/checks.py | 399 | 411| 137 | 29252 | 149829 | 
| 114 | 32 django/forms/fields.py | 547 | 566| 171 | 29423 | 149829 | 
| 115 | 32 django/contrib/postgres/fields/array.py | 156 | 173| 146 | 29569 | 149829 | 
| 116 | 32 django/forms/models.py | 1128 | 1153| 226 | 29795 | 149829 | 
| 117 | 32 django/contrib/admin/checks.py | 521 | 544| 230 | 30025 | 149829 | 
| 118 | 32 django/contrib/postgres/forms/array.py | 62 | 102| 248 | 30273 | 149829 | 
| 119 | 33 django/contrib/postgres/fields/ranges.py | 43 | 87| 330 | 30603 | 151815 | 
| 120 | 34 django/contrib/auth/migrations/0007_alter_validators_add_error_messages.py | 1 | 25| 137 | 30740 | 151952 | 
| 121 | 34 django/contrib/admin/checks.py | 262 | 275| 135 | 30875 | 151952 | 
| 122 | 35 django/db/models/fields/reverse_related.py | 117 | 130| 132 | 31007 | 154066 | 
| 123 | **35 django/db/models/fields/__init__.py** | 1160 | 1178| 180 | 31187 | 154066 | 
| 124 | 35 django/db/models/fields/related.py | 1611 | 1645| 266 | 31453 | 154066 | 
| 125 | 35 django/db/backends/base/schema.py | 765 | 805| 511 | 31964 | 154066 | 
| 126 | 35 django/forms/models.py | 1221 | 1234| 176 | 32140 | 154066 | 
| 127 | 35 django/db/models/fields/related.py | 127 | 154| 202 | 32342 | 154066 | 
| 128 | 35 django/contrib/admin/checks.py | 583 | 594| 128 | 32470 | 154066 | 
| 129 | 35 django/db/models/base.py | 1149 | 1177| 213 | 32683 | 154066 | 
| 130 | **35 django/db/models/fields/__init__.py** | 1 | 85| 678 | 33361 | 154066 | 
| 131 | 35 django/db/models/base.py | 1058 | 1101| 404 | 33765 | 154066 | 
| 132 | 36 django/contrib/postgres/forms/ranges.py | 66 | 110| 284 | 34049 | 154770 | 
| 133 | 37 django/db/models/constraints.py | 30 | 66| 309 | 34358 | 155769 | 
| 134 | **37 django/db/models/fields/__init__.py** | 1216 | 1257| 332 | 34690 | 155769 | 
| 135 | 38 django/db/backends/mysql/operations.py | 189 | 232| 329 | 35019 | 158896 | 
| 136 | 38 django/db/backends/base/schema.py | 1 | 37| 287 | 35306 | 158896 | 
| 137 | 39 django/contrib/auth/migrations/0003_alter_user_email_max_length.py | 1 | 17| 0 | 35306 | 158974 | 
| 138 | 39 django/contrib/admin/checks.py | 718 | 749| 229 | 35535 | 158974 | 
| 139 | 39 django/contrib/admin/filters.py | 246 | 261| 154 | 35689 | 158974 | 
| 140 | 39 django/db/models/base.py | 1240 | 1269| 242 | 35931 | 158974 | 
| 141 | **39 django/db/models/fields/__init__.py** | 1058 | 1074| 175 | 36106 | 158974 | 
| 142 | 39 django/forms/fields.py | 323 | 349| 227 | 36333 | 158974 | 
| 143 | 39 django/core/validators.py | 389 | 416| 224 | 36557 | 158974 | 
| 144 | **39 django/db/models/fields/__init__.py** | 1010 | 1023| 104 | 36661 | 158974 | 
| 145 | 40 django/db/models/sql/query.py | 2223 | 2239| 177 | 36838 | 180508 | 
| 146 | 40 django/db/models/base.py | 1545 | 1570| 183 | 37021 | 180508 | 
| 147 | 40 django/db/migrations/questioner.py | 56 | 81| 220 | 37241 | 180508 | 
| 148 | 40 django/db/models/fields/related.py | 108 | 125| 155 | 37396 | 180508 | 
| 149 | **40 django/db/models/fields/__init__.py** | 2156 | 2176| 163 | 37559 | 180508 | 
| 150 | 40 django/db/models/fields/related.py | 824 | 845| 169 | 37728 | 180508 | 
| 151 | 40 django/contrib/admin/checks.py | 867 | 877| 129 | 37857 | 180508 | 
| 152 | 40 django/db/models/base.py | 1378 | 1433| 491 | 38348 | 180508 | 
| 153 | **40 django/db/models/fields/__init__.py** | 1576 | 1601| 206 | 38554 | 180508 | 
| 154 | 40 django/contrib/postgres/fields/ranges.py | 111 | 174| 362 | 38916 | 180508 | 
| 155 | 40 django/forms/fields.py | 351 | 368| 160 | 39076 | 180508 | 
| 156 | 40 django/forms/models.py | 1267 | 1298| 266 | 39342 | 180508 | 
| 157 | 40 django/forms/models.py | 679 | 750| 732 | 40074 | 180508 | 
| 158 | 41 django/contrib/admin/helpers.py | 123 | 149| 220 | 40294 | 183701 | 
| 159 | **41 django/db/models/fields/__init__.py** | 1734 | 1779| 279 | 40573 | 183701 | 
| 160 | 41 django/contrib/postgres/forms/ranges.py | 1 | 63| 419 | 40992 | 183701 | 
| 161 | 41 django/db/models/base.py | 969 | 997| 230 | 41222 | 183701 | 
| 162 | 41 django/contrib/postgres/fields/ranges.py | 238 | 300| 348 | 41570 | 183701 | 
| 163 | 42 django/contrib/contenttypes/fields.py | 111 | 159| 328 | 41898 | 189113 | 
| 164 | 42 django/contrib/admin/filters.py | 422 | 430| 107 | 42005 | 189113 | 
| 165 | 42 django/db/models/fields/related.py | 156 | 169| 144 | 42149 | 189113 | 
| 166 | 42 django/forms/models.py | 309 | 348| 387 | 42536 | 189113 | 
| 167 | 42 django/contrib/admin/checks.py | 843 | 865| 217 | 42753 | 189113 | 
| 168 | 42 django/contrib/postgres/validators.py | 24 | 77| 370 | 43123 | 189113 | 
| 169 | 42 django/db/models/fields/related.py | 487 | 507| 138 | 43261 | 189113 | 
| 170 | 42 django/contrib/admin/checks.py | 641 | 668| 232 | 43493 | 189113 | 
| 171 | 42 django/db/models/base.py | 1329 | 1359| 244 | 43737 | 189113 | 
| 172 | 42 django/db/models/base.py | 1484 | 1516| 231 | 43968 | 189113 | 
| 173 | 42 django/contrib/admin/options.py | 368 | 420| 504 | 44472 | 189113 | 
| 174 | 43 django/contrib/postgres/fields/jsonb.py | 30 | 83| 361 | 44833 | 190330 | 
| 175 | 43 django/db/backends/base/schema.py | 695 | 764| 752 | 45585 | 190330 | 
| 176 | 43 django/contrib/admin/checks.py | 1013 | 1040| 204 | 45789 | 190330 | 
| 177 | 43 django/contrib/admin/options.py | 422 | 465| 350 | 46139 | 190330 | 
| 178 | 43 django/forms/models.py | 1156 | 1219| 520 | 46659 | 190330 | 
| 179 | 44 django/core/checks/model_checks.py | 129 | 153| 268 | 46927 | 192117 | 
| 180 | 44 django/contrib/admin/helpers.py | 355 | 364| 134 | 47061 | 192117 | 
| 181 | 45 django/contrib/contenttypes/admin.py | 1 | 80| 615 | 47676 | 193142 | 
| 182 | 45 django/contrib/admin/checks.py | 447 | 473| 190 | 47866 | 193142 | 


## Patch

```diff
diff --git a/django/db/models/fields/__init__.py b/django/db/models/fields/__init__.py
--- a/django/db/models/fields/__init__.py
+++ b/django/db/models/fields/__init__.py
@@ -257,6 +257,7 @@ def is_value(value, accept_promise=True):
                 )
             ]
 
+        choice_max_length = 0
         # Expect [group_name, [value, display]]
         for choices_group in self.choices:
             try:
@@ -270,16 +271,32 @@ def is_value(value, accept_promise=True):
                     for value, human_name in group_choices
                 ):
                     break
+                if self.max_length is not None and group_choices:
+                    choice_max_length = max(
+                        choice_max_length,
+                        *(len(value) for value, _ in group_choices if isinstance(value, str)),
+                    )
             except (TypeError, ValueError):
                 # No groups, choices in the form [value, display]
                 value, human_name = group_name, group_choices
                 if not is_value(value) or not is_value(human_name):
                     break
+                if self.max_length is not None and isinstance(value, str):
+                    choice_max_length = max(choice_max_length, len(value))
 
             # Special case: choices=['ab']
             if isinstance(choices_group, str):
                 break
         else:
+            if self.max_length is not None and choice_max_length > self.max_length:
+                return [
+                    checks.Error(
+                        "'max_length' is too small to fit the longest value "
+                        "in 'choices' (%d characters)." % choice_max_length,
+                        obj=self,
+                        id='fields.E009',
+                    ),
+                ]
             return []
 
         return [

```

## Test Patch

```diff
diff --git a/tests/invalid_models_tests/test_ordinary_fields.py b/tests/invalid_models_tests/test_ordinary_fields.py
--- a/tests/invalid_models_tests/test_ordinary_fields.py
+++ b/tests/invalid_models_tests/test_ordinary_fields.py
@@ -304,6 +304,32 @@ class Model(models.Model):
 
         self.assertEqual(Model._meta.get_field('field').check(), [])
 
+    def test_choices_in_max_length(self):
+        class Model(models.Model):
+            field = models.CharField(
+                max_length=2, choices=[
+                    ('ABC', 'Value Too Long!'), ('OK', 'Good')
+                ],
+            )
+            group = models.CharField(
+                max_length=2, choices=[
+                    ('Nested', [('OK', 'Good'), ('Longer', 'Longer')]),
+                    ('Grouped', [('Bad', 'Bad')]),
+                ],
+            )
+
+        for name, choice_max_length in (('field', 3), ('group', 6)):
+            with self.subTest(name):
+                field = Model._meta.get_field(name)
+                self.assertEqual(field.check(), [
+                    Error(
+                        "'max_length' is too small to fit the longest value "
+                        "in 'choices' (%d characters)." % choice_max_length,
+                        obj=field,
+                        id='fields.E009',
+                    ),
+                ])
+
     def test_bad_db_index_value(self):
         class Model(models.Model):
             field = models.CharField(max_length=10, db_index='bad')

```


## Code snippets

### 1 - django/db/models/fields/__init__.py:

Start line: 947, End line: 979

```python
class CharField(Field):
    description = _("String (up to %(max_length)s)")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.validators.append(validators.MaxLengthValidator(self.max_length))

    def check(self, **kwargs):
        return [
            *super().check(**kwargs),
            *self._check_max_length_attribute(**kwargs),
        ]

    def _check_max_length_attribute(self, **kwargs):
        if self.max_length is None:
            return [
                checks.Error(
                    "CharFields must define a 'max_length' attribute.",
                    obj=self,
                    id='fields.E120',
                )
            ]
        elif (not isinstance(self.max_length, int) or isinstance(self.max_length, bool) or
                self.max_length <= 0):
            return [
                checks.Error(
                    "'max_length' must be a positive integer.",
                    obj=self,
                    id='fields.E121',
                )
            ]
        else:
            return []
```
### 2 - django/db/models/fields/__init__.py:

Start line: 244, End line: 292

```python
@total_ordering
class Field(RegisterLookupMixin):

    def _check_choices(self):
        if not self.choices:
            return []

        def is_value(value, accept_promise=True):
            return isinstance(value, (str, Promise) if accept_promise else str) or not is_iterable(value)

        if is_value(self.choices, accept_promise=False):
            return [
                checks.Error(
                    "'choices' must be an iterable (e.g., a list or tuple).",
                    obj=self,
                    id='fields.E004',
                )
            ]

        # Expect [group_name, [value, display]]
        for choices_group in self.choices:
            try:
                group_name, group_choices = choices_group
            except (TypeError, ValueError):
                # Containing non-pairs
                break
            try:
                if not all(
                    is_value(value) and is_value(human_name)
                    for value, human_name in group_choices
                ):
                    break
            except (TypeError, ValueError):
                # No groups, choices in the form [value, display]
                value, human_name = group_name, group_choices
                if not is_value(value) or not is_value(human_name):
                    break

            # Special case: choices=['ab']
            if isinstance(choices_group, str):
                break
        else:
            return []

        return [
            checks.Error(
                "'choices' must be an iterable containing "
                "(actual value, human readable name) tuples.",
                obj=self,
                id='fields.E005',
            )
        ]
```
### 3 - django/db/backends/mysql/validation.py:

Start line: 29, End line: 61

```python
class DatabaseValidation(BaseDatabaseValidation):

    def check_field_type(self, field, field_type):
        """
        MySQL has the following field length restriction:
        No character (varchar) fields can have a length exceeding 255
        characters if they have a unique index on them.
        MySQL doesn't support a database index on some data types.
        """
        errors = []
        if (field_type.startswith('varchar') and field.unique and
                (field.max_length is None or int(field.max_length) > 255)):
            errors.append(
                checks.Error(
                    'MySQL does not allow unique CharFields to have a max_length > 255.',
                    obj=field,
                    id='mysql.E001',
                )
            )

        if field.db_index and field_type.lower() in self.connection._limited_data_types:
            errors.append(
                checks.Warning(
                    '%s does not support a database index on %s columns.'
                    % (self.connection.display_name, field_type),
                    hint=(
                        "An index won't be created. Silence this warning if "
                        "you don't care about it."
                    ),
                    obj=field,
                    id='fields.W162',
                )
            )
        return errors
```
### 4 - django/db/models/base.py:

Start line: 1745, End line: 1816

```python
class Model(metaclass=ModelBase):

    @classmethod
    def _check_long_column_names(cls):
        """
        Check that any auto-generated column names are shorter than the limits
        for each database in which the model will be created.
        """
        errors = []
        allowed_len = None
        db_alias = None

        # Find the minimum max allowed length among all specified db_aliases.
        for db in settings.DATABASES:
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
### 5 - django/forms/fields.py:

Start line: 848, End line: 887

```python
class MultipleChoiceField(ChoiceField):
    hidden_widget = MultipleHiddenInput
    widget = SelectMultiple
    default_error_messages = {
        'invalid_choice': _('Select a valid choice. %(value)s is not one of the available choices.'),
        'invalid_list': _('Enter a list of values.'),
    }

    def to_python(self, value):
        if not value:
            return []
        elif not isinstance(value, (list, tuple)):
            raise ValidationError(self.error_messages['invalid_list'], code='invalid_list')
        return [str(val) for val in value]

    def validate(self, value):
        """Validate that the input is a list or tuple."""
        if self.required and not value:
            raise ValidationError(self.error_messages['required'], code='required')
        # Validate that each value in the value list is in self.choices.
        for val in value:
            if not self.valid_value(val):
                raise ValidationError(
                    self.error_messages['invalid_choice'],
                    code='invalid_choice',
                    params={'value': val},
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
### 6 - django/db/models/fields/__init__.py:

Start line: 1680, End line: 1703

```python
class IntegerField(Field):
    empty_strings_allowed = False
    default_error_messages = {
        'invalid': _('“%(value)s” value must be an integer.'),
    }
    description = _("Integer")

    def check(self, **kwargs):
        return [
            *super().check(**kwargs),
            *self._check_max_length_warning(),
        ]

    def _check_max_length_warning(self):
        if self.max_length is not None:
            return [
                checks.Warning(
                    "'max_length' is ignored when used with %s." % self.__class__.__name__,
                    hint="Remove 'max_length' from field",
                    obj=self,
                    id='fields.W122',
                )
            ]
        return []
```
### 7 - django/forms/fields.py:

Start line: 890, End line: 923

```python
class TypedMultipleChoiceField(MultipleChoiceField):
    def __init__(self, *, coerce=lambda val: val, **kwargs):
        self.coerce = coerce
        self.empty_value = kwargs.pop('empty_value', [])
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
                    self.error_messages['invalid_choice'],
                    code='invalid_choice',
                    params={'value': choice},
                )
        return new_value

    def clean(self, value):
        value = super().clean(value)
        return self._coerce(value)

    def validate(self, value):
        if value != self.empty_value:
            super().validate(value)
        elif self.required:
            raise ValidationError(self.error_messages['required'], code='required')
```
### 8 - django/forms/fields.py:

Start line: 759, End line: 818

```python
class ChoiceField(Field):
    widget = Select
    default_error_messages = {
        'invalid_choice': _('Select a valid choice. %(value)s is not one of the available choices.'),
    }

    def __init__(self, *, choices=(), **kwargs):
        super().__init__(**kwargs)
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
            return ''
        return str(value)

    def validate(self, value):
        """Validate that the input is in self.choices."""
        super().validate(value)
        if value and not self.valid_value(value):
            raise ValidationError(
                self.error_messages['invalid_choice'],
                code='invalid_choice',
                params={'value': value},
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
### 9 - django/forms/fields.py:

Start line: 821, End line: 845

```python
class TypedChoiceField(ChoiceField):
    def __init__(self, *, coerce=lambda val: val, empty_value='', **kwargs):
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
                self.error_messages['invalid_choice'],
                code='invalid_choice',
                params={'value': value},
            )
        return value

    def clean(self, value):
        value = super().clean(value)
        return self._coerce(value)
```
### 10 - django/contrib/postgres/validators.py:

Start line: 1, End line: 21

```python
from django.core.exceptions import ValidationError
from django.core.validators import (
    MaxLengthValidator, MaxValueValidator, MinLengthValidator,
    MinValueValidator,
)
from django.utils.deconstruct import deconstructible
from django.utils.translation import gettext_lazy as _, ngettext_lazy


class ArrayMaxLengthValidator(MaxLengthValidator):
    message = ngettext_lazy(
        'List contains %(show_value)d item, it should contain no more than %(limit_value)d.',
        'List contains %(show_value)d items, it should contain no more than %(limit_value)d.',
        'limit_value')


class ArrayMinLengthValidator(MinLengthValidator):
    message = ngettext_lazy(
        'List contains %(show_value)d item, it should contain no fewer than %(limit_value)d.',
        'List contains %(show_value)d items, it should contain no fewer than %(limit_value)d.',
        'limit_value')
```
### 11 - django/db/models/fields/__init__.py:

Start line: 594, End line: 623

```python
@total_ordering
class Field(RegisterLookupMixin):

    def validate(self, value, model_instance):
        """
        Validate value and raise ValidationError if necessary. Subclasses
        should override this to provide validation logic.
        """
        if not self.editable:
            # Skip validation for non-editable fields.
            return

        if self.choices is not None and value not in self.empty_values:
            for option_key, option_value in self.choices:
                if isinstance(option_value, (list, tuple)):
                    # This is an optgroup, so look inside the group for
                    # options.
                    for optgroup_key, optgroup_value in option_value:
                        if value == optgroup_key:
                            return
                elif value == option_key:
                    return
            raise exceptions.ValidationError(
                self.error_messages['invalid_choice'],
                code='invalid_choice',
                params={'value': value},
            )

        if value is None and not self.null:
            raise exceptions.ValidationError(self.error_messages['null'], code='null')

        if not self.blank and value in self.empty_values:
            raise exceptions.ValidationError(self.error_messages['blank'], code='blank')
```
### 14 - django/db/models/fields/__init__.py:

Start line: 1406, End line: 1428

```python
class DecimalField(Field):

    def _check_max_digits(self):
        try:
            max_digits = int(self.max_digits)
            if max_digits <= 0:
                raise ValueError()
        except TypeError:
            return [
                checks.Error(
                    "DecimalFields must define a 'max_digits' attribute.",
                    obj=self,
                    id='fields.E132',
                )
            ]
        except ValueError:
            return [
                checks.Error(
                    "'max_digits' must be a positive integer.",
                    obj=self,
                    id='fields.E133',
                )
            ]
        else:
            return []
```
### 15 - django/db/models/fields/__init__.py:

Start line: 324, End line: 348

```python
@total_ordering
class Field(RegisterLookupMixin):

    def _check_backend_specific_checks(self, **kwargs):
        app_label = self.model._meta.app_label
        for db in connections:
            if router.allow_migrate(db, app_label, model_name=self.model._meta.model_name):
                return connections[db].validation.check_field(self, **kwargs)
        return []

    def _check_validators(self):
        errors = []
        for i, validator in enumerate(self.validators):
            if not callable(validator):
                errors.append(
                    checks.Error(
                        "All 'validators' must be callable.",
                        hint=(
                            "validators[{i}] ({repr}) isn't a function or "
                            "instance of a validator class.".format(
                                i=i, repr=repr(validator),
                            )
                        ),
                        obj=self,
                        id='fields.E008',
                    )
                )
        return errors
```
### 20 - django/db/models/fields/__init__.py:

Start line: 1026, End line: 1055

```python
class DateTimeCheckMixin:

    def check(self, **kwargs):
        return [
            *super().check(**kwargs),
            *self._check_mutually_exclusive_options(),
            *self._check_fix_default_value(),
        ]

    def _check_mutually_exclusive_options(self):
        # auto_now, auto_now_add, and default are mutually exclusive
        # options. The use of more than one of these options together
        # will trigger an Error
        mutually_exclusive_options = [self.auto_now_add, self.auto_now, self.has_default()]
        enabled_options = [option not in (None, False) for option in mutually_exclusive_options].count(True)
        if enabled_options > 1:
            return [
                checks.Error(
                    "The options auto_now, auto_now_add, and default "
                    "are mutually exclusive. Only one of these options "
                    "may be present.",
                    obj=self,
                    id='fields.E160',
                )
            ]
        else:
            return []

    def _check_fix_default_value(self):
        return []
```
### 23 - django/db/models/fields/__init__.py:

Start line: 294, End line: 322

```python
@total_ordering
class Field(RegisterLookupMixin):

    def _check_db_index(self):
        if self.db_index not in (None, True, False):
            return [
                checks.Error(
                    "'db_index' must be None, True or False.",
                    obj=self,
                    id='fields.E006',
                )
            ]
        else:
            return []

    def _check_null_allowed_for_primary_keys(self):
        if (self.primary_key and self.null and
                not connection.features.interprets_empty_strings_as_nulls):
            # We cannot reliably check this for backends like Oracle which
            # consider NULL and '' to be equal (and thus set up
            # character-based fields a little differently).
            return [
                checks.Error(
                    'Primary keys must not have null=True.',
                    hint=('Set null=False on the field, or '
                          'remove primary_key=True argument.'),
                    obj=self,
                    id='fields.E007',
                )
            ]
        else:
            return []
```
### 24 - django/db/models/fields/__init__.py:

Start line: 350, End line: 376

```python
@total_ordering
class Field(RegisterLookupMixin):

    def _check_deprecation_details(self):
        if self.system_check_removed_details is not None:
            return [
                checks.Error(
                    self.system_check_removed_details.get(
                        'msg',
                        '%s has been removed except for support in historical '
                        'migrations.' % self.__class__.__name__
                    ),
                    hint=self.system_check_removed_details.get('hint'),
                    obj=self,
                    id=self.system_check_removed_details.get('id', 'fields.EXXX'),
                )
            ]
        elif self.system_check_deprecated_details is not None:
            return [
                checks.Warning(
                    self.system_check_deprecated_details.get(
                        'msg',
                        '%s has been deprecated.' % self.__class__.__name__
                    ),
                    hint=self.system_check_deprecated_details.get('hint'),
                    obj=self,
                    id=self.system_check_deprecated_details.get('id', 'fields.WXXX'),
                )
            ]
        return []
```
### 27 - django/db/models/fields/__init__.py:

Start line: 212, End line: 242

```python
@total_ordering
class Field(RegisterLookupMixin):

    def _check_field_name(self):
        """
        Check if field name is valid, i.e. 1) does not end with an
        underscore, 2) does not contain "__" and 3) is not "pk".
        """
        if self.name.endswith('_'):
            return [
                checks.Error(
                    'Field names must not end with an underscore.',
                    obj=self,
                    id='fields.E001',
                )
            ]
        elif LOOKUP_SEP in self.name:
            return [
                checks.Error(
                    'Field names must not contain "%s".' % (LOOKUP_SEP,),
                    obj=self,
                    id='fields.E002',
                )
            ]
        elif self.name == 'pk':
            return [
                checks.Error(
                    "'pk' is a reserved word that cannot be used as a field name.",
                    obj=self,
                    id='fields.E003',
                )
            ]
        else:
            return []
```
### 31 - django/db/models/fields/__init__.py:

Start line: 1430, End line: 1489

```python
class DecimalField(Field):

    def _check_decimal_places_and_max_digits(self, **kwargs):
        if int(self.decimal_places) > int(self.max_digits):
            return [
                checks.Error(
                    "'max_digits' must be greater or equal to 'decimal_places'.",
                    obj=self,
                    id='fields.E134',
                )
            ]
        return []

    @cached_property
    def validators(self):
        return super().validators + [
            validators.DecimalValidator(self.max_digits, self.decimal_places)
        ]

    @cached_property
    def context(self):
        return decimal.Context(prec=self.max_digits)

    def deconstruct(self):
        name, path, args, kwargs = super().deconstruct()
        if self.max_digits is not None:
            kwargs['max_digits'] = self.max_digits
        if self.decimal_places is not None:
            kwargs['decimal_places'] = self.decimal_places
        return name, path, args, kwargs

    def get_internal_type(self):
        return "DecimalField"

    def to_python(self, value):
        if value is None:
            return value
        if isinstance(value, float):
            return self.context.create_decimal_from_float(value)
        try:
            return decimal.Decimal(value)
        except decimal.InvalidOperation:
            raise exceptions.ValidationError(
                self.error_messages['invalid'],
                code='invalid',
                params={'value': value},
            )

    def get_db_prep_save(self, value, connection):
        return connection.ops.adapt_decimalfield_value(self.to_python(value), self.max_digits, self.decimal_places)

    def get_prep_value(self, value):
        value = super().get_prep_value(value)
        return self.to_python(value)

    def formfield(self, **kwargs):
        return super().formfield(**{
            'max_digits': self.max_digits,
            'decimal_places': self.decimal_places,
            'form_class': forms.DecimalField,
            **kwargs,
        })
```
### 32 - django/db/models/fields/__init__.py:

Start line: 981, End line: 1007

```python
class CharField(Field):

    def cast_db_type(self, connection):
        if self.max_length is None:
            return connection.ops.cast_char_field_without_max_length
        return super().cast_db_type(connection)

    def get_internal_type(self):
        return "CharField"

    def to_python(self, value):
        if isinstance(value, str) or value is None:
            return value
        return str(value)

    def get_prep_value(self, value):
        value = super().get_prep_value(value)
        return self.to_python(value)

    def formfield(self, **kwargs):
        # Passing max_length to forms.CharField means that the value's length
        # will be validated twice. This is considered acceptable since we want
        # the value in the form field (to pass into widget for example).
        defaults = {'max_length': self.max_length}
        # TODO: Handle multiple backends with different feature flags.
        if self.null and not connection.features.interprets_empty_strings_as_nulls:
            defaults['empty_value'] = None
        defaults.update(kwargs)
        return super().formfield(**defaults)
```
### 34 - django/db/models/fields/__init__.py:

Start line: 1552, End line: 1573

```python
class EmailField(CharField):
    default_validators = [validators.validate_email]
    description = _("Email address")

    def __init__(self, *args, **kwargs):
        # max_length=254 to be compliant with RFCs 3696 and 5321
        kwargs.setdefault('max_length', 254)
        super().__init__(*args, **kwargs)

    def deconstruct(self):
        name, path, args, kwargs = super().deconstruct()
        # We do not exclude max_length if it matches default as we want to change
        # the default in future.
        return name, path, args, kwargs

    def formfield(self, **kwargs):
        # As with CharField, this will cause email validation to be performed
        # twice.
        return super().formfield(**{
            'form_class': forms.EmailField,
            **kwargs,
        })
```
### 35 - django/db/models/fields/__init__.py:

Start line: 809, End line: 833

```python
@total_ordering
class Field(RegisterLookupMixin):

    def get_choices(self, include_blank=True, blank_choice=BLANK_CHOICE_DASH, limit_choices_to=None, ordering=()):
        """
        Return choices with a default blank choices included, for use
        as <select> choices for this field.
        """
        if self.choices is not None:
            choices = list(self.choices)
            if include_blank:
                blank_defined = any(choice in ('', None) for choice, _ in self.flatchoices)
                if not blank_defined:
                    choices = blank_choice + choices
            return choices
        rel_model = self.remote_field.model
        limit_choices_to = limit_choices_to or self.get_limit_choices_to()
        choice_func = operator.attrgetter(
            self.remote_field.get_related_field().attname
            if hasattr(self.remote_field, 'get_related_field')
            else 'pk'
        )
        qs = rel_model._default_manager.complex_filter(limit_choices_to)
        if ordering:
            qs = qs.order_by(*ordering)
        return (blank_choice if include_blank else []) + [
            (choice_func(x), str(x)) for x in qs
        ]
```
### 46 - django/db/models/fields/__init__.py:

Start line: 1705, End line: 1732

```python
class IntegerField(Field):

    @cached_property
    def validators(self):
        # These validators can't be added at field initialization time since
        # they're based on values retrieved from `connection`.
        validators_ = super().validators
        internal_type = self.get_internal_type()
        min_value, max_value = connection.ops.integer_field_range(internal_type)
        if min_value is not None and not any(
            (
                isinstance(validator, validators.MinValueValidator) and (
                    validator.limit_value()
                    if callable(validator.limit_value)
                    else validator.limit_value
                ) >= min_value
            ) for validator in validators_
        ):
            validators_.append(validators.MinValueValidator(min_value))
        if max_value is not None and not any(
            (
                isinstance(validator, validators.MaxValueValidator) and (
                    validator.limit_value()
                    if callable(validator.limit_value)
                    else validator.limit_value
                ) <= max_value
            ) for validator in validators_
        ):
            validators_.append(validators.MaxValueValidator(max_value))
        return validators_
```
### 50 - django/db/models/fields/__init__.py:

Start line: 1958, End line: 1988

```python
class SlugField(CharField):
    default_validators = [validators.validate_slug]
    description = _("Slug (up to %(max_length)s)")

    def __init__(self, *args, max_length=50, db_index=True, allow_unicode=False, **kwargs):
        self.allow_unicode = allow_unicode
        if self.allow_unicode:
            self.default_validators = [validators.validate_unicode_slug]
        super().__init__(*args, max_length=max_length, db_index=db_index, **kwargs)

    def deconstruct(self):
        name, path, args, kwargs = super().deconstruct()
        if kwargs.get("max_length") == 50:
            del kwargs['max_length']
        if self.db_index is False:
            kwargs['db_index'] = False
        else:
            del kwargs['db_index']
        if self.allow_unicode is not False:
            kwargs['allow_unicode'] = self.allow_unicode
        return name, path, args, kwargs

    def get_internal_type(self):
        return "SlugField"

    def formfield(self, **kwargs):
        return super().formfield(**{
            'form_class': forms.SlugField,
            'allow_unicode': self.allow_unicode,
            **kwargs,
        })
```
### 52 - django/db/models/fields/__init__.py:

Start line: 1357, End line: 1380

```python
class DecimalField(Field):
    empty_strings_allowed = False
    default_error_messages = {
        'invalid': _('“%(value)s” value must be a decimal number.'),
    }
    description = _("Decimal number")

    def __init__(self, verbose_name=None, name=None, max_digits=None,
                 decimal_places=None, **kwargs):
        self.max_digits, self.decimal_places = max_digits, decimal_places
        super().__init__(verbose_name, name, **kwargs)

    def check(self, **kwargs):
        errors = super().check(**kwargs)

        digits_errors = [
            *self._check_decimal_places(),
            *self._check_max_digits(),
        ]
        if not digits_errors:
            errors.extend(self._check_decimal_places_and_max_digits(**kwargs))
        else:
            errors.extend(digits_errors)
        return errors
```
### 62 - django/db/models/fields/__init__.py:

Start line: 1076, End line: 1114

```python
class DateField(DateTimeCheckMixin, Field):

    def _check_fix_default_value(self):
        """
        Warn that using an actual date or datetime value is probably wrong;
        it's only evaluated on server startup.
        """
        if not self.has_default():
            return []

        now = timezone.now()
        if not timezone.is_naive(now):
            now = timezone.make_naive(now, timezone.utc)
        value = self.default
        if isinstance(value, datetime.datetime):
            if not timezone.is_naive(value):
                value = timezone.make_naive(value, timezone.utc)
            value = value.date()
        elif isinstance(value, datetime.date):
            # Nothing to do, as dates don't have tz information
            pass
        else:
            # No explicit date / datetime value -- no checks necessary
            return []
        offset = datetime.timedelta(days=1)
        lower = (now - offset).date()
        upper = (now + offset).date()
        if lower <= value <= upper:
            return [
                checks.Warning(
                    'Fixed default value provided.',
                    hint='It seems you set a fixed date / time / datetime '
                         'value as default for this field. This may not be '
                         'what you want. If you want to have the current date '
                         'as default, use `django.utils.timezone.now`',
                    obj=self,
                    id='fields.W161',
                )
            ]

        return []
```
### 72 - django/db/models/fields/__init__.py:

Start line: 2042, End line: 2083

```python
class TimeField(DateTimeCheckMixin, Field):

    def _check_fix_default_value(self):
        """
        Warn that using an actual date or datetime value is probably wrong;
        it's only evaluated on server startup.
        """
        if not self.has_default():
            return []

        now = timezone.now()
        if not timezone.is_naive(now):
            now = timezone.make_naive(now, timezone.utc)
        value = self.default
        if isinstance(value, datetime.datetime):
            second_offset = datetime.timedelta(seconds=10)
            lower = now - second_offset
            upper = now + second_offset
            if timezone.is_aware(value):
                value = timezone.make_naive(value, timezone.utc)
        elif isinstance(value, datetime.time):
            second_offset = datetime.timedelta(seconds=10)
            lower = now - second_offset
            upper = now + second_offset
            value = datetime.datetime.combine(now.date(), value)
            if timezone.is_aware(value):
                value = timezone.make_naive(value, timezone.utc).time()
        else:
            # No explicit time / datetime value -- no checks necessary
            return []
        if lower <= value <= upper:
            return [
                checks.Warning(
                    'Fixed default value provided.',
                    hint='It seems you set a fixed date / time / datetime '
                         'value as default for this field. This may not be '
                         'what you want. If you want to have the current date '
                         'as default, use `django.utils.timezone.now`',
                    obj=self,
                    id='fields.W161',
                )
            ]

        return []
```
### 75 - django/db/models/fields/__init__.py:

Start line: 1991, End line: 2021

```python
class SmallIntegerField(IntegerField):
    description = _("Small integer")

    def get_internal_type(self):
        return "SmallIntegerField"


class TextField(Field):
    description = _("Text")

    def get_internal_type(self):
        return "TextField"

    def to_python(self, value):
        if isinstance(value, str) or value is None:
            return value
        return str(value)

    def get_prep_value(self, value):
        value = super().get_prep_value(value)
        return self.to_python(value)

    def formfield(self, **kwargs):
        # Passing max_length to forms.CharField means that the value's length
        # will be validated twice. This is considered acceptable since we want
        # the value in the form field (to pass into widget for example).
        return super().formfield(**{
            'max_length': self.max_length,
            **({} if self.choices is not None else {'widget': forms.Textarea}),
            **kwargs,
        })
```
### 84 - django/db/models/fields/__init__.py:

Start line: 1382, End line: 1404

```python
class DecimalField(Field):

    def _check_decimal_places(self):
        try:
            decimal_places = int(self.decimal_places)
            if decimal_places < 0:
                raise ValueError()
        except TypeError:
            return [
                checks.Error(
                    "DecimalFields must define a 'decimal_places' attribute.",
                    obj=self,
                    id='fields.E130',
                )
            ]
        except ValueError:
            return [
                checks.Error(
                    "'decimal_places' must be a non-negative integer.",
                    obj=self,
                    id='fields.E131',
                )
            ]
        else:
            return []
```
### 94 - django/db/models/fields/__init__.py:

Start line: 928, End line: 944

```python
class BooleanField(Field):

    def get_prep_value(self, value):
        value = super().get_prep_value(value)
        if value is None:
            return None
        return self.to_python(value)

    def formfield(self, **kwargs):
        if self.choices is not None:
            include_blank = not (self.has_default() or 'initial' in kwargs)
            defaults = {'choices': self.get_choices(include_blank=include_blank)}
        else:
            form_class = forms.NullBooleanField if self.null else forms.BooleanField
            # In HTML checkboxes, 'required' means "must be checked" which is
            # different from the choices case ("must select some value").
            # required=False allows unchecked checkboxes.
            defaults = {'form_class': form_class, 'required': False}
        return super().formfield(**{**defaults, **kwargs})
```
### 96 - django/db/models/fields/__init__.py:

Start line: 551, End line: 592

```python
@total_ordering
class Field(RegisterLookupMixin):

    def get_pk_value_on_save(self, instance):
        """
        Hook to generate new PK values on save. This method is called when
        saving instances with no primary key value set. If this method returns
        something else than None, then the returned value is used when saving
        the new instance.
        """
        if self.default:
            return self.get_default()
        return None

    def to_python(self, value):
        """
        Convert the input value into the expected Python data type, raising
        django.core.exceptions.ValidationError if the data can't be converted.
        Return the converted value. Subclasses should override this.
        """
        return value

    @cached_property
    def validators(self):
        """
        Some validators can't be created at field initialization time.
        This method provides a way to delay their creation until required.
        """
        return [*self.default_validators, *self._validators]

    def run_validators(self, value):
        if value in self.empty_values:
            return

        errors = []
        for v in self.validators:
            try:
                v(value)
            except exceptions.ValidationError as e:
                if hasattr(e, 'code') and e.code in self.error_messages:
                    e.message = self.error_messages[e.code]
                errors.extend(e.error_list)

        if errors:
            raise exceptions.ValidationError(errors)
```
### 105 - django/db/models/fields/__init__.py:

Start line: 2243, End line: 2293

```python
class UUIDField(Field):
    default_error_messages = {
        'invalid': _('“%(value)s” is not a valid UUID.'),
    }
    description = _('Universally unique identifier')
    empty_strings_allowed = False

    def __init__(self, verbose_name=None, **kwargs):
        kwargs['max_length'] = 32
        super().__init__(verbose_name, **kwargs)

    def deconstruct(self):
        name, path, args, kwargs = super().deconstruct()
        del kwargs['max_length']
        return name, path, args, kwargs

    def get_internal_type(self):
        return "UUIDField"

    def get_prep_value(self, value):
        value = super().get_prep_value(value)
        return self.to_python(value)

    def get_db_prep_value(self, value, connection, prepared=False):
        if value is None:
            return None
        if not isinstance(value, uuid.UUID):
            value = self.to_python(value)

        if connection.features.has_native_uuid_field:
            return value
        return value.hex

    def to_python(self, value):
        if value is not None and not isinstance(value, uuid.UUID):
            input_form = 'int' if isinstance(value, int) else 'hex'
            try:
                return uuid.UUID(**{input_form: value})
            except (AttributeError, ValueError):
                raise exceptions.ValidationError(
                    self.error_messages['invalid'],
                    code='invalid',
                    params={'value': value},
                )
        return value

    def formfield(self, **kwargs):
        return super().formfield(**{
            'form_class': forms.UUIDField,
            **kwargs,
        })
```
### 108 - django/db/models/fields/__init__.py:

Start line: 2296, End line: 2344

```python
class AutoFieldMixin:

    def __init__(self, *args, **kwargs):
        kwargs['blank'] = True
        super().__init__(*args, **kwargs)

    def check(self, **kwargs):
        return [
            *super().check(**kwargs),
            *self._check_primary_key(),
        ]

    def _check_primary_key(self):
        if not self.primary_key:
            return [
                checks.Error(
                    'AutoFields must set primary_key=True.',
                    obj=self,
                    id='fields.E100',
                ),
            ]
        else:
            return []

    def deconstruct(self):
        name, path, args, kwargs = super().deconstruct()
        del kwargs['blank']
        kwargs['primary_key'] = True
        return name, path, args, kwargs

    def validate(self, value, model_instance):
        pass

    def get_db_prep_value(self, value, connection, prepared=False):
        if not prepared:
            value = self.get_prep_value(value)
            value = connection.ops.validate_autopk_value(value)
        return value

    def contribute_to_class(self, cls, name, **kwargs):
        assert not cls._meta.auto_field, (
            "Model %s can't have more than one auto-generated field."
            % cls._meta.label
        )
        super().contribute_to_class(cls, name, **kwargs)
        cls._meta.auto_field = self

    def formfield(self, **kwargs):
        return None
```
### 111 - django/db/models/fields/__init__.py:

Start line: 2024, End line: 2040

```python
class TimeField(DateTimeCheckMixin, Field):
    empty_strings_allowed = False
    default_error_messages = {
        'invalid': _('“%(value)s” value has an invalid format. It must be in '
                     'HH:MM[:ss[.uuuuuu]] format.'),
        'invalid_time': _('“%(value)s” value has the correct format '
                          '(HH:MM[:ss[.uuuuuu]]) but it is an invalid time.'),
    }
    description = _("Time")

    def __init__(self, verbose_name=None, name=None, auto_now=False,
                 auto_now_add=False, **kwargs):
        self.auto_now, self.auto_now_add = auto_now, auto_now_add
        if auto_now or auto_now_add:
            kwargs['editable'] = False
            kwargs['blank'] = True
        super().__init__(verbose_name, name, **kwargs)
```
### 123 - django/db/models/fields/__init__.py:

Start line: 1160, End line: 1178

```python
class DateField(DateTimeCheckMixin, Field):

    def pre_save(self, model_instance, add):
        if self.auto_now or (self.auto_now_add and add):
            value = datetime.date.today()
            setattr(model_instance, self.attname, value)
            return value
        else:
            return super().pre_save(model_instance, add)

    def contribute_to_class(self, cls, name, **kwargs):
        super().contribute_to_class(cls, name, **kwargs)
        if not self.null:
            setattr(
                cls, 'get_next_by_%s' % self.name,
                partialmethod(cls._get_next_or_previous_by_FIELD, field=self, is_next=True)
            )
            setattr(
                cls, 'get_previous_by_%s' % self.name,
                partialmethod(cls._get_next_or_previous_by_FIELD, field=self, is_next=False)
            )
```
### 130 - django/db/models/fields/__init__.py:

Start line: 1, End line: 85

```python
import collections.abc
import copy
import datetime
import decimal
import operator
import uuid
import warnings
from base64 import b64decode, b64encode
from functools import partialmethod, total_ordering

from django import forms
from django.apps import apps
from django.conf import settings
from django.core import checks, exceptions, validators
# When the _meta object was formalized, this exception was moved to
# django.core.exceptions. It is retained here for backwards compatibility
# purposes.
from django.core.exceptions import FieldDoesNotExist  # NOQA
from django.db import connection, connections, router
from django.db.models.constants import LOOKUP_SEP
from django.db.models.query_utils import DeferredAttribute, RegisterLookupMixin
from django.utils import timezone
from django.utils.datastructures import DictWrapper
from django.utils.dateparse import (
    parse_date, parse_datetime, parse_duration, parse_time,
)
from django.utils.duration import duration_microseconds, duration_string
from django.utils.functional import Promise, cached_property
from django.utils.ipv6 import clean_ipv6_address
from django.utils.itercompat import is_iterable
from django.utils.text import capfirst
from django.utils.translation import gettext_lazy as _

__all__ = [
    'AutoField', 'BLANK_CHOICE_DASH', 'BigAutoField', 'BigIntegerField',
    'BinaryField', 'BooleanField', 'CharField', 'CommaSeparatedIntegerField',
    'DateField', 'DateTimeField', 'DecimalField', 'DurationField',
    'EmailField', 'Empty', 'Field', 'FieldDoesNotExist', 'FilePathField',
    'FloatField', 'GenericIPAddressField', 'IPAddressField', 'IntegerField',
    'NOT_PROVIDED', 'NullBooleanField', 'PositiveIntegerField',
    'PositiveSmallIntegerField', 'SlugField', 'SmallAutoField',
    'SmallIntegerField', 'TextField', 'TimeField', 'URLField', 'UUIDField',
]


class Empty:
    pass


class NOT_PROVIDED:
    pass


# The values to use for "blank" in SelectFields. Will be appended to the start
# of most "choices" lists.
BLANK_CHOICE_DASH = [("", "---------")]


def _load_field(app_label, model_name, field_name):
    return apps.get_model(app_label, model_name)._meta.get_field(field_name)


# A guide to Field parameters:
#
#   * name:      The name of the field specified in the model.
#   * attname:   The attribute to use on the model object. This is the same as
#                "name", except in the case of ForeignKeys, where "_id" is
#                appended.
#   * db_column: The db_column specified in the model (or None).
#   * column:    The database column for this field. This is the same as
#                "attname", except if db_column is specified.
#
# Code that introspects values, or does other dynamic things, should use
# attname. For example, this gets the primary key value of object "obj":
#
#     getattr(obj, opts.pk.attname)

def _empty(of_cls):
    new = Empty()
    new.__class__ = of_cls
    return new


def return_None():
    return None
```
### 134 - django/db/models/fields/__init__.py:

Start line: 1216, End line: 1257

```python
class DateTimeField(DateField):

    def _check_fix_default_value(self):
        """
        Warn that using an actual date or datetime value is probably wrong;
        it's only evaluated on server startup.
        """
        if not self.has_default():
            return []

        now = timezone.now()
        if not timezone.is_naive(now):
            now = timezone.make_naive(now, timezone.utc)
        value = self.default
        if isinstance(value, datetime.datetime):
            second_offset = datetime.timedelta(seconds=10)
            lower = now - second_offset
            upper = now + second_offset
            if timezone.is_aware(value):
                value = timezone.make_naive(value, timezone.utc)
        elif isinstance(value, datetime.date):
            second_offset = datetime.timedelta(seconds=10)
            lower = now - second_offset
            lower = datetime.datetime(lower.year, lower.month, lower.day)
            upper = now + second_offset
            upper = datetime.datetime(upper.year, upper.month, upper.day)
            value = datetime.datetime(value.year, value.month, value.day)
        else:
            # No explicit date / datetime value -- no checks necessary
            return []
        if lower <= value <= upper:
            return [
                checks.Warning(
                    'Fixed default value provided.',
                    hint='It seems you set a fixed date / time / datetime '
                         'value as default for this field. This may not be '
                         'what you want. If you want to have the current date '
                         'as default, use `django.utils.timezone.now`',
                    obj=self,
                    id='fields.W161',
                )
            ]

        return []
```
### 141 - django/db/models/fields/__init__.py:

Start line: 1058, End line: 1074

```python
class DateField(DateTimeCheckMixin, Field):
    empty_strings_allowed = False
    default_error_messages = {
        'invalid': _('“%(value)s” value has an invalid date format. It must be '
                     'in YYYY-MM-DD format.'),
        'invalid_date': _('“%(value)s” value has the correct format (YYYY-MM-DD) '
                          'but it is an invalid date.'),
    }
    description = _("Date (without time)")

    def __init__(self, verbose_name=None, name=None, auto_now=False,
                 auto_now_add=False, **kwargs):
        self.auto_now, self.auto_now_add = auto_now, auto_now_add
        if auto_now or auto_now_add:
            kwargs['editable'] = False
            kwargs['blank'] = True
        super().__init__(verbose_name, name, **kwargs)
```
### 144 - django/db/models/fields/__init__.py:

Start line: 1010, End line: 1023

```python
class CommaSeparatedIntegerField(CharField):
    default_validators = [validators.validate_comma_separated_integer_list]
    description = _("Comma-separated integers")
    system_check_removed_details = {
        'msg': (
            'CommaSeparatedIntegerField is removed except for support in '
            'historical migrations.'
        ),
        'hint': (
            'Use CharField(validators=[validate_comma_separated_integer_list]) '
            'instead.'
        ),
        'id': 'fields.E901',
    }
```
### 149 - django/db/models/fields/__init__.py:

Start line: 2156, End line: 2176

```python
class URLField(CharField):
    default_validators = [validators.URLValidator()]
    description = _("URL")

    def __init__(self, verbose_name=None, name=None, **kwargs):
        kwargs.setdefault('max_length', 200)
        super().__init__(verbose_name, name, **kwargs)

    def deconstruct(self):
        name, path, args, kwargs = super().deconstruct()
        if kwargs.get("max_length") == 200:
            del kwargs['max_length']
        return name, path, args, kwargs

    def formfield(self, **kwargs):
        # As with CharField, this will cause URL validation to be performed
        # twice.
        return super().formfield(**{
            'form_class': forms.URLField,
            **kwargs,
        })
```
### 153 - django/db/models/fields/__init__.py:

Start line: 1576, End line: 1601

```python
class FilePathField(Field):
    description = _("File path")

    def __init__(self, verbose_name=None, name=None, path='', match=None,
                 recursive=False, allow_files=True, allow_folders=False, **kwargs):
        self.path, self.match, self.recursive = path, match, recursive
        self.allow_files, self.allow_folders = allow_files, allow_folders
        kwargs.setdefault('max_length', 100)
        super().__init__(verbose_name, name, **kwargs)

    def check(self, **kwargs):
        return [
            *super().check(**kwargs),
            *self._check_allowing_files_or_folders(**kwargs),
        ]

    def _check_allowing_files_or_folders(self, **kwargs):
        if not self.allow_files and not self.allow_folders:
            return [
                checks.Error(
                    "FilePathFields must have either 'allow_files' or 'allow_folders' set to True.",
                    obj=self,
                    id='fields.E140',
                )
            ]
        return []
```
### 159 - django/db/models/fields/__init__.py:

Start line: 1734, End line: 1779

```python
class IntegerField(Field):

    def get_prep_value(self, value):
        value = super().get_prep_value(value)
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError) as e:
            raise e.__class__(
                "Field '%s' expected a number but got %r." % (self.name, value),
            ) from e

    def get_internal_type(self):
        return "IntegerField"

    def to_python(self, value):
        if value is None:
            return value
        try:
            return int(value)
        except (TypeError, ValueError):
            raise exceptions.ValidationError(
                self.error_messages['invalid'],
                code='invalid',
                params={'value': value},
            )

    def formfield(self, **kwargs):
        return super().formfield(**{
            'form_class': forms.IntegerField,
            **kwargs,
        })


class BigIntegerField(IntegerField):
    description = _("Big (8 byte) integer")
    MAX_BIGINT = 9223372036854775807

    def get_internal_type(self):
        return "BigIntegerField"

    def formfield(self, **kwargs):
        return super().formfield(**{
            'min_value': -BigIntegerField.MAX_BIGINT - 1,
            'max_value': BigIntegerField.MAX_BIGINT,
            **kwargs,
        })
```
