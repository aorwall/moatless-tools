# django__django-12306

| **django/django** | `8b3e714ecf409ed6c9628c3f2a4e033cbfa4253b` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 448 |
| **Any found context length** | 448 |
| **Avg pos** | 1.0 |
| **Min pos** | 1 |
| **Max pos** | 1 |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/db/models/fields/__init__.py b/django/db/models/fields/__init__.py
--- a/django/db/models/fields/__init__.py
+++ b/django/db/models/fields/__init__.py
@@ -269,10 +269,10 @@ def _check_choices(self):
                 ):
                     break
                 if self.max_length is not None and group_choices:
-                    choice_max_length = max(
+                    choice_max_length = max([
                         choice_max_length,
                         *(len(value) for value, _ in group_choices if isinstance(value, str)),
-                    )
+                    ])
             except (TypeError, ValueError):
                 # No groups, choices in the form [value, display]
                 value, human_name = group_name, group_choices

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/db/models/fields/__init__.py | 272 | 275 | 1 | 1 | 448


## Problem Statement

```
Named groups in choices are not properly validated in case of non str typed values.
Description
	
In case of using typed choices and string value to store it (in my case it is multiple values stored in char field as JSON) it is possible to catch error while run makemigrations (_check_choices error):
main.MultiValueFieldModel.multi_value_field_integer_with_grouped_choices: (fields.E005) 'choices' must be an iterable containing (actual value, human readable name) tuples.
Looking deeper into the django code, we see actual error message: 'int' object is not iterable and it happens in this block of code (​https://github.com/django/django/blob/aa6c620249bc8c2a6245c8d7b928b05e7e5e78fc/django/db/models/fields/__init__.py#L271-L275):
if self.max_length is not None and group_choices:
	choice_max_length = max(
		choice_max_length,
		*(len(value) for value, _ in group_choices if isinstance(value, str)),
	)
If we have CharField (any other with max_length defined) and grouped choices with non str typed values like:
choices=(
	('one', ((1, 'One',), (11, 'Eleven',),),),
	('two', ((2, 'Two',), (22, 'Twenty two',),),),
)
we will have the situation, when max function receives only one integer value (choice_max_length), because (len(value) for value, _ in group_choices if isinstance(value, str)) will return empty generator, and that is why error 'int' object is not iterable raises (max function waits iterable if there is only one argument).
Code block:
choice_max_length = max(
	choice_max_length,
	*(len(value) for value, _ in group_choices if isinstance(value, str)),
)
in this case works like:
choice_max_length = max(
	choice_max_length,
	*[],
)
which is incorrect.
The simples solution is to add one additional argument to max function, which will be usefull only in this partucular situation:
choice_max_length = max(
	choice_max_length, 0,
	*(len(value) for value, _ in group_choices if isinstance(value, str)),
)

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 django/db/models/fields/__init__.py** | 244 | 306| 448 | 448 | 17512 | 
| 2 | 2 django/forms/fields.py | 901 | 934| 235 | 683 | 26525 | 
| 3 | 2 django/forms/fields.py | 859 | 898| 298 | 981 | 26525 | 
| 4 | 3 django/forms/models.py | 1254 | 1284| 242 | 1223 | 38127 | 
| 5 | 3 django/forms/fields.py | 770 | 829| 416 | 1639 | 38127 | 
| 6 | 3 django/forms/fields.py | 832 | 856| 177 | 1816 | 38127 | 
| 7 | 3 django/forms/models.py | 1357 | 1384| 209 | 2025 | 38127 | 
| 8 | 3 django/forms/models.py | 1129 | 1171| 310 | 2335 | 38127 | 
| 9 | **3 django/db/models/fields/__init__.py** | 973 | 1005| 208 | 2543 | 38127 | 
| 10 | 3 django/forms/models.py | 1320 | 1355| 286 | 2829 | 38127 | 
| 11 | **3 django/db/models/fields/__init__.py** | 608 | 637| 234 | 3063 | 38127 | 
| 12 | **3 django/db/models/fields/__init__.py** | 1706 | 1729| 146 | 3209 | 38127 | 
| 13 | 3 django/forms/models.py | 1287 | 1318| 266 | 3475 | 38127 | 
| 14 | 3 django/forms/models.py | 1239 | 1252| 176 | 3651 | 38127 | 
| 15 | 3 django/forms/fields.py | 1006 | 1058| 450 | 4101 | 38127 | 
| 16 | 4 django/db/models/enums.py | 62 | 83| 118 | 4219 | 38720 | 
| 17 | **4 django/db/models/fields/__init__.py** | 1760 | 1805| 279 | 4498 | 38720 | 
| 18 | 4 django/forms/fields.py | 261 | 285| 195 | 4693 | 38720 | 
| 19 | **4 django/db/models/fields/__init__.py** | 208 | 242| 235 | 4928 | 38720 | 
| 20 | **4 django/db/models/fields/__init__.py** | 1007 | 1033| 229 | 5157 | 38720 | 
| 21 | 5 django/contrib/admin/filters.py | 278 | 302| 217 | 5374 | 42813 | 
| 22 | 5 django/forms/models.py | 1174 | 1237| 520 | 5894 | 42813 | 
| 23 | **5 django/db/models/fields/__init__.py** | 1731 | 1758| 215 | 6109 | 42813 | 
| 24 | 5 django/contrib/admin/filters.py | 397 | 419| 211 | 6320 | 42813 | 
| 25 | 5 django/db/models/enums.py | 37 | 59| 183 | 6503 | 42813 | 
| 26 | 5 django/forms/fields.py | 242 | 259| 164 | 6667 | 42813 | 
| 27 | **5 django/db/models/fields/__init__.py** | 835 | 859| 244 | 6911 | 42813 | 
| 28 | 5 django/forms/models.py | 1086 | 1126| 306 | 7217 | 42813 | 
| 29 | **5 django/db/models/fields/__init__.py** | 1456 | 1515| 398 | 7615 | 42813 | 
| 30 | 6 django/db/migrations/questioner.py | 84 | 107| 187 | 7802 | 44887 | 
| 31 | 7 django/contrib/admin/widgets.py | 423 | 447| 232 | 8034 | 48753 | 
| 32 | **7 django/db/models/fields/__init__.py** | 2030 | 2060| 199 | 8233 | 48753 | 
| 33 | 8 django/db/models/fields/related.py | 1169 | 1200| 180 | 8413 | 62265 | 
| 34 | 9 django/db/backends/mysql/validation.py | 29 | 61| 246 | 8659 | 62753 | 
| 35 | **9 django/db/models/fields/__init__.py** | 1036 | 1049| 104 | 8763 | 62753 | 
| 36 | 9 django/db/models/fields/related.py | 1202 | 1313| 939 | 9702 | 62753 | 
| 37 | **9 django/db/models/fields/__init__.py** | 1432 | 1454| 119 | 9821 | 62753 | 
| 38 | 10 django/contrib/postgres/fields/ranges.py | 42 | 90| 362 | 10183 | 64870 | 
| 39 | 11 django/db/models/base.py | 1762 | 1833| 565 | 10748 | 80186 | 
| 40 | 11 django/contrib/admin/filters.py | 209 | 226| 190 | 10938 | 80186 | 
| 41 | 12 django/contrib/postgres/forms/ranges.py | 81 | 103| 149 | 11087 | 80863 | 
| 42 | 12 django/db/models/fields/related.py | 1315 | 1387| 616 | 11703 | 80863 | 
| 43 | **12 django/db/models/fields/__init__.py** | 1383 | 1406| 171 | 11874 | 80863 | 
| 44 | 13 django/contrib/postgres/validators.py | 1 | 21| 181 | 12055 | 81414 | 
| 45 | 14 django/core/validators.py | 330 | 375| 308 | 12363 | 85649 | 
| 46 | 15 django/forms/widgets.py | 763 | 786| 204 | 12567 | 93655 | 
| 47 | 15 django/core/validators.py | 407 | 453| 415 | 12982 | 93655 | 
| 48 | 15 django/forms/widgets.py | 637 | 666| 238 | 13220 | 93655 | 
| 49 | 16 django/contrib/auth/migrations/0001_initial.py | 1 | 105| 849 | 14069 | 94504 | 
| 50 | 17 django/contrib/admin/checks.py | 1093 | 1123| 188 | 14257 | 103552 | 
| 51 | **17 django/db/models/fields/__init__.py** | 1052 | 1081| 218 | 14475 | 103552 | 
| 52 | **17 django/db/models/fields/__init__.py** | 338 | 362| 184 | 14659 | 103552 | 
| 53 | 17 django/forms/widgets.py | 669 | 700| 261 | 14920 | 103552 | 
| 54 | 17 django/contrib/admin/filters.py | 264 | 276| 149 | 15069 | 103552 | 
| 55 | 17 django/forms/fields.py | 1184 | 1214| 182 | 15251 | 103552 | 
| 56 | 17 django/db/models/base.py | 1662 | 1760| 717 | 15968 | 103552 | 
| 57 | 17 django/core/validators.py | 284 | 294| 109 | 16077 | 103552 | 
| 58 | 17 django/forms/fields.py | 208 | 239| 274 | 16351 | 103552 | 
| 59 | 17 django/contrib/postgres/forms/ranges.py | 31 | 78| 325 | 16676 | 103552 | 
| 60 | 18 django/contrib/admin/migrations/0003_logentry_add_action_flag_choices.py | 1 | 21| 111 | 16787 | 103663 | 
| 61 | 18 django/db/migrations/questioner.py | 143 | 160| 183 | 16970 | 103663 | 
| 62 | 19 django/contrib/postgres/forms/array.py | 1 | 37| 258 | 17228 | 105257 | 
| 63 | 19 django/db/models/base.py | 1612 | 1660| 348 | 17576 | 105257 | 
| 64 | 19 django/db/models/base.py | 1556 | 1581| 183 | 17759 | 105257 | 
| 65 | 19 django/db/models/fields/related.py | 1611 | 1645| 266 | 18025 | 105257 | 
| 66 | **19 django/db/models/fields/__init__.py** | 1 | 81| 633 | 18658 | 105257 | 
| 67 | 19 django/forms/fields.py | 1089 | 1130| 353 | 19011 | 105257 | 
| 68 | 20 django/db/models/fields/reverse_related.py | 117 | 134| 161 | 19172 | 107400 | 
| 69 | 20 django/core/validators.py | 141 | 176| 388 | 19560 | 107400 | 
| 70 | 20 django/forms/widgets.py | 741 | 760| 144 | 19704 | 107400 | 
| 71 | 20 django/db/migrations/questioner.py | 227 | 240| 123 | 19827 | 107400 | 
| 72 | 21 django/contrib/admin/options.py | 188 | 204| 171 | 19998 | 125887 | 
| 73 | 21 django/forms/fields.py | 351 | 368| 160 | 20158 | 125887 | 
| 74 | 22 django/contrib/auth/migrations/0008_alter_user_username_max_length.py | 1 | 25| 138 | 20296 | 126025 | 
| 75 | 22 django/contrib/admin/filters.py | 246 | 261| 154 | 20450 | 126025 | 
| 76 | **22 django/db/models/fields/__init__.py** | 1958 | 1994| 198 | 20648 | 126025 | 
| 77 | **22 django/db/models/fields/__init__.py** | 1408 | 1430| 121 | 20769 | 126025 | 
| 78 | **22 django/db/models/fields/__init__.py** | 954 | 970| 176 | 20945 | 126025 | 
| 79 | 22 django/db/models/enums.py | 1 | 35| 298 | 21243 | 126025 | 
| 80 | 22 django/contrib/admin/checks.py | 425 | 446| 191 | 21434 | 126025 | 
| 81 | 22 django/forms/fields.py | 323 | 349| 227 | 21661 | 126025 | 
| 82 | 23 django/contrib/auth/migrations/0007_alter_validators_add_error_messages.py | 1 | 25| 137 | 21798 | 126162 | 
| 83 | 23 django/forms/models.py | 752 | 773| 194 | 21992 | 126162 | 
| 84 | 24 django/db/migrations/exceptions.py | 1 | 55| 250 | 22242 | 126413 | 
| 85 | 25 django/contrib/postgres/forms/jsonb.py | 1 | 63| 345 | 22587 | 126758 | 
| 86 | 25 django/db/models/base.py | 1160 | 1188| 213 | 22800 | 126758 | 
| 87 | **25 django/db/models/fields/__init__.py** | 2282 | 2332| 339 | 23139 | 126758 | 
| 88 | 25 django/db/models/base.py | 1529 | 1554| 183 | 23322 | 126758 | 
| 89 | 25 django/core/validators.py | 378 | 405| 224 | 23546 | 126758 | 
| 90 | 25 django/contrib/postgres/forms/array.py | 62 | 102| 248 | 23794 | 126758 | 
| 91 | 26 django/contrib/postgres/fields/array.py | 18 | 51| 288 | 24082 | 128839 | 
| 92 | 26 django/contrib/admin/checks.py | 1048 | 1090| 343 | 24425 | 128839 | 
| 93 | 26 django/db/migrations/questioner.py | 162 | 185| 246 | 24671 | 128839 | 
| 94 | 26 django/forms/widgets.py | 582 | 614| 232 | 24903 | 128839 | 
| 95 | **26 django/db/models/fields/__init__.py** | 639 | 663| 206 | 25109 | 128839 | 
| 96 | 26 django/forms/widgets.py | 546 | 580| 277 | 25386 | 128839 | 
| 97 | 27 django/contrib/auth/checks.py | 97 | 167| 525 | 25911 | 130012 | 
| 98 | 27 django/contrib/admin/widgets.py | 352 | 378| 328 | 26239 | 130012 | 
| 99 | 28 django/core/serializers/base.py | 273 | 298| 218 | 26457 | 132437 | 
| 100 | 29 django/contrib/auth/migrations/0004_alter_user_username_opts.py | 1 | 24| 150 | 26607 | 132587 | 
| 101 | **29 django/db/models/fields/__init__.py** | 1997 | 2027| 252 | 26859 | 132587 | 
| 102 | **29 django/db/models/fields/__init__.py** | 1666 | 1703| 226 | 27085 | 132587 | 
| 103 | 29 django/db/models/base.py | 1495 | 1527| 231 | 27316 | 132587 | 
| 104 | 30 django/db/models/fields/mixins.py | 31 | 57| 173 | 27489 | 132930 | 
| 105 | 31 django/contrib/auth/migrations/0010_alter_group_name_max_length.py | 1 | 17| 0 | 27489 | 133007 | 
| 106 | 32 django/contrib/contenttypes/checks.py | 24 | 42| 125 | 27614 | 133256 | 
| 107 | 32 django/db/models/fields/related.py | 896 | 916| 178 | 27792 | 133256 | 
| 108 | 33 django/db/models/lookups.py | 292 | 342| 306 | 28098 | 138079 | 
| 109 | 33 django/core/validators.py | 297 | 327| 227 | 28325 | 138079 | 
| 110 | 33 django/db/models/base.py | 1251 | 1280| 242 | 28567 | 138079 | 
| 111 | 34 django/db/models/fields/related_lookups.py | 46 | 60| 224 | 28791 | 139528 | 
| 112 | 34 django/db/models/fields/related.py | 1389 | 1419| 322 | 29113 | 139528 | 
| 113 | 34 django/db/models/base.py | 1069 | 1112| 404 | 29517 | 139528 | 
| 114 | 35 django/db/models/options.py | 39 | 62| 203 | 29720 | 146612 | 
| 115 | 36 django/db/models/functions/text.py | 1 | 21| 183 | 29903 | 149068 | 
| 116 | 36 django/contrib/postgres/fields/ranges.py | 114 | 161| 262 | 30165 | 149068 | 
| 117 | 36 django/db/backends/mysql/validation.py | 1 | 27| 248 | 30413 | 149068 | 
| 118 | **36 django/db/models/fields/__init__.py** | 364 | 390| 199 | 30612 | 149068 | 
| 119 | 36 django/forms/widgets.py | 616 | 635| 189 | 30801 | 149068 | 
| 120 | 37 django/contrib/postgres/fields/jsonb.py | 29 | 82| 361 | 31162 | 150291 | 
| 121 | 38 django/db/migrations/state.py | 349 | 399| 471 | 31633 | 155509 | 
| 122 | 38 django/db/migrations/state.py | 580 | 599| 188 | 31821 | 155509 | 
| 123 | **38 django/db/models/fields/__init__.py** | 861 | 882| 162 | 31983 | 155509 | 
| 124 | 38 django/forms/models.py | 1 | 28| 215 | 32198 | 155509 | 
| 125 | 38 django/forms/models.py | 679 | 750| 732 | 32930 | 155509 | 
| 126 | 39 django/core/exceptions.py | 99 | 194| 649 | 33579 | 156564 | 
| 127 | 39 django/forms/fields.py | 288 | 320| 228 | 33807 | 156564 | 
| 128 | 39 django/db/models/base.py | 1143 | 1158| 138 | 33945 | 156564 | 
| 129 | 39 django/db/migrations/questioner.py | 56 | 81| 220 | 34165 | 156564 | 
| 130 | 39 django/forms/fields.py | 961 | 1004| 372 | 34537 | 156564 | 
| 131 | 39 django/contrib/admin/options.py | 368 | 420| 504 | 35041 | 156564 | 
| 132 | 39 django/forms/models.py | 382 | 410| 240 | 35281 | 156564 | 
| 133 | 39 django/contrib/admin/checks.py | 880 | 928| 416 | 35697 | 156564 | 
| 134 | 39 django/db/migrations/questioner.py | 109 | 141| 290 | 35987 | 156564 | 
| 135 | **39 django/db/models/fields/__init__.py** | 308 | 336| 205 | 36192 | 156564 | 
| 136 | 40 django/forms/boundfield.py | 35 | 50| 149 | 36341 | 158685 | 
| 137 | 40 django/contrib/postgres/fields/array.py | 1 | 15| 110 | 36451 | 158685 | 
| 138 | **40 django/db/models/fields/__init__.py** | 1578 | 1599| 183 | 36634 | 158685 | 
| 139 | 40 django/db/models/base.py | 1389 | 1444| 491 | 37125 | 158685 | 
| 140 | 40 django/db/models/base.py | 1835 | 1859| 175 | 37300 | 158685 | 
| 141 | 40 django/db/migrations/state.py | 601 | 612| 136 | 37436 | 158685 | 
| 142 | 40 django/db/models/options.py | 149 | 208| 587 | 38023 | 158685 | 
| 143 | 40 django/db/models/base.py | 1340 | 1370| 244 | 38267 | 158685 | 
| 144 | 40 django/contrib/admin/checks.py | 448 | 474| 190 | 38457 | 158685 | 
| 145 | 40 django/contrib/admin/checks.py | 510 | 520| 134 | 38591 | 158685 | 
| 146 | 40 django/db/models/base.py | 1446 | 1469| 176 | 38767 | 158685 | 
| 147 | 40 django/db/migrations/questioner.py | 207 | 224| 171 | 38938 | 158685 | 
| 148 | 40 django/contrib/admin/checks.py | 149 | 159| 123 | 39061 | 158685 | 
| 149 | 40 django/db/models/fields/related.py | 509 | 563| 409 | 39470 | 158685 | 
| 150 | 40 django/contrib/admin/checks.py | 326 | 352| 221 | 39691 | 158685 | 
| 151 | 40 django/forms/fields.py | 735 | 767| 241 | 39932 | 158685 | 
| 152 | **40 django/db/models/fields/__init__.py** | 927 | 952| 209 | 40141 | 158685 | 
| 153 | 40 django/db/models/base.py | 1471 | 1493| 171 | 40312 | 158685 | 
| 154 | 40 django/db/models/fields/related_lookups.py | 102 | 117| 215 | 40527 | 158685 | 
| 155 | 40 django/contrib/admin/checks.py | 792 | 842| 443 | 40970 | 158685 | 
| 156 | 40 django/contrib/auth/checks.py | 1 | 94| 646 | 41616 | 158685 | 
| 157 | 40 django/db/models/base.py | 505 | 547| 347 | 41963 | 158685 | 
| 158 | 40 django/core/serializers/base.py | 301 | 323| 207 | 42170 | 158685 | 
| 159 | 40 django/contrib/admin/options.py | 1 | 96| 769 | 42939 | 158685 | 
| 160 | 40 django/core/validators.py | 456 | 491| 249 | 43188 | 158685 | 
| 161 | 41 django/db/backends/mysql/operations.py | 193 | 236| 329 | 43517 | 162016 | 
| 162 | 41 django/contrib/admin/filters.py | 422 | 429| 107 | 43624 | 162016 | 
| 163 | 41 django/forms/fields.py | 480 | 505| 174 | 43798 | 162016 | 
| 164 | 41 django/contrib/postgres/validators.py | 24 | 77| 370 | 44168 | 162016 | 
| 165 | 41 django/db/models/fields/related_lookups.py | 62 | 99| 447 | 44615 | 162016 | 
| 166 | 42 django/contrib/auth/models.py | 80 | 124| 309 | 44924 | 165205 | 
| 167 | 43 django/utils/datastructures.py | 151 | 190| 300 | 45224 | 167462 | 
| 168 | **43 django/db/models/fields/__init__.py** | 1102 | 1140| 293 | 45517 | 167462 | 
| 169 | 43 django/db/models/fields/related.py | 401 | 419| 165 | 45682 | 167462 | 
| 170 | 43 django/contrib/postgres/fields/array.py | 160 | 177| 146 | 45828 | 167462 | 
| 171 | 43 django/forms/fields.py | 540 | 556| 180 | 46008 | 167462 | 
| 172 | **43 django/db/models/fields/__init__.py** | 1839 | 1916| 567 | 46575 | 167462 | 
| 173 | 43 django/db/models/base.py | 1 | 50| 330 | 46905 | 167462 | 
| 174 | 43 django/contrib/admin/checks.py | 161 | 202| 325 | 47230 | 167462 | 
| 175 | 44 django/db/models/deletion.py | 1 | 76| 566 | 47796 | 171121 | 
| 176 | 44 django/contrib/admin/checks.py | 522 | 545| 230 | 48026 | 171121 | 
| 177 | 45 django/db/backends/sqlite3/introspection.py | 23 | 54| 296 | 48322 | 174832 | 
| 178 | 45 django/db/models/options.py | 262 | 294| 331 | 48653 | 174832 | 
| 179 | 45 django/db/models/fields/related.py | 952 | 962| 121 | 48774 | 174832 | 
| 180 | 45 django/db/models/options.py | 363 | 385| 164 | 48938 | 174832 | 
| 181 | 45 django/db/models/lookups.py | 208 | 243| 308 | 49246 | 174832 | 
| 182 | 45 django/db/models/lookups.py | 377 | 406| 337 | 49583 | 174832 | 
| 183 | **45 django/db/models/fields/__init__.py** | 2335 | 2384| 311 | 49894 | 174832 | 
| 184 | 45 django/db/models/lookups.py | 345 | 375| 263 | 50157 | 174832 | 
| 185 | **45 django/db/models/fields/__init__.py** | 2081 | 2122| 325 | 50482 | 174832 | 
| 186 | 46 django/contrib/postgres/fields/hstore.py | 1 | 69| 435 | 50917 | 175532 | 
| 187 | 46 django/core/validators.py | 222 | 266| 330 | 51247 | 175532 | 
| 188 | **46 django/db/models/fields/__init__.py** | 1242 | 1283| 332 | 51579 | 175532 | 
| 189 | 46 django/db/models/fields/related.py | 255 | 282| 269 | 51848 | 175532 | 
| 190 | 46 django/contrib/postgres/fields/ranges.py | 92 | 111| 160 | 52008 | 175532 | 
| 191 | **46 django/db/models/fields/__init__.py** | 2415 | 2440| 143 | 52151 | 175532 | 
| 192 | 46 django/contrib/admin/checks.py | 231 | 261| 229 | 52380 | 175532 | 
| 193 | 46 django/utils/datastructures.py | 42 | 149| 766 | 53146 | 175532 | 
| 194 | 47 django/core/checks/translation.py | 1 | 65| 445 | 53591 | 175977 | 
| 195 | 48 django/db/backends/base/features.py | 1 | 115| 904 | 54495 | 178556 | 
| 196 | 49 django/contrib/admin/helpers.py | 1 | 30| 198 | 54693 | 181749 | 
| 197 | 49 django/core/validators.py | 494 | 530| 219 | 54912 | 181749 | 
| 198 | 49 django/db/models/options.py | 1 | 36| 301 | 55213 | 181749 | 
| 199 | 49 django/contrib/admin/checks.py | 844 | 866| 217 | 55430 | 181749 | 
| 200 | 49 django/contrib/admin/checks.py | 476 | 486| 149 | 55579 | 181749 | 
| 201 | 49 django/forms/models.py | 309 | 348| 387 | 55966 | 181749 | 
| 202 | **49 django/db/models/fields/__init__.py** | 1919 | 1938| 164 | 56130 | 181749 | 


### Hint

```
Thanks for this report. Regression in b6251956b69512bf230322bd7a49b629ca8455c6.
```

## Patch

```diff
diff --git a/django/db/models/fields/__init__.py b/django/db/models/fields/__init__.py
--- a/django/db/models/fields/__init__.py
+++ b/django/db/models/fields/__init__.py
@@ -269,10 +269,10 @@ def _check_choices(self):
                 ):
                     break
                 if self.max_length is not None and group_choices:
-                    choice_max_length = max(
+                    choice_max_length = max([
                         choice_max_length,
                         *(len(value) for value, _ in group_choices if isinstance(value, str)),
-                    )
+                    ])
             except (TypeError, ValueError):
                 # No groups, choices in the form [value, display]
                 value, human_name = group_name, group_choices

```

## Test Patch

```diff
diff --git a/tests/invalid_models_tests/test_ordinary_fields.py b/tests/invalid_models_tests/test_ordinary_fields.py
--- a/tests/invalid_models_tests/test_ordinary_fields.py
+++ b/tests/invalid_models_tests/test_ordinary_fields.py
@@ -1,4 +1,5 @@
 import unittest
+import uuid
 
 from django.core.checks import Error, Warning as DjangoWarning
 from django.db import connection, models
@@ -769,3 +770,20 @@ class Model(models.Model):
                 id='fields.W162',
             )
         ])
+
+
+@isolate_apps('invalid_models_tests')
+class UUIDFieldTests(TestCase):
+    def test_choices_named_group(self):
+        class Model(models.Model):
+            field = models.UUIDField(
+                choices=[
+                    ['knights', [
+                        [uuid.UUID('5c859437-d061-4847-b3f7-e6b78852f8c8'), 'Lancelot'],
+                        [uuid.UUID('c7853ec1-2ea3-4359-b02d-b54e8f1bcee2'), 'Galahad'],
+                    ]],
+                    [uuid.UUID('25d405be-4895-4d50-9b2e-d6695359ce47'), 'Other'],
+                ],
+            )
+
+        self.assertEqual(Model._meta.get_field('field').check(), [])

```


## Code snippets

### 1 - django/db/models/fields/__init__.py:

Start line: 244, End line: 306

```python
@total_ordering
class Field(RegisterLookupMixin):

    def _check_choices(self):
        if not self.choices:
            return []

        if not is_iterable(self.choices) or isinstance(self.choices, str):
            return [
                checks.Error(
                    "'choices' must be an iterable (e.g., a list or tuple).",
                    obj=self,
                    id='fields.E004',
                )
            ]

        choice_max_length = 0
        # Expect [group_name, [value, display]]
        for choices_group in self.choices:
            try:
                group_name, group_choices = choices_group
            except (TypeError, ValueError):
                # Containing non-pairs
                break
            try:
                if not all(
                    self._choices_is_value(value) and self._choices_is_value(human_name)
                    for value, human_name in group_choices
                ):
                    break
                if self.max_length is not None and group_choices:
                    choice_max_length = max(
                        choice_max_length,
                        *(len(value) for value, _ in group_choices if isinstance(value, str)),
                    )
            except (TypeError, ValueError):
                # No groups, choices in the form [value, display]
                value, human_name = group_name, group_choices
                if not self._choices_is_value(value) or not self._choices_is_value(human_name):
                    break
                if self.max_length is not None and isinstance(value, str):
                    choice_max_length = max(choice_max_length, len(value))

            # Special case: choices=['ab']
            if isinstance(choices_group, str):
                break
        else:
            if self.max_length is not None and choice_max_length > self.max_length:
                return [
                    checks.Error(
                        "'max_length' is too small to fit the longest value "
                        "in 'choices' (%d characters)." % choice_max_length,
                        obj=self,
                        id='fields.E009',
                    ),
                ]
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
### 2 - django/forms/fields.py:

Start line: 901, End line: 934

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
### 3 - django/forms/fields.py:

Start line: 859, End line: 898

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
### 4 - django/forms/models.py:

Start line: 1254, End line: 1284

```python
class ModelChoiceField(ChoiceField):

    choices = property(_get_choices, ChoiceField._set_choices)

    def prepare_value(self, value):
        if hasattr(value, '_meta'):
            if self.to_field_name:
                return value.serializable_value(self.to_field_name)
            else:
                return value.pk
        return super().prepare_value(value)

    def to_python(self, value):
        if value in self.empty_values:
            return None
        try:
            key = self.to_field_name or 'pk'
            if isinstance(value, self.queryset.model):
                value = getattr(value, key)
            value = self.queryset.get(**{key: value})
        except (ValueError, TypeError, self.queryset.model.DoesNotExist):
            raise ValidationError(self.error_messages['invalid_choice'], code='invalid_choice')
        return value

    def validate(self, value):
        return Field.validate(self, value)

    def has_changed(self, initial, data):
        if self.disabled:
            return False
        initial_value = initial if initial is not None else ''
        data_value = data if data is not None else ''
        return str(self.prepare_value(initial_value)) != str(data_value)
```
### 5 - django/forms/fields.py:

Start line: 770, End line: 829

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
### 6 - django/forms/fields.py:

Start line: 832, End line: 856

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
### 7 - django/forms/models.py:

Start line: 1357, End line: 1384

```python
class ModelMultipleChoiceField(ModelChoiceField):

    def prepare_value(self, value):
        if (hasattr(value, '__iter__') and
                not isinstance(value, str) and
                not hasattr(value, '_meta')):
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
    return hasattr(form_class, '_meta') and (
        form_class._meta.fields is not None or
        form_class._meta.exclude is not None
    )
```
### 8 - django/forms/models.py:

Start line: 1129, End line: 1171

```python
class ModelChoiceIteratorValue:
    def __init__(self, value, instance):
        self.value = value
        self.instance = instance

    def __str__(self):
        return str(self.value)

    def __eq__(self, other):
        if isinstance(other, ModelChoiceIteratorValue):
            other = other.value
        return self.value == other


class ModelChoiceIterator:
    def __init__(self, field):
        self.field = field
        self.queryset = field.queryset

    def __iter__(self):
        if self.field.empty_label is not None:
            yield ("", self.field.empty_label)
        queryset = self.queryset
        # Can't use iterator() when queryset uses prefetch_related()
        if not queryset._prefetch_related_lookups:
            queryset = queryset.iterator()
        for obj in queryset:
            yield self.choice(obj)

    def __len__(self):
        # count() adds a query but uses less memory since the QuerySet results
        # won't be cached. In most cases, the choices will only be iterated on,
        # and __len__() won't be called.
        return self.queryset.count() + (1 if self.field.empty_label is not None else 0)

    def __bool__(self):
        return self.field.empty_label is not None or self.queryset.exists()

    def choice(self, obj):
        return (
            ModelChoiceIteratorValue(self.field.prepare_value(obj), obj),
            self.field.label_from_instance(obj),
        )
```
### 9 - django/db/models/fields/__init__.py:

Start line: 973, End line: 1005

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
### 10 - django/forms/models.py:

Start line: 1320, End line: 1355

```python
class ModelMultipleChoiceField(ModelChoiceField):

    def _check_values(self, value):
        """
        Given a list of possible PK values, return a QuerySet of the
        corresponding objects. Raise a ValidationError if a given value is
        invalid (not a valid PK, not in the queryset, etc.)
        """
        key = self.to_field_name or 'pk'
        # deduplicate given values to avoid creating many querysets or
        # requiring the database backend deduplicate efficiently.
        try:
            value = frozenset(value)
        except TypeError:
            # list of lists isn't hashable, for example
            raise ValidationError(
                self.error_messages['list'],
                code='list',
            )
        for pk in value:
            try:
                self.queryset.filter(**{key: pk})
            except (ValueError, TypeError):
                raise ValidationError(
                    self.error_messages['invalid_pk_value'],
                    code='invalid_pk_value',
                    params={'pk': pk},
                )
        qs = self.queryset.filter(**{'%s__in' % key: value})
        pks = {str(getattr(o, key)) for o in qs}
        for val in value:
            if str(val) not in pks:
                raise ValidationError(
                    self.error_messages['invalid_choice'],
                    code='invalid_choice',
                    params={'value': val},
                )
        return qs
```
### 11 - django/db/models/fields/__init__.py:

Start line: 608, End line: 637

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
### 12 - django/db/models/fields/__init__.py:

Start line: 1706, End line: 1729

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
### 17 - django/db/models/fields/__init__.py:

Start line: 1760, End line: 1805

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
### 19 - django/db/models/fields/__init__.py:

Start line: 208, End line: 242

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

    @classmethod
    def _choices_is_value(cls, value):
        return isinstance(value, (str, Promise)) or not is_iterable(value)
```
### 20 - django/db/models/fields/__init__.py:

Start line: 1007, End line: 1033

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
### 23 - django/db/models/fields/__init__.py:

Start line: 1731, End line: 1758

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
### 27 - django/db/models/fields/__init__.py:

Start line: 835, End line: 859

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
### 29 - django/db/models/fields/__init__.py:

Start line: 1456, End line: 1515

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

Start line: 2030, End line: 2060

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
### 35 - django/db/models/fields/__init__.py:

Start line: 1036, End line: 1049

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
### 37 - django/db/models/fields/__init__.py:

Start line: 1432, End line: 1454

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
### 43 - django/db/models/fields/__init__.py:

Start line: 1383, End line: 1406

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
### 51 - django/db/models/fields/__init__.py:

Start line: 1052, End line: 1081

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
### 52 - django/db/models/fields/__init__.py:

Start line: 338, End line: 362

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
### 66 - django/db/models/fields/__init__.py:

Start line: 1, End line: 81

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
    'EmailField', 'Empty', 'Field', 'FilePathField', 'FloatField',
    'GenericIPAddressField', 'IPAddressField', 'IntegerField', 'NOT_PROVIDED',
    'NullBooleanField', 'PositiveBigIntegerField', 'PositiveIntegerField',
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
### 76 - django/db/models/fields/__init__.py:

Start line: 1958, End line: 1994

```python
class PositiveBigIntegerField(PositiveIntegerRelDbTypeMixin, IntegerField):
    description = _('Positive big integer')

    def get_internal_type(self):
        return 'PositiveBigIntegerField'

    def formfield(self, **kwargs):
        return super().formfield(**{
            'min_value': 0,
            **kwargs,
        })


class PositiveIntegerField(PositiveIntegerRelDbTypeMixin, IntegerField):
    description = _("Positive integer")

    def get_internal_type(self):
        return "PositiveIntegerField"

    def formfield(self, **kwargs):
        return super().formfield(**{
            'min_value': 0,
            **kwargs,
        })


class PositiveSmallIntegerField(PositiveIntegerRelDbTypeMixin, IntegerField):
    description = _("Positive small integer")

    def get_internal_type(self):
        return "PositiveSmallIntegerField"

    def formfield(self, **kwargs):
        return super().formfield(**{
            'min_value': 0,
            **kwargs,
        })
```
### 77 - django/db/models/fields/__init__.py:

Start line: 1408, End line: 1430

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
### 78 - django/db/models/fields/__init__.py:

Start line: 954, End line: 970

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
### 87 - django/db/models/fields/__init__.py:

Start line: 2282, End line: 2332

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
### 95 - django/db/models/fields/__init__.py:

Start line: 639, End line: 663

```python
@total_ordering
class Field(RegisterLookupMixin):

    def clean(self, value, model_instance):
        """
        Convert the value's type and run validation. Validation errors
        from to_python() and validate() are propagated. Return the correct
        value if no error is raised.
        """
        value = self.to_python(value)
        self.validate(value, model_instance)
        self.run_validators(value)
        return value

    def db_type_parameters(self, connection):
        return DictWrapper(self.__dict__, connection.ops.quote_name, 'qn_')

    def db_check(self, connection):
        """
        Return the database column check constraint for this field, for the
        provided connection. Works the same way as db_type() for the case that
        get_internal_type() does not map to a preexisting model field.
        """
        data = self.db_type_parameters(connection)
        try:
            return connection.data_type_check_constraints[self.get_internal_type()] % data
        except KeyError:
            return None
```
### 101 - django/db/models/fields/__init__.py:

Start line: 1997, End line: 2027

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
### 102 - django/db/models/fields/__init__.py:

Start line: 1666, End line: 1703

```python
class FloatField(Field):
    empty_strings_allowed = False
    default_error_messages = {
        'invalid': _('“%(value)s” value must be a float.'),
    }
    description = _("Floating point number")

    def get_prep_value(self, value):
        value = super().get_prep_value(value)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError) as e:
            raise e.__class__(
                "Field '%s' expected a number but got %r." % (self.name, value),
            ) from e

    def get_internal_type(self):
        return "FloatField"

    def to_python(self, value):
        if value is None:
            return value
        try:
            return float(value)
        except (TypeError, ValueError):
            raise exceptions.ValidationError(
                self.error_messages['invalid'],
                code='invalid',
                params={'value': value},
            )

    def formfield(self, **kwargs):
        return super().formfield(**{
            'form_class': forms.FloatField,
            **kwargs,
        })
```
### 118 - django/db/models/fields/__init__.py:

Start line: 364, End line: 390

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
### 123 - django/db/models/fields/__init__.py:

Start line: 861, End line: 882

```python
@total_ordering
class Field(RegisterLookupMixin):

    def value_to_string(self, obj):
        """
        Return a string value of this field from the passed obj.
        This is used by the serialization framework.
        """
        return str(self.value_from_object(obj))

    def _get_flatchoices(self):
        """Flattened version of choices tuple."""
        if self.choices is None:
            return []
        flat = []
        for choice, value in self.choices:
            if isinstance(value, (list, tuple)):
                flat.extend(value)
            else:
                flat.append((choice, value))
        return flat
    flatchoices = property(_get_flatchoices)

    def save_form_data(self, instance, data):
        setattr(instance, self.name, data)
```
### 135 - django/db/models/fields/__init__.py:

Start line: 308, End line: 336

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
### 138 - django/db/models/fields/__init__.py:

Start line: 1578, End line: 1599

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
### 152 - django/db/models/fields/__init__.py:

Start line: 927, End line: 952

```python
class BooleanField(Field):
    empty_strings_allowed = False
    default_error_messages = {
        'invalid': _('“%(value)s” value must be either True or False.'),
        'invalid_nullable': _('“%(value)s” value must be either True, False, or None.'),
    }
    description = _("Boolean (Either True or False)")

    def get_internal_type(self):
        return "BooleanField"

    def to_python(self, value):
        if self.null and value in self.empty_values:
            return None
        if value in (True, False):
            # 1/0 are equal to True/False. bool() converts former to latter.
            return bool(value)
        if value in ('t', 'True', '1'):
            return True
        if value in ('f', 'False', '0'):
            return False
        raise exceptions.ValidationError(
            self.error_messages['invalid_nullable' if self.null else 'invalid'],
            code='invalid',
            params={'value': value},
        )
```
### 168 - django/db/models/fields/__init__.py:

Start line: 1102, End line: 1140

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
### 172 - django/db/models/fields/__init__.py:

Start line: 1839, End line: 1916

```python
class GenericIPAddressField(Field):
    empty_strings_allowed = False
    description = _("IP address")
    default_error_messages = {}

    def __init__(self, verbose_name=None, name=None, protocol='both',
                 unpack_ipv4=False, *args, **kwargs):
        self.unpack_ipv4 = unpack_ipv4
        self.protocol = protocol
        self.default_validators, invalid_error_message = \
            validators.ip_address_validators(protocol, unpack_ipv4)
        self.default_error_messages['invalid'] = invalid_error_message
        kwargs['max_length'] = 39
        super().__init__(verbose_name, name, *args, **kwargs)

    def check(self, **kwargs):
        return [
            *super().check(**kwargs),
            *self._check_blank_and_null_values(**kwargs),
        ]

    def _check_blank_and_null_values(self, **kwargs):
        if not getattr(self, 'null', False) and getattr(self, 'blank', False):
            return [
                checks.Error(
                    'GenericIPAddressFields cannot have blank=True if null=False, '
                    'as blank values are stored as nulls.',
                    obj=self,
                    id='fields.E150',
                )
            ]
        return []

    def deconstruct(self):
        name, path, args, kwargs = super().deconstruct()
        if self.unpack_ipv4 is not False:
            kwargs['unpack_ipv4'] = self.unpack_ipv4
        if self.protocol != "both":
            kwargs['protocol'] = self.protocol
        if kwargs.get("max_length") == 39:
            del kwargs['max_length']
        return name, path, args, kwargs

    def get_internal_type(self):
        return "GenericIPAddressField"

    def to_python(self, value):
        if value is None:
            return None
        if not isinstance(value, str):
            value = str(value)
        value = value.strip()
        if ':' in value:
            return clean_ipv6_address(value, self.unpack_ipv4, self.error_messages['invalid'])
        return value

    def get_db_prep_value(self, value, connection, prepared=False):
        if not prepared:
            value = self.get_prep_value(value)
        return connection.ops.adapt_ipaddressfield_value(value)

    def get_prep_value(self, value):
        value = super().get_prep_value(value)
        if value is None:
            return None
        if value and ':' in value:
            try:
                return clean_ipv6_address(value, self.unpack_ipv4)
            except exceptions.ValidationError:
                pass
        return str(value)

    def formfield(self, **kwargs):
        return super().formfield(**{
            'protocol': self.protocol,
            'form_class': forms.GenericIPAddressField,
            **kwargs,
        })
```
### 183 - django/db/models/fields/__init__.py:

Start line: 2335, End line: 2384

```python
class AutoFieldMixin:
    db_returning = True

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
### 185 - django/db/models/fields/__init__.py:

Start line: 2081, End line: 2122

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
### 188 - django/db/models/fields/__init__.py:

Start line: 1242, End line: 1283

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
### 191 - django/db/models/fields/__init__.py:

Start line: 2415, End line: 2440

```python
class AutoField(AutoFieldMixin, IntegerField, metaclass=AutoFieldMeta):

    def get_internal_type(self):
        return 'AutoField'

    def rel_db_type(self, connection):
        return IntegerField().db_type(connection=connection)


class BigAutoField(AutoFieldMixin, BigIntegerField):

    def get_internal_type(self):
        return 'BigAutoField'

    def rel_db_type(self, connection):
        return BigIntegerField().db_type(connection=connection)


class SmallAutoField(AutoFieldMixin, SmallIntegerField):

    def get_internal_type(self):
        return 'SmallAutoField'

    def rel_db_type(self, connection):
        return SmallIntegerField().db_type(connection=connection)
```
### 202 - django/db/models/fields/__init__.py:

Start line: 1919, End line: 1938

```python
class NullBooleanField(BooleanField):
    default_error_messages = {
        'invalid': _('“%(value)s” value must be either None, True or False.'),
        'invalid_nullable': _('“%(value)s” value must be either None, True or False.'),
    }
    description = _("Boolean (Either True, False or None)")

    def __init__(self, *args, **kwargs):
        kwargs['null'] = True
        kwargs['blank'] = True
        super().__init__(*args, **kwargs)

    def deconstruct(self):
        name, path, args, kwargs = super().deconstruct()
        del kwargs['null']
        del kwargs['blank']
        return name, path, args, kwargs

    def get_internal_type(self):
        return "NullBooleanField"
```
