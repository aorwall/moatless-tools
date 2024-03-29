# django__django-16111

| **django/django** | `f71b0cf769d9ac582ee3d1a8c33d73dad3a770da` |
| ---- | ---- |
| **No of patches** | 2 |
| **All found context length** | 7295 |
| **Any found context length** | 7295 |
| **Avg pos** | 9.5 |
| **Min pos** | 19 |
| **Max pos** | 19 |
| **Top file pos** | 3 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/db/backends/mysql/features.py b/django/db/backends/mysql/features.py
--- a/django/db/backends/mysql/features.py
+++ b/django/db/backends/mysql/features.py
@@ -81,7 +81,7 @@ def test_collations(self):
             "swedish_ci": f"{charset}_swedish_ci",
         }
 
-    test_now_utc_template = "UTC_TIMESTAMP"
+    test_now_utc_template = "UTC_TIMESTAMP(6)"
 
     @cached_property
     def django_test_skips(self):
diff --git a/django/db/models/functions/datetime.py b/django/db/models/functions/datetime.py
--- a/django/db/models/functions/datetime.py
+++ b/django/db/models/functions/datetime.py
@@ -223,6 +223,19 @@ def as_postgresql(self, compiler, connection, **extra_context):
             compiler, connection, template="STATEMENT_TIMESTAMP()", **extra_context
         )
 
+    def as_mysql(self, compiler, connection, **extra_context):
+        return self.as_sql(
+            compiler, connection, template="CURRENT_TIMESTAMP(6)", **extra_context
+        )
+
+    def as_sqlite(self, compiler, connection, **extra_context):
+        return self.as_sql(
+            compiler,
+            connection,
+            template="STRFTIME('%%Y-%%m-%%d %%H:%%M:%%f', 'NOW')",
+            **extra_context,
+        )
+
 
 class TruncBase(TimezoneMixin, Transform):
     kind = None

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/db/backends/mysql/features.py | 84 | 84 | 19 | 3 | 7295
| django/db/models/functions/datetime.py | 226 | 226 | - | 17 | -


## Problem Statement

```
Add support for microseconds to Now() on MySQL and SQLite.
Description
	
Add support for microseconds to Now() on MySQL and SQLite.
​PR

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 django/db/models/fields/__init__.py | 1744 | 1806| 359 | 359 | 18735 | 
| 2 | 2 django/db/backends/sqlite3/_functions.py | 227 | 271| 337 | 696 | 22591 | 
| 3 | **3 django/db/backends/mysql/features.py** | 86 | 160| 597 | 1293 | 24962 | 
| 4 | 4 django/db/backends/mysql/operations.py | 255 | 272| 125 | 1418 | 29140 | 
| 5 | 4 django/db/backends/mysql/operations.py | 113 | 136| 333 | 1751 | 29140 | 
| 6 | **4 django/db/backends/mysql/features.py** | 1 | 64| 457 | 2208 | 29140 | 
| 7 | 4 django/db/backends/mysql/operations.py | 274 | 297| 172 | 2380 | 29140 | 
| 8 | 4 django/db/backends/mysql/operations.py | 344 | 368| 280 | 2660 | 29140 | 
| 9 | 4 django/db/backends/mysql/operations.py | 138 | 149| 135 | 2795 | 29140 | 
| 10 | 5 django/db/backends/base/features.py | 6 | 221| 1745 | 4540 | 32246 | 
| 11 | **5 django/db/backends/mysql/features.py** | 271 | 330| 439 | 4979 | 32246 | 
| 12 | 5 django/db/backends/sqlite3/_functions.py | 196 | 224| 411 | 5390 | 32246 | 
| 13 | 5 django/db/backends/sqlite3/_functions.py | 274 | 288| 141 | 5531 | 32246 | 
| 14 | 6 django/db/backends/sqlite3/operations.py | 262 | 294| 205 | 5736 | 35733 | 
| 15 | **6 django/db/backends/mysql/features.py** | 162 | 269| 728 | 6464 | 35733 | 
| 16 | 6 django/db/backends/sqlite3/_functions.py | 144 | 161| 180 | 6644 | 35733 | 
| 17 | 7 django/db/models/functions/mixins.py | 26 | 58| 262 | 6906 | 36163 | 
| 18 | 7 django/db/backends/mysql/operations.py | 68 | 86| 214 | 7120 | 36163 | 
| **-> 19 <-** | **7 django/db/backends/mysql/features.py** | 66 | 84| 175 | 7295 | 36163 | 
| 20 | 8 django/db/backends/oracle/features.py | 1 | 84| 718 | 8013 | 37480 | 
| 21 | 9 django/db/backends/oracle/operations.py | 170 | 189| 281 | 8294 | 43700 | 
| 22 | 9 django/db/backends/sqlite3/operations.py | 72 | 140| 613 | 8907 | 43700 | 
| 23 | 10 django/db/backends/oracle/functions.py | 1 | 27| 194 | 9101 | 43894 | 
| 24 | 10 django/db/backends/sqlite3/_functions.py | 164 | 193| 257 | 9358 | 43894 | 
| 25 | 10 django/db/backends/sqlite3/_functions.py | 1 | 37| 163 | 9521 | 43894 | 
| 26 | 10 django/db/backends/mysql/operations.py | 1 | 42| 354 | 9875 | 43894 | 
| 27 | 11 django/db/models/expressions.py | 1 | 34| 228 | 10103 | 57098 | 
| 28 | 11 django/db/backends/oracle/operations.py | 191 | 204| 232 | 10335 | 57098 | 
| 29 | 12 django/db/backends/sqlite3/features.py | 115 | 149| 245 | 10580 | 58345 | 
| 30 | 12 django/db/backends/sqlite3/operations.py | 1 | 42| 314 | 10894 | 58345 | 
| 31 | 12 django/db/backends/sqlite3/_functions.py | 106 | 122| 161 | 11055 | 58345 | 
| 32 | 12 django/db/backends/sqlite3/_functions.py | 291 | 513| 979 | 12034 | 58345 | 
| 33 | 12 django/db/backends/oracle/operations.py | 149 | 168| 309 | 12343 | 58345 | 
| 34 | 13 django/db/backends/base/operations.py | 180 | 259| 628 | 12971 | 64334 | 
| 35 | 13 django/db/backends/base/features.py | 222 | 360| 1208 | 14179 | 64334 | 
| 36 | 14 django/db/backends/sqlite3/base.py | 355 | 377| 183 | 14362 | 67424 | 
| 37 | 15 django/db/models/functions/text.py | 1 | 39| 266 | 14628 | 69793 | 
| 38 | 15 django/db/backends/mysql/operations.py | 370 | 389| 225 | 14853 | 69793 | 
| 39 | 15 django/db/backends/sqlite3/features.py | 1 | 64| 632 | 15485 | 69793 | 
| 40 | 16 django/db/backends/oracle/utils.py | 1 | 60| 339 | 15824 | 70394 | 
| 41 | **17 django/db/models/functions/datetime.py** | 349 | 428| 431 | 16255 | 73218 | 
| 42 | 17 django/db/backends/sqlite3/_functions.py | 125 | 141| 230 | 16485 | 73218 | 
| 43 | 17 django/db/backends/oracle/operations.py | 107 | 119| 220 | 16705 | 73218 | 
| 44 | 17 django/db/backends/sqlite3/features.py | 66 | 113| 383 | 17088 | 73218 | 
| 45 | 18 django/db/models/fields/json.py | 264 | 282| 170 | 17258 | 77709 | 
| 46 | 19 django/contrib/gis/db/backends/mysql/features.py | 1 | 22| 178 | 17436 | 77888 | 
| 47 | 19 django/db/models/expressions.py | 759 | 793| 270 | 17706 | 77888 | 
| 48 | 19 django/db/backends/sqlite3/operations.py | 356 | 415| 577 | 18283 | 77888 | 
| 49 | 19 django/db/backends/sqlite3/_functions.py | 40 | 84| 671 | 18954 | 77888 | 
| 50 | 19 django/db/backends/mysql/operations.py | 44 | 66| 325 | 19279 | 77888 | 
| 51 | 19 django/db/backends/mysql/operations.py | 151 | 201| 425 | 19704 | 77888 | 
| 52 | 19 django/db/backends/oracle/operations.py | 582 | 609| 249 | 19953 | 77888 | 
| 53 | 19 django/db/backends/oracle/operations.py | 611 | 631| 212 | 20165 | 77888 | 
| 54 | 19 django/db/models/fields/__init__.py | 1541 | 1570| 284 | 20449 | 77888 | 
| 55 | 19 django/db/backends/base/operations.py | 1 | 101| 815 | 21264 | 77888 | 
| 56 | 19 django/db/backends/mysql/operations.py | 436 | 465| 274 | 21538 | 77888 | 
| 57 | 20 django/conf/locale/nl/formats.py | 36 | 93| 1456 | 22994 | 79870 | 
| 58 | 20 django/db/models/fields/__init__.py | 2493 | 2521| 217 | 23211 | 79870 | 
| 59 | 20 django/db/backends/mysql/operations.py | 88 | 111| 273 | 23484 | 79870 | 
| 60 | 20 django/db/backends/oracle/operations.py | 83 | 105| 323 | 23807 | 79870 | 
| 61 | 21 django/conf/locale/nb/formats.py | 5 | 42| 610 | 24417 | 80525 | 
| 62 | 22 django/utils/timesince.py | 27 | 102| 716 | 25133 | 81501 | 
| 63 | 22 django/db/backends/sqlite3/operations.py | 333 | 354| 155 | 25288 | 81501 | 
| 64 | **22 django/db/models/functions/datetime.py** | 125 | 224| 551 | 25839 | 81501 | 
| 65 | 23 django/db/backends/mysql/compiler.py | 51 | 81| 240 | 26079 | 82137 | 
| 66 | 24 django/db/migrations/questioner.py | 249 | 267| 177 | 26256 | 84833 | 
| 67 | 25 django/db/backends/mysql/base.py | 1 | 50| 404 | 26660 | 88345 | 
| 68 | 25 django/conf/locale/nl/formats.py | 5 | 35| 481 | 27141 | 88345 | 
| 69 | 26 django/conf/locale/hu/formats.py | 5 | 31| 304 | 27445 | 88694 | 
| 70 | 26 django/db/backends/oracle/operations.py | 21 | 81| 630 | 28075 | 88694 | 
| 71 | **26 django/db/models/functions/datetime.py** | 227 | 268| 325 | 28400 | 88694 | 
| 72 | 27 django/db/backends/mysql/validation.py | 38 | 78| 291 | 28691 | 89225 | 
| 73 | 28 django/conf/locale/ky/formats.py | 5 | 33| 414 | 29105 | 89684 | 
| 74 | 29 django/contrib/gis/db/backends/mysql/operations.py | 57 | 89| 186 | 29291 | 90537 | 
| 75 | 30 django/db/backends/postgresql/features.py | 1 | 103| 792 | 30083 | 91329 | 
| 76 | 31 django/conf/locale/cs/formats.py | 5 | 44| 609 | 30692 | 91983 | 
| 77 | 31 django/db/backends/base/operations.py | 167 | 178| 117 | 30809 | 91983 | 
| 78 | 32 django/conf/locale/nn/formats.py | 5 | 42| 610 | 31419 | 92638 | 
| 79 | 33 django/conf/locale/sv/formats.py | 5 | 36| 480 | 31899 | 93163 | 
| 80 | 33 django/db/backends/sqlite3/operations.py | 44 | 70| 231 | 32130 | 93163 | 
| 81 | 34 django/conf/locale/it/formats.py | 5 | 44| 772 | 32902 | 93980 | 
| 82 | 35 django/conf/locale/uk/formats.py | 5 | 36| 424 | 33326 | 94449 | 
| 83 | 36 django/conf/locale/sr_Latn/formats.py | 5 | 45| 741 | 34067 | 95235 | 
| 84 | 37 django/conf/locale/zh_Hans/formats.py | 5 | 43| 635 | 34702 | 95915 | 
| 85 | 38 django/conf/locale/sr/formats.py | 5 | 45| 742 | 35444 | 96702 | 
| 86 | 39 django/conf/locale/lt/formats.py | 5 | 46| 681 | 36125 | 97428 | 
| 87 | 39 django/db/models/fields/__init__.py | 1283 | 1315| 267 | 36392 | 97428 | 
| 88 | 40 django/conf/locale/ckb/formats.py | 5 | 22| 152 | 36544 | 97624 | 
| 89 | 40 django/db/models/fields/__init__.py | 2406 | 2427| 197 | 36741 | 97624 | 
| 90 | 41 django/conf/locale/es_NI/formats.py | 3 | 27| 271 | 37012 | 97911 | 
| 91 | 42 django/conf/locale/es_PR/formats.py | 3 | 28| 253 | 37265 | 98180 | 
| 92 | 43 django/conf/locale/lv/formats.py | 5 | 47| 705 | 37970 | 98930 | 
| 93 | 43 django/db/backends/mysql/operations.py | 391 | 434| 353 | 38323 | 98930 | 
| 94 | 44 django/conf/locale/fi/formats.py | 5 | 37| 434 | 38757 | 99409 | 
| 95 | 45 django/conf/locale/uz/formats.py | 5 | 31| 416 | 39173 | 99870 | 
| 96 | 46 django/conf/locale/es_MX/formats.py | 3 | 27| 290 | 39463 | 100176 | 
| 97 | 47 django/db/models/functions/comparison.py | 39 | 48| 130 | 39593 | 101931 | 
| 98 | 48 django/conf/locale/ml/formats.py | 5 | 44| 631 | 40224 | 102607 | 
| 99 | 49 django/db/models/functions/math.py | 157 | 173| 143 | 40367 | 104039 | 
| 100 | 49 django/utils/timesince.py | 1 | 24| 260 | 40627 | 104039 | 
| 101 | 49 django/db/backends/base/operations.py | 127 | 165| 314 | 40941 | 104039 | 
| 102 | 50 django/conf/locale/id/formats.py | 5 | 50| 678 | 41619 | 104762 | 
| 103 | 51 django/contrib/humanize/templatetags/humanize.py | 281 | 323| 379 | 41998 | 107761 | 
| 104 | 51 django/db/backends/sqlite3/operations.py | 142 | 167| 280 | 42278 | 107761 | 
| 105 | 51 django/db/backends/oracle/operations.py | 696 | 727| 314 | 42592 | 107761 | 
| 106 | 51 django/db/backends/mysql/base.py | 397 | 421| 211 | 42803 | 107761 | 
| 107 | 52 django/db/backends/sqlite3/schema.py | 123 | 174| 527 | 43330 | 112344 | 
| 108 | 53 django/conf/locale/pl/formats.py | 5 | 31| 332 | 43662 | 112721 | 
| 109 | 54 django/conf/locale/zh_Hant/formats.py | 5 | 43| 635 | 44297 | 113401 | 
| 110 | 54 django/db/backends/mysql/operations.py | 299 | 310| 165 | 44462 | 113401 | 
| 111 | 55 django/db/models/sql/query.py | 2259 | 2290| 266 | 44728 | 136212 | 
| 112 | 56 django/conf/locale/ru/formats.py | 5 | 31| 366 | 45094 | 136623 | 
| 113 | 56 django/db/backends/oracle/operations.py | 131 | 147| 248 | 45342 | 136623 | 
| 114 | 57 django/db/backends/postgresql/operations.py | 246 | 317| 556 | 45898 | 139612 | 
| 115 | 57 django/db/backends/mysql/base.py | 370 | 395| 213 | 46111 | 139612 | 
| 116 | 58 django/template/defaulttags.py | 1141 | 1161| 160 | 46271 | 150385 | 
| 117 | 58 django/contrib/gis/db/backends/mysql/operations.py | 1 | 55| 461 | 46732 | 150385 | 
| 118 | 58 django/db/backends/oracle/operations.py | 564 | 580| 190 | 46922 | 150385 | 
| 119 | 59 django/conf/locale/hr/formats.py | 5 | 45| 747 | 47669 | 151177 | 
| 120 | 60 django/conf/locale/mk/formats.py | 5 | 41| 599 | 48268 | 151821 | 
| 121 | 61 django/conf/locale/th/formats.py | 5 | 34| 354 | 48622 | 152220 | 
| 122 | 61 django/db/backends/sqlite3/operations.py | 313 | 331| 157 | 48779 | 152220 | 
| 123 | 62 django/conf/locale/cy/formats.py | 5 | 34| 531 | 49310 | 152796 | 
| 124 | 63 django/conf/locale/ko/formats.py | 38 | 55| 384 | 49694 | 153705 | 
| 125 | 64 django/db/models/functions/window.py | 94 | 121| 150 | 49844 | 154364 | 
| 126 | 65 django/forms/utils.py | 207 | 244| 274 | 50118 | 156083 | 
| 127 | 65 django/contrib/humanize/templatetags/humanize.py | 207 | 257| 647 | 50765 | 156083 | 
| 128 | 66 django/conf/locale/ka/formats.py | 29 | 49| 476 | 51241 | 156922 | 
| 129 | 67 django/conf/locale/sl/formats.py | 5 | 45| 713 | 51954 | 157680 | 
| 130 | 67 django/db/backends/base/features.py | 362 | 380| 173 | 52127 | 157680 | 
| 131 | 68 django/db/models/sql/compiler.py | 1942 | 2003| 588 | 52715 | 174052 | 
| 132 | 69 django/conf/locale/fr/formats.py | 5 | 34| 458 | 53173 | 174555 | 
| 133 | 69 django/db/models/fields/__init__.py | 1219 | 1242| 153 | 53326 | 174555 | 
| 134 | 69 django/db/backends/oracle/operations.py | 1 | 18| 149 | 53475 | 174555 | 
| 135 | 70 django/conf/locale/en_GB/formats.py | 5 | 42| 673 | 54148 | 175273 | 
| 136 | 70 django/db/backends/oracle/operations.py | 339 | 366| 271 | 54419 | 175273 | 
| 137 | 71 django/conf/locale/sk/formats.py | 5 | 31| 336 | 54755 | 175654 | 
| 138 | 72 django/conf/locale/fa/formats.py | 5 | 22| 148 | 54903 | 175846 | 
| 139 | 73 django/conf/locale/el/formats.py | 5 | 35| 460 | 55363 | 176351 | 
| 140 | 73 django/conf/locale/ko/formats.py | 5 | 37| 480 | 55843 | 176351 | 
| 141 | 74 django/conf/locale/en/formats.py | 7 | 52| 709 | 56552 | 177231 | 
| 142 | 75 django/conf/locale/de_CH/formats.py | 5 | 36| 409 | 56961 | 177685 | 
| 143 | 75 django/db/models/functions/math.py | 176 | 213| 251 | 57212 | 177685 | 
| 144 | 75 django/db/backends/oracle/features.py | 86 | 150| 604 | 57816 | 177685 | 
| 145 | 76 django/conf/locale/ca/formats.py | 5 | 31| 279 | 58095 | 178009 | 
| 146 | 76 django/db/backends/oracle/operations.py | 299 | 317| 213 | 58308 | 178009 | 
| 147 | 76 django/db/models/fields/__init__.py | 1404 | 1428| 190 | 58498 | 178009 | 
| 148 | 77 django/contrib/gis/db/models/functions.py | 108 | 134| 169 | 58667 | 182046 | 
| 149 | 78 django/db/backends/mysql/schema.py | 45 | 54| 134 | 58801 | 184000 | 
| 150 | 79 django/conf/locale/eo/formats.py | 5 | 45| 705 | 59506 | 184750 | 
| 151 | 80 django/conf/locale/ar_DZ/formats.py | 5 | 30| 251 | 59757 | 185046 | 
| 152 | 80 django/db/backends/sqlite3/_functions.py | 85 | 103| 334 | 60091 | 185046 | 
| 153 | 80 django/contrib/humanize/templatetags/humanize.py | 170 | 204| 280 | 60371 | 185046 | 
| 154 | 81 django/conf/locale/pt/formats.py | 5 | 40| 589 | 60960 | 185680 | 
| 155 | 82 django/conf/locale/de/formats.py | 5 | 30| 311 | 61271 | 186036 | 
| 156 | 83 django/conf/locale/bn/formats.py | 5 | 33| 293 | 61564 | 186373 | 
| 157 | 84 django/db/models/lookups.py | 360 | 411| 306 | 61870 | 191691 | 
| 158 | 84 django/db/backends/oracle/operations.py | 235 | 283| 411 | 62281 | 191691 | 
| 159 | 84 django/db/models/functions/math.py | 59 | 107| 293 | 62574 | 191691 | 
| 160 | 85 django/conf/locale/es_CO/formats.py | 3 | 27| 263 | 62837 | 191970 | 
| 161 | 86 django/conf/locale/es_AR/formats.py | 5 | 31| 273 | 63110 | 192288 | 
| 162 | 86 django/db/models/functions/text.py | 42 | 64| 156 | 63266 | 192288 | 
| 163 | 87 django/conf/locale/mn/formats.py | 5 | 22| 120 | 63386 | 192452 | 
| 164 | 87 django/db/backends/sqlite3/base.py | 174 | 231| 511 | 63897 | 192452 | 
| 165 | 88 django/conf/locale/en_AU/formats.py | 5 | 42| 673 | 64570 | 193170 | 
| 166 | 88 django/db/backends/mysql/schema.py | 1 | 43| 471 | 65041 | 193170 | 
| 167 | 89 django/conf/locale/az/formats.py | 5 | 31| 363 | 65404 | 193578 | 
| 168 | 90 django/conf/locale/et/formats.py | 5 | 22| 132 | 65536 | 193754 | 
| 169 | 91 django/conf/locale/sq/formats.py | 5 | 22| 127 | 65663 | 193925 | 
| 170 | 92 django/conf/locale/pt_BR/formats.py | 5 | 35| 468 | 66131 | 194438 | 
| 171 | 93 django/contrib/postgres/fields/ranges.py | 162 | 209| 264 | 66395 | 196848 | 
| 172 | 93 django/db/backends/base/operations.py | 745 | 771| 222 | 66617 | 196848 | 
| 173 | 94 django/conf/locale/tg/formats.py | 5 | 33| 401 | 67018 | 197294 | 
| 174 | 95 django/conf/locale/es/formats.py | 5 | 31| 286 | 67304 | 197625 | 
| 175 | 96 django/conf/locale/ig/formats.py | 5 | 33| 387 | 67691 | 198057 | 
| 176 | 97 django/conf/locale/ms/formats.py | 5 | 39| 607 | 68298 | 198709 | 
| 177 | 98 django/conf/locale/ar/formats.py | 5 | 22| 134 | 68432 | 198887 | 
| 178 | 99 django/conf/locale/da/formats.py | 5 | 27| 249 | 68681 | 199181 | 


## Patch

```diff
diff --git a/django/db/backends/mysql/features.py b/django/db/backends/mysql/features.py
--- a/django/db/backends/mysql/features.py
+++ b/django/db/backends/mysql/features.py
@@ -81,7 +81,7 @@ def test_collations(self):
             "swedish_ci": f"{charset}_swedish_ci",
         }
 
-    test_now_utc_template = "UTC_TIMESTAMP"
+    test_now_utc_template = "UTC_TIMESTAMP(6)"
 
     @cached_property
     def django_test_skips(self):
diff --git a/django/db/models/functions/datetime.py b/django/db/models/functions/datetime.py
--- a/django/db/models/functions/datetime.py
+++ b/django/db/models/functions/datetime.py
@@ -223,6 +223,19 @@ def as_postgresql(self, compiler, connection, **extra_context):
             compiler, connection, template="STATEMENT_TIMESTAMP()", **extra_context
         )
 
+    def as_mysql(self, compiler, connection, **extra_context):
+        return self.as_sql(
+            compiler, connection, template="CURRENT_TIMESTAMP(6)", **extra_context
+        )
+
+    def as_sqlite(self, compiler, connection, **extra_context):
+        return self.as_sql(
+            compiler,
+            connection,
+            template="STRFTIME('%%Y-%%m-%%d %%H:%%M:%%f', 'NOW')",
+            **extra_context,
+        )
+
 
 class TruncBase(TimezoneMixin, Transform):
     kind = None

```

## Test Patch

```diff
diff --git a/tests/db_functions/datetime/test_now.py b/tests/db_functions/datetime/test_now.py
--- a/tests/db_functions/datetime/test_now.py
+++ b/tests/db_functions/datetime/test_now.py
@@ -1,6 +1,8 @@
 from datetime import datetime, timedelta
 
-from django.db.models.functions import Now
+from django.db import connection
+from django.db.models import TextField
+from django.db.models.functions import Cast, Now
 from django.test import TestCase
 from django.utils import timezone
 
@@ -47,3 +49,17 @@ def test_basic(self):
             ["How to Time Travel"],
             lambda a: a.title,
         )
+
+    def test_microseconds(self):
+        Article.objects.create(
+            title="How to Django",
+            text=lorem_ipsum,
+            written=timezone.now(),
+        )
+        now_string = (
+            Article.objects.annotate(now_string=Cast(Now(), TextField()))
+            .get()
+            .now_string
+        )
+        precision = connection.features.time_cast_precision
+        self.assertRegex(now_string, rf"^.*\.\d{{1,{precision}}}")

```


## Code snippets

### 1 - django/db/models/fields/__init__.py:

Start line: 1744, End line: 1806

```python
class DurationField(Field):
    """
    Store timedelta objects.

    Use interval on PostgreSQL, INTERVAL DAY TO SECOND on Oracle, and bigint
    of microseconds on other databases.
    """

    empty_strings_allowed = False
    default_error_messages = {
        "invalid": _(
            "“%(value)s” value has an invalid format. It must be in "
            "[DD] [[HH:]MM:]ss[.uuuuuu] format."
        )
    }
    description = _("Duration")

    def get_internal_type(self):
        return "DurationField"

    def to_python(self, value):
        if value is None:
            return value
        if isinstance(value, datetime.timedelta):
            return value
        try:
            parsed = parse_duration(value)
        except ValueError:
            pass
        else:
            if parsed is not None:
                return parsed

        raise exceptions.ValidationError(
            self.error_messages["invalid"],
            code="invalid",
            params={"value": value},
        )

    def get_db_prep_value(self, value, connection, prepared=False):
        if connection.features.has_native_duration_field:
            return value
        if value is None:
            return None
        return duration_microseconds(value)

    def get_db_converters(self, connection):
        converters = []
        if not connection.features.has_native_duration_field:
            converters.append(connection.ops.convert_durationfield_value)
        return converters + super().get_db_converters(connection)

    def value_to_string(self, obj):
        val = self.value_from_object(obj)
        return "" if val is None else duration_string(val)

    def formfield(self, **kwargs):
        return super().formfield(
            **{
                "form_class": forms.DurationField,
                **kwargs,
            }
        )
```
### 2 - django/db/backends/sqlite3/_functions.py:

Start line: 227, End line: 271

```python
def _sqlite_time_extract(lookup_type, dt):
    if dt is None:
        return None
    try:
        dt = typecast_time(dt)
    except (ValueError, TypeError):
        return None
    return getattr(dt, lookup_type)


def _sqlite_prepare_dtdelta_param(conn, param):
    if conn in ["+", "-"]:
        if isinstance(param, int):
            return timedelta(0, 0, param)
        else:
            return typecast_timestamp(param)
    return param


def _sqlite_format_dtdelta(connector, lhs, rhs):
    """
    LHS and RHS can be either:
    - An integer number of microseconds
    - A string representing a datetime
    - A scalar value, e.g. float
    """
    if connector is None or lhs is None or rhs is None:
        return None
    connector = connector.strip()
    try:
        real_lhs = _sqlite_prepare_dtdelta_param(connector, lhs)
        real_rhs = _sqlite_prepare_dtdelta_param(connector, rhs)
    except (ValueError, TypeError):
        return None
    if connector == "+":
        # typecast_timestamp() returns a date or a datetime without timezone.
        # It will be formatted as "%Y-%m-%d" or "%Y-%m-%d %H:%M:%S[.%f]"
        out = str(real_lhs + real_rhs)
    elif connector == "-":
        out = str(real_lhs - real_rhs)
    elif connector == "*":
        out = real_lhs * real_rhs
    else:
        out = real_lhs / real_rhs
    return out
```
### 3 - django/db/backends/mysql/features.py:

Start line: 86, End line: 160

```python
class DatabaseFeatures(BaseDatabaseFeatures):

    @cached_property
    def django_test_skips(self):
        skips = {
            "This doesn't work on MySQL.": {
                "db_functions.comparison.test_greatest.GreatestTests."
                "test_coalesce_workaround",
                "db_functions.comparison.test_least.LeastTests."
                "test_coalesce_workaround",
            },
            "Running on MySQL requires utf8mb4 encoding (#18392).": {
                "model_fields.test_textfield.TextFieldTests.test_emoji",
                "model_fields.test_charfield.TestCharField.test_emoji",
            },
            "MySQL doesn't support functional indexes on a function that "
            "returns JSON": {
                "schema.tests.SchemaTests.test_func_index_json_key_transform",
            },
            "MySQL supports multiplying and dividing DurationFields by a "
            "scalar value but it's not implemented (#25287).": {
                "expressions.tests.FTimeDeltaTests.test_durationfield_multiply_divide",
            },
            "UPDATE ... ORDER BY syntax on MySQL/MariaDB does not support ordering by"
            "related fields.": {
                "update.tests.AdvancedTests."
                "test_update_ordered_by_inline_m2m_annotation",
                "update.tests.AdvancedTests.test_update_ordered_by_m2m_annotation",
            },
        }
        if "ONLY_FULL_GROUP_BY" in self.connection.sql_mode:
            skips.update(
                {
                    "GROUP BY optimization does not work properly when "
                    "ONLY_FULL_GROUP_BY mode is enabled on MySQL, see #31331.": {
                        "aggregation.tests.AggregateTestCase."
                        "test_aggregation_subquery_annotation_multivalued",
                        "annotations.tests.NonAggregateAnnotationTestCase."
                        "test_annotation_aggregate_with_m2o",
                    },
                }
            )
        if self.connection.mysql_is_mariadb and (
            10,
            4,
            3,
        ) < self.connection.mysql_version < (10, 5, 2):
            skips.update(
                {
                    "https://jira.mariadb.org/browse/MDEV-19598": {
                        "schema.tests.SchemaTests."
                        "test_alter_not_unique_field_to_primary_key",
                    },
                }
            )
        if self.connection.mysql_is_mariadb and (
            10,
            4,
            12,
        ) < self.connection.mysql_version < (10, 5):
            skips.update(
                {
                    "https://jira.mariadb.org/browse/MDEV-22775": {
                        "schema.tests.SchemaTests."
                        "test_alter_pk_with_self_referential_field",
                    },
                }
            )
        if not self.supports_explain_analyze:
            skips.update(
                {
                    "MariaDB and MySQL >= 8.0.18 specific.": {
                        "queries.test_explain.ExplainTests.test_mysql_analyze",
                    },
                }
            )
        return skips
```
### 4 - django/db/backends/mysql/operations.py:

Start line: 255, End line: 272

```python
class DatabaseOperations(BaseDatabaseOperations):

    def adapt_datetimefield_value(self, value):
        if value is None:
            return None

        # Expression values are adapted by the database.
        if hasattr(value, "resolve_expression"):
            return value

        # MySQL doesn't support tz-aware datetimes
        if timezone.is_aware(value):
            if settings.USE_TZ:
                value = timezone.make_naive(value, self.connection.timezone)
            else:
                raise ValueError(
                    "MySQL backend does not support timezone-aware datetimes when "
                    "USE_TZ is False."
                )
        return str(value)
```
### 5 - django/db/backends/mysql/operations.py:

Start line: 113, End line: 136

```python
class DatabaseOperations(BaseDatabaseOperations):

    def datetime_trunc_sql(self, lookup_type, sql, params, tzname):
        sql, params = self._convert_sql_to_tz(sql, params, tzname)
        fields = ["year", "month", "day", "hour", "minute", "second"]
        format = ("%Y-", "%m", "-%d", " %H:", "%i", ":%s")
        format_def = ("0000-", "01", "-01", " 00:", "00", ":00")
        if lookup_type == "quarter":
            return (
                f"CAST(DATE_FORMAT(MAKEDATE(YEAR({sql}), 1) + "
                f"INTERVAL QUARTER({sql}) QUARTER - "
                f"INTERVAL 1 QUARTER, %s) AS DATETIME)"
            ), (*params, *params, "%Y-%m-01 00:00:00")
        if lookup_type == "week":
            return (
                f"CAST(DATE_FORMAT("
                f"DATE_SUB({sql}, INTERVAL WEEKDAY({sql}) DAY), %s) AS DATETIME)"
            ), (*params, *params, "%Y-%m-%d 00:00:00")
        try:
            i = fields.index(lookup_type) + 1
        except ValueError:
            pass
        else:
            format_str = "".join(format[:i] + format_def[i:])
            return f"CAST(DATE_FORMAT({sql}, %s) AS DATETIME)", (*params, format_str)
        return sql, params
```
### 6 - django/db/backends/mysql/features.py:

Start line: 1, End line: 64

```python
import operator

from django.db.backends.base.features import BaseDatabaseFeatures
from django.utils.functional import cached_property


class DatabaseFeatures(BaseDatabaseFeatures):
    empty_fetchmany_value = ()
    allows_group_by_pk = True
    related_fields_match_type = True
    # MySQL doesn't support sliced subqueries with IN/ALL/ANY/SOME.
    allow_sliced_subqueries_with_in = False
    has_select_for_update = True
    supports_forward_references = False
    supports_regex_backreferencing = False
    supports_date_lookup_using_string = False
    supports_timezones = False
    requires_explicit_null_ordering_when_grouping = True
    can_release_savepoints = True
    atomic_transactions = False
    can_clone_databases = True
    supports_temporal_subtraction = True
    supports_select_intersection = False
    supports_select_difference = False
    supports_slicing_ordering_in_compound = True
    supports_index_on_text_field = False
    supports_update_conflicts = True
    create_test_procedure_without_params_sql = """
        CREATE PROCEDURE test_procedure ()
        BEGIN
            DECLARE V_I INTEGER;
            SET V_I = 1;
        END;
    """
    create_test_procedure_with_int_param_sql = """
        CREATE PROCEDURE test_procedure (P_I INTEGER)
        BEGIN
            DECLARE V_I INTEGER;
            SET V_I = P_I;
        END;
    """
    create_test_table_with_composite_primary_key = """
        CREATE TABLE test_table_composite_pk (
            column_1 INTEGER NOT NULL,
            column_2 INTEGER NOT NULL,
            PRIMARY KEY(column_1, column_2)
        )
    """
    # Neither MySQL nor MariaDB support partial indexes.
    supports_partial_indexes = False
    # COLLATE must be wrapped in parentheses because MySQL treats COLLATE as an
    # indexed expression.
    collate_as_index_expression = True

    supports_order_by_nulls_modifier = False
    order_by_nulls_first = True
    supports_logical_xor = True

    @cached_property
    def minimum_database_version(self):
        if self.connection.mysql_is_mariadb:
            return (10, 4)
        else:
            return (8,)
```
### 7 - django/db/backends/mysql/operations.py:

Start line: 274, End line: 297

```python
class DatabaseOperations(BaseDatabaseOperations):

    def adapt_timefield_value(self, value):
        if value is None:
            return None

        # Expression values are adapted by the database.
        if hasattr(value, "resolve_expression"):
            return value

        # MySQL doesn't support tz-aware times
        if timezone.is_aware(value):
            raise ValueError("MySQL backend does not support timezone-aware times.")

        return value.isoformat(timespec="microseconds")

    def max_name_length(self):
        return 64

    def pk_default_value(self):
        return "NULL"

    def bulk_insert_sql(self, fields, placeholder_rows):
        placeholder_rows_sql = (", ".join(row) for row in placeholder_rows)
        values_sql = ", ".join("(%s)" % sql for sql in placeholder_rows_sql)
        return "VALUES " + values_sql
```
### 8 - django/db/backends/mysql/operations.py:

Start line: 344, End line: 368

```python
class DatabaseOperations(BaseDatabaseOperations):

    def subtract_temporals(self, internal_type, lhs, rhs):
        lhs_sql, lhs_params = lhs
        rhs_sql, rhs_params = rhs
        if internal_type == "TimeField":
            if self.connection.mysql_is_mariadb:
                # MariaDB includes the microsecond component in TIME_TO_SEC as
                # a decimal. MySQL returns an integer without microseconds.
                return (
                    "CAST((TIME_TO_SEC(%(lhs)s) - TIME_TO_SEC(%(rhs)s)) "
                    "* 1000000 AS SIGNED)"
                ) % {
                    "lhs": lhs_sql,
                    "rhs": rhs_sql,
                }, (
                    *lhs_params,
                    *rhs_params,
                )
            return (
                "((TIME_TO_SEC(%(lhs)s) * 1000000 + MICROSECOND(%(lhs)s)) -"
                " (TIME_TO_SEC(%(rhs)s) * 1000000 + MICROSECOND(%(rhs)s)))"
            ) % {"lhs": lhs_sql, "rhs": rhs_sql}, tuple(lhs_params) * 2 + tuple(
                rhs_params
            ) * 2
        params = (*rhs_params, *lhs_params)
        return "TIMESTAMPDIFF(MICROSECOND, %s, %s)" % (rhs_sql, lhs_sql), params
```
### 9 - django/db/backends/mysql/operations.py:

Start line: 138, End line: 149

```python
class DatabaseOperations(BaseDatabaseOperations):

    def time_trunc_sql(self, lookup_type, sql, params, tzname=None):
        sql, params = self._convert_sql_to_tz(sql, params, tzname)
        fields = {
            "hour": "%H:00:00",
            "minute": "%H:%i:00",
            "second": "%H:%i:%s",
        }
        if lookup_type in fields:
            format_str = fields[lookup_type]
            return f"CAST(DATE_FORMAT({sql}, %s) AS TIME)", (*params, format_str)
        else:
            return f"TIME({sql})", params
```
### 10 - django/db/backends/base/features.py:

Start line: 6, End line: 221

```python
class BaseDatabaseFeatures:
    # An optional tuple indicating the minimum supported database version.
    minimum_database_version = None
    gis_enabled = False
    # Oracle can't group by LOB (large object) data types.
    allows_group_by_lob = True
    allows_group_by_pk = False
    allows_group_by_selected_pks = False
    empty_fetchmany_value = []
    update_can_self_select = True

    # Does the backend distinguish between '' and None?
    interprets_empty_strings_as_nulls = False

    # Does the backend allow inserting duplicate NULL rows in a nullable
    # unique field? All core backends implement this correctly, but other
    # databases such as SQL Server do not.
    supports_nullable_unique_constraints = True

    # Does the backend allow inserting duplicate rows when a unique_together
    # constraint exists and some fields are nullable but not all of them?
    supports_partially_nullable_unique_constraints = True
    # Does the backend support initially deferrable unique constraints?
    supports_deferrable_unique_constraints = False

    can_use_chunked_reads = True
    can_return_columns_from_insert = False
    can_return_rows_from_bulk_insert = False
    has_bulk_insert = True
    uses_savepoints = True
    can_release_savepoints = False

    # If True, don't use integer foreign keys referring to, e.g., positive
    # integer primary keys.
    related_fields_match_type = False
    allow_sliced_subqueries_with_in = True
    has_select_for_update = False
    has_select_for_update_nowait = False
    has_select_for_update_skip_locked = False
    has_select_for_update_of = False
    has_select_for_no_key_update = False
    # Does the database's SELECT FOR UPDATE OF syntax require a column rather
    # than a table?
    select_for_update_of_column = False

    # Does the default test database allow multiple connections?
    # Usually an indication that the test database is in-memory
    test_db_allows_multiple_connections = True

    # Can an object be saved without an explicit primary key?
    supports_unspecified_pk = False

    # Can a fixture contain forward references? i.e., are
    # FK constraints checked at the end of transaction, or
    # at the end of each save operation?
    supports_forward_references = True

    # Does the backend truncate names properly when they are too long?
    truncates_names = False

    # Is there a REAL datatype in addition to floats/doubles?
    has_real_datatype = False
    supports_subqueries_in_group_by = True

    # Does the backend ignore unnecessary ORDER BY clauses in subqueries?
    ignores_unnecessary_order_by_in_subqueries = True

    # Is there a true datatype for uuid?
    has_native_uuid_field = False

    # Is there a true datatype for timedeltas?
    has_native_duration_field = False

    # Does the database driver supports same type temporal data subtraction
    # by returning the type used to store duration field?
    supports_temporal_subtraction = False

    # Does the __regex lookup support backreferencing and grouping?
    supports_regex_backreferencing = True

    # Can date/datetime lookups be performed using a string?
    supports_date_lookup_using_string = True

    # Can datetimes with timezones be used?
    supports_timezones = True

    # Does the database have a copy of the zoneinfo database?
    has_zoneinfo_database = True

    # When performing a GROUP BY, is an ORDER BY NULL required
    # to remove any ordering?
    requires_explicit_null_ordering_when_grouping = False

    # Does the backend order NULL values as largest or smallest?
    nulls_order_largest = False

    # Does the backend support NULLS FIRST and NULLS LAST in ORDER BY?
    supports_order_by_nulls_modifier = True

    # Does the backend orders NULLS FIRST by default?
    order_by_nulls_first = False

    # The database's limit on the number of query parameters.
    max_query_params = None

    # Can an object have an autoincrement primary key of 0?
    allows_auto_pk_0 = True

    # Do we need to NULL a ForeignKey out, or can the constraint check be
    # deferred
    can_defer_constraint_checks = False

    # Does the backend support tablespaces? Default to False because it isn't
    # in the SQL standard.
    supports_tablespaces = False

    # Does the backend reset sequences between tests?
    supports_sequence_reset = True

    # Can the backend introspect the default value of a column?
    can_introspect_default = True

    # Confirm support for introspected foreign keys
    # Every database can do this reliably, except MySQL,
    # which can't do it for MyISAM tables
    can_introspect_foreign_keys = True

    # Map fields which some backends may not be able to differentiate to the
    # field it's introspected as.
    introspected_field_types = {
        "AutoField": "AutoField",
        "BigAutoField": "BigAutoField",
        "BigIntegerField": "BigIntegerField",
        "BinaryField": "BinaryField",
        "BooleanField": "BooleanField",
        "CharField": "CharField",
        "DurationField": "DurationField",
        "GenericIPAddressField": "GenericIPAddressField",
        "IntegerField": "IntegerField",
        "PositiveBigIntegerField": "PositiveBigIntegerField",
        "PositiveIntegerField": "PositiveIntegerField",
        "PositiveSmallIntegerField": "PositiveSmallIntegerField",
        "SmallAutoField": "SmallAutoField",
        "SmallIntegerField": "SmallIntegerField",
        "TimeField": "TimeField",
    }

    # Can the backend introspect the column order (ASC/DESC) for indexes?
    supports_index_column_ordering = True

    # Does the backend support introspection of materialized views?
    can_introspect_materialized_views = False

    # Support for the DISTINCT ON clause
    can_distinct_on_fields = False

    # Does the backend prevent running SQL queries in broken transactions?
    atomic_transactions = True

    # Can we roll back DDL in a transaction?
    can_rollback_ddl = False

    # Does it support operations requiring references rename in a transaction?
    supports_atomic_references_rename = True

    # Can we issue more than one ALTER COLUMN clause in an ALTER TABLE?
    supports_combined_alters = False

    # Does it support foreign keys?
    supports_foreign_keys = True

    # Can it create foreign key constraints inline when adding columns?
    can_create_inline_fk = True

    # Can an index be renamed?
    can_rename_index = False

    # Does it automatically index foreign keys?
    indexes_foreign_keys = True

    # Does it support CHECK constraints?
    supports_column_check_constraints = True
    supports_table_check_constraints = True
    # Does the backend support introspection of CHECK constraints?
    can_introspect_check_constraints = True

    # Does the backend support 'pyformat' style ("... %(name)s ...", {'name': value})
    # parameter passing? Note this can be provided by the backend even if not
    # supported by the Python driver
    supports_paramstyle_pyformat = True

    # Does the backend require literal defaults, rather than parameterized ones?
    requires_literal_defaults = False

    # Does the backend require a connection reset after each material schema change?
    connection_persists_old_columns = False

    # What kind of error does the backend throw when accessing closed cursor?
    closed_cursor_error_class = ProgrammingError

    # Does 'a' LIKE 'A' match?
    has_case_insensitive_like = False

    # Suffix for backends that don't support "SELECT xxx;" queries.
    bare_select_suffix = ""

    # If NULL is implied on columns without needing to be explicitly specified
    implied_column_null = False

    # Does the backend support "select for update" queries with limit (and offset)?
    supports_select_for_update_with_limit = True

    # Does the backend ignore null expressions in GREATEST and LEAST queries unless
    # every expression is null?
    greatest_least_ignores_nulls = False

    # Can the backend clone databases for parallel test execution?
    # ... other code
```
### 11 - django/db/backends/mysql/features.py:

Start line: 271, End line: 330

```python
class DatabaseFeatures(BaseDatabaseFeatures):

    @cached_property
    def supported_explain_formats(self):
        # Alias MySQL's TRADITIONAL to TEXT for consistency with other
        # backends.
        formats = {"JSON", "TEXT", "TRADITIONAL"}
        if not self.connection.mysql_is_mariadb and self.connection.mysql_version >= (
            8,
            0,
            16,
        ):
            formats.add("TREE")
        return formats

    @cached_property
    def supports_transactions(self):
        """
        All storage engines except MyISAM support transactions.
        """
        return self._mysql_storage_engine != "MyISAM"

    uses_savepoints = property(operator.attrgetter("supports_transactions"))
    can_release_savepoints = property(operator.attrgetter("supports_transactions"))

    @cached_property
    def ignores_table_name_case(self):
        return self.connection.mysql_server_data["lower_case_table_names"]

    @cached_property
    def supports_default_in_lead_lag(self):
        # To be added in https://jira.mariadb.org/browse/MDEV-12981.
        return not self.connection.mysql_is_mariadb

    @cached_property
    def can_introspect_json_field(self):
        if self.connection.mysql_is_mariadb:
            return self.can_introspect_check_constraints
        return True

    @cached_property
    def supports_index_column_ordering(self):
        if self._mysql_storage_engine != "InnoDB":
            return False
        if self.connection.mysql_is_mariadb:
            return self.connection.mysql_version >= (10, 8)
        return self.connection.mysql_version >= (8, 0, 1)

    @cached_property
    def supports_expression_indexes(self):
        return (
            not self.connection.mysql_is_mariadb
            and self._mysql_storage_engine != "MyISAM"
            and self.connection.mysql_version >= (8, 0, 13)
        )

    @cached_property
    def can_rename_index(self):
        if self.connection.mysql_is_mariadb:
            return self.connection.mysql_version >= (10, 5, 2)
        return True
```
### 15 - django/db/backends/mysql/features.py:

Start line: 162, End line: 269

```python
class DatabaseFeatures(BaseDatabaseFeatures):

    @cached_property
    def _mysql_storage_engine(self):
        "Internal method used in Django tests. Don't rely on this from your code"
        return self.connection.mysql_server_data["default_storage_engine"]

    @cached_property
    def allows_auto_pk_0(self):
        """
        Autoincrement primary key can be set to 0 if it doesn't generate new
        autoincrement values.
        """
        return "NO_AUTO_VALUE_ON_ZERO" in self.connection.sql_mode

    @cached_property
    def update_can_self_select(self):
        return self.connection.mysql_is_mariadb and self.connection.mysql_version >= (
            10,
            3,
            2,
        )

    @cached_property
    def can_introspect_foreign_keys(self):
        "Confirm support for introspected foreign keys"
        return self._mysql_storage_engine != "MyISAM"

    @cached_property
    def introspected_field_types(self):
        return {
            **super().introspected_field_types,
            "BinaryField": "TextField",
            "BooleanField": "IntegerField",
            "DurationField": "BigIntegerField",
            "GenericIPAddressField": "CharField",
        }

    @cached_property
    def can_return_columns_from_insert(self):
        return self.connection.mysql_is_mariadb and self.connection.mysql_version >= (
            10,
            5,
            0,
        )

    can_return_rows_from_bulk_insert = property(
        operator.attrgetter("can_return_columns_from_insert")
    )

    @cached_property
    def has_zoneinfo_database(self):
        return self.connection.mysql_server_data["has_zoneinfo_database"]

    @cached_property
    def is_sql_auto_is_null_enabled(self):
        return self.connection.mysql_server_data["sql_auto_is_null"]

    @cached_property
    def supports_over_clause(self):
        if self.connection.mysql_is_mariadb:
            return True
        return self.connection.mysql_version >= (8, 0, 2)

    supports_frame_range_fixed_distance = property(
        operator.attrgetter("supports_over_clause")
    )

    @cached_property
    def supports_column_check_constraints(self):
        if self.connection.mysql_is_mariadb:
            return True
        return self.connection.mysql_version >= (8, 0, 16)

    supports_table_check_constraints = property(
        operator.attrgetter("supports_column_check_constraints")
    )

    @cached_property
    def can_introspect_check_constraints(self):
        if self.connection.mysql_is_mariadb:
            return True
        return self.connection.mysql_version >= (8, 0, 16)

    @cached_property
    def has_select_for_update_skip_locked(self):
        if self.connection.mysql_is_mariadb:
            return self.connection.mysql_version >= (10, 6)
        return self.connection.mysql_version >= (8, 0, 1)

    @cached_property
    def has_select_for_update_nowait(self):
        if self.connection.mysql_is_mariadb:
            return True
        return self.connection.mysql_version >= (8, 0, 1)

    @cached_property
    def has_select_for_update_of(self):
        return (
            not self.connection.mysql_is_mariadb
            and self.connection.mysql_version >= (8, 0, 1)
        )

    @cached_property
    def supports_explain_analyze(self):
        return self.connection.mysql_is_mariadb or self.connection.mysql_version >= (
            8,
            0,
            18,
        )
```
### 19 - django/db/backends/mysql/features.py:

Start line: 66, End line: 84

```python
class DatabaseFeatures(BaseDatabaseFeatures):

    @cached_property
    def test_collations(self):
        charset = "utf8"
        if (
            self.connection.mysql_is_mariadb
            and self.connection.mysql_version >= (10, 6)
        ) or (
            not self.connection.mysql_is_mariadb
            and self.connection.mysql_version >= (8, 0, 30)
        ):
            # utf8 is an alias for utf8mb3 in MariaDB 10.6+ and MySQL 8.0.30+.
            charset = "utf8mb3"
        return {
            "ci": f"{charset}_general_ci",
            "non_default": f"{charset}_esperanto_ci",
            "swedish_ci": f"{charset}_swedish_ci",
        }

    test_now_utc_template = "UTC_TIMESTAMP"
```
### 41 - django/db/models/functions/datetime.py:

Start line: 349, End line: 428

```python
class Trunc(TruncBase):

    # RemovedInDjango50Warning: when the deprecation ends, remove is_dst
    # argument.
    def __init__(
        self,
        expression,
        kind,
        output_field=None,
        tzinfo=None,
        is_dst=timezone.NOT_PASSED,
        **extra,
    ):
        self.kind = kind
        super().__init__(
            expression, output_field=output_field, tzinfo=tzinfo, is_dst=is_dst, **extra
        )


class TruncYear(TruncBase):
    kind = "year"


class TruncQuarter(TruncBase):
    kind = "quarter"


class TruncMonth(TruncBase):
    kind = "month"


class TruncWeek(TruncBase):
    """Truncate to midnight on the Monday of the week."""

    kind = "week"


class TruncDay(TruncBase):
    kind = "day"


class TruncDate(TruncBase):
    kind = "date"
    lookup_name = "date"
    output_field = DateField()

    def as_sql(self, compiler, connection):
        # Cast to date rather than truncate to date.
        sql, params = compiler.compile(self.lhs)
        tzname = self.get_tzname()
        return connection.ops.datetime_cast_date_sql(sql, tuple(params), tzname)


class TruncTime(TruncBase):
    kind = "time"
    lookup_name = "time"
    output_field = TimeField()

    def as_sql(self, compiler, connection):
        # Cast to time rather than truncate to time.
        sql, params = compiler.compile(self.lhs)
        tzname = self.get_tzname()
        return connection.ops.datetime_cast_time_sql(sql, tuple(params), tzname)


class TruncHour(TruncBase):
    kind = "hour"


class TruncMinute(TruncBase):
    kind = "minute"


class TruncSecond(TruncBase):
    kind = "second"


DateTimeField.register_lookup(TruncDate)
DateTimeField.register_lookup(TruncTime)
```
### 64 - django/db/models/functions/datetime.py:

Start line: 125, End line: 224

```python
class ExtractYear(Extract):
    lookup_name = "year"


class ExtractIsoYear(Extract):
    """Return the ISO-8601 week-numbering year."""

    lookup_name = "iso_year"


class ExtractMonth(Extract):
    lookup_name = "month"


class ExtractDay(Extract):
    lookup_name = "day"


class ExtractWeek(Extract):
    """
    Return 1-52 or 53, based on ISO-8601, i.e., Monday is the first of the
    week.
    """

    lookup_name = "week"


class ExtractWeekDay(Extract):
    """
    Return Sunday=1 through Saturday=7.

    To replicate this in Python: (mydatetime.isoweekday() % 7) + 1
    """

    lookup_name = "week_day"


class ExtractIsoWeekDay(Extract):
    """Return Monday=1 through Sunday=7, based on ISO-8601."""

    lookup_name = "iso_week_day"


class ExtractQuarter(Extract):
    lookup_name = "quarter"


class ExtractHour(Extract):
    lookup_name = "hour"


class ExtractMinute(Extract):
    lookup_name = "minute"


class ExtractSecond(Extract):
    lookup_name = "second"


DateField.register_lookup(ExtractYear)
DateField.register_lookup(ExtractMonth)
DateField.register_lookup(ExtractDay)
DateField.register_lookup(ExtractWeekDay)
DateField.register_lookup(ExtractIsoWeekDay)
DateField.register_lookup(ExtractWeek)
DateField.register_lookup(ExtractIsoYear)
DateField.register_lookup(ExtractQuarter)

TimeField.register_lookup(ExtractHour)
TimeField.register_lookup(ExtractMinute)
TimeField.register_lookup(ExtractSecond)

DateTimeField.register_lookup(ExtractHour)
DateTimeField.register_lookup(ExtractMinute)
DateTimeField.register_lookup(ExtractSecond)

ExtractYear.register_lookup(YearExact)
ExtractYear.register_lookup(YearGt)
ExtractYear.register_lookup(YearGte)
ExtractYear.register_lookup(YearLt)
ExtractYear.register_lookup(YearLte)

ExtractIsoYear.register_lookup(YearExact)
ExtractIsoYear.register_lookup(YearGt)
ExtractIsoYear.register_lookup(YearGte)
ExtractIsoYear.register_lookup(YearLt)
ExtractIsoYear.register_lookup(YearLte)


class Now(Func):
    template = "CURRENT_TIMESTAMP"
    output_field = DateTimeField()

    def as_postgresql(self, compiler, connection, **extra_context):
        # PostgreSQL's CURRENT_TIMESTAMP means "the time at the start of the
        # transaction". Use STATEMENT_TIMESTAMP to be cross-compatible with
        # other databases.
        return self.as_sql(
            compiler, connection, template="STATEMENT_TIMESTAMP()", **extra_context
        )
```
### 71 - django/db/models/functions/datetime.py:

Start line: 227, End line: 268

```python
class TruncBase(TimezoneMixin, Transform):
    kind = None
    tzinfo = None

    # RemovedInDjango50Warning: when the deprecation ends, remove is_dst
    # argument.
    def __init__(
        self,
        expression,
        output_field=None,
        tzinfo=None,
        is_dst=timezone.NOT_PASSED,
        **extra,
    ):
        self.tzinfo = tzinfo
        self.is_dst = is_dst
        super().__init__(expression, output_field=output_field, **extra)

    def as_sql(self, compiler, connection):
        sql, params = compiler.compile(self.lhs)
        tzname = None
        if isinstance(self.lhs.output_field, DateTimeField):
            tzname = self.get_tzname()
        elif self.tzinfo is not None:
            raise ValueError("tzinfo can only be used with DateTimeField.")
        if isinstance(self.output_field, DateTimeField):
            sql, params = connection.ops.datetime_trunc_sql(
                self.kind, sql, tuple(params), tzname
            )
        elif isinstance(self.output_field, DateField):
            sql, params = connection.ops.date_trunc_sql(
                self.kind, sql, tuple(params), tzname
            )
        elif isinstance(self.output_field, TimeField):
            sql, params = connection.ops.time_trunc_sql(
                self.kind, sql, tuple(params), tzname
            )
        else:
            raise ValueError(
                "Trunc only valid on DateField, TimeField, or DateTimeField."
            )
        return sql, params
```
