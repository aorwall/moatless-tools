# django__django-14792

| **django/django** | `d89f976bddb49fb168334960acc8979c3de991fa` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 5426 |
| **Any found context length** | 5426 |
| **Avg pos** | 20.0 |
| **Min pos** | 20 |
| **Max pos** | 20 |
| **Top file pos** | 7 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/utils/timezone.py b/django/utils/timezone.py
--- a/django/utils/timezone.py
+++ b/django/utils/timezone.py
@@ -72,8 +72,11 @@ def get_current_timezone_name():
 
 
 def _get_timezone_name(timezone):
-    """Return the name of ``timezone``."""
-    return str(timezone)
+    """
+    Return the offset for fixed offset timezones, or the name of timezone if
+    not set.
+    """
+    return timezone.tzname(None) or str(timezone)
 
 # Timezone selection functions.
 

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/utils/timezone.py | 75 | 76 | 20 | 7 | 5426


## Problem Statement

```
Reverse time zone conversion in Trunc()/Extract() database functions.
Description
	
When using a time zone of "Etc/GMT-10" (or similar) for a Trunc class tzinfo, it appears there's a different behavior as of Django 3.2 in the resulting database query. I think it's due to a change in the return value of timezone._get_timezone_name() that's called by the TimezoneMixin.
On Django 3.1 the TimezoneMixin method get_tzname() returns "+10" for a "Etc/GMT-10" time zone after calling ​_get_timezone_name(). This later becomes "-10" in the resulting query due to the return value of _prepare_tzname_delta() of the Postgres DatabaseOperations class, i.e. the time zone 10 hours east from UTC.
SELECT ... DATE_TRUNC(\'day\', "my_model"."start_at" AT TIME ZONE \'-10\') AS "date" ...
On Django 3.2 the TimezoneMixin method get_tzname() returns "Etc/GMT-10" for a "Etc/GMT-10" time zone after calling ​_get_timezone_name(). This later, incorrectly, becomes "Etc/GMT+10" in the resulting query due to the return value of _prepare_tzname_delta() of the Postgres DatabaseOperations class, i.e. the time zone 10 hours west from UTC, which is the opposite direction from the behavior in Django 3.1.
SELECT ... DATE_TRUNC(\'day\', "my_model"."start_at" AT TIME ZONE \'Etc/GMT+10\') AS "date" ...
# Django 3.1
>>> timezone._get_timezone_name(pytz.timezone("Etc/GMT-10"))
'+10'
# Django 3.2
>>> timezone._get_timezone_name(pytz.timezone("Etc/GMT-10"))
'Etc/GMT-10'
The above is the same when using Python's zoneinfo.ZoneInfo() too.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 django/db/models/functions/datetime.py | 187 | 211| 264 | 264 | 2623 | 
| 2 | 1 django/db/models/functions/datetime.py | 243 | 262| 170 | 434 | 2623 | 
| 3 | 1 django/db/models/functions/datetime.py | 213 | 241| 426 | 860 | 2623 | 
| 4 | 1 django/db/models/functions/datetime.py | 265 | 336| 407 | 1267 | 2623 | 
| 5 | 1 django/db/models/functions/datetime.py | 31 | 63| 324 | 1591 | 2623 | 
| 6 | 2 django/db/backends/sqlite3/base.py | 466 | 482| 169 | 1760 | 8686 | 
| 7 | 3 django/utils/dateformat.py | 174 | 191| 185 | 1945 | 11280 | 
| 8 | 4 django/db/backends/mysql/operations.py | 132 | 143| 151 | 2096 | 15007 | 
| 9 | 4 django/db/models/functions/datetime.py | 65 | 88| 270 | 2366 | 15007 | 
| 10 | 4 django/db/models/functions/datetime.py | 1 | 28| 236 | 2602 | 15007 | 
| 11 | 4 django/db/backends/sqlite3/base.py | 517 | 538| 377 | 2979 | 15007 | 
| 12 | 4 django/db/backends/sqlite3/base.py | 448 | 463| 215 | 3194 | 15007 | 
| 13 | 4 django/db/backends/mysql/operations.py | 106 | 130| 350 | 3544 | 15007 | 
| 14 | 4 django/utils/dateformat.py | 128 | 140| 131 | 3675 | 15007 | 
| 15 | 4 django/db/backends/sqlite3/base.py | 426 | 445| 196 | 3871 | 15007 | 
| 16 | 4 django/db/backends/mysql/operations.py | 58 | 76| 225 | 4096 | 15007 | 
| 17 | 5 django/forms/utils.py | 154 | 189| 272 | 4368 | 16308 | 
| 18 | 5 django/db/backends/sqlite3/base.py | 485 | 514| 258 | 4626 | 16308 | 
| 19 | 6 django/db/backends/base/operations.py | 144 | 153| 112 | 4738 | 22070 | 
| **-> 20 <-** | **7 django/utils/timezone.py** | 1 | 106| 688 | 5426 | 23910 | 
| 21 | 7 django/db/models/functions/datetime.py | 91 | 184| 548 | 5974 | 23910 | 
| 22 | 7 django/utils/dateformat.py | 155 | 172| 138 | 6112 | 23910 | 
| 23 | 7 django/utils/dateformat.py | 47 | 126| 557 | 6669 | 23910 | 
| 24 | 8 django/db/backends/oracle/operations.py | 152 | 169| 309 | 6978 | 29903 | 
| 25 | 9 django/db/backends/postgresql/operations.py | 41 | 87| 528 | 7506 | 32464 | 
| 26 | 9 django/db/backends/oracle/operations.py | 171 | 182| 227 | 7733 | 32464 | 
| 27 | 9 django/db/backends/base/operations.py | 113 | 142| 289 | 8022 | 32464 | 
| 28 | 9 django/utils/dateformat.py | 273 | 328| 460 | 8482 | 32464 | 
| 29 | 9 django/db/backends/base/operations.py | 102 | 111| 112 | 8594 | 32464 | 
| 30 | 10 django/db/backends/sqlite3/operations.py | 71 | 133| 611 | 9205 | 35736 | 
| 31 | 10 django/db/backends/oracle/operations.py | 92 | 102| 222 | 9427 | 35736 | 
| 32 | 10 django/db/backends/mysql/operations.py | 78 | 104| 275 | 9702 | 35736 | 
| 33 | 11 django/templatetags/tz.py | 37 | 78| 288 | 9990 | 36921 | 
| 34 | 12 django/conf/locale/tg/formats.py | 5 | 33| 402 | 10392 | 37368 | 
| 35 | 13 django/db/backends/base/base.py | 117 | 138| 186 | 10578 | 42271 | 
| 36 | 14 django/conf/locale/eo/formats.py | 5 | 48| 707 | 11285 | 43023 | 
| 37 | 15 django/conf/locale/lt/formats.py | 5 | 44| 676 | 11961 | 43744 | 
| 38 | 16 django/conf/locale/tk/formats.py | 5 | 33| 402 | 12363 | 44191 | 
| 39 | 17 django/conf/locale/zh_Hans/formats.py | 5 | 43| 635 | 12998 | 44871 | 
| 40 | 17 django/db/backends/oracle/operations.py | 133 | 150| 291 | 13289 | 44871 | 
| 41 | 17 django/db/backends/base/base.py | 140 | 182| 335 | 13624 | 44871 | 
| 42 | 17 django/db/backends/oracle/operations.py | 104 | 115| 212 | 13836 | 44871 | 
| 43 | 18 django/conf/locale/id/formats.py | 5 | 47| 670 | 14506 | 45586 | 
| 44 | 19 django/conf/locale/zh_Hant/formats.py | 5 | 43| 635 | 15141 | 46266 | 
| 45 | 20 django/conf/locale/el/formats.py | 5 | 33| 455 | 15596 | 46766 | 
| 46 | 21 django/conf/locale/sr_Latn/formats.py | 5 | 40| 726 | 16322 | 47537 | 
| 47 | 22 django/conf/locale/fr/formats.py | 5 | 32| 448 | 16770 | 48030 | 
| 48 | 23 django/conf/locale/sr/formats.py | 5 | 40| 726 | 17496 | 48801 | 
| 49 | 24 django/conf/locale/en_AU/formats.py | 5 | 37| 655 | 18151 | 49501 | 
| 50 | 25 django/conf/locale/az/formats.py | 5 | 31| 364 | 18515 | 49910 | 
| 51 | 26 django/conf/locale/sl/formats.py | 5 | 43| 708 | 19223 | 50663 | 
| 52 | 27 django/conf/locale/cy/formats.py | 5 | 33| 529 | 19752 | 51237 | 
| 53 | 28 django/conf/locale/en_GB/formats.py | 5 | 37| 655 | 20407 | 51937 | 
| 54 | 29 django/conf/locale/cs/formats.py | 5 | 41| 600 | 21007 | 52582 | 
| 55 | 30 django/conf/locale/nl/formats.py | 32 | 67| 1371 | 22378 | 54465 | 
| 56 | 31 django/conf/locale/hr/formats.py | 5 | 43| 742 | 23120 | 55252 | 
| 57 | 32 django/conf/locale/pt/formats.py | 5 | 36| 577 | 23697 | 55874 | 
| 58 | 32 django/db/backends/oracle/operations.py | 117 | 131| 229 | 23926 | 55874 | 
| 59 | 33 django/conf/locale/uz/formats.py | 5 | 31| 418 | 24344 | 56337 | 
| 60 | 34 django/conf/locale/hu/formats.py | 5 | 31| 305 | 24649 | 56687 | 
| 61 | 35 django/conf/locale/et/formats.py | 5 | 22| 133 | 24782 | 56864 | 
| 62 | 36 django/conf/locale/bn/formats.py | 5 | 33| 294 | 25076 | 57202 | 
| 63 | 37 django/conf/locale/ka/formats.py | 5 | 43| 773 | 25849 | 58020 | 
| 64 | 38 django/conf/locale/pt_BR/formats.py | 5 | 32| 459 | 26308 | 58524 | 
| 65 | 39 django/conf/locale/uk/formats.py | 5 | 36| 425 | 26733 | 58994 | 
| 66 | 40 django/db/models/fields/__init__.py | 1386 | 1414| 281 | 27014 | 77141 | 
| 67 | 41 django/conf/locale/en/formats.py | 5 | 38| 610 | 27624 | 77796 | 
| 68 | 42 django/utils/dateparse.py | 1 | 66| 763 | 28387 | 79392 | 
| 69 | 43 django/conf/locale/lv/formats.py | 5 | 45| 700 | 29087 | 80137 | 
| 70 | 44 django/conf/locale/ml/formats.py | 5 | 38| 610 | 29697 | 80792 | 
| 71 | 45 django/utils/timesince.py | 1 | 24| 262 | 29959 | 81769 | 
| 72 | 46 django/conf/locale/pl/formats.py | 5 | 29| 321 | 30280 | 82135 | 
| 73 | 46 django/utils/dateformat.py | 194 | 258| 536 | 30816 | 82135 | 
| 74 | 47 django/conf/locale/mk/formats.py | 5 | 39| 594 | 31410 | 82774 | 
| 75 | 47 django/utils/dateformat.py | 260 | 271| 121 | 31531 | 82774 | 
| 76 | 48 django/conf/locale/ru/formats.py | 5 | 31| 367 | 31898 | 83186 | 
| 77 | 49 django/conf/locale/fi/formats.py | 5 | 38| 435 | 32333 | 83666 | 
| 78 | 50 django/conf/locale/de/formats.py | 5 | 28| 305 | 32638 | 84016 | 
| 79 | 51 django/conf/locale/ig/formats.py | 5 | 33| 388 | 33026 | 84449 | 
| 80 | 52 django/conf/locale/it/formats.py | 5 | 41| 764 | 33790 | 85258 | 
| 81 | 53 django/contrib/humanize/templatetags/humanize.py | 177 | 211| 615 | 34405 | 88142 | 
| 82 | 54 django/conf/locale/nn/formats.py | 5 | 37| 593 | 34998 | 88780 | 
| 83 | 55 django/conf/locale/es_PR/formats.py | 3 | 28| 252 | 35250 | 89048 | 
| 84 | 56 django/conf/locale/ar_DZ/formats.py | 5 | 30| 252 | 35502 | 89345 | 
| 85 | 57 django/conf/locale/es_MX/formats.py | 3 | 26| 289 | 35791 | 89650 | 
| 86 | 58 django/db/backends/utils.py | 152 | 176| 234 | 36025 | 91532 | 
| 87 | 58 django/contrib/humanize/templatetags/humanize.py | 140 | 174| 280 | 36305 | 91532 | 
| 88 | 58 django/contrib/humanize/templatetags/humanize.py | 222 | 260| 370 | 36675 | 91532 | 
| 89 | 59 django/conf/locale/sk/formats.py | 5 | 29| 330 | 37005 | 91907 | 
| 90 | 60 django/conf/locale/sv/formats.py | 5 | 36| 481 | 37486 | 92433 | 
| 91 | 61 django/conf/locale/da/formats.py | 5 | 27| 250 | 37736 | 92728 | 
| 92 | 61 django/db/backends/sqlite3/operations.py | 246 | 275| 198 | 37934 | 92728 | 
| 93 | 62 django/conf/locale/de_CH/formats.py | 5 | 34| 403 | 38337 | 93176 | 
| 94 | 63 django/conf/locale/es_CO/formats.py | 3 | 27| 262 | 38599 | 93454 | 
| 95 | 64 django/conf/locale/ta/formats.py | 5 | 22| 125 | 38724 | 93623 | 
| 96 | 65 django/conf/locale/nb/formats.py | 5 | 37| 593 | 39317 | 94261 | 
| 97 | 66 django/conf/global_settings.py | 350 | 400| 785 | 40102 | 99961 | 
| 98 | 66 django/db/backends/sqlite3/base.py | 541 | 584| 324 | 40426 | 99961 | 
| 99 | 67 django/conf/locale/bg/formats.py | 5 | 22| 131 | 40557 | 100136 | 
| 100 | 68 django/conf/locale/ko/formats.py | 32 | 50| 385 | 40942 | 101026 | 
| 101 | 69 django/conf/locale/ky/formats.py | 5 | 33| 414 | 41356 | 101485 | 
| 102 | 70 django/conf/locale/gd/formats.py | 5 | 22| 144 | 41500 | 101673 | 
| 103 | **70 django/utils/timezone.py** | 138 | 155| 148 | 41648 | 101673 | 
| 104 | 71 django/conf/locale/tr/formats.py | 5 | 29| 319 | 41967 | 102037 | 
| 105 | 72 django/conf/locale/is/formats.py | 5 | 22| 130 | 42097 | 102212 | 
| 106 | 73 django/conf/locale/es_NI/formats.py | 3 | 27| 270 | 42367 | 102498 | 
| 107 | 73 django/db/backends/utils.py | 133 | 149| 141 | 42508 | 102498 | 
| 108 | 74 django/conf/locale/es_AR/formats.py | 5 | 31| 275 | 42783 | 102818 | 
| 109 | 75 django/conf/locale/ro/formats.py | 5 | 36| 262 | 43045 | 103125 | 
| 110 | 75 django/db/models/fields/__init__.py | 2252 | 2278| 213 | 43258 | 103125 | 
| 111 | 75 django/db/backends/mysql/operations.py | 324 | 339| 264 | 43522 | 103125 | 
| 112 | 75 django/db/models/fields/__init__.py | 1338 | 1384| 329 | 43851 | 103125 | 
| 113 | **75 django/utils/timezone.py** | 180 | 226| 319 | 44170 | 103125 | 
| 114 | 76 django/conf/locale/ca/formats.py | 5 | 31| 287 | 44457 | 103457 | 
| 115 | 77 django/conf/locale/es/formats.py | 5 | 31| 285 | 44742 | 103787 | 
| 116 | 78 django/conf/locale/bs/formats.py | 5 | 22| 139 | 44881 | 103970 | 
| 117 | **78 django/utils/timezone.py** | 109 | 135| 188 | 45069 | 103970 | 
| 118 | **78 django/utils/timezone.py** | 245 | 270| 212 | 45281 | 103970 | 
| 119 | 78 django/db/backends/postgresql/operations.py | 29 | 39| 170 | 45451 | 103970 | 
| 120 | 79 django/contrib/gis/gdal/field.py | 163 | 175| 171 | 45622 | 105643 | 
| 121 | 80 django/conf/locale/te/formats.py | 5 | 22| 123 | 45745 | 105810 | 
| 122 | 81 django/conf/locale/sq/formats.py | 5 | 22| 128 | 45873 | 105982 | 
| 123 | 82 django/conf/locale/ga/formats.py | 5 | 22| 124 | 45997 | 106150 | 
| 124 | 82 django/db/models/fields/__init__.py | 2170 | 2186| 185 | 46182 | 106150 | 
| 125 | 82 django/db/backends/mysql/operations.py | 1 | 35| 282 | 46464 | 106150 | 
| 126 | 83 django/conf/locale/gl/formats.py | 5 | 22| 170 | 46634 | 106364 | 
| 127 | 83 django/conf/locale/nl/formats.py | 5 | 31| 467 | 47101 | 106364 | 
| 128 | 84 django/conf/locale/kn/formats.py | 5 | 22| 123 | 47224 | 106531 | 
| 129 | 85 django/conf/locale/he/formats.py | 5 | 22| 142 | 47366 | 106717 | 
| 130 | 85 django/db/backends/oracle/operations.py | 520 | 544| 242 | 47608 | 106717 | 
| 131 | 86 django/conf/locale/ar/formats.py | 5 | 22| 135 | 47743 | 106896 | 
| 132 | 87 django/conf/locale/km/formats.py | 5 | 22| 164 | 47907 | 107104 | 
| 133 | 88 django/utils/duration.py | 1 | 45| 304 | 48211 | 107409 | 
| 134 | 88 django/conf/locale/ko/formats.py | 5 | 31| 460 | 48671 | 107409 | 
| 135 | 88 django/db/models/fields/__init__.py | 2210 | 2250| 289 | 48960 | 107409 | 
| 136 | 88 django/db/backends/mysql/operations.py | 37 | 56| 296 | 49256 | 107409 | 
| 137 | 88 django/contrib/humanize/templatetags/humanize.py | 212 | 220| 205 | 49461 | 107409 | 
| 138 | 88 django/utils/dateformat.py | 142 | 153| 151 | 49612 | 107409 | 
| 139 | 89 django/utils/datetime_safe.py | 76 | 109| 309 | 49921 | 108209 | 
| 140 | 90 django/utils/dates.py | 1 | 50| 679 | 50600 | 108888 | 
| 141 | 91 django/conf/locale/hi/formats.py | 5 | 22| 125 | 50725 | 109057 | 
| 142 | 91 django/templatetags/tz.py | 125 | 145| 176 | 50901 | 109057 | 
| 143 | 92 django/conf/locale/mn/formats.py | 5 | 22| 120 | 51021 | 109221 | 
| 144 | 92 django/db/backends/base/operations.py | 155 | 228| 615 | 51636 | 109221 | 
| 145 | **92 django/utils/timezone.py** | 229 | 242| 141 | 51777 | 109221 | 
| 146 | 93 django/db/backends/postgresql/base.py | 221 | 253| 260 | 52037 | 112130 | 
| 147 | 94 django/conf/locale/ja/formats.py | 5 | 22| 149 | 52186 | 112323 | 
| 148 | 95 django/conf/locale/fa/formats.py | 5 | 22| 149 | 52335 | 112516 | 
| 149 | 96 django/conf/locale/th/formats.py | 5 | 34| 355 | 52690 | 112916 | 
| 150 | 96 django/templatetags/tz.py | 173 | 191| 149 | 52839 | 112916 | 
| 151 | **96 django/utils/timezone.py** | 158 | 177| 140 | 52979 | 112916 | 
| 152 | 97 django/conf/locale/eu/formats.py | 5 | 22| 171 | 53150 | 113132 | 
| 153 | 98 django/conf/locale/vi/formats.py | 5 | 22| 179 | 53329 | 113355 | 
| 154 | 98 django/db/backends/base/operations.py | 486 | 527| 284 | 53613 | 113355 | 
| 155 | 98 django/utils/datetime_safe.py | 1 | 73| 489 | 54102 | 113355 | 
| 156 | 98 django/db/backends/sqlite3/base.py | 587 | 600| 135 | 54237 | 113355 | 
| 157 | 98 django/db/backends/sqlite3/operations.py | 294 | 312| 157 | 54394 | 113355 | 
| 158 | 98 django/utils/dateparse.py | 106 | 131| 267 | 54661 | 113355 | 
| 159 | 98 django/db/backends/oracle/operations.py | 546 | 565| 209 | 54870 | 113355 | 
| 160 | 98 django/utils/dateparse.py | 69 | 103| 311 | 55181 | 113355 | 
| 161 | 99 django/views/generic/dates.py | 634 | 730| 806 | 55987 | 118796 | 
| 162 | 99 django/db/models/fields/__init__.py | 1416 | 1430| 121 | 56108 | 118796 | 
| 163 | 99 django/templatetags/tz.py | 1 | 34| 148 | 56256 | 118796 | 
| 164 | 100 django/utils/http.py | 1 | 37| 464 | 56720 | 122038 | 
| 165 | 100 django/contrib/gis/gdal/field.py | 60 | 71| 158 | 56878 | 122038 | 
| 166 | 100 django/db/models/fields/__init__.py | 1235 | 1263| 189 | 57067 | 122038 | 
| 167 | 100 django/db/models/fields/__init__.py | 1306 | 1336| 261 | 57328 | 122038 | 
| 168 | 100 django/db/backends/oracle/operations.py | 75 | 90| 281 | 57609 | 122038 | 
| 169 | 100 django/db/backends/oracle/operations.py | 1 | 18| 143 | 57752 | 122038 | 
| 170 | 100 django/db/backends/postgresql/operations.py | 189 | 276| 696 | 58448 | 122038 | 
| 171 | 101 django/forms/fields.py | 1140 | 1173| 293 | 58741 | 131459 | 
| 172 | 101 django/db/models/fields/__init__.py | 2188 | 2208| 181 | 58922 | 131459 | 
| 173 | 102 django/contrib/postgres/apps.py | 1 | 20| 158 | 59080 | 132055 | 
| 174 | 103 django/db/backends/oracle/functions.py | 1 | 23| 188 | 59268 | 132243 | 
| 175 | 104 django/core/cache/backends/db.py | 229 | 248| 242 | 59510 | 134350 | 
| 176 | 104 django/db/backends/base/operations.py | 1 | 100| 829 | 60339 | 134350 | 
| 177 | 104 django/utils/http.py | 81 | 92| 119 | 60458 | 134350 | 
| 178 | 104 django/db/models/fields/__init__.py | 1265 | 1283| 180 | 60638 | 134350 | 
| 179 | 104 django/utils/dateformat.py | 1 | 28| 225 | 60863 | 134350 | 
| 180 | 105 django/template/defaultfilters.py | 707 | 784| 443 | 61306 | 140580 | 
| 181 | 105 django/utils/dateformat.py | 31 | 44| 121 | 61427 | 140580 | 
| 182 | 105 django/forms/fields.py | 421 | 446| 168 | 61595 | 140580 | 
| 183 | 106 django/db/models/expressions.py | 549 | 578| 263 | 61858 | 151691 | 
| 184 | 106 django/db/backends/sqlite3/base.py | 603 | 625| 143 | 62001 | 151691 | 
| 185 | 107 django/db/backends/mysql/base.py | 363 | 385| 208 | 62209 | 155150 | 
| 186 | 107 django/db/models/fields/__init__.py | 1285 | 1303| 149 | 62358 | 155150 | 
| 187 | 108 django/db/backends/base/features.py | 1 | 112| 895 | 63253 | 158157 | 
| 188 | 108 django/core/cache/backends/db.py | 40 | 95| 431 | 63684 | 158157 | 
| 189 | 109 django/utils/formats.py | 140 | 161| 205 | 63889 | 160570 | 
| 190 | 109 django/core/cache/backends/db.py | 112 | 196| 797 | 64686 | 160570 | 
| 191 | 109 django/db/backends/base/operations.py | 577 | 668| 743 | 65429 | 160570 | 
| 192 | 109 django/utils/dateparse.py | 134 | 159| 255 | 65684 | 160570 | 


### Hint

```
Thanks for the report. Regression in 10d126198434810529e0220b0c6896ed64ca0e88. Reproduced at 4fe3774c729f3fd5105b3001fe69a70bdca95ac3.
This problem is also affecting MySQL, the timezone "Etc/GMT-10" is returning "-10" instead of "-10:00". #33037
```

## Patch

```diff
diff --git a/django/utils/timezone.py b/django/utils/timezone.py
--- a/django/utils/timezone.py
+++ b/django/utils/timezone.py
@@ -72,8 +72,11 @@ def get_current_timezone_name():
 
 
 def _get_timezone_name(timezone):
-    """Return the name of ``timezone``."""
-    return str(timezone)
+    """
+    Return the offset for fixed offset timezones, or the name of timezone if
+    not set.
+    """
+    return timezone.tzname(None) or str(timezone)
 
 # Timezone selection functions.
 

```

## Test Patch

```diff
diff --git a/tests/utils_tests/test_timezone.py b/tests/utils_tests/test_timezone.py
--- a/tests/utils_tests/test_timezone.py
+++ b/tests/utils_tests/test_timezone.py
@@ -260,6 +260,31 @@ def test_make_aware_zoneinfo_non_existent(self):
         self.assertEqual(std.utcoffset(), datetime.timedelta(hours=1))
         self.assertEqual(dst.utcoffset(), datetime.timedelta(hours=2))
 
+    def test_get_timezone_name(self):
+        """
+        The _get_timezone_name() helper must return the offset for fixed offset
+        timezones, for usage with Trunc DB functions.
+
+        The datetime.timezone examples show the current behavior.
+        """
+        tests = [
+            # datetime.timezone, fixed offset with and without `name`.
+            (datetime.timezone(datetime.timedelta(hours=10)), 'UTC+10:00'),
+            (datetime.timezone(datetime.timedelta(hours=10), name='Etc/GMT-10'), 'Etc/GMT-10'),
+            # pytz, named and fixed offset.
+            (pytz.timezone('Europe/Madrid'), 'Europe/Madrid'),
+            (pytz.timezone('Etc/GMT-10'), '+10'),
+        ]
+        if HAS_ZONEINFO:
+            tests += [
+                # zoneinfo, named and fixed offset.
+                (zoneinfo.ZoneInfo('Europe/Madrid'), 'Europe/Madrid'),
+                (zoneinfo.ZoneInfo('Etc/GMT-10'), '+10'),
+            ]
+        for tz, expected in tests:
+            with self.subTest(tz=tz, expected=expected):
+                self.assertEqual(timezone._get_timezone_name(tz), expected)
+
     def test_get_default_timezone(self):
         self.assertEqual(timezone.get_default_timezone_name(), 'America/Chicago')
 

```


## Code snippets

### 1 - django/db/models/functions/datetime.py:

Start line: 187, End line: 211

```python
class TruncBase(TimezoneMixin, Transform):
    kind = None
    tzinfo = None

    def __init__(self, expression, output_field=None, tzinfo=None, is_dst=None, **extra):
        self.tzinfo = tzinfo
        self.is_dst = is_dst
        super().__init__(expression, output_field=output_field, **extra)

    def as_sql(self, compiler, connection):
        inner_sql, inner_params = compiler.compile(self.lhs)
        tzname = None
        if isinstance(self.lhs.output_field, DateTimeField):
            tzname = self.get_tzname()
        elif self.tzinfo is not None:
            raise ValueError('tzinfo can only be used with DateTimeField.')
        if isinstance(self.output_field, DateTimeField):
            sql = connection.ops.datetime_trunc_sql(self.kind, inner_sql, tzname)
        elif isinstance(self.output_field, DateField):
            sql = connection.ops.date_trunc_sql(self.kind, inner_sql, tzname)
        elif isinstance(self.output_field, TimeField):
            sql = connection.ops.time_trunc_sql(self.kind, inner_sql, tzname)
        else:
            raise ValueError('Trunc only valid on DateField, TimeField, or DateTimeField.')
        return sql, inner_params
```
### 2 - django/db/models/functions/datetime.py:

Start line: 243, End line: 262

```python
class TruncBase(TimezoneMixin, Transform):

    def convert_value(self, value, expression, connection):
        if isinstance(self.output_field, DateTimeField):
            if not settings.USE_TZ:
                pass
            elif value is not None:
                value = value.replace(tzinfo=None)
                value = timezone.make_aware(value, self.tzinfo, is_dst=self.is_dst)
            elif not connection.features.has_zoneinfo_database:
                raise ValueError(
                    'Database returned an invalid datetime value. Are time '
                    'zone definitions for your database installed?'
                )
        elif isinstance(value, datetime):
            if value is None:
                pass
            elif isinstance(self.output_field, DateField):
                value = value.date()
            elif isinstance(self.output_field, TimeField):
                value = value.time()
        return value
```
### 3 - django/db/models/functions/datetime.py:

Start line: 213, End line: 241

```python
class TruncBase(TimezoneMixin, Transform):

    def resolve_expression(self, query=None, allow_joins=True, reuse=None, summarize=False, for_save=False):
        copy = super().resolve_expression(query, allow_joins, reuse, summarize, for_save)
        field = copy.lhs.output_field
        # DateTimeField is a subclass of DateField so this works for both.
        if not isinstance(field, (DateField, TimeField)):
            raise TypeError(
                "%r isn't a DateField, TimeField, or DateTimeField." % field.name
            )
        # If self.output_field was None, then accessing the field will trigger
        # the resolver to assign it to self.lhs.output_field.
        if not isinstance(copy.output_field, (DateField, DateTimeField, TimeField)):
            raise ValueError('output_field must be either DateField, TimeField, or DateTimeField')
        # Passing dates or times to functions expecting datetimes is most
        # likely a mistake.
        class_output_field = self.__class__.output_field if isinstance(self.__class__.output_field, Field) else None
        output_field = class_output_field or copy.output_field
        has_explicit_output_field = class_output_field or field.__class__ is not copy.output_field.__class__
        if type(field) == DateField and (
                isinstance(output_field, DateTimeField) or copy.kind in ('hour', 'minute', 'second', 'time')):
            raise ValueError("Cannot truncate DateField '%s' to %s." % (
                field.name, output_field.__class__.__name__ if has_explicit_output_field else 'DateTimeField'
            ))
        elif isinstance(field, TimeField) and (
                isinstance(output_field, DateTimeField) or
                copy.kind in ('year', 'quarter', 'month', 'week', 'day', 'date')):
            raise ValueError("Cannot truncate TimeField '%s' to %s." % (
                field.name, output_field.__class__.__name__ if has_explicit_output_field else 'DateTimeField'
            ))
        return copy
```
### 4 - django/db/models/functions/datetime.py:

Start line: 265, End line: 336

```python
class Trunc(TruncBase):

    def __init__(self, expression, kind, output_field=None, tzinfo=None, is_dst=None, **extra):
        self.kind = kind
        super().__init__(
            expression, output_field=output_field, tzinfo=tzinfo,
            is_dst=is_dst, **extra
        )


class TruncYear(TruncBase):
    kind = 'year'


class TruncQuarter(TruncBase):
    kind = 'quarter'


class TruncMonth(TruncBase):
    kind = 'month'


class TruncWeek(TruncBase):
    """Truncate to midnight on the Monday of the week."""
    kind = 'week'


class TruncDay(TruncBase):
    kind = 'day'


class TruncDate(TruncBase):
    kind = 'date'
    lookup_name = 'date'
    output_field = DateField()

    def as_sql(self, compiler, connection):
        # Cast to date rather than truncate to date.
        lhs, lhs_params = compiler.compile(self.lhs)
        tzname = self.get_tzname()
        sql = connection.ops.datetime_cast_date_sql(lhs, tzname)
        return sql, lhs_params


class TruncTime(TruncBase):
    kind = 'time'
    lookup_name = 'time'
    output_field = TimeField()

    def as_sql(self, compiler, connection):
        # Cast to time rather than truncate to time.
        lhs, lhs_params = compiler.compile(self.lhs)
        tzname = self.get_tzname()
        sql = connection.ops.datetime_cast_time_sql(lhs, tzname)
        return sql, lhs_params


class TruncHour(TruncBase):
    kind = 'hour'


class TruncMinute(TruncBase):
    kind = 'minute'


class TruncSecond(TruncBase):
    kind = 'second'


DateTimeField.register_lookup(TruncDate)
DateTimeField.register_lookup(TruncTime)
```
### 5 - django/db/models/functions/datetime.py:

Start line: 31, End line: 63

```python
class Extract(TimezoneMixin, Transform):
    lookup_name = None
    output_field = IntegerField()

    def __init__(self, expression, lookup_name=None, tzinfo=None, **extra):
        if self.lookup_name is None:
            self.lookup_name = lookup_name
        if self.lookup_name is None:
            raise ValueError('lookup_name must be provided')
        self.tzinfo = tzinfo
        super().__init__(expression, **extra)

    def as_sql(self, compiler, connection):
        sql, params = compiler.compile(self.lhs)
        lhs_output_field = self.lhs.output_field
        if isinstance(lhs_output_field, DateTimeField):
            tzname = self.get_tzname()
            sql = connection.ops.datetime_extract_sql(self.lookup_name, sql, tzname)
        elif self.tzinfo is not None:
            raise ValueError('tzinfo can only be used with DateTimeField.')
        elif isinstance(lhs_output_field, DateField):
            sql = connection.ops.date_extract_sql(self.lookup_name, sql)
        elif isinstance(lhs_output_field, TimeField):
            sql = connection.ops.time_extract_sql(self.lookup_name, sql)
        elif isinstance(lhs_output_field, DurationField):
            if not connection.features.has_native_duration_field:
                raise ValueError('Extract requires native DurationField database support.')
            sql = connection.ops.time_extract_sql(self.lookup_name, sql)
        else:
            # resolve_expression has already validated the output_field so this
            # assert should never be hit.
            assert False, "Tried to Extract from an invalid type."
        return sql, params
```
### 6 - django/db/backends/sqlite3/base.py:

Start line: 466, End line: 482

```python
def _sqlite_time_trunc(lookup_type, dt, tzname, conn_tzname):
    if dt is None:
        return None
    dt_parsed = _sqlite_datetime_parse(dt, tzname, conn_tzname)
    if dt_parsed is None:
        try:
            dt = backend_utils.typecast_time(dt)
        except (ValueError, TypeError):
            return None
    else:
        dt = dt_parsed
    if lookup_type == 'hour':
        return "%02i:00:00" % dt.hour
    elif lookup_type == 'minute':
        return "%02i:%02i:00" % (dt.hour, dt.minute)
    elif lookup_type == 'second':
        return "%02i:%02i:%02i" % (dt.hour, dt.minute, dt.second)
```
### 7 - django/utils/dateformat.py:

Start line: 174, End line: 191

```python
class TimeFormat(Formatter):

    def Z(self):
        """
        Time zone offset in seconds (i.e. '-43200' to '43200'). The offset for
        timezones west of UTC is always negative, and for those east of UTC is
        always positive.

        If timezone information is not available, return an empty string.
        """
        if self._no_timezone_or_datetime_is_ambiguous_or_imaginary:
            return ""

        offset = self.timezone.utcoffset(self.data)

        # `offset` is a datetime.timedelta. For negative values (to the west of
        # UTC) only days can be negative (days=-1) and seconds are always
        # positive. e.g. UTC-1 -> timedelta(days=-1, seconds=82800, microseconds=0)
        # Positive offsets have days=0
        return offset.days * 86400 + offset.seconds
```
### 8 - django/db/backends/mysql/operations.py:

Start line: 132, End line: 143

```python
class DatabaseOperations(BaseDatabaseOperations):

    def time_trunc_sql(self, lookup_type, field_name, tzname=None):
        field_name = self._convert_field_to_tz(field_name, tzname)
        fields = {
            'hour': '%%H:00:00',
            'minute': '%%H:%%i:00',
            'second': '%%H:%%i:%%s',
        }  # Use double percents to escape.
        if lookup_type in fields:
            format_str = fields[lookup_type]
            return "CAST(DATE_FORMAT(%s, '%s') AS TIME)" % (field_name, format_str)
        else:
            return "TIME(%s)" % (field_name)
```
### 9 - django/db/models/functions/datetime.py:

Start line: 65, End line: 88

```python
class Extract(TimezoneMixin, Transform):

    def resolve_expression(self, query=None, allow_joins=True, reuse=None, summarize=False, for_save=False):
        copy = super().resolve_expression(query, allow_joins, reuse, summarize, for_save)
        field = getattr(copy.lhs, 'output_field', None)
        if field is None:
            return copy
        if not isinstance(field, (DateField, DateTimeField, TimeField, DurationField)):
            raise ValueError(
                'Extract input expression must be DateField, DateTimeField, '
                'TimeField, or DurationField.'
            )
        # Passing dates to functions expecting datetimes is most likely a mistake.
        if type(field) == DateField and copy.lookup_name in ('hour', 'minute', 'second'):
            raise ValueError(
                "Cannot extract time component '%s' from DateField '%s'." % (copy.lookup_name, field.name)
            )
        if (
            isinstance(field, DurationField) and
            copy.lookup_name in ('year', 'iso_year', 'month', 'week', 'week_day', 'iso_week_day', 'quarter')
        ):
            raise ValueError(
                "Cannot extract component '%s' from DurationField '%s'."
                % (copy.lookup_name, field.name)
            )
        return copy
```
### 10 - django/db/models/functions/datetime.py:

Start line: 1, End line: 28

```python
from datetime import datetime

from django.conf import settings
from django.db.models.expressions import Func
from django.db.models.fields import (
    DateField, DateTimeField, DurationField, Field, IntegerField, TimeField,
)
from django.db.models.lookups import (
    Transform, YearExact, YearGt, YearGte, YearLt, YearLte,
)
from django.utils import timezone


class TimezoneMixin:
    tzinfo = None

    def get_tzname(self):
        # Timezone conversions must happen to the input datetime *before*
        # applying a function. 2015-12-31 23:00:00 -02:00 is stored in the
        # database as 2016-01-01 01:00:00 +00:00. Any results should be
        # based on the input datetime not the stored datetime.
        tzname = None
        if settings.USE_TZ:
            if self.tzinfo is None:
                tzname = timezone.get_current_timezone_name()
            else:
                tzname = timezone._get_timezone_name(self.tzinfo)
        return tzname
```
### 20 - django/utils/timezone.py:

Start line: 1, End line: 106

```python
"""
Timezone-related classes and functions.
"""

import functools
from contextlib import ContextDecorator
from datetime import datetime, timedelta, timezone, tzinfo

import pytz
from asgiref.local import Local

from django.conf import settings

__all__ = [
    'utc', 'get_fixed_timezone',
    'get_default_timezone', 'get_default_timezone_name',
    'get_current_timezone', 'get_current_timezone_name',
    'activate', 'deactivate', 'override',
    'localtime', 'now',
    'is_aware', 'is_naive', 'make_aware', 'make_naive',
]


# UTC time zone as a tzinfo instance.
utc = pytz.utc

_PYTZ_BASE_CLASSES = (pytz.tzinfo.BaseTzInfo, pytz._FixedOffset)
# In releases prior to 2018.4, pytz.UTC was not a subclass of BaseTzInfo
if not isinstance(pytz.UTC, pytz._FixedOffset):
    _PYTZ_BASE_CLASSES = _PYTZ_BASE_CLASSES + (type(pytz.UTC),)


def get_fixed_timezone(offset):
    """Return a tzinfo instance with a fixed offset from UTC."""
    if isinstance(offset, timedelta):
        offset = offset.total_seconds() // 60
    sign = '-' if offset < 0 else '+'
    hhmm = '%02d%02d' % divmod(abs(offset), 60)
    name = sign + hhmm
    return timezone(timedelta(minutes=offset), name)


# In order to avoid accessing settings at compile time,
# wrap the logic in a function and cache the result.
@functools.lru_cache()
def get_default_timezone():
    """
    Return the default time zone as a tzinfo instance.

    This is the time zone defined by settings.TIME_ZONE.
    """
    return pytz.timezone(settings.TIME_ZONE)


# This function exists for consistency with get_current_timezone_name
def get_default_timezone_name():
    """Return the name of the default time zone."""
    return _get_timezone_name(get_default_timezone())


_active = Local()


def get_current_timezone():
    """Return the currently active time zone as a tzinfo instance."""
    return getattr(_active, "value", get_default_timezone())


def get_current_timezone_name():
    """Return the name of the currently active time zone."""
    return _get_timezone_name(get_current_timezone())


def _get_timezone_name(timezone):
    """Return the name of ``timezone``."""
    return str(timezone)

# Timezone selection functions.

# These functions don't change os.environ['TZ'] and call time.tzset()
# because it isn't thread safe.


def activate(timezone):
    """
    Set the time zone for the current thread.

    The ``timezone`` argument must be an instance of a tzinfo subclass or a
    time zone name.
    """
    if isinstance(timezone, tzinfo):
        _active.value = timezone
    elif isinstance(timezone, str):
        _active.value = pytz.timezone(timezone)
    else:
        raise ValueError("Invalid timezone: %r" % timezone)


def deactivate():
    """
    Unset the time zone for the current thread.

    Django will then use the time zone defined by settings.TIME_ZONE.
    """
    if hasattr(_active, "value"):
        del _active.value
```
### 103 - django/utils/timezone.py:

Start line: 138, End line: 155

```python
# Templates

def template_localtime(value, use_tz=None):
    """
    Check if value is a datetime and converts it to local time if necessary.

    If use_tz is provided and is not None, that will force the value to
    be converted (or not), overriding the value of settings.USE_TZ.

    This function is designed for use by the template engine.
    """
    should_convert = (
        isinstance(value, datetime) and
        (settings.USE_TZ if use_tz is None else use_tz) and
        not is_naive(value) and
        getattr(value, 'convert_to_local_time', True)
    )
    return localtime(value) if should_convert else value
```
### 113 - django/utils/timezone.py:

Start line: 180, End line: 226

```python
def localdate(value=None, timezone=None):
    """
    Convert an aware datetime to local time and return the value's date.

    Only aware datetimes are allowed. When value is omitted, it defaults to
    now().

    Local time is defined by the current time zone, unless another time zone is
    specified.
    """
    return localtime(value, timezone).date()


def now():
    """
    Return an aware or naive datetime.datetime, depending on settings.USE_TZ.
    """
    return datetime.now(tz=utc if settings.USE_TZ else None)


# By design, these four functions don't perform any checks on their arguments.
# The caller should ensure that they don't receive an invalid value like None.

def is_aware(value):
    """
    Determine if a given datetime.datetime is aware.

    The concept is defined in Python's docs:
    https://docs.python.org/library/datetime.html#datetime.tzinfo

    Assuming value.tzinfo is either None or a proper datetime.tzinfo,
    value.utcoffset() implements the appropriate logic.
    """
    return value.utcoffset() is not None


def is_naive(value):
    """
    Determine if a given datetime.datetime is naive.

    The concept is defined in Python's docs:
    https://docs.python.org/library/datetime.html#datetime.tzinfo

    Assuming value.tzinfo is either None or a proper datetime.tzinfo,
    value.utcoffset() implements the appropriate logic.
    """
    return value.utcoffset() is None
```
### 117 - django/utils/timezone.py:

Start line: 109, End line: 135

```python
class override(ContextDecorator):
    """
    Temporarily set the time zone for the current thread.

    This is a context manager that uses django.utils.timezone.activate()
    to set the timezone on entry and restores the previously active timezone
    on exit.

    The ``timezone`` argument must be an instance of a ``tzinfo`` subclass, a
    time zone name, or ``None``. If it is ``None``, Django enables the default
    time zone.
    """
    def __init__(self, timezone):
        self.timezone = timezone

    def __enter__(self):
        self.old_timezone = getattr(_active, 'value', None)
        if self.timezone is None:
            deactivate()
        else:
            activate(self.timezone)

    def __exit__(self, exc_type, exc_value, traceback):
        if self.old_timezone is None:
            deactivate()
        else:
            _active.value = self.old_timezone
```
### 118 - django/utils/timezone.py:

Start line: 245, End line: 270

```python
def make_naive(value, timezone=None):
    """Make an aware datetime.datetime naive in a given time zone."""
    if timezone is None:
        timezone = get_current_timezone()
    # Emulate the behavior of astimezone() on Python < 3.6.
    if is_naive(value):
        raise ValueError("make_naive() cannot be applied to a naive datetime")
    return value.astimezone(timezone).replace(tzinfo=None)


def _is_pytz_zone(tz):
    """Checks if a zone is a pytz zone."""
    return isinstance(tz, _PYTZ_BASE_CLASSES)


def _datetime_ambiguous_or_imaginary(dt, tz):
    if _is_pytz_zone(tz):
        try:
            tz.utcoffset(dt)
        except (pytz.AmbiguousTimeError, pytz.NonExistentTimeError):
            return True
        else:
            return False

    return tz.utcoffset(dt.replace(fold=not dt.fold)) != tz.utcoffset(dt)
```
### 145 - django/utils/timezone.py:

Start line: 229, End line: 242

```python
def make_aware(value, timezone=None, is_dst=None):
    """Make a naive datetime.datetime in a given time zone aware."""
    if timezone is None:
        timezone = get_current_timezone()
    if _is_pytz_zone(timezone):
        # This method is available for pytz time zones.
        return timezone.localize(value, is_dst=is_dst)
    else:
        # Check that we won't overwrite the timezone of an aware datetime.
        if is_aware(value):
            raise ValueError(
                "make_aware expects a naive datetime, got %s" % value)
        # This may be wrong around DST changes!
        return value.replace(tzinfo=timezone)
```
### 151 - django/utils/timezone.py:

Start line: 158, End line: 177

```python
# Utilities

def localtime(value=None, timezone=None):
    """
    Convert an aware datetime.datetime to local time.

    Only aware datetimes are allowed. When value is omitted, it defaults to
    now().

    Local time is defined by the current time zone, unless another time zone
    is specified.
    """
    if value is None:
        value = now()
    if timezone is None:
        timezone = get_current_timezone()
    # Emulate the behavior of astimezone() on Python < 3.6.
    if is_naive(value):
        raise ValueError("localtime() cannot be applied to a naive datetime")
    return value.astimezone(timezone)
```
