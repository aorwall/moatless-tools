# django__django-12125

| **django/django** | `89d41cba392b759732ba9f1db4ff29ed47da6a56` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 39228 |
| **Any found context length** | 39228 |
| **Avg pos** | 115.0 |
| **Min pos** | 115 |
| **Max pos** | 115 |
| **Top file pos** | 4 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/db/migrations/serializer.py b/django/db/migrations/serializer.py
--- a/django/db/migrations/serializer.py
+++ b/django/db/migrations/serializer.py
@@ -269,7 +269,7 @@ def serialize(self):
             if module == builtins.__name__:
                 return self.value.__name__, set()
             else:
-                return "%s.%s" % (module, self.value.__name__), {"import %s" % module}
+                return "%s.%s" % (module, self.value.__qualname__), {"import %s" % module}
 
 
 class UUIDSerializer(BaseSerializer):

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/db/migrations/serializer.py | 272 | 272 | 115 | 4 | 39228


## Problem Statement

```
makemigrations produces incorrect path for inner classes
Description
	
When you define a subclass from django.db.models.Field as an inner class of some other class, and use this field inside a django.db.models.Model class, then when you run manage.py makemigrations, a migrations file is created which refers to the inner class as if it were a top-level class of the module it is in.
To reproduce, create the following as your model:
class Outer(object):
	class Inner(models.CharField):
		pass
class A(models.Model):
	field = Outer.Inner(max_length=20)
After running manage.py makemigrations, the generated migrations file contains the following:
migrations.CreateModel(
	name='A',
	fields=[
		('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
		('field', test1.models.Inner(max_length=20)),
	],
),
Note the test1.models.Inner, which should have been test1.models.Outer.Inner.
The real life case involved an EnumField from django-enumfields, defined as an inner class of a Django Model class, similar to this:
import enum
from enumfields import Enum, EnumField
class Thing(models.Model):
	@enum.unique
	class State(Enum):
		on = 'on'
		off = 'off'
	state = EnumField(enum=State)
This results in the following migrations code:
migrations.CreateModel(
	name='Thing',
	fields=[
		('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
		('state', enumfields.fields.EnumField(enum=test1.models.State, max_length=10)),
	],
),
This refers to test1.models.State, instead of to test1.models.Thing.State.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 django/contrib/contenttypes/migrations/0001_initial.py | 1 | 35| 207 | 207 | 207 | 
| 2 | 2 django/db/migrations/state.py | 580 | 599| 188 | 395 | 5425 | 
| 3 | 3 django/db/models/fields/related.py | 1202 | 1313| 939 | 1334 | 18937 | 
| 4 | 3 django/db/models/fields/related.py | 1044 | 1088| 407 | 1741 | 18937 | 
| 5 | **4 django/db/migrations/serializer.py** | 196 | 217| 183 | 1924 | 21485 | 
| 6 | 5 django/db/migrations/autodetector.py | 904 | 985| 876 | 2800 | 33220 | 
| 7 | 6 django/contrib/admin/migrations/0001_initial.py | 1 | 48| 320 | 3120 | 33540 | 
| 8 | 7 django/contrib/sites/migrations/0001_initial.py | 1 | 32| 191 | 3311 | 33731 | 
| 9 | 8 django/contrib/auth/migrations/0001_initial.py | 1 | 105| 849 | 4160 | 34580 | 
| 10 | 9 django/db/migrations/operations/__init__.py | 1 | 18| 195 | 4355 | 34775 | 
| 11 | 10 django/contrib/flatpages/migrations/0001_initial.py | 1 | 40| 307 | 4662 | 35082 | 
| 12 | 10 django/db/models/fields/related.py | 1556 | 1591| 475 | 5137 | 35082 | 
| 13 | 10 django/db/models/fields/related.py | 1593 | 1609| 286 | 5423 | 35082 | 
| 14 | 11 django/contrib/sessions/migrations/0001_initial.py | 1 | 31| 162 | 5585 | 35244 | 
| 15 | 11 django/db/migrations/state.py | 496 | 528| 250 | 5835 | 35244 | 
| 16 | 12 django/db/migrations/questioner.py | 56 | 81| 220 | 6055 | 37318 | 
| 17 | 13 django/contrib/redirects/migrations/0001_initial.py | 1 | 41| 274 | 6329 | 37592 | 
| 18 | 13 django/db/migrations/state.py | 601 | 612| 136 | 6465 | 37592 | 
| 19 | 13 django/db/models/fields/related.py | 255 | 282| 269 | 6734 | 37592 | 
| 20 | 13 django/db/migrations/autodetector.py | 1182 | 1207| 245 | 6979 | 37592 | 
| 21 | 13 django/db/migrations/autodetector.py | 525 | 671| 1109 | 8088 | 37592 | 
| 22 | 14 django/db/models/base.py | 1611 | 1659| 348 | 8436 | 52903 | 
| 23 | 15 django/core/management/commands/makemigrations.py | 1 | 20| 149 | 8585 | 55652 | 
| 24 | 16 django/db/migrations/migration.py | 1 | 88| 714 | 9299 | 57285 | 
| 25 | 16 django/db/migrations/state.py | 349 | 399| 471 | 9770 | 57285 | 
| 26 | 16 django/db/models/base.py | 1388 | 1443| 491 | 10261 | 57285 | 
| 27 | 17 django/db/models/fields/__init__.py | 2415 | 2440| 143 | 10404 | 74797 | 
| 28 | 18 django/db/migrations/operations/models.py | 345 | 394| 493 | 10897 | 81493 | 
| 29 | 18 django/core/management/commands/makemigrations.py | 60 | 146| 788 | 11685 | 81493 | 
| 30 | 18 django/db/models/base.py | 404 | 503| 856 | 12541 | 81493 | 
| 31 | 19 django/db/migrations/operations/fields.py | 1 | 37| 237 | 12778 | 84760 | 
| 32 | 19 django/db/models/fields/__init__.py | 1760 | 1805| 279 | 13057 | 84760 | 
| 33 | 19 django/db/models/fields/related.py | 1315 | 1387| 616 | 13673 | 84760 | 
| 34 | 19 django/db/models/fields/related.py | 738 | 756| 222 | 13895 | 84760 | 
| 35 | 19 django/db/models/base.py | 212 | 322| 866 | 14761 | 84760 | 
| 36 | 19 django/db/migrations/autodetector.py | 1209 | 1221| 131 | 14892 | 84760 | 
| 37 | 19 django/db/models/fields/related.py | 284 | 318| 293 | 15185 | 84760 | 
| 38 | 19 django/db/migrations/questioner.py | 227 | 240| 123 | 15308 | 84760 | 
| 39 | 20 django/db/backends/base/schema.py | 885 | 904| 296 | 15604 | 96075 | 
| 40 | 21 django/db/backends/sqlite3/schema.py | 367 | 413| 444 | 16048 | 100040 | 
| 41 | 21 django/db/migrations/operations/fields.py | 357 | 384| 289 | 16337 | 100040 | 
| 42 | 21 django/db/models/base.py | 1250 | 1279| 242 | 16579 | 100040 | 
| 43 | 22 django/db/migrations/exceptions.py | 1 | 55| 250 | 16829 | 100291 | 
| 44 | 22 django/db/models/fields/related.py | 171 | 188| 166 | 16995 | 100291 | 
| 45 | 22 django/db/migrations/operations/fields.py | 241 | 251| 146 | 17141 | 100291 | 
| 46 | 22 django/db/backends/sqlite3/schema.py | 101 | 138| 486 | 17627 | 100291 | 
| 47 | 22 django/core/management/commands/makemigrations.py | 147 | 184| 302 | 17929 | 100291 | 
| 48 | 22 django/db/migrations/autodetector.py | 224 | 237| 199 | 18128 | 100291 | 
| 49 | 22 django/db/migrations/operations/fields.py | 302 | 355| 535 | 18663 | 100291 | 
| 50 | 22 django/db/models/base.py | 1494 | 1526| 231 | 18894 | 100291 | 
| 51 | 22 django/db/models/fields/__init__.py | 2030 | 2060| 199 | 19093 | 100291 | 
| 52 | 22 django/db/models/fields/__init__.py | 1 | 81| 633 | 19726 | 100291 | 
| 53 | 23 django/contrib/contenttypes/fields.py | 431 | 452| 248 | 19974 | 105703 | 
| 54 | 23 django/db/models/fields/related.py | 1389 | 1419| 322 | 20296 | 105703 | 
| 55 | 23 django/db/migrations/operations/models.py | 304 | 343| 406 | 20702 | 105703 | 
| 56 | 23 django/db/models/base.py | 1761 | 1832| 565 | 21267 | 105703 | 
| 57 | 23 django/db/migrations/autodetector.py | 49 | 87| 322 | 21589 | 105703 | 
| 58 | 23 django/db/models/fields/related.py | 616 | 638| 185 | 21774 | 105703 | 
| 59 | 23 django/db/migrations/autodetector.py | 1067 | 1084| 180 | 21954 | 105703 | 
| 60 | 23 django/db/models/fields/related.py | 127 | 154| 202 | 22156 | 105703 | 
| 61 | 23 django/db/migrations/autodetector.py | 847 | 881| 339 | 22495 | 105703 | 
| 62 | 23 django/db/migrations/autodetector.py | 1086 | 1121| 312 | 22807 | 105703 | 
| 63 | 23 django/db/migrations/operations/models.py | 102 | 118| 147 | 22954 | 105703 | 
| 64 | 23 django/db/migrations/state.py | 530 | 555| 228 | 23182 | 105703 | 
| 65 | 23 django/db/backends/sqlite3/schema.py | 140 | 221| 820 | 24002 | 105703 | 
| 66 | 23 django/db/migrations/questioner.py | 187 | 205| 237 | 24239 | 105703 | 
| 67 | 23 django/db/migrations/operations/models.py | 120 | 239| 827 | 25066 | 105703 | 
| 68 | 24 django/db/migrations/executor.py | 298 | 393| 841 | 25907 | 109124 | 
| 69 | 24 django/db/migrations/autodetector.py | 1123 | 1144| 231 | 26138 | 109124 | 
| 70 | **24 django/db/migrations/serializer.py** | 119 | 138| 136 | 26274 | 109124 | 
| 71 | 24 django/db/models/fields/__init__.py | 1731 | 1758| 215 | 26489 | 109124 | 
| 72 | 24 django/db/migrations/operations/fields.py | 103 | 123| 223 | 26712 | 109124 | 
| 73 | 24 django/db/models/fields/__init__.py | 1958 | 1994| 198 | 26910 | 109124 | 
| 74 | 24 django/db/migrations/autodetector.py | 1027 | 1043| 188 | 27098 | 109124 | 
| 75 | 24 django/db/migrations/operations/models.py | 1 | 38| 238 | 27336 | 109124 | 
| 76 | 25 django/forms/models.py | 207 | 276| 623 | 27959 | 120637 | 
| 77 | 25 django/db/models/fields/related.py | 565 | 598| 318 | 28277 | 120637 | 
| 78 | 25 django/db/models/fields/__init__.py | 2282 | 2332| 339 | 28616 | 120637 | 
| 79 | 25 django/db/models/fields/related.py | 83 | 106| 162 | 28778 | 120637 | 
| 80 | 25 django/db/migrations/autodetector.py | 508 | 524| 186 | 28964 | 120637 | 
| 81 | 25 django/db/migrations/operations/models.py | 460 | 485| 279 | 29243 | 120637 | 
| 82 | 25 django/db/migrations/operations/fields.py | 274 | 300| 158 | 29401 | 120637 | 
| 83 | 25 django/db/models/fields/related.py | 952 | 962| 121 | 29522 | 120637 | 
| 84 | 25 django/db/models/fields/__init__.py | 84 | 176| 822 | 30344 | 120637 | 
| 85 | 25 django/db/models/fields/__init__.py | 2335 | 2384| 311 | 30655 | 120637 | 
| 86 | 25 django/db/backends/base/schema.py | 31 | 41| 120 | 30775 | 120637 | 
| 87 | 26 django/contrib/contenttypes/migrations/0002_remove_content_type_name.py | 1 | 40| 217 | 30992 | 120854 | 
| 88 | 26 django/db/migrations/state.py | 401 | 495| 808 | 31800 | 120854 | 
| 89 | 26 django/db/migrations/autodetector.py | 437 | 463| 256 | 32056 | 120854 | 
| 90 | 26 django/db/models/fields/related.py | 600 | 614| 186 | 32242 | 120854 | 
| 91 | 26 django/db/models/base.py | 169 | 211| 406 | 32648 | 120854 | 
| 92 | 27 django/db/models/__init__.py | 1 | 52| 591 | 33239 | 121445 | 
| 93 | 27 django/db/migrations/autodetector.py | 200 | 222| 239 | 33478 | 121445 | 
| 94 | 27 django/db/migrations/autodetector.py | 465 | 506| 418 | 33896 | 121445 | 
| 95 | 27 django/db/models/fields/related.py | 108 | 125| 155 | 34051 | 121445 | 
| 96 | 28 django/db/backends/mysql/schema.py | 79 | 89| 138 | 34189 | 122840 | 
| 97 | 28 django/db/models/fields/related.py | 1611 | 1645| 266 | 34455 | 122840 | 
| 98 | 28 django/db/models/fields/__init__.py | 1706 | 1729| 146 | 34601 | 122840 | 
| 99 | 28 django/db/backends/sqlite3/schema.py | 223 | 305| 731 | 35332 | 122840 | 
| 100 | 28 django/forms/models.py | 382 | 410| 240 | 35572 | 122840 | 
| 101 | 28 django/db/models/base.py | 1470 | 1492| 171 | 35743 | 122840 | 
| 102 | 28 django/db/models/fields/related.py | 847 | 873| 240 | 35983 | 122840 | 
| 103 | 28 django/db/migrations/operations/models.py | 41 | 100| 497 | 36480 | 122840 | 
| 104 | 28 django/db/models/fields/related.py | 487 | 507| 138 | 36618 | 122840 | 
| 105 | 28 django/db/migrations/autodetector.py | 987 | 1003| 188 | 36806 | 122840 | 
| 106 | 28 django/db/models/base.py | 1339 | 1369| 244 | 37050 | 122840 | 
| 107 | 29 django/core/serializers/base.py | 273 | 294| 186 | 37236 | 125233 | 
| 108 | 29 django/db/models/fields/__init__.py | 364 | 390| 199 | 37435 | 125233 | 
| 109 | 29 django/db/migrations/operations/fields.py | 220 | 239| 205 | 37640 | 125233 | 
| 110 | 29 django/db/migrations/state.py | 293 | 317| 266 | 37906 | 125233 | 
| 111 | 29 django/db/models/fields/related.py | 509 | 563| 409 | 38315 | 125233 | 
| 112 | 29 django/contrib/contenttypes/fields.py | 677 | 702| 254 | 38569 | 125233 | 
| 113 | 29 django/db/migrations/operations/fields.py | 253 | 271| 161 | 38730 | 125233 | 
| 114 | 30 django/db/models/options.py | 262 | 294| 343 | 39073 | 132332 | 
| **-> 115 <-** | **30 django/db/migrations/serializer.py** | 258 | 277| 155 | 39228 | 132332 | 
| 116 | 31 django/db/models/fields/related_descriptors.py | 671 | 725| 511 | 39739 | 142678 | 
| 117 | 31 django/forms/models.py | 952 | 985| 367 | 40106 | 142678 | 
| 118 | 31 django/db/models/fields/related.py | 640 | 656| 163 | 40269 | 142678 | 
| 119 | 31 django/db/migrations/autodetector.py | 796 | 845| 570 | 40839 | 142678 | 
| 120 | 31 django/db/migrations/operations/fields.py | 91 | 101| 125 | 40964 | 142678 | 
| 121 | 31 django/db/models/fields/related.py | 1 | 34| 244 | 41208 | 142678 | 
| 122 | 31 django/db/models/fields/related.py | 964 | 991| 215 | 41423 | 142678 | 
| 123 | 32 django/db/migrations/operations/utils.py | 1 | 14| 138 | 41561 | 143158 | 
| 124 | 32 django/db/models/fields/__init__.py | 1007 | 1033| 229 | 41790 | 143158 | 
| 125 | 33 django/forms/fields.py | 1173 | 1203| 182 | 41972 | 152105 | 
| 126 | 33 django/db/migrations/operations/models.py | 591 | 607| 215 | 42187 | 152105 | 
| 127 | 33 django/db/models/fields/related.py | 1421 | 1461| 399 | 42586 | 152105 | 
| 128 | 33 django/db/models/fields/related.py | 156 | 169| 144 | 42730 | 152105 | 
| 129 | 33 django/forms/models.py | 1085 | 1125| 306 | 43036 | 152105 | 
| 130 | 34 django/core/serializers/python.py | 62 | 75| 129 | 43165 | 153340 | 
| 131 | 34 django/db/migrations/state.py | 1 | 24| 191 | 43356 | 153340 | 
| 132 | 34 django/db/models/fields/__init__.py | 244 | 306| 448 | 43804 | 153340 | 
| 133 | 34 django/db/models/fields/related_descriptors.py | 343 | 362| 156 | 43960 | 153340 | 
| 134 | 34 django/db/backends/base/schema.py | 44 | 119| 790 | 44750 | 153340 | 
| 135 | 34 django/db/migrations/autodetector.py | 883 | 902| 184 | 44934 | 153340 | 
| 136 | 34 django/core/management/commands/makemigrations.py | 23 | 58| 284 | 45218 | 153340 | 
| 137 | 34 django/db/migrations/autodetector.py | 673 | 705| 283 | 45501 | 153340 | 
| 138 | 35 django/db/migrations/operations/base.py | 1 | 102| 783 | 46284 | 154419 | 
| 139 | 35 django/db/models/fields/__init__.py | 338 | 362| 184 | 46468 | 154419 | 
| 140 | 35 django/db/models/fields/__init__.py | 308 | 336| 205 | 46673 | 154419 | 
| 141 | 35 django/db/migrations/autodetector.py | 358 | 372| 141 | 46814 | 154419 | 
| 142 | 35 django/db/migrations/questioner.py | 143 | 160| 183 | 46997 | 154419 | 
| 143 | 35 django/core/management/commands/makemigrations.py | 186 | 229| 450 | 47447 | 154419 | 
| 144 | 35 django/db/migrations/operations/models.py | 396 | 412| 182 | 47629 | 154419 | 
| 145 | 35 django/db/models/fields/related.py | 190 | 254| 673 | 48302 | 154419 | 
| 146 | 35 django/db/backends/base/schema.py | 699 | 768| 740 | 49042 | 154419 | 
| 147 | 35 django/db/migrations/autodetector.py | 707 | 794| 789 | 49831 | 154419 | 
| 148 | 35 django/db/models/base.py | 1861 | 1912| 351 | 50182 | 154419 | 
| 149 | 35 django/db/models/fields/__init__.py | 1997 | 2027| 252 | 50434 | 154419 | 
| 150 | 36 django/db/migrations/writer.py | 201 | 301| 619 | 51053 | 156666 | 
| 151 | 36 django/db/models/base.py | 1068 | 1111| 404 | 51457 | 156666 | 
| 152 | 37 django/db/migrations/recorder.py | 1 | 22| 153 | 51610 | 157336 | 
| 153 | 37 django/db/migrations/state.py | 557 | 578| 229 | 51839 | 157336 | 
| 154 | 37 django/db/backends/sqlite3/schema.py | 307 | 328| 218 | 52057 | 157336 | 
| 155 | 37 django/db/models/fields/related_descriptors.py | 308 | 322| 182 | 52239 | 157336 | 
| 156 | 37 django/db/models/base.py | 948 | 962| 212 | 52451 | 157336 | 
| 157 | 37 django/db/migrations/operations/fields.py | 39 | 67| 285 | 52736 | 157336 | 
| 158 | 38 django/db/backends/oracle/schema.py | 57 | 77| 249 | 52985 | 159092 | 
| 159 | 38 django/db/backends/base/schema.py | 1048 | 1062| 123 | 53108 | 159092 | 
| 160 | 38 django/db/models/fields/__init__.py | 2387 | 2412| 217 | 53325 | 159092 | 
| 161 | 38 django/db/backends/base/schema.py | 769 | 809| 511 | 53836 | 159092 | 
| 162 | 38 django/db/models/fields/related_descriptors.py | 1010 | 1037| 334 | 54170 | 159092 | 
| 163 | 38 django/db/migrations/state.py | 166 | 190| 213 | 54383 | 159092 | 
| 164 | 38 django/db/migrations/migration.py | 127 | 194| 585 | 54968 | 159092 | 
| 165 | 38 django/db/models/fields/related.py | 1463 | 1487| 295 | 55263 | 159092 | 
| 166 | 38 django/db/backends/base/schema.py | 626 | 698| 792 | 56055 | 159092 | 
| 167 | 39 django/db/migrations/__init__.py | 1 | 3| 0 | 56055 | 159116 | 
| 168 | 39 django/forms/models.py | 918 | 950| 353 | 56408 | 159116 | 
| 169 | 39 django/db/models/fields/related.py | 1091 | 1167| 524 | 56932 | 159116 | 
| 170 | 39 django/forms/models.py | 279 | 307| 288 | 57220 | 159116 | 
| 171 | 40 django/db/migrations/optimizer.py | 41 | 71| 249 | 57469 | 159712 | 
| 172 | 40 django/db/models/base.py | 1445 | 1468| 176 | 57645 | 159712 | 
| 173 | 40 django/db/models/base.py | 1 | 50| 330 | 57975 | 159712 | 
| 174 | 40 django/db/migrations/autodetector.py | 89 | 101| 118 | 58093 | 159712 | 
| 175 | 40 django/db/models/fields/related.py | 1169 | 1200| 180 | 58273 | 159712 | 
| 176 | 40 django/db/backends/base/schema.py | 441 | 494| 580 | 58853 | 159712 | 
| 177 | 41 django/db/models/enums.py | 36 | 58| 183 | 59036 | 160295 | 
| 178 | 41 django/core/management/commands/makemigrations.py | 231 | 311| 824 | 59860 | 160295 | 
| 179 | 42 django/db/models/fields/files.py | 150 | 209| 645 | 60505 | 164017 | 
| 180 | 42 django/db/models/fields/__init__.py | 750 | 772| 236 | 60741 | 164017 | 
| 181 | 42 django/db/models/fields/related.py | 896 | 916| 178 | 60919 | 164017 | 
| 182 | 43 django/db/models/fields/related_lookups.py | 62 | 99| 447 | 61366 | 165466 | 
| 183 | 43 django/db/backends/base/schema.py | 526 | 565| 470 | 61836 | 165466 | 
| 184 | **43 django/db/migrations/serializer.py** | 280 | 311| 270 | 62106 | 165466 | 
| 185 | 43 django/db/backends/base/schema.py | 278 | 299| 173 | 62279 | 165466 | 
| 186 | 43 django/db/backends/mysql/schema.py | 91 | 104| 148 | 62427 | 165466 | 
| 187 | 43 django/db/backends/base/schema.py | 370 | 384| 182 | 62609 | 165466 | 
| 188 | 43 django/db/models/fields/related_lookups.py | 46 | 60| 224 | 62833 | 165466 | 
| 189 | 43 django/db/migrations/writer.py | 118 | 199| 744 | 63577 | 165466 | 
| 190 | 44 django/core/management/commands/showmigrations.py | 42 | 63| 158 | 63735 | 166652 | 
| 191 | 44 django/db/migrations/state.py | 79 | 104| 252 | 63987 | 166652 | 
| 192 | 44 django/db/models/fields/related_descriptors.py | 1112 | 1158| 489 | 64476 | 166652 | 
| 193 | 44 django/forms/models.py | 1 | 28| 215 | 64691 | 166652 | 
| 194 | 44 django/db/migrations/autodetector.py | 1045 | 1065| 136 | 64827 | 166652 | 
| 195 | 44 django/db/models/enums.py | 61 | 82| 118 | 64945 | 166652 | 
| 196 | 45 django/contrib/auth/migrations/0007_alter_validators_add_error_messages.py | 1 | 25| 137 | 65082 | 166789 | 
| 197 | 45 django/db/backends/base/schema.py | 1031 | 1046| 170 | 65252 | 166789 | 
| 198 | 46 django/contrib/gis/db/backends/mysql/schema.py | 40 | 63| 190 | 65442 | 167422 | 
| 199 | 46 django/db/migrations/state.py | 154 | 164| 132 | 65574 | 167422 | 
| 200 | 47 django/db/migrations/loader.py | 146 | 172| 291 | 65865 | 170313 | 
| 201 | 47 django/db/models/fields/files.py | 1 | 130| 905 | 66770 | 170313 | 
| 202 | 47 django/contrib/contenttypes/fields.py | 21 | 109| 571 | 67341 | 170313 | 
| 203 | 47 django/db/migrations/questioner.py | 162 | 185| 246 | 67587 | 170313 | 
| 204 | 47 django/db/models/fields/related_descriptors.py | 607 | 639| 323 | 67910 | 170313 | 
| 205 | 48 django/db/backends/postgresql/schema.py | 79 | 147| 539 | 68449 | 172256 | 
| 206 | 49 django/contrib/sites/managers.py | 1 | 61| 385 | 68834 | 172641 | 


### Hint

```
This should be possible to do by relying on __qualname__ (instead of __name__) now that master is Python 3 only.
​PR
I think we should focus on using __qualname__ during migration serialization as well instead of simply solving the field subclasses case.
In fb0f987: Fixed #27914 -- Added support for nested classes in Field.deconstruct()/repr().
In 451b585: Refs #27914 -- Used qualname in model operations' deconstruct().
I am still encountering this issue when running makemigrations on models that include a django-enumfields EnumField. From tracing through the code, I believe the Enum is getting serialized using the django.db.migrations.serializer.TypeSerializer, which still uses the __name__ rather than __qualname__. As a result, the Enum's path gets resolved to app_name.models.enum_name and the generated migration file throws an error "app_name.models has no 'enum_name' member". The correct path for the inner class should be app_name.models.model_name.enum_name. ​https://github.com/django/django/blob/master/django/db/migrations/serializer.py#L266
Reopening it. Will recheck with nested enum field.
​PR for fixing enum class as an inner class of model.
In d3030dea: Refs #27914 -- Moved test enum.Enum subclasses outside of WriterTests.test_serialize_enums().
In 6452112: Refs #27914 -- Fixed serialization of nested enum.Enum classes in migrations.
In 1a4db2c: [3.0.x] Refs #27914 -- Moved test enum.Enum subclasses outside of WriterTests.test_serialize_enums(). Backport of d3030deaaa50b7814e34ef1e71f2afaf97c6bec6 from master
In 30271a47: [3.0.x] Refs #27914 -- Fixed serialization of nested enum.Enum classes in migrations. Backport of 6452112640081ac8838147a8ba192c45879203d8 from master
commit 6452112640081ac8838147a8ba192c45879203d8 does not resolve this ticket. The commit patched the EnumSerializer with __qualname__, which works for Enum members. However, the serializer_factory is returning TypeSerializer for the Enum subclass, which is still using __name__ With v3.0.x introducing models.Choices, models.IntegerChoices, using nested enums will become a common pattern; serializing them properly with __qualname__ seems prudent. Here's a patch for the 3.0rc1 build ​https://github.com/django/django/files/3879265/django_db_migrations_serializer_TypeSerializer.patch.txt
Agreed, we should fix this.
I will create a patch a soon as possible.
Submitted PR: ​https://github.com/django/django/pull/12125
PR: ​https://github.com/django/django/pull/12125
```

## Patch

```diff
diff --git a/django/db/migrations/serializer.py b/django/db/migrations/serializer.py
--- a/django/db/migrations/serializer.py
+++ b/django/db/migrations/serializer.py
@@ -269,7 +269,7 @@ def serialize(self):
             if module == builtins.__name__:
                 return self.value.__name__, set()
             else:
-                return "%s.%s" % (module, self.value.__name__), {"import %s" % module}
+                return "%s.%s" % (module, self.value.__qualname__), {"import %s" % module}
 
 
 class UUIDSerializer(BaseSerializer):

```

## Test Patch

```diff
diff --git a/tests/migrations/test_writer.py b/tests/migrations/test_writer.py
--- a/tests/migrations/test_writer.py
+++ b/tests/migrations/test_writer.py
@@ -26,6 +26,11 @@
 from .models import FoodManager, FoodQuerySet
 
 
+class DeconstructibleInstances:
+    def deconstruct(self):
+        return ('DeconstructibleInstances', [], {})
+
+
 class Money(decimal.Decimal):
     def deconstruct(self):
         return (
@@ -188,6 +193,10 @@ class NestedEnum(enum.IntEnum):
         A = 1
         B = 2
 
+    class NestedChoices(models.TextChoices):
+        X = 'X', 'X value'
+        Y = 'Y', 'Y value'
+
     def safe_exec(self, string, value=None):
         d = {}
         try:
@@ -383,6 +392,18 @@ class DateChoices(datetime.date, models.Choices):
             "default=datetime.date(1969, 11, 19))"
         )
 
+    def test_serialize_nested_class(self):
+        for nested_cls in [self.NestedEnum, self.NestedChoices]:
+            cls_name = nested_cls.__name__
+            with self.subTest(cls_name):
+                self.assertSerializedResultEqual(
+                    nested_cls,
+                    (
+                        "migrations.test_writer.WriterTests.%s" % cls_name,
+                        {'import migrations.test_writer'},
+                    ),
+                )
+
     def test_serialize_uuid(self):
         self.assertSerializedEqual(uuid.uuid1())
         self.assertSerializedEqual(uuid.uuid4())
@@ -726,10 +747,6 @@ def test_deconstruct_class_arguments(self):
         # Yes, it doesn't make sense to use a class as a default for a
         # CharField. It does make sense for custom fields though, for example
         # an enumfield that takes the enum class as an argument.
-        class DeconstructibleInstances:
-            def deconstruct(self):
-                return ('DeconstructibleInstances', [], {})
-
         string = MigrationWriter.serialize(models.CharField(default=DeconstructibleInstances))[0]
         self.assertEqual(string, "models.CharField(default=migrations.test_writer.DeconstructibleInstances)")
 

```


## Code snippets

### 1 - django/contrib/contenttypes/migrations/0001_initial.py:

Start line: 1, End line: 35

```python
import django.contrib.contenttypes.models
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='ContentType',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('name', models.CharField(max_length=100)),
                ('app_label', models.CharField(max_length=100)),
                ('model', models.CharField(max_length=100, verbose_name='python model class name')),
            ],
            options={
                'ordering': ('name',),
                'db_table': 'django_content_type',
                'verbose_name': 'content type',
                'verbose_name_plural': 'content types',
            },
            bases=(models.Model,),
            managers=[
                ('objects', django.contrib.contenttypes.models.ContentTypeManager()),
            ],
        ),
        migrations.AlterUniqueTogether(
            name='contenttype',
            unique_together={('app_label', 'model')},
        ),
    ]
```
### 2 - django/db/migrations/state.py:

Start line: 580, End line: 599

```python
class ModelState:

    def get_field_by_name(self, name):
        for fname, field in self.fields:
            if fname == name:
                return field
        raise ValueError("No field called %s on model %s" % (name, self.name))

    def get_index_by_name(self, name):
        for index in self.options['indexes']:
            if index.name == name:
                return index
        raise ValueError("No index named %s on model %s" % (name, self.name))

    def get_constraint_by_name(self, name):
        for constraint in self.options['constraints']:
            if constraint.name == name:
                return constraint
        raise ValueError('No constraint named %s on model %s' % (name, self.name))

    def __repr__(self):
        return "<%s: '%s.%s'>" % (self.__class__.__name__, self.app_label, self.name)
```
### 3 - django/db/models/fields/related.py:

Start line: 1202, End line: 1313

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
                                'use ForeignKey("self", symmetrical=False, through="%s").'
                            ) % relationship_model_name,
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
                                'use ForeignKey("self", symmetrical=False, through="%s").'
                            ) % relationship_model_name,
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
### 4 - django/db/models/fields/related.py:

Start line: 1044, End line: 1088

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
### 5 - django/db/migrations/serializer.py:

Start line: 196, End line: 217

```python
class ModelFieldSerializer(DeconstructableSerializer):
    def serialize(self):
        attr_name, path, args, kwargs = self.value.deconstruct()
        return self.serialize_deconstructed(path, args, kwargs)


class ModelManagerSerializer(DeconstructableSerializer):
    def serialize(self):
        as_manager, manager_path, qs_path, args, kwargs = self.value.deconstruct()
        if as_manager:
            name, imports = self._serialize_path(qs_path)
            return "%s.as_manager()" % name, imports
        else:
            return self.serialize_deconstructed(manager_path, args, kwargs)


class OperationSerializer(BaseSerializer):
    def serialize(self):
        from django.db.migrations.writer import OperationWriter
        string, imports = OperationWriter(self.value, indentation=0).serialize()
        # Nested operation, trailing comma is handled in upper OperationWriter._write()
        return string.rstrip(','), imports
```
### 6 - django/db/migrations/autodetector.py:

Start line: 904, End line: 985

```python
class MigrationAutodetector:

    def generate_altered_fields(self):
        """
        Make AlterField operations, or possibly RemovedField/AddField if alter
        isn's possible.
        """
        for app_label, model_name, field_name in sorted(self.old_field_keys & self.new_field_keys):
            # Did the field change?
            old_model_name = self.renamed_models.get((app_label, model_name), model_name)
            old_field_name = self.renamed_fields.get((app_label, model_name, field_name), field_name)
            old_field = self.old_apps.get_model(app_label, old_model_name)._meta.get_field(old_field_name)
            new_field = self.new_apps.get_model(app_label, model_name)._meta.get_field(field_name)
            dependencies = []
            # Implement any model renames on relations; these are handled by RenameModel
            # so we need to exclude them from the comparison
            if hasattr(new_field, "remote_field") and getattr(new_field.remote_field, "model", None):
                rename_key = (
                    new_field.remote_field.model._meta.app_label,
                    new_field.remote_field.model._meta.model_name,
                )
                if rename_key in self.renamed_models:
                    new_field.remote_field.model = old_field.remote_field.model
                # Handle ForeignKey which can only have a single to_field.
                remote_field_name = getattr(new_field.remote_field, 'field_name', None)
                if remote_field_name:
                    to_field_rename_key = rename_key + (remote_field_name,)
                    if to_field_rename_key in self.renamed_fields:
                        # Repoint both model and field name because to_field
                        # inclusion in ForeignKey.deconstruct() is based on
                        # both.
                        new_field.remote_field.model = old_field.remote_field.model
                        new_field.remote_field.field_name = old_field.remote_field.field_name
                # Handle ForeignObjects which can have multiple from_fields/to_fields.
                from_fields = getattr(new_field, 'from_fields', None)
                if from_fields:
                    from_rename_key = (app_label, model_name)
                    new_field.from_fields = tuple([
                        self.renamed_fields.get(from_rename_key + (from_field,), from_field)
                        for from_field in from_fields
                    ])
                    new_field.to_fields = tuple([
                        self.renamed_fields.get(rename_key + (to_field,), to_field)
                        for to_field in new_field.to_fields
                    ])
                dependencies.extend(self._get_dependencies_for_foreign_key(new_field))
            if hasattr(new_field, "remote_field") and getattr(new_field.remote_field, "through", None):
                rename_key = (
                    new_field.remote_field.through._meta.app_label,
                    new_field.remote_field.through._meta.model_name,
                )
                if rename_key in self.renamed_models:
                    new_field.remote_field.through = old_field.remote_field.through
            old_field_dec = self.deep_deconstruct(old_field)
            new_field_dec = self.deep_deconstruct(new_field)
            if old_field_dec != new_field_dec:
                both_m2m = old_field.many_to_many and new_field.many_to_many
                neither_m2m = not old_field.many_to_many and not new_field.many_to_many
                if both_m2m or neither_m2m:
                    # Either both fields are m2m or neither is
                    preserve_default = True
                    if (old_field.null and not new_field.null and not new_field.has_default() and
                            not new_field.many_to_many):
                        field = new_field.clone()
                        new_default = self.questioner.ask_not_null_alteration(field_name, model_name)
                        if new_default is not models.NOT_PROVIDED:
                            field.default = new_default
                            preserve_default = False
                    else:
                        field = new_field
                    self.add_operation(
                        app_label,
                        operations.AlterField(
                            model_name=model_name,
                            name=field_name,
                            field=field,
                            preserve_default=preserve_default,
                        ),
                        dependencies=dependencies,
                    )
                else:
                    # We cannot alter between m2m and concrete fields
                    self._generate_removed_field(app_label, model_name, field_name)
                    self._generate_added_field(app_label, model_name, field_name)
```
### 7 - django/contrib/admin/migrations/0001_initial.py:

Start line: 1, End line: 48

```python
import django.contrib.admin.models
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('contenttypes', '__first__'),
    ]

    operations = [
        migrations.CreateModel(
            name='LogEntry',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('action_time', models.DateTimeField(auto_now=True, verbose_name='action time')),
                ('object_id', models.TextField(null=True, verbose_name='object id', blank=True)),
                ('object_repr', models.CharField(max_length=200, verbose_name='object repr')),
                ('action_flag', models.PositiveSmallIntegerField(verbose_name='action flag')),
                ('change_message', models.TextField(verbose_name='change message', blank=True)),
                ('content_type', models.ForeignKey(
                    to_field='id',
                    on_delete=models.SET_NULL,
                    blank=True, null=True,
                    to='contenttypes.ContentType',
                    verbose_name='content type',
                )),
                ('user', models.ForeignKey(
                    to=settings.AUTH_USER_MODEL,
                    on_delete=models.CASCADE,
                    verbose_name='user',
                )),
            ],
            options={
                'ordering': ['-action_time'],
                'db_table': 'django_admin_log',
                'verbose_name': 'log entry',
                'verbose_name_plural': 'log entries',
            },
            bases=(models.Model,),
            managers=[
                ('objects', django.contrib.admin.models.LogEntryManager()),
            ],
        ),
    ]
```
### 8 - django/contrib/sites/migrations/0001_initial.py:

Start line: 1, End line: 32

```python
import django.contrib.sites.models
from django.contrib.sites.models import _simple_domain_name_validator
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = []

    operations = [
        migrations.CreateModel(
            name='Site',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('domain', models.CharField(
                    max_length=100, verbose_name='domain name', validators=[_simple_domain_name_validator]
                )),
                ('name', models.CharField(max_length=50, verbose_name='display name')),
            ],
            options={
                'ordering': ['domain'],
                'db_table': 'django_site',
                'verbose_name': 'site',
                'verbose_name_plural': 'sites',
            },
            bases=(models.Model,),
            managers=[
                ('objects', django.contrib.sites.models.SiteManager()),
            ],
        ),
    ]
```
### 9 - django/contrib/auth/migrations/0001_initial.py:

Start line: 1, End line: 105

```python
import django.contrib.auth.models
from django.contrib.auth import validators
from django.db import migrations, models
from django.utils import timezone


class Migration(migrations.Migration):

    dependencies = [
        ('contenttypes', '__first__'),
    ]

    operations = [
        migrations.CreateModel(
            name='Permission',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('name', models.CharField(max_length=50, verbose_name='name')),
                ('content_type', models.ForeignKey(
                    to='contenttypes.ContentType',
                    on_delete=models.CASCADE,
                    to_field='id',
                    verbose_name='content type',
                )),
                ('codename', models.CharField(max_length=100, verbose_name='codename')),
            ],
            options={
                'ordering': ['content_type__app_label', 'content_type__model', 'codename'],
                'unique_together': {('content_type', 'codename')},
                'verbose_name': 'permission',
                'verbose_name_plural': 'permissions',
            },
            managers=[
                ('objects', django.contrib.auth.models.PermissionManager()),
            ],
        ),
        migrations.CreateModel(
            name='Group',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('name', models.CharField(unique=True, max_length=80, verbose_name='name')),
                ('permissions', models.ManyToManyField(to='auth.Permission', verbose_name='permissions', blank=True)),
            ],
            options={
                'verbose_name': 'group',
                'verbose_name_plural': 'groups',
            },
            managers=[
                ('objects', django.contrib.auth.models.GroupManager()),
            ],
        ),
        migrations.CreateModel(
            name='User',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('password', models.CharField(max_length=128, verbose_name='password')),
                ('last_login', models.DateTimeField(default=timezone.now, verbose_name='last login')),
                ('is_superuser', models.BooleanField(
                    default=False,
                    help_text='Designates that this user has all permissions without explicitly assigning them.',
                    verbose_name='superuser status'
                )),
                ('username', models.CharField(
                    help_text='Required. 30 characters or fewer. Letters, digits and @/./+/-/_ only.', unique=True,
                    max_length=30, verbose_name='username',
                    validators=[validators.UnicodeUsernameValidator()],
                )),
                ('first_name', models.CharField(max_length=30, verbose_name='first name', blank=True)),
                ('last_name', models.CharField(max_length=30, verbose_name='last name', blank=True)),
                ('email', models.EmailField(max_length=75, verbose_name='email address', blank=True)),
                ('is_staff', models.BooleanField(
                    default=False, help_text='Designates whether the user can log into this admin site.',
                    verbose_name='staff status'
                )),
                ('is_active', models.BooleanField(
                    default=True, verbose_name='active', help_text=(
                        'Designates whether this user should be treated as active. Unselect this instead of deleting '
                        'accounts.'
                    )
                )),
                ('date_joined', models.DateTimeField(default=timezone.now, verbose_name='date joined')),
                ('groups', models.ManyToManyField(
                    to='auth.Group', verbose_name='groups', blank=True, related_name='user_set',
                    related_query_name='user', help_text=(
                        'The groups this user belongs to. A user will get all permissions granted to each of their '
                        'groups.'
                    )
                )),
                ('user_permissions', models.ManyToManyField(
                    to='auth.Permission', verbose_name='user permissions', blank=True,
                    help_text='Specific permissions for this user.', related_name='user_set',
                    related_query_name='user')
                 ),
            ],
            options={
                'swappable': 'AUTH_USER_MODEL',
                'verbose_name': 'user',
                'verbose_name_plural': 'users',
            },
            managers=[
                ('objects', django.contrib.auth.models.UserManager()),
            ],
        ),
    ]
```
### 10 - django/db/migrations/operations/__init__.py:

Start line: 1, End line: 18

```python
from .fields import AddField, AlterField, RemoveField, RenameField
from .models import (
    AddConstraint, AddIndex, AlterIndexTogether, AlterModelManagers,
    AlterModelOptions, AlterModelTable, AlterOrderWithRespectTo,
    AlterUniqueTogether, CreateModel, DeleteModel, RemoveConstraint,
    RemoveIndex, RenameModel,
)
from .special import RunPython, RunSQL, SeparateDatabaseAndState

__all__ = [
    'CreateModel', 'DeleteModel', 'AlterModelTable', 'AlterUniqueTogether',
    'RenameModel', 'AlterIndexTogether', 'AlterModelOptions', 'AddIndex',
    'RemoveIndex', 'AddField', 'RemoveField', 'AlterField', 'RenameField',
    'AddConstraint', 'RemoveConstraint',
    'SeparateDatabaseAndState', 'RunSQL', 'RunPython',
    'AlterOrderWithRespectTo', 'AlterModelManagers',
]
```
### 70 - django/db/migrations/serializer.py:

Start line: 119, End line: 138

```python
class EnumSerializer(BaseSerializer):
    def serialize(self):
        enum_class = self.value.__class__
        module = enum_class.__module__
        return (
            '%s.%s[%r]' % (module, enum_class.__qualname__, self.value.name),
            {'import %s' % module},
        )


class FloatSerializer(BaseSimpleSerializer):
    def serialize(self):
        if math.isnan(self.value) or math.isinf(self.value):
            return 'float("{}")'.format(self.value), set()
        return super().serialize()


class FrozensetSerializer(BaseSequenceSerializer):
    def _format(self):
        return "frozenset([%s])"
```
### 115 - django/db/migrations/serializer.py:

Start line: 258, End line: 277

```python
class TypeSerializer(BaseSerializer):
    def serialize(self):
        special_cases = [
            (models.Model, "models.Model", []),
            (type(None), 'type(None)', []),
        ]
        for case, string, imports in special_cases:
            if case is self.value:
                return string, set(imports)
        if hasattr(self.value, "__module__"):
            module = self.value.__module__
            if module == builtins.__name__:
                return self.value.__name__, set()
            else:
                return "%s.%s" % (module, self.value.__name__), {"import %s" % module}


class UUIDSerializer(BaseSerializer):
    def serialize(self):
        return "uuid.%s" % repr(self.value), {"import uuid"}
```
### 184 - django/db/migrations/serializer.py:

Start line: 280, End line: 311

```python
class Serializer:
    _registry = {
        # Some of these are order-dependent.
        frozenset: FrozensetSerializer,
        list: SequenceSerializer,
        set: SetSerializer,
        tuple: TupleSerializer,
        dict: DictionarySerializer,
        models.Choices: ChoicesSerializer,
        enum.Enum: EnumSerializer,
        datetime.datetime: DatetimeDatetimeSerializer,
        (datetime.date, datetime.timedelta, datetime.time): DateTimeSerializer,
        SettingsReference: SettingsReferenceSerializer,
        float: FloatSerializer,
        (bool, int, type(None), bytes, str, range): BaseSimpleSerializer,
        decimal.Decimal: DecimalSerializer,
        (functools.partial, functools.partialmethod): FunctoolsPartialSerializer,
        (types.FunctionType, types.BuiltinFunctionType, types.MethodType): FunctionTypeSerializer,
        collections.abc.Iterable: IterableSerializer,
        (COMPILED_REGEX_TYPE, RegexObject): RegexSerializer,
        uuid.UUID: UUIDSerializer,
    }

    @classmethod
    def register(cls, type_, serializer):
        if not issubclass(serializer, BaseSerializer):
            raise ValueError("'%s' must inherit from 'BaseSerializer'." % serializer.__name__)
        cls._registry[type_] = serializer

    @classmethod
    def unregister(cls, type_):
        cls._registry.pop(type_)
```
