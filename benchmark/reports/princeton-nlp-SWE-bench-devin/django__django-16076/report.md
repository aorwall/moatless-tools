# django__django-16076

| **django/django** | `cfe3008123ed7c9e3f3a4d51d4a22f9d96634e33` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | 20105 |
| **Avg pos** | 61.0 |
| **Min pos** | 61 |
| **Max pos** | 61 |
| **Top file pos** | 8 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/db/models/sql/query.py b/django/db/models/sql/query.py
--- a/django/db/models/sql/query.py
+++ b/django/db/models/sql/query.py
@@ -1278,10 +1278,6 @@ def build_lookup(self, lookups, lhs, rhs):
         # supports both transform and lookup for the name.
         lookup_class = lhs.get_lookup(lookup_name)
         if not lookup_class:
-            if lhs.field.is_relation:
-                raise FieldError(
-                    "Related Field got invalid lookup: {}".format(lookup_name)
-                )
             # A lookup wasn't found. Try to interpret the name as a transform
             # and do an Exact lookup against it.
             lhs = self.try_transform(lhs, lookup_name)
@@ -1450,12 +1446,6 @@ def build_filter(
             can_reuse.update(join_list)
 
         if join_info.final_field.is_relation:
-            # No support for transforms for relational fields
-            num_lookups = len(lookups)
-            if num_lookups > 1:
-                raise FieldError(
-                    "Related Field got invalid lookup: {}".format(lookups[0])
-                )
             if len(targets) == 1:
                 col = self._get_col(targets[0], join_info.final_field, alias)
             else:

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/db/models/sql/query.py | 1281 | 1284 | - | 8 | -
| django/db/models/sql/query.py | 1453 | 1458 | 61 | 8 | 20105


## Problem Statement

```
Registering lookups on relation fields should be supported.
Description
	 
		(last modified by Thomas)
	 
Hello,
I have a model, let's call it Parent, with a field called object_id. I have another model, let's call it Child, which has a ForeignKey field called parent_object[_id] pointing to Parent.object_id. I need to do a lookup on Child where the FK starts with a certain character (it's a normalized value so, in the context of my app, it makes sense... also, I didn't design this schema and changing it is not a possibility ATM).
The problem is that if I do:
qs = Child.objects.filter(parent_object_id__startswith='c')
I get:
django.core.exceptions.FieldError: Related Field got invalid lookup: startswith
The only way I could make it work is:
qs = Child.objects.filter(parent_object__object_id__startswith='c')
but it forces a join between the table and the view and that's a no-no in my case (way too costly).
Here's the MCVE (tested on Python 3.9 + Django 4.0.7 and Python 3.10 + Django 4.1.1):
import django
django.setup()
from django.db import models
class Parent(models.Model):
	class Meta:
		app_label = 'test'
	object_id = models.CharField('Object ID', max_length=20, unique=True)
class Child(models.Model):
	class Meta:
		app_label = 'test'
	parent_object = models.ForeignKey(
		Parent, to_field='object_id', related_name='%(class)s_set', on_delete=models.CASCADE
	)
if __name__ == '__main__':
	qs = Child.objects.filter(parent_object_id__startswith='c') # fails with `FieldError: Related Field got invalid lookup: startswith`
	qs = Child.objects.filter(parent_object__object_id__startswith='c') # works but forces a costly join
And the error:
Traceback (most recent call last):
 File "/opt/src/orm_test.py", line 26, in <module>
	qs = Child.objects.filter(parent_object_id__startswith='c')
 File "/opt/src/venv/lib/python3.10/site-packages/django/db/models/manager.py", line 85, in manager_method
	return getattr(self.get_queryset(), name)(*args, **kwargs)
 File "/opt/src/venv/lib/python3.10/site-packages/django/db/models/query.py", line 1420, in filter
	return self._filter_or_exclude(False, args, kwargs)
 File "/opt/src/venv/lib/python3.10/site-packages/django/db/models/query.py", line 1438, in _filter_or_exclude
	clone._filter_or_exclude_inplace(negate, args, kwargs)
 File "/opt/src/venv/lib/python3.10/site-packages/django/db/models/query.py", line 1445, in _filter_or_exclude_inplace
	self._query.add_q(Q(*args, **kwargs))
 File "/opt/src/venv/lib/python3.10/site-packages/django/db/models/sql/query.py", line 1532, in add_q
	clause, _ = self._add_q(q_object, self.used_aliases)
 File "/opt/src/venv/lib/python3.10/site-packages/django/db/models/sql/query.py", line 1562, in _add_q
	child_clause, needed_inner = self.build_filter(
 File "/opt/src/venv/lib/python3.10/site-packages/django/db/models/sql/query.py", line 1478, in build_filter
	condition = self.build_lookup(lookups, col, value)
 File "/opt/src/venv/lib/python3.10/site-packages/django/db/models/sql/query.py", line 1292, in build_lookup
	raise FieldError(
django.core.exceptions.FieldError: Related Field got invalid lookup: startswith
Thanks for your help,
Regards,

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 django/db/models/fields/related_lookups.py | 151 | 168| 216 | 216 | 1600 | 
| 2 | 1 django/db/models/fields/related_lookups.py | 75 | 108| 369 | 585 | 1600 | 
| 3 | 1 django/db/models/fields/related_lookups.py | 170 | 210| 250 | 835 | 1600 | 
| 4 | 1 django/db/models/fields/related_lookups.py | 110 | 148| 324 | 1159 | 1600 | 
| 5 | 2 django/db/models/lookups.py | 521 | 575| 308 | 1467 | 6918 | 
| 6 | 2 django/db/models/lookups.py | 414 | 452| 336 | 1803 | 6918 | 
| 7 | 3 django/db/models/fields/related.py | 871 | 898| 237 | 2040 | 21528 | 
| 8 | 3 django/db/models/fields/related.py | 154 | 185| 209 | 2249 | 21528 | 
| 9 | 3 django/db/models/fields/related.py | 1450 | 1579| 984 | 3233 | 21528 | 
| 10 | 3 django/db/models/fields/related.py | 302 | 339| 296 | 3529 | 21528 | 
| 11 | 3 django/db/models/fields/related.py | 1079 | 1101| 180 | 3709 | 21528 | 
| 12 | 3 django/db/models/lookups.py | 454 | 487| 344 | 4053 | 21528 | 
| 13 | 3 django/db/models/fields/related.py | 226 | 301| 696 | 4749 | 21528 | 
| 14 | 4 django/contrib/gis/db/models/lookups.py | 272 | 297| 222 | 4971 | 24177 | 
| 15 | 4 django/db/models/fields/related.py | 582 | 602| 138 | 5109 | 24177 | 
| 16 | 5 django/db/models/query.py | 2232 | 2366| 1129 | 6238 | 44627 | 
| 17 | 6 django/contrib/admin/options.py | 432 | 490| 509 | 6747 | 63847 | 
| 18 | 6 django/db/models/lookups.py | 490 | 518| 290 | 7037 | 63847 | 
| 19 | 7 django/db/models/fields/related_descriptors.py | 1 | 89| 727 | 7764 | 75143 | 
| 20 | 7 django/db/models/fields/related.py | 819 | 869| 378 | 8142 | 75143 | 
| 21 | 7 django/db/models/fields/related.py | 1 | 40| 251 | 8393 | 75143 | 
| 22 | 7 django/db/models/fields/related.py | 733 | 755| 172 | 8565 | 75143 | 
| 23 | **8 django/db/models/sql/query.py** | 1228 | 1262| 339 | 8904 | 98034 | 
| 24 | 8 django/db/models/lookups.py | 616 | 636| 231 | 9135 | 98034 | 
| 25 | 8 django/db/models/lookups.py | 343 | 357| 170 | 9305 | 98034 | 
| 26 | 8 django/db/models/fields/related.py | 604 | 670| 497 | 9802 | 98034 | 
| 27 | **8 django/db/models/sql/query.py** | 1643 | 1741| 846 | 10648 | 98034 | 
| 28 | 8 django/db/models/fields/related.py | 1103 | 1120| 133 | 10781 | 98034 | 
| 29 | 9 django/db/backends/base/schema.py | 39 | 72| 214 | 10995 | 111900 | 
| 30 | 9 django/db/models/fields/related.py | 126 | 152| 171 | 11166 | 111900 | 
| 31 | 9 django/db/models/fields/related.py | 707 | 731| 210 | 11376 | 111900 | 
| 32 | 9 django/db/models/lookups.py | 360 | 411| 306 | 11682 | 111900 | 
| 33 | **9 django/db/models/sql/query.py** | 1596 | 1625| 287 | 11969 | 111900 | 
| 34 | 9 django/db/models/fields/related.py | 1581 | 1679| 655 | 12624 | 111900 | 
| 35 | 9 django/db/models/fields/related_lookups.py | 1 | 40| 221 | 12845 | 111900 | 
| 36 | 10 django/db/models/query_utils.py | 366 | 392| 289 | 13134 | 115117 | 
| 37 | 11 django/db/models/sql/compiler.py | 1199 | 1317| 928 | 14062 | 131489 | 
| 38 | 11 django/db/models/fields/related.py | 757 | 775| 166 | 14228 | 131489 | 
| 39 | 11 django/db/models/lookups.py | 108 | 126| 198 | 14426 | 131489 | 
| 40 | 11 django/db/models/fields/related.py | 485 | 511| 177 | 14603 | 131489 | 
| 41 | 11 django/db/models/fields/related.py | 1018 | 1054| 261 | 14864 | 131489 | 
| 42 | 11 django/db/models/lookups.py | 234 | 255| 180 | 15044 | 131489 | 
| 43 | 11 django/db/models/fields/related.py | 777 | 803| 222 | 15266 | 131489 | 
| 44 | 11 django/db/models/fields/related.py | 187 | 206| 155 | 15421 | 131489 | 
| 45 | 11 django/db/models/lookups.py | 212 | 231| 216 | 15637 | 131489 | 
| 46 | 11 django/db/models/lookups.py | 686 | 719| 141 | 15778 | 131489 | 
| 47 | 11 django/db/models/lookups.py | 166 | 186| 200 | 15978 | 131489 | 
| 48 | 11 django/db/models/fields/related.py | 341 | 378| 297 | 16275 | 131489 | 
| 49 | 12 django/db/models/fields/reverse_related.py | 20 | 151| 765 | 17040 | 134021 | 
| 50 | 12 django/db/models/lookups.py | 258 | 294| 304 | 17344 | 134021 | 
| 51 | 12 django/db/models/fields/related.py | 1160 | 1175| 136 | 17480 | 134021 | 
| 52 | 12 django/db/models/fields/related_descriptors.py | 92 | 110| 196 | 17676 | 134021 | 
| 53 | 13 django/db/models/fields/__init__.py | 600 | 618| 182 | 17858 | 152756 | 
| 54 | 13 django/db/models/fields/related.py | 380 | 401| 219 | 18077 | 152756 | 
| 55 | 13 django/db/models/fields/related.py | 992 | 1016| 176 | 18253 | 152756 | 
| 56 | 13 django/db/models/fields/reverse_related.py | 185 | 203| 167 | 18420 | 152756 | 
| 57 | 13 django/db/models/fields/related.py | 208 | 224| 142 | 18562 | 152756 | 
| 58 | 13 django/db/models/fields/related.py | 1177 | 1209| 246 | 18808 | 152756 | 
| 59 | 13 django/db/models/lookups.py | 1 | 50| 360 | 19168 | 152756 | 
| 60 | 13 django/contrib/gis/db/models/lookups.py | 374 | 396| 124 | 19292 | 152756 | 
| **-> 61 <-** | **13 django/db/models/sql/query.py** | 1422 | 1503| 813 | 20105 | 152756 | 
| 62 | 13 django/db/models/fields/related_descriptors.py | 153 | 197| 418 | 20523 | 152756 | 
| 63 | 13 django/db/models/lookups.py | 128 | 164| 270 | 20793 | 152756 | 
| 64 | 13 django/db/models/lookups.py | 639 | 664| 163 | 20956 | 152756 | 
| 65 | 13 django/db/models/fields/related.py | 1122 | 1158| 284 | 21240 | 152756 | 
| 66 | 14 django/contrib/admin/filters.py | 180 | 225| 427 | 21667 | 157025 | 
| 67 | 14 django/contrib/gis/db/models/lookups.py | 242 | 269| 134 | 21801 | 157025 | 
| 68 | 14 django/db/models/fields/related.py | 1418 | 1448| 172 | 21973 | 157025 | 
| 69 | 15 django/contrib/contenttypes/fields.py | 174 | 221| 417 | 22390 | 162656 | 
| 70 | 15 django/db/models/fields/__init__.py | 376 | 409| 214 | 22604 | 162656 | 
| 71 | 15 django/db/models/fields/related_descriptors.py | 113 | 151| 266 | 22870 | 162656 | 
| 72 | 15 django/db/models/fields/related_descriptors.py | 404 | 425| 158 | 23028 | 162656 | 
| 73 | 16 django/db/models/base.py | 1149 | 1168| 228 | 23256 | 181207 | 
| 74 | 17 django/db/models/__init__.py | 1 | 116| 682 | 23938 | 181889 | 
| 75 | 17 django/contrib/contenttypes/fields.py | 22 | 111| 560 | 24498 | 181889 | 
| 76 | **17 django/db/models/sql/query.py** | 1196 | 1226| 291 | 24789 | 181889 | 
| 77 | 17 django/db/models/fields/related.py | 1681 | 1731| 431 | 25220 | 181889 | 
| 78 | 17 django/contrib/contenttypes/fields.py | 363 | 389| 181 | 25401 | 181889 | 
| 79 | 18 django/contrib/postgres/lookups.py | 1 | 71| 398 | 25799 | 182287 | 
| 80 | 18 django/db/models/fields/related.py | 805 | 817| 113 | 25912 | 182287 | 
| 81 | 18 django/db/models/base.py | 1074 | 1126| 516 | 26428 | 182287 | 
| 82 | 18 django/db/models/fields/related_lookups.py | 43 | 72| 227 | 26655 | 182287 | 
| 83 | 18 django/db/models/fields/related.py | 1882 | 1929| 505 | 27160 | 182287 | 
| 84 | 18 django/db/models/fields/reverse_related.py | 165 | 183| 160 | 27320 | 182287 | 
| 85 | 18 django/contrib/contenttypes/fields.py | 471 | 498| 258 | 27578 | 182287 | 
| 86 | 18 django/db/models/fields/related.py | 672 | 705| 335 | 27913 | 182287 | 
| 87 | 18 django/db/models/base.py | 1809 | 1844| 242 | 28155 | 182287 | 
| 88 | 18 django/contrib/gis/db/models/lookups.py | 56 | 70| 158 | 28313 | 182287 | 
| 89 | 18 django/db/models/fields/related.py | 403 | 421| 161 | 28474 | 182287 | 
| 90 | 18 django/db/models/query.py | 1553 | 1576| 218 | 28692 | 182287 | 
| 91 | 18 django/contrib/admin/filters.py | 227 | 250| 202 | 28894 | 182287 | 
| 92 | 18 django/db/models/fields/related_descriptors.py | 610 | 638| 220 | 29114 | 182287 | 
| 93 | 18 django/db/models/query.py | 2604 | 2627| 201 | 29315 | 182287 | 
| 94 | 18 django/db/models/fields/related_descriptors.py | 199 | 248| 452 | 29767 | 182287 | 
| 95 | 18 django/db/models/lookups.py | 578 | 595| 134 | 29901 | 182287 | 
| 96 | 18 django/db/models/query.py | 2142 | 2200| 487 | 30388 | 182287 | 
| 97 | **18 django/db/models/sql/query.py** | 1064 | 1095| 307 | 30695 | 182287 | 
| 98 | 18 django/db/models/fields/related_descriptors.py | 427 | 450| 194 | 30889 | 182287 | 
| 99 | 18 django/db/models/fields/related_descriptors.py | 797 | 872| 592 | 31481 | 182287 | 
| 100 | 18 django/contrib/contenttypes/fields.py | 391 | 431| 408 | 31889 | 182287 | 
| 101 | 19 django/db/models/options.py | 603 | 624| 156 | 32045 | 189875 | 
| 102 | 19 django/db/models/lookups.py | 52 | 67| 156 | 32201 | 189875 | 
| 103 | 19 django/db/models/fields/related.py | 514 | 580| 367 | 32568 | 189875 | 
| 104 | 19 django/db/models/query.py | 2203 | 2231| 246 | 32814 | 189875 | 
| 105 | 19 django/db/models/base.py | 2256 | 2447| 1302 | 34116 | 189875 | 
| 106 | 19 django/db/models/lookups.py | 598 | 614| 117 | 34233 | 189875 | 
| 107 | 19 django/db/models/fields/related.py | 463 | 483| 169 | 34402 | 189875 | 
| 108 | 19 django/db/models/fields/related.py | 1960 | 1994| 266 | 34668 | 189875 | 
| 109 | 19 django/contrib/gis/db/models/lookups.py | 333 | 355| 200 | 34868 | 189875 | 
| 110 | 19 django/contrib/admin/filters.py | 488 | 501| 119 | 34987 | 189875 | 
| 111 | 19 django/db/models/fields/related.py | 89 | 124| 232 | 35219 | 189875 | 
| 112 | 19 django/db/models/fields/__init__.py | 304 | 374| 462 | 35681 | 189875 | 
| 113 | 20 django/contrib/admin/views/main.py | 554 | 586| 227 | 35908 | 194406 | 
| 114 | **20 django/db/models/sql/query.py** | 1314 | 1334| 172 | 36080 | 194406 | 
| 115 | **20 django/db/models/sql/query.py** | 1567 | 1594| 220 | 36300 | 194406 | 
| 116 | 20 django/db/models/fields/related_descriptors.py | 674 | 712| 350 | 36650 | 194406 | 
| 117 | 20 django/db/models/base.py | 1785 | 1807| 171 | 36821 | 194406 | 
| 118 | **20 django/db/models/sql/query.py** | 1918 | 1966| 445 | 37266 | 194406 | 
| 119 | 20 django/db/models/fields/related.py | 1056 | 1077| 147 | 37413 | 194406 | 
| 120 | 20 django/db/models/fields/__init__.py | 268 | 302| 234 | 37647 | 194406 | 
| 121 | 20 django/db/models/fields/related_descriptors.py | 714 | 737| 223 | 37870 | 194406 | 
| 122 | **20 django/db/models/sql/query.py** | 1835 | 1865| 244 | 38114 | 194406 | 
| 123 | 20 django/db/models/base.py | 1995 | 2048| 355 | 38469 | 194406 | 
| 124 | 20 django/db/models/options.py | 1 | 57| 347 | 38816 | 194406 | 
| 125 | 20 django/db/models/fields/__init__.py | 571 | 582| 184 | 39000 | 194406 | 
| 126 | 21 django/contrib/postgres/fields/ranges.py | 282 | 372| 479 | 39479 | 196816 | 
| 127 | 21 django/db/models/base.py | 2050 | 2155| 736 | 40215 | 196816 | 
| 128 | 21 django/db/models/query.py | 1906 | 1971| 539 | 40754 | 196816 | 
| 129 | 21 django/db/models/fields/reverse_related.py | 205 | 238| 306 | 41060 | 196816 | 
| 130 | 21 django/contrib/gis/db/models/lookups.py | 300 | 330| 267 | 41327 | 196816 | 
| 131 | 21 django/db/models/fields/reverse_related.py | 153 | 163| 130 | 41457 | 196816 | 
| 132 | 21 django/db/models/fields/related.py | 1931 | 1958| 305 | 41762 | 196816 | 
| 133 | 21 django/db/models/query.py | 2499 | 2531| 314 | 42076 | 196816 | 
| 134 | 21 django/db/models/lookups.py | 69 | 106| 310 | 42386 | 196816 | 
| 135 | 21 django/db/models/query_utils.py | 211 | 311| 784 | 43170 | 196816 | 
| 136 | 21 django/db/models/fields/__init__.py | 411 | 441| 208 | 43378 | 196816 | 
| 137 | **21 django/db/models/sql/query.py** | 1968 | 2035| 667 | 44045 | 196816 | 
| 138 | 21 django/db/models/sql/compiler.py | 1096 | 1197| 743 | 44788 | 196816 | 
| 139 | **21 django/db/models/sql/query.py** | 1 | 79| 576 | 45364 | 196816 | 
| 140 | 21 django/contrib/admin/options.py | 492 | 536| 348 | 45712 | 196816 | 
| 141 | 21 django/db/models/fields/related_descriptors.py | 366 | 383| 188 | 45900 | 196816 | 
| 142 | 21 django/db/models/options.py | 286 | 328| 355 | 46255 | 196816 | 
| 143 | 21 django/db/models/base.py | 1298 | 1345| 409 | 46664 | 196816 | 
| 144 | 21 django/db/models/fields/related.py | 901 | 990| 583 | 47247 | 196816 | 
| 145 | **21 django/db/models/sql/query.py** | 963 | 990| 272 | 47519 | 196816 | 
| 146 | 21 django/contrib/admin/filters.py | 253 | 273| 205 | 47724 | 196816 | 
| 147 | 21 django/contrib/contenttypes/fields.py | 433 | 450| 129 | 47853 | 196816 | 
| 148 | 21 django/db/models/sql/compiler.py | 1357 | 1378| 199 | 48052 | 196816 | 


### Hint

```
Thanks for the report. Django 4.2 (cd1afd553f9c175ebccfc0f50e72b43b9604bd97) allows ​registering lookups per field instances, so you will be able to register __startswith for parent_object_id, e.g. parent_field = Child._meta.get_field("parent_object_id") with register_lookup(parent_field, StartsWith): Child.objects.filter(parent_object_id__startswith='c') Duplicate of #29799.
Also, Thomas, it makes sense to assume there is a JOIN in the second queryset, but apparently there isn't: >>> print(Child.objects.filter(parent_object__object_id__startswith='c').query) SELECT "test_child"."id", "test_child"."parent_object_id" FROM "test_child" WHERE "test_child"."parent_object_id" LIKE c% ESCAPE '\'
@Mariusz Felisiak: Thanks for the heads-up, I hadn't found that ticket in my searches. @Alex Morega: Thank you. I should have checked the SQL of my example for the join. I relied on my findings on my existing code base which uses parent_object__object_id__startswith to circumvent the RelatedField lookup problem. I have the join there but it must be coming from somewhere else.
This also has similarities with this very old ticket https://code.djangoproject.com/ticket/2331#comment:7 so it's likely that parent_object_id__startswith='c' was actually never supported which is surprising to me.
I noticed that registering transforms on related fields doesn't work at all as we have ​a guard that seems completely unnecessary. A regression test: tests/queries/tests.py diff --git a/tests/queries/tests.py b/tests/queries/tests.py index 1bd72dd8b8..facf0fc421 100644 a b class Queries4Tests(TestCase): 16211621 date_obj, 16221622 ) 16231623 1624 def test_related_transform(self): 1625 from django.db.models.functions import ExtractYear 1626 from django.test.utils import register_lookup 1627 1628 date_obj = DateTimePK.objects.create() 1629 extra_obj = ExtraInfo.objects.create(info="extra", date=date_obj) 1630 fk_field = ExtraInfo._meta.get_field("date") 1631 with register_lookup(fk_field, ExtractYear): 1632 self.assertSequenceEqual( 1633 ExtraInfo.objects.filter(date__year=2022), 1634 [extra_obj], 1635 ) 1636 16241637 def test_ticket10181(self): 16251638 # Avoid raising an EmptyResultSet if an inner query is probably 16261639 # empty (and hence, not executed). We could consider this a release blocker after 10178197d57476f69688d4535e550a1ea3a5eac5 🤔.
I'm not sure I understand the rationale behind making this a release blocker as lookups on related fields were never supported? I remember having to remove this check a while ago when trying to add support for a m2m__exists lookup to work around #10060 ​in a feature branch. I think this a limitation we should lift with proper test coverage but I fail to see how it relates to 10178197d57476f69688d4535e550a1ea3a5eac5
```

## Patch

```diff
diff --git a/django/db/models/sql/query.py b/django/db/models/sql/query.py
--- a/django/db/models/sql/query.py
+++ b/django/db/models/sql/query.py
@@ -1278,10 +1278,6 @@ def build_lookup(self, lookups, lhs, rhs):
         # supports both transform and lookup for the name.
         lookup_class = lhs.get_lookup(lookup_name)
         if not lookup_class:
-            if lhs.field.is_relation:
-                raise FieldError(
-                    "Related Field got invalid lookup: {}".format(lookup_name)
-                )
             # A lookup wasn't found. Try to interpret the name as a transform
             # and do an Exact lookup against it.
             lhs = self.try_transform(lhs, lookup_name)
@@ -1450,12 +1446,6 @@ def build_filter(
             can_reuse.update(join_list)
 
         if join_info.final_field.is_relation:
-            # No support for transforms for relational fields
-            num_lookups = len(lookups)
-            if num_lookups > 1:
-                raise FieldError(
-                    "Related Field got invalid lookup: {}".format(lookups[0])
-                )
             if len(targets) == 1:
                 col = self._get_col(targets[0], join_info.final_field, alias)
             else:

```

## Test Patch

```diff
diff --git a/tests/lookup/tests.py b/tests/lookup/tests.py
--- a/tests/lookup/tests.py
+++ b/tests/lookup/tests.py
@@ -786,10 +786,16 @@ def test_unsupported_lookups(self):
 
     def test_relation_nested_lookup_error(self):
         # An invalid nested lookup on a related field raises a useful error.
-        msg = "Related Field got invalid lookup: editor"
+        msg = (
+            "Unsupported lookup 'editor' for ForeignKey or join on the field not "
+            "permitted."
+        )
         with self.assertRaisesMessage(FieldError, msg):
             Article.objects.filter(author__editor__name="James")
-        msg = "Related Field got invalid lookup: foo"
+        msg = (
+            "Unsupported lookup 'foo' for ForeignKey or join on the field not "
+            "permitted."
+        )
         with self.assertRaisesMessage(FieldError, msg):
             Tag.objects.filter(articles__foo="bar")
 
diff --git a/tests/queries/models.py b/tests/queries/models.py
--- a/tests/queries/models.py
+++ b/tests/queries/models.py
@@ -1,6 +1,8 @@
 """
 Various complex queries that have been problematic in the past.
 """
+import datetime
+
 from django.db import models
 from django.db.models.functions import Now
 
@@ -64,7 +66,7 @@ def __str__(self):
 
 
 class DateTimePK(models.Model):
-    date = models.DateTimeField(primary_key=True, auto_now_add=True)
+    date = models.DateTimeField(primary_key=True, default=datetime.datetime.now)
 
 
 class ExtraInfo(models.Model):
diff --git a/tests/queries/tests.py b/tests/queries/tests.py
--- a/tests/queries/tests.py
+++ b/tests/queries/tests.py
@@ -7,12 +7,13 @@
 
 from django.core.exceptions import EmptyResultSet, FieldError
 from django.db import DEFAULT_DB_ALIAS, connection
-from django.db.models import Count, Exists, F, Max, OuterRef, Q
+from django.db.models import CharField, Count, Exists, F, Max, OuterRef, Q
 from django.db.models.expressions import RawSQL
+from django.db.models.functions import ExtractYear, Length, LTrim
 from django.db.models.sql.constants import LOUTER
 from django.db.models.sql.where import AND, OR, NothingNode, WhereNode
 from django.test import SimpleTestCase, TestCase, skipUnlessDBFeature
-from django.test.utils import CaptureQueriesContext, ignore_warnings
+from django.test.utils import CaptureQueriesContext, ignore_warnings, register_lookup
 from django.utils.deprecation import RemovedInDjango50Warning
 
 from .models import (
@@ -391,6 +392,33 @@ def test_order_by_join_unref(self):
         qs = qs.order_by("id")
         self.assertNotIn("OUTER JOIN", str(qs.query))
 
+    def test_filter_by_related_field_transform(self):
+        extra_old = ExtraInfo.objects.create(
+            info="extra 12",
+            date=DateTimePK.objects.create(date=datetime.datetime(2020, 12, 10)),
+        )
+        ExtraInfo.objects.create(info="extra 11", date=DateTimePK.objects.create())
+        a5 = Author.objects.create(name="a5", num=5005, extra=extra_old)
+
+        fk_field = ExtraInfo._meta.get_field("date")
+        with register_lookup(fk_field, ExtractYear):
+            self.assertSequenceEqual(
+                ExtraInfo.objects.filter(date__year=2020),
+                [extra_old],
+            )
+            self.assertSequenceEqual(
+                Author.objects.filter(extra__date__year=2020), [a5]
+            )
+
+    def test_filter_by_related_field_nested_transforms(self):
+        extra = ExtraInfo.objects.create(info=" extra")
+        a5 = Author.objects.create(name="a5", num=5005, extra=extra)
+        info_field = ExtraInfo._meta.get_field("info")
+        with register_lookup(info_field, Length), register_lookup(CharField, LTrim):
+            self.assertSequenceEqual(
+                Author.objects.filter(extra__info__ltrim__length=5), [a5]
+            )
+
     def test_get_clears_ordering(self):
         """
         get() should clear ordering for optimization purposes.

```


## Code snippets

### 1 - django/db/models/fields/related_lookups.py:

Start line: 151, End line: 168

```python
class RelatedLookupMixin:
    def get_prep_lookup(self):
        if not isinstance(self.lhs, MultiColSource) and not hasattr(
            self.rhs, "resolve_expression"
        ):
            # If we get here, we are dealing with single-column relations.
            self.rhs = get_normalized_value(self.rhs, self.lhs)[0]
            # We need to run the related field's get_prep_value(). Consider case
            # ForeignKey to IntegerField given value 'abc'. The ForeignKey itself
            # doesn't have validation for non-integers, so we must run validation
            # using the target field.
            if self.prepare_rhs and hasattr(self.lhs.output_field, "path_infos"):
                # Get the target field. We can safely assume there is only one
                # as we don't get to the direct value branch otherwise.
                target_field = self.lhs.output_field.path_infos[-1].target_fields[-1]
                self.rhs = target_field.get_prep_value(self.rhs)

        return super().get_prep_lookup()
```
### 2 - django/db/models/fields/related_lookups.py:

Start line: 75, End line: 108

```python
class RelatedIn(In):
    def get_prep_lookup(self):
        if not isinstance(self.lhs, MultiColSource):
            if self.rhs_is_direct_value():
                # If we get here, we are dealing with single-column relations.
                self.rhs = [get_normalized_value(val, self.lhs)[0] for val in self.rhs]
                # We need to run the related field's get_prep_value(). Consider
                # case ForeignKey to IntegerField given value 'abc'. The
                # ForeignKey itself doesn't have validation for non-integers,
                # so we must run validation using the target field.
                if hasattr(self.lhs.output_field, "path_infos"):
                    # Run the target field's get_prep_value. We can safely
                    # assume there is only one as we don't get to the direct
                    # value branch otherwise.
                    target_field = self.lhs.output_field.path_infos[-1].target_fields[
                        -1
                    ]
                    self.rhs = [target_field.get_prep_value(v) for v in self.rhs]
            elif not getattr(self.rhs, "has_select_fields", True) and not getattr(
                self.lhs.field.target_field, "primary_key", False
            ):
                if (
                    getattr(self.lhs.output_field, "primary_key", False)
                    and self.lhs.output_field.model == self.rhs.model
                ):
                    # A case like
                    # Restaurant.objects.filter(place__in=restaurant_qs), where
                    # place is a OneToOneField and the primary key of
                    # Restaurant.
                    target_field = self.lhs.field.name
                else:
                    target_field = self.lhs.field.target_field.name
                self.rhs.set_values([target_field])
        return super().get_prep_lookup()
```
### 3 - django/db/models/fields/related_lookups.py:

Start line: 170, End line: 210

```python
class RelatedLookupMixin:

    def as_sql(self, compiler, connection):
        if isinstance(self.lhs, MultiColSource):
            assert self.rhs_is_direct_value()
            self.rhs = get_normalized_value(self.rhs, self.lhs)
            from django.db.models.sql.where import AND, WhereNode

            root_constraint = WhereNode()
            for target, source, val in zip(
                self.lhs.targets, self.lhs.sources, self.rhs
            ):
                lookup_class = target.get_lookup(self.lookup_name)
                root_constraint.add(
                    lookup_class(target.get_col(self.lhs.alias, source), val), AND
                )
            return root_constraint.as_sql(compiler, connection)
        return super().as_sql(compiler, connection)


class RelatedExact(RelatedLookupMixin, Exact):
    pass


class RelatedLessThan(RelatedLookupMixin, LessThan):
    pass


class RelatedGreaterThan(RelatedLookupMixin, GreaterThan):
    pass


class RelatedGreaterThanOrEqual(RelatedLookupMixin, GreaterThanOrEqual):
    pass


class RelatedLessThanOrEqual(RelatedLookupMixin, LessThanOrEqual):
    pass


class RelatedIsNull(RelatedLookupMixin, IsNull):
    pass
```
### 4 - django/db/models/fields/related_lookups.py:

Start line: 110, End line: 148

```python
class RelatedIn(In):

    def as_sql(self, compiler, connection):
        if isinstance(self.lhs, MultiColSource):
            # For multicolumn lookups we need to build a multicolumn where clause.
            # This clause is either a SubqueryConstraint (for values that need
            # to be compiled to SQL) or an OR-combined list of
            # (col1 = val1 AND col2 = val2 AND ...) clauses.
            from django.db.models.sql.where import (
                AND,
                OR,
                SubqueryConstraint,
                WhereNode,
            )

            root_constraint = WhereNode(connector=OR)
            if self.rhs_is_direct_value():
                values = [get_normalized_value(value, self.lhs) for value in self.rhs]
                for value in values:
                    value_constraint = WhereNode()
                    for source, target, val in zip(
                        self.lhs.sources, self.lhs.targets, value
                    ):
                        lookup_class = target.get_lookup("exact")
                        lookup = lookup_class(
                            target.get_col(self.lhs.alias, source), val
                        )
                        value_constraint.add(lookup, AND)
                    root_constraint.add(value_constraint, OR)
            else:
                root_constraint.add(
                    SubqueryConstraint(
                        self.lhs.alias,
                        [target.column for target in self.lhs.targets],
                        [source.name for source in self.lhs.sources],
                        self.rhs,
                    ),
                    AND,
                )
            return root_constraint.as_sql(compiler, connection)
        return super().as_sql(compiler, connection)
```
### 5 - django/db/models/lookups.py:

Start line: 521, End line: 575

```python
@Field.register_lookup
class Contains(PatternLookup):
    lookup_name = "contains"


@Field.register_lookup
class IContains(Contains):
    lookup_name = "icontains"


@Field.register_lookup
class StartsWith(PatternLookup):
    lookup_name = "startswith"
    param_pattern = "%s%%"


@Field.register_lookup
class IStartsWith(StartsWith):
    lookup_name = "istartswith"


@Field.register_lookup
class EndsWith(PatternLookup):
    lookup_name = "endswith"
    param_pattern = "%%%s"


@Field.register_lookup
class IEndsWith(EndsWith):
    lookup_name = "iendswith"


@Field.register_lookup
class Range(FieldGetDbPrepValueIterableMixin, BuiltinLookup):
    lookup_name = "range"

    def get_rhs_op(self, connection, rhs):
        return "BETWEEN %s AND %s" % (rhs[0], rhs[1])


@Field.register_lookup
class IsNull(BuiltinLookup):
    lookup_name = "isnull"
    prepare_rhs = False

    def as_sql(self, compiler, connection):
        if not isinstance(self.rhs, bool):
            raise ValueError(
                "The QuerySet value for an isnull lookup must be True or False."
            )
        sql, params = compiler.compile(self.lhs)
        if self.rhs:
            return "%s IS NULL" % sql, params
        else:
            return "%s IS NOT NULL" % sql, params
```
### 6 - django/db/models/lookups.py:

Start line: 414, End line: 452

```python
@Field.register_lookup
class In(FieldGetDbPrepValueIterableMixin, BuiltinLookup):
    lookup_name = "in"

    def get_prep_lookup(self):
        from django.db.models.sql.query import Query  # avoid circular import

        if isinstance(self.rhs, Query):
            self.rhs.clear_ordering(clear_default=True)
            if not self.rhs.has_select_fields:
                self.rhs.clear_select_clause()
                self.rhs.add_fields(["pk"])
        return super().get_prep_lookup()

    def process_rhs(self, compiler, connection):
        db_rhs = getattr(self.rhs, "_db", None)
        if db_rhs is not None and db_rhs != connection.alias:
            raise ValueError(
                "Subqueries aren't allowed across different databases. Force "
                "the inner query to be evaluated using `list(inner_query)`."
            )

        if self.rhs_is_direct_value():
            # Remove None from the list as NULL is never equal to anything.
            try:
                rhs = OrderedSet(self.rhs)
                rhs.discard(None)
            except TypeError:  # Unhashable items in self.rhs
                rhs = [r for r in self.rhs if r is not None]

            if not rhs:
                raise EmptyResultSet

            # rhs should be an iterable; use batch_process_rhs() to
            # prepare/transform those values.
            sqls, sqls_params = self.batch_process_rhs(compiler, connection, rhs)
            placeholder = "(" + ", ".join(sqls) + ")"
            return (placeholder, sqls_params)
        return super().process_rhs(compiler, connection)
```
### 7 - django/db/models/fields/related.py:

Start line: 871, End line: 898

```python
class ForeignObject(RelatedField):

    def contribute_to_related_class(self, cls, related):
        # Internal FK's - i.e., those with a related name ending with '+' -
        # and swapped models don't get a related descriptor.
        if (
            not self.remote_field.is_hidden()
            and not related.related_model._meta.swapped
        ):
            setattr(
                cls._meta.concrete_model,
                related.get_accessor_name(),
                self.related_accessor_class(related),
            )
            # While 'limit_choices_to' might be a callable, simply pass
            # it along for later - this is too early because it's still
            # model load time.
            if self.remote_field.limit_choices_to:
                cls._meta.related_fkey_lookups.append(
                    self.remote_field.limit_choices_to
                )


ForeignObject.register_lookup(RelatedIn)
ForeignObject.register_lookup(RelatedExact)
ForeignObject.register_lookup(RelatedLessThan)
ForeignObject.register_lookup(RelatedGreaterThan)
ForeignObject.register_lookup(RelatedGreaterThanOrEqual)
ForeignObject.register_lookup(RelatedLessThanOrEqual)
ForeignObject.register_lookup(RelatedIsNull)
```
### 8 - django/db/models/fields/related.py:

Start line: 154, End line: 185

```python
class RelatedField(FieldCacheMixin, Field):

    def _check_related_query_name_is_valid(self):
        if self.remote_field.is_hidden():
            return []
        rel_query_name = self.related_query_name()
        errors = []
        if rel_query_name.endswith("_"):
            errors.append(
                checks.Error(
                    "Reverse query name '%s' must not end with an underscore."
                    % rel_query_name,
                    hint=(
                        "Add or change a related_name or related_query_name "
                        "argument for this field."
                    ),
                    obj=self,
                    id="fields.E308",
                )
            )
        if LOOKUP_SEP in rel_query_name:
            errors.append(
                checks.Error(
                    "Reverse query name '%s' must not contain '%s'."
                    % (rel_query_name, LOOKUP_SEP),
                    hint=(
                        "Add or change a related_name or related_query_name "
                        "argument for this field."
                    ),
                    obj=self,
                    id="fields.E309",
                )
            )
        return errors
```
### 9 - django/db/models/fields/related.py:

Start line: 1450, End line: 1579

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
### 10 - django/db/models/fields/related.py:

Start line: 302, End line: 339

```python
class RelatedField(FieldCacheMixin, Field):

    def _check_clashes(self):
        # ... other code
        for clash_field in potential_clashes:
            # i.e. "app_label.Model.m2m".
            clash_name = "%s.%s" % (
                clash_field.related_model._meta.label,
                clash_field.field.name,
            )
            if not rel_is_hidden and clash_field.get_accessor_name() == rel_name:
                errors.append(
                    checks.Error(
                        f"Reverse accessor '{rel_opts.object_name}.{rel_name}' "
                        f"for '{field_name}' clashes with reverse accessor for "
                        f"'{clash_name}'.",
                        hint=(
                            "Add or change a related_name argument "
                            "to the definition for '%s' or '%s'."
                        )
                        % (field_name, clash_name),
                        obj=self,
                        id="fields.E304",
                    )
                )

            if clash_field.get_accessor_name() == rel_query_name:
                errors.append(
                    checks.Error(
                        "Reverse query name for '%s' clashes with reverse query name "
                        "for '%s'." % (field_name, clash_name),
                        hint=(
                            "Add or change a related_name argument "
                            "to the definition for '%s' or '%s'."
                        )
                        % (field_name, clash_name),
                        obj=self,
                        id="fields.E305",
                    )
                )

        return errors
```
### 23 - django/db/models/sql/query.py:

Start line: 1228, End line: 1262

```python
class Query(BaseExpression):

    def check_related_objects(self, field, value, opts):
        """Check the type of object passed to query relations."""
        if field.is_relation:
            # Check that the field and the queryset use the same model in a
            # query like .filter(author=Author.objects.all()). For example, the
            # opts would be Author's (from the author field) and value.model
            # would be Author.objects.all() queryset's .model (Author also).
            # The field is the related field on the lhs side.
            if (
                isinstance(value, Query)
                and not value.has_select_fields
                and not check_rel_lookup_compatibility(value.model, opts, field)
            ):
                raise ValueError(
                    'Cannot use QuerySet for "%s": Use a QuerySet for "%s".'
                    % (value.model._meta.object_name, opts.object_name)
                )
            elif hasattr(value, "_meta"):
                self.check_query_object_type(value, opts, field)
            elif hasattr(value, "__iter__"):
                for v in value:
                    self.check_query_object_type(v, opts, field)

    def check_filterable(self, expression):
        """Raise an error if expression cannot be used in a WHERE clause."""
        if hasattr(expression, "resolve_expression") and not getattr(
            expression, "filterable", True
        ):
            raise NotSupportedError(
                expression.__class__.__name__ + " is disallowed in the filter "
                "clause."
            )
        if hasattr(expression, "get_source_expressions"):
            for expr in expression.get_source_expressions():
                self.check_filterable(expr)
```
### 27 - django/db/models/sql/query.py:

Start line: 1643, End line: 1741

```python
class Query(BaseExpression):

    def names_to_path(self, names, opts, allow_many=True, fail_on_missing=False):
        # ... other code
        for pos, name in enumerate(names):
            cur_names_with_path = (name, [])
            if name == "pk":
                name = opts.pk.name

            field = None
            filtered_relation = None
            try:
                if opts is None:
                    raise FieldDoesNotExist
                field = opts.get_field(name)
            except FieldDoesNotExist:
                if name in self.annotation_select:
                    field = self.annotation_select[name].output_field
                elif name in self._filtered_relations and pos == 0:
                    filtered_relation = self._filtered_relations[name]
                    if LOOKUP_SEP in filtered_relation.relation_name:
                        parts = filtered_relation.relation_name.split(LOOKUP_SEP)
                        filtered_relation_path, field, _, _ = self.names_to_path(
                            parts,
                            opts,
                            allow_many,
                            fail_on_missing,
                        )
                        path.extend(filtered_relation_path[:-1])
                    else:
                        field = opts.get_field(filtered_relation.relation_name)
            if field is not None:
                # Fields that contain one-to-many relations with a generic
                # model (like a GenericForeignKey) cannot generate reverse
                # relations and therefore cannot be used for reverse querying.
                if field.is_relation and not field.related_model:
                    raise FieldError(
                        "Field %r does not generate an automatic reverse "
                        "relation and therefore cannot be used for reverse "
                        "querying. If it is a GenericForeignKey, consider "
                        "adding a GenericRelation." % name
                    )
                try:
                    model = field.model._meta.concrete_model
                except AttributeError:
                    # QuerySet.annotate() may introduce fields that aren't
                    # attached to a model.
                    model = None
            else:
                # We didn't find the current field, so move position back
                # one step.
                pos -= 1
                if pos == -1 or fail_on_missing:
                    available = sorted(
                        [
                            *get_field_names_from_opts(opts),
                            *self.annotation_select,
                            *self._filtered_relations,
                        ]
                    )
                    raise FieldError(
                        "Cannot resolve keyword '%s' into field. "
                        "Choices are: %s" % (name, ", ".join(available))
                    )
                break
            # Check if we need any joins for concrete inheritance cases (the
            # field lives in parent, but we are currently in one of its
            # children)
            if opts is not None and model is not opts.model:
                path_to_parent = opts.get_path_to_parent(model)
                if path_to_parent:
                    path.extend(path_to_parent)
                    cur_names_with_path[1].extend(path_to_parent)
                    opts = path_to_parent[-1].to_opts
            if hasattr(field, "path_infos"):
                if filtered_relation:
                    pathinfos = field.get_path_info(filtered_relation)
                else:
                    pathinfos = field.path_infos
                if not allow_many:
                    for inner_pos, p in enumerate(pathinfos):
                        if p.m2m:
                            cur_names_with_path[1].extend(pathinfos[0 : inner_pos + 1])
                            names_with_path.append(cur_names_with_path)
                            raise MultiJoin(pos + 1, names_with_path)
                last = pathinfos[-1]
                path.extend(pathinfos)
                final_field = last.join_field
                opts = last.to_opts
                targets = last.target_fields
                cur_names_with_path[1].extend(pathinfos)
                names_with_path.append(cur_names_with_path)
            else:
                # Local non-relational field.
                final_field = field
                targets = (field,)
                if fail_on_missing and pos + 1 != len(names):
                    raise FieldError(
                        "Cannot resolve keyword %r into field. Join on '%s'"
                        " not permitted." % (names[pos + 1], name)
                    )
                break
        return path, final_field, targets, names[pos + 1 :]
```
### 33 - django/db/models/sql/query.py:

Start line: 1596, End line: 1625

```python
class Query(BaseExpression):

    def add_filtered_relation(self, filtered_relation, alias):
        filtered_relation.alias = alias
        lookups = dict(get_children_from_q(filtered_relation.condition))
        relation_lookup_parts, relation_field_parts, _ = self.solve_lookup_type(
            filtered_relation.relation_name
        )
        if relation_lookup_parts:
            raise ValueError(
                "FilteredRelation's relation_name cannot contain lookups "
                "(got %r)." % filtered_relation.relation_name
            )
        for lookup in chain(lookups):
            lookup_parts, lookup_field_parts, _ = self.solve_lookup_type(lookup)
            shift = 2 if not lookup_parts else 1
            lookup_field_path = lookup_field_parts[:-shift]
            for idx, lookup_field_part in enumerate(lookup_field_path):
                if len(relation_field_parts) > idx:
                    if relation_field_parts[idx] != lookup_field_part:
                        raise ValueError(
                            "FilteredRelation's condition doesn't support "
                            "relations outside the %r (got %r)."
                            % (filtered_relation.relation_name, lookup)
                        )
                else:
                    raise ValueError(
                        "FilteredRelation's condition doesn't support nested "
                        "relations deeper than the relation_name (got %r for "
                        "%r)." % (lookup, filtered_relation.relation_name)
                    )
        self._filtered_relations[filtered_relation.alias] = filtered_relation
```
### 61 - django/db/models/sql/query.py:

Start line: 1422, End line: 1503

```python
class Query(BaseExpression):

    def build_filter(
        self,
        filter_expr,
        branch_negated=False,
        current_negated=False,
        can_reuse=None,
        allow_joins=True,
        split_subq=True,
        reuse_with_filtered_relation=False,
        check_filterable=True,
    ):
        # ... other code

        try:
            join_info = self.setup_joins(
                parts,
                opts,
                alias,
                can_reuse=can_reuse,
                allow_many=allow_many,
                reuse_with_filtered_relation=reuse_with_filtered_relation,
            )

            # Prevent iterator from being consumed by check_related_objects()
            if isinstance(value, Iterator):
                value = list(value)
            self.check_related_objects(join_info.final_field, value, join_info.opts)

            # split_exclude() needs to know which joins were generated for the
            # lookup parts
            self._lookup_joins = join_info.joins
        except MultiJoin as e:
            return self.split_exclude(filter_expr, can_reuse, e.names_with_path)

        # Update used_joins before trimming since they are reused to determine
        # which joins could be later promoted to INNER.
        used_joins.update(join_info.joins)
        targets, alias, join_list = self.trim_joins(
            join_info.targets, join_info.joins, join_info.path
        )
        if can_reuse is not None:
            can_reuse.update(join_list)

        if join_info.final_field.is_relation:
            # No support for transforms for relational fields
            num_lookups = len(lookups)
            if num_lookups > 1:
                raise FieldError(
                    "Related Field got invalid lookup: {}".format(lookups[0])
                )
            if len(targets) == 1:
                col = self._get_col(targets[0], join_info.final_field, alias)
            else:
                col = MultiColSource(
                    alias, targets, join_info.targets, join_info.final_field
                )
        else:
            col = self._get_col(targets[0], join_info.final_field, alias)

        condition = self.build_lookup(lookups, col, value)
        lookup_type = condition.lookup_name
        clause = WhereNode([condition], connector=AND)

        require_outer = (
            lookup_type == "isnull" and condition.rhs is True and not current_negated
        )
        if (
            current_negated
            and (lookup_type != "isnull" or condition.rhs is False)
            and condition.rhs is not None
        ):
            require_outer = True
            if lookup_type != "isnull":
                # The condition added here will be SQL like this:
                # NOT (col IS NOT NULL), where the first NOT is added in
                # upper layers of code. The reason for addition is that if col
                # is null, then col != someval will result in SQL "unknown"
                # which isn't the same as in Python. The Python None handling
                # is wanted, and it can be gotten by
                # (col IS NULL OR col != someval)
                #   <=>
                # NOT (col IS NOT NULL AND col = someval).
                if (
                    self.is_nullable(targets[0])
                    or self.alias_map[join_list[-1]].join_type == LOUTER
                ):
                    lookup_class = targets[0].get_lookup("isnull")
                    col = self._get_col(targets[0], join_info.targets[0], alias)
                    clause.add(lookup_class(col, False), AND)
                # If someval is a nullable column, someval IS NOT NULL is
                # added.
                if isinstance(value, Col) and self.is_nullable(value.target):
                    lookup_class = value.target.get_lookup("isnull")
                    clause.add(lookup_class(value, False), AND)
        return clause, used_joins if not require_outer else ()
```
### 76 - django/db/models/sql/query.py:

Start line: 1196, End line: 1226

```python
class Query(BaseExpression):

    def solve_lookup_type(self, lookup):
        """
        Solve the lookup type from the lookup (e.g.: 'foobar__id__icontains').
        """
        lookup_splitted = lookup.split(LOOKUP_SEP)
        if self.annotations:
            expression, expression_lookups = refs_expression(
                lookup_splitted, self.annotations
            )
            if expression:
                return expression_lookups, (), expression
        _, field, _, lookup_parts = self.names_to_path(lookup_splitted, self.get_meta())
        field_parts = lookup_splitted[0 : len(lookup_splitted) - len(lookup_parts)]
        if len(lookup_parts) > 1 and not field_parts:
            raise FieldError(
                'Invalid lookup "%s" for model %s".'
                % (lookup, self.get_meta().model.__name__)
            )
        return lookup_parts, field_parts, False

    def check_query_object_type(self, value, opts, field):
        """
        Check whether the object passed while querying is of the correct type.
        If not, raise a ValueError specifying the wrong object.
        """
        if hasattr(value, "_meta"):
            if not check_rel_lookup_compatibility(value._meta.model, opts, field):
                raise ValueError(
                    'Cannot query "%s": Must be "%s" instance.'
                    % (value, opts.object_name)
                )
```
### 97 - django/db/models/sql/query.py:

Start line: 1064, End line: 1095

```python
class Query(BaseExpression):

    def join_parent_model(self, opts, model, alias, seen):
        """
        Make sure the given 'model' is joined in the query. If 'model' isn't
        a parent of 'opts' or if it is None this method is a no-op.

        The 'alias' is the root alias for starting the join, 'seen' is a dict
        of model -> alias of existing joins. It must also contain a mapping
        of None -> some alias. This will be returned in the no-op case.
        """
        if model in seen:
            return seen[model]
        chain = opts.get_base_chain(model)
        if not chain:
            return alias
        curr_opts = opts
        for int_model in chain:
            if int_model in seen:
                curr_opts = int_model._meta
                alias = seen[int_model]
                continue
            # Proxy model have elements in base chain
            # with no parents, assign the new options
            # object and skip to the next base in that
            # case
            if not curr_opts.parents[int_model]:
                curr_opts = int_model._meta
                continue
            link_field = curr_opts.get_ancestor_link(int_model)
            join_info = self.setup_joins([link_field.name], curr_opts, alias)
            curr_opts = int_model._meta
            alias = seen[int_model] = join_info.joins[-1]
        return alias or seen[None]
```
### 114 - django/db/models/sql/query.py:

Start line: 1314, End line: 1334

```python
class Query(BaseExpression):

    def try_transform(self, lhs, name):
        """
        Helper method for build_lookup(). Try to fetch and initialize
        a transform for name parameter from lhs.
        """
        transform_class = lhs.get_transform(name)
        if transform_class:
            return transform_class(lhs)
        else:
            output_field = lhs.output_field.__class__
            suggested_lookups = difflib.get_close_matches(
                name, output_field.get_lookups()
            )
            if suggested_lookups:
                suggestion = ", perhaps you meant %s?" % " or ".join(suggested_lookups)
            else:
                suggestion = "."
            raise FieldError(
                "Unsupported lookup '%s' for %s or join on the field not "
                "permitted%s" % (name, output_field.__name__, suggestion)
            )
```
### 115 - django/db/models/sql/query.py:

Start line: 1567, End line: 1594

```python
class Query(BaseExpression):

    def build_filtered_relation_q(
        self, q_object, reuse, branch_negated=False, current_negated=False
    ):
        """Add a FilteredRelation object to the current filter."""
        connector = q_object.connector
        current_negated ^= q_object.negated
        branch_negated = branch_negated or q_object.negated
        target_clause = WhereNode(connector=connector, negated=q_object.negated)
        for child in q_object.children:
            if isinstance(child, Node):
                child_clause = self.build_filtered_relation_q(
                    child,
                    reuse=reuse,
                    branch_negated=branch_negated,
                    current_negated=current_negated,
                )
            else:
                child_clause, _ = self.build_filter(
                    child,
                    can_reuse=reuse,
                    branch_negated=branch_negated,
                    current_negated=current_negated,
                    allow_joins=True,
                    split_subq=False,
                    reuse_with_filtered_relation=True,
                )
            target_clause.add(child_clause, connector)
        return target_clause
```
### 118 - django/db/models/sql/query.py:

Start line: 1918, End line: 1966

```python
class Query(BaseExpression):

    def resolve_ref(self, name, allow_joins=True, reuse=None, summarize=False):
        annotation = self.annotations.get(name)
        if annotation is not None:
            if not allow_joins:
                for alias in self._gen_col_aliases([annotation]):
                    if isinstance(self.alias_map[alias], Join):
                        raise FieldError(
                            "Joined field references are not permitted in this query"
                        )
            if summarize:
                # Summarize currently means we are doing an aggregate() query
                # which is executed as a wrapped subquery if any of the
                # aggregate() elements reference an existing annotation. In
                # that case we need to return a Ref to the subquery's annotation.
                if name not in self.annotation_select:
                    raise FieldError(
                        "Cannot aggregate over the '%s' alias. Use annotate() "
                        "to promote it." % name
                    )
                return Ref(name, self.annotation_select[name])
            else:
                return annotation
        else:
            field_list = name.split(LOOKUP_SEP)
            annotation = self.annotations.get(field_list[0])
            if annotation is not None:
                for transform in field_list[1:]:
                    annotation = self.try_transform(annotation, transform)
                return annotation
            join_info = self.setup_joins(
                field_list, self.get_meta(), self.get_initial_alias(), can_reuse=reuse
            )
            targets, final_alias, join_list = self.trim_joins(
                join_info.targets, join_info.joins, join_info.path
            )
            if not allow_joins and len(join_list) > 1:
                raise FieldError(
                    "Joined field references are not permitted in this query"
                )
            if len(targets) > 1:
                raise FieldError(
                    "Referencing multicolumn fields with F() objects isn't supported"
                )
            # Verify that the last lookup in name is a field or a transform:
            # transform_function() raises FieldError if not.
            transform = join_info.transform_function(targets[0], final_alias)
            if reuse is not None:
                reuse.update(join_list)
            return transform
```
### 122 - django/db/models/sql/query.py:

Start line: 1835, End line: 1865

```python
class Query(BaseExpression):

    def setup_joins(
        self,
        names,
        opts,
        alias,
        can_reuse=None,
        allow_many=True,
        reuse_with_filtered_relation=False,
    ):
        # ... other code
        for join in path:
            if join.filtered_relation:
                filtered_relation = join.filtered_relation.clone()
                table_alias = filtered_relation.alias
            else:
                filtered_relation = None
                table_alias = None
            opts = join.to_opts
            if join.direct:
                nullable = self.is_nullable(join.join_field)
            else:
                nullable = True
            connection = self.join_class(
                opts.db_table,
                alias,
                table_alias,
                INNER,
                join.join_field,
                nullable,
                filtered_relation=filtered_relation,
            )
            reuse = can_reuse if join.m2m or reuse_with_filtered_relation else None
            alias = self.join(
                connection,
                reuse=reuse,
                reuse_with_filtered_relation=reuse_with_filtered_relation,
            )
            joins.append(alias)
            if filtered_relation:
                filtered_relation.path = joins[:]
        return JoinInfo(final_field, targets, opts, joins, path, final_transformer)
```
### 137 - django/db/models/sql/query.py:

Start line: 1968, End line: 2035

```python
class Query(BaseExpression):

    def split_exclude(self, filter_expr, can_reuse, names_with_path):
        """
        When doing an exclude against any kind of N-to-many relation, we need
        to use a subquery. This method constructs the nested query, given the
        original exclude filter (filter_expr) and the portion up to the first
        N-to-many relation field.

        For example, if the origin filter is ~Q(child__name='foo'), filter_expr
        is ('child__name', 'foo') and can_reuse is a set of joins usable for
        filters in the original query.

        We will turn this into equivalent of:
            WHERE NOT EXISTS(
                SELECT 1
                FROM child
                WHERE name = 'foo' AND child.parent_id = parent.id
                LIMIT 1
            )
        """
        # Generate the inner query.
        query = self.__class__(self.model)
        query._filtered_relations = self._filtered_relations
        filter_lhs, filter_rhs = filter_expr
        if isinstance(filter_rhs, OuterRef):
            filter_rhs = OuterRef(filter_rhs)
        elif isinstance(filter_rhs, F):
            filter_rhs = OuterRef(filter_rhs.name)
        query.add_filter(filter_lhs, filter_rhs)
        query.clear_ordering(force=True)
        # Try to have as simple as possible subquery -> trim leading joins from
        # the subquery.
        trimmed_prefix, contains_louter = query.trim_start(names_with_path)

        col = query.select[0]
        select_field = col.target
        alias = col.alias
        if alias in can_reuse:
            pk = select_field.model._meta.pk
            # Need to add a restriction so that outer query's filters are in effect for
            # the subquery, too.
            query.bump_prefix(self)
            lookup_class = select_field.get_lookup("exact")
            # Note that the query.select[0].alias is different from alias
            # due to bump_prefix above.
            lookup = lookup_class(pk.get_col(query.select[0].alias), pk.get_col(alias))
            query.where.add(lookup, AND)
            query.external_aliases[alias] = True

        lookup_class = select_field.get_lookup("exact")
        lookup = lookup_class(col, ResolvedOuterRef(trimmed_prefix))
        query.where.add(lookup, AND)
        condition, needed_inner = self.build_filter(Exists(query))

        if contains_louter:
            or_null_condition, _ = self.build_filter(
                ("%s__isnull" % trimmed_prefix, True),
                current_negated=True,
                branch_negated=True,
                can_reuse=can_reuse,
            )
            condition.add(or_null_condition, OR)
            # Note that the end result will be:
            # (outercol NOT IN innerq AND outercol IS NOT NULL) OR outercol IS NULL.
            # This might look crazy but due to how IN works, this seems to be
            # correct. If the IS NOT NULL check is removed then outercol NOT
            # IN will return UNKNOWN. If the IS NULL check is removed, then if
            # outercol IS NULL we will not match the row.
        return condition, needed_inner
```
### 139 - django/db/models/sql/query.py:

Start line: 1, End line: 79

```python
"""
Create SQL statements for QuerySets.

The code in here encapsulates all of the SQL construction so that QuerySets
themselves do not have to (and could be backed by things other than SQL
databases). The abstraction barrier only works one way: this module has to know
all about the internals of models in order to get the information it needs.
"""
import copy
import difflib
import functools
import sys
from collections import Counter, namedtuple
from collections.abc import Iterator, Mapping
from itertools import chain, count, product
from string import ascii_uppercase

from django.core.exceptions import FieldDoesNotExist, FieldError
from django.db import DEFAULT_DB_ALIAS, NotSupportedError, connections
from django.db.models.aggregates import Count
from django.db.models.constants import LOOKUP_SEP
from django.db.models.expressions import (
    BaseExpression,
    Col,
    Exists,
    F,
    OuterRef,
    Ref,
    ResolvedOuterRef,
    Value,
)
from django.db.models.fields import Field
from django.db.models.fields.related_lookups import MultiColSource
from django.db.models.lookups import Lookup
from django.db.models.query_utils import (
    Q,
    check_rel_lookup_compatibility,
    refs_expression,
)
from django.db.models.sql.constants import INNER, LOUTER, ORDER_DIR, SINGLE
from django.db.models.sql.datastructures import BaseTable, Empty, Join, MultiJoin
from django.db.models.sql.where import AND, OR, ExtraWhere, NothingNode, WhereNode
from django.utils.functional import cached_property
from django.utils.regex_helper import _lazy_re_compile
from django.utils.tree import Node

__all__ = ["Query", "RawQuery"]

# Quotation marks ('"`[]), whitespace characters, semicolons, or inline
# SQL comments are forbidden in column aliases.
FORBIDDEN_ALIAS_PATTERN = _lazy_re_compile(r"['`\"\]\[;\s]|--|/\*|\*/")

# Inspired from
# https://www.postgresql.org/docs/current/sql-syntax-lexical.html#SQL-SYNTAX-IDENTIFIERS
EXPLAIN_OPTIONS_PATTERN = _lazy_re_compile(r"[\w\-]+")


def get_field_names_from_opts(opts):
    if opts is None:
        return set()
    return set(
        chain.from_iterable(
            (f.name, f.attname) if f.concrete else (f.name,) for f in opts.get_fields()
        )
    )


def get_children_from_q(q):
    for child in q.children:
        if isinstance(child, Node):
            yield from get_children_from_q(child)
        else:
            yield child


JoinInfo = namedtuple(
    "JoinInfo",
    ("final_field", "targets", "opts", "joins", "path", "transform_function"),
)
```
### 145 - django/db/models/sql/query.py:

Start line: 963, End line: 990

```python
class Query(BaseExpression):

    def bump_prefix(self, other_query, exclude=None):
        # ... other code

        if self.alias_prefix != other_query.alias_prefix:
            # No clashes between self and outer query should be possible.
            return

        # Explicitly avoid infinite loop. The constant divider is based on how
        # much depth recursive subquery references add to the stack. This value
        # might need to be adjusted when adding or removing function calls from
        # the code path in charge of performing these operations.
        local_recursion_limit = sys.getrecursionlimit() // 16
        for pos, prefix in enumerate(prefix_gen()):
            if prefix not in self.subq_aliases:
                self.alias_prefix = prefix
                break
            if pos > local_recursion_limit:
                raise RecursionError(
                    "Maximum recursion depth exceeded: too many subqueries."
                )
        self.subq_aliases = self.subq_aliases.union([self.alias_prefix])
        other_query.subq_aliases = other_query.subq_aliases.union(self.subq_aliases)
        if exclude is None:
            exclude = {}
        self.change_aliases(
            {
                alias: "%s%d" % (self.alias_prefix, pos)
                for pos, alias in enumerate(self.alias_map)
                if alias not in exclude
            }
        )
```
