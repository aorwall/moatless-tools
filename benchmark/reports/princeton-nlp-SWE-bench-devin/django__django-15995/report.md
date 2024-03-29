# django__django-15995

| **django/django** | `0701bb8e1f1771b36cdde45602ad377007e372b3` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 1749 |
| **Any found context length** | 330 |
| **Avg pos** | 6.0 |
| **Min pos** | 1 |
| **Max pos** | 5 |
| **Top file pos** | 1 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/db/models/fields/related_descriptors.py b/django/db/models/fields/related_descriptors.py
--- a/django/db/models/fields/related_descriptors.py
+++ b/django/db/models/fields/related_descriptors.py
@@ -647,15 +647,6 @@ def __init__(self, instance):
 
             self.core_filters = {self.field.name: instance}
 
-            # Even if this relation is not to pk, we require still pk value.
-            # The wish is that the instance has been already saved to DB,
-            # although having a pk value isn't a guarantee of that.
-            if self.instance.pk is None:
-                raise ValueError(
-                    f"{instance.__class__.__name__!r} instance needs to have a primary "
-                    f"key value before this relationship can be used."
-                )
-
         def __call__(self, *, manager):
             manager = getattr(self.model, manager)
             manager_class = create_reverse_many_to_one_manager(manager.__class__, rel)
@@ -720,6 +711,14 @@ def _remove_prefetched_objects(self):
                 pass  # nothing to clear from cache
 
         def get_queryset(self):
+            # Even if this relation is not to pk, we require still pk value.
+            # The wish is that the instance has been already saved to DB,
+            # although having a pk value isn't a guarantee of that.
+            if self.instance.pk is None:
+                raise ValueError(
+                    f"{self.instance.__class__.__name__!r} instance needs to have a "
+                    f"primary key value before this relationship can be used."
+                )
             try:
                 return self.instance._prefetched_objects_cache[
                     self.field.remote_field.get_cache_name()

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/db/models/fields/related_descriptors.py | 650 | 658 | 1 | 1 | 330
| django/db/models/fields/related_descriptors.py | 723 | 723 | 5 | 1 | 1749


## Problem Statement

```
Too aggressive pk control in create_reverse_many_to_one_manager
Description
	
In the context of #19580, Django now requires an instance pk to even instanciate a related manager [7ba6ebe9149a].
Now I have a use case where I need to introspect the model used by a related manager (MyModel().related_set.model) and Django 4.1 refuses that with ValueError: 'MyModel' instance needs to have a primary key value before this relationship can be used.
My opinion is that is is too aggressive of a check and would suggest to let the __init__ succeed even if the instance has no pk. Other calls to _check_fk_val in the class seems sufficient to me to safeguard against shooting in the foot.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 django/db/models/fields/related_descriptors.py** | 632 | 672| 330 | 330 | 11242 | 
| 2 | **1 django/db/models/fields/related_descriptors.py** | 789 | 864| 592 | 922 | 11242 | 
| 3 | **1 django/db/models/fields/related_descriptors.py** | 752 | 787| 264 | 1186 | 11242 | 
| 4 | **1 django/db/models/fields/related_descriptors.py** | 866 | 894| 229 | 1415 | 11242 | 
| **-> 5 <-** | **1 django/db/models/fields/related_descriptors.py** | 714 | 750| 334 | 1749 | 11242 | 
| 6 | **1 django/db/models/fields/related_descriptors.py** | 955 | 1021| 588 | 2337 | 11242 | 
| 7 | **1 django/db/models/fields/related_descriptors.py** | 1198 | 1232| 341 | 2678 | 11242 | 
| 8 | **1 django/db/models/fields/related_descriptors.py** | 1166 | 1196| 256 | 2934 | 11242 | 
| 9 | **1 django/db/models/fields/related_descriptors.py** | 1389 | 1439| 403 | 3337 | 11242 | 
| 10 | **1 django/db/models/fields/related_descriptors.py** | 1066 | 1108| 377 | 3714 | 11242 | 
| 11 | **1 django/db/models/fields/related_descriptors.py** | 1318 | 1387| 522 | 4236 | 11242 | 
| 12 | **1 django/db/models/fields/related_descriptors.py** | 1138 | 1164| 198 | 4434 | 11242 | 
| 13 | **1 django/db/models/fields/related_descriptors.py** | 674 | 712| 350 | 4784 | 11242 | 
| 14 | **1 django/db/models/fields/related_descriptors.py** | 1110 | 1136| 220 | 5004 | 11242 | 
| 15 | 2 django/db/models/fields/related.py | 1450 | 1579| 984 | 5988 | 25851 | 
| 16 | **2 django/db/models/fields/related_descriptors.py** | 1266 | 1283| 150 | 6138 | 25851 | 
| 17 | 3 django/db/models/base.py | 1809 | 1844| 242 | 6380 | 44402 | 
| 18 | **3 django/db/models/fields/related_descriptors.py** | 1023 | 1041| 195 | 6575 | 44402 | 
| 19 | **3 django/db/models/fields/related_descriptors.py** | 1043 | 1064| 199 | 6774 | 44402 | 
| 20 | **3 django/db/models/fields/related_descriptors.py** | 1285 | 1316| 344 | 7118 | 44402 | 
| 21 | 3 django/db/models/fields/related.py | 1079 | 1101| 180 | 7298 | 44402 | 
| 22 | 3 django/db/models/fields/related.py | 1581 | 1679| 655 | 7953 | 44402 | 
| 23 | 3 django/db/models/base.py | 1074 | 1126| 516 | 8469 | 44402 | 
| 24 | **3 django/db/models/fields/related_descriptors.py** | 1234 | 1264| 277 | 8746 | 44402 | 
| 25 | 3 django/db/models/base.py | 1560 | 1590| 243 | 8989 | 44402 | 
| 26 | 4 django/contrib/contenttypes/fields.py | 746 | 774| 254 | 9243 | 50033 | 
| 27 | 4 django/db/models/fields/related.py | 1018 | 1054| 261 | 9504 | 50033 | 
| 28 | 4 django/contrib/contenttypes/fields.py | 724 | 744| 188 | 9692 | 50033 | 
| 29 | **4 django/db/models/fields/related_descriptors.py** | 564 | 599| 245 | 9937 | 50033 | 
| 30 | 4 django/db/models/fields/related.py | 1681 | 1731| 431 | 10368 | 50033 | 
| 31 | **4 django/db/models/fields/related_descriptors.py** | 357 | 374| 188 | 10556 | 50033 | 
| 32 | **4 django/db/models/fields/related_descriptors.py** | 395 | 416| 158 | 10714 | 50033 | 
| 33 | 4 django/contrib/contenttypes/fields.py | 659 | 693| 284 | 10998 | 50033 | 
| 34 | 4 django/contrib/contenttypes/fields.py | 695 | 722| 214 | 11212 | 50033 | 
| 35 | 4 django/db/models/fields/related.py | 1160 | 1175| 136 | 11348 | 50033 | 
| 36 | 4 django/db/models/fields/related.py | 1418 | 1448| 172 | 11520 | 50033 | 
| 37 | 4 django/db/models/fields/related.py | 1882 | 1929| 505 | 12025 | 50033 | 
| 38 | 4 django/db/models/fields/related.py | 992 | 1016| 176 | 12201 | 50033 | 
| 39 | 4 django/db/models/fields/related.py | 1262 | 1316| 421 | 12622 | 50033 | 
| 40 | 4 django/db/models/base.py | 1650 | 1684| 248 | 12870 | 50033 | 
| 41 | 4 django/db/models/fields/related.py | 1960 | 1994| 266 | 13136 | 50033 | 
| 42 | 4 django/db/models/fields/related.py | 604 | 670| 497 | 13633 | 50033 | 
| 43 | 4 django/db/models/fields/related.py | 302 | 339| 296 | 13929 | 50033 | 
| 44 | 4 django/contrib/contenttypes/fields.py | 564 | 621| 439 | 14368 | 50033 | 
| 45 | 4 django/db/models/base.py | 1619 | 1648| 205 | 14573 | 50033 | 
| 46 | 5 django/db/backends/base/schema.py | 39 | 72| 214 | 14787 | 63894 | 
| 47 | 6 django/db/models/options.py | 330 | 366| 338 | 15125 | 71482 | 
| 48 | 6 django/contrib/contenttypes/fields.py | 623 | 657| 337 | 15462 | 71482 | 
| 49 | 6 django/db/models/fields/related.py | 154 | 185| 209 | 15671 | 71482 | 
| 50 | **6 django/db/models/fields/related_descriptors.py** | 144 | 188| 418 | 16089 | 71482 | 
| 51 | 6 django/db/models/base.py | 1298 | 1345| 409 | 16498 | 71482 | 
| 52 | **6 django/db/models/fields/related_descriptors.py** | 897 | 952| 411 | 16909 | 71482 | 
| 53 | 6 django/db/models/base.py | 1523 | 1558| 273 | 17182 | 71482 | 
| 54 | 6 django/db/models/base.py | 2450 | 2502| 341 | 17523 | 71482 | 
| 55 | 6 django/db/models/base.py | 1686 | 1703| 156 | 17679 | 71482 | 
| 56 | 6 django/db/models/fields/related.py | 126 | 152| 171 | 17850 | 71482 | 
| 57 | 6 django/db/models/base.py | 477 | 590| 953 | 18803 | 71482 | 
| 58 | 6 django/db/models/fields/related.py | 187 | 206| 155 | 18958 | 71482 | 
| 59 | 7 django/contrib/admin/options.py | 492 | 536| 348 | 19306 | 90702 | 
| 60 | 7 django/db/models/fields/related.py | 1931 | 1958| 305 | 19611 | 90702 | 
| 61 | 7 django/db/models/fields/related.py | 582 | 602| 138 | 19749 | 90702 | 
| 62 | 7 django/db/models/fields/related.py | 1319 | 1416| 574 | 20323 | 90702 | 
| 63 | 7 django/db/models/fields/related.py | 1103 | 1120| 133 | 20456 | 90702 | 
| 64 | 7 django/contrib/contenttypes/fields.py | 535 | 561| 182 | 20638 | 90702 | 
| 65 | 8 django/db/models/manager.py | 1 | 173| 1241 | 21879 | 92150 | 
| 66 | 8 django/db/models/manager.py | 176 | 214| 207 | 22086 | 92150 | 
| 67 | 8 django/db/models/fields/related.py | 901 | 990| 583 | 22669 | 92150 | 
| 68 | 8 django/db/models/fields/related.py | 208 | 224| 142 | 22811 | 92150 | 
| 69 | 8 django/db/models/fields/related.py | 226 | 301| 696 | 23507 | 92150 | 
| 70 | 9 django/contrib/admin/helpers.py | 499 | 527| 240 | 23747 | 95784 | 
| 71 | 9 django/db/models/base.py | 1995 | 2048| 355 | 24102 | 95784 | 
| 72 | 9 django/db/models/fields/related.py | 871 | 898| 237 | 24339 | 95784 | 
| 73 | 10 django/contrib/admin/checks.py | 217 | 264| 333 | 24672 | 105316 | 
| 74 | 10 django/contrib/admin/checks.py | 1244 | 1273| 196 | 24868 | 105316 | 
| 75 | **10 django/db/models/fields/related_descriptors.py** | 418 | 441| 194 | 25062 | 105316 | 
| 76 | 11 django/contrib/sites/managers.py | 1 | 46| 277 | 25339 | 105711 | 
| 77 | **11 django/db/models/fields/related_descriptors.py** | 241 | 323| 760 | 26099 | 105711 | 
| 78 | 11 django/db/models/fields/related.py | 1177 | 1209| 246 | 26345 | 105711 | 
| 79 | 11 django/contrib/admin/options.py | 432 | 490| 509 | 26854 | 105711 | 
| 80 | 12 django/db/models/constraints.py | 307 | 370| 497 | 27351 | 108639 | 
| 81 | 13 django/contrib/auth/models.py | 87 | 133| 309 | 27660 | 112022 | 
| 82 | 13 django/db/models/fields/related.py | 341 | 378| 297 | 27957 | 112022 | 
| 83 | 13 django/db/models/options.py | 440 | 462| 164 | 28121 | 112022 | 
| 84 | 13 django/db/models/base.py | 1193 | 1233| 306 | 28427 | 112022 | 
| 85 | 13 django/contrib/admin/checks.py | 1275 | 1319| 343 | 28770 | 112022 | 
| 86 | 13 django/db/models/base.py | 2256 | 2447| 1302 | 30072 | 112022 | 
| 87 | **13 django/db/models/fields/related_descriptors.py** | 601 | 629| 220 | 30292 | 112022 | 
| 88 | 13 django/db/models/fields/related.py | 757 | 775| 166 | 30458 | 112022 | 
| 89 | 14 django/db/models/fields/related_lookups.py | 75 | 109| 379 | 30837 | 113632 | 
| 90 | **14 django/db/models/fields/related_descriptors.py** | 488 | 561| 648 | 31485 | 113632 | 
| 91 | 15 django/forms/models.py | 1352 | 1387| 225 | 31710 | 125826 | 
| 92 | 16 django/contrib/contenttypes/admin.py | 1 | 88| 585 | 32295 | 126831 | 
| 93 | 16 django/db/models/fields/related.py | 1 | 40| 251 | 32546 | 126831 | 
| 94 | 17 django/db/models/query.py | 684 | 743| 479 | 33025 | 147281 | 
| 95 | 17 django/db/models/options.py | 1 | 57| 347 | 33372 | 147281 | 
| 96 | 17 django/contrib/admin/checks.py | 1040 | 1071| 229 | 33601 | 147281 | 
| 97 | 17 django/db/models/fields/related.py | 707 | 731| 210 | 33811 | 147281 | 
| 98 | 17 django/db/models/base.py | 1705 | 1758| 490 | 34301 | 147281 | 
| 99 | 17 django/db/models/base.py | 1785 | 1807| 171 | 34472 | 147281 | 
| 100 | 18 django/db/models/fields/reverse_related.py | 185 | 203| 167 | 34639 | 149813 | 
| 101 | 18 django/db/models/fields/related.py | 1733 | 1771| 413 | 35052 | 149813 | 
| 102 | 18 django/contrib/contenttypes/fields.py | 363 | 389| 181 | 35233 | 149813 | 
| 103 | 18 django/db/models/base.py | 195 | 247| 458 | 35691 | 149813 | 
| 104 | **18 django/db/models/fields/related_descriptors.py** | 1 | 83| 716 | 36407 | 149813 | 
| 105 | 19 django/core/checks/model_checks.py | 187 | 228| 345 | 36752 | 151623 | 
| 106 | 19 django/contrib/contenttypes/fields.py | 471 | 498| 258 | 37010 | 151623 | 
| 107 | 19 django/db/models/options.py | 368 | 386| 124 | 37134 | 151623 | 
| 108 | 19 django/db/backends/base/schema.py | 1718 | 1755| 303 | 37437 | 151623 | 
| 109 | 19 django/db/models/fields/related_lookups.py | 152 | 169| 216 | 37653 | 151623 | 
| 110 | 19 django/db/models/fields/related.py | 733 | 755| 172 | 37825 | 151623 | 
| 111 | 19 django/db/models/options.py | 493 | 516| 155 | 37980 | 151623 | 
| 112 | 19 django/db/models/fields/related.py | 1212 | 1259| 368 | 38348 | 151623 | 
| 113 | 20 django/db/migrations/autodetector.py | 600 | 776| 1231 | 39579 | 165084 | 
| 114 | 20 django/db/models/base.py | 1427 | 1449| 193 | 39772 | 165084 | 
| 115 | 21 django/db/models/fields/__init__.py | 377 | 410| 214 | 39986 | 183807 | 
| 116 | 21 django/contrib/admin/checks.py | 266 | 282| 138 | 40124 | 183807 | 
| 117 | 21 django/db/models/base.py | 1128 | 1147| 188 | 40312 | 183807 | 
| 118 | 21 django/db/migrations/autodetector.py | 811 | 906| 712 | 41024 | 183807 | 
| 119 | 21 django/db/models/base.py | 1874 | 1902| 188 | 41212 | 183807 | 
| 120 | 21 django/db/models/options.py | 464 | 491| 176 | 41388 | 183807 | 
| 121 | 21 django/db/models/fields/__init__.py | 2667 | 2717| 313 | 41701 | 183807 | 
| 122 | 21 django/db/models/fields/reverse_related.py | 153 | 163| 130 | 41831 | 183807 | 
| 123 | 21 django/db/models/base.py | 1395 | 1425| 216 | 42047 | 183807 | 
| 124 | 21 django/contrib/admin/checks.py | 284 | 312| 218 | 42265 | 183807 | 
| 125 | 21 django/contrib/admin/checks.py | 1231 | 1242| 116 | 42381 | 183807 | 
| 126 | 21 django/db/models/base.py | 1 | 64| 346 | 42727 | 183807 | 
| 127 | **21 django/db/models/fields/related_descriptors.py** | 190 | 239| 452 | 43179 | 183807 | 
| 128 | 21 django/db/backends/base/schema.py | 1497 | 1519| 199 | 43378 | 183807 | 
| 129 | 21 django/db/models/fields/reverse_related.py | 205 | 238| 306 | 43684 | 183807 | 
| 130 | 21 django/forms/models.py | 461 | 491| 244 | 43928 | 183807 | 
| 131 | 21 django/db/models/fields/__init__.py | 1246 | 1283| 245 | 44173 | 183807 | 
| 132 | 22 django/db/models/query_utils.py | 312 | 338| 289 | 44462 | 186600 | 
| 133 | 22 django/db/models/fields/reverse_related.py | 303 | 333| 159 | 44621 | 186600 | 
| 134 | 22 django/db/models/fields/reverse_related.py | 241 | 300| 386 | 45007 | 186600 | 
| 135 | 22 django/db/models/fields/__init__.py | 666 | 715| 342 | 45349 | 186600 | 
| 136 | 22 django/db/models/fields/related.py | 1122 | 1158| 284 | 45633 | 186600 | 
| 137 | 22 django/contrib/admin/options.py | 2376 | 2433| 465 | 46098 | 186600 | 
| 138 | 23 django/db/migrations/state.py | 68 | 90| 214 | 46312 | 194768 | 
| 139 | 23 django/db/models/fields/related.py | 1056 | 1077| 147 | 46459 | 194768 | 
| 140 | 23 django/db/models/base.py | 2157 | 2238| 588 | 47047 | 194768 | 
| 141 | 23 django/db/models/base.py | 1347 | 1376| 290 | 47337 | 194768 | 
| 142 | 23 django/forms/models.py | 790 | 881| 783 | 48120 | 194768 | 
| 143 | 23 django/db/models/fields/reverse_related.py | 1 | 17| 120 | 48240 | 194768 | 
| 144 | 23 django/contrib/contenttypes/fields.py | 22 | 111| 560 | 48800 | 194768 | 
| 145 | 23 django/db/migrations/state.py | 872 | 904| 250 | 49050 | 194768 | 
| 146 | 23 django/contrib/admin/options.py | 2435 | 2462| 254 | 49304 | 194768 | 
| 147 | 24 django/db/backends/mysql/schema.py | 120 | 136| 144 | 49448 | 196391 | 
| 148 | 24 django/db/models/base.py | 1845 | 1872| 187 | 49635 | 196391 | 
| 149 | 24 django/db/models/constraints.py | 233 | 249| 139 | 49774 | 196391 | 
| 150 | 24 django/db/models/base.py | 248 | 365| 874 | 50648 | 196391 | 
| 151 | 24 django/db/models/fields/__init__.py | 2223 | 2248| 206 | 50854 | 196391 | 
| 152 | 24 django/contrib/auth/models.py | 158 | 172| 164 | 51018 | 196391 | 
| 153 | 24 django/contrib/admin/options.py | 1884 | 1895| 149 | 51167 | 196391 | 
| 154 | 24 django/db/models/query.py | 2604 | 2627| 201 | 51368 | 196391 | 
| 155 | **24 django/db/models/fields/related_descriptors.py** | 104 | 142| 266 | 51634 | 196391 | 


### Hint

```
I think this worth doing, having get_queryset perform self._check_fk_val() should catch the currently tested .all() case as well. In the mean time I think you should be able to use MyModel._meta.get_field("related_set") for your introspection needs.
Replying to Simon Charette: ... In the mean time I think you should be able to use MyModel._meta.get_field("related_set") for your introspection needs. Thanks for the tip! I just needed to remove the _set as it's not part of the ManyToOneRel field name in _meta.
```

## Patch

```diff
diff --git a/django/db/models/fields/related_descriptors.py b/django/db/models/fields/related_descriptors.py
--- a/django/db/models/fields/related_descriptors.py
+++ b/django/db/models/fields/related_descriptors.py
@@ -647,15 +647,6 @@ def __init__(self, instance):
 
             self.core_filters = {self.field.name: instance}
 
-            # Even if this relation is not to pk, we require still pk value.
-            # The wish is that the instance has been already saved to DB,
-            # although having a pk value isn't a guarantee of that.
-            if self.instance.pk is None:
-                raise ValueError(
-                    f"{instance.__class__.__name__!r} instance needs to have a primary "
-                    f"key value before this relationship can be used."
-                )
-
         def __call__(self, *, manager):
             manager = getattr(self.model, manager)
             manager_class = create_reverse_many_to_one_manager(manager.__class__, rel)
@@ -720,6 +711,14 @@ def _remove_prefetched_objects(self):
                 pass  # nothing to clear from cache
 
         def get_queryset(self):
+            # Even if this relation is not to pk, we require still pk value.
+            # The wish is that the instance has been already saved to DB,
+            # although having a pk value isn't a guarantee of that.
+            if self.instance.pk is None:
+                raise ValueError(
+                    f"{self.instance.__class__.__name__!r} instance needs to have a "
+                    f"primary key value before this relationship can be used."
+                )
             try:
                 return self.instance._prefetched_objects_cache[
                     self.field.remote_field.get_cache_name()

```

## Test Patch

```diff
diff --git a/tests/many_to_one/tests.py b/tests/many_to_one/tests.py
--- a/tests/many_to_one/tests.py
+++ b/tests/many_to_one/tests.py
@@ -758,6 +758,9 @@ def test_relation_unsaved(self):
         )
         with self.assertRaisesMessage(ValueError, msg):
             th.child_set.count()
+        # The reverse foreign key manager can be created.
+        self.assertEqual(th.child_set.model, Third)
+
         th.save()
         # Now the model is saved, so we will need to execute a query.
         with self.assertNumQueries(1):

```


## Code snippets

### 1 - django/db/models/fields/related_descriptors.py:

Start line: 632, End line: 672

```python
def create_reverse_many_to_one_manager(superclass, rel):
    """
    Create a manager for the reverse side of a many-to-one relation.

    This manager subclasses another manager, generally the default manager of
    the related model, and adds behaviors specific to many-to-one relations.
    """

    class RelatedManager(superclass):
        def __init__(self, instance):
            super().__init__()

            self.instance = instance
            self.model = rel.related_model
            self.field = rel.field

            self.core_filters = {self.field.name: instance}

            # Even if this relation is not to pk, we require still pk value.
            # The wish is that the instance has been already saved to DB,
            # although having a pk value isn't a guarantee of that.
            if self.instance.pk is None:
                raise ValueError(
                    f"{instance.__class__.__name__!r} instance needs to have a primary "
                    f"key value before this relationship can be used."
                )

        def __call__(self, *, manager):
            manager = getattr(self.model, manager)
            manager_class = create_reverse_many_to_one_manager(manager.__class__, rel)
            return manager_class(self.instance)

        do_not_call_in_templates = True

        def _check_fk_val(self):
            for field in self.field.foreign_related_fields:
                if getattr(self.instance, field.attname) is None:
                    raise ValueError(
                        f'"{self.instance!r}" needs to have a value for field '
                        f'"{field.attname}" before this relationship can be used.'
                    )
    # ... other code
```
### 2 - django/db/models/fields/related_descriptors.py:

Start line: 789, End line: 864

```python
def create_reverse_many_to_one_manager(superclass, rel):

    class RelatedManager(superclass):

        add.alters_data = True

        def create(self, **kwargs):
            self._check_fk_val()
            kwargs[self.field.name] = self.instance
            db = router.db_for_write(self.model, instance=self.instance)
            return super(RelatedManager, self.db_manager(db)).create(**kwargs)

        create.alters_data = True

        def get_or_create(self, **kwargs):
            self._check_fk_val()
            kwargs[self.field.name] = self.instance
            db = router.db_for_write(self.model, instance=self.instance)
            return super(RelatedManager, self.db_manager(db)).get_or_create(**kwargs)

        get_or_create.alters_data = True

        def update_or_create(self, **kwargs):
            self._check_fk_val()
            kwargs[self.field.name] = self.instance
            db = router.db_for_write(self.model, instance=self.instance)
            return super(RelatedManager, self.db_manager(db)).update_or_create(**kwargs)

        update_or_create.alters_data = True

        # remove() and clear() are only provided if the ForeignKey can have a
        # value of null.
        if rel.field.null:

            def remove(self, *objs, bulk=True):
                if not objs:
                    return
                self._check_fk_val()
                val = self.field.get_foreign_related_value(self.instance)
                old_ids = set()
                for obj in objs:
                    if not isinstance(obj, self.model):
                        raise TypeError(
                            "'%s' instance expected, got %r"
                            % (
                                self.model._meta.object_name,
                                obj,
                            )
                        )
                    # Is obj actually part of this descriptor set?
                    if self.field.get_local_related_value(obj) == val:
                        old_ids.add(obj.pk)
                    else:
                        raise self.field.remote_field.model.DoesNotExist(
                            "%r is not related to %r." % (obj, self.instance)
                        )
                self._clear(self.filter(pk__in=old_ids), bulk)

            remove.alters_data = True

            def clear(self, *, bulk=True):
                self._check_fk_val()
                self._clear(self, bulk)

            clear.alters_data = True

            def _clear(self, queryset, bulk):
                self._remove_prefetched_objects()
                db = router.db_for_write(self.model, instance=self.instance)
                queryset = queryset.using(db)
                if bulk:
                    # `QuerySet.update()` is intrinsically atomic.
                    queryset.update(**{self.field.name: None})
                else:
                    with transaction.atomic(using=db, savepoint=False):
                        for obj in queryset:
                            setattr(obj, self.field.name, None)
                            obj.save(update_fields=[self.field.name])

            _clear.alters_data = True
    # ... other code
```
### 3 - django/db/models/fields/related_descriptors.py:

Start line: 752, End line: 787

```python
def create_reverse_many_to_one_manager(superclass, rel):

    class RelatedManager(superclass):

        def add(self, *objs, bulk=True):
            self._check_fk_val()
            self._remove_prefetched_objects()
            db = router.db_for_write(self.model, instance=self.instance)

            def check_and_update_obj(obj):
                if not isinstance(obj, self.model):
                    raise TypeError(
                        "'%s' instance expected, got %r"
                        % (
                            self.model._meta.object_name,
                            obj,
                        )
                    )
                setattr(obj, self.field.name, self.instance)

            if bulk:
                pks = []
                for obj in objs:
                    check_and_update_obj(obj)
                    if obj._state.adding or obj._state.db != db:
                        raise ValueError(
                            "%r instance isn't saved. Use bulk=False or save "
                            "the object first." % obj
                        )
                    pks.append(obj.pk)
                self.model._base_manager.using(db).filter(pk__in=pks).update(
                    **{
                        self.field.name: self.instance,
                    }
                )
            else:
                with transaction.atomic(using=db, savepoint=False):
                    for obj in objs:
                        check_and_update_obj(obj)
                        obj.save()
    # ... other code
```
### 4 - django/db/models/fields/related_descriptors.py:

Start line: 866, End line: 894

```python
def create_reverse_many_to_one_manager(superclass, rel):

    class RelatedManager(superclass):

        def set(self, objs, *, bulk=True, clear=False):
            self._check_fk_val()
            # Force evaluation of `objs` in case it's a queryset whose value
            # could be affected by `manager.clear()`. Refs #19816.
            objs = tuple(objs)

            if self.field.null:
                db = router.db_for_write(self.model, instance=self.instance)
                with transaction.atomic(using=db, savepoint=False):
                    if clear:
                        self.clear(bulk=bulk)
                        self.add(*objs, bulk=bulk)
                    else:
                        old_objs = set(self.using(db).all())
                        new_objs = []
                        for obj in objs:
                            if obj in old_objs:
                                old_objs.remove(obj)
                            else:
                                new_objs.append(obj)

                        self.remove(*old_objs, bulk=bulk)
                        self.add(*new_objs, bulk=bulk)
            else:
                self.add(*objs, bulk=bulk)

        set.alters_data = True

    return RelatedManager
```
### 5 - django/db/models/fields/related_descriptors.py:

Start line: 714, End line: 750

```python
def create_reverse_many_to_one_manager(superclass, rel):

    class RelatedManager(superclass):

        def _remove_prefetched_objects(self):
            try:
                self.instance._prefetched_objects_cache.pop(
                    self.field.remote_field.get_cache_name()
                )
            except (AttributeError, KeyError):
                pass  # nothing to clear from cache

        def get_queryset(self):
            try:
                return self.instance._prefetched_objects_cache[
                    self.field.remote_field.get_cache_name()
                ]
            except (AttributeError, KeyError):
                queryset = super().get_queryset()
                return self._apply_rel_filters(queryset)

        def get_prefetch_queryset(self, instances, queryset=None):
            if queryset is None:
                queryset = super().get_queryset()

            queryset._add_hints(instance=instances[0])
            queryset = queryset.using(queryset._db or self._db)

            rel_obj_attr = self.field.get_local_related_value
            instance_attr = self.field.get_foreign_related_value
            instances_dict = {instance_attr(inst): inst for inst in instances}
            queryset = _filter_prefetch_queryset(queryset, self.field.name, instances)

            # Since we just bypassed this class' get_queryset(), we must manage
            # the reverse relation manually.
            for rel_obj in queryset:
                if not self.field.is_cached(rel_obj):
                    instance = instances_dict[rel_obj_attr(rel_obj)]
                    setattr(rel_obj, self.field.name, instance)
            cache_name = self.field.remote_field.get_cache_name()
            return queryset, rel_obj_attr, instance_attr, False, cache_name, False
    # ... other code
```
### 6 - django/db/models/fields/related_descriptors.py:

Start line: 955, End line: 1021

```python
def create_forward_many_to_many_manager(superclass, rel, reverse):
    """
    Create a manager for the either side of a many-to-many relation.

    This manager subclasses another manager, generally the default manager of
    the related model, and adds behaviors specific to many-to-many relations.
    """

    class ManyRelatedManager(superclass):
        def __init__(self, instance=None):
            super().__init__()

            self.instance = instance

            if not reverse:
                self.model = rel.model
                self.query_field_name = rel.field.related_query_name()
                self.prefetch_cache_name = rel.field.name
                self.source_field_name = rel.field.m2m_field_name()
                self.target_field_name = rel.field.m2m_reverse_field_name()
                self.symmetrical = rel.symmetrical
            else:
                self.model = rel.related_model
                self.query_field_name = rel.field.name
                self.prefetch_cache_name = rel.field.related_query_name()
                self.source_field_name = rel.field.m2m_reverse_field_name()
                self.target_field_name = rel.field.m2m_field_name()
                self.symmetrical = False

            self.through = rel.through
            self.reverse = reverse

            self.source_field = self.through._meta.get_field(self.source_field_name)
            self.target_field = self.through._meta.get_field(self.target_field_name)

            self.core_filters = {}
            self.pk_field_names = {}
            for lh_field, rh_field in self.source_field.related_fields:
                core_filter_key = "%s__%s" % (self.query_field_name, rh_field.name)
                self.core_filters[core_filter_key] = getattr(instance, rh_field.attname)
                self.pk_field_names[lh_field.name] = rh_field.name

            self.related_val = self.source_field.get_foreign_related_value(instance)
            if None in self.related_val:
                raise ValueError(
                    '"%r" needs to have a value for field "%s" before '
                    "this many-to-many relationship can be used."
                    % (instance, self.pk_field_names[self.source_field_name])
                )
            # Even if this relation is not to pk, we require still pk value.
            # The wish is that the instance has been already saved to DB,
            # although having a pk value isn't a guarantee of that.
            if instance.pk is None:
                raise ValueError(
                    "%r instance needs to have a primary key value before "
                    "a many-to-many relationship can be used."
                    % instance.__class__.__name__
                )

        def __call__(self, *, manager):
            manager = getattr(self.model, manager)
            manager_class = create_forward_many_to_many_manager(
                manager.__class__, rel, reverse
            )
            return manager_class(instance=self.instance)

        do_not_call_in_templates = True
    # ... other code
```
### 7 - django/db/models/fields/related_descriptors.py:

Start line: 1198, End line: 1232

```python
def create_forward_many_to_many_manager(superclass, rel, reverse):

    class ManyRelatedManager(superclass):

        set.alters_data = True

        def create(self, *, through_defaults=None, **kwargs):
            db = router.db_for_write(self.instance.__class__, instance=self.instance)
            new_obj = super(ManyRelatedManager, self.db_manager(db)).create(**kwargs)
            self.add(new_obj, through_defaults=through_defaults)
            return new_obj

        create.alters_data = True

        def get_or_create(self, *, through_defaults=None, **kwargs):
            db = router.db_for_write(self.instance.__class__, instance=self.instance)
            obj, created = super(ManyRelatedManager, self.db_manager(db)).get_or_create(
                **kwargs
            )
            # We only need to add() if created because if we got an object back
            # from get() then the relationship already exists.
            if created:
                self.add(obj, through_defaults=through_defaults)
            return obj, created

        get_or_create.alters_data = True

        def update_or_create(self, *, through_defaults=None, **kwargs):
            db = router.db_for_write(self.instance.__class__, instance=self.instance)
            obj, created = super(
                ManyRelatedManager, self.db_manager(db)
            ).update_or_create(**kwargs)
            # We only need to add() if created because if we got an object back
            # from get() then the relationship already exists.
            if created:
                self.add(obj, through_defaults=through_defaults)
            return obj, created

        update_or_create.alters_data = True
    # ... other code
```
### 8 - django/db/models/fields/related_descriptors.py:

Start line: 1166, End line: 1196

```python
def create_forward_many_to_many_manager(superclass, rel, reverse):

    class ManyRelatedManager(superclass):

        def set(self, objs, *, clear=False, through_defaults=None):
            # Force evaluation of `objs` in case it's a queryset whose value
            # could be affected by `manager.clear()`. Refs #19816.
            objs = tuple(objs)

            db = router.db_for_write(self.through, instance=self.instance)
            with transaction.atomic(using=db, savepoint=False):
                if clear:
                    self.clear()
                    self.add(*objs, through_defaults=through_defaults)
                else:
                    old_ids = set(
                        self.using(db).values_list(
                            self.target_field.target_field.attname, flat=True
                        )
                    )

                    new_objs = []
                    for obj in objs:
                        fk_val = (
                            self.target_field.get_foreign_related_value(obj)[0]
                            if isinstance(obj, self.model)
                            else self.target_field.get_prep_value(obj)
                        )
                        if fk_val in old_ids:
                            old_ids.remove(fk_val)
                        else:
                            new_objs.append(obj)

                    self.remove(*old_ids)
                    self.add(*new_objs, through_defaults=through_defaults)
    # ... other code
```
### 9 - django/db/models/fields/related_descriptors.py:

Start line: 1389, End line: 1439

```python
def create_forward_many_to_many_manager(superclass, rel, reverse):

    class ManyRelatedManager(superclass):

        def _remove_items(self, source_field_name, target_field_name, *objs):
            # source_field_name: the PK colname in join table for the source object
            # target_field_name: the PK colname in join table for the target object
            # *objs - objects to remove. Either object instances, or primary
            # keys of object instances.
            if not objs:
                return

            # Check that all the objects are of the right type
            old_ids = set()
            for obj in objs:
                if isinstance(obj, self.model):
                    fk_val = self.target_field.get_foreign_related_value(obj)[0]
                    old_ids.add(fk_val)
                else:
                    old_ids.add(obj)

            db = router.db_for_write(self.through, instance=self.instance)
            with transaction.atomic(using=db, savepoint=False):
                # Send a signal to the other end if need be.
                signals.m2m_changed.send(
                    sender=self.through,
                    action="pre_remove",
                    instance=self.instance,
                    reverse=self.reverse,
                    model=self.model,
                    pk_set=old_ids,
                    using=db,
                )
                target_model_qs = super().get_queryset()
                if target_model_qs._has_filters():
                    old_vals = target_model_qs.using(db).filter(
                        **{"%s__in" % self.target_field.target_field.attname: old_ids}
                    )
                else:
                    old_vals = old_ids
                filters = self._build_remove_filters(old_vals)
                self.through._default_manager.using(db).filter(filters).delete()

                signals.m2m_changed.send(
                    sender=self.through,
                    action="post_remove",
                    instance=self.instance,
                    reverse=self.reverse,
                    model=self.model,
                    pk_set=old_ids,
                    using=db,
                )

    return ManyRelatedManager
```
### 10 - django/db/models/fields/related_descriptors.py:

Start line: 1066, End line: 1108

```python
def create_forward_many_to_many_manager(superclass, rel, reverse):

    class ManyRelatedManager(superclass):

        def get_prefetch_queryset(self, instances, queryset=None):
            if queryset is None:
                queryset = super().get_queryset()

            queryset._add_hints(instance=instances[0])
            queryset = queryset.using(queryset._db or self._db)
            queryset = _filter_prefetch_queryset(
                queryset._next_is_sticky(), self.query_field_name, instances
            )

            # M2M: need to annotate the query in order to get the primary model
            # that the secondary model was actually related to. We know that
            # there will already be a join on the join table, so we can just add
            # the select.

            # For non-autocreated 'through' models, can't assume we are
            # dealing with PK values.
            fk = self.through._meta.get_field(self.source_field_name)
            join_table = fk.model._meta.db_table
            connection = connections[queryset.db]
            qn = connection.ops.quote_name
            queryset = queryset.extra(
                select={
                    "_prefetch_related_val_%s"
                    % f.attname: "%s.%s"
                    % (qn(join_table), qn(f.column))
                    for f in fk.local_related_fields
                }
            )
            return (
                queryset,
                lambda result: tuple(
                    getattr(result, "_prefetch_related_val_%s" % f.attname)
                    for f in fk.local_related_fields
                ),
                lambda inst: tuple(
                    f.get_db_prep_value(getattr(inst, f.attname), connection)
                    for f in fk.foreign_related_fields
                ),
                False,
                self.prefetch_cache_name,
                False,
            )
    # ... other code
```
### 11 - django/db/models/fields/related_descriptors.py:

Start line: 1318, End line: 1387

```python
def create_forward_many_to_many_manager(superclass, rel, reverse):

    class ManyRelatedManager(superclass):

        def _add_items(
            self, source_field_name, target_field_name, *objs, through_defaults=None
        ):
            # source_field_name: the PK fieldname in join table for the source object
            # target_field_name: the PK fieldname in join table for the target object
            # *objs - objects to add. Either object instances, or primary keys
            # of object instances.
            if not objs:
                return

            through_defaults = dict(resolve_callables(through_defaults or {}))
            target_ids = self._get_target_ids(target_field_name, objs)
            db = router.db_for_write(self.through, instance=self.instance)
            can_ignore_conflicts, must_send_signals, can_fast_add = self._get_add_plan(
                db, source_field_name
            )
            if can_fast_add:
                self.through._default_manager.using(db).bulk_create(
                    [
                        self.through(
                            **{
                                "%s_id" % source_field_name: self.related_val[0],
                                "%s_id" % target_field_name: target_id,
                            }
                        )
                        for target_id in target_ids
                    ],
                    ignore_conflicts=True,
                )
                return

            missing_target_ids = self._get_missing_target_ids(
                source_field_name, target_field_name, db, target_ids
            )
            with transaction.atomic(using=db, savepoint=False):
                if must_send_signals:
                    signals.m2m_changed.send(
                        sender=self.through,
                        action="pre_add",
                        instance=self.instance,
                        reverse=self.reverse,
                        model=self.model,
                        pk_set=missing_target_ids,
                        using=db,
                    )
                # Add the ones that aren't there already.
                self.through._default_manager.using(db).bulk_create(
                    [
                        self.through(
                            **through_defaults,
                            **{
                                "%s_id" % source_field_name: self.related_val[0],
                                "%s_id" % target_field_name: target_id,
                            },
                        )
                        for target_id in missing_target_ids
                    ],
                    ignore_conflicts=can_ignore_conflicts,
                )

                if must_send_signals:
                    signals.m2m_changed.send(
                        sender=self.through,
                        action="post_add",
                        instance=self.instance,
                        reverse=self.reverse,
                        model=self.model,
                        pk_set=missing_target_ids,
                        using=db,
                    )
    # ... other code
```
### 12 - django/db/models/fields/related_descriptors.py:

Start line: 1138, End line: 1164

```python
def create_forward_many_to_many_manager(superclass, rel, reverse):

    class ManyRelatedManager(superclass):

        def clear(self):
            db = router.db_for_write(self.through, instance=self.instance)
            with transaction.atomic(using=db, savepoint=False):
                signals.m2m_changed.send(
                    sender=self.through,
                    action="pre_clear",
                    instance=self.instance,
                    reverse=self.reverse,
                    model=self.model,
                    pk_set=None,
                    using=db,
                )
                self._remove_prefetched_objects()
                filters = self._build_remove_filters(super().get_queryset().using(db))
                self.through._default_manager.using(db).filter(filters).delete()

                signals.m2m_changed.send(
                    sender=self.through,
                    action="post_clear",
                    instance=self.instance,
                    reverse=self.reverse,
                    model=self.model,
                    pk_set=None,
                    using=db,
                )

        clear.alters_data = True
    # ... other code
```
### 13 - django/db/models/fields/related_descriptors.py:

Start line: 674, End line: 712

```python
def create_reverse_many_to_one_manager(superclass, rel):

    class RelatedManager(superclass):

        def _apply_rel_filters(self, queryset):
            """
            Filter the queryset for the instance this manager is bound to.
            """
            db = self._db or router.db_for_read(self.model, instance=self.instance)
            empty_strings_as_null = connections[
                db
            ].features.interprets_empty_strings_as_nulls
            queryset._add_hints(instance=self.instance)
            if self._db:
                queryset = queryset.using(self._db)
            queryset._defer_next_filter = True
            queryset = queryset.filter(**self.core_filters)
            for field in self.field.foreign_related_fields:
                val = getattr(self.instance, field.attname)
                if val is None or (val == "" and empty_strings_as_null):
                    return queryset.none()
            if self.field.many_to_one:
                # Guard against field-like objects such as GenericRelation
                # that abuse create_reverse_many_to_one_manager() with reverse
                # one-to-many relationships instead and break known related
                # objects assignment.
                try:
                    target_field = self.field.target_field
                except FieldError:
                    # The relationship has multiple target fields. Use a tuple
                    # for related object id.
                    rel_obj_id = tuple(
                        [
                            getattr(self.instance, target_field.attname)
                            for target_field in self.field.path_infos[-1].target_fields
                        ]
                    )
                else:
                    rel_obj_id = getattr(self.instance, target_field.attname)
                queryset._known_related_objects = {
                    self.field: {rel_obj_id: self.instance}
                }
            return queryset
    # ... other code
```
### 14 - django/db/models/fields/related_descriptors.py:

Start line: 1110, End line: 1136

```python
def create_forward_many_to_many_manager(superclass, rel, reverse):

    class ManyRelatedManager(superclass):

        def add(self, *objs, through_defaults=None):
            self._remove_prefetched_objects()
            db = router.db_for_write(self.through, instance=self.instance)
            with transaction.atomic(using=db, savepoint=False):
                self._add_items(
                    self.source_field_name,
                    self.target_field_name,
                    *objs,
                    through_defaults=through_defaults,
                )
                # If this is a symmetrical m2m relation to self, add the mirror
                # entry in the m2m table.
                if self.symmetrical:
                    self._add_items(
                        self.target_field_name,
                        self.source_field_name,
                        *objs,
                        through_defaults=through_defaults,
                    )

        add.alters_data = True

        def remove(self, *objs):
            self._remove_prefetched_objects()
            self._remove_items(self.source_field_name, self.target_field_name, *objs)

        remove.alters_data = True
    # ... other code
```
### 16 - django/db/models/fields/related_descriptors.py:

Start line: 1266, End line: 1283

```python
def create_forward_many_to_many_manager(superclass, rel, reverse):

    class ManyRelatedManager(superclass):

        def _get_missing_target_ids(
            self, source_field_name, target_field_name, db, target_ids
        ):
            """
            Return the subset of ids of `objs` that aren't already assigned to
            this relationship.
            """
            vals = (
                self.through._default_manager.using(db)
                .values_list(target_field_name, flat=True)
                .filter(
                    **{
                        source_field_name: self.related_val[0],
                        "%s__in" % target_field_name: target_ids,
                    }
                )
            )
            return target_ids.difference(vals)
    # ... other code
```
### 18 - django/db/models/fields/related_descriptors.py:

Start line: 1023, End line: 1041

```python
def create_forward_many_to_many_manager(superclass, rel, reverse):

    class ManyRelatedManager(superclass):

        def _build_remove_filters(self, removed_vals):
            filters = Q.create([(self.source_field_name, self.related_val)])
            # No need to add a subquery condition if removed_vals is a QuerySet without
            # filters.
            removed_vals_filters = (
                not isinstance(removed_vals, QuerySet) or removed_vals._has_filters()
            )
            if removed_vals_filters:
                filters &= Q.create([(f"{self.target_field_name}__in", removed_vals)])
            if self.symmetrical:
                symmetrical_filters = Q.create(
                    [(self.target_field_name, self.related_val)]
                )
                if removed_vals_filters:
                    symmetrical_filters &= Q.create(
                        [(f"{self.source_field_name}__in", removed_vals)]
                    )
                filters |= symmetrical_filters
            return filters
    # ... other code
```
### 19 - django/db/models/fields/related_descriptors.py:

Start line: 1043, End line: 1064

```python
def create_forward_many_to_many_manager(superclass, rel, reverse):

    class ManyRelatedManager(superclass):

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
### 20 - django/db/models/fields/related_descriptors.py:

Start line: 1285, End line: 1316

```python
def create_forward_many_to_many_manager(superclass, rel, reverse):

    class ManyRelatedManager(superclass):

        def _get_add_plan(self, db, source_field_name):
            """
            Return a boolean triple of the way the add should be performed.

            The first element is whether or not bulk_create(ignore_conflicts)
            can be used, the second whether or not signals must be sent, and
            the third element is whether or not the immediate bulk insertion
            with conflicts ignored can be performed.
            """
            # Conflicts can be ignored when the intermediary model is
            # auto-created as the only possible collision is on the
            # (source_id, target_id) tuple. The same assertion doesn't hold for
            # user-defined intermediary models as they could have other fields
            # causing conflicts which must be surfaced.
            can_ignore_conflicts = (
                self.through._meta.auto_created is not False
                and connections[db].features.supports_ignore_conflicts
            )
            # Don't send the signal when inserting duplicate data row
            # for symmetrical reverse entries.
            must_send_signals = (
                self.reverse or source_field_name == self.source_field_name
            ) and (signals.m2m_changed.has_listeners(self.through))
            # Fast addition through bulk insertion can only be performed
            # if no m2m_changed listeners are connected for self.through
            # as they require the added set of ids to be provided via
            # pk_set.
            return (
                can_ignore_conflicts,
                must_send_signals,
                (can_ignore_conflicts and not must_send_signals),
            )
    # ... other code
```
### 24 - django/db/models/fields/related_descriptors.py:

Start line: 1234, End line: 1264

```python
def create_forward_many_to_many_manager(superclass, rel, reverse):

    class ManyRelatedManager(superclass):

        def _get_target_ids(self, target_field_name, objs):
            """
            Return the set of ids of `objs` that the target field references.
            """
            from django.db.models import Model

            target_ids = set()
            target_field = self.through._meta.get_field(target_field_name)
            for obj in objs:
                if isinstance(obj, self.model):
                    if not router.allow_relation(obj, self.instance):
                        raise ValueError(
                            'Cannot add "%r": instance is on database "%s", '
                            'value is on database "%s"'
                            % (obj, self.instance._state.db, obj._state.db)
                        )
                    target_id = target_field.get_foreign_related_value(obj)[0]
                    if target_id is None:
                        raise ValueError(
                            'Cannot add "%r": the value for field "%s" is None'
                            % (obj, target_field_name)
                        )
                    target_ids.add(target_id)
                elif isinstance(obj, Model):
                    raise TypeError(
                        "'%s' instance expected, got %r"
                        % (self.model._meta.object_name, obj)
                    )
                else:
                    target_ids.add(target_field.get_prep_value(obj))
            return target_ids
    # ... other code
```
### 29 - django/db/models/fields/related_descriptors.py:

Start line: 564, End line: 599

```python
class ReverseManyToOneDescriptor:
    """
    Accessor to the related objects manager on the reverse side of a
    many-to-one relation.

    In the example::

        class Child(Model):
            parent = ForeignKey(Parent, related_name='children')

    ``Parent.children`` is a ``ReverseManyToOneDescriptor`` instance.

    Most of the implementation is delegated to a dynamically defined manager
    class built by ``create_forward_many_to_many_manager()`` defined below.
    """

    def __init__(self, rel):
        self.rel = rel
        self.field = rel.field

    @cached_property
    def related_manager_cache_key(self):
        # Being able to access the manager instance precludes it from being
        # hidden. The rel's accessor name is used to allow multiple managers
        # to the same model to coexist. e.g. post.attached_comment_set and
        # post.attached_link_set are separately cached.
        return self.rel.get_cache_name()

    @cached_property
    def related_manager_cls(self):
        related_model = self.rel.related_model

        return create_reverse_many_to_one_manager(
            related_model._default_manager.__class__,
            self.rel,
        )
```
### 31 - django/db/models/fields/related_descriptors.py:

Start line: 357, End line: 374

```python
class ForwardOneToOneDescriptor(ForwardManyToOneDescriptor):

    def __set__(self, instance, value):
        super().__set__(instance, value)
        # If the primary key is a link to a parent model and a parent instance
        # is being set, update the value of the inherited pk(s).
        if self.field.primary_key and self.field.remote_field.parent_link:
            opts = instance._meta
            # Inherited primary key fields from this object's base classes.
            inherited_pk_fields = [
                field
                for field in opts.concrete_fields
                if field.primary_key and field.remote_field
            ]
            for field in inherited_pk_fields:
                rel_model_pk_name = field.remote_field.model._meta.pk.attname
                raw_value = (
                    getattr(value, rel_model_pk_name) if value is not None else None
                )
                setattr(instance, rel_model_pk_name, raw_value)
```
### 32 - django/db/models/fields/related_descriptors.py:

Start line: 395, End line: 416

```python
class ReverseOneToOneDescriptor:

    @cached_property
    def RelatedObjectDoesNotExist(self):
        # The exception isn't created at initialization time for the sake of
        # consistency with `ForwardManyToOneDescriptor`.
        return type(
            "RelatedObjectDoesNotExist",
            (self.related.related_model.DoesNotExist, AttributeError),
            {
                "__module__": self.related.model.__module__,
                "__qualname__": "%s.%s.RelatedObjectDoesNotExist"
                % (
                    self.related.model.__qualname__,
                    self.related.name,
                ),
            },
        )

    def is_cached(self, instance):
        return self.related.is_cached(instance)

    def get_queryset(self, **hints):
        return self.related.related_model._base_manager.db_manager(hints=hints).all()
```
### 50 - django/db/models/fields/related_descriptors.py:

Start line: 144, End line: 188

```python
class ForwardManyToOneDescriptor:

    def get_prefetch_queryset(self, instances, queryset=None):
        if queryset is None:
            queryset = self.get_queryset()
        queryset._add_hints(instance=instances[0])

        rel_obj_attr = self.field.get_foreign_related_value
        instance_attr = self.field.get_local_related_value
        instances_dict = {instance_attr(inst): inst for inst in instances}
        related_field = self.field.foreign_related_fields[0]
        remote_field = self.field.remote_field

        # FIXME: This will need to be revisited when we introduce support for
        # composite fields. In the meantime we take this practical approach to
        # solve a regression on 1.6 when the reverse manager in hidden
        # (related_name ends with a '+'). Refs #21410.
        # The check for len(...) == 1 is a special case that allows the query
        # to be join-less and smaller. Refs #21760.
        if remote_field.is_hidden() or len(self.field.foreign_related_fields) == 1:
            query = {
                "%s__in"
                % related_field.name: {instance_attr(inst)[0] for inst in instances}
            }
        else:
            query = {"%s__in" % self.field.related_query_name(): instances}
        queryset = queryset.filter(**query)

        # Since we're going to assign directly in the cache,
        # we must manage the reverse relation cache manually.
        if not remote_field.multiple:
            for rel_obj in queryset:
                instance = instances_dict[rel_obj_attr(rel_obj)]
                remote_field.set_cached_value(rel_obj, instance)
        return (
            queryset,
            rel_obj_attr,
            instance_attr,
            True,
            self.field.get_cache_name(),
            False,
        )

    def get_object(self, instance):
        qs = self.get_queryset(instance=instance)
        # Assuming the database enforces foreign keys, this won't fail.
        return qs.get(self.field.get_reverse_related_filter(instance))
```
### 52 - django/db/models/fields/related_descriptors.py:

Start line: 897, End line: 952

```python
class ManyToManyDescriptor(ReverseManyToOneDescriptor):
    """
    Accessor to the related objects manager on the forward and reverse sides of
    a many-to-many relation.

    In the example::

        class Pizza(Model):
            toppings = ManyToManyField(Topping, related_name='pizzas')

    ``Pizza.toppings`` and ``Topping.pizzas`` are ``ManyToManyDescriptor``
    instances.

    Most of the implementation is delegated to a dynamically defined manager
    class built by ``create_forward_many_to_many_manager()`` defined below.
    """

    def __init__(self, rel, reverse=False):
        super().__init__(rel)

        self.reverse = reverse

    @property
    def through(self):
        # through is provided so that you have easy access to the through
        # model (Book.authors.through) for inlines, etc. This is done as
        # a property to ensure that the fully resolved value is returned.
        return self.rel.through

    @cached_property
    def related_manager_cls(self):
        related_model = self.rel.related_model if self.reverse else self.rel.model

        return create_forward_many_to_many_manager(
            related_model._default_manager.__class__,
            self.rel,
            reverse=self.reverse,
        )

    @cached_property
    def related_manager_cache_key(self):
        if self.reverse:
            # Symmetrical M2Ms won't have an accessor name, but should never
            # end up in the reverse branch anyway, as the related_name ends up
            # being hidden, and no public manager is created.
            return self.rel.get_cache_name()
        else:
            # For forward managers, defer to the field name.
            return self.field.get_cache_name()

    def _get_set_deprecation_msg_params(self):
        return (
            "%s side of a many-to-many set"
            % ("reverse" if self.reverse else "forward"),
            self.rel.get_accessor_name() if self.reverse else self.field.name,
        )
```
### 75 - django/db/models/fields/related_descriptors.py:

Start line: 418, End line: 441

```python
class ReverseOneToOneDescriptor:

    def get_prefetch_queryset(self, instances, queryset=None):
        if queryset is None:
            queryset = self.get_queryset()
        queryset._add_hints(instance=instances[0])

        rel_obj_attr = self.related.field.get_local_related_value
        instance_attr = self.related.field.get_foreign_related_value
        instances_dict = {instance_attr(inst): inst for inst in instances}
        query = {"%s__in" % self.related.field.name: instances}
        queryset = queryset.filter(**query)

        # Since we're going to assign directly in the cache,
        # we must manage the reverse relation cache manually.
        for rel_obj in queryset:
            instance = instances_dict[rel_obj_attr(rel_obj)]
            self.related.field.set_cached_value(rel_obj, instance)
        return (
            queryset,
            rel_obj_attr,
            instance_attr,
            True,
            self.related.get_cache_name(),
            False,
        )
```
### 77 - django/db/models/fields/related_descriptors.py:

Start line: 241, End line: 323

```python
class ForwardManyToOneDescriptor:

    def __set__(self, instance, value):
        """
        Set the related instance through the forward relation.

        With the example above, when setting ``child.parent = parent``:

        - ``self`` is the descriptor managing the ``parent`` attribute
        - ``instance`` is the ``child`` instance
        - ``value`` is the ``parent`` instance on the right of the equal sign
        """
        # An object must be an instance of the related class.
        if value is not None and not isinstance(
            value, self.field.remote_field.model._meta.concrete_model
        ):
            raise ValueError(
                'Cannot assign "%r": "%s.%s" must be a "%s" instance.'
                % (
                    value,
                    instance._meta.object_name,
                    self.field.name,
                    self.field.remote_field.model._meta.object_name,
                )
            )
        elif value is not None:
            if instance._state.db is None:
                instance._state.db = router.db_for_write(
                    instance.__class__, instance=value
                )
            if value._state.db is None:
                value._state.db = router.db_for_write(
                    value.__class__, instance=instance
                )
            if not router.allow_relation(value, instance):
                raise ValueError(
                    'Cannot assign "%r": the current database router prevents this '
                    "relation." % value
                )

        remote_field = self.field.remote_field
        # If we're setting the value of a OneToOneField to None, we need to clear
        # out the cache on any old related object. Otherwise, deleting the
        # previously-related object will also cause this object to be deleted,
        # which is wrong.
        if value is None:
            # Look up the previously-related object, which may still be available
            # since we've not yet cleared out the related field.
            # Use the cache directly, instead of the accessor; if we haven't
            # populated the cache, then we don't care - we're only accessing
            # the object to invalidate the accessor cache, so there's no
            # need to populate the cache just to expire it again.
            related = self.field.get_cached_value(instance, default=None)

            # If we've got an old related object, we need to clear out its
            # cache. This cache also might not exist if the related object
            # hasn't been accessed yet.
            if related is not None:
                remote_field.set_cached_value(related, None)

            for lh_field, rh_field in self.field.related_fields:
                setattr(instance, lh_field.attname, None)

        # Set the values of the related field.
        else:
            for lh_field, rh_field in self.field.related_fields:
                setattr(instance, lh_field.attname, getattr(value, rh_field.attname))

        # Set the related instance cache used by __get__ to avoid an SQL query
        # when accessing the attribute we just set.
        self.field.set_cached_value(instance, value)

        # If this is a one-to-one relation, set the reverse accessor cache on
        # the related object to the current instance to avoid an extra SQL
        # query if it's accessed later on.
        if value is not None and not remote_field.multiple:
            remote_field.set_cached_value(value, instance)

    def __reduce__(self):
        """
        Pickling should return the instance attached by self.field on the
        model, not a new copy of that descriptor. Use getattr() to retrieve
        the instance directly from the model.
        """
        return getattr, (self.field.model, self.field.name)
```
### 87 - django/db/models/fields/related_descriptors.py:

Start line: 601, End line: 629

```python
class ReverseManyToOneDescriptor:

    def __get__(self, instance, cls=None):
        """
        Get the related objects through the reverse relation.

        With the example above, when getting ``parent.children``:

        - ``self`` is the descriptor managing the ``children`` attribute
        - ``instance`` is the ``parent`` instance
        - ``cls`` is the ``Parent`` class (unused)
        """
        if instance is None:
            return self
        key = self.related_manager_cache_key
        instance_cache = instance._state.related_managers_cache
        if key not in instance_cache:
            instance_cache[key] = self.related_manager_cls(instance)
        return instance_cache[key]

    def _get_set_deprecation_msg_params(self):
        return (
            "reverse side of a related set",
            self.rel.get_accessor_name(),
        )

    def __set__(self, instance, value):
        raise TypeError(
            "Direct assignment to the %s is prohibited. Use %s.set() instead."
            % self._get_set_deprecation_msg_params(),
        )
```
### 90 - django/db/models/fields/related_descriptors.py:

Start line: 488, End line: 561

```python
class ReverseOneToOneDescriptor:

    def __set__(self, instance, value):
        """
        Set the related instance through the reverse relation.

        With the example above, when setting ``place.restaurant = restaurant``:

        - ``self`` is the descriptor managing the ``restaurant`` attribute
        - ``instance`` is the ``place`` instance
        - ``value`` is the ``restaurant`` instance on the right of the equal sign

        Keep in mind that ``Restaurant`` holds the foreign key to ``Place``.
        """
        # The similarity of the code below to the code in
        # ForwardManyToOneDescriptor is annoying, but there's a bunch
        # of small differences that would make a common base class convoluted.

        if value is None:
            # Update the cached related instance (if any) & clear the cache.
            # Following the example above, this would be the cached
            # ``restaurant`` instance (if any).
            rel_obj = self.related.get_cached_value(instance, default=None)
            if rel_obj is not None:
                # Remove the ``restaurant`` instance from the ``place``
                # instance cache.
                self.related.delete_cached_value(instance)
                # Set the ``place`` field on the ``restaurant``
                # instance to None.
                setattr(rel_obj, self.related.field.name, None)
        elif not isinstance(value, self.related.related_model):
            # An object must be an instance of the related class.
            raise ValueError(
                'Cannot assign "%r": "%s.%s" must be a "%s" instance.'
                % (
                    value,
                    instance._meta.object_name,
                    self.related.get_accessor_name(),
                    self.related.related_model._meta.object_name,
                )
            )
        else:
            if instance._state.db is None:
                instance._state.db = router.db_for_write(
                    instance.__class__, instance=value
                )
            if value._state.db is None:
                value._state.db = router.db_for_write(
                    value.__class__, instance=instance
                )
            if not router.allow_relation(value, instance):
                raise ValueError(
                    'Cannot assign "%r": the current database router prevents this '
                    "relation." % value
                )

            related_pk = tuple(
                getattr(instance, field.attname)
                for field in self.related.field.foreign_related_fields
            )
            # Set the value of the related field to the value of the related
            # object's related field.
            for index, field in enumerate(self.related.field.local_related_fields):
                setattr(value, field.attname, related_pk[index])

            # Set the related instance cache used by __get__ to avoid an SQL query
            # when accessing the attribute we just set.
            self.related.set_cached_value(instance, value)

            # Set the forward accessor cache on the related object to the current
            # instance to avoid an extra SQL query if it's accessed later on.
            self.related.field.set_cached_value(value, instance)

    def __reduce__(self):
        # Same purpose as ForwardManyToOneDescriptor.__reduce__().
        return getattr, (self.related.model, self.related.name)
```
### 104 - django/db/models/fields/related_descriptors.py:

Start line: 1, End line: 83

```python
"""
Accessors for related objects.

When a field defines a relation between two models, each model class provides
an attribute to access related instances of the other model class (unless the
reverse accessor has been disabled with related_name='+').

Accessors are implemented as descriptors in order to customize access and
assignment. This module defines the descriptor classes.

Forward accessors follow foreign keys. Reverse accessors trace them back. For
example, with the following models::

    class Parent(Model):
        pass

    class Child(Model):
        parent = ForeignKey(Parent, related_name='children')

 ``child.parent`` is a forward many-to-one relation. ``parent.children`` is a
reverse many-to-one relation.

There are three types of relations (many-to-one, one-to-one, and many-to-many)
and two directions (forward and reverse) for a total of six combinations.

1. Related instance on the forward side of a many-to-one relation:
   ``ForwardManyToOneDescriptor``.

   Uniqueness of foreign key values is irrelevant to accessing the related
   instance, making the many-to-one and one-to-one cases identical as far as
   the descriptor is concerned. The constraint is checked upstream (unicity
   validation in forms) or downstream (unique indexes in the database).

2. Related instance on the forward side of a one-to-one
   relation: ``ForwardOneToOneDescriptor``.

   It avoids querying the database when accessing the parent link field in
   a multi-table inheritance scenario.

3. Related instance on the reverse side of a one-to-one relation:
   ``ReverseOneToOneDescriptor``.

   One-to-one relations are asymmetrical, despite the apparent symmetry of the
   name, because they're implemented in the database with a foreign key from
   one table to another. As a consequence ``ReverseOneToOneDescriptor`` is
   slightly different from ``ForwardManyToOneDescriptor``.

4. Related objects manager for related instances on the reverse side of a
   many-to-one relation: ``ReverseManyToOneDescriptor``.

   Unlike the previous two classes, this one provides access to a collection
   of objects. It returns a manager rather than an instance.

5. Related objects manager for related instances on the forward or reverse
   sides of a many-to-many relation: ``ManyToManyDescriptor``.

   Many-to-many relations are symmetrical. The syntax of Django models
   requires declaring them on one side but that's an implementation detail.
   They could be declared on the other side without any change in behavior.
   Therefore the forward and reverse descriptors can be the same.

   If you're looking for ``ForwardManyToManyDescriptor`` or
   ``ReverseManyToManyDescriptor``, use ``ManyToManyDescriptor`` instead.
"""

from django.core.exceptions import FieldError
from django.db import DEFAULT_DB_ALIAS, connections, router, transaction
from django.db.models import Q, Window, signals
from django.db.models.functions import RowNumber
from django.db.models.lookups import GreaterThan, LessThanOrEqual
from django.db.models.query import QuerySet
from django.db.models.query_utils import DeferredAttribute
from django.db.models.utils import resolve_callables
from django.utils.functional import cached_property


class ForeignKeyDeferredAttribute(DeferredAttribute):
    def __set__(self, instance, value):
        if instance.__dict__.get(self.field.attname) != value and self.field.is_cached(
            instance
        ):
            self.field.delete_cached_value(instance)
        instance.__dict__[self.field.attname] = value
```
### 127 - django/db/models/fields/related_descriptors.py:

Start line: 190, End line: 239

```python
class ForwardManyToOneDescriptor:

    def __get__(self, instance, cls=None):
        """
        Get the related instance through the forward relation.

        With the example above, when getting ``child.parent``:

        - ``self`` is the descriptor managing the ``parent`` attribute
        - ``instance`` is the ``child`` instance
        - ``cls`` is the ``Child`` class (we don't need it)
        """
        if instance is None:
            return self

        # The related instance is loaded from the database and then cached
        # by the field on the model instance state. It can also be pre-cached
        # by the reverse accessor (ReverseOneToOneDescriptor).
        try:
            rel_obj = self.field.get_cached_value(instance)
        except KeyError:
            has_value = None not in self.field.get_local_related_value(instance)
            ancestor_link = (
                instance._meta.get_ancestor_link(self.field.model)
                if has_value
                else None
            )
            if ancestor_link and ancestor_link.is_cached(instance):
                # An ancestor link will exist if this field is defined on a
                # multi-table inheritance parent of the instance's class.
                ancestor = ancestor_link.get_cached_value(instance)
                # The value might be cached on an ancestor if the instance
                # originated from walking down the inheritance chain.
                rel_obj = self.field.get_cached_value(ancestor, default=None)
            else:
                rel_obj = None
            if rel_obj is None and has_value:
                rel_obj = self.get_object(instance)
                remote_field = self.field.remote_field
                # If this is a one-to-one relation, set the reverse accessor
                # cache on the related object to the current instance to avoid
                # an extra SQL query if it's accessed later on.
                if not remote_field.multiple:
                    remote_field.set_cached_value(rel_obj, instance)
            self.field.set_cached_value(instance, rel_obj)

        if rel_obj is None and not self.field.null:
            raise self.RelatedObjectDoesNotExist(
                "%s has no %s." % (self.field.model.__name__, self.field.name)
            )
        else:
            return rel_obj
```
### 155 - django/db/models/fields/related_descriptors.py:

Start line: 104, End line: 142

```python
class ForwardManyToOneDescriptor:
    """
    Accessor to the related object on the forward side of a many-to-one or
    one-to-one (via ForwardOneToOneDescriptor subclass) relation.

    In the example::

        class Child(Model):
            parent = ForeignKey(Parent, related_name='children')

    ``Child.parent`` is a ``ForwardManyToOneDescriptor`` instance.
    """

    def __init__(self, field_with_rel):
        self.field = field_with_rel

    @cached_property
    def RelatedObjectDoesNotExist(self):
        # The exception can't be created at initialization time since the
        # related model might not be resolved yet; `self.field.model` might
        # still be a string model reference.
        return type(
            "RelatedObjectDoesNotExist",
            (self.field.remote_field.model.DoesNotExist, AttributeError),
            {
                "__module__": self.field.model.__module__,
                "__qualname__": "%s.%s.RelatedObjectDoesNotExist"
                % (
                    self.field.model.__qualname__,
                    self.field.name,
                ),
            },
        )

    def is_cached(self, instance):
        return self.field.is_cached(instance)

    def get_queryset(self, **hints):
        return self.field.remote_field.model._base_manager.db_manager(hints=hints).all()
```
