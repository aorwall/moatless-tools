# django__django-11085

| **django/django** | `f976ab1b117574db78d884c94e549a6b8e4c9f9b` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 820 |
| **Any found context length** | 820 |
| **Avg pos** | 2.0 |
| **Min pos** | 1 |
| **Max pos** | 1 |
| **Top file pos** | 1 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/db/models/base.py b/django/db/models/base.py
--- a/django/db/models/base.py
+++ b/django/db/models/base.py
@@ -84,9 +84,12 @@ def __new__(cls, name, bases, attrs, **kwargs):
         # Pass all attrs without a (Django-specific) contribute_to_class()
         # method to type.__new__() so that they're properly initialized
         # (i.e. __set_name__()).
+        contributable_attrs = {}
         for obj_name, obj in list(attrs.items()):
-            if not _has_contribute_to_class(obj):
-                new_attrs[obj_name] = attrs.pop(obj_name)
+            if _has_contribute_to_class(obj):
+                contributable_attrs[obj_name] = obj
+            else:
+                new_attrs[obj_name] = obj
         new_class = super_new(cls, name, bases, new_attrs, **kwargs)
 
         abstract = getattr(attr_meta, 'abstract', False)
@@ -146,8 +149,9 @@ def __new__(cls, name, bases, attrs, **kwargs):
         if is_proxy and base_meta and base_meta.swapped:
             raise TypeError("%s cannot proxy the swapped model '%s'." % (name, base_meta.swapped))
 
-        # Add all attributes to the class.
-        for obj_name, obj in attrs.items():
+        # Add remaining attributes (those with a contribute_to_class() method)
+        # to the class.
+        for obj_name, obj in contributable_attrs.items():
             new_class.add_to_class(obj_name, obj)
 
         # All the fields of any type declared on this model

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/db/models/base.py | 87 | 88 | 1 | 1 | 820
| django/db/models/base.py | 149 | 150 | 1 | 1 | 820


## Problem Statement

```
Custom model metaclasses cannot access the attribute dict in __init__
Description
	
In Django <=2.2, it is possible for models to define a custom metaclass (as a subclass of models.base.ModelBase) and access the attribute dict of the class being defined:
from django.db import models
class PageBase(models.base.ModelBase):
	def __init__(cls, name, bases, dct):
		super(PageBase, cls).__init__(name, bases, dct)
		if 'magic' in dct:
			print("enabling magic on %s" % (name))
class Page(models.Model, metaclass=PageBase):
	magic = True
	title = models.CharField(max_length=255)
As of commit a68ea231012434b522ce45c513d84add516afa60, this fails because all attributes without a contribute_to_class method are popped from the dict in ModelBase.__new__ .
(This pattern is used by Wagtail's Page model ​https://github.com/wagtail/wagtail/blob/3e1e67021e0a20783ed59e17b43e3c481897fce3/wagtail/core/models.py#L190 , so this is causing various failures against django stable/2.2.x.)

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 django/db/models/base.py** | 66 | 161| 820 | 820 | 14837 | 
| 2 | **1 django/db/models/base.py** | 394 | 498| 903 | 1723 | 14837 | 
| 3 | **1 django/db/models/base.py** | 1224 | 1253| 242 | 1965 | 14837 | 
| 4 | **1 django/db/models/base.py** | 202 | 312| 866 | 2831 | 14837 | 
| 5 | 2 django/forms/models.py | 204 | 273| 623 | 3454 | 26283 | 
| 6 | **2 django/db/models/base.py** | 314 | 372| 525 | 3979 | 26283 | 
| 7 | **2 django/db/models/base.py** | 162 | 201| 385 | 4364 | 26283 | 
| 8 | **2 django/db/models/base.py** | 1444 | 1466| 171 | 4535 | 26283 | 
| 9 | **2 django/db/models/base.py** | 1468 | 1500| 231 | 4766 | 26283 | 
| 10 | **2 django/db/models/base.py** | 1562 | 1608| 324 | 5090 | 26283 | 
| 11 | **2 django/db/models/base.py** | 1133 | 1161| 213 | 5303 | 26283 | 
| 12 | **2 django/db/models/base.py** | 544 | 560| 142 | 5445 | 26283 | 
| 13 | **2 django/db/models/base.py** | 1777 | 1798| 155 | 5600 | 26283 | 
| 14 | **2 django/db/models/base.py** | 1087 | 1114| 286 | 5886 | 26283 | 
| 15 | **2 django/db/models/base.py** | 500 | 542| 347 | 6233 | 26283 | 
| 16 | **2 django/db/models/base.py** | 1042 | 1085| 404 | 6637 | 26283 | 
| 17 | **2 django/db/models/base.py** | 938 | 951| 180 | 6817 | 26283 | 
| 18 | 3 django/db/models/options.py | 149 | 202| 518 | 7335 | 33149 | 
| 19 | **3 django/db/models/base.py** | 1116 | 1131| 138 | 7473 | 33149 | 
| 20 | **3 django/db/models/base.py** | 1419 | 1442| 176 | 7649 | 33149 | 
| 21 | **3 django/db/models/base.py** | 1362 | 1417| 491 | 8140 | 33149 | 
| 22 | **3 django/db/models/base.py** | 1282 | 1311| 205 | 8345 | 33149 | 
| 23 | **3 django/db/models/base.py** | 1801 | 1852| 351 | 8696 | 33149 | 
| 24 | **3 django/db/models/base.py** | 1704 | 1775| 565 | 9261 | 33149 | 
| 25 | 4 django/db/models/manager.py | 165 | 202| 211 | 9472 | 34583 | 
| 26 | 5 django/contrib/flatpages/models.py | 1 | 41| 311 | 9783 | 34894 | 
| 27 | 6 django/db/migrations/state.py | 349 | 399| 471 | 10254 | 40112 | 
| 28 | 7 django/contrib/sites/models.py | 78 | 100| 115 | 10369 | 40901 | 
| 29 | 7 django/forms/models.py | 276 | 304| 288 | 10657 | 40901 | 
| 30 | **7 django/db/models/base.py** | 1313 | 1343| 244 | 10901 | 40901 | 
| 31 | 8 django/core/cache/backends/db.py | 1 | 37| 229 | 11130 | 42987 | 
| 32 | 9 django/core/cache/backends/memcached.py | 110 | 126| 168 | 11298 | 44705 | 
| 33 | **9 django/db/models/base.py** | 1502 | 1527| 183 | 11481 | 44705 | 
| 34 | 10 django/forms/forms.py | 25 | 53| 229 | 11710 | 48778 | 
| 35 | 11 django/contrib/auth/checks.py | 97 | 167| 525 | 12235 | 49951 | 
| 36 | **11 django/db/models/base.py** | 1199 | 1222| 172 | 12407 | 49951 | 
| 37 | **11 django/db/models/base.py** | 47 | 63| 145 | 12552 | 49951 | 
| 38 | 11 django/db/models/manager.py | 1 | 162| 1223 | 13775 | 49951 | 
| 39 | 11 django/db/models/options.py | 369 | 395| 175 | 13950 | 49951 | 
| 40 | 12 django/contrib/admin/options.py | 1 | 96| 769 | 14719 | 68274 | 
| 41 | **12 django/db/models/base.py** | 1345 | 1360| 153 | 14872 | 68274 | 
| 42 | **12 django/db/models/base.py** | 1255 | 1280| 184 | 15056 | 68274 | 
| 43 | 13 django/contrib/contenttypes/models.py | 133 | 185| 381 | 15437 | 69690 | 
| 44 | **13 django/db/models/base.py** | 1163 | 1197| 230 | 15667 | 69690 | 
| 45 | **13 django/db/models/base.py** | 562 | 578| 133 | 15800 | 69690 | 
| 46 | 13 django/core/cache/backends/memcached.py | 92 | 108| 167 | 15967 | 69690 | 
| 47 | **13 django/db/models/base.py** | 922 | 936| 212 | 16179 | 69690 | 
| 48 | **13 django/db/models/base.py** | 1 | 44| 281 | 16460 | 69690 | 
| 49 | 14 django/db/models/fields/related.py | 1193 | 1316| 1010 | 17470 | 83185 | 
| 50 | 14 django/db/migrations/state.py | 496 | 528| 250 | 17720 | 83185 | 
| 51 | 14 django/contrib/admin/options.py | 1514 | 1593| 719 | 18439 | 83185 | 
| 52 | 15 django/contrib/flatpages/migrations/0001_initial.py | 1 | 40| 307 | 18746 | 83492 | 
| 53 | 16 django/db/models/constraints.py | 1 | 27| 203 | 18949 | 84491 | 
| 54 | 16 django/contrib/admin/options.py | 538 | 596| 408 | 19357 | 84491 | 
| 55 | 16 django/core/cache/backends/memcached.py | 65 | 90| 283 | 19640 | 84491 | 
| 56 | 17 django/contrib/admindocs/views.py | 249 | 314| 585 | 20225 | 87801 | 
| 57 | 17 django/core/cache/backends/memcached.py | 145 | 171| 264 | 20489 | 87801 | 
| 58 | 17 django/contrib/contenttypes/models.py | 1 | 32| 223 | 20712 | 87801 | 
| 59 | **17 django/db/models/base.py** | 375 | 391| 128 | 20840 | 87801 | 
| 60 | 17 django/db/models/options.py | 567 | 593| 197 | 21037 | 87801 | 
| 61 | **17 django/db/models/base.py** | 1529 | 1560| 237 | 21274 | 87801 | 
| 62 | 17 django/forms/models.py | 306 | 345| 387 | 21661 | 87801 | 
| 63 | 18 django/contrib/contenttypes/admin.py | 83 | 130| 410 | 22071 | 88826 | 
| 64 | 19 django/contrib/sessions/base_session.py | 26 | 48| 139 | 22210 | 89115 | 
| 65 | 20 django/utils/datastructures.py | 42 | 149| 766 | 22976 | 91372 | 
| 66 | 20 django/db/models/fields/related.py | 1596 | 1612| 286 | 23262 | 91372 | 
| 67 | 20 django/db/models/fields/related.py | 1559 | 1594| 475 | 23737 | 91372 | 
| 68 | 21 django/contrib/auth/models.py | 33 | 78| 404 | 24141 | 94300 | 
| 69 | 21 django/contrib/admin/options.py | 99 | 129| 223 | 24364 | 94300 | 
| 70 | 21 django/db/models/options.py | 345 | 367| 164 | 24528 | 94300 | 
| 71 | 22 django/apps/registry.py | 212 | 232| 237 | 24765 | 97707 | 
| 72 | 22 django/forms/models.py | 379 | 407| 240 | 25005 | 97707 | 
| 73 | **22 django/db/models/base.py** | 953 | 981| 230 | 25235 | 97707 | 
| 74 | 23 django/db/models/fields/__init__.py | 1114 | 1143| 218 | 25453 | 114545 | 
| 75 | 23 django/contrib/admin/options.py | 2141 | 2163| 201 | 25654 | 114545 | 
| 76 | 24 django/core/checks/model_checks.py | 1 | 42| 282 | 25936 | 115947 | 
| 77 | 24 django/contrib/auth/models.py | 81 | 125| 309 | 26245 | 115947 | 
| 78 | 25 django/contrib/admin/checks.py | 1008 | 1035| 204 | 26449 | 124908 | 
| 79 | 25 django/contrib/admin/options.py | 418 | 461| 350 | 26799 | 124908 | 
| 80 | 25 django/forms/models.py | 347 | 377| 233 | 27032 | 124908 | 
| 81 | 26 django/utils/deprecation.py | 33 | 73| 336 | 27368 | 125600 | 
| 82 | 26 django/db/models/options.py | 204 | 242| 362 | 27730 | 125600 | 
| 83 | 26 django/db/models/fields/__init__.py | 1069 | 1095| 229 | 27959 | 125600 | 
| 84 | 26 django/db/models/fields/__init__.py | 1 | 85| 673 | 28632 | 125600 | 
| 85 | 26 django/contrib/auth/models.py | 353 | 431| 460 | 29092 | 125600 | 
| 86 | 27 django/contrib/contenttypes/checks.py | 24 | 42| 125 | 29217 | 125849 | 
| 87 | 27 django/forms/models.py | 623 | 640| 167 | 29384 | 125849 | 
| 88 | **27 django/db/models/base.py** | 788 | 816| 307 | 29691 | 125849 | 
| 89 | 27 django/contrib/admindocs/views.py | 182 | 248| 584 | 30275 | 125849 | 
| 90 | 27 django/contrib/sites/models.py | 25 | 46| 192 | 30467 | 125849 | 
| 91 | 27 django/contrib/admin/options.py | 642 | 656| 136 | 30603 | 125849 | 
| 92 | 28 django/contrib/sessions/backends/base.py | 127 | 207| 547 | 31150 | 128374 | 
| 93 | 29 django/contrib/sites/managers.py | 1 | 61| 385 | 31535 | 128759 | 
| 94 | 29 django/db/models/options.py | 1 | 36| 304 | 31839 | 128759 | 
| 95 | 29 django/contrib/sessions/backends/base.py | 38 | 107| 513 | 32352 | 128759 | 
| 96 | 29 django/contrib/contenttypes/models.py | 62 | 102| 326 | 32678 | 128759 | 
| 97 | 30 django/core/exceptions.py | 1 | 91| 381 | 33059 | 129764 | 
| 98 | 30 django/contrib/admindocs/views.py | 155 | 179| 234 | 33293 | 129764 | 
| 99 | 31 django/core/cache/__init__.py | 90 | 125| 229 | 33522 | 130577 | 
| 100 | 31 django/forms/models.py | 1 | 28| 215 | 33737 | 130577 | 
| 101 | 32 django/core/paginator.py | 1 | 53| 305 | 34042 | 131892 | 
| 102 | 33 django/db/migrations/autodetector.py | 1020 | 1036| 188 | 34230 | 143563 | 
| 103 | 33 django/forms/models.py | 191 | 201| 131 | 34361 | 143563 | 
| 104 | 33 django/db/models/constraints.py | 30 | 66| 309 | 34670 | 143563 | 
| 105 | 34 django/forms/widgets.py | 155 | 191| 235 | 34905 | 151590 | 
| 106 | 34 django/contrib/admin/options.py | 2105 | 2139| 359 | 35264 | 151590 | 
| 107 | 34 django/db/migrations/state.py | 401 | 495| 808 | 36072 | 151590 | 
| 108 | 34 django/core/cache/backends/memcached.py | 128 | 142| 151 | 36223 | 151590 | 
| 109 | 35 django/db/migrations/operations/utils.py | 17 | 54| 340 | 36563 | 152070 | 
| 110 | 36 django/contrib/contenttypes/fields.py | 1 | 15| 126 | 36689 | 157373 | 
| 111 | 37 django/contrib/sessions/models.py | 1 | 36| 250 | 36939 | 157623 | 
| 112 | **37 django/db/models/base.py** | 1610 | 1702| 673 | 37612 | 157623 | 
| 113 | 37 django/core/cache/backends/memcached.py | 174 | 194| 189 | 37801 | 157623 | 
| 114 | 37 django/contrib/admin/checks.py | 874 | 922| 416 | 38217 | 157623 | 
| 115 | 38 django/core/management/base.py | 148 | 228| 755 | 38972 | 161987 | 
| 116 | 38 django/contrib/admindocs/views.py | 1 | 29| 216 | 39188 | 161987 | 
| 117 | 39 django/template/context.py | 135 | 169| 288 | 39476 | 163876 | 
| 118 | 40 django/apps/config.py | 204 | 217| 117 | 39593 | 165608 | 
| 119 | 40 django/db/models/fields/related.py | 1614 | 1648| 266 | 39859 | 165608 | 
| 120 | 40 django/db/models/options.py | 397 | 419| 154 | 40013 | 165608 | 
| 121 | 40 django/db/models/fields/__init__.py | 1035 | 1067| 208 | 40221 | 165608 | 
| 122 | 40 django/contrib/admin/checks.py | 615 | 634| 183 | 40404 | 165608 | 
| 123 | 41 django/contrib/gis/db/models/__init__.py | 1 | 19| 204 | 40608 | 165812 | 
| 124 | 42 django/contrib/contenttypes/migrations/0001_initial.py | 1 | 35| 207 | 40815 | 166019 | 
| 125 | 43 django/conf/__init__.py | 132 | 185| 472 | 41287 | 167805 | 
| 126 | 44 django/db/models/fields/related_descriptors.py | 1 | 70| 602 | 41889 | 178326 | 
| 127 | 44 django/forms/models.py | 854 | 878| 298 | 42187 | 178326 | 
| 128 | 44 django/forms/models.py | 913 | 945| 353 | 42540 | 178326 | 
| 129 | 44 django/contrib/auth/checks.py | 1 | 94| 646 | 43186 | 178326 | 
| 130 | 44 django/forms/models.py | 589 | 621| 270 | 43456 | 178326 | 
| 131 | 44 django/contrib/admin/options.py | 1463 | 1479| 231 | 43687 | 178326 | 
| 132 | 44 django/db/migrations/autodetector.py | 1202 | 1214| 131 | 43818 | 178326 | 
| 133 | 44 django/contrib/admin/options.py | 1960 | 1991| 249 | 44067 | 178326 | 
| 134 | 44 django/db/migrations/autodetector.py | 796 | 845| 570 | 44637 | 178326 | 
| 135 | 44 django/contrib/admin/checks.py | 578 | 589| 128 | 44765 | 178326 | 
| 136 | 45 django/core/serializers/base.py | 195 | 217| 183 | 44948 | 180719 | 
| 137 | 45 django/contrib/admin/options.py | 364 | 416| 504 | 45452 | 180719 | 
| 138 | **45 django/db/models/base.py** | 658 | 736| 785 | 46237 | 180719 | 
| 139 | 46 django/http/request.py | 487 | 507| 146 | 46383 | 185405 | 
| 140 | 46 django/db/models/fields/related.py | 255 | 282| 269 | 46652 | 185405 | 
| 141 | **46 django/db/models/base.py** | 870 | 895| 314 | 46966 | 185405 | 
| 142 | 46 django/contrib/contenttypes/fields.py | 428 | 449| 248 | 47214 | 185405 | 
| 143 | 46 django/db/models/fields/__init__.py | 1289 | 1302| 154 | 47368 | 185405 | 
| 144 | 46 django/db/migrations/state.py | 27 | 54| 233 | 47601 | 185405 | 
| 145 | 46 django/contrib/admin/options.py | 206 | 217| 135 | 47736 | 185405 | 
| 146 | 47 django/views/generic/base.py | 61 | 81| 203 | 47939 | 187008 | 
| 147 | 48 django/core/serializers/xml_serializer.py | 374 | 390| 126 | 48065 | 190402 | 
| 148 | 49 django/db/models/__init__.py | 1 | 49| 548 | 48613 | 190950 | 
| 149 | 49 django/contrib/admin/checks.py | 713 | 744| 227 | 48840 | 190950 | 
| 150 | 49 django/db/models/options.py | 65 | 147| 668 | 49508 | 190950 | 


## Patch

```diff
diff --git a/django/db/models/base.py b/django/db/models/base.py
--- a/django/db/models/base.py
+++ b/django/db/models/base.py
@@ -84,9 +84,12 @@ def __new__(cls, name, bases, attrs, **kwargs):
         # Pass all attrs without a (Django-specific) contribute_to_class()
         # method to type.__new__() so that they're properly initialized
         # (i.e. __set_name__()).
+        contributable_attrs = {}
         for obj_name, obj in list(attrs.items()):
-            if not _has_contribute_to_class(obj):
-                new_attrs[obj_name] = attrs.pop(obj_name)
+            if _has_contribute_to_class(obj):
+                contributable_attrs[obj_name] = obj
+            else:
+                new_attrs[obj_name] = obj
         new_class = super_new(cls, name, bases, new_attrs, **kwargs)
 
         abstract = getattr(attr_meta, 'abstract', False)
@@ -146,8 +149,9 @@ def __new__(cls, name, bases, attrs, **kwargs):
         if is_proxy and base_meta and base_meta.swapped:
             raise TypeError("%s cannot proxy the swapped model '%s'." % (name, base_meta.swapped))
 
-        # Add all attributes to the class.
-        for obj_name, obj in attrs.items():
+        # Add remaining attributes (those with a contribute_to_class() method)
+        # to the class.
+        for obj_name, obj in contributable_attrs.items():
             new_class.add_to_class(obj_name, obj)
 
         # All the fields of any type declared on this model

```

## Test Patch

```diff
diff --git a/tests/model_regress/tests.py b/tests/model_regress/tests.py
--- a/tests/model_regress/tests.py
+++ b/tests/model_regress/tests.py
@@ -2,9 +2,10 @@
 from operator import attrgetter
 
 from django.core.exceptions import ValidationError
-from django.db import router
+from django.db import models, router
 from django.db.models.sql import InsertQuery
 from django.test import TestCase, skipUnlessDBFeature
+from django.test.utils import isolate_apps
 from django.utils.timezone import get_fixed_timezone
 
 from .models import (
@@ -217,6 +218,23 @@ def test_chained_fks(self):
         m3 = Model3.objects.get(model2=1000)
         m3.model2
 
+    @isolate_apps('model_regress')
+    def test_metaclass_can_access_attribute_dict(self):
+        """
+        Model metaclasses have access to the class attribute dict in
+        __init__() (#30254).
+        """
+        class HorseBase(models.base.ModelBase):
+            def __init__(cls, name, bases, attrs):
+                super(HorseBase, cls).__init__(name, bases, attrs)
+                cls.horns = (1 if 'magic' in attrs else 0)
+
+        class Horse(models.Model, metaclass=HorseBase):
+            name = models.CharField(max_length=255)
+            magic = True
+
+        self.assertEqual(Horse.horns, 1)
+
 
 class ModelValidationTest(TestCase):
     def test_pk_validation(self):

```


## Code snippets

### 1 - django/db/models/base.py:

Start line: 66, End line: 161

```python
class ModelBase(type):
    """Metaclass for all models."""
    def __new__(cls, name, bases, attrs, **kwargs):
        super_new = super().__new__

        # Also ensure initialization is only performed for subclasses of Model
        # (excluding Model class itself).
        parents = [b for b in bases if isinstance(b, ModelBase)]
        if not parents:
            return super_new(cls, name, bases, attrs)

        # Create the class.
        module = attrs.pop('__module__')
        new_attrs = {'__module__': module}
        classcell = attrs.pop('__classcell__', None)
        if classcell is not None:
            new_attrs['__classcell__'] = classcell
        attr_meta = attrs.pop('Meta', None)
        # Pass all attrs without a (Django-specific) contribute_to_class()
        # method to type.__new__() so that they're properly initialized
        # (i.e. __set_name__()).
        for obj_name, obj in list(attrs.items()):
            if not _has_contribute_to_class(obj):
                new_attrs[obj_name] = attrs.pop(obj_name)
        new_class = super_new(cls, name, bases, new_attrs, **kwargs)

        abstract = getattr(attr_meta, 'abstract', False)
        meta = attr_meta or getattr(new_class, 'Meta', None)
        base_meta = getattr(new_class, '_meta', None)

        app_label = None

        # Look for an application configuration to attach the model to.
        app_config = apps.get_containing_app_config(module)

        if getattr(meta, 'app_label', None) is None:
            if app_config is None:
                if not abstract:
                    raise RuntimeError(
                        "Model class %s.%s doesn't declare an explicit "
                        "app_label and isn't in an application in "
                        "INSTALLED_APPS." % (module, name)
                    )

            else:
                app_label = app_config.label

        new_class.add_to_class('_meta', Options(meta, app_label))
        if not abstract:
            new_class.add_to_class(
                'DoesNotExist',
                subclass_exception(
                    'DoesNotExist',
                    tuple(
                        x.DoesNotExist for x in parents if hasattr(x, '_meta') and not x._meta.abstract
                    ) or (ObjectDoesNotExist,),
                    module,
                    attached_to=new_class))
            new_class.add_to_class(
                'MultipleObjectsReturned',
                subclass_exception(
                    'MultipleObjectsReturned',
                    tuple(
                        x.MultipleObjectsReturned for x in parents if hasattr(x, '_meta') and not x._meta.abstract
                    ) or (MultipleObjectsReturned,),
                    module,
                    attached_to=new_class))
            if base_meta and not base_meta.abstract:
                # Non-abstract child classes inherit some attributes from their
                # non-abstract parent (unless an ABC comes before it in the
                # method resolution order).
                if not hasattr(meta, 'ordering'):
                    new_class._meta.ordering = base_meta.ordering
                if not hasattr(meta, 'get_latest_by'):
                    new_class._meta.get_latest_by = base_meta.get_latest_by

        is_proxy = new_class._meta.proxy

        # If the model is a proxy, ensure that the base class
        # hasn't been swapped out.
        if is_proxy and base_meta and base_meta.swapped:
            raise TypeError("%s cannot proxy the swapped model '%s'." % (name, base_meta.swapped))

        # Add all attributes to the class.
        for obj_name, obj in attrs.items():
            new_class.add_to_class(obj_name, obj)

        # All the fields of any type declared on this model
        new_fields = chain(
            new_class._meta.local_fields,
            new_class._meta.local_many_to_many,
            new_class._meta.private_fields
        )
        field_names = {f.name for f in new_fields}

        # Basic setup for proxy models.
        # ... other code
```
### 2 - django/db/models/base.py:

Start line: 394, End line: 498

```python
class Model(metaclass=ModelBase):

    def __init__(self, *args, **kwargs):
        # Alias some things as locals to avoid repeat global lookups
        cls = self.__class__
        opts = self._meta
        _setattr = setattr
        _DEFERRED = DEFERRED

        pre_init.send(sender=cls, args=args, kwargs=kwargs)

        # Set up the storage for instance state
        self._state = ModelState()

        # There is a rather weird disparity here; if kwargs, it's set, then args
        # overrides it. It should be one or the other; don't duplicate the work
        # The reason for the kwargs check is that standard iterator passes in by
        # args, and instantiation for iteration is 33% faster.
        if len(args) > len(opts.concrete_fields):
            # Daft, but matches old exception sans the err msg.
            raise IndexError("Number of args exceeds number of fields")

        if not kwargs:
            fields_iter = iter(opts.concrete_fields)
            # The ordering of the zip calls matter - zip throws StopIteration
            # when an iter throws it. So if the first iter throws it, the second
            # is *not* consumed. We rely on this, so don't change the order
            # without changing the logic.
            for val, field in zip(args, fields_iter):
                if val is _DEFERRED:
                    continue
                _setattr(self, field.attname, val)
        else:
            # Slower, kwargs-ready version.
            fields_iter = iter(opts.fields)
            for val, field in zip(args, fields_iter):
                if val is _DEFERRED:
                    continue
                _setattr(self, field.attname, val)
                kwargs.pop(field.name, None)

        # Now we're left with the unprocessed fields that *must* come from
        # keywords, or default.

        for field in fields_iter:
            is_related_object = False
            # Virtual field
            if field.attname not in kwargs and field.column is None:
                continue
            if kwargs:
                if isinstance(field.remote_field, ForeignObjectRel):
                    try:
                        # Assume object instance was passed in.
                        rel_obj = kwargs.pop(field.name)
                        is_related_object = True
                    except KeyError:
                        try:
                            # Object instance wasn't passed in -- must be an ID.
                            val = kwargs.pop(field.attname)
                        except KeyError:
                            val = field.get_default()
                    else:
                        # Object instance was passed in. Special case: You can
                        # pass in "None" for related objects if it's allowed.
                        if rel_obj is None and field.null:
                            val = None
                else:
                    try:
                        val = kwargs.pop(field.attname)
                    except KeyError:
                        # This is done with an exception rather than the
                        # default argument on pop because we don't want
                        # get_default() to be evaluated, and then not used.
                        # Refs #12057.
                        val = field.get_default()
            else:
                val = field.get_default()

            if is_related_object:
                # If we are passed a related instance, set it using the
                # field.name instead of field.attname (e.g. "user" instead of
                # "user_id") so that the object gets properly cached (and type
                # checked) by the RelatedObjectDescriptor.
                if rel_obj is not _DEFERRED:
                    _setattr(self, field.name, rel_obj)
            else:
                if val is not _DEFERRED:
                    _setattr(self, field.attname, val)

        if kwargs:
            property_names = opts._property_names
            for prop in tuple(kwargs):
                try:
                    # Any remaining kwargs must correspond to properties or
                    # virtual fields.
                    if prop in property_names or opts.get_field(prop):
                        if kwargs[prop] is not _DEFERRED:
                            _setattr(self, prop, kwargs[prop])
                        del kwargs[prop]
                except (AttributeError, FieldDoesNotExist):
                    pass
            for kwarg in kwargs:
                raise TypeError("%s() got an unexpected keyword argument '%s'" % (cls.__name__, kwarg))
        super().__init__()
        post_init.send(sender=cls, instance=self)
```
### 3 - django/db/models/base.py:

Start line: 1224, End line: 1253

```python
class Model(metaclass=ModelBase):

    @classmethod
    def check(cls, **kwargs):
        errors = [*cls._check_swappable(), *cls._check_model(), *cls._check_managers(**kwargs)]
        if not cls._meta.swapped:
            errors += [
                *cls._check_fields(**kwargs),
                *cls._check_m2m_through_same_relationship(),
                *cls._check_long_column_names(),
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
                *cls._check_indexes(),
                *cls._check_ordering(),
                *cls._check_constraints(),
            ]

        return errors
```
### 4 - django/db/models/base.py:

Start line: 202, End line: 312

```python
class ModelBase(type):
    def __new__(cls, name, bases, attrs, **kwargs):
        # ... other code
        for base in new_class.mro():
            if base not in parents or not hasattr(base, '_meta'):
                # Things without _meta aren't functional models, so they're
                # uninteresting parents.
                inherited_attributes.update(base.__dict__)
                continue

            parent_fields = base._meta.local_fields + base._meta.local_many_to_many
            if not base._meta.abstract:
                # Check for clashes between locally declared fields and those
                # on the base classes.
                for field in parent_fields:
                    if field.name in field_names:
                        raise FieldError(
                            'Local field %r in class %r clashes with field of '
                            'the same name from base class %r.' % (
                                field.name,
                                name,
                                base.__name__,
                            )
                        )
                    else:
                        inherited_attributes.add(field.name)

                # Concrete classes...
                base = base._meta.concrete_model
                base_key = make_model_tuple(base)
                if base_key in parent_links:
                    field = parent_links[base_key]
                elif not is_proxy:
                    attr_name = '%s_ptr' % base._meta.model_name
                    field = OneToOneField(
                        base,
                        on_delete=CASCADE,
                        name=attr_name,
                        auto_created=True,
                        parent_link=True,
                    )

                    if attr_name in field_names:
                        raise FieldError(
                            "Auto-generated field '%s' in class %r for "
                            "parent_link to base class %r clashes with "
                            "declared field of the same name." % (
                                attr_name,
                                name,
                                base.__name__,
                            )
                        )

                    # Only add the ptr field if it's not already present;
                    # e.g. migrations will already have it specified
                    if not hasattr(new_class, attr_name):
                        new_class.add_to_class(attr_name, field)
                else:
                    field = None
                new_class._meta.parents[base] = field
            else:
                base_parents = base._meta.parents.copy()

                # Add fields from abstract base class if it wasn't overridden.
                for field in parent_fields:
                    if (field.name not in field_names and
                            field.name not in new_class.__dict__ and
                            field.name not in inherited_attributes):
                        new_field = copy.deepcopy(field)
                        new_class.add_to_class(field.name, new_field)
                        # Replace parent links defined on this base by the new
                        # field. It will be appropriately resolved if required.
                        if field.one_to_one:
                            for parent, parent_link in base_parents.items():
                                if field == parent_link:
                                    base_parents[parent] = new_field

                # Pass any non-abstract parent classes onto child.
                new_class._meta.parents.update(base_parents)

            # Inherit private fields (like GenericForeignKey) from the parent
            # class
            for field in base._meta.private_fields:
                if field.name in field_names:
                    if not base._meta.abstract:
                        raise FieldError(
                            'Local field %r in class %r clashes with field of '
                            'the same name from base class %r.' % (
                                field.name,
                                name,
                                base.__name__,
                            )
                        )
                else:
                    field = copy.deepcopy(field)
                    if not base._meta.abstract:
                        field.mti_inherited = True
                    new_class.add_to_class(field.name, field)

        # Copy indexes so that index names are unique when models extend an
        # abstract model.
        new_class._meta.indexes = [copy.deepcopy(idx) for idx in new_class._meta.indexes]

        if abstract:
            # Abstract base models can't be instantiated and don't appear in
            # the list of models for an app. We do the final setup for them a
            # little differently from normal models.
            attr_meta.abstract = False
            new_class.Meta = attr_meta
            return new_class

        new_class._prepare()
        new_class._meta.apps.register_model(new_class._meta.app_label, new_class)
        return new_class
```
### 5 - django/forms/models.py:

Start line: 204, End line: 273

```python
class ModelFormMetaclass(DeclarativeFieldsMetaclass):
    def __new__(mcs, name, bases, attrs):
        base_formfield_callback = None
        for b in bases:
            if hasattr(b, 'Meta') and hasattr(b.Meta, 'formfield_callback'):
                base_formfield_callback = b.Meta.formfield_callback
                break

        formfield_callback = attrs.pop('formfield_callback', base_formfield_callback)

        new_class = super(ModelFormMetaclass, mcs).__new__(mcs, name, bases, attrs)

        if bases == (BaseModelForm,):
            return new_class

        opts = new_class._meta = ModelFormOptions(getattr(new_class, 'Meta', None))

        # We check if a string was passed to `fields` or `exclude`,
        # which is likely to be a mistake where the user typed ('foo') instead
        # of ('foo',)
        for opt in ['fields', 'exclude', 'localized_fields']:
            value = getattr(opts, opt)
            if isinstance(value, str) and value != ALL_FIELDS:
                msg = ("%(model)s.Meta.%(opt)s cannot be a string. "
                       "Did you mean to type: ('%(value)s',)?" % {
                           'model': new_class.__name__,
                           'opt': opt,
                           'value': value,
                       })
                raise TypeError(msg)

        if opts.model:
            # If a model is defined, extract form fields from it.
            if opts.fields is None and opts.exclude is None:
                raise ImproperlyConfigured(
                    "Creating a ModelForm without either the 'fields' attribute "
                    "or the 'exclude' attribute is prohibited; form %s "
                    "needs updating." % name
                )

            if opts.fields == ALL_FIELDS:
                # Sentinel for fields_for_model to indicate "get the list of
                # fields from the model"
                opts.fields = None

            fields = fields_for_model(
                opts.model, opts.fields, opts.exclude, opts.widgets,
                formfield_callback, opts.localized_fields, opts.labels,
                opts.help_texts, opts.error_messages, opts.field_classes,
                # limit_choices_to will be applied during ModelForm.__init__().
                apply_limit_choices_to=False,
            )

            # make sure opts.fields doesn't specify an invalid field
            none_model_fields = {k for k, v in fields.items() if not v}
            missing_fields = none_model_fields.difference(new_class.declared_fields)
            if missing_fields:
                message = 'Unknown field(s) (%s) specified for %s'
                message = message % (', '.join(missing_fields),
                                     opts.model.__name__)
                raise FieldError(message)
            # Override default model fields with any custom declared ones
            # (plus, include all the other declared fields).
            fields.update(new_class.declared_fields)
        else:
            fields = new_class.declared_fields

        new_class.base_fields = fields

        return new_class
```
### 6 - django/db/models/base.py:

Start line: 314, End line: 372

```python
class ModelBase(type):

    def add_to_class(cls, name, value):
        if _has_contribute_to_class(value):
            value.contribute_to_class(cls, name)
        else:
            setattr(cls, name, value)

    def _prepare(cls):
        """Create some methods once self._meta has been populated."""
        opts = cls._meta
        opts._prepare(cls)

        if opts.order_with_respect_to:
            cls.get_next_in_order = partialmethod(cls._get_next_or_previous_in_order, is_next=True)
            cls.get_previous_in_order = partialmethod(cls._get_next_or_previous_in_order, is_next=False)

            # Defer creating accessors on the foreign class until it has been
            # created and registered. If remote_field is None, we're ordering
            # with respect to a GenericForeignKey and don't know what the
            # foreign class is - we'll add those accessors later in
            # contribute_to_class().
            if opts.order_with_respect_to.remote_field:
                wrt = opts.order_with_respect_to
                remote = wrt.remote_field.model
                lazy_related_operation(make_foreign_order_accessors, cls, remote)

        # Give the class a docstring -- its definition.
        if cls.__doc__ is None:
            cls.__doc__ = "%s(%s)" % (cls.__name__, ", ".join(f.name for f in opts.fields))

        get_absolute_url_override = settings.ABSOLUTE_URL_OVERRIDES.get(opts.label_lower)
        if get_absolute_url_override:
            setattr(cls, 'get_absolute_url', get_absolute_url_override)

        if not opts.managers:
            if any(f.name == 'objects' for f in opts.fields):
                raise ValueError(
                    "Model %s must specify a custom Manager, because it has a "
                    "field named 'objects'." % cls.__name__
                )
            manager = Manager()
            manager.auto_created = True
            cls.add_to_class('objects', manager)

        # Set the name of _meta.indexes. This can't be done in
        # Options.contribute_to_class() because fields haven't been added to
        # the model at that point.
        for index in cls._meta.indexes:
            if not index.name:
                index.set_name_with_model(cls)

        class_prepared.send(sender=cls)

    @property
    def _base_manager(cls):
        return cls._meta.base_manager

    @property
    def _default_manager(cls):
        return cls._meta.default_manager
```
### 7 - django/db/models/base.py:

Start line: 162, End line: 201

```python
class ModelBase(type):
    def __new__(cls, name, bases, attrs, **kwargs):
        # ... other code
        if is_proxy:
            base = None
            for parent in [kls for kls in parents if hasattr(kls, '_meta')]:
                if parent._meta.abstract:
                    if parent._meta.fields:
                        raise TypeError(
                            "Abstract base class containing model fields not "
                            "permitted for proxy model '%s'." % name
                        )
                    else:
                        continue
                if base is None:
                    base = parent
                elif parent._meta.concrete_model is not base._meta.concrete_model:
                    raise TypeError("Proxy model '%s' has more than one non-abstract model base class." % name)
            if base is None:
                raise TypeError("Proxy model '%s' has no non-abstract model base class." % name)
            new_class._meta.setup_proxy(base)
            new_class._meta.concrete_model = base._meta.concrete_model
        else:
            new_class._meta.concrete_model = new_class

        # Collect the parent links for multi-table inheritance.
        parent_links = {}
        for base in reversed([new_class] + parents):
            # Conceptually equivalent to `if base is Model`.
            if not hasattr(base, '_meta'):
                continue
            # Skip concrete parent classes.
            if base != new_class and not base._meta.abstract:
                continue
            # Locate OneToOneField instances.
            for field in base._meta.local_fields:
                if isinstance(field, OneToOneField):
                    related = resolve_relation(new_class, field.remote_field.model)
                    parent_links[make_model_tuple(related)] = field

        # Track fields inherited from base models.
        inherited_attributes = set()
        # Do the appropriate setup for any model parents.
        # ... other code
```
### 8 - django/db/models/base.py:

Start line: 1444, End line: 1466

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
### 9 - django/db/models/base.py:

Start line: 1468, End line: 1500

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
### 10 - django/db/models/base.py:

Start line: 1562, End line: 1608

```python
class Model(metaclass=ModelBase):

    @classmethod
    def _check_local_fields(cls, fields, option):
        from django.db import models

        # In order to avoid hitting the relation tree prematurely, we use our
        # own fields_map instead of using get_field()
        forward_fields_map = {
            field.name: field for field in cls._meta._get_fields(reverse=False)
        }

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
### 11 - django/db/models/base.py:

Start line: 1133, End line: 1161

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
### 12 - django/db/models/base.py:

Start line: 544, End line: 560

```python
class Model(metaclass=ModelBase):

    def __setstate__(self, state):
        msg = None
        pickled_version = state.get(DJANGO_VERSION_PICKLE_KEY)
        if pickled_version:
            current_version = get_version()
            if current_version != pickled_version:
                msg = (
                    "Pickled model instance's Django version %s does not match "
                    "the current version %s." % (pickled_version, current_version)
                )
        else:
            msg = "Pickled model instance's Django version is not specified."

        if msg:
            warnings.warn(msg, RuntimeWarning, stacklevel=2)

        self.__dict__.update(state)
```
### 13 - django/db/models/base.py:

Start line: 1777, End line: 1798

```python
class Model(metaclass=ModelBase):

    @classmethod
    def _check_constraints(cls):
        errors = []
        for db in settings.DATABASES:
            if not router.allow_migrate_model(db, cls):
                continue
            connection = connections[db]
            if connection.features.supports_table_check_constraints:
                continue
            if any(isinstance(constraint, CheckConstraint) for constraint in cls._meta.constraints):
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
        return errors
```
### 14 - django/db/models/base.py:

Start line: 1087, End line: 1114

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
### 15 - django/db/models/base.py:

Start line: 500, End line: 542

```python
class Model(metaclass=ModelBase):

    @classmethod
    def from_db(cls, db, field_names, values):
        if len(values) != len(cls._meta.concrete_fields):
            values_iter = iter(values)
            values = [
                next(values_iter) if f.attname in field_names else DEFERRED
                for f in cls._meta.concrete_fields
            ]
        new = cls(*values)
        new._state.adding = False
        new._state.db = db
        return new

    def __repr__(self):
        return '<%s: %s>' % (self.__class__.__name__, self)

    def __str__(self):
        return '%s object (%s)' % (self.__class__.__name__, self.pk)

    def __eq__(self, other):
        if not isinstance(other, Model):
            return False
        if self._meta.concrete_model != other._meta.concrete_model:
            return False
        my_pk = self.pk
        if my_pk is None:
            return self is other
        return my_pk == other.pk

    def __hash__(self):
        if self.pk is None:
            raise TypeError("Model instances without primary key value are unhashable")
        return hash(self.pk)

    def __reduce__(self):
        data = self.__getstate__()
        data[DJANGO_VERSION_PICKLE_KEY] = get_version()
        class_id = self._meta.app_label, self._meta.object_name
        return model_unpickle, (class_id,), data

    def __getstate__(self):
        """Hook to allow choosing the attributes to pickle."""
        return self.__dict__
```
### 16 - django/db/models/base.py:

Start line: 1042, End line: 1085

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
### 17 - django/db/models/base.py:

Start line: 938, End line: 951

```python
class Model(metaclass=ModelBase):

    def _get_next_or_previous_in_order(self, is_next):
        cachename = "__%s_order_cache" % is_next
        if not hasattr(self, cachename):
            op = 'gt' if is_next else 'lt'
            order = '_order' if is_next else '-_order'
            order_field = self._meta.order_with_respect_to
            filter_args = order_field.get_filter_kwargs_for_object(self)
            obj = self.__class__._default_manager.filter(**filter_args).filter(**{
                '_order__%s' % op: self.__class__._default_manager.values('_order').filter(**{
                    self._meta.pk.name: self.pk
                })
            }).order_by(order)[:1].get()
            setattr(self, cachename, obj)
        return getattr(self, cachename)
```
### 19 - django/db/models/base.py:

Start line: 1116, End line: 1131

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
### 20 - django/db/models/base.py:

Start line: 1419, End line: 1442

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
### 21 - django/db/models/base.py:

Start line: 1362, End line: 1417

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
### 22 - django/db/models/base.py:

Start line: 1282, End line: 1311

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
### 23 - django/db/models/base.py:

Start line: 1801, End line: 1852

```python
############################################
# HELPER FUNCTIONS (CURRIED MODEL METHODS) #
############################################

# ORDERING METHODS #########################

def method_set_order(self, ordered_obj, id_list, using=None):
    if using is None:
        using = DEFAULT_DB_ALIAS
    order_wrt = ordered_obj._meta.order_with_respect_to
    filter_args = order_wrt.get_forward_related_filter(self)
    ordered_obj.objects.db_manager(using).filter(**filter_args).bulk_update([
        ordered_obj(pk=pk, _order=order) for order, pk in enumerate(id_list)
    ], ['_order'])


def method_get_order(self, ordered_obj):
    order_wrt = ordered_obj._meta.order_with_respect_to
    filter_args = order_wrt.get_forward_related_filter(self)
    pk_name = ordered_obj._meta.pk.name
    return ordered_obj.objects.filter(**filter_args).values_list(pk_name, flat=True)


def make_foreign_order_accessors(model, related_model):
    setattr(
        related_model,
        'get_%s_order' % model.__name__.lower(),
        partialmethod(method_get_order, model)
    )
    setattr(
        related_model,
        'set_%s_order' % model.__name__.lower(),
        partialmethod(method_set_order, model)
    )

########
# MISC #
########


def model_unpickle(model_id):
    """Used to unpickle Model subclasses with deferred fields."""
    if isinstance(model_id, tuple):
        model = apps.get_model(*model_id)
    else:
        # Backwards compat - the model was cached directly in earlier versions.
        model = model_id
    return model.__new__(model)


model_unpickle.__safe_for_unpickle__ = True
```
### 24 - django/db/models/base.py:

Start line: 1704, End line: 1775

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
### 30 - django/db/models/base.py:

Start line: 1313, End line: 1343

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

Start line: 1502, End line: 1527

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
### 36 - django/db/models/base.py:

Start line: 1199, End line: 1222

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
### 37 - django/db/models/base.py:

Start line: 47, End line: 63

```python
def subclass_exception(name, bases, module, attached_to):
    """
    Create exception subclass. Used by ModelBase below.

    The exception is created in a way that allows it to be pickled, assuming
    that the returned exception class will be added as an attribute to the
    'attached_to' class.
    """
    return type(name, bases, {
        '__module__': module,
        '__qualname__': '%s.%s' % (attached_to.__qualname__, name),
    })


def _has_contribute_to_class(value):
    # Only call contribute_to_class() if it's bound.
    return not inspect.isclass(value) and hasattr(value, 'contribute_to_class')
```
### 41 - django/db/models/base.py:

Start line: 1345, End line: 1360

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
### 42 - django/db/models/base.py:

Start line: 1255, End line: 1280

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
### 44 - django/db/models/base.py:

Start line: 1163, End line: 1197

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
### 45 - django/db/models/base.py:

Start line: 562, End line: 578

```python
class Model(metaclass=ModelBase):

    def _get_pk_val(self, meta=None):
        meta = meta or self._meta
        return getattr(self, meta.pk.attname)

    def _set_pk_val(self, value):
        return setattr(self, self._meta.pk.attname, value)

    pk = property(_get_pk_val, _set_pk_val)

    def get_deferred_fields(self):
        """
        Return a set containing names of deferred fields for this instance.
        """
        return {
            f.attname for f in self._meta.concrete_fields
            if f.attname not in self.__dict__
        }
```
### 47 - django/db/models/base.py:

Start line: 922, End line: 936

```python
class Model(metaclass=ModelBase):

    def _get_next_or_previous_by_FIELD(self, field, is_next, **kwargs):
        if not self.pk:
            raise ValueError("get_next/get_previous cannot be used on unsaved objects.")
        op = 'gt' if is_next else 'lt'
        order = '' if is_next else '-'
        param = getattr(self, field.attname)
        q = Q(**{'%s__%s' % (field.name, op): param})
        q = q | Q(**{field.name: param, 'pk__%s' % op: self.pk})
        qs = self.__class__._default_manager.using(self._state.db).filter(**kwargs).filter(q).order_by(
            '%s%s' % (order, field.name), '%spk' % order
        )
        try:
            return qs[0]
        except IndexError:
            raise self.DoesNotExist("%s matching query does not exist." % self.__class__._meta.object_name)
```
### 48 - django/db/models/base.py:

Start line: 1, End line: 44

```python
import copy
import inspect
import warnings
from functools import partialmethod
from itertools import chain

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
from django.db.models.constants import LOOKUP_SEP
from django.db.models.constraints import CheckConstraint, UniqueConstraint
from django.db.models.deletion import CASCADE, Collector
from django.db.models.fields.related import (
    ForeignObjectRel, OneToOneField, lazy_related_operation, resolve_relation,
)
from django.db.models.manager import Manager
from django.db.models.options import Options
from django.db.models.query import Q
from django.db.models.signals import (
    class_prepared, post_init, post_save, pre_init, pre_save,
)
from django.db.models.utils import make_model_tuple
from django.utils.text import capfirst, get_text_list
from django.utils.translation import gettext_lazy as _
from django.utils.version import get_version


class Deferred:
    def __repr__(self):
        return '<Deferred field>'

    def __str__(self):
        return '<Deferred field>'


DEFERRED = Deferred()
```
### 59 - django/db/models/base.py:

Start line: 375, End line: 391

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
### 61 - django/db/models/base.py:

Start line: 1529, End line: 1560

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

    @classmethod
    def _check_indexes(cls):
        """Check the fields of indexes."""
        fields = [field for index in cls._meta.indexes for field, _ in index.fields_orders]
        return cls._check_local_fields(fields, 'indexes')
```
### 73 - django/db/models/base.py:

Start line: 953, End line: 981

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
### 88 - django/db/models/base.py:

Start line: 788, End line: 816

```python
class Model(metaclass=ModelBase):

    save_base.alters_data = True

    def _save_parents(self, cls, using, update_fields):
        """Save all the parents of cls using values from self."""
        meta = cls._meta
        inserted = False
        for parent, field in meta.parents.items():
            # Make sure the link fields are synced between parent and self.
            if (field and getattr(self, parent._meta.pk.attname) is None and
                    getattr(self, field.attname) is not None):
                setattr(self, parent._meta.pk.attname, getattr(self, field.attname))
            parent_inserted = self._save_parents(cls=parent, using=using, update_fields=update_fields)
            updated = self._save_table(
                cls=parent, using=using, update_fields=update_fields,
                force_insert=parent_inserted,
            )
            if not updated:
                inserted = True
            # Set the parent's PK value to self.
            if field:
                setattr(self, field.attname, self._get_pk_val(parent._meta))
                # Since we didn't have an instance of the parent handy set
                # attname directly, bypassing the descriptor. Invalidate
                # the related object cache, in case it's been accidentally
                # populated. A fresh instance will be re-built from the
                # database if necessary.
                if field.is_cached(self):
                    field.delete_cached_value(self)
        return inserted
```
### 112 - django/db/models/base.py:

Start line: 1610, End line: 1702

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
                    fld = _cls._meta.get_field(part)
                    if fld.is_relation:
                        _cls = fld.get_path_info()[-1].to_opts.model
                except (FieldDoesNotExist, AttributeError):
                    if fld is None or fld.get_transform(part) is None:
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
### 138 - django/db/models/base.py:

Start line: 658, End line: 736

```python
class Model(metaclass=ModelBase):

    def save(self, force_insert=False, force_update=False, using=None,
             update_fields=None):
        """
        Save the current instance. Override this in a subclass if you want to
        control the saving process.

        The 'force_insert' and 'force_update' parameters can be used to insist
        that the "save" must be an SQL insert or update (or equivalent for
        non-SQL backends), respectively. Normally, they should not be set.
        """
        # Ensure that a model instance without a PK hasn't been assigned to
        # a ForeignKey or OneToOneField on this model. If the field is
        # nullable, allowing the save() would result in silent data loss.
        for field in self._meta.concrete_fields:
            # If the related field isn't cached, then an instance hasn't
            # been assigned and there's no need to worry about this check.
            if field.is_relation and field.is_cached(self):
                obj = getattr(self, field.name, None)
                # A pk may have been assigned manually to a model instance not
                # saved to the database (or auto-generated in a case like
                # UUIDField), but we allow the save to proceed and rely on the
                # database to raise an IntegrityError if applicable. If
                # constraints aren't supported by the database, there's the
                # unavoidable risk of data corruption.
                if obj and obj.pk is None:
                    # Remove the object from a related instance cache.
                    if not field.remote_field.multiple:
                        field.remote_field.delete_cached_value(obj)
                    raise ValueError(
                        "save() prohibited to prevent data loss due to "
                        "unsaved related object '%s'." % field.name
                    )
                # If the relationship's pk/to_field was changed, clear the
                # cached relationship.
                if obj and getattr(obj, field.target_field.attname) != getattr(self, field.attname):
                    field.delete_cached_value(self)

        using = using or router.db_for_write(self.__class__, instance=self)
        if force_insert and (force_update or update_fields):
            raise ValueError("Cannot force both insert and updating in model saving.")

        deferred_fields = self.get_deferred_fields()
        if update_fields is not None:
            # If update_fields is empty, skip the save. We do also check for
            # no-op saves later on for inheritance cases. This bailout is
            # still needed for skipping signal sending.
            if not update_fields:
                return

            update_fields = frozenset(update_fields)
            field_names = set()

            for field in self._meta.fields:
                if not field.primary_key:
                    field_names.add(field.name)

                    if field.name != field.attname:
                        field_names.add(field.attname)

            non_model_fields = update_fields.difference(field_names)

            if non_model_fields:
                raise ValueError("The following fields do not exist in this "
                                 "model or are m2m fields: %s"
                                 % ', '.join(non_model_fields))

        # If saving to the same database, and this model is deferred, then
        # automatically do a "update_fields" save on the loaded fields.
        elif not force_insert and deferred_fields and using == self._state.db:
            field_names = set()
            for field in self._meta.concrete_fields:
                if not field.primary_key and not hasattr(field, 'through'):
                    field_names.add(field.attname)
            loaded_fields = field_names.difference(deferred_fields)
            if loaded_fields:
                update_fields = frozenset(loaded_fields)

        self.save_base(using=using, force_insert=force_insert,
                       force_update=force_update, update_fields=update_fields)
```
### 141 - django/db/models/base.py:

Start line: 870, End line: 895

```python
class Model(metaclass=ModelBase):

    def _do_update(self, base_qs, using, pk_val, values, update_fields, forced_update):
        """
        Try to update the model. Return True if the model was updated (if an
        update query was done and a matching row was found in the DB).
        """
        filtered = base_qs.filter(pk=pk_val)
        if not values:
            # We can end up here when saving a model in inheritance chain where
            # update_fields doesn't target any field in current model. In that
            # case we just say the update succeeded. Another case ending up here
            # is a model with just PK - in that case check that the PK still
            # exists.
            return update_fields is not None or filtered.exists()
        if self._meta.select_on_save and not forced_update:
            return (
                filtered.exists() and
                # It may happen that the object is deleted from the DB right after
                # this check, causing the subsequent UPDATE to return zero matching
                # rows. The same result can occur in some rare cases when the
                # database returns zero despite the UPDATE being executed
                # successfully (a row is matched and updated). In order to
                # distinguish these two cases, the object's existence in the
                # database is again checked for if the UPDATE query returns 0.
                (filtered._update(values) > 0 or filtered.exists())
            )
        return filtered._update(values) > 0
```
