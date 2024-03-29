# django__django-12325

| **django/django** | `29c126bb349526b5f1cd78facbe9f25906f18563` |
| ---- | ---- |
| **No of patches** | 2 |
| **All found context length** | 21779 |
| **Any found context length** | 1422 |
| **Avg pos** | 31.0 |
| **Min pos** | 3 |
| **Max pos** | 59 |
| **Top file pos** | 3 |
| **Missing snippets** | 3 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/db/models/base.py b/django/db/models/base.py
--- a/django/db/models/base.py
+++ b/django/db/models/base.py
@@ -202,7 +202,7 @@ def __new__(cls, name, bases, attrs, **kwargs):
                 continue
             # Locate OneToOneField instances.
             for field in base._meta.local_fields:
-                if isinstance(field, OneToOneField):
+                if isinstance(field, OneToOneField) and field.remote_field.parent_link:
                     related = resolve_relation(new_class, field.remote_field.model)
                     parent_links[make_model_tuple(related)] = field
 
diff --git a/django/db/models/options.py b/django/db/models/options.py
--- a/django/db/models/options.py
+++ b/django/db/models/options.py
@@ -5,7 +5,7 @@
 
 from django.apps import apps
 from django.conf import settings
-from django.core.exceptions import FieldDoesNotExist, ImproperlyConfigured
+from django.core.exceptions import FieldDoesNotExist
 from django.db import connections
 from django.db.models import Manager
 from django.db.models.fields import AutoField
@@ -251,10 +251,6 @@ def _prepare(self, model):
                     field = already_created[0]
                 field.primary_key = True
                 self.setup_pk(field)
-                if not field.remote_field.parent_link:
-                    raise ImproperlyConfigured(
-                        'Add parent_link=True to %s.' % field,
-                    )
             else:
                 auto = AutoField(verbose_name='ID', primary_key=True, auto_created=True)
                 model.add_to_class('id', auto)

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/db/models/base.py | 205 | 205 | 59 | 4 | 21779
| django/db/models/options.py | 8 | 8 | 3 | 3 | 1422
| django/db/models/options.py | 254 | 257 | - | 3 | -


## Problem Statement

```
pk setup for MTI to parent get confused by multiple OneToOne references.
Description
	
class Document(models.Model):
	pass
class Picking(Document):
	document_ptr = models.OneToOneField(Document, on_delete=models.CASCADE, parent_link=True, related_name='+')
	origin = models.OneToOneField(Document, related_name='picking', on_delete=models.PROTECT)
produces django.core.exceptions.ImproperlyConfigured: Add parent_link=True to appname.Picking.origin.
class Picking(Document):
	origin = models.OneToOneField(Document, related_name='picking', on_delete=models.PROTECT)
	document_ptr = models.OneToOneField(Document, on_delete=models.CASCADE, parent_link=True, related_name='+')
Works
First issue is that order seems to matter?
Even if ordering is required "by design"(It shouldn't be we have explicit parent_link marker) shouldn't it look from top to bottom like it does with managers and other things?

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 django/db/models/fields/related.py | 1202 | 1313| 939 | 939 | 13512 | 
| 2 | 2 django/db/models/fields/related_descriptors.py | 309 | 323| 182 | 1121 | 23870 | 
| **-> 3 <-** | **3 django/db/models/options.py** | 1 | 36| 301 | 1422 | 30954 | 
| 4 | 3 django/db/models/fields/related_descriptors.py | 120 | 154| 405 | 1827 | 30954 | 
| 5 | **4 django/db/models/base.py** | 1865 | 1916| 351 | 2178 | 46307 | 
| 6 | 4 django/db/models/fields/related_descriptors.py | 672 | 726| 511 | 2689 | 46307 | 
| 7 | 5 docs/_ext/djangodocs.py | 26 | 70| 385 | 3074 | 49352 | 
| 8 | 5 django/db/models/fields/related_descriptors.py | 642 | 671| 254 | 3328 | 49352 | 
| 9 | 5 django/db/models/fields/related_descriptors.py | 365 | 381| 184 | 3512 | 49352 | 
| 10 | 5 django/db/models/fields/related.py | 1556 | 1591| 475 | 3987 | 49352 | 
| 11 | 5 django/db/models/fields/related_descriptors.py | 203 | 275| 740 | 4727 | 49352 | 
| 12 | 5 django/db/models/fields/related_descriptors.py | 344 | 363| 156 | 4883 | 49352 | 
| 13 | 5 django/db/models/fields/related_descriptors.py | 608 | 640| 323 | 5206 | 49352 | 
| 14 | 5 django/db/models/fields/related.py | 994 | 1041| 368 | 5574 | 49352 | 
| 15 | 5 django/db/models/fields/related_descriptors.py | 1012 | 1039| 334 | 5908 | 49352 | 
| 16 | 6 django/db/models/__init__.py | 1 | 52| 591 | 6499 | 49943 | 
| 17 | 6 django/db/models/fields/related_descriptors.py | 728 | 754| 222 | 6721 | 49943 | 
| 18 | 6 django/db/models/fields/related_descriptors.py | 1114 | 1159| 484 | 7205 | 49943 | 
| 19 | 7 django/contrib/contenttypes/fields.py | 677 | 702| 254 | 7459 | 55355 | 
| 20 | 7 django/db/models/fields/related_descriptors.py | 903 | 940| 374 | 7833 | 55355 | 
| 21 | 7 django/contrib/contenttypes/fields.py | 431 | 452| 248 | 8081 | 55355 | 
| 22 | 7 django/db/models/fields/related.py | 1611 | 1645| 266 | 8347 | 55355 | 
| 23 | 7 django/db/models/fields/related_descriptors.py | 942 | 964| 218 | 8565 | 55355 | 
| 24 | **7 django/db/models/base.py** | 404 | 503| 856 | 9421 | 55355 | 
| 25 | 7 django/db/models/fields/related_descriptors.py | 803 | 862| 576 | 9997 | 55355 | 
| 26 | 7 django/db/models/fields/related.py | 1315 | 1387| 616 | 10613 | 55355 | 
| 27 | 7 django/db/models/fields/related_descriptors.py | 551 | 606| 478 | 11091 | 55355 | 
| 28 | 7 django/db/models/fields/related.py | 1389 | 1419| 322 | 11413 | 55355 | 
| 29 | 7 django/contrib/contenttypes/fields.py | 598 | 629| 278 | 11691 | 55355 | 
| 30 | 7 django/db/models/fields/related.py | 847 | 873| 240 | 11931 | 55355 | 
| 31 | 7 django/db/models/fields/related_descriptors.py | 966 | 983| 190 | 12121 | 55355 | 
| 32 | 7 django/db/models/fields/related_descriptors.py | 1161 | 1202| 392 | 12513 | 55355 | 
| 33 | 7 django/db/models/fields/related.py | 1593 | 1609| 286 | 12799 | 55355 | 
| 34 | 7 django/db/models/fields/related_descriptors.py | 494 | 548| 338 | 13137 | 55355 | 
| 35 | 7 django/db/models/fields/related_descriptors.py | 985 | 1011| 248 | 13385 | 55355 | 
| 36 | 7 django/contrib/contenttypes/fields.py | 565 | 596| 308 | 13693 | 55355 | 
| 37 | 7 django/db/models/fields/related.py | 1 | 34| 244 | 13937 | 55355 | 
| 38 | 8 django/db/migrations/autodetector.py | 525 | 671| 1109 | 15046 | 67090 | 
| 39 | 9 django/db/models/sql/datastructures.py | 104 | 115| 133 | 15179 | 68492 | 
| 40 | **9 django/db/models/base.py** | 1 | 50| 330 | 15509 | 68492 | 
| 41 | **9 django/db/models/base.py** | 212 | 322| 866 | 16375 | 68492 | 
| 42 | 10 django/db/models/query.py | 1621 | 1727| 1063 | 17438 | 85517 | 
| 43 | 10 django/contrib/contenttypes/fields.py | 656 | 676| 188 | 17626 | 85517 | 
| 44 | 10 django/db/models/fields/related.py | 255 | 282| 269 | 17895 | 85517 | 
| 45 | 11 django/contrib/admindocs/views.py | 183 | 249| 584 | 18479 | 88825 | 
| 46 | 12 django/contrib/admindocs/utils.py | 115 | 139| 185 | 18664 | 90731 | 
| 47 | 12 django/contrib/contenttypes/fields.py | 630 | 654| 214 | 18878 | 90731 | 
| 48 | 13 django/db/migrations/operations/models.py | 102 | 118| 147 | 19025 | 97427 | 
| 49 | 13 django/db/migrations/autodetector.py | 1182 | 1207| 245 | 19270 | 97427 | 
| 50 | **13 django/db/models/options.py** | 363 | 385| 164 | 19434 | 97427 | 
| 51 | **13 django/db/models/base.py** | 1343 | 1373| 244 | 19678 | 97427 | 
| 52 | 13 django/db/models/fields/related.py | 738 | 756| 222 | 19900 | 97427 | 
| 53 | **13 django/db/models/options.py** | 585 | 611| 197 | 20097 | 97427 | 
| 54 | 13 django/db/models/fields/related.py | 509 | 563| 409 | 20506 | 97427 | 
| 55 | **13 django/db/models/options.py** | 296 | 314| 136 | 20642 | 97427 | 
| 56 | **13 django/db/models/base.py** | 1498 | 1530| 231 | 20873 | 97427 | 
| 57 | 14 django/contrib/admin/migrations/0001_initial.py | 1 | 48| 320 | 21193 | 97747 | 
| 58 | 14 django/db/models/fields/related.py | 1169 | 1200| 180 | 21373 | 97747 | 
| **-> 59 <-** | **14 django/db/models/base.py** | 169 | 211| 406 | 21779 | 97747 | 
| 60 | 14 django/db/migrations/autodetector.py | 437 | 463| 256 | 22035 | 97747 | 
| 61 | **14 django/db/models/base.py** | 802 | 830| 307 | 22342 | 97747 | 
| 62 | **14 django/db/models/options.py** | 387 | 413| 175 | 22517 | 97747 | 
| 63 | 15 django/contrib/admindocs/urls.py | 1 | 51| 307 | 22824 | 98054 | 
| 64 | 16 django/forms/models.py | 886 | 916| 289 | 23113 | 109656 | 
| 65 | 16 django/db/models/fields/related_descriptors.py | 82 | 118| 264 | 23377 | 109656 | 
| 66 | **16 django/db/models/base.py** | 968 | 981| 180 | 23557 | 109656 | 
| 67 | 16 django/db/models/fields/related.py | 896 | 916| 178 | 23735 | 109656 | 
| 68 | 16 django/contrib/admindocs/views.py | 156 | 180| 234 | 23969 | 109656 | 
| 69 | 16 django/db/models/fields/related.py | 1463 | 1487| 295 | 24264 | 109656 | 
| 70 | 16 django/db/models/fields/related.py | 640 | 656| 163 | 24427 | 109656 | 
| 71 | 17 django/contrib/admin/options.py | 2148 | 2183| 315 | 24742 | 128143 | 
| 72 | 17 django/db/models/fields/related.py | 1044 | 1088| 407 | 25149 | 128143 | 
| 73 | **17 django/db/models/base.py** | 1665 | 1763| 717 | 25866 | 128143 | 
| 74 | 17 django/db/migrations/operations/models.py | 609 | 622| 137 | 26003 | 128143 | 
| 75 | 17 django/db/models/fields/related.py | 1421 | 1461| 399 | 26402 | 128143 | 
| 76 | 17 django/db/migrations/autodetector.py | 337 | 356| 196 | 26598 | 128143 | 
| 77 | 17 django/contrib/contenttypes/fields.py | 507 | 563| 439 | 27037 | 128143 | 
| 78 | **17 django/db/models/base.py** | 1392 | 1447| 491 | 27528 | 128143 | 
| 79 | **17 django/db/models/options.py** | 149 | 208| 587 | 28115 | 128143 | 
| 80 | 18 django/db/models/sql/query.py | 984 | 1015| 307 | 28422 | 149812 | 
| 81 | 18 django/contrib/contenttypes/fields.py | 21 | 109| 571 | 28993 | 149812 | 
| 82 | **18 django/db/models/options.py** | 262 | 294| 331 | 29324 | 149812 | 
| 83 | 19 django/db/models/fields/reverse_related.py | 230 | 249| 147 | 29471 | 151955 | 
| 84 | **19 django/db/models/base.py** | 1615 | 1663| 348 | 29819 | 151955 | 
| 85 | 19 django/db/migrations/autodetector.py | 1123 | 1144| 231 | 30050 | 151955 | 
| 86 | **19 django/db/models/base.py** | 952 | 966| 212 | 30262 | 151955 | 
| 87 | 20 django/db/models/fields/related_lookups.py | 62 | 99| 447 | 30709 | 153404 | 
| 88 | 21 django/contrib/flatpages/migrations/0001_initial.py | 1 | 40| 307 | 31016 | 153711 | 
| 89 | 22 django/db/models/deletion.py | 1 | 76| 566 | 31582 | 157370 | 
| 90 | 22 django/contrib/admin/options.py | 1466 | 1489| 319 | 31901 | 157370 | 
| 91 | 23 django/db/models/manager.py | 1 | 162| 1223 | 33124 | 158804 | 
| 92 | **23 django/db/models/base.py** | 1254 | 1283| 242 | 33366 | 158804 | 
| 93 | 24 django/contrib/redirects/migrations/0001_initial.py | 1 | 41| 274 | 33640 | 159078 | 
| 94 | 24 django/contrib/admindocs/views.py | 250 | 315| 585 | 34225 | 159078 | 
| 95 | 24 django/contrib/admindocs/utils.py | 86 | 112| 175 | 34400 | 159078 | 
| 96 | **24 django/db/models/options.py** | 65 | 147| 668 | 35068 | 159078 | 
| 97 | 25 django/db/migrations/exceptions.py | 1 | 55| 250 | 35318 | 159329 | 
| 98 | 26 django/contrib/admin/utils.py | 159 | 185| 239 | 35557 | 163446 | 
| 99 | 27 django/contrib/auth/migrations/0001_initial.py | 1 | 105| 849 | 36406 | 164295 | 
| 100 | **27 django/db/models/base.py** | 1474 | 1496| 171 | 36577 | 164295 | 
| 101 | 27 django/db/models/sql/datastructures.py | 117 | 137| 144 | 36721 | 164295 | 
| 102 | **27 django/db/models/base.py** | 1072 | 1115| 404 | 37125 | 164295 | 
| 103 | 27 django/contrib/admindocs/views.py | 1 | 30| 223 | 37348 | 164295 | 
| 104 | 27 django/db/models/fields/related_lookups.py | 46 | 60| 224 | 37572 | 164295 | 
| 105 | 28 django/conf/__init__.py | 1 | 40| 240 | 37812 | 166354 | 
| 106 | 29 django/db/migrations/state.py | 1 | 24| 191 | 38003 | 171572 | 
| 107 | 29 django/db/models/fields/related_descriptors.py | 1 | 79| 683 | 38686 | 171572 | 
| 108 | 29 django/db/models/fields/related.py | 824 | 845| 169 | 38855 | 171572 | 
| 109 | 29 django/db/models/fields/related_descriptors.py | 1085 | 1112| 338 | 39193 | 171572 | 
| 110 | **29 django/db/models/options.py** | 685 | 719| 378 | 39571 | 171572 | 
| 111 | 29 django/forms/models.py | 918 | 950| 353 | 39924 | 171572 | 
| 112 | 29 django/db/models/fields/related.py | 565 | 598| 318 | 40242 | 171572 | 
| 113 | **29 django/db/models/options.py** | 721 | 736| 144 | 40386 | 171572 | 
| 114 | 29 django/db/models/fields/related_descriptors.py | 156 | 201| 445 | 40831 | 171572 | 
| 115 | 29 django/db/models/query.py | 1842 | 1874| 314 | 41145 | 171572 | 
| 116 | **29 django/db/models/base.py** | 505 | 547| 347 | 41492 | 171572 | 
| 117 | 29 django/db/models/fields/related.py | 952 | 962| 121 | 41613 | 171572 | 
| 118 | 29 django/db/migrations/state.py | 601 | 612| 136 | 41749 | 171572 | 
| 119 | 29 django/contrib/admin/options.py | 422 | 465| 350 | 42099 | 171572 | 
| 120 | 29 django/db/models/fields/related.py | 600 | 614| 186 | 42285 | 171572 | 
| 121 | **29 django/db/models/options.py** | 415 | 437| 154 | 42439 | 171572 | 
| 122 | 29 django/db/models/manager.py | 165 | 202| 211 | 42650 | 171572 | 
| 123 | 30 django/core/serializers/xml_serializer.py | 111 | 147| 330 | 42980 | 174966 | 
| 124 | 31 django/contrib/contenttypes/migrations/0001_initial.py | 1 | 35| 207 | 43187 | 175173 | 
| 125 | 32 django/contrib/admin/checks.py | 1019 | 1046| 204 | 43391 | 184221 | 
| 126 | 32 django/forms/models.py | 679 | 750| 732 | 44123 | 184221 | 
| 127 | 32 django/db/models/fields/related_descriptors.py | 864 | 878| 190 | 44313 | 184221 | 
| 128 | 32 django/db/models/fields/related_descriptors.py | 880 | 901| 199 | 44512 | 184221 | 
| 129 | 33 django/db/migrations/operations/utils.py | 1 | 14| 138 | 44650 | 184701 | 
| 130 | **33 django/db/models/base.py** | 751 | 800| 456 | 45106 | 184701 | 
| 131 | 33 django/contrib/contenttypes/fields.py | 357 | 395| 405 | 45511 | 184701 | 
| 132 | 33 django/forms/models.py | 628 | 645| 167 | 45678 | 184701 | 
| 133 | 34 django/db/backends/base/schema.py | 370 | 384| 182 | 45860 | 196016 | 
| 134 | 34 django/forms/models.py | 382 | 410| 240 | 46100 | 196016 | 
| 135 | 34 django/db/migrations/operations/models.py | 1 | 38| 238 | 46338 | 196016 | 
| 136 | 34 django/db/models/query.py | 1938 | 1961| 200 | 46538 | 196016 | 
| 137 | 34 django/db/models/fields/related.py | 171 | 188| 166 | 46704 | 196016 | 


### Hint

```
That seems to be a bug, managed to reproduce. Does the error go away if you add primary_key=True to document_ptr like I assume you wanted to do? This makes me realized that the ​MTI documentation is not completely correcy, ​the automatically added `place_ptr` field end up with `primary_key=True`. Not sure why we're not checking and field.remote_field.parent_link on ​parent links connection.
Replying to Simon Charette: It does go away primary_key. Why is parent_link even necessary in that case? Having pk OneToOne with to MTI child should imply it's parent link.
Replying to Mārtiņš Šulcs: Replying to Simon Charette: It does go away primary_key. Why is parent_link even necessary in that case? Having pk OneToOne with to MTI child should imply it's parent link. I have an update. The warning goes away with primary_key, but model itself is still broken(complains about document_ptr_id not populated) unless field order is correct.
Been able to replicate this bug with a simpler case: class Document(models.Model): pass class Picking(Document): some_unrelated_document = models.OneToOneField(Document, related_name='something', on_delete=models.PROTECT) Produces the same error against some_unrelated_document.
Hi, Could this approach be appropriate? parent_links's params can be base and related class instance therefore just last sample columns are added into parent_links. We can extend key logic with related_name, e.g ('app', 'document', 'picking'). Or using this method, we can always ensure that the parent_link=True field is guaranteed into self.parents. ​PR +++ b/django/db/models/base.py @@ -196,10 +196,11 @@ class ModelBase(type): if base != new_class and not base._meta.abstract: continue # Locate OneToOneField instances. - for field in base._meta.local_fields: - if isinstance(field, OneToOneField): - related = resolve_relation(new_class, field.remote_field.model) - parent_links[make_model_tuple(related)] = field + fields = [field for field in base._meta.local_fields if isinstance(field, OneToOneField)] + for field in sorted(fields, key=lambda x: x.remote_field.parent_link, reverse=True): + related_key = make_model_tuple(resolve_relation(new_class, field.remote_field.model)) + if related_key not in parent_links: + parent_links[related_key] = field # Track fields inherited from base models. inherited_attributes = set()
What's the status on this? I'm encountering the same issue.
```

## Patch

```diff
diff --git a/django/db/models/base.py b/django/db/models/base.py
--- a/django/db/models/base.py
+++ b/django/db/models/base.py
@@ -202,7 +202,7 @@ def __new__(cls, name, bases, attrs, **kwargs):
                 continue
             # Locate OneToOneField instances.
             for field in base._meta.local_fields:
-                if isinstance(field, OneToOneField):
+                if isinstance(field, OneToOneField) and field.remote_field.parent_link:
                     related = resolve_relation(new_class, field.remote_field.model)
                     parent_links[make_model_tuple(related)] = field
 
diff --git a/django/db/models/options.py b/django/db/models/options.py
--- a/django/db/models/options.py
+++ b/django/db/models/options.py
@@ -5,7 +5,7 @@
 
 from django.apps import apps
 from django.conf import settings
-from django.core.exceptions import FieldDoesNotExist, ImproperlyConfigured
+from django.core.exceptions import FieldDoesNotExist
 from django.db import connections
 from django.db.models import Manager
 from django.db.models.fields import AutoField
@@ -251,10 +251,6 @@ def _prepare(self, model):
                     field = already_created[0]
                 field.primary_key = True
                 self.setup_pk(field)
-                if not field.remote_field.parent_link:
-                    raise ImproperlyConfigured(
-                        'Add parent_link=True to %s.' % field,
-                    )
             else:
                 auto = AutoField(verbose_name='ID', primary_key=True, auto_created=True)
                 model.add_to_class('id', auto)

```

## Test Patch

```diff
diff --git a/tests/invalid_models_tests/test_models.py b/tests/invalid_models_tests/test_models.py
--- a/tests/invalid_models_tests/test_models.py
+++ b/tests/invalid_models_tests/test_models.py
@@ -3,7 +3,6 @@
 from django.conf import settings
 from django.core.checks import Error, Warning
 from django.core.checks.model_checks import _check_lazy_references
-from django.core.exceptions import ImproperlyConfigured
 from django.db import connection, connections, models
 from django.db.models.functions import Lower
 from django.db.models.signals import post_init
@@ -1006,14 +1005,24 @@ class ShippingMethodPrice(models.Model):
 
         self.assertEqual(ShippingMethod.check(), [])
 
-    def test_missing_parent_link(self):
-        msg = 'Add parent_link=True to invalid_models_tests.ParkingLot.parent.'
-        with self.assertRaisesMessage(ImproperlyConfigured, msg):
-            class Place(models.Model):
-                pass
+    def test_onetoone_with_parent_model(self):
+        class Place(models.Model):
+            pass
+
+        class ParkingLot(Place):
+            other_place = models.OneToOneField(Place, models.CASCADE, related_name='other_parking')
+
+        self.assertEqual(ParkingLot.check(), [])
+
+    def test_onetoone_with_explicit_parent_link_parent_model(self):
+        class Place(models.Model):
+            pass
+
+        class ParkingLot(Place):
+            place = models.OneToOneField(Place, models.CASCADE, parent_link=True, primary_key=True)
+            other_place = models.OneToOneField(Place, models.CASCADE, related_name='other_parking')
 
-            class ParkingLot(Place):
-                parent = models.OneToOneField(Place, models.CASCADE)
+        self.assertEqual(ParkingLot.check(), [])
 
     def test_m2m_table_name_clash(self):
         class Foo(models.Model):
diff --git a/tests/invalid_models_tests/test_relative_fields.py b/tests/invalid_models_tests/test_relative_fields.py
--- a/tests/invalid_models_tests/test_relative_fields.py
+++ b/tests/invalid_models_tests/test_relative_fields.py
@@ -1291,6 +1291,33 @@ class Model(models.Model):
             ),
         ])
 
+    def test_clash_parent_link(self):
+        class Parent(models.Model):
+            pass
+
+        class Child(Parent):
+            other_parent = models.OneToOneField(Parent, models.CASCADE)
+
+        errors = [
+            ('fields.E304', 'accessor', 'parent_ptr', 'other_parent'),
+            ('fields.E305', 'query name', 'parent_ptr', 'other_parent'),
+            ('fields.E304', 'accessor', 'other_parent', 'parent_ptr'),
+            ('fields.E305', 'query name', 'other_parent', 'parent_ptr'),
+        ]
+        self.assertEqual(Child.check(), [
+            Error(
+                "Reverse %s for 'Child.%s' clashes with reverse %s for "
+                "'Child.%s'." % (attr, field_name, attr, clash_name),
+                hint=(
+                    "Add or change a related_name argument to the definition "
+                    "for 'Child.%s' or 'Child.%s'." % (field_name, clash_name)
+                ),
+                obj=Child._meta.get_field(field_name),
+                id=error_id,
+            )
+            for error_id, attr, field_name, clash_name in errors
+        ])
+
 
 @isolate_apps('invalid_models_tests')
 class M2mThroughFieldsTests(SimpleTestCase):
diff --git a/tests/migrations/test_state.py b/tests/migrations/test_state.py
--- a/tests/migrations/test_state.py
+++ b/tests/migrations/test_state.py
@@ -345,6 +345,7 @@ def test_render(self):
                     'migrations.Tag',
                     models.CASCADE,
                     auto_created=True,
+                    parent_link=True,
                     primary_key=True,
                     to_field='id',
                     serialize=False,

```


## Code snippets

### 1 - django/db/models/fields/related.py:

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
### 2 - django/db/models/fields/related_descriptors.py:

Start line: 309, End line: 323

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
                field for field in opts.concrete_fields
                if field.primary_key and field.remote_field
            ]
            for field in inherited_pk_fields:
                rel_model_pk_name = field.remote_field.model._meta.pk.attname
                raw_value = getattr(value, rel_model_pk_name) if value is not None else None
                setattr(instance, rel_model_pk_name, raw_value)
```
### 3 - django/db/models/options.py:

Start line: 1, End line: 36

```python
import bisect
import copy
import inspect
from collections import defaultdict

from django.apps import apps
from django.conf import settings
from django.core.exceptions import FieldDoesNotExist, ImproperlyConfigured
from django.db import connections
from django.db.models import Manager
from django.db.models.fields import AutoField
from django.db.models.fields.proxy import OrderWrt
from django.db.models.query_utils import PathInfo
from django.utils.datastructures import ImmutableList, OrderedSet
from django.utils.functional import cached_property
from django.utils.text import camel_case_to_spaces, format_lazy
from django.utils.translation import override

PROXY_PARENTS = object()

EMPTY_RELATION_TREE = ()

IMMUTABLE_WARNING = (
    "The return type of '%s' should never be mutated. If you want to manipulate this list "
    "for your own use, make a copy first."
)

DEFAULT_NAMES = (
    'verbose_name', 'verbose_name_plural', 'db_table', 'ordering',
    'unique_together', 'permissions', 'get_latest_by', 'order_with_respect_to',
    'app_label', 'db_tablespace', 'abstract', 'managed', 'proxy', 'swappable',
    'auto_created', 'index_together', 'apps', 'default_permissions',
    'select_on_save', 'default_related_name', 'required_db_features',
    'required_db_vendor', 'base_manager_name', 'default_manager_name',
    'indexes', 'constraints',
)
```
### 4 - django/db/models/fields/related_descriptors.py:

Start line: 120, End line: 154

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
            query = {'%s__in' % related_field.name: {instance_attr(inst)[0] for inst in instances}}
        else:
            query = {'%s__in' % self.field.related_query_name(): instances}
        queryset = queryset.filter(**query)

        # Since we're going to assign directly in the cache,
        # we must manage the reverse relation cache manually.
        if not remote_field.multiple:
            for rel_obj in queryset:
                instance = instances_dict[rel_obj_attr(rel_obj)]
                remote_field.set_cached_value(rel_obj, instance)
        return queryset, rel_obj_attr, instance_attr, True, self.field.get_cache_name(), False

    def get_object(self, instance):
        qs = self.get_queryset(instance=instance)
        # Assuming the database enforces foreign keys, this won't fail.
        return qs.get(self.field.get_reverse_related_filter(instance))
```
### 5 - django/db/models/base.py:

Start line: 1865, End line: 1916

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
### 6 - django/db/models/fields/related_descriptors.py:

Start line: 672, End line: 726

```python
def create_reverse_many_to_one_manager(superclass, rel):

    class RelatedManager(superclass):
        add.alters_data = True

        def create(self, **kwargs):
            kwargs[self.field.name] = self.instance
            db = router.db_for_write(self.model, instance=self.instance)
            return super(RelatedManager, self.db_manager(db)).create(**kwargs)
        create.alters_data = True

        def get_or_create(self, **kwargs):
            kwargs[self.field.name] = self.instance
            db = router.db_for_write(self.model, instance=self.instance)
            return super(RelatedManager, self.db_manager(db)).get_or_create(**kwargs)
        get_or_create.alters_data = True

        def update_or_create(self, **kwargs):
            kwargs[self.field.name] = self.instance
            db = router.db_for_write(self.model, instance=self.instance)
            return super(RelatedManager, self.db_manager(db)).update_or_create(**kwargs)
        update_or_create.alters_data = True

        # remove() and clear() are only provided if the ForeignKey can have a value of null.
        if rel.field.null:
            def remove(self, *objs, bulk=True):
                if not objs:
                    return
                val = self.field.get_foreign_related_value(self.instance)
                old_ids = set()
                for obj in objs:
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
### 7 - docs/_ext/djangodocs.py:

Start line: 26, End line: 70

```python
def setup(app):
    app.add_crossref_type(
        directivename="setting",
        rolename="setting",
        indextemplate="pair: %s; setting",
    )
    app.add_crossref_type(
        directivename="templatetag",
        rolename="ttag",
        indextemplate="pair: %s; template tag"
    )
    app.add_crossref_type(
        directivename="templatefilter",
        rolename="tfilter",
        indextemplate="pair: %s; template filter"
    )
    app.add_crossref_type(
        directivename="fieldlookup",
        rolename="lookup",
        indextemplate="pair: %s; field lookup type",
    )
    app.add_object_type(
        directivename="django-admin",
        rolename="djadmin",
        indextemplate="pair: %s; django-admin command",
        parse_node=parse_django_admin_node,
    )
    app.add_directive('django-admin-option', Cmdoption)
    app.add_config_value('django_next_version', '0.0', True)
    app.add_directive('versionadded', VersionDirective)
    app.add_directive('versionchanged', VersionDirective)
    app.add_builder(DjangoStandaloneHTMLBuilder)
    app.set_translator('djangohtml', DjangoHTMLTranslator)
    app.set_translator('json', DjangoHTMLTranslator)
    app.add_node(
        ConsoleNode,
        html=(visit_console_html, None),
        latex=(visit_console_dummy, depart_console_dummy),
        man=(visit_console_dummy, depart_console_dummy),
        text=(visit_console_dummy, depart_console_dummy),
        texinfo=(visit_console_dummy, depart_console_dummy),
    )
    app.add_directive('console', ConsoleDirective)
    app.connect('html-page-context', html_page_context_hook)
    return {'parallel_read_safe': True}
```
### 8 - django/db/models/fields/related_descriptors.py:

Start line: 642, End line: 671

```python
def create_reverse_many_to_one_manager(superclass, rel):

    class RelatedManager(superclass):

        def add(self, *objs, bulk=True):
            self._remove_prefetched_objects()
            objs = list(objs)
            db = router.db_for_write(self.model, instance=self.instance)

            def check_and_update_obj(obj):
                if not isinstance(obj, self.model):
                    raise TypeError("'%s' instance expected, got %r" % (
                        self.model._meta.object_name, obj,
                    ))
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
                self.model._base_manager.using(db).filter(pk__in=pks).update(**{
                    self.field.name: self.instance,
                })
            else:
                with transaction.atomic(using=db, savepoint=False):
                    for obj in objs:
                        check_and_update_obj(obj)
                        obj.save()
    # ... other code
```
### 9 - django/db/models/fields/related_descriptors.py:

Start line: 365, End line: 381

```python
class ReverseOneToOneDescriptor:

    def get_prefetch_queryset(self, instances, queryset=None):
        if queryset is None:
            queryset = self.get_queryset()
        queryset._add_hints(instance=instances[0])

        rel_obj_attr = self.related.field.get_local_related_value
        instance_attr = self.related.field.get_foreign_related_value
        instances_dict = {instance_attr(inst): inst for inst in instances}
        query = {'%s__in' % self.related.field.name: instances}
        queryset = queryset.filter(**query)

        # Since we're going to assign directly in the cache,
        # we must manage the reverse relation cache manually.
        for rel_obj in queryset:
            instance = instances_dict[rel_obj_attr(rel_obj)]
            self.related.field.set_cached_value(rel_obj, instance)
        return queryset, rel_obj_attr, instance_attr, True, self.related.get_cache_name(), False
```
### 10 - django/db/models/fields/related.py:

Start line: 1556, End line: 1591

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
                self.remote_field.model == "self" or self.remote_field.model == cls._meta.object_name):
            self.remote_field.related_name = "%s_rel_+" % name
        elif self.remote_field.is_hidden():
            # If the backwards relation is disabled, replace the original
            # related_name with one generated from the m2m field name. Django
            # still uses backwards relations internally and we need to avoid
            # clashes between multiple m2m fields with related_name == '+'.
            self.remote_field.related_name = "_%s_%s_+" % (cls.__name__.lower(), name)

        super().contribute_to_class(cls, name, **kwargs)

        # The intermediate m2m model is not auto created if:
        #  1) There is a manually specified intermediate, or
        #  2) The class owning the m2m field is abstract.
        #  3) The class owning the m2m field has been swapped out.
        if not cls._meta.abstract:
            if self.remote_field.through:
                def resolve_through_model(_, model, field):
                    field.remote_field.through = model
                lazy_related_operation(resolve_through_model, cls, self.remote_field.through, field=self)
            elif not cls._meta.swapped:
                self.remote_field.through = create_many_to_many_intermediary_model(self, cls)

        # Add the descriptor for the m2m relation.
        setattr(cls, self.name, ManyToManyDescriptor(self.remote_field, reverse=False))

        # Set up the accessor for the m2m table name for the relation.
        self.m2m_db_table = partial(self._get_m2m_db_table, cls._meta)
```
### 24 - django/db/models/base.py:

Start line: 404, End line: 503

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
### 40 - django/db/models/base.py:

Start line: 1, End line: 50

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
from django.utils.version import get_version


class Deferred:
    def __repr__(self):
        return '<Deferred field>'

    def __str__(self):
        return '<Deferred field>'


DEFERRED = Deferred()
```
### 41 - django/db/models/base.py:

Start line: 212, End line: 322

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
### 50 - django/db/models/options.py:

Start line: 363, End line: 385

```python
class Options:

    @cached_property
    def managers(self):
        managers = []
        seen_managers = set()
        bases = (b for b in self.model.mro() if hasattr(b, '_meta'))
        for depth, base in enumerate(bases):
            for manager in base._meta.local_managers:
                if manager.name in seen_managers:
                    continue

                manager = copy.copy(manager)
                manager.model = self.model
                seen_managers.add(manager.name)
                managers.append((depth, manager.creation_counter, manager))

        return make_immutable_fields_list(
            "managers",
            (m[2] for m in sorted(managers)),
        )

    @cached_property
    def managers_map(self):
        return {manager.name: manager for manager in self.managers}
```
### 51 - django/db/models/base.py:

Start line: 1343, End line: 1373

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
### 53 - django/db/models/options.py:

Start line: 585, End line: 611

```python
class Options:

    def get_base_chain(self, model):
        """
        Return a list of parent classes leading to `model` (ordered from
        closest to most distant ancestor). This has to handle the case where
        `model` is a grandparent or even more distant relation.
        """
        if not self.parents:
            return []
        if model in self.parents:
            return [model]
        for parent in self.parents:
            res = parent._meta.get_base_chain(model)
            if res:
                res.insert(0, parent)
                return res
        return []

    def get_parent_list(self):
        """
        Return all the ancestors of this model as a list ordered by MRO.
        Useful for determining if something is an ancestor, regardless of lineage.
        """
        result = OrderedSet(self.parents)
        for parent in self.parents:
            for ancestor in parent._meta.get_parent_list():
                result.add(ancestor)
        return list(result)
```
### 55 - django/db/models/options.py:

Start line: 296, End line: 314

```python
class Options:

    def setup_pk(self, field):
        if not self.pk and field.primary_key:
            self.pk = field
            field.serialize = False

    def setup_proxy(self, target):
        """
        Do the internal setup so that the current model is a proxy for
        "target".
        """
        self.pk = target._meta.pk
        self.proxy_for_model = target
        self.db_table = target._meta.db_table

    def __repr__(self):
        return '<Options for %s>' % self.object_name

    def __str__(self):
        return "%s.%s" % (self.app_label, self.model_name)
```
### 56 - django/db/models/base.py:

Start line: 1498, End line: 1530

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
### 59 - django/db/models/base.py:

Start line: 169, End line: 211

```python
class ModelBase(type):
    def __new__(cls, name, bases, attrs, **kwargs):
        # ... other code
        field_names = {f.name for f in new_fields}

        # Basic setup for proxy models.
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
### 61 - django/db/models/base.py:

Start line: 802, End line: 830

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
### 62 - django/db/models/options.py:

Start line: 387, End line: 413

```python
class Options:

    @cached_property
    def base_manager(self):
        base_manager_name = self.base_manager_name
        if not base_manager_name:
            # Get the first parent's base_manager_name if there's one.
            for parent in self.model.mro()[1:]:
                if hasattr(parent, '_meta'):
                    if parent._base_manager.name != '_base_manager':
                        base_manager_name = parent._base_manager.name
                    break

        if base_manager_name:
            try:
                return self.managers_map[base_manager_name]
            except KeyError:
                raise ValueError(
                    "%s has no manager named %r" % (
                        self.object_name,
                        base_manager_name,
                    )
                )

        manager = Manager()
        manager.name = '_base_manager'
        manager.model = self.model
        manager.auto_created = True
        return manager
```
### 66 - django/db/models/base.py:

Start line: 968, End line: 981

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
### 73 - django/db/models/base.py:

Start line: 1665, End line: 1763

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
### 78 - django/db/models/base.py:

Start line: 1392, End line: 1447

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
### 79 - django/db/models/options.py:

Start line: 149, End line: 208

```python
class Options:

    def contribute_to_class(self, cls, name):
        from django.db import connection
        from django.db.backends.utils import truncate_name

        cls._meta = self
        self.model = cls
        # First, construct the default values for these options.
        self.object_name = cls.__name__
        self.model_name = self.object_name.lower()
        self.verbose_name = camel_case_to_spaces(self.object_name)

        # Store the original user-defined values for each option,
        # for use when serializing the model definition
        self.original_attrs = {}

        # Next, apply any overridden values from 'class Meta'.
        if self.meta:
            meta_attrs = self.meta.__dict__.copy()
            for name in self.meta.__dict__:
                # Ignore any private attributes that Django doesn't care about.
                # NOTE: We can't modify a dictionary's contents while looping
                # over it, so we loop over the *original* dictionary instead.
                if name.startswith('_'):
                    del meta_attrs[name]
            for attr_name in DEFAULT_NAMES:
                if attr_name in meta_attrs:
                    setattr(self, attr_name, meta_attrs.pop(attr_name))
                    self.original_attrs[attr_name] = getattr(self, attr_name)
                elif hasattr(self.meta, attr_name):
                    setattr(self, attr_name, getattr(self.meta, attr_name))
                    self.original_attrs[attr_name] = getattr(self, attr_name)

            self.unique_together = normalize_together(self.unique_together)
            self.index_together = normalize_together(self.index_together)
            # App label/class name interpolation for names of constraints and
            # indexes.
            if not getattr(cls._meta, 'abstract', False):
                for attr_name in {'constraints', 'indexes'}:
                    objs = getattr(self, attr_name, [])
                    setattr(self, attr_name, self._format_names_with_class(cls, objs))

            # verbose_name_plural is a special case because it uses a 's'
            # by default.
            if self.verbose_name_plural is None:
                self.verbose_name_plural = format_lazy('{}s', self.verbose_name)

            # order_with_respect_and ordering are mutually exclusive.
            self._ordering_clash = bool(self.ordering and self.order_with_respect_to)

            # Any leftover attributes must be invalid.
            if meta_attrs != {}:
                raise TypeError("'class Meta' got invalid attribute(s): %s" % ','.join(meta_attrs))
        else:
            self.verbose_name_plural = format_lazy('{}s', self.verbose_name)
        del self.meta

        # If the db_table wasn't provided, use the app_label + model_name.
        if not self.db_table:
            self.db_table = "%s_%s" % (self.app_label, self.model_name)
            self.db_table = truncate_name(self.db_table, connection.ops.max_name_length())
```
### 82 - django/db/models/options.py:

Start line: 262, End line: 294

```python
class Options:

    def add_manager(self, manager):
        self.local_managers.append(manager)
        self._expire_cache()

    def add_field(self, field, private=False):
        # Insert the given field in the order in which it was created, using
        # the "creation_counter" attribute of the field.
        # Move many-to-many related fields from self.fields into
        # self.many_to_many.
        if private:
            self.private_fields.append(field)
        elif field.is_relation and field.many_to_many:
            bisect.insort(self.local_many_to_many, field)
        else:
            bisect.insort(self.local_fields, field)
            self.setup_pk(field)

        # If the field being added is a relation to another known field,
        # expire the cache on this field and the forward cache on the field
        # being referenced, because there will be new relationships in the
        # cache. Otherwise, expire the cache of references *to* this field.
        # The mechanism for getting at the related model is slightly odd -
        # ideally, we'd just ask for field.related_model. However, related_model
        # is a cached property, and all the models haven't been loaded yet, so
        # we need to make sure we don't cache a string reference.
        if field.is_relation and hasattr(field.remote_field, 'model') and field.remote_field.model:
            try:
                field.remote_field.model._meta._expire_cache(forward=False)
            except AttributeError:
                pass
            self._expire_cache()
        else:
            self._expire_cache(reverse=False)
```
### 84 - django/db/models/base.py:

Start line: 1615, End line: 1663

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
### 86 - django/db/models/base.py:

Start line: 952, End line: 966

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
### 92 - django/db/models/base.py:

Start line: 1254, End line: 1283

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
### 96 - django/db/models/options.py:

Start line: 65, End line: 147

```python
class Options:
    FORWARD_PROPERTIES = {
        'fields', 'many_to_many', 'concrete_fields', 'local_concrete_fields',
        '_forward_fields_map', 'managers', 'managers_map', 'base_manager',
        'default_manager',
    }
    REVERSE_PROPERTIES = {'related_objects', 'fields_map', '_relation_tree'}

    default_apps = apps

    def __init__(self, meta, app_label=None):
        self._get_fields_cache = {}
        self.local_fields = []
        self.local_many_to_many = []
        self.private_fields = []
        self.local_managers = []
        self.base_manager_name = None
        self.default_manager_name = None
        self.model_name = None
        self.verbose_name = None
        self.verbose_name_plural = None
        self.db_table = ''
        self.ordering = []
        self._ordering_clash = False
        self.indexes = []
        self.constraints = []
        self.unique_together = []
        self.index_together = []
        self.select_on_save = False
        self.default_permissions = ('add', 'change', 'delete', 'view')
        self.permissions = []
        self.object_name = None
        self.app_label = app_label
        self.get_latest_by = None
        self.order_with_respect_to = None
        self.db_tablespace = settings.DEFAULT_TABLESPACE
        self.required_db_features = []
        self.required_db_vendor = None
        self.meta = meta
        self.pk = None
        self.auto_field = None
        self.abstract = False
        self.managed = True
        self.proxy = False
        # For any class that is a proxy (including automatically created
        # classes for deferred object loading), proxy_for_model tells us
        # which class this model is proxying. Note that proxy_for_model
        # can create a chain of proxy models. For non-proxy models, the
        # variable is always None.
        self.proxy_for_model = None
        # For any non-abstract class, the concrete class is the model
        # in the end of the proxy_for_model chain. In particular, for
        # concrete models, the concrete_model is always the class itself.
        self.concrete_model = None
        self.swappable = None
        self.parents = {}
        self.auto_created = False

        # List of all lookups defined in ForeignKey 'limit_choices_to' options
        # from *other* models. Needed for some admin checks. Internal use only.
        self.related_fkey_lookups = []

        # A custom app registry to use, if you're making a separate model set.
        self.apps = self.default_apps

        self.default_related_name = None

    @property
    def label(self):
        return '%s.%s' % (self.app_label, self.object_name)

    @property
    def label_lower(self):
        return '%s.%s' % (self.app_label, self.model_name)

    @property
    def app_config(self):
        # Don't go through get_app_config to avoid triggering imports.
        return self.apps.app_configs.get(self.app_label)

    @property
    def installed(self):
        return self.app_config is not None
```
### 100 - django/db/models/base.py:

Start line: 1474, End line: 1496

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
### 102 - django/db/models/base.py:

Start line: 1072, End line: 1115

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
### 110 - django/db/models/options.py:

Start line: 685, End line: 719

```python
class Options:

    def _populate_directed_relation_graph(self):
        """
        This method is used by each model to find its reverse objects. As this
        method is very expensive and is accessed frequently (it looks up every
        field in a model, in every app), it is computed on first access and then
        is set as a property on every model.
        """
        related_objects_graph = defaultdict(list)

        all_models = self.apps.get_models(include_auto_created=True)
        for model in all_models:
            opts = model._meta
            # Abstract model's fields are copied to child models, hence we will
            # see the fields from the child models.
            if opts.abstract:
                continue
            fields_with_relations = (
                f for f in opts._get_fields(reverse=False, include_parents=False)
                if f.is_relation and f.related_model is not None
            )
            for f in fields_with_relations:
                if not isinstance(f.remote_field.model, str):
                    related_objects_graph[f.remote_field.model._meta.concrete_model._meta].append(f)

        for model in all_models:
            # Set the relation_tree using the internal __dict__. In this way
            # we avoid calling the cached property. In attribute lookup,
            # __dict__ takes precedence over a data descriptor (such as
            # @cached_property). This means that the _meta._relation_tree is
            # only called if related_objects is not in __dict__.
            related_objects = related_objects_graph[model._meta.concrete_model._meta]
            model._meta.__dict__['_relation_tree'] = related_objects
        # It seems it is possible that self is not in all_models, so guard
        # against that with default for get().
        return self.__dict__.get('_relation_tree', EMPTY_RELATION_TREE)
```
### 113 - django/db/models/options.py:

Start line: 721, End line: 736

```python
class Options:

    @cached_property
    def _relation_tree(self):
        return self._populate_directed_relation_graph()

    def _expire_cache(self, forward=True, reverse=True):
        # This method is usually called by apps.cache_clear(), when the
        # registry is finalized, or when a new field is added.
        if forward:
            for cache_key in self.FORWARD_PROPERTIES:
                if cache_key in self.__dict__:
                    delattr(self, cache_key)
        if reverse and not self.abstract:
            for cache_key in self.REVERSE_PROPERTIES:
                if cache_key in self.__dict__:
                    delattr(self, cache_key)
        self._get_fields_cache = {}
```
### 116 - django/db/models/base.py:

Start line: 505, End line: 547

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
            return NotImplemented
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
### 121 - django/db/models/options.py:

Start line: 415, End line: 437

```python
class Options:

    @cached_property
    def default_manager(self):
        default_manager_name = self.default_manager_name
        if not default_manager_name and not self.local_managers:
            # Get the first parent's default_manager_name if there's one.
            for parent in self.model.mro()[1:]:
                if hasattr(parent, '_meta'):
                    default_manager_name = parent._meta.default_manager_name
                    break

        if default_manager_name:
            try:
                return self.managers_map[default_manager_name]
            except KeyError:
                raise ValueError(
                    "%s has no manager named %r" % (
                        self.object_name,
                        default_manager_name,
                    )
                )

        if self.managers:
            return self.managers[0]
```
### 130 - django/db/models/base.py:

Start line: 751, End line: 800

```python
class Model(metaclass=ModelBase):
    save.alters_data = True

    def save_base(self, raw=False, force_insert=False,
                  force_update=False, using=None, update_fields=None):
        """
        Handle the parts of saving which should be done only once per save,
        yet need to be done in raw saves, too. This includes some sanity
        checks and signal sending.

        The 'raw' argument is telling save_base not to save any parent
        models and not to do any changes to the values before save. This
        is used by fixture loading.
        """
        using = using or router.db_for_write(self.__class__, instance=self)
        assert not (force_insert and (force_update or update_fields))
        assert update_fields is None or update_fields
        cls = origin = self.__class__
        # Skip proxies, but keep the origin as the proxy model.
        if cls._meta.proxy:
            cls = cls._meta.concrete_model
        meta = cls._meta
        if not meta.auto_created:
            pre_save.send(
                sender=origin, instance=self, raw=raw, using=using,
                update_fields=update_fields,
            )
        # A transaction isn't needed if one query is issued.
        if meta.parents:
            context_manager = transaction.atomic(using=using, savepoint=False)
        else:
            context_manager = transaction.mark_for_rollback_on_error(using=using)
        with context_manager:
            parent_inserted = False
            if not raw:
                parent_inserted = self._save_parents(cls, using, update_fields)
            updated = self._save_table(
                raw, cls, force_insert or parent_inserted,
                force_update, using, update_fields,
            )
        # Store the database on which the object was saved
        self._state.db = using
        # Once saved, this is no longer a to-be-added instance.
        self._state.adding = False

        # Signal that the save is complete
        if not meta.auto_created:
            post_save.send(
                sender=origin, instance=self, created=(not updated),
                update_fields=update_fields, raw=raw, using=using,
            )
```
