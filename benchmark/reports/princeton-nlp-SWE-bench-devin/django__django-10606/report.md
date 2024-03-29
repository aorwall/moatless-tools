# django__django-10606

| **django/django** | `0315c18fe170b1b611b7d10b5dde2f196b89a7e0` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 132 |
| **Any found context length** | 132 |
| **Avg pos** | 2.0 |
| **Min pos** | 1 |
| **Max pos** | 1 |
| **Top file pos** | 1 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/db/models/fields/reverse_related.py b/django/db/models/fields/reverse_related.py
--- a/django/db/models/fields/reverse_related.py
+++ b/django/db/models/fields/reverse_related.py
@@ -114,7 +114,10 @@ def __repr__(self):
             self.related_model._meta.model_name,
         )
 
-    def get_choices(self, include_blank=True, blank_choice=BLANK_CHOICE_DASH, ordering=()):
+    def get_choices(
+        self, include_blank=True, blank_choice=BLANK_CHOICE_DASH,
+        limit_choices_to=None, ordering=(),
+    ):
         """
         Return choices with a default blank choices included, for use
         as <select> choices for this field.
@@ -122,7 +125,8 @@ def get_choices(self, include_blank=True, blank_choice=BLANK_CHOICE_DASH, orderi
         Analog of django.db.models.fields.Field.get_choices(), provided
         initially for utilization by RelatedFieldListFilter.
         """
-        qs = self.related_model._default_manager.all()
+        limit_choices_to = limit_choices_to or self.limit_choices_to
+        qs = self.related_model._default_manager.complex_filter(limit_choices_to)
         if ordering:
             qs = qs.order_by(*ordering)
         return (blank_choice if include_blank else []) + [

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/db/models/fields/reverse_related.py | 117 | 117 | 1 | 1 | 132
| django/db/models/fields/reverse_related.py | 125 | 125 | 1 | 1 | 132


## Problem Statement

```
Using RelatedOnlyFieldListFilter with reverse ManyToMany crashes
Description
	 
		(last modified by Tim Graham)
	 
Using RelatedOnlyFieldListFilter with a reverse ManyToMany relation causes this exception:
get_choices() got an unexpected keyword argument 'limit_choices_to'
This method in ForeignObjectRel.get_choices is missing the parameter that Field.get_choices has.
Pull Request: ​https://github.com/django/django/pull/10606
Demo of how to trigger bug: ​https://github.com/mgrdcm/django-bug-reverse-related/blob/master/rrbug/rrapp/admin.py#L11-L15

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 django/db/models/fields/reverse_related.py** | 117 | 130| 132 | 132 | 2114 | 
| 2 | 2 django/contrib/admin/filters.py | 422 | 430| 107 | 239 | 5848 | 
| 3 | 2 django/contrib/admin/filters.py | 209 | 226| 190 | 429 | 5848 | 
| 4 | 3 django/db/models/fields/related.py | 401 | 419| 165 | 594 | 19360 | 
| 5 | 3 django/contrib/admin/filters.py | 162 | 207| 427 | 1021 | 19360 | 
| 6 | 3 django/db/models/fields/related.py | 255 | 282| 269 | 1290 | 19360 | 
| 7 | 3 django/db/models/fields/related.py | 1202 | 1313| 939 | 2229 | 19360 | 
| 8 | 3 django/db/models/fields/related.py | 320 | 341| 225 | 2454 | 19360 | 
| 9 | 3 django/db/models/fields/related.py | 738 | 756| 222 | 2676 | 19360 | 
| 10 | 3 django/db/models/fields/related.py | 127 | 154| 202 | 2878 | 19360 | 
| 11 | 3 django/db/models/fields/related.py | 1169 | 1200| 180 | 3058 | 19360 | 
| 12 | 3 django/db/models/fields/related.py | 343 | 360| 163 | 3221 | 19360 | 
| 13 | 3 django/db/models/fields/related.py | 190 | 254| 673 | 3894 | 19360 | 
| 14 | 4 django/db/models/fields/related_descriptors.py | 343 | 362| 156 | 4050 | 29701 | 
| 15 | 4 django/db/models/fields/related.py | 616 | 638| 185 | 4235 | 29701 | 
| 16 | 4 django/db/models/fields/related.py | 600 | 614| 186 | 4421 | 29701 | 
| 17 | 4 django/db/models/fields/related.py | 1315 | 1387| 616 | 5037 | 29701 | 
| 18 | **4 django/db/models/fields/reverse_related.py** | 1 | 16| 110 | 5147 | 29701 | 
| 19 | 4 django/db/models/fields/related.py | 487 | 507| 138 | 5285 | 29701 | 
| 20 | 4 django/db/models/fields/related.py | 698 | 736| 335 | 5620 | 29701 | 
| 21 | 4 django/db/models/fields/related_descriptors.py | 879 | 900| 199 | 5819 | 29701 | 
| 22 | **4 django/db/models/fields/reverse_related.py** | 132 | 150| 172 | 5991 | 29701 | 
| 23 | **4 django/db/models/fields/reverse_related.py** | 152 | 177| 269 | 6260 | 29701 | 
| 24 | 4 django/db/models/fields/related.py | 952 | 962| 121 | 6381 | 29701 | 
| 25 | 4 django/db/models/fields/related.py | 658 | 682| 218 | 6599 | 29701 | 
| 26 | 4 django/db/models/fields/related.py | 171 | 188| 166 | 6765 | 29701 | 
| 27 | 4 django/db/models/fields/related.py | 565 | 598| 318 | 7083 | 29701 | 
| 28 | **4 django/db/models/fields/reverse_related.py** | 19 | 115| 635 | 7718 | 29701 | 
| 29 | 4 django/db/models/fields/related.py | 509 | 563| 409 | 8127 | 29701 | 
| 30 | 5 django/db/models/sql/compiler.py | 856 | 948| 829 | 8956 | 43312 | 
| 31 | 5 django/db/models/fields/related.py | 1389 | 1419| 322 | 9278 | 43312 | 
| 32 | 5 django/db/models/fields/related_descriptors.py | 550 | 605| 478 | 9756 | 43312 | 
| 33 | 5 django/db/models/fields/related.py | 896 | 916| 178 | 9934 | 43312 | 
| 34 | 5 django/db/models/fields/related_descriptors.py | 1160 | 1201| 392 | 10326 | 43312 | 
| 35 | 5 django/db/models/fields/related_descriptors.py | 863 | 877| 190 | 10516 | 43312 | 
| 36 | 5 django/db/models/fields/related_descriptors.py | 607 | 639| 323 | 10839 | 43312 | 
| 37 | 5 django/db/models/fields/related.py | 875 | 894| 145 | 10984 | 43312 | 
| 38 | 5 django/db/models/fields/related.py | 1421 | 1461| 399 | 11383 | 43312 | 
| 39 | 5 django/db/models/fields/related_descriptors.py | 119 | 153| 405 | 11788 | 43312 | 
| 40 | 5 django/db/models/fields/related.py | 108 | 125| 155 | 11943 | 43312 | 
| 41 | 5 django/contrib/admin/filters.py | 229 | 244| 196 | 12139 | 43312 | 
| 42 | 5 django/db/models/fields/related.py | 1556 | 1591| 475 | 12614 | 43312 | 
| 43 | 5 django/db/models/fields/related_descriptors.py | 671 | 725| 511 | 13125 | 43312 | 
| 44 | 6 django/db/models/fields/related_lookups.py | 102 | 117| 215 | 13340 | 44761 | 
| 45 | 6 django/contrib/admin/filters.py | 278 | 302| 217 | 13557 | 44761 | 
| 46 | 6 django/db/models/fields/related_lookups.py | 46 | 60| 224 | 13781 | 44761 | 
| 47 | 6 django/contrib/admin/filters.py | 264 | 276| 149 | 13930 | 44761 | 
| 48 | 6 django/db/models/fields/related.py | 1593 | 1609| 286 | 14216 | 44761 | 
| 49 | **6 django/db/models/fields/reverse_related.py** | 180 | 223| 351 | 14567 | 44761 | 
| 50 | 6 django/db/models/fields/related.py | 362 | 399| 292 | 14859 | 44761 | 
| 51 | 7 django/contrib/admin/views/main.py | 465 | 496| 225 | 15084 | 48968 | 
| 52 | 7 django/db/models/fields/related.py | 156 | 169| 144 | 15228 | 48968 | 
| 53 | 7 django/db/models/fields/related_descriptors.py | 364 | 380| 184 | 15412 | 48968 | 
| 54 | 7 django/db/models/fields/related.py | 847 | 873| 240 | 15652 | 48968 | 
| 55 | 7 django/db/models/fields/related_lookups.py | 62 | 99| 447 | 16099 | 48968 | 
| 56 | 7 django/db/models/fields/related_descriptors.py | 965 | 982| 190 | 16289 | 48968 | 
| 57 | **7 django/db/models/fields/reverse_related.py** | 248 | 291| 318 | 16607 | 48968 | 
| 58 | 7 django/db/models/fields/related.py | 1 | 34| 244 | 16851 | 48968 | 
| 59 | 7 django/db/models/fields/related_descriptors.py | 1010 | 1037| 334 | 17185 | 48968 | 
| 60 | 7 django/db/models/fields/related_lookups.py | 119 | 155| 244 | 17429 | 48968 | 
| 61 | 7 django/db/models/fields/related_descriptors.py | 1070 | 1081| 138 | 17567 | 48968 | 
| 62 | 7 django/db/models/fields/related.py | 824 | 845| 169 | 17736 | 48968 | 
| 63 | 7 django/db/models/fields/related.py | 421 | 441| 166 | 17902 | 48968 | 
| 64 | 7 django/db/models/fields/related_descriptors.py | 727 | 753| 222 | 18124 | 48968 | 
| 65 | 8 django/db/backends/base/schema.py | 31 | 41| 120 | 18244 | 60284 | 
| 66 | 9 django/contrib/admin/options.py | 368 | 420| 504 | 18748 | 78650 | 
| 67 | 9 django/db/models/fields/related.py | 284 | 318| 293 | 19041 | 78650 | 
| 68 | 9 django/db/models/fields/related_descriptors.py | 941 | 963| 218 | 19259 | 78650 | 
| 69 | 9 django/db/models/fields/related_descriptors.py | 1112 | 1158| 489 | 19748 | 78650 | 
| 70 | 9 django/db/models/fields/related.py | 684 | 696| 116 | 19864 | 78650 | 
| 71 | 9 django/db/models/fields/related.py | 1611 | 1645| 266 | 20130 | 78650 | 
| 72 | 10 django/contrib/contenttypes/fields.py | 431 | 452| 248 | 20378 | 84062 | 
| 73 | 10 django/db/models/fields/related_descriptors.py | 984 | 1009| 241 | 20619 | 84062 | 
| 74 | 10 django/db/models/fields/related.py | 1489 | 1506| 184 | 20803 | 84062 | 
| 75 | **10 django/db/models/fields/reverse_related.py** | 226 | 245| 147 | 20950 | 84062 | 
| 76 | 10 django/db/models/fields/related.py | 640 | 656| 163 | 21113 | 84062 | 
| 77 | 10 django/contrib/admin/filters.py | 371 | 395| 294 | 21407 | 84062 | 
| 78 | 11 django/db/models/options.py | 513 | 528| 146 | 21553 | 91161 | 
| 79 | 11 django/contrib/contenttypes/fields.py | 414 | 429| 124 | 21677 | 91161 | 
| 80 | 11 django/contrib/admin/filters.py | 305 | 368| 627 | 22304 | 91161 | 
| 81 | 11 django/db/models/fields/related_descriptors.py | 902 | 939| 374 | 22678 | 91161 | 
| 82 | 11 django/contrib/admin/views/main.py | 121 | 203| 818 | 23496 | 91161 | 
| 83 | 11 django/db/models/fields/related_descriptors.py | 641 | 670| 254 | 23750 | 91161 | 
| 84 | 12 django/db/models/sql/query.py | 1435 | 1513| 734 | 24484 | 112734 | 
| 85 | 12 django/db/models/fields/related_descriptors.py | 802 | 861| 576 | 25060 | 112734 | 
| 86 | 12 django/db/models/fields/related.py | 964 | 991| 215 | 25275 | 112734 | 
| 87 | 13 django/db/models/query_utils.py | 221 | 254| 298 | 25573 | 115360 | 
| 88 | 13 django/db/models/options.py | 499 | 511| 109 | 25682 | 115360 | 
| 89 | 13 django/db/models/fields/related_descriptors.py | 1039 | 1068| 272 | 25954 | 115360 | 
| 90 | 13 django/contrib/contenttypes/fields.py | 397 | 412| 127 | 26081 | 115360 | 
| 91 | 13 django/contrib/contenttypes/fields.py | 333 | 355| 185 | 26266 | 115360 | 
| 92 | 13 django/contrib/admin/filters.py | 246 | 261| 154 | 26420 | 115360 | 
| 93 | 13 django/db/models/fields/related_descriptors.py | 1 | 78| 673 | 27093 | 115360 | 
| 94 | 14 django/db/models/fields/__init__.py | 240 | 305| 467 | 27560 | 132767 | 
| 95 | 14 django/db/models/fields/related.py | 1463 | 1487| 295 | 27855 | 132767 | 
| 96 | 14 django/db/models/fields/related_descriptors.py | 155 | 200| 445 | 28300 | 132767 | 
| 97 | 14 django/db/models/fields/related_descriptors.py | 308 | 322| 182 | 28482 | 132767 | 
| 98 | 14 django/contrib/contenttypes/fields.py | 274 | 331| 434 | 28916 | 132767 | 
| 99 | 14 django/db/models/fields/related_descriptors.py | 493 | 547| 338 | 29254 | 132767 | 
| 100 | 14 django/db/models/fields/related.py | 1091 | 1167| 524 | 29778 | 132767 | 
| 101 | 14 django/db/models/options.py | 439 | 471| 328 | 30106 | 132767 | 
| 102 | 14 django/contrib/admin/filters.py | 397 | 419| 211 | 30317 | 132767 | 
| 103 | 14 django/db/models/fields/related_lookups.py | 1 | 23| 170 | 30487 | 132767 | 
| 104 | 14 django/contrib/contenttypes/fields.py | 21 | 109| 571 | 31058 | 132767 | 
| 105 | 14 django/db/models/options.py | 685 | 719| 378 | 31436 | 132767 | 
| 106 | 15 django/db/models/deletion.py | 1 | 63| 482 | 31918 | 135853 | 
| 107 | 15 django/db/models/fields/related_descriptors.py | 81 | 117| 264 | 32182 | 135853 | 
| 108 | 15 django/db/models/fields/related_descriptors.py | 1083 | 1110| 338 | 32520 | 135853 | 
| 109 | 15 django/db/models/sql/query.py | 1108 | 1137| 325 | 32845 | 135853 | 
| 110 | 15 django/db/models/fields/related.py | 918 | 950| 279 | 33124 | 135853 | 
| 111 | 15 django/db/models/fields/related.py | 444 | 485| 273 | 33397 | 135853 | 
| 112 | 15 django/db/models/fields/related_descriptors.py | 382 | 427| 355 | 33752 | 135853 | 
| 113 | 15 django/contrib/admin/views/main.py | 420 | 463| 390 | 34142 | 135853 | 
| 114 | 15 django/db/models/fields/related.py | 759 | 822| 507 | 34649 | 135853 | 
| 115 | 15 django/contrib/admin/filters.py | 20 | 59| 295 | 34944 | 135853 | 
| 116 | 15 django/contrib/admin/filters.py | 118 | 159| 365 | 35309 | 135853 | 
| 117 | 16 django/db/models/__init__.py | 1 | 51| 576 | 35885 | 136429 | 
| 118 | 16 django/db/models/fields/related.py | 1044 | 1088| 407 | 36292 | 136429 | 
| 119 | 17 django/db/migrations/operations/utils.py | 1 | 14| 138 | 36430 | 136909 | 
| 120 | 17 django/contrib/admin/views/main.py | 1 | 45| 319 | 36749 | 136909 | 
| 121 | 18 django/contrib/admin/utils.py | 285 | 303| 175 | 36924 | 140997 | 
| 122 | 18 django/db/models/options.py | 721 | 736| 144 | 37068 | 140997 | 
| 123 | 19 django/db/models/base.py | 1492 | 1524| 231 | 37299 | 156286 | 
| 124 | 19 django/db/models/fields/related.py | 83 | 106| 162 | 37461 | 156286 | 
| 125 | 19 django/db/models/options.py | 530 | 558| 231 | 37692 | 156286 | 
| 126 | 19 django/db/models/sql/query.py | 1653 | 1688| 409 | 38101 | 156286 | 
| 127 | 19 django/db/models/options.py | 560 | 583| 228 | 38329 | 156286 | 
| 128 | 20 django/db/models/lookups.py | 259 | 275| 133 | 38462 | 160960 | 
| 129 | 20 django/db/models/sql/query.py | 1406 | 1417| 137 | 38599 | 160960 | 
| 130 | 20 django/contrib/contenttypes/fields.py | 677 | 702| 254 | 38853 | 160960 | 
| 131 | 20 django/db/models/fields/related.py | 1526 | 1554| 275 | 39128 | 160960 | 
| 132 | 21 django/db/models/query.py | 1605 | 1711| 1063 | 40191 | 177899 | 
| 133 | 21 django/db/models/query.py | 1031 | 1052| 214 | 40405 | 177899 | 
| 134 | 21 django/db/models/fields/related.py | 994 | 1041| 368 | 40773 | 177899 | 
| 135 | 21 django/db/models/options.py | 1 | 36| 304 | 41077 | 177899 | 
| 136 | 21 django/contrib/admin/options.py | 422 | 465| 350 | 41427 | 177899 | 
| 137 | 22 django/contrib/admin/widgets.py | 278 | 313| 382 | 41809 | 181765 | 
| 138 | 22 django/contrib/admin/widgets.py | 166 | 197| 243 | 42052 | 181765 | 
| 139 | 22 django/contrib/admin/options.py | 243 | 275| 357 | 42409 | 181765 | 
| 140 | 23 django/contrib/admin/checks.py | 791 | 841| 443 | 42852 | 190781 | 
| 141 | 23 django/contrib/contenttypes/fields.py | 630 | 654| 214 | 43066 | 190781 | 
| 142 | 23 django/db/models/options.py | 738 | 750| 131 | 43197 | 190781 | 


### Hint

```
I can't reproduce the crash on a ManyToManyField with limit_choices_to. Could you give more details?
Apologies for my lack of response on this, have been traveling. I'm going to be working on adding tests for my fix ASAP but here's a minimal example of how to trigger: Demo of how to trigger bug: ​https://github.com/mgrdcm/django-bug-reverse-related/blob/master/rrbug/rrapp/admin.py#L11-L15
Test case added: ​https://github.com/django/django/pull/10606/commits/da4785e82c9586c6f2ab41a0e9e5bc3eeeb8fd1c
Tim says "test should very the correct results of the Filter rather than just checking that an exception isn't raised."
```

## Patch

```diff
diff --git a/django/db/models/fields/reverse_related.py b/django/db/models/fields/reverse_related.py
--- a/django/db/models/fields/reverse_related.py
+++ b/django/db/models/fields/reverse_related.py
@@ -114,7 +114,10 @@ def __repr__(self):
             self.related_model._meta.model_name,
         )
 
-    def get_choices(self, include_blank=True, blank_choice=BLANK_CHOICE_DASH, ordering=()):
+    def get_choices(
+        self, include_blank=True, blank_choice=BLANK_CHOICE_DASH,
+        limit_choices_to=None, ordering=(),
+    ):
         """
         Return choices with a default blank choices included, for use
         as <select> choices for this field.
@@ -122,7 +125,8 @@ def get_choices(self, include_blank=True, blank_choice=BLANK_CHOICE_DASH, orderi
         Analog of django.db.models.fields.Field.get_choices(), provided
         initially for utilization by RelatedFieldListFilter.
         """
-        qs = self.related_model._default_manager.all()
+        limit_choices_to = limit_choices_to or self.limit_choices_to
+        qs = self.related_model._default_manager.complex_filter(limit_choices_to)
         if ordering:
             qs = qs.order_by(*ordering)
         return (blank_choice if include_blank else []) + [

```

## Test Patch

```diff
diff --git a/tests/admin_filters/tests.py b/tests/admin_filters/tests.py
--- a/tests/admin_filters/tests.py
+++ b/tests/admin_filters/tests.py
@@ -741,6 +741,43 @@ def test_relatedonlyfieldlistfilter_foreignkey(self):
         expected = [(self.alfred.pk, 'alfred'), (self.bob.pk, 'bob')]
         self.assertEqual(sorted(filterspec.lookup_choices), sorted(expected))
 
+    def test_relatedonlyfieldlistfilter_foreignkey_reverse_relationships(self):
+        class EmployeeAdminReverseRelationship(ModelAdmin):
+            list_filter = (
+                ('book', RelatedOnlyFieldListFilter),
+            )
+
+        self.djangonaut_book.employee = self.john
+        self.djangonaut_book.save()
+        self.django_book.employee = self.jack
+        self.django_book.save()
+
+        modeladmin = EmployeeAdminReverseRelationship(Employee, site)
+        request = self.request_factory.get('/')
+        request.user = self.alfred
+        changelist = modeladmin.get_changelist_instance(request)
+        filterspec = changelist.get_filters(request)[0][0]
+        self.assertEqual(filterspec.lookup_choices, [
+            (self.djangonaut_book.pk, 'Djangonaut: an art of living'),
+            (self.django_book.pk, 'The Django Book'),
+        ])
+
+    def test_relatedonlyfieldlistfilter_manytomany_reverse_relationships(self):
+        class UserAdminReverseRelationship(ModelAdmin):
+            list_filter = (
+                ('books_contributed', RelatedOnlyFieldListFilter),
+            )
+
+        modeladmin = UserAdminReverseRelationship(User, site)
+        request = self.request_factory.get('/')
+        request.user = self.alfred
+        changelist = modeladmin.get_changelist_instance(request)
+        filterspec = changelist.get_filters(request)[0][0]
+        self.assertEqual(
+            filterspec.lookup_choices,
+            [(self.guitar_book.pk, 'Guitar for dummies')],
+        )
+
     def test_relatedonlyfieldlistfilter_foreignkey_ordering(self):
         """RelatedOnlyFieldListFilter ordering respects ModelAdmin.ordering."""
         class EmployeeAdminWithOrdering(ModelAdmin):
diff --git a/tests/model_fields/tests.py b/tests/model_fields/tests.py
--- a/tests/model_fields/tests.py
+++ b/tests/model_fields/tests.py
@@ -266,3 +266,37 @@ def test_get_choices_reverse_related_field_default_ordering(self):
             self.field.remote_field.get_choices(include_blank=False),
             [self.bar2, self.bar1]
         )
+
+
+class GetChoicesLimitChoicesToTests(TestCase):
+    @classmethod
+    def setUpTestData(cls):
+        cls.foo1 = Foo.objects.create(a='a', d='12.34')
+        cls.foo2 = Foo.objects.create(a='b', d='12.34')
+        cls.bar1 = Bar.objects.create(a=cls.foo1, b='b')
+        cls.bar2 = Bar.objects.create(a=cls.foo2, b='a')
+        cls.field = Bar._meta.get_field('a')
+
+    def assertChoicesEqual(self, choices, objs):
+        self.assertEqual(choices, [(obj.pk, str(obj)) for obj in objs])
+
+    def test_get_choices(self):
+        self.assertChoicesEqual(
+            self.field.get_choices(include_blank=False, limit_choices_to={'a': 'a'}),
+            [self.foo1],
+        )
+        self.assertChoicesEqual(
+            self.field.get_choices(include_blank=False, limit_choices_to={}),
+            [self.foo1, self.foo2],
+        )
+
+    def test_get_choices_reverse_related_field(self):
+        field = self.field.remote_field
+        self.assertChoicesEqual(
+            field.get_choices(include_blank=False, limit_choices_to={'b': 'b'}),
+            [self.bar1],
+        )
+        self.assertChoicesEqual(
+            field.get_choices(include_blank=False, limit_choices_to={}),
+            [self.bar1, self.bar2],
+        )

```


## Code snippets

### 1 - django/db/models/fields/reverse_related.py:

Start line: 117, End line: 130

```python
class ForeignObjectRel(FieldCacheMixin):

    def get_choices(self, include_blank=True, blank_choice=BLANK_CHOICE_DASH, ordering=()):
        """
        Return choices with a default blank choices included, for use
        as <select> choices for this field.

        Analog of django.db.models.fields.Field.get_choices(), provided
        initially for utilization by RelatedFieldListFilter.
        """
        qs = self.related_model._default_manager.all()
        if ordering:
            qs = qs.order_by(*ordering)
        return (blank_choice if include_blank else []) + [
            (x.pk, str(x)) for x in qs
        ]
```
### 2 - django/contrib/admin/filters.py:

Start line: 422, End line: 430

```python
FieldListFilter.register(lambda f: True, AllValuesFieldListFilter)


class RelatedOnlyFieldListFilter(RelatedFieldListFilter):
    def field_choices(self, field, request, model_admin):
        pk_qs = model_admin.get_queryset(request).distinct().values_list('%s__pk' % self.field_path, flat=True)
        ordering = self.field_admin_ordering(field, request, model_admin)
        return field.get_choices(include_blank=False, limit_choices_to={'pk__in': pk_qs}, ordering=ordering)
```
### 3 - django/contrib/admin/filters.py:

Start line: 209, End line: 226

```python
class RelatedFieldListFilter(FieldListFilter):

    def choices(self, changelist):
        yield {
            'selected': self.lookup_val is None and not self.lookup_val_isnull,
            'query_string': changelist.get_query_string(remove=[self.lookup_kwarg, self.lookup_kwarg_isnull]),
            'display': _('All'),
        }
        for pk_val, val in self.lookup_choices:
            yield {
                'selected': self.lookup_val == str(pk_val),
                'query_string': changelist.get_query_string({self.lookup_kwarg: pk_val}, [self.lookup_kwarg_isnull]),
                'display': val,
            }
        if self.include_empty_choice:
            yield {
                'selected': bool(self.lookup_val_isnull),
                'query_string': changelist.get_query_string({self.lookup_kwarg_isnull: 'True'}, [self.lookup_kwarg]),
                'display': self.empty_value_display,
            }
```
### 4 - django/db/models/fields/related.py:

Start line: 401, End line: 419

```python
class RelatedField(FieldCacheMixin, Field):

    def formfield(self, **kwargs):
        """
        Pass ``limit_choices_to`` to the field being constructed.

        Only passes it if there is a type that supports related fields.
        This is a similar strategy used to pass the ``queryset`` to the field
        being constructed.
        """
        defaults = {}
        if hasattr(self.remote_field, 'get_related_field'):
            # If this is a callable, do not invoke it here. Just pass
            # it in the defaults for when the form class will later be
            # instantiated.
            limit_choices_to = self.remote_field.limit_choices_to
            defaults.update({
                'limit_choices_to': limit_choices_to,
            })
        defaults.update(kwargs)
        return super().formfield(**defaults)
```
### 5 - django/contrib/admin/filters.py:

Start line: 162, End line: 207

```python
class RelatedFieldListFilter(FieldListFilter):
    def __init__(self, field, request, params, model, model_admin, field_path):
        other_model = get_model_from_relation(field)
        self.lookup_kwarg = '%s__%s__exact' % (field_path, field.target_field.name)
        self.lookup_kwarg_isnull = '%s__isnull' % field_path
        self.lookup_val = params.get(self.lookup_kwarg)
        self.lookup_val_isnull = params.get(self.lookup_kwarg_isnull)
        super().__init__(field, request, params, model, model_admin, field_path)
        self.lookup_choices = self.field_choices(field, request, model_admin)
        if hasattr(field, 'verbose_name'):
            self.lookup_title = field.verbose_name
        else:
            self.lookup_title = other_model._meta.verbose_name
        self.title = self.lookup_title
        self.empty_value_display = model_admin.get_empty_value_display()

    @property
    def include_empty_choice(self):
        """
        Return True if a "(None)" choice should be included, which filters
        out everything except empty relationships.
        """
        return self.field.null or (self.field.is_relation and self.field.many_to_many)

    def has_output(self):
        if self.include_empty_choice:
            extra = 1
        else:
            extra = 0
        return len(self.lookup_choices) + extra > 1

    def expected_parameters(self):
        return [self.lookup_kwarg, self.lookup_kwarg_isnull]

    def field_admin_ordering(self, field, request, model_admin):
        """
        Return the model admin's ordering for related field, if provided.
        """
        related_admin = model_admin.admin_site._registry.get(field.remote_field.model)
        if related_admin is not None:
            return related_admin.get_ordering(request)
        return ()

    def field_choices(self, field, request, model_admin):
        ordering = self.field_admin_ordering(field, request, model_admin)
        return field.get_choices(include_blank=False, ordering=ordering)
```
### 6 - django/db/models/fields/related.py:

Start line: 255, End line: 282

```python
class RelatedField(FieldCacheMixin, Field):

    def _check_clashes(self):
        # ... other code
        for clash_field in potential_clashes:
            clash_name = "%s.%s" % (  # i. e. "Model.m2m"
                clash_field.related_model._meta.object_name,
                clash_field.field.name)
            if not rel_is_hidden and clash_field.get_accessor_name() == rel_name:
                errors.append(
                    checks.Error(
                        "Reverse accessor for '%s' clashes with reverse accessor for '%s'." % (field_name, clash_name),
                        hint=("Add or change a related_name argument "
                              "to the definition for '%s' or '%s'.") % (field_name, clash_name),
                        obj=self,
                        id='fields.E304',
                    )
                )

            if clash_field.get_accessor_name() == rel_query_name:
                errors.append(
                    checks.Error(
                        "Reverse query name for '%s' clashes with reverse query name for '%s'."
                        % (field_name, clash_name),
                        hint=("Add or change a related_name argument "
                              "to the definition for '%s' or '%s'.") % (field_name, clash_name),
                        obj=self,
                        id='fields.E305',
                    )
                )

        return errors
```
### 7 - django/db/models/fields/related.py:

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
### 8 - django/db/models/fields/related.py:

Start line: 320, End line: 341

```python
class RelatedField(FieldCacheMixin, Field):

    def deconstruct(self):
        name, path, args, kwargs = super().deconstruct()
        if self.remote_field.limit_choices_to:
            kwargs['limit_choices_to'] = self.remote_field.limit_choices_to
        if self.remote_field.related_name is not None:
            kwargs['related_name'] = self.remote_field.related_name
        if self.remote_field.related_query_name is not None:
            kwargs['related_query_name'] = self.remote_field.related_query_name
        return name, path, args, kwargs

    def get_forward_related_filter(self, obj):
        """
        Return the keyword arguments that when supplied to
        self.model.object.filter(), would select all instances related through
        this field to the remote obj. This is used to build the querysets
        returned by related descriptors. obj is an instance of
        self.related_field.model.
        """
        return {
            '%s__%s' % (self.name, rh_field.name): getattr(obj, rh_field.attname)
            for _, rh_field in self.related_fields
        }
```
### 9 - django/db/models/fields/related.py:

Start line: 738, End line: 756

```python
class ForeignObject(RelatedField):

    def contribute_to_related_class(self, cls, related):
        # Internal FK's - i.e., those with a related name ending with '+' -
        # and swapped models don't get a related descriptor.
        if not self.remote_field.is_hidden() and not related.related_model._meta.swapped:
            setattr(cls._meta.concrete_model, related.get_accessor_name(), self.related_accessor_class(related))
            # While 'limit_choices_to' might be a callable, simply pass
            # it along for later - this is too early because it's still
            # model load time.
            if self.remote_field.limit_choices_to:
                cls._meta.related_fkey_lookups.append(self.remote_field.limit_choices_to)


ForeignObject.register_lookup(RelatedIn)
ForeignObject.register_lookup(RelatedExact)
ForeignObject.register_lookup(RelatedLessThan)
ForeignObject.register_lookup(RelatedGreaterThan)
ForeignObject.register_lookup(RelatedGreaterThanOrEqual)
ForeignObject.register_lookup(RelatedLessThanOrEqual)
ForeignObject.register_lookup(RelatedIsNull)
```
### 10 - django/db/models/fields/related.py:

Start line: 127, End line: 154

```python
class RelatedField(FieldCacheMixin, Field):

    def _check_related_query_name_is_valid(self):
        if self.remote_field.is_hidden():
            return []
        rel_query_name = self.related_query_name()
        errors = []
        if rel_query_name.endswith('_'):
            errors.append(
                checks.Error(
                    "Reverse query name '%s' must not end with an underscore."
                    % (rel_query_name,),
                    hint=("Add or change a related_name or related_query_name "
                          "argument for this field."),
                    obj=self,
                    id='fields.E308',
                )
            )
        if LOOKUP_SEP in rel_query_name:
            errors.append(
                checks.Error(
                    "Reverse query name '%s' must not contain '%s'."
                    % (rel_query_name, LOOKUP_SEP),
                    hint=("Add or change a related_name or related_query_name "
                          "argument for this field."),
                    obj=self,
                    id='fields.E309',
                )
            )
        return errors
```
### 18 - django/db/models/fields/reverse_related.py:

Start line: 1, End line: 16

```python
"""
"Rel objects" for related fields.

"Rel objects" (for lack of a better name) carry information about the relation
modeled by a related field and provide some utility functions. They're stored
in the ``remote_field`` attribute of the field.

They also act as reverse fields for the purposes of the Meta API because
they're the closest concept currently available.
"""

from django.core import exceptions
from django.utils.functional import cached_property

from . import BLANK_CHOICE_DASH
from .mixins import FieldCacheMixin
```
### 22 - django/db/models/fields/reverse_related.py:

Start line: 132, End line: 150

```python
class ForeignObjectRel(FieldCacheMixin):

    def is_hidden(self):
        """Should the related object be hidden?"""
        return bool(self.related_name) and self.related_name[-1] == '+'

    def get_joining_columns(self):
        return self.field.get_reverse_joining_columns()

    def get_extra_restriction(self, where_class, alias, related_alias):
        return self.field.get_extra_restriction(where_class, related_alias, alias)

    def set_field_name(self):
        """
        Set the related field's name, this is not available until later stages
        of app loading, so set_field_name is called from
        set_attributes_from_rel()
        """
        # By default foreign object doesn't relate to any remote field (for
        # example custom multicolumn joins currently have no remote field).
        self.field_name = None
```
### 23 - django/db/models/fields/reverse_related.py:

Start line: 152, End line: 177

```python
class ForeignObjectRel(FieldCacheMixin):

    def get_accessor_name(self, model=None):
        # This method encapsulates the logic that decides what name to give an
        # accessor descriptor that retrieves related many-to-one or
        # many-to-many objects. It uses the lowercased object_name + "_set",
        # but this can be overridden with the "related_name" option. Due to
        # backwards compatibility ModelForms need to be able to provide an
        # alternate model. See BaseInlineFormSet.get_default_prefix().
        opts = model._meta if model else self.related_model._meta
        model = model or self.related_model
        if self.multiple:
            # If this is a symmetrical m2m relation on self, there is no reverse accessor.
            if self.symmetrical and model == self.model:
                return None
        if self.related_name:
            return self.related_name
        return opts.model_name + ('_set' if self.multiple else '')

    def get_path_info(self, filtered_relation=None):
        return self.field.get_reverse_path_info(filtered_relation)

    def get_cache_name(self):
        """
        Return the name of the cache key to use for storing an instance of the
        forward model on the reverse model.
        """
        return self.get_accessor_name()
```
### 28 - django/db/models/fields/reverse_related.py:

Start line: 19, End line: 115

```python
class ForeignObjectRel(FieldCacheMixin):
    """
    Used by ForeignObject to store information about the relation.

    ``_meta.get_fields()`` returns this class to provide access to the field
    flags for the reverse relation.
    """

    # Field flags
    auto_created = True
    concrete = False
    editable = False
    is_relation = True

    # Reverse relations are always nullable (Django can't enforce that a
    # foreign key on the related model points to this model).
    null = True

    def __init__(self, field, to, related_name=None, related_query_name=None,
                 limit_choices_to=None, parent_link=False, on_delete=None):
        self.field = field
        self.model = to
        self.related_name = related_name
        self.related_query_name = related_query_name
        self.limit_choices_to = {} if limit_choices_to is None else limit_choices_to
        self.parent_link = parent_link
        self.on_delete = on_delete

        self.symmetrical = False
        self.multiple = True

    # Some of the following cached_properties can't be initialized in
    # __init__ as the field doesn't have its model yet. Calling these methods
    # before field.contribute_to_class() has been called will result in
    # AttributeError
    @cached_property
    def hidden(self):
        return self.is_hidden()

    @cached_property
    def name(self):
        return self.field.related_query_name()

    @property
    def remote_field(self):
        return self.field

    @property
    def target_field(self):
        """
        When filtering against this relation, return the field on the remote
        model against which the filtering should happen.
        """
        target_fields = self.get_path_info()[-1].target_fields
        if len(target_fields) > 1:
            raise exceptions.FieldError("Can't use target_field for multicolumn relations.")
        return target_fields[0]

    @cached_property
    def related_model(self):
        if not self.field.model:
            raise AttributeError(
                "This property can't be accessed before self.field.contribute_to_class has been called.")
        return self.field.model

    @cached_property
    def many_to_many(self):
        return self.field.many_to_many

    @cached_property
    def many_to_one(self):
        return self.field.one_to_many

    @cached_property
    def one_to_many(self):
        return self.field.many_to_one

    @cached_property
    def one_to_one(self):
        return self.field.one_to_one

    def get_lookup(self, lookup_name):
        return self.field.get_lookup(lookup_name)

    def get_internal_type(self):
        return self.field.get_internal_type()

    @property
    def db_type(self):
        return self.field.db_type

    def __repr__(self):
        return '<%s: %s.%s>' % (
            type(self).__name__,
            self.related_model._meta.app_label,
            self.related_model._meta.model_name,
        )
```
### 49 - django/db/models/fields/reverse_related.py:

Start line: 180, End line: 223

```python
class ManyToOneRel(ForeignObjectRel):
    """
    Used by the ForeignKey field to store information about the relation.

    ``_meta.get_fields()`` returns this class to provide access to the field
    flags for the reverse relation.

    Note: Because we somewhat abuse the Rel objects by using them as reverse
    fields we get the funny situation where
    ``ManyToOneRel.many_to_one == False`` and
    ``ManyToOneRel.one_to_many == True``. This is unfortunate but the actual
    ManyToOneRel class is a private API and there is work underway to turn
    reverse relations into actual fields.
    """

    def __init__(self, field, to, field_name, related_name=None, related_query_name=None,
                 limit_choices_to=None, parent_link=False, on_delete=None):
        super().__init__(
            field, to,
            related_name=related_name,
            related_query_name=related_query_name,
            limit_choices_to=limit_choices_to,
            parent_link=parent_link,
            on_delete=on_delete,
        )

        self.field_name = field_name

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('related_model', None)
        return state

    def get_related_field(self):
        """
        Return the Field in the 'to' object to which this relationship is tied.
        """
        field = self.model._meta.get_field(self.field_name)
        if not field.concrete:
            raise exceptions.FieldDoesNotExist("No related field named '%s'" % self.field_name)
        return field

    def set_field_name(self):
        self.field_name = self.field_name or self.model._meta.pk.name
```
### 57 - django/db/models/fields/reverse_related.py:

Start line: 248, End line: 291

```python
class ManyToManyRel(ForeignObjectRel):
    """
    Used by ManyToManyField to store information about the relation.

    ``_meta.get_fields()`` returns this class to provide access to the field
    flags for the reverse relation.
    """

    def __init__(self, field, to, related_name=None, related_query_name=None,
                 limit_choices_to=None, symmetrical=True, through=None,
                 through_fields=None, db_constraint=True):
        super().__init__(
            field, to,
            related_name=related_name,
            related_query_name=related_query_name,
            limit_choices_to=limit_choices_to,
        )

        if through and not db_constraint:
            raise ValueError("Can't supply a through model and db_constraint=False")
        self.through = through

        if through_fields and not through:
            raise ValueError("Cannot specify through_fields without a through model")
        self.through_fields = through_fields

        self.symmetrical = symmetrical
        self.db_constraint = db_constraint

    def get_related_field(self):
        """
        Return the field in the 'to' object to which this relationship is tied.
        Provided for symmetry with ManyToOneRel.
        """
        opts = self.through._meta
        if self.through_fields:
            field = opts.get_field(self.through_fields[0])
        else:
            for field in opts.fields:
                rel = getattr(field, 'remote_field', None)
                if rel and rel.model == self.model:
                    break
        return field.foreign_related_fields[0]
```
### 75 - django/db/models/fields/reverse_related.py:

Start line: 226, End line: 245

```python
class OneToOneRel(ManyToOneRel):
    """
    Used by OneToOneField to store information about the relation.

    ``_meta.get_fields()`` returns this class to provide access to the field
    flags for the reverse relation.
    """

    def __init__(self, field, to, field_name, related_name=None, related_query_name=None,
                 limit_choices_to=None, parent_link=False, on_delete=None):
        super().__init__(
            field, to, field_name,
            related_name=related_name,
            related_query_name=related_query_name,
            limit_choices_to=limit_choices_to,
            parent_link=parent_link,
            on_delete=on_delete,
        )

        self.multiple = False
```
