# django__django-11298

| **django/django** | `a9179ab032cda80801e7f67ef20db5ee60989f21` |
| ---- | ---- |
| **No of patches** | 2 |
| **All found context length** | 2009 |
| **Any found context length** | 2009 |
| **Avg pos** | 5.0 |
| **Min pos** | 3 |
| **Max pos** | 7 |
| **Top file pos** | 1 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/db/models/fields/related.py b/django/db/models/fields/related.py
--- a/django/db/models/fields/related.py
+++ b/django/db/models/fields/related.py
@@ -1234,18 +1234,6 @@ def _check_relationship_model(self, from_model=None, **kwargs):
                 to_model_name = to_model._meta.object_name
             relationship_model_name = self.remote_field.through._meta.object_name
             self_referential = from_model == to_model
-
-            # Check symmetrical attribute.
-            if (self_referential and self.remote_field.symmetrical and
-                    not self.remote_field.through._meta.auto_created):
-                errors.append(
-                    checks.Error(
-                        'Many-to-many fields with intermediate tables must not be symmetrical.',
-                        obj=self,
-                        id='fields.E332',
-                    )
-                )
-
             # Count foreign keys in intermediate model
             if self_referential:
                 seen_self = sum(
diff --git a/django/db/models/fields/related_descriptors.py b/django/db/models/fields/related_descriptors.py
--- a/django/db/models/fields/related_descriptors.py
+++ b/django/db/models/fields/related_descriptors.py
@@ -938,11 +938,14 @@ def add(self, *objs, through_defaults=None):
                     through_defaults=through_defaults,
                 )
                 # If this is a symmetrical m2m relation to self, add the mirror
-                # entry in the m2m table. `through_defaults` aren't used here
-                # because of the system check error fields.E332: Many-to-many
-                # fields with intermediate tables must not be symmetrical.
+                # entry in the m2m table.
                 if self.symmetrical:
-                    self._add_items(self.target_field_name, self.source_field_name, *objs)
+                    self._add_items(
+                        self.target_field_name,
+                        self.source_field_name,
+                        *objs,
+                        through_defaults=through_defaults,
+                    )
         add.alters_data = True
 
         def remove(self, *objs):

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/db/models/fields/related.py | 1237 | 1248 | 3 | 1 | 2009
| django/db/models/fields/related_descriptors.py | 941 | 945 | 7 | 2 | 3546


## Problem Statement

```
Allow ManyToManyField using a intermediary table to be defined as symmetrical.
Description
	
Thanks to the work made by Collin Anderson in #9475 I think we can remove the check 
"fields.E332 Many-to-many fields with intermediate tables must not be symmetrical." with a little adjustment.
This change was discussed in the django-dev mailing list ​https://groups.google.com/forum/#!topic/django-developers/BuT0-Uq8pyc.
This would let have 
class Person(models.Model):
	name = models.CharField(max_length=20)
	friends = models.ManyToManyField('self', through='Friendship')
class Friendship(models.Model):
	first = models.ForeignKey(Person, models.CASCADE, related_name="+")
	second = models.ForeignKey(Person, models.CASCADE)
	friendship_date = models.DateTimeField()
and just do something like
joe.friends.add(anna, through_defaults={'friendship_date': date.datetime(...)})
where currently we would have to do
joe.friends.add(anna, through_defaults={'friendship_date': date.datetime(...)})
anna.friends.add(joe, through_defaults={'friendship_date': date.datetime(...)})

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 django/db/models/fields/related.py** | 1091 | 1167| 524 | 524 | 13571 | 
| 2 | **1 django/db/models/fields/related.py** | 1568 | 1603| 475 | 999 | 13571 | 
| **-> 3 <-** | **1 django/db/models/fields/related.py** | 1202 | 1325| 1010 | 2009 | 13571 | 
| 4 | **1 django/db/models/fields/related.py** | 1327 | 1399| 616 | 2625 | 13571 | 
| 5 | **1 django/db/models/fields/related.py** | 1044 | 1088| 407 | 3032 | 13571 | 
| 6 | **1 django/db/models/fields/related.py** | 1401 | 1431| 322 | 3354 | 13571 | 
| **-> 7 <-** | **2 django/db/models/fields/related_descriptors.py** | 932 | 945| 192 | 3546 | 23856 | 
| 8 | **2 django/db/models/fields/related_descriptors.py** | 1100 | 1146| 489 | 4035 | 23856 | 
| 9 | **2 django/db/models/fields/related_descriptors.py** | 998 | 1025| 334 | 4369 | 23856 | 
| 10 | **2 django/db/models/fields/related_descriptors.py** | 1148 | 1189| 392 | 4761 | 23856 | 
| 11 | **2 django/db/models/fields/related.py** | 1605 | 1621| 286 | 5047 | 23856 | 
| 12 | **2 django/db/models/fields/related_descriptors.py** | 946 | 970| 240 | 5287 | 23856 | 
| 13 | **2 django/db/models/fields/related_descriptors.py** | 972 | 997| 241 | 5528 | 23856 | 
| 14 | **2 django/db/models/fields/related_descriptors.py** | 893 | 930| 374 | 5902 | 23856 | 
| 15 | **2 django/db/models/fields/related.py** | 1433 | 1473| 399 | 6301 | 23856 | 
| 16 | **2 django/db/models/fields/related.py** | 1623 | 1657| 266 | 6567 | 23856 | 
| 17 | 3 django/db/backends/base/schema.py | 873 | 892| 296 | 6863 | 35086 | 
| 18 | **3 django/db/models/fields/related_descriptors.py** | 794 | 853| 576 | 7439 | 35086 | 
| 19 | 4 django/db/backends/sqlite3/schema.py | 366 | 412| 444 | 7883 | 39040 | 
| 20 | **4 django/db/models/fields/related_descriptors.py** | 855 | 869| 190 | 8073 | 39040 | 
| 21 | **4 django/db/models/fields/related_descriptors.py** | 663 | 717| 511 | 8584 | 39040 | 
| 22 | **4 django/db/models/fields/related.py** | 759 | 822| 499 | 9083 | 39040 | 
| 23 | **4 django/db/models/fields/related_descriptors.py** | 1071 | 1098| 338 | 9421 | 39040 | 
| 24 | 5 django/db/models/fields/reverse_related.py | 245 | 288| 318 | 9739 | 41139 | 
| 25 | **5 django/db/models/fields/related_descriptors.py** | 1058 | 1069| 138 | 9877 | 41139 | 
| 26 | **5 django/db/models/fields/related_descriptors.py** | 633 | 662| 254 | 10131 | 41139 | 
| 27 | 6 django/contrib/contenttypes/fields.py | 21 | 109| 571 | 10702 | 46551 | 
| 28 | **6 django/db/models/fields/related_descriptors.py** | 599 | 631| 323 | 11025 | 46551 | 
| 29 | **6 django/db/models/fields/related_descriptors.py** | 719 | 745| 222 | 11247 | 46551 | 
| 30 | **6 django/db/models/fields/related.py** | 1169 | 1200| 180 | 11427 | 46551 | 
| 31 | **6 django/db/models/fields/related_descriptors.py** | 1027 | 1056| 272 | 11699 | 46551 | 
| 32 | 6 django/contrib/contenttypes/fields.py | 431 | 452| 248 | 11947 | 46551 | 
| 33 | 7 django/db/models/base.py | 1325 | 1355| 244 | 12191 | 61511 | 
| 34 | **7 django/db/models/fields/related_descriptors.py** | 111 | 145| 405 | 12596 | 61511 | 
| 35 | 7 django/db/models/fields/reverse_related.py | 177 | 220| 351 | 12947 | 61511 | 
| 36 | **7 django/db/models/fields/related.py** | 444 | 485| 273 | 13220 | 61511 | 
| 37 | **7 django/db/models/fields/related_descriptors.py** | 871 | 891| 190 | 13410 | 61511 | 
| 38 | **7 django/db/models/fields/related_descriptors.py** | 542 | 597| 478 | 13888 | 61511 | 
| 39 | 8 django/core/serializers/xml_serializer.py | 111 | 147| 330 | 14218 | 64905 | 
| 40 | **8 django/db/models/fields/related.py** | 1501 | 1518| 184 | 14402 | 64905 | 
| 41 | **8 django/db/models/fields/related.py** | 994 | 1041| 368 | 14770 | 64905 | 
| 42 | **8 django/db/models/fields/related.py** | 738 | 756| 222 | 14992 | 64905 | 
| 43 | 8 django/db/models/fields/reverse_related.py | 223 | 242| 147 | 15139 | 64905 | 
| 44 | 9 django/db/models/fields/related_lookups.py | 62 | 99| 447 | 15586 | 66351 | 
| 45 | **9 django/db/models/fields/related.py** | 255 | 282| 269 | 15855 | 66351 | 
| 46 | **9 django/db/models/fields/related.py** | 616 | 638| 185 | 16040 | 66351 | 
| 47 | **9 django/db/models/fields/related.py** | 1538 | 1566| 275 | 16315 | 66351 | 
| 48 | **9 django/db/models/fields/related.py** | 847 | 873| 240 | 16555 | 66351 | 
| 49 | **9 django/db/models/fields/related_descriptors.py** | 300 | 314| 182 | 16737 | 66351 | 
| 50 | 9 django/contrib/contenttypes/fields.py | 333 | 355| 185 | 16922 | 66351 | 
| 51 | **9 django/db/models/fields/related.py** | 509 | 563| 409 | 17331 | 66351 | 
| 52 | **9 django/db/models/fields/related.py** | 600 | 614| 186 | 17517 | 66351 | 
| 53 | 9 django/db/models/fields/related_lookups.py | 102 | 117| 212 | 17729 | 66351 | 
| 54 | **9 django/db/models/fields/related.py** | 127 | 154| 202 | 17931 | 66351 | 
| 55 | **9 django/db/models/fields/related.py** | 284 | 318| 293 | 18224 | 66351 | 
| 56 | 9 django/db/models/fields/reverse_related.py | 19 | 115| 635 | 18859 | 66351 | 
| 57 | **9 django/db/models/fields/related.py** | 1520 | 1536| 184 | 19043 | 66351 | 
| 58 | **9 django/db/models/fields/related.py** | 487 | 507| 138 | 19181 | 66351 | 
| 59 | 9 django/db/models/fields/related_lookups.py | 46 | 60| 224 | 19405 | 66351 | 
| 60 | **9 django/db/models/fields/related.py** | 1475 | 1499| 295 | 19700 | 66351 | 
| 61 | **9 django/db/models/fields/related_descriptors.py** | 485 | 539| 338 | 20038 | 66351 | 
| 62 | **9 django/db/models/fields/related.py** | 824 | 845| 169 | 20207 | 66351 | 
| 63 | **9 django/db/models/fields/related.py** | 952 | 962| 121 | 20328 | 66351 | 
| 64 | 9 django/db/models/fields/related_lookups.py | 119 | 155| 244 | 20572 | 66351 | 
| 65 | **9 django/db/models/fields/related_descriptors.py** | 335 | 354| 156 | 20728 | 66351 | 
| 66 | **9 django/db/models/fields/related_descriptors.py** | 73 | 109| 264 | 20992 | 66351 | 
| 67 | 9 django/contrib/contenttypes/fields.py | 677 | 702| 254 | 21246 | 66351 | 
| 68 | 9 django/db/models/fields/reverse_related.py | 129 | 147| 172 | 21418 | 66351 | 
| 69 | **9 django/db/models/fields/related.py** | 171 | 188| 166 | 21584 | 66351 | 
| 70 | **9 django/db/models/fields/related.py** | 108 | 125| 155 | 21739 | 66351 | 
| 71 | 10 django/db/migrations/operations/models.py | 453 | 469| 203 | 21942 | 73097 | 
| 72 | **10 django/db/models/fields/related.py** | 964 | 991| 215 | 22157 | 73097 | 
| 73 | **10 django/db/models/fields/related.py** | 565 | 598| 318 | 22475 | 73097 | 
| 74 | 10 django/db/backends/base/schema.py | 514 | 553| 470 | 22945 | 73097 | 
| 75 | 10 django/db/models/base.py | 1374 | 1429| 491 | 23436 | 73097 | 
| 76 | **10 django/db/models/fields/related_descriptors.py** | 194 | 266| 740 | 24176 | 73097 | 
| 77 | 10 django/db/backends/base/schema.py | 1 | 37| 287 | 24463 | 73097 | 
| 78 | 11 django/db/migrations/autodetector.py | 904 | 978| 812 | 25275 | 84768 | 
| 79 | **11 django/db/models/fields/related.py** | 156 | 169| 144 | 25419 | 84768 | 
| 80 | **11 django/db/models/fields/related_descriptors.py** | 421 | 482| 625 | 26044 | 84768 | 
| 81 | **11 django/db/models/fields/related.py** | 190 | 254| 673 | 26717 | 84768 | 
| 82 | **11 django/db/models/fields/related_descriptors.py** | 356 | 372| 184 | 26901 | 84768 | 
| 83 | 12 django/db/migrations/operations/fields.py | 220 | 239| 205 | 27106 | 88035 | 
| 84 | **12 django/db/models/fields/related_descriptors.py** | 748 | 791| 314 | 27420 | 88035 | 
| 85 | 12 django/contrib/contenttypes/fields.py | 598 | 629| 278 | 27698 | 88035 | 
| 86 | 12 django/contrib/contenttypes/fields.py | 220 | 271| 459 | 28157 | 88035 | 
| 87 | 13 django/contrib/auth/admin.py | 25 | 37| 128 | 28285 | 89761 | 
| 88 | 13 django/contrib/contenttypes/fields.py | 656 | 676| 188 | 28473 | 89761 | 
| 89 | 14 django/contrib/admin/options.py | 2108 | 2133| 250 | 28723 | 108114 | 
| 90 | 15 django/db/backends/mysql/schema.py | 47 | 57| 138 | 28861 | 109199 | 
| 91 | 15 django/db/models/fields/related_lookups.py | 1 | 23| 170 | 29031 | 109199 | 
| 92 | 16 django/db/models/sql/query.py | 1562 | 1586| 227 | 29258 | 130389 | 
| 93 | 16 django/contrib/contenttypes/fields.py | 630 | 654| 214 | 29472 | 130389 | 
| 94 | 16 django/db/migrations/operations/models.py | 345 | 394| 493 | 29965 | 130389 | 
| 95 | **16 django/db/models/fields/related.py** | 658 | 682| 218 | 30183 | 130389 | 
| 96 | 16 django/db/migrations/operations/fields.py | 91 | 101| 125 | 30308 | 130389 | 
| 97 | 16 django/db/backends/base/schema.py | 687 | 756| 752 | 31060 | 130389 | 
| 98 | 16 django/db/migrations/operations/fields.py | 241 | 251| 146 | 31206 | 130389 | 
| 99 | **16 django/db/models/fields/related_descriptors.py** | 147 | 192| 445 | 31651 | 130389 | 
| 100 | 16 django/db/backends/base/schema.py | 757 | 797| 511 | 32162 | 130389 | 
| 101 | 17 django/db/models/options.py | 512 | 540| 231 | 32393 | 137255 | 
| 102 | 17 django/db/models/base.py | 1574 | 1622| 348 | 32741 | 137255 | 
| 103 | 18 django/core/serializers/base.py | 273 | 294| 186 | 32927 | 139648 | 
| 104 | 18 django/db/migrations/operations/models.py | 596 | 612| 215 | 33142 | 139648 | 
| 105 | 18 django/db/backends/base/schema.py | 1019 | 1034| 170 | 33312 | 139648 | 
| 106 | **18 django/db/models/fields/related.py** | 1 | 34| 240 | 33552 | 139648 | 
| 107 | 18 django/db/migrations/operations/models.py | 533 | 546| 139 | 33691 | 139648 | 
| 108 | **18 django/db/models/fields/related.py** | 684 | 696| 116 | 33807 | 139648 | 
| 109 | 19 django/db/models/sql/datastructures.py | 119 | 139| 144 | 33951 | 141077 | 
| 110 | 20 django/db/models/deletion.py | 1 | 61| 467 | 34418 | 144047 | 
| 111 | 20 django/db/models/sql/datastructures.py | 106 | 117| 133 | 34551 | 144047 | 
| 112 | 20 django/db/models/sql/query.py | 2257 | 2312| 827 | 35378 | 144047 | 
| 113 | 21 django/core/serializers/python.py | 62 | 75| 129 | 35507 | 145282 | 
| 114 | 21 django/db/models/sql/query.py | 1616 | 1643| 351 | 35858 | 145282 | 
| 115 | 21 django/contrib/contenttypes/fields.py | 274 | 331| 434 | 36292 | 145282 | 
| 116 | 21 django/db/backends/sqlite3/schema.py | 222 | 304| 729 | 37021 | 145282 | 
| 117 | **21 django/db/models/fields/related.py** | 83 | 106| 162 | 37183 | 145282 | 
| 118 | 21 django/db/migrations/operations/fields.py | 103 | 123| 223 | 37406 | 145282 | 
| 119 | 21 django/db/models/options.py | 481 | 493| 109 | 37515 | 145282 | 
| 120 | **21 django/db/models/fields/related.py** | 918 | 950| 279 | 37794 | 145282 | 
| 121 | **21 django/db/models/fields/related.py** | 896 | 916| 178 | 37972 | 145282 | 
| 122 | 22 django/db/models/lookups.py | 345 | 374| 337 | 38309 | 149436 | 
| 123 | 22 django/db/backends/base/schema.py | 1036 | 1050| 123 | 38432 | 149436 | 
| 124 | 22 django/db/backends/base/schema.py | 555 | 613| 676 | 39108 | 149436 | 
| 125 | 23 django/db/backends/ddl_references.py | 124 | 155| 283 | 39391 | 150755 | 
| 126 | 24 django/db/models/fields/__init__.py | 897 | 973| 477 | 39868 | 167749 | 
| 127 | 24 django/db/backends/sqlite3/schema.py | 347 | 364| 270 | 40138 | 167749 | 
| 128 | 24 django/db/models/base.py | 399 | 503| 903 | 41041 | 167749 | 
| 129 | 24 django/db/models/sql/query.py | 1406 | 1484| 734 | 41775 | 167749 | 
| 130 | 24 django/contrib/admin/options.py | 422 | 465| 350 | 42125 | 167749 | 
| 131 | 24 django/db/migrations/operations/fields.py | 302 | 355| 535 | 42660 | 167749 | 
| 132 | 24 django/db/migrations/operations/fields.py | 357 | 384| 289 | 42949 | 167749 | 
| 133 | 24 django/contrib/contenttypes/fields.py | 565 | 596| 308 | 43257 | 167749 | 
| 134 | 24 django/db/models/lookups.py | 313 | 343| 263 | 43520 | 167749 | 
| 135 | 24 django/db/migrations/autodetector.py | 1060 | 1077| 180 | 43700 | 167749 | 
| 136 | 24 django/db/backends/sqlite3/schema.py | 139 | 220| 820 | 44520 | 167749 | 
| 137 | 24 django/db/migrations/autodetector.py | 796 | 845| 570 | 45090 | 167749 | 
| 138 | 24 django/db/backends/mysql/schema.py | 74 | 88| 201 | 45291 | 167749 | 
| 139 | 24 django/db/migrations/operations/models.py | 522 | 531| 129 | 45420 | 167749 | 
| 140 | 24 django/db/backends/base/schema.py | 358 | 372| 182 | 45602 | 167749 | 
| 141 | 24 django/db/models/base.py | 1815 | 1866| 351 | 45953 | 167749 | 
| 142 | 25 django/db/migrations/operations/utils.py | 1 | 14| 138 | 46091 | 168229 | 
| 143 | 26 django/db/migrations/questioner.py | 56 | 81| 220 | 46311 | 170303 | 
| 144 | 26 django/db/migrations/autodetector.py | 200 | 222| 239 | 46550 | 170303 | 
| 145 | 26 django/db/backends/base/schema.py | 614 | 686| 792 | 47342 | 170303 | 
| 146 | 26 django/db/models/fields/__init__.py | 292 | 320| 205 | 47547 | 170303 | 
| 147 | 26 django/db/models/sql/query.py | 641 | 688| 511 | 48058 | 170303 | 
| 148 | 26 django/core/serializers/xml_serializer.py | 273 | 303| 278 | 48336 | 170303 | 
| 149 | **26 django/db/models/fields/related.py** | 421 | 441| 166 | 48502 | 170303 | 
| 150 | 26 django/db/migrations/operations/fields.py | 253 | 271| 161 | 48663 | 170303 | 
| 151 | 26 django/db/migrations/operations/fields.py | 39 | 67| 285 | 48948 | 170303 | 
| 152 | 26 django/db/models/base.py | 1480 | 1512| 231 | 49179 | 170303 | 
| 153 | 26 django/db/backends/sqlite3/schema.py | 329 | 345| 173 | 49352 | 170303 | 
| 154 | 26 django/contrib/admin/options.py | 243 | 275| 357 | 49709 | 170303 | 
| 155 | 26 django/db/models/fields/reverse_related.py | 1 | 16| 110 | 49819 | 170303 | 
| 156 | 26 django/db/migrations/questioner.py | 143 | 160| 183 | 50002 | 170303 | 
| 157 | 26 django/db/models/options.py | 244 | 276| 343 | 50345 | 170303 | 
| 158 | 26 django/db/models/base.py | 1718 | 1789| 565 | 50910 | 170303 | 
| 159 | 26 django/db/migrations/autodetector.py | 224 | 237| 199 | 51109 | 170303 | 
| 160 | 26 django/db/backends/sqlite3/schema.py | 306 | 327| 218 | 51327 | 170303 | 
| 161 | 26 django/db/backends/base/schema.py | 484 | 512| 289 | 51616 | 170303 | 
| 162 | 26 django/db/models/base.py | 1 | 45| 289 | 51905 | 170303 | 
| 163 | 26 django/db/migrations/operations/models.py | 1 | 38| 238 | 52143 | 170303 | 
| 164 | 26 django/db/models/base.py | 164 | 206| 406 | 52549 | 170303 | 
| 165 | 27 django/forms/models.py | 309 | 348| 387 | 52936 | 181798 | 
| 166 | 27 django/db/backends/mysql/schema.py | 90 | 108| 192 | 53128 | 181798 | 
| 167 | 27 django/db/backends/ddl_references.py | 1 | 38| 213 | 53341 | 181798 | 
| 168 | 27 django/db/models/fields/reverse_related.py | 149 | 174| 269 | 53610 | 181798 | 
| 169 | 28 django/db/backends/mysql/base.py | 247 | 281| 247 | 53857 | 184736 | 
| 170 | 29 django/contrib/admin/utils.py | 435 | 463| 225 | 54082 | 188809 | 
| 171 | 29 django/forms/models.py | 1337 | 1364| 209 | 54291 | 188809 | 
| 172 | 30 django/db/backends/oracle/schema.py | 79 | 123| 578 | 54869 | 190560 | 
| 173 | 30 django/db/migrations/operations/models.py | 614 | 627| 137 | 55006 | 190560 | 
| 174 | 30 django/db/models/options.py | 278 | 296| 136 | 55142 | 190560 | 
| 175 | 30 django/db/backends/sqlite3/schema.py | 85 | 98| 181 | 55323 | 190560 | 
| 176 | 30 django/db/backends/sqlite3/schema.py | 100 | 137| 486 | 55809 | 190560 | 
| 177 | 30 django/db/models/sql/query.py | 690 | 725| 389 | 56198 | 190560 | 
| 178 | 30 django/db/models/sql/datastructures.py | 1 | 23| 153 | 56351 | 190560 | 
| 179 | 30 django/db/models/fields/__init__.py | 242 | 290| 320 | 56671 | 190560 | 
| 180 | 30 django/db/models/sql/query.py | 1645 | 1714| 759 | 57430 | 190560 | 
| 181 | 30 django/core/serializers/base.py | 232 | 249| 208 | 57638 | 190560 | 
| 182 | 30 django/contrib/contenttypes/fields.py | 475 | 504| 222 | 57860 | 190560 | 
| 183 | 30 django/db/models/sql/query.py | 1964 | 1986| 229 | 58089 | 190560 | 
| 184 | 30 django/db/models/options.py | 667 | 701| 378 | 58467 | 190560 | 
| 185 | 30 django/db/backends/mysql/schema.py | 1 | 45| 441 | 58908 | 190560 | 
| 186 | 30 django/db/migrations/autodetector.py | 707 | 794| 789 | 59697 | 190560 | 
| 187 | 30 django/db/models/sql/query.py | 1792 | 1828| 318 | 60015 | 190560 | 
| 188 | 30 django/db/models/fields/__init__.py | 348 | 374| 199 | 60214 | 190560 | 
| 189 | **30 django/db/models/fields/related_descriptors.py** | 1 | 70| 602 | 60816 | 190560 | 
| 190 | 30 django/contrib/contenttypes/fields.py | 174 | 218| 411 | 61227 | 190560 | 
| 191 | 31 django/contrib/admin/widgets.py | 200 | 226| 216 | 61443 | 194416 | 
| 192 | 31 django/db/migrations/operations/models.py | 120 | 239| 827 | 62270 | 194416 | 
| 193 | 31 django/db/models/sql/query.py | 2229 | 2255| 176 | 62446 | 194416 | 
| 194 | 32 django/db/models/__init__.py | 1 | 49| 548 | 62994 | 194964 | 
| 195 | 32 django/db/models/fields/__init__.py | 1095 | 1108| 104 | 63098 | 194964 | 
| 196 | 32 django/contrib/contenttypes/fields.py | 507 | 563| 439 | 63537 | 194964 | 
| 197 | 32 django/db/models/sql/query.py | 1865 | 1906| 342 | 63879 | 194964 | 
| 198 | 32 django/db/migrations/autodetector.py | 89 | 101| 118 | 63997 | 194964 | 
| 199 | 32 django/contrib/admin/options.py | 368 | 420| 504 | 64501 | 194964 | 
| 200 | 32 django/db/migrations/autodetector.py | 525 | 671| 1109 | 65610 | 194964 | 
| 201 | 32 django/db/models/base.py | 1236 | 1265| 242 | 65852 | 194964 | 
| 202 | 32 django/db/migrations/questioner.py | 207 | 224| 171 | 66023 | 194964 | 
| 203 | 33 django/contrib/contenttypes/admin.py | 1 | 80| 615 | 66638 | 195989 | 
| 204 | 33 django/db/models/fields/__init__.py | 487 | 527| 329 | 66967 | 195989 | 


## Patch

```diff
diff --git a/django/db/models/fields/related.py b/django/db/models/fields/related.py
--- a/django/db/models/fields/related.py
+++ b/django/db/models/fields/related.py
@@ -1234,18 +1234,6 @@ def _check_relationship_model(self, from_model=None, **kwargs):
                 to_model_name = to_model._meta.object_name
             relationship_model_name = self.remote_field.through._meta.object_name
             self_referential = from_model == to_model
-
-            # Check symmetrical attribute.
-            if (self_referential and self.remote_field.symmetrical and
-                    not self.remote_field.through._meta.auto_created):
-                errors.append(
-                    checks.Error(
-                        'Many-to-many fields with intermediate tables must not be symmetrical.',
-                        obj=self,
-                        id='fields.E332',
-                    )
-                )
-
             # Count foreign keys in intermediate model
             if self_referential:
                 seen_self = sum(
diff --git a/django/db/models/fields/related_descriptors.py b/django/db/models/fields/related_descriptors.py
--- a/django/db/models/fields/related_descriptors.py
+++ b/django/db/models/fields/related_descriptors.py
@@ -938,11 +938,14 @@ def add(self, *objs, through_defaults=None):
                     through_defaults=through_defaults,
                 )
                 # If this is a symmetrical m2m relation to self, add the mirror
-                # entry in the m2m table. `through_defaults` aren't used here
-                # because of the system check error fields.E332: Many-to-many
-                # fields with intermediate tables must not be symmetrical.
+                # entry in the m2m table.
                 if self.symmetrical:
-                    self._add_items(self.target_field_name, self.source_field_name, *objs)
+                    self._add_items(
+                        self.target_field_name,
+                        self.source_field_name,
+                        *objs,
+                        through_defaults=through_defaults,
+                    )
         add.alters_data = True
 
         def remove(self, *objs):

```

## Test Patch

```diff
diff --git a/tests/invalid_models_tests/test_relative_fields.py b/tests/invalid_models_tests/test_relative_fields.py
--- a/tests/invalid_models_tests/test_relative_fields.py
+++ b/tests/invalid_models_tests/test_relative_fields.py
@@ -260,24 +260,6 @@ class Group(models.Model):
         field = Group._meta.get_field('members')
         self.assertEqual(field.check(from_model=Group), [])
 
-    def test_symmetrical_self_referential_field(self):
-        class Person(models.Model):
-            # Implicit symmetrical=False.
-            friends = models.ManyToManyField('self', through="Relationship")
-
-        class Relationship(models.Model):
-            first = models.ForeignKey(Person, models.CASCADE, related_name="rel_from_set")
-            second = models.ForeignKey(Person, models.CASCADE, related_name="rel_to_set")
-
-        field = Person._meta.get_field('friends')
-        self.assertEqual(field.check(from_model=Person), [
-            Error(
-                'Many-to-many fields with intermediate tables must not be symmetrical.',
-                obj=field,
-                id='fields.E332',
-            ),
-        ])
-
     def test_too_many_foreign_keys_in_self_referential_model(self):
         class Person(models.Model):
             friends = models.ManyToManyField('self', through="InvalidRelationship", symmetrical=False)
@@ -301,52 +283,6 @@ class InvalidRelationship(models.Model):
             ),
         ])
 
-    def test_symmetric_self_reference_with_intermediate_table(self):
-        class Person(models.Model):
-            # Explicit symmetrical=True.
-            friends = models.ManyToManyField('self', through="Relationship", symmetrical=True)
-
-        class Relationship(models.Model):
-            first = models.ForeignKey(Person, models.CASCADE, related_name="rel_from_set")
-            second = models.ForeignKey(Person, models.CASCADE, related_name="rel_to_set")
-
-        field = Person._meta.get_field('friends')
-        self.assertEqual(field.check(from_model=Person), [
-            Error(
-                'Many-to-many fields with intermediate tables must not be symmetrical.',
-                obj=field,
-                id='fields.E332',
-            ),
-        ])
-
-    def test_symmetric_self_reference_with_intermediate_table_and_through_fields(self):
-        """
-        Using through_fields in a m2m with an intermediate model shouldn't
-        mask its incompatibility with symmetry.
-        """
-        class Person(models.Model):
-            # Explicit symmetrical=True.
-            friends = models.ManyToManyField(
-                'self',
-                symmetrical=True,
-                through="Relationship",
-                through_fields=('first', 'second'),
-            )
-
-        class Relationship(models.Model):
-            first = models.ForeignKey(Person, models.CASCADE, related_name="rel_from_set")
-            second = models.ForeignKey(Person, models.CASCADE, related_name="rel_to_set")
-            referee = models.ForeignKey(Person, models.CASCADE, related_name="referred")
-
-        field = Person._meta.get_field('friends')
-        self.assertEqual(field.check(from_model=Person), [
-            Error(
-                'Many-to-many fields with intermediate tables must not be symmetrical.',
-                obj=field,
-                id='fields.E332',
-            ),
-        ])
-
     def test_foreign_key_to_abstract_model(self):
         class AbstractModel(models.Model):
             class Meta:
diff --git a/tests/m2m_recursive/models.py b/tests/m2m_recursive/models.py
--- a/tests/m2m_recursive/models.py
+++ b/tests/m2m_recursive/models.py
@@ -22,7 +22,14 @@
 class Person(models.Model):
     name = models.CharField(max_length=20)
     friends = models.ManyToManyField('self')
+    colleagues = models.ManyToManyField('self', symmetrical=True, through='Colleague')
     idols = models.ManyToManyField('self', symmetrical=False, related_name='stalkers')
 
     def __str__(self):
         return self.name
+
+
+class Colleague(models.Model):
+    first = models.ForeignKey(Person, models.CASCADE)
+    second = models.ForeignKey(Person, models.CASCADE, related_name='+')
+    first_meet = models.DateField()
diff --git a/tests/m2m_recursive/tests.py b/tests/m2m_recursive/tests.py
--- a/tests/m2m_recursive/tests.py
+++ b/tests/m2m_recursive/tests.py
@@ -1,3 +1,5 @@
+import datetime
+
 from django.test import TestCase
 
 from .models import Person
@@ -59,3 +61,59 @@ def test_recursive_m2m_related_to_self(self):
         self.a.idols.add(self.a)
         self.assertSequenceEqual(self.a.idols.all(), [self.a])
         self.assertSequenceEqual(self.a.stalkers.all(), [self.a])
+
+
+class RecursiveSymmetricalM2MThroughTests(TestCase):
+    @classmethod
+    def setUpTestData(cls):
+        cls.a, cls.b, cls.c, cls.d = [
+            Person.objects.create(name=name)
+            for name in ['Anne', 'Bill', 'Chuck', 'David']
+        ]
+        cls.a.colleagues.add(cls.b, cls.c, through_defaults={
+            'first_meet': datetime.date(2013, 1, 5),
+        })
+        # Add m2m for Anne and Chuck in reverse direction.
+        cls.d.colleagues.add(cls.a, cls.c, through_defaults={
+            'first_meet': datetime.date(2015, 6, 15),
+        })
+
+    def test_recursive_m2m_all(self):
+        for person, colleagues in (
+            (self.a, [self.b, self.c, self.d]),
+            (self.b, [self.a]),
+            (self.c, [self.a, self.d]),
+            (self.d, [self.a, self.c]),
+        ):
+            with self.subTest(person=person):
+                self.assertSequenceEqual(person.colleagues.all(), colleagues)
+
+    def test_recursive_m2m_reverse_add(self):
+        # Add m2m for Anne in reverse direction.
+        self.b.colleagues.add(self.a, through_defaults={
+            'first_meet': datetime.date(2013, 1, 5),
+        })
+        self.assertSequenceEqual(self.a.colleagues.all(), [self.b, self.c, self.d])
+        self.assertSequenceEqual(self.b.colleagues.all(), [self.a])
+
+    def test_recursive_m2m_remove(self):
+        self.b.colleagues.remove(self.a)
+        self.assertSequenceEqual(self.a.colleagues.all(), [self.c, self.d])
+        self.assertSequenceEqual(self.b.colleagues.all(), [])
+
+    def test_recursive_m2m_clear(self):
+        # Clear m2m for Anne.
+        self.a.colleagues.clear()
+        self.assertSequenceEqual(self.a.friends.all(), [])
+        # Reverse m2m relationships is removed.
+        self.assertSequenceEqual(self.c.colleagues.all(), [self.d])
+        self.assertSequenceEqual(self.d.colleagues.all(), [self.c])
+
+    def test_recursive_m2m_set(self):
+        # Set new relationships for Chuck.
+        self.c.colleagues.set([self.b, self.d], through_defaults={
+            'first_meet': datetime.date(2013, 1, 5),
+        })
+        self.assertSequenceEqual(self.c.colleagues.order_by('name'), [self.b, self.d])
+        # Reverse m2m relationships is removed.
+        self.assertSequenceEqual(self.a.colleagues.order_by('name'), [self.b, self.d])
diff --git a/tests/m2m_through/models.py b/tests/m2m_through/models.py
--- a/tests/m2m_through/models.py
+++ b/tests/m2m_through/models.py
@@ -72,6 +72,7 @@ class TestNoDefaultsOrNulls(models.Model):
 class PersonSelfRefM2M(models.Model):
     name = models.CharField(max_length=5)
     friends = models.ManyToManyField('self', through="Friendship", symmetrical=False)
+    sym_friends = models.ManyToManyField('self', through='SymmetricalFriendship', symmetrical=True)
 
     def __str__(self):
         return self.name
@@ -83,6 +84,12 @@ class Friendship(models.Model):
     date_friended = models.DateTimeField()
 
 
+class SymmetricalFriendship(models.Model):
+    first = models.ForeignKey(PersonSelfRefM2M, models.CASCADE)
+    second = models.ForeignKey(PersonSelfRefM2M, models.CASCADE, related_name='+')
+    date_friended = models.DateField()
+
+
 # Custom through link fields
 class Event(models.Model):
     title = models.CharField(max_length=50)
diff --git a/tests/m2m_through/tests.py b/tests/m2m_through/tests.py
--- a/tests/m2m_through/tests.py
+++ b/tests/m2m_through/tests.py
@@ -1,4 +1,4 @@
-from datetime import datetime
+from datetime import date, datetime
 from operator import attrgetter
 
 from django.db import IntegrityError
@@ -7,7 +7,7 @@
 from .models import (
     CustomMembership, Employee, Event, Friendship, Group, Ingredient,
     Invitation, Membership, Person, PersonSelfRefM2M, Recipe, RecipeIngredient,
-    Relationship,
+    Relationship, SymmetricalFriendship,
 )
 
 
@@ -401,7 +401,7 @@ def test_self_referential_non_symmetrical_clear_first_side(self):
             attrgetter("name")
         )
 
-    def test_self_referential_symmetrical(self):
+    def test_self_referential_non_symmetrical_both(self):
         tony = PersonSelfRefM2M.objects.create(name="Tony")
         chris = PersonSelfRefM2M.objects.create(name="Chris")
         Friendship.objects.create(
@@ -439,6 +439,71 @@ def test_through_fields_self_referential(self):
             attrgetter('name')
         )
 
+    def test_self_referential_symmetrical(self):
+        tony = PersonSelfRefM2M.objects.create(name='Tony')
+        chris = PersonSelfRefM2M.objects.create(name='Chris')
+        SymmetricalFriendship.objects.create(
+            first=tony, second=chris, date_friended=date.today(),
+        )
+        self.assertSequenceEqual(tony.sym_friends.all(), [chris])
+        # Manually created symmetrical m2m relation doesn't add mirror entry
+        # automatically.
+        self.assertSequenceEqual(chris.sym_friends.all(), [])
+        SymmetricalFriendship.objects.create(
+            first=chris, second=tony, date_friended=date.today()
+        )
+        self.assertSequenceEqual(chris.sym_friends.all(), [tony])
+
+    def test_add_on_symmetrical_m2m_with_intermediate_model(self):
+        tony = PersonSelfRefM2M.objects.create(name='Tony')
+        chris = PersonSelfRefM2M.objects.create(name='Chris')
+        date_friended = date(2017, 1, 3)
+        tony.sym_friends.add(chris, through_defaults={'date_friended': date_friended})
+        self.assertSequenceEqual(tony.sym_friends.all(), [chris])
+        self.assertSequenceEqual(chris.sym_friends.all(), [tony])
+        friendship = tony.symmetricalfriendship_set.get()
+        self.assertEqual(friendship.date_friended, date_friended)
+
+    def test_set_on_symmetrical_m2m_with_intermediate_model(self):
+        tony = PersonSelfRefM2M.objects.create(name='Tony')
+        chris = PersonSelfRefM2M.objects.create(name='Chris')
+        anne = PersonSelfRefM2M.objects.create(name='Anne')
+        kate = PersonSelfRefM2M.objects.create(name='Kate')
+        date_friended_add = date(2013, 1, 5)
+        date_friended_set = date.today()
+        tony.sym_friends.add(
+            anne, chris,
+            through_defaults={'date_friended': date_friended_add},
+        )
+        tony.sym_friends.set(
+            [anne, kate],
+            through_defaults={'date_friended': date_friended_set},
+        )
+        self.assertSequenceEqual(tony.sym_friends.all(), [anne, kate])
+        self.assertSequenceEqual(anne.sym_friends.all(), [tony])
+        self.assertSequenceEqual(kate.sym_friends.all(), [tony])
+        self.assertEqual(
+            kate.symmetricalfriendship_set.get().date_friended,
+            date_friended_set,
+        )
+        # Date is preserved.
+        self.assertEqual(
+            anne.symmetricalfriendship_set.get().date_friended,
+            date_friended_add,
+        )
+        # Recreate relationship.
+        tony.sym_friends.set(
+            [anne],
+            clear=True,
+            through_defaults={'date_friended': date_friended_set},
+        )
+        self.assertSequenceEqual(tony.sym_friends.all(), [anne])
+        self.assertSequenceEqual(anne.sym_friends.all(), [tony])
+        self.assertEqual(
+            anne.symmetricalfriendship_set.get().date_friended,
+            date_friended_set,
+        )
+
 
 class M2mThroughToFieldsTests(TestCase):
     @classmethod

```


## Code snippets

### 1 - django/db/models/fields/related.py:

Start line: 1091, End line: 1167

```python
class ManyToManyField(RelatedField):
    """
    Provide a many-to-many relation by using an intermediary model that
    holds two ForeignKey fields pointed at the two sides of the relation.

    Unless a ``through`` model was provided, ManyToManyField will use the
    create_many_to_many_intermediary_model factory to automatically generate
    the intermediary model.
    """

    # Field flags
    many_to_many = True
    many_to_one = False
    one_to_many = False
    one_to_one = False

    rel_class = ManyToManyRel

    description = _("Many-to-many relationship")

    def __init__(self, to, related_name=None, related_query_name=None,
                 limit_choices_to=None, symmetrical=None, through=None,
                 through_fields=None, db_constraint=True, db_table=None,
                 swappable=True, **kwargs):
        try:
            to._meta
        except AttributeError:
            assert isinstance(to, str), (
                "%s(%r) is invalid. First parameter to ManyToManyField must be "
                "either a model, a model name, or the string %r" %
                (self.__class__.__name__, to, RECURSIVE_RELATIONSHIP_CONSTANT)
            )

        if symmetrical is None:
            symmetrical = (to == RECURSIVE_RELATIONSHIP_CONSTANT)

        if through is not None:
            assert db_table is None, (
                "Cannot specify a db_table if an intermediary model is used."
            )

        kwargs['rel'] = self.rel_class(
            self, to,
            related_name=related_name,
            related_query_name=related_query_name,
            limit_choices_to=limit_choices_to,
            symmetrical=symmetrical,
            through=through,
            through_fields=through_fields,
            db_constraint=db_constraint,
        )
        self.has_null_arg = 'null' in kwargs

        super().__init__(**kwargs)

        self.db_table = db_table
        self.swappable = swappable

    def check(self, **kwargs):
        return [
            *super().check(**kwargs),
            *self._check_unique(**kwargs),
            *self._check_relationship_model(**kwargs),
            *self._check_ignored_options(**kwargs),
            *self._check_table_uniqueness(**kwargs),
        ]

    def _check_unique(self, **kwargs):
        if self.unique:
            return [
                checks.Error(
                    'ManyToManyFields cannot be unique.',
                    obj=self,
                    id='fields.E330',
                )
            ]
        return []
```
### 2 - django/db/models/fields/related.py:

Start line: 1568, End line: 1603

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
### 3 - django/db/models/fields/related.py:

Start line: 1202, End line: 1325

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

            # Check symmetrical attribute.
            if (self_referential and self.remote_field.symmetrical and
                    not self.remote_field.through._meta.auto_created):
                errors.append(
                    checks.Error(
                        'Many-to-many fields with intermediate tables must not be symmetrical.',
                        obj=self,
                        id='fields.E332',
                    )
                )

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

Start line: 1327, End line: 1399

```python
class ManyToManyField(RelatedField):

    def _check_relationship_model(self, from_model=None, **kwargs):

        # Validate `through_fields`.
        if self.remote_field.through_fields is not None:
            # Validate that we're given an iterable of at least two items
            # and that none of them is "falsy".
            if not (len(self.remote_field.through_fields) >= 2 and
                    self.remote_field.through_fields[0] and self.remote_field.through_fields[1]):
                errors.append(
                    checks.Error(
                        "Field specifies 'through_fields' but does not provide "
                        "the names of the two link fields that should be used "
                        "for the relation through model '%s'." % qualified_model_name,
                        hint="Make sure you specify 'through_fields' as through_fields=('field1', 'field2')",
                        obj=self,
                        id='fields.E337',
                    )
                )

            # Validate the given through fields -- they should be actual
            # fields on the through model, and also be foreign keys to the
            # expected models.
            else:
                assert from_model is not None, (
                    "ManyToManyField with intermediate "
                    "tables cannot be checked if you don't pass the model "
                    "where the field is attached to."
                )

                source, through, target = from_model, self.remote_field.through, self.remote_field.model
                source_field_name, target_field_name = self.remote_field.through_fields[:2]

                for field_name, related_model in ((source_field_name, source),
                                                  (target_field_name, target)):

                    possible_field_names = []
                    for f in through._meta.fields:
                        if hasattr(f, 'remote_field') and getattr(f.remote_field, 'model', None) == related_model:
                            possible_field_names.append(f.name)
                    if possible_field_names:
                        hint = "Did you mean one of the following foreign keys to '%s': %s?" % (
                            related_model._meta.object_name,
                            ', '.join(possible_field_names),
                        )
                    else:
                        hint = None

                    try:
                        field = through._meta.get_field(field_name)
                    except exceptions.FieldDoesNotExist:
                        errors.append(
                            checks.Error(
                                "The intermediary model '%s' has no field '%s'."
                                % (qualified_model_name, field_name),
                                hint=hint,
                                obj=self,
                                id='fields.E338',
                            )
                        )
                    else:
                        if not (hasattr(field, 'remote_field') and
                                getattr(field.remote_field, 'model', None) == related_model):
                            errors.append(
                                checks.Error(
                                    "'%s.%s' is not a foreign key to '%s'." % (
                                        through._meta.object_name, field_name,
                                        related_model._meta.object_name,
                                    ),
                                    hint=hint,
                                    obj=self,
                                    id='fields.E339',
                                )
                            )

        return errors
```
### 5 - django/db/models/fields/related.py:

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
### 6 - django/db/models/fields/related.py:

Start line: 1401, End line: 1431

```python
class ManyToManyField(RelatedField):

    def _check_table_uniqueness(self, **kwargs):
        if isinstance(self.remote_field.through, str) or not self.remote_field.through._meta.managed:
            return []
        registered_tables = {
            model._meta.db_table: model
            for model in self.opts.apps.get_models(include_auto_created=True)
            if model != self.remote_field.through and model._meta.managed
        }
        m2m_db_table = self.m2m_db_table()
        model = registered_tables.get(m2m_db_table)
        # The second condition allows multiple m2m relations on a model if
        # some point to a through model that proxies another through model.
        if model and model._meta.concrete_model != self.remote_field.through._meta.concrete_model:
            if model._meta.auto_created:
                def _get_field_name(model):
                    for field in model._meta.auto_created._meta.many_to_many:
                        if field.remote_field.through is model:
                            return field.name
                opts = model._meta.auto_created._meta
                clashing_obj = '%s.%s' % (opts.label, _get_field_name(model))
            else:
                clashing_obj = model._meta.label
            return [
                checks.Error(
                    "The field's intermediary table '%s' clashes with the "
                    "table name of '%s'." % (m2m_db_table, clashing_obj),
                    obj=self,
                    id='fields.E340',
                )
            ]
        return []
```
### 7 - django/db/models/fields/related_descriptors.py:

Start line: 932, End line: 945

```python
def create_forward_many_to_many_manager(superclass, rel, reverse):

    class ManyRelatedManager(superclass):

        def add(self, *objs, through_defaults=None):
            self._remove_prefetched_objects()
            db = router.db_for_write(self.through, instance=self.instance)
            with transaction.atomic(using=db, savepoint=False):
                self._add_items(
                    self.source_field_name, self.target_field_name, *objs,
                    through_defaults=through_defaults,
                )
                # If this is a symmetrical m2m relation to self, add the mirror
                # entry in the m2m table. `through_defaults` aren't used here
                # because of the system check error fields.E332: Many-to-many
                # fields with intermediate tables must not be symmetrical.
                if self.symmetrical:
                    self._add_items(self.target_field_name, self.source_field_name, *objs)
    # ... other code
```
### 8 - django/db/models/fields/related_descriptors.py:

Start line: 1100, End line: 1146

```python
def create_forward_many_to_many_manager(superclass, rel, reverse):

    class ManyRelatedManager(superclass):

        def _add_items(self, source_field_name, target_field_name, *objs, through_defaults=None):
            # source_field_name: the PK fieldname in join table for the source object
            # target_field_name: the PK fieldname in join table for the target object
            # *objs - objects to add. Either object instances, or primary keys of object instances.
            through_defaults = through_defaults or {}

            # If there aren't any objects, there is nothing to do.
            if objs:
                target_ids = self._get_target_ids(target_field_name, objs)
                db = router.db_for_write(self.through, instance=self.instance)
                can_ignore_conflicts, must_send_signals, can_fast_add = self._get_add_plan(db, source_field_name)
                if can_fast_add:
                    self.through._default_manager.using(db).bulk_create([
                        self.through(**{
                            '%s_id' % source_field_name: self.related_val[0],
                            '%s_id' % target_field_name: target_id,
                        })
                        for target_id in target_ids
                    ], ignore_conflicts=True)
                    return

                missing_target_ids = self._get_missing_target_ids(
                    source_field_name, target_field_name, db, target_ids
                )
                with transaction.atomic(using=db, savepoint=False):
                    if must_send_signals:
                        signals.m2m_changed.send(
                            sender=self.through, action='pre_add',
                            instance=self.instance, reverse=self.reverse,
                            model=self.model, pk_set=missing_target_ids, using=db,
                        )

                    # Add the ones that aren't there already.
                    self.through._default_manager.using(db).bulk_create([
                        self.through(**through_defaults, **{
                            '%s_id' % source_field_name: self.related_val[0],
                            '%s_id' % target_field_name: target_id,
                        })
                        for target_id in missing_target_ids
                    ], ignore_conflicts=can_ignore_conflicts)

                    if must_send_signals:
                        signals.m2m_changed.send(
                            sender=self.through, action='post_add',
                            instance=self.instance, reverse=self.reverse,
                            model=self.model, pk_set=missing_target_ids, using=db,
                        )
    # ... other code
```
### 9 - django/db/models/fields/related_descriptors.py:

Start line: 998, End line: 1025

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
            obj, created = super(ManyRelatedManager, self.db_manager(db)).get_or_create(**kwargs)
            # We only need to add() if created because if we got an object back
            # from get() then the relationship already exists.
            if created:
                self.add(obj, through_defaults=through_defaults)
            return obj, created
        get_or_create.alters_data = True

        def update_or_create(self, *, through_defaults=None, **kwargs):
            db = router.db_for_write(self.instance.__class__, instance=self.instance)
            obj, created = super(ManyRelatedManager, self.db_manager(db)).update_or_create(**kwargs)
            # We only need to add() if created because if we got an object back
            # from get() then the relationship already exists.
            if created:
                self.add(obj, through_defaults=through_defaults)
            return obj, created
        update_or_create.alters_data = True
    # ... other code
```
### 10 - django/db/models/fields/related_descriptors.py:

Start line: 1148, End line: 1189

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
                    sender=self.through, action="pre_remove",
                    instance=self.instance, reverse=self.reverse,
                    model=self.model, pk_set=old_ids, using=db,
                )
                target_model_qs = super().get_queryset()
                if target_model_qs._has_filters():
                    old_vals = target_model_qs.using(db).filter(**{
                        '%s__in' % self.target_field.target_field.attname: old_ids})
                else:
                    old_vals = old_ids
                filters = self._build_remove_filters(old_vals)
                self.through._default_manager.using(db).filter(filters).delete()

                signals.m2m_changed.send(
                    sender=self.through, action="post_remove",
                    instance=self.instance, reverse=self.reverse,
                    model=self.model, pk_set=old_ids, using=db,
                )

    return ManyRelatedManager
```
### 11 - django/db/models/fields/related.py:

Start line: 1605, End line: 1621

```python
class ManyToManyField(RelatedField):

    def contribute_to_related_class(self, cls, related):
        # Internal M2Ms (i.e., those with a related name ending with '+')
        # and swapped models don't get a related descriptor.
        if not self.remote_field.is_hidden() and not related.related_model._meta.swapped:
            setattr(cls, related.get_accessor_name(), ManyToManyDescriptor(self.remote_field, reverse=True))

        # Set up the accessors for the column names on the m2m table.
        self.m2m_column_name = partial(self._get_m2m_attr, related, 'column')
        self.m2m_reverse_name = partial(self._get_m2m_reverse_attr, related, 'column')

        self.m2m_field_name = partial(self._get_m2m_attr, related, 'name')
        self.m2m_reverse_field_name = partial(self._get_m2m_reverse_attr, related, 'name')

        get_m2m_rel = partial(self._get_m2m_attr, related, 'remote_field')
        self.m2m_target_field_name = lambda: get_m2m_rel().field_name
        get_m2m_reverse_rel = partial(self._get_m2m_reverse_attr, related, 'remote_field')
        self.m2m_reverse_target_field_name = lambda: get_m2m_reverse_rel().field_name
```
### 12 - django/db/models/fields/related_descriptors.py:

Start line: 946, End line: 970

```python
def create_forward_many_to_many_manager(superclass, rel, reverse):

    class ManyRelatedManager(superclass):
        add.alters_data = True

        def remove(self, *objs):
            self._remove_prefetched_objects()
            self._remove_items(self.source_field_name, self.target_field_name, *objs)
        remove.alters_data = True

        def clear(self):
            db = router.db_for_write(self.through, instance=self.instance)
            with transaction.atomic(using=db, savepoint=False):
                signals.m2m_changed.send(
                    sender=self.through, action="pre_clear",
                    instance=self.instance, reverse=self.reverse,
                    model=self.model, pk_set=None, using=db,
                )
                self._remove_prefetched_objects()
                filters = self._build_remove_filters(super().get_queryset().using(db))
                self.through._default_manager.using(db).filter(filters).delete()

                signals.m2m_changed.send(
                    sender=self.through, action="post_clear",
                    instance=self.instance, reverse=self.reverse,
                    model=self.model, pk_set=None, using=db,
                )
        clear.alters_data = True
    # ... other code
```
### 13 - django/db/models/fields/related_descriptors.py:

Start line: 972, End line: 997

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
                    old_ids = set(self.using(db).values_list(self.target_field.target_field.attname, flat=True))

                    new_objs = []
                    for obj in objs:
                        fk_val = (
                            self.target_field.get_foreign_related_value(obj)[0]
                            if isinstance(obj, self.model) else obj
                        )
                        if fk_val in old_ids:
                            old_ids.remove(fk_val)
                        else:
                            new_objs.append(obj)

                    self.remove(*old_ids)
                    self.add(*new_objs, through_defaults=through_defaults)
    # ... other code
```
### 14 - django/db/models/fields/related_descriptors.py:

Start line: 893, End line: 930

```python
def create_forward_many_to_many_manager(superclass, rel, reverse):

    class ManyRelatedManager(superclass):

        def get_prefetch_queryset(self, instances, queryset=None):
            if queryset is None:
                queryset = super().get_queryset()

            queryset._add_hints(instance=instances[0])
            queryset = queryset.using(queryset._db or self._db)

            query = {'%s__in' % self.query_field_name: instances}
            queryset = queryset._next_is_sticky().filter(**query)

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
            queryset = queryset.extra(select={
                '_prefetch_related_val_%s' % f.attname:
                '%s.%s' % (qn(join_table), qn(f.column)) for f in fk.local_related_fields})
            return (
                queryset,
                lambda result: tuple(
                    getattr(result, '_prefetch_related_val_%s' % f.attname)
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
### 15 - django/db/models/fields/related.py:

Start line: 1433, End line: 1473

```python
class ManyToManyField(RelatedField):

    def deconstruct(self):
        name, path, args, kwargs = super().deconstruct()
        # Handle the simpler arguments.
        if self.db_table is not None:
            kwargs['db_table'] = self.db_table
        if self.remote_field.db_constraint is not True:
            kwargs['db_constraint'] = self.remote_field.db_constraint
        # Rel needs more work.
        if isinstance(self.remote_field.model, str):
            kwargs['to'] = self.remote_field.model
        else:
            kwargs['to'] = "%s.%s" % (
                self.remote_field.model._meta.app_label,
                self.remote_field.model._meta.object_name,
            )
        if getattr(self.remote_field, 'through', None) is not None:
            if isinstance(self.remote_field.through, str):
                kwargs['through'] = self.remote_field.through
            elif not self.remote_field.through._meta.auto_created:
                kwargs['through'] = "%s.%s" % (
                    self.remote_field.through._meta.app_label,
                    self.remote_field.through._meta.object_name,
                )
        # If swappable is True, then see if we're actually pointing to the target
        # of a swap.
        swappable_setting = self.swappable_setting
        if swappable_setting is not None:
            # If it's already a settings reference, error.
            if hasattr(kwargs['to'], "setting_name"):
                if kwargs['to'].setting_name != swappable_setting:
                    raise ValueError(
                        "Cannot deconstruct a ManyToManyField pointing to a "
                        "model that is swapped in place of more than one model "
                        "(%s and %s)" % (kwargs['to'].setting_name, swappable_setting)
                    )

            kwargs['to'] = SettingsReference(
                kwargs['to'],
                swappable_setting,
            )
        return name, path, args, kwargs
```
### 16 - django/db/models/fields/related.py:

Start line: 1623, End line: 1657

```python
class ManyToManyField(RelatedField):

    def set_attributes_from_rel(self):
        pass

    def value_from_object(self, obj):
        return [] if obj.pk is None else list(getattr(obj, self.attname).all())

    def save_form_data(self, instance, data):
        getattr(instance, self.attname).set(data)

    def formfield(self, *, using=None, **kwargs):
        defaults = {
            'form_class': forms.ModelMultipleChoiceField,
            'queryset': self.remote_field.model._default_manager.using(using),
            **kwargs,
        }
        # If initial is passed in, it's a list of related objects, but the
        # MultipleChoiceField takes a list of IDs.
        if defaults.get('initial') is not None:
            initial = defaults['initial']
            if callable(initial):
                initial = initial()
            defaults['initial'] = [i.pk for i in initial]
        return super().formfield(**defaults)

    def db_check(self, connection):
        return None

    def db_type(self, connection):
        # A ManyToManyField is not represented by a single column,
        # so return None.
        return None

    def db_parameters(self, connection):
        return {"type": None, "check": None}
```
### 18 - django/db/models/fields/related_descriptors.py:

Start line: 794, End line: 853

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
                core_filter_key = '%s__%s' % (self.query_field_name, rh_field.name)
                self.core_filters[core_filter_key] = getattr(instance, rh_field.attname)
                self.pk_field_names[lh_field.name] = rh_field.name

            self.related_val = self.source_field.get_foreign_related_value(instance)
            if None in self.related_val:
                raise ValueError('"%r" needs to have a value for field "%s" before '
                                 'this many-to-many relationship can be used.' %
                                 (instance, self.pk_field_names[self.source_field_name]))
            # Even if this relation is not to pk, we require still pk value.
            # The wish is that the instance has been already saved to DB,
            # although having a pk value isn't a guarantee of that.
            if instance.pk is None:
                raise ValueError("%r instance needs to have a primary key value before "
                                 "a many-to-many relationship can be used." %
                                 instance.__class__.__name__)

        def __call__(self, *, manager):
            manager = getattr(self.model, manager)
            manager_class = create_forward_many_to_many_manager(manager.__class__, rel, reverse)
            return manager_class(instance=self.instance)
        do_not_call_in_templates = True
    # ... other code
```
### 20 - django/db/models/fields/related_descriptors.py:

Start line: 855, End line: 869

```python
def create_forward_many_to_many_manager(superclass, rel, reverse):

    class ManyRelatedManager(superclass):

        def _build_remove_filters(self, removed_vals):
            filters = Q(**{self.source_field_name: self.related_val})
            # No need to add a subquery condition if removed_vals is a QuerySet without
            # filters.
            removed_vals_filters = (not isinstance(removed_vals, QuerySet) or
                                    removed_vals._has_filters())
            if removed_vals_filters:
                filters &= Q(**{'%s__in' % self.target_field_name: removed_vals})
            if self.symmetrical:
                symmetrical_filters = Q(**{self.target_field_name: self.related_val})
                if removed_vals_filters:
                    symmetrical_filters &= Q(
                        **{'%s__in' % self.source_field_name: removed_vals})
                filters |= symmetrical_filters
            return filters
    # ... other code
```
### 21 - django/db/models/fields/related_descriptors.py:

Start line: 663, End line: 717

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
### 22 - django/db/models/fields/related.py:

Start line: 759, End line: 822

```python
class ForeignKey(ForeignObject):
    """
    Provide a many-to-one relation by adding a column to the local model
    to hold the remote value.

    By default ForeignKey will target the pk of the remote model but this
    behavior can be changed by using the ``to_field`` argument.
    """

    # Field flags
    many_to_many = False
    many_to_one = True
    one_to_many = False
    one_to_one = False

    rel_class = ManyToOneRel

    empty_strings_allowed = False
    default_error_messages = {
        'invalid': _('%(model)s instance with %(field)s %(value)r does not exist.')
    }
    description = _("Foreign Key (type determined by related field)")

    def __init__(self, to, on_delete, related_name=None, related_query_name=None,
                 limit_choices_to=None, parent_link=False, to_field=None,
                 db_constraint=True, **kwargs):
        try:
            to._meta.model_name
        except AttributeError:
            assert isinstance(to, str), (
                "%s(%r) is invalid. First parameter to ForeignKey must be "
                "either a model, a model name, or the string %r" % (
                    self.__class__.__name__, to,
                    RECURSIVE_RELATIONSHIP_CONSTANT,
                )
            )
        else:
            # For backwards compatibility purposes, we need to *try* and set
            # the to_field during FK construction. It won't be guaranteed to
            # be correct until contribute_to_class is called. Refs #12190.
            to_field = to_field or (to._meta.pk and to._meta.pk.name)
        if not callable(on_delete):
            raise TypeError('on_delete must be callable.')

        kwargs['rel'] = self.rel_class(
            self, to, to_field,
            related_name=related_name,
            related_query_name=related_query_name,
            limit_choices_to=limit_choices_to,
            parent_link=parent_link,
            on_delete=on_delete,
        )
        kwargs.setdefault('db_index', True)

        super().__init__(to, on_delete, from_fields=['self'], to_fields=[to_field], **kwargs)

        self.db_constraint = db_constraint

    def check(self, **kwargs):
        return [
            *super().check(**kwargs),
            *self._check_on_delete(),
            *self._check_unique(),
        ]
```
### 23 - django/db/models/fields/related_descriptors.py:

Start line: 1071, End line: 1098

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
                connections[db].features.supports_ignore_conflicts and
                self.through._meta.auto_created is not False
            )
            # Don't send the signal when inserting duplicate data row
            # for symmetrical reverse entries.
            must_send_signals = (self.reverse or source_field_name == self.source_field_name) and (
                signals.m2m_changed.has_listeners(self.through)
            )
            # Fast addition through bulk insertion can only be performed
            # if no m2m_changed listeners are connected for self.through
            # as they require the added set of ids to be provided via
            # pk_set.
            return can_ignore_conflicts, must_send_signals, (can_ignore_conflicts and not must_send_signals)
    # ... other code
```
### 25 - django/db/models/fields/related_descriptors.py:

Start line: 1058, End line: 1069

```python
def create_forward_many_to_many_manager(superclass, rel, reverse):

    class ManyRelatedManager(superclass):

        def _get_missing_target_ids(self, source_field_name, target_field_name, db, target_ids):
            """
            Return the subset of ids of `objs` that aren't already assigned to
            this relationship.
            """
            vals = self.through._default_manager.using(db).values_list(
                target_field_name, flat=True
            ).filter(**{
                source_field_name: self.related_val[0],
                '%s__in' % target_field_name: target_ids,
            })
            return target_ids.difference(vals)
    # ... other code
```
### 26 - django/db/models/fields/related_descriptors.py:

Start line: 633, End line: 662

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
### 28 - django/db/models/fields/related_descriptors.py:

Start line: 599, End line: 631

```python
def create_reverse_many_to_one_manager(superclass, rel):

    class RelatedManager(superclass):

        def _remove_prefetched_objects(self):
            try:
                self.instance._prefetched_objects_cache.pop(self.field.remote_field.get_cache_name())
            except (AttributeError, KeyError):
                pass  # nothing to clear from cache

        def get_queryset(self):
            try:
                return self.instance._prefetched_objects_cache[self.field.remote_field.get_cache_name()]
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
            query = {'%s__in' % self.field.name: instances}
            queryset = queryset.filter(**query)

            # Since we just bypassed this class' get_queryset(), we must manage
            # the reverse relation manually.
            for rel_obj in queryset:
                instance = instances_dict[rel_obj_attr(rel_obj)]
                setattr(rel_obj, self.field.name, instance)
            cache_name = self.field.remote_field.get_cache_name()
            return queryset, rel_obj_attr, instance_attr, False, cache_name, False
    # ... other code
```
### 29 - django/db/models/fields/related_descriptors.py:

Start line: 719, End line: 745

```python
def create_reverse_many_to_one_manager(superclass, rel):

    class RelatedManager(superclass):

        def set(self, objs, *, bulk=True, clear=False):
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
### 30 - django/db/models/fields/related.py:

Start line: 1169, End line: 1200

```python
class ManyToManyField(RelatedField):

    def _check_ignored_options(self, **kwargs):
        warnings = []

        if self.has_null_arg:
            warnings.append(
                checks.Warning(
                    'null has no effect on ManyToManyField.',
                    obj=self,
                    id='fields.W340',
                )
            )

        if self._validators:
            warnings.append(
                checks.Warning(
                    'ManyToManyField does not support validators.',
                    obj=self,
                    id='fields.W341',
                )
            )
        if (self.remote_field.limit_choices_to and self.remote_field.through and
                not self.remote_field.through._meta.auto_created):
            warnings.append(
                checks.Warning(
                    'limit_choices_to has no effect on ManyToManyField '
                    'with a through model.',
                    obj=self,
                    id='fields.W343',
                )
            )

        return warnings
```
### 31 - django/db/models/fields/related_descriptors.py:

Start line: 1027, End line: 1056

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
                            'value is on database "%s"' %
                            (obj, self.instance._state.db, obj._state.db)
                        )
                    target_id = target_field.get_foreign_related_value(obj)[0]
                    if target_id is None:
                        raise ValueError(
                            'Cannot add "%r": the value for field "%s" is None' %
                            (obj, target_field_name)
                        )
                    target_ids.add(target_id)
                elif isinstance(obj, Model):
                    raise TypeError(
                        "'%s' instance expected, got %r" %
                        (self.model._meta.object_name, obj)
                    )
                else:
                    target_ids.add(obj)
            return target_ids
    # ... other code
```
### 34 - django/db/models/fields/related_descriptors.py:

Start line: 111, End line: 145

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
### 36 - django/db/models/fields/related.py:

Start line: 444, End line: 485

```python
class ForeignObject(RelatedField):
    """
    Abstraction of the ForeignKey relation to support multi-column relations.
    """

    # Field flags
    many_to_many = False
    many_to_one = True
    one_to_many = False
    one_to_one = False

    requires_unique_target = True
    related_accessor_class = ReverseManyToOneDescriptor
    forward_related_accessor_class = ForwardManyToOneDescriptor
    rel_class = ForeignObjectRel

    def __init__(self, to, on_delete, from_fields, to_fields, rel=None, related_name=None,
                 related_query_name=None, limit_choices_to=None, parent_link=False,
                 swappable=True, **kwargs):

        if rel is None:
            rel = self.rel_class(
                self, to,
                related_name=related_name,
                related_query_name=related_query_name,
                limit_choices_to=limit_choices_to,
                parent_link=parent_link,
                on_delete=on_delete,
            )

        super().__init__(rel=rel, **kwargs)

        self.from_fields = from_fields
        self.to_fields = to_fields
        self.swappable = swappable

    def check(self, **kwargs):
        return [
            *super().check(**kwargs),
            *self._check_to_fields_exist(),
            *self._check_unique_target(),
        ]
```
### 37 - django/db/models/fields/related_descriptors.py:

Start line: 871, End line: 891

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
### 38 - django/db/models/fields/related_descriptors.py:

Start line: 542, End line: 597

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

        def __call__(self, *, manager):
            manager = getattr(self.model, manager)
            manager_class = create_reverse_many_to_one_manager(manager.__class__, rel)
            return manager_class(self.instance)
        do_not_call_in_templates = True

        def _apply_rel_filters(self, queryset):
            """
            Filter the queryset for the instance this manager is bound to.
            """
            db = self._db or router.db_for_read(self.model, instance=self.instance)
            empty_strings_as_null = connections[db].features.interprets_empty_strings_as_nulls
            queryset._add_hints(instance=self.instance)
            if self._db:
                queryset = queryset.using(self._db)
            queryset = queryset.filter(**self.core_filters)
            for field in self.field.foreign_related_fields:
                val = getattr(self.instance, field.attname)
                if val is None or (val == '' and empty_strings_as_null):
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
                    rel_obj_id = tuple([
                        getattr(self.instance, target_field.attname)
                        for target_field in self.field.get_path_info()[-1].target_fields
                    ])
                else:
                    rel_obj_id = getattr(self.instance, target_field.attname)
                queryset._known_related_objects = {self.field: {rel_obj_id: self.instance}}
            return queryset
    # ... other code
```
### 40 - django/db/models/fields/related.py:

Start line: 1501, End line: 1518

```python
class ManyToManyField(RelatedField):

    def get_path_info(self, filtered_relation=None):
        return self._get_path_info(direct=True, filtered_relation=filtered_relation)

    def get_reverse_path_info(self, filtered_relation=None):
        return self._get_path_info(direct=False, filtered_relation=filtered_relation)

    def _get_m2m_db_table(self, opts):
        """
        Function that can be curried to provide the m2m table name for this
        relation.
        """
        if self.remote_field.through is not None:
            return self.remote_field.through._meta.db_table
        elif self.db_table:
            return self.db_table
        else:
            m2m_table_name = '%s_%s' % (utils.strip_quotes(opts.db_table), self.name)
            return utils.truncate_name(m2m_table_name, connection.ops.max_name_length())
```
### 41 - django/db/models/fields/related.py:

Start line: 994, End line: 1041

```python
class OneToOneField(ForeignKey):
    """
    A OneToOneField is essentially the same as a ForeignKey, with the exception
    that it always carries a "unique" constraint with it and the reverse
    relation always returns the object pointed to (since there will only ever
    be one), rather than returning a list.
    """

    # Field flags
    many_to_many = False
    many_to_one = False
    one_to_many = False
    one_to_one = True

    related_accessor_class = ReverseOneToOneDescriptor
    forward_related_accessor_class = ForwardOneToOneDescriptor
    rel_class = OneToOneRel

    description = _("One-to-one relationship")

    def __init__(self, to, on_delete, to_field=None, **kwargs):
        kwargs['unique'] = True
        super().__init__(to, on_delete, to_field=to_field, **kwargs)

    def deconstruct(self):
        name, path, args, kwargs = super().deconstruct()
        if "unique" in kwargs:
            del kwargs['unique']
        return name, path, args, kwargs

    def formfield(self, **kwargs):
        if self.remote_field.parent_link:
            return None
        return super().formfield(**kwargs)

    def save_form_data(self, instance, data):
        if isinstance(data, self.remote_field.model):
            setattr(instance, self.name, data)
        else:
            setattr(instance, self.attname, data)
            # Remote field object must be cleared otherwise Model.save()
            # will reassign attname using the related object pk.
            if data is None:
                setattr(instance, self.name, data)

    def _check_unique(self, **kwargs):
        # Override ForeignKey since check isn't applicable here.
        return []
```
### 42 - django/db/models/fields/related.py:

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
### 45 - django/db/models/fields/related.py:

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
### 46 - django/db/models/fields/related.py:

Start line: 616, End line: 638

```python
class ForeignObject(RelatedField):

    @property
    def related_fields(self):
        if not hasattr(self, '_related_fields'):
            self._related_fields = self.resolve_related_fields()
        return self._related_fields

    @property
    def reverse_related_fields(self):
        return [(rhs_field, lhs_field) for lhs_field, rhs_field in self.related_fields]

    @property
    def local_related_fields(self):
        return tuple(lhs_field for lhs_field, rhs_field in self.related_fields)

    @property
    def foreign_related_fields(self):
        return tuple(rhs_field for lhs_field, rhs_field in self.related_fields if rhs_field)

    def get_local_related_value(self, instance):
        return self.get_instance_value_for_fields(instance, self.local_related_fields)

    def get_foreign_related_value(self, instance):
        return self.get_instance_value_for_fields(instance, self.foreign_related_fields)
```
### 47 - django/db/models/fields/related.py:

Start line: 1538, End line: 1566

```python
class ManyToManyField(RelatedField):

    def _get_m2m_reverse_attr(self, related, attr):
        """
        Function that can be curried to provide the related accessor or DB
        column name for the m2m table.
        """
        cache_attr = '_m2m_reverse_%s_cache' % attr
        if hasattr(self, cache_attr):
            return getattr(self, cache_attr)
        found = False
        if self.remote_field.through_fields is not None:
            link_field_name = self.remote_field.through_fields[1]
        else:
            link_field_name = None
        for f in self.remote_field.through._meta.fields:
            if f.is_relation and f.remote_field.model == related.model:
                if link_field_name is None and related.related_model == related.model:
                    # If this is an m2m-intermediate to self,
                    # the first foreign key you find will be
                    # the source column. Keep searching for
                    # the second foreign key.
                    if found:
                        setattr(self, cache_attr, getattr(f, attr))
                        break
                    else:
                        found = True
                elif link_field_name is None or link_field_name == f.name:
                    setattr(self, cache_attr, getattr(f, attr))
                    break
        return getattr(self, cache_attr)
```
### 48 - django/db/models/fields/related.py:

Start line: 847, End line: 873

```python
class ForeignKey(ForeignObject):

    def _check_unique(self, **kwargs):
        return [
            checks.Warning(
                'Setting unique=True on a ForeignKey has the same effect as using a OneToOneField.',
                hint='ForeignKey(unique=True) is usually better served by a OneToOneField.',
                obj=self,
                id='fields.W342',
            )
        ] if self.unique else []

    def deconstruct(self):
        name, path, args, kwargs = super().deconstruct()
        del kwargs['to_fields']
        del kwargs['from_fields']
        # Handle the simpler arguments
        if self.db_index:
            del kwargs['db_index']
        else:
            kwargs['db_index'] = False
        if self.db_constraint is not True:
            kwargs['db_constraint'] = self.db_constraint
        # Rel needs more work.
        to_meta = getattr(self.remote_field.model, "_meta", None)
        if self.remote_field.field_name and (
                not to_meta or (to_meta.pk and self.remote_field.field_name != to_meta.pk.name)):
            kwargs['to_field'] = self.remote_field.field_name
        return name, path, args, kwargs
```
### 49 - django/db/models/fields/related_descriptors.py:

Start line: 300, End line: 314

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
### 51 - django/db/models/fields/related.py:

Start line: 509, End line: 563

```python
class ForeignObject(RelatedField):

    def _check_unique_target(self):
        rel_is_string = isinstance(self.remote_field.model, str)
        if rel_is_string or not self.requires_unique_target:
            return []

        try:
            self.foreign_related_fields
        except exceptions.FieldDoesNotExist:
            return []

        if not self.foreign_related_fields:
            return []

        unique_foreign_fields = {
            frozenset([f.name])
            for f in self.remote_field.model._meta.get_fields()
            if getattr(f, 'unique', False)
        }
        unique_foreign_fields.update({
            frozenset(ut)
            for ut in self.remote_field.model._meta.unique_together
        })
        foreign_fields = {f.name for f in self.foreign_related_fields}
        has_unique_constraint = any(u <= foreign_fields for u in unique_foreign_fields)

        if not has_unique_constraint and len(self.foreign_related_fields) > 1:
            field_combination = ', '.join(
                "'%s'" % rel_field.name for rel_field in self.foreign_related_fields
            )
            model_name = self.remote_field.model.__name__
            return [
                checks.Error(
                    "No subset of the fields %s on model '%s' is unique."
                    % (field_combination, model_name),
                    hint=(
                        "Add unique=True on any of those fields or add at "
                        "least a subset of them to a unique_together constraint."
                    ),
                    obj=self,
                    id='fields.E310',
                )
            ]
        elif not has_unique_constraint:
            field_name = self.foreign_related_fields[0].name
            model_name = self.remote_field.model.__name__
            return [
                checks.Error(
                    "'%s.%s' must set unique=True because it is referenced by "
                    "a foreign key." % (model_name, field_name),
                    obj=self,
                    id='fields.E311',
                )
            ]
        else:
            return []
```
### 52 - django/db/models/fields/related.py:

Start line: 600, End line: 614

```python
class ForeignObject(RelatedField):

    def resolve_related_fields(self):
        if not self.from_fields or len(self.from_fields) != len(self.to_fields):
            raise ValueError('Foreign Object from and to fields must be the same non-zero length')
        if isinstance(self.remote_field.model, str):
            raise ValueError('Related model %r cannot be resolved' % self.remote_field.model)
        related_fields = []
        for index in range(len(self.from_fields)):
            from_field_name = self.from_fields[index]
            to_field_name = self.to_fields[index]
            from_field = (self if from_field_name == 'self'
                          else self.opts.get_field(from_field_name))
            to_field = (self.remote_field.model._meta.pk if to_field_name is None
                        else self.remote_field.model._meta.get_field(to_field_name))
            related_fields.append((from_field, to_field))
        return related_fields
```
### 54 - django/db/models/fields/related.py:

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
### 55 - django/db/models/fields/related.py:

Start line: 284, End line: 318

```python
class RelatedField(FieldCacheMixin, Field):

    def db_type(self, connection):
        # By default related field will not have a column as it relates to
        # columns from another table.
        return None

    def contribute_to_class(self, cls, name, private_only=False, **kwargs):

        super().contribute_to_class(cls, name, private_only=private_only, **kwargs)

        self.opts = cls._meta

        if not cls._meta.abstract:
            if self.remote_field.related_name:
                related_name = self.remote_field.related_name
            else:
                related_name = self.opts.default_related_name
            if related_name:
                related_name = related_name % {
                    'class': cls.__name__.lower(),
                    'model_name': cls._meta.model_name.lower(),
                    'app_label': cls._meta.app_label.lower()
                }
                self.remote_field.related_name = related_name

            if self.remote_field.related_query_name:
                related_query_name = self.remote_field.related_query_name % {
                    'class': cls.__name__.lower(),
                    'app_label': cls._meta.app_label.lower(),
                }
                self.remote_field.related_query_name = related_query_name

            def resolve_related_class(model, related, field):
                field.remote_field.model = related
                field.do_related_class(related, model)
            lazy_related_operation(resolve_related_class, cls, self.remote_field.model, field=self)
```
### 57 - django/db/models/fields/related.py:

Start line: 1520, End line: 1536

```python
class ManyToManyField(RelatedField):

    def _get_m2m_attr(self, related, attr):
        """
        Function that can be curried to provide the source accessor or DB
        column name for the m2m table.
        """
        cache_attr = '_m2m_%s_cache' % attr
        if hasattr(self, cache_attr):
            return getattr(self, cache_attr)
        if self.remote_field.through_fields is not None:
            link_field_name = self.remote_field.through_fields[0]
        else:
            link_field_name = None
        for f in self.remote_field.through._meta.fields:
            if (f.is_relation and f.remote_field.model == related.related_model and
                    (link_field_name is None or link_field_name == f.name)):
                setattr(self, cache_attr, getattr(f, attr))
                return getattr(self, cache_attr)
```
### 58 - django/db/models/fields/related.py:

Start line: 487, End line: 507

```python
class ForeignObject(RelatedField):

    def _check_to_fields_exist(self):
        # Skip nonexistent models.
        if isinstance(self.remote_field.model, str):
            return []

        errors = []
        for to_field in self.to_fields:
            if to_field:
                try:
                    self.remote_field.model._meta.get_field(to_field)
                except exceptions.FieldDoesNotExist:
                    errors.append(
                        checks.Error(
                            "The to_field '%s' doesn't exist on the related "
                            "model '%s'."
                            % (to_field, self.remote_field.model._meta.label),
                            obj=self,
                            id='fields.E312',
                        )
                    )
        return errors
```
### 60 - django/db/models/fields/related.py:

Start line: 1475, End line: 1499

```python
class ManyToManyField(RelatedField):

    def _get_path_info(self, direct=False, filtered_relation=None):
        """Called by both direct and indirect m2m traversal."""
        int_model = self.remote_field.through
        linkfield1 = int_model._meta.get_field(self.m2m_field_name())
        linkfield2 = int_model._meta.get_field(self.m2m_reverse_field_name())
        if direct:
            join1infos = linkfield1.get_reverse_path_info()
            join2infos = linkfield2.get_path_info(filtered_relation)
        else:
            join1infos = linkfield2.get_reverse_path_info()
            join2infos = linkfield1.get_path_info(filtered_relation)

        # Get join infos between the last model of join 1 and the first model
        # of join 2. Assume the only reason these may differ is due to model
        # inheritance.
        join1_final = join1infos[-1].to_opts
        join2_initial = join2infos[0].from_opts
        if join1_final is join2_initial:
            intermediate_infos = []
        elif issubclass(join1_final.model, join2_initial.model):
            intermediate_infos = join1_final.get_path_to_parent(join2_initial.model)
        else:
            intermediate_infos = join2_initial.get_path_from_parent(join1_final.model)

        return [*join1infos, *intermediate_infos, *join2infos]
```
### 61 - django/db/models/fields/related_descriptors.py:

Start line: 485, End line: 539

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
    def related_manager_cls(self):
        related_model = self.rel.related_model

        return create_reverse_many_to_one_manager(
            related_model._default_manager.__class__,
            self.rel,
        )

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

        return self.related_manager_cls(instance)

    def _get_set_deprecation_msg_params(self):
        return (
            'reverse side of a related set',
            self.rel.get_accessor_name(),
        )

    def __set__(self, instance, value):
        raise TypeError(
            'Direct assignment to the %s is prohibited. Use %s.set() instead.'
            % self._get_set_deprecation_msg_params(),
        )
```
### 62 - django/db/models/fields/related.py:

Start line: 824, End line: 845

```python
class ForeignKey(ForeignObject):

    def _check_on_delete(self):
        on_delete = getattr(self.remote_field, 'on_delete', None)
        if on_delete == SET_NULL and not self.null:
            return [
                checks.Error(
                    'Field specifies on_delete=SET_NULL, but cannot be null.',
                    hint='Set null=True argument on the field, or change the on_delete rule.',
                    obj=self,
                    id='fields.E320',
                )
            ]
        elif on_delete == SET_DEFAULT and not self.has_default():
            return [
                checks.Error(
                    'Field specifies on_delete=SET_DEFAULT, but has no default value.',
                    hint='Set a default value, or change the on_delete rule.',
                    obj=self,
                    id='fields.E321',
                )
            ]
        else:
            return []
```
### 63 - django/db/models/fields/related.py:

Start line: 952, End line: 962

```python
class ForeignKey(ForeignObject):

    def formfield(self, *, using=None, **kwargs):
        if isinstance(self.remote_field.model, str):
            raise ValueError("Cannot create form field for %r yet, because "
                             "its related model %r has not been loaded yet" %
                             (self.name, self.remote_field.model))
        return super().formfield(**{
            'form_class': forms.ModelChoiceField,
            'queryset': self.remote_field.model._default_manager.using(using),
            'to_field_name': self.remote_field.field_name,
            **kwargs,
        })
```
### 65 - django/db/models/fields/related_descriptors.py:

Start line: 335, End line: 354

```python
class ReverseOneToOneDescriptor:

    @cached_property
    def RelatedObjectDoesNotExist(self):
        # The exception isn't created at initialization time for the sake of
        # consistency with `ForwardManyToOneDescriptor`.
        return type(
            'RelatedObjectDoesNotExist',
            (self.related.related_model.DoesNotExist, AttributeError), {
                '__module__': self.related.model.__module__,
                '__qualname__': '%s.%s.RelatedObjectDoesNotExist' % (
                    self.related.model.__qualname__,
                    self.related.name,
                )
            },
        )

    def is_cached(self, instance):
        return self.related.is_cached(instance)

    def get_queryset(self, **hints):
        return self.related.related_model._base_manager.db_manager(hints=hints).all()
```
### 66 - django/db/models/fields/related_descriptors.py:

Start line: 73, End line: 109

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
            'RelatedObjectDoesNotExist',
            (self.field.remote_field.model.DoesNotExist, AttributeError), {
                '__module__': self.field.model.__module__,
                '__qualname__': '%s.%s.RelatedObjectDoesNotExist' % (
                    self.field.model.__qualname__,
                    self.field.name,
                ),
            }
        )

    def is_cached(self, instance):
        return self.field.is_cached(instance)

    def get_queryset(self, **hints):
        return self.field.remote_field.model._base_manager.db_manager(hints=hints).all()
```
### 69 - django/db/models/fields/related.py:

Start line: 171, End line: 188

```python
class RelatedField(FieldCacheMixin, Field):

    def _check_referencing_to_swapped_model(self):
        if (self.remote_field.model not in self.opts.apps.get_models() and
                not isinstance(self.remote_field.model, str) and
                self.remote_field.model._meta.swapped):
            model = "%s.%s" % (
                self.remote_field.model._meta.app_label,
                self.remote_field.model._meta.object_name
            )
            return [
                checks.Error(
                    "Field defines a relation with the model '%s', which has "
                    "been swapped out." % model,
                    hint="Update the relation to point at 'settings.%s'." % self.remote_field.model._meta.swappable,
                    obj=self,
                    id='fields.E301',
                )
            ]
        return []
```
### 70 - django/db/models/fields/related.py:

Start line: 108, End line: 125

```python
class RelatedField(FieldCacheMixin, Field):

    def _check_related_name_is_valid(self):
        import keyword
        related_name = self.remote_field.related_name
        if related_name is None:
            return []
        is_valid_id = not keyword.iskeyword(related_name) and related_name.isidentifier()
        if not (is_valid_id or related_name.endswith('+')):
            return [
                checks.Error(
                    "The name '%s' is invalid related_name for field %s.%s" %
                    (self.remote_field.related_name, self.model._meta.object_name,
                     self.name),
                    hint="Related name must be a valid Python identifier or end with a '+'",
                    obj=self,
                    id='fields.E306',
                )
            ]
        return []
```
### 72 - django/db/models/fields/related.py:

Start line: 964, End line: 991

```python
class ForeignKey(ForeignObject):

    def db_check(self, connection):
        return []

    def db_type(self, connection):
        return self.target_field.rel_db_type(connection=connection)

    def db_parameters(self, connection):
        return {"type": self.db_type(connection), "check": self.db_check(connection)}

    def convert_empty_strings(self, value, expression, connection):
        if (not value) and isinstance(value, str):
            return None
        return value

    def get_db_converters(self, connection):
        converters = super().get_db_converters(connection)
        if connection.features.interprets_empty_strings_as_nulls:
            converters += [self.convert_empty_strings]
        return converters

    def get_col(self, alias, output_field=None):
        if output_field is None:
            output_field = self.target_field
            while isinstance(output_field, ForeignKey):
                output_field = output_field.target_field
                if output_field is self:
                    raise ValueError('Cannot resolve output_field.')
        return super().get_col(alias, output_field)
```
### 73 - django/db/models/fields/related.py:

Start line: 565, End line: 598

```python
class ForeignObject(RelatedField):

    def deconstruct(self):
        name, path, args, kwargs = super().deconstruct()
        kwargs['on_delete'] = self.remote_field.on_delete
        kwargs['from_fields'] = self.from_fields
        kwargs['to_fields'] = self.to_fields

        if self.remote_field.parent_link:
            kwargs['parent_link'] = self.remote_field.parent_link
        # Work out string form of "to"
        if isinstance(self.remote_field.model, str):
            kwargs['to'] = self.remote_field.model
        else:
            kwargs['to'] = "%s.%s" % (
                self.remote_field.model._meta.app_label,
                self.remote_field.model._meta.object_name,
            )
        # If swappable is True, then see if we're actually pointing to the target
        # of a swap.
        swappable_setting = self.swappable_setting
        if swappable_setting is not None:
            # If it's already a settings reference, error
            if hasattr(kwargs['to'], "setting_name"):
                if kwargs['to'].setting_name != swappable_setting:
                    raise ValueError(
                        "Cannot deconstruct a ForeignKey pointing to a model "
                        "that is swapped in place of more than one model (%s and %s)"
                        % (kwargs['to'].setting_name, swappable_setting)
                    )
            # Set it
            kwargs['to'] = SettingsReference(
                kwargs['to'],
                swappable_setting,
            )
        return name, path, args, kwargs
```
### 76 - django/db/models/fields/related_descriptors.py:

Start line: 194, End line: 266

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
        if value is not None and not isinstance(value, self.field.remote_field.model._meta.concrete_model):
            raise ValueError(
                'Cannot assign "%r": "%s.%s" must be a "%s" instance.' % (
                    value,
                    instance._meta.object_name,
                    self.field.name,
                    self.field.remote_field.model._meta.object_name,
                )
            )
        elif value is not None:
            if instance._state.db is None:
                instance._state.db = router.db_for_write(instance.__class__, instance=value)
            if value._state.db is None:
                value._state.db = router.db_for_write(value.__class__, instance=instance)
            if not router.allow_relation(value, instance):
                raise ValueError('Cannot assign "%r": the current database router prevents this relation.' % value)

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
### 79 - django/db/models/fields/related.py:

Start line: 156, End line: 169

```python
class RelatedField(FieldCacheMixin, Field):

    def _check_relation_model_exists(self):
        rel_is_missing = self.remote_field.model not in self.opts.apps.get_models()
        rel_is_string = isinstance(self.remote_field.model, str)
        model_name = self.remote_field.model if rel_is_string else self.remote_field.model._meta.object_name
        if rel_is_missing and (rel_is_string or not self.remote_field.model._meta.swapped):
            return [
                checks.Error(
                    "Field defines a relation with model '%s', which is either "
                    "not installed, or is abstract." % model_name,
                    obj=self,
                    id='fields.E300',
                )
            ]
        return []
```
### 80 - django/db/models/fields/related_descriptors.py:

Start line: 421, End line: 482

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
                'Cannot assign "%r": "%s.%s" must be a "%s" instance.' % (
                    value,
                    instance._meta.object_name,
                    self.related.get_accessor_name(),
                    self.related.related_model._meta.object_name,
                )
            )
        else:
            if instance._state.db is None:
                instance._state.db = router.db_for_write(instance.__class__, instance=value)
            if value._state.db is None:
                value._state.db = router.db_for_write(value.__class__, instance=instance)
            if not router.allow_relation(value, instance):
                raise ValueError('Cannot assign "%r": the current database router prevents this relation.' % value)

            related_pk = tuple(getattr(instance, field.attname) for field in self.related.field.foreign_related_fields)
            # Set the value of the related field to the value of the related object's related field
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
### 81 - django/db/models/fields/related.py:

Start line: 190, End line: 254

```python
class RelatedField(FieldCacheMixin, Field):

    def _check_clashes(self):
        """Check accessor and reverse query name clashes."""
        from django.db.models.base import ModelBase

        errors = []
        opts = self.model._meta

        # `f.remote_field.model` may be a string instead of a model. Skip if model name is
        # not resolved.
        if not isinstance(self.remote_field.model, ModelBase):
            return []

        # Consider that we are checking field `Model.foreign` and the models
        # are:
        #
        #     class Target(models.Model):
        #         model = models.IntegerField()
        #         model_set = models.IntegerField()
        #
        #     class Model(models.Model):
        #         foreign = models.ForeignKey(Target)
        #         m2m = models.ManyToManyField(Target)

        # rel_opts.object_name == "Target"
        rel_opts = self.remote_field.model._meta
        # If the field doesn't install a backward relation on the target model
        # (so `is_hidden` returns True), then there are no clashes to check
        # and we can skip these fields.
        rel_is_hidden = self.remote_field.is_hidden()
        rel_name = self.remote_field.get_accessor_name()  # i. e. "model_set"
        rel_query_name = self.related_query_name()  # i. e. "model"
        field_name = "%s.%s" % (opts.object_name, self.name)  # i. e. "Model.field"

        # Check clashes between accessor or reverse query name of `field`
        # and any other field name -- i.e. accessor for Model.foreign is
        # model_set and it clashes with Target.model_set.
        potential_clashes = rel_opts.fields + rel_opts.many_to_many
        for clash_field in potential_clashes:
            clash_name = "%s.%s" % (rel_opts.object_name, clash_field.name)  # i.e. "Target.model_set"
            if not rel_is_hidden and clash_field.name == rel_name:
                errors.append(
                    checks.Error(
                        "Reverse accessor for '%s' clashes with field name '%s'." % (field_name, clash_name),
                        hint=("Rename field '%s', or add/change a related_name "
                              "argument to the definition for field '%s'.") % (clash_name, field_name),
                        obj=self,
                        id='fields.E302',
                    )
                )

            if clash_field.name == rel_query_name:
                errors.append(
                    checks.Error(
                        "Reverse query name for '%s' clashes with field name '%s'." % (field_name, clash_name),
                        hint=("Rename field '%s', or add/change a related_name "
                              "argument to the definition for field '%s'.") % (clash_name, field_name),
                        obj=self,
                        id='fields.E303',
                    )
                )

        # Check clashes between accessors/reverse query names of `field` and
        # any other field accessor -- i. e. Model.foreign accessor clashes with
        # Model.m2m accessor.
        potential_clashes = (r for r in rel_opts.related_objects if r.field is not self)
        # ... other code
```
### 82 - django/db/models/fields/related_descriptors.py:

Start line: 356, End line: 372

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
### 84 - django/db/models/fields/related_descriptors.py:

Start line: 748, End line: 791

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

    def _get_set_deprecation_msg_params(self):
        return (
            '%s side of a many-to-many set' % ('reverse' if self.reverse else 'forward'),
            self.rel.get_accessor_name() if self.reverse else self.field.name,
        )
```
### 95 - django/db/models/fields/related.py:

Start line: 658, End line: 682

```python
class ForeignObject(RelatedField):

    def get_attname_column(self):
        attname, column = super().get_attname_column()
        return attname, None

    def get_joining_columns(self, reverse_join=False):
        source = self.reverse_related_fields if reverse_join else self.related_fields
        return tuple((lhs_field.column, rhs_field.column) for lhs_field, rhs_field in source)

    def get_reverse_joining_columns(self):
        return self.get_joining_columns(reverse_join=True)

    def get_extra_descriptor_filter(self, instance):
        """
        Return an extra filter condition for related object fetching when
        user does 'instance.fieldname', that is the extra filter is used in
        the descriptor of the field.

        The filter should be either a dict usable in .filter(**kwargs) call or
        a Q-object. The condition will be ANDed together with the relation's
        joining columns.

        A parallel method is get_extra_restriction() which is used in
        JOIN and subquery conditions.
        """
        return {}
```
### 99 - django/db/models/fields/related_descriptors.py:

Start line: 147, End line: 192

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
            ancestor_link = instance._meta.get_ancestor_link(self.field.model) if has_value else None
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
### 106 - django/db/models/fields/related.py:

Start line: 1, End line: 34

```python
import functools
import inspect
from functools import partial

from django import forms
from django.apps import apps
from django.conf import SettingsReference
from django.core import checks, exceptions
from django.db import connection, router
from django.db.backends import utils
from django.db.models import Q
from django.db.models.constants import LOOKUP_SEP
from django.db.models.deletion import CASCADE, SET_DEFAULT, SET_NULL
from django.db.models.query_utils import PathInfo
from django.db.models.utils import make_model_tuple
from django.utils.functional import cached_property
from django.utils.translation import gettext_lazy as _

from . import Field
from .mixins import FieldCacheMixin
from .related_descriptors import (
    ForwardManyToOneDescriptor, ForwardOneToOneDescriptor,
    ManyToManyDescriptor, ReverseManyToOneDescriptor,
    ReverseOneToOneDescriptor,
)
from .related_lookups import (
    RelatedExact, RelatedGreaterThan, RelatedGreaterThanOrEqual, RelatedIn,
    RelatedIsNull, RelatedLessThan, RelatedLessThanOrEqual,
)
from .reverse_related import (
    ForeignObjectRel, ManyToManyRel, ManyToOneRel, OneToOneRel,
)

RECURSIVE_RELATIONSHIP_CONSTANT = 'self'
```
### 108 - django/db/models/fields/related.py:

Start line: 684, End line: 696

```python
class ForeignObject(RelatedField):

    def get_extra_restriction(self, where_class, alias, related_alias):
        """
        Return a pair condition used for joining and subquery pushdown. The
        condition is something that responds to as_sql(compiler, connection)
        method.

        Note that currently referring both the 'alias' and 'related_alias'
        will not work in some conditions, like subquery pushdown.

        A parallel method is get_extra_descriptor_filter() which is used in
        instance.fieldname related object fetching.
        """
        return None
```
### 117 - django/db/models/fields/related.py:

Start line: 83, End line: 106

```python
class RelatedField(FieldCacheMixin, Field):
    """Base class that all relational fields inherit from."""

    # Field flags
    one_to_many = False
    one_to_one = False
    many_to_many = False
    many_to_one = False

    @cached_property
    def related_model(self):
        # Can't cache this property until all the models are loaded.
        apps.check_models_ready()
        return self.remote_field.model

    def check(self, **kwargs):
        return [
            *super().check(**kwargs),
            *self._check_related_name_is_valid(),
            *self._check_related_query_name_is_valid(),
            *self._check_relation_model_exists(),
            *self._check_referencing_to_swapped_model(),
            *self._check_clashes(),
        ]
```
### 120 - django/db/models/fields/related.py:

Start line: 918, End line: 950

```python
class ForeignKey(ForeignObject):

    def get_attname(self):
        return '%s_id' % self.name

    def get_attname_column(self):
        attname = self.get_attname()
        column = self.db_column or attname
        return attname, column

    def get_default(self):
        """Return the to_field if the default value is an object."""
        field_default = super().get_default()
        if isinstance(field_default, self.remote_field.model):
            return getattr(field_default, self.target_field.attname)
        return field_default

    def get_db_prep_save(self, value, connection):
        if value is None or (value == '' and
                             (not self.target_field.empty_strings_allowed or
                              connection.features.interprets_empty_strings_as_nulls)):
            return None
        else:
            return self.target_field.get_db_prep_save(value, connection=connection)

    def get_db_prep_value(self, value, connection, prepared=False):
        return self.target_field.get_db_prep_value(value, connection, prepared)

    def get_prep_value(self, value):
        return self.target_field.get_prep_value(value)

    def contribute_to_related_class(self, cls, related):
        super().contribute_to_related_class(cls, related)
        if self.remote_field.field_name is None:
            self.remote_field.field_name = cls._meta.pk.name
```
### 121 - django/db/models/fields/related.py:

Start line: 896, End line: 916

```python
class ForeignKey(ForeignObject):

    def validate(self, value, model_instance):
        if self.remote_field.parent_link:
            return
        super().validate(value, model_instance)
        if value is None:
            return

        using = router.db_for_read(self.remote_field.model, instance=model_instance)
        qs = self.remote_field.model._default_manager.using(using).filter(
            **{self.remote_field.field_name: value}
        )
        qs = qs.complex_filter(self.get_limit_choices_to())
        if not qs.exists():
            raise exceptions.ValidationError(
                self.error_messages['invalid'],
                code='invalid',
                params={
                    'model': self.remote_field.model._meta.verbose_name, 'pk': value,
                    'field': self.remote_field.field_name, 'value': value,
                },  # 'pk' is included for backwards compatibility
            )
```
### 149 - django/db/models/fields/related.py:

Start line: 421, End line: 441

```python
class RelatedField(FieldCacheMixin, Field):

    def related_query_name(self):
        """
        Define the name that can be used to identify this related object in a
        table-spanning query.
        """
        return self.remote_field.related_query_name or self.remote_field.related_name or self.opts.model_name

    @property
    def target_field(self):
        """
        When filtering against this relation, return the field on the remote
        model against which the filtering should happen.
        """
        target_fields = self.get_path_info()[-1].target_fields
        if len(target_fields) > 1:
            raise exceptions.FieldError(
                "The relation has multiple target fields, but only single target field was asked for")
        return target_fields[0]

    def get_cache_name(self):
        return self.name
```
### 189 - django/db/models/fields/related_descriptors.py:

Start line: 1, End line: 70

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
from django.db import connections, router, transaction
from django.db.models import Q, signals
from django.db.models.query import QuerySet
from django.utils.functional import cached_property
```
