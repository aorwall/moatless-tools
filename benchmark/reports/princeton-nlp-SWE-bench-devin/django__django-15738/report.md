# django__django-15738

| **django/django** | `6f73eb9d90cfec684529aab48d517e3d6449ba8c` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 33581 |
| **Any found context length** | 33581 |
| **Avg pos** | 86.0 |
| **Min pos** | 86 |
| **Max pos** | 86 |
| **Top file pos** | 6 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/db/migrations/autodetector.py b/django/db/migrations/autodetector.py
--- a/django/db/migrations/autodetector.py
+++ b/django/db/migrations/autodetector.py
@@ -1022,8 +1022,9 @@ def generate_added_fields(self):
 
     def _generate_added_field(self, app_label, model_name, field_name):
         field = self.to_state.models[app_label, model_name].get_field(field_name)
-        # Fields that are foreignkeys/m2ms depend on stuff
-        dependencies = []
+        # Adding a field always depends at least on its removal.
+        dependencies = [(app_label, model_name, field_name, False)]
+        # Fields that are foreignkeys/m2ms depend on stuff.
         if field.remote_field and field.remote_field.model:
             dependencies.extend(
                 self._get_dependencies_for_foreign_key(

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/db/migrations/autodetector.py | 1025 | 1026 | 86 | 6 | 33581


## Problem Statement

```
Models migration with change field foreign to many and deleting unique together.
Description
	 
		(last modified by Simon Charette)
	 
I have models like
class Authors(models.Model):
	project_data_set = models.ForeignKey(
		ProjectDataSet,
		on_delete=models.PROTECT
	)
	state = models.IntegerField()
	start_date = models.DateField()
	class Meta:
		 unique_together = (('project_data_set', 'state', 'start_date'),)
and
class DataSet(models.Model):
	name = models.TextField(max_length=50)
class Project(models.Model):
	data_sets = models.ManyToManyField(
		DataSet,
		through='ProjectDataSet',
	)
	name = models.TextField(max_length=50)
class ProjectDataSet(models.Model):
	"""
	Cross table of data set and project
	"""
	data_set = models.ForeignKey(DataSet, on_delete=models.PROTECT)
	project = models.ForeignKey(Project, on_delete=models.PROTECT)
	class Meta:
		unique_together = (('data_set', 'project'),)
when i want to change field project_data_set in Authors model from foreign key field to many to many field I must delete a unique_together, cause it can't be on many to many field.
Then my model should be like:
class Authors(models.Model):
	project_data_set = models.ManyToManyField(
		ProjectDataSet,
	)
	state = models.IntegerField()
	start_date = models.DateField()
But when I want to do a migrations.
python3 manage.py makemigrations
python3 manage.py migrate
I have error:
ValueError: Found wrong number (0) of constraints for app_authors(project_data_set, state, start_date)
The database is on production, so I can't delete previous initial migrations, and this error isn't depending on database, cause I delete it and error is still the same.
My solve is to first delete unique_together, then do a makemigrations and then migrate. After that change the field from foreign key to many to many field, then do a makemigrations and then migrate.
But in this way I have 2 migrations instead of one.
I added attachment with this project, download it and then do makemigrations and then migrate to see this error.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 django/db/backends/base/schema.py | 520 | 539| 196 | 196 | 13797 | 
| 2 | 2 django/db/models/fields/related.py | 1449 | 1578| 984 | 1180 | 28409 | 
| 3 | 3 django/db/migrations/state.py | 437 | 458| 204 | 1384 | 36573 | 
| 4 | 3 django/db/migrations/state.py | 291 | 345| 476 | 1860 | 36573 | 
| 5 | 3 django/db/migrations/state.py | 460 | 486| 150 | 2010 | 36573 | 
| 6 | 3 django/db/models/fields/related.py | 1680 | 1730| 431 | 2441 | 36573 | 
| 7 | 4 django/contrib/auth/migrations/0001_initial.py | 1 | 206| 1007 | 3448 | 37580 | 
| 8 | 5 django/db/backends/sqlite3/schema.py | 484 | 534| 395 | 3843 | 42153 | 
| 9 | 5 django/db/backends/base/schema.py | 1241 | 1271| 318 | 4161 | 42153 | 
| 10 | 5 django/db/migrations/state.py | 181 | 238| 598 | 4759 | 42153 | 
| 11 | 5 django/db/models/fields/related.py | 1580 | 1678| 655 | 5414 | 42153 | 
| 12 | 5 django/db/migrations/state.py | 240 | 263| 247 | 5661 | 42153 | 
| 13 | 5 django/db/migrations/state.py | 265 | 289| 238 | 5899 | 42153 | 
| 14 | **6 django/db/migrations/autodetector.py** | 809 | 902| 694 | 6593 | 55516 | 
| 15 | 7 django/db/migrations/operations/models.py | 370 | 425| 539 | 7132 | 63267 | 
| 16 | 7 django/db/migrations/state.py | 142 | 179| 395 | 7527 | 63267 | 
| 17 | 8 django/db/models/base.py | 1298 | 1345| 409 | 7936 | 81808 | 
| 18 | **8 django/db/migrations/autodetector.py** | 1499 | 1525| 217 | 8153 | 81808 | 
| 19 | 9 django/db/models/constraints.py | 238 | 252| 117 | 8270 | 84575 | 
| 20 | 9 django/db/models/fields/related.py | 603 | 669| 497 | 8767 | 84575 | 
| 21 | **9 django/db/migrations/autodetector.py** | 600 | 774| 1213 | 9980 | 84575 | 
| 22 | 9 django/db/migrations/operations/models.py | 645 | 669| 231 | 10211 | 84575 | 
| 23 | 10 django/forms/models.py | 796 | 887| 783 | 10994 | 96819 | 
| 24 | 10 django/db/migrations/state.py | 488 | 510| 225 | 11219 | 96819 | 
| 25 | **10 django/db/migrations/autodetector.py** | 1427 | 1472| 318 | 11537 | 96819 | 
| 26 | 10 django/db/migrations/state.py | 126 | 140| 163 | 11700 | 96819 | 
| 27 | 10 django/db/migrations/operations/models.py | 571 | 595| 213 | 11913 | 96819 | 
| 28 | 10 django/db/models/base.py | 1873 | 1901| 188 | 12101 | 96819 | 
| 29 | **10 django/db/migrations/autodetector.py** | 1392 | 1425| 287 | 12388 | 96819 | 
| 30 | 10 django/db/migrations/operations/models.py | 1 | 18| 137 | 12525 | 96819 | 
| 31 | 11 django/db/migrations/questioner.py | 57 | 87| 255 | 12780 | 99515 | 
| 32 | 12 django/db/migrations/operations/__init__.py | 1 | 43| 227 | 13007 | 99742 | 
| 33 | 13 django/db/models/deletion.py | 1 | 86| 569 | 13576 | 103674 | 
| 34 | 13 django/db/migrations/operations/models.py | 598 | 619| 148 | 13724 | 103674 | 
| 35 | 13 django/db/migrations/state.py | 970 | 986| 140 | 13864 | 103674 | 
| 36 | 13 django/db/models/constraints.py | 220 | 236| 139 | 14003 | 103674 | 
| 37 | 13 django/db/models/fields/related.py | 1881 | 1928| 505 | 14508 | 103674 | 
| 38 | 13 django/db/models/constraints.py | 202 | 218| 138 | 14646 | 103674 | 
| 39 | 13 django/db/migrations/operations/models.py | 560 | 569| 129 | 14775 | 103674 | 
| 40 | 13 django/db/models/base.py | 1395 | 1425| 216 | 14991 | 103674 | 
| 41 | 13 django/db/migrations/operations/models.py | 136 | 306| 968 | 15959 | 103674 | 
| 42 | 13 django/db/backends/base/schema.py | 541 | 560| 199 | 16158 | 103674 | 
| 43 | 13 django/db/models/fields/related.py | 1261 | 1315| 421 | 16579 | 103674 | 
| 44 | 13 django/db/models/fields/related.py | 1017 | 1053| 261 | 16840 | 103674 | 
| 45 | 13 django/db/models/fields/related.py | 1959 | 1993| 266 | 17106 | 103674 | 
| 46 | 14 django/db/models/fields/related_descriptors.py | 770 | 845| 592 | 17698 | 114737 | 
| 47 | 14 django/db/migrations/operations/models.py | 1069 | 1109| 337 | 18035 | 114737 | 
| 48 | 14 django/forms/models.py | 500 | 530| 243 | 18278 | 114737 | 
| 49 | 14 django/db/models/constraints.py | 293 | 350| 457 | 18735 | 114737 | 
| 50 | 15 django/contrib/auth/migrations/0007_alter_validators_add_error_messages.py | 1 | 28| 143 | 18878 | 114880 | 
| 51 | 15 django/db/backends/sqlite3/schema.py | 265 | 360| 807 | 19685 | 114880 | 
| 52 | 15 django/db/backends/sqlite3/schema.py | 123 | 174| 527 | 20212 | 114880 | 
| 53 | 15 django/db/models/base.py | 1650 | 1684| 248 | 20460 | 114880 | 
| 54 | 15 django/db/migrations/questioner.py | 291 | 342| 367 | 20827 | 114880 | 
| 55 | 15 django/db/models/base.py | 1074 | 1126| 516 | 21343 | 114880 | 
| 56 | 16 django/db/backends/mysql/schema.py | 138 | 156| 209 | 21552 | 116503 | 
| 57 | 16 django/db/backends/base/schema.py | 562 | 594| 268 | 21820 | 116503 | 
| 58 | 16 django/db/migrations/operations/models.py | 671 | 687| 159 | 21979 | 116503 | 
| 59 | 16 django/db/migrations/state.py | 869 | 901| 250 | 22229 | 116503 | 
| 60 | 17 django/contrib/admin/migrations/0001_initial.py | 1 | 77| 363 | 22592 | 116866 | 
| 61 | 17 django/db/models/base.py | 2255 | 2446| 1302 | 23894 | 116866 | 
| 62 | 17 django/db/models/fields/related.py | 1930 | 1957| 305 | 24199 | 116866 | 
| 63 | **17 django/db/migrations/autodetector.py** | 1096 | 1213| 982 | 25181 | 116866 | 
| 64 | 17 django/db/models/base.py | 1193 | 1233| 306 | 25487 | 116866 | 
| 65 | **17 django/db/migrations/autodetector.py** | 1341 | 1362| 197 | 25684 | 116866 | 
| 66 | 17 django/db/backends/base/schema.py | 38 | 71| 214 | 25898 | 116866 | 
| 67 | **17 django/db/migrations/autodetector.py** | 1588 | 1620| 258 | 26156 | 116866 | 
| 68 | 18 django/db/migrations/exceptions.py | 1 | 61| 249 | 26405 | 117116 | 
| 69 | 18 django/db/backends/base/schema.py | 1513 | 1551| 240 | 26645 | 117116 | 
| 70 | 19 django/contrib/contenttypes/migrations/0001_initial.py | 1 | 47| 225 | 26870 | 117341 | 
| 71 | 19 django/db/models/fields/related.py | 1732 | 1770| 413 | 27283 | 117341 | 
| 72 | 19 django/db/models/base.py | 1994 | 2047| 355 | 27638 | 117341 | 
| 73 | 19 django/db/models/base.py | 1809 | 1842| 232 | 27870 | 117341 | 
| 74 | 19 django/db/backends/base/schema.py | 1710 | 1747| 303 | 28173 | 117341 | 
| 75 | 19 django/db/models/fields/related.py | 302 | 339| 296 | 28469 | 117341 | 
| 76 | 19 django/db/migrations/operations/models.py | 1028 | 1066| 283 | 28752 | 117341 | 
| 77 | 19 django/db/models/fields/related_descriptors.py | 1368 | 1418| 403 | 29155 | 117341 | 
| 78 | 19 django/db/backends/base/schema.py | 1058 | 1137| 774 | 29929 | 117341 | 
| 79 | 20 django/db/migrations/operations/fields.py | 227 | 237| 146 | 30075 | 119853 | 
| 80 | 20 django/forms/models.py | 889 | 912| 198 | 30273 | 119853 | 
| 81 | 20 django/db/backends/base/schema.py | 794 | 884| 773 | 31046 | 119853 | 
| 82 | 20 django/db/backends/base/schema.py | 74 | 151| 798 | 31844 | 119853 | 
| 83 | 20 django/db/backends/sqlite3/schema.py | 176 | 264| 822 | 32666 | 119853 | 
| 84 | 20 django/db/migrations/state.py | 347 | 395| 371 | 33037 | 119853 | 
| 85 | 20 django/db/migrations/state.py | 512 | 536| 184 | 33221 | 119853 | 
| **-> 86 <-** | **20 django/db/migrations/autodetector.py** | 1023 | 1071| 360 | 33581 | 119853 | 
| 87 | 20 django/db/migrations/state.py | 411 | 435| 213 | 33794 | 119853 | 
| 88 | 20 django/db/models/base.py | 1128 | 1147| 188 | 33982 | 119853 | 
| 89 | 20 django/db/backends/base/schema.py | 1553 | 1604| 343 | 34325 | 119853 | 
| 90 | **20 django/db/migrations/autodetector.py** | 1215 | 1302| 719 | 35044 | 119853 | 
| 91 | 20 django/db/models/fields/related_descriptors.py | 1177 | 1211| 341 | 35385 | 119853 | 
| 92 | 20 django/db/models/base.py | 1523 | 1558| 273 | 35658 | 119853 | 
| 93 | 20 django/db/backends/base/schema.py | 1489 | 1511| 199 | 35857 | 119853 | 
| 94 | 20 django/db/backends/base/schema.py | 885 | 969| 775 | 36632 | 119853 | 
| 95 | 20 django/db/models/deletion.py | 310 | 370| 616 | 37248 | 119853 | 
| 96 | 20 django/db/backends/mysql/schema.py | 1 | 42| 456 | 37704 | 119853 | 
| 97 | 21 django/db/backends/postgresql/schema.py | 115 | 187| 602 | 38306 | 122076 | 
| 98 | **21 django/db/migrations/autodetector.py** | 583 | 599| 186 | 38492 | 122076 | 
| 99 | 21 django/db/backends/postgresql/schema.py | 189 | 235| 408 | 38900 | 122076 | 
| 100 | **21 django/db/migrations/autodetector.py** | 904 | 977| 623 | 39523 | 122076 | 
| 101 | 22 django/db/backends/oracle/schema.py | 75 | 103| 344 | 39867 | 124408 | 
| 102 | 22 django/db/backends/sqlite3/schema.py | 425 | 482| 488 | 40355 | 124408 | 
| 103 | 22 django/db/models/base.py | 2156 | 2237| 588 | 40943 | 124408 | 
| 104 | 22 django/db/models/fields/related_descriptors.py | 1297 | 1366| 522 | 41465 | 124408 | 
| 105 | 22 django/db/migrations/questioner.py | 189 | 215| 238 | 41703 | 124408 | 
| 106 | 22 django/db/models/fields/related.py | 900 | 989| 583 | 42286 | 124408 | 
| 107 | 23 django/contrib/sessions/migrations/0001_initial.py | 1 | 39| 173 | 42459 | 124581 | 
| 108 | 23 django/db/models/constraints.py | 266 | 291| 205 | 42664 | 124581 | 
| 109 | 23 django/db/backends/base/schema.py | 1606 | 1643| 243 | 42907 | 124581 | 
| 110 | 24 django/contrib/redirects/migrations/0001_initial.py | 1 | 66| 309 | 43216 | 124890 | 
| 111 | 24 django/db/migrations/operations/models.py | 427 | 440| 127 | 43343 | 124890 | 
| 112 | 24 django/db/backends/base/schema.py | 1447 | 1466| 175 | 43518 | 124890 | 
| 113 | 24 django/db/migrations/operations/fields.py | 239 | 267| 206 | 43724 | 124890 | 
| 114 | 25 django/contrib/sites/migrations/0001_initial.py | 1 | 45| 210 | 43934 | 125100 | 
| 115 | **25 django/db/migrations/autodetector.py** | 516 | 581| 482 | 44416 | 125100 | 
| 116 | 25 django/db/migrations/state.py | 397 | 409| 136 | 44552 | 125100 | 
| 117 | 25 django/db/models/fields/related.py | 581 | 601| 138 | 44690 | 125100 | 
| 118 | 25 django/db/backends/sqlite3/schema.py | 100 | 121| 195 | 44885 | 125100 | 
| 119 | 25 django/db/models/fields/related.py | 991 | 1015| 176 | 45061 | 125100 | 
| 120 | 25 django/db/models/fields/related_descriptors.py | 1089 | 1115| 220 | 45281 | 125100 | 
| 121 | 25 django/db/migrations/operations/fields.py | 270 | 337| 521 | 45802 | 125100 | 
| 122 | 25 django/db/models/fields/related.py | 1318 | 1415| 574 | 46376 | 125100 | 
| 123 | **25 django/db/migrations/autodetector.py** | 1474 | 1497| 161 | 46537 | 125100 | 
| 124 | 26 django/contrib/auth/migrations/0008_alter_user_username_max_length.py | 1 | 28| 144 | 46681 | 125244 | 
| 125 | 26 django/db/backends/base/schema.py | 970 | 1057| 824 | 47505 | 125244 | 
| 126 | 27 django/db/models/sql/datastructures.py | 126 | 144| 141 | 47646 | 126737 | 
| 127 | 27 django/db/models/constraints.py | 254 | 264| 163 | 47809 | 126737 | 
| 128 | 27 django/db/backends/oracle/schema.py | 105 | 170| 740 | 48549 | 126737 | 
| 129 | **27 django/db/migrations/autodetector.py** | 210 | 231| 238 | 48787 | 126737 | 
| 130 | 28 django/core/serializers/xml_serializer.py | 127 | 173| 366 | 49153 | 130333 | 
| 131 | 28 django/db/models/base.py | 1705 | 1758| 490 | 49643 | 130333 | 
| 132 | 29 django/contrib/flatpages/migrations/0001_initial.py | 1 | 70| 355 | 49998 | 130688 | 
| 133 | **29 django/db/migrations/autodetector.py** | 21 | 38| 185 | 50183 | 130688 | 
| 134 | 29 django/db/models/fields/related.py | 1078 | 1100| 180 | 50363 | 130688 | 
| 135 | 29 django/db/backends/sqlite3/schema.py | 398 | 423| 246 | 50609 | 130688 | 
| 136 | 29 django/db/backends/sqlite3/schema.py | 536 | 560| 162 | 50771 | 130688 | 
| 137 | 29 django/db/models/fields/related.py | 208 | 224| 142 | 50913 | 130688 | 
| 138 | 29 django/forms/models.py | 1358 | 1393| 225 | 51138 | 130688 | 
| 139 | 29 django/db/migrations/state.py | 651 | 677| 270 | 51408 | 130688 | 
| 140 | 30 django/contrib/gis/db/backends/mysql/schema.py | 49 | 73| 191 | 51599 | 131353 | 
| 141 | **30 django/db/migrations/autodetector.py** | 1364 | 1390| 144 | 51743 | 131353 | 
| 142 | 30 django/db/models/fields/related_descriptors.py | 1117 | 1143| 198 | 51941 | 131353 | 
| 143 | 31 django/contrib/auth/migrations/0004_alter_user_username_opts.py | 1 | 29| 158 | 52099 | 131511 | 
| 144 | 32 django/contrib/gis/db/backends/spatialite/schema.py | 137 | 192| 404 | 52503 | 132905 | 
| 145 | 32 django/db/migrations/state.py | 708 | 762| 472 | 52975 | 132905 | 
| 146 | 32 django/db/backends/base/schema.py | 1468 | 1487| 176 | 53151 | 132905 | 
| 147 | 32 django/db/migrations/state.py | 93 | 124| 256 | 53407 | 132905 | 
| 148 | 32 django/db/models/fields/related.py | 870 | 897| 237 | 53644 | 132905 | 
| 149 | 32 django/db/models/fields/related.py | 1417 | 1447| 172 | 53816 | 132905 | 
| 150 | 32 django/db/models/base.py | 1427 | 1449| 193 | 54009 | 132905 | 
| 151 | 32 django/db/migrations/operations/models.py | 533 | 558| 163 | 54172 | 132905 | 
| 152 | **32 django/db/migrations/autodetector.py** | 1622 | 1636| 135 | 54307 | 132905 | 
| 153 | 32 django/db/migrations/operations/fields.py | 101 | 113| 130 | 54437 | 132905 | 
| 154 | 32 django/db/backends/base/schema.py | 698 | 728| 293 | 54730 | 132905 | 
| 155 | 32 django/db/migrations/operations/models.py | 41 | 111| 524 | 55254 | 132905 | 
| 156 | 32 django/db/models/fields/related.py | 706 | 730| 210 | 55464 | 132905 | 
| 157 | 32 django/db/migrations/questioner.py | 166 | 187| 188 | 55652 | 132905 | 
| 158 | 32 django/db/models/deletion.py | 372 | 396| 273 | 55925 | 132905 | 
| 159 | 33 django/core/management/commands/makemigrations.py | 294 | 404| 927 | 56852 | 136020 | 
| 160 | 33 django/db/models/fields/related_descriptors.py | 733 | 768| 264 | 57116 | 136020 | 
| 161 | 34 django/contrib/contenttypes/migrations/0002_remove_content_type_name.py | 1 | 44| 232 | 57348 | 136252 | 
| 162 | 35 django/core/management/commands/migrate.py | 263 | 361| 813 | 58161 | 140159 | 
| 163 | 35 django/db/migrations/questioner.py | 269 | 288| 195 | 58356 | 140159 | 
| 164 | 35 django/db/backends/base/schema.py | 730 | 792| 524 | 58880 | 140159 | 
| 165 | 35 django/db/backends/sqlite3/schema.py | 362 | 378| 132 | 59012 | 140159 | 
| 166 | 35 django/db/models/deletion.py | 431 | 509| 601 | 59613 | 140159 | 
| 167 | **35 django/db/migrations/autodetector.py** | 1527 | 1546| 187 | 59800 | 140159 | 
| 168 | 35 django/db/migrations/state.py | 955 | 968| 138 | 59938 | 140159 | 
| 169 | 35 django/db/models/fields/related_descriptors.py | 694 | 731| 341 | 60279 | 140159 | 
| 170 | 36 django/db/migrations/recorder.py | 1 | 22| 148 | 60427 | 140846 | 
| 171 | 37 django/db/models/__init__.py | 1 | 116| 682 | 61109 | 141528 | 
| 172 | 37 django/db/models/fields/related.py | 671 | 704| 335 | 61444 | 141528 | 
| 173 | 37 django/db/models/constraints.py | 115 | 200| 684 | 62128 | 141528 | 
| 174 | 37 django/core/management/commands/makemigrations.py | 181 | 241| 441 | 62569 | 141528 | 
| 175 | 38 django/db/backends/mysql/operations.py | 441 | 470| 274 | 62843 | 145727 | 
| 176 | 38 django/db/backends/mysql/schema.py | 104 | 118| 144 | 62987 | 145727 | 
| 177 | 38 django/db/models/fields/related.py | 1102 | 1119| 133 | 63120 | 145727 | 
| 178 | 38 django/db/models/fields/related_descriptors.py | 847 | 875| 229 | 63349 | 145727 | 
| 179 | 39 django/contrib/sites/migrations/0002_alter_domain_unique.py | 1 | 23| 0 | 63349 | 145826 | 
| 180 | 40 django/contrib/auth/migrations/0011_update_proxy_permissions.py | 1 | 56| 455 | 63804 | 146394 | 
| 181 | 40 django/db/models/base.py | 592 | 630| 322 | 64126 | 146394 | 
| 182 | 41 django/db/models/fields/__init__.py | 1158 | 1178| 143 | 64269 | 165117 | 
| 183 | **41 django/db/migrations/autodetector.py** | 1073 | 1094| 188 | 64457 | 165117 | 
| 184 | 41 django/contrib/auth/migrations/0011_update_proxy_permissions.py | 59 | 77| 113 | 64570 | 165117 | 
| 185 | 41 django/db/models/fields/related_descriptors.py | 1145 | 1175| 256 | 64826 | 165117 | 
| 186 | 41 django/db/migrations/operations/fields.py | 154 | 195| 348 | 65174 | 165117 | 
| 187 | 41 django/db/backends/oracle/schema.py | 52 | 73| 146 | 65320 | 165117 | 
| 188 | 41 django/db/migrations/operations/fields.py | 115 | 127| 131 | 65451 | 165117 | 
| 189 | 41 django/db/models/fields/related.py | 1176 | 1208| 246 | 65697 | 165117 | 
| 190 | 41 django/db/models/base.py | 1844 | 1871| 187 | 65884 | 165117 | 


### Hint

```
Download this file and then do makemigrations and migrate to see this error.
Thanks for the report. Tentatively accepting, however I'm not sure if we can sort these operations properly, we should probably alter unique_together first migrations.AlterUniqueTogether( name='authors', unique_together=set(), ), migrations.RemoveField( model_name='authors', name='project_data_set', ), migrations.AddField( model_name='authors', name='project_data_set', field=models.ManyToManyField(to='dict.ProjectDataSet'), ), You should take into account that you'll lose all data because ForeignKey cannot be altered to ManyToManyField.
I agree that you'll loose data but Alter(Index|Unique)Together should always be sorted before RemoveField ​https://github.com/django/django/blob/b502061027b90499f2e20210f944292cecd74d24/django/db/migrations/autodetector.py#L910 ​https://github.com/django/django/blob/b502061027b90499f2e20210f944292cecd74d24/django/db/migrations/autodetector.py#L424-L430 So something's broken here in a few different ways and I suspect it's due to the fact the same field name project_data_set is reused for the many-to-many field. If you start from class Authors(models.Model): project_data_set = models.ForeignKey( ProjectDataSet, on_delete=models.PROTECT ) state = models.IntegerField() start_date = models.DateField() class Meta: unique_together = (('project_data_set', 'state', 'start_date'),) And generate makemigrations for class Authors(models.Model): project_data_set = models.ManyToManyField(ProjectDataSet) state = models.IntegerField() start_date = models.DateField() You'll get two migrations with the following operations # 0002 operations = [ migrations.AddField( model_name='authors', name='project_data_set', field=models.ManyToManyField(to='ticket_31788.ProjectDataSet'), ), migrations.AlterUniqueTogether( name='authors', unique_together=set(), ), migrations.RemoveField( model_name='authors', name='project_data_set', ), ] # 0003 operations = [ migrations.AddField( model_name='authors', name='project_data_set', field=models.ManyToManyField(to='ticket_31788.ProjectDataSet'), ), ] If you change the name of the field to something else like project_data_sets every work as expected operations = [ migrations.AddField( model_name='authors', name='project_data_sets', field=models.ManyToManyField(to='ticket_31788.ProjectDataSet'), ), migrations.AlterUniqueTogether( name='authors', unique_together=set(), ), migrations.RemoveField( model_name='authors', name='project_data_set', ), ] It seems like there's some bad interactions between generate_removed_fields and generate_added_fields when a field with the same name is added.
```

## Patch

```diff
diff --git a/django/db/migrations/autodetector.py b/django/db/migrations/autodetector.py
--- a/django/db/migrations/autodetector.py
+++ b/django/db/migrations/autodetector.py
@@ -1022,8 +1022,9 @@ def generate_added_fields(self):
 
     def _generate_added_field(self, app_label, model_name, field_name):
         field = self.to_state.models[app_label, model_name].get_field(field_name)
-        # Fields that are foreignkeys/m2ms depend on stuff
-        dependencies = []
+        # Adding a field always depends at least on its removal.
+        dependencies = [(app_label, model_name, field_name, False)]
+        # Fields that are foreignkeys/m2ms depend on stuff.
         if field.remote_field and field.remote_field.model:
             dependencies.extend(
                 self._get_dependencies_for_foreign_key(

```

## Test Patch

```diff
diff --git a/tests/migrations/test_autodetector.py b/tests/migrations/test_autodetector.py
--- a/tests/migrations/test_autodetector.py
+++ b/tests/migrations/test_autodetector.py
@@ -868,6 +868,18 @@ class AutodetectorTests(TestCase):
             "unique_together": {("title", "newfield2")},
         },
     )
+    book_unique_together = ModelState(
+        "otherapp",
+        "Book",
+        [
+            ("id", models.AutoField(primary_key=True)),
+            ("author", models.ForeignKey("testapp.Author", models.CASCADE)),
+            ("title", models.CharField(max_length=200)),
+        ],
+        {
+            "unique_together": {("author", "title")},
+        },
+    )
     attribution = ModelState(
         "otherapp",
         "Attribution",
@@ -3798,16 +3810,16 @@ def test_many_to_many_changed_to_concrete_field(self):
         # Right number/type of migrations?
         self.assertNumberMigrations(changes, "testapp", 1)
         self.assertOperationTypes(
-            changes, "testapp", 0, ["RemoveField", "AddField", "DeleteModel"]
+            changes, "testapp", 0, ["RemoveField", "DeleteModel", "AddField"]
         )
         self.assertOperationAttributes(
             changes, "testapp", 0, 0, name="publishers", model_name="author"
         )
+        self.assertOperationAttributes(changes, "testapp", 0, 1, name="Publisher")
         self.assertOperationAttributes(
-            changes, "testapp", 0, 1, name="publishers", model_name="author"
+            changes, "testapp", 0, 2, name="publishers", model_name="author"
         )
-        self.assertOperationAttributes(changes, "testapp", 0, 2, name="Publisher")
-        self.assertOperationFieldAttributes(changes, "testapp", 0, 1, max_length=100)
+        self.assertOperationFieldAttributes(changes, "testapp", 0, 2, max_length=100)
 
     def test_non_circular_foreignkey_dependency_removal(self):
         """
@@ -4346,6 +4358,36 @@ def test_fk_dependency_other_app(self):
             changes, "testapp", 0, [("otherapp", "__first__")]
         )
 
+    def test_alter_unique_together_fk_to_m2m(self):
+        changes = self.get_changes(
+            [self.author_name, self.book_unique_together],
+            [
+                self.author_name,
+                ModelState(
+                    "otherapp",
+                    "Book",
+                    [
+                        ("id", models.AutoField(primary_key=True)),
+                        ("author", models.ManyToManyField("testapp.Author")),
+                        ("title", models.CharField(max_length=200)),
+                    ],
+                ),
+            ],
+        )
+        self.assertNumberMigrations(changes, "otherapp", 1)
+        self.assertOperationTypes(
+            changes, "otherapp", 0, ["AlterUniqueTogether", "RemoveField", "AddField"]
+        )
+        self.assertOperationAttributes(
+            changes, "otherapp", 0, 0, name="book", unique_together=set()
+        )
+        self.assertOperationAttributes(
+            changes, "otherapp", 0, 1, model_name="book", name="author"
+        )
+        self.assertOperationAttributes(
+            changes, "otherapp", 0, 2, model_name="book", name="author"
+        )
+
     def test_alter_field_to_fk_dependency_other_app(self):
         changes = self.get_changes(
             [self.author_empty, self.book_with_no_author_fk],

```


## Code snippets

### 1 - django/db/backends/base/schema.py:

Start line: 520, End line: 539

```python
class BaseDatabaseSchemaEditor:

    def alter_unique_together(self, model, old_unique_together, new_unique_together):
        """
        Deal with a model changing its unique_together. The input
        unique_togethers must be doubly-nested, not the single-nested
        ["foo", "bar"] format.
        """
        olds = {tuple(fields) for fields in old_unique_together}
        news = {tuple(fields) for fields in new_unique_together}
        # Deleted uniques
        for fields in olds.difference(news):
            self._delete_composed_index(
                model,
                fields,
                {"unique": True, "primary_key": False},
                self.sql_delete_unique,
            )
        # Created uniques
        for field_names in news.difference(olds):
            fields = [model._meta.get_field(field) for field in field_names]
            self.execute(self._create_unique_sql(model, fields))
```
### 2 - django/db/models/fields/related.py:

Start line: 1449, End line: 1578

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
### 3 - django/db/migrations/state.py:

Start line: 437, End line: 458

```python
class ProjectState:

    def update_model_field_relation(
        self,
        model,
        model_key,
        field_name,
        field,
        concretes,
    ):
        remote_model_key = resolve_relation(model, *model_key)
        if remote_model_key[0] not in self.real_apps and remote_model_key in concretes:
            remote_model_key = concretes[remote_model_key]
        relations_to_remote_model = self._relations[remote_model_key]
        if field_name in self.models[model_key].fields:
            # The assert holds because it's a new relation, or an altered
            # relation, in which case references have been removed by
            # alter_field().
            assert field_name not in relations_to_remote_model[model_key]
            relations_to_remote_model[model_key][field_name] = field
        else:
            del relations_to_remote_model[model_key][field_name]
            if not relations_to_remote_model[model_key]:
                del relations_to_remote_model[model_key]
```
### 4 - django/db/migrations/state.py:

Start line: 291, End line: 345

```python
class ProjectState:

    def rename_field(self, app_label, model_name, old_name, new_name):
        model_key = app_label, model_name
        model_state = self.models[model_key]
        # Rename the field.
        fields = model_state.fields
        try:
            found = fields.pop(old_name)
        except KeyError:
            raise FieldDoesNotExist(
                f"{app_label}.{model_name} has no field named '{old_name}'"
            )
        fields[new_name] = found
        for field in fields.values():
            # Fix from_fields to refer to the new field.
            from_fields = getattr(field, "from_fields", None)
            if from_fields:
                field.from_fields = tuple(
                    [
                        new_name if from_field_name == old_name else from_field_name
                        for from_field_name in from_fields
                    ]
                )
        # Fix index/unique_together to refer to the new field.
        options = model_state.options
        for option in ("index_together", "unique_together"):
            if option in options:
                options[option] = [
                    [new_name if n == old_name else n for n in together]
                    for together in options[option]
                ]
        # Fix to_fields to refer to the new field.
        delay = True
        references = get_references(self, model_key, (old_name, found))
        for *_, field, reference in references:
            delay = False
            if reference.to:
                remote_field, to_fields = reference.to
                if getattr(remote_field, "field_name", None) == old_name:
                    remote_field.field_name = new_name
                if to_fields:
                    field.to_fields = tuple(
                        [
                            new_name if to_field_name == old_name else to_field_name
                            for to_field_name in to_fields
                        ]
                    )
        if self._relations is not None:
            old_name_lower = old_name.lower()
            new_name_lower = new_name.lower()
            for to_model in self._relations.values():
                if old_name_lower in to_model[model_key]:
                    field = to_model[model_key].pop(old_name_lower)
                    field.name = new_name_lower
                    to_model[model_key][new_name_lower] = field
        self.reload_model(*model_key, delay=delay)
```
### 5 - django/db/migrations/state.py:

Start line: 460, End line: 486

```python
class ProjectState:

    def resolve_model_field_relations(
        self,
        model_key,
        field_name,
        field,
        concretes=None,
    ):
        remote_field = field.remote_field
        if not remote_field:
            return
        if concretes is None:
            concretes, _ = self._get_concrete_models_mapping_and_proxy_models()

        self.update_model_field_relation(
            remote_field.model,
            model_key,
            field_name,
            field,
            concretes,
        )

        through = getattr(remote_field, "through", None)
        if not through:
            return
        self.update_model_field_relation(
            through, model_key, field_name, field, concretes
        )
```
### 6 - django/db/models/fields/related.py:

Start line: 1680, End line: 1730

```python
class ManyToManyField(RelatedField):

    def _check_table_uniqueness(self, **kwargs):
        if (
            isinstance(self.remote_field.through, str)
            or not self.remote_field.through._meta.managed
        ):
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
        if (
            model
            and model._meta.concrete_model
            != self.remote_field.through._meta.concrete_model
        ):
            if model._meta.auto_created:

                def _get_field_name(model):
                    for field in model._meta.auto_created._meta.many_to_many:
                        if field.remote_field.through is model:
                            return field.name

                opts = model._meta.auto_created._meta
                clashing_obj = "%s.%s" % (opts.label, _get_field_name(model))
            else:
                clashing_obj = model._meta.label
            if settings.DATABASE_ROUTERS:
                error_class, error_id = checks.Warning, "fields.W344"
                error_hint = (
                    "You have configured settings.DATABASE_ROUTERS. Verify "
                    "that the table of %r is correctly routed to a separate "
                    "database." % clashing_obj
                )
            else:
                error_class, error_id = checks.Error, "fields.E340"
                error_hint = None
            return [
                error_class(
                    "The field's intermediary table '%s' clashes with the "
                    "table name of '%s'." % (m2m_db_table, clashing_obj),
                    obj=self,
                    hint=error_hint,
                    id=error_id,
                )
            ]
        return []
```
### 7 - django/contrib/auth/migrations/0001_initial.py:

Start line: 1, End line: 206

```python
import django.contrib.auth.models
from django.contrib.auth import validators
from django.db import migrations, models
from django.utils import timezone


class Migration(migrations.Migration):

    dependencies = [
        ("contenttypes", "__first__"),
    ]

    operations = [
        migrations.CreateModel(
            name="Permission",
            fields=[
                (
                    "id",
                    models.AutoField(
                        verbose_name="ID",
                        serialize=False,
                        auto_created=True,
                        primary_key=True,
                    ),
                ),
                ("name", models.CharField(max_length=50, verbose_name="name")),
                (
                    "content_type",
                    models.ForeignKey(
                        to="contenttypes.ContentType",
                        on_delete=models.CASCADE,
                        verbose_name="content type",
                    ),
                ),
                ("codename", models.CharField(max_length=100, verbose_name="codename")),
            ],
            options={
                "ordering": [
                    "content_type__app_label",
                    "content_type__model",
                    "codename",
                ],
                "unique_together": {("content_type", "codename")},
                "verbose_name": "permission",
                "verbose_name_plural": "permissions",
            },
            managers=[
                ("objects", django.contrib.auth.models.PermissionManager()),
            ],
        ),
        migrations.CreateModel(
            name="Group",
            fields=[
                (
                    "id",
                    models.AutoField(
                        verbose_name="ID",
                        serialize=False,
                        auto_created=True,
                        primary_key=True,
                    ),
                ),
                (
                    "name",
                    models.CharField(unique=True, max_length=80, verbose_name="name"),
                ),
                (
                    "permissions",
                    models.ManyToManyField(
                        to="auth.Permission", verbose_name="permissions", blank=True
                    ),
                ),
            ],
            options={
                "verbose_name": "group",
                "verbose_name_plural": "groups",
            },
            managers=[
                ("objects", django.contrib.auth.models.GroupManager()),
            ],
        ),
        migrations.CreateModel(
            name="User",
            fields=[
                (
                    "id",
                    models.AutoField(
                        verbose_name="ID",
                        serialize=False,
                        auto_created=True,
                        primary_key=True,
                    ),
                ),
                ("password", models.CharField(max_length=128, verbose_name="password")),
                (
                    "last_login",
                    models.DateTimeField(
                        default=timezone.now, verbose_name="last login"
                    ),
                ),
                (
                    "is_superuser",
                    models.BooleanField(
                        default=False,
                        help_text=(
                            "Designates that this user has all permissions without "
                            "explicitly assigning them."
                        ),
                        verbose_name="superuser status",
                    ),
                ),
                (
                    "username",
                    models.CharField(
                        help_text=(
                            "Required. 30 characters or fewer. Letters, digits and "
                            "@/./+/-/_ only."
                        ),
                        unique=True,
                        max_length=30,
                        verbose_name="username",
                        validators=[validators.UnicodeUsernameValidator()],
                    ),
                ),
                (
                    "first_name",
                    models.CharField(
                        max_length=30, verbose_name="first name", blank=True
                    ),
                ),
                (
                    "last_name",
                    models.CharField(
                        max_length=30, verbose_name="last name", blank=True
                    ),
                ),
                (
                    "email",
                    models.EmailField(
                        max_length=75, verbose_name="email address", blank=True
                    ),
                ),
                (
                    "is_staff",
                    models.BooleanField(
                        default=False,
                        help_text=(
                            "Designates whether the user can log into this admin site."
                        ),
                        verbose_name="staff status",
                    ),
                ),
                (
                    "is_active",
                    models.BooleanField(
                        default=True,
                        verbose_name="active",
                        help_text=(
                            "Designates whether this user should be treated as active. "
                            "Unselect this instead of deleting accounts."
                        ),
                    ),
                ),
                (
                    "date_joined",
                    models.DateTimeField(
                        default=timezone.now, verbose_name="date joined"
                    ),
                ),
                (
                    "groups",
                    models.ManyToManyField(
                        to="auth.Group",
                        verbose_name="groups",
                        blank=True,
                        related_name="user_set",
                        related_query_name="user",
                        help_text=(
                            "The groups this user belongs to. A user will get all "
                            "permissions granted to each of their groups."
                        ),
                    ),
                ),
                (
                    "user_permissions",
                    models.ManyToManyField(
                        to="auth.Permission",
                        verbose_name="user permissions",
                        blank=True,
                        help_text="Specific permissions for this user.",
                        related_name="user_set",
                        related_query_name="user",
                    ),
                ),
            ],
            options={
                "swappable": "AUTH_USER_MODEL",
                "verbose_name": "user",
                "verbose_name_plural": "users",
            },
            managers=[
                ("objects", django.contrib.auth.models.UserManager()),
            ],
        ),
    ]
```
### 8 - django/db/backends/sqlite3/schema.py:

Start line: 484, End line: 534

```python
class DatabaseSchemaEditor(BaseDatabaseSchemaEditor):

    def _alter_many_to_many(self, model, old_field, new_field, strict):
        """Alter M2Ms to repoint their to= endpoints."""
        if (
            old_field.remote_field.through._meta.db_table
            == new_field.remote_field.through._meta.db_table
        ):
            # The field name didn't change, but some options did, so we have to
            # propagate this altering.
            self._remake_table(
                old_field.remote_field.through,
                alter_field=(
                    # The field that points to the target model is needed, so
                    # we can tell alter_field to change it - this is
                    # m2m_reverse_field_name() (as opposed to m2m_field_name(),
                    # which points to our model).
                    old_field.remote_field.through._meta.get_field(
                        old_field.m2m_reverse_field_name()
                    ),
                    new_field.remote_field.through._meta.get_field(
                        new_field.m2m_reverse_field_name()
                    ),
                ),
            )
            return

        # Make a new through table
        self.create_model(new_field.remote_field.through)
        # Copy the data across
        self.execute(
            "INSERT INTO %s (%s) SELECT %s FROM %s"
            % (
                self.quote_name(new_field.remote_field.through._meta.db_table),
                ", ".join(
                    [
                        "id",
                        new_field.m2m_column_name(),
                        new_field.m2m_reverse_name(),
                    ]
                ),
                ", ".join(
                    [
                        "id",
                        old_field.m2m_column_name(),
                        old_field.m2m_reverse_name(),
                    ]
                ),
                self.quote_name(old_field.remote_field.through._meta.db_table),
            )
        )
        # Delete the old through table
        self.delete_model(old_field.remote_field.through)
```
### 9 - django/db/backends/base/schema.py:

Start line: 1241, End line: 1271

```python
class BaseDatabaseSchemaEditor:

    def _alter_many_to_many(self, model, old_field, new_field, strict):
        """Alter M2Ms to repoint their to= endpoints."""
        # Rename the through table
        if (
            old_field.remote_field.through._meta.db_table
            != new_field.remote_field.through._meta.db_table
        ):
            self.alter_db_table(
                old_field.remote_field.through,
                old_field.remote_field.through._meta.db_table,
                new_field.remote_field.through._meta.db_table,
            )
        # Repoint the FK to the other side
        self.alter_field(
            new_field.remote_field.through,
            # The field that points to the target model is needed, so we can
            # tell alter_field to change it - this is m2m_reverse_field_name()
            # (as opposed to m2m_field_name(), which points to our model).
            old_field.remote_field.through._meta.get_field(
                old_field.m2m_reverse_field_name()
            ),
            new_field.remote_field.through._meta.get_field(
                new_field.m2m_reverse_field_name()
            ),
        )
        self.alter_field(
            new_field.remote_field.through,
            # for self-referential models we need to alter field from the other end too
            old_field.remote_field.through._meta.get_field(old_field.m2m_field_name()),
            new_field.remote_field.through._meta.get_field(new_field.m2m_field_name()),
        )
```
### 10 - django/db/migrations/state.py:

Start line: 181, End line: 238

```python
class ProjectState:

    def alter_model_options(self, app_label, model_name, options, option_keys=None):
        model_state = self.models[app_label, model_name]
        model_state.options = {**model_state.options, **options}
        if option_keys:
            for key in option_keys:
                if key not in options:
                    model_state.options.pop(key, False)
        self.reload_model(app_label, model_name, delay=True)

    def remove_model_options(self, app_label, model_name, option_name, value_to_remove):
        model_state = self.models[app_label, model_name]
        if objs := model_state.options.get(option_name):
            model_state.options[option_name] = [
                obj for obj in objs if tuple(obj) != tuple(value_to_remove)
            ]
        self.reload_model(app_label, model_name, delay=True)

    def alter_model_managers(self, app_label, model_name, managers):
        model_state = self.models[app_label, model_name]
        model_state.managers = list(managers)
        self.reload_model(app_label, model_name, delay=True)

    def _append_option(self, app_label, model_name, option_name, obj):
        model_state = self.models[app_label, model_name]
        model_state.options[option_name] = [*model_state.options[option_name], obj]
        self.reload_model(app_label, model_name, delay=True)

    def _remove_option(self, app_label, model_name, option_name, obj_name):
        model_state = self.models[app_label, model_name]
        objs = model_state.options[option_name]
        model_state.options[option_name] = [obj for obj in objs if obj.name != obj_name]
        self.reload_model(app_label, model_name, delay=True)

    def add_index(self, app_label, model_name, index):
        self._append_option(app_label, model_name, "indexes", index)

    def remove_index(self, app_label, model_name, index_name):
        self._remove_option(app_label, model_name, "indexes", index_name)

    def rename_index(self, app_label, model_name, old_index_name, new_index_name):
        model_state = self.models[app_label, model_name]
        objs = model_state.options["indexes"]

        new_indexes = []
        for obj in objs:
            if obj.name == old_index_name:
                obj = obj.clone()
                obj.name = new_index_name
            new_indexes.append(obj)

        model_state.options["indexes"] = new_indexes
        self.reload_model(app_label, model_name, delay=True)

    def add_constraint(self, app_label, model_name, constraint):
        self._append_option(app_label, model_name, "constraints", constraint)

    def remove_constraint(self, app_label, model_name, constraint_name):
        self._remove_option(app_label, model_name, "constraints", constraint_name)
```
### 14 - django/db/migrations/autodetector.py:

Start line: 809, End line: 902

```python
class MigrationAutodetector:

    def generate_deleted_models(self):
        """
        Find all deleted models (managed and unmanaged) and make delete
        operations for them as well as separate operations to delete any
        foreign key or M2M relationships (these are optimized later, if
        possible).

        Also bring forward removal of any model options that refer to
        collections of fields - the inverse of generate_created_models().
        """
        new_keys = self.new_model_keys | self.new_unmanaged_keys
        deleted_models = self.old_model_keys - new_keys
        deleted_unmanaged_models = self.old_unmanaged_keys - new_keys
        all_deleted_models = chain(
            sorted(deleted_models), sorted(deleted_unmanaged_models)
        )
        for app_label, model_name in all_deleted_models:
            model_state = self.from_state.models[app_label, model_name]
            # Gather related fields
            related_fields = {}
            for field_name, field in model_state.fields.items():
                if field.remote_field:
                    if field.remote_field.model:
                        related_fields[field_name] = field
                    if getattr(field.remote_field, "through", None):
                        related_fields[field_name] = field
            # Generate option removal first
            unique_together = model_state.options.pop("unique_together", None)
            index_together = model_state.options.pop("index_together", None)
            if unique_together:
                self.add_operation(
                    app_label,
                    operations.AlterUniqueTogether(
                        name=model_name,
                        unique_together=None,
                    ),
                )
            if index_together:
                self.add_operation(
                    app_label,
                    operations.AlterIndexTogether(
                        name=model_name,
                        index_together=None,
                    ),
                )
            # Then remove each related field
            for name in sorted(related_fields):
                self.add_operation(
                    app_label,
                    operations.RemoveField(
                        model_name=model_name,
                        name=name,
                    ),
                )
            # Finally, remove the model.
            # This depends on both the removal/alteration of all incoming fields
            # and the removal of all its own related fields, and if it's
            # a through model the field that references it.
            dependencies = []
            relations = self.from_state.relations
            for (
                related_object_app_label,
                object_name,
            ), relation_related_fields in relations[app_label, model_name].items():
                for field_name, field in relation_related_fields.items():
                    dependencies.append(
                        (related_object_app_label, object_name, field_name, False),
                    )
                    if not field.many_to_many:
                        dependencies.append(
                            (
                                related_object_app_label,
                                object_name,
                                field_name,
                                "alter",
                            ),
                        )

            for name in sorted(related_fields):
                dependencies.append((app_label, model_name, name, False))
            # We're referenced in another field's through=
            through_user = self.through_users.get((app_label, model_state.name_lower))
            if through_user:
                dependencies.append(
                    (through_user[0], through_user[1], through_user[2], False)
                )
            # Finally, make the operation, deduping any dependencies
            self.add_operation(
                app_label,
                operations.DeleteModel(
                    name=model_state.name,
                ),
                dependencies=list(set(dependencies)),
            )
```
### 18 - django/db/migrations/autodetector.py:

Start line: 1499, End line: 1525

```python
class MigrationAutodetector:

    def generate_removed_altered_unique_together(self):
        self._generate_removed_altered_foo_together(operations.AlterUniqueTogether)

    def generate_removed_altered_index_together(self):
        self._generate_removed_altered_foo_together(operations.AlterIndexTogether)

    def _generate_altered_foo_together(self, operation):
        for (
            old_value,
            new_value,
            app_label,
            model_name,
            dependencies,
        ) in self._get_altered_foo_together_operations(operation.option_name):
            removal_value = new_value.intersection(old_value)
            if new_value != removal_value:
                self.add_operation(
                    app_label,
                    operation(name=model_name, **{operation.option_name: new_value}),
                    dependencies=dependencies,
                )

    def generate_altered_unique_together(self):
        self._generate_altered_foo_together(operations.AlterUniqueTogether)

    def generate_altered_index_together(self):
        self._generate_altered_foo_together(operations.AlterIndexTogether)
```
### 21 - django/db/migrations/autodetector.py:

Start line: 600, End line: 774

```python
class MigrationAutodetector:

    def generate_created_models(self):
        # ... other code
        for app_label, model_name in all_added_models:
            model_state = self.to_state.models[app_label, model_name]
            # Gather related fields
            related_fields = {}
            primary_key_rel = None
            for field_name, field in model_state.fields.items():
                if field.remote_field:
                    if field.remote_field.model:
                        if field.primary_key:
                            primary_key_rel = field.remote_field.model
                        elif not field.remote_field.parent_link:
                            related_fields[field_name] = field
                    if getattr(field.remote_field, "through", None):
                        related_fields[field_name] = field

            # Are there indexes/unique|index_together to defer?
            indexes = model_state.options.pop("indexes")
            constraints = model_state.options.pop("constraints")
            unique_together = model_state.options.pop("unique_together", None)
            index_together = model_state.options.pop("index_together", None)
            order_with_respect_to = model_state.options.pop(
                "order_with_respect_to", None
            )
            # Depend on the deletion of any possible proxy version of us
            dependencies = [
                (app_label, model_name, None, False),
            ]
            # Depend on all bases
            for base in model_state.bases:
                if isinstance(base, str) and "." in base:
                    base_app_label, base_name = base.split(".", 1)
                    dependencies.append((base_app_label, base_name, None, True))
                    # Depend on the removal of base fields if the new model has
                    # a field with the same name.
                    old_base_model_state = self.from_state.models.get(
                        (base_app_label, base_name)
                    )
                    new_base_model_state = self.to_state.models.get(
                        (base_app_label, base_name)
                    )
                    if old_base_model_state and new_base_model_state:
                        removed_base_fields = (
                            set(old_base_model_state.fields)
                            .difference(
                                new_base_model_state.fields,
                            )
                            .intersection(model_state.fields)
                        )
                        for removed_base_field in removed_base_fields:
                            dependencies.append(
                                (base_app_label, base_name, removed_base_field, False)
                            )
            # Depend on the other end of the primary key if it's a relation
            if primary_key_rel:
                dependencies.append(
                    resolve_relation(
                        primary_key_rel,
                        app_label,
                        model_name,
                    )
                    + (None, True)
                )
            # Generate creation operation
            self.add_operation(
                app_label,
                operations.CreateModel(
                    name=model_state.name,
                    fields=[
                        d
                        for d in model_state.fields.items()
                        if d[0] not in related_fields
                    ],
                    options=model_state.options,
                    bases=model_state.bases,
                    managers=model_state.managers,
                ),
                dependencies=dependencies,
                beginning=True,
            )

            # Don't add operations which modify the database for unmanaged models
            if not model_state.options.get("managed", True):
                continue

            # Generate operations for each related field
            for name, field in sorted(related_fields.items()):
                dependencies = self._get_dependencies_for_foreign_key(
                    app_label,
                    model_name,
                    field,
                    self.to_state,
                )
                # Depend on our own model being created
                dependencies.append((app_label, model_name, None, True))
                # Make operation
                self.add_operation(
                    app_label,
                    operations.AddField(
                        model_name=model_name,
                        name=name,
                        field=field,
                    ),
                    dependencies=list(set(dependencies)),
                )
            # Generate other opns
            if order_with_respect_to:
                self.add_operation(
                    app_label,
                    operations.AlterOrderWithRespectTo(
                        name=model_name,
                        order_with_respect_to=order_with_respect_to,
                    ),
                    dependencies=[
                        (app_label, model_name, order_with_respect_to, True),
                        (app_label, model_name, None, True),
                    ],
                )
            related_dependencies = [
                (app_label, model_name, name, True) for name in sorted(related_fields)
            ]
            related_dependencies.append((app_label, model_name, None, True))
            for index in indexes:
                self.add_operation(
                    app_label,
                    operations.AddIndex(
                        model_name=model_name,
                        index=index,
                    ),
                    dependencies=related_dependencies,
                )
            for constraint in constraints:
                self.add_operation(
                    app_label,
                    operations.AddConstraint(
                        model_name=model_name,
                        constraint=constraint,
                    ),
                    dependencies=related_dependencies,
                )
            if unique_together:
                self.add_operation(
                    app_label,
                    operations.AlterUniqueTogether(
                        name=model_name,
                        unique_together=unique_together,
                    ),
                    dependencies=related_dependencies,
                )
            if index_together:
                self.add_operation(
                    app_label,
                    operations.AlterIndexTogether(
                        name=model_name,
                        index_together=index_together,
                    ),
                    dependencies=related_dependencies,
                )
            # Fix relationships if the model changed from a proxy model to a
            # concrete model.
            relations = self.to_state.relations
            if (app_label, model_name) in self.old_proxy_keys:
                for related_model_key, related_fields in relations[
                    app_label, model_name
                ].items():
                    related_model_state = self.to_state.models[related_model_key]
                    for related_field_name, related_field in related_fields.items():
                        self.add_operation(
                            related_model_state.app_label,
                            operations.AlterField(
                                model_name=related_model_state.name,
                                name=related_field_name,
                                field=related_field,
                            ),
                            dependencies=[(app_label, model_name, None, True)],
                        )
```
### 25 - django/db/migrations/autodetector.py:

Start line: 1427, End line: 1472

```python
class MigrationAutodetector:

    def _get_altered_foo_together_operations(self, option_name):
        for app_label, model_name in sorted(self.kept_model_keys):
            old_model_name = self.renamed_models.get(
                (app_label, model_name), model_name
            )
            old_model_state = self.from_state.models[app_label, old_model_name]
            new_model_state = self.to_state.models[app_label, model_name]

            # We run the old version through the field renames to account for those
            old_value = old_model_state.options.get(option_name)
            old_value = (
                {
                    tuple(
                        self.renamed_fields.get((app_label, model_name, n), n)
                        for n in unique
                    )
                    for unique in old_value
                }
                if old_value
                else set()
            )

            new_value = new_model_state.options.get(option_name)
            new_value = set(new_value) if new_value else set()

            if old_value != new_value:
                dependencies = []
                for foo_togethers in new_value:
                    for field_name in foo_togethers:
                        field = new_model_state.get_field(field_name)
                        if field.remote_field and field.remote_field.model:
                            dependencies.extend(
                                self._get_dependencies_for_foreign_key(
                                    app_label,
                                    model_name,
                                    field,
                                    self.to_state,
                                )
                            )
                yield (
                    old_value,
                    new_value,
                    app_label,
                    model_name,
                    dependencies,
                )
```
### 29 - django/db/migrations/autodetector.py:

Start line: 1392, End line: 1425

```python
class MigrationAutodetector:

    @staticmethod
    def _get_dependencies_for_foreign_key(app_label, model_name, field, project_state):
        remote_field_model = None
        if hasattr(field.remote_field, "model"):
            remote_field_model = field.remote_field.model
        else:
            relations = project_state.relations[app_label, model_name]
            for (remote_app_label, remote_model_name), fields in relations.items():
                if any(
                    field == related_field.remote_field
                    for related_field in fields.values()
                ):
                    remote_field_model = f"{remote_app_label}.{remote_model_name}"
                    break
        # Account for FKs to swappable models
        swappable_setting = getattr(field, "swappable_setting", None)
        if swappable_setting is not None:
            dep_app_label = "__setting__"
            dep_object_name = swappable_setting
        else:
            dep_app_label, dep_object_name = resolve_relation(
                remote_field_model,
                app_label,
                model_name,
            )
        dependencies = [(dep_app_label, dep_object_name, None, True)]
        if getattr(field.remote_field, "through", None):
            through_app_label, through_object_name = resolve_relation(
                remote_field_model,
                app_label,
                model_name,
            )
            dependencies.append((through_app_label, through_object_name, None, True))
        return dependencies
```
### 63 - django/db/migrations/autodetector.py:

Start line: 1096, End line: 1213

```python
class MigrationAutodetector:

    def generate_altered_fields(self):
        """
        Make AlterField operations, or possibly RemovedField/AddField if alter
        isn't possible.
        """
        for app_label, model_name, field_name in sorted(
            self.old_field_keys & self.new_field_keys
        ):
            # Did the field change?
            old_model_name = self.renamed_models.get(
                (app_label, model_name), model_name
            )
            old_field_name = self.renamed_fields.get(
                (app_label, model_name, field_name), field_name
            )
            old_field = self.from_state.models[app_label, old_model_name].get_field(
                old_field_name
            )
            new_field = self.to_state.models[app_label, model_name].get_field(
                field_name
            )
            dependencies = []
            # Implement any model renames on relations; these are handled by RenameModel
            # so we need to exclude them from the comparison
            if hasattr(new_field, "remote_field") and getattr(
                new_field.remote_field, "model", None
            ):
                rename_key = resolve_relation(
                    new_field.remote_field.model, app_label, model_name
                )
                if rename_key in self.renamed_models:
                    new_field.remote_field.model = old_field.remote_field.model
                # Handle ForeignKey which can only have a single to_field.
                remote_field_name = getattr(new_field.remote_field, "field_name", None)
                if remote_field_name:
                    to_field_rename_key = rename_key + (remote_field_name,)
                    if to_field_rename_key in self.renamed_fields:
                        # Repoint both model and field name because to_field
                        # inclusion in ForeignKey.deconstruct() is based on
                        # both.
                        new_field.remote_field.model = old_field.remote_field.model
                        new_field.remote_field.field_name = (
                            old_field.remote_field.field_name
                        )
                # Handle ForeignObjects which can have multiple from_fields/to_fields.
                from_fields = getattr(new_field, "from_fields", None)
                if from_fields:
                    from_rename_key = (app_label, model_name)
                    new_field.from_fields = tuple(
                        [
                            self.renamed_fields.get(
                                from_rename_key + (from_field,), from_field
                            )
                            for from_field in from_fields
                        ]
                    )
                    new_field.to_fields = tuple(
                        [
                            self.renamed_fields.get(rename_key + (to_field,), to_field)
                            for to_field in new_field.to_fields
                        ]
                    )
                dependencies.extend(
                    self._get_dependencies_for_foreign_key(
                        app_label,
                        model_name,
                        new_field,
                        self.to_state,
                    )
                )
            if hasattr(new_field, "remote_field") and getattr(
                new_field.remote_field, "through", None
            ):
                rename_key = resolve_relation(
                    new_field.remote_field.through, app_label, model_name
                )
                if rename_key in self.renamed_models:
                    new_field.remote_field.through = old_field.remote_field.through
            old_field_dec = self.deep_deconstruct(old_field)
            new_field_dec = self.deep_deconstruct(new_field)
            # If the field was confirmed to be renamed it means that only
            # db_column was allowed to change which generate_renamed_fields()
            # already accounts for by adding an AlterField operation.
            if old_field_dec != new_field_dec and old_field_name == field_name:
                both_m2m = old_field.many_to_many and new_field.many_to_many
                neither_m2m = not old_field.many_to_many and not new_field.many_to_many
                if both_m2m or neither_m2m:
                    # Either both fields are m2m or neither is
                    preserve_default = True
                    if (
                        old_field.null
                        and not new_field.null
                        and not new_field.has_default()
                        and not new_field.many_to_many
                    ):
                        field = new_field.clone()
                        new_default = self.questioner.ask_not_null_alteration(
                            field_name, model_name
                        )
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
### 65 - django/db/migrations/autodetector.py:

Start line: 1341, End line: 1362

```python
class MigrationAutodetector:

    def create_altered_constraints(self):
        option_name = operations.AddConstraint.option_name
        for app_label, model_name in sorted(self.kept_model_keys):
            old_model_name = self.renamed_models.get(
                (app_label, model_name), model_name
            )
            old_model_state = self.from_state.models[app_label, old_model_name]
            new_model_state = self.to_state.models[app_label, model_name]

            old_constraints = old_model_state.options[option_name]
            new_constraints = new_model_state.options[option_name]
            add_constraints = [c for c in new_constraints if c not in old_constraints]
            rem_constraints = [c for c in old_constraints if c not in new_constraints]

            self.altered_constraints.update(
                {
                    (app_label, model_name): {
                        "added_constraints": add_constraints,
                        "removed_constraints": rem_constraints,
                    }
                }
            )
```
### 67 - django/db/migrations/autodetector.py:

Start line: 1588, End line: 1620

```python
class MigrationAutodetector:

    def generate_altered_order_with_respect_to(self):
        for app_label, model_name in sorted(self.kept_model_keys):
            old_model_name = self.renamed_models.get(
                (app_label, model_name), model_name
            )
            old_model_state = self.from_state.models[app_label, old_model_name]
            new_model_state = self.to_state.models[app_label, model_name]
            if old_model_state.options.get(
                "order_with_respect_to"
            ) != new_model_state.options.get("order_with_respect_to"):
                # Make sure it comes second if we're adding
                # (removal dependency is part of RemoveField)
                dependencies = []
                if new_model_state.options.get("order_with_respect_to"):
                    dependencies.append(
                        (
                            app_label,
                            model_name,
                            new_model_state.options["order_with_respect_to"],
                            True,
                        )
                    )
                # Actually generate the operation
                self.add_operation(
                    app_label,
                    operations.AlterOrderWithRespectTo(
                        name=model_name,
                        order_with_respect_to=new_model_state.options.get(
                            "order_with_respect_to"
                        ),
                    ),
                    dependencies=dependencies,
                )
```
### 86 - django/db/migrations/autodetector.py:

Start line: 1023, End line: 1071

```python
class MigrationAutodetector:

    def _generate_added_field(self, app_label, model_name, field_name):
        field = self.to_state.models[app_label, model_name].get_field(field_name)
        # Fields that are foreignkeys/m2ms depend on stuff
        dependencies = []
        if field.remote_field and field.remote_field.model:
            dependencies.extend(
                self._get_dependencies_for_foreign_key(
                    app_label,
                    model_name,
                    field,
                    self.to_state,
                )
            )
        # You can't just add NOT NULL fields with no default or fields
        # which don't allow empty strings as default.
        time_fields = (models.DateField, models.DateTimeField, models.TimeField)
        preserve_default = (
            field.null
            or field.has_default()
            or field.many_to_many
            or (field.blank and field.empty_strings_allowed)
            or (isinstance(field, time_fields) and field.auto_now)
        )
        if not preserve_default:
            field = field.clone()
            if isinstance(field, time_fields) and field.auto_now_add:
                field.default = self.questioner.ask_auto_now_add_addition(
                    field_name, model_name
                )
            else:
                field.default = self.questioner.ask_not_null_addition(
                    field_name, model_name
                )
        if (
            field.unique
            and field.default is not models.NOT_PROVIDED
            and callable(field.default)
        ):
            self.questioner.ask_unique_callable_default_addition(field_name, model_name)
        self.add_operation(
            app_label,
            operations.AddField(
                model_name=model_name,
                name=field_name,
                field=field,
                preserve_default=preserve_default,
            ),
            dependencies=dependencies,
        )
```
### 90 - django/db/migrations/autodetector.py:

Start line: 1215, End line: 1302

```python
class MigrationAutodetector:

    def create_altered_indexes(self):
        option_name = operations.AddIndex.option_name
        self.renamed_index_together_values = defaultdict(list)

        for app_label, model_name in sorted(self.kept_model_keys):
            old_model_name = self.renamed_models.get(
                (app_label, model_name), model_name
            )
            old_model_state = self.from_state.models[app_label, old_model_name]
            new_model_state = self.to_state.models[app_label, model_name]

            old_indexes = old_model_state.options[option_name]
            new_indexes = new_model_state.options[option_name]
            added_indexes = [idx for idx in new_indexes if idx not in old_indexes]
            removed_indexes = [idx for idx in old_indexes if idx not in new_indexes]
            renamed_indexes = []
            # Find renamed indexes.
            remove_from_added = []
            remove_from_removed = []
            for new_index in added_indexes:
                new_index_dec = new_index.deconstruct()
                new_index_name = new_index_dec[2].pop("name")
                for old_index in removed_indexes:
                    old_index_dec = old_index.deconstruct()
                    old_index_name = old_index_dec[2].pop("name")
                    # Indexes are the same except for the names.
                    if (
                        new_index_dec == old_index_dec
                        and new_index_name != old_index_name
                    ):
                        renamed_indexes.append((old_index_name, new_index_name, None))
                        remove_from_added.append(new_index)
                        remove_from_removed.append(old_index)
            # Find index_together changed to indexes.
            for (
                old_value,
                new_value,
                index_together_app_label,
                index_together_model_name,
                dependencies,
            ) in self._get_altered_foo_together_operations(
                operations.AlterIndexTogether.option_name
            ):
                if (
                    app_label != index_together_app_label
                    or model_name != index_together_model_name
                ):
                    continue
                removed_values = old_value.difference(new_value)
                for removed_index_together in removed_values:
                    renamed_index_together_indexes = []
                    for new_index in added_indexes:
                        _, args, kwargs = new_index.deconstruct()
                        # Ensure only 'fields' are defined in the Index.
                        if (
                            not args
                            and new_index.fields == list(removed_index_together)
                            and set(kwargs) == {"name", "fields"}
                        ):
                            renamed_index_together_indexes.append(new_index)

                    if len(renamed_index_together_indexes) == 1:
                        renamed_index = renamed_index_together_indexes[0]
                        remove_from_added.append(renamed_index)
                        renamed_indexes.append(
                            (None, renamed_index.name, removed_index_together)
                        )
                        self.renamed_index_together_values[
                            index_together_app_label, index_together_model_name
                        ].append(removed_index_together)
            # Remove renamed indexes from the lists of added and removed
            # indexes.
            added_indexes = [
                idx for idx in added_indexes if idx not in remove_from_added
            ]
            removed_indexes = [
                idx for idx in removed_indexes if idx not in remove_from_removed
            ]

            self.altered_indexes.update(
                {
                    (app_label, model_name): {
                        "added_indexes": added_indexes,
                        "removed_indexes": removed_indexes,
                        "renamed_indexes": renamed_indexes,
                    }
                }
            )
```
### 98 - django/db/migrations/autodetector.py:

Start line: 583, End line: 599

```python
class MigrationAutodetector:

    def generate_created_models(self):
        """
        Find all new models (both managed and unmanaged) and make create
        operations for them as well as separate operations to create any
        foreign key or M2M relationships (these are optimized later, if
        possible).

        Defer any model options that refer to collections of fields that might
        be deferred (e.g. unique_together, index_together).
        """
        old_keys = self.old_model_keys | self.old_unmanaged_keys
        added_models = self.new_model_keys - old_keys
        added_unmanaged_models = self.new_unmanaged_keys - old_keys
        all_added_models = chain(
            sorted(added_models, key=self.swappable_first_key, reverse=True),
            sorted(added_unmanaged_models, key=self.swappable_first_key, reverse=True),
        )
        # ... other code
```
### 100 - django/db/migrations/autodetector.py:

Start line: 904, End line: 977

```python
class MigrationAutodetector:

    def generate_deleted_proxies(self):
        """Make DeleteModel options for proxy models."""
        deleted = self.old_proxy_keys - self.new_proxy_keys
        for app_label, model_name in sorted(deleted):
            model_state = self.from_state.models[app_label, model_name]
            assert model_state.options.get("proxy")
            self.add_operation(
                app_label,
                operations.DeleteModel(
                    name=model_state.name,
                ),
            )

    def create_renamed_fields(self):
        """Work out renamed fields."""
        self.renamed_operations = []
        old_field_keys = self.old_field_keys.copy()
        for app_label, model_name, field_name in sorted(
            self.new_field_keys - old_field_keys
        ):
            old_model_name = self.renamed_models.get(
                (app_label, model_name), model_name
            )
            old_model_state = self.from_state.models[app_label, old_model_name]
            new_model_state = self.to_state.models[app_label, model_name]
            field = new_model_state.get_field(field_name)
            # Scan to see if this is actually a rename!
            field_dec = self.deep_deconstruct(field)
            for rem_app_label, rem_model_name, rem_field_name in sorted(
                old_field_keys - self.new_field_keys
            ):
                if rem_app_label == app_label and rem_model_name == model_name:
                    old_field = old_model_state.get_field(rem_field_name)
                    old_field_dec = self.deep_deconstruct(old_field)
                    if (
                        field.remote_field
                        and field.remote_field.model
                        and "to" in old_field_dec[2]
                    ):
                        old_rel_to = old_field_dec[2]["to"]
                        if old_rel_to in self.renamed_models_rel:
                            old_field_dec[2]["to"] = self.renamed_models_rel[old_rel_to]
                    old_field.set_attributes_from_name(rem_field_name)
                    old_db_column = old_field.get_attname_column()[1]
                    if old_field_dec == field_dec or (
                        # Was the field renamed and db_column equal to the
                        # old field's column added?
                        old_field_dec[0:2] == field_dec[0:2]
                        and dict(old_field_dec[2], db_column=old_db_column)
                        == field_dec[2]
                    ):
                        if self.questioner.ask_rename(
                            model_name, rem_field_name, field_name, field
                        ):
                            self.renamed_operations.append(
                                (
                                    rem_app_label,
                                    rem_model_name,
                                    old_field.db_column,
                                    rem_field_name,
                                    app_label,
                                    model_name,
                                    field,
                                    field_name,
                                )
                            )
                            old_field_keys.remove(
                                (rem_app_label, rem_model_name, rem_field_name)
                            )
                            old_field_keys.add((app_label, model_name, field_name))
                            self.renamed_fields[
                                app_label, model_name, field_name
                            ] = rem_field_name
                            break
```
### 115 - django/db/migrations/autodetector.py:

Start line: 516, End line: 581

```python
class MigrationAutodetector:

    def generate_renamed_models(self):
        """
        Find any renamed models, generate the operations for them, and remove
        the old entry from the model lists. Must be run before other
        model-level generation.
        """
        self.renamed_models = {}
        self.renamed_models_rel = {}
        added_models = self.new_model_keys - self.old_model_keys
        for app_label, model_name in sorted(added_models):
            model_state = self.to_state.models[app_label, model_name]
            model_fields_def = self.only_relation_agnostic_fields(model_state.fields)

            removed_models = self.old_model_keys - self.new_model_keys
            for rem_app_label, rem_model_name in removed_models:
                if rem_app_label == app_label:
                    rem_model_state = self.from_state.models[
                        rem_app_label, rem_model_name
                    ]
                    rem_model_fields_def = self.only_relation_agnostic_fields(
                        rem_model_state.fields
                    )
                    if model_fields_def == rem_model_fields_def:
                        if self.questioner.ask_rename_model(
                            rem_model_state, model_state
                        ):
                            dependencies = []
                            fields = list(model_state.fields.values()) + [
                                field.remote_field
                                for relations in self.to_state.relations[
                                    app_label, model_name
                                ].values()
                                for field in relations.values()
                            ]
                            for field in fields:
                                if field.is_relation:
                                    dependencies.extend(
                                        self._get_dependencies_for_foreign_key(
                                            app_label,
                                            model_name,
                                            field,
                                            self.to_state,
                                        )
                                    )
                            self.add_operation(
                                app_label,
                                operations.RenameModel(
                                    old_name=rem_model_state.name,
                                    new_name=model_state.name,
                                ),
                                dependencies=dependencies,
                            )
                            self.renamed_models[app_label, model_name] = rem_model_name
                            renamed_models_rel_key = "%s.%s" % (
                                rem_model_state.app_label,
                                rem_model_state.name_lower,
                            )
                            self.renamed_models_rel[
                                renamed_models_rel_key
                            ] = "%s.%s" % (
                                model_state.app_label,
                                model_state.name_lower,
                            )
                            self.old_model_keys.remove((rem_app_label, rem_model_name))
                            self.old_model_keys.add((app_label, model_name))
                            break
```
### 123 - django/db/migrations/autodetector.py:

Start line: 1474, End line: 1497

```python
class MigrationAutodetector:

    def _generate_removed_altered_foo_together(self, operation):
        for (
            old_value,
            new_value,
            app_label,
            model_name,
            dependencies,
        ) in self._get_altered_foo_together_operations(operation.option_name):
            if operation == operations.AlterIndexTogether:
                old_value = {
                    value
                    for value in old_value
                    if value
                    not in self.renamed_index_together_values[app_label, model_name]
                }
            removal_value = new_value.intersection(old_value)
            if removal_value or old_value:
                self.add_operation(
                    app_label,
                    operation(
                        name=model_name, **{operation.option_name: removal_value}
                    ),
                    dependencies=dependencies,
                )
```
### 129 - django/db/migrations/autodetector.py:

Start line: 210, End line: 231

```python
class MigrationAutodetector:

    def _prepare_field_lists(self):
        """
        Prepare field lists and a list of the fields that used through models
        in the old state so dependencies can be made from the through model
        deletion to the field that uses it.
        """
        self.kept_model_keys = self.old_model_keys & self.new_model_keys
        self.kept_proxy_keys = self.old_proxy_keys & self.new_proxy_keys
        self.kept_unmanaged_keys = self.old_unmanaged_keys & self.new_unmanaged_keys
        self.through_users = {}
        self.old_field_keys = {
            (app_label, model_name, field_name)
            for app_label, model_name in self.kept_model_keys
            for field_name in self.from_state.models[
                app_label, self.renamed_models.get((app_label, model_name), model_name)
            ].fields
        }
        self.new_field_keys = {
            (app_label, model_name, field_name)
            for app_label, model_name in self.kept_model_keys
            for field_name in self.to_state.models[app_label, model_name].fields
        }
```
### 133 - django/db/migrations/autodetector.py:

Start line: 21, End line: 38

```python
class MigrationAutodetector:
    """
    Take a pair of ProjectStates and compare them to see what the first would
    need doing to make it match the second (the second usually being the
    project's current state).

    Note that this naturally operates on entire projects at a time,
    as it's likely that changes interact (for example, you can't
    add a ForeignKey without having a migration to add the table it
    depends on first). A user interface may offer single-app usage
    if it wishes, with the caveat that it may not always be possible.
    """

    def __init__(self, from_state, to_state, questioner=None):
        self.from_state = from_state
        self.to_state = to_state
        self.questioner = questioner or MigrationQuestioner()
        self.existing_apps = {app for app, model in from_state.models}
```
### 141 - django/db/migrations/autodetector.py:

Start line: 1364, End line: 1390

```python
class MigrationAutodetector:

    def generate_added_constraints(self):
        for (
            app_label,
            model_name,
        ), alt_constraints in self.altered_constraints.items():
            for constraint in alt_constraints["added_constraints"]:
                self.add_operation(
                    app_label,
                    operations.AddConstraint(
                        model_name=model_name,
                        constraint=constraint,
                    ),
                )

    def generate_removed_constraints(self):
        for (
            app_label,
            model_name,
        ), alt_constraints in self.altered_constraints.items():
            for constraint in alt_constraints["removed_constraints"]:
                self.add_operation(
                    app_label,
                    operations.RemoveConstraint(
                        model_name=model_name,
                        name=constraint.name,
                    ),
                )
```
### 152 - django/db/migrations/autodetector.py:

Start line: 1622, End line: 1636

```python
class MigrationAutodetector:

    def generate_altered_managers(self):
        for app_label, model_name in sorted(self.kept_model_keys):
            old_model_name = self.renamed_models.get(
                (app_label, model_name), model_name
            )
            old_model_state = self.from_state.models[app_label, old_model_name]
            new_model_state = self.to_state.models[app_label, model_name]
            if old_model_state.managers != new_model_state.managers:
                self.add_operation(
                    app_label,
                    operations.AlterModelManagers(
                        name=model_name,
                        managers=new_model_state.managers,
                    ),
                )
```
### 167 - django/db/migrations/autodetector.py:

Start line: 1527, End line: 1546

```python
class MigrationAutodetector:

    def generate_altered_db_table(self):
        models_to_check = self.kept_model_keys.union(
            self.kept_proxy_keys, self.kept_unmanaged_keys
        )
        for app_label, model_name in sorted(models_to_check):
            old_model_name = self.renamed_models.get(
                (app_label, model_name), model_name
            )
            old_model_state = self.from_state.models[app_label, old_model_name]
            new_model_state = self.to_state.models[app_label, model_name]
            old_db_table_name = old_model_state.options.get("db_table")
            new_db_table_name = new_model_state.options.get("db_table")
            if old_db_table_name != new_db_table_name:
                self.add_operation(
                    app_label,
                    operations.AlterModelTable(
                        name=model_name,
                        table=new_db_table_name,
                    ),
                )
```
### 183 - django/db/migrations/autodetector.py:

Start line: 1073, End line: 1094

```python
class MigrationAutodetector:

    def generate_removed_fields(self):
        """Make RemoveField operations."""
        for app_label, model_name, field_name in sorted(
            self.old_field_keys - self.new_field_keys
        ):
            self._generate_removed_field(app_label, model_name, field_name)

    def _generate_removed_field(self, app_label, model_name, field_name):
        self.add_operation(
            app_label,
            operations.RemoveField(
                model_name=model_name,
                name=field_name,
            ),
            # We might need to depend on the removal of an
            # order_with_respect_to or index/unique_together operation;
            # this is safely ignored if there isn't one
            dependencies=[
                (app_label, model_name, field_name, "order_wrt_unset"),
                (app_label, model_name, field_name, "foo_together_change"),
            ],
        )
```
