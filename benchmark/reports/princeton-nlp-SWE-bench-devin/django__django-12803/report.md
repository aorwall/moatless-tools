# django__django-12803

| **django/django** | `35f89d199c94ebc72b06d5c44077401aa2eae47f` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 346 |
| **Any found context length** | 346 |
| **Avg pos** | 1.0 |
| **Min pos** | 1 |
| **Max pos** | 1 |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/contrib/staticfiles/storage.py b/django/contrib/staticfiles/storage.py
--- a/django/contrib/staticfiles/storage.py
+++ b/django/contrib/staticfiles/storage.py
@@ -98,8 +98,7 @@ def hashed_name(self, name, content=None, filename=None):
                 content.close()
         path, filename = os.path.split(clean_name)
         root, ext = os.path.splitext(filename)
-        if file_hash is not None:
-            file_hash = ".%s" % file_hash
+        file_hash = ('.%s' % file_hash) if file_hash else ''
         hashed_name = os.path.join(path, "%s%s%s" %
                                    (root, file_hash, ext))
         unparsed_name = list(parsed_name)

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/contrib/staticfiles/storage.py | 101 | 102 | 1 | 1 | 346


## Problem Statement

```
ManifestFilesMixin.file_hash() returning None get's included in hashed filename as 'None'.
Description
	 
		(last modified by Mariusz Felisiak)
	 
When returning a string from a custom ManifestFilesMixin.file_hash() implementation, the resulting file name is <file_path>.<custom_hash>.<ext> as expected, whereas returning None results in <file_path>None.<ext>.
​Discussion on django-developers supports this behaviour being unintended.
Behavior appears to have been introduced with #17896 which split the file hashing into a separate method.
The following test, when included in the test_storage.TestCollectionManifestStorage test class demonstrates the bug:
def test_hashed_name_unchanged_when_file_hash_is_None(self):
	with mock.patch('django.contrib.staticfiles.storage.ManifestStaticFilesStorage.file_hash', return_value=None):
		self.assertEqual(storage.staticfiles_storage.hashed_name('test/file.txt'), 'test/file.txt')
As suggested by the name of my test, my opinion is that the correct behaviour should be that if file_hash returns None, then no hash is inserted into the filename and it therefore remains unchanged.
With that in mind, a possible solution is to change the following lines in the hashed_name() method (~line 100 in staticfiles.storage):
if file_hash is not None:
	file_hash = ".%s" % file_hash
hashed_name = os.path.join(path, "%s%s%s" % (root, file_hash, ext))
to
if file_hash is None:
	file_hash = ""
else:
	file_hash = ".%s" % file_hash
hashed_name = os.path.join(path, "%s%s%s" % (root, file_hash, ext))

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 django/contrib/staticfiles/storage.py** | 79 | 111| 346 | 346 | 3530 | 
| 2 | **1 django/contrib/staticfiles/storage.py** | 342 | 363| 230 | 576 | 3530 | 
| 3 | **1 django/contrib/staticfiles/storage.py** | 324 | 340| 147 | 723 | 3530 | 
| 4 | **1 django/contrib/staticfiles/storage.py** | 366 | 409| 328 | 1051 | 3530 | 
| 5 | **1 django/contrib/staticfiles/storage.py** | 411 | 443| 275 | 1326 | 3530 | 
| 6 | **1 django/contrib/staticfiles/storage.py** | 252 | 322| 575 | 1901 | 3530 | 
| 7 | **1 django/contrib/staticfiles/storage.py** | 44 | 77| 275 | 2176 | 3530 | 
| 8 | **1 django/contrib/staticfiles/storage.py** | 150 | 202| 425 | 2601 | 3530 | 
| 9 | 2 django/core/files/uploadedfile.py | 38 | 52| 134 | 2735 | 4381 | 
| 10 | 3 django/core/files/storage.py | 54 | 69| 138 | 2873 | 7252 | 
| 11 | 3 django/core/files/storage.py | 233 | 294| 526 | 3399 | 7252 | 
| 12 | **3 django/contrib/staticfiles/storage.py** | 113 | 148| 307 | 3706 | 7252 | 
| 13 | 4 django/db/models/fields/files.py | 160 | 219| 645 | 4351 | 11063 | 
| 14 | 4 django/db/models/fields/files.py | 222 | 341| 931 | 5282 | 11063 | 
| 15 | 5 django/contrib/auth/hashers.py | 85 | 107| 167 | 5449 | 15919 | 
| 16 | 6 django/core/files/base.py | 1 | 29| 174 | 5623 | 16971 | 
| 17 | 6 django/db/models/fields/files.py | 1 | 140| 930 | 6553 | 16971 | 
| 18 | 6 django/core/files/storage.py | 205 | 231| 215 | 6768 | 16971 | 
| 19 | 7 django/contrib/messages/storage/cookie.py | 126 | 149| 234 | 7002 | 18452 | 
| 20 | 7 django/core/files/storage.py | 296 | 368| 483 | 7485 | 18452 | 
| 21 | 7 django/contrib/auth/hashers.py | 570 | 603| 242 | 7727 | 18452 | 
| 22 | 7 django/core/files/storage.py | 71 | 99| 300 | 8027 | 18452 | 
| 23 | 7 django/contrib/auth/hashers.py | 64 | 82| 186 | 8213 | 18452 | 
| 24 | 7 django/contrib/auth/hashers.py | 412 | 422| 127 | 8340 | 18452 | 
| 25 | **7 django/contrib/staticfiles/storage.py** | 1 | 41| 307 | 8647 | 18452 | 
| 26 | 7 django/core/files/storage.py | 101 | 173| 610 | 9257 | 18452 | 
| 27 | 7 django/contrib/auth/hashers.py | 535 | 567| 220 | 9477 | 18452 | 
| 28 | 8 django/utils/cache.py | 215 | 239| 235 | 9712 | 22200 | 
| 29 | 8 django/contrib/auth/hashers.py | 1 | 27| 187 | 9899 | 22200 | 
| 30 | 8 django/contrib/auth/hashers.py | 504 | 532| 222 | 10121 | 22200 | 
| 31 | 8 django/contrib/auth/hashers.py | 424 | 453| 290 | 10411 | 22200 | 
| 32 | 8 django/core/files/storage.py | 1 | 22| 158 | 10569 | 22200 | 
| 33 | 8 django/core/files/storage.py | 176 | 191| 151 | 10720 | 22200 | 
| 34 | 8 django/contrib/auth/hashers.py | 473 | 501| 220 | 10940 | 22200 | 
| 35 | 9 django/forms/fields.py | 579 | 604| 243 | 11183 | 31213 | 
| 36 | 9 django/contrib/auth/hashers.py | 133 | 151| 189 | 11372 | 31213 | 
| 37 | **9 django/contrib/staticfiles/storage.py** | 204 | 250| 364 | 11736 | 31213 | 
| 38 | 10 django/contrib/auth/forms.py | 53 | 78| 162 | 11898 | 34422 | 
| 39 | 10 django/core/files/base.py | 75 | 118| 303 | 12201 | 34422 | 
| 40 | 10 django/contrib/auth/hashers.py | 606 | 643| 276 | 12477 | 34422 | 
| 41 | 11 django/core/files/__init__.py | 1 | 4| 0 | 12477 | 34437 | 
| 42 | 11 django/contrib/auth/hashers.py | 154 | 189| 245 | 12722 | 34437 | 
| 43 | 12 django/core/management/commands/makemessages.py | 37 | 58| 143 | 12865 | 39987 | 
| 44 | 13 django/db/models/fields/__init__.py | 1636 | 1650| 144 | 13009 | 57643 | 
| 45 | 13 django/core/files/uploadedfile.py | 99 | 118| 148 | 13157 | 57643 | 
| 46 | 13 django/forms/fields.py | 1089 | 1130| 353 | 13510 | 57643 | 
| 47 | 13 django/core/files/storage.py | 25 | 52| 211 | 13721 | 57643 | 
| 48 | 13 django/contrib/auth/hashers.py | 110 | 130| 139 | 13860 | 57643 | 
| 49 | 14 django/core/management/commands/diffsettings.py | 57 | 67| 128 | 13988 | 58334 | 
| 50 | 14 django/forms/fields.py | 1184 | 1214| 182 | 14170 | 58334 | 
| 51 | 15 django/contrib/staticfiles/testing.py | 1 | 14| 0 | 14170 | 58427 | 
| 52 | 16 django/contrib/auth/migrations/0002_alter_permission_name_max_length.py | 1 | 17| 0 | 14170 | 58495 | 
| 53 | 17 django/forms/widgets.py | 374 | 394| 138 | 14308 | 66572 | 
| 54 | 18 django/contrib/auth/migrations/0010_alter_group_name_max_length.py | 1 | 17| 0 | 14308 | 66649 | 
| 55 | 18 django/forms/fields.py | 558 | 577| 171 | 14479 | 66649 | 
| 56 | 18 django/db/models/fields/files.py | 374 | 420| 373 | 14852 | 66649 | 
| 57 | 19 django/db/models/fields/mixins.py | 31 | 57| 173 | 15025 | 66992 | 
| 58 | 20 django/contrib/staticfiles/finders.py | 243 | 255| 109 | 15134 | 69033 | 
| 59 | 21 django/template/defaultfilters.py | 806 | 850| 378 | 15512 | 75121 | 
| 60 | 21 django/db/models/fields/__init__.py | 1609 | 1634| 206 | 15718 | 75121 | 
| 61 | 22 django/core/cache/backends/filebased.py | 1 | 44| 284 | 16002 | 76294 | 
| 62 | 22 django/contrib/auth/hashers.py | 191 | 219| 237 | 16239 | 76294 | 
| 63 | 23 django/contrib/sessions/backends/file.py | 111 | 170| 530 | 16769 | 77795 | 
| 64 | 24 django/db/models/functions/text.py | 24 | 55| 217 | 16986 | 80250 | 
| 65 | 25 django/core/mail/backends/filebased.py | 1 | 65| 503 | 17489 | 80754 | 
| 66 | 25 django/contrib/auth/hashers.py | 334 | 347| 120 | 17609 | 80754 | 
| 67 | 26 django/core/files/utils.py | 1 | 53| 378 | 17987 | 81132 | 
| 68 | 26 django/contrib/auth/hashers.py | 349 | 364| 146 | 18133 | 81132 | 
| 69 | 27 django/core/mail/message.py | 169 | 177| 115 | 18248 | 84670 | 
| 70 | 28 django/core/files/images.py | 1 | 30| 147 | 18395 | 85209 | 
| 71 | 29 django/core/mail/backends/dummy.py | 1 | 11| 0 | 18395 | 85252 | 
| 72 | 29 django/contrib/auth/hashers.py | 235 | 279| 414 | 18809 | 85252 | 
| 73 | 30 django/http/multipartparser.py | 289 | 310| 213 | 19022 | 90279 | 
| 74 | 30 django/core/files/uploadedfile.py | 78 | 96| 149 | 19171 | 90279 | 
| 75 | 30 django/db/models/fields/__init__.py | 1652 | 1670| 134 | 19305 | 90279 | 
| 76 | 30 django/contrib/auth/hashers.py | 282 | 332| 391 | 19696 | 90279 | 
| 77 | 31 django/core/files/uploadhandler.py | 1 | 24| 126 | 19822 | 91609 | 
| 78 | 31 django/contrib/staticfiles/finders.py | 70 | 93| 202 | 20024 | 91609 | 
| 79 | 32 django/db/migrations/questioner.py | 56 | 81| 220 | 20244 | 93682 | 
| 80 | 32 django/core/files/storage.py | 193 | 203| 130 | 20374 | 93682 | 
| 81 | 32 django/core/files/base.py | 31 | 46| 129 | 20503 | 93682 | 
| 82 | 33 django/views/static.py | 108 | 136| 206 | 20709 | 94734 | 
| 83 | 34 django/db/backends/sqlite3/creation.py | 1 | 21| 140 | 20849 | 95585 | 
| 84 | 34 django/db/models/fields/__init__.py | 2012 | 2042| 252 | 21101 | 95585 | 
| 85 | 34 django/db/migrations/questioner.py | 162 | 185| 246 | 21347 | 95585 | 
| 86 | 34 django/contrib/auth/hashers.py | 456 | 470| 126 | 21473 | 95585 | 
| 87 | 35 django/contrib/auth/migrations/0012_alter_user_first_name_max_length.py | 1 | 17| 0 | 21473 | 95660 | 
| 88 | 36 django/db/models/fields/related.py | 127 | 154| 201 | 21674 | 109491 | 
| 89 | 36 django/core/files/base.py | 121 | 145| 154 | 21828 | 109491 | 
| 90 | 36 django/db/models/fields/mixins.py | 1 | 28| 168 | 21996 | 109491 | 
| 91 | 36 django/core/cache/backends/filebased.py | 61 | 96| 260 | 22256 | 109491 | 
| 92 | 37 django/db/migrations/serializer.py | 163 | 179| 151 | 22407 | 112040 | 
| 93 | 37 django/db/models/fields/__init__.py | 2233 | 2294| 425 | 22832 | 112040 | 
| 94 | 38 django/forms/forms.py | 429 | 451| 195 | 23027 | 116054 | 
| 95 | 38 django/db/models/fields/__init__.py | 1926 | 1953| 234 | 23261 | 116054 | 
| 96 | 38 django/db/models/fields/__init__.py | 2297 | 2347| 339 | 23600 | 116054 | 
| 97 | 38 django/contrib/auth/forms.py | 141 | 169| 235 | 23835 | 116054 | 
| 98 | 39 django/contrib/staticfiles/__init__.py | 1 | 2| 0 | 23835 | 116068 | 
| 99 | 40 django/db/backends/dummy/features.py | 1 | 7| 0 | 23835 | 116100 | 
| 100 | 40 django/core/cache/backends/filebased.py | 116 | 159| 337 | 24172 | 116100 | 
| 101 | 41 django/contrib/postgres/indexes.py | 188 | 205| 131 | 24303 | 117889 | 
| 102 | 41 django/db/models/fields/related.py | 860 | 886| 240 | 24543 | 117889 | 
| 103 | 41 django/core/cache/backends/filebased.py | 46 | 59| 145 | 24688 | 117889 | 
| 104 | 41 django/contrib/auth/forms.py | 32 | 50| 161 | 24849 | 117889 | 
| 105 | 41 django/core/management/commands/makemessages.py | 83 | 96| 118 | 24967 | 117889 | 
| 106 | 42 django/contrib/auth/migrations/0009_alter_user_last_name_max_length.py | 1 | 17| 0 | 24967 | 117967 | 
| 107 | 43 django/db/migrations/state.py | 591 | 607| 146 | 25113 | 123089 | 
| 108 | 43 django/forms/fields.py | 540 | 556| 180 | 25293 | 123089 | 
| 109 | 44 django/core/files/temp.py | 1 | 75| 517 | 25810 | 123607 | 
| 110 | 44 django/db/models/fields/files.py | 143 | 158| 112 | 25922 | 123607 | 
| 111 | 45 django/forms/boundfield.py | 1 | 34| 242 | 26164 | 125763 | 
| 112 | 45 django/db/migrations/state.py | 348 | 395| 428 | 26592 | 125763 | 
| 113 | 46 django/utils/crypto.py | 1 | 46| 369 | 26961 | 126475 | 
| 114 | 47 django/core/cache/utils.py | 1 | 13| 0 | 26961 | 126554 | 
| 115 | 48 django/utils/hashable.py | 1 | 20| 122 | 27083 | 126677 | 
| 116 | 48 django/utils/cache.py | 136 | 151| 188 | 27271 | 126677 | 
| 117 | 49 django/contrib/staticfiles/management/commands/collectstatic.py | 147 | 206| 501 | 27772 | 129519 | 
| 118 | 50 django/conf/global_settings.py | 263 | 345| 800 | 28572 | 135159 | 
| 119 | 50 django/core/files/uploadedfile.py | 1 | 36| 241 | 28813 | 135159 | 
| 120 | 51 django/core/cache/backends/memcached.py | 110 | 126| 168 | 28981 | 136976 | 
| 121 | 52 django/db/models/options.py | 332 | 355| 198 | 29179 | 144082 | 
| 122 | 52 django/db/models/fields/__init__.py | 2350 | 2399| 311 | 29490 | 144082 | 
| 123 | 52 django/db/models/fields/__init__.py | 1973 | 2009| 198 | 29688 | 144082 | 
| 124 | 52 django/contrib/auth/hashers.py | 30 | 61| 246 | 29934 | 144082 | 
| 125 | 53 django/db/models/lookups.py | 591 | 603| 124 | 30058 | 149008 | 
| 126 | 54 django/bin/django-admin.py | 1 | 22| 138 | 30196 | 149146 | 
| 127 | 54 django/db/models/fields/related.py | 255 | 282| 269 | 30465 | 149146 | 
| 128 | 55 django/contrib/sites/migrations/0002_alter_domain_unique.py | 1 | 21| 0 | 30465 | 149243 | 
| 129 | 55 django/db/models/options.py | 256 | 288| 331 | 30796 | 149243 | 
| 130 | 55 django/db/models/fields/__init__.py | 1 | 81| 633 | 31429 | 149243 | 
| 131 | 55 django/db/migrations/state.py | 576 | 589| 138 | 31567 | 149243 | 
| 132 | 56 django/contrib/auth/base_user.py | 47 | 140| 585 | 32152 | 150127 | 
| 133 | 57 django/contrib/auth/mixins.py | 88 | 110| 146 | 32298 | 150857 | 
| 134 | 57 django/contrib/auth/hashers.py | 221 | 232| 150 | 32448 | 150857 | 
| 135 | 58 django/contrib/humanize/__init__.py | 1 | 2| 0 | 32448 | 150873 | 
| 136 | 59 django/forms/__init__.py | 1 | 12| 0 | 32448 | 150963 | 
| 137 | 59 django/contrib/auth/hashers.py | 394 | 410| 127 | 32575 | 150963 | 
| 138 | 60 django/contrib/auth/migrations/0006_require_contenttypes_0002.py | 1 | 15| 0 | 32575 | 151039 | 
| 139 | 60 django/contrib/staticfiles/finders.py | 198 | 240| 282 | 32857 | 151039 | 
| 140 | 60 django/core/management/commands/makemessages.py | 61 | 81| 146 | 33003 | 151039 | 
| 141 | 60 django/forms/widgets.py | 443 | 461| 195 | 33198 | 151039 | 
| 142 | 61 django/middleware/common.py | 76 | 97| 227 | 33425 | 152550 | 
| 143 | 61 django/db/migrations/questioner.py | 187 | 205| 237 | 33662 | 152550 | 
| 144 | 61 django/db/models/fields/related.py | 108 | 125| 155 | 33817 | 152550 | 
| 145 | 62 django/contrib/contenttypes/fields.py | 160 | 171| 123 | 33940 | 157983 | 
| 146 | 63 django/contrib/messages/storage/__init__.py | 1 | 13| 0 | 33940 | 158053 | 
| 147 | 63 django/utils/crypto.py | 49 | 74| 212 | 34152 | 158053 | 
| 148 | 63 django/forms/fields.py | 173 | 205| 280 | 34432 | 158053 | 
| 149 | 63 django/core/files/uploadedfile.py | 55 | 75| 181 | 34613 | 158053 | 
| 150 | 63 django/utils/cache.py | 194 | 212| 184 | 34797 | 158053 | 
| 151 | 63 django/db/models/options.py | 381 | 407| 175 | 34972 | 158053 | 
| 152 | 63 django/db/models/fields/__init__.py | 1815 | 1843| 191 | 35163 | 158053 | 
| 153 | 64 django/core/validators.py | 456 | 491| 249 | 35412 | 162284 | 
| 154 | 64 django/core/cache/backends/memcached.py | 92 | 108| 167 | 35579 | 162284 | 
| 155 | 64 django/db/models/fields/__init__.py | 1846 | 1923| 567 | 36146 | 162284 | 
| 156 | 65 django/db/models/indexes.py | 79 | 118| 411 | 36557 | 163457 | 
| 157 | 66 django/core/management/utils.py | 112 | 125| 119 | 36676 | 164571 | 
| 158 | 67 django/core/management/commands/makemigrations.py | 147 | 184| 302 | 36978 | 167294 | 
| 159 | 68 django/core/cache/backends/base.py | 1 | 47| 245 | 37223 | 169451 | 
| 160 | 68 django/db/migrations/state.py | 492 | 524| 250 | 37473 | 169451 | 
| 161 | 68 django/contrib/auth/hashers.py | 366 | 391| 252 | 37725 | 169451 | 
| 162 | 68 django/contrib/sessions/backends/file.py | 57 | 73| 143 | 37868 | 169451 | 
| 163 | 69 django/db/models/constants.py | 1 | 7| 0 | 37868 | 169476 | 
| 164 | 70 django/core/mail/backends/__init__.py | 1 | 2| 0 | 37868 | 169484 | 
| 165 | 71 django/core/cache/backends/dummy.py | 1 | 40| 255 | 38123 | 169740 | 
| 166 | 71 django/core/files/uploadhandler.py | 27 | 58| 204 | 38327 | 169740 | 
| 167 | 71 django/db/migrations/questioner.py | 227 | 240| 123 | 38450 | 169740 | 
| 168 | 71 django/contrib/messages/storage/cookie.py | 151 | 185| 258 | 38708 | 169740 | 
| 169 | 72 django/utils/dateformat.py | 31 | 44| 121 | 38829 | 172424 | 
| 170 | 73 django/contrib/auth/migrations/0005_alter_user_last_login_null.py | 1 | 17| 0 | 38829 | 172499 | 
| 171 | 73 django/db/models/fields/__init__.py | 1059 | 1088| 218 | 39047 | 172499 | 
| 172 | 74 django/db/migrations/operations/fields.py | 289 | 332| 410 | 39457 | 175462 | 
| 173 | 75 django/db/models/base.py | 404 | 503| 856 | 40313 | 191105 | 
| 174 | 76 django/db/models/functions/mixins.py | 23 | 51| 253 | 40566 | 191519 | 
| 175 | 77 django/core/management/commands/loaddata.py | 305 | 351| 325 | 40891 | 194385 | 
| 176 | 78 django/core/signing.py | 146 | 182| 345 | 41236 | 196212 | 
| 177 | 78 django/db/models/fields/__init__.py | 1673 | 1710| 226 | 41462 | 196212 | 
| 178 | 78 django/contrib/sessions/backends/file.py | 172 | 203| 210 | 41672 | 196212 | 
| 179 | 78 django/contrib/staticfiles/management/commands/collectstatic.py | 295 | 329| 320 | 41992 | 196212 | 
| 180 | 79 django/db/migrations/utils.py | 1 | 18| 0 | 41992 | 196300 | 
| 181 | 80 django/conf/__init__.py | 138 | 158| 167 | 42159 | 198359 | 
| 182 | 80 django/db/models/options.py | 409 | 431| 154 | 42313 | 198359 | 
| 183 | 80 django/db/migrations/serializer.py | 76 | 103| 233 | 42546 | 198359 | 
| 184 | 80 django/core/files/uploadhandler.py | 151 | 206| 382 | 42928 | 198359 | 
| 185 | 80 django/conf/__init__.py | 161 | 220| 541 | 43469 | 198359 | 
| 186 | 80 django/core/mail/message.py | 147 | 166| 218 | 43687 | 198359 | 


## Patch

```diff
diff --git a/django/contrib/staticfiles/storage.py b/django/contrib/staticfiles/storage.py
--- a/django/contrib/staticfiles/storage.py
+++ b/django/contrib/staticfiles/storage.py
@@ -98,8 +98,7 @@ def hashed_name(self, name, content=None, filename=None):
                 content.close()
         path, filename = os.path.split(clean_name)
         root, ext = os.path.splitext(filename)
-        if file_hash is not None:
-            file_hash = ".%s" % file_hash
+        file_hash = ('.%s' % file_hash) if file_hash else ''
         hashed_name = os.path.join(path, "%s%s%s" %
                                    (root, file_hash, ext))
         unparsed_name = list(parsed_name)

```

## Test Patch

```diff
diff --git a/tests/staticfiles_tests/storage.py b/tests/staticfiles_tests/storage.py
--- a/tests/staticfiles_tests/storage.py
+++ b/tests/staticfiles_tests/storage.py
@@ -88,3 +88,8 @@ class ExtraPatternsStorage(ManifestStaticFilesStorage):
             ),
         ),
     )
+
+
+class NoneHashStorage(ManifestStaticFilesStorage):
+    def file_hash(self, name, content=None):
+        return None
diff --git a/tests/staticfiles_tests/test_storage.py b/tests/staticfiles_tests/test_storage.py
--- a/tests/staticfiles_tests/test_storage.py
+++ b/tests/staticfiles_tests/test_storage.py
@@ -386,6 +386,15 @@ def test_intermediate_files(self):
         )
 
 
+@override_settings(STATICFILES_STORAGE='staticfiles_tests.storage.NoneHashStorage')
+class TestCollectionNoneHashStorage(CollectionTestCase):
+    hashed_file_path = hashed_file_path
+
+    def test_hashed_name(self):
+        relpath = self.hashed_file_path('cached/styles.css')
+        self.assertEqual(relpath, 'cached/styles.css')
+
+
 @override_settings(STATICFILES_STORAGE='staticfiles_tests.storage.SimpleStorage')
 class TestCollectionSimpleStorage(CollectionTestCase):
     hashed_file_path = hashed_file_path

```


## Code snippets

### 1 - django/contrib/staticfiles/storage.py:

Start line: 79, End line: 111

```python
class HashedFilesMixin:

    def hashed_name(self, name, content=None, filename=None):
        # `filename` is the name of file to hash if `content` isn't given.
        # `name` is the base name to construct the new hashed filename from.
        parsed_name = urlsplit(unquote(name))
        clean_name = parsed_name.path.strip()
        filename = (filename and urlsplit(unquote(filename)).path.strip()) or clean_name
        opened = content is None
        if opened:
            if not self.exists(filename):
                raise ValueError("The file '%s' could not be found with %r." % (filename, self))
            try:
                content = self.open(filename)
            except OSError:
                # Handle directory paths and fragments
                return name
        try:
            file_hash = self.file_hash(clean_name, content)
        finally:
            if opened:
                content.close()
        path, filename = os.path.split(clean_name)
        root, ext = os.path.splitext(filename)
        if file_hash is not None:
            file_hash = ".%s" % file_hash
        hashed_name = os.path.join(path, "%s%s%s" %
                                   (root, file_hash, ext))
        unparsed_name = list(parsed_name)
        unparsed_name[2] = hashed_name
        # Special casing for a @font-face hack, like url(myfont.eot?#iefix")
        # http://www.fontspring.com/blog/the-new-bulletproof-font-face-syntax
        if '?#' in name and not unparsed_name[3]:
            unparsed_name[2] += '?'
        return urlunsplit(unparsed_name)
```
### 2 - django/contrib/staticfiles/storage.py:

Start line: 342, End line: 363

```python
class HashedFilesMixin:

    def stored_name(self, name):
        cleaned_name = self.clean_name(name)
        hash_key = self.hash_key(cleaned_name)
        cache_name = self.hashed_files.get(hash_key)
        if cache_name:
            return cache_name
        # No cached name found, recalculate it from the files.
        intermediate_name = name
        for i in range(self.max_post_process_passes + 1):
            cache_name = self.clean_name(
                self.hashed_name(name, content=None, filename=intermediate_name)
            )
            if intermediate_name == cache_name:
                # Store the hashed name if there was a miss.
                self.hashed_files[hash_key] = cache_name
                return cache_name
            else:
                # Move on to the next intermediate file.
                intermediate_name = cache_name
        # If the cache name can't be determined after the max number of passes,
        # the intermediate files on disk may be corrupt; avoid an infinite loop.
        raise ValueError("The name '%s' could not be hashed with %r." % (name, self))
```
### 3 - django/contrib/staticfiles/storage.py:

Start line: 324, End line: 340

```python
class HashedFilesMixin:

    def clean_name(self, name):
        return name.replace('\\', '/')

    def hash_key(self, name):
        return name

    def _stored_name(self, name, hashed_files):
        # Normalize the path to avoid multiple names for the same file like
        # ../foo/bar.css and ../foo/../foo/bar.css which normalize to the same
        # path.
        name = posixpath.normpath(name)
        cleaned_name = self.clean_name(name)
        hash_key = self.hash_key(cleaned_name)
        cache_name = hashed_files.get(hash_key)
        if cache_name is None:
            cache_name = self.clean_name(self.hashed_name(name))
        return cache_name
```
### 4 - django/contrib/staticfiles/storage.py:

Start line: 366, End line: 409

```python
class ManifestFilesMixin(HashedFilesMixin):
    manifest_version = '1.0'  # the manifest format standard
    manifest_name = 'staticfiles.json'
    manifest_strict = True
    keep_intermediate_files = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hashed_files = self.load_manifest()

    def read_manifest(self):
        try:
            with self.open(self.manifest_name) as manifest:
                return manifest.read().decode()
        except FileNotFoundError:
            return None

    def load_manifest(self):
        content = self.read_manifest()
        if content is None:
            return {}
        try:
            stored = json.loads(content)
        except json.JSONDecodeError:
            pass
        else:
            version = stored.get('version')
            if version == '1.0':
                return stored.get('paths', {})
        raise ValueError("Couldn't load manifest '%s' (version %s)" %
                         (self.manifest_name, self.manifest_version))

    def post_process(self, *args, **kwargs):
        self.hashed_files = {}
        yield from super().post_process(*args, **kwargs)
        if not kwargs.get('dry_run'):
            self.save_manifest()

    def save_manifest(self):
        payload = {'paths': self.hashed_files, 'version': self.manifest_version}
        if self.exists(self.manifest_name):
            self.delete(self.manifest_name)
        contents = json.dumps(payload).encode()
        self._save(self.manifest_name, ContentFile(contents))
```
### 5 - django/contrib/staticfiles/storage.py:

Start line: 411, End line: 443

```python
class ManifestFilesMixin(HashedFilesMixin):

    def stored_name(self, name):
        parsed_name = urlsplit(unquote(name))
        clean_name = parsed_name.path.strip()
        hash_key = self.hash_key(clean_name)
        cache_name = self.hashed_files.get(hash_key)
        if cache_name is None:
            if self.manifest_strict:
                raise ValueError("Missing staticfiles manifest entry for '%s'" % clean_name)
            cache_name = self.clean_name(self.hashed_name(name))
        unparsed_name = list(parsed_name)
        unparsed_name[2] = cache_name
        # Special casing for a @font-face hack, like url(myfont.eot?#iefix")
        # http://www.fontspring.com/blog/the-new-bulletproof-font-face-syntax
        if '?#' in name and not unparsed_name[3]:
            unparsed_name[2] += '?'
        return urlunsplit(unparsed_name)


class ManifestStaticFilesStorage(ManifestFilesMixin, StaticFilesStorage):
    """
    A static file system storage backend which also saves
    hashed copies of the files it saves.
    """
    pass


class ConfiguredStorage(LazyObject):
    def _setup(self):
        self._wrapped = get_storage_class(settings.STATICFILES_STORAGE)()


staticfiles_storage = ConfiguredStorage()
```
### 6 - django/contrib/staticfiles/storage.py:

Start line: 252, End line: 322

```python
class HashedFilesMixin:

    def _post_process(self, paths, adjustable_paths, hashed_files):
        # Sort the files by directory level
        def path_level(name):
            return len(name.split(os.sep))

        for name in sorted(paths, key=path_level, reverse=True):
            substitutions = True
            # use the original, local file, not the copied-but-unprocessed
            # file, which might be somewhere far away, like S3
            storage, path = paths[name]
            with storage.open(path) as original_file:
                cleaned_name = self.clean_name(name)
                hash_key = self.hash_key(cleaned_name)

                # generate the hash with the original content, even for
                # adjustable files.
                if hash_key not in hashed_files:
                    hashed_name = self.hashed_name(name, original_file)
                else:
                    hashed_name = hashed_files[hash_key]

                # then get the original's file content..
                if hasattr(original_file, 'seek'):
                    original_file.seek(0)

                hashed_file_exists = self.exists(hashed_name)
                processed = False

                # ..to apply each replacement pattern to the content
                if name in adjustable_paths:
                    old_hashed_name = hashed_name
                    content = original_file.read().decode('utf-8')
                    for extension, patterns in self._patterns.items():
                        if matches_patterns(path, (extension,)):
                            for pattern, template in patterns:
                                converter = self.url_converter(name, hashed_files, template)
                                try:
                                    content = pattern.sub(converter, content)
                                except ValueError as exc:
                                    yield name, None, exc, False
                    if hashed_file_exists:
                        self.delete(hashed_name)
                    # then save the processed result
                    content_file = ContentFile(content.encode())
                    if self.keep_intermediate_files:
                        # Save intermediate file for reference
                        self._save(hashed_name, content_file)
                    hashed_name = self.hashed_name(name, content_file)

                    if self.exists(hashed_name):
                        self.delete(hashed_name)

                    saved_name = self._save(hashed_name, content_file)
                    hashed_name = self.clean_name(saved_name)
                    # If the file hash stayed the same, this file didn't change
                    if old_hashed_name == hashed_name:
                        substitutions = False
                    processed = True

                if not processed:
                    # or handle the case in which neither processing nor
                    # a change to the original file happened
                    if not hashed_file_exists:
                        processed = True
                        saved_name = self._save(hashed_name, original_file)
                        hashed_name = self.clean_name(saved_name)

                # and then set the cache accordingly
                hashed_files[hash_key] = hashed_name

                yield name, hashed_name, processed, substitutions
```
### 7 - django/contrib/staticfiles/storage.py:

Start line: 44, End line: 77

```python
class HashedFilesMixin:
    default_template = """url("%s")"""
    max_post_process_passes = 5
    patterns = (
        ("*.css", (
            r"""(url\(['"]{0,1}\s*(.*?)["']{0,1}\))""",
            (r"""(@import\s*["']\s*(.*?)["'])""", """@import url("%s")"""),
        )),
    )
    keep_intermediate_files = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._patterns = {}
        self.hashed_files = {}
        for extension, patterns in self.patterns:
            for pattern in patterns:
                if isinstance(pattern, (tuple, list)):
                    pattern, template = pattern
                else:
                    template = self.default_template
                compiled = re.compile(pattern, re.IGNORECASE)
                self._patterns.setdefault(extension, []).append((compiled, template))

    def file_hash(self, name, content=None):
        """
        Return a hash of the file with the given name and optional content.
        """
        if content is None:
            return None
        md5 = hashlib.md5()
        for chunk in content.chunks():
            md5.update(chunk)
        return md5.hexdigest()[:12]
```
### 8 - django/contrib/staticfiles/storage.py:

Start line: 150, End line: 202

```python
class HashedFilesMixin:

    def url_converter(self, name, hashed_files, template=None):
        """
        Return the custom URL converter for the given file name.
        """
        if template is None:
            template = self.default_template

        def converter(matchobj):
            """
            Convert the matched URL to a normalized and hashed URL.

            This requires figuring out which files the matched URL resolves
            to and calling the url() method of the storage.
            """
            matched, url = matchobj.groups()

            # Ignore absolute/protocol-relative and data-uri URLs.
            if re.match(r'^[a-z]+:', url):
                return matched

            # Ignore absolute URLs that don't point to a static file (dynamic
            # CSS / JS?). Note that STATIC_URL cannot be empty.
            if url.startswith('/') and not url.startswith(settings.STATIC_URL):
                return matched

            # Strip off the fragment so a path-like fragment won't interfere.
            url_path, fragment = urldefrag(url)

            if url_path.startswith('/'):
                # Otherwise the condition above would have returned prematurely.
                assert url_path.startswith(settings.STATIC_URL)
                target_name = url_path[len(settings.STATIC_URL):]
            else:
                # We're using the posixpath module to mix paths and URLs conveniently.
                source_name = name if os.sep == '/' else name.replace(os.sep, '/')
                target_name = posixpath.join(posixpath.dirname(source_name), url_path)

            # Determine the hashed name of the target file with the storage backend.
            hashed_url = self._url(
                self._stored_name, unquote(target_name),
                force=True, hashed_files=hashed_files,
            )

            transformed_url = '/'.join(url_path.split('/')[:-1] + hashed_url.split('/')[-1:])

            # Restore the fragment that was stripped off earlier.
            if fragment:
                transformed_url += ('?#' if '?#' in url else '#') + fragment

            # Return the hashed version to the file
            return template % unquote(transformed_url)

        return converter
```
### 9 - django/core/files/uploadedfile.py:

Start line: 38, End line: 52

```python
class UploadedFile(File):

    def _set_name(self, name):
        # Sanitize the file name so that it can't be dangerous.
        if name is not None:
            # Just use the basename of the file -- anything else is dangerous.
            name = os.path.basename(name)

            # File names longer than 255 characters can cause problems on older OSes.
            if len(name) > 255:
                name, ext = os.path.splitext(name)
                ext = ext[:255]
                name = name[:255 - len(ext)] + ext

        self._name = name

    name = property(_get_name, _set_name)
```
### 10 - django/core/files/storage.py:

Start line: 54, End line: 69

```python
class Storage:

    # These methods are part of the public API, with default implementations.

    def get_valid_name(self, name):
        """
        Return a filename, based on the provided filename, that's suitable for
        use in the target storage system.
        """
        return get_valid_filename(name)

    def get_alternative_name(self, file_root, file_ext):
        """
        Return an alternative filename, by adding an underscore and a random 7
        character alphanumeric string (before the file extension, if one
        exists) to the filename.
        """
        return '%s_%s%s' % (file_root, get_random_string(7), file_ext)
```
### 12 - django/contrib/staticfiles/storage.py:

Start line: 113, End line: 148

```python
class HashedFilesMixin:

    def _url(self, hashed_name_func, name, force=False, hashed_files=None):
        """
        Return the non-hashed URL in DEBUG mode.
        """
        if settings.DEBUG and not force:
            hashed_name, fragment = name, ''
        else:
            clean_name, fragment = urldefrag(name)
            if urlsplit(clean_name).path.endswith('/'):  # don't hash paths
                hashed_name = name
            else:
                args = (clean_name,)
                if hashed_files is not None:
                    args += (hashed_files,)
                hashed_name = hashed_name_func(*args)

        final_url = super().url(hashed_name)

        # Special casing for a @font-face hack, like url(myfont.eot?#iefix")
        # http://www.fontspring.com/blog/the-new-bulletproof-font-face-syntax
        query_fragment = '?#' in name  # [sic!]
        if fragment or query_fragment:
            urlparts = list(urlsplit(final_url))
            if fragment and not urlparts[4]:
                urlparts[4] = fragment
            if query_fragment and not urlparts[3]:
                urlparts[2] += '?'
            final_url = urlunsplit(urlparts)

        return unquote(final_url)

    def url(self, name, force=False):
        """
        Return the non-hashed URL in DEBUG mode.
        """
        return self._url(self.stored_name, name, force)
```
### 25 - django/contrib/staticfiles/storage.py:

Start line: 1, End line: 41

```python
import hashlib
import json
import os
import posixpath
import re
from urllib.parse import unquote, urldefrag, urlsplit, urlunsplit

from django.conf import settings
from django.contrib.staticfiles.utils import check_settings, matches_patterns
from django.core.exceptions import ImproperlyConfigured
from django.core.files.base import ContentFile
from django.core.files.storage import FileSystemStorage, get_storage_class
from django.utils.functional import LazyObject


class StaticFilesStorage(FileSystemStorage):
    """
    Standard file system storage for static files.

    The defaults for ``location`` and ``base_url`` are
    ``STATIC_ROOT`` and ``STATIC_URL``.
    """
    def __init__(self, location=None, base_url=None, *args, **kwargs):
        if location is None:
            location = settings.STATIC_ROOT
        if base_url is None:
            base_url = settings.STATIC_URL
        check_settings(base_url)
        super().__init__(location, base_url, *args, **kwargs)
        # FileSystemStorage fallbacks to MEDIA_ROOT when location
        # is empty, so we restore the empty value.
        if not location:
            self.base_location = None
            self.location = None

    def path(self, name):
        if not self.location:
            raise ImproperlyConfigured("You're using the staticfiles app "
                                       "without having set the STATIC_ROOT "
                                       "setting to a filesystem path.")
        return super().path(name)
```
### 37 - django/contrib/staticfiles/storage.py:

Start line: 204, End line: 250

```python
class HashedFilesMixin:

    def post_process(self, paths, dry_run=False, **options):
        """
        Post process the given dictionary of files (called from collectstatic).

        Processing is actually two separate operations:

        1. renaming files to include a hash of their content for cache-busting,
           and copying those files to the target storage.
        2. adjusting files which contain references to other files so they
           refer to the cache-busting filenames.

        If either of these are performed on a file, then that file is considered
        post-processed.
        """
        # don't even dare to process the files if we're in dry run mode
        if dry_run:
            return

        # where to store the new paths
        hashed_files = {}

        # build a list of adjustable files
        adjustable_paths = [
            path for path in paths
            if matches_patterns(path, self._patterns)
        ]
        # Do a single pass first. Post-process all files once, then repeat for
        # adjustable files.
        for name, hashed_name, processed, _ in self._post_process(paths, adjustable_paths, hashed_files):
            yield name, hashed_name, processed

        paths = {path: paths[path] for path in adjustable_paths}

        for i in range(self.max_post_process_passes):
            substitutions = False
            for name, hashed_name, processed, subst in self._post_process(paths, adjustable_paths, hashed_files):
                yield name, hashed_name, processed
                substitutions = substitutions or subst

            if not substitutions:
                break

        if substitutions:
            yield 'All', None, RuntimeError('Max post-process passes exceeded.')

        # Store the processed paths
        self.hashed_files.update(hashed_files)
```
