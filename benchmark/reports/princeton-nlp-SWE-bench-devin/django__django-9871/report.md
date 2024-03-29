# django__django-9871

| **django/django** | `2919a08c20d5ae48e381d6bd251d3b0d400d47d9` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 35417 |
| **Any found context length** | 35417 |
| **Avg pos** | 188.0 |
| **Min pos** | 94 |
| **Max pos** | 94 |
| **Top file pos** | 2 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/core/management/base.py b/django/core/management/base.py
--- a/django/core/management/base.py
+++ b/django/core/management/base.py
@@ -228,6 +228,9 @@ def create_parser(self, prog_name, subcommand):
             self, prog="%s %s" % (os.path.basename(prog_name), subcommand),
             description=self.help or None,
         )
+        # Add command-specific arguments first so that they appear in the
+        # --help output before arguments common to all commands.
+        self.add_arguments(parser)
         parser.add_argument('--version', action='version', version=self.get_version())
         parser.add_argument(
             '-v', '--verbosity', action='store', dest='verbosity', default=1,
@@ -251,7 +254,6 @@ def create_parser(self, prog_name, subcommand):
             '--no-color', action='store_true', dest='no_color',
             help="Don't colorize the command output.",
         )
-        self.add_arguments(parser)
         return parser
 
     def add_arguments(self, parser):

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/core/management/base.py | 231 | 231 | 94 | 2 | 35417
| django/core/management/base.py | 254 | 254 | 94 | 2 | 35417


## Problem Statement

```
Reorder management command arguments in --help output to prioritize command-specific arguments
Description
	
Currently if you run a custom management command with --help, you will get output that looks like:
I have highlighted in yellow the useful information specific to the command that is *not* boilerplate. Notice that most of this yellow text is at the end of the output, with the boilerplate dominating what the user reads first.
I propose reordering the options in the output so that the useful information is at the *beginning* rather than the end, so that it looks like the following:
Discussion on django-developers: ​https://groups.google.com/forum/#!topic/django-developers/PByZfN_IccE

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 django/core/management/__init__.py | 151 | 193| 343 | 343 | 3275 | 
| 2 | **2 django/core/management/base.py** | 491 | 523| 240 | 583 | 7429 | 
| 3 | 2 django/core/management/__init__.py | 301 | 382| 742 | 1325 | 7429 | 
| 4 | **2 django/core/management/base.py** | 63 | 106| 297 | 1622 | 7429 | 
| 5 | 2 django/core/management/__init__.py | 227 | 299| 722 | 2344 | 7429 | 
| 6 | 3 django/core/management/commands/dumpdata.py | 68 | 141| 627 | 2971 | 9000 | 
| 7 | 4 django/core/management/commands/makemessages.py | 364 | 394| 245 | 3216 | 14784 | 
| 8 | 4 django/core/management/commands/makemessages.py | 284 | 363| 809 | 4025 | 14784 | 
| 9 | **4 django/core/management/base.py** | 257 | 305| 381 | 4406 | 14784 | 
| 10 | 4 django/core/management/commands/makemessages.py | 217 | 282| 681 | 5087 | 14784 | 
| 11 | 5 django/core/management/commands/squashmigrations.py | 1 | 38| 321 | 5408 | 16583 | 
| 12 | **5 django/core/management/base.py** | 109 | 189| 752 | 6160 | 16583 | 
| 13 | 6 django/core/management/commands/migrate.py | 61 | 135| 646 | 6806 | 19293 | 
| 14 | 7 django/core/management/commands/makemigrations.py | 54 | 137| 768 | 7574 | 22000 | 
| 15 | 8 django/core/management/commands/diffsettings.py | 68 | 79| 143 | 7717 | 22694 | 
| 16 | 8 django/core/management/commands/diffsettings.py | 41 | 54| 127 | 7844 | 22694 | 
| 17 | **8 django/core/management/base.py** | 455 | 488| 291 | 8135 | 22694 | 
| 18 | 8 django/core/management/commands/diffsettings.py | 56 | 66| 128 | 8263 | 22694 | 
| 19 | 8 django/core/management/commands/squashmigrations.py | 40 | 124| 751 | 9014 | 22694 | 
| 20 | 8 django/core/management/commands/squashmigrations.py | 126 | 190| 649 | 9663 | 22694 | 
| 21 | **8 django/core/management/base.py** | 190 | 220| 251 | 9914 | 22694 | 
| 22 | 9 django/core/management/commands/shell.py | 1 | 40| 281 | 10195 | 23521 | 
| 23 | 9 django/core/management/commands/shell.py | 42 | 81| 401 | 10596 | 23521 | 
| 24 | 9 django/core/management/commands/makemigrations.py | 138 | 175| 296 | 10892 | 23521 | 
| 25 | 9 django/core/management/__init__.py | 1 | 37| 254 | 11146 | 23521 | 
| 26 | 10 django/contrib/admin/views/main.py | 267 | 312| 436 | 11582 | 27152 | 
| 27 | 10 django/contrib/admin/views/main.py | 237 | 265| 248 | 11830 | 27152 | 
| 28 | 11 django/core/management/templates.py | 62 | 118| 487 | 12317 | 29810 | 
| 29 | 12 django/core/management/commands/showmigrations.py | 35 | 53| 158 | 12475 | 30902 | 
| 30 | 12 django/core/management/commands/makemessages.py | 1 | 34| 246 | 12721 | 30902 | 
| 31 | 12 django/core/management/commands/showmigrations.py | 1 | 33| 266 | 12987 | 30902 | 
| 32 | 13 django/core/management/commands/test.py | 28 | 60| 274 | 13261 | 31334 | 
| 33 | 13 django/core/management/commands/makemigrations.py | 21 | 52| 271 | 13532 | 31334 | 
| 34 | 13 django/contrib/admin/views/main.py | 314 | 350| 315 | 13847 | 31334 | 
| 35 | 13 django/core/management/__init__.py | 195 | 225| 305 | 14152 | 31334 | 
| 36 | 14 django/core/management/commands/sqlmigrate.py | 31 | 60| 297 | 14449 | 31891 | 
| 37 | 14 django/core/management/commands/dumpdata.py | 171 | 195| 224 | 14673 | 31891 | 
| 38 | 14 django/core/management/commands/shell.py | 83 | 103| 155 | 14828 | 31891 | 
| 39 | 15 django/core/management/commands/compilemessages.py | 26 | 48| 195 | 15023 | 32978 | 
| 40 | 15 django/core/management/commands/dumpdata.py | 1 | 66| 542 | 15565 | 32978 | 
| 41 | 15 django/core/management/templates.py | 120 | 181| 551 | 16116 | 32978 | 
| 42 | **15 django/core/management/base.py** | 1 | 29| 195 | 16311 | 32978 | 
| 43 | 16 django/db/migrations/autodetector.py | 326 | 342| 166 | 16477 | 44230 | 
| 44 | 16 django/core/management/commands/migrate.py | 19 | 59| 361 | 16838 | 44230 | 
| 45 | 16 django/core/management/commands/makemessages.py | 197 | 215| 185 | 17023 | 44230 | 
| 46 | 16 django/core/management/commands/sqlmigrate.py | 1 | 29| 265 | 17288 | 44230 | 
| 47 | 16 django/core/management/commands/compilemessages.py | 50 | 100| 441 | 17729 | 44230 | 
| 48 | 16 django/core/management/__init__.py | 75 | 148| 654 | 18383 | 44230 | 
| 49 | 17 django/core/management/commands/loaddata.py | 32 | 61| 291 | 18674 | 47076 | 
| 50 | 18 django/contrib/gis/management/commands/ogrinspect.py | 98 | 134| 411 | 19085 | 48326 | 
| 51 | 19 django/core/management/commands/inspectdb.py | 171 | 225| 478 | 19563 | 50898 | 
| 52 | 20 django/contrib/auth/management/commands/changepassword.py | 34 | 76| 337 | 19900 | 51445 | 
| 53 | 20 django/contrib/admin/views/main.py | 1 | 35| 245 | 20145 | 51445 | 
| 54 | 20 django/core/management/commands/dumpdata.py | 143 | 169| 239 | 20384 | 51445 | 
| 55 | 20 django/core/management/commands/makemessages.py | 420 | 450| 263 | 20647 | 51445 | 
| 56 | 21 django/contrib/admin/options.py | 1658 | 1736| 706 | 21353 | 68608 | 
| 57 | 21 django/core/management/__init__.py | 40 | 72| 265 | 21618 | 68608 | 
| 58 | 21 django/core/management/templates.py | 44 | 60| 181 | 21799 | 68608 | 
| 59 | 21 django/contrib/gis/management/commands/ogrinspect.py | 33 | 96| 628 | 22427 | 68608 | 
| 60 | 21 django/core/management/commands/loaddata.py | 63 | 79| 187 | 22614 | 68608 | 
| 61 | 22 django/core/management/utils.py | 51 | 73| 204 | 22818 | 69404 | 
| 62 | 23 docs/conf.py | 105 | 224| 871 | 23689 | 72355 | 
| 63 | 24 django/contrib/auth/management/commands/createsuperuser.py | 61 | 203| 1119 | 24808 | 73931 | 
| 64 | 24 docs/conf.py | 226 | 372| 1044 | 25852 | 73931 | 
| 65 | 25 django/core/management/commands/flush.py | 27 | 83| 496 | 26348 | 74636 | 
| 66 | 26 django/contrib/admin/utils.py | 380 | 414| 305 | 26653 | 78563 | 
| 67 | 26 django/core/management/commands/inspectdb.py | 1 | 34| 262 | 26915 | 78563 | 
| 68 | 26 django/core/management/commands/makemessages.py | 396 | 418| 200 | 27115 | 78563 | 
| 69 | 27 django/contrib/admin/templatetags/admin_list.py | 103 | 189| 742 | 27857 | 82338 | 
| 70 | 27 django/core/management/commands/migrate.py | 226 | 258| 337 | 28194 | 82338 | 
| 71 | 28 django/contrib/staticfiles/management/commands/collectstatic.py | 38 | 69| 310 | 28504 | 85204 | 
| 72 | 28 django/contrib/admin/options.py | 901 | 939| 269 | 28773 | 85204 | 
| 73 | 28 django/core/management/commands/diffsettings.py | 1 | 39| 309 | 29082 | 85204 | 
| 74 | 28 django/contrib/admin/views/main.py | 186 | 235| 423 | 29505 | 85204 | 
| 75 | 29 django/core/management/commands/sqlsequencereset.py | 1 | 24| 172 | 29677 | 85376 | 
| 76 | 30 django/db/models/options.py | 1 | 38| 330 | 30007 | 92261 | 
| 77 | 31 django/core/management/commands/check.py | 37 | 67| 214 | 30221 | 92711 | 
| 78 | 31 django/core/management/commands/migrate.py | 136 | 224| 865 | 31086 | 92711 | 
| 79 | 31 django/contrib/staticfiles/management/commands/collectstatic.py | 148 | 206| 503 | 31589 | 92711 | 
| 80 | 31 django/core/management/commands/check.py | 1 | 35| 241 | 31830 | 92711 | 
| 81 | 31 django/core/management/commands/makemigrations.py | 223 | 303| 820 | 32650 | 92711 | 
| 82 | 32 django/contrib/admindocs/utils.py | 1 | 33| 241 | 32891 | 94598 | 
| 83 | 32 django/contrib/auth/management/commands/changepassword.py | 1 | 32| 214 | 33105 | 94598 | 
| 84 | 33 django/db/models/base.py | 1688 | 1703| 146 | 33251 | 108676 | 
| 85 | 34 scripts/manage_translations.py | 1 | 29| 200 | 33451 | 110351 | 
| 86 | 34 django/core/management/commands/squashmigrations.py | 192 | 205| 112 | 33563 | 110351 | 
| 87 | 34 django/core/management/commands/test.py | 1 | 26| 163 | 33726 | 110351 | 
| 88 | 34 django/core/management/commands/makemessages.py | 658 | 688| 289 | 34015 | 110351 | 
| 89 | 35 django/core/management/commands/sendtestemail.py | 1 | 24| 195 | 34210 | 110662 | 
| 90 | 36 django/core/management/commands/runserver.py | 55 | 65| 120 | 34330 | 112120 | 
| 91 | 36 django/contrib/admin/options.py | 205 | 216| 135 | 34465 | 112120 | 
| 92 | 36 django/core/management/templates.py | 209 | 239| 235 | 34700 | 112120 | 
| 93 | 36 django/contrib/admin/views/main.py | 352 | 395| 390 | 35090 | 112120 | 
| **-> 94 <-** | **36 django/core/management/base.py** | 222 | 255| 327 | 35417 | 112120 | 
| 95 | 36 django/contrib/admin/views/main.py | 38 | 100| 553 | 35970 | 112120 | 
| 96 | 36 django/contrib/admin/options.py | 1577 | 1657| 741 | 36711 | 112120 | 
| 97 | 36 django/core/management/commands/flush.py | 1 | 25| 214 | 36925 | 112120 | 
| 98 | 37 django/contrib/staticfiles/management/commands/findstatic.py | 1 | 44| 318 | 37243 | 112438 | 
| 99 | 37 django/core/management/commands/runserver.py | 107 | 163| 518 | 37761 | 112438 | 
| 100 | 38 django/contrib/contenttypes/management/commands/remove_stale_contenttypes.py | 1 | 81| 605 | 38366 | 113063 | 
| 101 | 38 django/core/management/commands/runserver.py | 23 | 53| 248 | 38614 | 113063 | 
| 102 | 38 django/core/management/commands/makemessages.py | 37 | 58| 143 | 38757 | 113063 | 
| 103 | 38 django/contrib/auth/management/commands/createsuperuser.py | 1 | 59| 461 | 39218 | 113063 | 
| 104 | 39 django/db/migrations/operations/models.py | 587 | 608| 173 | 39391 | 119431 | 
| 105 | 39 scripts/manage_translations.py | 171 | 181| 116 | 39507 | 119431 | 
| 106 | 39 django/core/management/commands/loaddata.py | 1 | 29| 151 | 39658 | 119431 | 
| 107 | 39 django/core/management/utils.py | 29 | 48| 195 | 39853 | 119431 | 
| 108 | 39 django/core/management/commands/inspectdb.py | 36 | 169| 1228 | 41081 | 119431 | 
| 109 | 40 django/views/generic/list.py | 50 | 75| 244 | 41325 | 121004 | 
| 110 | 40 django/core/management/commands/showmigrations.py | 55 | 89| 360 | 41685 | 121004 | 
| 111 | 40 django/core/management/commands/loaddata.py | 81 | 145| 557 | 42242 | 121004 | 
| 112 | 41 django/utils/termcolors.py | 58 | 133| 747 | 42989 | 122891 | 
| 113 | 41 django/db/models/options.py | 398 | 420| 154 | 43143 | 122891 | 
| 114 | 41 django/contrib/admin/options.py | 275 | 360| 616 | 43759 | 122891 | 
| 115 | 41 django/contrib/admin/options.py | 686 | 719| 246 | 44005 | 122891 | 
| 116 | 41 django/core/management/commands/compilemessages.py | 1 | 23| 148 | 44153 | 122891 | 
| 117 | **41 django/core/management/base.py** | 32 | 60| 211 | 44364 | 122891 | 
| 118 | 41 django/core/management/commands/showmigrations.py | 91 | 132| 323 | 44687 | 122891 | 
| 119 | 42 docs/_ext/djangodocs.py | 209 | 237| 216 | 44903 | 126888 | 
| 120 | 42 django/core/management/commands/makemigrations.py | 177 | 221| 452 | 45355 | 126888 | 
| 121 | 43 django/core/management/commands/dbshell.py | 1 | 32| 239 | 45594 | 127127 | 
| 122 | 44 django/core/management/commands/sqlflush.py | 1 | 23| 161 | 45755 | 127288 | 
| 123 | 44 django/contrib/staticfiles/management/commands/collectstatic.py | 208 | 243| 248 | 46003 | 127288 | 
| 124 | 44 docs/conf.py | 1 | 104| 820 | 46823 | 127288 | 
| 125 | 44 django/contrib/admin/options.py | 1 | 96| 766 | 47589 | 127288 | 
| 126 | 45 django/template/defaultfilters.py | 491 | 584| 525 | 48114 | 133343 | 
| 127 | 46 django/db/models/sql/compiler.py | 326 | 354| 328 | 48442 | 146341 | 
| 128 | 46 django/contrib/admin/views/main.py | 102 | 184| 818 | 49260 | 146341 | 
| 129 | 46 django/contrib/admin/options.py | 1808 | 1845| 328 | 49588 | 146341 | 
| 130 | 47 django/forms/forms.py | 119 | 139| 178 | 49766 | 150445 | 
| 131 | 47 django/core/management/commands/loaddata.py | 211 | 267| 549 | 50315 | 150445 | 
| 132 | 48 django/db/models/sql/query.py | 1745 | 1768| 204 | 50519 | 170760 | 
| 133 | 48 django/core/management/commands/migrate.py | 1 | 16| 137 | 50656 | 170760 | 
| 134 | **48 django/core/management/base.py** | 353 | 418| 614 | 51270 | 170760 | 
| 135 | 49 django/conf/global_settings.py | 144 | 261| 869 | 52139 | 176304 | 
| 136 | 49 django/core/management/commands/makemessages.py | 98 | 116| 157 | 52296 | 176304 | 
| 137 | 50 django/contrib/admin/__init__.py | 1 | 30| 286 | 52582 | 176590 | 
| 138 | 50 django/db/models/sql/query.py | 2097 | 2108| 116 | 52698 | 176590 | 
| 139 | 50 django/db/models/options.py | 346 | 368| 164 | 52862 | 176590 | 
| 140 | 50 django/core/management/commands/runserver.py | 67 | 105| 395 | 53257 | 176590 | 
| 141 | 51 django/core/management/commands/testserver.py | 29 | 55| 234 | 53491 | 177024 | 
| 142 | 51 django/contrib/admin/options.py | 1738 | 1806| 585 | 54076 | 177024 | 
| 143 | 51 django/core/management/utils.py | 1 | 26| 173 | 54249 | 177024 | 
| 144 | **51 django/core/management/base.py** | 307 | 351| 333 | 54582 | 177024 | 
| 145 | 51 django/db/models/sql/compiler.py | 899 | 921| 180 | 54762 | 177024 | 
| 146 | 52 django/contrib/admin/checks.py | 489 | 518| 251 | 55013 | 185374 | 
| 147 | 53 django/db/models/__init__.py | 1 | 47| 520 | 55533 | 185894 | 
| 148 | 53 django/db/migrations/operations/models.py | 610 | 626| 216 | 55749 | 185894 | 
| 149 | 53 django/contrib/admin/options.py | 1482 | 1575| 857 | 56606 | 185894 | 
| 150 | 53 django/db/migrations/operations/models.py | 628 | 641| 138 | 56744 | 185894 | 
| 151 | 53 django/core/management/commands/makemessages.py | 452 | 519| 632 | 57376 | 185894 | 
| 152 | 53 django/core/management/commands/migrate.py | 260 | 308| 404 | 57780 | 185894 | 
| 153 | 53 django/db/models/base.py | 903 | 916| 180 | 57960 | 185894 | 
| 154 | 53 django/db/models/options.py | 370 | 396| 175 | 58135 | 185894 | 
| 155 | 53 django/db/models/base.py | 1547 | 1612| 517 | 58652 | 185894 | 
| 156 | 53 django/contrib/admin/options.py | 976 | 987| 149 | 58801 | 185894 | 
| 157 | 54 django/core/management/commands/createcachetable.py | 32 | 44| 121 | 58922 | 186769 | 
| 158 | 54 django/contrib/admin/options.py | 1079 | 1117| 399 | 59321 | 186769 | 
| 159 | 54 django/core/management/commands/loaddata.py | 147 | 209| 559 | 59880 | 186769 | 
| 160 | 54 django/core/management/commands/sendtestemail.py | 26 | 41| 121 | 60001 | 186769 | 
| 161 | **54 django/core/management/base.py** | 420 | 452| 282 | 60283 | 186769 | 
| 162 | 54 django/core/management/commands/makemessages.py | 156 | 168| 111 | 60394 | 186769 | 
| 163 | 54 django/db/models/sql/compiler.py | 647 | 676| 351 | 60745 | 186769 | 
| 164 | 54 docs/_ext/djangodocs.py | 24 | 79| 508 | 61253 | 186769 | 
| 165 | 55 django/core/management/commands/startproject.py | 1 | 21| 137 | 61390 | 186906 | 
| 166 | 55 django/contrib/admin/options.py | 989 | 1011| 198 | 61588 | 186906 | 
| 167 | 55 django/db/models/options.py | 150 | 203| 518 | 62106 | 186906 | 
| 168 | 55 django/core/management/commands/compilemessages.py | 102 | 132| 312 | 62418 | 186906 | 
| 169 | 56 django/contrib/admin/sites.py | 465 | 498| 228 | 62646 | 190981 | 
| 170 | 57 django/core/management/color.py | 1 | 26| 131 | 62777 | 191375 | 
| 171 | 57 django/db/models/options.py | 67 | 148| 661 | 63438 | 191375 | 
| 172 | 57 django/contrib/staticfiles/management/commands/collectstatic.py | 71 | 84| 127 | 63565 | 191375 | 
| 173 | 58 django/db/backends/base/operations.py | 189 | 220| 278 | 63843 | 196607 | 
| 174 | 58 django/contrib/admin/templatetags/admin_list.py | 428 | 486| 343 | 64186 | 196607 | 
| 175 | 58 django/contrib/admin/views/main.py | 397 | 428| 225 | 64411 | 196607 | 
| 176 | 58 django/contrib/admin/templatetags/admin_list.py | 301 | 351| 347 | 64758 | 196607 | 
| 177 | 58 django/contrib/admin/options.py | 2008 | 2054| 420 | 65178 | 196607 | 
| 178 | 58 django/contrib/admin/options.py | 1289 | 1305| 159 | 65337 | 196607 | 
| 179 | 58 django/db/models/options.py | 205 | 243| 362 | 65699 | 196607 | 
| 180 | 58 django/db/migrations/operations/models.py | 698 | 734| 225 | 65924 | 196607 | 
| 181 | 58 docs/_ext/djangodocs.py | 406 | 505| 825 | 66749 | 196607 | 
| 182 | 58 django/core/management/commands/loaddata.py | 300 | 346| 325 | 67074 | 196607 | 
| 183 | 59 django/core/serializers/__init__.py | 159 | 235| 690 | 67764 | 198269 | 
| 184 | 59 django/db/models/sql/query.py | 1770 | 1810| 304 | 68068 | 198269 | 
| 185 | 60 django/contrib/sessions/management/commands/clearsessions.py | 1 | 20| 122 | 68190 | 198391 | 
| 186 | 60 django/db/models/options.py | 245 | 277| 343 | 68533 | 198391 | 
| 187 | 61 django/contrib/admindocs/urls.py | 1 | 51| 307 | 68840 | 198698 | 
| 188 | 62 django/contrib/staticfiles/management/commands/runserver.py | 1 | 33| 252 | 69092 | 198951 | 
| 189 | 62 django/contrib/admin/options.py | 1307 | 1372| 581 | 69673 | 198951 | 


## Patch

```diff
diff --git a/django/core/management/base.py b/django/core/management/base.py
--- a/django/core/management/base.py
+++ b/django/core/management/base.py
@@ -228,6 +228,9 @@ def create_parser(self, prog_name, subcommand):
             self, prog="%s %s" % (os.path.basename(prog_name), subcommand),
             description=self.help or None,
         )
+        # Add command-specific arguments first so that they appear in the
+        # --help output before arguments common to all commands.
+        self.add_arguments(parser)
         parser.add_argument('--version', action='version', version=self.get_version())
         parser.add_argument(
             '-v', '--verbosity', action='store', dest='verbosity', default=1,
@@ -251,7 +254,6 @@ def create_parser(self, prog_name, subcommand):
             '--no-color', action='store_true', dest='no_color',
             help="Don't colorize the command output.",
         )
-        self.add_arguments(parser)
         return parser
 
     def add_arguments(self, parser):

```

## Test Patch

```diff
diff --git a/tests/admin_scripts/tests.py b/tests/admin_scripts/tests.py
--- a/tests/admin_scripts/tests.py
+++ b/tests/admin_scripts/tests.py
@@ -1495,6 +1495,13 @@ def test_specific_help(self):
         args = ['check', '--help']
         out, err = self.run_manage(args)
         self.assertNoOutput(err)
+        # Command-specific options like --tag appear before options common to
+        # all commands like --version.
+        tag_location = out.find('--tag')
+        version_location = out.find('--version')
+        self.assertNotEqual(tag_location, -1)
+        self.assertNotEqual(version_location, -1)
+        self.assertLess(tag_location, version_location)
         self.assertOutput(out, "Checks the entire Django project for potential problems.")
 
     def test_color_style(self):

```


## Code snippets

### 1 - django/core/management/__init__.py:

Start line: 151, End line: 193

```python
class ManagementUtility:
    """
    Encapsulate the logic of the django-admin and manage.py utilities.
    """
    def __init__(self, argv=None):
        self.argv = argv or sys.argv[:]
        self.prog_name = os.path.basename(self.argv[0])
        if self.prog_name == '__main__.py':
            self.prog_name = 'python -m django'
        self.settings_exception = None

    def main_help_text(self, commands_only=False):
        """Return the script's main help text, as a string."""
        if commands_only:
            usage = sorted(get_commands())
        else:
            usage = [
                "",
                "Type '%s help <subcommand>' for help on a specific subcommand." % self.prog_name,
                "",
                "Available subcommands:",
            ]
            commands_dict = defaultdict(lambda: [])
            for name, app in get_commands().items():
                if app == 'django.core':
                    app = 'django'
                else:
                    app = app.rpartition('.')[-1]
                commands_dict[app].append(name)
            style = color_style()
            for app in sorted(commands_dict):
                usage.append("")
                usage.append(style.NOTICE("[%s]" % app))
                for name in sorted(commands_dict[app]):
                    usage.append("    %s" % name)
            # Output an extra note if settings are not properly configured
            if self.settings_exception is not None:
                usage.append(style.NOTICE(
                    "Note that only Django core commands are listed "
                    "as settings are not properly configured (error: %s)."
                    % self.settings_exception))

        return '\n'.join(usage)
```
### 2 - django/core/management/base.py:

Start line: 491, End line: 523

```python
class LabelCommand(BaseCommand):
    """
    A management command which takes one or more arbitrary arguments
    (labels) on the command line, and does something with each of
    them.

    Rather than implementing ``handle()``, subclasses must implement
    ``handle_label()``, which will be called once for each label.

    If the arguments should be names of installed applications, use
    ``AppCommand`` instead.
    """
    label = 'label'
    missing_args_message = "Enter at least one %s." % label

    def add_arguments(self, parser):
        parser.add_argument('args', metavar=self.label, nargs='+')

    def handle(self, *labels, **options):
        output = []
        for label in labels:
            label_output = self.handle_label(label, **options)
            if label_output:
                output.append(label_output)
        return '\n'.join(output)

    def handle_label(self, label, **options):
        """
        Perform the command's actions for ``label``, which will be the
        string as given on the command line.
        """
        raise NotImplementedError('subclasses of LabelCommand must provide a handle_label() method')
```
### 3 - django/core/management/__init__.py:

Start line: 301, End line: 382

```python
class ManagementUtility:

    def execute(self):
        """
        Given the command-line arguments, figure out which subcommand is being
        run, create a parser appropriate to that command, and run it.
        """
        try:
            subcommand = self.argv[1]
        except IndexError:
            subcommand = 'help'  # Display help if no arguments were given.

        # Preprocess options to extract --settings and --pythonpath.
        # These options could affect the commands that are available, so they
        # must be processed early.
        parser = CommandParser(None, usage="%(prog)s subcommand [options] [args]", add_help=False)
        parser.add_argument('--settings')
        parser.add_argument('--pythonpath')
        parser.add_argument('args', nargs='*')  # catch-all
        try:
            options, args = parser.parse_known_args(self.argv[2:])
            handle_default_options(options)
        except CommandError:
            pass  # Ignore any option errors at this point.

        try:
            settings.INSTALLED_APPS
        except ImproperlyConfigured as exc:
            self.settings_exception = exc
        except ImportError as exc:
            self.settings_exception = exc

        if settings.configured:
            # Start the auto-reloading dev server even if the code is broken.
            # The hardcoded condition is a code smell but we can't rely on a
            # flag on the command class because we haven't located it yet.
            if subcommand == 'runserver' and '--noreload' not in self.argv:
                try:
                    autoreload.check_errors(django.setup)()
                except Exception:
                    # The exception will be raised later in the child process
                    # started by the autoreloader. Pretend it didn't happen by
                    # loading an empty list of applications.
                    apps.all_models = defaultdict(OrderedDict)
                    apps.app_configs = OrderedDict()
                    apps.apps_ready = apps.models_ready = apps.ready = True

                    # Remove options not compatible with the built-in runserver
                    # (e.g. options for the contrib.staticfiles' runserver).
                    # Changes here require manually testing as described in
                    # #27522.
                    _parser = self.fetch_command('runserver').create_parser('django', 'runserver')
                    _options, _args = _parser.parse_known_args(self.argv[2:])
                    for _arg in _args:
                        self.argv.remove(_arg)

            # In all other cases, django.setup() is required to succeed.
            else:
                django.setup()

        self.autocomplete()

        if subcommand == 'help':
            if '--commands' in args:
                sys.stdout.write(self.main_help_text(commands_only=True) + '\n')
            elif not options.args:
                sys.stdout.write(self.main_help_text() + '\n')
            else:
                self.fetch_command(options.args[0]).print_help(self.prog_name, options.args[0])
        # Special-cases: We want 'django-admin --version' and
        # 'django-admin --help' to work, for backwards compatibility.
        elif subcommand == 'version' or self.argv[1:] == ['--version']:
            sys.stdout.write(django.get_version() + '\n')
        elif self.argv[1:] in (['--help'], ['-h']):
            sys.stdout.write(self.main_help_text() + '\n')
        else:
            self.fetch_command(subcommand).run_from_argv(self.argv)


def execute_from_command_line(argv=None):
    """Run a ManagementUtility."""
    utility = ManagementUtility(argv)
    utility.execute()
```
### 4 - django/core/management/base.py:

Start line: 63, End line: 106

```python
def handle_default_options(options):
    """
    Include any default options that all commands should accept here
    so that ManagementUtility can handle them before searching for
    user commands.
    """
    if options.settings:
        os.environ['DJANGO_SETTINGS_MODULE'] = options.settings
    if options.pythonpath:
        sys.path.insert(0, options.pythonpath)


class OutputWrapper(TextIOBase):
    """
    Wrapper around stdout/stderr
    """
    @property
    def style_func(self):
        return self._style_func

    @style_func.setter
    def style_func(self, style_func):
        if style_func and self.isatty():
            self._style_func = style_func
        else:
            self._style_func = lambda x: x

    def __init__(self, out, style_func=None, ending='\n'):
        self._out = out
        self.style_func = None
        self.ending = ending

    def __getattr__(self, name):
        return getattr(self._out, name)

    def isatty(self):
        return hasattr(self._out, 'isatty') and self._out.isatty()

    def write(self, msg, style_func=None, ending=None):
        ending = self.ending if ending is None else ending
        if ending and not msg.endswith(ending):
            msg += ending
        style_func = style_func or self.style_func
        self._out.write(style_func(msg))
```
### 5 - django/core/management/__init__.py:

Start line: 227, End line: 299

```python
class ManagementUtility:

    def autocomplete(self):
        """
        Output completion suggestions for BASH.

        The output of this function is passed to BASH's `COMREPLY` variable and
        treated as completion suggestions. `COMREPLY` expects a space
        separated string as the result.

        The `COMP_WORDS` and `COMP_CWORD` BASH environment variables are used
        to get information about the cli input. Please refer to the BASH
        man-page for more information about this variables.

        Subcommand options are saved as pairs. A pair consists of
        the long option string (e.g. '--exclude') and a boolean
        value indicating if the option requires arguments. When printing to
        stdout, an equal sign is appended to options which require arguments.

        Note: If debugging this function, it is recommended to write the debug
        output in a separate file. Otherwise the debug output will be treated
        and formatted as potential completion suggestions.
        """
        # Don't complete if user hasn't sourced bash_completion file.
        if 'DJANGO_AUTO_COMPLETE' not in os.environ:
            return

        cwords = os.environ['COMP_WORDS'].split()[1:]
        cword = int(os.environ['COMP_CWORD'])

        try:
            curr = cwords[cword - 1]
        except IndexError:
            curr = ''

        subcommands = list(get_commands()) + ['help']
        options = [('--help', False)]

        # subcommand
        if cword == 1:
            print(' '.join(sorted(filter(lambda x: x.startswith(curr), subcommands))))
        # subcommand options
        # special case: the 'help' subcommand has no options
        elif cwords[0] in subcommands and cwords[0] != 'help':
            subcommand_cls = self.fetch_command(cwords[0])
            # special case: add the names of installed apps to options
            if cwords[0] in ('dumpdata', 'sqlmigrate', 'sqlsequencereset', 'test'):
                try:
                    app_configs = apps.get_app_configs()
                    # Get the last part of the dotted path as the app name.
                    options.extend((app_config.label, 0) for app_config in app_configs)
                except ImportError:
                    # Fail silently if DJANGO_SETTINGS_MODULE isn't set. The
                    # user will find out once they execute the command.
                    pass
            parser = subcommand_cls.create_parser('', cwords[0])
            options.extend(
                (min(s_opt.option_strings), s_opt.nargs != 0)
                for s_opt in parser._actions if s_opt.option_strings
            )
            # filter out previously specified options from available options
            prev_opts = {x.split('=')[0] for x in cwords[1:cword - 1]}
            options = (opt for opt in options if opt[0] not in prev_opts)

            # filter options by current input
            options = sorted((k, v) for k, v in options if k.startswith(curr))
            for opt_label, require_arg in options:
                # append '=' to options which require args
                if require_arg:
                    opt_label += '='
                print(opt_label)
        # Exit code of the bash completion function is never passed back to
        # the user, so it's safe to always exit with 0.
        # For more details see #25420.
        sys.exit(0)
```
### 6 - django/core/management/commands/dumpdata.py:

Start line: 68, End line: 141

```python
class Command(BaseCommand):

    def handle(self, *app_labels, **options):
        format = options['format']
        indent = options['indent']
        using = options['database']
        excludes = options['exclude']
        output = options['output']
        show_traceback = options['traceback']
        use_natural_foreign_keys = options['use_natural_foreign_keys']
        use_natural_primary_keys = options['use_natural_primary_keys']
        use_base_manager = options['use_base_manager']
        pks = options['primary_keys']

        if pks:
            primary_keys = [pk.strip() for pk in pks.split(',')]
        else:
            primary_keys = []

        excluded_models, excluded_apps = parse_apps_and_model_labels(excludes)

        if not app_labels:
            if primary_keys:
                raise CommandError("You can only use --pks option with one model")
            app_list = OrderedDict.fromkeys(
                app_config for app_config in apps.get_app_configs()
                if app_config.models_module is not None and app_config not in excluded_apps
            )
        else:
            if len(app_labels) > 1 and primary_keys:
                raise CommandError("You can only use --pks option with one model")
            app_list = OrderedDict()
            for label in app_labels:
                try:
                    app_label, model_label = label.split('.')
                    try:
                        app_config = apps.get_app_config(app_label)
                    except LookupError as e:
                        raise CommandError(str(e))
                    if app_config.models_module is None or app_config in excluded_apps:
                        continue
                    try:
                        model = app_config.get_model(model_label)
                    except LookupError:
                        raise CommandError("Unknown model: %s.%s" % (app_label, model_label))

                    app_list_value = app_list.setdefault(app_config, [])

                    # We may have previously seen a "all-models" request for
                    # this app (no model qualifier was given). In this case
                    # there is no need adding specific models to the list.
                    if app_list_value is not None:
                        if model not in app_list_value:
                            app_list_value.append(model)
                except ValueError:
                    if primary_keys:
                        raise CommandError("You can only use --pks option with one model")
                    # This is just an app - no model qualifier
                    app_label = label
                    try:
                        app_config = apps.get_app_config(app_label)
                    except LookupError as e:
                        raise CommandError(str(e))
                    if app_config.models_module is None or app_config in excluded_apps:
                        continue
                    app_list[app_config] = None

        # Check that the serialization format exists; this is a shortcut to
        # avoid collating all the objects and _then_ failing.
        if format not in serializers.get_public_serializer_formats():
            try:
                serializers.get_serializer(format)
            except serializers.SerializerDoesNotExist:
                pass

            raise CommandError("Unknown serialization format: %s" % format)
        # ... other code
```
### 7 - django/core/management/commands/makemessages.py:

Start line: 364, End line: 394

```python
class Command(BaseCommand):

    def handle(self, *args, **options):
        # ... other code
        looks_like_locale = re.compile(r'[a-z]{2}')
        locale_dirs = filter(os.path.isdir, glob.glob('%s/*' % self.default_locale_path))
        all_locales = [
            lang_code for lang_code in map(os.path.basename, locale_dirs)
            if looks_like_locale.match(lang_code)
        ]

        # Account for excluded locales
        if process_all:
            locales = all_locales
        else:
            locales = locale or all_locales
            locales = set(locales).difference(exclude)

        if locales:
            check_programs('msguniq', 'msgmerge', 'msgattrib')

        check_programs('xgettext')

        try:
            potfiles = self.build_potfiles()

            # Build po files for each selected locale
            for locale in locales:
                if self.verbosity > 0:
                    self.stdout.write("processing locale %s\n" % locale)
                for potfile in potfiles:
                    self.write_po_file(potfile, locale)
        finally:
            if not self.keep_pot:
                self.remove_potfiles()
```
### 8 - django/core/management/commands/makemessages.py:

Start line: 284, End line: 363

```python
class Command(BaseCommand):

    def handle(self, *args, **options):
        locale = options['locale']
        exclude = options['exclude']
        self.domain = options['domain']
        self.verbosity = options['verbosity']
        process_all = options['all']
        extensions = options['extensions']
        self.symlinks = options['symlinks']

        ignore_patterns = options['ignore_patterns']
        if options['use_default_ignore_patterns']:
            ignore_patterns += ['CVS', '.*', '*~', '*.pyc']
        self.ignore_patterns = list(set(ignore_patterns))

        # Avoid messing with mutable class variables
        if options['no_wrap']:
            self.msgmerge_options = self.msgmerge_options[:] + ['--no-wrap']
            self.msguniq_options = self.msguniq_options[:] + ['--no-wrap']
            self.msgattrib_options = self.msgattrib_options[:] + ['--no-wrap']
            self.xgettext_options = self.xgettext_options[:] + ['--no-wrap']
        if options['no_location']:
            self.msgmerge_options = self.msgmerge_options[:] + ['--no-location']
            self.msguniq_options = self.msguniq_options[:] + ['--no-location']
            self.msgattrib_options = self.msgattrib_options[:] + ['--no-location']
            self.xgettext_options = self.xgettext_options[:] + ['--no-location']
        if options['add_location']:
            if self.gettext_version < (0, 19):
                raise CommandError(
                    "The --add-location option requires gettext 0.19 or later. "
                    "You have %s." % '.'.join(str(x) for x in self.gettext_version)
                )
            arg_add_location = "--add-location=%s" % options['add_location']
            self.msgmerge_options = self.msgmerge_options[:] + [arg_add_location]
            self.msguniq_options = self.msguniq_options[:] + [arg_add_location]
            self.msgattrib_options = self.msgattrib_options[:] + [arg_add_location]
            self.xgettext_options = self.xgettext_options[:] + [arg_add_location]

        self.no_obsolete = options['no_obsolete']
        self.keep_pot = options['keep_pot']

        if self.domain not in ('django', 'djangojs'):
            raise CommandError("currently makemessages only supports domains "
                               "'django' and 'djangojs'")
        if self.domain == 'djangojs':
            exts = extensions or ['js']
        else:
            exts = extensions or ['html', 'txt', 'py']
        self.extensions = handle_extensions(exts)

        if (locale is None and not exclude and not process_all) or self.domain is None:
            raise CommandError(
                "Type '%s help %s' for usage information."
                % (os.path.basename(sys.argv[0]), sys.argv[1])
            )

        if self.verbosity > 1:
            self.stdout.write(
                'examining files with the extensions: %s\n'
                % get_text_list(list(self.extensions), 'and')
            )

        self.invoked_for_django = False
        self.locale_paths = []
        self.default_locale_path = None
        if os.path.isdir(os.path.join('conf', 'locale')):
            self.locale_paths = [os.path.abspath(os.path.join('conf', 'locale'))]
            self.default_locale_path = self.locale_paths[0]
            self.invoked_for_django = True
        else:
            if self.settings_available:
                self.locale_paths.extend(settings.LOCALE_PATHS)
            # Allow to run makemessages inside an app dir
            if os.path.isdir('locale'):
                self.locale_paths.append(os.path.abspath('locale'))
            if self.locale_paths:
                self.default_locale_path = self.locale_paths[0]
                if not os.path.exists(self.default_locale_path):
                    os.makedirs(self.default_locale_path)

        # Build locale list
        # ... other code
```
### 9 - django/core/management/base.py:

Start line: 257, End line: 305

```python
class BaseCommand:

    def add_arguments(self, parser):
        """
        Entry point for subclassed commands to add custom arguments.
        """
        pass

    def print_help(self, prog_name, subcommand):
        """
        Print the help message for this command, derived from
        ``self.usage()``.
        """
        parser = self.create_parser(prog_name, subcommand)
        parser.print_help()

    def run_from_argv(self, argv):
        """
        Set up any environment changes requested (e.g., Python path
        and Django settings), then run this command. If the
        command raises a ``CommandError``, intercept it and print it sensibly
        to stderr. If the ``--traceback`` option is present or the raised
        ``Exception`` is not ``CommandError``, raise it.
        """
        self._called_from_command_line = True
        parser = self.create_parser(argv[0], argv[1])

        options = parser.parse_args(argv[2:])
        cmd_options = vars(options)
        # Move positional args out of options to mimic legacy optparse
        args = cmd_options.pop('args', ())
        handle_default_options(options)
        try:
            self.execute(*args, **cmd_options)
        except Exception as e:
            if options.traceback or not isinstance(e, CommandError):
                raise

            # SystemCheckError takes care of its own formatting.
            if isinstance(e, SystemCheckError):
                self.stderr.write(str(e), lambda x: x)
            else:
                self.stderr.write('%s: %s' % (e.__class__.__name__, e))
            sys.exit(1)
        finally:
            try:
                connections.close_all()
            except ImproperlyConfigured:
                # Ignore if connections aren't setup at this point (e.g. no
                # configured settings).
                pass
```
### 10 - django/core/management/commands/makemessages.py:

Start line: 217, End line: 282

```python
class Command(BaseCommand):

    def add_arguments(self, parser):
        parser.add_argument(
            '--locale', '-l', default=[], dest='locale', action='append',
            help='Creates or updates the message files for the given locale(s) (e.g. pt_BR). '
                 'Can be used multiple times.',
        )
        parser.add_argument(
            '--exclude', '-x', default=[], dest='exclude', action='append',
            help='Locales to exclude. Default is none. Can be used multiple times.',
        )
        parser.add_argument(
            '--domain', '-d', default='django', dest='domain',
            help='The domain of the message files (default: "django").',
        )
        parser.add_argument(
            '--all', '-a', action='store_true', dest='all',
            help='Updates the message files for all existing locales.',
        )
        parser.add_argument(
            '--extension', '-e', dest='extensions', action='append',
            help='The file extension(s) to examine (default: "html,txt,py", or "js" '
                 'if the domain is "djangojs"). Separate multiple extensions with '
                 'commas, or use -e multiple times.',
        )
        parser.add_argument(
            '--symlinks', '-s', action='store_true', dest='symlinks',
            help='Follows symlinks to directories when examining source code '
                 'and templates for translation strings.',
        )
        parser.add_argument(
            '--ignore', '-i', action='append', dest='ignore_patterns',
            default=[], metavar='PATTERN',
            help='Ignore files or directories matching this glob-style pattern. '
                 'Use multiple times to ignore more.',
        )
        parser.add_argument(
            '--no-default-ignore', action='store_false', dest='use_default_ignore_patterns',
            help="Don't ignore the common glob-style patterns 'CVS', '.*', '*~' and '*.pyc'.",
        )
        parser.add_argument(
            '--no-wrap', action='store_true', dest='no_wrap',
            help="Don't break long message lines into several lines.",
        )
        parser.add_argument(
            '--no-location', action='store_true', dest='no_location',
            help="Don't write '#: filename:line' lines.",
        )
        parser.add_argument(
            '--add-location', dest='add_location',
            choices=('full', 'file', 'never'), const='full', nargs='?',
            help=(
                "Controls '#: filename:line' lines. If the option is 'full' "
                "(the default if not given), the lines  include both file name "
                "and line number. If it's 'file', the line number is omitted. If "
                "it's 'never', the lines are suppressed (same as --no-location). "
                "--add-location requires gettext 0.19 or newer."
            ),
        )
        parser.add_argument(
            '--no-obsolete', action='store_true', dest='no_obsolete',
            help="Remove obsolete message strings.",
        )
        parser.add_argument(
            '--keep-pot', action='store_true', dest='keep_pot',
            help="Keep .pot file after making messages. Useful when debugging.",
        )
```
### 12 - django/core/management/base.py:

Start line: 109, End line: 189

```python
class BaseCommand:
    """
    The base class from which all management commands ultimately
    derive.

    Use this class if you want access to all of the mechanisms which
    parse the command-line arguments and work out what code to call in
    response; if you don't need to change any of that behavior,
    consider using one of the subclasses defined in this file.

    If you are interested in overriding/customizing various aspects of
    the command-parsing and -execution behavior, the normal flow works
    as follows:

    1. ``django-admin`` or ``manage.py`` loads the command class
       and calls its ``run_from_argv()`` method.

    2. The ``run_from_argv()`` method calls ``create_parser()`` to get
       an ``ArgumentParser`` for the arguments, parses them, performs
       any environment changes requested by options like
       ``pythonpath``, and then calls the ``execute()`` method,
       passing the parsed arguments.

    3. The ``execute()`` method attempts to carry out the command by
       calling the ``handle()`` method with the parsed arguments; any
       output produced by ``handle()`` will be printed to standard
       output and, if the command is intended to produce a block of
       SQL statements, will be wrapped in ``BEGIN`` and ``COMMIT``.

    4. If ``handle()`` or ``execute()`` raised any exception (e.g.
       ``CommandError``), ``run_from_argv()`` will  instead print an error
       message to ``stderr``.

    Thus, the ``handle()`` method is typically the starting point for
    subclasses; many built-in commands and command types either place
    all of their logic in ``handle()``, or perform some additional
    parsing work in ``handle()`` and then delegate from it to more
    specialized methods as needed.

    Several attributes affect behavior at various steps along the way:

    ``help``
        A short description of the command, which will be printed in
        help messages.

    ``output_transaction``
        A boolean indicating whether the command outputs SQL
        statements; if ``True``, the output will automatically be
        wrapped with ``BEGIN;`` and ``COMMIT;``. Default value is
        ``False``.

    ``requires_migrations_checks``
        A boolean; if ``True``, the command prints a warning if the set of
        migrations on disk don't match the migrations in the database.

    ``requires_system_checks``
        A boolean; if ``True``, entire Django project will be checked for errors
        prior to executing the command. Default value is ``True``.
        To validate an individual application's models
        rather than all applications' models, call
        ``self.check(app_configs)`` from ``handle()``, where ``app_configs``
        is the list of application's configuration provided by the
        app registry.

    ``leave_locale_alone``
        A boolean indicating whether the locale set in settings should be
        preserved during the execution of the command instead of translations
        being deactivated.

        Default value is ``False``.

        Make sure you know what you are doing if you decide to change the value
        of this option in your custom command if it creates database content
        that is locale-sensitive and such content shouldn't contain any
        translations (like it happens e.g. with django.contrib.auth
        permissions) as activating any locale might cause unintended effects.

    ``stealth_options``
        A tuple of any options the command uses which aren't defined by the
        argument parser.
    """
```
### 17 - django/core/management/base.py:

Start line: 455, End line: 488

```python
class AppCommand(BaseCommand):
    """
    A management command which takes one or more installed application labels
    as arguments, and does something with each of them.

    Rather than implementing ``handle()``, subclasses must implement
    ``handle_app_config()``, which will be called once for each application.
    """
    missing_args_message = "Enter at least one application label."

    def add_arguments(self, parser):
        parser.add_argument('args', metavar='app_label', nargs='+', help='One or more application label.')

    def handle(self, *app_labels, **options):
        from django.apps import apps
        try:
            app_configs = [apps.get_app_config(app_label) for app_label in app_labels]
        except (LookupError, ImportError) as e:
            raise CommandError("%s. Are you sure your INSTALLED_APPS setting is correct?" % e)
        output = []
        for app_config in app_configs:
            app_output = self.handle_app_config(app_config, **options)
            if app_output:
                output.append(app_output)
        return '\n'.join(output)

    def handle_app_config(self, app_config, **options):
        """
        Perform the command's actions for app_config, an AppConfig instance
        corresponding to an application label given on the command line.
        """
        raise NotImplementedError(
            "Subclasses of AppCommand must provide"
            "a handle_app_config() method.")
```
### 21 - django/core/management/base.py:

Start line: 190, End line: 220

```python
class BaseCommand:
    # Metadata about this command.
    help = ''

    # Configuration shortcuts that alter various logic.
    _called_from_command_line = False
    output_transaction = False  # Whether to wrap the output in a "BEGIN; COMMIT;"
    leave_locale_alone = False
    requires_migrations_checks = False
    requires_system_checks = True
    # Arguments, common to all commands, which aren't defined by the argument
    # parser.
    base_stealth_options = ('skip_checks', 'stderr', 'stdout')
    # Command-specific options not defined by the argument parser.
    stealth_options = ()

    def __init__(self, stdout=None, stderr=None, no_color=False):
        self.stdout = OutputWrapper(stdout or sys.stdout)
        self.stderr = OutputWrapper(stderr or sys.stderr)
        if no_color:
            self.style = no_style()
        else:
            self.style = color_style()
            self.stderr.style_func = self.style.ERROR

    def get_version(self):
        """
        Return the Django version, which should be correct for all built-in
        Django commands. User-supplied commands can override this method to
        return their own version.
        """
        return django.get_version()
```
### 42 - django/core/management/base.py:

Start line: 1, End line: 29

```python
"""
Base classes for writing management commands (named commands which can
be executed through ``django-admin`` or ``manage.py``).
"""
import os
import sys
from argparse import ArgumentParser
from io import TextIOBase

import django
from django.core import checks
from django.core.exceptions import ImproperlyConfigured
from django.core.management.color import color_style, no_style
from django.db import DEFAULT_DB_ALIAS, connections


class CommandError(Exception):
    """
    Exception class indicating a problem while executing a management
    command.

    If this exception is raised during the execution of a management
    command, it will be caught and turned into a nicely-printed error
    message to the appropriate output stream (i.e., stderr); as a
    result, raising this exception (with a sensible description of the
    error) is the preferred way to indicate that something has gone
    wrong in the execution of a command.
    """
    pass
```
### 94 - django/core/management/base.py:

Start line: 222, End line: 255

```python
class BaseCommand:

    def create_parser(self, prog_name, subcommand):
        """
        Create and return the ``ArgumentParser`` which will be used to
        parse the arguments to this command.
        """
        parser = CommandParser(
            self, prog="%s %s" % (os.path.basename(prog_name), subcommand),
            description=self.help or None,
        )
        parser.add_argument('--version', action='version', version=self.get_version())
        parser.add_argument(
            '-v', '--verbosity', action='store', dest='verbosity', default=1,
            type=int, choices=[0, 1, 2, 3],
            help='Verbosity level; 0=minimal output, 1=normal output, 2=verbose output, 3=very verbose output',
        )
        parser.add_argument(
            '--settings',
            help=(
                'The Python path to a settings module, e.g. '
                '"myproject.settings.main". If this isn\'t provided, the '
                'DJANGO_SETTINGS_MODULE environment variable will be used.'
            ),
        )
        parser.add_argument(
            '--pythonpath',
            help='A directory to add to the Python path, e.g. "/home/djangoprojects/myproject".',
        )
        parser.add_argument('--traceback', action='store_true', help='Raise on CommandError exceptions')
        parser.add_argument(
            '--no-color', action='store_true', dest='no_color',
            help="Don't colorize the command output.",
        )
        self.add_arguments(parser)
        return parser
```
### 117 - django/core/management/base.py:

Start line: 32, End line: 60

```python
class SystemCheckError(CommandError):
    """
    The system check framework detected unrecoverable errors.
    """
    pass


class CommandParser(ArgumentParser):
    """
    Customized ArgumentParser class to improve some error messages and prevent
    SystemExit in several occasions, as SystemExit is unacceptable when a
    command is called programmatically.
    """
    def __init__(self, cmd, **kwargs):
        self.cmd = cmd
        super().__init__(**kwargs)

    def parse_args(self, args=None, namespace=None):
        # Catch missing argument for a better error message
        if (hasattr(self.cmd, 'missing_args_message') and
                not (args or any(not arg.startswith('-') for arg in args))):
            self.error(self.cmd.missing_args_message)
        return super().parse_args(args, namespace)

    def error(self, message):
        if self.cmd._called_from_command_line:
            super().error(message)
        else:
            raise CommandError("Error: %s" % message)
```
### 134 - django/core/management/base.py:

Start line: 353, End line: 418

```python
class BaseCommand:

    def check(self, app_configs=None, tags=None, display_num_errors=False,
              include_deployment_checks=False, fail_level=checks.ERROR):
        """
        Use the system check framework to validate entire Django project.
        Raise CommandError for any serious message (error or critical errors).
        If there are only light messages (like warnings), print them to stderr
        and don't raise an exception.
        """
        all_issues = self._run_checks(
            app_configs=app_configs,
            tags=tags,
            include_deployment_checks=include_deployment_checks,
        )

        header, body, footer = "", "", ""
        visible_issue_count = 0  # excludes silenced warnings

        if all_issues:
            debugs = [e for e in all_issues if e.level < checks.INFO and not e.is_silenced()]
            infos = [e for e in all_issues if checks.INFO <= e.level < checks.WARNING and not e.is_silenced()]
            warnings = [e for e in all_issues if checks.WARNING <= e.level < checks.ERROR and not e.is_silenced()]
            errors = [e for e in all_issues if checks.ERROR <= e.level < checks.CRITICAL and not e.is_silenced()]
            criticals = [e for e in all_issues if checks.CRITICAL <= e.level and not e.is_silenced()]
            sorted_issues = [
                (criticals, 'CRITICALS'),
                (errors, 'ERRORS'),
                (warnings, 'WARNINGS'),
                (infos, 'INFOS'),
                (debugs, 'DEBUGS'),
            ]

            for issues, group_name in sorted_issues:
                if issues:
                    visible_issue_count += len(issues)
                    formatted = (
                        self.style.ERROR(str(e))
                        if e.is_serious()
                        else self.style.WARNING(str(e))
                        for e in issues)
                    formatted = "\n".join(sorted(formatted))
                    body += '\n%s:\n%s\n' % (group_name, formatted)

        if visible_issue_count:
            header = "System check identified some issues:\n"

        if display_num_errors:
            if visible_issue_count:
                footer += '\n'
            footer += "System check identified %s (%s silenced)." % (
                "no issues" if visible_issue_count == 0 else
                "1 issue" if visible_issue_count == 1 else
                "%s issues" % visible_issue_count,
                len(all_issues) - visible_issue_count,
            )

        if any(e.is_serious(fail_level) and not e.is_silenced() for e in all_issues):
            msg = self.style.ERROR("SystemCheckError: %s" % header) + body + footer
            raise SystemCheckError(msg)
        else:
            msg = header + body + footer

        if msg:
            if visible_issue_count:
                self.stderr.write(msg, lambda x: x)
            else:
                self.stdout.write(msg)
```
### 144 - django/core/management/base.py:

Start line: 307, End line: 351

```python
class BaseCommand:

    def execute(self, *args, **options):
        """
        Try to execute this command, performing system checks if needed (as
        controlled by the ``requires_system_checks`` attribute, except if
        force-skipped).
        """
        if options['no_color']:
            self.style = no_style()
            self.stderr.style_func = None
        if options.get('stdout'):
            self.stdout = OutputWrapper(options['stdout'])
        if options.get('stderr'):
            self.stderr = OutputWrapper(options['stderr'], self.stderr.style_func)

        saved_locale = None
        if not self.leave_locale_alone:
            # Deactivate translations, because django-admin creates database
            # content like permissions, and those shouldn't contain any
            # translations.
            from django.utils import translation
            saved_locale = translation.get_language()
            translation.deactivate_all()

        try:
            if self.requires_system_checks and not options.get('skip_checks'):
                self.check()
            if self.requires_migrations_checks:
                self.check_migrations()
            output = self.handle(*args, **options)
            if output:
                if self.output_transaction:
                    connection = connections[options.get('database', DEFAULT_DB_ALIAS)]
                    output = '%s\n%s\n%s' % (
                        self.style.SQL_KEYWORD(connection.ops.start_transaction_sql()),
                        output,
                        self.style.SQL_KEYWORD(connection.ops.end_transaction_sql()),
                    )
                self.stdout.write(output)
        finally:
            if saved_locale is not None:
                translation.activate(saved_locale)
        return output

    def _run_checks(self, **kwargs):
        return checks.run_checks(**kwargs)
```
### 161 - django/core/management/base.py:

Start line: 420, End line: 452

```python
class BaseCommand:

    def check_migrations(self):
        """
        Print a warning if the set of migrations on disk don't match the
        migrations in the database.
        """
        from django.db.migrations.executor import MigrationExecutor
        try:
            executor = MigrationExecutor(connections[DEFAULT_DB_ALIAS])
        except ImproperlyConfigured:
            # No databases are configured (or the dummy one)
            return

        plan = executor.migration_plan(executor.loader.graph.leaf_nodes())
        if plan:
            apps_waiting_migration = sorted({migration.app_label for migration, backwards in plan})
            self.stdout.write(
                self.style.NOTICE(
                    "\nYou have %(unpplied_migration_count)s unapplied migration(s). "
                    "Your project may not work properly until you apply the "
                    "migrations for app(s): %(apps_waiting_migration)s." % {
                        "unpplied_migration_count": len(plan),
                        "apps_waiting_migration": ", ".join(apps_waiting_migration),
                    }
                )
            )
            self.stdout.write(self.style.NOTICE("Run 'python manage.py migrate' to apply them.\n"))

    def handle(self, *args, **options):
        """
        The actual logic of the command. Subclasses must implement
        this method.
        """
        raise NotImplementedError('subclasses of BaseCommand must provide a handle() method')
```
