# django__django-14751

| **django/django** | `274771df9133542df048cc104c19e7756f9d3715` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 2755 |
| **Any found context length** | 644 |
| **Avg pos** | 22.0 |
| **Min pos** | 2 |
| **Max pos** | 5 |
| **Top file pos** | 1 |
| **Missing snippets** | 6 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/core/management/commands/makemigrations.py b/django/core/management/commands/makemigrations.py
--- a/django/core/management/commands/makemigrations.py
+++ b/django/core/management/commands/makemigrations.py
@@ -57,9 +57,20 @@ def add_arguments(self, parser):
             '--check', action='store_true', dest='check_changes',
             help='Exit with a non-zero status if model changes are missing migrations.',
         )
+        parser.add_argument(
+            '--scriptable', action='store_true', dest='scriptable',
+            help=(
+                'Divert log output and input prompts to stderr, writing only '
+                'paths of generated migration files to stdout.'
+            ),
+        )
+
+    @property
+    def log_output(self):
+        return self.stderr if self.scriptable else self.stdout
 
     def log(self, msg):
-        self.stdout.write(msg)
+        self.log_output.write(msg)
 
     @no_translations
     def handle(self, *app_labels, **options):
@@ -73,6 +84,10 @@ def handle(self, *app_labels, **options):
             raise CommandError('The migration name must be a valid Python identifier.')
         self.include_header = options['include_header']
         check_changes = options['check_changes']
+        self.scriptable = options['scriptable']
+        # If logs and prompts are diverted to stderr, remove the ERROR style.
+        if self.scriptable:
+            self.stderr.style_func = None
 
         # Make sure the app they asked for exists
         app_labels = set(app_labels)
@@ -147,7 +162,7 @@ def handle(self, *app_labels, **options):
             questioner = InteractiveMigrationQuestioner(
                 specified_apps=app_labels,
                 dry_run=self.dry_run,
-                prompt_output=self.stdout,
+                prompt_output=self.log_output,
             )
         else:
             questioner = NonInteractiveMigrationQuestioner(
@@ -226,6 +241,8 @@ def write_migration_files(self, changes):
                     self.log('  %s\n' % self.style.MIGRATE_LABEL(migration_string))
                     for operation in migration.operations:
                         self.log('    - %s' % operation.describe())
+                    if self.scriptable:
+                        self.stdout.write(migration_string)
                 if not self.dry_run:
                     # Write the migrations file to the disk.
                     migrations_directory = os.path.dirname(writer.path)
@@ -254,7 +271,7 @@ def handle_merge(self, loader, conflicts):
         if it's safe; otherwise, advises on how to fix it.
         """
         if self.interactive:
-            questioner = InteractiveMigrationQuestioner(prompt_output=self.stdout)
+            questioner = InteractiveMigrationQuestioner(prompt_output=self.log_output)
         else:
             questioner = MigrationQuestioner(defaults={'ask_merge': True})
 
@@ -327,6 +344,8 @@ def all_items_equal(seq):
                         fh.write(writer.as_string())
                     if self.verbosity > 0:
                         self.log('\nCreated new merge migration %s' % writer.path)
+                        if self.scriptable:
+                            self.stdout.write(writer.path)
                 elif self.verbosity == 3:
                     # Alternatively, makemigrations --merge --dry-run --verbosity 3
                     # will log the merge migrations rather than saving the file

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/core/management/commands/makemigrations.py | 60 | 60 | 2 | 1 | 644
| django/core/management/commands/makemigrations.py | 76 | 76 | 3 | 1 | 1459
| django/core/management/commands/makemigrations.py | 150 | 150 | 3 | 1 | 1459
| django/core/management/commands/makemigrations.py | 229 | 229 | 4 | 1 | 1886
| django/core/management/commands/makemigrations.py | 257 | 257 | 5 | 1 | 2755
| django/core/management/commands/makemigrations.py | 330 | 330 | 5 | 1 | 2755


## Problem Statement

```
Make makemigrations scriptable / script-friendly
Description
	
Currently, the makemigrations management command doesn't lend itself well to scripting. For example, it writes its progress output to stdout rather than stderr. Also, there doesn't appear to be a structured / programmatic way to figure out what files it has created.
My use case is that in my development environment, I'd like to be able to run makemigrations in a Docker container, find out what files were added (e.g. from makemigrations's output), and then copy those files from the Docker container to my development machine so they can be added to source control.
Currently, there doesn't seem to be an easy way to do this. One way, for example, is to manually read makemigrations's output to find out what apps were affected, and then inspect the directories yourself for the new files.
Better, for example, would be if makemigrations could write the paths to the created files to stdout.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 django/core/management/commands/makemigrations.py** | 160 | 204| 347 | 347 | 2862 | 
| **-> 2 <-** | **1 django/core/management/commands/makemigrations.py** | 24 | 62| 297 | 644 | 2862 | 
| **-> 3 <-** | **1 django/core/management/commands/makemigrations.py** | 64 | 159| 815 | 1459 | 2862 | 
| **-> 4 <-** | **1 django/core/management/commands/makemigrations.py** | 206 | 249| 427 | 1886 | 2862 | 
| **-> 5 <-** | **1 django/core/management/commands/makemigrations.py** | 251 | 338| 869 | 2755 | 2862 | 
| 6 | **1 django/core/management/commands/makemigrations.py** | 1 | 21| 155 | 2910 | 2862 | 
| 7 | 2 django/core/management/commands/migrate.py | 162 | 227| 632 | 3542 | 6199 | 
| 8 | 2 django/core/management/commands/migrate.py | 71 | 160| 774 | 4316 | 6199 | 
| 9 | 2 django/core/management/commands/migrate.py | 228 | 279| 537 | 4853 | 6199 | 
| 10 | 3 django/core/management/commands/showmigrations.py | 46 | 67| 158 | 5011 | 7469 | 
| 11 | 4 django/core/management/commands/squashmigrations.py | 138 | 210| 686 | 5697 | 9377 | 
| 12 | 5 django/core/management/commands/sqlmigrate.py | 31 | 69| 379 | 6076 | 10010 | 
| 13 | 5 django/core/management/commands/migrate.py | 281 | 313| 349 | 6425 | 10010 | 
| 14 | 5 django/core/management/commands/squashmigrations.py | 47 | 136| 791 | 7216 | 10010 | 
| 15 | 5 django/core/management/commands/showmigrations.py | 69 | 113| 476 | 7692 | 10010 | 
| 16 | 5 django/core/management/commands/sqlmigrate.py | 1 | 29| 259 | 7951 | 10010 | 
| 17 | 5 django/core/management/commands/migrate.py | 21 | 69| 407 | 8358 | 10010 | 
| 18 | 5 django/core/management/commands/showmigrations.py | 1 | 44| 311 | 8669 | 10010 | 
| 19 | 6 django/db/migrations/writer.py | 118 | 199| 744 | 9413 | 12257 | 
| 20 | 7 django/db/migrations/questioner.py | 1 | 55| 469 | 9882 | 14911 | 
| 21 | 7 django/core/management/commands/squashmigrations.py | 1 | 45| 353 | 10235 | 14911 | 
| 22 | 8 django/db/migrations/autodetector.py | 356 | 370| 138 | 10373 | 27026 | 
| 23 | 8 django/db/migrations/questioner.py | 269 | 314| 361 | 10734 | 27026 | 
| 24 | 8 django/core/management/commands/showmigrations.py | 115 | 158| 340 | 11074 | 27026 | 
| 25 | 8 django/core/management/commands/squashmigrations.py | 212 | 225| 112 | 11186 | 27026 | 
| 26 | 9 django/db/migrations/executor.py | 251 | 274| 225 | 11411 | 30424 | 
| 27 | 9 django/db/migrations/executor.py | 1 | 71| 571 | 11982 | 30424 | 
| 28 | 9 django/db/migrations/autodetector.py | 37 | 47| 120 | 12102 | 30424 | 
| 29 | 10 django/db/migrations/loader.py | 210 | 289| 783 | 12885 | 33532 | 
| 30 | 10 django/db/migrations/autodetector.py | 262 | 333| 748 | 13633 | 33532 | 
| 31 | 10 django/db/migrations/writer.py | 201 | 301| 619 | 14252 | 33532 | 
| 32 | 11 django/core/management/base.py | 488 | 520| 281 | 14533 | 38207 | 
| 33 | 11 django/db/migrations/loader.py | 1 | 53| 409 | 14942 | 38207 | 
| 34 | 11 django/core/management/commands/migrate.py | 1 | 18| 140 | 15082 | 38207 | 
| 35 | 11 django/db/migrations/autodetector.py | 103 | 201| 817 | 15899 | 38207 | 
| 36 | 11 django/db/migrations/autodetector.py | 1359 | 1382| 240 | 16139 | 38207 | 
| 37 | 12 django/db/migrations/recorder.py | 46 | 97| 390 | 16529 | 38884 | 
| 38 | 12 django/db/migrations/questioner.py | 90 | 103| 157 | 16686 | 38884 | 
| 39 | 12 django/core/management/commands/migrate.py | 315 | 362| 396 | 17082 | 38884 | 
| 40 | 12 django/db/migrations/executor.py | 226 | 249| 203 | 17285 | 38884 | 
| 41 | 13 django/contrib/admin/migrations/0001_initial.py | 1 | 47| 314 | 17599 | 39198 | 
| 42 | 13 django/db/migrations/executor.py | 293 | 386| 843 | 18442 | 39198 | 
| 43 | 13 django/core/management/commands/migrate.py | 364 | 387| 186 | 18628 | 39198 | 
| 44 | 14 django/core/management/sql.py | 38 | 54| 128 | 18756 | 39554 | 
| 45 | 14 django/db/migrations/autodetector.py | 1309 | 1357| 436 | 19192 | 39554 | 
| 46 | 14 django/db/migrations/autodetector.py | 248 | 261| 178 | 19370 | 39554 | 
| 47 | 15 django/contrib/admin/migrations/0003_logentry_add_action_flag_choices.py | 1 | 21| 111 | 19481 | 39665 | 
| 48 | 15 django/db/migrations/autodetector.py | 335 | 354| 196 | 19677 | 39665 | 
| 49 | 16 django/core/management/commands/makemessages.py | 284 | 363| 814 | 20491 | 45319 | 
| 50 | 16 django/db/migrations/autodetector.py | 1171 | 1213| 320 | 20811 | 45319 | 
| 51 | 16 django/core/management/commands/makemessages.py | 364 | 401| 272 | 21083 | 45319 | 
| 52 | 16 django/db/migrations/executor.py | 73 | 89| 167 | 21250 | 45319 | 
| 53 | 16 django/db/migrations/recorder.py | 1 | 21| 148 | 21398 | 45319 | 
| 54 | 16 django/db/migrations/writer.py | 2 | 115| 886 | 22284 | 45319 | 
| 55 | 16 django/db/migrations/autodetector.py | 1232 | 1266| 296 | 22580 | 45319 | 
| 56 | 16 django/db/migrations/questioner.py | 105 | 120| 135 | 22715 | 45319 | 
| 57 | 16 django/db/migrations/autodetector.py | 1295 | 1307| 131 | 22846 | 45319 | 
| 58 | 16 django/db/migrations/autodetector.py | 1082 | 1102| 136 | 22982 | 45319 | 
| 59 | 17 django/contrib/admin/migrations/0002_logentry_remove_auto_add.py | 1 | 23| 0 | 22982 | 45420 | 
| 60 | 18 django/conf/global_settings.py | 502 | 632| 799 | 23781 | 51227 | 
| 61 | 18 django/db/migrations/loader.py | 68 | 132| 551 | 24332 | 51227 | 
| 62 | 18 django/db/migrations/autodetector.py | 1268 | 1293| 245 | 24577 | 51227 | 
| 63 | 19 django/db/migrations/__init__.py | 1 | 3| 0 | 24577 | 51251 | 
| 64 | 20 django/db/migrations/graph.py | 61 | 97| 337 | 24914 | 53854 | 
| 65 | 21 django/core/management/commands/dumpdata.py | 193 | 246| 474 | 25388 | 55772 | 
| 66 | 21 django/core/management/commands/makemessages.py | 198 | 215| 184 | 25572 | 55772 | 
| 67 | 21 django/db/migrations/autodetector.py | 1064 | 1080| 188 | 25760 | 55772 | 
| 68 | 21 django/core/management/commands/dumpdata.py | 81 | 153| 624 | 26384 | 55772 | 
| 69 | 22 django/db/migrations/migration.py | 1 | 89| 726 | 27110 | 57594 | 
| 70 | 22 django/core/management/commands/makemessages.py | 171 | 195| 225 | 27335 | 57594 | 
| 71 | 23 scripts/manage_translations.py | 1 | 29| 195 | 27530 | 59240 | 
| 72 | 23 django/db/migrations/migration.py | 91 | 126| 338 | 27868 | 59240 | 
| 73 | 24 django/contrib/sessions/migrations/0001_initial.py | 1 | 31| 162 | 28030 | 59402 | 
| 74 | 25 django/db/migrations/operations/__init__.py | 1 | 18| 195 | 28225 | 59597 | 
| 75 | 25 django/db/migrations/autodetector.py | 1135 | 1169| 300 | 28525 | 59597 | 
| 76 | 25 django/db/migrations/loader.py | 340 | 357| 152 | 28677 | 59597 | 
| 77 | 25 django/db/migrations/autodetector.py | 1215 | 1230| 179 | 28856 | 59597 | 
| 78 | 25 django/core/management/commands/makemessages.py | 84 | 97| 118 | 28974 | 59597 | 
| 79 | 25 django/core/management/commands/makemessages.py | 99 | 116| 139 | 29113 | 59597 | 
| 80 | 26 django/contrib/auth/migrations/0001_initial.py | 1 | 104| 843 | 29956 | 60440 | 
| 81 | 26 django/core/management/commands/makemessages.py | 427 | 457| 263 | 30219 | 60440 | 
| 82 | 26 django/db/migrations/loader.py | 159 | 185| 291 | 30510 | 60440 | 
| 83 | 27 django/contrib/contenttypes/migrations/0001_initial.py | 1 | 35| 207 | 30717 | 60647 | 
| 84 | 27 django/core/management/commands/makemessages.py | 217 | 282| 633 | 31350 | 60647 | 
| 85 | 27 django/db/migrations/autodetector.py | 537 | 688| 1175 | 32525 | 60647 | 
| 86 | 28 django/db/migrations/operations/special.py | 181 | 204| 246 | 32771 | 62205 | 
| 87 | 28 django/db/migrations/autodetector.py | 435 | 464| 265 | 33036 | 62205 | 
| 88 | 29 django/core/management/commands/shell.py | 96 | 116| 166 | 33202 | 63067 | 
| 89 | 29 django/db/migrations/loader.py | 187 | 208| 213 | 33415 | 63067 | 
| 90 | 29 scripts/manage_translations.py | 176 | 186| 116 | 33531 | 63067 | 
| 91 | 29 django/db/migrations/graph.py | 282 | 298| 159 | 33690 | 63067 | 
| 92 | 29 django/db/migrations/autodetector.py | 939 | 1022| 919 | 34609 | 63067 | 
| 93 | 30 django/contrib/redirects/migrations/0001_initial.py | 1 | 40| 268 | 34877 | 63335 | 
| 94 | 30 django/db/migrations/loader.py | 291 | 315| 205 | 35082 | 63335 | 
| 95 | 31 django/core/management/commands/diffsettings.py | 41 | 55| 134 | 35216 | 64025 | 
| 96 | 32 django/db/migrations/state.py | 170 | 205| 407 | 35623 | 71909 | 
| 97 | 33 django/core/management/__init__.py | 190 | 232| 343 | 35966 | 75472 | 
| 98 | 33 django/db/migrations/autodetector.py | 520 | 536| 186 | 36152 | 75472 | 
| 99 | 34 django/contrib/redirects/migrations/0002_alter_redirect_new_path_help_text.py | 1 | 25| 117 | 36269 | 75589 | 
| 100 | 34 django/db/migrations/migration.py | 128 | 177| 481 | 36750 | 75589 | 
| 101 | 34 django/db/migrations/executor.py | 276 | 291| 165 | 36915 | 75589 | 
| 102 | 34 django/core/management/commands/makemessages.py | 118 | 153| 270 | 37185 | 75589 | 
| 103 | 35 django/contrib/auth/migrations/0006_require_contenttypes_0002.py | 1 | 15| 0 | 37185 | 75665 | 
| 104 | 35 django/core/management/__init__.py | 340 | 426| 755 | 37940 | 75665 | 
| 105 | 35 django/db/migrations/migration.py | 179 | 219| 283 | 38223 | 75665 | 
| 106 | 36 django/contrib/flatpages/migrations/0001_initial.py | 1 | 40| 307 | 38530 | 75972 | 
| 107 | 37 django/db/utils.py | 256 | 298| 322 | 38852 | 77990 | 
| 108 | 37 django/db/migrations/loader.py | 317 | 338| 207 | 39059 | 77990 | 
| 109 | 37 django/db/migrations/autodetector.py | 1042 | 1062| 134 | 39193 | 77990 | 
| 110 | 38 django/utils/deconstruct.py | 1 | 58| 399 | 39592 | 78389 | 
| 111 | 39 django/db/migrations/operations/base.py | 1 | 109| 804 | 40396 | 79419 | 
| 112 | 39 django/db/migrations/executor.py | 140 | 163| 235 | 40631 | 79419 | 
| 113 | 39 django/db/migrations/operations/special.py | 63 | 114| 390 | 41021 | 79419 | 
| 114 | 39 django/db/migrations/autodetector.py | 874 | 916| 394 | 41415 | 79419 | 
| 115 | 39 django/db/migrations/autodetector.py | 1024 | 1040| 188 | 41603 | 79419 | 
| 116 | 40 django/core/management/commands/loaddata.py | 114 | 166| 436 | 42039 | 82526 | 
| 117 | 40 django/db/migrations/autodetector.py | 723 | 806| 680 | 42719 | 82526 | 
| 118 | 41 django/core/management/templates.py | 133 | 199| 583 | 43302 | 85442 | 
| 119 | 42 django/core/management/commands/check.py | 40 | 71| 221 | 43523 | 85914 | 
| 120 | 43 django/contrib/contenttypes/migrations/0002_remove_content_type_name.py | 1 | 40| 217 | 43740 | 86131 | 
| 121 | 44 django/contrib/admin/models.py | 23 | 36| 111 | 43851 | 87254 | 
| 122 | 45 django/contrib/sites/migrations/0001_initial.py | 1 | 32| 191 | 44042 | 87445 | 
| 123 | 45 django/db/migrations/executor.py | 165 | 224| 567 | 44609 | 87445 | 
| 124 | 46 django/core/management/commands/createcachetable.py | 31 | 43| 121 | 44730 | 88301 | 
| 125 | 46 django/db/migrations/graph.py | 193 | 243| 450 | 45180 | 88301 | 
| 126 | 46 django/db/migrations/operations/special.py | 116 | 130| 139 | 45319 | 88301 | 
| 127 | 46 django/db/migrations/recorder.py | 23 | 44| 145 | 45464 | 88301 | 
| 128 | 47 django/core/management/commands/flush.py | 27 | 83| 486 | 45950 | 88988 | 
| 129 | 47 django/db/migrations/autodetector.py | 1 | 35| 294 | 46244 | 88988 | 
| 130 | 47 django/db/migrations/state.py | 1 | 29| 230 | 46474 | 88988 | 
| 131 | 47 django/core/management/commands/check.py | 1 | 38| 256 | 46730 | 88988 | 
| 132 | 48 django/db/backends/mysql/creation.py | 58 | 69| 178 | 46908 | 89627 | 
| 133 | 49 django/contrib/staticfiles/storage.py | 275 | 345| 575 | 47483 | 93399 | 
| 134 | 49 django/core/management/commands/makemessages.py | 1 | 35| 265 | 47748 | 93399 | 
| 135 | 49 django/db/migrations/autodetector.py | 227 | 246| 234 | 47982 | 93399 | 
| 136 | 49 django/db/migrations/loader.py | 55 | 66| 116 | 48098 | 93399 | 
| 137 | 49 django/db/migrations/autodetector.py | 918 | 937| 184 | 48282 | 93399 | 
| 138 | 49 django/core/management/commands/makemessages.py | 459 | 504| 472 | 48754 | 93399 | 
| 139 | 49 django/core/management/base.py | 346 | 380| 297 | 49051 | 93399 | 
| 140 | 49 django/core/management/commands/loaddata.py | 69 | 85| 187 | 49238 | 93399 | 
| 141 | 50 django/db/migrations/exceptions.py | 1 | 55| 249 | 49487 | 93649 | 
| 142 | 51 django/contrib/gis/management/commands/ogrinspect.py | 98 | 135| 407 | 49894 | 94858 | 
| 143 | 52 django/db/backends/base/creation.py | 301 | 322| 258 | 50152 | 97646 | 
| 144 | 52 django/core/management/commands/makemessages.py | 403 | 425| 200 | 50352 | 97646 | 
| 145 | 52 django/core/management/commands/dumpdata.py | 155 | 191| 316 | 50668 | 97646 | 
| 146 | 53 docs/_ext/djangodocs.py | 26 | 71| 398 | 51066 | 100855 | 
| 147 | 53 django/core/management/commands/loaddata.py | 242 | 271| 303 | 51369 | 100855 | 
| 148 | 53 django/core/management/commands/dumpdata.py | 1 | 79| 565 | 51934 | 100855 | 
| 149 | 53 django/core/management/__init__.py | 173 | 187| 130 | 52064 | 100855 | 
| 150 | 53 django/core/management/commands/makemessages.py | 606 | 646| 413 | 52477 | 100855 | 
| 151 | 53 django/db/migrations/graph.py | 99 | 120| 192 | 52669 | 100855 | 
| 152 | 53 django/db/migrations/autodetector.py | 808 | 872| 676 | 53345 | 100855 | 
| 153 | 54 django/db/migrations/serializer.py | 143 | 162| 223 | 53568 | 103534 | 
| 154 | 54 django/core/management/sql.py | 1 | 35| 228 | 53796 | 103534 | 
| 155 | 55 django/core/management/utils.py | 1 | 27| 167 | 53963 | 104642 | 
| 156 | 56 django/contrib/staticfiles/management/commands/collectstatic.py | 38 | 69| 297 | 54260 | 107460 | 
| 157 | 57 django/utils/decorators.py | 114 | 152| 316 | 54576 | 108862 | 
| 158 | 57 django/db/migrations/autodetector.py | 690 | 721| 278 | 54854 | 108862 | 
| 159 | 57 django/db/migrations/autodetector.py | 466 | 518| 465 | 55319 | 108862 | 
| 160 | 58 django/core/management/commands/test.py | 26 | 48| 206 | 55525 | 109349 | 
| 161 | 58 django/db/migrations/state.py | 371 | 395| 213 | 55738 | 109349 | 
| 162 | 58 django/db/migrations/state.py | 520 | 532| 125 | 55863 | 109349 | 
| 163 | 58 django/conf/global_settings.py | 633 | 657| 184 | 56047 | 109349 | 
| 164 | 58 django/db/migrations/state.py | 84 | 115| 256 | 56303 | 109349 | 
| 165 | 58 django/core/management/commands/loaddata.py | 193 | 240| 404 | 56707 | 109349 | 
| 166 | 58 django/core/management/commands/makemessages.py | 38 | 59| 143 | 56850 | 109349 | 
| 167 | 58 django/core/management/commands/loaddata.py | 38 | 67| 261 | 57111 | 109349 | 
| 168 | 58 django/core/management/commands/makemessages.py | 62 | 82| 146 | 57257 | 109349 | 
| 169 | 58 django/db/migrations/operations/base.py | 111 | 141| 229 | 57486 | 109349 | 
| 170 | 58 django/db/migrations/executor.py | 91 | 138| 446 | 57932 | 109349 | 
| 171 | 59 django/db/backends/oracle/creation.py | 30 | 100| 722 | 58654 | 113242 | 
| 172 | 59 django/core/management/commands/shell.py | 42 | 94| 439 | 59093 | 113242 | 
| 173 | 59 django/db/migrations/operations/special.py | 44 | 60| 180 | 59273 | 113242 | 
| 174 | 59 django/core/management/commands/createcachetable.py | 45 | 108| 532 | 59805 | 113242 | 
| 175 | 59 django/db/migrations/loader.py | 134 | 157| 215 | 60020 | 113242 | 
| 176 | 59 django/core/management/commands/loaddata.py | 273 | 307| 337 | 60357 | 113242 | 
| 177 | 60 django/db/migrations/optimizer.py | 1 | 38| 344 | 60701 | 113832 | 
| 178 | 60 django/core/management/commands/diffsettings.py | 69 | 80| 143 | 60844 | 113832 | 
| 179 | 60 django/core/management/commands/flush.py | 1 | 25| 206 | 61050 | 113832 | 
| 180 | 60 django/db/migrations/state.py | 359 | 369| 132 | 61182 | 113832 | 
| 181 | 61 django/core/management/commands/sqlsequencereset.py | 1 | 26| 194 | 61376 | 114026 | 
| 182 | 62 django/db/models/options.py | 1 | 35| 300 | 61676 | 121388 | 
| 183 | 62 django/core/management/commands/diffsettings.py | 57 | 67| 128 | 61804 | 121388 | 
| 184 | 62 django/contrib/staticfiles/management/commands/collectstatic.py | 148 | 205| 497 | 62301 | 121388 | 
| 185 | 62 django/contrib/staticfiles/management/commands/collectstatic.py | 86 | 146| 480 | 62781 | 121388 | 
| 186 | 62 django/db/migrations/graph.py | 300 | 320| 180 | 62961 | 121388 | 
| 187 | 63 django/contrib/auth/management/commands/createsuperuser.py | 81 | 208| 1229 | 64190 | 123522 | 
| 188 | 63 django/core/management/commands/makemessages.py | 506 | 604| 778 | 64968 | 123522 | 
| 189 | 63 django/db/migrations/questioner.py | 207 | 225| 233 | 65201 | 123522 | 
| 190 | 63 django/core/management/base.py | 523 | 556| 291 | 65492 | 123522 | 
| 191 | 63 django/core/management/base.py | 157 | 237| 762 | 66254 | 123522 | 
| 192 | 64 django/core/management/commands/testserver.py | 29 | 55| 234 | 66488 | 123955 | 
| 193 | 65 django/contrib/contenttypes/apps.py | 1 | 24| 161 | 66649 | 124116 | 
| 194 | 66 django/db/migrations/operations/models.py | 630 | 686| 325 | 66974 | 130679 | 
| 195 | 66 django/db/migrations/autodetector.py | 1104 | 1133| 283 | 67257 | 130679 | 
| 196 | 66 django/db/migrations/graph.py | 259 | 280| 179 | 67436 | 130679 | 
| 197 | 67 django/db/backends/mysql/schema.py | 1 | 39| 428 | 67864 | 132253 | 
| 198 | 67 django/core/management/utils.py | 128 | 154| 198 | 68062 | 132253 | 
| 199 | 68 django/contrib/auth/management/__init__.py | 35 | 86| 471 | 68533 | 133363 | 
| 200 | 68 django/core/management/templates.py | 67 | 131| 563 | 69096 | 133363 | 
| 201 | 68 django/core/management/__init__.py | 78 | 172| 797 | 69893 | 133363 | 
| 202 | 69 django/db/backends/sqlite3/creation.py | 51 | 82| 317 | 70210 | 134214 | 
| 203 | 69 django/db/migrations/state.py | 535 | 581| 444 | 70654 | 134214 | 
| 204 | 70 django/core/files/storage.py | 245 | 308| 561 | 71215 | 137244 | 
| 205 | 70 django/db/backends/mysql/creation.py | 32 | 56| 253 | 71468 | 137244 | 
| 206 | 70 django/db/backends/base/creation.py | 1 | 100| 755 | 72223 | 137244 | 
| 207 | 70 django/core/management/__init__.py | 266 | 338| 721 | 72944 | 137244 | 
| 208 | 71 django/contrib/auth/migrations/0010_alter_group_name_max_length.py | 1 | 17| 0 | 72944 | 137321 | 


### Hint

```
The current makemigrations command would seem to allow for everything you have asked. # Define People and Poll models >> python -m manage makemigrations Migrations for 'newspaper': newspaper\migrations\0001_initial.py - Create model People Migrations for 'polls': polls\migrations\0001_initial.py - Create model Poll # Add new_field to People Model >> python -m manage makemigrations Migrations for 'newspaper': newspaper\migrations\0002_people_new_field.py - Add field new_field to people Addressing each point: Output written to stdout rather than stderr. You can redirect stdout to stderr for the execution of the command if you want the output in stderr. (Using sys in python, or pipes for a shell script) A structured / programmatic way to figure out what files it has created. Run a regex on the output of the migration command. Example pattern: r'Migrations for '(?P<app_name>[^']*)':\n (?P<migration_file>[^\n]*)' Is there a reason this would not meet your needs?
I don't consider parsing log output with regexes to be structured output or a programmatic API. That seems brittle and unnecessarily complicated, and also would be hard for Django to make backwards compatibility guarantees about. What I had in mind was something like log output going to stderr, and the paths of the created files going to stdout -- one per line. If something fancier was needed, json could be outputted. With something like that, there wouldn't be any need for regex parsing and the API would be well-defined.
It seems complicated. For example, what if makemigrations requires interactive input from the questioner?
My original use case was a non-interactive one (inside a container). But again, you raise a good point. Obviously, piping to stdout won't work if interactivity is required (because you'd want user prompts to go to stdout). This is true of any script, not just Django management commands. Other than that, I don't think the changes I've described would hurt things in that case, aside from possibly the "reliability" issue you mentioned here. That though could be addressed by my follow-up comment to yours. If we wanted a fancier solution, the "structured" stdout could be outputted only in non-interactive mode.
I'm skeptical, but I guess if you have a patch to propose, we can evaluate it.
I closed #29470 as a duplicate.
If you are developing with Docker, why are you not just mounting your development machines source directory into the container, execute makemigrations and then you have the migrations directly on your machine. This way you can save yourself from parsing anything.
```

## Patch

```diff
diff --git a/django/core/management/commands/makemigrations.py b/django/core/management/commands/makemigrations.py
--- a/django/core/management/commands/makemigrations.py
+++ b/django/core/management/commands/makemigrations.py
@@ -57,9 +57,20 @@ def add_arguments(self, parser):
             '--check', action='store_true', dest='check_changes',
             help='Exit with a non-zero status if model changes are missing migrations.',
         )
+        parser.add_argument(
+            '--scriptable', action='store_true', dest='scriptable',
+            help=(
+                'Divert log output and input prompts to stderr, writing only '
+                'paths of generated migration files to stdout.'
+            ),
+        )
+
+    @property
+    def log_output(self):
+        return self.stderr if self.scriptable else self.stdout
 
     def log(self, msg):
-        self.stdout.write(msg)
+        self.log_output.write(msg)
 
     @no_translations
     def handle(self, *app_labels, **options):
@@ -73,6 +84,10 @@ def handle(self, *app_labels, **options):
             raise CommandError('The migration name must be a valid Python identifier.')
         self.include_header = options['include_header']
         check_changes = options['check_changes']
+        self.scriptable = options['scriptable']
+        # If logs and prompts are diverted to stderr, remove the ERROR style.
+        if self.scriptable:
+            self.stderr.style_func = None
 
         # Make sure the app they asked for exists
         app_labels = set(app_labels)
@@ -147,7 +162,7 @@ def handle(self, *app_labels, **options):
             questioner = InteractiveMigrationQuestioner(
                 specified_apps=app_labels,
                 dry_run=self.dry_run,
-                prompt_output=self.stdout,
+                prompt_output=self.log_output,
             )
         else:
             questioner = NonInteractiveMigrationQuestioner(
@@ -226,6 +241,8 @@ def write_migration_files(self, changes):
                     self.log('  %s\n' % self.style.MIGRATE_LABEL(migration_string))
                     for operation in migration.operations:
                         self.log('    - %s' % operation.describe())
+                    if self.scriptable:
+                        self.stdout.write(migration_string)
                 if not self.dry_run:
                     # Write the migrations file to the disk.
                     migrations_directory = os.path.dirname(writer.path)
@@ -254,7 +271,7 @@ def handle_merge(self, loader, conflicts):
         if it's safe; otherwise, advises on how to fix it.
         """
         if self.interactive:
-            questioner = InteractiveMigrationQuestioner(prompt_output=self.stdout)
+            questioner = InteractiveMigrationQuestioner(prompt_output=self.log_output)
         else:
             questioner = MigrationQuestioner(defaults={'ask_merge': True})
 
@@ -327,6 +344,8 @@ def all_items_equal(seq):
                         fh.write(writer.as_string())
                     if self.verbosity > 0:
                         self.log('\nCreated new merge migration %s' % writer.path)
+                        if self.scriptable:
+                            self.stdout.write(writer.path)
                 elif self.verbosity == 3:
                     # Alternatively, makemigrations --merge --dry-run --verbosity 3
                     # will log the merge migrations rather than saving the file

```

## Test Patch

```diff
diff --git a/tests/migrations/test_commands.py b/tests/migrations/test_commands.py
--- a/tests/migrations/test_commands.py
+++ b/tests/migrations/test_commands.py
@@ -1667,6 +1667,47 @@ class Meta:
         self.assertIn("model_name='sillymodel',", out.getvalue())
         self.assertIn("name='silly_char',", out.getvalue())
 
+    def test_makemigrations_scriptable(self):
+        """
+        With scriptable=True, log output is diverted to stderr, and only the
+        paths of generated migration files are written to stdout.
+        """
+        out = io.StringIO()
+        err = io.StringIO()
+        with self.temporary_migration_module(
+            module='migrations.migrations.test_migrations',
+        ) as migration_dir:
+            call_command(
+                'makemigrations',
+                'migrations',
+                scriptable=True,
+                stdout=out,
+                stderr=err,
+            )
+        initial_file = os.path.join(migration_dir, '0001_initial.py')
+        self.assertEqual(out.getvalue(), f'{initial_file}\n')
+        self.assertIn('    - Create model ModelWithCustomBase\n', err.getvalue())
+
+    @mock.patch('builtins.input', return_value='Y')
+    def test_makemigrations_scriptable_merge(self, mock_input):
+        out = io.StringIO()
+        err = io.StringIO()
+        with self.temporary_migration_module(
+            module='migrations.test_migrations_conflict',
+        ) as migration_dir:
+            call_command(
+                'makemigrations',
+                'migrations',
+                merge=True,
+                name='merge',
+                scriptable=True,
+                stdout=out,
+                stderr=err,
+            )
+        merge_file = os.path.join(migration_dir, '0003_merge.py')
+        self.assertEqual(out.getvalue(), f'{merge_file}\n')
+        self.assertIn(f'Created new merge migration {merge_file}', err.getvalue())
+
     def test_makemigrations_migrations_modules_path_not_exist(self):
         """
         makemigrations creates migrations when specifying a custom location

```


## Code snippets

### 1 - django/core/management/commands/makemigrations.py:

Start line: 160, End line: 204

```python
class Command(BaseCommand):

    @no_translations
    def handle(self, *app_labels, **options):
        # ... other code
        autodetector = MigrationAutodetector(
            loader.project_state(),
            ProjectState.from_apps(apps),
            questioner,
        )

        # If they want to make an empty migration, make one for each app
        if self.empty:
            if not app_labels:
                raise CommandError("You must supply at least one app label when using --empty.")
            # Make a fake changes() result we can pass to arrange_for_graph
            changes = {
                app: [Migration("custom", app)]
                for app in app_labels
            }
            changes = autodetector.arrange_for_graph(
                changes=changes,
                graph=loader.graph,
                migration_name=self.migration_name,
            )
            self.write_migration_files(changes)
            return

        # Detect changes
        changes = autodetector.changes(
            graph=loader.graph,
            trim_to_apps=app_labels or None,
            convert_apps=app_labels or None,
            migration_name=self.migration_name,
        )

        if not changes:
            # No changes? Tell them.
            if self.verbosity >= 1:
                if app_labels:
                    if len(app_labels) == 1:
                        self.log("No changes detected in app '%s'" % app_labels.pop())
                    else:
                        self.log("No changes detected in apps '%s'" % ("', '".join(app_labels)))
                else:
                    self.log('No changes detected')
        else:
            self.write_migration_files(changes)
            if check_changes:
                sys.exit(1)
```
### 2 - django/core/management/commands/makemigrations.py:

Start line: 24, End line: 62

```python
class Command(BaseCommand):
    help = "Creates new migration(s) for apps."

    def add_arguments(self, parser):
        parser.add_argument(
            'args', metavar='app_label', nargs='*',
            help='Specify the app label(s) to create migrations for.',
        )
        parser.add_argument(
            '--dry-run', action='store_true',
            help="Just show what migrations would be made; don't actually write them.",
        )
        parser.add_argument(
            '--merge', action='store_true',
            help="Enable fixing of migration conflicts.",
        )
        parser.add_argument(
            '--empty', action='store_true',
            help="Create an empty migration.",
        )
        parser.add_argument(
            '--noinput', '--no-input', action='store_false', dest='interactive',
            help='Tells Django to NOT prompt the user for input of any kind.',
        )
        parser.add_argument(
            '-n', '--name',
            help="Use this name for migration file(s).",
        )
        parser.add_argument(
            '--no-header', action='store_false', dest='include_header',
            help='Do not add header comments to new migration file(s).',
        )
        parser.add_argument(
            '--check', action='store_true', dest='check_changes',
            help='Exit with a non-zero status if model changes are missing migrations.',
        )

    def log(self, msg):
        self.stdout.write(msg)
```
### 3 - django/core/management/commands/makemigrations.py:

Start line: 64, End line: 159

```python
class Command(BaseCommand):

    @no_translations
    def handle(self, *app_labels, **options):
        self.verbosity = options['verbosity']
        self.interactive = options['interactive']
        self.dry_run = options['dry_run']
        self.merge = options['merge']
        self.empty = options['empty']
        self.migration_name = options['name']
        if self.migration_name and not self.migration_name.isidentifier():
            raise CommandError('The migration name must be a valid Python identifier.')
        self.include_header = options['include_header']
        check_changes = options['check_changes']

        # Make sure the app they asked for exists
        app_labels = set(app_labels)
        has_bad_labels = False
        for app_label in app_labels:
            try:
                apps.get_app_config(app_label)
            except LookupError as err:
                self.stderr.write(str(err))
                has_bad_labels = True
        if has_bad_labels:
            sys.exit(2)

        # Load the current graph state. Pass in None for the connection so
        # the loader doesn't try to resolve replaced migrations from DB.
        loader = MigrationLoader(None, ignore_no_migrations=True)

        # Raise an error if any migrations are applied before their dependencies.
        consistency_check_labels = {config.label for config in apps.get_app_configs()}
        # Non-default databases are only checked if database routers used.
        aliases_to_check = connections if settings.DATABASE_ROUTERS else [DEFAULT_DB_ALIAS]
        for alias in sorted(aliases_to_check):
            connection = connections[alias]
            if (connection.settings_dict['ENGINE'] != 'django.db.backends.dummy' and any(
                    # At least one model must be migrated to the database.
                    router.allow_migrate(connection.alias, app_label, model_name=model._meta.object_name)
                    for app_label in consistency_check_labels
                    for model in apps.get_app_config(app_label).get_models()
            )):
                try:
                    loader.check_consistent_history(connection)
                except OperationalError as error:
                    warnings.warn(
                        "Got an error checking a consistent migration history "
                        "performed for database connection '%s': %s"
                        % (alias, error),
                        RuntimeWarning,
                    )
        # Before anything else, see if there's conflicting apps and drop out
        # hard if there are any and they don't want to merge
        conflicts = loader.detect_conflicts()

        # If app_labels is specified, filter out conflicting migrations for unspecified apps
        if app_labels:
            conflicts = {
                app_label: conflict for app_label, conflict in conflicts.items()
                if app_label in app_labels
            }

        if conflicts and not self.merge:
            name_str = "; ".join(
                "%s in %s" % (", ".join(names), app)
                for app, names in conflicts.items()
            )
            raise CommandError(
                "Conflicting migrations detected; multiple leaf nodes in the "
                "migration graph: (%s).\nTo fix them run "
                "'python manage.py makemigrations --merge'" % name_str
            )

        # If they want to merge and there's nothing to merge, then politely exit
        if self.merge and not conflicts:
            self.log('No conflicts detected to merge.')
            return

        # If they want to merge and there is something to merge, then
        # divert into the merge code
        if self.merge and conflicts:
            return self.handle_merge(loader, conflicts)

        if self.interactive:
            questioner = InteractiveMigrationQuestioner(
                specified_apps=app_labels,
                dry_run=self.dry_run,
                prompt_output=self.stdout,
            )
        else:
            questioner = NonInteractiveMigrationQuestioner(
                specified_apps=app_labels,
                dry_run=self.dry_run,
                verbosity=self.verbosity,
                log=self.log,
            )
        # Set up autodetector
        # ... other code
```
### 4 - django/core/management/commands/makemigrations.py:

Start line: 206, End line: 249

```python
class Command(BaseCommand):

    def write_migration_files(self, changes):
        """
        Take a changes dict and write them out as migration files.
        """
        directory_created = {}
        for app_label, app_migrations in changes.items():
            if self.verbosity >= 1:
                self.log(self.style.MIGRATE_HEADING("Migrations for '%s':" % app_label))
            for migration in app_migrations:
                # Describe the migration
                writer = MigrationWriter(migration, self.include_header)
                if self.verbosity >= 1:
                    # Display a relative path if it's below the current working
                    # directory, or an absolute path otherwise.
                    try:
                        migration_string = os.path.relpath(writer.path)
                    except ValueError:
                        migration_string = writer.path
                    if migration_string.startswith('..'):
                        migration_string = writer.path
                    self.log('  %s\n' % self.style.MIGRATE_LABEL(migration_string))
                    for operation in migration.operations:
                        self.log('    - %s' % operation.describe())
                if not self.dry_run:
                    # Write the migrations file to the disk.
                    migrations_directory = os.path.dirname(writer.path)
                    if not directory_created.get(app_label):
                        os.makedirs(migrations_directory, exist_ok=True)
                        init_path = os.path.join(migrations_directory, "__init__.py")
                        if not os.path.isfile(init_path):
                            open(init_path, "w").close()
                        # We just do this once per app
                        directory_created[app_label] = True
                    migration_string = writer.as_string()
                    with open(writer.path, "w", encoding='utf-8') as fh:
                        fh.write(migration_string)
                elif self.verbosity == 3:
                    # Alternatively, makemigrations --dry-run --verbosity 3
                    # will log the migrations rather than saving the file to
                    # the disk.
                    self.log(self.style.MIGRATE_HEADING(
                        "Full migrations file '%s':" % writer.filename
                    ))
                    self.log(writer.as_string())
```
### 5 - django/core/management/commands/makemigrations.py:

Start line: 251, End line: 338

```python
class Command(BaseCommand):

    def handle_merge(self, loader, conflicts):
        """
        Handles merging together conflicted migrations interactively,
        if it's safe; otherwise, advises on how to fix it.
        """
        if self.interactive:
            questioner = InteractiveMigrationQuestioner(prompt_output=self.stdout)
        else:
            questioner = MigrationQuestioner(defaults={'ask_merge': True})

        for app_label, migration_names in conflicts.items():
            # Grab out the migrations in question, and work out their
            # common ancestor.
            merge_migrations = []
            for migration_name in migration_names:
                migration = loader.get_migration(app_label, migration_name)
                migration.ancestry = [
                    mig for mig in loader.graph.forwards_plan((app_label, migration_name))
                    if mig[0] == migration.app_label
                ]
                merge_migrations.append(migration)

            def all_items_equal(seq):
                return all(item == seq[0] for item in seq[1:])

            merge_migrations_generations = zip(*(m.ancestry for m in merge_migrations))
            common_ancestor_count = sum(1 for common_ancestor_generation
                                        in takewhile(all_items_equal, merge_migrations_generations))
            if not common_ancestor_count:
                raise ValueError("Could not find common ancestor of %s" % migration_names)
            # Now work out the operations along each divergent branch
            for migration in merge_migrations:
                migration.branch = migration.ancestry[common_ancestor_count:]
                migrations_ops = (loader.get_migration(node_app, node_name).operations
                                  for node_app, node_name in migration.branch)
                migration.merged_operations = sum(migrations_ops, [])
            # In future, this could use some of the Optimizer code
            # (can_optimize_through) to automatically see if they're
            # mergeable. For now, we always just prompt the user.
            if self.verbosity > 0:
                self.log(self.style.MIGRATE_HEADING('Merging %s' % app_label))
                for migration in merge_migrations:
                    self.log(self.style.MIGRATE_LABEL('  Branch %s' % migration.name))
                    for operation in migration.merged_operations:
                        self.log('    - %s' % operation.describe())
            if questioner.ask_merge(app_label):
                # If they still want to merge it, then write out an empty
                # file depending on the migrations needing merging.
                numbers = [
                    MigrationAutodetector.parse_number(migration.name)
                    for migration in merge_migrations
                ]
                try:
                    biggest_number = max(x for x in numbers if x is not None)
                except ValueError:
                    biggest_number = 1
                subclass = type("Migration", (Migration,), {
                    "dependencies": [(app_label, migration.name) for migration in merge_migrations],
                })
                parts = ['%04i' % (biggest_number + 1)]
                if self.migration_name:
                    parts.append(self.migration_name)
                else:
                    parts.append('merge')
                    leaf_names = '_'.join(sorted(migration.name for migration in merge_migrations))
                    if len(leaf_names) > 47:
                        parts.append(get_migration_name_timestamp())
                    else:
                        parts.append(leaf_names)
                migration_name = '_'.join(parts)
                new_migration = subclass(migration_name, app_label)
                writer = MigrationWriter(new_migration, self.include_header)

                if not self.dry_run:
                    # Write the merge migrations file to the disk
                    with open(writer.path, "w", encoding='utf-8') as fh:
                        fh.write(writer.as_string())
                    if self.verbosity > 0:
                        self.log('\nCreated new merge migration %s' % writer.path)
                elif self.verbosity == 3:
                    # Alternatively, makemigrations --merge --dry-run --verbosity 3
                    # will log the merge migrations rather than saving the file
                    # to the disk.
                    self.log(self.style.MIGRATE_HEADING(
                        "Full merge migrations file '%s':" % writer.filename
                    ))
                    self.log(writer.as_string())
```
### 6 - django/core/management/commands/makemigrations.py:

Start line: 1, End line: 21

```python
import os
import sys
import warnings
from itertools import takewhile

from django.apps import apps
from django.conf import settings
from django.core.management.base import (
    BaseCommand, CommandError, no_translations,
)
from django.db import DEFAULT_DB_ALIAS, OperationalError, connections, router
from django.db.migrations import Migration
from django.db.migrations.autodetector import MigrationAutodetector
from django.db.migrations.loader import MigrationLoader
from django.db.migrations.questioner import (
    InteractiveMigrationQuestioner, MigrationQuestioner,
    NonInteractiveMigrationQuestioner,
)
from django.db.migrations.state import ProjectState
from django.db.migrations.utils import get_migration_name_timestamp
from django.db.migrations.writer import MigrationWriter
```
### 7 - django/core/management/commands/migrate.py:

Start line: 162, End line: 227

```python
class Command(BaseCommand):

    @no_translations
    def handle(self, *args, **options):
        # ... other code

        if options['plan']:
            self.stdout.write('Planned operations:', self.style.MIGRATE_LABEL)
            if not plan:
                self.stdout.write('  No planned migration operations.')
            for migration, backwards in plan:
                self.stdout.write(str(migration), self.style.MIGRATE_HEADING)
                for operation in migration.operations:
                    message, is_error = self.describe_operation(operation, backwards)
                    style = self.style.WARNING if is_error else None
                    self.stdout.write('    ' + message, style)
            if exit_dry:
                sys.exit(1)
            return
        if exit_dry:
            sys.exit(1)

        # At this point, ignore run_syncdb if there aren't any apps to sync.
        run_syncdb = options['run_syncdb'] and executor.loader.unmigrated_apps
        # Print some useful info
        if self.verbosity >= 1:
            self.stdout.write(self.style.MIGRATE_HEADING("Operations to perform:"))
            if run_syncdb:
                if options['app_label']:
                    self.stdout.write(
                        self.style.MIGRATE_LABEL("  Synchronize unmigrated app: %s" % app_label)
                    )
                else:
                    self.stdout.write(
                        self.style.MIGRATE_LABEL("  Synchronize unmigrated apps: ") +
                        (", ".join(sorted(executor.loader.unmigrated_apps)))
                    )
            if target_app_labels_only:
                self.stdout.write(
                    self.style.MIGRATE_LABEL("  Apply all migrations: ") +
                    (", ".join(sorted({a for a, n in targets})) or "(none)")
                )
            else:
                if targets[0][1] is None:
                    self.stdout.write(
                        self.style.MIGRATE_LABEL('  Unapply all migrations: ') +
                        str(targets[0][0])
                    )
                else:
                    self.stdout.write(self.style.MIGRATE_LABEL(
                        "  Target specific migration: ") + "%s, from %s"
                        % (targets[0][1], targets[0][0])
                    )

        pre_migrate_state = executor._create_project_state(with_applied_migrations=True)
        pre_migrate_apps = pre_migrate_state.apps
        emit_pre_migrate_signal(
            self.verbosity, self.interactive, connection.alias, stdout=self.stdout, apps=pre_migrate_apps, plan=plan,
        )

        # Run the syncdb phase.
        if run_syncdb:
            if self.verbosity >= 1:
                self.stdout.write(self.style.MIGRATE_HEADING("Synchronizing apps without migrations:"))
            if options['app_label']:
                self.sync_apps(connection, [app_label])
            else:
                self.sync_apps(connection, executor.loader.unmigrated_apps)

        # Migrate!
        if self.verbosity >= 1:
            self.stdout.write(self.style.MIGRATE_HEADING("Running migrations:"))
        # ... other code
```
### 8 - django/core/management/commands/migrate.py:

Start line: 71, End line: 160

```python
class Command(BaseCommand):

    @no_translations
    def handle(self, *args, **options):
        database = options['database']
        if not options['skip_checks']:
            self.check(databases=[database])

        self.verbosity = options['verbosity']
        self.interactive = options['interactive']

        # Import the 'management' module within each installed app, to register
        # dispatcher events.
        for app_config in apps.get_app_configs():
            if module_has_submodule(app_config.module, "management"):
                import_module('.management', app_config.name)

        # Get the database we're operating from
        connection = connections[database]

        # Hook for backends needing any database preparation
        connection.prepare_database()
        # Work out which apps have migrations and which do not
        executor = MigrationExecutor(connection, self.migration_progress_callback)

        # Raise an error if any migrations are applied before their dependencies.
        executor.loader.check_consistent_history(connection)

        # Before anything else, see if there's conflicting apps and drop out
        # hard if there are any
        conflicts = executor.loader.detect_conflicts()
        if conflicts:
            name_str = "; ".join(
                "%s in %s" % (", ".join(names), app)
                for app, names in conflicts.items()
            )
            raise CommandError(
                "Conflicting migrations detected; multiple leaf nodes in the "
                "migration graph: (%s).\nTo fix them run "
                "'python manage.py makemigrations --merge'" % name_str
            )

        # If they supplied command line arguments, work out what they mean.
        run_syncdb = options['run_syncdb']
        target_app_labels_only = True
        if options['app_label']:
            # Validate app_label.
            app_label = options['app_label']
            try:
                apps.get_app_config(app_label)
            except LookupError as err:
                raise CommandError(str(err))
            if run_syncdb:
                if app_label in executor.loader.migrated_apps:
                    raise CommandError("Can't use run_syncdb with app '%s' as it has migrations." % app_label)
            elif app_label not in executor.loader.migrated_apps:
                raise CommandError("App '%s' does not have migrations." % app_label)

        if options['app_label'] and options['migration_name']:
            migration_name = options['migration_name']
            if migration_name == "zero":
                targets = [(app_label, None)]
            else:
                try:
                    migration = executor.loader.get_migration_by_prefix(app_label, migration_name)
                except AmbiguityError:
                    raise CommandError(
                        "More than one migration matches '%s' in app '%s'. "
                        "Please be more specific." %
                        (migration_name, app_label)
                    )
                except KeyError:
                    raise CommandError("Cannot find a migration matching '%s' from app '%s'." % (
                        migration_name, app_label))
                target = (app_label, migration.name)
                # Partially applied squashed migrations are not included in the
                # graph, use the last replacement instead.
                if (
                    target not in executor.loader.graph.nodes and
                    target in executor.loader.replacements
                ):
                    incomplete_migration = executor.loader.replacements[target]
                    target = incomplete_migration.replaces[-1]
                targets = [target]
            target_app_labels_only = False
        elif options['app_label']:
            targets = [key for key in executor.loader.graph.leaf_nodes() if key[0] == app_label]
        else:
            targets = executor.loader.graph.leaf_nodes()

        plan = executor.migration_plan(targets)
        exit_dry = plan and options['check_unapplied']
        # ... other code
```
### 9 - django/core/management/commands/migrate.py:

Start line: 228, End line: 279

```python
class Command(BaseCommand):

    @no_translations
    def handle(self, *args, **options):
        # ... other code
        if not plan:
            if self.verbosity >= 1:
                self.stdout.write("  No migrations to apply.")
                # If there's changes that aren't in migrations yet, tell them how to fix it.
                autodetector = MigrationAutodetector(
                    executor.loader.project_state(),
                    ProjectState.from_apps(apps),
                )
                changes = autodetector.changes(graph=executor.loader.graph)
                if changes:
                    self.stdout.write(self.style.NOTICE(
                        "  Your models in app(s): %s have changes that are not "
                        "yet reflected in a migration, and so won't be "
                        "applied." % ", ".join(repr(app) for app in sorted(changes))
                    ))
                    self.stdout.write(self.style.NOTICE(
                        "  Run 'manage.py makemigrations' to make new "
                        "migrations, and then re-run 'manage.py migrate' to "
                        "apply them."
                    ))
            fake = False
            fake_initial = False
        else:
            fake = options['fake']
            fake_initial = options['fake_initial']
        post_migrate_state = executor.migrate(
            targets, plan=plan, state=pre_migrate_state.clone(), fake=fake,
            fake_initial=fake_initial,
        )
        # post_migrate signals have access to all models. Ensure that all models
        # are reloaded in case any are delayed.
        post_migrate_state.clear_delayed_apps_cache()
        post_migrate_apps = post_migrate_state.apps

        # Re-render models of real apps to include relationships now that
        # we've got a final state. This wouldn't be necessary if real apps
        # models were rendered with relationships in the first place.
        with post_migrate_apps.bulk_update():
            model_keys = []
            for model_state in post_migrate_apps.real_models:
                model_key = model_state.app_label, model_state.name_lower
                model_keys.append(model_key)
                post_migrate_apps.unregister_model(*model_key)
        post_migrate_apps.render_multiple([
            ModelState.from_model(apps.get_model(*model)) for model in model_keys
        ])

        # Send the post_migrate signal, so individual apps can do whatever they need
        # to do at this point.
        emit_post_migrate_signal(
            self.verbosity, self.interactive, connection.alias, stdout=self.stdout, apps=post_migrate_apps, plan=plan,
        )
```
### 10 - django/core/management/commands/showmigrations.py:

Start line: 46, End line: 67

```python
class Command(BaseCommand):

    def handle(self, *args, **options):
        self.verbosity = options['verbosity']

        # Get the database we're operating from
        db = options['database']
        connection = connections[db]

        if options['format'] == "plan":
            return self.show_plan(connection, options['app_label'])
        else:
            return self.show_list(connection, options['app_label'])

    def _validate_app_names(self, loader, app_names):
        has_bad_names = False
        for app_name in app_names:
            try:
                apps.get_app_config(app_name)
            except LookupError as err:
                self.stderr.write(str(err))
                has_bad_names = True
        if has_bad_names:
            sys.exit(2)
```
