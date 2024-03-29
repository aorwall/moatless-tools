# django__django-15018

| **django/django** | `1feb55d60736011ee94fbff9ba0c1c25acfd0b14` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 847 |
| **Any found context length** | 847 |
| **Avg pos** | 1.0 |
| **Min pos** | 1 |
| **Max pos** | 1 |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/core/management/__init__.py b/django/core/management/__init__.py
--- a/django/core/management/__init__.py
+++ b/django/core/management/__init__.py
@@ -149,6 +149,12 @@ def get_actions(parser):
             opt.dest in options and
             (opt.required or opt in mutually_exclusive_required_options)
         ):
+            opt_dest_count = sum(v == opt.dest for v in opt_mapping.values())
+            if opt_dest_count > 1:
+                raise TypeError(
+                    f'Cannot pass the dest {opt.dest!r} that matches multiple '
+                    f'arguments via **options.'
+                )
             parse_args.append(min(opt.option_strings))
             if isinstance(opt, (_AppendConstAction, _CountAction, _StoreConstAction)):
                 continue

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/core/management/__init__.py | 152 | 152 | 1 | 1 | 847


## Problem Statement

```
call_command() fails when required mutually exclusive arguments use the same `dest`.
Description
	
I have a command which accepts two different ways to specify a time -- either as a timestamp or as a duration in the future:
pause (--for duration | --until time)
class Command(BaseCommand):
	def add_arguments(self, parser) -> None:
		group = parser.add_mutually_exclusive_group(required=True)
		group.add_argument('--for', dest='until', action='store', type=parse_duration_to_time)
		group.add_argument('--until', action='store', type=parse_time)
	def handle(self, until: datetime, **_):
		pass
This works fine on the command line, however there doesn't seem to be a way to make this work through call_command. Specifically there are two sides to the failure:
while I can provide an until value (as a string, which is processed by parse_time) there is no mechanism to pass a for value if that's how I want to spell the input
the for value is always required and attempts to parse the (string) until value passed, which then errors since the input formats are very different

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 django/core/management/__init__.py** | 78 | 181| 847 | 847 | 3508 | 
| 2 | 2 django/db/models/functions/text.py | 64 | 92| 231 | 1078 | 5844 | 
| 3 | 3 django/core/management/commands/squashmigrations.py | 1 | 43| 350 | 1428 | 7717 | 
| 4 | 4 django/contrib/staticfiles/management/commands/collectstatic.py | 38 | 69| 297 | 1725 | 10535 | 
| 5 | 5 django/core/management/commands/makemessages.py | 216 | 281| 633 | 2358 | 16162 | 
| 6 | 6 django/core/management/commands/migrate.py | 21 | 69| 407 | 2765 | 19499 | 
| 7 | 7 django/core/management/base.py | 37 | 66| 230 | 2995 | 24174 | 
| 8 | 7 django/core/management/base.py | 559 | 591| 240 | 3235 | 24174 | 
| 9 | 7 django/core/management/commands/migrate.py | 71 | 160| 774 | 4009 | 24174 | 
| 10 | 8 django/core/management/commands/test.py | 26 | 48| 206 | 4215 | 24661 | 
| 11 | 9 django/core/management/commands/makemigrations.py | 64 | 155| 805 | 5020 | 27509 | 
| 12 | 9 django/core/management/commands/squashmigrations.py | 45 | 134| 791 | 5811 | 27509 | 
| 13 | 9 django/core/management/base.py | 382 | 417| 296 | 6107 | 27509 | 
| 14 | 9 django/core/management/base.py | 267 | 319| 460 | 6567 | 27509 | 
| 15 | 9 django/core/management/base.py | 1 | 34| 243 | 6810 | 27509 | 
| 16 | 10 django/core/management/commands/compilemessages.py | 30 | 57| 230 | 7040 | 28828 | 
| 17 | 11 django/core/management/commands/sendtestemail.py | 1 | 24| 186 | 7226 | 29130 | 
| 18 | 11 django/core/management/commands/compilemessages.py | 59 | 116| 504 | 7730 | 29130 | 
| 19 | 12 django/core/management/commands/testserver.py | 1 | 27| 204 | 7934 | 29563 | 
| 20 | 12 django/core/management/commands/makemessages.py | 283 | 362| 814 | 8748 | 29563 | 
| 21 | 12 django/core/management/base.py | 96 | 116| 168 | 8916 | 29563 | 
| 22 | 12 django/core/management/base.py | 523 | 556| 291 | 9207 | 29563 | 
| 23 | 13 django/core/management/commands/shell.py | 1 | 40| 267 | 9474 | 30425 | 
| 24 | 14 django/contrib/auth/management/commands/createsuperuser.py | 81 | 208| 1229 | 10703 | 32559 | 
| 25 | 14 django/core/management/commands/migrate.py | 162 | 227| 632 | 11335 | 32559 | 
| 26 | 15 django/core/management/templates.py | 133 | 199| 582 | 11917 | 35361 | 
| 27 | 15 django/core/management/templates.py | 67 | 131| 563 | 12480 | 35361 | 
| 28 | 16 django/core/management/commands/check.py | 40 | 71| 221 | 12701 | 35833 | 
| 29 | 16 django/core/management/commands/makemessages.py | 363 | 400| 272 | 12973 | 35833 | 
| 30 | 16 django/core/management/commands/shell.py | 96 | 116| 166 | 13139 | 35833 | 
| 31 | 17 django/core/management/commands/diffsettings.py | 69 | 80| 143 | 13282 | 36523 | 
| 32 | 17 django/core/management/base.py | 157 | 237| 762 | 14044 | 36523 | 
| 33 | 17 django/core/management/commands/diffsettings.py | 41 | 55| 134 | 14178 | 36523 | 
| 34 | 17 django/core/management/templates.py | 227 | 258| 236 | 14414 | 36523 | 
| 35 | 18 django/core/management/commands/sqlmigrate.py | 31 | 69| 379 | 14793 | 37156 | 
| 36 | 18 django/core/management/commands/makemigrations.py | 24 | 62| 297 | 15090 | 37156 | 
| 37 | 18 django/core/management/templates.py | 41 | 65| 246 | 15336 | 37156 | 
| 38 | 19 django/contrib/auth/management/commands/changepassword.py | 34 | 76| 333 | 15669 | 37690 | 
| 39 | 19 django/contrib/staticfiles/management/commands/collectstatic.py | 148 | 205| 497 | 16166 | 37690 | 
| 40 | 20 django/contrib/contenttypes/management/commands/remove_stale_contenttypes.py | 1 | 27| 188 | 16354 | 38410 | 
| 41 | 20 django/core/management/commands/squashmigrations.py | 136 | 204| 654 | 17008 | 38410 | 
| 42 | 20 django/core/management/commands/squashmigrations.py | 206 | 219| 112 | 17120 | 38410 | 
| 43 | 20 django/core/management/commands/shell.py | 42 | 94| 439 | 17559 | 38410 | 
| 44 | 21 django/contrib/gis/management/commands/ogrinspect.py | 33 | 96| 591 | 18150 | 39619 | 
| 45 | 21 django/contrib/auth/management/commands/createsuperuser.py | 1 | 79| 577 | 18727 | 39619 | 
| 46 | 21 django/core/management/commands/migrate.py | 281 | 313| 349 | 19076 | 39619 | 
| 47 | 21 django/core/management/commands/test.py | 50 | 63| 121 | 19197 | 39619 | 
| 48 | 21 django/core/management/commands/makemigrations.py | 156 | 200| 347 | 19544 | 39619 | 
| 49 | 21 django/core/management/commands/makemessages.py | 1 | 34| 260 | 19804 | 39619 | 
| 50 | 22 django/core/management/utils.py | 52 | 74| 204 | 20008 | 40727 | 
| 51 | **22 django/core/management/__init__.py** | 334 | 420| 755 | 20763 | 40727 | 
| 52 | 23 django/db/migrations/exceptions.py | 1 | 55| 249 | 21012 | 40977 | 
| 53 | 24 django/core/management/commands/dumpdata.py | 81 | 153| 624 | 21636 | 42895 | 
| 54 | 25 django/utils/timesince.py | 27 | 102| 715 | 22351 | 43872 | 
| 55 | 26 django/core/management/commands/loaddata.py | 38 | 67| 261 | 22612 | 46979 | 
| 56 | 26 django/core/management/commands/migrate.py | 228 | 279| 537 | 23149 | 46979 | 
| 57 | 26 django/core/management/commands/testserver.py | 29 | 55| 234 | 23383 | 46979 | 
| 58 | 26 django/core/management/base.py | 419 | 486| 622 | 24005 | 46979 | 
| 59 | 27 django/db/models/expressions.py | 549 | 578| 263 | 24268 | 58156 | 
| 60 | 27 django/core/management/base.py | 321 | 344| 167 | 24435 | 58156 | 
| 61 | 28 django/core/management/commands/dbshell.py | 23 | 44| 175 | 24610 | 58465 | 
| 62 | 28 django/core/management/base.py | 69 | 93| 161 | 24771 | 58465 | 
| 63 | 28 django/contrib/auth/management/commands/changepassword.py | 1 | 32| 205 | 24976 | 58465 | 
| 64 | 29 django/db/migrations/operations/models.py | 531 | 550| 148 | 25124 | 64915 | 
| 65 | 29 django/core/management/commands/check.py | 1 | 38| 256 | 25380 | 64915 | 
| 66 | 29 django/contrib/auth/management/commands/createsuperuser.py | 210 | 234| 204 | 25584 | 64915 | 
| 67 | 29 django/core/management/commands/makemessages.py | 197 | 214| 176 | 25760 | 64915 | 
| 68 | 29 django/core/management/commands/makemigrations.py | 247 | 334| 865 | 26625 | 64915 | 
| 69 | 30 django/contrib/postgres/constraints.py | 1 | 64| 526 | 27151 | 66437 | 
| 70 | 30 django/core/management/utils.py | 112 | 125| 119 | 27270 | 66437 | 
| 71 | 30 django/core/management/commands/loaddata.py | 69 | 85| 187 | 27457 | 66437 | 
| 72 | 31 django/db/models/sql/query.py | 708 | 743| 389 | 27846 | 88773 | 
| 73 | 31 django/contrib/staticfiles/management/commands/collectstatic.py | 326 | 346| 205 | 28051 | 88773 | 
| 74 | 31 django/core/management/commands/compilemessages.py | 118 | 169| 432 | 28483 | 88773 | 
| 75 | 32 django/contrib/sitemaps/management/commands/ping_google.py | 1 | 17| 119 | 28602 | 88892 | 
| 76 | 33 django/forms/fields.py | 485 | 510| 174 | 28776 | 98313 | 
| 77 | **33 django/core/management/__init__.py** | 228 | 258| 309 | 29085 | 98313 | 
| 78 | 34 django/core/management/commands/runserver.py | 112 | 165| 510 | 29595 | 99805 | 
| 79 | 35 django/db/models/fields/__init__.py | 1117 | 1149| 237 | 29832 | 117973 | 
| 80 | 36 django/db/models/fields/related.py | 1266 | 1383| 963 | 30795 | 131969 | 
| 81 | 36 django/core/management/commands/compilemessages.py | 1 | 27| 161 | 30956 | 131969 | 
| 82 | 37 django/db/models/base.py | 1440 | 1495| 491 | 31447 | 149299 | 
| 83 | 37 django/core/management/commands/sendtestemail.py | 26 | 41| 121 | 31568 | 149299 | 
| 84 | 38 django/core/management/commands/flush.py | 27 | 83| 486 | 32054 | 149986 | 
| 85 | 38 django/core/management/base.py | 346 | 380| 297 | 32351 | 149986 | 
| 86 | 38 django/core/management/commands/dbshell.py | 1 | 21| 139 | 32490 | 149986 | 
| 87 | 39 django/db/models/functions/mixins.py | 23 | 53| 257 | 32747 | 150404 | 
| 88 | 40 django/core/management/commands/createcachetable.py | 31 | 43| 121 | 32868 | 151260 | 
| 89 | 40 django/db/migrations/operations/models.py | 469 | 498| 168 | 33036 | 151260 | 
| 90 | 41 django/contrib/auth/checks.py | 105 | 211| 776 | 33812 | 152758 | 
| 91 | 42 django/db/models/functions/comparison.py | 76 | 85| 134 | 33946 | 154492 | 
| 92 | 42 django/core/management/commands/runserver.py | 24 | 58| 276 | 34222 | 154492 | 
| 93 | 42 django/core/management/commands/diffsettings.py | 57 | 67| 128 | 34350 | 154492 | 
| 94 | 43 django/dispatch/dispatcher.py | 175 | 222| 358 | 34708 | 156577 | 
| 95 | 43 django/db/models/expressions.py | 520 | 547| 214 | 34922 | 156577 | 
| 96 | 43 django/contrib/auth/management/commands/createsuperuser.py | 236 | 251| 139 | 35061 | 156577 | 
| 97 | 44 django/db/models/constraints.py | 38 | 90| 416 | 35477 | 158638 | 
| 98 | 44 django/db/models/fields/related.py | 1385 | 1457| 616 | 36093 | 158638 | 
| 99 | 44 django/db/migrations/operations/models.py | 511 | 528| 168 | 36261 | 158638 | 
| 100 | 45 django/core/management/commands/showmigrations.py | 46 | 67| 158 | 36419 | 159908 | 
| 101 | 45 django/core/management/commands/dumpdata.py | 1 | 79| 565 | 36984 | 159908 | 
| 102 | 45 django/core/management/commands/sqlmigrate.py | 1 | 29| 259 | 37243 | 159908 | 
| 103 | 45 django/contrib/staticfiles/management/commands/collectstatic.py | 294 | 324| 290 | 37533 | 159908 | 
| 104 | 45 django/core/management/commands/loaddata.py | 114 | 166| 436 | 37969 | 159908 | 
| 105 | 45 django/core/management/commands/loaddata.py | 242 | 271| 303 | 38272 | 159908 | 
| 106 | **45 django/core/management/__init__.py** | 260 | 332| 721 | 38993 | 159908 | 
| 107 | 45 django/core/management/commands/dumpdata.py | 193 | 246| 474 | 39467 | 159908 | 
| 108 | 45 django/contrib/contenttypes/management/commands/remove_stale_contenttypes.py | 29 | 95| 517 | 39984 | 159908 | 
| 109 | 45 django/db/migrations/operations/models.py | 500 | 509| 129 | 40113 | 159908 | 
| 110 | 45 django/contrib/staticfiles/management/commands/collectstatic.py | 244 | 292| 477 | 40590 | 159908 | 
| 111 | 45 django/core/management/base.py | 238 | 265| 239 | 40829 | 159908 | 
| 112 | 45 django/db/models/base.py | 1607 | 1632| 183 | 41012 | 159908 | 
| 113 | 46 django/utils/deprecation.py | 128 | 146| 122 | 41134 | 160933 | 
| 114 | 47 django/contrib/gis/geos/prototypes/coordseq.py | 33 | 52| 170 | 41304 | 161713 | 
| 115 | 47 django/contrib/postgres/constraints.py | 142 | 168| 231 | 41535 | 161713 | 
| 116 | 48 django/db/models/functions/datetime.py | 65 | 88| 270 | 41805 | 164392 | 
| 117 | 48 django/core/management/commands/loaddata.py | 87 | 112| 264 | 42069 | 164392 | 
| 118 | 49 django/db/models/sql/compiler.py | 1080 | 1120| 337 | 42406 | 179184 | 
| 119 | 49 django/db/models/sql/compiler.py | 1021 | 1043| 207 | 42613 | 179184 | 
| 120 | 50 django/db/backends/sqlite3/operations.py | 43 | 69| 232 | 42845 | 182456 | 
| 121 | 50 django/db/models/fields/related.py | 198 | 266| 687 | 43532 | 182456 | 
| 122 | 50 django/db/models/sql/compiler.py | 466 | 519| 569 | 44101 | 182456 | 
| 123 | 51 django/contrib/postgres/aggregates/general.py | 94 | 112| 178 | 44279 | 183273 | 
| 124 | 52 django/contrib/auth/mixins.py | 107 | 129| 146 | 44425 | 184137 | 
| 125 | 52 django/core/management/commands/makemessages.py | 426 | 456| 263 | 44688 | 184137 | 
| 126 | 53 django/contrib/sessions/management/commands/clearsessions.py | 1 | 22| 123 | 44811 | 184260 | 
| 127 | 53 django/core/management/commands/test.py | 1 | 24| 170 | 44981 | 184260 | 
| 128 | 53 django/core/management/commands/showmigrations.py | 1 | 44| 311 | 45292 | 184260 | 
| 129 | 54 django/core/mail/message.py | 55 | 71| 171 | 45463 | 187911 | 
| 130 | 55 django/utils/dateparse.py | 134 | 159| 255 | 45718 | 189507 | 
| 131 | 56 django/db/backends/mysql/client.py | 1 | 61| 560 | 46278 | 190068 | 
| 132 | 56 django/contrib/gis/management/commands/ogrinspect.py | 98 | 135| 407 | 46685 | 190068 | 
| 133 | 56 django/db/models/sql/compiler.py | 280 | 375| 710 | 47395 | 190068 | 
| 134 | 57 django/db/migrations/questioner.py | 238 | 257| 195 | 47590 | 192592 | 
| 135 | 58 django/db/models/aggregates.py | 50 | 68| 278 | 47868 | 194007 | 
| 136 | 58 django/db/models/fields/related.py | 1459 | 1500| 418 | 48286 | 194007 | 
| 137 | 58 django/contrib/staticfiles/management/commands/collectstatic.py | 86 | 146| 480 | 48766 | 194007 | 
| 138 | 58 django/db/models/functions/comparison.py | 88 | 102| 175 | 48941 | 194007 | 
| 139 | 59 django/conf/locale/nl/formats.py | 32 | 67| 1371 | 50312 | 195890 | 
| 140 | 59 django/db/models/expressions.py | 33 | 147| 836 | 51148 | 195890 | 
| 141 | 60 django/templatetags/tz.py | 149 | 169| 176 | 51324 | 197218 | 
| 142 | 60 django/db/models/base.py | 1773 | 1873| 729 | 52053 | 197218 | 
| 143 | 61 django/contrib/postgres/operations.py | 192 | 213| 207 | 52260 | 199604 | 
| 144 | 61 django/core/management/commands/makemessages.py | 402 | 424| 200 | 52460 | 199604 | 


### Hint

```
Thanks for the report. The following calls work as expected for me : management.call_command('pause', '--until=1') management.call_command('pause', '--until', '1') management.call_command('pause', '--for=1') management.call_command('pause', '--for', '1') however I confirmed an issue when passing arguments in keyword arguments: management.call_command('pause', until='1') management.call_command('pause', **{'for': '1'}) # Using "call_command('pause', for='1')" raises SyntaxError This is caused by using dest for mapping **options to arguments, see ​call_command().
I can create a patch to fix the two above-mentioned issues but using the command with both options together: management.call_command('pause', **{'for': '1', 'util': '1'}) won't raise an error because both options use the same dest. I will investigate it more. I don't know do we have to support passing both dest and arg name as keyword arguments? in the example of this ticket, if we call management.call_command('pause', until='1'), it should be considered as until arg or for (because dest of for is until as well)
Ah, interesting, I wasn't aware that those other spellings worked! I'd be happy to switch to using those spellings instead (I've confirmed they work for my original case). Perhaps just documenting this limitation (and the fact that those spellings work as alternatives) and/or improving the related error messages could be a way to go here?
Replying to Hasan Ramezani: I don't know do we have to support passing both dest and arg name as keyword arguments? in the example of this ticket, if we call management.call_command('pause', until='1'), it should be considered as until arg or for (because dest of for is until as well) We should support option names as ​documented: ** options named options accepted on the command-line. so management.call_command('pause', until='1') should work the same as calling pause --until 1 management.call_command('pause', **{'for': '1'}) should work the same as calling pause --for 1 management.call_command('pause', **{'for': '1', 'until': '1'}) should work the same as calling pause --for 1 --until 1 and raise an exception
Replying to Mariusz Felisiak: I am not sure about your second example: management.call_command('pause', **{'for': '1'}) should work the same as calling pause --for 1 Based on the documentation, it seems we have to pass dest as keyword argument name when we define dest for arguments. Some command options have different names when using call_command() instead of django-admin or manage.py. For example, django-admin createsuperuser --no-input translates to call_command('createsuperuser', interactive=False). To find what keyword argument name to use for call_command(), check the command’s source code for the dest argument passed to parser.add_argument(). Also, when Django ​adds required arguments in call command, it search for dest in options.
You're right, sorry, I missed "... check the command’s source code for the dest argument passed to parser.add_argument().". In that case I would raise an error that passing dest with multiple arguments via **options is not supported.
```

## Patch

```diff
diff --git a/django/core/management/__init__.py b/django/core/management/__init__.py
--- a/django/core/management/__init__.py
+++ b/django/core/management/__init__.py
@@ -149,6 +149,12 @@ def get_actions(parser):
             opt.dest in options and
             (opt.required or opt in mutually_exclusive_required_options)
         ):
+            opt_dest_count = sum(v == opt.dest for v in opt_mapping.values())
+            if opt_dest_count > 1:
+                raise TypeError(
+                    f'Cannot pass the dest {opt.dest!r} that matches multiple '
+                    f'arguments via **options.'
+                )
             parse_args.append(min(opt.option_strings))
             if isinstance(opt, (_AppendConstAction, _CountAction, _StoreConstAction)):
                 continue

```

## Test Patch

```diff
diff --git a/tests/user_commands/management/commands/mutually_exclusive_required_with_same_dest.py b/tests/user_commands/management/commands/mutually_exclusive_required_with_same_dest.py
new file mode 100644
--- /dev/null
+++ b/tests/user_commands/management/commands/mutually_exclusive_required_with_same_dest.py
@@ -0,0 +1,13 @@
+from django.core.management.base import BaseCommand
+
+
+class Command(BaseCommand):
+    def add_arguments(self, parser):
+        group = parser.add_mutually_exclusive_group(required=True)
+        group.add_argument('--for', dest='until', action='store')
+        group.add_argument('--until', action='store')
+
+    def handle(self, *args, **options):
+        for option, value in options.items():
+            if value is not None:
+                self.stdout.write('%s=%s' % (option, value))
diff --git a/tests/user_commands/tests.py b/tests/user_commands/tests.py
--- a/tests/user_commands/tests.py
+++ b/tests/user_commands/tests.py
@@ -274,6 +274,41 @@ def test_mutually_exclusive_group_required_const_options(self):
                 )
                 self.assertIn(expected_output, out.getvalue())
 
+    def test_mutually_exclusive_group_required_with_same_dest_options(self):
+        tests = [
+            {'until': '2'},
+            {'for': '1', 'until': '2'},
+        ]
+        msg = (
+            "Cannot pass the dest 'until' that matches multiple arguments via "
+            "**options."
+        )
+        for options in tests:
+            with self.subTest(options=options):
+                with self.assertRaisesMessage(TypeError, msg):
+                    management.call_command(
+                        'mutually_exclusive_required_with_same_dest',
+                        **options,
+                    )
+
+    def test_mutually_exclusive_group_required_with_same_dest_args(self):
+        tests = [
+            ('--until=1',),
+            ('--until', 1),
+            ('--for=1',),
+            ('--for', 1),
+        ]
+        for args in tests:
+            out = StringIO()
+            with self.subTest(options=args):
+                management.call_command(
+                    'mutually_exclusive_required_with_same_dest',
+                    *args,
+                    stdout=out,
+                )
+                output = out.getvalue()
+                self.assertIn('until=1', output)
+
     def test_required_list_option(self):
         tests = [
             (('--foo-list', [1, 2]), {}),

```


## Code snippets

### 1 - django/core/management/__init__.py:

Start line: 78, End line: 181

```python
def call_command(command_name, *args, **options):
    """
    Call the given command, with the given options and args/kwargs.

    This is the primary API you should use for calling specific commands.

    `command_name` may be a string or a command object. Using a string is
    preferred unless the command object is required for further processing or
    testing.

    Some examples:
        call_command('migrate')
        call_command('shell', plain=True)
        call_command('sqlmigrate', 'myapp')

        from django.core.management.commands import flush
        cmd = flush.Command()
        call_command(cmd, verbosity=0, interactive=False)
        # Do something with cmd ...
    """
    if isinstance(command_name, BaseCommand):
        # Command object passed in.
        command = command_name
        command_name = command.__class__.__module__.split('.')[-1]
    else:
        # Load the command object by name.
        try:
            app_name = get_commands()[command_name]
        except KeyError:
            raise CommandError("Unknown command: %r" % command_name)

        if isinstance(app_name, BaseCommand):
            # If the command is already loaded, use it directly.
            command = app_name
        else:
            command = load_command_class(app_name, command_name)

    # Simulate argument parsing to get the option defaults (see #10080 for details).
    parser = command.create_parser('', command_name)
    # Use the `dest` option name from the parser option
    opt_mapping = {
        min(s_opt.option_strings).lstrip('-').replace('-', '_'): s_opt.dest
        for s_opt in parser._actions if s_opt.option_strings
    }
    arg_options = {opt_mapping.get(key, key): value for key, value in options.items()}
    parse_args = []
    for arg in args:
        if isinstance(arg, (list, tuple)):
            parse_args += map(str, arg)
        else:
            parse_args.append(str(arg))

    def get_actions(parser):
        # Parser actions and actions from sub-parser choices.
        for opt in parser._actions:
            if isinstance(opt, _SubParsersAction):
                for sub_opt in opt.choices.values():
                    yield from get_actions(sub_opt)
            else:
                yield opt

    parser_actions = list(get_actions(parser))
    mutually_exclusive_required_options = {
        opt
        for group in parser._mutually_exclusive_groups
        for opt in group._group_actions if group.required
    }
    # Any required arguments which are passed in via **options must be passed
    # to parse_args().
    for opt in parser_actions:
        if (
            opt.dest in options and
            (opt.required or opt in mutually_exclusive_required_options)
        ):
            parse_args.append(min(opt.option_strings))
            if isinstance(opt, (_AppendConstAction, _CountAction, _StoreConstAction)):
                continue
            value = arg_options[opt.dest]
            if isinstance(value, (list, tuple)):
                parse_args += map(str, value)
            else:
                parse_args.append(str(value))
    defaults = parser.parse_args(args=parse_args)
    defaults = dict(defaults._get_kwargs(), **arg_options)
    # Raise an error if any unknown options were passed.
    stealth_options = set(command.base_stealth_options + command.stealth_options)
    dest_parameters = {action.dest for action in parser_actions}
    valid_options = (dest_parameters | stealth_options).union(opt_mapping)
    unknown_options = set(options) - valid_options
    if unknown_options:
        raise TypeError(
            "Unknown option(s) for %s command: %s. "
            "Valid options are: %s." % (
                command_name,
                ', '.join(sorted(unknown_options)),
                ', '.join(sorted(valid_options)),
            )
        )
    # Move positional args out of options to mimic legacy optparse
    args = defaults.pop('args', ())
    if 'skip_checks' not in options:
        defaults['skip_checks'] = True

    return command.execute(*args, **defaults)
```
### 2 - django/db/models/functions/text.py:

Start line: 64, End line: 92

```python
class ConcatPair(Func):
    """
    Concatenate two arguments together. This is used by `Concat` because not
    all backend databases support more than two arguments.
    """
    function = 'CONCAT'

    def as_sqlite(self, compiler, connection, **extra_context):
        coalesced = self.coalesce()
        return super(ConcatPair, coalesced).as_sql(
            compiler, connection, template='%(expressions)s', arg_joiner=' || ',
            **extra_context
        )

    def as_mysql(self, compiler, connection, **extra_context):
        # Use CONCAT_WS with an empty separator so that NULLs are ignored.
        return super().as_sql(
            compiler, connection, function='CONCAT_WS',
            template="%(function)s('', %(expressions)s)",
            **extra_context
        )

    def coalesce(self):
        # null on either side results in null for expression, wrap with coalesce
        c = self.copy()
        c.set_source_expressions([
            Coalesce(expression, Value('')) for expression in c.get_source_expressions()
        ])
        return c
```
### 3 - django/core/management/commands/squashmigrations.py:

Start line: 1, End line: 43

```python
from django.apps import apps
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db import DEFAULT_DB_ALIAS, connections, migrations
from django.db.migrations.loader import AmbiguityError, MigrationLoader
from django.db.migrations.migration import SwappableTuple
from django.db.migrations.optimizer import MigrationOptimizer
from django.db.migrations.writer import MigrationWriter
from django.utils.version import get_docs_version


class Command(BaseCommand):
    help = "Squashes an existing set of migrations (from first until specified) into a single new one."

    def add_arguments(self, parser):
        parser.add_argument(
            'app_label',
            help='App label of the application to squash migrations for.',
        )
        parser.add_argument(
            'start_migration_name', nargs='?',
            help='Migrations will be squashed starting from and including this migration.',
        )
        parser.add_argument(
            'migration_name',
            help='Migrations will be squashed until and including this migration.',
        )
        parser.add_argument(
            '--no-optimize', action='store_true',
            help='Do not try to optimize the squashed operations.',
        )
        parser.add_argument(
            '--noinput', '--no-input', action='store_false', dest='interactive',
            help='Tells Django to NOT prompt the user for input of any kind.',
        )
        parser.add_argument(
            '--squashed-name',
            help='Sets the name of the new squashed migration.',
        )
        parser.add_argument(
            '--no-header', action='store_false', dest='include_header',
            help='Do not add a header comment to the new squashed migration.',
        )
```
### 4 - django/contrib/staticfiles/management/commands/collectstatic.py:

Start line: 38, End line: 69

```python
class Command(BaseCommand):

    def add_arguments(self, parser):
        parser.add_argument(
            '--noinput', '--no-input', action='store_false', dest='interactive',
            help="Do NOT prompt the user for input of any kind.",
        )
        parser.add_argument(
            '--no-post-process', action='store_false', dest='post_process',
            help="Do NOT post process collected files.",
        )
        parser.add_argument(
            '-i', '--ignore', action='append', default=[],
            dest='ignore_patterns', metavar='PATTERN',
            help="Ignore files or directories matching this glob-style "
                 "pattern. Use multiple times to ignore more.",
        )
        parser.add_argument(
            '-n', '--dry-run', action='store_true',
            help="Do everything except modify the filesystem.",
        )
        parser.add_argument(
            '-c', '--clear', action='store_true',
            help="Clear the existing files using the storage "
                 "before trying to copy or link the original file.",
        )
        parser.add_argument(
            '-l', '--link', action='store_true',
            help="Create a symbolic link to each file instead of copying.",
        )
        parser.add_argument(
            '--no-default-ignore', action='store_false', dest='use_default_ignore_patterns',
            help="Don't ignore the common private glob-style patterns (defaults to 'CVS', '.*' and '*~').",
        )
```
### 5 - django/core/management/commands/makemessages.py:

Start line: 216, End line: 281

```python
class Command(BaseCommand):

    def add_arguments(self, parser):
        parser.add_argument(
            '--locale', '-l', default=[], action='append',
            help='Creates or updates the message files for the given locale(s) (e.g. pt_BR). '
                 'Can be used multiple times.',
        )
        parser.add_argument(
            '--exclude', '-x', default=[], action='append',
            help='Locales to exclude. Default is none. Can be used multiple times.',
        )
        parser.add_argument(
            '--domain', '-d', default='django',
            help='The domain of the message files (default: "django").',
        )
        parser.add_argument(
            '--all', '-a', action='store_true',
            help='Updates the message files for all existing locales.',
        )
        parser.add_argument(
            '--extension', '-e', dest='extensions', action='append',
            help='The file extension(s) to examine (default: "html,txt,py", or "js" '
                 'if the domain is "djangojs"). Separate multiple extensions with '
                 'commas, or use -e multiple times.',
        )
        parser.add_argument(
            '--symlinks', '-s', action='store_true',
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
            '--no-wrap', action='store_true',
            help="Don't break long message lines into several lines.",
        )
        parser.add_argument(
            '--no-location', action='store_true',
            help="Don't write '#: filename:line' lines.",
        )
        parser.add_argument(
            '--add-location',
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
            '--no-obsolete', action='store_true',
            help="Remove obsolete message strings.",
        )
        parser.add_argument(
            '--keep-pot', action='store_true',
            help="Keep .pot file after making messages. Useful when debugging.",
        )
```
### 6 - django/core/management/commands/migrate.py:

Start line: 21, End line: 69

```python
class Command(BaseCommand):
    help = "Updates database schema. Manages both apps with migrations and those without."
    requires_system_checks = []

    def add_arguments(self, parser):
        parser.add_argument(
            '--skip-checks', action='store_true',
            help='Skip system checks.',
        )
        parser.add_argument(
            'app_label', nargs='?',
            help='App label of an application to synchronize the state.',
        )
        parser.add_argument(
            'migration_name', nargs='?',
            help='Database state will be brought to the state after that '
                 'migration. Use the name "zero" to unapply all migrations.',
        )
        parser.add_argument(
            '--noinput', '--no-input', action='store_false', dest='interactive',
            help='Tells Django to NOT prompt the user for input of any kind.',
        )
        parser.add_argument(
            '--database',
            default=DEFAULT_DB_ALIAS,
            help='Nominates a database to synchronize. Defaults to the "default" database.',
        )
        parser.add_argument(
            '--fake', action='store_true',
            help='Mark migrations as run without actually running them.',
        )
        parser.add_argument(
            '--fake-initial', action='store_true',
            help='Detect if tables already exist and fake-apply initial migrations if so. Make sure '
                 'that the current database schema matches your initial migration before using this '
                 'flag. Django will only check for an existing table name.',
        )
        parser.add_argument(
            '--plan', action='store_true',
            help='Shows a list of the migration actions that will be performed.',
        )
        parser.add_argument(
            '--run-syncdb', action='store_true',
            help='Creates tables for apps without migrations.',
        )
        parser.add_argument(
            '--check', action='store_true', dest='check_unapplied',
            help='Exits with a non-zero status if unapplied migrations exist.',
        )
```
### 7 - django/core/management/base.py:

Start line: 37, End line: 66

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
    def __init__(self, *, missing_args_message=None, called_from_command_line=None, **kwargs):
        self.missing_args_message = missing_args_message
        self.called_from_command_line = called_from_command_line
        super().__init__(**kwargs)

    def parse_args(self, args=None, namespace=None):
        # Catch missing argument for a better error message
        if (self.missing_args_message and
                not (args or any(not arg.startswith('-') for arg in args))):
            self.error(self.missing_args_message)
        return super().parse_args(args, namespace)

    def error(self, message):
        if self.called_from_command_line:
            super().error(message)
        else:
            raise CommandError("Error: %s" % message)
```
### 8 - django/core/management/base.py:

Start line: 559, End line: 591

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
### 9 - django/core/management/commands/migrate.py:

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
### 10 - django/core/management/commands/test.py:

Start line: 26, End line: 48

```python
class Command(BaseCommand):

    def add_arguments(self, parser):
        parser.add_argument(
            'args', metavar='test_label', nargs='*',
            help='Module paths to test; can be modulename, modulename.TestCase or modulename.TestCase.test_method'
        )
        parser.add_argument(
            '--noinput', '--no-input', action='store_false', dest='interactive',
            help='Tells Django to NOT prompt the user for input of any kind.',
        )
        parser.add_argument(
            '--failfast', action='store_true',
            help='Tells Django to stop running the test suite after first failed test.',
        )
        parser.add_argument(
            '--testrunner',
            help='Tells Django to use specified test runner class instead of '
                 'the one specified by the TEST_RUNNER setting.',
        )

        test_runner_class = get_runner(settings, self.test_runner)

        if hasattr(test_runner_class, 'add_arguments'):
            test_runner_class.add_arguments(parser)
```
### 51 - django/core/management/__init__.py:

Start line: 334, End line: 420

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
        parser = CommandParser(
            prog=self.prog_name,
            usage='%(prog)s subcommand [options] [args]',
            add_help=False,
            allow_abbrev=False,
        )
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
                    apps.all_models = defaultdict(dict)
                    apps.app_configs = {}
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
### 77 - django/core/management/__init__.py:

Start line: 228, End line: 258

```python
class ManagementUtility:

    def fetch_command(self, subcommand):
        """
        Try to fetch the given subcommand, printing a message with the
        appropriate command called from the command line (usually
        "django-admin" or "manage.py") if it can't be found.
        """
        # Get commands outside of try block to prevent swallowing exceptions
        commands = get_commands()
        try:
            app_name = commands[subcommand]
        except KeyError:
            if os.environ.get('DJANGO_SETTINGS_MODULE'):
                # If `subcommand` is missing due to misconfigured settings, the
                # following line will retrigger an ImproperlyConfigured exception
                # (get_commands() swallows the original one) so the user is
                # informed about it.
                settings.INSTALLED_APPS
            elif not settings.configured:
                sys.stderr.write("No Django settings specified.\n")
            possible_matches = get_close_matches(subcommand, commands)
            sys.stderr.write('Unknown command: %r' % subcommand)
            if possible_matches:
                sys.stderr.write('. Did you mean %s?' % possible_matches[0])
            sys.stderr.write("\nType '%s help' for usage.\n" % self.prog_name)
            sys.exit(1)
        if isinstance(app_name, BaseCommand):
            # If the command is already loaded, use it directly.
            klass = app_name
        else:
            klass = load_command_class(app_name, subcommand)
        return klass
```
### 106 - django/core/management/__init__.py:

Start line: 260, End line: 332

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

        subcommands = [*get_commands(), 'help']
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
