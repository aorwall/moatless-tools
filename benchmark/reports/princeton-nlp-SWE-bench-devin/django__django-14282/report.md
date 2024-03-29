# django__django-14282

| **django/django** | `3fec16e8acf0724b061a9e3cce25da898052bc9b` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | - |
| **Avg pos** | - |
| **Min pos** | - |
| **Max pos** | - |
| **Top file pos** | - |
| **Missing snippets** | 1 |
| **Missing patch files** | 1 |


## Expected patch

```diff
diff --git a/django/contrib/auth/tokens.py b/django/contrib/auth/tokens.py
--- a/django/contrib/auth/tokens.py
+++ b/django/contrib/auth/tokens.py
@@ -12,12 +12,19 @@ class PasswordResetTokenGenerator:
     """
     key_salt = "django.contrib.auth.tokens.PasswordResetTokenGenerator"
     algorithm = None
-    secret = None
+    _secret = None
 
     def __init__(self):
-        self.secret = self.secret or settings.SECRET_KEY
         self.algorithm = self.algorithm or 'sha256'
 
+    def _get_secret(self):
+        return self._secret or settings.SECRET_KEY
+
+    def _set_secret(self, secret):
+        self._secret = secret
+
+    secret = property(_get_secret, _set_secret)
+
     def make_token(self, user):
         """
         Return a token that can be used once to do a password reset

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/contrib/auth/tokens.py | 15 | 18 | - | - | -


## Problem Statement

```
Cannot run makemigrations management command without a SECRET_KEY
Description
	
I believe #29324 intended to fix this issue.
Steps to reproduce:
$ cd $(mktemp -d)
$ python -m venv venv
$ source venv/bin/activate
$ pip install 'Django>=3.2'
$ python -m django startproject foo
$ sed -ri '/SECRET_KEY/d' foo/foo/settings.py # Remove SECRET_KEY from settings
$ PYTHONPATH=foo DJANGO_SETTINGS_MODULE="foo.settings" python -m django makemigrations --check
The output is attached.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 django/core/management/commands/startproject.py | 1 | 22| 159 | 159 | 159 | 
| 2 | 2 django/core/management/commands/makemigrations.py | 24 | 59| 284 | 443 | 2998 | 
| 3 | 2 django/core/management/commands/makemigrations.py | 154 | 192| 313 | 756 | 2998 | 
| 4 | 2 django/core/management/commands/makemigrations.py | 61 | 152| 822 | 1578 | 2998 | 
| 5 | 3 django/core/checks/security/base.py | 199 | 211| 110 | 1688 | 5046 | 
| 6 | 3 django/core/management/commands/makemigrations.py | 1 | 21| 155 | 1843 | 5046 | 
| 7 | 3 django/core/management/commands/makemigrations.py | 194 | 237| 434 | 2277 | 5046 | 
| 8 | 4 django/core/management/commands/migrate.py | 21 | 69| 407 | 2684 | 8310 | 
| 9 | 4 django/core/management/commands/migrate.py | 71 | 167| 834 | 3518 | 8310 | 
| 10 | 5 django/core/management/commands/sqlmigrate.py | 31 | 69| 379 | 3897 | 8943 | 
| 11 | 6 django/core/management/commands/showmigrations.py | 42 | 63| 158 | 4055 | 10138 | 
| 12 | 7 django/contrib/admin/migrations/0003_logentry_add_action_flag_choices.py | 1 | 21| 111 | 4166 | 10249 | 
| 13 | 8 django/db/migrations/loader.py | 156 | 182| 291 | 4457 | 13354 | 
| 14 | 9 django/core/management/utils.py | 77 | 109| 222 | 4679 | 14468 | 
| 15 | 9 django/core/management/commands/migrate.py | 169 | 251| 812 | 5491 | 14468 | 
| 16 | 10 django/core/management/commands/makemessages.py | 402 | 424| 200 | 5691 | 20055 | 
| 17 | 10 django/core/checks/security/base.py | 1 | 72| 667 | 6358 | 20055 | 
| 18 | 10 django/core/management/commands/migrate.py | 272 | 304| 349 | 6707 | 20055 | 
| 19 | 10 django/core/management/commands/makemessages.py | 283 | 362| 814 | 7521 | 20055 | 
| 20 | 11 django/contrib/admin/migrations/0001_initial.py | 1 | 48| 320 | 7841 | 20375 | 
| 21 | 11 django/core/management/commands/makemessages.py | 363 | 400| 272 | 8113 | 20375 | 
| 22 | 11 django/core/management/commands/migrate.py | 1 | 18| 140 | 8253 | 20375 | 
| 23 | 11 django/core/management/commands/sqlmigrate.py | 1 | 29| 259 | 8512 | 20375 | 
| 24 | 11 django/core/management/commands/makemessages.py | 197 | 214| 176 | 8688 | 20375 | 
| 25 | 12 django/middleware/csrf.py | 48 | 57| 111 | 8799 | 23725 | 
| 26 | 12 django/core/management/commands/showmigrations.py | 1 | 40| 292 | 9091 | 23725 | 
| 27 | 13 django/contrib/sessions/migrations/0001_initial.py | 1 | 31| 162 | 9253 | 23887 | 
| 28 | 14 django/conf/global_settings.py | 496 | 647| 939 | 10192 | 29583 | 
| 29 | 15 django/db/migrations/__init__.py | 1 | 3| 0 | 10192 | 29607 | 
| 30 | 15 django/core/management/commands/makemessages.py | 216 | 281| 633 | 10825 | 29607 | 
| 31 | 15 django/core/management/commands/migrate.py | 253 | 270| 212 | 11037 | 29607 | 
| 32 | 15 django/core/management/commands/makemigrations.py | 239 | 326| 873 | 11910 | 29607 | 
| 33 | 16 django/contrib/admin/migrations/0002_logentry_remove_auto_add.py | 1 | 23| 0 | 11910 | 29708 | 
| 34 | 17 django/db/migrations/recorder.py | 46 | 97| 390 | 12300 | 30385 | 
| 35 | 18 django/core/management/sql.py | 38 | 54| 128 | 12428 | 30741 | 
| 36 | 19 django/core/management/commands/dbshell.py | 23 | 44| 175 | 12603 | 31050 | 
| 37 | 20 django/core/management/commands/squashmigrations.py | 1 | 43| 350 | 12953 | 32923 | 
| 38 | 21 django/contrib/auth/migrations/0001_initial.py | 1 | 105| 849 | 13802 | 33772 | 
| 39 | 21 django/core/management/commands/squashmigrations.py | 206 | 219| 112 | 13914 | 33772 | 
| 40 | 22 django/contrib/auth/management/commands/changepassword.py | 34 | 76| 333 | 14247 | 34306 | 
| 41 | 22 django/conf/global_settings.py | 151 | 266| 859 | 15106 | 34306 | 
| 42 | 23 django/core/checks/security/csrf.py | 1 | 42| 304 | 15410 | 34768 | 
| 43 | 24 django/db/migrations/autodetector.py | 1230 | 1242| 131 | 15541 | 46348 | 
| 44 | 25 django/core/management/commands/check.py | 40 | 71| 221 | 15762 | 46820 | 
| 45 | 26 django/contrib/auth/management/commands/createsuperuser.py | 81 | 202| 1158 | 16920 | 48883 | 
| 46 | 27 django/core/cache/backends/base.py | 280 | 293| 112 | 17032 | 51104 | 
| 47 | 28 django/db/migrations/operations/special.py | 181 | 204| 246 | 17278 | 52662 | 
| 48 | 29 django/core/management/base.py | 21 | 42| 165 | 17443 | 57299 | 
| 49 | 30 django/contrib/redirects/migrations/0001_initial.py | 1 | 41| 274 | 17717 | 57573 | 
| 50 | 31 django/core/management/commands/flush.py | 27 | 83| 486 | 18203 | 58260 | 
| 51 | 32 django/db/utils.py | 1 | 49| 177 | 18380 | 60267 | 
| 52 | 32 django/core/management/commands/makemessages.py | 1 | 34| 260 | 18640 | 60267 | 
| 53 | 32 django/core/management/commands/squashmigrations.py | 136 | 204| 654 | 19294 | 60267 | 
| 54 | 33 django/core/management/commands/runserver.py | 111 | 164| 510 | 19804 | 61747 | 
| 55 | 33 django/core/management/commands/makemessages.py | 426 | 456| 263 | 20067 | 61747 | 
| 56 | 34 django/contrib/auth/migrations/0006_require_contenttypes_0002.py | 1 | 15| 0 | 20067 | 61823 | 
| 57 | 34 django/core/management/base.py | 479 | 511| 281 | 20348 | 61823 | 
| 58 | 34 django/contrib/auth/management/commands/changepassword.py | 1 | 32| 205 | 20553 | 61823 | 
| 59 | 35 django/contrib/sites/migrations/0001_initial.py | 1 | 32| 191 | 20744 | 62014 | 
| 60 | 35 django/db/migrations/autodetector.py | 1037 | 1053| 188 | 20932 | 62014 | 
| 61 | 36 django/db/migrations/operations/__init__.py | 1 | 18| 195 | 21127 | 62209 | 
| 62 | 36 django/db/migrations/autodetector.py | 1203 | 1228| 245 | 21372 | 62209 | 
| 63 | 36 django/core/checks/security/base.py | 74 | 168| 746 | 22118 | 62209 | 
| 64 | 36 django/contrib/auth/management/commands/createsuperuser.py | 1 | 79| 577 | 22695 | 62209 | 
| 65 | 37 django/core/management/commands/diffsettings.py | 41 | 55| 134 | 22829 | 62899 | 
| 66 | 38 django/contrib/flatpages/migrations/0001_initial.py | 1 | 40| 307 | 23136 | 63206 | 
| 67 | 38 django/db/migrations/autodetector.py | 1055 | 1075| 136 | 23272 | 63206 | 
| 68 | 38 django/core/management/commands/squashmigrations.py | 45 | 134| 791 | 24063 | 63206 | 
| 69 | 39 django/contrib/admin/checks.py | 232 | 253| 208 | 24271 | 72388 | 
| 70 | 40 django/contrib/auth/forms.py | 54 | 72| 124 | 24395 | 75500 | 
| 71 | 40 django/db/migrations/autodetector.py | 1105 | 1142| 317 | 24712 | 75500 | 
| 72 | 40 django/conf/global_settings.py | 267 | 349| 800 | 25512 | 75500 | 
| 73 | 41 django/db/migrations/questioner.py | 226 | 239| 123 | 25635 | 77557 | 
| 74 | 42 django/utils/crypto.py | 47 | 61| 128 | 25763 | 78167 | 
| 75 | 43 django/contrib/auth/migrations/0010_alter_group_name_max_length.py | 1 | 17| 0 | 25763 | 78244 | 
| 76 | 44 django/contrib/auth/migrations/0007_alter_validators_add_error_messages.py | 1 | 25| 137 | 25900 | 78381 | 
| 77 | 45 django/core/management/commands/shell.py | 42 | 82| 401 | 26301 | 79205 | 
| 78 | 46 django/db/migrations/serializer.py | 143 | 162| 223 | 26524 | 81876 | 
| 79 | 46 django/core/management/commands/migrate.py | 306 | 353| 396 | 26920 | 81876 | 
| 80 | 47 django/core/management/commands/inspectdb.py | 1 | 36| 266 | 27186 | 84509 | 
| 81 | 47 django/db/migrations/autodetector.py | 352 | 366| 138 | 27324 | 84509 | 
| 82 | 48 django/core/management/commands/dumpdata.py | 81 | 153| 624 | 27948 | 86427 | 
| 83 | 48 django/core/management/commands/runserver.py | 24 | 57| 264 | 28212 | 86427 | 
| 84 | 48 django/core/management/commands/flush.py | 1 | 25| 206 | 28418 | 86427 | 
| 85 | 49 django/core/management/commands/sqlsequencereset.py | 1 | 26| 194 | 28612 | 86621 | 
| 86 | 50 django/contrib/auth/migrations/0002_alter_permission_name_max_length.py | 1 | 17| 0 | 28612 | 86689 | 
| 87 | 50 django/utils/crypto.py | 1 | 44| 351 | 28963 | 86689 | 
| 88 | 51 django/core/management/commands/startapp.py | 1 | 15| 0 | 28963 | 86790 | 
| 89 | 52 django/contrib/auth/migrations/0003_alter_user_email_max_length.py | 1 | 17| 0 | 28963 | 86868 | 
| 90 | 52 django/core/management/commands/check.py | 1 | 38| 256 | 29219 | 86868 | 
| 91 | 53 django/db/migrations/state.py | 228 | 246| 173 | 29392 | 92610 | 
| 92 | 54 django/core/management/commands/sendtestemail.py | 26 | 41| 121 | 29513 | 92912 | 
| 93 | 54 django/db/migrations/state.py | 1 | 26| 197 | 29710 | 92912 | 
| 94 | 54 django/core/checks/security/base.py | 234 | 258| 210 | 29920 | 92912 | 
| 95 | 55 django/db/backends/dummy/features.py | 1 | 7| 0 | 29920 | 92944 | 
| 96 | 55 django/core/management/base.py | 239 | 273| 313 | 30233 | 92944 | 
| 97 | 56 django/contrib/auth/admin.py | 1 | 22| 188 | 30421 | 94670 | 
| 98 | 57 django/contrib/redirects/migrations/0002_alter_redirect_new_path_help_text.py | 1 | 25| 117 | 30538 | 94787 | 
| 99 | 57 django/core/management/commands/dumpdata.py | 193 | 246| 474 | 31012 | 94787 | 
| 100 | 57 django/core/management/commands/migrate.py | 355 | 378| 186 | 31198 | 94787 | 
| 101 | 57 django/core/management/commands/shell.py | 1 | 40| 267 | 31465 | 94787 | 
| 102 | 58 django/core/management/commands/loaddata.py | 38 | 67| 261 | 31726 | 97706 | 
| 103 | 59 django/views/debug.py | 150 | 162| 148 | 31874 | 102366 | 
| 104 | 59 django/core/management/commands/makemessages.py | 98 | 115| 139 | 32013 | 102366 | 
| 105 | 59 django/core/management/commands/loaddata.py | 87 | 157| 640 | 32653 | 102366 | 
| 106 | 60 django/db/__init__.py | 1 | 43| 272 | 32925 | 102638 | 
| 107 | 60 django/core/management/commands/shell.py | 84 | 104| 166 | 33091 | 102638 | 
| 108 | 61 django/core/management/commands/createcachetable.py | 31 | 43| 121 | 33212 | 103494 | 
| 109 | 62 django/core/management/commands/test.py | 25 | 59| 296 | 33508 | 103945 | 
| 110 | 63 django/db/backends/signals.py | 1 | 4| 0 | 33508 | 103956 | 
| 111 | 63 django/db/migrations/operations/special.py | 116 | 130| 139 | 33647 | 103956 | 
| 112 | 64 django/db/migrations/writer.py | 201 | 301| 619 | 34266 | 106203 | 
| 113 | 65 django/contrib/auth/hashers.py | 1 | 28| 194 | 34460 | 111560 | 
| 114 | 66 django/contrib/admin/models.py | 23 | 36| 111 | 34571 | 112683 | 
| 115 | 66 django/core/management/commands/loaddata.py | 1 | 35| 177 | 34748 | 112683 | 
| 116 | 67 django/db/models/base.py | 1 | 50| 328 | 35076 | 129987 | 
| 117 | 67 django/db/utils.py | 180 | 213| 224 | 35300 | 129987 | 
| 118 | 68 django/contrib/auth/migrations/0005_alter_user_last_login_null.py | 1 | 17| 0 | 35300 | 130062 | 
| 119 | 68 django/core/management/commands/runserver.py | 71 | 109| 397 | 35697 | 130062 | 
| 120 | 68 django/contrib/auth/management/commands/createsuperuser.py | 230 | 245| 139 | 35836 | 130062 | 
| 121 | 68 django/db/migrations/recorder.py | 1 | 21| 148 | 35984 | 130062 | 
| 122 | 68 django/core/management/commands/loaddata.py | 69 | 85| 187 | 36171 | 130062 | 
| 123 | 68 django/conf/global_settings.py | 401 | 495| 782 | 36953 | 130062 | 
| 124 | 69 django/db/migrations/executor.py | 264 | 279| 165 | 37118 | 133344 | 
| 125 | 70 django/db/backends/sqlite3/creation.py | 23 | 49| 239 | 37357 | 134195 | 
| 126 | 70 django/db/migrations/autodetector.py | 1144 | 1165| 231 | 37588 | 134195 | 
| 127 | 71 django/core/checks/messages.py | 54 | 77| 161 | 37749 | 134772 | 
| 128 | 71 django/db/migrations/loader.py | 184 | 205| 213 | 37962 | 134772 | 
| 129 | 72 django/contrib/auth/migrations/0012_alter_user_first_name_max_length.py | 1 | 17| 0 | 37962 | 134847 | 
| 130 | 72 django/core/management/base.py | 373 | 408| 296 | 38258 | 134847 | 
| 131 | 72 django/core/management/commands/showmigrations.py | 65 | 103| 420 | 38678 | 134847 | 
| 132 | 72 django/db/migrations/autodetector.py | 997 | 1013| 188 | 38866 | 134847 | 
| 133 | 73 django/contrib/auth/migrations/0004_alter_user_username_opts.py | 1 | 24| 150 | 39016 | 134997 | 
| 134 | 74 django/core/signing.py | 1 | 78| 741 | 39757 | 136879 | 
| 135 | 74 django/core/management/commands/sendtestemail.py | 1 | 24| 186 | 39943 | 136879 | 
| 136 | 74 django/db/migrations/serializer.py | 108 | 118| 114 | 40057 | 136879 | 
| 137 | 75 django/contrib/auth/migrations/0009_alter_user_last_name_max_length.py | 1 | 17| 0 | 40057 | 136957 | 
| 138 | 75 django/db/migrations/executor.py | 238 | 262| 227 | 40284 | 136957 | 
| 139 | 75 django/core/management/commands/diffsettings.py | 57 | 67| 128 | 40412 | 136957 | 
| 140 | 75 django/core/checks/messages.py | 27 | 51| 259 | 40671 | 136957 | 
| 141 | 76 django/db/backends/mysql/client.py | 1 | 58| 546 | 41217 | 137504 | 
| 142 | 77 django/core/mail/backends/dummy.py | 1 | 11| 0 | 41217 | 137547 | 
| 143 | 77 django/core/management/commands/makemessages.py | 37 | 58| 143 | 41360 | 137547 | 
| 144 | 77 django/db/migrations/autodetector.py | 258 | 329| 748 | 42108 | 137547 | 
| 145 | 78 django/contrib/auth/migrations/0008_alter_user_username_max_length.py | 1 | 25| 138 | 42246 | 137685 | 
| 146 | 79 django/db/backends/mysql/creation.py | 58 | 69| 178 | 42424 | 138324 | 
| 147 | 79 django/db/migrations/autodetector.py | 1167 | 1201| 296 | 42720 | 138324 | 
| 148 | 79 django/db/backends/sqlite3/creation.py | 51 | 82| 317 | 43037 | 138324 | 
| 149 | 80 django/core/management/commands/testserver.py | 1 | 27| 204 | 43241 | 138757 | 
| 150 | 81 django/core/checks/templates.py | 1 | 36| 259 | 43500 | 139017 | 
| 151 | 82 django/core/checks/compatibility/django_4_0.py | 1 | 19| 138 | 43638 | 139156 | 
| 152 | 82 django/core/management/commands/makemessages.py | 83 | 96| 118 | 43756 | 139156 | 
| 153 | 83 django/contrib/auth/signals.py | 1 | 6| 0 | 43756 | 139180 | 
| 154 | 83 django/db/migrations/autodetector.py | 1015 | 1035| 134 | 43890 | 139180 | 
| 155 | 84 django/utils/log.py | 1 | 75| 484 | 44374 | 140822 | 
| 156 | 85 django/contrib/auth/__init__.py | 1 | 38| 241 | 44615 | 142403 | 
| 157 | 85 django/db/migrations/questioner.py | 83 | 106| 187 | 44802 | 142403 | 
| 158 | 86 django/db/migrations/operations/models.py | 107 | 123| 156 | 44958 | 149387 | 
| 159 | 86 django/core/management/commands/testserver.py | 29 | 55| 234 | 45192 | 149387 | 
| 160 | 86 django/core/management/commands/dumpdata.py | 1 | 79| 565 | 45757 | 149387 | 
| 161 | 86 django/core/checks/security/base.py | 214 | 231| 127 | 45884 | 149387 | 
| 162 | 86 django/core/checks/security/csrf.py | 45 | 68| 157 | 46041 | 149387 | 
| 163 | 86 django/core/management/commands/diffsettings.py | 69 | 80| 143 | 46184 | 149387 | 
| 164 | 86 django/contrib/auth/management/commands/createsuperuser.py | 204 | 228| 204 | 46388 | 149387 | 
| 165 | 86 django/db/backends/sqlite3/creation.py | 1 | 21| 140 | 46528 | 149387 | 
| 166 | 87 django/db/models/options.py | 1 | 35| 300 | 46828 | 156754 | 
| 167 | 88 django/contrib/auth/middleware.py | 48 | 84| 360 | 47188 | 157759 | 
| 168 | 88 django/db/migrations/loader.py | 288 | 312| 205 | 47393 | 157759 | 
| 169 | 89 django/db/migrations/graph.py | 259 | 280| 179 | 47572 | 160362 | 
| 170 | 89 django/db/migrations/autodetector.py | 516 | 532| 186 | 47758 | 160362 | 
| 171 | 90 django/core/signals.py | 1 | 7| 0 | 47758 | 160389 | 
| 172 | 90 django/core/management/base.py | 410 | 477| 622 | 48380 | 160389 | 
| 173 | 91 django/db/backends/base/base.py | 1 | 23| 138 | 48518 | 165292 | 
| 174 | 91 django/db/migrations/executor.py | 64 | 80| 168 | 48686 | 165292 | 
| 175 | 91 django/core/management/base.py | 1 | 18| 115 | 48801 | 165292 | 
| 176 | 92 django/db/migrations/migration.py | 179 | 219| 283 | 49084 | 167114 | 
| 177 | 93 django/core/management/templates.py | 120 | 183| 560 | 49644 | 169789 | 
| 178 | 93 django/core/management/commands/dbshell.py | 1 | 21| 139 | 49783 | 169789 | 
| 179 | 94 django/views/csrf.py | 15 | 100| 835 | 50618 | 171333 | 
| 180 | 95 django/db/backends/oracle/creation.py | 300 | 315| 193 | 50811 | 175226 | 
| 181 | 95 django/core/management/commands/makemessages.py | 170 | 194| 225 | 51036 | 175226 | 
| 182 | 96 django/contrib/contenttypes/migrations/0001_initial.py | 1 | 35| 207 | 51243 | 175433 | 
| 183 | 97 django/db/backends/sqlite3/schema.py | 1 | 37| 317 | 51560 | 179607 | 
| 184 | 97 django/db/utils.py | 255 | 297| 322 | 51882 | 179607 | 
| 185 | 98 django/contrib/auth/management/__init__.py | 35 | 86| 471 | 52353 | 180717 | 
| 186 | 98 django/db/migrations/state.py | 658 | 674| 146 | 52499 | 180717 | 
| 187 | 98 django/db/migrations/autodetector.py | 1077 | 1103| 277 | 52776 | 180717 | 
| 188 | 98 django/core/management/commands/createcachetable.py | 45 | 108| 532 | 53308 | 180717 | 
| 189 | 99 django/db/migrations/exceptions.py | 1 | 55| 249 | 53557 | 180967 | 
| 190 | 99 django/db/backends/mysql/creation.py | 1 | 30| 221 | 53778 | 180967 | 
| 191 | 100 django/db/backends/postgresql/creation.py | 1 | 34| 238 | 54016 | 181613 | 
| 192 | 100 django/db/migrations/executor.py | 281 | 374| 843 | 54859 | 181613 | 
| 193 | 101 django/contrib/gis/db/backends/mysql/base.py | 1 | 17| 0 | 54859 | 181722 | 
| 194 | 101 django/core/management/commands/runserver.py | 1 | 21| 204 | 55063 | 181722 | 
| 195 | 102 django/core/checks/database.py | 1 | 15| 0 | 55063 | 181791 | 
| 196 | 103 django/core/mail/message.py | 338 | 353| 127 | 55190 | 185442 | 
| 197 | 103 django/db/migrations/autodetector.py | 223 | 242| 234 | 55424 | 185442 | 
| 198 | 104 django/conf/__init__.py | 1 | 30| 179 | 55603 | 187179 | 
| 199 | 104 django/core/management/commands/showmigrations.py | 105 | 148| 340 | 55943 | 187179 | 
| 200 | 104 django/db/migrations/autodetector.py | 462 | 514| 465 | 56408 | 187179 | 
| 201 | 105 django/core/management/commands/compilemessages.py | 30 | 57| 230 | 56638 | 188498 | 
| 202 | 106 django/contrib/gis/management/commands/ogrinspect.py | 33 | 96| 591 | 57229 | 189707 | 
| 203 | 107 django/db/backends/postgresql/client.py | 1 | 65| 461 | 57690 | 190168 | 
| 204 | 108 django/db/backends/mysql/validation.py | 1 | 31| 239 | 57929 | 190688 | 
| 205 | 108 django/core/management/templates.py | 211 | 242| 236 | 58165 | 190688 | 
| 206 | 108 django/core/management/templates.py | 58 | 118| 526 | 58691 | 190688 | 
| 207 | 108 django/core/cache/backends/base.py | 1 | 51| 254 | 58945 | 190688 | 
| 208 | 108 django/db/migrations/autodetector.py | 719 | 802| 679 | 59624 | 190688 | 
| 209 | 109 django/db/backends/mysql/operations.py | 1 | 35| 282 | 59906 | 194387 | 


## Missing Patch Files

 * 1: django/contrib/auth/tokens.py

### Hint

```
#29324 fix this issue for management commands that do not rely on the SECRET_KEY, as far as I'm aware check is not one of them. Have you tried with a custom management command?
I am using the makemigrations command with the --check toggle to verify no new migrations are needed. I don’t think it needs a SECRET_KEY?
Here’s a patch that solves the issue for me: ​https://github.com/django/django/pull/14282 That may help describe and reproduce the issue.
I'm not sure about fixing this. It's a reasonable assumption that SECRET_KEY is necessary for all built-in commands. We cannot not guarantee that makemigrations or any other built-in command will work in the future without a SECRET_KEY.
We cannot not guarantee that makemigrations or any other built-in command will work in the future without a SECRET_KEY. That’s true. I’m not asking for all management commands to work without a SECRET_KEY, but for the commands to fail only when the SECRET_KEY is accessed. My goal is to avoid defining a SECRET_KEY in environments that do not need it. That’s the same goal as #29324 and corresponding release note mention ​https://docs.djangoproject.com/en/3.2/releases/3.2/#security: running management commands that do not rely on the SECRET_KEY without needing to provide a value. My project (the same as Jon in #29324) works around the limitation by generating a random string for SECRET_KEY when none is available. It is annoying to maintain logic to create values for unused settings. A regression in this area isn’t a big deal. IMO, maintaining it on a best-effort basis is sufficient, and can help simplifying the running on management commands for other projects.
I think the proper fix would be to move PasswordResetTokenGenerator.secret to a property so it is only accessed when actually needed. This would also help tests that change the SECRET_KEY and then wonder why default_token_generator does not pick it up. Lazy* should be used as a last resort.
Thanks for the suggestion, it is a good improvement! I updated the patch ​https://github.com/django/django/pull/14282 to reflect it. Please let me know if that’s what you had in mind?
```

## Patch

```diff
diff --git a/django/contrib/auth/tokens.py b/django/contrib/auth/tokens.py
--- a/django/contrib/auth/tokens.py
+++ b/django/contrib/auth/tokens.py
@@ -12,12 +12,19 @@ class PasswordResetTokenGenerator:
     """
     key_salt = "django.contrib.auth.tokens.PasswordResetTokenGenerator"
     algorithm = None
-    secret = None
+    _secret = None
 
     def __init__(self):
-        self.secret = self.secret or settings.SECRET_KEY
         self.algorithm = self.algorithm or 'sha256'
 
+    def _get_secret(self):
+        return self._secret or settings.SECRET_KEY
+
+    def _set_secret(self, secret):
+        self._secret = secret
+
+    secret = property(_get_secret, _set_secret)
+
     def make_token(self, user):
         """
         Return a token that can be used once to do a password reset

```

## Test Patch

```diff
diff --git a/tests/auth_tests/test_tokens.py b/tests/auth_tests/test_tokens.py
--- a/tests/auth_tests/test_tokens.py
+++ b/tests/auth_tests/test_tokens.py
@@ -3,7 +3,9 @@
 from django.conf import settings
 from django.contrib.auth.models import User
 from django.contrib.auth.tokens import PasswordResetTokenGenerator
+from django.core.exceptions import ImproperlyConfigured
 from django.test import TestCase
+from django.test.utils import override_settings
 
 from .models import CustomEmailField
 
@@ -111,3 +113,30 @@ def test_token_with_different_secret(self):
         # Tokens created with a different secret don't validate.
         self.assertIs(p0.check_token(user, tk1), False)
         self.assertIs(p1.check_token(user, tk0), False)
+
+    def test_token_with_different_secret_subclass(self):
+        class CustomPasswordResetTokenGenerator(PasswordResetTokenGenerator):
+            secret = 'test-secret'
+
+        user = User.objects.create_user('tokentestuser', 'test2@example.com', 'testpw')
+        custom_password_generator = CustomPasswordResetTokenGenerator()
+        tk_custom = custom_password_generator.make_token(user)
+        self.assertIs(custom_password_generator.check_token(user, tk_custom), True)
+
+        default_password_generator = PasswordResetTokenGenerator()
+        self.assertNotEqual(
+            custom_password_generator.secret,
+            default_password_generator.secret,
+        )
+        self.assertEqual(default_password_generator.secret, settings.SECRET_KEY)
+        # Tokens created with a different secret don't validate.
+        tk_default = default_password_generator.make_token(user)
+        self.assertIs(custom_password_generator.check_token(user, tk_default), False)
+        self.assertIs(default_password_generator.check_token(user, tk_custom), False)
+
+    @override_settings(SECRET_KEY='')
+    def test_secret_lazy_validation(self):
+        default_token_generator = PasswordResetTokenGenerator()
+        msg = 'The SECRET_KEY setting must not be empty.'
+        with self.assertRaisesMessage(ImproperlyConfigured, msg):
+            default_token_generator.secret

```


## Code snippets

### 1 - django/core/management/commands/startproject.py:

Start line: 1, End line: 22

```python
from django.core.checks.security.base import SECRET_KEY_INSECURE_PREFIX
from django.core.management.templates import TemplateCommand

from ..utils import get_random_secret_key


class Command(TemplateCommand):
    help = (
        "Creates a Django project directory structure for the given project "
        "name in the current directory or optionally in the given directory."
    )
    missing_args_message = "You must provide a project name."

    def handle(self, **options):
        project_name = options.pop('name')
        target = options.pop('directory')

        # Create a random SECRET_KEY to put it in the main settings.
        options['secret_key'] = SECRET_KEY_INSECURE_PREFIX + get_random_secret_key()

        super().handle('project', project_name, target, **options)
```
### 2 - django/core/management/commands/makemigrations.py:

Start line: 24, End line: 59

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
```
### 3 - django/core/management/commands/makemigrations.py:

Start line: 154, End line: 192

```python
class Command(BaseCommand):

    @no_translations
    def handle(self, *app_labels, **options):

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
                        self.stdout.write("No changes detected in app '%s'" % app_labels.pop())
                    else:
                        self.stdout.write("No changes detected in apps '%s'" % ("', '".join(app_labels)))
                else:
                    self.stdout.write("No changes detected")
        else:
            self.write_migration_files(changes)
            if check_changes:
                sys.exit(1)
```
### 4 - django/core/management/commands/makemigrations.py:

Start line: 61, End line: 152

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
            self.stdout.write("No conflicts detected to merge.")
            return

        # If they want to merge and there is something to merge, then
        # divert into the merge code
        if self.merge and conflicts:
            return self.handle_merge(loader, conflicts)

        if self.interactive:
            questioner = InteractiveMigrationQuestioner(specified_apps=app_labels, dry_run=self.dry_run)
        else:
            questioner = NonInteractiveMigrationQuestioner(specified_apps=app_labels, dry_run=self.dry_run)
        # Set up autodetector
        autodetector = MigrationAutodetector(
            loader.project_state(),
            ProjectState.from_apps(apps),
            questioner,
        )
        # ... other code
```
### 5 - django/core/checks/security/base.py:

Start line: 199, End line: 211

```python
@register(Tags.security, deploy=True)
def check_secret_key(app_configs, **kwargs):
    try:
        secret_key = settings.SECRET_KEY
    except (ImproperlyConfigured, AttributeError):
        passed_check = False
    else:
        passed_check = (
            len(set(secret_key)) >= SECRET_KEY_MIN_UNIQUE_CHARACTERS and
            len(secret_key) >= SECRET_KEY_MIN_LENGTH and
            not secret_key.startswith(SECRET_KEY_INSECURE_PREFIX)
        )
    return [] if passed_check else [W009]
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
### 7 - django/core/management/commands/makemigrations.py:

Start line: 194, End line: 237

```python
class Command(BaseCommand):

    def write_migration_files(self, changes):
        """
        Take a changes dict and write them out as migration files.
        """
        directory_created = {}
        for app_label, app_migrations in changes.items():
            if self.verbosity >= 1:
                self.stdout.write(self.style.MIGRATE_HEADING("Migrations for '%s':" % app_label))
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
                    self.stdout.write('  %s\n' % self.style.MIGRATE_LABEL(migration_string))
                    for operation in migration.operations:
                        self.stdout.write('    - %s' % operation.describe())
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
                    # will output the migrations to stdout rather than saving
                    # the file to the disk.
                    self.stdout.write(self.style.MIGRATE_HEADING(
                        "Full migrations file '%s':" % writer.filename
                    ))
                    self.stdout.write(writer.as_string())
```
### 8 - django/core/management/commands/migrate.py:

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
### 9 - django/core/management/commands/migrate.py:

Start line: 71, End line: 167

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
                targets = [(app_label, migration.name)]
            target_app_labels_only = False
        elif options['app_label']:
            targets = [key for key in executor.loader.graph.leaf_nodes() if key[0] == app_label]
        else:
            targets = executor.loader.graph.leaf_nodes()

        plan = executor.migration_plan(targets)
        exit_dry = plan and options['check_unapplied']

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
        # ... other code
```
### 10 - django/core/management/commands/sqlmigrate.py:

Start line: 31, End line: 69

```python
class Command(BaseCommand):

    def handle(self, *args, **options):
        # Get the database we're operating from
        connection = connections[options['database']]

        # Load up an loader to get all the migration data, but don't replace
        # migrations.
        loader = MigrationLoader(connection, replace_migrations=False)

        # Resolve command-line arguments into a migration
        app_label, migration_name = options['app_label'], options['migration_name']
        # Validate app_label
        try:
            apps.get_app_config(app_label)
        except LookupError as err:
            raise CommandError(str(err))
        if app_label not in loader.migrated_apps:
            raise CommandError("App '%s' does not have migrations" % app_label)
        try:
            migration = loader.get_migration_by_prefix(app_label, migration_name)
        except AmbiguityError:
            raise CommandError("More than one migration matches '%s' in app '%s'. Please be more specific." % (
                migration_name, app_label))
        except KeyError:
            raise CommandError("Cannot find a migration matching '%s' from app '%s'. Is it in INSTALLED_APPS?" % (
                migration_name, app_label))
        target = (app_label, migration.name)

        # Show begin/end around output for atomic migrations, if the database
        # supports transactional DDL.
        self.output_transaction = migration.atomic and connection.features.can_rollback_ddl

        # Make a plan that represents just the requested migrations and show SQL
        # for it
        plan = [(loader.graph.nodes[target], options['backwards'])]
        sql_statements = loader.collect_sql(plan)
        if not sql_statements and options['verbosity'] >= 1:
            self.stderr.write('No operations found.')
        return '\n'.join(sql_statements)
```
