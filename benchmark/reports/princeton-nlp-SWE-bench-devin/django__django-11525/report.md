# django__django-11525

| **django/django** | `29240a99526f5f2a234a9093cd7e001f32ba1801` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 449 |
| **Any found context length** | 133 |
| **Avg pos** | 4.0 |
| **Min pos** | 1 |
| **Max pos** | 3 |
| **Top file pos** | 1 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/core/mail/__init__.py b/django/core/mail/__init__.py
--- a/django/core/mail/__init__.py
+++ b/django/core/mail/__init__.py
@@ -91,6 +91,8 @@ def mail_admins(subject, message, fail_silently=False, connection=None,
     """Send a message to the admins, as defined by the ADMINS setting."""
     if not settings.ADMINS:
         return
+    if not all(isinstance(a, (list, tuple)) and len(a) == 2 for a in settings.ADMINS):
+        raise ValueError('The ADMINS setting must be a list of 2-tuples.')
     mail = EmailMultiAlternatives(
         '%s%s' % (settings.EMAIL_SUBJECT_PREFIX, subject), message,
         settings.SERVER_EMAIL, [a[1] for a in settings.ADMINS],
@@ -106,6 +108,8 @@ def mail_managers(subject, message, fail_silently=False, connection=None,
     """Send a message to the managers, as defined by the MANAGERS setting."""
     if not settings.MANAGERS:
         return
+    if not all(isinstance(a, (list, tuple)) and len(a) == 2 for a in settings.MANAGERS):
+        raise ValueError('The MANAGERS setting must be a list of 2-tuples.')
     mail = EmailMultiAlternatives(
         '%s%s' % (settings.EMAIL_SUBJECT_PREFIX, subject), message,
         settings.SERVER_EMAIL, [a[1] for a in settings.MANAGERS],

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/core/mail/__init__.py | 94 | 94 | 3 | 1 | 449
| django/core/mail/__init__.py | 109 | 109 | 1 | 1 | 133


## Problem Statement

```
Raise exceptions in mail_admins()/mail_managers() when settings are not in expected formats.
Description
	
Hi,
First time writing a ticket so I apologize if I do anything improperly here. This issue just arose on a project I've been working on, and it goes as follows:
Our MANAGERS setting was set like so:
MANAGERS = ['one@example.com', 'two@example.com']
And after calling mail_managers, the result was:
smtplib.SMTPRecipientsRefused: {'=?utf-8?q?h?=': (550, b'5.1.1 <=?utf-8?q?h?=>: Recipient address rejected: User unknown in local recipient table'), '=?utf-8?q?u?=': (550, b'5.1.1 <=?utf-8?q?u?=>: Recipient address rejected: User unknown in local recipient table')}
After some investigation it became clear that this setting was in the improper format, but that was only because of ​this StackOverflow post. It would be nice if Django failed early if this setting was detected but improperly set, rather than waiting until the consequences become apparent.
Thank you,
Kevin

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 django/core/mail/__init__.py** | 104 | 117| 133 | 133 | 1014 | 
| 2 | 2 django/core/management/commands/sendtestemail.py | 1 | 24| 186 | 319 | 1316 | 
| **-> 3 <-** | **2 django/core/mail/__init__.py** | 89 | 101| 130 | 449 | 1316 | 
| 4 | 3 django/core/mail/message.py | 1 | 52| 339 | 788 | 4871 | 
| 5 | 4 django/conf/global_settings.py | 145 | 263| 876 | 1664 | 10475 | 
| 6 | 4 django/core/management/commands/sendtestemail.py | 26 | 41| 121 | 1785 | 10475 | 
| 7 | 5 django/utils/log.py | 79 | 134| 439 | 2224 | 12083 | 
| 8 | 6 django/core/validators.py | 233 | 277| 330 | 2554 | 16401 | 
| 9 | 6 django/core/mail/message.py | 239 | 264| 322 | 2876 | 16401 | 
| 10 | 6 django/conf/global_settings.py | 499 | 638| 853 | 3729 | 16401 | 
| 11 | 6 django/core/validators.py | 189 | 208| 157 | 3886 | 16401 | 
| 12 | 7 django/core/checks/translation.py | 1 | 62| 447 | 4333 | 16848 | 
| 13 | 8 django/core/checks/security/base.py | 1 | 86| 752 | 5085 | 18474 | 
| 14 | 9 django/middleware/common.py | 149 | 175| 254 | 5339 | 19985 | 
| 15 | 9 django/core/validators.py | 152 | 187| 388 | 5727 | 19985 | 
| 16 | 10 django/core/mail/backends/dummy.py | 1 | 11| 0 | 5727 | 20028 | 
| 17 | 11 django/core/mail/backends/__init__.py | 1 | 2| 0 | 5727 | 20036 | 
| 18 | 11 django/utils/log.py | 1 | 76| 492 | 6219 | 20036 | 
| 19 | 12 django/contrib/admin/options.py | 1 | 96| 769 | 6988 | 38402 | 
| 20 | 13 django/core/checks/templates.py | 1 | 36| 259 | 7247 | 38662 | 
| 21 | 14 django/conf/__init__.py | 132 | 185| 472 | 7719 | 40448 | 
| 22 | 14 django/core/mail/message.py | 172 | 180| 115 | 7834 | 40448 | 
| 23 | 15 django/contrib/admin/exceptions.py | 1 | 12| 0 | 7834 | 40515 | 
| 24 | 15 django/conf/global_settings.py | 347 | 397| 826 | 8660 | 40515 | 
| 25 | 16 django/core/handlers/exception.py | 41 | 102| 499 | 9159 | 41456 | 
| 26 | 17 django/contrib/auth/admin.py | 1 | 22| 188 | 9347 | 43182 | 
| 27 | 18 django/contrib/auth/password_validation.py | 1 | 32| 206 | 9553 | 44666 | 
| 28 | 18 django/core/checks/security/base.py | 88 | 190| 747 | 10300 | 44666 | 
| 29 | 19 django/core/mail/backends/locmem.py | 1 | 31| 183 | 10483 | 44850 | 
| 30 | 20 django/contrib/sites/managers.py | 1 | 61| 385 | 10868 | 45235 | 
| 31 | 21 django/http/response.py | 1 | 25| 144 | 11012 | 49735 | 
| 32 | **21 django/core/mail/__init__.py** | 1 | 35| 320 | 11332 | 49735 | 
| 33 | 22 setup.py | 69 | 138| 536 | 11868 | 50758 | 
| 34 | 22 django/conf/global_settings.py | 264 | 346| 800 | 12668 | 50758 | 
| 35 | 23 django/views/debug.py | 72 | 95| 196 | 12864 | 54976 | 
| 36 | 24 django/conf/locale/ka/formats.py | 5 | 22| 298 | 13162 | 55883 | 
| 37 | 24 django/core/mail/message.py | 329 | 344| 127 | 13289 | 55883 | 
| 38 | 25 django/conf/locale/en_AU/formats.py | 5 | 40| 708 | 13997 | 56636 | 
| 39 | 26 django/core/checks/messages.py | 53 | 76| 161 | 14158 | 57209 | 
| 40 | 27 django/conf/locale/sk/formats.py | 5 | 30| 348 | 14506 | 57602 | 
| 41 | 28 django/contrib/auth/migrations/0003_alter_user_email_max_length.py | 1 | 17| 0 | 14506 | 57680 | 
| 42 | 29 django/core/mail/backends/smtp.py | 1 | 39| 364 | 14870 | 58708 | 
| 43 | 30 django/conf/locale/cs/formats.py | 5 | 43| 640 | 15510 | 59393 | 
| 44 | 31 django/conf/locale/id/formats.py | 5 | 50| 708 | 16218 | 60146 | 
| 45 | 32 django/conf/locale/uk/formats.py | 5 | 38| 460 | 16678 | 60651 | 
| 46 | 33 django/conf/locale/ml/formats.py | 5 | 41| 663 | 17341 | 61359 | 
| 47 | 34 django/conf/locale/es_MX/formats.py | 3 | 26| 289 | 17630 | 61664 | 
| 48 | 35 django/conf/locale/pl/formats.py | 5 | 30| 339 | 17969 | 62048 | 
| 49 | 36 django/conf/locale/lt/formats.py | 5 | 46| 711 | 18680 | 62804 | 
| 50 | 37 django/http/request.py | 1 | 37| 251 | 18931 | 67634 | 
| 51 | 37 django/core/validators.py | 210 | 230| 139 | 19070 | 67634 | 
| 52 | 38 django/conf/locale/pt/formats.py | 5 | 39| 630 | 19700 | 68309 | 
| 53 | 39 django/core/management/commands/makemessages.py | 283 | 362| 816 | 20516 | 73869 | 
| 54 | 40 django/views/defaults.py | 100 | 119| 149 | 20665 | 74911 | 
| 55 | 41 django/conf/locale/en_GB/formats.py | 5 | 40| 708 | 21373 | 75664 | 
| 56 | 42 django/contrib/auth/management/commands/createsuperuser.py | 171 | 195| 189 | 21562 | 77379 | 
| 57 | 43 django/conf/locale/az/formats.py | 5 | 33| 399 | 21961 | 77823 | 
| 58 | 44 django/conf/locale/mk/formats.py | 5 | 43| 672 | 22633 | 78540 | 
| 59 | 45 django/conf/locale/hu/formats.py | 5 | 32| 323 | 22956 | 78908 | 
| 60 | 46 django/conf/locale/de/formats.py | 5 | 29| 323 | 23279 | 79276 | 
| 61 | 47 django/conf/locale/lv/formats.py | 5 | 47| 735 | 24014 | 80056 | 
| 62 | 48 django/core/mail/utils.py | 1 | 21| 113 | 24127 | 80169 | 
| 63 | 49 django/conf/locale/ru/formats.py | 5 | 33| 402 | 24529 | 80616 | 
| 64 | 49 django/contrib/auth/management/commands/createsuperuser.py | 197 | 212| 139 | 24668 | 80616 | 
| 65 | 50 django/conf/locale/el/formats.py | 5 | 36| 508 | 25176 | 81169 | 
| 66 | 50 django/core/mail/message.py | 55 | 71| 171 | 25347 | 81169 | 
| 67 | 50 django/conf/locale/ka/formats.py | 23 | 48| 564 | 25911 | 81169 | 
| 68 | 51 django/conf/locale/en/formats.py | 5 | 41| 663 | 26574 | 81877 | 
| 69 | 52 django/conf/locale/fr/formats.py | 5 | 34| 489 | 27063 | 82411 | 
| 70 | 53 django/contrib/auth/migrations/0007_alter_validators_add_error_messages.py | 1 | 25| 137 | 27200 | 82548 | 
| 71 | 54 django/conf/locale/pt_BR/formats.py | 5 | 34| 494 | 27694 | 83087 | 
| 72 | 55 django/conf/locale/nn/formats.py | 5 | 41| 664 | 28358 | 83796 | 
| 73 | 56 django/conf/locale/sr/formats.py | 5 | 22| 293 | 28651 | 84645 | 
| 74 | 57 django/contrib/admin/sites.py | 1 | 29| 175 | 28826 | 88836 | 
| 75 | 58 django/conf/locale/de_CH/formats.py | 5 | 35| 416 | 29242 | 89297 | 
| 76 | 59 django/conf/locale/es/formats.py | 5 | 31| 285 | 29527 | 89627 | 
| 77 | 60 django/contrib/admin/models.py | 1 | 20| 118 | 29645 | 90752 | 
| 78 | 61 django/conf/locale/km/formats.py | 5 | 22| 164 | 29809 | 90960 | 
| 79 | 62 django/conf/locale/ca/formats.py | 5 | 31| 287 | 30096 | 91292 | 
| 80 | 62 django/views/debug.py | 244 | 327| 745 | 30841 | 91292 | 
| 81 | 63 django/conf/locale/sr_Latn/formats.py | 5 | 22| 293 | 31134 | 92141 | 
| 82 | 64 django/conf/locale/es_NI/formats.py | 3 | 27| 270 | 31404 | 92427 | 
| 83 | 65 django/contrib/auth/forms.py | 240 | 262| 196 | 31600 | 95485 | 
| 84 | 66 django/conf/locale/nb/formats.py | 5 | 40| 646 | 32246 | 96176 | 
| 85 | 67 django/conf/locale/es_PR/formats.py | 3 | 28| 252 | 32498 | 96444 | 
| 86 | 68 django/conf/locale/kn/formats.py | 5 | 22| 123 | 32621 | 96611 | 
| 87 | 68 django/core/management/commands/makemessages.py | 1 | 33| 247 | 32868 | 96611 | 
| 88 | 68 django/conf/global_settings.py | 398 | 497| 793 | 33661 | 96611 | 
| 89 | 68 django/views/debug.py | 154 | 177| 176 | 33837 | 96611 | 
| 90 | 69 django/conf/locale/bn/formats.py | 5 | 33| 294 | 34131 | 96949 | 
| 91 | 70 django/views/csrf.py | 15 | 100| 835 | 34966 | 98493 | 
| 92 | 71 django/contrib/auth/checks.py | 1 | 94| 646 | 35612 | 99666 | 
| 93 | 72 django/conf/locale/es_CO/formats.py | 3 | 27| 262 | 35874 | 99944 | 
| 94 | 73 django/contrib/admin/checks.py | 129 | 146| 155 | 36029 | 108958 | 
| 95 | 74 django/conf/locale/fi/formats.py | 5 | 40| 470 | 36499 | 109473 | 
| 96 | 75 django/conf/locale/hr/formats.py | 5 | 21| 218 | 36717 | 110356 | 
| 97 | 75 django/core/mail/message.py | 150 | 169| 218 | 36935 | 110356 | 
| 98 | 76 django/contrib/admin/views/main.py | 1 | 34| 244 | 37179 | 114433 | 
| 99 | 77 django/core/checks/security/csrf.py | 1 | 41| 299 | 37478 | 114732 | 
| 100 | 77 django/views/debug.py | 179 | 191| 146 | 37624 | 114732 | 
| 101 | 78 django/contrib/auth/validators.py | 1 | 26| 165 | 37789 | 114898 | 
| 102 | 79 django/conf/locale/ro/formats.py | 5 | 36| 262 | 38051 | 115205 | 
| 103 | 80 django/conf/locale/cy/formats.py | 5 | 36| 582 | 38633 | 115832 | 
| 104 | 80 django/middleware/common.py | 118 | 147| 277 | 38910 | 115832 | 
| 105 | 81 django/contrib/admindocs/utils.py | 1 | 23| 133 | 39043 | 117732 | 
| 106 | 82 django/core/mail/backends/console.py | 1 | 43| 281 | 39324 | 118014 | 
| 107 | 82 django/conf/locale/sr/formats.py | 23 | 44| 511 | 39835 | 118014 | 
| 108 | 82 django/contrib/admin/checks.py | 620 | 639| 183 | 40018 | 118014 | 
| 109 | 83 django/conf/locale/da/formats.py | 5 | 27| 250 | 40268 | 118309 | 
| 110 | 84 django/core/management/base.py | 384 | 449| 614 | 40882 | 122696 | 
| 111 | 84 django/conf/locale/hr/formats.py | 22 | 48| 620 | 41502 | 122696 | 
| 112 | 85 django/conf/locale/sv/formats.py | 5 | 39| 534 | 42036 | 123275 | 
| 113 | 86 django/conf/locale/es_AR/formats.py | 5 | 31| 275 | 42311 | 123595 | 
| 114 | 86 django/conf/global_settings.py | 1 | 50| 366 | 42677 | 123595 | 
| 115 | 86 setup.py | 1 | 66| 508 | 43185 | 123595 | 
| 116 | 86 django/contrib/auth/admin.py | 128 | 189| 465 | 43650 | 123595 | 
| 117 | 86 django/core/management/commands/makemessages.py | 394 | 416| 200 | 43850 | 123595 | 
| 118 | 86 django/core/mail/backends/smtp.py | 116 | 131| 137 | 43987 | 123595 | 
| 119 | 87 django/conf/locale/gd/formats.py | 5 | 22| 144 | 44131 | 123783 | 
| 120 | 87 django/core/checks/messages.py | 26 | 50| 259 | 44390 | 123783 | 
| 121 | 88 django/core/checks/security/sessions.py | 1 | 98| 572 | 44962 | 124356 | 
| 122 | 89 django/conf/locale/is/formats.py | 5 | 22| 130 | 45092 | 124531 | 
| 123 | 90 django/conf/locale/et/formats.py | 5 | 22| 133 | 45225 | 124708 | 
| 124 | 91 django/conf/locale/eo/formats.py | 5 | 50| 742 | 45967 | 125495 | 
| 125 | 91 django/contrib/admin/checks.py | 1013 | 1040| 204 | 46171 | 125495 | 
| 126 | 91 django/contrib/admin/checks.py | 879 | 927| 416 | 46587 | 125495 | 
| 127 | 92 django/db/backends/mysql/validation.py | 1 | 27| 248 | 46835 | 125977 | 
| 128 | 93 django/contrib/admin/__init__.py | 1 | 30| 286 | 47121 | 126263 | 
| 129 | 93 django/conf/locale/sr_Latn/formats.py | 23 | 44| 511 | 47632 | 126263 | 
| 130 | 93 django/contrib/admin/models.py | 23 | 36| 111 | 47743 | 126263 | 
| 131 | 94 django/core/mail/backends/filebased.py | 1 | 68| 547 | 48290 | 126811 | 
| 132 | 95 django/contrib/auth/models.py | 1 | 30| 200 | 48490 | 129814 | 
| 133 | 95 django/core/handlers/exception.py | 105 | 130| 184 | 48674 | 129814 | 
| 134 | 96 django/contrib/auth/hashers.py | 80 | 102| 167 | 48841 | 134601 | 
| 135 | 97 django/forms/models.py | 309 | 348| 387 | 49228 | 146096 | 
| 136 | 98 django/conf/locale/zh_Hans/formats.py | 5 | 43| 635 | 49863 | 146776 | 
| 137 | 98 django/contrib/admin/options.py | 1517 | 1596| 719 | 50582 | 146776 | 
| 138 | 99 django/db/utils.py | 1 | 49| 154 | 50736 | 148888 | 
| 139 | 100 django/conf/locale/zh_Hant/formats.py | 5 | 43| 635 | 51371 | 149568 | 
| 140 | 101 django/conf/locale/sl/formats.py | 5 | 20| 208 | 51579 | 150417 | 
| 141 | 101 django/core/management/commands/makemessages.py | 363 | 392| 231 | 51810 | 150417 | 
| 142 | 101 django/core/checks/security/base.py | 193 | 211| 127 | 51937 | 150417 | 
| 143 | 102 django/contrib/gis/geoip2/base.py | 1 | 21| 141 | 52078 | 152453 | 
| 144 | 103 django/conf/locale/bg/formats.py | 5 | 22| 131 | 52209 | 152628 | 
| 145 | 103 django/conf/locale/sl/formats.py | 22 | 48| 596 | 52805 | 152628 | 
| 146 | 103 django/db/utils.py | 52 | 98| 312 | 53117 | 152628 | 
| 147 | 104 django/db/models/fields/__init__.py | 1637 | 1658| 183 | 53300 | 169639 | 
| 148 | 105 django/contrib/auth/migrations/0004_alter_user_username_opts.py | 1 | 24| 150 | 53450 | 169789 | 
| 149 | 106 django/conf/locale/sq/formats.py | 5 | 22| 128 | 53578 | 169961 | 
| 150 | 106 django/contrib/admin/checks.py | 1087 | 1117| 188 | 53766 | 169961 | 
| 151 | 106 django/db/models/fields/__init__.py | 1111 | 1140| 218 | 53984 | 169961 | 
| 152 | 107 django/conf/locale/gl/formats.py | 5 | 22| 170 | 54154 | 170175 | 
| 153 | 108 django/conf/locale/nl/formats.py | 5 | 71| 479 | 54633 | 172206 | 
| 154 | 109 django/conf/locale/mn/formats.py | 5 | 22| 120 | 54753 | 172370 | 
| 155 | 109 django/core/checks/messages.py | 1 | 24| 156 | 54909 | 172370 | 
| 156 | 109 django/contrib/admin/options.py | 1045 | 1068| 234 | 55143 | 172370 | 
| 157 | 109 django/contrib/admin/checks.py | 1042 | 1084| 343 | 55486 | 172370 | 
| 158 | 110 django/conf/locale/ar/formats.py | 5 | 22| 135 | 55621 | 172549 | 
| 159 | 111 django/conf/locale/ko/formats.py | 32 | 53| 438 | 56059 | 173492 | 
| 160 | 112 django/conf/locale/hi/formats.py | 5 | 22| 125 | 56184 | 173661 | 
| 161 | 112 django/views/debug.py | 1 | 45| 306 | 56490 | 173661 | 
| 162 | 113 django/db/models/base.py | 1294 | 1323| 205 | 56695 | 188621 | 
| 163 | 114 django/conf/locale/it/formats.py | 21 | 46| 564 | 57259 | 189518 | 
| 164 | 115 django/contrib/auth/migrations/0001_initial.py | 1 | 105| 849 | 58108 | 190367 | 
| 165 | 116 django/conf/locale/ga/formats.py | 5 | 22| 124 | 58232 | 190535 | 
| 166 | 116 django/contrib/admin/options.py | 2135 | 2170| 315 | 58547 | 190535 | 
| 167 | 117 django/conf/locale/te/formats.py | 5 | 22| 123 | 58670 | 190702 | 
| 168 | 117 django/db/models/fields/__init__.py | 1301 | 1342| 332 | 59002 | 190702 | 
| 169 | 117 django/db/models/base.py | 1 | 45| 289 | 59291 | 190702 | 
| 170 | 118 django/conf/locale/tr/formats.py | 5 | 30| 337 | 59628 | 191084 | 
| 171 | 119 django/conf/locale/he/formats.py | 5 | 22| 142 | 59770 | 191270 | 
| 172 | 120 django/contrib/admin/helpers.py | 33 | 67| 230 | 60000 | 194463 | 
| 173 | 120 django/db/models/base.py | 1236 | 1265| 242 | 60242 | 194463 | 
| 174 | 121 django/conf/locale/ta/formats.py | 5 | 22| 125 | 60367 | 194632 | 
| 175 | 121 django/contrib/admin/helpers.py | 1 | 30| 198 | 60565 | 194632 | 
| 176 | 122 django/contrib/admin/migrations/0001_initial.py | 1 | 48| 322 | 60887 | 194954 | 
| 177 | 122 django/contrib/admin/checks.py | 521 | 544| 230 | 61117 | 194954 | 
| 178 | 123 django/conf/locale/bs/formats.py | 5 | 22| 139 | 61256 | 195137 | 
| 179 | 123 django/views/debug.py | 125 | 152| 248 | 61504 | 195137 | 
| 180 | 123 django/contrib/admin/options.py | 1337 | 1402| 581 | 62085 | 195137 | 
| 181 | 123 django/contrib/auth/models.py | 128 | 158| 273 | 62358 | 195137 | 
| 182 | 123 django/core/management/base.py | 1 | 36| 223 | 62581 | 195137 | 


### Hint

```
Thanks for the report. It seems reasonable to raise ValueError in mail_admins() and mail_managers() when settings are not in expected formats.
```

## Patch

```diff
diff --git a/django/core/mail/__init__.py b/django/core/mail/__init__.py
--- a/django/core/mail/__init__.py
+++ b/django/core/mail/__init__.py
@@ -91,6 +91,8 @@ def mail_admins(subject, message, fail_silently=False, connection=None,
     """Send a message to the admins, as defined by the ADMINS setting."""
     if not settings.ADMINS:
         return
+    if not all(isinstance(a, (list, tuple)) and len(a) == 2 for a in settings.ADMINS):
+        raise ValueError('The ADMINS setting must be a list of 2-tuples.')
     mail = EmailMultiAlternatives(
         '%s%s' % (settings.EMAIL_SUBJECT_PREFIX, subject), message,
         settings.SERVER_EMAIL, [a[1] for a in settings.ADMINS],
@@ -106,6 +108,8 @@ def mail_managers(subject, message, fail_silently=False, connection=None,
     """Send a message to the managers, as defined by the MANAGERS setting."""
     if not settings.MANAGERS:
         return
+    if not all(isinstance(a, (list, tuple)) and len(a) == 2 for a in settings.MANAGERS):
+        raise ValueError('The MANAGERS setting must be a list of 2-tuples.')
     mail = EmailMultiAlternatives(
         '%s%s' % (settings.EMAIL_SUBJECT_PREFIX, subject), message,
         settings.SERVER_EMAIL, [a[1] for a in settings.MANAGERS],

```

## Test Patch

```diff
diff --git a/tests/mail/tests.py b/tests/mail/tests.py
--- a/tests/mail/tests.py
+++ b/tests/mail/tests.py
@@ -991,6 +991,23 @@ def test_empty_admins(self):
         mail_managers('hi', 'there')
         self.assertEqual(self.get_mailbox_content(), [])
 
+    def test_wrong_admins_managers(self):
+        tests = (
+            'test@example.com',
+            ('test@example.com',),
+            ['test@example.com', 'other@example.com'],
+            ('test@example.com', 'other@example.com'),
+        )
+        for setting, mail_func in (
+            ('ADMINS', mail_admins),
+            ('MANAGERS', mail_managers),
+        ):
+            msg = 'The %s setting must be a list of 2-tuples.' % setting
+            for value in tests:
+                with self.subTest(setting=setting, value=value), self.settings(**{setting: value}):
+                    with self.assertRaisesMessage(ValueError, msg):
+                        mail_func('subject', 'content')
+
     def test_message_cc_header(self):
         """
         Regression test for #7722
diff --git a/tests/middleware/tests.py b/tests/middleware/tests.py
--- a/tests/middleware/tests.py
+++ b/tests/middleware/tests.py
@@ -340,7 +340,7 @@ class MyCommonMiddleware(CommonMiddleware):
 
 @override_settings(
     IGNORABLE_404_URLS=[re.compile(r'foo')],
-    MANAGERS=['PHB@dilbert.com'],
+    MANAGERS=[('PHD', 'PHB@dilbert.com')],
 )
 class BrokenLinkEmailsMiddlewareTest(SimpleTestCase):
 

```


## Code snippets

### 1 - django/core/mail/__init__.py:

Start line: 104, End line: 117

```python
def mail_managers(subject, message, fail_silently=False, connection=None,
                  html_message=None):
    """Send a message to the managers, as defined by the MANAGERS setting."""
    if not settings.MANAGERS:
        return
    mail = EmailMultiAlternatives(
        '%s%s' % (settings.EMAIL_SUBJECT_PREFIX, subject), message,
        settings.SERVER_EMAIL, [a[1] for a in settings.MANAGERS],
        connection=connection,
    )
    if html_message:
        mail.attach_alternative(html_message, 'text/html')
    mail.send(fail_silently=fail_silently)
```
### 2 - django/core/management/commands/sendtestemail.py:

Start line: 1, End line: 24

```python
import socket

from django.core.mail import mail_admins, mail_managers, send_mail
from django.core.management.base import BaseCommand
from django.utils import timezone


class Command(BaseCommand):
    help = "Sends a test email to the email addresses specified as arguments."
    missing_args_message = "You must specify some email recipients, or pass the --managers or --admin options."

    def add_arguments(self, parser):
        parser.add_argument(
            'email', nargs='*',
            help='One or more email addresses to send a test email to.',
        )
        parser.add_argument(
            '--managers', action='store_true',
            help='Send a test email to the addresses specified in settings.MANAGERS.',
        )
        parser.add_argument(
            '--admins', action='store_true',
            help='Send a test email to the addresses specified in settings.ADMINS.',
        )
```
### 3 - django/core/mail/__init__.py:

Start line: 89, End line: 101

```python
def mail_admins(subject, message, fail_silently=False, connection=None,
                html_message=None):
    """Send a message to the admins, as defined by the ADMINS setting."""
    if not settings.ADMINS:
        return
    mail = EmailMultiAlternatives(
        '%s%s' % (settings.EMAIL_SUBJECT_PREFIX, subject), message,
        settings.SERVER_EMAIL, [a[1] for a in settings.ADMINS],
        connection=connection,
    )
    if html_message:
        mail.attach_alternative(html_message, 'text/html')
    mail.send(fail_silently=fail_silently)
```
### 4 - django/core/mail/message.py:

Start line: 1, End line: 52

```python
import mimetypes
from email import (
    charset as Charset, encoders as Encoders, generator, message_from_string,
)
from email.errors import HeaderParseError
from email.header import Header
from email.headerregistry import Address, parser
from email.message import Message
from email.mime.base import MIMEBase
from email.mime.message import MIMEMessage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formatdate, getaddresses, make_msgid
from io import BytesIO, StringIO
from pathlib import Path

from django.conf import settings
from django.core.mail.utils import DNS_NAME
from django.utils.encoding import force_str

# Don't BASE64-encode UTF-8 messages so that we avoid unwanted attention from
# some spam filters.
utf8_charset = Charset.Charset('utf-8')
utf8_charset.body_encoding = None  # Python defaults to BASE64
utf8_charset_qp = Charset.Charset('utf-8')
utf8_charset_qp.body_encoding = Charset.QP

# Default MIME type to use on attachments (if it is not explicitly given
# and cannot be guessed).
DEFAULT_ATTACHMENT_MIME_TYPE = 'application/octet-stream'

RFC5322_EMAIL_LINE_LENGTH_LIMIT = 998


class BadHeaderError(ValueError):
    pass


# Header names that contain structured address data (RFC #5322)
ADDRESS_HEADERS = {
    'from',
    'sender',
    'reply-to',
    'to',
    'cc',
    'bcc',
    'resent-from',
    'resent-sender',
    'resent-to',
    'resent-cc',
    'resent-bcc',
}
```
### 5 - django/conf/global_settings.py:

Start line: 145, End line: 263

```python
LANGUAGES_BIDI = ["he", "ar", "fa", "ur"]

# If you set this to False, Django will make some optimizations so as not
# to load the internationalization machinery.
USE_I18N = True
LOCALE_PATHS = []

# Settings for language cookie
LANGUAGE_COOKIE_NAME = 'django_language'
LANGUAGE_COOKIE_AGE = None
LANGUAGE_COOKIE_DOMAIN = None
LANGUAGE_COOKIE_PATH = '/'
LANGUAGE_COOKIE_SECURE = False
LANGUAGE_COOKIE_HTTPONLY = False
LANGUAGE_COOKIE_SAMESITE = None


# If you set this to True, Django will format dates, numbers and calendars
# according to user current locale.
USE_L10N = False

# Not-necessarily-technical managers of the site. They get broken link
# notifications and other various emails.
MANAGERS = ADMINS

# Default charset to use for all HttpResponse objects, if a MIME type isn't
# manually specified. It's used to construct the Content-Type header.
DEFAULT_CHARSET = 'utf-8'

# Encoding of files read from disk (template and initial SQL files).
FILE_CHARSET = 'utf-8'

# Email address that error messages come from.
SERVER_EMAIL = 'root@localhost'

# Database connection info. If left empty, will default to the dummy backend.
DATABASES = {}

# Classes used to implement DB routing behavior.
DATABASE_ROUTERS = []

# The email backend to use. For possible shortcuts see django.core.mail.
# The default is to use the SMTP backend.
# Third-party backends can be specified by providing a Python path
# to a module that defines an EmailBackend class.
EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'

# Host for sending email.
EMAIL_HOST = 'localhost'

# Port for sending email.
EMAIL_PORT = 25

# Whether to send SMTP 'Date' header in the local time zone or in UTC.
EMAIL_USE_LOCALTIME = False

# Optional SMTP authentication information for EMAIL_HOST.
EMAIL_HOST_USER = ''
EMAIL_HOST_PASSWORD = ''
EMAIL_USE_TLS = False
EMAIL_USE_SSL = False
EMAIL_SSL_CERTFILE = None
EMAIL_SSL_KEYFILE = None
EMAIL_TIMEOUT = None

# List of strings representing installed apps.
INSTALLED_APPS = []

TEMPLATES = []

# Default form rendering class.
FORM_RENDERER = 'django.forms.renderers.DjangoTemplates'

# Default email address to use for various automated correspondence from
# the site managers.
DEFAULT_FROM_EMAIL = 'webmaster@localhost'

# Subject-line prefix for email messages send with django.core.mail.mail_admins
# or ...mail_managers.  Make sure to include the trailing space.
EMAIL_SUBJECT_PREFIX = '[Django] '

# Whether to append trailing slashes to URLs.
APPEND_SLASH = True

# Whether to prepend the "www." subdomain to URLs that don't have it.
PREPEND_WWW = False

# Override the server-derived value of SCRIPT_NAME
FORCE_SCRIPT_NAME = None

# List of compiled regular expression objects representing User-Agent strings
# that are not allowed to visit any page, systemwide. Use this for bad
# robots/crawlers. Here are a few examples:
#     import re
#     DISALLOWED_USER_AGENTS = [
#         re.compile(r'^NaverBot.*'),
#         re.compile(r'^EmailSiphon.*'),
#         re.compile(r'^SiteSucker.*'),
#         re.compile(r'^sohu-search'),
#     ]
DISALLOWED_USER_AGENTS = []

ABSOLUTE_URL_OVERRIDES = {}

# List of compiled regular expression objects representing URLs that need not
# be reported by BrokenLinkEmailsMiddleware. Here are a few examples:
#    import re
#    IGNORABLE_404_URLS = [
#        re.compile(r'^/apple-touch-icon.*\.png$'),
#        re.compile(r'^/favicon.ico$'),
#        re.compile(r'^/robots.txt$'),
#        re.compile(r'^/phpmyadmin/'),
#        re.compile(r'\.(cgi|php|pl)$'),
#    ]
IGNORABLE_404_URLS = []

# A secret key for this particular Django installation. Used in secret-key
# hashing algorithms. Set this in your settings, or Django will complain
# loudly.
```
### 6 - django/core/management/commands/sendtestemail.py:

Start line: 26, End line: 41

```python
class Command(BaseCommand):

    def handle(self, *args, **kwargs):
        subject = 'Test email from %s on %s' % (socket.gethostname(), timezone.now())

        send_mail(
            subject=subject,
            message="If you\'re reading this, it was successful.",
            from_email=None,
            recipient_list=kwargs['email'],
        )

        if kwargs['managers']:
            mail_managers(subject, "This email was sent to the site managers.")

        if kwargs['admins']:
            mail_admins(subject, "This email was sent to the site admins.")
```
### 7 - django/utils/log.py:

Start line: 79, End line: 134

```python
class AdminEmailHandler(logging.Handler):
    """An exception log handler that emails log entries to site admins.

    If the request is passed as the first argument to the log record,
    request data will be provided in the email report.
    """

    def __init__(self, include_html=False, email_backend=None):
        super().__init__()
        self.include_html = include_html
        self.email_backend = email_backend

    def emit(self, record):
        try:
            request = record.request
            subject = '%s (%s IP): %s' % (
                record.levelname,
                ('internal' if request.META.get('REMOTE_ADDR') in settings.INTERNAL_IPS
                 else 'EXTERNAL'),
                record.getMessage()
            )
        except Exception:
            subject = '%s: %s' % (
                record.levelname,
                record.getMessage()
            )
            request = None
        subject = self.format_subject(subject)

        # Since we add a nicely formatted traceback on our own, create a copy
        # of the log record without the exception data.
        no_exc_record = copy(record)
        no_exc_record.exc_info = None
        no_exc_record.exc_text = None

        if record.exc_info:
            exc_info = record.exc_info
        else:
            exc_info = (None, record.getMessage(), None)

        reporter = ExceptionReporter(request, is_email=True, *exc_info)
        message = "%s\n\n%s" % (self.format(no_exc_record), reporter.get_traceback_text())
        html_message = reporter.get_traceback_html() if self.include_html else None
        self.send_mail(subject, message, fail_silently=True, html_message=html_message)

    def send_mail(self, subject, message, *args, **kwargs):
        mail.mail_admins(subject, message, *args, connection=self.connection(), **kwargs)

    def connection(self):
        return get_connection(backend=self.email_backend, fail_silently=True)

    def format_subject(self, subject):
        """
        Escape CR and LF characters.
        """
        return subject.replace('\n', '\\n').replace('\r', '\\r')
```
### 8 - django/core/validators.py:

Start line: 233, End line: 277

```python
validate_email = EmailValidator()

slug_re = _lazy_re_compile(r'^[-a-zA-Z0-9_]+\Z')
validate_slug = RegexValidator(
    slug_re,
    # Translators: "letters" means latin letters: a-z and A-Z.
    _('Enter a valid “slug” consisting of letters, numbers, underscores or hyphens.'),
    'invalid'
)

slug_unicode_re = _lazy_re_compile(r'^[-\w]+\Z')
validate_unicode_slug = RegexValidator(
    slug_unicode_re,
    _('Enter a valid “slug” consisting of Unicode letters, numbers, underscores, or hyphens.'),
    'invalid'
)


def validate_ipv4_address(value):
    try:
        ipaddress.IPv4Address(value)
    except ValueError:
        raise ValidationError(_('Enter a valid IPv4 address.'), code='invalid')


def validate_ipv6_address(value):
    if not is_valid_ipv6_address(value):
        raise ValidationError(_('Enter a valid IPv6 address.'), code='invalid')


def validate_ipv46_address(value):
    try:
        validate_ipv4_address(value)
    except ValidationError:
        try:
            validate_ipv6_address(value)
        except ValidationError:
            raise ValidationError(_('Enter a valid IPv4 or IPv6 address.'), code='invalid')


ip_address_validator_map = {
    'both': ([validate_ipv46_address], _('Enter a valid IPv4 or IPv6 address.')),
    'ipv4': ([validate_ipv4_address], _('Enter a valid IPv4 address.')),
    'ipv6': ([validate_ipv6_address], _('Enter a valid IPv6 address.')),
}
```
### 9 - django/core/mail/message.py:

Start line: 239, End line: 264

```python
class EmailMessage:

    def message(self):
        encoding = self.encoding or settings.DEFAULT_CHARSET
        msg = SafeMIMEText(self.body, self.content_subtype, encoding)
        msg = self._create_message(msg)
        msg['Subject'] = self.subject
        msg['From'] = self.extra_headers.get('From', self.from_email)
        self._set_list_header_if_not_empty(msg, 'To', self.to)
        self._set_list_header_if_not_empty(msg, 'Cc', self.cc)
        self._set_list_header_if_not_empty(msg, 'Reply-To', self.reply_to)

        # Email header names are case-insensitive (RFC 2045), so we have to
        # accommodate that when doing comparisons.
        header_names = [key.lower() for key in self.extra_headers]
        if 'date' not in header_names:
            # formatdate() uses stdlib methods to format the date, which use
            # the stdlib/OS concept of a timezone, however, Django sets the
            # TZ environment variable based on the TIME_ZONE setting which
            # will get picked up by formatdate().
            msg['Date'] = formatdate(localtime=settings.EMAIL_USE_LOCALTIME)
        if 'message-id' not in header_names:
            # Use cached DNS_NAME for performance
            msg['Message-ID'] = make_msgid(domain=DNS_NAME)
        for name, value in self.extra_headers.items():
            if name.lower() != 'from':  # From is already handled
                msg[name] = value
        return msg
```
### 10 - django/conf/global_settings.py:

Start line: 499, End line: 638

```python
AUTH_USER_MODEL = 'auth.User'

AUTHENTICATION_BACKENDS = ['django.contrib.auth.backends.ModelBackend']

LOGIN_URL = '/accounts/login/'

LOGIN_REDIRECT_URL = '/accounts/profile/'

LOGOUT_REDIRECT_URL = None

# The number of days a password reset link is valid for
PASSWORD_RESET_TIMEOUT_DAYS = 3

# the first hasher in this list is the preferred algorithm.  any
# password using different algorithms will be converted automatically
# upon login
PASSWORD_HASHERS = [
    'django.contrib.auth.hashers.PBKDF2PasswordHasher',
    'django.contrib.auth.hashers.PBKDF2SHA1PasswordHasher',
    'django.contrib.auth.hashers.Argon2PasswordHasher',
    'django.contrib.auth.hashers.BCryptSHA256PasswordHasher',
]

AUTH_PASSWORD_VALIDATORS = []

###########
# SIGNING #
###########

SIGNING_BACKEND = 'django.core.signing.TimestampSigner'

########
# CSRF #
########

# Dotted path to callable to be used as view when a request is
# rejected by the CSRF middleware.
CSRF_FAILURE_VIEW = 'django.views.csrf.csrf_failure'

# Settings for CSRF cookie.
CSRF_COOKIE_NAME = 'csrftoken'
CSRF_COOKIE_AGE = 60 * 60 * 24 * 7 * 52
CSRF_COOKIE_DOMAIN = None
CSRF_COOKIE_PATH = '/'
CSRF_COOKIE_SECURE = False
CSRF_COOKIE_HTTPONLY = False
CSRF_COOKIE_SAMESITE = 'Lax'
CSRF_HEADER_NAME = 'HTTP_X_CSRFTOKEN'
CSRF_TRUSTED_ORIGINS = []
CSRF_USE_SESSIONS = False

############
# MESSAGES #
############

# Class to use as messages backend
MESSAGE_STORAGE = 'django.contrib.messages.storage.fallback.FallbackStorage'

# Default values of MESSAGE_LEVEL and MESSAGE_TAGS are defined within
# django.contrib.messages to avoid imports in this settings file.

###########
# LOGGING #
###########

# The callable to use to configure logging
LOGGING_CONFIG = 'logging.config.dictConfig'

# Custom logging configuration.
LOGGING = {}

# Default exception reporter filter class used in case none has been
# specifically assigned to the HttpRequest instance.
DEFAULT_EXCEPTION_REPORTER_FILTER = 'django.views.debug.SafeExceptionReporterFilter'

###########
# TESTING #
###########

# The name of the class to use to run the test suite
TEST_RUNNER = 'django.test.runner.DiscoverRunner'

# Apps that don't need to be serialized at test database creation time
# (only apps with migrations are to start with)
TEST_NON_SERIALIZED_APPS = []

############
# FIXTURES #
############

# The list of directories to search for fixtures
FIXTURE_DIRS = []

###############
# STATICFILES #
###############

# A list of locations of additional static files
STATICFILES_DIRS = []

# The default file storage backend used during the build process
STATICFILES_STORAGE = 'django.contrib.staticfiles.storage.StaticFilesStorage'

# List of finder classes that know how to find static files in
# various locations.
STATICFILES_FINDERS = [
    'django.contrib.staticfiles.finders.FileSystemFinder',
    'django.contrib.staticfiles.finders.AppDirectoriesFinder',
    # 'django.contrib.staticfiles.finders.DefaultStorageFinder',
]

##############
# MIGRATIONS #
##############

# Migration module overrides for apps, by app label.
MIGRATION_MODULES = {}

#################
# SYSTEM CHECKS #
#################

# List of all issues generated by system checks that should be silenced. Light
# issues like warnings, infos or debugs will not generate a message. Silencing
# serious issues like errors and criticals does not result in hiding the
# message, but Django will not stop you from e.g. running server.
SILENCED_SYSTEM_CHECKS = []

#######################
# SECURITY MIDDLEWARE #
#######################
SECURE_BROWSER_XSS_FILTER = False
SECURE_CONTENT_TYPE_NOSNIFF = False
SECURE_HSTS_INCLUDE_SUBDOMAINS = False
SECURE_HSTS_PRELOAD = False
SECURE_HSTS_SECONDS = 0
SECURE_REDIRECT_EXEMPT = []
SECURE_SSL_HOST = None
SECURE_SSL_REDIRECT = False
```
### 32 - django/core/mail/__init__.py:

Start line: 1, End line: 35

```python
"""
Tools for sending email.
"""
from django.conf import settings
# Imported for backwards compatibility and for the sake
# of a cleaner namespace. These symbols used to be in
# django/core/mail.py before the introduction of email
# backends and the subsequent reorganization (See #10355)
from django.core.mail.message import (
    DEFAULT_ATTACHMENT_MIME_TYPE, BadHeaderError, EmailMessage,
    EmailMultiAlternatives, SafeMIMEMultipart, SafeMIMEText,
    forbid_multi_line_headers, make_msgid,
)
from django.core.mail.utils import DNS_NAME, CachedDnsName
from django.utils.module_loading import import_string

__all__ = [
    'CachedDnsName', 'DNS_NAME', 'EmailMessage', 'EmailMultiAlternatives',
    'SafeMIMEText', 'SafeMIMEMultipart', 'DEFAULT_ATTACHMENT_MIME_TYPE',
    'make_msgid', 'BadHeaderError', 'forbid_multi_line_headers',
    'get_connection', 'send_mail', 'send_mass_mail', 'mail_admins',
    'mail_managers',
]


def get_connection(backend=None, fail_silently=False, **kwds):
    """Load an email backend and return an instance of it.

    If backend is None (default), use settings.EMAIL_BACKEND.

    Both fail_silently and other keyword arguments are used in the
    constructor of the backend.
    """
    klass = import_string(backend or settings.EMAIL_BACKEND)
    return klass(fail_silently=fail_silently, **kwds)
```
