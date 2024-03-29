# django__django-11790

| **django/django** | `b1d6b35e146aea83b171c1b921178bbaae2795ed` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 1371 |
| **Any found context length** | 1371 |
| **Avg pos** | 9.0 |
| **Min pos** | 9 |
| **Max pos** | 9 |
| **Top file pos** | 3 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/contrib/auth/forms.py b/django/contrib/auth/forms.py
--- a/django/contrib/auth/forms.py
+++ b/django/contrib/auth/forms.py
@@ -191,7 +191,9 @@ def __init__(self, request=None, *args, **kwargs):
 
         # Set the max length and label for the "username" field.
         self.username_field = UserModel._meta.get_field(UserModel.USERNAME_FIELD)
-        self.fields['username'].max_length = self.username_field.max_length or 254
+        username_max_length = self.username_field.max_length or 254
+        self.fields['username'].max_length = username_max_length
+        self.fields['username'].widget.attrs['maxlength'] = username_max_length
         if self.fields['username'].label is None:
             self.fields['username'].label = capfirst(self.username_field.verbose_name)
 

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/contrib/auth/forms.py | 194 | 194 | 9 | 3 | 1371


## Problem Statement

```
AuthenticationForm's username field doesn't set maxlength HTML attribute.
Description
	
AuthenticationForm's username field doesn't render with maxlength HTML attribute anymore.
Regression introduced in #27515 and 5ceaf14686ce626404afb6a5fbd3d8286410bf13.
​https://groups.google.com/forum/?utm_source=digest&utm_medium=email#!topic/django-developers/qnfSqro0DlA
​https://forum.djangoproject.com/t/possible-authenticationform-max-length-regression-in-django-2-1/241

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 django/contrib/auth/migrations/0008_alter_user_username_max_length.py | 1 | 25| 138 | 138 | 138 | 
| 2 | 2 django/contrib/auth/migrations/0004_alter_user_username_opts.py | 1 | 24| 150 | 288 | 288 | 
| 3 | **3 django/contrib/auth/forms.py** | 44 | 69| 162 | 450 | 3334 | 
| 4 | 4 django/contrib/auth/migrations/0009_alter_user_last_name_max_length.py | 1 | 17| 0 | 450 | 3412 | 
| 5 | 5 django/contrib/auth/password_validation.py | 91 | 115| 189 | 639 | 4896 | 
| 6 | **5 django/contrib/auth/forms.py** | 132 | 160| 235 | 874 | 4896 | 
| 7 | 6 django/contrib/auth/migrations/0003_alter_user_email_max_length.py | 1 | 17| 0 | 874 | 4974 | 
| 8 | **6 django/contrib/auth/forms.py** | 1 | 20| 137 | 1011 | 4974 | 
| **-> 9 <-** | **6 django/contrib/auth/forms.py** | 163 | 209| 360 | 1371 | 4974 | 
| 10 | 7 django/forms/fields.py | 208 | 239| 274 | 1645 | 13918 | 
| 11 | 8 django/contrib/auth/validators.py | 1 | 26| 165 | 1810 | 14084 | 
| 12 | 9 django/db/models/fields/__init__.py | 1577 | 1598| 183 | 1993 | 31541 | 
| 13 | 10 django/contrib/admin/forms.py | 1 | 31| 184 | 2177 | 31725 | 
| 14 | 10 django/db/models/fields/__init__.py | 972 | 1004| 208 | 2385 | 31725 | 
| 15 | 11 django/contrib/auth/migrations/0002_alter_permission_name_max_length.py | 1 | 17| 0 | 2385 | 31793 | 
| 16 | **11 django/contrib/auth/forms.py** | 211 | 236| 176 | 2561 | 31793 | 
| 17 | 12 django/contrib/auth/migrations/0007_alter_validators_add_error_messages.py | 1 | 25| 137 | 2698 | 31930 | 
| 18 | 13 django/contrib/auth/migrations/0010_alter_group_name_max_length.py | 1 | 17| 0 | 2698 | 32007 | 
| 19 | **13 django/contrib/auth/forms.py** | 72 | 129| 415 | 3113 | 32007 | 
| 20 | **13 django/contrib/auth/forms.py** | 310 | 351| 290 | 3403 | 32007 | 
| 21 | 14 django/contrib/auth/base_user.py | 47 | 140| 585 | 3988 | 32891 | 
| 22 | **14 django/contrib/auth/forms.py** | 354 | 381| 190 | 4178 | 32891 | 
| 23 | **14 django/contrib/auth/forms.py** | 384 | 436| 359 | 4537 | 32891 | 
| 24 | 14 django/contrib/auth/password_validation.py | 135 | 157| 197 | 4734 | 32891 | 
| 25 | 15 django/contrib/auth/admin.py | 128 | 189| 465 | 5199 | 34617 | 
| 26 | 16 django/contrib/auth/views.py | 330 | 362| 239 | 5438 | 37281 | 
| 27 | 16 django/contrib/auth/password_validation.py | 118 | 133| 154 | 5592 | 37281 | 
| 28 | **16 django/contrib/auth/forms.py** | 23 | 41| 161 | 5753 | 37281 | 
| 29 | **16 django/contrib/auth/forms.py** | 239 | 261| 196 | 5949 | 37281 | 
| 30 | 16 django/forms/fields.py | 469 | 494| 174 | 6123 | 37281 | 
| 31 | 16 django/contrib/auth/admin.py | 40 | 99| 504 | 6627 | 37281 | 
| 32 | 16 django/contrib/auth/views.py | 208 | 222| 133 | 6760 | 37281 | 
| 33 | 16 django/db/models/fields/__init__.py | 2181 | 2201| 163 | 6923 | 37281 | 
| 34 | 16 django/contrib/auth/views.py | 247 | 284| 348 | 7271 | 37281 | 
| 35 | 16 django/forms/fields.py | 529 | 545| 180 | 7451 | 37281 | 
| 36 | 16 django/forms/fields.py | 1173 | 1203| 182 | 7633 | 37281 | 
| 37 | 17 django/forms/forms.py | 136 | 153| 142 | 7775 | 41354 | 
| 38 | 17 django/contrib/auth/views.py | 286 | 327| 314 | 8089 | 41354 | 
| 39 | 17 django/forms/forms.py | 431 | 453| 195 | 8284 | 41354 | 
| 40 | 17 django/contrib/auth/password_validation.py | 1 | 32| 206 | 8490 | 41354 | 
| 41 | 17 django/forms/fields.py | 242 | 259| 162 | 8652 | 41354 | 
| 42 | 18 django/contrib/auth/middleware.py | 85 | 110| 192 | 8844 | 42370 | 
| 43 | 19 django/forms/models.py | 382 | 410| 240 | 9084 | 53865 | 
| 44 | 19 django/contrib/auth/views.py | 40 | 63| 197 | 9281 | 53865 | 
| 45 | 20 django/forms/formsets.py | 1 | 25| 203 | 9484 | 57751 | 
| 46 | 20 django/forms/models.py | 350 | 380| 233 | 9717 | 57751 | 
| 47 | 21 django/contrib/postgres/validators.py | 1 | 21| 181 | 9898 | 58302 | 
| 48 | 22 django/contrib/admin/helpers.py | 366 | 382| 138 | 10036 | 61495 | 
| 49 | 23 django/contrib/auth/management/commands/createsuperuser.py | 230 | 245| 139 | 10175 | 63555 | 
| 50 | 24 django/forms/widgets.py | 298 | 334| 204 | 10379 | 71621 | 
| 51 | 24 django/contrib/auth/views.py | 224 | 244| 163 | 10542 | 71621 | 
| 52 | 25 django/contrib/postgres/forms/array.py | 1 | 37| 258 | 10800 | 73106 | 
| 53 | 26 django/forms/utils.py | 139 | 146| 132 | 10932 | 74342 | 
| 54 | 26 django/forms/models.py | 279 | 307| 288 | 11220 | 74342 | 
| 55 | **26 django/contrib/auth/forms.py** | 276 | 307| 264 | 11484 | 74342 | 
| 56 | 26 django/forms/forms.py | 282 | 290| 113 | 11597 | 74342 | 
| 57 | **26 django/contrib/auth/forms.py** | 263 | 274| 116 | 11713 | 74342 | 
| 58 | 26 django/db/models/fields/__init__.py | 1006 | 1032| 229 | 11942 | 74342 | 
| 59 | 27 django/contrib/auth/management/commands/changepassword.py | 34 | 76| 337 | 12279 | 74881 | 
| 60 | 27 django/forms/forms.py | 380 | 400| 210 | 12489 | 74881 | 
| 61 | 28 django/contrib/postgres/forms/jsonb.py | 1 | 63| 345 | 12834 | 75226 | 
| 62 | 29 django/db/backends/mysql/validation.py | 29 | 61| 246 | 13080 | 75714 | 
| 63 | 30 django/contrib/auth/backends.py | 183 | 234| 369 | 13449 | 77476 | 
| 64 | 30 django/contrib/auth/admin.py | 1 | 22| 188 | 13637 | 77476 | 
| 65 | 30 django/contrib/auth/middleware.py | 47 | 83| 360 | 13997 | 77476 | 
| 66 | 30 django/contrib/auth/admin.py | 101 | 126| 286 | 14283 | 77476 | 
| 67 | 31 django/contrib/admin/widgets.py | 316 | 328| 114 | 14397 | 81342 | 
| 68 | 32 django/contrib/admin/options.py | 1517 | 1596| 719 | 15116 | 99708 | 
| 69 | 32 django/contrib/admin/helpers.py | 1 | 30| 198 | 15314 | 99708 | 
| 70 | 33 django/contrib/auth/migrations/0005_alter_user_last_login_null.py | 1 | 17| 0 | 15314 | 99783 | 
| 71 | 34 django/views/csrf.py | 15 | 100| 835 | 16149 | 101327 | 
| 72 | 34 django/contrib/admin/helpers.py | 33 | 67| 230 | 16379 | 101327 | 
| 73 | 34 django/db/models/fields/__init__.py | 1983 | 2013| 252 | 16631 | 101327 | 
| 74 | 35 django/forms/boundfield.py | 1 | 33| 239 | 16870 | 103448 | 
| 75 | 36 django/core/validators.py | 190 | 209| 152 | 17022 | 107765 | 
| 76 | 36 django/contrib/admin/options.py | 2054 | 2106| 451 | 17473 | 107765 | 
| 77 | 37 django/contrib/auth/models.py | 384 | 465| 484 | 17957 | 110956 | 
| 78 | 37 django/forms/forms.py | 195 | 270| 634 | 18591 | 110956 | 
| 79 | 38 django/contrib/auth/hashers.py | 418 | 447| 290 | 18881 | 115743 | 
| 80 | 38 django/core/validators.py | 153 | 188| 388 | 19269 | 115743 | 
| 81 | 38 django/forms/fields.py | 547 | 566| 171 | 19440 | 115743 | 
| 82 | 38 django/forms/forms.py | 272 | 280| 132 | 19572 | 115743 | 
| 83 | 38 django/forms/models.py | 309 | 348| 387 | 19959 | 115743 | 
| 84 | 38 django/contrib/auth/models.py | 316 | 381| 463 | 20422 | 115743 | 
| 85 | 38 django/contrib/auth/hashers.py | 343 | 358| 146 | 20568 | 115743 | 
| 86 | 38 django/core/validators.py | 341 | 386| 308 | 20876 | 115743 | 
| 87 | 38 django/forms/fields.py | 438 | 466| 199 | 21075 | 115743 | 
| 88 | 38 django/contrib/admin/helpers.py | 123 | 149| 220 | 21295 | 115743 | 
| 89 | 38 django/forms/fields.py | 416 | 435| 131 | 21426 | 115743 | 
| 90 | 38 django/contrib/admin/helpers.py | 355 | 364| 134 | 21560 | 115743 | 
| 91 | 38 django/contrib/admin/options.py | 1111 | 1156| 482 | 22042 | 115743 | 
| 92 | 38 django/forms/fields.py | 323 | 349| 227 | 22269 | 115743 | 
| 93 | 38 django/contrib/auth/middleware.py | 1 | 24| 193 | 22462 | 115743 | 
| 94 | 38 django/contrib/auth/models.py | 128 | 158| 275 | 22737 | 115743 | 
| 95 | 38 django/core/validators.py | 234 | 278| 330 | 23067 | 115743 | 
| 96 | 38 django/forms/formsets.py | 28 | 42| 195 | 23262 | 115743 | 
| 97 | 39 django/contrib/admin/sites.py | 314 | 329| 158 | 23420 | 119934 | 
| 98 | 39 django/forms/widgets.py | 765 | 788| 204 | 23624 | 119934 | 
| 99 | 39 django/forms/fields.py | 46 | 127| 773 | 24397 | 119934 | 
| 100 | 39 django/contrib/auth/hashers.py | 1 | 27| 187 | 24584 | 119934 | 
| 101 | 39 django/forms/fields.py | 392 | 413| 144 | 24728 | 119934 | 
| 102 | 39 django/contrib/auth/views.py | 1 | 37| 278 | 25006 | 119934 | 
| 103 | 39 django/contrib/admin/options.py | 1 | 96| 769 | 25775 | 119934 | 
| 104 | 39 django/forms/fields.py | 351 | 368| 160 | 25935 | 119934 | 
| 105 | 40 django/conf/locale/hr/formats.py | 22 | 48| 620 | 26555 | 120817 | 
| 106 | 40 django/contrib/auth/admin.py | 25 | 37| 128 | 26683 | 120817 | 
| 107 | 40 django/forms/models.py | 679 | 750| 732 | 27415 | 120817 | 
| 108 | 40 django/contrib/admin/widgets.py | 166 | 197| 243 | 27658 | 120817 | 
| 109 | 41 django/conf/locale/en_AU/formats.py | 5 | 40| 708 | 28366 | 121570 | 
| 110 | 42 django/conf/locale/en_GB/formats.py | 5 | 40| 708 | 29074 | 122323 | 
| 111 | 42 django/contrib/auth/hashers.py | 406 | 416| 127 | 29201 | 122323 | 
| 112 | 42 django/contrib/auth/password_validation.py | 54 | 88| 277 | 29478 | 122323 | 
| 113 | 43 django/contrib/auth/tokens.py | 54 | 63| 134 | 29612 | 123107 | 
| 114 | 43 django/contrib/auth/models.py | 1 | 30| 200 | 29812 | 123107 | 
| 115 | 44 django/db/models/base.py | 1747 | 1818| 565 | 30377 | 138297 | 
| 116 | 44 django/db/models/fields/__init__.py | 1431 | 1453| 119 | 30496 | 138297 | 
| 117 | 44 django/contrib/admin/options.py | 1597 | 1621| 279 | 30775 | 138297 | 
| 118 | 44 django/forms/forms.py | 56 | 112| 525 | 31300 | 138297 | 
| 119 | 44 django/contrib/auth/backends.py | 31 | 49| 142 | 31442 | 138297 | 
| 120 | 45 django/contrib/auth/handlers/modwsgi.py | 1 | 44| 247 | 31689 | 138544 | 
| 121 | 46 django/conf/locale/id/formats.py | 5 | 50| 708 | 32397 | 139297 | 
| 122 | 47 django/conf/locale/da/formats.py | 5 | 27| 250 | 32647 | 139592 | 
| 123 | 47 django/forms/models.py | 752 | 773| 194 | 32841 | 139592 | 
| 124 | 47 django/forms/models.py | 207 | 276| 623 | 33464 | 139592 | 
| 125 | 48 django/conf/locale/bn/formats.py | 5 | 33| 294 | 33758 | 139930 | 
| 126 | 49 django/contrib/postgres/forms/ranges.py | 81 | 103| 149 | 33907 | 140607 | 
| 127 | 49 django/forms/fields.py | 848 | 887| 298 | 34205 | 140607 | 
| 128 | 49 django/contrib/admin/helpers.py | 92 | 120| 249 | 34454 | 140607 | 
| 129 | 49 django/db/models/fields/__init__.py | 244 | 309| 467 | 34921 | 140607 | 
| 130 | 50 django/conf/locale/ka/formats.py | 23 | 48| 564 | 35485 | 141514 | 
| 131 | 50 django/forms/fields.py | 1158 | 1170| 113 | 35598 | 141514 | 
| 132 | 51 django/conf/locale/ml/formats.py | 5 | 41| 663 | 36261 | 142222 | 
| 133 | 52 django/conf/locale/mk/formats.py | 5 | 43| 672 | 36933 | 142939 | 
| 134 | 53 django/conf/locale/sl/formats.py | 22 | 48| 596 | 37529 | 143788 | 
| 135 | 53 django/db/models/fields/__init__.py | 1705 | 1728| 146 | 37675 | 143788 | 
| 136 | 53 django/forms/fields.py | 497 | 526| 207 | 37882 | 143788 | 
| 137 | 54 django/conf/locale/ru/formats.py | 5 | 33| 402 | 38284 | 144235 | 
| 138 | 55 django/conf/locale/mn/formats.py | 5 | 22| 120 | 38404 | 144399 | 
| 139 | 56 django/conf/locale/ar/formats.py | 5 | 22| 135 | 38539 | 144578 | 
| 140 | 56 django/forms/models.py | 594 | 626| 270 | 38809 | 144578 | 
| 141 | 56 django/contrib/admin/options.py | 1996 | 2017| 221 | 39030 | 144578 | 
| 142 | 56 django/forms/forms.py | 402 | 429| 190 | 39220 | 144578 | 
| 143 | 57 django/conf/locale/sr_Latn/formats.py | 23 | 44| 511 | 39731 | 145427 | 
| 144 | 57 django/db/models/fields/__init__.py | 2268 | 2318| 339 | 40070 | 145427 | 
| 145 | 58 django/conf/locale/uk/formats.py | 5 | 38| 460 | 40530 | 145932 | 
| 146 | 58 django/contrib/auth/password_validation.py | 160 | 204| 351 | 40881 | 145932 | 
| 147 | 58 django/contrib/auth/views.py | 65 | 104| 295 | 41176 | 145932 | 
| 148 | 58 django/contrib/postgres/forms/ranges.py | 31 | 78| 325 | 41501 | 145932 | 
| 149 | 59 django/conf/locale/az/formats.py | 5 | 33| 399 | 41900 | 146376 | 
| 150 | 59 django/contrib/auth/hashers.py | 328 | 341| 120 | 42020 | 146376 | 
| 151 | 59 django/contrib/auth/middleware.py | 27 | 45| 178 | 42198 | 146376 | 
| 152 | 59 django/forms/models.py | 194 | 204| 131 | 42329 | 146376 | 
| 153 | 60 django/conf/locale/lt/formats.py | 5 | 46| 711 | 43040 | 147132 | 
| 154 | 60 django/contrib/auth/hashers.py | 64 | 77| 147 | 43187 | 147132 | 
| 155 | 61 django/conf/locale/nb/formats.py | 5 | 40| 646 | 43833 | 147823 | 
| 156 | 62 django/conf/locale/nn/formats.py | 5 | 41| 664 | 44497 | 148532 | 
| 157 | 62 django/conf/locale/sl/formats.py | 5 | 20| 208 | 44705 | 148532 | 
| 158 | 63 django/contrib/flatpages/forms.py | 1 | 50| 346 | 45051 | 149015 | 
| 159 | 64 django/conf/locale/ko/formats.py | 32 | 53| 438 | 45489 | 149958 | 
| 160 | 64 django/core/validators.py | 111 | 150| 398 | 45887 | 149958 | 
| 161 | 65 django/conf/locale/bs/formats.py | 5 | 22| 139 | 46026 | 150141 | 
| 162 | 65 django/forms/widgets.py | 464 | 504| 275 | 46301 | 150141 | 
| 163 | 66 django/contrib/auth/checks.py | 97 | 167| 525 | 46826 | 151314 | 
| 164 | 67 django/conf/locale/en/formats.py | 5 | 41| 663 | 47489 | 152022 | 
| 165 | 67 django/contrib/auth/hashers.py | 215 | 226| 150 | 47639 | 152022 | 
| 166 | 67 django/forms/forms.py | 292 | 308| 165 | 47804 | 152022 | 
| 167 | 68 django/conf/locale/sr/formats.py | 23 | 44| 511 | 48315 | 152871 | 
| 168 | 68 django/forms/forms.py | 357 | 378| 172 | 48487 | 152871 | 
| 169 | 68 django/contrib/auth/base_user.py | 1 | 44| 298 | 48785 | 152871 | 
| 170 | 68 django/db/models/fields/__init__.py | 1455 | 1514| 398 | 49183 | 152871 | 
| 171 | 69 django/conf/locale/pt_BR/formats.py | 5 | 34| 494 | 49677 | 153410 | 
| 172 | 69 django/contrib/admin/helpers.py | 70 | 89| 167 | 49844 | 153410 | 
| 173 | 69 django/forms/forms.py | 171 | 193| 182 | 50026 | 153410 | 
| 174 | 70 django/contrib/admin/checks.py | 57 | 126| 588 | 50614 | 162426 | 
| 175 | 71 django/conf/locale/pl/formats.py | 5 | 30| 339 | 50953 | 162810 | 
| 176 | 72 django/conf/locale/it/formats.py | 21 | 46| 564 | 51517 | 163707 | 
| 177 | 73 django/conf/locale/ro/formats.py | 5 | 36| 262 | 51779 | 164014 | 
| 178 | 74 django/conf/locale/pt/formats.py | 5 | 39| 630 | 52409 | 164689 | 
| 179 | 74 django/forms/utils.py | 1 | 41| 291 | 52700 | 164689 | 
| 180 | 75 django/conf/locale/zh_Hant/formats.py | 5 | 43| 635 | 53335 | 165369 | 
| 181 | 75 django/contrib/postgres/forms/array.py | 62 | 102| 248 | 53583 | 165369 | 
| 182 | 76 django/contrib/auth/mixins.py | 47 | 85| 277 | 53860 | 166101 | 
| 183 | 76 django/forms/models.py | 412 | 442| 243 | 54103 | 166101 | 
| 184 | 77 django/conf/locale/bg/formats.py | 5 | 22| 131 | 54234 | 166276 | 
| 185 | 77 django/forms/widgets.py | 374 | 391| 116 | 54350 | 166276 | 
| 186 | 77 django/conf/locale/hr/formats.py | 5 | 21| 218 | 54568 | 166276 | 
| 187 | 78 django/conf/locale/gl/formats.py | 5 | 22| 170 | 54738 | 166490 | 
| 188 | 78 django/core/validators.py | 389 | 416| 224 | 54962 | 166490 | 
| 189 | 79 django/conf/locale/de/formats.py | 5 | 29| 323 | 55285 | 166858 | 
| 190 | 79 django/db/models/fields/__init__.py | 367 | 393| 199 | 55484 | 166858 | 
| 191 | 80 django/forms/__init__.py | 1 | 12| 0 | 55484 | 166948 | 
| 192 | 80 django/contrib/auth/management/commands/changepassword.py | 1 | 32| 206 | 55690 | 166948 | 
| 193 | 80 django/contrib/admin/checks.py | 371 | 397| 281 | 55971 | 166948 | 
| 194 | 80 django/contrib/auth/hashers.py | 564 | 597| 242 | 56213 | 166948 | 
| 195 | 80 django/core/validators.py | 74 | 109| 554 | 56767 | 166948 | 
| 196 | 81 django/conf/locale/ga/formats.py | 5 | 22| 124 | 56891 | 167116 | 
| 197 | 82 django/conf/locale/cy/formats.py | 5 | 36| 582 | 57473 | 167743 | 
| 198 | 82 django/db/models/fields/__init__.py | 1035 | 1048| 104 | 57577 | 167743 | 
| 199 | 83 django/conf/locale/fi/formats.py | 5 | 40| 470 | 58047 | 168258 | 
| 200 | 84 django/conf/locale/te/formats.py | 5 | 22| 123 | 58170 | 168425 | 
| 201 | 84 django/forms/widgets.py | 671 | 702| 261 | 58431 | 168425 | 
| 202 | 85 django/conf/locale/sk/formats.py | 5 | 30| 348 | 58779 | 168818 | 
| 203 | 85 django/forms/models.py | 918 | 950| 353 | 59132 | 168818 | 
| 204 | 85 django/db/models/fields/__init__.py | 2016 | 2046| 199 | 59331 | 168818 | 
| 205 | 85 django/forms/fields.py | 288 | 320| 228 | 59559 | 168818 | 
| 206 | 85 django/contrib/auth/backends.py | 163 | 181| 146 | 59705 | 168818 | 
| 207 | 86 django/conf/locale/is/formats.py | 5 | 22| 130 | 59835 | 168993 | 
| 208 | 87 django/conf/locale/zh_Hans/formats.py | 5 | 43| 635 | 60470 | 169673 | 
| 209 | 87 django/forms/models.py | 952 | 985| 367 | 60837 | 169673 | 


### Hint

```
Regression test.
```

## Patch

```diff
diff --git a/django/contrib/auth/forms.py b/django/contrib/auth/forms.py
--- a/django/contrib/auth/forms.py
+++ b/django/contrib/auth/forms.py
@@ -191,7 +191,9 @@ def __init__(self, request=None, *args, **kwargs):
 
         # Set the max length and label for the "username" field.
         self.username_field = UserModel._meta.get_field(UserModel.USERNAME_FIELD)
-        self.fields['username'].max_length = self.username_field.max_length or 254
+        username_max_length = self.username_field.max_length or 254
+        self.fields['username'].max_length = username_max_length
+        self.fields['username'].widget.attrs['maxlength'] = username_max_length
         if self.fields['username'].label is None:
             self.fields['username'].label = capfirst(self.username_field.verbose_name)
 

```

## Test Patch

```diff
diff --git a/tests/auth_tests/test_forms.py b/tests/auth_tests/test_forms.py
--- a/tests/auth_tests/test_forms.py
+++ b/tests/auth_tests/test_forms.py
@@ -423,6 +423,7 @@ def test_username_field_max_length_matches_user_model(self):
         CustomEmailField.objects.create_user(**data)
         form = AuthenticationForm(None, data)
         self.assertEqual(form.fields['username'].max_length, 255)
+        self.assertEqual(form.fields['username'].widget.attrs.get('maxlength'), 255)
         self.assertEqual(form.errors, {})
 
     @override_settings(AUTH_USER_MODEL='auth_tests.IntegerUsernameUser')
@@ -435,6 +436,7 @@ def test_username_field_max_length_defaults_to_254(self):
         IntegerUsernameUser.objects.create_user(**data)
         form = AuthenticationForm(None, data)
         self.assertEqual(form.fields['username'].max_length, 254)
+        self.assertEqual(form.fields['username'].widget.attrs.get('maxlength'), 254)
         self.assertEqual(form.errors, {})
 
     def test_username_field_label(self):

```


## Code snippets

### 1 - django/contrib/auth/migrations/0008_alter_user_username_max_length.py:

Start line: 1, End line: 25

```python
from django.contrib.auth import validators
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('auth', '0007_alter_validators_add_error_messages'),
    ]

    operations = [
        migrations.AlterField(
            model_name='user',
            name='username',
            field=models.CharField(
                error_messages={'unique': 'A user with that username already exists.'},
                help_text='Required. 150 characters or fewer. Letters, digits and @/./+/-/_ only.',
                max_length=150,
                unique=True,
                validators=[validators.UnicodeUsernameValidator()],
                verbose_name='username',
            ),
        ),
    ]
```
### 2 - django/contrib/auth/migrations/0004_alter_user_username_opts.py:

Start line: 1, End line: 24

```python
from django.contrib.auth import validators
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('auth', '0003_alter_user_email_max_length'),
    ]

    # No database changes; modifies validators and error_messages (#13147).
    operations = [
        migrations.AlterField(
            model_name='user',
            name='username',
            field=models.CharField(
                error_messages={'unique': 'A user with that username already exists.'}, max_length=30,
                validators=[validators.UnicodeUsernameValidator()],
                help_text='Required. 30 characters or fewer. Letters, digits and @/./+/-/_ only.',
                unique=True, verbose_name='username'
            ),
        ),
    ]
```
### 3 - django/contrib/auth/forms.py:

Start line: 44, End line: 69

```python
class ReadOnlyPasswordHashField(forms.Field):
    widget = ReadOnlyPasswordHashWidget

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("required", False)
        super().__init__(*args, **kwargs)

    def bound_data(self, data, initial):
        # Always return initial because the widget doesn't
        # render an input field.
        return initial

    def has_changed(self, initial, data):
        return False


class UsernameField(forms.CharField):
    def to_python(self, value):
        return unicodedata.normalize('NFKC', super().to_python(value))

    def widget_attrs(self, widget):
        return {
            **super().widget_attrs(widget),
            'autocapitalize': 'none',
            'autocomplete': 'username',
        }
```
### 4 - django/contrib/auth/migrations/0009_alter_user_last_name_max_length.py:

Start line: 1, End line: 17

```python

```
### 5 - django/contrib/auth/password_validation.py:

Start line: 91, End line: 115

```python
class MinimumLengthValidator:
    """
    Validate whether the password is of a minimum length.
    """
    def __init__(self, min_length=8):
        self.min_length = min_length

    def validate(self, password, user=None):
        if len(password) < self.min_length:
            raise ValidationError(
                ngettext(
                    "This password is too short. It must contain at least %(min_length)d character.",
                    "This password is too short. It must contain at least %(min_length)d characters.",
                    self.min_length
                ),
                code='password_too_short',
                params={'min_length': self.min_length},
            )

    def get_help_text(self):
        return ngettext(
            "Your password must contain at least %(min_length)d character.",
            "Your password must contain at least %(min_length)d characters.",
            self.min_length
        ) % {'min_length': self.min_length}
```
### 6 - django/contrib/auth/forms.py:

Start line: 132, End line: 160

```python
class UserChangeForm(forms.ModelForm):
    password = ReadOnlyPasswordHashField(
        label=_("Password"),
        help_text=_(
            'Raw passwords are not stored, so there is no way to see this '
            'user’s password, but you can change the password using '
            '<a href="{}">this form</a>.'
        ),
    )

    class Meta:
        model = User
        fields = '__all__'
        field_classes = {'username': UsernameField}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        password = self.fields.get('password')
        if password:
            password.help_text = password.help_text.format('../password/')
        user_permissions = self.fields.get('user_permissions')
        if user_permissions:
            user_permissions.queryset = user_permissions.queryset.select_related('content_type')

    def clean_password(self):
        # Regardless of what the user provides, return the initial value.
        # This is done here, rather than on the field, because the
        # field does not have access to the initial value
        return self.initial.get('password')
```
### 7 - django/contrib/auth/migrations/0003_alter_user_email_max_length.py:

Start line: 1, End line: 17

```python

```
### 8 - django/contrib/auth/forms.py:

Start line: 1, End line: 20

```python
import unicodedata

from django import forms
from django.contrib.auth import (
    authenticate, get_user_model, password_validation,
)
from django.contrib.auth.hashers import (
    UNUSABLE_PASSWORD_PREFIX, identify_hasher,
)
from django.contrib.auth.models import User
from django.contrib.auth.tokens import default_token_generator
from django.contrib.sites.shortcuts import get_current_site
from django.core.mail import EmailMultiAlternatives
from django.template import loader
from django.utils.encoding import force_bytes
from django.utils.http import urlsafe_base64_encode
from django.utils.text import capfirst
from django.utils.translation import gettext, gettext_lazy as _

UserModel = get_user_model()
```
### 9 - django/contrib/auth/forms.py:

Start line: 163, End line: 209

```python
class AuthenticationForm(forms.Form):
    """
    Base class for authenticating users. Extend this to get a form that accepts
    username/password logins.
    """
    username = UsernameField(widget=forms.TextInput(attrs={'autofocus': True}))
    password = forms.CharField(
        label=_("Password"),
        strip=False,
        widget=forms.PasswordInput(attrs={'autocomplete': 'current-password'}),
    )

    error_messages = {
        'invalid_login': _(
            "Please enter a correct %(username)s and password. Note that both "
            "fields may be case-sensitive."
        ),
        'inactive': _("This account is inactive."),
    }

    def __init__(self, request=None, *args, **kwargs):
        """
        The 'request' parameter is set for custom auth use by subclasses.
        The form data comes in via the standard 'data' kwarg.
        """
        self.request = request
        self.user_cache = None
        super().__init__(*args, **kwargs)

        # Set the max length and label for the "username" field.
        self.username_field = UserModel._meta.get_field(UserModel.USERNAME_FIELD)
        self.fields['username'].max_length = self.username_field.max_length or 254
        if self.fields['username'].label is None:
            self.fields['username'].label = capfirst(self.username_field.verbose_name)

    def clean(self):
        username = self.cleaned_data.get('username')
        password = self.cleaned_data.get('password')

        if username is not None and password:
            self.user_cache = authenticate(self.request, username=username, password=password)
            if self.user_cache is None:
                raise self.get_invalid_login_error()
            else:
                self.confirm_login_allowed(self.user_cache)

        return self.cleaned_data
```
### 10 - django/forms/fields.py:

Start line: 208, End line: 239

```python
class CharField(Field):
    def __init__(self, *, max_length=None, min_length=None, strip=True, empty_value='', **kwargs):
        self.max_length = max_length
        self.min_length = min_length
        self.strip = strip
        self.empty_value = empty_value
        super().__init__(**kwargs)
        if min_length is not None:
            self.validators.append(validators.MinLengthValidator(int(min_length)))
        if max_length is not None:
            self.validators.append(validators.MaxLengthValidator(int(max_length)))
        self.validators.append(validators.ProhibitNullCharactersValidator())

    def to_python(self, value):
        """Return a string."""
        if value not in self.empty_values:
            value = str(value)
            if self.strip:
                value = value.strip()
        if value in self.empty_values:
            return self.empty_value
        return value

    def widget_attrs(self, widget):
        attrs = super().widget_attrs(widget)
        if self.max_length is not None and not widget.is_hidden:
            # The HTML attribute is maxlength, not max_length.
            attrs['maxlength'] = str(self.max_length)
        if self.min_length is not None and not widget.is_hidden:
            # The HTML attribute is minlength, not min_length.
            attrs['minlength'] = str(self.min_length)
        return attrs
```
### 16 - django/contrib/auth/forms.py:

Start line: 211, End line: 236

```python
class AuthenticationForm(forms.Form):

    def confirm_login_allowed(self, user):
        """
        Controls whether the given User may log in. This is a policy setting,
        independent of end-user authentication. This default behavior is to
        allow login by active users, and reject login by inactive users.

        If the given user cannot log in, this method should raise a
        ``forms.ValidationError``.

        If the given user may log in, this method should return None.
        """
        if not user.is_active:
            raise forms.ValidationError(
                self.error_messages['inactive'],
                code='inactive',
            )

    def get_user(self):
        return self.user_cache

    def get_invalid_login_error(self):
        return forms.ValidationError(
            self.error_messages['invalid_login'],
            code='invalid_login',
            params={'username': self.username_field.verbose_name},
        )
```
### 19 - django/contrib/auth/forms.py:

Start line: 72, End line: 129

```python
class UserCreationForm(forms.ModelForm):
    """
    A form that creates a user, with no privileges, from the given username and
    password.
    """
    error_messages = {
        'password_mismatch': _('The two password fields didn’t match.'),
    }
    password1 = forms.CharField(
        label=_("Password"),
        strip=False,
        widget=forms.PasswordInput(attrs={'autocomplete': 'new-password'}),
        help_text=password_validation.password_validators_help_text_html(),
    )
    password2 = forms.CharField(
        label=_("Password confirmation"),
        widget=forms.PasswordInput(attrs={'autocomplete': 'new-password'}),
        strip=False,
        help_text=_("Enter the same password as before, for verification."),
    )

    class Meta:
        model = User
        fields = ("username",)
        field_classes = {'username': UsernameField}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self._meta.model.USERNAME_FIELD in self.fields:
            self.fields[self._meta.model.USERNAME_FIELD].widget.attrs['autofocus'] = True

    def clean_password2(self):
        password1 = self.cleaned_data.get("password1")
        password2 = self.cleaned_data.get("password2")
        if password1 and password2 and password1 != password2:
            raise forms.ValidationError(
                self.error_messages['password_mismatch'],
                code='password_mismatch',
            )
        return password2

    def _post_clean(self):
        super()._post_clean()
        # Validate the password after self.instance is updated with form data
        # by super().
        password = self.cleaned_data.get('password2')
        if password:
            try:
                password_validation.validate_password(password, self.instance)
            except forms.ValidationError as error:
                self.add_error('password2', error)

    def save(self, commit=True):
        user = super().save(commit=False)
        user.set_password(self.cleaned_data["password1"])
        if commit:
            user.save()
        return user
```
### 20 - django/contrib/auth/forms.py:

Start line: 310, End line: 351

```python
class SetPasswordForm(forms.Form):
    """
    A form that lets a user change set their password without entering the old
    password
    """
    error_messages = {
        'password_mismatch': _('The two password fields didn’t match.'),
    }
    new_password1 = forms.CharField(
        label=_("New password"),
        widget=forms.PasswordInput(attrs={'autocomplete': 'new-password'}),
        strip=False,
        help_text=password_validation.password_validators_help_text_html(),
    )
    new_password2 = forms.CharField(
        label=_("New password confirmation"),
        strip=False,
        widget=forms.PasswordInput(attrs={'autocomplete': 'new-password'}),
    )

    def __init__(self, user, *args, **kwargs):
        self.user = user
        super().__init__(*args, **kwargs)

    def clean_new_password2(self):
        password1 = self.cleaned_data.get('new_password1')
        password2 = self.cleaned_data.get('new_password2')
        if password1 and password2:
            if password1 != password2:
                raise forms.ValidationError(
                    self.error_messages['password_mismatch'],
                    code='password_mismatch',
                )
        password_validation.validate_password(password2, self.user)
        return password2

    def save(self, commit=True):
        password = self.cleaned_data["new_password1"]
        self.user.set_password(password)
        if commit:
            self.user.save()
        return self.user
```
### 22 - django/contrib/auth/forms.py:

Start line: 354, End line: 381

```python
class PasswordChangeForm(SetPasswordForm):
    """
    A form that lets a user change their password by entering their old
    password.
    """
    error_messages = {
        **SetPasswordForm.error_messages,
        'password_incorrect': _("Your old password was entered incorrectly. Please enter it again."),
    }
    old_password = forms.CharField(
        label=_("Old password"),
        strip=False,
        widget=forms.PasswordInput(attrs={'autocomplete': 'current-password', 'autofocus': True}),
    )

    field_order = ['old_password', 'new_password1', 'new_password2']

    def clean_old_password(self):
        """
        Validate that the old_password field is correct.
        """
        old_password = self.cleaned_data["old_password"]
        if not self.user.check_password(old_password):
            raise forms.ValidationError(
                self.error_messages['password_incorrect'],
                code='password_incorrect',
            )
        return old_password
```
### 23 - django/contrib/auth/forms.py:

Start line: 384, End line: 436

```python
class AdminPasswordChangeForm(forms.Form):
    """
    A form used to change the password of a user in the admin interface.
    """
    error_messages = {
        'password_mismatch': _('The two password fields didn’t match.'),
    }
    required_css_class = 'required'
    password1 = forms.CharField(
        label=_("Password"),
        widget=forms.PasswordInput(attrs={'autocomplete': 'new-password', 'autofocus': True}),
        strip=False,
        help_text=password_validation.password_validators_help_text_html(),
    )
    password2 = forms.CharField(
        label=_("Password (again)"),
        widget=forms.PasswordInput(attrs={'autocomplete': 'new-password'}),
        strip=False,
        help_text=_("Enter the same password as before, for verification."),
    )

    def __init__(self, user, *args, **kwargs):
        self.user = user
        super().__init__(*args, **kwargs)

    def clean_password2(self):
        password1 = self.cleaned_data.get('password1')
        password2 = self.cleaned_data.get('password2')
        if password1 and password2:
            if password1 != password2:
                raise forms.ValidationError(
                    self.error_messages['password_mismatch'],
                    code='password_mismatch',
                )
        password_validation.validate_password(password2, self.user)
        return password2

    def save(self, commit=True):
        """Save the new password."""
        password = self.cleaned_data["password1"]
        self.user.set_password(password)
        if commit:
            self.user.save()
        return self.user

    @property
    def changed_data(self):
        data = super().changed_data
        for name in self.fields:
            if name not in data:
                return []
        return ['password']
```
### 28 - django/contrib/auth/forms.py:

Start line: 23, End line: 41

```python
class ReadOnlyPasswordHashWidget(forms.Widget):
    template_name = 'auth/widgets/read_only_password_hash.html'
    read_only = True

    def get_context(self, name, value, attrs):
        context = super().get_context(name, value, attrs)
        summary = []
        if not value or value.startswith(UNUSABLE_PASSWORD_PREFIX):
            summary.append({'label': gettext("No password set.")})
        else:
            try:
                hasher = identify_hasher(value)
            except ValueError:
                summary.append({'label': gettext("Invalid password format or unknown hashing algorithm.")})
            else:
                for key, value_ in hasher.safe_summary(value).items():
                    summary.append({'label': gettext(key), 'value': value_})
        context['summary'] = summary
        return context
```
### 29 - django/contrib/auth/forms.py:

Start line: 239, End line: 261

```python
class PasswordResetForm(forms.Form):
    email = forms.EmailField(
        label=_("Email"),
        max_length=254,
        widget=forms.EmailInput(attrs={'autocomplete': 'email'})
    )

    def send_mail(self, subject_template_name, email_template_name,
                  context, from_email, to_email, html_email_template_name=None):
        """
        Send a django.core.mail.EmailMultiAlternatives to `to_email`.
        """
        subject = loader.render_to_string(subject_template_name, context)
        # Email subject *must not* contain newlines
        subject = ''.join(subject.splitlines())
        body = loader.render_to_string(email_template_name, context)

        email_message = EmailMultiAlternatives(subject, body, from_email, [to_email])
        if html_email_template_name is not None:
            html_email = loader.render_to_string(html_email_template_name, context)
            email_message.attach_alternative(html_email, 'text/html')

        email_message.send()
```
### 55 - django/contrib/auth/forms.py:

Start line: 276, End line: 307

```python
class PasswordResetForm(forms.Form):

    def save(self, domain_override=None,
             subject_template_name='registration/password_reset_subject.txt',
             email_template_name='registration/password_reset_email.html',
             use_https=False, token_generator=default_token_generator,
             from_email=None, request=None, html_email_template_name=None,
             extra_email_context=None):
        """
        Generate a one-use only link for resetting password and send it to the
        user.
        """
        email = self.cleaned_data["email"]
        for user in self.get_users(email):
            if not domain_override:
                current_site = get_current_site(request)
                site_name = current_site.name
                domain = current_site.domain
            else:
                site_name = domain = domain_override
            context = {
                'email': email,
                'domain': domain,
                'site_name': site_name,
                'uid': urlsafe_base64_encode(force_bytes(user.pk)),
                'user': user,
                'token': token_generator.make_token(user),
                'protocol': 'https' if use_https else 'http',
                **(extra_email_context or {}),
            }
            self.send_mail(
                subject_template_name, email_template_name, context, from_email,
                email, html_email_template_name=html_email_template_name,
            )
```
### 57 - django/contrib/auth/forms.py:

Start line: 263, End line: 274

```python
class PasswordResetForm(forms.Form):

    def get_users(self, email):
        """Given an email, return matching user(s) who should receive a reset.

        This allows subclasses to more easily customize the default policies
        that prevent inactive users and users with unusable passwords from
        resetting their password.
        """
        active_users = UserModel._default_manager.filter(**{
            '%s__iexact' % UserModel.get_email_field_name(): email,
            'is_active': True,
        })
        return (u for u in active_users if u.has_usable_password())
```
