# django__django-8630

| **django/django** | `59841170ba1785ada10a2915b0b60efdb046ee39` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 1116 |
| **Any found context length** | 919 |
| **Avg pos** | 13.0 |
| **Min pos** | 4 |
| **Max pos** | 5 |
| **Top file pos** | 1 |
| **Missing snippets** | 3 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/contrib/auth/views.py b/django/contrib/auth/views.py
--- a/django/contrib/auth/views.py
+++ b/django/contrib/auth/views.py
@@ -43,6 +43,7 @@ class LoginView(SuccessURLAllowedHostsMixin, FormView):
     """
     form_class = AuthenticationForm
     authentication_form = None
+    next_page = None
     redirect_field_name = REDIRECT_FIELD_NAME
     template_name = 'registration/login.html'
     redirect_authenticated_user = False
@@ -63,8 +64,7 @@ def dispatch(self, request, *args, **kwargs):
         return super().dispatch(request, *args, **kwargs)
 
     def get_success_url(self):
-        url = self.get_redirect_url()
-        return url or resolve_url(settings.LOGIN_REDIRECT_URL)
+        return self.get_redirect_url() or self.get_default_redirect_url()
 
     def get_redirect_url(self):
         """Return the user-originating redirect URL if it's safe."""
@@ -79,6 +79,10 @@ def get_redirect_url(self):
         )
         return redirect_to if url_is_safe else ''
 
+    def get_default_redirect_url(self):
+        """Return the default redirect URL."""
+        return resolve_url(self.next_page or settings.LOGIN_REDIRECT_URL)
+
     def get_form_class(self):
         return self.authentication_form or self.form_class
 

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/contrib/auth/views.py | 46 | 46 | 5 | 1 | 1116
| django/contrib/auth/views.py | 66 | 67 | 4 | 1 | 919
| django/contrib/auth/views.py | 82 | 82 | 4 | 1 | 919


## Problem Statement

```
Add next_page to LoginView
Description
	
LogoutView has a next_page attribute used to override settings.LOGOUT_REDIRECT_URL.
It would be nice if LoginView had the same mechanism.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 django/contrib/auth/views.py** | 129 | 163| 269 | 269 | 2664 | 
| 2 | **1 django/contrib/auth/views.py** | 166 | 186| 182 | 451 | 2664 | 
| 3 | **1 django/contrib/auth/views.py** | 107 | 127| 173 | 624 | 2664 | 
| **-> 4 <-** | **1 django/contrib/auth/views.py** | 65 | 104| 295 | 919 | 2664 | 
| **-> 5 <-** | **1 django/contrib/auth/views.py** | 40 | 63| 197 | 1116 | 2664 | 
| 6 | **1 django/contrib/auth/views.py** | 330 | 362| 239 | 1355 | 2664 | 
| 7 | 2 django/views/generic/base.py | 188 | 219| 247 | 1602 | 4265 | 
| 8 | 3 django/contrib/auth/decorators.py | 38 | 74| 273 | 1875 | 4852 | 
| 9 | 4 django/contrib/admin/views/decorators.py | 1 | 19| 135 | 2010 | 4988 | 
| 10 | **4 django/contrib/auth/views.py** | 1 | 37| 278 | 2288 | 4988 | 
| 11 | **4 django/contrib/auth/views.py** | 286 | 327| 314 | 2602 | 4988 | 
| 12 | **4 django/contrib/auth/views.py** | 247 | 284| 348 | 2950 | 4988 | 
| 13 | 5 django/contrib/admin/sites.py | 220 | 239| 221 | 3171 | 9358 | 
| 14 | **5 django/contrib/auth/views.py** | 224 | 244| 163 | 3334 | 9358 | 
| 15 | 6 django/contrib/flatpages/views.py | 48 | 70| 191 | 3525 | 9948 | 
| 16 | 7 django/contrib/auth/urls.py | 1 | 21| 224 | 3749 | 10172 | 
| 17 | 7 django/contrib/auth/decorators.py | 1 | 35| 313 | 4062 | 10172 | 
| 18 | 7 django/contrib/admin/sites.py | 360 | 380| 157 | 4219 | 10172 | 
| 19 | 7 django/contrib/flatpages/views.py | 1 | 45| 399 | 4618 | 10172 | 
| 20 | 7 django/contrib/admin/sites.py | 382 | 414| 288 | 4906 | 10172 | 
| 21 | **7 django/contrib/auth/views.py** | 208 | 222| 133 | 5039 | 10172 | 
| 22 | 8 django/contrib/auth/mixins.py | 1 | 42| 267 | 5306 | 11036 | 
| 23 | 9 django/views/i18n.py | 1 | 64| 490 | 5796 | 13497 | 
| 24 | 10 django/contrib/admin/options.py | 1909 | 1947| 330 | 6126 | 32036 | 
| 25 | 11 django/contrib/admindocs/views.py | 138 | 156| 187 | 6313 | 35340 | 
| 26 | 11 django/contrib/admin/options.py | 1326 | 1351| 232 | 6545 | 35340 | 
| 27 | 11 django/contrib/admin/options.py | 1683 | 1754| 653 | 7198 | 35340 | 
| 28 | 12 django/contrib/admindocs/middleware.py | 1 | 29| 235 | 7433 | 35576 | 
| 29 | 13 django/contrib/flatpages/middleware.py | 1 | 21| 147 | 7580 | 35724 | 
| 30 | 14 django/views/decorators/clickjacking.py | 22 | 54| 238 | 7818 | 36100 | 
| 31 | 15 django/contrib/flatpages/templatetags/flatpages.py | 1 | 42| 300 | 8118 | 36874 | 
| 32 | 16 django/contrib/auth/middleware.py | 112 | 123| 107 | 8225 | 37868 | 
| 33 | 17 django/contrib/auth/admin.py | 191 | 206| 185 | 8410 | 39594 | 
| 34 | **17 django/contrib/auth/views.py** | 189 | 205| 122 | 8532 | 39594 | 
| 35 | 18 django/views/decorators/common.py | 1 | 15| 110 | 8642 | 39704 | 
| 36 | 18 django/contrib/admin/options.py | 611 | 632| 258 | 8900 | 39704 | 
| 37 | 19 django/urls/base.py | 27 | 86| 438 | 9338 | 40887 | 
| 38 | 19 django/contrib/admin/sites.py | 338 | 358| 182 | 9520 | 40887 | 
| 39 | 19 django/contrib/admindocs/views.py | 32 | 52| 159 | 9679 | 40887 | 
| 40 | 20 django/contrib/redirects/middleware.py | 1 | 51| 354 | 10033 | 41242 | 
| 41 | 20 django/contrib/auth/admin.py | 101 | 126| 286 | 10319 | 41242 | 
| 42 | 20 django/contrib/admin/options.py | 1627 | 1652| 291 | 10610 | 41242 | 
| 43 | 20 django/contrib/admin/sites.py | 416 | 431| 129 | 10739 | 41242 | 
| 44 | 21 django/contrib/admin/templatetags/admin_list.py | 46 | 70| 191 | 10930 | 44902 | 
| 45 | 21 django/contrib/admindocs/views.py | 158 | 182| 234 | 11164 | 44902 | 
| 46 | 21 django/views/generic/base.py | 154 | 186| 228 | 11392 | 44902 | 
| 47 | 22 django/http/response.py | 496 | 514| 186 | 11578 | 49478 | 
| 48 | 23 django/contrib/admin/views/main.py | 214 | 263| 420 | 11998 | 53874 | 
| 49 | 24 django/contrib/admindocs/urls.py | 1 | 51| 307 | 12305 | 54181 | 
| 50 | 24 django/contrib/admin/options.py | 1755 | 1837| 750 | 13055 | 54181 | 
| 51 | 25 django/middleware/common.py | 100 | 115| 165 | 13220 | 55710 | 
| 52 | 26 django/contrib/flatpages/models.py | 1 | 48| 363 | 13583 | 56073 | 
| 53 | 27 django/views/defaults.py | 27 | 76| 401 | 13984 | 57115 | 
| 54 | 27 django/contrib/admindocs/views.py | 320 | 349| 201 | 14185 | 57115 | 
| 55 | 28 django/contrib/contenttypes/views.py | 1 | 89| 711 | 14896 | 57826 | 
| 56 | 28 django/contrib/admindocs/views.py | 185 | 251| 584 | 15480 | 57826 | 
| 57 | 28 django/contrib/auth/mixins.py | 44 | 71| 235 | 15715 | 57826 | 
| 58 | 29 django/urls/resolvers.py | 622 | 695| 681 | 16396 | 63420 | 
| 59 | 29 django/contrib/admin/options.py | 1540 | 1626| 760 | 17156 | 63420 | 
| 60 | 29 django/middleware/common.py | 63 | 75| 136 | 17292 | 63420 | 
| 61 | 29 django/middleware/common.py | 34 | 61| 257 | 17549 | 63420 | 
| 62 | 29 django/contrib/admin/options.py | 1839 | 1907| 584 | 18133 | 63420 | 
| 63 | 29 django/contrib/admindocs/views.py | 252 | 317| 573 | 18706 | 63420 | 
| 64 | 29 django/views/defaults.py | 1 | 24| 149 | 18855 | 63420 | 
| 65 | 30 django/views/generic/__init__.py | 1 | 23| 189 | 19044 | 63610 | 
| 66 | 30 django/contrib/admin/views/main.py | 496 | 527| 224 | 19268 | 63610 | 
| 67 | 31 django/contrib/admin/templatetags/log.py | 1 | 23| 161 | 19429 | 64089 | 
| 68 | 32 django/contrib/redirects/models.py | 1 | 33| 230 | 19659 | 64319 | 
| 69 | 33 django/contrib/redirects/admin.py | 1 | 11| 0 | 19659 | 64387 | 
| 70 | 33 django/views/generic/base.py | 82 | 115| 324 | 19983 | 64387 | 
| 71 | 34 django/template/defaulttags.py | 402 | 436| 270 | 20253 | 74951 | 
| 72 | 34 django/contrib/admindocs/views.py | 117 | 135| 168 | 20421 | 74951 | 
| 73 | 35 django/contrib/flatpages/urls.py | 1 | 7| 0 | 20421 | 74989 | 
| 74 | 36 django/contrib/flatpages/forms.py | 1 | 51| 351 | 20772 | 75476 | 
| 75 | 37 django/contrib/admin/templatetags/admin_modify.py | 48 | 86| 391 | 21163 | 76444 | 
| 76 | 38 django/views/generic/dates.py | 634 | 730| 806 | 21969 | 81885 | 
| 77 | 38 django/http/response.py | 517 | 548| 153 | 22122 | 81885 | 
| 78 | 38 django/contrib/admin/options.py | 836 | 861| 212 | 22334 | 81885 | 
| 79 | 39 django/views/generic/edit.py | 202 | 242| 263 | 22597 | 83601 | 
| 80 | 39 django/contrib/admin/options.py | 804 | 834| 216 | 22813 | 83601 | 
| 81 | 40 django/contrib/auth/__init__.py | 138 | 166| 232 | 23045 | 85182 | 
| 82 | 40 django/contrib/admin/options.py | 1251 | 1324| 659 | 23704 | 85182 | 
| 83 | 41 django/views/csrf.py | 101 | 155| 577 | 24281 | 86726 | 
| 84 | 41 django/contrib/admin/sites.py | 241 | 295| 516 | 24797 | 86726 | 
| 85 | 41 django/contrib/admin/templatetags/admin_list.py | 28 | 43| 131 | 24928 | 86726 | 
| 86 | 41 django/contrib/admin/options.py | 1174 | 1249| 664 | 25592 | 86726 | 
| 87 | 42 django/views/decorators/cache.py | 1 | 24| 209 | 25801 | 87089 | 
| 88 | 42 django/views/defaults.py | 122 | 149| 214 | 26015 | 87089 | 
| 89 | 42 django/views/decorators/clickjacking.py | 1 | 19| 138 | 26153 | 87089 | 
| 90 | 43 django/urls/utils.py | 1 | 63| 460 | 26613 | 87549 | 
| 91 | 44 docs/_ext/djangodocs.py | 74 | 106| 255 | 26868 | 90705 | 
| 92 | 44 django/contrib/admin/views/main.py | 442 | 494| 440 | 27308 | 90705 | 
| 93 | 45 django/contrib/flatpages/admin.py | 1 | 20| 144 | 27452 | 90849 | 
| 94 | 45 django/contrib/auth/__init__.py | 90 | 135| 392 | 27844 | 90849 | 
| 95 | 46 django/contrib/sitemaps/views.py | 22 | 45| 250 | 28094 | 91625 | 
| 96 | 46 django/views/generic/edit.py | 1 | 67| 479 | 28573 | 91625 | 
| 97 | 46 django/contrib/auth/admin.py | 128 | 189| 465 | 29038 | 91625 | 
| 98 | 47 django/contrib/auth/forms.py | 210 | 235| 173 | 29211 | 94737 | 
| 99 | 47 django/contrib/sitemaps/views.py | 48 | 93| 392 | 29603 | 94737 | 
| 100 | 47 django/contrib/admin/sites.py | 197 | 219| 167 | 29770 | 94737 | 
| 101 | 47 django/contrib/admin/options.py | 1127 | 1172| 482 | 30252 | 94737 | 
| 102 | 48 django/views/generic/list.py | 77 | 111| 270 | 30522 | 96309 | 
| 103 | 49 django/contrib/gis/views.py | 1 | 21| 155 | 30677 | 96464 | 
| 104 | 50 django/contrib/redirects/apps.py | 1 | 9| 0 | 30677 | 96515 | 
| 105 | 50 django/contrib/auth/__init__.py | 1 | 38| 241 | 30918 | 96515 | 
| 106 | 50 django/contrib/admin/options.py | 1461 | 1480| 133 | 31051 | 96515 | 
| 107 | 51 django/contrib/redirects/migrations/0002_alter_redirect_new_path_help_text.py | 1 | 25| 117 | 31168 | 96632 | 
| 108 | 51 django/contrib/auth/admin.py | 1 | 22| 188 | 31356 | 96632 | 
| 109 | 52 django/contrib/auth/migrations/0005_alter_user_last_login_null.py | 1 | 17| 0 | 31356 | 96707 | 
| 110 | 53 django/contrib/messages/views.py | 1 | 19| 0 | 31356 | 96803 | 
| 111 | 53 django/views/generic/edit.py | 152 | 199| 340 | 31696 | 96803 | 
| 112 | 53 django/contrib/admindocs/views.py | 55 | 83| 285 | 31981 | 96803 | 
| 113 | 53 docs/_ext/djangodocs.py | 369 | 387| 201 | 32182 | 96803 | 
| 114 | 53 django/urls/resolvers.py | 610 | 620| 120 | 32302 | 96803 | 
| 115 | 53 django/contrib/flatpages/forms.py | 53 | 71| 142 | 32444 | 96803 | 
| 116 | 54 django/middleware/csrf.py | 158 | 179| 173 | 32617 | 99689 | 
| 117 | 55 django/contrib/admin/widgets.py | 326 | 344| 168 | 32785 | 103532 | 
| 118 | 56 django/contrib/admin/tests.py | 126 | 137| 153 | 32938 | 105010 | 
| 119 | 57 django/contrib/flatpages/apps.py | 1 | 9| 0 | 32938 | 105061 | 
| 120 | 57 django/contrib/auth/admin.py | 40 | 99| 504 | 33442 | 105061 | 
| 121 | 57 django/middleware/common.py | 77 | 98| 227 | 33669 | 105061 | 
| 122 | 58 django/contrib/syndication/views.py | 50 | 75| 202 | 33871 | 106801 | 
| 123 | 58 django/middleware/csrf.py | 181 | 203| 230 | 34101 | 106801 | 
| 124 | 58 django/contrib/admin/options.py | 2166 | 2201| 315 | 34416 | 106801 | 
| 125 | 58 django/contrib/admindocs/views.py | 1 | 29| 217 | 34633 | 106801 | 
| 126 | 58 django/template/defaulttags.py | 1250 | 1314| 504 | 35137 | 106801 | 
| 127 | 58 django/contrib/auth/middleware.py | 46 | 82| 360 | 35497 | 106801 | 
| 128 | 59 django/views/decorators/http.py | 1 | 52| 350 | 35847 | 107755 | 
| 129 | 60 django/contrib/admin/models.py | 74 | 94| 161 | 36008 | 108878 | 
| 130 | 61 django/core/paginator.py | 1 | 163| 1260 | 37268 | 110567 | 
| 131 | 61 django/contrib/auth/middleware.py | 84 | 109| 192 | 37460 | 110567 | 
| 132 | 61 django/contrib/admin/views/main.py | 1 | 45| 324 | 37784 | 110567 | 
| 133 | 61 django/views/csrf.py | 15 | 100| 835 | 38619 | 110567 | 
| 134 | 61 django/contrib/admin/sites.py | 321 | 336| 158 | 38777 | 110567 | 
| 135 | 62 django/contrib/admin/migrations/0003_logentry_add_action_flag_choices.py | 1 | 21| 111 | 38888 | 110678 | 
| 136 | 63 django/contrib/admin/views/autocomplete.py | 35 | 46| 131 | 39019 | 111428 | 
| 137 | 63 django/contrib/admin/sites.py | 536 | 569| 281 | 39300 | 111428 | 
| 138 | 63 django/views/generic/base.py | 48 | 80| 284 | 39584 | 111428 | 
| 139 | 63 django/contrib/admin/views/main.py | 48 | 121| 653 | 40237 | 111428 | 
| 140 | 64 django/contrib/auth/models.py | 1 | 32| 216 | 40453 | 114700 | 
| 141 | 64 django/contrib/admindocs/views.py | 86 | 114| 285 | 40738 | 114700 | 
| 142 | 64 django/contrib/admin/models.py | 39 | 72| 241 | 40979 | 114700 | 
| 143 | 64 django/contrib/admin/options.py | 1420 | 1459| 309 | 41288 | 114700 | 
| 144 | 65 django/contrib/admindocs/utils.py | 1 | 25| 151 | 41439 | 116605 | 
| 145 | 66 django/views/__init__.py | 1 | 4| 0 | 41439 | 116620 | 
| 146 | 66 django/contrib/admin/views/main.py | 123 | 212| 861 | 42300 | 116620 | 
| 147 | 66 django/contrib/syndication/views.py | 29 | 48| 196 | 42496 | 116620 | 
| 148 | 66 django/middleware/csrf.py | 205 | 330| 1222 | 43718 | 116620 | 
| 149 | 66 django/views/generic/base.py | 30 | 46| 136 | 43854 | 116620 | 
| 150 | 66 django/contrib/auth/middleware.py | 26 | 44| 178 | 44032 | 116620 | 
| 151 | 66 django/views/i18n.py | 200 | 210| 137 | 44169 | 116620 | 
| 152 | 66 django/contrib/admin/options.py | 717 | 750| 245 | 44414 | 116620 | 
| 153 | 66 django/contrib/admin/widgets.py | 161 | 192| 243 | 44657 | 116620 | 
| 154 | 66 django/contrib/admin/options.py | 634 | 651| 128 | 44785 | 116620 | 
| 155 | 66 django/views/generic/list.py | 139 | 158| 196 | 44981 | 116620 | 
| 156 | 66 django/contrib/auth/forms.py | 33 | 51| 161 | 45142 | 116620 | 
| 157 | 67 django/db/models/base.py | 964 | 978| 212 | 45354 | 133680 | 
| 158 | 67 django/contrib/admin/options.py | 377 | 429| 504 | 45858 | 133680 | 
| 159 | 68 django/core/checks/security/csrf.py | 45 | 68| 157 | 46015 | 134142 | 
| 160 | 68 django/contrib/admin/models.py | 23 | 36| 111 | 46126 | 134142 | 
| 161 | 68 django/core/paginator.py | 166 | 225| 428 | 46554 | 134142 | 
| 162 | 68 django/contrib/sitemaps/views.py | 1 | 19| 132 | 46686 | 134142 | 
| 163 | 69 django/conf/global_settings.py | 496 | 646| 926 | 47612 | 139825 | 
| 164 | 69 django/contrib/admin/options.py | 947 | 985| 269 | 47881 | 139825 | 
| 165 | 70 django/shortcuts.py | 23 | 54| 243 | 48124 | 140922 | 
| 166 | 70 django/contrib/admin/options.py | 1 | 97| 762 | 48886 | 140922 | 
| 167 | 71 django/contrib/admin/templatetags/admin_urls.py | 1 | 57| 405 | 49291 | 141327 | 
| 168 | 72 django/contrib/auth/password_validation.py | 135 | 157| 197 | 49488 | 142811 | 
| 169 | 72 django/contrib/syndication/views.py | 1 | 26| 220 | 49708 | 142811 | 
| 170 | 72 django/contrib/syndication/views.py | 168 | 221| 475 | 50183 | 142811 | 
| 171 | 73 django/contrib/admin/decorators.py | 32 | 71| 268 | 50451 | 143454 | 
| 172 | 73 django/db/models/base.py | 980 | 993| 180 | 50631 | 143454 | 
| 173 | 74 docs/conf.py | 102 | 206| 899 | 51530 | 146491 | 
| 174 | 74 django/shortcuts.py | 1 | 20| 155 | 51685 | 146491 | 
| 175 | 74 django/middleware/common.py | 1 | 32| 247 | 51932 | 146491 | 
| 176 | 74 django/contrib/auth/middleware.py | 1 | 23| 171 | 52103 | 146491 | 
| 177 | 75 django/urls/conf.py | 57 | 78| 162 | 52265 | 147100 | 
| 178 | 76 django/contrib/admin/migrations/0002_logentry_remove_auto_add.py | 1 | 23| 0 | 52265 | 147201 | 
| 179 | 77 django/views/debug.py | 486 | 555| 573 | 52838 | 151792 | 
| 180 | 78 django/views/static.py | 57 | 80| 211 | 53049 | 152844 | 
| 181 | 78 django/views/decorators/cache.py | 27 | 48| 153 | 53202 | 152844 | 
| 182 | 78 docs/_ext/djangodocs.py | 26 | 71| 398 | 53600 | 152844 | 
| 183 | 78 django/views/i18n.py | 277 | 294| 158 | 53758 | 152844 | 
| 184 | 78 django/middleware/common.py | 149 | 175| 254 | 54012 | 152844 | 
| 185 | 79 django/views/decorators/csrf.py | 1 | 57| 460 | 54472 | 153304 | 
| 186 | 79 django/contrib/admin/options.py | 1353 | 1418| 581 | 55053 | 153304 | 
| 187 | 80 django/contrib/sitemaps/__init__.py | 54 | 196| 1026 | 56079 | 154948 | 
| 188 | 80 django/views/generic/edit.py | 129 | 149| 182 | 56261 | 154948 | 
| 189 | 80 django/views/decorators/http.py | 77 | 122| 349 | 56610 | 154948 | 
| 190 | 80 django/contrib/admin/models.py | 136 | 151| 131 | 56741 | 154948 | 
| 191 | 81 django/views/decorators/debug.py | 77 | 92| 132 | 56873 | 155537 | 
| 192 | 81 django/contrib/admin/options.py | 1949 | 1992| 403 | 57276 | 155537 | 
| 193 | 82 django/contrib/auth/migrations/0009_alter_user_last_name_max_length.py | 1 | 17| 0 | 57276 | 155615 | 
| 194 | 83 django/utils/decorators.py | 114 | 152| 316 | 57592 | 157014 | 
| 195 | 83 django/urls/resolvers.py | 508 | 548| 282 | 57874 | 157014 | 
| 196 | 83 django/contrib/admin/options.py | 100 | 130| 223 | 58097 | 157014 | 
| 197 | 83 django/contrib/admin/sites.py | 37 | 77| 313 | 58410 | 157014 | 
| 198 | 83 django/urls/resolvers.py | 1 | 29| 209 | 58619 | 157014 | 
| 199 | 84 django/contrib/redirects/migrations/0001_initial.py | 1 | 41| 274 | 58893 | 157288 | 
| 200 | 85 django/utils/http.py | 346 | 427| 437 | 59330 | 161135 | 
| 201 | 85 django/contrib/admin/decorators.py | 1 | 29| 181 | 59511 | 161135 | 
| 202 | 85 django/views/decorators/http.py | 55 | 76| 272 | 59783 | 161135 | 
| 203 | 85 django/contrib/admin/templatetags/admin_list.py | 1 | 25| 175 | 59958 | 161135 | 
| 204 | 86 django/contrib/admin/filters.py | 20 | 59| 295 | 60253 | 165258 | 
| 205 | 86 django/contrib/admin/options.py | 1654 | 1665| 151 | 60404 | 165258 | 
| 206 | 86 django/contrib/admin/views/autocomplete.py | 48 | 102| 403 | 60807 | 165258 | 
| 207 | 86 django/contrib/auth/forms.py | 135 | 157| 179 | 60986 | 165258 | 


### Hint

```
Did you consider overriding the get_success_url() method? Perhaps that method could be documented. Also there is settings.LOGIN_REDIRECT_URL. Do you have a use case that requires customizing the redirect for different login views?
Yes I have, the issue with that is when redirect_authenticated_user = True, dispatch also has redirect logic. No I don't. It's mostly for symmetry with LogoutView so that I have redirects in the same view file, and not in the settings.
I guess we could see what a patch looks like.
```

## Patch

```diff
diff --git a/django/contrib/auth/views.py b/django/contrib/auth/views.py
--- a/django/contrib/auth/views.py
+++ b/django/contrib/auth/views.py
@@ -43,6 +43,7 @@ class LoginView(SuccessURLAllowedHostsMixin, FormView):
     """
     form_class = AuthenticationForm
     authentication_form = None
+    next_page = None
     redirect_field_name = REDIRECT_FIELD_NAME
     template_name = 'registration/login.html'
     redirect_authenticated_user = False
@@ -63,8 +64,7 @@ def dispatch(self, request, *args, **kwargs):
         return super().dispatch(request, *args, **kwargs)
 
     def get_success_url(self):
-        url = self.get_redirect_url()
-        return url or resolve_url(settings.LOGIN_REDIRECT_URL)
+        return self.get_redirect_url() or self.get_default_redirect_url()
 
     def get_redirect_url(self):
         """Return the user-originating redirect URL if it's safe."""
@@ -79,6 +79,10 @@ def get_redirect_url(self):
         )
         return redirect_to if url_is_safe else ''
 
+    def get_default_redirect_url(self):
+        """Return the default redirect URL."""
+        return resolve_url(self.next_page or settings.LOGIN_REDIRECT_URL)
+
     def get_form_class(self):
         return self.authentication_form or self.form_class
 

```

## Test Patch

```diff
diff --git a/tests/auth_tests/test_views.py b/tests/auth_tests/test_views.py
--- a/tests/auth_tests/test_views.py
+++ b/tests/auth_tests/test_views.py
@@ -52,8 +52,8 @@ def setUpTestData(cls):
         cls.u1 = User.objects.create_user(username='testclient', password='password', email='testclient@example.com')
         cls.u3 = User.objects.create_user(username='staff', password='password', email='staffmember@example.com')
 
-    def login(self, username='testclient', password='password'):
-        response = self.client.post('/login/', {
+    def login(self, username='testclient', password='password', url='/login/'):
+        response = self.client.post(url, {
             'username': username,
             'password': password,
         })
@@ -726,6 +726,31 @@ def test_login_session_without_hash_session_key(self):
         self.login()
         self.assertNotEqual(original_session_key, self.client.session.session_key)
 
+    def test_login_get_default_redirect_url(self):
+        response = self.login(url='/login/get_default_redirect_url/')
+        self.assertRedirects(response, '/custom/', fetch_redirect_response=False)
+
+    def test_login_next_page(self):
+        response = self.login(url='/login/next_page/')
+        self.assertRedirects(response, '/somewhere/', fetch_redirect_response=False)
+
+    def test_login_named_next_page_named(self):
+        response = self.login(url='/login/next_page/named/')
+        self.assertRedirects(response, '/password_reset/', fetch_redirect_response=False)
+
+    @override_settings(LOGIN_REDIRECT_URL='/custom/')
+    def test_login_next_page_overrides_login_redirect_url_setting(self):
+        response = self.login(url='/login/next_page/')
+        self.assertRedirects(response, '/somewhere/', fetch_redirect_response=False)
+
+    def test_login_redirect_url_overrides_next_page(self):
+        response = self.login(url='/login/next_page/?next=/test/')
+        self.assertRedirects(response, '/test/', fetch_redirect_response=False)
+
+    def test_login_redirect_url_overrides_get_default_redirect_url(self):
+        response = self.login(url='/login/get_default_redirect_url/?next=/test/')
+        self.assertRedirects(response, '/test/', fetch_redirect_response=False)
+
 
 class LoginURLSettings(AuthViewsTestCase):
     """Tests for settings.LOGIN_URL."""
diff --git a/tests/auth_tests/urls.py b/tests/auth_tests/urls.py
--- a/tests/auth_tests/urls.py
+++ b/tests/auth_tests/urls.py
@@ -3,6 +3,7 @@
 from django.contrib.auth.decorators import login_required, permission_required
 from django.contrib.auth.forms import AuthenticationForm
 from django.contrib.auth.urls import urlpatterns as auth_urlpatterns
+from django.contrib.auth.views import LoginView
 from django.contrib.messages.api import info
 from django.http import HttpRequest, HttpResponse
 from django.shortcuts import render
@@ -78,6 +79,11 @@ def login_and_permission_required_exception(request):
     pass
 
 
+class CustomDefaultRedirectURLLoginView(LoginView):
+    def get_default_redirect_url(self):
+        return '/custom/'
+
+
 # special urls for auth test cases
 urlpatterns = auth_urlpatterns + [
     path('logout/custom_query/', views.LogoutView.as_view(redirect_field_name='follow')),
@@ -149,6 +155,9 @@ def login_and_permission_required_exception(request):
          views.LoginView.as_view(redirect_authenticated_user=True)),
     path('login/allowed_hosts/',
          views.LoginView.as_view(success_url_allowed_hosts={'otherserver'})),
+    path('login/get_default_redirect_url/', CustomDefaultRedirectURLLoginView.as_view()),
+    path('login/next_page/', views.LoginView.as_view(next_page='/somewhere/')),
+    path('login/next_page/named/', views.LoginView.as_view(next_page='password_reset')),
 
     path('permission_required_redirect/', permission_required_redirect),
     path('permission_required_exception/', permission_required_exception),

```


## Code snippets

### 1 - django/contrib/auth/views.py:

Start line: 129, End line: 163

```python
class LogoutView(SuccessURLAllowedHostsMixin, TemplateView):

    def get_next_page(self):
        if self.next_page is not None:
            next_page = resolve_url(self.next_page)
        elif settings.LOGOUT_REDIRECT_URL:
            next_page = resolve_url(settings.LOGOUT_REDIRECT_URL)
        else:
            next_page = self.next_page

        if (self.redirect_field_name in self.request.POST or
                self.redirect_field_name in self.request.GET):
            next_page = self.request.POST.get(
                self.redirect_field_name,
                self.request.GET.get(self.redirect_field_name)
            )
            url_is_safe = url_has_allowed_host_and_scheme(
                url=next_page,
                allowed_hosts=self.get_success_url_allowed_hosts(),
                require_https=self.request.is_secure(),
            )
            # Security check -- Ensure the user-originating redirection URL is
            # safe.
            if not url_is_safe:
                next_page = self.request.path
        return next_page

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        current_site = get_current_site(self.request)
        context.update({
            'site': current_site,
            'site_name': current_site.name,
            'title': _('Logged out'),
            **(self.extra_context or {})
        })
        return context
```
### 2 - django/contrib/auth/views.py:

Start line: 166, End line: 186

```python
def logout_then_login(request, login_url=None):
    """
    Log out the user if they are logged in. Then redirect to the login page.
    """
    login_url = resolve_url(login_url or settings.LOGIN_URL)
    return LogoutView.as_view(next_page=login_url)(request)


def redirect_to_login(next, login_url=None, redirect_field_name=REDIRECT_FIELD_NAME):
    """
    Redirect the user to the login page, passing the given 'next' page.
    """
    resolved_url = resolve_url(login_url or settings.LOGIN_URL)

    login_url_parts = list(urlparse(resolved_url))
    if redirect_field_name:
        querystring = QueryDict(login_url_parts[4], mutable=True)
        querystring[redirect_field_name] = next
        login_url_parts[4] = querystring.urlencode(safe='/')

    return HttpResponseRedirect(urlunparse(login_url_parts))
```
### 3 - django/contrib/auth/views.py:

Start line: 107, End line: 127

```python
class LogoutView(SuccessURLAllowedHostsMixin, TemplateView):
    """
    Log out the user and display the 'You are logged out' message.
    """
    next_page = None
    redirect_field_name = REDIRECT_FIELD_NAME
    template_name = 'registration/logged_out.html'
    extra_context = None

    @method_decorator(never_cache)
    def dispatch(self, request, *args, **kwargs):
        auth_logout(request)
        next_page = self.get_next_page()
        if next_page:
            # Redirect to this page until the session has been cleared.
            return HttpResponseRedirect(next_page)
        return super().dispatch(request, *args, **kwargs)

    def post(self, request, *args, **kwargs):
        """Logout may be done via POST."""
        return self.get(request, *args, **kwargs)
```
### 4 - django/contrib/auth/views.py:

Start line: 65, End line: 104

```python
class LoginView(SuccessURLAllowedHostsMixin, FormView):

    def get_success_url(self):
        url = self.get_redirect_url()
        return url or resolve_url(settings.LOGIN_REDIRECT_URL)

    def get_redirect_url(self):
        """Return the user-originating redirect URL if it's safe."""
        redirect_to = self.request.POST.get(
            self.redirect_field_name,
            self.request.GET.get(self.redirect_field_name, '')
        )
        url_is_safe = url_has_allowed_host_and_scheme(
            url=redirect_to,
            allowed_hosts=self.get_success_url_allowed_hosts(),
            require_https=self.request.is_secure(),
        )
        return redirect_to if url_is_safe else ''

    def get_form_class(self):
        return self.authentication_form or self.form_class

    def get_form_kwargs(self):
        kwargs = super().get_form_kwargs()
        kwargs['request'] = self.request
        return kwargs

    def form_valid(self, form):
        """Security check complete. Log the user in."""
        auth_login(self.request, form.get_user())
        return HttpResponseRedirect(self.get_success_url())

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        current_site = get_current_site(self.request)
        context.update({
            self.redirect_field_name: self.get_redirect_url(),
            'site': current_site,
            'site_name': current_site.name,
            **(self.extra_context or {})
        })
        return context
```
### 5 - django/contrib/auth/views.py:

Start line: 40, End line: 63

```python
class LoginView(SuccessURLAllowedHostsMixin, FormView):
    """
    Display the login form and handle the login action.
    """
    form_class = AuthenticationForm
    authentication_form = None
    redirect_field_name = REDIRECT_FIELD_NAME
    template_name = 'registration/login.html'
    redirect_authenticated_user = False
    extra_context = None

    @method_decorator(sensitive_post_parameters())
    @method_decorator(csrf_protect)
    @method_decorator(never_cache)
    def dispatch(self, request, *args, **kwargs):
        if self.redirect_authenticated_user and self.request.user.is_authenticated:
            redirect_to = self.get_success_url()
            if redirect_to == self.request.path:
                raise ValueError(
                    "Redirection loop for authenticated user detected. Check that "
                    "your LOGIN_REDIRECT_URL doesn't point to a login page."
                )
            return HttpResponseRedirect(redirect_to)
        return super().dispatch(request, *args, **kwargs)
```
### 6 - django/contrib/auth/views.py:

Start line: 330, End line: 362

```python
class PasswordChangeView(PasswordContextMixin, FormView):
    form_class = PasswordChangeForm
    success_url = reverse_lazy('password_change_done')
    template_name = 'registration/password_change_form.html'
    title = _('Password change')

    @method_decorator(sensitive_post_parameters())
    @method_decorator(csrf_protect)
    @method_decorator(login_required)
    def dispatch(self, *args, **kwargs):
        return super().dispatch(*args, **kwargs)

    def get_form_kwargs(self):
        kwargs = super().get_form_kwargs()
        kwargs['user'] = self.request.user
        return kwargs

    def form_valid(self, form):
        form.save()
        # Updating the password logs out all other sessions for the user
        # except the current one.
        update_session_auth_hash(self.request, form.user)
        return super().form_valid(form)


class PasswordChangeDoneView(PasswordContextMixin, TemplateView):
    template_name = 'registration/password_change_done.html'
    title = _('Password change successful')

    @method_decorator(login_required)
    def dispatch(self, *args, **kwargs):
        return super().dispatch(*args, **kwargs)
```
### 7 - django/views/generic/base.py:

Start line: 188, End line: 219

```python
class RedirectView(View):

    def get(self, request, *args, **kwargs):
        url = self.get_redirect_url(*args, **kwargs)
        if url:
            if self.permanent:
                return HttpResponsePermanentRedirect(url)
            else:
                return HttpResponseRedirect(url)
        else:
            logger.warning(
                'Gone: %s', request.path,
                extra={'status_code': 410, 'request': request}
            )
            return HttpResponseGone()

    def head(self, request, *args, **kwargs):
        return self.get(request, *args, **kwargs)

    def post(self, request, *args, **kwargs):
        return self.get(request, *args, **kwargs)

    def options(self, request, *args, **kwargs):
        return self.get(request, *args, **kwargs)

    def delete(self, request, *args, **kwargs):
        return self.get(request, *args, **kwargs)

    def put(self, request, *args, **kwargs):
        return self.get(request, *args, **kwargs)

    def patch(self, request, *args, **kwargs):
        return self.get(request, *args, **kwargs)
```
### 8 - django/contrib/auth/decorators.py:

Start line: 38, End line: 74

```python
def login_required(function=None, redirect_field_name=REDIRECT_FIELD_NAME, login_url=None):
    """
    Decorator for views that checks that the user is logged in, redirecting
    to the log-in page if necessary.
    """
    actual_decorator = user_passes_test(
        lambda u: u.is_authenticated,
        login_url=login_url,
        redirect_field_name=redirect_field_name
    )
    if function:
        return actual_decorator(function)
    return actual_decorator


def permission_required(perm, login_url=None, raise_exception=False):
    """
    Decorator for views that checks whether a user has a particular permission
    enabled, redirecting to the log-in page if necessary.
    If the raise_exception parameter is given the PermissionDenied exception
    is raised.
    """
    def check_perms(user):
        if isinstance(perm, str):
            perms = (perm,)
        else:
            perms = perm
        # First check if the user has the permission (even anon users)
        if user.has_perms(perms):
            return True
        # In case the 403 handler should be called raise the exception
        if raise_exception:
            raise PermissionDenied
        # As the last resort, show the login form
        return False
    return user_passes_test(check_perms, login_url=login_url)
```
### 9 - django/contrib/admin/views/decorators.py:

Start line: 1, End line: 19

```python
from django.contrib.auth import REDIRECT_FIELD_NAME
from django.contrib.auth.decorators import user_passes_test


def staff_member_required(view_func=None, redirect_field_name=REDIRECT_FIELD_NAME,
                          login_url='admin:login'):
    """
    Decorator for views that checks that the user is logged in and is a staff
    member, redirecting to the login page if necessary.
    """
    actual_decorator = user_passes_test(
        lambda u: u.is_active and u.is_staff,
        login_url=login_url,
        redirect_field_name=redirect_field_name
    )
    if view_func:
        return actual_decorator(view_func)
    return actual_decorator
```
### 10 - django/contrib/auth/views.py:

Start line: 1, End line: 37

```python
from urllib.parse import urlparse, urlunparse

from django.conf import settings
# Avoid shadowing the login() and logout() views below.
from django.contrib.auth import (
    REDIRECT_FIELD_NAME, get_user_model, login as auth_login,
    logout as auth_logout, update_session_auth_hash,
)
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import (
    AuthenticationForm, PasswordChangeForm, PasswordResetForm, SetPasswordForm,
)
from django.contrib.auth.tokens import default_token_generator
from django.contrib.sites.shortcuts import get_current_site
from django.core.exceptions import ValidationError
from django.http import HttpResponseRedirect, QueryDict
from django.shortcuts import resolve_url
from django.urls import reverse_lazy
from django.utils.decorators import method_decorator
from django.utils.http import (
    url_has_allowed_host_and_scheme, urlsafe_base64_decode,
)
from django.utils.translation import gettext_lazy as _
from django.views.decorators.cache import never_cache
from django.views.decorators.csrf import csrf_protect
from django.views.decorators.debug import sensitive_post_parameters
from django.views.generic.base import TemplateView
from django.views.generic.edit import FormView

UserModel = get_user_model()


class SuccessURLAllowedHostsMixin:
    success_url_allowed_hosts = set()

    def get_success_url_allowed_hosts(self):
        return {self.request.get_host(), *self.success_url_allowed_hosts}
```
### 11 - django/contrib/auth/views.py:

Start line: 286, End line: 327

```python
class PasswordResetConfirmView(PasswordContextMixin, FormView):

    def get_user(self, uidb64):
        try:
            # urlsafe_base64_decode() decodes to bytestring
            uid = urlsafe_base64_decode(uidb64).decode()
            user = UserModel._default_manager.get(pk=uid)
        except (TypeError, ValueError, OverflowError, UserModel.DoesNotExist, ValidationError):
            user = None
        return user

    def get_form_kwargs(self):
        kwargs = super().get_form_kwargs()
        kwargs['user'] = self.user
        return kwargs

    def form_valid(self, form):
        user = form.save()
        del self.request.session[INTERNAL_RESET_SESSION_TOKEN]
        if self.post_reset_login:
            auth_login(self.request, user, self.post_reset_login_backend)
        return super().form_valid(form)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        if self.validlink:
            context['validlink'] = True
        else:
            context.update({
                'form': None,
                'title': _('Password reset unsuccessful'),
                'validlink': False,
            })
        return context


class PasswordResetCompleteView(PasswordContextMixin, TemplateView):
    template_name = 'registration/password_reset_complete.html'
    title = _('Password reset complete')

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['login_url'] = resolve_url(settings.LOGIN_URL)
        return context
```
### 12 - django/contrib/auth/views.py:

Start line: 247, End line: 284

```python
class PasswordResetConfirmView(PasswordContextMixin, FormView):
    form_class = SetPasswordForm
    post_reset_login = False
    post_reset_login_backend = None
    reset_url_token = 'set-password'
    success_url = reverse_lazy('password_reset_complete')
    template_name = 'registration/password_reset_confirm.html'
    title = _('Enter new password')
    token_generator = default_token_generator

    @method_decorator(sensitive_post_parameters())
    @method_decorator(never_cache)
    def dispatch(self, *args, **kwargs):
        assert 'uidb64' in kwargs and 'token' in kwargs

        self.validlink = False
        self.user = self.get_user(kwargs['uidb64'])

        if self.user is not None:
            token = kwargs['token']
            if token == self.reset_url_token:
                session_token = self.request.session.get(INTERNAL_RESET_SESSION_TOKEN)
                if self.token_generator.check_token(self.user, session_token):
                    # If the token is valid, display the password reset form.
                    self.validlink = True
                    return super().dispatch(*args, **kwargs)
            else:
                if self.token_generator.check_token(self.user, token):
                    # Store the token in the session and redirect to the
                    # password reset form at a URL without the token. That
                    # avoids the possibility of leaking the token in the
                    # HTTP Referer header.
                    self.request.session[INTERNAL_RESET_SESSION_TOKEN] = token
                    redirect_url = self.request.path.replace(token, self.reset_url_token)
                    return HttpResponseRedirect(redirect_url)

        # Display the "Password reset unsuccessful" page.
        return self.render_to_response(self.get_context_data())
```
### 14 - django/contrib/auth/views.py:

Start line: 224, End line: 244

```python
class PasswordResetView(PasswordContextMixin, FormView):

    def form_valid(self, form):
        opts = {
            'use_https': self.request.is_secure(),
            'token_generator': self.token_generator,
            'from_email': self.from_email,
            'email_template_name': self.email_template_name,
            'subject_template_name': self.subject_template_name,
            'request': self.request,
            'html_email_template_name': self.html_email_template_name,
            'extra_email_context': self.extra_email_context,
        }
        form.save(**opts)
        return super().form_valid(form)


INTERNAL_RESET_SESSION_TOKEN = '_password_reset_token'


class PasswordResetDoneView(PasswordContextMixin, TemplateView):
    template_name = 'registration/password_reset_done.html'
    title = _('Password reset sent')
```
### 21 - django/contrib/auth/views.py:

Start line: 208, End line: 222

```python
class PasswordResetView(PasswordContextMixin, FormView):
    email_template_name = 'registration/password_reset_email.html'
    extra_email_context = None
    form_class = PasswordResetForm
    from_email = None
    html_email_template_name = None
    subject_template_name = 'registration/password_reset_subject.txt'
    success_url = reverse_lazy('password_reset_done')
    template_name = 'registration/password_reset_form.html'
    title = _('Password reset')
    token_generator = default_token_generator

    @method_decorator(csrf_protect)
    def dispatch(self, *args, **kwargs):
        return super().dispatch(*args, **kwargs)
```
### 34 - django/contrib/auth/views.py:

Start line: 189, End line: 205

```python
# Class-based password reset views
# - PasswordResetView sends the mail
# - PasswordResetDoneView shows a success message for the above
# - PasswordResetConfirmView checks the link the user clicked and
#   prompts for a new password
# - PasswordResetCompleteView shows a success message for the above

class PasswordContextMixin:
    extra_context = None

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context.update({
            'title': self.title,
            **(self.extra_context or {})
        })
        return context
```
