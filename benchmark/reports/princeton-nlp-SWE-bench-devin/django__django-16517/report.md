# django__django-16517

| **django/django** | `c626173833784c86920b448793ac45005af4c058` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 13949 |
| **Any found context length** | 13949 |
| **Avg pos** | 90.0 |
| **Min pos** | 45 |
| **Max pos** | 45 |
| **Top file pos** | 4 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/contrib/admindocs/utils.py b/django/contrib/admindocs/utils.py
--- a/django/contrib/admindocs/utils.py
+++ b/django/contrib/admindocs/utils.py
@@ -101,6 +101,9 @@ def parse_rst(text, default_reference_context, thing_being_parsed=None):
 
 
 def create_reference_role(rolename, urlbase):
+    # Views and template names are case-sensitive.
+    is_case_sensitive = rolename in ["template", "view"]
+
     def _role(name, rawtext, text, lineno, inliner, options=None, content=None):
         if options is None:
             options = {}
@@ -111,7 +114,7 @@ def _role(name, rawtext, text, lineno, inliner, options=None, content=None):
                 urlbase
                 % (
                     inliner.document.settings.link_base,
-                    text.lower(),
+                    text if is_case_sensitive else text.lower(),
                 )
             ),
             **options,

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/contrib/admindocs/utils.py | 104 | 104 | 45 | 4 | 13949
| django/contrib/admindocs/utils.py | 114 | 114 | 45 | 4 | 13949


## Problem Statement

```
Mixed-case views/templates names causes 404 on :view:/:template: directive.
Description
	
​https://github.com/django/django/blob/main/django/contrib/admindocs/views.py#L168
Using a class based view, 
class OrderSearch(LoginRequiredMixin, UserPassesTestMixin, ListView):
add a doc comment such as
:view:orders.views.Orders
causes a 404 when you click on the link in the docs
Page not found (404)
Request Method:		GET
Request URL:		​http://localhost:8000/admin/doc/views/orders.views.orders/
Raised by:		django.contrib.admindocs.views.ViewDetailView
I'm not sure exactly where orders becomes lowercase, but I thought it might have something to do with the _get_view_func

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 django/contrib/admindocs/views.py | 164 | 182| 187 | 187 | 3488 | 
| 2 | 1 django/contrib/admindocs/views.py | 1 | 36| 252 | 439 | 3488 | 
| 3 | 1 django/contrib/admindocs/views.py | 141 | 161| 170 | 609 | 3488 | 
| 4 | 1 django/contrib/admindocs/views.py | 39 | 62| 163 | 772 | 3488 | 
| 5 | 1 django/contrib/admindocs/views.py | 184 | 210| 238 | 1010 | 3488 | 
| 6 | 1 django/contrib/admindocs/views.py | 395 | 428| 211 | 1221 | 3488 | 
| 7 | 2 django/contrib/admindocs/urls.py | 1 | 51| 307 | 1528 | 3795 | 
| 8 | 3 django/views/generic/__init__.py | 1 | 40| 204 | 1732 | 4000 | 
| 9 | 3 django/contrib/admindocs/views.py | 102 | 138| 301 | 2033 | 4000 | 
| 10 | **4 django/contrib/admindocs/utils.py** | 1 | 28| 185 | 2218 | 5955 | 
| 11 | 4 django/contrib/admindocs/views.py | 65 | 99| 297 | 2515 | 5955 | 
| 12 | 4 django/contrib/admindocs/views.py | 213 | 297| 615 | 3130 | 5955 | 
| 13 | 5 django/views/defaults.py | 1 | 26| 151 | 3281 | 6945 | 
| 14 | 6 django/contrib/admin/sites.py | 228 | 249| 221 | 3502 | 11391 | 
| 15 | 6 django/contrib/admindocs/views.py | 298 | 392| 640 | 4142 | 11391 | 
| 16 | 7 django/views/csrf.py | 15 | 100| 839 | 4981 | 12944 | 
| 17 | 7 django/contrib/admin/sites.py | 442 | 456| 129 | 5110 | 12944 | 
| 18 | 7 django/views/csrf.py | 101 | 161| 581 | 5691 | 12944 | 
| 19 | 8 django/contrib/admin/views/main.py | 1 | 51| 324 | 6015 | 17478 | 
| 20 | 9 django/contrib/admin/options.py | 1914 | 2002| 676 | 6691 | 36718 | 
| 21 | 9 django/views/defaults.py | 29 | 79| 377 | 7068 | 36718 | 
| 22 | 10 django/contrib/auth/urls.py | 1 | 37| 253 | 7321 | 36971 | 
| 23 | 11 django/contrib/auth/password_validation.py | 217 | 267| 386 | 7707 | 38865 | 
| 24 | 12 django/views/generic/base.py | 256 | 286| 246 | 7953 | 40771 | 
| 25 | 13 django/__init__.py | 1 | 25| 173 | 8126 | 40944 | 
| 26 | 14 django/contrib/auth/views.py | 1 | 32| 255 | 8381 | 43658 | 
| 27 | 14 django/contrib/auth/views.py | 348 | 380| 239 | 8620 | 43658 | 
| 28 | 15 django/views/generic/edit.py | 161 | 212| 340 | 8960 | 45521 | 
| 29 | 16 django/urls/conf.py | 61 | 96| 266 | 9226 | 46238 | 
| 30 | 16 django/views/defaults.py | 124 | 150| 197 | 9423 | 46238 | 
| 31 | 17 django/contrib/flatpages/views.py | 1 | 45| 399 | 9822 | 46828 | 
| 32 | 18 django/contrib/contenttypes/fields.py | 455 | 472| 123 | 9945 | 52656 | 
| 33 | 18 django/contrib/auth/views.py | 296 | 345| 325 | 10270 | 52656 | 
| 34 | 18 django/views/generic/base.py | 63 | 78| 131 | 10401 | 52656 | 
| 35 | 18 django/views/defaults.py | 102 | 121| 144 | 10545 | 52656 | 
| 36 | 18 django/views/generic/base.py | 145 | 179| 204 | 10749 | 52656 | 
| 37 | 18 django/contrib/admin/options.py | 683 | 723| 280 | 11029 | 52656 | 
| 38 | 18 django/views/csrf.py | 1 | 13| 132 | 11161 | 52656 | 
| 39 | 19 django/urls/resolvers.py | 739 | 827| 722 | 11883 | 58720 | 
| 40 | 20 django/contrib/admindocs/middleware.py | 1 | 34| 257 | 12140 | 58978 | 
| 41 | 20 django/views/generic/base.py | 36 | 61| 147 | 12287 | 58978 | 
| 42 | 21 django/views/i18n.py | 87 | 190| 702 | 12989 | 61464 | 
| 43 | 22 django/views/__init__.py | 1 | 4| 0 | 12989 | 61479 | 
| 44 | 22 django/contrib/admin/options.py | 1749 | 1851| 780 | 13769 | 61479 | 
| **-> 45 <-** | **22 django/contrib/admindocs/utils.py** | 91 | 121| 180 | 13949 | 61479 | 
| 46 | 22 django/contrib/admin/options.py | 1 | 114| 776 | 14725 | 61479 | 
| 47 | 22 django/contrib/admin/sites.py | 567 | 589| 163 | 14888 | 61479 | 
| 48 | 23 django/contrib/messages/views.py | 1 | 20| 0 | 14888 | 61575 | 
| 49 | 24 django/contrib/admin/views/decorators.py | 1 | 20| 137 | 15025 | 61713 | 
| 50 | 25 django/contrib/auth/admin.py | 121 | 147| 286 | 15311 | 63484 | 
| 51 | 25 django/contrib/admin/options.py | 2173 | 2231| 444 | 15755 | 63484 | 
| 52 | 25 django/contrib/admin/options.py | 1852 | 1883| 303 | 16058 | 63484 | 
| 53 | 25 django/contrib/admin/options.py | 2003 | 2094| 784 | 16842 | 63484 | 
| 54 | 25 django/contrib/auth/views.py | 229 | 249| 163 | 17005 | 63484 | 
| 55 | 26 django/contrib/admin/views/autocomplete.py | 66 | 123| 425 | 17430 | 64325 | 
| 56 | 26 django/contrib/auth/views.py | 252 | 294| 382 | 17812 | 64325 | 
| 57 | 26 django/contrib/auth/views.py | 213 | 227| 133 | 17945 | 64325 | 
| 58 | 27 django/urls/base.py | 27 | 88| 440 | 18385 | 65521 | 
| 59 | 27 django/contrib/admin/sites.py | 251 | 313| 527 | 18912 | 65521 | 
| 60 | 27 django/contrib/admin/views/main.py | 554 | 586| 227 | 19139 | 65521 | 
| 61 | 27 django/contrib/admin/views/main.py | 495 | 552| 465 | 19604 | 65521 | 
| 62 | 27 django/views/generic/edit.py | 241 | 275| 223 | 19827 | 65521 | 
| 63 | 27 django/contrib/flatpages/views.py | 48 | 71| 191 | 20018 | 65521 | 
| 64 | 27 django/contrib/admin/views/main.py | 311 | 341| 270 | 20288 | 65521 | 
| 65 | 28 django/db/models/base.py | 1812 | 1834| 175 | 20463 | 84237 | 
| 66 | 29 django/contrib/gis/views.py | 1 | 23| 160 | 20623 | 84397 | 
| 67 | 29 django/views/generic/edit.py | 1 | 73| 491 | 21114 | 84397 | 
| 68 | 29 django/contrib/auth/views.py | 194 | 210| 127 | 21241 | 84397 | 
| 69 | 29 django/contrib/admin/views/main.py | 256 | 309| 428 | 21669 | 84397 | 
| 70 | 29 django/contrib/admin/sites.py | 204 | 226| 167 | 21836 | 84397 | 
| 71 | 30 django/views/generic/detail.py | 113 | 181| 518 | 22354 | 85727 | 
| 72 | 30 django/contrib/auth/admin.py | 1 | 25| 195 | 22549 | 85727 | 
| 73 | 31 django/core/checks/security/csrf.py | 45 | 68| 159 | 22708 | 86192 | 
| 74 | 32 django/contrib/flatpages/urls.py | 1 | 7| 0 | 22708 | 86230 | 
| 75 | 33 django/contrib/flatpages/models.py | 1 | 50| 368 | 23076 | 86598 | 
| 76 | 34 django/contrib/sitemaps/views.py | 1 | 27| 162 | 23238 | 87670 | 
| 77 | 35 docs/_ext/djangodocs.py | 383 | 402| 204 | 23442 | 90894 | 
| 78 | 35 django/contrib/admin/views/main.py | 153 | 254| 863 | 24305 | 90894 | 
| 79 | 35 django/contrib/admindocs/views.py | 455 | 483| 194 | 24499 | 90894 | 
| 80 | 35 django/contrib/admin/options.py | 2467 | 2502| 315 | 24814 | 90894 | 
| 81 | 36 django/views/generic/list.py | 150 | 175| 205 | 25019 | 92496 | 
| 82 | 37 django/views/debug.py | 1 | 57| 357 | 25376 | 97552 | 
| 83 | 38 django/contrib/admin/templatetags/admin_urls.py | 1 | 67| 419 | 25795 | 97971 | 
| 84 | 38 django/views/i18n.py | 209 | 219| 137 | 25932 | 97971 | 
| 85 | 38 django/contrib/sitemaps/views.py | 42 | 88| 423 | 26355 | 97971 | 
| 86 | 38 django/urls/resolvers.py | 726 | 737| 120 | 26475 | 97971 | 
| 87 | 38 django/contrib/admin/options.py | 2233 | 2282| 450 | 26925 | 97971 | 
| 88 | 39 django/contrib/syndication/views.py | 29 | 49| 193 | 27118 | 99829 | 
| 89 | 39 django/views/generic/detail.py | 79 | 110| 241 | 27359 | 99829 | 
| 90 | 39 django/views/generic/edit.py | 76 | 108| 269 | 27628 | 99829 | 
| 91 | 40 django/contrib/admin/templatetags/admin_list.py | 1 | 32| 187 | 27815 | 103573 | 
| 92 | 41 django/contrib/flatpages/middleware.py | 1 | 21| 147 | 27962 | 103721 | 
| 93 | 42 django/views/decorators/cache.py | 29 | 46| 129 | 28091 | 104196 | 
| 94 | 42 django/contrib/admin/sites.py | 1 | 33| 224 | 28315 | 104196 | 
| 95 | 43 django/urls/utils.py | 1 | 67| 464 | 28779 | 104660 | 
| 96 | 44 django/views/decorators/common.py | 1 | 18| 112 | 28891 | 104773 | 
| 97 | 44 django/contrib/admin/options.py | 1510 | 1536| 233 | 29124 | 104773 | 
| 98 | 44 django/contrib/admin/templatetags/admin_list.py | 175 | 191| 140 | 29264 | 104773 | 
| 99 | 45 django/core/management/templates.py | 260 | 293| 240 | 29504 | 107800 | 
| 100 | 46 django/views/static.py | 56 | 79| 211 | 29715 | 108786 | 
| 101 | 46 docs/_ext/djangodocs.py | 74 | 108| 257 | 29972 | 108786 | 
| 102 | 46 docs/_ext/djangodocs.py | 26 | 71| 398 | 30370 | 108786 | 
| 103 | 47 django/views/generic/dates.py | 302 | 326| 204 | 30574 | 114296 | 
| 104 | 47 django/views/defaults.py | 82 | 99| 121 | 30695 | 114296 | 
| 105 | 47 django/contrib/auth/views.py | 65 | 88| 185 | 30880 | 114296 | 
| 106 | 48 django/contrib/admin/checks.py | 954 | 978| 197 | 31077 | 123822 | 
| 107 | 49 django/contrib/auth/decorators.py | 43 | 83| 276 | 31353 | 124414 | 
| 108 | 50 django/contrib/admin/helpers.py | 530 | 556| 199 | 31552 | 128044 | 
| 109 | 50 django/contrib/syndication/views.py | 1 | 26| 223 | 31775 | 128044 | 
| 110 | 50 django/contrib/sitemaps/views.py | 91 | 141| 369 | 32144 | 128044 | 
| 111 | 50 django/contrib/admin/views/main.py | 390 | 453| 513 | 32657 | 128044 | 
| 112 | 50 django/contrib/syndication/views.py | 51 | 76| 202 | 32859 | 128044 | 
| 113 | 50 django/contrib/admin/options.py | 1257 | 1319| 511 | 33370 | 128044 | 
| 114 | 50 django/views/generic/base.py | 80 | 123| 369 | 33739 | 128044 | 
| 115 | 50 django/contrib/admin/views/main.py | 54 | 151| 736 | 34475 | 128044 | 
| 116 | 51 django/contrib/admin/__init__.py | 1 | 51| 281 | 34756 | 128325 | 
| 117 | 52 django/middleware/csrf.py | 348 | 411| 585 | 35341 | 132436 | 
| 118 | 52 django/views/static.py | 1 | 14| 109 | 35450 | 132436 | 
| 119 | 52 django/views/debug.py | 590 | 648| 454 | 35904 | 132436 | 
| 120 | 52 django/views/generic/base.py | 125 | 143| 180 | 36084 | 132436 | 
| 121 | 53 django/contrib/admindocs/apps.py | 1 | 8| 0 | 36084 | 132478 | 
| 122 | 53 django/urls/base.py | 1 | 24| 170 | 36254 | 132478 | 
| 123 | 53 django/contrib/admin/options.py | 666 | 681| 123 | 36377 | 132478 | 
| 124 | 53 django/views/i18n.py | 300 | 323| 167 | 36544 | 132478 | 
| 125 | 54 docs/conf.py | 133 | 236| 934 | 37478 | 136032 | 
| 126 | 54 django/contrib/auth/decorators.py | 1 | 40| 315 | 37793 | 136032 | 
| 127 | 54 django/middleware/csrf.py | 413 | 468| 577 | 38370 | 136032 | 
| 128 | 54 django/middleware/csrf.py | 199 | 218| 151 | 38521 | 136032 | 
| 129 | 54 django/contrib/admin/options.py | 1411 | 1508| 710 | 39231 | 136032 | 
| 130 | 54 django/views/generic/list.py | 49 | 79| 251 | 39482 | 136032 | 
| 131 | 54 django/contrib/admin/options.py | 357 | 430| 500 | 39982 | 136032 | 
| 132 | 55 django/contrib/admin/utils.py | 1 | 28| 246 | 40228 | 140300 | 
| 133 | 55 django/urls/resolvers.py | 1 | 30| 216 | 40444 | 140300 | 
| 134 | 55 django/contrib/admin/options.py | 236 | 249| 139 | 40583 | 140300 | 
| 135 | 56 django/contrib/staticfiles/views.py | 1 | 40| 270 | 40853 | 140570 | 
| 136 | 56 django/contrib/admin/options.py | 725 | 742| 128 | 40981 | 140570 | 
| 137 | 57 django/shortcuts.py | 1 | 25| 161 | 41142 | 141694 | 
| 138 | 57 django/contrib/admin/options.py | 1678 | 1714| 337 | 41479 | 141694 | 
| 139 | 57 django/contrib/admin/checks.py | 704 | 740| 301 | 41780 | 141694 | 
| 140 | 58 django/template/defaulttags.py | 1325 | 1391| 508 | 42288 | 152467 | 
| 141 | 58 django/views/generic/dates.py | 235 | 280| 338 | 42626 | 152467 | 
| 142 | 58 django/contrib/admin/checks.py | 1090 | 1142| 430 | 43056 | 152467 | 
| 143 | 58 django/urls/resolvers.py | 499 | 528| 292 | 43348 | 152467 | 
| 144 | 58 django/shortcuts.py | 92 | 114| 208 | 43556 | 152467 | 
| 145 | 58 django/views/generic/base.py | 1 | 33| 170 | 43726 | 152467 | 
| 146 | 58 django/middleware/csrf.py | 296 | 346| 450 | 44176 | 152467 | 
| 147 | **58 django/contrib/admindocs/utils.py** | 124 | 165| 306 | 44482 | 152467 | 
| 148 | 59 django/contrib/flatpages/forms.py | 57 | 75| 142 | 44624 | 152967 | 
| 149 | 59 django/contrib/auth/views.py | 90 | 121| 216 | 44840 | 152967 | 
| 150 | 59 django/views/generic/dates.py | 124 | 168| 285 | 45125 | 152967 | 
| 151 | 59 django/contrib/admin/views/main.py | 455 | 493| 335 | 45460 | 152967 | 
| 152 | 60 django/core/checks/templates.py | 50 | 76| 167 | 45627 | 153449 | 
| 153 | 60 django/middleware/csrf.py | 470 | 483| 163 | 45790 | 153449 | 
| 154 | 61 django/contrib/flatpages/admin.py | 1 | 23| 148 | 45938 | 153597 | 
| 155 | 61 django/db/models/base.py | 2477 | 2529| 341 | 46279 | 153597 | 
| 156 | 62 django/contrib/admin/filters.py | 22 | 69| 311 | 46590 | 157866 | 
| 157 | 62 django/core/management/templates.py | 385 | 407| 200 | 46790 | 157866 | 
| 158 | 63 django/db/migrations/operations/models.py | 441 | 478| 267 | 47057 | 165865 | 
| 159 | 63 django/contrib/admin/options.py | 577 | 593| 198 | 47255 | 165865 | 
| 160 | 63 django/contrib/auth/admin.py | 149 | 214| 477 | 47732 | 165865 | 
| 161 | 64 django/http/request.py | 437 | 485| 338 | 48070 | 171427 | 
| 162 | 64 django/contrib/admin/options.py | 1060 | 1081| 153 | 48223 | 171427 | 
| 163 | 65 django/views/decorators/http.py | 1 | 59| 358 | 48581 | 172407 | 
| 164 | 65 django/views/generic/dates.py | 389 | 410| 140 | 48721 | 172407 | 
| 165 | 65 django/contrib/admin/options.py | 1140 | 1160| 201 | 48922 | 172407 | 
| 166 | 65 django/contrib/admin/options.py | 432 | 490| 509 | 49431 | 172407 | 
| 167 | 65 django/views/debug.py | 76 | 103| 178 | 49609 | 172407 | 
| 168 | 66 django/contrib/redirects/admin.py | 1 | 11| 0 | 49609 | 172476 | 
| 169 | 66 django/contrib/admin/options.py | 1321 | 1409| 689 | 50298 | 172476 | 
| 170 | 67 django/conf/urls/__init__.py | 1 | 10| 0 | 50298 | 172541 | 
| 171 | 67 django/contrib/flatpages/forms.py | 1 | 55| 364 | 50662 | 172541 | 
| 172 | 67 django/contrib/admindocs/views.py | 431 | 452| 132 | 50794 | 172541 | 
| 173 | 67 django/contrib/admin/templatetags/admin_list.py | 454 | 525| 372 | 51166 | 172541 | 
| 174 | 67 django/core/management/templates.py | 86 | 158| 606 | 51772 | 172541 | 
| 175 | 67 django/contrib/admin/checks.py | 1209 | 1227| 140 | 51912 | 172541 | 
| 176 | 67 django/views/static.py | 82 | 110| 169 | 52081 | 172541 | 
| 177 | 67 django/urls/resolvers.py | 440 | 453| 121 | 52202 | 172541 | 
| 178 | 67 django/contrib/admin/options.py | 2096 | 2171| 599 | 52801 | 172541 | 
| 179 | 68 django/contrib/flatpages/migrations/0001_initial.py | 1 | 69| 355 | 53156 | 172896 | 
| 180 | 69 django/db/models/__init__.py | 1 | 116| 682 | 53838 | 173578 | 
| 181 | 70 django/core/handlers/base.py | 174 | 226| 384 | 54222 | 176228 | 
| 182 | 70 django/views/generic/base.py | 182 | 217| 243 | 54465 | 176228 | 
| 183 | 71 django/urls/__init__.py | 1 | 54| 269 | 54734 | 176497 | 
| 184 | 71 docs/conf.py | 54 | 131| 711 | 55445 | 176497 | 
| 185 | 71 django/views/generic/list.py | 1 | 47| 333 | 55778 | 176497 | 


### Hint

```
Thanks for the ticket. It was originally reported in the #27409 among other issues "For views you can use generic class based views, in there by naming conventions can be capital letters. Class based views are classes and not methods and the naming of the view can be like BlogView or BlogEdit. But if generating link using the directives according to documentation it will not work since it will try to find view by name of blogview instead of BlogView. It will automatically put it to lowercase and may make it not working." but I think it deserves a separate ticket.
gonna look
Replying to Rahmat Faisal: gonna look Yeah if u have views for example class About(Views): ... and in the docs you write :view:`app_label.views.About` and if u acces, and the url gonna change to localhost:8000/admin/docs/app_label.views.about and return 404. the url should be localhost:8000/admin/docs/app_label.views.About
the problem from admindocs utils function ​https://github.com/django/django/blob/main/django/contrib/admindocs/utils.py#L59, that convert into lowercase
```

## Patch

```diff
diff --git a/django/contrib/admindocs/utils.py b/django/contrib/admindocs/utils.py
--- a/django/contrib/admindocs/utils.py
+++ b/django/contrib/admindocs/utils.py
@@ -101,6 +101,9 @@ def parse_rst(text, default_reference_context, thing_being_parsed=None):
 
 
 def create_reference_role(rolename, urlbase):
+    # Views and template names are case-sensitive.
+    is_case_sensitive = rolename in ["template", "view"]
+
     def _role(name, rawtext, text, lineno, inliner, options=None, content=None):
         if options is None:
             options = {}
@@ -111,7 +114,7 @@ def _role(name, rawtext, text, lineno, inliner, options=None, content=None):
                 urlbase
                 % (
                     inliner.document.settings.link_base,
-                    text.lower(),
+                    text if is_case_sensitive else text.lower(),
                 )
             ),
             **options,

```

## Test Patch

```diff
diff --git a/tests/admin_docs/test_utils.py b/tests/admin_docs/test_utils.py
--- a/tests/admin_docs/test_utils.py
+++ b/tests/admin_docs/test_utils.py
@@ -104,6 +104,22 @@ def test_parse_rst_with_docstring_no_leading_line_feed(self):
             self.assertEqual(parse_rst(body, ""), "<p>second line</p>\n")
         self.assertEqual(stderr.getvalue(), "")
 
+    def test_parse_rst_view_case_sensitive(self):
+        source = ":view:`myapp.views.Index`"
+        rendered = (
+            '<p><a class="reference external" '
+            'href="/admindocs/views/myapp.views.Index/">myapp.views.Index</a></p>'
+        )
+        self.assertHTMLEqual(parse_rst(source, "view"), rendered)
+
+    def test_parse_rst_template_case_sensitive(self):
+        source = ":template:`Index.html`"
+        rendered = (
+            '<p><a class="reference external" href="/admindocs/templates/Index.html/">'
+            "Index.html</a></p>"
+        )
+        self.assertHTMLEqual(parse_rst(source, "template"), rendered)
+
     def test_publish_parts(self):
         """
         Django shouldn't break the default role for interpreted text

```


## Code snippets

### 1 - django/contrib/admindocs/views.py:

Start line: 164, End line: 182

```python
class ViewDetailView(BaseAdminDocsView):
    template_name = "admin_doc/view_detail.html"

    @staticmethod
    def _get_view_func(view):
        urlconf = get_urlconf()
        if get_resolver(urlconf)._is_callback(view):
            mod, func = get_mod_func(view)
            try:
                # Separate the module and function, e.g.
                # 'mymodule.views.myview' -> 'mymodule.views', 'myview').
                return getattr(import_module(mod), func)
            except ImportError:
                # Import may fail because view contains a class name, e.g.
                # 'mymodule.views.ViewContainer.my_view', so mod takes the form
                # 'mymodule.views.ViewContainer'. Parse it again to separate
                # the module and class.
                mod, klass = get_mod_func(mod)
                return getattr(getattr(import_module(mod), klass), func)
```
### 2 - django/contrib/admindocs/views.py:

Start line: 1, End line: 36

```python
import inspect
from importlib import import_module
from inspect import cleandoc
from pathlib import Path

from django.apps import apps
from django.contrib import admin
from django.contrib.admin.views.decorators import staff_member_required
from django.contrib.admindocs import utils
from django.contrib.admindocs.utils import (
    remove_non_capturing_groups,
    replace_metacharacters,
    replace_named_groups,
    replace_unnamed_groups,
)
from django.core.exceptions import ImproperlyConfigured, ViewDoesNotExist
from django.db import models
from django.http import Http404
from django.template.engine import Engine
from django.urls import get_mod_func, get_resolver, get_urlconf
from django.utils._os import safe_join
from django.utils.decorators import method_decorator
from django.utils.functional import cached_property
from django.utils.inspect import (
    func_accepts_kwargs,
    func_accepts_var_args,
    get_func_full_args,
    method_has_no_args,
)
from django.utils.translation import gettext as _
from django.views.generic import TemplateView

from .utils import get_view_name

# Exclude methods starting with these strings from documentation
MODEL_METHODS_EXCLUDE = ("_", "add_", "delete", "save", "set_")
```
### 3 - django/contrib/admindocs/views.py:

Start line: 141, End line: 161

```python
class ViewIndexView(BaseAdminDocsView):
    template_name = "admin_doc/view_index.html"

    def get_context_data(self, **kwargs):
        views = []
        url_resolver = get_resolver(get_urlconf())
        try:
            view_functions = extract_views_from_urlpatterns(url_resolver.url_patterns)
        except ImproperlyConfigured:
            view_functions = []
        for func, regex, namespace, name in view_functions:
            views.append(
                {
                    "full_name": get_view_name(func),
                    "url": simplify_regex(regex),
                    "url_name": ":".join((namespace or []) + (name and [name] or [])),
                    "namespace": ":".join(namespace or []),
                    "name": name,
                }
            )
        return super().get_context_data(**{**kwargs, "views": views})
```
### 4 - django/contrib/admindocs/views.py:

Start line: 39, End line: 62

```python
class BaseAdminDocsView(TemplateView):
    """
    Base view for admindocs views.
    """

    @method_decorator(staff_member_required)
    def dispatch(self, request, *args, **kwargs):
        if not utils.docutils_is_available:
            # Display an error message for people without docutils
            self.template_name = "admin_doc/missing_docutils.html"
            return self.render_to_response(admin.site.each_context(request))
        return super().dispatch(request, *args, **kwargs)

    def get_context_data(self, **kwargs):
        return super().get_context_data(
            **{
                **kwargs,
                **admin.site.each_context(self.request),
            }
        )


class BookmarkletsView(BaseAdminDocsView):
    template_name = "admin_doc/bookmarklets.html"
```
### 5 - django/contrib/admindocs/views.py:

Start line: 184, End line: 210

```python
class ViewDetailView(BaseAdminDocsView):

    def get_context_data(self, **kwargs):
        view = self.kwargs["view"]
        view_func = self._get_view_func(view)
        if view_func is None:
            raise Http404
        title, body, metadata = utils.parse_docstring(view_func.__doc__)
        title = title and utils.parse_rst(title, "view", _("view:") + view)
        body = body and utils.parse_rst(body, "view", _("view:") + view)
        for key in metadata:
            metadata[key] = utils.parse_rst(metadata[key], "model", _("view:") + view)
        return super().get_context_data(
            **{
                **kwargs,
                "name": view,
                "summary": title,
                "body": body,
                "meta": metadata,
            }
        )


class ModelIndexView(BaseAdminDocsView):
    template_name = "admin_doc/model_index.html"

    def get_context_data(self, **kwargs):
        m_list = [m._meta for m in apps.get_models()]
        return super().get_context_data(**{**kwargs, "models": m_list})
```
### 6 - django/contrib/admindocs/views.py:

Start line: 395, End line: 428

```python
class TemplateDetailView(BaseAdminDocsView):
    template_name = "admin_doc/template_detail.html"

    def get_context_data(self, **kwargs):
        template = self.kwargs["template"]
        templates = []
        try:
            default_engine = Engine.get_default()
        except ImproperlyConfigured:
            # Non-trivial TEMPLATES settings aren't supported (#24125).
            pass
        else:
            # This doesn't account for template loaders (#24128).
            for index, directory in enumerate(default_engine.dirs):
                template_file = Path(safe_join(directory, template))
                if template_file.exists():
                    template_contents = template_file.read_text()
                else:
                    template_contents = ""
                templates.append(
                    {
                        "file": template_file,
                        "exists": template_file.exists(),
                        "contents": template_contents,
                        "order": index,
                    }
                )
        return super().get_context_data(
            **{
                **kwargs,
                "name": template,
                "templates": templates,
            }
        )
```
### 7 - django/contrib/admindocs/urls.py:

Start line: 1, End line: 51

```python
from django.contrib.admindocs import views
from django.urls import path, re_path

urlpatterns = [
    path(
        "",
        views.BaseAdminDocsView.as_view(template_name="admin_doc/index.html"),
        name="django-admindocs-docroot",
    ),
    path(
        "bookmarklets/",
        views.BookmarkletsView.as_view(),
        name="django-admindocs-bookmarklets",
    ),
    path(
        "tags/",
        views.TemplateTagIndexView.as_view(),
        name="django-admindocs-tags",
    ),
    path(
        "filters/",
        views.TemplateFilterIndexView.as_view(),
        name="django-admindocs-filters",
    ),
    path(
        "views/",
        views.ViewIndexView.as_view(),
        name="django-admindocs-views-index",
    ),
    path(
        "views/<view>/",
        views.ViewDetailView.as_view(),
        name="django-admindocs-views-detail",
    ),
    path(
        "models/",
        views.ModelIndexView.as_view(),
        name="django-admindocs-models-index",
    ),
    re_path(
        r"^models/(?P<app_label>[^\.]+)\.(?P<model_name>[^/]+)/$",
        views.ModelDetailView.as_view(),
        name="django-admindocs-models-detail",
    ),
    path(
        "templates/<path:template>/",
        views.TemplateDetailView.as_view(),
        name="django-admindocs-templates",
    ),
]
```
### 8 - django/views/generic/__init__.py:

Start line: 1, End line: 40

```python
from django.views.generic.base import RedirectView, TemplateView, View
from django.views.generic.dates import (
    ArchiveIndexView,
    DateDetailView,
    DayArchiveView,
    MonthArchiveView,
    TodayArchiveView,
    WeekArchiveView,
    YearArchiveView,
)
from django.views.generic.detail import DetailView
from django.views.generic.edit import CreateView, DeleteView, FormView, UpdateView
from django.views.generic.list import ListView

__all__ = [
    "View",
    "TemplateView",
    "RedirectView",
    "ArchiveIndexView",
    "YearArchiveView",
    "MonthArchiveView",
    "WeekArchiveView",
    "DayArchiveView",
    "TodayArchiveView",
    "DateDetailView",
    "DetailView",
    "FormView",
    "CreateView",
    "UpdateView",
    "DeleteView",
    "ListView",
    "GenericViewError",
]


class GenericViewError(Exception):
    """A problem in a generic view."""

    pass
```
### 9 - django/contrib/admindocs/views.py:

Start line: 102, End line: 138

```python
class TemplateFilterIndexView(BaseAdminDocsView):
    template_name = "admin_doc/template_filter_index.html"

    def get_context_data(self, **kwargs):
        filters = []
        try:
            engine = Engine.get_default()
        except ImproperlyConfigured:
            # Non-trivial TEMPLATES settings aren't supported (#24125).
            pass
        else:
            app_libs = sorted(engine.template_libraries.items())
            builtin_libs = [("", lib) for lib in engine.template_builtins]
            for module_name, library in builtin_libs + app_libs:
                for filter_name, filter_func in library.filters.items():
                    title, body, metadata = utils.parse_docstring(filter_func.__doc__)
                    title = title and utils.parse_rst(
                        title, "filter", _("filter:") + filter_name
                    )
                    body = body and utils.parse_rst(
                        body, "filter", _("filter:") + filter_name
                    )
                    for key in metadata:
                        metadata[key] = utils.parse_rst(
                            metadata[key], "filter", _("filter:") + filter_name
                        )
                    tag_library = module_name.split(".")[-1]
                    filters.append(
                        {
                            "name": filter_name,
                            "title": title,
                            "body": body,
                            "meta": metadata,
                            "library": tag_library,
                        }
                    )
        return super().get_context_data(**{**kwargs, "filters": filters})
```
### 10 - django/contrib/admindocs/utils.py:

Start line: 1, End line: 28

```python
"Misc. utility functions/classes for admin documentation generator."

import re
from email.errors import HeaderParseError
from email.parser import HeaderParser
from inspect import cleandoc

from django.urls import reverse
from django.utils.regex_helper import _lazy_re_compile
from django.utils.safestring import mark_safe

try:
    import docutils.core
    import docutils.nodes
    import docutils.parsers.rst.roles
except ImportError:
    docutils_is_available = False
else:
    docutils_is_available = True


def get_view_name(view_func):
    if hasattr(view_func, "view_class"):
        klass = view_func.view_class
        return f"{klass.__module__}.{klass.__qualname__}"
    mod_name = view_func.__module__
    view_name = getattr(view_func, "__qualname__", view_func.__class__.__name__)
    return mod_name + "." + view_name
```
### 45 - django/contrib/admindocs/utils.py:

Start line: 91, End line: 121

```python
#
# reST roles
#
ROLES = {
    "model": "%s/models/%s/",
    "view": "%s/views/%s/",
    "template": "%s/templates/%s/",
    "filter": "%s/filters/#%s",
    "tag": "%s/tags/#%s",
}


def create_reference_role(rolename, urlbase):
    def _role(name, rawtext, text, lineno, inliner, options=None, content=None):
        if options is None:
            options = {}
        node = docutils.nodes.reference(
            rawtext,
            text,
            refuri=(
                urlbase
                % (
                    inliner.document.settings.link_base,
                    text.lower(),
                )
            ),
            **options,
        )
        return [node], []

    docutils.parsers.rst.roles.register_canonical_role(rolename, _role)
```
### 147 - django/contrib/admindocs/utils.py:

Start line: 124, End line: 165

```python
def default_reference_role(
    name, rawtext, text, lineno, inliner, options=None, content=None
):
    if options is None:
        options = {}
    context = inliner.document.settings.default_reference_context
    node = docutils.nodes.reference(
        rawtext,
        text,
        refuri=(
            ROLES[context]
            % (
                inliner.document.settings.link_base,
                text.lower(),
            )
        ),
        **options,
    )
    return [node], []


if docutils_is_available:
    docutils.parsers.rst.roles.register_canonical_role(
        "cmsreference", default_reference_role
    )

    for name, urlbase in ROLES.items():
        create_reference_role(name, urlbase)

# Match the beginning of a named, unnamed, or non-capturing groups.
named_group_matcher = _lazy_re_compile(r"\(\?P(<\w+>)")
unnamed_group_matcher = _lazy_re_compile(r"\(")
non_capturing_group_matcher = _lazy_re_compile(r"\(\?\:")


def replace_metacharacters(pattern):
    """Remove unescaped metacharacters from the pattern."""
    return re.sub(
        r"((?:^|(?<!\\))(?:\\\\)*)(\\?)([?*+^$]|\\[bBAZ])",
        lambda m: m[1] + m[3] if m[2] else m[1],
        pattern,
    )
```
