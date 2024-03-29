# django__django-14634

| **django/django** | `37e8367c359cd115f109d82f99ff32be219f4928` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | 263 |
| **Avg pos** | 1.0 |
| **Min pos** | 1 |
| **Max pos** | 1 |
| **Top file pos** | 1 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/views/generic/edit.py b/django/views/generic/edit.py
--- a/django/views/generic/edit.py
+++ b/django/views/generic/edit.py
@@ -1,5 +1,5 @@
 from django.core.exceptions import ImproperlyConfigured
-from django.forms import models as model_forms
+from django.forms import Form, models as model_forms
 from django.http import HttpResponseRedirect
 from django.views.generic.base import ContextMixin, TemplateResponseMixin, View
 from django.views.generic.detail import (
@@ -225,12 +225,30 @@ def get_success_url(self):
                 "No URL to redirect to. Provide a success_url.")
 
 
-class BaseDeleteView(DeletionMixin, BaseDetailView):
+class BaseDeleteView(DeletionMixin, FormMixin, BaseDetailView):
     """
     Base view for deleting an object.
 
     Using this base class requires subclassing to provide a response mixin.
     """
+    form_class = Form
+
+    def post(self, request, *args, **kwargs):
+        # Set self.object before the usual form processing flow.
+        # Inlined because having DeletionMixin as the first base, for
+        # get_success_url(), makes leveraging super() with ProcessFormView
+        # overly complex.
+        self.object = self.get_object()
+        form = self.get_form()
+        if form.is_valid():
+            return self.form_valid(form)
+        else:
+            return self.form_invalid(form)
+
+    def form_valid(self, form):
+        success_url = self.get_success_url()
+        self.object.delete()
+        return HttpResponseRedirect(success_url)
 
 
 class DeleteView(SingleObjectTemplateResponseMixin, BaseDeleteView):

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/views/generic/edit.py | 2 | 2 | - | 1 | -
| django/views/generic/edit.py | 228 | 228 | 1 | 1 | 263


## Problem Statement

```
Allow delete to provide a success message through a mixin.
Description
	
Add a mixin to show a message on successful object deletion.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 django/views/generic/edit.py** | 202 | 242| 263 | 263 | 1716 | 
| 2 | 2 django/contrib/admin/options.py | 844 | 869| 211 | 474 | 20374 | 
| 3 | 3 django/contrib/admin/actions.py | 1 | 81| 616 | 1090 | 20990 | 
| 4 | 3 django/contrib/admin/options.py | 1849 | 1918| 590 | 1680 | 20990 | 
| 5 | 3 django/contrib/admin/options.py | 1471 | 1490| 133 | 1813 | 20990 | 
| 6 | 4 django/contrib/admin/utils.py | 123 | 158| 303 | 2116 | 25148 | 
| 7 | 5 django/db/models/deletion.py | 79 | 97| 199 | 2315 | 28976 | 
| 8 | 5 django/db/models/deletion.py | 99 | 121| 212 | 2527 | 28976 | 
| 9 | 5 django/db/models/deletion.py | 379 | 448| 580 | 3107 | 28976 | 
| 10 | 5 django/contrib/admin/utils.py | 495 | 553| 463 | 3570 | 28976 | 
| 11 | 6 django/core/checks/messages.py | 27 | 51| 259 | 3829 | 29553 | 
| 12 | 6 django/contrib/admin/options.py | 500 | 513| 165 | 3994 | 29553 | 
| 13 | 6 django/db/models/deletion.py | 214 | 268| 519 | 4513 | 29553 | 
| 14 | 7 django/contrib/admin/templatetags/admin_modify.py | 48 | 86| 391 | 4904 | 30521 | 
| 15 | 7 django/contrib/admin/options.py | 1975 | 2008| 345 | 5249 | 30521 | 
| 16 | 7 django/contrib/admin/options.py | 1430 | 1469| 309 | 5558 | 30521 | 
| 17 | 8 django/core/management/commands/makemessages.py | 363 | 400| 272 | 5830 | 36148 | 
| 18 | 9 django/utils/deprecation.py | 79 | 129| 372 | 6202 | 37187 | 
| 19 | 10 django/contrib/auth/views.py | 111 | 131| 173 | 6375 | 39917 | 
| 20 | 10 django/contrib/admin/options.py | 2101 | 2153| 451 | 6826 | 39917 | 
| 21 | 10 django/db/models/deletion.py | 269 | 344| 800 | 7626 | 39917 | 
| 22 | 10 django/contrib/admin/options.py | 1071 | 1094| 234 | 7860 | 39917 | 
| 23 | 11 django/contrib/auth/mixins.py | 1 | 42| 267 | 8127 | 40781 | 
| 24 | 11 django/core/management/commands/makemessages.py | 283 | 362| 814 | 8941 | 40781 | 
| 25 | 12 django/contrib/messages/middleware.py | 1 | 27| 174 | 9115 | 40956 | 
| 26 | 13 django/contrib/admin/sites.py | 342 | 362| 182 | 9297 | 45385 | 
| 27 | 13 django/db/models/deletion.py | 165 | 199| 325 | 9622 | 45385 | 
| 28 | 13 django/utils/deprecation.py | 131 | 149| 122 | 9744 | 45385 | 
| 29 | 14 django/views/generic/base.py | 121 | 154| 241 | 9985 | 47035 | 
| 30 | 15 django/contrib/messages/api.py | 1 | 34| 210 | 10195 | 47698 | 
| 31 | 15 django/db/models/deletion.py | 1 | 76| 566 | 10761 | 47698 | 
| 32 | 16 django/db/backends/mysql/compiler.py | 18 | 39| 219 | 10980 | 48292 | 
| 33 | 17 django/contrib/contenttypes/management/commands/remove_stale_contenttypes.py | 29 | 95| 517 | 11497 | 49012 | 
| 34 | 17 django/core/management/commands/makemessages.py | 426 | 456| 263 | 11760 | 49012 | 
| 35 | 18 django/utils/asyncio.py | 1 | 34| 212 | 11972 | 49225 | 
| 36 | 18 django/core/management/commands/makemessages.py | 117 | 152| 270 | 12242 | 49225 | 
| 37 | 18 django/contrib/admin/utils.py | 105 | 121| 135 | 12377 | 49225 | 
| 38 | 19 django/core/management/commands/flush.py | 27 | 83| 486 | 12863 | 49912 | 
| 39 | 19 django/core/checks/messages.py | 1 | 25| 160 | 13023 | 49912 | 
| 40 | 19 django/contrib/messages/api.py | 37 | 97| 453 | 13476 | 49912 | 
| 41 | 19 django/core/management/commands/makemessages.py | 170 | 194| 225 | 13701 | 49912 | 
| 42 | 20 django/db/backends/base/creation.py | 259 | 299| 342 | 14043 | 52700 | 
| 43 | 20 django/core/management/commands/makemessages.py | 605 | 644| 399 | 14442 | 52700 | 
| 44 | 20 django/contrib/admin/options.py | 1336 | 1361| 232 | 14674 | 52700 | 
| 45 | 20 django/core/management/commands/makemessages.py | 216 | 281| 633 | 15307 | 52700 | 
| 46 | 21 django/core/management/commands/makemigrations.py | 239 | 326| 873 | 16180 | 55539 | 
| 47 | 21 django/db/models/deletion.py | 361 | 377| 130 | 16310 | 55539 | 
| 48 | 22 django/core/mail/message.py | 177 | 185| 115 | 16425 | 59190 | 
| 49 | 22 django/core/management/commands/makemessages.py | 197 | 214| 176 | 16601 | 59190 | 
| 50 | 23 django/contrib/auth/migrations/0007_alter_validators_add_error_messages.py | 1 | 25| 137 | 16738 | 59327 | 
| 51 | 24 django/db/migrations/operations/models.py | 250 | 286| 254 | 16992 | 65777 | 
| 52 | 24 django/core/checks/messages.py | 54 | 77| 161 | 17153 | 65777 | 
| 53 | 24 django/core/mail/message.py | 338 | 353| 127 | 17280 | 65777 | 
| 54 | 25 django/db/models/sql/subqueries.py | 1 | 44| 320 | 17600 | 66978 | 
| 55 | 26 django/contrib/messages/storage/base.py | 1 | 41| 264 | 17864 | 68208 | 
| 56 | 27 django/db/models/base.py | 949 | 966| 177 | 18041 | 85528 | 
| 57 | 27 django/core/management/commands/makemessages.py | 402 | 424| 200 | 18241 | 85528 | 
| 58 | 28 django/forms/models.py | 451 | 478| 227 | 18468 | 97302 | 
| 59 | 28 django/contrib/admin/options.py | 1765 | 1847| 750 | 19218 | 97302 | 
| 60 | 28 django/core/management/commands/makemessages.py | 646 | 676| 284 | 19502 | 97302 | 
| 61 | 29 django/core/mail/__init__.py | 90 | 104| 175 | 19677 | 98422 | 
| 62 | 30 django/core/checks/security/sessions.py | 1 | 98| 572 | 20249 | 98995 | 
| 63 | 31 django/contrib/messages/storage/cookie.py | 1 | 25| 180 | 20429 | 100374 | 
| 64 | 32 django/core/exceptions.py | 107 | 218| 752 | 21181 | 101563 | 
| 65 | 33 django/views/debug.py | 189 | 201| 143 | 21324 | 106316 | 
| 66 | 34 django/db/backends/base/schema.py | 355 | 369| 141 | 21465 | 119139 | 
| 67 | 35 django/contrib/admin/decorators.py | 32 | 71| 268 | 21733 | 119782 | 
| 68 | 36 django/core/management/commands/squashmigrations.py | 136 | 204| 654 | 22387 | 121655 | 
| 69 | 37 django/contrib/postgres/aggregates/mixins.py | 22 | 34| 124 | 22511 | 122093 | 
| 70 | 38 django/core/management/commands/compilemessages.py | 59 | 116| 504 | 23015 | 123412 | 
| 71 | 39 django/contrib/postgres/utils.py | 1 | 30| 218 | 23233 | 123630 | 
| 72 | 40 django/db/models/sql/compiler.py | 1462 | 1494| 254 | 23487 | 138410 | 
| 73 | 40 django/views/debug.py | 150 | 162| 148 | 23635 | 138410 | 
| 74 | 41 django/forms/formsets.py | 228 | 243| 176 | 23811 | 142541 | 
| 75 | 41 django/core/management/commands/compilemessages.py | 30 | 57| 230 | 24041 | 142541 | 
| 76 | 42 django/contrib/auth/management/commands/createsuperuser.py | 230 | 245| 139 | 24180 | 144604 | 
| 77 | 42 django/db/models/base.py | 1178 | 1206| 213 | 24393 | 144604 | 
| 78 | 42 django/contrib/auth/mixins.py | 107 | 129| 146 | 24539 | 144604 | 
| 79 | 43 django/contrib/staticfiles/management/commands/collectstatic.py | 207 | 242| 248 | 24787 | 147422 | 
| 80 | 43 django/contrib/auth/mixins.py | 44 | 71| 235 | 25022 | 147422 | 
| 81 | 43 django/contrib/admin/options.py | 476 | 498| 241 | 25263 | 147422 | 
| 82 | 44 django/contrib/postgres/operations.py | 239 | 259| 163 | 25426 | 149808 | 
| 83 | 45 django/db/backends/oracle/creation.py | 167 | 185| 201 | 25627 | 153701 | 
| 84 | 46 django/db/models/query.py | 724 | 752| 261 | 25888 | 171239 | 
| 85 | 46 django/utils/deprecation.py | 36 | 76| 336 | 26224 | 171239 | 
| 86 | 46 django/db/backends/oracle/creation.py | 130 | 165| 399 | 26623 | 171239 | 
| 87 | 46 django/contrib/admin/options.py | 788 | 842| 414 | 27037 | 171239 | 
| 88 | 47 django/contrib/admin/models.py | 74 | 94| 161 | 27198 | 172362 | 
| 89 | 47 django/contrib/admin/options.py | 1637 | 1662| 291 | 27489 | 172362 | 
| 90 | 48 django/contrib/auth/middleware.py | 86 | 111| 192 | 27681 | 173367 | 
| 91 | 48 django/contrib/messages/storage/base.py | 133 | 149| 131 | 27812 | 173367 | 
| 92 | **48 django/views/generic/edit.py** | 70 | 101| 269 | 28081 | 173367 | 
| 93 | 48 django/forms/formsets.py | 330 | 383| 456 | 28537 | 173367 | 
| 94 | 49 django/contrib/messages/utils.py | 1 | 13| 0 | 28537 | 173417 | 
| 95 | 49 django/core/management/commands/makemessages.py | 1 | 34| 260 | 28797 | 173417 | 
| 96 | 50 django/core/files/utils.py | 27 | 79| 378 | 29175 | 173988 | 
| 97 | 51 django/forms/utils.py | 144 | 151| 132 | 29307 | 175289 | 
| 98 | 51 django/contrib/auth/views.py | 133 | 167| 269 | 29576 | 175289 | 
| 99 | 51 django/core/management/commands/flush.py | 1 | 25| 206 | 29782 | 175289 | 
| 100 | 51 django/db/models/query.py | 754 | 787| 274 | 30056 | 175289 | 
| 101 | 51 django/contrib/admin/decorators.py | 1 | 29| 181 | 30237 | 175289 | 
| 102 | 51 django/forms/models.py | 759 | 780| 194 | 30431 | 175289 | 
| 103 | 51 django/views/generic/base.py | 1 | 26| 142 | 30573 | 175289 | 
| 104 | 51 django/forms/utils.py | 79 | 142| 379 | 30952 | 175289 | 
| 105 | 51 django/core/management/commands/makemigrations.py | 61 | 152| 822 | 31774 | 175289 | 
| 106 | 51 django/contrib/admin/options.py | 1550 | 1636| 760 | 32534 | 175289 | 
| 107 | 52 django/contrib/messages/storage/__init__.py | 1 | 13| 0 | 32534 | 175359 | 
| 108 | 53 django/core/validators.py | 159 | 210| 518 | 33052 | 180058 | 
| 109 | 53 django/contrib/admin/options.py | 1137 | 1182| 482 | 33534 | 180058 | 
| 110 | 54 django/db/models/fields/mixins.py | 31 | 57| 173 | 33707 | 180401 | 
| 111 | 55 django/utils/datastructures.py | 221 | 255| 226 | 33933 | 182675 | 
| 112 | 55 django/utils/deprecation.py | 1 | 33| 209 | 34142 | 182675 | 
| 113 | 56 django/utils/decorators.py | 114 | 152| 316 | 34458 | 184074 | 
| 114 | 57 django/contrib/messages/views.py | 1 | 19| 0 | 34458 | 184170 | 
| 115 | 57 django/contrib/auth/views.py | 193 | 209| 122 | 34580 | 184170 | 
| 116 | 57 django/core/mail/__init__.py | 107 | 122| 180 | 34760 | 184170 | 
| 117 | 58 django/db/backends/oracle/schema.py | 44 | 58| 133 | 34893 | 186221 | 
| 118 | 59 django/db/migrations/autodetector.py | 804 | 854| 576 | 35469 | 197801 | 
| 119 | 59 django/core/management/commands/makemessages.py | 98 | 115| 139 | 35608 | 197801 | 
| 120 | 60 django/contrib/messages/__init__.py | 1 | 3| 0 | 35608 | 197825 | 
| 121 | 60 django/views/debug.py | 88 | 120| 258 | 35866 | 197825 | 
| 122 | 60 django/contrib/postgres/aggregates/mixins.py | 1 | 20| 182 | 36048 | 197825 | 


### Hint

```
The patch for this contribution
Your patch sets the message before the object is deleted. What if deleting the object fails? Otherwise, this addition makes sense.
Here is a pull request that approaches the problem from a different angle: ​https://github.com/django/django/pull/2585 Instead of adding a new mixin, the DeleteView is refactored to allow it to work with the existing SuccessMessageMixin. Thanks to @charettes for the approach and much of the code.
The approach seems sensible to me and the patch looks quite good. I'm going to mark this as ready for checkin so that we can get another set of eyes on this in case I've missed something. Thanks!
I see one small issue: the comment on line ​https://github.com/django/django/pull/2585/files#diff-2b2c9cb35ddf34bc38c90e322dcc39e8L201 still seems valid to me: the documented behaviour has changed, but I don't see a versionchanged annotation, which should be there in a case like this.
Bug #21926 was a duplicate of this one.
My main concern with the patch that I have provided is that there are changes related to using the DELETE method for deletions, but it doesn't work entirely as expected. For example, if an application uses a form to verify that a field should be deleted. If the view were to have code that checks that a confirmation box is checked or something similar, and they used the DELETE HTTP method, then the view would not work correctly, as data is not passed through with the request in the case of DELETE. So the documentation change where I changed it to read that you can use DELETE, and a few other changes in the code may not be valid. There was a comment in the pull request that got buried I think: "As I was updating the tests, I found that data cannot currently be sent with the DELETE method. When doing further research, I wasn't sure whether this should be allowed or not. The test client accepts a data parameter for DELETE, but the HTTP spec suggests that you shouldn't expect data, like you can for a POST: ​http://www.w3.org/Protocols/rfc2616/rfc2616-sec9.html#sec9.7 If we want to I can figure out how to actually get data through the chain. Otherwise I can update the documentation to reflect the changes instead." I wasn't sure how to proceed: to remove the parts related to supporting the DELETE method, or to try to figure out how to get the data through the DELETE chain.
I've updated the patch with feedback from review.
new pull request send with moving the conflicting tests.
Thanks for the new PR. For the record, this is ​https://github.com/django/django/pull/4256 I left a comment on GitHub regarding the code style issues.
Is there any status update on this?
We are waiting for someone to update the pull request as described in comment 13.
I will new PR
new PR ​https://github.com/django/django/pull/5992
re started working on it. will send the updated pr on master tomorrow
Just wanted to remind this is an important feature/fix. Thanks in advance
Thanks to all for their work on this. I did some minor changes to the original pull request that become inactive, and submitted a new pull request here ​https://github.com/django/django/pull/13362
Comments on PR. I think comment:9 is important: not clear adding the delete handling makes too much sense.
```

## Patch

```diff
diff --git a/django/views/generic/edit.py b/django/views/generic/edit.py
--- a/django/views/generic/edit.py
+++ b/django/views/generic/edit.py
@@ -1,5 +1,5 @@
 from django.core.exceptions import ImproperlyConfigured
-from django.forms import models as model_forms
+from django.forms import Form, models as model_forms
 from django.http import HttpResponseRedirect
 from django.views.generic.base import ContextMixin, TemplateResponseMixin, View
 from django.views.generic.detail import (
@@ -225,12 +225,30 @@ def get_success_url(self):
                 "No URL to redirect to. Provide a success_url.")
 
 
-class BaseDeleteView(DeletionMixin, BaseDetailView):
+class BaseDeleteView(DeletionMixin, FormMixin, BaseDetailView):
     """
     Base view for deleting an object.
 
     Using this base class requires subclassing to provide a response mixin.
     """
+    form_class = Form
+
+    def post(self, request, *args, **kwargs):
+        # Set self.object before the usual form processing flow.
+        # Inlined because having DeletionMixin as the first base, for
+        # get_success_url(), makes leveraging super() with ProcessFormView
+        # overly complex.
+        self.object = self.get_object()
+        form = self.get_form()
+        if form.is_valid():
+            return self.form_valid(form)
+        else:
+            return self.form_invalid(form)
+
+    def form_valid(self, form):
+        success_url = self.get_success_url()
+        self.object.delete()
+        return HttpResponseRedirect(success_url)
 
 
 class DeleteView(SingleObjectTemplateResponseMixin, BaseDeleteView):

```

## Test Patch

```diff
diff --git a/tests/generic_views/forms.py b/tests/generic_views/forms.py
--- a/tests/generic_views/forms.py
+++ b/tests/generic_views/forms.py
@@ -15,3 +15,12 @@ class Meta:
 class ContactForm(forms.Form):
     name = forms.CharField()
     message = forms.CharField(widget=forms.Textarea)
+
+
+class ConfirmDeleteForm(forms.Form):
+    confirm = forms.BooleanField()
+
+    def clean(self):
+        cleaned_data = super().clean()
+        if 'confirm' not in cleaned_data:
+            raise forms.ValidationError('You must confirm the delete.')
diff --git a/tests/generic_views/test_edit.py b/tests/generic_views/test_edit.py
--- a/tests/generic_views/test_edit.py
+++ b/tests/generic_views/test_edit.py
@@ -394,3 +394,35 @@ def test_delete_without_redirect(self):
         msg = 'No URL to redirect to. Provide a success_url.'
         with self.assertRaisesMessage(ImproperlyConfigured, msg):
             self.client.post('/edit/author/%d/delete/naive/' % self.author.pk)
+
+    def test_delete_with_form_as_post(self):
+        res = self.client.get('/edit/author/%d/delete/form/' % self.author.pk)
+        self.assertEqual(res.status_code, 200)
+        self.assertEqual(res.context['object'], self.author)
+        self.assertEqual(res.context['author'], self.author)
+        self.assertTemplateUsed(res, 'generic_views/author_confirm_delete.html')
+        res = self.client.post(
+            '/edit/author/%d/delete/form/' % self.author.pk, data={'confirm': True}
+        )
+        self.assertEqual(res.status_code, 302)
+        self.assertRedirects(res, '/list/authors/')
+        self.assertSequenceEqual(Author.objects.all(), [])
+
+    def test_delete_with_form_as_post_with_validation_error(self):
+        res = self.client.get('/edit/author/%d/delete/form/' % self.author.pk)
+        self.assertEqual(res.status_code, 200)
+        self.assertEqual(res.context['object'], self.author)
+        self.assertEqual(res.context['author'], self.author)
+        self.assertTemplateUsed(res, 'generic_views/author_confirm_delete.html')
+
+        res = self.client.post('/edit/author/%d/delete/form/' % self.author.pk)
+        self.assertEqual(res.status_code, 200)
+        self.assertEqual(len(res.context_data['form'].errors), 2)
+        self.assertEqual(
+            res.context_data['form'].errors['__all__'],
+            ['You must confirm the delete.'],
+        )
+        self.assertEqual(
+            res.context_data['form'].errors['confirm'],
+            ['This field is required.'],
+        )
diff --git a/tests/generic_views/urls.py b/tests/generic_views/urls.py
--- a/tests/generic_views/urls.py
+++ b/tests/generic_views/urls.py
@@ -101,6 +101,7 @@
     ),
     path('edit/author/<int:pk>/delete/', views.AuthorDelete.as_view()),
     path('edit/author/<int:pk>/delete/special/', views.SpecializedAuthorDelete.as_view()),
+    path('edit/author/<int:pk>/delete/form/', views.AuthorDeleteFormView.as_view()),
 
     # ArchiveIndexView
     path('dates/books/', views.BookArchive.as_view()),
diff --git a/tests/generic_views/views.py b/tests/generic_views/views.py
--- a/tests/generic_views/views.py
+++ b/tests/generic_views/views.py
@@ -4,7 +4,7 @@
 from django.utils.decorators import method_decorator
 from django.views import generic
 
-from .forms import AuthorForm, ContactForm
+from .forms import AuthorForm, ConfirmDeleteForm, ContactForm
 from .models import Artist, Author, Book, BookSigning, Page
 
 
@@ -179,6 +179,14 @@ class AuthorDelete(generic.DeleteView):
     success_url = '/list/authors/'
 
 
+class AuthorDeleteFormView(generic.DeleteView):
+    model = Author
+    form_class = ConfirmDeleteForm
+
+    def get_success_url(self):
+        return reverse('authors_list')
+
+
 class SpecializedAuthorDelete(generic.DeleteView):
     queryset = Author.objects.all()
     template_name = 'generic_views/confirm_delete.html'
diff --git a/tests/messages_tests/models.py b/tests/messages_tests/models.py
new file mode 100644
--- /dev/null
+++ b/tests/messages_tests/models.py
@@ -0,0 +1,5 @@
+from django.db import models
+
+
+class SomeObject(models.Model):
+    name = models.CharField(max_length=255)
diff --git a/tests/messages_tests/test_mixins.py b/tests/messages_tests/test_mixins.py
--- a/tests/messages_tests/test_mixins.py
+++ b/tests/messages_tests/test_mixins.py
@@ -1,12 +1,13 @@
 from django.core.signing import b64_decode
-from django.test import SimpleTestCase, override_settings
+from django.test import TestCase, override_settings
 from django.urls import reverse
 
-from .urls import ContactFormViewWithMsg
+from .models import SomeObject
+from .urls import ContactFormViewWithMsg, DeleteFormViewWithMsg
 
 
 @override_settings(ROOT_URLCONF='messages_tests.urls')
-class SuccessMessageMixinTests(SimpleTestCase):
+class SuccessMessageMixinTests(TestCase):
 
     def test_set_messages_success(self):
         author = {'name': 'John Doe', 'slug': 'success-msg'}
@@ -17,3 +18,9 @@ def test_set_messages_success(self):
             req.cookies['messages'].value.split(":")[0].encode(),
         ).decode()
         self.assertIn(ContactFormViewWithMsg.success_message % author, value)
+
+    def test_set_messages_success_on_delete(self):
+        object_to_delete = SomeObject.objects.create(name='MyObject')
+        delete_url = reverse('success_msg_on_delete', args=[object_to_delete.pk])
+        response = self.client.post(delete_url, follow=True)
+        self.assertContains(response, DeleteFormViewWithMsg.success_message)
diff --git a/tests/messages_tests/urls.py b/tests/messages_tests/urls.py
--- a/tests/messages_tests/urls.py
+++ b/tests/messages_tests/urls.py
@@ -6,7 +6,9 @@
 from django.template.response import TemplateResponse
 from django.urls import path, re_path, reverse
 from django.views.decorators.cache import never_cache
-from django.views.generic.edit import FormView
+from django.views.generic.edit import DeleteView, FormView
+
+from .models import SomeObject
 
 TEMPLATE = """{% if messages %}
 <ul class="messages">
@@ -63,9 +65,16 @@ class ContactFormViewWithMsg(SuccessMessageMixin, FormView):
     success_message = "%(name)s was created successfully"
 
 
+class DeleteFormViewWithMsg(SuccessMessageMixin, DeleteView):
+    model = SomeObject
+    success_url = '/show/'
+    success_message = 'Object was deleted successfully'
+
+
 urlpatterns = [
     re_path('^add/(debug|info|success|warning|error)/$', add, name='add_message'),
     path('add/msg/', ContactFormViewWithMsg.as_view(), name='add_success_msg'),
+    path('delete/msg/<int:pk>', DeleteFormViewWithMsg.as_view(), name='success_msg_on_delete'),
     path('show/', show, name='show_message'),
     re_path(
         '^template_response/add/(debug|info|success|warning|error)/$',

```


## Code snippets

### 1 - django/views/generic/edit.py:

Start line: 202, End line: 242

```python
class DeletionMixin:
    """Provide the ability to delete objects."""
    success_url = None

    def delete(self, request, *args, **kwargs):
        """
        Call the delete() method on the fetched object and then redirect to the
        success URL.
        """
        self.object = self.get_object()
        success_url = self.get_success_url()
        self.object.delete()
        return HttpResponseRedirect(success_url)

    # Add support for browsers which only accept GET and POST for now.
    def post(self, request, *args, **kwargs):
        return self.delete(request, *args, **kwargs)

    def get_success_url(self):
        if self.success_url:
            return self.success_url.format(**self.object.__dict__)
        else:
            raise ImproperlyConfigured(
                "No URL to redirect to. Provide a success_url.")


class BaseDeleteView(DeletionMixin, BaseDetailView):
    """
    Base view for deleting an object.

    Using this base class requires subclassing to provide a response mixin.
    """


class DeleteView(SingleObjectTemplateResponseMixin, BaseDeleteView):
    """
    View for deleting an object retrieved with self.get_object(), with a
    response rendered by a template.
    """
    template_name_suffix = '_confirm_delete'
```
### 2 - django/contrib/admin/options.py:

Start line: 844, End line: 869

```python
class ModelAdmin(BaseModelAdmin):

    def log_deletion(self, request, obj, object_repr):
        """
        Log that an object will be deleted. Note that this method must be
        called before the deletion.

        The default implementation creates an admin LogEntry object.
        """
        from django.contrib.admin.models import DELETION, LogEntry
        return LogEntry.objects.log_action(
            user_id=request.user.pk,
            content_type_id=get_content_type_for_model(obj).pk,
            object_id=obj.pk,
            object_repr=object_repr,
            action_flag=DELETION,
        )

    @display(description=mark_safe('<input type="checkbox" id="action-toggle">'))
    def action_checkbox(self, obj):
        """
        A list_display column containing a checkbox widget.
        """
        return helpers.checkbox.render(helpers.ACTION_CHECKBOX_NAME, str(obj.pk))

    @staticmethod
    def _get_action_description(func, name):
        return getattr(func, 'short_description', capfirst(name.replace('_', ' ')))
```
### 3 - django/contrib/admin/actions.py:

Start line: 1, End line: 81

```python
"""
Built-in, globally-available admin actions.
"""

from django.contrib import messages
from django.contrib.admin import helpers
from django.contrib.admin.decorators import action
from django.contrib.admin.utils import model_ngettext
from django.core.exceptions import PermissionDenied
from django.template.response import TemplateResponse
from django.utils.translation import gettext as _, gettext_lazy


@action(
    permissions=['delete'],
    description=gettext_lazy('Delete selected %(verbose_name_plural)s'),
)
def delete_selected(modeladmin, request, queryset):
    """
    Default action which deletes the selected objects.

    This action first displays a confirmation page which shows all the
    deletable objects, or, if the user has no permission one of the related
    childs (foreignkeys), a "permission denied" message.

    Next, it deletes all selected objects and redirects back to the change list.
    """
    opts = modeladmin.model._meta
    app_label = opts.app_label

    # Populate deletable_objects, a data structure of all related objects that
    # will also be deleted.
    deletable_objects, model_count, perms_needed, protected = modeladmin.get_deleted_objects(queryset, request)

    # The user has already confirmed the deletion.
    # Do the deletion and return None to display the change list view again.
    if request.POST.get('post') and not protected:
        if perms_needed:
            raise PermissionDenied
        n = queryset.count()
        if n:
            for obj in queryset:
                obj_display = str(obj)
                modeladmin.log_deletion(request, obj, obj_display)
            modeladmin.delete_queryset(request, queryset)
            modeladmin.message_user(request, _("Successfully deleted %(count)d %(items)s.") % {
                "count": n, "items": model_ngettext(modeladmin.opts, n)
            }, messages.SUCCESS)
        # Return None to display the change list page again.
        return None

    objects_name = model_ngettext(queryset)

    if perms_needed or protected:
        title = _("Cannot delete %(name)s") % {"name": objects_name}
    else:
        title = _("Are you sure?")

    context = {
        **modeladmin.admin_site.each_context(request),
        'title': title,
        'objects_name': str(objects_name),
        'deletable_objects': [deletable_objects],
        'model_count': dict(model_count).items(),
        'queryset': queryset,
        'perms_lacking': perms_needed,
        'protected': protected,
        'opts': opts,
        'action_checkbox_name': helpers.ACTION_CHECKBOX_NAME,
        'media': modeladmin.media,
    }

    request.current_app = modeladmin.admin_site.name

    # Display the confirmation page
    return TemplateResponse(request, modeladmin.delete_selected_confirmation_template or [
        "admin/%s/%s/delete_selected_confirmation.html" % (app_label, opts.model_name),
        "admin/%s/delete_selected_confirmation.html" % app_label,
        "admin/delete_selected_confirmation.html"
    ], context)
```
### 4 - django/contrib/admin/options.py:

Start line: 1849, End line: 1918

```python
class ModelAdmin(BaseModelAdmin):

    def get_deleted_objects(self, objs, request):
        """
        Hook for customizing the delete process for the delete view and the
        "delete selected" action.
        """
        return get_deleted_objects(objs, request, self.admin_site)

    @csrf_protect_m
    def delete_view(self, request, object_id, extra_context=None):
        with transaction.atomic(using=router.db_for_write(self.model)):
            return self._delete_view(request, object_id, extra_context)

    def _delete_view(self, request, object_id, extra_context):
        "The 'delete' admin view for this model."
        opts = self.model._meta
        app_label = opts.app_label

        to_field = request.POST.get(TO_FIELD_VAR, request.GET.get(TO_FIELD_VAR))
        if to_field and not self.to_field_allowed(request, to_field):
            raise DisallowedModelAdminToField("The field %s cannot be referenced." % to_field)

        obj = self.get_object(request, unquote(object_id), to_field)

        if not self.has_delete_permission(request, obj):
            raise PermissionDenied

        if obj is None:
            return self._get_obj_does_not_exist_redirect(request, opts, object_id)

        # Populate deleted_objects, a data structure of all related objects that
        # will also be deleted.
        deleted_objects, model_count, perms_needed, protected = self.get_deleted_objects([obj], request)

        if request.POST and not protected:  # The user has confirmed the deletion.
            if perms_needed:
                raise PermissionDenied
            obj_display = str(obj)
            attr = str(to_field) if to_field else opts.pk.attname
            obj_id = obj.serializable_value(attr)
            self.log_deletion(request, obj, obj_display)
            self.delete_model(request, obj)

            return self.response_delete(request, obj_display, obj_id)

        object_name = str(opts.verbose_name)

        if perms_needed or protected:
            title = _("Cannot delete %(name)s") % {"name": object_name}
        else:
            title = _("Are you sure?")

        context = {
            **self.admin_site.each_context(request),
            'title': title,
            'subtitle': None,
            'object_name': object_name,
            'object': obj,
            'deleted_objects': deleted_objects,
            'model_count': dict(model_count).items(),
            'perms_lacking': perms_needed,
            'protected': protected,
            'opts': opts,
            'app_label': app_label,
            'preserved_filters': self.get_preserved_filters(request),
            'is_popup': IS_POPUP_VAR in request.POST or IS_POPUP_VAR in request.GET,
            'to_field': to_field,
            **(extra_context or {}),
        }

        return self.render_delete_form(request, context)
```
### 5 - django/contrib/admin/options.py:

Start line: 1471, End line: 1490

```python
class ModelAdmin(BaseModelAdmin):

    def render_delete_form(self, request, context):
        opts = self.model._meta
        app_label = opts.app_label

        request.current_app = self.admin_site.name
        context.update(
            to_field_var=TO_FIELD_VAR,
            is_popup_var=IS_POPUP_VAR,
            media=self.media,
        )

        return TemplateResponse(
            request,
            self.delete_confirmation_template or [
                "admin/{}/{}/delete_confirmation.html".format(app_label, opts.model_name),
                "admin/{}/delete_confirmation.html".format(app_label),
                "admin/delete_confirmation.html",
            ],
            context,
        )
```
### 6 - django/contrib/admin/utils.py:

Start line: 123, End line: 158

```python
def get_deleted_objects(objs, request, admin_site):
    # ... other code

    def format_callback(obj):
        model = obj.__class__
        has_admin = model in admin_site._registry
        opts = obj._meta

        no_edit_link = '%s: %s' % (capfirst(opts.verbose_name), obj)

        if has_admin:
            if not admin_site._registry[model].has_delete_permission(request, obj):
                perms_needed.add(opts.verbose_name)
            try:
                admin_url = reverse('%s:%s_%s_change'
                                    % (admin_site.name,
                                       opts.app_label,
                                       opts.model_name),
                                    None, (quote(obj.pk),))
            except NoReverseMatch:
                # Change url doesn't exist -- don't display link to edit
                return no_edit_link

            # Display a link to the admin page.
            return format_html('{}: <a href="{}">{}</a>',
                               capfirst(opts.verbose_name),
                               admin_url,
                               obj)
        else:
            # Don't display link to edit, because it either has no
            # admin or is edited inline.
            return no_edit_link

    to_delete = collector.nested(format_callback)

    protected = [format_callback(obj) for obj in collector.protected]
    model_count = {model._meta.verbose_name_plural: len(objs) for model, objs in collector.model_objs.items()}

    return to_delete, model_count, perms_needed, protected
```
### 7 - django/db/models/deletion.py:

Start line: 79, End line: 97

```python
class Collector:
    def __init__(self, using):
        self.using = using
        # Initially, {model: {instances}}, later values become lists.
        self.data = defaultdict(set)
        # {model: {(field, value): {instances}}}
        self.field_updates = defaultdict(partial(defaultdict, set))
        # {model: {field: {instances}}}
        self.restricted_objects = defaultdict(partial(defaultdict, set))
        # fast_deletes is a list of queryset-likes that can be deleted without
        # fetching the objects into memory.
        self.fast_deletes = []

        # Tracks deletion-order dependency for databases without transactions
        # or ability to defer constraint checks. Only concrete model classes
        # should be included, as the dependencies exist only between actual
        # database tables; proxy models are represented here by their concrete
        # parent.
        self.dependencies = defaultdict(set)  # {model: {models}}
```
### 8 - django/db/models/deletion.py:

Start line: 99, End line: 121

```python
class Collector:

    def add(self, objs, source=None, nullable=False, reverse_dependency=False):
        """
        Add 'objs' to the collection of objects to be deleted.  If the call is
        the result of a cascade, 'source' should be the model that caused it,
        and 'nullable' should be set to True if the relation can be null.

        Return a list of all objects that were not already collected.
        """
        if not objs:
            return []
        new_objs = []
        model = objs[0].__class__
        instances = self.data[model]
        for obj in objs:
            if obj not in instances:
                new_objs.append(obj)
        instances.update(new_objs)
        # Nullable relationships can be ignored -- they are nulled out before
        # deleting, and therefore do not affect the order in which objects have
        # to be deleted.
        if source is not None and not nullable:
            self.add_dependency(source, model, reverse_dependency=reverse_dependency)
        return new_objs
```
### 9 - django/db/models/deletion.py:

Start line: 379, End line: 448

```python
class Collector:

    def delete(self):
        # sort instance collections
        for model, instances in self.data.items():
            self.data[model] = sorted(instances, key=attrgetter("pk"))

        # if possible, bring the models in an order suitable for databases that
        # don't support transactions or cannot defer constraint checks until the
        # end of a transaction.
        self.sort()
        # number of objects deleted for each model label
        deleted_counter = Counter()

        # Optimize for the case with a single obj and no dependencies
        if len(self.data) == 1 and len(instances) == 1:
            instance = list(instances)[0]
            if self.can_fast_delete(instance):
                with transaction.mark_for_rollback_on_error(self.using):
                    count = sql.DeleteQuery(model).delete_batch([instance.pk], self.using)
                setattr(instance, model._meta.pk.attname, None)
                return count, {model._meta.label: count}

        with transaction.atomic(using=self.using, savepoint=False):
            # send pre_delete signals
            for model, obj in self.instances_with_model():
                if not model._meta.auto_created:
                    signals.pre_delete.send(
                        sender=model, instance=obj, using=self.using
                    )

            # fast deletes
            for qs in self.fast_deletes:
                count = qs._raw_delete(using=self.using)
                if count:
                    deleted_counter[qs.model._meta.label] += count

            # update fields
            for model, instances_for_fieldvalues in self.field_updates.items():
                for (field, value), instances in instances_for_fieldvalues.items():
                    query = sql.UpdateQuery(model)
                    query.update_batch([obj.pk for obj in instances],
                                       {field.name: value}, self.using)

            # reverse instance collections
            for instances in self.data.values():
                instances.reverse()

            # delete instances
            for model, instances in self.data.items():
                query = sql.DeleteQuery(model)
                pk_list = [obj.pk for obj in instances]
                count = query.delete_batch(pk_list, self.using)
                if count:
                    deleted_counter[model._meta.label] += count

                if not model._meta.auto_created:
                    for obj in instances:
                        signals.post_delete.send(
                            sender=model, instance=obj, using=self.using
                        )

        # update collected instances
        for instances_for_fieldvalues in self.field_updates.values():
            for (field, value), instances in instances_for_fieldvalues.items():
                for obj in instances:
                    setattr(obj, field.attname, value)
        for model, instances in self.data.items():
            for instance in instances:
                setattr(instance, model._meta.pk.attname, None)
        return sum(deleted_counter.values()), dict(deleted_counter)
```
### 10 - django/contrib/admin/utils.py:

Start line: 495, End line: 553

```python
def construct_change_message(form, formsets, add):
    """
    Construct a JSON structure describing changes from a changed object.
    Translations are deactivated so that strings are stored untranslated.
    Translation happens later on LogEntry access.
    """
    # Evaluating `form.changed_data` prior to disabling translations is required
    # to avoid fields affected by localization from being included incorrectly,
    # e.g. where date formats differ such as MM/DD/YYYY vs DD/MM/YYYY.
    changed_data = form.changed_data
    with translation_override(None):
        # Deactivate translations while fetching verbose_name for form
        # field labels and using `field_name`, if verbose_name is not provided.
        # Translations will happen later on LogEntry access.
        changed_field_labels = _get_changed_field_labels_from_form(form, changed_data)

    change_message = []
    if add:
        change_message.append({'added': {}})
    elif form.changed_data:
        change_message.append({'changed': {'fields': changed_field_labels}})
    if formsets:
        with translation_override(None):
            for formset in formsets:
                for added_object in formset.new_objects:
                    change_message.append({
                        'added': {
                            'name': str(added_object._meta.verbose_name),
                            'object': str(added_object),
                        }
                    })
                for changed_object, changed_fields in formset.changed_objects:
                    change_message.append({
                        'changed': {
                            'name': str(changed_object._meta.verbose_name),
                            'object': str(changed_object),
                            'fields': _get_changed_field_labels_from_form(formset.forms[0], changed_fields),
                        }
                    })
                for deleted_object in formset.deleted_objects:
                    change_message.append({
                        'deleted': {
                            'name': str(deleted_object._meta.verbose_name),
                            'object': str(deleted_object),
                        }
                    })
    return change_message


def _get_changed_field_labels_from_form(form, changed_data):
    changed_field_labels = []
    for field_name in changed_data:
        try:
            verbose_field_name = form.fields[field_name].label or field_name
        except KeyError:
            verbose_field_name = field_name
        changed_field_labels.append(str(verbose_field_name))
    return changed_field_labels
```
### 92 - django/views/generic/edit.py:

Start line: 70, End line: 101

```python
class ModelFormMixin(FormMixin, SingleObjectMixin):
    """Provide a way to show and handle a ModelForm in a request."""
    fields = None

    def get_form_class(self):
        """Return the form class to use in this view."""
        if self.fields is not None and self.form_class:
            raise ImproperlyConfigured(
                "Specifying both 'fields' and 'form_class' is not permitted."
            )
        if self.form_class:
            return self.form_class
        else:
            if self.model is not None:
                # If a model has been explicitly provided, use it
                model = self.model
            elif getattr(self, 'object', None) is not None:
                # If this view is operating on a single object, use
                # the class of that object
                model = self.object.__class__
            else:
                # Try to get a queryset and extract the model class
                # from that
                model = self.get_queryset().model

            if self.fields is None:
                raise ImproperlyConfigured(
                    "Using ModelFormMixin (base class of %s) without "
                    "the 'fields' attribute is prohibited." % self.__class__.__name__
                )

            return model_forms.modelform_factory(model, fields=self.fields)
```
