# django__django-13233

| **django/django** | `41065cfed56d5408dd8f267b9e70089471a7f1be` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | - |
| **Avg pos** | - |
| **Min pos** | - |
| **Max pos** | - |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/db/models/fields/files.py b/django/db/models/fields/files.py
--- a/django/db/models/fields/files.py
+++ b/django/db/models/fields/files.py
@@ -299,6 +299,10 @@ def pre_save(self, model_instance, add):
             file.save(file.name, file.file, save=False)
         return file
 
+    def contribute_to_class(self, cls, name, **kwargs):
+        super().contribute_to_class(cls, name, **kwargs)
+        setattr(cls, self.attname, self.descriptor_class(self))
+
     def generate_filename(self, instance, filename):
         """
         Apply (if callable) or prepend (if a string) upload_to to the filename,

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/db/models/fields/files.py | 302 | 302 | - | 1 | -


## Problem Statement

```
The `model` attribute of image fields doesn't point to concrete model.
Description
	
In Django 3.1 and before, one could use the model attribute of image fields to find the concrete model the image field belongs to.
This isn't possible in 3.2 anymore, and I bisected the change to the fix of #31701.
I found this while investigating a CI failure of django-imagefield ​https://travis-ci.org/github/matthiask/django-imagefield/jobs/710794644
I'm not sure whether this is a bug or whether it is an intentional change. If it is the later, is there an alternative to find the concrete model an image field belongs to? I'm classifying this as a bug because the change made model and field introspection harder than it was before. Also, since behavior changed #31701 may possibly not be classified as a cleanup/optimization anymore...

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 django/db/models/fields/files.py** | 364 | 410| 373 | 373 | 3739 | 
| 2 | 2 django/db/models/base.py | 954 | 968| 212 | 585 | 20320 | 
| 3 | 2 django/db/models/base.py | 1654 | 1702| 348 | 933 | 20320 | 
| 4 | **2 django/db/models/fields/files.py** | 144 | 213| 711 | 1644 | 20320 | 
| 5 | **2 django/db/models/fields/files.py** | 334 | 361| 277 | 1921 | 20320 | 
| 6 | 3 django/db/models/options.py | 607 | 625| 178 | 2099 | 27426 | 
| 7 | 4 django/db/models/fields/related.py | 171 | 188| 166 | 2265 | 41302 | 
| 8 | 4 django/db/models/fields/related.py | 1235 | 1352| 963 | 3228 | 41302 | 
| 9 | 4 django/db/models/options.py | 332 | 355| 198 | 3426 | 41302 | 
| 10 | 4 django/db/models/base.py | 404 | 503| 856 | 4282 | 41302 | 
| 11 | 4 django/db/models/base.py | 1147 | 1162| 138 | 4420 | 41302 | 
| 12 | 4 django/db/models/fields/related.py | 156 | 169| 144 | 4564 | 41302 | 
| 13 | 5 django/contrib/contenttypes/fields.py | 110 | 158| 328 | 4892 | 46735 | 
| 14 | 5 django/db/models/fields/related.py | 284 | 318| 293 | 5185 | 46735 | 
| 15 | 5 django/db/models/base.py | 1345 | 1375| 244 | 5429 | 46735 | 
| 16 | 6 django/db/models/fields/__init__.py | 495 | 506| 184 | 5613 | 64424 | 
| 17 | 6 django/db/models/base.py | 1476 | 1498| 171 | 5784 | 64424 | 
| 18 | 6 django/db/models/base.py | 1118 | 1145| 286 | 6070 | 64424 | 
| 19 | **6 django/db/models/fields/files.py** | 1 | 141| 940 | 7010 | 64424 | 
| 20 | **6 django/db/models/fields/files.py** | 412 | 474| 551 | 7561 | 64424 | 
| 21 | 7 django/contrib/admin/utils.py | 261 | 284| 174 | 7735 | 68576 | 
| 22 | 8 django/db/migrations/autodetector.py | 806 | 855| 567 | 8302 | 80195 | 
| 23 | 8 django/db/models/base.py | 169 | 211| 413 | 8715 | 80195 | 
| 24 | 8 django/db/models/base.py | 1255 | 1285| 259 | 8974 | 80195 | 
| 25 | 8 django/db/models/base.py | 970 | 983| 180 | 9154 | 80195 | 
| 26 | 8 django/db/models/fields/related.py | 710 | 748| 335 | 9489 | 80195 | 
| 27 | 9 django/db/backends/base/schema.py | 31 | 41| 120 | 9609 | 92037 | 
| 28 | 9 django/db/models/fields/related.py | 935 | 948| 126 | 9735 | 92037 | 
| 29 | 9 django/db/models/fields/related.py | 1354 | 1426| 616 | 10351 | 92037 | 
| 30 | 10 django/db/migrations/state.py | 348 | 395| 428 | 10779 | 97159 | 
| 31 | 11 django/contrib/admin/options.py | 431 | 474| 350 | 11129 | 115728 | 
| 32 | 11 django/db/models/base.py | 650 | 665| 153 | 11282 | 115728 | 
| 33 | 11 django/db/models/fields/related.py | 1513 | 1537| 295 | 11577 | 115728 | 
| 34 | 12 django/db/migrations/questioner.py | 187 | 205| 237 | 11814 | 117801 | 
| 35 | 12 django/db/models/base.py | 1164 | 1192| 213 | 12027 | 117801 | 
| 36 | 12 django/db/models/base.py | 898 | 923| 314 | 12341 | 117801 | 
| 37 | 12 django/db/models/fields/related.py | 255 | 282| 269 | 12610 | 117801 | 
| 38 | 12 django/db/models/base.py | 505 | 547| 348 | 12958 | 117801 | 
| 39 | 12 django/db/models/fields/related.py | 1 | 34| 246 | 13204 | 117801 | 
| 40 | 12 django/db/models/fields/__init__.py | 178 | 206| 253 | 13457 | 117801 | 
| 41 | 12 django/db/models/fields/related.py | 1606 | 1643| 484 | 13941 | 117801 | 
| 42 | 12 django/db/models/fields/__init__.py | 367 | 393| 199 | 14140 | 117801 | 
| 43 | 12 django/db/models/options.py | 747 | 829| 827 | 14967 | 117801 | 
| 44 | 13 django/core/checks/model_checks.py | 129 | 153| 268 | 15235 | 119586 | 
| 45 | 14 django/db/models/fields/mixins.py | 1 | 28| 168 | 15403 | 119929 | 
| 46 | 15 django/forms/models.py | 991 | 1044| 470 | 15873 | 131703 | 
| 47 | 15 django/db/models/base.py | 212 | 322| 866 | 16739 | 131703 | 
| 48 | 15 django/db/models/base.py | 1500 | 1532| 231 | 16970 | 131703 | 
| 49 | 16 django/db/migrations/operations/fields.py | 273 | 299| 158 | 17128 | 134801 | 
| 50 | 17 django/db/backends/postgresql/introspection.py | 1 | 43| 336 | 17464 | 137049 | 
| 51 | 17 django/db/models/base.py | 1287 | 1312| 184 | 17648 | 137049 | 
| 52 | 18 django/contrib/gis/db/models/fields.py | 239 | 250| 148 | 17796 | 140100 | 
| 53 | 18 django/db/models/base.py | 1377 | 1392| 153 | 17949 | 140100 | 
| 54 | 18 django/db/models/base.py | 549 | 566| 142 | 18091 | 140100 | 
| 55 | 18 django/db/models/fields/related.py | 1645 | 1661| 286 | 18377 | 140100 | 
| 56 | 18 django/db/models/fields/related.py | 127 | 154| 201 | 18578 | 140100 | 
| 57 | 18 django/contrib/admin/options.py | 1540 | 1626| 760 | 19338 | 140100 | 
| 58 | 18 django/db/models/base.py | 2031 | 2082| 351 | 19689 | 140100 | 
| 59 | 18 django/db/models/fields/related.py | 1471 | 1511| 399 | 20088 | 140100 | 
| 60 | 18 django/db/models/fields/__init__.py | 1928 | 1955| 234 | 20322 | 140100 | 
| 61 | 18 django/db/models/base.py | 1897 | 2028| 976 | 21298 | 140100 | 
| 62 | 18 django/db/models/base.py | 1073 | 1116| 404 | 21702 | 140100 | 
| 63 | 18 django/db/models/base.py | 385 | 401| 128 | 21830 | 140100 | 
| 64 | 19 django/db/models/fields/reverse_related.py | 156 | 181| 269 | 22099 | 142243 | 
| 65 | 19 django/db/models/base.py | 1394 | 1449| 491 | 22590 | 142243 | 
| 66 | 20 django/db/backends/sqlite3/schema.py | 330 | 346| 173 | 22763 | 146359 | 
| 67 | 20 django/db/models/fields/related.py | 892 | 911| 145 | 22908 | 146359 | 
| 68 | 21 django/db/migrations/operations/models.py | 339 | 388| 493 | 23401 | 153254 | 
| 69 | 21 django/db/backends/base/schema.py | 386 | 405| 197 | 23598 | 153254 | 
| 70 | 21 django/db/migrations/autodetector.py | 463 | 507| 424 | 24022 | 153254 | 
| 71 | 21 django/db/models/fields/__init__.py | 1638 | 1652| 144 | 24166 | 153254 | 
| 72 | 21 django/db/backends/base/schema.py | 503 | 531| 289 | 24455 | 153254 | 
| 73 | 22 django/core/serializers/xml_serializer.py | 317 | 332| 144 | 24599 | 156766 | 
| 74 | 23 django/db/models/fields/related_lookups.py | 102 | 117| 215 | 24814 | 158215 | 
| 75 | 23 django/contrib/contenttypes/fields.py | 160 | 171| 123 | 24937 | 158215 | 
| 76 | 24 django/contrib/gis/db/backends/spatialite/schema.py | 84 | 101| 153 | 25090 | 159567 | 
| 77 | 24 django/db/backends/sqlite3/schema.py | 384 | 430| 444 | 25534 | 159567 | 
| 78 | 24 django/db/models/fields/__init__.py | 2432 | 2457| 143 | 25677 | 159567 | 
| 79 | 25 django/db/migrations/operations/utils.py | 1 | 32| 220 | 25897 | 160311 | 
| 80 | 25 django/db/models/fields/related.py | 576 | 609| 334 | 26231 | 160311 | 
| 81 | 25 django/core/checks/model_checks.py | 155 | 176| 263 | 26494 | 160311 | 
| 82 | 26 django/db/models/fields/related_descriptors.py | 344 | 363| 156 | 26650 | 170706 | 
| 83 | 26 django/contrib/gis/db/backends/spatialite/schema.py | 63 | 82| 133 | 26783 | 170706 | 
| 84 | 26 django/forms/models.py | 208 | 277| 616 | 27399 | 170706 | 
| 85 | 27 django/db/backends/oracle/introspection.py | 1 | 49| 431 | 27830 | 173357 | 
| 86 | 28 django/contrib/admin/views/main.py | 265 | 295| 270 | 28100 | 177756 | 
| 87 | 28 django/db/migrations/questioner.py | 56 | 81| 220 | 28320 | 177756 | 
| 88 | 28 django/db/models/fields/related_lookups.py | 46 | 60| 224 | 28544 | 177756 | 
| 89 | 28 django/db/backends/base/schema.py | 896 | 915| 296 | 28840 | 177756 | 
| 90 | 28 django/db/models/fields/__init__.py | 936 | 961| 209 | 29049 | 177756 | 
| 91 | 28 django/db/migrations/state.py | 56 | 75| 209 | 29258 | 177756 | 
| 92 | 29 django/db/backends/sqlite3/introspection.py | 23 | 54| 296 | 29554 | 181605 | 
| 93 | 29 django/db/models/fields/__init__.py | 1 | 81| 633 | 30187 | 181605 | 
| 94 | 29 django/db/models/fields/related.py | 108 | 125| 155 | 30342 | 181605 | 
| 95 | 30 django/contrib/gis/db/backends/mysql/schema.py | 40 | 63| 190 | 30532 | 182236 | 
| 96 | 30 django/db/migrations/operations/models.py | 312 | 337| 290 | 30822 | 182236 | 
| 97 | 31 django/db/models/query_utils.py | 25 | 54| 185 | 31007 | 184942 | 
| 98 | 31 django/db/models/query_utils.py | 284 | 309| 293 | 31300 | 184942 | 
| 99 | 31 django/db/backends/sqlite3/schema.py | 140 | 221| 820 | 32120 | 184942 | 


## Patch

```diff
diff --git a/django/db/models/fields/files.py b/django/db/models/fields/files.py
--- a/django/db/models/fields/files.py
+++ b/django/db/models/fields/files.py
@@ -299,6 +299,10 @@ def pre_save(self, model_instance, add):
             file.save(file.name, file.file, save=False)
         return file
 
+    def contribute_to_class(self, cls, name, **kwargs):
+        super().contribute_to_class(cls, name, **kwargs)
+        setattr(cls, self.attname, self.descriptor_class(self))
+
     def generate_filename(self, instance, filename):
         """
         Apply (if callable) or prepend (if a string) upload_to to the filename,

```

## Test Patch

```diff
diff --git a/tests/model_fields/test_filefield.py b/tests/model_fields/test_filefield.py
--- a/tests/model_fields/test_filefield.py
+++ b/tests/model_fields/test_filefield.py
@@ -8,8 +8,9 @@
 from django.core.files import File, temp
 from django.core.files.base import ContentFile
 from django.core.files.uploadedfile import TemporaryUploadedFile
-from django.db import IntegrityError
+from django.db import IntegrityError, models
 from django.test import TestCase, override_settings
+from django.test.utils import isolate_apps
 
 from .models import Document
 
@@ -147,3 +148,21 @@ def test_pickle(self):
                         self.assertEqual(document.myfile.field, loaded_myfile.field)
                     finally:
                         document.myfile.delete()
+
+    @isolate_apps('model_fields')
+    def test_abstract_filefield_model(self):
+        """
+        FileField.model returns the concrete model for fields defined in an
+        abstract model.
+        """
+        class AbstractMyDocument(models.Model):
+            myfile = models.FileField(upload_to='unused')
+
+            class Meta:
+                abstract = True
+
+        class MyDocument(AbstractMyDocument):
+            pass
+
+        document = MyDocument(myfile='test_file.py')
+        self.assertEqual(document.myfile.field.model, MyDocument)

```


## Code snippets

### 1 - django/db/models/fields/files.py:

Start line: 364, End line: 410

```python
class ImageField(FileField):
    attr_class = ImageFieldFile
    descriptor_class = ImageFileDescriptor
    description = _("Image")

    def __init__(self, verbose_name=None, name=None, width_field=None, height_field=None, **kwargs):
        self.width_field, self.height_field = width_field, height_field
        super().__init__(verbose_name, name, **kwargs)

    def check(self, **kwargs):
        return [
            *super().check(**kwargs),
            *self._check_image_library_installed(),
        ]

    def _check_image_library_installed(self):
        try:
            from PIL import Image  # NOQA
        except ImportError:
            return [
                checks.Error(
                    'Cannot use ImageField because Pillow is not installed.',
                    hint=('Get Pillow at https://pypi.org/project/Pillow/ '
                          'or run command "python -m pip install Pillow".'),
                    obj=self,
                    id='fields.E210',
                )
            ]
        else:
            return []

    def deconstruct(self):
        name, path, args, kwargs = super().deconstruct()
        if self.width_field:
            kwargs['width_field'] = self.width_field
        if self.height_field:
            kwargs['height_field'] = self.height_field
        return name, path, args, kwargs

    def contribute_to_class(self, cls, name, **kwargs):
        super().contribute_to_class(cls, name, **kwargs)
        # Attach update_dimension_fields so that dimension fields declared
        # after their corresponding image field don't stay cleared by
        # Model.__init__, see bug #11196.
        # Only run post-initialization dimension update on non-abstract models
        if not cls._meta.abstract:
            signals.post_init.connect(self.update_dimension_fields, sender=cls)
```
### 2 - django/db/models/base.py:

Start line: 954, End line: 968

```python
class Model(metaclass=ModelBase):

    def _get_next_or_previous_by_FIELD(self, field, is_next, **kwargs):
        if not self.pk:
            raise ValueError("get_next/get_previous cannot be used on unsaved objects.")
        op = 'gt' if is_next else 'lt'
        order = '' if is_next else '-'
        param = getattr(self, field.attname)
        q = Q(**{'%s__%s' % (field.name, op): param})
        q = q | Q(**{field.name: param, 'pk__%s' % op: self.pk})
        qs = self.__class__._default_manager.using(self._state.db).filter(**kwargs).filter(q).order_by(
            '%s%s' % (order, field.name), '%spk' % order
        )
        try:
            return qs[0]
        except IndexError:
            raise self.DoesNotExist("%s matching query does not exist." % self.__class__._meta.object_name)
```
### 3 - django/db/models/base.py:

Start line: 1654, End line: 1702

```python
class Model(metaclass=ModelBase):

    @classmethod
    def _check_local_fields(cls, fields, option):
        from django.db import models

        # In order to avoid hitting the relation tree prematurely, we use our
        # own fields_map instead of using get_field()
        forward_fields_map = {}
        for field in cls._meta._get_fields(reverse=False):
            forward_fields_map[field.name] = field
            if hasattr(field, 'attname'):
                forward_fields_map[field.attname] = field

        errors = []
        for field_name in fields:
            try:
                field = forward_fields_map[field_name]
            except KeyError:
                errors.append(
                    checks.Error(
                        "'%s' refers to the nonexistent field '%s'." % (
                            option, field_name,
                        ),
                        obj=cls,
                        id='models.E012',
                    )
                )
            else:
                if isinstance(field.remote_field, models.ManyToManyRel):
                    errors.append(
                        checks.Error(
                            "'%s' refers to a ManyToManyField '%s', but "
                            "ManyToManyFields are not permitted in '%s'." % (
                                option, field_name, option,
                            ),
                            obj=cls,
                            id='models.E013',
                        )
                    )
                elif field not in cls._meta.local_fields:
                    errors.append(
                        checks.Error(
                            "'%s' refers to field '%s' which is not local to model '%s'."
                            % (option, field_name, cls._meta.object_name),
                            hint="This issue may be caused by multi-table inheritance.",
                            obj=cls,
                            id='models.E016',
                        )
                    )
        return errors
```
### 4 - django/db/models/fields/files.py:

Start line: 144, End line: 213

```python
class FileDescriptor(DeferredAttribute):
    """
    The descriptor for the file attribute on the model instance. Return a
    FieldFile when accessed so you can write code like::

        >>> from myapp.models import MyModel
        >>> instance = MyModel.objects.get(pk=1)
        >>> instance.file.size

    Assign a file object on assignment so you can do::

        >>> with open('/path/to/hello.world') as f:
        ...     instance.file = File(f)
    """
    def __get__(self, instance, cls=None):
        if instance is None:
            return self

        # This is slightly complicated, so worth an explanation.
        # instance.file`needs to ultimately return some instance of `File`,
        # probably a subclass. Additionally, this returned object needs to have
        # the FieldFile API so that users can easily do things like
        # instance.file.path and have that delegated to the file storage engine.
        # Easy enough if we're strict about assignment in __set__, but if you
        # peek below you can see that we're not. So depending on the current
        # value of the field we have to dynamically construct some sort of
        # "thing" to return.

        # The instance dict contains whatever was originally assigned
        # in __set__.
        file = super().__get__(instance, cls)

        # If this value is a string (instance.file = "path/to/file") or None
        # then we simply wrap it with the appropriate attribute class according
        # to the file field. [This is FieldFile for FileFields and
        # ImageFieldFile for ImageFields; it's also conceivable that user
        # subclasses might also want to subclass the attribute class]. This
        # object understands how to convert a path to a file, and also how to
        # handle None.
        if isinstance(file, str) or file is None:
            attr = self.field.attr_class(instance, self.field, file)
            instance.__dict__[self.field.attname] = attr

        # Other types of files may be assigned as well, but they need to have
        # the FieldFile interface added to them. Thus, we wrap any other type of
        # File inside a FieldFile (well, the field's attr_class, which is
        # usually FieldFile).
        elif isinstance(file, File) and not isinstance(file, FieldFile):
            file_copy = self.field.attr_class(instance, self.field, file.name)
            file_copy.file = file
            file_copy._committed = False
            instance.__dict__[self.field.attname] = file_copy

        # Finally, because of the (some would say boneheaded) way pickle works,
        # the underlying FieldFile might not actually itself have an associated
        # file. So we need to reset the details of the FieldFile in those cases.
        elif isinstance(file, FieldFile) and not hasattr(file, 'field'):
            file.instance = instance
            file.field = self.field
            file.storage = self.field.storage

        # Make sure that the instance is correct.
        elif isinstance(file, FieldFile) and instance is not file.instance:
            file.instance = instance

        # That was fun, wasn't it?
        return instance.__dict__[self.field.attname]

    def __set__(self, instance, value):
        instance.__dict__[self.field.attname] = value
```
### 5 - django/db/models/fields/files.py:

Start line: 334, End line: 361

```python
class ImageFileDescriptor(FileDescriptor):
    """
    Just like the FileDescriptor, but for ImageFields. The only difference is
    assigning the width/height to the width_field/height_field, if appropriate.
    """
    def __set__(self, instance, value):
        previous_file = instance.__dict__.get(self.field.attname)
        super().__set__(instance, value)

        # To prevent recalculating image dimensions when we are instantiating
        # an object from the database (bug #11084), only update dimensions if
        # the field had a value before this assignment.  Since the default
        # value for FileField subclasses is an instance of field.attr_class,
        # previous_file will only be None when we are called from
        # Model.__init__().  The ImageField.update_dimension_fields method
        # hooked up to the post_init signal handles the Model.__init__() cases.
        # Assignment happening outside of Model.__init__() will trigger the
        # update right here.
        if previous_file is not None:
            self.field.update_dimension_fields(instance, force=True)


class ImageFieldFile(ImageFile, FieldFile):
    def delete(self, save=True):
        # Clear the image dimensions cache
        if hasattr(self, '_dimensions_cache'):
            del self._dimensions_cache
        super().delete(save)
```
### 6 - django/db/models/options.py:

Start line: 607, End line: 625

```python
class Options:

    def get_ancestor_link(self, ancestor):
        """
        Return the field on the current model which points to the given
        "ancestor". This is possible an indirect link (a pointer to a parent
        model, which points, eventually, to the ancestor). Used when
        constructing table joins for model inheritance.

        Return None if the model isn't an ancestor of this one.
        """
        if ancestor in self.parents:
            return self.parents[ancestor]
        for parent in self.parents:
            # Tries to get a link field from the immediate parent
            parent_link = parent._meta.get_ancestor_link(ancestor)
            if parent_link:
                # In case of a proxied model, the first link
                # of the chain to the ancestor is that parent
                # links
                return self.parents[parent] or parent_link
```
### 7 - django/db/models/fields/related.py:

Start line: 171, End line: 188

```python
class RelatedField(FieldCacheMixin, Field):

    def _check_referencing_to_swapped_model(self):
        if (self.remote_field.model not in self.opts.apps.get_models() and
                not isinstance(self.remote_field.model, str) and
                self.remote_field.model._meta.swapped):
            model = "%s.%s" % (
                self.remote_field.model._meta.app_label,
                self.remote_field.model._meta.object_name
            )
            return [
                checks.Error(
                    "Field defines a relation with the model '%s', which has "
                    "been swapped out." % model,
                    hint="Update the relation to point at 'settings.%s'." % self.remote_field.model._meta.swappable,
                    obj=self,
                    id='fields.E301',
                )
            ]
        return []
```
### 8 - django/db/models/fields/related.py:

Start line: 1235, End line: 1352

```python
class ManyToManyField(RelatedField):

    def _check_relationship_model(self, from_model=None, **kwargs):
        if hasattr(self.remote_field.through, '_meta'):
            qualified_model_name = "%s.%s" % (
                self.remote_field.through._meta.app_label, self.remote_field.through.__name__)
        else:
            qualified_model_name = self.remote_field.through

        errors = []

        if self.remote_field.through not in self.opts.apps.get_models(include_auto_created=True):
            # The relationship model is not installed.
            errors.append(
                checks.Error(
                    "Field specifies a many-to-many relation through model "
                    "'%s', which has not been installed." % qualified_model_name,
                    obj=self,
                    id='fields.E331',
                )
            )

        else:
            assert from_model is not None, (
                "ManyToManyField with intermediate "
                "tables cannot be checked if you don't pass the model "
                "where the field is attached to."
            )
            # Set some useful local variables
            to_model = resolve_relation(from_model, self.remote_field.model)
            from_model_name = from_model._meta.object_name
            if isinstance(to_model, str):
                to_model_name = to_model
            else:
                to_model_name = to_model._meta.object_name
            relationship_model_name = self.remote_field.through._meta.object_name
            self_referential = from_model == to_model
            # Count foreign keys in intermediate model
            if self_referential:
                seen_self = sum(
                    from_model == getattr(field.remote_field, 'model', None)
                    for field in self.remote_field.through._meta.fields
                )

                if seen_self > 2 and not self.remote_field.through_fields:
                    errors.append(
                        checks.Error(
                            "The model is used as an intermediate model by "
                            "'%s', but it has more than two foreign keys "
                            "to '%s', which is ambiguous. You must specify "
                            "which two foreign keys Django should use via the "
                            "through_fields keyword argument." % (self, from_model_name),
                            hint="Use through_fields to specify which two foreign keys Django should use.",
                            obj=self.remote_field.through,
                            id='fields.E333',
                        )
                    )

            else:
                # Count foreign keys in relationship model
                seen_from = sum(
                    from_model == getattr(field.remote_field, 'model', None)
                    for field in self.remote_field.through._meta.fields
                )
                seen_to = sum(
                    to_model == getattr(field.remote_field, 'model', None)
                    for field in self.remote_field.through._meta.fields
                )

                if seen_from > 1 and not self.remote_field.through_fields:
                    errors.append(
                        checks.Error(
                            ("The model is used as an intermediate model by "
                             "'%s', but it has more than one foreign key "
                             "from '%s', which is ambiguous. You must specify "
                             "which foreign key Django should use via the "
                             "through_fields keyword argument.") % (self, from_model_name),
                            hint=(
                                'If you want to create a recursive relationship, '
                                'use ManyToManyField("%s", through="%s").'
                            ) % (
                                RECURSIVE_RELATIONSHIP_CONSTANT,
                                relationship_model_name,
                            ),
                            obj=self,
                            id='fields.E334',
                        )
                    )

                if seen_to > 1 and not self.remote_field.through_fields:
                    errors.append(
                        checks.Error(
                            "The model is used as an intermediate model by "
                            "'%s', but it has more than one foreign key "
                            "to '%s', which is ambiguous. You must specify "
                            "which foreign key Django should use via the "
                            "through_fields keyword argument." % (self, to_model_name),
                            hint=(
                                'If you want to create a recursive relationship, '
                                'use ManyToManyField("%s", through="%s").'
                            ) % (
                                RECURSIVE_RELATIONSHIP_CONSTANT,
                                relationship_model_name,
                            ),
                            obj=self,
                            id='fields.E335',
                        )
                    )

                if seen_from == 0 or seen_to == 0:
                    errors.append(
                        checks.Error(
                            "The model is used as an intermediate model by "
                            "'%s', but it does not have a foreign key to '%s' or '%s'." % (
                                self, from_model_name, to_model_name
                            ),
                            obj=self.remote_field.through,
                            id='fields.E336',
                        )
                    )
        # ... other code
```
### 9 - django/db/models/options.py:

Start line: 332, End line: 355

```python
class Options:

    @property
    def swapped(self):
        """
        Has this model been swapped out for another? If so, return the model
        name of the replacement; otherwise, return None.

        For historical reasons, model name lookups using get_model() are
        case insensitive, so we make sure we are case insensitive here.
        """
        if self.swappable:
            swapped_for = getattr(settings, self.swappable, None)
            if swapped_for:
                try:
                    swapped_label, swapped_object = swapped_for.split('.')
                except ValueError:
                    # setting not in the format app_label.model_name
                    # raising ImproperlyConfigured here causes problems with
                    # test cleanup code - instead it is raised in get_user_model
                    # or as part of validation.
                    return swapped_for

                if '%s.%s' % (swapped_label, swapped_object.lower()) != self.label_lower:
                    return swapped_for
        return None
```
### 10 - django/db/models/base.py:

Start line: 404, End line: 503

```python
class Model(metaclass=ModelBase):

    def __init__(self, *args, **kwargs):
        # Alias some things as locals to avoid repeat global lookups
        cls = self.__class__
        opts = self._meta
        _setattr = setattr
        _DEFERRED = DEFERRED

        pre_init.send(sender=cls, args=args, kwargs=kwargs)

        # Set up the storage for instance state
        self._state = ModelState()

        # There is a rather weird disparity here; if kwargs, it's set, then args
        # overrides it. It should be one or the other; don't duplicate the work
        # The reason for the kwargs check is that standard iterator passes in by
        # args, and instantiation for iteration is 33% faster.
        if len(args) > len(opts.concrete_fields):
            # Daft, but matches old exception sans the err msg.
            raise IndexError("Number of args exceeds number of fields")

        if not kwargs:
            fields_iter = iter(opts.concrete_fields)
            # The ordering of the zip calls matter - zip throws StopIteration
            # when an iter throws it. So if the first iter throws it, the second
            # is *not* consumed. We rely on this, so don't change the order
            # without changing the logic.
            for val, field in zip(args, fields_iter):
                if val is _DEFERRED:
                    continue
                _setattr(self, field.attname, val)
        else:
            # Slower, kwargs-ready version.
            fields_iter = iter(opts.fields)
            for val, field in zip(args, fields_iter):
                if val is _DEFERRED:
                    continue
                _setattr(self, field.attname, val)
                kwargs.pop(field.name, None)

        # Now we're left with the unprocessed fields that *must* come from
        # keywords, or default.

        for field in fields_iter:
            is_related_object = False
            # Virtual field
            if field.attname not in kwargs and field.column is None:
                continue
            if kwargs:
                if isinstance(field.remote_field, ForeignObjectRel):
                    try:
                        # Assume object instance was passed in.
                        rel_obj = kwargs.pop(field.name)
                        is_related_object = True
                    except KeyError:
                        try:
                            # Object instance wasn't passed in -- must be an ID.
                            val = kwargs.pop(field.attname)
                        except KeyError:
                            val = field.get_default()
                else:
                    try:
                        val = kwargs.pop(field.attname)
                    except KeyError:
                        # This is done with an exception rather than the
                        # default argument on pop because we don't want
                        # get_default() to be evaluated, and then not used.
                        # Refs #12057.
                        val = field.get_default()
            else:
                val = field.get_default()

            if is_related_object:
                # If we are passed a related instance, set it using the
                # field.name instead of field.attname (e.g. "user" instead of
                # "user_id") so that the object gets properly cached (and type
                # checked) by the RelatedObjectDescriptor.
                if rel_obj is not _DEFERRED:
                    _setattr(self, field.name, rel_obj)
            else:
                if val is not _DEFERRED:
                    _setattr(self, field.attname, val)

        if kwargs:
            property_names = opts._property_names
            for prop in tuple(kwargs):
                try:
                    # Any remaining kwargs must correspond to properties or
                    # virtual fields.
                    if prop in property_names or opts.get_field(prop):
                        if kwargs[prop] is not _DEFERRED:
                            _setattr(self, prop, kwargs[prop])
                        del kwargs[prop]
                except (AttributeError, FieldDoesNotExist):
                    pass
            for kwarg in kwargs:
                raise TypeError("%s() got an unexpected keyword argument '%s'" % (cls.__name__, kwarg))
        super().__init__()
        post_init.send(sender=cls, instance=self)
```
### 19 - django/db/models/fields/files.py:

Start line: 1, End line: 141

```python
import datetime
import posixpath

from django import forms
from django.core import checks
from django.core.files.base import File
from django.core.files.images import ImageFile
from django.core.files.storage import Storage, default_storage
from django.db.models import signals
from django.db.models.fields import Field
from django.db.models.query_utils import DeferredAttribute
from django.utils.translation import gettext_lazy as _


class FieldFile(File):
    def __init__(self, instance, field, name):
        super().__init__(None, name)
        self.instance = instance
        self.field = field
        self.storage = field.storage
        self._committed = True

    def __eq__(self, other):
        # Older code may be expecting FileField values to be simple strings.
        # By overriding the == operator, it can remain backwards compatibility.
        if hasattr(other, 'name'):
            return self.name == other.name
        return self.name == other

    def __hash__(self):
        return hash(self.name)

    # The standard File contains most of the necessary properties, but
    # FieldFiles can be instantiated without a name, so that needs to
    # be checked for here.

    def _require_file(self):
        if not self:
            raise ValueError("The '%s' attribute has no file associated with it." % self.field.name)

    def _get_file(self):
        self._require_file()
        if getattr(self, '_file', None) is None:
            self._file = self.storage.open(self.name, 'rb')
        return self._file

    def _set_file(self, file):
        self._file = file

    def _del_file(self):
        del self._file

    file = property(_get_file, _set_file, _del_file)

    @property
    def path(self):
        self._require_file()
        return self.storage.path(self.name)

    @property
    def url(self):
        self._require_file()
        return self.storage.url(self.name)

    @property
    def size(self):
        self._require_file()
        if not self._committed:
            return self.file.size
        return self.storage.size(self.name)

    def open(self, mode='rb'):
        self._require_file()
        if getattr(self, '_file', None) is None:
            self.file = self.storage.open(self.name, mode)
        else:
            self.file.open(mode)
        return self
    # open() doesn't alter the file's contents, but it does reset the pointer
    open.alters_data = True

    # In addition to the standard File API, FieldFiles have extra methods
    # to further manipulate the underlying file, as well as update the
    # associated model instance.

    def save(self, name, content, save=True):
        name = self.field.generate_filename(self.instance, name)
        self.name = self.storage.save(name, content, max_length=self.field.max_length)
        setattr(self.instance, self.field.name, self.name)
        self._committed = True

        # Save the object because it has changed, unless save is False
        if save:
            self.instance.save()
    save.alters_data = True

    def delete(self, save=True):
        if not self:
            return
        # Only close the file if it's already open, which we know by the
        # presence of self._file
        if hasattr(self, '_file'):
            self.close()
            del self.file

        self.storage.delete(self.name)

        self.name = None
        setattr(self.instance, self.field.name, self.name)
        self._committed = False

        if save:
            self.instance.save()
    delete.alters_data = True

    @property
    def closed(self):
        file = getattr(self, '_file', None)
        return file is None or file.closed

    def close(self):
        file = getattr(self, '_file', None)
        if file is not None:
            file.close()

    def __getstate__(self):
        # FieldFile needs access to its associated model field, an instance and
        # the file's name. Everything else will be restored later, by
        # FileDescriptor below.
        return {
            'name': self.name,
            'closed': False,
            '_committed': True,
            '_file': None,
            'instance': self.instance,
            'field': self.field,
        }

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.storage = self.field.storage
```
### 20 - django/db/models/fields/files.py:

Start line: 412, End line: 474

```python
class ImageField(FileField):

    def update_dimension_fields(self, instance, force=False, *args, **kwargs):
        """
        Update field's width and height fields, if defined.

        This method is hooked up to model's post_init signal to update
        dimensions after instantiating a model instance.  However, dimensions
        won't be updated if the dimensions fields are already populated.  This
        avoids unnecessary recalculation when loading an object from the
        database.

        Dimensions can be forced to update with force=True, which is how
        ImageFileDescriptor.__set__ calls this method.
        """
        # Nothing to update if the field doesn't have dimension fields or if
        # the field is deferred.
        has_dimension_fields = self.width_field or self.height_field
        if not has_dimension_fields or self.attname not in instance.__dict__:
            return

        # getattr will call the ImageFileDescriptor's __get__ method, which
        # coerces the assigned value into an instance of self.attr_class
        # (ImageFieldFile in this case).
        file = getattr(instance, self.attname)

        # Nothing to update if we have no file and not being forced to update.
        if not file and not force:
            return

        dimension_fields_filled = not(
            (self.width_field and not getattr(instance, self.width_field)) or
            (self.height_field and not getattr(instance, self.height_field))
        )
        # When both dimension fields have values, we are most likely loading
        # data from the database or updating an image field that already had
        # an image stored.  In the first case, we don't want to update the
        # dimension fields because we are already getting their values from the
        # database.  In the second case, we do want to update the dimensions
        # fields and will skip this return because force will be True since we
        # were called from ImageFileDescriptor.__set__.
        if dimension_fields_filled and not force:
            return

        # file should be an instance of ImageFieldFile or should be None.
        if file:
            width = file.width
            height = file.height
        else:
            # No file, so clear dimensions fields.
            width = None
            height = None

        # Update the width and height fields.
        if self.width_field:
            setattr(instance, self.width_field, width)
        if self.height_field:
            setattr(instance, self.height_field, height)

    def formfield(self, **kwargs):
        return super().formfield(**{
            'form_class': forms.ImageField,
            **kwargs,
        })
```
