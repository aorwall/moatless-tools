# astropy__astropy-7008

| **astropy/astropy** | `264d967708a3dcdb2bce0ed9f9ca3391c40f3ff3` |
| ---- | ---- |
| **No of patches** | 4 |
| **All found context length** | - |
| **Any found context length** | 1215 |
| **Avg pos** | 3.5 |
| **Min pos** | 7 |
| **Max pos** | 7 |
| **Top file pos** | 1 |
| **Missing snippets** | 6 |
| **Missing patch files** | 1 |


## Expected patch

```diff
diff --git a/astropy/constants/__init__.py b/astropy/constants/__init__.py
--- a/astropy/constants/__init__.py
+++ b/astropy/constants/__init__.py
@@ -13,8 +13,8 @@
     <Quantity 0.510998927603161 MeV>
 
 """
-
-import itertools
+import inspect
+from contextlib import contextmanager
 
 # Hack to make circular imports with units work
 try:
@@ -23,10 +23,11 @@
 except ImportError:
     pass
 
-from .constant import Constant, EMConstant
-from . import si
-from . import cgs
-from . import codata2014, iau2015
+from .constant import Constant, EMConstant  # noqa
+from . import si  # noqa
+from . import cgs  # noqa
+from . import codata2014, iau2015  # noqa
+from . import utils as _utils
 
 # for updating the constants module docstring
 _lines = [
@@ -36,19 +37,65 @@
     '========== ============== ================ =========================',
 ]
 
-for _nm, _c in itertools.chain(sorted(vars(codata2014).items()),
-                               sorted(vars(iau2015).items())):
-    if isinstance(_c, Constant) and _c.abbrev not in locals():
-        locals()[_c.abbrev] = _c.__class__(_c.abbrev, _c.name, _c.value,
-                                           _c._unit_string, _c.uncertainty,
-                                           _c.reference)
-
-        _lines.append('{0:^10} {1:^14.9g} {2:^16} {3}'.format(
-            _c.abbrev, _c.value, _c._unit_string, _c.name))
+# NOTE: Update this when default changes.
+_utils._set_c(codata2014, iau2015, inspect.getmodule(inspect.currentframe()),
+              not_in_module_only=True, doclines=_lines, set_class=True)
 
 _lines.append(_lines[1])
 
 if __doc__ is not None:
     __doc__ += '\n'.join(_lines)
 
-del _lines, _nm, _c
+
+# TODO: Re-implement in a way that is more consistent with astropy.units.
+#       See https://github.com/astropy/astropy/pull/7008 discussions.
+@contextmanager
+def set_enabled_constants(modname):
+    """
+    Context manager to temporarily set values in the ``constants``
+    namespace to an older version.
+    See :ref:`astropy-constants-prior` for usage.
+
+    Parameters
+    ----------
+    modname : {'astropyconst13'}
+        Name of the module containing an older version.
+
+    """
+
+    # Re-import here because these were deleted from namespace on init.
+    import inspect
+    import warnings
+    from . import utils as _utils
+
+    # NOTE: Update this when default changes.
+    if modname == 'astropyconst13':
+        from .astropyconst13 import codata2010 as codata
+        from .astropyconst13 import iau2012 as iaudata
+    else:
+        raise ValueError(
+            'Context manager does not currently handle {}'.format(modname))
+
+    module = inspect.getmodule(inspect.currentframe())
+
+    # Ignore warnings about "Constant xxx already has a definition..."
+    with warnings.catch_warnings():
+        warnings.simplefilter('ignore')
+        _utils._set_c(codata, iaudata, module,
+                      not_in_module_only=False, set_class=True)
+
+    try:
+        yield
+    finally:
+        with warnings.catch_warnings():
+            warnings.simplefilter('ignore')
+            # NOTE: Update this when default changes.
+            _utils._set_c(codata2014, iau2015, module,
+                          not_in_module_only=False, set_class=True)
+
+
+# Clean up namespace
+del inspect
+del contextmanager
+del _utils
+del _lines
diff --git a/astropy/constants/astropyconst13.py b/astropy/constants/astropyconst13.py
--- a/astropy/constants/astropyconst13.py
+++ b/astropy/constants/astropyconst13.py
@@ -4,15 +4,12 @@
 See :mod:`astropy.constants` for a complete listing of constants
 defined in Astropy.
 """
-
-
-
-import itertools
-
-from .constant import Constant
+import inspect
+from . import utils as _utils
 from . import codata2010, iau2012
 
-for _nm, _c in itertools.chain(sorted(vars(codata2010).items()),
-                               sorted(vars(iau2012).items())):
-    if (isinstance(_c, Constant) and _c.abbrev not in locals()):
-        locals()[_c.abbrev] = _c
+_utils._set_c(codata2010, iau2012, inspect.getmodule(inspect.currentframe()))
+
+# Clean up namespace
+del inspect
+del _utils
diff --git a/astropy/constants/astropyconst20.py b/astropy/constants/astropyconst20.py
--- a/astropy/constants/astropyconst20.py
+++ b/astropy/constants/astropyconst20.py
@@ -3,15 +3,12 @@
 Astronomical and physics constants for Astropy v2.0.  See :mod:`astropy.constants`
 for a complete listing of constants defined in Astropy.
 """
-
-
-
-import itertools
-
-from .constant import Constant
+import inspect
+from . import utils as _utils
 from . import codata2014, iau2015
 
-for _nm, _c in itertools.chain(sorted(vars(codata2014).items()),
-                               sorted(vars(iau2015).items())):
-    if (isinstance(_c, Constant) and _c.abbrev not in locals()):
-        locals()[_c.abbrev] = _c
+_utils._set_c(codata2014, iau2015, inspect.getmodule(inspect.currentframe()))
+
+# Clean up namespace
+del inspect
+del _utils
diff --git a/astropy/constants/utils.py b/astropy/constants/utils.py
new file mode 100644
--- /dev/null
+++ b/astropy/constants/utils.py
@@ -0,0 +1,80 @@
+# Licensed under a 3-clause BSD style license - see LICENSE.rst
+"""Utility functions for ``constants`` sub-package."""
+import itertools
+
+__all__ = []
+
+
+def _get_c(codata, iaudata, module, not_in_module_only=True):
+    """
+    Generator to return a Constant object.
+
+    Parameters
+    ----------
+    codata, iaudata : obj
+        Modules containing CODATA and IAU constants of interest.
+
+    module : obj
+        Namespace module of interest.
+
+    not_in_module_only : bool
+        If ``True``, ignore constants that are already in the
+        namespace of ``module``.
+
+    Returns
+    -------
+    _c : Constant
+        Constant object to process.
+
+    """
+    from .constant import Constant
+
+    for _nm, _c in itertools.chain(sorted(vars(codata).items()),
+                                   sorted(vars(iaudata).items())):
+        if not isinstance(_c, Constant):
+            continue
+        elif (not not_in_module_only) or (_c.abbrev not in module.__dict__):
+            yield _c
+
+
+def _set_c(codata, iaudata, module, not_in_module_only=True, doclines=None,
+           set_class=False):
+    """
+    Set constants in a given module namespace.
+
+    Parameters
+    ----------
+    codata, iaudata : obj
+        Modules containing CODATA and IAU constants of interest.
+
+    module : obj
+        Namespace module to modify with the given ``codata`` and ``iaudata``.
+
+    not_in_module_only : bool
+        If ``True``, constants that are already in the namespace
+        of ``module`` will not be modified.
+
+    doclines : list or `None`
+        If a list is given, this list will be modified in-place to include
+        documentation of modified constants. This can be used to update
+        docstring of ``module``.
+
+    set_class : bool
+        Namespace of ``module`` is populated with ``_c.__class__``
+        instead of just ``_c`` from :func:`_get_c`.
+
+    """
+    for _c in _get_c(codata, iaudata, module,
+                     not_in_module_only=not_in_module_only):
+        if set_class:
+            value = _c.__class__(_c.abbrev, _c.name, _c.value,
+                                 _c._unit_string, _c.uncertainty,
+                                 _c.reference)
+        else:
+            value = _c
+
+        setattr(module, _c.abbrev, value)
+
+        if doclines is not None:
+            doclines.append('{0:^10} {1:^14.9g} {2:^16} {3}'.format(
+                _c.abbrev, _c.value, _c._unit_string, _c.name))

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| astropy/constants/__init__.py | 16 | 17 | - | 7 | -
| astropy/constants/__init__.py | 26 | 29 | 7 | 7 | 1215
| astropy/constants/__init__.py | 39 | 54 | 7 | 7 | 1215
| astropy/constants/astropyconst13.py | 7 | 18 | - | 1 | -
| astropy/constants/astropyconst20.py | 6 | 17 | - | 2 | -
| astropy/constants/utils.py | 0 | 0 | - | - | -


## Problem Statement

```
Context manager for constant versions
For some use cases it would be helpful to have a context manager to set the version set of the constants. E.g., something like 
\`\`\`
with constants_set(astropyconst13):
    ... code goes here ...
\`\`\``

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 astropy/constants/astropyconst13.py** | 10 | 19| 82 | 82 | 140 | 
| 2 | **2 astropy/constants/astropyconst20.py** | 9 | 18| 82 | 164 | 278 | 
| 3 | 3 astropy/constants/cgs.py | 7 | 17| 99 | 263 | 429 | 
| 4 | 4 astropy/utils/misc.py | 826 | 860| 225 | 488 | 9034 | 
| 5 | 5 astropy/constants/constant.py | 76 | 118| 343 | 831 | 10809 | 
| 6 | 6 astropy/constants/si.py | 9 | 19| 91 | 922 | 10952 | 
| **-> 7 <-** | **7 astropy/constants/__init__.py** | 17 | 55| 293 | 1215 | 11365 | 
| 8 | 8 astropy/config/paths.py | 195 | 237| 315 | 1530 | 13662 | 
| 9 | 8 astropy/constants/constant.py | 33 | 73| 382 | 1912 | 13662 | 
| 10 | 8 astropy/constants/constant.py | 120 | 217| 671 | 2583 | 13662 | 
| 11 | 8 astropy/constants/constant.py | 3 | 31| 260 | 2843 | 13662 | 
| 12 | 8 astropy/utils/misc.py | 114 | 158| 285 | 3128 | 13662 | 
| 13 | 9 astropy/utils/data_info.py | 47 | 63| 121 | 3249 | 18305 | 
| 14 | 9 astropy/config/paths.py | 240 | 269| 254 | 3503 | 18305 | 
| 15 | 10 astropy/table/index.py | 631 | 691| 538 | 4041 | 24751 | 
| 16 | 11 astropy/io/registry.py | 42 | 75| 203 | 4244 | 29378 | 
| 17 | 12 astropy/utils/compat/numpycompat.py | 6 | 20| 202 | 4446 | 29621 | 
| 18 | 13 ez_setup.py | 191 | 206| 134 | 4580 | 32452 | 
| 19 | 14 astropy/__init__.py | 43 | 87| 330 | 4910 | 34875 | 
| 20 | 15 astropy/config/configuration.py | 95 | 117| 145 | 5055 | 40236 | 
| 21 | 16 astropy/constants/codata2010.py | 7 | 71| 754 | 5809 | 41591 | 
| 22 | 17 setup.py | 1 | 84| 665 | 6474 | 42570 | 
| 23 | 17 ez_setup.py | 152 | 188| 261 | 6735 | 42570 | 
| 24 | 18 astropy/constants/codata2014.py | 7 | 68| 736 | 7471 | 43866 | 
| 25 | 18 setup.py | 87 | 121| 297 | 7768 | 43866 | 
| 26 | 18 astropy/constants/constant.py | 220 | 234| 120 | 7888 | 43866 | 
| 27 | 19 astropy/table/table.py | 567 | 586| 181 | 8069 | 66646 | 
| 28 | 19 astropy/__init__.py | 122 | 156| 291 | 8360 | 66646 | 
| 29 | 20 astropy/utils/console.py | 1 | 37| 150 | 8510 | 74307 | 
| 30 | 20 ez_setup.py | 81 | 95| 120 | 8630 | 74307 | 
| 31 | 20 astropy/config/configuration.py | 302 | 328| 152 | 8782 | 74307 | 
| 32 | 21 astropy/modeling/functional_models.py | 1092 | 1134| 205 | 8987 | 93579 | 
| 33 | 22 astropy/utils/state.py | 1 | 40| 227 | 9214 | 93995 | 
| 34 | 23 astropy/utils/metadata.py | 232 | 288| 500 | 9714 | 97215 | 
| 35 | 24 astropy/logger.py | 365 | 412| 345 | 10059 | 101168 | 
| 36 | 24 astropy/utils/state.py | 42 | 74| 192 | 10251 | 101168 | 
| 37 | 25 astropy/utils/data.py | 9 | 39| 209 | 10460 | 112033 | 
| 38 | 26 docs/conf.py | 1 | 111| 762 | 11222 | 114314 | 
| 39 | 27 astropy/utils/compat/futures/__init__.py | 1 | 8| 49 | 11271 | 114363 | 
| 40 | 27 astropy/config/paths.py | 162 | 192| 199 | 11470 | 114363 | 
| 41 | 28 astropy/constants/iau2012.py | 7 | 77| 749 | 12219 | 115162 | 
| 42 | 29 astropy/samp/__init__.py | 12 | 40| 142 | 12361 | 115411 | 
| 43 | 29 astropy/modeling/functional_models.py | 1169 | 1215| 257 | 12618 | 115411 | 
| 44 | 30 ah_bootstrap.py | 99 | 162| 436 | 13054 | 122974 | 
| 45 | 30 docs/conf.py | 201 | 253| 403 | 13457 | 122974 | 
| 46 | 31 astropy/units/deprecated.py | 1 | 37| 174 | 13631 | 123488 | 
| 47 | 32 astropy/nddata/__init__.py | 11 | 51| 246 | 13877 | 123831 | 
| 48 | 32 astropy/modeling/functional_models.py | 1136 | 1166| 224 | 14101 | 123831 | 
| 49 | 32 astropy/logger.py | 414 | 455| 290 | 14391 | 123831 | 
| 50 | 32 astropy/constants/codata2010.py | 73 | 111| 551 | 14942 | 123831 | 
| 51 | 32 astropy/constants/codata2014.py | 70 | 106| 510 | 15452 | 123831 | 
| 52 | 33 astropy/constants/iau2015.py | 7 | 77| 749 | 16201 | 124889 | 
| 53 | 33 astropy/__init__.py | 159 | 215| 454 | 16655 | 124889 | 
| 54 | 34 astropy/conftest.py | 7 | 25| 141 | 16796 | 125085 | 
| 55 | 35 astropy/config/__init__.py | 1 | 12| 62 | 16858 | 125147 | 
| 56 | 36 astropy/units/__init__.py | 13 | 40| 140 | 16998 | 125374 | 
| 57 | 37 astropy/extern/configobj/validate.py | 1 | 217| 679 | 17677 | 138006 | 
| 58 | 37 astropy/__init__.py | 218 | 258| 281 | 17958 | 138006 | 
| 59 | 38 astropy/table/bst.py | 58 | 89| 183 | 18141 | 142371 | 
| 60 | 39 astropy/io/ascii/core.py | 137 | 149| 121 | 18262 | 154138 | 
| 61 | 40 astropy/extern/configobj/configobj.py | 1948 | 1973| 212 | 18474 | 172565 | 
| 62 | 41 astropy/io/fits/__init__.py | 15 | 59| 438 | 18912 | 173326 | 
| 63 | 42 astropy/io/votable/__init__.py | 8 | 34| 218 | 19130 | 173590 | 
| 64 | 43 astropy/visualization/mpl_style.py | 7 | 87| 647 | 19777 | 174298 | 
| 65 | 44 astropy/utils/__init__.py | 13 | 17| 21 | 19798 | 174426 | 
| 66 | 45 astropy/extern/bundled/six.py | 618 | 721| 712 | 20510 | 181774 | 
| 67 | 45 astropy/extern/configobj/validate.py | 1336 | 1417| 960 | 21470 | 181774 | 
| 68 | 46 astropy/samp/constants.py | 7 | 26| 177 | 21647 | 181980 | 
| 69 | 46 ah_bootstrap.py | 802 | 833| 200 | 21847 | 181980 | 
| 70 | 46 astropy/extern/configobj/validate.py | 628 | 657| 281 | 22128 | 181980 | 
| 71 | 47 conftest.py | 1 | 7| 38 | 22166 | 182018 | 
| 72 | 47 astropy/config/configuration.py | 275 | 300| 162 | 22328 | 182018 | 
| 73 | 48 astropy/_erfa/__init__.py | 1 | 8| 54 | 22382 | 182073 | 
| 74 | 48 astropy/__init__.py | 319 | 342| 275 | 22657 | 182073 | 
| 75 | 49 astropy/time/__init__.py | 1 | 4| 27 | 22684 | 182100 | 
| 76 | 50 astropy/wcs/docstrings.py | 1537 | 1649| 675 | 23359 | 198049 | 


## Missing Patch Files

 * 1: astropy/constants/__init__.py
 * 2: astropy/constants/astropyconst13.py
 * 3: astropy/constants/astropyconst20.py
 * 4: astropy/constants/utils.py

### Hint

```
I am trying to take a stab at this but no promises.
```

## Patch

```diff
diff --git a/astropy/constants/__init__.py b/astropy/constants/__init__.py
--- a/astropy/constants/__init__.py
+++ b/astropy/constants/__init__.py
@@ -13,8 +13,8 @@
     <Quantity 0.510998927603161 MeV>
 
 """
-
-import itertools
+import inspect
+from contextlib import contextmanager
 
 # Hack to make circular imports with units work
 try:
@@ -23,10 +23,11 @@
 except ImportError:
     pass
 
-from .constant import Constant, EMConstant
-from . import si
-from . import cgs
-from . import codata2014, iau2015
+from .constant import Constant, EMConstant  # noqa
+from . import si  # noqa
+from . import cgs  # noqa
+from . import codata2014, iau2015  # noqa
+from . import utils as _utils
 
 # for updating the constants module docstring
 _lines = [
@@ -36,19 +37,65 @@
     '========== ============== ================ =========================',
 ]
 
-for _nm, _c in itertools.chain(sorted(vars(codata2014).items()),
-                               sorted(vars(iau2015).items())):
-    if isinstance(_c, Constant) and _c.abbrev not in locals():
-        locals()[_c.abbrev] = _c.__class__(_c.abbrev, _c.name, _c.value,
-                                           _c._unit_string, _c.uncertainty,
-                                           _c.reference)
-
-        _lines.append('{0:^10} {1:^14.9g} {2:^16} {3}'.format(
-            _c.abbrev, _c.value, _c._unit_string, _c.name))
+# NOTE: Update this when default changes.
+_utils._set_c(codata2014, iau2015, inspect.getmodule(inspect.currentframe()),
+              not_in_module_only=True, doclines=_lines, set_class=True)
 
 _lines.append(_lines[1])
 
 if __doc__ is not None:
     __doc__ += '\n'.join(_lines)
 
-del _lines, _nm, _c
+
+# TODO: Re-implement in a way that is more consistent with astropy.units.
+#       See https://github.com/astropy/astropy/pull/7008 discussions.
+@contextmanager
+def set_enabled_constants(modname):
+    """
+    Context manager to temporarily set values in the ``constants``
+    namespace to an older version.
+    See :ref:`astropy-constants-prior` for usage.
+
+    Parameters
+    ----------
+    modname : {'astropyconst13'}
+        Name of the module containing an older version.
+
+    """
+
+    # Re-import here because these were deleted from namespace on init.
+    import inspect
+    import warnings
+    from . import utils as _utils
+
+    # NOTE: Update this when default changes.
+    if modname == 'astropyconst13':
+        from .astropyconst13 import codata2010 as codata
+        from .astropyconst13 import iau2012 as iaudata
+    else:
+        raise ValueError(
+            'Context manager does not currently handle {}'.format(modname))
+
+    module = inspect.getmodule(inspect.currentframe())
+
+    # Ignore warnings about "Constant xxx already has a definition..."
+    with warnings.catch_warnings():
+        warnings.simplefilter('ignore')
+        _utils._set_c(codata, iaudata, module,
+                      not_in_module_only=False, set_class=True)
+
+    try:
+        yield
+    finally:
+        with warnings.catch_warnings():
+            warnings.simplefilter('ignore')
+            # NOTE: Update this when default changes.
+            _utils._set_c(codata2014, iau2015, module,
+                          not_in_module_only=False, set_class=True)
+
+
+# Clean up namespace
+del inspect
+del contextmanager
+del _utils
+del _lines
diff --git a/astropy/constants/astropyconst13.py b/astropy/constants/astropyconst13.py
--- a/astropy/constants/astropyconst13.py
+++ b/astropy/constants/astropyconst13.py
@@ -4,15 +4,12 @@
 See :mod:`astropy.constants` for a complete listing of constants
 defined in Astropy.
 """
-
-
-
-import itertools
-
-from .constant import Constant
+import inspect
+from . import utils as _utils
 from . import codata2010, iau2012
 
-for _nm, _c in itertools.chain(sorted(vars(codata2010).items()),
-                               sorted(vars(iau2012).items())):
-    if (isinstance(_c, Constant) and _c.abbrev not in locals()):
-        locals()[_c.abbrev] = _c
+_utils._set_c(codata2010, iau2012, inspect.getmodule(inspect.currentframe()))
+
+# Clean up namespace
+del inspect
+del _utils
diff --git a/astropy/constants/astropyconst20.py b/astropy/constants/astropyconst20.py
--- a/astropy/constants/astropyconst20.py
+++ b/astropy/constants/astropyconst20.py
@@ -3,15 +3,12 @@
 Astronomical and physics constants for Astropy v2.0.  See :mod:`astropy.constants`
 for a complete listing of constants defined in Astropy.
 """
-
-
-
-import itertools
-
-from .constant import Constant
+import inspect
+from . import utils as _utils
 from . import codata2014, iau2015
 
-for _nm, _c in itertools.chain(sorted(vars(codata2014).items()),
-                               sorted(vars(iau2015).items())):
-    if (isinstance(_c, Constant) and _c.abbrev not in locals()):
-        locals()[_c.abbrev] = _c
+_utils._set_c(codata2014, iau2015, inspect.getmodule(inspect.currentframe()))
+
+# Clean up namespace
+del inspect
+del _utils
diff --git a/astropy/constants/utils.py b/astropy/constants/utils.py
new file mode 100644
--- /dev/null
+++ b/astropy/constants/utils.py
@@ -0,0 +1,80 @@
+# Licensed under a 3-clause BSD style license - see LICENSE.rst
+"""Utility functions for ``constants`` sub-package."""
+import itertools
+
+__all__ = []
+
+
+def _get_c(codata, iaudata, module, not_in_module_only=True):
+    """
+    Generator to return a Constant object.
+
+    Parameters
+    ----------
+    codata, iaudata : obj
+        Modules containing CODATA and IAU constants of interest.
+
+    module : obj
+        Namespace module of interest.
+
+    not_in_module_only : bool
+        If ``True``, ignore constants that are already in the
+        namespace of ``module``.
+
+    Returns
+    -------
+    _c : Constant
+        Constant object to process.
+
+    """
+    from .constant import Constant
+
+    for _nm, _c in itertools.chain(sorted(vars(codata).items()),
+                                   sorted(vars(iaudata).items())):
+        if not isinstance(_c, Constant):
+            continue
+        elif (not not_in_module_only) or (_c.abbrev not in module.__dict__):
+            yield _c
+
+
+def _set_c(codata, iaudata, module, not_in_module_only=True, doclines=None,
+           set_class=False):
+    """
+    Set constants in a given module namespace.
+
+    Parameters
+    ----------
+    codata, iaudata : obj
+        Modules containing CODATA and IAU constants of interest.
+
+    module : obj
+        Namespace module to modify with the given ``codata`` and ``iaudata``.
+
+    not_in_module_only : bool
+        If ``True``, constants that are already in the namespace
+        of ``module`` will not be modified.
+
+    doclines : list or `None`
+        If a list is given, this list will be modified in-place to include
+        documentation of modified constants. This can be used to update
+        docstring of ``module``.
+
+    set_class : bool
+        Namespace of ``module`` is populated with ``_c.__class__``
+        instead of just ``_c`` from :func:`_get_c`.
+
+    """
+    for _c in _get_c(codata, iaudata, module,
+                     not_in_module_only=not_in_module_only):
+        if set_class:
+            value = _c.__class__(_c.abbrev, _c.name, _c.value,
+                                 _c._unit_string, _c.uncertainty,
+                                 _c.reference)
+        else:
+            value = _c
+
+        setattr(module, _c.abbrev, value)
+
+        if doclines is not None:
+            doclines.append('{0:^10} {1:^14.9g} {2:^16} {3}'.format(
+                _c.abbrev, _c.value, _c._unit_string, _c.name))

```

## Test Patch

```diff
diff --git a/astropy/constants/tests/test_prior_version.py b/astropy/constants/tests/test_prior_version.py
--- a/astropy/constants/tests/test_prior_version.py
+++ b/astropy/constants/tests/test_prior_version.py
@@ -1,7 +1,5 @@
 # Licensed under a 3-clause BSD style license - see LICENSE.rst
 
-
-
 import copy
 
 import pytest
@@ -155,3 +153,16 @@ def test_view():
 
     c4 = Q(c, subok=True, copy=False)
     assert c4 is c
+
+
+def test_context_manager():
+    from ... import constants as const
+
+    with const.set_enabled_constants('astropyconst13'):
+        assert const.h.value == 6.62606957e-34  # CODATA2010
+
+    assert const.h.value == 6.626070040e-34  # CODATA2014
+
+    with pytest.raises(ValueError):
+        with const.set_enabled_constants('notreal'):
+            const.h

```


## Code snippets

### 1 - astropy/constants/astropyconst13.py:

Start line: 10, End line: 19

```python
import itertools

from .constant import Constant
from . import codata2010, iau2012

for _nm, _c in itertools.chain(sorted(vars(codata2010).items()),
                               sorted(vars(iau2012).items())):
    if (isinstance(_c, Constant) and _c.abbrev not in locals()):
        locals()[_c.abbrev] = _c
```
### 2 - astropy/constants/astropyconst20.py:

Start line: 9, End line: 18

```python
import itertools

from .constant import Constant
from . import codata2014, iau2015

for _nm, _c in itertools.chain(sorted(vars(codata2014).items()),
                               sorted(vars(iau2015).items())):
    if (isinstance(_c, Constant) and _c.abbrev not in locals()):
        locals()[_c.abbrev] = _c
```
### 3 - astropy/constants/cgs.py:

Start line: 7, End line: 17

```python
import itertools

from .constant import Constant
from . import codata2014, iau2015

for _nm, _c in itertools.chain(sorted(vars(codata2014).items()),
                               sorted(vars(iau2015).items())):
    if (isinstance(_c, Constant) and _c.abbrev not in locals()
         and _c.system in ['esu', 'gauss', 'emu']):
        locals()[_c.abbrev] = _c
```
### 4 - astropy/utils/misc.py:

Start line: 826, End line: 860

```python
LOCALE_LOCK = threading.Lock()


@contextmanager
def set_locale(name):
    """
    Context manager to temporarily set the locale to ``name``.

    An example is setting locale to "C" so that the C strtod()
    function will use "." as the decimal point to enable consistent
    numerical string parsing.

    Note that one cannot nest multiple set_locale() context manager
    statements as this causes a threading lock.

    This code taken from https://stackoverflow.com/questions/18593661/how-do-i-strftime-a-date-object-in-a-different-locale.

    Parameters
    ==========
    name : str
        Locale name, e.g. "C" or "fr_FR".
    """
    name = str(name)

    with LOCALE_LOCK:
        saved = locale.setlocale(locale.LC_ALL)
        if saved == name:
            # Don't do anything if locale is already the requested locale
            yield
        else:
            try:
                locale.setlocale(locale.LC_ALL, name)
                yield
            finally:
                locale.setlocale(locale.LC_ALL, saved)
```
### 5 - astropy/constants/constant.py:

Start line: 76, End line: 118

```python
class Constant(Quantity, metaclass=ConstantMeta):
    """A physical or astronomical constant.

    These objects are quantities that are meant to represent physical
    constants.
    """
    _registry = {}
    _has_incompatible_units = set()

    def __new__(cls, abbrev, name, value, unit, uncertainty,
                reference=None, system=None):
        if reference is None:
            reference = getattr(cls, 'default_reference', None)
            if reference is None:
                raise TypeError("{} requires a reference.".format(cls))
        name_lower = name.lower()
        instances = cls._registry.setdefault(name_lower, {})
        # By-pass Quantity initialization, since units may not yet be
        # initialized here, and we store the unit in string form.
        inst = np.array(value).view(cls)

        if system in instances:
                warnings.warn('Constant {0!r} already has a definition in the '
                              '{1!r} system from {2!r} reference'.format(
                              name, system, reference), AstropyUserWarning)
        for c in instances.values():
            if system is not None and not hasattr(c.__class__, system):
                setattr(c, system, inst)
            if c.system is not None and not hasattr(inst.__class__, c.system):
                setattr(inst, c.system, c)

        instances[system] = inst

        inst._abbrev = abbrev
        inst._name = name
        inst._value = value
        inst._unit_string = unit
        inst._uncertainty = uncertainty
        inst._reference = reference
        inst._system = system

        inst._checked_units = False
        return inst
```
### 6 - astropy/constants/si.py:

Start line: 9, End line: 19

```python
import itertools

from .constant import Constant
from . import codata2014, iau2015

for _nm, _c in itertools.chain(sorted(vars(codata2014).items()),
                               sorted(vars(iau2015).items())):
    if (isinstance(_c, Constant) and _c.abbrev not in locals()
         and _c.system == 'si'):
        locals()[_c.abbrev] = _c
```
### 7 - astropy/constants/__init__.py:

Start line: 17, End line: 55

```python
import itertools

# Hack to make circular imports with units work
try:
    from .. import units
    del units
except ImportError:
    pass

from .constant import Constant, EMConstant
from . import si
from . import cgs
from . import codata2014, iau2015

# for updating the constants module docstring
_lines = [
    'The following constants are available:\n',
    '========== ============== ================ =========================',
    '   Name        Value            Unit       Description',
    '========== ============== ================ =========================',
]

for _nm, _c in itertools.chain(sorted(vars(codata2014).items()),
                               sorted(vars(iau2015).items())):
    if isinstance(_c, Constant) and _c.abbrev not in locals():
        locals()[_c.abbrev] = _c.__class__(_c.abbrev, _c.name, _c.value,
                                           _c._unit_string, _c.uncertainty,
                                           _c.reference)

        _lines.append('{0:^10} {1:^14.9g} {2:^16} {3}'.format(
            _c.abbrev, _c.value, _c._unit_string, _c.name))

_lines.append(_lines[1])

if __doc__ is not None:
    __doc__ += '\n'.join(_lines)

del _lines, _nm, _c
```
### 8 - astropy/config/paths.py:

Start line: 195, End line: 237

```python
class set_temp_config(_SetTempPath):
    """
    Context manager to set a temporary path for the Astropy config, primarily
    for use with testing.

    If the path set by this context manager does not already exist it will be
    created, if possible.

    This may also be used as a decorator on a function to set the config path
    just within that function.

    Parameters
    ----------

    path : str, optional
        The directory (which must exist) in which to find the Astropy config
        files, or create them if they do not already exist.  If None, this
        restores the config path to the user's default config path as returned
        by `get_config_dir` as though this context manager were not in effect
        (this is useful for testing).  In this case the ``delete`` argument is
        always ignored.

    delete : bool, optional
        If True, cleans up the temporary directory after exiting the temp
        context (default: False).
    """

    _default_path_getter = staticmethod(get_config_dir)

    def __enter__(self):
        # Special case for the config case, where we need to reset all the
        # cached config objects
        from .configuration import _cfgobjs

        path = super().__enter__()
        _cfgobjs.clear()
        return path

    def __exit__(self, *args):
        from .configuration import _cfgobjs

        super().__exit__(*args)
        _cfgobjs.clear()
```
### 9 - astropy/constants/constant.py:

Start line: 33, End line: 73

```python
class ConstantMeta(InheritDocstrings):

    def __new__(mcls, name, bases, d):
        def wrap(meth):
            @functools.wraps(meth)
            def wrapper(self, *args, **kwargs):
                name_lower = self.name.lower()
                instances = self._registry[name_lower]
                if not self._checked_units:
                    for inst in instances.values():
                        try:
                            self.unit.to(inst.unit)
                        except UnitsError:
                            self._has_incompatible_units.add(name_lower)
                    self._checked_units = True

                if (not self.system and
                        name_lower in self._has_incompatible_units):
                    systems = sorted([x for x in instances if x])
                    raise TypeError(
                        'Constant {0!r} does not have physically compatible '
                        'units across all systems of units and cannot be '
                        'combined with other values without specifying a '
                        'system (eg. {1}.{2})'.format(self.abbrev, self.abbrev,
                                                      systems[0]))

                return meth(self, *args, **kwargs)

            return wrapper

        # The wrapper applies to so many of the __ methods that it's easier to
        # just exclude the ones it doesn't apply to
        exclude = set(['__new__', '__array_finalize__', '__array_wrap__',
                       '__dir__', '__getattr__', '__init__', '__str__',
                       '__repr__', '__hash__', '__iter__', '__getitem__',
                       '__len__', '__bool__', '__quantity_subclass__'])
        for attr, value in vars(Quantity).items():
            if (isinstance(value, types.FunctionType) and
                    attr.startswith('__') and attr.endswith('__') and
                    attr not in exclude):
                d[attr] = wrap(value)

        return super().__new__(mcls, name, bases, d)
```
### 10 - astropy/constants/constant.py:

Start line: 120, End line: 217

```python
class Constant(Quantity, metaclass=ConstantMeta):

    def __repr__(self):
        return ('<{0} name={1!r} value={2} uncertainty={3} unit={4!r} '
                'reference={5!r}>'.format(self.__class__, self.name, self.value,
                                          self.uncertainty, str(self.unit),
                                          self.reference))

    def __str__(self):
        return ('  Name   = {0}\n'
                '  Value  = {1}\n'
                '  Uncertainty  = {2}\n'
                '  Unit  = {3}\n'
                '  Reference = {4}'.format(self.name, self.value,
                                           self.uncertainty, self.unit,
                                           self.reference))

    def __quantity_subclass__(self, unit):
        return super().__quantity_subclass__(unit)[0], False

    def copy(self):
        """
        Return a copy of this `Constant` instance.  Since they are by
        definition immutable, this merely returns another reference to
        ``self``.
        """
        return self
    __deepcopy__ = __copy__ = copy

    @property
    def abbrev(self):
        """A typical ASCII text abbreviation of the constant, also generally
        the same as the Python variable used for this constant.
        """

        return self._abbrev

    @property
    def name(self):
        """The full name of the constant."""

        return self._name

    @lazyproperty
    def _unit(self):
        """The unit(s) in which this constant is defined."""

        return Unit(self._unit_string)

    @property
    def uncertainty(self):
        """The known uncertainty in this constant's value."""

        return self._uncertainty

    @property
    def reference(self):
        """The source used for the value of this constant."""

        return self._reference

    @property
    def system(self):
        """The system of units in which this constant is defined (typically
        `None` so long as the constant's units can be directly converted
        between systems).
        """

        return self._system

    def _instance_or_super(self, key):
        instances = self._registry[self.name.lower()]
        inst = instances.get(key)
        if inst is not None:
            return inst
        else:
            return getattr(super(), key)

    @property
    def si(self):
        """If the Constant is defined in the SI system return that instance of
        the constant, else convert to a Quantity in the appropriate SI units.
        """

        return self._instance_or_super('si')

    @property
    def cgs(self):
        """If the Constant is defined in the CGS system return that instance of
        the constant, else convert to a Quantity in the appropriate CGS units.
        """

        return self._instance_or_super('cgs')

    def __array_finalize__(self, obj):
        for attr in ('_abbrev', '_name', '_value', '_unit_string',
                     '_uncertainty', '_reference', '_system'):
            setattr(self, attr, getattr(obj, attr, None))

        self._checked_units = getattr(obj, '_checked_units', False)
```
