# pylint-dev__astroid-1333

| **pylint-dev/astroid** | `d2a5b3c7b1e203fec3c7ca73c30eb1785d3d4d0a` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 6194 |
| **Any found context length** | 6194 |
| **Avg pos** | 4.0 |
| **Min pos** | 2 |
| **Max pos** | 2 |
| **Top file pos** | 2 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/astroid/modutils.py b/astroid/modutils.py
--- a/astroid/modutils.py
+++ b/astroid/modutils.py
@@ -297,6 +297,9 @@ def _get_relative_base_path(filename, path_to_check):
     if os.path.normcase(real_filename).startswith(path_to_check):
         importable_path = real_filename
 
+    # if "var" in path_to_check:
+    #     breakpoint()
+
     if importable_path:
         base_path = os.path.splitext(importable_path)[0]
         relative_base_path = base_path[len(path_to_check) :]
@@ -307,8 +310,11 @@ def _get_relative_base_path(filename, path_to_check):
 
 def modpath_from_file_with_callback(filename, path=None, is_package_cb=None):
     filename = os.path.expanduser(_path_from_filename(filename))
+    paths_to_check = sys.path.copy()
+    if path:
+        paths_to_check += path
     for pathname in itertools.chain(
-        path or [], map(_cache_normalize_path, sys.path), sys.path
+        paths_to_check, map(_cache_normalize_path, paths_to_check)
     ):
         if not pathname:
             continue

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| astroid/modutils.py | 297 | - | 2 | 2 | 6194
| astroid/modutils.py | 311 | 314 | 2 | 2 | 6194


## Problem Statement

```
astroid 2.9.1 breaks pylint with missing __init__.py: F0010: error while code parsing: Unable to load file __init__.py
### Steps to reproduce
> Steps provided are for Windows 11, but initial problem found in Ubuntu 20.04

> Update 2022-01-04: Corrected repro steps and added more environment details

1. Set up simple repo with following structure (all files can be empty):
\`\`\`
root_dir/
|--src/
|----project/ # Notice the missing __init__.py
|------file.py # It can be empty, but I added `import os` at the top
|----__init__.py
\`\`\`
2. Open a command prompt
3. `cd root_dir`
4. `python -m venv venv`
5. `venv/Scripts/activate`
6. `pip install pylint astroid==2.9.1` # I also repro'd on the latest, 2.9.2
7. `pylint src/project` # Updated from `pylint src`
8. Observe failure:
\`\`\`
src\project\__init__.py:1:0: F0010: error while code parsing: Unable to load file src\project\__init__.py:
\`\`\`

### Current behavior
Fails with `src\project\__init__.py:1:0: F0010: error while code parsing: Unable to load file src\project\__init__.py:`

### Expected behavior
Does not fail with error.
> If you replace step 6 with `pip install pylint astroid==2.9.0`, you get no failure with an empty output - since no files have content

### `python -c "from astroid import __pkginfo__; print(__pkginfo__.version)"` output
2.9.1

`python 3.9.1`
`pylint 2.12.2 `



This issue has been observed with astroid `2.9.1` and `2.9.2`

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- |
| 1 | 1 astroid/__pkginfo__.py | 0 | 29| 543 | 543 | 
| **-> 2 <-** | **2 astroid/modutils.py** | 0 | 675| 5651 | 6194 | 
| 3 | 3 astroid/_ast.py | 0 | 127| 811 | 7005 | 
| 4 | 4 astroid/raw_building.py | 0 | 500| 4127 | 11132 | 
| 5 | 5 astroid/const.py | 0 | 22| 148 | 11280 | 


### Hint

```
I can't seem to reproduce this in my `virtualenv`. This might be specific to `venv`? Needs some further investigation.
@interifter Which version of `pylint` are you using?
Right, ``pip install pylint astroid==2.9.0``, will keep the local version if you already have one, so I thought it was ``2.12.2`` but that could be false. In fact it probably isn't 2.12.2. For the record, you're not supposed to set the version of ``astroid`` yourself, pylint does, and bad thing will happen if you try to set the version of an incompatible astroid. We might want to update the issue's template to have this information next.
My apologies... I updated the repro steps with a critical missed detail: `pylint src/project`, instead of `pylint src`

But I verified that either with, or without, `venv`, the issue is reproduced.

Also, I never have specified the `astroid` version, before. 

However, this isn't the first time the issue has been observed.
Back in early 2019, a [similar issue](https://stackoverflow.com/questions/48024049/pylint-raises-error-if-directory-doesnt-contain-init-py-file) was observed with either `astroid 2.2.0` or `isort 4.3.5`, which led me to try pinning `astroid==2.9.0`, which worked.
> @interifter Which version of `pylint` are you using?

`2.12.2`

Full env info:

\`\`\`
Package           Version
----------------- -------
astroid           2.9.2
colorama          0.4.4
isort             5.10.1
lazy-object-proxy 1.7.1
mccabe            0.6.1
pip               20.2.3
platformdirs      2.4.1
pylint            2.12.2
setuptools        49.2.1
toml              0.10.2
typing-extensions 4.0.1
wrapt             1.13.3
\`\`\`

I confirm the bug and i'm able to reproduce it with `python 3.9.1`. 
\`\`\`
$> pip freeze
astroid==2.9.2
isort==5.10.1
lazy-object-proxy==1.7.1
mccabe==0.6.1
platformdirs==2.4.1
pylint==2.12.2
toml==0.10.2
typing-extensions==4.0.1
wrapt==1.13.3
\`\`\`
Bisected and this is the faulty commit:
https://github.com/PyCQA/astroid/commit/2ee20ccdf62450db611acc4a1a7e42f407ce8a14
Fix in #1333, no time to write tests yet so if somebody has any good ideas: please let me know!
```

## Patch

```diff
diff --git a/astroid/modutils.py b/astroid/modutils.py
--- a/astroid/modutils.py
+++ b/astroid/modutils.py
@@ -297,6 +297,9 @@ def _get_relative_base_path(filename, path_to_check):
     if os.path.normcase(real_filename).startswith(path_to_check):
         importable_path = real_filename
 
+    # if "var" in path_to_check:
+    #     breakpoint()
+
     if importable_path:
         base_path = os.path.splitext(importable_path)[0]
         relative_base_path = base_path[len(path_to_check) :]
@@ -307,8 +310,11 @@ def _get_relative_base_path(filename, path_to_check):
 
 def modpath_from_file_with_callback(filename, path=None, is_package_cb=None):
     filename = os.path.expanduser(_path_from_filename(filename))
+    paths_to_check = sys.path.copy()
+    if path:
+        paths_to_check += path
     for pathname in itertools.chain(
-        path or [], map(_cache_normalize_path, sys.path), sys.path
+        paths_to_check, map(_cache_normalize_path, paths_to_check)
     ):
         if not pathname:
             continue

```

## Test Patch

```diff
diff --git a/tests/unittest_modutils.py b/tests/unittest_modutils.py
--- a/tests/unittest_modutils.py
+++ b/tests/unittest_modutils.py
@@ -30,6 +30,7 @@
 import tempfile
 import unittest
 import xml
+from pathlib import Path
 from xml import etree
 from xml.etree import ElementTree
 
@@ -189,6 +190,30 @@ def test_load_from_module_symlink_on_symlinked_paths_in_syspath(self) -> None:
         # this should be equivalent to: import secret
         self.assertEqual(modutils.modpath_from_file(symlink_secret_path), ["secret"])
 
+    def test_load_packages_without_init(self) -> None:
+        """Test that we correctly find packages with an __init__.py file.
+
+        Regression test for issue reported in:
+        https://github.com/PyCQA/astroid/issues/1327
+        """
+        tmp_dir = Path(tempfile.gettempdir())
+        self.addCleanup(os.chdir, os.curdir)
+        os.chdir(tmp_dir)
+
+        self.addCleanup(shutil.rmtree, tmp_dir / "src")
+        os.mkdir(tmp_dir / "src")
+        os.mkdir(tmp_dir / "src" / "package")
+        with open(tmp_dir / "src" / "__init__.py", "w", encoding="utf-8"):
+            pass
+        with open(tmp_dir / "src" / "package" / "file.py", "w", encoding="utf-8"):
+            pass
+
+        # this should be equivalent to: import secret
+        self.assertEqual(
+            modutils.modpath_from_file(str(Path("src") / "package"), ["."]),
+            ["src", "package"],
+        )
+
 
 class LoadModuleFromPathTest(resources.SysPathSetup, unittest.TestCase):
     def test_do_not_load_twice(self) -> None:

```


## Code snippets

### 1 - astroid/__pkginfo__.py:

```python
# Copyright (c) 2006-2014 LOGILAB S.A. (Paris, FRANCE) <contact@logilab.fr>
# Copyright (c) 2014-2020 Claudiu Popa <pcmanticore@gmail.com>
# Copyright (c) 2014 Google, Inc.
# Copyright (c) 2015-2017 Ceridwen <ceridwenv@gmail.com>
# Copyright (c) 2015 Florian Bruhin <me@the-compiler.org>
# Copyright (c) 2015 Radosław Ganczarek <radoslaw@ganczarek.in>
# Copyright (c) 2016 Moises Lopez <moylop260@vauxoo.com>
# Copyright (c) 2017 Hugo <hugovk@users.noreply.github.com>
# Copyright (c) 2017 Łukasz Rogalski <rogalski.91@gmail.com>
# Copyright (c) 2017 Calen Pennington <cale@edx.org>
# Copyright (c) 2018 Ville Skyttä <ville.skytta@iki.fi>
# Copyright (c) 2018 Ashley Whetter <ashley@awhetter.co.uk>
# Copyright (c) 2018 Bryce Guinta <bryce.paul.guinta@gmail.com>
# Copyright (c) 2019 Uilian Ries <uilianries@gmail.com>
# Copyright (c) 2019 Thomas Hisch <t.hisch@gmail.com>
# Copyright (c) 2020-2021 hippo91 <guillaume.peillex@gmail.com>
# Copyright (c) 2020 David Gilman <davidgilman1@gmail.com>
# Copyright (c) 2020 Konrad Weihmann <kweihmann@outlook.com>
# Copyright (c) 2020 Felix Mölder <felix.moelder@uni-due.de>
# Copyright (c) 2020 Michael <michael-k@users.noreply.github.com>
# Copyright (c) 2021 Pierre Sassoulas <pierre.sassoulas@gmail.com>
# Copyright (c) 2021 Marc Mueller <30130371+cdce8p@users.noreply.github.com>

# Licensed under the LGPL: https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html
# For details: https://github.com/PyCQA/astroid/blob/main/LICENSE

__version__ = "2.10.0-dev0"
version = __version__

```
### 2 - astroid/modutils.py:

```python
# Copyright (c) 2014-2018, 2020 Claudiu Popa <pcmanticore@gmail.com>
# Copyright (c) 2014 Google, Inc.
# Copyright (c) 2014 Denis Laxalde <denis.laxalde@logilab.fr>
# Copyright (c) 2014 LOGILAB S.A. (Paris, FRANCE) <contact@logilab.fr>
# Copyright (c) 2014 Eevee (Alex Munroe) <amunroe@yelp.com>
# Copyright (c) 2015 Florian Bruhin <me@the-compiler.org>
# Copyright (c) 2015 Radosław Ganczarek <radoslaw@ganczarek.in>
# Copyright (c) 2016 Derek Gustafson <degustaf@gmail.com>
# Copyright (c) 2016 Jakub Wilk <jwilk@jwilk.net>
# Copyright (c) 2016 Ceridwen <ceridwenv@gmail.com>
# Copyright (c) 2018 Ville Skyttä <ville.skytta@iki.fi>
# Copyright (c) 2018 Mario Corchero <mcorcherojim@bloomberg.net>
# Copyright (c) 2018 Mario Corchero <mariocj89@gmail.com>
# Copyright (c) 2018 Anthony Sottile <asottile@umich.edu>
# Copyright (c) 2019 Hugo van Kemenade <hugovk@users.noreply.github.com>
# Copyright (c) 2019 markmcclain <markmcclain@users.noreply.github.com>
# Copyright (c) 2019 BasPH <BasPH@users.noreply.github.com>
# Copyright (c) 2020-2021 hippo91 <guillaume.peillex@gmail.com>
# Copyright (c) 2020 Peter Kolbus <peter.kolbus@gmail.com>
# Copyright (c) 2021 Pierre Sassoulas <pierre.sassoulas@gmail.com>
# Copyright (c) 2021 Daniël van Noord <13665637+DanielNoord@users.noreply.github.com>
# Copyright (c) 2021 Keichi Takahashi <hello@keichi.dev>
# Copyright (c) 2021 Nick Drozd <nicholasdrozd@gmail.com>
# Copyright (c) 2021 Marc Mueller <30130371+cdce8p@users.noreply.github.com>
# Copyright (c) 2021 DudeNr33 <3929834+DudeNr33@users.noreply.github.com>

# Licensed under the LGPL: https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html
# For details: https://github.com/PyCQA/astroid/blob/main/LICENSE

"""Python modules manipulation utility functions.

:type PY_SOURCE_EXTS: tuple(str)
:var PY_SOURCE_EXTS: list of possible python source file extension

:type STD_LIB_DIRS: set of str
:var STD_LIB_DIRS: directories where standard modules are located

:type BUILTIN_MODULES: dict
:var BUILTIN_MODULES: dictionary with builtin module names has key
"""

# We disable the import-error so pylint can work without distutils installed.
# pylint: disable=no-name-in-module,useless-suppression

import importlib
import importlib.machinery
import importlib.util
import itertools
import os
import platform
import sys
import types
from distutils.errors import DistutilsPlatformError  # pylint: disable=import-error
from distutils.sysconfig import get_python_lib  # pylint: disable=import-error
from typing import Dict, Set

from astroid.interpreter._import import spec, util

# distutils is replaced by virtualenv with a module that does
# weird path manipulations in order to get to the
# real distutils module.


if sys.platform.startswith("win"):
    PY_SOURCE_EXTS = ("py", "pyw")
    PY_COMPILED_EXTS = ("dll", "pyd")
else:
    PY_SOURCE_EXTS = ("py",)
    PY_COMPILED_EXTS = ("so",)


try:
    # The explicit sys.prefix is to work around a patch in virtualenv that
    # replaces the 'real' sys.prefix (i.e. the location of the binary)
    # with the prefix from which the virtualenv was created. This throws
    # off the detection logic for standard library modules, thus the
    # workaround.
    STD_LIB_DIRS = {
        get_python_lib(standard_lib=True, prefix=sys.prefix),
        # Take care of installations where exec_prefix != prefix.
        get_python_lib(standard_lib=True, prefix=sys.exec_prefix),
        get_python_lib(standard_lib=True),
    }
# get_python_lib(standard_lib=1) is not available on pypy, set STD_LIB_DIR to
# non-valid path, see https://bugs.pypy.org/issue1164
except DistutilsPlatformError:
    STD_LIB_DIRS = set()

if os.name == "nt":
    STD_LIB_DIRS.add(os.path.join(sys.prefix, "dlls"))
    try:
        # real_prefix is defined when running inside virtual environments,
        # created with the **virtualenv** library.
        # Deprecated in virtualenv==16.7.9
        # See: https://github.com/pypa/virtualenv/issues/1622
        STD_LIB_DIRS.add(os.path.join(sys.real_prefix, "dlls"))  # type: ignore[attr-defined]
    except AttributeError:
        # sys.base_exec_prefix is always defined, but in a virtual environment
        # created with the stdlib **venv** module, it points to the original
        # installation, if the virtual env is activated.
        try:
            STD_LIB_DIRS.add(os.path.join(sys.base_exec_prefix, "dlls"))
        except AttributeError:
            pass

if platform.python_implementation() == "PyPy":
    # The get_python_lib(standard_lib=True) function does not give valid
    # result with pypy in a virtualenv.
    # In a virtual environment, with CPython implementation the call to this function returns a path toward
    # the binary (its libraries) which has been used to create the virtual environment.
    # Not with pypy implementation.
    # The only way to retrieve such information is to use the sys.base_prefix hint.
    # It's worth noticing that under CPython implementation the return values of
    # get_python_lib(standard_lib=True) and get_python_lib(santdard_lib=True, prefix=sys.base_prefix)
    # are the same.
    # In the lines above, we could have replace the call to get_python_lib(standard=True)
    # with the one using prefix=sys.base_prefix but we prefer modifying only what deals with pypy.
    STD_LIB_DIRS.add(get_python_lib(standard_lib=True, prefix=sys.base_prefix))
    _root = os.path.join(sys.prefix, "lib_pypy")
    STD_LIB_DIRS.add(_root)
    try:
        # real_prefix is defined when running inside virtualenv.
        STD_LIB_DIRS.add(os.path.join(sys.base_prefix, "lib_pypy"))
    except AttributeError:
        pass
    del _root
if os.name == "posix":
    # Need the real prefix if we're in a virtualenv, otherwise
    # the usual one will do.
    # Deprecated in virtualenv==16.7.9
    # See: https://github.com/pypa/virtualenv/issues/1622
    try:
        prefix = sys.real_prefix  # type: ignore[attr-defined]
    except AttributeError:
        prefix = sys.prefix

    def _posix_path(path):
        base_python = "python%d.%d" % sys.version_info[:2]
        return os.path.join(prefix, path, base_python)

    STD_LIB_DIRS.add(_posix_path("lib"))
    if sys.maxsize > 2 ** 32:
        # This tries to fix a problem with /usr/lib64 builds,
        # where systems are running both 32-bit and 64-bit code
        # on the same machine, which reflects into the places where
        # standard library could be found. More details can be found
        # here http://bugs.python.org/issue1294959.
        # An easy reproducing case would be
        # https://github.com/PyCQA/pylint/issues/712#issuecomment-163178753
        STD_LIB_DIRS.add(_posix_path("lib64"))

EXT_LIB_DIRS = {get_python_lib(), get_python_lib(True)}
IS_JYTHON = platform.python_implementation() == "Jython"
BUILTIN_MODULES = dict.fromkeys(sys.builtin_module_names, True)


class NoSourceFile(Exception):
    """exception raised when we are not able to get a python
    source file for a precompiled file
    """


def _normalize_path(path: str) -> str:
    """Resolve symlinks in path and convert to absolute path.

    Note that environment variables and ~ in the path need to be expanded in
    advance.

    This can be cached by using _cache_normalize_path.
    """
    return os.path.normcase(os.path.realpath(path))


def _path_from_filename(filename, is_jython=IS_JYTHON):
    if not is_jython:
        return filename
    head, has_pyclass, _ = filename.partition("$py.class")
    if has_pyclass:
        return head + ".py"
    return filename


def _handle_blacklist(blacklist, dirnames, filenames):
    """remove files/directories in the black list

    dirnames/filenames are usually from os.walk
    """
    for norecurs in blacklist:
        if norecurs in dirnames:
            dirnames.remove(norecurs)
        elif norecurs in filenames:
            filenames.remove(norecurs)


_NORM_PATH_CACHE: Dict[str, str] = {}


def _cache_normalize_path(path: str) -> str:
    """Normalize path with caching."""
    # _module_file calls abspath on every path in sys.path every time it's
    # called; on a larger codebase this easily adds up to half a second just
    # assembling path components. This cache alleviates that.
    try:
        return _NORM_PATH_CACHE[path]
    except KeyError:
        if not path:  # don't cache result for ''
            return _normalize_path(path)
        result = _NORM_PATH_CACHE[path] = _normalize_path(path)
        return result


def load_module_from_name(dotted_name: str) -> types.ModuleType:
    """Load a Python module from its name.

    :type dotted_name: str
    :param dotted_name: python name of a module or package

    :raise ImportError: if the module or package is not found

    :rtype: module
    :return: the loaded module
    """
    try:
        return sys.modules[dotted_name]
    except KeyError:
        pass

    return importlib.import_module(dotted_name)


def load_module_from_modpath(parts):
    """Load a python module from its split name.

    :type parts: list(str) or tuple(str)
    :param parts:
      python name of a module or package split on '.'

    :raise ImportError: if the module or package is not found

    :rtype: module
    :return: the loaded module
    """
    return load_module_from_name(".".join(parts))


def load_module_from_file(filepath: str):
    """Load a Python module from it's path.

    :type filepath: str
    :param filepath: path to the python module or package

    :raise ImportError: if the module or package is not found

    :rtype: module
    :return: the loaded module
    """
    modpath = modpath_from_file(filepath)
    return load_module_from_modpath(modpath)


def check_modpath_has_init(path, mod_path):
    """check there are some __init__.py all along the way"""
    modpath = []
    for part in mod_path:
        modpath.append(part)
        path = os.path.join(path, part)
        if not _has_init(path):
            old_namespace = util.is_namespace(".".join(modpath))
            if not old_namespace:
                return False
    return True


def _get_relative_base_path(filename, path_to_check):
    """Extracts the relative mod path of the file to import from

    Check if a file is within the passed in path and if so, returns the
    relative mod path from the one passed in.

    If the filename is no in path_to_check, returns None

    Note this function will look for both abs and realpath of the file,
    this allows to find the relative base path even if the file is a
    symlink of a file in the passed in path

    Examples:
        _get_relative_base_path("/a/b/c/d.py", "/a/b") ->  ["c","d"]
        _get_relative_base_path("/a/b/c/d.py", "/dev") ->  None
    """
    importable_path = None
    path_to_check = os.path.normcase(path_to_check)
    abs_filename = os.path.abspath(filename)
    if os.path.normcase(abs_filename).startswith(path_to_check):
        importable_path = abs_filename

    real_filename = os.path.realpath(filename)
    if os.path.normcase(real_filename).startswith(path_to_check):
        importable_path = real_filename

    if importable_path:
        base_path = os.path.splitext(importable_path)[0]
        relative_base_path = base_path[len(path_to_check) :]
        return [pkg for pkg in relative_base_path.split(os.sep) if pkg]

    return None


def modpath_from_file_with_callback(filename, path=None, is_package_cb=None):
    filename = os.path.expanduser(_path_from_filename(filename))
    for pathname in itertools.chain(
        path or [], map(_cache_normalize_path, sys.path), sys.path
    ):
        if not pathname:
            continue
        modpath = _get_relative_base_path(filename, pathname)
        if not modpath:
            continue
        if is_package_cb(pathname, modpath[:-1]):
            return modpath

    raise ImportError(
        "Unable to find module for {} in {}".format(filename, ", \n".join(sys.path))
    )


def modpath_from_file(filename, path=None):
    """Get the corresponding split module's name from a filename

    This function will return the name of a module or package split on `.`.

    :type filename: str
    :param filename: file's path for which we want the module's name

    :type Optional[List[str]] path:
      Optional list of path where the module or package should be
      searched (use sys.path if nothing or None is given)

    :raise ImportError:
      if the corresponding module's name has not been found

    :rtype: list(str)
    :return: the corresponding split module's name
    """
    return modpath_from_file_with_callback(filename, path, check_modpath_has_init)


def file_from_modpath(modpath, path=None, context_file=None):
    return file_info_from_modpath(modpath, path, context_file).location


def file_info_from_modpath(modpath, path=None, context_file=None):
    """given a mod path (i.e. split module / package name), return the
    corresponding file, giving priority to source file over precompiled
    file if it exists

    :type modpath: list or tuple
    :param modpath:
      split module's name (i.e name of a module or package split
      on '.')
      (this means explicit relative imports that start with dots have
      empty strings in this list!)

    :type path: list or None
    :param path:
      optional list of path where the module or package should be
      searched (use sys.path if nothing or None is given)

    :type context_file: str or None
    :param context_file:
      context file to consider, necessary if the identifier has been
      introduced using a relative import unresolvable in the actual
      context (i.e. modutils)

    :raise ImportError: if there is no such module in the directory

    :rtype: (str or None, import type)
    :return:
      the path to the module's file or None if it's an integrated
      builtin module such as 'sys'
    """
    if context_file is not None:
        context = os.path.dirname(context_file)
    else:
        context = context_file
    if modpath[0] == "xml":
        # handle _xmlplus
        try:
            return _spec_from_modpath(["_xmlplus"] + modpath[1:], path, context)
        except ImportError:
            return _spec_from_modpath(modpath, path, context)
    elif modpath == ["os", "path"]:
        # FIXME: currently ignoring search_path...
        return spec.ModuleSpec(
            name="os.path",
            location=os.path.__file__,
            module_type=spec.ModuleType.PY_SOURCE,
        )
    return _spec_from_modpath(modpath, path, context)


def get_module_part(dotted_name, context_file=None):
    """given a dotted name return the module part of the name :

    >>> get_module_part('astroid.as_string.dump')
    'astroid.as_string'

    :type dotted_name: str
    :param dotted_name: full name of the identifier we are interested in

    :type context_file: str or None
    :param context_file:
      context file to consider, necessary if the identifier has been
      introduced using a relative import unresolvable in the actual
      context (i.e. modutils)


    :raise ImportError: if there is no such module in the directory

    :rtype: str or None
    :return:
      the module part of the name or None if we have not been able at
      all to import the given name

    XXX: deprecated, since it doesn't handle package precedence over module
    (see #10066)
    """
    # os.path trick
    if dotted_name.startswith("os.path"):
        return "os.path"
    parts = dotted_name.split(".")
    if context_file is not None:
        # first check for builtin module which won't be considered latter
        # in that case (path != None)
        if parts[0] in BUILTIN_MODULES:
            if len(parts) > 2:
                raise ImportError(dotted_name)
            return parts[0]
        # don't use += or insert, we want a new list to be created !
    path = None
    starti = 0
    if parts[0] == "":
        assert (
            context_file is not None
        ), "explicit relative import, but no context_file?"
        path = []  # prevent resolving the import non-relatively
        starti = 1
    while parts[starti] == "":  # for all further dots: change context
        starti += 1
        context_file = os.path.dirname(context_file)
    for i in range(starti, len(parts)):
        try:
            file_from_modpath(
                parts[starti : i + 1], path=path, context_file=context_file
            )
        except ImportError:
            if i < max(1, len(parts) - 2):
                raise
            return ".".join(parts[:i])
    return dotted_name


def get_module_files(src_directory, blacklist, list_all=False):
    """given a package directory return a list of all available python
    module's files in the package and its subpackages

    :type src_directory: str
    :param src_directory:
      path of the directory corresponding to the package

    :type blacklist: list or tuple
    :param blacklist: iterable
      list of files or directories to ignore.

    :type list_all: bool
    :param list_all:
        get files from all paths, including ones without __init__.py

    :rtype: list
    :return:
      the list of all available python module's files in the package and
      its subpackages
    """
    files = []
    for directory, dirnames, filenames in os.walk(src_directory):
        if directory in blacklist:
            continue
        _handle_blacklist(blacklist, dirnames, filenames)
        # check for __init__.py
        if not list_all and "__init__.py" not in filenames:
            dirnames[:] = ()
            continue
        for filename in filenames:
            if _is_python_file(filename):
                src = os.path.join(directory, filename)
                files.append(src)
    return files


def get_source_file(filename, include_no_ext=False):
    """given a python module's file name return the matching source file
    name (the filename will be returned identically if it's already an
    absolute path to a python source file...)

    :type filename: str
    :param filename: python module's file name


    :raise NoSourceFile: if no source file exists on the file system

    :rtype: str
    :return: the absolute path of the source file if it exists
    """
    filename = os.path.abspath(_path_from_filename(filename))
    base, orig_ext = os.path.splitext(filename)
    for ext in PY_SOURCE_EXTS:
        source_path = f"{base}.{ext}"
        if os.path.exists(source_path):
            return source_path
    if include_no_ext and not orig_ext and os.path.exists(base):
        return base
    raise NoSourceFile(filename)


def is_python_source(filename):
    """
    rtype: bool
    return: True if the filename is a python source file
    """
    return os.path.splitext(filename)[1][1:] in PY_SOURCE_EXTS


def is_standard_module(modname, std_path=None):
    """try to guess if a module is a standard python module (by default,
    see `std_path` parameter's description)

    :type modname: str
    :param modname: name of the module we are interested in

    :type std_path: list(str) or tuple(str)
    :param std_path: list of path considered has standard


    :rtype: bool
    :return:
      true if the module:
      - is located on the path listed in one of the directory in `std_path`
      - is a built-in module
    """
    modname = modname.split(".")[0]
    try:
        filename = file_from_modpath([modname])
    except ImportError:
        # import failed, i'm probably not so wrong by supposing it's
        # not standard...
        return False
    # modules which are not living in a file are considered standard
    # (sys and __builtin__ for instance)
    if filename is None:
        # we assume there are no namespaces in stdlib
        return not util.is_namespace(modname)
    filename = _normalize_path(filename)
    for path in EXT_LIB_DIRS:
        if filename.startswith(_cache_normalize_path(path)):
            return False
    if std_path is None:
        std_path = STD_LIB_DIRS

    return any(filename.startswith(_cache_normalize_path(path)) for path in std_path)


def is_relative(modname, from_file):
    """return true if the given module name is relative to the given
    file name

    :type modname: str
    :param modname: name of the module we are interested in

    :type from_file: str
    :param from_file:
      path of the module from which modname has been imported

    :rtype: bool
    :return:
      true if the module has been imported relatively to `from_file`
    """
    if not os.path.isdir(from_file):
        from_file = os.path.dirname(from_file)
    if from_file in sys.path:
        return False
    return bool(
        importlib.machinery.PathFinder.find_spec(
            modname.split(".", maxsplit=1)[0], [from_file]
        )
    )


# internal only functions #####################################################


def _spec_from_modpath(modpath, path=None, context=None):
    """given a mod path (i.e. split module / package name), return the
    corresponding spec

    this function is used internally, see `file_from_modpath`'s
    documentation for more information
    """
    assert modpath
    location = None
    if context is not None:
        try:
            found_spec = spec.find_spec(modpath, [context])
            location = found_spec.location
        except ImportError:
            found_spec = spec.find_spec(modpath, path)
            location = found_spec.location
    else:
        found_spec = spec.find_spec(modpath, path)
    if found_spec.type == spec.ModuleType.PY_COMPILED:
        try:
            location = get_source_file(found_spec.location)
            return found_spec._replace(
                location=location, type=spec.ModuleType.PY_SOURCE
            )
        except NoSourceFile:
            return found_spec._replace(location=location)
    elif found_spec.type == spec.ModuleType.C_BUILTIN:
        # integrated builtin module
        return found_spec._replace(location=None)
    elif found_spec.type == spec.ModuleType.PKG_DIRECTORY:
        location = _has_init(found_spec.location)
        return found_spec._replace(location=location, type=spec.ModuleType.PY_SOURCE)
    return found_spec


def _is_python_file(filename):
    """return true if the given filename should be considered as a python file

    .pyc and .pyo are ignored
    """
    return filename.endswith((".py", ".so", ".pyd", ".pyw"))


def _has_init(directory):
    """if the given directory has a valid __init__ file, return its path,
    else return None
    """
    mod_or_pack = os.path.join(directory, "__init__")
    for ext in PY_SOURCE_EXTS + ("pyc", "pyo"):
        if os.path.exists(mod_or_pack + "." + ext):
            return mod_or_pack + "." + ext
    return None


def is_namespace(specobj):
    return specobj.type == spec.ModuleType.PY_NAMESPACE


def is_directory(specobj):
    return specobj.type == spec.ModuleType.PKG_DIRECTORY


def is_module_name_part_of_extension_package_whitelist(
    module_name: str, package_whitelist: Set[str]
) -> bool:
    """
    Returns True if one part of the module name is in the package whitelist

    >>> is_module_name_part_of_extension_package_whitelist('numpy.core.umath', {'numpy'})
    True
    """
    parts = module_name.split(".")
    return any(
        ".".join(parts[:x]) in package_whitelist for x in range(1, len(parts) + 1)
    )

```
### 3 - astroid/_ast.py:

```python
import ast
import sys
import types
from collections import namedtuple
from functools import partial
from typing import Dict, Optional

from astroid.const import PY38_PLUS, Context

if sys.version_info >= (3, 8):
    # On Python 3.8, typed_ast was merged back into `ast`
    _ast_py3: Optional[types.ModuleType] = ast
else:
    try:
        import typed_ast.ast3 as _ast_py3
    except ImportError:
        _ast_py3 = None

FunctionType = namedtuple("FunctionType", ["argtypes", "returns"])


class ParserModule(
    namedtuple(
        "ParserModule",
        [
            "module",
            "unary_op_classes",
            "cmp_op_classes",
            "bool_op_classes",
            "bin_op_classes",
            "context_classes",
        ],
    )
):
    def parse(self, string: str, type_comments=True):
        if self.module is _ast_py3:
            if PY38_PLUS:
                parse_func = partial(self.module.parse, type_comments=type_comments)
            else:
                parse_func = partial(
                    self.module.parse, feature_version=sys.version_info.minor
                )
        else:
            parse_func = self.module.parse
        return parse_func(string)


def parse_function_type_comment(type_comment: str) -> Optional[FunctionType]:
    """Given a correct type comment, obtain a FunctionType object"""
    if _ast_py3 is None:
        return None

    func_type = _ast_py3.parse(type_comment, "<type_comment>", "func_type")  # type: ignore[attr-defined]
    return FunctionType(argtypes=func_type.argtypes, returns=func_type.returns)


def get_parser_module(type_comments=True) -> ParserModule:
    parser_module = ast
    if type_comments and _ast_py3:
        parser_module = _ast_py3

    unary_op_classes = _unary_operators_from_module(parser_module)
    cmp_op_classes = _compare_operators_from_module(parser_module)
    bool_op_classes = _bool_operators_from_module(parser_module)
    bin_op_classes = _binary_operators_from_module(parser_module)
    context_classes = _contexts_from_module(parser_module)

    return ParserModule(
        parser_module,
        unary_op_classes,
        cmp_op_classes,
        bool_op_classes,
        bin_op_classes,
        context_classes,
    )


def _unary_operators_from_module(module):
    return {module.UAdd: "+", module.USub: "-", module.Not: "not", module.Invert: "~"}


def _binary_operators_from_module(module):
    binary_operators = {
        module.Add: "+",
        module.BitAnd: "&",
        module.BitOr: "|",
        module.BitXor: "^",
        module.Div: "/",
        module.FloorDiv: "//",
        module.MatMult: "@",
        module.Mod: "%",
        module.Mult: "*",
        module.Pow: "**",
        module.Sub: "-",
        module.LShift: "<<",
        module.RShift: ">>",
    }
    return binary_operators


def _bool_operators_from_module(module):
    return {module.And: "and", module.Or: "or"}


def _compare_operators_from_module(module):
    return {
        module.Eq: "==",
        module.Gt: ">",
        module.GtE: ">=",
        module.In: "in",
        module.Is: "is",
        module.IsNot: "is not",
        module.Lt: "<",
        module.LtE: "<=",
        module.NotEq: "!=",
        module.NotIn: "not in",
    }


def _contexts_from_module(module) -> Dict[ast.expr_context, Context]:
    return {
        module.Load: Context.Load,
        module.Store: Context.Store,
        module.Del: Context.Del,
        module.Param: Context.Store,
    }

```
### 4 - astroid/raw_building.py:

```python
# Copyright (c) 2006-2014 LOGILAB S.A. (Paris, FRANCE) <contact@logilab.fr>
# Copyright (c) 2012 FELD Boris <lothiraldan@gmail.com>
# Copyright (c) 2014-2020 Claudiu Popa <pcmanticore@gmail.com>
# Copyright (c) 2014 Google, Inc.
# Copyright (c) 2015-2016 Ceridwen <ceridwenv@gmail.com>
# Copyright (c) 2015 Florian Bruhin <me@the-compiler.org>
# Copyright (c) 2015 Ovidiu Sabou <ovidiu@sabou.org>
# Copyright (c) 2016 Derek Gustafson <degustaf@gmail.com>
# Copyright (c) 2016 Jakub Wilk <jwilk@jwilk.net>
# Copyright (c) 2018 Ville Skyttä <ville.skytta@iki.fi>
# Copyright (c) 2018 Nick Drozd <nicholasdrozd@gmail.com>
# Copyright (c) 2018 Bryce Guinta <bryce.paul.guinta@gmail.com>
# Copyright (c) 2020-2021 hippo91 <guillaume.peillex@gmail.com>
# Copyright (c) 2020 Becker Awqatty <bawqatty@mide.com>
# Copyright (c) 2020 Robin Jarry <robin.jarry@6wind.com>
# Copyright (c) 2021 Pierre Sassoulas <pierre.sassoulas@gmail.com>
# Copyright (c) 2021 Daniël van Noord <13665637+DanielNoord@users.noreply.github.com>
# Copyright (c) 2021 Marc Mueller <30130371+cdce8p@users.noreply.github.com>
# Copyright (c) 2021 Andrew Haigh <hello@nelf.in>

# Licensed under the LGPL: https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html
# For details: https://github.com/PyCQA/astroid/blob/main/LICENSE

"""this module contains a set of functions to create astroid trees from scratch
(build_* functions) or from living object (object_build_* functions)
"""

import builtins
import inspect
import os
import sys
import types
import warnings
from typing import List, Optional

from astroid import bases, nodes
from astroid.manager import AstroidManager
from astroid.nodes import node_classes

# the keys of CONST_CLS eg python builtin types
_CONSTANTS = tuple(node_classes.CONST_CLS)
_BUILTINS = vars(builtins)
TYPE_NONE = type(None)
TYPE_NOTIMPLEMENTED = type(NotImplemented)
TYPE_ELLIPSIS = type(...)


def _io_discrepancy(member):
    # _io module names itself `io`: http://bugs.python.org/issue18602
    member_self = getattr(member, "__self__", None)
    return (
        member_self
        and inspect.ismodule(member_self)
        and member_self.__name__ == "_io"
        and member.__module__ == "io"
    )


def _attach_local_node(parent, node, name):
    node.name = name  # needed by add_local_node
    parent.add_local_node(node)


def _add_dunder_class(func, member):
    """Add a __class__ member to the given func node, if we can determine it."""
    python_cls = member.__class__
    cls_name = getattr(python_cls, "__name__", None)
    if not cls_name:
        return
    cls_bases = [ancestor.__name__ for ancestor in python_cls.__bases__]
    ast_klass = build_class(cls_name, cls_bases, python_cls.__doc__)
    func.instance_attrs["__class__"] = [ast_klass]


_marker = object()


def attach_dummy_node(node, name, runtime_object=_marker):
    """create a dummy node and register it in the locals of the given
    node with the specified name
    """
    enode = nodes.EmptyNode()
    enode.object = runtime_object
    _attach_local_node(node, enode, name)


def _has_underlying_object(self):
    return self.object is not None and self.object is not _marker


nodes.EmptyNode.has_underlying_object = _has_underlying_object


def attach_const_node(node, name, value):
    """create a Const node and register it in the locals of the given
    node with the specified name
    """
    if name not in node.special_attributes:
        _attach_local_node(node, nodes.const_factory(value), name)


def attach_import_node(node, modname, membername):
    """create a ImportFrom node and register it in the locals of the given
    node with the specified name
    """
    from_node = nodes.ImportFrom(modname, [(membername, None)])
    _attach_local_node(node, from_node, membername)


def build_module(name: str, doc: Optional[str] = None) -> nodes.Module:
    """create and initialize an astroid Module node"""
    node = nodes.Module(name, doc, pure_python=False)
    node.package = False
    node.parent = None
    return node


def build_class(name, basenames=(), doc=None):
    """create and initialize an astroid ClassDef node"""
    node = nodes.ClassDef(name, doc)
    for base in basenames:
        basenode = nodes.Name(name=base)
        node.bases.append(basenode)
        basenode.parent = node
    return node


def build_function(
    name,
    args: Optional[List[str]] = None,
    posonlyargs: Optional[List[str]] = None,
    defaults=None,
    doc=None,
    kwonlyargs: Optional[List[str]] = None,
) -> nodes.FunctionDef:
    """create and initialize an astroid FunctionDef node"""
    # first argument is now a list of decorators
    func = nodes.FunctionDef(name, doc)
    func.args = argsnode = nodes.Arguments(parent=func)
    argsnode.postinit(
        args=[nodes.AssignName(name=arg, parent=argsnode) for arg in args or ()],
        defaults=[],
        kwonlyargs=[
            nodes.AssignName(name=arg, parent=argsnode) for arg in kwonlyargs or ()
        ],
        kw_defaults=[],
        annotations=[],
        posonlyargs=[
            nodes.AssignName(name=arg, parent=argsnode) for arg in posonlyargs or ()
        ],
    )
    for default in defaults or ():
        argsnode.defaults.append(nodes.const_factory(default))
        argsnode.defaults[-1].parent = argsnode
    if args:
        register_arguments(func)
    return func


def build_from_import(fromname, names):
    """create and initialize an astroid ImportFrom import statement"""
    return nodes.ImportFrom(fromname, [(name, None) for name in names])


def register_arguments(func, args=None):
    """add given arguments to local

    args is a list that may contains nested lists
    (i.e. def func(a, (b, c, d)): ...)
    """
    if args is None:
        args = func.args.args
        if func.args.vararg:
            func.set_local(func.args.vararg, func.args)
        if func.args.kwarg:
            func.set_local(func.args.kwarg, func.args)
    for arg in args:
        if isinstance(arg, nodes.AssignName):
            func.set_local(arg.name, arg)
        else:
            register_arguments(func, arg.elts)


def object_build_class(node, member, localname):
    """create astroid for a living class object"""
    basenames = [base.__name__ for base in member.__bases__]
    return _base_class_object_build(node, member, basenames, localname=localname)


def object_build_function(node, member, localname):
    """create astroid for a living function object"""
    signature = inspect.signature(member)
    args = []
    defaults = []
    posonlyargs = []
    kwonlyargs = []
    for param_name, param in signature.parameters.items():
        if param.kind == inspect.Parameter.POSITIONAL_ONLY:
            posonlyargs.append(param_name)
        elif param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
            args.append(param_name)
        elif param.kind == inspect.Parameter.VAR_POSITIONAL:
            args.append(param_name)
        elif param.kind == inspect.Parameter.VAR_KEYWORD:
            args.append(param_name)
        elif param.kind == inspect.Parameter.KEYWORD_ONLY:
            kwonlyargs.append(param_name)
        if param.default is not inspect._empty:
            defaults.append(param.default)
    func = build_function(
        getattr(member, "__name__", None) or localname,
        args,
        posonlyargs,
        defaults,
        member.__doc__,
    )
    node.add_local_node(func, localname)


def object_build_datadescriptor(node, member, name):
    """create astroid for a living data descriptor object"""
    return _base_class_object_build(node, member, [], name)


def object_build_methoddescriptor(node, member, localname):
    """create astroid for a living method descriptor object"""
    # FIXME get arguments ?
    func = build_function(
        getattr(member, "__name__", None) or localname, doc=member.__doc__
    )
    # set node's arguments to None to notice that we have no information, not
    # and empty argument list
    func.args.args = None
    node.add_local_node(func, localname)
    _add_dunder_class(func, member)


def _base_class_object_build(node, member, basenames, name=None, localname=None):
    """create astroid for a living class object, with a given set of base names
    (e.g. ancestors)
    """
    klass = build_class(
        name or getattr(member, "__name__", None) or localname,
        basenames,
        member.__doc__,
    )
    klass._newstyle = isinstance(member, type)
    node.add_local_node(klass, localname)
    try:
        # limit the instantiation trick since it's too dangerous
        # (such as infinite test execution...)
        # this at least resolves common case such as Exception.args,
        # OSError.errno
        if issubclass(member, Exception):
            instdict = member().__dict__
        else:
            raise TypeError
    except TypeError:
        pass
    else:
        for item_name, obj in instdict.items():
            valnode = nodes.EmptyNode()
            valnode.object = obj
            valnode.parent = klass
            valnode.lineno = 1
            klass.instance_attrs[item_name] = [valnode]
    return klass


def _build_from_function(node, name, member, module):
    # verify this is not an imported function
    try:
        code = member.__code__
    except AttributeError:
        # Some implementations don't provide the code object,
        # such as Jython.
        code = None
    filename = getattr(code, "co_filename", None)
    if filename is None:
        assert isinstance(member, object)
        object_build_methoddescriptor(node, member, name)
    elif filename != getattr(module, "__file__", None):
        attach_dummy_node(node, name, member)
    else:
        object_build_function(node, member, name)


def _safe_has_attribute(obj, member):
    try:
        return hasattr(obj, member)
    except Exception:  # pylint: disable=broad-except
        return False


class InspectBuilder:
    """class for building nodes from living object

    this is actually a really minimal representation, including only Module,
    FunctionDef and ClassDef nodes and some others as guessed.
    """

    def __init__(self, manager_instance=None):
        self._manager = manager_instance or AstroidManager()
        self._done = {}
        self._module = None

    def inspect_build(
        self,
        module: types.ModuleType,
        modname: Optional[str] = None,
        path: Optional[str] = None,
    ) -> nodes.Module:
        """build astroid from a living module (i.e. using inspect)
        this is used when there is no python source code available (either
        because it's a built-in module or because the .py is not available)
        """
        self._module = module
        if modname is None:
            modname = module.__name__
        try:
            node = build_module(modname, module.__doc__)
        except AttributeError:
            # in jython, java modules have no __doc__ (see #109562)
            node = build_module(modname)
        node.file = node.path = os.path.abspath(path) if path else path
        node.name = modname
        self._manager.cache_module(node)
        node.package = hasattr(module, "__path__")
        self._done = {}
        self.object_build(node, module)
        return node

    def object_build(self, node, obj):
        """recursive method which create a partial ast from real objects
        (only function, class, and method are handled)
        """
        if obj in self._done:
            return self._done[obj]
        self._done[obj] = node
        for name in dir(obj):
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("error")
                    member = getattr(obj, name)
            except (AttributeError, DeprecationWarning):
                # damned ExtensionClass.Base, I know you're there !
                attach_dummy_node(node, name)
                continue
            if inspect.ismethod(member):
                member = member.__func__
            if inspect.isfunction(member):
                _build_from_function(node, name, member, self._module)
            elif inspect.isbuiltin(member):
                if not _io_discrepancy(member) and self.imported_member(
                    node, member, name
                ):
                    continue
                object_build_methoddescriptor(node, member, name)
            elif inspect.isclass(member):
                if self.imported_member(node, member, name):
                    continue
                if member in self._done:
                    class_node = self._done[member]
                    if class_node not in node.locals.get(name, ()):
                        node.add_local_node(class_node, name)
                else:
                    class_node = object_build_class(node, member, name)
                    # recursion
                    self.object_build(class_node, member)
                if name == "__class__" and class_node.parent is None:
                    class_node.parent = self._done[self._module]
            elif inspect.ismethoddescriptor(member):
                assert isinstance(member, object)
                object_build_methoddescriptor(node, member, name)
            elif inspect.isdatadescriptor(member):
                assert isinstance(member, object)
                object_build_datadescriptor(node, member, name)
            elif isinstance(member, _CONSTANTS):
                attach_const_node(node, name, member)
            elif inspect.isroutine(member):
                # This should be called for Jython, where some builtin
                # methods aren't caught by isbuiltin branch.
                _build_from_function(node, name, member, self._module)
            elif _safe_has_attribute(member, "__all__"):
                module = build_module(name)
                _attach_local_node(node, module, name)
                # recursion
                self.object_build(module, member)
            else:
                # create an empty node so that the name is actually defined
                attach_dummy_node(node, name, member)
        return None

    def imported_member(self, node, member, name):
        """verify this is not an imported class or handle it"""
        # /!\ some classes like ExtensionClass doesn't have a __module__
        # attribute ! Also, this may trigger an exception on badly built module
        # (see http://www.logilab.org/ticket/57299 for instance)
        try:
            modname = getattr(member, "__module__", None)
        except TypeError:
            modname = None
        if modname is None:
            if name in {"__new__", "__subclasshook__"}:
                # Python 2.5.1 (r251:54863, Sep  1 2010, 22:03:14)
                # >>> print object.__new__.__module__
                # None
                modname = builtins.__name__
            else:
                attach_dummy_node(node, name, member)
                return True

        real_name = {"gtk": "gtk_gtk", "_io": "io"}.get(modname, modname)

        if real_name != self._module.__name__:
            # check if it sounds valid and then add an import node, else use a
            # dummy node
            try:
                getattr(sys.modules[modname], name)
            except (KeyError, AttributeError):
                attach_dummy_node(node, name, member)
            else:
                attach_import_node(node, modname, name)
            return True
        return False


# astroid bootstrapping ######################################################

_CONST_PROXY = {}


def _set_proxied(const):
    # TODO : find a nicer way to handle this situation;
    return _CONST_PROXY[const.value.__class__]


def _astroid_bootstrapping():
    """astroid bootstrapping the builtins module"""
    # this boot strapping is necessary since we need the Const nodes to
    # inspect_build builtins, and then we can proxy Const
    builder = InspectBuilder()
    astroid_builtin = builder.inspect_build(builtins)

    for cls, node_cls in node_classes.CONST_CLS.items():
        if cls is TYPE_NONE:
            proxy = build_class("NoneType")
            proxy.parent = astroid_builtin
        elif cls is TYPE_NOTIMPLEMENTED:
            proxy = build_class("NotImplementedType")
            proxy.parent = astroid_builtin
        elif cls is TYPE_ELLIPSIS:
            proxy = build_class("Ellipsis")
            proxy.parent = astroid_builtin
        else:
            proxy = astroid_builtin.getattr(cls.__name__)[0]
        if cls in (dict, list, set, tuple):
            node_cls._proxied = proxy
        else:
            _CONST_PROXY[cls] = proxy

    # Set the builtin module as parent for some builtins.
    nodes.Const._proxied = property(_set_proxied)

    _GeneratorType = nodes.ClassDef(
        types.GeneratorType.__name__, types.GeneratorType.__doc__
    )
    _GeneratorType.parent = astroid_builtin
    bases.Generator._proxied = _GeneratorType
    builder.object_build(bases.Generator._proxied, types.GeneratorType)

    if hasattr(types, "AsyncGeneratorType"):
        _AsyncGeneratorType = nodes.ClassDef(
            types.AsyncGeneratorType.__name__, types.AsyncGeneratorType.__doc__
        )
        _AsyncGeneratorType.parent = astroid_builtin
        bases.AsyncGenerator._proxied = _AsyncGeneratorType
        builder.object_build(bases.AsyncGenerator._proxied, types.AsyncGeneratorType)
    builtin_types = (
        types.GetSetDescriptorType,
        types.GeneratorType,
        types.MemberDescriptorType,
        TYPE_NONE,
        TYPE_NOTIMPLEMENTED,
        types.FunctionType,
        types.MethodType,
        types.BuiltinFunctionType,
        types.ModuleType,
        types.TracebackType,
    )
    for _type in builtin_types:
        if _type.__name__ not in astroid_builtin:
            cls = nodes.ClassDef(_type.__name__, _type.__doc__)
            cls.parent = astroid_builtin
            builder.object_build(cls, _type)
            astroid_builtin[_type.__name__] = cls


_astroid_bootstrapping()

```
### 5 - astroid/const.py:

```python
import enum
import sys

PY38 = sys.version_info[:2] == (3, 8)
PY37_PLUS = sys.version_info >= (3, 7)
PY38_PLUS = sys.version_info >= (3, 8)
PY39_PLUS = sys.version_info >= (3, 9)
PY310_PLUS = sys.version_info >= (3, 10)
BUILTINS = "builtins"  # TODO Remove in 2.8


class Context(enum.Enum):
    Load = 1
    Store = 2
    Del = 3


# TODO Remove in 3.0 in favor of Context
Load = Context.Load
Store = Context.Store
Del = Context.Del

```
