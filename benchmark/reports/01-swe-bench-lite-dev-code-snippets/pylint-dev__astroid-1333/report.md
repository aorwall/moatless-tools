# pylint-dev__astroid-1333

| **pylint-dev/astroid** | `d2a5b3c7b1e203fec3c7ca73c30eb1785d3d4d0a` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 21818 |
| **Any found context length** | 833 |
| **Avg pos** | 101.0 |
| **Min pos** | 2 |
| **Max pos** | 76 |
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
| astroid/modutils.py | 297 | - | 17 | 2 | 4596
| astroid/modutils.py | 311 | 314 | 76 | 2 | 21818


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
| 1 | 1 astroid/__pkginfo__.py | 26 | 28| 18 | 18 | 
| **-> 2 <-** | **2 astroid/modutils.py** | 44 | 125| 815 | 833 | 
| 3 | 3 astroid/_ast.py | 0 | 18| 125 | 958 | 
| 4 | 4 astroid/raw_building.py | 27 | 60| 219 | 1177 | 
| 5 | 5 astroid/const.py | 0 | 21| 147 | 1324 | 
| **-> 6 <-** | **5 astroid/modutils.py** | 126 | 194| 567 | 1891 | 
| 7 | 6 astroid/interpreter/_import/util.py | 5 | 17| 62 | 1953 | 
| 8 | 7 astroid/manager.py | 31 | 60| 183 | 2136 | 
| 9 | 8 astroid/bases.py | 31 | 82| 330 | 2466 | 
| 10 | 9 astroid/nodes/scoped_nodes/scoped_nodes.py | 45 | 100| 342 | 2808 | 
| 11 | 10 astroid/rebuilder.py | 32 | 82| 307 | 3115 | 
| 12 | 11 astroid/builder.py | 114 | 150| 314 | 3429 | 
| 13 | 12 astroid/nodes/node_ng.py | 0 | 44| 262 | 3691 | 
| 14 | 13 astroid/interpreter/_import/spec.py | 237 | 263| 161 | 3852 | 
| 15 | 13 astroid/builder.py | 158 | 156| 233 | 4085 | 
| 16 | 13 astroid/_ast.py | 21 | 44| 138 | 4223 | 
| **-> 17 <-** | **13 astroid/modutils.py** | 211 | 270| 373 | 4596 | 
| 18 | 13 astroid/rebuilder.py | 500 | 796| 31 | 4627 | 
| 19 | 13 astroid/builder.py | 26 | 68| 301 | 4928 | 
| 20 | 13 astroid/rebuilder.py | 154 | 375| 1591 | 6519 | 
| 21 | 13 astroid/raw_building.py | 269 | 291| 168 | 6687 | 
| 22 | 13 astroid/raw_building.py | 437 | 499| 523 | 7210 | 
| 23 | 13 astroid/rebuilder.py | 502 | 725| 1621 | 8831 | 
| 24 | 13 astroid/raw_building.py | 393 | 434| 360 | 9191 | 
| 25 | 13 astroid/raw_building.py | 164 | 186| 234 | 9425 | 
| 26 | 14 astroid/helpers.py | 23 | 51| 200 | 9625 | 
| 27 | 14 astroid/interpreter/_import/spec.py | 17 | 38| 114 | 9739 | 
| 28 | 15 astroid/brain/brain_dataclasses.py | 448 | 464| 104 | 9843 | 
| 29 | 15 astroid/builder.py | 86 | 112| 304 | 10147 | 
| 30 | **15 astroid/modutils.py** | 634 | 658| 182 | 10329 | 
| 31 | 15 astroid/interpreter/_import/spec.py | 210 | 211| 222 | 10551 | 
| 32 | 15 astroid/interpreter/_import/spec.py | 300 | 327| 240 | 10791 | 
| 33 | 15 astroid/rebuilder.py | 487 | 498| 155 | 10946 | 
| 34 | 15 astroid/builder.py | 457 | 471| 150 | 11096 | 
| 35 | 15 astroid/rebuilder.py | 377 | 485| 786 | 11882 | 
| 36 | 16 astroid/brain/brain_re.py | 11 | 8| 259 | 12141 | 
| 37 | 17 astroid/nodes/scoped_nodes/__init__.py | 8 | 43| 171 | 12312 | 
| 38 | 17 astroid/nodes/scoped_nodes/scoped_nodes.py | 651 | 670| 161 | 12473 | 
| 39 | 17 astroid/manager.py | 222 | 247| 200 | 12673 | 
| 40 | 17 astroid/rebuilder.py | 727 | 783| 439 | 13112 | 
| 41 | 17 astroid/rebuilder.py | 785 | 796| 153 | 13265 | 
| 42 | 18 astroid/protocols.py | 277 | 293| 193 | 13458 | 
| 43 | 18 astroid/interpreter/_import/spec.py | 95 | 102| 562 | 14020 | 
| 44 | 19 astroid/nodes/node_classes.py | 40 | 88| 290 | 14310 | 
| 45 | 19 astroid/helpers.py | 192 | 212| 181 | 14491 | 
| 46 | 20 astroid/brain/brain_numpy_utils.py | 11 | 43| 223 | 14714 | 
| 47 | **20 astroid/modutils.py** | 461 | 495| 255 | 14969 | 
| 48 | 20 astroid/interpreter/_import/spec.py | 167 | 185| 150 | 15119 | 
| 49 | 20 astroid/raw_building.py | 189 | 216| 215 | 15334 | 
| 50 | 20 astroid/rebuilder.py | 2311 | 2324| 132 | 15466 | 
| 51 | 21 astroid/brain/brain_typing.py | 15 | 115| 583 | 16049 | 
| 52 | 21 astroid/bases.py | 343 | 361| 203 | 16252 | 
| 53 | 21 astroid/protocols.py | 764 | 847| 648 | 16900 | 
| 54 | 21 astroid/brain/brain_typing.py | 410 | 438| 217 | 17117 | 
| 55 | 21 astroid/rebuilder.py | 906 | 939| 198 | 17315 | 
| 56 | 21 astroid/manager.py | 270 | 286| 158 | 17473 | 
| 57 | 21 astroid/rebuilder.py | 974 | 972| 208 | 17681 | 
| 58 | 22 astroid/inference.py | 335 | 358| 168 | 17849 | 
| 59 | 22 astroid/raw_building.py | 294 | 304| 297 | 18146 | 
| 60 | 22 astroid/raw_building.py | 219 | 234| 148 | 18294 | 
| 61 | 22 astroid/inference.py | 255 | 275| 138 | 18432 | 
| 62 | 22 astroid/rebuilder.py | 2262 | 2290| 234 | 18666 | 
| 63 | 22 astroid/bases.py | 85 | 110| 210 | 18876 | 
| 64 | 22 astroid/raw_building.py | 237 | 266| 235 | 19111 | 
| 65 | 22 astroid/rebuilder.py | 941 | 964| 199 | 19310 | 
| 66 | 22 astroid/nodes/node_classes.py | 932 | 972| 244 | 19554 | 
| 67 | 22 astroid/raw_building.py | 127 | 156| 239 | 19793 | 
| 68 | 22 astroid/rebuilder.py | 1458 | 1500| 336 | 20129 | 
| 69 | 22 astroid/protocols.py | 691 | 762| 571 | 20700 | 
| 70 | 22 astroid/inference.py | 106 | 120| 111 | 20811 | 
| 71 | 22 astroid/rebuilder.py | 1991 | 2008| 162 | 20973 | 
| 72 | 22 astroid/interpreter/_import/spec.py | 41 | 63| 129 | 21102 | 
| 73 | 22 astroid/protocols.py | 324 | 338| 127 | 21229 | 
| 74 | 22 astroid/rebuilder.py | 106 | 133| 217 | 21446 | 
| 75 | 22 astroid/rebuilder.py | 2143 | 2168| 242 | 21688 | 
| **-> 76 <-** | **22 astroid/modutils.py** | 307 | 322| 130 | 21818 | 


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

Start line: 26, End line: 28

```python
__version__ = "2.10.0-dev0"
version = __version__
```
### 2 - astroid/modutils.py:

Start line: 44, End line: 125

```python
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
```
### 4 - astroid/raw_building.py:

Start line: 27, End line: 60

```python
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
### 6 - astroid/modutils.py:

Start line: 126, End line: 194

```python
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
```
### 7 - astroid/interpreter/_import/util.py:

Start line: 5, End line: 17

```python
try:
    import pkg_resources
except ImportError:
    pkg_resources = None  # type: ignore[assignment]


def is_namespace(modname):
    return (
        pkg_resources is not None
        and hasattr(pkg_resources, "_namespace_packages")
        and modname in pkg_resources._namespace_packages
    )
```
### 8 - astroid/manager.py:

Start line: 31, End line: 60

```python
import os
import types
import zipimport
from typing import TYPE_CHECKING, ClassVar, List, Optional

from astroid.exceptions import AstroidBuildingError, AstroidImportError
from astroid.interpreter._import import spec
from astroid.modutils import (
    NoSourceFile,
    file_info_from_modpath,
    get_source_file,
    is_module_name_part_of_extension_package_whitelist,
    is_python_source,
    is_standard_module,
    load_module_from_name,
    modpath_from_file,
)
from astroid.transforms import TransformVisitor

if TYPE_CHECKING:
    from astroid import nodes

ZIP_IMPORT_EXTS = (".zip", ".egg", ".whl", ".pyz", ".pyzw")


def safe_repr(obj):
    try:
        return repr(obj)
    except Exception:  # pylint: disable=broad-except
        return "???"
```
### 9 - astroid/bases.py:

Start line: 31, End line: 82

```python
import collections

from astroid import decorators
from astroid.const import PY310_PLUS
from astroid.context import (
    CallContext,
    InferenceContext,
    bind_context_to_node,
    copy_context,
)
from astroid.exceptions import (
    AstroidTypeError,
    AttributeInferenceError,
    InferenceError,
    NameInferenceError,
)
from astroid.util import Uninferable, lazy_descriptor, lazy_import

objectmodel = lazy_import("interpreter.objectmodel")
helpers = lazy_import("helpers")
manager = lazy_import("manager")


# TODO: check if needs special treatment
BOOL_SPECIAL_METHOD = "__bool__"
BUILTINS = "builtins"  # TODO Remove in 2.8

PROPERTIES = {"builtins.property", "abc.abstractproperty"}
if PY310_PLUS:
    PROPERTIES.add("enum.property")

# List of possible property names. We use this list in order
# to see if a method is a property or not. This should be
# pretty reliable and fast, the alternative being to check each
# decorator to see if its a real property-like descriptor, which
# can be too complicated.
# Also, these aren't qualified, because each project can
# define them, we shouldn't expect to know every possible
# property-like decorator!
POSSIBLE_PROPERTIES = {
    "cached_property",
    "cachedproperty",
    "lazyproperty",
    "lazy_property",
    "reify",
    "lazyattribute",
    "lazy_attribute",
    "LazyProperty",
    "lazy",
    "cache_readonly",
    "DynamicClassAttribute",
}
```
### 10 - astroid/nodes/scoped_nodes/scoped_nodes.py:

Start line: 45, End line: 100

```python
import builtins
import io
import itertools
import os
import sys
import typing
import warnings
from typing import List, Optional, TypeVar, Union, overload

from astroid import bases
from astroid import decorators as decorators_mod
from astroid import mixins, util
from astroid.const import PY39_PLUS
from astroid.context import (
    CallContext,
    InferenceContext,
    bind_context_to_node,
    copy_context,
)
from astroid.exceptions import (
    AstroidBuildingError,
    AstroidTypeError,
    AttributeInferenceError,
    DuplicateBasesError,
    InconsistentMroError,
    InferenceError,
    MroError,
    StatementMissing,
    TooManyLevelsError,
)
from astroid.filter_statements import _filter_stmts
from astroid.interpreter.dunder_lookup import lookup
from astroid.interpreter.objectmodel import ClassModel, FunctionModel, ModuleModel
from astroid.manager import AstroidManager
from astroid.nodes import Arguments, Const, node_classes

if sys.version_info >= (3, 6, 2):
    from typing import NoReturn
else:
    from typing_extensions import NoReturn


if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal


ITER_METHODS = ("__iter__", "__getitem__")
EXCEPTION_BASE_CLASSES = frozenset({"Exception", "BaseException"})
objects = util.lazy_import("objects")
BUILTIN_DESCRIPTORS = frozenset(
    {"classmethod", "staticmethod", "builtins.classmethod", "builtins.staticmethod"}
)

T = TypeVar("T")
```
### 17 - astroid/modutils.py:

Start line: 211, End line: 270

```python
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
```
### 30 - astroid/modutils.py:

Start line: 634, End line: 658

```python
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
```
### 47 - astroid/modutils.py:

Start line: 461, End line: 495

```python
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
```
### 76 - astroid/modutils.py:

Start line: 307, End line: 322

```python
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
```
