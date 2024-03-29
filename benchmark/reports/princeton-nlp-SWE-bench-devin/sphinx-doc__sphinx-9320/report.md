# sphinx-doc__sphinx-9320

| **sphinx-doc/sphinx** | `e05cef574b8f23ab1b57f57e7da6dee509a4e230` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 2991 |
| **Any found context length** | 1164 |
| **Avg pos** | 7.0 |
| **Min pos** | 2 |
| **Max pos** | 5 |
| **Top file pos** | 1 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sphinx/cmd/quickstart.py b/sphinx/cmd/quickstart.py
--- a/sphinx/cmd/quickstart.py
+++ b/sphinx/cmd/quickstart.py
@@ -95,6 +95,12 @@ def is_path(x: str) -> str:
     return x
 
 
+def is_path_or_empty(x: str) -> str:
+    if x == '':
+        return x
+    return is_path(x)
+
+
 def allow_empty(x: str) -> str:
     return x
 
@@ -223,7 +229,7 @@ def ask_user(d: Dict) -> None:
         print(__('sphinx-quickstart will not overwrite existing Sphinx projects.'))
         print()
         d['path'] = do_prompt(__('Please enter a new root path (or just Enter to exit)'),
-                              '', is_path)
+                              '', is_path_or_empty)
         if not d['path']:
             sys.exit(1)
 

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sphinx/cmd/quickstart.py | 98 | 98 | 5 | 1 | 2991
| sphinx/cmd/quickstart.py | 226 | 226 | 2 | 1 | 1164


## Problem Statement

```
`sphinx-quickstart` with existing conf.py doesn't exit easily
**Describe the bug**
I've attached a screenshot in the screenshots section which I think explains the bug better.

- I'm running `sphinx-quickstart` in a folder with a conf.py already existing. 
- It says *"Please enter a new root path name (or just Enter to exit)"*. 
- However, upon pressing 'Enter' it returns an error message *"Please enter a valid path name"*. 


**To Reproduce**
Steps to reproduce the behavior:
\`\`\`
$ sphinx-quickstart
$ sphinx-quickstart
\`\`\`

**Expected behavior**
After pressing Enter, sphinx-quickstart exits. 

**Your project**
n/a

**Screenshots**

![sphinx-enter-exit](https://user-images.githubusercontent.com/30437511/121676712-4bf54f00-caf8-11eb-992b-636e56999d54.png)
I press Enter for the first prompt.


**Environment info**
- OS: Ubuntu 20.04
- Python version: Python 3.8.5
- Sphinx version: sphinx-build 3.2.1 
- Sphinx extensions:  none
- Extra tools: none

**Additional context**
I had a quick search but couldn't find any similar existing issues. Sorry if this is a duplicate.


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 sphinx/cmd/quickstart.py** | 538 | 605| 491 | 491 | 5540 | 
| **-> 2 <-** | **1 sphinx/cmd/quickstart.py** | 185 | 260| 673 | 1164 | 5540 | 
| 3 | **1 sphinx/cmd/quickstart.py** | 262 | 320| 739 | 1903 | 5540 | 
| 4 | **1 sphinx/cmd/quickstart.py** | 122 | 157| 302 | 2205 | 5540 | 
| **-> 5 <-** | **1 sphinx/cmd/quickstart.py** | 11 | 119| 786 | 2991 | 5540 | 
| 6 | **1 sphinx/cmd/quickstart.py** | 453 | 519| 751 | 3742 | 5540 | 
| 7 | **1 sphinx/cmd/quickstart.py** | 424 | 450| 158 | 3900 | 5540 | 
| 8 | 2 setup.py | 176 | 252| 638 | 4538 | 7288 | 
| 9 | **2 sphinx/cmd/quickstart.py** | 393 | 421| 385 | 4923 | 7288 | 
| 10 | **2 sphinx/cmd/quickstart.py** | 520 | 535| 175 | 5098 | 7288 | 
| 11 | 3 sphinx/application.py | 300 | 313| 132 | 5230 | 18805 | 
| 12 | 4 sphinx/ext/apidoc.py | 369 | 397| 374 | 5604 | 23035 | 
| 13 | 5 doc/conf.py | 1 | 82| 731 | 6335 | 24499 | 
| 14 | 6 sphinx/setup_command.py | 91 | 118| 229 | 6564 | 26044 | 
| 15 | 6 sphinx/application.py | 332 | 381| 410 | 6974 | 26044 | 
| 16 | 7 sphinx/testing/fixtures.py | 11 | 40| 177 | 7151 | 27829 | 
| 17 | 7 sphinx/setup_command.py | 142 | 190| 415 | 7566 | 27829 | 
| 18 | 8 sphinx/testing/util.py | 86 | 135| 448 | 8014 | 29561 | 
| 19 | 8 sphinx/setup_command.py | 14 | 89| 415 | 8429 | 29561 | 
| 20 | 9 sphinx/config.py | 484 | 498| 154 | 8583 | 34017 | 
| 21 | 9 sphinx/testing/util.py | 137 | 151| 188 | 8771 | 34017 | 
| 22 | 10 sphinx/cmd/build.py | 202 | 299| 701 | 9472 | 36678 | 
| 23 | 11 sphinx/__main__.py | 1 | 16| 0 | 9472 | 36749 | 
| 24 | 12 doc/development/tutorials/examples/helloworld.py | 1 | 20| 0 | 9472 | 36838 | 
| 25 | 12 sphinx/config.py | 462 | 481| 223 | 9695 | 36838 | 
| 26 | 13 sphinx/testing/__init__.py | 1 | 15| 0 | 9695 | 36920 | 
| 27 | **13 sphinx/cmd/quickstart.py** | 160 | 182| 200 | 9895 | 36920 | 
| 28 | **13 sphinx/cmd/quickstart.py** | 323 | 391| 760 | 10655 | 36920 | 
| 29 | 13 setup.py | 1 | 78| 466 | 11121 | 36920 | 
| 30 | 14 sphinx/__init__.py | 14 | 60| 475 | 11596 | 37469 | 
| 31 | 15 sphinx/cmd/__init__.py | 1 | 10| 0 | 11596 | 37518 | 
| 32 | 15 sphinx/config.py | 91 | 150| 740 | 12336 | 37518 | 
| 33 | 16 sphinx/cmd/make_mode.py | 17 | 54| 532 | 12868 | 39220 | 
| 34 | 17 sphinx/errors.py | 77 | 134| 297 | 13165 | 40016 | 
| 35 | 17 sphinx/cmd/build.py | 169 | 199| 383 | 13548 | 40016 | 
| 36 | 18 utils/doclinter.py | 11 | 83| 496 | 14044 | 40564 | 
| 37 | 18 doc/conf.py | 141 | 162| 255 | 14299 | 40564 | 
| 38 | 18 sphinx/cmd/make_mode.py | 86 | 94| 114 | 14413 | 40564 | 
| 39 | 19 sphinx/builders/manpage.py | 11 | 29| 154 | 14567 | 41527 | 
| 40 | 20 sphinx/builders/html/__init__.py | 1294 | 1375| 1000 | 15567 | 53815 | 
| 41 | 20 sphinx/setup_command.py | 120 | 140| 172 | 15739 | 53815 | 
| 42 | 20 sphinx/cmd/build.py | 11 | 30| 132 | 15871 | 53815 | 
| 43 | 21 sphinx/util/osutil.py | 71 | 171| 718 | 16589 | 55449 | 
| 44 | 21 sphinx/cmd/build.py | 101 | 168| 762 | 17351 | 55449 | 
| 45 | 21 sphinx/application.py | 126 | 270| 1228 | 18579 | 55449 | 
| 46 | 22 sphinx/util/__init__.py | 11 | 64| 446 | 19025 | 60202 | 
| 47 | 23 sphinx/util/console.py | 85 | 98| 155 | 19180 | 61198 | 
| 48 | 24 sphinx/ext/__init__.py | 1 | 10| 0 | 19180 | 61248 | 
| 49 | 25 sphinx/environment/__init__.py | 11 | 82| 508 | 19688 | 66745 | 
| 50 | 26 sphinx/ext/intersphinx.py | 367 | 378| 125 | 19813 | 70425 | 
| 51 | 26 sphinx/ext/apidoc.py | 303 | 368| 751 | 20564 | 70425 | 
| 52 | 27 utils/bump_version.py | 67 | 102| 201 | 20765 | 71788 | 
| 53 | 28 sphinx/project.py | 11 | 39| 220 | 20985 | 72554 | 
| 54 | 28 sphinx/util/osutil.py | 11 | 45| 214 | 21199 | 72554 | 
| 55 | 28 sphinx/application.py | 13 | 58| 361 | 21560 | 72554 | 
| 56 | 28 sphinx/config.py | 252 | 287| 326 | 21886 | 72554 | 
| 57 | 29 utils/checks.py | 33 | 109| 545 | 22431 | 73460 | 
| 58 | 30 sphinx/ext/autodoc/__init__.py | 2738 | 2781| 528 | 22959 | 96894 | 
| 59 | 30 doc/conf.py | 83 | 138| 476 | 23435 | 96894 | 
| 60 | 30 sphinx/application.py | 315 | 330| 117 | 23552 | 96894 | 
| 61 | 31 sphinx/ext/todo.py | 225 | 248| 214 | 23766 | 98734 | 
| 62 | 31 sphinx/errors.py | 48 | 74| 222 | 23988 | 98734 | 
| 63 | 32 sphinx/ext/graphviz.py | 12 | 44| 243 | 24231 | 102470 | 
| 64 | 33 sphinx/testing/path.py | 83 | 99| 185 | 24416 | 104133 | 
| 65 | 33 sphinx/application.py | 61 | 123| 494 | 24910 | 104133 | 
| 66 | 33 sphinx/testing/path.py | 123 | 236| 802 | 25712 | 104133 | 
| 67 | 33 utils/bump_version.py | 149 | 180| 224 | 25936 | 104133 | 
| 68 | 34 sphinx/builders/text.py | 83 | 96| 114 | 26050 | 104823 | 
| 69 | 34 sphinx/testing/fixtures.py | 232 | 260| 174 | 26224 | 104823 | 
| 70 | 34 sphinx/util/osutil.py | 214 | 235| 149 | 26373 | 104823 | 
| 71 | 35 sphinx/ext/doctest.py | 556 | 573| 215 | 26588 | 109841 | 
| 72 | 35 sphinx/util/__init__.py | 506 | 532| 194 | 26782 | 109841 | 
| 73 | 35 sphinx/ext/apidoc.py | 17 | 73| 402 | 27184 | 109841 | 
| 74 | 36 sphinx/util/docutils.py | 146 | 170| 194 | 27378 | 113983 | 
| 75 | 36 sphinx/testing/fixtures.py | 140 | 174| 254 | 27632 | 113983 | 
| 76 | 37 sphinx/util/build_phase.py | 1 | 21| 0 | 27632 | 114096 | 
| 77 | 38 sphinx/theming.py | 11 | 48| 207 | 27839 | 115934 | 
| 78 | 39 sphinx/ext/imgconverter.py | 75 | 94| 204 | 28043 | 116680 | 
| 79 | 40 sphinx/builders/changes.py | 125 | 168| 438 | 28481 | 118193 | 
| 80 | 40 sphinx/util/console.py | 47 | 82| 247 | 28728 | 118193 | 
| 81 | 41 sphinx/domains/python.py | 1399 | 1413| 110 | 28838 | 130238 | 
| 82 | 42 sphinx/writers/__init__.py | 1 | 10| 0 | 28838 | 130287 | 
| 83 | 42 sphinx/config.py | 230 | 250| 174 | 29012 | 130287 | 
| 84 | 42 sphinx/errors.py | 12 | 45| 212 | 29224 | 130287 | 
| 85 | 42 sphinx/util/__init__.py | 214 | 248| 294 | 29518 | 130287 | 
| 86 | 43 doc/development/tutorials/examples/todo.py | 116 | 136| 161 | 29679 | 131195 | 
| 87 | 43 sphinx/environment/__init__.py | 442 | 520| 687 | 30366 | 131195 | 
| 88 | 43 sphinx/ext/intersphinx.py | 26 | 50| 157 | 30523 | 131195 | 
| 89 | 44 sphinx/util/logging.py | 566 | 598| 279 | 30802 | 134987 | 
| 90 | 45 sphinx/writers/texinfo.py | 1210 | 1275| 457 | 31259 | 147301 | 
| 91 | 46 sphinx/builders/epub3.py | 241 | 284| 607 | 31866 | 149888 | 
| 92 | 47 sphinx/util/nodes.py | 364 | 400| 313 | 32179 | 155373 | 
| 93 | 48 sphinx/addnodes.py | 530 | 579| 321 | 32500 | 159247 | 
| 94 | 49 sphinx/directives/__init__.py | 269 | 294| 167 | 32667 | 161499 | 
| 95 | 50 sphinx/builders/__init__.py | 11 | 48| 302 | 32969 | 166849 | 
| 96 | 51 sphinx/ext/githubpages.py | 11 | 37| 227 | 33196 | 167136 | 
| 97 | 51 sphinx/writers/texinfo.py | 394 | 402| 132 | 33328 | 167136 | 
| 98 | 52 sphinx/ext/napoleon/__init__.py | 271 | 294| 298 | 33626 | 171161 | 
| 99 | 53 sphinx/highlighting.py | 11 | 68| 620 | 34246 | 172707 | 
| 100 | 53 sphinx/testing/fixtures.py | 65 | 110| 391 | 34637 | 172707 | 
| 101 | 54 sphinx/util/pycompat.py | 49 | 60| 114 | 34751 | 173200 | 
| 102 | 55 sphinx/builders/latex/constants.py | 74 | 124| 537 | 35288 | 175392 | 
| 103 | 55 sphinx/util/__init__.py | 480 | 503| 195 | 35483 | 175392 | 
| 104 | 55 sphinx/testing/fixtures.py | 205 | 229| 167 | 35650 | 175392 | 
| 105 | 55 sphinx/ext/napoleon/__init__.py | 11 | 294| 57 | 35707 | 175392 | 
| 106 | 56 sphinx/jinja2glue.py | 11 | 44| 219 | 35926 | 177017 | 
| 107 | 56 sphinx/ext/doctest.py | 12 | 43| 227 | 36153 | 177017 | 
| 108 | 57 sphinx/util/smartypants.py | 380 | 392| 137 | 36290 | 181164 | 
| 109 | 57 sphinx/cmd/make_mode.py | 57 | 84| 318 | 36608 | 181164 | 
| 110 | 58 sphinx/locale/__init__.py | 241 | 269| 226 | 36834 | 183236 | 
| 111 | 59 sphinx/builders/texinfo.py | 11 | 36| 226 | 37060 | 185249 | 
| 112 | 59 utils/bump_version.py | 104 | 117| 123 | 37183 | 185249 | 
| 113 | 59 sphinx/ext/doctest.py | 246 | 273| 277 | 37460 | 185249 | 
| 114 | 60 sphinx/ext/autosummary/generate.py | 20 | 53| 257 | 37717 | 190548 | 
| 115 | 60 sphinx/ext/intersphinx.py | 343 | 364| 210 | 37927 | 190548 | 
| 116 | 61 sphinx/builders/singlehtml.py | 11 | 25| 112 | 38039 | 192344 | 
| 117 | 61 sphinx/builders/manpage.py | 110 | 129| 166 | 38205 | 192344 | 
| 118 | 61 sphinx/util/console.py | 32 | 44| 125 | 38330 | 192344 | 
| 119 | 61 sphinx/builders/texinfo.py | 198 | 220| 227 | 38557 | 192344 | 
| 120 | 61 sphinx/util/console.py | 101 | 142| 296 | 38853 | 192344 | 
| 121 | 61 sphinx/ext/graphviz.py | 405 | 421| 196 | 39049 | 192344 | 
| 122 | 61 sphinx/cmd/make_mode.py | 96 | 140| 375 | 39424 | 192344 | 
| 123 | 61 sphinx/ext/autosummary/generate.py | 637 | 658| 162 | 39586 | 192344 | 
| 124 | 61 sphinx/util/console.py | 11 | 29| 123 | 39709 | 192344 | 
| 125 | 61 sphinx/ext/doctest.py | 332 | 351| 164 | 39873 | 192344 | 
| 126 | 61 sphinx/ext/imgconverter.py | 11 | 49| 278 | 40151 | 192344 | 
| 127 | 61 sphinx/cmd/make_mode.py | 158 | 168| 110 | 40261 | 192344 | 
| 128 | 61 sphinx/cmd/build.py | 33 | 98| 647 | 40908 | 192344 | 
| 129 | 61 sphinx/util/pycompat.py | 11 | 46| 328 | 41236 | 192344 | 
| 130 | 61 sphinx/writers/texinfo.py | 964 | 1081| 846 | 42082 | 192344 | 
| 131 | 61 sphinx/ext/todo.py | 197 | 222| 211 | 42293 | 192344 | 
| 132 | 61 sphinx/ext/doctest.py | 200 | 229| 253 | 42546 | 192344 | 
| 133 | 61 sphinx/testing/path.py | 32 | 81| 269 | 42815 | 192344 | 
| 134 | 62 sphinx/ext/inheritance_diagram.py | 461 | 476| 173 | 42988 | 196210 | 
| 135 | 62 utils/bump_version.py | 119 | 146| 255 | 43243 | 196210 | 
| 136 | 62 sphinx/environment/__init__.py | 317 | 330| 136 | 43379 | 196210 | 
| 137 | 62 sphinx/ext/apidoc.py | 203 | 234| 296 | 43675 | 196210 | 


### Hint

```
I could try fix this myself (maybe?)
Good catch @dogenstein! In my opinion, if the selected path already has a `conf.py`, `sphinx-quickstart` should exit with status 1 immediately.
```

## Patch

```diff
diff --git a/sphinx/cmd/quickstart.py b/sphinx/cmd/quickstart.py
--- a/sphinx/cmd/quickstart.py
+++ b/sphinx/cmd/quickstart.py
@@ -95,6 +95,12 @@ def is_path(x: str) -> str:
     return x
 
 
+def is_path_or_empty(x: str) -> str:
+    if x == '':
+        return x
+    return is_path(x)
+
+
 def allow_empty(x: str) -> str:
     return x
 
@@ -223,7 +229,7 @@ def ask_user(d: Dict) -> None:
         print(__('sphinx-quickstart will not overwrite existing Sphinx projects.'))
         print()
         d['path'] = do_prompt(__('Please enter a new root path (or just Enter to exit)'),
-                              '', is_path)
+                              '', is_path_or_empty)
         if not d['path']:
             sys.exit(1)
 

```

## Test Patch

```diff
diff --git a/tests/test_quickstart.py b/tests/test_quickstart.py
--- a/tests/test_quickstart.py
+++ b/tests/test_quickstart.py
@@ -10,6 +10,7 @@
 
 import time
 from io import StringIO
+from os import path
 
 import pytest
 
@@ -250,3 +251,18 @@ def test_extensions(tempdir):
     ns = {}
     exec(conffile.read_text(), ns)
     assert ns['extensions'] == ['foo', 'bar', 'baz']
+
+
+def test_exits_when_existing_confpy(monkeypatch):
+    # The code detects existing conf.py with path.isfile() 
+    # so we mock it as True with pytest's monkeypatch
+    def mock_isfile(path):
+        return True
+    monkeypatch.setattr(path, 'isfile', mock_isfile)
+
+    qs.term_input = mock_input({
+        'Please enter a new root path (or just Enter to exit)': ''
+    })
+    d = {}
+    with pytest.raises(SystemExit):
+        qs.ask_user(d)

```


## Code snippets

### 1 - sphinx/cmd/quickstart.py:

Start line: 538, End line: 605

```python
def main(argv: List[str] = sys.argv[1:]) -> int:
    sphinx.locale.setlocale(locale.LC_ALL, '')
    sphinx.locale.init_console(os.path.join(package_dir, 'locale'), 'sphinx')

    if not color_terminal():
        nocolor()

    # parse options
    parser = get_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as err:
        return err.code

    d = vars(args)
    # delete None or False value
    d = {k: v for k, v in d.items() if v is not None}

    # handle use of CSV-style extension values
    d.setdefault('extensions', [])
    for ext in d['extensions'][:]:
        if ',' in ext:
            d['extensions'].remove(ext)
            d['extensions'].extend(ext.split(','))

    try:
        if 'quiet' in d:
            if not {'project', 'author'}.issubset(d):
                print(__('"quiet" is specified, but any of "project" or '
                         '"author" is not specified.'))
                return 1

        if {'quiet', 'project', 'author'}.issubset(d):
            # quiet mode with all required params satisfied, use default
            d.setdefault('version', '')
            d.setdefault('release', d['version'])
            d2 = DEFAULTS.copy()
            d2.update(d)
            d = d2

            if not valid_dir(d):
                print()
                print(bold(__('Error: specified path is not a directory, or sphinx'
                              ' files already exist.')))
                print(__('sphinx-quickstart only generate into a empty directory.'
                         ' Please specify a new root path.'))
                return 1
        else:
            ask_user(d)
    except (KeyboardInterrupt, EOFError):
        print()
        print('[Interrupted.]')
        return 130  # 128 + SIGINT

    for variable in d.get('variables', []):
        try:
            name, value = variable.split('=')
            d[name] = value
        except ValueError:
            print(__('Invalid template variable: %s') % variable)

    generate(d, overwrite=False, templatedir=args.templatedir)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
```
### 2 - sphinx/cmd/quickstart.py:

Start line: 185, End line: 260

```python
def ask_user(d: Dict) -> None:

    print(bold(__('Welcome to the Sphinx %s quickstart utility.')) % __display_version__)
    print()
    print(__('Please enter values for the following settings (just press Enter to\n'
             'accept a default value, if one is given in brackets).'))

    if 'path' in d:
        print()
        print(bold(__('Selected root path: %s')) % d['path'])
    else:
        print()
        print(__('Enter the root path for documentation.'))
        d['path'] = do_prompt(__('Root path for the documentation'), '.', is_path)

    while path.isfile(path.join(d['path'], 'conf.py')) or \
            path.isfile(path.join(d['path'], 'source', 'conf.py')):
        print()
        print(bold(__('Error: an existing conf.py has been found in the '
                      'selected root path.')))
        print(__('sphinx-quickstart will not overwrite existing Sphinx projects.'))
        print()
        d['path'] = do_prompt(__('Please enter a new root path (or just Enter to exit)'),
                              '', is_path)
        if not d['path']:
            sys.exit(1)

    if 'sep' not in d:
        print()
        print(__('You have two options for placing the build directory for Sphinx output.\n'
                 'Either, you use a directory "_build" within the root path, or you separate\n'
                 '"source" and "build" directories within the root path.'))
        d['sep'] = do_prompt(__('Separate source and build directories (y/n)'), 'n', boolean)

    if 'dot' not in d:
        print()
        print(__('Inside the root directory, two more directories will be created; "_templates"\n'      # NOQA
                 'for custom HTML templates and "_static" for custom stylesheets and other static\n'    # NOQA
                 'files. You can enter another prefix (such as ".") to replace the underscore.'))       # NOQA
        d['dot'] = do_prompt(__('Name prefix for templates and static dir'), '_', ok)

    if 'project' not in d:
        print()
        print(__('The project name will occur in several places in the built documentation.'))
        d['project'] = do_prompt(__('Project name'))
    if 'author' not in d:
        d['author'] = do_prompt(__('Author name(s)'))

    if 'version' not in d:
        print()
        print(__('Sphinx has the notion of a "version" and a "release" for the\n'
                 'software. Each version can have multiple releases. For example, for\n'
                 'Python the version is something like 2.5 or 3.0, while the release is\n'
                 'something like 2.5.1 or 3.0a1. If you don\'t need this dual structure,\n'
                 'just set both to the same value.'))
        d['version'] = do_prompt(__('Project version'), '', allow_empty)
    if 'release' not in d:
        d['release'] = do_prompt(__('Project release'), d['version'], allow_empty)
    # ... other code
```
### 3 - sphinx/cmd/quickstart.py:

Start line: 262, End line: 320

```python
def ask_user(d: Dict) -> None:
    # ... other code

    if 'language' not in d:
        print()
        print(__('If the documents are to be written in a language other than English,\n'
                 'you can select a language here by its language code. Sphinx will then\n'
                 'translate text that it generates into that language.\n'
                 '\n'
                 'For a list of supported codes, see\n'
                 'https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-language.'))  # NOQA
        d['language'] = do_prompt(__('Project language'), 'en')
        if d['language'] == 'en':
            d['language'] = None

    if 'suffix' not in d:
        print()
        print(__('The file name suffix for source files. Commonly, this is either ".txt"\n'
                 'or ".rst". Only files with this suffix are considered documents.'))
        d['suffix'] = do_prompt(__('Source file suffix'), '.rst', suffix)

    if 'master' not in d:
        print()
        print(__('One document is special in that it is considered the top node of the\n'
                 '"contents tree", that is, it is the root of the hierarchical structure\n'
                 'of the documents. Normally, this is "index", but if your "index"\n'
                 'document is a custom template, you can also set this to another filename.'))
        d['master'] = do_prompt(__('Name of your master document (without suffix)'), 'index')

    while path.isfile(path.join(d['path'], d['master'] + d['suffix'])) or \
            path.isfile(path.join(d['path'], 'source', d['master'] + d['suffix'])):
        print()
        print(bold(__('Error: the master file %s has already been found in the '
                      'selected root path.') % (d['master'] + d['suffix'])))
        print(__('sphinx-quickstart will not overwrite the existing file.'))
        print()
        d['master'] = do_prompt(__('Please enter a new file name, or rename the '
                                   'existing file and press Enter'), d['master'])

    if 'extensions' not in d:
        print(__('Indicate which of the following Sphinx extensions should be enabled:'))
        d['extensions'] = []
        for name, description in EXTENSIONS.items():
            if do_prompt('%s: %s (y/n)' % (name, description), 'n', boolean):
                d['extensions'].append('sphinx.ext.%s' % name)

        # Handle conflicting options
        if {'sphinx.ext.imgmath', 'sphinx.ext.mathjax'}.issubset(d['extensions']):
            print(__('Note: imgmath and mathjax cannot be enabled at the same time. '
                     'imgmath has been deselected.'))
            d['extensions'].remove('sphinx.ext.imgmath')

    if 'makefile' not in d:
        print()
        print(__('A Makefile and a Windows command file can be generated for you so that you\n'
                 'only have to run e.g. `make html\' instead of invoking sphinx-build\n'
                 'directly.'))
        d['makefile'] = do_prompt(__('Create Makefile? (y/n)'), 'y', boolean)

    if 'batchfile' not in d:
        d['batchfile'] = do_prompt(__('Create Windows command file? (y/n)'), 'y', boolean)
    print()
```
### 4 - sphinx/cmd/quickstart.py:

Start line: 122, End line: 157

```python
def suffix(x: str) -> str:
    if not (x[0:1] == '.' and len(x) > 1):
        raise ValidationError(__("Please enter a file suffix, e.g. '.rst' or '.txt'."))
    return x


def ok(x: str) -> str:
    return x


def do_prompt(text: str, default: str = None, validator: Callable[[str], Any] = nonempty) -> Union[str, bool]:  # NOQA
    while True:
        if default is not None:
            prompt = PROMPT_PREFIX + '%s [%s]: ' % (text, default)
        else:
            prompt = PROMPT_PREFIX + text + ': '
        if USE_LIBEDIT:
            # Note: libedit has a problem for combination of ``input()`` and escape
            # sequence (see #5335).  To avoid the problem, all prompts are not colored
            # on libedit.
            pass
        elif readline:
            # pass input_mode=True if readline available
            prompt = colorize(COLOR_QUESTION, prompt, input_mode=True)
        else:
            prompt = colorize(COLOR_QUESTION, prompt, input_mode=False)
        x = term_input(prompt).strip()
        if default and not x:
            x = default
        try:
            x = validator(x)
        except ValidationError as err:
            print(red('* ' + str(err)))
            continue
        break
    return x
```
### 5 - sphinx/cmd/quickstart.py:

Start line: 11, End line: 119

```python
import argparse
import locale
import os
import sys
import time
from collections import OrderedDict
from os import path
from typing import Any, Callable, Dict, List, Union

# try to import readline, unix specific enhancement
try:
    import readline
    if readline.__doc__ and 'libedit' in readline.__doc__:
        readline.parse_and_bind("bind ^I rl_complete")
        USE_LIBEDIT = True
    else:
        readline.parse_and_bind("tab: complete")
        USE_LIBEDIT = False
except ImportError:
    readline = None
    USE_LIBEDIT = False

from docutils.utils import column_width

import sphinx.locale
from sphinx import __display_version__, package_dir
from sphinx.locale import __
from sphinx.util.console import bold, color_terminal, colorize, nocolor, red  # type: ignore
from sphinx.util.osutil import ensuredir
from sphinx.util.template import SphinxRenderer

EXTENSIONS = OrderedDict([
    ('autodoc', __('automatically insert docstrings from modules')),
    ('doctest', __('automatically test code snippets in doctest blocks')),
    ('intersphinx', __('link between Sphinx documentation of different projects')),
    ('todo', __('write "todo" entries that can be shown or hidden on build')),
    ('coverage', __('checks for documentation coverage')),
    ('imgmath', __('include math, rendered as PNG or SVG images')),
    ('mathjax', __('include math, rendered in the browser by MathJax')),
    ('ifconfig', __('conditional inclusion of content based on config values')),
    ('viewcode', __('include links to the source code of documented Python objects')),
    ('githubpages', __('create .nojekyll file to publish the document on GitHub pages')),
])

DEFAULTS = {
    'path': '.',
    'sep': False,
    'dot': '_',
    'language': None,
    'suffix': '.rst',
    'master': 'index',
    'makefile': True,
    'batchfile': True,
}

PROMPT_PREFIX = '> '

if sys.platform == 'win32':
    # On Windows, show questions as bold because of color scheme of PowerShell (refs: #5294).
    COLOR_QUESTION = 'bold'
else:
    COLOR_QUESTION = 'purple'


# function to get input from terminal -- overridden by the test suite
def term_input(prompt: str) -> str:
    if sys.platform == 'win32':
        # Important: On windows, readline is not enabled by default.  In these
        #            environment, escape sequences have been broken.  To avoid the
        #            problem, quickstart uses ``print()`` to show prompt.
        print(prompt, end='')
        return input('')
    else:
        return input(prompt)


class ValidationError(Exception):
    """Raised for validation errors."""


def is_path(x: str) -> str:
    x = path.expanduser(x)
    if not path.isdir(x):
        raise ValidationError(__("Please enter a valid path name."))
    return x


def allow_empty(x: str) -> str:
    return x


def nonempty(x: str) -> str:
    if not x:
        raise ValidationError(__("Please enter some text."))
    return x


def choice(*l: str) -> Callable[[str], str]:
    def val(x: str) -> str:
        if x not in l:
            raise ValidationError(__('Please enter one of %s.') % ', '.join(l))
        return x
    return val


def boolean(x: str) -> bool:
    if x.upper() not in ('Y', 'YES', 'N', 'NO'):
        raise ValidationError(__("Please enter either 'y' or 'n'."))
    return x.upper() in ('Y', 'YES')
```
### 6 - sphinx/cmd/quickstart.py:

Start line: 453, End line: 519

```python
def get_parser() -> argparse.ArgumentParser:
    description = __(
        "\n"
        "Generate required files for a Sphinx project.\n"
        "\n"
        "sphinx-quickstart is an interactive tool that asks some questions about your\n"
        "project and then generates a complete documentation directory and sample\n"
        "Makefile to be used with sphinx-build.\n"
    )
    parser = argparse.ArgumentParser(
        usage='%(prog)s [OPTIONS] <PROJECT_DIR>',
        epilog=__("For more information, visit <http://sphinx-doc.org/>."),
        description=description)

    parser.add_argument('-q', '--quiet', action='store_true', dest='quiet',
                        default=None,
                        help=__('quiet mode'))
    parser.add_argument('--version', action='version', dest='show_version',
                        version='%%(prog)s %s' % __display_version__)

    parser.add_argument('path', metavar='PROJECT_DIR', default='.', nargs='?',
                        help=__('project root'))

    group = parser.add_argument_group(__('Structure options'))
    group.add_argument('--sep', action='store_true', dest='sep', default=None,
                       help=__('if specified, separate source and build dirs'))
    group.add_argument('--no-sep', action='store_false', dest='sep',
                       help=__('if specified, create build dir under source dir'))
    group.add_argument('--dot', metavar='DOT', default='_',
                       help=__('replacement for dot in _templates etc.'))

    group = parser.add_argument_group(__('Project basic options'))
    group.add_argument('-p', '--project', metavar='PROJECT', dest='project',
                       help=__('project name'))
    group.add_argument('-a', '--author', metavar='AUTHOR', dest='author',
                       help=__('author names'))
    group.add_argument('-v', metavar='VERSION', dest='version', default='',
                       help=__('version of project'))
    group.add_argument('-r', '--release', metavar='RELEASE', dest='release',
                       help=__('release of project'))
    group.add_argument('-l', '--language', metavar='LANGUAGE', dest='language',
                       help=__('document language'))
    group.add_argument('--suffix', metavar='SUFFIX', default='.rst',
                       help=__('source file suffix'))
    group.add_argument('--master', metavar='MASTER', default='index',
                       help=__('master document name'))
    group.add_argument('--epub', action='store_true', default=False,
                       help=__('use epub'))

    group = parser.add_argument_group(__('Extension options'))
    for ext in EXTENSIONS:
        group.add_argument('--ext-%s' % ext, action='append_const',
                           const='sphinx.ext.%s' % ext, dest='extensions',
                           help=__('enable %s extension') % ext)
    group.add_argument('--extensions', metavar='EXTENSIONS', dest='extensions',
                       action='append', help=__('enable arbitrary extensions'))

    group = parser.add_argument_group(__('Makefile and Batchfile creation'))
    group.add_argument('--makefile', action='store_true', dest='makefile', default=True,
                       help=__('create makefile'))
    group.add_argument('--no-makefile', action='store_false', dest='makefile',
                       help=__('do not create makefile'))
    group.add_argument('--batchfile', action='store_true', dest='batchfile', default=True,
                       help=__('create batchfile'))
    group.add_argument('--no-batchfile', action='store_false',
                       dest='batchfile',
                       help=__('do not create batchfile'))
    # ... other code
```
### 7 - sphinx/cmd/quickstart.py:

Start line: 424, End line: 450

```python
def valid_dir(d: Dict) -> bool:
    dir = d['path']
    if not path.exists(dir):
        return True
    if not path.isdir(dir):
        return False

    if {'Makefile', 'make.bat'} & set(os.listdir(dir)):
        return False

    if d['sep']:
        dir = os.path.join('source', dir)
        if not path.exists(dir):
            return True
        if not path.isdir(dir):
            return False

    reserved_names = [
        'conf.py',
        d['dot'] + 'static',
        d['dot'] + 'templates',
        d['master'] + d['suffix'],
    ]
    if set(reserved_names) & set(os.listdir(dir)):
        return False

    return True
```
### 8 - setup.py:

Start line: 176, End line: 252

```python
setup(
    name='Sphinx',
    version=sphinx.__version__,
    url='https://sphinx-doc.org/',
    download_url='https://pypi.org/project/Sphinx/',
    license='BSD',
    author='Georg Brandl',
    author_email='georg@python.org',
    description='Python documentation generator',
    long_description=long_desc,
    long_description_content_type='text/x-rst',
    project_urls={
        "Code": "https://github.com/sphinx-doc/sphinx",
        "Issue tracker": "https://github.com/sphinx-doc/sphinx/issues",
    },
    zip_safe=False,
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Environment :: Web Environment',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Science/Research',
        'Intended Audience :: System Administrators',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy',
        'Framework :: Setuptools Plugin',
        'Framework :: Sphinx',
        'Framework :: Sphinx :: Extension',
        'Framework :: Sphinx :: Theme',
        'Topic :: Documentation',
        'Topic :: Documentation :: Sphinx',
        'Topic :: Internet :: WWW/HTTP :: Site Management',
        'Topic :: Printing',
        'Topic :: Software Development',
        'Topic :: Software Development :: Documentation',
        'Topic :: Text Processing',
        'Topic :: Text Processing :: General',
        'Topic :: Text Processing :: Indexing',
        'Topic :: Text Processing :: Markup',
        'Topic :: Text Processing :: Markup :: HTML',
        'Topic :: Text Processing :: Markup :: LaTeX',
        'Topic :: Utilities',
    ],
    platforms='any',
    packages=find_packages(exclude=['tests', 'utils']),
    package_data = {
        'sphinx': ['py.typed'],
    },
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'sphinx-build = sphinx.cmd.build:main',
            'sphinx-quickstart = sphinx.cmd.quickstart:main',
            'sphinx-apidoc = sphinx.ext.apidoc:main',
            'sphinx-autogen = sphinx.ext.autosummary.generate:main',
        ],
        'distutils.commands': [
            'build_sphinx = sphinx.setup_command:BuildDoc',
        ],
    },
    python_requires=">=3.6",
    install_requires=install_requires,
    extras_require=extras_require,
    cmdclass=cmdclass,
)
```
### 9 - sphinx/cmd/quickstart.py:

Start line: 393, End line: 421

```python
def generate(d: Dict, overwrite: bool = True, silent: bool = False, templatedir: str = None
             ) -> None:
    # ... other code

    if d['makefile'] is True:
        d['rsrcdir'] = 'source' if d['sep'] else '.'
        d['rbuilddir'] = 'build' if d['sep'] else d['dot'] + 'build'
        # use binary mode, to avoid writing \r\n on Windows
        write_file(path.join(d['path'], 'Makefile'),
                   template.render(makefile_template, d), '\n')

    if d['batchfile'] is True:
        d['rsrcdir'] = 'source' if d['sep'] else '.'
        d['rbuilddir'] = 'build' if d['sep'] else d['dot'] + 'build'
        write_file(path.join(d['path'], 'make.bat'),
                   template.render(batchfile_template, d), '\r\n')

    if silent:
        return
    print()
    print(bold(__('Finished: An initial directory structure has been created.')))
    print()
    print(__('You should now populate your master file %s and create other documentation\n'
             'source files. ') % masterfile, end='')
    if d['makefile'] or d['batchfile']:
        print(__('Use the Makefile to build the docs, like so:\n'
                 '   make builder'))
    else:
        print(__('Use the sphinx-build command to build the docs, like so:\n'
                 '   sphinx-build -b builder %s %s') % (srcdir, builddir))
    print(__('where "builder" is one of the supported builders, '
             'e.g. html, latex or linkcheck.'))
    print()
```
### 10 - sphinx/cmd/quickstart.py:

Start line: 520, End line: 535

```python
def get_parser() -> argparse.ArgumentParser:
    # ... other code
    group.add_argument('-m', '--use-make-mode', action='store_true',
                       dest='make_mode', default=True,
                       help=__('use make-mode for Makefile/make.bat'))
    group.add_argument('-M', '--no-use-make-mode', action='store_false',
                       dest='make_mode',
                       help=__('do not use make-mode for Makefile/make.bat'))

    group = parser.add_argument_group(__('Project templating'))
    group.add_argument('-t', '--templatedir', metavar='TEMPLATEDIR',
                       dest='templatedir',
                       help=__('template directory for template files'))
    group.add_argument('-d', metavar='NAME=VALUE', action='append',
                       dest='variables',
                       help=__('define a template variable'))

    return parser
```
### 27 - sphinx/cmd/quickstart.py:

Start line: 160, End line: 182

```python
class QuickstartRenderer(SphinxRenderer):
    def __init__(self, templatedir: str) -> None:
        self.templatedir = templatedir or ''
        super().__init__()

    def _has_custom_template(self, template_name: str) -> bool:
        """Check if custom template file exists.

        Note: Please don't use this function from extensions.
              It will be removed in the future without deprecation period.
        """
        template = path.join(self.templatedir, path.basename(template_name))
        if self.templatedir and path.exists(template):
            return True
        else:
            return False

    def render(self, template_name: str, context: Dict) -> str:
        if self._has_custom_template(template_name):
            custom_template = path.join(self.templatedir, path.basename(template_name))
            return self.render_from_file(custom_template, context)
        else:
            return super().render(template_name, context)
```
### 28 - sphinx/cmd/quickstart.py:

Start line: 323, End line: 391

```python
def generate(d: Dict, overwrite: bool = True, silent: bool = False, templatedir: str = None
             ) -> None:
    """Generate project based on values in *d*."""
    template = QuickstartRenderer(templatedir=templatedir)

    if 'mastertoctree' not in d:
        d['mastertoctree'] = ''
    if 'mastertocmaxdepth' not in d:
        d['mastertocmaxdepth'] = 2

    d['root_doc'] = d['master']
    d['now'] = time.asctime()
    d['project_underline'] = column_width(d['project']) * '='
    d.setdefault('extensions', [])
    d['copyright'] = time.strftime('%Y') + ', ' + d['author']

    d["path"] = os.path.abspath(d['path'])
    ensuredir(d['path'])

    srcdir = path.join(d['path'], 'source') if d['sep'] else d['path']

    ensuredir(srcdir)
    if d['sep']:
        builddir = path.join(d['path'], 'build')
        d['exclude_patterns'] = ''
    else:
        builddir = path.join(srcdir, d['dot'] + 'build')
        exclude_patterns = map(repr, [
            d['dot'] + 'build',
            'Thumbs.db', '.DS_Store',
        ])
        d['exclude_patterns'] = ', '.join(exclude_patterns)
    ensuredir(builddir)
    ensuredir(path.join(srcdir, d['dot'] + 'templates'))
    ensuredir(path.join(srcdir, d['dot'] + 'static'))

    def write_file(fpath: str, content: str, newline: str = None) -> None:
        if overwrite or not path.isfile(fpath):
            if 'quiet' not in d:
                print(__('Creating file %s.') % fpath)
            with open(fpath, 'wt', encoding='utf-8', newline=newline) as f:
                f.write(content)
        else:
            if 'quiet' not in d:
                print(__('File %s already exists, skipping.') % fpath)

    conf_path = os.path.join(templatedir, 'conf.py_t') if templatedir else None
    if not conf_path or not path.isfile(conf_path):
        conf_path = os.path.join(package_dir, 'templates', 'quickstart', 'conf.py_t')
    with open(conf_path) as f:
        conf_text = f.read()

    write_file(path.join(srcdir, 'conf.py'), template.render_string(conf_text, d))

    masterfile = path.join(srcdir, d['master'] + d['suffix'])
    if template._has_custom_template('quickstart/master_doc.rst_t'):
        msg = ('A custom template `master_doc.rst_t` found. It has been renamed to '
               '`root_doc.rst_t`.  Please rename it on your project too.')
        print(colorize('red', msg))  # RemovedInSphinx60Warning
        write_file(masterfile, template.render('quickstart/master_doc.rst_t', d))
    else:
        write_file(masterfile, template.render('quickstart/root_doc.rst_t', d))

    if d.get('make_mode') is True:
        makefile_template = 'quickstart/Makefile.new_t'
        batchfile_template = 'quickstart/make.bat.new_t'
    else:
        makefile_template = 'quickstart/Makefile_t'
        batchfile_template = 'quickstart/make.bat_t'
    # ... other code
```
