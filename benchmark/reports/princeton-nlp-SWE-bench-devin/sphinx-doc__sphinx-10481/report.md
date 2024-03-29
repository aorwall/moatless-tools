# sphinx-doc__sphinx-10481

| **sphinx-doc/sphinx** | `004012b6df0fcec67312373f8d89327f5b09a7e6` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | - |
| **Avg pos** | - |
| **Min pos** | - |
| **Max pos** | - |
| **Top file pos** | 3 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sphinx/config.py b/sphinx/config.py
--- a/sphinx/config.py
+++ b/sphinx/config.py
@@ -163,6 +163,17 @@ def read(cls, confdir: str, overrides: Dict = None, tags: Tags = None) -> "Confi
             raise ConfigError(__("config directory doesn't contain a conf.py file (%s)") %
                               confdir)
         namespace = eval_config_file(filename, tags)
+
+        # Note: Old sphinx projects have been configured as "langugae = None" because
+        #       sphinx-quickstart previously generated this by default.
+        #       To keep compatibility, they should be fallback to 'en' for a while
+        #       (This conversion should not be removed before 2025-01-01).
+        if namespace.get("language", ...) is None:
+            logger.warning(__("Invalid configuration value found: 'language = None'. "
+                              "Update your configuration to a valid langauge code. "
+                              "Falling back to 'en' (English)."))
+            namespace["language"] = "en"
+
         return cls(namespace, overrides or {})
 
     def convert_overrides(self, name: str, value: Any) -> Any:

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sphinx/config.py | 166 | 166 | - | 3 | -


## Problem Statement

```
If a project defines "language = None" in conf.py, treat it like "en"
Hello, I started working on integrating Sphinx 5 to Fedora to ensure distribution packages work smoothly when the final is out.
I ran across is a side effect of the change inspired by #10062. 
If a project has already "language = None" defined in their conf.py (which, it seems, used to be an issue before [this](https://github.com/sphinx-doc/sphinx/commit/77b1d713a8d7b21ed6ad0f0a3d9f13a391b0a605) commit), the new behavior will cause the documentation build to error out. The projects created after the mentioned commit seem not to be affected.
In a sample of ~40 packages, 2 have run across this issue. 
A naive check using [grep.app](https://grep.app/search?current=3&q=language%20%3D%20None&filter[lang][0]=Python&filter[path.pattern][0]=/conf.py) shows that for a half a million indexed GitHub projects there is around 6k which have the string in their conf.py (I removed the commented strings from the equation).
For older projects using Sphinx, this change will be disruptive and will require the same commit in the same place for each and every one of them.

The exact error:
\`\`\`
+ python3 setup.py build_sphinx
running build_sphinx
Running Sphinx v5.0.0b1
loading translations [None]... not available for built-in messages
making output directory... done
WARNING: The config value `language' has type `NoneType'; expected `str'.

Extension error (sphinx.config):
Handler <function check_confval_types at 0x7fd1e67a6c00> for event 'config-inited' threw an exception (exception: 'NoneType' object has no attribute 'startswith')
\`\`\`

**Describe the solution you'd like**
When Sphinx encounters NoneType for language, it could set the language to English and log the fact for the user (possibly asking them to make adjustments to conf.py) instead of erroring.
It's not that different than the current behavior in 5.0.0b1. When if I run `sphinx-quickstart` and set no language, the variable is not present at all in conf.py, although in the background my project is processed as English. 

**Describe alternatives you've considered**
Aforementioned manual change for each affected project, which I'm afraid of.



```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 sphinx/locale/__init__.py | 233 | 261| 222 | 222 | 2046 | 
| 2 | 2 setup.py | 57 | 133| 644 | 866 | 3084 | 
| 3 | **3 sphinx/config.py** | 83 | 143| 756 | 1622 | 7595 | 
| 4 | 4 sphinx/__init__.py | 1 | 54| 539 | 2161 | 8135 | 
| 5 | 4 setup.py | 1 | 55| 394 | 2555 | 8135 | 
| 6 | 5 sphinx/builders/latex/__init__.py | 442 | 502| 518 | 3073 | 13674 | 
| 7 | 6 sphinx/builders/changes.py | 117 | 160| 438 | 3511 | 15142 | 
| 8 | **6 sphinx/config.py** | 467 | 486| 223 | 3734 | 15142 | 
| 9 | 7 doc/conf.py | 165 | 187| 269 | 4003 | 16867 | 
| 10 | 8 sphinx/application.py | 318 | 368| 407 | 4410 | 28673 | 
| 11 | 9 sphinx/errors.py | 68 | 125| 297 | 4707 | 29415 | 
| 12 | 10 sphinx/environment/__init__.py | 1 | 80| 569 | 5276 | 34917 | 
| 13 | 11 sphinx/util/pycompat.py | 1 | 38| 335 | 5611 | 35366 | 
| 14 | 12 sphinx/setup_command.py | 136 | 187| 440 | 6051 | 36905 | 
| 15 | 13 sphinx/cmd/quickstart.py | 183 | 258| 675 | 6726 | 42438 | 
| 16 | 14 sphinx/util/docutils.py | 169 | 189| 191 | 6917 | 47138 | 
| 17 | 14 sphinx/cmd/quickstart.py | 1 | 111| 767 | 7684 | 47138 | 
| 18 | 14 sphinx/setup_command.py | 6 | 83| 431 | 8115 | 47138 | 
| 19 | 14 sphinx/builders/latex/__init__.py | 35 | 90| 825 | 8940 | 47138 | 
| 20 | 15 sphinx/builders/latex/constants.py | 66 | 116| 537 | 9477 | 49336 | 
| 21 | 15 sphinx/cmd/quickstart.py | 536 | 603| 491 | 9968 | 49336 | 
| 22 | 16 sphinx/builders/html/__init__.py | 1331 | 1410| 1012 | 10980 | 61949 | 
| 23 | 16 sphinx/util/docutils.py | 145 | 166| 181 | 11161 | 61949 | 
| 24 | 16 sphinx/cmd/quickstart.py | 260 | 318| 739 | 11900 | 61949 | 
| 25 | 16 sphinx/builders/changes.py | 43 | 116| 805 | 12705 | 61949 | 
| 26 | 16 doc/conf.py | 84 | 140| 502 | 13207 | 61949 | 
| 27 | 16 sphinx/locale/__init__.py | 143 | 160| 132 | 13339 | 61949 | 
| 28 | **16 sphinx/config.py** | 412 | 464| 478 | 13817 | 61949 | 
| 29 | 16 sphinx/util/docutils.py | 125 | 142| 132 | 13949 | 61949 | 
| 30 | 16 sphinx/setup_command.py | 85 | 112| 230 | 14179 | 61949 | 
| 31 | 17 utils/babel_runner.py | 106 | 172| 382 | 14561 | 63040 | 
| 32 | 18 sphinx/ext/autodoc/__init__.py | 2797 | 2837| 535 | 15096 | 86699 | 
| 33 | 19 sphinx/builders/gettext.py | 282 | 298| 168 | 15264 | 89185 | 
| 34 | 19 sphinx/application.py | 286 | 299| 132 | 15396 | 89185 | 
| 35 | **19 sphinx/config.py** | 489 | 503| 154 | 15550 | 89185 | 
| 36 | 20 sphinx/builders/manpage.py | 104 | 123| 166 | 15716 | 90135 | 
| 37 | 21 utils/bump_version.py | 67 | 102| 207 | 15923 | 91527 | 
| 38 | 21 sphinx/builders/latex/constants.py | 118 | 209| 1040 | 16963 | 91527 | 
| 39 | 22 sphinx/transforms/i18n.py | 236 | 390| 1618 | 18581 | 96405 | 
| 40 | 23 sphinx/builders/texinfo.py | 195 | 218| 242 | 18823 | 98438 | 
| 41 | 24 sphinx/builders/latex/util.py | 23 | 49| 231 | 19054 | 98825 | 
| 42 | 24 sphinx/cmd/quickstart.py | 451 | 517| 752 | 19806 | 98825 | 
| 43 | 25 sphinx/domains/python.py | 1447 | 1461| 110 | 19916 | 111282 | 
| 44 | 26 sphinx/cmd/make_mode.py | 1 | 47| 596 | 20512 | 112937 | 
| 45 | 26 sphinx/environment/__init__.py | 83 | 186| 993 | 21505 | 112937 | 
| 46 | 27 sphinx/cmd/build.py | 95 | 162| 763 | 22268 | 115598 | 
| 47 | 28 sphinx/builders/linkcheck.py | 560 | 591| 325 | 22593 | 120372 | 
| 48 | 29 sphinx/ext/intersphinx.py | 629 | 643| 166 | 22759 | 126234 | 
| 49 | 30 sphinx/directives/code.py | 1 | 22| 148 | 22907 | 130040 | 
| 50 | 31 sphinx/roles.py | 339 | 381| 410 | 23317 | 133709 | 
| 51 | 32 sphinx/transforms/__init__.py | 420 | 442| 182 | 23499 | 136989 | 
| 52 | 33 sphinx/directives/patches.py | 1 | 33| 256 | 23755 | 138894 | 
| 53 | 33 sphinx/builders/latex/__init__.py | 505 | 538| 404 | 24159 | 138894 | 
| 54 | 34 sphinx/writers/texinfo.py | 210 | 250| 433 | 24592 | 151176 | 
| 55 | **34 sphinx/config.py** | 209 | 222| 126 | 24718 | 151176 | 
| 56 | 34 sphinx/environment/__init__.py | 287 | 313| 273 | 24991 | 151176 | 
| 57 | 35 sphinx/directives/__init__.py | 260 | 275| 122 | 25113 | 153321 | 
| 58 | 36 sphinx/directives/other.py | 1 | 31| 240 | 25353 | 156408 | 
| 59 | 36 sphinx/domains/python.py | 1063 | 1075| 132 | 25485 | 156408 | 
| 60 | 37 sphinx/util/cfamily.py | 274 | 293| 184 | 25669 | 160085 | 
| 61 | 38 sphinx/domains/changeset.py | 41 | 99| 516 | 26185 | 161290 | 
| 62 | 38 sphinx/domains/python.py | 1019 | 1037| 140 | 26325 | 161290 | 
| 63 | 39 sphinx/builders/dummy.py | 1 | 45| 232 | 26557 | 161522 | 
| 64 | 39 sphinx/application.py | 53 | 115| 494 | 27051 | 161522 | 
| 65 | 40 sphinx/project.py | 1 | 31| 233 | 27284 | 162251 | 
| 66 | 40 sphinx/environment/__init__.py | 440 | 518| 687 | 27971 | 162251 | 
| 67 | 41 sphinx/events.py | 1 | 45| 333 | 28304 | 163151 | 
| 68 | 41 sphinx/builders/latex/__init__.py | 204 | 240| 482 | 28786 | 163151 | 
| 69 | 42 sphinx/builders/latex/transforms.py | 608 | 625| 142 | 28928 | 167438 | 
| 70 | 42 sphinx/transforms/i18n.py | 121 | 235| 1056 | 29984 | 167438 | 
| 71 | 42 sphinx/transforms/i18n.py | 391 | 473| 999 | 30983 | 167438 | 
| 72 | 42 sphinx/directives/code.py | 399 | 462| 642 | 31625 | 167438 | 
| 73 | 43 sphinx/builders/epub3.py | 235 | 278| 607 | 32232 | 169984 | 
| 74 | **43 sphinx/config.py** | 1 | 47| 331 | 32563 | 169984 | 
| 75 | 44 sphinx/writers/html5.py | 671 | 757| 686 | 33249 | 177364 | 
| 76 | 44 sphinx/builders/epub3.py | 179 | 215| 415 | 33664 | 177364 | 
| 77 | 44 sphinx/transforms/__init__.py | 1 | 38| 250 | 33914 | 177364 | 
| 78 | 44 sphinx/application.py | 1 | 50| 382 | 34296 | 177364 | 
| 79 | 45 sphinx/ext/apidoc.py | 365 | 393| 374 | 34670 | 181575 | 
| 80 | 46 sphinx/domains/std.py | 234 | 251| 122 | 34792 | 191444 | 
| 81 | 46 sphinx/directives/code.py | 25 | 48| 176 | 34968 | 191444 | 
| 82 | 46 sphinx/setup_command.py | 114 | 134| 172 | 35140 | 191444 | 
| 83 | 47 sphinx/transforms/post_transforms/code.py | 106 | 135| 208 | 35348 | 192396 | 
| 84 | 47 sphinx/domains/python.py | 1422 | 1444| 208 | 35556 | 192396 | 
| 85 | 47 sphinx/cmd/quickstart.py | 321 | 389| 766 | 36322 | 192396 | 
| 86 | 48 sphinx/util/i18n.py | 232 | 259| 238 | 36560 | 194735 | 
| 87 | **48 sphinx/config.py** | 224 | 244| 174 | 36734 | 194735 | 
| 88 | 49 sphinx/pygments_styles.py | 1 | 27| 143 | 36877 | 195384 | 
| 89 | 49 utils/bump_version.py | 119 | 146| 267 | 37144 | 195384 | 
| 90 | 49 sphinx/cmd/make_mode.py | 89 | 133| 375 | 37519 | 195384 | 
| 91 | 49 sphinx/writers/html5.py | 144 | 208| 605 | 38124 | 195384 | 
| 92 | 49 utils/bump_version.py | 149 | 180| 224 | 38348 | 195384 | 
| 93 | 50 sphinx/transforms/references.py | 1 | 46| 274 | 38622 | 195658 | 
| 94 | 50 sphinx/builders/changes.py | 1 | 18| 129 | 38751 | 195658 | 
| 95 | 50 sphinx/builders/gettext.py | 162 | 186| 189 | 38940 | 195658 | 
| 96 | 50 sphinx/writers/texinfo.py | 852 | 960| 839 | 39779 | 195658 | 
| 97 | 50 sphinx/util/docutils.py | 272 | 296| 230 | 40009 | 195658 | 
| 98 | 50 sphinx/cmd/build.py | 1 | 24| 158 | 40167 | 195658 | 
| 99 | 50 sphinx/transforms/i18n.py | 475 | 495| 230 | 40397 | 195658 | 
| 100 | 51 sphinx/builders/text.py | 75 | 88| 114 | 40511 | 196305 | 


## Patch

```diff
diff --git a/sphinx/config.py b/sphinx/config.py
--- a/sphinx/config.py
+++ b/sphinx/config.py
@@ -163,6 +163,17 @@ def read(cls, confdir: str, overrides: Dict = None, tags: Tags = None) -> "Confi
             raise ConfigError(__("config directory doesn't contain a conf.py file (%s)") %
                               confdir)
         namespace = eval_config_file(filename, tags)
+
+        # Note: Old sphinx projects have been configured as "langugae = None" because
+        #       sphinx-quickstart previously generated this by default.
+        #       To keep compatibility, they should be fallback to 'en' for a while
+        #       (This conversion should not be removed before 2025-01-01).
+        if namespace.get("language", ...) is None:
+            logger.warning(__("Invalid configuration value found: 'language = None'. "
+                              "Update your configuration to a valid langauge code. "
+                              "Falling back to 'en' (English)."))
+            namespace["language"] = "en"
+
         return cls(namespace, overrides or {})
 
     def convert_overrides(self, name: str, value: Any) -> Any:

```

## Test Patch

```diff
diff --git a/tests/test_config.py b/tests/test_config.py
--- a/tests/test_config.py
+++ b/tests/test_config.py
@@ -381,3 +381,49 @@ def test_nitpick_ignore_regex_fullmatch(app, status, warning):
     assert len(warning) == len(nitpick_warnings)
     for actual, expected in zip(warning, nitpick_warnings):
         assert expected in actual
+
+
+def test_conf_py_language_none(tempdir):
+    """Regression test for #10474."""
+
+    # Given a conf.py file with language = None
+    (tempdir / 'conf.py').write_text("language = None", encoding='utf-8')
+
+    # When we load conf.py into a Config object
+    cfg = Config.read(tempdir, {}, None)
+    cfg.init_values()
+
+    # Then the language is coerced to English
+    assert cfg.language == "en"
+
+
+@mock.patch("sphinx.config.logger")
+def test_conf_py_language_none_warning(logger, tempdir):
+    """Regression test for #10474."""
+
+    # Given a conf.py file with language = None
+    (tempdir / 'conf.py').write_text("language = None", encoding='utf-8')
+
+    # When we load conf.py into a Config object
+    Config.read(tempdir, {}, None)
+
+    # Then a warning is raised
+    assert logger.warning.called
+    assert logger.warning.call_args[0][0] == (
+        "Invalid configuration value found: 'language = None'. "
+        "Update your configuration to a valid langauge code. "
+        "Falling back to 'en' (English).")
+
+
+def test_conf_py_no_language(tempdir):
+    """Regression test for #10474."""
+
+    # Given a conf.py file with no language attribute
+    (tempdir / 'conf.py').write_text("", encoding='utf-8')
+
+    # When we load conf.py into a Config object
+    cfg = Config.read(tempdir, {}, None)
+    cfg.init_values()
+
+    # Then the language is coerced to English
+    assert cfg.language == "en"

```


## Code snippets

### 1 - sphinx/locale/__init__.py:

Start line: 233, End line: 261

```python
# A shortcut for sphinx-core
#: Translation function for messages on documentation (menu, labels, themes and so on).
#: This function follows :confval:`language` setting.
_ = get_translation('sphinx')
#: Translation function for console messages
#: This function follows locale setting (`LC_ALL`, `LC_MESSAGES` and so on).
__ = get_translation('sphinx', 'console')


# labels
admonitionlabels = {
    'attention': _('Attention'),
    'caution':   _('Caution'),
    'danger':    _('Danger'),
    'error':     _('Error'),
    'hint':      _('Hint'),
    'important': _('Important'),
    'note':      _('Note'),
    'seealso':   _('See also'),
    'tip':       _('Tip'),
    'warning':   _('Warning'),
}

# Moved to sphinx.directives.other (will be overridden later)
versionlabels: Dict[str, str] = {}

# Moved to sphinx.domains.python (will be overridden later)
pairindextypes: Dict[str, str] = {}
```
### 2 - setup.py:

Start line: 57, End line: 133

```python
setup(
    name='Sphinx',
    version=sphinx.__version__,
    url='https://www.sphinx-doc.org/',
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
        'Programming Language :: Python :: 3.10',
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
)
```
### 3 - sphinx/config.py:

Start line: 83, End line: 143

```python
class Config:

    config_values: Dict[str, Tuple] = {
        # general options
        'project': ('Python', 'env', []),
        'author': ('unknown', 'env', []),
        'project_copyright': ('', 'html', [str]),
        'copyright': (lambda c: c.project_copyright, 'html', [str]),
        'version': ('', 'env', []),
        'release': ('', 'env', []),
        'today': ('', 'env', []),
        # the real default is locale-dependent
        'today_fmt': (None, 'env', [str]),

        'language': ('en', 'env', [str]),
        'locale_dirs': (['locales'], 'env', []),
        'figure_language_filename': ('{root}.{language}{ext}', 'env', [str]),
        'gettext_allow_fuzzy_translations': (False, 'gettext', []),

        'master_doc': ('index', 'env', []),
        'root_doc': (lambda config: config.master_doc, 'env', []),
        'source_suffix': ({'.rst': 'restructuredtext'}, 'env', Any),
        'source_encoding': ('utf-8-sig', 'env', []),
        'exclude_patterns': ([], 'env', []),
        'default_role': (None, 'env', [str]),
        'add_function_parentheses': (True, 'env', []),
        'add_module_names': (True, 'env', []),
        'trim_footnote_reference_space': (False, 'env', []),
        'show_authors': (False, 'env', []),
        'pygments_style': (None, 'html', [str]),
        'highlight_language': ('default', 'env', []),
        'highlight_options': ({}, 'env', []),
        'templates_path': ([], 'html', []),
        'template_bridge': (None, 'html', [str]),
        'keep_warnings': (False, 'env', []),
        'suppress_warnings': ([], 'env', []),
        'modindex_common_prefix': ([], 'html', []),
        'rst_epilog': (None, 'env', [str]),
        'rst_prolog': (None, 'env', [str]),
        'trim_doctest_flags': (True, 'env', []),
        'primary_domain': ('py', 'env', [NoneType]),
        'needs_sphinx': (None, None, [str]),
        'needs_extensions': ({}, None, []),
        'manpages_url': (None, 'env', []),
        'nitpicky': (False, None, []),
        'nitpick_ignore': ([], None, []),
        'nitpick_ignore_regex': ([], None, []),
        'numfig': (False, 'env', []),
        'numfig_secnum_depth': (1, 'env', []),
        'numfig_format': ({}, 'env', []),  # will be initialized in init_numfig_format()

        'math_number_all': (False, 'env', []),
        'math_eqref_format': (None, 'env', [str]),
        'math_numfig': (True, 'env', []),
        'tls_verify': (True, 'env', []),
        'tls_cacerts': (None, 'env', []),
        'user_agent': (None, 'env', [str]),
        'smartquotes': (True, 'env', []),
        'smartquotes_action': ('qDe', 'env', []),
        'smartquotes_excludes': ({'languages': ['ja'],
                                  'builders': ['man', 'text']},
                                 'env', []),
    }
```
### 4 - sphinx/__init__.py:

Start line: 1, End line: 54

```python
"""The Sphinx documentation toolchain."""

# Keep this file executable as-is in Python 3!
# (Otherwise getting the version out of it from setup.py is impossible.)

import os
import subprocess
import warnings
from os import path
from subprocess import PIPE

from .deprecation import RemovedInNextVersionWarning

# by default, all DeprecationWarning under sphinx package will be emit.
# Users can avoid this by using environment variable: PYTHONWARNINGS=
if 'PYTHONWARNINGS' not in os.environ:
    warnings.filterwarnings('default', category=RemovedInNextVersionWarning)
# docutils.io using mode='rU' for open
warnings.filterwarnings('ignore', "'U' mode is deprecated",
                        DeprecationWarning, module='docutils.io')
warnings.filterwarnings('ignore', 'The frontend.Option class .*',
                        DeprecationWarning, module='docutils.frontend')

__version__ = '5.0.0b1'
__released__ = '5.0.0b1'  # used when Sphinx builds its own docs

#: Version info for better programmatic use.
#:
#: A tuple of five elements; for Sphinx version 1.2.1 beta 3 this would be
#: ``(1, 2, 1, 'beta', 3)``. The fourth element can be one of: ``alpha``,
#: ``beta``, ``rc``, ``final``. ``final`` always has 0 as the last element.
#:
#: .. versionadded:: 1.2
#:    Before version 1.2, check the string ``sphinx.__version__``.
version_info = (5, 0, 0, 'beta', 1)

package_dir = path.abspath(path.dirname(__file__))

__display_version__ = __version__  # used for command line version
if __version__.endswith('+'):
    # try to find out the commit hash if checked out from git, and append
    # it to __version__ (since we use this value from setup.py, it gets
    # automatically propagated to an installed copy as well)
    __display_version__ = __version__
    __version__ = __version__[:-1]  # remove '+' for PEP-440 version spec.
    try:
        ret = subprocess.run(['git', 'show', '-s', '--pretty=format:%h'],
                             cwd=package_dir,
                             stdout=PIPE, stderr=PIPE, encoding='ascii')
        if ret.stdout:
            __display_version__ += '/' + ret.stdout.strip()
    except Exception:
        pass
```
### 5 - setup.py:

Start line: 1, End line: 55

```python
import sys

from setuptools import find_packages, setup

import sphinx

with open('README.rst', encoding='utf-8') as f:
    long_desc = f.read()

if sys.version_info < (3, 6):
    print('ERROR: Sphinx requires at least Python 3.6 to run.')
    sys.exit(1)

install_requires = [
    'sphinxcontrib-applehelp',
    'sphinxcontrib-devhelp',
    'sphinxcontrib-jsmath',
    'sphinxcontrib-htmlhelp>=2.0.0',
    'sphinxcontrib-serializinghtml>=1.1.5',
    'sphinxcontrib-qthelp',
    'Jinja2>=2.3',
    'Pygments>=2.0',
    'docutils>=0.14,<0.19',
    'snowballstemmer>=1.1',
    'babel>=1.3',
    'alabaster>=0.7,<0.8',
    'imagesize',
    'requests>=2.5.0',
    'packaging',
    "importlib-metadata>=4.4; python_version < '3.10'",
]

extras_require = {
    # Environment Marker works for wheel 0.24 or later
    ':sys_platform=="win32"': [
        'colorama>=0.3.5',
    ],
    'docs': [
        'sphinxcontrib-websupport',
    ],
    'lint': [
        'flake8>=3.5.0',
        'isort',
        'mypy>=0.950',
        'docutils-stubs',
        "types-typed-ast",
        "types-requests",
    ],
    'test': [
        'pytest>=4.6',
        'html5lib',
        "typed_ast; python_version < '3.8'",
        'cython',
    ],
}
```
### 6 - sphinx/builders/latex/__init__.py:

Start line: 442, End line: 502

```python
def validate_config_values(app: Sphinx, config: Config) -> None:
    for key in list(config.latex_elements):
        if key not in DEFAULT_SETTINGS:
            msg = __("Unknown configure key: latex_elements[%r], ignored.")
            logger.warning(msg % (key,))
            config.latex_elements.pop(key)


def validate_latex_theme_options(app: Sphinx, config: Config) -> None:
    for key in list(config.latex_theme_options):
        if key not in Theme.UPDATABLE_KEYS:
            msg = __("Unknown theme option: latex_theme_options[%r], ignored.")
            logger.warning(msg % (key,))
            config.latex_theme_options.pop(key)


def install_packages_for_ja(app: Sphinx) -> None:
    """Install packages for Japanese."""
    if app.config.language == 'ja' and app.config.latex_engine in ('platex', 'uplatex'):
        app.add_latex_package('pxjahyper', after_hyperref=True)


def default_latex_engine(config: Config) -> str:
    """ Better default latex_engine settings for specific languages. """
    if config.language == 'ja':
        return 'uplatex'
    elif config.language.startswith('zh'):
        return 'xelatex'
    elif config.language == 'el':
        return 'xelatex'
    else:
        return 'pdflatex'


def default_latex_docclass(config: Config) -> Dict[str, str]:
    """ Better default latex_docclass settings for specific languages. """
    if config.language == 'ja':
        if config.latex_engine == 'uplatex':
            return {'manual': 'ujbook',
                    'howto': 'ujreport'}
        else:
            return {'manual': 'jsbook',
                    'howto': 'jreport'}
    else:
        return {}


def default_latex_use_xindy(config: Config) -> bool:
    """ Better default latex_use_xindy settings for specific engines. """
    return config.latex_engine in {'xelatex', 'lualatex'}


def default_latex_documents(config: Config) -> List[Tuple[str, str, str, str, str]]:
    """ Better default latex_documents settings. """
    project = texescape.escape(config.project, config.latex_engine)
    author = texescape.escape(config.author, config.latex_engine)
    return [(config.root_doc,
             make_filename_from_project(config.project) + '.tex',
             texescape.escape_abbr(project),
             texescape.escape_abbr(author),
             config.latex_theme)]
```
### 7 - sphinx/builders/changes.py:

Start line: 117, End line: 160

```python
class ChangesBuilder(Builder):

    def write(self, *ignored: Any) -> None:
        # ... other code
        for docname in self.env.all_docs:
            with open(self.env.doc2path(docname),
                      encoding=self.env.config.source_encoding) as f:
                try:
                    lines = f.readlines()
                except UnicodeDecodeError:
                    logger.warning(__('could not read %r for changelog creation'), docname)
                    continue
            targetfn = path.join(self.outdir, 'rst', os_path(docname)) + '.html'
            ensuredir(path.dirname(targetfn))
            with open(targetfn, 'w', encoding='utf-8') as f:
                text = ''.join(hl(i + 1, line) for (i, line) in enumerate(lines))
                ctx = {
                    'filename': self.env.doc2path(docname, None),
                    'text': text
                }
                f.write(self.templates.render('changes/rstsource.html', ctx))
        themectx = {'theme_' + key: val for (key, val) in
                    self.theme.get_options({}).items()}
        copy_asset_file(path.join(package_dir, 'themes', 'default', 'static', 'default.css_t'),
                        self.outdir, context=themectx, renderer=self.templates)
        copy_asset_file(path.join(package_dir, 'themes', 'basic', 'static', 'basic.css'),
                        self.outdir)

    def hl(self, text: str, version: str) -> str:
        text = html.escape(text)
        for directive in ('versionchanged', 'versionadded', 'deprecated'):
            text = text.replace('.. %s:: %s' % (directive, version),
                                '<b>.. %s:: %s</b>' % (directive, version))
        return text

    def finish(self) -> None:
        pass


def setup(app: Sphinx) -> Dict[str, Any]:
    app.add_builder(ChangesBuilder)

    return {
        'version': 'builtin',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
```
### 8 - sphinx/config.py:

Start line: 467, End line: 486

```python
def check_primary_domain(app: "Sphinx", config: Config) -> None:
    primary_domain = config.primary_domain
    if primary_domain and not app.registry.has_domain(primary_domain):
        logger.warning(__('primary_domain %r not found, ignored.'), primary_domain)
        config.primary_domain = None  # type: ignore


def check_root_doc(app: "Sphinx", env: "BuildEnvironment", added: Set[str],
                   changed: Set[str], removed: Set[str]) -> Set[str]:
    """Adjust root_doc to 'contents' to support an old project which does not have
    any root_doc setting.
    """
    if (app.config.root_doc == 'index' and
            'index' not in app.project.docnames and
            'contents' in app.project.docnames):
        logger.warning(__('Since v2.0, Sphinx uses "index" as root_doc by default. '
                          'Please add "root_doc = \'contents\'" to your conf.py.'))
        app.config.root_doc = "contents"  # type: ignore

    return changed
```
### 9 - doc/conf.py:

Start line: 165, End line: 187

```python
def setup(app):
    from sphinx.ext.autodoc import cut_lines
    from sphinx.util.docfields import GroupedField
    app.connect('autodoc-process-docstring', cut_lines(4, what=['module']))
    app.connect('source-read', linkify_issues_in_changelog)
    app.add_object_type('confval', 'confval',
                        objname='configuration value',
                        indextemplate='pair: %s; configuration value')
    app.add_object_type('setuptools-confval', 'setuptools-confval',
                        objname='setuptools configuration value',
                        indextemplate='pair: %s; setuptools configuration value')
    fdesc = GroupedField('parameter', label='Parameters',
                         names=['param'], can_collapse=True)
    app.add_object_type('event', 'event', 'pair: %s; event', parse_event,
                        doc_field_types=[fdesc])

    # workaround for RTD
    from sphinx.util import logging
    logger = logging.getLogger(__name__)
    app.info = lambda *args, **kwargs: logger.info(*args, **kwargs)
    app.warn = lambda *args, **kwargs: logger.warning(*args, **kwargs)
    app.debug = lambda *args, **kwargs: logger.debug(*args, **kwargs)
```
### 10 - sphinx/application.py:

Start line: 318, End line: 368

```python
class Sphinx:

    def build(self, force_all: bool = False, filenames: List[str] = None) -> None:
        self.phase = BuildPhase.READING
        try:
            if force_all:
                self.builder.compile_all_catalogs()
                self.builder.build_all()
            elif filenames:
                self.builder.compile_specific_catalogs(filenames)
                self.builder.build_specific(filenames)
            else:
                self.builder.compile_update_catalogs()
                self.builder.build_update()

            self.events.emit('build-finished', None)
        except Exception as err:
            # delete the saved env to force a fresh build next time
            envfile = path.join(self.doctreedir, ENV_PICKLE_FILENAME)
            if path.isfile(envfile):
                os.unlink(envfile)
            self.events.emit('build-finished', err)
            raise

        if self._warncount and self.keep_going:
            self.statuscode = 1

        status = (__('succeeded') if self.statuscode == 0
                  else __('finished with problems'))
        if self._warncount:
            if self.warningiserror:
                if self._warncount == 1:
                    msg = __('build %s, %s warning (with warnings treated as errors).')
                else:
                    msg = __('build %s, %s warnings (with warnings treated as errors).')
            else:
                if self._warncount == 1:
                    msg = __('build %s, %s warning.')
                else:
                    msg = __('build %s, %s warnings.')

            logger.info(bold(msg % (status, self._warncount)))
        else:
            logger.info(bold(__('build %s.') % status))

        if self.statuscode == 0 and self.builder.epilog:
            logger.info('')
            logger.info(self.builder.epilog % {
                'outdir': relpath(self.outdir),
                'project': self.config.project
            })

        self.builder.cleanup()
```
### 28 - sphinx/config.py:

Start line: 412, End line: 464

```python
def check_confval_types(app: "Sphinx", config: Config) -> None:
    """Check all values for deviation from the default value's type, since
    that can result in TypeErrors all over the place NB.
    """
    for confval in config:
        default, rebuild, annotations = config.values[confval.name]

        if callable(default):
            default = default(config)  # evaluate default value
        if default is None and not annotations:
            continue  # neither inferable nor expliclitly annotated types

        if annotations is Any:
            # any type of value is accepted
            pass
        elif isinstance(annotations, ENUM):
            if not annotations.match(confval.value):
                msg = __("The config value `{name}` has to be a one of {candidates}, "
                         "but `{current}` is given.")
                logger.warning(msg.format(name=confval.name,
                                          current=confval.value,
                                          candidates=annotations.candidates), once=True)
        else:
            if type(confval.value) is type(default):
                continue
            if type(confval.value) in annotations:
                continue

            common_bases = (set(type(confval.value).__bases__ + (type(confval.value),)) &
                            set(type(default).__bases__))
            common_bases.discard(object)
            if common_bases:
                continue  # at least we share a non-trivial base class

            if annotations:
                msg = __("The config value `{name}' has type `{current.__name__}'; "
                         "expected {permitted}.")
                wrapped_annotations = ["`{}'".format(c.__name__) for c in annotations]
                if len(wrapped_annotations) > 2:
                    permitted = "{}, or {}".format(
                        ", ".join(wrapped_annotations[:-1]),
                        wrapped_annotations[-1])
                else:
                    permitted = " or ".join(wrapped_annotations)
                logger.warning(msg.format(name=confval.name,
                                          current=type(confval.value),
                                          permitted=permitted), once=True)
            else:
                msg = __("The config value `{name}' has type `{current.__name__}', "
                         "defaults to `{default.__name__}'.")
                logger.warning(msg.format(name=confval.name,
                                          current=type(confval.value),
                                          default=type(default)), once=True)
```
### 35 - sphinx/config.py:

Start line: 489, End line: 503

```python
def setup(app: "Sphinx") -> Dict[str, Any]:
    app.connect('config-inited', convert_source_suffix, priority=800)
    app.connect('config-inited', convert_highlight_options, priority=800)
    app.connect('config-inited', init_numfig_format, priority=800)
    app.connect('config-inited', correct_copyright_year, priority=800)
    app.connect('config-inited', check_confval_types, priority=800)
    app.connect('config-inited', check_primary_domain, priority=800)
    app.connect('env-get-outdated', check_root_doc)

    return {
        'version': 'builtin',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
```
### 55 - sphinx/config.py:

Start line: 209, End line: 222

```python
class Config:

    def pre_init_values(self) -> None:
        """
        Initialize some limited config variables before initializing i18n and loading
        extensions.
        """
        variables = ['needs_sphinx', 'suppress_warnings', 'language', 'locale_dirs']
        for name in variables:
            try:
                if name in self.overrides:
                    self.__dict__[name] = self.convert_overrides(name, self.overrides[name])
                elif name in self._raw_config:
                    self.__dict__[name] = self._raw_config[name]
            except ValueError as exc:
                logger.warning("%s", exc)
```
### 74 - sphinx/config.py:

Start line: 1, End line: 47

```python
"""Build configuration file handling."""

import re
import traceback
import types
from collections import OrderedDict
from os import getenv, path
from typing import (TYPE_CHECKING, Any, Callable, Dict, Generator, Iterator, List, NamedTuple,
                    Optional, Set, Tuple, Union)

from sphinx.errors import ConfigError, ExtensionError
from sphinx.locale import _, __
from sphinx.util import logging
from sphinx.util.i18n import format_date
from sphinx.util.osutil import cd, fs_encoding
from sphinx.util.tags import Tags
from sphinx.util.typing import NoneType

if TYPE_CHECKING:
    from sphinx.application import Sphinx
    from sphinx.environment import BuildEnvironment

logger = logging.getLogger(__name__)

CONFIG_FILENAME = 'conf.py'
UNSERIALIZABLE_TYPES = (type, types.ModuleType, types.FunctionType)
copyright_year_re = re.compile(r'^((\d{4}-)?)(\d{4})(?=[ ,])')


class ConfigValue(NamedTuple):
    name: str
    value: Any
    rebuild: Union[bool, str]


def is_serializable(obj: Any) -> bool:
    """Check if object is serializable or not."""
    if isinstance(obj, UNSERIALIZABLE_TYPES):
        return False
    elif isinstance(obj, dict):
        for key, value in obj.items():
            if not is_serializable(key) or not is_serializable(value):
                return False
    elif isinstance(obj, (list, tuple, set)):
        return all(is_serializable(i) for i in obj)

    return True
```
### 87 - sphinx/config.py:

Start line: 224, End line: 244

```python
class Config:

    def init_values(self) -> None:
        config = self._raw_config
        for valname, value in self.overrides.items():
            try:
                if '.' in valname:
                    realvalname, key = valname.split('.', 1)
                    config.setdefault(realvalname, {})[key] = value
                    continue
                elif valname not in self.values:
                    logger.warning(__('unknown config value %r in override, ignoring'),
                                   valname)
                    continue
                if isinstance(value, str):
                    config[valname] = self.convert_overrides(valname, value)
                else:
                    config[valname] = value
            except ValueError as exc:
                logger.warning("%s", exc)
        for name in config:
            if name in self.values:
                self.__dict__[name] = config[name]
```
