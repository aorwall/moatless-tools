# pytest-dev__pytest-6202

| **pytest-dev/pytest** | `3a668ea6ff24b0c8f00498c3144c63bac561d925` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | - |
| **Avg pos** | - |
| **Min pos** | - |
| **Max pos** | - |
| **Top file pos** | 16 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/src/_pytest/python.py b/src/_pytest/python.py
--- a/src/_pytest/python.py
+++ b/src/_pytest/python.py
@@ -285,8 +285,7 @@ def getmodpath(self, stopatmodule=True, includemodule=False):
                     break
             parts.append(name)
         parts.reverse()
-        s = ".".join(parts)
-        return s.replace(".[", "[")
+        return ".".join(parts)
 
     def reportinfo(self):
         # XXX caching?

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| src/_pytest/python.py | 288 | 289 | - | 16 | -


## Problem Statement

```
'.['  replaced with '[' in the headline shown of the test report
\`\`\`
bug.py F                                                                 [100%]

=================================== FAILURES ===================================
_________________________________ test_boo[.[] _________________________________

a = '..['

    @pytest.mark.parametrize("a",["..["])
    def test_boo(a):
>       assert 0
E       assert 0

bug.py:6: AssertionError
============================== 1 failed in 0.06s ===============================
\`\`\`

The `"test_boo[..[]"` replaced with `"test_boo[.[]"` in the headline shown with long report output.

**The same problem also causing the vscode-python test discovery error.**

## What causing the problem

I trace back the source code.

[https://github.com/pytest-dev/pytest/blob/92d6a0500b9f528a9adcd6bbcda46ebf9b6baf03/src/_pytest/reports.py#L129-L149](https://github.com/pytest-dev/pytest/blob/92d6a0500b9f528a9adcd6bbcda46ebf9b6baf03/src/_pytest/reports.py#L129-L149)

The headline comes from line 148.

[https://github.com/pytest-dev/pytest/blob/92d6a0500b9f528a9adcd6bbcda46ebf9b6baf03/src/_pytest/nodes.py#L432-L441](https://github.com/pytest-dev/pytest/blob/92d6a0500b9f528a9adcd6bbcda46ebf9b6baf03/src/_pytest/nodes.py#L432-L441)

`location` comes from line 437 `location = self.reportinfo()`

[https://github.com/pytest-dev/pytest/blob/92d6a0500b9f528a9adcd6bbcda46ebf9b6baf03/src/_pytest/python.py#L294-L308](https://github.com/pytest-dev/pytest/blob/92d6a0500b9f528a9adcd6bbcda46ebf9b6baf03/src/_pytest/python.py#L294-L308)

The headline comes from line 306 `modpath = self.getmodpath() `

[https://github.com/pytest-dev/pytest/blob/92d6a0500b9f528a9adcd6bbcda46ebf9b6baf03/src/_pytest/python.py#L274-L292](https://github.com/pytest-dev/pytest/blob/92d6a0500b9f528a9adcd6bbcda46ebf9b6baf03/src/_pytest/python.py#L274-L292)

This line of code `return s.replace(".[", "[")` causes the problem. We should replace it with `return s`. After changing this, run `tox -e linting,py37`, pass all the tests and resolve this issue. But I can't find this line of code for what purpose.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 src/pytest.py | 1 | 107| 708 | 708 | 708 | 
| 2 | 2 src/_pytest/reports.py | 129 | 149| 117 | 825 | 4085 | 
| 3 | 3 src/_pytest/doctest.py | 219 | 275| 495 | 1320 | 8585 | 
| 4 | 4 src/_pytest/terminal.py | 905 | 920| 187 | 1507 | 16817 | 
| 5 | 4 src/_pytest/terminal.py | 873 | 903| 250 | 1757 | 16817 | 
| 6 | 5 doc/en/example/xfail_demo.py | 1 | 39| 143 | 1900 | 16961 | 
| 7 | 5 src/_pytest/reports.py | 1 | 31| 244 | 2144 | 16961 | 
| 8 | 5 src/_pytest/terminal.py | 386 | 435| 420 | 2564 | 16961 | 
| 9 | 5 src/_pytest/terminal.py | 722 | 745| 142 | 2706 | 16961 | 
| 10 | 6 testing/python/metafunc.py | 1308 | 1346| 243 | 2949 | 30155 | 
| 11 | 6 src/_pytest/doctest.py | 87 | 128| 311 | 3260 | 30155 | 
| 12 | 6 src/_pytest/terminal.py | 811 | 832| 180 | 3440 | 30155 | 
| 13 | 7 src/_pytest/pastebin.py | 83 | 104| 181 | 3621 | 30960 | 
| 14 | 8 src/_pytest/_code/code.py | 1 | 32| 164 | 3785 | 39172 | 
| 15 | 8 src/_pytest/terminal.py | 700 | 720| 191 | 3976 | 39172 | 
| 16 | 9 testing/python/collect.py | 1205 | 1230| 186 | 4162 | 48448 | 
| 17 | 9 testing/python/collect.py | 559 | 658| 662 | 4824 | 48448 | 
| 18 | 10 src/_pytest/_code/source.py | 231 | 259| 162 | 4986 | 50830 | 
| 19 | 10 testing/python/collect.py | 1055 | 1073| 154 | 5140 | 50830 | 
| 20 | 10 testing/python/collect.py | 1146 | 1172| 181 | 5321 | 50830 | 
| 21 | 11 src/_pytest/junitxml.py | 249 | 277| 217 | 5538 | 55884 | 
| 22 | 12 src/_pytest/runner.py | 160 | 191| 238 | 5776 | 58731 | 
| 23 | 12 testing/python/collect.py | 881 | 918| 321 | 6097 | 58731 | 
| 24 | 13 src/_pytest/skipping.py | 120 | 178| 529 | 6626 | 60205 | 
| 25 | 13 testing/python/collect.py | 1 | 33| 225 | 6851 | 60205 | 
| 26 | 13 src/_pytest/terminal.py | 922 | 947| 199 | 7050 | 60205 | 
| 27 | 13 src/_pytest/skipping.py | 28 | 65| 363 | 7413 | 60205 | 
| 28 | 13 src/_pytest/junitxml.py | 214 | 231| 161 | 7574 | 60205 | 
| 29 | 13 src/_pytest/junitxml.py | 233 | 247| 132 | 7706 | 60205 | 
| 30 | 14 testing/python/fixtures.py | 2475 | 3503| 6208 | 13914 | 86511 | 
| 31 | 14 src/_pytest/terminal.py | 256 | 280| 161 | 14075 | 86511 | 
| 32 | 15 doc/en/example/assertion/failure_demo.py | 43 | 121| 680 | 14755 | 88170 | 
| 33 | **16 src/_pytest/python.py** | 1268 | 1337| 487 | 15242 | 99606 | 
| 34 | 16 src/_pytest/terminal.py | 834 | 847| 125 | 15367 | 99606 | 
| 35 | 16 testing/python/collect.py | 1034 | 1053| 167 | 15534 | 99606 | 
| 36 | 17 src/_pytest/nodes.py | 1 | 26| 148 | 15682 | 102901 | 
| 37 | 18 doc/en/example/py2py3/conftest.py | 1 | 17| 0 | 15682 | 102987 | 
| 38 | 18 src/_pytest/terminal.py | 295 | 316| 165 | 15847 | 102987 | 
| 39 | 18 src/_pytest/junitxml.py | 161 | 212| 333 | 16180 | 102987 | 
| 40 | 19 doc/en/conftest.py | 1 | 2| 0 | 16180 | 102994 | 
| 41 | 19 src/_pytest/terminal.py | 573 | 593| 187 | 16367 | 102994 | 
| 42 | 19 src/_pytest/doctest.py | 509 | 534| 237 | 16604 | 102994 | 
| 43 | 19 testing/python/collect.py | 196 | 249| 320 | 16924 | 102994 | 
| 44 | 19 testing/python/fixtures.py | 85 | 1065| 6303 | 23227 | 102994 | 
| 45 | 19 src/_pytest/terminal.py | 785 | 809| 203 | 23430 | 102994 | 
| 46 | 19 src/_pytest/terminal.py | 612 | 645| 331 | 23761 | 102994 | 
| 47 | 19 src/_pytest/reports.py | 182 | 191| 131 | 23892 | 102994 | 
| 48 | **19 src/_pytest/python.py** | 1210 | 1236| 208 | 24100 | 102994 | 
| 49 | 20 src/_pytest/_io/saferepr.py | 1 | 21| 159 | 24259 | 103486 | 
| 50 | 21 testing/python/integration.py | 1 | 35| 239 | 24498 | 106411 | 
| 51 | 21 testing/python/collect.py | 279 | 344| 471 | 24969 | 106411 | 
| 52 | 21 src/_pytest/_code/code.py | 1064 | 1074| 138 | 25107 | 106411 | 
| 53 | 21 testing/python/fixtures.py | 2074 | 2473| 2558 | 27665 | 106411 | 
| 54 | 22 doc/en/example/conftest.py | 1 | 2| 0 | 27665 | 106418 | 
| 55 | 22 src/_pytest/doctest.py | 477 | 507| 291 | 27956 | 106418 | 
| 56 | 23 src/_pytest/compat.py | 147 | 185| 244 | 28200 | 108923 | 
| 57 | 23 src/_pytest/_code/code.py | 998 | 1021| 173 | 28373 | 108923 | 
| 58 | 24 extra/get_issues.py | 55 | 86| 231 | 28604 | 109466 | 
| 59 | 24 src/_pytest/terminal.py | 747 | 783| 299 | 28903 | 109466 | 
| 60 | 24 src/_pytest/terminal.py | 336 | 349| 123 | 29026 | 109466 | 
| 61 | 25 testing/example_scripts/issue88_initial_file_multinodes/conftest.py | 1 | 15| 0 | 29026 | 109519 | 
| 62 | 25 testing/python/collect.py | 920 | 939| 178 | 29204 | 109519 | 
| 63 | 25 testing/python/integration.py | 37 | 68| 226 | 29430 | 109519 | 
| 64 | 25 src/_pytest/doctest.py | 392 | 411| 159 | 29589 | 109519 | 
| 65 | 25 src/_pytest/terminal.py | 506 | 544| 323 | 29912 | 109519 | 
| 66 | 25 testing/python/collect.py | 1017 | 1032| 133 | 30045 | 109519 | 
| 67 | 25 testing/python/fixtures.py | 1067 | 2072| 6201 | 36246 | 109519 | 
| 68 | 25 src/_pytest/junitxml.py | 118 | 159| 339 | 36585 | 109519 | 
| 69 | 25 src/_pytest/terminal.py | 140 | 167| 216 | 36801 | 109519 | 
| 70 | 26 testing/example_scripts/issue_519.py | 1 | 31| 350 | 37151 | 109985 | 
| 71 | 27 src/_pytest/outcomes.py | 39 | 54| 124 | 37275 | 111499 | 
| 72 | 27 src/_pytest/_code/code.py | 920 | 961| 294 | 37569 | 111499 | 
| 73 | 27 src/_pytest/_code/code.py | 614 | 657| 307 | 37876 | 111499 | 
| 74 | 27 testing/python/collect.py | 59 | 77| 189 | 38065 | 111499 | 
| 75 | 28 testing/python/setup_only.py | 63 | 94| 190 | 38255 | 113185 | 
| 76 | 29 testing/freeze/tox_run.py | 1 | 13| 0 | 38255 | 113270 | 
| 77 | 30 src/_pytest/pytester.py | 311 | 347| 210 | 38465 | 123606 | 
| 78 | 30 src/_pytest/terminal.py | 668 | 684| 122 | 38587 | 123606 | 
| 79 | 30 testing/python/collect.py | 1106 | 1143| 209 | 38796 | 123606 | 
| 80 | 30 src/_pytest/_code/code.py | 964 | 995| 277 | 39073 | 123606 | 
| 81 | 31 src/_pytest/faulthandler.py | 1 | 36| 249 | 39322 | 124181 | 
| 82 | 31 testing/python/fixtures.py | 3505 | 4210| 4568 | 43890 | 124181 | 
| 83 | 31 src/_pytest/_code/code.py | 695 | 719| 236 | 44126 | 124181 | 
| 84 | 31 testing/python/fixtures.py | 1 | 82| 492 | 44618 | 124181 | 
| 85 | 31 src/_pytest/nodes.py | 258 | 303| 373 | 44991 | 124181 | 
| 86 | 31 src/_pytest/pytester.py | 116 | 137| 243 | 45234 | 124181 | 
| 87 | 32 doc/en/example/costlysetup/conftest.py | 1 | 21| 0 | 45234 | 124261 | 
| 88 | 32 src/_pytest/junitxml.py | 1 | 81| 624 | 45858 | 124261 | 
| 89 | 32 doc/en/example/assertion/failure_demo.py | 191 | 203| 114 | 45972 | 124261 | 
| 90 | 32 src/_pytest/_code/code.py | 865 | 888| 185 | 46157 | 124261 | 
| 91 | 32 doc/en/example/assertion/failure_demo.py | 164 | 188| 159 | 46316 | 124261 | 
| 92 | 32 src/_pytest/reports.py | 34 | 127| 572 | 46888 | 124261 | 
| 93 | 32 doc/en/example/assertion/failure_demo.py | 256 | 283| 161 | 47049 | 124261 | 
| 94 | 32 src/_pytest/terminal.py | 437 | 456| 198 | 47247 | 124261 | 
| 95 | 33 src/_pytest/helpconfig.py | 139 | 206| 550 | 47797 | 125950 | 
| 96 | 33 src/_pytest/doctest.py | 313 | 339| 198 | 47995 | 125950 | 
| 97 | 33 src/_pytest/doctest.py | 1 | 34| 251 | 48246 | 125950 | 
| 98 | 33 src/_pytest/doctest.py | 180 | 217| 303 | 48549 | 125950 | 
| 99 | 33 src/_pytest/reports.py | 392 | 407| 151 | 48700 | 125950 | 
| 100 | 34 src/_pytest/fixtures.py | 737 | 761| 214 | 48914 | 137660 | 
| 101 | 35 src/_pytest/config/__init__.py | 223 | 233| 154 | 49068 | 146437 | 
| 102 | **35 src/_pytest/python.py** | 291 | 305| 141 | 49209 | 146437 | 
| 103 | 36 src/_pytest/debugging.py | 319 | 334| 117 | 49326 | 148868 | 
| 104 | 36 testing/python/setup_only.py | 34 | 60| 167 | 49493 | 148868 | 
| 105 | 36 src/_pytest/terminal.py | 282 | 293| 165 | 49658 | 148868 | 
| 106 | 36 src/_pytest/terminal.py | 686 | 698| 119 | 49777 | 148868 | 
| 107 | 37 testing/example_scripts/conftest_usageerror/conftest.py | 1 | 9| 0 | 49777 | 148899 | 
| 108 | 38 src/_pytest/python_api.py | 1 | 41| 229 | 50006 | 155469 | 
| 109 | 38 src/_pytest/terminal.py | 546 | 571| 258 | 50264 | 155469 | 
| 110 | 38 src/_pytest/pytester.py | 1 | 30| 175 | 50439 | 155469 | 
| 111 | **38 src/_pytest/python.py** | 111 | 129| 176 | 50615 | 155469 | 
| 112 | 39 testing/example_scripts/fixtures/fill_fixtures/test_extend_fixture_conftest_conftest/pkg/conftest.py | 1 | 7| 0 | 50615 | 155487 | 
| 113 | 39 src/_pytest/helpconfig.py | 209 | 245| 252 | 50867 | 155487 | 
| 114 | 39 testing/python/collect.py | 730 | 753| 170 | 51037 | 155487 | 
| 115 | 39 src/_pytest/terminal.py | 351 | 363| 121 | 51158 | 155487 | 
| 116 | 40 src/_pytest/main.py | 311 | 362| 303 | 51461 | 160783 | 
| 117 | 40 src/_pytest/_code/code.py | 216 | 240| 199 | 51660 | 160783 | 
| 118 | 41 testing/example_scripts/fixtures/fill_fixtures/test_extend_fixture_conftest_conftest/conftest.py | 1 | 7| 0 | 51660 | 160797 | 
| 119 | 41 src/_pytest/compat.py | 1 | 46| 225 | 51885 | 160797 | 
| 120 | 41 testing/python/metafunc.py | 1157 | 1186| 199 | 52084 | 160797 | 
| 121 | 41 src/_pytest/_code/code.py | 721 | 752| 310 | 52394 | 160797 | 
| 122 | 42 src/_pytest/setuponly.py | 47 | 80| 242 | 52636 | 161335 | 
| 123 | 42 src/_pytest/terminal.py | 218 | 241| 203 | 52839 | 161335 | 
| 124 | 42 src/_pytest/outcomes.py | 139 | 154| 130 | 52969 | 161335 | 
| 125 | 43 setup.py | 1 | 16| 156 | 53125 | 161606 | 
| 126 | 43 testing/python/collect.py | 35 | 57| 187 | 53312 | 161606 | 
| 127 | 44 doc/en/_themes/flask_theme_support.py | 1 | 88| 1273 | 54585 | 162879 | 
| 128 | 44 src/_pytest/pastebin.py | 41 | 54| 145 | 54730 | 162879 | 
| 129 | 44 src/_pytest/terminal.py | 50 | 137| 630 | 55360 | 162879 | 
| 130 | 44 src/_pytest/fixtures.py | 1 | 106| 687 | 56047 | 162879 | 
| 131 | 44 testing/python/setup_only.py | 230 | 249| 122 | 56169 | 162879 | 
| 132 | 44 testing/python/collect.py | 788 | 810| 195 | 56364 | 162879 | 
| 133 | 45 src/_pytest/mark/structures.py | 1 | 34| 188 | 56552 | 165742 | 
| 134 | 45 src/_pytest/pytester.py | 1211 | 1235| 162 | 56714 | 165742 | 
| 135 | 46 src/_pytest/unittest.py | 242 | 283| 286 | 57000 | 167739 | 
| 136 | 46 src/_pytest/fixtures.py | 764 | 793| 242 | 57242 | 167739 | 
| 137 | 47 src/_pytest/resultlog.py | 62 | 79| 144 | 57386 | 168460 | 
| 138 | 47 setup.py | 19 | 40| 115 | 57501 | 168460 | 
| 139 | 47 doc/en/example/assertion/failure_demo.py | 206 | 253| 228 | 57729 | 168460 | 
| 140 | 48 testing/example_scripts/fixtures/fill_fixtures/test_extend_fixture_conftest_module/conftest.py | 1 | 7| 0 | 57729 | 168474 | 
| 141 | 48 testing/python/collect.py | 540 | 557| 182 | 57911 | 168474 | 
| 142 | 48 src/_pytest/_code/code.py | 754 | 780| 210 | 58121 | 168474 | 
| 143 | 48 src/_pytest/junitxml.py | 446 | 465| 143 | 58264 | 168474 | 
| 144 | 48 testing/python/metafunc.py | 304 | 325| 229 | 58493 | 168474 | 
| 145 | 49 src/_pytest/monkeypatch.py | 39 | 65| 180 | 58673 | 170889 | 
| 146 | 50 src/_pytest/_code/__init__.py | 1 | 11| 0 | 58673 | 170990 | 
| 147 | 51 doc/en/conf.py | 1 | 116| 789 | 59462 | 173449 | 
| 148 | 51 testing/python/setup_only.py | 186 | 201| 119 | 59581 | 173449 | 
| 149 | 51 src/_pytest/runner.py | 251 | 280| 269 | 59850 | 173449 | 
| 150 | 51 testing/python/setup_only.py | 124 | 154| 183 | 60033 | 173449 | 
| 151 | 52 src/_pytest/assertion/rewrite.py | 367 | 398| 240 | 60273 | 182480 | 
| 152 | 53 testing/python/raises.py | 187 | 202| 135 | 60408 | 184202 | 
| 153 | 53 src/_pytest/pytester.py | 85 | 114| 193 | 60601 | 184202 | 
| 154 | 54 doc/en/example/costlysetup/sub_a/__init__.py | 1 | 2| 0 | 60601 | 184203 | 
| 155 | 54 src/_pytest/runner.py | 121 | 135| 136 | 60737 | 184203 | 
| 156 | 54 src/_pytest/unittest.py | 106 | 155| 349 | 61086 | 184203 | 
| 157 | 54 testing/python/collect.py | 1076 | 1103| 183 | 61269 | 184203 | 
| 158 | 54 doc/en/conf.py | 241 | 341| 656 | 61925 | 184203 | 
| 159 | 54 testing/python/metafunc.py | 891 | 910| 129 | 62054 | 184203 | 
| 160 | 54 src/_pytest/reports.py | 372 | 390| 163 | 62217 | 184203 | 
| 161 | 54 src/_pytest/assertion/rewrite.py | 1 | 36| 229 | 62446 | 184203 | 
| 162 | 54 testing/python/metafunc.py | 1250 | 1282| 199 | 62645 | 184203 | 
| 163 | 55 src/_pytest/__init__.py | 1 | 9| 0 | 62645 | 184259 | 
| 164 | 55 src/_pytest/pastebin.py | 1 | 38| 268 | 62913 | 184259 | 
| 165 | 55 src/_pytest/terminal.py | 849 | 871| 206 | 63119 | 184259 | 
| 166 | 55 src/_pytest/terminal.py | 647 | 666| 153 | 63272 | 184259 | 
| 167 | 55 doc/en/example/assertion/failure_demo.py | 124 | 161| 143 | 63415 | 184259 | 
| 168 | 56 src/_pytest/stepwise.py | 73 | 109| 290 | 63705 | 184973 | 
| 169 | 56 src/_pytest/reports.py | 444 | 484| 318 | 64023 | 184973 | 
| 170 | 57 src/_pytest/config/exceptions.py | 1 | 10| 0 | 64023 | 185018 | 
| 171 | 57 src/_pytest/resultlog.py | 81 | 98| 159 | 64182 | 185018 | 
| 172 | 58 doc/en/example/costlysetup/sub_b/__init__.py | 1 | 2| 0 | 64182 | 185019 | 
| 173 | 58 src/_pytest/runner.py | 41 | 67| 232 | 64414 | 185019 | 
| 174 | 58 testing/python/raises.py | 1 | 50| 300 | 64714 | 185019 | 
| 175 | 58 src/_pytest/terminal.py | 199 | 215| 142 | 64856 | 185019 | 
| 176 | 58 extra/get_issues.py | 33 | 52| 143 | 64999 | 185019 | 
| 177 | 58 testing/python/setup_only.py | 97 | 121| 150 | 65149 | 185019 | 
| 178 | 58 src/_pytest/doctest.py | 536 | 564| 291 | 65440 | 185019 | 
| 179 | 58 doc/en/example/assertion/failure_demo.py | 1 | 40| 169 | 65609 | 185019 | 
| 180 | 59 testing/example_scripts/fixtures/fill_fixtures/test_conftest_funcargs_only_available_in_subdir/sub1/conftest.py | 1 | 8| 0 | 65609 | 185045 | 
| 181 | 60 doc/en/example/assertion/global_testmodule_config/conftest.py | 1 | 15| 0 | 65609 | 185126 | 
| 182 | 60 src/_pytest/terminal.py | 1010 | 1043| 240 | 65849 | 185126 | 
| 183 | 60 testing/python/metafunc.py | 64 | 86| 178 | 66027 | 185126 | 
| 184 | 60 src/_pytest/main.py | 616 | 630| 132 | 66159 | 185126 | 
| 185 | 60 testing/python/integration.py | 71 | 86| 109 | 66268 | 185126 | 
| 186 | **60 src/_pytest/python.py** | 497 | 545| 468 | 66736 | 185126 | 
| 187 | 60 testing/python/collect.py | 846 | 878| 304 | 67040 | 185126 | 
| 188 | 60 testing/python/collect.py | 755 | 786| 204 | 67244 | 185126 | 
| 189 | 60 src/_pytest/junitxml.py | 508 | 533| 177 | 67421 | 185126 | 
| 190 | 60 testing/python/setup_only.py | 157 | 183| 167 | 67588 | 185126 | 
| 191 | **60 src/_pytest/python.py** | 1 | 53| 373 | 67961 | 185126 | 
| 192 | 61 testing/example_scripts/fixtures/fill_fixtures/test_conftest_funcargs_only_available_in_subdir/sub2/conftest.py | 1 | 7| 0 | 67961 | 185151 | 
| 193 | 62 testing/example_scripts/config/collect_pytest_prefix/conftest.py | 1 | 3| 0 | 67961 | 185159 | 
| 194 | 63 src/_pytest/config/findpaths.py | 85 | 107| 157 | 68118 | 186317 | 
| 195 | 63 doc/en/conf.py | 118 | 240| 811 | 68929 | 186317 | 
| 196 | 63 src/_pytest/_code/source.py | 286 | 326| 371 | 69300 | 186317 | 
| 197 | 63 src/_pytest/unittest.py | 223 | 239| 138 | 69438 | 186317 | 
| 198 | 63 testing/example_scripts/issue_519.py | 34 | 52| 115 | 69553 | 186317 | 
| 199 | 63 testing/python/metafunc.py | 1013 | 1027| 112 | 69665 | 186317 | 
| 200 | 63 src/_pytest/outcomes.py | 115 | 136| 174 | 69839 | 186317 | 
| 201 | 63 src/_pytest/pytester.py | 542 | 562| 177 | 70016 | 186317 | 
| 202 | 63 src/_pytest/terminal.py | 595 | 610| 128 | 70144 | 186317 | 
| 203 | 63 src/_pytest/doctest.py | 278 | 291| 130 | 70274 | 186317 | 
| 204 | 63 src/_pytest/fixtures.py | 683 | 734| 470 | 70744 | 186317 | 
| 205 | **63 src/_pytest/python.py** | 1238 | 1265| 217 | 70961 | 186317 | 
| 206 | 63 src/_pytest/_code/code.py | 1024 | 1044| 150 | 71111 | 186317 | 
| 207 | 63 src/_pytest/pytester.py | 58 | 83| 201 | 71312 | 186317 | 
| 208 | 63 testing/python/metafunc.py | 526 | 565| 329 | 71641 | 186317 | 
| 209 | 64 extra/setup-py.test/setup.py | 1 | 12| 0 | 71641 | 186395 | 
| 210 | 64 testing/python/metafunc.py | 968 | 994| 196 | 71837 | 186395 | 
| 211 | 64 src/_pytest/outcomes.py | 57 | 89| 198 | 72035 | 186395 | 
| 212 | 65 doc/en/example/nonpython/conftest.py | 1 | 47| 314 | 72349 | 186709 | 
| 213 | 65 testing/python/metafunc.py | 1476 | 1508| 225 | 72574 | 186709 | 
| 214 | 65 src/_pytest/python_api.py | 720 | 737| 149 | 72723 | 186709 | 
| 215 | 66 src/_pytest/mark/__init__.py | 77 | 95| 141 | 72864 | 187880 | 
| 216 | 66 testing/python/metafunc.py | 88 | 132| 448 | 73312 | 187880 | 
| 217 | 66 src/_pytest/reports.py | 300 | 318| 143 | 73455 | 187880 | 
| 218 | 66 src/_pytest/assertion/rewrite.py | 966 | 980| 181 | 73636 | 187880 | 
| 219 | 66 testing/python/metafunc.py | 362 | 376| 144 | 73780 | 187880 | 
| 220 | 66 src/_pytest/_code/code.py | 432 | 452| 188 | 73968 | 187880 | 
| 221 | 66 src/_pytest/terminal.py | 365 | 384| 196 | 74164 | 187880 | 
| 222 | 66 testing/python/setup_only.py | 273 | 295| 143 | 74307 | 187880 | 
| 223 | 67 testing/python/approx.py | 427 | 443| 137 | 74444 | 193681 | 
| 224 | 67 testing/python/collect.py | 1266 | 1303| 320 | 74764 | 193681 | 
| 225 | 67 src/_pytest/main.py | 44 | 152| 757 | 75521 | 193681 | 
| 226 | 67 src/_pytest/junitxml.py | 645 | 675| 261 | 75782 | 193681 | 
| 227 | 67 testing/python/metafunc.py | 944 | 966| 170 | 75952 | 193681 | 
| 228 | 67 testing/python/collect.py | 713 | 728| 143 | 76095 | 193681 | 
| 229 | 67 testing/python/metafunc.py | 1763 | 1800| 253 | 76348 | 193681 | 
| 230 | 67 src/_pytest/_code/code.py | 659 | 693| 330 | 76678 | 193681 | 
| 231 | 67 src/_pytest/fixtures.py | 660 | 680| 132 | 76810 | 193681 | 
| 232 | 67 testing/python/metafunc.py | 913 | 942| 221 | 77031 | 193681 | 
| 233 | 67 testing/python/metafunc.py | 1640 | 1654| 132 | 77163 | 193681 | 
| 234 | 67 testing/python/metafunc.py | 1624 | 1638| 127 | 77290 | 193681 | 
| 235 | 67 src/_pytest/config/__init__.py | 755 | 767| 139 | 77429 | 193681 | 
| 236 | 67 src/_pytest/doctest.py | 413 | 435| 208 | 77637 | 193681 | 
| 237 | 67 src/_pytest/reports.py | 321 | 344| 171 | 77808 | 193681 | 
| 238 | 68 testing/example_scripts/collect/package_infinite_recursion/conftest.py | 1 | 3| 0 | 77808 | 193691 | 
| 239 | 68 testing/python/metafunc.py | 501 | 524| 149 | 77957 | 193691 | 


### Hint

```
Thanks for the fantastic report @linw1995, this is really helpful :smile: 
I find out the purpose of replacing '.[' with '['. The older version of pytest, support to generate test by using the generator function.

[https://github.com/pytest-dev/pytest/blob/9eb1d55380ae7c25ffc600b65e348dca85f99221/py/test/testing/test_collect.py#L137-L153](https://github.com/pytest-dev/pytest/blob/9eb1d55380ae7c25ffc600b65e348dca85f99221/py/test/testing/test_collect.py#L137-L153)

[https://github.com/pytest-dev/pytest/blob/9eb1d55380ae7c25ffc600b65e348dca85f99221/py/test/pycollect.py#L263-L276](https://github.com/pytest-dev/pytest/blob/9eb1d55380ae7c25ffc600b65e348dca85f99221/py/test/pycollect.py#L263-L276)

the name of the generated test case function is `'[0]'`. And its parent is `'test_gen'`. The line of code `return s.replace('.[', '[')` avoids its `modpath` becoming `test_gen.[0]`. Since the yield tests were removed in pytest 4.0, this line of code can be replaced with `return s` safely.


@linw1995 
Great find and investigation.
Do you want to create a PR for it?
```

## Patch

```diff
diff --git a/src/_pytest/python.py b/src/_pytest/python.py
--- a/src/_pytest/python.py
+++ b/src/_pytest/python.py
@@ -285,8 +285,7 @@ def getmodpath(self, stopatmodule=True, includemodule=False):
                     break
             parts.append(name)
         parts.reverse()
-        s = ".".join(parts)
-        return s.replace(".[", "[")
+        return ".".join(parts)
 
     def reportinfo(self):
         # XXX caching?

```

## Test Patch

```diff
diff --git a/testing/test_collection.py b/testing/test_collection.py
--- a/testing/test_collection.py
+++ b/testing/test_collection.py
@@ -685,6 +685,8 @@ def test_2():
     def test_example_items1(self, testdir):
         p = testdir.makepyfile(
             """
+            import pytest
+
             def testone():
                 pass
 
@@ -693,19 +695,24 @@ def testmethod_one(self):
                     pass
 
             class TestY(TestX):
-                pass
+                @pytest.mark.parametrize("arg0", [".["])
+                def testmethod_two(self, arg0):
+                    pass
         """
         )
         items, reprec = testdir.inline_genitems(p)
-        assert len(items) == 3
+        assert len(items) == 4
         assert items[0].name == "testone"
         assert items[1].name == "testmethod_one"
         assert items[2].name == "testmethod_one"
+        assert items[3].name == "testmethod_two[.[]"
 
         # let's also test getmodpath here
         assert items[0].getmodpath() == "testone"
         assert items[1].getmodpath() == "TestX.testmethod_one"
         assert items[2].getmodpath() == "TestY.testmethod_one"
+        # PR #6202: Fix incorrect result of getmodpath method. (Resolves issue #6189)
+        assert items[3].getmodpath() == "TestY.testmethod_two[.[]"
 
         s = items[0].getmodpath(stopatmodule=False)
         assert s.endswith("test_example_items1.testone")

```


## Code snippets

### 1 - src/pytest.py:

Start line: 1, End line: 107

```python
# PYTHON_ARGCOMPLETE_OK
"""
pytest: unit and functional testing with Python.
"""
from _pytest import __version__
from _pytest.assertion import register_assert_rewrite
from _pytest.config import cmdline
from _pytest.config import hookimpl
from _pytest.config import hookspec
from _pytest.config import main
from _pytest.config import UsageError
from _pytest.debugging import pytestPDB as __pytestPDB
from _pytest.fixtures import fillfixtures as _fillfuncargs
from _pytest.fixtures import fixture
from _pytest.fixtures import yield_fixture
from _pytest.freeze_support import freeze_includes
from _pytest.main import ExitCode
from _pytest.main import Session
from _pytest.mark import MARK_GEN as mark
from _pytest.mark import param
from _pytest.nodes import Collector
from _pytest.nodes import File
from _pytest.nodes import Item
from _pytest.outcomes import exit
from _pytest.outcomes import fail
from _pytest.outcomes import importorskip
from _pytest.outcomes import skip
from _pytest.outcomes import xfail
from _pytest.python import Class
from _pytest.python import Function
from _pytest.python import Instance
from _pytest.python import Module
from _pytest.python import Package
from _pytest.python_api import approx
from _pytest.python_api import raises
from _pytest.recwarn import deprecated_call
from _pytest.recwarn import warns
from _pytest.warning_types import PytestAssertRewriteWarning
from _pytest.warning_types import PytestCacheWarning
from _pytest.warning_types import PytestCollectionWarning
from _pytest.warning_types import PytestConfigWarning
from _pytest.warning_types import PytestDeprecationWarning
from _pytest.warning_types import PytestExperimentalApiWarning
from _pytest.warning_types import PytestUnhandledCoroutineWarning
from _pytest.warning_types import PytestUnknownMarkWarning
from _pytest.warning_types import PytestWarning


set_trace = __pytestPDB.set_trace

__all__ = [
    "__version__",
    "_fillfuncargs",
    "approx",
    "Class",
    "cmdline",
    "Collector",
    "deprecated_call",
    "exit",
    "ExitCode",
    "fail",
    "File",
    "fixture",
    "freeze_includes",
    "Function",
    "hookimpl",
    "hookspec",
    "importorskip",
    "Instance",
    "Item",
    "main",
    "mark",
    "Module",
    "Package",
    "param",
    "PytestAssertRewriteWarning",
    "PytestCacheWarning",
    "PytestCollectionWarning",
    "PytestConfigWarning",
    "PytestDeprecationWarning",
    "PytestExperimentalApiWarning",
    "PytestUnhandledCoroutineWarning",
    "PytestUnknownMarkWarning",
    "PytestWarning",
    "raises",
    "register_assert_rewrite",
    "Session",
    "set_trace",
    "skip",
    "UsageError",
    "warns",
    "xfail",
    "yield_fixture",
]

if __name__ == "__main__":
    # if run as a script or by 'python -m pytest'
    # we trigger the below "else" condition by the following import
    import pytest

    raise SystemExit(pytest.main())
else:

    from _pytest.compat import _setup_collect_fakemodule

    _setup_collect_fakemodule()
```
### 2 - src/_pytest/reports.py:

Start line: 129, End line: 149

```python
class BaseReport:

    @property
    def head_line(self):
        """
        **Experimental**

        Returns the head line shown with longrepr output for this report, more commonly during
        traceback representation during failures::

            ________ Test.foo ________


        In the example above, the head_line is "Test.foo".

        .. note::

            This function is considered **experimental**, so beware that it is subject to changes
            even in patch releases.
        """
        if self.location is not None:
            fspath, lineno, domain = self.location
            return domain
```
### 3 - src/_pytest/doctest.py:

Start line: 219, End line: 275

```python
class DoctestItem(pytest.Item):

    def repr_failure(self, excinfo):
        import doctest

        failures = None
        if excinfo.errisinstance((doctest.DocTestFailure, doctest.UnexpectedException)):
            failures = [excinfo.value]
        elif excinfo.errisinstance(MultipleDoctestFailures):
            failures = excinfo.value.failures

        if failures is not None:
            reprlocation_lines = []
            for failure in failures:
                example = failure.example
                test = failure.test
                filename = test.filename
                if test.lineno is None:
                    lineno = None
                else:
                    lineno = test.lineno + example.lineno + 1
                message = type(failure).__name__
                reprlocation = ReprFileLocation(filename, lineno, message)
                checker = _get_checker()
                report_choice = _get_report_choice(
                    self.config.getoption("doctestreport")
                )
                if lineno is not None:
                    lines = failure.test.docstring.splitlines(False)
                    # add line numbers to the left of the error message
                    lines = [
                        "%03d %s" % (i + test.lineno + 1, x)
                        for (i, x) in enumerate(lines)
                    ]
                    # trim docstring error lines to 10
                    lines = lines[max(example.lineno - 9, 0) : example.lineno + 1]
                else:
                    lines = [
                        "EXAMPLE LOCATION UNKNOWN, not showing all tests of that example"
                    ]
                    indent = ">>>"
                    for line in example.source.splitlines():
                        lines.append("??? {} {}".format(indent, line))
                        indent = "..."
                if isinstance(failure, doctest.DocTestFailure):
                    lines += checker.output_difference(
                        example, failure.got, report_choice
                    ).split("\n")
                else:
                    inner_excinfo = ExceptionInfo(failure.exc_info)
                    lines += ["UNEXPECTED EXCEPTION: %s" % repr(inner_excinfo.value)]
                    lines += traceback.format_exception(*failure.exc_info)
                reprlocation_lines.append((reprlocation, lines))
            return ReprFailDoctest(reprlocation_lines)
        else:
            return super().repr_failure(excinfo)

    def reportinfo(self):
        return self.fspath, self.dtest.lineno, "[doctest] %s" % self.name
```
### 4 - src/_pytest/terminal.py:

Start line: 905, End line: 920

```python
class TerminalReporter:

    def short_test_summary(self):
        # ... other code

        def show_skipped(lines):
            skipped = self.stats.get("skipped", [])
            fskips = _folded_skips(skipped) if skipped else []
            if not fskips:
                return
            verbose_word = skipped[0]._get_verbose_word(self.config)
            for num, fspath, lineno, reason in fskips:
                if reason.startswith("Skipped: "):
                    reason = reason[9:]
                if lineno is not None:
                    lines.append(
                        "%s [%d] %s:%d: %s"
                        % (verbose_word, num, fspath, lineno + 1, reason)
                    )
                else:
                    lines.append("%s [%d] %s: %s" % (verbose_word, num, fspath, reason))
        # ... other code
```
### 5 - src/_pytest/terminal.py:

Start line: 873, End line: 903

```python
class TerminalReporter:

    def short_test_summary(self):
        if not self.reportchars:
            return

        def show_simple(stat, lines):
            failed = self.stats.get(stat, [])
            if not failed:
                return
            termwidth = self.writer.fullwidth
            config = self.config
            for rep in failed:
                line = _get_line_with_reprcrash_message(config, rep, termwidth)
                lines.append(line)

        def show_xfailed(lines):
            xfailed = self.stats.get("xfailed", [])
            for rep in xfailed:
                verbose_word = rep._get_verbose_word(self.config)
                pos = _get_pos(self.config, rep)
                lines.append("{} {}".format(verbose_word, pos))
                reason = rep.wasxfail
                if reason:
                    lines.append("  " + str(reason))

        def show_xpassed(lines):
            xpassed = self.stats.get("xpassed", [])
            for rep in xpassed:
                verbose_word = rep._get_verbose_word(self.config)
                pos = _get_pos(self.config, rep)
                reason = rep.wasxfail
                lines.append("{} {} {}".format(verbose_word, pos, reason))
        # ... other code
```
### 6 - doc/en/example/xfail_demo.py:

Start line: 1, End line: 39

```python
import pytest

xfail = pytest.mark.xfail


@xfail
def test_hello():
    assert 0


@xfail(run=False)
def test_hello2():
    assert 0


@xfail("hasattr(os, 'sep')")
def test_hello3():
    assert 0


@xfail(reason="bug 110")
def test_hello4():
    assert 0


@xfail('pytest.__version__[0] != "17"')
def test_hello5():
    assert 0


def test_hello6():
    pytest.xfail("reason")


@xfail(raises=IndexError)
def test_hello7():
    x = []
    x[1] = 1
```
### 7 - src/_pytest/reports.py:

Start line: 1, End line: 31

```python
from io import StringIO
from pprint import pprint
from typing import Optional
from typing import Union

import py

from _pytest._code.code import ExceptionChainRepr
from _pytest._code.code import ExceptionInfo
from _pytest._code.code import ReprEntry
from _pytest._code.code import ReprEntryNative
from _pytest._code.code import ReprExceptionInfo
from _pytest._code.code import ReprFileLocation
from _pytest._code.code import ReprFuncArgs
from _pytest._code.code import ReprLocals
from _pytest._code.code import ReprTraceback
from _pytest._code.code import TerminalRepr
from _pytest.outcomes import skip
from _pytest.pathlib import Path


def getslaveinfoline(node):
    try:
        return node._slaveinfocache
    except AttributeError:
        d = node.slaveinfo
        ver = "%s.%s.%s" % d["version_info"][:3]
        node._slaveinfocache = s = "[{}] {} -- Python {} {}".format(
            d["id"], d["sysplatform"], ver, d["executable"]
        )
        return s
```
### 8 - src/_pytest/terminal.py:

Start line: 386, End line: 435

```python
class TerminalReporter:

    def pytest_runtest_logreport(self, report):
        self._tests_ran = True
        rep = report
        res = self.config.hook.pytest_report_teststatus(report=rep, config=self.config)
        category, letter, word = res
        if isinstance(word, tuple):
            word, markup = word
        else:
            markup = None
        self.stats.setdefault(category, []).append(rep)
        if not letter and not word:
            # probably passed setup/teardown
            return
        running_xdist = hasattr(rep, "node")
        if markup is None:
            was_xfail = hasattr(report, "wasxfail")
            if rep.passed and not was_xfail:
                markup = {"green": True}
            elif rep.passed and was_xfail:
                markup = {"yellow": True}
            elif rep.failed:
                markup = {"red": True}
            elif rep.skipped:
                markup = {"yellow": True}
            else:
                markup = {}
        if self.verbosity <= 0:
            if not running_xdist and self.showfspath:
                self.write_fspath_result(rep.nodeid, letter, **markup)
            else:
                self._tw.write(letter, **markup)
        else:
            self._progress_nodeids_reported.add(rep.nodeid)
            line = self._locationline(rep.nodeid, *rep.location)
            if not running_xdist:
                self.write_ensure_prefix(line, word, **markup)
                if self._show_progress_info:
                    self._write_progress_information_filling_space()
            else:
                self.ensure_newline()
                self._tw.write("[%s]" % rep.node.gateway.id)
                if self._show_progress_info:
                    self._tw.write(
                        self._get_progress_information_message() + " ", cyan=True
                    )
                else:
                    self._tw.write(" ")
                self._tw.write(word, **markup)
                self._tw.write(" " + line)
                self.currentfspath = -2
```
### 9 - src/_pytest/terminal.py:

Start line: 722, End line: 745

```python
class TerminalReporter:

    def _getfailureheadline(self, rep):
        head_line = rep.head_line
        if head_line:
            return head_line
        return "test session"  # XXX?

    def _getcrashline(self, rep):
        try:
            return str(rep.longrepr.reprcrash)
        except AttributeError:
            try:
                return str(rep.longrepr)[:50]
            except AttributeError:
                return ""

    #
    # summaries for sessionfinish
    #
    def getreports(self, name):
        values = []
        for x in self.stats.get(name, []):
            if not hasattr(x, "_pdbshown"):
                values.append(x)
        return values
```
### 10 - testing/python/metafunc.py:

Start line: 1308, End line: 1346

```python
class TestMetafuncFunctional:

    def test_generate_same_function_names_issue403(self, testdir):
        testdir.makepyfile(
            """
            import pytest

            def make_tests():
                @pytest.mark.parametrize("x", range(2))
                def test_foo(x):
                    pass
                return test_foo

            test_x = make_tests()
            test_y = make_tests()
        """
        )
        reprec = testdir.runpytest()
        reprec.assert_outcomes(passed=4)

    @pytest.mark.parametrize("attr", ["parametrise", "parameterize", "parameterise"])
    def test_parametrize_misspelling(self, testdir, attr):
        """#463"""
        testdir.makepyfile(
            """
            import pytest

            @pytest.mark.{}("x", range(2))
            def test_foo(x):
                pass
        """.format(
                attr
            )
        )
        result = testdir.runpytest("--collectonly")
        result.stdout.fnmatch_lines(
            [
                "test_foo has '{}' mark, spelling should be 'parametrize'".format(attr),
                "*1 error in*",
            ]
        )
```
### 33 - src/_pytest/python.py:

Start line: 1268, End line: 1337

```python
def _showfixtures_main(config, session):
    import _pytest.config

    session.perform_collect()
    curdir = py.path.local()
    tw = _pytest.config.create_terminal_writer(config)
    verbose = config.getvalue("verbose")

    fm = session._fixturemanager

    available = []
    seen = set()

    for argname, fixturedefs in fm._arg2fixturedefs.items():
        assert fixturedefs is not None
        if not fixturedefs:
            continue
        for fixturedef in fixturedefs:
            loc = getlocation(fixturedef.func, curdir)
            if (fixturedef.argname, loc) in seen:
                continue
            seen.add((fixturedef.argname, loc))
            available.append(
                (
                    len(fixturedef.baseid),
                    fixturedef.func.__module__,
                    curdir.bestrelpath(loc),
                    fixturedef.argname,
                    fixturedef,
                )
            )

    available.sort()
    currentmodule = None
    for baseid, module, bestrel, argname, fixturedef in available:
        if currentmodule != module:
            if not module.startswith("_pytest."):
                tw.line()
                tw.sep("-", "fixtures defined from {}".format(module))
                currentmodule = module
        if verbose <= 0 and argname[0] == "_":
            continue
        tw.write(argname, green=True)
        if fixturedef.scope != "function":
            tw.write(" [%s scope]" % fixturedef.scope, cyan=True)
        if verbose > 0:
            tw.write(" -- %s" % bestrel, yellow=True)
        tw.write("\n")
        loc = getlocation(fixturedef.func, curdir)
        doc = fixturedef.func.__doc__ or ""
        if doc:
            write_docstring(tw, doc)
        else:
            tw.line("    {}: no docstring available".format(loc), red=True)
        tw.line()


def write_docstring(tw, doc, indent="    "):
    doc = doc.rstrip()
    if "\n" in doc:
        firstline, rest = doc.split("\n", 1)
    else:
        firstline, rest = doc, ""

    if firstline.strip():
        tw.line(indent + firstline.strip())

    if rest:
        for line in dedent(rest).split("\n"):
            tw.write(indent + line + "\n")
```
### 48 - src/_pytest/python.py:

Start line: 1210, End line: 1236

```python
def _show_fixtures_per_test(config, session):
    import _pytest.config

    session.perform_collect()
    curdir = py.path.local()
    tw = _pytest.config.create_terminal_writer(config)
    verbose = config.getvalue("verbose")

    def get_best_relpath(func):
        loc = getlocation(func, curdir)
        return curdir.bestrelpath(loc)

    def write_fixture(fixture_def):
        argname = fixture_def.argname
        if verbose <= 0 and argname.startswith("_"):
            return
        if verbose > 0:
            bestrel = get_best_relpath(fixture_def.func)
            funcargspec = "{} -- {}".format(argname, bestrel)
        else:
            funcargspec = argname
        tw.line(funcargspec, green=True)
        fixture_doc = fixture_def.func.__doc__
        if fixture_doc:
            write_docstring(tw, fixture_doc)
        else:
            tw.line("    no docstring available", red=True)
    # ... other code
```
### 102 - src/_pytest/python.py:

Start line: 291, End line: 305

```python
class PyobjMixin(PyobjContext):

    def reportinfo(self):
        # XXX caching?
        obj = self.obj
        compat_co_firstlineno = getattr(obj, "compat_co_firstlineno", None)
        if isinstance(compat_co_firstlineno, int):
            # nose compatibility
            fspath = sys.modules[obj.__module__].__file__
            if fspath.endswith(".pyc"):
                fspath = fspath[:-1]
            lineno = compat_co_firstlineno
        else:
            fspath, lineno = getfslineno(obj)
        modpath = self.getmodpath()
        assert isinstance(lineno, int)
        return fspath, lineno, modpath
```
### 111 - src/_pytest/python.py:

Start line: 111, End line: 129

```python
def pytest_cmdline_main(config):
    if config.option.showfixtures:
        showfixtures(config)
        return 0
    if config.option.show_fixtures_per_test:
        show_fixtures_per_test(config)
        return 0


def pytest_generate_tests(metafunc):
    # those alternative spellings are common - raise a specific error to alert
    # the user
    alt_spellings = ["parameterize", "parametrise", "parameterise"]
    for mark_name in alt_spellings:
        if metafunc.definition.get_closest_marker(mark_name):
            msg = "{0} has '{1}' mark, spelling should be 'parametrize'"
            fail(msg.format(metafunc.function.__name__, mark_name), pytrace=False)
    for marker in metafunc.definition.iter_markers(name="parametrize"):
        metafunc.parametrize(*marker.args, **marker.kwargs)
```
### 186 - src/_pytest/python.py:

Start line: 497, End line: 545

```python
class Module(nodes.File, PyCollector):

    def _importtestmodule(self):
        # we assume we are only called once per module
        importmode = self.config.getoption("--import-mode")
        try:
            mod = self.fspath.pyimport(ensuresyspath=importmode)
        except SyntaxError:
            raise self.CollectError(
                _pytest._code.ExceptionInfo.from_current().getrepr(style="short")
            )
        except self.fspath.ImportMismatchError:
            e = sys.exc_info()[1]
            raise self.CollectError(
                "import file mismatch:\n"
                "imported module %r has this __file__ attribute:\n"
                "  %s\n"
                "which is not the same as the test file we want to collect:\n"
                "  %s\n"
                "HINT: remove __pycache__ / .pyc files and/or use a "
                "unique basename for your test file modules" % e.args
            )
        except ImportError:
            from _pytest._code.code import ExceptionInfo

            exc_info = ExceptionInfo.from_current()
            if self.config.getoption("verbose") < 2:
                exc_info.traceback = exc_info.traceback.filter(filter_traceback)
            exc_repr = (
                exc_info.getrepr(style="short")
                if exc_info.traceback
                else exc_info.exconly()
            )
            formatted_tb = str(exc_repr)
            raise self.CollectError(
                "ImportError while importing test module '{fspath}'.\n"
                "Hint: make sure your test modules/packages have valid Python names.\n"
                "Traceback:\n"
                "{traceback}".format(fspath=self.fspath, traceback=formatted_tb)
            )
        except _pytest.runner.Skipped as e:
            if e.allow_module_level:
                raise
            raise self.CollectError(
                "Using pytest.skip outside of a test is not allowed. "
                "To decorate a test function, use the @pytest.mark.skip "
                "or @pytest.mark.skipif decorators instead, and to skip a "
                "module use `pytestmark = pytest.mark.{skip,skipif}."
            )
        self.config.pluginmanager.consider_module(mod)
        return mod
```
### 191 - src/_pytest/python.py:

Start line: 1, End line: 53

```python
""" Python test discovery, setup and run of test functions. """
import enum
import fnmatch
import inspect
import os
import sys
import warnings
from collections import Counter
from collections.abc import Sequence
from functools import partial
from textwrap import dedent

import py

import _pytest
from _pytest import fixtures
from _pytest import nodes
from _pytest._code import filter_traceback
from _pytest.compat import ascii_escaped
from _pytest.compat import get_default_arg_names
from _pytest.compat import get_real_func
from _pytest.compat import getfslineno
from _pytest.compat import getimfunc
from _pytest.compat import getlocation
from _pytest.compat import is_generator
from _pytest.compat import iscoroutinefunction
from _pytest.compat import NOTSET
from _pytest.compat import REGEX_TYPE
from _pytest.compat import safe_getattr
from _pytest.compat import safe_isclass
from _pytest.compat import STRING_TYPES
from _pytest.config import hookimpl
from _pytest.main import FSHookProxy
from _pytest.mark import MARK_GEN
from _pytest.mark.structures import get_unpacked_marks
from _pytest.mark.structures import normalize_mark_list
from _pytest.outcomes import fail
from _pytest.outcomes import skip
from _pytest.pathlib import parts
from _pytest.warning_types import PytestCollectionWarning
from _pytest.warning_types import PytestUnhandledCoroutineWarning


def pyobj_property(name):
    def get(self):
        node = self.getparent(getattr(__import__("pytest"), name))
        if node is not None:
            return node.obj

    doc = "python {} object this node was collected from (can be None).".format(
        name.lower()
    )
    return property(get, None, None, doc)
```
### 205 - src/_pytest/python.py:

Start line: 1238, End line: 1265

```python
def _show_fixtures_per_test(config, session):
    # ... other code

    def write_item(item):
        try:
            info = item._fixtureinfo
        except AttributeError:
            # doctests items have no _fixtureinfo attribute
            return
        if not info.name2fixturedefs:
            # this test item does not use any fixtures
            return
        tw.line()
        tw.sep("-", "fixtures used by {}".format(item.name))
        tw.sep("-", "({})".format(get_best_relpath(item.function)))
        # dict key not used in loop but needed for sorting
        for _, fixturedefs in sorted(info.name2fixturedefs.items()):
            assert fixturedefs is not None
            if not fixturedefs:
                continue
            # last item is expected to be the one used by the test item
            write_fixture(fixturedefs[-1])

    for session_item in session.items:
        write_item(session_item)


def showfixtures(config):
    from _pytest.main import wrap_session

    return wrap_session(config, _showfixtures_main)
```
