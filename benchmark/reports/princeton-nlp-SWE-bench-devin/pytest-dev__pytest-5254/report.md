# pytest-dev__pytest-5254

| **pytest-dev/pytest** | `654d8da9f7ffd7a88e02ae2081ffcb2ca2e765b3` |
| ---- | ---- |
| **No of patches** | 2 |
| **All found context length** | 30114 |
| **Any found context length** | 30114 |
| **Avg pos** | 92.5 |
| **Min pos** | 29 |
| **Max pos** | 51 |
| **Top file pos** | 5 |
| **Missing snippets** | 5 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/src/_pytest/fixtures.py b/src/_pytest/fixtures.py
--- a/src/_pytest/fixtures.py
+++ b/src/_pytest/fixtures.py
@@ -1129,18 +1129,40 @@ def __init__(self, session):
         self._nodeid_and_autousenames = [("", self.config.getini("usefixtures"))]
         session.config.pluginmanager.register(self, "funcmanage")
 
+    def _get_direct_parametrize_args(self, node):
+        """This function returns all the direct parametrization
+        arguments of a node, so we don't mistake them for fixtures
+
+        Check https://github.com/pytest-dev/pytest/issues/5036
+
+        This things are done later as well when dealing with parametrization
+        so this could be improved
+        """
+        from _pytest.mark import ParameterSet
+
+        parametrize_argnames = []
+        for marker in node.iter_markers(name="parametrize"):
+            if not marker.kwargs.get("indirect", False):
+                p_argnames, _ = ParameterSet._parse_parametrize_args(
+                    *marker.args, **marker.kwargs
+                )
+                parametrize_argnames.extend(p_argnames)
+
+        return parametrize_argnames
+
     def getfixtureinfo(self, node, func, cls, funcargs=True):
         if funcargs and not getattr(node, "nofuncargs", False):
             argnames = getfuncargnames(func, cls=cls)
         else:
             argnames = ()
+
         usefixtures = itertools.chain.from_iterable(
             mark.args for mark in node.iter_markers(name="usefixtures")
         )
         initialnames = tuple(usefixtures) + argnames
         fm = node.session._fixturemanager
         initialnames, names_closure, arg2fixturedefs = fm.getfixtureclosure(
-            initialnames, node
+            initialnames, node, ignore_args=self._get_direct_parametrize_args(node)
         )
         return FuncFixtureInfo(argnames, initialnames, names_closure, arg2fixturedefs)
 
@@ -1174,7 +1196,7 @@ def _getautousenames(self, nodeid):
                 autousenames.extend(basenames)
         return autousenames
 
-    def getfixtureclosure(self, fixturenames, parentnode):
+    def getfixtureclosure(self, fixturenames, parentnode, ignore_args=()):
         # collect the closure of all fixtures , starting with the given
         # fixturenames as the initial set.  As we have to visit all
         # factory definitions anyway, we also return an arg2fixturedefs
@@ -1202,6 +1224,8 @@ def merge(otherlist):
         while lastlen != len(fixturenames_closure):
             lastlen = len(fixturenames_closure)
             for argname in fixturenames_closure:
+                if argname in ignore_args:
+                    continue
                 if argname in arg2fixturedefs:
                     continue
                 fixturedefs = self.getfixturedefs(argname, parentid)
diff --git a/src/_pytest/mark/structures.py b/src/_pytest/mark/structures.py
--- a/src/_pytest/mark/structures.py
+++ b/src/_pytest/mark/structures.py
@@ -103,8 +103,11 @@ def extract_from(cls, parameterset, force_tuple=False):
         else:
             return cls(parameterset, marks=[], id=None)
 
-    @classmethod
-    def _for_parametrize(cls, argnames, argvalues, func, config, function_definition):
+    @staticmethod
+    def _parse_parametrize_args(argnames, argvalues, **_):
+        """It receives an ignored _ (kwargs) argument so this function can
+        take also calls from parametrize ignoring scope, indirect, and other
+        arguments..."""
         if not isinstance(argnames, (tuple, list)):
             argnames = [x.strip() for x in argnames.split(",") if x.strip()]
             force_tuple = len(argnames) == 1
@@ -113,6 +116,11 @@ def _for_parametrize(cls, argnames, argvalues, func, config, function_definition
         parameters = [
             ParameterSet.extract_from(x, force_tuple=force_tuple) for x in argvalues
         ]
+        return argnames, parameters
+
+    @classmethod
+    def _for_parametrize(cls, argnames, argvalues, func, config, function_definition):
+        argnames, parameters = cls._parse_parametrize_args(argnames, argvalues)
         del argvalues
 
         if parameters:

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| src/_pytest/fixtures.py | 1132 | 1132 | 51 | 5 | 35103
| src/_pytest/fixtures.py | 1177 | 1177 | 38 | 5 | 32648
| src/_pytest/fixtures.py | 1205 | 1205 | 38 | 5 | 32648
| src/_pytest/mark/structures.py | 106 | 107 | 29 | 8 | 30114
| src/_pytest/mark/structures.py | 116 | 116 | 29 | 8 | 30114


## Problem Statement

```
`pytest.mark.parametrize` does not correctly hide fixtures of the same name (it misses its dependencies)
From https://github.com/smarie/python-pytest-cases/issues/36

This works:

\`\`\`python
@pytest.fixture(params=['a', 'b'])
def arg(request):
    return request.param

@pytest.mark.parametrize("arg", [1])
def test_reference(arg, request):
    assert '[1]' in request.node.nodeid
\`\`\`

the `arg` parameter in the test correctly hides the `arg` fixture so the unique pytest node has id `[1]` (instead of there being two nodes because of the fixture).

However if the fixture that is hidden by the parameter depends on another fixture, that other fixture is mistakenly kept in the fixtures closure, even if it is not needed anymore. Therefore the test fails:

\`\`\`python
@pytest.fixture(params=['a', 'b'])
def argroot(request):
    return request.param

@pytest.fixture
def arg(argroot):
    return argroot

@pytest.mark.parametrize("arg", [1])
def test_reference(arg, request):
    assert '[1]' in request.node.nodeid
\`\`\`







```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 testing/python/fixtures.py | 1013 | 2016| 6226 | 6226 | 24799 | 
| 2 | 2 testing/python/setup_only.py | 157 | 183| 167 | 6393 | 26340 | 
| 3 | 3 testing/python/metafunc.py | 1138 | 1167| 200 | 6593 | 39462 | 
| 4 | 4 testing/python/collect.py | 449 | 478| 193 | 6786 | 49157 | 
| 5 | 4 testing/python/collect.py | 430 | 447| 124 | 6910 | 49157 | 
| 6 | 4 testing/python/fixtures.py | 47 | 1011| 6204 | 13114 | 49157 | 
| 7 | 4 testing/python/collect.py | 401 | 428| 194 | 13308 | 49157 | 
| 8 | 4 testing/python/fixtures.py | 3110 | 3953| 5054 | 18362 | 49157 | 
| 9 | 4 testing/python/fixtures.py | 2182 | 3108| 6112 | 24474 | 49157 | 
| 10 | 4 testing/python/fixtures.py | 2018 | 2180| 969 | 25443 | 49157 | 
| 11 | 4 testing/python/setup_only.py | 124 | 154| 183 | 25626 | 49157 | 
| 12 | 4 testing/python/setup_only.py | 63 | 94| 190 | 25816 | 49157 | 
| 13 | 4 testing/python/fixtures.py | 1 | 44| 259 | 26075 | 49157 | 
| 14 | **5 src/_pytest/fixtures.py** | 1223 | 1255| 247 | 26322 | 59965 | 
| 15 | 5 testing/python/metafunc.py | 1535 | 1554| 169 | 26491 | 59965 | 
| 16 | **5 src/_pytest/fixtures.py** | 1 | 111| 705 | 27196 | 59965 | 
| 17 | 5 testing/python/setup_only.py | 186 | 201| 119 | 27315 | 59965 | 
| 18 | 5 testing/python/metafunc.py | 518 | 557| 330 | 27645 | 59965 | 
| 19 | 6 src/_pytest/capture.py | 251 | 261| 108 | 27753 | 66132 | 
| 20 | **6 src/_pytest/fixtures.py** | 591 | 609| 183 | 27936 | 66132 | 
| 21 | 6 testing/python/metafunc.py | 1659 | 1698| 267 | 28203 | 66132 | 
| 22 | 6 testing/python/setup_only.py | 34 | 60| 167 | 28370 | 66132 | 
| 23 | 7 src/_pytest/python.py | 1221 | 1240| 189 | 28559 | 77964 | 
| 24 | **7 src/_pytest/fixtures.py** | 488 | 515| 208 | 28767 | 77964 | 
| 25 | 7 testing/python/metafunc.py | 1231 | 1263| 200 | 28967 | 77964 | 
| 26 | **7 src/_pytest/fixtures.py** | 699 | 750| 475 | 29442 | 77964 | 
| 27 | 7 testing/python/setup_only.py | 1 | 31| 188 | 29630 | 77964 | 
| 28 | 7 testing/python/metafunc.py | 1589 | 1603| 131 | 29761 | 77964 | 
| **-> 29 <-** | **8 src/_pytest/mark/structures.py** | 106 | 145| 353 | 30114 | 80800 | 
| 30 | 8 testing/python/metafunc.py | 1025 | 1047| 174 | 30288 | 80800 | 
| 31 | 8 testing/python/collect.py | 285 | 350| 472 | 30760 | 80800 | 
| 32 | 8 testing/python/collect.py | 480 | 499| 157 | 30917 | 80800 | 
| 33 | 8 testing/python/metafunc.py | 1744 | 1768| 183 | 31100 | 80800 | 
| 34 | 8 testing/python/collect.py | 565 | 664| 663 | 31763 | 80800 | 
| 35 | **8 src/_pytest/fixtures.py** | 374 | 388| 198 | 31961 | 80800 | 
| 36 | 9 testing/python/integration.py | 421 | 448| 161 | 32122 | 83581 | 
| 37 | 9 testing/python/metafunc.py | 1621 | 1635| 139 | 32261 | 83581 | 
| **-> 38 <-** | **9 src/_pytest/fixtures.py** | 1177 | 1221| 387 | 32648 | 83581 | 
| 39 | 9 testing/python/setup_only.py | 97 | 121| 150 | 32798 | 83581 | 
| 40 | 9 testing/python/metafunc.py | 636 | 664| 231 | 33029 | 83581 | 
| 41 | **9 src/_pytest/fixtures.py** | 873 | 903| 264 | 33293 | 83581 | 
| 42 | **9 src/_pytest/fixtures.py** | 611 | 638| 274 | 33567 | 83581 | 
| 43 | **9 src/_pytest/mark/structures.py** | 38 | 61| 202 | 33769 | 83581 | 
| 44 | 9 testing/python/metafunc.py | 1049 | 1070| 158 | 33927 | 83581 | 
| 45 | 9 testing/python/metafunc.py | 1637 | 1657| 202 | 34129 | 83581 | 
| 46 | **9 src/_pytest/fixtures.py** | 973 | 996| 181 | 34310 | 83581 | 
| 47 | 9 testing/python/metafunc.py | 1516 | 1533| 159 | 34469 | 83581 | 
| 48 | 9 src/_pytest/python.py | 1243 | 1269| 208 | 34677 | 83581 | 
| 49 | 9 testing/python/metafunc.py | 1605 | 1619| 134 | 34811 | 83581 | 
| 50 | 9 testing/python/integration.py | 397 | 419| 141 | 34952 | 83581 | 
| **-> 51 <-** | **9 src/_pytest/fixtures.py** | 1132 | 1145| 151 | 35103 | 83581 | 
| 52 | 9 testing/python/metafunc.py | 1556 | 1571| 136 | 35239 | 83581 | 
| 53 | 9 testing/python/metafunc.py | 1573 | 1587| 119 | 35358 | 83581 | 
| 54 | 9 testing/python/metafunc.py | 1289 | 1327| 244 | 35602 | 83581 | 
| 55 | 9 testing/python/metafunc.py | 1492 | 1514| 170 | 35772 | 83581 | 
| 56 | 10 testing/python/setup_plan.py | 1 | 21| 127 | 35899 | 83708 | 
| 57 | **10 src/_pytest/fixtures.py** | 114 | 169| 607 | 36506 | 83708 | 
| 58 | 11 testing/example_scripts/issue_519.py | 1 | 32| 358 | 36864 | 84182 | 
| 59 | 11 src/_pytest/python.py | 767 | 808| 336 | 37200 | 84182 | 
| 60 | 11 testing/python/metafunc.py | 666 | 684| 149 | 37349 | 84182 | 
| 61 | 11 src/_pytest/python.py | 477 | 502| 243 | 37592 | 84182 | 
| 62 | 11 testing/python/metafunc.py | 1187 | 1203| 140 | 37732 | 84182 | 
| 63 | 12 testing/python/raises.py | 186 | 211| 164 | 37896 | 86019 | 
| 64 | 12 testing/python/metafunc.py | 703 | 718| 131 | 38027 | 86019 | 
| 65 | 12 testing/python/metafunc.py | 1700 | 1723| 223 | 38250 | 86019 | 
| 66 | **12 src/_pytest/fixtures.py** | 1147 | 1175| 250 | 38500 | 86019 | 
| 67 | 12 testing/python/metafunc.py | 1725 | 1742| 128 | 38628 | 86019 | 
| 68 | **12 src/_pytest/fixtures.py** | 1257 | 1315| 469 | 39097 | 86019 | 
| 69 | 12 testing/python/metafunc.py | 977 | 992| 145 | 39242 | 86019 | 
| 70 | 12 testing/python/metafunc.py | 686 | 701| 131 | 39373 | 86019 | 
| 71 | 13 src/_pytest/unittest.py | 32 | 80| 397 | 39770 | 88074 | 
| 72 | 13 testing/python/collect.py | 546 | 563| 183 | 39953 | 88074 | 
| 73 | **13 src/_pytest/fixtures.py** | 277 | 300| 191 | 40144 | 88074 | 
| 74 | 14 src/pytest.py | 1 | 108| 731 | 40875 | 88805 | 
| 75 | **14 src/_pytest/mark/structures.py** | 64 | 85| 175 | 41050 | 88805 | 
| 76 | **14 src/_pytest/fixtures.py** | 172 | 206| 327 | 41377 | 88805 | 
| 77 | 14 testing/example_scripts/issue_519.py | 35 | 53| 115 | 41492 | 88805 | 
| 78 | 15 doc/en/example/assertion/failure_demo.py | 1 | 41| 178 | 41670 | 90468 | 
| 79 | 15 src/_pytest/python.py | 1301 | 1370| 493 | 42163 | 90468 | 
| 80 | 15 testing/python/metafunc.py | 57 | 78| 212 | 42375 | 90468 | 
| 81 | **15 src/_pytest/fixtures.py** | 780 | 809| 245 | 42620 | 90468 | 
| 82 | **15 src/_pytest/fixtures.py** | 1317 | 1335| 154 | 42774 | 90468 | 
| 83 | 16 src/_pytest/setuponly.py | 27 | 51| 206 | 42980 | 91063 | 
| 84 | 16 testing/python/collect.py | 372 | 399| 193 | 43173 | 91063 | 
| 85 | 16 src/_pytest/python.py | 1271 | 1298| 217 | 43390 | 91063 | 
| 86 | 16 testing/python/setup_only.py | 204 | 227| 129 | 43519 | 91063 | 
| 87 | 16 testing/python/metafunc.py | 720 | 735| 127 | 43646 | 91063 | 
| 88 | 16 testing/python/metafunc.py | 801 | 828| 211 | 43857 | 91063 | 
| 89 | 17 src/_pytest/compat.py | 322 | 459| 804 | 44661 | 94221 | 
| 90 | 17 testing/python/metafunc.py | 1205 | 1229| 200 | 44861 | 94221 | 
| 91 | 17 testing/python/metafunc.py | 296 | 317| 231 | 45092 | 94221 | 
| 92 | 17 testing/python/integration.py | 307 | 344| 225 | 45317 | 94221 | 
| 93 | 17 testing/python/metafunc.py | 80 | 124| 450 | 45767 | 94221 | 
| 94 | 17 testing/python/metafunc.py | 1457 | 1489| 226 | 45993 | 94221 | 
| 95 | 18 src/_pytest/mark/__init__.py | 1 | 18| 139 | 46132 | 95401 | 
| 96 | **18 src/_pytest/fixtures.py** | 999 | 1050| 550 | 46682 | 95401 | 
| 97 | 18 testing/python/metafunc.py | 830 | 870| 314 | 46996 | 95401 | 
| 98 | 18 testing/python/metafunc.py | 775 | 790| 213 | 47209 | 95401 | 
| 99 | 18 src/_pytest/mark/__init__.py | 21 | 38| 172 | 47381 | 95401 | 
| 100 | **18 src/_pytest/fixtures.py** | 946 | 970| 180 | 47561 | 95401 | 
| 101 | 18 testing/python/metafunc.py | 1096 | 1114| 124 | 47685 | 95401 | 
| 102 | 18 src/_pytest/python.py | 743 | 765| 215 | 47900 | 95401 | 
| 103 | 18 testing/python/metafunc.py | 1380 | 1395| 113 | 48013 | 95401 | 
| 104 | 18 testing/python/metafunc.py | 607 | 634| 225 | 48238 | 95401 | 
| 105 | 18 testing/python/integration.py | 90 | 127| 198 | 48436 | 95401 | 
| 106 | 18 testing/python/metafunc.py | 754 | 773| 151 | 48587 | 95401 | 
| 107 | 18 src/_pytest/unittest.py | 83 | 106| 157 | 48744 | 95401 | 
| 108 | 18 testing/python/collect.py | 352 | 370| 153 | 48897 | 95401 | 
| 109 | **18 src/_pytest/fixtures.py** | 342 | 372| 272 | 49169 | 95401 | 
| 110 | 18 src/_pytest/python.py | 389 | 434| 394 | 49563 | 95401 | 
| 111 | **18 src/_pytest/fixtures.py** | 390 | 486| 779 | 50342 | 95401 | 
| 112 | 18 testing/python/setup_only.py | 230 | 249| 122 | 50464 | 95401 | 
| 113 | 18 testing/python/metafunc.py | 1010 | 1023| 144 | 50608 | 95401 | 
| 114 | 19 src/_pytest/main.py | 725 | 775| 430 | 51038 | 101184 | 
| 115 | 19 testing/python/metafunc.py | 1169 | 1185| 150 | 51188 | 101184 | 
| 116 | 19 testing/python/metafunc.py | 1072 | 1094| 157 | 51345 | 101184 | 
| 117 | 19 testing/python/metafunc.py | 792 | 799| 137 | 51482 | 101184 | 
| 118 | 20 src/_pytest/junitxml.py | 285 | 297| 132 | 51614 | 106205 | 
| 119 | 20 testing/python/collect.py | 530 | 544| 151 | 51765 | 106205 | 
| 120 | 20 testing/python/metafunc.py | 409 | 425| 118 | 51883 | 106205 | 
| 121 | 20 src/_pytest/setuponly.py | 54 | 90| 265 | 52148 | 106205 | 
| 122 | 20 testing/python/metafunc.py | 1116 | 1136| 163 | 52311 | 106205 | 
| 123 | **20 src/_pytest/fixtures.py** | 846 | 871| 194 | 52505 | 106205 | 
| 124 | 20 testing/python/metafunc.py | 1397 | 1422| 200 | 52705 | 106205 | 
| 125 | **20 src/_pytest/fixtures.py** | 641 | 673| 320 | 53025 | 106205 | 
| 126 | 20 testing/python/metafunc.py | 894 | 923| 235 | 53260 | 106205 | 
| 127 | **20 src/_pytest/mark/structures.py** | 87 | 104| 139 | 53399 | 106205 | 
| 128 | 20 src/_pytest/main.py | 385 | 418| 201 | 53600 | 106205 | 
| 129 | 20 testing/python/metafunc.py | 872 | 891| 130 | 53730 | 106205 | 
| 130 | 20 testing/python/metafunc.py | 493 | 516| 150 | 53880 | 106205 | 
| 131 | 20 testing/python/collect.py | 1 | 34| 239 | 54119 | 106205 | 
| 132 | 20 testing/python/metafunc.py | 574 | 605| 271 | 54390 | 106205 | 
| 133 | **20 src/_pytest/fixtures.py** | 303 | 339| 376 | 54766 | 106205 | 
| 134 | 20 testing/python/metafunc.py | 737 | 752| 118 | 54884 | 106205 | 
| 135 | 20 testing/python/integration.py | 38 | 69| 227 | 55111 | 106205 | 
| 136 | 21 src/_pytest/debugging.py | 267 | 276| 128 | 55239 | 108553 | 
| 137 | **21 src/_pytest/fixtures.py** | 1088 | 1130| 357 | 55596 | 108553 | 
| 138 | 21 testing/python/integration.py | 248 | 264| 122 | 55718 | 108553 | 
| 139 | 21 testing/python/metafunc.py | 126 | 181| 406 | 56124 | 108553 | 
| 140 | 21 src/_pytest/python.py | 141 | 158| 212 | 56336 | 108553 | 
| 141 | 21 testing/python/metafunc.py | 1 | 37| 212 | 56548 | 108553 | 
| 142 | 21 testing/python/integration.py | 129 | 165| 229 | 56777 | 108553 | 
| 143 | 21 testing/python/collect.py | 889 | 926| 322 | 57099 | 108553 | 
| 144 | 21 src/_pytest/capture.py | 390 | 405| 128 | 57227 | 108553 | 
| 145 | 21 testing/python/metafunc.py | 925 | 947| 177 | 57404 | 108553 | 
| 146 | **21 src/_pytest/fixtures.py** | 753 | 777| 214 | 57618 | 108553 | 
| 147 | 21 testing/python/collect.py | 1293 | 1310| 130 | 57748 | 108553 | 
| 148 | 21 testing/python/metafunc.py | 994 | 1008| 119 | 57867 | 108553 | 
| 149 | 21 testing/python/integration.py | 1 | 36| 248 | 58115 | 108553 | 
| 150 | 22 src/_pytest/doctest.py | 389 | 408| 159 | 58274 | 112724 | 
| 151 | 22 testing/python/collect.py | 1366 | 1399| 239 | 58513 | 112724 | 
| 152 | 22 testing/python/collect.py | 1154 | 1197| 332 | 58845 | 112724 | 
| 153 | 22 src/_pytest/main.py | 345 | 382| 333 | 59178 | 112724 | 
| 154 | 22 testing/python/collect.py | 1114 | 1151| 209 | 59387 | 112724 | 
| 155 | 22 testing/python/metafunc.py | 1357 | 1378| 187 | 59574 | 112724 | 
| 156 | 22 testing/python/setup_only.py | 252 | 271| 125 | 59699 | 112724 | 
| 157 | 22 testing/python/collect.py | 36 | 58| 188 | 59887 | 112724 | 
| 158 | 22 testing/python/collect.py | 854 | 886| 304 | 60191 | 112724 | 
| 159 | 23 doc/en/example/xfail_demo.py | 1 | 40| 151 | 60342 | 112876 | 
| 160 | 23 src/_pytest/python.py | 1 | 62| 430 | 60772 | 112876 | 
| 161 | 23 testing/python/collect.py | 202 | 255| 321 | 61093 | 112876 | 
| 162 | 23 testing/python/collect.py | 1313 | 1323| 110 | 61203 | 112876 | 
| 163 | 23 testing/python/metafunc.py | 949 | 975| 203 | 61406 | 112876 | 
| 164 | 23 src/_pytest/mark/__init__.py | 129 | 167| 242 | 61648 | 112876 | 
| 165 | 23 src/_pytest/unittest.py | 249 | 290| 286 | 61934 | 112876 | 
| 166 | 23 testing/python/metafunc.py | 559 | 572| 146 | 62080 | 112876 | 
| 167 | **23 src/_pytest/mark/structures.py** | 281 | 328| 428 | 62508 | 112876 | 
| 168 | 23 testing/python/metafunc.py | 370 | 388| 161 | 62669 | 112876 | 
| 169 | **23 src/_pytest/fixtures.py** | 1053 | 1085| 196 | 62865 | 112876 | 
| 170 | 23 testing/python/collect.py | 949 | 974| 171 | 63036 | 112876 | 
| 171 | 23 testing/python/collect.py | 796 | 818| 196 | 63232 | 112876 | 
| 172 | 23 testing/python/collect.py | 1326 | 1363| 320 | 63552 | 112876 | 
| 173 | 24 src/_pytest/pytester.py | 334 | 367| 157 | 63709 | 123448 | 
| 174 | **24 src/_pytest/fixtures.py** | 676 | 696| 132 | 63841 | 123448 | 
| 175 | 24 testing/python/metafunc.py | 427 | 458| 230 | 64071 | 123448 | 
| 176 | 24 testing/python/collect.py | 501 | 528| 208 | 64279 | 123448 | 
| 177 | **24 src/_pytest/fixtures.py** | 517 | 589| 651 | 64930 | 123448 | 
| 178 | 24 src/_pytest/main.py | 552 | 627| 688 | 65618 | 123448 | 
| 179 | 24 src/_pytest/doctest.py | 89 | 130| 319 | 65937 | 123448 | 
| 180 | 24 testing/python/integration.py | 224 | 246| 184 | 66121 | 123448 | 
| 181 | 24 src/_pytest/unittest.py | 109 | 137| 202 | 66323 | 123448 | 
| 182 | 24 testing/python/metafunc.py | 1330 | 1355| 160 | 66483 | 123448 | 
| 183 | 24 src/_pytest/python.py | 120 | 138| 176 | 66659 | 123448 | 
| 184 | 24 testing/python/integration.py | 167 | 189| 170 | 66829 | 123448 | 
| 185 | 24 src/_pytest/doctest.py | 435 | 450| 111 | 66940 | 123448 | 
| 186 | 24 testing/python/metafunc.py | 390 | 407| 126 | 67066 | 123448 | 
| 187 | 25 testing/example_scripts/fixtures/fill_fixtures/test_conftest_funcargs_only_available_in_subdir/sub1/conftest.py | 1 | 9| 0 | 67066 | 123482 | 
| 188 | 25 src/_pytest/capture.py | 163 | 248| 573 | 67639 | 123482 | 
| 189 | 25 testing/python/metafunc.py | 1424 | 1455| 245 | 67884 | 123482 | 
| 190 | 26 src/_pytest/skipping.py | 33 | 70| 363 | 68247 | 125041 | 
| 191 | 26 testing/python/metafunc.py | 183 | 197| 130 | 68377 | 125041 | 
| 192 | **26 src/_pytest/fixtures.py** | 926 | 943| 165 | 68542 | 125041 | 
| 193 | 26 testing/python/collect.py | 1200 | 1226| 182 | 68724 | 125041 | 
| 194 | 26 testing/python/metafunc.py | 39 | 55| 118 | 68842 | 125041 | 
| 195 | 26 src/_pytest/pytester.py | 1 | 38| 234 | 69076 | 125041 | 
| 196 | 26 doc/en/example/assertion/failure_demo.py | 165 | 203| 256 | 69332 | 125041 | 
| 197 | 26 src/_pytest/python.py | 360 | 387| 249 | 69581 | 125041 | 
| 198 | 27 src/_pytest/monkeypatch.py | 1 | 43| 279 | 69860 | 127611 | 
| 199 | 27 src/_pytest/python.py | 1208 | 1218| 122 | 69982 | 127611 | 
| 200 | 28 testing/example_scripts/fixtures/fill_fixtures/test_conftest_funcargs_only_available_in_subdir/sub2/conftest.py | 1 | 8| 0 | 69982 | 127644 | 
| 201 | 28 testing/python/collect.py | 1084 | 1111| 183 | 70165 | 127644 | 
| 202 | 28 testing/python/raises.py | 213 | 228| 135 | 70300 | 127644 | 
| 203 | 28 testing/python/metafunc.py | 319 | 352| 265 | 70565 | 127644 | 
| 204 | 29 src/_pytest/deprecated.py | 71 | 97| 237 | 70802 | 128621 | 
| 205 | 29 testing/python/metafunc.py | 199 | 208| 114 | 70916 | 128621 | 
| 206 | 29 src/_pytest/doctest.py | 182 | 219| 309 | 71225 | 128621 | 
| 207 | 29 src/_pytest/skipping.py | 73 | 92| 167 | 71392 | 128621 | 
| 208 | 29 src/_pytest/python.py | 850 | 900| 358 | 71750 | 128621 | 
| 209 | 30 bench/manyparam.py | 1 | 16| 0 | 71750 | 128670 | 
| 210 | 30 src/_pytest/main.py | 505 | 550| 379 | 72129 | 128670 | 
| 211 | 30 testing/python/metafunc.py | 460 | 491| 226 | 72355 | 128670 | 
| 212 | 30 testing/python/integration.py | 191 | 222| 272 | 72627 | 128670 | 
| 213 | 30 testing/python/collect.py | 738 | 761| 171 | 72798 | 128670 | 
| 214 | 30 testing/python/collect.py | 1259 | 1290| 203 | 73001 | 128670 | 
| 215 | 30 src/_pytest/skipping.py | 125 | 187| 583 | 73584 | 128670 | 
| 216 | 30 testing/python/collect.py | 258 | 283| 182 | 73766 | 128670 | 
| 217 | 30 testing/python/collect.py | 763 | 794| 205 | 73971 | 128670 | 
| 218 | **30 src/_pytest/mark/structures.py** | 331 | 368| 217 | 74188 | 128670 | 
| 219 | **30 src/_pytest/mark/structures.py** | 1 | 35| 198 | 74386 | 128670 | 
| 220 | 30 src/_pytest/python.py | 1175 | 1205| 307 | 74693 | 128670 | 
| 221 | 30 src/_pytest/python.py | 202 | 236| 334 | 75027 | 128670 | 
| 222 | **30 src/_pytest/fixtures.py** | 209 | 234| 261 | 75288 | 128670 | 
| 223 | 30 src/_pytest/python.py | 609 | 636| 211 | 75499 | 128670 | 
| 224 | 30 testing/python/metafunc.py | 1265 | 1287| 225 | 75724 | 128670 | 
| 225 | 30 src/_pytest/main.py | 296 | 342| 339 | 76063 | 128670 | 
| 226 | 30 src/_pytest/main.py | 421 | 476| 473 | 76536 | 128670 | 
| 227 | 30 src/_pytest/unittest.py | 200 | 218| 217 | 76753 | 128670 | 
| 228 | 30 src/_pytest/python.py | 1073 | 1100| 303 | 77056 | 128670 | 
| 229 | 30 src/_pytest/main.py | 652 | 675| 180 | 77236 | 128670 | 
| 230 | 30 src/_pytest/python.py | 902 | 922| 225 | 77461 | 128670 | 
| 231 | 30 src/_pytest/python.py | 504 | 552| 469 | 77930 | 128670 | 
| 232 | 31 doc/en/example/multipython.py | 1 | 23| 125 | 78055 | 129125 | 
| 233 | 31 testing/python/raises.py | 1 | 68| 455 | 78510 | 129125 | 
| 234 | **31 src/_pytest/mark/structures.py** | 148 | 168| 147 | 78657 | 129125 | 
| 235 | 31 src/_pytest/monkeypatch.py | 46 | 72| 184 | 78841 | 129125 | 
| 236 | 32 src/_pytest/setupplan.py | 1 | 33| 194 | 79035 | 129320 | 
| 237 | 32 testing/python/metafunc.py | 354 | 368| 145 | 79180 | 129320 | 
| 238 | 32 src/_pytest/main.py | 629 | 650| 177 | 79357 | 129320 | 
| 239 | 32 testing/python/raises.py | 230 | 258| 195 | 79552 | 129320 | 
| 240 | 33 testing/example_scripts/fixtures/fill_fixtures/test_extend_fixture_conftest_conftest/pkg/conftest.py | 1 | 8| 0 | 79552 | 129346 | 
| 241 | 34 src/_pytest/nodes.py | 347 | 376| 253 | 79805 | 132444 | 
| 242 | 35 src/_pytest/python_api.py | 673 | 716| 379 | 80184 | 138881 | 
| 243 | 35 testing/python/collect.py | 928 | 947| 179 | 80363 | 138881 | 
| 244 | 35 testing/python/collect.py | 721 | 736| 144 | 80507 | 138881 | 
| 245 | 35 src/_pytest/python.py | 1102 | 1129| 252 | 80759 | 138881 | 
| 246 | 36 testing/example_scripts/tmpdir/tmpdir_fixture.py | 1 | 9| 0 | 80759 | 138930 | 
| 247 | 36 testing/python/collect.py | 976 | 995| 171 | 80930 | 138930 | 
| 248 | 36 src/_pytest/python.py | 555 | 580| 266 | 81196 | 138930 | 
| 249 | 36 testing/python/collect.py | 997 | 1022| 205 | 81401 | 138930 | 
| 250 | 36 src/_pytest/python.py | 1132 | 1172| 314 | 81715 | 138930 | 


## Patch

```diff
diff --git a/src/_pytest/fixtures.py b/src/_pytest/fixtures.py
--- a/src/_pytest/fixtures.py
+++ b/src/_pytest/fixtures.py
@@ -1129,18 +1129,40 @@ def __init__(self, session):
         self._nodeid_and_autousenames = [("", self.config.getini("usefixtures"))]
         session.config.pluginmanager.register(self, "funcmanage")
 
+    def _get_direct_parametrize_args(self, node):
+        """This function returns all the direct parametrization
+        arguments of a node, so we don't mistake them for fixtures
+
+        Check https://github.com/pytest-dev/pytest/issues/5036
+
+        This things are done later as well when dealing with parametrization
+        so this could be improved
+        """
+        from _pytest.mark import ParameterSet
+
+        parametrize_argnames = []
+        for marker in node.iter_markers(name="parametrize"):
+            if not marker.kwargs.get("indirect", False):
+                p_argnames, _ = ParameterSet._parse_parametrize_args(
+                    *marker.args, **marker.kwargs
+                )
+                parametrize_argnames.extend(p_argnames)
+
+        return parametrize_argnames
+
     def getfixtureinfo(self, node, func, cls, funcargs=True):
         if funcargs and not getattr(node, "nofuncargs", False):
             argnames = getfuncargnames(func, cls=cls)
         else:
             argnames = ()
+
         usefixtures = itertools.chain.from_iterable(
             mark.args for mark in node.iter_markers(name="usefixtures")
         )
         initialnames = tuple(usefixtures) + argnames
         fm = node.session._fixturemanager
         initialnames, names_closure, arg2fixturedefs = fm.getfixtureclosure(
-            initialnames, node
+            initialnames, node, ignore_args=self._get_direct_parametrize_args(node)
         )
         return FuncFixtureInfo(argnames, initialnames, names_closure, arg2fixturedefs)
 
@@ -1174,7 +1196,7 @@ def _getautousenames(self, nodeid):
                 autousenames.extend(basenames)
         return autousenames
 
-    def getfixtureclosure(self, fixturenames, parentnode):
+    def getfixtureclosure(self, fixturenames, parentnode, ignore_args=()):
         # collect the closure of all fixtures , starting with the given
         # fixturenames as the initial set.  As we have to visit all
         # factory definitions anyway, we also return an arg2fixturedefs
@@ -1202,6 +1224,8 @@ def merge(otherlist):
         while lastlen != len(fixturenames_closure):
             lastlen = len(fixturenames_closure)
             for argname in fixturenames_closure:
+                if argname in ignore_args:
+                    continue
                 if argname in arg2fixturedefs:
                     continue
                 fixturedefs = self.getfixturedefs(argname, parentid)
diff --git a/src/_pytest/mark/structures.py b/src/_pytest/mark/structures.py
--- a/src/_pytest/mark/structures.py
+++ b/src/_pytest/mark/structures.py
@@ -103,8 +103,11 @@ def extract_from(cls, parameterset, force_tuple=False):
         else:
             return cls(parameterset, marks=[], id=None)
 
-    @classmethod
-    def _for_parametrize(cls, argnames, argvalues, func, config, function_definition):
+    @staticmethod
+    def _parse_parametrize_args(argnames, argvalues, **_):
+        """It receives an ignored _ (kwargs) argument so this function can
+        take also calls from parametrize ignoring scope, indirect, and other
+        arguments..."""
         if not isinstance(argnames, (tuple, list)):
             argnames = [x.strip() for x in argnames.split(",") if x.strip()]
             force_tuple = len(argnames) == 1
@@ -113,6 +116,11 @@ def _for_parametrize(cls, argnames, argvalues, func, config, function_definition
         parameters = [
             ParameterSet.extract_from(x, force_tuple=force_tuple) for x in argvalues
         ]
+        return argnames, parameters
+
+    @classmethod
+    def _for_parametrize(cls, argnames, argvalues, func, config, function_definition):
+        argnames, parameters = cls._parse_parametrize_args(argnames, argvalues)
         del argvalues
 
         if parameters:

```

## Test Patch

```diff
diff --git a/testing/python/fixtures.py b/testing/python/fixtures.py
--- a/testing/python/fixtures.py
+++ b/testing/python/fixtures.py
@@ -3950,3 +3950,46 @@ def fix():
 
     with pytest.raises(pytest.fail.Exception):
         assert fix() == 1
+
+
+def test_fixture_param_shadowing(testdir):
+    """Parametrized arguments would be shadowed if a fixture with the same name also exists (#5036)"""
+    testdir.makepyfile(
+        """
+        import pytest
+
+        @pytest.fixture(params=['a', 'b'])
+        def argroot(request):
+            return request.param
+
+        @pytest.fixture
+        def arg(argroot):
+            return argroot
+
+        # This should only be parametrized directly
+        @pytest.mark.parametrize("arg", [1])
+        def test_direct(arg):
+            assert arg == 1
+
+        # This should be parametrized based on the fixtures
+        def test_normal_fixture(arg):
+            assert isinstance(arg, str)
+
+        # Indirect should still work:
+
+        @pytest.fixture
+        def arg2(request):
+            return 2*request.param
+
+        @pytest.mark.parametrize("arg2", [1], indirect=True)
+        def test_indirect(arg2):
+            assert arg2 == 2
+    """
+    )
+    # Only one test should have run
+    result = testdir.runpytest("-v")
+    result.assert_outcomes(passed=4)
+    result.stdout.fnmatch_lines(["*::test_direct[[]1[]]*"])
+    result.stdout.fnmatch_lines(["*::test_normal_fixture[[]a[]]*"])
+    result.stdout.fnmatch_lines(["*::test_normal_fixture[[]b[]]*"])
+    result.stdout.fnmatch_lines(["*::test_indirect[[]1[]]*"])

```


## Code snippets

### 1 - testing/python/fixtures.py:

Start line: 1013, End line: 2016

```python
class TestFixtureUsages(object):

    def test_receives_funcargs_scope_mismatch(self, testdir):
        testdir.makepyfile(
            """
            import pytest
            @pytest.fixture(scope="function")
            def arg1():
                return 1

            @pytest.fixture(scope="module")
            def arg2(arg1):
                return arg1 + 1

            def test_add(arg2):
                assert arg2 == 2
        """
        )
        result = testdir.runpytest()
        result.stdout.fnmatch_lines(
            [
                "*ScopeMismatch*involved factories*",
                "test_receives_funcargs_scope_mismatch.py:6:  def arg2(arg1)",
                "test_receives_funcargs_scope_mismatch.py:2:  def arg1()",
                "*1 error*",
            ]
        )

    def test_receives_funcargs_scope_mismatch_issue660(self, testdir):
        testdir.makepyfile(
            """
            import pytest
            @pytest.fixture(scope="function")
            def arg1():
                return 1

            @pytest.fixture(scope="module")
            def arg2(arg1):
                return arg1 + 1

            def test_add(arg1, arg2):
                assert arg2 == 2
        """
        )
        result = testdir.runpytest()
        result.stdout.fnmatch_lines(
            ["*ScopeMismatch*involved factories*", "* def arg2*", "*1 error*"]
        )

    def test_invalid_scope(self, testdir):
        testdir.makepyfile(
            """
            import pytest
            @pytest.fixture(scope="functions")
            def badscope():
                pass

            def test_nothing(badscope):
                pass
        """
        )
        result = testdir.runpytest_inprocess()
        result.stdout.fnmatch_lines(
            "*Fixture 'badscope' from test_invalid_scope.py got an unexpected scope value 'functions'"
        )

    def test_funcarg_parametrized_and_used_twice(self, testdir):
        testdir.makepyfile(
            """
            import pytest
            values = []
            @pytest.fixture(params=[1,2])
            def arg1(request):
                values.append(1)
                return request.param

            @pytest.fixture()
            def arg2(arg1):
                return arg1 + 1

            def test_add(arg1, arg2):
                assert arg2 == arg1 + 1
                assert len(values) == arg1
        """
        )
        result = testdir.runpytest()
        result.stdout.fnmatch_lines(["*2 passed*"])

    def test_factory_uses_unknown_funcarg_as_dependency_error(self, testdir):
        testdir.makepyfile(
            """
            import pytest

            @pytest.fixture()
            def fail(missing):
                return

            @pytest.fixture()
            def call_fail(fail):
                return

            def test_missing(call_fail):
                pass
            """
        )
        result = testdir.runpytest()
        result.stdout.fnmatch_lines(
            """
            *pytest.fixture()*
            *def call_fail(fail)*
            *pytest.fixture()*
            *def fail*
            *fixture*'missing'*not found*
        """
        )

    def test_factory_setup_as_classes_fails(self, testdir):
        testdir.makepyfile(
            """
            import pytest
            class arg1(object):
                def __init__(self, request):
                    self.x = 1
            arg1 = pytest.fixture()(arg1)

        """
        )
        reprec = testdir.inline_run()
        values = reprec.getfailedcollections()
        assert len(values) == 1

    @pytest.mark.filterwarnings("ignore::pytest.PytestDeprecationWarning")
    def test_request_can_be_overridden(self, testdir):
        testdir.makepyfile(
            """
            import pytest
            @pytest.fixture()
            def request(request):
                request.a = 1
                return request
            def test_request(request):
                assert request.a == 1
        """
        )
        reprec = testdir.inline_run()
        reprec.assertoutcome(passed=1)

    def test_usefixtures_marker(self, testdir):
        testdir.makepyfile(
            """
            import pytest

            values = []

            @pytest.fixture(scope="class")
            def myfix(request):
                request.cls.hello = "world"
                values.append(1)

            class TestClass(object):
                def test_one(self):
                    assert self.hello == "world"
                    assert len(values) == 1
                def test_two(self):
                    assert self.hello == "world"
                    assert len(values) == 1
            pytest.mark.usefixtures("myfix")(TestClass)
        """
        )
        reprec = testdir.inline_run()
        reprec.assertoutcome(passed=2)

    def test_usefixtures_ini(self, testdir):
        testdir.makeini(
            """
            [pytest]
            usefixtures = myfix
        """
        )
        testdir.makeconftest(
            """
            import pytest

            @pytest.fixture(scope="class")
            def myfix(request):
                request.cls.hello = "world"

        """
        )
        testdir.makepyfile(
            """
            class TestClass(object):
                def test_one(self):
                    assert self.hello == "world"
                def test_two(self):
                    assert self.hello == "world"
        """
        )
        reprec = testdir.inline_run()
        reprec.assertoutcome(passed=2)

    def test_usefixtures_seen_in_showmarkers(self, testdir):
        result = testdir.runpytest("--markers")
        result.stdout.fnmatch_lines(
            """
            *usefixtures(fixturename1*mark tests*fixtures*
        """
        )

    def test_request_instance_issue203(self, testdir):
        testdir.makepyfile(
            """
            import pytest

            class TestClass(object):
                @pytest.fixture
                def setup1(self, request):
                    assert self == request.instance
                    self.arg1 = 1
                def test_hello(self, setup1):
                    assert self.arg1 == 1
        """
        )
        reprec = testdir.inline_run()
        reprec.assertoutcome(passed=1)

    def test_fixture_parametrized_with_iterator(self, testdir):
        testdir.makepyfile(
            """
            import pytest

            values = []
            def f():
                yield 1
                yield 2
            dec = pytest.fixture(scope="module", params=f())

            @dec
            def arg(request):
                return request.param
            @dec
            def arg2(request):
                return request.param

            def test_1(arg):
                values.append(arg)
            def test_2(arg2):
                values.append(arg2*10)
        """
        )
        reprec = testdir.inline_run("-v")
        reprec.assertoutcome(passed=4)
        values = reprec.getcalls("pytest_runtest_call")[0].item.module.values
        assert values == [1, 2, 10, 20]

    def test_setup_functions_as_fixtures(self, testdir):
        """Ensure setup_* methods obey fixture scope rules (#517, #3094)."""
        testdir.makepyfile(
            """
            import pytest

            DB_INITIALIZED = None

            @pytest.yield_fixture(scope="session", autouse=True)
            def db():
                global DB_INITIALIZED
                DB_INITIALIZED = True
                yield
                DB_INITIALIZED = False

            def setup_module():
                assert DB_INITIALIZED

            def teardown_module():
                assert DB_INITIALIZED

            class TestClass(object):

                def setup_method(self, method):
                    assert DB_INITIALIZED

                def teardown_method(self, method):
                    assert DB_INITIALIZED

                def test_printer_1(self):
                    pass

                def test_printer_2(self):
                    pass
        """
        )
        result = testdir.runpytest()
        result.stdout.fnmatch_lines(["* 2 passed in *"])


class TestFixtureManagerParseFactories(object):
    @pytest.fixture
    def testdir(self, request):
        testdir = request.getfixturevalue("testdir")
        testdir.makeconftest(
            """
            import pytest

            @pytest.fixture
            def hello(request):
                return "conftest"

            @pytest.fixture
            def fm(request):
                return request._fixturemanager

            @pytest.fixture
            def item(request):
                return request._pyfuncitem
        """
        )
        return testdir

    def test_parsefactories_evil_objects_issue214(self, testdir):
        testdir.makepyfile(
            """
            class A(object):
                def __call__(self):
                    pass
                def __getattr__(self, name):
                    raise RuntimeError()
            a = A()
            def test_hello():
                pass
        """
        )
        reprec = testdir.inline_run()
        reprec.assertoutcome(passed=1, failed=0)

    def test_parsefactories_conftest(self, testdir):
        testdir.makepyfile(
            """
            def test_hello(item, fm):
                for name in ("fm", "hello", "item"):
                    faclist = fm.getfixturedefs(name, item.nodeid)
                    assert len(faclist) == 1
                    fac = faclist[0]
                    assert fac.func.__name__ == name
        """
        )
        reprec = testdir.inline_run("-s")
        reprec.assertoutcome(passed=1)

    def test_parsefactories_conftest_and_module_and_class(self, testdir):
        testdir.makepyfile(
            """
            import pytest
            import six

            @pytest.fixture
            def hello(request):
                return "module"
            class TestClass(object):
                @pytest.fixture
                def hello(self, request):
                    return "class"
                def test_hello(self, item, fm):
                    faclist = fm.getfixturedefs("hello", item.nodeid)
                    print(faclist)
                    assert len(faclist) == 3

                    assert faclist[0].func(item._request) == "conftest"
                    assert faclist[1].func(item._request) == "module"
                    assert faclist[2].func(item._request) == "class"
        """
        )
        reprec = testdir.inline_run("-s")
        reprec.assertoutcome(passed=1)

    def test_parsefactories_relative_node_ids(self, testdir):
        # example mostly taken from:
        # https://mail.python.org/pipermail/pytest-dev/2014-September/002617.html
        runner = testdir.mkdir("runner")
        package = testdir.mkdir("package")
        package.join("conftest.py").write(
            textwrap.dedent(
                """\
            import pytest
            @pytest.fixture
            def one():
                return 1
            """
            )
        )
        package.join("test_x.py").write(
            textwrap.dedent(
                """\
                def test_x(one):
                    assert one == 1
                """
            )
        )
        sub = package.mkdir("sub")
        sub.join("__init__.py").ensure()
        sub.join("conftest.py").write(
            textwrap.dedent(
                """\
                import pytest
                @pytest.fixture
                def one():
                    return 2
                """
            )
        )
        sub.join("test_y.py").write(
            textwrap.dedent(
                """\
                def test_x(one):
                    assert one == 2
                """
            )
        )
        reprec = testdir.inline_run()
        reprec.assertoutcome(passed=2)
        with runner.as_cwd():
            reprec = testdir.inline_run("..")
            reprec.assertoutcome(passed=2)

    def test_package_xunit_fixture(self, testdir):
        testdir.makepyfile(
            __init__="""\
            values = []
        """
        )
        package = testdir.mkdir("package")
        package.join("__init__.py").write(
            textwrap.dedent(
                """\
                from .. import values
                def setup_module():
                    values.append("package")
                def teardown_module():
                    values[:] = []
                """
            )
        )
        package.join("test_x.py").write(
            textwrap.dedent(
                """\
                from .. import values
                def test_x():
                    assert values == ["package"]
                """
            )
        )
        package = testdir.mkdir("package2")
        package.join("__init__.py").write(
            textwrap.dedent(
                """\
                from .. import values
                def setup_module():
                    values.append("package2")
                def teardown_module():
                    values[:] = []
                """
            )
        )
        package.join("test_x.py").write(
            textwrap.dedent(
                """\
                from .. import values
                def test_x():
                    assert values == ["package2"]
                """
            )
        )
        reprec = testdir.inline_run()
        reprec.assertoutcome(passed=2)

    def test_package_fixture_complex(self, testdir):
        testdir.makepyfile(
            __init__="""\
            values = []
        """
        )
        testdir.syspathinsert(testdir.tmpdir.dirname)
        package = testdir.mkdir("package")
        package.join("__init__.py").write("")
        package.join("conftest.py").write(
            textwrap.dedent(
                """\
                import pytest
                from .. import values
                @pytest.fixture(scope="package")
                def one():
                    values.append("package")
                    yield values
                    values.pop()
                @pytest.fixture(scope="package", autouse=True)
                def two():
                    values.append("package-auto")
                    yield values
                    values.pop()
                """
            )
        )
        package.join("test_x.py").write(
            textwrap.dedent(
                """\
                from .. import values
                def test_package_autouse():
                    assert values == ["package-auto"]
                def test_package(one):
                    assert values == ["package-auto", "package"]
                """
            )
        )
        reprec = testdir.inline_run()
        reprec.assertoutcome(passed=2)

    def test_collect_custom_items(self, testdir):
        testdir.copy_example("fixtures/custom_item")
        result = testdir.runpytest("foo")
        result.stdout.fnmatch_lines(["*passed*"])


class TestAutouseDiscovery(object):
    @pytest.fixture
    def testdir(self, testdir):
        testdir.makeconftest(
            """
            import pytest
            @pytest.fixture(autouse=True)
            def perfunction(request, tmpdir):
                pass

            @pytest.fixture()
            def arg1(tmpdir):
                pass
            @pytest.fixture(autouse=True)
            def perfunction2(arg1):
                pass

            @pytest.fixture
            def fm(request):
                return request._fixturemanager

            @pytest.fixture
            def item(request):
                return request._pyfuncitem
        """
        )
        return testdir

    def test_parsefactories_conftest(self, testdir):
        testdir.makepyfile(
            """
            from _pytest.pytester import get_public_names
            def test_check_setup(item, fm):
                autousenames = fm._getautousenames(item.nodeid)
                assert len(get_public_names(autousenames)) == 2
                assert "perfunction2" in autousenames
                assert "perfunction" in autousenames
        """
        )
        reprec = testdir.inline_run("-s")
        reprec.assertoutcome(passed=1)

    def test_two_classes_separated_autouse(self, testdir):
        testdir.makepyfile(
            """
            import pytest
            class TestA(object):
                values = []
                @pytest.fixture(autouse=True)
                def setup1(self):
                    self.values.append(1)
                def test_setup1(self):
                    assert self.values == [1]
            class TestB(object):
                values = []
                @pytest.fixture(autouse=True)
                def setup2(self):
                    self.values.append(1)
                def test_setup2(self):
                    assert self.values == [1]
        """
        )
        reprec = testdir.inline_run()
        reprec.assertoutcome(passed=2)

    def test_setup_at_classlevel(self, testdir):
        testdir.makepyfile(
            """
            import pytest
            class TestClass(object):
                @pytest.fixture(autouse=True)
                def permethod(self, request):
                    request.instance.funcname = request.function.__name__
                def test_method1(self):
                    assert self.funcname == "test_method1"
                def test_method2(self):
                    assert self.funcname == "test_method2"
        """
        )
        reprec = testdir.inline_run("-s")
        reprec.assertoutcome(passed=2)

    @pytest.mark.xfail(reason="'enabled' feature not implemented")
    def test_setup_enabled_functionnode(self, testdir):
        testdir.makepyfile(
            """
            import pytest

            def enabled(parentnode, markers):
                return "needsdb" in markers

            @pytest.fixture(params=[1,2])
            def db(request):
                return request.param

            @pytest.fixture(enabled=enabled, autouse=True)
            def createdb(db):
                pass

            def test_func1(request):
                assert "db" not in request.fixturenames

            @pytest.mark.needsdb
            def test_func2(request):
                assert "db" in request.fixturenames
        """
        )
        reprec = testdir.inline_run("-s")
        reprec.assertoutcome(passed=2)

    def test_callables_nocode(self, testdir):
        """
        an imported mock.call would break setup/factory discovery
        due to it being callable and __code__ not being a code object
        """
        testdir.makepyfile(
            """
           class _call(tuple):
               def __call__(self, *k, **kw):
                   pass
               def __getattr__(self, k):
                   return self

           call = _call()
        """
        )
        reprec = testdir.inline_run("-s")
        reprec.assertoutcome(failed=0, passed=0)

    def test_autouse_in_conftests(self, testdir):
        a = testdir.mkdir("a")
        b = testdir.mkdir("a1")
        conftest = testdir.makeconftest(
            """
            import pytest
            @pytest.fixture(autouse=True)
            def hello():
                xxx
        """
        )
        conftest.move(a.join(conftest.basename))
        a.join("test_something.py").write("def test_func(): pass")
        b.join("test_otherthing.py").write("def test_func(): pass")
        result = testdir.runpytest()
        result.stdout.fnmatch_lines(
            """
            *1 passed*1 error*
        """
        )

    def test_autouse_in_module_and_two_classes(self, testdir):
        testdir.makepyfile(
            """
            import pytest
            values = []
            @pytest.fixture(autouse=True)
            def append1():
                values.append("module")
            def test_x():
                assert values == ["module"]

            class TestA(object):
                @pytest.fixture(autouse=True)
                def append2(self):
                    values.append("A")
                def test_hello(self):
                    assert values == ["module", "module", "A"], values
            class TestA2(object):
                def test_world(self):
                    assert values == ["module", "module", "A", "module"], values
        """
        )
        reprec = testdir.inline_run()
        reprec.assertoutcome(passed=3)


class TestAutouseManagement(object):
    def test_autouse_conftest_mid_directory(self, testdir):
        pkgdir = testdir.mkpydir("xyz123")
        pkgdir.join("conftest.py").write(
            textwrap.dedent(
                """\
                import pytest
                @pytest.fixture(autouse=True)
                def app():
                    import sys
                    sys._myapp = "hello"
                """
            )
        )
        t = pkgdir.ensure("tests", "test_app.py")
        t.write(
            textwrap.dedent(
                """\
                import sys
                def test_app():
                    assert sys._myapp == "hello"
                """
            )
        )
        reprec = testdir.inline_run("-s")
        reprec.assertoutcome(passed=1)

    def test_funcarg_and_setup(self, testdir):
        testdir.makepyfile(
            """
            import pytest
            values = []
            @pytest.fixture(scope="module")
            def arg():
                values.append(1)
                return 0
            @pytest.fixture(scope="module", autouse=True)
            def something(arg):
                values.append(2)

            def test_hello(arg):
                assert len(values) == 2
                assert values == [1,2]
                assert arg == 0

            def test_hello2(arg):
                assert len(values) == 2
                assert values == [1,2]
                assert arg == 0
        """
        )
        reprec = testdir.inline_run()
        reprec.assertoutcome(passed=2)

    def test_uses_parametrized_resource(self, testdir):
        testdir.makepyfile(
            """
            import pytest
            values = []
            @pytest.fixture(params=[1,2])
            def arg(request):
                return request.param

            @pytest.fixture(autouse=True)
            def something(arg):
                values.append(arg)

            def test_hello():
                if len(values) == 1:
                    assert values == [1]
                elif len(values) == 2:
                    assert values == [1, 2]
                else:
                    0/0

        """
        )
        reprec = testdir.inline_run("-s")
        reprec.assertoutcome(passed=2)

    def test_session_parametrized_function(self, testdir):
        testdir.makepyfile(
            """
            import pytest

            values = []

            @pytest.fixture(scope="session", params=[1,2])
            def arg(request):
               return request.param

            @pytest.fixture(scope="function", autouse=True)
            def append(request, arg):
                if request.function.__name__ == "test_some":
                    values.append(arg)

            def test_some():
                pass

            def test_result(arg):
                assert len(values) == arg
                assert values[:arg] == [1,2][:arg]
        """
        )
        reprec = testdir.inline_run("-v", "-s")
        reprec.assertoutcome(passed=4)

    def test_class_function_parametrization_finalization(self, testdir):
        p = testdir.makeconftest(
            """
            import pytest
            import pprint

            values = []

            @pytest.fixture(scope="function", params=[1,2])
            def farg(request):
                return request.param

            @pytest.fixture(scope="class", params=list("ab"))
            def carg(request):
                return request.param

            @pytest.fixture(scope="function", autouse=True)
            def append(request, farg, carg):
                def fin():
                    values.append("fin_%s%s" % (carg, farg))
                request.addfinalizer(fin)
        """
        )
        testdir.makepyfile(
            """
            import pytest

            class TestClass(object):
                def test_1(self):
                    pass
            class TestClass2(object):
                def test_2(self):
                    pass
        """
        )
        confcut = "--confcutdir={}".format(testdir.tmpdir)
        reprec = testdir.inline_run("-v", "-s", confcut)
        reprec.assertoutcome(passed=8)
        config = reprec.getcalls("pytest_unconfigure")[0].config
        values = config.pluginmanager._getconftestmodules(p)[0].values
        assert values == ["fin_a1", "fin_a2", "fin_b1", "fin_b2"] * 2

    def test_scope_ordering(self, testdir):
        testdir.makepyfile(
            """
            import pytest
            values = []
            @pytest.fixture(scope="function", autouse=True)
            def fappend2():
                values.append(2)
            @pytest.fixture(scope="class", autouse=True)
            def classappend3():
                values.append(3)
            @pytest.fixture(scope="module", autouse=True)
            def mappend():
                values.append(1)

            class TestHallo(object):
                def test_method(self):
                    assert values == [1,3,2]
        """
        )
        reprec = testdir.inline_run()
        reprec.assertoutcome(passed=1)

    def test_parametrization_setup_teardown_ordering(self, testdir):
        testdir.makepyfile(
            """
            import pytest
            values = []
            def pytest_generate_tests(metafunc):
                if metafunc.cls is None:
                    assert metafunc.function is test_finish
                if metafunc.cls is not None:
                    metafunc.parametrize("item", [1,2], scope="class")
            class TestClass(object):
                @pytest.fixture(scope="class", autouse=True)
                def addteardown(self, item, request):
                    values.append("setup-%d" % item)
                    request.addfinalizer(lambda: values.append("teardown-%d" % item))
                def test_step1(self, item):
                    values.append("step1-%d" % item)
                def test_step2(self, item):
                    values.append("step2-%d" % item)

            def test_finish():
                print(values)
                assert values == ["setup-1", "step1-1", "step2-1", "teardown-1",
                             "setup-2", "step1-2", "step2-2", "teardown-2",]
        """
        )
        reprec = testdir.inline_run("-s")
        reprec.assertoutcome(passed=5)

    def test_ordering_autouse_before_explicit(self, testdir):
        testdir.makepyfile(
            """
            import pytest

            values = []
            @pytest.fixture(autouse=True)
            def fix1():
                values.append(1)
            @pytest.fixture()
            def arg1():
                values.append(2)
            def test_hello(arg1):
                assert values == [1,2]
        """
        )
        reprec = testdir.inline_run()
        reprec.assertoutcome(passed=1)

    @pytest.mark.parametrize("param1", ["", "params=[1]"], ids=["p00", "p01"])
    @pytest.mark.parametrize("param2", ["", "params=[1]"], ids=["p10", "p11"])
    def test_ordering_dependencies_torndown_first(self, testdir, param1, param2):
        """#226"""
        testdir.makepyfile(
            """
            import pytest
            values = []
            @pytest.fixture(%(param1)s)
            def arg1(request):
                request.addfinalizer(lambda: values.append("fin1"))
                values.append("new1")
            @pytest.fixture(%(param2)s)
            def arg2(request, arg1):
                request.addfinalizer(lambda: values.append("fin2"))
                values.append("new2")

            def test_arg(arg2):
                pass
            def test_check():
                assert values == ["new1", "new2", "fin2", "fin1"]
        """
            % locals()
        )
        reprec = testdir.inline_run("-s")
        reprec.assertoutcome(passed=2)


class TestFixtureMarker(object):
    def test_parametrize(self, testdir):
        testdir.makepyfile(
            """
            import pytest
            @pytest.fixture(params=["a", "b", "c"])
            def arg(request):
                return request.param
            values = []
            def test_param(arg):
                values.append(arg)
            def test_result():
                assert values == list("abc")
        """
        )
        reprec = testdir.inline_run()
        reprec.assertoutcome(passed=4)

    def test_multiple_parametrization_issue_736(self, testdir):
        testdir.makepyfile(
            """
            import pytest

            @pytest.fixture(params=[1,2,3])
            def foo(request):
                return request.param

            @pytest.mark.parametrize('foobar', [4,5,6])
            def test_issue(foo, foobar):
                assert foo in [1,2,3]
                assert foobar in [4,5,6]
        """
        )
        reprec = testdir.inline_run()
        reprec.assertoutcome(passed=9)

    @pytest.mark.parametrize(
        "param_args",
        ["'fixt, val'", "'fixt,val'", "['fixt', 'val']", "('fixt', 'val')"],
    )
    def test_override_parametrized_fixture_issue_979(self, testdir, param_args):
        """Make sure a parametrized argument can override a parametrized fixture.

        This was a regression introduced in the fix for #736.
        """
        testdir.makepyfile(
            """
            import pytest

            @pytest.fixture(params=[1, 2])
            def fixt(request):
                return request.param

            @pytest.mark.parametrize(%s, [(3, 'x'), (4, 'x')])
            def test_foo(fixt, val):
                pass
        """
            % param_args
        )
        reprec = testdir.inline_run()
        reprec.assertoutcome(passed=2)
```
### 2 - testing/python/setup_only.py:

Start line: 157, End line: 183

```python
def test_show_fixtures_with_parameter_ids(testdir, mode):
    testdir.makeconftest(
        '''
        import pytest
        @pytest.fixture(
            scope='session', params=['foo', 'bar'], ids=['spam', 'ham'])
        def arg_same():
            """session scoped fixture"""
        '''
    )
    p = testdir.makepyfile(
        '''
        import pytest
        @pytest.fixture(scope='function')
        def arg_other(arg_same):
            """function scoped fixture"""
        def test_arg1(arg_other):
            pass
    '''
    )

    result = testdir.runpytest(mode, p)
    assert result.ret == 0

    result.stdout.fnmatch_lines(
        ["SETUP    S arg_same?spam?", "SETUP    S arg_same?ham?"]
    )
```
### 3 - testing/python/metafunc.py:

Start line: 1138, End line: 1167

```python
class TestMetafuncFunctional(object):

    def test_fixture_parametrized_empty_ids(self, testdir):
        """Fixtures parametrized with empty ids cause an internal error (#1849)."""
        testdir.makepyfile(
            """
            import pytest

            @pytest.fixture(scope="module", ids=[], params=[])
            def temp(request):
               return request.param

            def test_temp(temp):
                 pass
        """
        )
        result = testdir.runpytest()
        result.stdout.fnmatch_lines(["* 1 skipped *"])

    def test_parametrized_empty_ids(self, testdir):
        """Tests parametrized with empty ids cause an internal error (#1849)."""
        testdir.makepyfile(
            """
            import pytest

            @pytest.mark.parametrize('temp', [], ids=list())
            def test_temp(temp):
                 pass
        """
        )
        result = testdir.runpytest()
        result.stdout.fnmatch_lines(["* 1 skipped *"])
```
### 4 - testing/python/collect.py:

Start line: 449, End line: 478

```python
class TestFunction(object):

    def test_parametrize_overrides_indirect_dependency_fixture(self, testdir):
        """Test parametrization when parameter overrides a fixture that a test indirectly depends on"""
        testdir.makepyfile(
            """
            import pytest

            fix3_instantiated = False

            @pytest.fixture
            def fix1(fix2):
               return fix2 + '1'

            @pytest.fixture
            def fix2(fix3):
               return fix3 + '2'

            @pytest.fixture
            def fix3():
               global fix3_instantiated
               fix3_instantiated = True
               return '3'

            @pytest.mark.parametrize('fix2', ['2'])
            def test_it(fix1):
               assert fix1 == '21'
               assert not fix3_instantiated
        """
        )
        rec = testdir.inline_run()
        rec.assertoutcome(passed=1)
```
### 5 - testing/python/collect.py:

Start line: 430, End line: 447

```python
class TestFunction(object):

    def test_parametrize_overrides_parametrized_fixture(self, testdir):
        """Test parametrization when parameter overrides existing parametrized fixture with same name."""
        testdir.makepyfile(
            """
            import pytest

            @pytest.fixture(params=[1, 2])
            def value(request):
                return request.param

            @pytest.mark.parametrize('value',
                                     ['overridden'])
            def test_overridden_via_param(value):
                assert value == 'overridden'
        """
        )
        rec = testdir.inline_run()
        rec.assertoutcome(passed=1)
```
### 6 - testing/python/fixtures.py:

Start line: 47, End line: 1011

```python
@pytest.mark.pytester_example_path("fixtures/fill_fixtures")
class TestFillFixtures(object):
    def test_fillfuncargs_exposed(self):
        # used by oejskit, kept for compatibility
        assert pytest._fillfuncargs == fixtures.fillfixtures

    def test_funcarg_lookupfails(self, testdir):
        testdir.copy_example()
        result = testdir.runpytest()  # "--collect-only")
        assert result.ret != 0
        result.stdout.fnmatch_lines(
            """
            *def test_func(some)*
            *fixture*some*not found*
            *xyzsomething*
            """
        )

    def test_detect_recursive_dependency_error(self, testdir):
        testdir.copy_example()
        result = testdir.runpytest()
        result.stdout.fnmatch_lines(
            ["*recursive dependency involving fixture 'fix1' detected*"]
        )

    def test_funcarg_basic(self, testdir):
        testdir.copy_example()
        item = testdir.getitem(Path("test_funcarg_basic.py"))
        fixtures.fillfixtures(item)
        del item.funcargs["request"]
        assert len(get_public_names(item.funcargs)) == 2
        assert item.funcargs["some"] == "test_func"
        assert item.funcargs["other"] == 42

    def test_funcarg_lookup_modulelevel(self, testdir):
        testdir.copy_example()
        reprec = testdir.inline_run()
        reprec.assertoutcome(passed=2)

    def test_funcarg_lookup_classlevel(self, testdir):
        p = testdir.copy_example()
        result = testdir.runpytest(p)
        result.stdout.fnmatch_lines(["*1 passed*"])

    def test_conftest_funcargs_only_available_in_subdir(self, testdir):
        testdir.copy_example()
        result = testdir.runpytest("-v")
        result.assert_outcomes(passed=2)

    def test_extend_fixture_module_class(self, testdir):
        testfile = testdir.copy_example()
        result = testdir.runpytest()
        result.stdout.fnmatch_lines(["*1 passed*"])
        result = testdir.runpytest(testfile)
        result.stdout.fnmatch_lines(["*1 passed*"])

    def test_extend_fixture_conftest_module(self, testdir):
        p = testdir.copy_example()
        result = testdir.runpytest()
        result.stdout.fnmatch_lines(["*1 passed*"])
        result = testdir.runpytest(next(p.visit("test_*.py")))
        result.stdout.fnmatch_lines(["*1 passed*"])

    def test_extend_fixture_conftest_conftest(self, testdir):
        p = testdir.copy_example()
        result = testdir.runpytest()
        result.stdout.fnmatch_lines(["*1 passed*"])
        result = testdir.runpytest(next(p.visit("test_*.py")))
        result.stdout.fnmatch_lines(["*1 passed*"])

    def test_extend_fixture_conftest_plugin(self, testdir):
        testdir.makepyfile(
            testplugin="""
            import pytest

            @pytest.fixture
            def foo():
                return 7
        """
        )
        testdir.syspathinsert()
        testdir.makeconftest(
            """
            import pytest

            pytest_plugins = 'testplugin'

            @pytest.fixture
            def foo(foo):
                return foo + 7
        """
        )
        testdir.makepyfile(
            """
            def test_foo(foo):
                assert foo == 14
        """
        )
        result = testdir.runpytest("-s")
        assert result.ret == 0

    def test_extend_fixture_plugin_plugin(self, testdir):
        # Two plugins should extend each order in loading order
        testdir.makepyfile(
            testplugin0="""
            import pytest

            @pytest.fixture
            def foo():
                return 7
        """
        )
        testdir.makepyfile(
            testplugin1="""
            import pytest

            @pytest.fixture
            def foo(foo):
                return foo + 7
        """
        )
        testdir.syspathinsert()
        testdir.makepyfile(
            """
            pytest_plugins = ['testplugin0', 'testplugin1']

            def test_foo(foo):
                assert foo == 14
        """
        )
        result = testdir.runpytest()
        assert result.ret == 0

    def test_override_parametrized_fixture_conftest_module(self, testdir):
        """Test override of the parametrized fixture with non-parametrized one on the test module level."""
        testdir.makeconftest(
            """
            import pytest

            @pytest.fixture(params=[1, 2, 3])
            def spam(request):
                return request.param
        """
        )
        testfile = testdir.makepyfile(
            """
            import pytest

            @pytest.fixture
            def spam():
                return 'spam'

            def test_spam(spam):
                assert spam == 'spam'
        """
        )
        result = testdir.runpytest()
        result.stdout.fnmatch_lines(["*1 passed*"])
        result = testdir.runpytest(testfile)
        result.stdout.fnmatch_lines(["*1 passed*"])

    def test_override_parametrized_fixture_conftest_conftest(self, testdir):
        """Test override of the parametrized fixture with non-parametrized one on the conftest level."""
        testdir.makeconftest(
            """
            import pytest

            @pytest.fixture(params=[1, 2, 3])
            def spam(request):
                return request.param
        """
        )
        subdir = testdir.mkpydir("subdir")
        subdir.join("conftest.py").write(
            textwrap.dedent(
                """\
                import pytest

                @pytest.fixture
                def spam():
                    return 'spam'
                """
            )
        )
        testfile = subdir.join("test_spam.py")
        testfile.write(
            textwrap.dedent(
                """\
                def test_spam(spam):
                    assert spam == "spam"
                """
            )
        )
        result = testdir.runpytest()
        result.stdout.fnmatch_lines(["*1 passed*"])
        result = testdir.runpytest(testfile)
        result.stdout.fnmatch_lines(["*1 passed*"])

    def test_override_non_parametrized_fixture_conftest_module(self, testdir):
        """Test override of the non-parametrized fixture with parametrized one on the test module level."""
        testdir.makeconftest(
            """
            import pytest

            @pytest.fixture
            def spam():
                return 'spam'
        """
        )
        testfile = testdir.makepyfile(
            """
            import pytest

            @pytest.fixture(params=[1, 2, 3])
            def spam(request):
                return request.param

            params = {'spam': 1}

            def test_spam(spam):
                assert spam == params['spam']
                params['spam'] += 1
        """
        )
        result = testdir.runpytest()
        result.stdout.fnmatch_lines(["*3 passed*"])
        result = testdir.runpytest(testfile)
        result.stdout.fnmatch_lines(["*3 passed*"])

    def test_override_non_parametrized_fixture_conftest_conftest(self, testdir):
        """Test override of the non-parametrized fixture with parametrized one on the conftest level."""
        testdir.makeconftest(
            """
            import pytest

            @pytest.fixture
            def spam():
                return 'spam'
        """
        )
        subdir = testdir.mkpydir("subdir")
        subdir.join("conftest.py").write(
            textwrap.dedent(
                """\
                import pytest

                @pytest.fixture(params=[1, 2, 3])
                def spam(request):
                    return request.param
                """
            )
        )
        testfile = subdir.join("test_spam.py")
        testfile.write(
            textwrap.dedent(
                """\
                params = {'spam': 1}

                def test_spam(spam):
                    assert spam == params['spam']
                    params['spam'] += 1
                """
            )
        )
        result = testdir.runpytest()
        result.stdout.fnmatch_lines(["*3 passed*"])
        result = testdir.runpytest(testfile)
        result.stdout.fnmatch_lines(["*3 passed*"])

    def test_override_autouse_fixture_with_parametrized_fixture_conftest_conftest(
        self, testdir
    ):
        """Test override of the autouse fixture with parametrized one on the conftest level.
        This test covers the issue explained in issue 1601
        """
        testdir.makeconftest(
            """
            import pytest

            @pytest.fixture(autouse=True)
            def spam():
                return 'spam'
        """
        )
        subdir = testdir.mkpydir("subdir")
        subdir.join("conftest.py").write(
            textwrap.dedent(
                """\
                import pytest

                @pytest.fixture(params=[1, 2, 3])
                def spam(request):
                    return request.param
                """
            )
        )
        testfile = subdir.join("test_spam.py")
        testfile.write(
            textwrap.dedent(
                """\
                params = {'spam': 1}

                def test_spam(spam):
                    assert spam == params['spam']
                    params['spam'] += 1
                """
            )
        )
        result = testdir.runpytest()
        result.stdout.fnmatch_lines(["*3 passed*"])
        result = testdir.runpytest(testfile)
        result.stdout.fnmatch_lines(["*3 passed*"])

    def test_autouse_fixture_plugin(self, testdir):
        # A fixture from a plugin has no baseid set, which screwed up
        # the autouse fixture handling.
        testdir.makepyfile(
            testplugin="""
            import pytest

            @pytest.fixture(autouse=True)
            def foo(request):
                request.function.foo = 7
        """
        )
        testdir.syspathinsert()
        testdir.makepyfile(
            """
            pytest_plugins = 'testplugin'

            def test_foo(request):
                assert request.function.foo == 7
        """
        )
        result = testdir.runpytest()
        assert result.ret == 0

    def test_funcarg_lookup_error(self, testdir):
        testdir.makeconftest(
            """
            import pytest

            @pytest.fixture
            def a_fixture(): pass

            @pytest.fixture
            def b_fixture(): pass

            @pytest.fixture
            def c_fixture(): pass

            @pytest.fixture
            def d_fixture(): pass
        """
        )
        testdir.makepyfile(
            """
            def test_lookup_error(unknown):
                pass
        """
        )
        result = testdir.runpytest()
        result.stdout.fnmatch_lines(
            [
                "*ERROR at setup of test_lookup_error*",
                "  def test_lookup_error(unknown):*",
                "E       fixture 'unknown' not found",
                ">       available fixtures:*a_fixture,*b_fixture,*c_fixture,*d_fixture*monkeypatch,*",  # sorted
                ">       use 'py*test --fixtures *' for help on them.",
                "*1 error*",
            ]
        )
        assert "INTERNAL" not in result.stdout.str()

    def test_fixture_excinfo_leak(self, testdir):
        # on python2 sys.excinfo would leak into fixture executions
        testdir.makepyfile(
            """
            import sys
            import traceback
            import pytest

            @pytest.fixture
            def leak():
                if sys.exc_info()[0]:  # python3 bug :)
                    traceback.print_exc()
                #fails
                assert sys.exc_info() == (None, None, None)

            def test_leak(leak):
                if sys.exc_info()[0]:  # python3 bug :)
                    traceback.print_exc()
                assert sys.exc_info() == (None, None, None)
        """
        )
        result = testdir.runpytest()
        assert result.ret == 0


class TestRequestBasic(object):
    def test_request_attributes(self, testdir):
        item = testdir.getitem(
            """
            import pytest

            @pytest.fixture
            def something(request): pass
            def test_func(something): pass
        """
        )
        req = fixtures.FixtureRequest(item)
        assert req.function == item.obj
        assert req.keywords == item.keywords
        assert hasattr(req.module, "test_func")
        assert req.cls is None
        assert req.function.__name__ == "test_func"
        assert req.config == item.config
        assert repr(req).find(req.function.__name__) != -1

    def test_request_attributes_method(self, testdir):
        item, = testdir.getitems(
            """
            import pytest
            class TestB(object):

                @pytest.fixture
                def something(self, request):
                    return 1
                def test_func(self, something):
                    pass
        """
        )
        req = item._request
        assert req.cls.__name__ == "TestB"
        assert req.instance.__class__ == req.cls

    def test_request_contains_funcarg_arg2fixturedefs(self, testdir):
        modcol = testdir.getmodulecol(
            """
            import pytest
            @pytest.fixture
            def something(request):
                pass
            class TestClass(object):
                def test_method(self, something):
                    pass
        """
        )
        item1, = testdir.genitems([modcol])
        assert item1.name == "test_method"
        arg2fixturedefs = fixtures.FixtureRequest(item1)._arg2fixturedefs
        assert len(arg2fixturedefs) == 1
        assert arg2fixturedefs["something"][0].argname == "something"

    @pytest.mark.skipif(
        hasattr(sys, "pypy_version_info"),
        reason="this method of test doesn't work on pypy",
    )
    def test_request_garbage(self, testdir):
        try:
            import xdist  # noqa
        except ImportError:
            pass
        else:
            pytest.xfail("this test is flaky when executed with xdist")
        testdir.makepyfile(
            """
            import sys
            import pytest
            from _pytest.fixtures import PseudoFixtureDef
            import gc

            @pytest.fixture(autouse=True)
            def something(request):
                original = gc.get_debug()
                gc.set_debug(gc.DEBUG_SAVEALL)
                gc.collect()

                yield

                try:
                    gc.collect()
                    leaked = [x for _ in gc.garbage if isinstance(_, PseudoFixtureDef)]
                    assert leaked == []
                finally:
                    gc.set_debug(original)

            def test_func():
                pass
        """
        )
        result = testdir.runpytest_subprocess()
        result.stdout.fnmatch_lines(["* 1 passed in *"])

    def test_getfixturevalue_recursive(self, testdir):
        testdir.makeconftest(
            """
            import pytest

            @pytest.fixture
            def something(request):
                return 1
        """
        )
        testdir.makepyfile(
            """
            import pytest

            @pytest.fixture
            def something(request):
                return request.getfixturevalue("something") + 1
            def test_func(something):
                assert something == 2
        """
        )
        reprec = testdir.inline_run()
        reprec.assertoutcome(passed=1)

    def test_getfixturevalue_teardown(self, testdir):
        """
        Issue #1895

        `test_inner` requests `inner` fixture, which in turn requests `resource`
        using `getfixturevalue`. `test_func` then requests `resource`.

        `resource` is teardown before `inner` because the fixture mechanism won't consider
        `inner` dependent on `resource` when it is used via `getfixturevalue`: `test_func`
        will then cause the `resource`'s finalizer to be called first because of this.
        """
        testdir.makepyfile(
            """
            import pytest

            @pytest.fixture(scope='session')
            def resource():
                r = ['value']
                yield r
                r.pop()

            @pytest.fixture(scope='session')
            def inner(request):
                resource = request.getfixturevalue('resource')
                assert resource == ['value']
                yield
                assert resource == ['value']

            def test_inner(inner):
                pass

            def test_func(resource):
                pass
        """
        )
        result = testdir.runpytest()
        result.stdout.fnmatch_lines(["* 2 passed in *"])

    @pytest.mark.parametrize("getfixmethod", ("getfixturevalue", "getfuncargvalue"))
    def test_getfixturevalue(self, testdir, getfixmethod):
        item = testdir.getitem(
            """
            import pytest
            values = [2]
            @pytest.fixture
            def something(request): return 1
            @pytest.fixture
            def other(request):
                return values.pop()
            def test_func(something): pass
        """
        )
        import contextlib

        if getfixmethod == "getfuncargvalue":
            warning_expectation = pytest.warns(DeprecationWarning)
        else:
            # see #1830 for a cleaner way to accomplish this
            @contextlib.contextmanager
            def expecting_no_warning():
                yield

            warning_expectation = expecting_no_warning()

        req = item._request
        with warning_expectation:
            fixture_fetcher = getattr(req, getfixmethod)
            with pytest.raises(FixtureLookupError):
                fixture_fetcher("notexists")
            val = fixture_fetcher("something")
            assert val == 1
            val = fixture_fetcher("something")
            assert val == 1
            val2 = fixture_fetcher("other")
            assert val2 == 2
            val2 = fixture_fetcher("other")  # see about caching
            assert val2 == 2
            pytest._fillfuncargs(item)
            assert item.funcargs["something"] == 1
            assert len(get_public_names(item.funcargs)) == 2
            assert "request" in item.funcargs

    def test_request_addfinalizer(self, testdir):
        item = testdir.getitem(
            """
            import pytest
            teardownlist = []
            @pytest.fixture
            def something(request):
                request.addfinalizer(lambda: teardownlist.append(1))
            def test_func(something): pass
        """
        )
        item.session._setupstate.prepare(item)
        pytest._fillfuncargs(item)
        # successively check finalization calls
        teardownlist = item.getparent(pytest.Module).obj.teardownlist
        ss = item.session._setupstate
        assert not teardownlist
        ss.teardown_exact(item, None)
        print(ss.stack)
        assert teardownlist == [1]

    def test_request_addfinalizer_failing_setup(self, testdir):
        testdir.makepyfile(
            """
            import pytest
            values = [1]
            @pytest.fixture
            def myfix(request):
                request.addfinalizer(values.pop)
                assert 0
            def test_fix(myfix):
                pass
            def test_finalizer_ran():
                assert not values
        """
        )
        reprec = testdir.inline_run("-s")
        reprec.assertoutcome(failed=1, passed=1)

    def test_request_addfinalizer_failing_setup_module(self, testdir):
        testdir.makepyfile(
            """
            import pytest
            values = [1, 2]
            @pytest.fixture(scope="module")
            def myfix(request):
                request.addfinalizer(values.pop)
                request.addfinalizer(values.pop)
                assert 0
            def test_fix(myfix):
                pass
        """
        )
        reprec = testdir.inline_run("-s")
        mod = reprec.getcalls("pytest_runtest_setup")[0].item.module
        assert not mod.values

    def test_request_addfinalizer_partial_setup_failure(self, testdir):
        p = testdir.makepyfile(
            """
            import pytest
            values = []
            @pytest.fixture
            def something(request):
                request.addfinalizer(lambda: values.append(None))
            def test_func(something, missingarg):
                pass
            def test_second():
                assert len(values) == 1
        """
        )
        result = testdir.runpytest(p)
        result.stdout.fnmatch_lines(
            ["*1 error*"]  # XXX the whole module collection fails
        )

    def test_request_subrequest_addfinalizer_exceptions(self, testdir):
        """
        Ensure exceptions raised during teardown by a finalizer are suppressed
        until all finalizers are called, re-raising the first exception (#2440)
        """
        testdir.makepyfile(
            """
            import pytest
            values = []
            def _excepts(where):
                raise Exception('Error in %s fixture' % where)
            @pytest.fixture
            def subrequest(request):
                return request
            @pytest.fixture
            def something(subrequest):
                subrequest.addfinalizer(lambda: values.append(1))
                subrequest.addfinalizer(lambda: values.append(2))
                subrequest.addfinalizer(lambda: _excepts('something'))
            @pytest.fixture
            def excepts(subrequest):
                subrequest.addfinalizer(lambda: _excepts('excepts'))
                subrequest.addfinalizer(lambda: values.append(3))
            def test_first(something, excepts):
                pass
            def test_second():
                assert values == [3, 2, 1]
        """
        )
        result = testdir.runpytest()
        result.stdout.fnmatch_lines(
            ["*Exception: Error in excepts fixture", "* 2 passed, 1 error in *"]
        )

    def test_request_getmodulepath(self, testdir):
        modcol = testdir.getmodulecol("def test_somefunc(): pass")
        item, = testdir.genitems([modcol])
        req = fixtures.FixtureRequest(item)
        assert req.fspath == modcol.fspath

    def test_request_fixturenames(self, testdir):
        testdir.makepyfile(
            """
            import pytest
            from _pytest.pytester import get_public_names
            @pytest.fixture()
            def arg1():
                pass
            @pytest.fixture()
            def farg(arg1):
                pass
            @pytest.fixture(autouse=True)
            def sarg(tmpdir):
                pass
            def test_function(request, farg):
                assert set(get_public_names(request.fixturenames)) == \
                       set(["tmpdir", "sarg", "arg1", "request", "farg",
                            "tmp_path", "tmp_path_factory"])
        """
        )
        reprec = testdir.inline_run()
        reprec.assertoutcome(passed=1)

    def test_request_fixturenames_dynamic_fixture(self, testdir):
        """Regression test for #3057"""
        testdir.copy_example("fixtures/test_getfixturevalue_dynamic.py")
        result = testdir.runpytest()
        result.stdout.fnmatch_lines(["*1 passed*"])

    def test_funcargnames_compatattr(self, testdir):
        testdir.makepyfile(
            """
            import pytest
            def pytest_generate_tests(metafunc):
                assert metafunc.funcargnames == metafunc.fixturenames
            @pytest.fixture
            def fn(request):
                assert request._pyfuncitem.funcargnames == \
                       request._pyfuncitem.fixturenames
                return request.funcargnames, request.fixturenames

            def test_hello(fn):
                assert fn[0] == fn[1]
        """
        )
        reprec = testdir.inline_run()
        reprec.assertoutcome(passed=1)

    def test_setupdecorator_and_xunit(self, testdir):
        testdir.makepyfile(
            """
            import pytest
            values = []
            @pytest.fixture(scope='module', autouse=True)
            def setup_module():
                values.append("module")
            @pytest.fixture(autouse=True)
            def setup_function():
                values.append("function")

            def test_func():
                pass

            class TestClass(object):
                @pytest.fixture(scope="class", autouse=True)
                def setup_class(self):
                    values.append("class")
                @pytest.fixture(autouse=True)
                def setup_method(self):
                    values.append("method")
                def test_method(self):
                    pass
            def test_all():
                assert values == ["module", "function", "class",
                             "function", "method", "function"]
        """
        )
        reprec = testdir.inline_run("-v")
        reprec.assertoutcome(passed=3)

    def test_fixtures_sub_subdir_normalize_sep(self, testdir):
        # this tests that normalization of nodeids takes place
        b = testdir.mkdir("tests").mkdir("unit")
        b.join("conftest.py").write(
            textwrap.dedent(
                """\
                import pytest
                @pytest.fixture
                def arg1():
                    pass
                """
            )
        )
        p = b.join("test_module.py")
        p.write("def test_func(arg1): pass")
        result = testdir.runpytest(p, "--fixtures")
        assert result.ret == 0
        result.stdout.fnmatch_lines(
            """
            *fixtures defined*conftest*
            *arg1*
        """
        )

    def test_show_fixtures_color_yes(self, testdir):
        testdir.makepyfile("def test_this(): assert 1")
        result = testdir.runpytest("--color=yes", "--fixtures")
        assert "\x1b[32mtmpdir" in result.stdout.str()

    def test_newstyle_with_request(self, testdir):
        testdir.makepyfile(
            """
            import pytest
            @pytest.fixture()
            def arg(request):
                pass
            def test_1(arg):
                pass
        """
        )
        reprec = testdir.inline_run()
        reprec.assertoutcome(passed=1)

    def test_setupcontext_no_param(self, testdir):
        testdir.makepyfile(
            """
            import pytest
            @pytest.fixture(params=[1,2])
            def arg(request):
                return request.param

            @pytest.fixture(autouse=True)
            def mysetup(request, arg):
                assert not hasattr(request, "param")
            def test_1(arg):
                assert arg in (1,2)
        """
        )
        reprec = testdir.inline_run()
        reprec.assertoutcome(passed=2)


class TestRequestMarking(object):
    def test_applymarker(self, testdir):
        item1, item2 = testdir.getitems(
            """
            import pytest

            @pytest.fixture
            def something(request):
                pass
            class TestClass(object):
                def test_func1(self, something):
                    pass
                def test_func2(self, something):
                    pass
        """
        )
        req1 = fixtures.FixtureRequest(item1)
        assert "xfail" not in item1.keywords
        req1.applymarker(pytest.mark.xfail)
        assert "xfail" in item1.keywords
        assert "skipif" not in item1.keywords
        req1.applymarker(pytest.mark.skipif)
        assert "skipif" in item1.keywords
        with pytest.raises(ValueError):
            req1.applymarker(42)

    def test_accesskeywords(self, testdir):
        testdir.makepyfile(
            """
            import pytest
            @pytest.fixture()
            def keywords(request):
                return request.keywords
            @pytest.mark.XYZ
            def test_function(keywords):
                assert keywords["XYZ"]
                assert "abc" not in keywords
        """
        )
        reprec = testdir.inline_run()
        reprec.assertoutcome(passed=1)

    def test_accessmarker_dynamic(self, testdir):
        testdir.makeconftest(
            """
            import pytest
            @pytest.fixture()
            def keywords(request):
                return request.keywords

            @pytest.fixture(scope="class", autouse=True)
            def marking(request):
                request.applymarker(pytest.mark.XYZ("hello"))
        """
        )
        testdir.makepyfile(
            """
            import pytest
            def test_fun1(keywords):
                assert keywords["XYZ"] is not None
                assert "abc" not in keywords
            def test_fun2(keywords):
                assert keywords["XYZ"] is not None
                assert "abc" not in keywords
        """
        )
        reprec = testdir.inline_run()
        reprec.assertoutcome(passed=2)


class TestFixtureUsages(object):
    def test_noargfixturedec(self, testdir):
        testdir.makepyfile(
            """
            import pytest
            @pytest.fixture
            def arg1():
                return 1

            def test_func(arg1):
                assert arg1 == 1
        """
        )
        reprec = testdir.inline_run()
        reprec.assertoutcome(passed=1)

    def test_receives_funcargs(self, testdir):
        testdir.makepyfile(
            """
            import pytest
            @pytest.fixture()
            def arg1():
                return 1

            @pytest.fixture()
            def arg2(arg1):
                return arg1 + 1

            def test_add(arg2):
                assert arg2 == 2
            def test_all(arg1, arg2):
                assert arg1 == 1
                assert arg2 == 2
        """
        )
        reprec = testdir.inline_run()
        reprec.assertoutcome(passed=2)
```
### 7 - testing/python/collect.py:

Start line: 401, End line: 428

```python
class TestFunction(object):

    def test_parametrize_overrides_fixture(self, testdir):
        """Test parametrization when parameter overrides existing fixture with same name."""
        testdir.makepyfile(
            """
            import pytest

            @pytest.fixture
            def value():
                return 'value'

            @pytest.mark.parametrize('value',
                                     ['overridden'])
            def test_overridden_via_param(value):
                assert value == 'overridden'

            @pytest.mark.parametrize('somevalue', ['overridden'])
            def test_not_overridden(value, somevalue):
                assert value == 'value'
                assert somevalue == 'overridden'

            @pytest.mark.parametrize('other,value', [('foo', 'overridden')])
            def test_overridden_via_multiparam(other, value):
                assert other == 'foo'
                assert value == 'overridden'
        """
        )
        rec = testdir.inline_run()
        rec.assertoutcome(passed=3)
```
### 8 - testing/python/fixtures.py:

Start line: 3110, End line: 3953

```python
class TestShowFixtures(object):

    def test_show_fixtures_trimmed_doc(self, testdir):
        p = testdir.makepyfile(
            textwrap.dedent(
                '''\
                import pytest
                @pytest.fixture
                def arg1():
                    """
                    line1
                    line2

                    """
                @pytest.fixture
                def arg2():
                    """
                    line1
                    line2

                    """
                '''
            )
        )
        result = testdir.runpytest("--fixtures", p)
        result.stdout.fnmatch_lines(
            textwrap.dedent(
                """\
                * fixtures defined from test_show_fixtures_trimmed_doc *
                arg2
                    line1
                    line2
                arg1
                    line1
                    line2
                """
            )
        )

    def test_show_fixtures_indented_doc(self, testdir):
        p = testdir.makepyfile(
            textwrap.dedent(
                '''\
                import pytest
                @pytest.fixture
                def fixture1():
                    """
                    line1
                        indented line
                    """
                '''
            )
        )
        result = testdir.runpytest("--fixtures", p)
        result.stdout.fnmatch_lines(
            textwrap.dedent(
                """\
                * fixtures defined from test_show_fixtures_indented_doc *
                fixture1
                    line1
                        indented line
                """
            )
        )

    def test_show_fixtures_indented_doc_first_line_unindented(self, testdir):
        p = testdir.makepyfile(
            textwrap.dedent(
                '''\
                import pytest
                @pytest.fixture
                def fixture1():
                    """line1
                    line2
                        indented line
                    """
                '''
            )
        )
        result = testdir.runpytest("--fixtures", p)
        result.stdout.fnmatch_lines(
            textwrap.dedent(
                """\
                * fixtures defined from test_show_fixtures_indented_doc_first_line_unindented *
                fixture1
                    line1
                    line2
                        indented line
                """
            )
        )

    def test_show_fixtures_indented_in_class(self, testdir):
        p = testdir.makepyfile(
            textwrap.dedent(
                '''\
                import pytest
                class TestClass(object):
                    @pytest.fixture
                    def fixture1(self):
                        """line1
                        line2
                            indented line
                        """
                '''
            )
        )
        result = testdir.runpytest("--fixtures", p)
        result.stdout.fnmatch_lines(
            textwrap.dedent(
                """\
                * fixtures defined from test_show_fixtures_indented_in_class *
                fixture1
                    line1
                    line2
                        indented line
                """
            )
        )

    def test_show_fixtures_different_files(self, testdir):
        """
        #833: --fixtures only shows fixtures from first file
        """
        testdir.makepyfile(
            test_a='''
            import pytest

            @pytest.fixture
            def fix_a():
                """Fixture A"""
                pass

            def test_a(fix_a):
                pass
        '''
        )
        testdir.makepyfile(
            test_b='''
            import pytest

            @pytest.fixture
            def fix_b():
                """Fixture B"""
                pass

            def test_b(fix_b):
                pass
        '''
        )
        result = testdir.runpytest("--fixtures")
        result.stdout.fnmatch_lines(
            """
            * fixtures defined from test_a *
            fix_a
                Fixture A

            * fixtures defined from test_b *
            fix_b
                Fixture B
        """
        )

    def test_show_fixtures_with_same_name(self, testdir):
        testdir.makeconftest(
            '''
            import pytest
            @pytest.fixture
            def arg1():
                """Hello World in conftest.py"""
                return "Hello World"
        '''
        )
        testdir.makepyfile(
            """
            def test_foo(arg1):
                assert arg1 == "Hello World"
        """
        )
        testdir.makepyfile(
            '''
            import pytest
            @pytest.fixture
            def arg1():
                """Hi from test module"""
                return "Hi"
            def test_bar(arg1):
                assert arg1 == "Hi"
        '''
        )
        result = testdir.runpytest("--fixtures")
        result.stdout.fnmatch_lines(
            """
            * fixtures defined from conftest *
            arg1
                Hello World in conftest.py

            * fixtures defined from test_show_fixtures_with_same_name *
            arg1
                Hi from test module
        """
        )

    def test_fixture_disallow_twice(self):
        """Test that applying @pytest.fixture twice generates an error (#2334)."""
        with pytest.raises(ValueError):

            @pytest.fixture
            @pytest.fixture
            def foo():
                pass


class TestContextManagerFixtureFuncs(object):
    @pytest.fixture(params=["fixture", "yield_fixture"])
    def flavor(self, request, testdir, monkeypatch):
        monkeypatch.setenv("PYTEST_FIXTURE_FLAVOR", request.param)
        testdir.makepyfile(
            test_context="""
            import os
            import pytest
            import warnings
            VAR = "PYTEST_FIXTURE_FLAVOR"
            if VAR not in os.environ:
                warnings.warn("PYTEST_FIXTURE_FLAVOR was not set, assuming fixture")
                fixture = pytest.fixture
            else:
                fixture = getattr(pytest, os.environ[VAR])
        """
        )

    def test_simple(self, testdir, flavor):
        testdir.makepyfile(
            """
            from __future__ import print_function
            from test_context import fixture
            @fixture
            def arg1():
                print("setup")
                yield 1
                print("teardown")
            def test_1(arg1):
                print("test1", arg1)
            def test_2(arg1):
                print("test2", arg1)
                assert 0
        """
        )
        result = testdir.runpytest("-s")
        result.stdout.fnmatch_lines(
            """
            *setup*
            *test1 1*
            *teardown*
            *setup*
            *test2 1*
            *teardown*
        """
        )

    def test_scoped(self, testdir, flavor):
        testdir.makepyfile(
            """
            from __future__ import print_function
            from test_context import fixture
            @fixture(scope="module")
            def arg1():
                print("setup")
                yield 1
                print("teardown")
            def test_1(arg1):
                print("test1", arg1)
            def test_2(arg1):
                print("test2", arg1)
        """
        )
        result = testdir.runpytest("-s")
        result.stdout.fnmatch_lines(
            """
            *setup*
            *test1 1*
            *test2 1*
            *teardown*
        """
        )

    def test_setup_exception(self, testdir, flavor):
        testdir.makepyfile(
            """
            from test_context import fixture
            @fixture(scope="module")
            def arg1():
                pytest.fail("setup")
                yield 1
            def test_1(arg1):
                pass
        """
        )
        result = testdir.runpytest("-s")
        result.stdout.fnmatch_lines(
            """
            *pytest.fail*setup*
            *1 error*
        """
        )

    def test_teardown_exception(self, testdir, flavor):
        testdir.makepyfile(
            """
            from test_context import fixture
            @fixture(scope="module")
            def arg1():
                yield 1
                pytest.fail("teardown")
            def test_1(arg1):
                pass
        """
        )
        result = testdir.runpytest("-s")
        result.stdout.fnmatch_lines(
            """
            *pytest.fail*teardown*
            *1 passed*1 error*
        """
        )

    def test_yields_more_than_one(self, testdir, flavor):
        testdir.makepyfile(
            """
            from test_context import fixture
            @fixture(scope="module")
            def arg1():
                yield 1
                yield 2
            def test_1(arg1):
                pass
        """
        )
        result = testdir.runpytest("-s")
        result.stdout.fnmatch_lines(
            """
            *fixture function*
            *test_yields*:2*
        """
        )

    def test_custom_name(self, testdir, flavor):
        testdir.makepyfile(
            """
            from test_context import fixture
            @fixture(name='meow')
            def arg1():
                return 'mew'
            def test_1(meow):
                print(meow)
        """
        )
        result = testdir.runpytest("-s")
        result.stdout.fnmatch_lines(["*mew*"])


class TestParameterizedSubRequest(object):
    def test_call_from_fixture(self, testdir):
        testdir.makepyfile(
            test_call_from_fixture="""
            import pytest

            @pytest.fixture(params=[0, 1, 2])
            def fix_with_param(request):
                return request.param

            @pytest.fixture
            def get_named_fixture(request):
                return request.getfixturevalue('fix_with_param')

            def test_foo(request, get_named_fixture):
                pass
            """
        )
        result = testdir.runpytest()
        result.stdout.fnmatch_lines(
            [
                "The requested fixture has no parameter defined for test:",
                "    test_call_from_fixture.py::test_foo",
                "Requested fixture 'fix_with_param' defined in:",
                "test_call_from_fixture.py:4",
                "Requested here:",
                "test_call_from_fixture.py:9",
                "*1 error in*",
            ]
        )

    def test_call_from_test(self, testdir):
        testdir.makepyfile(
            test_call_from_test="""
            import pytest

            @pytest.fixture(params=[0, 1, 2])
            def fix_with_param(request):
                return request.param

            def test_foo(request):
                request.getfixturevalue('fix_with_param')
            """
        )
        result = testdir.runpytest()
        result.stdout.fnmatch_lines(
            [
                "The requested fixture has no parameter defined for test:",
                "    test_call_from_test.py::test_foo",
                "Requested fixture 'fix_with_param' defined in:",
                "test_call_from_test.py:4",
                "Requested here:",
                "test_call_from_test.py:8",
                "*1 failed*",
            ]
        )

    def test_external_fixture(self, testdir):
        testdir.makeconftest(
            """
            import pytest

            @pytest.fixture(params=[0, 1, 2])
            def fix_with_param(request):
                return request.param
            """
        )

        testdir.makepyfile(
            test_external_fixture="""
            def test_foo(request):
                request.getfixturevalue('fix_with_param')
            """
        )
        result = testdir.runpytest()
        result.stdout.fnmatch_lines(
            [
                "The requested fixture has no parameter defined for test:",
                "    test_external_fixture.py::test_foo",
                "",
                "Requested fixture 'fix_with_param' defined in:",
                "conftest.py:4",
                "Requested here:",
                "test_external_fixture.py:2",
                "*1 failed*",
            ]
        )

    def test_non_relative_path(self, testdir):
        tests_dir = testdir.mkdir("tests")
        fixdir = testdir.mkdir("fixtures")
        fixfile = fixdir.join("fix.py")
        fixfile.write(
            textwrap.dedent(
                """\
                import pytest

                @pytest.fixture(params=[0, 1, 2])
                def fix_with_param(request):
                    return request.param
                """
            )
        )

        testfile = tests_dir.join("test_foos.py")
        testfile.write(
            textwrap.dedent(
                """\
                from fix import fix_with_param

                def test_foo(request):
                    request.getfixturevalue('fix_with_param')
                """
            )
        )

        tests_dir.chdir()
        testdir.syspathinsert(fixdir)
        result = testdir.runpytest()
        result.stdout.fnmatch_lines(
            [
                "The requested fixture has no parameter defined for test:",
                "    test_foos.py::test_foo",
                "",
                "Requested fixture 'fix_with_param' defined in:",
                "*fix.py:4",
                "Requested here:",
                "test_foos.py:4",
                "*1 failed*",
            ]
        )


def test_pytest_fixture_setup_and_post_finalizer_hook(testdir):
    testdir.makeconftest(
        """
        from __future__ import print_function
        def pytest_fixture_setup(fixturedef, request):
            print('ROOT setup hook called for {0} from {1}'.format(fixturedef.argname, request.node.name))
        def pytest_fixture_post_finalizer(fixturedef, request):
            print('ROOT finalizer hook called for {0} from {1}'.format(fixturedef.argname, request.node.name))
    """
    )
    testdir.makepyfile(
        **{
            "tests/conftest.py": """
            from __future__ import print_function
            def pytest_fixture_setup(fixturedef, request):
                print('TESTS setup hook called for {0} from {1}'.format(fixturedef.argname, request.node.name))
            def pytest_fixture_post_finalizer(fixturedef, request):
                print('TESTS finalizer hook called for {0} from {1}'.format(fixturedef.argname, request.node.name))
        """,
            "tests/test_hooks.py": """
            from __future__ import print_function
            import pytest

            @pytest.fixture()
            def my_fixture():
                return 'some'

            def test_func(my_fixture):
                print('TEST test_func')
                assert my_fixture == 'some'
        """,
        }
    )
    result = testdir.runpytest("-s")
    assert result.ret == 0
    result.stdout.fnmatch_lines(
        [
            "*TESTS setup hook called for my_fixture from test_func*",
            "*ROOT setup hook called for my_fixture from test_func*",
            "*TEST test_func*",
            "*TESTS finalizer hook called for my_fixture from test_func*",
            "*ROOT finalizer hook called for my_fixture from test_func*",
        ]
    )


class TestScopeOrdering(object):
    """Class of tests that ensure fixtures are ordered based on their scopes (#2405)"""

    @pytest.mark.parametrize("variant", ["mark", "autouse"])
    def test_func_closure_module_auto(self, testdir, variant, monkeypatch):
        """Semantically identical to the example posted in #2405 when ``use_mark=True``"""
        monkeypatch.setenv("FIXTURE_ACTIVATION_VARIANT", variant)
        testdir.makepyfile(
            """
            import warnings
            import os
            import pytest
            VAR = 'FIXTURE_ACTIVATION_VARIANT'
            VALID_VARS = ('autouse', 'mark')

            VARIANT = os.environ.get(VAR)
            if VARIANT is None or VARIANT not in VALID_VARS:
                warnings.warn("{!r} is not  in {}, assuming autouse".format(VARIANT, VALID_VARS) )
                variant = 'mark'

            @pytest.fixture(scope='module', autouse=VARIANT == 'autouse')
            def m1(): pass

            if VARIANT=='mark':
                pytestmark = pytest.mark.usefixtures('m1')

            @pytest.fixture(scope='function', autouse=True)
            def f1(): pass

            def test_func(m1):
                pass
        """
        )
        items, _ = testdir.inline_genitems()
        request = FixtureRequest(items[0])
        assert request.fixturenames == "m1 f1".split()

    def test_func_closure_with_native_fixtures(self, testdir, monkeypatch):
        """Sanity check that verifies the order returned by the closures and the actual fixture execution order:
        The execution order may differ because of fixture inter-dependencies.
        """
        monkeypatch.setattr(pytest, "FIXTURE_ORDER", [], raising=False)
        testdir.makepyfile(
            """
            import pytest

            FIXTURE_ORDER = pytest.FIXTURE_ORDER

            @pytest.fixture(scope="session")
            def s1():
                FIXTURE_ORDER.append('s1')

            @pytest.fixture(scope="package")
            def p1():
                FIXTURE_ORDER.append('p1')

            @pytest.fixture(scope="module")
            def m1():
                FIXTURE_ORDER.append('m1')

            @pytest.fixture(scope='session')
            def my_tmpdir_factory():
                FIXTURE_ORDER.append('my_tmpdir_factory')

            @pytest.fixture
            def my_tmpdir(my_tmpdir_factory):
                FIXTURE_ORDER.append('my_tmpdir')

            @pytest.fixture
            def f1(my_tmpdir):
                FIXTURE_ORDER.append('f1')

            @pytest.fixture
            def f2():
                FIXTURE_ORDER.append('f2')

            def test_foo(f1, p1, m1, f2, s1): pass
        """
        )
        items, _ = testdir.inline_genitems()
        request = FixtureRequest(items[0])
        # order of fixtures based on their scope and position in the parameter list
        assert (
            request.fixturenames == "s1 my_tmpdir_factory p1 m1 f1 f2 my_tmpdir".split()
        )
        testdir.runpytest()
        # actual fixture execution differs: dependent fixtures must be created first ("my_tmpdir")
        assert (
            pytest.FIXTURE_ORDER == "s1 my_tmpdir_factory p1 m1 my_tmpdir f1 f2".split()
        )

    def test_func_closure_module(self, testdir):
        testdir.makepyfile(
            """
            import pytest

            @pytest.fixture(scope='module')
            def m1(): pass

            @pytest.fixture(scope='function')
            def f1(): pass

            def test_func(f1, m1):
                pass
        """
        )
        items, _ = testdir.inline_genitems()
        request = FixtureRequest(items[0])
        assert request.fixturenames == "m1 f1".split()

    def test_func_closure_scopes_reordered(self, testdir):
        """Test ensures that fixtures are ordered by scope regardless of the order of the parameters, although
        fixtures of same scope keep the declared order
        """
        testdir.makepyfile(
            """
            import pytest

            @pytest.fixture(scope='session')
            def s1(): pass

            @pytest.fixture(scope='module')
            def m1(): pass

            @pytest.fixture(scope='function')
            def f1(): pass

            @pytest.fixture(scope='function')
            def f2(): pass

            class Test:

                @pytest.fixture(scope='class')
                def c1(cls): pass

                def test_func(self, f2, f1, c1, m1, s1):
                    pass
        """
        )
        items, _ = testdir.inline_genitems()
        request = FixtureRequest(items[0])
        assert request.fixturenames == "s1 m1 c1 f2 f1".split()

    def test_func_closure_same_scope_closer_root_first(self, testdir):
        """Auto-use fixtures of same scope are ordered by closer-to-root first"""
        testdir.makeconftest(
            """
            import pytest

            @pytest.fixture(scope='module', autouse=True)
            def m_conf(): pass
        """
        )
        testdir.makepyfile(
            **{
                "sub/conftest.py": """
                import pytest

                @pytest.fixture(scope='package', autouse=True)
                def p_sub(): pass

                @pytest.fixture(scope='module', autouse=True)
                def m_sub(): pass
            """,
                "sub/__init__.py": "",
                "sub/test_func.py": """
                import pytest

                @pytest.fixture(scope='module', autouse=True)
                def m_test(): pass

                @pytest.fixture(scope='function')
                def f1(): pass

                def test_func(m_test, f1):
                    pass
        """,
            }
        )
        items, _ = testdir.inline_genitems()
        request = FixtureRequest(items[0])
        assert request.fixturenames == "p_sub m_conf m_sub m_test f1".split()

    def test_func_closure_all_scopes_complex(self, testdir):
        """Complex test involving all scopes and mixing autouse with normal fixtures"""
        testdir.makeconftest(
            """
            import pytest

            @pytest.fixture(scope='session')
            def s1(): pass

            @pytest.fixture(scope='package', autouse=True)
            def p1(): pass
        """
        )
        testdir.makepyfile(**{"__init__.py": ""})
        testdir.makepyfile(
            """
            import pytest

            @pytest.fixture(scope='module', autouse=True)
            def m1(): pass

            @pytest.fixture(scope='module')
            def m2(s1): pass

            @pytest.fixture(scope='function')
            def f1(): pass

            @pytest.fixture(scope='function')
            def f2(): pass

            class Test:

                @pytest.fixture(scope='class', autouse=True)
                def c1(self):
                    pass

                def test_func(self, f2, f1, m2):
                    pass
        """
        )
        items, _ = testdir.inline_genitems()
        request = FixtureRequest(items[0])
        assert request.fixturenames == "s1 p1 m1 m2 c1 f2 f1".split()

    def test_multiple_packages(self, testdir):
        """Complex test involving multiple package fixtures. Make sure teardowns
        are executed in order.
        .
        └── root
            ├── __init__.py
            ├── sub1
            │   ├── __init__.py
            │   ├── conftest.py
            │   └── test_1.py
            └── sub2
                ├── __init__.py
                ├── conftest.py
                └── test_2.py
        """
        root = testdir.mkdir("root")
        root.join("__init__.py").write("values = []")
        sub1 = root.mkdir("sub1")
        sub1.ensure("__init__.py")
        sub1.join("conftest.py").write(
            textwrap.dedent(
                """\
            import pytest
            from .. import values
            @pytest.fixture(scope="package")
            def fix():
                values.append("pre-sub1")
                yield values
                assert values.pop() == "pre-sub1"
        """
            )
        )
        sub1.join("test_1.py").write(
            textwrap.dedent(
                """\
            from .. import values
            def test_1(fix):
                assert values == ["pre-sub1"]
        """
            )
        )
        sub2 = root.mkdir("sub2")
        sub2.ensure("__init__.py")
        sub2.join("conftest.py").write(
            textwrap.dedent(
                """\
            import pytest
            from .. import values
            @pytest.fixture(scope="package")
            def fix():
                values.append("pre-sub2")
                yield values
                assert values.pop() == "pre-sub2"
        """
            )
        )
        sub2.join("test_2.py").write(
            textwrap.dedent(
                """\
            from .. import values
            def test_2(fix):
                assert values == ["pre-sub2"]
        """
            )
        )
        reprec = testdir.inline_run()
        reprec.assertoutcome(passed=2)


def test_call_fixture_function_error():
    """Check if an error is raised if a fixture function is called directly (#4545)"""

    @pytest.fixture
    def fix():
        return 1

    with pytest.raises(pytest.fail.Exception):
        assert fix() == 1
```
### 9 - testing/python/fixtures.py:

Start line: 2182, End line: 3108

```python
class TestFixtureMarker(object):

    def test_scope_mismatch_various(self, testdir):
        testdir.makeconftest(
            """
            import pytest
            finalized = []
            created = []
            @pytest.fixture(scope="function")
            def arg(request):
                pass
        """
        )
        testdir.makepyfile(
            test_mod1="""
                import pytest
                @pytest.fixture(scope="session")
                def arg(request):
                    request.getfixturevalue("arg")
                def test_1(arg):
                    pass
            """
        )
        result = testdir.runpytest(SHOW_PYTEST_WARNINGS_ARG)
        assert result.ret != 0
        result.stdout.fnmatch_lines(
            ["*ScopeMismatch*You tried*function*session*request*"]
        )

    def test_register_only_with_mark(self, testdir):
        testdir.makeconftest(
            """
            import pytest
            @pytest.fixture()
            def arg():
                return 1
        """
        )
        testdir.makepyfile(
            test_mod1="""
                import pytest
                @pytest.fixture()
                def arg(arg):
                    return arg + 1
                def test_1(arg):
                    assert arg == 2
            """
        )
        reprec = testdir.inline_run()
        reprec.assertoutcome(passed=1)

    def test_parametrize_and_scope(self, testdir):
        testdir.makepyfile(
            """
            import pytest
            @pytest.fixture(scope="module", params=["a", "b", "c"])
            def arg(request):
                return request.param
            values = []
            def test_param(arg):
                values.append(arg)
        """
        )
        reprec = testdir.inline_run("-v")
        reprec.assertoutcome(passed=3)
        values = reprec.getcalls("pytest_runtest_call")[0].item.module.values
        assert len(values) == 3
        assert "a" in values
        assert "b" in values
        assert "c" in values

    def test_scope_mismatch(self, testdir):
        testdir.makeconftest(
            """
            import pytest
            @pytest.fixture(scope="function")
            def arg(request):
                pass
        """
        )
        testdir.makepyfile(
            """
            import pytest
            @pytest.fixture(scope="session")
            def arg(arg):
                pass
            def test_mismatch(arg):
                pass
        """
        )
        result = testdir.runpytest()
        result.stdout.fnmatch_lines(["*ScopeMismatch*", "*1 error*"])

    def test_parametrize_separated_order(self, testdir):
        testdir.makepyfile(
            """
            import pytest

            @pytest.fixture(scope="module", params=[1, 2])
            def arg(request):
                return request.param

            values = []
            def test_1(arg):
                values.append(arg)
            def test_2(arg):
                values.append(arg)
        """
        )
        reprec = testdir.inline_run("-v")
        reprec.assertoutcome(passed=4)
        values = reprec.getcalls("pytest_runtest_call")[0].item.module.values
        assert values == [1, 1, 2, 2]

    def test_module_parametrized_ordering(self, testdir):
        testdir.makeini(
            """
            [pytest]
            console_output_style=classic
        """
        )
        testdir.makeconftest(
            """
            import pytest

            @pytest.fixture(scope="session", params="s1 s2".split())
            def sarg():
                pass
            @pytest.fixture(scope="module", params="m1 m2".split())
            def marg():
                pass
        """
        )
        testdir.makepyfile(
            test_mod1="""
            def test_func(sarg):
                pass
            def test_func1(marg):
                pass
        """,
            test_mod2="""
            def test_func2(sarg):
                pass
            def test_func3(sarg, marg):
                pass
            def test_func3b(sarg, marg):
                pass
            def test_func4(marg):
                pass
        """,
        )
        result = testdir.runpytest("-v")
        result.stdout.fnmatch_lines(
            """
            test_mod1.py::test_func[s1] PASSED
            test_mod2.py::test_func2[s1] PASSED
            test_mod2.py::test_func3[s1-m1] PASSED
            test_mod2.py::test_func3b[s1-m1] PASSED
            test_mod2.py::test_func3[s1-m2] PASSED
            test_mod2.py::test_func3b[s1-m2] PASSED
            test_mod1.py::test_func[s2] PASSED
            test_mod2.py::test_func2[s2] PASSED
            test_mod2.py::test_func3[s2-m1] PASSED
            test_mod2.py::test_func3b[s2-m1] PASSED
            test_mod2.py::test_func4[m1] PASSED
            test_mod2.py::test_func3[s2-m2] PASSED
            test_mod2.py::test_func3b[s2-m2] PASSED
            test_mod2.py::test_func4[m2] PASSED
            test_mod1.py::test_func1[m1] PASSED
            test_mod1.py::test_func1[m2] PASSED
        """
        )

    def test_dynamic_parametrized_ordering(self, testdir):
        testdir.makeini(
            """
            [pytest]
            console_output_style=classic
        """
        )
        testdir.makeconftest(
            """
            import pytest

            def pytest_configure(config):
                class DynamicFixturePlugin(object):
                    @pytest.fixture(scope='session', params=['flavor1', 'flavor2'])
                    def flavor(self, request):
                        return request.param
                config.pluginmanager.register(DynamicFixturePlugin(), 'flavor-fixture')

            @pytest.fixture(scope='session', params=['vxlan', 'vlan'])
            def encap(request):
                return request.param

            @pytest.fixture(scope='session', autouse='True')
            def reprovision(request, flavor, encap):
                pass
        """
        )
        testdir.makepyfile(
            """
            def test(reprovision):
                pass
            def test2(reprovision):
                pass
        """
        )
        result = testdir.runpytest("-v")
        result.stdout.fnmatch_lines(
            """
            test_dynamic_parametrized_ordering.py::test[flavor1-vxlan] PASSED
            test_dynamic_parametrized_ordering.py::test2[flavor1-vxlan] PASSED
            test_dynamic_parametrized_ordering.py::test[flavor2-vxlan] PASSED
            test_dynamic_parametrized_ordering.py::test2[flavor2-vxlan] PASSED
            test_dynamic_parametrized_ordering.py::test[flavor2-vlan] PASSED
            test_dynamic_parametrized_ordering.py::test2[flavor2-vlan] PASSED
            test_dynamic_parametrized_ordering.py::test[flavor1-vlan] PASSED
            test_dynamic_parametrized_ordering.py::test2[flavor1-vlan] PASSED
        """
        )

    def test_class_ordering(self, testdir):
        testdir.makeini(
            """
            [pytest]
            console_output_style=classic
        """
        )
        testdir.makeconftest(
            """
            import pytest

            values = []

            @pytest.fixture(scope="function", params=[1,2])
            def farg(request):
                return request.param

            @pytest.fixture(scope="class", params=list("ab"))
            def carg(request):
                return request.param

            @pytest.fixture(scope="function", autouse=True)
            def append(request, farg, carg):
                def fin():
                    values.append("fin_%s%s" % (carg, farg))
                request.addfinalizer(fin)
        """
        )
        testdir.makepyfile(
            """
            import pytest

            class TestClass2(object):
                def test_1(self):
                    pass
                def test_2(self):
                    pass
            class TestClass(object):
                def test_3(self):
                    pass
        """
        )
        result = testdir.runpytest("-vs")
        result.stdout.re_match_lines(
            r"""
            test_class_ordering.py::TestClass2::test_1\[a-1\] PASSED
            test_class_ordering.py::TestClass2::test_1\[a-2\] PASSED
            test_class_ordering.py::TestClass2::test_2\[a-1\] PASSED
            test_class_ordering.py::TestClass2::test_2\[a-2\] PASSED
            test_class_ordering.py::TestClass2::test_1\[b-1\] PASSED
            test_class_ordering.py::TestClass2::test_1\[b-2\] PASSED
            test_class_ordering.py::TestClass2::test_2\[b-1\] PASSED
            test_class_ordering.py::TestClass2::test_2\[b-2\] PASSED
            test_class_ordering.py::TestClass::test_3\[a-1\] PASSED
            test_class_ordering.py::TestClass::test_3\[a-2\] PASSED
            test_class_ordering.py::TestClass::test_3\[b-1\] PASSED
            test_class_ordering.py::TestClass::test_3\[b-2\] PASSED
        """
        )

    def test_parametrize_separated_order_higher_scope_first(self, testdir):
        testdir.makepyfile(
            """
            import pytest

            @pytest.fixture(scope="function", params=[1, 2])
            def arg(request):
                param = request.param
                request.addfinalizer(lambda: values.append("fin:%s" % param))
                values.append("create:%s" % param)
                return request.param

            @pytest.fixture(scope="module", params=["mod1", "mod2"])
            def modarg(request):
                param = request.param
                request.addfinalizer(lambda: values.append("fin:%s" % param))
                values.append("create:%s" % param)
                return request.param

            values = []
            def test_1(arg):
                values.append("test1")
            def test_2(modarg):
                values.append("test2")
            def test_3(arg, modarg):
                values.append("test3")
            def test_4(modarg, arg):
                values.append("test4")
        """
        )
        reprec = testdir.inline_run("-v")
        reprec.assertoutcome(passed=12)
        values = reprec.getcalls("pytest_runtest_call")[0].item.module.values
        expected = [
            "create:1",
            "test1",
            "fin:1",
            "create:2",
            "test1",
            "fin:2",
            "create:mod1",
            "test2",
            "create:1",
            "test3",
            "fin:1",
            "create:2",
            "test3",
            "fin:2",
            "create:1",
            "test4",
            "fin:1",
            "create:2",
            "test4",
            "fin:2",
            "fin:mod1",
            "create:mod2",
            "test2",
            "create:1",
            "test3",
            "fin:1",
            "create:2",
            "test3",
            "fin:2",
            "create:1",
            "test4",
            "fin:1",
            "create:2",
            "test4",
            "fin:2",
            "fin:mod2",
        ]
        import pprint

        pprint.pprint(list(zip(values, expected)))
        assert values == expected

    def test_parametrized_fixture_teardown_order(self, testdir):
        testdir.makepyfile(
            """
            import pytest
            @pytest.fixture(params=[1,2], scope="class")
            def param1(request):
                return request.param

            values = []

            class TestClass(object):
                @classmethod
                @pytest.fixture(scope="class", autouse=True)
                def setup1(self, request, param1):
                    values.append(1)
                    request.addfinalizer(self.teardown1)
                @classmethod
                def teardown1(self):
                    assert values.pop() == 1
                @pytest.fixture(scope="class", autouse=True)
                def setup2(self, request, param1):
                    values.append(2)
                    request.addfinalizer(self.teardown2)
                @classmethod
                def teardown2(self):
                    assert values.pop() == 2
                def test(self):
                    pass

            def test_finish():
                assert not values
        """
        )
        result = testdir.runpytest("-v")
        result.stdout.fnmatch_lines(
            """
            *3 passed*
        """
        )
        assert "error" not in result.stdout.str()

    def test_fixture_finalizer(self, testdir):
        testdir.makeconftest(
            """
            import pytest
            import sys

            @pytest.fixture
            def browser(request):

                def finalize():
                    sys.stdout.write('Finalized')
                request.addfinalizer(finalize)
                return {}
        """
        )
        b = testdir.mkdir("subdir")
        b.join("test_overridden_fixture_finalizer.py").write(
            textwrap.dedent(
                """\
                import pytest
                @pytest.fixture
                def browser(browser):
                    browser['visited'] = True
                    return browser

                def test_browser(browser):
                    assert browser['visited'] is True
                """
            )
        )
        reprec = testdir.runpytest("-s")
        for test in ["test_browser"]:
            reprec.stdout.fnmatch_lines(["*Finalized*"])

    def test_class_scope_with_normal_tests(self, testdir):
        testpath = testdir.makepyfile(
            """
            import pytest

            class Box(object):
                value = 0

            @pytest.fixture(scope='class')
            def a(request):
                Box.value += 1
                return Box.value

            def test_a(a):
                assert a == 1

            class Test1(object):
                def test_b(self, a):
                    assert a == 2

            class Test2(object):
                def test_c(self, a):
                    assert a == 3"""
        )
        reprec = testdir.inline_run(testpath)
        for test in ["test_a", "test_b", "test_c"]:
            assert reprec.matchreport(test).passed

    def test_request_is_clean(self, testdir):
        testdir.makepyfile(
            """
            import pytest
            values = []
            @pytest.fixture(params=[1, 2])
            def fix(request):
                request.addfinalizer(lambda: values.append(request.param))
            def test_fix(fix):
                pass
        """
        )
        reprec = testdir.inline_run("-s")
        values = reprec.getcalls("pytest_runtest_call")[0].item.module.values
        assert values == [1, 2]

    def test_parametrize_separated_lifecycle(self, testdir):
        testdir.makepyfile(
            """
            import pytest

            values = []
            @pytest.fixture(scope="module", params=[1, 2])
            def arg(request):
                x = request.param
                request.addfinalizer(lambda: values.append("fin%s" % x))
                return request.param
            def test_1(arg):
                values.append(arg)
            def test_2(arg):
                values.append(arg)
        """
        )
        reprec = testdir.inline_run("-vs")
        reprec.assertoutcome(passed=4)
        values = reprec.getcalls("pytest_runtest_call")[0].item.module.values
        import pprint

        pprint.pprint(values)
        # assert len(values) == 6
        assert values[0] == values[1] == 1
        assert values[2] == "fin1"
        assert values[3] == values[4] == 2
        assert values[5] == "fin2"

    def test_parametrize_function_scoped_finalizers_called(self, testdir):
        testdir.makepyfile(
            """
            import pytest

            @pytest.fixture(scope="function", params=[1, 2])
            def arg(request):
                x = request.param
                request.addfinalizer(lambda: values.append("fin%s" % x))
                return request.param

            values = []
            def test_1(arg):
                values.append(arg)
            def test_2(arg):
                values.append(arg)
            def test_3():
                assert len(values) == 8
                assert values == [1, "fin1", 2, "fin2", 1, "fin1", 2, "fin2"]
        """
        )
        reprec = testdir.inline_run("-v")
        reprec.assertoutcome(passed=5)

    @pytest.mark.parametrize("scope", ["session", "function", "module"])
    def test_finalizer_order_on_parametrization(self, scope, testdir):
        """#246"""
        testdir.makepyfile(
            """
            import pytest
            values = []

            @pytest.fixture(scope=%(scope)r, params=["1"])
            def fix1(request):
                return request.param

            @pytest.fixture(scope=%(scope)r)
            def fix2(request, base):
                def cleanup_fix2():
                    assert not values, "base should not have been finalized"
                request.addfinalizer(cleanup_fix2)

            @pytest.fixture(scope=%(scope)r)
            def base(request, fix1):
                def cleanup_base():
                    values.append("fin_base")
                    print("finalizing base")
                request.addfinalizer(cleanup_base)

            def test_begin():
                pass
            def test_baz(base, fix2):
                pass
            def test_other():
                pass
        """
            % {"scope": scope}
        )
        reprec = testdir.inline_run("-lvs")
        reprec.assertoutcome(passed=3)

    def test_class_scope_parametrization_ordering(self, testdir):
        """#396"""
        testdir.makepyfile(
            """
            import pytest
            values = []
            @pytest.fixture(params=["John", "Doe"], scope="class")
            def human(request):
                request.addfinalizer(lambda: values.append("fin %s" % request.param))
                return request.param

            class TestGreetings(object):
                def test_hello(self, human):
                    values.append("test_hello")

            class TestMetrics(object):
                def test_name(self, human):
                    values.append("test_name")

                def test_population(self, human):
                    values.append("test_population")
        """
        )
        reprec = testdir.inline_run()
        reprec.assertoutcome(passed=6)
        values = reprec.getcalls("pytest_runtest_call")[0].item.module.values
        assert values == [
            "test_hello",
            "fin John",
            "test_hello",
            "fin Doe",
            "test_name",
            "test_population",
            "fin John",
            "test_name",
            "test_population",
            "fin Doe",
        ]

    def test_parametrize_setup_function(self, testdir):
        testdir.makepyfile(
            """
            import pytest

            @pytest.fixture(scope="module", params=[1, 2])
            def arg(request):
                return request.param

            @pytest.fixture(scope="module", autouse=True)
            def mysetup(request, arg):
                request.addfinalizer(lambda: values.append("fin%s" % arg))
                values.append("setup%s" % arg)

            values = []
            def test_1(arg):
                values.append(arg)
            def test_2(arg):
                values.append(arg)
            def test_3():
                import pprint
                pprint.pprint(values)
                if arg == 1:
                    assert values == ["setup1", 1, 1, ]
                elif arg == 2:
                    assert values == ["setup1", 1, 1, "fin1",
                                 "setup2", 2, 2, ]

        """
        )
        reprec = testdir.inline_run("-v")
        reprec.assertoutcome(passed=6)

    def test_fixture_marked_function_not_collected_as_test(self, testdir):
        testdir.makepyfile(
            """
            import pytest
            @pytest.fixture
            def test_app():
                return 1

            def test_something(test_app):
                assert test_app == 1
        """
        )
        reprec = testdir.inline_run()
        reprec.assertoutcome(passed=1)

    def test_params_and_ids(self, testdir):
        testdir.makepyfile(
            """
            import pytest

            @pytest.fixture(params=[object(), object()],
                            ids=['alpha', 'beta'])
            def fix(request):
                return request.param

            def test_foo(fix):
                assert 1
        """
        )
        res = testdir.runpytest("-v")
        res.stdout.fnmatch_lines(["*test_foo*alpha*", "*test_foo*beta*"])

    def test_params_and_ids_yieldfixture(self, testdir):
        testdir.makepyfile(
            """
            import pytest

            @pytest.yield_fixture(params=[object(), object()],
                                  ids=['alpha', 'beta'])
            def fix(request):
                 yield request.param

            def test_foo(fix):
                assert 1
        """
        )
        res = testdir.runpytest("-v")
        res.stdout.fnmatch_lines(["*test_foo*alpha*", "*test_foo*beta*"])

    def test_deterministic_fixture_collection(self, testdir, monkeypatch):
        """#920"""
        testdir.makepyfile(
            """
            import pytest

            @pytest.fixture(scope="module",
                            params=["A",
                                    "B",
                                    "C"])
            def A(request):
                return request.param

            @pytest.fixture(scope="module",
                            params=["DDDDDDDDD", "EEEEEEEEEEEE", "FFFFFFFFFFF", "banansda"])
            def B(request, A):
                return request.param

            def test_foo(B):
                # Something funky is going on here.
                # Despite specified seeds, on what is collected,
                # sometimes we get unexpected passes. hashing B seems
                # to help?
                assert hash(B) or True
            """
        )
        monkeypatch.setenv("PYTHONHASHSEED", "1")
        out1 = testdir.runpytest_subprocess("-v")
        monkeypatch.setenv("PYTHONHASHSEED", "2")
        out2 = testdir.runpytest_subprocess("-v")
        out1 = [
            line
            for line in out1.outlines
            if line.startswith("test_deterministic_fixture_collection.py::test_foo")
        ]
        out2 = [
            line
            for line in out2.outlines
            if line.startswith("test_deterministic_fixture_collection.py::test_foo")
        ]
        assert len(out1) == 12
        assert out1 == out2


class TestRequestScopeAccess(object):
    pytestmark = pytest.mark.parametrize(
        ("scope", "ok", "error"),
        [
            ["session", "", "fspath class function module"],
            ["module", "module fspath", "cls function"],
            ["class", "module fspath cls", "function"],
            ["function", "module fspath cls function", ""],
        ],
    )

    def test_setup(self, testdir, scope, ok, error):
        testdir.makepyfile(
            """
            import pytest
            @pytest.fixture(scope=%r, autouse=True)
            def myscoped(request):
                for x in %r:
                    assert hasattr(request, x)
                for x in %r:
                    pytest.raises(AttributeError, lambda:
                        getattr(request, x))
                assert request.session
                assert request.config
            def test_func():
                pass
        """
            % (scope, ok.split(), error.split())
        )
        reprec = testdir.inline_run("-l")
        reprec.assertoutcome(passed=1)

    def test_funcarg(self, testdir, scope, ok, error):
        testdir.makepyfile(
            """
            import pytest
            @pytest.fixture(scope=%r)
            def arg(request):
                for x in %r:
                    assert hasattr(request, x)
                for x in %r:
                    pytest.raises(AttributeError, lambda:
                        getattr(request, x))
                assert request.session
                assert request.config
            def test_func(arg):
                pass
        """
            % (scope, ok.split(), error.split())
        )
        reprec = testdir.inline_run()
        reprec.assertoutcome(passed=1)


class TestErrors(object):
    def test_subfactory_missing_funcarg(self, testdir):
        testdir.makepyfile(
            """
            import pytest
            @pytest.fixture()
            def gen(qwe123):
                return 1
            def test_something(gen):
                pass
        """
        )
        result = testdir.runpytest()
        assert result.ret != 0
        result.stdout.fnmatch_lines(
            ["*def gen(qwe123):*", "*fixture*qwe123*not found*", "*1 error*"]
        )

    def test_issue498_fixture_finalizer_failing(self, testdir):
        testdir.makepyfile(
            """
            import pytest
            @pytest.fixture
            def fix1(request):
                def f():
                    raise KeyError
                request.addfinalizer(f)
                return object()

            values = []
            def test_1(fix1):
                values.append(fix1)
            def test_2(fix1):
                values.append(fix1)
            def test_3():
                assert values[0] != values[1]
        """
        )
        result = testdir.runpytest()
        result.stdout.fnmatch_lines(
            """
            *ERROR*teardown*test_1*
            *KeyError*
            *ERROR*teardown*test_2*
            *KeyError*
            *3 pass*2 error*
        """
        )

    def test_setupfunc_missing_funcarg(self, testdir):
        testdir.makepyfile(
            """
            import pytest
            @pytest.fixture(autouse=True)
            def gen(qwe123):
                return 1
            def test_something():
                pass
        """
        )
        result = testdir.runpytest()
        assert result.ret != 0
        result.stdout.fnmatch_lines(
            ["*def gen(qwe123):*", "*fixture*qwe123*not found*", "*1 error*"]
        )


class TestShowFixtures(object):
    def test_funcarg_compat(self, testdir):
        config = testdir.parseconfigure("--funcargs")
        assert config.option.showfixtures

    def test_show_fixtures(self, testdir):
        result = testdir.runpytest("--fixtures")
        result.stdout.fnmatch_lines(
            [
                "tmpdir_factory [[]session scope[]]",
                "*for the test session*",
                "tmpdir",
                "*temporary directory*",
            ]
        )

    def test_show_fixtures_verbose(self, testdir):
        result = testdir.runpytest("--fixtures", "-v")
        result.stdout.fnmatch_lines(
            [
                "tmpdir_factory [[]session scope[]] -- *tmpdir.py*",
                "*for the test session*",
                "tmpdir -- *tmpdir.py*",
                "*temporary directory*",
            ]
        )

    def test_show_fixtures_testmodule(self, testdir):
        p = testdir.makepyfile(
            '''
            import pytest
            @pytest.fixture
            def _arg0():
                """ hidden """
            @pytest.fixture
            def arg1():
                """  hello world """
        '''
        )
        result = testdir.runpytest("--fixtures", p)
        result.stdout.fnmatch_lines(
            """
            *tmpdir
            *fixtures defined from*
            *arg1*
            *hello world*
        """
        )
        assert "arg0" not in result.stdout.str()

    @pytest.mark.parametrize("testmod", [True, False])
    def test_show_fixtures_conftest(self, testdir, testmod):
        testdir.makeconftest(
            '''
            import pytest
            @pytest.fixture
            def arg1():
                """  hello world """
        '''
        )
        if testmod:
            testdir.makepyfile(
                """
                def test_hello():
                    pass
            """
            )
        result = testdir.runpytest("--fixtures")
        result.stdout.fnmatch_lines(
            """
            *tmpdir*
            *fixtures defined from*conftest*
            *arg1*
            *hello world*
        """
        )
```
### 10 - testing/python/fixtures.py:

Start line: 2018, End line: 2180

```python
class TestFixtureMarker(object):

    def test_scope_session(self, testdir):
        testdir.makepyfile(
            """
            import pytest
            values = []
            @pytest.fixture(scope="module")
            def arg():
                values.append(1)
                return 1

            def test_1(arg):
                assert arg == 1
            def test_2(arg):
                assert arg == 1
                assert len(values) == 1
            class TestClass(object):
                def test3(self, arg):
                    assert arg == 1
                    assert len(values) == 1
        """
        )
        reprec = testdir.inline_run()
        reprec.assertoutcome(passed=3)

    def test_scope_session_exc(self, testdir):
        testdir.makepyfile(
            """
            import pytest
            values = []
            @pytest.fixture(scope="session")
            def fix():
                values.append(1)
                pytest.skip('skipping')

            def test_1(fix):
                pass
            def test_2(fix):
                pass
            def test_last():
                assert values == [1]
        """
        )
        reprec = testdir.inline_run()
        reprec.assertoutcome(skipped=2, passed=1)

    def test_scope_session_exc_two_fix(self, testdir):
        testdir.makepyfile(
            """
            import pytest
            values = []
            m = []
            @pytest.fixture(scope="session")
            def a():
                values.append(1)
                pytest.skip('skipping')
            @pytest.fixture(scope="session")
            def b(a):
                m.append(1)

            def test_1(b):
                pass
            def test_2(b):
                pass
            def test_last():
                assert values == [1]
                assert m == []
        """
        )
        reprec = testdir.inline_run()
        reprec.assertoutcome(skipped=2, passed=1)

    def test_scope_exc(self, testdir):
        testdir.makepyfile(
            test_foo="""
                def test_foo(fix):
                    pass
            """,
            test_bar="""
                def test_bar(fix):
                    pass
            """,
            conftest="""
                import pytest
                reqs = []
                @pytest.fixture(scope="session")
                def fix(request):
                    reqs.append(1)
                    pytest.skip()
                @pytest.fixture
                def req_list():
                    return reqs
            """,
            test_real="""
                def test_last(req_list):
                    assert req_list == [1]
            """,
        )
        reprec = testdir.inline_run()
        reprec.assertoutcome(skipped=2, passed=1)

    def test_scope_module_uses_session(self, testdir):
        testdir.makepyfile(
            """
            import pytest
            values = []
            @pytest.fixture(scope="module")
            def arg():
                values.append(1)
                return 1

            def test_1(arg):
                assert arg == 1
            def test_2(arg):
                assert arg == 1
                assert len(values) == 1
            class TestClass(object):
                def test3(self, arg):
                    assert arg == 1
                    assert len(values) == 1
        """
        )
        reprec = testdir.inline_run()
        reprec.assertoutcome(passed=3)

    def test_scope_module_and_finalizer(self, testdir):
        testdir.makeconftest(
            """
            import pytest
            finalized_list = []
            created_list = []
            @pytest.fixture(scope="module")
            def arg(request):
                created_list.append(1)
                assert request.scope == "module"
                request.addfinalizer(lambda: finalized_list.append(1))
            @pytest.fixture
            def created(request):
                return len(created_list)
            @pytest.fixture
            def finalized(request):
                return len(finalized_list)
        """
        )
        testdir.makepyfile(
            test_mod1="""
                def test_1(arg, created, finalized):
                    assert created == 1
                    assert finalized == 0
                def test_2(arg, created, finalized):
                    assert created == 1
                    assert finalized == 0""",
            test_mod2="""
                def test_3(arg, created, finalized):
                    assert created == 2
                    assert finalized == 1""",
            test_mode3="""
                def test_4(arg, created, finalized):
                    assert created == 3
                    assert finalized == 2
            """,
        )
        reprec = testdir.inline_run()
        reprec.assertoutcome(passed=4)
```
### 14 - src/_pytest/fixtures.py:

Start line: 1223, End line: 1255

```python
class FixtureManager(object):

    def pytest_generate_tests(self, metafunc):
        for argname in metafunc.fixturenames:
            faclist = metafunc._arg2fixturedefs.get(argname)
            if faclist:
                fixturedef = faclist[-1]
                if fixturedef.params is not None:
                    markers = list(metafunc.definition.iter_markers("parametrize"))
                    for parametrize_mark in markers:
                        if "argnames" in parametrize_mark.kwargs:
                            argnames = parametrize_mark.kwargs["argnames"]
                        else:
                            argnames = parametrize_mark.args[0]

                        if not isinstance(argnames, (tuple, list)):
                            argnames = [
                                x.strip() for x in argnames.split(",") if x.strip()
                            ]
                        if argname in argnames:
                            break
                    else:
                        metafunc.parametrize(
                            argname,
                            fixturedef.params,
                            indirect=True,
                            scope=fixturedef.scope,
                            ids=fixturedef.ids,
                        )
            else:
                continue  # will raise FixtureLookupError at setup time

    def pytest_collection_modifyitems(self, items):
        # separate parametrized setups
        items[:] = reorder_items(items)
```
### 16 - src/_pytest/fixtures.py:

Start line: 1, End line: 111

```python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import inspect
import itertools
import sys
import warnings
from collections import defaultdict
from collections import deque
from collections import OrderedDict

import attr
import py
import six

import _pytest
from _pytest import nodes
from _pytest._code.code import FormattedExcinfo
from _pytest._code.code import TerminalRepr
from _pytest.compat import _format_args
from _pytest.compat import _PytestWrapper
from _pytest.compat import exc_clear
from _pytest.compat import FuncargnamesCompatAttr
from _pytest.compat import get_real_func
from _pytest.compat import get_real_method
from _pytest.compat import getfslineno
from _pytest.compat import getfuncargnames
from _pytest.compat import getimfunc
from _pytest.compat import getlocation
from _pytest.compat import is_generator
from _pytest.compat import isclass
from _pytest.compat import NOTSET
from _pytest.compat import safe_getattr
from _pytest.deprecated import FIXTURE_FUNCTION_CALL
from _pytest.deprecated import FIXTURE_NAMED_REQUEST
from _pytest.outcomes import fail
from _pytest.outcomes import TEST_OUTCOME


@attr.s(frozen=True)
class PseudoFixtureDef(object):
    cached_result = attr.ib()
    scope = attr.ib()


def pytest_sessionstart(session):
    import _pytest.python
    import _pytest.nodes

    scopename2class.update(
        {
            "package": _pytest.python.Package,
            "class": _pytest.python.Class,
            "module": _pytest.python.Module,
            "function": _pytest.nodes.Item,
            "session": _pytest.main.Session,
        }
    )
    session._fixturemanager = FixtureManager(session)


scopename2class = {}


scope2props = dict(session=())
scope2props["package"] = ("fspath",)
scope2props["module"] = ("fspath", "module")
scope2props["class"] = scope2props["module"] + ("cls",)
scope2props["instance"] = scope2props["class"] + ("instance",)
scope2props["function"] = scope2props["instance"] + ("function", "keywords")


def scopeproperty(name=None, doc=None):
    def decoratescope(func):
        scopename = name or func.__name__

        def provide(self):
            if func.__name__ in scope2props[self.scope]:
                return func(self)
            raise AttributeError(
                "%s not available in %s-scoped context" % (scopename, self.scope)
            )

        return property(provide, None, None, func.__doc__)

    return decoratescope


def get_scope_package(node, fixturedef):
    import pytest

    cls = pytest.Package
    current = node
    fixture_package_name = "%s/%s" % (fixturedef.baseid, "__init__.py")
    while current and (
        type(current) is not cls or fixture_package_name != current.nodeid
    ):
        current = current.parent
    if current is None:
        return node.session
    return current


def get_scope_node(node, scope):
    cls = scopename2class.get(scope)
    if cls is None:
        raise ValueError("unknown scope")
    return node.getparent(cls)
```
### 20 - src/_pytest/fixtures.py:

Start line: 591, End line: 609

```python
class FixtureRequest(FuncargnamesCompatAttr):

    def _schedule_finalizers(self, fixturedef, subrequest):
        # if fixture function failed it might have registered finalizers
        self.session._setupstate.addfinalizer(
            functools.partial(fixturedef.finish, request=subrequest), subrequest.node
        )

    def _check_scope(self, argname, invoking_scope, requested_scope):
        if argname == "request":
            return
        if scopemismatch(invoking_scope, requested_scope):
            # try to report something helpful
            lines = self._factorytraceback()
            fail(
                "ScopeMismatch: You tried to access the %r scoped "
                "fixture %r with a %r scoped request object, "
                "involved factories\n%s"
                % ((requested_scope, argname, invoking_scope, "\n".join(lines))),
                pytrace=False,
            )
```
### 24 - src/_pytest/fixtures.py:

Start line: 488, End line: 515

```python
class FixtureRequest(FuncargnamesCompatAttr):

    def _get_active_fixturedef(self, argname):
        try:
            return self._fixture_defs[argname]
        except KeyError:
            try:
                fixturedef = self._getnextfixturedef(argname)
            except FixtureLookupError:
                if argname == "request":
                    cached_result = (self, [0], None)
                    scope = "function"
                    return PseudoFixtureDef(cached_result, scope)
                raise
        # remove indent to prevent the python3 exception
        # from leaking into the call
        self._compute_fixture_value(fixturedef)
        self._fixture_defs[argname] = fixturedef
        return fixturedef

    def _get_fixturestack(self):
        current = self
        values = []
        while 1:
            fixturedef = getattr(current, "_fixturedef", None)
            if fixturedef is None:
                values.reverse()
                return values
            values.append(fixturedef)
            current = current._parent_request
```
### 26 - src/_pytest/fixtures.py:

Start line: 699, End line: 750

```python
class FixtureLookupError(LookupError):
    """ could not return a requested Fixture (missing or invalid). """

    def __init__(self, argname, request, msg=None):
        self.argname = argname
        self.request = request
        self.fixturestack = request._get_fixturestack()
        self.msg = msg

    def formatrepr(self):
        tblines = []
        addline = tblines.append
        stack = [self.request._pyfuncitem.obj]
        stack.extend(map(lambda x: x.func, self.fixturestack))
        msg = self.msg
        if msg is not None:
            # the last fixture raise an error, let's present
            # it at the requesting side
            stack = stack[:-1]
        for function in stack:
            fspath, lineno = getfslineno(function)
            try:
                lines, _ = inspect.getsourcelines(get_real_func(function))
            except (IOError, IndexError, TypeError):
                error_msg = "file %s, line %s: source code not available"
                addline(error_msg % (fspath, lineno + 1))
            else:
                addline("file %s, line %s" % (fspath, lineno + 1))
                for i, line in enumerate(lines):
                    line = line.rstrip()
                    addline("  " + line)
                    if line.lstrip().startswith("def"):
                        break

        if msg is None:
            fm = self.request._fixturemanager
            available = set()
            parentid = self.request._pyfuncitem.parent.nodeid
            for name, fixturedefs in fm._arg2fixturedefs.items():
                faclist = list(fm._matchfactories(fixturedefs, parentid))
                if faclist:
                    available.add(name)
            if self.argname in available:
                msg = " recursive dependency involving fixture '{}' detected".format(
                    self.argname
                )
            else:
                msg = "fixture '{}' not found".format(self.argname)
            msg += "\n available fixtures: {}".format(", ".join(sorted(available)))
            msg += "\n use 'pytest --fixtures [testpath]' for help on them."

        return FixtureLookupErrorRepr(fspath, lineno, tblines, msg, self.argname)
```
### 29 - src/_pytest/mark/structures.py:

Start line: 106, End line: 145

```python
class ParameterSet(namedtuple("ParameterSet", "values, marks, id")):

    @classmethod
    def _for_parametrize(cls, argnames, argvalues, func, config, function_definition):
        if not isinstance(argnames, (tuple, list)):
            argnames = [x.strip() for x in argnames.split(",") if x.strip()]
            force_tuple = len(argnames) == 1
        else:
            force_tuple = False
        parameters = [
            ParameterSet.extract_from(x, force_tuple=force_tuple) for x in argvalues
        ]
        del argvalues

        if parameters:
            # check all parameter sets have the correct number of values
            for param in parameters:
                if len(param.values) != len(argnames):
                    msg = (
                        '{nodeid}: in "parametrize" the number of names ({names_len}):\n'
                        "  {names}\n"
                        "must be equal to the number of values ({values_len}):\n"
                        "  {values}"
                    )
                    fail(
                        msg.format(
                            nodeid=function_definition.nodeid,
                            values=param.values,
                            names=argnames,
                            names_len=len(argnames),
                            values_len=len(param.values),
                        ),
                        pytrace=False,
                    )
        else:
            # empty parameter set (likely computed at runtime): create a single
            # parameter set with NOTSET values, with the "empty parameter set" mark applied to it
            mark = get_empty_parameterset_mark(config, argnames, func)
            parameters.append(
                ParameterSet(values=(NOTSET,) * len(argnames), marks=[mark], id=None)
            )
        return argnames, parameters
```
### 35 - src/_pytest/fixtures.py:

Start line: 374, End line: 388

```python
class FixtureRequest(FuncargnamesCompatAttr):

    def _getnextfixturedef(self, argname):
        fixturedefs = self._arg2fixturedefs.get(argname, None)
        if fixturedefs is None:
            # we arrive here because of a dynamic call to
            # getfixturevalue(argname) usage which was naturally
            # not known at parsing/collection time
            parentid = self._pyfuncitem.parent.nodeid
            fixturedefs = self._fixturemanager.getfixturedefs(argname, parentid)
            self._arg2fixturedefs[argname] = fixturedefs
        # fixturedefs list is immutable so we maintain a decreasing index
        index = self._arg2index.get(argname, 0) - 1
        if fixturedefs is None or (-index > len(fixturedefs)):
            raise FixtureLookupError(argname, self)
        self._arg2index[argname] = index
        return fixturedefs[index]
```
### 38 - src/_pytest/fixtures.py:

Start line: 1177, End line: 1221

```python
class FixtureManager(object):

    def getfixtureclosure(self, fixturenames, parentnode):
        # collect the closure of all fixtures , starting with the given
        # fixturenames as the initial set.  As we have to visit all
        # factory definitions anyway, we also return an arg2fixturedefs
        # mapping so that the caller can reuse it and does not have
        # to re-discover fixturedefs again for each fixturename
        # (discovering matching fixtures for a given name/node is expensive)

        parentid = parentnode.nodeid
        fixturenames_closure = self._getautousenames(parentid)

        def merge(otherlist):
            for arg in otherlist:
                if arg not in fixturenames_closure:
                    fixturenames_closure.append(arg)

        merge(fixturenames)

        # at this point, fixturenames_closure contains what we call "initialnames",
        # which is a set of fixturenames the function immediately requests. We
        # need to return it as well, so save this.
        initialnames = tuple(fixturenames_closure)

        arg2fixturedefs = {}
        lastlen = -1
        while lastlen != len(fixturenames_closure):
            lastlen = len(fixturenames_closure)
            for argname in fixturenames_closure:
                if argname in arg2fixturedefs:
                    continue
                fixturedefs = self.getfixturedefs(argname, parentid)
                if fixturedefs:
                    arg2fixturedefs[argname] = fixturedefs
                    merge(fixturedefs[-1].argnames)

        def sort_by_scope(arg_name):
            try:
                fixturedefs = arg2fixturedefs[arg_name]
            except KeyError:
                return scopes.index("function")
            else:
                return fixturedefs[-1].scopenum

        fixturenames_closure.sort(key=sort_by_scope)
        return initialnames, fixturenames_closure, arg2fixturedefs
```
### 41 - src/_pytest/fixtures.py:

Start line: 873, End line: 903

```python
class FixtureDef(object):

    def execute(self, request):
        # get required arguments and register our own finish()
        # with their finalization
        for argname in self.argnames:
            fixturedef = request._get_active_fixturedef(argname)
            if argname != "request":
                fixturedef.addfinalizer(functools.partial(self.finish, request=request))

        my_cache_key = request.param_index
        cached_result = getattr(self, "cached_result", None)
        if cached_result is not None:
            result, cache_key, err = cached_result
            if my_cache_key == cache_key:
                if err is not None:
                    six.reraise(*err)
                else:
                    return result
            # we have a previous but differently parametrized fixture instance
            # so we need to tear it down before creating a new one
            self.finish(request)
            assert not hasattr(self, "cached_result")

        hook = self._fixturemanager.session.gethookproxy(request.node.fspath)
        return hook.pytest_fixture_setup(fixturedef=self, request=request)

    def __repr__(self):
        return "<FixtureDef argname=%r scope=%r baseid=%r>" % (
            self.argname,
            self.scope,
            self.baseid,
        )
```
### 42 - src/_pytest/fixtures.py:

Start line: 611, End line: 638

```python
class FixtureRequest(FuncargnamesCompatAttr):

    def _factorytraceback(self):
        lines = []
        for fixturedef in self._get_fixturestack():
            factory = fixturedef.func
            fs, lineno = getfslineno(factory)
            p = self._pyfuncitem.session.fspath.bestrelpath(fs)
            args = _format_args(factory)
            lines.append("%s:%d:  def %s%s" % (p, lineno + 1, factory.__name__, args))
        return lines

    def _getscopeitem(self, scope):
        if scope == "function":
            # this might also be a non-function Item despite its attribute name
            return self._pyfuncitem
        if scope == "package":
            node = get_scope_package(self._pyfuncitem, self._fixturedef)
        else:
            node = get_scope_node(self._pyfuncitem, scope)
        if node is None and scope == "class":
            # fallback to function item itself
            node = self._pyfuncitem
        assert node, 'Could not obtain a node for scope "{}" for function {!r}'.format(
            scope, self._pyfuncitem
        )
        return node

    def __repr__(self):
        return "<FixtureRequest for %r>" % (self.node)
```
### 43 - src/_pytest/mark/structures.py:

Start line: 38, End line: 61

```python
def get_empty_parameterset_mark(config, argnames, func):
    from ..nodes import Collector

    requested_mark = config.getini(EMPTY_PARAMETERSET_OPTION)
    if requested_mark in ("", None, "skip"):
        mark = MARK_GEN.skip
    elif requested_mark == "xfail":
        mark = MARK_GEN.xfail(run=False)
    elif requested_mark == "fail_at_collect":
        f_name = func.__name__
        _, lineno = getfslineno(func)
        raise Collector.CollectError(
            "Empty parameter set in '%s' at line %d" % (f_name, lineno + 1)
        )
    else:
        raise LookupError(requested_mark)
    fs, lineno = getfslineno(func)
    reason = "got empty parameter set %r, function %s at %s:%d" % (
        argnames,
        func.__name__,
        fs,
        lineno,
    )
    return mark(reason=reason)
```
### 46 - src/_pytest/fixtures.py:

Start line: 973, End line: 996

```python
@attr.s(frozen=True)
class FixtureFunctionMarker(object):
    scope = attr.ib()
    params = attr.ib(converter=attr.converters.optional(tuple))
    autouse = attr.ib(default=False)
    ids = attr.ib(default=None, converter=_ensure_immutable_ids)
    name = attr.ib(default=None)

    def __call__(self, function):
        if isclass(function):
            raise ValueError("class fixtures not supported (maybe in the future)")

        if getattr(function, "_pytestfixturefunction", False):
            raise ValueError(
                "fixture is being applied more than once to the same function"
            )

        function = wrap_function_to_error_out_if_called_directly(function, self)

        name = self.name or function.__name__
        if name == "request":
            warnings.warn(FIXTURE_NAMED_REQUEST)
        function._pytestfixturefunction = self
        return function
```
### 51 - src/_pytest/fixtures.py:

Start line: 1132, End line: 1145

```python
class FixtureManager(object):

    def getfixtureinfo(self, node, func, cls, funcargs=True):
        if funcargs and not getattr(node, "nofuncargs", False):
            argnames = getfuncargnames(func, cls=cls)
        else:
            argnames = ()
        usefixtures = itertools.chain.from_iterable(
            mark.args for mark in node.iter_markers(name="usefixtures")
        )
        initialnames = tuple(usefixtures) + argnames
        fm = node.session._fixturemanager
        initialnames, names_closure, arg2fixturedefs = fm.getfixtureclosure(
            initialnames, node
        )
        return FuncFixtureInfo(argnames, initialnames, names_closure, arg2fixturedefs)
```
### 57 - src/_pytest/fixtures.py:

Start line: 114, End line: 169

```python
def add_funcarg_pseudo_fixture_def(collector, metafunc, fixturemanager):
    # this function will transform all collected calls to a functions
    # if they use direct funcargs (i.e. direct parametrization)
    # because we want later test execution to be able to rely on
    # an existing FixtureDef structure for all arguments.
    # XXX we can probably avoid this algorithm  if we modify CallSpec2
    # to directly care for creating the fixturedefs within its methods.
    if not metafunc._calls[0].funcargs:
        return  # this function call does not have direct parametrization
    # collect funcargs of all callspecs into a list of values
    arg2params = {}
    arg2scope = {}
    for callspec in metafunc._calls:
        for argname, argvalue in callspec.funcargs.items():
            assert argname not in callspec.params
            callspec.params[argname] = argvalue
            arg2params_list = arg2params.setdefault(argname, [])
            callspec.indices[argname] = len(arg2params_list)
            arg2params_list.append(argvalue)
            if argname not in arg2scope:
                scopenum = callspec._arg2scopenum.get(argname, scopenum_function)
                arg2scope[argname] = scopes[scopenum]
        callspec.funcargs.clear()

    # register artificial FixtureDef's so that later at test execution
    # time we can rely on a proper FixtureDef to exist for fixture setup.
    arg2fixturedefs = metafunc._arg2fixturedefs
    for argname, valuelist in arg2params.items():
        # if we have a scope that is higher than function we need
        # to make sure we only ever create an according fixturedef on
        # a per-scope basis. We thus store and cache the fixturedef on the
        # node related to the scope.
        scope = arg2scope[argname]
        node = None
        if scope != "function":
            node = get_scope_node(collector, scope)
            if node is None:
                assert scope == "class" and isinstance(collector, _pytest.python.Module)
                # use module-level collector for class-scope (for now)
                node = collector
        if node and argname in node._name2pseudofixturedef:
            arg2fixturedefs[argname] = [node._name2pseudofixturedef[argname]]
        else:
            fixturedef = FixtureDef(
                fixturemanager,
                "",
                argname,
                get_direct_param_fixture_func,
                arg2scope[argname],
                valuelist,
                False,
                False,
            )
            arg2fixturedefs[argname] = [fixturedef]
            if node is not None:
                node._name2pseudofixturedef[argname] = fixturedef
```
### 66 - src/_pytest/fixtures.py:

Start line: 1147, End line: 1175

```python
class FixtureManager(object):

    def pytest_plugin_registered(self, plugin):
        nodeid = None
        try:
            p = py.path.local(plugin.__file__).realpath()
        except AttributeError:
            pass
        else:
            # construct the base nodeid which is later used to check
            # what fixtures are visible for particular tests (as denoted
            # by their test id)
            if p.basename.startswith("conftest.py"):
                nodeid = p.dirpath().relto(self.config.rootdir)
                if p.sep != nodes.SEP:
                    nodeid = nodeid.replace(p.sep, nodes.SEP)

        self.parsefactories(plugin, nodeid)

    def _getautousenames(self, nodeid):
        """ return a tuple of fixture names to be used. """
        autousenames = []
        for baseid, basenames in self._nodeid_and_autousenames:
            if nodeid.startswith(baseid):
                if baseid:
                    i = len(baseid)
                    nextchar = nodeid[i : i + 1]
                    if nextchar and nextchar not in ":/":
                        continue
                autousenames.extend(basenames)
        return autousenames
```
### 68 - src/_pytest/fixtures.py:

Start line: 1257, End line: 1315

```python
class FixtureManager(object):

    def parsefactories(self, node_or_obj, nodeid=NOTSET, unittest=False):
        if nodeid is not NOTSET:
            holderobj = node_or_obj
        else:
            holderobj = node_or_obj.obj
            nodeid = node_or_obj.nodeid
        if holderobj in self._holderobjseen:
            return

        self._holderobjseen.add(holderobj)
        autousenames = []
        for name in dir(holderobj):
            # The attribute can be an arbitrary descriptor, so the attribute
            # access below can raise. safe_getatt() ignores such exceptions.
            obj = safe_getattr(holderobj, name, None)
            marker = getfixturemarker(obj)
            if not isinstance(marker, FixtureFunctionMarker):
                # magic globals  with __getattr__ might have got us a wrong
                # fixture attribute
                continue

            if marker.name:
                name = marker.name

            # during fixture definition we wrap the original fixture function
            # to issue a warning if called directly, so here we unwrap it in order to not emit the warning
            # when pytest itself calls the fixture function
            if six.PY2 and unittest:
                # hack on Python 2 because of the unbound methods
                obj = get_real_func(obj)
            else:
                obj = get_real_method(obj, holderobj)

            fixture_def = FixtureDef(
                self,
                nodeid,
                name,
                obj,
                marker.scope,
                marker.params,
                unittest=unittest,
                ids=marker.ids,
            )

            faclist = self._arg2fixturedefs.setdefault(name, [])
            if fixture_def.has_location:
                faclist.append(fixture_def)
            else:
                # fixturedefs with no location are at the front
                # so this inserts the current fixturedef after the
                # existing fixturedefs from external plugins but
                # before the fixturedefs provided in conftests.
                i = len([f for f in faclist if not f.has_location])
                faclist.insert(i, fixture_def)
            if marker.autouse:
                autousenames.append(name)

        if autousenames:
            self._nodeid_and_autousenames.append((nodeid or "", autousenames))
```
### 73 - src/_pytest/fixtures.py:

Start line: 277, End line: 300

```python
def fillfixtures(function):
    """ fill missing funcargs for a test function. """
    try:
        request = function._request
    except AttributeError:
        # XXX this special code path is only expected to execute
        # with the oejskit plugin.  It uses classes with funcargs
        # and we thus have to work a bit to allow this.
        fm = function.session._fixturemanager
        fi = fm.getfixtureinfo(function.parent, function.obj, None)
        function._fixtureinfo = fi
        request = function._request = FixtureRequest(function)
        request._fillfixtures()
        # prune out funcargs for jstests
        newfuncargs = {}
        for name in fi.argnames:
            newfuncargs[name] = function.funcargs[name]
        function.funcargs = newfuncargs
    else:
        request._fillfixtures()


def get_direct_param_fixture_func(request):
    return request.param
```
### 75 - src/_pytest/mark/structures.py:

Start line: 64, End line: 85

```python
class ParameterSet(namedtuple("ParameterSet", "values, marks, id")):
    @classmethod
    def param(cls, *values, **kwargs):
        marks = kwargs.pop("marks", ())
        if isinstance(marks, MarkDecorator):
            marks = (marks,)
        else:
            assert isinstance(marks, (tuple, list, set))

        id_ = kwargs.pop("id", None)
        if id_ is not None:
            if not isinstance(id_, six.string_types):
                raise TypeError(
                    "Expected id to be a string, got {}: {!r}".format(type(id_), id_)
                )
            id_ = ascii_escaped(id_)

        if kwargs:
            warnings.warn(
                PYTEST_PARAM_UNKNOWN_KWARGS.format(args=sorted(kwargs)), stacklevel=3
            )
        return cls(values, marks, id_)
```
### 76 - src/_pytest/fixtures.py:

Start line: 172, End line: 206

```python
def getfixturemarker(obj):
    """ return fixturemarker or None if it doesn't exist or raised
    exceptions."""
    try:
        return getattr(obj, "_pytestfixturefunction", None)
    except TEST_OUTCOME:
        # some objects raise errors like request (from flask import request)
        # we don't expect them to be fixture functions
        return None


def get_parametrized_fixture_keys(item, scopenum):
    """ return list of keys for all parametrized arguments which match
    the specified scope. """
    assert scopenum < scopenum_function  # function
    try:
        cs = item.callspec
    except AttributeError:
        pass
    else:
        # cs.indices.items() is random order of argnames.  Need to
        # sort this so that different calls to
        # get_parametrized_fixture_keys will be deterministic.
        for argname, param_index in sorted(cs.indices.items()):
            if cs._arg2scopenum[argname] != scopenum:
                continue
            if scopenum == 0:  # session
                key = (argname, param_index)
            elif scopenum == 1:  # package
                key = (argname, param_index, item.fspath.dirpath())
            elif scopenum == 2:  # module
                key = (argname, param_index, item.fspath)
            elif scopenum == 3:  # class
                key = (argname, param_index, item.fspath, item.cls)
            yield key
```
### 81 - src/_pytest/fixtures.py:

Start line: 780, End line: 809

```python
def fail_fixturefunc(fixturefunc, msg):
    fs, lineno = getfslineno(fixturefunc)
    location = "%s:%s" % (fs, lineno + 1)
    source = _pytest._code.Source(fixturefunc)
    fail(msg + ":\n\n" + str(source.indent()) + "\n" + location, pytrace=False)


def call_fixture_func(fixturefunc, request, kwargs):
    yieldctx = is_generator(fixturefunc)
    if yieldctx:
        it = fixturefunc(**kwargs)
        res = next(it)
        finalizer = functools.partial(_teardown_yield_fixture, fixturefunc, it)
        request.addfinalizer(finalizer)
    else:
        res = fixturefunc(**kwargs)
    return res


def _teardown_yield_fixture(fixturefunc, it):
    """Executes the teardown of a fixture function by advancing the iterator after the
    yield and ensure the iteration ends (if not it means there is more than one yield in the function)"""
    try:
        next(it)
    except StopIteration:
        pass
    else:
        fail_fixturefunc(
            fixturefunc, "yield_fixture function has more than one 'yield'"
        )
```
### 82 - src/_pytest/fixtures.py:

Start line: 1317, End line: 1335

```python
class FixtureManager(object):

    def getfixturedefs(self, argname, nodeid):
        """
        Gets a list of fixtures which are applicable to the given node id.

        :param str argname: name of the fixture to search for
        :param str nodeid: full node id of the requesting test.
        :return: list[FixtureDef]
        """
        try:
            fixturedefs = self._arg2fixturedefs[argname]
        except KeyError:
            return None
        return tuple(self._matchfactories(fixturedefs, nodeid))

    def _matchfactories(self, fixturedefs, nodeid):
        for fixturedef in fixturedefs:
            if nodes.ischildnode(fixturedef.baseid, nodeid):
                yield fixturedef
```
### 96 - src/_pytest/fixtures.py:

Start line: 999, End line: 1050

```python
def fixture(scope="function", params=None, autouse=False, ids=None, name=None):
    """Decorator to mark a fixture factory function.

    This decorator can be used, with or without parameters, to define a
    fixture function.

    The name of the fixture function can later be referenced to cause its
    invocation ahead of running tests: test
    modules or classes can use the ``pytest.mark.usefixtures(fixturename)``
    marker.

    Test functions can directly use fixture names as input
    arguments in which case the fixture instance returned from the fixture
    function will be injected.

    Fixtures can provide their values to test functions using ``return`` or ``yield``
    statements. When using ``yield`` the code block after the ``yield`` statement is executed
    as teardown code regardless of the test outcome, and must yield exactly once.

    :arg scope: the scope for which this fixture is shared, one of
                ``"function"`` (default), ``"class"``, ``"module"``,
                ``"package"`` or ``"session"``.

                ``"package"`` is considered **experimental** at this time.

    :arg params: an optional list of parameters which will cause multiple
                invocations of the fixture function and all of the tests
                using it.
                The current parameter is available in ``request.param``.

    :arg autouse: if True, the fixture func is activated for all tests that
                can see it.  If False (the default) then an explicit
                reference is needed to activate the fixture.

    :arg ids: list of string ids each corresponding to the params
                so that they are part of the test id. If no ids are provided
                they will be generated automatically from the params.

    :arg name: the name of the fixture. This defaults to the name of the
                decorated function. If a fixture is used in the same module in
                which it is defined, the function name of the fixture will be
                shadowed by the function arg that requests the fixture; one way
                to resolve this is to name the decorated function
                ``fixture_<fixturename>`` and then use
                ``@pytest.fixture(name='<fixturename>')``.
    """
    if callable(scope) and params is None and autouse is False:
        # direct decoration
        return FixtureFunctionMarker("function", params, autouse, name=name)(scope)
    if params is not None and not isinstance(params, (list, tuple)):
        params = list(params)
    return FixtureFunctionMarker(scope, params, autouse, ids=ids, name=name)
```
### 100 - src/_pytest/fixtures.py:

Start line: 946, End line: 970

```python
def _ensure_immutable_ids(ids):
    if ids is None:
        return
    if callable(ids):
        return ids
    return tuple(ids)


def wrap_function_to_error_out_if_called_directly(function, fixture_marker):
    """Wrap the given fixture function so we can raise an error about it being called directly,
    instead of used as an argument in a test function.
    """
    message = FIXTURE_FUNCTION_CALL.format(
        name=fixture_marker.name or function.__name__
    )

    @six.wraps(function)
    def result(*args, **kwargs):
        fail(message, pytrace=False)

    # keep reference to the original function in our own custom attribute so we don't unwrap
    # further than this point and lose useful wrappings like @mock.patch (#3774)
    result.__pytest_wrapped__ = _PytestWrapper(function)

    return result
```
### 109 - src/_pytest/fixtures.py:

Start line: 342, End line: 372

```python
class FixtureRequest(FuncargnamesCompatAttr):
    """ A request for a fixture from a test or fixture function.

    A request object gives access to the requesting test context
    and has an optional ``param`` attribute in case
    the fixture is parametrized indirectly.
    """

    def __init__(self, pyfuncitem):
        self._pyfuncitem = pyfuncitem
        #: fixture for which this request is being performed
        self.fixturename = None
        #: Scope string, one of "function", "class", "module", "session"
        self.scope = "function"
        self._fixture_defs = {}  # argname -> FixtureDef
        fixtureinfo = pyfuncitem._fixtureinfo
        self._arg2fixturedefs = fixtureinfo.name2fixturedefs.copy()
        self._arg2index = {}
        self._fixturemanager = pyfuncitem.session._fixturemanager

    @property
    def fixturenames(self):
        """names of all active fixtures in this request"""
        result = list(self._pyfuncitem._fixtureinfo.names_closure)
        result.extend(set(self._fixture_defs).difference(result))
        return result

    @property
    def node(self):
        """ underlying collection node (depends on current request scope)"""
        return self._getscopeitem(self.scope)
```
### 111 - src/_pytest/fixtures.py:

Start line: 390, End line: 486

```python
class FixtureRequest(FuncargnamesCompatAttr):

    @property
    def config(self):
        """ the pytest config object associated with this request. """
        return self._pyfuncitem.config

    @scopeproperty()
    def function(self):
        """ test function object if the request has a per-function scope. """
        return self._pyfuncitem.obj

    @scopeproperty("class")
    def cls(self):
        """ class (can be None) where the test function was collected. """
        clscol = self._pyfuncitem.getparent(_pytest.python.Class)
        if clscol:
            return clscol.obj

    @property
    def instance(self):
        """ instance (can be None) on which test function was collected. """
        # unittest support hack, see _pytest.unittest.TestCaseFunction
        try:
            return self._pyfuncitem._testcase
        except AttributeError:
            function = getattr(self, "function", None)
            return getattr(function, "__self__", None)

    @scopeproperty()
    def module(self):
        """ python module object where the test function was collected. """
        return self._pyfuncitem.getparent(_pytest.python.Module).obj

    @scopeproperty()
    def fspath(self):
        """ the file system path of the test module which collected this test. """
        return self._pyfuncitem.fspath

    @property
    def keywords(self):
        """ keywords/markers dictionary for the underlying node. """
        return self.node.keywords

    @property
    def session(self):
        """ pytest session object. """
        return self._pyfuncitem.session

    def addfinalizer(self, finalizer):
        """ add finalizer/teardown function to be called after the
        last test within the requesting test context finished
        execution. """
        # XXX usually this method is shadowed by fixturedef specific ones
        self._addfinalizer(finalizer, scope=self.scope)

    def _addfinalizer(self, finalizer, scope):
        colitem = self._getscopeitem(scope)
        self._pyfuncitem.session._setupstate.addfinalizer(
            finalizer=finalizer, colitem=colitem
        )

    def applymarker(self, marker):
        """ Apply a marker to a single test function invocation.
        This method is useful if you don't want to have a keyword/marker
        on all function invocations.

        :arg marker: a :py:class:`_pytest.mark.MarkDecorator` object
            created by a call to ``pytest.mark.NAME(...)``.
        """
        self.node.add_marker(marker)

    def raiseerror(self, msg):
        """ raise a FixtureLookupError with the given message. """
        raise self._fixturemanager.FixtureLookupError(None, self, msg)

    def _fillfixtures(self):
        item = self._pyfuncitem
        fixturenames = getattr(item, "fixturenames", self.fixturenames)
        for argname in fixturenames:
            if argname not in item.funcargs:
                item.funcargs[argname] = self.getfixturevalue(argname)

    def getfixturevalue(self, argname):
        """ Dynamically run a named fixture function.

        Declaring fixtures via function argument is recommended where possible.
        But if you can only decide whether to use another fixture at test
        setup time, you may use this function to retrieve it inside a fixture
        or test function body.
        """
        return self._get_active_fixturedef(argname).cached_result[0]

    def getfuncargvalue(self, argname):
        """ Deprecated, use getfixturevalue. """
        from _pytest import deprecated

        warnings.warn(deprecated.GETFUNCARGVALUE, stacklevel=2)
        return self.getfixturevalue(argname)
```
### 123 - src/_pytest/fixtures.py:

Start line: 846, End line: 871

```python
class FixtureDef(object):

    def finish(self, request):
        exceptions = []
        try:
            while self._finalizers:
                try:
                    func = self._finalizers.pop()
                    func()
                except:  # noqa
                    exceptions.append(sys.exc_info())
            if exceptions:
                e = exceptions[0]
                del (
                    exceptions
                )  # ensure we don't keep all frames alive because of the traceback
                six.reraise(*e)

        finally:
            hook = self._fixturemanager.session.gethookproxy(request.node.fspath)
            hook.pytest_fixture_post_finalizer(fixturedef=self, request=request)
            # even if finalization fails, we invalidate
            # the cached fixture value and remove
            # all finalizers because they may be bound methods which will
            # keep instances alive
            if hasattr(self, "cached_result"):
                del self.cached_result
            self._finalizers = []
```
### 125 - src/_pytest/fixtures.py:

Start line: 641, End line: 673

```python
class SubRequest(FixtureRequest):
    """ a sub request for handling getting a fixture from a
    test function/fixture. """

    def __init__(self, request, scope, param, param_index, fixturedef):
        self._parent_request = request
        self.fixturename = fixturedef.argname
        if param is not NOTSET:
            self.param = param
        self.param_index = param_index
        self.scope = scope
        self._fixturedef = fixturedef
        self._pyfuncitem = request._pyfuncitem
        self._fixture_defs = request._fixture_defs
        self._arg2fixturedefs = request._arg2fixturedefs
        self._arg2index = request._arg2index
        self._fixturemanager = request._fixturemanager

    def __repr__(self):
        return "<SubRequest %r for %r>" % (self.fixturename, self._pyfuncitem)

    def addfinalizer(self, finalizer):
        self._fixturedef.addfinalizer(finalizer)

    def _schedule_finalizers(self, fixturedef, subrequest):
        # if the executing fixturedef was not explicitly requested in the argument list (via
        # getfixturevalue inside the fixture call) then ensure this fixture def will be finished
        # first
        if fixturedef.argname not in self.funcargnames:
            fixturedef.addfinalizer(
                functools.partial(self._fixturedef.finish, request=self)
            )
        super(SubRequest, self)._schedule_finalizers(fixturedef, subrequest)
```
### 127 - src/_pytest/mark/structures.py:

Start line: 87, End line: 104

```python
class ParameterSet(namedtuple("ParameterSet", "values, marks, id")):

    @classmethod
    def extract_from(cls, parameterset, force_tuple=False):
        """
        :param parameterset:
            a legacy style parameterset that may or may not be a tuple,
            and may or may not be wrapped into a mess of mark objects

        :param force_tuple:
            enforce tuple wrapping so single argument tuple values
            don't get decomposed and break tests
        """

        if isinstance(parameterset, cls):
            return parameterset
        if force_tuple:
            return cls.param(parameterset)
        else:
            return cls(parameterset, marks=[], id=None)
```
### 133 - src/_pytest/fixtures.py:

Start line: 303, End line: 339

```python
@attr.s(slots=True)
class FuncFixtureInfo(object):
    # original function argument names
    argnames = attr.ib(type=tuple)
    # argnames that function immediately requires. These include argnames +
    # fixture names specified via usefixtures and via autouse=True in fixture
    # definitions.
    initialnames = attr.ib(type=tuple)
    names_closure = attr.ib()  # List[str]
    name2fixturedefs = attr.ib()  # List[str, List[FixtureDef]]

    def prune_dependency_tree(self):
        """Recompute names_closure from initialnames and name2fixturedefs

        Can only reduce names_closure, which means that the new closure will
        always be a subset of the old one. The order is preserved.

        This method is needed because direct parametrization may shadow some
        of the fixtures that were included in the originally built dependency
        tree. In this way the dependency tree can get pruned, and the closure
        of argnames may get reduced.
        """
        closure = set()
        working_set = set(self.initialnames)
        while working_set:
            argname = working_set.pop()
            # argname may be smth not included in the original names_closure,
            # in which case we ignore it. This currently happens with pseudo
            # FixtureDefs which wrap 'get_direct_param_fixture_func(request)'.
            # So they introduce the new dependency 'request' which might have
            # been missing in the original tree (closure).
            if argname not in closure and argname in self.names_closure:
                closure.add(argname)
                if argname in self.name2fixturedefs:
                    working_set.update(self.name2fixturedefs[argname][-1].argnames)

        self.names_closure[:] = sorted(closure, key=self.names_closure.index)
```
### 137 - src/_pytest/fixtures.py:

Start line: 1088, End line: 1130

```python
class FixtureManager(object):
    """
    pytest fixtures definitions and information is stored and managed
    from this class.

    During collection fm.parsefactories() is called multiple times to parse
    fixture function definitions into FixtureDef objects and internal
    data structures.

    During collection of test functions, metafunc-mechanics instantiate
    a FuncFixtureInfo object which is cached per node/func-name.
    This FuncFixtureInfo object is later retrieved by Function nodes
    which themselves offer a fixturenames attribute.

    The FuncFixtureInfo object holds information about fixtures and FixtureDefs
    relevant for a particular function.  An initial list of fixtures is
    assembled like this:

    - ini-defined usefixtures
    - autouse-marked fixtures along the collection chain up from the function
    - usefixtures markers at module/class/function level
    - test function funcargs

    Subsequently the funcfixtureinfo.fixturenames attribute is computed
    as the closure of the fixtures needed to setup the initial fixtures,
    i. e. fixtures needed by fixture functions themselves are appended
    to the fixturenames list.

    Upon the test-setup phases all fixturenames are instantiated, retrieved
    by a lookup of their FuncFixtureInfo.
    """

    FixtureLookupError = FixtureLookupError
    FixtureLookupErrorRepr = FixtureLookupErrorRepr

    def __init__(self, session):
        self.session = session
        self.config = session.config
        self._arg2fixturedefs = {}
        self._holderobjseen = set()
        self._arg2finish = {}
        self._nodeid_and_autousenames = [("", self.config.getini("usefixtures"))]
        session.config.pluginmanager.register(self, "funcmanage")
```
### 146 - src/_pytest/fixtures.py:

Start line: 753, End line: 777

```python
class FixtureLookupErrorRepr(TerminalRepr):
    def __init__(self, filename, firstlineno, tblines, errorstring, argname):
        self.tblines = tblines
        self.errorstring = errorstring
        self.filename = filename
        self.firstlineno = firstlineno
        self.argname = argname

    def toterminal(self, tw):
        # tw.line("FixtureLookupError: %s" %(self.argname), red=True)
        for tbline in self.tblines:
            tw.line(tbline.rstrip())
        lines = self.errorstring.split("\n")
        if lines:
            tw.line(
                "{}       {}".format(FormattedExcinfo.fail_marker, lines[0].strip()),
                red=True,
            )
            for line in lines[1:]:
                tw.line(
                    "{}       {}".format(FormattedExcinfo.flow_marker, line.strip()),
                    red=True,
                )
        tw.line()
        tw.line("%s:%d" % (self.filename, self.firstlineno + 1))
```
### 167 - src/_pytest/mark/structures.py:

Start line: 281, End line: 328

```python
class MarkGenerator(object):
    """ Factory for :class:`MarkDecorator` objects - exposed as
    a ``pytest.mark`` singleton instance.  Example::

         import pytest
         @pytest.mark.slowtest
         def test_function():
            pass

    will set a 'slowtest' :class:`MarkInfo` object
    on the ``test_function`` object. """

    _config = None
    _markers = set()

    def __getattr__(self, name):
        if name[0] == "_":
            raise AttributeError("Marker name must NOT start with underscore")

        if self._config is not None:
            # We store a set of markers as a performance optimisation - if a mark
            # name is in the set we definitely know it, but a mark may be known and
            # not in the set.  We therefore start by updating the set!
            if name not in self._markers:
                for line in self._config.getini("markers"):
                    # example lines: "skipif(condition): skip the given test if..."
                    # or "hypothesis: tests which use Hypothesis", so to get the
                    # marker name we split on both `:` and `(`.
                    marker = line.split(":")[0].split("(")[0].strip()
                    self._markers.add(marker)

            # If the name is not in the set of known marks after updating,
            # then it really is time to issue a warning or an error.
            if name not in self._markers:
                if self._config.option.strict_markers:
                    fail(
                        "{!r} not found in `markers` configuration option".format(name),
                        pytrace=False,
                    )
                else:
                    warnings.warn(
                        "Unknown pytest.mark.%s - is this a typo?  You can register "
                        "custom marks to avoid this warning - for details, see "
                        "https://docs.pytest.org/en/latest/mark.html" % name,
                        PytestUnknownMarkWarning,
                    )

        return MarkDecorator(Mark(name, (), {}))
```
### 169 - src/_pytest/fixtures.py:

Start line: 1053, End line: 1085

```python
def yield_fixture(scope="function", params=None, autouse=False, ids=None, name=None):
    """ (return a) decorator to mark a yield-fixture factory function.

    .. deprecated:: 3.0
        Use :py:func:`pytest.fixture` directly instead.
    """
    return fixture(scope=scope, params=params, autouse=autouse, ids=ids, name=name)


defaultfuncargprefixmarker = fixture()


@fixture(scope="session")
def pytestconfig(request):
    """Session-scoped fixture that returns the :class:`_pytest.config.Config` object.

    Example::

        def test_foo(pytestconfig):
            if pytestconfig.getoption("verbose") > 0:
                ...

    """
    return request.config


def pytest_addoption(parser):
    parser.addini(
        "usefixtures",
        type="args",
        default=[],
        help="list of default fixtures to be used with this project",
    )
```
### 174 - src/_pytest/fixtures.py:

Start line: 676, End line: 696

```python
scopes = "session package module class function".split()
scopenum_function = scopes.index("function")


def scopemismatch(currentscope, newscope):
    return scopes.index(newscope) > scopes.index(currentscope)


def scope2index(scope, descr, where=None):
    """Look up the index of ``scope`` and raise a descriptive value error
    if not defined.
    """
    try:
        return scopes.index(scope)
    except ValueError:
        fail(
            "{} {}got an unexpected scope value '{}'".format(
                descr, "from {} ".format(where) if where else "", scope
            ),
            pytrace=False,
        )
```
### 177 - src/_pytest/fixtures.py:

Start line: 517, End line: 589

```python
class FixtureRequest(FuncargnamesCompatAttr):

    def _compute_fixture_value(self, fixturedef):
        """
        Creates a SubRequest based on "self" and calls the execute method of the given fixturedef object. This will
        force the FixtureDef object to throw away any previous results and compute a new fixture value, which
        will be stored into the FixtureDef object itself.

        :param FixtureDef fixturedef:
        """
        # prepare a subrequest object before calling fixture function
        # (latter managed by fixturedef)
        argname = fixturedef.argname
        funcitem = self._pyfuncitem
        scope = fixturedef.scope
        try:
            param = funcitem.callspec.getparam(argname)
        except (AttributeError, ValueError):
            param = NOTSET
            param_index = 0
            has_params = fixturedef.params is not None
            fixtures_not_supported = getattr(funcitem, "nofuncargs", False)
            if has_params and fixtures_not_supported:
                msg = (
                    "{name} does not support fixtures, maybe unittest.TestCase subclass?\n"
                    "Node id: {nodeid}\n"
                    "Function type: {typename}"
                ).format(
                    name=funcitem.name,
                    nodeid=funcitem.nodeid,
                    typename=type(funcitem).__name__,
                )
                fail(msg, pytrace=False)
            if has_params:
                frame = inspect.stack()[3]
                frameinfo = inspect.getframeinfo(frame[0])
                source_path = frameinfo.filename
                source_lineno = frameinfo.lineno
                source_path = py.path.local(source_path)
                if source_path.relto(funcitem.config.rootdir):
                    source_path = source_path.relto(funcitem.config.rootdir)
                msg = (
                    "The requested fixture has no parameter defined for test:\n"
                    "    {}\n\n"
                    "Requested fixture '{}' defined in:\n{}"
                    "\n\nRequested here:\n{}:{}".format(
                        funcitem.nodeid,
                        fixturedef.argname,
                        getlocation(fixturedef.func, funcitem.config.rootdir),
                        source_path,
                        source_lineno,
                    )
                )
                fail(msg, pytrace=False)
        else:
            param_index = funcitem.callspec.indices[argname]
            # if a parametrize invocation set a scope it will override
            # the static scope defined with the fixture function
            paramscopenum = funcitem.callspec._arg2scopenum.get(argname)
            if paramscopenum is not None:
                scope = scopes[paramscopenum]

        subrequest = SubRequest(self, scope, param, param_index, fixturedef)

        # check if a higher-level scoped fixture accesses a lower level one
        subrequest._check_scope(argname, self.scope, scope)

        # clear sys.exc_info before invoking the fixture (python bug?)
        # if it's not explicitly cleared it will leak into the call
        exc_clear()
        try:
            # call the fixture function
            fixturedef.execute(request=subrequest)
        finally:
            self._schedule_finalizers(fixturedef, subrequest)
```
### 192 - src/_pytest/fixtures.py:

Start line: 926, End line: 943

```python
def pytest_fixture_setup(fixturedef, request):
    """ Execution of fixture setup. """
    kwargs = {}
    for argname in fixturedef.argnames:
        fixdef = request._get_active_fixturedef(argname)
        result, arg_cache_key, exc = fixdef.cached_result
        request._check_scope(argname, request.scope, fixdef.scope)
        kwargs[argname] = result

    fixturefunc = resolve_fixture_function(fixturedef, request)
    my_cache_key = request.param_index
    try:
        result = call_fixture_func(fixturefunc, request, kwargs)
    except TEST_OUTCOME:
        fixturedef.cached_result = (None, my_cache_key, sys.exc_info())
        raise
    fixturedef.cached_result = (result, my_cache_key, None)
    return result
```
### 218 - src/_pytest/mark/structures.py:

Start line: 331, End line: 368

```python
MARK_GEN = MarkGenerator()


class NodeKeywords(MappingMixin):
    def __init__(self, node):
        self.node = node
        self.parent = node.parent
        self._markers = {node.name: True}

    def __getitem__(self, key):
        try:
            return self._markers[key]
        except KeyError:
            if self.parent is None:
                raise
            return self.parent.keywords[key]

    def __setitem__(self, key, value):
        self._markers[key] = value

    def __delitem__(self, key):
        raise ValueError("cannot delete key in keywords dict")

    def __iter__(self):
        seen = self._seen()
        return iter(seen)

    def _seen(self):
        seen = set(self._markers)
        if self.parent is not None:
            seen.update(self.parent.keywords)
        return seen

    def __len__(self):
        return len(self._seen())

    def __repr__(self):
        return "<NodeKeywords for node %s>" % (self.node,)
```
### 219 - src/_pytest/mark/structures.py:

Start line: 1, End line: 35

```python
# -*- coding: utf-8 -*-
import inspect
import warnings
from collections import namedtuple
from operator import attrgetter

import attr
import six

from ..compat import ascii_escaped
from ..compat import getfslineno
from ..compat import MappingMixin
from ..compat import NOTSET
from _pytest.deprecated import PYTEST_PARAM_UNKNOWN_KWARGS
from _pytest.outcomes import fail
from _pytest.warning_types import PytestUnknownMarkWarning

EMPTY_PARAMETERSET_OPTION = "empty_parameter_set_mark"


def alias(name, warning=None):
    getter = attrgetter(name)

    def warned(self):
        warnings.warn(warning, stacklevel=2)
        return getter(self)

    return property(getter if warning is None else warned, doc="alias for " + name)


def istestfunc(func):
    return (
        hasattr(func, "__call__")
        and getattr(func, "__name__", "<lambda>") != "<lambda>"
    )
```
### 222 - src/_pytest/fixtures.py:

Start line: 209, End line: 234

```python
# algorithm for sorting on a per-parametrized resource setup basis
# it is called for scopenum==0 (session) first and performs sorting
# down to the lower scopes such as to minimize number of "high scope"
# setups and teardowns


def reorder_items(items):
    argkeys_cache = {}
    items_by_argkey = {}
    for scopenum in range(0, scopenum_function):
        argkeys_cache[scopenum] = d = {}
        items_by_argkey[scopenum] = item_d = defaultdict(deque)
        for item in items:
            keys = OrderedDict.fromkeys(get_parametrized_fixture_keys(item, scopenum))
            if keys:
                d[item] = keys
                for key in keys:
                    item_d[key].append(item)
    items = OrderedDict.fromkeys(items)
    return list(reorder_items_atscope(items, argkeys_cache, items_by_argkey, 0))


def fix_cache_order(item, argkeys_cache, items_by_argkey):
    for scopenum in range(0, scopenum_function):
        for key in argkeys_cache[scopenum].get(item, []):
            items_by_argkey[scopenum][key].appendleft(item)
```
### 234 - src/_pytest/mark/structures.py:

Start line: 148, End line: 168

```python
@attr.s(frozen=True)
class Mark(object):
    #: name of the mark
    name = attr.ib(type=str)
    #: positional arguments of the mark decorator
    args = attr.ib()  # List[object]
    #: keyword arguments of the mark decorator
    kwargs = attr.ib()  # Dict[str, object]

    def combined_with(self, other):
        """
        :param other: the mark to combine with
        :type other: Mark
        :rtype: Mark

        combines by appending args and merging the mappings
        """
        assert self.name == other.name
        return Mark(
            self.name, self.args + other.args, dict(self.kwargs, **other.kwargs)
        )
```
