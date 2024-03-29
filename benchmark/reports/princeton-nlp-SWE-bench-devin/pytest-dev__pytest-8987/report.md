# pytest-dev__pytest-8987

| **pytest-dev/pytest** | `a446ee81fd6674c2b7d1f0ee76467f1ffc1619fc` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 65327 |
| **Any found context length** | 16680 |
| **Avg pos** | 199.0 |
| **Min pos** | 48 |
| **Max pos** | 151 |
| **Top file pos** | 10 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/src/_pytest/mark/expression.py b/src/_pytest/mark/expression.py
--- a/src/_pytest/mark/expression.py
+++ b/src/_pytest/mark/expression.py
@@ -6,7 +6,7 @@
 expr:       and_expr ('or' and_expr)*
 and_expr:   not_expr ('and' not_expr)*
 not_expr:   'not' not_expr | '(' expr ')' | ident
-ident:      (\w|:|\+|-|\.|\[|\])+
+ident:      (\w|:|\+|-|\.|\[|\]|\\)+
 
 The semantics are:
 
@@ -88,7 +88,7 @@ def lex(self, input: str) -> Iterator[Token]:
                 yield Token(TokenType.RPAREN, ")", pos)
                 pos += 1
             else:
-                match = re.match(r"(:?\w|:|\+|-|\.|\[|\])+", input[pos:])
+                match = re.match(r"(:?\w|:|\+|-|\.|\[|\]|\\)+", input[pos:])
                 if match:
                     value = match.group(0)
                     if value == "or":

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| src/_pytest/mark/expression.py | 9 | 9 | 48 | 10 | 16680
| src/_pytest/mark/expression.py | 91 | 91 | 151 | 10 | 65327


## Problem Statement

```
pytest -k doesn't work with "\"?
### Discussed in https://github.com/pytest-dev/pytest/discussions/8982

<div type='discussions-op-text'>

<sup>Originally posted by **nguydavi** August  7, 2021</sup>
Hey!

I've been trying to use `pytest -k` passing the name I got by parametrizing my test. For example,

\`\`\`
$ pytest -vk 'test_solution[foo.py-5\n10\n-16\n]' validate.py
=========================================================================================================== test session starts ============================================================================================================platform linux -- Python 3.8.10, pytest-6.2.4, py-1.10.0, pluggy-0.13.1 -- /usr/bin/python3
cachedir: .pytest_cache
rootdir: /home/david/foo
collected 4 items

========================================================================================================== no tests ran in 0.01s ===========================================================================================================ERROR: Wrong expression passed to '-k': test_solution[foo.py-5\n10\n-16\n]: at column 23: unexpected character "\"
\`\`\`

Note the error message
\`\`\`
ERROR: Wrong expression passed to '-k': test_solution[foo.py-5\n10\n-16\n]: at column 23: unexpected character "\"
\`\`\`

I tried escaping the `\` but that didn't work, the only way I can make it work is to remove the backslashes completely,

\`\`\`
$ pytest -vk 'test_solution[foo.py-5 and 10' validate.py
\`\`\`

Is `\` just not supported by `-k` ? Or am I missing something ?

Thanks!

EDIT:
A possible test case
\`\`\`
@pytest.mark.parametrize(
    "param1, param2",
    [
        pytest.param(
            '5\n10\n', '16\n'
        ),
    ],
)
def test_solution(param1, param2):
  pass
\`\`\`
Which is then referred by `pytest` as `test_solution[5\n10\n-16\n]` . Essentially a new line character `\n` in the string brings the issue (or any escaped character probably)</div>

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 testing/python/metafunc.py | 1412 | 1436| 214 | 214 | 14658 | 
| 2 | 2 testing/python/collect.py | 361 | 379| 158 | 372 | 25133 | 
| 3 | 3 doc/en/conf.py | 1 | 110| 794 | 1166 | 28298 | 
| 4 | 3 testing/python/collect.py | 608 | 636| 214 | 1380 | 28298 | 
| 5 | 3 testing/python/collect.py | 653 | 681| 202 | 1582 | 28298 | 
| 6 | 3 testing/python/collect.py | 516 | 527| 110 | 1692 | 28298 | 
| 7 | 3 testing/python/metafunc.py | 1710 | 1724| 138 | 1830 | 28298 | 
| 8 | 3 testing/python/metafunc.py | 116 | 138| 190 | 2020 | 28298 | 
| 9 | 3 testing/python/metafunc.py | 1694 | 1708| 133 | 2153 | 28298 | 
| 10 | 3 testing/python/collect.py | 638 | 651| 112 | 2265 | 28298 | 
| 11 | 3 testing/python/collect.py | 346 | 359| 114 | 2379 | 28298 | 
| 12 | 3 testing/python/collect.py | 706 | 719| 124 | 2503 | 28298 | 
| 13 | 4 testing/python/fixtures.py | 2944 | 2997| 415 | 2918 | 57060 | 
| 14 | 4 testing/python/collect.py | 593 | 606| 111 | 3029 | 57060 | 
| 15 | 4 testing/python/collect.py | 381 | 410| 202 | 3231 | 57060 | 
| 16 | 5 testing/python/integration.py | 427 | 449| 146 | 3377 | 60188 | 
| 17 | 5 testing/python/metafunc.py | 1835 | 1871| 265 | 3642 | 60188 | 
| 18 | 6 src/_pytest/mark/__init__.py | 74 | 113| 357 | 3999 | 62229 | 
| 19 | 6 testing/python/fixtures.py | 2031 | 2942| 6163 | 10162 | 62229 | 
| 20 | 6 testing/python/metafunc.py | 1791 | 1814| 230 | 10392 | 62229 | 
| 21 | 7 testing/python/raises.py | 215 | 232| 205 | 10597 | 64456 | 
| 22 | 7 testing/python/collect.py | 441 | 460| 133 | 10730 | 64456 | 
| 23 | 7 testing/python/metafunc.py | 1726 | 1746| 203 | 10933 | 64456 | 
| 24 | 7 testing/python/metafunc.py | 1662 | 1676| 118 | 11051 | 64456 | 
| 25 | 7 testing/python/integration.py | 451 | 478| 166 | 11217 | 64456 | 
| 26 | 7 src/_pytest/mark/__init__.py | 47 | 71| 216 | 11433 | 64456 | 
| 27 | 7 testing/python/metafunc.py | 1645 | 1660| 135 | 11568 | 64456 | 
| 28 | 7 testing/python/metafunc.py | 1873 | 1908| 296 | 11864 | 64456 | 
| 29 | 7 testing/python/metafunc.py | 1678 | 1692| 130 | 11994 | 64456 | 
| 30 | 7 testing/python/metafunc.py | 591 | 617| 238 | 12232 | 64456 | 
| 31 | 7 testing/python/metafunc.py | 1816 | 1833| 133 | 12365 | 64456 | 
| 32 | 7 testing/python/metafunc.py | 1771 | 1789| 118 | 12483 | 64456 | 
| 33 | 7 testing/python/metafunc.py | 1748 | 1769| 166 | 12649 | 64456 | 
| 34 | 7 testing/python/collect.py | 495 | 514| 162 | 12811 | 64456 | 
| 35 | 7 testing/python/collect.py | 412 | 439| 199 | 13010 | 64456 | 
| 36 | 7 testing/python/metafunc.py | 1336 | 1368| 211 | 13221 | 64456 | 
| 37 | 7 testing/python/collect.py | 963 | 1002| 364 | 13585 | 64456 | 
| 38 | 7 testing/python/metafunc.py | 930 | 970| 319 | 13904 | 64456 | 
| 39 | 7 testing/python/metafunc.py | 1269 | 1286| 180 | 14084 | 64456 | 
| 40 | 7 testing/python/collect.py | 292 | 344| 402 | 14486 | 64456 | 
| 41 | 7 testing/python/metafunc.py | 193 | 227| 298 | 14784 | 64456 | 
| 42 | 7 testing/python/metafunc.py | 1602 | 1624| 175 | 14959 | 64456 | 
| 43 | 7 testing/python/metafunc.py | 1196 | 1214| 129 | 15088 | 64456 | 
| 44 | 7 testing/python/metafunc.py | 1626 | 1643| 158 | 15246 | 64456 | 
| 45 | 8 src/pytest/__init__.py | 76 | 148| 377 | 15623 | 65493 | 
| 46 | 8 testing/python/metafunc.py | 1110 | 1123| 149 | 15772 | 65493 | 
| 47 | 9 src/_pytest/pytester.py | 1 | 86| 522 | 16294 | 81209 | 
| **-> 48 <-** | **10 src/_pytest/mark/expression.py** | 1 | 69| 386 | 16680 | 82655 | 
| 49 | 10 testing/python/fixtures.py | 1978 | 2029| 316 | 16996 | 82655 | 
| 50 | 10 testing/python/metafunc.py | 726 | 756| 239 | 17235 | 82655 | 
| 51 | 11 src/_pytest/skipping.py | 46 | 82| 383 | 17618 | 84946 | 
| 52 | 11 testing/python/metafunc.py | 854 | 873| 156 | 17774 | 84946 | 
| 53 | 11 testing/python/fixtures.py | 3976 | 4462| 3250 | 21024 | 84946 | 
| 54 | 11 testing/python/collect.py | 1299 | 1322| 207 | 21231 | 84946 | 
| 55 | 12 src/_pytest/main.py | 53 | 172| 771 | 22002 | 91869 | 
| 56 | 13 src/_pytest/compat.py | 195 | 234| 274 | 22276 | 94903 | 
| 57 | 13 src/_pytest/main.py | 173 | 229| 400 | 22676 | 94903 | 
| 58 | 13 testing/python/integration.py | 1 | 42| 296 | 22972 | 94903 | 
| 59 | 13 testing/python/collect.py | 574 | 591| 188 | 23160 | 94903 | 
| 60 | 13 testing/python/metafunc.py | 1567 | 1599| 231 | 23391 | 94903 | 
| 61 | 13 testing/python/metafunc.py | 1149 | 1170| 163 | 23554 | 94903 | 
| 62 | 14 doc/en/example/assertion/failure_demo.py | 1 | 39| 163 | 23717 | 96552 | 
| 63 | 14 testing/python/metafunc.py | 780 | 797| 140 | 23857 | 96552 | 
| 64 | 15 testing/example_scripts/issue_519.py | 1 | 33| 362 | 24219 | 97026 | 
| 65 | 15 testing/python/collect.py | 558 | 572| 156 | 24375 | 97026 | 
| 66 | 16 src/_pytest/_code/code.py | 1 | 54| 348 | 24723 | 106966 | 
| 67 | 17 src/_pytest/python.py | 1349 | 1368| 177 | 24900 | 120934 | 
| 68 | 17 src/_pytest/python.py | 1293 | 1303| 125 | 25025 | 120934 | 
| 69 | 17 src/_pytest/python.py | 130 | 162| 330 | 25355 | 120934 | 
| 70 | 17 testing/python/metafunc.py | 901 | 928| 222 | 25577 | 120934 | 
| 71 | 17 testing/python/collect.py | 928 | 960| 300 | 25877 | 120934 | 
| 72 | 17 testing/python/metafunc.py | 818 | 835| 136 | 26013 | 120934 | 
| 73 | 17 testing/python/metafunc.py | 1238 | 1267| 211 | 26224 | 120934 | 
| 74 | 17 testing/python/metafunc.py | 758 | 778| 158 | 26382 | 120934 | 
| 75 | 17 testing/python/fixtures.py | 41 | 993| 6183 | 32565 | 120934 | 
| 76 | 17 testing/python/metafunc.py | 1125 | 1147| 179 | 32744 | 120934 | 
| 77 | 17 testing/python/raises.py | 184 | 213| 260 | 33004 | 120934 | 
| 78 | 18 doc/en/example/xfail_demo.py | 1 | 39| 143 | 33147 | 121078 | 
| 79 | 19 src/_pytest/doctest.py | 581 | 666| 792 | 33939 | 126749 | 
| 80 | 19 testing/python/metafunc.py | 71 | 84| 181 | 34120 | 126749 | 
| 81 | 20 src/_pytest/mark/structures.py | 75 | 101| 195 | 34315 | 131185 | 
| 82 | 20 testing/python/collect.py | 462 | 493| 202 | 34517 | 131185 | 
| 83 | 20 testing/python/metafunc.py | 1394 | 1410| 110 | 34627 | 131185 | 
| 84 | 20 testing/python/collect.py | 1 | 42| 293 | 34920 | 131185 | 
| 85 | 20 testing/python/metafunc.py | 799 | 816| 140 | 35060 | 131185 | 
| 86 | 20 testing/python/metafunc.py | 837 | 852| 123 | 35183 | 131185 | 
| 87 | 20 src/_pytest/pytester.py | 1162 | 1183| 246 | 35429 | 131185 | 
| 88 | 20 testing/python/metafunc.py | 1216 | 1236| 168 | 35597 | 131185 | 
| 89 | 20 src/pytest/__init__.py | 1 | 74| 659 | 36256 | 131185 | 
| 90 | 20 testing/python/metafunc.py | 430 | 442| 138 | 36394 | 131185 | 
| 91 | **20 src/_pytest/mark/expression.py** | 190 | 226| 233 | 36627 | 131185 | 
| 92 | 20 testing/python/metafunc.py | 698 | 724| 227 | 36854 | 131185 | 
| 93 | 20 testing/python/fixtures.py | 995 | 1976| 6180 | 43034 | 131185 | 
| 94 | 20 testing/python/metafunc.py | 247 | 291| 339 | 43373 | 131185 | 
| 95 | 21 src/_pytest/terminal.py | 114 | 226| 781 | 44154 | 142504 | 
| 96 | 21 testing/python/metafunc.py | 1172 | 1194| 162 | 44316 | 142504 | 
| 97 | 21 testing/python/collect.py | 1240 | 1266| 192 | 44508 | 142504 | 
| 98 | 21 testing/python/metafunc.py | 1 | 29| 151 | 44659 | 142504 | 
| 99 | 22 src/_pytest/junitxml.py | 40 | 64| 280 | 44939 | 148250 | 
| 100 | 22 src/_pytest/python.py | 1405 | 1437| 266 | 45205 | 148250 | 
| 101 | 22 src/_pytest/mark/__init__.py | 153 | 184| 241 | 45446 | 148250 | 
| 102 | 22 src/_pytest/mark/__init__.py | 187 | 217| 247 | 45693 | 148250 | 
| 103 | 22 src/_pytest/mark/structures.py | 132 | 153| 253 | 45946 | 148250 | 
| 104 | 22 testing/python/collect.py | 1200 | 1237| 221 | 46167 | 148250 | 
| 105 | 22 testing/python/collect.py | 1378 | 1418| 356 | 46523 | 148250 | 
| 106 | 22 testing/python/collect.py | 1004 | 1023| 184 | 46707 | 148250 | 
| 107 | 22 testing/python/metafunc.py | 86 | 114| 261 | 46968 | 148250 | 
| 108 | 22 src/_pytest/doctest.py | 139 | 173| 292 | 47260 | 148250 | 
| 109 | 22 src/_pytest/python.py | 84 | 127| 306 | 47566 | 148250 | 
| 110 | 22 testing/python/metafunc.py | 1288 | 1306| 149 | 47715 | 148250 | 
| 111 | 22 src/_pytest/mark/structures.py | 155 | 195| 402 | 48117 | 148250 | 
| 112 | 22 src/_pytest/junitxml.py | 443 | 462| 160 | 48277 | 148250 | 
| 113 | 23 src/_pytest/python_api.py | 1 | 40| 215 | 48492 | 156681 | 
| 114 | 23 testing/python/metafunc.py | 566 | 589| 155 | 48647 | 156681 | 
| 115 | 23 src/_pytest/skipping.py | 27 | 43| 113 | 48760 | 156681 | 
| 116 | 24 testing/conftest.py | 139 | 178| 377 | 49137 | 158118 | 
| 117 | 24 testing/python/metafunc.py | 376 | 393| 217 | 49354 | 158118 | 
| 118 | 24 testing/python/metafunc.py | 892 | 899| 139 | 49493 | 158118 | 
| 119 | 24 src/_pytest/python.py | 1306 | 1346| 345 | 49838 | 158118 | 
| 120 | 25 src/_pytest/config/argparsing.py | 1 | 30| 169 | 50007 | 162659 | 
| 121 | 25 src/_pytest/config/argparsing.py | 435 | 471| 382 | 50389 | 162659 | 
| 122 | 25 testing/python/metafunc.py | 1308 | 1334| 212 | 50601 | 162659 | 
| 123 | 25 src/_pytest/pytester.py | 1469 | 1483| 145 | 50746 | 162659 | 
| 124 | 26 src/_pytest/_io/saferepr.py | 1 | 35| 262 | 51008 | 163762 | 
| 125 | 26 src/_pytest/doctest.py | 312 | 378| 605 | 51613 | 163762 | 
| 126 | 26 testing/python/collect.py | 1170 | 1197| 188 | 51801 | 163762 | 
| 127 | 26 testing/python/fixtures.py | 1 | 38| 209 | 52010 | 163762 | 
| 128 | 26 testing/python/collect.py | 683 | 704| 121 | 52131 | 163762 | 
| 129 | 26 src/_pytest/python.py | 1 | 81| 601 | 52732 | 163762 | 
| 130 | 26 testing/python/metafunc.py | 875 | 890| 218 | 52950 | 163762 | 
| 131 | 26 src/_pytest/compat.py | 237 | 270| 290 | 53240 | 163762 | 
| 132 | 26 testing/python/integration.py | 335 | 374| 251 | 53491 | 163762 | 
| 133 | 26 src/_pytest/pytester.py | 1663 | 1736| 701 | 54192 | 163762 | 
| 134 | 26 testing/python/collect.py | 866 | 897| 279 | 54471 | 163762 | 
| 135 | 26 testing/python/metafunc.py | 140 | 191| 480 | 54951 | 163762 | 
| 136 | 26 testing/python/metafunc.py | 1488 | 1503| 118 | 55069 | 163762 | 
| 137 | 27 src/_pytest/outcomes.py | 71 | 120| 348 | 55417 | 165536 | 
| 138 | 27 testing/python/metafunc.py | 654 | 696| 366 | 55783 | 165536 | 
| 139 | 27 testing/python/fixtures.py | 2999 | 3974| 6078 | 61861 | 165536 | 
| 140 | 27 testing/python/metafunc.py | 1465 | 1486| 192 | 62053 | 165536 | 
| 141 | 28 src/_pytest/nodes.py | 1 | 48| 323 | 62376 | 170951 | 
| 142 | 28 testing/python/metafunc.py | 1049 | 1075| 208 | 62584 | 170951 | 
| 143 | 28 testing/python/metafunc.py | 1505 | 1530| 205 | 62789 | 170951 | 
| 144 | 28 src/_pytest/mark/structures.py | 46 | 72| 218 | 63007 | 170951 | 
| 145 | 28 testing/python/collect.py | 209 | 262| 342 | 63349 | 170951 | 
| 146 | 29 src/_pytest/runner.py | 159 | 182| 184 | 63533 | 175249 | 
| 147 | 30 src/_pytest/fixtures.py | 1 | 116| 777 | 64310 | 189391 | 
| 148 | 30 testing/python/collect.py | 1421 | 1456| 270 | 64580 | 189391 | 
| 149 | 30 src/_pytest/_code/code.py | 1173 | 1193| 161 | 64741 | 189391 | 
| 150 | 30 src/_pytest/doctest.py | 496 | 532| 299 | 65040 | 189391 | 
| **-> 151 <-** | **30 src/_pytest/mark/expression.py** | 72 | 108| 287 | 65327 | 189391 | 
| 152 | 30 src/_pytest/config/argparsing.py | 418 | 433| 158 | 65485 | 189391 | 
| 153 | 31 src/_pytest/faulthandler.py | 1 | 32| 225 | 65710 | 190133 | 
| 154 | 31 src/_pytest/_code/code.py | 954 | 968| 119 | 65829 | 190133 | 
| 155 | 31 testing/python/collect.py | 1365 | 1375| 122 | 65951 | 190133 | 
| 156 | 31 src/_pytest/mark/__init__.py | 268 | 283| 143 | 66094 | 190133 | 
| 157 | 31 testing/python/integration.py | 44 | 78| 253 | 66347 | 190133 | 
| 158 | 31 testing/python/collect.py | 1077 | 1105| 225 | 66572 | 190133 | 
| 159 | 31 src/_pytest/skipping.py | 245 | 259| 152 | 66724 | 190133 | 
| 160 | 31 src/_pytest/compat.py | 1 | 73| 437 | 67161 | 190133 | 
| 161 | 31 testing/python/metafunc.py | 1077 | 1092| 144 | 67305 | 190133 | 
| 162 | 31 src/_pytest/doctest.py | 66 | 113| 326 | 67631 | 190133 | 
| 163 | 31 testing/python/collect.py | 1325 | 1342| 125 | 67756 | 190133 | 
| 164 | 32 src/_pytest/helpconfig.py | 101 | 131| 234 | 67990 | 192023 | 
| 165 | 32 doc/en/example/assertion/failure_demo.py | 42 | 120| 680 | 68670 | 192023 | 
| 166 | 32 testing/python/collect.py | 1053 | 1075| 194 | 68864 | 192023 | 
| 167 | 32 doc/en/example/assertion/failure_demo.py | 163 | 202| 270 | 69134 | 192023 | 
| 168 | 33 src/_pytest/hookspec.py | 378 | 393| 145 | 69279 | 199080 | 
| 169 | 33 testing/python/metafunc.py | 1025 | 1047| 176 | 69455 | 199080 | 
| 170 | 33 src/_pytest/python.py | 1439 | 1464| 258 | 69713 | 199080 | 
| 171 | 33 src/_pytest/python_api.py | 892 | 926| 370 | 70083 | 199080 | 


## Patch

```diff
diff --git a/src/_pytest/mark/expression.py b/src/_pytest/mark/expression.py
--- a/src/_pytest/mark/expression.py
+++ b/src/_pytest/mark/expression.py
@@ -6,7 +6,7 @@
 expr:       and_expr ('or' and_expr)*
 and_expr:   not_expr ('and' not_expr)*
 not_expr:   'not' not_expr | '(' expr ')' | ident
-ident:      (\w|:|\+|-|\.|\[|\])+
+ident:      (\w|:|\+|-|\.|\[|\]|\\)+
 
 The semantics are:
 
@@ -88,7 +88,7 @@ def lex(self, input: str) -> Iterator[Token]:
                 yield Token(TokenType.RPAREN, ")", pos)
                 pos += 1
             else:
-                match = re.match(r"(:?\w|:|\+|-|\.|\[|\])+", input[pos:])
+                match = re.match(r"(:?\w|:|\+|-|\.|\[|\]|\\)+", input[pos:])
                 if match:
                     value = match.group(0)
                     if value == "or":

```

## Test Patch

```diff
diff --git a/testing/test_mark_expression.py b/testing/test_mark_expression.py
--- a/testing/test_mark_expression.py
+++ b/testing/test_mark_expression.py
@@ -66,6 +66,20 @@ def test_syntax_oddeties(expr: str, expected: bool) -> None:
     assert evaluate(expr, matcher) is expected
 
 
+def test_backslash_not_treated_specially() -> None:
+    r"""When generating nodeids, if the source name contains special characters
+    like a newline, they are escaped into two characters like \n. Therefore, a
+    user will never need to insert a literal newline, only \n (two chars). So
+    mark expressions themselves do not support escaping, instead they treat
+    backslashes as regular identifier characters."""
+    matcher = {r"\nfoo\n"}.__contains__
+
+    assert evaluate(r"\nfoo\n", matcher)
+    assert not evaluate(r"foo", matcher)
+    with pytest.raises(ParseError):
+        evaluate("\nfoo\n", matcher)
+
+
 @pytest.mark.parametrize(
     ("expr", "column", "message"),
     (
@@ -129,6 +143,7 @@ def test_syntax_errors(expr: str, column: int, message: str) -> None:
         ":::",
         "a:::c",
         "a+-b",
+        r"\nhe\\l\lo\n\t\rbye",
         "אבגד",
         "aaאבגדcc",
         "a[bcd]",
@@ -156,7 +171,6 @@ def test_valid_idents(ident: str) -> None:
     "ident",
     (
         "/",
-        "\\",
         "^",
         "*",
         "=",

```


## Code snippets

### 1 - testing/python/metafunc.py:

Start line: 1412, End line: 1436

```python
class TestMetafuncFunctional:

    def test_parametrize_misspelling(self, pytester: Pytester) -> None:
        """#463"""
        pytester.makepyfile(
            """
            import pytest

            @pytest.mark.parametrise("x", range(2))
            def test_foo(x):
                pass
        """
        )
        result = pytester.runpytest("--collectonly")
        result.stdout.fnmatch_lines(
            [
                "collected 0 items / 1 error",
                "",
                "*= ERRORS =*",
                "*_ ERROR collecting test_parametrize_misspelling.py _*",
                "test_parametrize_misspelling.py:3: in <module>",
                '    @pytest.mark.parametrise("x", range(2))',
                "E   Failed: Unknown 'parametrise' mark, did you mean 'parametrize'?",
                "*! Interrupted: 1 error during collection !*",
                "*= no tests collected, 1 error in *",
            ]
        )
```
### 2 - testing/python/collect.py:

Start line: 361, End line: 379

```python
class TestFunction:

    def test_parametrize_with_non_hashable_values(self, pytester: Pytester) -> None:
        """Test parametrization with non-hashable values."""
        pytester.makepyfile(
            """
            archival_mapping = {
                '1.0': {'tag': '1.0'},
                '1.2.2a1': {'tag': 'release-1.2.2a1'},
            }

            import pytest
            @pytest.mark.parametrize('key value'.split(),
                                     archival_mapping.items())
            def test_archival_to_version(key, value):
                assert key in archival_mapping
                assert value == archival_mapping[key]
        """
        )
        rec = pytester.inline_run()
        rec.assertoutcome(passed=2)
```
### 3 - doc/en/conf.py:

Start line: 1, End line: 110

```python
#
# pytest documentation build configuration file, created by
# sphinx-quickstart on Fri Oct  8 17:54:28 2010.
#
# This file is execfile()d with the current directory set to its containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.
# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The full version, including alpha/beta/rc tags.
# The short X.Y version.
import ast
import os
import sys
from typing import List
from typing import TYPE_CHECKING

from _pytest import __version__ as version

if TYPE_CHECKING:
    import sphinx.application


release = ".".join(version.split(".")[:2])

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
# sys.path.insert(0, os.path.abspath('.'))

autodoc_member_order = "bysource"
autodoc_typehints = "description"
todo_include_todos = 1

# -- General configuration -----------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be extensions
# coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    "pallets_sphinx_themes",
    "pygments_pytest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx_removed_in",
    "sphinxcontrib_trio",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix of source filenames.
source_suffix = ".rst"

# The encoding of source files.
# source_encoding = 'utf-8-sig'

# The master toctree document.
master_doc = "contents"

# General information about the project.
project = "pytest"
copyright = "2015–2021, holger krekel and pytest-dev team"


# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
# language = None

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
# today = ''
# Else, today_fmt is used as the format for a strftime call.
# today_fmt = '%B %d, %Y'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = [
    "_build",
    "naming20.rst",
    "test/*",
    "old_*",
    "*attic*",
    "*/attic*",
    "funcargs.rst",
    "setup.rst",
    "example/remoteinterp.rst",
]


# The reST default role (used for this markup: `text`) to use for all documents.
default_role = "literal"

# If true, '()' will be appended to :func: etc. cross-reference text.
# add_function_parentheses = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
add_module_names = False
```
### 4 - testing/python/collect.py:

Start line: 608, End line: 636

```python
class TestFunction:

    def test_parametrize_skip(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            import pytest

            m = pytest.mark.skip('')

            @pytest.mark.parametrize('x', [0, 1, pytest.param(2, marks=m)])
            def test_skip(x):
                assert x < 2
        """
        )
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(["* 2 passed, 1 skipped in *"])

    def test_parametrize_skipif_no_skip(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            import pytest

            m = pytest.mark.skipif('False')

            @pytest.mark.parametrize('x', [0, 1, m(2)])
            def test_skipif_no_skip(x):
                assert x < 2
        """
        )
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(["* 1 failed, 2 passed in *"])
```
### 5 - testing/python/collect.py:

Start line: 653, End line: 681

```python
class TestFunction:

    def test_parametrize_passed(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            import pytest

            m = pytest.mark.xfail('True')

            @pytest.mark.parametrize('x', [0, 1, pytest.param(2, marks=m)])
            def test_xfail(x):
                pass
        """
        )
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(["* 2 passed, 1 xpassed in *"])

    def test_parametrize_xfail_passed(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            import pytest

            m = pytest.mark.xfail('False')

            @pytest.mark.parametrize('x', [0, 1, m(2)])
            def test_passed(x):
                pass
        """
        )
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(["* 3 passed in *"])
```
### 6 - testing/python/collect.py:

Start line: 516, End line: 527

```python
class TestFunction:

    def test_parametrize_with_empty_string_arguments(self, pytester: Pytester) -> None:
        items = pytester.getitems(
            """\
            import pytest

            @pytest.mark.parametrize('v', ('', ' '))
            @pytest.mark.parametrize('w', ('', ' '))
            def test(v, w): ...
            """
        )
        names = {item.name for item in items}
        assert names == {"test[-]", "test[ -]", "test[- ]", "test[ - ]"}
```
### 7 - testing/python/metafunc.py:

Start line: 1710, End line: 1724

```python
class TestMarkersWithParametrization:

    def test_xfail_with_arg_and_kwarg(self, pytester: Pytester) -> None:
        s = """
            import pytest

            @pytest.mark.parametrize(("n", "expected"), [
                (1, 2),
                pytest.param(1, 3, marks=pytest.mark.xfail("True", reason="some bug")),
                (2, 3),
            ])
            def test_increment(n, expected):
                assert n + 1 == expected
        """
        pytester.makepyfile(s)
        reprec = pytester.inline_run()
        reprec.assertoutcome(passed=2, skipped=1)
```
### 8 - testing/python/metafunc.py:

Start line: 116, End line: 138

```python
class TestMetafunc:

    def test_parametrize_bad_scope(self) -> None:
        def func(x):
            pass

        metafunc = self.Metafunc(func)
        with pytest.raises(
            fail.Exception,
            match=r"parametrize\(\) call in func got an unexpected scope value 'doggy'",
        ):
            metafunc.parametrize("x", [1], scope="doggy")  # type: ignore[arg-type]

    def test_parametrize_request_name(self, pytester: Pytester) -> None:
        """Show proper error  when 'request' is used as a parameter name in parametrize (#6183)"""

        def func(request):
            raise NotImplementedError()

        metafunc = self.Metafunc(func)
        with pytest.raises(
            fail.Exception,
            match=r"'request' is a reserved name and cannot be used in @pytest.mark.parametrize",
        ):
            metafunc.parametrize("request", [1])
```
### 9 - testing/python/metafunc.py:

Start line: 1694, End line: 1708

```python
class TestMarkersWithParametrization:

    def test_xfail_with_kwarg(self, pytester: Pytester) -> None:
        s = """
            import pytest

            @pytest.mark.parametrize(("n", "expected"), [
                (1, 2),
                pytest.param(1, 3, marks=pytest.mark.xfail(reason="some bug")),
                (2, 3),
            ])
            def test_increment(n, expected):
                assert n + 1 == expected
        """
        pytester.makepyfile(s)
        reprec = pytester.inline_run()
        reprec.assertoutcome(passed=2, skipped=1)
```
### 10 - testing/python/collect.py:

Start line: 638, End line: 651

```python
class TestFunction:

    def test_parametrize_xfail(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            import pytest

            m = pytest.mark.xfail('True')

            @pytest.mark.parametrize('x', [0, 1, pytest.param(2, marks=m)])
            def test_xfail(x):
                assert x < 2
        """
        )
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(["* 2 passed, 1 xfailed in *"])
```
### 48 - src/_pytest/mark/expression.py:

Start line: 1, End line: 69

```python
r"""Evaluate match expressions, as used by `-k` and `-m`.

The grammar is:

expression: expr? EOF
expr:       and_expr ('or' and_expr)*
and_expr:   not_expr ('and' not_expr)*
not_expr:   'not' not_expr | '(' expr ')' | ident
ident:      (\w|:|\+|-|\.|\[|\])+

The semantics are:

- Empty expression evaluates to False.
- ident evaluates to True of False according to a provided matcher function.
- or/and/not evaluate according to the usual boolean semantics.
"""
import ast
import enum
import re
import types
from typing import Callable
from typing import Iterator
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import TYPE_CHECKING

import attr

if TYPE_CHECKING:
    from typing import NoReturn


__all__ = [
    "Expression",
    "ParseError",
]


class TokenType(enum.Enum):
    LPAREN = "left parenthesis"
    RPAREN = "right parenthesis"
    OR = "or"
    AND = "and"
    NOT = "not"
    IDENT = "identifier"
    EOF = "end of input"


@attr.s(frozen=True, slots=True)
class Token:
    type = attr.ib(type=TokenType)
    value = attr.ib(type=str)
    pos = attr.ib(type=int)


class ParseError(Exception):
    """The expression contains invalid syntax.

    :param column: The column in the line where the error occurred (1-based).
    :param message: A description of the error.
    """

    def __init__(self, column: int, message: str) -> None:
        self.column = column
        self.message = message

    def __str__(self) -> str:
        return f"at column {self.column}: {self.message}"
```
### 91 - src/_pytest/mark/expression.py:

Start line: 190, End line: 226

```python
class Expression:
    """A compiled match expression as used by -k and -m.

    The expression can be evaulated against different matchers.
    """

    __slots__ = ("code",)

    def __init__(self, code: types.CodeType) -> None:
        self.code = code

    @classmethod
    def compile(self, input: str) -> "Expression":
        """Compile a match expression.

        :param input: The input expression - one line.
        """
        astexpr = expression(Scanner(input))
        code: types.CodeType = compile(
            astexpr,
            filename="<pytest match expression>",
            mode="eval",
        )
        return Expression(code)

    def evaluate(self, matcher: Callable[[str], bool]) -> bool:
        """Evaluate the match expression.

        :param matcher:
            Given an identifier, should return whether it matches or not.
            Should be prepared to handle arbitrary strings as input.

        :returns: Whether the expression matches or not.
        """
        ret: bool = eval(self.code, {"__builtins__": {}}, MatcherAdapter(matcher))
        return ret
```
### 151 - src/_pytest/mark/expression.py:

Start line: 72, End line: 108

```python
class Scanner:
    __slots__ = ("tokens", "current")

    def __init__(self, input: str) -> None:
        self.tokens = self.lex(input)
        self.current = next(self.tokens)

    def lex(self, input: str) -> Iterator[Token]:
        pos = 0
        while pos < len(input):
            if input[pos] in (" ", "\t"):
                pos += 1
            elif input[pos] == "(":
                yield Token(TokenType.LPAREN, "(", pos)
                pos += 1
            elif input[pos] == ")":
                yield Token(TokenType.RPAREN, ")", pos)
                pos += 1
            else:
                match = re.match(r"(:?\w|:|\+|-|\.|\[|\])+", input[pos:])
                if match:
                    value = match.group(0)
                    if value == "or":
                        yield Token(TokenType.OR, value, pos)
                    elif value == "and":
                        yield Token(TokenType.AND, value, pos)
                    elif value == "not":
                        yield Token(TokenType.NOT, value, pos)
                    else:
                        yield Token(TokenType.IDENT, value, pos)
                    pos += len(value)
                else:
                    raise ParseError(
                        pos + 1,
                        f'unexpected character "{input[pos]}"',
                    )
        yield Token(TokenType.EOF, "", pos)
```
