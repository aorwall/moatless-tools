# pytest-dev__pytest-9249

| **pytest-dev/pytest** | `1824349f74298112722396be6f84a121bc9d6d63` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | - |
| **Avg pos** | - |
| **Min pos** | - |
| **Max pos** | - |
| **Top file pos** | - |
| **Missing snippets** | 2 |
| **Missing patch files** | 1 |


## Expected patch

```diff
diff --git a/src/_pytest/mark/expression.py b/src/_pytest/mark/expression.py
--- a/src/_pytest/mark/expression.py
+++ b/src/_pytest/mark/expression.py
@@ -6,7 +6,7 @@
 expr:       and_expr ('or' and_expr)*
 and_expr:   not_expr ('and' not_expr)*
 not_expr:   'not' not_expr | '(' expr ')' | ident
-ident:      (\w|:|\+|-|\.|\[|\]|\\)+
+ident:      (\w|:|\+|-|\.|\[|\]|\\|/)+
 
 The semantics are:
 
@@ -88,7 +88,7 @@ def lex(self, input: str) -> Iterator[Token]:
                 yield Token(TokenType.RPAREN, ")", pos)
                 pos += 1
             else:
-                match = re.match(r"(:?\w|:|\+|-|\.|\[|\]|\\)+", input[pos:])
+                match = re.match(r"(:?\w|:|\+|-|\.|\[|\]|\\|/)+", input[pos:])
                 if match:
                     value = match.group(0)
                     if value == "or":

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| src/_pytest/mark/expression.py | 9 | 9 | - | - | -
| src/_pytest/mark/expression.py | 91 | 91 | - | - | -


## Problem Statement

```
test ids with `/`s cannot be selected with `-k`
By default pytest 6.2.2 parametrize does user arguments to generate IDs, but some of these ids cannot be used with `-k` option because you endup with errors like  `unexpected character "/"` when trying to do so.

The solution for this bug is to assure that auto-generated IDs are sanitized so they can be used with -k option.

Example:
\`\`\`
@pytest.mark.parametrize(
    ('path', 'kind'),
    (
        ("foo/playbook.yml", "playbook"),
    ),
)
def test_auto_detect(path: str, kind: FileType) -> None:
   ...
\`\`\`

As you can see the first parameter includes a slash, and for good reasons. It is far from practical to have to add custom "ids" for all of these, as you can have LOTS of them.

There is another annoyance related to the -k selecting for parameterized tests, is the fact that square braces `[]` have special meanings for some shells and in order to use it you must remember to quote the strings. It would be much easier if the display and selecting of parametrized tests would use only shell-safe format, so we can easily copy/paste a failed test in run it. For example I think that using colon would be safe and arguably even easier to read: `test_name:param1:param2`.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 testing/python/integration.py | 427 | 449| 146 | 146 | 3126 | 
| 2 | 1 testing/python/integration.py | 451 | 478| 166 | 312 | 3126 | 
| 3 | 2 testing/python/metafunc.py | 591 | 617| 238 | 550 | 17784 | 
| 4 | 3 testing/python/collect.py | 575 | 592| 188 | 738 | 28464 | 
| 5 | 3 testing/python/metafunc.py | 1835 | 1871| 265 | 1003 | 28464 | 
| 6 | 3 testing/python/metafunc.py | 1172 | 1194| 162 | 1165 | 28464 | 
| 7 | 4 testing/python/fixtures.py | 2840 | 3845| 6143 | 7308 | 57333 | 
| 8 | 4 testing/python/metafunc.py | 193 | 227| 298 | 7606 | 57333 | 
| 9 | 4 testing/python/metafunc.py | 1873 | 1908| 296 | 7902 | 57333 | 
| 10 | 4 testing/python/metafunc.py | 1771 | 1789| 118 | 8020 | 57333 | 
| 11 | 4 testing/python/metafunc.py | 1196 | 1214| 129 | 8149 | 57333 | 
| 12 | 4 testing/python/metafunc.py | 1412 | 1436| 214 | 8363 | 57333 | 
| 13 | 4 testing/python/metafunc.py | 1567 | 1599| 231 | 8594 | 57333 | 
| 14 | 4 testing/python/metafunc.py | 1816 | 1833| 133 | 8727 | 57333 | 
| 15 | 4 testing/python/metafunc.py | 1269 | 1286| 180 | 8907 | 57333 | 
| 16 | 4 testing/python/fixtures.py | 41 | 993| 6183 | 15090 | 57333 | 
| 17 | 4 testing/python/fixtures.py | 995 | 1964| 6120 | 21210 | 57333 | 
| 18 | 4 testing/python/metafunc.py | 1288 | 1306| 149 | 21359 | 57333 | 
| 19 | 5 doc/en/conf.py | 1 | 115| 838 | 22197 | 60805 | 
| 20 | 5 testing/python/metafunc.py | 619 | 637| 225 | 22422 | 60805 | 
| 21 | 5 testing/python/metafunc.py | 1216 | 1236| 168 | 22590 | 60805 | 
| 22 | 5 testing/python/metafunc.py | 1710 | 1724| 138 | 22728 | 60805 | 
| 23 | 5 testing/python/metafunc.py | 444 | 458| 148 | 22876 | 60805 | 
| 24 | 5 testing/python/metafunc.py | 376 | 393| 217 | 23093 | 60805 | 
| 25 | 5 testing/python/metafunc.py | 229 | 245| 197 | 23290 | 60805 | 
| 26 | 5 testing/python/collect.py | 609 | 637| 214 | 23504 | 60805 | 
| 27 | 5 testing/python/fixtures.py | 1966 | 2838| 5920 | 29424 | 60805 | 
| 28 | 6 src/_pytest/pytester.py | 1 | 85| 519 | 29943 | 76634 | 
| 29 | 6 testing/python/metafunc.py | 116 | 138| 190 | 30133 | 76634 | 
| 30 | 7 src/_pytest/python.py | 1375 | 1394| 177 | 30310 | 90850 | 
| 31 | 7 testing/python/metafunc.py | 531 | 564| 226 | 30536 | 90850 | 
| 32 | 7 testing/python/metafunc.py | 1694 | 1708| 133 | 30669 | 90850 | 
| 33 | 7 testing/python/metafunc.py | 354 | 374| 187 | 30856 | 90850 | 
| 34 | 7 testing/python/metafunc.py | 1626 | 1643| 158 | 31014 | 90850 | 
| 35 | 7 testing/python/metafunc.py | 1748 | 1769| 166 | 31180 | 90850 | 
| 36 | 7 testing/python/metafunc.py | 1439 | 1463| 165 | 31345 | 90850 | 
| 37 | 7 testing/python/metafunc.py | 1238 | 1267| 211 | 31556 | 90850 | 
| 38 | 7 testing/python/metafunc.py | 247 | 291| 339 | 31895 | 90850 | 
| 39 | 7 testing/python/collect.py | 654 | 682| 202 | 32097 | 90850 | 
| 40 | 7 testing/python/metafunc.py | 1488 | 1503| 118 | 32215 | 90850 | 
| 41 | 8 src/_pytest/fixtures.py | 1139 | 1162| 139 | 32354 | 104975 | 
| 42 | 9 src/_pytest/mark/__init__.py | 74 | 113| 357 | 32711 | 107026 | 
| 43 | 9 testing/python/collect.py | 362 | 380| 158 | 32869 | 107026 | 
| 44 | 9 testing/python/collect.py | 442 | 461| 133 | 33002 | 107026 | 
| 45 | 9 src/_pytest/mark/__init__.py | 47 | 71| 216 | 33218 | 107026 | 
| 46 | 10 src/_pytest/main.py | 53 | 172| 771 | 33989 | 114010 | 
| 47 | 10 testing/python/metafunc.py | 71 | 84| 181 | 34170 | 114010 | 
| 48 | 10 testing/python/metafunc.py | 1505 | 1530| 205 | 34375 | 114010 | 
| 49 | 10 testing/python/metafunc.py | 1662 | 1676| 118 | 34493 | 114010 | 
| 50 | 10 testing/python/metafunc.py | 496 | 529| 231 | 34724 | 114010 | 
| 51 | 10 src/_pytest/main.py | 173 | 229| 400 | 35124 | 114010 | 
| 52 | 10 testing/python/collect.py | 382 | 411| 202 | 35326 | 114010 | 
| 53 | 10 testing/python/collect.py | 496 | 515| 162 | 35488 | 114010 | 
| 54 | 10 testing/python/collect.py | 594 | 607| 111 | 35599 | 114010 | 
| 55 | 10 testing/python/metafunc.py | 1678 | 1692| 130 | 35729 | 114010 | 
| 56 | 10 testing/python/metafunc.py | 1465 | 1486| 192 | 35921 | 114010 | 
| 57 | 10 src/_pytest/python.py | 1177 | 1212| 338 | 36259 | 114010 | 
| 58 | 10 src/_pytest/python.py | 1397 | 1428| 277 | 36536 | 114010 | 
| 59 | 10 testing/python/collect.py | 517 | 528| 110 | 36646 | 114010 | 
| 60 | 10 testing/python/collect.py | 639 | 652| 112 | 36758 | 114010 | 
| 61 | 10 testing/python/metafunc.py | 430 | 442| 138 | 36896 | 114010 | 
| 62 | 10 testing/python/metafunc.py | 1645 | 1660| 135 | 37031 | 114010 | 
| 63 | 10 src/_pytest/python.py | 1332 | 1372| 343 | 37374 | 114010 | 
| 64 | 10 testing/python/metafunc.py | 930 | 970| 319 | 37693 | 114010 | 
| 65 | 11 src/pytest/__init__.py | 80 | 156| 399 | 38092 | 115106 | 
| 66 | 11 testing/python/collect.py | 413 | 440| 199 | 38291 | 115106 | 
| 67 | 11 testing/python/metafunc.py | 566 | 589| 155 | 38446 | 115106 | 
| 68 | 11 testing/python/metafunc.py | 1110 | 1123| 149 | 38595 | 115106 | 
| 69 | 11 testing/python/metafunc.py | 1336 | 1368| 211 | 38806 | 115106 | 
| 70 | 11 testing/python/integration.py | 44 | 78| 252 | 39058 | 115106 | 
| 71 | 11 src/_pytest/python.py | 1 | 83| 601 | 39659 | 115106 | 
| 72 | 11 testing/python/metafunc.py | 1726 | 1746| 203 | 39862 | 115106 | 
| 73 | 11 testing/python/integration.py | 1 | 42| 295 | 40157 | 115106 | 
| 74 | 11 testing/python/collect.py | 293 | 345| 400 | 40557 | 115106 | 
| 75 | 11 testing/python/metafunc.py | 313 | 340| 221 | 40778 | 115106 | 
| 76 | 11 testing/python/metafunc.py | 1791 | 1814| 230 | 41008 | 115106 | 
| 77 | 11 testing/python/metafunc.py | 1602 | 1624| 175 | 41183 | 115106 | 
| 78 | 11 src/_pytest/python.py | 1086 | 1142| 596 | 41779 | 115106 | 
| 79 | 11 testing/python/collect.py | 559 | 573| 156 | 41935 | 115106 | 
| 80 | 12 doc/en/example/xfail_demo.py | 1 | 39| 143 | 42078 | 115250 | 
| 81 | 12 testing/python/metafunc.py | 86 | 114| 261 | 42339 | 115250 | 
| 82 | 12 src/pytest/__init__.py | 1 | 78| 696 | 43035 | 115250 | 
| 83 | 13 src/_pytest/skipping.py | 46 | 82| 382 | 43417 | 117533 | 
| 84 | 13 testing/python/metafunc.py | 479 | 494| 116 | 43533 | 117533 | 
| 85 | 13 testing/python/collect.py | 1231 | 1268| 221 | 43754 | 117533 | 
| 86 | 14 testing/example_scripts/issue_519.py | 1 | 33| 362 | 44116 | 118007 | 
| 87 | 14 testing/python/metafunc.py | 1149 | 1170| 163 | 44279 | 118007 | 
| 88 | 14 testing/python/collect.py | 707 | 720| 124 | 44403 | 118007 | 
| 89 | 14 src/_pytest/python.py | 132 | 164| 330 | 44733 | 118007 | 
| 90 | 14 testing/python/metafunc.py | 342 | 352| 148 | 44881 | 118007 | 
| 91 | 14 testing/python/metafunc.py | 698 | 724| 227 | 45108 | 118007 | 
| 92 | 14 src/_pytest/python.py | 86 | 129| 306 | 45414 | 118007 | 
| 93 | 14 testing/python/metafunc.py | 460 | 477| 129 | 45543 | 118007 | 
| 94 | 14 testing/python/metafunc.py | 875 | 890| 218 | 45761 | 118007 | 
| 95 | 14 testing/python/metafunc.py | 901 | 928| 222 | 45983 | 118007 | 
| 96 | 14 testing/python/collect.py | 1201 | 1228| 188 | 46171 | 118007 | 
| 97 | 14 testing/python/metafunc.py | 1532 | 1565| 264 | 46435 | 118007 | 
| 98 | 14 testing/python/metafunc.py | 854 | 873| 156 | 46591 | 118007 | 
| 99 | 14 testing/python/metafunc.py | 1394 | 1410| 110 | 46701 | 118007 | 
| 100 | 14 testing/python/collect.py | 347 | 360| 114 | 46815 | 118007 | 
| 101 | 14 src/_pytest/python.py | 1431 | 1463| 266 | 47081 | 118007 | 
| 102 | 14 testing/python/metafunc.py | 1 | 29| 151 | 47232 | 118007 | 
| 103 | 14 testing/python/metafunc.py | 726 | 756| 239 | 47471 | 118007 | 
| 104 | 15 src/_pytest/_code/code.py | 1 | 56| 357 | 47828 | 127933 | 
| 105 | 15 testing/python/collect.py | 463 | 494| 202 | 48030 | 127933 | 
| 106 | 16 src/_pytest/deprecated.py | 89 | 139| 435 | 48465 | 129112 | 
| 107 | 16 testing/python/metafunc.py | 780 | 797| 140 | 48605 | 129112 | 
| 108 | 16 testing/python/metafunc.py | 892 | 899| 139 | 48744 | 129112 | 
| 109 | 16 src/_pytest/skipping.py | 27 | 43| 113 | 48857 | 129112 | 
| 110 | 17 src/_pytest/mark/structures.py | 75 | 101| 195 | 49052 | 133701 | 
| 111 | 17 testing/python/metafunc.py | 818 | 835| 136 | 49188 | 133701 | 
| 112 | 17 src/_pytest/python.py | 919 | 963| 364 | 49552 | 133701 | 
| 113 | 17 src/_pytest/mark/structures.py | 155 | 195| 402 | 49954 | 133701 | 
| 114 | 18 src/_pytest/terminal.py | 114 | 226| 781 | 50735 | 145012 | 
| 115 | 18 src/_pytest/mark/__init__.py | 187 | 217| 247 | 50982 | 145012 | 
| 116 | 18 testing/python/metafunc.py | 654 | 696| 366 | 51348 | 145012 | 
| 117 | 18 testing/python/metafunc.py | 799 | 816| 140 | 51488 | 145012 | 
| 118 | 19 src/_pytest/doctest.py | 1 | 63| 425 | 51913 | 150670 | 
| 119 | 19 testing/python/collect.py | 994 | 1033| 364 | 52277 | 150670 | 
| 120 | 19 testing/python/collect.py | 959 | 991| 300 | 52577 | 150670 | 
| 121 | 20 doc/en/example/assertion/failure_demo.py | 1 | 39| 163 | 52740 | 152319 | 
| 122 | 21 src/_pytest/compat.py | 190 | 229| 274 | 53014 | 155328 | 
| 123 | 21 testing/python/metafunc.py | 395 | 428| 283 | 53297 | 155328 | 
| 124 | 21 src/_pytest/fixtures.py | 1 | 116| 777 | 54074 | 155328 | 
| 125 | 21 src/_pytest/pytester.py | 1178 | 1199| 247 | 54321 | 155328 | 
| 126 | 21 testing/python/metafunc.py | 837 | 852| 123 | 54444 | 155328 | 
| 127 | 21 src/_pytest/mark/structures.py | 132 | 153| 253 | 54697 | 155328 | 
| 128 | 22 src/_pytest/config/__init__.py | 1 | 82| 496 | 55193 | 168534 | 
| 129 | 23 src/_pytest/faulthandler.py | 1 | 32| 225 | 55418 | 169276 | 
| 130 | 24 src/_pytest/stepwise.py | 1 | 36| 218 | 55636 | 170191 | 
| 131 | 25 testing/python/raises.py | 215 | 232| 205 | 55841 | 172418 | 
| 132 | 25 src/_pytest/fixtures.py | 1230 | 1245| 114 | 55955 | 172418 | 
| 133 | 25 testing/python/metafunc.py | 1125 | 1147| 179 | 56134 | 172418 | 
| 134 | 25 testing/python/metafunc.py | 140 | 191| 480 | 56614 | 172418 | 
| 135 | 25 testing/python/metafunc.py | 758 | 778| 158 | 56772 | 172418 | 
| 136 | 25 testing/python/metafunc.py | 1308 | 1334| 212 | 56984 | 172418 | 
| 137 | 25 testing/python/collect.py | 1 | 43| 300 | 57284 | 172418 | 
| 138 | 25 src/_pytest/config/__init__.py | 201 | 266| 387 | 57671 | 172418 | 
| 139 | 25 src/_pytest/doctest.py | 66 | 113| 326 | 57997 | 172418 | 
| 140 | 26 src/_pytest/nodes.py | 1 | 49| 336 | 58333 | 177875 | 
| 141 | 26 testing/python/collect.py | 1409 | 1449| 356 | 58689 | 177875 | 
| 142 | 26 src/_pytest/pytester.py | 664 | 750| 717 | 59406 | 177875 | 
| 143 | 26 testing/python/raises.py | 184 | 213| 260 | 59666 | 177875 | 
| 144 | 26 testing/python/fixtures.py | 1 | 38| 209 | 59875 | 177875 | 
| 145 | 26 src/_pytest/fixtures.py | 1559 | 1606| 405 | 60280 | 177875 | 
| 146 | 27 testing/conftest.py | 139 | 178| 377 | 60657 | 179312 | 
| 147 | 27 src/_pytest/skipping.py | 245 | 259| 152 | 60809 | 179312 | 
| 148 | 27 src/_pytest/pytester.py | 1675 | 1748| 688 | 61497 | 179312 | 
| 149 | 27 src/_pytest/doctest.py | 578 | 663| 786 | 62283 | 179312 | 
| 150 | 27 src/_pytest/fixtures.py | 1248 | 1263| 116 | 62399 | 179312 | 
| 151 | 27 src/_pytest/python.py | 1465 | 1490| 258 | 62657 | 179312 | 
| 152 | 27 src/_pytest/pytester.py | 1601 | 1673| 765 | 63422 | 179312 | 
| 153 | 27 testing/python/metafunc.py | 1025 | 1047| 176 | 63598 | 179312 | 
| 154 | 28 src/_pytest/unittest.py | 1 | 41| 254 | 63852 | 182298 | 
| 155 | 28 src/_pytest/doctest.py | 139 | 173| 292 | 64144 | 182298 | 
| 156 | 28 testing/python/collect.py | 822 | 846| 188 | 64332 | 182298 | 
| 157 | 28 src/_pytest/mark/__init__.py | 153 | 184| 246 | 64578 | 182298 | 
| 158 | 28 src/_pytest/doctest.py | 494 | 529| 290 | 64868 | 182298 | 
| 159 | 29 src/_pytest/runner.py | 159 | 182| 184 | 65052 | 186596 | 
| 160 | 30 testing/example_scripts/fixtures/fill_fixtures/test_conftest_funcargs_only_available_in_subdir/sub1/conftest.py | 1 | 8| 0 | 65052 | 186627 | 
| 161 | 31 doc/en/example/multipython.py | 1 | 22| 115 | 65167 | 187066 | 
| 162 | 32 src/_pytest/config/argparsing.py | 1 | 30| 169 | 65336 | 191599 | 
| 163 | 33 src/_pytest/setuponly.py | 1 | 28| 172 | 65508 | 192337 | 
| 164 | 33 src/_pytest/pytester.py | 481 | 493| 138 | 65646 | 192337 | 
| 165 | 33 testing/python/metafunc.py | 1049 | 1075| 208 | 65854 | 192337 | 
| 166 | 34 testing/example_scripts/fixtures/fill_fixtures/test_conftest_funcargs_only_available_in_subdir/sub2/conftest.py | 1 | 7| 0 | 65854 | 192362 | 
| 167 | 34 src/_pytest/unittest.py | 240 | 290| 364 | 66218 | 192362 | 
| 168 | 34 testing/python/metafunc.py | 1370 | 1392| 238 | 66456 | 192362 | 
| 169 | 34 testing/python/metafunc.py | 994 | 1023| 227 | 66683 | 192362 | 
| 170 | 34 testing/python/collect.py | 72 | 90| 208 | 66891 | 192362 | 
| 171 | 34 src/_pytest/config/argparsing.py | 434 | 470| 382 | 67273 | 192362 | 
| 172 | 34 doc/en/example/assertion/failure_demo.py | 163 | 202| 270 | 67543 | 192362 | 
| 173 | 34 src/_pytest/pytester.py | 88 | 123| 237 | 67780 | 192362 | 
| 174 | 35 src/_pytest/config/findpaths.py | 137 | 167| 239 | 68019 | 194068 | 
| 175 | 36 src/_pytest/cacheprovider.py | 445 | 500| 416 | 68435 | 198735 | 


## Missing Patch Files

 * 1: src/_pytest/mark/expression.py

### Hint

```
The test ids are not invalid, keyword expressions are simply not able to express slashes

It's not clear to me if that should be added 
I am not sure either, but I wanted to underline the issue, hoping that we can find a way to improve the UX. The idea is what what we display should also be easily used to run a test or to find the test within the source code.

I was contemplating the idea of using indexes instead of test id.
Actual test ids can be passed as such, no need for keyword expressions 
updated the title to more accurately reflect that the ids aren't invalid, they just can't be selected using `-k`
```

## Patch

```diff
diff --git a/src/_pytest/mark/expression.py b/src/_pytest/mark/expression.py
--- a/src/_pytest/mark/expression.py
+++ b/src/_pytest/mark/expression.py
@@ -6,7 +6,7 @@
 expr:       and_expr ('or' and_expr)*
 and_expr:   not_expr ('and' not_expr)*
 not_expr:   'not' not_expr | '(' expr ')' | ident
-ident:      (\w|:|\+|-|\.|\[|\]|\\)+
+ident:      (\w|:|\+|-|\.|\[|\]|\\|/)+
 
 The semantics are:
 
@@ -88,7 +88,7 @@ def lex(self, input: str) -> Iterator[Token]:
                 yield Token(TokenType.RPAREN, ")", pos)
                 pos += 1
             else:
-                match = re.match(r"(:?\w|:|\+|-|\.|\[|\]|\\)+", input[pos:])
+                match = re.match(r"(:?\w|:|\+|-|\.|\[|\]|\\|/)+", input[pos:])
                 if match:
                     value = match.group(0)
                     if value == "or":

```

## Test Patch

```diff
diff --git a/testing/test_mark.py b/testing/test_mark.py
--- a/testing/test_mark.py
+++ b/testing/test_mark.py
@@ -1111,7 +1111,7 @@ def test_pytest_param_id_allows_none_or_string(s) -> None:
     assert pytest.param(id=s)
 
 
-@pytest.mark.parametrize("expr", ("NOT internal_err", "NOT (internal_err)", "bogus/"))
+@pytest.mark.parametrize("expr", ("NOT internal_err", "NOT (internal_err)", "bogus="))
 def test_marker_expr_eval_failure_handling(pytester: Pytester, expr) -> None:
     foo = pytester.makepyfile(
         """
diff --git a/testing/test_mark_expression.py b/testing/test_mark_expression.py
--- a/testing/test_mark_expression.py
+++ b/testing/test_mark_expression.py
@@ -144,6 +144,7 @@ def test_syntax_errors(expr: str, column: int, message: str) -> None:
         "a:::c",
         "a+-b",
         r"\nhe\\l\lo\n\t\rbye",
+        "a/b",
         "אבגד",
         "aaאבגדcc",
         "a[bcd]",
@@ -170,7 +171,6 @@ def test_valid_idents(ident: str) -> None:
 @pytest.mark.parametrize(
     "ident",
     (
-        "/",
         "^",
         "*",
         "=",

```


## Code snippets

### 1 - testing/python/integration.py:

Start line: 427, End line: 449

```python
class TestParameterize:
    """#351"""

    def test_idfn_marker(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            import pytest

            def idfn(param):
                if param == 0:
                    return 'spam'
                elif param == 1:
                    return 'ham'
                else:
                    return None

            @pytest.mark.parametrize('a,b', [(0, 2), (1, 2)], ids=idfn)
            def test_params(a, b):
                pass
        """
        )
        res = pytester.runpytest("--collect-only")
        res.stdout.fnmatch_lines(["*spam-2*", "*ham-2*"])
```
### 2 - testing/python/integration.py:

Start line: 451, End line: 478

```python
class TestParameterize:

    def test_idfn_fixture(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            import pytest

            def idfn(param):
                if param == 0:
                    return 'spam'
                elif param == 1:
                    return 'ham'
                else:
                    return None

            @pytest.fixture(params=[0, 1], ids=idfn)
            def a(request):
                return request.param

            @pytest.fixture(params=[1, 2], ids=idfn)
            def b(request):
                return request.param

            def test_params(a, b):
                pass
        """
        )
        res = pytester.runpytest("--collect-only")
        res.stdout.fnmatch_lines(["*spam-2*", "*ham-2*"])
```
### 3 - testing/python/metafunc.py:

Start line: 591, End line: 617

```python
class TestMetafunc:

    def test_parametrize_ids_returns_non_string(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """\
            import pytest

            def ids(d):
                return d

            @pytest.mark.parametrize("arg", ({1: 2}, {3, 4}), ids=ids)
            def test(arg):
                assert arg

            @pytest.mark.parametrize("arg", (1, 2.0, True), ids=ids)
            def test_int(arg):
                assert arg
            """
        )
        result = pytester.runpytest("-vv", "-s")
        result.stdout.fnmatch_lines(
            [
                "test_parametrize_ids_returns_non_string.py::test[arg0] PASSED",
                "test_parametrize_ids_returns_non_string.py::test[arg1] PASSED",
                "test_parametrize_ids_returns_non_string.py::test_int[1] PASSED",
                "test_parametrize_ids_returns_non_string.py::test_int[2.0] PASSED",
                "test_parametrize_ids_returns_non_string.py::test_int[True] PASSED",
            ]
        )
```
### 4 - testing/python/collect.py:

Start line: 575, End line: 592

```python
class TestFunction:

    def test_issue751_multiple_parametrize_with_ids(self, pytester: Pytester) -> None:
        modcol = pytester.getmodulecol(
            """
            import pytest
            @pytest.mark.parametrize('x', [0], ids=['c'])
            @pytest.mark.parametrize('y', [0, 1], ids=['a', 'b'])
            class Test(object):
                def test1(self, x, y):
                    pass
                def test2(self, x, y):
                    pass
        """
        )
        colitems = modcol.collect()[0].collect()[0].collect()
        assert colitems[0].name == "test1[a-c]"
        assert colitems[1].name == "test1[b-c]"
        assert colitems[2].name == "test2[a-c]"
        assert colitems[3].name == "test2[b-c]"
```
### 5 - testing/python/metafunc.py:

Start line: 1835, End line: 1871

```python
class TestMarkersWithParametrization:

    def test_pytest_make_parametrize_id_with_argname(self, pytester: Pytester) -> None:
        pytester.makeconftest(
            """
            def pytest_make_parametrize_id(config, val, argname):
                return str(val * 2 if argname == 'x' else val * 10)
        """
        )
        pytester.makepyfile(
            """
                import pytest

                @pytest.mark.parametrize("x", range(2))
                def test_func_a(x):
                    pass

                @pytest.mark.parametrize("y", [1])
                def test_func_b(y):
                    pass
                """
        )
        result = pytester.runpytest("-v")
        result.stdout.fnmatch_lines(
            ["*test_func_a*0*PASS*", "*test_func_a*2*PASS*", "*test_func_b*10*PASS*"]
        )

    def test_parametrize_positional_args(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            import pytest

            @pytest.mark.parametrize("a", [1], False)
            def test_foo(a):
                pass
        """
        )
        result = pytester.runpytest()
        result.assert_outcomes(passed=1)
```
### 6 - testing/python/metafunc.py:

Start line: 1172, End line: 1194

```python
class TestMetafuncFunctional:

    def test_parametrize_with_ids(self, pytester: Pytester) -> None:
        pytester.makeini(
            """
            [pytest]
            console_output_style=classic
        """
        )
        pytester.makepyfile(
            """
            import pytest
            def pytest_generate_tests(metafunc):
                metafunc.parametrize(("a", "b"), [(1,1), (1,2)],
                                     ids=["basic", "advanced"])

            def test_function(a, b):
                assert a == b
        """
        )
        result = pytester.runpytest("-v")
        assert result.ret == 1
        result.stdout.fnmatch_lines_random(
            ["*test_function*basic*PASSED", "*test_function*advanced*FAILED"]
        )
```
### 7 - testing/python/fixtures.py:

Start line: 2840, End line: 3845

```python
class TestFixtureMarker:

    def test_parametrized_fixture_teardown_order(self, pytester: Pytester) -> None:
        pytester.makepyfile(
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
        result = pytester.runpytest("-v")
        result.stdout.fnmatch_lines(
            """
            *3 passed*
        """
        )
        result.stdout.no_fnmatch_line("*error*")

    def test_fixture_finalizer(self, pytester: Pytester) -> None:
        pytester.makeconftest(
            """
            import pytest
            import sys

            @pytest.fixture
            def browser(request):

                def finalize():
                    sys.stdout.write_text('Finalized')
                request.addfinalizer(finalize)
                return {}
        """
        )
        b = pytester.mkdir("subdir")
        b.joinpath("test_overridden_fixture_finalizer.py").write_text(
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
        reprec = pytester.runpytest("-s")
        for test in ["test_browser"]:
            reprec.stdout.fnmatch_lines(["*Finalized*"])

    def test_class_scope_with_normal_tests(self, pytester: Pytester) -> None:
        testpath = pytester.makepyfile(
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
        reprec = pytester.inline_run(testpath)
        for test in ["test_a", "test_b", "test_c"]:
            assert reprec.matchreport(test).passed

    def test_request_is_clean(self, pytester: Pytester) -> None:
        pytester.makepyfile(
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
        reprec = pytester.inline_run("-s")
        values = reprec.getcalls("pytest_runtest_call")[0].item.module.values
        assert values == [1, 2]

    def test_parametrize_separated_lifecycle(self, pytester: Pytester) -> None:
        pytester.makepyfile(
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
        reprec = pytester.inline_run("-vs")
        reprec.assertoutcome(passed=4)
        values = reprec.getcalls("pytest_runtest_call")[0].item.module.values
        import pprint

        pprint.pprint(values)
        # assert len(values) == 6
        assert values[0] == values[1] == 1
        assert values[2] == "fin1"
        assert values[3] == values[4] == 2
        assert values[5] == "fin2"

    def test_parametrize_function_scoped_finalizers_called(
        self, pytester: Pytester
    ) -> None:
        pytester.makepyfile(
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
        reprec = pytester.inline_run("-v")
        reprec.assertoutcome(passed=5)

    @pytest.mark.parametrize("scope", ["session", "function", "module"])
    def test_finalizer_order_on_parametrization(
        self, scope, pytester: Pytester
    ) -> None:
        """#246"""
        pytester.makepyfile(
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
        reprec = pytester.inline_run("-lvs")
        reprec.assertoutcome(passed=3)

    def test_class_scope_parametrization_ordering(self, pytester: Pytester) -> None:
        """#396"""
        pytester.makepyfile(
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
        reprec = pytester.inline_run()
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

    def test_parametrize_setup_function(self, pytester: Pytester) -> None:
        pytester.makepyfile(
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
        reprec = pytester.inline_run("-v")
        reprec.assertoutcome(passed=6)

    def test_fixture_marked_function_not_collected_as_test(
        self, pytester: Pytester
    ) -> None:
        pytester.makepyfile(
            """
            import pytest
            @pytest.fixture
            def test_app():
                return 1

            def test_something(test_app):
                assert test_app == 1
        """
        )
        reprec = pytester.inline_run()
        reprec.assertoutcome(passed=1)

    def test_params_and_ids(self, pytester: Pytester) -> None:
        pytester.makepyfile(
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
        res = pytester.runpytest("-v")
        res.stdout.fnmatch_lines(["*test_foo*alpha*", "*test_foo*beta*"])

    def test_params_and_ids_yieldfixture(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            import pytest

            @pytest.fixture(params=[object(), object()], ids=['alpha', 'beta'])
            def fix(request):
                 yield request.param

            def test_foo(fix):
                assert 1
        """
        )
        res = pytester.runpytest("-v")
        res.stdout.fnmatch_lines(["*test_foo*alpha*", "*test_foo*beta*"])

    def test_deterministic_fixture_collection(
        self, pytester: Pytester, monkeypatch
    ) -> None:
        """#920"""
        pytester.makepyfile(
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
        out1 = pytester.runpytest_subprocess("-v")
        monkeypatch.setenv("PYTHONHASHSEED", "2")
        out2 = pytester.runpytest_subprocess("-v")
        output1 = [
            line
            for line in out1.outlines
            if line.startswith("test_deterministic_fixture_collection.py::test_foo")
        ]
        output2 = [
            line
            for line in out2.outlines
            if line.startswith("test_deterministic_fixture_collection.py::test_foo")
        ]
        assert len(output1) == 12
        assert output1 == output2


class TestRequestScopeAccess:
    pytestmark = pytest.mark.parametrize(
        ("scope", "ok", "error"),
        [
            ["session", "", "path class function module"],
            ["module", "module path", "cls function"],
            ["class", "module path cls", "function"],
            ["function", "module path cls function", ""],
        ],
    )

    def test_setup(self, pytester: Pytester, scope, ok, error) -> None:
        pytester.makepyfile(
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
        reprec = pytester.inline_run("-l")
        reprec.assertoutcome(passed=1)

    def test_funcarg(self, pytester: Pytester, scope, ok, error) -> None:
        pytester.makepyfile(
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
        reprec = pytester.inline_run()
        reprec.assertoutcome(passed=1)


class TestErrors:
    def test_subfactory_missing_funcarg(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            import pytest
            @pytest.fixture()
            def gen(qwe123):
                return 1
            def test_something(gen):
                pass
        """
        )
        result = pytester.runpytest()
        assert result.ret != 0
        result.stdout.fnmatch_lines(
            ["*def gen(qwe123):*", "*fixture*qwe123*not found*", "*1 error*"]
        )

    def test_issue498_fixture_finalizer_failing(self, pytester: Pytester) -> None:
        pytester.makepyfile(
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
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(
            """
            *ERROR*teardown*test_1*
            *KeyError*
            *ERROR*teardown*test_2*
            *KeyError*
            *3 pass*2 errors*
        """
        )

    def test_setupfunc_missing_funcarg(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            import pytest
            @pytest.fixture(autouse=True)
            def gen(qwe123):
                return 1
            def test_something():
                pass
        """
        )
        result = pytester.runpytest()
        assert result.ret != 0
        result.stdout.fnmatch_lines(
            ["*def gen(qwe123):*", "*fixture*qwe123*not found*", "*1 error*"]
        )


class TestShowFixtures:
    def test_funcarg_compat(self, pytester: Pytester) -> None:
        config = pytester.parseconfigure("--funcargs")
        assert config.option.showfixtures

    def test_show_fixtures(self, pytester: Pytester) -> None:
        result = pytester.runpytest("--fixtures")
        result.stdout.fnmatch_lines(
            [
                "tmp_path_factory [[]session scope[]] -- *tmpdir.py*",
                "*for the test session*",
                "tmp_path -- *",
                "*temporary directory*",
            ]
        )

    def test_show_fixtures_verbose(self, pytester: Pytester) -> None:
        result = pytester.runpytest("--fixtures", "-v")
        result.stdout.fnmatch_lines(
            [
                "tmp_path_factory [[]session scope[]] -- *tmpdir.py*",
                "*for the test session*",
                "tmp_path -- *tmpdir.py*",
                "*temporary directory*",
            ]
        )

    def test_show_fixtures_testmodule(self, pytester: Pytester) -> None:
        p = pytester.makepyfile(
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
        result = pytester.runpytest("--fixtures", p)
        result.stdout.fnmatch_lines(
            """
            *tmp_path -- *
            *fixtures defined from*
            *arg1 -- test_show_fixtures_testmodule.py:6*
            *hello world*
        """
        )
        result.stdout.no_fnmatch_line("*arg0*")

    @pytest.mark.parametrize("testmod", [True, False])
    def test_show_fixtures_conftest(self, pytester: Pytester, testmod) -> None:
        pytester.makeconftest(
            '''
            import pytest
            @pytest.fixture
            def arg1():
                """  hello world """
        '''
        )
        if testmod:
            pytester.makepyfile(
                """
                def test_hello():
                    pass
            """
            )
        result = pytester.runpytest("--fixtures")
        result.stdout.fnmatch_lines(
            """
            *tmp_path*
            *fixtures defined from*conftest*
            *arg1*
            *hello world*
        """
        )

    def test_show_fixtures_trimmed_doc(self, pytester: Pytester) -> None:
        p = pytester.makepyfile(
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
        result = pytester.runpytest("--fixtures", p)
        result.stdout.fnmatch_lines(
            textwrap.dedent(
                """\
                * fixtures defined from test_show_fixtures_trimmed_doc *
                arg2 -- test_show_fixtures_trimmed_doc.py:10
                    line1
                    line2
                arg1 -- test_show_fixtures_trimmed_doc.py:3
                    line1
                    line2
                """
            )
        )

    def test_show_fixtures_indented_doc(self, pytester: Pytester) -> None:
        p = pytester.makepyfile(
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
        result = pytester.runpytest("--fixtures", p)
        result.stdout.fnmatch_lines(
            textwrap.dedent(
                """\
                * fixtures defined from test_show_fixtures_indented_doc *
                fixture1 -- test_show_fixtures_indented_doc.py:3
                    line1
                        indented line
                """
            )
        )

    def test_show_fixtures_indented_doc_first_line_unindented(
        self, pytester: Pytester
    ) -> None:
        p = pytester.makepyfile(
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
        result = pytester.runpytest("--fixtures", p)
        result.stdout.fnmatch_lines(
            textwrap.dedent(
                """\
                * fixtures defined from test_show_fixtures_indented_doc_first_line_unindented *
                fixture1 -- test_show_fixtures_indented_doc_first_line_unindented.py:3
                    line1
                    line2
                        indented line
                """
            )
        )

    def test_show_fixtures_indented_in_class(self, pytester: Pytester) -> None:
        p = pytester.makepyfile(
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
        result = pytester.runpytest("--fixtures", p)
        result.stdout.fnmatch_lines(
            textwrap.dedent(
                """\
                * fixtures defined from test_show_fixtures_indented_in_class *
                fixture1 -- test_show_fixtures_indented_in_class.py:4
                    line1
                    line2
                        indented line
                """
            )
        )

    def test_show_fixtures_different_files(self, pytester: Pytester) -> None:
        """`--fixtures` only shows fixtures from first file (#833)."""
        pytester.makepyfile(
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
        pytester.makepyfile(
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
        result = pytester.runpytest("--fixtures")
        result.stdout.fnmatch_lines(
            """
            * fixtures defined from test_a *
            fix_a -- test_a.py:4
                Fixture A

            * fixtures defined from test_b *
            fix_b -- test_b.py:4
                Fixture B
        """
        )

    def test_show_fixtures_with_same_name(self, pytester: Pytester) -> None:
        pytester.makeconftest(
            '''
            import pytest
            @pytest.fixture
            def arg1():
                """Hello World in conftest.py"""
                return "Hello World"
        '''
        )
        pytester.makepyfile(
            """
            def test_foo(arg1):
                assert arg1 == "Hello World"
        """
        )
        pytester.makepyfile(
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
        result = pytester.runpytest("--fixtures")
        result.stdout.fnmatch_lines(
            """
            * fixtures defined from conftest *
            arg1 -- conftest.py:3
                Hello World in conftest.py

            * fixtures defined from test_show_fixtures_with_same_name *
            arg1 -- test_show_fixtures_with_same_name.py:3
                Hi from test module
        """
        )

    def test_fixture_disallow_twice(self):
        """Test that applying @pytest.fixture twice generates an error (#2334)."""
        with pytest.raises(ValueError):

            @pytest.fixture
            @pytest.fixture
            def foo():
                raise NotImplementedError()


class TestContextManagerFixtureFuncs:
    def test_simple(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            import pytest
            @pytest.fixture
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
        result = pytester.runpytest("-s")
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

    def test_scoped(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            import pytest
            @pytest.fixture(scope="module")
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
        result = pytester.runpytest("-s")
        result.stdout.fnmatch_lines(
            """
            *setup*
            *test1 1*
            *test2 1*
            *teardown*
        """
        )

    def test_setup_exception(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            import pytest
            @pytest.fixture(scope="module")
            def arg1():
                pytest.fail("setup")
                yield 1
            def test_1(arg1):
                pass
        """
        )
        result = pytester.runpytest("-s")
        result.stdout.fnmatch_lines(
            """
            *pytest.fail*setup*
            *1 error*
        """
        )

    def test_teardown_exception(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            import pytest
            @pytest.fixture(scope="module")
            def arg1():
                yield 1
                pytest.fail("teardown")
            def test_1(arg1):
                pass
        """
        )
        result = pytester.runpytest("-s")
        result.stdout.fnmatch_lines(
            """
            *pytest.fail*teardown*
            *1 passed*1 error*
        """
        )

    def test_yields_more_than_one(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            import pytest
            @pytest.fixture(scope="module")
            def arg1():
                yield 1
                yield 2
            def test_1(arg1):
                pass
        """
        )
        result = pytester.runpytest("-s")
        result.stdout.fnmatch_lines(
            """
            *fixture function*
            *test_yields*:2*
        """
        )

    def test_custom_name(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            import pytest
            @pytest.fixture(name='meow')
            def arg1():
                return 'mew'
            def test_1(meow):
                print(meow)
        """
        )
        result = pytester.runpytest("-s")
        result.stdout.fnmatch_lines(["*mew*"])


class TestParameterizedSubRequest:
    def test_call_from_fixture(self, pytester: Pytester) -> None:
        pytester.makepyfile(
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
        result = pytester.runpytest()
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

    def test_call_from_test(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            test_call_from_test="""
            import pytest

            @pytest.fixture(params=[0, 1, 2])
            def fix_with_param(request):
                return request.param

            def test_foo(request):
                request.getfixturevalue('fix_with_param')
            """
        )
        result = pytester.runpytest()
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

    def test_external_fixture(self, pytester: Pytester) -> None:
        pytester.makeconftest(
            """
            import pytest

            @pytest.fixture(params=[0, 1, 2])
            def fix_with_param(request):
                return request.param
            """
        )

        pytester.makepyfile(
            test_external_fixture="""
            def test_foo(request):
                request.getfixturevalue('fix_with_param')
            """
        )
        result = pytester.runpytest()
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
```
### 8 - testing/python/metafunc.py:

Start line: 193, End line: 227

```python
class TestMetafunc:

    def test_parametrize_and_id(self) -> None:
        def func(x, y):
            pass

        metafunc = self.Metafunc(func)

        metafunc.parametrize("x", [1, 2], ids=["basic", "advanced"])
        metafunc.parametrize("y", ["abc", "def"])
        ids = [x.id for x in metafunc._calls]
        assert ids == ["basic-abc", "basic-def", "advanced-abc", "advanced-def"]

    def test_parametrize_and_id_unicode(self) -> None:
        """Allow unicode strings for "ids" parameter in Python 2 (##1905)"""

        def func(x):
            pass

        metafunc = self.Metafunc(func)
        metafunc.parametrize("x", [1, 2], ids=["basic", "advanced"])
        ids = [x.id for x in metafunc._calls]
        assert ids == ["basic", "advanced"]

    def test_parametrize_with_wrong_number_of_ids(self) -> None:
        def func(x, y):
            pass

        metafunc = self.Metafunc(func)

        with pytest.raises(fail.Exception):
            metafunc.parametrize("x", [1, 2], ids=["basic"])

        with pytest.raises(fail.Exception):
            metafunc.parametrize(
                ("x", "y"), [("abc", "def"), ("ghi", "jkl")], ids=["one"]
            )
```
### 9 - testing/python/metafunc.py:

Start line: 1873, End line: 1908

```python
class TestMarkersWithParametrization:

    def test_parametrize_iterator(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            import itertools
            import pytest

            id_parametrize = pytest.mark.parametrize(
                ids=("param%d" % i for i in itertools.count())
            )

            @id_parametrize('y', ['a', 'b'])
            def test1(y):
                pass

            @id_parametrize('y', ['a', 'b'])
            def test2(y):
                pass

            @pytest.mark.parametrize("a, b", [(1, 2), (3, 4)], ids=itertools.count())
            def test_converted_to_str(a, b):
                pass
        """
        )
        result = pytester.runpytest("-vv", "-s")
        result.stdout.fnmatch_lines(
            [
                "test_parametrize_iterator.py::test1[param0] PASSED",
                "test_parametrize_iterator.py::test1[param1] PASSED",
                "test_parametrize_iterator.py::test2[param0] PASSED",
                "test_parametrize_iterator.py::test2[param1] PASSED",
                "test_parametrize_iterator.py::test_converted_to_str[0] PASSED",
                "test_parametrize_iterator.py::test_converted_to_str[1] PASSED",
                "*= 6 passed in *",
            ]
        )
```
### 10 - testing/python/metafunc.py:

Start line: 1771, End line: 1789

```python
class TestMarkersWithParametrization:

    def test_parametrize_ID_generation_string_int_works(
        self, pytester: Pytester
    ) -> None:
        """#290"""
        pytester.makepyfile(
            """
            import pytest

            @pytest.fixture
            def myfixture():
                return 'example'
            @pytest.mark.parametrize(
                'limit', (0, '0'))
            def test_limit(limit, myfixture):
                return
        """
        )
        reprec = pytester.inline_run()
        reprec.assertoutcome(passed=2)
```
