# astropy__astropy-13032

| **astropy/astropy** | `d707b792d3ca45518a53b4a395c81ee86bd7b451` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 1414 |
| **Any found context length** | 569 |
| **Avg pos** | 7.0 |
| **Min pos** | 1 |
| **Max pos** | 4 |
| **Top file pos** | 1 |
| **Missing snippets** | 4 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/astropy/modeling/bounding_box.py b/astropy/modeling/bounding_box.py
--- a/astropy/modeling/bounding_box.py
+++ b/astropy/modeling/bounding_box.py
@@ -694,6 +694,12 @@ def _validate_dict(self, bounding_box: dict):
         for key, value in bounding_box.items():
             self[key] = value
 
+    @property
+    def _available_input_index(self):
+        model_input_index = [self._get_index(_input) for _input in self._model.inputs]
+
+        return [_input for _input in model_input_index if _input not in self._ignored]
+
     def _validate_sequence(self, bounding_box, order: str = None):
         """Validate passing tuple of tuples representation (or related) and setting them."""
         order = self._get_order(order)
@@ -703,7 +709,7 @@ def _validate_sequence(self, bounding_box, order: str = None):
             bounding_box = bounding_box[::-1]
 
         for index, value in enumerate(bounding_box):
-            self[index] = value
+            self[self._available_input_index[index]] = value
 
     @property
     def _n_inputs(self) -> int:
@@ -727,7 +733,7 @@ def _validate_iterable(self, bounding_box, order: str = None):
     def _validate(self, bounding_box, order: str = None):
         """Validate and set any representation"""
         if self._n_inputs == 1 and not isinstance(bounding_box, dict):
-            self[0] = bounding_box
+            self[self._available_input_index[0]] = bounding_box
         else:
             self._validate_iterable(bounding_box, order)
 
@@ -751,7 +757,7 @@ def validate(cls, model, bounding_box,
             order = bounding_box.order
             if _preserve_ignore:
                 ignored = bounding_box.ignored
-            bounding_box = bounding_box.intervals
+            bounding_box = bounding_box.named_intervals
 
         new = cls({}, model, ignored=ignored, order=order)
         new._validate(bounding_box)

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| astropy/modeling/bounding_box.py | 697 | 697 | 1 | 1 | 569
| astropy/modeling/bounding_box.py | 706 | 706 | 1 | 1 | 569
| astropy/modeling/bounding_box.py | 730 | 730 | 1 | 1 | 569
| astropy/modeling/bounding_box.py | 754 | 754 | 4 | 1 | 1414


## Problem Statement

```
Incorrect ignored usage in `ModelBoundingBox`
<!-- This comments are hidden when you submit the issue,
so you do not need to remove them! -->

<!-- Please be sure to check out our contributing guidelines,
https://github.com/astropy/astropy/blob/main/CONTRIBUTING.md .
Please be sure to check out our code of conduct,
https://github.com/astropy/astropy/blob/main/CODE_OF_CONDUCT.md . -->

<!-- Please have a search on our GitHub repository to see if a similar
issue has already been posted.
If a similar issue is closed, have a quick look to see if you are satisfied
by the resolution.
If not please go ahead and open an issue! -->

<!-- Please check that the development version still produces the same bug.
You can install development version with
pip install git+https://github.com/astropy/astropy
command. -->

### Description
<!-- Provide a general description of the bug. -->

Providing `ignored` inputs to `ModelBoundingBox` does not always work as expected.

Running the following code:
\`\`\`python
from astropy.modeling.bounding_box import ModelBoundingBox
from astropy.modeling import models as astropy_models

bbox = ModelBoundingBox((9, 10), astropy_models.Polynomial2D(1), ignored=["x"])
print(bbox)
print(bbox.ignored_inputs)
\`\`\`
Produces:
\`\`\`
ModelBoundingBox(
    intervals={
        x: Interval(lower=9, upper=10)
    }
    model=Polynomial2D(inputs=('x', 'y'))
    order='C'
)
[]
\`\`\`
This is incorrect. It instead should produce:
\`\`\`
ModelBoundingBox(
    intervals={
        y: Interval(lower=9, upper=10)
    }
    model=Polynomial2D(inputs=('x', 'y'))
    order='C'
)
['x']
\`\`\`

Somehow the `ignored` status of the `x` input is being accounted for during the validation which occurs during the construction of the bounding box; however, it is getting "lost" somehow resulting in the weird behavior we see above.

Oddly enough ignoring `y` does not have an issue. E.G. this code:
\`\`\`python
from astropy.modeling.bounding_box import ModelBoundingBox
from astropy.modeling import models as astropy_models

bbox = ModelBoundingBox((11, 12), astropy_models.Polynomial2D(1), ignored=["y"])
print(bbox)
print(bbox.ignored_inputs)
\`\`\`
Produces:
\`\`\`
ModelBoundingBox(
    intervals={
        x: Interval(lower=11, upper=12)
    }
    ignored=['y']
    model=Polynomial2D(inputs=('x', 'y'))
    order='C'
)
['y']
\`\`\`
as expected.

### System Details
This is present in both astropy 5.03 and astropy develop


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 astropy/modeling/bounding_box.py** | 667 | 732| 569 | 569 | 10114 | 
| 2 | **1 astropy/modeling/bounding_box.py** | 555 | 606| 396 | 965 | 10114 | 
| 3 | **1 astropy/modeling/bounding_box.py** | 761 | 798| 252 | 1217 | 10114 | 
| **-> 4 <-** | **1 astropy/modeling/bounding_box.py** | 734 | 759| 197 | 1414 | 10114 | 
| 5 | **1 astropy/modeling/bounding_box.py** | 1488 | 1504| 165 | 1579 | 10114 | 
| 6 | **1 astropy/modeling/bounding_box.py** | 608 | 625| 138 | 1717 | 10114 | 
| 7 | **1 astropy/modeling/bounding_box.py** | 627 | 645| 147 | 1864 | 10114 | 
| 8 | **1 astropy/modeling/bounding_box.py** | 1506 | 1542| 265 | 2129 | 10114 | 
| 9 | **1 astropy/modeling/bounding_box.py** | 1332 | 1382| 365 | 2494 | 10114 | 
| 10 | **1 astropy/modeling/bounding_box.py** | 899 | 930| 162 | 2656 | 10114 | 
| 11 | **1 astropy/modeling/bounding_box.py** | 800 | 834| 230 | 2886 | 10114 | 
| 12 | **1 astropy/modeling/bounding_box.py** | 1233 | 1262| 185 | 3071 | 10114 | 
| 13 | **1 astropy/modeling/bounding_box.py** | 932 | 950| 130 | 3201 | 10114 | 
| 14 | **1 astropy/modeling/bounding_box.py** | 862 | 896| 255 | 3456 | 10114 | 
| 15 | **1 astropy/modeling/bounding_box.py** | 1039 | 1104| 330 | 3786 | 10114 | 
| 16 | **1 astropy/modeling/bounding_box.py** | 162 | 262| 693 | 4479 | 10114 | 
| 17 | **1 astropy/modeling/bounding_box.py** | 836 | 860| 175 | 4654 | 10114 | 
| 18 | **1 astropy/modeling/bounding_box.py** | 1466 | 1486| 181 | 4835 | 10114 | 
| 19 | **1 astropy/modeling/bounding_box.py** | 1420 | 1440| 170 | 5005 | 10114 | 
| 20 | **1 astropy/modeling/bounding_box.py** | 1442 | 1464| 170 | 5175 | 10114 | 
| 21 | 2 astropy/modeling/functional_models.py | 2354 | 2380| 235 | 5410 | 38107 | 
| 22 | **2 astropy/modeling/bounding_box.py** | 130 | 159| 222 | 5632 | 38107 | 
| 23 | **2 astropy/modeling/bounding_box.py** | 79 | 127| 298 | 5930 | 38107 | 
| 24 | **2 astropy/modeling/bounding_box.py** | 55 | 77| 205 | 6135 | 38107 | 
| 25 | **2 astropy/modeling/bounding_box.py** | 1140 | 1193| 296 | 6431 | 38107 | 
| 26 | 2 astropy/modeling/functional_models.py | 344 | 393| 436 | 6867 | 38107 | 
| 27 | **2 astropy/modeling/bounding_box.py** | 8 | 53| 206 | 7073 | 38107 | 
| 28 | **2 astropy/modeling/bounding_box.py** | 1106 | 1138| 210 | 7283 | 38107 | 
| 29 | 2 astropy/modeling/functional_models.py | 1968 | 1989| 168 | 7451 | 38107 | 
| 30 | **2 astropy/modeling/bounding_box.py** | 264 | 286| 169 | 7620 | 38107 | 
| 31 | **2 astropy/modeling/bounding_box.py** | 1312 | 1330| 173 | 7793 | 38107 | 
| 32 | 2 astropy/modeling/functional_models.py | 2454 | 2476| 192 | 7985 | 38107 | 
| 33 | **2 astropy/modeling/bounding_box.py** | 1384 | 1418| 297 | 8282 | 38107 | 
| 34 | **2 astropy/modeling/bounding_box.py** | 390 | 429| 296 | 8578 | 38107 | 
| 35 | **2 astropy/modeling/bounding_box.py** | 362 | 388| 194 | 8772 | 38107 | 
| 36 | **2 astropy/modeling/bounding_box.py** | 647 | 665| 156 | 8928 | 38107 | 
| 37 | 2 astropy/modeling/functional_models.py | 125 | 167| 308 | 9236 | 38107 | 
| 38 | 3 astropy/modeling/physical_models.py | 323 | 360| 289 | 9525 | 44583 | 
| 39 | 4 astropy/modeling/core.py | 4068 | 4471| 3343 | 12868 | 80968 | 
| 40 | **4 astropy/modeling/bounding_box.py** | 1265 | 1310| 353 | 13221 | 80968 | 
| 41 | 4 astropy/modeling/core.py | 85 | 880| 6529 | 19750 | 80968 | 
| 42 | **4 astropy/modeling/bounding_box.py** | 328 | 360| 232 | 19982 | 80968 | 
| 43 | 4 astropy/modeling/functional_models.py | 2976 | 2993| 205 | 20187 | 80968 | 
| 44 | **4 astropy/modeling/bounding_box.py** | 308 | 326| 133 | 20320 | 80968 | 
| 45 | 4 astropy/modeling/functional_models.py | 2169 | 2187| 145 | 20465 | 80968 | 
| 46 | 4 astropy/modeling/functional_models.py | 2522 | 2540| 162 | 20627 | 80968 | 
| 47 | **4 astropy/modeling/bounding_box.py** | 466 | 502| 276 | 20903 | 80968 | 
| 48 | 5 astropy/timeseries/periodograms/bls/core.py | 605 | 642| 313 | 21216 | 88573 | 
| 49 | 6 astropy/stats/bayesian_blocks.py | 216 | 280| 462 | 21678 | 93315 | 
| 50 | 7 astropy/modeling/tabular.py | 173 | 207| 294 | 21972 | 96205 | 
| 51 | **7 astropy/modeling/bounding_box.py** | 991 | 1009| 125 | 22097 | 96205 | 
| 52 | 8 astropy/modeling/models.py | 9 | 72| 668 | 22765 | 96920 | 
| 53 | 8 astropy/modeling/functional_models.py | 1991 | 2002| 170 | 22935 | 96920 | 
| 54 | **8 astropy/modeling/bounding_box.py** | 288 | 306| 124 | 23059 | 96920 | 
| 55 | 9 astropy/modeling/fitting.py | 940 | 1027| 744 | 23803 | 112151 | 
| 56 | **9 astropy/modeling/bounding_box.py** | 527 | 552| 226 | 24029 | 112151 | 
| 57 | **9 astropy/modeling/bounding_box.py** | 504 | 525| 121 | 24150 | 112151 | 
| 58 | 10 astropy/io/misc/asdf/tags/transform/functional_models.py | 105 | 115| 133 | 24283 | 119618 | 
| 59 | 10 astropy/modeling/functional_models.py | 3229 | 3250| 170 | 24453 | 119618 | 
| 60 | 10 astropy/modeling/functional_models.py | 1534 | 1561| 226 | 24679 | 119618 | 
| 61 | **10 astropy/modeling/bounding_box.py** | 1011 | 1036| 157 | 24836 | 119618 | 
| 62 | **10 astropy/modeling/bounding_box.py** | 431 | 464| 258 | 25094 | 119618 | 
| 63 | 11 astropy/convolution/utils.py | 162 | 196| 402 | 25496 | 122555 | 
| 64 | 11 astropy/modeling/functional_models.py | 2339 | 2352| 157 | 25653 | 122555 | 
| 65 | **11 astropy/modeling/bounding_box.py** | 1214 | 1231| 116 | 25769 | 122555 | 
| 66 | 11 astropy/modeling/core.py | 1700 | 2461| 6325 | 32094 | 122555 | 
| 67 | 11 astropy/io/misc/asdf/tags/transform/functional_models.py | 54 | 80| 244 | 32338 | 122555 | 
| 68 | 11 astropy/modeling/functional_models.py | 2202 | 2293| 595 | 32933 | 122555 | 
| 69 | **11 astropy/modeling/bounding_box.py** | 952 | 989| 222 | 33155 | 122555 | 
| 70 | 11 astropy/modeling/functional_models.py | 2072 | 2081| 153 | 33308 | 122555 | 
| 71 | 12 astropy/modeling/mappings.py | 128 | 178| 380 | 33688 | 125124 | 
| 72 | 12 astropy/modeling/functional_models.py | 459 | 477| 225 | 33913 | 125124 | 
| 73 | 12 astropy/modeling/fitting.py | 857 | 939| 779 | 34692 | 125124 | 
| 74 | 12 astropy/modeling/functional_models.py | 2189 | 2199| 163 | 34855 | 125124 | 
| 75 | 13 astropy/io/misc/asdf/tags/transform/basic.py | 3 | 67| 476 | 35331 | 127346 | 
| 76 | 13 astropy/modeling/functional_models.py | 288 | 342| 632 | 35963 | 127346 | 
| 77 | 13 astropy/modeling/core.py | 882 | 1698| 6380 | 42343 | 127346 | 
| 78 | 13 astropy/modeling/functional_models.py | 2795 | 2811| 203 | 42546 | 127346 | 
| 79 | 14 astropy/nddata/utils.py | 713 | 798| 726 | 43272 | 134775 | 
| 80 | 14 astropy/modeling/functional_models.py | 2296 | 2337| 331 | 43603 | 134775 | 
| 81 | 15 astropy/extern/configobj/validate.py | 356 | 443| 679 | 44282 | 147407 | 
| 82 | 15 astropy/io/misc/asdf/tags/transform/functional_models.py | 270 | 281| 145 | 44427 | 147407 | 
| 83 | **15 astropy/modeling/bounding_box.py** | 1195 | 1212| 117 | 44544 | 147407 | 
| 84 | 15 astropy/modeling/functional_models.py | 2542 | 2552| 171 | 44715 | 147407 | 
| 85 | 15 astropy/io/misc/asdf/tags/transform/functional_models.py | 176 | 187| 144 | 44859 | 147407 | 
| 86 | 15 astropy/io/misc/asdf/tags/transform/functional_models.py | 424 | 434| 134 | 44993 | 147407 | 
| 87 | 15 astropy/modeling/fitting.py | 1703 | 1731| 236 | 45229 | 147407 | 
| 88 | 15 astropy/timeseries/periodograms/bls/core.py | 26 | 88| 639 | 45868 | 147407 | 
| 89 | 16 astropy/units/core.py | 1894 | 1955| 454 | 46322 | 166194 | 
| 90 | 16 astropy/io/misc/asdf/tags/transform/functional_models.py | 390 | 399| 120 | 46442 | 166194 | 
| 91 | 17 astropy/utils/masked/function_helpers.py | 80 | 155| 774 | 47216 | 174967 | 
| 92 | 17 astropy/convolution/utils.py | 2 | 31| 143 | 47359 | 174967 | 
| 93 | 17 astropy/modeling/fitting.py | 575 | 661| 847 | 48206 | 174967 | 
| 94 | 17 astropy/io/misc/asdf/tags/transform/functional_models.py | 752 | 761| 125 | 48331 | 174967 | 
| 95 | 17 astropy/modeling/fitting.py | 1555 | 1601| 488 | 48819 | 174967 | 
| 96 | 17 astropy/modeling/functional_models.py | 3099 | 3116| 211 | 49030 | 174967 | 
| 97 | 17 astropy/io/misc/asdf/tags/transform/functional_models.py | 3 | 19| 270 | 49300 | 174967 | 
| 98 | 17 astropy/modeling/functional_models.py | 5 | 30| 374 | 49674 | 174967 | 
| 99 | 17 astropy/modeling/tabular.py | 241 | 261| 211 | 49885 | 174967 | 
| 100 | 18 astropy/units/quantity_helper/function_helpers.py | 102 | 142| 482 | 50367 | 184408 | 
| 101 | 19 astropy/io/ascii/core.py | 183 | 267| 404 | 50771 | 197878 | 
| 102 | 19 astropy/convolution/utils.py | 84 | 160| 741 | 51512 | 197878 | 
| 103 | 19 astropy/modeling/mappings.py | 1 | 98| 796 | 52308 | 197878 | 


### Hint

```
You just can't differentiate between a robot and the very best of humans.

*(A special day message.)*
```

## Patch

```diff
diff --git a/astropy/modeling/bounding_box.py b/astropy/modeling/bounding_box.py
--- a/astropy/modeling/bounding_box.py
+++ b/astropy/modeling/bounding_box.py
@@ -694,6 +694,12 @@ def _validate_dict(self, bounding_box: dict):
         for key, value in bounding_box.items():
             self[key] = value
 
+    @property
+    def _available_input_index(self):
+        model_input_index = [self._get_index(_input) for _input in self._model.inputs]
+
+        return [_input for _input in model_input_index if _input not in self._ignored]
+
     def _validate_sequence(self, bounding_box, order: str = None):
         """Validate passing tuple of tuples representation (or related) and setting them."""
         order = self._get_order(order)
@@ -703,7 +709,7 @@ def _validate_sequence(self, bounding_box, order: str = None):
             bounding_box = bounding_box[::-1]
 
         for index, value in enumerate(bounding_box):
-            self[index] = value
+            self[self._available_input_index[index]] = value
 
     @property
     def _n_inputs(self) -> int:
@@ -727,7 +733,7 @@ def _validate_iterable(self, bounding_box, order: str = None):
     def _validate(self, bounding_box, order: str = None):
         """Validate and set any representation"""
         if self._n_inputs == 1 and not isinstance(bounding_box, dict):
-            self[0] = bounding_box
+            self[self._available_input_index[0]] = bounding_box
         else:
             self._validate_iterable(bounding_box, order)
 
@@ -751,7 +757,7 @@ def validate(cls, model, bounding_box,
             order = bounding_box.order
             if _preserve_ignore:
                 ignored = bounding_box.ignored
-            bounding_box = bounding_box.intervals
+            bounding_box = bounding_box.named_intervals
 
         new = cls({}, model, ignored=ignored, order=order)
         new._validate(bounding_box)

```

## Test Patch

```diff
diff --git a/astropy/modeling/tests/test_bounding_box.py b/astropy/modeling/tests/test_bounding_box.py
--- a/astropy/modeling/tests/test_bounding_box.py
+++ b/astropy/modeling/tests/test_bounding_box.py
@@ -12,7 +12,7 @@
                                            _ignored_interval, _Interval, _SelectorArgument,
                                            _SelectorArguments)
 from astropy.modeling.core import Model, fix_inputs
-from astropy.modeling.models import Gaussian1D, Gaussian2D, Identity, Scale, Shift
+from astropy.modeling.models import Gaussian1D, Gaussian2D, Identity, Polynomial2D, Scale, Shift
 
 
 class Test_Interval:
@@ -1633,6 +1633,15 @@ def test_prepare_inputs(self):
         assert (valid_index[0] == []).all()
         assert all_out and isinstance(all_out, bool)
 
+    def test_bounding_box_ignore(self):
+        """Regression test for #13028"""
+
+        bbox_x = ModelBoundingBox((9, 10), Polynomial2D(1), ignored=["x"])
+        assert bbox_x.ignored_inputs == ['x']
+
+        bbox_y = ModelBoundingBox((11, 12), Polynomial2D(1), ignored=["y"])
+        assert bbox_y.ignored_inputs == ['y']
+
 
 class Test_SelectorArgument:
     def test_create(self):
@@ -2098,15 +2107,17 @@ def test___repr__(self):
             "    bounding_boxes={\n" + \
             "        (1,) = ModelBoundingBox(\n" + \
             "                intervals={\n" + \
-            "                    x: Interval(lower=-1, upper=1)\n" + \
+            "                    y: Interval(lower=-1, upper=1)\n" + \
             "                }\n" + \
+            "                ignored=['x']\n" + \
             "                model=Gaussian2D(inputs=('x', 'y'))\n" + \
             "                order='C'\n" + \
             "            )\n" + \
             "        (2,) = ModelBoundingBox(\n" + \
             "                intervals={\n" + \
-            "                    x: Interval(lower=-2, upper=2)\n" + \
+            "                    y: Interval(lower=-2, upper=2)\n" + \
             "                }\n" + \
+            "                ignored=['x']\n" + \
             "                model=Gaussian2D(inputs=('x', 'y'))\n" + \
             "                order='C'\n" + \
             "            )\n" + \
@@ -2650,3 +2661,12 @@ def test_fix_inputs(self):
         assert bbox._bounding_boxes[(1,)] == (-np.inf, np.inf)
         assert bbox._bounding_boxes[(1,)].order == 'F'
         assert len(bbox._bounding_boxes) == 2
+
+    def test_complex_compound_bounding_box(self):
+        model = Identity(4)
+        bounding_boxes = {(2.5, 1.3): ((-1, 1), (-3, 3)), (2.5, 2.71): ((-3, 3), (-1, 1))}
+        selector_args = (('x0', True), ('x1', True))
+
+        bbox = CompoundBoundingBox.validate(model, bounding_boxes, selector_args)
+        assert bbox[(2.5, 1.3)] == ModelBoundingBox(((-1, 1), (-3, 3)), model, ignored=['x0', 'x1'])
+        assert bbox[(2.5, 2.71)] == ModelBoundingBox(((-3, 3), (-1, 1)), model, ignored=['x0', 'x1'])

```


## Code snippets

### 1 - astropy/modeling/bounding_box.py:

Start line: 667, End line: 732

```python
class ModelBoundingBox(_BoundingDomain):

    def __eq__(self, value):
        """Note equality can be either with old representation or new one."""
        if isinstance(value, tuple):
            return self.bounding_box() == value
        elif isinstance(value, ModelBoundingBox):
            return (self.intervals == value.intervals) and (self.ignored == value.ignored)
        else:
            return False

    def __setitem__(self, key, value):
        """Validate and store interval under key (input index or input name)."""
        index = self._get_index(key)
        if index in self._ignored:
            self._ignored.remove(index)

        self._intervals[index] = _Interval.validate(value)

    def __delitem__(self, key):
        """Delete stored interval"""
        index = self._get_index(key)
        if index in self._ignored:
            raise RuntimeError(f"Cannot delete ignored input: {key}!")
        del self._intervals[index]
        self._ignored.append(index)

    def _validate_dict(self, bounding_box: dict):
        """Validate passing dictionary of intervals and setting them."""
        for key, value in bounding_box.items():
            self[key] = value

    def _validate_sequence(self, bounding_box, order: str = None):
        """Validate passing tuple of tuples representation (or related) and setting them."""
        order = self._get_order(order)
        if order == 'C':
            # If bounding_box is C/python ordered, it needs to be reversed
            # to be in Fortran/mathematical/input order.
            bounding_box = bounding_box[::-1]

        for index, value in enumerate(bounding_box):
            self[index] = value

    @property
    def _n_inputs(self) -> int:
        n_inputs = self._model.n_inputs - len(self._ignored)
        if n_inputs > 0:
            return n_inputs
        else:
            return 0

    def _validate_iterable(self, bounding_box, order: str = None):
        """Validate and set any iterable representation"""
        if len(bounding_box) != self._n_inputs:
            raise ValueError(f"Found {len(bounding_box)} intervals, "
                             f"but must have exactly {self._n_inputs}.")

        if isinstance(bounding_box, dict):
            self._validate_dict(bounding_box)
        else:
            self._validate_sequence(bounding_box, order)

    def _validate(self, bounding_box, order: str = None):
        """Validate and set any representation"""
        if self._n_inputs == 1 and not isinstance(bounding_box, dict):
            self[0] = bounding_box
        else:
            self._validate_iterable(bounding_box, order)
```
### 2 - astropy/modeling/bounding_box.py:

Start line: 555, End line: 606

```python
class ModelBoundingBox(_BoundingDomain):
    """
    A model's bounding box

    Parameters
    ----------
    intervals : dict
        A dictionary containing all the intervals for each model input
            keys   -> input index
            values -> interval for that index

    model : `~astropy.modeling.Model`
        The Model this bounding_box is for.

    ignored : list
        A list containing all the inputs (index) which will not be
        checked for whether or not their elements are in/out of an interval.

    order : optional, str
        The ordering that is assumed for the tuple representation of this
        bounding_box. Options: 'C': C/Python order, e.g. z, y, x.
        (default), 'F': Fortran/mathematical notation order, e.g. x, y, z.
    """

    def __init__(self, intervals: Dict[int, _Interval], model,
                 ignored: List[int] = None, order: str = 'C'):
        super().__init__(model, ignored, order)

        self._intervals = {}
        if intervals != () and intervals != {}:
            self._validate(intervals, order=order)

    def copy(self, ignored=None):
        intervals = {index: interval.copy()
                     for index, interval in self._intervals.items()}

        if ignored is None:
            ignored = self._ignored.copy()

        return ModelBoundingBox(intervals, self._model,
                                ignored=ignored,
                                order=self._order)

    @property
    def intervals(self) -> Dict[int, _Interval]:
        """Return bounding_box labeled using input positions"""
        return self._intervals

    @property
    def named_intervals(self) -> Dict[str, _Interval]:
        """Return bounding_box labeled using input names"""
        return {self._get_name(index): bbox for index, bbox in self._intervals.items()}
```
### 3 - astropy/modeling/bounding_box.py:

Start line: 761, End line: 798

```python
class ModelBoundingBox(_BoundingDomain):

    def fix_inputs(self, model, fixed_inputs: dict, _keep_ignored=False):
        """
        Fix the bounding_box for a `fix_inputs` compound model.

        Parameters
        ----------
        model : `~astropy.modeling.Model`
            The new model for which this will be a bounding_box
        fixed_inputs : dict
            Dictionary of inputs which have been fixed by this bounding box.
        keep_ignored : bool
            Keep the ignored inputs of the bounding box (internal argument only)
        """

        new = self.copy()

        for _input in fixed_inputs.keys():
            del new[_input]

        if _keep_ignored:
            ignored = new.ignored
        else:
            ignored = None

        return ModelBoundingBox.validate(model, new.named_intervals,
                                    ignored=ignored, order=new._order)

    @property
    def dimension(self):
        return len(self)

    def domain(self, resolution, order: str = None):
        inputs = self._model.inputs
        order = self._get_order(order)
        if order == 'C':
            inputs = inputs[::-1]

        return [self[input_name].domain(resolution) for input_name in inputs]
```
### 4 - astropy/modeling/bounding_box.py:

Start line: 734, End line: 759

```python
class ModelBoundingBox(_BoundingDomain):

    @classmethod
    def validate(cls, model, bounding_box,
                 ignored: list = None, order: str = 'C', _preserve_ignore: bool = False, **kwargs):
        """
        Construct a valid bounding box for a model.

        Parameters
        ----------
        model : `~astropy.modeling.Model`
            The model for which this will be a bounding_box
        bounding_box : dict, tuple
            A possible representation of the bounding box
        order : optional, str
            The order that a tuple representation will be assumed to be
                Default: 'C'
        """
        if isinstance(bounding_box, ModelBoundingBox):
            order = bounding_box.order
            if _preserve_ignore:
                ignored = bounding_box.ignored
            bounding_box = bounding_box.intervals

        new = cls({}, model, ignored=ignored, order=order)
        new._validate(bounding_box)

        return new
```
### 5 - astropy/modeling/bounding_box.py:

Start line: 1488, End line: 1504

```python
class CompoundBoundingBox(_BoundingDomain):

    def _fix_input_selector_arg(self, argument, value):
        matching_bounding_boxes = self._matching_bounding_boxes(argument, value)

        if len(self.selector_args) == 1:
            return matching_bounding_boxes[()]
        else:
            return CompoundBoundingBox(matching_bounding_boxes, self._model,
                                       self.selector_args.reduce(self._model, argument))

    def _fix_input_bbox_arg(self, argument, value):
        bounding_boxes = {}
        for selector_key, bbox in self._bounding_boxes.items():
            bounding_boxes[selector_key] = bbox.fix_inputs(self._model, {argument: value},
                                                        _keep_ignored=True)

        return CompoundBoundingBox(bounding_boxes, self._model,
                                   self.selector_args.add_ignore(self._model, argument))
```
### 6 - astropy/modeling/bounding_box.py:

Start line: 608, End line: 625

```python
class ModelBoundingBox(_BoundingDomain):

    def __repr__(self):
        parts = [
            'ModelBoundingBox(',
            '    intervals={'
        ]

        for name, interval in self.named_intervals.items():
            parts.append(f"        {name}: {interval}")

        parts.append('    }')
        if len(self._ignored) > 0:
            parts.append(f"    ignored={self.ignored_inputs}")

        parts.append(f'    model={self._model.__class__.__name__}(inputs={self._model.inputs})')
        parts.append(f"    order='{self._order}'")
        parts.append(')')

        return '\n'.join(parts)
```
### 7 - astropy/modeling/bounding_box.py:

Start line: 627, End line: 645

```python
class ModelBoundingBox(_BoundingDomain):

    def __len__(self):
        return len(self._intervals)

    def __contains__(self, key):
        try:
            return self._get_index(key) in self._intervals or self._ignored
        except (IndexError, ValueError):
            return False

    def has_interval(self, key):
        return self._get_index(key) in self._intervals

    def __getitem__(self, key):
        """Get bounding_box entries by either input name or input index"""
        index = self._get_index(key)
        if index in self._ignored:
            return _ignored_interval
        else:
            return self._intervals[self._get_index(key)]
```
### 8 - astropy/modeling/bounding_box.py:

Start line: 1506, End line: 1542

```python
class CompoundBoundingBox(_BoundingDomain):

    def fix_inputs(self, model, fixed_inputs: dict):
        """
        Fix the bounding_box for a `fix_inputs` compound model.

        Parameters
        ----------
        model : `~astropy.modeling.Model`
            The new model for which this will be a bounding_box
        fixed_inputs : dict
            Dictionary of inputs which have been fixed by this bounding box.
        """

        fixed_input_keys = list(fixed_inputs.keys())
        argument = fixed_input_keys.pop()
        value = fixed_inputs[argument]

        if self.selector_args.is_argument(self._model, argument):
            bbox = self._fix_input_selector_arg(argument, value)
        else:
            bbox = self._fix_input_bbox_arg(argument, value)

        if len(fixed_input_keys) > 0:
            new_fixed_inputs = fixed_inputs.copy()
            del new_fixed_inputs[argument]

            bbox = bbox.fix_inputs(model, new_fixed_inputs)

        if isinstance(bbox, CompoundBoundingBox):
            selector_args = bbox.named_selector_tuple
            bbox_dict = bbox
        elif isinstance(bbox, ModelBoundingBox):
            selector_args = None
            bbox_dict = bbox.named_intervals

        return bbox.__class__.validate(model, bbox_dict,
                                       order=bbox.order, selector_args=selector_args)
```
### 9 - astropy/modeling/bounding_box.py:

Start line: 1332, End line: 1382

```python
class CompoundBoundingBox(_BoundingDomain):

    @property
    def bounding_boxes(self) -> Dict[Any, ModelBoundingBox]:
        return self._bounding_boxes

    @property
    def selector_args(self) -> _SelectorArguments:
        return self._selector_args

    @selector_args.setter
    def selector_args(self, value):
        self._selector_args = _SelectorArguments.validate(self._model, value)

        warnings.warn("Overriding selector_args may cause problems you should re-validate "
                      "the compound bounding box before use!", RuntimeWarning)

    @property
    def named_selector_tuple(self) -> tuple:
        return self._selector_args.named_tuple(self._model)

    @property
    def create_selector(self):
        return self._create_selector

    @staticmethod
    def _get_selector_key(key):
        if isiterable(key):
            return tuple(key)
        else:
            return (key,)

    def __setitem__(self, key, value):
        _selector = self._get_selector_key(key)
        if not self.selector_args.is_selector(_selector):
            raise ValueError(f"{_selector} is not a selector!")

        ignored = self.selector_args.ignore + self.ignored
        self._bounding_boxes[_selector] = ModelBoundingBox.validate(self._model, value,
                                                                    ignored,
                                                                    order=self._order)

    def _validate(self, bounding_boxes: dict):
        for _selector, bounding_box in bounding_boxes.items():
            self[_selector] = bounding_box

    def __eq__(self, value):
        if isinstance(value, CompoundBoundingBox):
            return (self.bounding_boxes == value.bounding_boxes) and \
                (self.selector_args == value.selector_args) and \
                (self.create_selector == value.create_selector)
        else:
            return False
```
### 10 - astropy/modeling/bounding_box.py:

Start line: 899, End line: 930

```python
_BaseSelectorArgument = namedtuple('_BaseSelectorArgument', "index ignore")


class _SelectorArgument(_BaseSelectorArgument):
    """
    Contains a single CompoundBoundingBox slicing input.

    Parameters
    ----------
    index : int
        The index of the input in the input list

    ignore : bool
        Whether or not this input will be ignored by the bounding box.

    Methods
    -------
    validate :
        Returns a valid SelectorArgument for a given model.

    get_selector :
        Returns the value of the input for use in finding the correct
        bounding_box.

    get_fixed_value :
        Gets the slicing value from a fix_inputs set of values.
    """

    def __new__(cls, index, ignore):
        self = super().__new__(cls, index, ignore)

        return self
```
### 11 - astropy/modeling/bounding_box.py:

Start line: 800, End line: 834

```python
class ModelBoundingBox(_BoundingDomain):

    def _outside(self,  input_shape, inputs):
        """
        Get all the input positions which are outside the bounding_box,
        so that the corresponding outputs can be filled with the fill
        value (default NaN).

        Parameters
        ----------
        input_shape : tuple
            The shape that all inputs have be reshaped/broadcasted into
        inputs : list
            List of all the model inputs

        Returns
        -------
        outside_index : bool-numpy array
            True  -> position outside bounding_box
            False -> position inside  bounding_box
        all_out : bool
            if all of the inputs are outside the bounding_box
        """
        all_out = False

        outside_index = np.zeros(input_shape, dtype=bool)
        for index, _input in enumerate(inputs):
            _input = np.asanyarray(_input)

            outside = np.broadcast_to(self[index].outside(_input), input_shape)
            outside_index[outside] = True

            if outside_index.all():
                all_out = True
                break

        return outside_index, all_out
```
### 12 - astropy/modeling/bounding_box.py:

Start line: 1233, End line: 1262

```python
class _SelectorArguments(tuple):

    def add_ignore(self, model, argument):
        """
        Add argument to the kept_ignore list

        Parameters
        ----------
        model : `~astropy.modeling.Model`
            The Model these selector arguments are for.

        argument : int or str
            A representation of which argument is being used
        """

        if self.is_argument(model, argument):
            raise ValueError(f"{argument}: is a selector argument and cannot be ignored.")

        kept_ignore = [get_index(model, argument)]

        return _SelectorArguments.validate(model, self, kept_ignore)

    def named_tuple(self, model):
        """
        Get a tuple of selector argument tuples using input names

        Parameters
        ----------
        model : `~astropy.modeling.Model`
            The Model these selector arguments are for.
        """
        return tuple([selector_arg.named_tuple(model) for selector_arg in self])
```
### 13 - astropy/modeling/bounding_box.py:

Start line: 932, End line: 950

```python
class _SelectorArgument(_BaseSelectorArgument):

    @classmethod
    def validate(cls, model, argument, ignored: bool = True):
        """
        Construct a valid selector argument for a CompoundBoundingBox.

        Parameters
        ----------
        model : `~astropy.modeling.Model`
            The model for which this will be an argument for.
        argument : int or str
            A representation of which evaluation input to use
        ignored : optional, bool
            Whether or not to ignore this argument in the ModelBoundingBox.

        Returns
        -------
        Validated selector_argument
        """
        return cls(get_index(model, argument), ignored)
```
### 14 - astropy/modeling/bounding_box.py:

Start line: 862, End line: 896

```python
class ModelBoundingBox(_BoundingDomain):

    def prepare_inputs(self, input_shape, inputs) -> Tuple[Any, Any, Any]:
        """
        Get prepare the inputs with respect to the bounding box.

        Parameters
        ----------
        input_shape : tuple
            The shape that all inputs have be reshaped/broadcasted into
        inputs : list
            List of all the model inputs

        Returns
        -------
        valid_inputs : list
            The inputs reduced to just those inputs which are all inside
            their respective bounding box intervals
        valid_index : array_like
            array of all indices inside the bounding box
        all_out: bool
            if all of the inputs are outside the bounding_box
        """
        valid_index, all_out = self._valid_index(input_shape, inputs)

        valid_inputs = []
        if not all_out:
            for _input in inputs:
                if input_shape:
                    valid_input = np.broadcast_to(np.atleast_1d(_input), input_shape)[valid_index]
                    if np.isscalar(_input):
                        valid_input = valid_input.item(0)
                    valid_inputs.append(valid_input)
                else:
                    valid_inputs.append(_input)

        return tuple(valid_inputs), valid_index, all_out
```
### 15 - astropy/modeling/bounding_box.py:

Start line: 1039, End line: 1104

```python
class _SelectorArguments(tuple):
    """
    Contains the CompoundBoundingBox slicing description

    Parameters
    ----------
    input_ :
        The SelectorArgument values

    Methods
    -------
    validate :
        Returns a valid SelectorArguments for its model.

    get_selector :
        Returns the selector a set of inputs corresponds to.

    is_selector :
        Determines if a selector is correctly formatted for this CompoundBoundingBox.

    get_fixed_value :
        Gets the selector from a fix_inputs set of values.
    """

    _kept_ignore = None

    def __new__(cls, input_: Tuple[_SelectorArgument], kept_ignore: List = None):
        self = super().__new__(cls, input_)

        if kept_ignore is None:
            self._kept_ignore = []
        else:
            self._kept_ignore = kept_ignore

        return self

    def pretty_repr(self, model):
        """
        Get a pretty-print representation of this object

        Parameters
        ----------
        model : `~astropy.modeling.Model`
            The Model these selector arguments are for.
        """
        parts = ['SelectorArguments(']
        for argument in self:
            parts.append(
                f"    {argument.pretty_repr(model)}"
            )
        parts.append(')')

        return '\n'.join(parts)

    @property
    def ignore(self):
        """Get the list of ignored inputs"""
        ignore = [argument.index for argument in self if argument.ignore]
        ignore.extend(self._kept_ignore)

        return ignore

    @property
    def kept_ignore(self):
        """The arguments to persist in ignoring"""
        return self._kept_ignore
```
### 16 - astropy/modeling/bounding_box.py:

Start line: 162, End line: 262

```python
class _BoundingDomain(abc.ABC):
    """
    Base class for ModelBoundingBox and CompoundBoundingBox.
        This is where all the `~astropy.modeling.core.Model` evaluation
        code for evaluating with a bounding box is because it is common
        to both types of bounding box.

    Parameters
    ----------
    model : `~astropy.modeling.Model`
        The Model this bounding domain is for.

    prepare_inputs :
        Generates the necessary input information so that model can
        be evaluated only for input points entirely inside bounding_box.
        This needs to be implemented by a subclass. Note that most of
        the implementation is in ModelBoundingBox.

    prepare_outputs :
        Fills the output values in for any input points outside the
        bounding_box.

    evaluate :
        Performs a complete model evaluation while enforcing the bounds
        on the inputs and returns a complete output.
    """

    def __init__(self, model, ignored: List[int] = None, order: str = 'C'):
        self._model = model
        self._ignored = self._validate_ignored(ignored)
        self._order = self._get_order(order)

    @property
    def model(self):
        return self._model

    @property
    def order(self) -> str:
        return self._order

    @property
    def ignored(self) -> List[int]:
        return self._ignored

    def _get_order(self, order: str = None) -> str:
        """
        Get if bounding_box is C/python ordered or Fortran/mathematically
        ordered
        """
        if order is None:
            order = self._order

        if order not in ('C', 'F'):
            raise ValueError("order must be either 'C' (C/python order) or "
                             f"'F' (Fortran/mathematical order), got: {order}.")

        return order

    def _get_index(self, key) -> int:
        """
        Get the input index corresponding to the given key.
            Can pass in either:
                the string name of the input or
                the input index itself.
        """

        return get_index(self._model, key)

    def _get_name(self, index: int):
        """Get the input name corresponding to the input index"""
        return get_name(self._model, index)

    @property
    def ignored_inputs(self) -> List[str]:
        return [self._get_name(index) for index in self._ignored]

    def _validate_ignored(self, ignored: list) -> List[int]:
        if ignored is None:
            return []
        else:
            return [self._get_index(key) for key in ignored]

    def __call__(self, *args, **kwargs):
        raise NotImplementedError(
            "This bounding box is fixed by the model and does not have "
            "adjustable parameters.")

    @abc.abstractmethod
    def fix_inputs(self, model, fixed_inputs: dict):
        """
        Fix the bounding_box for a `fix_inputs` compound model.

        Parameters
        ----------
        model : `~astropy.modeling.Model`
            The new model for which this will be a bounding_box
        fixed_inputs : dict
            Dictionary of inputs which have been fixed by this bounding box.
        """

        raise NotImplementedError("This should be implemented by a child class.")
```
### 17 - astropy/modeling/bounding_box.py:

Start line: 836, End line: 860

```python
class ModelBoundingBox(_BoundingDomain):

    def _valid_index(self, input_shape, inputs):
        """
        Get the indices of all the inputs inside the bounding_box.

        Parameters
        ----------
        input_shape : tuple
            The shape that all inputs have be reshaped/broadcasted into
        inputs : list
            List of all the model inputs

        Returns
        -------
        valid_index : numpy array
            array of all indices inside the bounding box
        all_out : bool
            if all of the inputs are outside the bounding_box
        """
        outside_index, all_out = self._outside(input_shape, inputs)

        valid_index = np.atleast_1d(np.logical_not(outside_index)).nonzero()
        if len(valid_index[0]) == 0:
            all_out = True

        return valid_index, all_out
```
### 18 - astropy/modeling/bounding_box.py:

Start line: 1466, End line: 1486

```python
class CompoundBoundingBox(_BoundingDomain):

    def _matching_bounding_boxes(self, argument, value) -> Dict[Any, ModelBoundingBox]:
        selector_index = self.selector_args.selector_index(self._model, argument)
        matching = {}
        for selector_key, bbox in self._bounding_boxes.items():
            if selector_key[selector_index] == value:
                new_selector_key = list(selector_key)
                new_selector_key.pop(selector_index)

                if bbox.has_interval(argument):
                    new_bbox = bbox.fix_inputs(self._model, {argument: value},
                                               _keep_ignored=True)
                else:
                    new_bbox = bbox.copy()

                matching[tuple(new_selector_key)] = new_bbox

        if len(matching) == 0:
            raise ValueError(f"Attempting to fix input {argument}, but there are no "
                             f"bounding boxes for argument value {value}.")

        return matching
```
### 19 - astropy/modeling/bounding_box.py:

Start line: 1420, End line: 1440

```python
class CompoundBoundingBox(_BoundingDomain):

    def __contains__(self, key):
        return key in self._bounding_boxes

    def _create_bounding_box(self, _selector):
        self[_selector] = self._create_selector(_selector, model=self._model)

        return self[_selector]

    def __getitem__(self, key):
        _selector = self._get_selector_key(key)
        if _selector in self:
            return self._bounding_boxes[_selector]
        elif self._create_selector is not None:
            return self._create_bounding_box(_selector)
        else:
            raise RuntimeError(f"No bounding box is defined for selector: {_selector}.")

    def _select_bounding_box(self, inputs) -> ModelBoundingBox:
        _selector = self.selector_args.get_selector(*inputs)

        return self[_selector]
```
### 20 - astropy/modeling/bounding_box.py:

Start line: 1442, End line: 1464

```python
class CompoundBoundingBox(_BoundingDomain):

    def prepare_inputs(self, input_shape, inputs) -> Tuple[Any, Any, Any]:
        """
        Get prepare the inputs with respect to the bounding box.

        Parameters
        ----------
        input_shape : tuple
            The shape that all inputs have be reshaped/broadcasted into
        inputs : list
            List of all the model inputs

        Returns
        -------
        valid_inputs : list
            The inputs reduced to just those inputs which are all inside
            their respective bounding box intervals
        valid_index : array_like
            array of all indices inside the bounding box
        all_out: bool
            if all of the inputs are outside the bounding_box
        """
        bounding_box = self._select_bounding_box(inputs)
        return bounding_box.prepare_inputs(input_shape, inputs)
```
### 22 - astropy/modeling/bounding_box.py:

Start line: 130, End line: 159

```python
# The interval where all ignored inputs can be found.
_ignored_interval = _Interval.validate((-np.inf, np.inf))


def get_index(model, key) -> int:
    """
    Get the input index corresponding to the given key.
        Can pass in either:
            the string name of the input or
            the input index itself.
    """
    if isinstance(key, str):
        if key in model.inputs:
            index = model.inputs.index(key)
        else:
            raise ValueError(f"'{key}' is not one of the inputs: {model.inputs}.")
    elif np.issubdtype(type(key), np.integer):
        if 0 <= key < len(model.inputs):
            index = key
        else:
            raise IndexError(f"Integer key: {key} must be non-negative and < {len(model.inputs)}.")
    else:
        raise ValueError(f"Key value: {key} must be string or integer.")

    return index


def get_name(model, index: int):
    """Get the input name corresponding to the input index"""
    return model.inputs[index]
```
### 23 - astropy/modeling/bounding_box.py:

Start line: 79, End line: 127

```python
class _Interval(_BaseInterval):

    @classmethod
    def _validate_bounds(cls, lower, upper):
        """Validate the bounds are reasonable and construct an interval from them."""
        if (np.asanyarray(lower) > np.asanyarray(upper)).all():
            warnings.warn(f"Invalid interval: upper bound {upper} "
                          f"is strictly less than lower bound {lower}.", RuntimeWarning)

        return cls(lower, upper)

    @classmethod
    def validate(cls, interval):
        """
        Construct and validate an interval

        Parameters
        ----------
        interval : iterable
            A representation of the interval.

        Returns
        -------
        A validated interval.
        """
        cls._validate_shape(interval)

        if len(interval) == 1:
            interval = tuple(interval[0])
        else:
            interval = tuple(interval)

        return cls._validate_bounds(interval[0], interval[1])

    def outside(self, _input: np.ndarray):
        """
        Parameters
        ----------
        _input : np.ndarray
            The evaluation input in the form of an array.

        Returns
        -------
        Boolean array indicating which parts of _input are outside the interval:
            True  -> position outside interval
            False -> position inside  interval
        """
        return np.logical_or(_input < self.lower, _input > self.upper)

    def domain(self, resolution):
        return np.arange(self.lower, self.upper + resolution, resolution)
```
### 24 - astropy/modeling/bounding_box.py:

Start line: 55, End line: 77

```python
class _Interval(_BaseInterval):

    @staticmethod
    def _validate_shape(interval):
        """Validate the shape of an interval representation"""
        MESSAGE = """An interval must be some sort of sequence of length 2"""

        try:
            shape = np.shape(interval)
        except TypeError:
            try:
                # np.shape does not work with lists of Quantities
                if len(interval) == 1:
                    interval = interval[0]
                shape = np.shape([b.to_value() for b in interval])
            except (ValueError, TypeError, AttributeError):
                raise ValueError(MESSAGE)

        valid_shape = shape in ((2,), (1, 2), (2, 0))
        if not valid_shape:
            valid_shape = (len(shape) > 0) and (shape[0] == 2) and \
                all(isinstance(b, np.ndarray) for b in interval)

        if not isiterable(interval) or not valid_shape:
            raise ValueError(MESSAGE)
```
### 25 - astropy/modeling/bounding_box.py:

Start line: 1140, End line: 1193

```python
class _SelectorArguments(tuple):

    def get_selector(self, *inputs):
        """
        Get the selector corresponding to these inputs

        Parameters
        ----------
        *inputs :
            All the processed model evaluation inputs.
        """
        return tuple([argument.get_selector(*inputs) for argument in self])

    def is_selector(self, _selector):
        """
        Determine if this is a reasonable selector

        Parameters
        ----------
        _selector : tuple
            The selector to check
        """
        return isinstance(_selector, tuple) and len(_selector) == len(self)

    def get_fixed_values(self, model, values: dict):
        """
        Gets the value fixed input corresponding to this argument

        Parameters
        ----------
        model : `~astropy.modeling.Model`
            The Model these selector arguments are for.

        values : dict
            Dictionary of fixed inputs.
        """
        return tuple([argument.get_fixed_value(model, values) for argument in self])

    def is_argument(self, model, argument) -> bool:
        """
        Determine if passed argument is one of the selector arguments

        Parameters
        ----------
        model : `~astropy.modeling.Model`
            The Model these selector arguments are for.

        argument : int or str
            A representation of which evaluation input is being used
        """

        for selector_arg in self:
            if selector_arg.is_argument(model, argument):
                return True
        else:
            return False
```
### 27 - astropy/modeling/bounding_box.py:

Start line: 8, End line: 53

```python
import abc
import copy
import warnings
from collections import namedtuple
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from astropy.units import Quantity
from astropy.utils import isiterable

__all__ = ['ModelBoundingBox', 'CompoundBoundingBox']


_BaseInterval = namedtuple('_BaseInterval', "lower upper")


class _Interval(_BaseInterval):
    """
    A single input's bounding box interval.

    Parameters
    ----------
    lower : float
        The lower bound of the interval

    upper : float
        The upper bound of the interval

    Methods
    -------
    validate :
        Contructs a valid interval

    outside :
        Determine which parts of an input array are outside the interval.

    domain :
        Contructs a discretization of the points inside the interval.
    """

    def __repr__(self):
        return f"Interval(lower={self.lower}, upper={self.upper})"

    def copy(self):
        return copy.deepcopy(self)
```
### 28 - astropy/modeling/bounding_box.py:

Start line: 1106, End line: 1138

```python
class _SelectorArguments(tuple):

    @classmethod
    def validate(cls, model, arguments, kept_ignore: List=None):
        """
        Construct a valid Selector description for a CompoundBoundingBox.

        Parameters
        ----------
        model : `~astropy.modeling.Model`
            The Model these selector arguments are for.

        arguments :
            The individual argument informations

        kept_ignore :
            Arguments to persist as ignored
        """
        inputs = []
        for argument in arguments:
            _input = _SelectorArgument.validate(model, *argument)
            if _input.index in [this.index for this in inputs]:
                raise ValueError(f"Input: '{get_name(model, _input.index)}' has been repeated.")
            inputs.append(_input)

        if len(inputs) == 0:
            raise ValueError("There must be at least one selector argument.")

        if isinstance(arguments, _SelectorArguments):
            if kept_ignore is None:
                kept_ignore = []

            kept_ignore.extend(arguments.kept_ignore)

        return cls(tuple(inputs), kept_ignore)
```
### 30 - astropy/modeling/bounding_box.py:

Start line: 264, End line: 286

```python
class _BoundingDomain(abc.ABC):

    @abc.abstractmethod
    def prepare_inputs(self, input_shape, inputs) -> Tuple[Any, Any, Any]:
        """
        Get prepare the inputs with respect to the bounding box.

        Parameters
        ----------
        input_shape : tuple
            The shape that all inputs have be reshaped/broadcasted into
        inputs : list
            List of all the model inputs

        Returns
        -------
        valid_inputs : list
            The inputs reduced to just those inputs which are all inside
            their respective bounding box intervals
        valid_index : array_like
            array of all indices inside the bounding box
        all_out: bool
            if all of the inputs are outside the bounding_box
        """
        raise NotImplementedError("This has not been implemented for BoundingDomain.")
```
### 31 - astropy/modeling/bounding_box.py:

Start line: 1312, End line: 1330

```python
class CompoundBoundingBox(_BoundingDomain):

    def __repr__(self):
        parts = ['CompoundBoundingBox(',
                 '    bounding_boxes={']
        # bounding_boxes
        for _selector, bbox in self._bounding_boxes.items():
            bbox_repr = bbox.__repr__().split('\n')
            parts.append(f"        {_selector} = {bbox_repr.pop(0)}")
            for part in bbox_repr:
                parts.append(f"            {part}")
        parts.append('    }')

        # selector_args
        selector_args_repr = self.selector_args.pretty_repr(self._model).split('\n')
        parts.append(f"    selector_args = {selector_args_repr.pop(0)}")
        for part in selector_args_repr:
            parts.append(f"        {part}")
        parts.append(')')

        return '\n'.join(parts)
```
### 33 - astropy/modeling/bounding_box.py:

Start line: 1384, End line: 1418

```python
class CompoundBoundingBox(_BoundingDomain):

    @classmethod
    def validate(cls, model, bounding_box: dict, selector_args=None, create_selector=None,
                 ignored: list = None, order: str = 'C', _preserve_ignore: bool = False, **kwarg):
        """
        Construct a valid compound bounding box for a model.

        Parameters
        ----------
        model : `~astropy.modeling.Model`
            The model for which this will be a bounding_box
        bounding_box : dict
            Dictionary of possible bounding_box respresentations
        selector_args : optional
            Description of the selector arguments
        create_selector : optional, callable
            Method for generating new selectors
        order : optional, str
            The order that a tuple representation will be assumed to be
                Default: 'C'
        """
        if isinstance(bounding_box, CompoundBoundingBox):
            if selector_args is None:
                selector_args = bounding_box.selector_args
            if create_selector is None:
                create_selector = bounding_box.create_selector
            order = bounding_box.order
            if _preserve_ignore:
                ignored = bounding_box.ignored
            bounding_box = bounding_box.bounding_boxes

        if selector_args is None:
            raise ValueError("Selector arguments must be provided (can be passed as part of bounding_box argument)!")

        return cls(bounding_box, model, selector_args,
                   create_selector=create_selector, ignored=ignored, order=order)
```
### 34 - astropy/modeling/bounding_box.py:

Start line: 390, End line: 429

```python
class _BoundingDomain(abc.ABC):

    def prepare_outputs(self, valid_outputs, valid_index, input_shape, fill_value):
        """
        Fill in all the outputs of the model corresponding to inputs
        outside the bounding_box, adjusting any single output model so that
        its output becomes a list of containing that output.

        Parameters
        ----------
        valid_outputs : list
            The list of outputs from the model corresponding to inputs
            inside the bounding box
        valid_index : array_like
            array of all indices of inputs inside the bounding box
        input_shape : tuple
            The shape that all inputs have be reshaped/broadcasted into
        fill_value : float
            The value which will be assigned to inputs which are outside
            the bounding box
        """
        if self._model.n_outputs == 1:
            valid_outputs = [valid_outputs]

        return self._prepare_outputs(valid_outputs, valid_index, input_shape, fill_value)

    @staticmethod
    def _get_valid_outputs_unit(valid_outputs, with_units: bool):
        """
        Get the unit for outputs if one is required.

        Parameters
        ----------
        valid_outputs : list of numpy array
            The list of outputs from the model corresponding to inputs
            inside the bounding box
        with_units : bool
            whether or not a unit is required
        """

        if with_units:
            return getattr(valid_outputs, 'unit', None)
```
### 35 - astropy/modeling/bounding_box.py:

Start line: 362, End line: 388

```python
class _BoundingDomain(abc.ABC):

    def _prepare_outputs(self, valid_outputs, valid_index, input_shape, fill_value):
        """
        Fill in all the outputs of the model corresponding to inputs
        outside the bounding_box.

        Parameters
        ----------
        valid_outputs : list of numpy array
            The list of outputs from the model corresponding to inputs
            inside the bounding box
        valid_index : numpy array
            array of all indices of inputs inside the bounding box
        input_shape : tuple
            The shape that all inputs have be reshaped/broadcasted into
        fill_value : float
            The value which will be assigned to inputs which are outside
            the bounding box

        Returns
        -------
        List of filled in output arrays.
        """
        outputs = []
        for valid_output in valid_outputs:
            outputs.append(self._modify_output(valid_output, valid_index, input_shape, fill_value))

        return outputs
```
### 36 - astropy/modeling/bounding_box.py:

Start line: 647, End line: 665

```python
class ModelBoundingBox(_BoundingDomain):

    def bounding_box(self, order: str = None):
        """
        Return the old tuple of tuples representation of the bounding_box
            order='C' corresponds to the old bounding_box ordering
            order='F' corresponds to the gwcs bounding_box ordering.
        """
        if len(self._intervals) == 1:
            return tuple(list(self._intervals.values())[0])
        else:
            order = self._get_order(order)
            inputs = self._model.inputs
            if order == 'C':
                inputs = inputs[::-1]

            bbox = tuple([tuple(self[input_name]) for input_name in inputs])
            if len(bbox) == 1:
                bbox = bbox[0]

            return bbox
```
### 40 - astropy/modeling/bounding_box.py:

Start line: 1265, End line: 1310

```python
class CompoundBoundingBox(_BoundingDomain):
    """
    A model's compound bounding box

    Parameters
    ----------
    bounding_boxes : dict
        A dictionary containing all the ModelBoundingBoxes that are possible
            keys   -> _selector (extracted from model inputs)
            values -> ModelBoundingBox

    model : `~astropy.modeling.Model`
        The Model this compound bounding_box is for.

    selector_args : _SelectorArguments
        A description of how to extract the selectors from model inputs.

    create_selector : optional
        A method which takes in the selector and the model to return a
        valid bounding corresponding to that selector. This can be used
        to construct new bounding_boxes for previously undefined selectors.
        These new boxes are then stored for future lookups.

    order : optional, str
        The ordering that is assumed for the tuple representation of the
        bounding_boxes.
    """
    def __init__(self, bounding_boxes: Dict[Any, ModelBoundingBox], model,
                 selector_args: _SelectorArguments, create_selector: Callable = None,
                 ignored: List[int] = None, order: str = 'C'):
        super().__init__(model, ignored, order)

        self._create_selector = create_selector
        self._selector_args = _SelectorArguments.validate(model, selector_args)

        self._bounding_boxes = {}
        self._validate(bounding_boxes)

    def copy(self):
        bounding_boxes = {selector: bbox.copy(self.selector_args.ignore)
                          for selector, bbox in self._bounding_boxes.items()}

        return CompoundBoundingBox(bounding_boxes, self._model,
                                   selector_args=self._selector_args,
                                   create_selector=copy.deepcopy(self._create_selector),
                                   order=self._order)
```
### 42 - astropy/modeling/bounding_box.py:

Start line: 328, End line: 360

```python
class _BoundingDomain(abc.ABC):

    def _modify_output(self, valid_output, valid_index, input_shape, fill_value):
        """
        For a single output fill in all the parts corresponding to inputs
        outside the bounding box.

        Parameters
        ----------
        valid_output : numpy array
            The output from the model corresponding to inputs inside the
            bounding box
        valid_index : numpy array
            array of all indices of inputs inside the bounding box
        input_shape : tuple
            The shape that all inputs have be reshaped/broadcasted into
        fill_value : float
            The value which will be assigned to inputs which are outside
            the bounding box

        Returns
        -------
        An output array with all the indices corresponding to inputs
        outside the bounding box filled in by fill_value
        """
        output = self._base_output(input_shape, fill_value)
        if not output.shape:
            output = np.array(valid_output)
        else:
            output[valid_index] = valid_output

        if np.isscalar(valid_output):
            output = output.item(0)

        return output
```
### 44 - astropy/modeling/bounding_box.py:

Start line: 308, End line: 326

```python
class _BoundingDomain(abc.ABC):

    def _all_out_output(self, input_shape, fill_value):
        """
        Create output if all inputs are outside the domain

        Parameters
        ----------
        input_shape : tuple
            The shape that all inputs have be reshaped/broadcasted into
        fill_value : float
            The value which will be assigned to inputs which are outside
            the bounding box

        Returns
        -------
        A full set of outputs for case that all inputs are outside domain.
        """

        return [self._base_output(input_shape, fill_value)
                for _ in range(self._model.n_outputs)], None
```
### 47 - astropy/modeling/bounding_box.py:

Start line: 466, End line: 502

```python
class _BoundingDomain(abc.ABC):

    def _evaluate(self, evaluate: Callable, inputs, input_shape,
                  fill_value, with_units: bool):
        """
        Perform model evaluation steps:
            prepare_inputs -> evaluate -> prepare_outputs

        Parameters
        ----------
        evaluate : Callable
            callable which takes in the valid inputs to evaluate model
        valid_inputs : list of numpy arrays
            The inputs reduced to just those inputs which are all inside
            their respective bounding box intervals
        valid_index : numpy array
            array of all indices inside the bounding box
        input_shape : tuple
            The shape that all inputs have be reshaped/broadcasted into
        fill_value : float
            The value which will be assigned to inputs which are outside
            the bounding box
        with_units : bool
            whether or not a unit is required

        Returns
        -------
        outputs :
            list containing filled in output values
        valid_outputs_unit :
            the unit that will be attached to the outputs
        """
        valid_inputs, valid_index, all_out = self.prepare_inputs(input_shape, inputs)

        if all_out:
            return self._all_out_output(input_shape, fill_value)
        else:
            return self._evaluate_model(evaluate, valid_inputs, valid_index,
                                        input_shape, fill_value, with_units)
```
### 51 - astropy/modeling/bounding_box.py:

Start line: 991, End line: 1009

```python
class _SelectorArgument(_BaseSelectorArgument):

    def get_fixed_value(self, model, values: dict):
        """
        Gets the value fixed input corresponding to this argument

        Parameters
        ----------
        model : `~astropy.modeling.Model`
            The Model this selector argument is for.

        values : dict
            Dictionary of fixed inputs.
        """
        if self.index in values:
            return values[self.index]
        else:
            if self.name(model) in values:
                return values[self.name(model)]
            else:
                raise RuntimeError(f"{self.pretty_repr(model)} was not found in {values}")
```
### 54 - astropy/modeling/bounding_box.py:

Start line: 288, End line: 306

```python
class _BoundingDomain(abc.ABC):

    @staticmethod
    def _base_output(input_shape, fill_value):
        """
        Create a baseline output, assuming that the entire input is outside
        the bounding box

        Parameters
        ----------
        input_shape : tuple
            The shape that all inputs have be reshaped/broadcasted into
        fill_value : float
            The value which will be assigned to inputs which are outside
            the bounding box

        Returns
        -------
        An array of the correct shape containing all fill_value
        """
        return np.zeros(input_shape) + fill_value
```
### 56 - astropy/modeling/bounding_box.py:

Start line: 527, End line: 552

```python
class _BoundingDomain(abc.ABC):

    def evaluate(self, evaluate: Callable, inputs, fill_value):
        """
        Perform full model evaluation steps:
            prepare_inputs -> evaluate -> prepare_outputs -> set output units

        Parameters
        ----------
        evaluate : callable
            callable which takes in the valid inputs to evaluate model
        valid_inputs : list
            The inputs reduced to just those inputs which are all inside
            their respective bounding box intervals
        valid_index : array_like
            array of all indices inside the bounding box
        fill_value : float
            The value which will be assigned to inputs which are outside
            the bounding box
        """
        input_shape = self._model.input_shape(inputs)

        # NOTE: CompoundModel does not currently support units during
        #   evaluation for bounding_box so this feature is turned off
        #   for CompoundModel(s).
        outputs, valid_outputs_unit = self._evaluate(evaluate, inputs, input_shape,
                                                     fill_value, self._model.bbox_with_units)
        return tuple(self._set_outputs_unit(outputs, valid_outputs_unit))
```
### 57 - astropy/modeling/bounding_box.py:

Start line: 504, End line: 525

```python
class _BoundingDomain(abc.ABC):

    @staticmethod
    def _set_outputs_unit(outputs, valid_outputs_unit):
        """
        Set the units on the outputs
            prepare_inputs -> evaluate -> prepare_outputs -> set output units

        Parameters
        ----------
        outputs :
            list containing filled in output values
        valid_outputs_unit :
            the unit that will be attached to the outputs

        Returns
        -------
        List containing filled in output values and units
        """

        if valid_outputs_unit is not None:
            return Quantity(outputs, valid_outputs_unit, copy=False)

        return outputs
```
### 61 - astropy/modeling/bounding_box.py:

Start line: 1011, End line: 1036

```python
class _SelectorArgument(_BaseSelectorArgument):

    def is_argument(self, model, argument) -> bool:
        """
        Determine if passed argument is described by this selector argument

        Parameters
        ----------
        model : `~astropy.modeling.Model`
            The Model this selector argument is for.

        argument : int or str
            A representation of which evaluation input is being used
        """

        return self.index == get_index(model, argument)

    def named_tuple(self, model):
        """
        Get a tuple representation of this argument using the input
        name from the model.

        Parameters
        ----------
        model : `~astropy.modeling.Model`
            The Model this selector argument is for.
        """
        return (self.name(model), self.ignore)
```
### 62 - astropy/modeling/bounding_box.py:

Start line: 431, End line: 464

```python
class _BoundingDomain(abc.ABC):

    def _evaluate_model(self, evaluate: Callable, valid_inputs, valid_index,
                        input_shape, fill_value, with_units: bool):
        """
        Evaluate the model using the given evaluate routine

        Parameters
        ----------
        evaluate : Callable
            callable which takes in the valid inputs to evaluate model
        valid_inputs : list of numpy arrays
            The inputs reduced to just those inputs which are all inside
            their respective bounding box intervals
        valid_index : numpy array
            array of all indices inside the bounding box
        input_shape : tuple
            The shape that all inputs have be reshaped/broadcasted into
        fill_value : float
            The value which will be assigned to inputs which are outside
            the bounding box
        with_units : bool
            whether or not a unit is required

        Returns
        -------
        outputs :
            list containing filled in output values
        valid_outputs_unit :
            the unit that will be attached to the outputs
        """
        valid_outputs = evaluate(valid_inputs)
        valid_outputs_unit = self._get_valid_outputs_unit(valid_outputs, with_units)

        return self.prepare_outputs(valid_outputs, valid_index,
                                    input_shape, fill_value), valid_outputs_unit
```
### 65 - astropy/modeling/bounding_box.py:

Start line: 1214, End line: 1231

```python
class _SelectorArguments(tuple):

    def reduce(self, model, argument):
        """
        Reduce the selector arguments by the argument given

        Parameters
        ----------
        model : `~astropy.modeling.Model`
            The Model these selector arguments are for.

        argument : int or str
            A representation of which argument is being used
        """

        arguments = list(self)
        kept_ignore = [arguments.pop(self.selector_index(model, argument)).index]
        kept_ignore.extend(self._kept_ignore)

        return _SelectorArguments.validate(model, tuple(arguments), kept_ignore)
```
### 69 - astropy/modeling/bounding_box.py:

Start line: 952, End line: 989

```python
class _SelectorArgument(_BaseSelectorArgument):

    def get_selector(self, *inputs):
        """
        Get the selector value corresponding to this argument

        Parameters
        ----------
        *inputs :
            All the processed model evaluation inputs.
        """
        _selector = inputs[self.index]
        if isiterable(_selector):
            if len(_selector) == 1:
                return _selector[0]
            else:
                return tuple(_selector)
        return _selector

    def name(self, model) -> str:
        """
        Get the name of the input described by this selector argument

        Parameters
        ----------
        model : `~astropy.modeling.Model`
            The Model this selector argument is for.
        """
        return get_name(model, self.index)

    def pretty_repr(self, model):
        """
        Get a pretty-print representation of this object

        Parameters
        ----------
        model : `~astropy.modeling.Model`
            The Model this selector argument is for.
        """
        return f"Argument(name='{self.name(model)}', ignore={self.ignore})"
```
### 83 - astropy/modeling/bounding_box.py:

Start line: 1195, End line: 1212

```python
class _SelectorArguments(tuple):

    def selector_index(self, model, argument):
        """
        Get the index of the argument passed in the selector tuples

        Parameters
        ----------
        model : `~astropy.modeling.Model`
            The Model these selector arguments are for.

        argument : int or str
            A representation of which argument is being used
        """

        for index, selector_arg in enumerate(self):
            if selector_arg.is_argument(model, argument):
                return index
        else:
            raise ValueError(f"{argument} does not correspond to any selector argument.")
```
