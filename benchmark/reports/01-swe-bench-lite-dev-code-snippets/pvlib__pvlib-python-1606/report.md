# pvlib__pvlib-python-1606

| **pvlib/pvlib-python** | `c78b50f4337ecbe536a961336ca91a1176efc0e8` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | - |
| **Avg pos** | - |
| **Min pos** | - |
| **Max pos** | - |
| **Top file pos** | - |
| **Missing snippets** | 3 |
| **Missing patch files** | 1 |


## Expected patch

```diff
diff --git a/pvlib/tools.py b/pvlib/tools.py
--- a/pvlib/tools.py
+++ b/pvlib/tools.py
@@ -341,6 +341,8 @@ def _golden_sect_DataFrame(params, lower, upper, func, atol=1e-8):
     --------
     pvlib.singlediode._pwr_optfcn
     """
+    if np.any(upper - lower < 0.):
+        raise ValueError('upper >= lower is required')
 
     phim1 = (np.sqrt(5) - 1) / 2
 
@@ -349,16 +351,8 @@ def _golden_sect_DataFrame(params, lower, upper, func, atol=1e-8):
     df['VL'] = lower
 
     converged = False
-    iterations = 0
 
-    # handle all NaN case gracefully
-    with warnings.catch_warnings():
-        warnings.filterwarnings(action='ignore',
-                                message='All-NaN slice encountered')
-        iterlimit = 1 + np.nanmax(
-            np.trunc(np.log(atol / (df['VH'] - df['VL'])) / np.log(phim1)))
-
-    while not converged and (iterations <= iterlimit):
+    while not converged:
 
         phi = phim1 * (df['VH'] - df['VL'])
         df['V1'] = df['VL'] + phi
@@ -373,22 +367,16 @@ def _golden_sect_DataFrame(params, lower, upper, func, atol=1e-8):
 
         err = abs(df['V2'] - df['V1'])
 
-        # works with single value because err is np.float64
-        converged = (err[~np.isnan(err)] < atol).all()
-        # err will be less than atol before iterations hit the limit
-        # but just to be safe
-        iterations += 1
-
-    if iterations > iterlimit:
-        raise Exception("Iterations exceeded maximum. Check that func",
-                        " is not NaN in (lower, upper)")  # pragma: no cover
+        # handle all NaN case gracefully
+        with warnings.catch_warnings():
+            warnings.filterwarnings(action='ignore',
+                                    message='All-NaN slice encountered')
+            converged = np.all(err[~np.isnan(err)] < atol)
 
-    try:
-        func_result = func(df, 'V1')
-        x = np.where(np.isnan(func_result), np.nan, df['V1'])
-    except KeyError:
-        func_result = np.full_like(upper, np.nan)
-        x = func_result.copy()
+    # best estimate of location of maximum
+    df['max'] = 0.5 * (df['V1'] + df['V2'])
+    func_result = func(df, 'max')
+    x = np.where(np.isnan(func_result), np.nan, df['max'])
 
     return func_result, x
 

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| pvlib/tools.py | 341 | - | - | - | -
| pvlib/tools.py | 352 | 355 | - | - | -
| pvlib/tools.py | 376 | 379 | - | - | -


## Problem Statement

```
golden-section search fails when upper and lower bounds are equal
**Describe the bug**
I was using pvlib for sometime now and until now I was always passing a big dataframe containing readings of a long period. Because of some changes in our software architecture, I need to pass the weather readings as a single reading (a dataframe with only one row) and I noticed that for readings that GHI-DHI are zero pvlib fails to calculate the output and returns below error while the same code executes correctly with weather information that has non-zero GHI-DHI:
\`\`\`python
import os
import pathlib
import time
import json
from datetime import datetime
from time import mktime, gmtime

import pandas as pd

from pvlib import pvsystem
from pvlib import location as pvlocation
from pvlib import modelchain
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS as PARAMS # not used -- to remove
from pvlib.bifacial.pvfactors import pvfactors_timeseries
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS

class PV:
    def pv_transform_time(self, val):
        # tt = gmtime(val / 1000)
        tt = gmtime(val)
        dd = datetime.fromtimestamp(mktime(tt))
        timestamp = pd.Timestamp(dd)
        return timestamp

    def __init__(self, model: str, inverter: str, latitude: float, longitude: float, **kwargs):
        # super().__init__(**kwargs)

        temperature_model_parameters = TEMPERATURE_MODEL_PARAMETERS["sapm"][
            "open_rack_glass_glass"
        ]
        # Load the database of CEC module model parameters
        modules = pvsystem.retrieve_sam("cecmod")
        # Load the database of CEC inverter model parameters
        inverters = pvsystem.retrieve_sam("cecinverter")


        # A bare bone PV simulator

        # Load the database of CEC module model parameters
        modules = pvsystem.retrieve_sam('cecmod')
        inverters = pvsystem.retrieve_sam('cecinverter')
        module_parameters = modules[model]
        inverter_parameters = inverters[inverter]

        location = pvlocation.Location(latitude=latitude, longitude=longitude)
        system = pvsystem.PVSystem(module_parameters=module_parameters, inverter_parameters=inverter_parameters, temperature_model_parameters=temperature_model_parameters)
        self.modelchain = modelchain.ModelChain(system, location, aoi_model='no_loss', spectral_model="no_loss")

    def process(self, data):
        weather = pd.read_json(data)
        # print(f"raw_weather: {weather}")
        weather.drop('time.1', axis=1, inplace=True)
        weather['time'] = pd.to_datetime(weather['time']).map(datetime.timestamp) # --> this works for the new process_weather code and also the old weather file
        weather["time"] = weather["time"].apply(self.pv_transform_time)
        weather.index = weather["time"]
        # print(f"weather: {weather}")
        # print(weather.dtypes)
        # print(weather['ghi'][0])
        # print(type(weather['ghi'][0]))

        # simulate
        self.modelchain.run_model(weather)
        # print(self.modelchain.results.ac.to_frame().to_json())
        print(self.modelchain.results.ac)


# good data
good_data = "{\"time\":{\"12\":\"2010-01-01 13:30:00+00:00\"},\"ghi\":{\"12\":36},\"dhi\":{\"12\":36},\"dni\":{\"12\":0},\"Tamb\":{\"12\":8.0},\"WindVel\":{\"12\":5.0},\"WindDir\":{\"12\":270},\"time.1\":{\"12\":\"2010-01-01 13:30:00+00:00\"}}"

# data that causes error
data = "{\"time\":{\"4\":\"2010-01-01 05:30:00+00:00\"},\"ghi\":{\"4\":0},\"dhi\":{\"4\":0},\"dni\":{\"4\":0},\"Tamb\":{\"4\":8.0},\"WindVel\":{\"4\":4.0},\"WindDir\":{\"4\":240},\"time.1\":{\"4\":\"2010-01-01 05:30:00+00:00\"}}"
p1 = PV(model="Trina_Solar_TSM_300DEG5C_07_II_", inverter="ABB__MICRO_0_25_I_OUTD_US_208__208V_", latitude=51.204483, longitude=5.265472)
p1.process(good_data)
print("=====")
p1.process(data)
\`\`\`
Error:
\`\`\`log
$ python3 ./tmp-pv.py 
time
2010-01-01 13:30:00    7.825527
dtype: float64
=====
/home/user/.local/lib/python3.10/site-packages/pvlib/tools.py:340: RuntimeWarning: divide by zero encountered in divide
  np.trunc(np.log(atol / (df['VH'] - df['VL'])) / np.log(phim1)))
Traceback (most recent call last):
  File "/home/user/workspace/enorch/simulator/simulator_processor/src/pv/./tmp-pv.py", line 88, in <module>
    p1.process(data)
  File "/home/user/workspace/enorch/simulator/simulator_processor/src/pv/./tmp-pv.py", line 75, in process
    self.modelchain.run_model(weather)
  File "/home/user/.local/lib/python3.10/site-packages/pvlib/modelchain.py", line 1770, in run_model
    self._run_from_effective_irrad(weather)
  File "/home/user/.local/lib/python3.10/site-packages/pvlib/modelchain.py", line 1858, in _run_from_effective_irrad
    self.dc_model()
  File "/home/user/.local/lib/python3.10/site-packages/pvlib/modelchain.py", line 790, in cec
    return self._singlediode(self.system.calcparams_cec)
  File "/home/user/.local/lib/python3.10/site-packages/pvlib/modelchain.py", line 772, in _singlediode
    self.results.dc = tuple(itertools.starmap(
  File "/home/user/.local/lib/python3.10/site-packages/pvlib/pvsystem.py", line 931, in singlediode
    return singlediode(photocurrent, saturation_current,
  File "/home/user/.local/lib/python3.10/site-packages/pvlib/pvsystem.py", line 2826, in singlediode
    out = _singlediode._lambertw(
  File "/home/user/.local/lib/python3.10/site-packages/pvlib/singlediode.py", line 651, in _lambertw
    p_mp, v_mp = _golden_sect_DataFrame(params, 0., v_oc * 1.14,
  File "/home/user/.local/lib/python3.10/site-packages/pvlib/tools.py", line 364, in _golden_sect_DataFrame
    raise Exception("Iterations exceeded maximum. Check that func",
Exception: ('Iterations exceeded maximum. Check that func', ' is not NaN in (lower, upper)')
\`\`\`

I have to mention that for now the workaround that I am using is to pass the weather data as a dataframe with two rows, the first row is a good weather data that pvlib can process and the second row is the incoming weather reading (I can also post that code if you want).

**Expected behavior**
PVlib should have consistent behavior and regardless of GHI-DHI readings.

**Versions:**
\`\`\`python
>>> import pvlib
>>> import pandas
>>> pvlib.__version__
'0.9.1'
>>> pandas.__version__
'1.4.3'
\`\`\` 
 - python: 3.10.6
- OS: Ubuntu 22.04.1 LTS

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- |
| 1 | 1 pvlib/modelchain.py | 1300 | 1331| 399 | 399 | 
| 2 | 2 pvlib/pvsystem.py | 0 | 46| 482 | 881 | 
| 3 | 3 pvlib/irradiance.py | 2010 | 2087| 993 | 1874 | 
| 4 | 4 docs/examples/irradiance-transposition/plot_seasonal_tilt.py | 41 | 98| 612 | 2486 | 
| 5 | 5 pvlib/iotools/sodapro.py | 5 | 40| 448 | 2934 | 
| 6 | 6 docs/examples/bifacial/plot_bifi_model_pvwatts.py | 0 | 98| 828 | 3762 | 
| 7 | 7 pvlib/iotools/__init__.py | 0 | 23| 507 | 4269 | 
| 8 | 8 pvlib/bifacial/pvfactors.py | 86 | 113| 338 | 4607 | 
| 9 | 9 docs/examples/irradiance-transposition/plot_interval_transposition_error.py | 87 | 170| 893 | 5500 | 
| 10 | 9 pvlib/pvsystem.py | 2534 | 2596| 761 | 6261 | 
| 11 | 9 pvlib/modelchain.py | 1230 | 1298| 625 | 6886 | 
| 12 | 10 pvlib/iotools/pvgis.py | 190 | 226| 541 | 7427 | 
| 13 | 10 pvlib/iotools/pvgis.py | 0 | 42| 434 | 7861 | 
| 14 | 11 pvlib/temperature.py | 911 | 958| 780 | 8641 | 
| 15 | 12 docs/examples/bifacial/plot_bifi_model_mc.py | 0 | 89| 800 | 9441 | 
| 16 | 12 pvlib/bifacial/pvfactors.py | 124 | 136| 207 | 9648 | 
| 17 | 12 pvlib/irradiance.py | 1942 | 1969| 441 | 10089 | 
| 18 | 13 docs/examples/solar-tracking/plot_discontinuous_tracking.py | 49 | 89| 423 | 10512 | 
| 19 | 13 pvlib/modelchain.py | 0 | 63| 644 | 11156 | 
| 20 | 13 docs/examples/irradiance-transposition/plot_interval_transposition_error.py | 0 | 46| 501 | 11657 | 
| 21 | 13 pvlib/bifacial/pvfactors.py | 114 | 122| 230 | 11887 | 
| 22 | 14 pvlib/clearsky.py | 95 | 144| 628 | 12515 | 
| 23 | 14 pvlib/iotools/pvgis.py | 55 | 190| 1700 | 14215 | 
| 24 | 14 pvlib/modelchain.py | 266 | 385| 1311 | 15526 | 
| 25 | 14 pvlib/modelchain.py | 1487 | 1566| 692 | 16218 | 
| 26 | 15 pvlib/forecast.py | 796 | 884| 705 | 16923 | 
| 27 | 15 pvlib/temperature.py | 670 | 719| 676 | 17599 | 
| 28 | 15 pvlib/temperature.py | 55 | 147| 858 | 18457 | 
| 29 | 15 pvlib/temperature.py | 614 | 669| 813 | 19270 | 
| 30 | 16 docs/examples/bifacial/plot_pvfactors_fixed_tilt.py | 0 | 68| 583 | 19853 | 
| 31 | 17 pvlib/inverter.py | 281 | 329| 563 | 20416 | 
| 32 | 18 pvlib/iotools/psm3.py | 0 | 40| 401 | 20817 | 
| 33 | 19 pvlib/iotools/tmy.py | 261 | 387| 1904 | 22721 | 
| 34 | 19 pvlib/modelchain.py | 161 | 222| 668 | 23389 | 
| 35 | 20 setup.py | 0 | 93| 898 | 24287 | 
| 36 | 21 pvlib/ivtools/sdm.py | 303 | 416| 1155 | 25442 | 
| 37 | 21 pvlib/clearsky.py | 837 | 858| 267 | 25709 | 
| 38 | 21 docs/examples/bifacial/plot_bifi_model_pvwatts.py | 100 | 115| 117 | 25826 | 
| 39 | 21 pvlib/clearsky.py | 752 | 836| 839 | 26665 | 
| 40 | 21 docs/examples/bifacial/plot_bifi_model_mc.py | 90 | 120| 253 | 26918 | 
| 41 | 21 pvlib/forecast.py | 0 | 28| 225 | 27143 | 
| 42 | 21 pvlib/temperature.py | 291 | 387| 1090 | 28233 | 
| 43 | 21 pvlib/irradiance.py | 1435 | 1456| 259 | 28492 | 
| 44 | 21 pvlib/iotools/pvgis.py | 449 | 503| 651 | 29143 | 
| 45 | 21 pvlib/temperature.py | 809 | 825| 264 | 29407 | 
| 46 | 22 docs/examples/soiling/plot_fig3A_hsu_soiling_example.py | 0 | 79| 768 | 30175 | 
| 47 | 22 pvlib/forecast.py | 771 | 793| 193 | 30368 | 
| 48 | 22 pvlib/pvsystem.py | 3173 | 3220| 473 | 30841 | 
| 49 | 22 pvlib/iotools/tmy.py | 8 | 158| 2214 | 33055 | 
| 50 | 22 pvlib/ivtools/sdm.py | 418 | 478| 718 | 33773 | 
| 51 | 22 pvlib/inverter.py | 508 | 544| 472 | 34245 | 
| 52 | 22 pvlib/forecast.py | 1187 | 1211| 193 | 34438 | 
| 53 | 22 pvlib/modelchain.py | 1568 | 1592| 305 | 34743 | 
| 54 | 22 pvlib/iotools/tmy.py | 387 | 395| 583 | 35326 | 
| 55 | 22 pvlib/modelchain.py | 66 | 159| 707 | 36033 | 
| 56 | 23 docs/examples/iv-modeling/plot_singlediode.py | 0 | 75| 709 | 36742 | 
| 57 | 23 pvlib/clearsky.py | 861 | 934| 770 | 37512 | 
| 58 | 23 pvlib/forecast.py | 1103 | 1124| 157 | 37669 | 
| 59 | 23 pvlib/temperature.py | 390 | 456| 686 | 38355 | 
| 60 | 23 pvlib/iotools/pvgis.py | 517 | 554| 607 | 38962 | 
| 61 | 23 pvlib/forecast.py | 943 | 966| 193 | 39155 | 
| 62 | 24 docs/examples/reflections/plot_diffuse_aoi_correction.py | 66 | 119| 614 | 39769 | 
| 63 | 24 pvlib/pvsystem.py | 100 | 220| 1049 | 40818 | 
| 64 | 25 docs/examples/spectrum/plot_spectrl2_fig51A.py | 0 | 79| 765 | 41583 | 
| 65 | 25 pvlib/pvsystem.py | 1844 | 2004| 1547 | 43130 | 
| 66 | 25 pvlib/temperature.py | 959 | 978| 267 | 43397 | 
| 67 | 25 pvlib/pvsystem.py | 1117 | 1252| 894 | 44291 | 
| 68 | 25 pvlib/iotools/pvgis.py | 45 | 226| 119 | 44410 | 
| 69 | 25 pvlib/forecast.py | 1027 | 1055| 263 | 44673 | 
| 70 | 25 pvlib/forecast.py | 30 | 129| 756 | 45429 | 
| 71 | 25 pvlib/pvsystem.py | 3223 | 3274| 406 | 45835 | 
| 72 | 25 pvlib/ivtools/sdm.py | 735 | 791| 759 | 46594 | 
| 73 | 25 pvlib/iotools/pvgis.py | 241 | 288| 545 | 47139 | 
| 74 | 25 pvlib/modelchain.py | 1747 | 1816| 526 | 47665 | 
| 75 | 25 pvlib/temperature.py | 1037 | 1111| 642 | 48307 | 
| 76 | 25 pvlib/forecast.py | 131 | 155| 186 | 48493 | 
| 77 | 26 pvlib/solarposition.py | 106 | 128| 229 | 48722 | 
| 78 | 26 pvlib/clearsky.py | 592 | 616| 282 | 49004 | 
| 79 | 26 pvlib/iotools/tmy.py | 235 | 257| 440 | 49444 | 


## Missing Patch Files

 * 1: pvlib/tools.py

### Hint

```
Confirmed. This appears to be an oversight in `pvlib.tools._golden_section_DataFrame` involving error messaging, likely introduced with #1089 .

In this code when processing the content of `data`, photocurrent is 0., hence the shunt resistance is infinite and v_oc is 0. That sets the range for the golden section search to be [0., 0.]. [iterlimit](https://github.com/pvlib/pvlib-python/blob/582b956c63c463e5178fbb7a88fa545fa5b1c257/pvlib/tools.py#L358) is then -infinity, which skips the loop (`iterations <= iterlimit`) but since `iterations > iterlimit` raises the "Iterations exceeded..." exception.

```

## Patch

```diff
diff --git a/pvlib/tools.py b/pvlib/tools.py
--- a/pvlib/tools.py
+++ b/pvlib/tools.py
@@ -341,6 +341,8 @@ def _golden_sect_DataFrame(params, lower, upper, func, atol=1e-8):
     --------
     pvlib.singlediode._pwr_optfcn
     """
+    if np.any(upper - lower < 0.):
+        raise ValueError('upper >= lower is required')
 
     phim1 = (np.sqrt(5) - 1) / 2
 
@@ -349,16 +351,8 @@ def _golden_sect_DataFrame(params, lower, upper, func, atol=1e-8):
     df['VL'] = lower
 
     converged = False
-    iterations = 0
 
-    # handle all NaN case gracefully
-    with warnings.catch_warnings():
-        warnings.filterwarnings(action='ignore',
-                                message='All-NaN slice encountered')
-        iterlimit = 1 + np.nanmax(
-            np.trunc(np.log(atol / (df['VH'] - df['VL'])) / np.log(phim1)))
-
-    while not converged and (iterations <= iterlimit):
+    while not converged:
 
         phi = phim1 * (df['VH'] - df['VL'])
         df['V1'] = df['VL'] + phi
@@ -373,22 +367,16 @@ def _golden_sect_DataFrame(params, lower, upper, func, atol=1e-8):
 
         err = abs(df['V2'] - df['V1'])
 
-        # works with single value because err is np.float64
-        converged = (err[~np.isnan(err)] < atol).all()
-        # err will be less than atol before iterations hit the limit
-        # but just to be safe
-        iterations += 1
-
-    if iterations > iterlimit:
-        raise Exception("Iterations exceeded maximum. Check that func",
-                        " is not NaN in (lower, upper)")  # pragma: no cover
+        # handle all NaN case gracefully
+        with warnings.catch_warnings():
+            warnings.filterwarnings(action='ignore',
+                                    message='All-NaN slice encountered')
+            converged = np.all(err[~np.isnan(err)] < atol)
 
-    try:
-        func_result = func(df, 'V1')
-        x = np.where(np.isnan(func_result), np.nan, df['V1'])
-    except KeyError:
-        func_result = np.full_like(upper, np.nan)
-        x = func_result.copy()
+    # best estimate of location of maximum
+    df['max'] = 0.5 * (df['V1'] + df['V2'])
+    func_result = func(df, 'max')
+    x = np.where(np.isnan(func_result), np.nan, df['max'])
 
     return func_result, x
 

```

## Test Patch

```diff
diff --git a/pvlib/tests/test_tools.py b/pvlib/tests/test_tools.py
--- a/pvlib/tests/test_tools.py
+++ b/pvlib/tests/test_tools.py
@@ -45,6 +45,22 @@ def test__golden_sect_DataFrame_vector():
     v, x = tools._golden_sect_DataFrame(params, lower, upper,
                                         _obj_test_golden_sect)
     assert np.allclose(x, expected, atol=1e-8)
+    # some upper and lower bounds equal
+    params = {'c': np.array([1., 2., 1.]), 'n': np.array([1., 1., 1.])}
+    lower = np.array([0., 0.001, 1.])
+    upper = np.array([1., 1.2, 1.])
+    expected = np.array([0.5, 0.25, 1.0])  # x values for maxima
+    v, x = tools._golden_sect_DataFrame(params, lower, upper,
+                                        _obj_test_golden_sect)
+    assert np.allclose(x, expected, atol=1e-8)
+    # all upper and lower bounds equal, arrays of length 1
+    params = {'c': np.array([1.]), 'n': np.array([1.])}
+    lower = np.array([1.])
+    upper = np.array([1.])
+    expected = np.array([1.])  # x values for maxima
+    v, x = tools._golden_sect_DataFrame(params, lower, upper,
+                                        _obj_test_golden_sect)
+    assert np.allclose(x, expected, atol=1e-8)
 
 
 def test__golden_sect_DataFrame_nans():

```


## Code snippets

### 1 - pvlib/modelchain.py:

Start line: 1300, End line: 1331

```python
class ModelChain:

    def _complete_irradiance(self, weather):
        icolumns = set(weather.columns)
        wrn_txt = ("This function is not safe at the moment.\n" +
                   "Results can be too high or negative.\n" +
                   "Help to improve this function on github:\n" +
                   "https://github.com/pvlib/pvlib-python \n")
        if {'ghi', 'dhi'} <= icolumns and 'dni' not in icolumns:
            clearsky = self.location.get_clearsky(
                weather.index, solar_position=self.results.solar_position)
            complete_irrad_df = pvlib.irradiance.complete_irradiance(
                solar_zenith=self.results.solar_position.zenith,
                ghi=weather.ghi,
                dhi=weather.dhi,
                dni=None,
                dni_clear=clearsky.dni)
            weather.loc[:, 'dni'] = complete_irrad_df.dni
        elif {'dni', 'dhi'} <= icolumns and 'ghi' not in icolumns:
            warnings.warn(wrn_txt, UserWarning)
            complete_irrad_df = pvlib.irradiance.complete_irradiance(
                solar_zenith=self.results.solar_position.zenith,
                ghi=None,
                dhi=weather.dhi,
                dni=weather.dni)
            weather.loc[:, 'ghi'] = complete_irrad_df.ghi
        elif {'dni', 'ghi'} <= icolumns and 'dhi' not in icolumns:
            warnings.warn(wrn_txt, UserWarning)
            complete_irrad_df = pvlib.irradiance.complete_irradiance(
                solar_zenith=self.results.solar_position.zenith,
                ghi=weather.ghi,
                dhi=None,
                dni=weather.dni)
            weather.loc[:, 'dhi'] = complete_irrad_df.dhi
```
### 2 - pvlib/pvsystem.py:

```python
"""
The ``pvsystem`` module contains functions for modeling the output and
performance of PV modules and inverters.
"""

from collections import OrderedDict
import functools
import io
import itertools
import os
from urllib.request import urlopen
import numpy as np
import pandas as pd
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Optional

from pvlib._deprecation import deprecated

from pvlib import (atmosphere, iam, inverter, irradiance,
                   singlediode as _singlediode, temperature)
from pvlib.tools import _build_kwargs, _build_args


# a dict of required parameter names for each DC power model
_DC_MODEL_PARAMS = {
    'sapm': {
        'A0', 'A1', 'A2', 'A3', 'A4', 'B0', 'B1', 'B2', 'B3',
        'B4', 'B5', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6',
        'C7', 'Isco', 'Impo', 'Voco', 'Vmpo', 'Aisc', 'Aimp', 'Bvoco',
        'Mbvoc', 'Bvmpo', 'Mbvmp', 'N', 'Cells_in_Series',
        'IXO', 'IXXO', 'FD'},
    'desoto': {
        'alpha_sc', 'a_ref', 'I_L_ref', 'I_o_ref',
        'R_sh_ref', 'R_s'},
    'cec': {
        'alpha_sc', 'a_ref', 'I_L_ref', 'I_o_ref',
        'R_sh_ref', 'R_s', 'Adjust'},
    'pvsyst': {
        'gamma_ref', 'mu_gamma', 'I_L_ref', 'I_o_ref',
        'R_sh_ref', 'R_sh_0', 'R_s', 'alpha_sc', 'EgRef',
        'cells_in_series'},
    'singlediode': {
        'alpha_sc', 'a_ref', 'I_L_ref', 'I_o_ref',
        'R_sh_ref', 'R_s'},
    'pvwatts': {'pdc0', 'gamma_pdc'}
}
```
### 3 - pvlib/irradiance.py:

Start line: 2010, End line: 2087

```python
def _gti_dirint_lt_90(poa_global, aoi, aoi_lt_90, solar_zenith, solar_azimuth,
                      times, surface_tilt, surface_azimuth, pressure=101325.,
                      use_delta_kt_prime=True, temp_dew=None, albedo=.25,
                      model='perez', model_perez='allsitescomposite1990',
                      max_iterations=30):
    # ... other code

    for iteration, coeff in enumerate(coeffs):

        # test if difference between modeled GTI and
        # measured GTI (poa_global) is less than 1 W/m^2
        # only test for aoi less than 90 deg
        best_diff_lte_1 = best_diff <= 1
        best_diff_lte_1_lt_90 = best_diff_lte_1[aoi_lt_90]
        if best_diff_lte_1_lt_90.all():
            # all aoi < 90 points have a difference <= 1, so break loop
            break

        # calculate kt and DNI from GTI
        kt = clearness_index(poa_global_i, aoi, I0)  # kt from Marion eqn 2
        disc_dni = np.maximum(_disc_kn(kt, airmass)[0] * I0, 0)
        kt_prime = clearness_index_zenith_independent(kt, airmass)
        # dirint DNI in Marion eqn 3
        dni = _dirint_from_dni_ktprime(disc_dni, kt_prime, solar_zenith,
                                       use_delta_kt_prime, temp_dew)

        # calculate DHI using Marion eqn 3 (identify 1st term on RHS as GHI)
        # I0h has a minimum zenith projection, but multiplier of DNI does not
        ghi = kt * I0h                  # Kt * I0 * max(0.065, cos(zen))
        dhi = ghi - dni * cos_zenith    # no cos(zen) restriction here

        # following SSC code
        dni = np.maximum(dni, 0)
        ghi = np.maximum(ghi, 0)
        dhi = np.maximum(dhi, 0)

        # use DNI and DHI to model GTI
        # GTI-DIRINT uses perez transposition model, but we allow for
        # any model here
        all_irrad = get_total_irradiance(
            surface_tilt, surface_azimuth, solar_zenith, solar_azimuth,
            dni, ghi, dhi, dni_extra=I0, airmass=airmass,
            albedo=albedo, model=model, model_perez=model_perez)

        gti_model = all_irrad['poa_global']

        # calculate new diff
        diff = gti_model - poa_global

        # determine if the new diff is smaller in magnitude
        # than the old diff
        diff_abs = diff.abs()
        smallest_diff = diff_abs < best_diff

        # save the best differences
        best_diff = diff_abs.where(smallest_diff, best_diff)

        # on first iteration, the best values are the only values
        if iteration == 0:
            best_ghi = ghi
            best_dni = dni
            best_dhi = dhi
            best_kt_prime = kt_prime
        else:
            # save new DNI, DHI, DHI if they provide the best consistency
            # otherwise use the older values.
            best_ghi = ghi.where(smallest_diff, best_ghi)
            best_dni = dni.where(smallest_diff, best_dni)
            best_dhi = dhi.where(smallest_diff, best_dhi)
            best_kt_prime = kt_prime.where(smallest_diff, best_kt_prime)

        # calculate adjusted inputs for next iteration. Marion eqn 4
        poa_global_i = np.maximum(1.0, poa_global_i - coeff * diff)
    else:
        # we are here because we ran out of coeffs to loop over and
        # therefore we have exceeded max_iterations
        import warnings
        failed_points = best_diff[aoi_lt_90][~best_diff_lte_1_lt_90]
        warnings.warn(
            ('%s points failed to converge after %s iterations. best_diff:\n%s'
             % (len(failed_points), max_iterations, failed_points)),
            RuntimeWarning)

    # return the best data, whether or not the solution converged
    return best_ghi, best_dni, best_dhi, best_kt_prime
```
### 4 - docs/examples/irradiance-transposition/plot_seasonal_tilt.py:

Start line: 41, End line: 98

```python
# %%
# First let's grab some weather data and make sure our mount produces tilts
# like we expect:

DATA_DIR = pathlib.Path(pvlib.__file__).parent / 'data'
tmy, metadata = iotools.read_tmy3(DATA_DIR / '723170TYA.CSV', coerce_year=1990)
# shift from TMY3 right-labeled index to left-labeled index:
tmy.index = tmy.index - pd.Timedelta(hours=1)
weather = pd.DataFrame({
    'ghi': tmy['GHI'], 'dhi': tmy['DHI'], 'dni': tmy['DNI'],
    'temp_air': tmy['DryBulb'], 'wind_speed': tmy['Wspd'],
})
loc = location.Location.from_tmy(metadata)
solpos = loc.get_solarposition(weather.index)
# same default monthly tilts as SAM:
tilts = [40, 40, 40, 20, 20, 20, 20, 20, 20, 40, 40, 40]
mount = SeasonalTiltMount(monthly_tilts=tilts)
orientation = mount.get_orientation(solpos.apparent_zenith, solpos.azimuth)
orientation['surface_tilt'].plot()
plt.ylabel('Surface Tilt [degrees]')
plt.show()

# %%
# With our custom tilt strategy defined, we can create the corresponding
# Array and PVSystem, and then run a ModelChain as usual:

module_parameters = {'pdc0': 1, 'gamma_pdc': -0.004, 'b': 0.05}
temp_params = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_polymer']
array = pvsystem.Array(mount=mount, module_parameters=module_parameters,
                       temperature_model_parameters=temp_params)
system = pvsystem.PVSystem(arrays=[array], inverter_parameters={'pdc0': 1})
mc = modelchain.ModelChain(system, loc, spectral_model='no_loss')

_ = mc.run_model(weather)

# %%
# Now let's re-run the simulation assuming tilt=30 for the entire year:

array2 = pvsystem.Array(mount=pvsystem.FixedMount(30, 180),
                        module_parameters=module_parameters,
                        temperature_model_parameters=temp_params)
system2 = pvsystem.PVSystem(arrays=[array2], inverter_parameters={'pdc0': 1})
mc2 = modelchain.ModelChain(system2, loc, spectral_model='no_loss')
_ = mc2.run_model(weather)

# %%
# And finally, compare simulated monthly generation between the two tilt
# strategies:

# sphinx_gallery_thumbnail_number = 2
results = pd.DataFrame({
    'Seasonal 20/40 Production': mc.results.ac,
    'Fixed 30 Production': mc2.results.ac,
})
results.resample('m').sum().plot()
plt.ylabel('Monthly Production')
plt.show()
```
### 5 - pvlib/iotools/sodapro.py:

Start line: 5, End line: 40

```python
import pandas as pd
import requests
import io
import warnings


CAMS_INTEGRATED_COLUMNS = [
    'TOA', 'Clear sky GHI', 'Clear sky BHI', 'Clear sky DHI', 'Clear sky BNI',
    'GHI', 'BHI', 'DHI', 'BNI',
    'GHI no corr', 'BHI no corr', 'DHI no corr', 'BNI no corr']

# Dictionary mapping CAMS Radiation and McClear variables to pvlib names
VARIABLE_MAP = {
    'TOA': 'ghi_extra',
    'Clear sky GHI': 'ghi_clear',
    'Clear sky BHI': 'bhi_clear',
    'Clear sky DHI': 'dhi_clear',
    'Clear sky BNI': 'dni_clear',
    'GHI': 'ghi',
    'BHI': 'bhi',
    'DHI': 'dhi',
    'BNI': 'dni',
    'sza': 'solar_zenith',
}

# Dictionary mapping time steps to CAMS time step format
TIME_STEPS_MAP = {'1min': 'PT01M', '15min': 'PT15M', '1h': 'PT01H',
                  '1d': 'P01D', '1M': 'P01M'}

TIME_STEPS_IN_HOURS = {'1min': 1/60, '15min': 15/60, '1h': 1, '1d': 24}

SUMMATION_PERIOD_TO_TIME_STEP = {'0 year 0 month 0 day 0 h 1 min 0 s': '1min',
                                 '0 year 0 month 0 day 0 h 15 min 0 s': '15min',  # noqa
                                 '0 year 0 month 0 day 1 h 0 min 0 s': '1h',
                                 '0 year 0 month 1 day 0 h 0 min 0 s': '1d',
                                 '0 year 1 month 0 day 0 h 0 min 0 s': '1M'}
```
### 6 - docs/examples/bifacial/plot_bifi_model_pvwatts.py:

```python
"""
Bifacial Modeling - procedural
==============================

Example of bifacial modeling using pvfactors and procedural method
"""

# %%
# This example shows how to complete a bifacial modeling example using the
# :py:func:`pvlib.pvsystem.pvwatts_dc` with the
# :py:func:`pvlib.bifacial.pvfactors.pvfactors_timeseries` function to
# transpose GHI data to both front and rear Plane of Array (POA) irradiance.

import pandas as pd
from pvlib import location
from pvlib import tracking
from pvlib.bifacial.pvfactors import pvfactors_timeseries
from pvlib import temperature
from pvlib import pvsystem
import matplotlib.pyplot as plt
import warnings

# supressing shapely warnings that occur on import of pvfactors
warnings.filterwarnings(action='ignore', module='pvfactors')

# using Greensboro, NC for this example
lat, lon = 36.084, -79.817
tz = 'Etc/GMT+5'
times = pd.date_range('2021-06-21', '2021-06-22', freq='1T', tz=tz)

# create location object and get clearsky data
site_location = location.Location(lat, lon, tz=tz, name='Greensboro, NC')
cs = site_location.get_clearsky(times)

# get solar position data
solar_position = site_location.get_solarposition(times)

# set ground coverage ratio and max_angle to
# pull orientation data for a single-axis tracker
gcr = 0.35
max_phi = 60
orientation = tracking.singleaxis(solar_position['apparent_zenith'],
                                  solar_position['azimuth'],
                                  max_angle=max_phi,
                                  backtrack=True,
                                  gcr=gcr
                                  )

# set axis_azimuth, albedo, pvrow width and height, and use
# the pvfactors engine for both front and rear-side absorbed irradiance
axis_azimuth = 180
pvrow_height = 3
pvrow_width = 4
albedo = 0.2

# explicity simulate on pvarray with 3 rows, with sensor placed in middle row
# users may select different values depending on needs
irrad = pvfactors_timeseries(solar_position['azimuth'],
                             solar_position['apparent_zenith'],
                             orientation['surface_azimuth'],
                             orientation['surface_tilt'],
                             axis_azimuth,
                             cs.index,
                             cs['dni'],
                             cs['dhi'],
                             gcr,
                             pvrow_height,
                             pvrow_width,
                             albedo,
                             n_pvrows=3,
                             index_observed_pvrow=1
                             )

# turn into pandas DataFrame
irrad = pd.concat(irrad, axis=1)

# using bifaciality factor and pvfactors results, create effective irradiance
bifaciality = 0.75
effective_irrad_bifi = irrad['total_abs_front'] + (irrad['total_abs_back']
                                                   * bifaciality)

# get cell temperature using the Faiman model
temp_cell = temperature.faiman(effective_irrad_bifi, temp_air=25,
                               wind_speed=1)

# using the pvwatts_dc model and parameters detailed above,
# set pdc0 and return DC power for both bifacial and monofacial
pdc0 = 1
gamma_pdc = -0.0043
pdc_bifi = pvsystem.pvwatts_dc(effective_irrad_bifi,
                               temp_cell,
                               pdc0,
                               gamma_pdc=gamma_pdc
                               ).fillna(0)
pdc_bifi.plot(title='Bifacial Simulation on June Solstice', ylabel='DC Power')

# %%
# For illustration, perform monofacial simulation using pvfactors front-side
# irradiance (AOI-corrected), and plot along with bifacial results.
```
### 7 - pvlib/iotools/__init__.py:

```python
from pvlib.iotools.tmy import read_tmy2, read_tmy3  # noqa: F401
from pvlib.iotools.epw import read_epw, parse_epw  # noqa: F401
from pvlib.iotools.srml import read_srml  # noqa: F401
from pvlib.iotools.srml import read_srml_month_from_solardat  # noqa: F401
from pvlib.iotools.surfrad import read_surfrad  # noqa: F401
from pvlib.iotools.midc import read_midc  # noqa: F401
from pvlib.iotools.midc import read_midc_raw_data_from_nrel  # noqa: F401
from pvlib.iotools.ecmwf_macc import read_ecmwf_macc  # noqa: F401
from pvlib.iotools.ecmwf_macc import get_ecmwf_macc  # noqa: F401
from pvlib.iotools.crn import read_crn  # noqa: F401
from pvlib.iotools.solrad import read_solrad  # noqa: F401
from pvlib.iotools.psm3 import get_psm3  # noqa: F401
from pvlib.iotools.psm3 import read_psm3  # noqa: F401
from pvlib.iotools.psm3 import parse_psm3  # noqa: F401
from pvlib.iotools.pvgis import get_pvgis_tmy, read_pvgis_tmy  # noqa: F401
from pvlib.iotools.pvgis import read_pvgis_hourly  # noqa: F401
from pvlib.iotools.pvgis import get_pvgis_hourly  # noqa: F401
from pvlib.iotools.bsrn import get_bsrn  # noqa: F401
from pvlib.iotools.bsrn import read_bsrn  # noqa: F401
from pvlib.iotools.bsrn import parse_bsrn  # noqa: F401
from pvlib.iotools.sodapro import get_cams  # noqa: F401
from pvlib.iotools.sodapro import read_cams  # noqa: F401
from pvlib.iotools.sodapro import parse_cams  # noqa: F401
```
### 8 - pvlib/bifacial/pvfactors.py:

Start line: 86, End line: 113

```python
def pvfactors_timeseries(
        solar_azimuth, solar_zenith, surface_azimuth, surface_tilt,
        axis_azimuth, timestamps, dni, dhi, gcr, pvrow_height, pvrow_width,
        albedo, n_pvrows=3, index_observed_pvrow=1,
        rho_front_pvrow=0.03, rho_back_pvrow=0.05,
        horizon_band_angle=15.):
    # Convert Series, list, float inputs to numpy arrays
    solar_azimuth = np.array(solar_azimuth)
    solar_zenith = np.array(solar_zenith)
    dni = np.array(dni)
    dhi = np.array(dhi)
    # GH 1127, GH 1332
    surface_tilt = np.full_like(solar_zenith, surface_tilt)
    surface_azimuth = np.full_like(solar_zenith, surface_azimuth)

    # Import pvfactors functions for timeseries calculations.
    from pvfactors.run import run_timeseries_engine

    # Build up pv array configuration parameters
    pvarray_parameters = {
        'n_pvrows': n_pvrows,
        'axis_azimuth': axis_azimuth,
        'pvrow_height': pvrow_height,
        'pvrow_width': pvrow_width,
        'gcr': gcr
    }

    irradiance_model_params = {
        'rho_front': rho_front_pvrow,
        'rho_back': rho_back_pvrow,
        'horizon_band_angle': horizon_band_angle
    }

    # Create report function
    # ... other code
```
### 9 - docs/examples/irradiance-transposition/plot_interval_transposition_error.py:

Start line: 87, End line: 170

```python
# %%
# Now, calculate the "ground truth" irradiance data.  We'll simulate
# clear-sky irradiance components at 1-second intervals and calculate
# the corresponding POA irradiance.  At such a short timescale, the
# difference between instantaneous and interval-averaged irradiance
# is negligible.

# baseline: all calculations done at 1-second scale
location = pvlib.location.Location(40, -80, tz='Etc/GMT+5')
times = pd.date_range('2019-06-01 05:00', '2019-06-01 19:00',
                      freq='1s', tz='Etc/GMT+5')
solpos = location.get_solarposition(times)
clearsky = location.get_clearsky(times, solar_position=solpos)
poa_1s = transpose(clearsky, timeshift=0)  # no shift needed for 1s data

# %%
# Now, we will aggregate the 1-second values into interval averages.
# To see how the averaging interval affects results, we'll loop over
# a few common data intervals and accumulate the results.

fig, ax = plt.subplots(figsize=(5, 3))

results = []

for timescale_minutes in [1, 5, 10, 15, 30, 60]:

    timescale_str = f'{timescale_minutes}min'
    # get the "true" interval average of poa as the baseline for comparison
    poa_avg = poa_1s.resample(timescale_str).mean()
    # get interval averages of irradiance components to use for transposition
    clearsky_avg = clearsky.resample(timescale_str).mean()

    # low-res interval averages of 1-second data, with NO shift
    poa_avg_noshift = transpose(clearsky_avg, timeshift=0)

    # low-res interval averages of 1-second data, with half-interval shift
    poa_avg_halfshift = transpose(clearsky_avg, timeshift=timescale_minutes/2)

    df = pd.DataFrame({
        'ground truth': poa_avg,
        'modeled, half shift': poa_avg_halfshift,
        'modeled, no shift': poa_avg_noshift,
    })
    error = df.subtract(df['ground truth'], axis=0)
    # add another trace to the error plot
    error['modeled, no shift'].plot(ax=ax, label=timescale_str)
    # calculate error statistics and save for later
    stats = error.abs().mean()  # average absolute error across daylight hours
    stats['timescale_minutes'] = timescale_minutes
    results.append(stats)

ax.legend(ncol=2)
ax.set_ylabel('Transposition Error [W/m$^2$]')
fig.tight_layout()

df_results = pd.DataFrame(results).set_index('timescale_minutes')
print(df_results)

# %%
# The errors shown above are the average absolute difference in :math:`W/m^2`.
# In this example, using the timestamps unadjusted creates an error that
# increases with increasing interval length, up to a ~40% error
# at hourly resolution.  In contrast, incorporating a half-interval shift
# so that solar position is calculated in the middle of the interval
# instead of the edge reduces the error by one or two orders of magnitude:

fig, ax = plt.subplots(figsize=(5, 3))
df_results[['modeled, no shift', 'modeled, half shift']].plot.bar(rot=0, ax=ax)
ax.set_ylabel('Mean Absolute Error [W/m$^2$]')
ax.set_xlabel('Transposition Timescale [minutes]')
fig.tight_layout()

# %%
# We can also plot the underlying time series results of the last
# iteration (hourly in this case).  The modeled irradiance using
# no shift is effectively time-lagged compared with ground truth.
# In contrast, the half-shift model is nearly identical to the ground
# truth irradiance.

fig, ax = plt.subplots(figsize=(5, 3))
ax = df.plot(ax=ax, style=[None, ':', None], lw=3)
ax.set_ylabel('Irradiance [W/m$^2$]')
fig.tight_layout()
```
### 10 - pvlib/pvsystem.py:

Start line: 2534, End line: 2596

```python
def sapm(effective_irradiance, temp_cell, module):

    # TODO: someday, change temp_ref and irrad_ref to reference_temperature and
    # reference_irradiance and expose
    temp_ref = 25
    irrad_ref = 1000

    q = 1.60218e-19  # Elementary charge in units of coulombs
    kb = 1.38066e-23  # Boltzmann's constant in units of J/K

    # avoid problem with integer input
    Ee = np.array(effective_irradiance, dtype='float64') / irrad_ref

    # set up masking for 0, positive, and nan inputs
    Ee_gt_0 = np.full_like(Ee, False, dtype='bool')
    Ee_eq_0 = np.full_like(Ee, False, dtype='bool')
    notnan = ~np.isnan(Ee)
    np.greater(Ee, 0, where=notnan, out=Ee_gt_0)
    np.equal(Ee, 0, where=notnan, out=Ee_eq_0)

    Bvmpo = module['Bvmpo'] + module['Mbvmp']*(1 - Ee)
    Bvoco = module['Bvoco'] + module['Mbvoc']*(1 - Ee)
    delta = module['N'] * kb * (temp_cell + 273.15) / q

    # avoid repeated computation
    logEe = np.full_like(Ee, np.nan)
    np.log(Ee, where=Ee_gt_0, out=logEe)
    logEe = np.where(Ee_eq_0, -np.inf, logEe)
    # avoid repeated __getitem__
    cells_in_series = module['Cells_in_Series']

    out = OrderedDict()

    out['i_sc'] = (
        module['Isco'] * Ee * (1 + module['Aisc']*(temp_cell - temp_ref)))

    out['i_mp'] = (
        module['Impo'] * (module['C0']*Ee + module['C1']*(Ee**2)) *
        (1 + module['Aimp']*(temp_cell - temp_ref)))

    out['v_oc'] = np.maximum(0, (
        module['Voco'] + cells_in_series * delta * logEe +
        Bvoco*(temp_cell - temp_ref)))

    out['v_mp'] = np.maximum(0, (
        module['Vmpo'] +
        module['C2'] * cells_in_series * delta * logEe +
        module['C3'] * cells_in_series * ((delta * logEe) ** 2) +
        Bvmpo*(temp_cell - temp_ref)))

    out['p_mp'] = out['i_mp'] * out['v_mp']

    out['i_x'] = (
        module['IXO'] * (module['C4']*Ee + module['C5']*(Ee**2)) *
        (1 + module['Aisc']*(temp_cell - temp_ref)))

    # the Ixx calculation in King 2004 has a typo (mixes up Aisc and Aimp)
    out['i_xx'] = (
        module['IXXO'] * (module['C6']*Ee + module['C7']*(Ee**2)) *
        (1 + module['Aisc']*(temp_cell - temp_ref)))

    if isinstance(out['i_sc'], pd.Series):
        out = pd.DataFrame(out)

    return out
```
