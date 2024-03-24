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
| 1 | 1 pvlib/modelchain.py | 0 | 2063| 17399 | 17399 | 
| 2 | 2 pvlib/pvsystem.py | 0 | 3400| 29030 | 46429 | 


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

```python
"""
The ``modelchain`` module contains functions and classes that combine
many of the PV power modeling steps. These tools make it easy to
get started with pvlib and demonstrate standard ways to use the
library. With great power comes great responsibility: users should take
the time to read the source code for the module.
"""

from functools import partial
import itertools
import warnings
import pandas as pd
from dataclasses import dataclass, field
from typing import Union, Tuple, Optional, TypeVar

from pvlib import (atmosphere, clearsky, inverter, pvsystem, solarposition,
                   temperature, tools)
from pvlib.tracking import SingleAxisTracker
import pvlib.irradiance  # avoid name conflict with full import
from pvlib.pvsystem import _DC_MODEL_PARAMS
from pvlib._deprecation import pvlibDeprecationWarning
from pvlib.tools import _build_kwargs

from pvlib._deprecation import deprecated

# keys that are used to detect input data and assign data to appropriate
# ModelChain attribute
# for ModelChain.weather
WEATHER_KEYS = ('ghi', 'dhi', 'dni', 'wind_speed', 'temp_air',
                'precipitable_water')

# for ModelChain.total_irrad
POA_KEYS = ('poa_global', 'poa_direct', 'poa_diffuse')

# Optional keys to communicate temperature data. If provided,
# 'cell_temperature' overrides ModelChain.temperature_model and sets
# ModelChain.cell_temperature to the data. If 'module_temperature' is provdied,
# overrides ModelChain.temperature_model with
# pvlib.temperature.sapm_celL_from_module
TEMPERATURE_KEYS = ('module_temperature', 'cell_temperature')

DATA_KEYS = WEATHER_KEYS + POA_KEYS + TEMPERATURE_KEYS

# these dictionaries contain the default configuration for following
# established modeling sequences. They can be used in combination with
# basic_chain and ModelChain. They are used by the ModelChain methods
# ModelChain.with_pvwatts, ModelChain.with_sapm, etc.

# pvwatts documentation states that it uses the following reference for
# a temperature model: Fuentes, M. K. (1987). A Simplified Thermal Model
# for Flat-Plate Photovoltaic Arrays. SAND85-0330. Albuquerque, NM:
# Sandia National Laboratories. Accessed September 3, 2013:
# http://prod.sandia.gov/techlib/access-control.cgi/1985/850330.pdf
# pvlib python does not implement that model, so use the SAPM instead.
PVWATTS_CONFIG = dict(
    dc_model='pvwatts', ac_model='pvwatts', losses_model='pvwatts',
    transposition_model='perez', aoi_model='physical',
    spectral_model='no_loss', temperature_model='sapm'
)

SAPM_CONFIG = dict(
    dc_model='sapm', ac_model='sandia', losses_model='no_loss',
    aoi_model='sapm', spectral_model='sapm', temperature_model='sapm'
)


@deprecated(
    since='0.9.1',
    name='pvlib.modelchain.basic_chain',
    alternative=('pvlib.modelchain.ModelChain.with_pvwatts'
                 ' or pvlib.modelchain.ModelChain.with_sapm'),
    addendum='Note that the with_xyz methods take different model parameters.'
)
def basic_chain(times, latitude, longitude,
                surface_tilt, surface_azimuth,
                module_parameters, temperature_model_parameters,
                inverter_parameters,
                irradiance=None, weather=None,
                transposition_model='haydavies',
                solar_position_method='nrel_numpy',
                airmass_model='kastenyoung1989',
                altitude=None, pressure=None,
                **kwargs):
    """
    An experimental function that computes all of the modeling steps
    necessary for calculating power or energy for a PV system at a given
    location.

    Parameters
    ----------
    times : DatetimeIndex
        Times at which to evaluate the model.

    latitude : float.
        Positive is north of the equator.
        Use decimal degrees notation.

    longitude : float.
        Positive is east of the prime meridian.
        Use decimal degrees notation.

    surface_tilt : numeric
        Surface tilt angles in decimal degrees.
        The tilt angle is defined as degrees from horizontal
        (e.g. surface facing up = 0, surface facing horizon = 90)

    surface_azimuth : numeric
        Surface azimuth angles in decimal degrees.
        The azimuth convention is defined
        as degrees east of north
        (North=0, South=180, East=90, West=270).

    module_parameters : None, dict or Series
        Module parameters as defined by the SAPM. See pvsystem.sapm for
        details.

    temperature_model_parameters : None, dict or Series.
        Temperature model parameters as defined by the SAPM.
        See temperature.sapm_cell for details.

    inverter_parameters : None, dict or Series
        Inverter parameters as defined by the CEC. See
        :py:func:`inverter.sandia` for details.

    irradiance : None or DataFrame, default None
        If None, calculates clear sky data.
        Columns must be 'dni', 'ghi', 'dhi'.

    weather : None or DataFrame, default None
        If None, assumes air temperature is 20 C and
        wind speed is 0 m/s.
        Columns must be 'wind_speed', 'temp_air'.

    transposition_model : str, default 'haydavies'
        Passed to system.get_irradiance.

    solar_position_method : str, default 'nrel_numpy'
        Passed to solarposition.get_solarposition.

    airmass_model : str, default 'kastenyoung1989'
        Passed to atmosphere.relativeairmass.

    altitude : None or float, default None
        If None, computed from pressure. Assumed to be 0 m
        if pressure is also None.

    pressure : None or float, default None
        If None, computed from altitude. Assumed to be 101325 Pa
        if altitude is also None.

    **kwargs
        Arbitrary keyword arguments.
        See code for details.

    Returns
    -------
    output : (dc, ac)
        Tuple of DC power (with SAPM parameters) (DataFrame) and AC
        power (Series).
    """

    if altitude is None and pressure is None:
        altitude = 0.
        pressure = 101325.
    elif altitude is None:
        altitude = atmosphere.pres2alt(pressure)
    elif pressure is None:
        pressure = atmosphere.alt2pres(altitude)

    solar_position = solarposition.get_solarposition(
        times, latitude, longitude, altitude=altitude, pressure=pressure,
        method=solar_position_method, **kwargs)

    # possible error with using apparent zenith with some models
    airmass = atmosphere.get_relative_airmass(
        solar_position['apparent_zenith'], model=airmass_model)
    airmass = atmosphere.get_absolute_airmass(airmass, pressure)
    dni_extra = pvlib.irradiance.get_extra_radiation(solar_position.index)

    aoi = pvlib.irradiance.aoi(surface_tilt, surface_azimuth,
                               solar_position['apparent_zenith'],
                               solar_position['azimuth'])

    if irradiance is None:
        linke_turbidity = clearsky.lookup_linke_turbidity(
            solar_position.index, latitude, longitude)
        irradiance = clearsky.ineichen(
            solar_position['apparent_zenith'],
            airmass,
            linke_turbidity,
            altitude=altitude,
            dni_extra=dni_extra
        )

    total_irrad = pvlib.irradiance.get_total_irradiance(
        surface_tilt,
        surface_azimuth,
        solar_position['apparent_zenith'],
        solar_position['azimuth'],
        irradiance['dni'],
        irradiance['ghi'],
        irradiance['dhi'],
        model=transposition_model,
        dni_extra=dni_extra)

    if weather is None:
        weather = {'wind_speed': 0, 'temp_air': 20}

    cell_temperature = temperature.sapm_cell(
        total_irrad['poa_global'], weather['temp_air'], weather['wind_speed'],
        temperature_model_parameters['a'], temperature_model_parameters['b'],
        temperature_model_parameters['deltaT'])

    effective_irradiance = pvsystem.sapm_effective_irradiance(
        total_irrad['poa_direct'], total_irrad['poa_diffuse'], airmass, aoi,
        module_parameters)

    dc = pvsystem.sapm(effective_irradiance, cell_temperature,
                       module_parameters)

    ac = inverter.sandia(dc['v_mp'], dc['p_mp'], inverter_parameters)

    return dc, ac


def get_orientation(strategy, **kwargs):
    """
    Determine a PV system's surface tilt and surface azimuth
    using a named strategy.

    Parameters
    ----------
    strategy: str
        The orientation strategy.
        Allowed strategies include 'flat', 'south_at_latitude_tilt'.
    **kwargs:
        Strategy-dependent keyword arguments. See code for details.

    Returns
    -------
    surface_tilt, surface_azimuth
    """

    if strategy == 'south_at_latitude_tilt':
        surface_azimuth = 180
        surface_tilt = kwargs['latitude']
    elif strategy == 'flat':
        surface_azimuth = 180
        surface_tilt = 0
    else:
        raise ValueError('invalid orientation strategy. strategy must '
                         'be one of south_at_latitude, flat,')

    return surface_tilt, surface_azimuth


# Type for fields that vary between arrays
T = TypeVar('T')


PerArray = Union[T, Tuple[T, ...]]


@dataclass
class ModelChainResult:
    # these attributes are used in __setattr__ to determine the correct type.
    _singleton_tuples: bool = field(default=False)
    _per_array_fields = {'total_irrad', 'aoi', 'aoi_modifier',
                         'spectral_modifier', 'cell_temperature',
                         'effective_irradiance', 'dc', 'diode_params',
                         'dc_ohmic_losses', 'weather', 'albedo'}

    # system-level information
    solar_position: Optional[pd.DataFrame] = field(default=None)
    """Solar position in a DataFrame containing columns ``'apparent_zenith'``,
    ``'zenith'``, ``'apparent_elevation'``, ``'elevation'``, ``'azimuth'``
    (all in degrees), with possibly other columns depending on the solar
    position method; see :py:func:`~pvlib.solarposition.get_solarposition`
    for details."""

    airmass: Optional[pd.DataFrame] = field(default=None)
    """Air mass in a DataFrame containing columns ``'airmass_relative'``,
    ``'airmass_absolute'`` (unitless); see
    :py:meth:`~pvlib.location.Location.get_airmass` for details."""

    ac: Optional[pd.Series] = field(default=None)
    """AC power from the PV system, in a Series [W]"""

    tracking: Optional[pd.DataFrame] = field(default=None)
    """Orientation of modules on a single axis tracker, in a DataFrame with
    columns ``'surface_tilt'``, ``'surface_azimuth'``, ``'aoi'``; see
    :py:func:`~pvlib.tracking.singleaxis` for details.
    """

    losses: Optional[Union[pd.Series, float]] = field(default=None)
    """Series containing DC loss as a fraction of total DC power, as
    calculated by ``ModelChain.losses_model``.
    """

    # per DC array information
    total_irrad: Optional[PerArray[pd.DataFrame]] = field(default=None)
    """ DataFrame (or tuple of DataFrame, one for each array) containing
    columns ``'poa_global'``, ``'poa_direct'`` ``'poa_diffuse'``,
    ``poa_sky_diffuse'``, ``'poa_ground_diffuse'`` (W/m2); see
    :py:func:`~pvlib.irradiance.get_total_irradiance` for details.
    """

    aoi: Optional[PerArray[pd.Series]] = field(default=None)
    """
    Series (or tuple of Series, one for each array) containing angle of
    incidence (degrees); see :py:func:`~pvlib.irradiance.aoi` for details.
    """

    aoi_modifier: Optional[PerArray[Union[pd.Series, float]]] = \
        field(default=None)
    """Series (or tuple of Series, one for each array) containing angle of
    incidence modifier (unitless) calculated by ``ModelChain.aoi_model``,
    which reduces direct irradiance for reflections;
    see :py:meth:`~pvlib.pvsystem.PVSystem.get_iam` for details.
    """

    spectral_modifier: Optional[PerArray[Union[pd.Series, float]]] = \
        field(default=None)
    """Series (or tuple of Series, one for each array) containing spectral
    modifier (unitless) calculated by ``ModelChain.spectral_model``, which
    adjusts broadband plane-of-array irradiance for spectral content.
    """

    cell_temperature: Optional[PerArray[pd.Series]] = field(default=None)
    """Series (or tuple of Series, one for each array) containing cell
    temperature (C).
    """

    effective_irradiance: Optional[PerArray[pd.Series]] = field(default=None)
    """Series (or tuple of Series, one for each array) containing effective
    irradiance (W/m2) which is total plane-of-array irradiance adjusted for
    reflections and spectral content.
    """

    dc: Optional[PerArray[Union[pd.Series, pd.DataFrame]]] = \
        field(default=None)
    """Series or DataFrame (or tuple of Series or DataFrame, one for
    each array) containing DC power (W) for each array, calculated by
    ``ModelChain.dc_model``.
    """

    diode_params: Optional[PerArray[pd.DataFrame]] = field(default=None)
    """DataFrame (or tuple of DataFrame, one for each array) containing diode
    equation parameters (columns ``'I_L'``, ``'I_o'``, ``'R_s'``, ``'R_sh'``,
    ``'nNsVth'``, present when ModelChain.dc_model is a single diode model;
    see :py:func:`~pvlib.pvsystem.singlediode` for details.
    """

    dc_ohmic_losses: Optional[PerArray[pd.Series]] = field(default=None)
    """Series (or tuple of Series, one for each array) containing DC ohmic
    loss (W) calculated by ``ModelChain.dc_ohmic_model``.
    """

    # copies of input data, for user convenience
    weather: Optional[PerArray[pd.DataFrame]] = None
    """DataFrame (or tuple of DataFrame, one for each array) contains a
    copy of the input weather data.
    """

    times: Optional[pd.DatetimeIndex] = None
    """DatetimeIndex containing a copy of the index of the input weather data.
    """

    albedo: Optional[PerArray[pd.Series]] = None
    """Series (or tuple of Series, one for each array) containing albedo.
    """

    def _result_type(self, value):
        """Coerce `value` to the correct type according to
        ``self._singleton_tuples``."""
        # Allow None to pass through without being wrapped in a tuple
        if (self._singleton_tuples
                and not isinstance(value, tuple)
                and value is not None):
            return (value,)
        return value

    def __setattr__(self, key, value):
        if key in ModelChainResult._per_array_fields:
            value = self._result_type(value)
        super().__setattr__(key, value)


class ModelChain:
    """
    The ModelChain class to provides a standardized, high-level
    interface for all of the modeling steps necessary for calculating PV
    power from a time series of weather inputs. The same models are applied
    to all ``pvsystem.Array`` objects, so each Array must contain the
    appropriate model parameters. For example, if ``dc_model='pvwatts'``,
    then each ``Array.module_parameters`` must contain ``'pdc0'``.

    See :ref:`modelchaindoc` for examples.

    Parameters
    ----------
    system : PVSystem
        A :py:class:`~pvlib.pvsystem.PVSystem` object that represents
        the connected set of modules, inverters, etc.

    location : Location
        A :py:class:`~pvlib.location.Location` object that represents
        the physical location at which to evaluate the model.

    clearsky_model : str, default 'ineichen'
        Passed to location.get_clearsky.

    transposition_model : str, default 'haydavies'
        Passed to system.get_irradiance.

    solar_position_method : str, default 'nrel_numpy'
        Passed to location.get_solarposition.

    airmass_model : str, default 'kastenyoung1989'
        Passed to location.get_airmass.

    dc_model: None, str, or function, default None
        If None, the model will be inferred from the parameters that
        are common to all of system.arrays[i].module_parameters.
        Valid strings are 'sapm', 'desoto', 'cec', 'pvsyst', 'pvwatts'.
        The ModelChain instance will be passed as the first argument
        to a user-defined function.

    ac_model: None, str, or function, default None
        If None, the model will be inferred from the parameters that
        are common to all of system.inverter_parameters.
        Valid strings are 'sandia', 'adr', 'pvwatts'. The
        ModelChain instance will be passed as the first argument to a
        user-defined function.

    aoi_model: None, str, or function, default None
        If None, the model will be inferred from the parameters that
        are common to all of system.arrays[i].module_parameters.
        Valid strings are 'physical', 'ashrae', 'sapm', 'martin_ruiz',
        'no_loss'. The ModelChain instance will be passed as the
        first argument to a user-defined function.

    spectral_model: None, str, or function, default None
        If None, the model will be inferred from the parameters that
        are common to all of system.arrays[i].module_parameters.
        Valid strings are 'sapm', 'first_solar', 'no_loss'.
        The ModelChain instance will be passed as the first argument to
        a user-defined function.

    temperature_model: None, str or function, default None
        Valid strings are: 'sapm', 'pvsyst', 'faiman', 'fuentes', 'noct_sam'.
        The ModelChain instance will be passed as the first argument to a
        user-defined function.

    dc_ohmic_model: str or function, default 'no_loss'
        Valid strings are 'dc_ohms_from_percent', 'no_loss'. The ModelChain
        instance will be passed as the first argument to a user-defined
        function.

    losses_model: str or function, default 'no_loss'
        Valid strings are 'pvwatts', 'no_loss'. The ModelChain instance
        will be passed as the first argument to a user-defined function.

    name: None or str, default None
        Name of ModelChain instance.
    """

    # list of deprecated attributes
    _deprecated_attrs = ['solar_position', 'airmass', 'total_irrad',
                         'aoi', 'aoi_modifier', 'spectral_modifier',
                         'cell_temperature', 'effective_irradiance',
                         'dc', 'ac', 'diode_params', 'tracking',
                         'weather', 'times', 'losses']

    def __init__(self, system, location,
                 clearsky_model='ineichen',
                 transposition_model='haydavies',
                 solar_position_method='nrel_numpy',
                 airmass_model='kastenyoung1989',
                 dc_model=None, ac_model=None, aoi_model=None,
                 spectral_model=None, temperature_model=None,
                 dc_ohmic_model='no_loss',
                 losses_model='no_loss', name=None):

        self.name = name
        self.system = system

        self.location = location
        self.clearsky_model = clearsky_model
        self.transposition_model = transposition_model
        self.solar_position_method = solar_position_method
        self.airmass_model = airmass_model

        # calls setters
        self.dc_model = dc_model
        self.ac_model = ac_model
        self.aoi_model = aoi_model
        self.spectral_model = spectral_model
        self.temperature_model = temperature_model

        self.dc_ohmic_model = dc_ohmic_model
        self.losses_model = losses_model

        self.results = ModelChainResult()

    def __getattr__(self, key):
        if key in ModelChain._deprecated_attrs:
            msg = f'ModelChain.{key} is deprecated and will' \
                  f' be removed in v0.10. Use' \
                  f' ModelChain.results.{key} instead'
            warnings.warn(msg, pvlibDeprecationWarning)
            return getattr(self.results, key)
        # __getattr__ is only called if __getattribute__ fails.
        # In that case we should check if key is a deprecated attribute,
        # and fail with an AttributeError if it is not.
        raise AttributeError

    def __setattr__(self, key, value):
        if key in ModelChain._deprecated_attrs:
            msg = f'ModelChain.{key} is deprecated from v0.9. Use' \
                  f' ModelChain.results.{key} instead'
            warnings.warn(msg, pvlibDeprecationWarning)
            setattr(self.results, key, value)
        else:
            super().__setattr__(key, value)

    @classmethod
    def with_pvwatts(cls, system, location,
                     clearsky_model='ineichen',
                     airmass_model='kastenyoung1989',
                     name=None,
                     **kwargs):
        """
        ModelChain that follows the PVWatts methods.

        Parameters
        ----------
        system : PVSystem
            A :py:class:`~pvlib.pvsystem.PVSystem` object that represents
            the connected set of modules, inverters, etc.

        location : Location
            A :py:class:`~pvlib.location.Location` object that represents
            the physical location at which to evaluate the model.

        clearsky_model : str, default 'ineichen'
            Passed to location.get_clearsky.

        airmass_model : str, default 'kastenyoung1989'
            Passed to location.get_airmass.

        name: None or str, default None
            Name of ModelChain instance.

        **kwargs
            Parameters supplied here are passed to the ModelChain
            constructor and take precedence over the default
            configuration.

        Examples
        --------
        >>> module_parameters = dict(gamma_pdc=-0.003, pdc0=4500)
        >>> inverter_parameters = dict(pac0=4000)
        >>> tparams = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']
        >>> system = PVSystem(surface_tilt=30, surface_azimuth=180,
        ...     module_parameters=module_parameters,
        ...     inverter_parameters=inverter_parameters,
        ...     temperature_model_parameters=tparams)
        >>> location = Location(32.2, -110.9)
        >>> ModelChain.with_pvwatts(system, location)
        ModelChain:
          name: None
          clearsky_model: ineichen
          transposition_model: perez
          solar_position_method: nrel_numpy
          airmass_model: kastenyoung1989
          dc_model: pvwatts_dc
          ac_model: pvwatts_inverter
          aoi_model: physical_aoi_loss
          spectral_model: no_spectral_loss
          temperature_model: sapm_temp
          losses_model: pvwatts_losses
        """  # noqa: E501
        config = PVWATTS_CONFIG.copy()
        config.update(kwargs)
        return ModelChain(
            system, location,
            clearsky_model=clearsky_model,
            airmass_model=airmass_model,
            name=name,
            **config
        )

    @classmethod
    def with_sapm(cls, system, location,
                  clearsky_model='ineichen',
                  transposition_model='haydavies',
                  solar_position_method='nrel_numpy',
                  airmass_model='kastenyoung1989',
                  name=None,
                  **kwargs):
        """
        ModelChain that follows the Sandia Array Performance Model
        (SAPM) methods.

        Parameters
        ----------
        system : PVSystem
            A :py:class:`~pvlib.pvsystem.PVSystem` object that represents
            the connected set of modules, inverters, etc.

        location : Location
            A :py:class:`~pvlib.location.Location` object that represents
            the physical location at which to evaluate the model.

        clearsky_model : str, default 'ineichen'
            Passed to location.get_clearsky.

        transposition_model : str, default 'haydavies'
            Passed to system.get_irradiance.

        solar_position_method : str, default 'nrel_numpy'
            Passed to location.get_solarposition.

        airmass_model : str, default 'kastenyoung1989'
            Passed to location.get_airmass.

        name: None or str, default None
            Name of ModelChain instance.

        **kwargs
            Parameters supplied here are passed to the ModelChain
            constructor and take precedence over the default
            configuration.

        Examples
        --------
        >>> mods = pvlib.pvsystem.retrieve_sam('sandiamod')
        >>> invs = pvlib.pvsystem.retrieve_sam('cecinverter')
        >>> module_parameters = mods['Canadian_Solar_CS5P_220M___2009_']
        >>> inverter_parameters = invs['ABB__MICRO_0_25_I_OUTD_US_240__240V_']
        >>> tparams = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']
        >>> system = PVSystem(surface_tilt=30, surface_azimuth=180,
        ...     module_parameters=module_parameters,
        ...     inverter_parameters=inverter_parameters,
        ...     temperature_model_parameters=tparams)
        >>> location = Location(32.2, -110.9)
        >>> ModelChain.with_sapm(system, location)
        ModelChain:
          name: None
          clearsky_model: ineichen
          transposition_model: haydavies
          solar_position_method: nrel_numpy
          airmass_model: kastenyoung1989
          dc_model: sapm
          ac_model: snlinverter
          aoi_model: sapm_aoi_loss
          spectral_model: sapm_spectral_loss
          temperature_model: sapm_temp
          losses_model: no_extra_losses
        """  # noqa: E501
        config = SAPM_CONFIG.copy()
        config.update(kwargs)
        return ModelChain(
            system, location,
            clearsky_model=clearsky_model,
            transposition_model=transposition_model,
            solar_position_method=solar_position_method,
            airmass_model=airmass_model,
            name=name,
            **config
        )

    def __repr__(self):
        attrs = [
            'name', 'clearsky_model',
            'transposition_model', 'solar_position_method',
            'airmass_model', 'dc_model', 'ac_model', 'aoi_model',
            'spectral_model', 'temperature_model', 'losses_model'
        ]

        def getmcattr(self, attr):
            """needed to avoid recursion in property lookups"""
            out = getattr(self, attr)
            try:
                out = out.__name__
            except AttributeError:
                pass
            return out

        return ('ModelChain: \n  ' + '\n  '.join(
            f'{attr}: {getmcattr(self, attr)}' for attr in attrs))

    @property
    def dc_model(self):
        return self._dc_model

    @dc_model.setter
    def dc_model(self, model):
        # guess at model if None
        if model is None:
            self._dc_model, model = self.infer_dc_model()

        # Set model and validate parameters
        if isinstance(model, str):
            model = model.lower()
            if model in _DC_MODEL_PARAMS.keys():
                # validate module parameters
                module_parameters = tuple(
                    array.module_parameters for array in self.system.arrays)
                missing_params = (
                    _DC_MODEL_PARAMS[model] - _common_keys(module_parameters))
                if missing_params:  # some parameters are not in module.keys()
                    raise ValueError(model + ' selected for the DC model but '
                                     'one or more Arrays are missing '
                                     'one or more required parameters '
                                     ' : ' + str(missing_params))
                if model == 'sapm':
                    self._dc_model = self.sapm
                elif model == 'desoto':
                    self._dc_model = self.desoto
                elif model == 'cec':
                    self._dc_model = self.cec
                elif model == 'pvsyst':
                    self._dc_model = self.pvsyst
                elif model == 'pvwatts':
                    self._dc_model = self.pvwatts_dc
            else:
                raise ValueError(model + ' is not a valid DC power model')
        else:
            self._dc_model = partial(model, self)

    def infer_dc_model(self):
        """Infer DC power model from Array module parameters."""
        params = _common_keys(
            tuple(array.module_parameters for array in self.system.arrays))
        if {'A0', 'A1', 'C7'} <= params:
            return self.sapm, 'sapm'
        elif {'a_ref', 'I_L_ref', 'I_o_ref', 'R_sh_ref', 'R_s',
              'Adjust'} <= params:
            return self.cec, 'cec'
        elif {'a_ref', 'I_L_ref', 'I_o_ref', 'R_sh_ref', 'R_s'} <= params:
            return self.desoto, 'desoto'
        elif {'gamma_ref', 'mu_gamma', 'I_L_ref', 'I_o_ref', 'R_sh_ref',
              'R_sh_0', 'R_sh_exp', 'R_s'} <= params:
            return self.pvsyst, 'pvsyst'
        elif {'pdc0', 'gamma_pdc'} <= params:
            return self.pvwatts_dc, 'pvwatts'
        else:
            raise ValueError(
                'Could not infer DC model from the module_parameters '
                'attributes of system.arrays. Check the module_parameters '
                'attributes or explicitly set the model with the dc_model '
                'keyword argument.')

    def sapm(self):
        dc = self.system.sapm(self.results.effective_irradiance,
                              self.results.cell_temperature)
        self.results.dc = self.system.scale_voltage_current_power(dc)
        return self

    def _singlediode(self, calcparams_model_function):
        def _make_diode_params(photocurrent, saturation_current,
                               resistance_series, resistance_shunt,
                               nNsVth):
            return pd.DataFrame(
                {'I_L': photocurrent, 'I_o': saturation_current,
                 'R_s': resistance_series, 'R_sh': resistance_shunt,
                 'nNsVth': nNsVth}
            )
        params = calcparams_model_function(self.results.effective_irradiance,
                                           self.results.cell_temperature,
                                           unwrap=False)
        self.results.diode_params = tuple(itertools.starmap(
            _make_diode_params, params))
        self.results.dc = tuple(itertools.starmap(
            self.system.singlediode, params))
        self.results.dc = self.system.scale_voltage_current_power(
            self.results.dc,
            unwrap=False
        )
        self.results.dc = tuple(dc.fillna(0) for dc in self.results.dc)
        # If the system has one Array, unwrap the single return value
        # to preserve the original behavior of ModelChain
        if self.system.num_arrays == 1:
            self.results.diode_params = self.results.diode_params[0]
            self.results.dc = self.results.dc[0]
        return self

    def desoto(self):
        return self._singlediode(self.system.calcparams_desoto)

    def cec(self):
        return self._singlediode(self.system.calcparams_cec)

    def pvsyst(self):
        return self._singlediode(self.system.calcparams_pvsyst)

    def pvwatts_dc(self):
        """Calculate DC power using the PVWatts model.

        Results are stored in ModelChain.results.dc. DC power is computed
        from PVSystem.arrays[i].module_parameters['pdc0'] and then scaled by
        PVSystem.modules_per_string and PVSystem.strings_per_inverter.

        Returns
        -------
        self

        See also
        --------
        pvlib.pvsystem.PVSystem.pvwatts_dc
        pvlib.pvsystem.PVSystem.scale_voltage_current_power
        """
        dc = self.system.pvwatts_dc(
            self.results.effective_irradiance,
            self.results.cell_temperature,
            unwrap=False
        )
        p_mp = tuple(pd.DataFrame(s, columns=['p_mp']) for s in dc)
        scaled = self.system.scale_voltage_current_power(p_mp)
        self.results.dc = _tuple_from_dfs(scaled, "p_mp")
        return self

    @property
    def ac_model(self):
        return self._ac_model

    @ac_model.setter
    def ac_model(self, model):
        if model is None:
            self._ac_model = self.infer_ac_model()
        elif isinstance(model, str):
            model = model.lower()
            if model == 'sandia':
                self._ac_model = self.sandia_inverter
            elif model in 'adr':
                self._ac_model = self.adr_inverter
            elif model == 'pvwatts':
                self._ac_model = self.pvwatts_inverter
            else:
                raise ValueError(model + ' is not a valid AC power model')
        else:
            self._ac_model = partial(model, self)

    def infer_ac_model(self):
        """Infer AC power model from system attributes."""
        inverter_params = set(self.system.inverter_parameters.keys())
        if _snl_params(inverter_params):
            return self.sandia_inverter
        if _adr_params(inverter_params):
            if self.system.num_arrays > 1:
                raise ValueError(
                    'The adr inverter function cannot be used for an inverter',
                    ' with multiple MPPT inputs')
            else:
                return self.adr_inverter
        if _pvwatts_params(inverter_params):
            return self.pvwatts_inverter
        raise ValueError('could not infer AC model from '
                         'system.inverter_parameters. Check '
                         'system.inverter_parameters or explicitly '
                         'set the model with the ac_model kwarg.')

    def sandia_inverter(self):
        self.results.ac = self.system.get_ac(
            'sandia',
            _tuple_from_dfs(self.results.dc, 'p_mp'),
            v_dc=_tuple_from_dfs(self.results.dc, 'v_mp')
        )
        return self

    def adr_inverter(self):
        self.results.ac = self.system.get_ac(
            'adr',
            self.results.dc['p_mp'],
            v_dc=self.results.dc['v_mp']
        )
        return self

    def pvwatts_inverter(self):
        ac = self.system.get_ac('pvwatts', self.results.dc)
        self.results.ac = ac.fillna(0)
        return self

    @property
    def aoi_model(self):
        return self._aoi_model

    @aoi_model.setter
    def aoi_model(self, model):
        if model is None:
            self._aoi_model = self.infer_aoi_model()
        elif isinstance(model, str):
            model = model.lower()
            if model == 'ashrae':
                self._aoi_model = self.ashrae_aoi_loss
            elif model == 'physical':
                self._aoi_model = self.physical_aoi_loss
            elif model == 'sapm':
                self._aoi_model = self.sapm_aoi_loss
            elif model == 'martin_ruiz':
                self._aoi_model = self.martin_ruiz_aoi_loss
            elif model == 'no_loss':
                self._aoi_model = self.no_aoi_loss
            else:
                raise ValueError(model + ' is not a valid aoi loss model')
        else:
            self._aoi_model = partial(model, self)

    def infer_aoi_model(self):
        module_parameters = tuple(
            array.module_parameters for array in self.system.arrays)
        params = _common_keys(module_parameters)
        if {'K', 'L', 'n'} <= params:
            return self.physical_aoi_loss
        elif {'B5', 'B4', 'B3', 'B2', 'B1', 'B0'} <= params:
            return self.sapm_aoi_loss
        elif {'b'} <= params:
            return self.ashrae_aoi_loss
        elif {'a_r'} <= params:
            return self.martin_ruiz_aoi_loss
        else:
            raise ValueError('could not infer AOI model from '
                             'system.arrays[i].module_parameters. Check that '
                             'the module_parameters for all Arrays in '
                             'system.arrays contain parameters for '
                             'the physical, aoi, ashrae or martin_ruiz model; '
                             'explicitly set the model with the aoi_model '
                             'kwarg; or set aoi_model="no_loss".')

    def ashrae_aoi_loss(self):
        self.results.aoi_modifier = self.system.get_iam(
            self.results.aoi,
            iam_model='ashrae'
        )
        return self

    def physical_aoi_loss(self):
        self.results.aoi_modifier = self.system.get_iam(
            self.results.aoi,
            iam_model='physical'
        )
        return self

    def sapm_aoi_loss(self):
        self.results.aoi_modifier = self.system.get_iam(
            self.results.aoi,
            iam_model='sapm'
        )
        return self

    def martin_ruiz_aoi_loss(self):
        self.results.aoi_modifier = self.system.get_iam(
            self.results.aoi, iam_model='martin_ruiz'
        )
        return self

    def no_aoi_loss(self):
        if self.system.num_arrays == 1:
            self.results.aoi_modifier = 1.0
        else:
            self.results.aoi_modifier = (1.0,) * self.system.num_arrays
        return self

    @property
    def spectral_model(self):
        return self._spectral_model

    @spectral_model.setter
    def spectral_model(self, model):
        if model is None:
            self._spectral_model = self.infer_spectral_model()
        elif isinstance(model, str):
            model = model.lower()
            if model == 'first_solar':
                self._spectral_model = self.first_solar_spectral_loss
            elif model == 'sapm':
                self._spectral_model = self.sapm_spectral_loss
            elif model == 'no_loss':
                self._spectral_model = self.no_spectral_loss
            else:
                raise ValueError(model + ' is not a valid spectral loss model')
        else:
            self._spectral_model = partial(model, self)

    def infer_spectral_model(self):
        """Infer spectral model from system attributes."""
        module_parameters = tuple(
            array.module_parameters for array in self.system.arrays)
        params = _common_keys(module_parameters)
        if {'A4', 'A3', 'A2', 'A1', 'A0'} <= params:
            return self.sapm_spectral_loss
        elif ((('Technology' in params or
                'Material' in params) and
               (self.system._infer_cell_type() is not None)) or
              'first_solar_spectral_coefficients' in params):
            return self.first_solar_spectral_loss
        else:
            raise ValueError('could not infer spectral model from '
                             'system.arrays[i].module_parameters. Check that '
                             'the module_parameters for all Arrays in '
                             'system.arrays contain valid '
                             'first_solar_spectral_coefficients, a valid '
                             'Material or Technology value, or set '
                             'spectral_model="no_loss".')

    def first_solar_spectral_loss(self):
        self.results.spectral_modifier = self.system.first_solar_spectral_loss(
            _tuple_from_dfs(self.results.weather, 'precipitable_water'),
            self.results.airmass['airmass_absolute']
        )
        return self

    def sapm_spectral_loss(self):
        self.results.spectral_modifier = self.system.sapm_spectral_loss(
            self.results.airmass['airmass_absolute']
        )
        return self

    def no_spectral_loss(self):
        if self.system.num_arrays == 1:
            self.results.spectral_modifier = 1
        else:
            self.results.spectral_modifier = (1,) * self.system.num_arrays
        return self

    @property
    def temperature_model(self):
        return self._temperature_model

    @temperature_model.setter
    def temperature_model(self, model):
        if model is None:
            self._temperature_model = self.infer_temperature_model()
        elif isinstance(model, str):
            model = model.lower()
            if model == 'sapm':
                self._temperature_model = self.sapm_temp
            elif model == 'pvsyst':
                self._temperature_model = self.pvsyst_temp
            elif model == 'faiman':
                self._temperature_model = self.faiman_temp
            elif model == 'fuentes':
                self._temperature_model = self.fuentes_temp
            elif model == 'noct_sam':
                self._temperature_model = self.noct_sam_temp
            else:
                raise ValueError(model + ' is not a valid temperature model')
            # check system.temperature_model_parameters for consistency
            name_from_params = self.infer_temperature_model().__name__
            if self._temperature_model.__name__ != name_from_params:
                common_params = _common_keys(tuple(
                    array.temperature_model_parameters
                    for array in self.system.arrays))
                raise ValueError(
                    f'Temperature model {self._temperature_model.__name__} is '
                    f'inconsistent with PVSystem temperature model '
                    f'parameters. All Arrays in system.arrays must have '
                    f'consistent parameters. Common temperature model '
                    f'parameters: {common_params}'
                )
        else:
            self._temperature_model = partial(model, self)

    def infer_temperature_model(self):
        """Infer temperature model from system attributes."""
        temperature_model_parameters = tuple(
            array.temperature_model_parameters for array in self.system.arrays)
        params = _common_keys(temperature_model_parameters)
        # remove or statement in v0.9
        if {'a', 'b', 'deltaT'} <= params or (
                not params and self.system.racking_model is None
                and self.system.module_type is None):
            return self.sapm_temp
        elif {'u_c', 'u_v'} <= params:
            return self.pvsyst_temp
        elif {'u0', 'u1'} <= params:
            return self.faiman_temp
        elif {'noct_installed'} <= params:
            return self.fuentes_temp
        elif {'noct', 'module_efficiency'} <= params:
            return self.noct_sam_temp
        else:
            raise ValueError(f'could not infer temperature model from '
                             f'system.temperature_model_parameters. Check '
                             f'that all Arrays in system.arrays have '
                             f'parameters for the same temperature model. '
                             f'Common temperature model parameters: {params}.')

    def _set_celltemp(self, model):
        """Set self.results.cell_temperature using the given cell
        temperature model.

        Parameters
        ----------
        model : str
            A cell temperature model name to pass to
            :py:meth:`pvlib.pvsystem.PVSystem.get_cell_temperature`.
            Valid names are 'sapm', 'pvsyst', 'faiman', 'fuentes', 'noct_sam'

        Returns
        -------
        self
        """

        poa = _irrad_for_celltemp(self.results.total_irrad,
                                  self.results.effective_irradiance)
        temp_air = _tuple_from_dfs(self.results.weather, 'temp_air')
        wind_speed = _tuple_from_dfs(self.results.weather, 'wind_speed')
        kwargs = {}
        if model == 'noct_sam':
            kwargs['effective_irradiance'] = self.results.effective_irradiance
        self.results.cell_temperature = self.system.get_cell_temperature(
            poa, temp_air, wind_speed, model=model, **kwargs)
        return self

    def sapm_temp(self):
        return self._set_celltemp('sapm')

    def pvsyst_temp(self):
        return self._set_celltemp('pvsyst')

    def faiman_temp(self):
        return self._set_celltemp('faiman')

    def fuentes_temp(self):
        return self._set_celltemp('fuentes')

    def noct_sam_temp(self):
        return self._set_celltemp('noct_sam')

    @property
    def dc_ohmic_model(self):
        return self._dc_ohmic_model

    @dc_ohmic_model.setter
    def dc_ohmic_model(self, model):
        if isinstance(model, str):
            model = model.lower()
            if model == 'dc_ohms_from_percent':
                self._dc_ohmic_model = self.dc_ohms_from_percent
            elif model == 'no_loss':
                self._dc_ohmic_model = self.no_dc_ohmic_loss
            else:
                raise ValueError(model + ' is not a valid losses model')
        else:
            self._dc_ohmic_model = partial(model, self)

    def dc_ohms_from_percent(self):
        """
        Calculate time series of ohmic losses and apply those to the mpp power
        output of the `dc_model` based on the pvsyst equivalent resistance
        method. Uses a `dc_ohmic_percent` parameter in the `losses_parameters`
        of the PVsystem.
        """
        Rw = self.system.dc_ohms_from_percent()
        if isinstance(self.results.dc, tuple):
            self.results.dc_ohmic_losses = tuple(
                pvsystem.dc_ohmic_losses(Rw, df['i_mp'])
                for Rw, df in zip(Rw, self.results.dc)
            )
            for df, loss in zip(self.results.dc, self.results.dc_ohmic_losses):
                df['p_mp'] = df['p_mp'] - loss
        else:
            self.results.dc_ohmic_losses = pvsystem.dc_ohmic_losses(
                Rw, self.results.dc['i_mp']
            )
            self.results.dc['p_mp'] = (self.results.dc['p_mp']
                                       - self.results.dc_ohmic_losses)
        return self

    def no_dc_ohmic_loss(self):
        return self

    @property
    def losses_model(self):
        return self._losses_model

    @losses_model.setter
    def losses_model(self, model):
        if model is None:
            self._losses_model = self.infer_losses_model()
        elif isinstance(model, str):
            model = model.lower()
            if model == 'pvwatts':
                self._losses_model = self.pvwatts_losses
            elif model == 'no_loss':
                self._losses_model = self.no_extra_losses
            else:
                raise ValueError(model + ' is not a valid losses model')
        else:
            self._losses_model = partial(model, self)

    def infer_losses_model(self):
        raise NotImplementedError

    def pvwatts_losses(self):
        self.results.losses = (100 - self.system.pvwatts_losses()) / 100.
        if isinstance(self.results.dc, tuple):
            for dc in self.results.dc:
                dc *= self.results.losses
        else:
            self.results.dc *= self.results.losses
        return self

    def no_extra_losses(self):
        self.results.losses = 1
        return self

    def effective_irradiance_model(self):
        def _eff_irrad(module_parameters, total_irrad, spect_mod, aoi_mod):
            fd = module_parameters.get('FD', 1.)
            return spect_mod * (total_irrad['poa_direct'] * aoi_mod +
                                fd * total_irrad['poa_diffuse'])
        if isinstance(self.results.total_irrad, tuple):
            self.results.effective_irradiance = tuple(
                _eff_irrad(array.module_parameters, ti, sm, am) for
                array, ti, sm, am in zip(
                    self.system.arrays, self.results.total_irrad,
                    self.results.spectral_modifier, self.results.aoi_modifier))
        else:
            self.results.effective_irradiance = _eff_irrad(
                self.system.arrays[0].module_parameters,
                self.results.total_irrad,
                self.results.spectral_modifier,
                self.results.aoi_modifier
            )
        return self

    def complete_irradiance(self, weather):
        """
        Determine the missing irradiation columns. Only two of the
        following data columns (dni, ghi, dhi) are needed to calculate
        the missing data.

        This function is not safe at the moment. Results can be too high
        or negative. Please contribute and help to improve this function
        on https://github.com/pvlib/pvlib-python

        Parameters
        ----------
        weather : DataFrame, or tuple or list of DataFrame
            Column names must be ``'dni'``, ``'ghi'``, ``'dhi'``,
            ``'wind_speed'``, ``'temp_air'``. All irradiance components
            are required. Air temperature of 20 C and wind speed
            of 0 m/s will be added to the DataFrame if not provided.
            If `weather` is a tuple it must be the same length as the number
            of Arrays in the system and the indices for each DataFrame must
            be the same.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            if the number of dataframes in `weather` is not the same as the
            number of Arrays in the system or if the indices of all elements
            of `weather` are not the same.

        Notes
        -----
        Assigns attributes to ``results``: ``times``, ``weather``

        Examples
        --------
        This example does not work until the parameters `my_system`,
        `my_location`, and `my_weather` are defined but shows the basic idea
        how this method can be used.

        >>> from pvlib.modelchain import ModelChain

        >>> # my_weather containing 'dhi' and 'ghi'.
        >>> mc = ModelChain(my_system, my_location)  # doctest: +SKIP
        >>> mc.complete_irradiance(my_weather)  # doctest: +SKIP
        >>> mc.run_model(mc.results.weather)  # doctest: +SKIP

        >>> # my_weather containing 'dhi', 'ghi' and 'dni'.
        >>> mc = ModelChain(my_system, my_location)  # doctest: +SKIP
        >>> mc.run_model(my_weather)  # doctest: +SKIP
        """
        weather = _to_tuple(weather)
        self._check_multiple_input(weather)
        # Don't use ModelChain._assign_weather() here because it adds
        # temperature and wind-speed columns which we do not need here.
        self.results.weather = _copy(weather)
        self._assign_times()
        self.results.solar_position = self.location.get_solarposition(
            self.results.times, method=self.solar_position_method)
        # Calculate the irradiance using the component sum equations,
        # if needed
        if isinstance(weather, tuple):
            for w in self.results.weather:
                self._complete_irradiance(w)
        else:
            self._complete_irradiance(self.results.weather)
        return self

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

    def _prep_inputs_solar_pos(self, weather):
        """
        Assign solar position
        """
        # build weather kwargs for solar position calculation
        kwargs = _build_kwargs(['pressure', 'temp_air'],
                               weather[0] if isinstance(weather, tuple)
                               else weather)
        try:
            kwargs['temperature'] = kwargs.pop('temp_air')
        except KeyError:
            pass

        self.results.solar_position = self.location.get_solarposition(
            self.results.times, method=self.solar_position_method,
            **kwargs)
        return self

    def _prep_inputs_albedo(self, weather):
        """
        Get albedo from weather
        """
        try:
            self.results.albedo = _tuple_from_dfs(weather, 'albedo')
        except KeyError:
            self.results.albedo = tuple([
                a.albedo for a in self.system.arrays])
        return self

    def _prep_inputs_airmass(self):
        """
        Assign airmass
        """
        self.results.airmass = self.location.get_airmass(
            solar_position=self.results.solar_position,
            model=self.airmass_model)
        return self

    def _prep_inputs_tracking(self):
        """
        Calculate tracker position and AOI
        """
        self.results.tracking = self.system.singleaxis(
            self.results.solar_position['apparent_zenith'],
            self.results.solar_position['azimuth'])
        self.results.tracking['surface_tilt'] = (
            self.results.tracking['surface_tilt']
                .fillna(self.system.axis_tilt))
        self.results.tracking['surface_azimuth'] = (
            self.results.tracking['surface_azimuth']
                .fillna(self.system.axis_azimuth))
        self.results.aoi = self.results.tracking['aoi']
        return self

    def _prep_inputs_fixed(self):
        """
        Calculate AOI for fixed tilt system
        """
        self.results.aoi = self.system.get_aoi(
            self.results.solar_position['apparent_zenith'],
            self.results.solar_position['azimuth'])
        return self

    def _verify_df(self, data, required):
        """ Checks data for column names in required

        Parameters
        ----------
        data : Dataframe
        required : List of str

        Raises
        ------
        ValueError if any of required are not in data.columns.
        """
        def _verify(data, index=None):
            if not set(required) <= set(data.columns):
                tuple_txt = "" if index is None else f"in element {index} "
                raise ValueError(
                    "Incomplete input data. Data needs to contain "
                    f"{required}. Detected data {tuple_txt}contains: "
                    f"{list(data.columns)}")
        if not isinstance(data, tuple):
            _verify(data)
        else:
            for (i, array_data) in enumerate(data):
                _verify(array_data, i)

    def _configure_results(self, per_array_data):
        """Configure the type used for per-array fields in
        ModelChainResult.

        If ``per_array_data`` is True and the number of arrays in the
        system is 1, then per-array results are stored as length-1
        tuples. This overrides the PVSystem defaults of unpacking a 1
        length tuple into a singleton.

        Parameters
        ----------
        per_array_data : bool
            If input data is provided for each array, pass True. If a
            single input data is provided for all arrays, pass False.
        """
        self.results._singleton_tuples = (
            self.system.num_arrays == 1 and per_array_data
        )

    def _assign_weather(self, data):
        def _build_weather(data):
            key_list = [k for k in WEATHER_KEYS if k in data]
            weather = data[key_list].copy()
            if weather.get('wind_speed') is None:
                weather['wind_speed'] = 0
            if weather.get('temp_air') is None:
                weather['temp_air'] = 20
            return weather
        if isinstance(data, tuple):
            weather = tuple(_build_weather(wx) for wx in data)
            self._configure_results(per_array_data=True)
        else:
            weather = _build_weather(data)
            self._configure_results(per_array_data=False)
        self.results.weather = weather
        self._assign_times()
        return self

    def _assign_total_irrad(self, data):
        def _build_irrad(data):
            key_list = [k for k in POA_KEYS if k in data]
            return data[key_list].copy()
        if isinstance(data, tuple):
            self.results.total_irrad = tuple(
                _build_irrad(irrad_data) for irrad_data in data
            )
            return self
        self.results.total_irrad = _build_irrad(data)
        return self

    def _assign_times(self):
        """Assign self.results.times according the the index of
        self.results.weather.

        If there are multiple DataFrames in self.results.weather then
        the index of the first one is assigned. It is assumed that the
        indices of each DataFrame in self.results.weather are the same.
        This can be verified by calling :py:func:`_all_same_index` or
        :py:meth:`self._check_multiple_weather` before calling this
        method.
        """
        if isinstance(self.results.weather, tuple):
            self.results.times = self.results.weather[0].index
        else:
            self.results.times = self.results.weather.index

    def prepare_inputs(self, weather):
        """
        Prepare the solar position, irradiance, and weather inputs to
        the model, starting with GHI, DNI and DHI.

        Parameters
        ----------
        weather : DataFrame, or tuple or list of DataFrames
            Required column names include ``'dni'``, ``'ghi'``, ``'dhi'``.
            Optional column names are ``'wind_speed'``, ``'temp_air'``,
            ``'albedo'``.

            If optional columns ``'wind_speed'``, ``'temp_air'`` are not
            provided, air temperature of 20 C and wind speed
            of 0 m/s will be added to the ``weather`` DataFrame.

            If optional column ``'albedo'`` is provided, albedo values in the
            ModelChain's PVSystem.arrays are ignored.

            If `weather` is a tuple or list, it must be of the same length and
            order as the Arrays of the ModelChain's PVSystem.

        Raises
        ------
        ValueError
            If any `weather` DataFrame(s) is missing an irradiance component.
        ValueError
            If `weather` is a tuple or list and the DataFrames it contains have
            different indices.
        ValueError
            If `weather` is a tuple or list with a different length than the
            number of Arrays in the system.

        Notes
        -----
        Assigns attributes to ``results``: ``times``, ``weather``,
        ``solar_position``, ``airmass``, ``total_irrad``, ``aoi``, ``albedo``.

        See also
        --------
        ModelChain.complete_irradiance
        """
        weather = _to_tuple(weather)
        self._check_multiple_input(weather, strict=False)
        self._verify_df(weather, required=['ghi', 'dni', 'dhi'])
        self._assign_weather(weather)

        self._prep_inputs_solar_pos(weather)
        self._prep_inputs_airmass()
        self._prep_inputs_albedo(weather)

        # PVSystem.get_irradiance and SingleAxisTracker.get_irradiance
        # and PVSystem.get_aoi and SingleAxisTracker.get_aoi
        # have different method signatures. Use partial to handle
        # the differences.
        if isinstance(self.system, SingleAxisTracker):
            self._prep_inputs_tracking()
            get_irradiance = partial(
                self.system.get_irradiance,
                self.results.tracking['surface_tilt'],
                self.results.tracking['surface_azimuth'],
                self.results.solar_position['apparent_zenith'],
                self.results.solar_position['azimuth'])
        else:
            self._prep_inputs_fixed()
            get_irradiance = partial(
                self.system.get_irradiance,
                self.results.solar_position['apparent_zenith'],
                self.results.solar_position['azimuth'])

        self.results.total_irrad = get_irradiance(
            _tuple_from_dfs(self.results.weather, 'dni'),
            _tuple_from_dfs(self.results.weather, 'ghi'),
            _tuple_from_dfs(self.results.weather, 'dhi'),
            albedo=self.results.albedo,
            airmass=self.results.airmass['airmass_relative'],
            model=self.transposition_model
        )

        return self

    def _check_multiple_input(self, data, strict=True):
        """Check that the number of elements in `data` is the same as
        the number of Arrays in `self.system`.

        In most cases if ``self.system.num_arrays`` is greater than 1 we
        want to raise an error when `data` is not a tuple; however, that
        behavior can be suppressed by setting ``strict=False``. This is
        useful for validating inputs such as GHI, DHI, DNI, wind speed, or
        air temperature that can be applied a ``PVSystem`` as a system-wide
        input. In this case we want to ensure that when a tuple is provided
        it has the same length as the number of Arrays, but we do not want
        to fail if the input is not a tuple.
        """
        if (not strict or self.system.num_arrays == 1) \
                and not isinstance(data, tuple):
            return
        if strict and not isinstance(data, tuple):
            raise TypeError("Input must be a tuple of length "
                            f"{self.system.num_arrays}, "
                            f"got {type(data).__name__}.")
        if len(data) != self.system.num_arrays:
            raise ValueError("Input must be same length as number of Arrays "
                             f"in system. Expected {self.system.num_arrays}, "
                             f"got {len(data)}.")
        _all_same_index(data)

    def prepare_inputs_from_poa(self, data):
        """
        Prepare the solar position, irradiance and weather inputs to
        the model, starting with plane-of-array irradiance.

        Parameters
        ----------
        data : DataFrame, or tuple or list of DataFrame
            Contains plane-of-array irradiance data. Required column names
            include ``'poa_global'``, ``'poa_direct'`` and ``'poa_diffuse'``.
            Columns with weather-related data are ssigned to the
            ``weather`` attribute.  If columns for ``'temp_air'`` and
            ``'wind_speed'`` are not provided, air temperature of 20 C and wind
            speed of 0 m/s are assumed.

            If list or tuple, must be of the same length and order as the
            Arrays of the ModelChain's PVSystem.

        Raises
        ------
        ValueError
             If the number of DataFrames passed in `data` is not the same
             as the number of Arrays in the system.

        Notes
        -----
        Assigns attributes to ``results``: ``times``, ``weather``,
        ``total_irrad``, ``solar_position``, ``airmass``, ``aoi``.

        See also
        --------
        pvlib.modelchain.ModelChain.prepare_inputs
        """
        data = _to_tuple(data)
        self._check_multiple_input(data)
        self._assign_weather(data)

        self._verify_df(data, required=['poa_global', 'poa_direct',
                                        'poa_diffuse'])
        self._assign_total_irrad(data)

        self._prep_inputs_solar_pos(data)
        self._prep_inputs_airmass()

        if isinstance(self.system, SingleAxisTracker):
            self._prep_inputs_tracking()
        else:
            self._prep_inputs_fixed()

        return self

    def _get_cell_temperature(self, data,
                              poa, temperature_model_parameters):
        """Extract the cell temperature data from a DataFrame.

        If 'cell_temperature' column exists in data then it is returned. If
        'module_temperature' column exists in data, then it is used with poa to
        calculate the cell temperature. If neither column exists then None is
        returned.

        Parameters
        ----------
        data : DataFrame (not a tuple of DataFrame)
        poa : Series (not a tuple of Series)

        Returns
        -------
        Series
        """
        if 'cell_temperature' in data:
            return data['cell_temperature']
        # cell_temperature is not in input. Calculate cell_temperature using
        # a temperature_model.
        # If module_temperature is in input data we can use the SAPM cell
        # temperature model.
        if (('module_temperature' in data) and
                (self.temperature_model == self.sapm_temp)):
            # use SAPM cell temperature model only
            return pvlib.temperature.sapm_cell_from_module(
                module_temperature=data['module_temperature'],
                poa_global=poa,
                deltaT=temperature_model_parameters['deltaT'])

    def _prepare_temperature_single_array(self, data, poa):
        """Set cell_temperature using a single data frame."""
        self.results.cell_temperature = self._get_cell_temperature(
            data,
            poa,
            self.system.arrays[0].temperature_model_parameters
        )
        if self.results.cell_temperature is None:
            self.temperature_model()
        return self

    def _prepare_temperature(self, data=None):
        """
        Sets cell_temperature using inputs in data and the specified
        temperature model.

        If 'data' contains 'cell_temperature', these values are assigned to
        attribute ``cell_temperature``. If 'data' contains 'module_temperature`
        and `temperature_model' is 'sapm', cell temperature is calculated using
        :py:func:`pvlib.temperature.sapm_cell_from_module`. Otherwise, cell
        temperature is calculated by 'temperature_model'.

        Parameters
        ----------
        data : DataFrame, default None
            May contain columns ``'cell_temperature'`` or
            ``'module_temperaure'``.

        Returns
        -------
        self

        Assigns attribute ``results.cell_temperature``.

        """
        poa = _irrad_for_celltemp(self.results.total_irrad,
                                  self.results.effective_irradiance)
        # handle simple case first, single array, data not iterable
        if not isinstance(data, tuple) and self.system.num_arrays == 1:
            return self._prepare_temperature_single_array(data, poa)
        if not isinstance(data, tuple):
            # broadcast data to all arrays
            data = (data,) * self.system.num_arrays
        # data is tuple, so temperature_model_parameters must also be
        # tuple. system.temperature_model_parameters is reduced to a dict
        # if system.num_arrays == 1, so manually access parameters. GH 1192
        t_mod_params = tuple(array.temperature_model_parameters
                             for array in self.system.arrays)
        # find where cell or module temperature is specified in input data
        given_cell_temperature = tuple(itertools.starmap(
            self._get_cell_temperature, zip(data, poa, t_mod_params)
        ))
        # If cell temperature has been specified for all arrays return
        # immediately and do not try to compute it.
        if all(cell_temp is not None for cell_temp in given_cell_temperature):
            self.results.cell_temperature = given_cell_temperature
            return self
        # Calculate cell temperature from weather data. If cell_temperature
        # has not been provided for some arrays then it is computed.
        self.temperature_model()
        # replace calculated cell temperature with temperature given in `data`
        # where available.
        self.results.cell_temperature = tuple(
            itertools.starmap(
                lambda given, modeled: modeled if given is None else given,
                zip(given_cell_temperature, self.results.cell_temperature)
            )
        )
        return self

    def run_model(self, weather):
        """
        Run the model chain starting with broadband global, diffuse and/or
        direct irradiance.

        Parameters
        ----------
        weather : DataFrame, or tuple or list of DataFrame
            Column names must include:

            - ``'dni'``
            - ``'ghi'``
            - ``'dhi'``

            Optional columns are:

            - ``'temp_air'``
            - ``'cell_temperature'``
            - ``'module_temperature'``
            - ``'wind_speed'``
            - ``'albedo'``

            If optional columns ``'temp_air'`` and ``'wind_speed'``
            are not provided, air temperature of 20 C and wind speed of 0 m/s
            are added to the DataFrame. If optional column
            ``'cell_temperature'`` is provided, these values are used instead
            of `temperature_model`. If optional column ``'module_temperature'``
            is provided, ``temperature_model`` must be ``'sapm'``.

            If optional column ``'albedo'`` is provided, ``'albedo'`` may not
            be present on the ModelChain's PVSystem.Arrays.

            If weather is a list or tuple, it must be of the same length and
            order as the Arrays of the ModelChain's PVSystem.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If the number of DataFrames in `data` is different than the number
            of Arrays in the PVSystem.
        ValueError
            If the DataFrames in `data` have different indexes.

        Notes
        -----
        Assigns attributes to ``results``: ``times``, ``weather``,
        ``solar_position``, ``airmass``, ``total_irrad``, ``aoi``,
        ``aoi_modifier``, ``spectral_modifier``, and
        ``effective_irradiance``, ``cell_temperature``, ``dc``, ``ac``,
        ``losses``, ``diode_params`` (if dc_model is a single diode
        model).

        See also
        --------
        pvlib.modelchain.ModelChain.run_model_from_poa
        pvlib.modelchain.ModelChain.run_model_from_effective_irradiance
        """
        weather = _to_tuple(weather)
        self.prepare_inputs(weather)
        self.aoi_model()
        self.spectral_model()
        self.effective_irradiance_model()

        self._run_from_effective_irrad(weather)

        return self

    def run_model_from_poa(self, data):
        """
        Run the model starting with broadband irradiance in the plane of array.

        Data must include direct, diffuse and total irradiance (W/m2) in the
        plane of array. Reflections and spectral adjustments are made to
        calculate effective irradiance (W/m2).

        Parameters
        ----------
        data : DataFrame, or tuple or list of DataFrame
            Required column names include ``'poa_global'``,
            ``'poa_direct'`` and ``'poa_diffuse'``. If optional columns
            ``'temp_air'`` and ``'wind_speed'`` are not provided, air
            temperature of 20 C and wind speed of 0 m/s are assumed.
            If optional column ``'cell_temperature'`` is provided, these values
            are used instead of `temperature_model`. If optional column
            ``'module_temperature'`` is provided, `temperature_model` must be
            ``'sapm'``.

            If the ModelChain's PVSystem has multiple arrays, `data` must be a
            list or tuple with the same length and order as the PVsystem's
            Arrays. Each element of `data` provides the irradiance and weather
            for the corresponding array.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If the number of DataFrames in `data` is different than the number
            of Arrays in the PVSystem.
        ValueError
            If the DataFrames in `data` have different indexes.

        Notes
        -----
        Assigns attributes to results: ``times``, ``weather``,
        ``solar_position``, ``airmass``, ``total_irrad``, ``aoi``,
        ``aoi_modifier``, ``spectral_modifier``, and
        ``effective_irradiance``, ``cell_temperature``, ``dc``, ``ac``,
        ``losses``, ``diode_params`` (if dc_model is a single diode
        model).

        See also
        --------
        pvlib.modelchain.ModelChain.run_model
        pvlib.modelchain.ModelChain.run_model_from_effective_irradiance
        """
        data = _to_tuple(data)
        self.prepare_inputs_from_poa(data)

        self.aoi_model()
        self.spectral_model()
        self.effective_irradiance_model()

        self._run_from_effective_irrad(data)

        return self

    def _run_from_effective_irrad(self, data=None):
        """
        Executes the temperature, DC, losses and AC models.

        Parameters
        ----------
        data : DataFrame, or tuple of DataFrame, default None
            If optional column ``'cell_temperature'`` is provided, these values
            are used instead of `temperature_model`. If optional column
            `module_temperature` is provided, `temperature_model` must be
            ``'sapm'``.

        Returns
        -------
        self

        Notes
        -----
        Assigns attributes:``cell_temperature``, ``dc``, ``ac``, ``losses``,
        ``diode_params`` (if dc_model is a single diode model).
        """
        self._prepare_temperature(data)
        self.dc_model()
        self.dc_ohmic_model()
        self.losses_model()
        self.ac_model()

        return self

    def run_model_from_effective_irradiance(self, data=None):
        """
        Run the model starting with effective irradiance in the plane of array.

        Effective irradiance is irradiance in the plane-of-array after any
        adjustments for soiling, reflections and spectrum.

        Parameters
        ----------
        data : DataFrame, or list or tuple of DataFrame
            Required column is ``'effective_irradiance'``.
            Optional columns include ``'cell_temperature'``,
            ``'module_temperature'`` and ``'poa_global'``.

            If the ModelChain's PVSystem has multiple arrays, `data` must be a
            list or tuple with the same length and order as the PVsystem's
            Arrays. Each element of `data` provides the irradiance and weather
            for the corresponding array.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If the number of DataFrames in `data` is different than the number
            of Arrays in the PVSystem.
        ValueError
            If the DataFrames in `data` have different indexes.

        Notes
        -----
        Optional ``data`` columns ``'cell_temperature'``,
        ``'module_temperature'`` and ``'poa_global'`` are used for determining
        cell temperature.

        * If optional column ``'cell_temperature'`` is present, these values
          are used and `temperature_model` is ignored.
        * If optional column ``'module_temperature'`` is preset,
          `temperature_model` must be ``'sapm'``.
        * Otherwise, cell temperature is calculated using `temperature_model`.

        The cell temperature models require plane-of-array irradiance as input.
        If optional column ``'poa_global'`` is present, these data are used.
        If ``'poa_global'`` is not present, ``'effective_irradiance'`` is used.

        Assigns attributes to results: ``times``, ``weather``, ``total_irrad``,
        ``effective_irradiance``, ``cell_temperature``, ``dc``, ``ac``,
        ``losses``, ``diode_params`` (if dc_model is a single diode model).

        See also
        --------
        pvlib.modelchain.ModelChain.run_model
        pvlib.modelchain.ModelChain.run_model_from_poa
        """
        data = _to_tuple(data)
        self._check_multiple_input(data)
        self._verify_df(data, required=['effective_irradiance'])
        self._assign_weather(data)
        self._assign_total_irrad(data)
        self.results.effective_irradiance = _tuple_from_dfs(
            data, 'effective_irradiance')
        self._run_from_effective_irrad(data)

        return self


def _irrad_for_celltemp(total_irrad, effective_irradiance):
    """
    Determine irradiance to use for cell temperature models, in order
    of preference 'poa_global' then 'effective_irradiance'

    Returns
    -------
    Series or tuple of Series
        tuple if total_irrad is a tuple of DataFrame

    """
    if isinstance(total_irrad, tuple):
        if all(['poa_global' in df for df in total_irrad]):
            return _tuple_from_dfs(total_irrad, 'poa_global')
        else:
            return effective_irradiance
    else:
        if 'poa_global' in total_irrad:
            return total_irrad['poa_global']
        else:
            return effective_irradiance


def _snl_params(inverter_params):
    """Return True if `inverter_params` includes parameters for the
    Sandia inverter model."""
    return {'C0', 'C1', 'C2'} <= inverter_params


def _adr_params(inverter_params):
    """Return True if `inverter_params` includes parameters for the ADR
    inverter model."""
    return {'ADRCoefficients'} <= inverter_params


def _pvwatts_params(inverter_params):
    """Return True if `inverter_params` includes parameters for the
    PVWatts inverter model."""
    return {'pdc0'} <= inverter_params


def _copy(data):
    """Return a copy of each DataFrame in `data` if it is a tuple,
    otherwise return a copy of `data`."""
    if not isinstance(data, tuple):
        return data.copy()
    return tuple(df.copy() for df in data)


def _all_same_index(data):
    """Raise a ValueError if all DataFrames in `data` do not have the
    same index."""
    indexes = map(lambda df: df.index, data)
    next(indexes, None)
    for index in indexes:
        if not index.equals(data[0].index):
            raise ValueError("Input DataFrames must have same index.")


def _common_keys(dicts):
    """Return the intersection of the set of keys for each dictionary
    in `dicts`"""
    def _keys(x):
        return set(x.keys())
    if isinstance(dicts, tuple):
        return set.intersection(*map(_keys, dicts))
    return _keys(dicts)


def _tuple_from_dfs(dfs, name):
    """Extract a column from each DataFrame in `dfs` if `dfs` is a tuple.

    Returns a tuple of Series if `dfs` is a tuple or a Series if `dfs` is
    a DataFrame.
    """
    if isinstance(dfs, tuple):
        return tuple(df[name] for df in dfs)
    else:
        return dfs[name]


def _to_tuple(x):
    if not isinstance(x, (tuple, list)):
        return x
    return tuple(x)

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


def _unwrap_single_value(func):
    """Decorator for functions that return iterables.

    If the length of the iterable returned by `func` is 1, then
    the single member of the iterable is returned. If the length is
    greater than 1, then entire iterable is returned.

    Adds 'unwrap' as a keyword argument that can be set to False
    to force the return value to be a tuple, regardless of its length.
    """
    @functools.wraps(func)
    def f(*args, **kwargs):
        unwrap = kwargs.pop('unwrap', True)
        x = func(*args, **kwargs)
        if unwrap and len(x) == 1:
            return x[0]
        return x
    return f


def _check_deprecated_passthrough(func):
    """
    Decorator to warn or error when getting and setting the "pass-through"
    PVSystem properties that have been moved to Array.  Emits a warning for
    PVSystems with only one Array and raises an error for PVSystems with
    more than one Array.
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        pvsystem_attr = func.__name__
        class_name = self.__class__.__name__  # PVSystem or SingleAxisTracker
        overrides = {  # some Array attrs aren't the same as PVSystem
            'strings_per_inverter': 'strings',
        }
        array_attr = overrides.get(pvsystem_attr, pvsystem_attr)
        alternative = f'{class_name}.arrays[i].{array_attr}'

        if len(self.arrays) > 1:
            raise AttributeError(
                f'{class_name}.{pvsystem_attr} not supported for multi-array '
                f'systems. Set {array_attr} for each Array in '
                f'{class_name}.arrays instead.')

        wrapped = deprecated('0.9', alternative=alternative, removal='0.10',
                             name=f"{class_name}.{pvsystem_attr}")(func)
        return wrapped(self, *args, **kwargs)

    return wrapper


# not sure if this belongs in the pvsystem module.
# maybe something more like core.py? It may eventually grow to
# import a lot more functionality from other modules.
class PVSystem:
    """
    The PVSystem class defines a standard set of PV system attributes
    and modeling functions. This class describes the collection and
    interactions of PV system components rather than an installed system
    on the ground. It is typically used in combination with
    :py:class:`~pvlib.location.Location` and
    :py:class:`~pvlib.modelchain.ModelChain`
    objects.

    The class supports basic system topologies consisting of:

        * `N` total modules arranged in series
          (`modules_per_string=N`, `strings_per_inverter=1`).
        * `M` total modules arranged in parallel
          (`modules_per_string=1`, `strings_per_inverter=M`).
        * `NxM` total modules arranged in `M` strings of `N` modules each
          (`modules_per_string=N`, `strings_per_inverter=M`).

    The class is complementary to the module-level functions.

    The attributes should generally be things that don't change about
    the system, such the type of module and the inverter. The instance
    methods accept arguments for things that do change, such as
    irradiance and temperature.

    Parameters
    ----------
    arrays : iterable of Array, optional
        List of arrays that are part of the system. If not specified
        a single array is created from the other parameters (e.g.
        `surface_tilt`, `surface_azimuth`). Must contain at least one Array,
        if length of arrays is 0 a ValueError is raised. If `arrays` is
        specified the following PVSystem parameters are ignored:

        - `surface_tilt`
        - `surface_azimuth`
        - `albedo`
        - `surface_type`
        - `module`
        - `module_type`
        - `module_parameters`
        - `temperature_model_parameters`
        - `modules_per_string`
        - `strings_per_inverter`

    surface_tilt: float or array-like, default 0
        Surface tilt angles in decimal degrees.
        The tilt angle is defined as degrees from horizontal
        (e.g. surface facing up = 0, surface facing horizon = 90)

    surface_azimuth: float or array-like, default 180
        Azimuth angle of the module surface.
        North=0, East=90, South=180, West=270.

    albedo : None or float, default None
        Ground surface albedo. If ``None``, then ``surface_type`` is used
        to look up a value in ``irradiance.SURFACE_ALBEDOS``.
        If ``surface_type`` is also None then a ground surface albedo
        of 0.25 is used.

    surface_type : None or string, default None
        The ground surface type. See ``irradiance.SURFACE_ALBEDOS`` for
        valid values.

    module : None or string, default None
        The model name of the modules.
        May be used to look up the module_parameters dictionary
        via some other method.

    module_type : None or string, default 'glass_polymer'
         Describes the module's construction. Valid strings are 'glass_polymer'
         and 'glass_glass'. Used for cell and module temperature calculations.

    module_parameters : None, dict or Series, default None
        Module parameters as defined by the SAPM, CEC, or other.

    temperature_model_parameters : None, dict or Series, default None.
        Temperature model parameters as required by one of the models in
        pvlib.temperature (excluding poa_global, temp_air and wind_speed).

    modules_per_string: int or float, default 1
        See system topology discussion above.

    strings_per_inverter: int or float, default 1
        See system topology discussion above.

    inverter : None or string, default None
        The model name of the inverters.
        May be used to look up the inverter_parameters dictionary
        via some other method.

    inverter_parameters : None, dict or Series, default None
        Inverter parameters as defined by the SAPM, CEC, or other.

    racking_model : None or string, default 'open_rack'
        Valid strings are 'open_rack', 'close_mount', and 'insulated_back'.
        Used to identify a parameter set for the SAPM cell temperature model.

    losses_parameters : None, dict or Series, default None
        Losses parameters as defined by PVWatts or other.

    name : None or string, default None

    **kwargs
        Arbitrary keyword arguments.
        Included for compatibility, but not used.

    Raises
    ------
    ValueError
        If `arrays` is not None and has length 0.

    See also
    --------
    pvlib.location.Location
    pvlib.tracking.SingleAxisTracker
    """

    def __init__(self,
                 arrays=None,
                 surface_tilt=0, surface_azimuth=180,
                 albedo=None, surface_type=None,
                 module=None, module_type=None,
                 module_parameters=None,
                 temperature_model_parameters=None,
                 modules_per_string=1, strings_per_inverter=1,
                 inverter=None, inverter_parameters=None,
                 racking_model=None, losses_parameters=None, name=None):

        if arrays is None:
            if losses_parameters is None:
                array_losses_parameters = {}
            else:
                array_losses_parameters = _build_kwargs(['dc_ohmic_percent'],
                                                        losses_parameters)
            self.arrays = (Array(
                FixedMount(surface_tilt, surface_azimuth, racking_model),
                albedo,
                surface_type,
                module,
                module_type,
                module_parameters,
                temperature_model_parameters,
                modules_per_string,
                strings_per_inverter,
                array_losses_parameters,
            ),)
        elif len(arrays) == 0:
            raise ValueError("PVSystem must have at least one Array. "
                             "If you want to create a PVSystem instance "
                             "with a single Array pass `arrays=None` and pass "
                             "values directly to PVSystem attributes, e.g., "
                             "`surface_tilt=30`")
        else:
            self.arrays = tuple(arrays)

        self.inverter = inverter
        if inverter_parameters is None:
            self.inverter_parameters = {}
        else:
            self.inverter_parameters = inverter_parameters

        if losses_parameters is None:
            self.losses_parameters = {}
        else:
            self.losses_parameters = losses_parameters

        self.name = name

    def __repr__(self):
        repr = f'PVSystem:\n  name: {self.name}\n  '
        for array in self.arrays:
            repr += '\n  '.join(array.__repr__().split('\n'))
            repr += '\n  '
        repr += f'inverter: {self.inverter}'
        return repr

    def _validate_per_array(self, values, system_wide=False):
        """Check that `values` is a tuple of the same length as
        `self.arrays`.

        If `values` is not a tuple it is packed in to a length-1 tuple before
        the check. If the lengths are not the same a ValueError is raised,
        otherwise the tuple `values` is returned.

        When `system_wide` is True and `values` is not a tuple, `values`
        is replicated to a tuple of the same length as `self.arrays` and that
        tuple is returned.
        """
        if system_wide and not isinstance(values, tuple):
            return (values,) * self.num_arrays
        if not isinstance(values, tuple):
            values = (values,)
        if len(values) != len(self.arrays):
            raise ValueError("Length mismatch for per-array parameter")
        return values

    @_unwrap_single_value
    def _infer_cell_type(self):
        """
        Examines module_parameters and maps the Technology key for the CEC
        database and the Material key for the Sandia database to a common
        list of strings for cell type.

        Returns
        -------
        cell_type: str
        """
        return tuple(array._infer_cell_type() for array in self.arrays)

    @_unwrap_single_value
    def get_aoi(self, solar_zenith, solar_azimuth):
        """Get the angle of incidence on the Array(s) in the system.

        Parameters
        ----------
        solar_zenith : float or Series.
            Solar zenith angle.
        solar_azimuth : float or Series.
            Solar azimuth angle.

        Returns
        -------
        aoi : Series or tuple of Series
            The angle of incidence
        """

        return tuple(array.get_aoi(solar_zenith, solar_azimuth)
                     for array in self.arrays)

    @_unwrap_single_value
    def get_irradiance(self, solar_zenith, solar_azimuth, dni, ghi, dhi,
                       dni_extra=None, airmass=None, albedo=None,
                       model='haydavies', **kwargs):
        """
        Uses the :py:func:`irradiance.get_total_irradiance` function to
        calculate the plane of array irradiance components on the tilted
        surfaces defined by each array's ``surface_tilt`` and
        ``surface_azimuth``.

        Parameters
        ----------
        solar_zenith : float or Series
            Solar zenith angle.
        solar_azimuth : float or Series
            Solar azimuth angle.
        dni : float or Series or tuple of float or Series
            Direct Normal Irradiance. [W/m2]
        ghi : float or Series or tuple of float or Series
            Global horizontal irradiance. [W/m2]
        dhi : float or Series or tuple of float or Series
            Diffuse horizontal irradiance. [W/m2]
        dni_extra : None, float, Series or tuple of float or Series,\
            default None
            Extraterrestrial direct normal irradiance. [W/m2]
        airmass : None, float or Series, default None
            Airmass. [unitless]
        albedo : None, float or Series, default None
            Ground surface albedo. [unitless]
        model : String, default 'haydavies'
            Irradiance model.

        kwargs
            Extra parameters passed to :func:`irradiance.get_total_irradiance`.

        Notes
        -----
        Each of `dni`, `ghi`, and `dni` parameters may be passed as a tuple
        to provide different irradiance for each array in the system. If not
        passed as a tuple then the same value is used for input to each Array.
        If passed as a tuple the length must be the same as the number of
        Arrays.

        Returns
        -------
        poa_irradiance : DataFrame or tuple of DataFrame
            Column names are: ``'poa_global', 'poa_direct', 'poa_diffuse',
            'poa_sky_diffuse', 'poa_ground_diffuse'``.

        See also
        --------
        pvlib.irradiance.get_total_irradiance
        """
        dni = self._validate_per_array(dni, system_wide=True)
        ghi = self._validate_per_array(ghi, system_wide=True)
        dhi = self._validate_per_array(dhi, system_wide=True)

        albedo = self._validate_per_array(albedo, system_wide=True)

        return tuple(
            array.get_irradiance(solar_zenith, solar_azimuth,
                                 dni, ghi, dhi,
                                 dni_extra=dni_extra, airmass=airmass,
                                 albedo=albedo, model=model, **kwargs)
            for array, dni, ghi, dhi, albedo in zip(
                self.arrays, dni, ghi, dhi, albedo
            )
        )

    @_unwrap_single_value
    def get_iam(self, aoi, iam_model='physical'):
        """
        Determine the incidence angle modifier using the method specified by
        ``iam_model``.

        Parameters for the selected IAM model are expected to be in
        ``PVSystem.module_parameters``. Default parameters are available for
        the 'physical', 'ashrae' and 'martin_ruiz' models.

        Parameters
        ----------
        aoi : numeric or tuple of numeric
            The angle of incidence in degrees.

        aoi_model : string, default 'physical'
            The IAM model to be used. Valid strings are 'physical', 'ashrae',
            'martin_ruiz' and 'sapm'.
        Returns
        -------
        iam : numeric or tuple of numeric
            The AOI modifier.

        Raises
        ------
        ValueError
            if `iam_model` is not a valid model name.
        """
        aoi = self._validate_per_array(aoi)
        return tuple(array.get_iam(aoi, iam_model)
                     for array, aoi in zip(self.arrays, aoi))

    @_unwrap_single_value
    def get_cell_temperature(self, poa_global, temp_air, wind_speed, model,
                             effective_irradiance=None):
        """
        Determine cell temperature using the method specified by ``model``.

        Parameters
        ----------
        poa_global : numeric or tuple of numeric
            Total incident irradiance in W/m^2.

        temp_air : numeric or tuple of numeric
            Ambient dry bulb temperature in degrees C.

        wind_speed : numeric or tuple of numeric
            Wind speed in m/s.

        model : str
            Supported models include ``'sapm'``, ``'pvsyst'``,
            ``'faiman'``, ``'fuentes'``, and ``'noct_sam'``

        effective_irradiance : numeric or tuple of numeric, optional
            The irradiance that is converted to photocurrent in W/m^2.
            Only used for some models.

        Returns
        -------
        numeric or tuple of numeric
            Values in degrees C.

        See Also
        --------
        Array.get_cell_temperature

        Notes
        -----
        The `temp_air` and `wind_speed` parameters may be passed as tuples
        to provide different values for each Array in the system. If passed as
        a tuple the length must be the same as the number of Arrays. If not
        passed as a tuple then the same value is used for each Array.
        """
        poa_global = self._validate_per_array(poa_global)
        temp_air = self._validate_per_array(temp_air, system_wide=True)
        wind_speed = self._validate_per_array(wind_speed, system_wide=True)
        # Not used for all models, but Array.get_cell_temperature handles it
        effective_irradiance = self._validate_per_array(effective_irradiance,
                                                        system_wide=True)

        return tuple(
            array.get_cell_temperature(poa_global, temp_air, wind_speed,
                                       model, effective_irradiance)
            for array, poa_global, temp_air, wind_speed, effective_irradiance
            in zip(
                self.arrays, poa_global, temp_air, wind_speed,
                effective_irradiance
            )
        )

    @_unwrap_single_value
    def calcparams_desoto(self, effective_irradiance, temp_cell):
        """
        Use the :py:func:`calcparams_desoto` function, the input
        parameters and ``self.module_parameters`` to calculate the
        module currents and resistances.

        Parameters
        ----------
        effective_irradiance : numeric or tuple of numeric
            The irradiance (W/m2) that is converted to photocurrent.

        temp_cell : float or Series or tuple of float or Series
            The average cell temperature of cells within a module in C.

        Returns
        -------
        See pvsystem.calcparams_desoto for details
        """
        effective_irradiance = self._validate_per_array(effective_irradiance)
        temp_cell = self._validate_per_array(temp_cell)

        build_kwargs = functools.partial(
            _build_kwargs,
            ['a_ref', 'I_L_ref', 'I_o_ref', 'R_sh_ref',
             'R_s', 'alpha_sc', 'EgRef', 'dEgdT',
             'irrad_ref', 'temp_ref']
        )

        return tuple(
            calcparams_desoto(
                effective_irradiance, temp_cell,
                **build_kwargs(array.module_parameters)
            )
            for array, effective_irradiance, temp_cell
            in zip(self.arrays, effective_irradiance, temp_cell)
        )

    @_unwrap_single_value
    def calcparams_cec(self, effective_irradiance, temp_cell):
        """
        Use the :py:func:`calcparams_cec` function, the input
        parameters and ``self.module_parameters`` to calculate the
        module currents and resistances.

        Parameters
        ----------
        effective_irradiance : numeric or tuple of numeric
            The irradiance (W/m2) that is converted to photocurrent.

        temp_cell : float or Series or tuple of float or Series
            The average cell temperature of cells within a module in C.

        Returns
        -------
        See pvsystem.calcparams_cec for details
        """
        effective_irradiance = self._validate_per_array(effective_irradiance)
        temp_cell = self._validate_per_array(temp_cell)

        build_kwargs = functools.partial(
            _build_kwargs,
            ['a_ref', 'I_L_ref', 'I_o_ref', 'R_sh_ref',
             'R_s', 'alpha_sc', 'Adjust', 'EgRef', 'dEgdT',
             'irrad_ref', 'temp_ref']
        )

        return tuple(
            calcparams_cec(
                effective_irradiance, temp_cell,
                **build_kwargs(array.module_parameters)
            )
            for array, effective_irradiance, temp_cell
            in zip(self.arrays, effective_irradiance, temp_cell)
        )

    @_unwrap_single_value
    def calcparams_pvsyst(self, effective_irradiance, temp_cell):
        """
        Use the :py:func:`calcparams_pvsyst` function, the input
        parameters and ``self.module_parameters`` to calculate the
        module currents and resistances.

        Parameters
        ----------
        effective_irradiance : numeric or tuple of numeric
            The irradiance (W/m2) that is converted to photocurrent.

        temp_cell : float or Series or tuple of float or Series
            The average cell temperature of cells within a module in C.

        Returns
        -------
        See pvsystem.calcparams_pvsyst for details
        """
        effective_irradiance = self._validate_per_array(effective_irradiance)
        temp_cell = self._validate_per_array(temp_cell)

        build_kwargs = functools.partial(
            _build_kwargs,
            ['gamma_ref', 'mu_gamma', 'I_L_ref', 'I_o_ref',
             'R_sh_ref', 'R_sh_0', 'R_sh_exp',
             'R_s', 'alpha_sc', 'EgRef',
             'irrad_ref', 'temp_ref',
             'cells_in_series']
        )

        return tuple(
            calcparams_pvsyst(
                effective_irradiance, temp_cell,
                **build_kwargs(array.module_parameters)
            )
            for array, effective_irradiance, temp_cell
            in zip(self.arrays, effective_irradiance, temp_cell)
        )

    @_unwrap_single_value
    def sapm(self, effective_irradiance, temp_cell):
        """
        Use the :py:func:`sapm` function, the input parameters,
        and ``self.module_parameters`` to calculate
        Voc, Isc, Ix, Ixx, Vmp, and Imp.

        Parameters
        ----------
        effective_irradiance : numeric or tuple of numeric
            The irradiance (W/m2) that is converted to photocurrent.

        temp_cell : float or Series or tuple of float or Series
            The average cell temperature of cells within a module in C.

        Returns
        -------
        See pvsystem.sapm for details
        """
        effective_irradiance = self._validate_per_array(effective_irradiance)
        temp_cell = self._validate_per_array(temp_cell)

        return tuple(
            sapm(effective_irradiance, temp_cell, array.module_parameters)
            for array, effective_irradiance, temp_cell
            in zip(self.arrays, effective_irradiance, temp_cell)
        )

    @deprecated('0.9', alternative='PVSystem.get_cell_temperature',
                removal='0.10.0')
    def sapm_celltemp(self, poa_global, temp_air, wind_speed):
        """Uses :py:func:`pvlib.temperature.sapm_cell` to calculate cell
        temperatures.

        Parameters
        ----------
        poa_global : numeric or tuple of numeric
            Total incident irradiance in W/m^2.

        temp_air : numeric or tuple of numeric
            Ambient dry bulb temperature in degrees C.

        wind_speed : numeric or tuple of numeric
            Wind speed in m/s at a height of 10 meters.

        Returns
        -------
        numeric or tuple of numeric
            values in degrees C.

        Notes
        -----
        The `temp_air` and `wind_speed` parameters may be passed as tuples
        to provide different values for each Array in the system. If not
        passed as a tuple then the same value is used for input to each Array.
        If passed as a tuple the length must be the same as the number of
        Arrays.
        """
        return self.get_cell_temperature(poa_global, temp_air, wind_speed,
                                         model='sapm')

    @_unwrap_single_value
    def sapm_spectral_loss(self, airmass_absolute):
        """
        Use the :py:func:`sapm_spectral_loss` function, the input
        parameters, and ``self.module_parameters`` to calculate F1.

        Parameters
        ----------
        airmass_absolute : numeric
            Absolute airmass.

        Returns
        -------
        F1 : numeric or tuple of numeric
            The SAPM spectral loss coefficient.
        """
        return tuple(
            sapm_spectral_loss(airmass_absolute, array.module_parameters)
            for array in self.arrays
        )

    @_unwrap_single_value
    def sapm_effective_irradiance(self, poa_direct, poa_diffuse,
                                  airmass_absolute, aoi,
                                  reference_irradiance=1000):
        """
        Use the :py:func:`sapm_effective_irradiance` function, the input
        parameters, and ``self.module_parameters`` to calculate
        effective irradiance.

        Parameters
        ----------
        poa_direct : numeric or tuple of numeric
            The direct irradiance incident upon the module.  [W/m2]

        poa_diffuse : numeric or tuple of numeric
            The diffuse irradiance incident on module.  [W/m2]

        airmass_absolute : numeric
            Absolute airmass. [unitless]

        aoi : numeric or tuple of numeric
            Angle of incidence. [degrees]

        Returns
        -------
        effective_irradiance : numeric or tuple of numeric
            The SAPM effective irradiance. [W/m2]
        """
        poa_direct = self._validate_per_array(poa_direct)
        poa_diffuse = self._validate_per_array(poa_diffuse)
        aoi = self._validate_per_array(aoi)
        return tuple(
            sapm_effective_irradiance(
                poa_direct, poa_diffuse, airmass_absolute, aoi,
                array.module_parameters)
            for array, poa_direct, poa_diffuse, aoi
            in zip(self.arrays, poa_direct, poa_diffuse, aoi)
        )

    @deprecated('0.9', alternative='PVSystem.get_cell_temperature',
                removal='0.10.0')
    def pvsyst_celltemp(self, poa_global, temp_air, wind_speed=1.0):
        """Uses :py:func:`pvlib.temperature.pvsyst_cell` to calculate cell
        temperature.

        Parameters
        ----------
        poa_global : numeric or tuple of numeric
            Total incident irradiance in W/m^2.

        temp_air : numeric or tuple of numeric
            Ambient dry bulb temperature in degrees C.

        wind_speed : numeric or tuple of numeric, default 1.0
            Wind speed in m/s measured at the same height for which the wind
            loss factor was determined.  The default value is 1.0, which is
            the wind speed at module height used to determine NOCT.

        Returns
        -------
        numeric or tuple of numeric
            values in degrees C.

        Notes
        -----
        The `temp_air` and `wind_speed` parameters may be passed as tuples
        to provide different values for each Array in the system. If not
        passed as a tuple then the same value is used for input to each Array.
        If passed as a tuple the length must be the same as the number of
        Arrays.
        """
        return self.get_cell_temperature(poa_global, temp_air, wind_speed,
                                         model='pvsyst')

    @deprecated('0.9', alternative='PVSystem.get_cell_temperature',
                removal='0.10.0')
    def faiman_celltemp(self, poa_global, temp_air, wind_speed=1.0):
        """
        Use :py:func:`pvlib.temperature.faiman` to calculate cell temperature.

        Parameters
        ----------
        poa_global : numeric or tuple of numeric
            Total incident irradiance [W/m^2].

        temp_air : numeric or tuple of numeric
            Ambient dry bulb temperature [C].

        wind_speed : numeric or tuple of numeric, default 1.0
            Wind speed in m/s measured at the same height for which the wind
            loss factor was determined.  The default value 1.0 m/s is the wind
            speed at module height used to determine NOCT. [m/s]

        Returns
        -------
        numeric or tuple of numeric
            values in degrees C.

        Notes
        -----
        The `temp_air` and `wind_speed` parameters may be passed as tuples
        to provide different values for each Array in the system. If not
        passed as a tuple then the same value is used for input to each Array.
        If passed as a tuple the length must be the same as the number of
        Arrays.
        """
        return self.get_cell_temperature(poa_global, temp_air, wind_speed,
                                         model='faiman')

    @deprecated('0.9', alternative='PVSystem.get_cell_temperature',
                removal='0.10.0')
    def fuentes_celltemp(self, poa_global, temp_air, wind_speed):
        """
        Use :py:func:`pvlib.temperature.fuentes` to calculate cell temperature.

        Parameters
        ----------
        poa_global : pandas Series or tuple of Series
            Total incident irradiance [W/m^2]

        temp_air : pandas Series or tuple of Series
            Ambient dry bulb temperature [C]

        wind_speed : pandas Series or tuple of Series
            Wind speed [m/s]

        Returns
        -------
        temperature_cell : Series or tuple of Series
            The modeled cell temperature [C]

        Notes
        -----
        The Fuentes thermal model uses the module surface tilt for convection
        modeling. The SAM implementation of PVWatts hardcodes the surface tilt
        value at 30 degrees, ignoring whatever value is used for irradiance
        transposition.  If you want to match the PVWatts behavior you can
        either leave ``surface_tilt`` unspecified to use the PVWatts default
        of 30, or specify a ``surface_tilt`` value in the Array's
        ``temperature_model_parameters``.

        The `temp_air`, `wind_speed`, and `surface_tilt` parameters may be
        passed as tuples
        to provide different values for each Array in the system. If not
        passed as a tuple then the same value is used for input to each Array.
        If passed as a tuple the length must be the same as the number of
        Arrays.
        """
        return self.get_cell_temperature(poa_global, temp_air, wind_speed,
                                         model='fuentes')

    @deprecated('0.9', alternative='PVSystem.get_cell_temperature',
                removal='0.10.0')
    def noct_sam_celltemp(self, poa_global, temp_air, wind_speed,
                          effective_irradiance=None):
        """
        Use :py:func:`pvlib.temperature.noct_sam` to calculate cell
        temperature.

        Parameters
        ----------
        poa_global : numeric or tuple of numeric
            Total incident irradiance in W/m^2.

        temp_air : numeric or tuple of numeric
            Ambient dry bulb temperature in degrees C.

        wind_speed : numeric or tuple of numeric
            Wind speed in m/s at a height of 10 meters.

        effective_irradiance : numeric, tuple of numeric, or None.
            The irradiance that is converted to photocurrent. If None,
            assumed equal to ``poa_global``. [W/m^2]

        Returns
        -------
        temperature_cell : numeric or tuple of numeric
            The modeled cell temperature [C]

        Notes
        -----
        The `temp_air` and `wind_speed` parameters may be passed as tuples
        to provide different values for each Array in the system. If not
        passed as a tuple then the same value is used for input to each Array.
        If passed as a tuple the length must be the same as the number of
        Arrays.
        """
        return self.get_cell_temperature(
            poa_global, temp_air, wind_speed, model='noct_sam',
            effective_irradiance=effective_irradiance)

    @_unwrap_single_value
    def first_solar_spectral_loss(self, pw, airmass_absolute):
        """
        Use :py:func:`pvlib.atmosphere.first_solar_spectral_correction` to
        calculate the spectral loss modifier. The model coefficients are
        specific to the module's cell type, and are determined by searching
        for one of the following keys in self.module_parameters (in order):

        - 'first_solar_spectral_coefficients' (user-supplied coefficients)
        - 'Technology' - a string describing the cell type, can be read from
          the CEC module parameter database
        - 'Material' - a string describing the cell type, can be read from
          the Sandia module database.

        Parameters
        ----------
        pw : array-like
            atmospheric precipitable water (cm).

        airmass_absolute : array-like
            absolute (pressure corrected) airmass.

        Returns
        -------
        modifier: array-like or tuple of array-like
            spectral mismatch factor (unitless) which can be multiplied
            with broadband irradiance reaching a module's cells to estimate
            effective irradiance, i.e., the irradiance that is converted to
            electrical current.
        """
        pw = self._validate_per_array(pw, system_wide=True)

        def _spectral_correction(array, pw):
            if 'first_solar_spectral_coefficients' in \
                    array.module_parameters.keys():
                coefficients = \
                    array.module_parameters[
                        'first_solar_spectral_coefficients'
                    ]
                module_type = None
            else:
                module_type = array._infer_cell_type()
                coefficients = None

            return atmosphere.first_solar_spectral_correction(
                pw, airmass_absolute,
                module_type, coefficients
            )
        return tuple(
            itertools.starmap(_spectral_correction, zip(self.arrays, pw))
        )

    def singlediode(self, photocurrent, saturation_current,
                    resistance_series, resistance_shunt, nNsVth,
                    ivcurve_pnts=None):
        """Wrapper around the :py:func:`pvlib.pvsystem.singlediode` function.

        See :py:func:`pvsystem.singlediode` for details
        """
        return singlediode(photocurrent, saturation_current,
                           resistance_series, resistance_shunt, nNsVth,
                           ivcurve_pnts=ivcurve_pnts)

    def i_from_v(self, resistance_shunt, resistance_series, nNsVth, voltage,
                 saturation_current, photocurrent):
        """Wrapper around the :py:func:`pvlib.pvsystem.i_from_v` function.

        See :py:func:`pvsystem.i_from_v` for details
        """
        return i_from_v(resistance_shunt, resistance_series, nNsVth, voltage,
                        saturation_current, photocurrent)

    def get_ac(self, model, p_dc, v_dc=None):
        r"""Calculates AC power from p_dc using the inverter model indicated
        by model and self.inverter_parameters.

        Parameters
        ----------
        model : str
            Must be one of 'sandia', 'adr', or 'pvwatts'.
        p_dc : numeric, or tuple, list or array of numeric
            DC power on each MPPT input of the inverter. Use tuple, list or
            array for inverters with multiple MPPT inputs. If type is array,
            p_dc must be 2d with axis 0 being the MPPT inputs. [W]
        v_dc : numeric, or tuple, list or array of numeric
            DC voltage on each MPPT input of the inverter. Required when
            model='sandia' or model='adr'. Use tuple, list or
            array for inverters with multiple MPPT inputs. If type is array,
            v_dc must be 2d with axis 0 being the MPPT inputs. [V]

        Returns
        -------
        power_ac : numeric
            AC power output for the inverter. [W]

        Raises
        ------
        ValueError
            If model is not one of 'sandia', 'adr' or 'pvwatts'.
        ValueError
            If model='adr' and the PVSystem has more than one array.

        See also
        --------
        pvlib.inverter.sandia
        pvlib.inverter.sandia_multi
        pvlib.inverter.adr
        pvlib.inverter.pvwatts
        pvlib.inverter.pvwatts_multi
        """
        model = model.lower()
        multiple_arrays = self.num_arrays > 1
        if model == 'sandia':
            p_dc = self._validate_per_array(p_dc)
            v_dc = self._validate_per_array(v_dc)
            if multiple_arrays:
                return inverter.sandia_multi(
                    v_dc, p_dc, self.inverter_parameters)
            return inverter.sandia(v_dc[0], p_dc[0], self.inverter_parameters)
        elif model == 'pvwatts':
            kwargs = _build_kwargs(['eta_inv_nom', 'eta_inv_ref'],
                                   self.inverter_parameters)
            p_dc = self._validate_per_array(p_dc)
            if multiple_arrays:
                return inverter.pvwatts_multi(
                    p_dc, self.inverter_parameters['pdc0'], **kwargs)
            return inverter.pvwatts(
                p_dc[0], self.inverter_parameters['pdc0'], **kwargs)
        elif model == 'adr':
            if multiple_arrays:
                raise ValueError(
                    'The adr inverter function cannot be used for an inverter',
                    ' with multiple MPPT inputs')
            # While this is only used for single-array systems, calling
            # _validate_per_arry lets us pass in singleton tuples.
            p_dc = self._validate_per_array(p_dc)
            v_dc = self._validate_per_array(v_dc)
            return inverter.adr(v_dc[0], p_dc[0], self.inverter_parameters)
        else:
            raise ValueError(
                model + ' is not a valid AC power model.',
                ' model must be one of "sandia", "adr" or "pvwatts"')

    @deprecated('0.9', alternative='PVSystem.get_ac', removal='0.10')
    def snlinverter(self, v_dc, p_dc):
        """Uses :py:func:`pvlib.inverter.sandia` to calculate AC power based on
        ``self.inverter_parameters`` and the input voltage and power.

        See :py:func:`pvlib.inverter.sandia` for details
        """
        return inverter.sandia(v_dc, p_dc, self.inverter_parameters)

    @deprecated('0.9', alternative='PVSystem.get_ac', removal='0.10')
    def adrinverter(self, v_dc, p_dc):
        """Uses :py:func:`pvlib.inverter.adr` to calculate AC power based on
        ``self.inverter_parameters`` and the input voltage and power.

        See :py:func:`pvlib.inverter.adr` for details
        """
        return inverter.adr(v_dc, p_dc, self.inverter_parameters)

    @_unwrap_single_value
    def scale_voltage_current_power(self, data):
        """
        Scales the voltage, current, and power of the `data` DataFrame
        by `self.modules_per_string` and `self.strings_per_inverter`.

        Parameters
        ----------
        data: DataFrame or tuple of DataFrame
            May contain columns `'v_mp', 'v_oc', 'i_mp' ,'i_x', 'i_xx',
            'i_sc', 'p_mp'`.

        Returns
        -------
        scaled_data: DataFrame or tuple of DataFrame
            A scaled copy of the input data.
        """
        data = self._validate_per_array(data)
        return tuple(
            scale_voltage_current_power(data,
                                        voltage=array.modules_per_string,
                                        current=array.strings)
            for array, data in zip(self.arrays, data)
        )

    @_unwrap_single_value
    def pvwatts_dc(self, g_poa_effective, temp_cell):
        """
        Calcuates DC power according to the PVWatts model using
        :py:func:`pvlib.pvsystem.pvwatts_dc`, `self.module_parameters['pdc0']`,
        and `self.module_parameters['gamma_pdc']`.

        See :py:func:`pvlib.pvsystem.pvwatts_dc` for details.
        """
        g_poa_effective = self._validate_per_array(g_poa_effective)
        temp_cell = self._validate_per_array(temp_cell)
        return tuple(
            pvwatts_dc(g_poa_effective, temp_cell,
                       array.module_parameters['pdc0'],
                       array.module_parameters['gamma_pdc'],
                       **_build_kwargs(['temp_ref'], array.module_parameters))
            for array, g_poa_effective, temp_cell
            in zip(self.arrays, g_poa_effective, temp_cell)
        )

    def pvwatts_losses(self):
        """
        Calculates DC power losses according the PVwatts model using
        :py:func:`pvlib.pvsystem.pvwatts_losses` and
        ``self.losses_parameters``.

        See :py:func:`pvlib.pvsystem.pvwatts_losses` for details.
        """
        kwargs = _build_kwargs(['soiling', 'shading', 'snow', 'mismatch',
                                'wiring', 'connections', 'lid',
                                'nameplate_rating', 'age', 'availability'],
                               self.losses_parameters)
        return pvwatts_losses(**kwargs)

    @deprecated('0.9', alternative='PVSystem.get_ac', removal='0.10')
    def pvwatts_ac(self, pdc):
        """
        Calculates AC power according to the PVWatts model using
        :py:func:`pvlib.inverter.pvwatts`, `self.module_parameters["pdc0"]`,
        and `eta_inv_nom=self.inverter_parameters["eta_inv_nom"]`.

        See :py:func:`pvlib.inverter.pvwatts` for details.
        """
        kwargs = _build_kwargs(['eta_inv_nom', 'eta_inv_ref'],
                               self.inverter_parameters)

        return inverter.pvwatts(pdc, self.inverter_parameters['pdc0'],
                                **kwargs)

    @_unwrap_single_value
    def dc_ohms_from_percent(self):
        """
        Calculates the equivalent resistance of the wires for each array using
        :py:func:`pvlib.pvsystem.dc_ohms_from_percent`

        See :py:func:`pvlib.pvsystem.dc_ohms_from_percent` for details.
        """

        return tuple(array.dc_ohms_from_percent() for array in self.arrays)

    @property
    @_unwrap_single_value
    @_check_deprecated_passthrough
    def module_parameters(self):
        return tuple(array.module_parameters for array in self.arrays)

    @module_parameters.setter
    @_check_deprecated_passthrough
    def module_parameters(self, value):
        for array in self.arrays:
            array.module_parameters = value

    @property
    @_unwrap_single_value
    @_check_deprecated_passthrough
    def module(self):
        return tuple(array.module for array in self.arrays)

    @module.setter
    @_check_deprecated_passthrough
    def module(self, value):
        for array in self.arrays:
            array.module = value

    @property
    @_unwrap_single_value
    @_check_deprecated_passthrough
    def module_type(self):
        return tuple(array.module_type for array in self.arrays)

    @module_type.setter
    @_check_deprecated_passthrough
    def module_type(self, value):
        for array in self.arrays:
            array.module_type = value

    @property
    @_unwrap_single_value
    @_check_deprecated_passthrough
    def temperature_model_parameters(self):
        return tuple(array.temperature_model_parameters
                     for array in self.arrays)

    @temperature_model_parameters.setter
    @_check_deprecated_passthrough
    def temperature_model_parameters(self, value):
        for array in self.arrays:
            array.temperature_model_parameters = value

    @property
    @_unwrap_single_value
    @_check_deprecated_passthrough
    def surface_tilt(self):
        return tuple(array.mount.surface_tilt for array in self.arrays)

    @surface_tilt.setter
    @_check_deprecated_passthrough
    def surface_tilt(self, value):
        for array in self.arrays:
            array.mount.surface_tilt = value

    @property
    @_unwrap_single_value
    @_check_deprecated_passthrough
    def surface_azimuth(self):
        return tuple(array.mount.surface_azimuth for array in self.arrays)

    @surface_azimuth.setter
    @_check_deprecated_passthrough
    def surface_azimuth(self, value):
        for array in self.arrays:
            array.mount.surface_azimuth = value

    @property
    @_unwrap_single_value
    @_check_deprecated_passthrough
    def albedo(self):
        return tuple(array.albedo for array in self.arrays)

    @albedo.setter
    @_check_deprecated_passthrough
    def albedo(self, value):
        for array in self.arrays:
            array.albedo = value

    @property
    @_unwrap_single_value
    @_check_deprecated_passthrough
    def racking_model(self):
        return tuple(array.mount.racking_model for array in self.arrays)

    @racking_model.setter
    @_check_deprecated_passthrough
    def racking_model(self, value):
        for array in self.arrays:
            array.mount.racking_model = value

    @property
    @_unwrap_single_value
    @_check_deprecated_passthrough
    def modules_per_string(self):
        return tuple(array.modules_per_string for array in self.arrays)

    @modules_per_string.setter
    @_check_deprecated_passthrough
    def modules_per_string(self, value):
        for array in self.arrays:
            array.modules_per_string = value

    @property
    @_unwrap_single_value
    @_check_deprecated_passthrough
    def strings_per_inverter(self):
        return tuple(array.strings for array in self.arrays)

    @strings_per_inverter.setter
    @_check_deprecated_passthrough
    def strings_per_inverter(self, value):
        for array in self.arrays:
            array.strings = value

    @property
    def num_arrays(self):
        """The number of Arrays in the system."""
        return len(self.arrays)


class Array:
    """
    An Array is a set of of modules at the same orientation.

    Specifically, an array is defined by its mount, the
    module parameters, the number of parallel strings of modules
    and the number of modules on each string.

    Parameters
    ----------
    mount: FixedMount, SingleAxisTrackerMount, or other
        Mounting for the array, either on fixed-tilt racking or horizontal
        single axis tracker. Mounting is used to determine module orientation.
        If not provided, a FixedMount with zero tilt is used.

    albedo : None or float, default None
        Ground surface albedo. If ``None``, then ``surface_type`` is used
        to look up a value in ``irradiance.SURFACE_ALBEDOS``.
        If ``surface_type`` is also None then a ground surface albedo
        of 0.25 is used.

    surface_type : None or string, default None
        The ground surface type. See ``irradiance.SURFACE_ALBEDOS`` for valid
        values.

    module : None or string, default None
        The model name of the modules.
        May be used to look up the module_parameters dictionary
        via some other method.

    module_type : None or string, default None
         Describes the module's construction. Valid strings are 'glass_polymer'
         and 'glass_glass'. Used for cell and module temperature calculations.

    module_parameters : None, dict or Series, default None
        Parameters for the module model, e.g., SAPM, CEC, or other.

    temperature_model_parameters : None, dict or Series, default None.
        Parameters for the module temperature model, e.g., SAPM, Pvsyst, or
        other.

    modules_per_string: int, default 1
        Number of modules per string in the array.

    strings: int, default 1
        Number of parallel strings in the array.

    array_losses_parameters: None, dict or Series, default None.
        Supported keys are 'dc_ohmic_percent'.

    name: None or str, default None
        Name of Array instance.
    """

    def __init__(self, mount,
                 albedo=None, surface_type=None,
                 module=None, module_type=None,
                 module_parameters=None,
                 temperature_model_parameters=None,
                 modules_per_string=1, strings=1,
                 array_losses_parameters=None,
                 name=None):
        self.mount = mount

        self.surface_type = surface_type
        if albedo is None:
            self.albedo = irradiance.SURFACE_ALBEDOS.get(surface_type, 0.25)
        else:
            self.albedo = albedo

        self.module = module
        if module_parameters is None:
            self.module_parameters = {}
        else:
            self.module_parameters = module_parameters

        self.module_type = module_type

        self.strings = strings
        self.modules_per_string = modules_per_string

        if temperature_model_parameters is None:
            self.temperature_model_parameters = \
                self._infer_temperature_model_params()
        else:
            self.temperature_model_parameters = temperature_model_parameters

        if array_losses_parameters is None:
            self.array_losses_parameters = {}
        else:
            self.array_losses_parameters = array_losses_parameters

        self.name = name

    def __repr__(self):
        attrs = ['name', 'mount', 'module',
                 'albedo', 'module_type',
                 'temperature_model_parameters',
                 'strings', 'modules_per_string']

        return 'Array:\n  ' + '\n  '.join(
            f'{attr}: {getattr(self, attr)}' for attr in attrs
        )

    def _infer_temperature_model_params(self):
        # try to infer temperature model parameters from from racking_model
        # and module_type
        param_set = f'{self.mount.racking_model}_{self.module_type}'
        if param_set in temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']:
            return temperature._temperature_model_params('sapm', param_set)
        elif 'freestanding' in param_set:
            return temperature._temperature_model_params('pvsyst',
                                                         'freestanding')
        elif 'insulated' in param_set:  # after SAPM to avoid confusing keys
            return temperature._temperature_model_params('pvsyst',
                                                         'insulated')
        else:
            return {}

    def _infer_cell_type(self):
        """
        Examines module_parameters and maps the Technology key for the CEC
        database and the Material key for the Sandia database to a common
        list of strings for cell type.

        Returns
        -------
        cell_type: str

        """

        _cell_type_dict = {'Multi-c-Si': 'multisi',
                           'Mono-c-Si': 'monosi',
                           'Thin Film': 'cigs',
                           'a-Si/nc': 'asi',
                           'CIS': 'cigs',
                           'CIGS': 'cigs',
                           '1-a-Si': 'asi',
                           'CdTe': 'cdte',
                           'a-Si': 'asi',
                           '2-a-Si': None,
                           '3-a-Si': None,
                           'HIT-Si': 'monosi',
                           'mc-Si': 'multisi',
                           'c-Si': 'multisi',
                           'Si-Film': 'asi',
                           'EFG mc-Si': 'multisi',
                           'GaAs': None,
                           'a-Si / mono-Si': 'monosi'}

        if 'Technology' in self.module_parameters.keys():
            # CEC module parameter set
            cell_type = _cell_type_dict[self.module_parameters['Technology']]
        elif 'Material' in self.module_parameters.keys():
            # Sandia module parameter set
            cell_type = _cell_type_dict[self.module_parameters['Material']]
        else:
            cell_type = None

        return cell_type

    def get_aoi(self, solar_zenith, solar_azimuth):
        """
        Get the angle of incidence on the array.

        Parameters
        ----------
        solar_zenith : float or Series
            Solar zenith angle.
        solar_azimuth : float or Series
            Solar azimuth angle

        Returns
        -------
        aoi : Series
            Then angle of incidence.
        """
        orientation = self.mount.get_orientation(solar_zenith, solar_azimuth)
        return irradiance.aoi(orientation['surface_tilt'],
                              orientation['surface_azimuth'],
                              solar_zenith, solar_azimuth)

    def get_irradiance(self, solar_zenith, solar_azimuth, dni, ghi, dhi,
                       dni_extra=None, airmass=None, albedo=None,
                       model='haydavies', **kwargs):
        """
        Get plane of array irradiance components.

        Uses the :py:func:`pvlib.irradiance.get_total_irradiance` function to
        calculate the plane of array irradiance components for a surface
        defined by ``self.surface_tilt`` and ``self.surface_azimuth``.

        Parameters
        ----------
        solar_zenith : float or Series.
            Solar zenith angle.
        solar_azimuth : float or Series.
            Solar azimuth angle.
        dni : float or Series
            Direct normal irradiance. [W/m2]
        ghi : float or Series. [W/m2]
            Global horizontal irradiance
        dhi : float or Series
            Diffuse horizontal irradiance. [W/m2]
        dni_extra : None, float or Series, default None
            Extraterrestrial direct normal irradiance. [W/m2]
        airmass : None, float or Series, default None
            Airmass. [unitless]
        albedo : None, float or Series, default None
            Ground surface albedo. [unitless]
        model : String, default 'haydavies'
            Irradiance model.

        kwargs
            Extra parameters passed to
            :py:func:`pvlib.irradiance.get_total_irradiance`.

        Returns
        -------
        poa_irradiance : DataFrame
            Column names are: ``'poa_global', 'poa_direct', 'poa_diffuse',
            'poa_sky_diffuse', 'poa_ground_diffuse'``.

        See also
        --------
        :py:func:`pvlib.irradiance.get_total_irradiance`
        """
        if albedo is None:
            albedo = self.albedo

        # not needed for all models, but this is easier
        if dni_extra is None:
            dni_extra = irradiance.get_extra_radiation(solar_zenith.index)

        if airmass is None:
            airmass = atmosphere.get_relative_airmass(solar_zenith)

        orientation = self.mount.get_orientation(solar_zenith, solar_azimuth)
        return irradiance.get_total_irradiance(orientation['surface_tilt'],
                                               orientation['surface_azimuth'],
                                               solar_zenith, solar_azimuth,
                                               dni, ghi, dhi,
                                               dni_extra=dni_extra,
                                               airmass=airmass,
                                               albedo=albedo,
                                               model=model,
                                               **kwargs)

    def get_iam(self, aoi, iam_model='physical'):
        """
        Determine the incidence angle modifier using the method specified by
        ``iam_model``.

        Parameters for the selected IAM model are expected to be in
        ``Array.module_parameters``. Default parameters are available for
        the 'physical', 'ashrae' and 'martin_ruiz' models.

        Parameters
        ----------
        aoi : numeric
            The angle of incidence in degrees.

        aoi_model : string, default 'physical'
            The IAM model to be used. Valid strings are 'physical', 'ashrae',
            'martin_ruiz' and 'sapm'.

        Returns
        -------
        iam : numeric
            The AOI modifier.

        Raises
        ------
        ValueError
            if `iam_model` is not a valid model name.
        """
        model = iam_model.lower()
        if model in ['ashrae', 'physical', 'martin_ruiz']:
            param_names = iam._IAM_MODEL_PARAMS[model]
            kwargs = _build_kwargs(param_names, self.module_parameters)
            func = getattr(iam, model)
            return func(aoi, **kwargs)
        elif model == 'sapm':
            return iam.sapm(aoi, self.module_parameters)
        elif model == 'interp':
            raise ValueError(model + ' is not implemented as an IAM model '
                             'option for Array')
        else:
            raise ValueError(model + ' is not a valid IAM model')

    def get_cell_temperature(self, poa_global, temp_air, wind_speed, model,
                             effective_irradiance=None):
        """
        Determine cell temperature using the method specified by ``model``.

        Parameters
        ----------
        poa_global : numeric
            Total incident irradiance [W/m^2]

        temp_air : numeric
            Ambient dry bulb temperature [C]

        wind_speed : numeric
            Wind speed [m/s]

        model : str
            Supported models include ``'sapm'``, ``'pvsyst'``,
            ``'faiman'``, ``'fuentes'``, and ``'noct_sam'``

        effective_irradiance : numeric, optional
            The irradiance that is converted to photocurrent in W/m^2.
            Only used for some models.

        Returns
        -------
        numeric
            Values in degrees C.

        See Also
        --------
        pvlib.temperature.sapm_cell, pvlib.temperature.pvsyst_cell,
        pvlib.temperature.faiman, pvlib.temperature.fuentes,
        pvlib.temperature.noct_sam

        Notes
        -----
        Some temperature models have requirements for the input types;
        see the documentation of the underlying model function for details.
        """
        # convenience wrapper to avoid passing args 2 and 3 every call
        _build_tcell_args = functools.partial(
            _build_args, input_dict=self.temperature_model_parameters,
            dict_name='temperature_model_parameters')

        if model == 'sapm':
            func = temperature.sapm_cell
            required = _build_tcell_args(['a', 'b', 'deltaT'])
            optional = _build_kwargs(['irrad_ref'],
                                     self.temperature_model_parameters)
        elif model == 'pvsyst':
            func = temperature.pvsyst_cell
            required = tuple()
            optional = {
                # TODO remove 'eta_m' after deprecation of this parameter
                **_build_kwargs(['eta_m', 'module_efficiency',
                                 'alpha_absorption'],
                                self.module_parameters),
                **_build_kwargs(['u_c', 'u_v'],
                                self.temperature_model_parameters)
            }
        elif model == 'faiman':
            func = temperature.faiman
            required = tuple()
            optional = _build_kwargs(['u0', 'u1'],
                                     self.temperature_model_parameters)
        elif model == 'fuentes':
            func = temperature.fuentes
            required = _build_tcell_args(['noct_installed'])
            optional = _build_kwargs([
                'wind_height', 'emissivity', 'absorption',
                'surface_tilt', 'module_width', 'module_length'],
                self.temperature_model_parameters)
            if self.mount.module_height is not None:
                optional['module_height'] = self.mount.module_height
        elif model == 'noct_sam':
            func = functools.partial(temperature.noct_sam,
                                     effective_irradiance=effective_irradiance)
            required = _build_tcell_args(['noct', 'module_efficiency'])
            optional = _build_kwargs(['transmittance_absorptance',
                                      'array_height', 'mount_standoff'],
                                     self.temperature_model_parameters)
        else:
            raise ValueError(f'{model} is not a valid cell temperature model')

        temperature_cell = func(poa_global, temp_air, wind_speed,
                                *required, **optional)
        return temperature_cell

    def dc_ohms_from_percent(self):
        """
        Calculates the equivalent resistance of the wires using
        :py:func:`pvlib.pvsystem.dc_ohms_from_percent`

        Makes use of array module parameters according to the
        following DC models:

        CEC:

            * `self.module_parameters["V_mp_ref"]`
            * `self.module_parameters["I_mp_ref"]`

        SAPM:

            * `self.module_parameters["Vmpo"]`
            * `self.module_parameters["Impo"]`

        PVsyst-like or other:

            * `self.module_parameters["Vmpp"]`
            * `self.module_parameters["Impp"]`

        Other array parameters that are used are:
        `self.losses_parameters["dc_ohmic_percent"]`,
        `self.modules_per_string`, and
        `self.strings`.

        See :py:func:`pvlib.pvsystem.dc_ohms_from_percent` for more details.
        """

        # get relevent Vmp and Imp parameters from CEC parameters
        if all([elem in self.module_parameters
                for elem in ['V_mp_ref', 'I_mp_ref']]):
            vmp_ref = self.module_parameters['V_mp_ref']
            imp_ref = self.module_parameters['I_mp_ref']

        # get relevant Vmp and Imp parameters from SAPM parameters
        elif all([elem in self.module_parameters
                  for elem in ['Vmpo', 'Impo']]):
            vmp_ref = self.module_parameters['Vmpo']
            imp_ref = self.module_parameters['Impo']

        # get relevant Vmp and Imp parameters if they are PVsyst-like
        elif all([elem in self.module_parameters
                  for elem in ['Vmpp', 'Impp']]):
            vmp_ref = self.module_parameters['Vmpp']
            imp_ref = self.module_parameters['Impp']

        # raise error if relevant Vmp and Imp parameters are not found
        else:
            raise ValueError('Parameters for Vmp and Imp could not be found '
                             'in the array module parameters. Module '
                             'parameters must include one set of '
                             '{"V_mp_ref", "I_mp_Ref"}, '
                             '{"Vmpo", "Impo"}, or '
                             '{"Vmpp", "Impp"}.'
                             )

        return dc_ohms_from_percent(
            vmp_ref,
            imp_ref,
            self.array_losses_parameters['dc_ohmic_percent'],
            self.modules_per_string,
            self.strings)


@dataclass
class AbstractMount(ABC):
    """
    A base class for Mount classes to extend. It is not intended to be
    instantiated directly.
    """

    @abstractmethod
    def get_orientation(self, solar_zenith, solar_azimuth):
        """
        Determine module orientation.

        Parameters
        ----------
        solar_zenith : numeric
            Solar apparent zenith angle [degrees]
        solar_azimuth : numeric
            Solar azimuth angle [degrees]

        Returns
        -------
        orientation : dict-like
            A dict-like object with keys `'surface_tilt', 'surface_azimuth'`
            (typically a dict or pandas.DataFrame)
        """


@dataclass
class FixedMount(AbstractMount):
    """
    Racking at fixed (static) orientation.

    Parameters
    ----------
    surface_tilt : float, default 0
        Surface tilt angle. The tilt angle is defined as angle from horizontal
        (e.g. surface facing up = 0, surface facing horizon = 90) [degrees]

    surface_azimuth : float, default 180
        Azimuth angle of the module surface. North=0, East=90, South=180,
        West=270. [degrees]

    racking_model : str, optional
        Valid strings are 'open_rack', 'close_mount', and 'insulated_back'.
        Used to identify a parameter set for the SAPM cell temperature model.

    module_height : float, optional
       The height above ground of the center of the module [m]. Used for
       the Fuentes cell temperature model.
    """

    surface_tilt: float = 0.0
    surface_azimuth: float = 180.0
    racking_model: Optional[str] = None
    module_height: Optional[float] = None

    def get_orientation(self, solar_zenith, solar_azimuth):
        # note -- docstring is automatically inherited from AbstractMount
        return {
            'surface_tilt': self.surface_tilt,
            'surface_azimuth': self.surface_azimuth,
        }


@dataclass
class SingleAxisTrackerMount(AbstractMount):
    """
    Single-axis tracker racking for dynamic solar tracking.

    Parameters
    ----------
    axis_tilt : float, default 0
        The tilt of the axis of rotation (i.e, the y-axis defined by
        axis_azimuth) with respect to horizontal. [degrees]

    axis_azimuth : float, default 180
        A value denoting the compass direction along which the axis of
        rotation lies, measured east of north. [degrees]

    max_angle : float, default 90
        A value denoting the maximum rotation angle
        of the one-axis tracker from its horizontal position (horizontal
        if axis_tilt = 0). A max_angle of 90 degrees allows the tracker
        to rotate to a vertical position to point the panel towards a
        horizon. max_angle of 180 degrees allows for full rotation. [degrees]

    backtrack : bool, default True
        Controls whether the tracker has the capability to "backtrack"
        to avoid row-to-row shading. False denotes no backtrack
        capability. True denotes backtrack capability.

    gcr : float, default 2.0/7.0
        A value denoting the ground coverage ratio of a tracker system
        which utilizes backtracking; i.e. the ratio between the PV array
        surface area to total ground area. A tracker system with modules
        2 meters wide, centered on the tracking axis, with 6 meters
        between the tracking axes has a gcr of 2/6=0.333. If gcr is not
        provided, a gcr of 2/7 is default. gcr must be <=1. [unitless]

    cross_axis_tilt : float, default 0.0
        The angle, relative to horizontal, of the line formed by the
        intersection between the slope containing the tracker axes and a plane
        perpendicular to the tracker axes. Cross-axis tilt should be specified
        using a right-handed convention. For example, trackers with axis
        azimuth of 180 degrees (heading south) will have a negative cross-axis
        tilt if the tracker axes plane slopes down to the east and positive
        cross-axis tilt if the tracker axes plane slopes up to the east. Use
        :func:`~pvlib.tracking.calc_cross_axis_tilt` to calculate
        `cross_axis_tilt`. [degrees]

    racking_model : str, optional
        Valid strings are 'open_rack', 'close_mount', and 'insulated_back'.
        Used to identify a parameter set for the SAPM cell temperature model.

    module_height : float, optional
       The height above ground of the center of the module [m]. Used for
       the Fuentes cell temperature model.
    """
    axis_tilt: float = 0.0
    axis_azimuth: float = 0.0
    max_angle: float = 90.0
    backtrack: bool = True
    gcr: float = 2.0/7.0
    cross_axis_tilt: float = 0.0
    racking_model: Optional[str] = None
    module_height: Optional[float] = None

    def get_orientation(self, solar_zenith, solar_azimuth):
        # note -- docstring is automatically inherited from AbstractMount
        from pvlib import tracking  # avoid circular import issue
        tracking_data = tracking.singleaxis(
            solar_zenith, solar_azimuth,
            self.axis_tilt, self.axis_azimuth,
            self.max_angle, self.backtrack,
            self.gcr, self.cross_axis_tilt
        )
        return tracking_data


def calcparams_desoto(effective_irradiance, temp_cell,
                      alpha_sc, a_ref, I_L_ref, I_o_ref, R_sh_ref, R_s,
                      EgRef=1.121, dEgdT=-0.0002677,
                      irrad_ref=1000, temp_ref=25):
    '''
    Calculates five parameter values for the single diode equation at
    effective irradiance and cell temperature using the De Soto et al.
    model described in [1]_. The five values returned by calcparams_desoto
    can be used by singlediode to calculate an IV curve.

    Parameters
    ----------
    effective_irradiance : numeric
        The irradiance (W/m2) that is converted to photocurrent.

    temp_cell : numeric
        The average cell temperature of cells within a module in C.

    alpha_sc : float
        The short-circuit current temperature coefficient of the
        module in units of A/C.

    a_ref : float
        The product of the usual diode ideality factor (n, unitless),
        number of cells in series (Ns), and cell thermal voltage at reference
        conditions, in units of V.

    I_L_ref : float
        The light-generated current (or photocurrent) at reference conditions,
        in amperes.

    I_o_ref : float
        The dark or diode reverse saturation current at reference conditions,
        in amperes.

    R_sh_ref : float
        The shunt resistance at reference conditions, in ohms.

    R_s : float
        The series resistance at reference conditions, in ohms.

    EgRef : float
        The energy bandgap at reference temperature in units of eV.
        1.121 eV for crystalline silicon. EgRef must be >0.  For parameters
        from the SAM CEC module database, EgRef=1.121 is implicit for all
        cell types in the parameter estimation algorithm used by NREL.

    dEgdT : float
        The temperature dependence of the energy bandgap at reference
        conditions in units of 1/K. May be either a scalar value
        (e.g. -0.0002677 as in [1]_) or a DataFrame (this may be useful if
        dEgdT is a modeled as a function of temperature). For parameters from
        the SAM CEC module database, dEgdT=-0.0002677 is implicit for all cell
        types in the parameter estimation algorithm used by NREL.

    irrad_ref : float (optional, default=1000)
        Reference irradiance in W/m^2.

    temp_ref : float (optional, default=25)
        Reference cell temperature in C.

    Returns
    -------
    Tuple of the following results:

    photocurrent : numeric
        Light-generated current in amperes

    saturation_current : numeric
        Diode saturation curent in amperes

    resistance_series : float
        Series resistance in ohms

    resistance_shunt : numeric
        Shunt resistance in ohms

    nNsVth : numeric
        The product of the usual diode ideality factor (n, unitless),
        number of cells in series (Ns), and cell thermal voltage at
        specified effective irradiance and cell temperature.

    References
    ----------
    .. [1] W. De Soto et al., "Improvement and validation of a model for
       photovoltaic array performance", Solar Energy, vol 80, pp. 78-88,
       2006.

    .. [2] System Advisor Model web page. https://sam.nrel.gov.

    .. [3] A. Dobos, "An Improved Coefficient Calculator for the California
       Energy Commission 6 Parameter Photovoltaic Module Model", Journal of
       Solar Energy Engineering, vol 134, 2012.

    .. [4] O. Madelung, "Semiconductors: Data Handbook, 3rd ed." ISBN
       3-540-40488-0

    See Also
    --------
    singlediode
    retrieve_sam

    Notes
    -----
    If the reference parameters in the ModuleParameters struct are read
    from a database or library of parameters (e.g. System Advisor
    Model), it is important to use the same EgRef and dEgdT values that
    were used to generate the reference parameters, regardless of the
    actual bandgap characteristics of the semiconductor. For example, in
    the case of the System Advisor Model library, created as described
    in [3], EgRef and dEgdT for all modules were 1.121 and -0.0002677,
    respectively.

    This table of reference bandgap energies (EgRef), bandgap energy
    temperature dependence (dEgdT), and "typical" airmass response (M)
    is provided purely as reference to those who may generate their own
    reference module parameters (a_ref, IL_ref, I0_ref, etc.) based upon
    the various PV semiconductors. Again, we stress the importance of
    using identical EgRef and dEgdT when generation reference parameters
    and modifying the reference parameters (for irradiance, temperature,
    and airmass) per DeSoto's equations.

     Crystalline Silicon (Si):
         * EgRef = 1.121
         * dEgdT = -0.0002677

         >>> M = np.polyval([-1.26E-4, 2.816E-3, -0.024459, 0.086257, 0.9181],
         ...                AMa) # doctest: +SKIP

         Source: [1]

     Cadmium Telluride (CdTe):
         * EgRef = 1.475
         * dEgdT = -0.0003

         >>> M = np.polyval([-2.46E-5, 9.607E-4, -0.0134, 0.0716, 0.9196],
         ...                AMa) # doctest: +SKIP

         Source: [4]

     Copper Indium diSelenide (CIS):
         * EgRef = 1.010
         * dEgdT = -0.00011

         >>> M = np.polyval([-3.74E-5, 0.00125, -0.01462, 0.0718, 0.9210],
         ...                AMa) # doctest: +SKIP

         Source: [4]

     Copper Indium Gallium diSelenide (CIGS):
         * EgRef = 1.15
         * dEgdT = ????

         >>> M = np.polyval([-9.07E-5, 0.0022, -0.0202, 0.0652, 0.9417],
         ...                AMa) # doctest: +SKIP

         Source: Wikipedia

     Gallium Arsenide (GaAs):
         * EgRef = 1.424
         * dEgdT = -0.000433
         * M = unknown

         Source: [4]
    '''

    # Boltzmann constant in eV/K
    k = 8.617332478e-05

    # reference temperature
    Tref_K = temp_ref + 273.15
    Tcell_K = temp_cell + 273.15

    E_g = EgRef * (1 + dEgdT*(Tcell_K - Tref_K))

    nNsVth = a_ref * (Tcell_K / Tref_K)

    # In the equation for IL, the single factor effective_irradiance is
    # used, in place of the product S*M in [1]. effective_irradiance is
    # equivalent to the product of S (irradiance reaching a module's cells) *
    # M (spectral adjustment factor) as described in [1].
    IL = effective_irradiance / irrad_ref * \
        (I_L_ref + alpha_sc * (Tcell_K - Tref_K))
    I0 = (I_o_ref * ((Tcell_K / Tref_K) ** 3) *
          (np.exp(EgRef / (k*(Tref_K)) - (E_g / (k*(Tcell_K))))))
    # Note that the equation for Rsh differs from [1]. In [1] Rsh is given as
    # Rsh = Rsh_ref * (S_ref / S) where S is broadband irradiance reaching
    # the module's cells. If desired this model behavior can be duplicated
    # by applying reflection and soiling losses to broadband plane of array
    # irradiance and not applying a spectral loss modifier, i.e.,
    # spectral_modifier = 1.0.
    # use errstate to silence divide by warning
    with np.errstate(divide='ignore'):
        Rsh = R_sh_ref * (irrad_ref / effective_irradiance)
    Rs = R_s

    return IL, I0, Rs, Rsh, nNsVth


def calcparams_cec(effective_irradiance, temp_cell,
                   alpha_sc, a_ref, I_L_ref, I_o_ref, R_sh_ref, R_s,
                   Adjust, EgRef=1.121, dEgdT=-0.0002677,
                   irrad_ref=1000, temp_ref=25):
    '''
    Calculates five parameter values for the single diode equation at
    effective irradiance and cell temperature using the CEC
    model. The CEC model [1]_ differs from the De soto et al.
    model [3]_ by the parameter Adjust. The five values returned by
    calcparams_cec can be used by singlediode to calculate an IV curve.

    Parameters
    ----------
    effective_irradiance : numeric
        The irradiance (W/m2) that is converted to photocurrent.

    temp_cell : numeric
        The average cell temperature of cells within a module in C.

    alpha_sc : float
        The short-circuit current temperature coefficient of the
        module in units of A/C.

    a_ref : float
        The product of the usual diode ideality factor (n, unitless),
        number of cells in series (Ns), and cell thermal voltage at reference
        conditions, in units of V.

    I_L_ref : float
        The light-generated current (or photocurrent) at reference conditions,
        in amperes.

    I_o_ref : float
        The dark or diode reverse saturation current at reference conditions,
        in amperes.

    R_sh_ref : float
        The shunt resistance at reference conditions, in ohms.

    R_s : float
        The series resistance at reference conditions, in ohms.

    Adjust : float
        The adjustment to the temperature coefficient for short circuit
        current, in percent

    EgRef : float
        The energy bandgap at reference temperature in units of eV.
        1.121 eV for crystalline silicon. EgRef must be >0.  For parameters
        from the SAM CEC module database, EgRef=1.121 is implicit for all
        cell types in the parameter estimation algorithm used by NREL.

    dEgdT : float
        The temperature dependence of the energy bandgap at reference
        conditions in units of 1/K. May be either a scalar value
        (e.g. -0.0002677 as in [3]) or a DataFrame (this may be useful if
        dEgdT is a modeled as a function of temperature). For parameters from
        the SAM CEC module database, dEgdT=-0.0002677 is implicit for all cell
        types in the parameter estimation algorithm used by NREL.

    irrad_ref : float (optional, default=1000)
        Reference irradiance in W/m^2.

    temp_ref : float (optional, default=25)
        Reference cell temperature in C.

    Returns
    -------
    Tuple of the following results:

    photocurrent : numeric
        Light-generated current in amperes

    saturation_current : numeric
        Diode saturation curent in amperes

    resistance_series : float
        Series resistance in ohms

    resistance_shunt : numeric
        Shunt resistance in ohms

    nNsVth : numeric
        The product of the usual diode ideality factor (n, unitless),
        number of cells in series (Ns), and cell thermal voltage at
        specified effective irradiance and cell temperature.

    References
    ----------
    .. [1] A. Dobos, "An Improved Coefficient Calculator for the California
       Energy Commission 6 Parameter Photovoltaic Module Model", Journal of
       Solar Energy Engineering, vol 134, 2012.

    .. [2] System Advisor Model web page. https://sam.nrel.gov.

    .. [3] W. De Soto et al., "Improvement and validation of a model for
       photovoltaic array performance", Solar Energy, vol 80, pp. 78-88,
       2006.

    See Also
    --------
    calcparams_desoto
    singlediode
    retrieve_sam

    '''

    # pass adjusted temperature coefficient to desoto
    return calcparams_desoto(effective_irradiance, temp_cell,
                             alpha_sc*(1.0 - Adjust/100),
                             a_ref, I_L_ref, I_o_ref,
                             R_sh_ref, R_s,
                             EgRef=EgRef, dEgdT=dEgdT,
                             irrad_ref=irrad_ref, temp_ref=temp_ref)


def calcparams_pvsyst(effective_irradiance, temp_cell,
                      alpha_sc, gamma_ref, mu_gamma,
                      I_L_ref, I_o_ref,
                      R_sh_ref, R_sh_0, R_s,
                      cells_in_series,
                      R_sh_exp=5.5,
                      EgRef=1.121,
                      irrad_ref=1000, temp_ref=25):
    '''
    Calculates five parameter values for the single diode equation at
    effective irradiance and cell temperature using the PVsyst v6
    model.  The PVsyst v6 model is described in [1]_, [2]_, [3]_.
    The five values returned by calcparams_pvsyst can be used by singlediode
    to calculate an IV curve.

    Parameters
    ----------
    effective_irradiance : numeric
        The irradiance (W/m2) that is converted to photocurrent.

    temp_cell : numeric
        The average cell temperature of cells within a module in C.

    alpha_sc : float
        The short-circuit current temperature coefficient of the
        module in units of A/C.

    gamma_ref : float
        The diode ideality factor

    mu_gamma : float
        The temperature coefficient for the diode ideality factor, 1/K

    I_L_ref : float
        The light-generated current (or photocurrent) at reference conditions,
        in amperes.

    I_o_ref : float
        The dark or diode reverse saturation current at reference conditions,
        in amperes.

    R_sh_ref : float
        The shunt resistance at reference conditions, in ohms.

    R_sh_0 : float
        The shunt resistance at zero irradiance conditions, in ohms.

    R_s : float
        The series resistance at reference conditions, in ohms.

    cells_in_series : integer
        The number of cells connected in series.

    R_sh_exp : float
        The exponent in the equation for shunt resistance, unitless. Defaults
        to 5.5.

    EgRef : float
        The energy bandgap at reference temperature in units of eV.
        1.121 eV for crystalline silicon. EgRef must be >0.

    irrad_ref : float (optional, default=1000)
        Reference irradiance in W/m^2.

    temp_ref : float (optional, default=25)
        Reference cell temperature in C.

    Returns
    -------
    Tuple of the following results:

    photocurrent : numeric
        Light-generated current in amperes

    saturation_current : numeric
        Diode saturation current in amperes

    resistance_series : float
        Series resistance in ohms

    resistance_shunt : numeric
        Shunt resistance in ohms

    nNsVth : numeric
        The product of the usual diode ideality factor (n, unitless),
        number of cells in series (Ns), and cell thermal voltage at
        specified effective irradiance and cell temperature.

    References
    ----------
    .. [1] K. Sauer, T. Roessler, C. W. Hansen, Modeling the Irradiance and
       Temperature Dependence of Photovoltaic Modules in PVsyst,
       IEEE Journal of Photovoltaics v5(1), January 2015.

    .. [2] A. Mermoud, PV modules modelling, Presentation at the 2nd PV
       Performance Modeling Workshop, Santa Clara, CA, May 2013

    .. [3] A. Mermoud, T. Lejeune, Performance Assessment of a Simulation Model
       for PV modules of any available technology, 25th European Photovoltaic
       Solar Energy Conference, Valencia, Spain, Sept. 2010

    See Also
    --------
    calcparams_desoto
    singlediode

    '''

    # Boltzmann constant in J/K
    k = 1.38064852e-23

    # elementary charge in coulomb
    q = 1.6021766e-19

    # reference temperature
    Tref_K = temp_ref + 273.15
    Tcell_K = temp_cell + 273.15

    gamma = gamma_ref + mu_gamma * (Tcell_K - Tref_K)
    nNsVth = gamma * k / q * cells_in_series * Tcell_K

    IL = effective_irradiance / irrad_ref * \
        (I_L_ref + alpha_sc * (Tcell_K - Tref_K))

    I0 = I_o_ref * ((Tcell_K / Tref_K) ** 3) * \
        (np.exp((q * EgRef) / (k * gamma) * (1 / Tref_K - 1 / Tcell_K)))

    Rsh_tmp = \
        (R_sh_ref - R_sh_0 * np.exp(-R_sh_exp)) / (1.0 - np.exp(-R_sh_exp))
    Rsh_base = np.maximum(0.0, Rsh_tmp)

    Rsh = Rsh_base + (R_sh_0 - Rsh_base) * \
        np.exp(-R_sh_exp * effective_irradiance / irrad_ref)

    Rs = R_s

    return IL, I0, Rs, Rsh, nNsVth


def retrieve_sam(name=None, path=None):
    '''
    Retrieve latest module and inverter info from a local file or the
    SAM website.

    This function will retrieve either:

        * CEC module database
        * Sandia Module database
        * CEC Inverter database
        * Anton Driesse Inverter database

    and return it as a pandas DataFrame.

    Parameters
    ----------
    name : None or string, default None
        Name can be one of:

        * 'CECMod' - returns the CEC module database
        * 'CECInverter' - returns the CEC Inverter database
        * 'SandiaInverter' - returns the CEC Inverter database
          (CEC is only current inverter db available; tag kept for
          backwards compatibility)
        * 'SandiaMod' - returns the Sandia Module database
        * 'ADRInverter' - returns the ADR Inverter database

    path : None or string, default None
        Path to the SAM file. May also be a URL.

    Returns
    -------
    samfile : DataFrame
        A DataFrame containing all the elements of the desired database.
        Each column represents a module or inverter, and a specific
        dataset can be retrieved by the command

    Raises
    ------
    ValueError
        If no name or path is provided.

    Notes
    -----
    Files available at
        https://github.com/NREL/SAM/tree/develop/deploy/libraries
    Documentation for module and inverter data sets:
        https://sam.nrel.gov/photovoltaic/pv-sub-page-2.html

    Examples
    --------

    >>> from pvlib import pvsystem
    >>> invdb = pvsystem.retrieve_sam('CECInverter')
    >>> inverter = invdb.AE_Solar_Energy__AE6_0__277V__277V__CEC_2012_
    >>> inverter
    Vac           277.000000
    Paco         6000.000000
    Pdco         6165.670000
    Vdco          361.123000
    Pso            36.792300
    C0             -0.000002
    C1             -0.000047
    C2             -0.001861
    C3              0.000721
    Pnt             0.070000
    Vdcmax        600.000000
    Idcmax         32.000000
    Mppt_low      200.000000
    Mppt_high     500.000000
    Name: AE_Solar_Energy__AE6_0__277V__277V__CEC_2012_, dtype: float64
    '''

    if name is not None:
        name = name.lower()
        data_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'data')
        if name == 'cecmod':
            csvdata = os.path.join(
                data_path, 'sam-library-cec-modules-2019-03-05.csv')
        elif name == 'sandiamod':
            csvdata = os.path.join(
                data_path, 'sam-library-sandia-modules-2015-6-30.csv')
        elif name == 'adrinverter':
            csvdata = os.path.join(data_path, 'adr-library-2013-10-01.csv')
        elif name in ['cecinverter', 'sandiainverter']:
            # Allowing either, to provide for old code,
            # while aligning with current expectations
            csvdata = os.path.join(
                data_path, 'sam-library-cec-inverters-2019-03-05.csv')
        else:
            raise ValueError(f'invalid name {name}')
    elif path is not None:
        if path.startswith('http'):
            response = urlopen(path)
            csvdata = io.StringIO(response.read().decode(errors='ignore'))
        else:
            csvdata = path
    elif name is None and path is None:
        raise ValueError("A name or path must be provided!")

    return _parse_raw_sam_df(csvdata)


def _normalize_sam_product_names(names):
    '''
    Replace special characters within the product names to make them more
    suitable for use as Dataframe column names.
    '''
    # Contributed by Anton Driesse (@adriesse), PV Performance Labs. July, 2019

    import warnings

    BAD_CHARS = ' -.()[]:+/",'
    GOOD_CHARS = '____________'

    mapping = str.maketrans(BAD_CHARS, GOOD_CHARS)
    names = pd.Series(data=names)
    norm_names = names.str.translate(mapping)

    n_duplicates = names.duplicated().sum()
    if n_duplicates > 0:
        warnings.warn('Original names contain %d duplicate(s).' % n_duplicates)

    n_duplicates = norm_names.duplicated().sum()
    if n_duplicates > 0:
        warnings.warn(
            'Normalized names contain %d duplicate(s).' % n_duplicates)

    return norm_names.values


def _parse_raw_sam_df(csvdata):

    df = pd.read_csv(csvdata, index_col=0, skiprows=[1, 2])

    df.columns = df.columns.str.replace(' ', '_')
    df.index = _normalize_sam_product_names(df.index)
    df = df.transpose()

    if 'ADRCoefficients' in df.index:
        ad_ce = 'ADRCoefficients'
        # for each inverter, parses a string of coefficients like
        # ' 1.33, 2.11, 3.12' into a list containing floats:
        # [1.33, 2.11, 3.12]
        df.loc[ad_ce] = df.loc[ad_ce].map(lambda x: list(
            map(float, x.strip(' []').split())))

    return df


def sapm(effective_irradiance, temp_cell, module):
    '''
    The Sandia PV Array Performance Model (SAPM) generates 5 points on a
    PV module's I-V curve (Voc, Isc, Ix, Ixx, Vmp/Imp) according to
    SAND2004-3535. Assumes a reference cell temperature of 25 C.

    Parameters
    ----------
    effective_irradiance : numeric
        Irradiance reaching the module's cells, after reflections and
        adjustment for spectrum. [W/m2]

    temp_cell : numeric
        Cell temperature [C].

    module : dict-like
        A dict or Series defining the SAPM parameters. See the notes section
        for more details.

    Returns
    -------
    A DataFrame with the columns:

        * i_sc : Short-circuit current (A)
        * i_mp : Current at the maximum-power point (A)
        * v_oc : Open-circuit voltage (V)
        * v_mp : Voltage at maximum-power point (V)
        * p_mp : Power at maximum-power point (W)
        * i_x : Current at module V = 0.5Voc, defines 4th point on I-V
          curve for modeling curve shape
        * i_xx : Current at module V = 0.5(Voc+Vmp), defines 5th point on
          I-V curve for modeling curve shape

    Notes
    -----
    The SAPM parameters which are required in ``module`` are
    listed in the following table.

    The Sandia module database contains parameter values for a limited set
    of modules. The CEC module database does not contain these parameters.
    Both databases can be accessed using :py:func:`retrieve_sam`.

    ================   ========================================================
    Key                Description
    ================   ========================================================
    A0-A4              The airmass coefficients used in calculating
                       effective irradiance
    B0-B5              The angle of incidence coefficients used in calculating
                       effective irradiance
    C0-C7              The empirically determined coefficients relating
                       Imp, Vmp, Ix, and Ixx to effective irradiance
    Isco               Short circuit current at reference condition (amps)
    Impo               Maximum power current at reference condition (amps)
    Voco               Open circuit voltage at reference condition (amps)
    Vmpo               Maximum power voltage at reference condition (amps)
    Aisc               Short circuit current temperature coefficient at
                       reference condition (1/C)
    Aimp               Maximum power current temperature coefficient at
                       reference condition (1/C)
    Bvoco              Open circuit voltage temperature coefficient at
                       reference condition (V/C)
    Mbvoc              Coefficient providing the irradiance dependence for the
                       BetaVoc temperature coefficient at reference irradiance
                       (V/C)
    Bvmpo              Maximum power voltage temperature coefficient at
                       reference condition
    Mbvmp              Coefficient providing the irradiance dependence for the
                       BetaVmp temperature coefficient at reference irradiance
                       (V/C)
    N                  Empirically determined "diode factor" (dimensionless)
    Cells_in_Series    Number of cells in series in a module's cell string(s)
    IXO                Ix at reference conditions
    IXXO               Ixx at reference conditions
    FD                 Fraction of diffuse irradiance used by module
    ================   ========================================================

    References
    ----------
    .. [1] King, D. et al, 2004, "Sandia Photovoltaic Array Performance
       Model", SAND Report 3535, Sandia National Laboratories, Albuquerque,
       NM.

    See Also
    --------
    retrieve_sam
    pvlib.temperature.sapm_cell
    pvlib.temperature.sapm_module
    '''

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


def sapm_spectral_loss(airmass_absolute, module):
    """
    Calculates the SAPM spectral loss coefficient, F1.

    Parameters
    ----------
    airmass_absolute : numeric
        Absolute airmass

    module : dict-like
        A dict, Series, or DataFrame defining the SAPM performance
        parameters. See the :py:func:`sapm` notes section for more
        details.

    Returns
    -------
    F1 : numeric
        The SAPM spectral loss coefficient.

    Notes
    -----
    nan airmass values will result in 0 output.
    """

    am_coeff = [module['A4'], module['A3'], module['A2'], module['A1'],
                module['A0']]

    spectral_loss = np.polyval(am_coeff, airmass_absolute)

    spectral_loss = np.where(np.isnan(spectral_loss), 0, spectral_loss)

    spectral_loss = np.maximum(0, spectral_loss)

    if isinstance(airmass_absolute, pd.Series):
        spectral_loss = pd.Series(spectral_loss, airmass_absolute.index)

    return spectral_loss


def sapm_effective_irradiance(poa_direct, poa_diffuse, airmass_absolute, aoi,
                              module):
    r"""
    Calculates the SAPM effective irradiance using the SAPM spectral
    loss and SAPM angle of incidence loss functions.

    Parameters
    ----------
    poa_direct : numeric
        The direct irradiance incident upon the module. [W/m2]

    poa_diffuse : numeric
        The diffuse irradiance incident on module.  [W/m2]

    airmass_absolute : numeric
        Absolute airmass. [unitless]

    aoi : numeric
        Angle of incidence. [degrees]

    module : dict-like
        A dict, Series, or DataFrame defining the SAPM performance
        parameters. See the :py:func:`sapm` notes section for more
        details.

    Returns
    -------
    effective_irradiance : numeric
        Effective irradiance accounting for reflections and spectral content.
        [W/m2]

    Notes
    -----
    The SAPM model for effective irradiance [1]_ translates broadband direct
    and diffuse irradiance on the plane of array to the irradiance absorbed by
    a module's cells.

    The model is
    .. math::

        `Ee = f_1(AM_a) (E_b f_2(AOI) + f_d E_d)`

    where :math:`Ee` is effective irradiance (W/m2), :math:`f_1` is a fourth
    degree polynomial in air mass :math:`AM_a`, :math:`E_b` is beam (direct)
    irradiance on the plane of array, :math:`E_d` is diffuse irradiance on the
    plane of array, :math:`f_2` is a fifth degree polynomial in the angle of
    incidence :math:`AOI`, and :math:`f_d` is the fraction of diffuse
    irradiance on the plane of array that is not reflected away.

    References
    ----------
    .. [1] D. King et al, "Sandia Photovoltaic Array Performance Model",
       SAND2004-3535, Sandia National Laboratories, Albuquerque, NM

    See also
    --------
    pvlib.iam.sapm
    pvlib.pvsystem.sapm_spectral_loss
    pvlib.pvsystem.sapm
    """

    F1 = sapm_spectral_loss(airmass_absolute, module)
    F2 = iam.sapm(aoi, module)

    Ee = F1 * (poa_direct * F2 + module['FD'] * poa_diffuse)

    return Ee


def singlediode(photocurrent, saturation_current, resistance_series,
                resistance_shunt, nNsVth, ivcurve_pnts=None,
                method='lambertw'):
    r"""
    Solve the single-diode equation to obtain a photovoltaic IV curve.

    Solves the single diode equation [1]_

    .. math::

        I = I_L -
            I_0 \left[
                \exp \left(\frac{V+I R_s}{n N_s V_{th}} \right)-1
            \right] -
            \frac{V + I R_s}{R_{sh}}

    for :math:`I` and :math:`V` when given :math:`I_L, I_0, R_s, R_{sh},` and
    :math:`n N_s V_{th}` which are described later. Returns a DataFrame
    which contains the 5 points on the I-V curve specified in
    [3]_. If all :math:`I_L, I_0, R_s, R_{sh},` and
    :math:`n N_s V_{th}` are scalar, a single curve is returned, if any
    are Series (of the same length), multiple IV curves are calculated.

    The input parameters can be calculated from meteorological data using a
    function for a single diode model, e.g.,
    :py:func:`~pvlib.pvsystem.calcparams_desoto`.

    Parameters
    ----------
    photocurrent : numeric
        Light-generated current :math:`I_L` (photocurrent)
        ``0 <= photocurrent``. [A]

    saturation_current : numeric
        Diode saturation :math:`I_0` current under desired IV curve
        conditions. ``0 < saturation_current``. [A]

    resistance_series : numeric
        Series resistance :math:`R_s` under desired IV curve conditions.
        ``0 <= resistance_series < numpy.inf``.  [ohm]

    resistance_shunt : numeric
        Shunt resistance :math:`R_{sh}` under desired IV curve conditions.
        ``0 < resistance_shunt <= numpy.inf``.  [ohm]

    nNsVth : numeric
        The product of three components: 1) the usual diode ideality factor
        :math:`n`, 2) the number of cells in series :math:`N_s`, and 3)
        the cell thermal voltage
        :math:`V_{th}`. The thermal voltage of the cell (in volts) may be
        calculated as :math:`k_B T_c / q`, where :math:`k_B` is
        Boltzmann's constant (J/K), :math:`T_c` is the temperature of the p-n
        junction in Kelvin, and :math:`q` is the charge of an electron
        (coulombs). ``0 < nNsVth``.  [V]

    ivcurve_pnts : None or int, default None
        Number of points in the desired IV curve. If None or 0, no points on
        the IV curves will be produced.

    method : str, default 'lambertw'
        Determines the method used to calculate points on the IV curve. The
        options are ``'lambertw'``, ``'newton'``, or ``'brentq'``.

    Returns
    -------
    OrderedDict or DataFrame

    The returned dict-like object always contains the keys/columns:

        * i_sc - short circuit current in amperes.
        * v_oc - open circuit voltage in volts.
        * i_mp - current at maximum power point in amperes.
        * v_mp - voltage at maximum power point in volts.
        * p_mp - power at maximum power point in watts.
        * i_x - current, in amperes, at ``v = 0.5*v_oc``.
        * i_xx - current, in amperes, at ``V = 0.5*(v_oc+v_mp)``.

    If ivcurve_pnts is greater than 0, the output dictionary will also
    include the keys:

        * i - IV curve current in amperes.
        * v - IV curve voltage in volts.

    The output will be an OrderedDict if photocurrent is a scalar,
    array, or ivcurve_pnts is not None.

    The output will be a DataFrame if photocurrent is a Series and
    ivcurve_pnts is None.

    See also
    --------
    calcparams_desoto
    calcparams_cec
    calcparams_pvsyst
    sapm
    pvlib.singlediode.bishop88

    Notes
    -----
    If the method is ``'lambertw'`` then the solution employed to solve the
    implicit diode equation utilizes the Lambert W function to obtain an
    explicit function of :math:`V=f(I)` and :math:`I=f(V)` as shown in [2]_.

    If the method is ``'newton'`` then the root-finding Newton-Raphson method
    is used. It should be safe for well behaved IV-curves, but the ``'brentq'``
    method is recommended for reliability.

    If the method is ``'brentq'`` then Brent's bisection search method is used
    that guarantees convergence by bounding the voltage between zero and
    open-circuit.

    If the method is either ``'newton'`` or ``'brentq'`` and ``ivcurve_pnts``
    are indicated, then :func:`pvlib.singlediode.bishop88` [4]_ is used to
    calculate the points on the IV curve points at diode voltages from zero to
    open-circuit voltage with a log spacing that gets closer as voltage
    increases. If the method is ``'lambertw'`` then the calculated points on
    the IV curve are linearly spaced.

    References
    ----------
    .. [1] S.R. Wenham, M.A. Green, M.E. Watt, "Applied Photovoltaics" ISBN
       0 86758 909 4

    .. [2] A. Jain, A. Kapoor, "Exact analytical solutions of the
       parameters of real solar cells using Lambert W-function", Solar
       Energy Materials and Solar Cells, 81 (2004) 269-277.

    .. [3] D. King et al, "Sandia Photovoltaic Array Performance Model",
       SAND2004-3535, Sandia National Laboratories, Albuquerque, NM

    .. [4] "Computer simulation of the effects of electrical mismatches in
       photovoltaic cell interconnection circuits" JW Bishop, Solar Cell (1988)
       https://doi.org/10.1016/0379-6787(88)90059-2
    """
    # Calculate points on the IV curve using the LambertW solution to the
    # single diode equation
    if method.lower() == 'lambertw':
        out = _singlediode._lambertw(
            photocurrent, saturation_current, resistance_series,
            resistance_shunt, nNsVth, ivcurve_pnts
        )
        i_sc, v_oc, i_mp, v_mp, p_mp, i_x, i_xx = out[:7]
        if ivcurve_pnts:
            ivcurve_i, ivcurve_v = out[7:]
    else:
        # Calculate points on the IV curve using either 'newton' or 'brentq'
        # methods. Voltages are determined by first solving the single diode
        # equation for the diode voltage V_d then backing out voltage
        args = (photocurrent, saturation_current, resistance_series,
                resistance_shunt, nNsVth)  # collect args
        v_oc = _singlediode.bishop88_v_from_i(
            0.0, *args, method=method.lower()
        )
        i_mp, v_mp, p_mp = _singlediode.bishop88_mpp(
            *args, method=method.lower()
        )
        i_sc = _singlediode.bishop88_i_from_v(
            0.0, *args, method=method.lower()
        )
        i_x = _singlediode.bishop88_i_from_v(
            v_oc / 2.0, *args, method=method.lower()
        )
        i_xx = _singlediode.bishop88_i_from_v(
            (v_oc + v_mp) / 2.0, *args, method=method.lower()
        )

        # calculate the IV curve if requested using bishop88
        if ivcurve_pnts:
            vd = v_oc * (
                (11.0 - np.logspace(np.log10(11.0), 0.0, ivcurve_pnts)) / 10.0
            )
            ivcurve_i, ivcurve_v, _ = _singlediode.bishop88(vd, *args)

    out = OrderedDict()
    out['i_sc'] = i_sc
    out['v_oc'] = v_oc
    out['i_mp'] = i_mp
    out['v_mp'] = v_mp
    out['p_mp'] = p_mp
    out['i_x'] = i_x
    out['i_xx'] = i_xx

    if ivcurve_pnts:

        out['v'] = ivcurve_v
        out['i'] = ivcurve_i

    if isinstance(photocurrent, pd.Series) and not ivcurve_pnts:
        out = pd.DataFrame(out, index=photocurrent.index)

    return out


def max_power_point(photocurrent, saturation_current, resistance_series,
                    resistance_shunt, nNsVth, d2mutau=0, NsVbi=np.Inf,
                    method='brentq'):
    """
    Given the single diode equation coefficients, calculates the maximum power
    point (MPP).

    Parameters
    ----------
    photocurrent : numeric
        photo-generated current [A]
    saturation_current : numeric
        diode reverse saturation current [A]
    resistance_series : numeric
        series resitance [ohms]
    resistance_shunt : numeric
        shunt resitance [ohms]
    nNsVth : numeric
        product of thermal voltage ``Vth`` [V], diode ideality factor ``n``,
        and number of serices cells ``Ns``
    d2mutau : numeric, default 0
        PVsyst parameter for cadmium-telluride (CdTe) and amorphous-silicon
        (a-Si) modules that accounts for recombination current in the
        intrinsic layer. The value is the ratio of intrinsic layer thickness
        squared :math:`d^2` to the diffusion length of charge carriers
        :math:`\\mu \\tau`. [V]
    NsVbi : numeric, default np.inf
        PVsyst parameter for cadmium-telluride (CdTe) and amorphous-silicon
        (a-Si) modules that is the product of the PV module number of series
        cells ``Ns`` and the builtin voltage ``Vbi`` of the intrinsic layer.
        [V].
    method : str
        either ``'newton'`` or ``'brentq'``

    Returns
    -------
    OrderedDict or pandas.Datafrane
        ``(i_mp, v_mp, p_mp)``

    Notes
    -----
    Use this function when you only want to find the maximum power point. Use
    :func:`singlediode` when you need to find additional points on the IV
    curve. This function uses Brent's method by default because it is
    guaranteed to converge.
    """
    i_mp, v_mp, p_mp = _singlediode.bishop88_mpp(
        photocurrent, saturation_current, resistance_series,
        resistance_shunt, nNsVth, d2mutau=0, NsVbi=np.Inf,
        method=method.lower()
    )
    if isinstance(photocurrent, pd.Series):
        ivp = {'i_mp': i_mp, 'v_mp': v_mp, 'p_mp': p_mp}
        out = pd.DataFrame(ivp, index=photocurrent.index)
    else:
        out = OrderedDict()
        out['i_mp'] = i_mp
        out['v_mp'] = v_mp
        out['p_mp'] = p_mp
    return out


def v_from_i(resistance_shunt, resistance_series, nNsVth, current,
             saturation_current, photocurrent, method='lambertw'):
    '''
    Device voltage at the given device current for the single diode model.

    Uses the single diode model (SDM) as described in, e.g.,
    Jain and Kapoor 2004 [1]_.
    The solution is per Eq 3 of [1]_ except when resistance_shunt=numpy.inf,
    in which case the explict solution for voltage is used.
    Ideal device parameters are specified by resistance_shunt=np.inf and
    resistance_series=0.
    Inputs to this function can include scalars and pandas.Series, but it is
    the caller's responsibility to ensure that the arguments are all float64
    and within the proper ranges.

    Parameters
    ----------
    resistance_shunt : numeric
        Shunt resistance in ohms under desired IV curve conditions.
        Often abbreviated ``Rsh``.
        0 < resistance_shunt <= numpy.inf

    resistance_series : numeric
        Series resistance in ohms under desired IV curve conditions.
        Often abbreviated ``Rs``.
        0 <= resistance_series < numpy.inf

    nNsVth : numeric
        The product of three components. 1) The usual diode ideal factor
        (n), 2) the number of cells in series (Ns), and 3) the cell
        thermal voltage under the desired IV curve conditions (Vth). The
        thermal voltage of the cell (in volts) may be calculated as
        ``k*temp_cell/q``, where k is Boltzmann's constant (J/K),
        temp_cell is the temperature of the p-n junction in Kelvin, and
        q is the charge of an electron (coulombs).
        0 < nNsVth

    current : numeric
        The current in amperes under desired IV curve conditions.

    saturation_current : numeric
        Diode saturation current in amperes under desired IV curve
        conditions. Often abbreviated ``I_0``.
        0 < saturation_current

    photocurrent : numeric
        Light-generated current (photocurrent) in amperes under desired
        IV curve conditions. Often abbreviated ``I_L``.
        0 <= photocurrent

    method : str
        Method to use: ``'lambertw'``, ``'newton'``, or ``'brentq'``. *Note*:
        ``'brentq'`` is limited to 1st quadrant only.

    Returns
    -------
    current : np.ndarray or scalar

    References
    ----------
    .. [1] A. Jain, A. Kapoor, "Exact analytical solutions of the
       parameters of real solar cells using Lambert W-function", Solar
       Energy Materials and Solar Cells, 81 (2004) 269-277.
    '''
    if method.lower() == 'lambertw':
        return _singlediode._lambertw_v_from_i(
            resistance_shunt, resistance_series, nNsVth, current,
            saturation_current, photocurrent
        )
    else:
        # Calculate points on the IV curve using either 'newton' or 'brentq'
        # methods. Voltages are determined by first solving the single diode
        # equation for the diode voltage V_d then backing out voltage
        args = (current, photocurrent, saturation_current,
                resistance_series, resistance_shunt, nNsVth)
        V = _singlediode.bishop88_v_from_i(*args, method=method.lower())
        # find the right size and shape for returns
        size, shape = _singlediode._get_size_and_shape(args)
        if size <= 1:
            if shape is not None:
                V = np.tile(V, shape)
        if np.isnan(V).any() and size <= 1:
            V = np.repeat(V, size)
            if shape is not None:
                V = V.reshape(shape)
        return V


def i_from_v(resistance_shunt, resistance_series, nNsVth, voltage,
             saturation_current, photocurrent, method='lambertw'):
    '''
    Device current at the given device voltage for the single diode model.

    Uses the single diode model (SDM) as described in, e.g.,
     Jain and Kapoor 2004 [1]_.
    The solution is per Eq 2 of [1] except when resistance_series=0,
     in which case the explict solution for current is used.
    Ideal device parameters are specified by resistance_shunt=np.inf and
     resistance_series=0.
    Inputs to this function can include scalars and pandas.Series, but it is
     the caller's responsibility to ensure that the arguments are all float64
     and within the proper ranges.

    Parameters
    ----------
    resistance_shunt : numeric
        Shunt resistance in ohms under desired IV curve conditions.
        Often abbreviated ``Rsh``.
        0 < resistance_shunt <= numpy.inf

    resistance_series : numeric
        Series resistance in ohms under desired IV curve conditions.
        Often abbreviated ``Rs``.
        0 <= resistance_series < numpy.inf

    nNsVth : numeric
        The product of three components. 1) The usual diode ideal factor
        (n), 2) the number of cells in series (Ns), and 3) the cell
        thermal voltage under the desired IV curve conditions (Vth). The
        thermal voltage of the cell (in volts) may be calculated as
        ``k*temp_cell/q``, where k is Boltzmann's constant (J/K),
        temp_cell is the temperature of the p-n junction in Kelvin, and
        q is the charge of an electron (coulombs).
        0 < nNsVth

    voltage : numeric
        The voltage in Volts under desired IV curve conditions.

    saturation_current : numeric
        Diode saturation current in amperes under desired IV curve
        conditions. Often abbreviated ``I_0``.
        0 < saturation_current

    photocurrent : numeric
        Light-generated current (photocurrent) in amperes under desired
        IV curve conditions. Often abbreviated ``I_L``.
        0 <= photocurrent

    method : str
        Method to use: ``'lambertw'``, ``'newton'``, or ``'brentq'``. *Note*:
        ``'brentq'`` is limited to 1st quadrant only.

    Returns
    -------
    current : np.ndarray or scalar

    References
    ----------
    .. [1] A. Jain, A. Kapoor, "Exact analytical solutions of the
       parameters of real solar cells using Lambert W-function", Solar
       Energy Materials and Solar Cells, 81 (2004) 269-277.
    '''
    if method.lower() == 'lambertw':
        return _singlediode._lambertw_i_from_v(
            resistance_shunt, resistance_series, nNsVth, voltage,
            saturation_current, photocurrent
        )
    else:
        # Calculate points on the IV curve using either 'newton' or 'brentq'
        # methods. Voltages are determined by first solving the single diode
        # equation for the diode voltage V_d then backing out voltage
        args = (voltage, photocurrent, saturation_current, resistance_series,
                resistance_shunt, nNsVth)
        current = _singlediode.bishop88_i_from_v(*args, method=method.lower())
        # find the right size and shape for returns
        size, shape = _singlediode._get_size_and_shape(args)
        if size <= 1:
            if shape is not None:
                current = np.tile(current, shape)
        if np.isnan(current).any() and size <= 1:
            current = np.repeat(current, size)
            if shape is not None:
                current = current.reshape(shape)
        return current


def scale_voltage_current_power(data, voltage=1, current=1):
    """
    Scales the voltage, current, and power in data by the voltage
    and current factors.

    Parameters
    ----------
    data: DataFrame
        May contain columns `'v_mp', 'v_oc', 'i_mp' ,'i_x', 'i_xx',
        'i_sc', 'p_mp'`.
    voltage: numeric, default 1
        The amount by which to multiply the voltages.
    current: numeric, default 1
        The amount by which to multiply the currents.

    Returns
    -------
    scaled_data: DataFrame
        A scaled copy of the input data.
        `'p_mp'` is scaled by `voltage * current`.
    """

    # as written, only works with a DataFrame
    # could make it work with a dict, but it would be more verbose
    voltage_keys = ['v_mp', 'v_oc']
    current_keys = ['i_mp', 'i_x', 'i_xx', 'i_sc']
    power_keys = ['p_mp']
    voltage_df = data.filter(voltage_keys, axis=1) * voltage
    current_df = data.filter(current_keys, axis=1) * current
    power_df = data.filter(power_keys, axis=1) * voltage * current
    df = pd.concat([voltage_df, current_df, power_df], axis=1)
    df_sorted = df[data.columns]  # retain original column order
    return df_sorted


def pvwatts_dc(g_poa_effective, temp_cell, pdc0, gamma_pdc, temp_ref=25.):
    r"""
    Implements NREL's PVWatts DC power model. The PVWatts DC model [1]_ is:

    .. math::

        P_{dc} = \frac{G_{poa eff}}{1000} P_{dc0} ( 1 + \gamma_{pdc} (T_{cell} - T_{ref}))

    Note that the pdc0 is also used as a symbol in
    :py:func:`pvlib.inverter.pvwatts`. pdc0 in this function refers to the DC
    power of the modules at reference conditions. pdc0 in
    :py:func:`pvlib.inverter.pvwatts` refers to the DC power input limit of
    the inverter.

    Parameters
    ----------
    g_poa_effective: numeric
        Irradiance transmitted to the PV cells. To be
        fully consistent with PVWatts, the user must have already
        applied angle of incidence losses, but not soiling, spectral,
        etc. [W/m^2]
    temp_cell: numeric
        Cell temperature [C].
    pdc0: numeric
        Power of the modules at 1000 W/m^2 and cell reference temperature. [W]
    gamma_pdc: numeric
        The temperature coefficient of power. Typically -0.002 to
        -0.005 per degree C. [1/C]
    temp_ref: numeric, default 25.0
        Cell reference temperature. PVWatts defines it to be 25 C and
        is included here for flexibility. [C]

    Returns
    -------
    pdc: numeric
        DC power.

    References
    ----------
    .. [1] A. P. Dobos, "PVWatts Version 5 Manual"
           http://pvwatts.nrel.gov/downloads/pvwattsv5.pdf
           (2014).
    """  # noqa: E501

    pdc = (g_poa_effective * 0.001 * pdc0 *
           (1 + gamma_pdc * (temp_cell - temp_ref)))

    return pdc


def pvwatts_losses(soiling=2, shading=3, snow=0, mismatch=2, wiring=2,
                   connections=0.5, lid=1.5, nameplate_rating=1, age=0,
                   availability=3):
    r"""
    Implements NREL's PVWatts system loss model.
    The PVWatts loss model [1]_ is:

    .. math::

        L_{total}(\%) = 100 [ 1 - \Pi_i ( 1 - \frac{L_i}{100} ) ]

    All parameters must be in units of %. Parameters may be
    array-like, though all array sizes must match.

    Parameters
    ----------
    soiling: numeric, default 2
    shading: numeric, default 3
    snow: numeric, default 0
    mismatch: numeric, default 2
    wiring: numeric, default 2
    connections: numeric, default 0.5
    lid: numeric, default 1.5
        Light induced degradation
    nameplate_rating: numeric, default 1
    age: numeric, default 0
    availability: numeric, default 3

    Returns
    -------
    losses: numeric
        System losses in units of %.

    References
    ----------
    .. [1] A. P. Dobos, "PVWatts Version 5 Manual"
           http://pvwatts.nrel.gov/downloads/pvwattsv5.pdf
           (2014).
    """

    params = [soiling, shading, snow, mismatch, wiring, connections, lid,
              nameplate_rating, age, availability]

    # manually looping over params allows for numpy/pandas to handle any
    # array-like broadcasting that might be necessary.
    perf = 1
    for param in params:
        perf *= 1 - param/100

    losses = (1 - perf) * 100.

    return losses


def dc_ohms_from_percent(vmp_ref, imp_ref, dc_ohmic_percent,
                         modules_per_string=1,
                         strings=1):
    """
    Calculates the equivalent resistance of the wires from a percent
    ohmic loss at STC.

    Equivalent resistance is calculated with the function:

    .. math::
        Rw = (L_{stc} / 100) * (Varray / Iarray)

    :math:`Rw` is the equivalent resistance in ohms
    :math:`Varray` is the Vmp of the modules times modules per string
    :math:`Iarray` is the Imp of the modules times strings per array
    :math:`L_{stc}` is the input dc loss percent

    Parameters
    ----------
    vmp_ref: numeric
        Voltage at maximum power in reference conditions [V]
    imp_ref: numeric
        Current at maximum power in reference conditions [V]
    dc_ohmic_percent: numeric, default 0
        input dc loss as a percent, e.g. 1.5% loss is input as 1.5
    modules_per_string: int, default 1
        Number of modules per string in the array.
    strings: int, default 1
        Number of parallel strings in the array.

    Returns
    ----------
    Rw: numeric
        Equivalent resistance [ohm]

    See Also
    --------
    pvlib.pvsystem.dc_ohmic_losses

    References
    ----------
    .. [1] PVsyst 7 Help. "Array ohmic wiring loss".
       https://www.pvsyst.com/help/ohmic_loss.htm
    """
    vmp = modules_per_string * vmp_ref

    imp = strings * imp_ref

    Rw = (dc_ohmic_percent / 100) * (vmp / imp)

    return Rw


def dc_ohmic_losses(resistance, current):
    """
    Returns ohmic losses in units of power from the equivalent
    resistance of the wires and the operating current.

    Parameters
    ----------
    resistance: numeric
        Equivalent resistance of wires [ohm]
    current: numeric, float or array-like
        Operating current [A]

    Returns
    ----------
    loss: numeric
        Power Loss [W]

    See Also
    --------
    pvlib.pvsystem.dc_ohms_from_percent

    References
    ----------
    .. [1] PVsyst 7 Help. "Array ohmic wiring loss".
       https://www.pvsyst.com/help/ohmic_loss.htm
    """
    return resistance * current * current


def combine_loss_factors(index, *losses, fill_method='ffill'):
    r"""
    Combines Series loss fractions while setting a common index.

    The separate losses are compounded using the following equation:

    .. math::

        L_{total} = 1 - [ 1 - \Pi_i ( 1 - L_i ) ]

    :math:`L_{total}` is the total loss returned
    :math:`L_i` is each individual loss factor input

    Note the losses must each be a series with a DatetimeIndex.
    All losses will be resampled to match the index parameter using
    the fill method specified (defaults to "fill forward").

    Parameters
    ----------
    index : DatetimeIndex
        The index of the returned loss factors

    *losses : Series
        One or more Series of fractions to be compounded

    fill_method : {'ffill', 'bfill', 'nearest'}, default 'ffill'
        Method to use for filling holes in reindexed DataFrame

    Returns
    -------
    Series
        Fractions resulting from the combination of each loss factor
    """
    combined_factor = 1

    for loss in losses:
        loss = loss.reindex(index, method=fill_method)
        combined_factor *= (1 - loss)

    return 1 - combined_factor

```
