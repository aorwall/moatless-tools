# pvlib__pvlib-python-1072

| **pvlib/pvlib-python** | `04a523fafbd61bc2e49420963b84ed8e2bd1b3cf` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 1493 |
| **Any found context length** | 1493 |
| **Avg pos** | 2.0 |
| **Min pos** | 2 |
| **Max pos** | 2 |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/pvlib/temperature.py b/pvlib/temperature.py
--- a/pvlib/temperature.py
+++ b/pvlib/temperature.py
@@ -599,8 +599,9 @@ def fuentes(poa_global, temp_air, wind_speed, noct_installed, module_height=5,
     # n.b. the way Fuentes calculates the first timedelta makes it seem like
     # the value doesn't matter -- rather than recreate it here, just assume
     # it's the same as the second timedelta:
-    timedelta_hours = np.diff(poa_global.index).astype(float) / 1e9 / 60 / 60
-    timedelta_hours = np.append([timedelta_hours[0]], timedelta_hours)
+    timedelta_seconds = poa_global.index.to_series().diff().dt.total_seconds()
+    timedelta_hours = timedelta_seconds / 3600
+    timedelta_hours.iloc[0] = timedelta_hours.iloc[1]
 
     tamb_array = temp_air + 273.15
     sun_array = poa_global * absorp

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| pvlib/temperature.py | 602 | 604 | 2 | 1 | 1493


## Problem Statement

```
temperature.fuentes errors when given tz-aware inputs on pandas>=1.0.0
**Describe the bug**
When the weather timeseries inputs to `temperature.fuentes` have tz-aware index, an internal call to `np.diff(index)` returns an array of `Timedelta` objects instead of an array of nanosecond ints, throwing an error immediately after.  The error only happens when using pandas>=1.0.0; using 0.25.3 runs successfully, but emits the warning:

\`\`\`
  /home/kevin/anaconda3/envs/pvlib-dev/lib/python3.7/site-packages/numpy/lib/function_base.py:1243: FutureWarning: Converting timezone-aware DatetimeArray to timezone-naive ndarray with 'datetime64[ns]' dtype. In the future, this will return an ndarray with 'object' dtype where each element is a 'pandas.Timestamp' with the correct 'tz'.
  	To accept the future behavior, pass 'dtype=object'.
  	To keep the old behavior, pass 'dtype="datetime64[ns]"'.
    a = asanyarray(a)
\`\`\`

**To Reproduce**
\`\`\`python
In [1]: import pvlib
   ...: import pandas as pd
   ...: 
   ...: index_naive = pd.date_range('2019-01-01', freq='h', periods=3)
   ...: 
   ...: kwargs = {
   ...:     'poa_global': pd.Series(1000, index_naive),
   ...:     'temp_air': pd.Series(20, index_naive),
   ...:     'wind_speed': pd.Series(1, index_naive),
   ...:     'noct_installed': 45
   ...: }
   ...: 

In [2]: print(pvlib.temperature.fuentes(**kwargs))
2019-01-01 00:00:00    47.85
2019-01-01 01:00:00    50.85
2019-01-01 02:00:00    50.85
Freq: H, Name: tmod, dtype: float64

In [3]: kwargs['poa_global'].index = index_naive.tz_localize('UTC')
   ...: print(pvlib.temperature.fuentes(**kwargs))
   ...: 
Traceback (most recent call last):

  File "<ipython-input-3-ff99badadc91>", line 2, in <module>
    print(pvlib.temperature.fuentes(**kwargs))

  File "/home/kevin/anaconda3/lib/python3.7/site-packages/pvlib/temperature.py", line 602, in fuentes
    timedelta_hours = np.diff(poa_global.index).astype(float) / 1e9 / 60 / 60

TypeError: float() argument must be a string or a number, not 'Timedelta'
\`\`\`

**Expected behavior**
`temperature.fuentes` should work with both tz-naive and tz-aware inputs.


**Versions:**
 - ``pvlib.__version__``: 0.8.0
 - ``pandas.__version__``: 1.0.0+
 - python: 3.7.4 (default, Aug 13 2019, 20:35:49) \n[GCC 7.3.0]



```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- |
| 1 | **1 pvlib/temperature.py** | 608 | 659| 679 | 679 | 
| **-> 2 <-** | **1 pvlib/temperature.py** | 553 | 607| 814 | 1493 | 
| 3 | 2 benchmarks/benchmarks/temperature.py | 18 | 15| 368 | 1861 | 
| 4 | **2 pvlib/temperature.py** | 476 | 552| 767 | 2628 | 
| 5 | 3 pvlib/bifacial.py | 84 | 128| 486 | 3114 | 
| 6 | 4 pvlib/solarposition.py | 478 | 501| 199 | 3313 | 
| 7 | **4 pvlib/temperature.py** | 377 | 442| 688 | 4001 | 
| 8 | 5 pvlib/iotools/tmy.py | 10 | 158| 2526 | 6527 | 
| 9 | 5 pvlib/iotools/tmy.py | 252 | 380| 2061 | 8588 | 
| 10 | 6 pvlib/tools.py | 86 | 110| 158 | 8746 | 
| 11 | 7 pvlib/forecast.py | 0 | 20| 157 | 8903 | 
| 12 | **7 pvlib/temperature.py** | 445 | 468| 423 | 9326 | 
| 13 | 8 pvlib/iotools/epw.py | 10 | 217| 2136 | 11462 | 
| 14 | 8 pvlib/bifacial.py | 139 | 151| 207 | 11669 | 
| 15 | 9 pvlib/irradiance.py | 115 | 148| 423 | 12092 | 
| 16 | 10 pvlib/__init__.py | 0 | 18| 230 | 12322 | 
| 17 | 10 pvlib/solarposition.py | 1399 | 1418| 206 | 12528 | 
| 18 | 10 pvlib/forecast.py | 166 | 183| 136 | 12664 | 


## Patch

```diff
diff --git a/pvlib/temperature.py b/pvlib/temperature.py
--- a/pvlib/temperature.py
+++ b/pvlib/temperature.py
@@ -599,8 +599,9 @@ def fuentes(poa_global, temp_air, wind_speed, noct_installed, module_height=5,
     # n.b. the way Fuentes calculates the first timedelta makes it seem like
     # the value doesn't matter -- rather than recreate it here, just assume
     # it's the same as the second timedelta:
-    timedelta_hours = np.diff(poa_global.index).astype(float) / 1e9 / 60 / 60
-    timedelta_hours = np.append([timedelta_hours[0]], timedelta_hours)
+    timedelta_seconds = poa_global.index.to_series().diff().dt.total_seconds()
+    timedelta_hours = timedelta_seconds / 3600
+    timedelta_hours.iloc[0] = timedelta_hours.iloc[1]
 
     tamb_array = temp_air + 273.15
     sun_array = poa_global * absorp

```

## Test Patch

```diff
diff --git a/pvlib/tests/test_temperature.py b/pvlib/tests/test_temperature.py
--- a/pvlib/tests/test_temperature.py
+++ b/pvlib/tests/test_temperature.py
@@ -190,3 +190,17 @@ def test_fuentes(filename, inoct):
     night_difference = expected_tcell[is_night] - actual_tcell[is_night]
     assert night_difference.max() < 6
     assert night_difference.min() > 0
+
+
+@pytest.mark.parametrize('tz', [None, 'Etc/GMT+5'])
+def test_fuentes_timezone(tz):
+    index = pd.date_range('2019-01-01', freq='h', periods=3, tz=tz)
+
+    df = pd.DataFrame({'poa_global': 1000, 'temp_air': 20, 'wind_speed': 1},
+                      index)
+
+    out = temperature.fuentes(df['poa_global'], df['temp_air'],
+                              df['wind_speed'], noct_installed=45)
+
+    assert_series_equal(out, pd.Series([47.85, 50.85, 50.85], index=index,
+                                       name='tmod'))

```


## Code snippets

### 1 - pvlib/temperature.py:

Start line: 608, End line: 659

```python
def fuentes(poa_global, temp_air, wind_speed, noct_installed, module_height=5,
            wind_height=9.144, emissivity=0.84, absorption=0.83,
            surface_tilt=30, module_width=0.31579, module_length=1.2):
    # sky temperature -- Equation 24
    tsky_array = 0.68 * (0.0552 * tamb_array**1.5) + 0.32 * tamb_array
    # wind speed at module height -- Equation 22
    # not sure why the 1e-4 factor is included -- maybe the equations don't
    # behave well if wind == 0?
    windmod_array = wind_speed * (module_height/wind_height)**0.2 + 1e-4

    tmod0 = 293.15
    tmod_array = np.zeros_like(poa_global)

    iterator = zip(tamb_array, sun_array, windmod_array, tsky_array,
                   timedelta_hours)
    for i, (tamb, sun, windmod, tsky, dtime) in enumerate(iterator):
        # solve the heat transfer equation, iterating because the heat loss
        # terms depend on tmod. NB Fuentes doesn't show that 10 iterations is
        # sufficient for convergence.
        tmod = tmod0
        for j in range(10):
            # overall convective coefficient
            tave = (tmod + tamb) / 2
            hconv = convrat * _fuentes_hconv(tave, windmod, tinoct,
                                             abs(tmod-tamb), xlen,
                                             surface_tilt, True)
            # sky radiation coefficient (Equation 3)
            hsky = emiss * boltz * (tmod**2 + tsky**2) * (tmod + tsky)
            # ground radiation coeffieicient (Equation 4)
            tground = tamb + tgrat * (tmod - tamb)
            hground = emiss * boltz * (tmod**2 + tground**2) * (tmod + tground)
            # thermal lag -- Equation 8
            eigen = - (hconv + hsky + hground) / cap * dtime * 3600
            # not sure why this check is done, maybe as a speed optimization?
            if eigen > -10:
                ex = np.exp(eigen)
            else:
                ex = 0
            # Equation 7 -- note that `sun` and `sun0` already account for
            # absorption (alpha)
            tmod = tmod0 * ex + (
                (1 - ex) * (
                    hconv * tamb
                    + hsky * tsky
                    + hground * tground
                    + sun0
                    + (sun - sun0) / eigen
                ) + sun - sun0
            ) / (hconv + hsky + hground)
        tmod_array[i] = tmod
        tmod0 = tmod
        sun0 = sun

    return pd.Series(tmod_array - 273.15, index=poa_global.index, name='tmod')
```
### 2 - pvlib/temperature.py:

Start line: 553, End line: 607

```python
def fuentes(poa_global, temp_air, wind_speed, noct_installed, module_height=5,
            wind_height=9.144, emissivity=0.84, absorption=0.83,
            surface_tilt=30, module_width=0.31579, module_length=1.2):
    # ported from the FORTRAN77 code provided in Appendix A of Fuentes 1987;
    # nearly all variable names are kept the same for ease of comparison.

    boltz = 5.669e-8
    emiss = emissivity
    absorp = absorption
    xlen = _hydraulic_diameter(module_width, module_length)
    # cap0 has units of [J / (m^2 K)], equal to mass per unit area times
    # specific heat of the module.
    cap0 = 11000
    tinoct = noct_installed + 273.15

    # convective coefficient of top surface of module at NOCT
    windmod = 1.0
    tave = (tinoct + 293.15) / 2
    hconv = _fuentes_hconv(tave, windmod, tinoct, tinoct - 293.15, xlen,
                           surface_tilt, False)

    # determine the ground temperature ratio and the ratio of the total
    # convection to the top side convection
    hground = emiss * boltz * (tinoct**2 + 293.15**2) * (tinoct + 293.15)
    backrat = (
        absorp * 800.0
        - emiss * boltz * (tinoct**4 - 282.21**4)
        - hconv * (tinoct - 293.15)
    ) / ((hground + hconv) * (tinoct - 293.15))
    tground = (tinoct**4 - backrat * (tinoct**4 - 293.15**4))**0.25
    tground = np.clip(tground, 293.15, tinoct)

    tgrat = (tground - 293.15) / (tinoct - 293.15)
    convrat = (absorp * 800 - emiss * boltz * (
        2 * tinoct**4 - 282.21**4 - tground**4)) / (hconv * (tinoct - 293.15))

    # adjust the capacitance (thermal mass) of the module based on the INOCT.
    # It is a function of INOCT because high INOCT implies thermal coupling
    # with the racking (e.g. roofmount), so the thermal mass is increased.
    # `cap` has units J/(m^2 C) -- see Table 3, Equations 26 & 27
    cap = cap0
    if tinoct > 321.15:
        cap = cap * (1 + (tinoct - 321.15) / 12)

    # iterate through timeseries inputs
    sun0 = 0
    tmod0 = 293.15

    # n.b. the way Fuentes calculates the first timedelta makes it seem like
    # the value doesn't matter -- rather than recreate it here, just assume
    # it's the same as the second timedelta:
    timedelta_hours = np.diff(poa_global.index).astype(float) / 1e9 / 60 / 60
    timedelta_hours = np.append([timedelta_hours[0]], timedelta_hours)

    tamb_array = temp_air + 273.15
    sun_array = poa_global * absorp

    # Two of the calculations are easily vectorized, so precalculate them:
    # ... other code
```
### 3 - benchmarks/benchmarks/temperature.py:

Start line: 18, End line: 15

```python
"""
ASV benchmarks for irradiance.py
"""

import pandas as pd
import pvlib
from pkg_resources import parse_version
from functools import partial


def set_weather_data(obj):
    obj.times = pd.date_range(start='20180601', freq='1min',
                              periods=14400)
    obj.poa = pd.Series(1000, index=obj.times)
    obj.tamb = pd.Series(20, index=obj.times)
    obj.wind_speed = pd.Series(2, index=obj.times)


class SAPM:

    def setup(self):
        set_weather_data(self)
        if parse_version(pvlib.__version__) >= parse_version('0.7.0'):
            kwargs = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']
            kwargs = kwargs['open_rack_glass_glass']
            self.sapm_cell_wrapper = partial(pvlib.temperature.sapm_cell,
                                             **kwargs)
        else:
            sapm_celltemp = pvlib.pvsystem.sapm_celltemp

            def sapm_cell_wrapper(poa_global, temp_air, wind_speed):
                # just swap order; model params are provided by default
                return sapm_celltemp(poa_global, wind_speed, temp_air)
            self.sapm_cell_wrapper = sapm_cell_wrapper

    def time_sapm_cell(self):
        # use version-appropriate wrapper
        self.sapm_cell_wrapper(self.poa, self.tamb, self.wind_speed)


class Fuentes:

    def setup(self):
        if parse_version(pvlib.__version__) < parse_version('0.8.0'):
            raise NotImplementedError

        set_weather_data(self)

    def time_fuentes(self):
        pvlib.temperature.fuentes(self.poa, self.tamb, self.wind_speed,
                                  noct_installed=45)
```
### 4 - pvlib/temperature.py:

Start line: 476, End line: 552

```python
def fuentes(poa_global, temp_air, wind_speed, noct_installed, module_height=5,
            wind_height=9.144, emissivity=0.84, absorption=0.83,
            surface_tilt=30, module_width=0.31579, module_length=1.2):
    """
    Calculate cell or module temperature using the Fuentes model.

    The Fuentes model is a first-principles heat transfer energy balance
    model [1]_ that is used in PVWatts for cell temperature modeling [2]_.

    Parameters
    ----------
    poa_global : pandas Series
        Total incident irradiance [W/m^2]

    temp_air : pandas Series
        Ambient dry bulb temperature [C]

    wind_speed : pandas Series
        Wind speed [m/s]

    noct_installed : float
        The "installed" nominal operating cell temperature as defined in [1]_.
        PVWatts assumes this value to be 45 C for rack-mounted arrays and
        49 C for roof mount systems with restricted air flow around the
        module.  [C]

    module_height : float, default 5.0
        The height above ground of the center of the module. The PVWatts
        default is 5.0 [m]

    wind_height : float, default 9.144
        The height above ground at which ``wind_speed`` is measured. The
        PVWatts defauls is 9.144 [m]

    emissivity : float, default 0.84
        The effectiveness of the module at radiating thermal energy. [unitless]

    absorption : float, default 0.83
        The fraction of incident irradiance that is converted to thermal
        energy in the module. [unitless]

    surface_tilt : float, default 30
        Module tilt from horizontal. If not provided, the default value
        of 30 degrees from [1]_ and [2]_ is used. [degrees]

    module_width : float, default 0.31579
        Module width. The default value of 0.31579 meters in combination with
        the default `module_length` gives a hydraulic diameter of 0.5 as
        assumed in [1]_ and [2]_. [m]

    module_length : float, default 1.2
        Module length. The default value of 1.2 meters in combination with
        the default `module_width` gives a hydraulic diameter of 0.5 as
        assumed in [1]_ and [2]_. [m]

    Returns
    -------
    temperature_cell : pandas Series
        The modeled cell temperature [C]

    Notes
    -----
    This function returns slightly different values from PVWatts at night
    and just after dawn. This is because the SAM SSC assumes that module
    temperature equals ambient temperature when irradiance is zero so it can
    skip the heat balance calculation at night.

    References
    ----------
    .. [1] Fuentes, M. K., 1987, "A Simplifed Thermal Model for Flat-Plate
           Photovoltaic Arrays", SAND85-0330, Sandia National Laboratories,
           Albuquerque NM.
           http://prod.sandia.gov/techlib/access-control.cgi/1985/850330.pdf
    .. [2] Dobos, A. P., 2014, "PVWatts Version 5 Manual", NREL/TP-6A20-62641,
           National Renewable Energy Laboratory, Golden CO.
           doi:10.2172/1158421.
    """
    # ... other code
```
### 5 - pvlib/bifacial.py:

Start line: 84, End line: 128

```python
def pvfactors_timeseries(
        solar_azimuth, solar_zenith, surface_azimuth, surface_tilt,
        axis_azimuth, timestamps, dni, dhi, gcr, pvrow_height, pvrow_width,
        albedo, n_pvrows=3, index_observed_pvrow=1,
        rho_front_pvrow=0.03, rho_back_pvrow=0.05,
        horizon_band_angle=15.):
    # Convert pandas Series inputs (and some lists) to numpy arrays
    if isinstance(solar_azimuth, pd.Series):
        solar_azimuth = solar_azimuth.values
    elif isinstance(solar_azimuth, list):
        solar_azimuth = np.array(solar_azimuth)
    if isinstance(solar_zenith, pd.Series):
        solar_zenith = solar_zenith.values
    elif isinstance(solar_zenith, list):
        solar_zenith = np.array(solar_zenith)
    if isinstance(surface_azimuth, pd.Series):
        surface_azimuth = surface_azimuth.values
    elif isinstance(surface_azimuth, list):
        surface_azimuth = np.array(surface_azimuth)
    if isinstance(surface_tilt, pd.Series):
        surface_tilt = surface_tilt.values
    elif isinstance(surface_tilt, list):
        surface_tilt = np.array(surface_tilt)
    if isinstance(dni, pd.Series):
        dni = dni.values
    elif isinstance(dni, list):
        dni = np.array(dni)
    if isinstance(dhi, pd.Series):
        dhi = dhi.values
    elif isinstance(dhi, list):
        dhi = np.array(dhi)

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
### 6 - pvlib/solarposition.py:

Start line: 478, End line: 501

```python
def _ephem_to_timezone(date, tzinfo):
    # utility from unreleased PyEphem 3.6.7.1
    """"Convert a PyEphem Date into a timezone aware python datetime"""
    seconds, microseconds = _ephem_convert_to_seconds_and_microseconds(date)
    date = dt.datetime.fromtimestamp(seconds, tzinfo)
    date = date.replace(microsecond=microseconds)
    return date


def _ephem_setup(latitude, longitude, altitude, pressure, temperature,
                 horizon):
    import ephem
    # initialize a PyEphem observer
    obs = ephem.Observer()
    obs.lat = str(latitude)
    obs.lon = str(longitude)
    obs.elevation = altitude
    obs.pressure = pressure / 100.  # convert to mBar
    obs.temp = temperature
    obs.horizon = horizon

    # the PyEphem sun
    sun = ephem.Sun()
    return obs, sun
```
### 7 - pvlib/temperature.py:

Start line: 377, End line: 442

```python
def faiman(poa_global, temp_air, wind_speed=1.0, u0=25.0, u1=6.84):
    r'''
    Calculate cell or module temperature using the Faiman model.  The Faiman
    model uses an empirical heat loss factor model [1]_ and is adopted in the
    IEC 61853 standards [2]_ and [3]_.

    Usage of this model in the IEC 61853 standard does not distinguish
    between cell and module temperature.

    Parameters
    ----------
    poa_global : numeric
        Total incident irradiance [W/m^2].

    temp_air : numeric
        Ambient dry bulb temperature [C].

    wind_speed : numeric, default 1.0
        Wind speed in m/s measured at the same height for which the wind loss
        factor was determined.  The default value 1.0 m/s is the wind
        speed at module height used to determine NOCT. [m/s]

    u0 : numeric, default 25.0
        Combined heat loss factor coefficient. The default value is one
        determined by Faiman for 7 silicon modules.
        :math:`\left[\frac{\text{W}/{\text{m}^2}}{\text{C}}\right]`

    u1 : numeric, default 6.84
        Combined heat loss factor influenced by wind. The default value is one
        determined by Faiman for 7 silicon modules.
        :math:`\left[ \frac{\text{W}/\text{m}^2}{\text{C}\ \left( \text{m/s} \right)} \right]`

    Returns
    -------
    numeric, values in degrees Celsius

    Notes
    -----
    All arguments may be scalars or vectors. If multiple arguments
    are vectors they must be the same length.

    References
    ----------
    .. [1] Faiman, D. (2008). "Assessing the outdoor operating temperature of
       photovoltaic modules." Progress in Photovoltaics 16(4): 307-315.

    .. [2] "IEC 61853-2 Photovoltaic (PV) module performance testing and energy
       rating - Part 2: Spectral responsivity, incidence angle and module
       operating temperature measurements". IEC, Geneva, 2018.

    .. [3] "IEC 61853-3 Photovoltaic (PV) module performance testing and energy
       rating - Part 3: Energy rating of PV modules". IEC, Geneva, 2018.

    '''
    # Contributed by Anton Driesse (@adriesse), PV Performance Labs. Dec., 2019

    # The following lines may seem odd since u0 & u1 are probably scalar,
    # but it serves an indirect and easy way of allowing lists and
    # tuples for the other function arguments.
    u0 = np.asanyarray(u0)
    u1 = np.asanyarray(u1)

    total_loss_factor = u0 + u1 * wind_speed
    heat_input = poa_global
    temp_difference = heat_input / total_loss_factor
    return temp_air + temp_difference
```
### 8 - pvlib/iotools/tmy.py:

Start line: 10, End line: 158

```python
def read_tmy3(filename, coerce_year=None, recolumn=True):
    '''
    Read a TMY3 file in to a pandas dataframe.

    Note that values contained in the metadata dictionary are unchanged
    from the TMY3 file (i.e. units are retained). In the case of any
    discrepancies between this documentation and the TMY3 User's Manual
    [1]_, the TMY3 User's Manual takes precedence.

    The TMY3 files were updated in Jan. 2015. This function requires the
    use of the updated files.

    Parameters
    ----------
    filename : str
        A relative file path or absolute file path.

    coerce_year : None or int, default None
        If supplied, the year of the index will be set to `coerce_year`, except
        for the last index value which will be set to the *next* year so that
        the index increases monotonically.

    recolumn : bool, default True
        If ``True``, apply standard names to TMY3 columns. Typically this
        results in stripping the units from the column name.

    Returns
    -------
    Tuple of the form (data, metadata).

    data : DataFrame
        A pandas dataframe with the columns described in the table
        below. For more detailed descriptions of each component, please
        consult the TMY3 User's Manual ([1]), especially tables 1-1
        through 1-6.

    metadata : dict
        The site metadata available in the file.

    Notes
    -----
    The returned structures have the following fields.

    ===============   ======  ===================
    key               format  description
    ===============   ======  ===================
    altitude          Float   site elevation
    latitude          Float   site latitudeitude
    longitude         Float   site longitudeitude
    Name              String  site name
    State             String  state
    TZ                Float   UTC offset
    USAF              Int     USAF identifier
    ===============   ======  ===================

    =============================       ======================================================================================================================================================
    TMYData field                       description
    =============================       ======================================================================================================================================================
    TMYData.Index                       A pandas datetime index. NOTE, the index is currently timezone unaware, and times are set to local standard time (daylight savings is not included)
    TMYData.ETR                         Extraterrestrial horizontal radiation recv'd during 60 minutes prior to timestamp, Wh/m^2
    TMYData.ETRN                        Extraterrestrial normal radiation recv'd during 60 minutes prior to timestamp, Wh/m^2
    TMYData.GHI                         Direct and diffuse horizontal radiation recv'd during 60 minutes prior to timestamp, Wh/m^2
    TMYData.GHISource                   See [1]_, Table 1-4
    TMYData.GHIUncertainty              Uncertainty based on random and bias error estimates                        see [2]_
    TMYData.DNI                         Amount of direct normal radiation (modeled) recv'd during 60 mintues prior to timestamp, Wh/m^2
    TMYData.DNISource                   See [1]_, Table 1-4
    TMYData.DNIUncertainty              Uncertainty based on random and bias error estimates                        see [2]_
    TMYData.DHI                         Amount of diffuse horizontal radiation recv'd during 60 minutes prior to timestamp, Wh/m^2
    TMYData.DHISource                   See [1]_, Table 1-4
    TMYData.DHIUncertainty              Uncertainty based on random and bias error estimates                        see [2]_
    TMYData.GHillum                     Avg. total horizontal illuminance recv'd during the 60 minutes prior to timestamp, lx
    TMYData.GHillumSource               See [1]_, Table 1-4
    TMYData.GHillumUncertainty          Uncertainty based on random and bias error estimates                        see [2]_
    TMYData.DNillum                     Avg. direct normal illuminance recv'd during the 60 minutes prior to timestamp, lx
    TMYData.DNillumSource               See [1]_, Table 1-4
    TMYData.DNillumUncertainty          Uncertainty based on random and bias error estimates                        see [2]_
    TMYData.DHillum                     Avg. horizontal diffuse illuminance recv'd during the 60 minutes prior to timestamp, lx
    TMYData.DHillumSource               See [1]_, Table 1-4
    TMYData.DHillumUncertainty          Uncertainty based on random and bias error estimates                        see [2]_
    TMYData.Zenithlum                   Avg. luminance at the sky's zenith during the 60 minutes prior to timestamp, cd/m^2
    TMYData.ZenithlumSource             See [1]_, Table 1-4
    TMYData.ZenithlumUncertainty        Uncertainty based on random and bias error estimates                        see [1]_ section 2.10
    TMYData.TotCld                      Amount of sky dome covered by clouds or obscuring phenonema at time stamp, tenths of sky
    TMYData.TotCldSource                See [1]_, Table 1-5, 8760x1 cell array of strings
    TMYData.TotCldUncertainty           See [1]_, Table 1-6
    TMYData.OpqCld                      Amount of sky dome covered by clouds or obscuring phenonema that prevent observing the sky at time stamp, tenths of sky
    TMYData.OpqCldSource                See [1]_, Table 1-5, 8760x1 cell array of strings
    TMYData.OpqCldUncertainty           See [1]_, Table 1-6
    TMYData.DryBulb                     Dry bulb temperature at the time indicated, deg C
    TMYData.DryBulbSource               See [1]_, Table 1-5, 8760x1 cell array of strings
    TMYData.DryBulbUncertainty          See [1]_, Table 1-6
    TMYData.DewPoint                    Dew-point temperature at the time indicated, deg C
    TMYData.DewPointSource              See [1]_, Table 1-5, 8760x1 cell array of strings
    TMYData.DewPointUncertainty         See [1]_, Table 1-6
    TMYData.RHum                        Relatitudeive humidity at the time indicated, percent
    TMYData.RHumSource                  See [1]_, Table 1-5, 8760x1 cell array of strings
    TMYData.RHumUncertainty             See [1]_, Table 1-6
    TMYData.Pressure                    Station pressure at the time indicated, 1 mbar
    TMYData.PressureSource              See [1]_, Table 1-5, 8760x1 cell array of strings
    TMYData.PressureUncertainty         See [1]_, Table 1-6
    TMYData.Wdir                        Wind direction at time indicated, degrees from north (360 = north; 0 = undefined,calm)
    TMYData.WdirSource                  See [1]_, Table 1-5, 8760x1 cell array of strings
    TMYData.WdirUncertainty             See [1]_, Table 1-6
    TMYData.Wspd                        Wind speed at the time indicated, meter/second
    TMYData.WspdSource                  See [1]_, Table 1-5, 8760x1 cell array of strings
    TMYData.WspdUncertainty             See [1]_, Table 1-6
    TMYData.Hvis                        Distance to discernable remote objects at time indicated (7777=unlimited), meter
    TMYData.HvisSource                  See [1]_, Table 1-5, 8760x1 cell array of strings
    TMYData.HvisUncertainty             See [1]_, Table 1-6
    TMYData.CeilHgt                     Height of cloud base above local terrain (7777=unlimited), meter
    TMYData.CeilHgtSource               See [1]_, Table 1-5, 8760x1 cell array of strings
    TMYData.CeilHgtUncertainty          See [1]_, Table 1-6
    TMYData.Pwat                        Total precipitable water contained in a column of unit cross section from earth to top of atmosphere, cm
    TMYData.PwatSource                  See [1]_, Table 1-5, 8760x1 cell array of strings
    TMYData.PwatUncertainty             See [1]_, Table 1-6
    TMYData.AOD                         The broadband aerosol optical depth per unit of air mass due to extinction by aerosol component of atmosphere, unitless
    TMYData.AODSource                   See [1]_, Table 1-5, 8760x1 cell array of strings
    TMYData.AODUncertainty              See [1]_, Table 1-6
    TMYData.Alb                         The ratio of reflected solar irradiance to global horizontal irradiance, unitless
    TMYData.AlbSource                   See [1]_, Table 1-5, 8760x1 cell array of strings
    TMYData.AlbUncertainty              See [1]_, Table 1-6
    TMYData.Lprecipdepth                The amount of liquid precipitation observed at indicated time for the period indicated in the liquid precipitation quantity field, millimeter
    TMYData.Lprecipquantity             The period of accumulatitudeion for the liquid precipitation depth field, hour
    TMYData.LprecipSource               See [1]_, Table 1-5, 8760x1 cell array of strings
    TMYData.LprecipUncertainty          See [1]_, Table 1-6
    TMYData.PresWth                     Present weather code, see [2]_.
    TMYData.PresWthSource               Present weather code source, see [2]_.
    TMYData.PresWthUncertainty          Present weather code uncertainty, see [2]_.
    =============================       ======================================================================================================================================================

    .. warning:: TMY3 irradiance data corresponds to the *previous* hour, so
        the first index is 1AM, corresponding to the irradiance from midnight
        to 1AM, and the last index is midnight of the *next* year. For example,
        if the last index in the TMY3 file was 1988-12-31 24:00:00 this becomes
        1989-01-01 00:00:00 after calling :func:`~pvlib.iotools.read_tmy3`.

    .. warning:: When coercing the year, the last index in the dataframe will
        become midnight of the *next* year. For example, if the last index in
        the TMY3 was 1988-12-31 24:00:00, and year is coerced to 1990 then this
        becomes 1991-01-01 00:00:00.

    References
    ----------

    .. [1] Wilcox, S and Marion, W. "Users Manual for TMY3 Data Sets".
       NREL/TP-581-43156, Revised May 2008.

    .. [2] Wilcox, S. (2007). National Solar Radiation Database 1991 2005
       Update: Users Manual. 472 pp.; NREL Report No. TP-581-41364.
    '''
    # ... other code
```
### 9 - pvlib/iotools/tmy.py:

Start line: 252, End line: 380

```python
def read_tmy2(filename):
    '''
    Read a TMY2 file in to a DataFrame.

    Note that values contained in the DataFrame are unchanged from the
    TMY2 file (i.e. units  are retained). Time/Date and location data
    imported from the TMY2 file have been modified to a "friendlier"
    form conforming to modern conventions (e.g. N latitude is postive, E
    longitude is positive, the "24th" hour of any day is technically the
    "0th" hour of the next day). In the case of any discrepencies
    between this documentation and the TMY2 User's Manual [1]_, the TMY2
    User's Manual takes precedence.

    Parameters
    ----------
    filename : str
        A relative or absolute file path.

    Returns
    -------
    Tuple of the form (data, metadata).

    data : DataFrame
        A dataframe with the columns described in the table below. For a
        more detailed descriptions of each component, please consult the
        TMY2 User's Manual ([1]_), especially tables 3-1 through 3-6, and
        Appendix B.

    metadata : dict
        The site metadata available in the file.

    Notes
    -----

    The returned structures have the following fields.

    =============    ==================================
    key              description
    =============    ==================================
    WBAN             Site identifier code (WBAN number)
    City             Station name
    State            Station state 2 letter designator
    TZ               Hours from Greenwich
    latitude         Latitude in decimal degrees
    longitude        Longitude in decimal degrees
    altitude         Site elevation in meters
    =============    ==================================

    ============================   ==========================================================================================================================================================================
    TMYData field                   description
    ============================   ==========================================================================================================================================================================
    index                           Pandas timeseries object containing timestamps
    year
    month
    day
    hour
    ETR                             Extraterrestrial horizontal radiation recv'd during 60 minutes prior to timestamp, Wh/m^2
    ETRN                            Extraterrestrial normal radiation recv'd during 60 minutes prior to timestamp, Wh/m^2
    GHI                             Direct and diffuse horizontal radiation recv'd during 60 minutes prior to timestamp, Wh/m^2
    GHISource                       See [1]_, Table 3-3
    GHIUncertainty                  See [1]_, Table 3-4
    DNI                             Amount of direct normal radiation (modeled) recv'd during 60 mintues prior to timestamp, Wh/m^2
    DNISource                       See [1]_, Table 3-3
    DNIUncertainty                  See [1]_, Table 3-4
    DHI                             Amount of diffuse horizontal radiation recv'd during 60 minutes prior to timestamp, Wh/m^2
    DHISource                       See [1]_, Table 3-3
    DHIUncertainty                  See [1]_, Table 3-4
    GHillum                         Avg. total horizontal illuminance recv'd during the 60 minutes prior to timestamp, units of 100 lux (e.g. value of 50 = 5000 lux)
    GHillumSource                   See [1]_, Table 3-3
    GHillumUncertainty              See [1]_, Table 3-4
    DNillum                         Avg. direct normal illuminance recv'd during the 60 minutes prior to timestamp, units of 100 lux
    DNillumSource                   See [1]_, Table 3-3
    DNillumUncertainty              See [1]_, Table 3-4
    DHillum                         Avg. horizontal diffuse illuminance recv'd during the 60 minutes prior to timestamp, units of 100 lux
    DHillumSource                   See [1]_, Table 3-3
    DHillumUncertainty              See [1]_, Table 3-4
    Zenithlum                       Avg. luminance at the sky's zenith during the 60 minutes prior to timestamp, units of 10 Cd/m^2 (e.g. value of 700 = 7,000 Cd/m^2)
    ZenithlumSource                 See [1]_, Table 3-3
    ZenithlumUncertainty            See [1]_, Table 3-4
    TotCld                          Amount of sky dome covered by clouds or obscuring phenonema at time stamp, tenths of sky
    TotCldSource                    See [1]_, Table 3-5, 8760x1 cell array of strings
    TotCldUncertainty                See [1]_, Table 3-6
    OpqCld                          Amount of sky dome covered by clouds or obscuring phenonema that prevent observing the sky at time stamp, tenths of sky
    OpqCldSource                    See [1]_, Table 3-5, 8760x1 cell array of strings
    OpqCldUncertainty               See [1]_, Table 3-6
    DryBulb                         Dry bulb temperature at the time indicated, in tenths of degree C (e.g. 352 = 35.2 C).
    DryBulbSource                   See [1]_, Table 3-5, 8760x1 cell array of strings
    DryBulbUncertainty              See [1]_, Table 3-6
    DewPoint                        Dew-point temperature at the time indicated, in tenths of degree C (e.g. 76 = 7.6 C).
    DewPointSource                  See [1]_, Table 3-5, 8760x1 cell array of strings
    DewPointUncertainty             See [1]_, Table 3-6
    RHum                            Relative humidity at the time indicated, percent
    RHumSource                      See [1]_, Table 3-5, 8760x1 cell array of strings
    RHumUncertainty                 See [1]_, Table 3-6
    Pressure                        Station pressure at the time indicated, 1 mbar
    PressureSource                  See [1]_, Table 3-5, 8760x1 cell array of strings
    PressureUncertainty             See [1]_, Table 3-6
    Wdir                            Wind direction at time indicated, degrees from east of north (360 = 0 = north; 90 = East; 0 = undefined,calm)
    WdirSource                      See [1]_, Table 3-5, 8760x1 cell array of strings
    WdirUncertainty                 See [1]_, Table 3-6
    Wspd                            Wind speed at the time indicated, in tenths of meters/second (e.g. 212 = 21.2 m/s)
    WspdSource                      See [1]_, Table 3-5, 8760x1 cell array of strings
    WspdUncertainty                 See [1]_, Table 3-6
    Hvis                            Distance to discernable remote objects at time indicated (7777=unlimited, 9999=missing data), in tenths of kilometers (e.g. 341 = 34.1 km).
    HvisSource                      See [1]_, Table 3-5, 8760x1 cell array of strings
    HvisUncertainty                 See [1]_, Table 3-6
    CeilHgt                         Height of cloud base above local terrain (7777=unlimited, 88888=cirroform, 99999=missing data), in meters
    CeilHgtSource                   See [1]_, Table 3-5, 8760x1 cell array of strings
    CeilHgtUncertainty              See [1]_, Table 3-6
    Pwat                            Total precipitable water contained in a column of unit cross section from Earth to top of atmosphere, in millimeters
    PwatSource                      See [1]_, Table 3-5, 8760x1 cell array of strings
    PwatUncertainty                 See [1]_, Table 3-6
    AOD                             The broadband aerosol optical depth (broadband turbidity) in thousandths on the day indicated (e.g. 114 = 0.114)
    AODSource                       See [1]_, Table 3-5, 8760x1 cell array of strings
    AODUncertainty                  See [1]_, Table 3-6
    SnowDepth                       Snow depth in centimeters on the day indicated, (999 = missing data).
    SnowDepthSource                 See [1]_, Table 3-5, 8760x1 cell array of strings
    SnowDepthUncertainty            See [1]_, Table 3-6
    LastSnowfall                    Number of days since last snowfall (maximum value of 88, where 88 = 88 or greater days; 99 = missing data)
    LastSnowfallSource              See [1]_, Table 3-5, 8760x1 cell array of strings
    LastSnowfallUncertainty         See [1]_, Table 3-6
    PresentWeather                  See [1]_, Appendix B, an 8760x1 cell array of strings. Each string contains 10 numeric values. The string can be parsed to determine each of 10 observed weather metrics.
    ============================   ==========================================================================================================================================================================

    References
    ----------

    .. [1] Marion, W and Urban, K. "Wilcox, S and Marion, W. "User's Manual
       for TMY2s". NREL 1995.
    '''
    # ... other code
```
### 10 - pvlib/tools.py:

Start line: 86, End line: 110

```python
def localize_to_utc(time, location):
    """
    Converts or localizes a time series to UTC.

    Parameters
    ----------
    time : datetime.datetime, pandas.DatetimeIndex,
           or pandas.Series/DataFrame with a DatetimeIndex.
    location : pvlib.Location object

    Returns
    -------
    pandas object localized to UTC.
    """
    if isinstance(time, dt.datetime):
        if time.tzinfo is None:
            time = pytz.timezone(location.tz).localize(time)
        time_utc = time.astimezone(pytz.utc)
    else:
        try:
            time_utc = time.tz_convert('UTC')
        except TypeError:
            time_utc = time.tz_localize(location.tz).tz_convert('UTC')

    return time_utc
```
### 12 - pvlib/temperature.py:

Start line: 445, End line: 468

```python
def _fuentes_hconv(tave, windmod, tinoct, temp_delta, xlen, tilt,
                   check_reynold):
    # Calculate the convective coefficient as in Fuentes 1987 -- a mixture of
    # free, laminar, and turbulent convection.
    densair = 0.003484 * 101325.0 / tave  # density
    visair = 0.24237e-6 * tave**0.76 / densair  # kinematic viscosity
    condair = 2.1695e-4 * tave**0.84  # thermal conductivity
    reynold = windmod * xlen / visair
    # the boundary between laminar and turbulent is modeled as an abrupt
    # change at Re = 1.2e5:
    if check_reynold and reynold > 1.2e5:
        # turbulent convection
        hforce = 0.0282 / reynold**0.2 * densair * windmod * 1007 / 0.71**0.4
    else:
        # laminar convection
        hforce = 0.8600 / reynold**0.5 * densair * windmod * 1007 / 0.71**0.67
    # free convection via Grashof number
    # NB: Fuentes hardwires sind(tilt) as 0.5 for tilt=30
    grashof = 9.8 / tave * temp_delta * xlen**3 / visair**2 * sind(tilt)
    # product of Nusselt number and (k/l)
    hfree = 0.21 * (grashof * 0.71)**0.32 * condair / xlen
    # combine free and forced components
    hconv = (hfree**3 + hforce**3)**(1/3)
    return hconv
```
