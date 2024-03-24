# pvlib__pvlib-python-1154

| **pvlib/pvlib-python** | `0b8f24c265d76320067a5ee908a57d475cd1bb24` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 302 |
| **Any found context length** | 302 |
| **Avg pos** | 1.0 |
| **Min pos** | 1 |
| **Max pos** | 1 |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/pvlib/irradiance.py b/pvlib/irradiance.py
--- a/pvlib/irradiance.py
+++ b/pvlib/irradiance.py
@@ -886,8 +886,9 @@ def reindl(surface_tilt, surface_azimuth, dhi, dni, ghi, dni_extra,
     # these are the () and [] sub-terms of the second term of eqn 8
     term1 = 1 - AI
     term2 = 0.5 * (1 + tools.cosd(surface_tilt))
-    term3 = 1 + np.sqrt(HB / ghi) * (tools.sind(0.5 * surface_tilt) ** 3)
-
+    with np.errstate(invalid='ignore', divide='ignore'):
+        hb_to_ghi = np.where(ghi == 0, 0, np.divide(HB, ghi))
+    term3 = 1 + np.sqrt(hb_to_ghi) * (tools.sind(0.5 * surface_tilt)**3)
     sky_diffuse = dhi * (AI * Rb + term1 * term2 * term3)
     sky_diffuse = np.maximum(sky_diffuse, 0)
 

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| pvlib/irradiance.py | 889 | 891 | 1 | 1 | 302


## Problem Statement

```
pvlib.irradiance.reindl() model generates NaNs when GHI = 0
**Describe the bug**
The reindl function should give zero sky diffuse when GHI is zero. Instead it generates NaN or Inf values due to "term3" having a quotient that divides by GHI.  

**Expected behavior**
The reindl function should result in zero sky diffuse when GHI is zero.


pvlib.irradiance.reindl() model generates NaNs when GHI = 0
**Describe the bug**
The reindl function should give zero sky diffuse when GHI is zero. Instead it generates NaN or Inf values due to "term3" having a quotient that divides by GHI.  

**Expected behavior**
The reindl function should result in zero sky diffuse when GHI is zero.



```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- |
| **-> 1 <-** | **1 pvlib/irradiance.py** | 870 | 893| 302 | 302 | 
| 2 | **1 pvlib/irradiance.py** | 792 | 869| 749 | 1051 | 
| 3 | **1 pvlib/irradiance.py** | 1131 | 1170| 342 | 1393 | 
| 4 | **1 pvlib/irradiance.py** | 2802 | 2871| 649 | 2042 | 
| 5 | 2 pvlib/clearsky.py | 93 | 142| 628 | 2670 | 
| 6 | **2 pvlib/irradiance.py** | 1930 | 2007| 993 | 3663 | 
| 7 | **2 pvlib/irradiance.py** | 1744 | 1760| 221 | 3884 | 
| 8 | **2 pvlib/irradiance.py** | 610 | 697| 841 | 4725 | 
| 9 | **2 pvlib/irradiance.py** | 2091 | 2184| 864 | 5589 | 
| 10 | **2 pvlib/irradiance.py** | 369 | 437| 547 | 6136 | 
| 11 | **2 pvlib/irradiance.py** | 1862 | 1889| 441 | 6577 | 
| 12 | **2 pvlib/irradiance.py** | 1048 | 1107| 779 | 7356 | 
| 13 | **2 pvlib/irradiance.py** | 1500 | 1517| 226 | 7582 | 
| 14 | 2 pvlib/clearsky.py | 356 | 402| 452 | 8034 | 
| 15 | **2 pvlib/irradiance.py** | 1425 | 1498| 744 | 8778 | 
| 16 | **2 pvlib/irradiance.py** | 1109 | 1128| 227 | 9005 | 
| 17 | 3 pvlib/spectrum/spectrl2.py | 32 | 47| 732 | 9737 | 
| 18 | 3 pvlib/clearsky.py | 17 | 14| 829 | 10566 | 
| 19 | **3 pvlib/irradiance.py** | 2241 | 2296| 470 | 11036 | 
| 20 | 4 pvlib/forecast.py | 456 | 500| 372 | 11408 | 
| 21 | 5 pvlib/modelchain.py | 1169 | 1194| 320 | 11728 | 
| 22 | **5 pvlib/irradiance.py** | 700 | 789| 866 | 12594 | 


### Hint

```
Verified. Looks like an easy fix.
Verified. Looks like an easy fix.
```

## Patch

```diff
diff --git a/pvlib/irradiance.py b/pvlib/irradiance.py
--- a/pvlib/irradiance.py
+++ b/pvlib/irradiance.py
@@ -886,8 +886,9 @@ def reindl(surface_tilt, surface_azimuth, dhi, dni, ghi, dni_extra,
     # these are the () and [] sub-terms of the second term of eqn 8
     term1 = 1 - AI
     term2 = 0.5 * (1 + tools.cosd(surface_tilt))
-    term3 = 1 + np.sqrt(HB / ghi) * (tools.sind(0.5 * surface_tilt) ** 3)
-
+    with np.errstate(invalid='ignore', divide='ignore'):
+        hb_to_ghi = np.where(ghi == 0, 0, np.divide(HB, ghi))
+    term3 = 1 + np.sqrt(hb_to_ghi) * (tools.sind(0.5 * surface_tilt)**3)
     sky_diffuse = dhi * (AI * Rb + term1 * term2 * term3)
     sky_diffuse = np.maximum(sky_diffuse, 0)
 

```

## Test Patch

```diff
diff --git a/pvlib/tests/test_irradiance.py b/pvlib/tests/test_irradiance.py
--- a/pvlib/tests/test_irradiance.py
+++ b/pvlib/tests/test_irradiance.py
@@ -203,7 +203,7 @@ def test_reindl(irrad_data, ephem_data, dni_et):
         40, 180, irrad_data['dhi'], irrad_data['dni'], irrad_data['ghi'],
         dni_et, ephem_data['apparent_zenith'], ephem_data['azimuth'])
     # values from matlab 1.4 code
-    assert_allclose(result, [np.nan, 27.9412, 104.1317, 34.1663], atol=1e-4)
+    assert_allclose(result, [0., 27.9412, 104.1317, 34.1663], atol=1e-4)
 
 
 def test_king(irrad_data, ephem_data):

```


## Code snippets

### 1 - pvlib/irradiance.py:

Start line: 870, End line: 893

```python
def reindl(surface_tilt, surface_azimuth, dhi, dni, ghi, dni_extra,
           solar_zenith, solar_azimuth):
    # ... other code
    cos_tt = np.maximum(cos_tt, 0)  # GH 526

    # do not apply cos(zen) limit here (needed for HB below)
    cos_solar_zenith = tools.cosd(solar_zenith)

    # ratio of titled and horizontal beam irradiance
    Rb = cos_tt / np.maximum(cos_solar_zenith, 0.01745)  # GH 432

    # Anisotropy Index
    AI = dni / dni_extra

    # DNI projected onto horizontal
    HB = dni * cos_solar_zenith
    HB = np.maximum(HB, 0)

    # these are the () and [] sub-terms of the second term of eqn 8
    term1 = 1 - AI
    term2 = 0.5 * (1 + tools.cosd(surface_tilt))
    term3 = 1 + np.sqrt(HB / ghi) * (tools.sind(0.5 * surface_tilt) ** 3)

    sky_diffuse = dhi * (AI * Rb + term1 * term2 * term3)
    sky_diffuse = np.maximum(sky_diffuse, 0)

    return sky_diffuse
```
### 2 - pvlib/irradiance.py:

Start line: 792, End line: 869

```python
def reindl(surface_tilt, surface_azimuth, dhi, dni, ghi, dni_extra,
           solar_zenith, solar_azimuth):
    r'''
    Determine diffuse irradiance from the sky on a tilted surface using
    Reindl's 1990 model

    .. math::

       I_{d} = DHI (A R_b + (1 - A) (\frac{1 + \cos\beta}{2})
       (1 + \sqrt{\frac{I_{hb}}{I_h}} \sin^3(\beta/2)) )

    Reindl's 1990 model determines the diffuse irradiance from the sky
    (ground reflected irradiance is not included in this algorithm) on a
    tilted surface using the surface tilt angle, surface azimuth angle,
    diffuse horizontal irradiance, direct normal irradiance, global
    horizontal irradiance, extraterrestrial irradiance, sun zenith
    angle, and sun azimuth angle.

    Parameters
    ----------
    surface_tilt : numeric
        Surface tilt angles in decimal degrees. The tilt angle is
        defined as degrees from horizontal (e.g. surface facing up = 0,
        surface facing horizon = 90)

    surface_azimuth : numeric
        Surface azimuth angles in decimal degrees. The azimuth
        convention is defined as degrees east of north (e.g. North = 0,
        South=180 East = 90, West = 270).

    dhi : numeric
        diffuse horizontal irradiance in W/m^2.

    dni : numeric
        direct normal irradiance in W/m^2.

    ghi: numeric
        Global irradiance in W/m^2.

    dni_extra : numeric
        Extraterrestrial normal irradiance in W/m^2.

    solar_zenith : numeric
        Apparent (refraction-corrected) zenith angles in decimal degrees.

    solar_azimuth : numeric
        Sun azimuth angles in decimal degrees. The azimuth convention is
        defined as degrees east of north (e.g. North = 0, East = 90,
        West = 270).

    Returns
    -------
    poa_sky_diffuse : numeric
        The sky diffuse component of the solar radiation.

    Notes
    -----
    The poa_sky_diffuse calculation is generated from the Loutzenhiser et al.
    (2007) paper, equation 8. Note that I have removed the beam and ground
    reflectance portion of the equation and this generates ONLY the diffuse
    radiation from the sky and circumsolar, so the form of the equation
    varies slightly from equation 8.

    References
    ----------
    .. [1] Loutzenhiser P.G. et. al. "Empirical validation of models to
       compute solar irradiance on inclined surfaces for building energy
       simulation" 2007, Solar Energy vol. 81. pp. 254-267

    .. [2] Reindl, D.T., Beckmann, W.A., Duffie, J.A., 1990a. Diffuse
       fraction correlations. Solar Energy 45(1), 1-7.

    .. [3] Reindl, D.T., Beckmann, W.A., Duffie, J.A., 1990b. Evaluation of
       hourly tilted surface radiation models. Solar Energy 45(1), 9-17.
    '''

    cos_tt = aoi_projection(surface_tilt, surface_azimuth,
                            solar_zenith, solar_azimuth)
    # ... other code
```
### 3 - pvlib/irradiance.py:

Start line: 1131, End line: 1170

```python
def clearsky_index(ghi, clearsky_ghi, max_clearsky_index=2.0):
    """
    Calculate the clearsky index.

    The clearsky index is the ratio of global to clearsky global irradiance.
    Negative and non-finite clearsky index values will be truncated to zero.

    Parameters
    ----------
    ghi : numeric
        Global horizontal irradiance in W/m^2.

    clearsky_ghi : numeric
        Modeled clearsky GHI

    max_clearsky_index : numeric, default 2.0
        Maximum value of the clearsky index. The default, 2.0, allows
        for over-irradiance events typically seen in sub-hourly data.

    Returns
    -------
    clearsky_index : numeric
        Clearsky index
    """
    clearsky_index = ghi / clearsky_ghi
    # set +inf, -inf, and nans to zero
    clearsky_index = np.where(~np.isfinite(clearsky_index), 0,
                              clearsky_index)
    # but preserve nans in the input arrays
    input_is_nan = ~np.isfinite(ghi) | ~np.isfinite(clearsky_ghi)
    clearsky_index = np.where(input_is_nan, np.nan, clearsky_index)

    clearsky_index = np.maximum(clearsky_index, 0)
    clearsky_index = np.minimum(clearsky_index, max_clearsky_index)

    # preserve input type
    if isinstance(ghi, pd.Series):
        clearsky_index = pd.Series(clearsky_index, index=ghi.index)

    return clearsky_index
```
### 4 - pvlib/irradiance.py:

Start line: 2802, End line: 2871

```python
def dni(ghi, dhi, zenith, clearsky_dni=None, clearsky_tolerance=1.1,
        zenith_threshold_for_zero_dni=88.0,
        zenith_threshold_for_clearsky_limit=80.0):
    """
    Determine DNI from GHI and DHI.

    When calculating the DNI from GHI and DHI the calculated DNI may be
    unreasonably high or negative for zenith angles close to 90 degrees
    (sunrise/sunset transitions). This function identifies unreasonable DNI
    values and sets them to NaN. If the clearsky DNI is given unreasonably high
    values are cut off.

    Parameters
    ----------
    ghi : Series
        Global horizontal irradiance.

    dhi : Series
        Diffuse horizontal irradiance.

    zenith : Series
        True (not refraction-corrected) zenith angles in decimal
        degrees. Angles must be >=0 and <=180.

    clearsky_dni : None or Series, default None
        Clearsky direct normal irradiance.

    clearsky_tolerance : float, default 1.1
        If 'clearsky_dni' is given this parameter can be used to allow a
        tolerance by how much the calculated DNI value can be greater than
        the clearsky value before it is identified as an unreasonable value.

    zenith_threshold_for_zero_dni : float, default 88.0
        Non-zero DNI values for zenith angles greater than or equal to
        'zenith_threshold_for_zero_dni' will be set to NaN.

    zenith_threshold_for_clearsky_limit : float, default 80.0
        DNI values for zenith angles greater than or equal to
        'zenith_threshold_for_clearsky_limit' and smaller the
        'zenith_threshold_for_zero_dni' that are greater than the clearsky DNI
        (times allowed tolerance) will be corrected. Only applies if
        'clearsky_dni' is not None.

    Returns
    -------
    dni : Series
        The modeled direct normal irradiance.
    """

    # calculate DNI
    dni = (ghi - dhi) / tools.cosd(zenith)

    # cutoff negative values
    dni[dni < 0] = float('nan')

    # set non-zero DNI values for zenith angles >=
    # zenith_threshold_for_zero_dni to NaN
    dni[(zenith >= zenith_threshold_for_zero_dni) & (dni != 0)] = float('nan')

    # correct DNI values for zenith angles greater or equal to the
    # zenith_threshold_for_clearsky_limit and smaller than the
    # upper_cutoff_zenith that are greater than the clearsky DNI (times
    # clearsky_tolerance)
    if clearsky_dni is not None:
        max_dni = clearsky_dni * clearsky_tolerance
        dni[(zenith >= zenith_threshold_for_clearsky_limit) &
            (zenith < zenith_threshold_for_zero_dni) &
            (dni > max_dni)] = max_dni
    return dni
```
### 5 - pvlib/clearsky.py:

Start line: 93, End line: 142

```python
def ineichen(apparent_zenith, airmass_absolute, linke_turbidity,
             altitude=0, dni_extra=1364., perez_enhancement=False):
    # want NaNs in other inputs to propagate through to the output. This
    # is accomplished by judicious use and placement of np.maximum,
    # np.minimum, and np.fmax

    # use max so that nighttime values will result in 0s instead of
    # negatives. propagates nans.
    cos_zenith = np.maximum(tools.cosd(apparent_zenith), 0)

    tl = linke_turbidity

    fh1 = np.exp(-altitude/8000.)
    fh2 = np.exp(-altitude/1250.)
    cg1 = 5.09e-05 * altitude + 0.868
    cg2 = 3.92e-05 * altitude + 0.0387

    ghi = np.exp(-cg2*airmass_absolute*(fh1 + fh2*(tl - 1)))

    # https://github.com/pvlib/pvlib-python/issues/435
    if perez_enhancement:
        ghi *= np.exp(0.01*airmass_absolute**1.8)

    # use fmax to map airmass nans to 0s. multiply and divide by tl to
    # reinsert tl nans
    ghi = cg1 * dni_extra * cos_zenith * tl / tl * np.fmax(ghi, 0)

    # From [1] (Following [2] leads to 0.664 + 0.16268 / fh1)
    # See https://github.com/pvlib/pvlib-python/pull/808
    b = 0.664 + 0.163/fh1
    # BncI = "normal beam clear sky radiation"
    bnci = b * np.exp(-0.09 * airmass_absolute * (tl - 1))
    bnci = dni_extra * np.fmax(bnci, 0)

    # "empirical correction" SE 73, 157 & SE 73, 312.
    bnci_2 = ((1 - (0.1 - 0.2*np.exp(-tl))/(0.1 + 0.882/fh1)) /
              cos_zenith)
    bnci_2 = ghi * np.fmin(np.fmax(bnci_2, 0), 1e20)

    dni = np.minimum(bnci, bnci_2)

    dhi = ghi - dni*cos_zenith

    irrads = OrderedDict()
    irrads['ghi'] = ghi
    irrads['dni'] = dni
    irrads['dhi'] = dhi

    if isinstance(dni, pd.Series):
        irrads = pd.DataFrame.from_dict(irrads)

    return irrads
```
### 6 - pvlib/irradiance.py:

Start line: 1930, End line: 2007

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
### 7 - pvlib/irradiance.py:

Start line: 1744, End line: 1760

```python
def dirindex(ghi, ghi_clearsky, dni_clearsky, zenith, times, pressure=101325.,
             use_delta_kt_prime=True, temp_dew=None, min_cos_zenith=0.065,
             max_zenith=87):

    dni_dirint = dirint(ghi, zenith, times, pressure=pressure,
                        use_delta_kt_prime=use_delta_kt_prime,
                        temp_dew=temp_dew, min_cos_zenith=min_cos_zenith,
                        max_zenith=max_zenith)

    dni_dirint_clearsky = dirint(ghi_clearsky, zenith, times,
                                 pressure=pressure,
                                 use_delta_kt_prime=use_delta_kt_prime,
                                 temp_dew=temp_dew,
                                 min_cos_zenith=min_cos_zenith,
                                 max_zenith=max_zenith)

    dni_dirindex = dni_clearsky * dni_dirint / dni_dirint_clearsky

    dni_dirindex[dni_dirindex < 0] = 0.

    return dni_dirindex
```
### 8 - pvlib/irradiance.py:

Start line: 610, End line: 697

```python
def klucher(surface_tilt, surface_azimuth, dhi, ghi, solar_zenith,
            solar_azimuth):
    r'''
    Determine diffuse irradiance from the sky on a tilted surface
    using Klucher's 1979 model

    .. math::

       I_{d} = DHI \frac{1 + \cos\beta}{2} (1 + F' \sin^3(\beta/2))
       (1 + F' \cos^2\theta\sin^3\theta_z)

    where

    .. math::

       F' = 1 - (I_{d0} / GHI)^2

    Klucher's 1979 model determines the diffuse irradiance from the sky
    (ground reflected irradiance is not included in this algorithm) on a
    tilted surface using the surface tilt angle, surface azimuth angle,
    diffuse horizontal irradiance, direct normal irradiance, global
    horizontal irradiance, extraterrestrial irradiance, sun zenith
    angle, and sun azimuth angle.

    Parameters
    ----------
    surface_tilt : numeric
        Surface tilt angles in decimal degrees. surface_tilt must be >=0
        and <=180. The tilt angle is defined as degrees from horizontal
        (e.g. surface facing up = 0, surface facing horizon = 90)

    surface_azimuth : numeric
        Surface azimuth angles in decimal degrees. surface_azimuth must
        be >=0 and <=360. The Azimuth convention is defined as degrees
        east of north (e.g. North = 0, South=180 East = 90, West = 270).

    dhi : numeric
        Diffuse horizontal irradiance in W/m^2. DHI must be >=0.

    ghi : numeric
        Global irradiance in W/m^2. DNI must be >=0.

    solar_zenith : numeric
        Apparent (refraction-corrected) zenith angles in decimal
        degrees. solar_zenith must be >=0 and <=180.

    solar_azimuth : numeric
        Sun azimuth angles in decimal degrees. solar_azimuth must be >=0
        and <=360. The Azimuth convention is defined as degrees east of
        north (e.g. North = 0, East = 90, West = 270).

    Returns
    -------
    diffuse : numeric
        The sky diffuse component of the solar radiation.

    References
    ----------
    .. [1] Loutzenhiser P.G. et. al. "Empirical validation of models to compute
       solar irradiance on inclined surfaces for building energy simulation"
       2007, Solar Energy vol. 81. pp. 254-267

    .. [2] Klucher, T.M., 1979. Evaluation of models to predict insolation on
       tilted surfaces. Solar Energy 23 (2), 111-114.
    '''

    # zenith angle with respect to panel normal.
    cos_tt = aoi_projection(surface_tilt, surface_azimuth,
                            solar_zenith, solar_azimuth)
    cos_tt = np.maximum(cos_tt, 0)  # GH 526

    # silence warning from 0 / 0
    with np.errstate(invalid='ignore'):
        F = 1 - ((dhi / ghi) ** 2)

    try:
        # fails with single point input
        F.fillna(0, inplace=True)
    except AttributeError:
        F = np.where(np.isnan(F), 0, F)

    term1 = 0.5 * (1 + tools.cosd(surface_tilt))
    term2 = 1 + F * (tools.sind(0.5 * surface_tilt) ** 3)
    term3 = 1 + F * (cos_tt ** 2) * (tools.sind(solar_zenith) ** 3)

    sky_diffuse = dhi * term1 * term2 * term3

    return sky_diffuse
```
### 9 - pvlib/irradiance.py:

Start line: 2091, End line: 2184

```python
def erbs(ghi, zenith, datetime_or_doy, min_cos_zenith=0.065, max_zenith=87):
    r"""
    Estimate DNI and DHI from GHI using the Erbs model.

    The Erbs model [1]_ estimates the diffuse fraction DF from global
    horizontal irradiance through an empirical relationship between DF
    and the ratio of GHI to extraterrestrial irradiance, Kt. The
    function uses the diffuse fraction to compute DHI as

    .. math::

        DHI = DF \times GHI

    DNI is then estimated as

    .. math::

        DNI = (GHI - DHI)/\cos(Z)

    where Z is the zenith angle.

    Parameters
    ----------
    ghi: numeric
        Global horizontal irradiance in W/m^2.
    zenith: numeric
        True (not refraction-corrected) zenith angles in decimal degrees.
    datetime_or_doy : int, float, array, pd.DatetimeIndex
        Day of year or array of days of year e.g.
        pd.DatetimeIndex.dayofyear, or pd.DatetimeIndex.
    min_cos_zenith : numeric, default 0.065
        Minimum value of cos(zenith) to allow when calculating global
        clearness index `kt`. Equivalent to zenith = 86.273 degrees.
    max_zenith : numeric, default 87
        Maximum value of zenith to allow in DNI calculation. DNI will be
        set to 0 for times with zenith values greater than `max_zenith`.

    Returns
    -------
    data : OrderedDict or DataFrame
        Contains the following keys/columns:

            * ``dni``: the modeled direct normal irradiance in W/m^2.
            * ``dhi``: the modeled diffuse horizontal irradiance in
              W/m^2.
            * ``kt``: Ratio of global to extraterrestrial irradiance
              on a horizontal plane.

    References
    ----------
    .. [1] D. G. Erbs, S. A. Klein and J. A. Duffie, Estimation of the
       diffuse radiation fraction for hourly, daily and monthly-average
       global radiation, Solar Energy 28(4), pp 293-302, 1982. Eq. 1

    See also
    --------
    dirint
    disc
    """

    dni_extra = get_extra_radiation(datetime_or_doy)

    kt = clearness_index(ghi, zenith, dni_extra, min_cos_zenith=min_cos_zenith,
                         max_clearness_index=1)

    # For Kt <= 0.22, set the diffuse fraction
    df = 1 - 0.09*kt

    # For Kt > 0.22 and Kt <= 0.8, set the diffuse fraction
    df = np.where((kt > 0.22) & (kt <= 0.8),
                  0.9511 - 0.1604*kt + 4.388*kt**2 -
                  16.638*kt**3 + 12.336*kt**4,
                  df)

    # For Kt > 0.8, set the diffuse fraction
    df = np.where(kt > 0.8, 0.165, df)

    dhi = df * ghi

    dni = (ghi - dhi) / tools.cosd(zenith)
    bad_values = (zenith > max_zenith) | (ghi < 0) | (dni < 0)
    dni = np.where(bad_values, 0, dni)
    # ensure that closure relationship remains valid
    dhi = np.where(bad_values, ghi, dhi)

    data = OrderedDict()
    data['dni'] = dni
    data['dhi'] = dhi
    data['kt'] = kt

    if isinstance(datetime_or_doy, pd.DatetimeIndex):
        data = pd.DataFrame(data, index=datetime_or_doy)

    return data
```
### 10 - pvlib/irradiance.py:

Start line: 369, End line: 437

```python
def get_sky_diffuse(surface_tilt, surface_azimuth,
                    solar_zenith, solar_azimuth,
                    dni, ghi, dhi, dni_extra=None, airmass=None,
                    model='isotropic',
                    model_perez='allsitescomposite1990'):
    r"""
    Determine in-plane sky diffuse irradiance component
    using the specified sky diffuse irradiance model.

    Sky diffuse models include:
        * isotropic (default)
        * klucher
        * haydavies
        * reindl
        * king
        * perez

    Parameters
    ----------
    surface_tilt : numeric
        Panel tilt from horizontal.
    surface_azimuth : numeric
        Panel azimuth from north.
    solar_zenith : numeric
        Solar zenith angle.
    solar_azimuth : numeric
        Solar azimuth angle.
    dni : numeric
        Direct Normal Irradiance
    ghi : numeric
        Global horizontal irradiance
    dhi : numeric
        Diffuse horizontal irradiance
    dni_extra : None or numeric, default None
        Extraterrestrial direct normal irradiance
    airmass : None or numeric, default None
        Airmass
    model : String, default 'isotropic'
        Irradiance model.
    model_perez : String, default 'allsitescomposite1990'
        See perez.

    Returns
    -------
    poa_sky_diffuse : numeric
    """

    model = model.lower()
    if model == 'isotropic':
        sky = isotropic(surface_tilt, dhi)
    elif model == 'klucher':
        sky = klucher(surface_tilt, surface_azimuth, dhi, ghi,
                      solar_zenith, solar_azimuth)
    elif model == 'haydavies':
        sky = haydavies(surface_tilt, surface_azimuth, dhi, dni, dni_extra,
                        solar_zenith, solar_azimuth)
    elif model == 'reindl':
        sky = reindl(surface_tilt, surface_azimuth, dhi, dni, ghi, dni_extra,
                     solar_zenith, solar_azimuth)
    elif model == 'king':
        sky = king(surface_tilt, dhi, ghi, solar_zenith)
    elif model == 'perez':
        sky = perez(surface_tilt, surface_azimuth, dhi, dni, dni_extra,
                    solar_zenith, solar_azimuth, airmass,
                    model=model_perez)
    else:
        raise ValueError(f'invalid model selection {model}')

    return sky
```
### 11 - pvlib/irradiance.py:

Start line: 1862, End line: 1889

```python
def gti_dirint(poa_global, aoi, solar_zenith, solar_azimuth, times,
               surface_tilt, surface_azimuth, pressure=101325.,
               use_delta_kt_prime=True, temp_dew=None, albedo=.25,
               model='perez', model_perez='allsitescomposite1990',
               calculate_gt_90=True, max_iterations=30):

    aoi_lt_90 = aoi < 90

    # for AOI less than 90 degrees
    ghi, dni, dhi, kt_prime = _gti_dirint_lt_90(
        poa_global, aoi, aoi_lt_90, solar_zenith, solar_azimuth, times,
        surface_tilt, surface_azimuth, pressure=pressure,
        use_delta_kt_prime=use_delta_kt_prime, temp_dew=temp_dew,
        albedo=albedo, model=model, model_perez=model_perez,
        max_iterations=max_iterations)

    # for AOI greater than or equal to 90 degrees
    if calculate_gt_90:
        ghi_gte_90, dni_gte_90, dhi_gte_90 = _gti_dirint_gte_90(
            poa_global, aoi, solar_zenith, solar_azimuth,
            surface_tilt, times, kt_prime,
            pressure=pressure, temp_dew=temp_dew, albedo=albedo)
    else:
        ghi_gte_90, dni_gte_90, dhi_gte_90 = np.nan, np.nan, np.nan

    # put the AOI < 90 and AOI >= 90 conditions together
    output = OrderedDict()
    output['ghi'] = ghi.where(aoi_lt_90, ghi_gte_90)
    output['dni'] = dni.where(aoi_lt_90, dni_gte_90)
    output['dhi'] = dhi.where(aoi_lt_90, dhi_gte_90)

    output = pd.DataFrame(output, index=times)

    return output
```
### 12 - pvlib/irradiance.py:

Start line: 1048, End line: 1107

```python
def perez(surface_tilt, surface_azimuth, dhi, dni, dni_extra,
          solar_zenith, solar_azimuth, airmass,
          model='allsitescomposite1990', return_components=False):

    kappa = 1.041  # for solar_zenith in radians
    z = np.radians(solar_zenith)  # convert to radians

    # delta is the sky's "brightness"
    delta = dhi * airmass / dni_extra

    # epsilon is the sky's "clearness"
    with np.errstate(invalid='ignore'):
        eps = ((dhi + dni) / dhi + kappa * (z ** 3)) / (1 + kappa * (z ** 3))

    # numpy indexing below will not work with a Series
    if isinstance(eps, pd.Series):
        eps = eps.values

    # Perez et al define clearness bins according to the following
    # rules. 1 = overcast ... 8 = clear (these names really only make
    # sense for small zenith angles, but...) these values will
    # eventually be used as indicies for coeffecient look ups
    ebin = np.digitize(eps, (0., 1.065, 1.23, 1.5, 1.95, 2.8, 4.5, 6.2))
    ebin = np.array(ebin)  # GH 642
    ebin[np.isnan(eps)] = 0

    # correct for 0 indexing in coeffecient lookup
    # later, ebin = -1 will yield nan coefficients
    ebin -= 1

    # The various possible sets of Perez coefficients are contained
    # in a subfunction to clean up the code.
    F1c, F2c = _get_perez_coefficients(model)

    # results in invalid eps (ebin = -1) being mapped to nans
    nans = np.array([np.nan, np.nan, np.nan])
    F1c = np.vstack((F1c, nans))
    F2c = np.vstack((F2c, nans))

    F1 = (F1c[ebin, 0] + F1c[ebin, 1] * delta + F1c[ebin, 2] * z)
    F1 = np.maximum(F1, 0)

    F2 = (F2c[ebin, 0] + F2c[ebin, 1] * delta + F2c[ebin, 2] * z)
    F2 = np.maximum(F2, 0)

    A = aoi_projection(surface_tilt, surface_azimuth,
                       solar_zenith, solar_azimuth)
    A = np.maximum(A, 0)

    B = tools.cosd(solar_zenith)
    B = np.maximum(B, tools.cosd(85))

    # Calculate Diffuse POA from sky dome
    term1 = 0.5 * (1 - F1) * (1 + tools.cosd(surface_tilt))
    term2 = F1 * A / B
    term3 = F2 * tools.sind(surface_tilt)

    sky_diffuse = np.maximum(dhi * (term1 + term2 + term3), 0)

    # we've preserved the input type until now, so don't ruin it!
    if isinstance(sky_diffuse, pd.Series):
        sky_diffuse[np.isnan(airmass)] = 0
    else:
        sky_diffuse = np.where(np.isnan(airmass), 0, sky_diffuse)
    # ... other code
```
### 13 - pvlib/irradiance.py:

Start line: 1500, End line: 1517

```python
def dirint(ghi, solar_zenith, times, pressure=101325., use_delta_kt_prime=True,
           temp_dew=None, min_cos_zenith=0.065, max_zenith=87):

    disc_out = disc(ghi, solar_zenith, times, pressure=pressure,
                    min_cos_zenith=min_cos_zenith, max_zenith=max_zenith)
    airmass = disc_out['airmass']
    kt = disc_out['kt']

    kt_prime = clearness_index_zenith_independent(
        kt, airmass, max_clearness_index=1)
    delta_kt_prime = _delta_kt_prime_dirint(kt_prime, use_delta_kt_prime,
                                            times)
    w = _temp_dew_dirint(temp_dew, times)

    dirint_coeffs = _dirint_coeffs(times, kt_prime, solar_zenith, w,
                                   delta_kt_prime)

    # Perez eqn 5
    dni = disc_out['dni'] * dirint_coeffs

    return dni
```
### 15 - pvlib/irradiance.py:

Start line: 1425, End line: 1498

```python
def dirint(ghi, solar_zenith, times, pressure=101325., use_delta_kt_prime=True,
           temp_dew=None, min_cos_zenith=0.065, max_zenith=87):
    """
    Determine DNI from GHI using the DIRINT modification of the DISC
    model.

    Implements the modified DISC model known as "DIRINT" introduced in
    [1]. DIRINT predicts direct normal irradiance (DNI) from measured
    global horizontal irradiance (GHI). DIRINT improves upon the DISC
    model by using time-series GHI data and dew point temperature
    information. The effectiveness of the DIRINT model improves with
    each piece of information provided.

    The pvlib implementation limits the clearness index to 1.

    Parameters
    ----------
    ghi : array-like
        Global horizontal irradiance in W/m^2.

    solar_zenith : array-like
        True (not refraction-corrected) solar_zenith angles in decimal
        degrees.

    times : DatetimeIndex

    pressure : float or array-like, default 101325.0
        The site pressure in Pascal. Pressure may be measured or an
        average pressure may be calculated from site altitude.

    use_delta_kt_prime : bool, default True
        If True, indicates that the stability index delta_kt_prime is
        included in the model. The stability index adjusts the estimated
        DNI in response to dynamics in the time series of GHI. It is
        recommended that delta_kt_prime is not used if the time between
        GHI points is 1.5 hours or greater. If use_delta_kt_prime=True,
        input data must be Series.

    temp_dew : None, float, or array-like, default None
        Surface dew point temperatures, in degrees C. Values of temp_dew
        may be numeric or NaN. Any single time period point with a
        temp_dew=NaN does not have dew point improvements applied. If
        temp_dew is not provided, then dew point improvements are not
        applied.

    min_cos_zenith : numeric, default 0.065
        Minimum value of cos(zenith) to allow when calculating global
        clearness index `kt`. Equivalent to zenith = 86.273 degrees.

    max_zenith : numeric, default 87
        Maximum value of zenith to allow in DNI calculation. DNI will be
        set to 0 for times with zenith values greater than `max_zenith`.

    Returns
    -------
    dni : array-like
        The modeled direct normal irradiance in W/m^2 provided by the
        DIRINT model.

    Notes
    -----
    DIRINT model requires time series data (ie. one of the inputs must
    be a vector of length > 2).

    References
    ----------
    .. [1] Perez, R., P. Ineichen, E. Maxwell, R. Seals and A. Zelenka,
       (1992). "Dynamic Global-to-Direct Irradiance Conversion Models".
       ASHRAE Transactions-Research Series, pp. 354-369

    .. [2] Maxwell, E. L., "A Quasi-Physical Model for Converting Hourly
       Global Horizontal to Direct Normal Insolation", Technical Report No.
       SERI/TR-215-3087, Golden, CO: Solar Energy Research Institute, 1987.
    """
    # ... other code
```
### 16 - pvlib/irradiance.py:

Start line: 1109, End line: 1128

```python
def perez(surface_tilt, surface_azimuth, dhi, dni, dni_extra,
          solar_zenith, solar_azimuth, airmass,
          model='allsitescomposite1990', return_components=False):
    # ... other code

    if return_components:
        diffuse_components = OrderedDict()
        diffuse_components['sky_diffuse'] = sky_diffuse

        # Calculate the different components
        diffuse_components['isotropic'] = dhi * term1
        diffuse_components['circumsolar'] = dhi * term2
        diffuse_components['horizon'] = dhi * term3

        # Set values of components to 0 when sky_diffuse is 0
        mask = sky_diffuse == 0
        if isinstance(sky_diffuse, pd.Series):
            diffuse_components = pd.DataFrame(diffuse_components)
            diffuse_components.loc[mask] = 0
        else:
            diffuse_components = {k: np.where(mask, 0, v) for k, v in
                                  diffuse_components.items()}
        return diffuse_components
    else:
        return sky_diffuse
```
### 19 - pvlib/irradiance.py:

Start line: 2241, End line: 2296

```python
def _liujordan(zenith, transmittance, airmass, dni_extra=1367.0):
    '''
    Determine DNI, DHI, GHI from extraterrestrial flux, transmittance,
    and optical air mass number.

    Liu and Jordan, 1960, developed a simplified direct radiation model.
    DHI is from an empirical equation for diffuse radiation from Liu and
    Jordan, 1960.

    Parameters
    ----------
    zenith: pd.Series
        True (not refraction-corrected) zenith angles in decimal
        degrees. If Z is a vector it must be of the same size as all
        other vector inputs. Z must be >=0 and <=180.

    transmittance: float
        Atmospheric transmittance between 0 and 1.

    pressure: float, default 101325.0
        Air pressure

    dni_extra: float, default 1367.0
        Direct irradiance incident at the top of the atmosphere.

    Returns
    -------
    irradiance: DataFrame
        Modeled direct normal irradiance, direct horizontal irradiance,
        and global horizontal irradiance in W/m^2

    References
    ----------
    .. [1] Campbell, G. S., J. M. Norman (1998) An Introduction to
       Environmental Biophysics. 2nd Ed. New York: Springer.

    .. [2] Liu, B. Y., R. C. Jordan, (1960). "The interrelationship and
       characteristic distribution of direct, diffuse, and total solar
       radiation".  Solar Energy 4:1-19
    '''

    tau = transmittance

    dni = dni_extra*tau**airmass
    dhi = 0.3 * (1.0 - tau**airmass) * dni_extra * np.cos(np.radians(zenith))
    ghi = dhi + dni * np.cos(np.radians(zenith))

    irrads = OrderedDict()
    irrads['ghi'] = ghi
    irrads['dni'] = dni
    irrads['dhi'] = dhi

    if isinstance(ghi, pd.Series):
        irrads = pd.DataFrame(irrads)

    return irrads
```
### 22 - pvlib/irradiance.py:

Start line: 700, End line: 789

```python
def haydavies(surface_tilt, surface_azimuth, dhi, dni, dni_extra,
              solar_zenith=None, solar_azimuth=None, projection_ratio=None):
    r'''
    Determine diffuse irradiance from the sky on a tilted surface using
    Hay & Davies' 1980 model

    .. math::
        I_{d} = DHI ( A R_b + (1 - A) (\frac{1 + \cos\beta}{2}) )

    Hay and Davies' 1980 model determines the diffuse irradiance from
    the sky (ground reflected irradiance is not included in this
    algorithm) on a tilted surface using the surface tilt angle, surface
    azimuth angle, diffuse horizontal irradiance, direct normal
    irradiance, extraterrestrial irradiance, sun zenith angle, and sun
    azimuth angle.

    Parameters
    ----------
    surface_tilt : numeric
        Surface tilt angles in decimal degrees. The tilt angle is
        defined as degrees from horizontal (e.g. surface facing up = 0,
        surface facing horizon = 90)

    surface_azimuth : numeric
        Surface azimuth angles in decimal degrees. The azimuth
        convention is defined as degrees east of north (e.g. North=0,
        South=180, East=90, West=270).

    dhi : numeric
        Diffuse horizontal irradiance in W/m^2.

    dni : numeric
        Direct normal irradiance in W/m^2.

    dni_extra : numeric
        Extraterrestrial normal irradiance in W/m^2.

    solar_zenith : None or numeric, default None
        Solar apparent (refraction-corrected) zenith angles in decimal
        degrees. Must supply ``solar_zenith`` and ``solar_azimuth`` or
        supply ``projection_ratio``.

    solar_azimuth : None or numeric, default None
        Solar azimuth angles in decimal degrees. Must supply
        ``solar_zenith`` and ``solar_azimuth`` or supply
        ``projection_ratio``.

    projection_ratio : None or numeric, default None
        Ratio of angle of incidence projection to solar zenith angle
        projection. Must supply ``solar_zenith`` and ``solar_azimuth``
        or supply ``projection_ratio``.

    Returns
    --------
    sky_diffuse : numeric
        The sky diffuse component of the solar radiation.

    References
    -----------
    .. [1] Loutzenhiser P.G. et. al. "Empirical validation of models to
       compute solar irradiance on inclined surfaces for building energy
       simulation" 2007, Solar Energy vol. 81. pp. 254-267

    .. [2] Hay, J.E., Davies, J.A., 1980. Calculations of the solar
       radiation incident on an inclined surface. In: Hay, J.E., Won, T.K.
       (Eds.), Proc. of First Canadian Solar Radiation Data Workshop, 59.
       Ministry of Supply and Services, Canada.
    '''

    # if necessary, calculate ratio of titled and horizontal beam irradiance
    if projection_ratio is None:
        cos_tt = aoi_projection(surface_tilt, surface_azimuth,
                                solar_zenith, solar_azimuth)
        cos_tt = np.maximum(cos_tt, 0)  # GH 526
        cos_solar_zenith = tools.cosd(solar_zenith)
        Rb = cos_tt / np.maximum(cos_solar_zenith, 0.01745)  # GH 432
    else:
        Rb = projection_ratio

    # Anisotropy Index
    AI = dni / dni_extra

    # these are the () and [] sub-terms of the second term of eqn 7
    term1 = 1 - AI
    term2 = 0.5 * (1 + tools.cosd(surface_tilt))

    sky_diffuse = dhi * (AI * Rb + term1 * term2)
    sky_diffuse = np.maximum(sky_diffuse, 0)

    return sky_diffuse
```
