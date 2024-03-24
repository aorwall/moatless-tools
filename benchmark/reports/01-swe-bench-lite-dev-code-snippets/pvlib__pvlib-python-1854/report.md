# pvlib__pvlib-python-1854

| **pvlib/pvlib-python** | `27a3a07ebc84b11014d3753e4923902adf9a38c0` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 2234 |
| **Any found context length** | 425 |
| **Avg pos** | 15.0 |
| **Min pos** | 1 |
| **Max pos** | 8 |
| **Top file pos** | 1 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/pvlib/pvsystem.py b/pvlib/pvsystem.py
--- a/pvlib/pvsystem.py
+++ b/pvlib/pvsystem.py
@@ -101,10 +101,11 @@ class PVSystem:
 
     Parameters
     ----------
-    arrays : iterable of Array, optional
-        List of arrays that are part of the system. If not specified
-        a single array is created from the other parameters (e.g.
-        `surface_tilt`, `surface_azimuth`). Must contain at least one Array,
+    arrays : Array or iterable of Array, optional
+        An Array or list of arrays that are part of the system. If not
+        specified a single array is created from the other parameters (e.g.
+        `surface_tilt`, `surface_azimuth`). If specified as a list, the list
+        must contain at least one Array;
         if length of arrays is 0 a ValueError is raised. If `arrays` is
         specified the following PVSystem parameters are ignored:
 
@@ -220,6 +221,8 @@ def __init__(self,
                 strings_per_inverter,
                 array_losses_parameters,
             ),)
+        elif isinstance(arrays, Array):
+            self.arrays = (arrays,)
         elif len(arrays) == 0:
             raise ValueError("PVSystem must have at least one Array. "
                              "If you want to create a PVSystem instance "

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| pvlib/pvsystem.py | 104 | 106 | 3 | 1 | 2234
| pvlib/pvsystem.py | 220 | - | 8 | 1 | 3413


## Problem Statement

```
PVSystem with single Array generates an error
**Is your feature request related to a problem? Please describe.**

When a PVSystem has a single Array, you can't assign just the Array instance when constructing the PVSystem.

\`\`\`
mount = pvlib.pvsystem.FixedMount(surface_tilt=35, surface_azimuth=180)
array = pvlib.pvsystem.Array(mount=mount)
pv = pvlib.pvsystem.PVSystem(arrays=array)

---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-13-f5424e3db16a> in <module>
      3 mount = pvlib.pvsystem.FixedMount(surface_tilt=35, surface_azimuth=180)
      4 array = pvlib.pvsystem.Array(mount=mount)
----> 5 pv = pvlib.pvsystem.PVSystem(arrays=array)

~\anaconda3\lib\site-packages\pvlib\pvsystem.py in __init__(self, arrays, surface_tilt, surface_azimuth, albedo, surface_type, module, module_type, module_parameters, temperature_model_parameters, modules_per_string, strings_per_inverter, inverter, inverter_parameters, racking_model, losses_parameters, name)
    251                 array_losses_parameters,
    252             ),)
--> 253         elif len(arrays) == 0:
    254             raise ValueError("PVSystem must have at least one Array. "
    255                              "If you want to create a PVSystem instance "

TypeError: object of type 'Array' has no len()

\`\`\`

Not a bug per se, since the PVSystem docstring requests that `arrays` be iterable. Still, a bit inconvenient to have to do this

\`\`\`
mount = pvlib.pvsystem.FixedMount(surface_tilt=35, surface_azimuth=180)
array = pvlib.pvsystem.Array(mount=mount)
pv = pvlib.pvsystem.PVSystem(arrays=[array])
\`\`\`

**Describe the solution you'd like**
Handle `arrays=array` where `array` is an instance of `Array`

**Describe alternatives you've considered**
Status quo - either make the single Array into a list, or use the PVSystem kwargs.


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- |
| **-> 1 <-** | **1 pvlib/pvsystem.py** | 193 | 250| 425 | 425 | 
| 2 | **1 pvlib/pvsystem.py** | 890 | 992| 769 | 1194 | 
| **-> 3 <-** | **1 pvlib/pvsystem.py** | 72 | 191| 1040 | 2234 | 
| 4 | **1 pvlib/pvsystem.py** | 994 | 1007| 149 | 2383 | 
| 5 | **1 pvlib/pvsystem.py** | 252 | 270| 196 | 2579 | 
| 6 | **1 pvlib/pvsystem.py** | 873 | 887| 122 | 2701 | 
| 7 | **1 pvlib/pvsystem.py** | 285 | 283| 209 | 2910 | 
| **-> 8 <-** | **1 pvlib/pvsystem.py** | 0 | 49| 503 | 3413 | 
| 9 | **1 pvlib/pvsystem.py** | 1051 | 1070| 133 | 3546 | 
| 10 | **1 pvlib/pvsystem.py** | 1009 | 1049| 328 | 3874 | 
| 11 | **1 pvlib/pvsystem.py** | 1072 | 1136| 568 | 4442 | 
| 12 | 2 pvlib/modelchain.py | 1578 | 1602| 305 | 4747 | 
| 13 | **2 pvlib/pvsystem.py** | 540 | 578| 314 | 5061 | 
| 14 | 3 setup.py | 0 | 93| 876 | 5937 | 
| 15 | 4 docs/examples/bifacial/plot_pvfactors_fixed_tilt.py | 0 | 74| 660 | 6597 | 
| 16 | 4 pvlib/modelchain.py | 440 | 517| 791 | 7388 | 
| 17 | **4 pvlib/pvsystem.py** | 1267 | 1329| 515 | 7903 | 
| 18 | **4 pvlib/pvsystem.py** | 720 | 729| 113 | 8016 | 
| 19 | **4 pvlib/pvsystem.py** | 305 | 372| 661 | 8677 | 
| 20 | **4 pvlib/pvsystem.py** | 580 | 606| 225 | 8902 | 
| 21 | **4 pvlib/pvsystem.py** | 2462 | 2523| 734 | 9636 | 
| 22 | **4 pvlib/pvsystem.py** | 1359 | 1393| 293 | 9929 | 
| 23 | 5 pvlib/__init__.py | 0 | 28| 126 | 10055 | 
| 24 | 6 pvlib/bifacial/pvfactors.py | 91 | 118| 338 | 10393 | 
| 25 | **6 pvlib/pvsystem.py** | 1396 | 1468| 773 | 11166 | 
| 26 | **6 pvlib/pvsystem.py** | 1138 | 1178| 343 | 11509 | 


## Patch

```diff
diff --git a/pvlib/pvsystem.py b/pvlib/pvsystem.py
--- a/pvlib/pvsystem.py
+++ b/pvlib/pvsystem.py
@@ -101,10 +101,11 @@ class PVSystem:
 
     Parameters
     ----------
-    arrays : iterable of Array, optional
-        List of arrays that are part of the system. If not specified
-        a single array is created from the other parameters (e.g.
-        `surface_tilt`, `surface_azimuth`). Must contain at least one Array,
+    arrays : Array or iterable of Array, optional
+        An Array or list of arrays that are part of the system. If not
+        specified a single array is created from the other parameters (e.g.
+        `surface_tilt`, `surface_azimuth`). If specified as a list, the list
+        must contain at least one Array;
         if length of arrays is 0 a ValueError is raised. If `arrays` is
         specified the following PVSystem parameters are ignored:
 
@@ -220,6 +221,8 @@ def __init__(self,
                 strings_per_inverter,
                 array_losses_parameters,
             ),)
+        elif isinstance(arrays, Array):
+            self.arrays = (arrays,)
         elif len(arrays) == 0:
             raise ValueError("PVSystem must have at least one Array. "
                              "If you want to create a PVSystem instance "

```

## Test Patch

```diff
diff --git a/pvlib/tests/test_pvsystem.py b/pvlib/tests/test_pvsystem.py
--- a/pvlib/tests/test_pvsystem.py
+++ b/pvlib/tests/test_pvsystem.py
@@ -1887,8 +1887,6 @@ def test_PVSystem_multiple_array_creation():
     assert pv_system.arrays[0].module_parameters == {}
     assert pv_system.arrays[1].module_parameters == {'pdc0': 1}
     assert pv_system.arrays == (array_one, array_two)
-    with pytest.raises(TypeError):
-        pvsystem.PVSystem(arrays=array_one)
 
 
 def test_PVSystem_get_aoi():
@@ -2362,6 +2360,14 @@ def test_PVSystem_at_least_one_array():
         pvsystem.PVSystem(arrays=[])
 
 
+def test_PVSystem_single_array():
+    # GH 1831
+    single_array = pvsystem.Array(pvsystem.FixedMount())
+    system = pvsystem.PVSystem(arrays=single_array)
+    assert isinstance(system.arrays, tuple)
+    assert system.arrays[0] is single_array
+
+
 def test_combine_loss_factors():
     test_index = pd.date_range(start='1990/01/01T12:00', periods=365, freq='D')
     loss_1 = pd.Series(.10, index=test_index)

```


## Code snippets

### 1 - pvlib/pvsystem.py:

Start line: 193, End line: 250

```python
class PVSystem:

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
```
### 2 - pvlib/pvsystem.py:

Start line: 890, End line: 992

```python
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
```
### 3 - pvlib/pvsystem.py:

Start line: 72, End line: 191

```python
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
    """
```
### 4 - pvlib/pvsystem.py:

Start line: 994, End line: 1007

```python
class Array:

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
```
### 5 - pvlib/pvsystem.py:

Start line: 252, End line: 270

```python
class PVSystem:

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
```
### 6 - pvlib/pvsystem.py:

Start line: 873, End line: 887

```python
class PVSystem:

    @_unwrap_single_value
    def dc_ohms_from_percent(self):
        """
        Calculates the equivalent resistance of the wires for each array using
        :py:func:`pvlib.pvsystem.dc_ohms_from_percent`

        See :py:func:`pvlib.pvsystem.dc_ohms_from_percent` for details.
        """

        return tuple(array.dc_ohms_from_percent() for array in self.arrays)

    @property
    def num_arrays(self):
        """The number of Arrays in the system."""
        return len(self.arrays)
```
### 7 - pvlib/pvsystem.py:

Start line: 285, End line: 283

```python
class PVSystem:

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
```
### 8 - pvlib/pvsystem.py:

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
import inspect
from urllib.request import urlopen
import numpy as np
from scipy import constants
import pandas as pd
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Optional

from pvlib._deprecation import deprecated, warn_deprecated

from pvlib import (atmosphere, iam, inverter, irradiance,
                   singlediode as _singlediode, spectrum, temperature)
from pvlib.tools import _build_kwargs, _build_args
import pvlib.tools as tools


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
### 9 - pvlib/pvsystem.py:

Start line: 1051, End line: 1070

```python
class Array:

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
```
### 10 - pvlib/pvsystem.py:

Start line: 1009, End line: 1049

```python
class Array:

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
```
### 11 - pvlib/pvsystem.py:

Start line: 1072, End line: 1136

```python
class Array:

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
```
### 13 - pvlib/pvsystem.py:

Start line: 540, End line: 578

```python
class PVSystem:

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
```
### 17 - pvlib/pvsystem.py:

Start line: 1267, End line: 1329

```python
class Array:

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
        if all(elem in self.module_parameters
               for elem in ['V_mp_ref', 'I_mp_ref']):
            vmp_ref = self.module_parameters['V_mp_ref']
            imp_ref = self.module_parameters['I_mp_ref']

        # get relevant Vmp and Imp parameters from SAPM parameters
        elif all(elem in self.module_parameters for elem in ['Vmpo', 'Impo']):
            vmp_ref = self.module_parameters['Vmpo']
            imp_ref = self.module_parameters['Impo']

        # get relevant Vmp and Imp parameters if they are PVsyst-like
        elif all(elem in self.module_parameters for elem in ['Vmpp', 'Impp']):
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
```
### 18 - pvlib/pvsystem.py:

Start line: 720, End line: 729

```python
class PVSystem:

    def singlediode(self, photocurrent, saturation_current,
                    resistance_series, resistance_shunt, nNsVth,
                    ivcurve_pnts=None):
        """Wrapper around the :py:func:`pvlib.pvsystem.singlediode` function.

        See :py:func:`pvsystem.singlediode` for details
        """
        return singlediode(photocurrent, saturation_current,
                           resistance_series, resistance_shunt, nNsVth,
                           ivcurve_pnts=ivcurve_pnts)
```
### 19 - pvlib/pvsystem.py:

Start line: 305, End line: 372

```python
class PVSystem:

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
```
### 20 - pvlib/pvsystem.py:

Start line: 580, End line: 606

```python
class PVSystem:

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
```
### 21 - pvlib/pvsystem.py:

Start line: 2462, End line: 2523

```python
def singlediode(photocurrent, saturation_current, resistance_series,
                resistance_shunt, nNsVth, ivcurve_pnts=None,
                method='lambertw'):
    if ivcurve_pnts:
        warn_deprecated('0.10.0', name='pvlib.pvsystem.singlediode',
                        alternative=('pvlib.pvsystem.v_from_i and '
                                     'pvlib.pvsystem.i_from_v'),
                        obj_type='parameter ivcurve_pnts',
                        removal='0.11.0')
    args = (photocurrent, saturation_current, resistance_series,
            resistance_shunt, nNsVth)  # collect args
    # Calculate points on the IV curve using the LambertW solution to the
    # single diode equation
    if method.lower() == 'lambertw':
        out = _singlediode._lambertw(*args, ivcurve_pnts)
        points = out[:7]
        if ivcurve_pnts:
            ivcurve_i, ivcurve_v = out[7:]
    else:
        # Calculate points on the IV curve using either 'newton' or 'brentq'
        # methods. Voltages are determined by first solving the single diode
        # equation for the diode voltage V_d then backing out voltage
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
        points = i_sc, v_oc, i_mp, v_mp, p_mp, i_x, i_xx

        # calculate the IV curve if requested using bishop88
        if ivcurve_pnts:
            vd = v_oc * (
                (11.0 - np.logspace(np.log10(11.0), 0.0, ivcurve_pnts)) / 10.0
            )
            ivcurve_i, ivcurve_v, _ = _singlediode.bishop88(vd, *args)

    columns = ('i_sc', 'v_oc', 'i_mp', 'v_mp', 'p_mp', 'i_x', 'i_xx')

    if all(map(np.isscalar, args)) or ivcurve_pnts:
        out = {c: p for c, p in zip(columns, points)}

        if ivcurve_pnts:
            out.update(i=ivcurve_i, v=ivcurve_v)

        return out

    points = np.atleast_1d(*points)  # convert scalars to 1d-arrays
    points = np.vstack(points).T  # collect rows into DataFrame columns

    # save the first available pd.Series index, otherwise set to None
    index = next((a.index for a in args if isinstance(a, pd.Series)), None)

    out = pd.DataFrame(points, columns=columns, index=index)

    return out
```
### 22 - pvlib/pvsystem.py:

Start line: 1359, End line: 1393

```python
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
```
### 25 - pvlib/pvsystem.py:

Start line: 1396, End line: 1468

```python
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
```
### 26 - pvlib/pvsystem.py:

Start line: 1138, End line: 1178

```python
class Array:

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
            'martin_ruiz', 'sapm' and 'interp'.

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
        if model in ['ashrae', 'physical', 'martin_ruiz', 'interp']:
            func = getattr(iam, model)  # get function at pvlib.iam
            # get all parameters from function signature to retrieve them from
            # module_parameters if present
            params = set(inspect.signature(func).parameters.keys())
            params.discard('aoi')  # exclude aoi so it can't be repeated
            kwargs = _build_kwargs(params, self.module_parameters)
            return func(aoi, **kwargs)
        elif model == 'sapm':
            return iam.sapm(aoi, self.module_parameters)
        else:
            raise ValueError(model + ' is not a valid IAM model')
```
