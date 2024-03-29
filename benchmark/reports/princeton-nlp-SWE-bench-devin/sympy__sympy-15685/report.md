# sympy__sympy-15685

| **sympy/sympy** | `9ac430347eb80809a1dd89bbf5dad7ca593bbe63` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 3841 |
| **Any found context length** | 897 |
| **Avg pos** | 9.0 |
| **Min pos** | 2 |
| **Max pos** | 7 |
| **Top file pos** | 2 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sympy/physics/units/definitions.py b/sympy/physics/units/definitions.py
--- a/sympy/physics/units/definitions.py
+++ b/sympy/physics/units/definitions.py
@@ -50,22 +50,13 @@
 meter.set_dimension(length)
 meter.set_scale_factor(One)
 
-# gram; used to define its prefixed units
-g = gram = grams = Quantity("gram", abbrev="g")
-gram.set_dimension(mass)
-gram.set_scale_factor(One)
-
-# NOTE: the `kilogram` has scale factor 1000. In SI, kg is a base unit, but
-# nonetheless we are trying to be compatible with the `kilo` prefix. In a
-# similar manner, people using CGS or gaussian units could argue that the
-# `centimeter` rather than `meter` is the fundamental unit for length, but the
-# scale factor of `centimeter` will be kept as 1/100 to be compatible with the
-# `centi` prefix.  The current state of the code assumes SI unit dimensions, in
+# NOTE: the `kilogram` has scale factor of 1 in SI.
+# The current state of the code assumes SI unit dimensions, in
 # the future this module will be modified in order to be unit system-neutral
 # (that is, support all kinds of unit systems).
 kg = kilogram = kilograms = Quantity("kilogram", abbrev="kg")
 kilogram.set_dimension(mass)
-kilogram.set_scale_factor(kilo*gram)
+kilogram.set_scale_factor(One)
 
 s = second = seconds = Quantity("second", abbrev="s")
 second.set_dimension(time)
@@ -87,6 +78,9 @@
 candela.set_dimension(luminous_intensity)
 candela.set_scale_factor(One)
 
+g = gram = grams = Quantity("gram", abbrev="g")
+gram.set_dimension(mass)
+gram.set_scale_factor(kilogram/kilo)
 
 mg = milligram = milligrams = Quantity("milligram", abbrev="mg")
 milligram.set_dimension(mass)

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sympy/physics/units/definitions.py | 53 | 68 | 2 | 2 | 897
| sympy/physics/units/definitions.py | 90 | 90 | 7 | 2 | 3841


## Problem Statement

```
Make .scale_factor private in the units module
* sympy version: 1.3
* Python version: 3.6.6
* Operating System: Win10

### Description

Dividing a Quantity with dimension voltage by a Quantity with dimension current yields ohm/1000 when I expected ohm. In the SI system, 1 V/ 1 A = 1 Ω.

### What I Did

\`\`\`
>>> from sympy.physics.units import Quantity, voltage, current, ohm, convert_to
>>> vs = Quantity('vs')
>>> vs.set_dimension(voltage)
>>> vs_i = Quantity('vs_i')
>>> vs_i.set_dimension(current)
>>> convert_to(vs/vs_i, ohm)
ohm/1000
\`\`\`

### Further discussion
The problem is related to the kilogram workaround and the property `scale_factor`. The default scale_factor for a Quantity is 1.
\`\`\`
>>> vs.scale_factor
1.0
\`\`\`

The docstring for `scale_factor' states:

> Overall magnitude of the quantity as compared to the canonical units.

But, the scale factor for ohm is 1000.
\`\`\`
>>> ohm.scale_factor
1000

This value of 1000 conflicts with the definition. `scale_factor` is a user facing property and should be consistent with the unit system definition, in this case the SI. The kilogram workaround should be an internal implementation factor and not exposed to the user.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 sympy/physics/units/quantities.py | 102 | 112| 141 | 141 | 1832 | 
| **-> 2 <-** | **2 sympy/physics/units/definitions.py** | 1 | 86| 756 | 897 | 7043 | 
| 3 | **2 sympy/physics/units/definitions.py** | 514 | 592| 712 | 1609 | 7043 | 
| 4 | 3 sympy/physics/units/__init__.py | 1 | 83| 515 | 2124 | 8983 | 
| 5 | 3 sympy/physics/units/quantities.py | 74 | 100| 221 | 2345 | 8983 | 
| 6 | **3 sympy/physics/units/definitions.py** | 178 | 275| 743 | 3088 | 8983 | 
| **-> 7 <-** | **3 sympy/physics/units/definitions.py** | 87 | 177| 753 | 3841 | 8983 | 
| 8 | **3 sympy/physics/units/definitions.py** | 433 | 513| 741 | 4582 | 8983 | 
| 9 | 3 sympy/physics/units/quantities.py | 226 | 250| 154 | 4736 | 8983 | 
| 10 | **3 sympy/physics/units/definitions.py** | 277 | 364| 757 | 5493 | 8983 | 
| 11 | 4 sympy/physics/units/util.py | 133 | 167| 286 | 5779 | 10780 | 
| 12 | 4 sympy/physics/units/quantities.py | 114 | 154| 237 | 6016 | 10780 | 
| 13 | 4 sympy/physics/units/quantities.py | 156 | 176| 194 | 6210 | 10780 | 
| 14 | **4 sympy/physics/units/definitions.py** | 365 | 432| 746 | 6956 | 10780 | 
| 15 | 4 sympy/physics/units/quantities.py | 18 | 72| 368 | 7324 | 10780 | 
| 16 | 5 sympy/physics/units/unitsystem.py | 1 | 77| 551 | 7875 | 11528 | 
| 17 | 5 sympy/physics/units/util.py | 61 | 130| 669 | 8544 | 11528 | 
| 18 | 6 sympy/physics/units/systems/si.py | 1 | 42| 400 | 8944 | 11928 | 
| 19 | 6 sympy/physics/units/__init__.py | 85 | 211| 998 | 9942 | 11928 | 
| 20 | 6 sympy/physics/units/quantities.py | 1 | 15| 107 | 10049 | 11928 | 
| 21 | 7 sympy/physics/units/dimensions.py | 272 | 308| 322 | 10371 | 16910 | 
| 22 | 7 sympy/physics/units/util.py | 1 | 16| 112 | 10483 | 16910 | 
| 23 | 7 sympy/physics/units/unitsystem.py | 79 | 115| 201 | 10684 | 16910 | 
| 24 | 7 sympy/physics/units/dimensions.py | 667 | 712| 425 | 11109 | 16910 | 
| 25 | 8 sympy/physics/units/systems/natural.py | 1 | 32| 228 | 11337 | 17138 | 
| 26 | 8 sympy/physics/units/dimensions.py | 1 | 24| 190 | 11527 | 17138 | 
| 27 | 8 sympy/physics/units/quantities.py | 178 | 224| 439 | 11966 | 17138 | 
| 28 | 9 sympy/physics/units/systems/mksa.py | 1 | 33| 251 | 12217 | 17389 | 
| 29 | 9 sympy/physics/units/__init__.py | 214 | 263| 427 | 12644 | 17389 | 
| 30 | 10 sympy/physics/units/systems/mks.py | 1 | 37| 288 | 12932 | 17677 | 
| 31 | 10 sympy/physics/units/dimensions.py | 27 | 79| 355 | 13287 | 17677 | 
| 32 | 10 sympy/physics/units/dimensions.py | 602 | 664| 369 | 13656 | 17677 | 
| 33 | 10 sympy/physics/units/dimensions.py | 154 | 236| 627 | 14283 | 17677 | 
| 34 | 11 sympy/physics/units/prefixes.py | 1 | 118| 805 | 15088 | 19333 | 
| 35 | 11 sympy/physics/units/dimensions.py | 401 | 445| 300 | 15388 | 19333 | 
| 36 | 11 sympy/physics/units/util.py | 19 | 37| 153 | 15541 | 19333 | 
| 37 | 11 sympy/physics/units/dimensions.py | 310 | 399| 694 | 16235 | 19333 | 
| 38 | 11 sympy/physics/units/util.py | 40 | 58| 226 | 16461 | 19333 | 
| 39 | 11 sympy/physics/units/dimensions.py | 521 | 559| 237 | 16698 | 19333 | 
| 40 | 11 sympy/physics/units/dimensions.py | 447 | 461| 142 | 16840 | 19333 | 
| 41 | 11 sympy/physics/units/dimensions.py | 105 | 134| 161 | 17001 | 19333 | 
| 42 | 12 sympy/core/numbers.py | 3880 | 3936| 358 | 17359 | 48966 | 
| 43 | 12 sympy/physics/units/dimensions.py | 481 | 499| 125 | 17484 | 48966 | 
| 44 | 12 sympy/physics/units/dimensions.py | 136 | 152| 159 | 17643 | 48966 | 
| 45 | 12 sympy/physics/units/dimensions.py | 81 | 103| 142 | 17785 | 48966 | 
| 46 | 13 sympy/physics/units/systems/__init__.py | 1 | 7| 0 | 17785 | 49043 | 
| 47 | 13 sympy/physics/units/dimensions.py | 583 | 600| 136 | 17921 | 49043 | 
| 48 | 13 sympy/physics/units/dimensions.py | 561 | 581| 172 | 18093 | 49043 | 
| 49 | 13 sympy/physics/units/dimensions.py | 501 | 519| 125 | 18218 | 49043 | 
| 50 | 13 sympy/core/numbers.py | 3800 | 3843| 236 | 18454 | 49043 | 
| 51 | 13 sympy/physics/units/dimensions.py | 463 | 479| 149 | 18603 | 49043 | 
| 52 | 13 sympy/physics/units/prefixes.py | 153 | 217| 600 | 19203 | 49043 | 
| 53 | 13 sympy/physics/units/util.py | 170 | 213| 347 | 19550 | 49043 | 
| 54 | 14 sympy/series/fourier.py | 318 | 345| 236 | 19786 | 52723 | 
| 55 | 15 sympy/physics/quantum/constants.py | 1 | 62| 362 | 20148 | 53085 | 
| 56 | 16 sympy/physics/quantum/sho1d.py | 380 | 395| 139 | 20287 | 58555 | 
| 57 | 17 sympy/simplify/simplify.py | 1 | 38| 404 | 20691 | 74609 | 
| 58 | 18 sympy/core/backend.py | 1 | 24| 357 | 21048 | 74967 | 
| 59 | 19 sympy/physics/quantum/operatorordering.py | 37 | 143| 843 | 21891 | 77304 | 
| 60 | 20 sympy/__init__.py | 1 | 91| 601 | 22492 | 77905 | 
| 61 | 21 sympy/calculus/util.py | 1 | 15| 174 | 22666 | 88215 | 
| 62 | 21 sympy/physics/quantum/sho1d.py | 1 | 35| 248 | 22914 | 88215 | 
| 63 | 21 sympy/physics/quantum/sho1d.py | 397 | 409| 133 | 23047 | 88215 | 
| 64 | 22 sympy/series/limits.py | 1 | 15| 151 | 23198 | 90353 | 
| 65 | 22 sympy/core/numbers.py | 1 | 37| 320 | 23518 | 90353 | 
| 66 | 22 sympy/core/numbers.py | 3845 | 3878| 288 | 23806 | 90353 | 
| 67 | 22 sympy/physics/units/prefixes.py | 121 | 150| 249 | 24055 | 90353 | 
| 68 | 23 sympy/solvers/ode.py | 4948 | 5088| 1543 | 25598 | 195120 | 


### Hint

```
@asmeurer I want to work on this issue, can you please help? I am understanding the issue but not sure how to fix it.
I would like to work on this issue.
You are not setting the scale factors of the quantities you define.
```

## Patch

```diff
diff --git a/sympy/physics/units/definitions.py b/sympy/physics/units/definitions.py
--- a/sympy/physics/units/definitions.py
+++ b/sympy/physics/units/definitions.py
@@ -50,22 +50,13 @@
 meter.set_dimension(length)
 meter.set_scale_factor(One)
 
-# gram; used to define its prefixed units
-g = gram = grams = Quantity("gram", abbrev="g")
-gram.set_dimension(mass)
-gram.set_scale_factor(One)
-
-# NOTE: the `kilogram` has scale factor 1000. In SI, kg is a base unit, but
-# nonetheless we are trying to be compatible with the `kilo` prefix. In a
-# similar manner, people using CGS or gaussian units could argue that the
-# `centimeter` rather than `meter` is the fundamental unit for length, but the
-# scale factor of `centimeter` will be kept as 1/100 to be compatible with the
-# `centi` prefix.  The current state of the code assumes SI unit dimensions, in
+# NOTE: the `kilogram` has scale factor of 1 in SI.
+# The current state of the code assumes SI unit dimensions, in
 # the future this module will be modified in order to be unit system-neutral
 # (that is, support all kinds of unit systems).
 kg = kilogram = kilograms = Quantity("kilogram", abbrev="kg")
 kilogram.set_dimension(mass)
-kilogram.set_scale_factor(kilo*gram)
+kilogram.set_scale_factor(One)
 
 s = second = seconds = Quantity("second", abbrev="s")
 second.set_dimension(time)
@@ -87,6 +78,9 @@
 candela.set_dimension(luminous_intensity)
 candela.set_scale_factor(One)
 
+g = gram = grams = Quantity("gram", abbrev="g")
+gram.set_dimension(mass)
+gram.set_scale_factor(kilogram/kilo)
 
 mg = milligram = milligrams = Quantity("milligram", abbrev="mg")
 milligram.set_dimension(mass)

```

## Test Patch

```diff
diff --git a/sympy/physics/units/tests/test_unitsystem.py b/sympy/physics/units/tests/test_unitsystem.py
--- a/sympy/physics/units/tests/test_unitsystem.py
+++ b/sympy/physics/units/tests/test_unitsystem.py
@@ -53,7 +53,7 @@ def test_print_unit_base():
 
     mksa = UnitSystem((m, kg, s, A), (Js,))
     with warns_deprecated_sympy():
-        assert mksa.print_unit_base(Js) == m**2*kg*s**-1/1000
+        assert mksa.print_unit_base(Js) == m**2*kg*s**-1
 
 
 def test_extend():

```


## Code snippets

### 1 - sympy/physics/units/quantities.py:

Start line: 102, End line: 112

```python
class Quantity(AtomicExpr):

    def set_scale_factor(self, scale_factor, unit_system="SI"):
        if unit_system != "SI":
            # TODO: add support for more units and dimension systems:
            raise NotImplementedError("Currently only SI is supported")

        scale_factor = sympify(scale_factor)
        # replace all prefixes by their ratio to canonical units:
        scale_factor = scale_factor.replace(lambda x: isinstance(x, Prefix), lambda x: x.scale_factor)
        # replace all quantities by their ratio to canonical units:
        scale_factor = scale_factor.replace(lambda x: isinstance(x, Quantity), lambda x: x.scale_factor)
        Quantity.SI_quantity_scale_factors[self] = scale_factor
```
### 2 - sympy/physics/units/definitions.py:

Start line: 1, End line: 86

```python
from sympy import Rational, pi, sqrt, sympify, S
from sympy.physics.units.quantities import Quantity
from sympy.physics.units.dimensions import (
    acceleration, action, amount_of_substance, capacitance, charge,
    conductance, current, energy, force, frequency, information, impedance, inductance,
    length, luminous_intensity, magnetic_density, magnetic_flux, mass, power,
    pressure, temperature, time, velocity, voltage)
from sympy.physics.units.dimensions import dimsys_default, Dimension
from sympy.physics.units.prefixes import (
    centi, deci, kilo, micro, milli, nano, pico,
    kibi, mebi, gibi, tebi, pebi, exbi)

One = S.One

#### UNITS ####

# Dimensionless:

percent = percents = Quantity("percent")
percent.set_dimension(One)
percent.set_scale_factor(Rational(1, 100))

permille = Quantity("permille")
permille.set_dimension(One)
permille.set_scale_factor(Rational(1, 1000))


# Angular units (dimensionless)

rad = radian = radians = Quantity("radian")
radian.set_dimension(One)
radian.set_scale_factor(One)

deg = degree = degrees = Quantity("degree", abbrev="deg")
degree.set_dimension(One)
degree.set_scale_factor(pi/180)

sr = steradian = steradians = Quantity("steradian", abbrev="sr")
steradian.set_dimension(One)
steradian.set_scale_factor(One)

mil = angular_mil = angular_mils = Quantity("angular_mil", abbrev="mil")
angular_mil.set_dimension(One)
angular_mil.set_scale_factor(2*pi/6400)


# Base units:

m = meter = meters = Quantity("meter", abbrev="m")
meter.set_dimension(length)
meter.set_scale_factor(One)

# gram; used to define its prefixed units
g = gram = grams = Quantity("gram", abbrev="g")
gram.set_dimension(mass)
gram.set_scale_factor(One)

# NOTE: the `kilogram` has scale factor 1000. In SI, kg is a base unit, but
# nonetheless we are trying to be compatible with the `kilo` prefix. In a
# similar manner, people using CGS or gaussian units could argue that the
# `centimeter` rather than `meter` is the fundamental unit for length, but the
# scale factor of `centimeter` will be kept as 1/100 to be compatible with the
# `centi` prefix.  The current state of the code assumes SI unit dimensions, in
# the future this module will be modified in order to be unit system-neutral
# (that is, support all kinds of unit systems).
kg = kilogram = kilograms = Quantity("kilogram", abbrev="kg")
kilogram.set_dimension(mass)
kilogram.set_scale_factor(kilo*gram)

s = second = seconds = Quantity("second", abbrev="s")
second.set_dimension(time)
second.set_scale_factor(One)

A = ampere = amperes = Quantity("ampere", abbrev='A')
ampere.set_dimension(current)
ampere.set_scale_factor(One)

K = kelvin = kelvins = Quantity("kelvin", abbrev='K')
kelvin.set_dimension(temperature)
kelvin.set_scale_factor(One)

mol = mole = moles = Quantity("mole", abbrev="mol")
mole.set_dimension(amount_of_substance)
mole.set_scale_factor(One)

cd = candela = candelas = Quantity("candela", abbrev="cd")
```
### 3 - sympy/physics/units/definitions.py:

Start line: 514, End line: 592

```python
planck_density.set_scale_factor(planck_mass / planck_length**3)

planck_energy_density = Quantity("planck_energy_density", abbrev="rho^E_P")
planck_energy_density.set_dimension(energy / length**3)
planck_energy_density.set_scale_factor(planck_energy / planck_length**3)

planck_intensity = Quantity("planck_intensity", abbrev="I_P")
planck_intensity.set_dimension(mass * time**(-3))
planck_intensity.set_scale_factor(planck_energy_density * speed_of_light)

planck_angular_frequency = Quantity("planck_angular_frequency", abbrev="omega_P")
planck_angular_frequency.set_dimension(1 / time)
planck_angular_frequency.set_scale_factor(1 / planck_time)

planck_pressure = Quantity("planck_pressure", abbrev="p_P")
planck_pressure.set_dimension(pressure)
planck_pressure.set_scale_factor(planck_force / planck_length**2)

planck_current = Quantity("planck_current", abbrev="I_P")
planck_current.set_dimension(current)
planck_current.set_scale_factor(planck_charge / planck_time)

planck_voltage = Quantity("planck_voltage", abbrev="V_P")
planck_voltage.set_dimension(voltage)
planck_voltage.set_scale_factor(planck_energy / planck_charge)

planck_impedance = Quantity("planck_impedance", abbrev="Z_P")
planck_impedance.set_dimension(impedance)
planck_impedance.set_scale_factor(planck_voltage / planck_current)

planck_acceleration = Quantity("planck_acceleration", abbrev="a_P")
planck_acceleration.set_dimension(acceleration)
planck_acceleration.set_scale_factor(speed_of_light / planck_time)


# Information theory units:
bit = bits = Quantity("bit")
bit.set_dimension(information)
bit.set_scale_factor(One)

byte = bytes = Quantity("byte")
byte.set_dimension(information)
byte.set_scale_factor(8*bit)

kibibyte = kibibytes = Quantity("kibibyte")
kibibyte.set_dimension(information)
kibibyte.set_scale_factor(kibi*byte)

mebibyte = mebibytes = Quantity("mebibyte")
mebibyte.set_dimension(information)
mebibyte.set_scale_factor(mebi*byte)

gibibyte = gibibytes = Quantity("gibibyte")
gibibyte.set_dimension(information)
gibibyte.set_scale_factor(gibi*byte)

tebibyte = tebibytes = Quantity("tebibyte")
tebibyte.set_dimension(information)
tebibyte.set_scale_factor(tebi*byte)

pebibyte = pebibytes = Quantity("pebibyte")
pebibyte.set_dimension(information)
pebibyte.set_scale_factor(pebi*byte)

exbibyte = exbibytes = Quantity("exbibyte")
exbibyte.set_dimension(information)
exbibyte.set_scale_factor(exbi*byte)


# check that scale factors are the right SI dimensions:
for _scale_factor, _dimension in zip(
        Quantity.SI_quantity_scale_factors.values(),
        Quantity.SI_quantity_dimension_map.values()):
    dimex = Quantity.get_dimensional_expr(_scale_factor)
    if dimex != 1:
        if not dimsys_default.equivalent_dims(_dimension, Dimension(dimex)):
            raise ValueError("quantity value and dimension mismatch")
del _scale_factor, _dimension
```
### 4 - sympy/physics/units/__init__.py:

Start line: 1, End line: 83

```python
# -*- coding: utf-8 -*-
# isort:skip_file
"""
Dimensional analysis and unit systems.

This module defines dimension/unit systems and physical quantities. It is
based on a group-theoretical construction where dimensions are represented as
vectors (coefficients being the exponents), and units are defined as a dimension
to which we added a scale.

Quantities are built from a factor and a unit, and are the basic objects that
one will use when doing computations.

All objects except systems and prefixes can be used in sympy expressions.
Note that as part of a CAS, various objects do not combine automatically
under operations.

Details about the implementation can be found in the documentation, and we
will not repeat all the explanations we gave there concerning our approach.
Ideas about future developments can be found on the `Github wiki
<https://github.com/sympy/sympy/wiki/Unit-systems>`_, and you should consult
this page if you are willing to help.

Useful functions:

- ``find_unit``: easily lookup pre-defined units.
- ``convert_to(expr, newunit)``: converts an expression into the same
    expression expressed in another unit.

"""

from sympy.core.compatibility import string_types
from .dimensions import Dimension, DimensionSystem
from .unitsystem import UnitSystem
from .util import convert_to
from .quantities import Quantity

from .dimensions import (
    amount_of_substance, acceleration, action,
    capacitance, charge, conductance, current, energy,
    force, frequency, impedance, inductance, length,
    luminous_intensity, magnetic_density,
    magnetic_flux, mass, momentum, power, pressure, temperature, time,
    velocity, voltage, volume
)

Unit = Quantity

speed = velocity
luminosity = luminous_intensity
magnetic_flux_density = magnetic_density
amount = amount_of_substance

from .prefixes import (
    # 10-power based:
    yotta,
    zetta,
    exa,
    peta,
    tera,
    giga,
    mega,
    kilo,
    hecto,
    deca,
    deci,
    centi,
    milli,
    micro,
    nano,
    pico,
    femto,
    atto,
    zepto,
    yocto,
    # 2-power based:
    kibi,
    mebi,
    gibi,
    tebi,
    pebi,
    exbi,
)
```
### 5 - sympy/physics/units/quantities.py:

Start line: 74, End line: 100

```python
class Quantity(AtomicExpr):

    ### Currently only SI is supported: ###

    # Dimensional representations for the SI units:
    SI_quantity_dimension_map = {}
    # Scale factors in SI units:
    SI_quantity_scale_factors = {}

    def set_dimension(self, dimension, unit_system="SI"):
        from sympy.physics.units.dimensions import dimsys_default, DimensionSystem

        if unit_system != "SI":
            # TODO: add support for more units and dimension systems:
            raise NotImplementedError("Currently only SI is supported")

        dim_sys = dimsys_default

        if not isinstance(dimension, dimensions.Dimension):
            if dimension == 1:
                dimension = Dimension(1)
            else:
                raise ValueError("expected dimension or 1")
        else:
            for dim_sym in dimension.name.atoms(Dimension):
                if dim_sym not in [i.name for i in dim_sys._dimensional_dependencies]:
                    raise ValueError("Dimension %s is not registered in the "
                                     "dimensional dependency tree." % dim_sym)
        Quantity.SI_quantity_dimension_map[self] = dimension
```
### 6 - sympy/physics/units/definitions.py:

Start line: 178, End line: 275

```python
becquerel = Bq = Quantity("becquerel")
becquerel.set_dimension(1/time)
becquerel.set_scale_factor(1/second)


# Common length units

km = kilometer = kilometers = Quantity("kilometer", abbrev="km")
kilometer.set_dimension(length)
kilometer.set_scale_factor(kilo*meter)

dm = decimeter = decimeters = Quantity("decimeter", abbrev="dm")
decimeter.set_dimension(length)
decimeter.set_scale_factor(deci*meter)

cm = centimeter = centimeters = Quantity("centimeter", abbrev="cm")
centimeter.set_dimension(length)
centimeter.set_scale_factor(centi*meter)

mm = millimeter = millimeters = Quantity("millimeter", abbrev="mm")
millimeter.set_dimension(length)
millimeter.set_scale_factor(milli*meter)

um = micrometer = micrometers = micron = microns = Quantity("micrometer", abbrev="um")
micrometer.set_dimension(length)
micrometer.set_scale_factor(micro*meter)

nm = nanometer = nanometers = Quantity("nanometer", abbrev="nn")
nanometer.set_dimension(length)
nanometer.set_scale_factor(nano*meter)

pm = picometer = picometers = Quantity("picometer", abbrev="pm")
picometer.set_dimension(length)
picometer.set_scale_factor(pico*meter)


ft = foot = feet = Quantity("foot", abbrev="ft")
foot.set_dimension(length)
foot.set_scale_factor(Rational(3048, 10000)*meter)

inch = inches = Quantity("inch")
inch.set_dimension(length)
inch.set_scale_factor(foot/12)

yd = yard = yards = Quantity("yard", abbrev="yd")
yard.set_dimension(length)
yard.set_scale_factor(3*feet)

mi = mile = miles = Quantity("mile")
mile.set_dimension(length)
mile.set_scale_factor(5280*feet)

nmi = nautical_mile = nautical_miles = Quantity("nautical_mile")
nautical_mile.set_dimension(length)
nautical_mile.set_scale_factor(6076*feet)


# Common volume and area units

l = liter = liters = Quantity("liter")
liter.set_dimension(length**3)
liter.set_scale_factor(meter**3 / 1000)

dl = deciliter = deciliters = Quantity("deciliter")
deciliter.set_dimension(length**3)
deciliter.set_scale_factor(liter / 10)

cl = centiliter = centiliters = Quantity("centiliter")
centiliter.set_dimension(length**3)
centiliter.set_scale_factor(liter / 100)

ml = milliliter = milliliters = Quantity("milliliter")
milliliter.set_dimension(length**3)
milliliter.set_scale_factor(liter / 1000)


# Common time units

ms = millisecond = milliseconds = Quantity("millisecond", abbrev="ms")
millisecond.set_dimension(time)
millisecond.set_scale_factor(milli*second)

us = microsecond = microseconds = Quantity("microsecond", abbrev="us")
microsecond.set_dimension(time)
microsecond.set_scale_factor(micro*second)

ns = nanosecond = nanoseconds = Quantity("nanosecond", abbrev="ns")
nanosecond.set_dimension(time)
nanosecond.set_scale_factor(nano*second)

ps = picosecond = picoseconds = Quantity("picosecond", abbrev="ps")
picosecond.set_dimension(time)
picosecond.set_scale_factor(pico*second)


minute = minutes = Quantity("minute")
minute.set_dimension(time)
minute.set_scale_factor(60*second)
```
### 7 - sympy/physics/units/definitions.py:

Start line: 87, End line: 177

```python
candela.set_dimension(luminous_intensity)
candela.set_scale_factor(One)


mg = milligram = milligrams = Quantity("milligram", abbrev="mg")
milligram.set_dimension(mass)
milligram.set_scale_factor(milli*gram)

ug = microgram = micrograms = Quantity("microgram", abbrev="ug")
microgram.set_dimension(mass)
microgram.set_scale_factor(micro*gram)


# derived units
newton = newtons = N = Quantity("newton", abbrev="N")
newton.set_dimension(force)
newton.set_scale_factor(kilogram*meter/second**2)

joule = joules = J = Quantity("joule", abbrev="J")
joule.set_dimension(energy)
joule.set_scale_factor(newton*meter)

watt = watts = W = Quantity("watt", abbrev="W")
watt.set_dimension(power)
watt.set_scale_factor(joule/second)

pascal = pascals = Pa = pa = Quantity("pascal", abbrev="Pa")
pascal.set_dimension(pressure)
pascal.set_scale_factor(newton/meter**2)

hertz = hz = Hz = Quantity("hertz", abbrev="Hz")
hertz.set_dimension(frequency)
hertz.set_scale_factor(One)


# MKSA extension to MKS: derived units

coulomb = coulombs = C = Quantity("coulomb", abbrev='C')
coulomb.set_dimension(charge)
coulomb.set_scale_factor(One)

volt = volts = v = V = Quantity("volt", abbrev='V')
volt.set_dimension(voltage)
volt.set_scale_factor(joule/coulomb)

ohm = ohms = Quantity("ohm", abbrev='ohm')
ohm.set_dimension(impedance)
ohm.set_scale_factor(volt/ampere)

siemens = S = mho = mhos = Quantity("siemens", abbrev='S')
siemens.set_dimension(conductance)
siemens.set_scale_factor(ampere/volt)

farad = farads = F = Quantity("farad", abbrev='F')
farad.set_dimension(capacitance)
farad.set_scale_factor(coulomb/volt)

henry = henrys = H = Quantity("henry", abbrev='H')
henry.set_dimension(inductance)
henry.set_scale_factor(volt*second/ampere)

tesla = teslas = T = Quantity("tesla", abbrev='T')
tesla.set_dimension(magnetic_density)
tesla.set_scale_factor(volt*second/meter**2)

weber = webers = Wb = wb = Quantity("weber", abbrev='Wb')
weber.set_dimension(magnetic_flux)
weber.set_scale_factor(joule/ampere)


# Other derived units:

optical_power = dioptre = D = Quantity("dioptre")
dioptre.set_dimension(1/length)
dioptre.set_scale_factor(1/meter)

lux = lx = Quantity("lux")
lux.set_dimension(luminous_intensity/length**2)
lux.set_scale_factor(steradian*candela/meter**2)

# katal is the SI unit of catalytic activity
katal = kat = Quantity("katal")
katal.set_dimension(amount_of_substance/time)
katal.set_scale_factor(mol/second)

# gray is the SI unit of absorbed dose
gray = Gy = Quantity("gray")
gray.set_dimension(energy/mass)
gray.set_scale_factor(meter**2/second**2)

# becquerel is the SI unit of radioactivity
```
### 8 - sympy/physics/units/definitions.py:

Start line: 433, End line: 513

```python
pound.set_dimension(mass)
pound.set_scale_factor(Rational(45359237, 100000000) * kg)

psi = Quantity("psi")
psi.set_dimension(pressure)
psi.set_scale_factor(pound * gee / inch ** 2)

dHg0 = 13.5951  # approx value at 0 C
mmHg = torr = Quantity("mmHg")
mmHg.set_dimension(pressure)
mmHg.set_scale_factor(dHg0 * acceleration_due_to_gravity * kilogram / meter**2)

mmu = mmus = milli_mass_unit = Quantity("milli_mass_unit")
milli_mass_unit.set_dimension(mass)
milli_mass_unit.set_scale_factor(atomic_mass_unit/1000)

quart = quarts = Quantity("quart")
quart.set_dimension(length**3)
quart.set_scale_factor(Rational(231, 4) * inch**3)


# Other convenient units and magnitudes

ly = lightyear = lightyears = Quantity("lightyear", abbrev="ly")
lightyear.set_dimension(length)
lightyear.set_scale_factor(speed_of_light*julian_year)

au = astronomical_unit = astronomical_units = Quantity("astronomical_unit", abbrev="AU")
astronomical_unit.set_dimension(length)
astronomical_unit.set_scale_factor(149597870691*meter)


# Fundamental Planck units:
planck_mass = Quantity("planck_mass", abbrev="m_P")
planck_mass.set_dimension(mass)
planck_mass.set_scale_factor(sqrt(hbar*speed_of_light/G))

planck_time = Quantity("planck_time", abbrev="t_P")
planck_time.set_dimension(time)
planck_time.set_scale_factor(sqrt(hbar*G/speed_of_light**5))

planck_temperature = Quantity("planck_temperature", abbrev="T_P")
planck_temperature.set_dimension(temperature)
planck_temperature.set_scale_factor(sqrt(hbar*speed_of_light**5/G/boltzmann**2))

planck_length = Quantity("planck_length", abbrev="l_P")
planck_length.set_dimension(length)
planck_length.set_scale_factor(sqrt(hbar*G/speed_of_light**3))

planck_charge = Quantity("planck_charge", abbrev="q_P")
planck_charge.set_dimension(charge)
planck_charge.set_scale_factor(sqrt(4*pi*electric_constant*hbar*speed_of_light))


# Derived Planck units:
planck_area = Quantity("planck_area")
planck_area.set_dimension(length**2)
planck_area.set_scale_factor(planck_length**2)

planck_volume = Quantity("planck_volume")
planck_volume.set_dimension(length**3)
planck_volume.set_scale_factor(planck_length**3)

planck_momentum = Quantity("planck_momentum")
planck_momentum.set_dimension(mass*velocity)
planck_momentum.set_scale_factor(planck_mass * speed_of_light)

planck_energy = Quantity("planck_energy", abbrev="E_P")
planck_energy.set_dimension(energy)
planck_energy.set_scale_factor(planck_mass * speed_of_light**2)

planck_force = Quantity("planck_force", abbrev="F_P")
planck_force.set_dimension(force)
planck_force.set_scale_factor(planck_energy / planck_length)

planck_power = Quantity("planck_power", abbrev="P_P")
planck_power.set_dimension(power)
planck_power.set_scale_factor(planck_energy / planck_time)

planck_density = Quantity("planck_density", abbrev="rho_P")
planck_density.set_dimension(mass/length**3)
```
### 9 - sympy/physics/units/quantities.py:

Start line: 226, End line: 250

```python
class Quantity(AtomicExpr):

    def convert_to(self, other):
        """
        Convert the quantity to another quantity of same dimensions.

        Examples
        ========

        >>> from sympy.physics.units import speed_of_light, meter, second
        >>> speed_of_light
        speed_of_light
        >>> speed_of_light.convert_to(meter/second)
        299792458*meter/second

        >>> from sympy.physics.units import liter
        >>> liter.convert_to(meter**3)
        meter**3/1000
        """
        from .util import convert_to
        return convert_to(self, other)

    @property
    def free_symbols(self):
        """Return free symbols from quantity."""
        return self.scale_factor.free_symbols
```
### 10 - sympy/physics/units/definitions.py:

Start line: 277, End line: 364

```python
h = hour = hours = Quantity("hour")
hour.set_dimension(time)
hour.set_scale_factor(60*minute)

day = days = Quantity("day")
day.set_dimension(time)
day.set_scale_factor(24*hour)


anomalistic_year = anomalistic_years = Quantity("anomalistic_year")
anomalistic_year.set_dimension(time)
anomalistic_year.set_scale_factor(365.259636*day)

sidereal_year = sidereal_years = Quantity("sidereal_year")
sidereal_year.set_dimension(time)
sidereal_year.set_scale_factor(31558149.540)

tropical_year = tropical_years = Quantity("tropical_year")
tropical_year.set_dimension(time)
tropical_year.set_scale_factor(365.24219*day)

common_year = common_years = Quantity("common_year")
common_year.set_dimension(time)
common_year.set_scale_factor(365*day)

julian_year = julian_years = Quantity("julian_year")
julian_year.set_dimension(time)
julian_year.set_scale_factor((365 + One/4)*day)

draconic_year = draconic_years = Quantity("draconic_year")
draconic_year.set_dimension(time)
draconic_year.set_scale_factor(346.62*day)

gaussian_year = gaussian_years = Quantity("gaussian_year")
gaussian_year.set_dimension(time)
gaussian_year.set_scale_factor(365.2568983*day)

full_moon_cycle = full_moon_cycles = Quantity("full_moon_cycle")
full_moon_cycle.set_dimension(time)
full_moon_cycle.set_scale_factor(411.78443029*day)


year = years = tropical_year

#### CONSTANTS ####

# Newton constant
G = gravitational_constant = Quantity("gravitational_constant", abbrev="G")
gravitational_constant.set_dimension(length**3*mass**-1*time**-2)
gravitational_constant.set_scale_factor(6.67408e-11*m**3/(kg*s**2))

# speed of light
c = speed_of_light = Quantity("speed_of_light", abbrev="c")
speed_of_light.set_dimension(velocity)
speed_of_light.set_scale_factor(299792458*meter/second)

# Reduced Planck constant
hbar = Quantity("hbar", abbrev="hbar")
hbar.set_dimension(action)
hbar.set_scale_factor(1.05457266e-34*joule*second)

# Planck constant
planck = Quantity("planck", abbrev="h")
planck.set_dimension(action)
planck.set_scale_factor(2*pi*hbar)

# Electronvolt
eV = electronvolt = electronvolts = Quantity("electronvolt", abbrev="eV")
electronvolt.set_dimension(energy)
electronvolt.set_scale_factor(1.60219e-19*joule)

# Avogadro number
avogadro_number = Quantity("avogadro_number")
avogadro_number.set_dimension(One)
avogadro_number.set_scale_factor(6.022140857e23)

# Avogadro constant
avogadro = avogadro_constant = Quantity("avogadro_constant")
avogadro_constant.set_dimension(amount_of_substance**-1)
avogadro_constant.set_scale_factor(avogadro_number / mol)

# Boltzmann constant
boltzmann = boltzmann_constant = Quantity("boltzmann_constant")
boltzmann_constant.set_dimension(energy/temperature)
boltzmann_constant.set_scale_factor(1.38064852e-23*joule/kelvin)

# Stefan-Boltzmann constant
stefan = stefan_boltzmann_constant = Quantity("stefan_boltzmann_constant")
```
### 14 - sympy/physics/units/definitions.py:

Start line: 365, End line: 432

```python
stefan_boltzmann_constant.set_dimension(energy*time**-1*length**-2*temperature**-4)
stefan_boltzmann_constant.set_scale_factor(5.670367e-8*joule/(s*m**2*kelvin**4))

# Atomic mass
amu = amus = atomic_mass_unit = atomic_mass_constant = Quantity("atomic_mass_constant")
atomic_mass_constant.set_dimension(mass)
atomic_mass_constant.set_scale_factor(1.660539040e-24*gram)

# Molar gas constant
R = molar_gas_constant = Quantity("molar_gas_constant", abbrev="R")
molar_gas_constant.set_dimension(energy/(temperature * amount_of_substance))
molar_gas_constant.set_scale_factor(8.3144598*joule/kelvin/mol)

# Faraday constant
faraday_constant = Quantity("faraday_constant")
faraday_constant.set_dimension(charge/amount_of_substance)
faraday_constant.set_scale_factor(96485.33289*C/mol)

# Josephson constant
josephson_constant = Quantity("josephson_constant", abbrev="K_j")
josephson_constant.set_dimension(frequency/voltage)
josephson_constant.set_scale_factor(483597.8525e9*hertz/V)

# Von Klitzing constant
von_klitzing_constant = Quantity("von_klitzing_constant", abbrev="R_k")
von_klitzing_constant.set_dimension(voltage/current)
von_klitzing_constant.set_scale_factor(25812.8074555*ohm)

# Acceleration due to gravity (on the Earth surface)
gee = gees = acceleration_due_to_gravity = Quantity("acceleration_due_to_gravity", abbrev="g")
acceleration_due_to_gravity.set_dimension(acceleration)
acceleration_due_to_gravity.set_scale_factor(9.80665*meter/second**2)

# magnetic constant:
u0 = magnetic_constant = vacuum_permeability = Quantity("magnetic_constant")
magnetic_constant.set_dimension(force/current**2)
magnetic_constant.set_scale_factor(4*pi/10**7 * newton/ampere**2)

# electric constat:
e0 = electric_constant = vacuum_permittivity = Quantity("vacuum_permittivity")
vacuum_permittivity.set_dimension(capacitance/length)
vacuum_permittivity.set_scale_factor(1/(u0 * c**2))

# vacuum impedance:
Z0 = vacuum_impedance = Quantity("vacuum_impedance", abbrev='Z_0')
vacuum_impedance.set_dimension(impedance)
vacuum_impedance.set_scale_factor(u0 * c)

# Coulomb's constant:
coulomb_constant = coulombs_constant = electric_force_constant = Quantity("coulomb_constant", abbrev="k_e")
coulomb_constant.set_dimension(force*length**2/charge**2)
coulomb_constant.set_scale_factor(1/(4*pi*vacuum_permittivity))


atmosphere = atmospheres = atm = Quantity("atmosphere", abbrev="atm")
atmosphere.set_dimension(pressure)
atmosphere.set_scale_factor(101325 * pascal)


kPa = kilopascal = Quantity("kilopascal", abbrev="kPa")
kilopascal.set_dimension(pressure)
kilopascal.set_scale_factor(kilo*Pa)

bar = bars = Quantity("bar", abbrev="bar")
bar.set_dimension(pressure)
bar.set_scale_factor(100*kPa)

pound = pounds = Quantity("pound")  # exact
```
