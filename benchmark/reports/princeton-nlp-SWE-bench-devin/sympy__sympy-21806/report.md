# sympy__sympy-21806

| **sympy/sympy** | `5824415f287a1842e47b75241ca4929efd0fbc7b` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 7426 |
| **Any found context length** | 1487 |
| **Avg pos** | 27.0 |
| **Min pos** | 6 |
| **Max pos** | 21 |
| **Top file pos** | 1 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sympy/algebras/quaternion.py b/sympy/algebras/quaternion.py
--- a/sympy/algebras/quaternion.py
+++ b/sympy/algebras/quaternion.py
@@ -8,6 +8,7 @@
 from sympy import integrate
 from sympy import Matrix
 from sympy import sympify
+from sympy.core.evalf import prec_to_dps
 from sympy.core.expr import Expr
 
 
@@ -490,6 +491,31 @@ def _ln(self):
 
         return Quaternion(a, b, c, d)
 
+    def _eval_evalf(self, prec):
+        """Returns the floating point approximations (decimal numbers) of the quaternion.
+
+        Returns
+        =======
+
+        Quaternion
+            Floating point approximations of quaternion(self)
+
+        Examples
+        ========
+
+        >>> from sympy.algebras.quaternion import Quaternion
+        >>> from sympy import sqrt
+        >>> q = Quaternion(1/sqrt(1), 1/sqrt(2), 1/sqrt(3), 1/sqrt(4))
+        >>> q.evalf()
+        1.00000000000000
+        + 0.707106781186547*i
+        + 0.577350269189626*j
+        + 0.500000000000000*k
+
+        """
+
+        return Quaternion(*[arg.evalf(n=prec_to_dps(prec)) for arg in self.args])
+
     def pow_cos_sin(self, p):
         """Computes the pth power in the cos-sin form.
 

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sympy/algebras/quaternion.py | 11 | 11 | 6 | 1 | 1487
| sympy/algebras/quaternion.py | 493 | 493 | 21 | 1 | 7426


## Problem Statement

```
Quaternion class has no overridden evalf method
`Quaternion` class has no overridden `evalf` method.

\`\`\`python
import sympy as sp
q = sp.Quaternion(1/sp.sqrt(2), 0, 0, 1/sp.sqrt(2))
q.evalf()  # does not work
# output: sqrt(2)/2 + 0*i + 0*j + sqrt(2)/2*k
\`\`\`

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 sympy/algebras/quaternion.py** | 164 | 196| 241 | 241 | 6132 | 
| 2 | **1 sympy/algebras/quaternion.py** | 436 | 465| 276 | 517 | 6132 | 
| 3 | **1 sympy/algebras/quaternion.py** | 366 | 388| 207 | 724 | 6132 | 
| 4 | **1 sympy/algebras/quaternion.py** | 361 | 364| 147 | 871 | 6132 | 
| 5 | **1 sympy/algebras/quaternion.py** | 44 | 79| 231 | 1102 | 6132 | 
| **-> 6 <-** | **1 sympy/algebras/quaternion.py** | 1 | 42| 385 | 1487 | 6132 | 
| 7 | **1 sympy/algebras/quaternion.py** | 293 | 359| 682 | 2169 | 6132 | 
| 8 | **1 sympy/algebras/quaternion.py** | 528 | 559| 243 | 2412 | 6132 | 
| 9 | **1 sympy/algebras/quaternion.py** | 198 | 250| 490 | 2902 | 6132 | 
| 10 | **1 sympy/algebras/quaternion.py** | 646 | 723| 779 | 3681 | 6132 | 
| 11 | **1 sympy/algebras/quaternion.py** | 467 | 491| 263 | 3944 | 6132 | 
| 12 | **1 sympy/algebras/quaternion.py** | 606 | 644| 250 | 4194 | 6132 | 
| 13 | **1 sympy/algebras/quaternion.py** | 120 | 162| 482 | 4676 | 6132 | 
| 14 | 2 sympy/core/evalf.py | 1556 | 1593| 304 | 4980 | 20401 | 
| 15 | **2 sympy/algebras/quaternion.py** | 252 | 291| 351 | 5331 | 20401 | 
| 16 | 2 sympy/core/evalf.py | 1513 | 1554| 350 | 5681 | 20401 | 
| 17 | **2 sympy/algebras/quaternion.py** | 390 | 434| 238 | 5919 | 20401 | 
| 18 | **2 sympy/algebras/quaternion.py** | 81 | 118| 296 | 6215 | 20401 | 
| 19 | 2 sympy/core/evalf.py | 1420 | 1512| 776 | 6991 | 20401 | 
| 20 | 3 sympy/polys/polyclasses.py | 739 | 751| 150 | 7141 | 34847 | 
| **-> 21 <-** | **3 sympy/algebras/quaternion.py** | 493 | 526| 285 | 7426 | 34847 | 
| 22 | 4 sympy/vector/orienters.py | 300 | 331| 342 | 7768 | 37845 | 
| 23 | 5 sympy/functions/special/hyper.py | 670 | 702| 383 | 8151 | 48768 | 
| 24 | 5 sympy/core/evalf.py | 1273 | 1329| 634 | 8785 | 48768 | 
| 25 | 6 sympy/functions/elementary/complexes.py | 1023 | 1062| 320 | 9105 | 59295 | 
| 26 | 6 sympy/core/evalf.py | 910 | 935| 218 | 9323 | 59295 | 
| 27 | 6 sympy/core/evalf.py | 758 | 773| 202 | 9525 | 59295 | 
| 28 | 7 sympy/physics/quantum/qexpr.py | 304 | 328| 178 | 9703 | 62330 | 
| 29 | 7 sympy/core/evalf.py | 1247 | 1270| 160 | 9863 | 62330 | 
| 30 | 7 sympy/core/evalf.py | 1039 | 1052| 138 | 10001 | 62330 | 
| 31 | 8 sympy/physics/quantum/state.py | 805 | 831| 152 | 10153 | 69423 | 
| 32 | 9 sympy/external/pythonmpq.py | 308 | 325| 139 | 10292 | 72330 | 
| 33 | 10 sympy/core/mod.py | 181 | 225| 416 | 10708 | 74124 | 
| 34 | 11 sympy/physics/quantum/operator.py | 138 | 182| 316 | 11024 | 78494 | 
| 35 | 11 sympy/core/evalf.py | 937 | 957| 141 | 11165 | 78494 | 
| 36 | 11 sympy/core/evalf.py | 222 | 239| 179 | 11344 | 78494 | 
| 37 | 11 sympy/core/evalf.py | 1055 | 1078| 214 | 11558 | 78494 | 
| 38 | 12 sympy/vector/basisdependent.py | 57 | 70| 134 | 11692 | 81075 | 
| 39 | 12 sympy/functions/special/hyper.py | 184 | 210| 321 | 12013 | 81075 | 
| 40 | 12 sympy/core/evalf.py | 842 | 886| 472 | 12485 | 81075 | 
| 41 | 12 sympy/core/evalf.py | 675 | 757| 848 | 13333 | 81075 | 
| 42 | 12 sympy/core/evalf.py | 1198 | 1244| 477 | 13810 | 81075 | 
| 43 | 12 sympy/core/evalf.py | 960 | 1037| 783 | 14593 | 81075 | 
| 44 | 13 sympy/physics/quantum/spin.py | 852 | 893| 593 | 15186 | 101467 | 
| 45 | 14 sympy/__init__.py | 226 | 254| 335 | 15521 | 109640 | 
| 46 | 15 sympy/core/function.py | 1619 | 1637| 266 | 15787 | 138025 | 
| 47 | 15 sympy/core/evalf.py | 632 | 672| 446 | 16233 | 138025 | 
| 48 | 15 sympy/core/evalf.py | 519 | 558| 379 | 16612 | 138025 | 
| 49 | 15 sympy/physics/quantum/spin.py | 595 | 615| 193 | 16805 | 138025 | 
| 50 | 15 sympy/core/function.py | 482 | 506| 193 | 16998 | 138025 | 
| 51 | 15 sympy/physics/quantum/spin.py | 145 | 158| 123 | 17121 | 138025 | 
| 52 | 16 examples/advanced/qft.py | 85 | 138| 607 | 17728 | 139328 | 
| 53 | 17 sympy/holonomic/holonomic.py | 1775 | 1848| 840 | 18568 | 164065 | 
| 54 | 18 sympy/physics/vector/frame.py | 1085 | 1111| 436 | 19004 | 177041 | 
| 55 | 19 sympy/functions/special/bessel.py | 1932 | 1944| 165 | 19169 | 194818 | 
| 56 | 19 sympy/core/evalf.py | 561 | 631| 611 | 19780 | 194818 | 


## Patch

```diff
diff --git a/sympy/algebras/quaternion.py b/sympy/algebras/quaternion.py
--- a/sympy/algebras/quaternion.py
+++ b/sympy/algebras/quaternion.py
@@ -8,6 +8,7 @@
 from sympy import integrate
 from sympy import Matrix
 from sympy import sympify
+from sympy.core.evalf import prec_to_dps
 from sympy.core.expr import Expr
 
 
@@ -490,6 +491,31 @@ def _ln(self):
 
         return Quaternion(a, b, c, d)
 
+    def _eval_evalf(self, prec):
+        """Returns the floating point approximations (decimal numbers) of the quaternion.
+
+        Returns
+        =======
+
+        Quaternion
+            Floating point approximations of quaternion(self)
+
+        Examples
+        ========
+
+        >>> from sympy.algebras.quaternion import Quaternion
+        >>> from sympy import sqrt
+        >>> q = Quaternion(1/sqrt(1), 1/sqrt(2), 1/sqrt(3), 1/sqrt(4))
+        >>> q.evalf()
+        1.00000000000000
+        + 0.707106781186547*i
+        + 0.577350269189626*j
+        + 0.500000000000000*k
+
+        """
+
+        return Quaternion(*[arg.evalf(n=prec_to_dps(prec)) for arg in self.args])
+
     def pow_cos_sin(self, p):
         """Computes the pth power in the cos-sin form.
 

```

## Test Patch

```diff
diff --git a/sympy/algebras/tests/test_quaternion.py b/sympy/algebras/tests/test_quaternion.py
--- a/sympy/algebras/tests/test_quaternion.py
+++ b/sympy/algebras/tests/test_quaternion.py
@@ -57,6 +57,11 @@ def test_quaternion_complex_real_addition():
     assert q1 - q1 == q0
 
 
+def test_quaternion_evalf():
+    assert Quaternion(sqrt(2), 0, 0, sqrt(3)).evalf() == Quaternion(sqrt(2).evalf(), 0, 0, sqrt(3).evalf())
+    assert Quaternion(1/sqrt(2), 0, 0, 1/sqrt(2)).evalf() == Quaternion((1/sqrt(2)).evalf(), 0, 0, (1/sqrt(2)).evalf())
+
+
 def test_quaternion_functions():
     q = Quaternion(w, x, y, z)
     q1 = Quaternion(1, 2, 3, 4)

```


## Code snippets

### 1 - sympy/algebras/quaternion.py:

Start line: 164, End line: 196

```python
class Quaternion(Expr):

    def __add__(self, other):
        return self.add(other)

    def __radd__(self, other):
        return self.add(other)

    def __sub__(self, other):
        return self.add(other*-1)

    def __mul__(self, other):
        return self._generic_mul(self, other)

    def __rmul__(self, other):
        return self._generic_mul(other, self)

    def __pow__(self, p):
        return self.pow(p)

    def __neg__(self):
        return Quaternion(-self._a, -self._b, -self._c, -self.d)

    def __truediv__(self, other):
        return self * sympify(other)**-1

    def __rtruediv__(self, other):
        return sympify(other) * self**-1

    def _eval_Integral(self, *args):
        return self.integrate(*args)

    def diff(self, *symbols, **kwargs):
        kwargs.setdefault('evaluate', True)
        return self.func(*[a.diff(*symbols, **kwargs) for a  in self.args])
```
### 2 - sympy/algebras/quaternion.py:

Start line: 436, End line: 465

```python
class Quaternion(Expr):

    def exp(self):
        """Returns the exponential of q (e^q).

        Returns
        =======

        Quaternion
            Exponential of q (e^q).

        Examples
        ========

        >>> from sympy.algebras.quaternion import Quaternion
        >>> q = Quaternion(1, 2, 3, 4)
        >>> q.exp()
        E*cos(sqrt(29))
        + 2*sqrt(29)*E*sin(sqrt(29))/29*i
        + 3*sqrt(29)*E*sin(sqrt(29))/29*j
        + 4*sqrt(29)*E*sin(sqrt(29))/29*k

        """
        # exp(q) = e^a(cos||v|| + v/||v||*sin||v||)
        q = self
        vector_norm = sqrt(q.b**2 + q.c**2 + q.d**2)
        a = exp(q.a) * cos(vector_norm)
        b = exp(q.a) * sin(vector_norm) * q.b / vector_norm
        c = exp(q.a) * sin(vector_norm) * q.c / vector_norm
        d = exp(q.a) * sin(vector_norm) * q.d / vector_norm

        return Quaternion(a, b, c, d)
```
### 3 - sympy/algebras/quaternion.py:

Start line: 366, End line: 388

```python
class Quaternion(Expr):

    def _eval_conjugate(self):
        """Returns the conjugate of the quaternion."""
        q = self
        return Quaternion(q.a, -q.b, -q.c, -q.d)

    def norm(self):
        """Returns the norm of the quaternion."""
        q = self
        # trigsimp is used to simplify sin(x)^2 + cos(x)^2 (these terms
        # arise when from_axis_angle is used).
        return sqrt(trigsimp(q.a**2 + q.b**2 + q.c**2 + q.d**2))

    def normalize(self):
        """Returns the normalized form of the quaternion."""
        q = self
        return q * (1/q.norm())

    def inverse(self):
        """Returns the inverse of the quaternion."""
        q = self
        if not q.norm():
            raise ValueError("Cannot compute inverse for a quaternion with zero norm")
        return conjugate(q) * (1/q.norm()**2)
```
### 4 - sympy/algebras/quaternion.py:

Start line: 361, End line: 364

```python
class Quaternion(Expr):

    @staticmethod
    def _generic_mul(q1, q2):
        # ... other code

        return Quaternion(-q1.b*q2.b - q1.c*q2.c - q1.d*q2.d + q1.a*q2.a,
                          q1.b*q2.a + q1.c*q2.d - q1.d*q2.c + q1.a*q2.b,
                          -q1.b*q2.d + q1.c*q2.a + q1.d*q2.b + q1.a*q2.c,
                          q1.b*q2.c - q1.c*q2.b + q1.d*q2.a + q1.a * q2.d)
```
### 5 - sympy/algebras/quaternion.py:

Start line: 44, End line: 79

```python
class Quaternion(Expr):

    def __new__(cls, a=0, b=0, c=0, d=0, real_field=True):
        a = sympify(a)
        b = sympify(b)
        c = sympify(c)
        d = sympify(d)

        if any(i.is_commutative is False for i in [a, b, c, d]):
            raise ValueError("arguments have to be commutative")
        else:
            obj = Expr.__new__(cls, a, b, c, d)
            obj._a = a
            obj._b = b
            obj._c = c
            obj._d = d
            obj._real_field = real_field
            return obj

    @property
    def a(self):
        return self._a

    @property
    def b(self):
        return self._b

    @property
    def c(self):
        return self._c

    @property
    def d(self):
        return self._d

    @property
    def real_field(self):
        return self._real_field
```
### 6 - sympy/algebras/quaternion.py:

Start line: 1, End line: 42

```python
# References :
# http://www.euclideanspace.com/maths/algebra/realNormedAlgebra/quaternions/
# https://en.wikipedia.org/wiki/Quaternion
from sympy import S, Rational
from sympy import re, im, conjugate, sign
from sympy import sqrt, sin, cos, acos, exp, ln
from sympy import trigsimp
from sympy import integrate
from sympy import Matrix
from sympy import sympify
from sympy.core.expr import Expr


class Quaternion(Expr):
    """Provides basic quaternion operations.
    Quaternion objects can be instantiated as Quaternion(a, b, c, d)
    as in (a + b*i + c*j + d*k).

    Examples
    ========

    >>> from sympy.algebras.quaternion import Quaternion
    >>> q = Quaternion(1, 2, 3, 4)
    >>> q
    1 + 2*i + 3*j + 4*k

    Quaternions over complex fields can be defined as :

    >>> from sympy.algebras.quaternion import Quaternion
    >>> from sympy import symbols, I
    >>> x = symbols('x')
    >>> q1 = Quaternion(x, x**3, x, x**2, real_field = False)
    >>> q2 = Quaternion(3 + 4*I, 2 + 5*I, 0, 7 + 8*I, real_field = False)
    >>> q1
    x + x**3*i + x*j + x**2*k
    >>> q2
    (3 + 4*I) + (2 + 5*I)*i + 0*j + (7 + 8*I)*k

    """
    _op_priority = 11.0

    is_commutative = False
```
### 7 - sympy/algebras/quaternion.py:

Start line: 293, End line: 359

```python
class Quaternion(Expr):

    @staticmethod
    def _generic_mul(q1, q2):
        """Generic multiplication.

        Parameters
        ==========

        q1 : Quaternion or symbol
        q2 : Quaternion or symbol

        It's important to note that if neither q1 nor q2 is a Quaternion,
        this function simply returns q1 * q2.

        Returns
        =======

        Quaternion
            The resultant quaternion after multiplying q1 and q2

        Examples
        ========

        >>> from sympy.algebras.quaternion import Quaternion
        >>> from sympy import Symbol
        >>> q1 = Quaternion(1, 2, 3, 4)
        >>> q2 = Quaternion(5, 6, 7, 8)
        >>> Quaternion._generic_mul(q1, q2)
        (-60) + 12*i + 30*j + 24*k
        >>> Quaternion._generic_mul(q1, 2)
        2 + 4*i + 6*j + 8*k
        >>> x = Symbol('x', real = True)
        >>> Quaternion._generic_mul(q1, x)
        x + 2*x*i + 3*x*j + 4*x*k

        Quaternions over complex fields :

        >>> from sympy.algebras.quaternion import Quaternion
        >>> from sympy import I
        >>> q3 = Quaternion(3 + 4*I, 2 + 5*I, 0, 7 + 8*I, real_field = False)
        >>> Quaternion._generic_mul(q3, 2 + 3*I)
        (2 + 3*I)*(3 + 4*I) + (2 + 3*I)*(2 + 5*I)*i + 0*j + (2 + 3*I)*(7 + 8*I)*k

        """
        q1 = sympify(q1)
        q2 = sympify(q2)

        # None is a Quaternion:
        if not isinstance(q1, Quaternion) and not isinstance(q2, Quaternion):
            return q1 * q2

        # If q1 is a number or a sympy expression instead of a quaternion
        if not isinstance(q1, Quaternion):
            if q2.real_field and q1.is_complex:
                return Quaternion(re(q1), im(q1), 0, 0) * q2
            elif q1.is_commutative:
                return Quaternion(q1 * q2.a, q1 * q2.b, q1 * q2.c, q1 * q2.d)
            else:
                raise ValueError("Only commutative expressions can be multiplied with a Quaternion.")

        # If q2 is a number or a sympy expression instead of a quaternion
        if not isinstance(q2, Quaternion):
            if q1.real_field and q2.is_complex:
                return q1 * Quaternion(re(q2), im(q2), 0, 0)
            elif q2.is_commutative:
                return Quaternion(q2 * q1.a, q2 * q1.b, q2 * q1.c, q2 * q1.d)
            else:
                raise ValueError("Only commutative expressions can be multiplied with a Quaternion.")
        # ... other code
```
### 8 - sympy/algebras/quaternion.py:

Start line: 528, End line: 559

```python
class Quaternion(Expr):

    def integrate(self, *args):
        """Computes integration of quaternion.

        Returns
        =======

        Quaternion
            Integration of the quaternion(self) with the given variable.

        Examples
        ========

        Indefinite Integral of quaternion :

        >>> from sympy.algebras.quaternion import Quaternion
        >>> from sympy.abc import x
        >>> q = Quaternion(1, 2, 3, 4)
        >>> q.integrate(x)
        x + 2*x*i + 3*x*j + 4*x*k

        Definite integral of quaternion :

        >>> from sympy.algebras.quaternion import Quaternion
        >>> from sympy.abc import x
        >>> q = Quaternion(1, 2, 3, 4)
        >>> q.integrate((x, 1, 5))
        4 + 8*i + 12*j + 16*k

        """
        # TODO: is this expression correct?
        return Quaternion(integrate(self.a, *args), integrate(self.b, *args),
                          integrate(self.c, *args), integrate(self.d, *args))
```
### 9 - sympy/algebras/quaternion.py:

Start line: 198, End line: 250

```python
class Quaternion(Expr):

    def add(self, other):
        """Adds quaternions.

        Parameters
        ==========

        other : Quaternion
            The quaternion to add to current (self) quaternion.

        Returns
        =======

        Quaternion
            The resultant quaternion after adding self to other

        Examples
        ========

        >>> from sympy.algebras.quaternion import Quaternion
        >>> from sympy import symbols
        >>> q1 = Quaternion(1, 2, 3, 4)
        >>> q2 = Quaternion(5, 6, 7, 8)
        >>> q1.add(q2)
        6 + 8*i + 10*j + 12*k
        >>> q1 + 5
        6 + 2*i + 3*j + 4*k
        >>> x = symbols('x', real = True)
        >>> q1.add(x)
        (x + 1) + 2*i + 3*j + 4*k

        Quaternions over complex fields :

        >>> from sympy.algebras.quaternion import Quaternion
        >>> from sympy import I
        >>> q3 = Quaternion(3 + 4*I, 2 + 5*I, 0, 7 + 8*I, real_field = False)
        >>> q3.add(2 + 3*I)
        (5 + 7*I) + (2 + 5*I)*i + 0*j + (7 + 8*I)*k

        """
        q1 = self
        q2 = sympify(other)

        # If q2 is a number or a sympy expression instead of a quaternion
        if not isinstance(q2, Quaternion):
            if q1.real_field and q2.is_complex:
                return Quaternion(re(q2) + q1.a, im(q2) + q1.b, q1.c, q1.d)
            elif q2.is_commutative:
                return Quaternion(q1.a + q2, q1.b, q1.c, q1.d)
            else:
                raise ValueError("Only commutative expressions can be added with a Quaternion.")

        return Quaternion(q1.a + q2.a, q1.b + q2.b, q1.c + q2.c, q1.d
                          + q2.d)
```
### 10 - sympy/algebras/quaternion.py:

Start line: 646, End line: 723

```python
class Quaternion(Expr):

    def to_rotation_matrix(self, v=None):
        """Returns the equivalent rotation transformation matrix of the quaternion
        which represents rotation about the origin if v is not passed.

        Parameters
        ==========

        v : tuple or None
            Default value: None

        Returns
        =======

        tuple
            Returns the equivalent rotation transformation matrix of the quaternion
            which represents rotation about the origin if v is not passed.

        Examples
        ========

        >>> from sympy.algebras.quaternion import Quaternion
        >>> from sympy import symbols, trigsimp, cos, sin
        >>> x = symbols('x')
        >>> q = Quaternion(cos(x/2), 0, 0, sin(x/2))
        >>> trigsimp(q.to_rotation_matrix())
        Matrix([
        [cos(x), -sin(x), 0],
        [sin(x),  cos(x), 0],
        [     0,       0, 1]])

        Generates a 4x4 transformation matrix (used for rotation about a point
        other than the origin) if the point(v) is passed as an argument.

        Examples
        ========

        >>> from sympy.algebras.quaternion import Quaternion
        >>> from sympy import symbols, trigsimp, cos, sin
        >>> x = symbols('x')
        >>> q = Quaternion(cos(x/2), 0, 0, sin(x/2))
        >>> trigsimp(q.to_rotation_matrix((1, 1, 1)))
         Matrix([
        [cos(x), -sin(x), 0,  sin(x) - cos(x) + 1],
        [sin(x),  cos(x), 0, -sin(x) - cos(x) + 1],
        [     0,       0, 1,                    0],
        [     0,       0, 0,                    1]])

        """

        q = self
        s = q.norm()**-2
        m00 = 1 - 2*s*(q.c**2 + q.d**2)
        m01 = 2*s*(q.b*q.c - q.d*q.a)
        m02 = 2*s*(q.b*q.d + q.c*q.a)

        m10 = 2*s*(q.b*q.c + q.d*q.a)
        m11 = 1 - 2*s*(q.b**2 + q.d**2)
        m12 = 2*s*(q.c*q.d - q.b*q.a)

        m20 = 2*s*(q.b*q.d - q.c*q.a)
        m21 = 2*s*(q.c*q.d + q.b*q.a)
        m22 = 1 - 2*s*(q.b**2 + q.c**2)

        if not v:
            return Matrix([[m00, m01, m02], [m10, m11, m12], [m20, m21, m22]])

        else:
            (x, y, z) = v

            m03 = x - x*m00 - y*m01 - z*m02
            m13 = y - x*m10 - y*m11 - z*m12
            m23 = z - x*m20 - y*m21 - z*m22
            m30 = m31 = m32 = 0
            m33 = 1

            return Matrix([[m00, m01, m02, m03], [m10, m11, m12, m13],
                          [m20, m21, m22, m23], [m30, m31, m32, m33]])
```
### 11 - sympy/algebras/quaternion.py:

Start line: 467, End line: 491

```python
class Quaternion(Expr):

    def _ln(self):
        """Returns the natural logarithm of the quaternion (_ln(q)).

        Examples
        ========

        >>> from sympy.algebras.quaternion import Quaternion
        >>> q = Quaternion(1, 2, 3, 4)
        >>> q._ln()
        log(sqrt(30))
        + 2*sqrt(29)*acos(sqrt(30)/30)/29*i
        + 3*sqrt(29)*acos(sqrt(30)/30)/29*j
        + 4*sqrt(29)*acos(sqrt(30)/30)/29*k

        """
        # _ln(q) = _ln||q|| + v/||v||*arccos(a/||q||)
        q = self
        vector_norm = sqrt(q.b**2 + q.c**2 + q.d**2)
        q_norm = q.norm()
        a = ln(q_norm)
        b = q.b * acos(q.a / q_norm) / vector_norm
        c = q.c * acos(q.a / q_norm) / vector_norm
        d = q.d * acos(q.a / q_norm) / vector_norm

        return Quaternion(a, b, c, d)
```
### 12 - sympy/algebras/quaternion.py:

Start line: 606, End line: 644

```python
class Quaternion(Expr):

    def to_axis_angle(self):
        """Returns the axis and angle of rotation of a quaternion

        Returns
        =======

        tuple
            Tuple of (axis, angle)

        Examples
        ========

        >>> from sympy.algebras.quaternion import Quaternion
        >>> q = Quaternion(1, 1, 1, 1)
        >>> (axis, angle) = q.to_axis_angle()
        >>> axis
        (sqrt(3)/3, sqrt(3)/3, sqrt(3)/3)
        >>> angle
        2*pi/3

        """
        q = self
        if q.a.is_negative:
            q = q * -1

        q = q.normalize()
        angle = trigsimp(2 * acos(q.a))

        # Since quaternion is normalised, q.a is less than 1.
        s = sqrt(1 - q.a*q.a)

        x = trigsimp(q.b / s)
        y = trigsimp(q.c / s)
        z = trigsimp(q.d / s)

        v = (x, y, z)
        t = (v, angle)

        return t
```
### 13 - sympy/algebras/quaternion.py:

Start line: 120, End line: 162

```python
class Quaternion(Expr):

    @classmethod
    def from_rotation_matrix(cls, M):
        """Returns the equivalent quaternion of a matrix. The quaternion will be normalized
        only if the matrix is special orthogonal (orthogonal and det(M) = 1).

        Parameters
        ==========

        M : Matrix
            Input matrix to be converted to equivalent quaternion. M must be special
            orthogonal (orthogonal and det(M) = 1) for the quaternion to be normalized.

        Returns
        =======

        Quaternion
            The quaternion equivalent to given matrix.

        Examples
        ========

        >>> from sympy.algebras.quaternion import Quaternion
        >>> from sympy import Matrix, symbols, cos, sin, trigsimp
        >>> x = symbols('x')
        >>> M = Matrix([[cos(x), -sin(x), 0], [sin(x), cos(x), 0], [0, 0, 1]])
        >>> q = trigsimp(Quaternion.from_rotation_matrix(M))
        >>> q
        sqrt(2)*sqrt(cos(x) + 1)/2 + 0*i + 0*j + sqrt(2 - 2*cos(x))*sign(sin(x))/2*k

        """

        absQ = M.det()**Rational(1, 3)

        a = sqrt(absQ + M[0, 0] + M[1, 1] + M[2, 2]) / 2
        b = sqrt(absQ + M[0, 0] - M[1, 1] - M[2, 2]) / 2
        c = sqrt(absQ - M[0, 0] + M[1, 1] - M[2, 2]) / 2
        d = sqrt(absQ - M[0, 0] - M[1, 1] + M[2, 2]) / 2

        b = b * sign(M[2, 1] - M[1, 2])
        c = c * sign(M[0, 2] - M[2, 0])
        d = d * sign(M[1, 0] - M[0, 1])

        return Quaternion(a, b, c, d)
```
### 15 - sympy/algebras/quaternion.py:

Start line: 252, End line: 291

```python
class Quaternion(Expr):

    def mul(self, other):
        """Multiplies quaternions.

        Parameters
        ==========

        other : Quaternion or symbol
            The quaternion to multiply to current (self) quaternion.

        Returns
        =======

        Quaternion
            The resultant quaternion after multiplying self with other

        Examples
        ========

        >>> from sympy.algebras.quaternion import Quaternion
        >>> from sympy import symbols
        >>> q1 = Quaternion(1, 2, 3, 4)
        >>> q2 = Quaternion(5, 6, 7, 8)
        >>> q1.mul(q2)
        (-60) + 12*i + 30*j + 24*k
        >>> q1.mul(2)
        2 + 4*i + 6*j + 8*k
        >>> x = symbols('x', real = True)
        >>> q1.mul(x)
        x + 2*x*i + 3*x*j + 4*x*k

        Quaternions over complex fields :

        >>> from sympy.algebras.quaternion import Quaternion
        >>> from sympy import I
        >>> q3 = Quaternion(3 + 4*I, 2 + 5*I, 0, 7 + 8*I, real_field = False)
        >>> q3.mul(2 + 3*I)
        (2 + 3*I)*(3 + 4*I) + (2 + 3*I)*(2 + 5*I)*i + 0*j + (2 + 3*I)*(7 + 8*I)*k

        """
        return self._generic_mul(self, other)
```
### 17 - sympy/algebras/quaternion.py:

Start line: 390, End line: 434

```python
class Quaternion(Expr):

    def pow(self, p):
        """Finds the pth power of the quaternion.

        Parameters
        ==========

        p : int
            Power to be applied on quaternion.

        Returns
        =======

        Quaternion
            Returns the p-th power of the current quaternion.
            Returns the inverse if p = -1.

        Examples
        ========

        >>> from sympy.algebras.quaternion import Quaternion
        >>> q = Quaternion(1, 2, 3, 4)
        >>> q.pow(4)
        668 + (-224)*i + (-336)*j + (-448)*k

        """
        p = sympify(p)
        q = self
        if p == -1:
            return q.inverse()
        res = 1

        if not p.is_Integer:
            return NotImplemented

        if p < 0:
            q, p = q.inverse(), -p

        while p > 0:
            if p % 2 == 1:
                res = q * res

            p = p//2
            q = q * q

        return res
```
### 18 - sympy/algebras/quaternion.py:

Start line: 81, End line: 118

```python
class Quaternion(Expr):

    @classmethod
    def from_axis_angle(cls, vector, angle):
        """Returns a rotation quaternion given the axis and the angle of rotation.

        Parameters
        ==========

        vector : tuple of three numbers
            The vector representation of the given axis.
        angle : number
            The angle by which axis is rotated (in radians).

        Returns
        =======

        Quaternion
            The normalized rotation quaternion calculated from the given axis and the angle of rotation.

        Examples
        ========

        >>> from sympy.algebras.quaternion import Quaternion
        >>> from sympy import pi, sqrt
        >>> q = Quaternion.from_axis_angle((sqrt(3)/3, sqrt(3)/3, sqrt(3)/3), 2*pi/3)
        >>> q
        1/2 + 1/2*i + 1/2*j + 1/2*k

        """
        (x, y, z) = vector
        norm = sqrt(x**2 + y**2 + z**2)
        (x, y, z) = (x / norm, y / norm, z / norm)
        s = sin(angle * S.Half)
        a = cos(angle * S.Half)
        b = x * s
        c = y * s
        d = z * s

        return cls(a, b, c, d).normalize()
```
### 21 - sympy/algebras/quaternion.py:

Start line: 493, End line: 526

```python
class Quaternion(Expr):

    def pow_cos_sin(self, p):
        """Computes the pth power in the cos-sin form.

        Parameters
        ==========

        p : int
            Power to be applied on quaternion.

        Returns
        =======

        Quaternion
            The p-th power in the cos-sin form.

        Examples
        ========

        >>> from sympy.algebras.quaternion import Quaternion
        >>> q = Quaternion(1, 2, 3, 4)
        >>> q.pow_cos_sin(4)
        900*cos(4*acos(sqrt(30)/30))
        + 1800*sqrt(29)*sin(4*acos(sqrt(30)/30))/29*i
        + 2700*sqrt(29)*sin(4*acos(sqrt(30)/30))/29*j
        + 3600*sqrt(29)*sin(4*acos(sqrt(30)/30))/29*k

        """
        # q = ||q||*(cos(a) + u*sin(a))
        # q^p = ||q||^p * (cos(p*a) + u*sin(p*a))

        q = self
        (v, angle) = q.to_axis_angle()
        q2 = Quaternion.from_axis_angle(v, p * angle)
        return q2 * (q.norm()**p)
```
