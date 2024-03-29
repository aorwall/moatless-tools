# sympy__sympy-15349

| **sympy/sympy** | `768da1c6f6ec907524b8ebbf6bf818c92b56101b` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 727 |
| **Any found context length** | 727 |
| **Avg pos** | 1.0 |
| **Min pos** | 1 |
| **Max pos** | 1 |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sympy/algebras/quaternion.py b/sympy/algebras/quaternion.py
--- a/sympy/algebras/quaternion.py
+++ b/sympy/algebras/quaternion.py
@@ -529,7 +529,7 @@ def to_rotation_matrix(self, v=None):
 
         m10 = 2*s*(q.b*q.c + q.d*q.a)
         m11 = 1 - 2*s*(q.b**2 + q.d**2)
-        m12 = 2*s*(q.c*q.d + q.b*q.a)
+        m12 = 2*s*(q.c*q.d - q.b*q.a)
 
         m20 = 2*s*(q.b*q.d - q.c*q.a)
         m21 = 2*s*(q.c*q.d + q.b*q.a)

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sympy/algebras/quaternion.py | 532 | 532 | 1 | 1 | 727


## Problem Statement

```
Incorrect result with Quaterniont.to_rotation_matrix()
https://github.com/sympy/sympy/blob/ab14b02dba5a7e3e4fb1e807fc8a954f1047a1a1/sympy/algebras/quaternion.py#L489

There appears to be an error in the `Quaternion.to_rotation_matrix()` output.  The simplest example I created to illustrate the problem is as follows:

\`\`\`
>>import sympy
>>print('Sympy version: ', sympy.__version__)
Sympy version: 1.2

>> from sympy import *
>> x = symbols('x')
>> q = Quaternion(cos(x/2), sin(x/2), 0, 0)
>> trigsimp(q.to_rotation_matrix())
Matrix([
[1,      0,      0],
[0, cos(x), sin(x)],
[0, sin(x), cos(x)]])
\`\`\`
One of the `sin(x)` functions should be negative.  What was the reference of the original equations?  

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 sympy/algebras/quaternion.py** | 489 | 552| 727 | 727 | 5265 | 
| 2 | **1 sympy/algebras/quaternion.py** | 252 | 286| 405 | 1132 | 5265 | 
| 3 | **1 sympy/algebras/quaternion.py** | 143 | 178| 238 | 1370 | 5265 | 
| 4 | **1 sympy/algebras/quaternion.py** | 109 | 141| 438 | 1808 | 5265 | 
| 5 | **1 sympy/algebras/quaternion.py** | 1 | 17| 163 | 1971 | 5265 | 
| 6 | **1 sympy/algebras/quaternion.py** | 450 | 487| 286 | 2257 | 5265 | 
| 7 | **1 sympy/algebras/quaternion.py** | 288 | 310| 207 | 2464 | 5265 | 
| 8 | **1 sympy/algebras/quaternion.py** | 49 | 83| 231 | 2695 | 5265 | 
| 9 | **1 sympy/algebras/quaternion.py** | 85 | 107| 237 | 2932 | 5265 | 
| 10 | **1 sympy/algebras/quaternion.py** | 180 | 222| 453 | 3385 | 5265 | 
| 11 | **1 sympy/algebras/quaternion.py** | 224 | 250| 310 | 3695 | 5265 | 
| 12 | **1 sympy/algebras/quaternion.py** | 344 | 366| 257 | 3952 | 5265 | 
| 13 | **1 sympy/algebras/quaternion.py** | 424 | 448| 305 | 4257 | 5265 | 
| 14 | **1 sympy/algebras/quaternion.py** | 20 | 47| 280 | 4537 | 5265 | 
| 15 | **1 sympy/algebras/quaternion.py** | 393 | 422| 335 | 4872 | 5265 | 
| 16 | **1 sympy/algebras/quaternion.py** | 368 | 391| 263 | 5135 | 5265 | 
| 17 | 2 sympy/vector/orienters.py | 330 | 394| 432 | 5567 | 8265 | 
| 18 | **2 sympy/algebras/quaternion.py** | 312 | 342| 200 | 5767 | 8265 | 
| 19 | 2 sympy/vector/orienters.py | 68 | 99| 213 | 5980 | 8265 | 
| 20 | 3 sympy/physics/quantum/spin.py | 596 | 616| 196 | 6176 | 28636 | 
| 21 | 3 sympy/vector/orienters.py | 297 | 328| 349 | 6525 | 28636 | 
| 22 | 4 sympy/matrices/expressions/inverse.py | 71 | 95| 174 | 6699 | 29272 | 
| 23 | 5 sympy/polys/subresultants_qq_zz.py | 2269 | 2300| 303 | 7002 | 52895 | 
| 24 | 6 sympy/physics/quantum/qexpr.py | 1 | 25| 135 | 7137 | 55955 | 
| 25 | 6 sympy/polys/subresultants_qq_zz.py | 2118 | 2196| 814 | 7951 | 55955 | 
| 26 | 7 sympy/vector/coordsysrect.py | 933 | 986| 413 | 8364 | 64226 | 
| 27 | 8 sympy/physics/secondquant.py | 178 | 197| 162 | 8526 | 86784 | 
| 28 | 8 sympy/vector/coordsysrect.py | 438 | 490| 295 | 8821 | 86784 | 
| 29 | 9 sympy/matrices/expressions/blockmatrix.py | 1 | 20| 219 | 9040 | 90531 | 
| 30 | 10 examples/advanced/qft.py | 85 | 138| 607 | 9647 | 91846 | 
| 31 | 10 sympy/vector/coordsysrect.py | 1 | 23| 227 | 9874 | 91846 | 
| 32 | 10 sympy/physics/quantum/spin.py | 618 | 644| 234 | 10108 | 91846 | 
| 33 | 10 sympy/physics/quantum/spin.py | 646 | 678| 316 | 10424 | 91846 | 
| 34 | 11 sympy/physics/quantum/matrixcache.py | 83 | 103| 434 | 10858 | 92837 | 
| 35 | 12 sympy/plotting/pygletplot/plot_rotation.py | 1 | 28| 214 | 11072 | 93392 | 
| 36 | 13 sympy/matrices/dense.py | 870 | 910| 298 | 11370 | 105043 | 
| 37 | 14 sympy/parsing/autolev/test-examples/ruletest6.py | 1 | 37| 681 | 12051 | 105724 | 
| 38 | 14 sympy/vector/coordsysrect.py | 401 | 436| 231 | 12282 | 105724 | 
| 39 | 14 sympy/matrices/dense.py | 913 | 953| 293 | 12575 | 105724 | 
| 40 | 14 sympy/physics/quantum/spin.py | 1 | 59| 482 | 13057 | 105724 | 
| 41 | 14 sympy/polys/subresultants_qq_zz.py | 1217 | 1261| 492 | 13549 | 105724 | 
| 42 | 15 sympy/physics/vector/functions.py | 322 | 372| 1165 | 14714 | 112340 | 
| 43 | 16 sympy/plotting/pygletplot/util.py | 1 | 77| 560 | 15274 | 113788 | 
| 44 | 16 sympy/matrices/dense.py | 827 | 867| 296 | 15570 | 113788 | 
| 45 | 16 sympy/physics/quantum/spin.py | 680 | 687| 120 | 15690 | 113788 | 
| 46 | 17 examples/advanced/curvilinear_coordinates.py | 77 | 117| 327 | 16017 | 114734 | 
| 47 | 18 sympy/diffgeom/diffgeom.py | 1 | 20| 174 | 16191 | 129243 | 
| 48 | 18 sympy/physics/secondquant.py | 292 | 309| 151 | 16342 | 129243 | 
| 49 | 18 sympy/physics/quantum/spin.py | 422 | 511| 612 | 16954 | 129243 | 
| 50 | 18 sympy/polys/subresultants_qq_zz.py | 1556 | 1574| 224 | 17178 | 129243 | 
| 51 | 18 sympy/polys/subresultants_qq_zz.py | 1929 | 1953| 168 | 17346 | 129243 | 
| 52 | 19 sympy/polys/rootisolation.py | 728 | 782| 647 | 17993 | 149845 | 
| 53 | 20 sympy/physics/quantum/qubit.py | 391 | 464| 634 | 18627 | 155586 | 
| 54 | 20 sympy/polys/subresultants_qq_zz.py | 1457 | 1518| 548 | 19175 | 155586 | 
| 55 | 20 sympy/diffgeom/diffgeom.py | 246 | 258| 162 | 19337 | 155586 | 
| 56 | 21 sympy/core/backend.py | 1 | 24| 357 | 19694 | 155944 | 
| 57 | 22 sympy/matrices/expressions/matmul.py | 305 | 340| 229 | 19923 | 158703 | 
| 58 | 22 sympy/matrices/expressions/matmul.py | 78 | 136| 501 | 20424 | 158703 | 
| 59 | 23 sympy/functions/elementary/trigonometric.py | 800 | 849| 599 | 21023 | 183393 | 
| 60 | 24 sympy/assumptions/handlers/matrices.py | 50 | 99| 364 | 21387 | 187632 | 
| 61 | 25 sympy/matrices/expressions/transpose.py | 73 | 98| 142 | 21529 | 188230 | 


### Hint

```
@hamid-m @smichr I'd like to try my hands at this issue.
```

## Patch

```diff
diff --git a/sympy/algebras/quaternion.py b/sympy/algebras/quaternion.py
--- a/sympy/algebras/quaternion.py
+++ b/sympy/algebras/quaternion.py
@@ -529,7 +529,7 @@ def to_rotation_matrix(self, v=None):
 
         m10 = 2*s*(q.b*q.c + q.d*q.a)
         m11 = 1 - 2*s*(q.b**2 + q.d**2)
-        m12 = 2*s*(q.c*q.d + q.b*q.a)
+        m12 = 2*s*(q.c*q.d - q.b*q.a)
 
         m20 = 2*s*(q.b*q.d - q.c*q.a)
         m21 = 2*s*(q.c*q.d + q.b*q.a)

```

## Test Patch

```diff
diff --git a/sympy/algebras/tests/test_quaternion.py b/sympy/algebras/tests/test_quaternion.py
--- a/sympy/algebras/tests/test_quaternion.py
+++ b/sympy/algebras/tests/test_quaternion.py
@@ -96,12 +96,12 @@ def test_quaternion_conversions():
                                    2 * acos(sqrt(30)/30))
 
     assert q1.to_rotation_matrix() == Matrix([[-S(2)/3, S(2)/15, S(11)/15],
-                                     [S(2)/3, -S(1)/3, S(14)/15],
+                                     [S(2)/3, -S(1)/3, S(2)/3],
                                      [S(1)/3, S(14)/15, S(2)/15]])
 
     assert q1.to_rotation_matrix((1, 1, 1)) == Matrix([[-S(2)/3, S(2)/15, S(11)/15, S(4)/5],
-                                                  [S(2)/3, -S(1)/3, S(14)/15, -S(4)/15],
-                                                  [S(1)/3, S(14)/15, S(2)/15, -S(2)/5],
+                                                  [S(2)/3, -S(1)/3, S(2)/3, S(0)],
+                                                       [S(1)/3, S(14)/15, S(2)/15, -S(2)/5],
                                                   [S(0), S(0), S(0), S(1)]])
 
     theta = symbols("theta", real=True)
@@ -120,3 +120,19 @@ def test_quaternion_conversions():
                [sin(theta),  cos(theta), 0, -sin(theta) - cos(theta) + 1],
                [0,           0,          1,  0],
                [0,           0,          0,  1]])
+
+
+def test_quaternion_rotation_iss1593():
+    """
+    There was a sign mistake in the definition,
+    of the rotation matrix. This tests that particular sign mistake.
+    See issue 1593 for reference.
+    See wikipedia
+    https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Quaternion-derived_rotation_matrix
+    for the correct definition
+    """
+    q = Quaternion(cos(x/2), sin(x/2), 0, 0)
+    assert(trigsimp(q.to_rotation_matrix()) == Matrix([
+                [1,      0,      0],
+                [0, cos(x), -sin(x)],
+                [0, sin(x), cos(x)]]))

```


## Code snippets

### 1 - sympy/algebras/quaternion.py:

Start line: 489, End line: 552

```python
class Quaternion(Expr):

    def to_rotation_matrix(self, v=None):
        """Returns the equivalent rotation transformation matrix of the quaternion
        which represents rotation about the origin if v is not passed.

        Example
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

        Example
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
        m12 = 2*s*(q.c*q.d + q.b*q.a)

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
### 2 - sympy/algebras/quaternion.py:

Start line: 252, End line: 286

```python
class Quaternion(Expr):

    @staticmethod
    def _generic_mul(q1, q2):

        q1 = sympify(q1)
        q2 = sympify(q2)

        # None is a Quaternion:
        if not isinstance(q1, Quaternion) and not isinstance(q2, Quaternion):
            return q1 * q2

        # If q1 is a number or a sympy expression instead of a quaternion
        if not isinstance(q1, Quaternion):
            if q2.real_field:
                if q1.is_complex:
                    return q2 * Quaternion(re(q1), im(q1), 0, 0)
                else:
                    return Mul(q1, q2)
            else:
                return Quaternion(q1 * q2.a, q1 * q2.b, q1 * q2.c, q1 * q2.d)


        # If q2 is a number or a sympy expression instead of a quaternion
        if not isinstance(q2, Quaternion):
            if q1.real_field:
                if q2.is_complex:
                    return q1 * Quaternion(re(q2), im(q2), 0, 0)
                else:
                    return Mul(q1, q2)
            else:
                return Quaternion(q2 * q1.a, q2 * q1.b, q2 * q1.c, q2 * q1.d)

        return Quaternion(-q1.b*q2.b - q1.c*q2.c - q1.d*q2.d + q1.a*q2.a,
                          q1.b*q2.a + q1.c*q2.d - q1.d*q2.c + q1.a*q2.b,
                          -q1.b*q2.d + q1.c*q2.a + q1.d*q2.b + q1.a*q2.c,
                          q1.b*q2.c - q1.c*q2.b + q1.d*q2.a + q1.a * q2.d)
```
### 3 - sympy/algebras/quaternion.py:

Start line: 143, End line: 178

```python
class Quaternion(Expr):

    @staticmethod
    def __copysign(x, y):

        # Takes the sign from the second term and sets the sign of the first
        # without altering the magnitude.

        if y == 0:
            return 0
        return x if x*y > 0 else -x

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

    def _eval_Integral(self, *args):
        return self.integrate(*args)

    def _eval_diff(self, *symbols, **kwargs):
        return self.diff(*symbols)
```
### 4 - sympy/algebras/quaternion.py:

Start line: 109, End line: 141

```python
class Quaternion(Expr):

    @classmethod
    def from_rotation_matrix(cls, M):
        """Returns the equivalent quaternion of a matrix. The quaternion will be normalized
        only if the matrix is special orthogonal (orthogonal and det(M) = 1).

        Example
        ========

        >>> from sympy.algebras.quaternion import Quaternion
        >>> from sympy import Matrix, symbols, cos, sin, trigsimp
        >>> x = symbols('x')
        >>> M = Matrix([[cos(x), -sin(x), 0], [sin(x), cos(x), 0], [0, 0, 1]])
        >>> q = trigsimp(Quaternion.from_rotation_matrix(M))
        >>> q
        sqrt(2)*sqrt(cos(x) + 1)/2 + 0*i + 0*j + sqrt(-2*cos(x) + 2)/2*k
        """

        absQ = M.det()**Rational(1, 3)

        a = sqrt(absQ + M[0, 0] + M[1, 1] + M[2, 2]) / 2
        b = sqrt(absQ + M[0, 0] - M[1, 1] - M[2, 2]) / 2
        c = sqrt(absQ - M[0, 0] + M[1, 1] - M[2, 2]) / 2
        d = sqrt(absQ - M[0, 0] - M[1, 1] + M[2, 2]) / 2

        try:
            b = Quaternion.__copysign(b, M[2, 1] - M[1, 2])
            c = Quaternion.__copysign(c, M[0, 2] - M[2, 0])
            d = Quaternion.__copysign(d, M[1, 0] - M[0, 1])

        except Exception:
            pass

        return Quaternion(a, b, c, d)
```
### 5 - sympy/algebras/quaternion.py:

Start line: 1, End line: 17

```python
# References :
# http://www.euclideanspace.com/maths/algebra/realNormedAlgebra/quaternions/
# https://en.wikipedia.org/wiki/Quaternion
from __future__ import print_function

from sympy.core.expr import Expr
from sympy import Rational
from sympy import re, im, conjugate
from sympy import sqrt, sin, cos, acos, asin, exp, ln
from sympy import trigsimp
from sympy import diff, integrate
from sympy import Matrix, Add, Mul
from sympy import symbols, sympify
from sympy.printing.latex import latex
from sympy.printing import StrPrinter
from sympy.core.numbers import Integer
from sympy.core.compatibility import SYMPY_INTS
```
### 6 - sympy/algebras/quaternion.py:

Start line: 450, End line: 487

```python
class Quaternion(Expr):

    def to_axis_angle(self):
        """Returns the axis and angle of rotation of a quaternion

        Example
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
        try:
            # Skips it if it doesn't know whether q.a is negative
            if q.a < 0:
                # avoid error with acos
                # axis and angle of rotation of q and q*-1 will be the same
                q = q * -1
        except BaseException:
            pass

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
### 7 - sympy/algebras/quaternion.py:

Start line: 288, End line: 310

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
### 8 - sympy/algebras/quaternion.py:

Start line: 49, End line: 83

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
### 9 - sympy/algebras/quaternion.py:

Start line: 85, End line: 107

```python
class Quaternion(Expr):

    @classmethod
    def from_axis_angle(cls, vector, angle):
        """Returns a rotation quaternion given the axis and the angle of rotation.

        Example
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
        s = sin(angle * Rational(1, 2))
        a = cos(angle * Rational(1, 2))
        b = x * s
        c = y * s
        d = z * s

        return cls(a, b, c, d).normalize()
```
### 10 - sympy/algebras/quaternion.py:

Start line: 180, End line: 222

```python
class Quaternion(Expr):

    def add(self, other):
        """Adds quaternions.

        Example
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
        ========

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
            if q1.real_field:
                if q2.is_complex:
                    return Quaternion(re(q2) + q1.a, im(q2) + q1.b, q1.c, q1.d)
                else:
                    # q2 is something strange, do not evaluate:
                    return Add(q1, q2)
            else:
                return Quaternion(q1.a + q2, q1.b, q1.c, q1.d)

        return Quaternion(q1.a + q2.a, q1.b + q2.b, q1.c + q2.c, q1.d
                          + q2.d)
```
### 11 - sympy/algebras/quaternion.py:

Start line: 224, End line: 250

```python
class Quaternion(Expr):

    def mul(self, other):
        """Multiplies quaternions.

        Example
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
        ========
        >>> from sympy.algebras.quaternion import Quaternion
        >>> from sympy import I
        >>> q3 = Quaternion(3 + 4*I, 2 + 5*I, 0, 7 + 8*I, real_field = False)
        >>> q3.mul(2 + 3*I)
        (2 + 3*I)*(3 + 4*I) + (2 + 3*I)*(2 + 5*I)*i + 0*j + (2 + 3*I)*(7 + 8*I)*k
        """
        return self._generic_mul(self, other)
```
### 12 - sympy/algebras/quaternion.py:

Start line: 344, End line: 366

```python
class Quaternion(Expr):

    def exp(self):
        """Returns the exponential of q (e^q).

        Example
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
### 13 - sympy/algebras/quaternion.py:

Start line: 424, End line: 448

```python
class Quaternion(Expr):

    @staticmethod
    def rotate_point(pin, r):
        """Returns the coordinates of the point pin(a 3 tuple) after rotation.

        Example
        ========

        >>> from sympy.algebras.quaternion import Quaternion
        >>> from sympy import symbols, trigsimp, cos, sin
        >>> x = symbols('x')
        >>> q = Quaternion(cos(x/2), 0, 0, sin(x/2))
        >>> trigsimp(Quaternion.rotate_point((1, 1, 1), q))
        (sqrt(2)*cos(x + pi/4), sqrt(2)*sin(x + pi/4), 1)
        >>> (axis, angle) = q.to_axis_angle()
        >>> trigsimp(Quaternion.rotate_point((1, 1, 1), (axis, angle)))
        (sqrt(2)*cos(x + pi/4), sqrt(2)*sin(x + pi/4), 1)
        """
        if isinstance(r, tuple):
            # if r is of the form (vector, angle)
            q = Quaternion.from_axis_angle(r[0], r[1])
        else:
            # if r is a quaternion
            q = r.normalize()
        pout = q * Quaternion(0, pin[0], pin[1], pin[2]) * conjugate(q)
        return (pout.b, pout.c, pout.d)
```
### 14 - sympy/algebras/quaternion.py:

Start line: 20, End line: 47

```python
class Quaternion(Expr):
    """Provides basic quaternion operations.
    Quaternion objects can be instantiated as Quaternion(a, b, c, d)
    as in (a + b*i + c*j + d*k).

    Example
    ========

    >>> from sympy.algebras.quaternion import Quaternion
    >>> q = Quaternion(1, 2, 3, 4)
    >>> q
    1 + 2*i + 3*j + 4*k

    Quaternions over complex fields can be defined as :
    ========
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
### 15 - sympy/algebras/quaternion.py:

Start line: 393, End line: 422

```python
class Quaternion(Expr):

    def pow_cos_sin(self, p):
        """Computes the pth power in the cos-sin form.

        Example
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

    def diff(self, *args):
        return Quaternion(diff(self.a, *args), diff(self.b, *args),
                          diff(self.c, *args), diff(self.d, *args))

    def integrate(self, *args):
        # TODO: is this expression correct?
        return Quaternion(integrate(self.a, *args), integrate(self.b, *args),
                          integrate(self.c, *args), integrate(self.d, *args))
```
### 16 - sympy/algebras/quaternion.py:

Start line: 368, End line: 391

```python
class Quaternion(Expr):

    def _ln(self):
        """Returns the natural logarithm of the quaternion (_ln(q)).

        Example
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
### 18 - sympy/algebras/quaternion.py:

Start line: 312, End line: 342

```python
class Quaternion(Expr):

    def pow(self, p):
        """Finds the pth power of the quaternion.
        Returns the inverse if p = -1.

        Example
        ========

        >>> from sympy.algebras.quaternion import Quaternion
        >>> q = Quaternion(1, 2, 3, 4)
        >>> q.pow(4)
        668 + (-224)*i + (-336)*j + (-448)*k
        """
        q = self
        if p == -1:
            return q.inverse()
        res = 1

        if p < 0:
            q, p = q.inverse(), -p

        if not (isinstance(p, (Integer, SYMPY_INTS))):
            return NotImplemented

        while p > 0:
            if p & 1:
                res = q * res

            p = p >> 1
            q = q * q

        return res
```
