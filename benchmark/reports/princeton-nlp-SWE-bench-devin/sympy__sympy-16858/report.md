# sympy__sympy-16858

| **sympy/sympy** | `6ffc2f04ad820e3f592b2107e66a16fd4585ac02` |
| ---- | ---- |
| **No of patches** | 2 |
| **All found context length** | - |
| **Any found context length** | 6578 |
| **Avg pos** | 279.0 |
| **Min pos** | 13 |
| **Max pos** | 65 |
| **Top file pos** | 8 |
| **Missing snippets** | 28 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sympy/stats/crv_types.py b/sympy/stats/crv_types.py
--- a/sympy/stats/crv_types.py
+++ b/sympy/stats/crv_types.py
@@ -163,6 +163,9 @@ def rv(symbol, cls, args):
 class ArcsinDistribution(SingleContinuousDistribution):
     _argnames = ('a', 'b')
 
+    def set(self):
+        return Interval(self.a, self.b)
+
     def pdf(self, x):
         return 1/(pi*sqrt((x - self.a)*(self.b - x)))
 
@@ -871,6 +874,8 @@ def ChiSquared(name, k):
 class DagumDistribution(SingleContinuousDistribution):
     _argnames = ('p', 'a', 'b')
 
+    set = Interval(0, oo)
+
     @staticmethod
     def check(p, a, b):
         _value_check(p > 0, "Shape parameter p must be positive.")
@@ -1205,6 +1210,13 @@ def FDistribution(name, d1, d2):
 class FisherZDistribution(SingleContinuousDistribution):
     _argnames = ('d1', 'd2')
 
+    set = Interval(-oo, oo)
+
+    @staticmethod
+    def check(d1, d2):
+        _value_check(d1 > 0, "Degree of freedom d1 must be positive.")
+        _value_check(d2 > 0, "Degree of freedom d2 must be positive.")
+
     def pdf(self, x):
         d1, d2 = self.d1, self.d2
         return (2*d1**(d1/2)*d2**(d2/2) / beta_fn(d1/2, d2/2) *
@@ -1276,6 +1288,11 @@ class FrechetDistribution(SingleContinuousDistribution):
 
     set = Interval(0, oo)
 
+    @staticmethod
+    def check(a, s, m):
+        _value_check(a > 0, "Shape parameter alpha must be positive.")
+        _value_check(s > 0, "Scale parameter s must be positive.")
+
     def __new__(cls, a, s=1, m=0):
         a, s, m = list(map(sympify, (a, s, m)))
         return Basic.__new__(cls, a, s, m)
@@ -1551,6 +1568,10 @@ class GumbelDistribution(SingleContinuousDistribution):
 
     set = Interval(-oo, oo)
 
+    @staticmethod
+    def check(beta, mu):
+        _value_check(beta > 0, "Scale parameter beta must be positive.")
+
     def pdf(self, x):
         beta, mu = self.beta, self.mu
         z = (x - mu)/beta
@@ -1564,7 +1585,7 @@ def _characteristic_function(self, t):
         return gamma(1 - I*self.beta*t) * exp(I*self.mu*t)
 
     def _moment_generating_function(self, t):
-        return gamma(1 - self.beta*t) * exp(I*self.mu*t)
+        return gamma(1 - self.beta*t) * exp(self.mu*t)
 
 def Gumbel(name, beta, mu):
     r"""
@@ -1765,6 +1786,13 @@ def Kumaraswamy(name, a, b):
 class LaplaceDistribution(SingleContinuousDistribution):
     _argnames = ('mu', 'b')
 
+    set = Interval(-oo, oo)
+
+    @staticmethod
+    def check(mu, b):
+        _value_check(b > 0, "Scale parameter b must be positive.")
+        _value_check(mu.is_real, "Location parameter mu should be real")
+
     def pdf(self, x):
         mu, b = self.mu, self.b
         return 1/(2*b)*exp(-Abs(x - mu)/b)
@@ -1852,6 +1880,12 @@ def Laplace(name, mu, b):
 class LogisticDistribution(SingleContinuousDistribution):
     _argnames = ('mu', 's')
 
+    set = Interval(-oo, oo)
+
+    @staticmethod
+    def check(mu, s):
+        _value_check(s > 0, "Scale parameter s must be positive.")
+
     def pdf(self, x):
         mu, s = self.mu, self.s
         return exp(-(x - mu)/s)/(s*(1 + exp(-(x - mu)/s))**2)
@@ -1864,7 +1898,7 @@ def _characteristic_function(self, t):
         return Piecewise((exp(I*t*self.mu) * pi*self.s*t / sinh(pi*self.s*t), Ne(t, 0)), (S.One, True))
 
     def _moment_generating_function(self, t):
-        return exp(self.mu*t) * Beta(1 - self.s*t, 1 + self.s*t)
+        return exp(self.mu*t) * beta_fn(1 - self.s*t, 1 + self.s*t)
 
     def _quantile(self, p):
         return self.mu - self.s*log(-S.One + S.One/p)
@@ -2015,6 +2049,10 @@ class MaxwellDistribution(SingleContinuousDistribution):
 
     set = Interval(0, oo)
 
+    @staticmethod
+    def check(a):
+        _value_check(a > 0, "Parameter a must be positive.")
+
     def pdf(self, x):
         a = self.a
         return sqrt(2/pi)*x**2*exp(-x**2/(2*a**2))/a**3
@@ -2085,6 +2123,11 @@ class NakagamiDistribution(SingleContinuousDistribution):
 
     set = Interval(0, oo)
 
+    @staticmethod
+    def check(mu, omega):
+        _value_check(mu >= S.Half, "Shape parameter mu must be greater than equal to 1/2.")
+        _value_check(omega > 0, "Spread parameter omega must be positive.")
+
     def pdf(self, x):
         mu, omega = self.mu, self.omega
         return 2*mu**mu/(gamma(mu)*omega**mu)*x**(2*mu - 1)*exp(-mu/omega*x**2)
@@ -2385,6 +2428,10 @@ class QuadraticUDistribution(SingleContinuousDistribution):
     def set(self):
         return Interval(self.a, self.b)
 
+    @staticmethod
+    def check(a, b):
+        _value_check(b > a, "Parameter b must be in range (%s, oo)."%(a))
+
     def pdf(self, x):
         a, b = self.a, self.b
         alpha = 12 / (b-a)**3
@@ -2553,6 +2600,10 @@ class RayleighDistribution(SingleContinuousDistribution):
 
     set = Interval(0, oo)
 
+    @staticmethod
+    def check(sigma):
+        _value_check(sigma > 0, "Scale parameter sigma must be positive.")
+
     def pdf(self, x):
         sigma = self.sigma
         return x/sigma**2*exp(-x**2/(2*sigma**2))
@@ -2690,6 +2741,12 @@ def ShiftedGompertz(name, b, eta):
 class StudentTDistribution(SingleContinuousDistribution):
     _argnames = ('nu',)
 
+    set = Interval(-oo, oo)
+
+    @staticmethod
+    def check(nu):
+        _value_check(nu > 0, "Degrees of freedom nu must be positive.")
+
     def pdf(self, x):
         nu = self.nu
         return 1/(sqrt(nu)*beta_fn(S(1)/2, nu/2))*(1 + x**2/nu)**(-(nu + 1)/2)
@@ -2770,6 +2827,19 @@ def StudentT(name, nu):
 class TrapezoidalDistribution(SingleContinuousDistribution):
     _argnames = ('a', 'b', 'c', 'd')
 
+    @property
+    def set(self):
+        return Interval(self.a, self.d)
+
+    @staticmethod
+    def check(a, b, c, d):
+        _value_check(a < d, "Lower bound parameter a < %s. a = %s"%(d, a))
+        _value_check((a <= b, b < c),
+        "Level start parameter b must be in range [%s, %s). b = %s"%(a, c, b))
+        _value_check((b < c, c <= d),
+        "Level end parameter c must be in range (%s, %s]. c = %s"%(b, d, c))
+        _value_check(d >= c, "Upper bound parameter d > %s. d = %s"%(c, d))
+
     def pdf(self, x):
         a, b, c, d = self.a, self.b, self.c, self.d
         return Piecewise(
@@ -2850,6 +2920,16 @@ def Trapezoidal(name, a, b, c, d):
 class TriangularDistribution(SingleContinuousDistribution):
     _argnames = ('a', 'b', 'c')
 
+    @property
+    def set(self):
+        return Interval(self.a, self.b)
+
+    @staticmethod
+    def check(a, b, c):
+        _value_check(b > a, "Parameter b > %s. b = %s"%(a, b))
+        _value_check((a <= c, c <= b),
+        "Parameter c must be in range [%s, %s]. c = %s"%(a, b, c))
+
     def pdf(self, x):
         a, b, c = self.a, self.b, self.c
         return Piecewise(
@@ -2864,7 +2944,7 @@ def _characteristic_function(self, t):
 
     def _moment_generating_function(self, t):
         a, b, c = self.a, self.b, self.c
-        return 2 * ((b - c) * exp(a * t) - (b - a) * exp(c * t) + (c + a) * exp(b * t)) / (
+        return 2 * ((b - c) * exp(a * t) - (b - a) * exp(c * t) + (c - a) * exp(b * t)) / (
         (b - a) * (c - a) * (b - c) * t ** 2)
 
 
@@ -2940,6 +3020,14 @@ def Triangular(name, a, b, c):
 class UniformDistribution(SingleContinuousDistribution):
     _argnames = ('left', 'right')
 
+    @property
+    def set(self):
+        return Interval(self.left, self.right)
+
+    @staticmethod
+    def check(left, right):
+        _value_check(left < right, "Lower limit should be less than Upper limit.")
+
     def pdf(self, x):
         left, right = self.left, self.right
         return Piecewise(
@@ -3047,6 +3135,11 @@ class UniformSumDistribution(SingleContinuousDistribution):
     def set(self):
         return Interval(0, self.n)
 
+    @staticmethod
+    def check(n):
+        _value_check((n > 0, n.is_integer),
+        "Parameter n must be positive integer.")
+
     def pdf(self, x):
         n = self.n
         k = Dummy("k")
@@ -3292,6 +3385,10 @@ class WignerSemicircleDistribution(SingleContinuousDistribution):
     def set(self):
         return Interval(-self.R, self.R)
 
+    @staticmethod
+    def check(R):
+        _value_check(R > 0, "Radius R must be positive.")
+
     def pdf(self, x):
         R = self.R
         return 2/(pi*R**2)*sqrt(R**2 - x**2)
diff --git a/sympy/stats/joint_rv_types.py b/sympy/stats/joint_rv_types.py
--- a/sympy/stats/joint_rv_types.py
+++ b/sympy/stats/joint_rv_types.py
@@ -76,7 +76,8 @@ def set(self):
         k = len(self.mu)
         return S.Reals**k
 
-    def check(self, mu, sigma):
+    @staticmethod
+    def check(mu, sigma):
         _value_check(len(mu) == len(sigma.col(0)),
             "Size of the mean vector and covariance matrix are incorrect.")
         #check if covariance matrix is positive definite or not.
@@ -117,7 +118,8 @@ def set(self):
         k = len(self.mu)
         return S.Reals**k
 
-    def check(self, mu, sigma):
+    @staticmethod
+    def check(mu, sigma):
         _value_check(len(mu) == len(sigma.col(0)),
             "Size of the mean vector and covariance matrix are incorrect.")
         #check if covariance matrix is positive definite or not.
@@ -151,7 +153,8 @@ def set(self):
         k = len(self.mu)
         return S.Reals**k
 
-    def check(self, mu, sigma, v):
+    @staticmethod
+    def check(mu, sigma, v):
         _value_check(len(mu) == len(sigma.col(0)),
             "Size of the location vector and shape matrix are incorrect.")
         #check if covariance matrix is positive definite or not.
@@ -196,7 +199,8 @@ class NormalGammaDistribution(JointDistribution):
     _argnames = ['mu', 'lamda', 'alpha', 'beta']
     is_Continuous=True
 
-    def check(self, mu, lamda, alpha, beta):
+    @staticmethod
+    def check(mu, lamda, alpha, beta):
         _value_check(mu.is_real, "Location must be real.")
         _value_check(lamda > 0, "Lambda must be positive")
         _value_check(alpha > 0, "alpha must be positive")
@@ -258,7 +262,8 @@ class MultivariateBetaDistribution(JointDistribution):
     _argnames = ['alpha']
     is_Continuous = True
 
-    def check(self, alpha):
+    @staticmethod
+    def check(alpha):
         _value_check(len(alpha) >= 2, "At least two categories should be passed.")
         for a_k in alpha:
             _value_check((a_k > 0) != False, "Each concentration parameter"
@@ -331,7 +336,8 @@ class MultivariateEwensDistribution(JointDistribution):
     is_Discrete = True
     is_Continuous = False
 
-    def check(self, n, theta):
+    @staticmethod
+    def check(n, theta):
         _value_check(isinstance(n, Integer) and (n > 0) == True,
                         "sample size should be positive integer.")
         _value_check(theta.is_positive, "mutation rate should be positive.")
@@ -403,7 +409,8 @@ class MultinomialDistribution(JointDistribution):
     is_Continuous=False
     is_Discrete = True
 
-    def check(self, n, p):
+    @staticmethod
+    def check(n, p):
         _value_check(n > 0,
                         "number of trials must be a positve integer")
         for p_k in p:
@@ -471,7 +478,8 @@ class NegativeMultinomialDistribution(JointDistribution):
     is_Continuous=False
     is_Discrete = True
 
-    def check(self, k0, p):
+    @staticmethod
+    def check(k0, p):
         _value_check(k0 > 0,
                         "number of failures must be a positve integer")
         for p_k in p:

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sympy/stats/crv_types.py | 166 | 166 | - | 8 | -
| sympy/stats/crv_types.py | 874 | 874 | 13 | 8 | 6578
| sympy/stats/crv_types.py | 1208 | 1208 | - | 8 | -
| sympy/stats/crv_types.py | 1279 | 1279 | 34 | 8 | 14456
| sympy/stats/crv_types.py | 1554 | 1554 | 49 | 8 | 17992
| sympy/stats/crv_types.py | 1567 | 1567 | 49 | 8 | 17992
| sympy/stats/crv_types.py | 1768 | 1768 | - | 8 | -
| sympy/stats/crv_types.py | 1855 | 1855 | - | 8 | -
| sympy/stats/crv_types.py | 1867 | 1867 | - | 8 | -
| sympy/stats/crv_types.py | 2018 | 2018 | - | 8 | -
| sympy/stats/crv_types.py | 2088 | 2088 | 57 | 8 | 19793
| sympy/stats/crv_types.py | 2388 | 2388 | - | 8 | -
| sympy/stats/crv_types.py | 2556 | 2556 | 55 | 8 | 19425
| sympy/stats/crv_types.py | 2693 | 2693 | 40 | 8 | 16069
| sympy/stats/crv_types.py | 2773 | 2773 | - | 8 | -
| sympy/stats/crv_types.py | 2853 | 2853 | - | 8 | -
| sympy/stats/crv_types.py | 2867 | 2867 | - | 8 | -
| sympy/stats/crv_types.py | 2943 | 2943 | 42 | 8 | 16583
| sympy/stats/crv_types.py | 3050 | 3050 | 30 | 8 | 13039
| sympy/stats/crv_types.py | 3295 | 3295 | - | 8 | -
| sympy/stats/joint_rv_types.py | 79 | 79 | 39 | 14 | 15894
| sympy/stats/joint_rv_types.py | 120 | 120 | 25 | 14 | 10721
| sympy/stats/joint_rv_types.py | 154 | 154 | - | 14 | -
| sympy/stats/joint_rv_types.py | 199 | 199 | 60 | 14 | 20299
| sympy/stats/joint_rv_types.py | 261 | 261 | - | 14 | -
| sympy/stats/joint_rv_types.py | 334 | 334 | 65 | 14 | 21863
| sympy/stats/joint_rv_types.py | 406 | 406 | - | 14 | -
| sympy/stats/joint_rv_types.py | 474 | 474 | - | 14 | -


## Problem Statement

```
Added missing checks and attributes to sympy.stats
<!-- Your title above should be a short description of what
was changed. Do not include the issue number in the title. -->

#### References to other Issues or PRs
<!-- If this pull request fixes an issue, write "Fixes #NNNN" in that exact
format, e.g. "Fixes #1234". See
https://github.com/blog/1506-closing-issues-via-pull-requests . Please also
write a comment on that issue linking back to this pull request once it is
open. -->
N/A


#### Brief description of what is fixed or changed
Missing checks for parameters and set
attributes have been added to various
distributions to enhance consistency
and correctness.


#### Other comments
These changes are made for enhancement of the code. This PR is made for receiving regular feedback on the code additions.
Status - Work In Progress
Please discuss with me on the changes I have made, so that I can present my view if I haven't made satisfactory changes. 

#### Release Notes

<!-- Write the release notes for this release below. See
https://github.com/sympy/sympy/wiki/Writing-Release-Notes for more information
on how to write release notes. The bot will check your release notes
automatically to see if they are formatted correctly. -->

<!-- BEGIN RELEASE NOTES -->
* stats
  * missing checks and attributes added to sympy.stats for distributions.
<!-- END RELEASE NOTES -->


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 setup.py | 307 | 382| 738 | 738 | 3544 | 
| 2 | 2 sympy/__init__.py | 1 | 93| 616 | 1354 | 4160 | 
| 3 | 2 setup.py | 79 | 163| 782 | 2136 | 4160 | 
| 4 | 3 release/fabfile.py | 851 | 895| 552 | 2688 | 15342 | 
| 5 | 3 release/fabfile.py | 797 | 849| 612 | 3300 | 15342 | 
| 6 | 4 sympy/stats/frv.py | 189 | 231| 310 | 3610 | 18098 | 
| 7 | 4 setup.py | 384 | 434| 476 | 4086 | 18098 | 
| 8 | 5 sympy/integrals/risch.py | 600 | 637| 393 | 4479 | 35903 | 
| 9 | 6 sympy/stats/__init__.py | 1 | 77| 708 | 5187 | 36611 | 
| 10 | 6 release/fabfile.py | 356 | 431| 765 | 5952 | 36611 | 
| 11 | 7 sympy/stats/drv.py | 1 | 24| 221 | 6173 | 39441 | 
| 12 | **8 sympy/stats/crv_types.py** | 230 | 254| 203 | 6376 | 63160 | 
| **-> 13 <-** | **8 sympy/stats/crv_types.py** | 867 | 887| 202 | 6578 | 63160 | 
| 14 | 9 sympy/core/add.py | 486 | 513| 391 | 6969 | 71817 | 
| 15 | **9 sympy/stats/crv_types.py** | 318 | 343| 199 | 7168 | 71817 | 
| 16 | 10 sympy/stats/joint_rv.py | 294 | 310| 136 | 7304 | 74574 | 
| 17 | 11 sympy/utilities/runtests.py | 64 | 77| 224 | 7528 | 94941 | 
| 18 | 12 doc/src/conf.py | 1 | 103| 832 | 8360 | 96687 | 
| 19 | 12 sympy/utilities/runtests.py | 1 | 63| 695 | 9055 | 96687 | 
| 20 | **12 sympy/stats/crv_types.py** | 2167 | 2198| 262 | 9317 | 96687 | 
| 21 | 12 release/fabfile.py | 725 | 748| 173 | 9490 | 96687 | 
| 22 | 13 sympy/core/compatibility.py | 1 | 65| 585 | 10075 | 104539 | 
| 23 | 13 setup.py | 165 | 194| 231 | 10306 | 104539 | 
| 24 | 13 sympy/stats/drv.py | 341 | 373| 275 | 10581 | 104539 | 
| **-> 25 <-** | **14 sympy/stats/joint_rv_types.py** | 108 | 125| 140 | 10721 | 109264 | 
| 26 | 15 isympy.py | 171 | 283| 811 | 11532 | 111984 | 
| 27 | 16 sympy/stats/rv_interface.py | 1 | 12| 197 | 11729 | 113841 | 
| 28 | **16 sympy/stats/crv_types.py** | 2624 | 2640| 132 | 11861 | 113841 | 
| 29 | 17 bin/coverage_doctest.py | 582 | 674| 930 | 12791 | 119438 | 
| **-> 30 <-** | **17 sympy/stats/crv_types.py** | 3039 | 3068| 248 | 13039 | 119438 | 
| 31 | 18 doc/api/conf.py | 1 | 135| 927 | 13966 | 120433 | 
| 32 | 19 sympy/stats/drv_types.py | 449 | 472| 167 | 14133 | 123567 | 
| 33 | 19 release/fabfile.py | 750 | 772| 130 | 14263 | 123567 | 
| **-> 34 <-** | **19 sympy/stats/crv_types.py** | 1271 | 1290| 193 | 14456 | 123567 | 
| 35 | **19 sympy/stats/crv_types.py** | 2463 | 2492| 287 | 14743 | 123567 | 
| 36 | 19 sympy/integrals/risch.py | 340 | 391| 624 | 15367 | 123567 | 
| 37 | **19 sympy/stats/crv_types.py** | 493 | 509| 125 | 15492 | 123567 | 
| 38 | 20 sympy/stats/frv_types.py | 263 | 279| 186 | 15678 | 126282 | 
| **-> 39 <-** | **20 sympy/stats/joint_rv_types.py** | 66 | 93| 216 | 15894 | 126282 | 
| **-> 40 <-** | **20 sympy/stats/crv_types.py** | 2686 | 2703| 175 | 16069 | 126282 | 
| 41 | 20 sympy/stats/drv_types.py | 44 | 64| 167 | 16236 | 126282 | 
| **-> 42 <-** | **20 sympy/stats/crv_types.py** | 2936 | 2977| 347 | 16583 | 126282 | 
| 43 | 20 sympy/core/add.py | 1 | 21| 137 | 16720 | 126282 | 
| 44 | **20 sympy/stats/crv_types.py** | 3208 | 3227| 145 | 16865 | 126282 | 
| 45 | **20 sympy/stats/crv_types.py** | 2291 | 2327| 279 | 17144 | 126282 | 
| 46 | 20 sympy/utilities/runtests.py | 1136 | 1142| 130 | 17274 | 126282 | 
| 47 | **20 sympy/stats/crv_types.py** | 1120 | 1143| 208 | 17482 | 126282 | 
| 48 | 21 sympy/core/numbers.py | 1 | 39| 340 | 17822 | 156263 | 
| **-> 49 <-** | **21 sympy/stats/crv_types.py** | 1545 | 1567| 170 | 17992 | 156263 | 
| 50 | 21 sympy/integrals/risch.py | 393 | 425| 370 | 18362 | 156263 | 
| 51 | 21 sympy/utilities/runtests.py | 1144 | 1165| 268 | 18630 | 156263 | 
| 52 | 21 sympy/core/add.py | 594 | 646| 383 | 19013 | 156263 | 
| 53 | 22 sympy/assumptions/__init__.py | 1 | 4| 0 | 19013 | 156300 | 
| 54 | **22 sympy/stats/crv_types.py** | 1012 | 1046| 201 | 19214 | 156300 | 
| **-> 55 <-** | **22 sympy/stats/crv_types.py** | 2547 | 2570| 211 | 19425 | 156300 | 
| 56 | **22 sympy/stats/crv_types.py** | 562 | 588| 220 | 19645 | 156300 | 
| **-> 57 <-** | **22 sympy/stats/crv_types.py** | 2079 | 2096| 148 | 19793 | 156300 | 
| 58 | 22 sympy/core/add.py | 648 | 661| 139 | 19932 | 156300 | 
| 59 | 22 sympy/integrals/risch.py | 427 | 440| 133 | 20065 | 156300 | 
| **-> 60 <-** | **22 sympy/stats/joint_rv_types.py** | 191 | 215| 234 | 20299 | 156300 | 
| 61 | 22 sympy/core/compatibility.py | 927 | 957| 240 | 20539 | 156300 | 
| 62 | 22 sympy/stats/drv_types.py | 370 | 398| 253 | 20792 | 156300 | 
| 63 | 22 sympy/core/add.py | 663 | 676| 139 | 20931 | 156300 | 
| 64 | 23 bin/authors_update.py | 71 | 135| 766 | 21697 | 157826 | 
| **-> 65 <-** | **23 sympy/stats/joint_rv_types.py** | 323 | 344| 166 | 21863 | 157826 | 
| 66 | **23 sympy/stats/crv_types.py** | 782 | 811| 238 | 22101 | 157826 | 
| 67 | **23 sympy/stats/crv_types.py** | 402 | 421| 214 | 22315 | 157826 | 
| 68 | 24 sympy/core/mul.py | 1 | 34| 205 | 22520 | 172578 | 
| 69 | 24 sympy/utilities/runtests.py | 1363 | 1454| 802 | 23322 | 172578 | 
| 70 | **24 sympy/stats/crv_types.py** | 635 | 658| 234 | 23556 | 172578 | 
| 71 | 25 sympy/stats/symbolic_probability.py | 228 | 259| 249 | 23805 | 175858 | 
| 72 | 25 release/fabfile.py | 987 | 1062| 652 | 24457 | 175858 | 


### Hint

```
:white_check_mark:

Hi, I am the [SymPy bot](https://github.com/sympy/sympy-bot) (v147). I'm here to help you write a release notes entry. Please read the [guide on how to write release notes](https://github.com/sympy/sympy/wiki/Writing-Release-Notes).



Your release notes are in good order.

Here is what the release notes will look like:
* stats
  * missing checks and attributes added to sympy.stats for distributions. ([#16571](https://github.com/sympy/sympy/pull/16571) by [@czgdp1807](https://github.com/czgdp1807))

This will be added to https://github.com/sympy/sympy/wiki/Release-Notes-for-1.5.

Note: This comment will be updated with the latest check if you edit the pull request. You need to reload the page to see it. <details><summary>Click here to see the pull request description that was parsed.</summary>

    <!-- Your title above should be a short description of what
    was changed. Do not include the issue number in the title. -->

    #### References to other Issues or PRs
    <!-- If this pull request fixes an issue, write "Fixes #NNNN" in that exact
    format, e.g. "Fixes #1234". See
    https://github.com/blog/1506-closing-issues-via-pull-requests . Please also
    write a comment on that issue linking back to this pull request once it is
    open. -->
    N/A


    #### Brief description of what is fixed or changed
    Missing checks for parameters and set
    attributes have been added to various
    distributions to enhance consistency
    and correctness.


    #### Other comments
    These changes are made for enhancement of the code. This PR is made for receiving regular feedback on the code additions.
    Status - Work In Progress
    Please discuss with me on the changes I have made, so that I can present my view if I haven't made satisfactory changes. 

    #### Release Notes

    <!-- Write the release notes for this release below. See
    https://github.com/sympy/sympy/wiki/Writing-Release-Notes for more information
    on how to write release notes. The bot will check your release notes
    automatically to see if they are formatted correctly. -->

    <!-- BEGIN RELEASE NOTES -->
    * stats
      * missing checks and attributes added to sympy.stats for distributions.
    <!-- END RELEASE NOTES -->


</details><p>

# [Codecov](https://codecov.io/gh/sympy/sympy/pull/16571?src=pr&el=h1) Report
> Merging [#16571](https://codecov.io/gh/sympy/sympy/pull/16571?src=pr&el=desc) into [master](https://codecov.io/gh/sympy/sympy/commit/fa19fc79ed1053b67c761962b1c13d22806c5de8?src=pr&el=desc) will **increase** coverage by `0.07%`.
> The diff coverage is `94.871%`.

\`\`\`diff
@@             Coverage Diff              @@
##            master    #16571      +/-   ##
============================================
+ Coverage   73.748%   73.819%   +0.07%     
============================================
  Files          619       619              
  Lines       158656    159426     +770     
  Branches     37185     37400     +215     
============================================
+ Hits        117006    117687     +681     
- Misses       36236     36282      +46     
- Partials      5414      5457      +43
\`\`\`

@czgdp1807 
I have also added some tests for the same module in  #16557
Also done some documentation work. Currently, improving the documentation 
but do check thos  out.
We can improve the tests for the module togather. It will be more effective.

@jksuom Any reviews/comments on my additions, [this](https://github.com/sympy/sympy/pull/16571/commits/7586750516e43c9c07cd8041e54a177838624c84) and [this](https://github.com/sympy/sympy/pull/16571/commits/6bbb90102e5e68290434ee53263ae94c9999fb72). I am adding commits in chunks so that it's easier to review.
I have corrected the tests for sampling according to [#16741 (comment)](https://github.com/sympy/sympy/issues/16741#issuecomment-487299356) in [the latest commit](https://github.com/sympy/sympy/pull/16571/commits/8ce1aa8ffcb87cd684f1bf5ae643820916448340). Please let me know if any changes are required.
@jksuom Please review the [latest commit](https://github.com/sympy/sympy/pull/16571/commits/cdc6df358644c2c79792e75e1b7e03f883592d3b). 
Note:
I cannot add `set` property to `UnifromDistribution` because it results in the following `NotImplemented` error. It would be great if you can tell me the reason behind this. I will be able to investigate only after a few days.
\`\`\`
>>> from sympy import *
>>> from sympy.stats import *
>>> l = Symbol('l', real=True, finite=True)
>>> w = Symbol('w', positive=True, finite=True)
>>> X = Uniform('x', l, l + w)
>>> P(X < l)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/home/gagandeep/sympy/sympy/stats/rv.py", line 756, in probability
    return result.doit()
  File "/home/gagandeep/sympy/sympy/integrals/integrals.py", line 636, in doit
    evalued_pw = piecewise_fold(Add(*piecewises))._eval_interval(x, a, b)
  File "/home/gagandeep/sympy/sympy/functions/elementary/piecewise.py", line 621, in _eval_interval
    return super(Piecewise, self)._eval_interval(sym, a, b)
  File "/home/gagandeep/sympy/sympy/core/expr.py", line 887, in _eval_interval
    B = self.subs(x, b)
  File "/home/gagandeep/sympy/sympy/core/basic.py", line 997, in subs
    rv = rv._subs(old, new, **kwargs)
  File "/home/gagandeep/sympy/sympy/core/cache.py", line 94, in wrapper
    retval = cfunc(*args, **kwargs)
  File "/home/gagandeep/sympy/sympy/core/basic.py", line 1109, in _subs
    rv = self._eval_subs(old, new)
  File "/home/gagandeep/sympy/sympy/functions/elementary/piecewise.py", line 873, in _eval_subs
    c = c._subs(old, new)
  File "/home/gagandeep/sympy/sympy/core/cache.py", line 94, in wrapper
    retval = cfunc(*args, **kwargs)
  File "/home/gagandeep/sympy/sympy/core/basic.py", line 1111, in _subs
    rv = fallback(self, old, new)
  File "/home/gagandeep/sympy/sympy/core/basic.py", line 1083, in fallback
    arg = arg._subs(old, new, **hints)
  File "/home/gagandeep/sympy/sympy/core/cache.py", line 94, in wrapper
    retval = cfunc(*args, **kwargs)
  File "/home/gagandeep/sympy/sympy/core/basic.py", line 1111, in _subs
    rv = fallback(self, old, new)
  File "/home/gagandeep/sympy/sympy/core/basic.py", line 1088, in fallback
    rv = self.func(*args)
  File "/home/gagandeep/sympy/sympy/core/relational.py", line 637, in __new__
    r = cls._eval_relation(lhs, rhs)
  File "/home/gagandeep/sympy/sympy/core/relational.py", line 916, in _eval_relation
    return _sympify(lhs.__ge__(rhs))
  File "/home/gagandeep/sympy/sympy/core/sympify.py", line 417, in _sympify
    return sympify(a, strict=True)
  File "/home/gagandeep/sympy/sympy/core/sympify.py", line 339, in sympify
    raise SympifyError(a)
sympy.core.sympify.SympifyError: SympifyError: NotImplemented

\`\`\` 
@supreet11agrawal @smichr Thanks for the comments and reviews. I will complete it after few clarifications in other PRs.
```

## Patch

```diff
diff --git a/sympy/stats/crv_types.py b/sympy/stats/crv_types.py
--- a/sympy/stats/crv_types.py
+++ b/sympy/stats/crv_types.py
@@ -163,6 +163,9 @@ def rv(symbol, cls, args):
 class ArcsinDistribution(SingleContinuousDistribution):
     _argnames = ('a', 'b')
 
+    def set(self):
+        return Interval(self.a, self.b)
+
     def pdf(self, x):
         return 1/(pi*sqrt((x - self.a)*(self.b - x)))
 
@@ -871,6 +874,8 @@ def ChiSquared(name, k):
 class DagumDistribution(SingleContinuousDistribution):
     _argnames = ('p', 'a', 'b')
 
+    set = Interval(0, oo)
+
     @staticmethod
     def check(p, a, b):
         _value_check(p > 0, "Shape parameter p must be positive.")
@@ -1205,6 +1210,13 @@ def FDistribution(name, d1, d2):
 class FisherZDistribution(SingleContinuousDistribution):
     _argnames = ('d1', 'd2')
 
+    set = Interval(-oo, oo)
+
+    @staticmethod
+    def check(d1, d2):
+        _value_check(d1 > 0, "Degree of freedom d1 must be positive.")
+        _value_check(d2 > 0, "Degree of freedom d2 must be positive.")
+
     def pdf(self, x):
         d1, d2 = self.d1, self.d2
         return (2*d1**(d1/2)*d2**(d2/2) / beta_fn(d1/2, d2/2) *
@@ -1276,6 +1288,11 @@ class FrechetDistribution(SingleContinuousDistribution):
 
     set = Interval(0, oo)
 
+    @staticmethod
+    def check(a, s, m):
+        _value_check(a > 0, "Shape parameter alpha must be positive.")
+        _value_check(s > 0, "Scale parameter s must be positive.")
+
     def __new__(cls, a, s=1, m=0):
         a, s, m = list(map(sympify, (a, s, m)))
         return Basic.__new__(cls, a, s, m)
@@ -1551,6 +1568,10 @@ class GumbelDistribution(SingleContinuousDistribution):
 
     set = Interval(-oo, oo)
 
+    @staticmethod
+    def check(beta, mu):
+        _value_check(beta > 0, "Scale parameter beta must be positive.")
+
     def pdf(self, x):
         beta, mu = self.beta, self.mu
         z = (x - mu)/beta
@@ -1564,7 +1585,7 @@ def _characteristic_function(self, t):
         return gamma(1 - I*self.beta*t) * exp(I*self.mu*t)
 
     def _moment_generating_function(self, t):
-        return gamma(1 - self.beta*t) * exp(I*self.mu*t)
+        return gamma(1 - self.beta*t) * exp(self.mu*t)
 
 def Gumbel(name, beta, mu):
     r"""
@@ -1765,6 +1786,13 @@ def Kumaraswamy(name, a, b):
 class LaplaceDistribution(SingleContinuousDistribution):
     _argnames = ('mu', 'b')
 
+    set = Interval(-oo, oo)
+
+    @staticmethod
+    def check(mu, b):
+        _value_check(b > 0, "Scale parameter b must be positive.")
+        _value_check(mu.is_real, "Location parameter mu should be real")
+
     def pdf(self, x):
         mu, b = self.mu, self.b
         return 1/(2*b)*exp(-Abs(x - mu)/b)
@@ -1852,6 +1880,12 @@ def Laplace(name, mu, b):
 class LogisticDistribution(SingleContinuousDistribution):
     _argnames = ('mu', 's')
 
+    set = Interval(-oo, oo)
+
+    @staticmethod
+    def check(mu, s):
+        _value_check(s > 0, "Scale parameter s must be positive.")
+
     def pdf(self, x):
         mu, s = self.mu, self.s
         return exp(-(x - mu)/s)/(s*(1 + exp(-(x - mu)/s))**2)
@@ -1864,7 +1898,7 @@ def _characteristic_function(self, t):
         return Piecewise((exp(I*t*self.mu) * pi*self.s*t / sinh(pi*self.s*t), Ne(t, 0)), (S.One, True))
 
     def _moment_generating_function(self, t):
-        return exp(self.mu*t) * Beta(1 - self.s*t, 1 + self.s*t)
+        return exp(self.mu*t) * beta_fn(1 - self.s*t, 1 + self.s*t)
 
     def _quantile(self, p):
         return self.mu - self.s*log(-S.One + S.One/p)
@@ -2015,6 +2049,10 @@ class MaxwellDistribution(SingleContinuousDistribution):
 
     set = Interval(0, oo)
 
+    @staticmethod
+    def check(a):
+        _value_check(a > 0, "Parameter a must be positive.")
+
     def pdf(self, x):
         a = self.a
         return sqrt(2/pi)*x**2*exp(-x**2/(2*a**2))/a**3
@@ -2085,6 +2123,11 @@ class NakagamiDistribution(SingleContinuousDistribution):
 
     set = Interval(0, oo)
 
+    @staticmethod
+    def check(mu, omega):
+        _value_check(mu >= S.Half, "Shape parameter mu must be greater than equal to 1/2.")
+        _value_check(omega > 0, "Spread parameter omega must be positive.")
+
     def pdf(self, x):
         mu, omega = self.mu, self.omega
         return 2*mu**mu/(gamma(mu)*omega**mu)*x**(2*mu - 1)*exp(-mu/omega*x**2)
@@ -2385,6 +2428,10 @@ class QuadraticUDistribution(SingleContinuousDistribution):
     def set(self):
         return Interval(self.a, self.b)
 
+    @staticmethod
+    def check(a, b):
+        _value_check(b > a, "Parameter b must be in range (%s, oo)."%(a))
+
     def pdf(self, x):
         a, b = self.a, self.b
         alpha = 12 / (b-a)**3
@@ -2553,6 +2600,10 @@ class RayleighDistribution(SingleContinuousDistribution):
 
     set = Interval(0, oo)
 
+    @staticmethod
+    def check(sigma):
+        _value_check(sigma > 0, "Scale parameter sigma must be positive.")
+
     def pdf(self, x):
         sigma = self.sigma
         return x/sigma**2*exp(-x**2/(2*sigma**2))
@@ -2690,6 +2741,12 @@ def ShiftedGompertz(name, b, eta):
 class StudentTDistribution(SingleContinuousDistribution):
     _argnames = ('nu',)
 
+    set = Interval(-oo, oo)
+
+    @staticmethod
+    def check(nu):
+        _value_check(nu > 0, "Degrees of freedom nu must be positive.")
+
     def pdf(self, x):
         nu = self.nu
         return 1/(sqrt(nu)*beta_fn(S(1)/2, nu/2))*(1 + x**2/nu)**(-(nu + 1)/2)
@@ -2770,6 +2827,19 @@ def StudentT(name, nu):
 class TrapezoidalDistribution(SingleContinuousDistribution):
     _argnames = ('a', 'b', 'c', 'd')
 
+    @property
+    def set(self):
+        return Interval(self.a, self.d)
+
+    @staticmethod
+    def check(a, b, c, d):
+        _value_check(a < d, "Lower bound parameter a < %s. a = %s"%(d, a))
+        _value_check((a <= b, b < c),
+        "Level start parameter b must be in range [%s, %s). b = %s"%(a, c, b))
+        _value_check((b < c, c <= d),
+        "Level end parameter c must be in range (%s, %s]. c = %s"%(b, d, c))
+        _value_check(d >= c, "Upper bound parameter d > %s. d = %s"%(c, d))
+
     def pdf(self, x):
         a, b, c, d = self.a, self.b, self.c, self.d
         return Piecewise(
@@ -2850,6 +2920,16 @@ def Trapezoidal(name, a, b, c, d):
 class TriangularDistribution(SingleContinuousDistribution):
     _argnames = ('a', 'b', 'c')
 
+    @property
+    def set(self):
+        return Interval(self.a, self.b)
+
+    @staticmethod
+    def check(a, b, c):
+        _value_check(b > a, "Parameter b > %s. b = %s"%(a, b))
+        _value_check((a <= c, c <= b),
+        "Parameter c must be in range [%s, %s]. c = %s"%(a, b, c))
+
     def pdf(self, x):
         a, b, c = self.a, self.b, self.c
         return Piecewise(
@@ -2864,7 +2944,7 @@ def _characteristic_function(self, t):
 
     def _moment_generating_function(self, t):
         a, b, c = self.a, self.b, self.c
-        return 2 * ((b - c) * exp(a * t) - (b - a) * exp(c * t) + (c + a) * exp(b * t)) / (
+        return 2 * ((b - c) * exp(a * t) - (b - a) * exp(c * t) + (c - a) * exp(b * t)) / (
         (b - a) * (c - a) * (b - c) * t ** 2)
 
 
@@ -2940,6 +3020,14 @@ def Triangular(name, a, b, c):
 class UniformDistribution(SingleContinuousDistribution):
     _argnames = ('left', 'right')
 
+    @property
+    def set(self):
+        return Interval(self.left, self.right)
+
+    @staticmethod
+    def check(left, right):
+        _value_check(left < right, "Lower limit should be less than Upper limit.")
+
     def pdf(self, x):
         left, right = self.left, self.right
         return Piecewise(
@@ -3047,6 +3135,11 @@ class UniformSumDistribution(SingleContinuousDistribution):
     def set(self):
         return Interval(0, self.n)
 
+    @staticmethod
+    def check(n):
+        _value_check((n > 0, n.is_integer),
+        "Parameter n must be positive integer.")
+
     def pdf(self, x):
         n = self.n
         k = Dummy("k")
@@ -3292,6 +3385,10 @@ class WignerSemicircleDistribution(SingleContinuousDistribution):
     def set(self):
         return Interval(-self.R, self.R)
 
+    @staticmethod
+    def check(R):
+        _value_check(R > 0, "Radius R must be positive.")
+
     def pdf(self, x):
         R = self.R
         return 2/(pi*R**2)*sqrt(R**2 - x**2)
diff --git a/sympy/stats/joint_rv_types.py b/sympy/stats/joint_rv_types.py
--- a/sympy/stats/joint_rv_types.py
+++ b/sympy/stats/joint_rv_types.py
@@ -76,7 +76,8 @@ def set(self):
         k = len(self.mu)
         return S.Reals**k
 
-    def check(self, mu, sigma):
+    @staticmethod
+    def check(mu, sigma):
         _value_check(len(mu) == len(sigma.col(0)),
             "Size of the mean vector and covariance matrix are incorrect.")
         #check if covariance matrix is positive definite or not.
@@ -117,7 +118,8 @@ def set(self):
         k = len(self.mu)
         return S.Reals**k
 
-    def check(self, mu, sigma):
+    @staticmethod
+    def check(mu, sigma):
         _value_check(len(mu) == len(sigma.col(0)),
             "Size of the mean vector and covariance matrix are incorrect.")
         #check if covariance matrix is positive definite or not.
@@ -151,7 +153,8 @@ def set(self):
         k = len(self.mu)
         return S.Reals**k
 
-    def check(self, mu, sigma, v):
+    @staticmethod
+    def check(mu, sigma, v):
         _value_check(len(mu) == len(sigma.col(0)),
             "Size of the location vector and shape matrix are incorrect.")
         #check if covariance matrix is positive definite or not.
@@ -196,7 +199,8 @@ class NormalGammaDistribution(JointDistribution):
     _argnames = ['mu', 'lamda', 'alpha', 'beta']
     is_Continuous=True
 
-    def check(self, mu, lamda, alpha, beta):
+    @staticmethod
+    def check(mu, lamda, alpha, beta):
         _value_check(mu.is_real, "Location must be real.")
         _value_check(lamda > 0, "Lambda must be positive")
         _value_check(alpha > 0, "alpha must be positive")
@@ -258,7 +262,8 @@ class MultivariateBetaDistribution(JointDistribution):
     _argnames = ['alpha']
     is_Continuous = True
 
-    def check(self, alpha):
+    @staticmethod
+    def check(alpha):
         _value_check(len(alpha) >= 2, "At least two categories should be passed.")
         for a_k in alpha:
             _value_check((a_k > 0) != False, "Each concentration parameter"
@@ -331,7 +336,8 @@ class MultivariateEwensDistribution(JointDistribution):
     is_Discrete = True
     is_Continuous = False
 
-    def check(self, n, theta):
+    @staticmethod
+    def check(n, theta):
         _value_check(isinstance(n, Integer) and (n > 0) == True,
                         "sample size should be positive integer.")
         _value_check(theta.is_positive, "mutation rate should be positive.")
@@ -403,7 +409,8 @@ class MultinomialDistribution(JointDistribution):
     is_Continuous=False
     is_Discrete = True
 
-    def check(self, n, p):
+    @staticmethod
+    def check(n, p):
         _value_check(n > 0,
                         "number of trials must be a positve integer")
         for p_k in p:
@@ -471,7 +478,8 @@ class NegativeMultinomialDistribution(JointDistribution):
     is_Continuous=False
     is_Discrete = True
 
-    def check(self, k0, p):
+    @staticmethod
+    def check(k0, p):
         _value_check(k0 > 0,
                         "number of failures must be a positve integer")
         for p_k in p:

```

## Test Patch

```diff
diff --git a/sympy/stats/tests/test_continuous_rv.py b/sympy/stats/tests/test_continuous_rv.py
--- a/sympy/stats/tests/test_continuous_rv.py
+++ b/sympy/stats/tests/test_continuous_rv.py
@@ -1,13 +1,16 @@
-from sympy import (Symbol, Abs, exp, S, N, pi, simplify, Interval, erf, erfc, Ne,
-                   Eq, log, lowergamma, uppergamma, Sum, symbols, sqrt, And, gamma, beta,
-                   Piecewise, Integral, sin, cos, tan, atan, besseli, factorial, binomial,
-                   floor, expand_func, Rational, I, re, im, lambdify, hyper, diff, Or, Mul)
+from sympy import (Symbol, Abs, exp, expint, S, N, pi, simplify, Interval, erf, erfc, Ne,
+                   EulerGamma, Eq, log, lowergamma, uppergamma, Sum, symbols, sqrt, And,
+                   gamma, beta, Piecewise, Integral, sin, cos, tan, atan, sinh, cosh,
+                   besseli, factorial, binomial, floor, expand_func, Rational, I, re,
+                   im, lambdify, hyper, diff, Or, Mul)
 from sympy.core.compatibility import range
 from sympy.external import import_module
 from sympy.functions.special.error_functions import erfinv
+from sympy.functions.special.hyper import meijerg
 from sympy.sets.sets import Intersection, FiniteSet
 from sympy.stats import (P, E, where, density, variance, covariance, skewness,
-                         given, pspace, cdf, characteristic_function, ContinuousRV, sample,
+                         given, pspace, cdf, characteristic_function,
+                         moment_generating_function, ContinuousRV, sample,
                          Arcsin, Benini, Beta, BetaNoncentral, BetaPrime, Cauchy,
                          Chi, ChiSquared,
                          ChiNoncentral, Dagum, Erlang, Exponential,
@@ -22,6 +25,7 @@
 from sympy.stats.joint_rv import JointPSpace
 from sympy.utilities.pytest import raises, XFAIL, slow, skip
 from sympy.utilities.randtest import verify_numerically as tn
+from sympy import E as e
 
 oo = S.Infinity
 
@@ -34,8 +38,8 @@ def test_single_normal():
     X = Normal('x', 0, 1)
     Y = X*sigma + mu
 
-    assert simplify(E(Y)) == mu
-    assert simplify(variance(Y)) == sigma**2
+    assert E(Y) == mu
+    assert variance(Y) == sigma**2
     pdf = density(Y)
     x = Symbol('x')
     assert (pdf(x) ==
@@ -46,12 +50,12 @@ def test_single_normal():
     assert E(X, Eq(X, mu)) == mu
 
 
-@XFAIL
 def test_conditional_1d():
     X = Normal('x', 0, 1)
     Y = given(X, X >= 0)
+    z = Symbol('z')
 
-    assert density(Y) == 2 * density(X)
+    assert density(Y)(z) == 2 * density(X)(z)
 
     assert Y.pspace.domain.set == Interval(0, oo)
     assert E(Y) == sqrt(2) / sqrt(pi)
@@ -108,7 +112,7 @@ def test_symbolic():
     assert E(X + Y) == mu1 + mu2
     assert E(a*X + b) == a*E(X) + b
     assert variance(X) == s1**2
-    assert simplify(variance(X + a*Y + b)) == variance(X) + a**2*variance(Y)
+    assert variance(X + a*Y + b) == variance(X) + a**2*variance(Y)
 
     assert E(Z) == 1/rate
     assert E(a*Z + b) == a*E(Z) + b
@@ -147,12 +151,144 @@ def test_characteristic_function():
     Y = Normal('y', 1, 1)
     cf = characteristic_function(Y)
     assert cf(0) == 1
-    assert simplify(cf(1)) == exp(I - S(1)/2)
+    assert cf(1) == exp(I - S(1)/2)
 
     Z = Exponential('z', 5)
     cf = characteristic_function(Z)
     assert cf(0) == 1
-    assert simplify(cf(1)) == S(25)/26 + 5*I/26
+    assert cf(1).expand() == S(25)/26 + 5*I/26
+
+def test_moment_generating_function():
+    t = symbols('t', positive=True)
+
+    # Symbolic tests
+    a, b, c = symbols('a b c')
+
+    mgf = moment_generating_function(Beta('x', a, b))(t)
+    assert mgf == hyper((a,), (a + b,), t)
+
+    mgf = moment_generating_function(Chi('x', a))(t)
+    assert mgf == sqrt(2)*t*gamma(a/2 + S(1)/2)*\
+        hyper((a/2 + S(1)/2,), (S(3)/2,), t**2/2)/gamma(a/2) +\
+        hyper((a/2,), (S(1)/2,), t**2/2)
+
+    mgf = moment_generating_function(ChiSquared('x', a))(t)
+    assert mgf == (1 - 2*t)**(-a/2)
+
+    mgf = moment_generating_function(Erlang('x', a, b))(t)
+    assert mgf == (1 - t/b)**(-a)
+
+    mgf = moment_generating_function(Exponential('x', a))(t)
+    assert mgf == a/(a - t)
+
+    mgf = moment_generating_function(Gamma('x', a, b))(t)
+    assert mgf == (-b*t + 1)**(-a)
+
+    mgf = moment_generating_function(Gumbel('x', a, b))(t)
+    assert mgf == exp(b*t)*gamma(-a*t + 1)
+
+    mgf = moment_generating_function(Gompertz('x', a, b))(t)
+    assert mgf == b*exp(b)*expint(t/a, b)
+
+    mgf = moment_generating_function(Laplace('x', a, b))(t)
+    assert mgf == exp(a*t)/(-b**2*t**2 + 1)
+
+    mgf = moment_generating_function(Logistic('x', a, b))(t)
+    assert mgf == exp(a*t)*beta(-b*t + 1, b*t + 1)
+
+    mgf = moment_generating_function(Normal('x', a, b))(t)
+    assert mgf == exp(a*t + b**2*t**2/2)
+
+    mgf = moment_generating_function(Pareto('x', a, b))(t)
+    assert mgf == b*(-a*t)**b*uppergamma(-b, -a*t)
+
+    mgf = moment_generating_function(QuadraticU('x', a, b))(t)
+    assert str(mgf) == ("(3*(t*(-4*b + (a + b)**2) + 4)*exp(b*t) - "
+    "3*(t*(a**2 + 2*a*(b - 2) + b**2) + 4)*exp(a*t))/(t**2*(a - b)**3)")
+
+    mgf = moment_generating_function(RaisedCosine('x', a, b))(t)
+    assert mgf == pi**2*exp(a*t)*sinh(b*t)/(b*t*(b**2*t**2 + pi**2))
+
+    mgf = moment_generating_function(Rayleigh('x', a))(t)
+    assert mgf == sqrt(2)*sqrt(pi)*a*t*(erf(sqrt(2)*a*t/2) + 1)\
+        *exp(a**2*t**2/2)/2 + 1
+
+    mgf = moment_generating_function(Triangular('x', a, b, c))(t)
+    assert str(mgf) == ("(-2*(-a + b)*exp(c*t) + 2*(-a + c)*exp(b*t) + "
+    "2*(b - c)*exp(a*t))/(t**2*(-a + b)*(-a + c)*(b - c))")
+
+    mgf = moment_generating_function(Uniform('x', a, b))(t)
+    assert mgf == (-exp(a*t) + exp(b*t))/(t*(-a + b))
+
+    mgf = moment_generating_function(UniformSum('x', a))(t)
+    assert mgf == ((exp(t) - 1)/t)**a
+
+    mgf = moment_generating_function(WignerSemicircle('x', a))(t)
+    assert mgf == 2*besseli(1, a*t)/(a*t)
+
+    # Numeric tests
+
+    mgf = moment_generating_function(Beta('x', 1, 1))(t)
+    assert mgf.diff(t).subs(t, 1) == hyper((2,), (3,), 1)/2
+
+    mgf = moment_generating_function(Chi('x', 1))(t)
+    assert mgf.diff(t).subs(t, 1) == sqrt(2)*hyper((1,), (S(3)/2,), S(1)/2
+    )/sqrt(pi) + hyper((S(3)/2,), (S(3)/2,), S(1)/2) + 2*sqrt(2)*hyper((2,),
+    (S(5)/2,), S(1)/2)/(3*sqrt(pi))
+
+    mgf = moment_generating_function(ChiSquared('x', 1))(t)
+    assert mgf.diff(t).subs(t, 1) == I
+
+    mgf = moment_generating_function(Erlang('x', 1, 1))(t)
+    assert mgf.diff(t).subs(t, 0) == 1
+
+    mgf = moment_generating_function(Exponential('x', 1))(t)
+    assert mgf.diff(t).subs(t, 0) == 1
+
+    mgf = moment_generating_function(Gamma('x', 1, 1))(t)
+    assert mgf.diff(t).subs(t, 0) == 1
+
+    mgf = moment_generating_function(Gumbel('x', 1, 1))(t)
+    assert mgf.diff(t).subs(t, 0) == EulerGamma + 1
+
+    mgf = moment_generating_function(Gompertz('x', 1, 1))(t)
+    assert mgf.diff(t).subs(t, 1) == -e*meijerg(((), (1, 1)),
+    ((0, 0, 0), ()), 1)
+
+    mgf = moment_generating_function(Laplace('x', 1, 1))(t)
+    assert mgf.diff(t).subs(t, 0) == 1
+
+    mgf = moment_generating_function(Logistic('x', 1, 1))(t)
+    assert mgf.diff(t).subs(t, 0) == beta(1, 1)
+
+    mgf = moment_generating_function(Normal('x', 0, 1))(t)
+    assert mgf.diff(t).subs(t, 1) == exp(S(1)/2)
+
+    mgf = moment_generating_function(Pareto('x', 1, 1))(t)
+    assert mgf.diff(t).subs(t, 0) == expint(1, 0)
+
+    mgf = moment_generating_function(QuadraticU('x', 1, 2))(t)
+    assert mgf.diff(t).subs(t, 1) == -12*e - 3*exp(2)
+
+    mgf = moment_generating_function(RaisedCosine('x', 1, 1))(t)
+    assert mgf.diff(t).subs(t, 1) == -2*e*pi**2*sinh(1)/\
+    (1 + pi**2)**2 + e*pi**2*cosh(1)/(1 + pi**2)
+
+    mgf = moment_generating_function(Rayleigh('x', 1))(t)
+    assert mgf.diff(t).subs(t, 0) == sqrt(2)*sqrt(pi)/2
+
+    mgf = moment_generating_function(Triangular('x', 1, 3, 2))(t)
+    assert mgf.diff(t).subs(t, 1) == -e + exp(3)
+
+    mgf = moment_generating_function(Uniform('x', 0, 1))(t)
+    assert mgf.diff(t).subs(t, 1) == 1
+
+    mgf = moment_generating_function(UniformSum('x', 1))(t)
+    assert mgf.diff(t).subs(t, 1) == 1
+
+    mgf = moment_generating_function(WignerSemicircle('x', 1))(t)
+    assert mgf.diff(t).subs(t, 1) == -2*besseli(1, 1) + besseli(2, 1) +\
+        besseli(0, 1)
 
 
 def test_sample_continuous():
@@ -451,7 +587,7 @@ def test_gamma():
     X = Gamma('x', k, theta)
     assert E(X) == k*theta
     assert variance(X) == k*theta**2
-    assert simplify(skewness(X)) == 2/sqrt(k)
+    assert skewness(X).expand() == 2/sqrt(k)
 
 
 def test_gamma_inverse():
@@ -554,7 +690,7 @@ def test_maxwell():
     assert density(X)(x) == (sqrt(2)*x**2*exp(-x**2/(2*a**2))/
         (sqrt(pi)*a**3))
     assert E(X) == 2*sqrt(2)*a/sqrt(pi)
-    assert simplify(variance(X)) == a**2*(-8 + 3*pi)/pi
+    assert variance(X) == -8*a**2/pi + 3*a**2
     assert cdf(X)(x) == erf(sqrt(2)*x/(2*a)) - sqrt(2)*x*exp(-x**2/(2*a**2))/(sqrt(pi)*a)
     assert diff(cdf(X)(x), x) == density(X)(x)
 
@@ -653,18 +789,14 @@ def test_trapezoidal():
     assert variance(X) == S(5)/12
     assert P(X < 2) == S(3)/4
 
-@XFAIL
 def test_triangular():
     a = Symbol("a")
     b = Symbol("b")
     c = Symbol("c")
 
     X = Triangular('x', a, b, c)
-    assert density(X)(x) == Piecewise(
-                 ((2*x - 2*a)/((-a + b)*(-a + c)), And(a <= x, x < c)),
-                 (2/(-a + b), x == c),
-                 ((-2*x + 2*b)/((-a + b)*(b - c)), And(x <= b, c < x)),
-                 (0, True))
+    assert str(density(X)(x)) == ("Piecewise(((-2*a + 2*x)/((-a + b)*(-a + c)), (a <= x) & (c > x)), "
+    "(2/(-a + b), Eq(c, x)), ((2*b - 2*x)/((-a + b)*(b - c)), (b >= x) & (c < x)), (0, True))")
 
 
 def test_quadratic_u():
@@ -681,8 +813,8 @@ def test_uniform():
     w = Symbol('w', positive=True, finite=True)
     X = Uniform('x', l, l + w)
 
-    assert simplify(E(X)) == l + w/2
-    assert simplify(variance(X)) == w**2/12
+    assert E(X) == l + w/2
+    assert variance(X).expand() == w**2/12
 
     # With numbers all is well
     X = Uniform('x', 3, 5)
@@ -700,7 +832,7 @@ def test_uniform():
     assert c(S(7)/2) == S(1)/4
     assert c(5) == 1 and c(6) == 1
 
-
+@XFAIL
 def test_uniform_P():
     """ This stopped working because SingleContinuousPSpace.compute_density no
     longer calls integrate on a DiracDelta but rather just solves directly.
@@ -738,8 +870,8 @@ def test_weibull():
     a, b = symbols('a b', positive=True)
     X = Weibull('x', a, b)
 
-    assert simplify(E(X)) == simplify(a * gamma(1 + 1/b))
-    assert simplify(variance(X)) == simplify(a**2 * gamma(1 + 2/b) - E(X)**2)
+    assert E(X).expand() == a * gamma(1 + 1/b)
+    assert variance(X).expand() == (a**2 * gamma(1 + 2/b) - E(X)**2).expand()
     assert simplify(skewness(X)) == (2*gamma(1 + 1/b)**3 - 3*gamma(1 + 1/b)*gamma(1 + 2/b) + gamma(1 + 3/b))/(-gamma(1 + 1/b)**2 + gamma(1 + 2/b))**(S(3)/2)
 
 def test_weibull_numeric():
@@ -795,22 +927,18 @@ def test_input_value_assertions():
         fn('x', p, q)  # No error raised
 
 
-@XFAIL
 def test_unevaluated():
     X = Normal('x', 0, 1)
-    assert E(X, evaluate=False) == (
-        Integral(sqrt(2)*x*exp(-x**2/2)/(2*sqrt(pi)), (x, -oo, oo)))
+    assert str(E(X, evaluate=False)) == ("Integral(sqrt(2)*x*exp(-x**2/2)/"
+    "(2*sqrt(pi)), (x, -oo, oo))")
 
-    assert E(X + 1, evaluate=False) == (
-        Integral(sqrt(2)*x*exp(-x**2/2)/(2*sqrt(pi)), (x, -oo, oo)) + 1)
+    assert str(E(X + 1, evaluate=False)) == ("Integral(sqrt(2)*x*exp(-x**2/2)/"
+    "(2*sqrt(pi)), (x, -oo, oo)) + 1")
 
-    assert P(X > 0, evaluate=False) == (
-        Integral(sqrt(2)*exp(-x**2/2)/(2*sqrt(pi)), (x, 0, oo)))
+    assert str(P(X > 0, evaluate=False)) == ("Integral(sqrt(2)*exp(-_z**2/2)/"
+    "(2*sqrt(pi)), (_z, 0, oo))")
 
-    assert P(X > 0, X**2 < 1, evaluate=False) == (
-        Integral(sqrt(2)*exp(-x**2/2)/(2*sqrt(pi)*
-            Integral(sqrt(2)*exp(-x**2/2)/(2*sqrt(pi)),
-                (x, -1, 1))), (x, 0, 1)))
+    assert P(X > 0, X**2 < 1, evaluate=False) == S(1)/2
 
 
 def test_probability_unevaluated():

```


## Code snippets

### 1 - setup.py:

Start line: 307, End line: 382

```python
# Check that this list is uptodate against the result of the command:
# python bin/generate_test_list.py
tests = [
    'sympy.algebras.tests',
    'sympy.assumptions.tests',
    'sympy.calculus.tests',
    'sympy.categories.tests',
    'sympy.codegen.tests',
    'sympy.combinatorics.tests',
    'sympy.concrete.tests',
    'sympy.core.tests',
    'sympy.crypto.tests',
    'sympy.deprecated.tests',
    'sympy.diffgeom.tests',
    'sympy.discrete.tests',
    'sympy.external.tests',
    'sympy.functions.combinatorial.tests',
    'sympy.functions.elementary.tests',
    'sympy.functions.special.tests',
    'sympy.geometry.tests',
    'sympy.holonomic.tests',
    'sympy.integrals.rubi.parsetools.tests',
    'sympy.integrals.rubi.rubi_tests.tests',
    'sympy.integrals.rubi.tests',
    'sympy.integrals.tests',
    'sympy.interactive.tests',
    'sympy.liealgebras.tests',
    'sympy.logic.tests',
    'sympy.matrices.expressions.tests',
    'sympy.matrices.tests',
    'sympy.multipledispatch.tests',
    'sympy.ntheory.tests',
    'sympy.parsing.tests',
    'sympy.physics.continuum_mechanics.tests',
    'sympy.physics.hep.tests',
    'sympy.physics.mechanics.tests',
    'sympy.physics.optics.tests',
    'sympy.physics.quantum.tests',
    'sympy.physics.tests',
    'sympy.physics.units.tests',
    'sympy.physics.vector.tests',
    'sympy.plotting.intervalmath.tests',
    'sympy.plotting.pygletplot.tests',
    'sympy.plotting.tests',
    'sympy.polys.agca.tests',
    'sympy.polys.domains.tests',
    'sympy.polys.tests',
    'sympy.printing.pretty.tests',
    'sympy.printing.tests',
    'sympy.sandbox.tests',
    'sympy.series.tests',
    'sympy.sets.tests',
    'sympy.simplify.tests',
    'sympy.solvers.tests',
    'sympy.stats.tests',
    'sympy.strategies.branch.tests',
    'sympy.strategies.tests',
    'sympy.tensor.array.tests',
    'sympy.tensor.tests',
    'sympy.unify.tests',
    'sympy.utilities._compilation.tests',
    'sympy.utilities.tests',
    'sympy.vector.tests',
]

long_description = '''SymPy is a Python library for symbolic mathematics. It aims
to become a full-featured computer algebra system (CAS) while keeping the code
as simple as possible in order to be comprehensible and easily extensible.
SymPy is written entirely in Python.'''

with open(os.path.join(dir_setup, 'sympy', 'release.py')) as f:
    # Defines __version__
    exec(f.read())

with open(os.path.join(dir_setup, 'sympy', '__init__.py')) as f:
    long_description = f.read().split('"""')[1]
```
### 2 - sympy/__init__.py:

Start line: 1, End line: 93

```python
"""
SymPy is a Python library for symbolic mathematics. It aims to become a
full-featured computer algebra system (CAS) while keeping the code as simple
as possible in order to be comprehensible and easily extensible.  SymPy is
written entirely in Python. It depends on mpmath, and other external libraries
may be optionally for things like plotting support.

See the webpage for more information and documentation:

    https://sympy.org

"""


from __future__ import absolute_import, print_function
del absolute_import, print_function

try:
    import mpmath
except ImportError:
    raise ImportError("SymPy now depends on mpmath as an external library. "
    "See https://docs.sympy.org/latest/install.html#mpmath for more information.")

del mpmath

from sympy.release import __version__

if 'dev' in __version__:
    def enable_warnings():
        import warnings
        warnings.filterwarnings('default',   '.*',   DeprecationWarning, module='sympy.*')
        del warnings
    enable_warnings()
    del enable_warnings


import sys
if ((sys.version_info[0] == 2 and sys.version_info[1] < 7) or
    (sys.version_info[0] == 3 and sys.version_info[1] < 4)):
    raise ImportError("Python version 2.7 or 3.4 or above "
                      "is required for SymPy.")

del sys


def __sympy_debug():
    # helper function so we don't import os globally
    import os
    debug_str = os.getenv('SYMPY_DEBUG', 'False')
    if debug_str in ('True', 'False'):
        return eval(debug_str)
    else:
        raise RuntimeError("unrecognized value for SYMPY_DEBUG: %s" %
                           debug_str)
SYMPY_DEBUG = __sympy_debug()

from .core import *
from .logic import *
from .assumptions import *
from .polys import *
from .series import *
from .functions import *
from .ntheory import *
from .concrete import *
from .discrete import *
from .simplify import *
from .sets import *
from .solvers import *
from .matrices import *
from .geometry import *
from .utilities import *
from .integrals import *
from .tensor import *
from .parsing import *
from .calculus import *
from .algebras import *
# This module causes conflicts with other modules:
# from .stats import *
# Adds about .04-.05 seconds of import time
# from combinatorics import *
# This module is slow to import:
#from physics import units
from .plotting import plot, textplot, plot_backends, plot_implicit
from .printing import *
from .interactive import init_session, init_printing

evalf._create_evalf_table()

# This is slow to import:
#import abc

from .deprecated import *
```
### 3 - setup.py:

Start line: 79, End line: 163

```python
modules = [
    'sympy.algebras',
    'sympy.assumptions',
    'sympy.assumptions.handlers',
    'sympy.benchmarks',
    'sympy.calculus',
    'sympy.categories',
    'sympy.codegen',
    'sympy.combinatorics',
    'sympy.concrete',
    'sympy.core',
    'sympy.core.benchmarks',
    'sympy.crypto',
    'sympy.deprecated',
    'sympy.diffgeom',
    'sympy.discrete',
    'sympy.external',
    'sympy.functions',
    'sympy.functions.combinatorial',
    'sympy.functions.elementary',
    'sympy.functions.elementary.benchmarks',
    'sympy.functions.special',
    'sympy.functions.special.benchmarks',
    'sympy.geometry',
    'sympy.holonomic',
    'sympy.integrals',
    'sympy.integrals.benchmarks',
    'sympy.integrals.rubi',
    'sympy.integrals.rubi.parsetools',
    'sympy.integrals.rubi.rubi_tests',
    'sympy.integrals.rubi.rules',
    'sympy.interactive',
    'sympy.liealgebras',
    'sympy.logic',
    'sympy.logic.algorithms',
    'sympy.logic.utilities',
    'sympy.matrices',
    'sympy.matrices.benchmarks',
    'sympy.matrices.expressions',
    'sympy.multipledispatch',
    'sympy.ntheory',
    'sympy.parsing',
    'sympy.parsing.autolev',
    'sympy.parsing.autolev._antlr',
    'sympy.parsing.autolev.test-examples',
    'sympy.parsing.autolev.test-examples.pydy-example-repo',
    'sympy.parsing.latex',
    'sympy.parsing.latex._antlr',
    'sympy.physics',
    'sympy.physics.continuum_mechanics',
    'sympy.physics.hep',
    'sympy.physics.mechanics',
    'sympy.physics.optics',
    'sympy.physics.quantum',
    'sympy.physics.units',
    'sympy.physics.units.systems',
    'sympy.physics.vector',
    'sympy.plotting',
    'sympy.plotting.intervalmath',
    'sympy.plotting.pygletplot',
    'sympy.polys',
    'sympy.polys.agca',
    'sympy.polys.benchmarks',
    'sympy.polys.domains',
    'sympy.printing',
    'sympy.printing.pretty',
    'sympy.sandbox',
    'sympy.series',
    'sympy.series.benchmarks',
    'sympy.sets',
    'sympy.sets.handlers',
    'sympy.simplify',
    'sympy.solvers',
    'sympy.solvers.benchmarks',
    'sympy.stats',
    'sympy.strategies',
    'sympy.strategies.branch',
    'sympy.tensor',
    'sympy.tensor.array',
    'sympy.unify',
    'sympy.utilities',
    'sympy.utilities._compilation',
    'sympy.utilities.mathml',
    'sympy.vector',
]
```
### 4 - release/fabfile.py:

Start line: 851, End line: 895

```python
@task
def update_sympy_org(website_location=None):
    """
    Update sympy.org

    This just means adding an entry to the news section.
    """
    website_location = website_location or get_location("sympy.github.com")

    # Check that the website directory is clean
    local("cd {website_location} && git diff --exit-code > /dev/null".format(website_location=website_location))
    local("cd {website_location} && git diff --cached --exit-code > /dev/null".format(website_location=website_location))

    release_date = time.gmtime(os.path.getctime(os.path.join("release",
        tarball_formatter()['source'])))
    release_year = str(release_date.tm_year)
    release_month = str(release_date.tm_mon)
    release_day = str(release_date.tm_mday)
    version = get_sympy_version()

    with open(os.path.join(website_location, "templates", "index.html"), 'r') as f:
        lines = f.read().split('\n')
        # We could try to use some html parser, but this way is easier
        try:
            news = lines.index(r"    <h3>{% trans %}News{% endtrans %}</h3>")
        except ValueError:
            error("index.html format not as expected")
        lines.insert(news + 2,  # There is a <p> after the news line. Put it
            # after that.
            r"""        <span class="date">{{ datetime(""" + release_year + """, """ + release_month + """, """ + release_day + """) }}</span> {% trans v='""" + version + """' %}Version {{ v }} released{% endtrans %} (<a href="https://github.com/sympy/sympy/wiki/Release-Notes-for-""" + version + """">{% trans %}changes{% endtrans %}</a>)<br/>
        <p>""")

    with open(os.path.join(website_location, "templates", "index.html"), 'w') as f:
        print("Updating index.html template")
        f.write('\n'.join(lines))

    print("Generating website pages")
    local("cd {website_location} && ./generate".format(website_location=website_location))

    print("Committing")
    local("cd {website_location} && git commit -a -m \'Add {version} to the news\'".format(website_location=website_location,
        version=version))

    print("Pushing")
    local("cd {website_location} && git push origin".format(website_location=website_location))
```
### 5 - release/fabfile.py:

Start line: 797, End line: 849

```python
@task
def update_docs(docs_location=None):
    """
    Update the docs hosted at docs.sympy.org
    """
    docs_location = docs_location or get_location("docs")

    print("Docs location:", docs_location)

    # Check that the docs directory is clean
    local("cd {docs_location} && git diff --exit-code > /dev/null".format(docs_location=docs_location))
    local("cd {docs_location} && git diff --cached --exit-code > /dev/null".format(docs_location=docs_location))

    # See the README of the docs repo. We have to remove the old redirects,
    # move in the new docs, and create redirects.
    current_version = get_sympy_version()
    previous_version = get_previous_version_tag().lstrip('sympy-')
    print("Removing redirects from previous version")
    local("cd {docs_location} && rm -r {previous_version}".format(docs_location=docs_location,
        previous_version=previous_version))
    print("Moving previous latest docs to old version")
    local("cd {docs_location} && mv latest {previous_version}".format(docs_location=docs_location,
        previous_version=previous_version))

    print("Unzipping docs into repo")
    release_dir = os.path.abspath(os.path.expanduser(os.path.join(os.path.curdir, 'release')))
    docs_zip = os.path.abspath(os.path.join(release_dir, get_tarball_name('html')))
    local("cd {docs_location} && unzip {docs_zip} > /dev/null".format(docs_location=docs_location,
        docs_zip=docs_zip))
    local("cd {docs_location} && mv {docs_zip_name} {version}".format(docs_location=docs_location,
        docs_zip_name=get_tarball_name("html-nozip"), version=current_version))

    print("Writing new version to releases.txt")
    with open(os.path.join(docs_location, "releases.txt"), 'a') as f:
        f.write("{version}:SymPy {version}\n".format(version=current_version))

    print("Generating indexes")
    local("cd {docs_location} && ./generate_indexes.py".format(docs_location=docs_location))
    local("cd {docs_location} && mv {version} latest".format(docs_location=docs_location,
        version=current_version))

    print("Generating redirects")
    local("cd {docs_location} && ./generate_redirects.py latest {version} ".format(docs_location=docs_location,
        version=current_version))

    print("Committing")
    local("cd {docs_location} && git add -A {version} latest".format(docs_location=docs_location,
        version=current_version))
    local("cd {docs_location} && git commit -a -m \'Updating docs to {version}\'".format(docs_location=docs_location,
        version=current_version))

    print("Pushing")
    local("cd {docs_location} && git push origin".format(docs_location=docs_location))
```
### 6 - sympy/stats/frv.py:

Start line: 189, End line: 231

```python
class SingleFiniteDistribution(Basic, NamedArgsMixin):
    def __new__(cls, *args):
        args = list(map(sympify, args))
        return Basic.__new__(cls, *args)

    @staticmethod
    def check(*args):
        pass

    @property
    @cacheit
    def dict(self):
        return dict((k, self.pdf(k)) for k in self.set)

    @property
    def pdf(self):
        x = Symbol('x')
        return Lambda(x, Piecewise(*(
            [(v, Eq(k, x)) for k, v in self.dict.items()] + [(0, True)])))

    @property
    def characteristic_function(self):
        t = Dummy('t', real=True)
        return Lambda(t, sum(exp(I*k*t)*v for k, v in self.dict.items()))

    @property
    def moment_generating_function(self):
        t = Dummy('t', real=True)
        return Lambda(t, sum(exp(k * t) * v for k, v in self.dict.items()))

    @property
    def set(self):
        return list(self.dict.keys())

    values = property(lambda self: self.dict.values)
    items = property(lambda self: self.dict.items)
    __iter__ = property(lambda self: self.dict.__iter__)
    __getitem__ = property(lambda self: self.dict.__getitem__)

    __call__ = pdf

    def __contains__(self, other):
        return other in self.set
```
### 7 - setup.py:

Start line: 384, End line: 434

```python
if __name__ == '__main__':
    setup(name='sympy',
          version=__version__,
          description='Computer algebra system (CAS) in Python',
          long_description=long_description,
          author='SymPy development team',
          author_email='sympy@googlegroups.com',
          license='BSD',
          keywords="Math CAS",
          url='https://sympy.org',
          py_modules=['isympy'],
          packages=['sympy'] + modules + tests,
          ext_modules=[],
          package_data={
              'sympy.utilities.mathml': ['data/*.xsl'],
              'sympy.logic.benchmarks': ['input/*.cnf'],
              'sympy.parsing.autolev': ['*.g4'],
              'sympy.parsing.autolev.test-examples': ['*.al'],
              'sympy.parsing.autolev.test-examples.pydy-example-repo': ['*.al'],
              'sympy.parsing.latex': ['*.txt', '*.g4'],
              'sympy.integrals.rubi.parsetools': ['header.py.txt'],
              },
          data_files=[('share/man/man1', ['doc/man/isympy.1'])],
          cmdclass={'test': test_sympy,
                    'bench': run_benchmarks,
                    'clean': clean,
                    'audit': audit,
                    'antlr': antlr,
                    },
          classifiers=[
            'License :: OSI Approved :: BSD License',
            'Operating System :: OS Independent',
            'Programming Language :: Python',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Mathematics',
            'Topic :: Scientific/Engineering :: Physics',
            'Programming Language :: Python :: 2',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.4',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: Implementation :: CPython',
            'Programming Language :: Python :: Implementation :: PyPy',
            ],
          install_requires=[
            'mpmath>=%s' % mpmath_version,
            ],
          **extra_kwargs
          )
```
### 8 - sympy/integrals/risch.py:

Start line: 600, End line: 637

```python
class DifferentialExtension(object):

    @property
    def _important_attrs(self):
        """
        Returns some of the more important attributes of self.

        Used for testing and debugging purposes.

        The attributes are (fa, fd, D, T, Tfuncs, backsubs,
        exts, extargs).
        """
        return (self.fa, self.fd, self.D, self.T, self.Tfuncs,
            self.backsubs, self.exts, self.extargs)

    # NOTE: this printing doesn't follow the Python's standard
    # eval(repr(DE)) == DE, where DE is the DifferentialExtension object
    # , also this printing is supposed to contain all the important
    # attributes of a DifferentialExtension object
    def __repr__(self):
        # no need to have GeneratorType object printed in it
        r = [(attr, getattr(self, attr)) for attr in self.__slots__
                if not isinstance(getattr(self, attr), GeneratorType)]
        return self.__class__.__name__ + '(dict(%r))' % (r)

    # fancy printing of DifferentialExtension object
    def __str__(self):
        return (self.__class__.__name__ + '({fa=%s, fd=%s, D=%s})' %
                (self.fa, self.fd, self.D))

    # should only be used for debugging purposes, internally
    # f1 = f2 = log(x) at different places in code execution
    # may return D1 != D2 as True, since 'level' or other attribute
    # may differ
    def __eq__(self, other):
        for attr in self.__class__.__slots__:
            d1, d2 = getattr(self, attr), getattr(other, attr)
            if not (isinstance(d1, GeneratorType) or d1 == d2):
                return False
        return True
```
### 9 - sympy/stats/__init__.py:

Start line: 1, End line: 77

```python
"""
SymPy statistics module

Introduces a random variable type into the SymPy language.

Random variables may be declared using prebuilt functions such as
Normal, Exponential, Coin, Die, etc...  or built with functions like FiniteRV.

Queries on random expressions can be made using the functions

========================= =============================
    Expression                    Meaning
------------------------- -----------------------------
 ``P(condition)``          Probability
 ``E(expression)``         Expected value
 ``H(expression)``         Entropy
 ``variance(expression)``  Variance
 ``density(expression)``   Probability Density Function
 ``sample(expression)``    Produce a realization
 ``where(condition)``      Where the condition is true
========================= =============================

Examples
========

>>> from sympy.stats import P, E, variance, Die, Normal
>>> from sympy import Eq, simplify
>>> X, Y = Die('X', 6), Die('Y', 6) # Define two six sided dice
>>> Z = Normal('Z', 0, 1) # Declare a Normal random variable with mean 0, std 1
>>> P(X>3) # Probability X is greater than 3
1/2
>>> E(X+Y) # Expectation of the sum of two dice
7
>>> variance(X+Y) # Variance of the sum of two dice
35/6
>>> simplify(P(Z>1)) # Probability of Z being greater than 1
1/2 - erf(sqrt(2)/2)/2
"""

__all__ = []

from . import rv_interface
from .rv_interface import (
    cdf, characteristic_function, covariance, density, dependent, E, given, independent, P, pspace,
    random_symbols, sample, sample_iter, skewness, std, variance, where,
    correlation, moment, cmoment, smoment, sampling_density, moment_generating_function, entropy, H,
    quantile
)
__all__.extend(rv_interface.__all__)

from . import frv_types
from .frv_types import (
    Bernoulli, Binomial, Coin, Die, DiscreteUniform, FiniteRV, Hypergeometric,
    Rademacher,
)
__all__.extend(frv_types.__all__)

from . import crv_types
from .crv_types import (
    ContinuousRV,
    Arcsin, Benini, Beta, BetaNoncentral, BetaPrime, Cauchy, Chi, ChiNoncentral, ChiSquared,
    Dagum, Erlang, Exponential, FDistribution, FisherZ, Frechet, Gamma,
    GammaInverse, Gumbel, Gompertz, Kumaraswamy, Laplace, Logistic, LogNormal,
    Maxwell, Nakagami, Normal, Pareto, QuadraticU, RaisedCosine, Rayleigh,
    ShiftedGompertz, StudentT, Trapezoidal, Triangular, Uniform, UniformSum, VonMises,
    Weibull, WignerSemicircle
)
__all__.extend(crv_types.__all__)

from . import drv_types
from .drv_types import (Geometric, Logarithmic, NegativeBinomial, Poisson, YuleSimon, Zeta)
__all__.extend(drv_types.__all__)

from . import symbolic_probability
from .symbolic_probability import Probability, Expectation, Variance, Covariance
__all__.extend(symbolic_probability.__all__)
```
### 10 - release/fabfile.py:

Start line: 356, End line: 431

```python
# If a file does not end up in the tarball that should, add it to setup.py if
# it is Python, or MANIFEST.in if it is not.  (There is a command at the top
# of setup.py to gather all the things that should be there).

# TODO: Also check that this whitelist isn't growning out of date from files
# removed from git.

# TODO: Address the "why?" comments below.

# Files that are in git that should not be in the tarball
git_whitelist = {
    # Git specific dotfiles
    '.gitattributes',
    '.gitignore',
    '.mailmap',
    # Travis
    '.travis.yml',
    # Code of conduct
    'CODE_OF_CONDUCT.md',
    # Nothing from bin/ should be shipped unless we intend to install it. Most
    # of this stuff is for development anyway. To run the tests from the
    # tarball, use setup.py test, or import sympy and run sympy.test() or
    # sympy.doctest().
    'bin/adapt_paths.py',
    'bin/ask_update.py',
    'bin/authors_update.py',
    'bin/coverage_doctest.py',
    'bin/coverage_report.py',
    'bin/build_doc.sh',
    'bin/deploy_doc.sh',
    'bin/diagnose_imports',
    'bin/doctest',
    'bin/generate_test_list.py',
    'bin/get_sympy.py',
    'bin/py.bench',
    'bin/mailmap_update.py',
    'bin/strip_whitespace',
    'bin/sympy_time.py',
    'bin/sympy_time_cache.py',
    'bin/test',
    'bin/test_import',
    'bin/test_import.py',
    'bin/test_isolated',
    'bin/test_travis.sh',
    # The notebooks are not ready for shipping yet. They need to be cleaned
    # up, and preferably doctested.  See also
    # https://github.com/sympy/sympy/issues/6039.
    'examples/advanced/identitysearch_example.ipynb',
    'examples/beginner/plot_advanced.ipynb',
    'examples/beginner/plot_colors.ipynb',
    'examples/beginner/plot_discont.ipynb',
    'examples/beginner/plot_gallery.ipynb',
    'examples/beginner/plot_intro.ipynb',
    'examples/intermediate/limit_examples_advanced.ipynb',
    'examples/intermediate/schwarzschild.ipynb',
    'examples/notebooks/density.ipynb',
    'examples/notebooks/fidelity.ipynb',
    'examples/notebooks/fresnel_integrals.ipynb',
    'examples/notebooks/qubits.ipynb',
    'examples/notebooks/sho1d_example.ipynb',
    'examples/notebooks/spin.ipynb',
    'examples/notebooks/trace.ipynb',
    'examples/notebooks/README.txt',
    # This stuff :)
    'release/.gitignore',
    'release/README.md',
    'release/Vagrantfile',
    'release/fabfile.py',
    # This is just a distribute version of setup.py. Used mainly for setup.py
    # develop, which we don't care about in the release tarball
    'setupegg.py',
    # Example on how to use tox to test Sympy. For development.
    'tox.ini.sample',
    }

# Files that should be in the tarball should not be in git
```
### 12 - sympy/stats/crv_types.py:

Start line: 230, End line: 254

```python
#-------------------------------------------------------------------------------
# Benini distribution ----------------------------------------------------------


class BeniniDistribution(SingleContinuousDistribution):
    _argnames = ('alpha', 'beta', 'sigma')

    @staticmethod
    def check(alpha, beta, sigma):
        _value_check(alpha > 0, "Shape parameter Alpha must be positive.")
        _value_check(beta > 0, "Shape parameter Beta must be positive.")
        _value_check(sigma > 0, "Scale parameter Sigma must be positive.")

    @property
    def set(self):
        return Interval(self.sigma, oo)

    def pdf(self, x):
        alpha, beta, sigma = self.alpha, self.beta, self.sigma
        return (exp(-alpha*log(x/sigma) - beta*log(x/sigma)**2)
               *(alpha/x + 2*beta*log(x/sigma)/x))

    def _moment_generating_function(self, t):
        raise NotImplementedError('The moment generating function of the '
                                  'Benini distribution does not exist.')
```
### 13 - sympy/stats/crv_types.py:

Start line: 867, End line: 887

```python
#-------------------------------------------------------------------------------
# Dagum distribution -----------------------------------------------------------


class DagumDistribution(SingleContinuousDistribution):
    _argnames = ('p', 'a', 'b')

    @staticmethod
    def check(p, a, b):
        _value_check(p > 0, "Shape parameter p must be positive.")
        _value_check(a > 0, "Shape parameter a must be positive.")
        _value_check(b > 0, "Scale parameter b must be positive.")

    def pdf(self, x):
        p, a, b = self.p, self.a, self.b
        return a*p/x*((x/b)**(a*p)/(((x/b)**a + 1)**(p + 1)))

    def _cdf(self, x):
        p, a, b = self.p, self.a, self.b
        return Piecewise(((S.One + (S(x)/b)**-a)**-p, x>=0),
                    (S.Zero, True))
```
### 15 - sympy/stats/crv_types.py:

Start line: 318, End line: 343

```python
#-------------------------------------------------------------------------------
# Beta distribution ------------------------------------------------------------


class BetaDistribution(SingleContinuousDistribution):
    _argnames = ('alpha', 'beta')

    set = Interval(0, 1)

    @staticmethod
    def check(alpha, beta):
        _value_check(alpha > 0, "Shape parameter Alpha must be positive.")
        _value_check(beta > 0, "Shape parameter Beta must be positive.")

    def pdf(self, x):
        alpha, beta = self.alpha, self.beta
        return x**(alpha - 1) * (1 - x)**(beta - 1) / beta_fn(alpha, beta)

    def sample(self):
        return random.betavariate(self.alpha, self.beta)

    def _characteristic_function(self, t):
        return hyper((self.alpha,), (self.alpha + self.beta,), I*t)

    def _moment_generating_function(self, t):
        return hyper((self.alpha,), (self.alpha + self.beta,), t)
```
### 20 - sympy/stats/crv_types.py:

Start line: 2167, End line: 2198

```python
#-------------------------------------------------------------------------------
# Normal distribution ----------------------------------------------------------


class NormalDistribution(SingleContinuousDistribution):
    _argnames = ('mean', 'std')

    @staticmethod
    def check(mean, std):
        _value_check(std > 0, "Standard deviation must be positive")

    def pdf(self, x):
        return exp(-(x - self.mean)**2 / (2*self.std**2)) / (sqrt(2*pi)*self.std)

    def sample(self):
        return random.normalvariate(self.mean, self.std)

    def _cdf(self, x):
        mean, std = self.mean, self.std
        return erf(sqrt(2)*(-mean + x)/(2*std))/2 + S.Half

    def _characteristic_function(self, t):
        mean, std = self.mean, self.std
        return exp(I*mean*t - std**2*t**2/2)

    def _moment_generating_function(self, t):
        mean, std = self.mean, self.std
        return exp(mean*t + std**2*t**2/2)

    def _quantile(self, p):
        mean, std = self.mean, self.std
        return mean + std*sqrt(2)*erfinv(2*p - 1)
```
### 25 - sympy/stats/joint_rv_types.py:

Start line: 108, End line: 125

```python
#-------------------------------------------------------------------------------
# Multivariate Laplace distribution ---------------------------------------------------------

class MultivariateLaplaceDistribution(JointDistribution):
    _argnames = ['mu', 'sigma']
    is_Continuous=True

    @property
    def set(self):
        k = len(self.mu)
        return S.Reals**k

    def check(self, mu, sigma):
        _value_check(len(mu) == len(sigma.col(0)),
            "Size of the mean vector and covariance matrix are incorrect.")
        #check if covariance matrix is positive definite or not.
        _value_check((i > 0 for i in sigma.eigenvals().keys()),
            "The covariance matrix must be positive definite. ")
```
### 28 - sympy/stats/crv_types.py:

Start line: 2624, End line: 2640

```python
#-------------------------------------------------------------------------------
# Shifted Gompertz distribution ------------------------------------------------


class ShiftedGompertzDistribution(SingleContinuousDistribution):
    _argnames = ('b', 'eta')

    set = Interval(0, oo)

    @staticmethod
    def check(b, eta):
        _value_check(b > 0, "b must be positive")
        _value_check(eta > 0, "eta must be positive")

    def pdf(self, x):
        b, eta = self.b, self.eta
        return b*exp(-b*x)*exp(-eta*exp(-b*x))*(1+eta*(1-exp(-b*x)))
```
### 30 - sympy/stats/crv_types.py:

Start line: 3039, End line: 3068

```python
#-------------------------------------------------------------------------------
# UniformSum distribution ------------------------------------------------------


class UniformSumDistribution(SingleContinuousDistribution):
    _argnames = ('n',)

    @property
    def set(self):
        return Interval(0, self.n)

    def pdf(self, x):
        n = self.n
        k = Dummy("k")
        return 1/factorial(
            n - 1)*Sum((-1)**k*binomial(n, k)*(x - k)**(n - 1), (k, 0, floor(x)))

    def _cdf(self, x):
        n = self.n
        k = Dummy("k")
        return Piecewise((S.Zero, x < 0),
                        (1/factorial(n)*Sum((-1)**k*binomial(n, k)*(x - k)**(n),
                        (k, 0, floor(x))), x <= n),
                        (S.One, True))

    def _characteristic_function(self, t):
        return ((exp(I*t) - 1) / (I*t))**self.n

    def _moment_generating_function(self, t):
        return ((exp(t) - 1) / t)**self.n
```
### 34 - sympy/stats/crv_types.py:

Start line: 1271, End line: 1290

```python
#-------------------------------------------------------------------------------
# Frechet distribution ---------------------------------------------------------

class FrechetDistribution(SingleContinuousDistribution):
    _argnames = ('a', 's', 'm')

    set = Interval(0, oo)

    def __new__(cls, a, s=1, m=0):
        a, s, m = list(map(sympify, (a, s, m)))
        return Basic.__new__(cls, a, s, m)

    def pdf(self, x):
        a, s, m = self.a, self.s, self.m
        return a/s * ((x-m)/s)**(-1-a) * exp(-((x-m)/s)**(-a))

    def _cdf(self, x):
        a, s, m = self.a, self.s, self.m
        return Piecewise((exp(-((x-m)/s)**(-a)), x >= m),
                        (S.Zero, True))
```
### 35 - sympy/stats/crv_types.py:

Start line: 2463, End line: 2492

```python
#-------------------------------------------------------------------------------
# RaisedCosine distribution ----------------------------------------------------


class RaisedCosineDistribution(SingleContinuousDistribution):
    _argnames = ('mu', 's')

    @property
    def set(self):
        return Interval(self.mu - self.s, self.mu + self.s)

    @staticmethod
    def check(mu, s):
        _value_check(s > 0, "s must be positive")

    def pdf(self, x):
        mu, s = self.mu, self.s
        return Piecewise(
                ((1+cos(pi*(x-mu)/s)) / (2*s), And(mu-s<=x, x<=mu+s)),
                (S.Zero, True))

    def _characteristic_function(self, t):
        mu, s = self.mu, self.s
        return Piecewise((exp(-I*pi*mu/s)/2, Eq(t, -pi/s)),
                         (exp(I*pi*mu/s)/2, Eq(t, pi/s)),
                         (pi**2*sin(s*t)*exp(I*mu*t) / (s*t*(pi**2 - s**2*t**2)), True))

    def _moment_generating_function(self, t):
        mu, s = self.mu, self.s
        return pi**2 * sinh(s*t) * exp(mu*t) /  (s*t*(pi**2 + s**2*t**2))
```
### 37 - sympy/stats/crv_types.py:

Start line: 493, End line: 509

```python
#-------------------------------------------------------------------------------
# Beta prime distribution ------------------------------------------------------


class BetaPrimeDistribution(SingleContinuousDistribution):
    _argnames = ('alpha', 'beta')

    @staticmethod
    def check(alpha, beta):
        _value_check(alpha > 0, "Shape parameter Alpha must be positive.")
        _value_check(beta > 0, "Shape parameter Beta must be positive.")

    set = Interval(0, oo)

    def pdf(self, x):
        alpha, beta = self.alpha, self.beta
        return x**(alpha - 1)*(1 + x)**(-alpha - beta)/beta_fn(alpha, beta)
```
### 39 - sympy/stats/joint_rv_types.py:

Start line: 66, End line: 93

```python
#-------------------------------------------------------------------------------
# Multivariate Normal distribution ---------------------------------------------------------

class MultivariateNormalDistribution(JointDistribution):
    _argnames = ['mu', 'sigma']

    is_Continuous=True

    @property
    def set(self):
        k = len(self.mu)
        return S.Reals**k

    def check(self, mu, sigma):
        _value_check(len(mu) == len(sigma.col(0)),
            "Size of the mean vector and covariance matrix are incorrect.")
        #check if covariance matrix is positive definite or not.
        _value_check((i > 0 for i in sigma.eigenvals().keys()),
            "The covariance matrix must be positive definite. ")

    def pdf(self, *args):
        mu, sigma = self.mu, self.sigma
        k = len(mu)
        args = ImmutableMatrix(args)
        x = args - mu
        return  S(1)/sqrt((2*pi)**(k)*det(sigma))*exp(
            -S(1)/2*x.transpose()*(sigma.inv()*\
                x))[0]
```
### 40 - sympy/stats/crv_types.py:

Start line: 2686, End line: 2703

```python
#-------------------------------------------------------------------------------
# StudentT distribution --------------------------------------------------------


class StudentTDistribution(SingleContinuousDistribution):
    _argnames = ('nu',)

    def pdf(self, x):
        nu = self.nu
        return 1/(sqrt(nu)*beta_fn(S(1)/2, nu/2))*(1 + x**2/nu)**(-(nu + 1)/2)

    def _cdf(self, x):
        nu = self.nu
        return S.Half + x*gamma((nu+1)/2)*hyper((S.Half, (nu+1)/2),
                                (S(3)/2,), -x**2/nu)/(sqrt(pi*nu)*gamma(nu/2))

    def _moment_generating_function(self, t):
        raise NotImplementedError('The moment generating function for the Student-T distribution is undefined.')
```
### 42 - sympy/stats/crv_types.py:

Start line: 2936, End line: 2977

```python
#-------------------------------------------------------------------------------
# Uniform distribution ---------------------------------------------------------


class UniformDistribution(SingleContinuousDistribution):
    _argnames = ('left', 'right')

    def pdf(self, x):
        left, right = self.left, self.right
        return Piecewise(
            (S.One/(right - left), And(left <= x, x <= right)),
            (S.Zero, True)
        )

    def _cdf(self, x):
        left, right = self.left, self.right
        return Piecewise(
            (S.Zero, x < left),
            ((x - left)/(right - left), x <= right),
            (S.One, True)
        )

    def _characteristic_function(self, t):
        left, right = self.left, self.right
        return Piecewise(((exp(I*t*right) - exp(I*t*left)) / (I*t*(right - left)), Ne(t, 0)),
                         (S.One, True))

    def _moment_generating_function(self, t):
        left, right = self.left, self.right
        return Piecewise(((exp(t*right) - exp(t*left)) / (t * (right - left)), Ne(t, 0)),
                         (S.One, True))

    def expectation(self, expr, var, **kwargs):
        from sympy import Max, Min
        kwargs['evaluate'] = True
        result = SingleContinuousDistribution.expectation(self, expr, var, **kwargs)
        result = result.subs({Max(self.left, self.right): self.right,
                              Min(self.left, self.right): self.left})
        return result

    def sample(self):
        return random.uniform(self.left, self.right)
```
### 44 - sympy/stats/crv_types.py:

Start line: 3208, End line: 3227

```python
#-------------------------------------------------------------------------------
# Weibull distribution ---------------------------------------------------------


class WeibullDistribution(SingleContinuousDistribution):
    _argnames = ('alpha', 'beta')

    set = Interval(0, oo)

    @staticmethod
    def check(alpha, beta):
        _value_check(alpha > 0, "Alpha must be positive")
        _value_check(beta > 0, "Beta must be positive")

    def pdf(self, x):
        alpha, beta = self.alpha, self.beta
        return beta * (x/alpha)**(beta - 1) * exp(-(x/alpha)**beta) / alpha

    def sample(self):
        return random.weibullvariate(self.alpha, self.beta)
```
### 45 - sympy/stats/crv_types.py:

Start line: 2291, End line: 2327

```python
#-------------------------------------------------------------------------------
# Pareto distribution ----------------------------------------------------------


class ParetoDistribution(SingleContinuousDistribution):
    _argnames = ('xm', 'alpha')

    @property
    def set(self):
        return Interval(self.xm, oo)

    @staticmethod
    def check(xm, alpha):
        _value_check(xm > 0, "Xm must be positive")
        _value_check(alpha > 0, "Alpha must be positive")

    def pdf(self, x):
        xm, alpha = self.xm, self.alpha
        return alpha * xm**alpha / x**(alpha + 1)

    def sample(self):
        return random.paretovariate(self.alpha)

    def _cdf(self, x):
        xm, alpha = self.xm, self.alpha
        return Piecewise(
                (S.One - xm**alpha/x**alpha, x>=xm),
                (0, True),
        )

    def _moment_generating_function(self, t):
        xm, alpha = self.xm, self.alpha
        return alpha * (-xm*t)**alpha * uppergamma(-alpha, -xm*t)

    def _characteristic_function(self, t):
        xm, alpha = self.xm, self.alpha
        return alpha * (-I * xm * t) ** alpha * uppergamma(-alpha, -I * xm * t)
```
### 47 - sympy/stats/crv_types.py:

Start line: 1120, End line: 1143

```python
#-------------------------------------------------------------------------------
# F distribution ---------------------------------------------------------------


class FDistributionDistribution(SingleContinuousDistribution):
    _argnames = ('d1', 'd2')

    set = Interval(0, oo)

    @staticmethod
    def check(d1, d2):
        _value_check((d1 > 0, d1.is_integer),
            "Degrees of freedom d1 must be positive integer.")
        _value_check((d2 > 0, d2.is_integer),
            "Degrees of freedom d2 must be positive integer.")

    def pdf(self, x):
        d1, d2 = self.d1, self.d2
        return (sqrt((d1*x)**d1*d2**d2 / (d1*x+d2)**(d1+d2))
               / (x * beta_fn(d1/2, d2/2)))

    def _moment_generating_function(self, t):
        raise NotImplementedError('The moment generating function for the '
                                  'F-distribution does not exist.')
```
### 49 - sympy/stats/crv_types.py:

Start line: 1545, End line: 1567

```python
#-------------------------------------------------------------------------------
# Gumbel distribution --------------------------------------------------------


class GumbelDistribution(SingleContinuousDistribution):
    _argnames = ('beta', 'mu')

    set = Interval(-oo, oo)

    def pdf(self, x):
        beta, mu = self.beta, self.mu
        z = (x - mu)/beta
        return (1/beta)*exp(-(z + exp(-z)))

    def _cdf(self, x):
        beta, mu = self.beta, self.mu
        return exp(-exp((mu - x)/beta))

    def _characteristic_function(self, t):
        return gamma(1 - I*self.beta*t) * exp(I*self.mu*t)

    def _moment_generating_function(self, t):
        return gamma(1 - self.beta*t) * exp(I*self.mu*t)
```
### 54 - sympy/stats/crv_types.py:

Start line: 1012, End line: 1046

```python
#-------------------------------------------------------------------------------
# Exponential distribution -----------------------------------------------------


class ExponentialDistribution(SingleContinuousDistribution):
    _argnames = ('rate',)

    set  = Interval(0, oo)

    @staticmethod
    def check(rate):
        _value_check(rate > 0, "Rate must be positive.")

    def pdf(self, x):
        return self.rate * exp(-self.rate*x)

    def sample(self):
        return random.expovariate(self.rate)

    def _cdf(self, x):
        return Piecewise(
                (S.One - exp(-self.rate*x), x >= 0),
                (0, True),
        )

    def _characteristic_function(self, t):
        rate = self.rate
        return rate / (rate - I*t)

    def _moment_generating_function(self, t):
        rate = self.rate
        return rate / (rate - t)

    def _quantile(self, p):
        return -log(1-p)/self.rate
```
### 55 - sympy/stats/crv_types.py:

Start line: 2547, End line: 2570

```python
#-------------------------------------------------------------------------------
# Rayleigh distribution --------------------------------------------------------


class RayleighDistribution(SingleContinuousDistribution):
    _argnames = ('sigma',)

    set = Interval(0, oo)

    def pdf(self, x):
        sigma = self.sigma
        return x/sigma**2*exp(-x**2/(2*sigma**2))

    def _cdf(self, x):
        sigma = self.sigma
        return 1 - exp(-(x**2/(2*sigma**2)))

    def _characteristic_function(self, t):
        sigma = self.sigma
        return 1 - sigma*t*exp(-sigma**2*t**2/2) * sqrt(pi/2) * (erfi(sigma*t/sqrt(2)) - I)

    def _moment_generating_function(self, t):
        sigma = self.sigma
        return 1 + sigma*t*exp(sigma**2*t**2/2) * sqrt(pi/2) * (erf(sigma*t/sqrt(2)) + 1)
```
### 56 - sympy/stats/crv_types.py:

Start line: 562, End line: 588

```python
#-------------------------------------------------------------------------------
# Cauchy distribution ----------------------------------------------------------


class CauchyDistribution(SingleContinuousDistribution):
    _argnames = ('x0', 'gamma')

    @staticmethod
    def check(x0, gamma):
        _value_check(gamma > 0, "Scale parameter Gamma must be positive.")

    def pdf(self, x):
        return 1/(pi*self.gamma*(1 + ((x - self.x0)/self.gamma)**2))

    def _cdf(self, x):
        x0, gamma = self.x0, self.gamma
        return (1/pi)*atan((x - x0)/gamma) + S.Half

    def _characteristic_function(self, t):
        return exp(self.x0 * I * t -  self.gamma * Abs(t))

    def _moment_generating_function(self, t):
        raise NotImplementedError("The moment generating function for the "
                                  "Cauchy distribution does not exist.")

    def _quantile(self, p):
        return self.x0 + self.gamma*tan(pi*(p - S.Half))
```
### 57 - sympy/stats/crv_types.py:

Start line: 2079, End line: 2096

```python
#-------------------------------------------------------------------------------
# Nakagami distribution --------------------------------------------------------


class NakagamiDistribution(SingleContinuousDistribution):
    _argnames = ('mu', 'omega')

    set = Interval(0, oo)

    def pdf(self, x):
        mu, omega = self.mu, self.omega
        return 2*mu**mu/(gamma(mu)*omega**mu)*x**(2*mu - 1)*exp(-mu/omega*x**2)

    def _cdf(self, x):
        mu, omega = self.mu, self.omega
        return Piecewise(
                    (lowergamma(mu, (mu/omega)*x**2)/gamma(mu), x > 0),
                    (S.Zero, True))
```
### 60 - sympy/stats/joint_rv_types.py:

Start line: 191, End line: 215

```python
#-------------------------------------------------------------------------------
# Multivariate Normal Gamma distribution ---------------------------------------------------------

class NormalGammaDistribution(JointDistribution):

    _argnames = ['mu', 'lamda', 'alpha', 'beta']
    is_Continuous=True

    def check(self, mu, lamda, alpha, beta):
        _value_check(mu.is_real, "Location must be real.")
        _value_check(lamda > 0, "Lambda must be positive")
        _value_check(alpha > 0, "alpha must be positive")
        _value_check(beta > 0, "beta must be positive")

    @property
    def set(self):
        return S.Reals*Interval(0, S.Infinity)

    def pdf(self, x, tau):
        beta, alpha, lamda = self.beta, self.alpha, self.lamda
        mu = self.mu

        return beta**alpha*sqrt(lamda)/(gamma(alpha)*sqrt(2*pi))*\
        tau**(alpha - S(1)/2)*exp(-1*beta*tau)*\
        exp(-1*(lamda*tau*(x - mu)**2)/S(2))
```
### 65 - sympy/stats/joint_rv_types.py:

Start line: 323, End line: 344

```python
Dirichlet = MultivariateBeta

#-------------------------------------------------------------------------------
# Multivariate Ewens distribution ---------------------------------------------------------

class MultivariateEwensDistribution(JointDistribution):

    _argnames = ['n', 'theta']
    is_Discrete = True
    is_Continuous = False

    def check(self, n, theta):
        _value_check(isinstance(n, Integer) and (n > 0) == True,
                        "sample size should be positive integer.")
        _value_check(theta.is_positive, "mutation rate should be positive.")

    @property
    def set(self):
        prod_set = Range(0, self.n//1 + 1)
        for i in range(2, self.n + 1):
            prod_set *= Range(0, self.n//i + 1)
        return prod_set
```
### 66 - sympy/stats/crv_types.py:

Start line: 782, End line: 811

```python
#-------------------------------------------------------------------------------
# Chi squared distribution -----------------------------------------------------


class ChiSquaredDistribution(SingleContinuousDistribution):
    _argnames = ('k',)

    @staticmethod
    def check(k):
        _value_check(k > 0, "Number of degrees of freedom (k) must be positive.")
        _value_check(k.is_integer, "Number of degrees of freedom (k) must be an integer.")

    set = Interval(0, oo)

    def pdf(self, x):
        k = self.k
        return 1/(2**(k/2)*gamma(k/2))*x**(k/2 - 1)*exp(-x/2)

    def _cdf(self, x):
        k = self.k
        return Piecewise(
                (S.One/gamma(k/2)*lowergamma(k/2, x/2), x >= 0),
                (0, True)
        )

    def _characteristic_function(self, t):
        return (1 - 2*I*t)**(-self.k/2)

    def  _moment_generating_function(self, t):
        return (1 - 2*t)**(-self.k/2)
```
### 67 - sympy/stats/crv_types.py:

Start line: 402, End line: 421

```python
#-------------------------------------------------------------------------------
# Noncentral Beta distribution ------------------------------------------------------------


class BetaNoncentralDistribution(SingleContinuousDistribution):
    _argnames = ('alpha', 'beta', 'lamda')

    set = Interval(0, 1)

    @staticmethod
    def check(alpha, beta, lamda):
        _value_check(alpha > 0, "Shape parameter Alpha must be positive.")
        _value_check(beta > 0, "Shape parameter Beta must be positive.")
        _value_check(lamda >= 0, "Noncentrality parameter Lambda must be positive")

    def pdf(self, x):
        alpha, beta, lamda = self.alpha, self.beta, self.lamda
        k = Dummy("k")
        return Sum(exp(-lamda / 2) * (lamda / 2)**k * x**(alpha + k - 1) *(
            1 - x)**(beta - 1) / (factorial(k) * beta_fn(alpha + k, beta)), (k, 0, oo))
```
### 70 - sympy/stats/crv_types.py:

Start line: 635, End line: 658

```python
#-------------------------------------------------------------------------------
# Chi distribution -------------------------------------------------------------


class ChiDistribution(SingleContinuousDistribution):
    _argnames = ('k',)

    @staticmethod
    def check(k):
        _value_check(k > 0, "Number of degrees of freedom (k) must be positive.")
        _value_check(k.is_integer, "Number of degrees of freedom (k) must be an integer.")

    set = Interval(0, oo)

    def pdf(self, x):
        return 2**(1 - self.k/2)*x**(self.k - 1)*exp(-x**2/2)/gamma(self.k/2)

    def _characteristic_function(self, t):
        k = self.k

        part_1 = hyper((k/2,), (S(1)/2,), -t**2/2)
        part_2 = I*t*sqrt(2)*gamma((k+1)/2)/gamma(k/2)
        part_3 = hyper(((k+1)/2,), (S(3)/2,), -t**2/2)
        return part_1 + part_2*part_3
```
