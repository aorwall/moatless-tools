# sympy__sympy-24539

| **sympy/sympy** | `193e3825645d93c73e31cdceb6d742cc6919624d` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 1824 |
| **Any found context length** | 1824 |
| **Avg pos** | 7.0 |
| **Min pos** | 7 |
| **Max pos** | 7 |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sympy/polys/rings.py b/sympy/polys/rings.py
--- a/sympy/polys/rings.py
+++ b/sympy/polys/rings.py
@@ -616,10 +616,13 @@ def set_ring(self, new_ring):
             return new_ring.from_dict(self, self.ring.domain)
 
     def as_expr(self, *symbols):
-        if symbols and len(symbols) != self.ring.ngens:
-            raise ValueError("not enough symbols, expected %s got %s" % (self.ring.ngens, len(symbols)))
-        else:
+        if not symbols:
             symbols = self.ring.symbols
+        elif len(symbols) != self.ring.ngens:
+            raise ValueError(
+                "Wrong number of symbols, expected %s got %s" %
+                (self.ring.ngens, len(symbols))
+            )
 
         return expr_from_dict(self.as_expr_dict(), *symbols)
 

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sympy/polys/rings.py | 619 | 621 | 7 | 1 | 1824


## Problem Statement

```
`PolyElement.as_expr()` not accepting symbols
The method `PolyElement.as_expr()`

https://github.com/sympy/sympy/blob/193e3825645d93c73e31cdceb6d742cc6919624d/sympy/polys/rings.py#L618-L624

is supposed to let you set the symbols you want to use, but, as it stands, either you pass the wrong number of symbols, and get an error message, or you pass the right number of symbols, and it ignores them, using `self.ring.symbols` instead:

\`\`\`python
>>> from sympy import ring, ZZ, symbols
>>> R, x, y, z = ring("x,y,z", ZZ)
>>> f = 3*x**2*y - x*y*z + 7*z**3 + 1
>>> U, V, W = symbols("u,v,w")
>>> f.as_expr(U, V, W)
3*x**2*y - x*y*z + 7*z**3 + 1
\`\`\`

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 sympy/polys/rings.py** | 787 | 833| 419 | 419 | 17629 | 
| 2 | **1 sympy/polys/rings.py** | 2291 | 2333| 280 | 699 | 17629 | 
| 3 | **1 sympy/polys/rings.py** | 2374 | 2404| 247 | 946 | 17629 | 
| 4 | **1 sympy/polys/rings.py** | 181 | 194| 121 | 1067 | 17629 | 
| 5 | **1 sympy/polys/rings.py** | 2335 | 2372| 244 | 1311 | 17629 | 
| 6 | **1 sympy/polys/rings.py** | 1984 | 1998| 139 | 1450 | 17629 | 
| **-> 7 <-** | **1 sympy/polys/rings.py** | 609 | 651| 374 | 1824 | 17629 | 
| 8 | **1 sympy/polys/rings.py** | 370 | 401| 277 | 2101 | 17629 | 
| 9 | **1 sympy/polys/rings.py** | 2150 | 2164| 230 | 2331 | 17629 | 
| 10 | **1 sympy/polys/rings.py** | 1362 | 1393| 296 | 2627 | 17629 | 
| 11 | **1 sympy/polys/rings.py** | 706 | 733| 231 | 2858 | 17629 | 
| 12 | **1 sympy/polys/rings.py** | 1471 | 1525| 391 | 3249 | 17629 | 
| 13 | **1 sympy/polys/rings.py** | 1198 | 1241| 305 | 3554 | 17629 | 
| 14 | **1 sympy/polys/rings.py** | 835 | 914| 515 | 4069 | 17629 | 
| 15 | **1 sympy/polys/rings.py** | 2060 | 2091| 199 | 4268 | 17629 | 
| 16 | **1 sympy/polys/rings.py** | 2166 | 2198| 274 | 4542 | 17629 | 
| 17 | **1 sympy/polys/rings.py** | 2406 | 2468| 448 | 4990 | 17629 | 
| 18 | **1 sympy/polys/rings.py** | 2093 | 2119| 180 | 5170 | 17629 | 
| 19 | 2 sympy/polys/specialpolys.py | 308 | 310| 128 | 5298 | 22085 | 
| 20 | **2 sympy/polys/rings.py** | 764 | 785| 188 | 5486 | 22085 | 
| 21 | **2 sympy/polys/rings.py** | 329 | 349| 155 | 5641 | 22085 | 
| 22 | **2 sympy/polys/rings.py** | 1329 | 1360| 267 | 5908 | 22085 | 
| 23 | **2 sympy/polys/rings.py** | 2121 | 2148| 265 | 6173 | 22085 | 
| 24 | **2 sympy/polys/rings.py** | 735 | 762| 193 | 6366 | 22085 | 
| 25 | 2 sympy/polys/specialpolys.py | 312 | 314| 127 | 6493 | 22085 | 
| 26 | 2 sympy/polys/specialpolys.py | 304 | 306| 174 | 6667 | 22085 | 
| 27 | 3 sympy/polys/solvers.py | 132 | 184| 438 | 7105 | 25598 | 
| 28 | **3 sympy/polys/rings.py** | 2000 | 2058| 384 | 7489 | 25598 | 
| 29 | **3 sympy/polys/rings.py** | 1761 | 1787| 197 | 7686 | 25598 | 
| 30 | 3 sympy/polys/specialpolys.py | 316 | 318| 379 | 8065 | 25598 | 
| 31 | 3 sympy/polys/specialpolys.py | 320 | 326| 243 | 8308 | 25598 | 
| 32 | **3 sympy/polys/rings.py** | 1279 | 1299| 190 | 8498 | 25598 | 
| 33 | 4 sympy/core/symbol.py | 323 | 385| 535 | 9033 | 32745 | 
| 34 | **4 sympy/polys/rings.py** | 66 | 95| 245 | 9278 | 32745 | 
| 35 | **4 sympy/polys/rings.py** | 1301 | 1327| 211 | 9489 | 32745 | 
| 36 | 4 sympy/core/symbol.py | 298 | 321| 300 | 9789 | 32745 | 
| 37 | 4 sympy/polys/specialpolys.py | 328 | 330| 423 | 10212 | 32745 | 
| 38 | 4 sympy/polys/specialpolys.py | 296 | 302| 117 | 10329 | 32745 | 
| 39 | **4 sympy/polys/rings.py** | 35 | 64| 239 | 10568 | 32745 | 
| 40 | **4 sympy/polys/rings.py** | 1664 | 1685| 174 | 10742 | 32745 | 
| 41 | **4 sympy/polys/rings.py** | 2256 | 2289| 288 | 11030 | 32745 | 
| 42 | **4 sympy/polys/rings.py** | 1243 | 1277| 278 | 11308 | 32745 | 
| 43 | **4 sympy/polys/rings.py** | 128 | 179| 401 | 11709 | 32745 | 
| 44 | **4 sympy/polys/rings.py** | 97 | 126| 247 | 11956 | 32745 | 
| 45 | 4 sympy/polys/specialpolys.py | 332 | 341| 244 | 12200 | 32745 | 
| 46 | 5 sympy/polys/__init__.py | 3 | 66| 856 | 13056 | 34350 | 
| 47 | 6 sympy/core/expr.py | 3981 | 4005| 208 | 13264 | 69050 | 
| 48 | 7 sympy/polys/ring_series.py | 1874 | 1957| 817 | 14081 | 87216 | 
| 49 | 8 sympy/polys/fields.py | 603 | 632| 306 | 14387 | 92308 | 
| 50 | **8 sympy/polys/rings.py** | 653 | 677| 212 | 14599 | 92308 | 
| 51 | **8 sympy/polys/rings.py** | 351 | 368| 132 | 14731 | 92308 | 
| 52 | 8 sympy/core/expr.py | 342 | 406| 560 | 15291 | 92308 | 
| 53 | **8 sympy/polys/rings.py** | 1 | 33| 325 | 15616 | 92308 | 
| 54 | 8 sympy/core/symbol.py | 274 | 296| 199 | 15815 | 92308 | 
| 55 | **8 sympy/polys/rings.py** | 556 | 579| 198 | 16013 | 92308 | 
| 56 | 8 sympy/core/symbol.py | 586 | 701| 1078 | 17091 | 92308 | 
| 57 | 9 sympy/functions/elementary/hyperbolic.py | 1 | 22| 257 | 17348 | 110710 | 
| 58 | 10 sympy/matrices/expressions/matexpr.py | 1 | 33| 282 | 17630 | 117314 | 
| 59 | 10 sympy/core/symbol.py | 1 | 22| 152 | 17782 | 117314 | 
| 60 | 10 sympy/core/expr.py | 305 | 340| 504 | 18286 | 117314 | 
| 61 | 10 sympy/core/expr.py | 2427 | 2456| 237 | 18523 | 117314 | 
| 62 | 11 sympy/concrete/expr_with_limits.py | 1 | 19| 189 | 18712 | 122315 | 
| 63 | 12 sympy/solvers/bivariate.py | 19 | 44| 235 | 18947 | 127582 | 
| 64 | 13 sympy/polys/polytools.py | 67 | 4263| 226 | 19173 | 180334 | 
| 65 | 13 sympy/core/symbol.py | 702 | 796| 741 | 19914 | 180334 | 


## Patch

```diff
diff --git a/sympy/polys/rings.py b/sympy/polys/rings.py
--- a/sympy/polys/rings.py
+++ b/sympy/polys/rings.py
@@ -616,10 +616,13 @@ def set_ring(self, new_ring):
             return new_ring.from_dict(self, self.ring.domain)
 
     def as_expr(self, *symbols):
-        if symbols and len(symbols) != self.ring.ngens:
-            raise ValueError("not enough symbols, expected %s got %s" % (self.ring.ngens, len(symbols)))
-        else:
+        if not symbols:
             symbols = self.ring.symbols
+        elif len(symbols) != self.ring.ngens:
+            raise ValueError(
+                "Wrong number of symbols, expected %s got %s" %
+                (self.ring.ngens, len(symbols))
+            )
 
         return expr_from_dict(self.as_expr_dict(), *symbols)
 

```

## Test Patch

```diff
diff --git a/sympy/polys/tests/test_rings.py b/sympy/polys/tests/test_rings.py
--- a/sympy/polys/tests/test_rings.py
+++ b/sympy/polys/tests/test_rings.py
@@ -259,11 +259,11 @@ def test_PolyElement_as_expr():
     assert f != g
     assert f.as_expr() == g
 
-    X, Y, Z = symbols("x,y,z")
-    g = 3*X**2*Y - X*Y*Z + 7*Z**3 + 1
+    U, V, W = symbols("u,v,w")
+    g = 3*U**2*V - U*V*W + 7*W**3 + 1
 
     assert f != g
-    assert f.as_expr(X, Y, Z) == g
+    assert f.as_expr(U, V, W) == g
 
     raises(ValueError, lambda: f.as_expr(X))
 

```


## Code snippets

### 1 - sympy/polys/rings.py:

Start line: 787, End line: 833

```python
class PolyElement(DomainElement, DefaultPrinting, CantSympify, dict):

    def str(self, printer, precedence, exp_pattern, mul_symbol):
        if not self:
            return printer._print(self.ring.domain.zero)
        prec_mul = precedence["Mul"]
        prec_atom = precedence["Atom"]
        ring = self.ring
        symbols = ring.symbols
        ngens = ring.ngens
        zm = ring.zero_monom
        sexpvs = []
        for expv, coeff in self.terms():
            negative = ring.domain.is_negative(coeff)
            sign = " - " if negative else " + "
            sexpvs.append(sign)
            if expv == zm:
                scoeff = printer._print(coeff)
                if negative and scoeff.startswith("-"):
                    scoeff = scoeff[1:]
            else:
                if negative:
                    coeff = -coeff
                if coeff != self.ring.domain.one:
                    scoeff = printer.parenthesize(coeff, prec_mul, strict=True)
                else:
                    scoeff = ''
            sexpv = []
            for i in range(ngens):
                exp = expv[i]
                if not exp:
                    continue
                symbol = printer.parenthesize(symbols[i], prec_atom, strict=True)
                if exp != 1:
                    if exp != int(exp) or exp < 0:
                        sexp = printer.parenthesize(exp, prec_atom, strict=False)
                    else:
                        sexp = exp
                    sexpv.append(exp_pattern % (symbol, sexp))
                else:
                    sexpv.append('%s' % symbol)
            if scoeff:
                sexpv = [scoeff] + sexpv
            sexpvs.append(mul_symbol.join(sexpv))
        if sexpvs[0] in [" + ", " - "]:
            head = sexpvs.pop(0)
            if head == " - ":
                sexpvs.insert(0, "-")
        return "".join(sexpvs)
```
### 2 - sympy/polys/rings.py:

Start line: 2291, End line: 2333

```python
class PolyElement(DomainElement, DefaultPrinting, CantSympify, dict):

    def evaluate(self, x, a=None):
        f = self

        if isinstance(x, list) and a is None:
            (X, a), x = x[0], x[1:]
            f = f.evaluate(X, a)

            if not x:
                return f
            else:
                x = [ (Y.drop(X), a) for (Y, a) in x ]
                return f.evaluate(x)

        ring = f.ring
        i = ring.index(x)
        a = ring.domain.convert(a)

        if ring.ngens == 1:
            result = ring.domain.zero

            for (n,), coeff in f.iterterms():
                result += coeff*a**n

            return result
        else:
            poly = ring.drop(x).zero

            for monom, coeff in f.iterterms():
                n, monom = monom[i], monom[:i] + monom[i+1:]
                coeff = coeff*a**n

                if monom in poly:
                    coeff = coeff + poly[monom]

                    if coeff:
                        poly[monom] = coeff
                    else:
                        del poly[monom]
                else:
                    if coeff:
                        poly[monom] = coeff

            return poly
```
### 3 - sympy/polys/rings.py:

Start line: 2374, End line: 2404

```python
class PolyElement(DomainElement, DefaultPrinting, CantSympify, dict):

    def compose(f, x, a=None):
        ring = f.ring
        poly = ring.zero
        gens_map = dict(zip(ring.gens, range(ring.ngens)))

        if a is not None:
            replacements = [(x, a)]
        else:
            if isinstance(x, list):
                replacements = list(x)
            elif isinstance(x, dict):
                replacements = sorted(list(x.items()), key=lambda k: gens_map[k[0]])
            else:
                raise ValueError("expected a generator, value pair a sequence of such pairs")

        for k, (x, g) in enumerate(replacements):
            replacements[k] = (gens_map[x], ring.ring_new(g))

        for monom, coeff in f.iterterms():
            monom = list(monom)
            subpoly = ring.one

            for i, g in replacements:
                n, monom[i] = monom[i], 0
                if n:
                    subpoly *= g**n

            subpoly = subpoly.mul_term((tuple(monom), coeff))
            poly += subpoly

        return poly
```
### 4 - sympy/polys/rings.py:

Start line: 181, End line: 194

```python
def _parse_symbols(symbols):
    if isinstance(symbols, str):
        return _symbols(symbols, seq=True) if symbols else ()
    elif isinstance(symbols, Expr):
        return (symbols,)
    elif is_sequence(symbols):
        if all(isinstance(s, str) for s in symbols):
            return _symbols(symbols)
        elif all(isinstance(s, Expr) for s in symbols):
            return symbols

    raise GeneratorsError("expected a string, Symbol or expression or a non-empty sequence of strings, Symbols or expressions")

_ring_cache: dict[Any, Any] = {}
```
### 5 - sympy/polys/rings.py:

Start line: 2335, End line: 2372

```python
class PolyElement(DomainElement, DefaultPrinting, CantSympify, dict):

    def subs(self, x, a=None):
        f = self

        if isinstance(x, list) and a is None:
            for X, a in x:
                f = f.subs(X, a)
            return f

        ring = f.ring
        i = ring.index(x)
        a = ring.domain.convert(a)

        if ring.ngens == 1:
            result = ring.domain.zero

            for (n,), coeff in f.iterterms():
                result += coeff*a**n

            return ring.ground_new(result)
        else:
            poly = ring.zero

            for monom, coeff in f.iterterms():
                n, monom = monom[i], monom[:i] + (0,) + monom[i+1:]
                coeff = coeff*a**n

                if monom in poly:
                    coeff = coeff + poly[monom]

                    if coeff:
                        poly[monom] = coeff
                    else:
                        del poly[monom]
                else:
                    if coeff:
                        poly[monom] = coeff

            return poly
```
### 6 - sympy/polys/rings.py:

Start line: 1984, End line: 1998

```python
class PolyElement(DomainElement, DefaultPrinting, CantSympify, dict):

    def quo_ground(f, x):
        domain = f.ring.domain

        if not x:
            raise ZeroDivisionError('polynomial division')
        if not f or x == domain.one:
            return f

        if domain.is_Field:
            quo = domain.quo
            terms = [ (monom, quo(coeff, x)) for monom, coeff in f.iterterms() ]
        else:
            terms = [ (monom, coeff // x) for monom, coeff in f.iterterms() if not (coeff % x) ]

        return f.new(terms)
```
### 7 - sympy/polys/rings.py:

Start line: 609, End line: 651

```python
class PolyElement(DomainElement, DefaultPrinting, CantSympify, dict):

    def set_ring(self, new_ring):
        if self.ring == new_ring:
            return self
        elif self.ring.symbols != new_ring.symbols:
            terms = list(zip(*_dict_reorder(self, self.ring.symbols, new_ring.symbols)))
            return new_ring.from_terms(terms, self.ring.domain)
        else:
            return new_ring.from_dict(self, self.ring.domain)

    def as_expr(self, *symbols):
        if symbols and len(symbols) != self.ring.ngens:
            raise ValueError("not enough symbols, expected %s got %s" % (self.ring.ngens, len(symbols)))
        else:
            symbols = self.ring.symbols

        return expr_from_dict(self.as_expr_dict(), *symbols)

    def as_expr_dict(self):
        to_sympy = self.ring.domain.to_sympy
        return {monom: to_sympy(coeff) for monom, coeff in self.iterterms()}

    def clear_denoms(self):
        domain = self.ring.domain

        if not domain.is_Field or not domain.has_assoc_Ring:
            return domain.one, self

        ground_ring = domain.get_ring()
        common = ground_ring.one
        lcm = ground_ring.lcm
        denom = domain.denom

        for coeff in self.values():
            common = lcm(common, denom(coeff))

        poly = self.new([ (k, v*common) for k, v in self.items() ])
        return common, poly

    def strip_zero(self):
        """Eliminate monomials with zero coefficient. """
        for k, v in list(self.items()):
            if not v:
                del self[k]
```
### 8 - sympy/polys/rings.py:

Start line: 370, End line: 401

```python
class PolyRing(DefaultPrinting, IPolys):

    def _rebuild_expr(self, expr, mapping):
        domain = self.domain

        def _rebuild(expr):
            generator = mapping.get(expr)

            if generator is not None:
                return generator
            elif expr.is_Add:
                return reduce(add, list(map(_rebuild, expr.args)))
            elif expr.is_Mul:
                return reduce(mul, list(map(_rebuild, expr.args)))
            else:
                # XXX: Use as_base_exp() to handle Pow(x, n) and also exp(n)
                # XXX: E can be a generator e.g. sring([exp(2)]) -> ZZ[E]
                base, exp = expr.as_base_exp()
                if exp.is_Integer and exp > 1:
                    return _rebuild(base)**int(exp)
                else:
                    return self.ground_new(domain.convert(expr))

        return _rebuild(sympify(expr))

    def from_expr(self, expr):
        mapping = dict(list(zip(self.symbols, self.gens)))

        try:
            poly = self._rebuild_expr(expr, mapping)
        except CoercionFailed:
            raise ValueError("expected an expression convertible to a polynomial in %s, got %s" % (self, expr))
        else:
            return self.ring_new(poly)
```
### 9 - sympy/polys/rings.py:

Start line: 2150, End line: 2164

```python
class PolyElement(DomainElement, DefaultPrinting, CantSympify, dict):

    def _gcd_monom(f, g):
        ring = f.ring
        ground_gcd = ring.domain.gcd
        ground_quo = ring.domain.quo
        monomial_gcd = ring.monomial_gcd
        monomial_ldiv = ring.monomial_ldiv
        mf, cf = list(f.iterterms())[0]
        _mgcd, _cgcd = mf, cf
        for mg, cg in g.iterterms():
            _mgcd = monomial_gcd(_mgcd, mg)
            _cgcd = ground_gcd(_cgcd, cg)
        h = f.new([(_mgcd, _cgcd)])
        cff = f.new([(monomial_ldiv(mf, _mgcd), ground_quo(cf, _cgcd))])
        cfg = f.new([(monomial_ldiv(mg, _mgcd), ground_quo(cg, _cgcd)) for mg, cg in g.iterterms()])
        return h, cff, cfg
```
### 10 - sympy/polys/rings.py:

Start line: 1362, End line: 1393

```python
class PolyElement(DomainElement, DefaultPrinting, CantSympify, dict):

    def _term_div(self):
        zm = self.ring.zero_monom
        domain = self.ring.domain
        domain_quo = domain.quo
        monomial_div = self.ring.monomial_div

        if domain.is_Field:
            def term_div(a_lm_a_lc, b_lm_b_lc):
                a_lm, a_lc = a_lm_a_lc
                b_lm, b_lc = b_lm_b_lc
                if b_lm == zm: # apparently this is a very common case
                    monom = a_lm
                else:
                    monom = monomial_div(a_lm, b_lm)
                if monom is not None:
                    return monom, domain_quo(a_lc, b_lc)
                else:
                    return None
        else:
            def term_div(a_lm_a_lc, b_lm_b_lc):
                a_lm, a_lc = a_lm_a_lc
                b_lm, b_lc = b_lm_b_lc
                if b_lm == zm: # apparently this is a very common case
                    monom = a_lm
                else:
                    monom = monomial_div(a_lm, b_lm)
                if not (monom is None or a_lc % b_lc):
                    return monom, domain_quo(a_lc, b_lc)
                else:
                    return None

        return term_div
```
### 11 - sympy/polys/rings.py:

Start line: 706, End line: 733

```python
class PolyElement(DomainElement, DefaultPrinting, CantSympify, dict):

    def sort_key(self):
        return (len(self), self.terms())

    def _cmp(p1, p2, op):
        if isinstance(p2, p1.ring.dtype):
            return op(p1.sort_key(), p2.sort_key())
        else:
            return NotImplemented

    def __lt__(p1, p2):
        return p1._cmp(p2, lt)
    def __le__(p1, p2):
        return p1._cmp(p2, le)
    def __gt__(p1, p2):
        return p1._cmp(p2, gt)
    def __ge__(p1, p2):
        return p1._cmp(p2, ge)

    def _drop(self, gen):
        ring = self.ring
        i = ring.index(gen)

        if ring.ngens == 1:
            return i, ring.domain
        else:
            symbols = list(ring.symbols)
            del symbols[i]
            return i, ring.clone(symbols=symbols)
```
### 12 - sympy/polys/rings.py:

Start line: 1471, End line: 1525

```python
class PolyElement(DomainElement, DefaultPrinting, CantSympify, dict):

    def rem(self, G):
        f = self
        if isinstance(G, PolyElement):
            G = [G]
        if not all(G):
            raise ZeroDivisionError("polynomial division")
        ring = f.ring
        domain = ring.domain
        zero = domain.zero
        monomial_mul = ring.monomial_mul
        r = ring.zero
        term_div = f._term_div()
        ltf = f.LT
        f = f.copy()
        get = f.get
        while f:
            for g in G:
                tq = term_div(ltf, g.LT)
                if tq is not None:
                    m, c = tq
                    for mg, cg in g.iterterms():
                        m1 = monomial_mul(mg, m)
                        c1 = get(m1, zero) - c*cg
                        if not c1:
                            del f[m1]
                        else:
                            f[m1] = c1
                    ltm = f.leading_expv()
                    if ltm is not None:
                        ltf = ltm, f[ltm]

                    break
            else:
                ltm, ltc = ltf
                if ltm in r:
                    r[ltm] += ltc
                else:
                    r[ltm] = ltc
                del f[ltm]
                ltm = f.leading_expv()
                if ltm is not None:
                    ltf = ltm, f[ltm]

        return r

    def quo(f, G):
        return f.div(G)[0]

    def exquo(f, G):
        q, r = f.div(G)

        if not r:
            return q
        else:
            raise ExactQuotientFailed(f, G)
```
### 13 - sympy/polys/rings.py:

Start line: 1198, End line: 1241

```python
class PolyElement(DomainElement, DefaultPrinting, CantSympify, dict):

    def _pow_generic(self, n):
        p = self.ring.one
        c = self

        while True:
            if n & 1:
                p = p*c
                n -= 1
                if not n:
                    break

            c = c.square()
            n = n // 2

        return p

    def _pow_multinomial(self, n):
        multinomials = multinomial_coefficients(len(self), n).items()
        monomial_mulpow = self.ring.monomial_mulpow
        zero_monom = self.ring.zero_monom
        terms = self.items()
        zero = self.ring.domain.zero
        poly = self.ring.zero

        for multinomial, multinomial_coeff in multinomials:
            product_monom = zero_monom
            product_coeff = multinomial_coeff

            for exp, (monom, coeff) in zip(multinomial, terms):
                if exp:
                    product_monom = monomial_mulpow(product_monom, monom, exp)
                    product_coeff *= coeff**exp

            monom = tuple(product_monom)
            coeff = product_coeff

            coeff = poly.get(monom, zero) + coeff

            if coeff:
                poly[monom] = coeff
            elif monom in poly:
                del poly[monom]

        return poly
```
### 14 - sympy/polys/rings.py:

Start line: 835, End line: 914

```python
class PolyElement(DomainElement, DefaultPrinting, CantSympify, dict):

    @property
    def is_generator(self):
        return self in self.ring._gens_set

    @property
    def is_ground(self):
        return not self or (len(self) == 1 and self.ring.zero_monom in self)

    @property
    def is_monomial(self):
        return not self or (len(self) == 1 and self.LC == 1)

    @property
    def is_term(self):
        return len(self) <= 1

    @property
    def is_negative(self):
        return self.ring.domain.is_negative(self.LC)

    @property
    def is_positive(self):
        return self.ring.domain.is_positive(self.LC)

    @property
    def is_nonnegative(self):
        return self.ring.domain.is_nonnegative(self.LC)

    @property
    def is_nonpositive(self):
        return self.ring.domain.is_nonpositive(self.LC)

    @property
    def is_zero(f):
        return not f

    @property
    def is_one(f):
        return f == f.ring.one

    @property
    def is_monic(f):
        return f.ring.domain.is_one(f.LC)

    @property
    def is_primitive(f):
        return f.ring.domain.is_one(f.content())

    @property
    def is_linear(f):
        return all(sum(monom) <= 1 for monom in f.itermonoms())

    @property
    def is_quadratic(f):
        return all(sum(monom) <= 2 for monom in f.itermonoms())

    @property
    def is_squarefree(f):
        if not f.ring.ngens:
            return True
        return f.ring.dmp_sqf_p(f)

    @property
    def is_irreducible(f):
        if not f.ring.ngens:
            return True
        return f.ring.dmp_irreducible_p(f)

    @property
    def is_cyclotomic(f):
        if f.ring.is_univariate:
            return f.ring.dup_cyclotomic_p(f)
        else:
            raise MultivariatePolynomialError("cyclotomic polynomial")

    def __neg__(self):
        return self.new([ (monom, -coeff) for monom, coeff in self.iterterms() ])

    def __pos__(self):
        return self
```
### 15 - sympy/polys/rings.py:

Start line: 2060, End line: 2091

```python
class PolyElement(DomainElement, DefaultPrinting, CantSympify, dict):

    def deflate(f, *G):
        ring = f.ring
        polys = [f] + list(G)

        J = [0]*ring.ngens

        for p in polys:
            for monom in p.itermonoms():
                for i, m in enumerate(monom):
                    J[i] = igcd(J[i], m)

        for i, b in enumerate(J):
            if not b:
                J[i] = 1

        J = tuple(J)

        if all(b == 1 for b in J):
            return J, polys

        H = []

        for p in polys:
            h = ring.zero

            for I, coeff in p.iterterms():
                N = [ i // j for i, j in zip(I, J) ]
                h[tuple(N)] = coeff

            H.append(h)

        return J, H
```
### 16 - sympy/polys/rings.py:

Start line: 2166, End line: 2198

```python
class PolyElement(DomainElement, DefaultPrinting, CantSympify, dict):

    def _gcd(f, g):
        ring = f.ring

        if ring.domain.is_QQ:
            return f._gcd_QQ(g)
        elif ring.domain.is_ZZ:
            return f._gcd_ZZ(g)
        else: # TODO: don't use dense representation (port PRS algorithms)
            return ring.dmp_inner_gcd(f, g)

    def _gcd_ZZ(f, g):
        return heugcd(f, g)

    def _gcd_QQ(self, g):
        f = self
        ring = f.ring
        new_ring = ring.clone(domain=ring.domain.get_ring())

        cf, f = f.clear_denoms()
        cg, g = g.clear_denoms()

        f = f.set_ring(new_ring)
        g = g.set_ring(new_ring)

        h, cff, cfg = f._gcd_ZZ(g)

        h = h.set_ring(ring)
        c, h = h.LC, h.monic()

        cff = cff.set_ring(ring).mul_ground(ring.domain.quo(c, cf))
        cfg = cfg.set_ring(ring).mul_ground(ring.domain.quo(c, cg))

        return h, cff, cfg
```
### 17 - sympy/polys/rings.py:

Start line: 2406, End line: 2468

```python
class PolyElement(DomainElement, DefaultPrinting, CantSympify, dict):

    # TODO: following methods should point to polynomial
    # representation independent algorithm implementations.

    def pdiv(f, g):
        return f.ring.dmp_pdiv(f, g)

    def prem(f, g):
        return f.ring.dmp_prem(f, g)

    def pquo(f, g):
        return f.ring.dmp_quo(f, g)

    def pexquo(f, g):
        return f.ring.dmp_exquo(f, g)

    def half_gcdex(f, g):
        return f.ring.dmp_half_gcdex(f, g)

    def gcdex(f, g):
        return f.ring.dmp_gcdex(f, g)

    def subresultants(f, g):
        return f.ring.dmp_subresultants(f, g)

    def resultant(f, g):
        return f.ring.dmp_resultant(f, g)

    def discriminant(f):
        return f.ring.dmp_discriminant(f)

    def decompose(f):
        if f.ring.is_univariate:
            return f.ring.dup_decompose(f)
        else:
            raise MultivariatePolynomialError("polynomial decomposition")

    def shift(f, a):
        if f.ring.is_univariate:
            return f.ring.dup_shift(f, a)
        else:
            raise MultivariatePolynomialError("polynomial shift")

    def sturm(f):
        if f.ring.is_univariate:
            return f.ring.dup_sturm(f)
        else:
            raise MultivariatePolynomialError("sturm sequence")

    def gff_list(f):
        return f.ring.dmp_gff_list(f)

    def sqf_norm(f):
        return f.ring.dmp_sqf_norm(f)

    def sqf_part(f):
        return f.ring.dmp_sqf_part(f)

    def sqf_list(f, all=False):
        return f.ring.dmp_sqf_list(f, all=all)

    def factor_list(f):
        return f.ring.dmp_factor_list(f)
```
### 18 - sympy/polys/rings.py:

Start line: 2093, End line: 2119

```python
class PolyElement(DomainElement, DefaultPrinting, CantSympify, dict):

    def inflate(f, J):
        poly = f.ring.zero

        for I, coeff in f.iterterms():
            N = [ i*j for i, j in zip(I, J) ]
            poly[tuple(N)] = coeff

        return poly

    def lcm(self, g):
        f = self
        domain = f.ring.domain

        if not domain.is_Field:
            fc, f = f.primitive()
            gc, g = g.primitive()
            c = domain.lcm(fc, gc)

        h = (f*g).quo(f.gcd(g))

        if not domain.is_Field:
            return h.mul_ground(c)
        else:
            return h.monic()

    def gcd(f, g):
        return f.cofactors(g)[0]
```
### 20 - sympy/polys/rings.py:

Start line: 764, End line: 785

```python
class PolyElement(DomainElement, DefaultPrinting, CantSympify, dict):

    def drop_to_ground(self, gen):
        if self.ring.ngens == 1:
            raise ValueError("Cannot drop only generator to ground")

        i, ring = self._drop_to_ground(gen)
        poly = ring.zero
        gen = ring.domain.gens[0]

        for monom, coeff in self.iterterms():
            mon = monom[:i] + monom[i+1:]
            if mon not in poly:
                poly[mon] = (gen**monom[i]).mul_ground(coeff)
            else:
                poly[mon] += (gen**monom[i]).mul_ground(coeff)

        return poly

    def to_dense(self):
        return dmp_from_dict(self, self.ring.ngens-1, self.ring.domain)

    def to_dict(self):
        return dict(self)
```
### 21 - sympy/polys/rings.py:

Start line: 329, End line: 349

```python
class PolyRing(DefaultPrinting, IPolys):

    def ring_new(self, element):
        if isinstance(element, PolyElement):
            if self == element.ring:
                return element
            elif isinstance(self.domain, PolynomialRing) and self.domain.ring == element.ring:
                return self.ground_new(element)
            else:
                raise NotImplementedError("conversion")
        elif isinstance(element, str):
            raise NotImplementedError("parsing")
        elif isinstance(element, dict):
            return self.from_dict(element)
        elif isinstance(element, list):
            try:
                return self.from_terms(element)
            except ValueError:
                return self.from_list(element)
        elif isinstance(element, Expr):
            return self.from_expr(element)
        else:
            return self.ground_new(element)
```
### 22 - sympy/polys/rings.py:

Start line: 1329, End line: 1360

```python
class PolyElement(DomainElement, DefaultPrinting, CantSympify, dict):

    def __truediv__(p1, p2):
        ring = p1.ring

        if not p2:
            raise ZeroDivisionError("polynomial division")
        elif isinstance(p2, ring.dtype):
            if p2.is_monomial:
                return p1*(p2**(-1))
            else:
                return p1.quo(p2)
        elif isinstance(p2, PolyElement):
            if isinstance(ring.domain, PolynomialRing) and ring.domain.ring == p2.ring:
                pass
            elif isinstance(p2.ring.domain, PolynomialRing) and p2.ring.domain.ring == ring:
                return p2.__rtruediv__(p1)
            else:
                return NotImplemented

        try:
            p2 = ring.domain_new(p2)
        except CoercionFailed:
            return NotImplemented
        else:
            return p1.quo_ground(p2)

    def __rtruediv__(p1, p2):
        return NotImplemented

    __floordiv__ = __truediv__
    __rfloordiv__ = __rtruediv__

    # TODO: use // (__floordiv__) for exquo()?
```
### 23 - sympy/polys/rings.py:

Start line: 2121, End line: 2148

```python
class PolyElement(DomainElement, DefaultPrinting, CantSympify, dict):

    def cofactors(f, g):
        if not f and not g:
            zero = f.ring.zero
            return zero, zero, zero
        elif not f:
            h, cff, cfg = f._gcd_zero(g)
            return h, cff, cfg
        elif not g:
            h, cfg, cff = g._gcd_zero(f)
            return h, cff, cfg
        elif len(f) == 1:
            h, cff, cfg = f._gcd_monom(g)
            return h, cff, cfg
        elif len(g) == 1:
            h, cfg, cff = g._gcd_monom(f)
            return h, cff, cfg

        J, (f, g) = f.deflate(g)
        h, cff, cfg = f._gcd(g)

        return (h.inflate(J), cff.inflate(J), cfg.inflate(J))

    def _gcd_zero(f, g):
        one, zero = f.ring.one, f.ring.zero
        if g.is_nonnegative:
            return g, zero, one
        else:
            return -g, zero, -one
```
### 24 - sympy/polys/rings.py:

Start line: 735, End line: 762

```python
class PolyElement(DomainElement, DefaultPrinting, CantSympify, dict):

    def drop(self, gen):
        i, ring = self._drop(gen)

        if self.ring.ngens == 1:
            if self.is_ground:
                return self.coeff(1)
            else:
                raise ValueError("Cannot drop %s" % gen)
        else:
            poly = ring.zero

            for k, v in self.items():
                if k[i] == 0:
                    K = list(k)
                    del K[i]
                    poly[tuple(K)] = v
                else:
                    raise ValueError("Cannot drop %s" % gen)

            return poly

    def _drop_to_ground(self, gen):
        ring = self.ring
        i = ring.index(gen)

        symbols = list(ring.symbols)
        del symbols[i]
        return i, ring.clone(symbols=symbols, domain=ring[i])
```
### 28 - sympy/polys/rings.py:

Start line: 2000, End line: 2058

```python
class PolyElement(DomainElement, DefaultPrinting, CantSympify, dict):

    def quo_term(f, term):
        monom, coeff = term

        if not coeff:
            raise ZeroDivisionError("polynomial division")
        elif not f:
            return f.ring.zero
        elif monom == f.ring.zero_monom:
            return f.quo_ground(coeff)

        term_div = f._term_div()

        terms = [ term_div(t, term) for t in f.iterterms() ]
        return f.new([ t for t in terms if t is not None ])

    def trunc_ground(f, p):
        if f.ring.domain.is_ZZ:
            terms = []

            for monom, coeff in f.iterterms():
                coeff = coeff % p

                if coeff > p // 2:
                    coeff = coeff - p

                terms.append((monom, coeff))
        else:
            terms = [ (monom, coeff % p) for monom, coeff in f.iterterms() ]

        poly = f.new(terms)
        poly.strip_zero()
        return poly

    rem_ground = trunc_ground

    def extract_ground(self, g):
        f = self
        fc = f.content()
        gc = g.content()

        gcd = f.ring.domain.gcd(fc, gc)

        f = f.quo_ground(gcd)
        g = g.quo_ground(gcd)

        return gcd, f, g

    def _norm(f, norm_func):
        if not f:
            return f.ring.domain.zero
        else:
            ground_abs = f.ring.domain.abs
            return norm_func([ ground_abs(coeff) for coeff in f.itercoeffs() ])

    def max_norm(f):
        return f._norm(max)

    def l1_norm(f):
        return f._norm(sum)
```
### 29 - sympy/polys/rings.py:

Start line: 1761, End line: 1787

```python
class PolyElement(DomainElement, DefaultPrinting, CantSympify, dict):

    @property
    def LT(self):
        expv = self.leading_expv()
        if expv is None:
            return (self.ring.zero_monom, self.ring.domain.zero)
        else:
            return (expv, self._get_coeff(expv))

    def leading_term(self):
        """Leading term as a polynomial element.

        Examples
        ========

        >>> from sympy.polys.rings import ring
        >>> from sympy.polys.domains import ZZ

        >>> _, x, y = ring('x, y', ZZ)
        >>> (3*x*y + y**2).leading_term()
        3*x*y

        """
        p = self.ring.zero
        expv = self.leading_expv()
        if expv is not None:
            p[expv] = self[expv]
        return p
```
### 32 - sympy/polys/rings.py:

Start line: 1279, End line: 1299

```python
class PolyElement(DomainElement, DefaultPrinting, CantSympify, dict):

    def __divmod__(p1, p2):
        ring = p1.ring

        if not p2:
            raise ZeroDivisionError("polynomial division")
        elif isinstance(p2, ring.dtype):
            return p1.div(p2)
        elif isinstance(p2, PolyElement):
            if isinstance(ring.domain, PolynomialRing) and ring.domain.ring == p2.ring:
                pass
            elif isinstance(p2.ring.domain, PolynomialRing) and p2.ring.domain.ring == ring:
                return p2.__rdivmod__(p1)
            else:
                return NotImplemented

        try:
            p2 = ring.domain_new(p2)
        except CoercionFailed:
            return NotImplemented
        else:
            return (p1.quo_ground(p2), p1.rem_ground(p2))
```
### 34 - sympy/polys/rings.py:

Start line: 66, End line: 95

```python
@public
def xring(symbols, domain, order=lex):
    """Construct a polynomial ring returning ``(ring, (x_1, ..., x_n))``.

    Parameters
    ==========

    symbols : str
        Symbol/Expr or sequence of str, Symbol/Expr (non-empty)
    domain : :class:`~.Domain` or coercible
    order : :class:`~.MonomialOrder` or coercible, optional, defaults to ``lex``

    Examples
    ========

    >>> from sympy.polys.rings import xring
    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.orderings import lex

    >>> R, (x, y, z) = xring("x,y,z", ZZ, lex)
    >>> R
    Polynomial ring in x, y, z over ZZ with lex order
    >>> x + y + z
    x + y + z
    >>> type(_)
    <class 'sympy.polys.rings.PolyElement'>

    """
    _ring = PolyRing(symbols, domain, order)
    return (_ring, _ring.gens)
```
### 35 - sympy/polys/rings.py:

Start line: 1301, End line: 1327

```python
class PolyElement(DomainElement, DefaultPrinting, CantSympify, dict):

    def __rdivmod__(p1, p2):
        return NotImplemented

    def __mod__(p1, p2):
        ring = p1.ring

        if not p2:
            raise ZeroDivisionError("polynomial division")
        elif isinstance(p2, ring.dtype):
            return p1.rem(p2)
        elif isinstance(p2, PolyElement):
            if isinstance(ring.domain, PolynomialRing) and ring.domain.ring == p2.ring:
                pass
            elif isinstance(p2.ring.domain, PolynomialRing) and p2.ring.domain.ring == ring:
                return p2.__rmod__(p1)
            else:
                return NotImplemented

        try:
            p2 = ring.domain_new(p2)
        except CoercionFailed:
            return NotImplemented
        else:
            return p1.rem_ground(p2)

    def __rmod__(p1, p2):
        return NotImplemented
```
### 39 - sympy/polys/rings.py:

Start line: 35, End line: 64

```python
@public
def ring(symbols, domain, order=lex):
    """Construct a polynomial ring returning ``(ring, x_1, ..., x_n)``.

    Parameters
    ==========

    symbols : str
        Symbol/Expr or sequence of str, Symbol/Expr (non-empty)
    domain : :class:`~.Domain` or coercible
    order : :class:`~.MonomialOrder` or coercible, optional, defaults to ``lex``

    Examples
    ========

    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.orderings import lex

    >>> R, x, y, z = ring("x,y,z", ZZ, lex)
    >>> R
    Polynomial ring in x, y, z over ZZ with lex order
    >>> x + y + z
    x + y + z
    >>> type(_)
    <class 'sympy.polys.rings.PolyElement'>

    """
    _ring = PolyRing(symbols, domain, order)
    return (_ring,) + _ring.gens
```
### 40 - sympy/polys/rings.py:

Start line: 1664, End line: 1685

```python
class PolyElement(DomainElement, DefaultPrinting, CantSympify, dict):

    def leading_expv(self):
        """Leading monomial tuple according to the monomial ordering.

        Examples
        ========

        >>> from sympy.polys.rings import ring
        >>> from sympy.polys.domains import ZZ

        >>> _, x, y, z = ring('x, y, z', ZZ)
        >>> p = x**4 + x**3*y + x**2*z**2 + z**7
        >>> p.leading_expv()
        (4, 0, 0)

        """
        if self:
            return self.ring.leading_expv(self)
        else:
            return None

    def _get_coeff(self, expv):
        return self.get(expv, self.ring.domain.zero)
```
### 41 - sympy/polys/rings.py:

Start line: 2256, End line: 2289

```python
class PolyElement(DomainElement, DefaultPrinting, CantSympify, dict):

    def canonical_unit(f):
        domain = f.ring.domain
        return domain.canonical_unit(f.LC)

    def diff(f, x):
        """Computes partial derivative in ``x``.

        Examples
        ========

        >>> from sympy.polys.rings import ring
        >>> from sympy.polys.domains import ZZ

        >>> _, x, y = ring("x,y", ZZ)
        >>> p = x + x**2*y**3
        >>> p.diff(x)
        2*x*y**3 + 1

        """
        ring = f.ring
        i = ring.index(x)
        m = ring.monomial_basis(i)
        g = ring.zero
        for expv, coeff in f.iterterms():
            if expv[i]:
                e = ring.monomial_ldiv(expv, m)
                g[e] = ring.domain_new(coeff*expv[i])
        return g

    def __call__(f, *values):
        if 0 < len(values) <= f.ring.ngens:
            return f.evaluate(list(zip(f.ring.gens, values)))
        else:
            raise ValueError("expected at least 1 and at most %s values, got %s" % (f.ring.ngens, len(values)))
```
### 42 - sympy/polys/rings.py:

Start line: 1243, End line: 1277

```python
class PolyElement(DomainElement, DefaultPrinting, CantSympify, dict):

    def square(self):
        """square of a polynomial

        Examples
        ========

        >>> from sympy.polys.rings import ring
        >>> from sympy.polys.domains import ZZ

        >>> _, x, y = ring('x, y', ZZ)
        >>> p = x + y**2
        >>> p.square()
        x**2 + 2*x*y**2 + y**4

        """
        ring = self.ring
        p = ring.zero
        get = p.get
        keys = list(self.keys())
        zero = ring.domain.zero
        monomial_mul = ring.monomial_mul
        for i in range(len(keys)):
            k1 = keys[i]
            pk = self[k1]
            for j in range(i):
                k2 = keys[j]
                exp = monomial_mul(k1, k2)
                p[exp] = get(exp, zero) + pk*self[k2]
        p = p.imul_num(2)
        get = p.get
        for k, v in self.items():
            k2 = monomial_mul(k, k)
            p[k2] = get(k2, zero) + v**2
        p.strip_zero()
        return p
```
### 43 - sympy/polys/rings.py:

Start line: 128, End line: 179

```python
@public
def sring(exprs, *symbols, **options):
    """Construct a ring deriving generators and domain from options and input expressions.

    Parameters
    ==========

    exprs : :class:`~.Expr` or sequence of :class:`~.Expr` (sympifiable)
    symbols : sequence of :class:`~.Symbol`/:class:`~.Expr`
    options : keyword arguments understood by :class:`~.Options`

    Examples
    ========

    >>> from sympy import sring, symbols

    >>> x, y, z = symbols("x,y,z")
    >>> R, f = sring(x + 2*y + 3*z)
    >>> R
    Polynomial ring in x, y, z over ZZ with lex order
    >>> f
    x + 2*y + 3*z
    >>> type(_)
    <class 'sympy.polys.rings.PolyElement'>

    """
    single = False

    if not is_sequence(exprs):
        exprs, single = [exprs], True

    exprs = list(map(sympify, exprs))
    opt = build_options(symbols, options)

    # TODO: rewrite this so that it doesn't use expand() (see poly()).
    reps, opt = _parallel_dict_from_expr(exprs, opt)

    if opt.domain is None:
        coeffs = sum([ list(rep.values()) for rep in reps ], [])

        opt.domain, coeffs_dom = construct_domain(coeffs, opt=opt)

        coeff_map = dict(zip(coeffs, coeffs_dom))
        reps = [{m: coeff_map[c] for m, c in rep.items()} for rep in reps]

    _ring = PolyRing(opt.gens, opt.domain, opt.order)
    polys = list(map(_ring.from_dict, reps))

    if single:
        return (_ring, polys[0])
    else:
        return (_ring, polys)
```
### 44 - sympy/polys/rings.py:

Start line: 97, End line: 126

```python
@public
def vring(symbols, domain, order=lex):
    """Construct a polynomial ring and inject ``x_1, ..., x_n`` into the global namespace.

    Parameters
    ==========

    symbols : str
        Symbol/Expr or sequence of str, Symbol/Expr (non-empty)
    domain : :class:`~.Domain` or coercible
    order : :class:`~.MonomialOrder` or coercible, optional, defaults to ``lex``

    Examples
    ========

    >>> from sympy.polys.rings import vring
    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.orderings import lex

    >>> vring("x,y,z", ZZ, lex)
    Polynomial ring in x, y, z over ZZ with lex order
    >>> x + y + z # noqa:
    x + y + z
    >>> type(_)
    <class 'sympy.polys.rings.PolyElement'>

    """
    _ring = PolyRing(symbols, domain, order)
    pollute([ sym.name for sym in _ring.symbols ], _ring.gens)
    return _ring
```
### 50 - sympy/polys/rings.py:

Start line: 653, End line: 677

```python
class PolyElement(DomainElement, DefaultPrinting, CantSympify, dict):

    def __eq__(p1, p2):
        """Equality test for polynomials.

        Examples
        ========

        >>> from sympy.polys.domains import ZZ
        >>> from sympy.polys.rings import ring

        >>> _, x, y = ring('x, y', ZZ)
        >>> p1 = (x + y)**2 + (x - y)**2
        >>> p1 == 4*x*y
        False
        >>> p1 == 2*(x**2 + y**2)
        True

        """
        if not p2:
            return not p1
        elif isinstance(p2, PolyElement) and p2.ring == p1.ring:
            return dict.__eq__(p1, p2)
        elif len(p1) > 1:
            return False
        else:
            return p1.get(p1.ring.zero_monom) == p2
```
### 51 - sympy/polys/rings.py:

Start line: 351, End line: 368

```python
class PolyRing(DefaultPrinting, IPolys):

    __call__ = ring_new

    def from_dict(self, element, orig_domain=None):
        domain_new = self.domain_new
        poly = self.zero

        for monom, coeff in element.items():
            coeff = domain_new(coeff, orig_domain)
            if coeff:
                poly[monom] = coeff

        return poly

    def from_terms(self, element, orig_domain=None):
        return self.from_dict(dict(element), orig_domain)

    def from_list(self, element):
        return self.from_dict(dmp_to_dict(element, self.ngens-1, self.domain))
```
### 53 - sympy/polys/rings.py:

Start line: 1, End line: 33

```python
"""Sparse polynomial rings. """

from __future__ import annotations
from typing import Any

from operator import add, mul, lt, le, gt, ge
from functools import reduce
from types import GeneratorType

from sympy.core.expr import Expr
from sympy.core.numbers import igcd, oo
from sympy.core.symbol import Symbol, symbols as _symbols
from sympy.core.sympify import CantSympify, sympify
from sympy.ntheory.multinomial import multinomial_coefficients
from sympy.polys.compatibility import IPolys
from sympy.polys.constructor import construct_domain
from sympy.polys.densebasic import dmp_to_dict, dmp_from_dict
from sympy.polys.domains.domainelement import DomainElement
from sympy.polys.domains.polynomialring import PolynomialRing
from sympy.polys.heuristicgcd import heugcd
from sympy.polys.monomials import MonomialOps
from sympy.polys.orderings import lex
from sympy.polys.polyerrors import (
    CoercionFailed, GeneratorsError,
    ExactQuotientFailed, MultivariatePolynomialError)
from sympy.polys.polyoptions import (Domain as DomainOpt,
                                     Order as OrderOpt, build_options)
from sympy.polys.polyutils import (expr_from_dict, _dict_reorder,
                                   _parallel_dict_from_expr)
from sympy.printing.defaults import DefaultPrinting
from sympy.utilities import public
from sympy.utilities.iterables import is_sequence
from sympy.utilities.magic import pollute
```
### 55 - sympy/polys/rings.py:

Start line: 556, End line: 579

```python
class PolyElement(DomainElement, DefaultPrinting, CantSympify, dict):
    """Element of multivariate distributed polynomial ring. """

    def new(self, init):
        return self.__class__(init)

    def parent(self):
        return self.ring.to_domain()

    def __getnewargs__(self):
        return (self.ring, list(self.iterterms()))

    _hash = None

    def __hash__(self):
        # XXX: This computes a hash of a dictionary, but currently we don't
        # protect dictionary from being changed so any use site modifications
        # will make hashing go wrong. Use this feature with caution until we
        # figure out how to make a safe API without compromising speed of this
        # low-level class.
        _hash = self._hash
        if _hash is None:
            self._hash = _hash = hash((self.ring, frozenset(self.items())))
        return _hash
```
