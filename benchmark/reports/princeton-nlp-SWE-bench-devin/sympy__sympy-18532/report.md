# sympy__sympy-18532

| **sympy/sympy** | `74227f900b05009d4eed62e34a166228788a32ca` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 1353 |
| **Any found context length** | 1353 |
| **Avg pos** | 3.0 |
| **Min pos** | 3 |
| **Max pos** | 3 |
| **Top file pos** | 2 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sympy/core/basic.py b/sympy/core/basic.py
--- a/sympy/core/basic.py
+++ b/sympy/core/basic.py
@@ -503,12 +503,11 @@ def atoms(self, *types):
         if types:
             types = tuple(
                 [t if isinstance(t, type) else type(t) for t in types])
+        nodes = preorder_traversal(self)
+        if types:
+            result = {node for node in nodes if isinstance(node, types)}
         else:
-            types = (Atom,)
-        result = set()
-        for expr in preorder_traversal(self):
-            if isinstance(expr, types):
-                result.add(expr)
+            result = {node for node in nodes if not node.args}
         return result
 
     @property

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sympy/core/basic.py | 506 | 510 | 3 | 2 | 1353


## Problem Statement

```
expr.atoms() should return objects with no args instead of subclasses of Atom
`expr.atoms()` with no arguments returns subclasses of `Atom` in `expr`. But the correct definition of a leaf node should be that it has no `.args`. 

This should be easy to fix, but one needs to check that this doesn't affect the performance. 


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 sympy/core/expr.py | 3781 | 3823| 311 | 311 | 33277 | 
| 2 | **2 sympy/core/basic.py** | 1826 | 1869| 301 | 612 | 49252 | 
| **-> 3 <-** | **2 sympy/core/basic.py** | 434 | 512| 741 | 1353 | 49252 | 
| 4 | 2 sympy/core/expr.py | 2377 | 2399| 167 | 1520 | 49252 | 
| 5 | 3 sympy/parsing/latex/_parse_latex_antlr.py | 201 | 213| 119 | 1639 | 53637 | 
| 6 | 4 sympy/utilities/lambdify.py | 1031 | 1069| 373 | 2012 | 65065 | 
| 7 | 5 sympy/simplify/epathtools.py | 180 | 224| 288 | 2300 | 67371 | 
| 8 | 6 sympy/printing/dot.py | 278 | 295| 224 | 2524 | 69611 | 
| 9 | 7 sympy/parsing/latex/_antlr/latexparser.py | 2687 | 2711| 175 | 2699 | 99886 | 
| 10 | 7 sympy/parsing/latex/_parse_latex_antlr.py | 281 | 324| 372 | 3071 | 99886 | 
| 11 | 7 sympy/core/expr.py | 800 | 857| 554 | 3625 | 99886 | 
| 12 | 8 sympy/plotting/experimental_lambdify.py | 618 | 678| 546 | 4171 | 105784 | 
| 13 | 9 sympy/physics/quantum/qexpr.py | 306 | 330| 178 | 4349 | 108837 | 
| 14 | 10 sympy/simplify/cse_main.py | 515 | 559| 256 | 4605 | 114802 | 
| 15 | 10 sympy/parsing/latex/_parse_latex_antlr.py | 107 | 121| 111 | 4716 | 114802 | 
| 16 | 10 sympy/simplify/epathtools.py | 118 | 154| 274 | 4990 | 114802 | 
| 17 | **10 sympy/core/basic.py** | 514 | 531| 171 | 5161 | 114802 | 
| 18 | 11 sympy/physics/secondquant.py | 2391 | 2473| 626 | 5787 | 137337 | 
| 19 | 11 sympy/simplify/epathtools.py | 252 | 280| 178 | 5965 | 137337 | 
| 20 | 11 sympy/simplify/cse_main.py | 561 | 625| 427 | 6392 | 137337 | 
| 21 | 11 sympy/core/expr.py | 126 | 146| 178 | 6570 | 137337 | 
| 22 | 11 sympy/parsing/latex/_antlr/latexparser.py | 2716 | 2748| 277 | 6847 | 137337 | 
| 23 | 12 sympy/simplify/radsimp.py | 667 | 720| 405 | 7252 | 147378 | 
| 24 | 12 sympy/simplify/cse_main.py | 422 | 456| 252 | 7504 | 147378 | 
| 25 | 12 sympy/parsing/latex/_antlr/latexparser.py | 2750 | 2774| 175 | 7679 | 147378 | 
| 26 | 13 sympy/simplify/simplify.py | 1141 | 1162| 176 | 7855 | 165802 | 
| 27 | 13 sympy/core/expr.py | 148 | 210| 571 | 8426 | 165802 | 
| 28 | 14 sympy/core/function.py | 2129 | 2191| 564 | 8990 | 193289 | 
| 29 | 14 sympy/physics/quantum/qexpr.py | 294 | 304| 125 | 9115 | 193289 | 
| 30 | 14 sympy/core/expr.py | 229 | 280| 498 | 9613 | 193289 | 
| 31 | 14 sympy/printing/dot.py | 140 | 184| 302 | 9915 | 193289 | 
| 32 | 14 sympy/parsing/latex/_antlr/latexparser.py | 1787 | 1814| 193 | 10108 | 193289 | 


### Hint

```
The docstring should also be updated. 

Hi, can i work on this?

Sure. Did you read https://github.com/sympy/sympy/wiki/Introduction-to-contributing? 

How should I remove .args? Should I try to remove ._args from object instance or add a new attribute to class Atom(), is_leave. Which when assigned as false, will raise attribute error on .args. Or if creating a new object, what attributes should it have?

I think you're misunderstanding the issue. The issue is not to remove .args. Indeed, every SymPy object should have .args in order to be valid. 

The issue is that the `atoms()` method currently uses `x.is_Atom` to check for "atomic" expressions (expressions with no subexpressions), but it really should be checking `not x.args`. It should be a simple one-line fix to the `atoms` function definition, but a new test should be added, and the full test suite run to make sure it doesn't break anything (`./bin/test` from the sympy directory). 

Okay. But, Basic() also return .args to be null. So will not that also appear in the result of .atoms()?

Yes, that's an example of an object with no args but that isn't a subclass of Atom. `atoms` should return that, because it's a leaf in the expression tree. 

Okay, but if I am understanding you correct, won't this test fail?
https://github.com/sympy/sympy/blob/master/sympy/core/tests/test_basic.py#L73

Yes, it would need to be changed. This is a slight redefinition of what `atoms` means (although hopefully not enough of a breaking behavior to require deprecation). 

Can you look over it once and look if it is okay?
https://github.com/sympy/sympy/pull/10246

@asmeurer 
When I ran the full suite of tests, sympy/vector/tests/test_field_functions.py failed on all the tests. 

\`\`\`
     Original-
            if not (types or expr.args):
                result.add(expr)

     Case 1-     
            if not types:
                if isinstance(expr, Atom):
                    result.add(expr)

     Case 2-
            if not (types or expr.args):
                if isinstance(expr, Atom):
                    result.add(expr)
\`\`\`

I saw that fails even on the second case. Then I saw the items that case1 had but case2 did not. Which were all either `C.z <class 'sympy.vector.scalar.BaseScalar'>` or `C.k <class 'sympy.vector.vector.BaseVector'>`. 

Elements of the class sympy.vector.scaler.BaseScalar or class sympy.vector.vector.BaseVector were earlier considered but not now, as they were Atom but had arguments. So what should we do?

I want to fix this if no one is working on it.

I am unable to figure out why 'Atom' has been assigned to 'types' . We can add the result while checking for the types and if there are no types then we can simply add x.args to the result. That way it will return null and we will not be having subclasses of Atom.

ping @asmeurer 

@darkcoderrises I have some fixes at https://github.com/sympy/sympy/pull/10084 which might make your issues go away. Once that is merged you should try merging your branch into master and see if it fixes the problems. 

ok

I merged the pull requests, and now the tests are passing. What should be my next step.
https://github.com/sympy/sympy/pull/10246

I am working on this issue
```

## Patch

```diff
diff --git a/sympy/core/basic.py b/sympy/core/basic.py
--- a/sympy/core/basic.py
+++ b/sympy/core/basic.py
@@ -503,12 +503,11 @@ def atoms(self, *types):
         if types:
             types = tuple(
                 [t if isinstance(t, type) else type(t) for t in types])
+        nodes = preorder_traversal(self)
+        if types:
+            result = {node for node in nodes if isinstance(node, types)}
         else:
-            types = (Atom,)
-        result = set()
-        for expr in preorder_traversal(self):
-            if isinstance(expr, types):
-                result.add(expr)
+            result = {node for node in nodes if not node.args}
         return result
 
     @property

```

## Test Patch

```diff
diff --git a/sympy/codegen/tests/test_cnodes.py b/sympy/codegen/tests/test_cnodes.py
--- a/sympy/codegen/tests/test_cnodes.py
+++ b/sympy/codegen/tests/test_cnodes.py
@@ -1,6 +1,6 @@
 from sympy.core.symbol import symbols
 from sympy.printing.ccode import ccode
-from sympy.codegen.ast import Declaration, Variable, float64, int64
+from sympy.codegen.ast import Declaration, Variable, float64, int64, String
 from sympy.codegen.cnodes import (
     alignof, CommaOperator, goto, Label, PreDecrement, PostDecrement, PreIncrement, PostIncrement,
     sizeof, union, struct
@@ -66,7 +66,7 @@ def test_sizeof():
     assert ccode(sz) == 'sizeof(%s)' % typename
     assert sz.func(*sz.args) == sz
     assert not sz.is_Atom
-    assert all(atom == typename for atom in sz.atoms())
+    assert sz.atoms() == {String('unsigned int'), String('sizeof')}
 
 
 def test_struct():
diff --git a/sympy/core/tests/test_basic.py b/sympy/core/tests/test_basic.py
--- a/sympy/core/tests/test_basic.py
+++ b/sympy/core/tests/test_basic.py
@@ -137,7 +137,7 @@ def test_subs_with_unicode_symbols():
 
 
 def test_atoms():
-    assert b21.atoms() == set()
+    assert b21.atoms() == set([Basic()])
 
 
 def test_free_symbols_empty():

```


## Code snippets

### 1 - sympy/core/expr.py:

Start line: 3781, End line: 3823

```python
class AtomicExpr(Atom, Expr):
    """
    A parent class for object which are both atoms and Exprs.

    For example: Symbol, Number, Rational, Integer, ...
    But not: Add, Mul, Pow, ...
    """
    is_number = False
    is_Atom = True

    __slots__ = ()

    def _eval_derivative(self, s):
        if self == s:
            return S.One
        return S.Zero

    def _eval_derivative_n_times(self, s, n):
        from sympy import Piecewise, Eq
        from sympy import Tuple, MatrixExpr
        from sympy.matrices.common import MatrixCommon
        if isinstance(s, (MatrixCommon, Tuple, Iterable, MatrixExpr)):
            return super(AtomicExpr, self)._eval_derivative_n_times(s, n)
        if self == s:
            return Piecewise((self, Eq(n, 0)), (1, Eq(n, 1)), (0, True))
        else:
            return Piecewise((self, Eq(n, 0)), (0, True))

    def _eval_is_polynomial(self, syms):
        return True

    def _eval_is_rational_function(self, syms):
        return True

    def _eval_is_algebraic_expr(self, syms):
        return True

    def _eval_nseries(self, x, n, logx):
        return self

    @property
    def expr_free_symbols(self):
        return {self}
```
### 2 - sympy/core/basic.py:

Start line: 1826, End line: 1869

```python
class Atom(Basic):
    """
    A parent class for atomic things. An atom is an expression with no subexpressions.

    Examples
    ========

    Symbol, Number, Rational, Integer, ...
    But not: Add, Mul, Pow, ...
    """

    is_Atom = True

    __slots__ = ()

    def matches(self, expr, repl_dict={}, old=False):
        if self == expr:
            return repl_dict

    def xreplace(self, rule, hack2=False):
        return rule.get(self, self)

    def doit(self, **hints):
        return self

    @classmethod
    def class_key(cls):
        return 2, 0, cls.__name__

    @cacheit
    def sort_key(self, order=None):
        return self.class_key(), (1, (str(self),)), S.One.sort_key(), S.One

    def _eval_simplify(self, **kwargs):
        return self

    @property
    def _sorted_args(self):
        # this is here as a safeguard against accidentally using _sorted_args
        # on Atoms -- they cannot be rebuilt as atom.func(*atom._sorted_args)
        # since there are no args. So the calling routine should be checking
        # to see that this property is not called for Atoms.
        raise AttributeError('Atoms have no args. It might be necessary'
        ' to make a check for Atoms in the calling code.')
```
### 3 - sympy/core/basic.py:

Start line: 434, End line: 512

```python
class Basic(metaclass=ManagedProperties):

    def atoms(self, *types):
        """Returns the atoms that form the current object.

        By default, only objects that are truly atomic and can't
        be divided into smaller pieces are returned: symbols, numbers,
        and number symbols like I and pi. It is possible to request
        atoms of any type, however, as demonstrated below.

        Examples
        ========

        >>> from sympy import I, pi, sin
        >>> from sympy.abc import x, y
        >>> (1 + x + 2*sin(y + I*pi)).atoms()
        {1, 2, I, pi, x, y}

        If one or more types are given, the results will contain only
        those types of atoms.

        >>> from sympy import Number, NumberSymbol, Symbol
        >>> (1 + x + 2*sin(y + I*pi)).atoms(Symbol)
        {x, y}

        >>> (1 + x + 2*sin(y + I*pi)).atoms(Number)
        {1, 2}

        >>> (1 + x + 2*sin(y + I*pi)).atoms(Number, NumberSymbol)
        {1, 2, pi}

        >>> (1 + x + 2*sin(y + I*pi)).atoms(Number, NumberSymbol, I)
        {1, 2, I, pi}

        Note that I (imaginary unit) and zoo (complex infinity) are special
        types of number symbols and are not part of the NumberSymbol class.

        The type can be given implicitly, too:

        >>> (1 + x + 2*sin(y + I*pi)).atoms(x) # x is a Symbol
        {x, y}

        Be careful to check your assumptions when using the implicit option
        since ``S(1).is_Integer = True`` but ``type(S(1))`` is ``One``, a special type
        of sympy atom, while ``type(S(2))`` is type ``Integer`` and will find all
        integers in an expression:

        >>> from sympy import S
        >>> (1 + x + 2*sin(y + I*pi)).atoms(S(1))
        {1}

        >>> (1 + x + 2*sin(y + I*pi)).atoms(S(2))
        {1, 2}

        Finally, arguments to atoms() can select more than atomic atoms: any
        sympy type (loaded in core/__init__.py) can be listed as an argument
        and those types of "atoms" as found in scanning the arguments of the
        expression recursively:

        >>> from sympy import Function, Mul
        >>> from sympy.core.function import AppliedUndef
        >>> f = Function('f')
        >>> (1 + f(x) + 2*sin(y + I*pi)).atoms(Function)
        {f(x), sin(y + I*pi)}
        >>> (1 + f(x) + 2*sin(y + I*pi)).atoms(AppliedUndef)
        {f(x)}

        >>> (1 + x + 2*sin(y + I*pi)).atoms(Mul)
        {I*pi, 2*sin(y + I*pi)}

        """
        if types:
            types = tuple(
                [t if isinstance(t, type) else type(t) for t in types])
        else:
            types = (Atom,)
        result = set()
        for expr in preorder_traversal(self):
            if isinstance(expr, types):
                result.add(expr)
        return result
```
### 4 - sympy/core/expr.py:

Start line: 2377, End line: 2399

```python
@sympify_method_args
class Expr(Basic, EvalfMixin):

    @property
    def expr_free_symbols(self):
        """
        Like ``free_symbols``, but returns the free symbols only if they are contained in an expression node.

        Examples
        ========

        >>> from sympy.abc import x, y
        >>> (x + y).expr_free_symbols
        {x, y}

        If the expression is contained in a non-expression object, don't return
        the free symbols. Compare:

        >>> from sympy import Tuple
        >>> t = Tuple(x + y)
        >>> t.expr_free_symbols
        set()
        >>> t.free_symbols
        {x, y}
        """
        return {j for i in self.args for j in i.expr_free_symbols}
```
### 5 - sympy/parsing/latex/_parse_latex_antlr.py:

Start line: 201, End line: 213

```python
def do_subs(expr, at):
    if at.expr():
        at_expr = convert_expr(at.expr())
        syms = at_expr.atoms(sympy.Symbol)
        if len(syms) == 0:
            return expr
        elif len(syms) > 0:
            sym = next(iter(syms))
            return expr.subs(sym, at_expr)
    elif at.equality():
        lh = convert_expr(at.equality().expr(0))
        rh = convert_expr(at.equality().expr(1))
        return expr.subs(lh, rh)
```
### 6 - sympy/utilities/lambdify.py:

Start line: 1031, End line: 1069

```python
class _EvaluatorPrinter(object):

    def _preprocess(self, args, expr):
        """Preprocess args, expr to replace arguments that do not map
        to valid Python identifiers.

        Returns string form of args, and updated expr.
        """
        from sympy import Dummy, Function, flatten, Derivative, ordered, Basic
        from sympy.matrices import DeferredVector
        from sympy.core.symbol import _uniquely_named_symbol
        from sympy.core.expr import Expr

        # Args of type Dummy can cause name collisions with args
        # of type Symbol.  Force dummify of everything in this
        # situation.
        dummify = self._dummify or any(
            isinstance(arg, Dummy) for arg in flatten(args))

        argstrs = [None]*len(args)
        for arg, i in reversed(list(ordered(zip(args, range(len(args)))))):
            if iterable(arg):
                s, expr = self._preprocess(arg, expr)
            elif isinstance(arg, DeferredVector):
                s = str(arg)
            elif isinstance(arg, Basic) and arg.is_symbol:
                s = self._argrepr(arg)
                if dummify or not self._is_safe_ident(s):
                    dummy = Dummy()
                    if isinstance(expr, Expr):
                        dummy = _uniquely_named_symbol(dummy.name, expr)
                    s = self._argrepr(dummy)
                    expr = self._subexpr(expr, {arg: dummy})
            elif dummify or isinstance(arg, (Function, Derivative)):
                dummy = Dummy()
                s = self._argrepr(dummy)
                expr = self._subexpr(expr, {arg: dummy})
            else:
                s = str(arg)
            argstrs[i] = s
        return argstrs, expr
```
### 7 - sympy/simplify/epathtools.py:

Start line: 180, End line: 224

```python
class EPath(object):

    def apply(self, expr, func, args=None, kwargs=None):
        def _apply(path, expr, func):
            if not path:
                return func(expr)
            else:
                selector, path = path[0], path[1:]
                attrs, types, span = selector

                if isinstance(expr, Basic):
                    if not expr.is_Atom:
                        args, basic = self._get_ordered_args(expr), True
                    else:
                        return expr
                elif hasattr(expr, '__iter__'):
                    args, basic = expr, False
                else:
                    return expr

                args = list(args)

                if span is not None:
                    if type(span) == slice:
                        indices = range(*span.indices(len(args)))
                    else:
                        indices = [span]
                else:
                    indices = range(len(args))

                for i in indices:
                    try:
                        arg = args[i]
                    except IndexError:
                        continue

                    if self._has(arg, attrs, types):
                        args[i] = _apply(path, arg, func)

                if basic:
                    return expr.func(*args)
                else:
                    return expr.__class__(args)

        _args, _kwargs = args or (), kwargs or {}
        _func = lambda expr: func(expr, *_args, **_kwargs)

        return _apply(self._epath, expr, _func)
```
### 8 - sympy/printing/dot.py:

Start line: 278, End line: 295

```python
def dotprint(expr,
    styles=default_styles, atom=lambda x: not isinstance(x, Basic),
    maxdepth=None, repeat=True, labelfunc=str, **kwargs):
    # Pow will have the tuple (1, 0), meaning it is expr.args[1].args[0].
    graphstyle = _graphstyle.copy()
    graphstyle.update(kwargs)

    nodes = []
    edges = []
    def traverse(e, depth, pos=()):
        nodes.append(dotnode(e, styles, labelfunc=labelfunc, pos=pos, repeat=repeat))
        if maxdepth and depth >= maxdepth:
            return
        edges.extend(dotedges(e, atom=atom, pos=pos, repeat=repeat))
        [traverse(arg, depth+1, pos + (i,)) for i, arg in enumerate(e.args) if not atom(arg)]
    traverse(expr, 0)

    return template%{'graphstyle': attrprint(graphstyle, delimiter='\n'),
                     'nodes': '\n'.join(nodes),
                     'edges': '\n'.join(edges)}
```
### 9 - sympy/parsing/latex/_antlr/latexparser.py:

Start line: 2687, End line: 2711

```python
class LaTeXParser ( Parser ):

    class SubexprContext(ParserRuleContext):

        def __init__(self, parser, parent=None, invokingState=-1):
            super(LaTeXParser.SubexprContext, self).__init__(parent, invokingState)
            self.parser = parser

        def UNDERSCORE(self):
            return self.getToken(LaTeXParser.UNDERSCORE, 0)

        def atom(self):
            return self.getTypedRuleContext(LaTeXParser.AtomContext,0)


        def L_BRACE(self):
            return self.getToken(LaTeXParser.L_BRACE, 0)

        def expr(self):
            return self.getTypedRuleContext(LaTeXParser.ExprContext,0)


        def R_BRACE(self):
            return self.getToken(LaTeXParser.R_BRACE, 0)

        def getRuleIndex(self):
            return LaTeXParser.RULE_subexpr
```
### 10 - sympy/parsing/latex/_parse_latex_antlr.py:

Start line: 281, End line: 324

```python
def convert_atom(atom):
    if atom.LETTER():
        subscriptName = ''
        if atom.subexpr():
            subscript = None
            if atom.subexpr().expr():  # subscript is expr
                subscript = convert_expr(atom.subexpr().expr())
            else:  # subscript is atom
                subscript = convert_atom(atom.subexpr().atom())
            subscriptName = '_{' + StrPrinter().doprint(subscript) + '}'
        return sympy.Symbol(atom.LETTER().getText() + subscriptName)
    elif atom.SYMBOL():
        s = atom.SYMBOL().getText()[1:]
        if s == "infty":
            return sympy.oo
        else:
            if atom.subexpr():
                subscript = None
                if atom.subexpr().expr():  # subscript is expr
                    subscript = convert_expr(atom.subexpr().expr())
                else:  # subscript is atom
                    subscript = convert_atom(atom.subexpr().atom())
                subscriptName = StrPrinter().doprint(subscript)
                s += '_{' + subscriptName + '}'
            return sympy.Symbol(s)
    elif atom.NUMBER():
        s = atom.NUMBER().getText().replace(",", "")
        return sympy.Number(s)
    elif atom.DIFFERENTIAL():
        var = get_differential_var(atom.DIFFERENTIAL())
        return sympy.Symbol('d' + var.name)
    elif atom.mathit():
        text = rule2text(atom.mathit().mathit_text())
        return sympy.Symbol(text)


def rule2text(ctx):
    stream = ctx.start.getInputStream()
    # starting index of starting token
    startIdx = ctx.start.start
    # stopping index of stopping token
    stopIdx = ctx.stop.stop

    return stream.getText(startIdx, stopIdx)
```
### 17 - sympy/core/basic.py:

Start line: 514, End line: 531

```python
class Basic(metaclass=ManagedProperties):

    @property
    def free_symbols(self):
        """Return from the atoms of self those which are free symbols.

        For most expressions, all symbols are free symbols. For some classes
        this is not true. e.g. Integrals use Symbols for the dummy variables
        which are bound variables, so Integral has a method to return all
        symbols except those. Derivative keeps track of symbols with respect
        to which it will perform a derivative; those are
        bound variables, too, so it has its own free_symbols method.

        Any other method that uses bound variables should implement a
        free_symbols method."""
        return set().union(*[a.free_symbols for a in self.args])

    @property
    def expr_free_symbols(self):
        return set([])
```
