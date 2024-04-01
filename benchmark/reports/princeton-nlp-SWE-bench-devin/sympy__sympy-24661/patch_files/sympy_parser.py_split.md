

## EpicSplitter

29 chunks

#### Split 1
200 tokens, line: 1 - 28

```python
"""Transform a string with Python-like source code into SymPy expression. """

from tokenize import (generate_tokens, untokenize, TokenError,
    NUMBER, STRING, NAME, OP, ENDMARKER, ERRORTOKEN, NEWLINE)

from keyword import iskeyword

import ast
import unicodedata
from io import StringIO
import builtins
import types
from typing import Tuple as tTuple, Dict as tDict, Any, Callable, \
    List, Optional, Union as tUnion

from sympy.assumptions.ask import AssumptionKeys
from sympy.core.basic import Basic
from sympy.core import Symbol
from sympy.core.function import Function
from sympy.utilities.misc import func_name
from sympy.functions.elementary.miscellaneous import Max, Min


null = ''

TOKEN = tTuple[int, str]
DICT = tDict[str, Any]
TRANS = Callable[[List[TOKEN], DICT, DICT], List[TOKEN]]
```



#### Split 2
222 tokens, line: 30 - 56

```python
def _token_splittable(token_name: str) -> bool:
    """
    Predicate for whether a token name can be split into multiple tokens.

    A token is splittable if it does not contain an underscore character and
    it is not the name of a Greek letter. This is used to implicitly convert
    expressions like 'xyz' into 'x*y*z'.
    """
    if '_' in token_name:
        return False
    try:
        return not unicodedata.lookup('GREEK SMALL LETTER ' + token_name)
    except KeyError:
        return len(token_name) > 1


def _token_callable(token: TOKEN, local_dict: DICT, global_dict: DICT, nextToken=None):
    """
    Predicate for whether a token name represents a callable function.

    Essentially wraps ``callable``, but looks up the token name in the
    locals and globals.
    """
    func = local_dict.get(token[1])
    if not func:
        func = global_dict.get(token[1])
    return callable(func) and not isinstance(func, Symbol)
```



#### Split 3
201 tokens, line: 59 - 84

```python
def _add_factorial_tokens(name: str, result: List[TOKEN]) -> List[TOKEN]:
    if result == [] or result[-1][1] == '(':
        raise TokenError()

    beginning = [(NAME, name), (OP, '(')]
    end = [(OP, ')')]

    diff = 0
    length = len(result)

    for index, token in enumerate(result[::-1]):
        toknum, tokval = token
        i = length - index - 1

        if tokval == ')':
            diff += 1
        elif tokval == '(':
            diff -= 1

        if diff == 0:
            if i - 1 >= 0 and result[i - 1][0] == NAME:
                return result[:i - 1] + beginning + result[i - 1:] + end
            else:
                return result[:i] + beginning + result[i:] + end

    return result
```



#### Split 4
267 tokens, line: 87 - 125

```python
class ParenthesisGroup(List[TOKEN]):
    """List of tokens representing an expression in parentheses."""
    pass


class AppliedFunction:
    """
    A group of tokens representing a function and its arguments.

    `exponent` is for handling the shorthand sin^2, ln^2, etc.
    """
    def __init__(self, function: TOKEN, args: ParenthesisGroup, exponent=None):
        if exponent is None:
            exponent = []
        self.function = function
        self.args = args
        self.exponent = exponent
        self.items = ['function', 'args', 'exponent']

    def expand(self) -> List[TOKEN]:
        """Return a list of tokens representing the function"""
        return [self.function, *self.args]

    def __getitem__(self, index):
        return getattr(self, self.items[index])

    def __repr__(self):
        return "AppliedFunction(%s, %s, %s)" % (self.function, self.args,
                                                self.exponent)


def _flatten(result: List[tUnion[TOKEN, AppliedFunction]]):
    result2: List[TOKEN] = []
    for tok in result:
        if isinstance(tok, AppliedFunction):
            result2.extend(tok.expand())
        else:
            result2.append(tok)
    return result2
```



#### Split 5
318 tokens, line: 128 - 169

```python
def _group_parentheses(recursor: TRANS):
    def _inner(tokens: List[TOKEN], local_dict: DICT, global_dict: DICT):
        """Group tokens between parentheses with ParenthesisGroup.

        Also processes those tokens recursively.

        """
        result: List[tUnion[TOKEN, ParenthesisGroup]] = []
        stacks: List[ParenthesisGroup] = []
        stacklevel = 0
        for token in tokens:
            if token[0] == OP:
                if token[1] == '(':
                    stacks.append(ParenthesisGroup([]))
                    stacklevel += 1
                elif token[1] == ')':
                    stacks[-1].append(token)
                    stack = stacks.pop()

                    if len(stacks) > 0:
                        # We don't recurse here since the upper-level stack
                        # would reprocess these tokens
                        stacks[-1].extend(stack)
                    else:
                        # Recurse here to handle nested parentheses
                        # Strip off the outer parentheses to avoid an infinite loop
                        inner = stack[1:-1]
                        inner = recursor(inner,
                                         local_dict,
                                         global_dict)
                        parenGroup = [stack[0]] + inner + [stack[-1]]
                        result.append(ParenthesisGroup(parenGroup))
                    stacklevel -= 1
                    continue
            if stacklevel:
                stacks[-1].append(token)
            else:
                result.append(token)
        if stacklevel:
            raise TokenError("Mismatched parentheses")
        return result
    return _inner
```



#### Split 6
178 tokens, line: 172 - 194

```python
def _apply_functions(tokens: List[tUnion[TOKEN, ParenthesisGroup]], local_dict: DICT, global_dict: DICT):
    """Convert a NAME token + ParenthesisGroup into an AppliedFunction.

    Note that ParenthesisGroups, if not applied to any function, are
    converted back into lists of tokens.

    """
    result: List[tUnion[TOKEN, AppliedFunction]] = []
    symbol = None
    for tok in tokens:
        if isinstance(tok, ParenthesisGroup):
            if symbol and _token_callable(symbol, local_dict, global_dict):
                result[-1] = AppliedFunction(symbol, tok)
                symbol = None
            else:
                result.extend(tok)
        elif tok[0] == NAME:
            symbol = tok
            result.append(tok)
        else:
            symbol = None
            result.append(tok)
    return result
```



#### Split 7
567 tokens, line: 197 - 259

```python
def _implicit_multiplication(tokens: List[tUnion[TOKEN, AppliedFunction]], local_dict: DICT, global_dict: DICT):
    """Implicitly adds '*' tokens.

    Cases:

    - Two AppliedFunctions next to each other ("sin(x)cos(x)")

    - AppliedFunction next to an open parenthesis ("sin x (cos x + 1)")

    - A close parenthesis next to an AppliedFunction ("(x+2)sin x")\

    - A close parenthesis next to an open parenthesis ("(x+2)(x+3)")

    - AppliedFunction next to an implicitly applied function ("sin(x)cos x")

    """
    result: List[tUnion[TOKEN, AppliedFunction]] = []
    skip = False
    for tok, nextTok in zip(tokens, tokens[1:]):
        result.append(tok)
        if skip:
            skip = False
            continue
        if tok[0] == OP and tok[1] == '.' and nextTok[0] == NAME:
            # Dotted name. Do not do implicit multiplication
            skip = True
            continue
        if isinstance(tok, AppliedFunction):
            if isinstance(nextTok, AppliedFunction):
                result.append((OP, '*'))
            elif nextTok == (OP, '('):
                # Applied function followed by an open parenthesis
                if tok.function[1] == "Function":
                    tok.function = (tok.function[0], 'Symbol')
                result.append((OP, '*'))
            elif nextTok[0] == NAME:
                # Applied function followed by implicitly applied function
                result.append((OP, '*'))
        else:
            if tok == (OP, ')'):
                if isinstance(nextTok, AppliedFunction):
                    # Close parenthesis followed by an applied function
                    result.append((OP, '*'))
                elif nextTok[0] == NAME:
                    # Close parenthesis followed by an implicitly applied function
                    result.append((OP, '*'))
                elif nextTok == (OP, '('):
                    # Close parenthesis followed by an open parenthesis
                    result.append((OP, '*'))
            elif tok[0] == NAME and not _token_callable(tok, local_dict, global_dict):
                if isinstance(nextTok, AppliedFunction) or \
                    (nextTok[0] == NAME and _token_callable(nextTok, local_dict, global_dict)):
                    # Constant followed by (implicitly applied) function
                    result.append((OP, '*'))
                elif nextTok == (OP, '('):
                    # Constant followed by parenthesis
                    result.append((OP, '*'))
                elif nextTok[0] == NAME:
                    # Constant followed by constant
                    result.append((OP, '*'))
    if tokens:
        result.append(tokens[-1])
    return result
```



#### Split 8
531 tokens, line: 262 - 310

```python
def _implicit_application(tokens: List[tUnion[TOKEN, AppliedFunction]], local_dict: DICT, global_dict: DICT):
    """Adds parentheses as needed after functions."""
    result: List[tUnion[TOKEN, AppliedFunction]] = []
    appendParen = 0  # number of closing parentheses to add
    skip = 0  # number of tokens to delay before adding a ')' (to
              # capture **, ^, etc.)
    exponentSkip = False  # skipping tokens before inserting parentheses to
                          # work with function exponentiation
    for tok, nextTok in zip(tokens, tokens[1:]):
        result.append(tok)
        if (tok[0] == NAME and nextTok[0] not in [OP, ENDMARKER, NEWLINE]):
            if _token_callable(tok, local_dict, global_dict, nextTok):  # type: ignore
                result.append((OP, '('))
                appendParen += 1
        # name followed by exponent - function exponentiation
        elif (tok[0] == NAME and nextTok[0] == OP and nextTok[1] == '**'):
            if _token_callable(tok, local_dict, global_dict):  # type: ignore
                exponentSkip = True
        elif exponentSkip:
            # if the last token added was an applied function (i.e. the
            # power of the function exponent) OR a multiplication (as
            # implicit multiplication would have added an extraneous
            # multiplication)
            if (isinstance(tok, AppliedFunction)
                or (tok[0] == OP and tok[1] == '*')):
                # don't add anything if the next token is a multiplication
                # or if there's already a parenthesis (if parenthesis, still
                # stop skipping tokens)
                if not (nextTok[0] == OP and nextTok[1] == '*'):
                    if not(nextTok[0] == OP and nextTok[1] == '('):
                        result.append((OP, '('))
                        appendParen += 1
                    exponentSkip = False
        elif appendParen:
            if nextTok[0] == OP and nextTok[1] in ('^', '**', '*'):
                skip = 1
                continue
            if skip:
                skip -= 1
                continue
            result.append((OP, ')'))
            appendParen -= 1

    if tokens:
        result.append(tokens[-1])

    if appendParen:
        result.extend([(OP, ')')] * appendParen)
    return result
```



#### Split 9
451 tokens, line: 313 - 362

```python
def function_exponentiation(tokens: List[TOKEN], local_dict: DICT, global_dict: DICT):
    """Allows functions to be exponentiated, e.g. ``cos**2(x)``.

    Examples
    ========

    >>> from sympy.parsing.sympy_parser import (parse_expr,
    ... standard_transformations, function_exponentiation)
    >>> transformations = standard_transformations + (function_exponentiation,)
    >>> parse_expr('sin**4(x)', transformations=transformations)
    sin(x)**4
    """
    result: List[TOKEN] = []
    exponent: List[TOKEN] = []
    consuming_exponent = False
    level = 0
    for tok, nextTok in zip(tokens, tokens[1:]):
        if tok[0] == NAME and nextTok[0] == OP and nextTok[1] == '**':
            if _token_callable(tok, local_dict, global_dict):
                consuming_exponent = True
        elif consuming_exponent:
            if tok[0] == NAME and tok[1] == 'Function':
                tok = (NAME, 'Symbol')
            exponent.append(tok)

            # only want to stop after hitting )
            if tok[0] == nextTok[0] == OP and tok[1] == ')' and nextTok[1] == '(':
                consuming_exponent = False
            # if implicit multiplication was used, we may have )*( instead
            if tok[0] == nextTok[0] == OP and tok[1] == '*' and nextTok[1] == '(':
                consuming_exponent = False
                del exponent[-1]
            continue
        elif exponent and not consuming_exponent:
            if tok[0] == OP:
                if tok[1] == '(':
                    level += 1
                elif tok[1] == ')':
                    level -= 1
            if level == 0:
                result.append(tok)
                result.extend(exponent)
                exponent = []
                continue
        result.append(tok)
    if tokens:
        result.append(tokens[-1])
    if exponent:
        result.extend(exponent)
    return result
```



#### Split 10
200 tokens, line: 365 - 386

```python
def split_symbols_custom(predicate: Callable[[str], bool]):
    """Creates a transformation that splits symbol names.

    ``predicate`` should return True if the symbol name is to be split.

    For instance, to retain the default behavior but avoid splitting certain
    symbol names, a predicate like this would work:


    >>> from sympy.parsing.sympy_parser import (parse_expr, _token_splittable,
    ... standard_transformations, implicit_multiplication,
    ... split_symbols_custom)
    >>> def can_split(symbol):
    ...     if symbol not in ('list', 'of', 'unsplittable', 'names'):
    ...             return _token_splittable(symbol)
    ...     return False
    ...
    >>> transformation = split_symbols_custom(can_split)
    >>> parse_expr('unsplittable', transformations=standard_transformations +
    ... (transformation, implicit_multiplication))
    unsplittable
    """
    # ... other code
```



#### Split 11
410 tokens, line: 387 - 443

```python
def split_symbols_custom(predicate: Callable[[str], bool]):
    def _split_symbols(tokens: List[TOKEN], local_dict: DICT, global_dict: DICT):
        result: List[TOKEN] = []
        split = False
        split_previous=False

        for tok in tokens:
            if split_previous:
                # throw out closing parenthesis of Symbol that was split
                split_previous=False
                continue
            split_previous=False

            if tok[0] == NAME and tok[1] in ['Symbol', 'Function']:
                split = True

            elif split and tok[0] == NAME:
                symbol = tok[1][1:-1]

                if predicate(symbol):
                    tok_type = result[-2][1]  # Symbol or Function
                    del result[-2:]  # Get rid of the call to Symbol

                    i = 0
                    while i < len(symbol):
                        char = symbol[i]
                        if char in local_dict or char in global_dict:
                            result.append((NAME, "%s" % char))
                        elif char.isdigit():
                            chars = [char]
                            for i in range(i + 1, len(symbol)):
                                if not symbol[i].isdigit():
                                  i -= 1
                                  break
                                chars.append(symbol[i])
                            char = ''.join(chars)
                            result.extend([(NAME, 'Number'), (OP, '('),
                                           (NAME, "'%s'" % char), (OP, ')')])
                        else:
                            use = tok_type if i == len(symbol) else 'Symbol'
                            result.extend([(NAME, use), (OP, '('),
                                           (NAME, "'%s'" % char), (OP, ')')])
                        i += 1

                    # Set split_previous=True so will skip
                    # the closing parenthesis of the original Symbol
                    split = False
                    split_previous = True
                    continue

                else:
                    split = False

            result.append(tok)

        return result

    return _split_symbols
```



#### Split 12
317 tokens, line: 446 - 476

```python
#: Splits symbol names for implicit multiplication.
#:
#: Intended to let expressions like ``xyz`` be parsed as ``x*y*z``. Does not
#: split Greek character names, so ``theta`` will *not* become
#: ``t*h*e*t*a``. Generally this should be used with
#: ``implicit_multiplication``.
split_symbols = split_symbols_custom(_token_splittable)


def implicit_multiplication(tokens: List[TOKEN], local_dict: DICT,
                            global_dict: DICT) -> List[TOKEN]:
    """Makes the multiplication operator optional in most cases.

    Use this before :func:`implicit_application`, otherwise expressions like
    ``sin 2x`` will be parsed as ``x * sin(2)`` rather than ``sin(2*x)``.

    Examples
    ========

    >>> from sympy.parsing.sympy_parser import (parse_expr,
    ... standard_transformations, implicit_multiplication)
    >>> transformations = standard_transformations + (implicit_multiplication,)
    >>> parse_expr('3 x y', transformations=transformations)
    3*x*y
    """
    # These are interdependent steps, so we don't expose them separately
    res1 = _group_parentheses(implicit_multiplication)(tokens, local_dict, global_dict)
    res2 = _apply_functions(res1, local_dict, global_dict)
    res3 = _implicit_multiplication(res2, local_dict, global_dict)
    result = _flatten(res3)
    return result
```



#### Split 13
221 tokens, line: 479 - 500

```python
def implicit_application(tokens: List[TOKEN], local_dict: DICT,
                         global_dict: DICT) -> List[TOKEN]:
    """Makes parentheses optional in some cases for function calls.

    Use this after :func:`implicit_multiplication`, otherwise expressions
    like ``sin 2x`` will be parsed as ``x * sin(2)`` rather than
    ``sin(2*x)``.

    Examples
    ========

    >>> from sympy.parsing.sympy_parser import (parse_expr,
    ... standard_transformations, implicit_application)
    >>> transformations = standard_transformations + (implicit_application,)
    >>> parse_expr('cot z + csc z', transformations=transformations)
    cot(z) + csc(z)
    """
    res1 = _group_parentheses(implicit_application)(tokens, local_dict, global_dict)
    res2 = _apply_functions(res1, local_dict, global_dict)
    res3 = _implicit_application(res2, local_dict, global_dict)
    result = _flatten(res3)
    return result
```



#### Split 14
213 tokens, line: 503 - 531

```python
def implicit_multiplication_application(result: List[TOKEN], local_dict: DICT,
                                        global_dict: DICT) -> List[TOKEN]:
    """Allows a slightly relaxed syntax.

    - Parentheses for single-argument method calls are optional.

    - Multiplication is implicit.

    - Symbol names can be split (i.e. spaces are not needed between
      symbols).

    - Functions can be exponentiated.

    Examples
    ========

    >>> from sympy.parsing.sympy_parser import (parse_expr,
    ... standard_transformations, implicit_multiplication_application)
    >>> parse_expr("10sin**2 x**2 + 3xyz + tan theta",
    ... transformations=(standard_transformations +
    ... (implicit_multiplication_application,)))
    3*x*y*z + 10*sin(x**2)**2 + tan(theta)

    """
    for step in (split_symbols, implicit_multiplication,
                 implicit_application, function_exponentiation):
        result = step(result, local_dict, global_dict)

    return result
```



#### Split 15
417 tokens, line: 534 - 582

```python
def auto_symbol(tokens: List[TOKEN], local_dict: DICT, global_dict: DICT):
    """Inserts calls to ``Symbol``/``Function`` for undefined variables."""
    result: List[TOKEN] = []
    prevTok = (-1, '')

    tokens.append((-1, ''))  # so zip traverses all tokens
    for tok, nextTok in zip(tokens, tokens[1:]):
        tokNum, tokVal = tok
        nextTokNum, nextTokVal = nextTok
        if tokNum == NAME:
            name = tokVal

            if (name in ['True', 'False', 'None']
                    or iskeyword(name)
                    # Don't convert attribute access
                    or (prevTok[0] == OP and prevTok[1] == '.')
                    # Don't convert keyword arguments
                    or (prevTok[0] == OP and prevTok[1] in ('(', ',')
                        and nextTokNum == OP and nextTokVal == '=')
                    # the name has already been defined
                    or name in local_dict and local_dict[name] is not null):
                result.append((NAME, name))
                continue
            elif name in local_dict:
                local_dict.setdefault(null, set()).add(name)
                if nextTokVal == '(':
                    local_dict[name] = Function(name)
                else:
                    local_dict[name] = Symbol(name)
                result.append((NAME, name))
                continue
            elif name in global_dict:
                obj = global_dict[name]
                if isinstance(obj, (AssumptionKeys, Basic, type)) or callable(obj):
                    result.append((NAME, name))
                    continue

            result.extend([
                (NAME, 'Symbol' if nextTokVal != '(' else 'Function'),
                (OP, '('),
                (NAME, repr(str(name))),
                (OP, ')'),
            ])
        else:
            result.append((tokNum, tokVal))

        prevTok = (tokNum, tokVal)

    return result
```



#### Split 16
317 tokens, line: 585 - 622

```python
def lambda_notation(tokens: List[TOKEN], local_dict: DICT, global_dict: DICT):
    """Substitutes "lambda" with its SymPy equivalent Lambda().
    However, the conversion does not take place if only "lambda"
    is passed because that is a syntax error.

    """
    result: List[TOKEN] = []
    flag = False
    toknum, tokval = tokens[0]
    tokLen = len(tokens)

    if toknum == NAME and tokval == 'lambda':
        if tokLen == 2 or tokLen == 3 and tokens[1][0] == NEWLINE:
            # In Python 3.6.7+, inputs without a newline get NEWLINE added to
            # the tokens
            result.extend(tokens)
        elif tokLen > 2:
            result.extend([
                (NAME, 'Lambda'),
                (OP, '('),
                (OP, '('),
                (OP, ')'),
                (OP, ')'),
            ])
            for tokNum, tokVal in tokens[1:]:
                if tokNum == OP and tokVal == ':':
                    tokVal = ','
                    flag = True
                if not flag and tokNum == OP and tokVal in ('*', '**'):
                    raise TokenError("Starred arguments in lambda not supported")
                if flag:
                    result.insert(-1, (tokNum, tokVal))
                else:
                    result.insert(-2, (tokNum, tokVal))
    else:
        result.extend(tokens)

    return result
```



#### Split 17
300 tokens, line: 625 - 661

```python
def factorial_notation(tokens: List[TOKEN], local_dict: DICT, global_dict: DICT):
    """Allows standard notation for factorial."""
    result: List[TOKEN] = []
    nfactorial = 0
    for toknum, tokval in tokens:
        if toknum == ERRORTOKEN:
            op = tokval
            if op == '!':
                nfactorial += 1
            else:
                nfactorial = 0
                result.append((OP, op))
        else:
            if nfactorial == 1:
                result = _add_factorial_tokens('factorial', result)
            elif nfactorial == 2:
                result = _add_factorial_tokens('factorial2', result)
            elif nfactorial > 2:
                raise TokenError
            nfactorial = 0
            result.append((toknum, tokval))
    return result


def convert_xor(tokens: List[TOKEN], local_dict: DICT, global_dict: DICT):
    """Treats XOR, ``^``, as exponentiation, ``**``."""
    result: List[TOKEN] = []
    for toknum, tokval in tokens:
        if toknum == OP:
            if tokval == '^':
                result.append((OP, '**'))
            else:
                result.append((toknum, tokval))
        else:
            result.append((toknum, tokval))

    return result
```



#### Split 18
723 tokens, line: 664 - 752

```python
def repeated_decimals(tokens: List[TOKEN], local_dict: DICT, global_dict: DICT):
    """
    Allows 0.2[1] notation to represent the repeated decimal 0.2111... (19/90)

    Run this before auto_number.

    """
    result: List[TOKEN] = []

    def is_digit(s):
        return all(i in '0123456789_' for i in s)

    # num will running match any DECIMAL [ INTEGER ]
    num: List[TOKEN] = []
    for toknum, tokval in tokens:
        if toknum == NUMBER:
            if (not num and '.' in tokval and 'e' not in tokval.lower() and
                'j' not in tokval.lower()):
                num.append((toknum, tokval))
            elif is_digit(tokval)and  len(num) == 2:
                num.append((toknum, tokval))
            elif is_digit(tokval) and len(num) == 3 and is_digit(num[-1][1]):
                # Python 2 tokenizes 00123 as '00', '123'
                # Python 3 tokenizes 01289 as '012', '89'
                num.append((toknum, tokval))
            else:
                num = []
        elif toknum == OP:
            if tokval == '[' and len(num) == 1:
                num.append((OP, tokval))
            elif tokval == ']' and len(num) >= 3:
                num.append((OP, tokval))
            elif tokval == '.' and not num:
                # handle .[1]
                num.append((NUMBER, '0.'))
            else:
                num = []
        else:
            num = []

        result.append((toknum, tokval))

        if num and num[-1][1] == ']':
            # pre.post[repetend] = a + b/c + d/e where a = pre, b/c = post,
            # and d/e = repetend
            result = result[:-len(num)]
            pre, post = num[0][1].split('.')
            repetend = num[2][1]
            if len(num) == 5:
                repetend += num[3][1]

            pre = pre.replace('_', '')
            post = post.replace('_', '')
            repetend = repetend.replace('_', '')

            zeros = '0'*len(post)
            post, repetends = [w.lstrip('0') for w in [post, repetend]]
                                        # or else interpreted as octal

            a = pre or '0'
            b, c = post or '0', '1' + zeros
            d, e = repetends, ('9'*len(repetend)) + zeros

            seq = [
                (OP, '('),
                    (NAME, 'Integer'),
                    (OP, '('),
                        (NUMBER, a),
                    (OP, ')'),
                    (OP, '+'),
                    (NAME, 'Rational'),
                    (OP, '('),
                        (NUMBER, b),
                        (OP, ','),
                        (NUMBER, c),
                    (OP, ')'),
                    (OP, '+'),
                    (NAME, 'Rational'),
                    (OP, '('),
                        (NUMBER, d),
                        (OP, ','),
                        (NUMBER, e),
                    (OP, ')'),
                (OP, ')'),
            ]
            result.extend(seq)
            num = []

    return result
```



#### Split 19
245 tokens, line: 755 - 786

```python
def auto_number(tokens: List[TOKEN], local_dict: DICT, global_dict: DICT):
    """
    Converts numeric literals to use SymPy equivalents.

    Complex numbers use ``I``, integer literals use ``Integer``, and float
    literals use ``Float``.

    """
    result: List[TOKEN] = []

    for toknum, tokval in tokens:
        if toknum == NUMBER:
            number = tokval
            postfix = []

            if number.endswith('j') or number.endswith('J'):
                number = number[:-1]
                postfix = [(OP, '*'), (NAME, 'I')]

            if '.' in number or (('e' in number or 'E' in number) and
                    not (number.startswith('0x') or number.startswith('0X'))):
                seq = [(NAME, 'Float'), (OP, '('),
                    (NUMBER, repr(str(number))), (OP, ')')]
            else:
                seq = [(NAME, 'Integer'), (OP, '('), (
                    NUMBER, number), (OP, ')')]

            result.extend(seq + postfix)
        else:
            result.append((toknum, tokval))

    return result
```



#### Split 20
146 tokens, line: 789 - 805

```python
def rationalize(tokens: List[TOKEN], local_dict: DICT, global_dict: DICT):
    """Converts floats into ``Rational``. Run AFTER ``auto_number``."""
    result: List[TOKEN] = []
    passed_float = False
    for toknum, tokval in tokens:
        if toknum == NAME:
            if tokval == 'Float':
                passed_float = True
                tokval = 'Rational'
            result.append((toknum, tokval))
        elif passed_float == True and toknum == NUMBER:
            passed_float = False
            result.append((STRING, tokval))
        else:
            result.append((toknum, tokval))

    return result
```



#### Split 21
224 tokens, line: 808 - 834

```python
def _transform_equals_sign(tokens: List[TOKEN], local_dict: DICT, global_dict: DICT):
    """Transforms the equals sign ``=`` to instances of Eq.

    This is a helper function for ``convert_equals_signs``.
    Works with expressions containing one equals sign and no
    nesting. Expressions like ``(1=2)=False`` will not work with this
    and should be used with ``convert_equals_signs``.

    Examples: 1=2     to Eq(1,2)
              1*2=x   to Eq(1*2, x)

    This does not deal with function arguments yet.

    """
    result: List[TOKEN] = []
    if (OP, "=") in tokens:
        result.append((NAME, "Eq"))
        result.append((OP, "("))
        for token in tokens:
            if token == (OP, "="):
                result.append((OP, ","))
                continue
            result.append(token)
        result.append((OP, ")"))
    else:
        result = tokens
    return result
```



#### Split 22
308 tokens, line: 837 - 870

```python
def convert_equals_signs(tokens: List[TOKEN], local_dict: DICT,
                         global_dict: DICT) -> List[TOKEN]:
    """ Transforms all the equals signs ``=`` to instances of Eq.

    Parses the equals signs in the expression and replaces them with
    appropriate Eq instances. Also works with nested equals signs.

    Does not yet play well with function arguments.
    For example, the expression ``(x=y)`` is ambiguous and can be interpreted
    as x being an argument to a function and ``convert_equals_signs`` will not
    work for this.

    See also
    ========
    convert_equality_operators

    Examples
    ========

    >>> from sympy.parsing.sympy_parser import (parse_expr,
    ... standard_transformations, convert_equals_signs)
    >>> parse_expr("1*2=x", transformations=(
    ... standard_transformations + (convert_equals_signs,)))
    Eq(2, x)
    >>> parse_expr("(1*2=x)=False", transformations=(
    ... standard_transformations + (convert_equals_signs,)))
    Eq(Eq(2, x), False)

    """
    res1 = _group_parentheses(convert_equals_signs)(tokens, local_dict, global_dict)
    res2 = _apply_functions(res1, local_dict, global_dict)
    res3 = _transform_equals_sign(res2, local_dict, global_dict)
    result = _flatten(res3)
    return result
```



#### Split 23
283 tokens, line: 873 - 908

```python
#: Standard transformations for :func:`parse_expr`.
#: Inserts calls to :class:`~.Symbol`, :class:`~.Integer`, and other SymPy
#: datatypes and allows the use of standard factorial notation (e.g. ``x!``).
standard_transformations: tTuple[TRANS, ...] \
    = (lambda_notation, auto_symbol, repeated_decimals, auto_number,
       factorial_notation)


def stringify_expr(s: str, local_dict: DICT, global_dict: DICT,
        transformations: tTuple[TRANS, ...]) -> str:
    """
    Converts the string ``s`` to Python code, in ``local_dict``

    Generally, ``parse_expr`` should be used.
    """

    tokens = []
    input_code = StringIO(s.strip())
    for toknum, tokval, _, _, _ in generate_tokens(input_code.readline):
        tokens.append((toknum, tokval))

    for transform in transformations:
        tokens = transform(tokens, local_dict, global_dict)

    return untokenize(tokens)


def eval_expr(code, local_dict: DICT, global_dict: DICT):
    """
    Evaluate Python code generated by ``stringify_expr``.

    Generally, ``parse_expr`` should be used.
    """
    expr = eval(
        code, global_dict, local_dict)  # take local objects in preference
    return expr
```



#### Split 24
962 tokens, line: 911 - 1038

```python
def parse_expr(s: str, local_dict: Optional[DICT] = None,
               transformations: tUnion[tTuple[TRANS, ...], str] \
                   = standard_transformations,
               global_dict: Optional[DICT] = None, evaluate=True):
    """Converts the string ``s`` to a SymPy expression, in ``local_dict``.

    Parameters
    ==========

    s : str
        The string to parse.

    local_dict : dict, optional
        A dictionary of local variables to use when parsing.

    global_dict : dict, optional
        A dictionary of global variables. By default, this is initialized
        with ``from sympy import *``; provide this parameter to override
        this behavior (for instance, to parse ``"Q & S"``).

    transformations : tuple or str
        A tuple of transformation functions used to modify the tokens of the
        parsed expression before evaluation. The default transformations
        convert numeric literals into their SymPy equivalents, convert
        undefined variables into SymPy symbols, and allow the use of standard
        mathematical factorial notation (e.g. ``x!``). Selection via
        string is available (see below).

    evaluate : bool, optional
        When False, the order of the arguments will remain as they were in the
        string and automatic simplification that would normally occur is
        suppressed. (see examples)

    Examples
    ========

    >>> from sympy.parsing.sympy_parser import parse_expr
    >>> parse_expr("1/2")
    1/2
    >>> type(_)
    <class 'sympy.core.numbers.Half'>
    >>> from sympy.parsing.sympy_parser import standard_transformations,\\
    ... implicit_multiplication_application
    >>> transformations = (standard_transformations +
    ...     (implicit_multiplication_application,))
    >>> parse_expr("2x", transformations=transformations)
    2*x

    When evaluate=False, some automatic simplifications will not occur:

    >>> parse_expr("2**3"), parse_expr("2**3", evaluate=False)
    (8, 2**3)

    In addition the order of the arguments will not be made canonical.
    This feature allows one to tell exactly how the expression was entered:

    >>> a = parse_expr('1 + x', evaluate=False)
    >>> b = parse_expr('x + 1', evaluate=0)
    >>> a == b
    False
    >>> a.args
    (1, x)
    >>> b.args
    (x, 1)

    Note, however, that when these expressions are printed they will
    appear the same:

    >>> assert str(a) == str(b)

    As a convenience, transformations can be seen by printing ``transformations``:

    >>> from sympy.parsing.sympy_parser import transformations

    >>> print(transformations)
    0: lambda_notation
    1: auto_symbol
    2: repeated_decimals
    3: auto_number
    4: factorial_notation
    5: implicit_multiplication_application
    6: convert_xor
    7: implicit_application
    8: implicit_multiplication
    9: convert_equals_signs
    10: function_exponentiation
    11: rationalize

    The ``T`` object provides a way to select these transformations:

    >>> from sympy.parsing.sympy_parser import T

    If you print it, you will see the same list as shown above.

    >>> str(T) == str(transformations)
    True

    Standard slicing will return a tuple of transformations:

    >>> T[:5] == standard_transformations
    True

    So ``T`` can be used to specify the parsing transformations:

    >>> parse_expr("2x", transformations=T[:5])
    Traceback (most recent call last):
    ...
    SyntaxError: invalid syntax
    >>> parse_expr("2x", transformations=T[:6])
    2*x
    >>> parse_expr('.3', transformations=T[3, 11])
    3/10
    >>> parse_expr('.3x', transformations=T[:])
    3*x/10

    As a further convenience, strings 'implicit' and 'all' can be used
    to select 0-5 and all the transformations, respectively.

    >>> parse_expr('.3x', transformations='all')
    3*x/10

    See Also
    ========

    stringify_expr, eval_expr, standard_transformations,
    implicit_multiplication_application

    """
    # ... other code
```



#### Split 25
411 tokens, line: 1040 - 1087

```python
def parse_expr(s: str, local_dict: Optional[DICT] = None,
               transformations: tUnion[tTuple[TRANS, ...], str] \
                   = standard_transformations,
               global_dict: Optional[DICT] = None, evaluate=True):

    if local_dict is None:
        local_dict = {}
    elif not isinstance(local_dict, dict):
        raise TypeError('expecting local_dict to be a dict')
    elif null in local_dict:
        raise ValueError('cannot use "" in local_dict')

    if global_dict is None:
        global_dict = {}
        exec('from sympy import *', global_dict)

        builtins_dict = vars(builtins)
        for name, obj in builtins_dict.items():
            if isinstance(obj, types.BuiltinFunctionType):
                global_dict[name] = obj
        global_dict['max'] = Max
        global_dict['min'] = Min

    elif not isinstance(global_dict, dict):
        raise TypeError('expecting global_dict to be a dict')

    transformations = transformations or ()
    if isinstance(transformations, str):
        if transformations == 'all':
            _transformations = T[:]
        elif transformations == 'implicit':
            _transformations = T[:6]
        else:
            raise ValueError('unknown transformation group name')
    else:
        _transformations = transformations

    code = stringify_expr(s, local_dict, global_dict, _transformations)

    if not evaluate:
        code = compile(evaluateFalse(code), '<string>', 'eval')

    try:
        rv = eval_expr(code, local_dict, global_dict)
        # restore neutral definitions for names
        for i in local_dict.pop(null, ()):
            local_dict[i] = null
        return rv
    except Exception as e:
        # restore neutral definitions for names
        for i in local_dict.pop(null, ()):
            local_dict[i] = null
        raise e from ValueError(f"Error from parse_expr with transformed code: {code!r}")
```



#### Split 26
379 tokens, line: 1090 - 1135

```python
def evaluateFalse(s: str):
    """
    Replaces operators with the SymPy equivalent and sets evaluate=False.
    """
    node = ast.parse(s)
    transformed_node = EvaluateFalseTransformer().visit(node)
    # node is a Module, we want an Expression
    transformed_node = ast.Expression(transformed_node.body[0].value)

    return ast.fix_missing_locations(transformed_node)


class EvaluateFalseTransformer(ast.NodeTransformer):
    operators = {
        ast.Add: 'Add',
        ast.Mult: 'Mul',
        ast.Pow: 'Pow',
        ast.Sub: 'Add',
        ast.Div: 'Mul',
        ast.BitOr: 'Or',
        ast.BitAnd: 'And',
        ast.BitXor: 'Not',
    }
    functions = (
        'Abs', 'im', 're', 'sign', 'arg', 'conjugate',
        'acos', 'acot', 'acsc', 'asec', 'asin', 'atan',
        'acosh', 'acoth', 'acsch', 'asech', 'asinh', 'atanh',
        'cos', 'cot', 'csc', 'sec', 'sin', 'tan',
        'cosh', 'coth', 'csch', 'sech', 'sinh', 'tanh',
        'exp', 'ln', 'log', 'sqrt', 'cbrt',
    )

    def flatten(self, args, func):
        result = []
        for arg in args:
            if isinstance(arg, ast.Call):
                arg_func = arg.func
                if isinstance(arg_func, ast.Call):
                    arg_func = arg_func.func
                if arg_func.id == func:
                    result.extend(self.flatten(arg.args, func))
                else:
                    result.append(arg)
            else:
                result.append(arg)
        return result
```



#### Split 27
532 tokens, line: 1137 - 1193

```python
class EvaluateFalseTransformer(ast.NodeTransformer):

    def visit_BinOp(self, node):
        if node.op.__class__ in self.operators:
            sympy_class = self.operators[node.op.__class__]
            right = self.visit(node.right)
            left = self.visit(node.left)

            rev = False
            if isinstance(node.op, ast.Sub):
                right = ast.Call(
                    func=ast.Name(id='Mul', ctx=ast.Load()),
                    args=[ast.UnaryOp(op=ast.USub(), operand=ast.Num(1)), right],
                    keywords=[ast.keyword(arg='evaluate', value=ast.NameConstant(value=False, ctx=ast.Load()))],
                    starargs=None,
                    kwargs=None
                )
            elif isinstance(node.op, ast.Div):
                if isinstance(node.left, ast.UnaryOp):
                    left, right = right, left
                    rev = True
                    left = ast.Call(
                    func=ast.Name(id='Pow', ctx=ast.Load()),
                    args=[left, ast.UnaryOp(op=ast.USub(), operand=ast.Num(1))],
                    keywords=[ast.keyword(arg='evaluate', value=ast.NameConstant(value=False, ctx=ast.Load()))],
                    starargs=None,
                    kwargs=None
                )
                else:
                    right = ast.Call(
                    func=ast.Name(id='Pow', ctx=ast.Load()),
                    args=[right, ast.UnaryOp(op=ast.USub(), operand=ast.Num(1))],
                    keywords=[ast.keyword(arg='evaluate', value=ast.NameConstant(value=False, ctx=ast.Load()))],
                    starargs=None,
                    kwargs=None
                )

            if rev:  # undo reversal
                left, right = right, left
            new_node = ast.Call(
                func=ast.Name(id=sympy_class, ctx=ast.Load()),
                args=[left, right],
                keywords=[ast.keyword(arg='evaluate', value=ast.NameConstant(value=False, ctx=ast.Load()))],
                starargs=None,
                kwargs=None
            )

            if sympy_class in ('Add', 'Mul'):
                # Denest Add or Mul as appropriate
                new_node.args = self.flatten(new_node.args, sympy_class)

            return new_node
        return node

    def visit_Call(self, node):
        new_node = self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id in self.functions:
            new_node.keywords.append(ast.keyword(arg='evaluate', value=ast.NameConstant(value=False, ctx=ast.Load())))
        return new_node
```



#### Split 28
117 tokens, line: 1196 - 1210

```python
_transformation = {  # items can be added but never re-ordered
0: lambda_notation,
1: auto_symbol,
2: repeated_decimals,
3: auto_number,
4: factorial_notation,
5: implicit_multiplication_application,
6: convert_xor,
7: implicit_application,
8: implicit_multiplication,
9: convert_equals_signs,
10: function_exponentiation,
11: rationalize}

transformations = '\n'.join('%s: %s' % (i, func_name(f)) for i, f in _transformation.items())
```



#### Split 29
181 tokens, line: 1213 - 1242

```python
class _T():
    """class to retrieve transformations from a given slice

    EXAMPLES
    ========

    >>> from sympy.parsing.sympy_parser import T, standard_transformations
    >>> assert T[:5] == standard_transformations
    """
    def __init__(self):
        self.N = len(_transformation)

    def __str__(self):
        return transformations

    def __getitem__(self, t):
        if not type(t) is tuple:
            t = (t,)
        i = []
        for ti in t:
            if type(ti) is int:
                i.append(range(self.N)[ti])
            elif type(ti) is slice:
                i.extend(range(*ti.indices(self.N)))
            else:
                raise TypeError('unexpected slice arg')
        return tuple([_transformation[_] for _ in i])

T = _T()
```
