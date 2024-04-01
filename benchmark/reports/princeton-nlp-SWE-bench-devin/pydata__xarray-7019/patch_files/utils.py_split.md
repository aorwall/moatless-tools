

## EpicSplitter

38 chunks

#### Split 1
357 tokens, line: 1 - 102

```python
"""Internal utilities; not for external use"""
# Some functions in this module are derived from functions in pandas. For
from __future__ import annotations

import contextlib
import functools
import importlib
import inspect
import io
import itertools
import math
import os
import re
import sys
import warnings
from collections.abc import (
    Collection,
    Container,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    MutableSet,
    Sequence,
)
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Literal,
    TypeVar,
    cast,
    overload,
)

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from xarray.core.types import Dims, ErrorOptionsWithWarn, OrderedDims

K = TypeVar("K")
V = TypeVar("V")
T = TypeVar("T")


def alias_message(old_name: str, new_name: str) -> str:
    return f"{old_name} has been deprecated. Use {new_name} instead."


def alias_warning(old_name: str, new_name: str, stacklevel: int = 3) -> None:
    warnings.warn(
        alias_message(old_name, new_name), FutureWarning, stacklevel=stacklevel
    )


def alias(obj: Callable[..., T], old_name: str) -> Callable[..., T]:
    assert isinstance(old_name, str)

    @functools.wraps(obj)
    def wrapper(*args, **kwargs):
        alias_warning(old_name, obj.__name__)
        return obj(*args, **kwargs)

    wrapper.__doc__ = alias_message(old_name, obj.__name__)
    return wrapper
```



#### Split 2
121 tokens, line: 105 - 122

```python
def get_valid_numpy_dtype(array: np.ndarray | pd.Index):
    """Return a numpy compatible dtype from either
    a numpy array or a pandas.Index.

    Used for wrapping a pandas.Index as an xarray,Variable.

    """
    if isinstance(array, pd.PeriodIndex):
        dtype = np.dtype("O")
    elif hasattr(array, "categories"):
        # category isn't a real numpy dtype
        dtype = array.categories.dtype
    elif not is_valid_numpy_dtype(array.dtype):
        dtype = np.dtype("O")
    else:
        dtype = array.dtype

    return dtype
```



#### Split 3
211 tokens, line: 125 - 153

```python
def maybe_coerce_to_str(index, original_coords):
    """maybe coerce a pandas Index back to a nunpy array of type str

    pd.Index uses object-dtype to store str - try to avoid this for coords
    """
    from xarray.core import dtypes

    try:
        result_type = dtypes.result_type(*original_coords)
    except TypeError:
        pass
    else:
        if result_type.kind in "SU":
            index = np.asarray(index, dtype=result_type.type)

    return index


def maybe_wrap_array(original, new_array):
    """Wrap a transformed array with __array_wrap__ if it can be done safely.

    This lets us treat arbitrary functions that take and return ndarray objects
    like ufuncs, as long as they return an array with the same shape.
    """
    # in case func lost array's metadata
    if isinstance(new_array, np.ndarray) and new_array.shape == original.shape:
        return original.__array_wrap__(new_array)
    else:
        return new_array
```



#### Split 4
152 tokens, line: 156 - 170

```python
def equivalent(first: T, second: T) -> bool:
    """Compare two objects for equivalence (identity or equality), using
    array_equiv if either object is an ndarray. If both objects are lists,
    equivalent is sequentially called on all the elements.
    """
    # TODO: refactor to avoid circular import
    from xarray.core import duck_array_ops

    if first is second:
        return True
    if isinstance(first, np.ndarray) or isinstance(second, np.ndarray):
        return duck_array_ops.array_equiv(first, second)
    if isinstance(first, list) or isinstance(second, list):
        return list_equiv(first, second)
    return (first == second) or (pd.isnull(first) and pd.isnull(second))
```



#### Split 5
121 tokens, line: 173 - 189

```python
def list_equiv(first, second):
    equiv = True
    if len(first) != len(second):
        return False
    else:
        for f, s in zip(first, second):
            equiv = equiv and equivalent(f, s)
    return equiv


def peek_at(iterable: Iterable[T]) -> tuple[T, Iterator[T]]:
    """Returns the first value from iterable, as well as a new iterator with
    the same content as the original iterable
    """
    gen = iter(iterable)
    peek = next(gen)
    return peek, itertools.chain([peek], gen)
```



#### Split 6
207 tokens, line: 192 - 217

```python
def update_safety_check(
    first_dict: Mapping[K, V],
    second_dict: Mapping[K, V],
    compat: Callable[[V, V], bool] = equivalent,
) -> None:
    """Check the safety of updating one dictionary with another.

    Raises ValueError if dictionaries have non-compatible values for any key,
    where compatibility is determined by identity (they are the same item) or
    the `compat` function.

    Parameters
    ----------
    first_dict, second_dict : dict-like
        All items in the second dictionary are checked against for conflicts
        against items in the first dictionary.
    compat : function, optional
        Binary operator to determine if two values are compatible. By default,
        checks for equivalence.
    """
    for k, v in second_dict.items():
        if k in first_dict and not compat(v, first_dict[k]):
            raise ValueError(
                "unsafe to merge dictionaries without "
                f"overriding values; conflicting key {k!r}"
            )
```



#### Split 7
156 tokens, line: 220 - 240

```python
def remove_incompatible_items(
    first_dict: MutableMapping[K, V],
    second_dict: Mapping[K, V],
    compat: Callable[[V, V], bool] = equivalent,
) -> None:
    """Remove incompatible items from the first dictionary in-place.

    Items are retained if their keys are found in both dictionaries and the
    values are compatible.

    Parameters
    ----------
    first_dict, second_dict : dict-like
        Mappings to merge.
    compat : function, optional
        Binary operator to determine if two values are compatible. By default,
        checks for equivalence.
    """
    for k in list(first_dict):
        if k not in second_dict or not compat(first_dict[k], second_dict[k]):
            del first_dict[k]
```



#### Split 8
189 tokens, line: 243 - 267

```python
# It's probably OK to give this as a TypeGuard; though it's not perfectly robust.
def is_dict_like(value: Any) -> TypeGuard[Mapping]:
    return hasattr(value, "keys") and hasattr(value, "__getitem__")


def is_full_slice(value: Any) -> bool:
    return isinstance(value, slice) and value == slice(None)


def is_list_like(value: Any) -> TypeGuard[list | tuple]:
    return isinstance(value, (list, tuple))


def is_duck_array(value: Any) -> bool:
    if isinstance(value, np.ndarray):
        return True
    return (
        hasattr(value, "ndim")
        and hasattr(value, "shape")
        and hasattr(value, "dtype")
        and (
            (hasattr(value, "__array_function__") and hasattr(value, "__array_ufunc__"))
            or hasattr(value, "__array_namespace__")
        )
    )
```



#### Split 9
160 tokens, line: 270 - 286

```python
def either_dict_or_kwargs(
    pos_kwargs: Mapping[Any, T] | None,
    kw_kwargs: Mapping[str, T],
    func_name: str,
) -> Mapping[Hashable, T]:
    if pos_kwargs is None or pos_kwargs == {}:
        # Need an explicit cast to appease mypy due to invariance; see
        # https://github.com/python/mypy/issues/6228
        return cast(Mapping[Hashable, T], kw_kwargs)

    if not is_dict_like(pos_kwargs):
        raise ValueError(f"the first argument to .{func_name} must be a dictionary")
    if kw_kwargs:
        raise ValueError(
            f"cannot specify both keyword and positional arguments to .{func_name}"
        )
    return pos_kwargs
```



#### Split 10
113 tokens, line: 289 - 302

```python
def _is_scalar(value, include_0d):
    from xarray.core.variable import NON_NUMPY_SUPPORTED_ARRAY_TYPES

    if include_0d:
        include_0d = getattr(value, "ndim", None) == 0
    return (
        include_0d
        or isinstance(value, (str, bytes))
        or not (
            isinstance(value, (Iterable,) + NON_NUMPY_SUPPORTED_ARRAY_TYPES)
            or hasattr(value, "__array_function__")
            or hasattr(value, "__array_namespace__")
        )
    )
```



#### Split 11
391 tokens, line: 305 - 356

```python
# See GH5624, this is a convoluted way to allow type-checking to use `TypeGuard` without
# requiring typing_extensions as a required dependency to _run_ the code (it is required
# to type-check).
try:
    if sys.version_info >= (3, 10):
        from typing import TypeGuard
    else:
        from typing_extensions import TypeGuard
except ImportError:
    if TYPE_CHECKING:
        raise
    else:

        def is_scalar(value: Any, include_0d: bool = True) -> bool:
            """Whether to treat a value as a scalar.

            Any non-iterable, string, or 0-D array
            """
            return _is_scalar(value, include_0d)

else:

    def is_scalar(value: Any, include_0d: bool = True) -> TypeGuard[Hashable]:
        """Whether to treat a value as a scalar.

        Any non-iterable, string, or 0-D array
        """
        return _is_scalar(value, include_0d)


def is_valid_numpy_dtype(dtype: Any) -> bool:
    try:
        np.dtype(dtype)
    except (TypeError, ValueError):
        return False
    else:
        return True


def to_0d_object_array(value: Any) -> np.ndarray:
    """Given a value, wrap it in a 0-D numpy.ndarray with dtype=object."""
    result = np.empty((), dtype=object)
    result[()] = value
    return result


def to_0d_array(value: Any) -> np.ndarray:
    """Given a value, wrap it in a 0-D numpy.ndarray."""
    if np.isscalar(value) or (isinstance(value, np.ndarray) and value.ndim == 0):
        return np.array(value)
    else:
        return to_0d_object_array(value)
```



#### Split 12
166 tokens, line: 359 - 383

```python
def dict_equiv(
    first: Mapping[K, V],
    second: Mapping[K, V],
    compat: Callable[[V, V], bool] = equivalent,
) -> bool:
    """Test equivalence of two dict-like objects. If any of the values are
    numpy arrays, compare them correctly.

    Parameters
    ----------
    first, second : dict-like
        Dictionaries to compare for equality
    compat : function, optional
        Binary operator to determine if two values are compatible. By default,
        checks for equivalence.

    Returns
    -------
    equals : bool
        True if the dictionaries are equal
    """
    for k in first:
        if k not in second or not compat(first[k], second[k]):
            return False
    return all(k in first for k in second)
```



#### Split 13
168 tokens, line: 386 - 411

```python
def compat_dict_intersection(
    first_dict: Mapping[K, V],
    second_dict: Mapping[K, V],
    compat: Callable[[V, V], bool] = equivalent,
) -> MutableMapping[K, V]:
    """Return the intersection of two dictionaries as a new dictionary.

    Items are retained if their keys are found in both dictionaries and the
    values are compatible.

    Parameters
    ----------
    first_dict, second_dict : dict-like
        Mappings to merge.
    compat : function, optional
        Binary operator to determine if two values are compatible. By default,
        checks for equivalence.

    Returns
    -------
    intersection : dict
        Intersection of the contents.
    """
    new_dict = dict(first_dict)
    remove_incompatible_items(new_dict, second_dict, compat)
    return new_dict
```



#### Split 14
177 tokens, line: 414 - 440

```python
def compat_dict_union(
    first_dict: Mapping[K, V],
    second_dict: Mapping[K, V],
    compat: Callable[[V, V], bool] = equivalent,
) -> MutableMapping[K, V]:
    """Return the union of two dictionaries as a new dictionary.

    An exception is raised if any keys are found in both dictionaries and the
    values are not compatible.

    Parameters
    ----------
    first_dict, second_dict : dict-like
        Mappings to merge.
    compat : function, optional
        Binary operator to determine if two values are compatible. By default,
        checks for equivalence.

    Returns
    -------
    union : dict
        union of the contents.
    """
    new_dict = dict(first_dict)
    update_safety_check(first_dict, second_dict, compat)
    new_dict.update(second_dict)
    return new_dict
```



#### Split 15
202 tokens, line: 443 - 471

```python
class Frozen(Mapping[K, V]):
    """Wrapper around an object implementing the mapping interface to make it
    immutable. If you really want to modify the mapping, the mutable version is
    saved under the `mapping` attribute.
    """

    __slots__ = ("mapping",)

    def __init__(self, mapping: Mapping[K, V]):
        self.mapping = mapping

    def __getitem__(self, key: K) -> V:
        return self.mapping[key]

    def __iter__(self) -> Iterator[K]:
        return iter(self.mapping)

    def __len__(self) -> int:
        return len(self.mapping)

    def __contains__(self, key: object) -> bool:
        return key in self.mapping

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.mapping!r})"


def FrozenDict(*args, **kwargs) -> Frozen:
    return Frozen(dict(*args, **kwargs))
```



#### Split 16
217 tokens, line: 474 - 500

```python
class HybridMappingProxy(Mapping[K, V]):
    """Implements the Mapping interface. Uses the wrapped mapping for item lookup
    and a separate wrapped keys collection for iteration.

    Can be used to construct a mapping object from another dict-like object without
    eagerly accessing its items or when a mapping object is expected but only
    iteration over keys is actually used.

    Note: HybridMappingProxy does not validate consistency of the provided `keys`
    and `mapping`. It is the caller's responsibility to ensure that they are
    suitable for the task at hand.
    """

    __slots__ = ("_keys", "mapping")

    def __init__(self, keys: Collection[K], mapping: Mapping[K, V]):
        self._keys = keys
        self.mapping = mapping

    def __getitem__(self, key: K) -> V:
        return self.mapping[key]

    def __iter__(self) -> Iterator[K]:
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)
```



#### Split 17
274 tokens, line: 503 - 543

```python
class OrderedSet(MutableSet[T]):
    """A simple ordered set.

    The API matches the builtin set, but it preserves insertion order of elements, like
    a dict. Note that, unlike in an OrderedDict, equality tests are not order-sensitive.
    """

    _d: dict[T, None]

    __slots__ = ("_d",)

    def __init__(self, values: Iterable[T] | None = None):
        self._d = {}
        if values is not None:
            self.update(values)

    # Required methods for MutableSet

    def __contains__(self, value: Hashable) -> bool:
        return value in self._d

    def __iter__(self) -> Iterator[T]:
        return iter(self._d)

    def __len__(self) -> int:
        return len(self._d)

    def add(self, value: T) -> None:
        self._d[value] = None

    def discard(self, value: T) -> None:
        del self._d[value]

    # Additional methods

    def update(self, values: Iterable[T]) -> None:
        for v in values:
            self._d[v] = None

    def __repr__(self) -> str:
        return f"{type(self).__name__}({list(self)!r})"
```



#### Split 18
195 tokens, line: 546 - 581

```python
class NdimSizeLenMixin:
    """Mixin class that extends a class that defines a ``shape`` property to
    one that also defines ``ndim``, ``size`` and ``__len__``.
    """

    __slots__ = ()

    @property
    def ndim(self: Any) -> int:
        """
        Number of array dimensions.

        See Also
        --------
        numpy.ndarray.ndim
        """
        return len(self.shape)

    @property
    def size(self: Any) -> int:
        """
        Number of elements in the array.

        Equal to ``np.prod(a.shape)``, i.e., the product of the array’s dimensions.

        See Also
        --------
        numpy.ndarray.size
        """
        return math.prod(self.shape)

    def __len__(self: Any) -> int:
        try:
            return self.shape[0]
        except IndexError:
            raise TypeError("len() of unsized object")
```



#### Split 19
167 tokens, line: 584 - 606

```python
class NDArrayMixin(NdimSizeLenMixin):
    """Mixin class for making wrappers of N-dimensional arrays that conform to
    the ndarray interface required for the data argument to Variable objects.

    A subclass should set the `array` property and override one or more of
    `dtype`, `shape` and `__getitem__`.
    """

    __slots__ = ()

    @property
    def dtype(self: Any) -> np.dtype:
        return self.array.dtype

    @property
    def shape(self: Any) -> tuple[int, ...]:
        return self.array.shape

    def __getitem__(self: Any, key):
        return self.array[key]

    def __repr__(self: Any) -> str:
        return f"{type(self).__name__}(array={self.array!r})"
```



#### Split 20
151 tokens, line: 609 - 631

```python
class ReprObject:
    """Object that prints as the given value, for use with sentinel values."""

    __slots__ = ("_value",)

    def __init__(self, value: str):
        self._value = value

    def __repr__(self) -> str:
        return self._value

    def __eq__(self, other) -> bool:
        if isinstance(other, ReprObject):
            return self._value == other._value
        return False

    def __hash__(self) -> int:
        return hash((type(self), self._value))

    def __dask_tokenize__(self):
        from dask.base import normalize_token

        return normalize_token((type(self), self._value))
```



#### Split 21
140 tokens, line: 634 - 652

```python
@contextlib.contextmanager
def close_on_error(f):
    """Context manager to ensure that a file opened by xarray is closed if an
    exception is raised before the user sees the file object.
    """
    try:
        yield
    except Exception:
        f.close()
        raise


def is_remote_uri(path: str) -> bool:
    """Finds URLs of the form protocol:// or protocol::

    This also matches for http[s]://, which were the only remote URLs
    supported in <=v0.16.2.
    """
    return bool(re.search(r"^[a-z][a-z0-9]*(\://|\:\:)", path))
```



#### Split 22
119 tokens, line: 655 - 666

```python
def read_magic_number_from_file(filename_or_obj, count=8) -> bytes:
    # check byte header to determine file type
    if isinstance(filename_or_obj, bytes):
        magic_number = filename_or_obj[:count]
    elif isinstance(filename_or_obj, io.IOBase):
        if filename_or_obj.tell() != 0:
            filename_or_obj.seek(0)
        magic_number = filename_or_obj.read(count)
        filename_or_obj.seek(0)
    else:
        raise TypeError(f"cannot read the magic number from {type(filename_or_obj)}")
    return magic_number
```



#### Split 23
160 tokens, line: 669 - 687

```python
def try_read_magic_number_from_path(pathlike, count=8) -> bytes | None:
    if isinstance(pathlike, str) or hasattr(pathlike, "__fspath__"):
        path = os.fspath(pathlike)
        try:
            with open(path, "rb") as f:
                return read_magic_number_from_file(f, count)
        except (FileNotFoundError, TypeError):
            pass
    return None


def try_read_magic_number_from_file_or_path(filename_or_obj, count=8) -> bytes | None:
    magic_number = try_read_magic_number_from_path(filename_or_obj, count)
    if magic_number is None:
        try:
            magic_number = read_magic_number_from_file(filename_or_obj, count)
        except TypeError:
            pass
    return magic_number
```



#### Split 24
105 tokens, line: 690 - 702

```python
def is_uniform_spaced(arr, **kwargs) -> bool:
    """Return True if values of an array are uniformly spaced and sorted.

    >>> is_uniform_spaced(range(5))
    True
    >>> is_uniform_spaced([-4, 0, 100])
    False

    kwargs are additional arguments to ``np.isclose``
    """
    arr = np.array(arr, dtype=float)
    diffs = np.diff(arr)
    return bool(np.isclose(diffs.min(), diffs.max(), **kwargs))
```



#### Split 25
328 tokens, line: 705 - 752

```python
def hashable(v: Any) -> TypeGuard[Hashable]:
    """Determine whether `v` can be hashed."""
    try:
        hash(v)
    except TypeError:
        return False
    return True


def iterable(v: Any) -> TypeGuard[Iterable[Any]]:
    """Determine whether `v` is iterable."""
    try:
        iter(v)
    except TypeError:
        return False
    return True


def iterable_of_hashable(v: Any) -> TypeGuard[Iterable[Hashable]]:
    """Determine whether `v` is an Iterable of Hashables."""
    try:
        it = iter(v)
    except TypeError:
        return False
    return all(hashable(elm) for elm in it)


def decode_numpy_dict_values(attrs: Mapping[K, V]) -> dict[K, V]:
    """Convert attribute values from numpy objects to native Python objects,
    for use in to_dict
    """
    attrs = dict(attrs)
    for k, v in attrs.items():
        if isinstance(v, np.ndarray):
            attrs[k] = v.tolist()
        elif isinstance(v, np.generic):
            attrs[k] = v.item()
    return attrs


def ensure_us_time_resolution(val):
    """Convert val out of numpy time, for use in to_dict.
    Needed because of numpy bug GH#7619"""
    if np.issubdtype(val.dtype, np.datetime64):
        val = val.astype("datetime64[us]")
    elif np.issubdtype(val.dtype, np.timedelta64):
        val = val.astype("timedelta64[us]")
    return val
```



#### Split 26
305 tokens, line: 755 - 790

```python
class HiddenKeyDict(MutableMapping[K, V]):
    """Acts like a normal dictionary, but hides certain keys."""

    __slots__ = ("_data", "_hidden_keys")

    # ``__init__`` method required to create instance from class.

    def __init__(self, data: MutableMapping[K, V], hidden_keys: Iterable[K]):
        self._data = data
        self._hidden_keys = frozenset(hidden_keys)

    def _raise_if_hidden(self, key: K) -> None:
        if key in self._hidden_keys:
            raise KeyError(f"Key `{key!r}` is hidden.")

    # The next five methods are requirements of the ABC.
    def __setitem__(self, key: K, value: V) -> None:
        self._raise_if_hidden(key)
        self._data[key] = value

    def __getitem__(self, key: K) -> V:
        self._raise_if_hidden(key)
        return self._data[key]

    def __delitem__(self, key: K) -> None:
        self._raise_if_hidden(key)
        del self._data[key]

    def __iter__(self) -> Iterator[K]:
        for k in self._data:
            if k not in self._hidden_keys:
                yield k

    def __len__(self) -> int:
        num_hidden = len(self._hidden_keys & self._data.keys())
        return len(self._data) - num_hidden
```



#### Split 27
258 tokens, line: 793 - 820

```python
def infix_dims(
    dims_supplied: Collection,
    dims_all: Collection,
    missing_dims: ErrorOptionsWithWarn = "raise",
) -> Iterator:
    """
    Resolves a supplied list containing an ellipsis representing other items, to
    a generator with the 'realized' list of all items
    """
    if ... in dims_supplied:
        if len(set(dims_all)) != len(dims_all):
            raise ValueError("Cannot use ellipsis with repeated dims")
        if list(dims_supplied).count(...) > 1:
            raise ValueError("More than one ellipsis supplied")
        other_dims = [d for d in dims_all if d not in dims_supplied]
        existing_dims = drop_missing_dims(dims_supplied, dims_all, missing_dims)
        for d in existing_dims:
            if d is ...:
                yield from other_dims
            else:
                yield d
    else:
        existing_dims = drop_missing_dims(dims_supplied, dims_all, missing_dims)
        if set(existing_dims) ^ set(dims_all):
            raise ValueError(
                f"{dims_supplied} must be a permuted list of {dims_all}, unless `...` is included"
            )
        yield from existing_dims
```



#### Split 28
145 tokens, line: 823 - 838

```python
def get_temp_dimname(dims: Container[Hashable], new_dim: Hashable) -> Hashable:
    """Get an new dimension name based on new_dim, that is not used in dims.
    If the same name exists, we add an underscore(s) in the head.

    Example1:
        dims: ['a', 'b', 'c']
        new_dim: ['_rolling']
        -> ['_rolling']
    Example2:
        dims: ['a', 'b', 'c', '_rolling']
        new_dim: ['_rolling']
        -> ['__rolling']
    """
    while new_dim in dims:
        new_dim = "_" + str(new_dim)
    return new_dim
```



#### Split 29
292 tokens, line: 841 - 885

```python
def drop_dims_from_indexers(
    indexers: Mapping[Any, Any],
    dims: Iterable[Hashable] | Mapping[Any, int],
    missing_dims: ErrorOptionsWithWarn,
) -> Mapping[Hashable, Any]:
    """Depending on the setting of missing_dims, drop any dimensions from indexers that
    are not present in dims.

    Parameters
    ----------
    indexers : dict
    dims : sequence
    missing_dims : {"raise", "warn", "ignore"}
    """

    if missing_dims == "raise":
        invalid = indexers.keys() - set(dims)
        if invalid:
            raise ValueError(
                f"Dimensions {invalid} do not exist. Expected one or more of {dims}"
            )

        return indexers

    elif missing_dims == "warn":
        # don't modify input
        indexers = dict(indexers)

        invalid = indexers.keys() - set(dims)
        if invalid:
            warnings.warn(
                f"Dimensions {invalid} do not exist. Expected one or more of {dims}"
            )
        for key in invalid:
            indexers.pop(key)

        return indexers

    elif missing_dims == "ignore":
        return {key: val for key, val in indexers.items() if key in dims}

    else:
        raise ValueError(
            f"Unrecognised option {missing_dims} for missing_dims argument"
        )
```



#### Split 30
291 tokens, line: 888 - 928

```python
def drop_missing_dims(
    supplied_dims: Iterable[Hashable],
    dims: Iterable[Hashable],
    missing_dims: ErrorOptionsWithWarn,
) -> Iterable[Hashable]:
    """Depending on the setting of missing_dims, drop any dimensions from supplied_dims that
    are not present in dims.

    Parameters
    ----------
    supplied_dims : Iterable of Hashable
    dims : Iterable of Hashable
    missing_dims : {"raise", "warn", "ignore"}
    """

    if missing_dims == "raise":
        supplied_dims_set = {val for val in supplied_dims if val is not ...}
        invalid = supplied_dims_set - set(dims)
        if invalid:
            raise ValueError(
                f"Dimensions {invalid} do not exist. Expected one or more of {dims}"
            )

        return supplied_dims

    elif missing_dims == "warn":
        invalid = set(supplied_dims) - set(dims)
        if invalid:
            warnings.warn(
                f"Dimensions {invalid} do not exist. Expected one or more of {dims}"
            )

        return [val for val in supplied_dims if val in dims or val is ...]

    elif missing_dims == "ignore":
        return [val for val in supplied_dims if val in dims or val is ...]

    else:
        raise ValueError(
            f"Unrecognised option {missing_dims} for missing_dims argument"
        )
```



#### Split 31
146 tokens, line: 931 - 953

```python
T_None = TypeVar("T_None", None, "ellipsis")


@overload
def parse_dims(
    dim: str | Iterable[Hashable] | T_None,
    all_dims: tuple[Hashable, ...],
    *,
    check_exists: bool = True,
    replace_none: Literal[True] = True,
) -> tuple[Hashable, ...]:
    ...


@overload
def parse_dims(
    dim: str | Iterable[Hashable] | T_None,
    all_dims: tuple[Hashable, ...],
    *,
    check_exists: bool = True,
    replace_none: Literal[False],
) -> tuple[Hashable, ...] | T_None:
    ...
```



#### Split 32
290 tokens, line: 956 - 994

```python
def parse_dims(
    dim: Dims,
    all_dims: tuple[Hashable, ...],
    *,
    check_exists: bool = True,
    replace_none: bool = True,
) -> tuple[Hashable, ...] | None | ellipsis:
    """Parse one or more dimensions.

    A single dimension must be always a str, multiple dimensions
    can be Hashables. This supports e.g. using a tuple as a dimension.
    If you supply e.g. a set of dimensions the order cannot be
    conserved, but for sequences it will be.

    Parameters
    ----------
    dim : str, Iterable of Hashable, "..." or None
        Dimension(s) to parse.
    all_dims : tuple of Hashable
        All possible dimensions.
    check_exists: bool, default: True
        if True, check if dim is a subset of all_dims.
    replace_none : bool, default: True
        If True, return all_dims if dim is None or "...".

    Returns
    -------
    parsed_dims : tuple of Hashable
        Input dimensions as a tuple.
    """
    if dim is None or dim is ...:
        if replace_none:
            return all_dims
        return dim
    if isinstance(dim, str):
        dim = (dim,)
    if check_exists:
        _check_dims(set(dim), set(all_dims))
    return tuple(dim)
```



#### Split 33
140 tokens, line: 997 - 1016

```python
@overload
def parse_ordered_dims(
    dim: str | Sequence[Hashable | ellipsis] | T_None,
    all_dims: tuple[Hashable, ...],
    *,
    check_exists: bool = True,
    replace_none: Literal[True] = True,
) -> tuple[Hashable, ...]:
    ...


@overload
def parse_ordered_dims(
    dim: str | Sequence[Hashable | ellipsis] | T_None,
    all_dims: tuple[Hashable, ...],
    *,
    check_exists: bool = True,
    replace_none: Literal[False],
) -> tuple[Hashable, ...] | T_None:
    ...
```



#### Split 34
479 tokens, line: 1019 - 1071

```python
def parse_ordered_dims(
    dim: OrderedDims,
    all_dims: tuple[Hashable, ...],
    *,
    check_exists: bool = True,
    replace_none: bool = True,
) -> tuple[Hashable, ...] | None | ellipsis:
    """Parse one or more dimensions.

    A single dimension must be always a str, multiple dimensions
    can be Hashables. This supports e.g. using a tuple as a dimension.
    An ellipsis ("...") in a sequence of dimensions will be
    replaced with all remaining dimensions. This only makes sense when
    the input is a sequence and not e.g. a set.

    Parameters
    ----------
    dim : str, Sequence of Hashable or "...", "..." or None
        Dimension(s) to parse. If "..." appears in a Sequence
        it always gets replaced with all remaining dims
    all_dims : tuple of Hashable
        All possible dimensions.
    check_exists: bool, default: True
        if True, check if dim is a subset of all_dims.
    replace_none : bool, default: True
        If True, return all_dims if dim is None.

    Returns
    -------
    parsed_dims : tuple of Hashable
        Input dimensions as a tuple.
    """
    if dim is not None and dim is not ... and not isinstance(dim, str) and ... in dim:
        dims_set: set[Hashable | ellipsis] = set(dim)
        all_dims_set = set(all_dims)
        if check_exists:
            _check_dims(dims_set, all_dims_set)
        if len(all_dims_set) != len(all_dims):
            raise ValueError("Cannot use ellipsis with repeated dims")
        dims = tuple(dim)
        if dims.count(...) > 1:
            raise ValueError("More than one ellipsis supplied")
        other_dims = tuple(d for d in all_dims if d not in dims_set)
        idx = dims.index(...)
        return dims[:idx] + other_dims + dims[idx + 1 :]
    else:
        # mypy cannot resolve that the sequence cannot contain "..."
        return parse_dims(  # type: ignore[call-overload]
            dim=dim,
            all_dims=all_dims,
            check_exists=check_exists,
            replace_none=replace_none,
        )
```



#### Split 35
106 tokens, line: 1074 - 1083

```python
def _check_dims(dim: set[Hashable | ellipsis], all_dims: set[Hashable]) -> None:
    wrong_dims = dim - all_dims
    if wrong_dims and wrong_dims != {...}:
        wrong_dims_str = ", ".join(f"'{d!s}'" for d in wrong_dims)
        raise ValueError(
            f"Dimension(s) {wrong_dims_str} do not exist. Expected one or more of {all_dims}"
        )


_Accessor = TypeVar("_Accessor")
```



#### Split 36
191 tokens, line: 1086 - 1109

```python
class UncachedAccessor(Generic[_Accessor]):
    """Acts like a property, but on both classes and class instances

    This class is necessary because some tools (e.g. pydoc and sphinx)
    inspect classes for which property returns itself and not the
    accessor.
    """

    def __init__(self, accessor: type[_Accessor]) -> None:
        self._accessor = accessor

    @overload
    def __get__(self, obj: None, cls) -> type[_Accessor]:
        ...

    @overload
    def __get__(self, obj: object, cls) -> _Accessor:
        ...

    def __get__(self, obj: None | object, cls) -> type[_Accessor] | _Accessor:
        if obj is None:
            return self._accessor

        return self._accessor(obj)  # type: ignore  # assume it is a valid accessor!
```



#### Split 37
266 tokens, line: 1112 - 1161

```python
# Singleton type, as per https://github.com/python/typing/pull/240
class Default(Enum):
    token = 0


_default = Default.token


def iterate_nested(nested_list):
    for item in nested_list:
        if isinstance(item, list):
            yield from iterate_nested(item)
        else:
            yield item


def contains_only_dask_or_numpy(obj) -> bool:
    """Returns True if xarray object contains only numpy or dask arrays.

    Expects obj to be Dataset or DataArray"""
    from xarray.core.dataarray import DataArray
    from xarray.core.pycompat import is_duck_dask_array

    if isinstance(obj, DataArray):
        obj = obj._to_temp_dataset()

    return all(
        [
            isinstance(var.data, np.ndarray) or is_duck_dask_array(var.data)
            for var in obj.variables.values()
        ]
    )


def module_available(module: str) -> bool:
    """Checks whether a module is installed without importing it.

    Use this for a lightweight check and lazy imports.

    Parameters
    ----------
    module : str
        Name of the module.

    Returns
    -------
    available : bool
        Whether the module is installed.
    """
    return importlib.util.find_spec(module) is not None
```



#### Split 38
289 tokens, line: 1164 - 1205

```python
def find_stack_level(test_mode=False) -> int:
    """Find the first place in the stack that is not inside xarray.

    This is unless the code emanates from a test, in which case we would prefer
    to see the xarray source.

    This function is taken from pandas.

    Parameters
    ----------
    test_mode : bool
        Flag used for testing purposes to switch off the detection of test
        directories in the stack trace.

    Returns
    -------
    stacklevel : int
        First level in the stack that is not part of xarray.
    """
    import xarray as xr

    pkg_dir = os.path.dirname(xr.__file__)
    test_dir = os.path.join(pkg_dir, "tests")

    # https://stackoverflow.com/questions/17407119/python-inspect-stack-is-slow
    frame = inspect.currentframe()
    n = 0
    while frame:
        fname = inspect.getfile(frame)
        if fname.startswith(pkg_dir) and (not fname.startswith(test_dir) or test_mode):
            frame = frame.f_back
            n += 1
        else:
            break
    return n


def emit_user_level_warning(message, category=None):
    """Emit a warning at the user level by inspecting the stack trace."""
    stacklevel = find_stack_level()
    warnings.warn(message, category=category, stacklevel=stacklevel)
```
