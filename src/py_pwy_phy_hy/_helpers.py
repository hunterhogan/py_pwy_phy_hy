from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from functools import wraps
from py_pwy_phy_hy import IdentityCallable, PSpec, RVar, SupportsIntIndex, T_co, TVar
from torch import Tensor
from typing import Concatenate, overload, TypeGuard

def compact(arr: Iterable[T_co | None]) -> list[T_co]:
	"""Filter `None` values from `arr` and return the remaining elements as a `list`.

	You can use `compact` to remove all `None` values from an iterable, producing a typed `list` of
	non-`None` elements. `compact` applies `exists` [1] as the predicate for `filter` [2]. The `safe`
	[3] decorator uses `compact` to strip `None` values from a `Sequence` of `Tensor` before passing
	the result to the decorated function.

	Parameters
	----------
	arr : Iterable[T_co | None]
		An iterable that may contain a mix of non-`None` values and `None` values to discard.

	Returns
	-------
	compacted : list[T_co]
		A `list` containing only the non-`None` elements of `arr`, in iteration order.

	See Also
	--------
	exists : Test whether a value is not `None`.
	safe : Decorator that applies `compact` to filter `None` values before calling the wrapped function.

	References
	----------
	[1] py_pwy_phy_hy.exists

	[2] filter - Python documentation
		https://docs.python.org/3/library/functions.html#filter
	[3] py_pwy_phy_hy.safe
	"""
	return [*filter(exists, arr)]

def default(v: TVar | None, d: TVar) -> TVar:
	"""Return `v` when `v` is not `None`, or `d` when `v` is `None`.

	You can use `default` to supply a fallback value for optional parameters and accumulators.
	`default` calls `exists` [1] to test `v`. When `exists(v)` is `True`, `default` returns `v`
	unchanged. When `v` is `None`, `default` returns `d`. Two overloads ensure the return type
	narrows correctly under static analysis.

	Parameters
	----------
	v : TVar | None
		The primary value to test.
	d : TVar
		The fallback value returned when `v` is `None`.

	Returns
	-------
	result : TVar
		`v` when `v` is not `None`, otherwise `d`.

	See Also
	--------
	exists : Test whether a value is not `None`.

	References
	----------
	[1] py_pwy_phy_hy.exists
	"""
	return v if exists(v) else d

def divisible_by(num: float, den: float) -> bool:
	"""Test whether `num` is evenly divisible by `den`.

	You can use `divisible_by` to check divisibility without raising a `ZeroDivisionError` when `den`
	is zero. `divisible_by` returns `False` whenever `den` is zero, and otherwise returns `True` when
	`num % den == 0`.

	Parameters
	----------
	num : float
		The numerator to test.
	den : float
		The denominator. When `den` is `0`, `divisible_by` returns `False` without evaluating `num %
		den`.

	Returns
	-------
	is_divisible : bool
		`True` when `den != 0` and `num % den == 0`, otherwise `False`.
	"""
	return (den != 0) and ((num % den) == 0)

def exists(v: TVar | None) -> TypeGuard[TVar]:
	"""Test whether `v` is not `None`.

	You can use `exists` as a `None`-guard throughout this package. `exists` returns `True` for any
	value that is not `None`, including falsy values such as `0`, `False`, and empty collections. The
	return type is annotated as `TypeGuard[TVar]` [1] so that static analyzers narrow the type of `v`
	to `TVar` in branches guarded by `exists`.

	Parameters
	----------
	v : TVar | None
		The value to test.

	Returns
	-------
	result : bool
		`True` when `v is not None`, otherwise `False`.

	See Also
	--------
	default : Return a fallback value when `v` is `None`.
	compact : Filter `None` values from an iterable.

	Examples
	--------
	From `py_pwy_phy_hy.lens_to_mask` [2], guarding optional parameter `max_len` before use:

		```python
		if not exists(max_len):
			max_len = lens.amax().item()
		```

	References
	----------
	[1] TypeGuard - Python typing documentation
		https://docs.python.org/3/library/typing.html#typing.TypeGuard
	[2] py_pwy_phy_hy.lens_to_mask
	"""
	return v is not None

def first(arr: SupportsIntIndex[TVar]) -> TVar:
	"""Return the element at index `0` of `arr`.

	You can use `first` to retrieve the first element of any sequence that supports integer indexing
	via `SupportsIntIndex` [1]. `first` delegates to `arr[0]`.

	Parameters
	----------
	arr : SupportsIntIndex[TVar]
		A sequence that supports integer indexing. Access with index `0` must be valid.

	Returns
	-------
	element : TVar
		The element at position `0` in `arr`.

	References
	----------
	[1] py_pwy_phy_hy.SupportsIntIndex
	"""
	return arr[0]

def identity(t: TVar, *_args: object, **_kwargs: object) -> TVar:
	"""Return `t` unchanged, ignoring all other arguments.

	You can use `identity` as a no-op callable in contexts that require a function but no
	transformation. `identity` accepts and discards any additional positional or keyword arguments,
	making `identity` a drop-in substitute for any unary or multi-argument callable.
	`py_pwy_phy_hy.maybe` [1] returns `identity` when its `fn` argument is `None`.

	Parameters
	----------
	t : TVar
		The value to return unchanged.
	*args : Any
		Additional positional arguments. Accepted and discarded.
	**kwargs : Any
		Additional keyword arguments. Accepted and discarded.

	Returns
	-------
	result : TVar
		`t`, the same object that was passed as the first argument.

	See Also
	--------
	maybe : Return `identity` when `fn` is `None`, otherwise wrap `fn` to skip `None` inputs.

	References
	----------
	[1] py_pwy_phy_hy.identity
	"""
	return t

def map_values(fn: Callable[[TVar], TVar], v: TVar) -> TVar:
	"""Apply `fn` to every leaf value in a nested `list`, `tuple`, or `dict` structure.

	You can use `map_values` to transform the leaf values of an arbitrarily nested container without
	changing its shape. `map_values` recurses into `list` and `tuple` elements and into `dict`
	values, reassembling each container using its original type. Any value that is not a `list`,
	`tuple`, or `dict` is treated as a leaf and passed directly to `fn`. `dehydrate_config` [1] and
	`rehydrate_config` [2] both use `map_values` to traverse nested checkpoint configuration
	structures.

	Parameters
	----------
	fn : Callable[[TVar], TVar]
		The function to apply to each leaf value. `fn` receives each non-container value and must
		return a value of the same type.
	v : TVar
		The value to transform. May be a `list`, `tuple`, `dict`, or any leaf value.

	Returns
	-------
	transformed : TVar
		The input structure with all leaf values replaced by the results of `fn`.

	See Also
	--------
	dehydrate_config : Serialize nested `Module` instances using `map_values`.
	rehydrate_config : Reconstruct nested `Module` instances using `map_values`.

	References
	----------
	[1] py_pwy_phy_hy.save_load.dehydrate_config

	[2] py_pwy_phy_hy.save_load.rehydrate_config

	[3] tests.test_helpers.test_map_values_transforms_structure
	"""
	if isinstance(v, (list, tuple)):
		return type(v)(map_values(fn, el) for el in v) # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]

	if isinstance(v, dict):
		v = {key: map_values(fn, val) for key, val in v.items()} # pyright: ignore[reportAssignmentType, reportUnknownArgumentType, reportUnknownVariableType]  # ty:ignore[invalid-assignment]

	return fn(v)

@overload
def maybe(fn: Callable[Concatenate[TVar, PSpec], RVar]) -> Callable[Concatenate[TVar | None, PSpec], RVar | None]: ...
@overload
def maybe(fn: None) -> IdentityCallable: ...
def maybe(
	fn: Callable[Concatenate[TVar, PSpec], RVar] | None,
) -> Callable[Concatenate[TVar | None, PSpec], RVar | None] | IdentityCallable:
	"""Wrap `fn` so that the wrapped function returns `None` when the first argument is `None`.

	You can use this function to conditionally apply `fn` without adding an explicit `if`/`else`
	guard at every call site. The returned callable passes all positional and keyword arguments to
	`fn` unchanged when the first argument is not `None`, and returns `None` immediately when the
	first argument is `None`. When `fn` is `None`, the function returns `identity` [1], which passes
	its first argument through without modification.

	Parameters
	----------
	fn : Callable[Concatenate[TVar, PSpec], RVar] | None
		The callable to wrap, or `None`. The first positional parameter of `fn` is the value that may
		be `None`. Pass `None` to receive an identity function.

	Returns
	-------
	wrapped : Callable[Concatenate[TVar | None, PSpec], RVar | None] | Callable[..., TVar]
		A wrapped version of `fn` that short-circuits to `None` when the first argument is `None`, or
		`identity` when `fn` is `None`.

	See Also
	--------
	identity : Return the first argument unchanged.

	Examples
	--------
	Skip the function when the first argument is `None` [2]:

		```python
		from py_pwy_phy_hy import maybe

		result = maybe(lambda t: t + 1)(None)
		# result is None
		```

	Pass `None` as `fn` to receive an identity function [2]:

		```python
		result = maybe(None)(1)
		# result == 1
		```

	Conditionally convert episode lengths to a mask [3]:

		```python
		from py_pwy_phy_hy import maybe, lens_to_mask

		mask = maybe(lens_to_mask)(episode_lens, seq_len)
		# mask is None when episode_lens is None,
		# otherwise mask == lens_to_mask(episode_lens, seq_len)
		```

	References
	----------
	[1] py_pwy_phy_hy.identity

	[2] tests.test_utils.test_maybe

	[3] metacontroller.metacontroller.ratio_loss
		https://context7.com/lucidrains/metacontroller
	"""
	if not exists(fn):
		return identity

	@wraps(fn)
	def inner(t: TVar | None, *args: PSpec.args, **kwargs: PSpec.kwargs) -> RVar | None:
		if not exists(t):
			return None

		return fn(t, *args, **kwargs)

	return inner

def once(fn: Callable[PSpec, RVar]) -> Callable[PSpec, RVar | None]:
	"""Wrap a callable so the callable executes at most once.

	You can use this function to restrict `fn` (***f***u***n***ction) to a single execution. On all
	subsequent calls, `fn` returns `None`.

	Parameters
	----------
	fn : Callable[PSpec, RVar]
		(***f***u***n***ction) The `Callable` to wrap.

	Returns
	-------
	single_use_fn : Callable[PSpec, RVar | None]
		A wrapper that invokes fn on the first call and returns None on all subsequent
		calls. The wrapper accepts the same arguments as fn.

	Examples
	--------
	This example creates a variant of the built-in `print` that emits output only once:

		```python
		print_once = once(print)

		if device_properties.major == 8 and device_properties.minor == 0:
			print_once('A100 GPU detected, using flash attention if input tensor is on cuda')
		```

	References
	----------
	[1] functools.wraps - Python standard library
		https://docs.python.org/3/library/functools.html#functools.wraps
	[2] py_pwy_phy_hy (PSpec, RVar)
	"""
	called: bool = False

	@wraps(fn)
	def inner(*args: PSpec.args, **kwargs: PSpec.kwargs) -> RVar | None:
		nonlocal called

		if called:
			return None

		called = True
		return fn(*args, **kwargs)

	return inner

def safe(
	fn: Callable[Concatenate[Sequence[Tensor], PSpec], Tensor | None],
) -> Callable[Concatenate[Sequence[Tensor | None], PSpec], Tensor | None]:
	"""Wrap `fn` so that `None` values are filtered from the first argument before the call.

	You can use `safe` as a decorator to make a function that accepts a `Sequence[Tensor]` tolerate
	`None` values in the sequence. The decorated function accepts a `Sequence[Tensor | None]`. Before
	calling `fn`, `safe` compacts [1] the sequence to remove all `None` values. When the compacted
	sequence is empty, `safe` returns `None` without calling `fn`. When at least one non-`None`
	`Tensor` remains, `safe` passes the compacted sequence to `fn`.

	`safe` is applied as a decorator to `safe_stack` [2] and `safe_cat` [3] to produce null-safe
	stacking and concatenation. `safe` is also applied to `reduce_masks` [4] to produce null-safe
	mask reduction.

	Parameters
	----------
	fn : Callable[Concatenate[Sequence[Tensor], PSpec], Tensor | None]
		A callable whose first argument is a `Sequence[Tensor]` with at least one element. `fn` must
		handle a compacted sequence of any length ≥ 1.

	Returns
	-------
	wrapped : Callable[Concatenate[Sequence[Tensor | None], PSpec], Tensor | None]
		A wrapped version of `fn` that accepts `None` values in the first argument and returns `None`
		when no non-`None` `Tensor` values are present.

	See Also
	--------
	compact : Filter `None` values from an iterable.
	safe_cat : Concatenate tensors along an existing dimension, skipping `None` values.
	safe_stack : Stack tensors along a new dimension, skipping `None` values.

	References
	----------
	[1] py_pwy_phy_hy.compact

	[2] py_pwy_phy_hy.safe_stack

	[3] py_pwy_phy_hy.safe_cat

	[4] py_pwy_phy_hy.reduce_masks
	"""

	@wraps(fn)
	def inner(tensors: Sequence[Tensor | None], *args: PSpec.args, **kwargs: PSpec.kwargs) -> Tensor | None:
		safe_tensors: list[Tensor] = compact(tensors)
		if len(safe_tensors) == 0:
			return None
		return fn(safe_tensors, *args, **kwargs)

	return inner

"""
Some or all of the logic in this module may be protected by the following.

MIT License

Copyright (c) 2026 Phil Wang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
