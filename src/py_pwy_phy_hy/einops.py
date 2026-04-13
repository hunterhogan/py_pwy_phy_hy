"""Provide `einops` pack, unpack, and paired inverse utilities.

You can use this module to pack one or more `torch.Tensor` objects using an `einops` pattern [1]
and to recover the original shapes through paired inverse functions. `pack_one` and `unpack_one`
handle the single-tensor case. `pack_with_inverse` handles both the single-tensor and
list-of-tensors case and is re-exported from the package root [2].

Contents
--------
Functions
	pack_one
		Pack one `Tensor` and return shape metadata for paired reconstruction.
	pack_with_inverse
		Pack `t` with an einops pattern and return a paired inverse unpacking function.
	unpack_one
		Unpack one `Tensor` using packed-shape metadata produced by `pack_one`.

References
----------
[1] einops pack/unpack API
	https://einops.rocks/api/pack_unpack/
[2] py_pwy_phy_hy
"""
from collections.abc import Callable, Sequence
from einops import pack, unpack
from py_pwy_phy_hy import default, first
from torch import is_tensor, Tensor
from typing import overload

def pack_one(t: Tensor, pattern: str) -> tuple[Tensor, Sequence[tuple[int, ...] | list[int]]]:
	"""Pack one `Tensor` and return shape metadata for paired reconstruction.

	You can use `pack_one` to wrap a single `Tensor` in the `einops.pack` [1] interface,
	capturing packed-shape metadata needed to restore the original shape with `unpack_one` [2].

	Parameters
	----------
	t : Tensor
		Input `Tensor` to pack.
	pattern : str
		Einops packing pattern string passed to `einops.pack` [1].

	Returns
	-------
	packedTensorAndPackedShape : tuple[Tensor, Sequence[tuple[int, ...] | list[int]]]
		Packed `Tensor` output and packed-shape metadata for reconstruction.

	See Also
	--------
	unpack_one : Reconstruct one `Tensor` using packed-shape metadata from `pack_one`.

	References
	----------
	[1] einops pack/unpack API
		https://einops.rocks/api/pack_unpack/
	[2] py_pwy_phy_hy.einops.unpack_one
	"""
	return pack([t], pattern)

@overload
def pack_with_inverse(t: Tensor, pattern: str) -> tuple[Tensor, Callable[[Tensor, str | None], Tensor]]: ...
@overload
def pack_with_inverse(t: list[Tensor], pattern: str) -> tuple[Tensor, Callable[[Tensor, str | None], list[Tensor]]]: ...
def pack_with_inverse(t: Tensor | list[Tensor], pattern: str) -> tuple[Tensor, Callable[[Tensor, str | None], Tensor | list[Tensor]]]:
	"""Pack `t` with `pattern` using einops and return a paired inverse unpacking function.

	You can use this function to merge one or more tensors into a single packed tensor using an
	einops `pack` pattern [1] and to later restore the original shapes. When `t` is a single
	`torch.Tensor`, the function wraps `t` in a list before packing and unwraps the result inside the
	inverse function. When `t` is a list of tensors, the inverse function returns a list of tensors.
	The inverse function accepts an optional `inv_pattern` argument to override the pattern used for
	unpacking; when `inv_pattern` is `None`, the original `pattern` is reused.

	Parameters
	----------
	t : Tensor | list[Tensor]
		A single tensor or a list of tensors to pack.
	pattern : str
		An einops pack pattern string such as `'b * d'`, where `*` collects the packed dimensions.

	Returns
	-------
	packed : Tensor
		The packed tensor produced by `einops.pack` [1].
	inverse : Callable[[Tensor, str | None], Tensor | list[Tensor]]
		A function that accepts the packed (or transformed) tensor and an optional override pattern
		and returns the unpacked tensor or list of tensors.

	See Also
	--------
	tree_flatten_with_inverse : Flatten a PyTree and return an inverse reconstruction function.

	Examples
	--------
	Pack a single tensor and recover the original shape [2]:

		```python
		import torch
		from py_pwy_phy_hy import pack_with_inverse

		t = torch.randn(3, 12, 2, 2)
		packed, inverse = pack_with_inverse(t, "b * d")
		# packed.shape == (3, 24, 2)
		recovered = inverse(packed)
		# recovered.shape == (3, 12, 2, 2)
		```

	Pack a list of tensors and unpack with an overriding pattern [2]:

		```python
		t = torch.randn(3, 12, 2)
		u = torch.randn(3, 4, 2)
		packed, inverse = pack_with_inverse([t, u], "b * d")
		# packed.shape == (3, 28, 2)

		reduced = packed.sum(dim=-1)
		t_out, u_out = inverse(reduced, "b *")
		# t_out.shape == (3, 12)
		# u_out.shape == (3, 4)
		```

	References
	----------
	[1] einops.pack - einops documentation
		https://einops.rocks/api/pack/
	[2] tests.test_utils.test_pack_with_inverse

	"""
	is_one: bool = is_tensor(t)

	if is_one:
		t = [t]

	packed, packed_shape = pack(t, pattern)

	def inverse(out: Tensor, inv_pattern: str | None = None) -> Tensor | list[Tensor]:
		inv_pattern = default(inv_pattern, pattern)
		unpacked: list[Tensor] = unpack(out, packed_shape, inv_pattern)

		if is_one:
			return first(unpacked)

		return unpacked

	return packed, inverse

def unpack_one(t: Tensor, ps: Sequence[tuple[int, ...] | list[int]], pattern: str) -> Tensor:
	"""Unpack one `Tensor` using packed-shape metadata produced by `pack_one`.

	You can use `unpack_one` with metadata from a paired `pack_one` [1] call to reconstruct the
	original `Tensor` shape. The function delegates to `einops.unpack` [2] and extracts the single
	unpacked output.

	Parameters
	----------
	t : Tensor
		Packed `Tensor` produced by `pack_one` [1].
	ps : Sequence[tuple[int, ...] | list[int]]
		Packed-shape metadata returned by `pack_one` [1].
	pattern : str
		Einops unpacking pattern string passed to `einops.unpack` [2].

	Returns
	-------
	unpackedTensor : Tensor
		Reconstructed `Tensor` matching the original shape before packing.

	See Also
	--------
	pack_one : Pack one `Tensor` and return shape metadata for paired reconstruction.

	References
	----------
	[1] py_pwy_phy_hy.einops.pack_one

	[2] einops pack/unpack API
		https://einops.rocks/api/pack_unpack/
	"""
	return unpack(t, list(ps), pattern)[0]

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
