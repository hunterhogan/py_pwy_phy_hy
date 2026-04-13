"""Provide PyTorch tensor normalization, PyTree manipulation, and masked mean utilities.

You can use this module to normalize tensor vectors to unit length, compute masked means, apply
functions to tensor leaves in PyTree structures, and flatten and reconstruct nested data.

Contents
--------
Functions
	l2norm
		Normalize `Tensor` vectors to unit length along the last dimension.
	masked_mean
		Compute the mean of a tensor over positions selected by a boolean mask.
	tree_flatten_with_inverse
		Flatten a PyTree into a list of leaves and return a paired inverse function.
	tree_map_tensor
		Apply a function to every tensor leaf in a PyTree, leaving non-tensor leaves unchanged.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from py_pwy_phy_hy import exists, pad_right_ndim
from torch import is_tensor, Tensor
from torch.utils._pytree import PyTree, tree_flatten, tree_map, tree_unflatten
from typing import Any
import torch
import torch.nn.functional as F

def l2norm(t: Tensor) -> Tensor:
	"""Normalize `Tensor` vectors to unit length.

	You can use `l2norm` to normalize attention query and key vectors before computing similarity
	scores. Normalizing query and key vectors prevents those with large magnitudes from dominating
	similarity scores.

	Parameters
	----------
	t : Tensor
		Input `Tensor` to normalize.

	Returns
	-------
	normalizedTensor : Tensor
		`Tensor` with each vector scaled to unit length.

	torch
	-----
	`l2norm` calls `torch.nn.functional.normalize` [1] with `p=2` and `dim=-1`, which divides each
	vector by its Euclidean length (L2 norm) along the last dimension.

	Examples
	--------
	Normalize `Tensor` attention query, `q`, and `Tensor` attention key, `k`, before computing
	similarity scores: [2]

		```python
		q, k = map(l2norm, (q, k))
		```

	References
	----------
	[1] torch.nn.functional.normalize
		https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html
	[2] BS-RoFormer.mel_band_roformer.LinearAttention
		https://github.com/lucidrains/BS-RoFormer
	"""
	return F.normalize(t, dim = -1, p = 2)

def masked_mean(
	t: Tensor,
	mask: Tensor | None = None,
	dim: torch.Size | list[int] | tuple[int, ...] | int | None = None,
	eps: float = 1e-5,
) -> Tensor:
	"""Compute the mean of `t` over positions selected by `mask`.

	You can use this function to average only the elements of `t` where `mask` is `True`, ignoring
	masked-out positions. When `mask` is `None`, the function falls back to the standard
	`torch.Tensor.mean` [1]. When `mask` has fewer dimensions than `t`, the function right-pads
	`mask` with singleton dimensions using `pad_right_ndim` [2] before broadcasting. When all
	positions in `mask` are `False` and `dim` is `None`, the function returns zero by summing over
	the empty selection.

	Parameters
	----------
	t : Tensor
		The input tensor to be averaged.
	mask : Tensor | None = None
		A boolean tensor selecting which positions contribute to the mean. When `mask` has fewer
		dimensions than `t`, singleton dimensions are appended on the right before broadcasting. Pass
		`None` to compute an unmasked mean.
	dim : torch.Size | list[int] | tuple[int, ...] | int | None = None
		The dimension or dimensions along which to compute the mean. Pass `None` to reduce over all
		dimensions.
	eps : float = 1e-5
		A small value added to the denominator to prevent division by zero when computing the masked
		mean along a dimension.

	Returns
	-------
	result : Tensor
		The masked mean of `t`. The shape matches `t` with the reduced dimension removed when `dim`
		is specified, or a scalar tensor when `dim` is `None`.

	See Also
	--------
	pad_right_ndim : Pad singleton dimensions on the right of a tensor to reach a target number of dimensions.

	Examples
	--------
	Compute the mean of all elements with no mask [3]:

		```python
		from torch import tensor
		from py_pwy_phy_hy import masked_mean

		t = tensor([1.0, 2.0, 3.0, 4.0])
		result = masked_mean(t)
		# result == tensor(2.5)
		```

	Select only the `True` positions using a boolean mask [3]:

		```python
		mask = tensor([True, False, True, False])
		result = masked_mean(t, mask=mask)
		# result == tensor(2.0)
		```

	Average along a specific dimension [3]:

		```python
		t = tensor([[1.0, 2.0], [3.0, 4.0]])
		mask = tensor([[True, False], [True, True]])
		result = masked_mean(t, mask=mask, dim=1)
		# result == tensor([1.0, 3.5])
		```

	References
	----------
	[1] torch.Tensor.mean - PyTorch documentation
		https://pytorch.org/docs/stable/generated/torch.Tensor.mean.html
	[2] py_pwy_phy_hy.pad_right_ndim

	[3] tests.test_utils.test_masked_mean

	"""
	if not exists(mask):
		return t.mean(dim = dim) if exists(dim) else t.mean()

	if mask.ndim < t.ndim:
		mask = pad_right_ndim(mask, t.ndim - mask.ndim)

	mask = mask.expand_as(t)

	if not exists(dim):
		return t[mask].mean() if mask.any() else t[mask].sum()

	num: Tensor = (t * mask).sum(dim = dim)
	den: Tensor = mask.sum(dim = dim)

	return num / den.clamp(min = eps)

def tree_map_tensor(fn: Callable[[Tensor], Tensor], tree: PyTree) -> PyTree:
	"""Apply `fn` to every `torch.Tensor` leaf in `tree`, leaving non-tensor leaves unchanged.

	You can use this function to transform only the tensor leaves of a PyTree [1] structure without
	disturbing the non-tensor leaves. The function wraps `fn` with an identity pass-through for
	non-tensor values and delegates structural traversal to `torch.utils._pytree.tree_map` [2].

	Parameters
	----------
	fn : Callable[[Tensor], Tensor]
		A function to apply to each `torch.Tensor` leaf in `tree`.
	tree : PyTree
		A nested Python structure such as a tuple, list, or dictionary containing a mix of
		`torch.Tensor` values and other Python objects.

	Returns
	-------
	mappedTree : PyTree
		A PyTree with the same structure as `tree`, where each `torch.Tensor` leaf has been replaced
		by the result of `fn(leaf)` and all non-tensor leaves are unchanged.

	See Also
	--------
	tree_flatten_with_inverse : Flatten a PyTree into a list and return an inverse function.

	Examples
	--------
	Increment only the tensor leaf while preserving non-tensor leaves [3]:

		```python
		from torch import tensor
		from py_pwy_phy_hy import tree_map_tensor

		tree = (1, tensor(2), 3)
		result = tree_map_tensor(lambda t: t + 1, tree)
		# result[0] == 1
		# result[1] == tensor(3)
		# result[2] == 3
		```

	Detach all tensors nested inside a state container [4]:

		```python
		from py_pwy_phy_hy import tree_map_tensor

		nextMemory = tree_map_tensor(lambda t: t.detach(), nextMemory)
		```

	References
	----------
	[1] PyTree - PyTorch documentation
		https://pytorch.org/docs/stable/pytree.html
	[2] torch.utils._pytree.tree_map - PyTorch documentation
		https://pytorch.org/docs/stable/pytree.html#torch.utils._pytree.tree_map
	[3] tests.test_utils.test_tree_map_tensor

	[4] fast_weight_attention.chunk_manager.ChunkManager.forward
		https://context7.com/lucidrains/fast-weight-attention
	"""
	def func(t: object) -> object:
		if is_tensor(t):
			return fn(t)
		return t

	return tree_map(func, tree)

def tree_flatten_with_inverse(tree: PyTree) -> tuple[list[Any], Callable[[Iterable[Any]], PyTree]]:
	"""Flatten `tree` into a list of leaves and return a paired inverse function.

	You can use this function to decompose a nested PyTree [1] structure into a flat list of leaves
	and to recover the original nested structure from a modified list. The paired inverse function
	calls `torch.utils._pytree.tree_unflatten` [2] with the `TreeSpec` captured at flatten time, so
	the structure can be reconstructed even after the leaves have been modified.

	Parameters
	----------
	tree : PyTree
		A nested Python structure such as a tuple, list, or dictionary to flatten.

	Returns
	-------
	flattened : list[Any]
		A flat list of all leaves in `tree` in left-to-right traversal order.
	inverse : Callable[[Iterable[Any]], PyTree]
		A function that accepts an iterable of leaves and reconstructs a PyTree with the same
		structure as the original `tree`.

	See Also
	--------
	tree_map_tensor : Apply a function to every tensor leaf in a PyTree.

	Examples
	--------
	Modify a single leaf and reconstruct the original nested structure [3]:

		```python
		from py_pwy_phy_hy import tree_flatten_with_inverse

		tree = (1, (2, 3), 4)
		(first, *rest), inverse = tree_flatten_with_inverse(tree)
		result = inverse((first + 1, *rest))
		# result == (2, (2, 3), 4)
		```

	References
	----------
	[1] PyTree - PyTorch documentation
		https://pytorch.org/docs/stable/pytree.html
	[2] torch.utils._pytree.tree_unflatten - PyTorch documentation
		https://pytorch.org/docs/stable/pytree.html#torch.utils._pytree.tree_unflatten
	[3] tests.test_utils.test_tree_flatten_with_inverse

	"""
	flattened, spec = tree_flatten(tree)

	def inverse(out: Iterable[Any]) -> PyTree:
		return tree_unflatten(out, spec)

	return flattened, inverse


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
