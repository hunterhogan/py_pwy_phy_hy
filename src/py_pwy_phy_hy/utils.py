"""Provide PyTree manipulation utilities for PyTorch tensors.

You can use this module to apply functions to tensor leaves in PyTree structures and to flatten
and reconstruct nested data.

Contents
--------
Functions
	tree_flatten_with_inverse
		Flatten a PyTree into a list of leaves and return a paired inverse function.
	tree_map_tensor
		Apply a function to every tensor leaf in a PyTree, leaving non-tensor leaves unchanged.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from torch import is_tensor, Tensor
from torch.utils._pytree import PyTree, tree_flatten, tree_map, tree_unflatten
from typing import Any

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
