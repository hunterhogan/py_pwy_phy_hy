from __future__ import annotations

from collections.abc import Callable, Sequence
from einops import rearrange
from py_pwy_phy_hy import exists, safe
from torch import arange, Tensor
from torch.types import Number
import torch

def lens_to_mask(lens: Tensor, max_len: Number | None = None) -> Tensor:
	"""Convert a sequence of length values into a boolean mask `Tensor`.

	You can use `lens_to_mask` to create sequence masks from integer length values. For each scalar
	in `lens`, `lens_to_mask` produces a row of `True` values for positions less than that length
	value and `False` values for all positions equal to or greater than it. The output `Tensor` has
	one more dimension than `lens`, appended at the last axis, with length `max_len`.

	Parameters
	----------
	lens : Tensor
		A `Tensor` of non-negative integers representing sequence lengths. `lens` may have any shape;
		the output shape is `(*lens.shape, max_len)`.
	max_len : int | None = None
		The size of the last dimension of the output `Tensor`. If `None`, `max_len` is set to
		`int(lens.amax().item())`.

	Returns
	-------
	mask : Tensor
		A boolean `Tensor` of shape `(*lens.shape, max_len)`. Position `i` along the last axis is
		`True` if `i < lens[...]` for the corresponding element of `lens`.

	Examples
	--------
	From the test suite [1], verifying that `lens_to_mask` sets exactly `length` leading `True`
	values per row:

		>>> import torch
		>>> from py_pwy_phy_hy import lens_to_mask
		>>> lens = torch.tensor([4, 3, 1])
		>>> mask = lens_to_mask(lens)
		>>> mask.shape
		torch.Size([3, 4])
		>>> (mask.sum(dim=-1) == lens).all()
		tensor(True)

	Passing an explicit `max_len` produces a wider mask than the maximum length in `lens`:

		>>> lens_to_mask(lens, max_len=6).shape
		torch.Size([3, 6])

	From dreamer4 [2], masking padded time steps in variable-length rollouts:

		```python
		mask_for_gae = lens_to_mask(experience.lens, time)
		```

	References
	----------
	[1] tests/test_masking.py

	[2] lucidrains/dreamer4
		https://github.com/lucidrains/dreamer4
	"""
	device: torch.device = lens.device

	if not exists(max_len):
		max_len = lens.amax().item()

	seq: Tensor = arange(max_len, device = device)
	lens = rearrange(lens, '... -> ... 1')
	return seq < lens

@safe
def reduce_masks(masks: Sequence[Tensor], op: Callable[[Tensor, Tensor], Tensor]) -> Tensor | None:
	"""Reduce a sequence of boolean mask `Tensor` values to a single mask using a binary operator.

	You can use `reduce_masks` to apply any binary element-wise callable reduction over a sequence of
	boolean masks. The `safe` [1] decorator filters out `None` values from `masks` before `op` is
	applied. If `masks` contains no non-`None` values, `reduce_masks` returns `None`. Reduction
	proceeds left-to-right over the non-`None` elements of `masks`.

	Parameters
	----------
	masks : Sequence[Tensor | None]
		A `Sequence` of boolean `Tensor` or `None` values. `None` values are filtered out before `op`
		is applied. All non-`None` `Tensor` values must have the same shape.
	op : Callable[[Tensor, Tensor], Tensor]
		A binary callable that accepts two `Tensor` arguments and returns a `Tensor`. Common choices
		are `torch.logical_and` [2] and `torch.logical_or` [3].

	Returns
	-------
	mask : Tensor | None
		The result of applying `op` cumulatively, left-to-right, over the non-`None` members of
		`masks`. Returns `None` if no non-`None` values remain after filtering.

	See Also
	--------
	and_masks : Reduce masks using element-wise logical AND.
	or_masks : Reduce masks using element-wise logical OR.

	Examples
	--------
	From the test suite [4]:

	>>> import torch
	>>> from py_pwy_phy_hy import reduce_masks
	>>> mask1 = torch.tensor([True, True])
	>>> mask2 = torch.tensor([True, False])
	>>> reduce_masks([None, None], torch.logical_and) is None
	True
	>>> reduce_masks([mask1, None, mask2], torch.logical_and)
	tensor([ True, False])
	>>> reduce_masks([mask1, None, mask2], torch.logical_or)
	tensor([ True,  True])

	References
	----------
	[1] py_pwy_phy_hy.safe

	[2] torch.logical_and - PyTorch documentation
		https://pytorch.org/docs/stable/generated/torch.logical_and.html
	[3] torch.logical_or - PyTorch documentation
		https://pytorch.org/docs/stable/generated/torch.logical_or.html
	[4] tests/test_masking.py

	"""
	mask, *rest_masks = masks

	for rest_mask in rest_masks:
		mask: Tensor = op(mask, rest_mask)

	return mask

def and_masks(masks: Sequence[Tensor | None]) -> Tensor | None:
	"""Reduce a `Sequence` of boolean mask `Tensor` values to a single mask using element-wise logical AND.

	You can use `and_masks` to combine multiple boolean masks so that the result is `True` only where
	all non-`None` input masks are `True`. `and_masks` calls `reduce_masks` [1] with
	`torch.logical_and` [2]. `None` values in `masks` are filtered out before reduction. If all
	values in `masks` are `None`, `and_masks` returns `None`.

	Parameters
	----------
	masks : Sequence[Tensor | None]
		A `Sequence` of boolean `Tensor` or `None` values. `None` values are treated as absent and
		filtered out before reduction. All non-`None` `Tensor` values must have the same shape.

	Returns
	-------
	mask : Tensor | None
		A boolean `Tensor` that is `True` only at positions where every non-`None` input mask
		is `True`. Returns `None` if `masks` contains no non-`None` values.

	See Also
	--------
	or_masks : Reduce masks using element-wise logical OR.
	reduce_masks : Reduce masks using a caller-supplied binary operator.

	Examples
	--------
	From the test suite [3]:

		>>> from torch import tensor
		>>> from py_pwy_phy_hy import and_masks
		>>> and_masks([None]) is None
		True
		>>> mask1 = tensor([True, True])
		>>> mask2 = tensor([True, False])
		>>> and_masks([mask1, None, mask2])
		tensor([ True, False])

	From sdft_pytorch [4], intersecting an end-of-sequence mask with an initial-token mask to exclude
	padding and masked prefix positions from the loss calculation:

		```python
		mask = and_masks([eos_mask, init_tokens_mask])
		```

	References
	----------
	[1] py_pwy_phy_hy.reduce_masks

	[2] torch.logical_and - PyTorch documentation
		https://pytorch.org/docs/stable/generated/torch.logical_and.html
	[3] tests/test_utils.py

	[4] lucidrains/sdft-pytorch
		https://github.com/lucidrains/sdft-pytorch
	"""
	return reduce_masks(masks, torch.logical_and)

def or_masks(masks: Sequence[Tensor | None]) -> Tensor | None:
	"""Reduce a sequence of boolean mask `Tensor` values to a single mask using element-wise logical OR.

	You can use `or_masks` to combine multiple boolean masks so that the result is `True` wherever at
	least one non-`None` input mask is `True`. `or_masks` calls `reduce_masks` [1] with
	`torch.logical_or` [2]. `None` values in `masks` are filtered out before reduction. If all values
	in `masks` are `None`, `or_masks` returns `None`.

	Parameters
	----------
	masks : Sequence[Tensor | None]
		A sequence of boolean `Tensor` or `None` values. `None` values are treated as absent and
		filtered out before reduction. All non-`None` `Tensor` values must have the same shape.

	Returns
	-------
	mask : Tensor | None
		A boolean `Tensor` that is `True` at any position where at least one non-`None` input mask is
		`True`. Returns `None` if `masks` contains no non-`None` values.

	See Also
	--------
	and_masks : Reduce masks using element-wise logical AND.
	reduce_masks : Reduce masks using a caller-supplied binary operator.

	References
	----------
	[1] py_pwy_phy_hy.reduce_masks

	[2] torch.logical_or - PyTorch documentation
		https://pytorch.org/docs/stable/generated/torch.logical_or.html
	"""
	return reduce_masks(masks, torch.logical_or)

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
