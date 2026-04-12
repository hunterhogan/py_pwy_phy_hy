from __future__ import annotations

from collections.abc import Sequence
from py_pwy_phy_hy import safe
from torch import cat, stack, Tensor

@safe
def safe_stack(tensors: Sequence[Tensor], dim: int = 0) -> Tensor | None:
	"""Stack tensors from `tensors` along a new dimension, skipping `None` values.

	You can use `safe_stack` to stack a mixed sequence of `Tensor` and `None` values. The `safe` [1]
	decorator filters out `None` values from `tensors` before passing the remaining `Tensor` values
	to `torch.stack` [2]. If `tensors` contains no non-`None` values, `safe_stack` returns `None`.

	Parameters
	----------
	tensors : Sequence[Tensor | None]
		A `Sequence` of `Tensor` or `None` values. `None` values are filtered out before stacking.
		All non-`None` `Tensor` values must have the same shape.
	dim : int = 0
		The dimension along which to stack. The result has one more dimension than each input
		`Tensor`.

	Returns
	-------
	stacked : Tensor | None
		The stacked `Tensor`, or `None` if `tensors` contains no non-`None` values.

	See Also
	--------
	safe_cat : Concatenate tensors along an existing dimension, skipping `None` values.

	Examples
	--------
	From the test suite [3]:

		>>> import torch
		>>> from py_pwy_phy_hy import safe_stack
		>>> t1, t2 = torch.randn(2, 3), torch.randn(2, 3)
		>>> safe_stack([]) is None
		True
		>>> safe_stack([None]) is None
		True
		>>> safe_stack([t1]).shape
		torch.Size([1, 2, 3])
		>>> safe_stack([t1, None]).shape  # None is skipped; only t1 is stacked
		torch.Size([1, 2, 3])
		>>> safe_stack([t1, t2]).shape
		torch.Size([2, 2, 3])

	From dreamer4 [4], collecting optional per-layer intermediates where some layers
	may not produce output (returning `None`):

		```python
			intermediates = TransformerIntermediates(
				stack(time_attn_kv_caches),
				safe_stack(normed_time_attn_inputs),
				safe_stack(normed_space_attn_inputs),
				safe_stack(rnn_hiddens),
				hiddens,
		)

	References
	----------
	[1] py_pwy_phy_hy.safe

	[2] torch.stack - PyTorch documentation
		https://pytorch.org/docs/stable/generated/torch.stack.html
	[3] tests/test_utils.py

	[4] lucidrains/dreamer4
		https://github.com/lucidrains/dreamer4
	"""
	return stack(tensors, dim = dim)  # pyright: ignore[reportArgumentType] https://github.com/pytorch/pytorch/issues/179391

@safe
def safe_cat(tensors: Sequence[Tensor], dim: int = 0) -> Tensor | None:
	"""Concatenate tensors from `tensors` along an existing dimension, skipping `None` values.

	You can use `safe_cat` to concatenate a mixed sequence of `Tensor` and `None` values. The `safe`
	[1] decorator filters out `None` values from `tensors` before passing the remaining `Tensor`
	values to `torch.cat` [2]. If `tensors` contains no non-`None` values, `safe_cat` returns `None`.

	A common pattern is iterative accumulation where the accumulator starts as `None`. On the first
	iteration, `safe_cat` receives one non-`None` `Tensor` and returns it unchanged. On subsequent
	iterations, `safe_cat` concatenates the accumulator with the new `Tensor`.

	Parameters
	----------
	tensors : Sequence[Tensor | None]
		A sequence of `Tensor` or `None` values. `None` values are filtered out before concatenation.
		All non-`None` `Tensor` values must have the same shape in every dimension except `dim`.
	dim : int = 0
		The dimension along which to concatenate.

	Returns
	-------
	concatenated : Tensor | None
		The concatenated `Tensor`, or `None` if `tensors` contains no non-`None` values.

	See Also
	--------
	safe_stack : Stack tensors along a new dimension, skipping `None` values.

	Examples
	--------
	From the test suite [3]:

		>>> import torch
		>>> from py_pwy_phy_hy import safe_cat
		>>> t1, t2 = torch.randn(2, 3), torch.randn(2, 3)
		>>> safe_cat([]) is None
		True
		>>> safe_cat([None]) is None
		True
		>>> safe_cat([t1, None]).shape  # None is skipped; only t1 is returned
		torch.Size([2, 3])
		>>> safe_cat([t1, t2]).shape
		torch.Size([4, 3])

	From sdft_pytorch [4], accumulating per-step token losses across a generation loop where
	`token_kl_div_losses` is initialized to `None` before the loop:

		```python
		token_kl_div_losses = safe_cat((token_kl_div_losses, token_kl_div), dim=1)
		```

	References
	----------
	[1] py_pwy_phy_hy.safe

	[2] torch.cat - PyTorch documentation
		https://pytorch.org/docs/stable/generated/torch.cat.html
	[3] tests/test_utils.py

	[4] lucidrains/sdft-pytorch
		https://github.com/lucidrains/sdft-pytorch
	"""
	return cat(tensors, dim = dim)  # pyright: ignore[reportUnknownVariableType, reportCallIssue, reportArgumentType] https://github.com/pytorch/pytorch/issues/179391

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
