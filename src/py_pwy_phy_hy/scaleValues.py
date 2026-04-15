"""Provide vector normalization, masked mean computation, and learned RMS normalization.

You can use this module to normalize feature vectors to unit length, compute masked mean
reductions over selected tensor positions, and apply learned root-mean-square normalization
to transformer and neural network feature channels.

Contents
--------
Functions
	l2norm
		Normalize `Tensor` vectors to unit length along the last dimension.
	masked_mean
		Compute the mean of a tensor over positions selected by a boolean mask.

Classes
	RMSNorm
		Normalize feature vectors with root-mean-square scaling and a learned `gamma` parameter.
"""
from __future__ import annotations

from py_pwy_phy_hy import exists, pad_right_ndim
from torch import nn, SymInt, Tensor
from torch.nn import Module
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

def masked_mean(t: Tensor, mask: Tensor | None = None, dim: torch.Size | list[int] | tuple[int, ...] | int | None = None, eps: float = 1e-5) -> Tensor:
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

class RMSNorm(Module):
	"""Normalize feature vectors with root-mean-square scaling and a learned rescaling parameter.

	You can use `RMSNorm` as a pre-normalization layer before attention, feedforward, or linear
	projection sublayers in transformer-style modules. `RMSNorm` normalizes each feature vector along
	the last axis to unit length, scales the result by `√dim` to restore the original magnitude
	range, and then applies the learned per-feature `gamma` parameter.

	The normalization is mathematically equivalent to dividing each vector by its root mean square
	[1]. This formulation omits the mean-centering step of `torch.nn.LayerNorm` [2], which the
	original paper [1] found to be unnecessary for stable training. The implementation delegates the
	per-vector division to `torch.nn.functional.normalize` [3] and carries the `√dim` factor in a
	stored `scale` attribute.

	Parameters
	----------
	dim : int | SymInt
		Feature dimension of the last axis normalized by `RMSNorm`.

	Attributes
	----------
	scale : float
		Fixed multiplier equal to `√dim`. Stored so the value is computed once at construction time
		rather than recomputed on every forward pass.
	gamma : nn.Parameter
		Learned per-feature rescaling weights of shape `(dim,)`. Initialized to ones so `RMSNorm`
		starts as an identity-scale transform.

	See Also
	--------
	l2norm : Normalize `Tensor` vectors to unit length along the last dimension.

	Examples
	--------
	Apply `RMSNorm` as a pre-norm before a linear projection, following the lucidrains BS-RoFormer
	pattern [4] [5]:

		```python from torch import nn from py_pwy_phy_hy.scaleValues import RMSNorm

		dim = 128 pre_norm_proj = nn.Sequential(
			RMSNorm(dim), nn.Linear(dim, dim * 4),
		)
		```

	Store a `RMSNorm` instance on an `Attention` module and call it in `forward`, following the
	lucidrains BS-RoFormer pattern [4] [5]:

		```python class Attention(nn.Module):
			def __init__(self, dim: int) -> None:
				super().__init__() self.norm = RMSNorm(dim) self.to_qkv = nn.Linear(dim, dim * 3,
				bias=False)

			def forward(self, x):
				x = self.norm(x) # ... query, key, value projection follows
		```

	References
	----------
	[1] Zhang, B., and Sennrich, R. (2019).
		Root Mean Square Layer Normalization. NeurIPS 32.
		https://papers.nips.cc/paper_files/paper/2019/hash/1e8a19426224ca89e83cef47f1e7f53b-Abstract.html
	[2] torch.nn.LayerNorm - PyTorch documentation
		https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
	[3] torch.nn.functional.normalize - PyTorch documentation
		https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html
	[4] bs_roformer.bs_roformer - BS-RoFormer source
		https://github.com/lucidrains/BS-RoFormer
	[5] bs_roformer.mel_band_roformer - Mel-Band RoFormer source
		https://github.com/lucidrains/BS-RoFormer
	"""
	def __init__(self, dim: int | SymInt) -> None:
		super().__init__()
		self.scale: float = float(dim) ** 0.5
		self.gamma: nn.Parameter = nn.Parameter(torch.ones((dim,)))

	def forward(self, x: Tensor) -> Tensor:
		"""Return feature activations after RMS normalization and learned rescaling.

		This method normalizes `x` along the last axis to unit length, multiplies the normalized
		result by `self.scale` (`√dim`), and applies the learned per-feature `self.gamma` parameter
		element-wise.

		Parameters
		----------
		x : Tensor
			Input `Tensor` whose last axis stores feature channels. `x` may have any number of
			leading batch or sequence dimensions.

		Returns
		-------
		normalizedX : Tensor
			`Tensor` with the same shape as `x` after RMS normalization and learned rescaling.

		References
		----------
		[1] torch.nn.functional.normalize - PyTorch documentation
			https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html
		[2] Zhang, B., and Sennrich, R. (2019).
			Root Mean Square Layer Normalization. NeurIPS 32.
			https://papers.nips.cc/paper_files/paper/2019/hash/1e8a19426224ca89e83cef47f1e7f53b-Abstract.html
		"""
		return F.normalize(x, dim=-1) * self.scale * self.gamma

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
